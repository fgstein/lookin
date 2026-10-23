"""Calibration: what a model's confidence is worth, and what can be done with it.

* **Can the confidence be believed?** The summary table (ECE / MCE / AURC),
  a reliability diagram and a right-vs-wrong confidence histogram per
  experiment, and the fix -- one fitted temperature, with the ECE before
  and after it.
* **What does it buy?** The risk-coverage curve, a threshold explorer that
  runs both directions ("at 0.9, what do I get" and "to reach 99%, what do I
  set"), and per-class thresholds, each with the crops behind an operating
  point one click away.

The expensive part is one pass per experiment to sort its confidences
(core.calibration.ThresholdIndex), done in the background when the tab is
first shown and again on "Update now"; after that the slider and the tables
are lookups. Per-class thresholds and the temperature fit read the merge
again, each on its own job, behind its own button.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from lookin.core import calibration as cal
from lookin.core import charts
from lookin.core import class_focus as cf
from lookin.core.chart_style import color_for_experiment
from lookin.core.metrics import compute_reliability, diagnose_scores
from lookin.web import html as h
from lookin.web.views.base import RebuildView
from lookin.web.views.carousel import CarouselDialog
from lookin.web.views.common import score_diagnostics_banner
from lookin.web.views.media import chart_html, fig_png, png_size
from lookin.web.views.widgets import ClassFocus, class_select

CHART_MAX_WIDTH = 1200
CHART_PX_PER_INCH = 90

DEFAULT_THRESHOLD = 0.9
SLIDER_DIVISIONS = 200
# Which classes the per-class thresholds table shows past FOCUS_MAX (see ClassFocus).
CALIBRATION_FOCUS_CRITERIA = (
    (cf.FOCUS_LOWEST_COVERAGE, "Lowest coverage"),
    (cf.FOCUS_MOST_PREDICTED, "Most predictions"),
    (cf.FOCUS_MANUAL, "Hand-picked"),
)
DEFAULT_TARGET_ACCURACY = 0.99
DEFAULT_TARGET_PRECISION = 0.95

# On-screen widths of the per-experiment charts.
RELIABILITY_WIDTH = 380
HISTOGRAM_WIDTH = 400

DASH = "—"


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value == value else DASH


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%" if value == value else DASH


def _chart_width(fig) -> int:
    fig_w, _ = fig.get_size_inches()
    return int(min(CHART_MAX_WIDTH, max(600.0, fig_w * CHART_PX_PER_INCH)))


def _parse_float(text, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


@dataclass
class ExperimentCalibration:
    """Everything this tab keeps for one experiment between renders (the
    threshold index is the only per-row structure kept)."""
    name: str
    index: cal.ThresholdIndex
    curve: cal.RiskCoverage
    histogram: cal.ConfidenceHistogram
    reliability: object                 # metrics.ReliabilityData
    diagnostics: object                 # metrics.ScoreDiagnostics
    n: int
    accuracy: float
    mean_confidence: float
    temperature: Optional[object] = None  # cal.TemperatureFit, once fitted


@dataclass
class _Chart:
    """A rendered chart: its PNG and the on-screen width it is drawn at."""
    png: bytes
    title: str
    width: int
    key: str = ""


class CalibrationView(RebuildView):
    name = "calibration"
    title = "Calibration"

    def __init__(self, session):
        super().__init__(session)
        self.checks: dict = {}          # experiment name -> ticked
        self._cache: dict = {}          # experiment name -> ExperimentCalibration
        self._class_rows: dict = {}     # (name, target) -> list[ClassThreshold]
        self._cache_stale = False
        self.built = False

        # ---- what the last rebuild drew ----
        self.placeholder = ""           # a message instead of the results
        self.placeholder_error = False
        self.shown: list = []           # experiment names the results are for
        self.reliability_charts: dict = {}   # name -> _Chart | None (disabled)
        self.histogram_charts: dict = {}     # name -> _Chart
        self.curve_chart: Optional[_Chart] = None

        # ---- threshold explorer ----
        self.threshold = DEFAULT_THRESHOLD
        self.target_text = str(DEFAULT_TARGET_ACCURACY)
        self.target_points: dict = {}   # name -> ThresholdPoint, for target_value
        self.target_value: Optional[float] = None

        # ---- per-class thresholds ----
        self.class_experiment: Optional[str] = None
        self.class_target_text = str(DEFAULT_TARGET_PRECISION)
        self.class_busy = False
        self.class_message = ""         # "Pick an experiment first." / an error
        self.class_message_error = False
        self._class_table_input = None  # (rows, name, target)
        self.class_table_rows: list = []
        self.class_focus = ClassFocus(self.name, "focus", CALIBRATION_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the per-class table",
                                      support_criteria=(cf.FOCUS_LOWEST_COVERAGE,),
                                      support_noun="predictions")

        # ---- temperature scaling ----
        self.temperature_busy = False
        self.temperature_error = ""

    # ---- experiment picker ---------------------------------------------

    def _sync_checks(self) -> None:
        """One tick per loaded experiment, in AppState's order (so the
        colours match everywhere); a new experiment comes in ticked. The
        per-class dropdown is derived from the ticked ones."""
        names = list(self.state.experiments)
        self.checks = {name: self.checks.get(name, True) for name in names}
        selected = self._selected_names()
        if self.class_experiment not in selected:
            self.class_experiment = selected[0] if selected else None

    def _selected_names(self) -> list:
        return [name for name, ticked in self.checks.items() if ticked]

    # ---- lifecycle -------------------------------------------------------

    def on_shown(self) -> None:
        if not self.built and self.state.dataset is not None and self.state.experiments:
            self.built = True
            self.rebuild(self._refresh_now)

    def on_data_change(self) -> None:
        """Flag stale, recompute when asked. The cached bundles were computed from the old data, so whatever
        recompute comes next starts from scratch."""
        self.dirty = True
        self._cache_stale = True

    def can_update(self) -> bool:
        return self.built and self.state.dataset is not None and bool(self.state.experiments)

    def on_update(self, **_):
        self._cache.clear()
        self._class_rows.clear()
        self._cache_stale = False
        self.built = True
        self.progress_text = "Computing calibration..."
        self.rebuild(self._refresh_now)

    def recompute_after_restore(self) -> None:
        self._sync_checks()
        if self.state.dataset is None or not self._selected_names():
            return
        self.on_update()

    def on_colors_changed(self) -> None:
        # Drawn again from the cached bundles: only the colors changed.
        if self.built and self._cache:
            self.rebuild(self._refresh_now, clear_stale=False)

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if old_name in self._cache:
            bundle = self._cache.pop(old_name)
            bundle.name = new_name
            self._cache[new_name] = bundle
        for key in [k for k in self._class_rows if k[0] == old_name]:
            self._class_rows[(new_name, key[1])] = self._class_rows.pop(key)
        if old_name in self.checks:
            self.checks = {(new_name if n == old_name else n): v for n, v in self.checks.items()}
        if self.class_experiment == old_name:
            self.class_experiment = new_name
        for store in (self.target_points,):
            if old_name in store:
                store[new_name] = store.pop(old_name)
        if self._class_table_input is not None and self._class_table_input[1] == old_name:
            rows, _, target = self._class_table_input
            self._class_table_input = (rows, new_name, target)
        self.shown = [new_name if n == old_name else n for n in self.shown]
        self._sync_checks()
        # The charts carry the name in their titles: redraw them from the
        # cached bundles (cheap), unless the view is waiting for an update.
        if self.built and not self.dirty:
            self.rebuild(self._refresh_now)
        self.refresh()

    # ---- computation ----------------------------------------------------

    def _bundle(self, name: str) -> ExperimentCalibration:
        if name in self._cache:
            return self._cache[name]

        state = self.state
        experiment = state.experiments[name]
        merged = state.merged(name)
        _log(f"computing calibration for '{name}' ({len(merged):,} merged rows)...")
        t0 = time.monotonic()

        confidence = merged["predicted_score"].to_numpy(dtype=float)
        correct = merged["correct"].to_numpy().astype(bool)
        labels_for_cols = experiment.class_labels[: len(experiment.score_cols)]
        # As stored (float32): a float64 copy of the whole score matrix is
        # 1.6 GB at a hundred thousand rows and two thousand classes.
        diagnostics = diagnose_scores(merged[experiment.score_cols].to_numpy(), labels_for_cols)

        index = cal.build_index(confidence, correct)
        bundle = ExperimentCalibration(
            name=name,
            index=index,
            curve=cal.risk_coverage(index),
            histogram=cal.confidence_histogram(confidence, correct),
            reliability=compute_reliability(merged, diagnostics=diagnostics),
            diagnostics=diagnostics,
            n=len(merged),
            accuracy=float(correct.mean()) if len(correct) else float("nan"),
            mean_confidence=float(np.nanmean(confidence)) if confidence.size else float("nan"),
        )
        _log(f"calibration for '{name}' done in {time.monotonic() - t0:.1f}s")
        self._cache[name] = bundle
        return bundle

    def _clear_results(self) -> None:
        self.shown = []
        self.reliability_charts = {}
        self.histogram_charts = {}
        self.curve_chart = None
        self.target_points = {}
        self._class_table_input = None
        self.class_table_rows = []
        self.class_message = ""
        self.class_focus.hide()

    def _show_placeholder(self, message: str, error: bool = False) -> None:
        self._clear_results()
        self.placeholder = message
        self.placeholder_error = error

    def _refresh_now(self) -> None:
        """The whole tab, in the background: bundles, then every chart, then
        the tables' lookups."""
        if self._cache_stale:
            self._cache.clear()
            self._class_rows.clear()
            self._cache_stale = False
        self._sync_checks()
        state = self.state
        self.placeholder = ""
        if state.dataset is None or not state.experiments:
            self._show_placeholder(
                "Load a dataset and at least one experiment (results CSV) in the 'Sources' tab first.")
            return
        selected = self._selected_names()
        if not selected:
            self._show_placeholder("Select at least one experiment above.")
            return
        try:
            bundles = {}
            for i, name in enumerate(selected):
                if name not in self._cache:
                    self.set_progress(f"Sorting confidences -- {name} ({i + 1}/{len(selected)})")
                bundles[name] = self._bundle(name)
            all_names = list(state.experiments)
            self.set_progress("Drawing reliability diagrams...")
            reliability = {name: self._reliability_chart(name, b.reliability, color_for_experiment(name, self.state.color_slots))
                           for name, b in bundles.items()}
            self.set_progress("Drawing confidence distributions...")
            histograms = {name: self._histogram_chart(name, b, color_for_experiment(name, self.state.color_slots))
                          for name, b in bundles.items()}
            self.set_progress("Drawing the risk-coverage curve...")
            curve = self._curve_chart(bundles, all_names)
            # The class table stays only if it is still about a shown experiment.
            keep_class = (self._class_table_input is not None and self._class_table_input[1] in bundles)
            self.reliability_charts = reliability
            self.histogram_charts = histograms
            self.curve_chart = curve
            self.shown = list(bundles)
            if not keep_class:
                self._class_table_input = None
                self.class_table_rows = []
                self.class_focus.hide()
            self._apply_target()
        except Exception as ex:  # noqa: BLE001
            _log(f"error computing calibration: {ex}")
            traceback.print_exc()
            self._show_placeholder(f"Error computing calibration: {ex}", error=True)

    def _bundles(self) -> dict:
        return {n: self._cache[n] for n in self.shown if n in self._cache}

    # ---- charts ------------------------------------------------------------

    def _reliability_chart(self, name: str, data, color) -> Optional[_Chart]:
        fig = charts.reliability_diagram(name, data, color)
        if fig is None:
            return None
        return _Chart(fig_png(fig), f"Calibration — {name}", RELIABILITY_WIDTH)

    def _histogram_chart(self, name: str, bundle: ExperimentCalibration, color) -> _Chart:
        fig = charts.confidence_split_histogram(name, bundle.histogram, color)
        return _Chart(fig_png(fig), f"Confidence distribution — {name}", HISTOGRAM_WIDTH)

    def _curve_chart(self, bundles: dict, all_names: list) -> _Chart:
        colors = {name: color_for_experiment(name, self.state.color_slots) for name in bundles}
        fig = charts.risk_coverage({name: bundle.curve for name, bundle in bundles.items()}, colors)
        return _Chart(fig_png(fig), "Risk-coverage", _chart_width(fig))

    def _chart_html(self, chart: _Chart) -> str:
        png_w, _ = png_size(chart.png)
        scale = chart.width / png_w if png_w else 0.5
        if not chart.key:
            chart.key = self.s.put_chart(chart.png)
        # Put again on every render: the session's chart store is bounded,
        # and this keeps the key (so the browser does not reload it).
        return chart_html(self.s, chart.png, chart.title, scale=scale, key=chart.key)

    # ---- threshold explorer --------------------------------------------

    def _apply_target(self) -> None:
        """Read the target field (clamped to [0, 1] and written back) and
        work back to each shown experiment's
        threshold -- one pass over the index, so not done in render()."""
        target = _parse_float(self.target_text, DEFAULT_TARGET_ACCURACY)
        self.target_text = f"{target:g}"
        self.target_value = target
        self.target_points = {name: cal.threshold_for_target(bundle.index, target)
                              for name, bundle in self._bundles().items()}

    def on_threshold(self, value="", **_):
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return
        self.threshold = max(0.0, min(1.0, threshold))

    def on_target_typed(self, value="", **_):
        self.target_text = str(value)

    def on_target_find(self, value=None, **_):
        if value is not None:
            self.target_text = str(value)
        if not self._cache:
            return
        self._apply_target()

    # ---- per-class thresholds ------------------------------------------

    def on_class_experiment(self, value="", **_):
        if value in self._selected_names():
            self.class_experiment = value

    def on_class_target_typed(self, value="", **_):
        self.class_target_text = str(value)

    def on_class_compute(self, target=None, **_):
        if target is not None:
            self.class_target_text = str(target)
        name = self.class_experiment
        if not name:
            self.class_message = "Pick an experiment first."
            self.class_message_error = False
            self._class_table_input = None
            self.class_table_rows = []
            self.class_focus.hide()
            return
        self.class_busy = True
        self.class_message = ""
        self.s.job("Per-class thresholds").run(self._compute_class_thresholds, on_error=self._on_class_error)

    def _on_class_error(self, ex: Exception) -> None:
        self.class_busy = False
        self.class_message = f"Error computing per-class thresholds: {ex}"
        self.class_message_error = True
        self._class_table_input = None
        self.class_table_rows = []
        self.refresh()

    def _compute_class_thresholds(self) -> None:
        name = self.class_experiment
        target = _parse_float(self.class_target_text, DEFAULT_TARGET_PRECISION)
        self.class_target_text = f"{target:g}"
        key = (name, target)
        if key not in self._class_rows:
            state = self.state
            experiment = state.experiments[name]
            merged = state.merged(name)
            labels = experiment.class_labels[: len(experiment.score_cols)]
            self._class_rows[key] = cal.per_class_thresholds(
                merged["predicted_label"].to_numpy(),
                merged["predicted_score"].to_numpy(dtype=float),
                merged["correct"].to_numpy().astype(bool),
                labels, target,
            )
        self._resolve_class_table(self._class_rows[key], name, target)
        self.class_busy = False
        self.refresh()

    def on_focus(self, op: str = "", **args):
        self.class_focus.handle(op, args)

    def _on_focus_change(self) -> None:
        if self._class_table_input is None:
            return
        self._resolve_class_table(*self._class_table_input)

    def _resolve_class_table(self, rows_data: list, name: str, target: float) -> None:
        """Which rows the per-class table shows (all of them, or the focus
        set past FOCUS_MAX classes) -- done on the action, not in render."""
        self._class_table_input = (rows_data, name, target)
        by_label = {row.label: row for row in rows_data}
        table = pd.DataFrame({
            "label": [row.label for row in rows_data],
            "support": [row.n_predicted for row in rows_data],
            # An unreachable target is the lowest coverage there is; a class
            # never predicted has nothing to threshold and goes last.
            "coverage": [np.nan if not row.n_predicted else (0.0 if row.unreachable else row.coverage)
                         for row in rows_data],
        })
        rankings = {
            cf.FOCUS_LOWEST_COVERAGE: ("coverage", True),
            cf.FOCUS_MOST_PREDICTED: ("support", False),
        }
        focus = self.class_focus.resolve(
            table["label"].tolist(),
            lambda criterion: cf.rank_by(table, *rankings[criterion],
                                         min_support=self.class_focus.min_support_for(criterion))
            if criterion in rankings else [])
        if self.class_focus.many:
            rows_data = [by_label[label] for label in focus if label in by_label]
        self.class_table_rows = rows_data

    # ---- temperature scaling -------------------------------------------

    def on_fit_temperature(self, **_):
        self.temperature_busy = True
        self.temperature_error = ""
        self.s.job("Temperature").run(self._fit_temperatures, on_error=self._on_temperature_error)

    def _on_temperature_error(self, ex: Exception) -> None:
        self.temperature_busy = False
        self.temperature_error = f"Error fitting the temperature: {ex}"
        self.refresh()

    def _fit_temperatures(self) -> None:
        state = self.state
        for name in self._selected_names():
            bundle = self._bundle(name)
            if bundle.temperature is not None:
                continue
            if not bundle.diagnostics.normalized:
                bundle.temperature = cal.TemperatureFit(
                    float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, bundle.n,
                    disabled_reason="the scores are not a probability distribution, so there is no "
                                    "likelihood to minimise")
                continue
            experiment = state.experiments[name]
            merged = state.merged(name)
            labels = experiment.class_labels[: len(experiment.score_cols)]
            _log(f"fitting temperature for '{name}'...")
            bundle.temperature = cal.fit_temperature(
                merged[experiment.score_cols].to_numpy(),
                merged[state.dataset.columns.label_col].to_numpy(),
                labels,
                y_pred=merged["predicted_label"].to_numpy(),
            )
            _log(f"temperature for '{name}': T={bundle.temperature.temperature:.3f}")
        self.temperature_busy = False
        self.refresh()

    # ---- drilling into an operating point -------------------------------

    def on_crops(self, experiment: str = "", threshold=None, which: str = "", **_):
        """The crops behind a number in the threshold table: the errors a
        deployment would let through, or the queue a human would be handed."""
        state = self.state
        if experiment not in state.experiments or state.dataset is None:
            return
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return
        merged = state.merged(experiment)
        cols = state.dataset.columns
        confidence = merged["predicted_score"].to_numpy(dtype=float)
        accepted = confidence >= np.float32(threshold)
        correct = merged["correct"].to_numpy().astype(bool)

        if which == "accepted_errors":
            subset = merged[accepted & ~correct]
            title = f"accepted and wrong at >= {threshold:.3f}"
        else:
            subset = merged[~accepted]
            title = f"abstained below {threshold:.3f}"

        uuid_col = cols.uuid_col or None
        suggestions = ({str(u): str(p) for u, p in zip(subset[uuid_col], subset["predicted_label"])}
                       if uuid_col and uuid_col in subset.columns else {})
        dialog = CarouselDialog(
            self.s, f"{experiment}  ·  {title}  ·  {len(subset):,} crop(s)",
            state.dataset, len(subset),
            lambda start, end: subset.iloc[start:end].to_dict("records"),
            cols.crop_col, uuid_col,
            review_source="calibration",
            suggestions=suggestions,
            label_col=cols.label_col,
        )
        self.s.open_dialog(dialog)

    # ---- the experiment ticks ------------------------------------------

    def on_tick(self, experiment: str = "", value=False, **_):
        self._sync_checks()
        if experiment in self.checks:
            self.checks[experiment] = bool(value)
        self._sync_checks()
        self.built = True
        self.progress_text = "Computing calibration..."
        self.rebuild(self._refresh_now)

    # ---- rendering ------------------------------------------------------

    def render(self) -> str:
        self._sync_checks()
        self.on_shown()
        state = self.state
        all_names = list(state.experiments)
        ticks = "".join(
            f'<span class="cal-exp">{h.dot(color_for_experiment(name, self.state.color_slots), size=10)}'
            f'{h.checkbox(name, ticked, change=self.c("tick", experiment=name), ellipsis=True)}</span>'
            for name, ticked in self.checks.items())
        if not ticks:
            ticks = '<span class="muted small">No experiment loaded.</span>'
        head = h.section(
            "Calibration & abstention",
            f'<div class="row-wrap"><b class="small">Experiments</b>{ticks}</div>',
            manual="calibration-error",
            sub="Computed from the same merged rows the Metrics tab reads, so the review "
                "overlay, the row filter and the split all apply here too.",
            right=h.button("Update now", act_attr=self.a("update"), icon_name="refresh", small=True, kind="ghost",
                           disabled=self.running, title="Already updating" if self.running else ""))
        parts = [self.stale_html(), head]

        # The two cheap placeholders are decided here; the rest by the last rebuild.
        if state.dataset is None or not state.experiments:
            parts.append(h.banner(
                "Load a dataset and at least one experiment (results CSV) in the 'Sources' tab first.", "info"))
            return f'<div class="stack cal">{"".join(parts)}</div>'
        if not self._selected_names() and not self.progress_text:
            parts.append(h.banner("Select at least one experiment above.", "info"))
            return f'<div class="stack cal">{"".join(parts)}</div>'
        if self.placeholder:
            parts.append(h.banner(self.placeholder, "error" if self.placeholder_error else "info"))
            parts.append(self.progress_html())
            return f'<div class="stack cal">{"".join(parts)}</div>'
        if self.error:
            parts.append(h.banner(self.error, "error"))
        bundles = self._bundles()
        for name, bundle in bundles.items():
            banner = score_diagnostics_banner(bundle.diagnostics, name)
            if banner:
                parts.append(banner)
        if self.progress_text:
            parts.append(self.progress_html())
        if not bundles:
            return f'<div class="stack cal">{"".join(parts)}</div>'

        parts.append(self._summary_html(bundles, all_names))
        parts.append(self._believable_html(bundles))
        if self.curve_chart is not None:
            parts.append(h.section("What does abstaining buy?", self._chart_html(self.curve_chart),
                                   manual="risk-coverage"))
        parts.append(self._operating_point_html(bundles))
        parts.append(self._temperature_html(bundles))
        return f'<div class="stack cal">{"".join(parts)}</div>'

    @staticmethod
    def _table(headers, rows: list, num_from: int = 1) -> str:
        head = "".join(f'<th{" class=num" if num_from <= i and h_ else ""}>{h.e(h_)}</th>'
                       for i, h_ in enumerate(headers))
        return (f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>')

    def _summary_html(self, bundles: dict, all_names: list) -> str:
        headers = ["Experiment", "N", "Accuracy", "Mean confidence", "Confidence gap",
                   "ECE", "MCE", "AURC", "Best possible AURC", "Excess AURC"]
        rows = []
        for name, bundle in bundles.items():
            gap = bundle.mean_confidence - bundle.accuracy
            excess = bundle.curve.aurc - bundle.curve.optimal_aurc
            gap_cls = " c-warn" if abs(gap) > 0.05 else ""
            rows.append(
                f'<tr><td><span class="row-nw">{h.dot(color_for_experiment(name, self.state.color_slots), size=10)}{h.name(name)}</span></td>'
                f'<td class="num">{bundle.n:,}</td>'
                f'<td class="num">{_fmt(bundle.accuracy)}</td>'
                f'<td class="num">{_fmt(bundle.mean_confidence)}</td>'
                f'<td class="num{gap_cls}">{h.e(f"{gap:+.3f}" if gap == gap else DASH)}</td>'
                f'<td class="num">{_fmt(bundle.reliability.ece)}</td>'
                f'<td class="num">{_fmt(bundle.reliability.mce)}</td>'
                f'<td class="num">{_fmt(bundle.curve.aurc)}</td>'
                f'<td class="num">{_fmt(bundle.curve.optimal_aurc)}</td>'
                f'<td class="num"><b>{_fmt(excess)}</b></td></tr>')
        return h.section("Summary", self._table(headers, rows), manual="calibration-summary")

    def _believable_html(self, bundles: dict) -> str:
        reliability = []
        for name in bundles:
            chart = self.reliability_charts.get(name)
            data = bundles[name].reliability
            if chart is None and data.disabled_reason:
                reliability.append(
                    f'<div class="cal-tile cal-off" data-key="calrel-{h.e(name)}">'
                    f'<b class="small row-nw">{h.name(name)}<span class="nowrap">— calibration unavailable</span></b>'
                    + h.banner(f"No reliability diagram: {data.disabled_reason}. ECE and MCE are blank for the "
                               f"same reason, and so is temperature scaling below -- the threshold sections are "
                               f"unaffected, since they only ever rank by confidence.", "info")
                    + '</div>')
            elif chart is not None:
                reliability.append(f'<div class="cal-tile" data-key="calrel-{h.e(name)}">{self._chart_html(chart)}</div>')
        histograms = [f'<div class="cal-tile" data-key="calhist-{h.e(name)}">{self._chart_html(self.histogram_charts[name])}</div>'
                      for name in bundles if name in self.histogram_charts]
        body = (f'<div class="cal-charts">{"".join(reliability)}</div>'
                f'<div class="cal-charts">{"".join(histograms)}</div>')
        return h.section("Is the confidence believable?", body, manual="reliability-diagram")

    def _operating_point_html(self, bundles: dict) -> str:
        return h.section("Pick an operating point",
                         self._threshold_block(bundles) + self._target_block(bundles) + self._class_block(),
                         manual="confidence-threshold")

    def _threshold_block(self, bundles: dict) -> str:
        threshold = self.threshold
        headers = ["Experiment", "Accepted", "Coverage", "Accuracy on accepted",
                   "Errors let through", "Sent to a human", "Accuracy of those", ""]
        rows = []
        for name, bundle in bundles.items():
            point = cal.threshold_point(bundle.index, threshold)
            buttons = (
                h.button("Errors let through", icon_name="warn", small=True, kind="ghost",
                         disabled=point.n_errors_accepted == 0,
                         act_attr=self.a("crops", experiment=name, threshold=threshold, which="accepted_errors"))
                + h.button("Abstained", icon_name="pan", small=True, kind="ghost",
                           disabled=point.n_abstained == 0,
                           act_attr=self.a("crops", experiment=name, threshold=threshold, which="abstained")))
            rows.append(
                f'<tr><td>{h.name(name)}</td>'
                f'<td class="num">{point.n_accepted:,}</td>'
                f'<td class="num">{_pct(point.coverage)}</td>'
                f'<td class="num"><b>{_fmt(point.accuracy)}</b></td>'
                f'<td class="num">{point.n_errors_accepted:,}</td>'
                f'<td class="num">{point.n_abstained:,}</td>'
                f'<td class="num">{_fmt(point.accuracy_abstained)}</td>'
                f'<td><div class="row cal-acts">{buttons}</div></td></tr>')
        step = 1.0 / SLIDER_DIVISIONS
        slider = (f'<input type="range" class="cal-rng" min="0" max="1" step="{step!r}" value="{threshold!r}" '
                  f'data-out="cal-thr-out" data-force="1" aria-label="Threshold"{self.c("threshold")}>')
        label = (f'<b class="small cal-thr">Accept at or above '
                 f'<span id="cal-thr-out" class="mono">{threshold:.3f}</span></b>')
        return (f'<div class="cal-sub"><h3>From a threshold</h3>'
                f'<div class="row cal-slider">{label}{slider}</div>'
                f'{self._table(headers, rows)}</div>')

    def _target_block(self, bundles: dict) -> str:
        field = h.text_input(self.target_text, name="value", label="Target accuracy on accepted", width="90px",
                             input_=self.i("target_typed"))
        controls = (f'<form class="row"{h.on_submit(self.name, "target_find")}>{field}'
                    f'{h.button("Find the threshold", kind="primary", icon_name="search", small=True, type_="submit")}</form>')
        headers = ["Experiment", "Required threshold", "Coverage reached", "Accepted",
                   "Accuracy on accepted", "Errors let through", "Sent to a human"]
        rows = []
        for name in bundles:
            point = self.target_points.get(name)
            if point is None:
                continue
            if point.unreachable:
                rows.append(f'<tr><td>{h.name(name)}</td><td class="num c-warn">unreachable</td>'
                            + "".join(f'<td class="num">{DASH}</td>' for _ in range(5)) + '</tr>')
                continue
            rows.append(
                f'<tr><td>{h.name(name)}</td>'
                f'<td class="num"><b>{_fmt(point.threshold, 4)}</b></td>'
                f'<td class="num"><b>{_pct(point.coverage)}</b></td>'
                f'<td class="num">{point.n_accepted:,}</td>'
                f'<td class="num">{_fmt(point.accuracy)}</td>'
                f'<td class="num">{point.n_errors_accepted:,}</td>'
                f'<td class="num">{point.n_abstained:,}</td></tr>')
        # Its own (i): this table answers the reverse question.
        table_head = (f'<div class="row cal-th"><b class="small">Working back from a target accuracy</b>'
                      f'{h.info_btn("target-accuracy")}</div>')
        note = ('<div class="muted small"><i>The most permissive threshold that still reaches the target -- i.e. the most the '
                'model can be trusted with. \'Unreachable\' means no threshold gets there, including '
                'one that accepts a single crop.</i></div>')
        return (f'<div class="cal-sub"><h3>From a target</h3>{controls}'
                f'{table_head}{self._table(headers, rows)}{note}</div>')

    def _class_block(self) -> str:
        selected = self._selected_names()
        picker = h.select([(n, n) for n in selected], self.class_experiment or "",
                          change=self.c("class_experiment"), label="Experiment")
        field = h.text_input(self.class_target_text, name="target", label="Target precision per class", width="90px",
                             input_=self.i("class_target_typed"))
        controls = (f'<form class="row-wrap"{h.on_submit(self.name, "class_compute")}>{picker}{field}'
                    f'{h.button("Compute", kind="primary", icon_name="chart", small=True, disabled=self.class_busy, type_="submit")}</form>')
        parts = [f'<div class="cal-sub"><h3>One threshold per class</h3>{controls}']
        if self.class_busy:
            parts.append(h.spinner("Computing per-class thresholds..."))
        if self.class_message:
            parts.append(h.banner(self.class_message, "error" if self.class_message_error else "info"))
        elif self._class_table_input is not None:
            parts.append(self.class_focus.render())
            parts.append(self._class_table_html())
        parts.append('</div>')
        return "".join(parts)

    def _class_table_html(self) -> str:
        all_rows, name, target = self._class_table_input
        rows_data = self.class_table_rows
        n_all_classes = len(all_rows)
        headers = ["Predicted class", "Predictions", "Threshold", "Kept", "Coverage in class", "Precision"]
        rows = []
        for row in rows_data:
            unreachable = row.unreachable
            threshold = ('<span class="c-warn">unreachable</span>' if unreachable else _fmt(row.threshold, 4))
            rows.append(
                f'<tr><td><b>{h.e(row.label)}</b></td>'
                f'<td class="num">{row.n_predicted:,}</td>'
                f'<td class="num">{threshold}</td>'
                f'<td class="num">{DASH if unreachable else f"{row.n_kept:,}"}</td>'
                f'<td class="num">{DASH if unreachable else _pct(row.coverage)}</td>'
                f'<td class="num">{DASH if unreachable else _fmt(row.precision)}</td></tr>')
        title = (f"{name} — target precision {target:g}"
                 + (f" -- {len(rows_data)} of {n_all_classes:,} classes, picked above"
                    if self.class_focus.many else ""))
        return (f'<div class="row cal-th"><b class="small">{h.e(title)}</b>{h.info_btn("per-class-thresholds")}</div>'
                f'{self._table(headers, rows)}')

    def _temperature_html(self, bundles: dict) -> str:
        button = h.button("Fit a temperature", act_attr=self.a("fit_temperature"), kind="primary",
                          small=True, disabled=self.temperature_busy)
        parts = [f'<div class="row">{button}</div>']
        if self.temperature_busy:
            parts.append(h.spinner("Fitting the temperature..."))
        if self.temperature_error:
            parts.append(h.banner(self.temperature_error, "error"))
        else:
            parts.append(self._temperature_table(bundles))
        return h.section("Fix the confidence: temperature scaling", "".join(parts),
                         manual="temperature-scaling",
                         sub="Changes no prediction: only the confidence attached to it moves.")

    def _temperature_table(self, bundles: dict) -> str:
        fitted = [(name, bundle) for name, bundle in bundles.items() if bundle.temperature is not None]
        if not fitted:
            return h.banner("Not fitted yet. Fitting reads every selected experiment's score matrix once; "
                            "it changes no prediction, only the confidence attached to it.", "info")
        headers = ["Experiment", "Temperature", "Log loss before", "after",
                   "ECE before", "after", "Rows fitted on"]
        rows = []
        notes = []
        for name, bundle in fitted:
            fit = bundle.temperature
            if fit.disabled_reason:
                notes.append(f"{name}: not fitted -- {fit.disabled_reason}.")
                continue
            improved = fit.ece_after < fit.ece_before
            rows.append(
                f'<tr><td>{h.name(name)}</td>'
                f'<td class="num"><b>{_fmt(fit.temperature, 3)}</b></td>'
                f'<td class="num">{_fmt(fit.nll_before)}</td>'
                f'<td class="num">{_fmt(fit.nll_after)}</td>'
                f'<td class="num">{_fmt(fit.ece_before)}</td>'
                f'<td class="num {"c-ok" if improved else "c-warn"}"><b>{_fmt(fit.ece_after)}</b></td>'
                f'<td class="num">{fit.n_fit_rows:,} of {fit.n_rows:,}</td></tr>')
            if not improved:
                notes.append(
                    f"{name}: the fitted temperature improves the log loss but not the ECE. That "
                    f"happens when the confidence carries little information about correctness -- "
                    f"check the risk-coverage curve above before trusting either number.")
        return (self._table(headers, rows)
                + "".join(f'<div class="small c-warn"><i>{h.e(n)}</i></div>' for n in notes))

    # ---- snapshots (see core/view_state.py) ----

    def view_state(self) -> dict:
        self._sync_checks()
        return {
            "selected": self._selected_names(),
            "threshold": float(self.threshold),
            "target_accuracy": str(self.target_text or ""),
            "class_experiment": self.class_experiment or "",
            "target_precision": str(self.class_target_text or ""),
            "class_focus": self.class_focus.view_state(),
        }

    def restore_view_state(self, state: dict) -> list:
        warnings = []
        loaded = set(self.state.experiments)

        self._sync_checks()
        selected = state.get("selected")
        if isinstance(selected, list):
            wanted = {n for n in selected if isinstance(n, str)}
            gone = sorted(wanted - loaded)
            if gone:
                warnings.append(f"Calibration: experiment(s) {', '.join(gone)} are not loaded.")
            for name in self.checks:
                self.checks[name] = name in wanted
            # The per-class dropdown is derived from the ticks: re-derive it.
            self._sync_checks()

        try:
            threshold = float(state["threshold"])
        except (KeyError, TypeError, ValueError):
            threshold = None
        if threshold is not None:
            self.threshold = max(0.0, min(1.0, threshold))

        if state.get("target_accuracy"):
            self.target_text = str(state["target_accuracy"])
        if state.get("target_precision"):
            self.class_target_text = str(state["target_precision"])

        self.class_focus.restore_view_state(state.get("class_focus"))

        class_experiment = state.get("class_experiment") or ""
        if class_experiment:
            if class_experiment in loaded:
                self.class_experiment = class_experiment
            else:
                warnings.append(f"Calibration: experiment '{class_experiment}' is not loaded.")

        self.refresh()
        return warnings
