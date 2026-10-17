"""Confidence: per-prediction confidence for one experiment at a time --
margin between the top two scores and entropy of the softmax-normalized
score distribution (see core/confidence.py), independent of whether a
prediction was actually correct.

Same behaviour as lookin.ui.confidence_view: a summary row, the
correct/incorrect distributions of margin and entropy (for every row, then
for one true class), the per-class means (a ClassFocus past FOCUS_MAX
classes), the margin vs. entropy scatter (sampled), and the confidently
wrong predictions grouped by (true, predicted) pair, each opening in the
Gallery.

Everything is drawn in the background from the per-experiment cache; a
data change only marks the tab stale.
"""
from __future__ import annotations

import traceback

import numpy as np

from lookin.core import class_focus as cf
from lookin.core.chart_style import subplots
from lookin.core.confidence import MARGIN_COL, NORMALIZED_ENTROPY_COL, compute_confidence, confident_and_wrong
from lookin.ui.components import ACCENT, GRID_NONE, style_axes
from lookin.ui.confidence_view import (
    CHART_WIDTH, CONFIDENCE_FOCUS_CRITERIA, CORRECT_COLOR, INCORRECT_COLOR,
    PERCLASS_BAR_COLOR, SCATTER_MAX_POINTS_DEFAULT, SCATTER_SEED, _chart_box,
)
from lookin.web import html as h
from lookin.web.views.base import RebuildView
from lookin.web.views.common import score_diagnostics_banner, uuid_overlap_banner
from lookin.web.views.media import fig_html
from lookin.web.views.widgets import ClassFocus, class_select

# What a redraw can be asked for; "all" is the original's _render().
PART_REFRESH = "refresh"
PART_ALL = "all"
PART_CLASS = "class"
PART_PERCLASS = "perclass"
PART_SCATTER = "scatter"
PART_WRONG = "wrong"

METRIC_OPTIONS = (("margin", "Margin (top1 - top2)"), ("entropy", "Entropy (normalized)"))


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _int(text, default: int) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _float(text, default: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _boxed(chart: str, width: int) -> str:
    """A chart at most `width` pixels wide (the original's image width),
    narrower when the card is."""
    return f'<div class="conf-chart" style="max-width:{int(width)}px">{chart}</div>'


def _fmt(value) -> str:
    return f"{value:.3f}" if value == value else "—"


class ConfidenceView(RebuildView):
    name = "confidence"
    title = "Confidence"

    def __init__(self, session):
        super().__init__(session)
        self._confidence_cache: dict = {}   # experiment name -> ConfidenceData
        self.built = False
        self.exp_names: list = []
        self.experiment = None

        # what the user typed / picked (read when the matching part is drawn)
        self.scatter_sample = str(SCATTER_MAX_POINTS_DEFAULT)
        self.class_value = None
        self.class_labels: list = []
        self.cw_metric = "margin"
        self.cw_percentile = "25"
        self.class_focus = ClassFocus(self.name, "focus", CONFIDENCE_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the per-class chart",
                                      support_criteria=(cf.FOCUS_LEAST_CONFIDENT, cf.FOCUS_HIGHEST_ENTROPY))

        # drawn in the background, rendered from here
        self._drawn_for = None              # the experiment the parts below were drawn for
        self.empty_msg = ""
        self.warnings: list = []
        self.summary_html = ""
        self.hist_html = ""
        self.class_hist_html = ""
        self.perclass_html = ""
        self.scatter_html = ""
        self.wrong_html = ""
        self._pending: set = set()

    def subtitle(self) -> str:
        return ("How sure the model was about each prediction for the selected experiment -- margin "
                "between its top two scores and the entropy of its softmax-normalized distribution -- "
                "independent of whether the prediction was actually correct.")

    # ---- lifecycle ---------------------------------------------------

    def on_shown(self) -> None:
        if not self.built:
            self.built = True
            self._refresh_in_background()

    def on_update(self, **_):
        self._confidence_cache.clear()
        self._refresh_in_background()

    def _refresh_in_background(self) -> None:
        """The original's "Update now": the experiment list again, every
        part redrawn, the stale banner gone."""
        self.progress_text = "Computing confidence diagnostics..."
        self._queue(PART_REFRESH)

    def _redraw(self, part: str) -> None:
        """One part again from the cache (the original's per-control
        redraws), leaving the stale banner as it was."""
        if not self.progress_text:
            self.progress_text = "Updating..."
        self._queue(part)

    def _queue(self, part: str) -> None:
        """Every request runs the same work on this view's job; pending
        parts accumulate, so a request the job coalesces never loses one."""
        self._pending.add(part)
        self.error = ""

        def run():
            try:
                self._run_pending()
            finally:
                self.progress_text = ""
                self.s.mark(self.name, "chrome")

        def failed(ex):
            self.progress_text = ""
            self.error = f"Error computing confidence: {ex}"
            self.s.mark(self.name)

        self.s.job(self.title).run(run, on_error=failed)
        self.s.mark(self.name)

    def _run_pending(self) -> None:
        parts, self._pending = self._pending, set()
        if not parts:
            return
        if PART_REFRESH in parts:
            self._rebuild_experiment_list()
            self._render_all()
            self.dirty = False
            return
        name = self.experiment
        if PART_ALL in parts or name != self._drawn_for or name not in self._confidence_cache:
            self._render_all()
            return
        state = self.state
        if state.dataset is None:
            return
        data = self._confidence_cache[name]
        label_col = state.dataset.columns.label_col
        try:
            if PART_CLASS in parts:
                self.set_progress("Drawing confidence distributions for one class...")
                self._render_class_histograms(data, label_col)
            if PART_PERCLASS in parts:
                self.set_progress("Drawing per-class breakdown...")
                self._render_perclass(data, label_col)
            if PART_SCATTER in parts:
                self.set_progress("Drawing margin vs. entropy scatter...")
                self._render_scatter(data)
            if PART_WRONG in parts:
                self.set_progress("Finding confidently-wrong predictions...")
                self._render_confident_wrong(data, label_col)
        except Exception as ex:  # noqa: BLE001
            _log(f"error rendering confidence tab: {ex}")
            traceback.print_exc()
            self.summary_html = h.banner(f"Error: {ex}", "error")

    def on_data_change(self) -> None:
        self.dirty = True

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """The numbers are unchanged, only the label is (see the original)."""
        if old_name in self._confidence_cache:
            self._confidence_cache[new_name] = self._confidence_cache.pop(old_name)
        if self._drawn_for == old_name:
            self._drawn_for = new_name
        selected = self.experiment
        self._rebuild_experiment_list()
        if selected == old_name:
            self.experiment = new_name
        if self.built and not self.dirty:
            self._redraw(PART_ALL)
        self.refresh()

    # ---- experiment / class lists ------------------------------------

    def _rebuild_experiment_list(self) -> None:
        self.exp_names = list(self.state.experiments)
        if self.experiment not in self.exp_names:
            self.experiment = self.exp_names[0] if self.exp_names else None

    def _rebuild_class_list(self, data, label_col: str) -> None:
        labels = sorted(data.df[label_col].dropna().astype(str).unique().tolist())
        self.class_labels = labels
        if self.class_value not in labels:
            self.class_value = labels[0] if labels else None

    # ---- cached accessor ---------------------------------------------

    def _get_confidence(self, name: str):
        if name not in self._confidence_cache:
            state = self.state
            exp = state.experiments[name]
            merged = state.merged(name)
            self.set_progress(f"Computing margin / entropy for '{name}' ({len(merged):,} rows)...")
            _log(f"computing confidence diagnostics for '{name}' ({len(merged):,} merged rows)...")
            self._confidence_cache[name] = compute_confidence(merged, exp.score_cols, exp.class_labels)
        return self._confidence_cache[name]

    # ---- the whole tab (the original's _render) ----------------------

    def _clear_parts(self) -> None:
        self.warnings = []
        self.summary_html = self.hist_html = self.class_hist_html = ""
        self.perclass_html = self.scatter_html = self.wrong_html = ""

    def _render_all(self) -> None:
        state = self.state
        self._clear_parts()
        self.empty_msg = ""
        if state.dataset is None or not state.experiments:
            self.empty_msg = "Load a dataset and at least one experiment (results CSV) in the 'Data' tab first."
            self.class_focus.hide()
            self._drawn_for = None
            return
        if not self.experiment:
            self._rebuild_experiment_list()
        name = self.experiment
        if not name:
            return
        if name not in state.experiments:
            # renamed or removed since the list was built
            self._rebuild_experiment_list()
            name = self.experiment
        self._drawn_for = name

        warnings = []
        banner = uuid_overlap_banner(state.uuid_overlap(name), "dataset", name)
        if banner:
            warnings.append(banner)
        try:
            data = self._get_confidence(name)
            diag_banner = score_diagnostics_banner(data.score_diagnostics, name)
            if diag_banner:
                warnings.append(diag_banner)
            if data.softmax_applied:
                warnings.append(h.banner(
                    "Score columns do not sum to 1, so entropy is computed on softmax(scores) rather than "
                    "the raw scores directly -- margin is unaffected, it only compares the raw top two scores.",
                    "info"))
            self.warnings = warnings

            label_col = state.dataset.columns.label_col
            self.set_progress("Building the summary...")
            self._render_summary(data)
            self.set_progress("Drawing confidence distributions...")
            self._render_histograms(data)
            self.set_progress("Drawing confidence distributions for one class...")
            self._rebuild_class_list(data, label_col)
            self._render_class_histograms(data, label_col)
            self.set_progress("Drawing per-class breakdown...")
            self._render_perclass(data, label_col)
            self.set_progress("Drawing margin vs. entropy scatter...")
            self._render_scatter(data)
            self.set_progress("Finding confidently-wrong predictions...")
            self._render_confident_wrong(data, label_col)
        except Exception as ex:  # noqa: BLE001
            _log(f"error rendering confidence tab: {ex}")
            traceback.print_exc()
            self.warnings = warnings
            self.summary_html = h.banner(f"Error: {ex}", "error")

    # ---- summary -----------------------------------------------------

    def _render_summary(self, data) -> None:
        df = data.df
        n = len(df)
        accuracy = float(df["correct"].mean()) if n else float("nan")
        margin_valid = df[MARGIN_COL].dropna()
        entropy_valid = df[NORMALIZED_ENTROPY_COL].dropna()
        headers = ["N", "Accuracy", "Mean margin", "Median margin", "Mean entropy (norm)", "Median entropy (norm)"]
        dash = "—"
        values = [
            f"{n:,}",
            f"{accuracy:.3f}" if accuracy == accuracy else dash,
            f"{margin_valid.mean():.3f}" if len(margin_valid) else dash,
            f"{margin_valid.median():.3f}" if len(margin_valid) else dash,
            f"{entropy_valid.mean():.3f}" if len(entropy_valid) else dash,
            f"{entropy_valid.median():.3f}" if len(entropy_valid) else dash,
        ]
        table = (f'<div class="tbl-wrap"><table class="tbl compact conf-summary"><thead><tr>'
                 f'{"".join(f"<th class=num>{h.e(c)}</th>" for c in headers)}</tr></thead>'
                 f'<tbody><tr>{"".join(f"<td class=num>{h.e(v)}</td>" for v in values)}</tr></tbody></table></div>')
        self.summary_html = h.section("Summary", table, manual="margin-entropy")

    # ---- one correct/incorrect histogram, for any row subset ----------

    @staticmethod
    def _hist_fig(df, col: str, title: str, xlabel: str):
        values = df[col].dropna()
        if values.empty:
            return None
        correct_vals = df.loc[df["correct"], col].dropna()
        incorrect_vals = df.loc[~df["correct"], col].dropna()
        fig, ax = subplots(figsize=(9.5, 3.2))
        bins = np.linspace(float(values.min()), float(values.max()), 41)
        if len(bins) < 2 or bins[0] == bins[-1]:
            bins = 20
        if len(correct_vals):
            ax.hist(correct_vals, bins=bins, alpha=0.6, density=True,
                    color=CORRECT_COLOR, label=f"correct ({len(correct_vals):,})")
        if len(incorrect_vals):
            ax.hist(incorrect_vals, bins=bins, alpha=0.6, density=True,
                    color=INCORRECT_COLOR, label=f"incorrect ({len(incorrect_vals):,})")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("density", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9)
        style_axes(ax)
        fig.tight_layout()
        return fig

    def _hist_charts(self, df, name_prefix: str = "") -> list:
        charts = []
        width = CHART_WIDTH - 36
        margin_fig = self._hist_fig(df, MARGIN_COL, "Margin (top1 - top2)", "margin")
        if margin_fig is not None:
            charts.append(_boxed(fig_html(self.s, margin_fig, f"{name_prefix}Margin distribution"), width))
        entropy_fig = self._hist_fig(df, NORMALIZED_ENTROPY_COL, "Entropy (normalized)", "entropy / log(n classes)")
        if entropy_fig is not None:
            charts.append(_boxed(fig_html(self.s, entropy_fig, f"{name_prefix}Entropy distribution"), width))
        return charts

    # ---- correct/incorrect distributions ------------------------------

    def _render_histograms(self, data) -> None:
        charts = self._hist_charts(data.df)
        if not charts:
            self.hist_html = h.banner("Not enough data for a confidence distribution (needs at least two classes).",
                                      "info")
            return
        self.hist_html = h.section("Confidence distributions", f'<div class="conf-charts">{"".join(charts)}</div>',
                                   manual="margin-entropy", sub="Split by whether the prediction was correct.")

    # ---- the same, for one class -------------------------------------

    def _render_class_histograms(self, data, label_col: str) -> None:
        cls = self.class_value
        if not cls:
            self.class_hist_html = h.banner("No class to show yet.", "info")
            return
        df = data.df[data.df[label_col].astype(str) == cls]
        charts = self._hist_charts(df, name_prefix=f"{cls} -- ")
        if not charts:
            self.class_hist_html = ("not-enough", h.banner(f"Not enough data for class '{cls}'.", "info"))
            return
        self.class_hist_html = ("charts", f'<div class="conf-charts">{"".join(charts)}</div>')

    def _class_hist_section(self) -> str:
        part = self.class_hist_html
        if not isinstance(part, tuple):
            return part      # "" or the "No class" banner
        kind, body = part
        picker = class_select(self.class_labels, self.class_value, self.c("class"), label="Class", width=260)
        sub = ("Same as above, restricted to the rows whose true label is the class chosen below."
               if kind == "charts" else "")
        return h.section("Confidence distributions by class",
                         f'<div class="row">{picker}</div>{body}', manual="margin-entropy", sub=sub)

    # ---- per-class breakdown -----------------------------------------

    def _on_focus_change(self) -> None:
        self._redraw(PART_PERCLASS)

    def _render_perclass(self, data, label_col: str) -> None:
        df = data.df.dropna(subset=[MARGIN_COL, NORMALIZED_ENTROPY_COL])
        if df.empty:
            self.class_focus.hide()
            self.perclass_html = h.banner("Not enough data for a per-class breakdown (needs at least two classes).",
                                          "info")
            return

        grouped = df.groupby(label_col).agg(
            mean_margin=(MARGIN_COL, "mean"),
            mean_entropy=(NORMALIZED_ENTROPY_COL, "mean"),
            n=(MARGIN_COL, "count"),
        ).sort_values("mean_margin")
        n_all_classes = len(grouped)
        table = grouped.reset_index().rename(columns={label_col: "label", "n": "support"})
        table["label"] = table["label"].astype(str)
        rankings = {
            cf.FOCUS_LEAST_CONFIDENT: ("mean_margin", True),
            cf.FOCUS_HIGHEST_ENTROPY: ("mean_entropy", False),
            cf.FOCUS_LARGEST_SUPPORT: ("support", False),
        }
        focus = self.class_focus.resolve(
            table["label"].tolist(),
            lambda criterion: cf.rank_by(table, *rankings[criterion],
                                         min_support=self.class_focus.min_support_for(criterion))
            if criterion in rankings else [])
        if self.class_focus.many:
            # Kept least confident first, as the chart reads.
            keep = set(focus)
            grouped = grouped[grouped.index.astype(str).isin(keep)]

        labels = grouped.index.astype(str).tolist()
        n_classes = len(labels)
        fig_width = max(6.0, min(18.0, n_classes * 0.35))
        fig, (ax1, ax2) = subplots(2, 1, figsize=(fig_width, 6.0), sharex=True)
        x = np.arange(n_classes)
        ax1.bar(x, grouped["mean_margin"], color=ACCENT)
        ax1.set_ylabel("mean margin", fontsize=10)
        ax1.set_title("Confidence per class (true label) -- least confident first", fontsize=12)
        ax2.bar(x, grouped["mean_entropy"], color=PERCLASS_BAR_COLOR)
        ax2.set_ylabel("mean entropy\n(normalized)", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        for ax in (ax1, ax2):
            ax.tick_params(labelsize=9)
            style_axes(ax, x=GRID_NONE)
        fig.tight_layout()

        sub = ("Classes the model is systematically unsure about, even where its accuracy on them looks fine."
               + (f" {len(labels)} of {n_all_classes:,} classes, picked under 'Classes shown' above."
                  if self.class_focus.many else ""))
        box = _chart_box(fig)
        chart = _boxed(fig_html(self.s, fig, "Per-class confidence"), box["width"])
        self.perclass_html = h.section("Per-class confidence", chart, manual="margin-entropy", sub=sub)

    # ---- margin vs. entropy scatter -----------------------------------

    def _render_scatter(self, data) -> None:
        df = data.df.dropna(subset=[MARGIN_COL, NORMALIZED_ENTROPY_COL])
        if df.empty:
            self.scatter_html = h.banner(
                "Not enough data for the margin vs. entropy scatter (needs at least two classes).", "info")
            return

        sample_size = _int(self.scatter_sample, SCATTER_MAX_POINTS_DEFAULT)
        plotted = df
        if sample_size > 0 and sample_size < len(df):
            plotted = df.sample(n=sample_size, random_state=SCATTER_SEED)

        correct = plotted[plotted["correct"]]
        incorrect = plotted[~plotted["correct"]]
        fig, ax = subplots(figsize=(7.2, 5.4))
        if len(correct):
            ax.scatter(correct[MARGIN_COL], correct[NORMALIZED_ENTROPY_COL], s=10, alpha=0.5,
                       color=CORRECT_COLOR, label=f"correct ({len(correct):,})")
        if len(incorrect):
            ax.scatter(incorrect[MARGIN_COL], incorrect[NORMALIZED_ENTROPY_COL], s=10, alpha=0.6,
                       color=INCORRECT_COLOR, label=f"incorrect ({len(incorrect):,})")
        ax.set_xlabel("margin (top1 - top2)", fontsize=10)
        ax.set_ylabel("entropy (normalized)", fontsize=10)
        ax.set_title("Margin vs. entropy", fontsize=12)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9)
        style_axes(ax)
        fig.tight_layout()

        note = f" Showing {len(plotted):,} of {len(df):,} row(s)." if len(plotted) < len(df) else ""
        chart = _boxed(fig_html(self.s, fig, "Margin vs entropy scatter"), 640)
        self.scatter_html = ("ok", f"One point per prediction, colored by correctness.{note}", chart)

    def _scatter_section(self) -> str:
        part = self.scatter_html
        if not isinstance(part, tuple):
            return part
        _, sub, chart = part
        controls = (f'<div class="row">'
                    f'{h.text_input(self.scatter_sample, label="Scatter sample size", width="150px", input_=self.i("scatter_text"), enter=h.on_enter(self.name, "scatter_apply"), extra=" data-force=\"1\"")}'
                    f'{h.button(icon_name="refresh", act_attr=self.a("scatter_apply"), title="Redraw the scatter", small=True)}'
                    f'</div>')
        return h.section("Margin vs. entropy", controls + chart, manual="margin-entropy", sub=sub)

    # ---- confidently wrong -------------------------------------------

    def _render_confident_wrong(self, data, label_col: str) -> None:
        metric = self.cw_metric or "margin"
        top_pct = _float(self.cw_percentile, 25.0)
        metric_label = "margin" if metric == "margin" else "entropy"

        df = data.df
        n_total = len(df)
        n_wrong = int((~df["correct"]).sum()) if n_total else 0
        subset = confident_and_wrong(data, label_col, metric, top_pct) if n_total else df.iloc[0:0]
        n_confident_wrong = len(subset)

        if n_total == 0:
            self.wrong_html = h.banner("No predictions to check.", "info")
            return
        if n_confident_wrong == 0:
            self.wrong_html = h.banner(
                f"No confidently-wrong predictions among the top {top_pct:.0f}% most confident by {metric_label}.",
                "info")
            return
        pct_of_wrong = 100.0 * n_confident_wrong / n_wrong if n_wrong else 0.0
        pct_of_total = 100.0 * n_confident_wrong / n_total
        out = [h.banner(
            f"{n_confident_wrong:,} confidently-wrong prediction(s) -- {pct_of_wrong:.1f}% of all "
            f"{n_wrong:,} incorrect predictions, {pct_of_total:.2f}% of all {n_total:,} predictions, "
            f"among the top {top_pct:.0f}% most confident by {metric_label}.",
            "warn")]
        grouped = (
            subset.groupby([label_col, "predicted_label"])
            .agg(n=(MARGIN_COL, "count"), mean_margin=(MARGIN_COL, "mean"),
                 mean_entropy=(NORMALIZED_ENTROPY_COL, "mean"))
            .reset_index()
            .sort_values("n", ascending=False)
        )
        rows = []
        for _, r in grouped.iterrows():
            true_label, predicted_label = str(r[label_col]), str(r["predicted_label"])
            open_btn = h.button(icon_name="external", small=True, kind="ghost",
                                title="Open this pair in the Gallery",
                                act_attr=self.a("open_gallery", t=true_label, p=predicted_label))
            rows.append(f'<tr><td>{h.e(true_label)}</td><td>{h.e(predicted_label)}</td>'
                        f'<td class="num">{int(r["n"]):,}</td><td class="num">{r["mean_margin"]:.3f}</td>'
                        f'<td class="num">{_fmt(r["mean_entropy"])}</td><td>{open_btn}</td></tr>')
        heads = "".join(f'<th{" class=num" if c in ("count", "mean margin", "mean entropy") else ""}>{h.e(c)}</th>'
                        for c in ["true label", "predicted", "count", "mean margin", "mean entropy", ""])
        out.append(f'<div class="tbl-wrap conf-wrong"><table class="tbl compact"><thead><tr>{heads}</tr></thead>'
                   f'<tbody>{"".join(rows)}</tbody></table></div>')
        self.wrong_html = "".join(out)

    def _wrong_section(self) -> str:
        controls = (f'<div class="row">'
                    f'{h.select(METRIC_OPTIONS, self.cw_metric, change=self.c("cw_metric"), label="Confidence measure")}'
                    f'{h.text_input(self.cw_percentile, label="Top confidence percentile", width="120px", input_=self.i("cw_text"), enter=h.on_enter(self.name, "cw_apply"), extra=" data-force=\"1\"")}'
                    f'{h.button(icon_name="refresh", act_attr=self.a("cw_apply"), title="Recompute", small=True)}'
                    f'</div>')
        return h.section("Confidently wrong", controls + self.wrong_html, manual="confidently-wrong",
                         sub="Predictions that are both incorrect and among the model's most confident, by the "
                             "measure and percentile chosen below.")

    # ---- render ------------------------------------------------------

    def render(self) -> str:
        if not self.built:
            self.on_shown()
        head = [self.stale_html()]
        if self.empty_msg:
            return (f'<div class="stack conf">{"".join(head)}'
                    f'{h.banner(self.empty_msg, "info")}{self.progress_html()}</div>')
        exp_select = h.select([(n, n) for n in self.exp_names], self.experiment or "",
                              change=self.c("experiment"), label="Experiment")
        head.append(f'<div class="row">{exp_select}</div>')
        head += self.warnings
        head.append(self.progress_html())
        body = [self.summary_html]
        drawn = self._drawn_for is not None
        if drawn:
            body.append(f'<div class="grid2 conf-pair">{self.hist_html}{self._class_hist_section()}</div>')
            body.append(self.class_focus.render())
            body.append(self.perclass_html)
            body.append(f'<div class="grid2 conf-pair">{self._scatter_section()}{self._wrong_section()}</div>')
        return f'<div class="stack conf">{"".join(head)}{"".join(p for p in body if p)}</div>'

    # ---- actions -----------------------------------------------------

    def on_experiment(self, value: str = "", **_):
        if value in self.exp_names:
            self.experiment = value
            self._redraw(PART_ALL)

    def on_class(self, value: str = "", **_):
        value = str(value)
        if value not in self.class_labels:
            return
        self.class_value = value
        if self.experiment in self._confidence_cache and self.state.dataset is not None:
            self._redraw(PART_CLASS)

    def on_focus(self, op: str = "", **args):
        self.class_focus.handle(op, args)

    def on_scatter_text(self, value: str = "", **_):
        self.scatter_sample = str(value)

    def on_scatter_apply(self, value=None, **_):
        if value is not None:
            self.scatter_sample = str(value)
        if self.experiment in self._confidence_cache:
            self._redraw(PART_SCATTER)

    def on_cw_metric(self, value: str = "", **_):
        self.cw_metric = value if value in {k for k, _ in METRIC_OPTIONS} else "margin"
        self._cw_redraw()

    def on_cw_text(self, value: str = "", **_):
        self.cw_percentile = str(value)

    def on_cw_apply(self, value=None, **_):
        if value is not None:
            self.cw_percentile = str(value)
        self._cw_redraw()

    def _cw_redraw(self) -> None:
        if self.experiment in self._confidence_cache and self.state.dataset is not None:
            self._redraw(PART_WRONG)

    def on_open_gallery(self, t: str = "", p: str = "", **_):
        if self.state.dataset is None or self.experiment not in self.state.experiments:
            return
        self.s.view("gallery").apply_external_filter(self.experiment, t, p)
        self.s.goto("gallery")
