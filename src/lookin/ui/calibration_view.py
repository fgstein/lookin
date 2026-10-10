"""Tab: what a model's confidence is worth, and what can be done with it.

The Metrics tab answers "how often is it right". This one answers the two
questions that come next and that an accuracy cannot:

* **Can the confidence be believed?** The reliability diagram and ECE/MCE,
  plus the fix -- one fitted temperature, with the ECE before and after it.
* **What does it buy?** Accept a prediction above a threshold, send the
  rest to a human. The risk-coverage curve, a threshold explorer that runs
  both directions ("at 0.9, what do I get" and "to reach 99%, what do I
  set"), and per-class thresholds for when one class is not like the others.

Why the two live together
-------------------------
Because reading only one of them is how a decision goes wrong. A model can
be badly calibrated and perfectly usable behind a threshold -- its
confidence ranks the predictions correctly and merely reports the wrong
number, which a threshold does not care about. And a model can be
beautifully calibrated and useless, if it is calibrated at 0.5 everywhere.
Calibration is about the *value* of the confidence, abstention about its
*order*, and a tab showing one without the other invites acting on half the
picture.

What stays enabled on scores that are not probabilities
--------------------------------------------------------
Exactly what still means something (see core/calibration for the full
argument): everything that only ranks by confidence -- the risk-coverage
curve, the thresholds, the accepted/abstained counts -- keeps working on a
sigmoid head or raw logits, because those are empirical facts about the
rows. Everything that reads the confidence *as a probability* -- ECE/MCE,
the reliability diagram, temperature scaling -- is refused with the reason
shown, rather than drawn as a plausible-looking result of nothing.

Cost
----
The expensive part is one pass per experiment to sort its confidences (see
core/calibration.ThresholdIndex); after that the slider, the target search
and the whole summary table are lookups, so the explorer is live rather
than a form with an Apply button. Like Metrics, that pass happens only when
the user asks for it (Update now), never on a tab switch.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Optional

import flet as ft
import numpy as np
import pandas as pd

from lookin.core.chart_style import subplots
from lookin.core import calibration as cal
from lookin.core import class_focus as cf
from lookin.core.metrics import compute_reliability, diagnose_scores
from lookin.ui.carousel import Carousel
from lookin.ui.class_matrix import ClassFocus
from lookin.ui.components import (
    info_button,
    ACCENT, BackgroundRebuild, StaleBanner, card, color_for_experiment,
    fig_to_zoomable_image, info_banner, legend_below, legend_columns,
    legend_height_inches, reserve_legend_space, score_diagnostics_banner, section_title,
    style_axes,
)

CHART_MAX_WIDTH = 1200
CHART_PX_PER_INCH = 90

DEFAULT_THRESHOLD = 0.9
# Which classes the per-class thresholds table shows past FOCUS_MAX (see ClassFocus).
CALIBRATION_FOCUS_CRITERIA = (
    (cf.FOCUS_LOWEST_COVERAGE, "Lowest coverage"),
    (cf.FOCUS_MOST_PREDICTED, "Most predictions"),
    (cf.FOCUS_MANUAL, "Hand-picked"),
)
DEFAULT_TARGET_ACCURACY = 0.99
DEFAULT_TARGET_PRECISION = 0.95


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value == value else "—"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%" if value == value else "—"


def _chart_box(fig) -> dict:
    fig_w, fig_h = fig.get_size_inches()
    width = min(CHART_MAX_WIDTH, max(600.0, fig_w * CHART_PX_PER_INCH))
    return {"width": int(width), "height": int(width * fig_h / fig_w)}


@dataclass
class ExperimentCalibration:
    """Everything this tab keeps for one experiment between renders.

    Deliberately holds no copy of the merged frame and no per-row label
    array: the threshold index is the only per-row structure kept (8 bytes
    a row), and anything needing the rows themselves -- the crops behind an
    operating point, the per-class thresholds, the temperature fit -- reads
    the merge again on demand, one experiment at a time, behind a click.
    """
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


class CalibrationView:
    def __init__(self, ctx):
        self.ctx = ctx
        self.experiment_checks: dict = {}
        self._cache: dict = {}          # experiment name -> ExperimentCalibration
        self._class_rows: dict = {}     # (name, target) -> list[ClassThreshold]
        self._dirty = False

        self.selection_row = ft.Row(spacing=14, wrap=True)
        self.progress_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_banner = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2), self.progress_text],
            spacing=10, visible=False,
        )
        self.stale_banner = StaleBanner(self._on_update_click)
        self.rebuild_job = BackgroundRebuild(ctx, "Calibration")
        self.class_job = BackgroundRebuild(ctx, "Per-class thresholds")
        self.temperature_job = BackgroundRebuild(ctx, "Temperature")

        # ---- threshold explorer controls ----
        self.threshold_slider = ft.Slider(
            min=0.0, max=1.0, value=DEFAULT_THRESHOLD, divisions=200,
            label="{value}", expand=True,
            on_change=self._on_threshold_slide, on_change_end=self._on_threshold_settled,
        )
        self.threshold_label = ft.Text(f"Accept at or above {DEFAULT_THRESHOLD:.3f}",
                                       size=13, weight=ft.FontWeight.BOLD)
        self.target_field = ft.TextField(
            label="Target accuracy on accepted", value=str(DEFAULT_TARGET_ACCURACY),
            width=220, border_radius=8, on_submit=self._on_target_change,
        )
        self.threshold_table_area = ft.Column(spacing=8)
        self.target_table_area = ft.Column(spacing=8)

        # ---- per-class thresholds ----
        self.class_experiment_dropdown = ft.Dropdown(label="Experiment", width=240, border_radius=8)
        self.class_target_field = ft.TextField(
            label="Target precision per class", value=str(DEFAULT_TARGET_PRECISION),
            width=220, border_radius=8,
        )
        self.class_area = ft.Column(spacing=8)
        # Past FOCUS_MAX classes the per-class table shows a focus set.
        self.class_focus = ClassFocus(CALIBRATION_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the per-class table",
                                      support_criteria=(cf.FOCUS_LOWEST_COVERAGE,),
                                      support_noun="predictions")
        self._class_table_input = None
        self.class_progress = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2),
             ft.Text("Computing per-class thresholds...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)],
            spacing=10, visible=False,
        )

        # ---- temperature scaling ----
        self.temperature_area = ft.Column(spacing=8)
        self.temperature_progress = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2),
             ft.Text("Fitting the temperature...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)],
            spacing=10, visible=False,
        )

        # ---- output areas ----
        self.warnings_area = ft.Column(spacing=6)
        self.summary_area = ft.Column()
        self.reliability_area = ft.Row(spacing=16, wrap=True, run_spacing=16)
        self.histogram_area = ft.Row(spacing=16, wrap=True, run_spacing=16)
        self.curve_area = ft.Container()
        self.root = ft.Container(padding=20)

        self.carousel_title = ft.Text(weight=ft.FontWeight.BOLD)
        self.carousel_content = ft.Container()
        self.carousel_dialog = ft.AlertDialog(
            modal=False, title=self.carousel_title, content=self.carousel_content,
            on_dismiss=self._close_carousel,
            actions=[ft.TextButton("Close", on_click=self._close_carousel)],
        )

    # ---- experiment picker ---------------------------------------------
    def _rebuild_experiment_checks(self) -> None:
        """A plain wrapped row of checkboxes rather than the Metrics tab's
        drag-to-reorder list: the order still comes from AppState (so the
        colors match everywhere), this tab just does not own it."""
        state = self.ctx.state
        existing = {name: cb.value for name, cb in self.experiment_checks.items()}
        self.experiment_checks.clear()
        controls = []
        names = list(state.experiments)
        for name in names:
            cb = ft.Checkbox(label=name, value=existing.get(name, True), on_change=self._on_selection_change)
            self.experiment_checks[name] = cb
            controls.append(ft.Row([
                ft.Container(width=10, height=10, bgcolor=color_for_experiment(name, names), border_radius=5),
                cb,
            ], spacing=6, tight=True))
        self.selection_row.controls = controls

        selected = self._selected_names()
        self.class_experiment_dropdown.options = [ft.dropdown.Option(n) for n in selected]
        if self.class_experiment_dropdown.value not in selected:
            self.class_experiment_dropdown.value = selected[0] if selected else None

    def _selected_names(self) -> list:
        return [name for name, cb in self.experiment_checks.items() if cb.value]

    def _on_selection_change(self, e):
        self._rebuild_experiment_checks()
        self._refresh_in_background()

    # ---- progress -------------------------------------------------------
    def _set_progress(self, text: str) -> None:
        self.progress_banner.visible = True
        self.progress_text.value = text
        if self.progress_banner.page:
            self.progress_banner.update()

    def _clear_progress(self) -> None:
        if self.progress_banner.visible:
            self.progress_banner.visible = False
            if self.progress_banner.page:
                self.progress_banner.update()

    # ---- computation ----------------------------------------------------
    def _bundle(self, name: str) -> ExperimentCalibration:
        if name in self._cache:
            return self._cache[name]

        state = self.ctx.state
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

    def _float_field(self, field_control: ft.TextField, default: float,
                     low: float = 0.0, high: float = 1.0) -> float:
        try:
            value = float(str(field_control.value).strip())
        except (TypeError, ValueError):
            value = default
        value = max(low, min(high, value))
        field_control.value = f"{value:g}"
        return value

    # ---- render ---------------------------------------------------------
    def _render(self) -> None:
        state = self.ctx.state
        self.warnings_area.controls = []
        self.summary_area.controls = []

        if state.dataset is None or not state.experiments:
            self._show_placeholder(
                "Load a dataset and at least one experiment (results CSV) in the 'Data' tab first.")
            return
        selected = self._selected_names()
        if not selected:
            self._show_placeholder("Select at least one experiment above.")
            return

        try:
            bundles = {}
            for i, name in enumerate(selected):
                if name not in self._cache:
                    self._set_progress(f"Sorting confidences -- {name} ({i + 1}/{len(selected)})")
                bundles[name] = self._bundle(name)

            for name, bundle in bundles.items():
                banner = score_diagnostics_banner(bundle.diagnostics, name)
                if banner:
                    self.warnings_area.controls.append(banner)

            self._set_progress("Building the summary...")
            self._render_summary(bundles)
            self._set_progress("Drawing reliability diagrams...")
            self._render_reliability(bundles)
            self._set_progress("Drawing confidence distributions...")
            self._render_histograms(bundles)
            self._set_progress("Drawing the risk-coverage curve...")
            self._render_curve(bundles)
            self._render_threshold_tables(bundles)
            self._render_temperature(bundles)
        except Exception as ex:
            _log(f"error computing calibration: {ex}")
            traceback.print_exc()
            self._show_placeholder(f"Error computing calibration: {ex}", error=True)
        self._clear_progress()

    def _show_placeholder(self, message: str, error: bool = False) -> None:
        banner = info_banner(message, icon=ft.Icons.ERROR_OUTLINE if error else None)
        self.summary_area.controls = [banner]
        self.reliability_area.controls = []
        self.histogram_area.controls = []
        self.curve_area.content = None
        self.threshold_table_area.controls = []
        self.target_table_area.controls = []
        self.class_area.controls = []
        self.temperature_area.controls = []
        self._clear_progress()

    def _render_summary(self, bundles: dict) -> None:
        all_names = list(self.ctx.state.experiments)
        headers = ["Experiment", "N", "Accuracy", "Mean confidence", "Confidence gap",
                   "ECE", "MCE", "AURC", "Best possible AURC", "Excess AURC"]
        columns = [ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12)) for h in headers]

        rows = []
        for name, bundle in bundles.items():
            gap = bundle.mean_confidence - bundle.accuracy
            excess = bundle.curve.aurc - bundle.curve.optimal_aurc
            name_cell = ft.Row([
                ft.Container(width=10, height=10, border_radius=5,
                             bgcolor=color_for_experiment(name, all_names)),
                ft.Text(name, size=12),
            ], spacing=8)
            rows.append(ft.DataRow(cells=[
                ft.DataCell(name_cell),
                ft.DataCell(ft.Text(f"{bundle.n:,}", size=12)),
                ft.DataCell(ft.Text(_fmt(bundle.accuracy), size=12)),
                ft.DataCell(ft.Text(_fmt(bundle.mean_confidence), size=12)),
                ft.DataCell(ft.Text(
                    f"{gap:+.3f}" if gap == gap else "—", size=12,
                    color=ft.Colors.ORANGE if abs(gap) > 0.05 else None)),
                ft.DataCell(ft.Text(_fmt(bundle.reliability.ece), size=12)),
                ft.DataCell(ft.Text(_fmt(bundle.reliability.mce), size=12)),
                ft.DataCell(ft.Text(_fmt(bundle.curve.aurc), size=12)),
                ft.DataCell(ft.Text(_fmt(bundle.curve.optimal_aurc), size=12)),
                ft.DataCell(ft.Text(_fmt(excess), size=12, weight=ft.FontWeight.BOLD)),
            ]))

        table = ft.DataTable(columns=columns, rows=rows,
                             heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8)
        self.summary_area.controls = [card(ft.Column([
            section_title("Summary", icon=ft.Icons.SUMMARIZE_OUTLINED, ctx=self.ctx,
                          manual="calibration-summary"),
            ft.Row([table], scroll=ft.ScrollMode.ALWAYS),
        ], spacing=10))]

    def _render_reliability(self, bundles: dict) -> None:
        all_names = list(self.ctx.state.experiments)
        self.reliability_area.controls = [
            self._reliability_card(name, bundle.reliability, color_for_experiment(name, all_names))
            for name, bundle in bundles.items()
        ]

    def _reliability_card(self, name: str, data, color) -> ft.Control:
        """Reliability diagram (accuracy = f(confidence)) for one
        experiment: a bar per confidence bin shows the actual accuracy of
        predictions that fell in that bin, a red line traces the mean
        predicted confidence per bin for comparison, and the dashed diagonal
        is perfect calibration. The bottom panel is the same bins' share of
        all predictions, on a log scale since most of the mass is typically
        in the highest-confidence bins -- this is the same binning ECE/MCE
        are computed from, so the two always agree."""
        if data.disabled_reason:
            return card(ft.Column([
                ft.Text(f"{name} — calibration unavailable", weight=ft.FontWeight.BOLD, size=13),
                ft.Container(width=340, content=info_banner(
                    f"No reliability diagram: {data.disabled_reason}. ECE and MCE are blank for the "
                    f"same reason, and so is temperature scaling below -- the threshold sections are "
                    f"unaffected, since they only ever rank by confidence.",
                    icon=ft.Icons.RULE)),
            ], spacing=8))

        width = 1.0 / len(data.bin_centers)
        fig, (ax1, ax2) = subplots(2, 1, figsize=(4.6, 5.4),
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect calibration")
        ax1.bar(data.bin_centers, np.nan_to_num(data.bin_accuracy), width=width * 0.9,
                color=color, edgecolor="black", linewidth=0.4, label="accuracy")
        ax1.plot(data.bin_centers, data.bin_confidence, color="#d62728", marker="o",
                 markersize=3, linewidth=1.2, label="mean confidence")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("accuracy")
        ax1.set_title(f"{name} — ECE={data.ece:.3f} · MCE={data.mce:.3f}", fontsize=11)
        ax1.legend(fontsize=7, loc="upper left")

        ax2.bar(data.bin_centers, data.bin_fraction, width=width * 0.9, color=color)
        ax2.set_yscale("log")
        ax2.set_xlabel("predicted confidence")
        ax2.set_ylabel("fraction\n(log)", fontsize=9)
        ax2.set_xlim(0, 1)
        # Reading a reliability diagram is reading a vertical distance from
        # the diagonal, which is exactly what the grid makes possible. The
        # fraction axis below it is logarithmic, so style_axes leaves its
        # decade subdivisions alone rather than imposing even ones.
        style_axes(ax1)
        style_axes(ax2)
        fig.tight_layout()
        return card(ft.Column([
            fig_to_zoomable_image(fig, self.ctx, f"Calibration — {name}", width=380, height=440),
        ], spacing=6))

    def _render_histograms(self, bundles: dict) -> None:
        all_names = list(self.ctx.state.experiments)
        self.histogram_area.controls = [
            self._histogram_card(name, bundle, color_for_experiment(name, all_names))
            for name, bundle in bundles.items()
        ]

    def _histogram_card(self, name: str, bundle: ExperimentCalibration, color) -> ft.Control:
        """Where the correct and the wrong predictions sit on the confidence
        axis. This is the picture that says whether any threshold can work:
        two separated humps mean an operating point exists, one merged hump
        means no threshold will buy much however good the ECE looks."""
        hist = bundle.histogram
        centers = (hist.edges[:-1] + hist.edges[1:]) / 2
        width = (hist.edges[1] - hist.edges[0]) * 0.9

        fig, ax = subplots(figsize=(4.6, 3.4))
        ax.bar(centers, hist.correct, width=width, color=color, label="correct")
        ax.bar(centers, hist.wrong, width=width, bottom=hist.correct,
               color="#d62728", label="wrong")
        ax.set_yscale("log")
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("crops (log)")
        ax.set_title(f"{name} — confidence of right vs. wrong", fontsize=10)
        style_axes(ax)
        ax.legend(fontsize=7)
        fig.tight_layout()
        return card(ft.Column([
            fig_to_zoomable_image(fig, self.ctx, f"Confidence distribution — {name}",
                                  width=400, height=296),
        ], spacing=6))

    def _render_curve(self, bundles: dict) -> None:
        all_names = list(self.ctx.state.experiments)
        fig, ax = subplots(figsize=(
            7.4, 4.4 + legend_height_inches(len(bundles), legend_columns(len(bundles)))))
        for name, bundle in bundles.items():
            curve = bundle.curve
            if not len(curve.coverage):
                continue
            ax.plot(curve.coverage, curve.risk, color=color_for_experiment(name, all_names),
                    label=f"{name} (AURC={curve.aurc:.4f})")
        ax.set_xlabel("Coverage -- share of crops the model is allowed to decide")
        ax.set_ylabel("Selective risk -- error rate among those")
        ax.set_title("Risk-coverage: what abstaining buys")
        ax.set_xlim(0, 1)
        style_axes(ax)
        # One entry per experiment, under the plot: the interesting part of
        # a risk-coverage curve is its right-hand end, which is where an
        # in-axes legend would sit.
        reserve_legend_space(fig, len(bundles), legend_columns(len(bundles)))
        legend_below(ax, ncol=legend_columns(len(bundles)))
        self.curve_area.content = card(ft.Column([
            fig_to_zoomable_image(fig, self.ctx, "Risk-coverage", **_chart_box(fig)),
        ], spacing=8))

    # ---- threshold explorer --------------------------------------------
    def _render_threshold_tables(self, bundles: dict) -> None:
        self._render_threshold_table(bundles)
        self._render_target_table(bundles)

    def _render_threshold_table(self, bundles: dict) -> None:
        threshold = float(self.threshold_slider.value or 0.0)
        self.threshold_label.value = f"Accept at or above {threshold:.3f}"

        headers = ["Experiment", "Accepted", "Coverage", "Accuracy on accepted",
                   "Errors let through", "Sent to a human", "Accuracy of those", ""]
        columns = [ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12)) for h in headers]
        rows = []
        for name, bundle in bundles.items():
            point = cal.threshold_point(bundle.index, threshold)
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(name, size=12)),
                ft.DataCell(ft.Text(f"{point.n_accepted:,}", size=12)),
                ft.DataCell(ft.Text(_pct(point.coverage), size=12)),
                ft.DataCell(ft.Text(_fmt(point.accuracy), size=12, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{point.n_errors_accepted:,}", size=12)),
                ft.DataCell(ft.Text(f"{point.n_abstained:,}", size=12)),
                ft.DataCell(ft.Text(_fmt(point.accuracy_abstained), size=12)),
                ft.DataCell(ft.Row([
                    ft.TextButton(
                        "Errors let through", icon=ft.Icons.REPORT_GMAILERRORRED_OUTLINED,
                        disabled=point.n_errors_accepted == 0,
                        on_click=lambda e, n=name, t=threshold: self._open_crops(n, t, "accepted_errors"),
                    ),
                    ft.TextButton(
                        "Abstained", icon=ft.Icons.PAN_TOOL_OUTLINED,
                        disabled=point.n_abstained == 0,
                        on_click=lambda e, n=name, t=threshold: self._open_crops(n, t, "abstained"),
                    ),
                ], spacing=2, tight=True)),
            ]))
        table = ft.DataTable(columns=columns, rows=rows,
                             heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8)
        self.threshold_table_area.controls = [ft.Row([table], scroll=ft.ScrollMode.ALWAYS)]

    def _render_target_table(self, bundles: dict) -> None:
        target = self._float_field(self.target_field, DEFAULT_TARGET_ACCURACY)
        headers = ["Experiment", "Required threshold", "Coverage reached", "Accepted",
                   "Accuracy on accepted", "Errors let through", "Sent to a human"]
        columns = [ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12)) for h in headers]
        rows = []
        for name, bundle in bundles.items():
            point = cal.threshold_for_target(bundle.index, target)
            if point.unreachable:
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(name, size=12)),
                    ft.DataCell(ft.Text("unreachable", size=12, color=ft.Colors.ORANGE)),
                    *[ft.DataCell(ft.Text("—", size=12)) for _ in range(5)],
                ]))
                continue
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(name, size=12)),
                ft.DataCell(ft.Text(_fmt(point.threshold, 4), size=12, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(_pct(point.coverage), size=12, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{point.n_accepted:,}", size=12)),
                ft.DataCell(ft.Text(_fmt(point.accuracy), size=12)),
                ft.DataCell(ft.Text(f"{point.n_errors_accepted:,}", size=12)),
                ft.DataCell(ft.Text(f"{point.n_abstained:,}", size=12)),
            ]))
        table = ft.DataTable(columns=columns, rows=rows,
                             heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8)
        self.target_table_area.controls = [
            # Its own (i): this table answers the reverse question from the
            # one above it -- state the accuracy, get the threshold -- and
            # the search behind it has rules of its own.
            ft.Row([ft.Text("Working back from a target accuracy", size=12, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    info_button(self.ctx, "target-accuracy")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([table], scroll=ft.ScrollMode.ALWAYS),
            ft.Text(
                "The most permissive threshold that still reaches the target -- i.e. the most the "
                "model can be trusted with. 'Unreachable' means no threshold gets there, including "
                "one that accepts a single crop.",
                size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        ]

    def _on_threshold_slide(self, e):
        """Fires continuously while dragging: only the label moves, so the
        drag stays smooth."""
        self.threshold_label.value = f"Accept at or above {float(self.threshold_slider.value):.3f}"
        if self.threshold_label.page:
            self.threshold_label.update()

    def _on_threshold_settled(self, e):
        if not self._cache:
            return
        bundles = {n: self._cache[n] for n in self._selected_names() if n in self._cache}
        self._render_threshold_table(bundles)
        if self.threshold_table_area.page:
            self.threshold_table_area.update()

    def _on_target_change(self, e):
        if not self._cache:
            return
        bundles = {n: self._cache[n] for n in self._selected_names() if n in self._cache}
        self._render_target_table(bundles)
        if self.target_table_area.page:
            self.target_table_area.update()

    # ---- per-class thresholds ------------------------------------------
    def _on_class_thresholds_click(self, e):
        name = self.class_experiment_dropdown.value
        if not name:
            self.class_area.controls = [info_banner("Pick an experiment first.")]
            if self.class_area.page:
                self.class_area.update()
            return
        self.class_progress.visible = True
        if self.class_progress.page:
            self.class_progress.update()
        self.class_job.run(self._compute_class_thresholds, on_error=self._on_class_error)

    def _on_class_error(self, ex: Exception) -> None:
        self.class_progress.visible = False
        self.class_area.controls = [
            info_banner(f"Error computing per-class thresholds: {ex}", icon=ft.Icons.ERROR_OUTLINE)]
        for control in (self.class_progress, self.class_focus.build(), self.class_area):
            if control.page:
                control.update()

    def _compute_class_thresholds(self) -> None:
        name = self.class_experiment_dropdown.value
        target = self._float_field(self.class_target_field, DEFAULT_TARGET_PRECISION)
        key = (name, target)
        if key not in self._class_rows:
            state = self.ctx.state
            experiment = state.experiments[name]
            merged = state.merged(name)
            labels = experiment.class_labels[: len(experiment.score_cols)]
            self._class_rows[key] = cal.per_class_thresholds(
                merged["predicted_label"].to_numpy(),
                merged["predicted_score"].to_numpy(dtype=float),
                merged["correct"].to_numpy().astype(bool),
                labels, target,
            )
        self._render_class_table(self._class_rows[key], name, target)
        self.class_progress.visible = False
        for control in (self.class_progress, self.class_focus.build(), self.class_area):
            if control.page:
                control.update()

    def _on_focus_change(self):
        if self._class_table_input is None:
            return
        self._render_class_table(*self._class_table_input)
        for control in (self.class_focus.build(), self.class_area):
            if control.page:
                control.update()

    def _render_class_table(self, rows_data: list, name: str, target: float) -> None:
        self._class_table_input = (rows_data, name, target)
        n_all_classes = len(rows_data)
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
        headers = ["Predicted class", "Predictions", "Threshold", "Kept", "Coverage in class", "Precision"]
        columns = [ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12)) for h in headers]
        rows = []
        for row in rows_data:
            unreachable = row.unreachable
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(row.label, size=12, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{row.n_predicted:,}", size=12)),
                ft.DataCell(ft.Text(
                    "unreachable" if unreachable else _fmt(row.threshold, 4), size=12,
                    color=ft.Colors.ORANGE if unreachable else None)),
                ft.DataCell(ft.Text("—" if unreachable else f"{row.n_kept:,}", size=12)),
                ft.DataCell(ft.Text("—" if unreachable else _pct(row.coverage), size=12)),
                ft.DataCell(ft.Text("—" if unreachable else _fmt(row.precision), size=12)),
            ]))
        table = ft.DataTable(columns=columns, rows=rows,
                             heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8)
        self.class_area.controls = [
            ft.Row([ft.Text(f"{name} — target precision {target:g}"
                            + (f" -- {len(rows_data)} of {n_all_classes:,} classes, picked above"
                               if self.class_focus.many else ""),
                            size=12, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    info_button(self.ctx, "per-class-thresholds")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([table], scroll=ft.ScrollMode.ALWAYS),
        ]

    # ---- temperature scaling -------------------------------------------
    def _render_temperature(self, bundles: dict) -> None:
        fitted = [(name, bundle) for name, bundle in bundles.items() if bundle.temperature is not None]
        if not fitted:
            self.temperature_area.controls = [info_banner(
                "Not fitted yet. Fitting reads every selected experiment's score matrix once; "
                "it changes no prediction, only the confidence attached to it.")]
            return

        headers = ["Experiment", "Temperature", "Log loss before", "after",
                   "ECE before", "after", "Rows fitted on"]
        columns = [ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12)) for h in headers]
        rows = []
        notes = []
        for name, bundle in fitted:
            fit = bundle.temperature
            if fit.disabled_reason:
                notes.append(f"{name}: not fitted -- {fit.disabled_reason}.")
                continue
            improved = fit.ece_after < fit.ece_before
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(name, size=12)),
                ft.DataCell(ft.Text(_fmt(fit.temperature, 3), size=12, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(_fmt(fit.nll_before), size=12)),
                ft.DataCell(ft.Text(_fmt(fit.nll_after), size=12)),
                ft.DataCell(ft.Text(_fmt(fit.ece_before), size=12)),
                ft.DataCell(ft.Text(
                    _fmt(fit.ece_after), size=12, weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREEN if improved else ft.Colors.ORANGE)),
                ft.DataCell(ft.Text(f"{fit.n_fit_rows:,} of {fit.n_rows:,}", size=12)),
            ]))
            if not improved:
                notes.append(
                    f"{name}: the fitted temperature improves the log loss but not the ECE. That "
                    f"happens when the confidence carries little information about correctness -- "
                    f"check the risk-coverage curve above before trusting either number.")

        table = ft.DataTable(columns=columns, rows=rows,
                             heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8)
        self.temperature_area.controls = [
            ft.Row([table], scroll=ft.ScrollMode.ALWAYS),
            *[ft.Text(n, size=11, italic=True, color=ft.Colors.ORANGE) for n in notes],
        ]

    def _on_fit_temperature(self, e):
        self.temperature_progress.visible = True
        if self.temperature_progress.page:
            self.temperature_progress.update()
        self.temperature_job.run(self._fit_temperatures, on_error=self._on_temperature_error)

    def _on_temperature_error(self, ex: Exception) -> None:
        self.temperature_progress.visible = False
        self.temperature_area.controls = [
            info_banner(f"Error fitting the temperature: {ex}", icon=ft.Icons.ERROR_OUTLINE)]
        for control in (self.temperature_progress, self.temperature_area):
            if control.page:
                control.update()

    def _fit_temperatures(self) -> None:
        state = self.ctx.state
        bundles = {}
        for name in self._selected_names():
            bundle = self._bundle(name)
            bundles[name] = bundle
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

        self._render_temperature(bundles)
        self.temperature_progress.visible = False
        for control in (self.temperature_progress, self.temperature_area):
            if control.page:
                control.update()

    # ---- drilling into an operating point -------------------------------
    def _open_crops(self, experiment_name: str, threshold: float, which: str) -> None:
        """The crops behind a number in the threshold table.

        "Errors let through" is the set a deployment would get wrong
        silently, and "abstained" is the queue a human would be handed --
        the two things worth looking at before committing to a threshold.
        Both carry the usual review controls, so a crop that turns out to be
        mislabelled can be fixed on the spot.
        """
        state = self.ctx.state
        merged = state.merged(experiment_name)
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

        self.carousel_title.value = f"{experiment_name}  ·  {title}  ·  {len(subset):,} crop(s)"
        carousel = Carousel(
            state.dataset, len(subset),
            lambda start, end: subset.iloc[start:end].to_dict("records"),
            cols.crop_col, cols.uuid_col, ctx=self.ctx,
            review_source="calibration",
            suggestions={str(u): str(p) for u, p in
                         zip(subset[cols.uuid_col], subset["predicted_label"])},
            label_col=cols.label_col,
        )
        self.carousel_content.content = carousel.build()
        self.ctx.page.open(self.carousel_dialog)

    def _close_carousel(self, e) -> None:
        self.ctx.page.close(self.carousel_dialog)

    # ---- lifecycle ------------------------------------------------------
    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        return {
            "selected": [name for name, cb in self.experiment_checks.items() if cb.value],
            "threshold": float(self.threshold_slider.value),
            "target_accuracy": str(self.target_field.value or ""),
            "class_experiment": self.class_experiment_dropdown.value or "",
            "target_precision": str(self.class_target_field.value or ""),
            "class_focus": self.class_focus.view_state(),
        }

    def restore_view_state(self, state: dict) -> list:
        warnings = []
        loaded = set(self.ctx.state.experiments)

        self._rebuild_experiment_checks()
        selected = state.get("selected")
        if isinstance(selected, list):
            wanted = {n for n in selected if isinstance(n, str)}
            gone = sorted(wanted - loaded)
            if gone:
                warnings.append(f"Calibration: experiment(s) {', '.join(gone)} are not loaded.")
            for name, check in self.experiment_checks.items():
                check.value = name in wanted
            # The per-class dropdown is *derived* from which experiments are
            # ticked (see _rebuild_experiment_checks), so unticking them by
            # hand leaves it pointing at an experiment that is no longer
            # shown. Rebuilding re-derives it from the values just set --
            # without this, restoring a snapshot that had nothing ticked
            # came back with the first experiment selected there, and the
            # app bar then reported the restored state as already modified.
            self._rebuild_experiment_checks()

        try:
            threshold = float(state["threshold"])
        except (KeyError, TypeError, ValueError):
            threshold = None
        if threshold is not None:
            self.threshold_slider.value = max(self.threshold_slider.min, min(self.threshold_slider.max, threshold))
            self.threshold_label.value = f"Accept at or above {self.threshold_slider.value:.3f}"

        if state.get("target_accuracy"):
            self.target_field.value = str(state["target_accuracy"])
        if state.get("target_precision"):
            self.class_target_field.value = str(state["target_precision"])

        self.class_focus.restore_view_state(state.get("class_focus"))

        class_experiment = state.get("class_experiment") or ""
        if class_experiment:
            if class_experiment in loaded:
                self.class_experiment_dropdown.value = class_experiment
            else:
                warnings.append(f"Calibration: experiment '{class_experiment}' is not loaded.")

        if self.root.page:
            self.root.update()
        return warnings

    def refresh(self) -> None:
        """Data changed. Same contract as the Metrics tab: flag the view
        stale and let the user decide when to pay for the recompute, rather
        than sorting a million confidences because a review decision was
        recorded somewhere."""
        self._dirty = True
        if self.root.page:
            self.stale_banner.show()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if old_name in self._cache:
            bundle = self._cache.pop(old_name)
            bundle.name = new_name
            self._cache[new_name] = bundle
        for key in [k for k in self._class_rows if k[0] == old_name]:
            self._class_rows[(new_name, key[1])] = self._class_rows.pop(key)
        if old_name in self.experiment_checks:
            self.experiment_checks[new_name] = self.experiment_checks.pop(old_name)
        self._rebuild_experiment_checks()
        if not self._dirty:
            self._render()
        if self.root.page:
            self.root.update()

    def recompute_after_restore(self) -> None:
        if self.ctx.state.dataset is None or not self._selected_names():
            return
        self._on_update_click(None)

    def _on_update_click(self, e):
        self._cache.clear()
        self._class_rows.clear()
        self._refresh_in_background()

    def _refresh_in_background(self) -> None:
        self._set_progress("Computing calibration...")
        self.rebuild_job.run(self._refresh_now, on_error=self._on_rebuild_error)

    def _on_rebuild_error(self, ex: Exception) -> None:
        self._clear_progress()
        self._show_placeholder(f"Error computing calibration: {ex}", error=True)
        if self.root.page:
            self.root.update()

    def _refresh_now(self) -> None:
        self._rebuild_experiment_checks()
        self._render()
        self._dirty = False
        self.stale_banner.hide()
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        self.root.content = ft.Column(
            controls=[
                section_title("Calibration & abstention", icon=ft.Icons.RULE_OUTLINED, ctx=self.ctx, manual="calibration-error"),
                ft.Text(
                    "Computed from the same merged rows the Metrics tab reads, so the review "
                    "overlay, the row filter and the split all apply here too.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                self.stale_banner.control,
                card(ft.Column([
                    ft.Row([
                        ft.Text("Experiments", size=12, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.TextButton("Update now", icon=ft.Icons.REFRESH, on_click=self._on_update_click),
                    ]),
                    self.selection_row,
                ], spacing=8)),
                self.warnings_area,
                self.progress_banner,
                self.summary_area,

                section_title("Is the confidence believable?", icon=ft.Icons.STRAIGHTEN, ctx=self.ctx, manual="reliability-diagram"),
                self.reliability_area,
                self.histogram_area,

                section_title("What does abstaining buy?", icon=ft.Icons.CALL_SPLIT, ctx=self.ctx, manual="risk-coverage"),
                self.curve_area,

                section_title("Pick an operating point", icon=ft.Icons.TUNE, ctx=self.ctx, manual="confidence-threshold"),
                card(ft.Column([
                    ft.Text("From a threshold", size=13, weight=ft.FontWeight.BOLD),
                    ft.Row([self.threshold_label], spacing=10),
                    self.threshold_slider,
                    self.threshold_table_area,
                ], spacing=10)),
                card(ft.Column([
                    ft.Text("From a target", size=13, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        self.target_field,
                        ft.FilledButton("Find the threshold", icon=ft.Icons.SEARCH,
                                        on_click=self._on_target_change,
                                        style=ft.ButtonStyle(bgcolor=ACCENT)),
                    ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    self.target_table_area,
                ], spacing=10)),
                card(ft.Column([
                    ft.Text("One threshold per class", size=13, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        self.class_experiment_dropdown, self.class_target_field,
                        ft.FilledButton("Compute", icon=ft.Icons.CALCULATE_OUTLINED,
                                        on_click=self._on_class_thresholds_click,
                                        style=ft.ButtonStyle(bgcolor=ACCENT)),
                    ], spacing=12, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    self.class_progress,
                    self.class_focus.build(),
                    self.class_area,
                ], spacing=10)),

                section_title("Fix the confidence: temperature scaling", icon=ft.Icons.THERMOSTAT, ctx=self.ctx, manual="temperature-scaling"),
                ft.Text(
                    "Changes no prediction: only the confidence attached to it moves.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                card(ft.Column([
                    ft.Row([
                        ft.FilledButton("Fit a temperature", icon=ft.Icons.THERMOSTAT,
                                        on_click=self._on_fit_temperature,
                                        style=ft.ButtonStyle(bgcolor=ACCENT)),
                    ], spacing=12),
                    self.temperature_progress,
                    self.temperature_area,
                ], spacing=10)),
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
        )
        self._refresh_now()
        return self.root
