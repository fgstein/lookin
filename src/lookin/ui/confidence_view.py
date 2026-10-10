"""Tab: per-prediction confidence for one experiment at a time -- margin
between the top two scores and entropy of the softmax-normalized score
distribution (see core/confidence.py), independent of whether a prediction
was actually correct.

Deliberately not folded into the Metrics tab: Metrics answers "how good is
this model" against the ground truth; this tab answers "how sure was the
model," a question that makes sense even without labels at all. The two
combined answer the sharper one -- was the model sure about the
predictions it got wrong (see the "Confidently wrong" section below).
"""
from __future__ import annotations

import traceback

import numpy as np
import flet as ft

from lookin.core import class_focus as cf
from lookin.core.chart_style import subplots
from lookin.core.confidence import MARGIN_COL, NORMALIZED_ENTROPY_COL, compute_confidence, confident_and_wrong
from lookin.ui.class_matrix import ClassFocus, make_searchable
from lookin.ui.components import (
    ACCENT, BackgroundRebuild, GRID_NONE, StaleBanner, card, fig_to_zoomable_image, info_banner,
    score_diagnostics_banner, section_title, style_axes, uuid_overlap_banner,
)

# Which classes the per-class chart shows past FOCUS_MAX (see ClassFocus).
CONFIDENCE_FOCUS_CRITERIA = (
    (cf.FOCUS_LEAST_CONFIDENT, "Least confident"),
    (cf.FOCUS_HIGHEST_ENTROPY, "Highest entropy"),
    (cf.FOCUS_LARGEST_SUPPORT, "Largest support"),
    (cf.FOCUS_MANUAL, "Hand-picked"),
)

# Same red/blue pairing calibration_view.py and prototypes_view.py use for
# wrong/bad vs. right/good (e.g. "#d62728" for a wrong prediction's bar in
# the calibration histogram) -- one meaning for these two colors everywhere
# in the app, rather than a third pair invented for this tab.
CORRECT_COLOR = "#4c72b0"
INCORRECT_COLOR = "#d62728"
PERCLASS_BAR_COLOR = "#e8590c"
CHART_WIDTH = 860
CHART_HEIGHT = 300
SCATTER_MAX_POINTS_DEFAULT = 3000
SCATTER_SEED = 0
CHART_MAX_WIDTH = 1000
CHART_PX_PER_INCH = 90


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _chart_box(fig) -> dict:
    """width/height for fig_to_zoomable_image that keep `fig`'s own aspect
    ratio -- same helper as metrics_view._chart_box."""
    fig_w, fig_h = fig.get_size_inches()
    width = min(CHART_MAX_WIDTH, max(600.0, fig_w * CHART_PX_PER_INCH))
    return {"width": int(width), "height": int(width * fig_h / fig_w)}


class ConfidenceView:
    def __init__(self, ctx):
        self.ctx = ctx
        self._confidence_cache: dict = {}  # experiment name -> ConfidenceData
        self._dirty = False

        self.experiment_dropdown = ft.Dropdown(
            label="Experiment", width=260, on_change=self._on_experiment_change, border_radius=8)

        self.scatter_sample_field = ft.TextField(
            label="Scatter sample size", value=str(SCATTER_MAX_POINTS_DEFAULT), width=190, border_radius=8,
            on_submit=self._on_scatter_sample_change,
        )
        self.scatter_apply_btn = ft.IconButton(
            ft.Icons.REFRESH, tooltip="Redraw the scatter", on_click=self._on_scatter_sample_change)

        self.class_dropdown = ft.Dropdown(
            label="Class", width=260, on_change=self._on_class_change, border_radius=8)
        # Past FOCUS_MAX classes the per-class chart shows a focus set.
        self.class_focus = ClassFocus(CONFIDENCE_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the per-class chart",
                                      support_criteria=(cf.FOCUS_LEAST_CONFIDENT, cf.FOCUS_HIGHEST_ENTROPY))
        self._perclass_input = None

        self.confident_wrong_metric_dropdown = ft.Dropdown(
            label="Confidence measure", value="margin", width=230, border_radius=8,
            options=[
                ft.dropdown.Option("margin", "Margin (top1 - top2)"),
                ft.dropdown.Option("entropy", "Entropy (normalized)"),
            ],
            on_change=self._on_confident_wrong_change,
        )
        self.confident_wrong_percentile_field = ft.TextField(
            label="Top confidence percentile", value="25", width=200, border_radius=8,
            on_submit=self._on_confident_wrong_change,
        )
        self.confident_wrong_apply_btn = ft.IconButton(
            ft.Icons.REFRESH, tooltip="Recompute", on_click=self._on_confident_wrong_change)

        self.warnings_area = ft.Column(spacing=6)
        self.summary_area = ft.Container()
        self.hist_area = ft.Column(spacing=14)
        self.class_hist_area = ft.Column(spacing=14)
        self.perclass_area = ft.Container()
        self.scatter_area = ft.Container()
        self.confident_wrong_area = ft.Column(spacing=10)

        self.progress_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_banner = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2), self.progress_text],
            spacing=10, visible=False,
        )
        self.stale_banner = StaleBanner(self._on_update_click)
        self.rebuild_job = BackgroundRebuild(ctx, "Confidence")
        self.root = ft.Container(padding=20)

    # ---- progress ----------------------------------------------------
    def _set_progress(self, text: str):
        self.progress_banner.visible = True
        self.progress_text.value = text
        if self.progress_banner.page:
            self.progress_banner.update()

    def _clear_progress(self):
        if self.progress_banner.visible:
            self.progress_banner.visible = False
            if self.progress_banner.page:
                self.progress_banner.update()

    # ---- small field helpers ------------------------------------------
    def _int_field(self, field, default: int) -> int:
        try:
            return int(float(field.value))
        except (TypeError, ValueError):
            return default

    def _float_field(self, field, default: float) -> float:
        try:
            return float(field.value)
        except (TypeError, ValueError):
            return default

    # ---- experiment picker ---------------------------------------------
    def _rebuild_experiment_dropdown(self):
        names = list(self.ctx.state.experiments)
        self.experiment_dropdown.options = [ft.dropdown.Option(n) for n in names]
        if self.experiment_dropdown.value not in names:
            self.experiment_dropdown.value = names[0] if names else None

    def _on_experiment_change(self, e):
        self._render()
        if self.root.page:
            self.root.update()

    def _on_scatter_sample_change(self, e):
        name = self.experiment_dropdown.value
        if not name or name not in self._confidence_cache:
            return
        self._render_scatter(self._confidence_cache[name])
        if self.scatter_area.page:
            self.scatter_area.update()

    # ---- class picker ---------------------------------------------------
    def _rebuild_class_dropdown(self, data, label_col: str):
        labels = sorted(data.df[label_col].dropna().astype(str).unique().tolist())
        self.class_dropdown.options = [ft.dropdown.Option(c) for c in labels]
        make_searchable(self.class_dropdown)
        if self.class_dropdown.value not in labels:
            self.class_dropdown.value = labels[0] if labels else None

    def _on_class_change(self, e):
        name = self.experiment_dropdown.value
        if not name or name not in self._confidence_cache or self.ctx.state.dataset is None:
            return
        label_col = self.ctx.state.dataset.columns.label_col
        self._render_class_histograms(self._confidence_cache[name], label_col)
        if self.class_hist_area.page:
            self.class_hist_area.update()

    def _on_confident_wrong_change(self, e):
        name = self.experiment_dropdown.value
        if not name or name not in self._confidence_cache or self.ctx.state.dataset is None:
            return
        label_col = self.ctx.state.dataset.columns.label_col
        self._render_confident_wrong(self._confidence_cache[name], label_col, name)
        if self.confident_wrong_area.page:
            self.confident_wrong_area.update()

    # ---- stale-data lifecycle (see StatsView / MetricsView) -------------
    def _on_update_click(self, e):
        self._confidence_cache.clear()
        self._refresh_in_background()

    def _refresh_in_background(self):
        self._set_progress("Computing confidence diagnostics...")
        self.rebuild_job.run(self._refresh_now, on_error=self._on_rebuild_error)

    def _on_rebuild_error(self, ex: Exception):
        self._clear_progress()
        self.summary_area.content = info_banner(f"Error computing confidence: {ex}", icon=ft.Icons.ERROR_OUTLINE)
        if self.root.page:
            self.root.update()

    def _refresh_now(self):
        self._rebuild_experiment_dropdown()
        self._render()
        self._dirty = False
        self.stale_banner.hide()
        if self.root.page:
            self.root.update()

    # ---- cached accessor --------------------------------------------------
    def _get_confidence(self, name: str):
        if name not in self._confidence_cache:
            state = self.ctx.state
            exp = state.experiments[name]
            merged = state.merged(name)
            self._set_progress(f"Computing margin / entropy for '{name}' ({len(merged):,} rows)...")
            _log(f"computing confidence diagnostics for '{name}' ({len(merged):,} merged rows)...")
            self._confidence_cache[name] = compute_confidence(merged, exp.score_cols, exp.class_labels)
        return self._confidence_cache[name]

    # ---- top-level render ------------------------------------------------
    def _render(self):
        state = self.ctx.state
        self.warnings_area.controls.clear()
        if state.dataset is None or not state.experiments:
            self.summary_area.content = info_banner(
                "Load a dataset and at least one experiment (results CSV) in the 'Data' tab first.")
            self.hist_area.controls = []
            self.class_hist_area.controls = []
            self.perclass_area.content = None
            self.scatter_area.content = None
            self.confident_wrong_area.controls = []
            self._clear_progress()
            return

        if not self.experiment_dropdown.value:
            self._rebuild_experiment_dropdown()
        name = self.experiment_dropdown.value
        if not name:
            self._clear_progress()
            return

        overlap = state.uuid_overlap(name)
        banner = uuid_overlap_banner(overlap, "dataset", name)
        if banner:
            self.warnings_area.controls.append(banner)

        try:
            data = self._get_confidence(name)
            diag_banner = score_diagnostics_banner(data.score_diagnostics, name)
            if diag_banner:
                self.warnings_area.controls.append(diag_banner)
            if data.softmax_applied:
                self.warnings_area.controls.append(info_banner(
                    "Score columns do not sum to 1, so entropy is computed on softmax(scores) rather than "
                    "the raw scores directly -- margin is unaffected, it only compares the raw top two scores.",
                    icon=ft.Icons.FUNCTIONS,
                ))

            label_col = state.dataset.columns.label_col
            self._set_progress("Building the summary...")
            self._render_summary(data)
            self._set_progress("Drawing confidence distributions...")
            self._render_histograms(data)
            self._set_progress("Drawing confidence distributions for one class...")
            self._rebuild_class_dropdown(data, label_col)
            self._render_class_histograms(data, label_col)
            self._set_progress("Drawing per-class breakdown...")
            self._render_perclass(data, label_col)
            self._set_progress("Drawing margin vs. entropy scatter...")
            self._render_scatter(data)
            self._set_progress("Finding confidently-wrong predictions...")
            self._render_confident_wrong(data, label_col, name)
        except Exception as ex:
            _log(f"error rendering confidence tab: {ex}")
            traceback.print_exc()
            self.summary_area.content = info_banner(f"Error: {ex}", icon=ft.Icons.ERROR_OUTLINE)
        self._clear_progress()

    # ---- summary ------------------------------------------------------
    def _render_summary(self, data):
        df = data.df
        n = len(df)
        accuracy = float(df["correct"].mean()) if n else float("nan")
        margin_valid = df[MARGIN_COL].dropna()
        entropy_valid = df[NORMALIZED_ENTROPY_COL].dropna()

        headers = ["N", "Accuracy", "Mean margin", "Median margin", "Mean entropy (norm)", "Median entropy (norm)"]
        values = [
            f"{n:,}",
            f"{accuracy:.3f}" if accuracy == accuracy else "—",
            f"{margin_valid.mean():.3f}" if len(margin_valid) else "—",
            f"{margin_valid.median():.3f}" if len(margin_valid) else "—",
            f"{entropy_valid.mean():.3f}" if len(entropy_valid) else "—",
            f"{entropy_valid.median():.3f}" if len(entropy_valid) else "—",
        ]
        table = ft.DataTable(
            columns=[ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD)) for h in headers],
            rows=[ft.DataRow(cells=[ft.DataCell(ft.Text(v)) for v in values])],
            heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8,
        )
        self.summary_area.content = card(ft.Column([
            section_title("Summary", icon=ft.Icons.SUMMARIZE_OUTLINED, ctx=self.ctx, manual="margin-entropy"),
            ft.Row([table], scroll=ft.ScrollMode.ALWAYS),
        ], spacing=10))

    # ---- shared: one correct/incorrect histogram, for any row subset ----
    def _hist_fig(self, df, col: str, title: str, xlabel: str):
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

    def _hist_charts(self, df, name_prefix: str = ""):
        charts = []
        margin_fig = self._hist_fig(df, MARGIN_COL, "Margin (top1 - top2)", "margin")
        if margin_fig is not None:
            charts.append(fig_to_zoomable_image(
                margin_fig, self.ctx, f"{name_prefix}Margin distribution",
                width=CHART_WIDTH - 36, height=CHART_HEIGHT - 40))
        entropy_fig = self._hist_fig(df, NORMALIZED_ENTROPY_COL, "Entropy (normalized)", "entropy / log(n classes)")
        if entropy_fig is not None:
            charts.append(fig_to_zoomable_image(
                entropy_fig, self.ctx, f"{name_prefix}Entropy distribution",
                width=CHART_WIDTH - 36, height=CHART_HEIGHT - 40))
        return charts

    # ---- feature 1: correct/incorrect distributions --------------------
    def _render_histograms(self, data):
        charts = self._hist_charts(data.df)
        if not charts:
            self.hist_area.controls = [info_banner("Not enough data for a confidence distribution (needs at least two classes).")]
            return
        self.hist_area.controls = [card(ft.Column([
            section_title(
                "Confidence distributions", icon=ft.Icons.STACKED_BAR_CHART,
                ctx=self.ctx, manual="margin-entropy",
            ),
            ft.Text("Split by whether the prediction was correct.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            *charts,
        ], spacing=8))]

    # ---- feature 1b: correct/incorrect distributions for one class ------
    def _render_class_histograms(self, data, label_col: str):
        cls = self.class_dropdown.value
        if not cls:
            self.class_hist_area.controls = [info_banner("No class to show yet.")]
            return
        df = data.df[data.df[label_col].astype(str) == cls]
        charts = self._hist_charts(df, name_prefix=f"{cls} -- ")
        if not charts:
            self.class_hist_area.controls = [card(ft.Column([
                section_title(
                    "Confidence distributions by class", icon=ft.Icons.STACKED_BAR_CHART,
                    ctx=self.ctx, manual="margin-entropy",
                ),
                self.class_dropdown,
                info_banner(f"Not enough data for class '{cls}'."),
            ], spacing=10))]
            return
        self.class_hist_area.controls = [card(ft.Column([
            section_title(
                "Confidence distributions by class", icon=ft.Icons.STACKED_BAR_CHART,
                ctx=self.ctx, manual="margin-entropy",
            ),
            ft.Text(
                "Same as above, restricted to the rows whose true label is the class chosen below.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self.class_dropdown,
            *charts,
        ], spacing=10))]

    # ---- feature 2: per-class breakdown ---------------------------------
    def _on_focus_change(self):
        if self._perclass_input is None:
            return
        self._render_perclass(*self._perclass_input)
        for control in (self.class_focus.build(), self.perclass_area):
            if control.page:
                control.update()

    def _render_perclass(self, data, label_col: str):
        self._perclass_input = (data, label_col)
        df = data.df.dropna(subset=[MARGIN_COL, NORMALIZED_ENTROPY_COL])
        if df.empty:
            self.class_focus.hide()
            self.perclass_area.content = info_banner(
                "Not enough data for a per-class breakdown (needs at least two classes).")
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

        self.perclass_area.content = card(ft.Column([
            section_title(
                "Per-class confidence", icon=ft.Icons.BAR_CHART,
                ctx=self.ctx, manual="margin-entropy",
            ),
            ft.Text(
                "Classes the model is systematically unsure about, even where its accuracy on them looks fine."
                + (f" {len(labels)} of {n_all_classes:,} classes, picked under 'Classes shown' above."
                   if self.class_focus.many else ""),
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            fig_to_zoomable_image(fig, self.ctx, "Per-class confidence", **_chart_box(fig)),
        ], spacing=10))

    # ---- feature 3: margin vs. entropy scatter --------------------------
    def _render_scatter(self, data):
        df = data.df.dropna(subset=[MARGIN_COL, NORMALIZED_ENTROPY_COL])
        if df.empty:
            self.scatter_area.content = info_banner(
                "Not enough data for the margin vs. entropy scatter (needs at least two classes).")
            return

        sample_size = self._int_field(self.scatter_sample_field, SCATTER_MAX_POINTS_DEFAULT)
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
        self.scatter_area.content = card(ft.Column([
            section_title(
                "Margin vs. entropy", icon=ft.Icons.SCATTER_PLOT_OUTLINED,
                ctx=self.ctx, manual="margin-entropy",
            ),
            ft.Text(
                f"One point per prediction, colored by correctness.{note}",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([self.scatter_sample_field, self.scatter_apply_btn], spacing=6),
            fig_to_zoomable_image(fig, self.ctx, "Margin vs entropy scatter", width=640, height=480),
        ], spacing=10))

    # ---- feature 6: confidently wrong -----------------------------------
    def _render_confident_wrong(self, data, label_col: str, exp_name: str):
        metric = self.confident_wrong_metric_dropdown.value or "margin"
        top_pct = self._float_field(self.confident_wrong_percentile_field, 25.0)
        metric_label = "margin" if metric == "margin" else "entropy"

        df = data.df
        n_total = len(df)
        n_wrong = int((~df["correct"]).sum()) if n_total else 0
        subset = confident_and_wrong(data, label_col, metric, top_pct) if n_total else df.iloc[0:0]
        n_confident_wrong = len(subset)

        header = ft.Row([
            self.confident_wrong_metric_dropdown, self.confident_wrong_percentile_field,
            self.confident_wrong_apply_btn,
        ], spacing=10)

        controls = [
            section_title(
                "Confidently wrong", icon=ft.Icons.GPP_BAD_OUTLINED,
                ctx=self.ctx, manual="confidently-wrong",
            ),
            ft.Text(
                "Predictions that are both incorrect and among the model's most confident, by the measure "
                "and percentile chosen below.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            header,
        ]

        if n_total == 0:
            controls.append(info_banner("No predictions to check."))
        elif n_confident_wrong == 0:
            controls.append(info_banner(
                f"No confidently-wrong predictions among the top {top_pct:.0f}% most confident by {metric_label}."))
        else:
            pct_of_wrong = 100.0 * n_confident_wrong / n_wrong if n_wrong else 0.0
            pct_of_total = 100.0 * n_confident_wrong / n_total
            controls.append(info_banner(
                f"{n_confident_wrong:,} confidently-wrong prediction(s) -- {pct_of_wrong:.1f}% of all "
                f"{n_wrong:,} incorrect predictions, {pct_of_total:.2f}% of all {n_total:,} predictions, "
                f"among the top {top_pct:.0f}% most confident by {metric_label}.",
                icon=ft.Icons.REPORT_PROBLEM_OUTLINED,
            ))
            grouped = (
                subset.groupby([label_col, "predicted_label"])
                .agg(n=(MARGIN_COL, "count"), mean_margin=(MARGIN_COL, "mean"), mean_entropy=(NORMALIZED_ENTROPY_COL, "mean"))
                .reset_index()
                .sort_values("n", ascending=False)
            )
            rows = []
            for _, r in grouped.iterrows():
                true_label, predicted_label = str(r[label_col]), str(r["predicted_label"])
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(true_label)),
                    ft.DataCell(ft.Text(predicted_label)),
                    ft.DataCell(ft.Text(f"{int(r['n']):,}")),
                    ft.DataCell(ft.Text(f"{r['mean_margin']:.3f}")),
                    ft.DataCell(ft.Text(f"{r['mean_entropy']:.3f}" if r["mean_entropy"] == r["mean_entropy"] else "—")),
                    ft.DataCell(ft.IconButton(
                        ft.Icons.OPEN_IN_NEW, icon_size=16, tooltip="Open this pair in the Gallery",
                        on_click=lambda e, t=true_label, p=predicted_label: self._on_open_gallery(exp_name, t, p),
                    )),
                ]))
            table = ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD))
                         for c in ["true label", "predicted", "count", "mean margin", "mean entropy", ""]],
                rows=rows, heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8,
            )
            controls.append(ft.Row([table], scroll=ft.ScrollMode.ALWAYS))

        self.confident_wrong_area.controls = [card(ft.Column(controls, spacing=10))]

    def _on_open_gallery(self, exp_name: str, true_label: str, predicted_label: str):
        if self.ctx.go_to_gallery:
            self.ctx.go_to_gallery(exp_name, true_label, predicted_label)

    # ---- lifecycle ---------------------------------------------------
    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow an experiment rename (see AppContext.notify_rename)
        without recomputing anything -- same reasoning as MetricsView's
        rename_experiment: the numbers are unchanged, only the label is."""
        if old_name in self._confidence_cache:
            self._confidence_cache[new_name] = self._confidence_cache.pop(old_name)

        selected = self.experiment_dropdown.value
        self._rebuild_experiment_dropdown()
        if selected == old_name:
            self.experiment_dropdown.value = new_name
        if not self._dirty:
            self._render()
        if self.root.page:
            self.root.update()

    def refresh(self):
        """Called via ctx.notify_change() whenever the underlying data
        changes. Computing margin/entropy is cheap compared to the Metrics
        tab's sklearn passes, but it still needs a fresh merge of the full
        dataset with the experiment's scores -- real work at scale -- so
        this follows the same stale-banner pattern as Statistics/Metrics
        rather than recomputing on every data change."""
        self._dirty = True
        if self.root.page:
            self.stale_banner.show()

    def build(self) -> ft.Control:
        self.root.content = ft.Column(
            controls=[
                section_title("Confidence", icon=ft.Icons.PERCENT_OUTLINED),
                ft.Text(
                    "How sure the model was about each prediction for the selected experiment -- margin "
                    "between its top two scores and the entropy of its softmax-normalized distribution -- "
                    "independent of whether the prediction was actually correct.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                self.stale_banner.control,
                self.experiment_dropdown,
                self.warnings_area,
                self.progress_banner,
                self.summary_area,
                self.hist_area,
                self.class_hist_area,
                self.class_focus.build(),
                self.perclass_area,
                self.scatter_area,
                self.confident_wrong_area,
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
        )
        self._refresh_now()
        return self.root
