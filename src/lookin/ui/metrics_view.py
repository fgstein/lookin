"""Tab 4: classification metrics between true labels and predicted classes,
computed per loaded experiment and overlaid on the same charts so multiple
experiments (model runs) can be compared directly. Confusion matrix cells
are clickable and open an inline image carousel for that exact (true label,
predicted label, experiment) combination, without leaving this tab -- the
carousel dialog also offers a button to jump to the Image gallery with the
same filters preset.

At scale (many experiments x hundreds of thousands of rows each), computing
confusion matrices / per-class metrics / ROC-PR curves is real work -- this
view caches results per experiment (and per experiment+curve-target) so
toggling which experiments are shown, or which per-class metric/curve class
is selected, is instant after the first computation. The cache is only
cleared when the underlying data actually changes (refresh(), driven by
ctx.notify_change), not on pure UI interactions (_on_change).

Many experiments is the case this tab is arranged around. The picker at the
top is a drag-to-reorder list rather than a wrapped row of checkboxes, and
the order set there is the order everything reads: summary rows, bar groups,
curve legend entries, confusion matrices, slice tables -- but not the
colors, which stay with each experiment (AppState.color_slots). The order lives
on AppState.experiments, not here, so it is the same order in every view.

Renaming an experiment (Data tab) arrives through ctx.notify_rename rather
than ctx.notify_change: the numbers are unchanged, so the caches below are
re-keyed to the new name and the tab re-renders from them -- no recompute,
and no reload of the results CSV.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import flet as ft

from lookin.core.report import Report
from lookin.core.report_builder import build_report, default_report_name
from lookin.core import library
from lookin.core.report_pptx import PptxReport
from lookin.core.metrics import compute_metrics, compute_curves, MACRO_AVERAGE
from lookin.core import class_focus as cf
from lookin.ui.components import fig_to_zoomable_image, saved_path_message, section_title, info_banner, color_for_experiment, blues_color, card, grid_card_width, matrix_cell_size, matrix_text_size, matrix_label_chars, uuid_overlap_banner, score_diagnostics_banner, status_row, style_axes, legend_below, legend_columns, legend_height_inches, reserve_legend_space, GRID_MAJOR, GRID_NONE, ACCENT, BackgroundRebuild, StaleBanner
from lookin.ui.carousel import Carousel
from lookin.ui.class_matrix import ClassFocus, PAIRS_SHOWN, make_searchable, confused_pairs_table, overview_image, short_column_labels
from lookin.ui.server_browser import ServerFileBrowser
from lookin.ui.slice_panel import SlicePanel
from lookin.core.chart_style import subplots


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)

# Every per-class column compute_metrics() produces, so the chart below can
# show any of them -- "support" included, which is a count rather than a
# rate and so is the one that needs its own y-axis (see _render_perclass_bars).
METRIC_OPTIONS = ["f1", "precision", "recall", "specificity", "roc_auc", "average_precision", "support"]
COUNT_METRICS = {"support"}
CELL_SIZE = 34
LABEL_COL_WIDTH = 96
# Beside or under each confusion grid: the confused-pairs table, and past
# FOCUS_MAX classes the whole-matrix picture.
PAIRS_TABLE_WIDTH = 480
OVERVIEW_SIZE = 420
# One picker row is ~44px; past this many experiments the list scrolls
# instead of pushing every chart below the fold.
PICKER_ROW_HEIGHT = 44
PICKER_MAX_VISIBLE_ROWS = 7
# Inline chart box, in page pixels. Every chart is also click-to-enlarge
# (see components.fig_to_zoomable_image), so this only has to stay legible,
# not fit everything -- but it does have to match the figure's own aspect
# ratio: ft.Image uses ImageFit.CONTAIN, so a box shaped differently from
# the figure letterboxes it into a strip.
CHART_MAX_WIDTH = 1200
CHART_PX_PER_INCH = 90


def _class_tick_labels(labels_union: list, per_class_by_name: dict) -> tuple:
    """"car" -> "car [126]", where 126 is that class's support (ground-truth
    row count) in the first selected experiment that has it. Also returns
    the subset of labels_union whose support is not the same number in
    every selected experiment, so the caller can flag that the single count
    shown is not universal -- this happens when experiments don't share the
    exact same uuid overlap with the dataset."""
    tick_labels = []
    uneven = set()
    for label in labels_union:
        supports = [
            int(pc.loc[label, "support"])
            for pc in per_class_by_name.values()
            if label in pc.index and pc.loc[label, "support"] == pc.loc[label, "support"]
        ]
        if supports:
            tick_labels.append(f"{label} [{supports[0]}]")
            if len(set(supports)) > 1:
                uneven.add(label)
        else:
            tick_labels.append(label)
    return tick_labels, uneven


def _chart_box(fig) -> dict:
    """width/height for fig_to_zoomable_image that keep `fig`'s own aspect
    ratio, so a figure that had to grow wide for many experiments is shown
    wide rather than letterboxed inside a fixed box."""
    fig_w, fig_h = fig.get_size_inches()
    width = min(CHART_MAX_WIDTH, max(600.0, fig_w * CHART_PX_PER_INCH))
    return {"width": int(width), "height": int(width * fig_h / fig_w)}


class MetricsView:
    def __init__(self, ctx):
        self.ctx = ctx
        self.experiment_checks: dict = {}
        self._metrics_cache: dict = {}  # experiment name -> ExperimentMetrics
        self._curves_cache: dict = {}  # (experiment name, target) -> CurveData
        # experiment name -> UuidOverlap. Cheap per call but not free (two
        # sets over every uuid on both sides), and _render() re-runs it for
        # every selected experiment -- which used to mean a full pass over
        # every uuid just to tick a checkbox or drag a row.
        self._overlap_cache: dict = {}
        self._dirty = False

        self.metric_dropdown = ft.Dropdown(
            label="Per-class metric", value="f1",
            options=[ft.dropdown.Option(m) for m in METRIC_OPTIONS],
            width=200, on_change=self._on_change, border_radius=8,
        )
        self.curve_target_dropdown = ft.Dropdown(
            label="Class for ROC / PR curves", value=MACRO_AVERAGE,
            width=260, on_change=self._on_change, border_radius=8,
        )
        # Drag-to-reorder picker (see _rebuild_experiment_checks): the list
        # itself, plus the "n of m selected" line above it. Height is
        # capped so a dozen experiments scroll here rather than pushing the
        # charts off the screen.
        self.experiments_list = ft.ReorderableListView(
            controls=[], padding=ft.padding.symmetric(vertical=2),
            on_reorder=self._on_reorder,
            # The platform's own handles are only rendered on desktop
            # targets; each row carries its own explicit one instead, so
            # the affordance is visible wherever this is opened.
            show_default_drag_handles=False,
        )
        self.experiments_list_container = ft.Container(content=self.experiments_list, height=PICKER_ROW_HEIGHT)
        self.selection_summary = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_banner = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2), self.progress_text],
            spacing=10, visible=False,
        )
        self.stale_banner = StaleBanner(self._on_update_click)
        self.rebuild_job = BackgroundRebuild(ctx, "Metrics")

        # ---- report export ----
        # Everything this tab computes dies with the browser tab. One file,
        # self-contained, is what gets archived next to a training run.
        self.report_browser = ServerFileBrowser(ctx)
        self.report_dir_field = ft.TextField(label="Destination folder", expand=True,
                                             border_radius=8, prefix_icon=ft.Icons.FOLDER_OPEN)
        self.report_name_field = ft.TextField(label="File name", width=320, border_radius=8)
        self.report_per_class = ft.Checkbox(label="Per-class tables", value=True)
        self.report_confusion = ft.Checkbox(label="Confusion matrices", value=True)
        self.report_btn = ft.FilledButton("Export HTML report", icon=ft.Icons.DESCRIPTION_OUTLINED,
                                          on_click=self._on_export_report,
                                          style=ft.ButtonStyle(bgcolor=ACCENT))
        self.report_pptx_btn = ft.FilledButton("Export PowerPoint", icon=ft.Icons.SLIDESHOW_OUTLINED,
                                               on_click=self._on_export_pptx,
                                               style=ft.ButtonStyle(bgcolor=ACCENT))
        self.report_status = ft.Column(spacing=4)
        self.report_progress = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2),
             ft.Text("Building the report...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)],
            spacing=10, visible=False)
        self.report_job = BackgroundRebuild(ctx, "Report")
        self.warnings_area = ft.Column(spacing=6)
        self.summary_area = ft.Column()
        self.perclass_area = ft.Container()
        self.curves_area = ft.Container()
        # wrap=True rather than a single horizontally-scrolling strip: with
        # a dozen experiments selected, the strip put everything past the
        # third one behind a scrollbar most people never noticed, so the
        # comparison this tab exists for silently showed three experiments.
        self.confusion_area = ft.Row(spacing=16, wrap=True, run_spacing=16)
        # (card, natural grid width) per rendered matrix, so a window
        # resize can re-fit them without recomputing anything.
        self._confusion_cards: list = []
        # Past FOCUS_MAX classes the confusion grids and the per-class
        # chart show a focus set (see core/class_focus.py) -- one for the
        # whole tab, so every experiment's grid and every bar group are the
        # same classes. Changing it redraws from the metrics already
        # computed; nothing is recomputed.
        self.class_focus = ClassFocus(cf.FOCUS_CRITERIA, self._on_focus_change,
                                      what="the confusion grids and the per-class chart",
                                      support_criteria=(cf.FOCUS_WORST_F1,))
        self._focus: list = []
        self._many_classes = False
        self._render_inputs = None  # (metrics_by_name, all_names) of the last render
        # Full-matrix thumbnails, drawn once per confusion matrix (see
        # class_matrix.overview_image); emptied with the metrics cache.
        self._overview_cache: dict = {}
        # Per-metadata-slice breakdown. Its own module and its own cache:
        # picking a column there costs a set of boolean masks over merges
        # this tab has already made, and costs nothing at all until a
        # column is picked (see ui/slice_panel.py).
        self.slice_panel = SlicePanel(ctx)
        self.root = ft.Container(padding=20)

        self.carousel_title = ft.Text(weight=ft.FontWeight.BOLD)
        self.carousel_content = ft.Container()  # width comes from the carousel, which sizes itself to the page
        self._pending_gallery_nav = None
        self.gallery_action = ft.TextButton("Open in Gallery", icon=ft.Icons.GRID_VIEW, on_click=self._on_open_in_gallery)
        self.carousel_dialog = ft.AlertDialog(
            # modal=False so tapping outside dismisses; on_dismiss then has
            # to sync `open` back to False, which Flet does not do itself --
            # a stale True makes the next page.open() a no-op.
            modal=False,
            title=self.carousel_title,
            content=self.carousel_content,
            on_dismiss=self._on_carousel_close,
            actions=[
                self.gallery_action,
                ft.TextButton("Close", on_click=self._on_carousel_close),
            ],
        )

    # ---- experiment picker (select + order) ----
    def _rebuild_experiment_checks(self):
        """Rebuild the picker from AppState.experiments, preserving which
        experiments were ticked. The list order *is* state's order -- this
        never sorts -- so what the user dragged is what every chart reads."""
        state = self.ctx.state
        existing = {name: cb.value for name, cb in self.experiment_checks.items()}
        self.experiment_checks.clear()
        self.experiments_list.controls.clear()
        names = list(state.experiments)
        for position, name in enumerate(names):
            default = existing.get(name, True)
            cb = ft.Checkbox(label=name, value=default, on_change=self._on_change, expand=True)
            self.experiment_checks[name] = cb
            self.experiments_list.controls.append(self._build_picker_row(position, name, names, cb))

        visible_rows = max(1, min(len(names), PICKER_MAX_VISIBLE_ROWS))
        self.experiments_list_container.height = visible_rows * PICKER_ROW_HEIGHT
        self._update_selection_summary()

        dataset_labels = state.dataset.labels() if state.dataset is not None else []
        class_options = [MACRO_AVERAGE] + dataset_labels
        self.curve_target_dropdown.options = [ft.dropdown.Option(c) for c in class_options]
        make_searchable(self.curve_target_dropdown)
        if self.curve_target_dropdown.value not in class_options:
            self.curve_target_dropdown.value = MACRO_AVERAGE

    def _build_picker_row(self, position: int, name: str, all_names: list, checkbox: ft.Checkbox) -> ft.Control:
        """One row of the picker: the color the charts will use for this
        experiment, its checkbox, and two ways to move it.

        The drag handle is a ReorderableDraggable -- Flet's name for
        Flutter's ReorderableDragStartListener -- so only the handle starts
        a drag and the rest of the row stays clickable (dragging from
        anywhere on the row would swallow the checkbox tap). The up/down
        buttons are not a redundancy: they are the only way to reorder with
        a keyboard, or on a trackpad that makes press-and-drag awkward, and
        they do the exact same thing.
        """
        color = color_for_experiment(name, self.ctx.state.color_slots)
        handle = ft.ReorderableDraggable(
            index=position,
            content=ft.Container(
                content=ft.Icon(ft.Icons.DRAG_INDICATOR, size=18, color=ft.Colors.ON_SURFACE_VARIANT),
                padding=ft.padding.symmetric(horizontal=4),
                tooltip="Drag to reorder",
            ),
        )
        return ft.Container(
            key=name,
            height=PICKER_ROW_HEIGHT,
            padding=ft.padding.only(left=4, right=8),
            content=ft.Row(
                [
                    handle,
                    ft.Container(width=10, height=10, bgcolor=color, border_radius=5),
                    checkbox,
                    ft.IconButton(
                        ft.Icons.KEYBOARD_ARROW_UP, icon_size=16, tooltip="Move up",
                        disabled=position == 0,
                        on_click=lambda e, n=name: self._move_experiment(n, -1),
                    ),
                    ft.IconButton(
                        ft.Icons.KEYBOARD_ARROW_DOWN, icon_size=16, tooltip="Move down",
                        disabled=position == len(all_names) - 1,
                        on_click=lambda e, n=name: self._move_experiment(n, 1),
                    ),
                ],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _update_selection_summary(self):
        total = len(self.experiment_checks)
        selected = len(self._selected_names())
        self.selection_summary.value = (
            f"{selected} of {total} experiment(s) shown -- drag, or use the arrows, "
            f"to set the order used by every table, legend and chart below."
        )

    def _on_reorder(self, e: ft.OnReorderEvent):
        """Flet already normalizes new_index for the downward case (its Dart
        side sends new-1 when moving down), so this is a plain pop/insert."""
        names = list(self.ctx.state.experiments)
        if not (0 <= e.old_index < len(names)):
            return
        name = names.pop(e.old_index)
        names.insert(max(0, min(e.new_index, len(names))), name)
        self._apply_order(names)

    def _move_experiment(self, name: str, delta: int):
        names = list(self.ctx.state.experiments)
        if name not in names:
            return
        old = names.index(name)
        new = old + delta
        if not (0 <= new < len(names)):
            return
        names.insert(new, names.pop(old))
        self._apply_order(names)

    def _apply_order(self, names: list):
        """Commit a new order and redraw. Nothing is recomputed: reordering
        only changes which cached result is read first; colors stay put."""
        self.ctx.state.reorder_experiments(names)
        self._rebuild_experiment_checks()
        self._render()
        if self.root.page:
            self.root.update()

    def _on_select_all(self, e):
        self._set_all_checks(True)

    def _on_select_none(self, e):
        self._set_all_checks(False)

    def _set_all_checks(self, value: bool):
        for cb in self.experiment_checks.values():
            cb.value = value
        self._update_selection_summary()
        self._render()
        if self.root.page:
            self.root.update()

    def _on_sort_alphabetically(self, e):
        self._apply_order(sorted(self.ctx.state.experiments))

    def _selected_names(self):
        return [name for name, cb in self.experiment_checks.items() if cb.value]

    def _on_change(self, e):
        self._update_selection_summary()
        self._render()
        if self.root.page:
            self.root.update()

    # ---- progress feedback for cache-miss (actual) computation ----
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

    # ---- cached accessors ----
    def _get_merged(self, merged_cache: dict, name: str):
        if name not in merged_cache:
            merged_cache[name] = self.ctx.state.merged(name)
        return merged_cache[name]

    def _get_metrics(self, name: str, merged_cache: dict):
        if name not in self._metrics_cache:
            state = self.ctx.state
            exp = state.experiments[name]
            merged = self._get_merged(merged_cache, name)
            _log(f"computing metrics for '{name}' ({len(merged):,} merged rows)...")
            t0 = time.monotonic()
            self._metrics_cache[name] = compute_metrics(name, merged, state.dataset.columns.label_col, exp.score_cols, exp.class_labels)
            _log(f"metrics for '{name}' done in {time.monotonic() - t0:.1f}s")
        return self._metrics_cache[name]

    def _get_curves(self, name: str, target: str, merged_cache: dict):
        key = (name, target)
        if key not in self._curves_cache:
            state = self.ctx.state
            exp = state.experiments[name]
            merged = self._get_merged(merged_cache, name)
            _log(f"computing ROC/PR curves for '{name}' / target={target!r} ({len(merged):,} merged rows)...")
            t0 = time.monotonic()
            self._curves_cache[key] = compute_curves(merged, state.dataset.columns.label_col, exp.score_cols, exp.class_labels, target)
            _log(f"curves for '{name}' / {target!r} done in {time.monotonic() - t0:.1f}s")
        return self._curves_cache[key]

    def _render(self):
        state = self.ctx.state
        self.summary_area.controls.clear()
        self.warnings_area.controls.clear()
        if state.dataset is None or not state.experiments:
            self.perclass_area.content = info_banner("Load a dataset and at least one experiment (results CSV) in the 'Data' tab first.")
            self.curves_area.content = None
            self.confusion_area.controls = []
            self._confusion_cards = []
            self.class_focus.hide()
            self.slice_panel.set_experiments([])
            self._clear_progress()
            return

        selected = self._selected_names()
        if not selected:
            self.perclass_area.content = info_banner("Select at least one experiment above to see metrics.")
            self.curves_area.content = None
            self.confusion_area.controls = []
            self._confusion_cards = []
            self.class_focus.hide()
            self.slice_panel.set_experiments([])
            self._clear_progress()
            return

        for name in selected:
            if name not in self._overlap_cache:
                self._overlap_cache[name] = state.uuid_overlap(name)
            banner = uuid_overlap_banner(self._overlap_cache[name], "dataset", name)
            if banner:
                self.warnings_area.controls.append(banner)

        # Merges are fetched at most once per experiment per render pass
        # (AppState.merged() also caches the single most-recent experiment
        # across passes) and reused by both metrics and curves below.
        #
        # This whole pass is wrapped in try/except: an uncaught exception
        # here (a bad merge, an unexpected data shape) would otherwise leave
        # the progress banner set above stuck on-screen forever with no
        # indication anything went wrong -- exactly what looked like an
        # infinite "Computing metrics" hang before this was added.
        try:
            merged_cache: dict = {}
            metrics_by_name = {}
            for i, name in enumerate(selected):
                if name not in self._metrics_cache:
                    self._set_progress(f"Computing metrics -- {name} ({i + 1}/{len(selected)})")
                metrics_by_name[name] = self._get_metrics(name, merged_cache)

            all_names = list(state.experiments.keys())
            for name, m in metrics_by_name.items():
                banner = score_diagnostics_banner(m.score_diagnostics, name)
                if banner:
                    self.warnings_area.controls.append(banner)

            # Each phase announces itself. Only worth doing now that the
            # rebuild runs off the UI thread (see _refresh_in_background):
            # before that these messages could never paint, and the curves
            # -- by far the longest phase, one full ROC + PR sweep per class
            # per experiment -- ran under a banner still naming the metrics
            # step that had already finished.
            self._set_progress("Building the summary table...")
            self._render_summary(metrics_by_name, all_names)
            self._render_inputs = (metrics_by_name, all_names)
            self._resolve_focus(metrics_by_name)
            self._set_progress("Drawing per-class metrics...")
            self._render_perclass_bars(metrics_by_name, all_names)
            self._set_progress(f"Computing ROC / PR curves ({len(selected)} experiment(s))...")
            self._render_curves(selected, all_names, merged_cache)
            self._set_progress("Drawing confusion matrices...")
            self._render_confusion(metrics_by_name, all_names)
            # Re-renders the slice breakdown from its own cache when a
            # column is picked, and does nothing otherwise -- it is never
            # part of the cost of an Update.
            self.slice_panel.set_experiments(selected)
        except Exception as ex:
            _log(f"error computing metrics: {ex}")
            traceback.print_exc()
            self.perclass_area.content = info_banner(f"Error computing metrics: {ex}", icon=ft.Icons.ERROR_OUTLINE)
            self.curves_area.content = None
            self.confusion_area.controls = []
            self._confusion_cards = []
            self.class_focus.hide()
        self._clear_progress()

    def _render_summary(self, metrics_by_name, all_names):
        # Every scalar compute_metrics() produces. The three Brier
        # decomposition terms used to be computed on every pass and then
        # dropped on the floor here, so the tab reported a Brier score with
        # no way to see whether a bad one came from miscalibration
        # (reliability) or from a model that simply cannot separate the
        # classes (resolution) -- which is the whole point of decomposing it.
        headers = [
            "Experiment", "N", "Accuracy", "Balanced Acc", "Macro F1", "Weighted F1",
            "MCC", "Cohen's kappa", "Macro ROC-AUC", "Weighted ROC-AUC",
            "Macro AP", "Weighted AP", "Log loss", "Brier",
            "Brier reliability", "Brier resolution", "Brier uncertainty", "ECE", "MCE",
        ]
        columns = [ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD)) for c in headers]
        rows = []
        for name, m in metrics_by_name.items():
            color = color_for_experiment(name, self.ctx.state.color_slots)
            name_cell = ft.Row([ft.Container(width=10, height=10, bgcolor=color, border_radius=5), ft.Text(name)], spacing=8)
            values = [
                m.accuracy, m.balanced_accuracy, m.macro_f1, m.weighted_f1,
                m.mcc, m.cohen_kappa, m.macro_roc_auc, m.weighted_roc_auc,
                m.macro_avg_precision, m.weighted_avg_precision,
                m.log_loss, m.brier_score,
                m.brier_reliability, m.brier_resolution, m.brier_uncertainty,
                m.ece, m.mce,
            ]
            rows.append(ft.DataRow(cells=[
                ft.DataCell(name_cell),
                ft.DataCell(ft.Text(f"{m.n_samples:,}")),
                *[ft.DataCell(ft.Text(f"{v:.3f}" if v == v else "—")) for v in values],
            ]))
        table = ft.DataTable(columns=columns, rows=rows, heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8)
        self.summary_area.controls.append(card(ft.Column([
            section_title("Summary", icon=ft.Icons.SUMMARIZE_OUTLINED, ctx=self.ctx, manual="summary-table"),
            ft.Row([table], scroll=ft.ScrollMode.ALWAYS),
        ], spacing=10)))

    # Which manual page the (i) beside the per-class chart opens: the chart
    # draws whichever metric the dropdown holds, so a fixed page would
    # explain F1 to someone looking at a specificity bar chart.
    PERCLASS_MANUAL_PAGES = {
        "f1": "f1",
        "precision": "precision-recall",
        "recall": "precision-recall",
        "specificity": "precision-recall",
        "roc_auc": "roc-auc",
        "average_precision": "pr-auc",
        "support": "support-and-imbalance",
    }

    def _perclass_manual_page(self) -> str:
        return self.PERCLASS_MANUAL_PAGES.get(self.metric_dropdown.value, "f1")

    def _render_perclass_bars(self, metrics_by_name, all_names):
        metric_key = self.metric_dropdown.value
        per_class_by_name = {name: m.per_class.set_index("label") for name, m in metrics_by_name.items()}
        labels_union = sorted(set().union(*[set(pc.index) for pc in per_class_by_name.values()]))
        n_all_classes = len(labels_union)
        if self._many_classes:
            # The focus set, in its own order (worst first for a ranking).
            labels_union = [l for l in self._focus if l in set(labels_union)]
        n_exp = len(metrics_by_name)
        # Both axes have to grow with the number of experiments, not just
        # the number of classes: each class's group holds one bar per
        # experiment, so at a dozen experiments a fixed-width figure drew
        # bars a pixel wide and a legend that covered the plot.
        fig_width = max(6.0, min(18.0, len(labels_union) * (0.5 + 0.13 * n_exp)))
        # The legend always goes under the axes (see components.legend_below),
        # so the figure grows to hold it rather than the plot shrinking.
        legend_cols = legend_columns(n_exp)
        fig, ax = subplots(
            figsize=(fig_width, 4.3 + legend_height_inches(n_exp, legend_cols)))
        width = 0.8 / max(n_exp, 1)
        x = np.arange(len(labels_union))
        for i, (name, m) in enumerate(metrics_by_name.items()):
            per_class = per_class_by_name[name]
            values = [per_class[metric_key].get(l, np.nan) for l in labels_union]
            offset = (i - (n_exp - 1) / 2) * width
            ax.bar(x + offset, values, width, label=name, color=color_for_experiment(name, self.ctx.state.color_slots))
        ax.set_xticks(x)
        tick_labels, uneven_support = _class_tick_labels(labels_union, per_class_by_name)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylabel(metric_key)
        if metric_key in COUNT_METRICS:
            # Support is a row count, not a rate -- the shared 0..1 limit
            # would flatten every bar against the top of the axes.
            ax.set_ylabel(f"{metric_key} (rows)")
        else:
            ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-class {metric_key} by experiment")
        # x carries class names, so there is nothing between two of them for
        # a minor line to divide; y is a scale, where the minor lines are
        # what lets a bar be read to a hundredth.
        style_axes(ax, x=GRID_NONE)
        # Under the plot, always: an in-axes legend sits on top of the bars
        # it is labelling exactly when there are enough of them to need it.
        reserve_legend_space(fig, n_exp, legend_cols)
        legend_below(ax, ncol=legend_cols)
        notes = ["[n] under each class name is its support -- the number of ground-truth examples used to compute that class's metrics."]
        if self._many_classes:
            notes.insert(0, f"{len(labels_union)} of {n_all_classes:,} classes -- the ones picked under "
                            f"'Classes shown' above.")
        if uneven_support:
            first_name = next(iter(per_class_by_name))
            notes.append(
                f"Support differs across the selected experiments for: {', '.join(sorted(uneven_support))} "
                f"(the uuid overlap with the dataset differs by experiment); the count shown is {first_name}'s."
            )
        self.perclass_area.content = card(ft.Column([
            section_title("Per-class metric comparison", icon=ft.Icons.BAR_CHART,
                          ctx=self.ctx, manual=self._perclass_manual_page),
            self.metric_dropdown,
            fig_to_zoomable_image(fig, self.ctx, f"Per-class {metric_key} by experiment", **_chart_box(fig)),
            *[ft.Text(n, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT) for n in notes],
        ], spacing=10))

    def _render_curves(self, selected, all_names, merged_cache):
        target = self.curve_target_dropdown.value or MACRO_AVERAGE
        curves_by_name = {}
        for i, name in enumerate(selected):
            if (name, target) not in self._curves_cache:
                self._set_progress(f"Computing ROC / PR curves -- {name} ({i + 1}/{len(selected)})")
            curves_by_name[name] = self._get_curves(name, target, merged_cache)

        # A curve whose target class has a constant score column is not a
        # weak curve, it's a non-result (AUC exactly 0.5, AP exactly the
        # prevalence) -- drawing it invites reading a straight line as a
        # measurement, so those experiments are left off the axes entirely.
        drawable = {n: c for n, c in curves_by_name.items() if c.disabled_reason is None}
        if not drawable:
            reason = next(iter(curves_by_name.values())).disabled_reason
            self.curves_area.content = card(ft.Column([
                section_title("ROC & Precision-Recall curves", icon=ft.Icons.SHOW_CHART, ctx=self.ctx, manual="roc-auc"),
                self.curve_target_dropdown,
                info_banner(
                    f"ROC and Precision-Recall curves are disabled for {target}: {reason}. "
                    f"The decision-level metrics for this class (precision, recall, F1, and its "
                    f"confusion matrix row) are unaffected and shown above.",
                    icon=ft.Icons.RULE,
                ),
            ], spacing=10))
            return

        notes = []
        skipped = [n for n in curves_by_name if n not in drawable]
        if skipped:
            notes.append(f"Not shown: {', '.join(skipped)} ({curves_by_name[skipped[0]].disabled_reason}).")
        excluded = sorted({l for c in drawable.values() for l in c.excluded_labels})
        if excluded:
            notes.append(
                f"Macro-average taken over the classes whose score column varies; "
                f"excluded: {', '.join(excluded)}."
            )

        # The legend goes under each plot rather than in a corner of it: a
        # curve is at its most interesting near the corners, which is
        # exactly where an in-axes legend has to sit. The figure grows by
        # the height the legend needs so the axes keep theirs.
        n_curves = len(drawable)
        legend_cols = min(2, n_curves)
        fig, (ax1, ax2) = subplots(
            1, 2, figsize=(11, 4.4 + legend_height_inches(n_curves, legend_cols)))
        ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        for name, curve in drawable.items():
            color = color_for_experiment(name, self.ctx.state.color_slots)
            ax1.plot(curve.fpr, curve.tpr, color=color, label=f"{name} (AUC={curve.roc_auc:.3f})")
            ax2.plot(curve.recall, curve.precision, color=color, label=f"{name} (AP={curve.average_precision:.3f})")
        ax1.set_xlabel("False positive rate")
        ax1.set_ylabel("True positive rate")
        ax1.set_title(f"ROC curve — {target}")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision-Recall curve — {target}")
        # Both axes are rates on the same 0..1 scale, so both sides get the
        # full grid -- reading "95% TPR at what FPR" off a curve is the
        # whole use of these two plots.
        style_axes(ax1)
        style_axes(ax2)
        reserve_legend_space(fig, n_curves, legend_cols)
        for ax in (ax1, ax2):
            legend_below(ax, ncol=legend_cols, fontsize=7)
        self.curves_area.content = card(ft.Column([
            section_title("ROC & Precision-Recall curves", icon=ft.Icons.SHOW_CHART),
            self.curve_target_dropdown,
            fig_to_zoomable_image(fig, self.ctx, f"ROC & Precision-Recall — {target}", **_chart_box(fig)),
            *[ft.Text(n, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT) for n in notes],
        ], spacing=10))

    def _render_confusion(self, metrics_by_name, all_names):
        # Each card's width comes from the window (see _build_confusion_grid),
        # so on_window_resized has to find them again; a matrix is rebuilt
        # only when the metrics behind it are.
        self._confusion_cards = []
        self.confusion_area.controls = [
            self._build_confusion_grid(name, m, all_names, self._focus, self._many_classes)
            for name, m in metrics_by_name.items()
        ]

    # ---- focus set: which classes the grids and bars show (see core/class_focus.py) ----
    def _resolve_focus(self, metrics_by_name) -> None:
        confusions = [m.confusion for m in metrics_by_name.values()]
        per_class = [m.per_class for m in metrics_by_name.values()]
        self._focus = self.class_focus.resolve(
            cf.all_labels(confusions),
            lambda criterion: cf.rank_classes(criterion, confusions, per_class,
                                              min_support=self.class_focus.min_support_for(criterion)))
        self._many_classes = self.class_focus.many

    def _on_focus_change(self):
        """Redraw the per-class chart and the matrices from the metrics
        already on screen."""
        if self._render_inputs is None:
            return
        metrics_by_name, all_names = self._render_inputs
        self._resolve_focus(metrics_by_name)
        self._render_perclass_bars(metrics_by_name, all_names)
        self._render_confusion(metrics_by_name, all_names)
        for control in (self.class_focus.build(), self.perclass_area, self.confusion_area):
            if control.page:
                control.update()

    def _build_confusion_grid(self, name, m, all_names, focus: list, many: bool) -> ft.Control:
        cm = m.confusion
        # Past FOCUS_MAX classes the grid is the focus rows by the focus
        # columns, plus one "other" column per row for whatever was
        # predicted outside the set -- so a row still sums to its support.
        if many:
            grid, other = cf.focus_matrix(cm, focus)
        else:
            grid, other = cm, None
        row_labels = list(grid.index)
        col_labels = list(grid.columns)
        # Read the counts out of the array once rather than a pair of label
        # lookups per cell.
        counts = grid.to_numpy()
        max_val = int(counts.max()) if counts.size else 1
        other_counts = other.to_numpy() if other is not None else None
        n_columns = len(col_labels) + (1 if many else 0)

        # A few classes make a big grid, not a tiny one: the cells take the
        # room the missing columns would have used (see matrix_cell_size),
        # and the text in them follows.
        cell = matrix_cell_size(n_columns, CELL_SIZE)
        count_size = matrix_text_size(cell, CELL_SIZE, 10)
        header_size = matrix_text_size(cell, CELL_SIZE, 9, cap=14)
        header_chars = matrix_label_chars(cell, CELL_SIZE, 4)
        row_label_size = matrix_text_size(cell, CELL_SIZE, 10, cap=14)

        # Two lines of header text per column, without the prefix every
        # class shares (see short_column_labels).
        short_labels, dropped_prefix = short_column_labels(col_labels, 2 * header_chars)
        header_cells = [ft.Container(width=LABEL_COL_WIDTH)] + [
            ft.Container(
                content=ft.Text(short, size=header_size, text_align=ft.TextAlign.CENTER, max_lines=2),
                width=cell, height=2.6 * header_size + 4, alignment=ft.alignment.bottom_center, tooltip=pl,
            ) for short, pl in zip(short_labels, col_labels)
        ]
        if many:
            header_cells.append(ft.Container(
                content=ft.Text("other", size=header_size, italic=True),
                width=cell, alignment=ft.alignment.center,
                tooltip="Predicted as any class not shown in this grid",
            ))
        body_rows = [ft.Row(header_cells, spacing=2)]
        for i, tl in enumerate(row_labels):
            row_controls = [
                ft.Container(
                    content=ft.Text(tl, size=row_label_size, weight=ft.FontWeight.BOLD,
                                    no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                    width=LABEL_COL_WIDTH, alignment=ft.alignment.center_right,
                    padding=ft.padding.only(right=8), tooltip=tl,
                )
            ]
            for j, pl in enumerate(col_labels):
                count = int(counts[i, j])
                fraction = count / max_val if max_val else 0.0
                bg = blues_color(fraction)
                fg = "#ffffff" if fraction > 0.55 else "#111111"
                row_controls.append(
                    ft.Container(
                        width=cell, height=cell, bgcolor=bg, border_radius=4,
                        alignment=ft.alignment.center, ink=True,
                        content=ft.Text(str(count), size=count_size, color=fg),
                        tooltip=f"true = {tl}\npredicted = {pl}\nn = {count}",
                        on_click=lambda e, tl=tl, pl=pl: self._on_confusion_click(name, tl, pl),
                    )
                )
            if many:
                count = int(other_counts[i])
                row_controls.append(ft.Container(
                    width=cell, height=cell, bgcolor=ft.Colors.SURFACE, border_radius=4,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    alignment=ft.alignment.center, ink=bool(count),
                    content=ft.Text(str(count), size=count_size, italic=True),
                    tooltip=f"true = {tl}\npredicted = any class not in this grid\nn = {count}",
                    on_click=(lambda e, tl=tl: self._on_confusion_other_click(name, tl, focus)) if count else None,
                ))
            body_rows.append(ft.Row(row_controls, spacing=2))

        color = color_for_experiment(name, self.ctx.state.color_slots)
        pairs = cf.top_confused_pairs(cm)
        pairs_table = confused_pairs_table(
            pairs.head(PAIRS_SHOWN), len(pairs),
            on_click=lambda tl, pl: self._on_confusion_click(name, tl, pl),
        )
        extras = [ft.Container(pairs_table, width=PAIRS_TABLE_WIDTH)]
        if many:
            extras.append(ft.Column([
                ft.Text(f"All {len(cm.index):,} classes -- click to enlarge and zoom",
                        size=12, weight=ft.FontWeight.BOLD),
                overview_image(self.ctx, cm, focus, f"{name} -- full confusion matrix", OVERVIEW_SIZE,
                               cache=self._overview_cache),
            ], spacing=6))
        # Sized to what it holds rather than left to fill the row: an
        # unconstrained card stretches to the full page width, which turns
        # the wrapping row of matrices into one tall column and puts most
        # of the comparison off-screen (see grid_card_width).
        grid_width = LABEL_COL_WIDTH + n_columns * (cell + 2)
        content_width = max(grid_width, PAIRS_TABLE_WIDTH + (OVERVIEW_SIZE + 16 if many else 0))
        matrix = card(
            ft.Column([
                ft.Row([
                    ft.Container(width=10, height=10, bgcolor=color, border_radius=5),
                    ft.Text(f"{name} — accuracy {m.accuracy:.3f}", weight=ft.FontWeight.BOLD, size=13),
                ], spacing=8),
                # AUTO, not ALWAYS: most matrices have nothing to scroll,
                # and a permanent scrollbar under them is a lie.
                ft.Row([ft.Column(body_rows, spacing=2)], scroll=ft.ScrollMode.AUTO),
                ft.Row([
                    ft.Icon(ft.Icons.TOUCH_APP, size=13, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("Click a cell or a pair to preview those images right here"
                            + (f" -- column headers leave out the shared prefix '{dropped_prefix}'" if dropped_prefix else ""),
                            size=10, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                ], spacing=6),
                ft.Row(extras, spacing=16, wrap=True, vertical_alignment=ft.CrossAxisAlignment.START),
            ], spacing=10),
            width=grid_card_width(self.ctx.page, content_width),
        )
        self._confusion_cards.append((matrix, content_width))
        return matrix

    def on_window_resized(self) -> None:
        """The matrices are sized from the window, so they have to follow
        it: widening the app should widen the matrix that was clipped, not
        leave it at the width it happened to be built at. Nothing is rebuilt
        here -- only the card widths, which is what the window decides."""
        changed = False
        for matrix, grid_width in self._confusion_cards:
            width = grid_card_width(self.ctx.page, grid_width)
            if width != matrix.width:
                matrix.width = width
                changed = True
        if changed and self.confusion_area.page:
            self.confusion_area.update()

    def _on_confusion_click(self, experiment_name: str, true_label: str, predicted_label: str):
        state = self.ctx.state
        merged = state.merged(experiment_name)
        cols = state.dataset.columns
        # A single cell can match many thousands of rows at scale, so only
        # the DataFrame slice is kept here -- Carousel converts to records
        # one page at a time as the user pages through, via the closure below.
        subset = merged[(merged[cols.label_col] == true_label) & (merged["predicted_label"] == predicted_label)]
        self._open_confusion_carousel(
            experiment_name, subset, f"{true_label} -> {predicted_label}",
            gallery_nav=(experiment_name, true_label, predicted_label),
        )

    def _on_confusion_other_click(self, experiment_name: str, true_label: str, focus: list):
        """The "other" cell of a focus grid: crops of `true_label` predicted
        as any class the grid does not show."""
        state = self.ctx.state
        merged = state.merged(experiment_name)
        cols = state.dataset.columns
        subset = merged[(merged[cols.label_col] == true_label) & ~merged["predicted_label"].isin(focus)]
        # The Gallery filters on one predicted class, so "any class outside
        # the grid" has no equivalent there.
        self._open_confusion_carousel(experiment_name, subset, f"{true_label} -> other classes", gallery_nav=None)

    def _open_confusion_carousel(self, experiment_name: str, subset, what: str, gallery_nav):
        state = self.ctx.state
        cols = state.dataset.columns
        self.carousel_title.value = f"{what}  ·  {experiment_name}  ·  {len(subset):,} image(s)"
        carousel = Carousel(
            state.dataset, len(subset),
            lambda start, end: subset.iloc[start:end].to_dict("records"),
            cols.crop_col, cols.uuid_col, ctx=self.ctx,
            review_source="confusion_matrix",
            # The cell the user clicked *is* a hypothesis about the label:
            # a crop in (true cat, predicted dog) is either a model error or
            # a mislabelled dog, so the prediction pre-fills the picker.
            suggestions={str(u): str(p) for u, p in zip(subset[cols.uuid_col], subset["predicted_label"])},
            label_col=cols.label_col,
        )
        self.carousel_content.content = carousel.build()
        self._pending_gallery_nav = gallery_nav
        self.gallery_action.visible = gallery_nav is not None
        self.ctx.page.open(self.carousel_dialog)

    def _on_open_in_gallery(self, e):
        self.ctx.page.close(self.carousel_dialog)
        if self._pending_gallery_nav and self.ctx.go_to_gallery:
            self.ctx.go_to_gallery(*self._pending_gallery_nav)

    def _on_carousel_close(self, e):
        self.ctx.page.close(self.carousel_dialog)

    def refresh(self):
        """Called via ctx.notify_change() whenever the underlying data
        changes. Computing metrics/curves for every selected experiment is
        real work at scale (sklearn over the full merged data), so this
        never happens automatically -- it just flags the view as stale and
        shows a banner. The cache is intentionally *not* cleared here
        either: until the user clicks Update, toggling which experiments
        are shown or which per-class metric is selected keeps working
        instantly against the last-known-good (now possibly stale) results,
        rather than silently recomputing against data the user hasn't
        asked to see yet."""
        self._dirty = True
        if self.root.page:
            self.stale_banner.show()

    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        """Which experiments were shown, in which order, and what the two
        dropdowns were set to. The order belongs here rather than to the
        workspace: it is a reading preference, and this tab is where it is
        set."""
        return {
            "metric": self.metric_dropdown.value or "",
            "curve_target": self.curve_target_dropdown.value or MACRO_AVERAGE,
            "selected": self._selected_names(),
            "order": list(self.ctx.state.experiments),
            "focus": self.class_focus.view_state(),
        }

    def restore_view_state(self, state: dict) -> list:
        """Put the picker back. Nothing is recomputed -- the tab shows its
        usual "click to update" banner, same as after any data change."""
        warnings = []
        loaded = set(self.ctx.state.experiments)

        order = [n for n in state.get("order", []) or [] if n in loaded]
        if order:
            self.ctx.state.reorder_experiments(order)

        self._rebuild_experiment_checks()
        selected = state.get("selected")
        if isinstance(selected, list):
            wanted = {n for n in selected if isinstance(n, str)}
            gone = sorted(wanted - loaded)
            if gone:
                warnings.append(f"Metrics: experiment(s) {', '.join(gone)} are not loaded.")
            for name, check in self.experiment_checks.items():
                check.value = name in wanted
            self._update_selection_summary()

        metric = state.get("metric")
        if metric in METRIC_OPTIONS:
            self.metric_dropdown.value = metric

        # The curve target is either MACRO_AVERAGE (a key) or a class label
        # from the user's CSV, which may no longer exist.
        target = state.get("curve_target")
        if isinstance(target, str) and target:
            dataset = self.ctx.state.dataset
            known = [MACRO_AVERAGE] + (list(dataset.labels()) if dataset is not None else [])
            if target in known:
                self.curve_target_dropdown.value = target
            else:
                warnings.append(f"Metrics: class '{target}' is not in this dataset -- curves show the macro average.")

        # Which classes the grids and the per-class chart show.
        self.class_focus.restore_view_state(state.get("focus"))

        if self.root.page:
            self.root.update()
        return warnings

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow an experiment rename (see AppContext.notify_rename)
        without recomputing anything: re-key every cache, rebuild the
        picker, re-render from what is already cached. Renaming is a
        relabel, and at half a million rows per experiment the difference
        between a relabel and a recompute is minutes."""
        if old_name in self._metrics_cache:
            metrics = self._metrics_cache.pop(old_name)
            metrics.name = new_name
            self._metrics_cache[new_name] = metrics
        if old_name in self._overlap_cache:
            self._overlap_cache[new_name] = self._overlap_cache.pop(old_name)
        for key in [k for k in self._curves_cache if k[0] == old_name]:
            self._curves_cache[(new_name, key[1])] = self._curves_cache.pop(key)
        self.slice_panel.rename_experiment(old_name, new_name)
        # Carry the checkbox state across so a renamed experiment doesn't
        # silently reappear in the charts the user had just hidden.
        if old_name in self.experiment_checks:
            self.experiment_checks[new_name] = self.experiment_checks.pop(old_name)

        self._rebuild_experiment_checks()
        # Only redraw the charts when they are up to date. While the tab is
        # stale, some selected experiment may have no cached result yet, and
        # re-rendering would quietly start the very computation the stale
        # banner exists to let the user decide about -- for a rename. The
        # picker above already shows the new name; the charts pick it up on
        # the Update click.
        if not self._dirty:
            self._render()
        if self.root.page:
            self.root.update()

    # ---- report export --------------------------------------------------
    def _on_browse_report_dir(self, e):
        self.report_browser.open(
            self._set_report_dir, start_path=self.report_dir_field.value or "",
            dirs_only=True, title="Where to write the report",
        )

    def _set_report_dir(self, path: str) -> None:
        self.report_dir_field.value = path
        if self.report_dir_field.page:
            self.report_dir_field.update()

    def _prepare_report_fields(self) -> None:
        state = self.ctx.state
        if not self.report_name_field.value and state.dataset is not None:
            self.report_name_field.value = default_report_name(state)
        if not self.report_dir_field.value and state.dataset is not None and state.dataset.csv_path:
            self.report_dir_field.value = str(Path(state.dataset.csv_path).parent)

    def _report_status(self, message: str, ok: bool) -> None:
        self.report_progress.visible = False
        self.report_status.controls = [status_row(message, "success" if ok else "error")]
        for control in (self.report_progress, self.report_status):
            if control.page:
                control.update()

    def _on_export_report(self, e):
        self._start_export(Report, "html")

    def _on_export_pptx(self, e):
        self._start_export(PptxReport, "pptx")

    def _start_export(self, report_cls: type, extension: str) -> None:
        selected = self._selected_names()
        if not selected:
            self._report_status("Select at least one experiment above to report on.", ok=False)
            return
        self._prepare_report_fields()
        directory = (self.report_dir_field.value or "").strip()
        if not directory:
            self._report_status("Pick a destination folder.", ok=False)
            return

        self.report_progress.visible = True
        self.report_status.controls = []
        for control in (self.report_progress, self.report_status, self.report_name_field,
                        self.report_dir_field):
            if control.page:
                control.update()
        # Recomputes every metric and redraws every confusion matrix, which
        # is the same work the Update button does -- so it goes off the UI
        # thread the same way.
        self.report_job.run(lambda: self._write_report(selected, directory, report_cls, extension),
                            on_error=lambda ex: self._report_status(f"Export failed: {ex}", ok=False))

    def _report_file_name(self, extension: str) -> str:
        """The file name field's value, re-extensioned for whichever export
        button was just clicked -- so switching from HTML to PowerPoint (or
        back) never produces "name.html.pptx"."""
        typed = (self.report_name_field.value or "").strip()
        if not typed:
            return default_report_name(self.ctx.state, extension)
        for known in (".html", ".pptx"):
            if typed.lower().endswith(known):
                typed = typed[:-len(known)]
                break
        return f"{typed}.{extension}"

    def _write_report(self, selected: list, directory: str, report_cls: type, extension: str) -> None:
        name = self._report_file_name(extension)
        # A comparison goes in whenever exactly two experiments are selected:
        # that is the case where "which one shipped and why" is the question
        # the report will be read for.
        pair = tuple(selected[:2]) if len(selected) == 2 else None
        report = build_report(
            self.ctx.state, selected,
            include_per_class=bool(self.report_per_class.value),
            include_confusion=bool(self.report_confusion.value),
            comparison_pair=pair,
            snapshot_id=self._bookmark_for_report(name),
            report_cls=report_cls,
            # What 'Classes shown' holds now, so the report matches the tab.
            focus=self._focus if self._many_classes else None,
        )
        dest = report.write(Path(directory) / name)
        size_kb = dest.stat().st_size / 1024
        _log(f"report: wrote {dest} ({size_kb:,.0f} KB)")
        self._report_status(
            f"Wrote {saved_path_message(self.ctx, dest)} ({size_kb:,.0f} KB) -- self-contained, "
            f"opens with no network.", ok=True)

    def _bookmark_for_report(self, report_name: str) -> str:
        """Save the current view and return its id, so the report can say
        how to reopen itself.

        Only possible when this session came from a library workspace -- there
        is otherwise nowhere to put the snapshot. Nothing here may stop a
        report being written: the report is what the user asked for, the
        bookmark is a convenience on top of it.
        """
        workspace = getattr(self.ctx, "current_workspace", None)
        if workspace is None or not workspace.exists():
            return ""
        try:
            snapshot = library.create_snapshot(
                workspace, self.ctx.view_state.capture(),
                name=f"Report: {report_name}",
                description="Saved automatically when this report was exported.",
            )
        except Exception as ex:  # noqa: BLE001 -- a report must still be written
            _log(f"report: could not bookmark the view ({ex})")
            return ""
        return snapshot.id if snapshot else ""

    def recompute_after_restore(self) -> None:
        """Rebuild the tables and charts the snapshot was showing. The
        most expensive rebuild in the app, and the reason a snapshot is
        worth having at all -- see ui/view_state.py."""
        if self.ctx.state.dataset is None or not self._selected_names():
            return
        self._on_update_click(None)

    def _on_update_click(self, e):
        self._metrics_cache.clear()
        self._curves_cache.clear()
        self._overlap_cache.clear()
        self._overview_cache.clear()
        self.slice_panel.invalidate()
        self._refresh_in_background()

    def _refresh_in_background(self):
        """The most expensive rebuild in the app -- ~19s for a million rows
        (scaling.md) -- and until this ran off the UI thread, the
        per-experiment progress messages _render() already emits could
        never appear: the app was simply frozen for the duration. Ignored
        while a rebuild is already running, which the progress banner is
        already reporting.

        _render() has its own try/except, so on_error here is only for a
        failure outside it."""
        self._set_progress("Computing metrics...")
        self.rebuild_job.run(self._refresh_now, on_error=self._on_rebuild_error)

    def _on_rebuild_error(self, ex: Exception):
        self._clear_progress()
        self.perclass_area.content = info_banner(
            f"Error computing metrics: {ex}", icon=ft.Icons.ERROR_OUTLINE)
        if self.root.page:
            self.root.update()

    def _refresh_now(self):
        self._rebuild_experiment_checks()
        self._render()
        self._dirty = False
        self.stale_banner.hide()
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        self.root.content = ft.Column(
            controls=[
                section_title("Compare experiments", icon=ft.Icons.INSIGHTS, ctx=self.ctx, manual="metric-levels"),
                ft.Text("Toggle which loaded experiments to include; all charts overlay them for direct comparison.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                self.stale_banner.control,
                card(ft.Column([
                    ft.Row([
                        self.selection_summary,
                        ft.Container(expand=True),
                        ft.TextButton("Select all", icon=ft.Icons.CHECK_BOX_OUTLINED, on_click=self._on_select_all),
                        ft.TextButton("Select none", icon=ft.Icons.CHECK_BOX_OUTLINE_BLANK, on_click=self._on_select_none),
                        ft.TextButton("Sort A-Z", icon=ft.Icons.SORT_BY_ALPHA, on_click=self._on_sort_alphabetically),
                    ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    self.experiments_list_container,
                ], spacing=8)),
                self.warnings_area,
                self.progress_banner,
                self.summary_area,
                self.class_focus.build(),
                section_title("Confusion matrices", icon=ft.Icons.GRID_ON, ctx=self.ctx, manual="confusion-matrix"),
                self.confusion_area,
                self.perclass_area,
                self.curves_area,
                self.slice_panel.build(),
                section_title("Export a report", icon=ft.Icons.DESCRIPTION_OUTLINED, ctx=self.ctx, manual="report-export"),
                ft.Text(
                    "Provenance (which CSV, which split, which review version, whether the reviewed "
                    "labels were applied), the summary table, the per-class tables and the confusion "
                    "matrices. Selecting exactly two experiments adds the head-to-head from the "
                    "Compare tab. HTML is one self-contained file that opens with no network and no "
                    "sibling files, and a browser prints it to PDF. PowerPoint is a native .pptx with "
                    "the same tables, figures and provenance laid out on slides.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                card(ft.Column([
                    ft.Row([self.report_dir_field,
                            ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse",
                                          on_click=self._on_browse_report_dir)], spacing=6),
                    ft.Row([self.report_name_field, self.report_per_class, self.report_confusion],
                           spacing=14, wrap=True),
                    ft.Row([self.report_btn, self.report_pptx_btn], spacing=12),
                    self.report_progress,
                    self.report_status,
                ], spacing=10)),
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
        )
        self._refresh_now()
        return self.root
