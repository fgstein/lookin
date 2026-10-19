"""Metrics: classification metrics per loaded experiment, overlaid on the
same charts -- the port of lookin.ui.metrics_view.

* the experiment picker is the order every table, legend and chart reads
  (the arrows move a row); the order lives on AppState.experiments, so
  it is the same in every tab -- the colors do not follow it;
* results are cached per experiment (and per experiment + curve target):
  ticking an experiment, picking another metric or curve class only draws;
* a data change marks the tab stale, and it recomputes everything when
  it is next shown (views.base.StaleTracking);
* a confusion cell (or a confused pair) opens its crops in the carousel,
  with "Open in Gallery" for the same filter there;
* past FOCUS_MAX classes, 'Classes shown' picks which ones the grids and
  bars show; the whole matrix is a picture beside each grid;
* the per-slice breakdown (slice_panel.py) and the report export.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

from lookin.core import class_focus as cf
from lookin.core import library
from lookin.core.chart_style import short_name, subplots
from lookin.core.metrics import MACRO_AVERAGE, compute_curves, compute_metrics
from lookin.core.report import Report
from lookin.core.report_builder import build_report, default_report_name
from lookin.core.report_pptx import PptxReport
from lookin.ui.class_matrix import PAIRS_SHOWN, short_column_labels
from lookin.ui.components import (
    GRID_NONE, blues_color, color_for_experiment, legend_below, legend_columns, legend_height_inches,
    matrix_cell_size, matrix_label_chars, matrix_text_size, reserve_legend_space, style_axes,
)
from lookin.ui.metrics_view import (
    CELL_SIZE, COUNT_METRICS, LABEL_COL_WIDTH, METRIC_OPTIONS, OVERVIEW_SIZE, _class_tick_labels,
)
from lookin.web import html as h
from lookin.web.views.base import RebuildView
from lookin.web.views.carousel import CarouselDialog
from lookin.web.views.common import (
    confused_pairs_table, overview_image, overview_zoom, saved_path_message, score_diagnostics_banner,
    status_line, uuid_overlap_banner,
)
from lookin.web.views.files import FileBrowserDialog
from lookin.web.views.media import fig_html
from lookin.web.views.slice_panel import SlicePanel
from lookin.web.views.widgets import ClassFocus, class_select, experiment_color_menu

PERCLASS_MANUAL_PAGES = {
    "f1": "f1",
    "precision": "precision-recall",
    "recall": "precision-recall",
    "specificity": "precision-recall",
    "roc_auc": "roc-auc",
    "average_precision": "pr-auc",
    "support": "support-and-imbalance",
}
SUMMARY_HEADERS = [
    "Experiment", "N", "Accuracy", "Balanced Acc", "Macro F1", "Weighted F1",
    "MCC", "Cohen's kappa", "Macro ROC-AUC", "Weighted ROC-AUC",
    "Macro AP", "Weighted AP", "Log loss", "Brier",
    "Brier reliability", "Brier resolution", "Brier uncertainty", "ECE", "MCE",
]


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class MetricsView(RebuildView):
    name = "metrics"
    title = "Metrics"

    def __init__(self, session):
        super().__init__(session)
        self.checked: dict = {}                 # experiment name -> shown
        self._metrics_cache: dict = {}
        self._curves_cache: dict = {}
        self._overlap_cache: dict = {}
        self._overview_cache: dict = {}
        self.metric = "f1"
        self.curve_target = MACRO_AVERAGE
        self.class_focus = ClassFocus(self.name, "focus", cf.FOCUS_CRITERIA, self._on_focus_change,
                                      what="the confusion grids and the per-class chart",
                                      support_criteria=(cf.FOCUS_WORST_F1,))
        self._focus: list = []
        self._many_classes = False
        self._render_inputs = None
        self.slice_panel = SlicePanel(self, session)
        # What the last render pass produced, drawn by render().
        self.out = {"warnings": "", "summary": "", "perclass": "", "curves": "", "confusion": "", "message": ""}
        self.built = False
        # report export
        self.report_dir = ""
        self.report_name = ""
        self.report_per_class = True
        self.report_confusion = True
        self.report_busy = False
        self.report_status = ("", "")

    # ---- experiment picker ----

    def _sync_checks(self) -> None:
        existing = dict(self.checked)
        self.checked = {name: existing.get(name, True) for name in self.state.experiments}
        dataset = self.state.dataset
        options = [MACRO_AVERAGE] + (list(dataset.labels()) if dataset is not None else [])
        if self.curve_target not in options:
            self.curve_target = MACRO_AVERAGE

    def _selected_names(self) -> list:
        return [name for name in self.state.experiments if self.checked.get(name, True)]

    def _picker_html(self) -> str:
        names = list(self.state.experiments)
        total = len(names)
        selected = len(self._selected_names())
        summary = (f"{selected} of {total} experiment(s) shown -- the arrows set the order used by "
                   f"every table, legend and chart below, a dot its color.")
        rows = []
        for position, name in enumerate(names):
            color = color_for_experiment(name, self.state.color_slots)
            rows.append(
                f'<div class="mx-row" data-key="mx-{h.e(name)}">'
                f'{h.button(icon_name="chev-up", act_attr=self.a("move", name=name, delta=-1), small=True, kind="ghost", title="Move up", disabled=position == 0)}'
                f'{h.button(icon_name="chev-down", act_attr=self.a("move", name=name, delta=1), small=True, kind="ghost", title="Move down", disabled=position == total - 1)}'
                f'{experiment_color_menu(self.target, name, color, name in self.state.experiment_colors)}'
                f'{h.checkbox(name, self.checked.get(name, True), change=self.c("check", name=name), cls="grow", ellipsis=True)}'
                f'</div>')
        head = (f'<div class="row"><span class="muted small grow">{h.e(summary)}</span>'
                f'{h.button("Select all", act_attr=self.a("select_all"), small=True, kind="ghost", icon_name="check")}'
                f'{h.button("Select none", act_attr=self.a("select_none"), small=True, kind="ghost", icon_name="x")}'
                f'{h.button("Sort A-Z", act_attr=self.a("sort_az"), small=True, kind="ghost", icon_name="list")}</div>')
        return f'<section class="card"><div class="card-b">{head}<div class="mx-list">{"".join(rows)}</div></div></section>'

    def _apply_order(self, names: list) -> None:
        self.state.reorder_experiments(names)
        self._redraw()

    def on_color(self, name: str = "", value: str = "", **_):
        if self.state.set_experiment_color(name, value):
            self.s.notify_colors()

    def on_colors_changed(self) -> None:
        if self.built:
            self._redraw()

    def on_check(self, name: str = "", value=False, **_):
        if name in self.state.experiments:
            self.checked[name] = bool(value)
            self._redraw()

    def on_move(self, name: str = "", delta: int = 0, **_):
        names = list(self.state.experiments)
        if name not in names:
            return
        old = names.index(name)
        new = old + int(delta)
        if 0 <= new < len(names):
            names.insert(new, names.pop(old))
            self._apply_order(names)

    def on_select_all(self, **_):
        self.checked = {n: True for n in self.state.experiments}
        self._redraw()

    def on_select_none(self, **_):
        self.checked = {n: False for n in self.state.experiments}
        self._redraw()

    def on_sort_az(self, **_):
        self._apply_order(sorted(self.state.experiments))

    def on_metric(self, value: str = "", **_):
        if value in METRIC_OPTIONS:
            self.metric = value
            self._redraw()

    def on_curve_target(self, value: str = "", **_):
        dataset = self.state.dataset
        known = [MACRO_AVERAGE] + ([str(l) for l in dataset.labels()] if dataset is not None else [])
        if str(value) in known:
            self.curve_target = value
            self._redraw()

    def _redraw(self) -> None:
        """A UI change: draw again from the caches (computing only what is
        not cached yet -- a newly ticked experiment), off the request
        thread. The stale banner stays as it is."""
        self.rebuild(self._render_pass, clear_stale=False)

    # ---- computation ----

    def _get_merged(self, merged_cache: dict, name: str):
        if name not in merged_cache:
            merged_cache[name] = self.state.merged(name)
        return merged_cache[name]

    def _get_metrics(self, name: str, merged_cache: dict):
        if name not in self._metrics_cache:
            state = self.state
            exp = state.experiments[name]
            merged = self._get_merged(merged_cache, name)
            _log(f"computing metrics for '{name}' ({len(merged):,} merged rows)...")
            t0 = time.monotonic()
            self._metrics_cache[name] = compute_metrics(name, merged, state.dataset.columns.label_col,
                                                        exp.score_cols, exp.class_labels)
            _log(f"metrics for '{name}' done in {time.monotonic() - t0:.1f}s")
        return self._metrics_cache[name]

    def _get_curves(self, name: str, target: str, merged_cache: dict):
        key = (name, target)
        if key not in self._curves_cache:
            state = self.state
            exp = state.experiments[name]
            merged = self._get_merged(merged_cache, name)
            _log(f"computing ROC/PR curves for '{name}' / target={target!r} ({len(merged):,} merged rows)...")
            t0 = time.monotonic()
            self._curves_cache[key] = compute_curves(merged, state.dataset.columns.label_col, exp.score_cols,
                                                     exp.class_labels, target)
            _log(f"curves for '{name}' / {target!r} done in {time.monotonic() - t0:.1f}s")
        return self._curves_cache[key]

    def _clear_out(self, message: str = "") -> None:
        self.out = {"warnings": "", "summary": "", "perclass": "", "curves": "", "confusion": "", "message": message}
        self.class_focus.hide()

    def _render_pass(self) -> None:
        state = self.state
        self._sync_checks()
        if state.dataset is None or not state.experiments:
            self._clear_out("Load a dataset and at least one experiment (results CSV) in the 'Data' tab first.")
            self.slice_panel.set_experiments([])
            return
        selected = self._selected_names()
        if not selected:
            self._clear_out("Select at least one experiment above to see metrics.")
            self.slice_panel.set_experiments([])
            return
        warnings = []
        for name in selected:
            if name not in self._overlap_cache:
                self._overlap_cache[name] = state.uuid_overlap(name)
            warnings.append(uuid_overlap_banner(self._overlap_cache[name], "dataset", name))
        try:
            merged_cache: dict = {}
            metrics_by_name = {}
            for i, name in enumerate(selected):
                if name not in self._metrics_cache:
                    self.set_progress(f"Computing metrics -- {name} ({i + 1}/{len(selected)})")
                metrics_by_name[name] = self._get_metrics(name, merged_cache)
            all_names = list(state.experiments.keys())
            for name, m in metrics_by_name.items():
                warnings.append(score_diagnostics_banner(m.score_diagnostics, name))
            out = {"warnings": "".join(warnings), "message": ""}
            self.set_progress("Building the summary table...")
            out["summary"] = self._summary_html(metrics_by_name, all_names)
            self._render_inputs = (metrics_by_name, all_names)
            self._resolve_focus(metrics_by_name)
            self.set_progress("Drawing per-class metrics...")
            out["perclass"] = self._perclass_html(metrics_by_name, all_names)
            self.set_progress(f"Computing ROC / PR curves ({len(selected)} experiment(s))...")
            out["curves"] = self._curves_html(selected, all_names, merged_cache)
            self.set_progress("Drawing confusion matrices...")
            out["confusion"] = self._confusion_html(metrics_by_name, all_names)
            self.out = out
            self.slice_panel.set_experiments(selected)
        except Exception as ex:  # noqa: BLE001
            _log(f"error computing metrics: {ex}")
            traceback.print_exc()
            self.out = {"warnings": "".join(warnings), "summary": "", "perclass": "", "curves": "",
                        "confusion": "", "message": ""}
            self.out["perclass"] = h.banner(f"Error computing metrics: {ex}", "error")
            self.class_focus.hide()

    def _summary_html(self, metrics_by_name, all_names) -> str:
        head = "".join(f"<th{' class=num' if i else ''}>{h.e(c)}</th>" for i, c in enumerate(SUMMARY_HEADERS))
        rows = []
        for name, m in metrics_by_name.items():
            values = [m.accuracy, m.balanced_accuracy, m.macro_f1, m.weighted_f1, m.mcc, m.cohen_kappa,
                      m.macro_roc_auc, m.weighted_roc_auc, m.macro_avg_precision, m.weighted_avg_precision,
                      m.log_loss, m.brier_score, m.brier_reliability, m.brier_resolution, m.brier_uncertainty,
                      m.ece, m.mce]
            cells = "".join(f"<td class=num>{f'{v:.3f}' if v == v else '—'}</td>" for v in values)
            rows.append(f'<tr><td><span class="row-nw">{h.dot(color_for_experiment(name, self.state.color_slots), size=10)}'
                        f'{h.name(name)}</span></td><td class=num>{m.n_samples:,}</td>{cells}</tr>')
        table = (f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{head}</tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
        return h.section("Summary", table, manual="summary-table")

    def _perclass_html(self, metrics_by_name, all_names) -> str:
        metric_key = self.metric
        per_class_by_name = {name: m.per_class.set_index("label") for name, m in metrics_by_name.items()}
        labels_union = sorted(set().union(*[set(pc.index) for pc in per_class_by_name.values()]))
        n_all_classes = len(labels_union)
        if self._many_classes:
            labels_union = [l for l in self._focus if l in set(labels_union)]
        n_exp = len(metrics_by_name)
        fig_width = max(6.0, min(18.0, len(labels_union) * (0.5 + 0.13 * n_exp)))
        legend_cols = legend_columns(n_exp)
        fig, ax = subplots(figsize=(fig_width, 4.3 + legend_height_inches(n_exp, legend_cols)))
        width = 0.8 / max(n_exp, 1)
        x = np.arange(len(labels_union))
        for i, (name, m) in enumerate(metrics_by_name.items()):
            per_class = per_class_by_name[name]
            values = [per_class[metric_key].get(l, np.nan) for l in labels_union]
            offset = (i - (n_exp - 1) / 2) * width
            ax.bar(x + offset, values, width, label=short_name(name), color=color_for_experiment(name, self.state.color_slots))
        ax.set_xticks(x)
        tick_labels, uneven_support = _class_tick_labels(labels_union, per_class_by_name)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_ylabel(metric_key)
        if metric_key in COUNT_METRICS:
            ax.set_ylabel(f"{metric_key} (rows)")
        else:
            ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-class {metric_key} by experiment")
        style_axes(ax, x=GRID_NONE)
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
                f"(the uuid overlap with the dataset differs by experiment); the count shown is {first_name}'s.")
        img = fig_html(self.s, fig, f"Per-class {metric_key} by experiment")
        return img + "".join(f'<div class="muted small"><i>{h.e(n)}</i></div>' for n in notes)

    def _curves_html(self, selected, all_names, merged_cache) -> str:
        target = self.curve_target or MACRO_AVERAGE
        curves_by_name = {}
        for i, name in enumerate(selected):
            if (name, target) not in self._curves_cache:
                self.set_progress(f"Computing ROC / PR curves -- {name} ({i + 1}/{len(selected)})")
            curves_by_name[name] = self._get_curves(name, target, merged_cache)
        drawable = {n: c for n, c in curves_by_name.items() if c.disabled_reason is None}
        if not drawable:
            reason = next(iter(curves_by_name.values())).disabled_reason
            return h.banner(
                f"ROC and Precision-Recall curves are disabled for {target}: {reason}. "
                f"The decision-level metrics for this class (precision, recall, F1, and its "
                f"confusion matrix row) are unaffected and shown above.", "info", icon_name="list")
        notes = []
        skipped = [n for n in curves_by_name if n not in drawable]
        if skipped:
            notes.append(f"Not shown: {', '.join(skipped)} ({curves_by_name[skipped[0]].disabled_reason}).")
        excluded = sorted({l for c in drawable.values() for l in c.excluded_labels})
        if excluded:
            notes.append(f"Macro-average taken over the classes whose score column varies; "
                         f"excluded: {', '.join(excluded)}.")
        n_curves = len(drawable)
        legend_cols = min(2, n_curves)
        fig, (ax1, ax2) = subplots(1, 2, figsize=(11, 4.4 + legend_height_inches(n_curves, legend_cols)))
        ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        for name, curve in drawable.items():
            color = color_for_experiment(name, self.state.color_slots)
            ax1.plot(curve.fpr, curve.tpr, color=color, label=f"{short_name(name)} (AUC={curve.roc_auc:.3f})")
            ax2.plot(curve.recall, curve.precision, color=color, label=f"{short_name(name)} (AP={curve.average_precision:.3f})")
        ax1.set_xlabel("False positive rate")
        ax1.set_ylabel("True positive rate")
        ax1.set_title(f"ROC curve — {target}")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision-Recall curve — {target}")
        style_axes(ax1)
        style_axes(ax2)
        reserve_legend_space(fig, n_curves, legend_cols)
        for ax in (ax1, ax2):
            legend_below(ax, ncol=legend_cols, fontsize=7)
        return (fig_html(self.s, fig, f"ROC & Precision-Recall — {target}")
                + "".join(f'<div class="muted small"><i>{h.e(n)}</i></div>' for n in notes))

    # ---- focus set ----

    def _resolve_focus(self, metrics_by_name) -> None:
        confusions = [m.confusion for m in metrics_by_name.values()]
        per_class = [m.per_class for m in metrics_by_name.values()]
        self._focus = self.class_focus.resolve(
            cf.all_labels(confusions),
            lambda criterion: cf.rank_classes(criterion, confusions, per_class,
                                              min_support=self.class_focus.min_support_for(criterion)))
        self._many_classes = self.class_focus.many

    def _on_focus_change(self) -> None:
        if self._render_inputs is None:
            return

        def work():
            metrics_by_name, all_names = self._render_inputs
            self._resolve_focus(metrics_by_name)
            self.out = {**self.out, "perclass": self._perclass_html(metrics_by_name, all_names),
                        "confusion": self._confusion_html(metrics_by_name, all_names)}
        self.rebuild(work, clear_stale=False)

    def on_focus(self, op: str = "", **args):
        self.class_focus.handle(op, args)

    # ---- confusion grids ----

    def _confusion_html(self, metrics_by_name, all_names) -> str:
        return "".join(self._grid_html(name, m, all_names, self._focus, self._many_classes)
                       for name, m in metrics_by_name.items())

    def _grid_html(self, name, m, all_names, focus: list, many: bool) -> str:
        cm = m.confusion
        if many:
            grid, other = cf.focus_matrix(cm, focus)
        else:
            grid, other = cm, None
        row_labels = list(grid.index)
        col_labels = list(grid.columns)
        counts = grid.to_numpy()
        max_val = int(counts.max()) if counts.size else 1
        other_counts = other.to_numpy() if other is not None else None
        n_columns = len(col_labels) + (1 if many else 0)
        cell = matrix_cell_size(n_columns, CELL_SIZE)
        count_size = matrix_text_size(cell, CELL_SIZE, 10)
        header_size = matrix_text_size(cell, CELL_SIZE, 9, cap=14)
        header_chars = matrix_label_chars(cell, CELL_SIZE, 4)
        row_label_size = matrix_text_size(cell, CELL_SIZE, 10, cap=14)
        short_labels, dropped_prefix = short_column_labels(col_labels, 2 * header_chars)
        cols_css = f"{LABEL_COL_WIDTH}px repeat({n_columns}, {cell}px)"
        cells = ['<span></span>']
        for short, pl in zip(short_labels, col_labels):
            cells.append(f'<span class="cm-h" style="font-size:{header_size}px" title="{h.e(pl)}">{h.e(short)}</span>')
        if many:
            cells.append(f'<span class="cm-h" style="font-size:{header_size}px" title="Predicted as any class not shown in this grid"><i>other</i></span>')
        for i, tl in enumerate(row_labels):
            cells.append(f'<span class="cm-r" style="font-size:{row_label_size}px" title="{h.e(tl)}">{h.e(tl)}</span>')
            for j, pl in enumerate(col_labels):
                count = int(counts[i, j])
                fraction = count / max_val if max_val else 0.0
                bg = blues_color(fraction)
                fg = "#ffffff" if fraction > 0.55 else "#111111"
                cells.append(
                    f'<button type="button" class="cm-c" style="height:{cell}px;background:{bg};color:{fg};font-size:{count_size}px" '
                    f'title="true = {h.e(tl)}&#10;predicted = {h.e(pl)}&#10;n = {count}"'
                    f'{self.a("cell", experiment=name, true=str(tl), pred=str(pl))}>{count}</button>')
            if many:
                count = int(other_counts[i])
                act = self.a("other", experiment=name, true=str(tl)) if count else ""
                cells.append(
                    f'<button type="button" class="cm-c cm-o" style="height:{cell}px;font-size:{count_size}px"'
                    f' title="true = {h.e(tl)}&#10;predicted = any class not in this grid&#10;n = {count}"'
                    f'{act}{"" if count else " disabled"}><i>{count}</i></button>')
        grid_html = f'<div class="cm-grid" style="grid-template-columns:{cols_css}">{"".join(cells)}</div>'
        color = color_for_experiment(name, self.state.color_slots)
        pairs = cf.top_confused_pairs(cm)
        pairs_table = confused_pairs_table(
            pairs.head(PAIRS_SHOWN), len(pairs),
            act_for=lambda tl, pl: self.a("cell", experiment=name, true=str(tl), pred=str(pl)))
        extras = [f'<div class="cm-pairs">{pairs_table}</div>']
        if many:
            extras.append(
                f'<div><div class="small"><b>All {len(cm.index):,} classes -- click to enlarge and zoom</b></div>'
                f'{overview_image(self.s, cm, f"{name} -- full confusion matrix", OVERVIEW_SIZE, self._overview_cache, self.a("overview", experiment=name))}</div>')
        hint = ("Click a cell or a pair to preview those images right here"
                + (f" -- column headers leave out the shared prefix '{dropped_prefix}'" if dropped_prefix else ""))
        return (f'<section class="card cm-card" data-key="cm-{h.e(name)}"><div class="card-h">{h.dot(color, size=10)}'
                f'<h2 class="ell">{h.name(name)}<span class="nowrap">— accuracy {m.accuracy:.3f}</span></h2></div><div class="card-b">'
                f'<div class="cm-scroll">{grid_html}</div>'
                f'<div class="muted small"><i>{h.e(hint)}</i></div>'
                f'<div class="row-wrap" style="align-items:flex-start;gap:16px">{"".join(extras)}</div></div></section>')

    def on_cell(self, experiment: str = "", true="", pred="", **_):
        state = self.state
        if experiment not in state.experiments:
            return
        merged = state.merged(experiment)
        cols = state.dataset.columns
        true_label = self._label_value(merged[cols.label_col], true)
        pred_label = self._label_value(merged["predicted_label"], pred)
        subset = merged[(merged[cols.label_col] == true_label) & (merged["predicted_label"] == pred_label)]
        self._open_confusion_carousel(experiment, subset, f"{true_label} -> {pred_label}",
                                      gallery_nav=(experiment, true_label, pred_label))

    def on_other(self, experiment: str = "", true="", **_):
        state = self.state
        if experiment not in state.experiments:
            return
        merged = state.merged(experiment)
        cols = state.dataset.columns
        true_label = self._label_value(merged[cols.label_col], true)
        subset = merged[(merged[cols.label_col] == true_label) & ~merged["predicted_label"].isin(self._focus)]
        self._open_confusion_carousel(experiment, subset, f"{true_label} -> other classes", gallery_nav=None)

    @staticmethod
    def _label_value(series, text):
        """The label as the frame holds it (a CSV label can be read back as
        a number): the action carries its text."""
        for value in series.dropna().unique():
            if str(value) == str(text):
                return value
        return text

    def on_overview(self, experiment: str = "", **_):
        m = self._metrics_cache.get(experiment)
        if m is not None:
            overview_zoom(self.s, m.confusion, self._focus if self._many_classes else [],
                          f"{experiment} -- full confusion matrix")

    def _open_confusion_carousel(self, experiment_name: str, subset, what: str, gallery_nav) -> None:
        state = self.state
        cols = state.dataset.columns
        header = ""
        dialog = CarouselDialog(
            self.s, f"{what}  ·  {experiment_name}  ·  {len(subset):,} image(s)", state.dataset, len(subset),
            lambda start, end: subset.iloc[start:end].to_dict("records"), cols.crop_col, cols.uuid_col,
            review_source="confusion_matrix",
            suggestions={str(u): str(p) for u, p in zip(subset[cols.uuid_col], subset["predicted_label"])},
            label_col=cols.label_col)
        if gallery_nav is not None:
            dialog.gallery_nav = gallery_nav
            dialog.footer = lambda: (h.button("Open in Gallery", act_attr=h.act(self.name, "open_gallery", dialog=dialog.id),
                                              icon_name="grid") + h.button("Close", act_attr=dialog.a("close")))
        self.s.open_dialog(dialog)

    def on_open_gallery(self, dialog: str = "", **_):
        d = self.s.dialog(dialog)
        nav = getattr(d, "gallery_nav", None) if d is not None else None
        if d is not None:
            d.close()
        if nav:
            gallery = self.s.view("gallery")
            self.s.goto("gallery")
            gallery.apply_external_filter(*nav)

    # ---- rendering ----

    def render(self) -> str:
        state = self.state
        self.on_shown()
        parts = [self.stale_html()]
        if state.dataset is not None and state.experiments:
            self._sync_checks()
            parts.append(self._picker_html())
        out = self.out
        if out.get("message"):
            parts.append(h.banner(out["message"], "info"))
        parts.append(out.get("warnings", ""))
        parts.append(self.progress_html())
        parts.append(out.get("summary", ""))
        parts.append(self.class_focus.render())
        if out.get("confusion"):
            parts.append(f'<div class="sec-t"><h2>Confusion matrices</h2>{h.info_btn("confusion-matrix")}</div>'
                         f'<div class="cm-area">{out["confusion"]}</div>')
        if out.get("perclass"):
            metric_sel = h.select([(m, m) for m in METRIC_OPTIONS], self.metric, change=self.c("metric"),
                                  label="Per-class metric")
            parts.append(h.section("Per-class metric comparison", f'<div class="row">{metric_sel}</div>{out["perclass"]}',
                                   manual=PERCLASS_MANUAL_PAGES.get(self.metric, "f1")))
        if out.get("curves"):
            labels = [MACRO_AVERAGE] + ([str(l) for l in state.dataset.labels()] if state.dataset is not None else [])
            target_sel = class_select(labels, self.curve_target, self.c("curve_target"), label="Class for ROC / PR curves")
            parts.append(h.section("ROC & Precision-Recall curves", f'<div class="row">{target_sel}</div>{out["curves"]}',
                                   manual="roc-auc"))
        if state.dataset is not None and state.experiments and self._selected_names():
            parts.append(self.slice_panel.render())
        if state.dataset is not None:
            parts.append(self._report_html())
        head = (f'<div class="sec-t"><h2>Compare experiments</h2>{h.info_btn("metric-levels")}</div>')
        return f'<div class="stack">{head}{"".join(p for p in parts if p)}</div>'

    # ---- report export ----

    def _prepare_report_fields(self) -> None:
        state = self.state
        if not self.report_name and state.dataset is not None:
            self.report_name = default_report_name(state)
        if not self.report_dir and state.dataset is not None and state.dataset.csv_path:
            self.report_dir = str(Path(state.dataset.csv_path).parent)

    def _report_html(self) -> str:
        self._prepare_report_fields()
        text, kind = self.report_status
        body = (f'<div class="row">{h.text_input(self.report_dir, label="Destination folder", change=self.c("report_dir"), cls="grow", mono=True, extra=" data-force=1")}'
                f'{h.button(icon_name="folder", act_attr=self.a("report_browse"), title="Browse")}</div>'
                f'<div class="row-wrap">{h.text_input(self.report_name, label="File name", change=self.c("report_name"), width=300, extra=" data-force=1")}'
                f'{h.checkbox("Per-class tables", self.report_per_class, change=self.c("report_opt", which="per_class"))}'
                f'{h.checkbox("Confusion matrices", self.report_confusion, change=self.c("report_opt", which="confusion"))}</div>'
                f'<div class="row">{h.button("Export HTML report", act_attr=self.a("export", kind="html"), kind="primary", icon_name="file", disabled=self.report_busy)}'
                f'{h.button("Export PowerPoint", act_attr=self.a("export", kind="pptx"), kind="primary", icon_name="layers", disabled=self.report_busy)}</div>'
                f'{h.spinner("Building the report...") if self.report_busy else ""}'
                f'{status_line(text, kind) if text else ""}')
        return h.section("Export a report", body, manual="report-export",
                         sub="one self-contained HTML file, or a native PowerPoint deck")

    def on_report_dir(self, value: str = "", **_):
        self.report_dir = str(value or "")

    def on_report_name(self, value: str = "", **_):
        self.report_name = str(value or "")

    def on_report_opt(self, which: str = "", value=False, **_):
        if which == "per_class":
            self.report_per_class = bool(value)
        elif which == "confusion":
            self.report_confusion = bool(value)

    def on_report_browse(self, **_):
        def picked(path):
            self.report_dir = path
            self.refresh()
        self.s.open_dialog(FileBrowserDialog(picked, start_path=self.report_dir, dirs_only=True,
                                             title="Where to write the report"))

    def on_export(self, kind: str = "html", **_):
        report_cls, extension = (PptxReport, "pptx") if kind == "pptx" else (Report, "html")
        selected = self._selected_names()
        if not selected:
            self.report_status = ("Select at least one experiment above to report on.", "error")
            return
        self._prepare_report_fields()
        directory = (self.report_dir or "").strip()
        if not directory:
            self.report_status = ("Pick a destination folder.", "error")
            return
        self.report_busy = True
        self.report_status = ("", "")

        def failed(ex):
            self.report_busy = False
            self.report_status = (f"Export failed: {ex}", "error")
            self.refresh()

        self.s.job("Report").run(lambda: self._write_report(selected, directory, report_cls, extension), on_error=failed)

    def _report_file_name(self, extension: str) -> str:
        typed = (self.report_name or "").strip()
        if not typed:
            return default_report_name(self.state, extension)
        for known in (".html", ".pptx"):
            if typed.lower().endswith(known):
                typed = typed[:-len(known)]
                break
        return f"{typed}.{extension}"

    def _write_report(self, selected: list, directory: str, report_cls: type, extension: str) -> None:
        try:
            name = self._report_file_name(extension)
            pair = tuple(selected[:2]) if len(selected) == 2 else None
            report = build_report(
                self.state, selected,
                include_per_class=bool(self.report_per_class), include_confusion=bool(self.report_confusion),
                comparison_pair=pair, snapshot_id=self._bookmark_for_report(name), report_cls=report_cls,
                focus=self._focus if self._many_classes else None)
            dest = report.write(Path(directory) / name)
            size_kb = dest.stat().st_size / 1024
            _log(f"report: wrote {dest} ({size_kb:,.0f} KB)")
            self.report_status = (f"Wrote {saved_path_message(self.s, dest)} ({size_kb:,.0f} KB) -- self-contained, "
                                  f"opens with no network.", "success")
        finally:
            self.report_busy = False
            self.refresh()

    def _bookmark_for_report(self, report_name: str) -> str:
        workspace = getattr(self.s, "current_workspace", None)
        if workspace is None or not workspace.exists():
            return ""
        try:
            snapshot = library.create_snapshot(
                workspace, self.s.view_state.capture(), name=f"Report: {report_name}",
                description="Saved automatically when this report was exported.")
        except Exception as ex:  # noqa: BLE001
            _log(f"report: could not bookmark the view ({ex})")
            return ""
        return snapshot.id if snapshot else ""

    # ---- lifecycle ----

    def on_shown(self) -> None:
        if not self.built and self.state.dataset is not None and self.state.experiments:
            self.built = True
            self.rebuild(self._refresh_now)

    def can_update(self) -> bool:
        return self.built and self.state.dataset is not None and bool(self.state.experiments)

    def on_update(self, **_):
        self._metrics_cache.clear()
        self._curves_cache.clear()
        self._overlap_cache.clear()
        self._overview_cache.clear()
        self.slice_panel.invalidate()
        self.built = True
        self.set_progress("Computing metrics...")
        self.rebuild(self._refresh_now)

    def _refresh_now(self) -> None:
        self._sync_checks()
        self._render_pass()

    def recompute_after_restore(self) -> None:
        if self.state.dataset is None or not self._selected_names():
            return
        self.on_update()

    def on_slice(self, op: str = "", **args):
        self.slice_panel.handle(op, args)

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if old_name in self._metrics_cache:
            metrics = self._metrics_cache.pop(old_name)
            metrics.name = new_name
            self._metrics_cache[new_name] = metrics
        if old_name in self._overlap_cache:
            self._overlap_cache[new_name] = self._overlap_cache.pop(old_name)
        for key in [k for k in self._curves_cache if k[0] == old_name]:
            self._curves_cache[(new_name, key[1])] = self._curves_cache.pop(key)
        self.slice_panel.rename_experiment(old_name, new_name)
        if old_name in self.checked:
            self.checked[new_name] = self.checked.pop(old_name)
        self._sync_checks()
        if not self.dirty and self.built:
            self.rebuild(self._render_pass, clear_stale=False)
        self.refresh()

    # ---- snapshots (see ui/view_state.py) ----

    def view_state(self) -> dict:
        return {
            "metric": self.metric or "",
            "curve_target": self.curve_target or MACRO_AVERAGE,
            "selected": self._selected_names(),
            "order": list(self.state.experiments),
            "focus": self.class_focus.view_state(),
        }

    def restore_view_state(self, state: dict) -> list:
        warnings = []
        loaded = set(self.state.experiments)
        order = [n for n in state.get("order", []) or [] if n in loaded]
        if order:
            self.state.reorder_experiments(order)
        self._sync_checks()
        selected = state.get("selected")
        if isinstance(selected, list):
            wanted = {n for n in selected if isinstance(n, str)}
            gone = sorted(wanted - loaded)
            if gone:
                warnings.append(f"Metrics: experiment(s) {', '.join(gone)} are not loaded.")
            self.checked = {name: name in wanted for name in self.state.experiments}
        metric = state.get("metric")
        if metric in METRIC_OPTIONS:
            self.metric = metric
        target = state.get("curve_target")
        if isinstance(target, str) and target:
            dataset = self.state.dataset
            known = [MACRO_AVERAGE] + (list(dataset.labels()) if dataset is not None else [])
            if target in known:
                self.curve_target = target
            else:
                warnings.append(f"Metrics: class '{target}' is not in this dataset -- curves show the macro average.")
        self.class_focus.restore_view_state(state.get("focus"))
        self.built = True
        self.refresh()
        return warnings
