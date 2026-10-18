"""The Metrics tab's per-slice breakdown: one metadata column, one row per
value, the same accuracy every experiment reports globally.

Lives in its own module rather than inside metrics_view.py because it has
its own cost and its own lifecycle. Nothing is computed until a column is
picked, and picking one does not recompute the confusion matrices, the
curves or anything else the Metrics tab holds -- the slice pass is a set of
boolean masks over a frame the tab has already merged, and it is the only
work the dropdown triggers.

Reading the table
-----------------
Slices are ranked worst-first, because "where does it fail" is the question
the panel exists for -- but a slice below the row floor is pushed to the
bottom and greyed rather than allowed to lead the ranking (see
core/slices.sort_rows). Each accuracy carries its Wilson interval, so a
twelve-point gap between two slices of forty rows reads as what it is.

Reading the chart
-----------------
Two series on two axes, because they answer each other: grey bars are how
many crops the slice holds, and the marked lines are the metric, one per
experiment. Keeping the counts visible next to the result is what stops a
0.42 being read as a finding when it rests on nine crops -- the same job the
Wilson interval does numerically, done at a glance.

Which metric
------------
Accuracy is the default and the plainest. The rest of the list is there
because "does it work here" is rarely the real question: the one that gets
asked is "does class A still get raised on nothing here", and that is a
false alarm rate on one named class, per slice. So every metric that
depends on a class comes with a class picker beside it, and the counts it
is derived from carry the same Wilson interval accuracy does -- see
core/slices.metric_value.

Clicking a slice opens its errors in the shared carousel, so the answer to
"why is this sensor at 61%" is three clicks away from noticing it, and the
crops are relabellable from there like anywhere else in the app.
"""
from __future__ import annotations

import traceback

import flet as ft
import numpy as np

from lookin.core.chart_style import subplots
from lookin.core import slices as sl
from lookin.ui.carousel import Carousel
from lookin.ui.class_matrix import make_searchable
from lookin.ui.components import (
    GRID_MAJOR, GRID_NONE, BackgroundRebuild, card, color_for_experiment,
    fig_to_zoomable_image, info_banner, legend_below, legend_columns, legend_height_inches,
    reserve_legend_space, section_title, style_axes,
)

# The metric keys themselves live in core/slices.py, beside the code that
# computes them; re-exported here because this is where the rest of the UI
# has always asked for them.
METRIC_ACCURACY = sl.METRIC_ACCURACY
METRIC_MACRO_F1 = sl.METRIC_MACRO_F1
METRIC_MEAN_CONFIDENCE = sl.METRIC_MEAN_CONFIDENCE

# Metrics offered per slice, and the labels beside them (keys first, labels
# second -- see CLAUDE.md). Accuracy first and by default: it is a plain
# proportion, so its confidence interval means something, which is what
# makes a small slice legible instead of alarming. The per-class half of the
# list is spelled "... (one class)" so the class picker that appears beside
# it does not look like it came from nowhere.
SLICE_METRICS = (
    (METRIC_ACCURACY, "Accuracy"),
    (METRIC_MACRO_F1, "Macro F1"),
    (METRIC_MEAN_CONFIDENCE, "Mean confidence"),
    (sl.METRIC_RECALL, "Recall (one class)"),
    (sl.METRIC_PRECISION, "Precision (one class)"),
    (sl.METRIC_F1, "F1 (one class)"),
    (sl.METRIC_FALSE_ALARM, "False alarm rate (one class)"),
    (sl.METRIC_MISS_RATE, "Miss rate (one class)"),
    (sl.METRIC_SUPPORT, "Support (one class)"),
    (sl.METRIC_PREDICTED, "Predicted as (one class)"),
)
METRIC_LABELS = dict(SLICE_METRICS)

SORT_LABELS = (
    (sl.SORT_WORST, "Worst first"),
    (sl.SORT_VALUE, "By value"),
    (sl.SORT_SIZE, "By size"),
)

BINNING_LABELS = (
    (sl.BINNING_AUTO, "Automatic"),
    (sl.BINNING_VALUES, "One bar per value"),
    (sl.BINNING_QUANTILE, "Quantile bins"),
)

CHART_MAX_WIDTH = 1200
CHART_PX_PER_INCH = 90
# Slices drawn in the chart. A hundred-value categorical column is
# legitimate in the table (it scrolls) but unreadable as a chart.
MAX_CHART_SLICES = 30
# The count bars are background, not a series: a neutral grey that no
# experiment color can be confused with, and an axis labelled in the same
# grey so the eye reads it as the context it is.
COUNT_BAR_COLOR = "#ccd1d9"
COUNT_AXIS_COLOR = "#7b828d"


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value == value else "—"


def _counts_agree(rows_by_name: dict, names: list, values: list) -> bool:
    """Whether every experiment holds the same number of crops in every
    slice, which is the normal case -- n comes from the dataset, not from
    the model. They only diverge when the experiments join different
    subsets of it (partial uuid overlap)."""
    if not values:
        return True
    return all(len({rows_by_name[name][v].n for name in names}) == 1 for v in values)


class SlicePanel:
    def __init__(self, ctx):
        self.ctx = ctx
        self._selected_names: list = []
        # (column, bins, binning) -> SliceSpec, and the same plus the
        # experiment -> SliceReport. Keyed by how the column is cut as well
        # as by the column: changing that is a different question, not a
        # refinement of the same one.
        self._spec_cache: dict = {}
        self._report_cache: dict = {}

        self.column_dropdown = ft.Dropdown(
            label="Break down by column...", width=280, border_radius=8,
            on_change=self._on_column_change,
        )
        self.metric_dropdown = ft.Dropdown(
            label="Metric", value=METRIC_ACCURACY, width=230, border_radius=8,
            options=[ft.dropdown.Option(key, label) for key, label in SLICE_METRICS],
            on_change=self._on_display_change,
        )
        # Only shown for a metric that needs it (see _sync_controls): a
        # class picker beside "Accuracy" would suggest the accuracy shown is
        # that class's, which it is not.
        self.class_dropdown = ft.Dropdown(
            label="Class", width=190, border_radius=8, visible=False,
            on_change=self._on_display_change,
        )
        self.sort_dropdown = ft.Dropdown(
            label="Order", value=sl.SORT_WORST, width=160, border_radius=8,
            options=[ft.dropdown.Option(key, label) for key, label in SORT_LABELS],
            on_change=self._on_display_change,
        )
        self.binning_dropdown = ft.Dropdown(
            label="Binning", value=sl.BINNING_AUTO, width=190, border_radius=8,
            options=[ft.dropdown.Option(key, label) for key, label in BINNING_LABELS],
            on_change=self._on_column_change,
            tooltip="Automatic gives one bar per value to a column of few distinct values, "
                    "and quantile bins to a continuous one",
        )
        self.bins_field = ft.TextField(
            label="Bins", value=str(sl.DEFAULT_BINS), width=90, border_radius=8,
            on_submit=self._on_column_change, tooltip="Quantile bins for a numeric column",
        )
        self.min_rows_field = ft.TextField(
            label="Min rows", value=str(sl.DEFAULT_MIN_ROWS), width=110, border_radius=8,
            on_submit=self._on_display_change,
            tooltip="Slices smaller than this are shown but never ranked first",
        )
        self.progress = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2),
             ft.Text("Computing slices...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)],
            spacing=10, visible=False,
        )
        self.body = ft.Column(spacing=12)
        self.rebuild_job = BackgroundRebuild(ctx, "Slices")

        self.carousel_title = ft.Text(weight=ft.FontWeight.BOLD)
        self.carousel_content = ft.Container()
        self.carousel_dialog = ft.AlertDialog(
            modal=False, title=self.carousel_title, content=self.carousel_content,
            on_dismiss=self._close_carousel,
            actions=[ft.TextButton("Close", on_click=self._close_carousel)],
        )

    # ---- lifecycle ------------------------------------------------------
    def set_experiments(self, names: list) -> None:
        """Called from the Metrics tab's render pass (already off the UI
        thread) with the experiments it is showing. Renders from cache when
        a column is already picked; does nothing at all when none is."""
        self._selected_names = list(names)
        self._rebuild_column_options()
        self._rebuild_class_options()
        if self.column_dropdown.value:
            self._render()

    def invalidate(self) -> None:
        """The data changed -- drop every cached slice. Called from the
        Metrics tab's Update, so the two are never out of step."""
        self._spec_cache.clear()
        self._report_cache.clear()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow a rename without recomputing: the slices are the same
        rows under a different name (see AppContext.notify_rename)."""
        for key in [k for k in self._report_cache if k[-1] == old_name]:
            report = self._report_cache.pop(key)
            report.experiment = new_name
            self._report_cache[(*key[:-1], new_name)] = report
        self._selected_names = [new_name if n == old_name else n for n in self._selected_names]

    def _rebuild_column_options(self) -> None:
        columns = self._sliceable_columns()
        self.column_dropdown.options = [ft.dropdown.Option(c) for c in columns]
        if self.column_dropdown.value not in columns:
            self.column_dropdown.value = None

    def _rebuild_class_options(self) -> None:
        """The classes a per-class metric can be measured on -- the dataset's
        own labels, never translated or reordered (see CLAUDE.md)."""
        dataset = self.ctx.state.dataset
        labels = [str(l) for l in dataset.labels()] if dataset is not None else []
        self.class_dropdown.options = [ft.dropdown.Option(l) for l in labels]
        make_searchable(self.class_dropdown)
        if self.class_dropdown.value not in labels:
            self.class_dropdown.value = labels[0] if labels else None

    def _sync_controls(self) -> None:
        """Show only the controls that mean something for what is picked:
        the class beside a per-class metric, the bin count only where bins
        are actually cut."""
        self.class_dropdown.visible = self._metric() in sl.PER_CLASS_METRICS
        self.bins_field.disabled = self.binning_dropdown.value == sl.BINNING_VALUES
        for control in (self.class_dropdown, self.bins_field):
            if control.page:
                control.update()

    def _metric(self) -> str:
        return self.metric_dropdown.value or METRIC_ACCURACY

    def _class_label(self):
        """The class a per-class metric is measured on, or None for the
        metrics that do not take one."""
        return self.class_dropdown.value if self._metric() in sl.PER_CLASS_METRICS else None

    def _metric_title(self) -> str:
        """What the chart and the table call the current metric."""
        label = METRIC_LABELS.get(self._metric(), self._metric())
        class_label = self._class_label()
        return f"{label.replace(' (one class)', '')} — {class_label}" if class_label else label

    def _sliceable_columns(self) -> list:
        """Every metadata column the app already considers usable as a
        control: high-cardinality ones are excluded for the same reason the
        Gallery excludes them from its filters -- a 'session_id' column
        would produce one slice per row."""
        dataset = self.ctx.state.dataset
        if dataset is None:
            return []
        kinds = dataset.column_kinds()
        return sorted(kinds.binary + kinds.categorical + kinds.numeric)

    # ---- events ---------------------------------------------------------
    def _slice_manual_page(self) -> str:
        """The metrics that need a class are defined on their own page --
        a false alarm rate is not something the breakdown page explains."""
        metric = self.metric_dropdown.value
        return "slice-metrics" if metric in sl.PER_CLASS_METRICS else "slice-breakdown"

    def _on_column_change(self, e):
        self._sync_controls()
        if not self.column_dropdown.value:
            self.body.controls = []
            self._safe_update()
            return
        self._set_busy(True)
        self.rebuild_job.run(self._render_and_update, on_error=self._on_error)

    def _on_display_change(self, e):
        """Metric / class / order / row floor only reshuffle what is already
        computed, so they never go through the background job -- the
        per-class counts every metric here reads are already on every row
        (see core/slices.SliceRow)."""
        self._sync_controls()
        if self.column_dropdown.value:
            self._render()
            self._safe_update()

    def _on_error(self, ex: Exception) -> None:
        self._set_busy(False)
        self.body.controls = [info_banner(f"Error computing slices: {ex}", icon=ft.Icons.ERROR_OUTLINE)]
        self._safe_update()

    def _render_and_update(self) -> None:
        try:
            self._render()
        finally:
            self._set_busy(False)
        self._safe_update()

    def _set_busy(self, busy: bool) -> None:
        self.progress.visible = busy
        if self.progress.page:
            self.progress.update()

    def _safe_update(self) -> None:
        if self.body.page:
            self.body.update()

    # ---- computation ----------------------------------------------------
    def _int_field(self, field: ft.TextField, default: int, low: int, high: int) -> int:
        try:
            value = int(str(field.value).strip())
        except (TypeError, ValueError):
            value = default
        value = max(low, min(high, value))
        field.value = str(value)
        return value

    def _spec(self, column: str, bins: int, binning: str):
        key = (column, bins, binning)
        if key not in self._spec_cache:
            # Built from the dataset's own column, never from one
            # experiment's joined subset, so every experiment's table has
            # the same rows (see core/slices.SliceSpec).
            series = self.ctx.state.dataset.df[column]
            self._spec_cache[key] = sl.build_spec(series, bins=bins, binning=binning)
        return self._spec_cache[key]

    def _report(self, name: str, column: str, bins: int, binning: str, spec):
        key = (column, bins, binning, name)
        if key not in self._report_cache:
            state = self.ctx.state
            merged = state.merged(name)
            _log(f"slicing '{name}' by '{column}' ({len(merged):,} merged rows)...")
            self._report_cache[key] = sl.slice_report(
                name, merged, state.dataset.columns.label_col, spec,
                # The dataset's labels rather than this experiment's own, so
                # a class one experiment never predicts still gets a column
                # -- and the same one in every experiment's table.
                class_labels=[str(l) for l in state.dataset.labels()],
            )
        return self._report_cache[key]

    # ---- rendering ------------------------------------------------------
    def _render(self) -> None:
        state = self.ctx.state
        column = self.column_dropdown.value
        if state.dataset is None or not self._selected_names or not column:
            self.body.controls = []
            return

        bins = self._int_field(self.bins_field, sl.DEFAULT_BINS, 2, sl.MAX_BINS)
        min_rows = self._int_field(self.min_rows_field, sl.DEFAULT_MIN_ROWS, 0, 1_000_000)
        binning = self.binning_dropdown.value or sl.BINNING_AUTO

        try:
            spec = self._spec(column, bins, binning)
            reports = {name: self._report(name, column, bins, binning, spec)
                       for name in self._selected_names}
        except Exception as ex:
            _log(f"error slicing by '{column}': {ex}")
            traceback.print_exc()
            self.body.controls = [info_banner(f"Error computing slices: {ex}", icon=ft.Icons.ERROR_OUTLINE)]
            return

        first = next(iter(reports.values()))
        if not first.rows:
            self.body.controls = [info_banner(
                f"No rows to slice by '{column}' -- the column is absent from the merged data, "
                f"or nothing survived the dataset/experiment join.")]
            return

        ordered_values = [row.value for row in sl.sort_rows(
            first.rows, self.sort_dropdown.value or sl.SORT_WORST, min_rows,
            metric=self._metric(), class_label=self._class_label())]

        self.body.controls = [
            *self._notes(spec, reports, min_rows),
            self._table(ordered_values, reports, min_rows),
            self._chart(ordered_values, reports),
        ]

    def _notes(self, spec, reports, min_rows: int) -> list:
        notes = []
        if spec.note:
            notes.append(spec.note)
        if spec.n_values_dropped:
            notes.append(
                f"{spec.n_values_dropped} value(s) of '{spec.column}' are not shown: the column has "
                f"more distinct values than the app builds one control per.")
        unassigned = max(report.n_unassigned for report in reports.values())
        if unassigned:
            notes.append(f"{unassigned:,} row(s) fall outside the slices shown.")
        small = [row.value for row in next(iter(reports.values())).rows if row.is_small(min_rows)]
        if small:
            notes.append(
                f"{len(small)} slice(s) hold fewer than {min_rows} rows; they are shown greyed and "
                f"never ranked first -- their accuracy is not measured precisely enough to act on.")
        return [ft.Text(n, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT) for n in notes]

    def _table(self, ordered_values: list, reports: dict, min_rows: int) -> ft.Control:
        metric, class_label = self._metric(), self._class_label()
        # What the second column under each experiment says, which depends on
        # what kind of number the first one is: a rate over countable
        # successes gets its confidence interval, a rate that is not (F1, a
        # mean) gets its distance from the whole dataset, and a count gets
        # the share of that dataset it represents -- "support 347" against a
        # dataset-wide 1,490 is a slice holding a quarter of the class, which
        # is the fact that makes the 347 mean something. A count minus a
        # count would be a large negative number with no reading at all.
        has_interval = sl.metric_fraction(
            next(iter(reports.values())).overall, metric, class_label)[0] is not None
        is_count = metric in sl.COUNT_METRICS
        second = "95% CI" if has_interval else ("share of all rows" if is_count else "vs. all rows")
        names = list(reports)
        rows_by_name = {name: {row.value: row for row in report.rows} for name, report in reports.items()}

        headers = ["Slice", "n", "Classes"]
        for name in names:
            headers += [name, second]
        columns = [ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12)) for h in headers]

        table_rows = []
        for value in ordered_values:
            reference = rows_by_name[names[0]][value]
            faded = reference.is_small(min_rows)
            color = ft.Colors.ON_SURFACE_VARIANT if faded else None
            cells = [
                ft.DataCell(ft.Text(value, size=12, color=color, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{reference.n:,}", size=12, color=color)),
                ft.DataCell(ft.Text(str(reference.n_classes), size=12, color=color)),
            ]
            for name in names:
                row = rows_by_name[name][value]
                overall = reports[name].overall
                cells.append(ft.DataCell(
                    ft.Row([
                        ft.Text(self._fmt_metric(row), size=12, color=color),
                        ft.IconButton(
                            ft.Icons.IMAGE_SEARCH, icon_size=15,
                            tooltip=f"Show the {row.n_errors:,} error(s) in this slice",
                            disabled=row.n_errors == 0,
                            on_click=lambda e, n=name, v=value: self._open_errors(n, v),
                        ),
                    ], spacing=2, tight=True)
                ))
                if has_interval:
                    low, high = sl.metric_interval(row, metric, class_label)
                    cells.append(ft.DataCell(ft.Text(
                        f"[{_fmt(low, 2)}, {_fmt(high, 2)}]", size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT)))
                elif is_count:
                    cells.append(ft.DataCell(self._share_text(row, overall)))
                else:
                    cells.append(ft.DataCell(self._delta_text(
                        sl.metric_value(row, metric, class_label)
                        - sl.metric_value(overall, metric, class_label))))
            table_rows.append(ft.DataRow(cells=cells))

        # The "all rows" reference goes at the bottom rather than the top:
        # it is what every row above is compared against, and putting it
        # first would make it look like the worst or best slice.
        overall_cells = [
            ft.DataCell(ft.Text("all rows", size=12, italic=True)),
            ft.DataCell(ft.Text(f"{reports[names[0]].overall.n:,}", size=12, italic=True)),
            ft.DataCell(ft.Text(str(reports[names[0]].overall.n_classes), size=12, italic=True)),
        ]
        for name in names:
            overall_cells.append(ft.DataCell(ft.Text(
                self._fmt_metric(reports[name].overall), size=12, italic=True,
                weight=ft.FontWeight.BOLD)))
            overall_cells.append(ft.DataCell(ft.Text("")))
        table_rows.append(ft.DataRow(cells=overall_cells))

        table = ft.DataTable(columns=columns, rows=table_rows,
                             heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=8)
        return card(ft.Column([ft.Row([table], scroll=ft.ScrollMode.ALWAYS)], spacing=8))

    def _fmt_metric(self, row) -> str:
        """One slice's number, spelled for what it is: a count has no
        decimals, and printing "12.000 rows" reads as a rate."""
        metric, class_label = self._metric(), self._class_label()
        value = sl.metric_value(row, metric, class_label)
        if value != value:
            return "—"
        return f"{value:,.0f}" if metric in sl.COUNT_METRICS else _fmt(value)

    def _share_text(self, row, overall) -> ft.Control:
        """How much of the whole dataset's count this slice holds."""
        metric, class_label = self._metric(), self._class_label()
        total = sl.metric_value(overall, metric, class_label)
        value = sl.metric_value(row, metric, class_label)
        if not total or total != total or value != value:
            return ft.Text("—", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        return ft.Text(f"{value / total:.1%}", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

    def _delta_text(self, delta: float) -> ft.Control:
        if delta != delta:
            return ft.Text("—", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        color = ft.Colors.GREEN if delta >= 0 else ft.Colors.RED
        return ft.Text(f"{delta:+.3f}", size=11, color=color)

    def _chart(self, ordered_values: list, reports: dict) -> ft.Control:
        metric, class_label = self._metric(), self._class_label()
        label = self._metric_title()
        values = ordered_values[:MAX_CHART_SLICES]
        names = list(reports)
        all_names = list(self.ctx.state.experiments.keys())
        rows_by_name = {name: {row.value: row for row in report.rows} for name, report in reports.items()}

        n_exp = len(names)
        column = self.column_dropdown.value
        fig_width = max(6.0, min(18.0, len(values) * (0.5 + 0.06 * n_exp)))
        # The legend always sits under the axes (see components.legend_below),
        # so the figure is grown to hold it rather than the plot shrunk.
        legend_entries = n_exp + (n_exp if not _counts_agree(rows_by_name, names, values) else 1)
        legend_cols = legend_columns(legend_entries)
        # Two things on one figure, because they answer each other: how many
        # crops a slice holds, and how well the model does on it. Counts are
        # bars in the background (the shape of the test set), the metric is
        # a marked line on top (the result). Reading a 0.42 without seeing
        # that it rests on nine crops is the mistake this arrangement makes
        # hard.
        fig, ax_counts = subplots(
            figsize=(fig_width, 4.8 + legend_height_inches(legend_entries, legend_cols)))
        ax_metric = ax_counts.twinx()
        # twinx puts the metric axes on top, which is the z-order we want
        # (lines over bars); its ticks are moved to the left so the subject
        # of the chart sits on the primary side and the counts read as the
        # context they are.
        ax_metric.yaxis.tick_left()
        ax_metric.yaxis.set_label_position("left")
        ax_counts.yaxis.tick_right()
        ax_counts.yaxis.set_label_position("right")

        x = np.arange(len(values))

        # ---- bars: how many crops each slice holds ----
        # One neutral bar per slice while every experiment agrees on the
        # count; when they do not (see _counts_agree), hiding all but one
        # would misreport the very thing this axis is for.
        counts_agree = _counts_agree(rows_by_name, names, values)
        if counts_agree:
            ax_counts.bar(x, [rows_by_name[names[0]][v].n for v in values],
                          width=0.68, color=COUNT_BAR_COLOR, label="crops in slice")
        else:
            bar_width = 0.68 / max(n_exp, 1)
            for i, name in enumerate(names):
                ax_counts.bar(
                    x + (i - (n_exp - 1) / 2) * bar_width,
                    [rows_by_name[name][v].n for v in values], bar_width,
                    color=color_for_experiment(name, self.ctx.state.color_slots), alpha=0.28,
                    label=f"crops in slice ({name})",
                )

        ax_counts.set_ylabel("crops in slice", color=COUNT_AXIS_COLOR, fontsize=9)
        ax_counts.tick_params(axis="y", colors=COUNT_AXIS_COLOR, labelsize=8)
        ax_counts.set_ylim(0, max(
            [rows_by_name[name][v].n for name in names for v in values] or [1]) * 1.15)

        # ---- lines: the metric itself ----
        # The interval comes from the same counts the value does (see
        # core/slices.metric_fraction), so a false alarm rate resting on
        # nine negatives shows it the way a small slice's accuracy always
        # has -- and a metric that is not a proportion simply has no bars.
        for name in names:
            rows = [rows_by_name[name][v] for v in values]
            heights = [sl.metric_value(row, metric, class_label) for row in rows]
            color = color_for_experiment(name, self.ctx.state.color_slots)
            intervals = [sl.metric_interval(row, metric, class_label) for row in rows]
            if any(low == low for low, _ in intervals):
                lows = [max(0.0, h - low) if low == low else 0.0
                        for h, (low, _) in zip(heights, intervals)]
                highs = [max(0.0, high - h) if high == high else 0.0
                         for h, (_, high) in zip(heights, intervals)]
                ax_metric.errorbar(x, heights, yerr=[lows, highs], label=name, color=color,
                                   marker="o", markersize=5, linewidth=1.8, capsize=3,
                                   elinewidth=0.9, ecolor=color, alpha=0.95)
            else:
                ax_metric.plot(x, heights, label=name, color=color,
                               marker="o", markersize=5, linewidth=1.8)
            overall = sl.metric_value(reports[name].overall, metric, class_label)
            if overall == overall:
                ax_metric.axhline(overall, color=color, linestyle="--", linewidth=1, alpha=0.6)

        ax_metric.set_ylabel(label)
        # Rates share the 0..1 axis so two slices are comparable by eye;
        # a mean confidence and a row count do not, and forcing them onto it
        # would flatten every point against one edge.
        if metric not in sl.COUNT_METRICS and metric != METRIC_MEAN_CONFIDENCE:
            ax_metric.set_ylim(0, 1.05)

        ax_counts.set_xticks(x)
        ax_counts.set_xticklabels(values, rotation=45, ha="right", fontsize=8)
        ax_counts.set_xlim(-0.7, len(values) - 0.3)
        ax_metric.set_title(f"{label} by {column} (dashed line: all rows)")
        # Only the metric axis is gridded: the counts are context on their
        # own scale, and a second set of horizontal lines for them would
        # cross the first at meaningless angles. Nothing on the x axis to
        # subdivide either -- the ticks are slices, not a scale.
        style_axes(ax_counts, x=GRID_MAJOR, y=GRID_NONE)
        style_axes(ax_metric, x=GRID_NONE)

        # One legend for both axes -- two of them, one per corner, is how a
        # reader ends up thinking the bars and the lines are separate charts
        # -- and under the plot, where it cannot cover the points it names.
        handles = ax_metric.get_legend_handles_labels()[0] + ax_counts.get_legend_handles_labels()[0]
        labels = ax_metric.get_legend_handles_labels()[1] + ax_counts.get_legend_handles_labels()[1]
        reserve_legend_space(fig, legend_entries, legend_cols)
        legend_below(ax_metric, handles, labels, ncol=legend_cols)

        notes = [
            "Bars are the number of crops in each slice (right axis); the marked lines are the "
            "metric (left axis). The line follows the order chosen above, so it traces a real "
            "trend only when that order is 'By value'.",
        ]
        if not counts_agree:
            notes.append(
                "The experiments do not cover the same rows, so each gets its own count bars.")
        if len(ordered_values) > MAX_CHART_SLICES:
            notes.append(
                f"Chart shows the first {MAX_CHART_SLICES} slices in this order; the table above "
                f"has all {len(ordered_values)}.")

        fig_w, fig_h = fig.get_size_inches()
        box_width = int(min(CHART_MAX_WIDTH, max(600.0, fig_w * CHART_PX_PER_INCH)))
        return card(ft.Column([
            fig_to_zoomable_image(fig, self.ctx, f"{label} by {column}",
                                  width=box_width, height=int(box_width * fig_h / fig_w)),
            *[ft.Text(n, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT) for n in notes],
        ], spacing=8))

    # ---- drilling into a slice -----------------------------------------
    def _open_errors(self, experiment_name: str, value: str) -> None:
        """The crops this slice got wrong, in the shared carousel -- so a
        slice that stands out is one click from the images that explain it,
        and they carry the same review controls as everywhere else."""
        state = self.ctx.state
        column = self.column_dropdown.value
        bins = self._int_field(self.bins_field, sl.DEFAULT_BINS, 2, sl.MAX_BINS)
        spec = self._spec(column, bins, self.binning_dropdown.value or sl.BINNING_AUTO)
        merged = state.merged(experiment_name)
        cols = state.dataset.columns

        names = sl.assign(merged[spec.column], spec)
        subset = merged[(names == value) & (~merged["correct"].to_numpy().astype(bool))]

        self.carousel_title.value = (
            f"{column} = {value}  ·  {experiment_name}  ·  {len(subset):,} error(s)")
        carousel = Carousel(
            state.dataset, len(subset),
            lambda start, end: subset.iloc[start:end].to_dict("records"),
            cols.crop_col, cols.uuid_col, ctx=self.ctx,
            review_source="slice_breakdown",
            # Same reasoning as the confusion matrix: the model's own
            # prediction is the hypothesis worth pre-filling, since a crop
            # the model gets wrong in one slice and right everywhere else is
            # as likely to be mislabelled as to be hard.
            suggestions={str(u): str(p) for u, p in
                         zip(subset[cols.uuid_col], subset["predicted_label"])},
            label_col=cols.label_col,
        )
        self.carousel_content.content = carousel.build()
        self.ctx.page.open(self.carousel_dialog)

    def _close_carousel(self, e) -> None:
        self.ctx.page.close(self.carousel_dialog)

    # ---- assembly -------------------------------------------------------
    def build(self) -> ft.Control:
        return ft.Column([
            section_title("Break down by metadata", icon=ft.Icons.SEGMENT, ctx=self.ctx, manual=self._slice_manual_page),
            ft.Text(
                "Every number above is an average over the whole dataset. Pick a metadata column "
                "to recompute it per slice -- per sensor, per site, per crop size. Nothing is "
                "computed until a column is picked.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            card(ft.Column([
                ft.Row([self.column_dropdown, self.binning_dropdown, self.bins_field,
                        self.metric_dropdown, self.class_dropdown,
                        self.sort_dropdown, self.min_rows_field],
                       spacing=12, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.progress,
            ], spacing=10)),
            self.body,
        ], spacing=12)
