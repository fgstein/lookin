"""The Metrics tab's per-slice breakdown (core/slices.py, and the manual's
slices chapter for the reasoning). Nothing is computed until a
column is picked; metric / class / order / row floor only reshuffle what is
computed; clicking a slice's error count opens its errors in the carousel.

A component of MetricsView: its controls are `metrics:slice` actions with
an `op`.
"""
from __future__ import annotations

import traceback

from lookin.core import charts
from lookin.core import slices as sl
from lookin.core.chart_style import color_for_experiment
from lookin.web import html as h
from lookin.web.views.carousel import CarouselDialog
from lookin.web.views.media import fig_html
from lookin.web.views.widgets import class_select


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



def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value == value else "—"


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class SlicePanel:
    def __init__(self, owner, session, action: str = "slice"):
        self.owner = owner
        self.s = session
        self.action = action
        self._selected_names: list = []
        self._spec_cache: dict = {}
        self._report_cache: dict = {}
        self.column = None
        self.metric = METRIC_ACCURACY
        self.class_value = None
        self.sort = sl.SORT_WORST
        self.binning = sl.BINNING_AUTO
        self.bins_value = str(sl.DEFAULT_BINS)
        self.min_rows_value = str(sl.DEFAULT_MIN_ROWS)
        self.busy = False
        self.body_html = ""

    # ---- lifecycle ----

    def set_experiments(self, names: list) -> None:
        """From the Metrics render pass (off the request thread)."""
        self._selected_names = list(names)
        self._sync_options()
        if self.column:
            self._render()

    def invalidate(self) -> None:
        self._spec_cache.clear()
        self._report_cache.clear()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        for key in [k for k in self._report_cache if k[-1] == old_name]:
            report = self._report_cache.pop(key)
            report.experiment = new_name
            self._report_cache[(*key[:-1], new_name)] = report
        self._selected_names = [new_name if n == old_name else n for n in self._selected_names]

    def _sync_options(self) -> None:
        if self.column not in self._sliceable_columns():
            self.column = None
        labels = self._labels()
        if self.class_value not in labels:
            self.class_value = labels[0] if labels else None

    def _labels(self) -> list:
        dataset = self.s.state.dataset
        return [str(l) for l in dataset.labels()] if dataset is not None else []

    def _sliceable_columns(self) -> list:
        dataset = self.s.state.dataset
        if dataset is None:
            return []
        kinds = dataset.column_kinds()
        return sorted(kinds.binary + kinds.categorical + kinds.numeric)

    def _class_label(self):
        return self.class_value if self.metric in sl.PER_CLASS_METRICS else None

    def _metric_title(self) -> str:
        label = METRIC_LABELS.get(self.metric, self.metric)
        class_label = self._class_label()
        return f"{label.replace(' (one class)', '')} — {class_label}" if class_label else label

    def manual_page(self) -> str:
        return "slice-metrics" if self.metric in sl.PER_CLASS_METRICS else "slice-breakdown"

    @staticmethod
    def _int(value, default: int, low: int, high: int) -> int:
        try:
            v = int(str(value).strip())
        except (TypeError, ValueError):
            v = default
        return max(low, min(high, v))

    # ---- computation ----

    def _spec(self, column: str, bins: int, binning: str):
        key = (column, bins, binning)
        if key not in self._spec_cache:
            self._spec_cache[key] = sl.build_spec(self.s.state.dataset.df[column], bins=bins, binning=binning)
        return self._spec_cache[key]

    def _report(self, name: str, column: str, bins: int, binning: str, spec):
        key = (column, bins, binning, name)
        if key not in self._report_cache:
            state = self.s.state
            merged = state.merged(name)
            _log(f"slicing '{name}' by '{column}' ({len(merged):,} merged rows)...")
            self._report_cache[key] = sl.slice_report(
                name, merged, state.dataset.columns.label_col, spec,
                class_labels=[str(l) for l in state.dataset.labels()])
        return self._report_cache[key]

    def _render(self) -> None:
        state = self.s.state
        column = self.column
        if state.dataset is None or not self._selected_names or not column:
            self.body_html = ""
            return
        bins = self._int(self.bins_value, sl.DEFAULT_BINS, 2, sl.MAX_BINS)
        self.bins_value = str(bins)
        min_rows = self._int(self.min_rows_value, sl.DEFAULT_MIN_ROWS, 0, 1_000_000)
        self.min_rows_value = str(min_rows)
        binning = self.binning or sl.BINNING_AUTO
        try:
            spec = self._spec(column, bins, binning)
            reports = {name: self._report(name, column, bins, binning, spec) for name in self._selected_names}
        except Exception as ex:  # noqa: BLE001
            _log(f"error slicing by '{column}': {ex}")
            traceback.print_exc()
            self.body_html = h.banner(f"Error computing slices: {ex}", "error")
            return
        first = next(iter(reports.values()))
        if not first.rows:
            self.body_html = h.banner(
                f"No rows to slice by '{column}' -- the column is absent from the merged data, "
                f"or nothing survived the dataset/experiment join.", "info")
            return
        ordered_values = [row.value for row in sl.sort_rows(
            first.rows, self.sort or sl.SORT_WORST, min_rows, metric=self.metric, class_label=self._class_label())]
        self.body_html = (self._notes(spec, reports, min_rows) + self._table(ordered_values, reports, min_rows)
                          + self._chart(ordered_values, reports))

    def _notes(self, spec, reports, min_rows: int) -> str:
        notes = []
        if spec.note:
            notes.append(spec.note)
        if spec.n_values_dropped:
            notes.append(f"{spec.n_values_dropped} value(s) of '{spec.column}' are not shown: the column has "
                         f"more distinct values than the app builds one control per.")
        unassigned = max(report.n_unassigned for report in reports.values())
        if unassigned:
            notes.append(f"{unassigned:,} row(s) fall outside the slices shown.")
        small = [row.value for row in next(iter(reports.values())).rows if row.is_small(min_rows)]
        if small:
            notes.append(f"{len(small)} slice(s) hold fewer than {min_rows} rows; they are shown greyed and "
                         f"never ranked first -- their accuracy is not measured precisely enough to act on.")
        return "".join(f'<div class="muted small"><i>{h.e(n)}</i></div>' for n in notes)

    def _fmt_metric(self, row) -> str:
        value = sl.metric_value(row, self.metric, self._class_label())
        if value != value:
            return "—"
        return f"{value:,.0f}" if self.metric in sl.COUNT_METRICS else _fmt(value)

    def _table(self, ordered_values: list, reports: dict, min_rows: int) -> str:
        metric, class_label = self.metric, self._class_label()
        has_interval = sl.metric_fraction(next(iter(reports.values())).overall, metric, class_label)[0] is not None
        is_count = metric in sl.COUNT_METRICS
        second = "95% CI" if has_interval else ("share of all rows" if is_count else "vs. all rows")
        names = list(reports)
        rows_by_name = {name: {row.value: row for row in report.rows} for name, report in reports.items()}
        head = "<th>Slice</th><th class=num>n</th><th class=num>Classes</th>" + "".join(
            f"<th class=num>{h.e(n)}</th><th class=num>{h.e(second)}</th>" for n in names)
        body = []
        for value in ordered_values:
            reference = rows_by_name[names[0]][value]
            faded = " dim" if reference.is_small(min_rows) else ""
            cells = [f"<td><b>{h.e(value)}</b></td>", f"<td class=num>{reference.n:,}</td>",
                     f"<td class=num>{reference.n_classes}</td>"]
            for name in names:
                row = rows_by_name[name][value]
                overall = reports[name].overall
                btn = h.button(icon_name="image", small=True, kind="ghost",
                               act_attr=h.act(self.owner.target, self.action, op="errors", experiment=name, value=value),
                               title=f"Show the {row.n_errors:,} error(s) in this slice", disabled=row.n_errors == 0)
                cells.append(f'<td class="num"><span class="row-nw" style="gap:2px;justify-content:flex-end">{h.e(self._fmt_metric(row))}{btn}</span></td>')
                if has_interval:
                    low, high = sl.metric_interval(row, metric, class_label)
                    cells.append(f'<td class="num muted small">[{_fmt(low, 2)}, {_fmt(high, 2)}]</td>')
                elif is_count:
                    total = sl.metric_value(overall, metric, class_label)
                    v = sl.metric_value(row, metric, class_label)
                    share = "—" if (not total or total != total or v != v) else f"{v / total:.1%}"
                    cells.append(f'<td class="num muted small">{share}</td>')
                else:
                    delta = sl.metric_value(row, metric, class_label) - sl.metric_value(overall, metric, class_label)
                    if delta != delta:
                        cells.append('<td class="num muted small">—</td>')
                    else:
                        cells.append(f'<td class="num small {"c-ok" if delta >= 0 else "c-err"}">{delta:+.3f}</td>')
            body.append(f'<tr class="{faded}">{"".join(cells)}</tr>')
        overall_cells = ['<td><i>all rows</i></td>', f'<td class=num><i>{reports[names[0]].overall.n:,}</i></td>',
                         f'<td class=num><i>{reports[names[0]].overall.n_classes}</i></td>']
        for name in names:
            overall_cells.append(f'<td class=num><b><i>{h.e(self._fmt_metric(reports[name].overall))}</i></b></td><td></td>')
        body.append(f'<tr>{"".join(overall_cells)}</tr>')
        return (f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>')

    def _chart(self, ordered_values: list, reports: dict) -> str:
        label = self._metric_title()
        colors = {name: color_for_experiment(name, self.s.state.color_slots) for name in reports}
        fig, notes = charts.slice_chart(ordered_values, reports, self.metric, label, self.column, colors,
                                        class_label=self._class_label())
        return (fig_html(self.s, fig, f"{label} by {self.column}")
                + "".join(f'<div class="muted small"><i>{h.e(n)}</i></div>' for n in notes))

    # ---- rendering the panel ----

    def render(self) -> str:
        c = lambda op: h.on_change(self.owner.target, self.action, op=op)   # noqa: E731
        columns = self._sliceable_columns()
        controls = [
            h.select([("", "Break down by column...")] + [(col, col) for col in columns], self.column or "",
                     change=c("column"), cls="mono"),
            h.select(list(BINNING_LABELS), self.binning, change=c("binning"), label="Binning",
                     title="Automatic gives one bar per value to a column of few distinct values, "
                           "and quantile bins to a continuous one"),
            h.text_input(self.bins_value, label="Bins", width=56, change=c("bins"),
                         title="Quantile bins for a numeric column", disabled=self.binning == sl.BINNING_VALUES),
            h.select(list(SLICE_METRICS), self.metric, change=c("metric"), label="Metric"),
        ]
        if self.metric in sl.PER_CLASS_METRICS:
            controls.append(class_select(self._labels(), self.class_value, c("class"), label="Class"))
        controls += [
            h.select(list(SORT_LABELS), self.sort, change=c("sort"), label="Order"),
            h.text_input(self.min_rows_value, label="Min rows", width=70, change=c("min_rows"),
                         title="Slices smaller than this are shown but never ranked first"),
        ]
        busy = h.spinner("Computing slices...") if self.busy else ""
        body = f'<div class="row-wrap">{"".join(controls)}</div>{busy}'
        if self.body_html:
            body += f'<div class="stack" style="margin-top:8px">{self.body_html}</div>'
        return h.section("Break down by metadata", body, manual=self.manual_page(),
                         sub="pick a metadata column to recompute the metrics per slice")

    # ---- actions ----

    def handle(self, op: str, args: dict) -> None:
        value = args.get("value")
        if op == "column":
            self.column = value or None
            self._recompute()
        elif op == "binning":
            self.binning = value or sl.BINNING_AUTO
            self._recompute()
        elif op == "bins":
            self.bins_value = str(value or "")
            self._recompute()
        elif op in ("metric", "class", "sort", "min_rows"):
            if op == "metric":
                self.metric = value or METRIC_ACCURACY
            elif op == "class":
                if str(value) in self._labels():
                    self.class_value = str(value)
            elif op == "sort":
                self.sort = value or sl.SORT_WORST
            else:
                self.min_rows_value = str(value or "")
            if self.column:
                self._render()
        elif op == "errors":
            self._open_errors(str(args.get("experiment", "")), str(args.get("value", "")))

    def _recompute(self) -> None:
        if not self.column:
            self.body_html = ""
            return
        self.busy = True

        def run():
            try:
                self._render()
            finally:
                self.busy = False
                self.owner.refresh()

        def failed(ex):
            self.busy = False
            self.body_html = h.banner(f"Error computing slices: {ex}", "error")
            self.owner.refresh()

        self.s.job("Slices").run(run, on_error=failed)

    def _open_errors(self, experiment_name: str, value: str) -> None:
        state = self.s.state
        if experiment_name not in state.experiments or not self.column:
            return
        column = self.column
        bins = self._int(self.bins_value, sl.DEFAULT_BINS, 2, sl.MAX_BINS)
        spec = self._spec(column, bins, self.binning or sl.BINNING_AUTO)
        merged = state.merged(experiment_name)
        cols = state.dataset.columns
        names = sl.assign(merged[spec.column], spec)
        subset = merged[(names == value) & (~merged["correct"].to_numpy().astype(bool))]
        self.s.open_dialog(CarouselDialog(
            self.s, f"{column} = {value}  ·  {experiment_name}  ·  {len(subset):,} error(s)",
            state.dataset, len(subset), lambda start, end: subset.iloc[start:end].to_dict("records"),
            cols.crop_col, cols.uuid_col, review_source="slice_breakdown",
            suggestions={str(u): str(p) for u, p in zip(subset[cols.uuid_col], subset["predicted_label"])},
            label_col=cols.label_col))
