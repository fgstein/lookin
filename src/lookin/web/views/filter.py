"""Filter: remove rows from the loaded dataset, by conditions on its columns
or by a query written as a formula (see core/dataset_filter.py).

The tab edits a *draft*: every change re-counts what the draft keeps and
shows it, but nothing else in the app moves until Apply. Applying changes
the rows every other tab reads, which marks them all out of date -- that
should happen once, when the filter says what the user means, not on every
value ticked on the way there.
"""
from __future__ import annotations

import json
import secrets

import numpy as np
import pandas as pd

from lookin.core import dataset_filter as dsf
from lookin.web import html as h
from lookin.web.views.base import View

# Values listed under a "pick values" condition before a search narrows them.
VALUES_SHOWN = 40
# Rows in the preview table.
PREVIEW_ROWS = 50
PREVIEW_COL_CHARS = 60

OP_LABELS = (
    (dsf.OP_IN, "is one of"),
    (dsf.OP_NOT_IN, "is not one of"),
    (dsf.OP_RANGE, "is between"),
    (dsf.OP_CONTAINS, "contains"),
    (dsf.OP_NOT_CONTAINS, "does not contain"),
    (dsf.OP_REGEX, "matches regex"),
    (dsf.OP_MISSING, "is empty"),
    (dsf.OP_PRESENT, "is not empty"),
)

PREVIEW_KEPT = "kept"
PREVIEW_REMOVED = "removed"


def _new_cid() -> str:
    return "c" + secrets.token_hex(4)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


class FilterView(View):
    name = "filter"
    title = "Filter"

    def __init__(self, session):
        super().__init__(session)
        self.conditions: list = []          # dicts as in dsf, plus "_cid"
        self.query = ""
        self.query_on = True
        self.add_column = ""
        self.searches: dict = {}            # cid -> what is typed in its value search
        self.preview = PREVIEW_KEPT
        self._counts_cache: dict = {}       # (frame id, column) -> value counts
        self._draft_cache = None            # (key, mask, steps, error)

    def subtitle(self) -> str:
        return "Remove rows from the dataset: every tab then works on the rows kept."

    # ---- draft <-> applied ----

    def draft_spec(self) -> dict:
        return dsf.normalize_spec({dsf.CONDITIONS: self.conditions, dsf.QUERY: self.query,
                                   dsf.QUERY_ON: self.query_on})

    def _set_draft(self, spec: dict) -> None:
        spec = dsf.normalize_spec(spec)
        self.conditions = [{**c, "_cid": _new_cid()} for c in spec[dsf.CONDITIONS]]
        self.query = spec[dsf.QUERY]
        self.query_on = spec[dsf.QUERY_ON]
        self.searches = {}
        self._draft_cache = None

    def adopt_applied(self) -> None:
        """Make the draft what is applied -- after a workspace or snapshot
        put a filter in place without going through this tab."""
        self._set_draft(self.state.dataset_filter or {})
        self.refresh()

    def _applied_key(self) -> str:
        return json.dumps(dsf.normalize_spec(self.state.dataset_filter or {}) if dsf.is_active(
            self.state.dataset_filter or {}) else {}, sort_keys=True)

    def _draft_key(self) -> str:
        spec = self.draft_spec()
        return json.dumps(spec if dsf.is_active(spec) else {}, sort_keys=True)

    def has_changes(self) -> bool:
        return self._draft_key() != self._applied_key()

    def _full(self):
        dataset = self.state.dataset
        return None if dataset is None else dataset.full_df

    def _draft_result(self) -> tuple:
        """(mask, steps, error) of the draft on every row; cached until the
        draft or the data changes."""
        full = self._full()
        key = (id(full), self._draft_key())
        if self._draft_cache is not None and self._draft_cache[0] == key:
            return self._draft_cache[1:]
        try:
            mask, steps = dsf.apply_spec(full, self.draft_spec())
            result = (mask, steps, "")
        except dsf.FilterError as ex:
            result = (None, [], str(ex))
        self._draft_cache = (key, *result)
        return result

    def _value_counts(self, column: str) -> pd.Series:
        full = self._full()
        key = (id(full), column)
        counts = self._counts_cache.get(key)
        if counts is None:
            series = full[column]
            counts = dsf._as_text(series[series.notna()]).value_counts()
            if len(self._counts_cache) > 64:
                self._counts_cache.clear()
            self._counts_cache[key] = counts
        return counts

    # ---- rendering ----

    def render(self) -> str:
        state = self.state
        if state.dataset is None:
            return h.section("Filter the dataset", h.banner("Load a dataset in the Sources tab first.", "info"),
                             manual="dataset-filter")
        return (f'<div class="stack flt">{self._summary_html()}{self._conditions_html()}'
                f'{self._query_html()}{self._preview_html()}</div>')

    def _summary_html(self) -> str:
        state = self.state
        dataset = state.dataset
        total = len(dataset.full_df)
        mask, steps, error = self._draft_result()
        changes = self.has_changes()
        if state.dataset_filter_error:
            applied = h.banner(f"The filter saved with this data could not be applied, so every row is in view: "
                               f"{state.dataset_filter_error}", "error")
        elif dataset.is_filtered():
            kept = len(dataset.df)
            applied = h.status_line(f"Applied: every tab works on {kept:,} of {total:,} rows "
                                    f"({total - kept:,} removed).", "ok")
        else:
            applied = h.status_line(f"No filter applied: every tab works on all {total:,} rows.", "neutral")
        if error:
            draft = f'<div class="flt-draft err">{h.icon("warn", 14)}<span>{h.e(error)}</span></div>'
        else:
            kept = int(mask.sum())
            frac = kept / total if total else 0.0
            draft = (f'<div class="flt-draft"><div class="flt-bar"><div style="width:{100 * frac:.1f}%"></div></div>'
                     f'<span><b>{kept:,}</b> of {total:,} rows kept'
                     f'{f" · {total - kept:,} removed" if kept < total else ""}'
                     f'{" (not applied yet)" if changes else ""}</span></div>')
        steps_html = ""
        if steps and not error:
            steps_html = '<div class="flt-steps">' + "".join(
                f'<span class="flt-step"><span class="mono">{h.e(text)}</span>'
                f'<span class="muted">removes {n:,}</span></span>' for text, n in steps) + "</div>"
        empty = not error and mask is not None and not mask.any()
        buttons = (h.button("Apply", act_attr=self.a("apply"), kind="primary" if changes else "", icon_name="check",
                            disabled=not changes or bool(error) or empty,
                            title="Nothing to work on: this keeps no row" if empty else "Make every tab work on the rows kept")
                   + h.button("Discard changes", act_attr=self.a("discard"), disabled=not changes,
                              title="Back to the filter that is applied")
                   + h.button("Remove the filter", act_attr=self.a("clear"), kind="ghost", icon_name="x",
                              disabled=not (dsf.is_active(self.state.dataset_filter) or self.conditions or self.query),
                              title="Keep every row again"))
        body = f'{applied}{draft}{steps_html}<div class="row" style="margin-top:8px">{buttons}</div>'
        return h.section("Filter the dataset", body, manual="dataset-filter",
                         sub="rows failing any active condition, or the query, are removed from every tab")

    def _columns(self) -> list:
        return [str(c) for c in self._full().columns]

    def _conditions_html(self) -> str:
        columns = self._columns()
        if self.add_column not in columns:
            label_col = self.state.dataset.columns.label_col
            self.add_column = label_col if label_col in columns else (columns[0] if columns else "")
        adder = (f'<div class="row">{h.select([(c, c) for c in columns], self.add_column, change=self.c("choose_column"), label="Column", cls="mono")}'
                 f'{h.button("Add condition", act_attr=self.a("add"), icon_name="plus", disabled=not columns)}</div>')
        rows = "".join(self._condition_html(c) for c in self.conditions)
        if not rows:
            rows = '<div class="muted small">No condition yet: pick a column and click Add condition.</div>'
        return h.section("Conditions", f'{adder}<div class="flt-conds">{rows}</div>',
                         sub="all must hold for a row to be kept")

    def _op_options(self, series: pd.Series) -> list:
        ops = [dsf.OP_IN, dsf.OP_NOT_IN]
        if _is_numeric(series):
            ops.append(dsf.OP_RANGE)
        else:
            ops += [dsf.OP_CONTAINS, dsf.OP_NOT_CONTAINS, dsf.OP_REGEX]
            if pd.to_numeric(series.head(200), errors="coerce").notna().mean() > 0.9:
                ops.append(dsf.OP_RANGE)
        ops += [dsf.OP_MISSING, dsf.OP_PRESENT]
        return [(k, t) for k, t in OP_LABELS if k in ops]

    def _condition_html(self, cond: dict) -> str:
        cid = cond["_cid"]
        full = self._full()
        column = cond[dsf.COLUMN]
        missing = column not in full.columns
        head_bits = [h.checkbox("", cond.get(dsf.ON, True), change=self.c("toggle", cid=cid),
                                title="Turn this condition off without removing it"),
                     f'<span class="mono flt-col">{h.e(column)}</span>']
        if missing:
            body = h.banner(f"This dataset has no column '{column}'.", "warn")
        else:
            series = full[column]
            options = self._op_options(series)
            if cond[dsf.OP] not in [k for k, _ in options]:
                options = options + [(cond[dsf.OP], dict(OP_LABELS).get(cond[dsf.OP], cond[dsf.OP]))]
            head_bits.append(h.select(options, cond[dsf.OP], change=self.c("set_op", cid=cid)))
            body = self._condition_body(cond, series)
        effect = ""
        if not missing and dsf.condition_is_active(cond):
            try:
                removed = int((~dsf.condition_mask(full, cond)).sum())
                effect = f'<span class="muted small">removes {removed:,} row{"s" if removed != 1 else ""}</span>'
            except dsf.FilterError as ex:
                effect = f'<span class="small c-err">{h.e(ex)}</span>'
        elif not missing and cond.get(dsf.ON, True):
            effect = '<span class="muted small">keeps every row until it is filled in</span>'
        head = (f'<div class="flt-ch">{"".join(head_bits)}{effect}<span class="grow"></span>'
                f'{h.button(icon_name="x", act_attr=self.a("remove", cid=cid), kind="ghost", small=True, title="Remove this condition")}</div>')
        off = "" if cond.get(dsf.ON, True) else " off"
        return f'<div class="flt-cond{off}" data-key="{cid}">{head}{body}</div>'

    def _condition_body(self, cond: dict, series: pd.Series) -> str:
        cid, op = cond["_cid"], cond[dsf.OP]
        if op in dsf.VALUE_OPS:
            return self._values_html(cond, series)
        if op == dsf.OP_RANGE:
            numbers = series if _is_numeric(series) else pd.to_numeric(series, errors="coerce")
            lo, hi = numbers.min(), numbers.max()
            span = (f'<span class="muted small">values from {lo:g} to {hi:g}</span>'
                    if pd.notna(lo) and pd.notna(hi) else "")

            def bound(key, label, ph):
                value = cond.get(key)
                return h.text_input("" if value is None else f"{value:g}", type_="number", label=label, width=120,
                                    placeholder="" if pd.isna(ph) else f"{ph:g}", extra=' step="any"',
                                    change=self.c("set_bound", cid=cid, which=key),
                                    enter=h.on_enter(self.name, "set_bound", cid=cid, which=key))
            return (f'<div class="row">{bound(dsf.MIN, "at least", lo)}{bound(dsf.MAX, "at most", hi)}{span}'
                    f'<span class="muted small">empty or non-numeric values are removed</span></div>')
        if op in dsf.TEXT_OPS:
            placeholder = "a regular expression, e.g. ^img_0[0-9]" if op == dsf.OP_REGEX else "text (case ignored)"
            return (f'<div class="row">{h.text_input(cond.get(dsf.TEXT, ""), placeholder=placeholder, mono=True, width=320, input_=self.i("set_text", cid=cid), enter=h.on_enter(self.name, "set_text", cid=cid))}'
                    f'</div>')
        n_missing = int(series.isna().sum())
        if series.dtype == object:
            n_missing += int((series.dropna().astype(str).str.strip() == "").sum())
        return f'<div class="muted small">{n_missing:,} row{"s" if n_missing != 1 else ""} with no value in this column</div>'

    def _values_html(self, cond: dict, series: pd.Series) -> str:
        cid = cond["_cid"]
        counts = self._value_counts(cond[dsf.COLUMN])
        picked = list(cond.get(dsf.VALUES) or [])
        picked_set = set(picked)
        query = (self.searches.get(cid) or "").strip().lower()
        if query:
            index = counts.index.astype(str)
            shown = counts[index.str.lower().str.contains(query, regex=False)]
        else:
            shown = counts
        n_shown = len(shown)
        shown = shown.head(VALUES_SHOWN)
        boxes = "".join(
            f'<label class="flt-val{" on" if v in picked_set else ""}">'
            f'<input type="checkbox"{" checked" if v in picked_set else ""}{self.c("pick", cid=cid, v=str(v))}>'
            f'<span class="mono">{h.e(v)}</span><span class="muted">{int(n):,}</span></label>'
            for v, n in shown.items())
        more = ""
        if n_shown > VALUES_SHOWN:
            more = (f'<div class="muted small">{n_shown:,} values{" match" if query else ""} -- the {VALUES_SHOWN} '
                    f'most frequent are listed; type to find others.</div>')
        elif not n_shown:
            more = '<div class="muted small">No value matches.</div>'
        unseen = [v for v in picked if v not in counts.index]
        chips = ""
        # The ticked boxes already say it while every value is listed.
        if picked and (query or len(counts) > VALUES_SHOWN or unseen):
            chips = ('<div class="row-wrap flt-picked"><span class="muted small">Picked:</span>' + "".join(
                f'<span class="chip chip-x mono">{h.e(v)}<button type="button" class="chip-del" title="Remove"'
                f'{self.a("unpick", cid=cid, value=v)}>{h.icon("x", 11)}</button></span>' for v in picked[:60])
                + (f'<span class="muted small">+{len(picked) - 60} more</span>' if len(picked) > 60 else "")
                + h.button("Clear", act_attr=self.a("unpick_all", cid=cid), kind="link", small=True) + "</div>")
        warn = (f'<div class="small c-warn">{len(unseen)} picked value{"s are" if len(unseen) > 1 else " is"} '
                f'not in this column.</div>') if unseen else ""
        tools = (f'<div class="row">{h.text_input(self.searches.get(cid, ""), placeholder="Find a value", width=220, input_=self.i("search", cid=cid))}'
                 f'{h.button("Pick all listed", act_attr=self.a("pick_listed", cid=cid), small=True, disabled=not len(shown))}'
                 f'<span class="muted small">{len(counts):,} distinct value{"s" if len(counts) != 1 else ""}</span></div>')
        return f'{tools}<div class="flt-vals">{boxes}</div>{more}{chips}{warn}'

    def _query_html(self) -> str:
        full = self._full()
        error = ""
        info = ""
        if self.query.strip():
            try:
                kept = int(dsf.query_mask(full, self.query).sum())
                info = f"On its own, this query keeps {kept:,} of {len(full):,} rows."
            except dsf.FilterError as ex:
                error = str(ex)
        area = (f'<textarea class="inp ta mono flt-q" rows="3" spellcheck="false" placeholder="e.g. width &gt;= 32 and label in [\'cat\', \'dog\']"'
                f'{self.i("query")}{h.on_enter(self.name, "query_enter")} aria-label="Query">{h.e(self.query)}</textarea>')
        status = ""
        if error:
            status = f'<div class="flt-qs err">{h.icon("warn", 13)}<span>{h.e(error)}</span></div>'
        elif info:
            status = f'<div class="flt-qs ok">{h.icon("check", 13)}<span>{h.e(info)}{"" if self.query_on else " (turned off)"}</span></div>'
        examples = self._examples(full)
        example_html = ""
        if examples:
            example_html = ('<div class="row-wrap flt-ex"><span class="muted small">Examples:</span>' + "".join(
                f'<button type="button" class="chip mono"{self.a("example", text=ex)} title="Put this in the query">{h.e(ex)}</button>'
                for ex in examples) + "</div>")
        cols = ", ".join(dsf.quote_column(c) for c in self._columns())
        convert = h.button("Write the conditions as a query", act_attr=self.a("to_query"), small=True,
                           disabled=not any(dsf.condition_is_active(c) for c in self.conditions),
                           title="Replace the conditions with one query that says the same, to take it further by hand")
        right = h.checkbox("On", self.query_on, change=self.c("query_on"), title="Turn the query off without erasing it")
        body = (f'{area}{status}<div class="row" style="margin-top:6px">{convert}'
                f'<span class="muted small">Enter applies the filter.</span></div>{example_html}'
                f'<div class="muted small flt-cols">Columns: <span class="mono">{h.e(cols)}</span></div>')
        return h.section("Query", body, manual="dataset-filter-query", right=right,
                         sub="a formula, pandas-style: and / or / not, comparisons, isin, between, str.contains")

    def _examples(self, full) -> list:
        out = []
        label_col = self.state.dataset.columns.label_col
        if label_col in full.columns:
            labels = self._value_counts(label_col).index[:2].tolist()
            if labels:
                out.append(f"{dsf.quote_column(label_col)} in {labels!r}")
        numeric = [c for c in full.columns if _is_numeric(full[c])]
        if numeric:
            col = numeric[0]
            median = full[col].median()
            if pd.notna(median):
                out.append(f"{dsf.quote_column(str(col))} > {float(median):g}")
        text = [c for c in full.columns if full[c].dtype == object and c != label_col]
        crop_col = self.state.dataset.columns.crop_col
        if crop_col in text:
            out.append(f"{dsf.quote_column(crop_col)}.str.contains('png') == False")
        elif text:
            out.append(f"{dsf.quote_column(str(text[0]))}.notna()")
        return out[:4]

    def _preview_html(self) -> str:
        full = self._full()
        mask, _steps, error = self._draft_result()
        if error or mask is None:
            return ""
        rows = full[mask.to_numpy()] if self.preview == PREVIEW_KEPT else full[~mask.to_numpy()]
        seg = h.segmented(((PREVIEW_KEPT, "Kept"), (PREVIEW_REMOVED, "Removed")), self.preview,
                          act_for=lambda k: self.a("preview", which=k), label="Rows shown")
        if not len(rows):
            table = h.empty("No row is " + ("kept." if self.preview == PREVIEW_KEPT else "removed."))
        else:
            head = rows.head(PREVIEW_ROWS)
            cols = [str(c) for c in head.columns]
            ths = "".join(f'<th class="mono">{h.e(c)}</th>' for c in cols)
            trs = []
            for _, row in head.iterrows():
                cells = []
                for c in head.columns:
                    v = row[c]
                    text = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                    if len(text) > PREVIEW_COL_CHARS:
                        text = text[:PREVIEW_COL_CHARS - 1] + "…"
                    cells.append(f'<td class="mono">{h.e(text)}</td>')
                trs.append(f"<tr>{''.join(cells)}</tr>")
            table = (f'<div class="tbl-wrap flt-tbl"><table class="tbl"><thead><tr>{ths}</tr></thead>'
                     f'<tbody>{"".join(trs)}</tbody></table></div>')
        note = f'<span class="muted small">first {min(PREVIEW_ROWS, len(rows)):,} of {len(rows):,}</span>'
        return h.section("Preview", table, right=f'{note}{seg}', sub="the rows of the draft, from the file as loaded")

    # ---- actions ----

    def _cond(self, cid):
        return next((c for c in self.conditions if c["_cid"] == cid), None)

    def _changed(self) -> None:
        self._draft_cache = None

    def on_choose_column(self, value: str = "", **_):
        self.add_column = value

    def on_add(self, **_):
        full = self._full()
        if full is None or self.add_column not in full.columns:
            return
        series = full[self.add_column]
        if _is_numeric(series) and series.nunique(dropna=True) > 20:
            op = dsf.OP_RANGE
        elif not _is_numeric(series) and series.nunique(dropna=True) > 200:
            op = dsf.OP_CONTAINS
        else:
            op = dsf.OP_IN
        self.conditions.append({dsf.COLUMN: self.add_column, dsf.OP: op, dsf.ON: True, dsf.VALUES: [],
                                dsf.MIN: None, dsf.MAX: None, dsf.TEXT: "", "_cid": _new_cid()})
        self._changed()

    def on_remove(self, cid: str = "", **_):
        self.conditions = [c for c in self.conditions if c["_cid"] != cid]
        self.searches.pop(cid, None)
        self._changed()

    def on_toggle(self, cid: str = "", value=None, **_):
        cond = self._cond(cid)
        if cond is not None:
            cond[dsf.ON] = bool(value) if value is not None else not cond.get(dsf.ON, True)
            self._changed()

    def on_set_op(self, cid: str = "", value: str = "", **_):
        cond = self._cond(cid)
        if cond is not None and value in dsf.OPS:
            cond[dsf.OP] = value
            cond.setdefault(dsf.VALUES, [])
            cond.setdefault(dsf.TEXT, "")
            self._changed()

    def on_set_bound(self, cid: str = "", which: str = "", value="", **_):
        cond = self._cond(cid)
        if cond is not None and which in (dsf.MIN, dsf.MAX):
            cond[which] = dsf._number(value)
            self._changed()

    def on_set_text(self, cid: str = "", value: str = "", **_):
        cond = self._cond(cid)
        if cond is not None:
            cond[dsf.TEXT] = value or ""
            self._changed()

    def on_search(self, cid: str = "", value: str = "", **_):
        self.searches[cid] = value or ""

    def on_pick(self, cid: str = "", v: str = "", value=None, **_):
        """A value's checkbox: `value` is whether it is now ticked."""
        cond = self._cond(cid)
        if cond is None:
            return
        values = cond.setdefault(dsf.VALUES, [])
        tick = (v not in values) if value is None else bool(value)
        if not tick and v in values:
            values.remove(v)
        elif tick and v not in values and len(values) < dsf.VALUES_MAX:
            values.append(v)
        self._changed()

    def on_unpick(self, cid: str = "", value: str = "", **_):
        cond = self._cond(cid)
        if cond is not None:
            cond[dsf.VALUES] = [v for v in cond.get(dsf.VALUES, []) if v != value]
            self._changed()

    def on_unpick_all(self, cid: str = "", **_):
        cond = self._cond(cid)
        if cond is not None:
            cond[dsf.VALUES] = []
            self._changed()

    def on_pick_listed(self, cid: str = "", **_):
        cond = self._cond(cid)
        if cond is None:
            return
        counts = self._value_counts(cond[dsf.COLUMN])
        query = (self.searches.get(cid) or "").strip().lower()
        if query:
            counts = counts[counts.index.astype(str).str.lower().str.contains(query, regex=False)]
        values = cond.setdefault(dsf.VALUES, [])
        for v in counts.head(VALUES_SHOWN).index:
            if v not in values and len(values) < dsf.VALUES_MAX:
                values.append(str(v))
        self._changed()

    def on_query(self, value: str = "", **_):
        self.query = value or ""
        self._changed()

    def on_query_enter(self, value: str = "", **_):
        self.query = value or ""
        self._changed()
        if self.has_changes():
            self.on_apply()

    def on_query_on(self, value=None, **_):
        self.query_on = bool(value)
        self._changed()

    def on_example(self, text: str = "", **_):
        self.query = text
        self._changed()

    def on_to_query(self, **_):
        self.query = dsf.query_from_conditions(self.draft_spec())
        self.query_on = True
        self.conditions = []
        self._changed()

    def on_preview(self, which: str = PREVIEW_KEPT, **_):
        self.preview = which if which in (PREVIEW_KEPT, PREVIEW_REMOVED) else PREVIEW_KEPT

    def on_discard(self, **_):
        self._set_draft(self.state.dataset_filter or {})

    def on_clear(self, **_):
        self._set_draft({})
        if dsf.is_active(self.state.dataset_filter or {}) or self.state.dataset_filter_error:
            self._apply({})

    def on_apply(self, **_):
        spec = self.draft_spec()
        mask, _steps, error = self._draft_result()
        if error:
            self.s.toast(error, "error")
            return
        if mask is not None and not mask.any():
            # Every tab would be empty, and most of what they compute has
            # no meaning on zero rows.
            self.s.toast("This filter keeps no row: change it before applying.", "error")
            return
        self._apply(spec)

    def _apply(self, spec: dict) -> None:
        state = self.state
        kept, total = state.set_dataset_filter(spec)
        if state.dataset_filter_error:
            self.s.toast(f"The filter could not be applied: {state.dataset_filter_error}", "error")
        elif dsf.is_active(state.dataset_filter):
            self.s.toast(f"Filter applied: {kept:,} of {total:,} rows kept.")
        else:
            self.s.toast(f"Filter removed: all {total:,} rows are back.")
        self.s.notify_change()

    def on_data_change(self) -> None:
        self._draft_cache = None
        self._counts_cache.clear()

    # ---- snapshot ----

    def view_state(self) -> dict:
        """What is *applied* -- the draft is work in progress, and a snapshot
        restored to an unapplied draft would show numbers no tab ever had."""
        # Never {}: a snapshot taken with no filter must remove the one a
        # workspace brings, and an empty block is not saved at all.
        spec = self.state.dataset_filter or {}
        return dsf.normalize_spec(spec) if dsf.is_active(spec) else dsf.empty_spec()

    def restore_view_state(self, state) -> list:
        spec = dsf.normalize_spec(state) if isinstance(state, dict) else {}
        current = self.state.dataset_filter or {}
        if json.dumps(spec if dsf.is_active(spec) else {}, sort_keys=True) != json.dumps(
                dsf.normalize_spec(current) if dsf.is_active(current) else {}, sort_keys=True):
            self.state.set_dataset_filter(spec if dsf.is_active(spec) else {})
            self.s.notify_change()
        self._set_draft(self.state.dataset_filter or {})
        if self.state.dataset_filter_error:
            return [f"The dataset filter could not be applied: {self.state.dataset_filter_error}"]
        return []
