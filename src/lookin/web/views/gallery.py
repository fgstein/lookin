"""Gallery: a filterable mosaic of crops, rows x columns.

Same behaviour as lookin.ui.gallery_view: nothing is filtered until the
user adds a filter for a column (checkboxes for label / categorical /
binary columns, a class search past FOCUS_MAX values, a range for numeric
ones); optionally colour each thumbnail by whether an experiment's
prediction was right and filter on the predicted label; shuffle with a
seed (0 = the dataset's order); click a crop to open the shared carousel
over the page, the clicked crop first.

Building the filter panel scans every unique value of each filtered column,
so it runs in the background and a data change only marks the view stale.
Ticking filters, paging and the mosaic itself stay immediate.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lookin.core.class_focus import FOCUS_MAX
from lookin.web import html as h
from lookin.web.views.base import PromptDialog, RebuildView
from lookin.web.views.carousel import open_carousel
from lookin.web.views.media import crop_url
from lookin.web.views.widgets import ClassPicker

MAX_ROWS = 12
MAX_COLS = 12
NONE_EXPERIMENT = "__none__"
SLIDER_DIVISIONS = 100


class GalleryView(RebuildView):
    name = "gallery"
    title = "Gallery"

    def __init__(self, session):
        super().__init__(session)
        self.active_filter_cols: list = []
        # col -> {"type": "categorical", "values": [...], "selected": set(str), "picker": ClassPicker|None}
        #      | {"type": "numeric", "min", "max", "start", "end"}
        self.filter_controls: dict = {}
        self.predicted_values: list = []
        self.predicted_selected: set = set()
        self.predicted_picker = None
        self.experiment = NONE_EXPERIMENT
        self.rows_value = "4"
        self.cols_value = "6"
        self.shuffle_seed = 0
        self.page_index = 0
        self.add_choice = ""
        self._records: list = []
        self._grid_info = (0, 1)        # (total matching, total pages)
        self.grid_error = ""
        self.built = False

    def subtitle(self) -> str:
        return "Filter the dataset and browse its crops as a mosaic."

    def on_shown(self) -> None:
        if not self.built and self.state.dataset is not None:
            self.built = True
            self.rebuild(self._refresh_now)

    # ---- filter panel ----

    def _available_filter_columns(self):
        state = self.state
        if state.dataset is None:
            return []
        kinds = state.dataset.column_kinds()
        all_cols = [state.dataset.columns.label_col] + kinds.categorical + kinds.binary + kinds.numeric
        return [c for c in all_cols if c not in self.active_filter_cols]

    def _build_filter_control(self, col: str, is_label: bool, old=None) -> None:
        dataset = self.state.dataset
        df = dataset.df
        kinds = dataset.column_kinds()
        is_numeric = (not is_label) and col in kinds.numeric
        if is_numeric:
            series = df[col].dropna()
            lo, hi = float(series.min()), float(series.max())
            if lo == hi:
                hi = lo + 1.0
            self.filter_controls[col] = {"type": "numeric", "min": lo, "max": hi, "start": lo, "end": hi}
        else:
            values = sorted(df[col].dropna().unique().tolist(), key=str)
            picker = None
            if is_label and len(values) > FOCUS_MAX:
                picker = ClassPicker(self.name, "fpick", lambda picked, c=col: self._picked(c, picked),
                                     limit=None, label="Find a class to keep")
                picker.set_labels(values)
            self.filter_controls[col] = {"type": "categorical", "values": values, "selected": set(), "picker": picker}

    def _rebuild_filters(self) -> None:
        self.filter_controls = {}
        state = self.state
        if state.dataset is None:
            self.experiment = NONE_EXPERIMENT
            self.active_filter_cols = []
            self._rebuild_predicted_filter()
            return
        cols_cfg = state.dataset.columns
        valid_columns = set(state.dataset.df.columns) | {cols_cfg.label_col}
        self.active_filter_cols = [c for c in self.active_filter_cols if c in valid_columns]
        for col in self.active_filter_cols:
            self.set_progress(f"Building filters... {col}")
            self._build_filter_control(col, is_label=(col == cols_cfg.label_col))
        self.experiment = NONE_EXPERIMENT
        self._rebuild_predicted_filter()

    def _rebuild_predicted_filter(self) -> None:
        self.predicted_values = []
        self.predicted_selected = set()
        self.predicted_picker = None
        state = self.state
        if self.experiment == NONE_EXPERIMENT or state.dataset is None:
            return
        experiment = state.experiments.get(self.experiment)
        predictable = experiment.class_labels if experiment is not None else []
        labels = sorted(set(state.dataset.labels()) | {str(l) for l in predictable}, key=str)
        self.predicted_values = labels
        if len(labels) > FOCUS_MAX:
            self.predicted_picker = ClassPicker(self.name, "ppick", self._predicted_picked, limit=None,
                                                label="Find a predicted class to keep")
            self.predicted_picker.set_labels(labels)

    def _picked(self, col: str, picked: list) -> None:
        entry = self.filter_controls.get(col)
        if entry is not None:
            entry["selected"] = {str(v) for v in picked}
        self._on_filter_change()

    def _predicted_picked(self, picked: list) -> None:
        self.predicted_selected = {str(v) for v in picked}
        self._on_filter_change()

    def _refresh_now(self) -> None:
        self._rebuild_filters()
        self.page_index = 0
        self._render_grid()

    # ---- filtering and the grid ----

    def _filtered_df(self) -> pd.DataFrame:
        state = self.state
        df = state.dataset.df
        cols = state.dataset.columns
        mask = pd.Series(True, index=df.index)
        for col, entry in self.filter_controls.items():
            if col not in df.columns:
                continue
            if entry["type"] == "categorical":
                selected = [v for v in entry["values"] if str(v) in entry["selected"]]
                if selected:
                    mask &= df[col].isin(selected)
            else:
                mask &= df[col].between(entry["start"], entry["end"])
        filtered = df[mask]
        if self.experiment != NONE_EXPERIMENT and self.experiment in state.experiments:
            merged = state.merged(self.experiment)
            merged = merged[merged[cols.uuid_col].isin(filtered[cols.uuid_col])]
            selected_pred = [v for v in self.predicted_values if str(v) in self.predicted_selected]
            if selected_pred:
                merged = merged[merged["predicted_label"].isin(selected_pred)]
            return merged
        return filtered

    def _grid_shape(self) -> tuple:
        def parse(value, default, max_val):
            try:
                v = int(value)
            except (TypeError, ValueError):
                v = default
            return max(1, min(max_val, v))
        return parse(self.rows_value, 4, MAX_ROWS), parse(self.cols_value, 6, MAX_COLS)

    def _render_grid(self) -> None:
        self.grid_error = ""
        if self.state.dataset is None:
            self._records = []
            self._grid_info = (0, 1)
            return
        try:
            rows_n, cols_n = self._grid_shape()
            page_size = rows_n * cols_n
            filtered = self._filtered_df()
            if self.shuffle_seed:
                order = np.random.default_rng(self.shuffle_seed).permutation(len(filtered))
                filtered = filtered.iloc[order]
            total = len(filtered)
            total_pages = max(1, math.ceil(total / page_size))
            self.page_index = min(self.page_index, total_pages - 1)
            start = self.page_index * page_size
            self._records = filtered.iloc[start:start + page_size].to_dict("records")
            self._grid_info = (total, total_pages)
        except Exception as ex:  # noqa: BLE001
            self._records = []
            self.grid_error = f"Error rebuilding the gallery: {ex}"

    def _on_filter_change(self) -> None:
        self.page_index = 0
        self._render_grid()
        self.refresh()

    # ---- rendering ----

    def render(self) -> str:
        state = self.state
        if state.dataset is None:
            return h.banner("Load a dataset in the 'Data' tab first.", "info")
        self.on_shown()
        return (f'<div class="stack">{self.stale_html()}'
                f'<div class="gal">{self._filters_html()}<div class="gal-main stack">'
                f'{self._mosaic_controls()}{self.progress_html()}{self._grid_html()}</div></div></div>')

    def _filters_html(self) -> str:
        state = self.state
        available = self._available_filter_columns()
        if self.add_choice not in available:
            self.add_choice = available[0] if available else ""
        kinds = state.dataset.column_kinds()
        parts = [
            '<div class="muted small">Add a filter for each column you want to narrow down; nothing is filtered until you do.</div>',
            f'<div class="row">{h.select([(c, c) for c in available], self.add_choice, change=self.c("choose"), title="Add a filter for column...", cls="mono grow")}'
            f'{h.button("Add", act_attr=self.a("add"), kind="primary", icon_name="plus", disabled=not available)}</div>',
        ]
        if kinds.high_cardinality:
            parts.append(f'<div class="muted small"><i>Not offered above (too many unique values): '
                         f'{h.e(", ".join(kinds.high_cardinality))}</i></div>')
        if not self.active_filter_cols:
            parts.append(h.banner("No filters applied yet -- pick a column above and click Add.", "info"))
        for col in self.active_filter_cols:
            entry = self.filter_controls.get(col)
            head = (f'<div class="flt-h"><b class="mono">{h.e(col)}</b>'
                    f'{h.button(icon_name="x", act_attr=self.a("remove", column=col), small=True, kind="ghost", title="Remove this filter")}</div>')
            if entry is None:
                body = h.spinner("Building filters...")
            elif entry["type"] == "numeric":
                body = self._range_html(col, entry)
            elif entry["picker"] is not None:
                entry["picker"].set_picked([v for v in entry["values"] if str(v) in entry["selected"]])
                body = entry["picker"].render()
            else:
                body = self._checks_html(entry["values"], entry["selected"], "toggle", column=col)
            parts.append(f'<div class="flt" data-key="flt-{h.e(col)}">{head}{body}</div>')
        if self.experiment != NONE_EXPERIMENT and self.predicted_values:
            if self.predicted_picker is not None:
                self.predicted_picker.set_picked([v for v in self.predicted_values if str(v) in self.predicted_selected])
                body = self.predicted_picker.render()
            else:
                body = self._checks_html(self.predicted_values, self.predicted_selected, "toggle_pred")
            parts.append(f'<div class="flt"><div class="flt-h"><b>Predicted label</b></div>{body}</div>')
        return (f'<aside class="card gal-side"><div class="card-h"><h2>Filters</h2></div>'
                f'<div class="card-b stack">{"".join(parts)}</div></aside>')

    def _checks_html(self, values, selected, action: str, **kw) -> str:
        out = []
        for i, v in enumerate(values):
            out.append(h.checkbox(str(v), str(v) in selected, change=self.c(action, i=i, **kw)))
        return f'<div class="checks">{"".join(out)}</div>'

    def _range_html(self, col: str, entry: dict) -> str:
        lo, hi = entry["min"], entry["max"]
        step = (hi - lo) / SLIDER_DIVISIONS
        out_id = f"rng-{abs(hash(col)) % 10**8}"
        def rng(which, value):
            return (f'<input type="range" class="rng" min="{lo!r}" max="{hi!r}" step="{step!r}" value="{value!r}" '
                    f'data-out="{out_id}" data-which="{which}" data-force="1"'
                    f'{self.c("range", column=col, which=which)}>')
        return (f'<div class="dual" data-key="dual-{h.e(col)}">{rng("start", entry["start"])}{rng("end", entry["end"])}</div>'
                f'<div class="small mono" id="{out_id}">[{entry["start"]:.2f}, {entry["end"]:.2f}]</div>')

    def _mosaic_controls(self) -> str:
        rows_n, cols_n = self._grid_shape()
        options = [(NONE_EXPERIMENT, "(none)")] + [(n, n) for n in self.state.experiments]
        total, total_pages = self._grid_info
        shuffle = h.checkbox("Shuffle", bool(self.shuffle_seed), change=self.c("shuffle"), cls="sw")
        reshuffle = (h.button(icon_name="shuffle", act_attr=self.a("reshuffle"), small=True, kind="ghost",
                              title="Another random order") if self.shuffle_seed else "")
        controls = (f'<div class="row-wrap">'
                    f'{h.button(icon_name="sliders", act_attr=self.a("settings"), small=True, kind="ghost", title="Mosaic settings")}'
                    f'<span class="muted small">{rows_n} x {cols_n}</span><span class="vsep"></span>'
                    f'{h.select(options, self.experiment, change=self.c("experiment"), label="Color by experiment prediction")}'
                    f'<span class="vsep"></span>{shuffle}{reshuffle}</div>'
                    f'<div class="row">{h.button(icon_name="chev-left", act_attr=self.a("prev"), small=True, disabled=self.page_index <= 0, title="Previous page")}'
                    f'<span class="small">Page {self.page_index + 1} / {total_pages}  ({total} matching rows)</span>'
                    f'{h.button(icon_name="chev-right", act_attr=self.a("next"), small=True, title="Next page")}</div>')
        return h.section("Mosaic", controls)

    def _grid_html(self) -> str:
        if self.grid_error:
            return h.banner(self.grid_error, "error")
        state = self.state
        cols_cfg = state.dataset.columns
        rows_n, cols_n = self._grid_shape()
        thumb = max(56, min(150, 920 // cols_n))
        show_prediction = self.experiment != NONE_EXPERIMENT
        cells = []
        for i in range(rows_n * cols_n):
            if i >= len(self._records):
                cells.append('<div class="gcell gcell-empty"></div>')
                continue
            rec = self._records[i]
            img_path = state.dataset.resolve_crop_path(rec[cols_cfg.crop_col])
            label = rec.get(cols_cfg.label_col)
            annotated = label is not None and not pd.isna(label) and str(label) != ""
            caption = str(label) if annotated else ""
            border = ""
            if show_prediction:
                if annotated:
                    border = " ok" if rec.get("correct") else " bad"
                caption = f"{caption or '?'} -> {rec.get('predicted_label')}"
            url = crop_url(self.s, img_path, rec)
            cells.append(f'<button type="button" class="gcell{border}" {self.a("open", i=i)}>'
                         f'<img class="crop" src="{h.e(url)}" alt="" loading="lazy" style="object-fit:cover">'
                         f'<span class="gcap">{h.e(caption)}</span></button>')
        return (f'<div class="ggrid" style="grid-template-columns:repeat({cols_n},minmax(0,{thumb + 10}px))">'
                f'{"".join(cells)}</div>')

    # ---- actions ----

    def on_choose(self, value: str = "", **_):
        self.add_choice = value

    def on_add(self, **_):
        col = self.add_choice
        if not col or col in self.active_filter_cols:
            return
        self.active_filter_cols.append(col)
        self.rebuild(self._refresh_now)

    def on_remove(self, column: str = "", **_):
        if column in self.active_filter_cols:
            self.active_filter_cols.remove(column)
        self.rebuild(self._refresh_now)

    def on_toggle(self, column: str = "", i: int = 0, value=False, **_):
        entry = self.filter_controls.get(column)
        if entry is None or entry["type"] != "categorical" or not (0 <= i < len(entry["values"])):
            return
        key = str(entry["values"][i])
        (entry["selected"].add if value else entry["selected"].discard)(key)
        self._on_filter_change()

    def on_toggle_pred(self, i: int = 0, value=False, **_):
        if 0 <= i < len(self.predicted_values):
            key = str(self.predicted_values[i])
            (self.predicted_selected.add if value else self.predicted_selected.discard)(key)
            self._on_filter_change()

    def on_fpick(self, op: str = "", **args):
        # Only the label column ever has a picker.
        col = self.state.dataset.columns.label_col if self.state.dataset is not None else None
        entry = self.filter_controls.get(col)
        if entry is not None and entry.get("picker") is not None:
            entry["picker"].handle(op, args)
        self.refresh()

    def on_ppick(self, op: str = "", **args):
        if self.predicted_picker is not None:
            self.predicted_picker.handle(op, args)
        self.refresh()

    def on_range(self, column: str = "", which: str = "start", value="", **_):
        entry = self.filter_controls.get(column)
        if entry is None or entry["type"] != "numeric":
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        v = max(entry["min"], min(entry["max"], v))
        if which == "start":
            entry["start"] = min(v, entry["end"])
        else:
            entry["end"] = max(v, entry["start"])
        self._on_filter_change()

    def on_experiment(self, value: str = "", **_):
        self.experiment = value if value in self.state.experiments else NONE_EXPERIMENT
        self.page_index = 0
        self._rebuild_predicted_filter()
        self._render_grid()

    def on_settings(self, **_):
        def apply(values):
            self.rows_value = str(values.get("rows", ""))
            self.cols_value = str(values.get("cols", ""))
            self.page_index = 0
            self._render_grid()
            self.refresh()
        self.s.open_dialog(PromptDialog("Mosaic settings", [("rows", "rows", self.rows_value),
                                                            ("cols", "cols", self.cols_value)],
                                        apply, submit_label="Apply"))

    def on_shuffle(self, value=False, **_):
        self._set_shuffle(self._next_seed() if value else 0)

    def on_reshuffle(self, **_):
        self._set_shuffle(self._next_seed())

    @staticmethod
    def _next_seed() -> int:
        return int(np.random.default_rng().integers(1, 2**31 - 1))

    def _set_shuffle(self, seed: int) -> None:
        self.shuffle_seed = seed
        self.page_index = 0
        self._render_grid()

    def on_prev(self, **_):
        if self.page_index > 0:
            self.page_index -= 1
            self._render_grid()

    def on_next(self, **_):
        self.page_index += 1
        self._render_grid()

    def on_open(self, i: int = 0, **_):
        if not self._records or not (0 <= i < len(self._records)):
            return
        cols_cfg = self.state.dataset.columns
        ordered = self._records[i:] + self._records[:i]
        uuid_col = cols_cfg.uuid_col or None
        clicked_uuid = self._records[i].get(uuid_col) if uuid_col else None
        open_carousel(self.s, f"{len(ordered)} image(s) on this page", ordered, cols_cfg.crop_col, uuid_col,
                      highlight_uuid=clicked_uuid, review_source="gallery", label_col=cols_cfg.label_col)

    def on_update(self, **_):
        self.rebuild(self._refresh_now)

    # ---- the rest of the app talking to this view ----

    def on_data_change(self) -> None:
        self.dirty = True

    def on_display_changed(self) -> None:
        self.refresh()

    def apply_external_filter(self, experiment_name: str, true_label, predicted_label) -> None:
        """From a confusion-matrix cell: the crops of that cell, here."""
        state = self.state
        cols_cfg = state.dataset.columns
        self.built = True
        if self.dirty:
            self._rebuild_filters()
            self.dirty = False
        if cols_cfg.label_col not in self.active_filter_cols:
            self.active_filter_cols.append(cols_cfg.label_col)
            self._rebuild_filters()
        for entry in self.filter_controls.values():
            if entry["type"] == "categorical":
                entry["selected"] = set()
            else:
                entry["start"], entry["end"] = entry["min"], entry["max"]
        label_entry = self.filter_controls.get(cols_cfg.label_col)
        if label_entry and str(true_label) in {str(v) for v in label_entry["values"]}:
            label_entry["selected"] = {str(true_label)}
        self.experiment = experiment_name if experiment_name in state.experiments else NONE_EXPERIMENT
        self._rebuild_predicted_filter()
        if str(predicted_label) in {str(v) for v in self.predicted_values}:
            self.predicted_selected = {str(predicted_label)}
        self.page_index = 0
        self.progress_text = ""
        self._render_grid()
        self.refresh()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if self.experiment == old_name:
            self.experiment = new_name
        self.refresh()

    # ---- snapshots (see ui/view_state.py) ----

    def view_state(self) -> dict:
        filters = {}
        for col, entry in self.filter_controls.items():
            if entry["type"] == "categorical":
                filters[col] = {"kind": "categorical",
                                "selected": [str(v) for v in entry["values"] if str(v) in entry["selected"]]}
            else:
                filters[col] = {"kind": "numeric", "start": float(entry["start"]), "end": float(entry["end"])}
        return {
            "columns": list(self.active_filter_cols),
            "filters": filters,
            "experiment": self.experiment or NONE_EXPERIMENT,
            "predicted": [str(v) for v in self.predicted_values if str(v) in self.predicted_selected],
            "rows": str(self.rows_value or ""),
            "cols": str(self.cols_value or ""),
            "page": int(self.page_index),
            "shuffle": int(self.shuffle_seed),
        }

    def restore_view_state(self, state: dict) -> list:
        warnings = []
        if self.state.dataset is None:
            return ["Gallery: no dataset is loaded."]
        dataset = self.state.dataset
        valid = set(dataset.df.columns) | {dataset.columns.label_col}
        wanted = [c for c in state.get("columns", []) or [] if isinstance(c, str)]
        missing = [c for c in wanted if c not in valid]
        if missing:
            warnings.append(f"Gallery: no column named {', '.join(missing)} -- that filter was dropped.")
        self.active_filter_cols = [c for c in wanted if c in valid]
        self.dirty = False
        self.built = True
        self._rebuild_filters()
        self.progress_text = ""

        experiment = state.get("experiment") or NONE_EXPERIMENT
        if experiment != NONE_EXPERIMENT:
            if experiment in self.state.experiments:
                self.experiment = experiment
                self._rebuild_predicted_filter()
            else:
                warnings.append(f"Gallery: experiment '{experiment}' is not loaded -- colouring is off.")

        saved = state.get("filters") or {}
        for col, entry in self.filter_controls.items():
            spec = saved.get(col)
            if not isinstance(spec, dict):
                continue
            if entry["type"] == "categorical" and spec.get("kind") == "categorical":
                selected = {str(v) for v in spec.get("selected", []) or []}
                entry["selected"] = {str(v) for v in entry["values"] if str(v) in selected}
            elif entry["type"] == "numeric" and spec.get("kind") == "numeric":
                try:
                    start, end = float(spec["start"]), float(spec["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                entry["start"] = max(entry["min"], min(entry["max"], start))
                entry["end"] = max(entry["min"], min(entry["max"], end))

        predicted = {str(v) for v in state.get("predicted", []) or []}
        self.predicted_selected = {str(v) for v in self.predicted_values if str(v) in predicted}
        if state.get("rows"):
            self.rows_value = str(state["rows"])
        if state.get("cols"):
            self.cols_value = str(state["cols"])
        try:
            self.page_index = max(0, int(state.get("page", 0)))
        except (TypeError, ValueError):
            self.page_index = 0
        try:
            self.shuffle_seed = max(0, int(state.get("shuffle", 0) or 0))
        except (TypeError, ValueError):
            self.shuffle_seed = 0
        self._render_grid()
        self.refresh()
        return warnings
