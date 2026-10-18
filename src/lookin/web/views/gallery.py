"""Gallery: a filterable mosaic of crops that loads as it is scrolled.

Same behaviour as lookin.ui.gallery_view: nothing is filtered until the
user adds a filter for a column (checkboxes for label / categorical /
binary columns, a class search past FOCUS_MAX values, a range for numeric
ones); optionally colour each thumbnail by whether an experiment's
prediction was right and filter on the predicted label; shuffle with a
seed (0 = the dataset's order).

The mosaic is the shared carousel (views/carousel.py), drawn in the tab
itself: zoom steps, class badges, the metadata and review controls on every
crop, and a click shows the crop full size. It used to be a grid of its own,
and a click opened the carousel in a dialog on the very same set -- two
mosaics of the same crops. It has no pages: it opens with enough rows to
fill the window and appends the next ones as the bottom comes near (an
infinite list, see app.js).

Building the filter panel scans every unique value of each filtered column,
so it runs in the background and a data change only marks the view stale.
Ticking filters, scrolling and the mosaic itself stay immediate.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from lookin.core.class_focus import FOCUS_MAX
from lookin.web import html as h
from lookin.web.views.base import NO_RENDER, RebuildView
from lookin.web.views.carousel import Carousel
from lookin.web.views.widgets import ClassPicker

# Kept for snapshots, which have always carried "cols"; the carousel's zoom
# steps decide how many crops sit on a row now.
DEFAULT_COLS = 6
GRID_ID = "gal-grid"
# What the window has besides the mosaic: the navigation column, the
# filter column and the gaps (width); the top bar, the mosaic's toolbar and
# the paddings (height).
GALLERY_CHROME = (214 + 20 + 300 + 10 + 18, 44 + 20 + 34 + 4 + 18)
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
        self.cols_value = str(DEFAULT_COLS)
        self.shuffle_seed = 0
        self.add_choice = ""
        # The filtered (and possibly shuffled) frame the mosaic draws from,
        # and the carousel showing it -- a new one whenever the set changes.
        # `gen` is carried from one to the next, so a batch asked for by the
        # old grid is never appended to the new one.
        self._filtered = None
        self.carousel = None
        self.gen = 0
        self.grid_error = ""
        self.built = False

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

    def _total(self) -> int:
        return 0 if self._filtered is None else len(self._filtered)

    def _render_grid(self) -> None:
        """Start the mosaic over from the top of the current filtered set,
        keeping how it is shown (zoom, badges, details)."""
        self.grid_error = ""
        self.gen += 1
        previous = self.carousel
        self.carousel = None
        self._filtered = None
        if self.state.dataset is None:
            return
        try:
            filtered = self._filtered_df()
            if self.shuffle_seed:
                order = np.random.default_rng(self.shuffle_seed).permutation(len(filtered))
                filtered = filtered.iloc[order]
            self._filtered = filtered
            self.carousel = self._new_carousel(filtered, previous)
        except Exception as ex:  # noqa: BLE001
            self.carousel = None
            self.grid_error = f"Error rebuilding the gallery: {ex}"

    def _new_carousel(self, filtered, previous) -> Carousel:
        cols = self.state.dataset.columns
        carousel = Carousel(
            self, self.s, self.state.dataset, len(filtered),
            lambda start, end: filtered.iloc[start:end].to_dict("records"),
            cols.crop_col, cols.uuid_col or None, review_source="gallery", label_col=cols.label_col,
            chrome=GALLERY_CHROME, shuffle_toggle=False, outcome_borders=True, list_id=GRID_ID)
        if previous is not None:
            carousel.zoom_index = previous.zoom_index
            carousel.show_labels = previous.show_labels
            carousel.show_info = previous.show_info
            carousel.picks.update(previous.picks)
        carousel.gen = self.gen
        return carousel

    def _on_filter_change(self) -> None:
        self._render_grid()
        self._to_top()
        self.refresh()

    def _to_top(self) -> None:
        """Back to the top: the crops that were on screen are not the ones
        at that place in the new set."""
        self.s.cmd(scroll_top=f"#{GRID_ID}")

    # ---- rendering ----

    def render(self) -> str:
        state = self.state
        if state.dataset is None:
            return h.banner("Load a dataset in the 'Data' tab first.", "info")
        self.on_shown()
        return (f'<div class="stack">{self.stale_html()}'
                f'<div class="gal">{self._filters_html()}<div class="gal-main stack">'
                f'{self.progress_html()}{self._grid_html()}</div></div></div>')

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
        """The Gallery's own controls, at the start of the mosaic's toolbar."""
        options = [(NONE_EXPERIMENT, "(none)")] + [(n, n) for n in self.state.experiments]
        shuffle = h.checkbox("Shuffle", bool(self.shuffle_seed), change=self.c("shuffle"), cls="sw")
        reshuffle = (h.button(icon_name="shuffle", act_attr=self.a("reshuffle"), small=True, kind="ghost",
                              title="Another random order") if self.shuffle_seed else "")
        return (f'{h.select(options, self.experiment, change=self.c("experiment"), label="Color by experiment prediction")}'
                f'<span class="vsep"></span>{shuffle}{reshuffle}')

    def _grid_html(self) -> str:
        if self.grid_error:
            return h.banner(self.grid_error, "error")
        if self.carousel is None:
            return ""
        if not self._total():
            return (f'<div class="car"><div class="car-tb">{self._mosaic_controls()}</div>'
                    f'<div class="empty">No crop matches these filters.</div></div>')
        return f'<section class="card gal-car">{self.carousel.render(self._mosaic_controls())}</section>'

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
        self._rebuild_predicted_filter()
        self._render_grid()
        self._to_top()

    def on_shuffle(self, value=False, **_):
        self._set_shuffle(self._next_seed() if value else 0)

    def on_reshuffle(self, **_):
        self._set_shuffle(self._next_seed())

    @staticmethod
    def _next_seed() -> int:
        return int(np.random.default_rng().integers(1, 2**31 - 1))

    def _set_shuffle(self, seed: int) -> None:
        self.shuffle_seed = seed
        self._render_grid()
        self._to_top()

    # ---- the mosaic ----

    def on_car(self, op: str = "", **args):
        if self.carousel is None:
            return NO_RENDER
        return self.carousel.handle(op, args)

    def on_update(self, **_):
        self.rebuild(self._refresh_now)

    # ---- the rest of the app talking to this view ----

    def on_data_change(self) -> None:
        self.dirty = True

    def on_display_changed(self) -> None:
        # Same crops, new URLs: a new generation redraws every loaded cell.
        self.gen += 1
        if self.carousel is not None:
            self.carousel.gen = self.gen
        self.refresh()

    def on_review_change(self) -> None:
        # The decision under each crop, in details mode.
        if self.carousel is not None and self.carousel.show_info:
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
            "cols": str(self.cols_value or ""),
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
        # Snapshots from before the mosaic scrolled also carry "rows" and
        # "page"; there is nothing left for either to set.
        if state.get("cols"):
            self.cols_value = str(state["cols"])
        try:
            self.shuffle_seed = max(0, int(state.get("shuffle", 0) or 0))
        except (TypeError, ValueError):
            self.shuffle_seed = 0
        self._render_grid()
        self.refresh()
        return warnings
