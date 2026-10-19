"""Embeddings: the 2D projection of an experiment's embeddings -- the port
of lookin.ui.projection_view.

Two renderings of the very same projected coordinates: the interactive
map (drawn in the browser by static/js/projection.js: hover for the crop,
click to open it, lasso a group, wheel / box zoom, legend filters in place)
and the static matplotlib figure, the reference. A run computes the
projection once; colour, display filters and overlays only redraw. The last
run of each experiment is kept and put back when it is selected again.
Class overlap is measured below the settings.

Every key the original compares is kept as is: projection methods (`pca`,
`tsne`, `umap`, `umap_circle`), the positional colour keys ("0", "1", ...,
`source`), PROTO_FROM_* (`model`, `dataset`), the overlap focus criteria.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lookin.core import class_focus as cf
from lookin.core import embedding_analysis as ea
from lookin.core import point_coloring as pc
from lookin.core import point_filters as pf
from lookin.core.chart_style import short_name, subplots
from lookin.core.class_focus import FOCUS_MAX
from lookin.core.data_manager import DATASET_SOURCE_GROUP
from lookin.ui.class_matrix import short_column_labels
from lookin.ui.components import INLINE_DPI, blues_color, crop_display
from lookin.ui.embedding_context import build_analysis_context, coverage_note, experiments_with_embeddings
from lookin.ui.interactive_projection import MIN_PLOT_SIZE, POINT_KIND_DISC, POINT_KIND_SQUARE
from lookin.ui.projection_view import (
    CHROME_HEIGHT, CLASS_COLOR_KEY, CLASS_COLORING_LABEL, MAP_MAX, OVERLAP_CELL, OVERLAP_FOCUS_CRITERIA,
    OVERLAP_LABEL_WIDTH, PROTO_CLASSES_MAX, PROTO_FROM_DATASET, PROTO_FROM_MODEL, SOURCE_COLOR_KEY,
    SOURCE_COLORING_LABEL, SUBCENTER_MARKERS, _OverlayLayer, _RunResult, overlay_dimension_problem,
    overlay_indices,
)
from lookin.ui.prototype_source import KMEANS_COLOR, MODEL_COLOR
from lookin.web import html as h
from lookin.web.session import checkpoint
from lookin.web.views.base import StaleTracking, View
from lookin.web.views.carousel import open_carousel
from lookin.web.views.files import menu
from lookin.web.views.media import fig_html
from lookin.web.views.widgets import ClassFocus, ClassPicker

METHOD_OPTIONS = [("pca", "PCA"), ("tsne", "t-SNE"), ("umap", "UMAP"), ("umap_circle", "UMAP (spherical)")]
METHOD_MANUAL_PAGES = {"pca": "pca", "tsne": "tsne", "umap": "umap", "umap_circle": "umap"}


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class EmbeddingsView(StaleTracking, View):
    name = "embeddings"
    title = "Embeddings"
    # a map drawn from data that has since changed: run the same projection
    # again (StaleTracking); the controls follow the data by themselves
    stale_job = "Projection"
    stale_message = "Data has changed -- this map is out of date."
    updating_message = "Data has changed -- the map is being computed again."
    cancelled_message = "The update was cancelled -- this map is from the previous data."

    def __init__(self, session):
        super().__init__(session)
        self.init_stale()
        self._results_before = ""     # what the map area showed before a run, for a Cancel
        self._projection_generation = 0
        self._has_result = False
        self._last_result = None
        self._computing = False
        self._runs: dict = {}
        self._last_context = None
        self._runs_epoch = 0

        self.experiment = None
        self.method = "pca"
        self.sample_value = "1500"
        self.color_key = CLASS_COLOR_KEY
        self._color_column_map: dict = {CLASS_COLOR_KEY: None}
        self._color_options: list = [(CLASS_COLOR_KEY, "Class (true label)")]
        self._score_note = ""
        self._colorable: list = []
        self._color_options_key = None
        self._overlay_selection: set = set()
        self._overlay_problems: dict = {}
        self._refit_choice = False

        self._filter_columns: list = []
        self._filters: dict = {}
        self._pending_filters: dict = {}
        self._pending_color_column = ""
        self._column_kind: dict = {}

        self._class_filter: list = []
        self._many_classes = False
        self._classes: list = []
        self.class_picker = ClassPicker(self.name, "cpick", self._set_class_filter, limit=None,
                                        label="Restrict to classes")
        self._proto_filter = None
        self._proto_filter_chosen = False
        self.proto_picker = ClassPicker(self.name, "ppick", self._set_proto_filter, limit=None,
                                        label="Show prototypes of")
        self._proto_source_choice: dict = {}
        self._proto_source_options: list = []
        self._proto_source_note = ""

        # what is on screen
        self.results_html = h.banner("Click 'Run projection' to compute.", "info")
        self.legend_html = ""
        self.notes_html = ""
        self.status_html = ""
        self.controls_ready = False
        self._map_records: list = []
        self._map_uuids = None
        self._map_n_drawn = 0

        # class overlap
        self.overlap_busy = False
        self.overlap_html = ""
        self._overlap = None
        self.overlap_focus = ClassFocus(self.name, "ofocus", OVERLAP_FOCUS_CRITERIA, self._on_overlap_focus_change,
                                        what="the overlap grid", support_criteria=(cf.FOCUS_LEAST_SEPARATED,))
        self.built = False
        session.prototype_source.subscribe(self.on_prototype_source_changed)

    # ---- lifecycle ----

    def on_shown(self) -> None:
        if not self.built:
            self.built = True
            self.refresh_controls()

    def on_data_change(self) -> None:
        if self.built:
            self.refresh_controls()
        if self._has_result or self._last_result is not None:
            # every cached projection came from the old rows
            self._clear_runs()
            self.dirty = True

    def can_update(self) -> bool:
        return bool(self.experiment) and (self._has_result or self._last_result is not None)

    def on_update(self, **_):
        self.on_run()

    def on_cancelling(self) -> None:
        # the map that was there, under the "cancelled" banner
        self.results_html = self._results_before
        self._computing = False

    def refresh_controls(self) -> None:
        """The original's refresh(): the experiment list, then everything
        around the map (class / colour / filter / overlay controls), which
        needs the joined frame -- so off the request thread."""
        def work():
            self._refresh_experiment_dropdown()
            self._render()
        self.s.job("Embeddings controls").run(lambda: (work(), self.refresh()),
                                              on_error=lambda ex: (traceback.print_exc(), self.refresh()))

    def _refresh_experiment_dropdown(self) -> None:
        names = experiments_with_embeddings(self.state)
        if self.experiment not in names:
            self.experiment = names[0] if names else None
        for gone in [n for n in self._runs if n not in names]:
            del self._runs[gone]

    def _analysis_context(self, exp_name: str):
        context = build_analysis_context(self.state, exp_name)
        self._last_context = context
        return context

    def _render(self, ctx=None) -> None:
        state = self.state
        self.status_html = ""
        if state.dataset is None or not experiments_with_embeddings(state):
            self.controls_ready = False
            self.status_html = h.banner(
                "Attach an embeddings .npz sidecar to at least one loaded experiment in the Data tab "
                "(uuid + embeddings arrays) to unlock this tab.", "info")
            return
        self.controls_ready = True
        if not self.experiment:
            self._refresh_experiment_dropdown()
        exp_name = self.experiment
        if not exp_name:
            return
        if ctx is None:
            ctx = self._analysis_context(exp_name)
        if ctx is None:
            return
        self._last_context = ctx
        self._rebuild_class_options(ctx)
        self._rebuild_color_options(ctx)
        self._apply_pending_color()
        self._apply_pending_filters()
        self._prune_filters()
        self._prune_overlays()
        status = f'<div class="muted small">{len(ctx["embeddings"]):,} embeddings loaded for {h.name(exp_name)}</div>'
        note = coverage_note(ctx, state, exp_name)
        if note:
            status += h.banner(note, "warn")
        self.status_html = status

    def on_prototype_source_changed(self) -> None:
        had_cached = self._has_result or bool(self._runs)
        self._clear_runs()
        if had_cached:
            self._reset_result("Embeddings or prototype source changed — run the projection again.")
        else:
            self._reset_result()
        if self.built:
            self.refresh_controls()

    def _clear_runs(self) -> None:
        self._runs.clear()
        self._runs_epoch += 1

    def _reset_result(self, message: str = "Click 'Run projection' to compute.") -> None:
        self._projection_generation += 1
        self._has_result = False
        self._last_result = None
        self.results_html = h.banner(message, "info")
        self.legend_html = ""
        self.notes_html = ""
        self.refresh()

    # ---- classes and prototypes ----

    def _rebuild_class_options(self, ctx: dict) -> None:
        classes = sorted(set(ctx["labels"].tolist()))
        present = set(classes)
        self._classes = classes
        self._class_filter = [c for c in self._class_filter if c in present]
        self._many_classes = len(classes) > FOCUS_MAX
        if not self._proto_filter_chosen:
            self._proto_filter = [] if self._many_classes else None
        elif self._proto_filter is not None:
            self._proto_filter = [c for c in self._proto_filter if c in present]
        self.proto_picker.set_labels(classes)
        self._sync_proto_source(ctx["exp"].name, classes)
        self.class_picker.set_labels(classes)
        self.class_picker.set_picked(self._class_filter)

    def _set_class_filter(self, classes: list) -> None:
        self._class_filter = list(classes)
        if self._last_context is not None and self._many_classes:
            self._rebuild_color_options(self._last_context)
        self.class_picker.set_picked(self._class_filter)
        self._reset_result("Class selection changed — run the projection again.")

    def _set_proto_filter(self, classes) -> None:
        self._proto_filter = None if classes is None else list(classes)
        self._proto_filter_chosen = True
        self._reset_result("Prototype selection changed — run the projection again.")

    def _proto_coverage(self, exp_name: str, classes: list):
        model = self.s.prototype_source.model_set_for_dataset(exp_name)
        if model is None:
            return None
        index = model.label_index()
        return sum(1 for c in classes if c in index and not np.isnan(model.vectors[index[c]]).all())

    def _default_proto_source(self, exp_name: str, classes: list):
        covered = self._proto_coverage(exp_name, classes)
        if covered is None:
            source, _ = self.s.prototype_source.resolve(exp_name)
            return PROTO_FROM_DATASET if source == "kmeans" else None
        return PROTO_FROM_MODEL if covered == len(classes) else PROTO_FROM_DATASET

    def _proto_source_for(self, exp_name: str, classes: list):
        chosen = self._proto_source_choice.get(exp_name)
        if chosen == PROTO_FROM_MODEL and self._proto_coverage(exp_name, classes) is None:
            chosen = None
        return chosen or self._default_proto_source(exp_name, classes)

    def _sync_proto_source(self, exp_name: str, classes: list) -> None:
        covered = self._proto_coverage(exp_name, classes)
        options = []
        if covered is not None:
            options.append((PROTO_FROM_MODEL, "Network (imported)"))
        options.append((PROTO_FROM_DATASET, "Dataset labels"))
        self._proto_source_options = options
        if covered is None:
            self._proto_source_note = ("No network prototypes loaded for this experiment -- pick 'Dataset labels' to draw "
                                       "one per class.")
        elif covered < len(classes):
            self._proto_source_note = (f"The network's prototypes cover {covered:,} of the dataset's {len(classes):,} classes "
                                       f"(after the Data tab's class mapping).")
        else:
            self._proto_source_note = ""

    def _proto_summary(self) -> str:
        picked = self._proto_filter
        if picked is None:
            return "every class" + (f" (the first {PROTO_CLASSES_MAX} alphabetically)" if self._many_classes else "")
        if not picked:
            return "none"
        return f"{len(picked)} class(es)"

    # ---- colours ----

    def _rebuild_color_options(self, ctx: dict) -> None:
        state = self.state
        cols = state.dataset.columns
        exp = ctx["exp"]
        merged = ctx["merged"]
        inferences = state.inferences
        cache_key = (exp.name, len(merged), tuple(merged.columns),
                     tuple((name, len(i.df), tuple(i.df.columns)) for name, i in inferences.items()))
        if cache_key != self._color_options_key:
            self._colorable = pc.colorable_columns(
                merged, exclude={cols.uuid_col, cols.crop_col, cols.label_col, exp.uuid_col},
                score_cols=exp.score_cols)
            extras = []
            for inference in inferences.values():
                offered = pc.colorable_columns(inference.df, exclude={inference.uuid_col, inference.crop_col})
                extras += [c for c in offered if c not in self._colorable and c not in extras]
            scores = [c for c in exp.score_cols if c in self._colorable]
            self._colorable = [c for c in self._colorable if c not in scores] + extras + scores
            self._color_options_key = cache_key
        previous_key = self.color_key or CLASS_COLOR_KEY
        previous = self._color_column_map.get(previous_key)
        self._color_column_map = {CLASS_COLOR_KEY: None}
        options = [(CLASS_COLOR_KEY, "Class (true label)")]
        if inferences:
            self._color_column_map[SOURCE_COLOR_KEY] = None
            options.append((SOURCE_COLOR_KEY, "Source (dataset / inference)"))
        offered = self._colorable
        score_note = ""
        if len(exp.score_cols) > FOCUS_MAX:
            wanted = set(self._class_filter)
            allowed = {col for col, label in zip(exp.score_cols, exp.class_labels) if label in wanted}
            score_set = set(exp.score_cols)
            hidden = sum(1 for c in offered if c in score_set and c not in allowed)
            offered = [c for c in offered if c not in score_set or c in allowed]
            if hidden:
                score_note = (f"Class scores: restrict to classes to color by one ({hidden:,} not listed)"
                              if not allowed else f"Class scores: only the {len(allowed)} restricted classes")
        for i, column in enumerate(offered, start=1):
            self._color_column_map[str(i)] = column
            options.append((str(i), column))
        self._color_options = options
        self._score_note = score_note
        if previous_key in (CLASS_COLOR_KEY, SOURCE_COLOR_KEY) and previous_key in self._color_column_map:
            self.color_key = previous_key
        else:
            self.color_key = next((key for key, column in self._color_column_map.items()
                                   if column is not None and column == previous), CLASS_COLOR_KEY)

    def _apply_pending_color(self) -> None:
        wanted = self._pending_color_column
        if not wanted:
            return
        self._pending_color_column = ""
        for key, column in self._color_column_map.items():
            if column == wanted:
                self.color_key = key
                return

    def _column_reference(self, ctx: dict, column: str):
        merged = ctx["merged"]
        parts = [merged[column]] if column in merged.columns else []
        parts += [inference.df[column] for inference in self.state.inferences.values()
                  if column in inference.df.columns]
        if not parts:
            return pd.Series([np.nan] * len(merged))
        return pd.concat([p.reset_index(drop=True) for p in parts], ignore_index=True)

    def _plotted_values(self, result, layers: list, column: str):
        merged = result.ctx["merged"]
        series = merged[column] if column in merged.columns else pd.Series([np.nan] * len(merged))
        pool = series.iloc[result.idx].reset_index(drop=True)
        if not layers:
            return pool
        return pd.concat([pool] + self._overlay_values(layers, column), ignore_index=True)

    def _kind_of(self, column: str, reference) -> str:
        forced = self._column_kind.get(column)
        if forced in (pc.KIND_CATEGORICAL, pc.KIND_NUMERIC):
            return forced
        return pc.default_kind(reference)

    def _set_column_kind(self, column: str, kind: str) -> None:
        if self._column_kind.get(column) == kind:
            return
        self._column_kind[column] = kind
        if column in self._filters:
            self._filters[column] = self._build_filter(column)
        self._redraw_current("Recoloring...")

    def _color_scale_kind(self):
        """(column, kind) when the coloured column could be read either way, else None."""
        column = self._color_column_map.get(self.color_key or CLASS_COLOR_KEY)
        ctx = self._last_context
        if column is None or ctx is None:
            return None
        reference = self._column_reference(ctx, column)
        if not pc.can_be_continuous(reference):
            return None
        return column, self._kind_of(column, reference)

    # ---- display filters ----

    def _filterable_columns(self) -> list:
        state = self.state
        label_col = state.dataset.columns.label_col if state.dataset else None
        columns = [label_col] if label_col else []
        return columns + [c for c in self._colorable if c != label_col]

    def _build_filter(self, column: str):
        ctx = self._last_context
        if ctx is None:
            return None
        reference = self._column_reference(ctx, column)
        return pf.build_filter(column, reference, self._kind_of(column, reference))

    def _toggle_filter_column(self, column: str) -> None:
        if column in self._filter_columns:
            self._filter_columns.remove(column)
            self._filters.pop(column, None)
        else:
            column_filter = self._build_filter(column)
            if column_filter is None:
                return
            self._filter_columns.append(column)
            self._filters[column] = column_filter
        self._redraw_current("Filtering...")

    def _apply_pending_filters(self) -> None:
        if not self._pending_filters:
            return
        available = set(self._filterable_columns())
        for column in list(self._filter_columns):
            spec = self._pending_filters.pop(column, None)
            if spec is None:
                continue
            if column not in available:
                self._filter_columns.remove(column)
                continue
            kind = spec.get("kind")
            if kind in (pc.KIND_CATEGORICAL, pc.KIND_NUMERIC):
                self._column_kind[column] = kind
            column_filter = self._build_filter(column)
            if column_filter is None:
                self._filter_columns.remove(column)
                continue
            column_filter.hidden = {v for v in column_filter.values
                                    if str(v) in {str(x) for x in spec.get("hidden", []) or []}}
            window = spec.get("window")
            if isinstance(window, (list, tuple)) and len(window) == 2:
                try:
                    low, high = float(window[0]), float(window[1])
                    lo_bound, hi_bound = column_filter.bounds
                    column_filter.window = (max(lo_bound, min(hi_bound, low)), max(lo_bound, min(hi_bound, high)))
                except (TypeError, ValueError):
                    pass
            column_filter.include_missing = bool(spec.get("include_missing", True))
            self._filters[column] = column_filter
        self._pending_filters.clear()

    def _prune_filters(self) -> None:
        available = set(self._filterable_columns())
        for column in [c for c in self._filter_columns if c not in available]:
            self._filter_columns.remove(column)
            self._filters.pop(column, None)

    def _filter_mask(self, result, layers: list, n_points: int) -> np.ndarray:
        active = [self._filters[c] for c in self._filter_columns if c in self._filters and self._filters[c].is_active()]
        if not active:
            return np.ones(n_points, dtype=bool)
        values = {f.column: self._plotted_values(result, layers, f.column) for f in active}
        return pf.combined_mask(active, values, n_points)

    # ---- overlays ----

    def _prune_overlays(self) -> None:
        inferences = self.state.inferences
        kept = {n for n in self._overlay_selection if n in inferences}
        dropped = self._overlay_selection - kept
        self._overlay_selection = kept
        if dropped and self._last_result is not None:
            for name in dropped:
                self._last_result.overlay_proj.pop(name, None)
                self._last_result.overlay_rows.pop(name, None)
            self._redraw_current("Updating the map...")

    def _refit_forced(self) -> bool:
        return self.method not in ea.OUT_OF_SAMPLE_METHODS

    def _refit_active(self) -> bool:
        return True if self._refit_forced() else bool(self._refit_choice)

    def _selected_overlay_names(self) -> list:
        return [n for n in self.state.inferences if n in self._overlay_selection]

    def _overlay_changed(self) -> None:
        if not self._has_result and self._last_result is None:
            return
        if self._refit_active():
            self._reset_result("Overlay changed — run the projection again to fit it in.")
            return
        self._redraw_current("Placing points...")

    def _overlay_layers(self, result) -> list:
        state = self.state
        cols = state.dataset.columns
        dim = int(result.ctx["embeddings"].shape[1])
        layers = []
        self._overlay_problems = {}
        for name in self._selected_overlay_names():
            inference = state.inferences.get(name)
            if inference is None or not len(inference.df):
                continue
            problem = overlay_dimension_problem(inference, dim, result.exp_name)
            if problem:
                _log(f"overlay '{name}' skipped: {problem}")
                self._overlay_problems[name] = problem
                continue
            rows = result.overlay_rows.get(name)
            coords = result.overlay_proj.get(name)
            if coords is None or rows is None:
                fit = result.fit
                if fit is None or not fit.supports_transform():
                    continue
                rows = overlay_indices(len(inference.df))
                t0 = time.monotonic()
                try:
                    coords = fit.transform(inference.embeddings[rows])
                except Exception as ex:  # noqa: BLE001
                    _log(f"could not place '{name}' onto the {result.method} fit: {ex}")
                    traceback.print_exc()
                    self._overlay_problems[name] = (
                        f"'{name}' could not be placed onto this {result.method} fit ({ex}). "
                        f"Refitting the map on these points is the way in.")
                    continue
                _log(f"placed {len(rows):,} points of '{name}' onto the existing {result.method} fit "
                     f"in {time.monotonic() - t0:.1f}s")
                result.overlay_rows[name] = rows
                result.overlay_proj[name] = coords
            layers.append(_OverlayLayer(
                name=name, coords=np.asarray(coords, dtype=float), uuids=inference.uuids()[rows],
                records=inference.display_records(cols.uuid_col, cols.crop_col, rows=rows),
                rows=rows, total=len(inference.df)))
        return layers

    def _overlay_values(self, layers: list, column) -> list:
        out = []
        for layer in layers:
            inference = self.state.inferences.get(layer.name)
            if column is not None and inference is not None and column in inference.df.columns:
                out.append(inference.df[column].iloc[layer.rows].reset_index(drop=True))
            else:
                out.append(pd.Series([np.nan] * len(layer.coords)))
        return out

    def _fit_note(self, result, layers: list, n_fitted: int) -> str:
        selected = self._selected_overlay_names()
        if not selected:
            return ""
        placed = ", ".join(f"'{layer.name}'" for layer in layers)
        total = sum(len(layer.coords) for layer in layers)
        thinned = [f"'{layer.name}' thinned to {len(layer.coords):,} of {layer.total:,}"
                   for layer in layers if len(layer.coords) < layer.total]
        missing = [n for n in selected if n not in {layer.name for layer in layers}]
        if layers and result.refit_with_overlay and not set(result.overlay_names) ^ {l.name for l in layers}:
            note = (f"This projection was fitted on {n_fitted:,} points of '{result.exp_name}' together "
                    f"with {total:,} points from {placed} — every position accounts for both.")
        elif layers:
            approximate = " UMAP places them approximately, from the neighbours it was fitted on." \
                if result.method in ("umap", "umap_circle") else ""
            note = (f"This projection was fitted on {n_fitted:,} points of '{result.exp_name}'; "
                    f"{total:,} points from {placed} are only projected onto that fit, not fitted "
                    f"with it.{approximate}")
        else:
            note = ""
        if thinned:
            note += " " + "; ".join(thinned) + "."
        for name in missing:
            note += " " + (self._overlay_problems.get(name)
                           or f"'{name}' could not be placed on this map — run the projection "
                              f"again to fit it in.")
        return note.strip()

    # ---- running a projection ----

    def _sample_size(self) -> int:
        try:
            return int(self.sample_value)
        except ValueError:
            return 1500

    def on_run(self, **_):
        exp_name = self.experiment
        if not exp_name:
            return
        self._projection_generation += 1
        generation = self._projection_generation
        method = self.method
        n_sample = self._sample_size()
        if not self._computing:
            self._results_before = self.results_html
        self.cancelled = False
        self.results_html = h.spinner(f"Computing {method}...")
        self._computing = True
        class_filter = list(self._class_filter)
        epoch = self._runs_epoch
        overlays = self._selected_overlay_names()
        refit = self._refit_active()
        proto_filter = None if self._proto_filter is None else list(self._proto_filter)

        def work():
            try:
                ctx = self._analysis_context(exp_name)
                if ctx is None:
                    self._computing = False
                    self.results_html = h.banner(f"'{exp_name}' has no embeddings attached.", "info")
                    return
                checkpoint()
                proto_source = self._proto_source_for(exp_name, sorted(set(ctx["labels"].tolist())))
                self._run_projection(ctx, exp_name, method, n_sample, generation, class_filter, epoch,
                                     overlays, refit, proto_filter, proto_source)
                self.dirty = False
            finally:
                self.refresh()                # a Cancel included

        self.s.job("Projection").run(work, on_error=lambda ex: self._fail(ex, generation))

    def _fail(self, ex, generation) -> None:
        traceback.print_exc()
        self._computing = False
        if generation == self._projection_generation:
            self.results_html = h.banner(f"Error: {ex}", "error")
        self.refresh()

    def _run_projection(self, ctx, exp_name, method, n_sample, generation, class_filter, epoch,
                        overlay_names=None, refit_with_overlay=False, proto_filter=None,
                        proto_source=PROTO_FROM_MODEL) -> None:
        try:
            rng = np.random.default_rng(0)
            all_classes = sorted(set(ctx["labels"].tolist()))
            pool = np.arange(len(ctx["embeddings"]))
            if class_filter:
                pool = np.flatnonzero(np.isin(ctx["labels"], class_filter))
            if not len(pool):
                self.results_html = h.banner("No rows in the selected classes.", "warn")
                return
            idx = rng.choice(pool, size=min(n_sample, len(pool)), replace=False)
            embs = ctx["embeddings"][idx]
            labels = ctx["labels"][idx]
            proto_vecs = np.zeros((0, embs.shape[1]))
            proto_labels: list = []
            proto_ks: list = []
            n_subcenters = 0
            source_key = source_color = None
            source_note = ""
            registry = self.s.prototype_source
            proto_set = None
            if proto_source == PROTO_FROM_MODEL:
                proto_set = registry.model_set_for_dataset(exp_name)
                source_color, source_note = MODEL_COLOR, "network prototypes (imported)"
            if proto_set is None and proto_source == PROTO_FROM_DATASET:
                proto_set, source_note = registry.dataset_set(exp_name, ctx["embeddings"], ctx["labels"])
                proto_source, source_color = PROTO_FROM_DATASET, KMEANS_COLOR
            if proto_set is not None:
                source_key = proto_source
                wanted = all_classes if proto_filter is None else [l for l in all_classes if l in set(proto_filter)]
                present = [l for l in wanted if l in proto_set.label_index()][:PROTO_CLASSES_MAX]
                rows = []
                for l in present:
                    vectors = proto_set.vectors[proto_set.label_index()[l]]
                    for k_idx in range(vectors.shape[0]):
                        if np.isnan(vectors[k_idx]).any():
                            continue
                        rows.append(vectors[k_idx])
                        proto_labels.append(l)
                        proto_ks.append(k_idx)
                if rows:
                    proto_vecs = np.stack(rows)
                    n_subcenters = max(proto_ks) + 1
            state = self.state
            overlay_names = [n for n in (overlay_names or []) if n in state.inferences]
            overlay_rows: dict = {}
            fit_input = embs
            if refit_with_overlay and overlay_names:
                parts = [embs]
                for name in overlay_names:
                    inference = state.inferences[name]
                    problem = overlay_dimension_problem(inference, embs.shape[1], exp_name)
                    if problem:
                        _log(f"overlay '{name}' left out of the fit: {problem}")
                        continue
                    rows = overlay_indices(len(inference.df))
                    overlay_rows[name] = rows
                    parts.append(inference.embeddings[rows])
                fit_input = np.concatenate(parts, axis=0)
            checkpoint()                  # a Cancel lands here, or once the fit is done
            t0 = time.monotonic()
            fit = ea.fit_projection(fit_input, method=method, proto_vectors=proto_vecs if len(proto_vecs) else None)
            _log(f"projection ({method}) for '{exp_name}' computed in {time.monotonic() - t0:.1f}s "
                 f"over {len(fit_input):,} vectors")
            checkpoint()
            proj, proto_proj = fit.points[: len(embs)], fit.proto_points
            overlay_proj: dict = {}
            cursor = len(embs)
            for name in overlay_rows:
                count = len(overlay_rows[name])
                overlay_proj[name] = fit.points[cursor: cursor + count]
                cursor += count
            result = _RunResult(
                ctx=ctx, exp_name=exp_name, method=method, idx=idx, proj=proj, labels=labels,
                all_classes=all_classes, class_filter=class_filter, pool_size=len(pool),
                proto_filter=None if proto_filter is None else list(proto_filter),
                proto_proj=proto_proj, proto_labels=proto_labels, proto_ks=proto_ks,
                n_subcenters=n_subcenters, source_key=source_key, source_color=source_color,
                source_note=source_note, n_sample=n_sample, sample_uuids=ctx["uuids"][idx],
                run_generation=generation, fit=fit,
                overlay_names=list(overlay_rows) if refit_with_overlay else [],
                refit_with_overlay=bool(refit_with_overlay and overlay_rows),
                overlay_proj=overlay_proj, overlay_rows=overlay_rows)
        except Exception as ex:  # noqa: BLE001
            _log(f"error computing projection: {ex}")
            traceback.print_exc()
            if generation == self._projection_generation:
                self.results_html = h.banner(f"Error: {ex}", "error")
            return
        finally:
            self._computing = False
        if epoch == self._runs_epoch:
            previous = self._runs.get(exp_name)
            if previous is None or previous.run_generation < generation:
                self._runs[exp_name] = replace(result, ctx=None)
        if generation != self._projection_generation:
            _log(f"projection ({method}) for '{exp_name}' kept but not shown -- superseded by a newer request")
            return
        self._last_result = result
        self._render_projection(result, generation)

    def _redraw_current(self, message: str) -> None:
        if self._computing or self._last_result is None:
            self.refresh()
            return
        self._projection_generation += 1
        generation = self._projection_generation
        self.results_html = h.spinner(message)
        result = self._last_result
        self.s.job("Projection").run(lambda: (self._render_projection(result, generation), self.refresh()),
                                     on_error=lambda ex: self._fail(ex, generation))

    def _rebind(self, cached, ctx):
        if cached is None:
            return None
        uuids = ctx["uuids"]
        idx = cached.idx
        if not len(idx) or idx.max() >= len(uuids):
            return None
        if not np.array_equal(uuids[idx], cached.sample_uuids):
            return None
        labels = ctx["labels"]
        all_classes = sorted(set(labels.tolist()))
        if [c for c in cached.class_filter if c not in all_classes]:
            return None
        pool_size = int(np.isin(labels, cached.class_filter).sum()) if cached.class_filter else len(labels)
        return replace(cached, ctx=ctx, labels=labels[idx], all_classes=all_classes, pool_size=pool_size)

    # ---- drawing a run ----

    def _map_size(self) -> int:
        width, height = self.s.viewport
        pane = width * 0.5 - 40
        return int(max(MIN_PLOT_SIZE, min(MAP_MAX, pane, height - CHROME_HEIGHT)))

    def _color_mapping(self, result, layers=None, mask=None) -> tuple:
        ctx, idx = result.ctx, result.idx
        layers = layers or []
        n_overlay = sum(len(layer.coords) for layer in layers)
        key = self.color_key or CLASS_COLOR_KEY

        def with_overlay(pool_values, overlay_values):
            values = pd.Series(pool_values).reset_index(drop=True)
            if layers:
                values = pd.concat([values] + list(overlay_values), ignore_index=True)
            if mask is None:
                return values
            return values[mask].reset_index(drop=True)

        if key == SOURCE_COLOR_KEY:
            names = [DATASET_SOURCE_GROUP] * len(idx)
            for layer in layers:
                names += [layer.name] * len(layer.coords)
            shown = pd.Series(names, dtype=object)
            if mask is not None:
                shown = shown[mask].reset_index(drop=True)
            return pc.build_mapping(shown, SOURCE_COLORING_LABEL, kind=pc.KIND_CATEGORICAL), False
        column = self._color_column_map.get(key)
        if column is not None:
            reference = self._column_reference(ctx, column)
            values = self._plotted_values(result, layers, column)
            if mask is not None:
                values = values[mask].reset_index(drop=True)
            return pc.build_mapping(values, column, reference=reference, kind=self._kind_of(column, reference)), False
        labels = pd.Series(ctx["labels"])
        no_labels = [pd.Series([None] * n_overlay, dtype=object)] if layers else []
        reference = labels if not result.class_filter else labels[labels.isin(result.class_filter)]
        mapping = pc.build_mapping(with_overlay(labels.iloc[idx], no_labels), CLASS_COLORING_LABEL,
                                   reference=reference, kind=pc.KIND_CATEGORICAL)
        return mapping, True

    def _render_projection(self, result, generation: int) -> None:
        try:
            layers = self._overlay_layers(result)
            n_drawn = len(result.proj)
            proj = result.proj if not layers else np.vstack([result.proj] + [layer.coords for layer in layers])
            is_overlay = np.zeros(len(proj), dtype=bool)
            is_overlay[n_drawn:] = True
            total = len(proj)
            shown = self._filter_mask(result, layers, total)
            n_hidden = int((~shown).sum())
            if n_hidden:
                proj = proj[shown]
                is_overlay = is_overlay[shown]
                n_drawn = int(shown[:n_drawn].sum())
            proto_proj, proto_labels, proto_ks = result.proto_proj, result.proto_labels, result.proto_ks
            n_subcenters = result.n_subcenters
            mapping, colors_by_class = self._color_mapping(result, layers, shown if n_hidden else None)
            _log(f"rendering '{result.exp_name}' colored by {mapping.column} "
                 f"({mapping.kind}, {len(mapping.legend)} legend entries)"
                 + (f", with {len(proj) - n_drawn:,} overlaid points from {len(layers)} inference(s)" if layers else "")
                 + (f", {n_hidden:,} points hidden by {len(self._filter_columns)} filter(s)" if n_hidden else ""))
            map_size = self._map_size()
            static_html = self._static_figure(result, mapping, colors_by_class, proj, is_overlay, proto_proj,
                                              proto_labels, proto_ks, n_subcenters, layers, n_drawn, map_size)
            if static_html is None or generation != self._projection_generation:
                return

            ctx = result.ctx
            proto_colors = None
            if proto_proj is not None and len(proto_proj):
                proto_colors = ([mapping.color_of(l) for l in proto_labels] if colors_by_class
                                else [result.source_color] * len(proto_labels))
            uuids_all = ctx["uuids"][result.idx]
            records_all = [ctx["row_by_uuid"].get(str(u)) for u in uuids_all]
            point_kinds = [POINT_KIND_DISC] * len(result.idx)
            for layer in layers:
                uuids_all = np.concatenate([uuids_all, layer.uuids])
                records_all += layer.records
                point_kinds += [POINT_KIND_SQUARE] * len(layer.coords)
            if n_hidden:
                uuids_all = uuids_all[shown]
                records_all = [r for r, keep in zip(records_all, shown) if keep]
                point_kinds = [k for k, keep in zip(point_kinds, shown) if keep]

            map_html, legend_html = self._interactive_map(
                proj, mapping, records_all, uuids_all, point_kinds if layers else None,
                (proto_proj, proto_colors, proto_ks, proto_labels) if proto_colors is not None else None,
                ctx, static_html, result)
            self._map_records = records_all
            self._map_uuids = uuids_all
            self._map_n_drawn = n_drawn

            notes_above = []
            if result.class_filter:
                notes_above.append(h.banner(
                    f"Fitted to {len(result.idx):,} points from {len(result.class_filter)} of "
                    f"{len(result.all_classes)} classes ({result.pool_size:,} available) — positions "
                    "are not comparable with another selection's.", "info"))
            if n_hidden:
                kept = total - n_hidden
                notes_above.append(h.banner(
                    f"{kept:,} of {total:,} points shown — {n_hidden:,} hidden by the display "
                    f"filters. The projection is unchanged: what is left is where it always was.", "info",
                    icon_name="filter"))
            fit_note = self._fit_note(result, layers, len(result.idx))
            if fit_note:
                notes_above.append(h.banner(fit_note, "info"))
            self.notes_html = "".join(notes_above)
            self.legend_html = legend_html
            self.results_html = map_html
            self._has_result = True
        except Exception as ex:  # noqa: BLE001
            _log(f"error rendering projection: {ex}")
            traceback.print_exc()
            if generation != self._projection_generation:
                return
            self.results_html = h.banner(f"Error: {ex}", "error")

    def _static_figure(self, result, mapping, colors_by_class, proj, is_overlay, proto_proj, proto_labels,
                       proto_ks, n_subcenters, layers, n_drawn, map_size):
        fig, ax = subplots(figsize=(map_size / INLINE_DPI, map_size / INLINE_DPI * 0.97))
        color_legend = None
        if mapping.kind == pc.KIND_CATEGORICAL:
            for color, name in mapping.legend:
                mask = mapping.groups == name
                pool_mask = mask & ~is_overlay
                ax.scatter(proj[pool_mask, 0], proj[pool_mask, 1], s=6, alpha=0.7, color=color, label=short_name(name))
                over_mask = mask & is_overlay
                if over_mask.any():
                    ax.scatter(proj[over_mask, 0], proj[over_mask, 1], marker="s", s=24, color=color,
                               edgecolors="black", linewidths=0.4, zorder=4)
            legend_ncol = max(1, -(-len(mapping.legend) // 16))
            color_legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=legend_ncol,
                                     frameon=False, markerscale=1.8, title=mapping.column, title_fontsize=8)
        else:
            known = np.isfinite(mapping.values)
            missing = ~known
            if missing.any():
                ax.scatter(proj[missing & ~is_overlay, 0], proj[missing & ~is_overlay, 1], s=6, alpha=0.6,
                           color=pc.MISSING_COLOR, label=pc.MISSING_GROUP)
                if (missing & is_overlay).any():
                    ax.scatter(proj[missing & is_overlay, 0], proj[missing & is_overlay, 1], marker="s", s=24,
                               color=pc.MISSING_COLOR, edgecolors="black", linewidths=0.4, zorder=4)
                color_legend = ax.legend(loc="upper right", fontsize=7, frameon=False, markerscale=1.8)
            if known.any():
                pool_known = known & ~is_overlay
                sc = ax.scatter(proj[pool_known, 0], proj[pool_known, 1], s=6, alpha=0.6,
                                c=mapping.values[pool_known], cmap="viridis", vmin=mapping.vmin, vmax=mapping.vmax)
                over_known = known & is_overlay
                if over_known.any():
                    ax.scatter(proj[over_known, 0], proj[over_known, 1], marker="s", s=24,
                               c=mapping.values[over_known], cmap="viridis", vmin=mapping.vmin, vmax=mapping.vmax,
                               edgecolors="black", linewidths=0.4, zorder=4)
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(mapping.column, fontsize=8)
        if proto_proj is not None and len(proto_proj):
            ks = np.array(proto_ks)
            for k_idx in range(n_subcenters):
                sel = ks == k_idx
                if not sel.any():
                    continue
                marker = SUBCENTER_MARKERS[k_idx % len(SUBCENTER_MARKERS)]
                fill = ([mapping.color_of(proto_labels[i]) for i in np.flatnonzero(sel)] if colors_by_class
                        else result.source_color)
                ax.scatter(proto_proj[sel, 0], proto_proj[sel, 1], marker=marker, s=210 if marker == "*" else 130,
                           c=fill, edgecolors="black", linewidths=0.9, zorder=5)
            if n_subcenters > 1:
                marker_handles = [
                    Line2D([], [], linestyle="none", label=f"sub {k}", marker=SUBCENTER_MARKERS[k % len(SUBCENTER_MARKERS)],
                           color="#bbbbbb", markeredgecolor="black", markersize=10)
                    for k in range(n_subcenters)]
                if color_legend is not None:
                    fig.add_artist(color_legend)
                ax.legend(handles=marker_handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                          ncol=min(n_subcenters, 8), fontsize=7, frameon=False)
        if result.method == "umap_circle":
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=0.8, linestyle="--")
            ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if not (mapping.kind == pc.KIND_CATEGORICAL or n_subcenters > 1):
            fig.tight_layout()
        if colors_by_class:
            notes = [f"Color = {mapping.column} (legend alongside), for the points as well as the prototypes."]
        elif mapping.kind == pc.KIND_CATEGORICAL:
            notes = [f"Color = {mapping.column} (legend alongside)."]
        else:
            notes = [f"Point color = {mapping.column}."]
        if mapping.note:
            notes.append(mapping.note)
        if layers:
            notes.append(f"Squares = crops from {len(layers)} simple inference(s) "
                         f"({len(proj) - n_drawn:,} points), not part of the dataset.")
        if n_subcenters:
            shape_note = "one shape per subcenter (legend below the plot)" if n_subcenters > 1 else "large stars"
            proto_color = "" if colors_by_class else (
                " in blue" if result.source_key == PROTO_FROM_MODEL else " in purple")
            notes.append(f"Prototypes = large black-edged symbols{proto_color}, {shape_note} — {result.source_note}.")
        img = fig_html(self.s, fig, f"2D projection ({result.method}) — {result.exp_name}")
        return f'<div class="muted small">{h.e(" ".join(notes))}</div>{img}'

    def _interactive_map(self, proj, mapping, records, uuids, point_kinds, prototypes, ctx, static_html, result):
        """The map's data as a JSON file the page fetches once, and the
        HTML that hosts it (toolbar, canvas box, static figure) and its
        legend."""
        colors = list(mapping.colors)
        palette = sorted(set(colors))
        pindex = {c: i for i, c in enumerate(palette)}
        groups = np.asarray(mapping.groups)
        names = []
        gindex = {}
        g = []
        for name in groups.tolist():
            if name not in gindex:
                gindex[name] = len(names)
                names.append(name)
            g.append(gindex[name])
        label_col = self.state.dataset.columns.label_col
        hl, hl_index, hi = [], {}, []
        for i, rec in enumerate(records):
            label = rec.get(label_col) if rec and label_col else None
            text = str(label) if label is not None else str(uuids[i])[:8]
            if text not in hl_index:
                hl_index[text] = len(hl)
                hl.append(text)
            hi.append(hl_index[text])
        data = {
            "x": np.round(proj[:, 0].astype(float), 6).tolist(),
            "y": np.round(proj[:, 1].astype(float), 6).tolist(),
            "c": [pindex[c] for c in colors], "palette": palette, "g": g, "groups": [str(n) for n in names],
            "k": None if point_kinds is None else [1 if k == POINT_KIND_SQUARE else 0 for k in point_kinds],
            "hl": hl, "hi": hi, "protos": None,
        }
        if prototypes is not None:
            coords, pcolors, ks, plabels = prototypes
            coords = np.asarray(coords, dtype=float)
            data["protos"] = {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(), "color": list(pcolors),
                              "k": [int(k) for k in ks], "label": [str(l) for l in plabels]}
        dkey = self.s.put_download(json.dumps(data).encode("utf-8"), "map.json", "application/json")
        session = self.s
        dataset = self.state.dataset
        crop_col = ctx["crop_col"]

        def resolve(i: int):
            if not 0 <= i < len(records) or not records[i]:
                return None
            rec = records[i]
            try:
                path = dataset.resolve_crop_path(rec[crop_col])
            except Exception:  # noqa: BLE001
                return None
            lut, turns = crop_display(session, rec)
            return path, lut, turns

        def describe(i: int):
            if not 0 <= i < len(records) or not records[i]:
                return None
            uid = uuids[i] if i < len(uuids) else None
            missing = uid is None or uid != uid or str(uid) == ""     # None, NaN or empty
            return (None if missing else str(uid)), records[i].get(crop_col)

        mkey = self.s.put_point_map(resolve, describe)
        src = f"/dl/{self.s.sid}/{dkey}?inline=1"
        pimg = f"/pimg/{self.s.sid}/{mkey}/"
        pinfo = f"/pinfo/{self.s.sid}/{mkey}/"
        # One row that never wraps (a button appearing never pushes the map
        # down), in groups that say what they act on: the view (how the map
        # is drawn and moved), the lasso selection, the gallery, the export.
        # Labels show when the map is wide enough, the least needed parts
        # fold away when it is narrow (embeddings.css).
        def icon_btn(key: str, icon_name: str, title: str, *, hidden: bool = False, cls: str = "",
                     label: str = "", attr: str = "") -> str:
            text = f'<span class="lbl">{h.e(label)}</span>' if label else ""
            return (f'<button type="button" class="btn btn-sm btn-ghost {"btn-icon " if not label else ""}{cls}" '
                    f'{attr or f"data-pm={chr(34)}{key}{chr(34)}"} title="{h.e(title)}"{" hidden" if hidden else ""}>'
                    f'{h.icon(icon_name, 14)}{text}</button>')

        def group(cls: str, title: str, body: str) -> str:
            return f'<span class="pm-grp {cls}" title="{h.e(title)}">{body}</span>'

        view = (
            f'<div class="seg pm-render"><button type="button" class="seg-b on" data-pm-render="interactive" title="Interactive map">{h.icon("target", 13)}</button>'
            f'<button type="button" class="seg-b" data-pm-render="static" title="Static matplotlib figure">{h.icon("image", 13)}</button></div>'
            f'<span class="pm-tools pm-in">'
            + group("pm-g-view", "Move around the map",
                    f'<span class="seg">'
                    f'<button type="button" class="seg-b on" data-pm-mode="pan" title="Pan: drag to move the map">{h.icon("pan", 13)}</button>'
                    f'<button type="button" class="seg-b" data-pm-mode="zoom" title="Zoom: drag a box to zoom into it">{h.icon("zoom", 13)}</button></span>'
                    f'<span class="pm-zoom">'
                    f'<button type="button" class="btn btn-sm btn-ghost btn-icon" data-pm="out" title="Zoom out">-</button>'
                    f'<span class="pm-zoomtext muted small">x1.0</span>'
                    f'<button type="button" class="btn btn-sm btn-ghost btn-icon" data-pm="in" title="Zoom in">+</button></span>'
                    + icon_btn("reset", "refresh", "Reset the view: the whole cloud, centred", hidden=True))
            + group("pm-g-sel", "Selection",
                    icon_btn("", "lasso", "Lasso: draw around points to select them", label="Lasso",
                             attr='data-pm-mode="lasso"')
                    + '<b class="pm-seltext small"></b>'
                    + icon_btn("show", "external", "Open the selected points in the carousel", hidden=True)
                    + icon_btn("clear", "x", "Clear the selection", hidden=True))
            + group("pm-g-gal", "Gallery",
                    icon_btn("gallery", "grid", "Gallery: the map moves to a corner, and a lasso fills the rest "
                             "with crops drawn at random from the selection", cls="pm-toggle", label="Gallery")
                    + icon_btn("crop-smaller", "minus", "Smaller crops (more of them)", cls="pm-gal-x")
                    + icon_btn("crop-larger", "plus", "Larger crops (fewer of them)", cls="pm-gal-x")
                    + icon_btn("shuffle", "refresh", "Draw another random sample of crops", hidden=True, cls="pm-gal-x"))
            + '<span class="grow"></span>'
            + group("pm-g-out", "Export what is shown",
                    icon_btn("copy", "copy", "Copy what is shown as a picture -- always the same size and framing for the same view")
                    + icon_btn("save", "download", "Save what is shown as a PNG -- always the same size and framing for the same view"))
            + '</span>')
        toolbar = f'<div class="pm-row">{view}</div>'
        map_html = (f'<div class="pmap" data-map="{dkey}" data-src="{h.e(src)}" data-pimg="{h.e(pimg)}" data-pinfo="{h.e(pinfo)}" data-target="{self.name}" '
                    f'data-static="{dkey}" title="">'
                    f'<div class="pm-bar">{toolbar}</div>'
                    f'<div class="pm-inter"><div class="pmap-box"></div></div>'
                    f'<div class="pm-stat">{static_html}</div></div>')
        legend_html = self._legend_html(mapping, names, dkey)
        return map_html, legend_html

    def _legend_html(self, mapping, names: list, dkey: str) -> str:
        if not mapping.legend:
            return ""
        known = set(names)
        filterable = all(label in known for _, label in mapping.legend)
        if not filterable:
            body = "".join(f'<div class="row-nw">{h.dot(color, size=10)}<span class="small xname" title="{h.e(label)}">{h.e(label)}</span></div>'
                           for color, label in mapping.legend)
            head = '<b class="small">Legend</b>'
        else:
            gi = {n: i for i, n in enumerate(names)}
            body = "".join(
                f'<div class="row-nw pm-lg"><button type="button" class="btn btn-sm btn-ghost btn-icon" data-front="{gi[label]}" '
                f'title="Draw on top of the other points">{h.icon("layers", 13)}</button>{h.dot(color, size=10)}'
                f'<label class="chk ell" title="{h.e(label)}"><input type="checkbox" checked data-g="{gi[label]}"><span>{h.e(label)}</span></label></div>'
                for color, label in mapping.legend)
            head = (f'<div class="row-nw"><b class="small">{h.e(f"Show {mapping.column}")}</b>'
                    f'<button type="button" class="btn btn-sm btn-link" data-pm="all">All</button>'
                    f'<button type="button" class="btn btn-sm btn-link" data-pm="none">None</button></div>')
        return (f'<section class="card pm-legend" data-map="{dkey}" data-static="lg-{dkey}"><div class="card-b">{head}'
                f'<div class="pm-lg-list">{body}</div></div></section>')

    # ---- crops behind points ----

    def _open_crops(self, indices: list, title: str = "") -> None:
        records = self._map_records
        n_pool = self._map_n_drawn
        picked = [records[i] for i in indices if 0 <= i < len(records) and records[i]]
        if not picked:
            return
        from_dataset = all(i < n_pool for i in indices)
        cols = self.state.dataset.columns
        title = title or f"Lasso selection — {len(picked):,} images"
        if not from_dataset:
            title += " — inference crops, no review"
        open_carousel(self.s, title, picked, cols.crop_col, cols.uuid_col,
                      review_source="projection" if from_dataset else "", label_col=cols.label_col)

    def on_point(self, i: int = 0, **_):
        i = int(i)
        if self._map_uuids is None or not 0 <= i < len(self._map_records):
            return
        self._open_crops([i], title=f"Crop {self._map_uuids[i]}")

    def on_select(self, idx=None, **_):
        self._open_crops([int(i) for i in (idx or [])])

    # ---- class overlap ----

    def on_overlap(self, **_):
        exp_name = self.experiment
        if not exp_name:
            return
        self.overlap_busy = True
        self.overlap_html = ""

        def failed(ex):
            self.overlap_busy = False
            self.overlap_html = h.banner(f"Could not measure class overlap: {ex}", "error")
            self.refresh()

        self.s.job("Class overlap").run(lambda: (self._compute_overlap(exp_name), self.refresh()), on_error=failed)

    def _compute_overlap(self, exp_name: str) -> None:
        context = self._analysis_context(exp_name)
        if context is None:
            self.overlap_busy = False
            self.overlap_html = h.banner(f"'{exp_name}' has no embeddings attached -- add a sidecar on the Data tab.", "info")
            return
        all_classes = sorted(set(context["labels"].tolist()))
        overlap = ea.class_overlap(context["embeddings"], context["labels"], class_order=all_classes)
        _log(f"class overlap for '{exp_name}': {len(overlap.labels)} classes")
        self._overlap = (overlap, exp_name)
        self.overlap_busy = False
        self.overlap_html = self._render_overlap(overlap, exp_name)

    def _on_overlap_focus_change(self) -> None:
        if self._overlap is not None:
            self.overlap_html = self._render_overlap(*self._overlap)

    def _render_overlap(self, overlap, exp_name: str) -> str:
        if not overlap.labels:
            return h.banner("No class has enough crops with embeddings to measure.", "info")
        finite = overlap.matrix[np.isfinite(overlap.matrix)]
        if not finite.size:
            return h.banner("Not enough crops per class to compare.", "info")
        low, high = float(finite.min()), float(finite.max())
        span = max(high - low, 1e-9)
        index_of = {label: i for i, label in enumerate(overlap.labels)}

        def rank(criterion):
            if criterion == cf.FOCUS_MOST_OVERLAPPING:
                return cf.pairs_to_focus((a, b) for a, b, _ in overlap.top_pairs(None))
            if criterion == cf.FOCUS_LEAST_SEPARATED:
                table = pd.DataFrame({"label": list(overlap.labels), "separation": overlap.separation,
                                      "support": overlap.counts})
                table = table[np.isfinite(table["separation"])]
                return cf.rank_by(table, "separation", ascending=True,
                                  min_support=self.overlap_focus.min_support_for(criterion))
            if criterion == cf.FOCUS_LARGEST_SUPPORT:
                order = np.argsort(-overlap.counts, kind="stable")
                return [overlap.labels[i] for i in order][: cf.FOCUS_MAX]
            return []

        shown = self.overlap_focus.resolve(list(overlap.labels), rank)
        shown_idx = [index_of[l] for l in shown]
        short_labels, _prefix = short_column_labels(shown, 5)
        cells = ['<span></span>'] + [f'<span class="cm-h" style="font-size:9px" title="{h.e(l)}">{h.e(s)}</span>'
                                     for s, l in zip(short_labels, shown)]
        for i in shown_idx:
            label_i = overlap.labels[i]
            cells.append(f'<span class="cm-r" style="font-size:10px">{h.e(label_i)}</span>')
            for j in shown_idx:
                label_j = overlap.labels[j]
                value = overlap.matrix[i, j]
                if not np.isfinite(value):
                    cells.append(f'<span class="ov-c ov-na" style="height:{OVERLAP_CELL}px"></span>')
                    continue
                fraction = (value - low) / span
                note = "how alike this class is to itself" if i == j else "how alike these two classes are"
                cells.append(f'<span class="ov-c" style="height:{OVERLAP_CELL}px;background:{blues_color(fraction)};'
                             f'color:{"#ffffff" if fraction > 0.55 else "#111111"}" '
                             f'title="{h.e(label_i)} vs {h.e(label_j)}&#10;mean cosine similarity {value:.4f}&#10;({note})">{value:.2f}</span>')
        grid = (f'<div class="cm-scroll"><div class="cm-grid" style="grid-template-columns:{OVERLAP_LABEL_WIDTH}px '
                f'repeat({len(shown_idx)}, {OVERLAP_CELL}px)">{"".join(cells)}</div></div>')
        pairs = "".join(f'<div class="small">{h.e(a)} / {h.e(b)}  —  {v:.3f}</div>' for a, b, v in overlap.top_pairs(8))
        order = np.argsort(np.where(np.isfinite(overlap.separation), overlap.separation, np.inf))
        seps = []
        for idx in order[:8]:
            value = overlap.separation[idx]
            if not np.isfinite(value):
                continue
            seps.append(f'<div class="small{" c-err" if value <= 0 else ""}">{h.e(overlap.labels[idx])}  —  {value:+.3f}  '
                        f'({int(overlap.counts[idx]):,} crops)</div>')
        side = (f'<div class="row-wrap" style="align-items:flex-start;gap:10px">'
                f'<section class="card" style="width:340px"><div class="card-b"><b class="small">Most-overlapping pairs</b>{pairs}</div></section>'
                f'<section class="card" style="width:380px"><div class="card-b"><b class="small">Least separated classes</b>'
                f'<div class="muted small"><i>Own-class similarity minus the closest other class. At or below zero, a class is no more like itself than like its neighbour.</i></div>'
                f'{"".join(seps)}</div></section></div>')
        return (f'<div class="muted small">Measured in {h.name(exp_name)} over {int(overlap.counts.sum()):,} crops.</div>'
                f'{self.overlap_focus.render()}{grid}{side}')

    def on_ofocus(self, op: str = "", **args):
        self.overlap_focus.handle(op, args)
        if self._overlap is not None:
            self.overlap_html = self._render_overlap(*self._overlap)

    # ---- rendering the tab ----

    def render(self) -> str:
        self.on_shown()
        state = self.state
        left = [f'<div class="sec-t"><h2>2D projection of the embeddings</h2>{h.info_btn(METHOD_MANUAL_PAGES.get(self.method, "projection-2d"))}</div>']
        if not self.controls_ready:
            left.append(self.status_html or h.spinner("Reading the embeddings..."))
            return f'<div class="emb"><div class="emb-left stack">{"".join(left)}</div></div>'
        names = experiments_with_embeddings(state)
        controls = [
            h.select([(n, n) for n in names], self.experiment or "", change=self.c("experiment"), label="Experiment"),
            h.select(METHOD_OPTIONS, self.method, change=self.c("method"), label="Projection"),
            h.text_input(self.sample_value, label="Sample size", width=70, change=self.c("sample"), extra=' data-force="1"'),
            h.select(self._color_options, self.color_key, change=self.c("color"), label="Color by"),
        ]
        scale = self._color_scale_kind()
        if scale is not None:
            controls.append(h.segmented([(pc.KIND_NUMERIC, "Continuous"), (pc.KIND_CATEGORICAL, "Discrete")], scale[1],
                                        act_for=lambda key: self.a("color_scale", kind=key)))
        if not self._many_classes:
            items = [("All classes", self.a("class_all"), None, False)] + [
                (str(c), self.a("class_toggle", label=str(c)), "check" if c in self._class_filter else None, False)
                for c in self._classes]
            controls.append(menu(items, icon_name="filter", label="Restrict to classes", right=False, title="Restrict to classes"))
        filter_items = [(str(c), self.a("filter_toggle", column=str(c)), "check" if c in self._filter_columns else None, False)
                        for c in self._filterable_columns()]
        controls.append(menu(filter_items, icon_name="sliders", label="Filter points", right=False, title="Filter points"))
        controls.append(h.button("Run projection", act_attr=self.a("run"), kind="primary", icon_name="play"))
        body = [f'<div class="row-wrap">{"".join(controls)}</div>']
        if self._score_note:
            body.append(f'<div class="muted small"><i>{h.e(self._score_note)}</i></div>')
        if self._many_classes:
            body.append(self.class_picker.render())
        elif self._class_filter:
            body.append('<div class="row-wrap">' + "".join(
                f'<span class="chip chip-x">{h.e(c)}<button type="button" class="chip-del" title="Remove this class"'
                f'{self.a("class_toggle", label=str(c))}>{h.icon("x", 11)}</button></span>' for c in self._class_filter) + '</div>')
        # prototypes
        self.proto_picker.set_picked(self._proto_filter or [])
        proto_value = self._proto_source_for(self.experiment, self._classes) if self.experiment else None
        proto = (f'<div class="row-wrap"><b class="small">Prototypes</b><span class="muted small">{h.e(self._proto_summary())}</span>'
                 f'{h.button("All classes", act_attr=self.a("proto_all"), small=True, kind="link")}'
                 f'{h.button("None", act_attr=self.a("proto_none"), small=True, kind="link")}</div>'
                 f'<div class="row-wrap">{h.select(self._proto_source_options, proto_value or "", change=self.c("proto_source"), label="Prototypes from")}</div>'
                 f'{f"<div class={chr(34)}muted small{chr(34)}>{h.e(self._proto_source_note)}</div>" if self._proto_source_note else ""}'
                 f'{self.proto_picker.render()}')
        body.append(f'<div class="emb-proto">{proto}</div>')
        body.append(self.notes_html)
        body.append(self.status_html)
        left.append(f'<section class="card"><div class="card-b stack">{"".join(b for b in body if b)}</div></section>')
        left.append(self._filters_card())
        left.append(self._overlay_card())
        left.append(self.legend_html)
        overlap_body = (f'<div class="row">{h.button("Measure class overlap", act_attr=self.a("overlap"), icon_name="layers", disabled=self.overlap_busy)}</div>'
                        f'{h.spinner("Measuring class overlap...") if self.overlap_busy else ""}{self.overlap_html}')
        left.append(h.section("How much the classes overlap", overlap_body, manual="class-overlap",
                              sub="mean cosine similarity within and between classes"))
        split = ('<div class="emb-split" title="Drag to resize the map; drag it to the left edge, or double-click, '
                 'for a full-width map"></div>')
        # The settings column folds away (the splitter, see projection.js)
        # only while there is a map to widen: folded with no map, the tab
        # showed one "Click 'Run projection'" line and hid the button. And a
        # folded column says how to get it back.
        has_map = 'class="pmap' in self.results_html
        unfold = (f'<button type="button" class="btn btn-sm emb-unfold" data-emb-unfold="1" '
                  f'title="The settings column is folded away -- show it again">'
                  f'{h.icon("sliders", 14)}<span>Show settings</span></button>')
        right = f'{split}<div class="emb-right stack">{unfold}{self.stale_html()}{self.results_html}</div>'
        return (f'<div class="emb{" emb-has-map" if has_map else ""}">'
                f'<div class="emb-left stack">{"".join(p for p in left if p)}</div>{right}</div>')

    def _filters_card(self) -> str:
        if not self._filter_columns:
            return ""
        active = [self._filters[c].describe() for c in self._filter_columns
                  if c in self._filters and self._filters[c].is_active()]
        sub = ("Independent of the coloring: hiding points here changes nothing about the projection."
               if not active else "Showing: " + "; ".join(active))
        blocks = []
        ctx = self._last_context
        for column in self._filter_columns:
            f = self._filters.get(column)
            if f is None:
                continue
            head = [f'<b class="small mono">{h.e(column)}</b>']
            if ctx is not None and pc.can_be_continuous(self._column_reference(ctx, column)):
                head.append(h.segmented([(pc.KIND_NUMERIC, "Continuous"), (pc.KIND_CATEGORICAL, "Discrete")], f.kind,
                                        act_for=lambda key, c=column: self.a("filter_kind", column=c, kind=key)))
            head.append(h.button(icon_name="x", act_attr=self.a("filter_toggle", column=column), small=True, kind="ghost",
                                 title="Stop filtering on this column"))
            if f.kind == pc.KIND_NUMERIC:
                lo, hi = f.bounds
                low, high = f.window
                out_id = f"pf-{abs(hash(column)) % 10**8}"
                rng = lambda which, v: (f'<input type="range" class="rng" min="{lo!r}" max="{hi!r}" step="any" value="{v!r}" '  # noqa: E731
                                        f'data-out="{out_id}" data-force="1"{self.c("filter_range", column=column, which=which)}>')
                body = (f'<div class="dual">{rng("start", low)}{rng("end", high)}</div>'
                        f'<div class="small muted" id="{out_id}">{low:.4g} to {high:.4g}</div>')
                if f.has_missing:
                    body += h.checkbox("include rows with no value", f.include_missing, change=self.c("filter_missing", column=column))
            else:
                checks = "".join(h.checkbox(str(v), v not in f.hidden, change=self.c("filter_value", column=column, i=i))
                                 for i, v in enumerate(f.values))
                if f.has_missing:
                    checks += h.checkbox(pc.MISSING_GROUP, f.include_missing, change=self.c("filter_missing", column=column))
                body = (f'<div class="row-nw">{h.button("All", act_attr=self.a("filter_all", column=column, shown=True), small=True, kind="link")}'
                        f'{h.button("None", act_attr=self.a("filter_all", column=column, shown=False), small=True, kind="link")}</div>'
                        f'<div class="checks">{checks}</div>')
                if f.truncated:
                    body += (f'<div class="muted small"><i>{f.truncated} rarer value(s) are not listed and are never hidden by '
                             f'this filter.</i></div>')
            blocks.append(f'<div class="flt" data-key="pf-{h.e(column)}"><div class="row-wrap">{"".join(head)}</div>{body}</div>')
        return h.section("Display filters", f'<div class="muted small">{h.e(sub)}</div>{"".join(blocks)}',
                         right=h.button("Reset all", act_attr=self.a("filter_reset"), small=True, kind="ghost", icon_name="rotate"))

    def _overlay_card(self) -> str:
        inferences = self.state.inferences
        if not inferences:
            return ""
        rows = []
        for name, inference in inferences.items():
            shown = len(overlay_indices(len(inference.df)))
            suffix = f"{shown:,} crops" if shown == len(inference.df) else f"{shown:,} of {len(inference.df):,} crops"
            rows.append(h.checkbox(f"{name} ({suffix})", name in self._overlay_selection,
                                   change=self.c("overlay_toggle", name=name), ellipsis=True))
        forced = self._refit_forced()
        tip = ("t-SNE cannot place points it was not fitted on, so overlaid crops are always part of "
               "the fit -- which moves every other point too." if forced else
               "Off: the map holds still and the overlaid crops are placed onto the fit it already "
               "has. On: everything is fitted together, so every point moves.")
        refit = h.checkbox("Refit the map on these points too", True if forced else self._refit_choice,
                           change=self.c("refit"), disabled=forced, title=tip)
        note = ("t-SNE learns no reusable map, so overlaid crops are fitted with the rest and the "
                "whole projection changes. PCA and UMAP can place them onto the existing fit." if forced else "")
        body = (f'<div class="muted small">Drawn as squares, with no label and no effect on any metric.</div>'
                f'<div class="stack" style="gap:2px">{"".join(rows)}</div>{refit}'
                f'{f"<div class={chr(34)}muted small{chr(34)}>{h.e(note)}</div>" if note else ""}')
        return h.section("Overlay simple inferences", body)

    # ---- actions ----

    def on_experiment(self, value: str = "", **_):
        names = experiments_with_embeddings(self.state)
        if value not in names:
            return
        self.experiment = value
        self._projection_generation += 1
        self._has_result = False
        self._last_result = None
        self.results_html = h.banner("Click 'Run projection' to compute.", "info")
        self.legend_html = ""
        self.notes_html = ""

        def work():
            ctx = self._analysis_context(value)
            restored = self._rebind(self._runs.get(value), ctx) if ctx is not None else None
            if restored is not None:
                self.method = restored.method
                self.sample_value = str(restored.n_sample)
                self._class_filter = list(restored.class_filter)
                self._proto_filter = None if restored.proto_filter is None else list(restored.proto_filter)
                self._proto_filter_chosen = True
                if restored.source_key:
                    self._proto_source_choice[value] = restored.source_key
            elif value in self._runs:
                del self._runs[value]
            self._render(ctx)
            if restored is not None:
                self._projection_generation += 1
                self._last_result = restored
                self.results_html = h.spinner("Restoring projection...")
                self.refresh()
                self._render_projection(restored, self._projection_generation)
            self.refresh()

        self.s.job("Embeddings controls").run(work, on_error=lambda ex: (traceback.print_exc(), self.refresh()))

    def on_method(self, value: str = "", **_):
        if value in dict(METHOD_OPTIONS):
            self.method = value
            self.on_run()

    def on_sample(self, value: str = "", **_):
        self.sample_value = str(value or "")

    def on_color(self, value: str = "", **_):
        if value not in self._color_column_map:
            return
        self.color_key = value
        self._redraw_current("Recoloring...")

    def on_color_scale(self, kind: str = "", **_):
        column = self._color_column_map.get(self.color_key or CLASS_COLOR_KEY)
        if column is not None and kind in (pc.KIND_NUMERIC, pc.KIND_CATEGORICAL):
            self._set_column_kind(column, kind)

    def on_class_all(self, **_):
        self._set_class_filter([])

    def on_class_toggle(self, label: str = "", **_):
        value = next((c for c in self._classes if str(c) == str(label)), None)
        if value is None:
            return
        if value in self._class_filter:
            self._set_class_filter([c for c in self._class_filter if c != value])
        else:
            self._set_class_filter(self._class_filter + [value])

    def on_cpick(self, op: str = "", **args):
        self.class_picker.handle(op, args)

    def on_ppick(self, op: str = "", **args):
        self.proto_picker.handle(op, args)

    def on_proto_all(self, **_):
        self._set_proto_filter(None)

    def on_proto_none(self, **_):
        self._set_proto_filter([])

    def on_proto_source(self, value: str = "", **_):
        if not self.experiment or value not in dict(self._proto_source_options):
            return
        self._proto_source_choice[self.experiment] = value
        self._reset_result("Prototype source changed — run the projection again.")

    def on_filter_toggle(self, column: str = "", **_):
        if column in self._filterable_columns():
            self._toggle_filter_column(column)

    def on_filter_kind(self, column: str = "", kind: str = "", **_):
        if kind in (pc.KIND_NUMERIC, pc.KIND_CATEGORICAL):
            self._set_column_kind(column, kind)

    def on_filter_range(self, column: str = "", which: str = "start", value="", **_):
        f = self._filters.get(column)
        if f is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        lo, hi = f.bounds
        v = max(lo, min(hi, v))
        low, high = f.window
        f.window = (min(v, high), high) if which == "start" else (low, max(v, low))
        self._redraw_current("Filtering...")

    def on_filter_missing(self, column: str = "", value=False, **_):
        f = self._filters.get(column)
        if f is not None:
            f.include_missing = bool(value)
            self._redraw_current("Filtering...")

    def on_filter_value(self, column: str = "", i: int = 0, value=False, **_):
        f = self._filters.get(column)
        if f is None or not 0 <= int(i) < len(f.values):
            return
        v = f.values[int(i)]
        (f.hidden.discard if value else f.hidden.add)(v)
        self._redraw_current("Filtering...")

    def on_filter_all(self, column: str = "", shown=True, **_):
        f = self._filters.get(column)
        if f is None:
            return
        f.hidden = set() if shown else set(f.values)
        f.include_missing = bool(shown)
        self._redraw_current("Filtering...")

    def on_filter_reset(self, **_):
        for f in self._filters.values():
            f.reset()
        self._redraw_current("Filtering...")

    def on_overlay_toggle(self, name: str = "", value=False, **_):
        if name not in self.state.inferences:
            return
        (self._overlay_selection.add if value else self._overlay_selection.discard)(name)
        self._overlay_changed()

    def on_refit(self, value=False, **_):
        self._refit_choice = bool(value)
        self._overlay_changed()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        cached = self._runs.pop(old_name, None)
        if cached is not None:
            self._runs[new_name] = replace(cached, exp_name=new_name)
        if old_name in self._proto_source_choice:
            self._proto_source_choice[new_name] = self._proto_source_choice.pop(old_name)
        if self.experiment == old_name:
            self.experiment = new_name
        if self.built:
            self.refresh_controls()

    def recompute_after_restore(self) -> None:
        state = self.state
        if state.dataset is None or not experiments_with_embeddings(state) or not self.experiment:
            return
        self.on_run()

    # ---- snapshots (see ui/view_state.py) ----

    def view_state(self) -> dict:
        filters = {}
        for column in self._filter_columns:
            f = self._filters.get(column)
            if f is None:
                continue
            filters[column] = {"kind": f.kind, "hidden": [str(v) for v in f.hidden],
                               "window": [float(f.window[0]), float(f.window[1])],
                               "include_missing": bool(f.include_missing)}
        if not self.experiment:
            # The original's dropdown is filled on every data change, shown or not.
            self._refresh_experiment_dropdown()
        return {
            "experiment": self.experiment or "",
            "method": self.method or "pca",
            "sample": str(self.sample_value or ""),
            "color_column": self._color_column_map.get(self.color_key or CLASS_COLOR_KEY) or "",
            "column_kind": dict(self._column_kind),
            "class_filter": [str(c) for c in self._class_filter],
            "proto_filter": None if self._proto_filter is None else [str(c) for c in self._proto_filter],
            "proto_filter_chosen": bool(self._proto_filter_chosen),
            "filter_columns": list(self._filter_columns),
            "filters": filters,
            "overlays": sorted(self._overlay_selection),
            "refit": bool(self._refit_choice),
        }

    def restore_view_state(self, state: dict) -> list:
        warnings = []
        experiment = state.get("experiment") or ""
        if experiment:
            if experiment in self.state.experiments:
                self.experiment = experiment
            else:
                warnings.append(f"Embeddings: experiment '{experiment}' is not loaded.")
        method = state.get("method")
        if method in ea.PROJECTION_METHODS:
            self.method = method
        elif method:
            warnings.append(f"Embeddings: unknown projection method '{method}'.")
        if state.get("sample"):
            self.sample_value = str(state["sample"])
        kinds = state.get("column_kind")
        if isinstance(kinds, dict):
            self._column_kind = {str(c): k for c, k in kinds.items() if k in (pc.KIND_CATEGORICAL, pc.KIND_NUMERIC)}
        self._class_filter = [c for c in state.get("class_filter", []) or [] if isinstance(c, str)]
        if state.get("proto_filter_chosen"):
            chosen = state.get("proto_filter")
            self._proto_filter = None if chosen is None else [c for c in chosen if isinstance(c, str)]
            self._proto_filter_chosen = True
        self._refit_choice = bool(state.get("refit", False))
        overlays = state.get("overlays")
        if isinstance(overlays, list):
            wanted = {n for n in overlays if isinstance(n, str)}
            gone = sorted(wanted - set(self.state.inferences))
            if gone:
                warnings.append(f"Embeddings: inference set(s) {', '.join(gone)} are not loaded.")
            self._overlay_selection = wanted & set(self.state.inferences)
        saved = state.get("filters")
        self._filter_columns = [c for c in state.get("filter_columns", []) or [] if isinstance(c, str)]
        self._pending_filters = {c: saved[c] for c in self._filter_columns
                                 if isinstance(saved, dict) and isinstance(saved.get(c), dict)}
        self._filters = {}
        self._pending_color_column = str(state.get("color_column") or "")
        # The original's refresh(), synchronously: recompute_after_restore
        # runs right after and reads the controls this sets.
        self.built = True
        self._refresh_experiment_dropdown()
        self._render()
        self.refresh()
        return warnings
