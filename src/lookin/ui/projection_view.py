"""Tab: the 2D projection of an experiment's embeddings, in two renderings
of the very same projected coordinates.

The interactive one (ui/interactive_projection.py) draws the cloud on a
canvas, so a point can be hovered for its crop, clicked to open it full
size, or lassoed along with its neighbours to browse them in the carousel.
The static one is matplotlib, and stays the reference: same points, same
colors, same prototypes, but with the exact legend layout and the PNG
export. A run computes the projection once and hands the same arrays to
both; the switch between them only toggles visibility, so it never
recomputes.

Nothing else recomputes on its own either. Projecting is the one expensive
thing here, so the last run of each experiment is kept (ProjectionView._runs)
and put back on screen when that experiment is selected again -- switching
between two runs to compare them used to cost a t-SNE each way. Only the
button projects. What does throw a kept run away is the data under it
changing (see _clear_runs), and a run is checked against a freshly built
context before it is shown again (see _rebind).

This is deliberately not in the Prototypes tab, where it used to live.
Every panel there needs an active prototype source and shows nothing
without one, while a projection needs only the embeddings sidecar -- so
with embeddings and no prototypes the one usable view was buried under
three "no prototype available" banners. Prototypes are still drawn here
when a source is active, read from ctx.prototype_source, which is exactly
why that state is a registry rather than a view's private attribute.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import flet as ft
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lookin.core.chart_style import short_name, subplots
from lookin.core import embedding_analysis as ea
from lookin.core import point_coloring as pc
from lookin.core import point_filters as pf
from lookin.core.data_manager import DATASET_SOURCE_GROUP
from lookin.core import class_focus as cf
from lookin.core.class_focus import FOCUS_MAX
from lookin.ui.class_matrix import ClassFocus, ClassPicker, make_searchable, short_column_labels
from lookin.ui.components import (
    name_option,
    INLINE_DPI, BackgroundRebuild, blues_color, card, fig_to_zoomable_image,
    info_banner, section_title,
)
from lookin.ui.carousel import build_carousel_from_list
from lookin.ui.embedding_context import build_analysis_context, coverage_note, experiments_with_embeddings
from lookin.ui.interactive_projection import (
    MIN_PLOT_SIZE, POINT_KIND_DISC, POINT_KIND_SQUARE, InteractiveProjection,
)
from lookin.ui.prototype_source import KMEANS_COLOR, MODEL_COLOR

# The "Color by" dropdown's entry for the true label, and the names the two
# colorings that are not columns are captioned with. Display strings only:
# what code compares is the dropdown *key*, and CLASS_COLOR_KEY is the only
# one of those spelled out here -- see _rebuild_color_options for why the
# column entries are keyed positionally instead.
CLASS_COLOR_KEY = "0"
# The other coloring that is not a column: where a point comes from -- the
# dataset, or one of the loaded simple inferences. Its key is a word rather
# than a number precisely *because* the column entries are keyed
# positionally ("1", "2", ...): no column can ever produce this key, so a
# CSV column called "source" cannot collide with it.
SOURCE_COLOR_KEY = "source"
# Most classes whose prototypes go on one map, when "every class" is asked
# for: the markers are drawn one per subcenter and fitted with the points.
PROTO_CLASSES_MAX = 300
# Where the map's prototypes come from -- dropdown keys, compared below.
# "model" is the network's imported set, renamed through the Data tab's
# class mapping; "dataset" is built from the dataset's own labels (class
# means, or the k-means estimate when the Prototypes tab made one active).
PROTO_FROM_MODEL = "model"
PROTO_FROM_DATASET = "dataset"
# A disabled line in the menu, never a coloring (see _rebuild_color_options).
SCORE_NOTE_KEY = "scores_note"
CLASS_COLORING_LABEL = "class"
SOURCE_COLORING_LABEL = "source"
# How many points one overlaid inference may put on the map. Not a sample
# in the statistical sense -- these are the points the user came to look
# at, so they are all drawn -- but a set of a million crops would freeze
# the canvas, and evenly spaced indices thin it without reordering it.
OVERLAY_MAX_POINTS = 20_000
# In the projection a prototype takes the color of the class it belongs to
# (so it can be matched to its own cloud of points), which means color is
# no longer free to distinguish one subcenter of that class from another --
# the marker shape carries that instead. Cycles if a model ever has more
# subcenters than there are shapes here.
SUBCENTER_MARKERS = ["*", "P", "X", "D", "^", "v", "s", "p"]
# What the window loses to everything that is not the map: the app bar and
# the tab strip (~130), this tab's padding, the renderer switch and the
# caption under it, and the map's own toolbar. Subtracted from the window
# height to get the square the map may occupy.
#
# Measured against the running app at 1440x900 rather than estimated: with
# the controls on one line the map's top edge sits 193px down, and a value
# below that does not make the map bigger, it makes the bottom of it fall
# off the window -- invisible in the layout, a cropped scatter on screen.
#
# The rest is one spare line of toolbar. Everything above the map fits one
# row until a lasso adds "Show selection" and "Clear", at which point it
# wraps; reserving the row here costs ~44px of map and is what keeps that
# wrap from pushing the bottom of the plot out of the window.
CHROME_HEIGHT = 250
# A square bigger than this stops adding information and starts costing
# coordinates, however large the screen is.
MAP_MAX = 1100
# Full screen hides the settings column and drops the tab's padding to a
# margin, so the map is bounded by the window instead of by half of it.
# There is less chrome to subtract too: the caption under the renderer
# switch goes with the column, and each end of the padding is 12px lighter.
FULLSCREEN_PADDING = 8
# Measured the same way, with the padding down to a margin: 181px above
# the map, plus the same spare toolbar row and breathing room below.
FULLSCREEN_CHROME_HEIGHT = 230
# The cap above exists to stop a map costing coordinates it gains nothing
# from. Full screen is a deliberate ask for the biggest map the screen
# will carry, so it is raised rather than dropped.
MAP_MAX_FULLSCREEN = 1600


# Which classes the overlap grid shows past FOCUS_MAX (see ClassFocus).
OVERLAP_FOCUS_CRITERIA = (
    (cf.FOCUS_MOST_OVERLAPPING, "Most overlapping pairs"),
    (cf.FOCUS_LEAST_SEPARATED, "Least separated"),
    (cf.FOCUS_LARGEST_SUPPORT, "Largest support"),
    (cf.FOCUS_MANUAL, "Hand-picked"),
)
# Class-overlap grid, sized like the confusion matrices it is read beside.
OVERLAP_CELL = 40
OVERLAP_LABEL_WIDTH = 104


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def overlay_indices(n: int, cap: int = OVERLAY_MAX_POINTS) -> np.ndarray:
    """Which rows of an overlaid inference get drawn.

    All of them, until there are more than the canvas can carry -- and then
    every k-th one rather than a random sample or the first k: an evenly
    spaced subset keeps the shape of the set (a random one is fine too, but
    would change from render to render, and a head would silently show only
    whatever the CSV happened to list first)."""
    n = int(n)
    if n <= cap:
        return np.arange(n)
    return np.arange(0, n, -(-n // cap))[:cap]


def overlay_dimension_problem(inference, dim: int, exp_name: str) -> str:
    """Why this inference cannot be drawn on this experiment's map, or "".

    The only mismatch that can be checked at all: an embedding of a
    different width did not come from the same model, so there is no shared
    space to place it in. Same width and a different model is undetectable
    and stays the user's responsibility -- which is why the Data tab says
    so where the file is loaded."""
    width = int(inference.embeddings.shape[1])
    if width == dim:
        return ""
    return (f"'{inference.name}' has {width}-dimensional embeddings and '{exp_name}' has {dim} -- "
            f"they have to come from the same model to share a map.")


@dataclass
class _RunResult:
    """One computed projection, kept so it can be *re-rendered* without
    being recomputed.

    The split matters because the two things a run produces cost wildly
    different amounts: the coordinates are tens of seconds of t-SNE/UMAP,
    while everything downstream of them -- the colors, the legend, the
    figure, the canvas -- is a redraw. Changing what the color means is
    only affordable because this is held onto.

    A run is also kept per experiment (ProjectionView._runs) so that
    switching experiment and coming back is a redraw too. Those copies
    carry ctx=None: the context holds a copy of the whole embedding matrix
    and of the joined frame, and one of those per experiment is exactly
    the kind of thing that makes a big dataset run out of memory. Only the
    cheap half is kept, and _rebind puts a freshly built context back
    under it -- which also picks up any relabelling done since the run.
    """
    ctx: dict | None
    exp_name: str
    method: str
    idx: np.ndarray                # which rows of ctx were sampled
    proj: np.ndarray               # (n, 2) projected coordinates
    labels: np.ndarray             # (n,) class label per plotted point
    all_classes: list
    class_filter: list
    pool_size: int                 # rows the sample was drawn from, for the caption
    proto_proj: object = None      # (m, 2) prototypes in the same space, or None
    proto_labels: list = field(default_factory=list)
    proto_ks: list = field(default_factory=list)
    n_subcenters: int = 0
    source_key: object = None
    source_color: object = None
    source_note: str = ""
    # What the run was asked for, so the controls can be put back the way
    # they were when it is restored, and the uuids of the sampled rows, so
    # `idx` can be checked against a rebuilt context before being reused.
    n_sample: int = 0
    # Whose prototypes were fitted and drawn: None for every class (capped,
    # see _run_projection), else those classes only.
    proto_filter: list | None = None
    sample_uuids: np.ndarray | None = None
    run_generation: int = 0
    # The fitted projection itself, kept so that points which were not part
    # of it can be placed onto it afterwards without moving anything that
    # is already on screen -- see ea.ProjectionFit. None for a restored run
    # whose fit was never computed.
    fit: object = None
    # Which simple inferences were overlaid, and whether they were part of
    # the fit. Both are properties of *this run*, not of the checkboxes:
    # the caption has to say what the map was actually fitted on, and the
    # user is free to tick another one afterwards.
    overlay_names: list = field(default_factory=list)
    refit_with_overlay: bool = False
    # name -> (n, 2) coordinates, and name -> which rows of that inference
    # they are. Filled by the run when the overlay was part of the fit, and
    # lazily by the renderer (through fit.transform) when it was not, so
    # ticking a set on costs a transform rather than a projection.
    overlay_proj: dict = field(default_factory=dict)
    overlay_rows: dict = field(default_factory=dict)


@dataclass
class _OverlayLayer:
    """One simple inference as it appears on one rendered map: where its
    points landed, which of its rows they are, and everything the shared
    crop widgets need to show them."""
    name: str
    coords: np.ndarray            # (n, 2) in the projection's own space
    uuids: np.ndarray             # (n,) for the hover label and "find similar"
    records: list                 # (n,) row dicts, keyed like a dataset row
    rows: np.ndarray              # (n,) positions in the inference's frame
    total: int                    # rows the set holds, for the "thinned to" note


class ProjectionView:
    def __init__(self, ctx):
        self.ctx = ctx
        self._projection_generation = 0  # bumped on every run; a stale worker discards its result
        self._has_result = False  # is a computed projection currently on screen?
        # The last computed projection, kept so the coloring can change
        # without projecting again, and whether one is being computed right
        # now -- a recolor has nothing to redraw while it is (see
        # _on_color_change).
        self._last_result: _RunResult | None = None
        self._computing = False
        # The last completed run of each experiment, ctx-less (see
        # _RunResult), so leaving an experiment does not throw its
        # projection away: coming back re-renders that one instead of
        # projecting the same points again. Only a click on "Run
        # projection" recomputes -- which is the whole point, since
        # comparing two experiments means going back and forth between
        # them and t-SNE/UMAP cost tens of seconds each way.
        self._runs: dict[str, _RunResult] = {}
        # The analysis context of the experiment on screen, kept so the
        # controls in the left column -- which are built and rebuilt
        # outside a render -- can read the columns they filter and colour
        # on without joining the frame again.
        self._last_context: dict | None = None
        # Bumped whenever the data under the cache changes, so a run that
        # was already in flight at that moment cannot file its result into
        # a cache that no longer describes the same embeddings.
        self._runs_epoch = 0

        self.experiment_dropdown = ft.Dropdown(
            label="Experiment", width=260, on_change=self._on_experiment_change, border_radius=8,
        )
        self.projection_method = ft.Dropdown(
            label="Projection", value="pca", width=180, on_change=self._on_method_change,
            options=[
                ft.dropdown.Option("pca", "PCA"),
                ft.dropdown.Option("tsne", "t-SNE"),
                ft.dropdown.Option("umap", "UMAP"),
                ft.dropdown.Option("umap_circle", "UMAP (spherical)"),
            ],
            border_radius=8,
        )
        self.projection_sample_field = ft.TextField(label="Sample size", value="1500", width=140, border_radius=8)
        # What the color means. Sits here in the left column with the rest
        # of the settings rather than on the map's toolbar, which is
        # already a full line of controls above a height-bound square.
        #
        # Changing it does *not* re-run: see _on_color_change.
        self.color_by = ft.Dropdown(
            label="Color by", value=CLASS_COLOR_KEY, width=220, border_radius=8,
            on_change=self._on_color_change,
            options=[ft.dropdown.Option(CLASS_COLOR_KEY, "Class (true label)")],
        )
        # Dropdown key -> column of the merged frame (None for the class
        # entry), and the cached list those options were built from.
        self._color_column_map: dict = {CLASS_COLOR_KEY: None}
        self._colorable: list = []
        self._color_options_key = None
        # Narrowing to a few classes. A menu, not a Dropdown, and that is
        # not cosmetic: a Dropdown keeps the value it was last set to, so
        # re-picking a class one had just removed would not fire a change
        # at all. A menu is an action list with no state of its own, and
        # the chips below it are the single source of truth.
        # ---- overlaid simple inferences ----
        # Which sets are ticked, by name. Held here rather than read off the
        # checkboxes because the checkboxes are rebuilt whenever the loaded
        # sets change, and a rebuilt control has no memory of what it was.
        self._overlay_selection: set = set()
        self._overlay_checks: dict = {}
        # name -> why that set is not on the map, when a render had to
        # leave it off. Read by the caption: a ticked box drawing nothing
        # with no explanation is the worst version of this.
        self._overlay_problems: dict = {}
        self.overlay_column = ft.Column(spacing=4)
        # What the user last chose for themselves, kept because t-SNE
        # overrides the control below (see _sync_refit_control) and
        # switching away from it must not leave the override behind.
        self._refit_choice = False
        # Off by default, and that is the whole feature: the map holds
        # still and the new crops are placed onto the fit it already has.
        # Ticking this makes them part of the fit instead, which is the
        # honest thing to do when they are numerous enough to deserve their
        # own structure -- and the only thing t-SNE can do at all.
        self.refit_check = ft.Checkbox(
            label="Refit the map on these points too", value=False,
            on_change=self._on_refit_change,
        )
        self.overlay_note = ft.Text(size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.overlay_card = ft.Container()

        # ---- display filters, independent of the coloring ----
        # Which columns the user chose to filter on, in the order they
        # picked them, and the filter each one carries (see
        # core/point_filters.py). Held by the view rather than by the map,
        # which is rebuilt on every recolor: a filter that vanished because
        # the color changed would be the very coupling this exists to undo.
        self._filter_columns: list = []
        self._filters: dict = {}
        # Filter choices restored from a snapshot, waiting for a context to
        # rebuild them against. A ColumnFilter is built from the column it
        # filters (pf.build_filter scans it for its values and its range),
        # and at restore time no projection has been run yet -- so the
        # user's *choices* are parked here and applied by
        # _apply_pending_filters() as soon as _render has a context.
        self._pending_filters: dict = {}
        # Same idea for the colour: the dropdown's keys only exist once the
        # options have been built from a context, so the restored *column
        # name* waits here (see _apply_pending_color).
        self._pending_color_column: str = ""
        # column -> the kind the user forced it to be read as, when they
        # have. Attached to the column, so it drives the coloring *and* the
        # shape of that column's filter, and survives a recolor.
        self._column_kind: dict = {}
        self.filter_menu = ft.PopupMenuButton(
            content=ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.TUNE, size=18),
                    ft.Text("Filter points", size=13),
                ], spacing=8, tight=True),
                padding=ft.padding.symmetric(10, 14), border_radius=8,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            ),
            items=[],
        )
        self.filters_column = ft.Column(spacing=10)
        self.filters_card = ft.Container()
        # Shown beside "Color by", and only for a column that could be read
        # either way: what a number means is the user's call, not the
        # dtype's (see pc.default_kind).
        self.color_scale_switch = ft.SegmentedButton(
            visible=False, show_selected_icon=False, allow_empty_selection=False,
            selected={pc.KIND_NUMERIC},
            segments=[
                ft.Segment(value=pc.KIND_NUMERIC, label=ft.Text("Continuous", size=12)),
                ft.Segment(value=pc.KIND_CATEGORICAL, label=ft.Text("Discrete", size=12)),
            ],
            on_change=self._on_color_scale_change,
        )

        self._class_filter: list = []    # empty means every class
        self.class_menu = ft.PopupMenuButton(
            content=ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.FILTER_LIST, size=18),
                    ft.Text("Restrict to classes", size=13),
                ], spacing=8, tight=True),
                padding=ft.padding.symmetric(10, 14), border_radius=8,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            ),
            items=[],
        )
        self.class_chips = ft.Row(wrap=True, spacing=6, run_spacing=6)
        # Past FOCUS_MAX classes the menu above would list thousands of
        # entries: a search picker takes its place (see _rebuild_class_options).
        self._many_classes = False
        self.class_picker = ClassPicker(self._set_class_filter, limit=None, label="Restrict to classes")
        self.class_picker_box = self.class_picker.build()
        self.class_picker_box.visible = False
        # Which classes' prototypes go on the map -- apart from which
        # classes' points do. Every class's, while they are few (None);
        # past FOCUS_MAX the markers bury the points they are meant to
        # locate, so none until some are picked. Set before the run, like
        # the class restriction: prototypes are fitted with the points,
        # so they shape the map as well as sit on it.
        self._proto_filter: list | None = None
        self._proto_filter_chosen = False
        self.proto_picker = ClassPicker(self._set_proto_filter, limit=None, label="Show prototypes of")
        self.proto_summary = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        # exp name -> PROTO_FROM_* the user picked. Absent: see
        # _default_proto_source.
        self._proto_source_choice: dict = {}
        self.proto_source_dd = ft.Dropdown(
            label="Prototypes from", width=220, dense=True, border_radius=8,
            on_change=self._on_proto_source_change,
        )
        self.proto_source_note = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.proto_box = ft.Column([
            ft.Row([ft.Text("Prototypes", size=13, weight=ft.FontWeight.BOLD), self.proto_summary,
                    ft.TextButton("All classes", on_click=lambda e: self._set_proto_filter(None)),
                    ft.TextButton("None", on_click=lambda e: self._set_proto_filter([]))],
                   spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.proto_source_dd,
            self.proto_source_note,
            self.proto_picker.build(),
        ], spacing=6)
        self.run_projection_btn = ft.FilledButton("Run projection", icon=ft.Icons.BUBBLE_CHART, on_click=self._on_run_projection)

        # One dialog for the life of the view, not one per projection run:
        # every run builds a fresh InteractiveProjection, and a dialog
        # created alongside each of those would pile up in the page overlay
        # for as long as the tab stays open.
        self.carousel_title = ft.Text(weight=ft.FontWeight.BOLD)
        self.carousel_content = ft.Container()  # width comes from the carousel, which sizes itself to the page
        self.carousel_dialog = ft.AlertDialog(
            # modal=False so tapping outside dismisses; on_dismiss then has
            # to sync `open` back to False, which Flet does not do itself --
            # a stale True makes the next page.open() a no-op.
            modal=False, title=self.carousel_title, content=self.carousel_content,
            on_dismiss=lambda e: self.ctx.page.close(self.carousel_dialog),
            actions=[ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.carousel_dialog))],
        )

        # ---- class overlap (its own computation, its own button) ----
        self.overlap_btn = ft.OutlinedButton(
            "Measure class overlap", icon=ft.Icons.JOIN_INNER_OUTLINED,
            on_click=self._on_overlap_click)
        self.overlap_progress = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2),
             ft.Text("Measuring class overlap...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)],
            spacing=10, visible=False)
        self.overlap_results = ft.Column(spacing=10)
        self.overlap_job = BackgroundRebuild(ctx, "Class overlap")
        # Past FOCUS_MAX classes the overlap grid shows a focus set.
        self.overlap_focus = ClassFocus(OVERLAP_FOCUS_CRITERIA, self._on_overlap_focus_change,
                                        what="the overlap grid",
                                        support_criteria=(cf.FOCUS_LEAST_SEPARATED,))
        self._overlap = None  # (ClassOverlap, experiment name) last measured

        # Filled by a run: the map's legend lives in the left column, so it
        # is reserved here and populated in _run_projection.
        self._interactive = None
        self.legend_slot = ft.Container()
        # Full screen gives the map the window and takes the settings
        # column away; _map_size and _set_fullscreen are the two places
        # that care. The caption is held onto because it is the one piece
        # of chrome above the map that full screen can afford to drop.
        self._fullscreen = False
        self.fullscreen_btn = None
        # The captions that have to sit *above* the map rather than under
        # the figure: what the projection was fitted on (a restricted set
        # of classes, an overlay that is only placed onto the fit), since
        # nothing else on screen would tell the difference.
        self.projection_note = ft.Container()

        self.status_area = ft.Column(spacing=6)
        self.projection_results_container = ft.Container()
        self.controls_row = ft.Row(
            [self.experiment_dropdown, self.projection_method, self.projection_sample_field,
             self.color_by, self.color_scale_switch, self.class_menu, self.filter_menu,
             self.run_projection_btn],
            spacing=12, wrap=True,
        )
        self.root = ft.Container(padding=20)

    # ---- class overlap -----------------------------------------------
    def _on_overlap_click(self, e):
        exp_name = self.experiment_dropdown.value
        if not exp_name:
            return
        self.overlap_progress.visible = True
        self.overlap_results.controls = []
        for control in (self.overlap_progress, self.overlap_results):
            if control.page:
                control.update()
        # Pairwise similarity over every class, off the UI thread for the
        # same reason the projection is.
        self.overlap_job.run(lambda: self._compute_overlap(exp_name),
                             on_error=self._on_overlap_error)

    def _on_overlap_error(self, ex: Exception):
        self.overlap_progress.visible = False
        self.overlap_results.controls = [
            info_banner(f"Could not measure class overlap: {ex}", ft.Icons.ERROR_OUTLINE)]
        self._push_overlap()

    def _push_overlap(self):
        for control in (self.overlap_progress, self.overlap_results):
            if control.page:
                control.update()

    def _compute_overlap(self, exp_name: str):
        context = self._analysis_context(exp_name)
        if context is None:
            self.overlap_progress.visible = False
            self.overlap_results.controls = [
                info_banner(f"'{exp_name}' has no embeddings attached -- add a sidecar on the Data tab.")]
            self._push_overlap()
            return

        all_classes = sorted(set(context["labels"].tolist()))
        overlap = ea.class_overlap(context["embeddings"], context["labels"], class_order=all_classes)
        _log(f"class overlap for '{exp_name}': {len(overlap.labels)} classes")
        self._overlap = (overlap, exp_name)
        self.overlap_progress.visible = False
        self.overlap_results.controls = self._render_overlap(overlap, exp_name)
        self._push_overlap()

    def _on_overlap_focus_change(self):
        """Redraw the grid from the overlap already measured."""
        if self._overlap is None:
            return
        self.overlap_results.controls = self._render_overlap(*self._overlap)
        self._push_overlap()

    def _render_overlap(self, overlap, exp_name: str) -> list:
        if not overlap.labels:
            return [info_banner("No class has enough crops with embeddings to measure.")]

        finite = overlap.matrix[np.isfinite(overlap.matrix)]
        if not finite.size:
            return [info_banner("Not enough crops per class to compare.")]
        low, high = float(finite.min()), float(finite.max())
        span = max(high - low, 1e-9)

        # Past FOCUS_MAX classes the grid is the focus set, both ways; the
        # colour scale stays the whole matrix's so a cell reads the same
        # whichever classes are around it.
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
        header = ft.Row(
            [ft.Container(width=OVERLAP_LABEL_WIDTH)] + [
                ft.Container(content=ft.Text(short, size=9), width=OVERLAP_CELL,
                             alignment=ft.alignment.center, tooltip=l)
                for short, l in zip(short_labels, shown)
            ], spacing=2)
        rows = [header]
        for i in shown_idx:
            label_i = overlap.labels[i]
            controls = [ft.Container(
                content=ft.Text(label_i, size=10, weight=ft.FontWeight.BOLD),
                width=OVERLAP_LABEL_WIDTH, alignment=ft.alignment.center_right,
                padding=ft.padding.only(right=8))]
            for j in shown_idx:
                label_j = overlap.labels[j]
                value = overlap.matrix[i, j]
                if not np.isfinite(value):
                    controls.append(ft.Container(width=OVERLAP_CELL, height=OVERLAP_CELL,
                                                 border_radius=4, bgcolor=ft.Colors.SURFACE))
                    continue
                fraction = (value - low) / span
                note = ("how alike this class is to itself" if i == j
                        else "how alike these two classes are")
                controls.append(ft.Container(
                    width=OVERLAP_CELL, height=OVERLAP_CELL, border_radius=4,
                    bgcolor=blues_color(fraction), alignment=ft.alignment.center,
                    content=ft.Text(f"{value:.2f}", size=9,
                                    color="#ffffff" if fraction > 0.55 else "#111111"),
                    tooltip=f"{label_i} vs {label_j}\nmean cosine similarity {value:.4f}\n({note})",
                ))
            rows.append(ft.Row(controls, spacing=2))

        pairs = overlap.top_pairs(8)
        pair_rows = [ft.Text("Most-overlapping pairs", size=12, weight=ft.FontWeight.BOLD)]
        for label_a, label_b, value in pairs:
            pair_rows.append(ft.Text(f"{label_a} / {label_b}  —  {value:.3f}", size=12))

        separation_rows = [ft.Text("Least separated classes", size=12, weight=ft.FontWeight.BOLD),
                           ft.Text("Own-class similarity minus the closest other class. At or below "
                                   "zero, a class is no more like itself than like its neighbour.",
                                   size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)]
        order = np.argsort(np.where(np.isfinite(overlap.separation), overlap.separation, np.inf))
        for idx in order[:8]:
            value = overlap.separation[idx]
            if not np.isfinite(value):
                continue
            separation_rows.append(ft.Text(
                f"{overlap.labels[idx]}  —  {value:+.3f}  ({int(overlap.counts[idx]):,} crops)",
                size=12, color=ft.Colors.RED_400 if value <= 0 else None))

        return [
            ft.Text(f"Measured in '{exp_name}' over {int(overlap.counts.sum()):,} crops.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            self.overlap_focus.build(),
            ft.Row([ft.Column(rows, spacing=2)], scroll=ft.ScrollMode.AUTO),
            ft.Row([
                card(ft.Column(pair_rows, spacing=4), width=340),
                card(ft.Column(separation_rows, spacing=4), width=380),
            ], spacing=12, wrap=True, vertical_alignment=ft.CrossAxisAlignment.START),
        ]

    # ---- helpers -----------------------------------------------------

    def _analysis_context(self, exp_name: str):
        context = build_analysis_context(self.ctx.state, exp_name)
        self._last_context = context
        return context

    def _rebuild_class_options(self, ctx: dict) -> None:
        """One menu entry per class, ticked when it is in the filter, so
        the menu doubles as the readout of what is on. The class label is
        carried in the closure rather than encoded into an item id -- a
        label comes from the user's CSV and must never double as a
        technical key."""
        classes = sorted(set(ctx["labels"].tolist()))
        # Classes this experiment no longer has cannot stay picked.
        present = set(classes)
        self._class_filter = [c for c in self._class_filter if c in present]
        self._many_classes = len(classes) > FOCUS_MAX
        if not self._proto_filter_chosen:
            self._proto_filter = [] if self._many_classes else None
        elif self._proto_filter is not None:
            self._proto_filter = [c for c in self._proto_filter if c in present]
        self.proto_picker.set_labels(classes)
        self._sync_proto_controls()
        # Always something to draw: the dataset's own labels give a
        # prototype per class even when the network brought none.
        self._sync_proto_source(ctx["exp"].name, classes)
        self.class_menu.visible = not self._many_classes
        self.class_picker_box.visible = self._many_classes
        if self._many_classes:
            self.class_menu.items = []
            self.class_picker.set_labels(classes)
            self.class_picker.set_picked(self._class_filter)
            if self.class_picker_box.page:
                self.class_picker_box.update()
        else:
            self.class_menu.items = [
                ft.PopupMenuItem(text="All classes", on_click=lambda e: self._set_class_filter([])),
            ] + [
                ft.PopupMenuItem(
                    text=str(c), checked=c in self._class_filter,
                    on_click=lambda e, label=c: self._toggle_class(label),
                )
                for c in classes
            ]
        if self.class_menu.page:
            self.class_menu.update()
        self._rebuild_class_chips()

    def _toggle_class(self, label) -> None:
        if label in self._class_filter:
            self._set_class_filter([c for c in self._class_filter if c != label])
        else:
            self._set_class_filter(self._class_filter + [label])

    def _set_class_filter(self, classes: list) -> None:
        self._class_filter = list(classes)
        # Past FOCUS_MAX classes the "Color by" menu lists the score columns
        # of the restricted classes only, so it follows the restriction.
        if self._last_context is not None and self._many_classes:
            self._rebuild_color_options(self._last_context)
            if self.color_by.page:
                self.color_by.update()
        if self._many_classes:
            self.class_picker.set_picked(self._class_filter)
            if self.class_picker_box.page:
                self.class_picker_box.update()
        for item in self.class_menu.items[1:]:
            item.checked = item.text in {str(c) for c in self._class_filter}
        if self.class_menu.page:
            self.class_menu.update()
        self._rebuild_class_chips()
        # Deliberately not re-running on its own: t-SNE and UMAP take tens
        # of seconds, and picking three classes in a row would leave three
        # of them churning in background threads for two results nobody
        # will see. The same "run it again" prompt the other invalidations
        # use is one click and no surprise.
        self._reset_result("Class selection changed — run the projection again.")
        if self.projection_results_container.page:
            self.projection_results_container.update()

    def _set_proto_filter(self, classes) -> None:
        """None for every class's prototypes, else a list (empty: none)."""
        self._proto_filter = None if classes is None else list(classes)
        self._proto_filter_chosen = True
        self._sync_proto_controls()
        self._reset_result("Prototype selection changed — run the projection again.")
        for control in (self.proto_box, self.projection_results_container):
            if control.page:
                control.update()

    def _proto_coverage(self, exp_name: str, classes: list) -> int | None:
        """How many of `classes` the network's prototypes cover, once its
        class names went through the class mapping; None without any."""
        model = self.ctx.prototype_source.model_set_for_dataset(exp_name)
        if model is None:
            return None
        index = model.label_index()
        return sum(1 for c in classes
                   if c in index and not np.isnan(model.vectors[index[c]]).all())

    def _default_proto_source(self, exp_name: str, classes: list) -> str | None:
        """The network's set when it has a prototype for every class of the
        dataset, the dataset's own when it misses some. With no network
        prototypes at all, none -- unless the Prototypes tab made a k-means
        estimate active, which the map has always drawn -- rather than
        putting markers on a map nobody asked to have them on."""
        covered = self._proto_coverage(exp_name, classes)
        if covered is None:
            source, _ = self.ctx.prototype_source.resolve(exp_name)
            return PROTO_FROM_DATASET if source == "kmeans" else None
        return PROTO_FROM_MODEL if covered == len(classes) else PROTO_FROM_DATASET

    def _proto_source_for(self, exp_name: str, classes: list) -> str | None:
        chosen = self._proto_source_choice.get(exp_name)
        if chosen == PROTO_FROM_MODEL and self._proto_coverage(exp_name, classes) is None:
            chosen = None  # its sidecar is gone
        return chosen or self._default_proto_source(exp_name, classes)

    def _sync_proto_source(self, exp_name: str, classes: list) -> None:
        covered = self._proto_coverage(exp_name, classes)
        options = []
        if covered is not None:
            options.append(ft.dropdown.Option(PROTO_FROM_MODEL, "Network (imported)"))
        options.append(ft.dropdown.Option(PROTO_FROM_DATASET, "Dataset labels"))
        self.proto_source_dd.options = options
        self.proto_source_dd.value = self._proto_source_for(exp_name, classes)
        if covered is None:
            self.proto_source_note.value = (
                "No network prototypes loaded for this experiment -- pick 'Dataset labels' to draw "
                "one per class.")
        elif covered < len(classes):
            self.proto_source_note.value = (
                f"The network's prototypes cover {covered:,} of the dataset's {len(classes):,} classes "
                f"(after the Data tab's class mapping).")
        else:
            self.proto_source_note.value = ""
        self.proto_source_note.visible = bool(self.proto_source_note.value)

    def _on_proto_source_change(self, e) -> None:
        exp_name = self.experiment_dropdown.value
        if not exp_name:
            return
        self._proto_source_choice[exp_name] = self.proto_source_dd.value
        self._reset_result("Prototype source changed — run the projection again.")
        if self.projection_results_container.page:
            self.projection_results_container.update()

    def _sync_proto_controls(self) -> None:
        picked = self._proto_filter
        self.proto_picker.set_picked(picked or [])
        if picked is None:
            self.proto_summary.value = "every class" + (
                f" (the first {PROTO_CLASSES_MAX} alphabetically)" if self._many_classes else "")
        elif not picked:
            self.proto_summary.value = "none"
        else:
            self.proto_summary.value = f"{len(picked)} class(es)"

    def _rebuild_class_chips(self) -> None:
        # The picker shows its own chips when it stands in for the menu.
        self.class_chips.controls = [] if self._many_classes else [
            ft.Chip(
                label=ft.Text(str(c), size=11),
                on_delete=lambda e, label=c: self._toggle_class(label),
                delete_icon_tooltip="Remove this class",
            )
            for c in self._class_filter
        ]
        if self.class_chips.page:
            self.class_chips.update()

    def _map_size(self) -> int:
        """Side of the square map, from the window.

        Half the width, bounded by the height left over once the app's
        chrome is taken out -- on a 16:9 screen the height is what runs out
        first, which is exactly why the map sits in a half-width column
        instead of across the page. page.width/height are None before the
        first frame, so the declared window size stands in.

        Full screen lifts the width term -- there is no settings column
        beside the map any more -- and raises the cap. The height term
        only loosens by the chrome that actually went away, so on a 16:9
        window the map still ends up height-bound: it is a *square*, and
        no amount of freed width changes that."""
        width = self.ctx.page.width or 1440
        height = self.ctx.page.height or 920
        if self._fullscreen:
            pane = width - 2 * FULLSCREEN_PADDING
            cap, chrome = MAP_MAX_FULLSCREEN, FULLSCREEN_CHROME_HEIGHT
        else:
            pane = width * 0.5 - 40  # minus the gap between the two panes
            cap, chrome = MAP_MAX, CHROME_HEIGHT
        return int(max(MIN_PLOT_SIZE, min(cap, pane, height - chrome)))

    def _set_fullscreen(self, on: bool) -> None:
        """Give the map the window, or give the settings column back.

        The map control is deliberately *not* re-parented: it stays in the
        pane it was built in, which is the pane with no scrolling ancestor.
        Moving it would put the wheel back in competition with a scrollbar
        (see build()), so only what sits around it changes."""
        self._fullscreen = on
        self.left_pane.visible = not on
        self.root.padding = FULLSCREEN_PADDING if on else 20
        # A square map rarely uses all the width it is offered, and left
        # against the edge of a wide window it reads as a layout bug.
        self.panes_row.alignment = (
            ft.MainAxisAlignment.CENTER if on else ft.MainAxisAlignment.START
        )
        if self.fullscreen_btn is not None:
            self._dress_fullscreen_button(self.fullscreen_btn)
        size = self._map_size()
        self.map_pane.width = size
        if self.root.page:
            self.root.update()
        # After the update: resize() pushes the new square itself, and it
        # has to land on a pane that has already been given the room.
        if self._interactive is not None:
            self._interactive.resize(size)

    def _dress_fullscreen_button(self, button: ft.IconButton) -> None:
        button.icon = ft.Icons.FULLSCREEN_EXIT if self._fullscreen else ft.Icons.FULLSCREEN
        button.tooltip = (
            "Leave full screen" if self._fullscreen
            else "Full screen -- hide the settings column and fill the window"
        )

    def on_window_resized(self) -> None:
        """The map is sized from the window, so it has to follow it. Only
        the map: the left column scrolls, so it needs no help. _map_size
        already knows whether we are full screen, so this one path serves
        both layouts."""
        size = self._map_size()
        self.map_pane.width = size
        if self.map_pane.page:
            self.map_pane.update()
        if self._interactive is not None:
            self._interactive.resize(size)

    def _refresh_experiment_dropdown(self):
        names = experiments_with_embeddings(self.ctx.state)
        self.experiment_dropdown.options = [name_option(n) for n in names]
        if self.experiment_dropdown.value not in names:
            self.experiment_dropdown.value = names[0] if names else None
        # An experiment that is gone (unloaded, or its sidecar detached)
        # takes its cached run with it, rather than waiting to be restored
        # under a name that may later belong to different data.
        for gone in [n for n in self._runs if n not in names]:
            del self._runs[gone]

    def _clear_runs(self) -> None:
        """Forget every cached run. For when what a projection was computed
        *from* changes rather than which one is being looked at: new
        embeddings or a new prototype source make every cached map wrong,
        including the ones for experiments not on screen. The epoch bump is
        what stops a run still in flight from re-filling the cache with a
        result computed from the data that just went away."""
        self._runs.clear()
        self._runs_epoch += 1

    def _reset_result(self, message: str = "Click 'Run projection' to compute.") -> None:
        """Drop whatever projection is on screen. Called whenever the data
        under it changes -- a different experiment, a re-attached sidecar, a
        different prototype source -- because a projection that no longer
        matches what it is captioned with is worse than no projection at
        all. Bumping the generation also makes any run still in flight
        discard its own result when it lands.

        This clears the *screen*, not the cache: the run being taken down
        stays in _runs so that coming back to its experiment can put it
        straight back up. Data changes clear the cache too, and say so by
        calling _clear_runs as well."""
        self._projection_generation += 1
        self._has_result = False
        # The cached run goes with the result it produced: recoloring a
        # projection that is no longer on screen would put a map back up
        # that the caller just decided no longer matches its data.
        self._last_result = None
        self.projection_results_container.content = info_banner(message)
        # The legend belongs to the projection that is going away, so it
        # goes with it rather than describing a map no longer on screen.
        self._interactive = None
        self.legend_slot.content = None
        self.projection_note.content = None
        for control in (self.legend_slot, self.projection_note):
            if control.page:
                control.update()
        # Full screen is only ever entered from the map's own toolbar, and
        # that toolbar just went away with the map -- staying in it would
        # hide the settings column with nothing left on screen to bring it
        # back, including the controls that run the next projection.
        if self._fullscreen:
            self._set_fullscreen(False)

    def _on_experiment_change(self, e):
        """Switch to another experiment's projection -- which is a redraw
        when that experiment has already been run, and the prompt to run it
        when it has not.

        The context is built here rather than left to _render because the
        cached run has to be checked against it first (see _rebind), and
        building it twice means merging the frame twice."""
        self._reset_result()
        exp_name = self.experiment_dropdown.value
        ctx = self._analysis_context(exp_name) if exp_name else None
        restored = self._rebind(self._runs.get(exp_name), ctx) if ctx is not None else None
        if restored is not None:
            # The controls describe the map: showing a t-SNE of three
            # classes under a dropdown that reads "UMAP" and no chips would
            # be a lie, and the next run would be of something else again.
            # Assigning a value from Python does not fire on_change, so
            # none of this re-runs the projection.
            self.projection_method.value = restored.method
            self.projection_sample_field.value = str(restored.n_sample)
            self._class_filter = list(restored.class_filter)
            self._proto_filter = None if restored.proto_filter is None else list(restored.proto_filter)
            self._proto_filter_chosen = True
            self._sync_proto_controls()
            if restored.source_key:
                self._proto_source_choice[exp_name] = restored.source_key
        elif exp_name in self._runs:
            # The rows moved under it (rows added, removed or reordered),
            # so its sample no longer points at what it was drawn from.
            del self._runs[exp_name]
        self._render(ctx)
        if restored is not None:
            self._show_restored(restored)
        if self.root.page:
            self.root.update()

    def _rebind(self, cached: "_RunResult | None", ctx: dict) -> "_RunResult | None":
        """A cached run, put back on top of a freshly built context -- or
        None when it cannot honestly be.

        `idx` are positions in the context the run was computed from, so
        reusing them is only valid if the same rows still sit at the same
        positions; the sampled uuids recorded alongside them are what
        proves it. Everything that follows from the labels is taken from
        the new context rather than from the run, so a crop relabelled
        since then is drawn in its new class."""
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
        # A class that is no longer in the data cannot stay in the filter,
        # and the caption ("N of M classes") would be wrong if it did.
        if [c for c in cached.class_filter if c not in all_classes]:
            return None
        pool_size = (int(np.isin(labels, cached.class_filter).sum())
                     if cached.class_filter else len(labels))
        return replace(cached, ctx=ctx, labels=labels[idx],
                       all_classes=all_classes, pool_size=pool_size)

    def _show_restored(self, result: "_RunResult") -> None:
        """Put a cached run back on screen. Same path a recolor takes: off
        the UI thread, under a fresh generation, since rebuilding the
        figure and the canvas is not instant even though projecting again
        is what it avoids."""
        self._projection_generation += 1
        generation = self._projection_generation
        self._last_result = result
        self.projection_results_container.content = ft.Row(
            [ft.ProgressRing(width=16, height=16, stroke_width=2),
             ft.Text("Restoring projection...", size=12)],
            spacing=10,
        )
        if self.projection_results_container.page:
            self.projection_results_container.update()
        self.ctx.page.run_thread(self._render_projection, result, generation)

    def on_prototype_source_changed(self) -> None:
        """Subscribed to the prototype-source registry: an attached sidecar
        or a newly selected source changes both the point cloud and the
        prototypes drawn over it. Only worth explaining when there was
        something on screen to invalidate -- on a first sidecar attach
        nothing has been computed yet, and the plain prompt is the honest
        message.

        Every cached run goes, not only the one on screen: the change is
        to the data the projections were computed from, and an experiment
        that is not being looked at right now is no more immune to it than
        the one that is."""
        had_cached = self._has_result or bool(self._runs)
        self._clear_runs()
        if had_cached:
            self._reset_result("Embeddings or prototype source changed — run the projection again.")
        else:
            self._reset_result()
        self.refresh()

    def _open_crops(self, indices: list, records: list, n_pool: int, title: str = "") -> None:
        """Show a group of points as crops. The carousel pages lazily (see
        ui/carousel.py), so a loop thrown around a whole cluster costs one
        page of images, not one per selected point.

        Points are named by position (see InteractiveProjection.records):
        below n_pool they are dataset rows, above it they are crops from an
        overlaid inference. The review controls follow that line -- a
        decision is a statement about a *dataset* row, keyed by a uuid the
        review store knows, so a selection that reaches outside the dataset
        is shown without them rather than offering a verdict that would be
        written against nothing. "Find similar" is offered either way: it
        is the one thing an inference crop is on the map for.
        """
        picked = [records[i] for i in indices if 0 <= i < len(records) and records[i]]
        if not picked:
            return
        from_dataset = all(i < n_pool for i in indices)
        cols = self.ctx.state.dataset.columns
        self.carousel_title.value = title or f"Lasso selection — {len(picked):,} images"
        if not from_dataset:
            self.carousel_title.value += " — inference crops, no review"
        self.carousel_content.content = build_carousel_from_list(
            self.ctx.state.dataset, picked, cols.crop_col, cols.uuid_col, ctx=self.ctx,
            review_source="projection" if from_dataset else "",
            label_col=cols.label_col,
        )
        self.ctx.page.open(self.carousel_dialog)

    def _fit_note(self, result: "_RunResult", layers: list, n_fitted: int) -> str:
        """What the map was fitted on, and what was merely placed onto it.

        Worth a line of the screen because the two pictures are genuinely
        different claims: points fitted together were free to arrange
        themselves around each other, while a point placed onto an existing
        fit was put where the map says it goes -- approximately, for UMAP,
        which learns that placement from the neighbours of what it was
        fitted on."""
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

    # ---- projection (threaded so a re-click cancels the stale result) ---

    def _on_run_projection(self, e):
        exp_name = self.experiment_dropdown.value
        if not exp_name:
            return
        ctx = self._analysis_context(exp_name)
        if ctx is None:
            return
        try:
            n_sample = int(self.projection_sample_field.value)
        except ValueError:
            n_sample = 1500

        self._projection_generation += 1
        generation = self._projection_generation
        method = self.projection_method.value

        self.projection_results_container.content = ft.Row(
            [ft.ProgressRing(width=16, height=16, stroke_width=2), ft.Text(f"Computing {method}...", size=12)],
            spacing=10,
        )
        if self.projection_results_container.page:
            self.projection_results_container.update()

        # Heavy, CPU-bound (t-SNE/UMAP) work runs off the UI thread via
        # Flet's own executor (page.run_thread) so the spinner above
        # actually paints instead of the whole app freezing until it's
        # done. `generation` lets a *newer* click's result win even if an
        # older, still-running computation finishes later -- the old one
        # isn't forcibly stopped (plain Python can't safely do that), its
        # result is just discarded instead of overwriting the new one.
        # Passed by value, not read from self inside the worker: the user
        # can change the picks while a run is in flight, and the result
        # that lands has to match the caption it is given.
        self._computing = True
        self.ctx.page.run_thread(
            self._run_projection, ctx, exp_name, method, n_sample, generation,
            list(self._class_filter), self._runs_epoch,
            self._selected_overlay_names(), self._refit_active(),
            None if self._proto_filter is None else list(self._proto_filter),
            self._proto_source_for(exp_name, sorted(set(ctx["labels"].tolist()))))

    # The map on screen was made by one of four methods, and what may be
    # read off it differs sharply between them -- so the (i) opens the page
    # for the method currently selected.
    METHOD_MANUAL_PAGES = {
        "pca": "pca",
        "tsne": "tsne",
        "umap": "umap",
        "umap_circle": "umap",
    }

    def _projection_manual_page(self) -> str:
        return self.METHOD_MANUAL_PAGES.get(self.projection_method.value, "projection-2d")

    def _on_method_change(self, e):
        """Switching method re-runs, as it always has -- but the refit
        switch has to follow first: with t-SNE selected there is no choice
        left to make, and the run below reads the switch."""
        self._sync_refit_control()
        self._on_run_projection(e)

    def _switcher(self, interactive_view, static_view, toolbar: list | None = None) -> ft.Control:
        """Both renderings, one shown at a time, under a single line of
        controls.

        They are both built and mounted, and only visibility is toggled:
        switching back and forth must not recompute or re-encode anything,
        since the projection behind them can take tens of seconds to
        produce.

        The one line is the point. The map is square and height-bound on
        any 16:9 window (see _map_size), so every row of chrome above it is
        subtracted from the picture -- which is why the map's own controls
        are pulled up here from interactive_projection rather than drawn
        over the plot, and why there is no caption line any more.
        """
        interactive_box = ft.Container(content=interactive_view)
        static_box = ft.Container(content=static_view, visible=False)
        def on_change(e):
            selected = next(iter(e.control.selected), "interactive")
            interactive_box.visible = selected == "interactive"
            static_box.visible = selected == "static"
            # The map's controls go with it: they drive the canvas and do
            # nothing to a PNG. Asking the projection rather than setting
            # visible here, because half of those buttons own their own
            # visibility -- see InteractiveProjection._sync_toolbar.
            if self._interactive is not None:
                self._interactive.set_toolbar_visible(selected == "interactive")
            if self.projection_results_container.page:
                self.projection_results_container.update()

        # Icons without labels: this switch and the map's Pan/Lasso/Zoom
        # have to share a line with the zoom controls, and two sets of
        # words do not fit the half-width pane. Tooltips carry the names --
        # Pan/Lasso/Zoom keeps its labels, being the one a user reaches for
        # mid-task and the one whose icons say least on their own.
        switch = ft.SegmentedButton(
            selected={"interactive"},
            on_change=on_change,
            show_selected_icon=False,
            segments=[
                ft.Segment(value="interactive", tooltip="Interactive map",
                           icon=ft.Icon(ft.Icons.TOUCH_APP_OUTLINED)),
                ft.Segment(value="static", tooltip="Static matplotlib figure",
                           icon=ft.Icon(ft.Icons.IMAGE_OUTLINED)),
            ],
        )
        # Built fresh with every switcher rather than kept as one control:
        # a run replaces this whole subtree, and the button has to sit in
        # this row rather than among the map's own controls -- those are
        # hidden in "Static", which would strand a full-screen user with no
        # way back.
        self.fullscreen_btn = ft.IconButton(icon_size=20, on_click=lambda e: self._set_fullscreen(not self._fullscreen))
        self._dress_fullscreen_button(self.fullscreen_btn)
        # One flat row, not a row of rows. A nested Row is laid out as a
        # single block: it cannot share the first line with these controls
        # and, having no width of its own to wrap inside, it runs off the
        # edge of the pane instead -- which is what cut the end off "Reset
        # view". Flat, every control wraps individually and in order, so
        # the line only becomes two once a lasso adds its buttons.
        header = ft.Row(
            [switch, self.fullscreen_btn, *(toolbar or [])],
            spacing=8, wrap=True, run_spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        return ft.Column([header, interactive_box, static_box], spacing=10)

    def _run_projection(self, ctx: dict, exp_name: str, method: str, n_sample: int,
                        generation: int, class_filter: list, epoch: int,
                        overlay_names: list | None = None, refit_with_overlay: bool = False,
                        proto_filter: list | None = None, proto_source: str | None = PROTO_FROM_MODEL):
        """The expensive half: sample, gather the prototypes, project. The
        result is handed to _render_projection, which owns everything that
        can be redone without projecting again -- which is what recoloring
        is."""
        try:
            rng = np.random.default_rng(0)
            all_classes = sorted(set(ctx["labels"].tolist()))
            # The sample is drawn from the picked classes rather than from
            # the whole pool, so "sample size" keeps meaning "points you
            # will see" -- picking a rare class out of a 10k pool would
            # otherwise put a handful of it on screen.
            #
            # The consequence, which the caption states: the projection is
            # then *fitted to those classes*. It uses its whole budget to
            # spread them, which is what makes a single class worth looking
            # at, but the coordinates are not comparable with another
            # selection's -- a point is not in the same place in two
            # different runs.
            pool = np.arange(len(ctx["embeddings"]))
            if class_filter:
                pool = np.flatnonzero(np.isin(ctx["labels"], class_filter))
            if not len(pool):
                # Cannot normally happen -- picks are pruned to the classes
                # this experiment has -- but returning here would skip the
                # push at the end of the method and leave the spinner up,
                # so the banner is pushed on the way out.
                self.projection_results_container.content = info_banner(
                    "No rows in the selected classes.", icon=ft.Icons.WARNING_AMBER_OUTLINED)
                if self.projection_results_container.page:
                    self.projection_results_container.update()
                return
            idx = rng.choice(pool, size=min(n_sample, len(pool)), replace=False)
            embs = ctx["embeddings"][idx]
            labels = ctx["labels"][idx]

            # Each prototype row is kept alongside *which* class and *which*
            # subcenter it is, so the plot below can color it like its class
            # and shape it by subcenter. (An earlier version filtered NaN
            # rows out with a mask, which silently lost the subcenter index
            # -- every prototype then had to be drawn identically.)
            proto_vecs = np.zeros((0, embs.shape[1]))
            proto_labels: list = []
            proto_ks: list = []
            n_subcenters = 0
            source_key = source_color = None
            source_note = ""
            registry = self.ctx.prototype_source
            proto_set = None
            if proto_source == PROTO_FROM_MODEL:
                proto_set = registry.model_set_for_dataset(exp_name)
                source_color, source_note = MODEL_COLOR, "network prototypes (imported)"
            if proto_set is None and proto_source == PROTO_FROM_DATASET:
                # Built from every row, not only the sample: a class mean
                # over the forty points on screen is not that class's mean.
                proto_set, source_note = registry.dataset_set(exp_name, ctx["embeddings"], ctx["labels"])
                proto_source, source_color = PROTO_FROM_DATASET, KMEANS_COLOR
            if proto_set is not None:
                source_key = proto_source
                # Every class the source has, not only the sampled ones:
                # when the picker narrows the points to two classes, the
                # other prototypes are precisely what tells you where those
                # two sit in the wider space. Still capped, since the
                # markers are drawn one per subcenter.
                # Whose prototypes: every class's (capped, since the markers
                # are drawn one per subcenter and are fitted with the
                # points), or the ones picked under "Prototypes".
                wanted = all_classes if proto_filter is None else [l for l in all_classes if l in set(proto_filter)]
                present = [l for l in wanted if l in proto_set.label_index()][:PROTO_CLASSES_MAX]
                rows = []
                for l in present:
                    vectors = proto_set.vectors[proto_set.label_index()[l]]  # (K, D)
                    for k_idx in range(vectors.shape[0]):
                        # NaN = this class was absent from whatever pool the
                        # prototypes were built from; skip it but keep every
                        # other subcenter's real index intact.
                        if np.isnan(vectors[k_idx]).any():
                            continue
                        rows.append(vectors[k_idx])
                        proto_labels.append(l)
                        proto_ks.append(k_idx)
                if rows:
                    proto_vecs = np.stack(rows)
                    n_subcenters = max(proto_ks) + 1

            # Simple inferences, when they are to be part of the fit
            # rather than placed onto it: their vectors go in with the
            # pool's, and the coordinates are split back out below in the
            # same order. (With the switch off, none of this runs -- the
            # renderer places them through the fitted model instead, which
            # is what leaves the map where it was.)
            state = self.ctx.state
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

            t0 = time.monotonic()
            fit = ea.fit_projection(
                fit_input, method=method, proto_vectors=proto_vecs if len(proto_vecs) else None)
            _log(f"projection ({method}) for '{exp_name}' computed in {time.monotonic() - t0:.1f}s "
                 f"over {len(fit_input):,} vectors")
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
                source_note=source_note, n_sample=n_sample, sample_uuids=ctx["uuids"][idx], run_generation=generation,
                fit=fit, overlay_names=list(overlay_rows) if refit_with_overlay else [],
                refit_with_overlay=bool(refit_with_overlay and overlay_rows),
                overlay_proj=overlay_proj, overlay_rows=overlay_rows,
            )
        except Exception as ex:
            _log(f"error computing projection: {ex}")
            traceback.print_exc()
            if generation != self._projection_generation:
                return
            self.projection_results_container.content = info_banner(f"Error: {ex}", icon=ft.Icons.ERROR_OUTLINE)
            if self.projection_results_container.page:
                self.projection_results_container.update()
            return
        finally:
            self._computing = False

        # Filed under its experiment whether or not it still has the
        # screen: what superseded it may well be a switch to another
        # experiment, and the projection has been computed either way --
        # dropping it is precisely what used to make going back and forth
        # cost one t-SNE each way. Two guards, both about the result being
        # *wrong* rather than merely late: an epoch bump means the data it
        # was computed from is gone, and a later run of the same experiment
        # may have landed first (nothing stops the old one from finishing
        # afterwards).
        if epoch == self._runs_epoch:
            previous = self._runs.get(exp_name)
            if previous is None or previous.run_generation < generation:
                self._runs[exp_name] = replace(result, ctx=None)

        if generation != self._projection_generation:
            _log(f"projection ({method}) for '{exp_name}' kept but not shown -- "
                 "superseded by a newer request")
            return
        # Kept so that changing the coloring can redraw from it rather than
        # projecting the same points a second time. Set before the render,
        # since the render is what a later recolor is redoing.
        self._last_result = result
        self._render_projection(result, generation)

    # ---- overlaid simple inferences ------------------------------------

    def _rebuild_overlay_controls(self) -> None:
        """One checkbox per loaded simple inference, plus the refit switch.

        Rebuilt from the loaded sets rather than kept in step by hand: a
        set can be loaded, replaced or removed on the Data tab at any time,
        and a checkbox for one that is gone would tick nothing. What
        survives the rebuild is the *selection*, which is held separately
        for exactly that reason."""
        inferences = self.ctx.state.inferences
        kept = {n for n in self._overlay_selection if n in inferences}
        # A set that was unloaded (or reloaded under another name) while its
        # points were on the map takes them with it: leaving them drawn
        # would be a map showing crops from a file that is no longer open.
        dropped = self._overlay_selection - kept
        self._overlay_selection = kept
        self._overlay_checks = {}
        controls = []
        for name, inference in inferences.items():
            shown = len(overlay_indices(len(inference.df)))
            suffix = f"{shown:,} crops" if shown == len(inference.df) else \
                f"{shown:,} of {len(inference.df):,} crops"
            check = ft.Checkbox(
                label=f"{short_name(name)} ({suffix})", tooltip=name, value=name in self._overlay_selection,
                on_change=lambda e, n=name: self._on_overlay_toggle(n, bool(e.control.value)),
            )
            self._overlay_checks[name] = check
            controls.append(check)
        self.overlay_column.controls = controls
        self._sync_refit_control()
        if self.overlay_column.page:
            self.overlay_column.update()
        if dropped and self._last_result is not None:
            for name in dropped:
                self._last_result.overlay_proj.pop(name, None)
                self._last_result.overlay_rows.pop(name, None)
            self._redraw_current("Updating the map...")

    def _sync_refit_control(self) -> None:
        """Keep the refit switch honest about what the method can do.

        t-SNE has no way to place a point it was not fitted on (see
        ea.OutOfSampleUnsupported), so with it there is no choice to
        offer -- the switch is forced on and locked, and says why. The
        user's own preference is remembered underneath, so going back to
        PCA or UMAP restores it rather than leaving the forced value
        behind."""
        forced = self.projection_method.value not in ea.OUT_OF_SAMPLE_METHODS
        self.refit_check.disabled = forced
        self.refit_check.value = True if forced else self._refit_choice
        self.refit_check.tooltip = (
            "t-SNE cannot place points it was not fitted on, so overlaid crops are always part of "
            "the fit -- which moves every other point too."
            if forced else
            "Off: the map holds still and the overlaid crops are placed onto the fit it already "
            "has. On: everything is fitted together, so every point moves."
        )
        self.overlay_note.value = (
            "t-SNE learns no reusable map, so overlaid crops are fitted with the rest and the "
            "whole projection changes. PCA and UMAP can place them onto the existing fit."
            if forced else ""
        )
        for control in (self.refit_check, self.overlay_note):
            if control.page:
                control.update()

    def _refit_active(self) -> bool:
        return bool(self.refit_check.value)

    def _selected_overlay_names(self) -> list:
        """The ticked sets that are actually loaded, in the order they were
        loaded -- so the legend and the caption list them the same way
        twice running."""
        return [n for n in self.ctx.state.inferences if n in self._overlay_selection]

    def _on_overlay_toggle(self, name: str, on: bool) -> None:
        if on:
            self._overlay_selection.add(name)
        else:
            self._overlay_selection.discard(name)
        self._overlay_changed()

    def _on_refit_change(self, e):
        self._refit_choice = bool(e.control.value)
        self._overlay_changed()

    def _overlay_changed(self) -> None:
        """What a change to the overlay costs, which depends entirely on
        the refit switch.

        With it off, the map is untouched and the new points are placed
        onto the fit already in hand -- a redraw, exactly like a recolor.
        With it on, the overlay is part of what the projection is computed
        from, so the answer is a new projection, and that is asked for
        rather than launched: t-SNE and UMAP take tens of seconds, and
        ticking three sets in a row would start three of them."""
        if not self._has_result and self._last_result is None:
            return
        if self._refit_active():
            self._reset_result("Overlay changed — run the projection again to fit it in.")
            if self.projection_results_container.page:
                self.projection_results_container.update()
            return
        self._redraw_current("Placing points...")

    def _overlay_layers(self, result: "_RunResult") -> list:
        """The ticked inferences as they appear on this map: where their
        points land, and the rows behind them.

        Coordinates come from the run itself when the overlay was part of
        the fit, and otherwise from the fitted model -- computed once and
        kept on the result, so switching the coloring six times does not
        pay for six transforms.
        """
        state = self.ctx.state
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
                # Two models' embedding spaces have nothing to do with each
                # other even when they happen to share a width, so this is
                # the cheap half of a mistake worth catching: the expensive
                # half (same width, different model) nothing can detect.
                _log(f"overlay '{name}' skipped: {problem}")
                self._overlay_problems[name] = problem
                continue
            rows = result.overlay_rows.get(name)
            coords = result.overlay_proj.get(name)
            if coords is None or rows is None:
                fit = result.fit
                if fit is None or not fit.supports_transform():
                    # Only reachable for a run whose fit is gone; ticking a
                    # set with t-SNE selected asks for a re-run instead.
                    continue
                rows = overlay_indices(len(inference.df))
                t0 = time.monotonic()
                try:
                    coords = fit.transform(inference.embeddings[rows])
                except Exception as ex:
                    # A failed placement must not take the map down with
                    # it: the projection is still valid, this one set is
                    # not on it, and the caption says so.
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
                name=name, coords=np.asarray(coords, dtype=float),
                uuids=inference.uuids()[rows],
                records=inference.display_records(cols.uuid_col, cols.crop_col, rows=rows),
                rows=rows, total=len(inference.df),
            ))
        return layers

    # ---- values of a column, wherever the points come from -------------

    def _column_reference(self, ctx: dict, column: str):
        """Every value of `column`, across the joined frame and every
        loaded inference.

        The reference, in the sense core/point_coloring gives the word: it
        is what decides a value's color and what a filter offers, so it has
        to cover every point that could be drawn rather than the ones
        currently on screen. Otherwise a color would change as the sample
        changed, and a filter's boxes would rearrange themselves as another
        filter narrowed the cloud.
        """
        merged = ctx["merged"]
        parts = [merged[column]] if column in merged.columns else []
        parts += [inference.df[column] for inference in self.ctx.state.inferences.values()
                  if column in inference.df.columns]
        if not parts:
            return pd.Series([np.nan] * len(merged))
        return pd.concat([p.reset_index(drop=True) for p in parts], ignore_index=True)

    def _plotted_values(self, result: "_RunResult", layers: list, column: str):
        """`column`'s value for every point on the map, pool first, then
        each overlaid layer -- the order everything else is built in."""
        merged = result.ctx["merged"]
        series = merged[column] if column in merged.columns \
            else pd.Series([np.nan] * len(merged))
        pool = series.iloc[result.idx].reset_index(drop=True)
        if not layers:
            return pool
        return pd.concat([pool] + self._overlay_values(layers, column), ignore_index=True)

    def _kind_of(self, column: str, reference) -> str:
        """How this column is read -- the user's choice where they made
        one, the guess otherwise."""
        forced = self._column_kind.get(column)
        if forced in (pc.KIND_CATEGORICAL, pc.KIND_NUMERIC):
            return forced
        return pc.default_kind(reference)

    def _set_column_kind(self, column: str, kind: str) -> None:
        """Read this column as a scale, or as a set of labels, everywhere
        it is used. Its filter is rebuilt rather than translated: a window
        over a range and a set of ticked values have nothing to say to each
        other, and silently keeping "half" of one in the other would hide
        points with no way to tell why."""
        if self._column_kind.get(column) == kind:
            return
        self._column_kind[column] = kind
        if column in self._filters:
            self._filters[column] = self._build_filter(column)
            self._build_filters_card()
        self._redraw_current("Recoloring...")

    def _on_color_scale_change(self, e):
        column = self._color_column_map.get(self.color_by.value or CLASS_COLOR_KEY)
        if column is None:
            return
        self._set_column_kind(column, next(iter(e.control.selected), pc.KIND_NUMERIC))

    def _sync_color_scale_switch(self, ctx: dict | None = None) -> None:
        """Show the continuous/discrete switch when the colored column
        could be read either way, and set it to what that column is
        currently read as."""
        column = self._color_column_map.get(self.color_by.value or CLASS_COLOR_KEY)
        ctx = ctx or self._last_context
        reference = None if (column is None or ctx is None) else self._column_reference(ctx, column)
        visible = reference is not None and pc.can_be_continuous(reference)
        self.color_scale_switch.visible = bool(visible)
        if visible:
            self.color_scale_switch.selected = {self._kind_of(column, reference)}
        if self.color_scale_switch.page:
            self.color_scale_switch.update()

    # ---- display filters -----------------------------------------------

    def _filterable_columns(self) -> list:
        """What the "Filter points" menu offers: the class, then every
        column a color could come from.

        The class column is offered here although it is not in the colour
        list (where the "Class" entry already stands for it): hiding two
        classes to see what is behind them is one of the things this is
        for, and it is *not* the class picker -- that one re-samples and
        re-projects, this one hides points in place."""
        state = self.ctx.state
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
        self._rebuild_filter_menu()
        self._build_filters_card()
        # Adding a column shows everything it has, so nothing moves on the
        # map until a box or a slider does -- but dropping one that *was*
        # hiding points has to put them back.
        self._redraw_current("Filtering...")

    def _rebuild_filter_menu(self) -> None:
        """One entry per filterable column, ticked when it has a panel.
        The column name travels in the closure rather than as an item id:
        it comes from the user's CSV and must never double as a key."""
        self.filter_menu.items = [
            ft.PopupMenuItem(
                text=str(column), checked=column in self._filter_columns,
                on_click=lambda e, c=column: self._toggle_filter_column(c),
            )
            for column in self._filterable_columns()
        ]
        if self.filter_menu.page:
            self.filter_menu.update()

    def _apply_pending_filters(self) -> None:
        """Turn snapshot-restored filter choices into real ColumnFilters.

        Runs once a context exists, because that is the first moment a
        filter *can* be built: its list of values and its range come from
        the column itself. Anything whose column is no longer there is
        dropped here rather than earlier -- at restore time there was
        nothing to check it against.
        """
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
            # Only the user's choices are restored. What the column offers
            # -- its values, its bounds -- is taken from the data as it is
            # now, so a filter saved against a wider range still makes sense.
            column_filter.hidden = {
                v for v in column_filter.values if str(v) in {str(h) for h in spec.get("hidden", []) or []}
            }
            window = spec.get("window")
            if isinstance(window, (list, tuple)) and len(window) == 2:
                try:
                    low, high = float(window[0]), float(window[1])
                    lo_bound, hi_bound = column_filter.bounds
                    column_filter.window = (max(lo_bound, min(hi_bound, low)),
                                            max(lo_bound, min(hi_bound, high)))
                except (TypeError, ValueError):
                    pass
            column_filter.include_missing = bool(spec.get("include_missing", True))
            self._filters[column] = column_filter
        self._pending_filters.clear()

    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        """The settings that describe a map, never the map itself.

        The colour is stored as the *column name*: the dropdown is keyed
        positionally ("0", "1", ...) precisely so a user's column name can
        never double as a key (see _rebuild_color_options), which means a
        stored key would point at whatever happened to be third next time.
        """
        filters = {}
        for column in self._filter_columns:
            column_filter = self._filters.get(column)
            if column_filter is None:
                continue
            filters[column] = {
                "kind": column_filter.kind,
                "hidden": [str(v) for v in column_filter.hidden],
                "window": [float(column_filter.window[0]), float(column_filter.window[1])],
                "include_missing": bool(column_filter.include_missing),
            }
        return {
            "experiment": self.experiment_dropdown.value or "",
            "method": self.projection_method.value or "pca",
            "sample": str(self.projection_sample_field.value or ""),
            "color_column": self._color_column_map.get(self.color_by.value or CLASS_COLOR_KEY) or "",
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
        """Put the settings back and stop there.

        A projection is minutes of work and t-SNE cannot be interrupted, so
        this never starts one -- the tab shows its usual prompt to run, with
        every control already set to what it was.
        """
        warnings = []

        experiment = state.get("experiment") or ""
        if experiment:
            if experiment in self.ctx.state.experiments:
                self.experiment_dropdown.value = experiment
            else:
                warnings.append(f"Embeddings: experiment '{experiment}' is not loaded.")

        method = state.get("method")
        if method in ea.PROJECTION_METHODS:
            self.projection_method.value = method
        elif method:
            warnings.append(f"Embeddings: unknown projection method '{method}'.")

        if state.get("sample"):
            self.projection_sample_field.value = str(state["sample"])

        kinds = state.get("column_kind")
        if isinstance(kinds, dict):
            self._column_kind = {
                str(c): k for c, k in kinds.items()
                if k in (pc.KIND_CATEGORICAL, pc.KIND_NUMERIC)
            }

        self._class_filter = [c for c in state.get("class_filter", []) or [] if isinstance(c, str)]
        if state.get("proto_filter_chosen"):
            chosen = state.get("proto_filter")
            self._proto_filter = None if chosen is None else [c for c in chosen if isinstance(c, str)]
            self._proto_filter_chosen = True
        self._refit_choice = bool(state.get("refit", False))

        overlays = state.get("overlays")
        if isinstance(overlays, list):
            wanted = {n for n in overlays if isinstance(n, str)}
            gone = sorted(wanted - set(self.ctx.state.inferences))
            if gone:
                warnings.append(f"Embeddings: inference set(s) {', '.join(gone)} are not loaded.")
            self._overlay_selection = wanted & set(self.ctx.state.inferences)

        # Parked rather than built: see _apply_pending_filters.
        saved = state.get("filters")
        self._filter_columns = [c for c in state.get("filter_columns", []) or [] if isinstance(c, str)]
        self._pending_filters = {
            c: saved[c] for c in self._filter_columns
            if isinstance(saved, dict) and isinstance(saved.get(c), dict)
        }

        # The colour column is resolved once the options exist, which
        # happens in _render below.
        self._pending_color_column = str(state.get("color_column") or "")

        self.refresh()
        return warnings

    def _apply_pending_color(self) -> None:
        """Select the colour entry whose column matches the restored name.
        Runs after _rebuild_color_options, which is what fills the map."""
        wanted = getattr(self, "_pending_color_column", "")
        if not wanted:
            return
        self._pending_color_column = ""
        for key, column in self._color_column_map.items():
            if column == wanted:
                self.color_by.value = key
                return

    def _prune_filters(self) -> None:
        """Drop the filters whose column is no longer anywhere -- another
        experiment, an inference unloaded. A filter on a column that does
        not exist would hide nothing and say it was hiding something."""
        available = set(self._filterable_columns())
        gone = [c for c in self._filter_columns if c not in available]
        for column in gone:
            self._filter_columns.remove(column)
            self._filters.pop(column, None)

    def _filter_mask(self, result: "_RunResult", layers: list, n_points: int) -> np.ndarray:
        """(n_points,) bool: what every active filter keeps, in the order
        the points are drawn."""
        active = [self._filters[c] for c in self._filter_columns
                  if c in self._filters and self._filters[c].is_active()]
        if not active:
            return np.ones(n_points, dtype=bool)
        values = {f.column: self._plotted_values(result, layers, f.column) for f in active}
        return pf.combined_mask(active, values, n_points)

    def _on_filter_changed(self) -> None:
        self._build_filters_card()
        self._redraw_current("Filtering...")

    def _build_filters_card(self) -> None:
        """The panel: one block per picked column, nothing at all when none
        is picked. The menu that picks them sits with the other controls
        above, so an empty panel here would be a heading with no content."""
        self.filters_column.controls = [
            self._filter_control(self._filters[column])
            for column in self._filter_columns if column in self._filters
        ]
        if not self._filter_columns:
            self.filters_card.content = None
        else:
            active = [self._filters[c].describe() for c in self._filter_columns
                      if c in self._filters and self._filters[c].is_active()]
            self.filters_card.content = card(ft.Column([
                ft.Row([
                    section_title("Display filters", icon=ft.Icons.TUNE),
                    ft.TextButton("Reset all", icon=ft.Icons.RESTART_ALT,
                                  on_click=lambda e: self._reset_all_filters()),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(
                    "Independent of the coloring: filter on as many columns as you like and color by "
                    "another. Hiding points here changes nothing about the projection -- the points "
                    "that remain are where they always were."
                    if not active else "Showing: " + "; ".join(active),
                    size=11, color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                self.filters_column,
            ], spacing=10))
        for control in (self.filters_column, self.filters_card):
            if control.page:
                control.update()

    def _reset_all_filters(self) -> None:
        for column_filter in self._filters.values():
            column_filter.reset()
        self._on_filter_changed()

    def _filter_control(self, column_filter) -> ft.Control:
        """One column's controls: a window over a scale, or a box per
        value. Which one follows the column's kind, and the switch that
        sets that kind sits right here -- a column read the wrong way is
        noticed while filtering on it at least as often as while coloring
        by it."""
        column = column_filter.column
        header = [ft.Text(column, size=12, weight=ft.FontWeight.BOLD)]
        ctx = self._last_context
        if ctx is not None and pc.can_be_continuous(self._column_reference(ctx, column)):
            header.append(ft.SegmentedButton(
                show_selected_icon=False, allow_empty_selection=False,
                selected={column_filter.kind},
                segments=[
                    ft.Segment(value=pc.KIND_NUMERIC, label=ft.Text("Continuous", size=11)),
                    ft.Segment(value=pc.KIND_CATEGORICAL, label=ft.Text("Discrete", size=11)),
                ],
                on_change=lambda e, c=column: self._set_column_kind(
                    c, next(iter(e.control.selected), pc.KIND_NUMERIC)),
            ))
        header.append(ft.IconButton(
            ft.Icons.CLOSE, icon_size=16, tooltip="Stop filtering on this column",
            on_click=lambda e, c=column: self._toggle_filter_column(c)))

        if column_filter.kind == pc.KIND_NUMERIC:
            body = self._numeric_filter_body(column_filter)
        else:
            body = self._categorical_filter_body(column_filter)
        return ft.Container(
            padding=10, border_radius=8, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            content=ft.Column(
                [ft.Row(header, spacing=8, wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER)] + body,
                spacing=6, tight=True),
        )

    def _numeric_filter_body(self, column_filter) -> list:
        low, high = column_filter.window
        lo, hi = column_filter.bounds
        slider = ft.RangeSlider(
            min=lo, max=hi, start_value=low, end_value=high,
            label="{value}", round=4,
            # on_change_end, not on_change: a drag fires continuously, and
            # every change here redraws both renderings of the map.
            on_change_end=lambda e, f=column_filter: self._on_range_changed(f, e.control),
        )
        readout = ft.Text(f"{low:.4g} to {high:.4g}", size=11,
                          color=ft.Colors.ON_SURFACE_VARIANT)
        body = [slider, readout]
        if column_filter.has_missing:
            body.append(ft.Checkbox(
                label="include rows with no value", value=column_filter.include_missing,
                scale=0.85,
                on_change=lambda e, f=column_filter: self._on_missing_changed(
                    f, bool(e.control.value)),
            ))
        return body

    def _on_range_changed(self, column_filter, slider) -> None:
        column_filter.window = (float(slider.start_value), float(slider.end_value))
        self._on_filter_changed()

    def _on_missing_changed(self, column_filter, include: bool) -> None:
        column_filter.include_missing = include
        self._on_filter_changed()

    def _categorical_filter_body(self, column_filter) -> list:
        boxes = []
        entries = list(column_filter.values)
        for value in entries:
            boxes.append(ft.Checkbox(
                label=str(value), value=value not in column_filter.hidden, scale=0.85,
                on_change=lambda e, f=column_filter, v=value: self._on_value_toggled(
                    f, v, bool(e.control.value)),
            ))
        if column_filter.has_missing:
            boxes.append(ft.Checkbox(
                label=pc.MISSING_GROUP, value=column_filter.include_missing, scale=0.85,
                on_change=lambda e, f=column_filter: self._on_missing_changed(
                    f, bool(e.control.value)),
            ))
        body = [
            ft.Row([
                ft.TextButton("All", on_click=lambda e, f=column_filter: self._set_all_values(f, True)),
                ft.TextButton("None", on_click=lambda e, f=column_filter: self._set_all_values(f, False)),
            ], spacing=4),
            ft.Column(boxes, spacing=0, tight=True),
        ]
        if column_filter.truncated:
            body.append(ft.Text(
                f"{column_filter.truncated} rarer value(s) are not listed and are never hidden by "
                f"this filter.", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))
        return body

    def _on_value_toggled(self, column_filter, value, shown: bool) -> None:
        if shown:
            column_filter.hidden.discard(value)
        else:
            column_filter.hidden.add(value)
        self._on_filter_changed()

    def _set_all_values(self, column_filter, shown: bool) -> None:
        column_filter.hidden = set() if shown else set(column_filter.values)
        column_filter.include_missing = shown
        self._on_filter_changed()

    # ---- coloring ----------------------------------------------------

    def _rebuild_color_options(self, ctx: dict) -> None:
        """One dropdown entry per column of the joined frame that can carry
        a color, with "Class" first.

        Keyed positionally ("0", "1", ...) with the real column name held in
        _color_column_map, for the same reason the case-study dropdown in
        ui/prototypes_view.py is: a column name comes from the user's CSV,
        and a column called "0" or "class" must not be able to collide with
        a key that code compares against.
        """
        state = self.ctx.state
        cols = state.dataset.columns
        exp = ctx["exp"]
        merged = ctx["merged"]
        inferences = state.inferences
        # Deciding which columns qualify counts distinct values across the
        # whole joined frame, which is not free on a large one -- and
        # _render runs on every refresh of the tab.
        cache_key = (exp.name, len(merged), tuple(merged.columns),
                     tuple((name, len(i.df), tuple(i.df.columns)) for name, i in inferences.items()))
        if cache_key != self._color_options_key:
            self._colorable = pc.colorable_columns(
                merged,
                exclude={cols.uuid_col, cols.crop_col, cols.label_col, exp.uuid_col},
                score_cols=exp.score_cols,
            )
            # A simple inference brings its own columns, and they are
            # offered on the same footing: whatever its CSV carried beyond
            # the id and the path -- a confidence, a sensor, a capture date
            # -- is a coloring and nothing else (it cannot reach a metric;
            # see InferenceSet).
            #
            # They go in *before* the per-class score columns, for the same
            # reason those are last to begin with: a 200-class experiment
            # contributes 200 of them, and a column the user loaded on
            # purpose must not be buried under that.
            extras = []
            for inference in inferences.values():
                offered = pc.colorable_columns(
                    inference.df, exclude={inference.uuid_col, inference.crop_col})
                extras += [c for c in offered if c not in self._colorable and c not in extras]
            scores = [c for c in exp.score_cols if c in self._colorable]
            self._colorable = [c for c in self._colorable if c not in scores] + extras + scores
            self._color_options_key = cache_key

        previous_key = self.color_by.value or CLASS_COLOR_KEY
        previous = self._color_column_map.get(previous_key)
        self._color_column_map = {CLASS_COLOR_KEY: None}
        options = [ft.dropdown.Option(CLASS_COLOR_KEY, "Class (true label)")]
        if inferences:
            # Only worth offering once there is more than one place a point
            # can come from: with no inference loaded it would be a legend
            # with a single entry.
            self._color_column_map[SOURCE_COLOR_KEY] = None
            options.append(ft.dropdown.Option(SOURCE_COLOR_KEY, "Source (dataset / inference)"))
        offered = self._colorable
        score_note = None
        if len(exp.score_cols) > FOCUS_MAX:
            # One score column per class is two thousand entries at that
            # many classes -- more than the menu can even open. Only the
            # classes the map is restricted to keep theirs.
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
            options.append(ft.dropdown.Option(str(i), column))
        if score_note:
            # Not an entry: a line in the menu saying where the rest went.
            options.append(ft.dropdown.Option(SCORE_NOTE_KEY, score_note, disabled=True))
        self.color_by.options = options
        # Past FOCUS_MAX entries (one per class score column, at thousands
        # of classes) the menu filters as you type.
        make_searchable(self.color_by)
        # The pick follows the column, not its position: another experiment
        # can offer a different set of columns, and keeping the index would
        # quietly start coloring by a different one. The two entries that
        # are not columns follow their own key instead, since they have no
        # column to be found by.
        if previous_key in (CLASS_COLOR_KEY, SOURCE_COLOR_KEY) and previous_key in self._color_column_map:
            self.color_by.value = previous_key
        else:
            self.color_by.value = next(
                (key for key, column in self._color_column_map.items()
                 if column is not None and column == previous),
                CLASS_COLOR_KEY,
            )
        if self.color_by.page:
            self.color_by.update()

    def _overlay_values(self, layers: list, column: str | None) -> list:
        """One Series per overlaid layer: that layer's values for `column`,
        or nothing at all where it has no such column.

        A missing column is not an error and not a reason to leave the
        points out -- they are painted `(missing)` like any other row with
        no value, which is the honest answer to "color these by whether the
        model got them right" for crops nobody has scored."""
        out = []
        for layer in layers:
            inference = self.ctx.state.inferences.get(layer.name)
            if column is not None and inference is not None and column in inference.df.columns:
                out.append(inference.df[column].iloc[layer.rows].reset_index(drop=True))
            else:
                out.append(pd.Series([np.nan] * len(layer.coords)))
        return out

    def _color_mapping(self, result: "_RunResult", layers: list | None = None,
                       mask: np.ndarray | None = None) -> tuple:
        """The coloring for one run's plotted points -- the pool's, then
        every overlaid layer's, in that order -- and whether it is the class
        coloring (the one case where a prototype has a color of its own to
        take from the cloud).

        Coloring by class is not a special path with its own code -- it is
        the label column through the same builder as any other column,
        which is what keeps the two renderings identical whichever column
        the color comes from, past pc.MAX_CATEGORIES classes included.

        Overlaid points go through the same builder as everything else, so
        a value they share with the dataset comes out the same color. Where
        they cannot answer at all -- no label, or a column their CSV does
        not have -- they land in `(missing)`, and the shape of the marker
        is what still tells them apart.
        """
        ctx, idx = result.ctx, result.idx
        layers = layers or []
        n_overlay = sum(len(layer.coords) for layer in layers)
        key = self.color_by.value or CLASS_COLOR_KEY

        def with_overlay(pool_values, overlay_values):
            values = pd.Series(pool_values).reset_index(drop=True)
            if layers:
                values = pd.concat([values] + list(overlay_values), ignore_index=True)
            # The filtered-out points are dropped *before* the mapping is
            # built, not painted and then hidden: the legend only ever
            # lists groups that have a point on screen, and a checkbox for
            # a value the filters have already removed would filter
            # nothing.
            if mask is None:
                return values
            return values[mask].reset_index(drop=True)

        if key == SOURCE_COLOR_KEY:
            # Where each point comes from, as a coloring in its own right:
            # the one question about an overlay that no column can answer,
            # since "which file was this row in" is not a column anywhere.
            names = [DATASET_SOURCE_GROUP] * len(idx)
            for layer in layers:
                names += [layer.name] * len(layer.coords)
            shown = pd.Series(names, dtype=object)
            if mask is not None:
                shown = shown[mask].reset_index(drop=True)
            return pc.build_mapping(shown, SOURCE_COLORING_LABEL,
                                    kind=pc.KIND_CATEGORICAL), False

        column = self._color_column_map.get(key)
        if column is not None:
            # The whole column is the reference, so a value keeps its color
            # across sample sizes, class filters, methods and filters --
            # and every overlaid set's own values are in it, so a value
            # only they have still gets a color of its own. The kind is the
            # column's own (the user may have said to read it as a scale
            # rather than as labels), never guessed twice in two places.
            reference = self._column_reference(ctx, column)
            values = self._plotted_values(result, layers, column)
            if mask is not None:
                values = values[mask].reset_index(drop=True)
            return pc.build_mapping(values, column, reference=reference,
                                    kind=self._kind_of(column, reference)), False

        labels = pd.Series(ctx["labels"])
        # An overlaid crop has no true label -- that is what makes it an
        # inference rather than a dataset row.
        no_labels = [pd.Series([None] * n_overlay, dtype=object)] if layers else []
        # Always by class. Past pc.MAX_CATEGORIES classes the builder keeps
        # the most common ones in its colors and puts the rest in (other),
        # and says so in the caption -- rather than silently switching to a
        # ramp of class sizes, which answered a question nobody had asked
        # under a menu entry that still read "Class". "Most common" is
        # counted among the classes the map is restricted to, when it is:
        # counted over the whole pool, a restriction to rare classes would
        # have drawn every one of them in (other) grey.
        reference = labels if not result.class_filter else labels[labels.isin(result.class_filter)]
        # kind is forced rather than inferred: a dataset whose labels are
        # integers would otherwise be read as a scale and shaded on a ramp,
        # which is exactly what a class is not.
        mapping = pc.build_mapping(with_overlay(labels.iloc[idx], no_labels), CLASS_COLORING_LABEL,
                                   reference=reference, kind=pc.KIND_CATEGORICAL)
        # The prototypes may still take their class colors: the overlay
        # changed which points are drawn, not what a class is colored.
        return mapping, True

    def _on_color_change(self, e):
        """Recolor what is already on screen.

        No reprojection: the coordinates do not depend on the color, and
        t-SNE/UMAP cost tens of seconds -- so unlike the class picker,
        which changes what is *in* the projection, this redraws the run
        already in hand. While a run is computing there is nothing to
        redraw and nothing to do: it reads the dropdown when it renders, so
        it will come out in the new colors by itself.

        The map is rebuilt rather than repainted, which does cost the
        current zoom and pan. Repainting in place was tried and reverted:
        the shapes and the legend come out right in Python, but Flet drops
        updates made to controls that are already mounted -- the canvas
        keeps its old colors while every value behind it has changed, which
        is a far worse failure than a reset viewport. What Flet does send
        reliably is a replaced subtree, so that is what this does.
        """
        if (self.color_by.value or CLASS_COLOR_KEY) not in self._color_column_map:
            return  # text typed into the filter, not an entry picked
        self._sync_color_scale_switch()
        self._redraw_current("Recoloring...")

    def _redraw_current(self, message: str) -> None:
        """Re-render the projection already in hand, off the UI thread.

        Everything that changes the *picture* and not the coordinates goes
        through here -- the coloring, and ticking an overlay on or off when
        it is not part of the fit. While a run is computing there is
        nothing to redraw and nothing to do: the render reads the controls
        when it happens, so it comes out right by itself."""
        if self._computing or self._last_result is None:
            return
        self._projection_generation += 1
        generation = self._projection_generation
        self.projection_results_container.content = ft.Row(
            [ft.ProgressRing(width=16, height=16, stroke_width=2),
             ft.Text(message, size=12)],
            spacing=10,
        )
        if self.projection_results_container.page:
            self.projection_results_container.update()
        self.ctx.page.run_thread(self._render_projection, self._last_result, generation)

    # ---- rendering ---------------------------------------------------

    def _render_projection(self, result: "_RunResult", generation: int):
        """Both renderings of one already-projected cloud. Everything here
        is redoable without projecting again, which is what lets the color
        be changed for the price of a redraw."""
        try:
            # The pool's points first, then every overlaid inference's, and
            # that order holds for the whole method: colors, groups, uuids,
            # records and marker shapes are all built as one array with the
            # overlay on the end, so a point's index means the same thing
            # in each of them.
            layers = self._overlay_layers(result)
            n_drawn = len(result.proj)
            proj = result.proj if not layers else np.vstack(
                [result.proj] + [layer.coords for layer in layers])
            is_overlay = np.zeros(len(proj), dtype=bool)
            is_overlay[n_drawn:] = True

            # The display filters, applied once, here: everything below --
            # the colors, the legend, the markers, the rows behind the
            # points -- is built from what survives, so no part of the
            # render has to remember that some points are hidden. Dropping
            # them rather than drawing them invisibly is also what keeps a
            # lasso and a click honest, since both work on positions.
            total = len(proj)
            shown = self._filter_mask(result, layers, total)
            n_hidden = int((~shown).sum())
            if n_hidden:
                proj = proj[shown]
                is_overlay = is_overlay[shown]
                n_drawn = int(shown[:n_drawn].sum())
            proto_proj, proto_labels, proto_ks = result.proto_proj, result.proto_labels, result.proto_ks
            n_subcenters = result.n_subcenters
            mapping, colors_by_class = self._color_mapping(
                result, layers, shown if n_hidden else None)
            _log(f"rendering '{result.exp_name}' colored by {mapping.column} "
                 f"({mapping.kind}, {len(mapping.legend)} legend entries)"
                 + (f", with {len(proj) - n_drawn:,} overlaid points from "
                    f"{len(layers)} inference(s)" if layers else "")
                 + (f", {n_hidden:,} points hidden by {len(self._filter_columns)} filter(s)"
                    if n_hidden else ""))

            # Sized so the figure's own pixels land roughly 1:1 in the box
            # it is displayed in: rendering it small and stretching the
            # bitmap up to the map's size is what makes an inline chart
            # look soft, and the two renderings sit one click apart.
            map_size = self._map_size()
            fig, ax = subplots(figsize=(map_size / INLINE_DPI, map_size / INLINE_DPI * 0.97))
            color_legend = None
            if mapping.kind == pc.KIND_CATEGORICAL:
                for color, name in mapping.legend:
                    mask = mapping.groups == name
                    pool_mask = mask & ~is_overlay
                    ax.scatter(proj[pool_mask, 0], proj[pool_mask, 1], s=6, alpha=0.7,
                               color=color, label=short_name(name))
                    # The same color, a different marker: an overlaid crop
                    # is not a dataset row and has to be recognizable as
                    # one whatever the coloring says.
                    over_mask = mask & is_overlay
                    if over_mask.any():
                        ax.scatter(proj[over_mask, 0], proj[over_mask, 1], marker="s", s=24,
                                   color=color, edgecolors="black", linewidths=0.4, zorder=4)
                # ~16 entries per column: past that the legend grows taller
                # than the plot itself, so it wraps into a second column
                # instead (32 groups -> 2 x 16, the readable upper bound
                # MAX_CATEGORIES caps it at).
                legend_ncol = max(1, -(-len(mapping.legend) // 16))
                color_legend = ax.legend(
                    loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=legend_ncol,
                    frameon=False, markerscale=1.8, title=mapping.column, title_fontsize=8,
                )
            else:
                # The figure gets the raw values and a continuous ramp; the
                # canvas gets the same ramp in discrete bands (see
                # core/point_coloring). Rows with no value are drawn in the
                # same grey the canvas gives them rather than dropped --
                # "the column is empty here" is a finding, not a gap.
                known = np.isfinite(mapping.values)
                missing = ~known
                if missing.any():
                    ax.scatter(proj[missing & ~is_overlay, 0], proj[missing & ~is_overlay, 1],
                               s=6, alpha=0.6, color=pc.MISSING_COLOR, label=pc.MISSING_GROUP)
                    if (missing & is_overlay).any():
                        ax.scatter(proj[missing & is_overlay, 0], proj[missing & is_overlay, 1],
                                   marker="s", s=24, color=pc.MISSING_COLOR,
                                   edgecolors="black", linewidths=0.4, zorder=4)
                    color_legend = ax.legend(loc="upper right", fontsize=7, frameon=False,
                                             markerscale=1.8)
                if known.any():
                    pool_known = known & ~is_overlay
                    sc = ax.scatter(proj[pool_known, 0], proj[pool_known, 1], s=6, alpha=0.6,
                                    c=mapping.values[pool_known], cmap="viridis",
                                    vmin=mapping.vmin, vmax=mapping.vmax)
                    over_known = known & is_overlay
                    if over_known.any():
                        ax.scatter(proj[over_known, 0], proj[over_known, 1], marker="s", s=24,
                                   c=mapping.values[over_known], cmap="viridis",
                                   vmin=mapping.vmin, vmax=mapping.vmax,
                                   edgecolors="black", linewidths=0.4, zorder=4)
                    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(mapping.column, fontsize=8)

            if proto_proj is not None and len(proto_proj):
                # One scatter call per subcenter index, since a marker shape
                # can't vary within a single call the way a color can.
                # Colored by class when the points are (so a prototype is
                # matchable to its own cloud); when the color means anything
                # else there is no class color to borrow, so it falls back
                # to the flat per-source color (blue = model, purple =
                # kmeans).
                ks = np.array(proto_ks)
                for k_idx in range(n_subcenters):
                    sel = ks == k_idx
                    if not sel.any():
                        continue
                    marker = SUBCENTER_MARKERS[k_idx % len(SUBCENTER_MARKERS)]
                    if colors_by_class:
                        fill = [mapping.color_of(proto_labels[i]) for i in np.flatnonzero(sel)]
                    else:
                        fill = result.source_color
                    ax.scatter(
                        proto_proj[sel, 0], proto_proj[sel, 1], marker=marker,
                        s=210 if marker == "*" else 130, c=fill,
                        edgecolors="black", linewidths=0.9, zorder=5,
                    )
                if n_subcenters > 1:
                    # Sits under the plot rather than beside it, so it never
                    # collides with the (tall, right-hand) color legend.
                    # "sub 0" rather than "subcenter 0", and no legend title:
                    # the caption under the figure already says the shape is
                    # the subcenter, so spelling it out again here only cost
                    # width and an extra line.
                    marker_handles = [
                        Line2D(
                            [], [], linestyle="none", label=f"sub {k}",
                            marker=SUBCENTER_MARKERS[k % len(SUBCENTER_MARKERS)],
                            color="#bbbbbb", markeredgecolor="black", markersize=10,
                        )
                        for k in range(n_subcenters)
                    ]
                    if color_legend is not None:
                        # A second ax.legend() replaces the first, so the color
                        # legend has to be re-attached -- to the *figure*, not
                        # the axes: an axes-level add_artist() legend is left
                        # out of get_default_bbox_extra_artists(), so
                        # fig_to_base64's bbox_inches="tight" would crop it
                        # straight back off the saved PNG. It keeps the
                        # axes-relative anchor it was created with either way.
                        fig.add_artist(color_legend)
                    ax.legend(
                        handles=marker_handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                        ncol=min(n_subcenters, 8), fontsize=7, frameon=False,
                    )
            if result.method == "umap_circle":
                theta = np.linspace(0, 2 * np.pi, 200)
                ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=0.8, linestyle="--")
                ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            has_outside_legend = mapping.kind == pc.KIND_CATEGORICAL or n_subcenters > 1
            if not has_outside_legend:
                # Deliberately skipped whenever a legend sits outside the axes:
                # tight_layout() would shrink the axes to squeeze it back inside
                # the canvas -- halving the scatter at 32 groups.
                # fig_to_base64's bbox_inches="tight" already grows the saved
                # PNG to include it, keeping the plot at full size.
                fig.tight_layout()

            if generation != self._projection_generation:
                _log(f"projection ({result.method}) for '{result.exp_name}' discarded -- "
                     "superseded by a newer request")
                plt.close(fig)
                return

            if colors_by_class:
                notes = [f"Color = {mapping.column} (legend alongside), for the points as well as "
                         "the prototypes."]
            elif mapping.kind == pc.KIND_CATEGORICAL:
                notes = [f"Color = {mapping.column} (legend alongside)."]
            else:
                notes = [f"Point color = {mapping.column}."]
            if mapping.note:
                notes.append(mapping.note)
            if layers:
                notes.append(
                    f"Squares = crops from {len(layers)} simple inference(s) "
                    f"({len(proj) - n_drawn:,} points), not part of the dataset.")
            if n_subcenters:
                source_label = result.source_note
                shape_note = (
                    "one shape per subcenter (legend below the plot)" if n_subcenters > 1
                    else "large stars"
                )
                proto_color = "" if colors_by_class else (
                    " in blue" if result.source_key == PROTO_FROM_MODEL else " in purple")
                notes.append(f"Prototypes = large black-edged symbols{proto_color}, {shape_note} — {source_label}.")
            static_view = ft.Column([
                ft.Text(" ".join(notes), size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                # The legends are drawn outside the axes and bbox_inches="tight"
                # keeps them in the saved PNG, so they add real width/height --
                # grow the displayed box to match instead of letting a fixed
                # size squeeze the scatter itself down.
                fig_to_zoomable_image(
                    fig, self.ctx, f"2D projection ({result.method}) — {result.exp_name}",
                    width=map_size, height=map_size,
                ),
            ], spacing=10)

            # The interactive view is fed exactly the same projected
            # coordinates, the same colors and the same prototypes as the
            # figure above -- it is the same plot drawn by a different
            # renderer, not a second analysis.
            ctx = result.ctx
            interactive_prototypes = None
            proto_colors = None
            if proto_proj is not None and len(proto_proj):
                # Same rule as the figure: a prototype takes its class
                # color where there is one to take, and the flat per-source
                # color otherwise.
                proto_colors = (
                    [mapping.color_of(l) for l in proto_labels] if colors_by_class
                    else [result.source_color] * len(proto_labels)
                )
                # The class labels travel with the coordinates: the map
                # needs them to name a prototype on hover, and a marker in
                # the middle of a cluster is unreadable without that.
                interactive_prototypes = (proto_proj, proto_colors, proto_ks, proto_labels)
            # Rows and uuids for every point on the map, pool then
            # overlay, in the order the coordinates are in. Rows are passed
            # positionally rather than looked up by uuid: with two pools on
            # one map a uuid is no longer a unique key, and the crop shown
            # on hover has to be the crop that was clicked.
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

            interactive = InteractiveProjection(
                self.ctx, proj, mapping.colors, mapping.legend, uuids_all,
                ctx["row_by_uuid"], ctx["crop_col"], ctx["uuid_col"],
                title=f"2D projection ({result.method}) — {result.exp_name}",
                prototypes=interactive_prototypes,
                on_select=lambda indices, r=records_all, n=n_drawn: self._open_crops(indices, r, n),
                on_point=lambda i, r=records_all, n=n_drawn, u=uuids_all: self._open_crops(
                    [i], r, n, title=f"Crop {u[i]}"),
                label_col=self.ctx.state.dataset.columns.label_col,
                point_groups=mapping.groups,
                legend_title=f"Show {mapping.column}",
                size=map_size,
                records=records_all, point_kinds=point_kinds if layers else None,
            )
            # The legend belongs to the left column, not to the map: only
            # the map may live outside a scrolling region, and only the map
            # is worth the width.
            self._interactive = interactive
            self.legend_slot.content = interactive.legend_control()

            # The plain "N points from M classes" line above the map is
            # gone -- it cost a row of the picture to say what the left
            # column already says. The warning that went with a *restricted*
            # projection is not the same thing and is kept, beside the class
            # chips that caused it: a projection fitted to two classes looks
            # exactly like one fitted to twenty, and nothing else on screen
            # would tell the difference.
            notes_above = []
            if result.class_filter:
                notes_above.append(info_banner(
                    f"Fitted to {len(result.idx):,} points from {len(result.class_filter)} of "
                    f"{len(result.all_classes)} classes ({result.pool_size:,} available) — positions "
                    "are not comparable with another selection's.",
                    icon=ft.Icons.INFO_OUTLINE,
                ))
            if n_hidden:
                kept = total - n_hidden
                notes_above.append(info_banner(
                    f"{kept:,} of {total:,} points shown — {n_hidden:,} hidden by the display "
                    f"filters. The projection is unchanged: what is left is where it always was.",
                    icon=ft.Icons.TUNE,
                ))
            # The *fitted* count, not the shown one: hiding points does not
            # unfit them, and a caption that said otherwise would describe
            # a projection nobody computed.
            fit_note = self._fit_note(result, layers, len(result.idx))
            if fit_note:
                notes_above.append(info_banner(fit_note, icon=ft.Icons.INFO_OUTLINE))
            self.projection_note.content = None if not notes_above else ft.Column(
                notes_above, spacing=6)
            self.projection_results_container.content = self._switcher(
                interactive.map_control(), static_view, interactive.toolbar_controls())
            self._has_result = True
        except Exception as ex:
            _log(f"error rendering projection: {ex}")
            traceback.print_exc()
            if generation != self._projection_generation:
                return
            self.projection_results_container.content = info_banner(f"Error: {ex}", icon=ft.Icons.ERROR_OUTLINE)
        # The map and the legend are two separate controls in two different
        # panes, so each needs its own push -- updating the results
        # container alone would leave the legend missing from the column.
        for control in (self.projection_results_container, self.legend_slot, self.projection_note):
            if control.page:
                control.update()

    # ---- lifecycle ---------------------------------------------------

    def _render(self, ctx: dict | None = None):
        """Redraw everything around the map: the class and color pickers,
        the row counts, the coverage note. `ctx` is the caller's already
        built context for the selected experiment, passed in only to avoid
        joining the frame a second time (see _on_experiment_change)."""
        state = self.ctx.state
        self.status_area.controls.clear()
        if state.dataset is None or not experiments_with_embeddings(state):
            self.controls_row.visible = False
            self.status_area.controls.append(info_banner(
                "Attach an embeddings .npz sidecar to at least one loaded experiment in the Data tab "
                "(uuid + embeddings arrays) to unlock this tab."
            ))
            self.projection_results_container.visible = False
            return
        self.controls_row.visible = True
        self.projection_results_container.visible = True

        if not self.experiment_dropdown.value:
            self._refresh_experiment_dropdown()
        exp_name = self.experiment_dropdown.value
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
        self._sync_color_scale_switch(ctx)
        self._apply_pending_filters()
        self._prune_filters()
        self._rebuild_filter_menu()
        self._build_filters_card()
        self._rebuild_overlay_controls()
        self._build_overlay_card()
        self.status_area.controls.append(ft.Text(
            f"{len(ctx['embeddings']):,} embeddings loaded for '{exp_name}'.",
            size=11, color=ft.Colors.ON_SURFACE_VARIANT,
        ))
        note = coverage_note(ctx, state, exp_name)
        if note:
            self.status_area.controls.append(info_banner(note, icon=ft.Icons.WARNING_AMBER_OUTLINED))

    def _build_overlay_card(self) -> None:
        """The overlay controls, or nothing at all.

        Nothing, when no simple inference is loaded: an empty box titled
        "Simple inferences" is a feature advertising itself in the middle
        of a tab that has work to do. The Data tab is where they are
        loaded, and it says so there."""
        if not self.ctx.state.inferences:
            self.overlay_card.content = None
            if self.overlay_card.page:
                self.overlay_card.update()
            return
        self.overlay_card.content = card(ft.Column([
            section_title("Overlay simple inferences", icon=ft.Icons.LAYERS_OUTLINED),
            ft.Text(
                "Drawn as squares, with no label and no effect on any metric. Ticking one costs a "
                "redraw, not another projection, unless the map is refitted.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self.overlay_column,
            self.refit_check,
            self.overlay_note,
        ], spacing=10))
        if self.overlay_card.page:
            self.overlay_card.update()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow an experiment rename (see AppContext.notify_rename). The
        only per-experiment state here is which one is selected -- the
        projection on screen is still valid, it just belongs to a name that
        changed, so it is left alone rather than thrown away."""
        selected = self.experiment_dropdown.value
        # Its cached run follows the name, before the dropdown refresh --
        # which drops the cache of any experiment it no longer sees, and
        # under the old name that is exactly what this one now looks like.
        cached = self._runs.pop(old_name, None)
        if cached is not None:
            self._runs[new_name] = replace(cached, exp_name=new_name)
        if old_name in self._proto_source_choice:
            self._proto_source_choice[new_name] = self._proto_source_choice.pop(old_name)
        self._refresh_experiment_dropdown()
        if selected == old_name:
            self.experiment_dropdown.value = new_name
        self._render()
        if self.root.page:
            self.root.update()

    def recompute_after_restore(self) -> None:
        """Run the projection the snapshot described.

        Deliberately not free -- t-SNE on a large sample is minutes of work
        -- and it runs anyway, because a snapshot of this tab that comes
        back as an empty map under a "Run projection" button is not the view
        that was saved. The point filters it carries also have nothing to
        act on until a map exists.
        """
        state = self.ctx.state
        if state.dataset is None or not experiments_with_embeddings(state):
            return
        if not self.experiment_dropdown.value:
            return
        self._on_run_projection(None)

    def refresh(self):
        self._refresh_experiment_dropdown()
        self._render()
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        """Two panes, and the split is load-bearing rather than cosmetic.

        A wheel event in Flutter is consumed by any enclosing scrollable as
        well as by us, and Flet gives no way to claim it -- so a map inside
        a scrolling page zooms *and* scrolls at once. Here the map sits in
        a fixed right-hand pane with no scrolling ancestor, and everything
        that legitimately wants to scroll is in the left column beside it.

        It also happens to be the better use of the screen: the map is
        square and the scarce dimension on a 16:9 display is the height, so
        a square fitted to the height leaves half the width free -- which
        is precisely what the settings column needs.

        Full screen hides that column (see _set_fullscreen) without moving
        the map out of this pane, so the wheel keeps the scroll-free
        ancestry the split was built for.
        """
        self.projection_results_container.content = info_banner("Click 'Run projection' to compute.")

        self.left_pane = ft.Column(
            controls=[
                section_title("2D projection of the embeddings", icon=ft.Icons.BUBBLE_CHART, ctx=self.ctx, manual=self._projection_manual_page),
                ft.Text(
                    "Hover a point for its crop, click it to open that crop in the carousel, lasso "
                    "a group to browse them all there with the review controls. 'Color by' and "
                    "'Filter points' both redraw the same map without refitting it -- colour says "
                    "what a point is, the filters say which points are drawn at all. The switch "
                    "beside 'Color by' reads a numeric column either as a scale or as labels.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                card(ft.Column([self.controls_row, self.class_picker_box, self.class_chips, self.proto_box,
                                self.projection_note,
                                self.status_area], spacing=10)),
                self.filters_card,
                self.overlay_card,
                self.legend_slot,
                section_title("How much the classes overlap", icon=ft.Icons.JOIN_INNER_OUTLINED, ctx=self.ctx, manual="class-overlap"),
                ft.Text(
                    "Mean cosine similarity within and between classes, measured on the crops "
                    "themselves. Needs only the embeddings, so it works on any run with a sidecar.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                card(ft.Column([
                    ft.Row([self.overlap_btn], spacing=12),
                    self.overlap_progress,
                    self.overlap_results,
                ], spacing=10)),
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        self.map_pane = ft.Container(
            width=self._map_size(),
            content=self.projection_results_container,
        )
        self.panes_row = ft.Row(
            [self.left_pane, self.map_pane],
            spacing=24, vertical_alignment=ft.CrossAxisAlignment.START, expand=True,
            # Flet's own default, spelled out because _set_fullscreen swaps
            # this to CENTER and back -- leaving it implicit would make
            # "back" a different value from where it started.
            alignment=ft.MainAxisAlignment.START,
        )
        self.root.content = self.panes_row
        return self.root
