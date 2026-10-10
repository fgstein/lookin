"""Interactive companion to the Embeddings tab's 2D projection: the same
point cloud, drawn natively on a Flet canvas instead of rendered by
matplotlib, so that every point stays addressable -- hovering one shows a
small crop at the pointer, clicking it opens that crop in the carousel,
lassoing a group opens all of them there too, and the wheel zooms into a
cluster to pull its points apart.

Why not put a hit-test layer over the matplotlib PNG instead: the figure's
layout is matplotlib's to decide (dpi, then bbox_inches="tight" growing the
canvas to fit legends drawn outside the axes, then Flet scaling the
resulting bitmap into its box), so mapping a cursor position back to a data
point means reproducing three transforms this code does not own -- and
being a few pixels off is invisible right up until a user clicks and gets
the wrong image. Here the data->screen transform is _to_screen(): it is
ours, and the *same* arrays it produces both place the dots and answer
"which point is under the cursor", so the two cannot drift apart. The lasso
and the box zoom run on those same screen coordinates.

The matplotlib rendering is not replaced -- it stays one click away (see
projection_view) as the reference, and it is the one that exports to PNG.

Hover and click answer different questions -- "what is this?" against
"show me this one" -- but both show the *original* file, and that is not
negotiable: this is a tool for evaluating image algorithms, so a preview
that had been resampled or re-encoded would show artefacts that are not in
the data, and the small ones are exactly where a person would not think to
doubt what they see. The card at the pointer is therefore the original
bytes displayed in a small box, not a thumbnail built from them -- seen
through the session's display setting (LUT and roll correction) like every
other crop in the app, which is a display choice the user made, and for
the roll a lossless quarter turn rather than a resampling. The cost
is one full image pushed per point the pointer lands on; if that ever
becomes too slow, the lever is to *debounce* the hover -- load only once
the pointer rests on a point -- never to degrade the picture.

The click target is the carousel: it is where the review controls live, so
a point that looks wrong can be relabelled in the same gesture that opened
it. A side panel beside the map was tried instead and taken back out -- at
300px of image plus a column of metadata it did not fit the screens this
runs on, and it bought nothing the carousel does not already do.

Scale: points are drawn as canvas `Points` shapes, one per *color* rather
than one per point -- a single shape carries a whole class's coordinates,
so 10k points cost a few dozen shapes instead of 10k controls, and only the
ones currently inside the box are sent at all. Picking is a brute-force
nearest-neighbor over the same arrays, which for 10k points is tens of
microseconds in numpy -- orders of magnitude below the websocket round-trip
that delivered the cursor position, so there is no spatial index worth
keeping in sync.

What does not scale for free is the *frame*: a redraw serializes every
visible coordinate to JSON and pushes it, which at 50k points is ~130ms and
three quarters of a megabyte. A drag asks for one of those every 40ms, so
the frames have to be rationed rather than merely made faster -- see
_RedrawPump, which owns the one thread that draws, and the thinned "draft"
frames it serves while a gesture is still moving.

Nothing here is scrollable, and that is load-bearing: a wheel event in
Flutter is a pointer signal that any enclosing scrollable also consumes,
and Flet exposes no way to claim it. The map therefore has to sit outside
the tab's scrolling region entirely (see projection_view), or the wheel
would zoom *and* scroll the page at the same time.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import flet as ft
import numpy as np
from flet import canvas as cv
from matplotlib.path import Path as MplPath

from lookin.ui.components import crop_display, file_to_full_base64

# The map is square and sized by the *height* of the window: on any 16:9
# screen height is the scarce dimension and width is abundant, so a square
# fitted to the height wastes nothing. These bound what the layout asks for.
DEFAULT_PLOT_SIZE = 820
MIN_PLOT_SIZE = 420
# Keeps the outermost points off the border, and leaves room for the hover
# ring (which is drawn wider than a point) to stay fully inside the plot.
# It scales with the map because it reads as a margin, not as a number of
# pixels: a flat 22px was comfortable at MIN_PLOT_SIZE and left the cloud
# running into the edge at the sizes the map gets on a large screen.
PLOT_PADDING_MIN = 22
PLOT_PADDING_FRAC = 0.06
POINT_DIAMETER = 5.0
# How far the cursor may be from a point and still select it. Larger than
# the point itself, so a dot does not have to be hit dead-centre; small
# enough that in a dense cloud the nearest point is unambiguous.
PICK_RADIUS = 11.0
# The box the hovered crop is displayed in, and the gap between it and the
# point it describes -- big enough not to cover it, small enough that the
# eye does not have to travel. The image inside is the original, scaled to
# fit by the browser; only its display size is bounded here.
HOVER_THUMB = 120
HOVER_OFFSET = 16
# Client-side throttle on hover and drag events. At 40ms the highlight and
# the lasso trail still feel immediate while the server sees ~25 events/s
# at most, however fast the cursor moves.
HOVER_INTERVAL_MS = 40
DRAG_INTERVAL_MS = 40
# The plot is drawn on white rather than on the app's dark surface: the
# categorical palette in components.py was picked to be distinguishable on
# matplotlib's white figure background, and the same colors on a dark
# ground read as different colors entirely.
PLOT_BGCOLOR = "#ffffff"
MARKER_COLOR = "#111111"
LASSO_COLOR = "#1f6feb"
ZOOM_BOX_COLOR = "#a04ec9"
# A dragged path is sampled rather than recorded point for point: below
# this many pixels of travel the extra vertex changes neither the drawn
# trail nor which points fall inside it, and every vertex is JSON that
# goes over the wire again on the next drag event.
LASSO_MIN_STEP = 4.0
# Under this many vertices there is no loop to speak of -- that is a click
# that wobbled, not a selection, and treating it as one would throw away
# the user's previous selection for nothing.
LASSO_MIN_VERTICES = 4
# ...and the same idea for the box zoom: a box smaller than this is a click,
# not a rectangle, and zooming to it would jump to a meaningless extreme.
ZOOM_BOX_MIN_SIDE = 12.0
# One wheel notch multiplies the zoom by this. Small enough that a zoom
# feels continuous, large enough that reaching a dense cluster does not
# take twenty notches.
ZOOM_STEP = 1.25
# Zooming out past the fitted view only shrinks the cloud into a dot in the
# middle of an empty box, so the fit *is* the minimum. The maximum is where
# a 5px dot stops resolving anything new.
ZOOM_MAX = 100.0
# Redrawing means re-projecting and re-serializing the visible points, so a
# margin is drawn beyond the box: without it, a point whose centre is just
# outside would pop in and out as the cursor moves it across the edge.
DRAW_MARGIN = 24.0
# How many points a *draft* frame draws -- the frames produced while a drag
# or a wheel zoom is still in flight (see _RedrawPump). Every point on the
# map is a coordinate pair serialized to JSON and pushed over the websocket,
# so the cost of a frame is linear in how many are visible: 50,000 of them
# measured at ~130ms a frame, which a drag sampled every 40ms cannot keep up
# with. A draft thins the cloud to this many by taking every k-th point, so
# a 50k map drags at the cost of a 10k one and the full cloud comes back the
# moment the gesture stops. Below this, nothing is thinned at all.
DRAFT_MAX_POINTS = 10_000
# How long the map must sit still before the full-detail frame is drawn.
# Long enough that the pauses inside a drag do not each trigger one, short
# enough to feel like part of releasing the mouse.
REDRAW_SETTLE_S = 0.15
# The redraw thread is started on demand and gives up after this long with
# nothing to draw, so an abandoned projection leaves no thread behind.
REDRAW_IDLE_S = 5.0
# Radius of a prototype marker. Deliberately much larger than a data point:
# in matplotlib these are s=130..210 against s=6 for the cloud.
PROTOTYPE_RADIUS = 9.0
# Marker shapes indexed by subcenter -- the canvas equivalent of
# projection_view.SUBCENTER_MARKERS, in the same order, so a prototype
# keeps its shape across both renderings. Drawn as polygons rather than
# glyphs: a text marker would depend on a font having that character, and
# the app ships one Latin subset and no network to fetch anything else.
SUBCENTER_SHAPES = ["star", "plus", "cross", "diamond", "triangle_up", "triangle_down", "square", "pentagon"]

# What a point of each *source* is drawn as. Keys, compared in code, never
# shown: the dataset's own crops are round dots, and anything overlaid onto
# the map from elsewhere (a simple inference -- see core/data_manager.py)
# is a slightly larger square. Shape rather than color, because color is
# already spent on whatever "Color by" means: with both, a square in the
# middle of a cluster says "this crop is not from the dataset" *and* keeps
# saying what its confidence or its sensor was.
#
# Both are drawn by the same fast path as before -- a Points shape whose
# stroke cap turns a zero-length segment into a dot -- so a second marker
# shape costs nothing per point: ROUND gives a disc, SQUARE a square.
POINT_KIND_DISC = "disc"
POINT_KIND_SQUARE = "square"
POINT_STYLES = {
    POINT_KIND_DISC: (ft.StrokeCap.ROUND, 1.0),
    POINT_KIND_SQUARE: (ft.StrokeCap.SQUARE, 1.3),
}


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _plot_padding(size: float) -> float:
    """Blank margin kept between the fitted cloud and the edge of the map."""
    return max(PLOT_PADDING_MIN, size * PLOT_PADDING_FRAC)


def _marker_polygon(kind: str, cx: float, cy: float, r: float) -> list:
    """Vertices of one prototype marker, centred on (cx, cy). Every shape is
    a closed polygon, so it can be both filled with the class color and
    stroked black the way the matplotlib markers are."""
    def ring(n: int, start_deg: float, radius: float) -> list:
        return [
            (cx + radius * math.cos(math.radians(start_deg + i * 360 / n)),
             cy + radius * math.sin(math.radians(start_deg + i * 360 / n)))
            for i in range(n)
        ]

    if kind == "star":
        # Alternating outer/inner radius, starting at the top.
        return [
            (cx + (r if i % 2 == 0 else r * 0.45) * math.cos(math.radians(-90 + i * 36)),
             cy + (r if i % 2 == 0 else r * 0.45) * math.sin(math.radians(-90 + i * 36)))
            for i in range(10)
        ]
    if kind in ("plus", "cross"):
        a = r * 0.34  # half-width of an arm
        pts = [
            (-a, -r), (a, -r), (a, -a), (r, -a), (r, a), (a, a),
            (a, r), (-a, r), (-a, a), (-r, a), (-r, -a), (-a, -a),
        ]
        if kind == "cross":  # the same cross, turned 45 degrees
            c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
            pts = [(x * c - y * s, x * s + y * c) for x, y in pts]
        return [(cx + x, cy + y) for x, y in pts]
    if kind == "diamond":
        return ring(4, -90, r)
    if kind == "triangle_up":
        return ring(3, -90, r)
    if kind == "triangle_down":
        return ring(3, 90, r)
    if kind == "square":
        return ring(4, -45, r)
    return ring(5, -90, r)  # pentagon, and the fallback for anything else


class _RedrawPump:
    """Turns a stream of viewport changes into a stream of frames the server
    can actually keep up with.

    Flet dispatches every event handler on its own thread from a pool (see
    Page.run_thread), so a drag does not queue politely: each pointer sample
    the client sends -- one every DRAG_INTERVAL_MS -- starts a redraw of its
    own, and at 50,000 points those take ~130ms each. They overlap, they
    pile up, and the map ends up seconds behind the cursor while every one
    of them re-serializes the whole cloud. Left alone this gets worse the
    more points there are, which is exactly backwards.

    So redraws go through here instead, and this owns the only thread that
    draws:

    * one frame at a time, so two of them can never interleave on the same
      coordinate arrays;
    * the next frame starts no sooner than the previous one took, so
      drawing never eats more than half the wall clock and a request that
      arrives mid-frame replaces the one behind it instead of adding to it;
    * intermediate frames are *drafts* (a thinned cloud, see
      _point_shapes), and a full-detail frame always lands once the map has
      been still for REDRAW_SETTLE_S -- so what the user ends up looking at
      is never a draft.

    `draw(draft)` is called on that thread. Flet control mutations from a
    worker thread are fine and are what every heavy rebuild in this app
    does (see components.BackgroundRebuild); it is only the page that must
    not be blocked.
    """

    def __init__(self, draw):
        self._draw = draw
        # Guards every field below *and* is the wake-up signal for the
        # thread: requesting a frame and the thread deciding to exit have to
        # be one atomic decision, or a request can be dropped on the floor.
        self._cv = threading.Condition()
        self._pending: str | None = None  # None | "draft" | "full"
        self._thread: threading.Thread | None = None

    def request(self, full: bool = False) -> None:
        """Ask for a frame. `full` for the end of a gesture -- a released
        drag, a finished box zoom -- where the next frame is the one the
        user will be reading."""
        with self._cv:
            # "full" wins over a draft still waiting: they draw the same
            # viewport, and only the detail differs.
            self._pending = "full" if full or self._pending == "full" else "draft"
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, daemon=True, name="lookin-projection-redraw")
                self._thread.start()
            self._cv.notify()

    def _loop(self) -> None:
        while True:
            with self._cv:
                if self._pending is None:
                    self._cv.wait(REDRAW_IDLE_S)
                    if self._pending is None:
                        self._thread = None
                        return
                mode, self._pending = self._pending, None
            t0 = time.monotonic()
            try:
                self._draw(draft=(mode == "draft"))
            except Exception as ex:
                # A dead redraw thread would leave the map frozen with no
                # sign of why, and the next gesture would start a new one
                # into the same failure.
                _log(f"projection redraw failed: {ex}")
            cost = time.monotonic() - t0
            with self._cv:
                if self._pending is not None:
                    # Something moved while we drew: pace the next frame so
                    # redrawing stays at most half the wall clock.
                    self._cv.wait(cost)
                elif mode == "draft":
                    # Nothing new: give the gesture a moment to prove it is
                    # over, then finish in full detail.
                    self._cv.wait(REDRAW_SETTLE_S)
                    if self._pending is None:
                        self._pending = "full"


class InteractiveProjection:
    """A clickable 2D scatter of already-projected points.

    points: (N, 2) projected coordinates -- whatever projection method
        produced them is irrelevant here.
    point_colors: (N,) hex colors, one per point. The caller owns the
        coloring decision (by class, by any column of the joined frame, ...)
        so that this view and the matplotlib one can be given exactly the
        same colors -- see core/point_coloring.py.
    legend: [(color, label)] rendered by legend_control(), or an empty list.
    uuids: (N,) row uuids, index-aligned with `points`.
    row_by_uuid / crop_col / label_col: enough to resolve a point to its
        crop, its class and the rest of its row -- all straight from the
        analysis context built in embedding_context.build_analysis_context.
    records: (N,) rows, index-aligned with `points`, used instead of
        row_by_uuid when given -- the map can show more than one pool at
        once and a uuid is only unique within one of them.
    point_kinds: (N,) POINT_KIND_* per point, or None for a map of nothing
        but dataset crops. What tells an overlaid inference crop from a
        dataset one at a glance, whatever the coloring says.
    point_groups: (N,) name of the group each point was colored by -- its
        class, its value of whatever column the coloring uses, or the band
        it fell into on a numeric ramp. When every legend entry is one of
        these, the legend becomes a set of checkboxes that show and hide
        groups *in place* -- no recompute, unlike the class picker in the
        toolbar which re-samples and re-projects.
    legend_title: the heading over those checkboxes, naming what they
        filter on ("Show classes", "Show correct", ...).
    prototypes: (coords (M, 2), colors, subcenter indices, class labels)
        in the same projected space as `points`, or None. Drawn over the
        cloud, colored like their class and shaped by subcenter -- and
        hoverable, since a marker sitting in the middle of a cluster
        otherwise gives no way to tell whose prototype it is.
    on_select: called with the positions of the points inside a finished
        lasso. What that means is the caller's decision (the projection tab
        opens them in the carousel), so this control stays about the plot
        itself. Positions rather than uuids because the map can hold points
        from more than one pool -- see `records`.
    on_point: called with the position of a clicked point. The carousel is
        where the review controls live, so this is the path from "that
        point looks wrong" to relabelling it. With no owner to hand it to
        a click falls back to the shared full-size viewer, which at least
        shows the crop rather than swallowing the gesture.
    size: side of the square map in pixels. The layout computes it from the
        window (see projection_view) and calls resize() when that changes.

    The pieces it builds -- map_control(), toolbar_controls() and
    legend_control() -- are placed separately by the caller, because only
    the map may sit outside a scrolling region (see the module docstring)
    and because the toolbar shares a line with the caller's own controls.
    """

    def __init__(
        self, ctx, points, point_colors: list, legend: list, uuids,
        row_by_uuid: dict, crop_col: str, uuid_col: str, title: str = "projection",
        prototypes=None, on_select=None, on_point=None, label_col: str | None = None,
        point_groups=None, legend_title: str = "Show classes",
        size: int = DEFAULT_PLOT_SIZE, records: list | None = None, point_kinds=None,
    ):
        self.ctx = ctx
        self.points = np.asarray(points, dtype=float)
        self.point_colors = point_colors
        # The same colors as an array: grouping the visible points by color
        # is done on every frame, and at 50,000 points a Python loop over
        # them costs more than the numpy comparison that replaces it.
        self._color_array = np.asarray(point_colors)
        self.legend = legend
        self.uuids = uuids
        self.row_by_uuid = row_by_uuid
        # Index-aligned rows, when the caller has them: with points from
        # more than one pool on the map, a uuid is no longer a unique key
        # (nothing stops a set of external crops from numbering its rows
        # the way the dataset does), and resolving a point through one
        # would eventually show the wrong crop. Position always identifies
        # exactly one point.
        self.records = records
        self.point_kinds = None if point_kinds is None else np.asarray(point_kinds)
        self.crop_col = crop_col
        self.uuid_col = uuid_col
        self.label_col = label_col
        self.point_groups = None if point_groups is None else np.asarray(point_groups)
        self.legend_title = legend_title
        self.title = title
        self.prototypes = prototypes
        self.on_select = on_select
        self.on_point = on_point
        self.size = max(MIN_PLOT_SIZE, int(size))

        # The viewport: a scale and a data-space centre. _fit() sets the
        # values that show everything, and they are the *only* thing zoom,
        # pan and resize change -- every screen coordinate is derived from
        # them by _to_screen, so there is nothing else to keep in step.
        self._fit()
        self.sx, self.sy = self._to_screen(self.points[:, 0], self.points[:, 1])
        self.psx, self.psy = self._prototype_screen()
        # What the pointer is over: ("point", i) or ("prototype", j), or
        # None. A tuple rather than a bare index because the two answer
        # different hovers and the card has to know which it is showing.
        self._current: tuple | None = None
        # The tab hides these controls while the static rendering is on
        # screen. A flag rather than an assignment onto the controls, since
        # half of them own their own visibility -- Reset view appears with
        # a zoom, the selection buttons with a lasso -- and would turn
        # themselves straight back on. Everything goes through _sync_toolbar.
        self._toolbar_visible = True
        # Pan is the default because it is the harmless one: a drag that
        # moves the view can always be undone by dragging back, where a
        # stray drag in lasso mode replaces a selection.
        self._mode = "pan"  # or "lasso" / "zoom" -- gesture events carry no modifier keys
        self._lasso: list = []
        # The finished loop, kept in *data* coordinates: it is redrawn from
        # these on every zoom, which a screen-space copy could not survive.
        self._lasso_data: np.ndarray | None = None
        self._zoom_box: tuple | None = None
        self._selected: np.ndarray = np.array([], dtype=int)
        self._hidden_groups: set = set()
        # Groups brought to the front from the legend, in the order they
        # were: the last one is drawn over everything. Without it the draw
        # order is whatever order the colors sort in, and a small class
        # that happens to sort first can sit entirely under a big one.
        self._front_groups: list = []
        # The displayed bytes, keyed by (point index, lut, quarter turns):
        # the display setting is in the key so changing it re-encodes
        # instead of serving the crop as it looked before.
        self._b64_by_index: dict = {}
        # The signature of the frame _redraw last drew (see
        # _frame_signature). None means "something else painted over it",
        # which is what the two paths that repaint a layer on their own --
        # _render_selection and _redraw_shown -- set it to.
        self._last_frame: tuple | None = None
        # Every viewport change goes through this rather than redrawing on
        # the event's own thread -- see _RedrawPump.
        self._pump = _RedrawPump(self._redraw)

        self._build_canvases()
        self._build_hover_card()
        self._build_toolbar()

    # ---- construction --------------------------------------------------

    def _build_canvases(self) -> None:
        # Five canvases rather than one, split by what gets redrawn
        # together: the cloud and the prototypes change on a viewport
        # change, the selection once per lasso, and the cursor layer on
        # every mouse move. Updating a shape means updating its canvas, so
        # keeping the coordinates out of the layer that moves is what makes
        # a mouse move cheap.
        self.points_canvas = cv.Canvas(shapes=self._point_shapes(), width=self.size, height=self.size)
        self.prototypes_canvas = cv.Canvas(shapes=self._prototype_shapes(), width=self.size, height=self.size)
        self.selection_canvas = cv.Canvas(shapes=[], width=self.size, height=self.size)
        self.marker = cv.Circle(
            x=0, y=0, radius=POINT_DIAMETER * 1.8, visible=False,
            paint=ft.Paint(color=MARKER_COLOR, style=ft.PaintingStyle.STROKE, stroke_width=2.0, anti_alias=True),
        )
        self.lasso_shape = cv.Points(
            points=[], point_mode=cv.PointMode.POLYGON, visible=False,
            paint=ft.Paint(color=LASSO_COLOR, style=ft.PaintingStyle.STROKE, stroke_width=1.8, anti_alias=True),
        )
        self.zoom_box_shape = cv.Rect(
            x=0, y=0, width=0, height=0, visible=False,
            paint=ft.Paint(color=ZOOM_BOX_COLOR, style=ft.PaintingStyle.STROKE, stroke_width=1.5, anti_alias=True),
        )
        self.cursor_canvas = cv.Canvas(
            shapes=[self.lasso_shape, self.zoom_box_shape, self.marker], width=self.size, height=self.size,
        )

    def _build_hover_card(self) -> None:
        self.hover_image = ft.Image(
            width=HOVER_THUMB, height=HOVER_THUMB, fit=ft.ImageFit.CONTAIN, border_radius=4,
        )
        self.hover_label = ft.Text(size=10, color=ft.Colors.ON_SURFACE_VARIANT, no_wrap=True)
        # A prototype has no crop, so the same card is reused with the
        # image hidden and this line carrying the answer instead.
        self.hover_note = ft.Text(size=11, weight=ft.FontWeight.BOLD, no_wrap=True, visible=False)
        # Positioned inside the plot Stack, *under* the gesture detector:
        # the detector's own (all but invisible) surface stays on top, so
        # the card never steals the pointer events that place it.
        self.hover_card = ft.Container(
            left=0, top=0, visible=False, padding=6, border_radius=8,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.35, "#000000")),
            content=ft.Column([self.hover_image, self.hover_note, self.hover_label], spacing=4, tight=True),
        )

    def _build_toolbar(self) -> None:
        # Drag does one thing at a time and there is no modifier key to tell
        # them apart -- Flet's gesture events carry none -- so the meanings
        # of a drag are an explicit choice.
        self.mode_switch = ft.SegmentedButton(
            selected={"pan"},
            on_change=lambda e: self._set_mode(next(iter(e.control.selected), "pan")),
            segments=[
                ft.Segment(value="pan", label=ft.Text("Pan"), icon=ft.Icon(ft.Icons.OPEN_WITH)),
                ft.Segment(value="lasso", label=ft.Text("Lasso"), icon=ft.Icon(ft.Icons.GESTURE)),
                ft.Segment(value="zoom", label=ft.Text("Zoom"), icon=ft.Icon(ft.Icons.CROP_FREE)),
            ],
        )
        self.zoom_text = ft.Text("×1.0", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        # Buttons as well as the wheel: a trackpad, or a browser that
        # handles wheel events its own way, must not be able to lock a user
        # out of zooming at all.
        self.zoom_in_btn = ft.IconButton(
            ft.Icons.ADD, tooltip="Zoom in", icon_size=18, on_click=lambda e: self._zoom_by(ZOOM_STEP))
        self.zoom_out_btn = ft.IconButton(
            ft.Icons.REMOVE, tooltip="Zoom out", icon_size=18, on_click=lambda e: self._zoom_by(1 / ZOOM_STEP))
        # An icon, not a labelled button: it appears the moment anyone
        # zooms, and it shares one line with the renderer switch and the
        # mode switch (see projection_view._switcher). At half-width the
        # word "Reset view" is the difference between that line fitting and
        # the whole toolbar folding onto a second one, which comes out of
        # the map.
        self.reset_view_btn = ft.IconButton(
            ft.Icons.ZOOM_OUT_MAP, tooltip="Reset view", icon_size=18,
            visible=False, on_click=lambda e: self._reset_view())
        self.selection_text = ft.Text(size=11, weight=ft.FontWeight.BOLD)
        self.show_selection_btn = ft.TextButton(
            "Show selection", icon=ft.Icons.PHOTO_LIBRARY_OUTLINED, visible=False,
            on_click=lambda e: self._emit_selection())
        self.clear_selection_btn = ft.TextButton(
            "Clear", icon=ft.Icons.CLOSE, visible=False, on_click=lambda e: self._clear_selection())
        self._set_selection_text()

    def set_toolbar_visible(self, visible: bool) -> None:
        """Show or hide the whole toolbar. The tab calls this when the
        user switches renderings: these controls drive the canvas and do
        nothing at all to a matplotlib PNG."""
        self._toolbar_visible = visible
        self._sync_toolbar()
        for control in self.toolbar_controls():
            if control.page:
                control.update()

    def _sync_toolbar(self) -> None:
        """The one place a toolbar control's visibility is decided: the
        tab's show/hide and the control's own condition, resolved together.
        Setting either on its own is what would let a hidden toolbar
        reappear one button at a time as the user zooms or lassos."""
        shown = self._toolbar_visible
        for control in (self.mode_switch, self.zoom_out_btn, self.zoom_text, self.zoom_in_btn):
            control.visible = shown
        self.reset_view_btn.visible = shown and self._scale > self._fit_scale * 1.001
        self.selection_text.visible = shown
        self.show_selection_btn.visible = shown and len(self._selected) > 0
        self.clear_selection_btn.visible = shown and self._lasso_data is not None

    # ---- geometry ----------------------------------------------------
    #
    # One transform, used for drawing, for picking, for the lasso and for
    # the box zoom. x and y share a single scale factor so the aspect ratio
    # of the projection is preserved -- a spherical UMAP has to stay a
    # circle, and even for PCA a stretched axis would misrepresent the
    # distances the plot is read for.

    def _fit(self) -> None:
        """Set the viewport to the one that shows everything -- the initial
        view, and what "Reset view" goes back to.

        Prototypes are inside the fitted extent rather than clipped against
        one fitted to the cloud alone: matplotlib autoscales over
        everything it draws, so leaving them out would put the same
        prototype in two different places in the two renderings."""
        xs, ys = [self.points[:, 0]], [self.points[:, 1]]
        if self.prototypes is not None and len(self.prototypes[0]):
            coords = np.asarray(self.prototypes[0], dtype=float)
            xs.append(coords[:, 0])
            ys.append(coords[:, 1])
        all_x, all_y = np.concatenate(xs), np.concatenate(ys)

        self._span = max(float(np.ptp(all_x)), float(np.ptp(all_y))) or 1.0
        self._fit_scale = (self.size - 2 * _plot_padding(self.size)) / self._span
        self._scale = self._fit_scale
        self._cx = (float(all_x.min()) + float(all_x.max())) / 2
        self._cy = (float(all_y.min()) + float(all_y.max())) / 2
        self._base_centre = (self._cx, self._cy)

    def _to_screen(self, x, y):
        half = self.size / 2
        # y is negated: screen coordinates grow downwards, and the plot has
        # to match matplotlib's upwards y or the two views would be mirror
        # images of each other.
        return half + (x - self._cx) * self._scale, half - (y - self._cy) * self._scale

    def _to_data(self, px, py):
        """Inverse of _to_screen -- what data point sits under a cursor.
        Takes scalars or arrays, the latter to carry a finished lasso or a
        zoom box back into data space. Zooming needs it to keep the point
        under the pointer pinned while the scale changes, which is what
        makes a wheel zoom land where one is looking."""
        half = self.size / 2
        return self._cx + (px - half) / self._scale, self._cy - (py - half) / self._scale

    def _visible(self) -> np.ndarray:
        """Indices of the points currently inside the box. Zoomed out this
        is everything; zoomed in it is a fraction, which is what keeps a
        redraw from getting *more* expensive the further one zooms in --
        the coordinates that go over the wire are only the ones on screen.

        Picking is deliberately not filtered this way: an off-screen point
        has off-screen coordinates, so the cursor cannot be near it anyway."""
        return np.flatnonzero(self._in_box() & self._shown_mask())

    def _in_box(self) -> np.ndarray:
        return (
            (self.sx >= -DRAW_MARGIN) & (self.sx <= self.size + DRAW_MARGIN)
            & (self.sy >= -DRAW_MARGIN) & (self.sy <= self.size + DRAW_MARGIN)
        )

    def _shown_mask(self) -> np.ndarray:
        """False for points whose class is unticked in the legend. It gates
        picking as well as drawing: a hidden point still has coordinates,
        and a hover that produced the crop of something invisible would be
        the worst kind of wrong answer."""
        if not self._hidden_groups or self.point_groups is None:
            return np.ones(len(self.sx), dtype=bool)
        return ~np.isin(self.point_groups, list(self._hidden_groups))

    def _draft_sample(self, indices: np.ndarray) -> np.ndarray:
        """Thin a set of point indices down to DRAFT_MAX_POINTS for a frame
        drawn mid-gesture.

        Every k-th one rather than a random sample, and taken from indices
        that are already in order, so the subset is the same from one frame
        to the next: the cloud thins evenly while the drag lasts instead of
        shimmering as a different sample is drawn each time. Nothing else
        changes -- the same points keep their colors and positions, and the
        frame that lands when the gesture stops has all of them back."""
        if len(indices) <= DRAFT_MAX_POINTS:
            return indices
        return indices[:: -(-len(indices) // DRAFT_MAX_POINTS)]

    def _point_shapes(self, draft: bool = False) -> list:
        """One Points shape per (color, marker shape). A Points shape
        carries a single Paint, so those two are what force a split --
        everything sharing both is drawn in one shape, whatever class it
        belongs to."""
        indices = self._visible()
        if draft:
            indices = self._draft_sample(indices)
        return self._styled_shapes(indices)

    def _styled_shapes(self, indices) -> list:
        """The Points shapes for `indices`, layer by layer: a canvas paints
        its shapes in list order, so the groups brought to the front are
        simply the last ones in the list."""
        return [
            self._points_shape(color, kind, idx)
            for layer in self._layers(indices)
            for (color, kind), idx in self._indices_by_style(layer).items()
        ]

    def _layers(self, indices) -> list:
        """Split point indices into draw layers, bottom first: every group
        not brought to the front, then each front group in the order it was
        brought there. Split by *group* rather than by color, because with
        more classes than palette entries two classes share a color and
        only one of them was asked for."""
        indices = np.asarray(indices, dtype=int)
        front = self._front_groups
        if not front or self.point_groups is None:
            return [indices]
        groups = self.point_groups[indices]
        layers = [indices[~np.isin(groups, front)]]
        layers += [indices[groups == group] for group in front]
        return [layer for layer in layers if len(layer)]

    def _layer_rank(self) -> np.ndarray:
        """Per point, which layer it is drawn in (0 is the bottom one) --
        what lets a hover answer with the point that is actually visible
        where several overlap."""
        rank = np.zeros(len(self.sx), dtype=int)
        for position, group in enumerate(self._front_groups, start=1):
            rank[self.point_groups == group] = position
        return rank

    def _points_shape(self, color: str, kind: str, idx) -> cv.Points:
        # A round stroke cap is what turns a zero-length point into a
        # filled disc (a square one, into a square); stroke_width is
        # therefore the marker size. It does not scale with the zoom, on
        # purpose: resolving points that overlap is the reason to zoom in
        # at all.
        cap, scale = POINT_STYLES.get(kind, POINT_STYLES[POINT_KIND_DISC])
        return cv.Points(
            points=self._offsets(idx), point_mode=cv.PointMode.POINTS,
            paint=ft.Paint(
                color=color, stroke_width=POINT_DIAMETER * scale,
                stroke_cap=cap, anti_alias=True,
            ),
        )

    def _indices_by_style(self, indices) -> dict:
        """Split point indices by (color, marker shape) -- how they are
        drawn.

        One numpy pass per distinct style rather than one Python step per
        point: a palette has a dozen entries, there are two shapes, and a
        map can have hundreds of thousands of points, so the loop belongs
        on the styles. A map with one shape (the usual case: nothing
        overlaid) splits by color exactly as it did before."""
        indices = np.asarray(indices, dtype=int)
        colors = self._color_array[indices]
        if self.point_kinds is None:
            return {(str(color), POINT_KIND_DISC): indices[colors == color]
                    for color in np.unique(colors)}
        kinds = self.point_kinds[indices]
        out = {}
        for color in np.unique(colors):
            for kind in np.unique(kinds):
                members = indices[(colors == color) & (kinds == kind)]
                if len(members):
                    out[(str(color), str(kind))] = members
        return out

    def _offsets(self, indices) -> list:
        # Rounded to a tenth of a pixel: sub-pixel precision is not
        # visible, and the coordinates are serialized to JSON and sent over
        # the wire -- for 10k points the shorter numbers are worth more
        # than the invisible accuracy.
        #
        # Built by numpy and handed over in one .tolist(): the same rounding
        # written as a Python comprehension is ten times slower, and this
        # runs on every frame of every drag.
        idx = np.asarray(indices, dtype=int)
        return np.round(np.column_stack([self.sx[idx], self.sy[idx]]), 1).tolist()

    def _prototype_shapes(self) -> list:
        """Prototypes as filled, black-edged polygons: colored like their
        class (so one can be matched to its own cloud) and shaped by
        subcenter, exactly as in the matplotlib rendering.

        Grouped by (color, shape) rather than drawn one at a time: a single
        canvas Path holds as many closed contours as we put in it, so a
        31-class, 2-subcenter model costs a few dozen shapes instead of a
        few hundred."""
        if self.prototypes is None:
            return []
        coords = self.prototypes[0]
        colors, ks = self.prototypes[1], self.prototypes[2]
        if not len(coords):
            return []
        px, py = self._prototype_screen()

        grouped: dict = {}
        for i in range(len(coords)):
            kind = SUBCENTER_SHAPES[int(ks[i]) % len(SUBCENTER_SHAPES)]
            grouped.setdefault((colors[i], kind), []).append(i)

        shapes = []
        for (color, kind), members in grouped.items():
            elements = []
            for i in members:
                polygon = _marker_polygon(kind, float(px[i]), float(py[i]), PROTOTYPE_RADIUS)
                elements.append(cv.Path.MoveTo(polygon[0][0], polygon[0][1]))
                elements += [cv.Path.LineTo(x, y) for x, y in polygon[1:]]
                elements.append(cv.Path.Close())
            # Two passes over the same contours, because a Paint is either
            # a fill or a stroke: the fill carries the class color, the
            # stroke is the black edge that lifts a prototype off a cloud
            # of its own color.
            shapes.append(cv.Path(elements=elements, paint=ft.Paint(
                color=color, style=ft.PaintingStyle.FILL, anti_alias=True,
            )))
            shapes.append(cv.Path(elements=elements, paint=ft.Paint(
                color="#000000", style=ft.PaintingStyle.STROKE, stroke_width=1.2, anti_alias=True,
            )))
        return shapes

    def _prototype_screen(self):
        """Prototype positions in screen space, or (None, None) when there
        are none. Computed alongside the points so that drawing them and
        picking them read the same arrays -- the same reason the cloud has
        only one transform."""
        if self.prototypes is None or not len(self.prototypes[0]):
            return None, None
        coords = np.asarray(self.prototypes[0], dtype=float)
        return self._to_screen(coords[:, 0], coords[:, 1])

    def _pick(self, px: float, py: float) -> int | None:
        d2 = (self.sx - px) ** 2 + (self.sy - py) ** 2
        if self._hidden_groups:
            d2 = np.where(self._shown_mask(), d2, np.inf)
        if self._front_groups and self.point_groups is not None:
            # Among the points within reach, the ones on the topmost layer
            # win: a point drawn over another is the one the user sees, so
            # it is the one the cursor is on.
            near = d2 <= PICK_RADIUS ** 2
            if near.any():
                rank = self._layer_rank()
                d2 = np.where(near & (rank == rank[near].max()), d2, np.inf)
        i = int(np.argmin(d2))
        return i if d2[i] <= PICK_RADIUS ** 2 else None

    def _pick_prototype(self, px: float, py: float) -> int | None:
        """Prototypes are tested before points and with a wider radius:
        they are drawn on top and are three times the size, so a cursor
        over one is pointing at it, not at whatever point happens to be
        underneath."""
        if self.psx is None:
            return None
        d2 = (self.psx - px) ** 2 + (self.psy - py) ** 2
        j = int(np.argmin(d2))
        return j if d2[j] <= (PROTOTYPE_RADIUS + 4) ** 2 else None

    # ---- viewport ------------------------------------------------------

    def _frame_signature(self, draft: bool) -> tuple:
        """What a drawn frame depends on: the viewport, the detail level,
        and which classes are drawn. Two frames with the same signature are
        the same picture -- as long as nothing outside _redraw has painted
        since, which is why those paths clear _last_frame instead of being
        listed here (a selection is tens of thousands of indices, far too
        much to compare on every frame)."""
        return (self._cx, self._cy, self._scale, self.size, draft, frozenset(self._hidden_groups),
                tuple(self._front_groups))

    def _request_redraw(self, full: bool = False) -> None:
        """The viewport moved -- ask for a frame. Everything that changes
        the viewport goes through here rather than drawing inline, so that
        a continuous gesture cannot outrun the drawing (see _RedrawPump).
        `full` is for the changes that are not part of a gesture, or that
        end one: they get full detail straight away."""
        self._pump.request(full=full)

    def _redraw(self, draft: bool = False) -> None:
        """Re-project everything and push the layers that depend on the
        viewport. This is the whole cost of a zoom, a pan or a resize: a
        few numbers changed, the coordinates recomputed from them, and only
        the visible ones sent.

        Called on the redraw thread, never inline from an event handler.
        A `draft` frame draws a thinned cloud, which is what keeps a drag
        responsive on a map with more points than a frame can carry."""
        # A gesture ends with a request for full detail and is usually
        # followed by the settle timer asking for the same frame again; and
        # a drag that stopped moving keeps sending the pointer position.
        # Drawing the picture already on screen is pure cost -- 40ms and
        # three quarters of a megabyte at 50k points -- so the frame it
        # would produce is compared first.
        signature = self._frame_signature(draft)
        if signature == self._last_frame:
            return
        self._last_frame = signature
        self.sx, self.sy = self._to_screen(self.points[:, 0], self.points[:, 1])
        self.psx, self.psy = self._prototype_screen()
        self.points_canvas.shapes = self._point_shapes(draft=draft)
        self.prototypes_canvas.shapes = self._prototype_shapes()
        self._render_selection_shapes(draft=draft)
        self._redraw_lasso_outline()
        # The hover card describes a point at a position that just moved,
        # and the pointer has not said where it is since -- so it is
        # dismissed rather than left pointing at the wrong dot.
        self._hide_hover()
        self.zoom_text.value = f"×{self._scale / self._fit_scale:.1f}"
        self._sync_toolbar()
        if self.points_canvas.page:
            self.points_canvas.update()
            self.prototypes_canvas.update()
            self.selection_canvas.update()
            self.cursor_canvas.update()
            self.hover_card.update()
            self.zoom_text.update()
            self.reset_view_btn.update()

    def _apply_zoom(self, new_scale: float, anchor: tuple | None = None) -> None:
        """Move to `new_scale`, keeping the data point under `anchor` (a
        screen position, the box centre if omitted) exactly where it is.
        That anchoring is what makes a wheel zoom land where one is looking
        instead of drifting towards the middle."""
        new_scale = max(self._fit_scale, min(ZOOM_MAX * self._fit_scale, new_scale))
        if new_scale == self._scale:
            return
        half = self.size / 2
        ax, ay = anchor if anchor is not None else (half, half)
        dx, dy = self._to_data(ax, ay)
        self._scale = new_scale
        self._cx = dx - (ax - half) / new_scale
        self._cy = dy + (ay - half) / new_scale
        self._clamp_centre()
        self._request_redraw()

    def _zoom_by(self, factor: float) -> None:
        self._apply_zoom(self._scale * factor)

    def _on_scroll(self, e) -> None:
        if not e.scroll_delta_y:
            return
        factor = ZOOM_STEP if e.scroll_delta_y < 0 else 1 / ZOOM_STEP
        self._apply_zoom(self._scale * factor, anchor=(e.local_x, e.local_y))

    def _clamp_centre(self) -> None:
        """Keep the fitted extent from being dragged entirely off screen.
        Without this, one flick of the pan leaves an empty white box and no
        way to tell which direction the data went."""
        half_span = (self.size / 2) / self._scale
        fit_half = (self.size / 2 - _plot_padding(self.size)) / self._fit_scale
        limit = max(0.0, fit_half - half_span)
        base_cx, base_cy = self._base_centre
        self._cx = float(np.clip(self._cx, base_cx - limit, base_cx + limit))
        self._cy = float(np.clip(self._cy, base_cy - limit, base_cy + limit))

    def _reset_view(self) -> None:
        self._fit()
        self._request_redraw(full=True)

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self.mode_switch.selected = {mode}
        if self.mode_switch.page:
            self.mode_switch.update()

    def resize(self, size: int) -> None:
        """Re-lay the map to a new square side, keeping the current zoom.

        The scale is in pixels per data unit, so it has to grow with the
        box: leaving it alone would make the view appear to zoom out every
        time the window is enlarged. What is preserved is the *ratio* to
        the fitted scale, which is what the user reads as the zoom level."""
        size = max(MIN_PLOT_SIZE, int(size))
        if size == self.size:
            return
        zoom_ratio = self._scale / self._fit_scale
        self.size = size
        self._fit_scale = (size - 2 * _plot_padding(size)) / self._span
        self._scale = self._fit_scale * zoom_ratio
        for canvas in (self.points_canvas, self.prototypes_canvas, self.selection_canvas, self.cursor_canvas):
            canvas.width = canvas.height = size
        self.plot_container.width = self.plot_container.height = size
        self.gesture_surface.width = self.gesture_surface.height = size
        self._clamp_centre()
        if self.plot_container.page:
            self.plot_container.update()
        self._request_redraw()

    # ---- hover / click -----------------------------------------------

    def _on_hover(self, e) -> None:
        j = self._pick_prototype(e.local_x, e.local_y)
        target = ("prototype", j) if j is not None else None
        if target is None:
            i = self._pick(e.local_x, e.local_y)
            target = ("point", i) if i is not None else None
        if target == self._current:
            return  # same thing (or still empty space): nothing to redraw
        self._current = target
        if target is None:
            self._hide_hover()
            self._push_cursor()
            self._push_hover()
        elif target[0] == "prototype":
            self._show_prototype_hover(target[1])
        else:
            self._show_hover(target[1])

    def _on_exit(self, e) -> None:
        self._current = None
        self._hide_hover()
        self._push_cursor()
        self._push_hover()

    def _on_tap(self, e) -> None:
        # A prototype has no crop to inspect, and it covers whatever points
        # sit under it -- so clicking one does nothing rather than opening
        # a neighbour the user was not pointing at.
        if self._pick_prototype(e.local_x, e.local_y) is not None:
            return
        i = self._pick(e.local_x, e.local_y)
        if i is None:
            return
        uuid = str(self.uuids[i])
        if self.on_point is not None:
            self.on_point(i)
            return
        # No owner to open it in: the shared viewer at least shows the
        # crop, which beats a click that looks broken.
        b64 = self._image_for(i)
        if b64 is not None:
            self.ctx.image_zoom.open(b64, uuid)

    def _show_hover(self, i: int) -> None:
        """Place the crop beside the hovered *point*, not under the
        cursor itself: anchored to the point it never jitters as the mouse
        wobbles, and it only moves when the highlighted point changes --
        which is also what keeps the update rate down."""
        self.marker.x = float(self.sx[i])
        self.marker.y = float(self.sy[i])
        self.marker.visible = True

        self.marker.radius = POINT_DIAMETER * 1.8
        record = self._record_for(i)
        self.hover_image.src_base64 = self._image_for(i)
        self.hover_image.visible = True
        self.hover_note.visible = False
        label = record.get(self.label_col) if record and self.label_col else None
        self.hover_label.value = str(label) if label is not None else str(self.uuids[i])[:8]

        self._place_hover_card(float(self.sx[i]), float(self.sy[i]), HOVER_THUMB + 12, HOVER_THUMB + 28)
        self._push_cursor()
        self._push_hover()

    def _show_prototype_hover(self, j: int) -> None:
        """Name the prototype under the pointer. A marker sitting inside a
        cluster is unreadable otherwise: the shape says which subcenter but
        nothing says of which class, and the color only helps when the
        legend is short enough to scan."""
        label = self.prototypes[3][j] if len(self.prototypes) > 3 else None
        k_idx = int(self.prototypes[2][j])
        self.marker.x = float(self.psx[j])
        self.marker.y = float(self.psy[j])
        self.marker.radius = PROTOTYPE_RADIUS + 4
        self.marker.visible = True

        self.hover_image.visible = False
        self.hover_note.visible = True
        self.hover_note.value = str(label) if label is not None else "prototype"
        self.hover_label.value = f"subcenter {k_idx}"
        self._place_hover_card(float(self.psx[j]), float(self.psy[j]), 170, 46)
        self._push_cursor()
        self._push_hover()

    def _place_hover_card(self, x0: float, y0: float, card_w: float, card_h: float) -> None:
        """Put the card beside (x0, y0), flipping to the other side near an
        edge so it is never clipped by the plot border."""
        x, y = x0 + HOVER_OFFSET, y0 + HOVER_OFFSET
        if x + card_w > self.size:
            x = x0 - HOVER_OFFSET - card_w
        if y + card_h > self.size:
            y = y0 - HOVER_OFFSET - card_h
        self.hover_card.left = float(np.clip(x, 2, max(2, self.size - card_w - 2)))
        self.hover_card.top = float(np.clip(y, 2, max(2, self.size - card_h - 2)))
        self.hover_card.visible = True

    def _hide_hover(self) -> None:
        self.marker.visible = False
        self.hover_card.visible = False

    # ---- lasso / box zoom ----------------------------------------------

    def _on_pan_start(self, e) -> None:
        if self._mode == "pan":
            return
        if self._mode == "zoom":
            self._zoom_box = (e.local_x, e.local_y, e.local_x, e.local_y)
            self._update_zoom_box_shape()
            self._push_cursor()
            return
        self._lasso = [(e.local_x, e.local_y)]
        self._hide_hover()
        self._push_hover()
        self.lasso_shape.points = list(self._lasso)
        self.lasso_shape.visible = True
        self._push_cursor()

    def _on_pan_update(self, e) -> None:
        if self._mode == "pan":
            # A drag moves the view under the cursor, so the centre travels
            # the opposite way, and by a data distance -- hence the divide
            # by the scale rather than a raw pixel offset.
            self._cx -= e.delta_x / self._scale
            self._cy += e.delta_y / self._scale
            self._clamp_centre()
            self._request_redraw()
            return
        if self._mode == "zoom":
            if self._zoom_box is None:
                return
            x0, y0, _, _ = self._zoom_box
            self._zoom_box = (x0, y0, e.local_x, e.local_y)
            self._update_zoom_box_shape()
            self._push_cursor()
            return
        if not self._lasso:
            return
        lx, ly = e.local_x, e.local_y
        px, py = self._lasso[-1]
        if math.hypot(lx - px, ly - py) < LASSO_MIN_STEP:
            return
        self._lasso.append((lx, ly))
        self.lasso_shape.points = list(self._lasso)
        self._push_cursor()

    def _on_pan_end(self, e) -> None:
        if self._mode == "pan":
            # The drag is over, so the frames it was being served -- thinned
            # ones, drawn as fast as they could be pushed -- give way to the
            # full cloud at the position it was let go at.
            self._request_redraw(full=True)
            return
        if self._mode == "zoom":
            self._finish_zoom_box()
            return
        self._finish_lasso()

    def _update_zoom_box_shape(self) -> None:
        x0, y0, x1, y1 = self._zoom_box
        self.zoom_box_shape.x = min(x0, x1)
        self.zoom_box_shape.y = min(y0, y1)
        self.zoom_box_shape.width = abs(x1 - x0)
        self.zoom_box_shape.height = abs(y1 - y0)
        self.zoom_box_shape.visible = True

    def _finish_zoom_box(self) -> None:
        box, self._zoom_box = self._zoom_box, None
        self.zoom_box_shape.visible = False
        if box is None:
            self._push_cursor()
            return
        x0, y0, x1, y1 = box
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < ZOOM_BOX_MIN_SIDE or h < ZOOM_BOX_MIN_SIDE:
            # A click, not a rectangle: zooming to it would jump to a
            # meaningless extreme.
            self._push_cursor()
            return
        # The box is not square and the map is, so the side that needs the
        # most room decides the scale -- everything the user framed stays
        # inside, with slack on the other axis.
        (dx0, dy0), (dx1, dy1) = self._to_data(x0, y0), self._to_data(x1, y1)
        span = max(abs(dx1 - dx0), abs(dy1 - dy0)) or 1e-9
        self._cx = (dx0 + dx1) / 2
        self._cy = (dy0 + dy1) / 2
        self._scale = max(
            self._fit_scale,
            min(ZOOM_MAX * self._fit_scale, (self.size - 2 * _plot_padding(self.size)) / span),
        )
        self._clamp_centre()
        _log(f"projection box zoom: ×{self._scale / self._fit_scale:.1f}")
        self._request_redraw(full=True)

    def _finish_lasso(self) -> None:
        vertices, self._lasso = self._lasso, []
        if len(vertices) < LASSO_MIN_VERTICES:
            # A click that wobbled: leave whatever was selected alone.
            self.lasso_shape.visible = self._lasso_data is not None
            self._redraw_lasso_outline()
            self._push_cursor()
            return

        # The loop is closed on the user's behalf -- a mouse is released
        # near where the drag started, not exactly on it, and an open path
        # would leave a wedge of points wrongly outside.
        closed = vertices + [vertices[0]]
        self.lasso_shape.points = closed
        self.lasso_shape.visible = True
        screen = np.asarray(closed, dtype=float)
        dx, dy = self._to_data(screen[:, 0], screen[:, 1])
        self._lasso_data = np.column_stack([dx, dy])
        # Hidden classes are not selected, however the loop was drawn: an
        # unticked legend entry is the user saying "not these", and a lasso
        # that swept them up anyway would open crops of a class they had
        # just taken off the map. The mask is applied here rather than only
        # when the selection is drawn (see _render_selection_shapes) so that
        # what is counted, what is highlighted and what the carousel opens
        # are the same set of points.
        inside = MplPath(closed).contains_points(np.column_stack([self.sx, self.sy]))
        self._selected = np.flatnonzero(inside & self._shown_mask())
        n_hidden = int(inside.sum()) - len(self._selected)
        _log(f"projection lasso: {len(self._selected):,} of {len(self.sx):,} points selected"
             + (f" ({n_hidden:,} hidden point(s) skipped)" if n_hidden else ""))

        self._render_selection()
        self._push_cursor()
        if len(self._selected):
            self._emit_selection()

    def _render_selection_shapes(self, draft: bool = False) -> None:
        """A dark halo under the selected points, then those points redrawn
        over it in their own colors -- which highlights them without
        repainting them one flat color, hiding the very class information
        the selection is being judged on.

        Selection is held as point *indices*, so it survives a zoom
        untouched; only where those points land on screen changes, which is
        why this is rebuilt from scratch on every redraw -- and why the
        common case, nothing selected, gets out before working out what is
        on screen at all."""
        if not len(self._selected):
            self.selection_canvas.shapes = []
            return
        selected = self._selected[self._in_box()[self._selected] & self._shown_mask()[self._selected]]
        if draft:
            selected = self._draft_sample(selected)
        if not len(selected):
            self.selection_canvas.shapes = []
            return
        shapes = [cv.Points(
            points=self._offsets(selected), point_mode=cv.PointMode.POINTS,
            paint=ft.Paint(
                color=MARKER_COLOR, stroke_width=POINT_DIAMETER + 4.0,
                stroke_cap=ft.StrokeCap.ROUND, anti_alias=True,
            ),
        )]
        shapes += self._styled_shapes(selected)
        self.selection_canvas.shapes = shapes

    def _render_selection(self) -> None:
        # Paints a layer _redraw also owns, so the frame comparison there
        # can no longer trust what it last drew.
        self._last_frame = None
        self._render_selection_shapes()
        self._set_selection_text()
        if self.selection_canvas.page:
            self.selection_canvas.update()
            self.selection_text.update()
            self.show_selection_btn.update()
            self.clear_selection_btn.update()

    def _reselect_inside_lasso(self) -> None:
        """Recompute the selection from the loop that is still on screen.

        Called when the shown classes change: the loop is the thing the user
        drew and it has not moved, so the answer to "what is selected" is
        re-derived from it rather than pruned -- which is what makes ticking
        a class back on restore its points instead of losing them.
        """
        if self._lasso_data is None:
            return
        lx, ly = self._to_screen(self._lasso_data[:, 0], self._lasso_data[:, 1])
        inside = MplPath(np.column_stack([lx, ly])).contains_points(
            np.column_stack([self.sx, self.sy]))
        self._selected = np.flatnonzero(inside & self._shown_mask())
        self._set_selection_text()

    def _redraw_lasso_outline(self) -> None:
        """Re-project the finished loop. It is kept in data coordinates for
        exactly this: a lasso drawn before a zoom has to stay around the
        same points after it, not around the same pixels."""
        if self._lasso_data is None:
            self.lasso_shape.visible = False
            return
        lx, ly = self._to_screen(self._lasso_data[:, 0], self._lasso_data[:, 1])
        self.lasso_shape.points = [(round(float(x), 1), round(float(y), 1)) for x, y in zip(lx, ly)]
        self.lasso_shape.visible = True

    def _set_selection_text(self) -> None:
        """What the toolbar says about the current selection.

        Driven by whether a *lasso* exists, not by how many points it
        caught: a loop drawn around empty space selects nothing but is
        still on screen, and keying the Clear button off the count left it
        drawn with no way to get rid of it. Saying "0 points selected" is
        also the answer to the question the user just asked."""
        n = len(self._selected)
        has_lasso = self._lasso_data is not None
        self.selection_text.value = f"{n:,} points selected" if has_lasso else ""
        self._sync_toolbar()

    def _clear_selection(self) -> None:
        self._selected = np.array([], dtype=int)
        self._lasso_data = None
        self.lasso_shape.visible = False
        self._render_selection()
        self._push_cursor()

    def _emit_selection(self) -> None:
        if self.on_select is None or not len(self._selected):
            return
        self.on_select([int(i) for i in self._selected])

    # ---- images ---------------------------------------------------------

    def _record_for(self, i: int) -> dict | None:
        if self.records is not None:
            return self.records[i] if 0 <= i < len(self.records) else None
        return self.row_by_uuid.get(str(self.uuids[i]))

    def _crop_path(self, i: int):
        record = self._record_for(i)
        if record is None:
            return None
        return Path(self.ctx.state.dataset.resolve_crop_path(record[self.crop_col]))

    def _image_for(self, i: int) -> str | None:
        """The original file, for the hover card and the full-size viewer
        alike -- see the module docstring for why the small one is not
        allowed to be a cheaper derivative. Shown through the session's
        display setting, like every other crop."""
        record = self._record_for(i)
        if record is None:
            return None
        display = crop_display(self.ctx, record)
        cached = self._b64_by_index.get((i, *display))
        if cached is not None:
            return cached
        path = self._crop_path(i)
        if path is None:
            return None
        t0 = time.monotonic()
        b64 = file_to_full_base64(path, *display)
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._b64_by_index[(i, *display)] = b64
        # Logged once per crop (later hovers and clicks hit the cache
        # above), so that if full-resolution previews ever turn out to be
        # too slow it is visible here whether the cost is reading the file
        # or the size of what gets pushed to the browser.
        _log(f"projection crop: {path.name} -- {len(b64) / 1024:.0f} KB base64, read in {elapsed_ms:.0f} ms")
        return b64

    def _push_cursor(self) -> None:
        # The cursor layer holds a circle, a polyline and a rectangle, so
        # that is what a mouse move costs -- the cloud, the prototypes and
        # the selection all sit in canvases this never touches.
        if self.cursor_canvas.page:
            self.cursor_canvas.update()

    def _push_hover(self) -> None:
        if self.hover_card.page:
            self.hover_card.update()

    # ---- the three pieces the layout places -----------------------------

    def _legend_is_filterable(self) -> bool:
        """Whether every legend entry names a group some point actually
        belongs to. Only then can ticking one mean anything -- a legend
        built from something other than the points (a colorbar's ticks,
        say) has nothing for a checkbox to hide."""
        if self.point_groups is None or not self.legend:
            return False
        known = set(self.point_groups.tolist())
        return all(label in known for _, label in self.legend)

    def _set_group_visible(self, label, visible: bool) -> None:
        if visible:
            self._hidden_groups.discard(label)
        else:
            self._hidden_groups.add(label)
        self._redraw_shown()

    def _show_all_groups(self, visible: bool) -> None:
        self._hidden_groups = set() if visible else {label for _, label in self.legend}
        for box in self._legend_checkboxes:
            box.value = visible
            if box.page:
                box.update()
        self._redraw_shown()

    def _toggle_front(self, label) -> None:
        """Bring a group to the front, or send it back among the others.
        Brought again after another one, it goes back on top of it."""
        if label in self._front_groups:
            self._front_groups.remove(label)
        else:
            self._front_groups.append(label)
        for group, button in self._front_buttons.items():
            button.selected = group in self._front_groups
            if button.page:
                button.update()
        self._redraw_shown()

    def _redraw_shown(self) -> None:
        """Redraw the layers that depend on which classes are shown. Only
        those: hiding a class is a *display* filter, so it re-serializes
        the points that remain and touches nothing else -- no resampling,
        no reprojection, and the viewport does not move."""
        self._last_frame = None  # same reason as in _render_selection
        # A selection made before the toggle has to follow it: unticking a
        # class while its points are selected would otherwise leave them in
        # the count and in the carousel while they are off the map.
        self._reselect_inside_lasso()
        self.points_canvas.shapes = self._point_shapes()
        self._render_selection_shapes()
        self._hide_hover()
        if self.points_canvas.page:
            self.points_canvas.update()
            self.selection_canvas.update()
            self.cursor_canvas.update()
            self.hover_card.update()
            # ...and the count beside them, which _reselect_inside_lasso
            # above may have just changed.
            for control in (self.selection_text, self.show_selection_btn,
                            self.clear_selection_btn):
                if control.page:
                    control.update()

    def legend_control(self) -> ft.Control:
        """The legend, and -- when its entries name groups of points -- the
        display filter. Ticking one off hides its points in place; the class
        picker in the toolbar is the other filter, the one that re-samples
        and re-projects. Prototypes are never hidden by either: they are
        the landmarks one is trying to place the points against."""
        if not self.legend:
            return ft.Container(width=0, height=0)
        self._legend_checkboxes = []
        self._front_buttons = {}
        if not self._legend_is_filterable():
            body = [
                ft.Row([
                    ft.Container(width=10, height=10, bgcolor=color, border_radius=5),
                    ft.Text(str(label), size=11),
                ], spacing=6)
                for color, label in self.legend
            ]
            header = [ft.Text("Legend", size=12, weight=ft.FontWeight.BOLD)]
        else:
            body = []
            for color, label in self.legend:
                box = ft.Checkbox(
                    label=str(label), value=label not in self._hidden_groups, scale=0.85,
                    on_change=lambda e, l=label: self._set_group_visible(l, bool(e.control.value)),
                )
                self._legend_checkboxes.append(box)
                front = ft.IconButton(
                    icon=ft.Icons.FLIP_TO_FRONT_OUTLINED, selected_icon=ft.Icons.FLIP_TO_FRONT,
                    selected=label in self._front_groups, icon_size=16,
                    visual_density=ft.VisualDensity.COMPACT,
                    tooltip="Draw on top of the other points",
                    on_click=lambda e, l=label: self._toggle_front(l),
                )
                self._front_buttons[label] = front
                # The button first: labels differ in length, and a column
                # of buttons after them would not line up.
                body.append(ft.Row([
                    front,
                    ft.Container(width=10, height=10, bgcolor=color, border_radius=5),
                    box,
                ], spacing=6))
            header = [
                ft.Row([
                    ft.Text(self.legend_title, size=12, weight=ft.FontWeight.BOLD),
                    ft.TextButton("All", on_click=lambda e: self._show_all_groups(True)),
                    ft.TextButton("None", on_click=lambda e: self._show_all_groups(False)),
                ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ]
        return ft.Container(
            padding=12, border_radius=10,
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            content=ft.Column(header + [ft.Column(body, spacing=0 if self._legend_is_filterable() else 4)],
                              spacing=8, tight=True),
        )

    def toolbar_controls(self) -> list:
        """The map's own controls, handed over rather than drawn above the
        plot: the tab lays them out on one line with its renderer switch,
        because the map is height-bound on any 16:9 window and a second row
        of chrome above it comes straight out of the picture.

        A list rather than a Row, so the caller can put them in a single
        flat row with its own controls -- a nested row wraps as one block,
        which is what took a whole second line to begin with."""
        return [
            self.mode_switch, self.zoom_out_btn, self.zoom_text, self.zoom_in_btn,
            self.reset_view_btn, self.selection_text,
            self.show_selection_btn, self.clear_selection_btn,
        ]

    def map_control(self) -> ft.Control:
        """The plot itself -- the piece that must be placed outside any
        scrolling region, see the module docstring. Its controls come
        separately, from toolbar_controls()."""
        self.gesture_surface = ft.Container(
            width=self.size, height=self.size,
            # A gesture detector only receives pointer events over
            # something that paints, so it wraps a container filled with an
            # (invisible) color rather than nothing at all.
            bgcolor=ft.Colors.with_opacity(0.01, "#ffffff"),
        )
        gesture = ft.GestureDetector(
            hover_interval=HOVER_INTERVAL_MS,
            drag_interval=DRAG_INTERVAL_MS,
            mouse_cursor=ft.MouseCursor.PRECISE,
            on_hover=self._on_hover,
            on_exit=self._on_exit,
            on_tap_down=self._on_tap,
            on_pan_start=self._on_pan_start,
            on_pan_update=self._on_pan_update,
            on_pan_end=self._on_pan_end,
            on_scroll=self._on_scroll,
            content=self.gesture_surface,
        )
        self.plot_container = ft.Container(
            width=self.size, height=self.size,
            border_radius=8, border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            bgcolor=PLOT_BGCOLOR,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Stack([
                self.points_canvas, self.prototypes_canvas, self.selection_canvas,
                self.cursor_canvas, self.hover_card, gesture,
            ]),
        )
        return self.plot_container
