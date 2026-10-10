"""How the interactive projection turns viewport changes into frames.

Not a UI test: everything here is the arithmetic and the pacing behind the
map, which is what decides whether dragging a 50,000-point cloud is usable.
The failure modes are all silent -- a drag that lags seconds behind the
cursor, a thinned "draft" cloud left on screen after the gesture ended, a
selection halo drawn around the wrong points -- so none of them would show
up as an error anywhere.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from lookin.ui.interactive_projection import (
    DRAFT_MAX_POINTS, InteractiveProjection, _RedrawPump,
)

PALETTE = ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]


def _projection(n: int = 500, size: int = 700) -> InteractiveProjection:
    """A projection with only the parts a frame needs, so no page, no
    canvases and no Flet at all -- the same trick as test_carousel_layout."""
    rng = np.random.default_rng(0)
    view = InteractiveProjection.__new__(InteractiveProjection)
    view.points = rng.normal(size=(n, 2))
    colors = [PALETTE[i] for i in rng.integers(0, len(PALETTE), n)]
    view.point_colors = colors
    view._color_array = np.asarray(colors)
    view.point_groups = np.asarray([f"class_{i}" for i in rng.integers(0, len(PALETTE), n)])
    view.prototypes = None
    view.point_kinds = None   # one pool on the map: every point is a disc
    view.size = size
    view._hidden_groups = set()
    view._front_groups = []
    view._selected = np.array([], dtype=int)
    view._last_frame = None
    view._fit()
    view.sx, view.sy = view._to_screen(view.points[:, 0], view.points[:, 1])
    return view


# ---- thinning a frame ---------------------------------------------------

def test_a_draft_is_capped_and_a_small_cloud_is_untouched():
    view = _projection(n=DRAFT_MAX_POINTS * 3)
    visible = view._visible()
    assert len(view._draft_sample(visible)) <= DRAFT_MAX_POINTS
    small = visible[:100]
    assert np.array_equal(view._draft_sample(small), small)


def test_the_same_points_are_dropped_every_frame():
    """A different random subset per frame would make the cloud shimmer for
    the whole length of a drag."""
    view = _projection(n=DRAFT_MAX_POINTS * 3)
    visible = view._visible()
    assert np.array_equal(view._draft_sample(visible), view._draft_sample(visible))


def test_a_draft_frame_is_the_cheap_one():
    """The point of the draft: what goes over the wire is bounded, however
    many points the map holds."""
    view = _projection(n=DRAFT_MAX_POINTS * 3)
    drafted = sum(len(shape.points) for shape in view._point_shapes(draft=True))
    full = sum(len(shape.points) for shape in view._point_shapes(draft=False))
    assert drafted <= DRAFT_MAX_POINTS < full


# ---- grouping and projecting -------------------------------------------

def test_points_are_grouped_by_the_color_they_are_drawn_in():
    view = _projection()
    indices = view._visible()
    grouped = view._indices_by_style(indices)
    assert sum(len(idx) for idx in grouped.values()) == len(indices)
    for (color, _kind), idx in grouped.items():
        assert {view.point_colors[i] for i in idx} == {color}


def test_a_second_marker_shape_splits_the_shapes_it_needs_to_and_no_more():
    """Overlaid points are drawn as squares rather than discs, and a Paint
    holds one shape -- so a color shared by both sources becomes two
    shapes, not one, and no point is dropped or drawn twice."""
    from lookin.ui.interactive_projection import POINT_KIND_DISC, POINT_KIND_SQUARE

    view = _projection(n=40)
    kinds = np.array([POINT_KIND_DISC] * 30 + [POINT_KIND_SQUARE] * 10)
    view.point_kinds = kinds
    indices = view._visible()
    grouped = view._indices_by_style(indices)

    assert sum(len(idx) for idx in grouped.values()) == len(indices)
    for (color, kind), idx in grouped.items():
        assert {view.point_colors[i] for i in idx} == {color}
        assert set(kinds[idx]) == {kind}
    assert {kind for _color, kind in grouped}.issubset({POINT_KIND_DISC, POINT_KIND_SQUARE})


def test_offsets_are_the_screen_coordinates_rounded():
    view = _projection()
    idx = np.array([3, 17, 40])
    assert view._offsets(idx) == [
        [round(float(view.sx[i]), 1), round(float(view.sy[i]), 1)] for i in idx
    ]


# ---- draw order ---------------------------------------------------------

def _drawn_groups(view, shapes) -> list:
    """The group of every point, in the order the canvas paints them."""
    by_offset = {tuple(o): view.point_groups[i]
                 for i, o in enumerate(view._offsets(np.arange(len(view.sx))))}
    return [by_offset[tuple(o)] for shape in shapes for o in shape.points]


def test_a_group_brought_to_the_front_is_painted_last():
    """A canvas paints its shapes in list order, so "on top" means "last"
    -- and every point of that group, not just the ones sharing a color."""
    view = _projection()
    view._front_groups = ["class_0"]
    order = _drawn_groups(view, view._point_shapes())
    n = int(np.sum(view.point_groups == "class_0"))
    assert set(order[-n:]) == {"class_0"}
    assert "class_0" not in order[:-n]


def test_the_last_group_brought_forward_is_the_top_one():
    view = _projection()
    view._front_groups = ["class_0", "class_1"]
    order = _drawn_groups(view, view._point_shapes())
    n1 = int(np.sum(view.point_groups == "class_1"))
    assert set(order[-n1:]) == {"class_1"}
    assert order.index("class_0") > max(i for i, g in enumerate(order) if g in ("class_2", "class_3"))


def test_layering_neither_drops_nor_duplicates_a_point():
    view = _projection()
    plain = sum(len(s.points) for s in view._point_shapes())
    view._front_groups = ["class_2", "class_0"]
    assert sum(len(s.points) for s in view._point_shapes()) == plain


def test_a_hover_over_overlapping_points_finds_the_one_on_top():
    view = _projection(n=2)
    view.points = np.zeros((2, 2))
    view.point_groups = np.array(["under", "over"])
    view.sx, view.sy = np.array([100.0, 101.0]), np.array([100.0, 100.0])
    assert view._pick(100.0, 100.0) == 0
    view._front_groups = ["over"]
    assert view._pick(100.0, 100.0) == 1


def test_changing_the_order_changes_the_frame():
    """Otherwise the redraw it asks for would be skipped as a repeat."""
    view = _projection()
    before = view._frame_signature(draft=False)
    view._front_groups = ["class_1"]
    assert view._frame_signature(draft=False) != before


# ---- the selection layer ------------------------------------------------

class _FakeCanvas:
    shapes: list = []


def test_nothing_selected_draws_no_selection_layer():
    view = _projection()
    view.selection_canvas = _FakeCanvas()
    view._render_selection_shapes()
    assert view.selection_canvas.shapes == []


def test_the_halo_covers_exactly_the_selected_points_on_screen():
    view = _projection()
    view.selection_canvas = _FakeCanvas()
    view._selected = np.array([1, 2, 3, 4, 5])
    view._render_selection_shapes()
    # The first shape is the dark halo under the whole selection; the rest
    # redraw those same points in their own colors.
    halo, *by_color = view.selection_canvas.shapes
    assert len(halo.points) == 5
    assert sum(len(shape.points) for shape in by_color) == 5


def test_a_hidden_group_is_not_drawn_as_selected():
    """Hiding a legend group hides its points, selected or not -- a halo
    left hanging in empty space would point at nothing."""
    view = _projection()
    view.selection_canvas = _FakeCanvas()
    view._selected = np.arange(20)
    hidden = view.point_groups[0]
    view._hidden_groups = {hidden}
    view._render_selection_shapes()
    still_shown = int(np.sum(view.point_groups[:20] != hidden))
    if still_shown:
        assert len(view.selection_canvas.shapes[0].points) == still_shown
    else:
        assert view.selection_canvas.shapes == []


# ---- pacing the frames --------------------------------------------------

SLOW = 0.05  # one "frame", long enough for requests to pile up behind it


def _pump(drawn: list, cost: float = SLOW) -> _RedrawPump:
    def draw(draft: bool):
        time.sleep(cost)
        drawn.append("draft" if draft else "full")
    return _RedrawPump(draw)


def _wait_for_settle(drawn: list, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if drawn and drawn[-1] == "full":
            return
        time.sleep(0.01)
    pytest.fail(f"no full-detail frame ever landed (drew {drawn})")


def test_a_burst_of_requests_does_not_become_a_burst_of_frames():
    """The whole point: a drag sends a request every 40ms whatever the
    frame costs, and without coalescing those queue up and the map falls
    further behind the cursor the longer the drag lasts."""
    drawn = []
    pump = _pump(drawn)
    for _ in range(20):
        pump.request()
    _wait_for_settle(drawn)
    assert len(drawn) < 20, f"every request became its own frame ({drawn})"


def test_the_last_frame_is_always_full_detail():
    """A gesture that ends on a draft would leave the thinned cloud on
    screen, which is a wrong picture, not a slow one."""
    drawn = []
    pump = _pump(drawn)
    pump.request()
    _wait_for_settle(drawn)
    assert drawn[-1] == "full"


def test_the_drawing_thread_gives_up_when_there_is_nothing_to_draw():
    """A projection the user has moved on from must not leave a thread
    behind -- there is one of these per run of the projection."""
    drawn = []
    pump = _pump(drawn, cost=0.0)
    pump.request()
    _wait_for_settle(drawn)
    # It is still alive (idling), but it is a daemon and it is one thread.
    assert threading.active_count() < 50


def test_drawing_the_picture_already_on_screen_is_skipped():
    """A pan at the fitted zoom is clamped to no movement, and a gesture
    ends with a full-detail request that the settle timer then repeats:
    both would otherwise re-serialize the whole cloud for nothing."""
    view = _projection()
    view._last_frame = view._frame_signature(draft=False)
    drew = []
    view._to_screen = lambda x, y: drew.append(1) or (view.sx, view.sy)
    view._redraw(draft=False)
    assert drew == [], "an identical frame was drawn again"
