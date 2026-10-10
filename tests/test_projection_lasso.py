"""What a lasso on the Embeddings map is allowed to catch.

The map has two filters on it at once: the legend's checkboxes, which hide
a class in place, and the lasso, which picks points out of what is drawn.
The bug this guards against is the two disagreeing -- a loop drawn around a
cluster returning the crops of a class the user had just taken off the map,
which is the one answer they had explicitly asked not to see.

Built like test_projection_frames: the object without its Flet half, since
none of this is about widgets.
"""
from __future__ import annotations

import numpy as np
import pytest

from lookin.ui.interactive_projection import InteractiveProjection

PALETTE = ["#4c78a8", "#f58518"]


class FakeShape:
    def __init__(self):
        self.points, self.visible = [], False


def _projection(size: int = 400) -> InteractiveProjection:
    """Twenty points in two classes, all of them inside the loop drawn
    below -- so what is selected is decided by the hidden classes alone."""
    view = InteractiveProjection.__new__(InteractiveProjection)
    groups = np.array(["cat"] * 10 + ["dog"] * 10)
    view.points = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    view.point_colors = [PALETTE[0 if g == "cat" else 1] for g in groups]
    view._color_array = np.asarray(view.point_colors)
    view.point_groups = groups
    view.legend = [(PALETTE[0], "cat"), (PALETTE[1], "dog")]
    view.prototypes = None
    view.point_kinds = None
    view.size = size
    view._hidden_groups = set()
    view._front_groups = []
    view._selected = np.array([], dtype=int)
    view._lasso_data = None
    view._lasso = []
    view._last_frame = None
    view.lasso_shape = FakeShape()
    view._fit()
    view.sx, view.sy = view._to_screen(view.points[:, 0], view.points[:, 1])

    # The widget half, replaced by nothing: every one of these only pushes
    # what the assertions read straight off the object.
    view._render_selection = lambda: None
    view._render_selection_shapes = lambda draft=False: None
    view._set_selection_text = lambda: None
    view._push_cursor = lambda: None
    view._hide_hover = lambda: None
    view._point_shapes = lambda: []
    view.points_canvas = type("NoPage", (), {"page": None, "shapes": []})()

    view.emitted = []
    view.on_select = view.emitted.append
    return view


def _loop_around_everything(view) -> list:
    return [(0.0, 0.0), (float(view.size), 0.0),
            (float(view.size), float(view.size)), (0.0, float(view.size))]


def test_a_lasso_over_everything_selects_everything():
    view = _projection()
    view._lasso = _loop_around_everything(view)
    view._finish_lasso()
    assert len(view._selected) == 20


def test_a_lasso_skips_the_points_of_a_hidden_class():
    """The whole point: an unticked legend entry is the user saying 'not
    these', and a lasso that swept them up anyway would open crops of a
    class they had just taken off the map."""
    view = _projection()
    view._hidden_groups = {"dog"}
    view._lasso = _loop_around_everything(view)
    view._finish_lasso()
    assert len(view._selected) == 10
    assert set(view.point_groups[view._selected]) == {"cat"}


def test_what_is_emitted_is_what_was_selected():
    view = _projection()
    view._hidden_groups = {"dog"}
    view._lasso = _loop_around_everything(view)
    view._finish_lasso()
    assert view.emitted == [[int(i) for i in view._selected]]


def test_hiding_a_class_after_the_lasso_drops_its_points():
    view = _projection()
    view._lasso = _loop_around_everything(view)
    view._finish_lasso()
    view._set_group_visible("dog", False)
    assert set(view.point_groups[view._selected]) == {"cat"}


def test_showing_it_again_brings_them_back():
    """The loop is what the user drew and it has not moved, so the
    selection is re-derived from it rather than pruned -- otherwise
    unticking a class would quietly destroy half a selection."""
    view = _projection()
    view._lasso = _loop_around_everything(view)
    view._finish_lasso()
    view._set_group_visible("dog", False)
    view._set_group_visible("dog", True)
    assert len(view._selected) == 20


def test_a_toggle_with_no_lasso_selects_nothing():
    view = _projection()
    view._set_group_visible("dog", False)
    assert len(view._selected) == 0
