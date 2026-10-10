"""The Embeddings tab's display filters, and the switch that decides
whether a number is a scale or a set of labels.

What is being protected here is a separation. The legend has always been
both the key to the colors *and* the filter, which means filtering has
always been filtering on whatever column the color came from. Colour by
confidence, filter by sensor, and hide the blurry crops while you are at
it -- that is one color and three columns, and it needs the two jobs to
come apart.

The other half is that nothing here may cost a projection. Hiding points
changes which of them are drawn; it does not change where any of them are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core import point_coloring as pc
from lookin.ui.projection_view import CLASS_COLOR_KEY, ProjectionView


class FakePage:
    width, height = 1440, 900

    def run_thread(self, fn, *args):
        fn(*args)

    def open(self, *a):
        pass

    def close(self, *a):
        pass


class FakeCtx:
    def __init__(self, state):
        self.page = FakePage()
        self.state = state
        self.image_zoom = None
        self.prototype_source = type("NoPrototypes", (), {
            "active_set": lambda self, name: None,
            "resolve": lambda self, name: (None, None),
            "model_set_for_dataset": lambda self, name: None,
            "dataset_set": lambda self, name, embeddings, labels: (None, ""),
        })()


@pytest.fixture
def view(loaded_state, embeddings_npz) -> ProjectionView:
    loaded_state.experiments["exp"].load_embeddings(str(embeddings_npz))
    view = ProjectionView(FakeCtx(loaded_state))
    view.build()
    view.refresh()
    view.projection_method.value = "pca"
    view._on_run_projection(None)
    assert view._has_result
    return view


def _key_for(view, column: str) -> str:
    return next(key for key, value in view._color_column_map.items() if value == column)


def _shown(view) -> int:
    return len(view._interactive.points)


# ---- picking the columns to filter on ------------------------------------

def test_only_the_columns_asked_for_get_a_panel(view):
    """The point of choosing them: a panel per column of the joined frame
    would be taller than the map it filters."""
    assert view._filter_columns == []
    assert view.filters_card.content is None

    view._toggle_filter_column("site")
    assert view._filter_columns == ["site"]
    assert view.filters_card.content is not None
    assert len(view.filters_column.controls) == 1

    view._toggle_filter_column("site")
    assert view._filter_columns == [] and view.filters_card.content is None


def test_the_class_is_offered_as_a_filter_although_it_is_not_a_colour_entry(view):
    """Colouring by class has its own entry, so the label column is kept
    out of the colour list -- but hiding two classes to see what is behind
    them is exactly what a display filter is for."""
    offered = view._filterable_columns()
    assert offered[0] == "label"
    assert "correct" in offered and "site" in offered
    assert "uuid" not in offered and "crop" not in offered


def test_a_filter_on_a_column_that_went_away_goes_with_it(view):
    view._toggle_filter_column("site")
    view._colorable = [c for c in view._colorable if c != "site"]
    view._prune_filters()
    assert view._filter_columns == [] and "site" not in view._filters


# ---- filtering, and what it costs ----------------------------------------

def test_hiding_points_is_a_redraw_not_a_projection(view, monkeypatch):
    import lookin.ui.projection_view as projection_view

    calls = []
    real = projection_view.ea.fit_projection
    monkeypatch.setattr(projection_view.ea, "fit_projection",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    before = view._last_result.proj.copy()
    view._toggle_filter_column("site")
    view._filters["site"].hidden = {"north"}
    view._on_filter_changed()

    assert calls == [], "the coordinates do not depend on which points are drawn"
    assert np.array_equal(view._last_result.proj, before)
    assert _shown(view) < len(before)


def test_the_points_that_remain_are_where_they_always_were(view):
    """A filter must not re-fit the projection to what is left: the whole
    value of hiding half the cloud is comparing the half that stays
    against the picture you were already reading."""
    positions = {str(u): tuple(p) for u, p in zip(view._interactive.uuids, view._interactive.points)}
    view._toggle_filter_column("site")
    view._filters["site"].hidden = {"north"}
    view._on_filter_changed()

    for uuid, point in zip(view._interactive.uuids, view._interactive.points):
        assert tuple(point) == positions[str(uuid)]


def test_a_hidden_point_takes_its_row_with_it(view):
    """The map resolves a click by position, so the rows, the uuids and the
    marker shapes have to be filtered in step with the coordinates -- a
    click would otherwise open the crop of a point that is not there."""
    view._toggle_filter_column("site")
    view._filters["site"].hidden = {"north"}
    view._on_filter_changed()

    interactive = view._interactive
    assert len(interactive.records) == len(interactive.points) == len(interactive.uuids)
    for record in interactive.records:
        assert record["site"] != "north"


def test_two_columns_at_once_and_neither_is_the_colour(view):
    view.color_by.value = _key_for(view, "predicted_score")
    view._on_color_change(None)

    view._toggle_filter_column("site")
    view._toggle_filter_column("flag")
    view._filters["site"].hidden = {"north"}
    view._filters["flag"].hidden = {"1"}
    view._on_filter_changed()

    assert view._interactive.legend_title == "Show predicted_score"
    for record in view._interactive.records:
        assert record["site"] != "north" and str(record["flag"]) != "1"


def test_a_filter_outlives_the_colour_it_was_never_tied_to(view):
    """The legend's own hidden set cannot survive a recolour -- its group
    names mean nothing under the new colors. These are not group names,
    they are columns, and they stay."""
    view._toggle_filter_column("site")
    view._filters["site"].hidden = {"north"}
    view._on_filter_changed()
    hidden_count = _shown(view)

    view.color_by.value = _key_for(view, "correct")
    view._on_color_change(None)

    assert view._filters["site"].hidden == {"north"}
    assert _shown(view) == hidden_count


def test_the_legend_still_hides_its_own_groups(view):
    """Kept deliberately: it is the fastest gesture on the tab. It just is
    not the only one any more."""
    interactive = view._interactive
    assert interactive._legend_is_filterable()
    hidden = interactive.legend[0][1]
    interactive._hidden_groups = {hidden}
    assert int(interactive._shown_mask().sum()) == int(np.sum(interactive.point_groups != hidden))


def test_resetting_puts_every_point_back(view):
    total = _shown(view)
    view._toggle_filter_column("site")
    view._filters["site"].hidden = {"north"}
    view._on_filter_changed()
    assert _shown(view) < total

    view._reset_all_filters()
    assert _shown(view) == total


def test_the_caption_says_how_many_points_went(view):
    view._toggle_filter_column("site")
    view._filters["site"].hidden = {"north"}
    view._on_filter_changed()
    banners = view.projection_note.content.controls
    text = " ".join(b.content.controls[1].value for b in banners)
    assert "points shown" in text and "hidden by the display filters" in text


# ---- a number is a scale, or it is a set of labels ------------------------

def test_a_discrete_float_can_be_read_either_way(view):
    """Six distinct values in a float column: labels to one person, a
    coarse measurement to another, and nothing in the data says which."""
    column = "area"
    key = _key_for(view, column)
    view.color_by.value = key

    reference = view._column_reference(view._last_context, column)
    assert pc.can_be_continuous(reference)
    assert pc.default_kind(reference) == pc.KIND_CATEGORICAL, "few distinct values reads as labels"

    mapping, _ = view._color_mapping(view._last_result)
    assert mapping.kind == pc.KIND_CATEGORICAL

    view._set_column_kind(column, pc.KIND_NUMERIC)
    mapping, _ = view._color_mapping(view._last_result)
    assert mapping.kind == pc.KIND_NUMERIC
    # The ramp spans the column's own extremes, not the sample's.
    assert (mapping.vmin, mapping.vmax) == (float(reference.min()), float(reference.max()))


def test_the_switch_is_only_offered_where_there_is_a_choice(view):
    view.color_by.value = _key_for(view, "area")
    view._sync_color_scale_switch()
    assert view.color_scale_switch.visible
    assert view.color_scale_switch.selected == {pc.KIND_CATEGORICAL}

    view.color_by.value = _key_for(view, "site")
    view._sync_color_scale_switch()
    assert not view.color_scale_switch.visible, "there is no ramp between two place names"

    view.color_by.value = CLASS_COLOR_KEY
    view._sync_color_scale_switch()
    assert not view.color_scale_switch.visible


def test_the_choice_follows_the_column_into_its_filter(view):
    """Set once, wherever it is set: the coloring and the filter read the
    same column the same way, or the legend and the panel would disagree
    about what the values even are."""
    view._toggle_filter_column("area")
    assert view._filters["area"].kind == pc.KIND_CATEGORICAL
    assert view._filters["area"].values, "a box per value"

    view._set_column_kind("area", pc.KIND_NUMERIC)
    assert view._filters["area"].kind == pc.KIND_NUMERIC
    assert view._filters["area"].bounds[0] < view._filters["area"].bounds[1], "a window over a range"

    view._set_column_kind("area", pc.KIND_CATEGORICAL)
    assert view._filters["area"].kind == pc.KIND_CATEGORICAL


def test_a_window_over_a_float_column_hides_what_is_outside_it(view):
    view._toggle_filter_column("area")
    view._set_column_kind("area", pc.KIND_NUMERIC)
    column_filter = view._filters["area"]
    low, high = column_filter.bounds
    column_filter.window = (low, low + (high - low) / 2)
    view._on_filter_changed()

    for record in view._interactive.records:
        assert record["area"] <= column_filter.window[1] + 1e-9
