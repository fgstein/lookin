"""The Embeddings tab's "Color by" control: what it offers, and what
changing it is allowed to cost.

The expensive half of this tab is the projection -- tens of seconds of
t-SNE or UMAP -- and the coordinates do not depend on the color at all. So
the one thing this must never do is re-run: a user cycling through six
columns to find the one that explains a cluster would otherwise be paying
six projections for six redraws. The other trap is the dropdown's keys:
column names come from the user's CSV, and one called "class" or "0" must
not be able to collide with something code compares against.

No Flet session: the controls are real, nothing is mounted, and page.run_thread
runs its worker inline so a "background" run is finished by the time the
call returns.
"""
from __future__ import annotations

import numpy as np
import pytest

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
    return view


def _columns(view) -> list:
    return [column for column in view._color_column_map.values() if column is not None]


def _key_for(view, column: str) -> str:
    return next(key for key, value in view._color_column_map.items() if value == column)


# ---- what is on offer ---------------------------------------------------

def test_the_join_is_what_is_offered_not_just_the_dataset(view):
    """The point of colouring by column is that the *joined* frame holds
    more than the CSV did: whether the model got it right, and how sure it
    was, are columns the merge invents."""
    offered = _columns(view)
    for column in ("predicted_label", "predicted_score", "correct", "site", "area", "flag"):
        assert column in offered
    assert offered[:3] == ["predicted_label", "predicted_score", "correct"]


def test_what_a_color_would_waste_itself_on_is_not_offered(view):
    """The uuid is one color per row, the crop path likewise, and the label
    column is the "Class" entry -- offering it twice is not a second
    coloring."""
    offered = _columns(view)
    for column in ("uuid", "crop", "label"):
        assert column not in offered


def test_the_options_are_keyed_positionally(view):
    """A column named "class" or "0" comes from the user's CSV and must not
    be able to collide with a key code compares against -- so the keys are
    positions and the names live in the map beside them."""
    keys = [option.key for option in view.color_by.options]
    assert keys == [str(i) for i in range(len(keys))]
    assert view.color_by.options[0].key == CLASS_COLOR_KEY
    labels = [option.text for option in view.color_by.options[1:]]
    assert labels == _columns(view), "the column name is the label, never the key"


def test_the_pick_follows_the_column_rather_than_its_position(view):
    """Another experiment offers another set of columns. Keeping the index
    across a rebuild would silently start coloring by a different one."""
    view.color_by.value = _key_for(view, "site")
    view._rebuild_color_options(view._analysis_context("exp"))
    assert view._color_column_map[view.color_by.value] == "site"


def test_a_pick_that_the_next_experiment_has_no_column_for_falls_back(view):
    view.color_by.value = _key_for(view, "site")
    ctx = view._analysis_context("exp")
    ctx["merged"] = ctx["merged"].drop(columns=["site"])
    view._color_options_key = None  # the frame changed under the cache
    view._rebuild_color_options(ctx)
    assert view.color_by.value == CLASS_COLOR_KEY


# ---- what changing it costs ---------------------------------------------

@pytest.fixture
def projected(view, monkeypatch):
    """A tab with a projection on screen, and a count of how many times the
    projection itself was computed."""
    import lookin.ui.projection_view as projection_view

    calls = []
    real = projection_view.ea.fit_projection

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(projection_view.ea, "fit_projection", counted)
    view._on_run_projection(None)
    assert view._has_result and len(calls) == 1
    return view, calls


def test_recoloring_redraws_without_projecting_again(projected):
    view, calls = projected
    view.color_by.value = _key_for(view, "correct")
    view._on_color_change(None)
    assert len(calls) == 1, "the coordinates do not depend on the color"
    assert view._has_result
    assert view._interactive.legend_title == "Show correct"


def test_the_two_renderings_are_recolored_together(projected):
    """They are one plot drawn twice, and the switch between them is a
    visibility toggle -- a recolor that reached only the canvas would leave
    a stale PNG one click away."""
    view, _ = projected
    view.color_by.value = _key_for(view, "site")
    view._on_color_change(None)
    header, interactive_box, static_box = view.projection_results_container.content.controls
    caption = static_box.content.controls[0]
    assert "site" in caption.value
    assert view._interactive.legend_title == "Show site"


def test_a_filter_from_the_old_coloring_does_not_survive_the_new_one(projected):
    """The old group names mean nothing under the new colors, so a filter
    carried across would hide a slice of the cloud that the legend no
    longer explains."""
    view, _ = projected
    view._interactive._hidden_groups = {view._interactive.legend[0][1]}
    view.color_by.value = _key_for(view, "correct")
    view._on_color_change(None)
    assert view._interactive._hidden_groups == set()
    assert int(view._interactive._shown_mask().sum()) == len(view._interactive.points)


def test_the_canvas_filters_on_the_names_its_legend_shows(projected):
    """Ticking a legend entry off hides those points in place. That only
    works while every entry names a group some point is actually in."""
    view, _ = projected
    view.color_by.value = _key_for(view, "site")
    view._on_color_change(None)
    interactive = view._interactive
    assert interactive._legend_is_filterable()
    hidden = interactive.legend[0][1]
    interactive._hidden_groups = {hidden}
    expected = int(np.sum(interactive.point_groups != hidden))
    assert int(interactive._shown_mask().sum()) == expected


def test_a_recolor_while_a_run_is_computing_is_left_to_the_run(projected):
    """The run reads the dropdown when it renders, so it comes out in the
    new colors by itself -- and bumping the generation here would discard
    the projection it is halfway through instead."""
    view, _ = projected
    view._computing = True
    generation = view._projection_generation
    view.color_by.value = _key_for(view, "correct")
    view._on_color_change(None)
    assert view._projection_generation == generation


def test_dropping_the_result_drops_what_a_recolor_would_redraw(projected):
    """A recolor renders from the run it kept. Keeping that past a reset
    would let it put a map back up that the tab has just decided no longer
    matches its data."""
    view, calls = projected
    view._reset_result()
    assert view._last_result is None
    view.color_by.value = _key_for(view, "correct")
    view._on_color_change(None)
    assert not view._has_result
    assert len(calls) == 1
