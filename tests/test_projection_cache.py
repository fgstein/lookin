"""The Embeddings tab keeps one computed projection per experiment.

Comparing two runs means going back and forth between them, and the
projection is the expensive thing in this tab -- tens of seconds of t-SNE
or UMAP. Recomputing it on the way back made that comparison cost one
projection per switch, so a completed run is kept per experiment and put
back on screen when its experiment is selected again. Only the button
computes.

What that buys has to be paid for honestly: a kept run must never be shown
against data it was not computed from, and the controls above the map must
describe the map that is actually on screen.

Same fakes as test_projection_coloring: real controls, nothing mounted, and
page.run_thread runs its worker inline.
"""
from __future__ import annotations

import numpy as np
import pytest

from lookin.ui.projection_view import ProjectionView


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
def two_experiments(loaded_state, results_csv, embeddings_npz):
    """Two experiments, both with the same sidecar attached -- the case the
    cache exists for."""
    loaded_state.add_experiment(str(results_csv), "other")
    for name in ("exp", "other"):
        loaded_state.experiments[name].load_embeddings(str(embeddings_npz))
    return loaded_state


@pytest.fixture
def view(two_experiments) -> ProjectionView:
    view = ProjectionView(FakeCtx(two_experiments))
    view.build()
    view.refresh()
    return view


@pytest.fixture
def counted(view, monkeypatch):
    """The view, and one entry per actual projection computed."""
    import lookin.ui.projection_view as projection_view

    calls = []
    real = projection_view.ea.fit_projection

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(projection_view.ea, "fit_projection", counting)
    return view, calls


def _select(view, name: str) -> None:
    """What picking an experiment in the dropdown does."""
    view.experiment_dropdown.value = name
    view._on_experiment_change(None)


# ---- going back to an experiment ---------------------------------------

def test_coming_back_to_an_experiment_redraws_instead_of_projecting(counted):
    view, calls = counted
    _select(view, "exp")
    view._on_run_projection(None)
    _select(view, "other")
    view._on_run_projection(None)
    assert len(calls) == 2

    _select(view, "exp")
    assert view._has_result, "the map that was computed for 'exp' is back on screen"
    assert view._last_result.exp_name == "exp"
    assert len(calls) == 2, "coming back is a redraw, not another projection"


def test_leaving_an_experiment_that_was_never_run_still_prompts(counted):
    """Nothing is cached for it, and nothing computes on its own -- the
    button is the only thing that projects."""
    view, calls = counted
    _select(view, "exp")
    view._on_run_projection(None)
    _select(view, "other")
    assert not view._has_result
    assert len(calls) == 1


def test_the_controls_describe_the_map_that_came_back(counted):
    """A t-SNE of one class under a dropdown reading "UMAP" with no chips
    would be a lie, and the next run would be of something else again."""
    view, _ = counted
    _select(view, "exp")
    view.projection_method.value = "pca"
    view.projection_sample_field.value = "3"
    view._set_class_filter(["cat"])
    view._on_run_projection(None)

    _select(view, "other")
    view.projection_method.value = "tsne"
    view.projection_sample_field.value = "1500"

    _select(view, "exp")
    assert view.projection_method.value == "pca"
    assert view.projection_sample_field.value == "3"
    assert view._class_filter == ["cat"]
    assert [chip.label.value for chip in view.class_chips.controls] == ["cat"]


def test_the_button_still_computes(counted):
    """The cache is about switching, not about refusing to work: an
    explicit click projects again even when a run is held for it."""
    view, calls = counted
    _select(view, "exp")
    view._on_run_projection(None)
    view._on_run_projection(None)
    assert len(calls) == 2


def test_a_run_that_lands_after_a_switch_is_kept_rather_than_dropped(counted):
    """t-SNE finishing just after the user moved on is still a projection
    that was paid for. It must not take the screen from whatever is on it,
    but throwing it away is what this whole cache exists to stop."""
    view, calls = counted
    _select(view, "exp")
    stale = view._projection_generation  # a generation already superseded
    view._projection_generation += 1
    view._run_projection(view._analysis_context("exp"), "exp", "pca", 1500,
                         stale, [], view._runs_epoch)
    assert "exp" in view._runs
    assert not view._has_result, "a superseded run does not grab the screen"
    assert view._last_result is None


# ---- when a kept run stops being true -----------------------------------

def test_new_embeddings_drop_every_kept_run(counted):
    """Not only the one on screen: the change is to the data they were all
    computed from, and an experiment that is not being looked at right now
    is no more immune to it."""
    view, calls = counted
    _select(view, "exp")
    view._on_run_projection(None)
    _select(view, "other")
    view._on_run_projection(None)

    view.on_prototype_source_changed()
    assert view._runs == {}
    assert not view._has_result

    _select(view, "exp")
    assert not view._has_result
    assert len(calls) == 2, "nothing recomputes by itself either"


def test_a_run_in_flight_cannot_refill_the_cache_it_was_cleared_from(view):
    """It was computed from the embeddings that just went away, so landing
    late must not put it back."""
    _select(view, "exp")
    ctx = view._analysis_context("exp")
    epoch = view._runs_epoch
    view._clear_runs()
    view._run_projection(ctx, "exp", "pca", 1500, view._projection_generation, [], epoch)
    assert view._runs == {}


def test_rows_moving_under_a_kept_run_makes_it_a_miss(view):
    """`idx` are positions in the context the run was computed from. Rows
    added, removed or reordered leave those positions pointing at other
    crops -- the map would be drawn from the right coordinates and the
    wrong points."""
    _select(view, "exp")
    view._on_run_projection(None)
    cached = view._runs["exp"]
    ctx = view._analysis_context("exp")

    assert view._rebind(cached, ctx) is not None
    shuffled = dict(ctx, uuids=ctx["uuids"][::-1])
    assert view._rebind(cached, shuffled) is None
    shorter = dict(ctx, uuids=ctx["uuids"][:1], labels=ctx["labels"][:1])
    assert view._rebind(cached, shorter) is None


def test_a_class_the_filter_named_going_away_makes_it_a_miss(view):
    """The caption reads "N of M classes"; a filter naming a class the data
    no longer has cannot be described by it."""
    _select(view, "exp")
    view._set_class_filter(["cat"])
    view._on_run_projection(None)
    ctx = view._analysis_context("exp")
    relabelled = dict(ctx, labels=np.array(["dog"] * len(ctx["labels"])))
    assert view._rebind(view._runs["exp"], relabelled) is None


def test_a_kept_run_is_redrawn_with_the_labels_of_today(view):
    """Only the coordinates are worth keeping. Everything the labels decide
    -- the colors, the legend, the counts -- is taken from the context
    built when the run comes back, so a crop relabelled in the carousel
    since then is drawn in its new class."""
    _select(view, "exp")
    view._on_run_projection(None)
    ctx = view._analysis_context("exp")
    relabelled = dict(ctx, labels=np.array(["fish"] * len(ctx["labels"])))
    restored = view._rebind(view._runs["exp"], relabelled)
    assert restored.all_classes == ["fish"]
    assert set(restored.labels.tolist()) == {"fish"}


def test_an_experiment_that_goes_away_takes_its_run_with_it(view):
    """A name freed by an unload can come back attached to different data;
    a run kept under it would then be restored against a stranger."""
    _select(view, "exp")
    view._on_run_projection(None)
    del view.ctx.state.experiments["exp"]
    view.refresh()
    assert "exp" not in view._runs


def test_a_kept_run_follows_a_rename(view):
    """The projection is still valid -- only the name it is filed under
    changed (see rename_experiment)."""
    _select(view, "exp")
    view._on_run_projection(None)
    view.ctx.state.rename_experiment("exp", "renamed")
    view.rename_experiment("exp", "renamed")
    assert "exp" not in view._runs
    assert view._runs["renamed"].exp_name == "renamed"
