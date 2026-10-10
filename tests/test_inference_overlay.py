"""Simple inferences: crops that are not in the dataset, drawn on the same
map as the ones that are.

Two things are being protected here, and they pull in opposite directions.

A pool with no ground truth must never reach a number. These crops carry no
label -- nothing about them can be right or wrong -- so the moment one of
them is counted in an accuracy, that accuracy is a lie. The defence is
structural rather than a flag: they live in their own slot, and
AppState.merged (which every metric is computed from) does not know it
exists.

And the map must hold still. The whole point of overlaying a batch of new
images is to see where they fall in a picture the user has already read;
re-projecting everything would rearrange the cloud underneath them and
answer a different question. That is what the kept fit is for, and what
"refit" deliberately gives up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.data_manager import DATASET_SOURCE_GROUP
from lookin.core.embedding_analysis import (
    OutOfSampleUnsupported, fit_projection, nearest_to_vector,
)
from lookin.ui.projection_view import (
    CLASS_COLOR_KEY, SOURCE_COLOR_KEY, ProjectionView, overlay_indices,
)


# ---- fixtures -----------------------------------------------------------

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
def inference_files(tmp_path):
    """A three-crop inference: its own ids, its own crop column name, one
    column the dataset also has ('site') and one only it has ('sensor')."""
    csv = tmp_path / "probe.csv"
    pd.DataFrame({
        "uuid": ["p1", "p2", "p3"],
        "img": ["a.png", "b.png", "sub/c.png"],
        "sensor": ["alpha", "beta", "alpha"],
        "site": ["north", "north", "west"],
    }).to_csv(csv, index=False)
    npz = tmp_path / "probe_embeddings.npz"
    np.savez(npz, uuid=np.array(["p1", "p2", "p3"], dtype="<U8"),
             embeddings=np.eye(3, 3, dtype=np.float32))
    return csv, npz


@pytest.fixture
def state_with_inference(loaded_state, embeddings_npz, inference_files):
    csv, npz = inference_files
    loaded_state.experiments["exp"].load_embeddings(str(embeddings_npz))
    loaded_state.load_inference("probe", str(csv), str(npz), uuid_col="uuid", crop_col="img")
    return loaded_state


@pytest.fixture
def view(state_with_inference) -> ProjectionView:
    view = ProjectionView(FakeCtx(state_with_inference))
    view.build()
    view.refresh()
    view.projection_method.value = "pca"
    return view


def _run(view, overlay: bool = False, refit: bool = False) -> None:
    if overlay:
        view._on_overlay_toggle("probe", True)
    view.refit_check.value = refit
    view._refit_choice = refit
    view._on_run_projection(None)


# ---- loading ------------------------------------------------------------

def test_rows_and_embeddings_are_aligned_once_and_for_all(state_with_inference):
    inference = state_with_inference.inferences["probe"]
    assert len(inference.df) == len(inference.embeddings) == 3
    assert list(inference.uuids()) == ["p1", "p2", "p3"]
    # Row i is embedding i, so nothing downstream has to re-align anything.
    for i, uuid in enumerate(inference.uuids()):
        assert np.array_equal(inference.vector_for(uuid), inference.embeddings[i])


def test_a_row_with_no_embedding_is_dropped_rather_than_drawn_nowhere(tmp_path, loaded_state):
    csv = tmp_path / "half.csv"
    pd.DataFrame({"uuid": ["p1", "p2"], "img": ["a.png", "b.png"]}).to_csv(csv, index=False)
    npz = tmp_path / "half_embeddings.npz"
    np.savez(npz, uuid=np.array(["p1"], dtype="<U8"), embeddings=np.eye(1, 3, dtype=np.float32))

    inference = loaded_state.load_inference("half", str(csv), str(npz), uuid_col="uuid", crop_col="img")
    assert list(inference.uuids()) == ["p1"]
    # ...and the counts are kept so the Data tab can say what happened.
    assert (inference.csv_rows, inference.npz_rows) == (2, 1)


def test_two_files_that_do_not_match_are_refused_with_a_reason(tmp_path, loaded_state):
    csv = tmp_path / "other.csv"
    pd.DataFrame({"uuid": ["x1"], "img": ["a.png"]}).to_csv(csv, index=False)
    npz = tmp_path / "other_embeddings.npz"
    np.savez(npz, uuid=np.array(["p9"], dtype="<U8"), embeddings=np.eye(1, 3, dtype=np.float32))

    with pytest.raises(ValueError, match="same run"):
        loaded_state.load_inference("other", str(csv), str(npz), uuid_col="uuid", crop_col="img")


def test_the_name_the_dataset_answers_to_is_reserved(loaded_state, inference_files):
    """Sources are told apart by name in the legend, so two of them cannot
    share one."""
    csv, npz = inference_files
    with pytest.raises(ValueError, match="reserved"):
        loaded_state.load_inference(DATASET_SOURCE_GROUP, str(csv), str(npz),
                                    uuid_col="uuid", crop_col="img")


def test_crop_paths_come_out_absolute(state_with_inference, tmp_path):
    """Which is what makes these crops displayable by widgets that resolve
    every path through the dataset: an absolute path is returned untouched,
    so an inference crop needs no second image root anywhere."""
    inference = state_with_inference.inferences["probe"]
    paths = list(inference.df["img"])
    assert all(p.startswith(str(tmp_path)) for p in paths)
    assert paths[2].endswith("sub/c.png")
    resolved = state_with_inference.dataset.resolve_crop_path(paths[0])
    assert str(resolved) == paths[0]


def test_rows_are_handed_over_keyed_the_way_the_crop_widgets_read_them(state_with_inference):
    inference = state_with_inference.inferences["probe"]
    records = inference.display_records("uuid", "crop")
    assert records[0]["uuid"] == "p1"
    assert records[0]["crop"].endswith("a.png")
    assert "img" not in records[0], "the path must not also stay under its own column name"
    assert records[0]["sensor"] == "alpha"
    # A subset, for a map that only drew some of them.
    assert [r["uuid"] for r in inference.display_records("uuid", "crop", rows=[2])] == ["p3"]


def test_nothing_that_computes_a_metric_can_see_them(state_with_inference):
    """The defence that matters: not a flag callers must remember to test,
    but a frame they are not in."""
    merged = state_with_inference.merged("exp")
    assert "sensor" not in merged.columns
    assert not set(merged["uuid"]) & {"p1", "p2", "p3"}
    assert len(merged) == 6  # the dataset's own six rows, and only those


# ---- placing points on an existing fit ----------------------------------

def test_a_fit_can_place_points_it_never_saw(loaded_state):
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(40, 6)).astype(np.float32)
    fit = fit_projection(pool, method="pca")
    placed = fit.transform(pool[:3])
    # Exactly where those rows landed in the fit itself: PCA is a linear
    # map, so placing a point twice cannot give two answers.
    assert np.allclose(placed, fit.points[:3], atol=1e-5)


def test_tsne_says_so_instead_of_guessing(loaded_state):
    rng = np.random.default_rng(0)
    fit = fit_projection(rng.normal(size=(30, 6)).astype(np.float32), method="tsne")
    assert not fit.supports_transform()
    with pytest.raises(OutOfSampleUnsupported):
        fit.transform(rng.normal(size=(2, 6)).astype(np.float32))


def test_an_overlay_capped_at_a_few_points_keeps_the_shape_of_the_set():
    assert list(overlay_indices(5, cap=10)) == [0, 1, 2, 3, 4]
    thinned = overlay_indices(100, cap=10)
    assert len(thinned) <= 10
    assert thinned[0] == 0 and thinned[-1] < 100
    assert len(set(np.diff(thinned))) == 1, "evenly spaced, not a head and not a random draw"


# ---- the map --------------------------------------------------------------

def test_ticking_an_overlay_places_it_without_projecting_again(view, monkeypatch):
    """The whole point: the cloud the user is reading must not rearrange
    itself to let three new crops in."""
    import lookin.ui.projection_view as projection_view

    _run(view)
    before = view._last_result.proj.copy()

    calls = []
    real = projection_view.ea.fit_projection
    monkeypatch.setattr(projection_view.ea, "fit_projection",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    view._on_overlay_toggle("probe", True)

    assert calls == [], "placing points onto an existing fit is not a projection"
    assert np.array_equal(view._last_result.proj, before)
    layers = view._overlay_layers(view._last_result)
    assert [layer.name for layer in layers] == ["probe"]
    assert layers[0].coords.shape == (3, 2)


def test_placed_coordinates_are_computed_once_and_kept(view):
    _run(view, overlay=True)
    result = view._last_result
    first = view._overlay_layers(result)[0].coords
    assert "probe" in result.overlay_proj
    again = view._overlay_layers(result)[0].coords
    assert again is first, "a recolor must not pay for another transform"


def test_refitting_puts_them_in_the_fit_and_says_so(view):
    _run(view, overlay=True, refit=True)
    result = view._last_result
    assert result.refit_with_overlay and result.overlay_names == ["probe"]
    assert result.overlay_proj["probe"].shape == (3, 2)
    note = view._fit_note(result, view._overlay_layers(result), len(result.proj))
    assert "together with" in note

    view.refit_check.value = False
    _run(view, refit=False)
    note = view._fit_note(view._last_result, view._overlay_layers(view._last_result),
                          len(view._last_result.proj))
    assert "only projected onto that fit" in note


def test_changing_the_overlay_while_refitting_asks_for_a_run_instead_of_lying(view):
    """With the overlay part of the fit, ticking another set changes what
    the projection is computed from -- so the map on screen is dropped
    rather than redrawn with points the fit never saw."""
    _run(view, overlay=True, refit=True)
    assert view._has_result
    view._on_overlay_toggle("probe", False)
    assert not view._has_result


def test_tsne_leaves_no_choice_and_explains_itself(view):
    view.refit_check.value = False
    view._refit_choice = False
    view.projection_method.value = "tsne"
    view._sync_refit_control()
    assert view.refit_check.value and view.refit_check.disabled
    assert "t-SNE" in view.overlay_note.value

    # ...and going back restores what the user had chosen for themselves.
    view.projection_method.value = "pca"
    view._sync_refit_control()
    assert not view.refit_check.value and not view.refit_check.disabled


# ---- what the points are colored by ---------------------------------------

def _legend(view, key):
    view.color_by.value = key
    result = view._last_result
    mapping, _ = view._color_mapping(result, view._overlay_layers(result))
    return mapping, [name for _color, name in mapping.legend]


def test_where_a_point_comes_from_is_a_coloring_of_its_own(view):
    _run(view, overlay=True)
    mapping, names = _legend(view, SOURCE_COLOR_KEY)
    assert names == [DATASET_SOURCE_GROUP, "probe"]
    assert len(mapping.colors) == len(view._last_result.proj) + 3


def test_the_source_entry_is_only_offered_when_there_is_a_second_source(loaded_state, embeddings_npz):
    loaded_state.experiments["exp"].load_embeddings(str(embeddings_npz))
    bare = ProjectionView(FakeCtx(loaded_state))
    bare.build()
    bare.refresh()
    assert SOURCE_COLOR_KEY not in bare._color_column_map


def test_an_inference_only_column_is_offered_and_leaves_the_dataset_grey(view):
    _run(view, overlay=True)
    key = next(k for k, column in view._color_column_map.items() if column == "sensor")
    mapping, names = _legend(view, key)
    assert set(names) == {"alpha", "beta", "(missing)"}
    # The dataset has no 'sensor'; its points say so rather than borrowing
    # a color that would mean something.
    assert set(mapping.groups[: len(view._last_result.proj)]) == {"(missing)"}


def test_a_column_only_the_dataset_has_leaves_the_overlay_grey(view):
    _run(view, overlay=True)
    key = next(k for k, column in view._color_column_map.items() if column == "correct")
    mapping, _ = _legend(view, key)
    assert set(mapping.groups[-3:]) == {"(missing)"}


def test_a_value_they_share_gets_one_colour_across_both_pools(view):
    """'site' exists on both sides, and north is north wherever it is read
    -- otherwise comparing the two clouds by eye means nothing."""
    _run(view, overlay=True)
    key = next(k for k, column in view._color_column_map.items() if column == "site")
    mapping, names = _legend(view, key)
    assert "north" in names and "west" in names
    assert mapping.color_of("north") != mapping.color_of("west")
    assert mapping.groups[-3] == "north"  # p1, from the inference


def test_an_overlaid_crop_has_no_class_and_is_not_given_one(view):
    _run(view, overlay=True)
    mapping, names = _legend(view, CLASS_COLOR_KEY)
    assert set(mapping.groups[-3:]) == {"(missing)"}
    assert "(missing)" in names


# ---- finding what looks like an inference crop ----------------------------

def test_a_crop_outside_the_dataset_can_still_ask_what_looks_like_it(state_with_inference):
    inference = state_with_inference.inferences["probe"]
    exp = state_with_inference.experiments["exp"]
    found = state_with_inference.inference_for_uuid("p2")
    assert found is inference

    neighbours = nearest_to_vector(
        exp.embeddings, exp.embeddings_uuids, inference.vector_for("p2"), k=2,
    )
    assert len(neighbours) == 2
    # The neighbours are dataset crops -- which is the answer being asked
    # for: "what in the dataset looks like this?"
    assert set(n.uuid for n in neighbours) <= {"a", "b", "c", "d"}


def test_a_dataset_uuid_is_not_claimed_by_an_inference(state_with_inference):
    assert state_with_inference.inference_for_uuid("a") is None


def test_an_inference_column_is_not_buried_under_the_class_scores(view):
    """The per-class score columns are last on purpose -- a 200-class run
    contributes 200 of them. A column that arrived with a set the user
    loaded on purpose belongs with the metadata, above that wall."""
    offered = [column for column in view._color_column_map.values() if column is not None]
    assert "sensor" in offered
    scores = [c for c in ("cat", "dog", "bird") if c in offered]
    assert scores, "the experiment's score columns should be offered too"
    assert offered.index("sensor") < min(offered.index(c) for c in scores)


def test_embeddings_from_another_model_are_refused_in_words_a_user_can_act_on(view, tmp_path):
    """Same map means same space. A different width proves a different
    model; the same width and a different model is undetectable, which is
    exactly why this says what it checked."""
    csv = tmp_path / "wide.csv"
    pd.DataFrame({"uuid": ["w1"], "img": ["w.png"]}).to_csv(csv, index=False)
    npz = tmp_path / "wide_embeddings.npz"
    np.savez(npz, uuid=np.array(["w1"], dtype="<U8"), embeddings=np.ones((1, 9), dtype=np.float32))
    view.ctx.state.load_inference("wide", str(csv), str(npz), uuid_col="uuid", crop_col="img")
    view.refresh()

    _run(view)
    view._on_overlay_toggle("wide", True)
    result = view._last_result
    layers = view._overlay_layers(result)

    assert layers == [], "a set from another model is left off rather than placed anywhere"
    note = view._fit_note(result, layers, len(result.proj))
    assert "9-dimensional" in note and "same model" in note
