"""The in-memory data model: loading, column typing, the join, and the
review overlay applied to it.

The cache invalidation cases matter as much as the happy paths here: a
merge that survives a change it should not have survived shows the user
stale numbers with no indication anything is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core import data_manager as dm
from lookin.core.data_manager import AppState, ColumnConfig, Dataset, sidecar_paths_for_csv
from lookin.core.review import STATUS_EXCLUDE, STATUS_OK, STATUS_RELABEL


# ---- sidecar naming -----------------------------------------------------

@pytest.mark.parametrize("csv,expected_stem", [
    ("baseline_cnn_results.csv", "baseline_cnn"),   # the _results suffix is swapped out
    ("baseline_cnn.csv", "baseline_cnn"),           # no suffix: used as-is
    ("results.csv", "results"),                     # a bare "results.csv" is not "_results"
    ("a_results_results.csv", "a_results"),         # only the trailing one goes
])
def test_sidecar_paths_for_csv(csv, expected_stem):
    embed, proto = sidecar_paths_for_csv(csv)
    assert embed.name == f"{expected_stem}_embeddings.npz"
    assert proto.name == f"{expected_stem}_prototypes.npz"


def test_sidecar_paths_keep_the_directory():
    """Works the same on a filesystem path and an MLflow artifact path."""
    embed, _ = sidecar_paths_for_csv("run/artifacts/deep/preds_results.csv")
    assert embed.as_posix() == "run/artifacts/deep/preds_embeddings.npz"


# ---- column typing ------------------------------------------------------

def test_column_kinds_sorts_metadata_by_shape(tiny_dataset_df, monkeypatch):
    monkeypatch.setattr(dm, "HIGH_CARDINALITY_MAX_UNIQUE", 4)
    dataset = Dataset(df=tiny_dataset_df, columns=ColumnConfig(), base_dir=".")
    kinds = dataset.column_kinds()

    assert kinds.binary == ["flag"], "two values means two checkboxes"
    assert "site" in kinds.categorical
    assert "note" in kinds.high_cardinality, "free text must stay out of the filter panel"
    assert "uuid" not in kinds.categorical + kinds.numeric + kinds.binary, \
        "the id/crop/label columns are roles, not metadata"


def test_a_numeric_column_is_only_a_range_slider_once_it_has_many_values(tiny_dataset_df):
    """Below CATEGORICAL_MAX_UNIQUE, checkboxes are more useful than a
    slider even for numbers -- above it, the reverse."""
    few = tiny_dataset_df.assign(area=[1, 1, 2, 2, 3, 3])
    assert "area" in Dataset(df=few, columns=ColumnConfig(), base_dir=".").column_kinds().categorical

    many = pd.DataFrame({
        "uuid": [str(i) for i in range(50)], "crop": ["c.png"] * 50,
        "label": ["a", "b"] * 25, "area": np.arange(50, dtype=float),
    })
    assert "area" in Dataset(df=many, columns=ColumnConfig(), base_dir=".").column_kinds().numeric


# ---- crop paths ---------------------------------------------------------

def test_relative_crops_resolve_against_the_csv_directory(tmp_path, tiny_dataset_df):
    dataset = Dataset(df=tiny_dataset_df, columns=ColumnConfig(), base_dir=tmp_path)
    assert dataset.resolve_crop_path("crops/a.png") == tmp_path / "crops/a.png"


def test_image_root_overrides_the_csv_directory(tmp_path, tiny_dataset_df):
    """The CSV can live in a repo while the crops sit on a mounted volume."""
    dataset = Dataset(df=tiny_dataset_df, columns=ColumnConfig(), base_dir=tmp_path,
                      image_root=tmp_path / "elsewhere")
    assert dataset.resolve_crop_path("crops/a.png") == tmp_path / "elsewhere/crops/a.png"


def test_absolute_crop_paths_are_left_alone(tmp_path, tiny_dataset_df):
    dataset = Dataset(df=tiny_dataset_df, columns=ColumnConfig(), base_dir=tmp_path,
                      image_root=tmp_path / "elsewhere")
    assert dataset.resolve_crop_path("/mnt/data/a.png").as_posix() == "/mnt/data/a.png"


# ---- loading ------------------------------------------------------------

def test_load_dataset_reports_a_missing_column_by_name(dataset_csv):
    state = AppState()
    with pytest.raises(ValueError, match="not_a_column"):
        state.load_dataset(str(dataset_csv), ColumnConfig(uuid_col="not_a_column"))


def test_load_dataset_applies_a_row_filter(dataset_csv):
    state = AppState()
    dataset = state.load_dataset(str(dataset_csv), ColumnConfig(), row_filter=("site", "north"))
    assert len(dataset.df) == 2
    assert set(dataset.df["site"]) == {"north"}


def test_an_unknown_filter_column_leaves_the_data_alone(dataset_csv):
    """Silently returning an unfiltered dataset is bad, but refusing to load
    is worse -- the load logs and continues."""
    state = AppState()
    dataset = state.load_dataset(str(dataset_csv), ColumnConfig(), row_filter=("nope", "x"))
    assert len(dataset.df) == 6


def test_a_label_column_with_too_many_values_is_refused(tmp_path, monkeypatch):
    """Pointing label_col at a uuid column would build one UI control per
    row and look exactly like a hang."""
    monkeypatch.setattr(dm, "LABEL_MAX_CLASSES", 3)
    df = pd.DataFrame({"uuid": list("abcde"), "crop": list("abcde"), "label": list("abcde")})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="unique values"):
        AppState().load_dataset(str(csv), ColumnConfig())


def test_numeric_labels_match_the_score_column_headers(tmp_path):
    """Labels 0/1 read as ints, but the experiment's classes are its score
    column headers "0"/"1" -- without normalizing, no prediction is ever
    correct and the metrics refuse the string/number mix."""
    from lookin.core.metrics import compute_metrics

    pd.DataFrame({"uuid": list("abc"), "crop": ["c.png"] * 3, "label": [0, 1, 1]}).to_csv(
        tmp_path / "d.csv", index=False)
    pd.DataFrame({"uuid": list("abc"), "0": [0.9, 0.2, 0.6], "1": [0.1, 0.8, 0.4]}).to_csv(
        tmp_path / "r.csv", index=False)
    state = AppState()
    dataset = state.load_dataset(str(tmp_path / "d.csv"), ColumnConfig())
    experiment = state.add_experiment(str(tmp_path / "r.csv"), "exp")

    assert dataset.labels() == ["0", "1"]
    merged = state.merged("exp")
    assert merged["correct"].tolist() == [True, True, False]
    metrics = compute_metrics("exp", merged, "label", experiment.score_cols, experiment.class_labels)
    assert list(metrics.confusion.index) == ["0", "1"]


def test_whole_number_labels_with_a_gap_keep_their_integer_spelling(tmp_path):
    """A missing value makes pandas read 0/1 as floats; "1.0" would still not
    match a "1" header, and the gap must stay a gap, not become "nan"."""
    pd.DataFrame({"uuid": list("abc"), "crop": ["c.png"] * 3, "label": [0, None, 1]}).to_csv(
        tmp_path / "d.csv", index=False)
    labels = AppState().load_dataset(str(tmp_path / "d.csv"), ColumnConfig()).df["label"]
    assert labels.iloc[0] == "0" and labels.iloc[2] == "1"
    assert pd.isna(labels.iloc[1])


def test_fractional_labels_are_left_spelled_as_read(tmp_path):
    pd.DataFrame({"uuid": list("ab"), "crop": ["c.png"] * 2, "label": [0.5, 1.5]}).to_csv(
        tmp_path / "d.csv", index=False)
    labels = AppState().load_dataset(str(tmp_path / "d.csv"), ColumnConfig()).df["label"]
    assert labels.tolist() == ["0.5", "1.5"]


def test_chunked_and_unchunked_reads_agree(tmp_path, monkeypatch):
    """The chunked path only exists to feed a progress bar; it must not
    change what is read."""
    df = pd.DataFrame({"uuid": [str(i) for i in range(500)],
                       "crop": ["c.png"] * 500, "label": ["a", "b"] * 250})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    plain = AppState().load_dataset(str(csv), ColumnConfig()).df

    monkeypatch.setattr(dm, "CHUNK_ROWS", 64)
    seen = []
    chunked = AppState().load_dataset(str(csv), ColumnConfig(),
                                      progress_cb=lambda done, total: seen.append((done, total))).df
    pd.testing.assert_frame_equal(plain, chunked)
    assert seen and seen[-1][0] == seen[-1][1] == 500, "progress ends at 100%"


def test_numeric_columns_are_downcast(dataset_csv):
    state = AppState()
    dataset = state.load_dataset(str(dataset_csv), ColumnConfig())
    assert dataset.df["area"].dtype == np.float32


def test_the_id_column_is_never_downcast(tmp_path):
    """A numeric-looking uuid downcast to int32 would stop matching the
    experiment CSV's string uuids."""
    df = pd.DataFrame({"uuid": [1, 2, 3], "crop": ["c.png"] * 3, "label": list("abc")})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    dataset = AppState().load_dataset(str(csv), ColumnConfig())
    assert dataset.df["uuid"].dtype == np.int64


# ---- a dataset with no id column ---------------------------------------

def test_a_dataset_can_be_loaded_with_no_uuid_column(tmp_path, tiny_dataset_df):
    csv = tmp_path / "d.csv"
    tiny_dataset_df.drop(columns=["uuid"]).to_csv(csv, index=False)
    state = AppState()
    dataset = state.load_dataset(str(csv), ColumnConfig(uuid_col=""))
    assert len(dataset.df) == 6
    assert "site" in dataset.metadata_columns(), "metadata is still there to filter and chart"
    assert state.review is None, "decisions are keyed by the id: no store without one"


def test_no_experiment_can_be_loaded_against_a_dataset_with_no_ids(tmp_path, tiny_dataset_df, results_csv):
    csv = tmp_path / "d.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    state = AppState()
    state.load_dataset(str(csv), ColumnConfig(uuid_col=""))
    with pytest.raises(ValueError, match="no uuid column"):
        state.add_experiment(str(results_csv), "exp")
    assert not state.experiments


def test_loading_a_dataset_with_no_ids_unloads_the_experiments(loaded_state, dataset_csv):
    """They were joined to the previous dataset; every view would fail on
    them against this one."""
    loaded_state.load_dataset(str(dataset_csv), ColumnConfig(uuid_col=""))
    assert not loaded_state.experiments


# ---- experiments --------------------------------------------------------

def test_score_columns_exclude_the_id_and_the_argmax_label(loaded_state):
    exp = loaded_state.experiments["exp"]
    assert exp.score_cols == ["cat", "dog", "bird"]
    assert exp.class_labels == ["cat", "dog", "bird"], "class names are the column headers"


def test_a_results_csv_with_no_scores_is_refused(tmp_path):
    pd.DataFrame({"uuid": ["a"], "label": ["cat"]}).to_csv(tmp_path / "r.csv", index=False)
    with pytest.raises(ValueError, match="score"):
        AppState().add_experiment(str(tmp_path / "r.csv"), "exp")


def test_rename_keeps_position_and_loaded_data(loaded_state, results_csv):
    loaded_state.add_experiment(str(results_csv), "second")
    df_before = loaded_state.experiments["exp"].df
    loaded_state.rename_experiment("exp", "renamed")

    assert list(loaded_state.experiments) == ["renamed", "second"], "order is user-visible everywhere"
    assert loaded_state.experiments["renamed"].df is df_before, "renaming must never re-read the CSV"
    assert loaded_state.experiments["renamed"].name == "renamed"


def test_rename_onto_an_existing_name_is_refused(loaded_state, results_csv):
    loaded_state.add_experiment(str(results_csv), "second")
    with pytest.raises(ValueError):
        loaded_state.rename_experiment("exp", "second")


def test_rename_of_an_unknown_experiment_is_refused(loaded_state):
    with pytest.raises(KeyError):
        loaded_state.rename_experiment("nope", "x")


def test_reorder_tolerates_a_stale_name_list(loaded_state, results_csv):
    loaded_state.add_experiment(str(results_csv), "b")
    loaded_state.add_experiment(str(results_csv), "c")
    loaded_state.reorder_experiments(["c", "gone", "exp"])
    assert list(loaded_state.experiments) == ["c", "exp", "b"], "unlisted ones keep their order at the end"


def test_uuid_overlap_counts_both_sides_and_duplicates(tmp_path, dataset_csv):
    pd.DataFrame({
        "uuid": ["a", "a", "b", "zz"],
        "cat": [0.9, 0.9, 0.1, 0.5], "dog": [0.1, 0.1, 0.9, 0.5],
    }).to_csv(tmp_path / "r.csv", index=False)
    state = AppState()
    state.load_dataset(str(dataset_csv), ColumnConfig())
    state.add_experiment(str(tmp_path / "r.csv"), "exp")

    overlap = state.uuid_overlap("exp")
    assert overlap.matched == 2                 # a, b
    assert overlap.dataset_only == 4            # c, d, e, f
    assert overlap.experiment_only == 1         # zz
    assert overlap.experiment_dupes == 1        # the second "a"
    assert overlap.dataset_dupes == 0


# ---- the join -----------------------------------------------------------

def test_merged_derives_prediction_columns(loaded_state):
    merged = loaded_state.merged("exp").set_index("uuid")
    assert merged.loc["a", "predicted_label"] == "cat"
    assert merged.loc["a", "predicted_score"] == pytest.approx(0.8)
    assert bool(merged.loc["a", "correct"]) is True
    assert merged.loc["f", "predicted_label"] == "dog", "true label is bird -- a real error"
    assert bool(merged.loc["f", "correct"]) is False


def test_merged_is_cached(loaded_state):
    assert loaded_state.merged("exp") is loaded_state.merged("exp")


def test_editing_a_class_label_invalidates_the_merge(loaded_state):
    first = loaded_state.merged("exp")
    loaded_state.set_class_label("exp", 0, "feline")
    second = loaded_state.merged("exp")
    assert second is not first
    assert second.set_index("uuid").loc["a", "predicted_label"] == "feline"


def test_renaming_an_experiment_keeps_the_merge_cached(loaded_state):
    """A rename changes no row -- paying for a re-merge to fix a typo is the
    thing notify_rename exists to avoid."""
    first = loaded_state.merged("exp")
    loaded_state.rename_experiment("exp", "renamed")
    assert loaded_state.merged("renamed") is first


def test_label_remap_cascades_into_every_experiment(loaded_state):
    loaded_state.apply_label_remap({"cat": "feline"})
    assert "feline" in loaded_state.dataset.labels()
    assert loaded_state.experiments["exp"].class_labels[0] == "feline"
    merged = loaded_state.merged("exp").set_index("uuid")
    assert bool(merged.loc["a", "correct"]) is True, "the remap must not break correctness"


# ---- the review overlay through the join --------------------------------

def test_the_overlay_is_off_by_default(loaded_state):
    loaded_state.review.create_version("v1")
    loaded_state.review.decide("a", STATUS_EXCLUDE)
    assert len(loaded_state.merged("exp")) == 6


def test_the_overlay_relabels_and_drops_before_correctness_is_computed(loaded_state):
    loaded_state.review.create_version("v1")
    loaded_state.review.decide("a", STATUS_EXCLUDE)
    loaded_state.review.decide("f", STATUS_RELABEL, new_label="dog")
    loaded_state.use_reviewed_labels = True

    merged = loaded_state.merged("exp").set_index("uuid")
    assert "a" not in merged.index, "an excluded crop is not a sample the model got wrong"
    assert merged.loc["f", "label"] == "dog"
    assert bool(merged.loc["f", "correct"]) is True, "correctness follows the corrected label"


def test_a_decision_only_invalidates_the_merge_while_the_overlay_is_on(loaded_state):
    loaded_state.review.create_version("v1")
    first = loaded_state.merged("exp")
    loaded_state.review.decide("a", STATUS_OK)
    assert loaded_state.merged("exp") is first, "reviewing must not re-merge 500k rows"

    loaded_state.use_reviewed_labels = True
    with_overlay = loaded_state.merged("exp")
    assert with_overlay is not first
    loaded_state.review.decide("b", STATUS_EXCLUDE)
    assert loaded_state.merged("exp") is not with_overlay, "...but now it has to"


def test_clear_all_drops_the_review_store(loaded_state):
    """Keeping a store bound to an unloaded dataset would let a decision be
    written against the wrong CSV."""
    loaded_state.clear_all()
    assert loaded_state.review is None
    assert loaded_state.dataset is None
    assert loaded_state.use_reviewed_labels is False


# ---- sidecars -----------------------------------------------------------

def test_embeddings_are_l2_normalized_on_load(tmp_path, loaded_state):
    path = tmp_path / "e.npz"
    np.savez(path, uuid=np.array(["a", "b"], dtype="<U8"),
             embeddings=np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    exp = loaded_state.experiments["exp"]
    assert exp.load_embeddings(str(path)) == 2
    assert np.allclose(np.linalg.norm(exp.embeddings, axis=1), 1.0)


def test_a_length_mismatch_in_the_sidecar_is_refused(tmp_path, loaded_state):
    path = tmp_path / "e.npz"
    np.savez(path, uuid=np.array(["a", "b"], dtype="<U8"),
             embeddings=np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="mismatch"):
        loaded_state.experiments["exp"].load_embeddings(str(path))


def test_a_sidecar_missing_an_array_names_what_it_found(tmp_path, loaded_state):
    path = tmp_path / "e.npz"
    np.savez(path, uuid=np.array(["a"], dtype="<U8"))
    with pytest.raises(ValueError, match="embeddings"):
        loaded_state.experiments["exp"].load_embeddings(str(path))


def test_a_pickled_object_array_is_refused_with_the_actual_fix(tmp_path, loaded_state):
    """np.array(list_of_str) pickles; the numpy traceback for that is not
    something a user can act on, so the message says what to do instead."""
    path = tmp_path / "e.npz"
    np.savez(path, uuid=np.array(["a", "b"], dtype=object),
             embeddings=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="dtype='<U64'"):
        loaded_state.experiments["exp"].load_embeddings(str(path))


def test_partial_embedding_coverage_is_reported_by_the_mask(loaded_state, embeddings_npz):
    """A sidecar covering only some rows is normal (a subsampled
    extraction), so it must not silently pass zeros off as embeddings."""
    exp = loaded_state.experiments["exp"]
    exp.load_embeddings(str(embeddings_npz))
    merged = loaded_state.merged("exp")

    embeddings, mask = exp.aligned_embeddings(merged, "uuid")
    assert embeddings.shape == (len(merged), 3)
    assert mask.tolist() == [True, True, True, True, False, False]
    assert not embeddings[~mask].any()


def test_aligned_embeddings_survives_a_dataset_with_no_matching_uuid(loaded_state, embeddings_npz):
    """A results CSV loaded against a dataset it has no uuid in common with
    merges to zero rows -- easy to do from the MLflow browser, where the run
    picked is not necessarily the one the loaded dataset came from. That has
    to come back empty, not raise: it used to blow up on an empty float64
    index array, which killed the whole notify_change() fan-out and left the
    other tabs unrefreshed."""
    exp = loaded_state.experiments["exp"]
    exp.load_embeddings(str(embeddings_npz))
    empty = loaded_state.merged("exp").iloc[:0]

    embeddings, mask = exp.aligned_embeddings(empty, "uuid")
    assert embeddings.shape == (0, 3)
    assert mask.tolist() == []


def test_prototypes_accept_the_k_equals_one_shape(tmp_path, loaded_state):
    path = tmp_path / "p.npz"
    np.savez(path, labels=np.array(["cat", "dog"], dtype="<U8"),
             prototypes=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    proto = loaded_state.experiments["exp"].load_model_prototypes(str(path))
    assert proto.vectors.shape == (2, 1, 2), "(C, D) is read as (C, 1, D)"
    assert proto.class_labels == ["cat", "dog"]


def test_a_class_absent_from_the_prototype_pool_stays_nan(tmp_path, loaded_state):
    """NaN rows are preserved so downstream code can skip them rather than
    silently scoring against a zero vector."""
    path = tmp_path / "p.npz"
    np.savez(path, labels=np.array(["cat", "dog"], dtype="<U8"),
             prototypes=np.array([[1.0, 0.0], [np.nan, np.nan]], dtype=np.float32))
    proto = loaded_state.experiments["exp"].load_model_prototypes(str(path))
    assert np.isnan(proto.vectors[1]).all()
    assert not np.isnan(proto.vectors[0]).any()


def test_a_label_column_with_thousands_of_classes_loads(tmp_path):
    """Fine-grained datasets have thousands of classes; the label guard is
    only there to catch a uuid or free-text column."""
    labels = [f"c{i}" for i in range(2000)]
    df = pd.DataFrame({"uuid": [f"u{i}" for i in range(4000)], "crop": "x.png", "label": labels * 2})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    state = AppState()
    state.load_dataset(str(csv), ColumnConfig())
    assert len(state.dataset.labels()) == 2000
