"""Similarity search and class overlap.

Both are built on cosine similarity over L2-normalized vectors, so the
tests use hand-placed unit vectors whose neighbours and overlaps can be
read off by eye rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from lookin.core.embedding_analysis import class_overlap, l2_normalize, nearest_neighbours


@pytest.fixture
def toy():
    """Four crops: a and b nearly identical, c orthogonal, d opposite to a."""
    embeddings = l2_normalize(np.array([
        [1.0, 0.0],
        [0.99, 0.14],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32))
    uuids = np.array(["a", "b", "c", "d"])
    labels = np.array(["cat", "cat", "dog", "bird"])
    return embeddings, uuids, labels


# ---- nearest neighbours -------------------------------------------------

def test_neighbours_come_back_nearest_first(toy):
    embeddings, uuids, labels = toy
    found = nearest_neighbours(embeddings, uuids, "a", k=3, labels=labels)
    assert [n.uuid for n in found] == ["b", "c", "d"]
    assert found[0].similarity > found[1].similarity > found[2].similarity
    assert found[0].label == "cat"


def test_the_query_is_never_its_own_neighbour(toy):
    """It is trivially the closest thing to itself and would waste a row."""
    embeddings, uuids, _ = toy
    assert "a" not in [n.uuid for n in nearest_neighbours(embeddings, uuids, "a", k=4)]


def test_k_is_capped_at_what_exists(toy):
    embeddings, uuids, _ = toy
    assert len(nearest_neighbours(embeddings, uuids, "a", k=100)) == 3


def test_an_unknown_uuid_returns_nothing_rather_than_raising(toy):
    """A crop can be in the dataset and absent from a subsampled sidecar."""
    embeddings, uuids, _ = toy
    assert nearest_neighbours(embeddings, uuids, "not-here", k=3) == []


def test_no_embeddings_returns_nothing(toy):
    _, uuids, _ = toy
    assert nearest_neighbours(np.zeros((0, 2)), np.array([]), "a", k=3) == []


def test_similarity_is_the_cosine(toy):
    embeddings, uuids, _ = toy
    found = nearest_neighbours(embeddings, uuids, "a", k=3)
    by_uuid = {n.uuid: n.similarity for n in found}
    assert by_uuid["c"] == pytest.approx(0.0, abs=1e-6)   # orthogonal
    assert by_uuid["d"] == pytest.approx(-1.0, abs=1e-6)  # opposite


def test_neighbours_scale_to_a_large_pool():
    """argpartition rather than a full sort -- this is the path a click in
    the carousel takes on a real dataset."""
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.normal(size=(20_000, 32)).astype(np.float32))
    uuids = np.array([f"u{i}" for i in range(20_000)])
    found = nearest_neighbours(embeddings, uuids, "u7", k=10)
    assert len(found) == 10
    assert all(found[i].similarity >= found[i + 1].similarity for i in range(9))


# ---- class overlap ------------------------------------------------------

def test_a_class_is_more_like_itself_than_like_the_others():
    rng = np.random.default_rng(1)
    a = l2_normalize(np.array([1.0, 0.0]) + rng.normal(scale=0.05, size=(60, 2)))
    b = l2_normalize(np.array([0.0, 1.0]) + rng.normal(scale=0.05, size=(60, 2)))
    embeddings = np.vstack([a, b]).astype(np.float32)
    labels = np.array(["cat"] * 60 + ["dog"] * 60)

    overlap = class_overlap(embeddings, labels)
    assert overlap.labels == ["cat", "dog"]
    assert overlap.matrix[0, 0] > overlap.matrix[0, 1]
    assert (overlap.separation > 0).all(), "well-separated classes have positive separation"


def test_two_classes_sitting_on_top_of_each_other_show_it():
    """The point of the measure: the confusion matrix says the model mixes
    them up, this says they genuinely overlap."""
    rng = np.random.default_rng(2)
    shared = l2_normalize(np.array([1.0, 0.0]) + rng.normal(scale=0.05, size=(120, 2)))
    embeddings = shared.astype(np.float32)
    labels = np.array(["cat"] * 60 + ["dog"] * 60)

    overlap = class_overlap(embeddings, labels)
    assert overlap.matrix[0, 1] == pytest.approx(overlap.matrix[0, 0], abs=0.02)
    assert abs(overlap.separation[0]) < 0.02, "no separation to speak of"


def test_the_matrix_is_symmetric_and_the_diagonal_excludes_self_similarity():
    rng = np.random.default_rng(3)
    embeddings = l2_normalize(rng.normal(size=(80, 8))).astype(np.float32)
    labels = np.array(["a", "b"] * 40)
    overlap = class_overlap(embeddings, labels)

    assert np.allclose(overlap.matrix, overlap.matrix.T, equal_nan=True)
    assert overlap.matrix[0, 0] < 0.99, "a self-similarity of 1 per crop would pin the diagonal at 1"


def test_a_single_crop_class_has_no_own_similarity():
    """Nothing to compare it with; NaN rather than a fabricated 1.0."""
    embeddings = l2_normalize(np.eye(3, 3)).astype(np.float32)
    labels = np.array(["lonely", "b", "b"])
    overlap = class_overlap(embeddings, labels)
    lonely = overlap.labels.index("lonely")
    assert np.isnan(overlap.matrix[lonely, lonely])
    assert np.isfinite(overlap.matrix[lonely, overlap.labels.index("b")]), \
        "it still compares against other classes"


def test_top_pairs_lists_the_worst_overlaps_first():
    rng = np.random.default_rng(4)
    close_a = l2_normalize(np.array([1.0, 0.0, 0.0]) + rng.normal(scale=0.03, size=(40, 3)))
    close_b = l2_normalize(np.array([1.0, 0.05, 0.0]) + rng.normal(scale=0.03, size=(40, 3)))
    far = l2_normalize(np.array([0.0, 0.0, 1.0]) + rng.normal(scale=0.03, size=(40, 3)))
    embeddings = np.vstack([close_a, close_b, far]).astype(np.float32)
    labels = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)

    pairs = class_overlap(embeddings, labels).top_pairs(3)
    assert {pairs[0][0], pairs[0][1]} == {"a", "b"}, "the two that sit on top of each other"
    assert pairs[0][2] > pairs[-1][2]


def test_class_order_is_honoured_and_absent_classes_dropped():
    embeddings = l2_normalize(np.eye(4, 4)).astype(np.float32)
    labels = np.array(["b", "b", "a", "a"])
    overlap = class_overlap(embeddings, labels, class_order=["a", "b", "ghost"])
    assert overlap.labels == ["a", "b"]
    assert list(overlap.counts) == [2, 2]


def test_large_classes_are_subsampled_not_scanned_whole():
    rng = np.random.default_rng(5)
    embeddings = l2_normalize(rng.normal(size=(3000, 16))).astype(np.float32)
    labels = np.array(["a"] * 1500 + ["b"] * 1500)
    overlap = class_overlap(embeddings, labels, max_per_class=100)
    assert list(overlap.counts) == [1500, 1500], "counts report the real class sizes"
    assert np.isfinite(overlap.matrix).all()
