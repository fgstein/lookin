"""Candidate sources and the class filter.

The filter's one trap is *when* it runs: after the limit, a class's worst
crops would be whatever few of them happened to rank in the global top N.
Several tests below use limit=1 precisely to catch that.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lookin.core import candidates as cand
from lookin.core.candidates import MATCH_LABEL, MATCH_SUGGESTION, Candidate, ClassFilter

CLASSES = ["cat", "dog", "bird"]


@pytest.fixture
def dataset_df() -> pd.DataFrame:
    return pd.DataFrame({
        "uuid": list("abcdef"),
        "label": ["cat", "cat", "dog", "dog", "bird", "bird"],
    })


def _experiment(name: str, rows: list) -> SimpleNamespace:
    df = pd.DataFrame(rows, columns=["uuid"] + CLASSES)
    return SimpleNamespace(name=name, df=df, score_cols=CLASSES, class_labels=CLASSES, uuid_col="uuid")


@pytest.fixture
def experiment() -> SimpleNamespace:
    # a, d, f right. b: cat->dog at 0.7, c: dog->cat at 0.95, e: bird->dog at 0.6.
    return _experiment("exp", [
        ["a", 0.90, 0.05, 0.05],
        ["b", 0.20, 0.70, 0.10],
        ["c", 0.95, 0.03, 0.02],
        ["d", 0.10, 0.80, 0.10],
        ["e", 0.10, 0.60, 0.30],
        ["f", 0.05, 0.05, 0.90],
    ])


# ---- the filter itself ----------------------------------------------------

def test_class_filter_matches_either_side():
    candidate = Candidate(uuid="x", label="cat", suggestion="dog")
    assert ClassFilter("cat", MATCH_LABEL).keeps(candidate)
    assert not ClassFilter("dog", MATCH_LABEL).keeps(candidate)
    assert ClassFilter("dog", MATCH_SUGGESTION).keeps(candidate)
    assert not ClassFilter("cat", MATCH_SUGGESTION).keeps(candidate)


def test_class_filter_compares_labels_as_strings():
    """Class labels are never normalized, so an integer label in the CSV
    has to match its string form on the other side."""
    assert ClassFilter(3).keeps(Candidate(uuid="x", label="3"))
    assert ClassFilter("3").keeps(Candidate(uuid="x", label=3))


# ---- all errors -----------------------------------------------------------

def test_all_errors_lists_every_error_most_confident_first(dataset_df, experiment):
    result = cand.all_errors(dataset_df, experiment, "uuid", "label")
    assert result.uuids() == ["c", "b", "e"]
    assert [c.suggestion for c in result.candidates] == ["cat", "dog", "dog"]
    assert result.candidates[0].detail["confidence"] == pytest.approx(0.95)


def test_all_errors_of_an_annotated_class(dataset_df, experiment):
    result = cand.all_errors(dataset_df, experiment, "uuid", "label", class_filter=ClassFilter("cat"))
    assert result.uuids() == ["b"]


def test_all_errors_toward_a_suggested_class(dataset_df, experiment):
    result = cand.all_errors(dataset_df, experiment, "uuid", "label",
                             class_filter=ClassFilter("dog", MATCH_SUGGESTION))
    assert result.uuids() == ["b", "e"]


def test_all_errors_has_no_cap(dataset_df):
    n = 1_200  # well past the default limit of every capped source
    df = pd.DataFrame({"uuid": [f"u{i}" for i in range(n)], "label": ["cat"] * n})
    experiment = _experiment("big", [[f"u{i}", 0.1, 0.9, 0.0] for i in range(n)])
    assert len(cand.all_errors(df, experiment, "uuid", "label")) == n


# ---- filtering happens before the limit -----------------------------------

def test_margin_filters_before_the_limit(dataset_df, experiment):
    assert cand.by_margin(dataset_df, experiment, "uuid", "label", limit=1).uuids() == ["c"]
    filtered = cand.by_margin(dataset_df, experiment, "uuid", "label", limit=1,
                              class_filter=ClassFilter("bird"))
    assert filtered.uuids() == ["e"]


def test_consensus_takes_the_class_filter(dataset_df, experiment):
    twin = _experiment("twin", experiment.df.values.tolist())
    result = cand.by_consensus(dataset_df, [experiment, twin], "uuid", "label",
                               class_filter=ClassFilter("dog", MATCH_SUGGESTION))
    assert sorted(result.uuids()) == ["b", "e"]


def test_embedding_flags_filter_before_the_limit():
    flags = SimpleNamespace(
        uuid=np.array(["u1", "u2", "u3"]),
        label=np.array(["cat", "dog", "cat"]),
        best_other_label=np.array(["dog", "cat", "bird"]),
        own_similarity=np.array([0.1, 0.2, 0.5]),
        best_other_similarity=np.array([0.6, 0.6, 0.6]),
        flagged=np.array([True, True, True]),
        reasons=["r", "r", "r"],
    )
    assert cand.from_embedding_flags(flags, "exp", limit=1).uuids() == ["u1"]
    assert cand.from_embedding_flags(flags, "exp", limit=1, class_filter=ClassFilter("dog")).uuids() == ["u2"]
    assert cand.from_embedding_flags(
        flags, "exp", class_filter=ClassFilter("bird", MATCH_SUGGESTION)).uuids() == ["u3"]


def test_near_duplicates_can_be_restricted_to_one_class():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    uuids = np.array(["a", "b", "c", "d"])
    labels = np.array(["cat", "cat", "dog", "dog"])
    pairs, _ = cand.near_duplicates(embeddings, uuids, labels)
    assert {p.label for p in pairs} == {"cat", "dog"}
    pairs, _ = cand.near_duplicates(embeddings, uuids, labels, only_label="dog")
    assert [(p.uuid_a, p.uuid_b) for p in pairs] == [("c", "d")]


# ---- subcenters -----------------------------------------------------------

@pytest.mark.parametrize("class_filter", [
    ClassFilter("cat"), ClassFilter("dog"), ClassFilter("bird"),
    ClassFilter("cat", MATCH_SUGGESTION), ClassFilter("dog", MATCH_SUGGESTION),
])
def test_subcenter_counts_agree_with_what_opening_it_shows(class_filter):
    """The ranking table's count and the opened list must never disagree --
    one is computed without building candidates, the other by filtering
    them."""
    row = cand.SubcenterRow(
        class_label="cat", k_idx=0, n_assigned=3, purity=0.67, compactness=0.5,
        separation=0.1, noise_score=0.3, member_uuids=["a", "b", "c"],
    )
    labels_by_uuid = {"a": "cat", "b": "cat", "c": "dog"}
    opened = cand.apply_class_filter(cand.subcenter_candidates(row, labels_by_uuid), class_filter)
    assert cand.subcenter_matches(row, labels_by_uuid, class_filter) == len(opened)


def test_a_suggested_class_opens_a_subcenter_on_its_intruders():
    row = cand.SubcenterRow(
        class_label="cat", k_idx=0, n_assigned=3, purity=0.67, compactness=0.5,
        separation=0.1, noise_score=0.3, member_uuids=["a", "b", "c"],
    )
    opened = cand.apply_class_filter(
        cand.subcenter_candidates(row, {"a": "cat", "b": "cat", "c": "dog"}),
        ClassFilter("cat", MATCH_SUGGESTION),
    )
    assert opened.uuids() == ["c"]
    assert opened.name.endswith("-- suggested cat")
