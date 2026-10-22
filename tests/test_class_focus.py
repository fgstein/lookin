"""Which classes a confusion grid shows once there are more than it can hold
(core/class_focus.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lookin.core import class_focus as cf


def _matrix(labels, cells: dict, diagonal: int = 10) -> pd.DataFrame:
    m = pd.DataFrame(0, index=labels, columns=labels, dtype=np.int64)
    for label in labels:
        m.loc[label, label] = diagonal
    for (true, pred), n in cells.items():
        m.loc[true, pred] = n
    return m


def _per_class(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["label", "f1", "support"])


def test_pairs_are_off_diagonal_largest_first_with_share_of_the_true_class():
    m = _matrix(["a", "b", "c"], {("a", "b"): 5, ("c", "a"): 9, ("b", "c"): 1})
    pairs = cf.top_confused_pairs(m)
    assert list(zip(pairs["true"], pairs["predicted"], pairs["count"])) == [
        ("c", "a", 9), ("a", "b", 5), ("b", "c", 1)]
    assert pairs["share"].iloc[0] == 9 / (9 + 10)


def test_pairs_limit_and_an_all_diagonal_matrix():
    m = _matrix(["a", "b", "c"], {("a", "b"): 5, ("c", "a"): 9})
    assert len(cf.top_confused_pairs(m, limit=1)) == 1
    assert cf.top_confused_pairs(_matrix(["a", "b"], {})).empty


def test_most_confused_takes_both_ends_of_the_largest_pairs():
    labels = [f"c{i}" for i in range(10)]
    m = _matrix(labels, {("c3", "c7"): 50, ("c1", "c7"): 40, ("c5", "c2"): 30})
    assert cf.rank_classes(cf.FOCUS_MOST_CONFUSED, [m], [], limit=4) == ["c3", "c7", "c1", "c5"]


def test_most_confused_sums_experiments():
    labels = ["a", "b", "c", "d"]
    first = _matrix(labels, {("a", "b"): 5, ("c", "d"): 4})
    second = _matrix(labels, {("c", "d"): 4})
    assert cf.rank_classes(cf.FOCUS_MOST_CONFUSED, [first, second], [], limit=2) == ["c", "d"]


def test_worst_f1_breaks_ties_by_larger_support():
    per_class = _per_class([("a", 0.0, 3), ("b", 0.9, 100), ("c", 0.0, 50), ("d", 0.4, 10)])
    assert cf.rank_classes(cf.FOCUS_WORST_F1, [], [per_class], limit=3) == ["c", "a", "d"]


def test_worst_f1_averages_across_experiments():
    one = _per_class([("a", 0.2, 10), ("b", 0.5, 10)])
    two = _per_class([("a", 0.8, 10), ("b", 0.4, 10)])
    assert cf.rank_classes(cf.FOCUS_WORST_F1, [], [one, two], limit=1) == ["b"]


def test_largest_support_and_manual():
    per_class = _per_class([("a", 0.5, 3), ("b", 0.5, 100), ("c", 0.5, 50)])
    assert cf.rank_classes(cf.FOCUS_LARGEST_SUPPORT, [], [per_class], limit=2) == ["b", "c"]
    assert cf.rank_classes(cf.FOCUS_MANUAL, [], [per_class]) == []


def test_focus_matrix_keeps_rows_summing_to_support():
    labels = ["a", "b", "c", "d"]
    m = _matrix(labels, {("a", "c"): 3, ("a", "d"): 2, ("b", "a"): 1})
    grid, other = cf.focus_matrix(m, ["b", "a"])
    assert list(grid.index) == ["b", "a"] and list(grid.columns) == ["b", "a"]
    assert other.to_dict() == {"b": 0, "a": 5}
    assert (grid.sum(axis=1) + other).to_dict() == m.sum(axis=1)[["b", "a"]].to_dict()


def test_focus_matrix_gives_a_class_an_experiment_never_saw_a_zero_row():
    m = _matrix(["a", "b"], {("a", "b"): 2})
    grid, other = cf.focus_matrix(m, ["a", "z"])
    assert grid.loc["z"].tolist() == [0, 0] and grid["z"].tolist() == [0, 0]
    assert other.to_dict() == {"a": 2, "z": 0}


def test_summed_confusion_covers_the_union_of_classes():
    total = cf.summed_confusion([_matrix(["a", "b"], {}), _matrix(["b", "c"], {("b", "c"): 4})])
    assert list(total.index) == ["a", "b", "c"]
    assert total.loc["b", "b"] == 20 and total.loc["b", "c"] == 4


def test_ranking_2000_classes_is_quick():
    import time

    rng = np.random.default_rng(0)
    labels = [f"class_{i}" for i in range(2000)]
    counts = np.zeros((2000, 2000), dtype=np.int64)
    np.fill_diagonal(counts, 40)
    counts[rng.integers(0, 2000, 5000), rng.integers(0, 2000, 5000)] += 1
    m = pd.DataFrame(counts, index=labels, columns=labels)
    t0 = time.monotonic()
    focus = cf.rank_classes(cf.FOCUS_MOST_CONFUSED, [m], [])
    cf.focus_matrix(m, focus)
    cf.top_confused_pairs(m, limit=15)
    assert len(focus) == cf.FOCUS_MAX
    assert time.monotonic() - t0 < 5


def test_column_headers_drop_the_shared_prefix_at_a_separator():
    from lookin.web.views.common import short_column_labels

    assert short_column_labels(["species_1247", "species_1246"], 8) == (["1247", "1246"], "species_")
    assert short_column_labels(["Quercus robur", "Quercus rubra"], 8) == (["robur", "rubra"], "Quercus ")
    assert short_column_labels(["cat", "car"], 8) == (["cat", "car"], "")
    assert short_column_labels(["sp_12", "sp_1"], 8) == (["12", "1"], "sp_")
    assert short_column_labels(["abcdefghijk", "x"], 4) == (["abc…", "x"], "")


def test_worst_f1_leaves_out_classes_under_the_minimum_support():
    per_class = _per_class([("tiny", 0.0, 2), ("big", 0.3, 400), ("mid", 0.1, 12)])
    assert cf.rank_classes(cf.FOCUS_WORST_F1, [], [per_class], min_support=10) == ["mid", "big"]
    # Averaged over experiments: two experiments of 6 crops each are 6 crops, not 12.
    two = [_per_class([("a", 0.0, 6)]), _per_class([("a", 0.0, 6)])]
    assert cf.rank_classes(cf.FOCUS_WORST_F1, [], two, min_support=10) == []


def test_rank_by_minimum_support():
    table = pd.DataFrame({"label": ["a", "b", "c"], "rate": [0.1, 0.2, 0.3], "support": [3, 50, 20]})
    assert cf.rank_by(table, "rate", ascending=True, min_support=10) == ["b", "c"]
    assert cf.rank_by(table, "rate", ascending=True) == ["a", "b", "c"]


def test_overview_names_that_would_collide_are_merged():
    ticks, names = cf._merged_ticks([10, 11, 500], ["species_0010", "species_0011", "species_0500"], 2000, 22, 12)
    assert ticks == [10.5, 500] and names == ["species_0010, 0011", "species_0500"]
    # Far enough apart on a small matrix, both stay.
    assert cf._merged_ticks([1, 30], ["a_1", "a_30"], 61, 6, 7)[1] == ["a_1", "a_30"]
