"""Head-to-head comparison of two experiments.

The whole point of this module is to surface what two accuracy numbers
hide, so the cases that matter most are the ones where the summary looks
fine: equal accuracy with churn underneath, and a difference matrix whose
most important cell is the one a naive sign-coloring would paint wrong.
"""
from __future__ import annotations

import pandas as pd
import pytest

from lookin.core.comparison import (
    COL_OUTCOME, COL_SCORE_B, COL_TRUE, MATRIX_MOVED_IN, MATRIX_MOVED_OUT, MATRIX_NET,
    MATRIX_VIEWS, OUTCOME_BOTH_RIGHT, OUTCOME_BOTH_WRONG, OUTCOME_FIX, OUTCOME_REGRESSION,
    COL_CAME_FROM, COL_WENT_TO, build_comparison, cell_flows, cell_improvement,
    compare_experiments, flow_improvement, flow_labels, flow_matrix, movement_confusion,
    per_class_change, rows_for_outcome, rows_in_flow_cell, rows_in_movement_cell,
    rows_moved_through_cell,
)
from lookin.core.data_manager import AppState, ColumnConfig

CLASSES = ["cat", "dog", "bird"]


def _write_results(path, uuids, predicted, confidence=0.9):
    """A results CSV whose argmax is exactly `predicted`."""
    rows = []
    for pred in predicted:
        row = {c: (1.0 - confidence) / (len(CLASSES) - 1) for c in CLASSES}
        row[pred] = confidence
        rows.append(row)
    df = pd.DataFrame(rows, columns=CLASSES)
    df.insert(0, "uuid", list(uuids))
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def state(tmp_path):
    """Six crops. Baseline and candidate disagree in a controlled way:

        uuid  true  baseline    candidate   outcome
        a     cat   cat         cat         both right
        b     cat   cat         dog         regression
        c     dog   cat         dog         fix
        d     dog   dog         dog         both right
        e     bird  cat         dog         both wrong
        f     bird  bird        cat         regression
    """
    df = pd.DataFrame({
        "uuid": list("abcdef"),
        "crop": [f"crops/{u}.png" for u in "abcdef"],
        "label": ["cat", "cat", "dog", "dog", "bird", "bird"],
    })
    dataset_csv = tmp_path / "dataset.csv"
    df.to_csv(dataset_csv, index=False)

    _write_results(tmp_path / "a.csv", df["uuid"], ["cat", "cat", "cat", "dog", "cat", "bird"])
    _write_results(tmp_path / "b.csv", df["uuid"], ["cat", "dog", "dog", "dog", "dog", "cat"])

    st = AppState()
    st.load_dataset(str(dataset_csv), ColumnConfig())
    st.add_experiment(str(tmp_path / "a.csv"), "base")
    st.add_experiment(str(tmp_path / "b.csv"), "cand")
    return st


# ---- outcomes -----------------------------------------------------------

def test_every_crop_gets_exactly_one_outcome(state):
    joined = build_comparison(state, "base", "cand")
    by_uuid = joined.set_index("uuid")[COL_OUTCOME]
    assert by_uuid["a"] == OUTCOME_BOTH_RIGHT
    assert by_uuid["b"] == OUTCOME_REGRESSION
    assert by_uuid["c"] == OUTCOME_FIX
    assert by_uuid["d"] == OUTCOME_BOTH_RIGHT
    assert by_uuid["e"] == OUTCOME_BOTH_WRONG
    assert by_uuid["f"] == OUTCOME_REGRESSION
    assert len(joined) == 6


def test_the_summary_counts_add_up(state):
    diff = compare_experiments(state, "base", "cand")
    assert sum(diff.counts.values()) == diff.n_common == 6
    assert diff.counts[OUTCOME_REGRESSION] == 2
    assert diff.counts[OUTCOME_FIX] == 1


def test_accuracies_match_the_outcome_counts(state):
    diff = compare_experiments(state, "base", "cand")
    assert diff.baseline_accuracy == pytest.approx(4 / 6)   # a, b, d, f
    assert diff.candidate_accuracy == pytest.approx(3 / 6)  # a, c, d
    assert diff.accuracy_delta == pytest.approx(-1 / 6)


def test_the_accuracy_delta_only_shows_the_net_of_the_churn(state):
    """Three crops changed verdict; the accuracy moved by one crop's worth.
    Seeing the other two is the reason this tab exists."""
    diff = compare_experiments(state, "base", "cand")
    changed = diff.counts[OUTCOME_REGRESSION] + diff.counts[OUTCOME_FIX]
    assert changed == 3
    assert diff.net_change() == -1
    assert diff.accuracy_delta == pytest.approx(diff.net_change() / diff.n_common)


def test_identical_accuracy_can_still_hide_a_pile_of_churn(tmp_path):
    """The worst case for a summary table: the two numbers agree exactly
    and every single crop that changed is invisible."""
    df = pd.DataFrame({
        "uuid": list("abcd"), "crop": ["c.png"] * 4, "label": ["cat", "cat", "dog", "dog"],
    })
    df.to_csv(tmp_path / "dataset.csv", index=False)
    _write_results(tmp_path / "a.csv", df["uuid"], ["cat", "dog", "dog", "cat"])   # right on a, c
    _write_results(tmp_path / "b.csv", df["uuid"], ["dog", "cat", "cat", "dog"])   # right on b, d

    st = AppState()
    st.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    st.add_experiment(str(tmp_path / "a.csv"), "base")
    st.add_experiment(str(tmp_path / "b.csv"), "cand")

    diff = compare_experiments(st, "base", "cand")
    assert diff.baseline_accuracy == diff.candidate_accuracy == pytest.approx(0.5)
    assert diff.accuracy_delta == pytest.approx(0.0)
    assert diff.counts[OUTCOME_REGRESSION] == 2 and diff.counts[OUTCOME_FIX] == 2
    assert diff.net_change() == 0


def test_comparing_an_experiment_with_itself_is_all_agreement(state):
    diff = compare_experiments(state, "base", "base")
    assert diff.counts[OUTCOME_REGRESSION] == 0
    assert diff.counts[OUTCOME_FIX] == 0
    assert diff.accuracy_delta == pytest.approx(0.0)
    assert (diff.delta_confusion.to_numpy() == 0).all()


# ---- the intersection ---------------------------------------------------

def test_only_shared_crops_are_compared_and_the_rest_is_reported(tmp_path):
    """Counts taken over different row sets would be meaningless, so the
    overlap is explicit rather than assumed."""
    df = pd.DataFrame({
        "uuid": list("abcd"), "crop": ["c.png"] * 4, "label": ["cat"] * 4,
    })
    df.to_csv(tmp_path / "dataset.csv", index=False)
    _write_results(tmp_path / "a.csv", ["a", "b", "c"], ["cat", "cat", "cat"])
    _write_results(tmp_path / "b.csv", ["b", "c", "d"], ["cat", "cat", "dog"])

    st = AppState()
    st.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    st.add_experiment(str(tmp_path / "a.csv"), "base")
    st.add_experiment(str(tmp_path / "b.csv"), "cand")

    diff = compare_experiments(st, "base", "cand")
    assert diff.n_common == 2            # b, c
    assert diff.n_baseline_only == 1     # a
    assert diff.n_candidate_only == 1    # d


def test_an_unknown_experiment_is_refused(state):
    with pytest.raises(KeyError):
        build_comparison(state, "base", "nope")


# ---- the differential confusion matrix ----------------------------------

def test_delta_confusion_is_candidate_minus_baseline(state):
    diff = compare_experiments(state, "base", "cand")
    delta = diff.delta_confusion
    # 'b' moved out of (cat -> cat) and into (cat -> dog)
    assert delta.loc["cat", "cat"] == -1
    assert delta.loc["cat", "dog"] == +1
    # 'c' moved out of (dog -> cat) and into (dog -> dog)
    assert delta.loc["dog", "cat"] == -1
    assert delta.loc["dog", "dog"] == +1
    assert delta.to_numpy().sum() == 0, "rows only move between cells, none appear or vanish"


def test_delta_confusion_keeps_a_class_only_one_model_predicts(state):
    diff = compare_experiments(state, "base", "cand")
    assert set(diff.delta_confusion.index) == set(CLASSES)
    assert set(diff.delta_confusion.columns) == set(CLASSES)


def test_more_on_the_diagonal_is_an_improvement_not_a_regression():
    """A difference matrix colored by the raw sign would paint the single
    most important cell -- the candidate getting a class right more often --
    the same as a brand-new error."""
    assert cell_improvement("cat", "cat", +5) == +5      # 5 more correct
    assert cell_improvement("cat", "cat", -5) == -5      # 5 fewer correct
    assert cell_improvement("cat", "dog", +5) == -5      # 5 more confusions
    assert cell_improvement("cat", "dog", -5) == +5      # 5 fewer confusions


# ---- the crop lists -----------------------------------------------------

def test_regressions_are_listed_most_confident_first(state):
    """A regression the new model is 0.99 sure about is a different problem
    from one it was hesitating over."""
    joined = build_comparison(state, "base", "cand")
    rows = rows_for_outcome(joined, OUTCOME_REGRESSION)
    assert set(rows["uuid"]) == {"b", "f"}
    assert rows[COL_SCORE_B].is_monotonic_decreasing


def test_rows_for_an_outcome_with_no_matches_is_empty_not_an_error(state):
    joined = build_comparison(state, "base", "base")
    assert len(rows_for_outcome(joined, OUTCOME_REGRESSION)) == 0


def test_crop_and_label_survive_into_the_rows_for_the_carousel(state):
    """These feed the image carousel, which needs the path and the truth."""
    joined = build_comparison(state, "base", "cand")
    rows = rows_for_outcome(joined, OUTCOME_REGRESSION)
    assert "crop" in rows.columns and COL_TRUE in rows.columns
    assert rows["crop"].str.endswith(".png").all()


# ---- per-class breakdown ------------------------------------------------

def test_per_class_change_shows_a_model_that_traded_one_class_for_another(state):
    table = per_class_change(build_comparison(state, "base", "cand"), CLASSES).set_index("label")
    assert table.loc["dog", "fixes"] == 1 and table.loc["dog", "regressions"] == 0
    assert table.loc["cat", "regressions"] == 1
    assert table.loc["bird", "regressions"] == 1
    assert table["net"].sum() == -1


def test_per_class_change_skips_a_class_with_no_rows(state):
    table = per_class_change(build_comparison(state, "base", "cand"), CLASSES + ["ghost"])
    assert "ghost" not in set(table["label"])


# ---- the review overlay -------------------------------------------------

def test_the_comparison_follows_the_review_overlay(state):
    """Both models must be judged against the same corrected labels, or the
    comparison stops meaning anything."""
    from lookin.core.review import STATUS_RELABEL

    state.review.create_version("v1")
    state.review.decide("b", STATUS_RELABEL, new_label="dog")  # b's true label was cat
    state.use_reviewed_labels = True

    joined = build_comparison(state, "base", "cand")
    by_uuid = joined.set_index("uuid")
    assert by_uuid.loc["b", COL_TRUE] == "dog"
    assert by_uuid.loc["b", COL_OUTCOME] == OUTCOME_FIX, "was a regression against the wrong label"


# ---- the gross movement matrices ----------------------------------------

def test_moved_in_counts_the_crops_by_where_the_candidate_put_them(state):
    diff = compare_experiments(state, "base", "cand")
    moved_in = diff.moved_in_confusion
    assert moved_in.loc["cat", "dog"] == 1    # b: cat -> dog
    assert moved_in.loc["dog", "dog"] == 1    # c: arrived on the diagonal, i.e. was fixed
    assert moved_in.loc["bird", "dog"] == 1   # e: one wrong class for another
    assert moved_in.loc["bird", "cat"] == 1   # f
    assert moved_in.to_numpy().sum() == diff.n_moved == 4


def test_moved_out_counts_the_same_crops_by_where_the_baseline_had_them(state):
    diff = compare_experiments(state, "base", "cand")
    moved_out = diff.moved_out_confusion
    assert moved_out.loc["cat", "cat"] == 1   # b left the diagonal, i.e. was broken
    assert moved_out.loc["dog", "cat"] == 1   # c: the confusion the candidate cleared up
    assert moved_out.loc["bird", "cat"] == 1  # e
    assert moved_out.loc["bird", "bird"] == 1  # f
    # Every crop that moved is one arrival and one departure, so the two
    # gross matrices always have the same total.
    assert moved_out.to_numpy().sum() == diff.n_moved == 4


def test_moved_in_minus_moved_out_is_the_difference_matrix(state):
    """The identity that makes the three matrices one story rather than
    three: the net is what the gross flows leave behind."""
    diff = compare_experiments(state, "base", "cand")
    net = diff.moved_in_confusion - diff.moved_out_confusion
    assert net.equals(diff.delta_confusion)


def test_a_change_from_one_wrong_class_to_another_still_moves(state):
    """Crop e is wrong under both models, so no outcome changed -- but it
    left (bird -> cat) and arrived in (bird -> dog), and leaving it out
    would break the identity with the difference matrix."""
    diff = compare_experiments(state, "base", "cand")
    joined = build_comparison(state, "base", "cand")
    e = joined[joined["uuid"] == "e"].iloc[0]
    assert e[COL_OUTCOME] == OUTCOME_BOTH_WRONG
    assert diff.moved_out_confusion.loc["bird", "cat"] == 1
    assert diff.moved_in_confusion.loc["bird", "dog"] == 1


def test_a_cell_the_net_matrix_flattens_still_shows_its_traffic(state):
    """The reason three matrices exist rather than one.

    For true=bird, one crop moved out of (bird -> cat) and another moved
    in, so the difference is flatly zero -- and two moved crops are
    invisible in it.
    """
    diff = compare_experiments(state, "base", "cand")
    assert diff.delta_confusion.loc["bird", "cat"] == 0
    assert diff.moved_in_confusion.loc["bird", "cat"] == 1
    assert diff.moved_out_confusion.loc["bird", "cat"] == 1


def test_all_three_matrices_share_index_and_columns(state):
    """They are flipped between in place in the Compare tab, so they have to
    read the same way."""
    diff = compare_experiments(state, "base", "cand")
    for key in MATRIX_VIEWS:
        matrix = diff.confusion(key)
        assert list(matrix.index) == list(diff.delta_confusion.index)
        assert list(matrix.columns) == list(diff.delta_confusion.columns)
    assert diff.confusion(MATRIX_NET) is diff.delta_confusion


def test_comparing_an_experiment_with_itself_moves_nothing(state):
    """Nothing moved, so the matrix has to come back full of zeros rather
    than empty or raising."""
    joined = build_comparison(state, "base", "base")
    moved_in = movement_confusion(joined, CLASSES, MATRIX_MOVED_IN)
    assert list(moved_in.index) == CLASSES
    assert moved_in.to_numpy().sum() == 0


def test_a_direction_that_is_not_a_movement_is_refused(state):
    joined = build_comparison(state, "base", "cand")
    with pytest.raises(ValueError):
        movement_confusion(joined, CLASSES, OUTCOME_REGRESSION)


def test_rows_in_movement_cell_opens_exactly_what_arrived(state):
    joined = build_comparison(state, "base", "cand")
    rows = rows_in_movement_cell(joined, MATRIX_MOVED_IN, "bird", "cat")
    assert list(rows["uuid"]) == ["f"]
    # The crop that moved *out* of the same cell is a departure, not an
    # arrival, and must not come along.
    assert "e" not in set(rows["uuid"])
    assert list(rows_in_movement_cell(joined, MATRIX_MOVED_OUT, "bird", "cat")["uuid"]) == ["e"]


def test_rows_moved_through_a_cell_opens_both_directions(state):
    """What a cell of the difference matrix stands for: its number is zero
    here, and two crops are behind it."""
    joined = build_comparison(state, "base", "cand")
    rows = rows_moved_through_cell(joined, "bird", "cat")
    assert set(rows["uuid"]) == {"e", "f"}


# ---- the per-class flow matrix ------------------------------------------

def test_the_flow_matrix_is_baseline_against_candidate_for_one_class(state):
    joined = build_comparison(state, "base", "cand")
    flow = flow_matrix(joined, CLASSES, "bird")
    assert flow.loc["bird", "cat"] == 1   # f: the baseline had it right, the candidate says cat
    assert flow.loc["cat", "dog"] == 1    # e: one mistake traded for another
    # Crops of another true class have no business here.
    assert flow.to_numpy().sum() == 2


def test_the_flow_matrix_keeps_what_did_not_move_on_its_diagonal(state):
    """The diagonal is the scale the moves are read against -- without it a
    cell holding three moved crops looks the same whether the class has ten
    crops or ten thousand."""
    joined = build_comparison(state, "base", "cand")
    flow = flow_matrix(joined, CLASSES, "dog")
    assert flow.loc["dog", "dog"] == 1    # d, unchanged
    assert flow.loc["cat", "dog"] == 1    # c, fixed


def test_a_flow_column_is_where_the_arrivals_came_from(state):
    """The bridge between the two panels: the off-diagonal column of the
    flow matrix sums to the arrivals the gross matrix counts."""
    diff = compare_experiments(state, "base", "cand")
    joined = build_comparison(state, "base", "cand")
    flow = flow_matrix(joined, diff.labels, "bird")
    for predicted in diff.labels:
        arrivals = flow[predicted].drop(index=predicted).sum()
        departures = flow.loc[predicted].drop(index=predicted).sum()
        assert arrivals == diff.moved_in_confusion.loc["bird", predicted]
        assert departures == diff.moved_out_confusion.loc["bird", predicted]


def test_flow_labels_drop_the_classes_nothing_touches(state):
    """A hundred-class dataset gives a hundred-square flow matrix with a
    handful of non-zero cells, and a grid of zeros is not a panel."""
    joined = build_comparison(state, "base", "cand")
    flow = flow_matrix(joined, CLASSES, "dog")
    assert flow_labels(flow, "dog") == ["cat", "dog"]


def test_flow_labels_keep_the_true_class_even_when_it_is_empty(tmp_path, state):
    """An absent row would read as missing rather than as zero, and that
    row is exactly what "broke" means in this panel."""
    joined = build_comparison(state, "base", "cand")
    flow = flow_matrix(joined, CLASSES, "bird")
    flow.loc["bird"] = 0
    flow["bird"] = 0
    assert "bird" in flow_labels(flow, "bird")


def test_flow_improvement_only_signs_the_moves_that_change_the_verdict():
    assert flow_improvement("cat", "dog", "cat", 3) == 3     # became right
    assert flow_improvement("cat", "cat", "dog", 3) == -3    # became wrong
    assert flow_improvement("cat", "dog", "bird", 3) == 0    # one mistake for another
    assert flow_improvement("cat", "dog", "dog", 3) == 0     # never moved


# ---- one cell's own flows -----------------------------------------------

def test_cell_flows_count_only_what_touches_the_cell(state):
    """The difference with flow_matrix. For (bird -> cat): 'f' arrived from
    bird and 'e' left for dog. The class also holds other crops, and none
    of them belong here."""
    joined = build_comparison(state, "base", "cand")
    table = cell_flows(joined, CLASSES, "bird", "cat")
    # Order is the busiest-first rule tested below; what matters here is
    # that nothing else of the bird class got counted.
    assert set(table.index) == {"bird", "dog"}
    assert table.loc["bird", COL_CAME_FROM] == 1
    assert table.loc["bird", COL_WENT_TO] == 0
    assert table.loc["dog", COL_WENT_TO] == 1


def test_cell_flows_drop_the_classes_the_cell_never_traded_with(state):
    """On a hundred-class dataset a cell usually exchanges crops with two
    or three of them, and the rest would be a page of zeros."""
    joined = build_comparison(state, "base", "cand")
    assert "bird" not in cell_flows(joined, CLASSES, "dog", "cat").index


def test_cell_flows_add_up_to_the_two_gross_matrices(state):
    """The bridge to the matrix on the left: the panel's two columns total
    exactly the arrivals and departures that cell counts."""
    diff = compare_experiments(state, "base", "cand")
    joined = build_comparison(state, "base", "cand")
    for true_label in diff.labels:
        for predicted in diff.labels:
            table = cell_flows(joined, diff.labels, true_label, predicted)
            assert table[COL_CAME_FROM].sum() == diff.moved_in_confusion.loc[true_label, predicted]
            assert table[COL_WENT_TO].sum() == diff.moved_out_confusion.loc[true_label, predicted]


def test_cell_flows_of_a_cell_nothing_moved_through_is_empty(state):
    joined = build_comparison(state, "base", "cand")
    table = cell_flows(joined, CLASSES, "cat", "bird")
    assert table.empty
    assert list(table.columns) == [COL_CAME_FROM, COL_WENT_TO]


def test_cell_flows_put_the_busiest_class_first(tmp_path):
    """Which class a cell mostly swaps with is the first question, so it is
    the first line."""
    df = pd.DataFrame({"uuid": list("abcd"), "crop": [f"c/{u}.png" for u in "abcd"],
                       "label": ["cat"] * 4})
    df.to_csv(tmp_path / "dataset.csv", index=False)
    _write_results(tmp_path / "a.csv", df["uuid"], ["bird", "dog", "dog", "dog"])
    _write_results(tmp_path / "b.csv", df["uuid"], ["cat", "cat", "cat", "cat"])
    st = AppState()
    st.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    st.add_experiment(str(tmp_path / "a.csv"), "base")
    st.add_experiment(str(tmp_path / "b.csv"), "cand")

    joined = build_comparison(st, "base", "cand")
    table = cell_flows(joined, CLASSES, "cat", "cat")
    assert list(table.index) == ["dog", "bird"]   # three arrivals against one


def test_rows_in_flow_cell_opens_exactly_that_move(state):
    joined = build_comparison(state, "base", "cand")
    rows = rows_in_flow_cell(joined, "bird", "cat", "dog")
    assert list(rows["uuid"]) == ["e"]
    assert rows_in_flow_cell(joined, "bird", "dog", "cat").empty
