"""Classification metrics.

Two kinds of test here, for two kinds of risk:

* the metrics taken from sklearn are checked against sklearn on a case
  small enough to also verify by hand -- what is being tested is the wiring
  (which column is which class, what gets weighted by what), not sklearn;
* the three metrics written by hand because no sklearn equivalent exists
  (Brier, its Murphy decomposition, ECE/MCE) are checked against their
  defining identities and against inputs whose answer is known exactly.

The ScoreDiagnostics cases matter most of all: they are what stops the tool
reporting a log loss or a ROC-AUC that looks fine and means nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.metrics import (
    MACRO_AVERAGE, compute_curves, compute_metrics, compute_reliability, diagnose_scores,
)

LABELS = ["cat", "dog", "bird"]
SCORE_COLS = ["cat", "dog", "bird"]


def _merged(y_true, scores) -> pd.DataFrame:
    """A merged frame shaped like AppState.merged builds it."""
    scores = np.asarray(scores, dtype=float)
    df = pd.DataFrame(scores, columns=SCORE_COLS)
    df.insert(0, "label", list(y_true))
    pred_idx = scores.argmax(axis=1)
    df["predicted_label"] = [SCORE_COLS[i] for i in pred_idx]
    df["predicted_score"] = scores.max(axis=1)
    df["correct"] = df["predicted_label"] == df["label"]
    return df


# ---- score diagnostics --------------------------------------------------

def test_a_softmax_is_recognised_as_a_distribution():
    diag = diagnose_scores(np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]), LABELS)
    assert diag.normalized is True
    assert diag.n_rows_not_normalized == 0
    assert diag.constant_labels == []


def test_rows_that_do_not_sum_to_one_are_flagged():
    """Sigmoid heads and raw logits: log loss on these is not a mistake
    sklearn will raise on, it just silently means nothing."""
    diag = diagnose_scores(np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]]), LABELS)
    assert diag.normalized is False
    assert diag.n_rows_not_normalized == 2


def test_float32_rounding_still_counts_as_normalized():
    """Scores are downcast to float32 on load, so an exact softmax lands a
    few 1e-7 off 1.0 -- that must not disable every probabilistic metric."""
    scores = np.array([[0.7, 0.2, 0.1]], dtype=np.float32).astype(np.float64)
    assert diagnose_scores(scores, LABELS).normalized is True


def test_a_constant_column_is_flagged_with_its_value():
    """A threshold encoded as a constant column: argmax is still right, but
    that class has no ranking at all."""
    scores = np.array([[0.5, 0.2, 0.3], [0.5, 0.4, 0.1], [0.5, 0.1, 0.4]])
    diag = diagnose_scores(scores, LABELS)
    assert diag.constant_labels == ["cat"]
    assert diag.constant_values["cat"] == pytest.approx(0.5)


def test_diagnostics_on_an_empty_matrix_do_not_blow_up():
    diag = diagnose_scores(np.zeros((0, 3)), LABELS)
    assert diag.n_rows == 0 and diag.normalized is True


# ---- decision-level metrics --------------------------------------------

def test_a_perfect_classifier():
    merged = _merged(["cat", "dog", "bird"], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.balanced_accuracy == 1.0
    assert m.mcc == pytest.approx(1.0)
    assert m.cohen_kappa == pytest.approx(1.0)
    assert m.n_samples == 3


def test_accuracy_and_confusion_on_a_hand_checkable_case():
    # cat -> cat, cat -> dog, dog -> dog, bird -> cat  => 2/4 correct
    merged = _merged(
        ["cat", "cat", "dog", "bird"],
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]],
    )
    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    assert m.accuracy == pytest.approx(0.5)
    assert m.confusion.loc["cat", "cat"] == 1
    assert m.confusion.loc["cat", "dog"] == 1
    assert m.confusion.loc["bird", "cat"] == 1
    assert m.confusion.to_numpy().sum() == 4, "every sample lands in exactly one cell"


def test_per_class_rows_line_up_with_the_labels():
    merged = _merged(["cat", "cat", "dog"], [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    by_label = m.per_class.set_index("label")
    assert by_label.loc["cat", "support"] == 2
    assert by_label.loc["cat", "recall"] == pytest.approx(1.0)
    assert by_label.loc["dog", "precision"] == pytest.approx(1.0)


def test_metrics_agree_with_sklearn_on_random_data():
    """Guards the wiring -- which score column is which class, and what the
    macro/weighted averages are taken over."""
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    rng = np.random.default_rng(0)
    scores = rng.dirichlet(np.ones(3), size=200)
    y_true = rng.choice(LABELS, size=200)
    merged = _merged(y_true, scores)
    y_pred = merged["predicted_label"].to_numpy()

    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    assert m.accuracy == pytest.approx(accuracy_score(y_true, y_pred))
    assert m.macro_f1 == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert m.weighted_f1 == pytest.approx(f1_score(y_true, y_pred, average="weighted"))
    assert m.mcc == pytest.approx(matthews_corrcoef(y_true, y_pred))


def test_an_empty_join_returns_nan_rather_than_raising():
    """sklearn raises on empty input; from the UI that would look like a
    crash rather than 'these two CSVs share no ids'."""
    m = compute_metrics("exp", _merged([], np.zeros((0, 3))), "label", SCORE_COLS, LABELS)
    assert m.n_samples == 0
    assert np.isnan(m.accuracy) or m.accuracy == 0


# ---- what the diagnostics disable ---------------------------------------

def test_probabilistic_metrics_are_withheld_when_rows_do_not_sum_to_one():
    """Reporting sklearn's log loss here would be worse than reporting
    nothing: it ignores the false-positive term entirely."""
    merged = _merged(["cat", "dog"], [[0.9, 0.9, 0.9], [0.8, 0.9, 0.7]])
    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    assert m.score_diagnostics.normalized is False
    assert np.isnan(m.log_loss)
    assert np.isnan(m.brier_score)
    assert np.isnan(m.ece)


def test_a_constant_column_is_kept_out_of_the_macro_roc_auc():
    """Otherwise a fixed 0.5 is silently averaged into the headline number."""
    rng = np.random.default_rng(1)
    n = 120
    y_true = rng.choice(LABELS, size=n)
    scores = rng.dirichlet(np.ones(3), size=n)
    scores[:, 0] = 0.5  # 'cat' carries no ranking
    merged = _merged(y_true, scores)

    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    assert m.score_diagnostics.constant_labels == ["cat"]
    by_label = m.per_class.set_index("label")
    assert np.isnan(by_label.loc["cat", "roc_auc"]), "no ranking means no ROC-AUC"
    assert not np.isnan(m.macro_roc_auc), "the other classes still average"


# ---- the hand-written metrics -------------------------------------------

def test_murphy_decomposition_identity():
    """brier == reliability - resolution + uncertainty, which is the whole
    point of the decomposition."""
    rng = np.random.default_rng(2)
    scores = rng.dirichlet(np.ones(3), size=400)
    y_true = [LABELS[i] for i in scores.argmax(axis=1)]
    m = compute_metrics("exp", _merged(y_true, scores), "label", SCORE_COLS, LABELS)

    assert m.brier_score == pytest.approx(
        m.brier_reliability - m.brier_resolution + m.brier_uncertainty, abs=2e-2,
    )


def test_brier_score_of_a_perfectly_confident_correct_model_is_zero():
    merged = _merged(["cat", "dog"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    assert m.brier_score == pytest.approx(0.0)
    assert m.brier_reliability == pytest.approx(0.0)


def test_brier_score_of_a_perfectly_confident_wrong_model_is_two():
    """One-vs-rest summed: the true class is off by 1 and the predicted one
    is off by 1, so 1^2 + 1^2."""
    merged = _merged(["cat", "cat"], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    assert m.brier_score == pytest.approx(2.0)


def test_a_perfectly_calibrated_model_has_almost_no_calibration_error():
    """Confidence 0.8 on a set that really is right 80% of the time."""
    rng = np.random.default_rng(3)
    n = 4000
    correct = rng.random(n) < 0.8
    y_true, scores = [], []
    for is_correct in correct:
        row = [0.8, 0.1, 0.1]
        scores.append(row)
        y_true.append("cat" if is_correct else "dog")
    m = compute_metrics("exp", _merged(y_true, scores), "label", SCORE_COLS, LABELS)
    assert m.ece == pytest.approx(0.0, abs=0.02)


def test_an_overconfident_model_has_a_large_calibration_error():
    """Confidence 1.0 while being right half the time -- gap of ~0.5."""
    y_true = ["cat", "dog"] * 100
    scores = [[1.0, 0.0, 0.0]] * 200
    m = compute_metrics("exp", _merged(y_true, scores), "label", SCORE_COLS, LABELS)
    assert m.ece == pytest.approx(0.5, abs=0.02)
    assert m.mce == pytest.approx(0.5, abs=0.02)


# ---- reliability diagram ------------------------------------------------

def test_reliability_bins_sum_to_one_and_match_the_summary_ece():
    rng = np.random.default_rng(4)
    scores = rng.dirichlet(np.ones(3), size=300)
    y_true = rng.choice(LABELS, size=300)
    merged = _merged(y_true, scores)

    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    rel = compute_reliability(merged, diagnostics=m.score_diagnostics)
    assert rel.bin_fraction.sum() == pytest.approx(1.0)
    assert rel.ece == pytest.approx(m.ece), "the diagram and the table must never disagree"


def test_the_reliability_diagram_is_disabled_for_non_distributions():
    """A confidence axis built from non-probabilities reads as a calibration
    result while measuring nothing of the sort."""
    merged = _merged(["cat", "dog"], [[0.9, 0.9, 0.9], [0.8, 0.9, 0.7]])
    m = compute_metrics("exp", merged, "label", SCORE_COLS, LABELS)
    rel = compute_reliability(merged, diagnostics=m.score_diagnostics)
    assert rel.disabled_reason


# ---- curves -------------------------------------------------------------

def test_a_perfect_ranker_has_auc_one():
    y_true = ["cat"] * 50 + ["dog"] * 50
    scores = [[0.9, 0.05, 0.05]] * 50 + [[0.1, 0.85, 0.05]] * 50
    curve = compute_curves(_merged(y_true, scores), "label", SCORE_COLS, LABELS, "cat")
    assert curve.roc_auc == pytest.approx(1.0)
    assert curve.average_precision == pytest.approx(1.0)


def test_the_macro_curve_excludes_a_constant_column_and_says_so():
    rng = np.random.default_rng(5)
    n = 120
    y_true = rng.choice(LABELS, size=n)
    scores = rng.dirichlet(np.ones(3), size=n)
    scores[:, 0] = 0.5
    curve = compute_curves(_merged(y_true, scores), "label", SCORE_COLS, LABELS, MACRO_AVERAGE)
    assert curve.excluded_labels == ["cat"]
    assert not np.isnan(curve.roc_auc)


def test_a_constant_target_class_disables_its_own_curve_with_a_reason():
    rng = np.random.default_rng(6)
    n = 60
    y_true = rng.choice(LABELS, size=n)
    scores = rng.dirichlet(np.ones(3), size=n)
    scores[:, 0] = 0.5
    curve = compute_curves(_merged(y_true, scores), "label", SCORE_COLS, LABELS, "cat")
    assert curve.disabled_reason and "constant" in curve.disabled_reason
    assert np.isnan(curve.roc_auc)


def test_a_class_absent_from_the_pool_gives_no_curve():
    """One-vs-rest against an empty positive set is undefined, not zero."""
    merged = _merged(["cat", "cat"], [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]])
    curve = compute_curves(merged, "label", SCORE_COLS, LABELS, "bird")
    assert np.isnan(curve.roc_auc)


def test_curves_on_an_empty_join_do_not_raise():
    curve = compute_curves(_merged([], np.zeros((0, 3))), "label", SCORE_COLS, LABELS, MACRO_AVERAGE)
    assert np.isnan(curve.roc_auc)
