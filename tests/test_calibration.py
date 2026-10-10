"""Calibration and abstention.

Three kinds of risk are covered here:

* the threshold queries have to agree with each other. A target-accuracy
  answer that threshold_point() then contradicts is worse than no answer:
  the user reads "threshold 0.9 gives 80% coverage", types 0.9, and sees
  something else.
* the risk-coverage numbers have to be right at the two ends, where they
  are checkable by hand -- full coverage is just the error rate, and a
  model that ranks its errors perfectly has to reach the optimal AURC.
* temperature scaling must not change a single prediction. It is only
  acceptable as a calibration fix because it cannot: if it moved the
  argmax it would silently invalidate every other number in the app.
"""
from __future__ import annotations

import numpy as np
import pytest

from lookin.core.calibration import (
    build_index, confidence_histogram, fit_temperature, per_class_thresholds,
    risk_coverage, threshold_for_target, threshold_point,
)

LABELS = ["cat", "dog", "bird"]


def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---- threshold queries --------------------------------------------------

def test_a_threshold_accepts_everything_at_or_above_it():
    """`>=`, not `>`: with hard 0/1 scores the two differ by the whole
    dataset."""
    index = build_index([0.9, 0.8, 0.7], [True, True, False])
    assert threshold_point(index, 0.8).n_accepted == 2
    assert threshold_point(index, 0.9).n_accepted == 1
    assert threshold_point(index, 0.0).n_accepted == 3


def test_a_round_threshold_accepts_the_rows_sitting_exactly_on_it():
    """The index stores float32 (scores are float32 on load); a threshold
    typed as a float64 0.9 is *greater* than the float32 0.9 the model
    emitted, so an unquantised comparison rejects every row at exactly the
    threshold -- and thresholds are set to round numbers the scores are full
    of, so that is most of the data, not an edge case."""
    index = build_index(np.array([0.9, 0.9, 0.8], dtype=np.float32), [True, True, False])
    assert threshold_point(index, 0.9).n_accepted == 2
    assert threshold_point(index, 0.5).n_accepted == 3


def test_accepted_and_abstained_accuracies_split_the_rows():
    index = build_index([0.9, 0.8, 0.4, 0.3], [True, True, False, True])
    point = threshold_point(index, 0.5)
    assert point.n_accepted == 2
    assert point.accuracy == 1.0
    assert point.n_errors_accepted == 0
    assert point.n_abstained == 2
    assert point.accuracy_abstained == 0.5
    assert point.coverage == 0.5


def test_a_threshold_above_every_score_accepts_nothing_without_dividing_by_zero():
    index = build_index([0.9, 0.8], [True, False])
    point = threshold_point(index, 1.0)
    assert point.n_accepted == 0
    assert np.isnan(point.accuracy)
    assert point.n_abstained == 2


def test_target_accuracy_returns_the_largest_coverage_that_meets_it():
    """Selective accuracy is not monotone in coverage: stopping at the first
    dip would hand back a threshold that automates far less than the model
    supports."""
    # Ranked: right, wrong, right, right, right -> accuracy dips at k=2 and
    # recovers to 0.8 at k=5.
    index = build_index([0.99, 0.98, 0.97, 0.96, 0.95], [True, False, True, True, True])
    point = threshold_for_target(index, 0.8)
    assert point.n_accepted == 5
    assert point.coverage == 1.0


def test_an_unreachable_target_says_so_rather_than_returning_a_threshold():
    index = build_index([0.9, 0.8], [False, False])
    point = threshold_for_target(index, 0.99)
    assert point.unreachable is True
    assert point.n_accepted == 0


def test_the_two_threshold_queries_agree_with_each_other():
    """The headline consistency property: a threshold reported for a target
    must produce that same operating point when applied."""
    rng = np.random.default_rng(0)
    confidence = rng.random(500)
    correct = rng.random(500) < confidence  # confidence actually informative
    index = build_index(confidence, correct)
    for target in (0.7, 0.85, 0.95):
        answer = threshold_for_target(index, target)
        if answer.unreachable:
            continue
        applied = threshold_point(index, answer.threshold)
        assert applied.n_accepted == answer.n_accepted
        assert applied.accuracy == pytest.approx(answer.accuracy)


def test_tied_confidences_never_produce_a_threshold_that_cannot_be_applied():
    """A threshold cannot split rows that share a confidence, so a cut point
    inside a run of equal values has to be refused."""
    index = build_index([0.5] * 4 + [0.4] * 4, [True, True, False, False, True, True, True, True])
    answer = threshold_for_target(index, 0.6)
    applied = threshold_point(index, answer.threshold)
    assert applied.n_accepted == answer.n_accepted


# ---- risk-coverage ------------------------------------------------------

def test_risk_at_full_coverage_is_the_plain_error_rate():
    index = build_index([0.9, 0.8, 0.7, 0.6], [True, True, True, False])
    curve = risk_coverage(index)
    assert curve.coverage[-1] == 1.0
    assert curve.risk[-1] == pytest.approx(0.25)


def test_a_perfect_ranking_reaches_the_optimal_aurc():
    """Every correct prediction ranked above every mistake: nothing about
    the *ordering* can be improved, whatever the error rate is."""
    index = build_index([0.9, 0.8, 0.7, 0.6], [True, True, False, False])
    curve = risk_coverage(index)
    assert curve.aurc == pytest.approx(curve.optimal_aurc)


def test_a_worse_ranking_has_a_higher_aurc_at_the_same_accuracy():
    """Same two errors, moved to the top of the ranking -- accuracy is
    identical and AURC is the number that tells them apart."""
    good = risk_coverage(build_index([0.9, 0.8, 0.7, 0.6], [True, True, False, False]))
    bad = risk_coverage(build_index([0.9, 0.8, 0.7, 0.6], [False, False, True, True]))
    assert bad.aurc > good.aurc


def test_the_drawn_curve_is_downsampled_but_the_aurc_is_not():
    """AURC goes in a table and must not depend on how many points were
    kept for the picture."""
    rng = np.random.default_rng(1)
    index = build_index(rng.random(5000), rng.random(5000) < 0.8)
    coarse = risk_coverage(index, max_points=50)
    fine = risk_coverage(index, max_points=2000)
    assert len(coarse.coverage) <= 50
    assert coarse.aurc == pytest.approx(fine.aurc)


# ---- per-class thresholds ----------------------------------------------

def test_each_class_gets_its_own_threshold():
    predicted = np.array(["cat"] * 4 + ["dog"] * 4)
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.75, 0.65])
    correct = np.array([True, True, False, False, True, True, True, False])
    rows = {r.label: r for r in per_class_thresholds(predicted, confidence, correct, ["cat", "dog"], 1.0)}
    assert rows["cat"].n_kept == 2
    assert rows["cat"].threshold == pytest.approx(0.8)
    assert rows["dog"].n_kept == 3
    assert rows["dog"].precision == 1.0


def test_a_class_the_model_never_predicts_is_reported_not_skipped():
    rows = per_class_thresholds(
        np.array(["cat", "cat"]), np.array([0.9, 0.8]), np.array([True, True]),
        ["cat", "dog"], 0.9,
    )
    dog = [r for r in rows if r.label == "dog"][0]
    assert dog.n_predicted == 0
    assert dog.unreachable is True


# ---- confidence histogram ----------------------------------------------

def test_the_histogram_splits_correct_from_wrong_without_losing_rows():
    confidence = np.array([0.1, 0.4, 0.6, 0.9])
    correct = np.array([False, False, True, True])
    hist = confidence_histogram(confidence, correct, n_bins=4)
    assert hist.correct.sum() == 2
    assert hist.wrong.sum() == 2


# ---- temperature scaling -----------------------------------------------

def test_temperature_scaling_never_changes_a_prediction():
    """The property that makes it safe to offer at all: accuracy, the
    confusion matrix and every ranking metric are untouched."""
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(200, 3)) * 4
    scores = _softmax(logits)
    y_true = np.array(LABELS, dtype=object)[rng.integers(0, 3, 200)]
    fit = fit_temperature(scores, y_true, LABELS)
    scaled_argmax = np.argmax(_softmax(np.log(np.clip(scores, 1e-12, None)) / fit.temperature), axis=1)
    assert np.array_equal(scaled_argmax, scores.argmax(axis=1))


def test_an_overconfident_model_is_cooled_back_to_its_true_temperature():
    """Labels drawn from softmax(logits), scores reported as softmax(3 *
    logits): a model whose ranking is right and whose confidence is inflated
    by exactly a factor of three. The fit has a known answer, so this checks
    the optimiser lands on it rather than merely moving in the right
    direction."""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(8000, 3)) * 1.5
    truth = _softmax(logits)
    true_idx = np.array([rng.choice(3, p=row) for row in truth])
    scores = _softmax(logits * 3)

    fit = fit_temperature(scores, np.array(LABELS, dtype=object)[true_idx], LABELS)
    assert fit.temperature == pytest.approx(3.0, rel=0.1)
    assert fit.nll_after < fit.nll_before
    assert fit.ece_after < fit.ece_before


def test_an_already_calibrated_model_is_left_almost_alone():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(4000, 3))
    scores = _softmax(logits)
    # Labels drawn *from* the model's own distribution: perfectly calibrated
    # by construction, so the best temperature is 1.
    true_idx = np.array([rng.choice(3, p=row) for row in scores])
    fit = fit_temperature(scores, np.array(LABELS, dtype=object)[true_idx], LABELS)
    assert fit.temperature == pytest.approx(1.0, abs=0.15)


def test_a_true_label_with_no_score_column_does_not_poison_the_fit():
    scores = _softmax(np.random.default_rng(5).normal(size=(50, 3)))
    y_true = np.array(["cat"] * 49 + ["ghost"], dtype=object)
    fit = fit_temperature(scores, y_true, LABELS)
    assert fit.disabled_reason is None
    assert np.isfinite(fit.temperature)


def test_fitting_on_nothing_is_refused_rather_than_returning_a_number():
    scores = _softmax(np.random.default_rng(6).normal(size=(10, 3)))
    fit = fit_temperature(scores, np.array(["ghost"] * 10, dtype=object), LABELS)
    assert fit.disabled_reason is not None


def test_the_fit_is_capped_but_the_report_covers_every_row():
    rng = np.random.default_rng(7)
    scores = _softmax(rng.normal(size=(500, 3)))
    y_true = np.array(LABELS, dtype=object)[rng.integers(0, 3, 500)]
    fit = fit_temperature(scores, y_true, LABELS, max_fit_rows=100)
    assert fit.n_fit_rows == 100
    assert fit.n_rows == 500
    assert len(fit.scaled_confidence) == 500
