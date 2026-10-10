"""Per-prediction confidence: margin and softmax entropy (core/confidence.py).

Two things matter more than the arithmetic itself:

* whether softmax gets applied at all has to follow diagnose_scores(), not
  run unconditionally -- softmax-ing an already-normalized row flattens
  real differences toward uniform, which would silently understate how
  sure a well-calibrated model actually was.
* "confidently wrong" has to threshold against the confidence of *every*
  prediction, not just the wrong ones, or "top 25% most confident" would
  mean a different thing depending on how error-prone the pool already is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.confidence import (
    ENTROPY_COL, MARGIN_COL, NORMALIZED_ENTROPY_COL, ConfidenceData, compute_confidence, confident_and_wrong,
)

LABELS = ["a", "b", "c"]


def _merged(scores, predicted_label, true_label):
    df = pd.DataFrame(np.asarray(scores, dtype=float), columns=LABELS)
    df["label"] = true_label
    df["predicted_label"] = predicted_label
    df["correct"] = df["predicted_label"] == df["label"]
    return df


# ---- margin ---------------------------------------------------------------

def test_margin_is_the_gap_between_the_top_two_scores():
    merged = _merged([[0.9, 0.08, 0.02]], ["a"], ["a"])
    data = compute_confidence(merged, LABELS, LABELS)
    assert data.df[MARGIN_COL].iloc[0] == pytest.approx(0.82)


def test_margin_is_zero_on_an_exact_tie():
    merged = _merged([[0.5, 0.5, 0.0]], ["a"], ["a"])
    data = compute_confidence(merged, LABELS, LABELS)
    assert data.df[MARGIN_COL].iloc[0] == pytest.approx(0.0)


def test_margin_does_not_care_which_class_is_second_vs_third():
    """Only the top two matter -- moving mass between the two losing
    classes must not move the margin at all."""
    merged = _merged([[0.7, 0.2, 0.1], [0.7, 0.1, 0.2]], ["a", "a"], ["a", "a"])
    data = compute_confidence(merged, LABELS, LABELS)
    assert data.df[MARGIN_COL].iloc[0] == pytest.approx(data.df[MARGIN_COL].iloc[1])


# ---- entropy: normalized vs. raw logits -----------------------------------

def test_entropy_uses_the_raw_scores_directly_when_already_normalized():
    """A genuine softmax export (rows sum to 1) must not be pushed through
    softmax a second time -- that would flatten it toward uniform and
    understate how sure the model actually was."""
    merged = _merged([[0.98, 0.01, 0.01]], ["a"], ["a"])
    data = compute_confidence(merged, LABELS, LABELS)
    assert data.score_diagnostics.normalized
    assert not data.softmax_applied
    expected = -np.sum([0.98, 0.01, 0.01] * np.log([0.98, 0.01, 0.01]))
    assert data.df[ENTROPY_COL].iloc[0] == pytest.approx(expected, abs=1e-6)


def test_entropy_applies_softmax_to_raw_logits():
    """Rows that don't sum to 1 are logits, not probabilities -- entropy on
    the raw numbers would be meaningless, so softmax runs first."""
    merged = _merged([[1.0, 1.0, 1.0]], ["a"], ["a"])  # uniform logits
    data = compute_confidence(merged, LABELS, LABELS)
    assert not data.score_diagnostics.normalized
    assert data.softmax_applied
    # softmax(uniform logits) is itself uniform -> maximum entropy -> 1.0
    # once normalized by log(n_classes).
    assert data.df[NORMALIZED_ENTROPY_COL].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_normalized_entropy_is_near_zero_for_a_near_certain_prediction():
    merged = _merged([[0.999, 0.0005, 0.0005]], ["a"], ["a"])
    data = compute_confidence(merged, LABELS, LABELS)
    assert data.df[NORMALIZED_ENTROPY_COL].iloc[0] < 0.05


def test_a_single_class_experiment_does_not_crash():
    """No second score to compare against -- margin and normalized entropy
    are undefined, not zero, and must not raise."""
    df = pd.DataFrame({"a": [1.0, 1.0]})
    df["label"] = ["a", "a"]
    df["predicted_label"] = ["a", "a"]
    df["correct"] = True
    data = compute_confidence(df, ["a"], ["a"])
    assert data.df[MARGIN_COL].isna().all()
    assert data.df[NORMALIZED_ENTROPY_COL].isna().all()


def test_empty_dataframe_does_not_crash():
    df = pd.DataFrame({"a": [], "b": [], "label": [], "predicted_label": [], "correct": []})
    data = compute_confidence(df, ["a", "b"], ["a", "b"])
    assert len(data.df) == 0


# ---- confident and wrong ----------------------------------------------

def _confidence_data(margins, entropies, correct) -> ConfidenceData:
    """A ConfidenceData built directly from already-computed columns, so
    confident_and_wrong's thresholding logic is tested independently of
    compute_confidence's own arithmetic."""
    df = pd.DataFrame({
        MARGIN_COL: margins,
        NORMALIZED_ENTROPY_COL: entropies,
        "correct": correct,
        "label": ["a"] * len(margins),
        "predicted_label": ["b" if not c else "a" for c in correct],
    })
    return ConfidenceData(df=df, score_diagnostics=None, softmax_applied=False)


def test_confident_and_wrong_keeps_only_wrong_rows_above_the_margin_threshold():
    # Margins: 0.9 (right), 0.8 (wrong), 0.1 (wrong), 0.05 (right).
    data = _confidence_data(
        margins=[0.9, 0.8, 0.1, 0.05], entropies=[0.1, 0.2, 0.9, 0.95],
        correct=[True, False, False, True],
    )
    # Top 50% by margin means threshold at the median (0.45) -- only the
    # 0.8-margin wrong row clears it.
    out = confident_and_wrong(data, "label", "margin", top_percentile=50.0)
    assert list(out[MARGIN_COL]) == [0.8]


def test_confident_and_wrong_keeps_only_wrong_rows_below_the_entropy_threshold():
    """Entropy runs the opposite direction from margin: low entropy is
    confident, so 'most confident' means the *lowest* entropy rows."""
    data = _confidence_data(
        margins=[0.9, 0.8, 0.1, 0.05], entropies=[0.05, 0.1, 0.9, 0.95],
        correct=[True, False, False, True],
    )
    out = confident_and_wrong(data, "label", "entropy", top_percentile=50.0)
    assert list(out[NORMALIZED_ENTROPY_COL]) == [0.1]


def test_confident_and_wrong_at_100_percent_keeps_every_wrong_row():
    data = _confidence_data(
        margins=[0.9, 0.1], entropies=[0.1, 0.9], correct=[False, False],
    )
    out = confident_and_wrong(data, "label", "margin", top_percentile=100.0)
    assert len(out) == 2


def test_confident_and_wrong_never_includes_a_correct_row():
    """However low the threshold, a correct prediction is never a failure
    mode -- it must never show up here regardless of how confident it was."""
    data = _confidence_data(
        margins=[0.99], entropies=[0.01], correct=[True],
    )
    out = confident_and_wrong(data, "label", "margin", top_percentile=100.0)
    assert out.empty


def test_confident_and_wrong_on_an_empty_frame_does_not_crash():
    data = _confidence_data(margins=[], entropies=[], correct=[])
    out = confident_and_wrong(data, "label", "margin", top_percentile=25.0)
    assert out.empty
