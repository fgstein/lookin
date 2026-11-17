"""compute_metrics derives its numbers from label counts and one sort per
score column rather than calling sklearn per class (core/metrics.py) -- the
difference between two minutes and five seconds at two thousand classes.
These pin every such number to what sklearn itself returns, on data with
ties, constant columns, NaN scores and absent classes."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score,
)

from lookin.core.metrics import compute_metrics, ovr_ranking


def _frame(seed: int, n: int, k: int, ties: bool = False, nan_col: bool = False, absent: bool = False):
    rng = np.random.default_rng(seed)
    labels = [f"c{i}" for i in range(k)]
    scores = rng.dirichlet(np.ones(k) * 0.5, n)
    if ties:
        scores = np.round(scores, 1) + 1e-9
        scores /= scores.sum(axis=1, keepdims=True)
    scores = scores.astype(np.float32)
    if nan_col:
        scores[3, 1] = np.nan
    true = rng.integers(0, k - 1 if absent else k, n)
    df = pd.DataFrame(scores, columns=labels)
    df["label"] = [labels[i] for i in true]
    df["predicted_label"] = [labels[i] for i in np.nan_to_num(scores, nan=-1).argmax(axis=1)]
    return df, labels


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_ovr_ranking_is_sklearn(seed, ties):
    df, labels = _frame(seed, 500, 7, ties=ties)
    scores = df[labels].to_numpy()
    y = df["label"].to_numpy()
    positives = [np.flatnonzero(y == l) for l in labels]
    auc, ap, f1_max = ovr_ranking(scores, positives, workers=2)
    for j, label in enumerate(labels):
        y_bin = (y == label).astype(int)
        assert auc[j] == pytest.approx(roc_auc_score(y_bin, scores[:, j].astype(float)), abs=1e-12)
        assert ap[j] == pytest.approx(average_precision_score(y_bin, scores[:, j].astype(float)), abs=1e-12)
        assert f1_max[j] == pytest.approx(_sklearn_f1_max(y_bin, scores[:, j].astype(float)), abs=1e-12)


def _sklearn_f1_max(y_bin, column):
    """The highest F1 over every point of sklearn's precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_bin, column)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(f1.max())


def test_ovr_ranking_skips_what_sklearn_refuses():
    df, labels = _frame(0, 200, 4, nan_col=True, absent=True)
    y = df["label"].to_numpy()
    auc, ap, f1_max = ovr_ranking(df[labels].to_numpy(), [np.flatnonzero(y == l) for l in labels])
    assert np.isnan(auc[1]) and np.isnan(ap[1]) and np.isnan(f1_max[1]), "a column holding NaN"
    assert np.isnan(auc[3]) and np.isnan(ap[3]) and np.isnan(f1_max[3]), "a class with no positive"


def test_f1_max_in_per_class_and_summary():
    df, labels = _frame(5, 600, 6, ties=True)
    m = compute_metrics("x", df, "label", labels, labels)
    y = df["label"].to_numpy()
    per_class = m.per_class.set_index("label")
    expected = {l: _sklearn_f1_max((y == l).astype(int), df[l].to_numpy(dtype=float)) for l in labels}
    for label, value in expected.items():
        assert per_class.loc[label, "f1_max"] == pytest.approx(value, abs=1e-12)
    assert m.macro_f1_max == pytest.approx(np.mean(list(expected.values())), abs=1e-12)
    support = per_class.loc[labels, "support"].to_numpy(dtype=float)
    assert m.weighted_f1_max == pytest.approx(
        np.average([expected[l] for l in labels], weights=support), abs=1e-12)


@pytest.mark.parametrize("absent", [False, True])
def test_decision_and_probabilistic_metrics_are_sklearn(absent):
    df, labels = _frame(3, 800, 9, absent=absent)
    m = compute_metrics("x", df, "label", labels, labels)
    y, p = df["label"].to_numpy(), df["predicted_label"].to_numpy()
    present = [l for l in labels if l in set(y) or l in set(p)]
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=present, zero_division=0)
    per_class = m.per_class.set_index("label").loc[present]
    np.testing.assert_allclose(per_class["precision"], precision, atol=1e-12)
    np.testing.assert_allclose(per_class["recall"], recall, atol=1e-12)
    np.testing.assert_allclose(per_class["f1"], f1, atol=1e-12)
    np.testing.assert_array_equal(per_class["support"], support)
    assert m.mcc == pytest.approx(matthews_corrcoef(y, p), abs=1e-12)
    assert m.cohen_kappa == pytest.approx(cohen_kappa_score(y, p, labels=present), abs=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        order = np.argsort(np.asarray(labels, dtype=object))
        expected = log_loss(y, df[labels].to_numpy(dtype=float)[:, order], labels=[labels[i] for i in order])
    assert m.log_loss == pytest.approx(expected, abs=1e-9)


def test_macro_curves_agree_with_per_class_sklearn():
    from sklearn.metrics import auc, roc_curve
    from lookin.core.metrics import MACRO_AVERAGE, compute_curves

    df, labels = _frame(5, 900, 8, ties=True)
    curves = compute_curves(df, "label", labels, labels, MACRO_AVERAGE)
    y = df["label"].to_numpy()
    scores = df[labels].to_numpy(dtype=float)
    per_class = [roc_curve((y == l).astype(int), scores[:, j]) for j, l in enumerate(labels)]
    grid = np.unique(np.concatenate([c[0] for c in per_class]))
    mean_tpr = np.mean([np.interp(grid, f, t) for f, t, _ in per_class], axis=0)
    assert curves.roc_auc == pytest.approx(auc(grid, mean_tpr), abs=1e-3)
    expected_ap = np.mean([average_precision_score((y == l).astype(int), scores[:, j]) for j, l in enumerate(labels)])
    assert curves.average_precision == pytest.approx(expected_ap, abs=1e-12)
    assert curves.precision[0] == 1.0 and np.all(np.diff(curves.fpr) >= 0)
