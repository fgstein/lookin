"""Classification metrics computed between true labels and predicted classes.

All functions take an already-merged DataFrame (see AppState.merged) plus the
list of score columns and class labels, and return plain dicts/DataFrames so
the UI layer can plot them however it likes.

Metric choices deliberately mirror metric.py (the torchmetrics-based
ClassificationReport meant for training-time use) so the same numbers show up
whether a model is being watched during training or explored here after the
fact. They're reimplemented in plain numpy/sklearn rather than by importing
that class, because this tool runs entirely on numpy arrays already
materialised from a merged DataFrame -- one shot over an in-memory array,
never an incremental many-small-batches loop -- which is exactly the case
where sklearn/numpy is already the fast path, and it avoids adding a torch
dependency to what is otherwise a lightweight, offline-friendly viewer.
Three metrics have no sklearn equivalent (multiclass Brier score, its
reliability/resolution/uncertainty decomposition, multiclass calibration
error) and are computed by hand below using the same formulas as metric.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

MACRO_AVERAGE = "macro-average (all classes)"

ECE_BINS = 15

PER_CLASS_COLUMNS = [
    "label", "precision", "recall", "specificity", "f1", "support", "roc_auc", "average_precision",
]

# How far a row of scores may stray from summing to 1.0 before the columns
# stop being treated as a probability distribution. Generous on purpose:
# scores are downcast to float32 on load (see data_manager._downcast_numeric),
# so an exactly-normalized softmax still lands a few 1e-7 away from 1.
ROW_SUM_TOLERANCE = 1e-3
# Spread below which a score column is treated as carrying no ranking at all.
# The motivating case is a threshold encoded as a constant column (a detector
# with no score for one class, e.g. "predict background when max(p) < 0.3",
# written as a background column filled with 0.3): argmax then reproduces the
# threshold rule exactly, so every decision-level metric is right, but the
# column has no order, so every ranking-level metric on it is a non-result --
# roc_auc_score returns exactly 0.5 and average_precision_score exactly the
# class's prevalence, whatever the model does.
CONSTANT_COLUMN_TOLERANCE = 1e-12
# Rows read at a time wherever the whole (n, k) score matrix would otherwise
# be copied, converted or expanded in one go. At a hundred thousand rows and
# two thousand classes one float64 copy of it is 1.6 GB, and the one-shot
# versions of these computations made four or five of them.
ROW_BLOCK = 8192
# Every such block is made row-major before it is worked on: a score matrix
# read off a loaded frame is column-major (data_manager builds the frame a
# column at a time), and a per-row reduction over it strides through the
# whole matrix -- a twelvefold slowdown measured at two thousand columns.
# Score columns sorted per task when ranking every class (see ovr_ranking).
COLUMN_BLOCK = 64


@dataclass
class ScoreDiagnostics:
    """What the score columns actually are, as opposed to what the
    probabilistic metrics assume they are.

    Two independent things can be false about a results CSV, and they
    disable two disjoint sets of metrics:

    * rows that don't sum to 1 (sigmoid/one-vs-rest heads, raw logits, an
      appended constant column) invalidate everything that reads a row as a
      distribution -- log loss, Brier and its decomposition, ECE/MCE and the
      reliability diagram. Note sklearn's log_loss does NOT renormalize and
      does NOT raise: it computes -mean(log p_true), which ignores the
      false-positive term entirely and is trivially beaten by a model that
      outputs 1.0 everywhere. Reporting it would be worse than reporting
      nothing.
    * a constant column invalidates everything that ranks that one class --
      its ROC-AUC, its average precision, its ROC/PR curves -- and would
      otherwise silently drag a fixed 0.5 into the macro ROC-AUC.
    """
    normalized: bool
    max_row_sum_deviation: float
    n_rows_not_normalized: int
    n_rows: int
    constant_labels: list = field(default_factory=list)
    constant_values: dict = field(default_factory=dict)  # label -> its single value


def diagnose_scores(scores: np.ndarray, labels_for_cols: list) -> ScoreDiagnostics:
    """Inspect a (n, k) score matrix for the two properties above. Cheap --
    one sum and one min/max per column -- so it runs on every metrics pass
    rather than being cached separately. Read in row blocks (see ROW_BLOCK)
    so no copy of the whole matrix is ever made."""
    if scores.size == 0:
        return ScoreDiagnostics(normalized=True, max_row_sum_deviation=0.0, n_rows_not_normalized=0, n_rows=0)

    n, k = scores.shape
    deviation = np.empty(n)
    col_min = np.full(k, np.inf)
    col_max = np.full(k, -np.inf)
    for start in range(0, n, ROW_BLOCK):
        block = scores[start:start + ROW_BLOCK]
        # Summed in float64: two thousand float32 columns accumulate enough
        # rounding to matter against ROW_SUM_TOLERANCE's neighbourhood.
        deviation[start:start + len(block)] = np.abs(block.sum(axis=1, dtype=np.float64) - 1.0)
        finite = np.isfinite(block)
        col_min = np.minimum(col_min, np.where(finite, block, np.inf).min(axis=0))
        col_max = np.maximum(col_max, np.where(finite, block, -np.inf).max(axis=0))
    # A row with a missing (NaN) score is not a distribution either: as
    # NaN it compared False to the tolerance and passed as one, and then
    # turned log loss, Brier and ECE into NaN with no banner saying why.
    deviation[np.isnan(deviation)] = np.inf
    n_off = int((deviation > ROW_SUM_TOLERANCE).sum())

    constant_labels = []
    constant_values = {}
    has_finite = np.isfinite(col_min)
    for j in np.flatnonzero(has_finite & (col_max - col_min <= CONSTANT_COLUMN_TOLERANCE)):
        label = labels_for_cols[j]
        constant_labels.append(label)
        constant_values[label] = float(col_min[j])

    return ScoreDiagnostics(
        normalized=n_off == 0,
        max_row_sum_deviation=float(deviation.max()),
        n_rows_not_normalized=n_off,
        n_rows=n,
        constant_labels=constant_labels,
        constant_values=constant_values,
    )


@dataclass
class ExperimentMetrics:
    name: str
    accuracy: float
    n_samples: int
    per_class: pd.DataFrame  # columns: see PER_CLASS_COLUMNS
    confusion: pd.DataFrame  # rows/cols indexed by label
    macro_f1: float
    macro_roc_auc: float
    # -- decision level (argmax) --
    balanced_accuracy: float  # macro recall
    weighted_f1: float
    mcc: float
    cohen_kappa: float
    # -- ranking level (threshold-free) --
    weighted_roc_auc: float
    macro_avg_precision: float
    weighted_avg_precision: float
    # -- probabilistic level (proper scoring rules + calibration) --
    log_loss: float
    brier_score: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    ece: float
    mce: float
    # Why some of the fields above may be NaN -- see ScoreDiagnostics.
    score_diagnostics: ScoreDiagnostics


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted average that skips NaN entries (and their weight) instead of
    propagating NaN, e.g. for a per-class ROC-AUC that's undefined because a
    class never appears in y_true."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    if not mask.any() or weights[mask].sum() == 0:
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _brier_and_decomposition(true_col: np.ndarray, scores: np.ndarray, extra_true_cols: list,
                              n_bins: int = ECE_BINS):
    """Multiclass Brier score plus Murphy's reliability/resolution/uncertainty
    decomposition, computed one-vs-rest per class and summed -- same formulas
    as BrierDecomposition in metric.py.

    `true_col[i]` is the score column of row i's true class (-1 when it has
    none), and `extra_true_cols` lists further (rows, column) pairs for the
    rare case of two score columns mapped to the same class -- so the
    one-hot target is never materialised, and neither is the (n, k) bin
    index: both are replaced by per-(column, bin) sums accumulated a block
    of rows at a time with np.bincount.
    """
    n, k = scores.shape
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    inner = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cells = k * n_bins
    count = np.zeros(cells)
    sum_pred = np.zeros(cells)
    sum_obs = np.zeros(cells)
    sum_sq = 0.0
    sum_true = 0.0
    n_true = 0
    for start in range(0, n, ROW_BLOCK):
        # Row-major (see ROW_BLOCK): a loaded frame's matrix is column-major.
        block = np.ascontiguousarray(scores[start:start + ROW_BLOCK])
        rows = len(block)
        # np.digitize(x, inner, right=True) == searchsorted(inner, x, "left")
        bins = np.searchsorted(inner, block, side="left")
        keys = (np.arange(k) * n_bins)[None, :] + bins
        count += np.bincount(keys.ravel(), minlength=cells)
        sum_pred += np.bincount(keys.ravel(), weights=block.ravel().astype(np.float64), minlength=cells)
        sum_sq += float(np.square(block, dtype=np.float64).sum())
        for cols in [true_col] + list(extra_true_cols):
            cols = cols[start:start + rows]
            hit = np.flatnonzero(cols >= 0)
            if not len(hit):
                continue
            c = cols[hit]
            sum_obs += np.bincount(keys[hit, c], minlength=cells)
            sum_true += float(block[hit, c].astype(np.float64).sum())
            n_true += len(hit)

    brier = (sum_sq - 2.0 * sum_true + n_true) / n
    filled = count > 0
    mean_pred = np.where(filled, sum_pred / np.where(filled, count, 1), 0.0)
    mean_obs = np.where(filled, sum_obs / np.where(filled, count, 1), 0.0)
    base_rate = (sum_obs.reshape(k, n_bins).sum(axis=1) / n)
    weight = count / n
    reliability = float((weight * (mean_pred - mean_obs) ** 2)[filled].sum())
    resolution = float((weight * (mean_obs - np.repeat(base_rate, n_bins)) ** 2)[filled].sum())
    uncertainty = float((base_rate * (1 - base_rate)).sum())
    return float(brier), reliability, resolution, uncertainty


def _each_sorted_column(scores: np.ndarray, positives: list, visit, workers: int | None = None) -> None:
    """Call visit(j, ordered, pos) for every column j with a usable class:
    `ordered` is the column sorted ascending, `pos` the scores of its
    positive rows. Skipped: columns whose positives are None, classes that
    are all or none of the rows, and columns holding NaN or inf (which
    sklearn refuses too). Columns are sorted a block at a time on a thread
    pool -- numpy releases the GIL while it sorts."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    n, k = scores.shape
    wanted = [j for j in range(k) if positives[j] is not None and 0 < len(positives[j]) < n]

    def run(columns: list) -> None:
        block = np.ascontiguousarray(scores[:, columns].T)
        block.sort(axis=1)
        for i, j in enumerate(columns):
            ordered = block[i]
            # NaN sorts last and -inf first.
            if not (np.isfinite(ordered[0]) and np.isfinite(ordered[-1])):
                continue
            visit(j, ordered, scores[positives[j], j])

    chunks = [wanted[i:i + COLUMN_BLOCK] for i in range(0, len(wanted), COLUMN_BLOCK)]
    workers = workers or min(8, os.cpu_count() or 1)
    if workers <= 1 or len(chunks) <= 1:
        for chunk in chunks:
            run(chunk)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, chunks))


def _positive_thresholds(ordered: np.ndarray, pos: np.ndarray) -> tuple:
    """For each distinct score of the positives, highest first: how many
    positives score exactly that, how many score at least that (tp), and
    how many rows of either kind score at least / strictly more than it."""
    n = len(ordered)
    values, counts = np.unique(pos, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    tp = np.cumsum(counts)
    at_least = n - np.searchsorted(ordered, values, side="left")
    above = n - np.searchsorted(ordered, values, side="right")
    return counts, tp, at_least, above


def ovr_ranking(scores: np.ndarray, positives: list, workers: int | None = None) -> tuple:
    """One-vs-rest ROC-AUC and average precision for many classes at once.

    `positives[j]` is None to skip column j, else the row indices whose
    true class column j scores. Returns (auc, ap), NaN where skipped (see
    _each_sorted_column).

    Exactly sklearn's roc_auc_score and average_precision_score, ties
    included, from one sort of each column: ROC-AUC is the Mann-Whitney
    statistic on average ranks (what the trapezoid under a tied ROC curve
    comes to), and average precision sums, over the distinct scores of the
    positives, the recall gained times the precision at that threshold --
    which needs only how many rows score at least that much. sklearn
    builds both full curves per class instead, twice, which at two thousand
    classes is the better part of two minutes.
    """
    n, k = scores.shape
    auc_out = np.full(k, np.nan)
    ap_out = np.full(k, np.nan)

    def visit(j, ordered, pos):
        n_pos = len(pos)
        below = np.searchsorted(ordered, pos, side="left")
        upto = np.searchsorted(ordered, pos, side="right")
        rank_sum = float((below + (upto - below + 1) / 2.0).sum())
        auc_out[j] = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * (n - n_pos))
        counts, tp, at_least, _above = _positive_thresholds(ordered, pos)
        ap_out[j] = float((counts / n_pos * tp / at_least).sum())

    _each_sorted_column(scores, positives, visit, workers)
    return auc_out, ap_out


def ovr_curves(scores: np.ndarray, positives: list, workers: int | None = None) -> dict:
    """Per-class ROC and precision-recall curves, one-vs-rest, from the
    same single sort per column as ovr_ranking: {column: (fpr, tpr, recall,
    precision, ap)}.

    The ROC curve is its corners only -- for each distinct positive score,
    the point before its crops are let in and the point after -- which is
    what sklearn's roc_curve keeps with drop_intermediate, so interpolating
    either gives the same curve. The precision-recall curve is the
    (recall, precision) pair at each of those thresholds, from recall 0 at
    precision 1: the points average precision is summed over.
    """
    n, _k = scores.shape
    out: dict = {}

    def visit(j, ordered, pos):
        n_pos = len(pos)
        n_neg = n - n_pos
        counts, tp, at_least, above = _positive_thresholds(ordered, pos)
        tp_before = tp - counts
        fp_before = above - tp_before
        fp_after = at_least - tp
        fpr = np.concatenate([[0.0], np.column_stack([fp_before, fp_after]).ravel() / n_neg, [1.0]])
        tpr = np.concatenate([[0.0], np.column_stack([tp_before, tp]).ravel() / n_pos, [1.0]])
        recall = np.concatenate([[0.0], tp / n_pos])
        precision = np.concatenate([[1.0], tp / at_least])
        ap = float((counts / n_pos * tp / at_least).sum())
        out[j] = (fpr, tpr, recall, precision, ap)

    _each_sorted_column(scores, positives, visit, workers)
    return out


def _codes(values: np.ndarray, index: dict) -> np.ndarray:
    """values -> their position in `index` (label -> int), -1 if absent."""
    return pd.Series(values, dtype=object).map(index).fillna(-1).to_numpy(dtype=np.int64)


def calibration_error(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, n_bins: int = ECE_BINS):
    """Top-label ECE/MCE: bin by predicted confidence (max probability),
    compare each bin's mean confidence to its mean accuracy. Same definition
    as torchmetrics' MulticlassCalibrationError.

    Public because core/calibration.py measures the same thing before and
    after temperature scaling, and a second implementation of the binning
    would let the Calibration tab and the Summary table disagree about the
    ECE of the same experiment."""
    if len(y_true) == 0:
        return float("nan"), float("nan")
    return calibration_error_from_confidence(scores.max(axis=1), y_true == y_pred, n_bins)


def calibration_error_from_confidence(confidence: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS):
    """calibration_error from the top-label confidence of each row and
    whether it was right -- for callers that have those and not a score
    matrix (the temperature-scaled confidence, see core/calibration.py)."""
    n = len(confidence)
    if n == 0:
        return float("nan"), float("nan")
    confidence = np.asarray(confidence).astype(np.float64)
    correct = np.asarray(correct).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        in_bin = bin_idx == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        gap = abs(confidence[in_bin].mean() - correct[in_bin].mean())
        ece += (count / n) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _matthews(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sklearn's matthews_corrcoef, from label counts: its multiclass form
    only needs each class's true and predicted totals and the trace."""
    values = pd.unique(pd.concat([pd.Series(y_true, dtype=object), pd.Series(y_pred, dtype=object)]))
    index = {v: i for i, v in enumerate(values)}
    t, p = _codes(y_true, index), _codes(y_pred, index)
    n = float(len(t))
    if not n:
        return float("nan")
    t_sum = np.bincount(t, minlength=len(values)).astype(np.float64)
    p_sum = np.bincount(p, minlength=len(values)).astype(np.float64)
    n_correct = float((t == p).sum())
    cov_ytyp = n_correct * n - float(t_sum @ p_sum)
    cov_ypyp = n * n - float(p_sum @ p_sum)
    cov_ytyt = n * n - float(t_sum @ t_sum)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    return float(cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp))


def _cohen_kappa(cm: np.ndarray) -> float:
    """sklearn's cohen_kappa_score (unweighted) from the confusion matrix:
    observed and expected disagreement are its total minus a trace."""
    total = float(cm.sum())
    if not total:
        return float("nan")
    observed = total - float(np.trace(cm))
    expected = total - float(cm.sum(axis=0).astype(np.float64) @ cm.sum(axis=1).astype(np.float64)) / total
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(1 - observed / expected) if expected else float("nan")


def compute_metrics(name: str, merged: pd.DataFrame, label_col: str, score_cols: list, class_labels: list) -> ExperimentMetrics:
    if len(merged) == 0:
        # No rows survived the dataset<->experiment join -- most commonly a
        # uuid mismatch (wrong column picked, or the two CSVs just don't
        # share any ids). sklearn's metric functions raise ValueError on
        # empty input, so this has to be short-circuited here rather than
        # letting that exception propagate: it would look, from the UI's
        # perspective, like the "Computing metrics" step never finishes.
        empty_per_class = pd.DataFrame(columns=PER_CLASS_COLUMNS)
        nan = float("nan")
        return ExperimentMetrics(
            name=name, accuracy=nan, n_samples=0,
            per_class=empty_per_class, confusion=pd.DataFrame(),
            macro_f1=nan, macro_roc_auc=nan,
            balanced_accuracy=nan, weighted_f1=nan, mcc=nan, cohen_kappa=nan,
            weighted_roc_auc=nan, macro_avg_precision=nan, weighted_avg_precision=nan,
            log_loss=nan, brier_score=nan, brier_reliability=nan, brier_resolution=nan, brier_uncertainty=nan,
            ece=nan, mce=nan,
            score_diagnostics=ScoreDiagnostics(
                normalized=True, max_row_sum_deviation=0.0, n_rows_not_normalized=0, n_rows=0,
            ),
        )

    labels_for_cols = class_labels[: len(score_cols)]
    y_true = merged[label_col].to_numpy()
    y_pred = merged["predicted_label"].to_numpy()
    # As stored (float32 after load): no float64 copy of the whole matrix.
    # Everything summed from it below accumulates in float64.
    scores = merged[score_cols].to_numpy()
    if not np.issubdtype(scores.dtype, np.floating):
        scores = scores.astype(np.float64)
    diagnostics = diagnose_scores(scores, labels_for_cols)
    constant_labels = set(diagnostics.constant_labels)

    # Every label as an integer once, instead of one pass over the rows per
    # class per metric -- two thousand of them at a hundred thousand rows.
    true_values = set(pd.unique(pd.Series(y_true, dtype=object)))
    pred_values = set(pd.unique(pd.Series(y_pred, dtype=object)))
    present_labels = [l for l in labels_for_cols if l in true_values or l in pred_values]
    if not present_labels:
        present_labels = sorted(v for v in true_values if v == v)
    present_labels = list(dict.fromkeys(present_labels))  # a label mapped to two columns counts once
    present_index = {l: i for i, l in enumerate(present_labels)}
    n_present = len(present_labels)
    t = _codes(y_true, present_index)
    p = _codes(y_pred, present_index)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    # precision_recall_fscore_support(labels=present_labels, zero_division=0):
    # its per-label counts run over every row, not only rows whose true
    # label is in `labels`, hence three bincounts rather than the matrix.
    true_sum = np.bincount(t[t >= 0], minlength=n_present).astype(np.float64)
    pred_sum = np.bincount(p[p >= 0], minlength=n_present).astype(np.float64)
    tp_sum = np.bincount(t[(t >= 0) & (t == p)], minlength=n_present).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(pred_sum > 0, tp_sum / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp_sum / true_sum, 0.0)
        f1 = np.where(true_sum + pred_sum > 0, 2 * tp_sum / (true_sum + pred_sum), 0.0)
    support = true_sum.astype(np.int64)

    both = (t >= 0) & (p >= 0)
    cm = np.bincount(t[both] * n_present + p[both], minlength=n_present * n_present).reshape(n_present, n_present)
    confusion = pd.DataFrame(cm, index=present_labels, columns=present_labels)

    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    with np.errstate(invalid="ignore", divide="ignore"):
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)

    # Ranking metrics, one-vs-rest, for every class with a score column that
    # varies and at least one positive and one negative row.
    first_col = {}
    for j, label in enumerate(labels_for_cols):
        first_col.setdefault(label, j)
    col_code = _codes(y_true, first_col)
    order = np.argsort(col_code, kind="stable")
    bounds = np.searchsorted(col_code[order], np.arange(len(labels_for_cols) + 1))
    positives = [None] * len(labels_for_cols)
    for label in present_labels:
        j = first_col.get(label)
        if j is None or label in constant_labels:
            continue
        positives[j] = order[bounds[j]:bounds[j + 1]]
    col_auc, col_ap = ovr_ranking(scores, positives)
    roc_aucs = [float(col_auc[first_col[l]]) if l in first_col else float("nan") for l in present_labels]
    avg_precisions = [float(col_ap[first_col[l]]) if l in first_col else float("nan") for l in present_labels]

    per_class = pd.DataFrame({
        "label": present_labels,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "support": support,
        "roc_auc": roc_aucs,
        "average_precision": avg_precisions,
    })

    macro_f1 = float(np.nanmean(f1)) if len(f1) else float("nan")
    macro_roc_auc = float(np.nanmean(roc_aucs)) if roc_aucs and not np.all(np.isnan(roc_aucs)) else float("nan")
    weighted_roc_auc = _weighted_nanmean(roc_aucs, support)
    macro_avg_precision = float(np.nanmean(avg_precisions)) if avg_precisions and not np.all(np.isnan(avg_precisions)) else float("nan")
    weighted_avg_precision = _weighted_nanmean(avg_precisions, support)

    balanced_accuracy = float(np.nanmean(recall)) if len(recall) else float("nan")
    weighted_f1 = _weighted_nanmean(f1, support)

    mcc = _matthews(y_true, y_pred)
    kappa = _cohen_kappa(cm)

    if diagnostics.normalized:
        # log_loss(y_true, scores, labels=labels_for_cols) over the rows whose
        # true label has a score column: sklearn clips to [eps, 1 - eps] at
        # the matrix's float precision and averages -log p_true -- which
        # only ever reads one entry per row.
        has_col = col_code >= 0
        if has_col.any():
            eps = np.finfo(np.float64).eps
            picked = scores[np.flatnonzero(has_col), col_code[has_col]].astype(np.float64)
            ll = float(-np.log(np.clip(picked, eps, 1 - eps)).mean())
        else:
            ll = float("nan")

        # Two columns mapped to the same class both carry the one-hot 1.
        extra = [np.where(col_code == first_col[label], j, -1)
                 for j, label in enumerate(labels_for_cols) if first_col[label] != j]
        brier, reliability, resolution, uncertainty = _brier_and_decomposition(col_code, scores, extra)
        ece, mce = calibration_error(y_true, y_pred, scores)
    else:
        # Rows aren't a probability distribution, so none of these mean what
        # their name says -- see ScoreDiagnostics for why reporting them
        # anyway would be actively misleading rather than merely imprecise.
        ll = brier = reliability = resolution = uncertainty = ece = mce = float("nan")

    return ExperimentMetrics(
        name=name,
        accuracy=accuracy,
        n_samples=len(merged),
        per_class=per_class,
        confusion=confusion,
        macro_f1=macro_f1,
        macro_roc_auc=macro_roc_auc,
        balanced_accuracy=balanced_accuracy,
        weighted_f1=weighted_f1,
        mcc=mcc,
        cohen_kappa=kappa,
        weighted_roc_auc=weighted_roc_auc,
        macro_avg_precision=macro_avg_precision,
        weighted_avg_precision=weighted_avg_precision,
        log_loss=ll,
        brier_score=brier,
        brier_reliability=reliability,
        brier_resolution=resolution,
        brier_uncertainty=uncertainty,
        ece=ece,
        mce=mce,
        score_diagnostics=diagnostics,
    )


@dataclass
class ReliabilityData:
    """Per-confidence-bin accuracy vs. mean predicted confidence -- the
    decomposition ECE/MCE (see calibration_error above) are computed from,
    exposed separately so the UI can plot the actual reliability diagram
    rather than just the two summary numbers."""
    bin_centers: np.ndarray    # (n_bins,) center of each confidence bin
    bin_confidence: np.ndarray  # (n_bins,) mean predicted confidence in the bin, NaN if empty
    bin_accuracy: np.ndarray    # (n_bins,) mean accuracy in the bin, NaN if empty
    bin_fraction: np.ndarray    # (n_bins,) fraction of all samples falling in the bin
    ece: float
    mce: float
    # Set when the diagram was not computed at all -- the scores aren't a
    # probability distribution, so "accuracy vs. confidence" has no meaning.
    disabled_reason: str | None = None


def compute_reliability(merged: pd.DataFrame, n_bins: int = ECE_BINS,
                        diagnostics: ScoreDiagnostics | None = None) -> ReliabilityData:
    """Top-label reliability diagram data, computed directly from the
    predicted_score/correct columns compute_metrics already relies on (see
    AppState.merged) -- same binning as calibration_error, so the ECE/MCE
    shown here always match the summary table.

    Pass `diagnostics` (from the matching ExperimentMetrics) to have the
    diagram disabled in exactly the cases where ECE/MCE are: a confidence
    axis built from scores that aren't a distribution reads as a calibration
    result while measuring nothing of the sort."""
    if diagnostics is not None and not diagnostics.normalized:
        nan_bins = np.full(n_bins, float("nan"))
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        return ReliabilityData(
            (edges[:-1] + edges[1:]) / 2, nan_bins, nan_bins, np.zeros(n_bins),
            float("nan"), float("nan"),
            disabled_reason="scores do not sum to 1, so predicted confidence is not a probability",
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    n = len(merged)
    if n == 0:
        nan_bins = np.full(n_bins, float("nan"))
        return ReliabilityData(centers, nan_bins, nan_bins, np.zeros(n_bins), float("nan"), float("nan"))

    confidence = merged["predicted_score"].to_numpy(dtype=float)
    correct = merged["correct"].to_numpy().astype(float)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    bin_confidence = np.full(n_bins, float("nan"))
    bin_accuracy = np.full(n_bins, float("nan"))
    bin_fraction = np.zeros(n_bins)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        in_bin = bin_idx == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_confidence[b] = confidence[in_bin].mean()
        bin_accuracy[b] = correct[in_bin].mean()
        bin_fraction[b] = count / n
        gap = abs(bin_confidence[b] - bin_accuracy[b])
        ece += bin_fraction[b] * gap
        mce = max(mce, gap)

    return ReliabilityData(centers, bin_confidence, bin_accuracy, bin_fraction, float(ece), float(mce))


@dataclass
class CurveData:
    fpr: np.ndarray
    tpr: np.ndarray
    roc_auc: float
    recall: np.ndarray
    precision: np.ndarray
    average_precision: float
    # Set when the target class has nothing rankable (a constant score
    # column): the curve would be a straight line through one real operating
    # point, with AUC exactly 0.5 and AP exactly the class's prevalence.
    disabled_reason: str | None = None
    # Classes dropped from a macro-average for that same reason.
    excluded_labels: list = field(default_factory=list)


def compute_curves(merged: pd.DataFrame, label_col: str, score_cols: list, class_labels: list, target: str) -> CurveData:
    """ROC and Precision-Recall curve points for one class (one-vs-rest), or
    the macro-average across every class present in class_labels.

    The macro-average interpolates each class's curve on a shared axis and
    averages. Where a curve has several points at one x -- the vertical
    steps of a ROC curve, the flat runs of a PR curve -- which of them an
    interpolation lands on used to depend on sklearn's point list and an
    unstable sort; the curves now come from ovr_curves, whose points are
    the corners (ROC) and the thresholds average precision is summed over
    (PR). The macro ROC-AUC moves by a few 1e-6 from that; the average
    precision is unchanged."""
    if len(merged) == 0:
        # Same empty-join case as compute_metrics() above.
        zeros = np.array([0.0, 1.0])
        return CurveData(zeros, zeros, float("nan"), zeros, zeros, float("nan"))

    labels_for_cols = class_labels[: len(score_cols)]
    y_true = merged[label_col].to_numpy()
    scores = merged[score_cols].to_numpy()
    constant_labels = set(diagnose_scores(scores, labels_for_cols).constant_labels)

    if target == MACRO_AVERAGE:
        # Every class's curves from one sort per score column (ovr_curves)
        # rather than three sklearn sweeps per class.
        first_col: dict = {}
        for j, label in enumerate(labels_for_cols):
            first_col.setdefault(label, j)
        col_code = _codes(y_true, first_col)
        order = np.argsort(col_code, kind="stable")
        bounds = np.searchsorted(col_code[order], np.arange(len(labels_for_cols) + 1))
        positives: list = [None] * len(labels_for_cols)
        for label, j in first_col.items():
            if label not in constant_labels:
                positives[j] = order[bounds[j]:bounds[j + 1]]
        curves = ovr_curves(scores, positives)
        per_label = [l for l in labels_for_cols if first_col[l] in curves]
        excluded = [l for l in labels_for_cols if l in constant_labels]
        if not per_label:
            zeros = np.array([0.0, 1.0])
            return CurveData(
                zeros, zeros, float("nan"), zeros, zeros, float("nan"),
                disabled_reason="no class has a score column that varies",
                excluded_labels=excluded,
            )
        chosen = [curves[first_col[l]] for l in per_label]

        all_fpr = np.unique(np.concatenate([c[0] for c in chosen]))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr_l, tpr_l, _rec, _prec, _ap in chosen:
            mean_tpr += np.interp(all_fpr, fpr_l, tpr_l)
        mean_tpr /= len(chosen)
        roc_auc_val = float(auc(all_fpr, mean_tpr))

        common_recall = np.linspace(0, 1, 200)
        mean_precision = np.zeros_like(common_recall)
        for _fpr, _tpr, rec_l, prec_l, _ap in chosen:
            mean_precision += np.interp(common_recall, rec_l, prec_l)
        mean_precision /= len(chosen)
        return CurveData(all_fpr, mean_tpr, roc_auc_val, common_recall, mean_precision,
                         float(np.mean([c[4] for c in chosen])), excluded_labels=excluded)

    if target not in labels_for_cols:
        zeros = np.array([0.0, 1.0])
        return CurveData(zeros, zeros, float("nan"), zeros, zeros, float("nan"))

    if target in constant_labels:
        zeros = np.array([0.0, 1.0])
        return CurveData(
            zeros, zeros, float("nan"), zeros, zeros, float("nan"),
            disabled_reason=f"the '{target}' score column is constant, so there is nothing to rank",
        )

    col_idx = labels_for_cols.index(target)
    y_bin = (y_true == target).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        zeros = np.array([0.0, 1.0])
        return CurveData(zeros, zeros, float("nan"), zeros, zeros, float("nan"))

    fpr, tpr, _ = roc_curve(y_bin, scores[:, col_idx])
    roc_auc_val = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(y_bin, scores[:, col_idx])
    ap = float(average_precision_score(y_bin, scores[:, col_idx]))
    return CurveData(fpr, tpr, roc_auc_val, recall, precision, ap)
