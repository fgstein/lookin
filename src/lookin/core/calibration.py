"""Confidence: whether it can be believed (calibration) and what can be
done with it (abstention).

Two different questions, deliberately in one module because they share the
same two arrays -- the predicted confidence and whether the prediction was
right -- and because answering only the first one is a trap. A model can be
badly calibrated and still perfectly usable behind a threshold (its
confidence orders the predictions correctly, it just reports the wrong
number), and it can be beautifully calibrated and useless (well-calibrated
at 0.5 everywhere). Calibration is about the *value* of the confidence;
abstention is about its *order*.

What needs a probability and what does not
------------------------------------------
This distinction runs through the whole module and decides what stays
enabled on a results CSV whose rows do not sum to 1 (see
metrics.ScoreDiagnostics):

* ECE/MCE, the reliability diagram and temperature scaling read the
  confidence as a probability. On a sigmoid head or raw logits they measure
  nothing, and are refused rather than shown.
* the risk-coverage curve, every threshold query and the per-class
  thresholds only ever *rank* by confidence and then count outcomes. They
  are empirical, they stay correct for any monotone score, and they stay
  enabled -- the accepted accuracy at a threshold is a fact about the rows,
  whatever the score is called.

Selective prediction, in one paragraph
--------------------------------------
Accept a prediction when its confidence is at or above a threshold,
otherwise abstain and send the crop to a human. Coverage is the share
accepted; selective risk is the error rate *among those accepted*. Sweeping
the threshold traces the risk-coverage curve, and the area under it (AURC)
is a single number for "how well does this model know what it doesn't
know" -- lower is better, and it is comparable between models in a way a
single threshold never is, since two models rarely put their errors at the
same confidence.

Cost
----
Every threshold query is answered from one sorted array plus its cumulative
correct-count (see ThresholdIndex), so a slider sweeping thresholds over
several experiments is a binary search each, not a pass over the rows. That
index costs 8 bytes per row per experiment -- about 4 MB for a 500k-row
experiment, next to the ~100 MB its merged frame already occupies -- and
callers are expected to drop it when the data changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lookin.core.metrics import ECE_BINS, calibration_error, calibration_error_from_confidence

def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


# Points kept when a curve is handed to matplotlib. A risk-coverage curve
# has one point per row; drawing 500k of them costs seconds and renders
# exactly like 500 do.
CURVE_POINTS = 500

# Bounds of the temperature search. 0.05 and 20 bracket anything a real
# network produces by a wide margin (an over-confident classifier lands
# around 1.5-3).
T_MIN = 0.05
T_MAX = 20.0
# The search runs on beta = 1/T, where the loss is convex (see
# fit_temperature), and stops once beta is known to this much.
BETA_TOLERANCE = 1e-9
# Rows per task in the temperature passes. Smaller than metrics.ROW_BLOCK:
# these run eight at a time, each with a few float64 temporaries of
# (block x classes), and at two thousand classes a full ROW_BLOCK apiece
# would hold several GB at once.
T_BLOCK = 2048
# Rows sampled to fit the temperature. The fit is a 1-D optimisation over a
# scalar; 200k rows already pin it to more decimals than anyone reads, and
# the full pass is then only paid twice (before/after), not eighty times.
T_FIT_MAX_ROWS = 200_000


@dataclass
class ThresholdIndex:
    """Confidences sorted high to low, with the running count of correct
    predictions above each position -- everything the threshold queries
    below need, precomputed once per experiment.

    `cum_correct[k]` is how many of the k most confident predictions were
    right, so accepting the top k costs one array lookup, and finding the k
    for a given threshold costs one binary search.
    """
    confidence: np.ndarray    # descending, float32
    cum_correct: np.ndarray   # length n+1, int32, cum_correct[0] == 0
    n: int

    @property
    def n_correct(self) -> int:
        return int(self.cum_correct[-1])


def build_index(confidence: np.ndarray, correct: np.ndarray) -> ThresholdIndex:
    """Sort once so every later query is a lookup.

    float32/int32 rather than the numpy defaults because this is the one
    structure a caller is expected to keep for every selected experiment at
    once: 8 bytes per row instead of 16 halves what a tab holding twenty
    500k-row experiments costs. No precision is lost -- scores are already
    downcast to float32 when the results CSV is read (see
    data_manager._downcast_numeric), so the confidences were never wider
    than this.
    """
    confidence = np.asarray(confidence, dtype=np.float32)
    correct = np.asarray(correct).astype(bool)
    order = np.argsort(-confidence, kind="stable")
    sorted_conf = confidence[order]
    cum = np.zeros(len(sorted_conf) + 1, dtype=np.int32)
    np.cumsum(correct[order], dtype=np.int32, out=cum[1:])
    return ThresholdIndex(confidence=sorted_conf, cum_correct=cum, n=len(sorted_conf))


@dataclass
class ThresholdPoint:
    """What one operating point costs and buys."""
    threshold: float
    n_accepted: int
    coverage: float
    accuracy: float            # among accepted -- "selective accuracy"
    n_errors_accepted: int
    n_abstained: int
    accuracy_abstained: float  # among abstained: what the humans would be handed
    # Set when a target could not be met at any threshold (see
    # threshold_for_target).
    unreachable: bool = False


def _point_for_k(index: ThresholdIndex, k: int, threshold: float) -> ThresholdPoint:
    n = index.n
    k = int(max(0, min(k, n)))
    correct_accepted = int(index.cum_correct[k])
    n_abstained = n - k
    correct_abstained = index.n_correct - correct_accepted
    return ThresholdPoint(
        threshold=float(threshold),
        n_accepted=k,
        coverage=k / n if n else float("nan"),
        accuracy=correct_accepted / k if k else float("nan"),
        n_errors_accepted=k - correct_accepted,
        n_abstained=n_abstained,
        accuracy_abstained=correct_abstained / n_abstained if n_abstained else float("nan"),
    )


def threshold_point(index: ThresholdIndex, threshold: float) -> ThresholdPoint:
    """Accept every prediction whose confidence is >= `threshold`.

    `>=` rather than `>` so that a threshold set to a confidence a model
    actually emits accepts those rows -- with hard 0/1 scores (a rounded
    export, a one-hot head) the two differ by the whole dataset.

    The threshold is quantised to the index's own float32 precision before
    the search. Without that, a threshold of 0.9 read as a float64 sits
    *above* the float32 0.9 the model actually emitted, and asking for
    "confidence >= 0.9" would reject every row at exactly 0.9 -- which,
    given how often a threshold is set to a round number the scores are
    full of, is the whole dataset rather than an edge case. The cost is
    that the effective threshold can move by half a float32 ulp (~6e-8),
    which is far below any threshold anyone sets.
    """
    # searchsorted needs an ascending array; the index is descending, so
    # search the reversed view and convert back.
    ascending = index.confidence[::-1]
    query = np.float32(threshold) if index.confidence.dtype == np.float32 else threshold
    n_below = int(np.searchsorted(ascending, query, side="left"))
    return _point_for_k(index, index.n - n_below, threshold)


def threshold_for_target(index: ThresholdIndex, target_accuracy: float) -> ThresholdPoint:
    """The *most permissive* threshold whose accepted predictions still reach
    `target_accuracy` -- i.e. the largest coverage meeting the bar.

    Largest coverage rather than "first k that works" because selective
    accuracy is not monotone in k: a run of correct predictions in the
    middle of the ranking can lift it back above the target after it dipped,
    and stopping at the first dip would hand back a threshold that automates
    far less than the model can actually support.

    Only cut points that a threshold can actually express are considered: a
    threshold cannot split rows that share a confidence, so cutting inside a
    run of equal values would return an operating point that
    threshold_point() then disagrees with. With hard 0/1 scores that is not
    a corner case, it is most of the data.
    """
    n = index.n
    if n == 0:
        return ThresholdPoint(float("nan"), 0, float("nan"), float("nan"), 0, 0, float("nan"), True)

    k = np.arange(1, n + 1)
    accuracy = index.cum_correct[1:] / k
    expressible = np.concatenate([index.confidence[1:] < index.confidence[:-1], [True]])
    meeting = np.nonzero((accuracy >= target_accuracy) & expressible)[0]
    if meeting.size == 0:
        point = _point_for_k(index, 0, float("nan"))
        point.unreachable = True
        return point
    best_k = int(meeting[-1]) + 1
    return _point_for_k(index, best_k, float(index.confidence[best_k - 1]))


@dataclass
class RiskCoverage:
    coverage: np.ndarray      # increasing, (0, 1]
    risk: np.ndarray          # selective error rate at that coverage
    threshold: np.ndarray     # the confidence cut-off producing that coverage
    aurc: float               # area under the risk-coverage curve, lower is better
    # AURC of a model that ranks its errors perfectly (all mistakes last),
    # at the same accuracy -- the floor AURC is measured against, since a
    # model with more errors has a higher floor through no fault of its
    # ranking.
    optimal_aurc: float
    n: int


def risk_coverage(index: ThresholdIndex, max_points: int = CURVE_POINTS) -> RiskCoverage:
    n = index.n
    if n == 0:
        empty = np.array([])
        return RiskCoverage(empty, empty, empty, float("nan"), float("nan"), 0)

    k = np.arange(1, n + 1)
    risk_full = 1.0 - index.cum_correct[1:] / k
    # AURC is averaged over every k, not over the downsampled curve below:
    # the number goes in a table and must not depend on how many points were
    # kept for drawing.
    aurc = float(risk_full.mean())

    n_errors = n - index.n_correct
    # The ideal ranking puts every correct prediction first, so risk stays 0
    # until coverage reaches accuracy and then climbs linearly.
    ideal_correct = np.minimum(k, n - n_errors)
    optimal = float((1.0 - ideal_correct / k).mean())

    if n > max_points:
        picks = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    else:
        picks = np.arange(n)
    return RiskCoverage(
        coverage=(picks + 1) / n,
        risk=risk_full[picks],
        threshold=index.confidence[picks],
        aurc=aurc,
        optimal_aurc=optimal,
        n=n,
    )


@dataclass
class ClassThreshold:
    """One class's own accept threshold, for a target precision on the
    predictions carrying that class."""
    label: str
    n_predicted: int
    threshold: float
    n_kept: int
    precision: float          # among the kept predictions of this class
    coverage: float           # kept / predicted, within this class
    unreachable: bool = False


def per_class_thresholds(predicted: np.ndarray, confidence: np.ndarray, correct: np.ndarray,
                         labels: list, target_precision: float) -> list:
    """A separate threshold per predicted class.

    One global threshold is the wrong control whenever the classes are not
    equally hard or equally costly: a single cut-off that makes the rare,
    dangerous class safe usually throws away most of the easy majority
    class for nothing. Here each class gets the most permissive threshold
    that still reaches `target_precision` on its own predictions.

    Precision, not accuracy, because the population is "rows the model
    called c": a row is right exactly when its true label is c, so this is
    the precision of class c at that threshold.
    """
    predicted = np.asarray(predicted)
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct).astype(bool)

    # The rows grouped by predicted class once, then one slice per class --
    # rather than comparing every row against every label, which at two
    # thousand classes was most of this function's time.
    index_of = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    codes = pd.Series(predicted, dtype=object).map(index_of).fillna(-1).to_numpy(dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(index_of) + 1))

    out = []
    for label in labels:
        i = index_of[label]
        rows = order[bounds[i]:bounds[i + 1]]
        n_predicted = int(len(rows))
        if n_predicted == 0:
            out.append(ClassThreshold(str(label), 0, float("nan"), 0, float("nan"), float("nan"), True))
            continue
        index = build_index(confidence[rows], correct[rows])
        point = threshold_for_target(index, target_precision)
        out.append(ClassThreshold(
            label=str(label),
            n_predicted=n_predicted,
            threshold=point.threshold,
            n_kept=point.n_accepted,
            precision=point.accuracy,
            coverage=point.coverage,
            unreachable=point.unreachable,
        ))
    return out


@dataclass
class ConfidenceHistogram:
    """Confidence distribution, split by whether the prediction was right.

    The picture that says whether a threshold can work at all: two separated
    humps mean an operating point exists, one merged hump means no threshold
    will ever buy much, however good the ECE looks.
    """
    edges: np.ndarray
    correct: np.ndarray    # counts per bin, correct predictions
    wrong: np.ndarray      # counts per bin, wrong predictions


def confidence_histogram(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 25) -> ConfidenceHistogram:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct).astype(bool)
    lo = float(np.nanmin(confidence)) if confidence.size else 0.0
    hi = float(np.nanmax(confidence)) if confidence.size else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    right, _ = np.histogram(confidence[correct], bins=edges)
    wrong, _ = np.histogram(confidence[~correct], bins=edges)
    return ConfidenceHistogram(edges=edges, correct=right, wrong=wrong)


# ---- temperature scaling -------------------------------------------------

@dataclass
class TemperatureFit:
    """A single scalar divided into the logits before the softmax -- the
    standard post-hoc calibration fix (Guo et al., 2017).

    It cannot change which class is predicted (dividing by a positive scalar
    is monotone, so the argmax is untouched), which is exactly why it is
    worth offering: accuracy, the confusion matrix and every ranking metric
    are bit-for-bit unchanged, and only the confidence values move. T > 1
    means the model was over-confident and its scores were pushed towards
    the uniform distribution; T < 1 means the opposite.
    """
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    n_fit_rows: int
    n_rows: int
    # Scaled confidence for every row, in the caller's row order -- what the
    # "after" reliability diagram is drawn from. One float per row, not the
    # whole scaled matrix.
    scaled_confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    disabled_reason: Optional[str] = None


def _log_scores(scores: np.ndarray) -> np.ndarray:
    """Logits, up to the per-row constant a softmax ignores. Clipped because
    an exported softmax routinely contains exact zeros, and log(0) would
    make every temperature produce NaN. In float64, from whatever the scores
    are stored as: callers pass a block of rows at a time, so this is never
    a copy of the whole matrix."""
    return np.log(np.clip(np.ascontiguousarray(scores, dtype=np.float64), 1e-12, None))


def _beta_moments(scores: np.ndarray, rows: np.ndarray, true_idx: np.ndarray, beta: float) -> tuple:
    """Over `rows`, with z the logits and the softmax taken at beta * z:
    (mean NLL, its first and second derivative in beta).

    The NLL is mean(logsumexp(beta z) - beta z_true), so its derivative is
    mean(E[z] - z_true) and its second derivative mean(Var[z]) -- both
    under the softmax -- which is why it is convex in beta. One pass over
    the rows, a block at a time, in float64 for the sums only."""
    def block(start: int) -> tuple:
        idx = rows[start:start + T_BLOCK]
        # A run of consecutive rows (every row, below max_fit_rows) is a
        # slice rather than a gather -- which matters on a column-major
        # matrix, where gathering rows reads every column once per row.
        contiguous = len(idx) and idx[-1] - idx[0] + 1 == len(idx)
        z = _log_scores(scores[idx[0]:idx[-1] + 1] if contiguous else scores[idx])
        z_true = z[np.arange(len(idx)), true_idx[start:start + T_BLOCK]]
        scaled = beta * z
        top = scaled.max(axis=1, keepdims=True)
        w = np.exp(scaled - top)
        norm = w.sum(axis=1)
        wz = w * z
        mean_z = wz.sum(axis=1) / norm
        mean_z2 = (wz * z).sum(axis=1) / norm
        return (float((np.log(norm) + top[:, 0] - beta * z_true).sum()),
                float((mean_z - z_true).sum()),
                float(np.maximum(mean_z2 - mean_z ** 2, 0.0).sum()))

    parts = _map_blocks(block, range(0, len(rows), T_BLOCK))
    n = max(len(rows), 1)
    return tuple(sum(part[i] for part in parts) / n for i in range(3))


def _map_blocks(fn, starts) -> list:
    """fn over row-block starts on a small thread pool -- numpy releases the
    GIL for the exp/log/sum work each block is made of."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    starts = list(starts)
    workers = min(8, os.cpu_count() or 1, len(starts))
    if workers <= 1:
        return [fn(start) for start in starts]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, starts))


def _scaled_confidence(scores: np.ndarray, beta: float) -> np.ndarray:
    """Top softmax probability of every row at beta -- 1 / sum(exp(beta (z
    - z_max))) -- without the scaled matrix."""
    out = np.empty(len(scores))

    def block(start: int) -> None:
        scaled = beta * _log_scores(scores[start:start + T_BLOCK])
        scaled -= scaled.max(axis=1, keepdims=True)
        out[start:start + len(scaled)] = 1.0 / np.exp(scaled).sum(axis=1)

    _map_blocks(block, range(0, len(scores), T_BLOCK))
    return out


def fit_temperature(scores: np.ndarray, y_true: np.ndarray, labels_for_cols: list,
                    y_pred: Optional[np.ndarray] = None, n_bins: int = ECE_BINS,
                    max_fit_rows: int = T_FIT_MAX_ROWS, seed: int = 0) -> TemperatureFit:
    """Fit one temperature by minimising the negative log-likelihood, and
    report ECE before and after.

    The fit reads at most `max_fit_rows` rows (sampled with a fixed seed, so
    the same data always gives the same temperature); the before/after
    numbers are then computed on every row.

    The search is over beta = 1/T, where the NLL is convex (its second
    derivative is a variance, see _beta_moments): a Newton step from beta =
    1, kept inside a shrinking bracket so it can never leave the bounds or
    stall, lands on the minimum in a handful of passes. That replaced an
    80-point grid plus a golden-section refinement -- some 120 softmaxes
    over the sampled rows, which at two thousand classes took minutes.

    Two caveats the caller must surface rather than hide:

    * fitting on the same rows the ECE is measured on is optimistic. A
      temperature is one parameter, so the optimism is small, but it is not
      zero -- a temperature meant for production belongs on a held-out
      split, which here means loading that split and reading the value
      fitted there.
    * the temperature minimises NLL, which is a proper scoring rule, not
      ECE, which is a binned proxy. They usually agree, but not always: a
      model whose confidence carries little information about correctness
      can have its NLL improved and its ECE made *worse* by the same
      temperature. That is why both numbers are returned and both are
      shown -- an implementation that only reported the fitted T would be
      claiming an improvement it had not checked.
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    n = len(scores)
    if n == 0 or scores.ndim != 2 or scores.shape[1] == 0:
        return TemperatureFit(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                              0, 0, disabled_reason="no rows to fit on")

    label_index: dict = {}
    for j, label in enumerate(labels_for_cols):
        label_index[label] = j  # the last column of a label, as the previous dict comprehension did
    true_idx_all = pd.Series(y_true, dtype=object).map(label_index).fillna(-1).to_numpy(dtype=np.int64)
    in_columns = true_idx_all >= 0
    if not in_columns.any():
        return TemperatureFit(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                              0, n, disabled_reason="no row's true label has a score column")

    usable = np.nonzero(in_columns)[0]
    if usable.size > max_fit_rows:
        rng = np.random.default_rng(seed)
        fit_rows = rng.choice(usable, size=max_fit_rows, replace=False)
    else:
        fit_rows = usable
    # Sorted: the order of the rows does not change the loss, and ascending
    # indices keep each block's reads as close together as they can be.
    fit_rows = np.sort(fit_rows)
    true_idx_fit = true_idx_all[fit_rows]

    low, high = 1.0 / T_MAX, 1.0 / T_MIN
    beta = min(max(1.0, low), high)
    passes = 0
    for _ in range(100):
        passes += 1
        _loss, grad, curvature = _beta_moments(scores, fit_rows, true_idx_fit, beta)
        if grad > 0:
            high = beta
        else:
            low = beta
        if grad == 0 or high - low < BETA_TOLERANCE:
            break
        step = beta - grad / curvature if curvature > 0 else np.nan
        if low < step < high:
            # Newton converges from one side, so the bracket alone would
            # never close: stop once the step itself is below tolerance.
            done = abs(step - beta) < BETA_TOLERANCE
            beta = step
            if done:
                break
        else:
            # A step that leaves the bracket is replaced by bisection.
            beta = 0.5 * (low + high)
    temperature = float(1.0 / beta)
    _log(f"temperature fitted in {passes} pass(es) over {len(fit_rows):,} rows: T={temperature:.4f}")

    nll_before = _beta_moments(scores, usable, true_idx_all[usable], 1.0)[0]
    nll_after = _beta_moments(scores, usable, true_idx_all[usable], beta)[0]
    scaled_confidence = _scaled_confidence(scores, beta)

    if y_pred is None:
        y_pred = np.asarray(labels_for_cols, dtype=object)[scores.argmax(axis=1)]
    ece_before, _ = calibration_error(y_true, y_pred, scores, n_bins=n_bins)
    ece_after, _ = calibration_error_from_confidence(scaled_confidence, y_true == y_pred, n_bins=n_bins)

    return TemperatureFit(
        temperature=temperature,
        nll_before=nll_before, nll_after=nll_after,
        ece_before=float(ece_before), ece_after=float(ece_after),
        n_fit_rows=int(len(fit_rows)), n_rows=n,
        scaled_confidence=scaled_confidence,
    )

