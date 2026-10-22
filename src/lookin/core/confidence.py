"""Per-prediction confidence diagnostics for one experiment: margin (top
score minus second score) and the entropy of the softmax-normalized score
distribution -- see web/views/confidence.py.

This answers a different question than core/metrics.py: those functions
compare a prediction to the ground truth (was the model right); these
describe how sure the model was, independent of whether it was right.
The two only get interesting together -- a wrong prediction the model was
confident about is a very different failure than one it was already unsure
about (see confident_and_wrong below).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lookin.core.metrics import ROW_BLOCK, ScoreDiagnostics, diagnose_scores

MARGIN_COL = "confidence_margin"
ENTROPY_COL = "confidence_entropy"
NORMALIZED_ENTROPY_COL = "confidence_entropy_norm"

# Floor applied to a probability before taking its log, so a class softmax
# underflowed to exactly 0.0 for (an entropy term of 0 log 0, not -inf).
_PROB_FLOOR = 1e-12


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class ConfidenceData:
    df: pd.DataFrame  # `merged` plus MARGIN_COL / ENTROPY_COL / NORMALIZED_ENTROPY_COL
    score_diagnostics: ScoreDiagnostics
    softmax_applied: bool  # False when the score columns already summed to 1


def compute_confidence(merged: pd.DataFrame, score_cols: list, class_labels: list) -> ConfidenceData:
    """`merged` without its score columns, plus per-row margin/entropy columns.

    margin = top score - second score, defined on the raw scores whatever
    they are (a rank gap does not need a probability distribution).

    Entropy does need one, so it is computed on the raw scores when they
    already sum to 1 (a genuine softmax export -- see diagnose_scores) and
    on softmax(scores) otherwise. Applying softmax to an already-normalized
    row would flatten real differences toward uniform, which is why the
    check comes first rather than softmax-ing unconditionally.
    """
    # As stored (float32 after load), read a block of rows at a time in
    # float64: one float64 copy of the whole matrix is 1.6 GB at a hundred
    # thousand rows and two thousand classes, and the one-shot version of
    # this made four of them.
    scores = merged[score_cols].to_numpy()
    labels_for_cols = class_labels[: len(score_cols)]
    diagnostics = diagnose_scores(scores, labels_for_cols)

    n = len(scores)
    n_classes = scores.shape[1] if scores.ndim == 2 else 0
    if n == 0 or n_classes == 0:
        margin = np.zeros(n)
        entropy = np.zeros(n)
        normalized_entropy = np.full(n, np.nan)
    else:
        margin = np.full(n, np.nan)
        entropy = np.empty(n)
        distribution = diagnostics.n_rows_not_normalized == diagnostics.n_rows_missing
        for start in range(0, n, ROW_BLOCK):
            # Row-major: a matrix read off a loaded frame is column-major
            # (the loader builds it a column at a time), which makes every
            # per-row operation below stride across the whole matrix.
            block = np.ascontiguousarray(scores[start:start + ROW_BLOCK], dtype=np.float64)
            stop = start + len(block)
            if n_classes >= 2:
                top2 = np.partition(block, -2, axis=1)[:, -2:]  # [:, 0]=2nd largest, [:, 1]=largest
                margin[start:stop] = top2[:, 1] - top2[:, 0]
            # A row with a missing score makes the matrix "not normalized"
            # (see diagnose_scores), but its entropy is NaN either way: it
            # must not push every complete row of a genuine softmax export
            # through a second softmax.
            probs = block if distribution else _softmax(block)
            probs = np.clip(probs, _PROB_FLOOR, 1.0)
            entropy[start:stop] = -(probs * np.log(probs)).sum(axis=1)
        max_entropy = np.log(n_classes) if n_classes > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else np.full(n, np.nan)

    # Every column but the scores: nothing downstream reads those, and the
    # rows of this frame are what the carousel shows -- two thousand score
    # columns would be two thousand lines of metadata under each crop.
    out = merged.drop(columns=[c for c in score_cols if c in merged.columns])
    out[MARGIN_COL] = margin
    out[ENTROPY_COL] = entropy
    out[NORMALIZED_ENTROPY_COL] = normalized_entropy
    softmax_applied = diagnostics.n_rows_not_normalized != diagnostics.n_rows_missing
    return ConfidenceData(df=out, score_diagnostics=diagnostics, softmax_applied=softmax_applied)


def confident_and_wrong(data: ConfidenceData, label_col: str, metric: str, top_percentile: float) -> pd.DataFrame:
    """Rows that are both incorrect and among the `top_percentile`% most
    confident predictions by `metric` ("margin" or "entropy") -- the
    failure mode a model doesn't hedge on, and arguably the most concerning
    one: a low-confidence error at least leaves a trace to catch it by.

    The confidence threshold is taken from the distribution over *every*
    prediction (not just the wrong ones), so "top 25% most confident" means
    the same thing whether the pool being checked is 1% wrong or 40% wrong.
    """
    df = data.df
    if df.empty:
        return df.iloc[0:0]

    top_percentile = min(max(top_percentile, 0.0), 100.0)
    if metric == "entropy":
        threshold = np.nanpercentile(df[NORMALIZED_ENTROPY_COL], top_percentile)
        confident = df[NORMALIZED_ENTROPY_COL] <= threshold
    else:
        threshold = np.nanpercentile(df[MARGIN_COL], 100.0 - top_percentile)
        confident = df[MARGIN_COL] >= threshold

    return df[confident & ~df["correct"]]
