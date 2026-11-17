"""Decision rule: one bias per class added to the scores before the argmax.

A multiclass model decides with ``argmax_c s_c``. Adding a bias ``b_c`` to
each class's log-score -- ``argmax_c (s_c + b_c)`` -- is the multiclass form
of moving a threshold: every row still gets exactly one class, so unlike a
one-vs-rest threshold per class it is a rule that can be deployed. It is also
exactly what the same network would do with ``b`` added to the bias of its
last layer: the scores it would output are ``softmax(logits + b)``, which is
what biased_scores() returns. That is the reason the bias is on the
log-scores and not of the (in theory finer) affine form ``a_c p_c - b_c``: no
network outputs the latter, so it could not be applied to a workspace as if
the experiment had been trained that way.

The scores are put in log space once (score_space / prepare): rows that sum
to 1 are probabilities and are logged; scores outside [0, 1] are taken as
logits and used as they are. Scores inside [0, 1] that do not sum to 1
(independent sigmoids) are refused: a bias on them is not what the network
would do with a bias on its logits.

fit_biases() looks for the biases that maximise an objective on a tuning
pool -- macro F1 or weighted F1 over chosen classes, or balanced accuracy --
by exact coordinate ascent: with every other bias fixed, the prediction of a
row only depends on whether ``b_c`` is above ``u_i = max_{k != c}(s_ik + b_k)
- s_ic``, so sorting the ``u_i`` gives the objective for every value of
``b_c`` in one pass (_sweep). The same sweep, with nothing optimised, is the
"argmax + bias" precision-recall curve of a class (class_sweep): the
deployable counterpart of its one-vs-rest curve.

Biases are defined up to a common constant (adding the same value to every
bias changes nothing). One class is therefore kept at 0: any class the user
locks at 0, or else the class with the largest support in the tuning pool.

The tuning pool is the validation split, read from the dataset CSV
(read_tuning_pool), or the evaluation pool itself when the user chooses so;
check_pools() refuses a validation pool that shares a uuid with the
evaluation pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


# Where the biases are tuned -- keys, compared in code.
POOL_VAL = "val"
POOL_TEST = "test"
POOL_KEYS = (POOL_VAL, POOL_TEST)
# The split value the validation rows carry by default.
DEFAULT_VAL_VALUE = "val"
DEFAULT_SPLIT_COLUMN = "split"

# What is maximised -- keys, compared in code.
OBJ_MACRO_F1 = "macro_f1"
OBJ_WEIGHTED_F1 = "weighted_f1"
OBJ_BALANCED_ACCURACY = "balanced_accuracy"
OBJECTIVES = (
    (OBJ_MACRO_F1, "Macro F1"),
    (OBJ_WEIGHTED_F1, "Weighted F1"),
    (OBJ_BALANCED_ACCURACY, "Balanced accuracy"),
)
# The objectives a subset of classes makes sense for. Balanced accuracy is a
# mean of recalls, which never penalise a false positive: restricted to some
# classes, its optimum predicts those classes everywhere.
SUBSET_OBJECTIVES = (OBJ_MACRO_F1, OBJ_WEIGHTED_F1)

# How the scores are put in log space -- keys, compared in code.
SPACE_LOG_PROB = "log_prob"   # rows sum to 1: log p
SPACE_LOGIT = "logit"         # raw logits: as they are
# A row sums to 1 within this: metrics.ROW_SUM_TOLERANCE, repeated here so that
# this module (imported when a workspace opens) does not import scikit-learn.
SPACE_TOLERANCE = 1e-3

DEFAULT_MAX_PASSES = 20
# Row block for the passes over the whole score matrix.
ROW_BLOCK = 8192
# An improvement smaller than this is noise, not a reason to move a bias.
MIN_GAIN = 1e-12
# The search (fit_biases): random starting points tried besides all-zero
# and the class prior, while there are few enough classes for it to be
# cheap; and how many pairs of classes are tried as a joint move once no
# single bias helps -- every pair up to PAIRS_ALL_MAX_CLASSES free classes,
# else the PAIRS_PER_CLASS most confused pairs per class.
# Breakpoints closer than this are one: they come from the same scores
# summed in a different order, and a bias between them would land on
# either side depending on that order.
BREAKPOINT_TOLERANCE = 1e-9
RANDOM_STARTS = 4
RANDOM_STARTS_MAX_CLASSES = 50
RANDOM_START_SCALE = 1.0
PAIRS_ALL_MAX_CLASSES = 40
PAIRS_PER_CLASS = 4


# ---- pools -----------------------------------------------------------------

@dataclass
class TuningPool:
    """The rows the biases are tuned on, merged with one experiment."""
    key: str                      # POOL_VAL or POOL_TEST
    split_column: str
    split_value: str
    frame: Optional[pd.DataFrame]  # merged rows (None when unusable)
    n_split_rows: int             # rows of that split in the dataset
    n_rows: int                   # rows with a prediction and a label
    n_without_prediction: int     # rows of the split the results CSV does not cover
    disabled_reason: Optional[str] = None
    warnings: list = field(default_factory=list)


def read_tuning_pool(state, experiment_name: str, split_column: str = DEFAULT_SPLIT_COLUMN,
                     split_value: str = DEFAULT_VAL_VALUE) -> TuningPool:
    """The validation rows of the dataset CSV (`split_column == split_value`)
    merged with `experiment_name`'s predictions. Never raises for a missing
    split or missing predictions: disabled_reason says what is wrong, in
    words meant for the user."""
    from lookin.core.data_manager import read_dataset_split

    def unusable(reason, n_split=0, n_missing=0):
        return TuningPool(POOL_VAL, split_column, split_value, None, n_split, 0, n_missing, reason)

    try:
        rows = read_dataset_split(state.dataset, split_column, split_value)
    except ValueError as ex:
        return unusable(str(ex))
    if rows.empty:
        return unusable(f"The dataset has no row where '{split_column}' is '{split_value}'.")
    uuid_col = state.dataset.columns.uuid_col
    experiment = state.experiments[experiment_name]
    predicted = set(experiment.df[experiment.uuid_col].astype(str))
    has_prediction = rows[uuid_col].isin(predicted).to_numpy()
    n_missing = int((~has_prediction).sum())
    if not has_prediction.any():
        return unusable(
            f"The results of '{experiment_name}' hold no prediction for any of the {len(rows):,} rows "
            f"where '{split_column}' is '{split_value}'. Run the model on those rows and add them to its "
            f"results CSV.", len(rows), n_missing)
    frame = state.merge_rows(rows, experiment_name)
    warnings = []
    if n_missing:
        warnings.append(
            f"{n_missing:,} of the {len(rows):,} rows where '{split_column}' is '{split_value}' have no "
            f"prediction in the results of '{experiment_name}': the biases are tuned on the other "
            f"{len(rows) - n_missing:,}.")
    return TuningPool(POOL_VAL, split_column, split_value, frame, len(rows), len(frame), n_missing,
                      None, warnings)


def evaluation_as_tuning_pool(frame: pd.DataFrame, split_column: str = "", split_value: str = "") -> TuningPool:
    """The evaluation pool itself used for tuning -- the case the interface
    has to make loud, since every metric measured on it afterwards is
    optimistic."""
    return TuningPool(POOL_TEST, split_column, split_value, frame, len(frame), len(frame), 0)


@dataclass
class PoolCheck:
    """Whether the tuning pool and the evaluation pool are kept apart."""
    same_pool: bool          # tuning on the evaluation pool, by choice
    n_shared: int            # uuids in both (expected only when same_pool)
    n_tuning_duplicates: int
    n_evaluation_duplicates: int
    blocking: Optional[str] = None


def check_pools(tuning_uuids, evaluation_uuids, same_pool: bool) -> PoolCheck:
    """Refuse a validation pool that shares rows with the evaluation pool:
    a bias tuned on a row and then scored on it is not a validation."""
    tuning = pd.Series(tuning_uuids, dtype=object).astype(str)
    evaluation = pd.Series(evaluation_uuids, dtype=object).astype(str)
    shared = len(set(tuning) & set(evaluation))
    check = PoolCheck(same_pool, shared, int(tuning.duplicated().sum()), int(evaluation.duplicated().sum()))
    if not same_pool and shared:
        check.blocking = (
            f"{shared:,} uuid(s) are both in the tuning rows and in the evaluated rows. Tuned on them, "
            f"the biases would be scored on the rows they were fitted to. Make the two splits disjoint "
            f"in the dataset, or choose explicitly to tune on the evaluated rows.")
    return check


# ---- score space -------------------------------------------------------------

def score_space(scores: np.ndarray) -> tuple:
    """(space, reason): how the scores are put in log space, or None and why
    no bias can be applied to them."""
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return None, "the score columns hold no finite value"
    if finite.min() < -SPACE_TOLERANCE or finite.max() > 1 + SPACE_TOLERANCE:
        return SPACE_LOGIT, None
    scored = np.isfinite(scores).any(axis=1)
    sums = np.nansum(scores[scored], axis=1, dtype=np.float64)
    if np.all(np.abs(sums - 1.0) <= SPACE_TOLERANCE):
        return SPACE_LOG_PROB, None
    return None, ("the scores lie in [0, 1] but the rows do not sum to 1 (independent sigmoids, or "
                  "probabilities of a subset of the classes): a bias on them is not what the network "
                  "would do with a bias on its logits")


def log_scores(scores: np.ndarray, space: str) -> np.ndarray:
    """The scores in the space the biases are added in, as float32 with
    -inf for a missing score (never the highest, like argmax_ignoring_nan)."""
    out = np.asarray(scores, dtype=np.float32)
    if space == SPACE_LOG_PROB:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(np.clip(out, 0.0, None))
    else:
        out = out.copy()
    out[np.isnan(out)] = -np.inf
    return out


def biased_scores(scores: np.ndarray, space: str, biases: np.ndarray) -> np.ndarray:
    """The scores the same network would output with `biases` added to its
    logits: softmax(log p + b) for probabilities -- p_c e^{b_c} renormalised
    -- and s + b for logits. Missing scores stay missing."""
    biases = np.asarray(biases, dtype=np.float64)
    if space == SPACE_LOGIT:
        return (np.asarray(scores, dtype=np.float64) + biases).astype(np.float32)
    out = np.empty(scores.shape, dtype=np.float32)
    weights = np.exp(biases - biases.max())
    for start in range(0, len(scores), ROW_BLOCK):
        block = np.asarray(scores[start:start + ROW_BLOCK], dtype=np.float64)
        weighted = np.clip(block, 0.0, None) * weights
        total = np.nansum(weighted, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:start + ROW_BLOCK] = weighted / total
    return out


def predict(logs: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """argmax_c(logs + biases) per row, the first column on a tie, -1 for a
    row with no finite score."""
    n = len(logs)
    out = np.full(n, -1, dtype=np.int64)
    biases = np.asarray(biases, dtype=np.float64)
    for start in range(0, n, ROW_BLOCK):
        block = logs[start:start + ROW_BLOCK] + biases
        idx = block.argmax(axis=1)
        some = np.isfinite(block[np.arange(len(block)), idx])
        out[start:start + len(block)] = np.where(some, idx, -1)
    return out


# ---- the problem ---------------------------------------------------------------

@dataclass
class Problem:
    """A pool of rows as the optimiser sees it: one log-score column per
    class, the true class of each row as a column index."""
    labels: list                 # class of each score column
    space: str
    logs: np.ndarray             # (n, k) float32, -inf where missing
    y: np.ndarray                # (n,) column of the true class, -1 if it has none
    n_true: np.ndarray           # (k,) rows of each class
    labels_without_column: dict  # label -> rows whose class has no score column

    @property
    def n_rows(self) -> int:
        return len(self.y)

    def subset(self, rows: np.ndarray) -> "Problem":
        y = self.y[rows]
        return Problem(self.labels, self.space, self.logs[rows], y,
                       np.bincount(y[y >= 0], minlength=len(self.labels)), self.labels_without_column)


def prepare(frame: pd.DataFrame, label_col: str, score_cols: list, labels_for_cols: list,
            space: Optional[str] = None) -> Problem:
    """A merged frame (AppState.merged / merge_rows) as a Problem. Raises
    ValueError, in words for the user, when no bias can be applied: two score
    columns for one class, or scores that are not log-able (score_space).
    `space` forces the score space -- the evaluation pool's, so that the
    tuning and the evaluated rows are read the same way."""
    labels = list(labels_for_cols[: len(score_cols)])
    duplicated = sorted({str(l) for l in labels if labels.count(l) > 1})
    if duplicated:
        raise ValueError(f"several score columns map to the class(es) {', '.join(duplicated)}: "
                         f"one bias per class is not defined")
    scores = frame[score_cols].to_numpy()
    if space is None:
        space, reason = score_space(scores)
        if space is None:
            raise ValueError(reason)
    index = {label: j for j, label in enumerate(labels)}
    truth = frame[label_col].to_numpy()
    y = np.fromiter((index.get(v, -1) for v in truth), dtype=np.int64, count=len(truth))
    without = pd.Series(truth[y < 0]).value_counts()
    return Problem(labels, space, log_scores(scores, space), y,
                   np.bincount(y[y >= 0], minlength=len(labels)),
                   {str(k): int(v) for k, v in without.items()})


# ---- objectives -----------------------------------------------------------------

@dataclass
class TuningSpec:
    objective: str = OBJ_MACRO_F1
    # Labels whose score counts in the objective; None for every class.
    classes: Optional[tuple] = None
    # Labels whose bias stays at 0.
    locked: tuple = ()
    # |bias| never exceeds this (None: unbounded).
    max_abs_bias: Optional[float] = None
    max_passes: int = DEFAULT_MAX_PASSES
    # Random starting points tried besides all-zero and the class prior;
    # None: RANDOM_STARTS up to RANDOM_STARTS_MAX_CLASSES free classes, else 0.
    random_starts: Optional[int] = None
    seed: int = 0
    # Try moving two classes' biases together once no single one helps.
    pair_moves: bool = True
    # Start from the class prior (logit adjustment) besides all-zero.
    prior_start: bool = True


@dataclass
class _Weights:
    w: np.ndarray        # (k,) weight of each class in the objective
    recall: bool         # True: the term is a recall, else an F1
    counted: list        # labels in the objective
    absent: list         # labels asked for but with no row in the pool


def _weights(problem: Problem, spec: TuningSpec) -> _Weights:
    if spec.objective not in dict(OBJECTIVES):
        raise ValueError(f"unknown objective {spec.objective!r}; one of {', '.join(dict(OBJECTIVES))}")
    k = len(problem.labels)
    asked = np.ones(k, dtype=bool)
    if spec.classes is not None:
        if spec.objective not in SUBSET_OBJECTIVES:
            raise ValueError("balanced accuracy is a mean of recalls, which never count a false positive: "
                             "over a subset of the classes it is maximised by predicting those classes "
                             "everywhere, so it is only defined over every class")
        wanted = set(map(str, spec.classes))
        unknown = sorted(wanted - set(map(str, problem.labels)))
        if unknown:
            raise ValueError(f"no score column for: {', '.join(unknown)}")
        asked = np.array([str(l) in wanted for l in problem.labels])
    present = problem.n_true > 0
    use = asked & present
    if not use.any():
        raise ValueError("none of the classes in the objective has a row in the tuning pool")
    if spec.objective == OBJ_WEIGHTED_F1:
        w = np.where(use, problem.n_true, 0).astype(np.float64)
    else:
        w = use.astype(np.float64)
    w /= w.sum()
    return _Weights(w, spec.objective == OBJ_BALANCED_ACCURACY,
                    [problem.labels[j] for j in np.flatnonzero(use)],
                    [problem.labels[j] for j in np.flatnonzero(asked & ~present)])


def _terms(tp, pred, n, recall: bool):
    """The per-class term of the objective: recall or F1, from counts (any
    shape), 0 where undefined -- only weighted where the class has rows."""
    tp = np.asarray(tp, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if recall:
            value = tp / n
        else:
            value = 2 * tp / (pred + n)
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _counts(problem: Problem, pred: np.ndarray) -> tuple:
    k = len(problem.labels)
    ok = pred >= 0
    predicted = np.bincount(pred[ok], minlength=k).astype(np.float64)
    hit = ok & (pred == problem.y)
    tp = np.bincount(pred[hit], minlength=k).astype(np.float64)
    return tp, predicted


def _objective(problem: Problem, weights: _Weights, pred: np.ndarray) -> float:
    tp, predicted = _counts(problem, pred)
    return float((weights.w * _terms(tp, predicted, problem.n_true, weights.recall)).sum())


def objective_value(problem: Problem, spec: TuningSpec, biases: np.ndarray) -> float:
    """The objective of `spec` on `problem` under `biases`."""
    return _objective(problem, _weights(problem, spec), predict(problem.logs, biases))


# ---- the sweep of one class's bias ---------------------------------------------

class _Tops:
    """The three highest biased log-scores of every row, and their columns,
    kept up to date as biases change: what the rest of a row offers class c
    -- or a pair of classes -- is the best of them outside it."""

    DEPTH = 3

    def __init__(self, logs: np.ndarray, biases: np.ndarray):
        self.logs = logs
        self.biases = biases
        n = len(logs)
        self.v = np.full((self.DEPTH, n), -np.inf)
        self.i = np.full((self.DEPTH, n), -1, dtype=np.int64)
        self.rescan(np.arange(n))

    @property
    def i1(self) -> np.ndarray:
        return self.i[0]

    def rescan(self, rows: np.ndarray) -> None:
        k = self.logs.shape[1]
        for start in range(0, len(rows), ROW_BLOCK):
            part = rows[start:start + ROW_BLOCK]
            block = self.logs[part] + self.biases
            r = np.arange(len(part))
            for level in range(self.DEPTH):
                if level >= k:
                    self.v[level, part] = -np.inf
                    self.i[level, part] = -1
                    continue
                # argmax, so ties go to the first column -- as predict() does.
                idx = block.argmax(axis=1)
                val = block[r, idx]
                self.v[level, part] = val
                self.i[level, part] = np.where(np.isfinite(val), idx, -1)
                block[r, idx] = -np.inf

    def set_biases(self, changes: dict) -> None:
        """{column: new bias}; rows where a changed column was or becomes
        one of the top three are scanned again."""
        touched = np.zeros(len(self.logs), dtype=bool)
        for c, value in changes.items():
            self.biases[c] = value
            column = self.logs[:, c].astype(np.float64) + value
            touched |= (self.i == c).any(axis=0) | (column >= self.v[-1])
        self.rescan(np.flatnonzero(touched))

    def set_bias(self, c: int, value: float) -> None:
        self.set_biases({c: value})

    def prediction(self) -> np.ndarray:
        return self.i[0].copy()

    def competitor(self, c: int) -> tuple:
        """(value, column) of the best class other than c on every row."""
        return self.outside((c,))

    def outside(self, group) -> tuple:
        """(value, column) of the best class not in `group` (at most
        DEPTH - 1 columns) on every row; -inf and -1 where there is none."""
        value = np.full(len(self.logs), -np.inf)
        column = np.full(len(self.logs), -1, dtype=np.int64)
        found = np.zeros(len(self.logs), dtype=bool)
        for level in range(self.DEPTH):
            ok = ~found & ~np.isin(self.i[level], list(group))
            value = np.where(ok, self.v[level], value)
            column = np.where(ok, self.i[level], column)
            found |= ok
        return value, column

    def inside(self, group) -> tuple:
        """(value, column) of the best class of `group` on every row, ties to
        the first column."""
        group = sorted(group)
        value = np.full(len(self.logs), -np.inf)
        column = np.full(len(self.logs), -1, dtype=np.int64)
        for c in group:
            v = self.logs[:, c].astype(np.float64) + self.biases[c]
            better = v > value
            value = np.where(better, v, value)
            column = np.where(better & np.isfinite(v), c, column)
        return value, column


@dataclass
class _Sweep:
    """Everything that changes when only b_c moves, one entry per interval
    between consecutive breakpoints: interval j is b_c in (edges[j-1],
    edges[j]), with edges[-1] = -inf and edges[len] = +inf."""
    edges: np.ndarray        # (m,) distinct breakpoints, ascending (the largest of each run of near-equal ones)
    tp_c: np.ndarray         # (m + 1,) true positives of c in each interval
    pred_c: np.ndarray       # (m + 1,) rows predicted c
    objective: np.ndarray    # (m + 1,) objective (NaN without weights)
    lows: Optional[np.ndarray] = None  # (m,) the smallest breakpoint of each run


def _sweep(problem: Problem, c: int, competitor: tuple, weights: Optional[_Weights]) -> _Sweep:
    comp_value, comp_col = competitor
    own = problem.logs[:, c].astype(np.float64)
    with np.errstate(invalid="ignore"):
        u = comp_value - own          # the row is c iff b_c > u
    # A row whose own score is missing never becomes c, one with no other
    # score always is: neither moves with b_c.
    always = np.isneginf(u) | (np.isfinite(own) & np.isneginf(comp_value))
    movable = np.isfinite(u)
    order = np.flatnonzero(movable)
    order = order[np.argsort(u[order], kind="stable")]
    u_sorted = u[order]
    y = problem.y
    k = len(problem.labels)

    # Base: b_c below every breakpoint -- every row at its competitor
    # (-1 when it has no other score), except those that are always c.
    pred0 = np.where(always, c, comp_col)
    hit_c = (y[order] == c).astype(np.float64)
    tp_c = np.concatenate([[float(((pred0 == c) & (y == c)).sum())], hit_c])
    tp_c = np.cumsum(tp_c)
    pred_c = float((pred0 == c).sum()) + np.arange(len(order) + 1, dtype=np.float64)

    # Positions where b_c can stop: after a run of equal breakpoints.
    if len(order):
        last_of_run = np.r_[np.diff(u_sorted) > BREAKPOINT_TOLERANCE, True]
        stops = np.r_[0, np.flatnonzero(last_of_run) + 1]
        edges = u_sorted[last_of_run]
        lows = u_sorted[np.r_[True, last_of_run[:-1]]]
    else:
        stops = np.array([0])
        edges = lows = np.empty(0)

    if weights is None:
        objective = np.full(len(stops), np.nan)
    else:
        tp, predicted = _counts(problem, pred0)
        w = weights.w
        base = float((w * _terms(tp, predicted, problem.n_true, weights.recall)).sum())
        own_term = w[c] * _terms(tp_c, pred_c, problem.n_true[c], weights.recall)
        # Each movable row, as it joins c, leaves its competitor class a.
        alt = comp_col[order]
        hit_alt = (y[order] == alt).astype(np.float64)
        perm = np.argsort(alt, kind="stable")
        a_p = alt[perm]
        starts = np.r_[0, np.flatnonzero(np.diff(a_p)) + 1] if len(a_p) else np.array([], dtype=int)
        group = np.cumsum(np.r_[True, np.diff(a_p) != 0]) - 1 if len(a_p) else np.array([], dtype=int)
        cnt_p = np.arange(len(a_p)) - starts[group] + 1 if len(a_p) else np.array([])
        cs = np.cumsum(hit_alt[perm])
        cs_before = np.r_[0.0, cs][starts] if len(a_p) else np.array([])
        tpc_p = cs - cs_before[group] if len(a_p) else np.array([])
        cnt = np.empty(len(alt))
        tpc = np.empty(len(alt))
        cnt[perm] = cnt_p
        tpc[perm] = tpc_p
        after = _terms(tp[alt] - tpc, predicted[alt] - cnt, problem.n_true[alt], weights.recall)
        before = _terms(tp[alt] - tpc + hit_alt, predicted[alt] - cnt + 1, problem.n_true[alt], weights.recall)
        delta = w[alt] * (after - before) if len(alt) else np.empty(0)
        others = base - w[c] * _terms(tp[c], predicted[c], problem.n_true[c], weights.recall) \
            + np.r_[0.0, np.cumsum(delta)]
        objective = (others + own_term)[stops]
    return _Sweep(edges, tp_c[stops], pred_c[stops], objective, lows)


def _interval_value(edges: np.ndarray, j: int, bound: Optional[float], lows=None) -> Optional[float]:
    """A bias inside interval j (see _Sweep), within [-bound, bound], or None
    when the interval lies outside the bound. Inner intervals take their
    midpoint, the two outer ones a point one unit past the last breakpoint."""
    if bound is None:
        return _interval_point(edges, j, -np.inf, np.inf, lows)
    return _interval_point(edges, j, -bound, bound, lows)


def _interval_point(edges: np.ndarray, j: int, low: float, high: float, lows=None) -> Optional[float]:
    """_interval_value within [low, high]. Interval j runs from the largest
    breakpoint of run j-1 to the smallest of run j (`lows`; the same as
    `edges` when every run is a single value)."""
    lo = edges[j - 1] if j > 0 else -np.inf
    hi = (edges if lows is None else lows)[j] if j < len(edges) else np.inf
    if hi <= low or lo >= high:
        return None
    lo_b, hi_b = max(lo, low), min(hi, high)
    if lo_b >= hi_b:
        return None
    if np.isfinite(lo_b) and np.isfinite(hi_b):
        if np.isfinite(lo) and np.isfinite(hi):
            return float((lo_b + hi_b) / 2)
        # An outer interval clipped by the bound: near its breakpoint.
        if np.isfinite(hi):
            return float(max(hi - 1.0, (lo_b + hi_b) / 2))
        return float(min(lo + 1.0, (lo_b + hi_b) / 2))
    if np.isfinite(hi_b):
        return float(hi_b - 1.0)
    if np.isfinite(lo_b):
        return float(lo_b + 1.0)
    return 0.0


def _group_counts(cls: np.ndarray, hit: np.ndarray) -> tuple:
    """For a sequence of events on classes `cls`, how many events -- and how
    many hits -- each class has had up to and including each one."""
    n = len(cls)
    if not n:
        return np.empty(0), np.empty(0)
    perm = np.argsort(cls, kind="stable")
    c_p = cls[perm]
    new_group = np.r_[True, c_p[1:] != c_p[:-1]]
    starts = np.flatnonzero(new_group)
    group = np.cumsum(new_group) - 1
    cnt_p = np.arange(n) - starts[group] + 1
    cs = np.cumsum(hit[perm])
    tpc_p = cs - np.r_[0.0, cs][starts][group]
    cnt, tpc = np.empty(n), np.empty(n)
    cnt[perm], tpc[perm] = cnt_p, tpc_p
    return cnt, tpc


def _group_sweep(problem: Problem, inside: tuple, outside: tuple, weights: _Weights) -> tuple:
    """(edges, objective): the objective as the same shift t is added to
    the biases of a group of classes, the others fixed -- interval j is t in
    (edges[j-1], edges[j]). A row goes to the group's best class exactly
    when t exceeds u = (best outside) - (best inside), so one sort of the u
    gives every value of t, as _sweep does for one class."""
    in_val, in_col = inside
    out_val, out_col = outside
    with np.errstate(invalid="ignore"):
        u = out_val - in_val
    always = np.isfinite(in_val) & np.isneginf(out_val)
    movable = np.isfinite(u)
    order = np.flatnonzero(movable)
    order = order[np.argsort(u[order], kind="stable")]
    u_sorted = u[order]
    y = problem.y
    pred0 = np.where(always, in_col, out_col)
    tp, predicted = _counts(problem, pred0)
    w, recall, n_true = weights.w, weights.recall, problem.n_true
    base = float((w * _terms(tp, predicted, n_true, recall)).sum())

    gain, loss = in_col[order], out_col[order]
    hit_g = (y[order] == gain).astype(np.float64)
    hit_l = (y[order] == loss).astype(np.float64)
    cnt_g, tpc_g = _group_counts(gain, hit_g)
    cnt_l, tpc_l = _group_counts(loss, hit_l)
    delta = (w[gain] * (_terms(tp[gain] + tpc_g, predicted[gain] + cnt_g, n_true[gain], recall)
                        - _terms(tp[gain] + tpc_g - hit_g, predicted[gain] + cnt_g - 1, n_true[gain], recall))
             + w[loss] * (_terms(tp[loss] - tpc_l, predicted[loss] - cnt_l, n_true[loss], recall)
                          - _terms(tp[loss] - tpc_l + hit_l, predicted[loss] - cnt_l + 1, n_true[loss], recall)))
    objective = base + np.r_[0.0, np.cumsum(delta)]
    if len(order):
        last_of_run = np.r_[np.diff(u_sorted) > BREAKPOINT_TOLERANCE, True]
        stops = np.r_[0, np.flatnonzero(last_of_run) + 1]
        edges = u_sorted[last_of_run]
        lows = u_sorted[np.r_[True, last_of_run[:-1]]]
    else:
        stops, edges, lows = np.array([0]), np.empty(0), np.empty(0)
    return edges, objective[stops], lows


# ---- fitting ---------------------------------------------------------------------

@dataclass
class TuningResult:
    biases: np.ndarray           # (k,) per score column
    labels: list
    reference: Optional[str]     # the class kept at 0 to fix the common constant, if not locked by the user
    objective: str
    objective_before: float      # all biases at 0 (the plain argmax)
    objective_after: float
    counted: list                # classes in the objective
    absent: list                 # classes asked for but with no row in the pool
    passes: int
    converged: bool
    n_rows: int
    starts: int = 1              # starting points tried; the best end point is kept

    def as_dict(self) -> dict:
        return {str(l): float(b) for l, b in zip(self.labels, self.biases)}


def fit_biases(problem: Problem, spec: TuningSpec, start: Optional[np.ndarray] = None,
               progress: Optional[Callable[[int, int, int, int, int], None]] = None) -> TuningResult:
    """The biases maximising spec's objective on `problem`.

    From each starting point -- `start` if given, else all biases at 0 (the
    plain argmax), the class prior (b_c = -log of the class's share, the
    "logit adjustment"), and a few random points while the classes are few
    -- an ascent alternates two exact moves until neither helps: the best
    value of one class's bias with the others fixed (_sweep), and the best
    common shift of a pair of classes' biases (_group_sweep), which is what
    frees two classes confused with each other, neither of which gains by
    moving alone. The best end point over the starts is returned. The
    objective never decreases along an ascent; the result is the best
    point found, not a proven global optimum. `progress` gets (start,
    starts, pass, steps done, steps in the pass)."""
    weights = _weights(problem, spec)
    k = len(problem.labels)
    names = [str(l) for l in problem.labels]
    unknown = sorted(set(map(str, spec.locked)) - set(names))
    if unknown:
        raise ValueError(f"no score column for: {', '.join(unknown)}")
    locked = {names.index(str(l)) for l in spec.locked}
    reference = None
    if not locked:
        ref = int(np.argmax(problem.n_true))
        locked = {ref}
        reference = problem.labels[ref]
    free = [j for j in range(k) if j not in locked]
    before = _objective(problem, weights, predict(problem.logs, np.zeros(k)))

    starts = [np.zeros(k) if start is None else np.asarray(start, dtype=np.float64).copy()]
    if start is None:
        share = problem.n_true / max(1, problem.n_true.sum())
        with np.errstate(divide="ignore"):
            prior = np.where(share > 0, -np.log(np.where(share > 0, share, 1.0)), 0.0)
        anchor = sorted(locked)[0]
        if spec.prior_start:
            starts.append(prior - prior[anchor])
        n_random = spec.random_starts
        if n_random is None:
            n_random = RANDOM_STARTS if len(free) <= RANDOM_STARTS_MAX_CLASSES else 0
        rng = np.random.default_rng(spec.seed)
        starts += [rng.normal(0.0, RANDOM_START_SCALE, k) for _ in range(n_random)]

    best = None
    for index, point in enumerate(starts):
        for j in locked:
            point[j] = 0.0
        if spec.max_abs_bias is not None:
            point = np.clip(point, -spec.max_abs_bias, spec.max_abs_bias)
        step = None if progress is None else (lambda p, d, t, i=index: progress(i + 1, len(starts), p, d, t))
        found = _ascend(problem, spec, weights, point, free, step)
        if best is None or found[1] > best[1] + MIN_GAIN:
            best = found
    biases, current, passes, converged = best
    return TuningResult(biases, list(problem.labels), reference, spec.objective, before, current,
                        weights.counted, weights.absent, passes, converged, problem.n_rows, len(starts))


def _pairs(problem: Problem, free: list, pred: np.ndarray) -> list:
    """The pairs of free classes tried as a joint move: all of them while
    they are few, else the most confused pairs (rows of one predicted as
    the other, both ways) -- PAIRS_PER_CLASS per class."""
    if len(free) < 2:
        return []
    if len(free) <= PAIRS_ALL_MAX_CLASSES:
        return [(a, b) for i, a in enumerate(free) for b in free[i + 1:]]
    k = len(problem.labels)
    ok = (problem.y >= 0) & (pred >= 0) & (problem.y != pred)
    counts = np.bincount(problem.y[ok] * k + pred[ok], minlength=k * k).reshape(k, k)
    counts = counts + counts.T
    is_free = np.zeros(k, dtype=bool)
    is_free[free] = True
    counts[~is_free, :] = 0
    counts[:, ~is_free] = 0
    upper = np.triu(counts, 1)
    flat = np.flatnonzero(upper)
    flat = flat[np.argsort(-upper.ravel()[flat], kind="stable")][: PAIRS_PER_CLASS * len(free)]
    return [(int(f // k), int(f % k)) for f in flat]


def _ascend(problem: Problem, spec: TuningSpec, weights: _Weights, biases: np.ndarray, free: list,
            progress) -> tuple:
    """One ascent from `biases`: single-class moves, then pair moves when no
    single one helps, until neither does. Returns (biases, objective,
    passes, converged)."""
    bound = spec.max_abs_bias
    tops = _Tops(problem.logs, biases)
    pred = tops.prediction()
    current = _objective(problem, weights, pred)

    def accept(changes: dict) -> bool:
        nonlocal pred, current
        old = {c: biases[c] for c in changes}
        tops.set_biases(changes)
        candidate = tops.prediction()
        score = _objective(problem, weights, candidate)
        if score > current + MIN_GAIN:
            pred, current = candidate, score
            return True
        tops.set_biases(old)  # a tie the sweep could not see: keep the old biases
        return False

    passes, converged = 0, False
    for passes in range(1, spec.max_passes + 1):
        improved = False
        for done, c in enumerate(free):
            if progress is not None:
                progress(passes, done, len(free))
            sweep = _sweep(problem, c, tops.competitor(c), weights)
            for j in np.argsort(-sweep.objective, kind="stable"):
                if sweep.objective[j] <= current + MIN_GAIN:
                    break
                value = _interval_value(sweep.edges, int(j), bound, sweep.lows)
                if value is None:
                    continue
                improved |= accept({c: value})
                break
        if improved:
            continue
        pairs = _pairs(problem, free, pred) if spec.pair_moves else []
        for done, (a, b) in enumerate(pairs):
            if progress is not None:
                progress(passes, len(free) + done, len(free) + len(pairs))
            edges, objective, lows = _group_sweep(problem, tops.inside((a, b)), tops.outside((a, b)), weights)
            low = -np.inf if bound is None else max(-bound - biases[a], -bound - biases[b])
            high = np.inf if bound is None else min(bound - biases[a], bound - biases[b])
            for j in np.argsort(-objective, kind="stable"):
                if objective[j] <= current + MIN_GAIN:
                    break
                shift = _interval_point(edges, int(j), low, high, lows)
                if shift is None:
                    continue
                if accept({a: biases[a] + shift, b: biases[b] + shift}):
                    improved = True
                break
        if not improved:
            converged = True
            break
    return biases.copy(), current, passes, converged


@dataclass
class CrossFit:
    """How optimistic a tuning is: the biases fitted on one half of the rows
    and scored on the other half, both ways round."""
    in_sample: float        # fitted and scored on every row
    held_out: float         # mean over the folds of the score on the half not fitted on
    before: float           # the plain argmax, on every row
    folds: list             # (fitted on n rows, scored on n rows, score)


def cross_fit(problem: Problem, spec: TuningSpec, folds: int = 2, seed: int = 0,
              in_sample: Optional[float] = None) -> CrossFit:
    """The optimism of fitting the biases on the rows they are scored on --
    the case of tuning on the evaluation pool. Rows are split per class,
    so every fold keeps the class balance."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(problem.n_rows, dtype=np.int64)
    for cls in np.unique(problem.y):
        rows = np.flatnonzero(problem.y == cls)
        rng.shuffle(rows)
        fold_of[rows] = (np.arange(len(rows)) + rng.integers(folds)) % folds
    if in_sample is None:
        in_sample = fit_biases(problem, spec).objective_after
    results = []
    for f in range(folds):
        train, test = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
        fitted = fit_biases(problem.subset(train), spec)
        scored = objective_value(problem.subset(test), spec, fitted.biases)
        results.append((len(train), len(test), scored))
    before = objective_value(problem, spec, np.zeros(len(problem.labels)))
    return CrossFit(float(in_sample), float(np.mean([r[2] for r in results])), before, results)


# ---- one class's curve ---------------------------------------------------------------

@dataclass
class ClassSweep:
    """One class's bias swept with the others fixed: the deployable
    counterpart of its one-vs-rest precision-recall curve. One entry per
    interval of the bias between consecutive breakpoints (see _Sweep):
    `bias` is a value inside it."""
    label: str
    bias: np.ndarray
    edges: np.ndarray
    precision: np.ndarray    # NaN where nothing is predicted as the class
    recall: np.ndarray       # NaN when the class has no row
    f1: np.ndarray
    objective: np.ndarray    # NaN without a spec
    current_bias: float
    current: int             # the interval holding current_bias
    n_true: int


def class_sweep(problem: Problem, biases: np.ndarray, label, spec: Optional[TuningSpec] = None) -> ClassSweep:
    """Precision, recall, F1 of `label` -- and spec's objective -- for every
    value of its bias, the other biases as in `biases`."""
    names = [str(l) for l in problem.labels]
    if str(label) not in names:
        raise ValueError(f"no score column for {label!r}")
    c = names.index(str(label))
    biases = np.asarray(biases, dtype=np.float64).copy()
    tops = _Tops(problem.logs, biases)
    weights = _weights(problem, spec) if spec is not None else None
    sweep = _sweep(problem, c, tops.competitor(c), weights)
    n = float(problem.n_true[c])
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(sweep.pred_c > 0, sweep.tp_c / sweep.pred_c, np.nan)
        recall = sweep.tp_c / n if n > 0 else np.full(len(sweep.tp_c), np.nan)
        f1 = 2 * sweep.tp_c / (sweep.pred_c + n) if n > 0 else np.full(len(sweep.tp_c), np.nan)
    values = np.array([_interval_value(sweep.edges, j, None, sweep.lows) for j in range(len(sweep.tp_c))])
    current = int(np.searchsorted(sweep.edges, biases[c], side="left"))
    return ClassSweep(str(label), values, sweep.edges, precision, recall, f1, sweep.objective,
                      float(biases[c]), current, int(n))


# ---- applying to a frame ------------------------------------------------------------

def apply_to_frame(frame: pd.DataFrame, label_col: str, score_cols: list, labels_for_cols: list,
                   space: str, biases: np.ndarray) -> pd.DataFrame:
    """A copy of a merged frame as if the experiment had been trained with
    the biases: biased scores (biased_scores), and the predicted label,
    confidence and correctness derived from them -- what compute_metrics
    reads, so the metrics after the rule are computed by the same code."""
    from lookin.core.data_manager import argmax_ignoring_nan

    out = frame.copy()
    scores = biased_scores(frame[score_cols].to_numpy(), space, biases)
    out[score_cols] = scores
    idx, value = argmax_ignoring_nan(scores)
    labels = list(labels_for_cols[: len(score_cols)])
    out["predicted_label"] = [labels[i] if i >= 0 else None for i in idx]
    out["predicted_score"] = value
    out["correct"] = out["predicted_label"] == out[label_col]
    return out


# ---- the record of an applied rule ---------------------------------------------------
# What AppState.apply_decision_rule takes and the workspace stores (as JSON,
# MountExperiment.decision_rule). Its keys are a format: never renamed.

def make_record(score_cols: list, labels_for_cols: list, biases, space: str, tuned_on: str,
                split_column: str = "", split_value: str = "", n_rows: int = 0, objective: str = "",
                classes=None, locked=(), max_abs_bias=None, objective_before=None,
                objective_after=None) -> dict:
    """The record of a rule: the bias of each score column (keyed by the
    column, which a relabel does not change) and how it was obtained --
    what the badge, the workspace and a reader of an export need."""
    from datetime import datetime, timezone

    biases = np.asarray(biases, dtype=np.float64)
    labels = list(labels_for_cols[: len(score_cols)])

    def number(value):
        return None if value is None or value != value else float(value)

    return {
        "version": 1,
        "biases": {str(c): float(b) for c, b in zip(score_cols, biases)},
        "labels": {str(c): str(l) for c, l in zip(score_cols, labels)},
        "space": space,
        "tuned_on": tuned_on,
        "split_column": str(split_column or ""),
        "split_value": str(split_value or ""),
        "n_rows": int(n_rows),
        "objective": str(objective or ""),
        "classes": None if classes is None else [str(c) for c in classes],
        "locked": [str(c) for c in locked],
        "max_abs_bias": number(max_abs_bias),
        "objective_before": number(objective_before),
        "objective_after": number(objective_after),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def check_record(rule: dict, score_cols: list, raw_scores: np.ndarray) -> np.ndarray:
    """The bias of each of `score_cols` in `rule`, after checking that the
    rule fits these scores; raises ValueError, in words for the user,
    when it does not (a rule saved for another results file)."""
    if not isinstance(rule, dict) or not isinstance(rule.get("biases"), dict):
        raise ValueError("the decision rule has no biases")
    unknown = sorted(set(rule["biases"]) - set(map(str, score_cols)))
    if unknown:
        raise ValueError(f"the decision rule names score columns this experiment does not have: "
                         f"{', '.join(unknown[:5])}")
    space, reason = score_space(raw_scores)
    if space is None:
        raise ValueError(f"no bias can be applied to these scores: {reason}")
    if rule.get("space") != space:
        raise ValueError(f"the decision rule was made for scores read as {rule.get('space')!r}, but these "
                         f"read as {space!r}")
    try:
        biases = np.array([float(rule["biases"].get(str(c), 0.0)) for c in score_cols])
    except (TypeError, ValueError) as ex:
        raise ValueError(f"the decision rule holds a bias that is not a number: {ex}") from ex
    if not np.all(np.isfinite(biases)):
        raise ValueError("the decision rule holds a bias that is not finite")
    return biases


def is_tuned_on_evaluation(rule: Optional[dict]) -> bool:
    return bool(rule) and rule.get("tuned_on") == POOL_TEST


def record_to_text(rule: Optional[dict]) -> str:
    """A rule as the workspace stores it (MountExperiment.decision_rule);
    "" for none."""
    import json

    return json.dumps(rule, sort_keys=True) if rule else ""


def restore(state, experiment_name: str, text: str) -> Optional[str]:
    """Apply the rule a workspace stored for an experiment just loaded.
    Returns None, or a warning saying why the experiment is shown without
    it -- never raises: an unreadable rule must not stop a workspace from
    opening."""
    import json

    if not text:
        return None
    try:
        state.apply_decision_rule(experiment_name, json.loads(text))
    except (ValueError, TypeError, KeyError) as ex:
        return (f"The decision rule saved for experiment {experiment_name!r} was not applied, so it is "
                f"shown with its scores as read: {ex}")
    return None


# ---- references for one class ------------------------------------------------------

def class_point(problem: Problem, biases, label) -> tuple:
    """(recall, precision) of `label` under `biases`; NaN where undefined."""
    c = [str(l) for l in problem.labels].index(str(label))
    pred = predict(problem.logs, np.asarray(biases, dtype=np.float64))
    tp = float(((pred == c) & (problem.y == c)).sum())
    n_pred, n_true = float((pred == c).sum()), float(problem.n_true[c])
    return (tp / n_true if n_true else float("nan"), tp / n_pred if n_pred else float("nan"))


def ovr_curve(frame: pd.DataFrame, label_col: str, score_col: str, label) -> Optional[tuple]:
    """(recall, precision) of the one-vs-rest precision-recall curve of
    `label` on its own score column, as the Metrics tab draws it -- the
    reference no single-label rule can reach. None when it is undefined
    (no positive, a constant or non-finite column)."""
    from lookin.core.metrics import ovr_curves

    scores = frame[[score_col]].to_numpy()
    positives = [np.flatnonzero(frame[label_col].to_numpy() == label)]
    curves = ovr_curves(scores, positives)
    if 0 not in curves:
        return None
    _fpr, _tpr, recall, precision, _ap = curves[0]
    return recall, precision


# ---- one call for what a tool needs ---------------------------------------------------

@dataclass
class Setup:
    """The evaluated rows and the tuning rows of one experiment, read and
    checked: what tune and apply both start from."""
    eval_frame: pd.DataFrame
    eval_problem: Problem
    pool: TuningPool
    tuning_problem: Problem
    check: PoolCheck


def setup(state, experiment_name: str, tune_on: str = POOL_VAL, split_column: str = "",
          split_value: str = DEFAULT_VAL_VALUE) -> Setup:
    """Read and check everything a tuning needs, from the scores as read
    (whatever rule is applied). Raises ValueError, in words for the user,
    when the rows cannot be used -- never falls back to other rows."""
    if tune_on not in POOL_KEYS:
        raise ValueError(f"tune_on must be one of {', '.join(POOL_KEYS)}; got {tune_on!r}.")
    if state.dataset is None:
        raise ValueError("The workspace has no dataset loaded.")
    experiment = state.experiments[experiment_name]
    cols = state.dataset.columns
    labels = list(experiment.class_labels[: len(experiment.score_cols)])
    eval_frame = state.raw_merged(experiment_name)
    eval_problem = prepare(eval_frame, cols.label_col, experiment.score_cols, labels)
    if tune_on == POOL_TEST:
        return Setup(eval_frame, eval_problem, evaluation_as_tuning_pool(eval_frame), eval_problem,
                     check_pools([], [], same_pool=True))
    if not split_column:
        row_filter = state.dataset.row_filter
        split_column = row_filter[0] if row_filter and row_filter[0] else DEFAULT_SPLIT_COLUMN
    pool = read_tuning_pool(state, experiment_name, split_column, split_value or DEFAULT_VAL_VALUE)
    if pool.frame is None:
        raise ValueError(pool.disabled_reason)
    check = check_pools(pool.frame[cols.uuid_col], eval_frame[cols.uuid_col], same_pool=False)
    if check.blocking:
        raise ValueError(check.blocking)
    tuning_problem = prepare(pool.frame, cols.label_col, experiment.score_cols, labels, space=eval_problem.space)
    return Setup(eval_frame, eval_problem, pool, tuning_problem, check)


def biases_from_labels(labels: list, given: dict) -> tuple:
    """(bias per score column, labels left at 0) from {class label: bias};
    raises ValueError on a label with no score column or a bias that is not
    a finite number."""
    names = [str(l) for l in labels]
    unknown = sorted(set(map(str, given)) - set(names))
    if unknown:
        raise ValueError(f"no score column for: {', '.join(unknown[:10])}")
    biases = np.zeros(len(names))
    for label, value in given.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as ex:
            raise ValueError(f"the bias of {label!r} is not a number: {value!r}") from ex
        if not np.isfinite(value):
            raise ValueError(f"the bias of {label!r} is not finite")
        biases[names.index(str(label))] = value
    return biases, [l for l in names if l not in set(map(str, given))]
