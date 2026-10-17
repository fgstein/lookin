"""Metrics restricted to a slice of the data: the same accuracy the Metrics
tab reports globally, recomputed on the rows where one metadata column takes
one value.

Why this exists
---------------
"The model is at 94%" is an average, and an average over a dataset that
mixes sensors, sites, acquisition times and crop sizes hides exactly the
thing a user is looking for. The per-class table already splits that average
by *label*; nothing split it by *metadata*, even though the merged frame has
carried every metadata column alongside `correct` all along. A model at 94%
overall and 61% on one sensor is a different problem from a model at 94%
everywhere, and only the first one is fixable by looking at that sensor.

What a slice is
---------------
One value of one column, for a categorical/binary column or a numeric one
whose values are *discrete*; one quantile bin, for a continuous numeric
column. Quantile bins rather than equal-width ones because the point is to
compare slices to each other: equal-width bins on a skewed column put 99%
of the rows in the first bin and three rows in the last, and the last bin's
"40% accuracy" is then noise being read as a finding.
Rows where the column is null get their own slice (MISSING_SLICE) rather
than being dropped -- "the model fails where this metadata is absent" is a
real answer, and silently dropping those rows would hide it.

Discrete columns get one slice per value
----------------------------------------
A column of numbers is not necessarily a scale. A sensor id, a gain
setting, a number of channels, a count of objects in the frame: these are
labels that happen to be written as integers, and cutting them into
deciles produces bins like "[2, 5)" that mix three settings the user
wanted told apart -- and empty bins wherever the values are not evenly
spread. So a numeric column with no more than DISCRETE_MAX_VALUES distinct
values is sliced per value, and the choice is overridable either way
(BINNING_VALUES / BINNING_QUANTILE) because no threshold gets every column
right.

Metrics that depend on a class
------------------------------
Accuracy over a slice answers "does the model work here". It does not
answer "does the model still confuse class A with class B here", and that
is usually the actual question: a false-alarm rate on one class can triple
on one sensor while overall accuracy moves a point, because that class is
5% of the rows. Every row therefore also carries per-class counts (tp, fp,
fn, tn, one set per class), from which recall, precision, F1, the false
alarm rate (FPR) and the miss rate are derived on demand -- see
metric_value(). They are counts rather than pre-computed rates because
counts are what the confidence intervals need, and because storing five
rates per class per slice would be five times the memory for the same
information.

The small-slice trap
--------------------
Slicing turns one number into twenty, and the worst of twenty numbers is
low even when nothing is wrong: with 30 rows in a slice, an accuracy of
0.73 and one of 0.93 are the same measurement. That is why every row here
carries its Wilson score interval and its own n, and why sort_rows() can
hold slices below a minimum size out of the ranking instead of letting them
top a "worst first" list they only reached by being small. The interval is
Wilson rather than the normal approximation because the normal one is
actively wrong in exactly this regime (small n, proportion near 1) -- it
happily produces upper bounds above 1.

Class mix is not constant across slices
---------------------------------------
Accuracy and macro-F1 are comparable between two slices only to the extent
that the slices pose the same problem. A slice holding two classes and a
slice holding ten do not, so every row also reports how many classes it
actually contains -- the number that says whether the comparison above it
means anything.

Everything here is plain pandas/numpy over the merged frame (see
AppState.merged), so it is testable without a UI and costs one groupby-like
pass per experiment rather than anything sklearn-shaped per row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lookin.core.data_manager import HIGH_CARDINALITY_MAX_UNIQUE
from lookin.core.point_coloring import group_name, natural_sort_key

# Slice kinds. Keys, compared in code (the view branches on them to decide
# whether to offer a bin-count field), never shown as-is to the user.
KIND_CATEGORICAL = "categorical"
KIND_NUMERIC = "numeric"

# The slice holding rows whose column is null. A display string, but also
# the value a row is grouped under, so it has to be a value the column
# itself cannot produce -- parentheses are not a legal float and would be a
# bizarre category.
MISSING_SLICE = "(missing)"

# Row-count floor under which a slice is flagged as too small to rank on.
# 30 is a convention, not a theorem: it is roughly where the Wilson interval
# on a mid-range proportion narrows to something you can act on, and it is
# adjustable from the UI.
DEFAULT_MIN_ROWS = 30

# Quantile bins for a numeric column. Ten deciles is enough resolution to
# see a monotone trend (accuracy falling with crop area) without producing a
# table nobody reads.
DEFAULT_BINS = 10
MAX_BINS = 50

# How a column is cut. Keys, compared in code and held by the panel's
# dropdown, never shown as-is to the user.
BINNING_AUTO = "auto"          # discrete -> one slice per value, continuous -> bins
BINNING_VALUES = "values"      # one slice per distinct value, whatever the column is
BINNING_QUANTILE = "quantile"  # quantile bins, even for a column of few values

# Above this many distinct values, a numeric column is treated as a scale
# and cut into bins; at or below it, the numbers are read as labels and each
# gets its own slice. Well above the 20 of data_manager.CATEGORICAL_MAX_UNIQUE
# (which decides whether a column gets one *UI control* per value) because a
# bar in a chart is far cheaper than a checkbox, and a gain setting with 30
# steps is still a setting rather than a scale.
DISCRETE_MAX_VALUES = 40

# How rows can be ordered. Keys, compared in code.
SORT_WORST = "worst"
SORT_VALUE = "value"
SORT_SIZE = "size"

# 95% two-sided normal quantile, for the Wilson interval below.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple:
    """Wilson score interval for a binomial proportion.

    Used instead of the textbook normal approximation because slices are
    routinely small and routinely near-perfect, which is precisely where the
    normal interval breaks: at 29/30 it returns an upper bound above 1 and a
    width that understates the real uncertainty. Wilson stays inside [0, 1]
    and degrades gracefully at n = 0 (the whole interval), which is what a
    table of slices needs, since some slice always ends up nearly empty.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class SliceSpec:
    """How one column is cut into slices, computed once and reused for every
    experiment.

    Reused rather than recomputed per experiment on purpose: each experiment
    joins to a different subset of the dataset (uuid overlap is rarely
    total), so deriving quantile edges from each experiment's own merged
    frame would give each one slightly different bins, and the resulting
    table would compare "[1.2, 3.4)" against "[1.3, 3.5)" as though they
    were the same slice.
    """
    column: str
    kind: str
    names: list = field(default_factory=list)   # slice names, in display order
    edges: Optional[np.ndarray] = None          # numeric only: len(names)+1 bin edges
    n_missing: int = 0
    # Categorical values dropped because the column had more distinct values
    # than the cap. Reported rather than silently applied: a table missing
    # half its slices with no explanation is worse than no table.
    n_values_dropped: int = 0
    note: str = ""


def _format_edge(value: float) -> str:
    return f"{value:.4g}"


def build_spec(series: pd.Series, bins: int = DEFAULT_BINS,
               max_values: int = HIGH_CARDINALITY_MAX_UNIQUE,
               binning: str = BINNING_AUTO) -> SliceSpec:
    """Decide the slices for one column, from the dataset's own values.

    Under BINNING_AUTO a numeric column is cut into `bins` quantile bins
    only if it has enough distinct values to be a scale (see
    DISCRETE_MAX_VALUES); everything else -- text, booleans, and numbers
    that are really labels -- gets one slice per distinct value.
    BINNING_VALUES and BINNING_QUANTILE say so explicitly, for the columns
    the threshold gets wrong in either direction.
    """
    column = str(series.name)
    n_missing = int(series.isna().sum())
    present = series.dropna()

    is_numeric = pd.api.types.is_numeric_dtype(series)
    n_unique = int(present.nunique())

    if is_numeric and n_unique > 2 and _wants_bins(binning, n_unique, bins):
        values = present.to_numpy(dtype=float)
        bins = max(2, min(int(bins), MAX_BINS))
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.nanquantile(values, quantiles))
        if edges.size < 2:
            # A column that is constant apart from its nulls: one slice is
            # the honest answer, not an error.
            edges = np.array([edges[0], edges[0]]) if edges.size else np.array([0.0, 0.0])
        names = []
        for i in range(len(edges) - 1):
            closing = "]" if i == len(edges) - 2 else ")"
            names.append(f"[{_format_edge(edges[i])}, {_format_edge(edges[i + 1])}{closing}")
        note = ""
        if len(names) < bins:
            # Repeated quantiles collapse: a column where 60% of the rows
            # share one value cannot produce ten distinct deciles.
            note = (f"{bins} bins requested, {len(names)} produced -- repeated values "
                    f"collapse quantile edges.")
        return SliceSpec(column=column, kind=KIND_NUMERIC, names=names, edges=edges,
                         n_missing=n_missing, note=note)

    ordered = sorted(present.unique().tolist(), key=natural_sort_key)
    dropped = 0
    if len(ordered) > max_values:
        dropped = len(ordered) - max_values
        ordered = ordered[:max_values]
    # group_name() rather than str(): the dataset frame is downcast to
    # float32 on load, so str() on a value of 34.7 spells it
    # "34.70000076293945" -- and that string would be the slice's name, its
    # bar label and the key assign() matches rows against.
    names = [group_name(v) for v in ordered]
    note = ""
    if is_numeric and binning == BINNING_AUTO and n_unique > 2:
        note = (f"'{column}' has {n_unique} distinct value(s), so it gets one slice per value "
                f"rather than quantile bins. Set Binning to 'Quantile bins' to cut it into a "
                f"scale instead.")
    return SliceSpec(column=column, kind=KIND_CATEGORICAL, names=names,
                     n_missing=n_missing, n_values_dropped=dropped, note=note)


def _wants_bins(binning: str, n_unique: int, bins: int) -> bool:
    """Whether a numeric column with `n_unique` distinct values gets cut
    into quantile bins. Split out because the auto rule is the one thing
    here a reader is likely to come looking for."""
    if binning == BINNING_VALUES:
        return False
    if binning == BINNING_QUANTILE:
        return True
    # Auto: more distinct values than the discrete cap means a scale. The
    # requested bin count is part of the test for the same reason it always
    # was -- asking for 50 bins from a column of 45 values cannot produce
    # them, so per-value is both the honest and the finer answer.
    return n_unique > max(DISCRETE_MAX_VALUES, min(bins, MAX_BINS))


def assign(series: pd.Series, spec: SliceSpec) -> np.ndarray:
    """Map every row to its slice name, using the edges/values decided by
    build_spec. Rows that fall outside the spec -- a null, or a categorical
    value dropped by the cap -- get an empty name and are excluded from the
    report by slice_report(), except nulls, which become MISSING_SLICE."""
    out = np.full(len(series), "", dtype=object)
    missing = series.isna().to_numpy()
    out[missing] = MISSING_SLICE

    if spec.kind == KIND_NUMERIC and spec.edges is not None and len(spec.names):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        # right=False everywhere except the last bin, which has to be closed
        # so the maximum value lands in a bin rather than one past the end.
        idx = np.searchsorted(spec.edges, values, side="right") - 1
        idx = np.clip(idx, 0, len(spec.names) - 1)
        valid = ~missing & np.isfinite(values)
        out[valid] = np.asarray(spec.names, dtype=object)[idx[valid]]
        return out

    # group_name(), matching build_spec above: the two have to spell a value
    # the same way or every row would fall outside its own slice.
    known = set(spec.names)
    valid = ~missing
    spelled = [group_name(v) for v in series.to_numpy(dtype=object)[valid]]
    out[valid] = [name if name in known else "" for name in spelled]
    return out


@dataclass
class ClassCounts:
    """One class's one-vs-rest confusion inside one slice.

    Counts rather than rates, for two reasons: every rate below is derived
    from them in one line, and a rate on its own cannot carry a confidence
    interval -- "false alarm rate 0.20" is a finding at 40 negatives and
    noise at 5, and only the counts know which.
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def support(self) -> int:
        """Rows whose true label is this class."""
        return self.tp + self.fn

    @property
    def predicted(self) -> int:
        """Rows the model assigned to this class."""
        return self.tp + self.fp

    @property
    def negatives(self) -> int:
        """Rows whose true label is anything else -- what a false alarm
        rate is a rate *of*."""
        return self.fp + self.tn


# Metrics a slice can be measured with. Keys, compared in code and held by
# the panel's dropdown; the labels beside them live in ui/slice_panel.py.
METRIC_ACCURACY = "accuracy"
METRIC_MACRO_F1 = "macro_f1"
METRIC_MEAN_CONFIDENCE = "mean_confidence"
# ...and the ones that only mean something for one named class.
METRIC_RECALL = "recall"
METRIC_PRECISION = "precision"
METRIC_F1 = "f1"
METRIC_FALSE_ALARM = "false_alarm"
METRIC_MISS_RATE = "miss_rate"
METRIC_SUPPORT = "support"
METRIC_PREDICTED = "predicted"

PER_CLASS_METRICS = (
    METRIC_RECALL, METRIC_PRECISION, METRIC_F1, METRIC_FALSE_ALARM,
    METRIC_MISS_RATE, METRIC_SUPPORT, METRIC_PREDICTED,
)
# Row counts rather than rates: they share no y axis with the rest and have
# no confidence interval to draw.
COUNT_METRICS = (METRIC_SUPPORT, METRIC_PREDICTED)

# Metrics a *high* value of which is the bad news. "Worst first" is the
# default order, so getting this wrong would put the best slices at the top
# of a table read as a list of problems.
HIGHER_IS_WORSE = (METRIC_FALSE_ALARM, METRIC_MISS_RATE)


@dataclass
class SliceRow:
    """One slice's numbers for one experiment."""
    value: str
    n: int
    n_errors: int
    accuracy: float
    ci_low: float
    ci_high: float
    macro_f1: float
    mean_confidence: float
    # How many distinct true labels the slice actually contains -- the
    # number that says whether comparing this row's accuracy to another
    # row's is comparing like with like (see the module docstring).
    n_classes: int
    # class label -> its one-vs-rest counts in this slice. Every row in a
    # report carries the same keys, including classes with no row in this
    # slice at all: a class that disappears from a slice is a finding, and
    # a table whose columns changed per row could not show it.
    per_class: dict = field(default_factory=dict)

    @property
    def small(self) -> bool:
        return self.n < DEFAULT_MIN_ROWS

    def is_small(self, min_rows: int) -> bool:
        return self.n < min_rows


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def metric_fraction(row: SliceRow, metric: str, class_label=None) -> tuple:
    """(successes, total) behind one metric, or (None, None) when it is not
    a proportion.

    The one place that knows what each rate is a rate *of*, so the value and
    its confidence interval can never be computed from different counts.
    """
    if metric == METRIC_ACCURACY:
        return row.n - row.n_errors, row.n
    if metric not in PER_CLASS_METRICS or metric in COUNT_METRICS:
        return None, None
    counts = row.per_class.get(class_label)
    if counts is None:
        return None, None
    if metric == METRIC_RECALL:
        return counts.tp, counts.support
    if metric == METRIC_PRECISION:
        return counts.tp, counts.predicted
    if metric == METRIC_FALSE_ALARM:
        return counts.fp, counts.negatives
    if metric == METRIC_MISS_RATE:
        return counts.fn, counts.support
    return None, None  # F1 is not a proportion of anything countable


def metric_value(row: SliceRow, metric: str, class_label=None) -> float:
    """What one slice scores on one metric. NaN when the metric is
    undefined here -- a class with no row in the slice, a precision with no
    prediction to divide by -- rather than 0, which would plot as a real
    result at the bottom of the axis."""
    if metric in (METRIC_ACCURACY, METRIC_MACRO_F1, METRIC_MEAN_CONFIDENCE):
        return float(getattr(row, metric))
    counts = row.per_class.get(class_label)
    if counts is None:
        return float("nan")
    if metric == METRIC_SUPPORT:
        return float(counts.support)
    if metric == METRIC_PREDICTED:
        return float(counts.predicted)
    if metric == METRIC_F1:
        # 2*tp / (2*tp + fp + fn) -- the same number as the harmonic mean of
        # precision and recall, without the two divisions that make it NaN
        # when a class is absent from the slice entirely.
        return _ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn)
    successes, total = metric_fraction(row, metric, class_label)
    if successes is None:
        return float("nan")
    return _ratio(successes, total)


def metric_interval(row: SliceRow, metric: str, class_label=None) -> tuple:
    """The 95% Wilson interval around metric_value(), or (NaN, NaN) for the
    metrics that are not proportions of a countable denominator."""
    successes, total = metric_fraction(row, metric, class_label)
    if successes is None:
        return (float("nan"), float("nan"))
    return wilson_interval(int(successes), int(total))


@dataclass
class SliceReport:
    experiment: str
    column: str
    rows: list = field(default_factory=list)     # list[SliceRow], spec order
    overall: Optional[SliceRow] = None           # every joined row, for reference
    n_unassigned: int = 0                        # rows whose value fell outside the spec
    # The classes every row carries counts for, in display order.
    class_labels: list = field(default_factory=list)


def _row(value: str, correct: np.ndarray, confidence: np.ndarray,
         y_true: np.ndarray, y_pred: np.ndarray, class_labels: list) -> SliceRow:
    n = int(len(correct))
    n_correct = int(correct.sum())
    accuracy = n_correct / n if n else float("nan")
    lo, hi = wilson_interval(n_correct, n)

    if n:
        # A crop with no score has no prediction (NaN): it is not a class,
        # and sorting it among the class names raised. It still counts
        # against its true class, as f1_score counts a prediction outside
        # `labels` -- the counts below are f1_score(labels=present,
        # average="macro", zero_division=0) without its refusal of the mix.
        present = sorted(_present(y_true) | _present(y_pred))
        counts = _class_counts(y_true, y_pred, present)
        macro_f1 = float(np.mean([2 * c.tp / (2 * c.tp + c.fp + c.fn) if c.tp + c.fp + c.fn else 0.0
                                  for c in counts.values()])) if counts else float("nan")
        n_classes = int(len(_present(y_true)))
        mean_confidence = float(np.nanmean(confidence)) if confidence.size else float("nan")
    else:
        macro_f1 = float("nan")
        n_classes = 0
        mean_confidence = float("nan")

    return SliceRow(
        value=value, n=n, n_errors=n - n_correct, accuracy=accuracy,
        ci_low=lo, ci_high=hi, macro_f1=macro_f1,
        mean_confidence=mean_confidence, n_classes=n_classes,
        per_class=_class_counts(y_true, y_pred, class_labels),
    )


def _present(values: np.ndarray) -> set:
    """The distinct labels among `values`, a missing one left out."""
    return {v for v in pd.unique(pd.Series(values, dtype=object)) if not pd.isna(v)}


def _class_counts(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list) -> dict:
    """One-vs-rest tp/fp/fn/tn per class over one slice.

    Three counts over integer-coded labels rather than two passes over the
    rows per class: at two thousand classes the per-class version was ten
    seconds for a three-slice column, and every one of those passes came
    down to these same three numbers.
    """
    n = int(len(y_true))
    index = {label: i for i, label in enumerate(dict.fromkeys(class_labels))}
    k = len(index)

    def codes(values: np.ndarray) -> np.ndarray:
        return pd.Series(values, dtype=object).map(index).fillna(-1).to_numpy(dtype=np.int64)

    t, p = codes(y_true), codes(y_pred)
    true_n = np.bincount(t[t >= 0], minlength=k)
    pred_n = np.bincount(p[p >= 0], minlength=k)
    tp = np.bincount(t[(t >= 0) & (t == p)], minlength=k)
    return {
        label: ClassCounts(tp=int(tp[i]), fp=int(pred_n[i] - tp[i]), fn=int(true_n[i] - tp[i]),
                           tn=n - int(true_n[i] + pred_n[i] - tp[i]))
        for label, i in index.items()
    }


def slice_report(experiment: str, merged: pd.DataFrame, label_col: str, spec: SliceSpec,
                 class_labels: Optional[list] = None) -> SliceReport:
    """Per-slice metrics for one experiment, in the spec's order.

    `merged` is AppState.merged(experiment) -- so it already carries the
    review overlay, the row filter and the split the rest of the app is
    looking at, and a slice here means the same population a confusion-matrix
    cell would.

    `class_labels` is which classes get one-vs-rest counts on every row.
    Passed in rather than derived here so that two experiments covering
    different subsets of the data still produce tables with the same
    columns; derived from this experiment's own rows when omitted.
    """
    if spec.column not in merged.columns or len(merged) == 0:
        return SliceReport(experiment=experiment, column=spec.column)

    names = assign(merged[spec.column], spec)
    correct_all = merged["correct"].to_numpy().astype(bool)
    confidence_all = merged["predicted_score"].to_numpy(dtype=float)
    y_true_all = merged[label_col].to_numpy()
    y_pred_all = merged["predicted_label"].to_numpy()
    if class_labels is None:
        class_labels = sorted(_present(y_true_all) | _present(y_pred_all), key=natural_sort_key)

    report = SliceReport(
        experiment=experiment, column=spec.column, class_labels=list(class_labels),
        overall=_row("all rows", correct_all, confidence_all, y_true_all, y_pred_all,
                     class_labels),
    )

    order = list(spec.names)
    if spec.n_missing:
        order.append(MISSING_SLICE)

    # One boolean mask per slice rather than a groupby: the slice names come
    # from the spec (so an empty slice still gets a row, which a groupby
    # would drop) and the masks are what the per-slice sklearn calls need
    # anyway.
    for value in order:
        mask = names == value
        if not mask.any() and value == MISSING_SLICE:
            continue
        report.rows.append(_row(
            value, correct_all[mask], confidence_all[mask], y_true_all[mask], y_pred_all[mask],
            class_labels,
        ))

    report.n_unassigned = int((names == "").sum())
    return report


def sort_rows(rows: list, how: str = SORT_WORST, min_rows: int = DEFAULT_MIN_ROWS,
              metric: str = METRIC_ACCURACY, class_label=None) -> list:
    """Order slices for display.

    SORT_WORST is the default because it is the question the tab is for --
    but slices under `min_rows` are pushed to the bottom rather than allowed
    to lead it. A slice of four crops at 50% is not the worst slice, it is
    an unmeasured one, and letting it head the table sends the user to look
    at four images while a real 12-point drop sits three rows down.

    "Worst" is worst *on the metric being shown*: a table of false alarm
    rates ranked by accuracy would lead with a slice that is not the one the
    user is looking at. A count has no worst, so it ranks on accuracy.
    """
    if how == SORT_VALUE:
        return list(rows)
    if how == SORT_SIZE:
        return sorted(rows, key=lambda r: -r.n)
    ranked = METRIC_ACCURACY if metric in COUNT_METRICS else metric
    sign = -1.0 if ranked in HIGHER_IS_WORSE else 1.0
    def rank(row: SliceRow):
        value = metric_value(row, ranked, class_label)
        return (
            row.is_small(min_rows),
            sign * value if value == value else 2.0,  # NaN last within its group
        )
    return sorted(rows, key=rank)
