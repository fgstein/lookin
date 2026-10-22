"""Head-to-head comparison of two loaded experiments, crop by crop.

The Metrics tab overlays N experiments and answers "how good is each one".
This answers a different question, the one actually asked after a
retraining run: *what did the new model break, and what did it fix?* Two
accuracy numbers cannot answer that -- a model can gain a point overall
while breaking two hundred crops it used to get right, and those two
hundred are the ones worth looking at.

Everything here works on the *intersection* of the two experiments' uuids.
Two results CSVs need not cover the same crops (a subsampled evaluation, a
run that crashed halfway), and comparing counts over different row sets
would be meaningless, so the overlap is computed explicitly and reported
rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lookin.core.metrics import ExperimentMetrics, compute_metrics

# Outcome of one crop under the two models. Keys, compared in code and used
# as dict keys below -- only their labels in web/views/compare.py are display
# text.
OUTCOME_REGRESSION = "regression"    # baseline right, candidate wrong
OUTCOME_FIX = "fix"                  # baseline wrong, candidate right
OUTCOME_BOTH_RIGHT = "both_right"
OUTCOME_BOTH_WRONG = "both_wrong"
OUTCOMES = (OUTCOME_REGRESSION, OUTCOME_FIX, OUTCOME_BOTH_RIGHT, OUTCOME_BOTH_WRONG)

# The three confusion matrices the Compare tab can show. Keys, compared in
# code -- only their labels in web/views/compare.py are display text.
# MATRIX_NET is the candidate-minus-baseline difference; the other two are
# the gross flows it nets against each other, and they satisfy
# moved_in - moved_out == net cell by cell (see movement_confusion).
MATRIX_NET = "net"
MATRIX_MOVED_IN = "moved_in"
MATRIX_MOVED_OUT = "moved_out"
MATRIX_VIEWS = (MATRIX_NET, MATRIX_MOVED_IN, MATRIX_MOVED_OUT)
MOVEMENTS = (MATRIX_MOVED_IN, MATRIX_MOVED_OUT)

# The two things the provenance panel can show for the cell that is picked
# in one of those matrices. Keys, compared in code -- only their labels in
# web/views/compare.py are display text.
FLOW_CELL = "cell"      # only the crops that touch the picked cell
FLOW_CLASS = "class"    # every crop of the picked cell's true class
FLOW_MODES = (FLOW_CELL, FLOW_CLASS)

# Columns of cell_flows. Also keys.
COL_CAME_FROM = "came_from"
COL_WENT_TO = "went_to"

# Column names added to the joined frame. Also keys.
COL_OUTCOME = "outcome"
COL_TRUE = "true_label"
COL_PRED_A = "predicted_baseline"
COL_PRED_B = "predicted_candidate"
COL_SCORE_A = "confidence_baseline"
COL_SCORE_B = "confidence_candidate"


@dataclass
class ExperimentDiff:
    """Everything the Compare tab shows, computed in one pass."""
    baseline: str
    candidate: str
    labels: list
    n_common: int
    n_baseline_only: int
    n_candidate_only: int
    counts: dict = field(default_factory=dict)  # outcome key -> n
    baseline_accuracy: float = float("nan")
    candidate_accuracy: float = float("nan")
    # (candidate - baseline) counts per (true, predicted) cell, over the
    # common rows only. Same index/columns as either model's own confusion
    # matrix, so the two read the same way.
    delta_confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    # The same cells, but counting one direction of the movement only and
    # netting nothing -- see movement_confusion. Same index/columns again,
    # so all three read the same way and can be flipped between in place.
    moved_in_confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    moved_out_confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Crops whose predicted class is not the same under the two models.
    # Every one of them is one arrival and one departure, so this is the
    # total of either gross matrix.
    n_moved: int = 0

    def confusion(self, matrix_view: str) -> pd.DataFrame:
        """One of the three matrices, by MATRIX_VIEWS key."""
        return {
            MATRIX_NET: self.delta_confusion,
            MATRIX_MOVED_IN: self.moved_in_confusion,
            MATRIX_MOVED_OUT: self.moved_out_confusion,
        }[matrix_view]

    @property
    def accuracy_delta(self) -> float:
        return self.candidate_accuracy - self.baseline_accuracy

    def net_change(self) -> int:
        """Fixes minus regressions -- the number the accuracy delta hides.
        A model can gain accuracy overall and still be a wash here."""
        return self.counts.get(OUTCOME_FIX, 0) - self.counts.get(OUTCOME_REGRESSION, 0)


def build_comparison(state, baseline: str, candidate: str) -> pd.DataFrame:
    """One row per crop the two experiments have in common, carrying both
    models' verdicts plus everything the image carousel needs.

    Built by joining the two merged frames rather than merging three tables
    at once, so it goes through AppState.merged() and inherits its review
    overlay: with reviewed labels on, both models are judged against the
    same corrected labels, which is the only way the comparison stays
    honest.
    """
    if state.dataset is None:
        raise ValueError("No dataset loaded.")
    for name in (baseline, candidate):
        if name not in state.experiments:
            raise KeyError(f"No experiment named '{name}' is loaded.")

    cols = state.dataset.columns
    keep = [cols.uuid_col, cols.crop_col, cols.label_col, "predicted_label", "predicted_score"]

    # Fetched one at a time and immediately narrowed: AppState.merged()
    # caches a single experiment, so asking for the second evicts the
    # first. Copying five columns is cheap; re-merging is not.
    left = state.merged(baseline)[keep].copy()
    right = state.merged(candidate)[[cols.uuid_col, "predicted_label", "predicted_score"]].copy()

    joined = left.merge(right, on=cols.uuid_col, how="inner", suffixes=("_a", "_b"))
    joined = joined.rename(columns={
        cols.label_col: COL_TRUE,
        "predicted_label_a": COL_PRED_A, "predicted_score_a": COL_SCORE_A,
        "predicted_label_b": COL_PRED_B, "predicted_score_b": COL_SCORE_B,
    })

    correct_a = joined[COL_PRED_A] == joined[COL_TRUE]
    correct_b = joined[COL_PRED_B] == joined[COL_TRUE]
    joined[COL_OUTCOME] = np.select(
        [correct_a & ~correct_b, ~correct_a & correct_b, correct_a & correct_b],
        [OUTCOME_REGRESSION, OUTCOME_FIX, OUTCOME_BOTH_RIGHT],
        default=OUTCOME_BOTH_WRONG,
    )
    return joined


def compare_experiments(state, baseline: str, candidate: str,
                        joined: pd.DataFrame | None = None) -> ExperimentDiff:
    """Summary + differential confusion matrix. Pass `joined` (from
    build_comparison) to avoid recomputing it when the caller already has
    it -- the view keeps it for the crop lists."""
    cols = state.dataset.columns
    if joined is None:
        joined = build_comparison(state, baseline, candidate)

    uuids_a = set(state.merged(baseline)[cols.uuid_col].astype(str))
    uuids_b = set(state.merged(candidate)[cols.uuid_col].astype(str))

    counts = {outcome: int((joined[COL_OUTCOME] == outcome).sum()) for outcome in OUTCOMES}
    n = len(joined)

    # Labels come from the union of what either model can predict plus what
    # actually appears, so a class only one of them ever predicts still gets
    # a row/column instead of silently vanishing from the difference.
    labels = sorted(set(joined[COL_TRUE].dropna()) |
                    set(joined[COL_PRED_A].dropna()) | set(joined[COL_PRED_B].dropna()))

    delta = _delta_confusion(joined, labels)

    return ExperimentDiff(
        baseline=baseline, candidate=candidate, labels=labels,
        n_common=n,
        n_baseline_only=len(uuids_a - uuids_b),
        n_candidate_only=len(uuids_b - uuids_a),
        counts=counts,
        baseline_accuracy=(counts[OUTCOME_BOTH_RIGHT] + counts[OUTCOME_REGRESSION]) / n if n else float("nan"),
        candidate_accuracy=(counts[OUTCOME_BOTH_RIGHT] + counts[OUTCOME_FIX]) / n if n else float("nan"),
        delta_confusion=delta,
        moved_in_confusion=movement_confusion(joined, labels, MATRIX_MOVED_IN),
        moved_out_confusion=movement_confusion(joined, labels, MATRIX_MOVED_OUT),
        n_moved=int((joined[COL_PRED_A] != joined[COL_PRED_B]).sum()),
    )


def _delta_confusion(joined: pd.DataFrame, labels: list) -> pd.DataFrame:
    """candidate minus baseline, cell by cell, over the common rows."""
    def counts_for(pred_col: str) -> pd.DataFrame:
        table = pd.crosstab(joined[COL_TRUE], joined[pred_col])
        return table.reindex(index=labels, columns=labels, fill_value=0)

    return (counts_for(COL_PRED_B) - counts_for(COL_PRED_A)).astype(int)


def movement_confusion(joined: pd.DataFrame, labels: list, direction: str) -> pd.DataFrame:
    """Where the crops that changed prediction arrived, or where they left
    from, cell by cell of the usual true-by-predicted grid.

    The difference matrix nets the two against each other, which is the
    right answer to "did this cell get better" and the wrong one to "what
    landed here". A cell that took in twelve crops and lost twelve others
    reads as no change at all, and twenty-four moved crops are invisible.
    These two matrices net nothing: `moved_in` counts only the crops the
    candidate put in a cell the baseline had put them elsewhere than, and
    `moved_out` only the ones it took out. Cell by cell,
    `moved_in - moved_out` is exactly the difference matrix.

    Every crop that changed prediction is counted once in each -- as a
    departure from the baseline's cell and an arrival in the candidate's --
    so both matrices sum to `n_moved`. A crop that changed from one wrong
    class to another wrong one moves like any other: leaving it out would
    break the identity with the difference matrix, and it is a real change
    in the confusion the user is looking at.

    The true class is the same under both models, so a crop can only ever
    move along its own row. That is what makes the per-class flow matrix
    (see flow_matrix) the place the provenance actually lives.
    """
    if direction not in MOVEMENTS:
        raise ValueError(f"Not a movement direction: {direction!r}")
    pred_col = COL_PRED_B if direction == MATRIX_MOVED_IN else COL_PRED_A
    subset = joined[joined[COL_PRED_A] != joined[COL_PRED_B]]
    table = pd.crosstab(subset[COL_TRUE], subset[pred_col])
    return table.reindex(index=labels, columns=labels, fill_value=0).astype(int)


def rows_in_movement_cell(joined: pd.DataFrame, direction: str,
                          true_label, predicted_label) -> pd.DataFrame:
    """The crops behind one cell of a movement_confusion matrix, most
    confident first."""
    if direction not in MOVEMENTS:
        raise ValueError(f"Not a movement direction: {direction!r}")
    pred_col = COL_PRED_B if direction == MATRIX_MOVED_IN else COL_PRED_A
    subset = joined[(joined[COL_PRED_A] != joined[COL_PRED_B]) &
                    (joined[COL_TRUE] == true_label) &
                    (joined[pred_col] == predicted_label)]
    return _by_confidence(subset)


def rows_moved_through_cell(joined: pd.DataFrame, true_label, predicted_label) -> pd.DataFrame:
    """Everything that moved into or out of one cell -- what a cell of the
    difference matrix actually stands for.

    A cell of that matrix is a set of *changes*, not of crops: its number is
    arrivals minus departures, and a cell reading zero can still be two
    hundred crops swapped for two hundred others. Opening it has to open
    both sides.
    """
    in_cell_now = ((joined[COL_TRUE] == true_label) &
                   (joined[COL_PRED_B] == predicted_label) &
                   (joined[COL_PRED_A] != predicted_label))
    in_cell_before = ((joined[COL_TRUE] == true_label) &
                      (joined[COL_PRED_A] == predicted_label) &
                      (joined[COL_PRED_B] != predicted_label))
    return _by_confidence(joined[in_cell_now | in_cell_before])


def flow_matrix(joined: pd.DataFrame, labels: list, true_label) -> pd.DataFrame:
    """For one true class: what the baseline predicted (rows) against what
    the candidate predicts (columns).

    This is where "the twelve crops that arrived here -- where were they
    before?" is answered, and it answers the opposite direction in the same
    grid. Since the true class never changes between the two models, a crop
    only ever moves within its own row of the true-by-predicted matrix, and
    that row's whole story is this square:

    * column `p`, off the diagonal, is everything that moved into (true, p)
      -- one row per class it came from;
    * row `p`, off the diagonal, is everything that moved out of it, one
      column per class it went to;
    * column `true_label` is the crops the candidate fixed, row
      `true_label` the ones it broke;
    * the diagonal is what did not move, kept for scale.
    """
    subset = joined[joined[COL_TRUE] == true_label]
    table = pd.crosstab(subset[COL_PRED_A], subset[COL_PRED_B])
    return table.reindex(index=labels, columns=labels, fill_value=0).astype(int)


def flow_labels(flow: pd.DataFrame, true_label) -> list:
    """The classes worth drawing in a flow matrix: the ones some crop of
    this class actually involves, plus the true class itself.

    A hundred-class dataset gives a hundred-by-hundred flow matrix with a
    handful of non-zero cells, and a grid of zeros is not a panel. The true
    class stays in even when empty, since its row and column are what
    "broke" and "fixed" mean here and an absent one would read as missing
    rather than as zero.
    """
    used = {label for label in flow.index if flow.loc[label].any()}
    used |= {label for label in flow.columns if flow[label].any()}
    if true_label in flow.index:
        used.add(true_label)
    return [label for label in flow.index if label in used]


def cell_flows(joined: pd.DataFrame, labels: list,
               true_label, predicted_label) -> pd.DataFrame:
    """One row per class the picked cell traded crops with: how many came
    from it, and how many went to it.

    The difference with flow_matrix is what gets counted. That one takes
    every crop of the true class, which answers "what happened to this
    class" -- a wider question than the one asked by clicking a cell. This
    counts only the crops that touch the cell itself: the ones the
    candidate moved *into* (true, predicted), by the class the baseline had
    them in, and the ones it moved *out* of it, by the class it moved them
    to. Nothing else of the class appears.

    Both are slices of the same square -- the arrivals are that column of
    flow_matrix and the departures that row -- which is why this is the
    focused reading of it rather than a separate count.

    Classes that traded nothing with the cell are dropped: on a hundred-class
    dataset, a cell usually exchanges crops with two or three of them and the
    rest would be a page of zeros. What is left is ordered by traffic, so
    the class the cell mostly swaps with is the first line.
    """
    flow = flow_matrix(joined, labels, true_label)
    if predicted_label not in flow.index:
        return pd.DataFrame(columns=[COL_CAME_FROM, COL_WENT_TO], dtype=int)
    table = pd.DataFrame({
        COL_CAME_FROM: flow[predicted_label].drop(index=predicted_label),
        COL_WENT_TO: flow.loc[predicted_label].drop(index=predicted_label),
    }).astype(int)
    table = table[(table[COL_CAME_FROM] != 0) | (table[COL_WENT_TO] != 0)]
    traffic = table[COL_CAME_FROM] + table[COL_WENT_TO]
    # Negated and sorted ascending rather than sorted and reversed:
    # mergesort is the stable one, and reversing its result would undo that
    # stability on ties instead of leaving them in the matrix's own order.
    return table.iloc[(-traffic).to_numpy().argsort(kind="mergesort")]


def rows_in_flow_cell(joined: pd.DataFrame, true_label,
                      baseline_label, candidate_label) -> pd.DataFrame:
    """The crops behind one cell of a flow_matrix -- one true class, one
    baseline prediction, one candidate prediction."""
    subset = joined[(joined[COL_TRUE] == true_label) &
                    (joined[COL_PRED_A] == baseline_label) &
                    (joined[COL_PRED_B] == candidate_label)]
    return _by_confidence(subset)


def flow_improvement(true_label, baseline_label, candidate_label, value: int) -> int:
    """Signed "is this flow good news", for colouring a flow matrix.

    Three cases, and only two of them have a sign. A crop that arrived on
    the true class was fixed; one that left it was broken; one that went
    from a wrong class to another wrong class is a change with no verdict
    attached, and painting it green or red would claim something the data
    does not say. The diagonal never moved at all.
    """
    if baseline_label == candidate_label:
        return 0
    if candidate_label == true_label:
        return value
    if baseline_label == true_label:
        return -value
    return 0


def _by_confidence(subset: pd.DataFrame) -> pd.DataFrame:
    """Most confident under the candidate first -- see rows_for_outcome for
    why that is the useful order."""
    if len(subset):
        subset = subset.sort_values(COL_SCORE_B, ascending=False)
    return subset.reset_index(drop=True)


def cell_improvement(true_label: str, predicted_label: str, delta: int) -> int:
    """Signed "is this cell better under the candidate", as opposed to the
    raw count change.

    On the diagonal more is better; everywhere else more is worse. Coloring
    a difference matrix by the sign of the raw delta would therefore paint
    the single most important cell -- the one where the candidate got a
    class right more often -- in the same color as a new error.
    """
    return delta if true_label == predicted_label else -delta


def rows_for_outcome(joined: pd.DataFrame, outcome: str, sort_by_confidence: bool = True) -> pd.DataFrame:
    """The crops with one outcome, worst first.

    "Worst" is the candidate's own confidence: a regression the new model
    is 0.99 sure about is a different problem from one it was hesitating
    over, and it is the confident ones that point at something real
    (a label error, a leaked duplicate, a genuinely ambiguous crop).
    """
    subset = joined[joined[COL_OUTCOME] == outcome]
    if sort_by_confidence and len(subset):
        subset = subset.sort_values(COL_SCORE_B, ascending=False)
    return subset.reset_index(drop=True)


def per_class_change(joined: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Fixes and regressions broken down by true class -- which classes the
    candidate helped and which it hurt, since a flat net number can hide a
    model that traded one class for another.

    One grouped count over the rows rather than a filter per class, which
    at two thousand classes was most of the Compare tab's wait."""
    columns = ["label", "support", "fixes", "regressions", "net"]
    wanted = list(dict.fromkeys(labels))
    counts = pd.crosstab(joined[COL_TRUE], joined[COL_OUTCOME]) if len(joined) else pd.DataFrame()
    counts = counts.reindex(index=[l for l in wanted if l in counts.index])
    if counts.empty:
        return pd.DataFrame(columns=columns)
    none = pd.Series(0, index=counts.index)
    table = pd.DataFrame({
        "label": counts.index,
        "support": counts.sum(axis=1).to_numpy(),
        "fixes": counts.get(OUTCOME_FIX, none).to_numpy(dtype=int),
        "regressions": counts.get(OUTCOME_REGRESSION, none).to_numpy(dtype=int),
    })
    table["net"] = table["fixes"] - table["regressions"]
    return table[columns].reset_index(drop=True)


# ---- what a cleaning pass actually bought -------------------------------

@dataclass
class ReviewImpact:
    """The same experiment scored with and without the review overlay.

    "I cleaned 800 crops" is not a result; "+2.1 points of macro F1" is. The
    app can already compute either number -- the overlay switch swaps
    between them -- but showing one *instead of* the other means the user
    has to remember the previous figure to know whether the pass was worth
    it. This computes both in one pass so they can sit side by side.
    """
    experiment: str
    version_id: str
    before: ExperimentMetrics
    after: ExperimentMetrics
    n_excluded: int
    n_relabelled: int

    def delta(self, field_name: str) -> float:
        return getattr(self.after, field_name) - getattr(self.before, field_name)


# Metrics worth showing before/after. Keys are ExperimentMetrics attribute
# names, compared in code; only the labels are display text.
IMPACT_METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("mcc", "MCC"),
    ("macro_roc_auc", "Macro ROC-AUC"),
)


def review_impact(state, experiment_name: str) -> ReviewImpact:
    """Score `experiment_name` twice: on the raw labels, then through the
    review overlay.

    The overlay flag is flipped and restored here rather than asking the
    caller to do it, because getting that wrong leaves the whole app
    silently showing cleaned metrics -- the exact failure the overlay's own
    banner exists to prevent.
    """
    from lookin.core.review import STATUS_EXCLUDE, STATUS_RELABEL

    if state.dataset is None:
        raise ValueError("No dataset loaded.")
    store = getattr(state, "review", None)
    if store is None or not store.versions:
        raise ValueError("No review versions to measure -- decide on some crops first.")

    version_id = store.active_version_id or store.resume_version_id() or list(store.versions)[-1]
    counts = store.counts(version_id)
    cols = state.dataset.columns
    experiment = state.experiments[experiment_name]

    previous = state.use_reviewed_labels
    try:
        state.use_reviewed_labels = False
        before = compute_metrics(experiment_name, state.merged(experiment_name),
                                 cols.label_col, experiment.score_cols, experiment.class_labels)
        state.use_reviewed_labels = True
        after = compute_metrics(experiment_name, state.merged(experiment_name),
                                cols.label_col, experiment.score_cols, experiment.class_labels)
    finally:
        state.use_reviewed_labels = previous
        # The cached merge belongs to whichever overlay state ran last, and
        # the stamp already records that -- but leaving the cache holding a
        # frame built under the *other* setting is exactly the kind of thing
        # that shows stale numbers, so it is dropped explicitly.
        state._invalidate_merged_cache()

    return ReviewImpact(
        experiment=experiment_name, version_id=version_id, before=before, after=after,
        n_excluded=counts.get(STATUS_EXCLUDE, 0), n_relabelled=counts.get(STATUS_RELABEL, 0),
    )
