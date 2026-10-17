"""Candidate sources for review: the several independent ways of guessing
which crops are worth a human look, all reduced to one shape.

Embeddings are *one* method among several, not the price of entry. The
strategies here are deliberately ordered by what they need:

  by_margin / by_confident_error   the results CSV alone
  by_consensus                     two or more results CSVs
  from_embedding_flags             embeddings + the model's real prototypes
  noisy_subcenters                 the same, at subcenter granularity
  near_duplicates                  embeddings (+ a split column, for leakage)

Every one of them returns a CandidateSet, so the review queue, the results
table and the triage panel are written once and work with all of them.

A note on in-sample evidence: on a train pool, the model has memorized the
samples it is being asked to judge, so these rankings surface hard-but-
correct crops as readily as genuine annotation errors. That is not fixable
here -- it is why the UI marks a train pool and pushes by_consensus, which
is the one strategy whose signal doesn't come from a single model's memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Pairwise similarity is computed per class (see near_duplicates), which
# keeps the cost to the sum of squared class sizes rather than the square of
# the whole pool. A single class larger than this still blows that up, so it
# is capped and reported rather than left to look like a hang.
MAX_CLASS_SIZE_FOR_PAIRS = 12_000


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


@dataclass
class Candidate:
    """One crop worth looking at, plus why."""
    uuid: str
    label: str  # its current label
    suggestion: str = ""  # what the evidence suggests instead, "" if nothing specific
    score: float = 0.0  # higher = more suspicious; the queue is ordered by this
    detail: dict = field(default_factory=dict)  # extra per-strategy columns for the table


@dataclass
class CandidateSet:
    name: str  # what the user sees, e.g. "Confident errors -- baseline_cnn"
    source: str  # provenance string stored on every decision made from here
    columns: list  # names of the per-strategy detail columns, in table order
    candidates: list
    note: str = ""  # caveat shown above the table, e.g. an in-sample warning

    def uuids(self) -> list:
        return [c.uuid for c in self.candidates]

    def __len__(self) -> int:
        return len(self.candidates)


# Which side of a candidate the class filter looks at. Keys, compared in
# code -- the readable wording lives in the Cleaning view.
MATCH_LABEL = "label"  # the class the crop is annotated with
MATCH_SUGGESTION = "suggestion"  # the class the evidence proposes instead


@dataclass(frozen=True)
class ClassFilter:
    """Restrict a source to one class, on either side of a candidate.

    Every source applies it *before* its `limit`, never after: cutting the
    global top 300 and then filtering would leave whatever few crops of the
    class happened to rank there, not that class's worst crops.

    Compared as strings, since class labels come from the user's CSV
    untouched and may well be integers on one side and text on the other.
    """
    label: object
    match: str = MATCH_LABEL

    def keeps(self, candidate: Candidate) -> bool:
        value = candidate.suggestion if self.match == MATCH_SUGGESTION else candidate.label
        return str(value) == str(self.label)

    def describe(self) -> str:
        return f"suggested {self.label}" if self.match == MATCH_SUGGESTION else f"labelled {self.label}"


def _keeps(class_filter: Optional[ClassFilter], candidate: Candidate) -> bool:
    return class_filter is None or class_filter.keeps(candidate)


def apply_class_filter(candidate_set: CandidateSet, class_filter: Optional[ClassFilter]) -> CandidateSet:
    """For a source with no limit (a subcenter's members), filtering after
    the fact is the same thing as filtering inside it."""
    if class_filter is None:
        return candidate_set
    return CandidateSet(
        name=f"{candidate_set.name} -- {class_filter.describe()}",
        source=candidate_set.source, columns=candidate_set.columns,
        candidates=[c for c in candidate_set.candidates if class_filter.keeps(c)],
        note=candidate_set.note,
    )


def _class_score_lookup(experiment, class_labels: list) -> dict:
    """label -> index into the experiment's score columns."""
    return {label: i for i, label in enumerate(class_labels)}


def _predictions(experiment, uuid_col: str) -> tuple:
    """(uuids, predicted_label, predicted_score, score_matrix) for one
    experiment, straight from its own DataFrame.

    Deliberately not AppState.merged(): that cache holds a single
    experiment at a time, so walking several of them (by_consensus) through
    it would evict and re-merge hundreds of thousands of rows per
    experiment. Everything here only needs the scores keyed by uuid.
    """
    from lookin.core.data_manager import argmax_ignoring_nan

    scores = experiment.df[experiment.score_cols].to_numpy()
    labels = experiment.class_labels[: len(experiment.score_cols)]
    # A missing (NaN) score is never the prediction -- see merged().
    idx, best = argmax_ignoring_nan(scores)
    predicted = np.array([labels[i] if 0 <= i < len(labels) else "" for i in idx], dtype=object)
    return (
        experiment.df[experiment.uuid_col].astype(str).to_numpy(),
        predicted,
        best,
        scores,
    )


def by_margin(
    dataset_df: pd.DataFrame, experiment, uuid_col: str, label_col: str,
    min_margin: float = 0.0, limit: int = 500, confident_errors_only: bool = False,
    class_filter: Optional[ClassFilter] = None,
) -> CandidateSet:
    """Rank by how much the scores prefer some other class over the labeled
    one: margin = best other-class score - the labeled class's own score.

    This is the classic label-error heuristic, and it needs nothing but the
    results CSV. A positive margin is a disagreement with the annotation; a
    large positive margin on a confident model is the single best cheap
    signal there is. `confident_errors_only` keeps only rows the model got
    wrong, which is the narrower, higher-precision version of the same
    question.
    """
    uuids, predicted, predicted_score, scores = _predictions(experiment, uuid_col)
    true_by_uuid = dict(zip(dataset_df[uuid_col].astype(str), dataset_df[label_col]))
    class_index = _class_score_lookup(experiment, experiment.class_labels[: len(experiment.score_cols)])

    candidates = []
    for i, uid in enumerate(uuids):
        true_label = true_by_uuid.get(uid)
        if true_label is None:
            continue
        col = class_index.get(true_label)
        if col is None:
            # The labeled class has no score column in this experiment --
            # a class mapping mismatch, not a noisy sample; scoring it would
            # flag every crop of that class.
            continue
        own = float(scores[i, col])
        if np.isnan(own):
            # No score for the labeled class: no margin to measure (as NaN
            # it passed the `margin < min_margin` test and was listed).
            continue
        row = np.where(np.isnan(scores[i]), -np.inf, scores[i])
        row[col] = -np.inf
        best_other = int(row.argmax())
        margin = float(row[best_other]) - own
        if margin < min_margin:
            continue
        if confident_errors_only and predicted[i] == true_label:
            continue
        candidate = Candidate(
            uuid=uid, label=true_label, suggestion=str(predicted[i]), score=margin,
            detail={
                "own score": round(own, 4),
                "best other": str(experiment.class_labels[best_other]),
                "other score": round(float(scores[i, best_other]), 4),
                "margin": round(margin, 4),
            },
        )
        if _keeps(class_filter, candidate):
            candidates.append(candidate)
    candidates.sort(key=lambda c: -c.score)
    kind = "confident errors" if confident_errors_only else "score margin"
    return CandidateSet(
        name=f"{kind.capitalize()} -- {experiment.name}",
        source=f"margin:{experiment.name}",
        columns=["own score", "best other", "other score", "margin"],
        candidates=candidates[:limit],
    )


def all_errors(
    dataset_df: pd.DataFrame, experiment, uuid_col: str, label_col: str,
    class_filter: Optional[ClassFilter] = None,
) -> CandidateSet:
    """Every crop the model gets wrong, most confident first -- no cap.

    The exhaustive counterpart to by_margin: a ranking plus a limit answers
    "where are the worst few hundred", this answers "show me all of them",
    which is what finishing the review of a class takes. Ordered by the
    confidence of the wrong prediction, so the crops the model is surest
    are mislabelled come first and the coin flips come last.
    """
    uuids, predicted, predicted_score, scores = _predictions(experiment, uuid_col)
    true_by_uuid = dict(zip(dataset_df[uuid_col].astype(str), dataset_df[label_col]))
    class_index = _class_score_lookup(experiment, experiment.class_labels[: len(experiment.score_cols)])

    candidates = []
    for i, uid in enumerate(uuids):
        true_label = true_by_uuid.get(uid)
        if true_label is None or predicted[i] == true_label or np.isnan(predicted_score[i]):
            # (No score at all: no prediction to have got wrong.)
            continue
        col = class_index.get(true_label)
        if col is None:
            # Same guard as by_margin: a class with no score column would
            # count every one of its crops as an error.
            continue
        candidate = Candidate(
            uuid=uid, label=true_label, suggestion=str(predicted[i]), score=float(predicted_score[i]),
            detail={
                "predicted": str(predicted[i]),
                "confidence": round(float(predicted_score[i]), 4),
                "own score": round(float(scores[i, col]), 4),
            },
        )
        if _keeps(class_filter, candidate):
            candidates.append(candidate)
    candidates.sort(key=lambda c: -c.score)
    return CandidateSet(
        name=f"All errors -- {experiment.name}",
        source=f"all_errors:{experiment.name}",
        columns=["predicted", "confidence", "own score"],
        candidates=candidates,
    )


def by_consensus(
    dataset_df: pd.DataFrame, experiments: list, uuid_col: str, label_col: str,
    min_agreeing: int = 2, limit: int = 500, class_filter: Optional[ClassFilter] = None,
) -> CandidateSet:
    """Crops that several independently-trained models get wrong *in the
    same way*.

    One model disagreeing with an annotation is a model error as often as
    an annotation error. Four models independently predicting the same
    other class is a much stronger claim about the annotation -- and unlike
    every other strategy here, its signal doesn't come from a single
    model's memory of the sample, which is what makes it the one to reach
    for on a train pool.

    Wide disagreement (models wrong in *different* directions) is the
    opposite finding: a genuinely ambiguous crop, which is what `uncertain`
    is for. It is reported in the detail column rather than ranked up.
    """
    true_by_uuid = dict(zip(dataset_df[uuid_col].astype(str), dataset_df[label_col]))
    votes: dict = {}
    for experiment in experiments:
        uuids, predicted, predicted_score, _ = _predictions(experiment, uuid_col)
        for uid, pred, conf in zip(uuids, predicted, predicted_score):
            if pred == "":
                continue  # no score at all: no vote
            votes.setdefault(uid, []).append((experiment.name, str(pred), float(conf)))

    candidates = []
    for uid, entries in votes.items():
        true_label = true_by_uuid.get(uid)
        # A crop with no label (NaN) disagreed with every model and was
        # listed as a consensus error with "nan" for its label.
        if true_label is None or pd.isna(true_label) or len(entries) < min_agreeing:
            continue
        wrong = [e for e in entries if e[1] != true_label]
        if len(wrong) < min_agreeing:
            continue
        tally: dict = {}
        for _, pred, conf in wrong:
            slot = tally.setdefault(pred, [0, 0.0])
            slot[0] += 1
            slot[1] += conf
        best_label, (n_agree, conf_sum) = max(tally.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        if n_agree < min_agreeing:
            continue
        candidate = Candidate(
            uuid=uid, label=true_label, suggestion=best_label,
            # Agreement first, mean confidence as the tie-break: five models
            # agreeing weakly beats two agreeing loudly.
            score=n_agree + min(conf_sum / max(n_agree, 1), 0.999) / 1000,
            detail={
                "models wrong": f"{len(wrong)}/{len(entries)}",
                "agreeing on": best_label,
                "agreement": n_agree,
                "mean confidence": round(conf_sum / max(n_agree, 1), 4),
                "unanimous": "yes" if n_agree == len(entries) else "no",
            },
        )
        if _keeps(class_filter, candidate):
            candidates.append(candidate)
    candidates.sort(key=lambda c: -c.score)
    return CandidateSet(
        name=f"Consensus errors -- {len(experiments)} experiments",
        source=f"consensus:{len(experiments)}",
        columns=["models wrong", "agreeing on", "agreement", "mean confidence", "unanimous"],
        candidates=candidates[:limit],
    )


def from_embedding_flags(flags, experiment_name: str, limit: int = 500,
                         class_filter: Optional[ClassFilter] = None) -> CandidateSet:
    """Wrap embedding_analysis.flag_noisy_samples' output in the common
    shape, ranked by how much closer the sample sits to another class's
    prototype than to its own."""
    gap = np.nan_to_num(flags.best_other_similarity - flags.own_similarity, nan=-np.inf)
    order = np.argsort(-gap)
    candidates = []
    for i in order:
        if not flags.flagged[i]:
            continue
        candidate = Candidate(
            uuid=str(flags.uuid[i]), label=str(flags.label[i]),
            suggestion=str(flags.best_other_label[i] or ""), score=float(gap[i]),
            detail={
                "nearest other class": str(flags.best_other_label[i] or ""),
                "own sim.": round(float(flags.own_similarity[i]), 4),
                "other sim.": round(float(flags.best_other_similarity[i]), 4),
                "reasons": flags.reasons[i],
            },
        )
        if not _keeps(class_filter, candidate):
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return CandidateSet(
        name=f"Prototype flags -- {experiment_name}",
        source=f"embedding_flags:{experiment_name}",
        columns=["nearest other class", "own sim.", "other sim.", "reasons"],
        candidates=candidates,
    )


@dataclass
class SubcenterRow:
    class_label: str
    k_idx: int
    n_assigned: int
    purity: float
    compactness: float
    separation: float
    noise_score: float
    member_uuids: list = field(default_factory=list)


def noisy_subcenters(
    embeddings: np.ndarray, labels: np.ndarray, uuids: np.ndarray, protos,
    assignment: Optional[tuple] = None, min_assigned: int = 1,
) -> list:
    """Rank each (class, subcenter) prototype by how much junk it looks to
    have collected, and carry its members along so the ranking is directly
    actionable.

    The score blends the three things that make a subcenter suspect:

      low purity        it is absorbing other classes' samples
      low compactness   what it holds isn't tight around it
      high separation   it sits on top of another class's prototype

    A near-empty subcenter is reported too (it is usually a degenerate
    center that captured whatever didn't fit anywhere else) but can't be
    scored on purity, so it is ranked by its emptiness instead.
    """
    from lookin.core import embedding_analysis as ea

    protos.require_model_source("subcenter noise ranking")
    if assignment is None:
        best_class, best_k, _ = ea.assign_nearest_prototype(embeddings, protos)
        assignment = (best_class, best_k)
    stats = ea.per_prototype_stats(embeddings, labels, protos, assignment=assignment)
    best_class, best_k = assignment
    index_of = protos.label_index()

    rows = []
    for s in stats:
        if s.n_assigned < min_assigned:
            continue
        purity = 0.0 if np.isnan(s.purity) else float(s.purity)
        compactness = 0.0 if np.isnan(s.compactness) else float(s.compactness)
        separation = 0.0 if np.isnan(s.separation) else float(s.separation)
        noise = (1.0 - purity) + max(0.0, separation - compactness)
        mask = (best_class == index_of[s.class_label]) & (best_k == s.k_idx)
        rows.append(SubcenterRow(
            class_label=s.class_label, k_idx=s.k_idx, n_assigned=s.n_assigned,
            purity=purity, compactness=compactness, separation=separation,
            noise_score=noise, member_uuids=[str(u) for u in uuids[mask]],
        ))
    rows.sort(key=lambda r: -r.noise_score)
    return rows


def subcenter_matches(row: SubcenterRow, labels_by_uuid: dict, class_filter: ClassFilter) -> int:
    """How many of a subcenter's members the class filter would keep once
    it is opened -- the same rule as subcenter_candidates, counted without
    building a Candidate per member, so the ranking table can be filtered
    on every change of class without walking the whole pool into objects."""
    target = str(class_filter.label)
    if class_filter.match == MATCH_SUGGESTION:
        # A member's suggestion is its subcenter's class, and only when it
        # is labelled something else.
        if str(row.class_label) != target:
            return 0
        return sum(1 for uid in row.member_uuids if str(labels_by_uuid.get(uid, "")) != target)
    return sum(1 for uid in row.member_uuids if str(labels_by_uuid.get(uid, "")) == target)


def subcenter_candidates(row: SubcenterRow, labels_by_uuid: dict, majority_label: str = "") -> CandidateSet:
    """One subcenter's members as a reviewable set. The suggestion is the
    subcenter's own class for a sample that doesn't share it -- which is
    exactly the "this subcenter absorbed another class" case the ranking
    points at."""
    candidates = []
    for uid in row.member_uuids:
        label = str(labels_by_uuid.get(uid, ""))
        suggestion = majority_label or row.class_label
        candidates.append(Candidate(
            uuid=uid, label=label,
            suggestion="" if label == suggestion else suggestion,
            score=0.0 if label == row.class_label else 1.0,
            detail={"assigned to": f"{row.class_label}#{row.k_idx}", "own class": label},
        ))
    candidates.sort(key=lambda c: -c.score)
    return CandidateSet(
        name=f"Subcenter {row.class_label}#{row.k_idx} -- {row.n_assigned:,} samples",
        source=f"subcenter:{row.class_label}#{row.k_idx}",
        columns=["assigned to", "own class"],
        candidates=candidates,
    )


@dataclass
class DuplicatePair:
    uuid_a: str
    uuid_b: str
    similarity: float
    label: str
    split_a: str = ""
    split_b: str = ""

    def cross_split(self) -> bool:
        return bool(self.split_a) and bool(self.split_b) and self.split_a != self.split_b


def near_duplicates(
    embeddings: np.ndarray, uuids: np.ndarray, labels: np.ndarray,
    splits: Optional[np.ndarray] = None, threshold: float = 0.98,
    limit: int = 300, cross_split_only: bool = False, only_label=None,
) -> tuple:
    """Pairs of crops whose embeddings are near-identical, blocked by class.

    Blocking by class is what makes this affordable (the cost is the sum of
    squared class sizes, not the square of the pool) and costs nothing that
    matters: a duplicate carrying two different labels is a labeling
    conflict the margin strategies already surface.

    With a split array, a pair straddling train and test is train/test
    leakage -- the finding that quietly invalidates an entire evaluation,
    and the reason this is worth running even on a clean-looking dataset.

    `only_label` restricts the search to one class -- skipped before the
    pairwise product rather than filtered after it, which is also the
    cheapest way to run this on a class too big to fit alongside the rest.

    Returns (pairs, skipped_classes).
    """
    pairs, skipped = [], []
    for label in np.unique(labels):
        if only_label is not None and str(label) != str(only_label):
            continue
        mask = labels == label
        n = int(mask.sum())
        if n < 2:
            continue
        if n > MAX_CLASS_SIZE_FOR_PAIRS:
            skipped.append((str(label), n))
            continue
        embs = embeddings[mask]
        ids = uuids[mask]
        class_splits = splits[mask] if splits is not None else None
        sim = embs @ embs.T
        np.fill_diagonal(sim, -np.inf)
        rows, cols = np.where(np.triu(sim >= threshold, k=1))
        for i, j in zip(rows, cols):
            split_a = str(class_splits[i]) if class_splits is not None else ""
            split_b = str(class_splits[j]) if class_splits is not None else ""
            if cross_split_only and split_a == split_b:
                continue
            pairs.append(DuplicatePair(
                uuid_a=str(ids[i]), uuid_b=str(ids[j]), similarity=float(sim[i, j]),
                label=str(label), split_a=split_a, split_b=split_b,
            ))
    # Leakage first, then by similarity: a cross-split pair is a different
    # (and worse) finding than a within-split one, not just a stronger one.
    pairs.sort(key=lambda p: (not p.cross_split(), -p.similarity))
    if skipped:
        _log(f"near-duplicates: skipped {len(skipped)} class(es) over "
             f"{MAX_CLASS_SIZE_FOR_PAIRS:,} samples: {', '.join(f'{c} ({n:,})' for c, n in skipped[:5])}")
    return pairs[:limit], skipped


def duplicate_candidates(pairs: list, keep: str = "b") -> CandidateSet:
    """The 'second' crop of each duplicate pair, as a reviewable set --
    exclude one of a pair and the duplication is gone, so the queue offers
    one side rather than both."""
    seen, candidates = set(), []
    for pair in pairs:
        uid = pair.uuid_b if keep == "b" else pair.uuid_a
        if uid in seen:
            continue
        seen.add(uid)
        candidates.append(Candidate(
            uuid=uid, label=pair.label, suggestion="", score=pair.similarity,
            detail={
                "duplicate of": pair.uuid_a if keep == "b" else pair.uuid_b,
                "similarity": round(pair.similarity, 4),
                "splits": f"{pair.split_a} / {pair.split_b}" if pair.split_a or pair.split_b else "",
                "leakage": "yes" if pair.cross_split() else "no",
            },
        ))
    return CandidateSet(
        name=f"Near-duplicates -- {len(candidates):,} crops",
        source="near_duplicates",
        columns=["duplicate of", "similarity", "splits", "leakage"],
        candidates=candidates,
    )


# ---- cleaning impact ----------------------------------------------------

@dataclass
class VersionImpact:
    version_id: str
    reviewed: int
    excluded: int
    relabelled: int
    uncertain: int
    accuracy_before: float
    accuracy_after: float
    pool_before: int
    pool_after: int
    strategy_precision: dict = field(default_factory=dict)  # source -> share of its flags that became a real correction

    @property
    def delta(self) -> float:
        return self.accuracy_after - self.accuracy_before


def version_impact(
    store, version_id: str, dataset_df: pd.DataFrame, experiment,
    uuid_col: str, label_col: str,
) -> VersionImpact:
    """What one review version does to an experiment's accuracy.

    Answers the question that decides whether another cleaning pass is
    worth the hours: how much of the error was annotation noise rather than
    model error. It also scores the *strategies*: what share of the crops
    each source flagged turned into an actual correction (a relabel or an
    exclusion) rather than a confirmation, which is what tells you to keep
    running one and drop another.
    """
    from lookin.core.review import STATUS_EXCLUDE, STATUS_RELABEL, STATUS_UNCERTAIN, STATUS_OK

    uuids, predicted, _, _ = _predictions(experiment, uuid_col)
    predicted_by_uuid = dict(zip(uuids, predicted))
    resolved = store.effective(version_id)

    before_hits = before_total = after_hits = after_total = 0
    for uid, true_label in zip(dataset_df[uuid_col].astype(str), dataset_df[label_col]):
        prediction = predicted_by_uuid.get(uid)
        # No label, or no score: neither right nor wrong (as merged() has it).
        if prediction is None or prediction == "" or pd.isna(true_label):
            continue
        before_total += 1
        before_hits += int(prediction == true_label)
        decision = resolved.get(uid)
        if decision is not None and decision.status == STATUS_EXCLUDE:
            continue
        after_total += 1
        after_hits += int(prediction == (decision.effective_label(true_label) if decision else true_label))

    per_source: dict = {}
    counts = {STATUS_OK: 0, STATUS_RELABEL: 0, STATUS_EXCLUDE: 0, STATUS_UNCERTAIN: 0}
    for decision in resolved.values():
        counts[decision.status] = counts.get(decision.status, 0) + 1
        source = (decision.source or "unknown").split(":")[0]
        slot = per_source.setdefault(source, [0, 0])
        slot[0] += 1
        slot[1] += int(decision.status in (STATUS_RELABEL, STATUS_EXCLUDE))
    strategy_precision = {
        source: (found / total if total else 0.0) for source, (total, found) in sorted(per_source.items())
    }

    return VersionImpact(
        version_id=version_id,
        reviewed=len(resolved),
        excluded=counts.get(STATUS_EXCLUDE, 0),
        relabelled=counts.get(STATUS_RELABEL, 0),
        uncertain=counts.get(STATUS_UNCERTAIN, 0),
        accuracy_before=before_hits / before_total if before_total else float("nan"),
        accuracy_after=after_hits / after_total if after_total else float("nan"),
        pool_before=before_total,
        pool_after=after_total,
        strategy_precision=strategy_precision,
    )


# ---- taxonomy -----------------------------------------------------------

@dataclass
class TaxonomyHint:
    kind: str  # "merge" or "split"
    a: str
    b: str = ""
    value: float = 0.0
    detail: str = ""


def taxonomy_hints(protos, per_class, top_n: int = 8, merge_threshold: float = 0.8,
                   split_threshold: float = 0.5) -> list:
    """Where the finding is "this taxonomy is wrong", not "these labels are".

    Two classes whose prototypes sit almost on top of each other are being
    asked to be distinguished by a model that cannot see a difference --
    possibly because there isn't one. A class whose own subcenters are far
    apart is the mirror image: one label covering two visually distinct
    populations, which a split would separate.
    """
    from lookin.core import embedding_analysis as ea

    hints = []
    for pair in ea.prototype_confusion_pairs(protos, top_n=top_n):
        if pair["sim"] >= merge_threshold:
            hints.append(TaxonomyHint(
                kind="merge", a=pair["a"], b=pair["b"], value=pair["sim"],
                detail=f"prototypes {pair['sim']:.3f} similar -- the model sees one class, not two",
            ))

    spreads = []
    for label, index in protos.label_index().items():
        vectors = protos.vectors[index]
        valid = ~np.isnan(vectors).any(axis=1)
        if valid.sum() < 2:
            continue
        sub = vectors[valid]
        sim = sub @ sub.T
        np.fill_diagonal(sim, np.nan)
        spreads.append((label, float(np.nanmin(sim))))

    below = [(label, worst) for label, worst in spreads if worst <= split_threshold]
    if spreads and len(below) > len(spreads) / 2:
        # Every class having distant subcenters is not a statement about any
        # one class -- it means K is higher than this data supports, and the
        # extra centers are splitting noise. Reporting each class separately
        # would bury that single real finding under one row per class, so
        # only the sharpest few are listed, with the actual conclusion said
        # out loud.
        n_below = len(below)
        below = sorted(below, key=lambda item: item[1])[:3]
        note = (f" ({n_below} of {len(spreads)} classes look like this -- more likely K is too high for "
                f"this dataset than that every class needs splitting)")
    else:
        note = ""
    for label, worst in below:
        hints.append(TaxonomyHint(
            kind="split", a=label, value=worst,
            detail=f"two subcenters only {worst:.3f} similar -- one label over two populations{note}",
        ))
    hints.sort(key=lambda h: (h.kind != "merge", -abs(h.value)))
    return hints
