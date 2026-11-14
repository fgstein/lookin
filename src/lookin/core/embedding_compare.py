"""One model's embedding space, seen through several datasets.

The Embeddings tab draws one experiment of one workspace: one dataset, one
model. The question this module answers is the other one -- "the same
model, run on two bases: where do they sit relative to each other?" -- and
a workspace cannot ask it, because a workspace is one dataset. So a
comparison is a list of *members*, each an experiment of some workspace of
the library (MemberRef), possibly of different workspaces, drawn in one
space.

Only the same model may be compared
-----------------------------------
Two embedding spaces from two models are two coordinate systems: a point
at the same place in both means nothing, and a projection fitted on their
union draws a difference that is only the difference between the models.
Nothing in the vectors themselves says which model made them (the same
dimension proves nothing), so the model's own prototypes -- the head's
class vectors, imported with each experiment -- are the fingerprint:
same model, same prototypes, to rounding. check_prototypes() compares
them and the comparison is refused when they differ or are missing. A
kmeans estimate is never accepted for this: it is fit on each base's own
crops, so it differs between bases by construction.

Two ways to project
-------------------
* PROJ_JOINT: one projection fitted on the union, the same number of
  points drawn from each member, so no base dominates the layout. Every
  base is treated alike, but adding one moves the map.
* PROJ_REFERENCE: fitted on the reference member alone; the others are
  placed onto that fit (ProjectionFit.transform). The map is the
  reference's and holds still: "where does base B land in base A's
  space?". t-SNE has no transform, so it is joint-only.

Measures
--------
Each member is measured against the reference (pair_stats):
* domain separability -- how well a linear classifier tells the two bases
  apart from the embeddings alone (cross-validated ROC AUC; 0.5 = cannot);
* coverage -- the share of the other base's crops that have a neighbour in
  the reference as close as the reference's crops usually have each other;
* label agreement -- how often the labels of a crop's nearest reference
  crops agree with its own label, beside the same measure inside the
  reference (two bases annotated by different conventions show here);
* per class: how far the class's centroid moved, how close its crops sit
  to the model's prototype in each base, accuracy in each base, coverage
  and agreement.
They run on samples (STATS_MAX_POINTS, KNN_POOL_MAX) so a comparison of
two half-million-crop bases still answers in seconds; the manual's pages
quote those constants. Searching neighbours of one crop in another member
(neighbours_across) is exact, over the whole member.

Keys stored in a workspace manifest, never reworded: PROJ_JOINT,
PROJ_REFERENCE, and the "members" / "reference" / "workspace" /
"experiment" fields of Workspace.embedding_comparison.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lookin.core.embedding_analysis import (
    MODEL_SOURCE, OutOfSampleUnsupported, PrototypeSet, fit_projection, nearest_to_vector,
)

PROJ_JOINT = "joint"
PROJ_REFERENCE = "reference"
PROJECTION_MODES = (PROJ_JOINT, PROJ_REFERENCE)
# Projection methods: the Embeddings tab's keys.
METHODS = ("pca", "umap", "umap_circle", "tsne")
TRANSFORM_METHODS = ("pca", "umap", "umap_circle")

# Two prototypes are "the same" when their cosine similarity is at least
# 1 - this: float32 rounding and a re-export leave ~1e-6, a different
# checkpoint leaves orders of magnitude more.
PROTOTYPE_TOLERANCE = 1e-4

# Points drawn per member on the map, by default and at most.
DEFAULT_POINTS_PER_MEMBER = 1500
MAX_POINTS_PER_MEMBER = 20_000
# Prototypes drawn on the map, at most (a thousand-class head is a cloud of
# stars, not landmarks).
MAX_MAP_PROTOTYPES = 300
# The samples the measures run on: crops of each side measured, and crops
# of the reference searched for their neighbours.
STATS_MAX_POINTS = 4000
KNN_POOL_MAX = 20_000
# Neighbours a crop's label is compared with (label agreement).
AGREEMENT_K = 10
# The reference's own nearest-neighbour similarities below this percentile
# are its outliers; the coverage threshold is set at it.
COVERAGE_PERCENTILE = 5.0
# The domain classifier is cross-validated over this many folds.
DOMAIN_FOLDS = 5
# Crops listed as "outside the reference" and "label disagreements".
TOP_LISTED = 200

# Members whose data is kept in memory (other workspaces read from disk).
MAX_CACHED_MEMBERS = 4


# ---- references --------------------------------------------------------------

@dataclass(frozen=True)
class MemberRef:
    workspace_id: str
    experiment: str

    def key(self) -> str:
        return f"{self.workspace_id}/{self.experiment}"

    def to_json(self) -> dict:
        return {"workspace": self.workspace_id, "experiment": self.experiment}


def refs_from_json(raw) -> list:
    out = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("workspace") and item.get("experiment"):
            ref = MemberRef(str(item["workspace"]), str(item["experiment"]))
            if ref not in out:
                out.append(ref)
    return out


def comparison_to_json(refs: list, reference: int) -> dict:
    if not refs:
        return {}
    return {"members": [r.to_json() for r in refs], "reference": max(0, min(int(reference), len(refs) - 1))}


def comparison_from_json(raw) -> tuple:
    """(refs, reference index) from Workspace.embedding_comparison."""
    raw = raw if isinstance(raw, dict) else {}
    refs = refs_from_json(raw.get("members"))
    try:
        reference = int(raw.get("reference", 0))
    except (TypeError, ValueError):
        reference = 0
    return refs, (reference if 0 <= reference < len(refs) else 0)


# ---- members -----------------------------------------------------------------

@dataclass
class Member:
    """One experiment's crops and embeddings, as far as a comparison needs
    them. Arrays are row-aligned: row i of `embeddings` is crop i."""
    ref: MemberRef
    workspace_name: str
    embeddings: np.ndarray            # (N, D) L2-normalized
    uuids: np.ndarray                 # (N,) str
    labels: np.ndarray                # (N,) str, "" for a crop with no label
    predicted: Optional[np.ndarray]   # (N,) str
    correct: Optional[np.ndarray]     # (N,) float: 1 right, 0 wrong, NaN unlabelled
    crops: np.ndarray                 # (N,) the crop column as stored
    image_root: Path                  # where relative crops resolve from
    prototypes: Optional[PrototypeSet] = None
    metadata: dict = field(default_factory=dict)   # column -> (N,) array, for the records
    warnings: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.workspace_name} / {self.ref.experiment}"

    def __len__(self) -> int:
        return len(self.uuids)

    def crop_path(self, i: int) -> Path:
        p = Path(str(self.crops[i]))
        return p if p.is_absolute() else self.image_root / p

    def record(self, i: int) -> dict:
        """Crop i as the carousel and the map read it: its metadata, then
        the keys every member shares."""
        rec = {}
        for column, values in self.metadata.items():
            value = values[i]
            rec[column] = value.item() if isinstance(value, np.generic) else value
        rec.update({
            "experiment": self.name, "uuid": str(self.uuids[i]), "label": self.labels[i] or None,
            "path": str(self.crop_path(i)),
        })
        if self.predicted is not None:
            rec["predicted_label"] = str(self.predicted[i])
        if self.correct is not None and np.isfinite(self.correct[i]):
            rec["correct"] = bool(self.correct[i])
        return rec

    def index_of(self, uuid: str) -> int:
        found = np.flatnonzero(self.uuids == str(uuid))
        if not len(found):
            raise ValueError(f"No crop {uuid!r} with an embedding in {self.name!r}.")
        return int(found[0])


def member_from_state(ref: MemberRef, workspace_name: str, state) -> Member:
    """A Member from a loaded AppState holding the experiment, embeddings
    attached. Raises ValueError when it cannot be one."""
    from lookin.core.embedding_context import build_analysis_context

    if state.dataset is None:
        raise ValueError(f"{workspace_name!r} has no dataset loaded.")
    if ref.experiment not in state.experiments:
        raise ValueError(f"{workspace_name!r} has no experiment {ref.experiment!r}.")
    exp = state.experiments[ref.experiment]
    if not exp.has_embeddings():
        raise ValueError(f"Experiment {ref.experiment!r} of {workspace_name!r} has no embeddings attached.")
    ctx = build_analysis_context(state, ref.experiment)
    if ctx is None or not len(ctx["embeddings"]):
        raise ValueError(f"No crop of experiment {ref.experiment!r} of {workspace_name!r} has an embedding.")
    merged = ctx["merged"]
    cols = state.dataset.columns
    raw_labels = merged[cols.label_col] if cols.label_col in merged.columns else pd.Series([None] * len(merged))
    has_label = raw_labels.notna() & (raw_labels.astype(str) != "")
    labels = np.where(has_label.to_numpy(), raw_labels.astype(str).to_numpy(), "").astype(object)
    predicted = merged["predicted_label"].astype(str).to_numpy() if "predicted_label" in merged.columns else None
    correct = None
    if "correct" in merged.columns:
        correct = merged["correct"].astype(float).to_numpy().copy()
        correct[~has_label.to_numpy()] = np.nan
    skip = set(exp.score_cols) | {cols.uuid_col, cols.crop_col, cols.label_col, "predicted_label", "correct"}
    metadata = {c: merged[c].to_numpy() for c in merged.columns if c not in skip}
    dataset = state.dataset
    member = Member(
        ref=ref, workspace_name=workspace_name, embeddings=ctx["embeddings"], uuids=ctx["uuids"].astype(str),
        labels=labels, predicted=predicted, correct=correct, crops=merged[cols.crop_col].astype(str).to_numpy(),
        image_root=Path(dataset.image_root or dataset.base_dir), prototypes=exp.model_prototypes, metadata=metadata)
    total = len(state.merged(ref.experiment))
    if len(member) < total:
        member.warnings.append(f"{member.name}: {len(member):,} of {total:,} crops have an embedding; "
                               f"only those are compared.")
    return member


_cache_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()   # ref -> (stamp, Member)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _mtime(path) -> float:
    try:
        return os.stat(path).st_mtime if path else 0.0
    except OSError:
        return 0.0


def _stamp(workspace) -> tuple:
    from lookin.core import library as lib

    mount = workspace.mount
    files = [Path(workspace.path) / lib.MANIFEST_NAME if workspace.path else "", mount.dataset_path]
    for e in mount.experiments:
        files += [e.path, e.embeddings_path, e.prototypes_path]
    return tuple(_mtime(f) for f in files)


def load_member(ref: MemberRef) -> Member:
    """A member read from its workspace in the library -- the dataset and
    that one experiment, with its embeddings and prototypes. The last few
    are kept until their files change."""
    from lookin.core import workspace_state as ws

    workspace = ws.find_workspace(ref.workspace_id)
    stamp = _stamp(workspace)
    with _cache_lock:
        hit = _cache.get(ref)
        if hit is not None and hit[0] == stamp:
            _cache.move_to_end(ref)
            return hit[1]
    if not any(e.name == ref.experiment for e in workspace.mount.experiments):
        raise ValueError(f"Workspace {workspace.label()!r} has no experiment {ref.experiment!r}.")
    loaded = ws.load_workspace_state(workspace, only=[ref.experiment])
    if ref.experiment not in loaded.state.experiments:
        reasons = "; ".join(loaded.warnings) or "it could not be loaded"
        raise ValueError(f"Experiment {ref.experiment!r} of {workspace.label()!r}: {reasons}")
    ws.attach_sidecars(workspace, loaded.state, ref.experiment, prototypes=True)
    member = member_from_state(ref, workspace.label(), loaded.state)
    member.warnings = [f"{member.name}: {w}" for w in loaded.warnings] + member.warnings
    with _cache_lock:
        _cache[ref] = (stamp, member)
        while len(_cache) > MAX_CACHED_MEMBERS:
            _cache.popitem(last=False)
    return member


def load_members(refs: list, local=None) -> tuple:
    """(members, problems): every ref loaded, or a sentence saying why it
    was not. `local` is (workspace_id, workspace_name, state) -- a session's
    own workspace, read from what it has loaded rather than from disk (it
    may hold changes not saved yet, and is already in memory)."""
    members, problems = [], []
    for ref in refs:
        try:
            if local is not None and ref.workspace_id == local[0] and ref.experiment in local[2].experiments \
                    and local[2].experiments[ref.experiment].has_embeddings():
                members.append(member_from_state(ref, local[1], local[2]))
            else:
                members.append(load_member(ref))
        except Exception as ex:  # noqa: BLE001 -- one member never stops the others
            problems.append(f"{ref.experiment!r} of {ref.workspace_id}: {ex}")
    return members, problems


def comparable_experiments(root=None) -> list:
    """Every experiment of the library that brings embeddings: what a
    comparison can be made of. Dicts with workspace_id, workspace,
    experiment, embeddings and prototypes (whether those files are on this
    machine)."""
    from lookin.core import library as lib

    out = []
    for workspace in lib.list_workspaces(root):
        for e in workspace.mount.experiments:
            if not e.embeddings_path:
                continue
            out.append({"workspace_id": workspace.id, "workspace": workspace.label(), "experiment": e.name,
                        "embeddings": Path(e.embeddings_path).is_file(),
                        "prototypes": bool(e.prototypes_path) and Path(e.prototypes_path).is_file(),
                        "prototypes_path": e.prototypes_path})
    return out


# ---- same model? ---------------------------------------------------------------

@dataclass
class PrototypeCheck:
    ok: bool
    reason: str
    min_cosine: Optional[float] = None
    n_compared: int = 0


def check_prototypes(a: Optional[PrototypeSet], b: Optional[PrototypeSet], name_a: str = "the reference",
                     name_b: str = "this member", tolerance: float = PROTOTYPE_TOLERANCE) -> PrototypeCheck:
    """Whether two experiments' imported prototypes are the same, which is
    what says their embeddings come from the same model. Classes are
    matched by label (the files may list them in another order); a class
    with no prototype on one side (a NaN row) is skipped."""
    for protos, name in ((a, name_a), (b, name_b)):
        if protos is None:
            return PrototypeCheck(False, f"{name} has no model prototypes: they are what shows that two "
                                         f"experiments come from the same model.")
        if protos.source != MODEL_SOURCE:
            return PrototypeCheck(False, f"{name}'s prototypes are estimated, not the model's own.")
    dim_a, dim_b = a.vectors.shape[-1], b.vectors.shape[-1]
    if dim_a != dim_b:
        return PrototypeCheck(False, f"The embeddings have {dim_a} dimensions in {name_a} and {dim_b} in {name_b}: "
                                     f"two different models.")
    if a.k != b.k:
        return PrototypeCheck(False, f"{a.k} prototype(s) per class in {name_a}, {b.k} in {name_b}: two different models.")
    labels_a, labels_b = [str(x) for x in a.class_labels], [str(x) for x in b.class_labels]
    only_a, only_b = sorted(set(labels_a) - set(labels_b)), sorted(set(labels_b) - set(labels_a))
    if only_a or only_b:
        def few(xs):
            return ", ".join(xs[:5]) + (f" and {len(xs) - 5} more" if len(xs) > 5 else "")
        parts = ([f"only in {name_a}: {few(only_a)}"] if only_a else []) + \
                ([f"only in {name_b}: {few(only_b)}"] if only_b else [])
        return PrototypeCheck(False, "The two heads do not have the same classes (" + "; ".join(parts) + ").")
    index_b = {label: i for i, label in enumerate(labels_b)}
    vb = b.vectors[[index_b[label] for label in labels_a]]
    va = a.vectors
    cos = np.sum(va * vb, axis=-1)                 # (C, K)
    finite = np.isfinite(cos)
    if not finite.any():
        return PrototypeCheck(False, "No class has a prototype on both sides, so nothing could be compared.")
    worst = float(np.min(cos[finite]))
    n = int(finite.sum())
    if worst < 1.0 - tolerance:
        c, k = np.unravel_index(np.argmin(np.where(finite, cos, np.inf)), cos.shape)
        return PrototypeCheck(False, f"The prototypes differ (cosine {worst:.4f} for class {labels_a[c]!r}"
                                     f"{f', sub-center {k}' if a.k > 1 else ''}): the embeddings come from "
                                     f"two different models, or two checkpoints of one.", worst, n)
    return PrototypeCheck(True, f"Same prototypes ({n:,} compared, lowest cosine {worst:.6f}).", worst, n)


def check_members(members: list, reference: int) -> list:
    """One PrototypeCheck per member against the reference (the reference
    itself is checked against nothing and is ok)."""
    ref = members[reference]
    out = []
    for i, member in enumerate(members):
        if i == reference:
            ok = ref.prototypes is not None
            out.append(PrototypeCheck(ok, "Reference." if ok else
                                      f"{ref.name} has no model prototypes: they are what shows that two "
                                      f"experiments come from the same model."))
        else:
            out.append(check_prototypes(ref.prototypes, member.prototypes, ref.name, member.name))
    return out


def comparison_problem(members: list, reference: int) -> str:
    """Why these members cannot be compared, or "" when they can."""
    if len(members) < 2:
        return "A comparison needs at least two experiments with embeddings."
    bad = [(m, c) for m, c in zip(members, check_members(members, reference)) if not c.ok]
    if bad:
        return " ".join(f"{m.name}: {c.reason}" for m, c in bad)
    return ""


# ---- the map -------------------------------------------------------------------

def sample_rows(n: int, cap: int, seed: int = 0) -> np.ndarray:
    """At most `cap` row indices of `n`, sorted; all of them when n <= cap."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))


@dataclass
class ComparisonMap:
    mode: str
    method: str
    rows: list            # per member: the row indices drawn
    points: list          # per member: (n_i, 2)
    proto_points: Optional[np.ndarray] = None
    proto_labels: list = field(default_factory=list)
    proto_ks: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _prototype_vectors(protos: Optional[PrototypeSet], classes: Optional[set]):
    """(vectors, labels, ks) of the prototypes to draw, or (None, [], [])."""
    if protos is None:
        return None, [], []
    vecs, labels, ks = [], [], []
    for c, label in enumerate(protos.class_labels):
        if classes is not None and str(label) not in classes:
            continue
        for k in range(protos.k):
            v = protos.vectors[c, k]
            if np.all(np.isfinite(v)):
                vecs.append(v)
                labels.append(str(label))
                ks.append(k)
    if not vecs or len(vecs) > MAX_MAP_PROTOTYPES:
        return None, [], []
    return np.stack(vecs).astype(np.float32), labels, ks


def project_members(members: list, reference: int, mode: str = PROJ_JOINT, method: str = "pca",
                    per_member: int = DEFAULT_POINTS_PER_MEMBER, seed: int = 0) -> ComparisonMap:
    """Every member's sample in one 2D space (see the module docstring for
    the two modes). The prototypes drawn are the reference's -- the same as
    every member's, once the comparison is allowed."""
    if mode not in PROJECTION_MODES:
        raise ValueError(f"Unknown projection mode {mode!r}: {', '.join(PROJECTION_MODES)}.")
    if method not in METHODS:
        raise ValueError(f"Unknown projection method {method!r}: {', '.join(METHODS)}.")
    if mode == PROJ_REFERENCE and method not in TRANSFORM_METHODS:
        raise ValueError("t-SNE cannot place points it was not fitted on, so it cannot project onto a "
                         "reference: use PCA or UMAP, or the joint projection.")
    per_member = max(10, min(int(per_member), MAX_POINTS_PER_MEMBER))
    rows = [sample_rows(len(m), per_member, seed + i) for i, m in enumerate(members)]
    classes = set().union(*({str(x) for x in m.labels[r] if x} for m, r in zip(members, rows)))
    proto_vecs, proto_labels, proto_ks = _prototype_vectors(members[reference].prototypes, classes)
    notes = []
    if members[reference].prototypes is not None and proto_vecs is None:
        notes.append(f"Prototypes not drawn: more than {MAX_MAP_PROTOTYPES} of them.")
    if mode == PROJ_JOINT:
        sizes = {len(r) for r in rows}
        if len(sizes) > 1:
            notes.append(f"Unequal samples: {', '.join(f'{len(r):,}' for r in rows)} crops -- a base with fewer "
                         f"crops than the sample size gives all it has.")
        combined = np.concatenate([m.embeddings[r] for m, r in zip(members, rows)], axis=0)
        fit = fit_projection(combined, method=method, proto_vectors=proto_vecs, seed=seed)
        bounds = np.cumsum([0] + [len(r) for r in rows])
        points = [fit.points[bounds[i]:bounds[i + 1]] for i in range(len(members))]
        proto_points = fit.proto_points
    else:
        ref_rows = rows[reference]
        fit = fit_projection(members[reference].embeddings[ref_rows], method=method, proto_vectors=proto_vecs,
                             seed=seed)
        points = []
        for i, (m, r) in enumerate(zip(members, rows)):
            if i == reference:
                points.append(fit.points)
                continue
            try:
                points.append(fit.transform(m.embeddings[r]))
            except OutOfSampleUnsupported as ex:
                raise ValueError(str(ex)) from ex
        proto_points = fit.proto_points
    return ComparisonMap(mode=mode, method=method, rows=rows, points=points, proto_points=proto_points,
                         proto_labels=proto_labels, proto_ks=proto_ks, notes=notes)


# ---- neighbours ----------------------------------------------------------------

def nearest_k(queries: np.ndarray, pool: np.ndarray, k: int, exclude_self: bool = False,
              max_cells: int = 50_000_000) -> tuple:
    """(similarities, indices), each (Q, k), of each query's k nearest rows
    of `pool`, nearest first, by cosine similarity on unit vectors. Blocks
    of queries keep the similarity matrix under `max_cells` floats.
    `exclude_self`: queries ARE the pool (same rows, same order), and a row
    is not its own neighbour."""
    queries = np.asarray(queries, dtype=np.float32)
    pool = np.asarray(pool, dtype=np.float32)
    k = max(0, min(int(k), len(pool) - (1 if exclude_self else 0)))
    nq = len(queries)
    sims = np.zeros((nq, k), dtype=np.float32)
    idx = np.zeros((nq, k), dtype=np.int64)
    if not k or not nq:
        return sims, idx
    block = max(1, int(max_cells // max(len(pool), 1)))
    for start in range(0, nq, block):
        s = queries[start:start + block] @ pool.T
        if exclude_self:
            rows = np.arange(len(s))
            s[rows, start + rows] = -np.inf
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(s, top, axis=1)
        order = np.argsort(-top_s, axis=1)
        idx[start:start + len(s)] = np.take_along_axis(top, order, axis=1)
        sims[start:start + len(s)] = np.take_along_axis(top_s, order, axis=1)
    return sims, idx


@dataclass
class CrossNeighbour:
    index: int            # row in the target member
    uuid: str
    similarity: float
    label: Optional[str]


def neighbours_across(query: Member, query_index: int, target: Member, k: int = 20) -> list:
    """The k crops of `target` nearest to crop `query_index` of `query`,
    over the whole of `target` (exact). When both are the same member the
    crop itself is left out."""
    same = query.ref == target.ref
    found = nearest_to_vector(target.embeddings, target.uuids, query.embeddings[query_index], k=k,
                              labels=target.labels, exclude=query_index if same else None)
    position = {u: i for i, u in enumerate(target.uuids.tolist())} if found else {}
    return [CrossNeighbour(index=position[n.uuid], uuid=n.uuid, similarity=n.similarity, label=n.label or None)
            for n in found]


# ---- measures ------------------------------------------------------------------

def domain_separability(a: np.ndarray, b: np.ndarray, seed: int = 0) -> Optional[float]:
    """Cross-validated ROC AUC of a logistic regression telling rows of `a`
    from rows of `b`, on equal samples of each. 0.5: the bases are
    indistinguishable to the model; 1: told apart perfectly. None with too
    few crops to cross-validate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    n = min(len(a), len(b), STATS_MAX_POINTS)
    if n < 2 * DOMAIN_FOLDS:
        return None
    rng = np.random.default_rng(seed)
    xa = a[rng.choice(len(a), n, replace=False)]
    xb = b[rng.choice(len(b), n, replace=False)]
    x = np.concatenate([xa, xb]).astype(np.float32)
    y = np.r_[np.zeros(n), np.ones(n)]
    model = LogisticRegression(max_iter=500, C=1.0)
    folds = StratifiedKFold(n_splits=DOMAIN_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_predict(model, x, y, cv=folds, method="decision_function")
    return float(roc_auc_score(y, scores))


def _majority(labels_2d: np.ndarray) -> tuple:
    """Per row: (most frequent non-empty label, its share of the row)."""
    out_label, out_share = [], []
    for row in labels_2d:
        values = [v for v in row if v]
        if not values:
            out_label.append("")
            out_share.append(0.0)
            continue
        names, counts = np.unique(np.asarray(values, dtype=object).astype(str), return_counts=True)
        best = int(np.argmax(counts))
        out_label.append(str(names[best]))
        out_share.append(float(counts[best]) / len(row))
    return np.asarray(out_label, dtype=object), np.asarray(out_share)


def _prototype_similarity(embeddings: np.ndarray, labels: np.ndarray, protos: Optional[PrototypeSet]) -> np.ndarray:
    """Per crop: cosine similarity to its own class's nearest prototype
    (NaN without a label, or with no prototype for its class)."""
    out = np.full(len(labels), np.nan)
    if protos is None:
        return out
    index = protos.label_index()
    for label in set(labels.tolist()) - {""}:
        c = index.get(label)
        if c is None:
            continue
        rows = np.flatnonzero(labels == label)
        vecs = protos.vectors[c]                     # (K, D)
        finite = np.all(np.isfinite(vecs), axis=1)
        if not finite.any():
            continue
        out[rows] = np.max(embeddings[rows] @ vecs[finite].T, axis=1)
    return out


@dataclass
class ClassShift:
    label: str
    n_reference: int
    n_other: int
    centroid_similarity: Optional[float]     # cosine between the class's two centroids
    prototype_reference: Optional[float]     # mean similarity to the class's prototype
    prototype_other: Optional[float]
    accuracy_reference: Optional[float]
    accuracy_other: Optional[float]
    coverage: Optional[float]                # share of the other's crops of the class covered
    agreement: Optional[float]               # share whose reference neighbours agree with the label


@dataclass
class PairStats:
    reference: str
    other: str
    n_reference: int            # crops measured on each side
    n_other: int
    n_shared: int               # crops (uuids) in both bases, left out of the domain test
    domain_auc: Optional[float]
    coverage_threshold: float
    coverage: float
    nn_similarity_reference: float    # median similarity to the nearest reference crop, inside the reference
    nn_similarity_other: float        # ... and from the other base
    agreement_reference: Optional[float]
    agreement: Optional[float]
    per_class: list = field(default_factory=list)       # ClassShift
    outside: list = field(default_factory=list)         # (row in other, similarity to nearest reference crop)
    disagreements: list = field(default_factory=list)   # (row in other, own label, neighbours' label, share)


def _mean(values) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def pair_stats(reference: Member, other: Member, seed: int = 0) -> PairStats:
    """Every measure of `other` against `reference` (see the module
    docstring). Rows quoted in `outside` and `disagreements` are rows of
    `other`."""
    pool_rows = sample_rows(len(reference), KNN_POOL_MAX, seed)
    pool = reference.embeddings[pool_rows]
    pool_labels = reference.labels[pool_rows]
    ref_rows = pool_rows[sample_rows(len(pool_rows), STATS_MAX_POINTS, seed + 1)]
    other_rows = sample_rows(len(other), STATS_MAX_POINTS, seed + 2)
    k = max(1, min(AGREEMENT_K, len(pool) - 1))
    # The reference against itself: its usual nearest-neighbour similarity,
    # and how often its own labels agree with their neighbours'.
    position = np.searchsorted(pool_rows, ref_rows)
    self_sims, self_idx = _self_neighbours(pool, position, k)
    ref_nn = self_sims[:, 0] if self_sims.shape[1] else np.zeros(len(ref_rows))
    threshold = float(np.percentile(ref_nn, COVERAGE_PERCENTILE)) if len(ref_nn) else 0.0
    ref_labels = reference.labels[ref_rows]
    self_major, _ = _majority(pool_labels[self_idx])
    labelled_ref = (ref_labels != "") & (self_major != "")
    agreement_reference = float(np.mean(self_major[labelled_ref] == ref_labels[labelled_ref])) \
        if labelled_ref.any() else None
    # The other base against the reference.
    other_sims, other_idx = nearest_k(other.embeddings[other_rows], pool, k)
    other_nn = other_sims[:, 0] if other_sims.shape[1] else np.zeros(len(other_rows))
    covered = other_nn >= threshold
    other_labels = other.labels[other_rows]
    major, share = _majority(pool_labels[other_idx])
    labelled = (other_labels != "") & (major != "")
    agrees = major == other_labels
    agreement = float(np.mean(agrees[labelled])) if labelled.any() else None
    # A crop in both bases is the same vector on both sides: in a held-out
    # fold its twin, labelled with the other side, pushes the classifier
    # below chance. They are left out of the domain test.
    shared = np.intersect1d(reference.uuids, other.uuids)
    only_ref = ~np.isin(reference.uuids, shared)
    only_other = ~np.isin(other.uuids, shared)
    stats = PairStats(
        reference=reference.name, other=other.name, n_reference=len(ref_rows), n_other=len(other_rows),
        n_shared=int(len(shared)),
        domain_auc=domain_separability(reference.embeddings[only_ref], other.embeddings[only_other], seed),
        coverage_threshold=threshold, coverage=float(covered.mean()) if len(covered) else 0.0,
        nn_similarity_reference=float(np.median(ref_nn)) if len(ref_nn) else 0.0,
        nn_similarity_other=float(np.median(other_nn)) if len(other_nn) else 0.0,
        agreement_reference=agreement_reference, agreement=agreement)
    order = np.argsort(other_nn)[:TOP_LISTED]
    stats.outside = [(int(other_rows[i]), float(other_nn[i])) for i in order if not covered[i]]
    wrong = np.flatnonzero(labelled & ~agrees)
    wrong = wrong[np.argsort(-share[wrong])][:TOP_LISTED]
    stats.disagreements = [(int(other_rows[i]), str(other_labels[i]), str(major[i]), float(share[i])) for i in wrong]
    stats.per_class = _class_shifts(reference, other, ref_rows, other_rows, covered, agrees, labelled)
    return stats


def _self_neighbours(pool: np.ndarray, position: np.ndarray, k: int) -> tuple:
    """Neighbours in `pool` of the pool rows at `position`, the row itself
    left out -- for a pool larger than the queries."""
    sims, idx = nearest_k(pool[position], pool, k + 1)
    # The row itself is normally first; a duplicate crop can tie with it,
    # so it is removed by index rather than by rank.
    is_self = idx == position[:, None]
    has_self = is_self.any(axis=1)
    # Rows without themselves in the list (k + 1 ties ahead of it) drop the last.
    drop = np.where(has_self, np.argmax(is_self, axis=1), k)
    keep = np.ones_like(is_self)
    keep[np.arange(len(position)), drop] = False
    return sims[keep].reshape(len(position), k), idx[keep].reshape(len(position), k)


def _class_shifts(reference: Member, other: Member, ref_rows, other_rows, covered, agrees, labelled) -> list:
    ref_labels, other_labels = reference.labels[ref_rows], other.labels[other_rows]
    ref_proto = _prototype_similarity(reference.embeddings[ref_rows], ref_labels, reference.prototypes)
    other_proto = _prototype_similarity(other.embeddings[other_rows], other_labels, reference.prototypes)
    classes = sorted((set(ref_labels.tolist()) | set(other_labels.tolist())) - {""}, key=str)
    out = []
    for label in classes:
        in_ref = ref_labels == label
        in_other = other_labels == label
        centroid = None
        if in_ref.any() and in_other.any():
            ca = reference.embeddings[ref_rows[in_ref]].mean(axis=0)
            cb = other.embeddings[other_rows[in_other]].mean(axis=0)
            denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))
            centroid = float(ca @ cb / denom) if denom > 0 else None
        acc_ref = _mean(reference.correct[ref_rows[in_ref]]) if reference.correct is not None else None
        acc_other = _mean(other.correct[other_rows[in_other]]) if other.correct is not None else None
        out.append(ClassShift(
            label=str(label), n_reference=int(in_ref.sum()), n_other=int(in_other.sum()),
            centroid_similarity=centroid, prototype_reference=_mean(ref_proto[in_ref]),
            prototype_other=_mean(other_proto[in_other]), accuracy_reference=acc_ref, accuracy_other=acc_other,
            coverage=float(covered[in_other].mean()) if in_other.any() else None,
            agreement=float(agrees[in_other & labelled].mean()) if (in_other & labelled).any() else None))
    return out


# Per-class sort keys of the Compare embeddings tab and the MCP tool (keys,
# compared in code): the worst first for each.
CLASS_SORTS = {
    "coverage": ("Coverage (lowest first)", lambda c: (c.coverage is None, c.coverage or 0.0)),
    "agreement": ("Label agreement (lowest first)", lambda c: (c.agreement is None, c.agreement or 0.0)),
    "centroid": ("Centroid similarity (lowest first)",
                 lambda c: (c.centroid_similarity is None, c.centroid_similarity or 0.0)),
    "accuracy_drop": ("Accuracy drop (largest first)",
                      lambda c: (c.accuracy_reference is None or c.accuracy_other is None,
                                 -((c.accuracy_reference or 0.0) - (c.accuracy_other or 0.0)))),
    "support": ("Crops in the other base (most first)", lambda c: -c.n_other),
}


def sorted_classes(rows: list, key: str) -> list:
    _, fn = CLASS_SORTS.get(key, CLASS_SORTS["coverage"])
    return sorted(rows, key=fn)


# ---- colouring the map (keys: the tab's dropdown and the MCP argument) ----------

COLOR_MEMBER = "experiment"
COLOR_CLASS = "class"
COLOR_CORRECT = "correct"
COLOR_MODES = (COLOR_MEMBER, COLOR_CLASS, COLOR_CORRECT)
# Right / wrong / no label, as the Gallery frames them.
CORRECT_COLORS = {"right": "#1D7148", "wrong": "#B42318", "unlabelled": "#9AA1AB"}
CORRECT_NAMES = {"right": "Right", "wrong": "Wrong", "unlabelled": "No label"}


def point_groups(member: Member, rows: np.ndarray, color_by: str) -> list:
    """The group each drawn point of `member` is colored as: its class, its
    outcome (right / wrong / no label), or the member itself."""
    if color_by == COLOR_CLASS:
        return [label or "(no label)" for label in member.labels[rows].tolist()]
    if color_by == COLOR_CORRECT:
        if member.correct is None:
            return [CORRECT_NAMES["unlabelled"]] * len(rows)
        values = member.correct[rows]
        return [CORRECT_NAMES["unlabelled"] if not np.isfinite(v) else CORRECT_NAMES["right" if v else "wrong"]
                for v in values]
    return [member.name] * len(rows)


def group_colors(members: list, cmap: ComparisonMap, color_by: str, member_colors: dict) -> dict:
    """group -> color for every group point_groups() can return."""
    if color_by == COLOR_CORRECT:
        return {CORRECT_NAMES[k]: v for k, v in CORRECT_COLORS.items()}
    if color_by == COLOR_CLASS:
        from lookin.core.point_coloring import categorical_colors

        labels = sorted({g for m, r in zip(members, cmap.rows) for g in point_groups(m, r, COLOR_CLASS)}, key=str)
        return dict(zip(labels, categorical_colors(len(labels))))
    return {m.name: member_colors.get(m.name, "#555555") for m in members}


# ---- which experiments of the library can join -------------------------------

_proto_lock = threading.Lock()
_proto_cache: dict = {}   # path -> (mtime, PrototypeSet | None)


def read_prototypes(path) -> Optional[PrototypeSet]:
    """A prototypes file on its own (small, unlike the embeddings), None
    when it is missing or unreadable -- enough to tell, before loading
    anything else, which experiments of the library share a model."""
    if not path:
        return None
    mtime = _mtime(path)
    with _proto_lock:
        hit = _proto_cache.get(str(path))
        if hit is not None and hit[0] == mtime:
            return hit[1]
    protos = None
    if mtime:
        from lookin.core.data_manager import Experiment

        try:
            protos = Experiment(name="", df=pd.DataFrame(), uuid_col="", score_cols=[]).load_model_prototypes(str(path))
        except Exception:  # noqa: BLE001 -- an unreadable file is "no prototypes"
            protos = None
    with _proto_lock:
        _proto_cache[str(path)] = (mtime, protos)
    return protos


def candidates_for(reference: Optional[PrototypeSet], listed: list) -> tuple:
    """(same model, others): `listed` (comparable_experiments() rows, each
    with a "prototypes_path") split by whether their prototypes match the
    reference's. With no reference yet, every experiment that has
    prototypes is a candidate."""
    same, others = [], []
    for row in listed:
        protos = read_prototypes(row.get("prototypes_path"))
        if protos is None:
            others.append(row)
        elif reference is None or check_prototypes(reference, protos).ok:
            same.append(row)
        else:
            others.append(row)
    return same, others


def rename_member_everywhere(workspace_id: str, old: str, new: str, root=None) -> int:
    """Follow an experiment's rename in every comparison of the library
    that names it (a comparison saved in another workspace may). Returns
    how many workspaces changed."""
    from lookin.core import library as lib

    if not workspace_id or old == new:
        return 0
    changed = 0
    for workspace in lib.list_workspaces(root):
        refs, reference = comparison_from_json(workspace.embedding_comparison)
        if MemberRef(workspace_id, old) not in refs:
            continue

        def change(fresh) -> bool:
            fresh_refs, fresh_reference = comparison_from_json(fresh.embedding_comparison)
            renamed = [MemberRef(r.workspace_id, new) if r == MemberRef(workspace_id, old) else r for r in fresh_refs]
            if renamed == fresh_refs:
                return False
            fresh.embedding_comparison = comparison_to_json(renamed, fresh_reference)
            return True

        if lib.update_workspace(workspace.path, change) is not None:
            changed += 1
    return changed
