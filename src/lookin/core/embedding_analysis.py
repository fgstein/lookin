"""Embedding-space analysis for a loaded experiment: prototype-based
diagnostics (compactness, separation, nearest/farthest/impostor case
studies, dataset cleaning) and a cross-class "prototype role" breakdown.

This mirrors the functionality of arcface-suite's report.html / explorer.html
/ flagged_gallery.html, adapted to lookin's data model: everything here is a
plain function over numpy arrays (embeddings, string labels, uuids) plus a
PrototypeSet, computed against whatever pool is currently loaded in lookin
-- which is always the test split.

Two kinds of prototype sets, and they must never be conflated
---------------------------------------------------------------
* "model" -- imported from the actual trained head (e.g. real ArcFace
  sub-centers). Independent of the loaded pool, so it's the only source ever
  used for a *scored* diagnostic: compactness, purity, cleaning flags.
* "kmeans" -- estimated in-tool via per-class KMeans over the loaded pool's
  own embeddings and labels. Since the loaded pool is always the test split,
  a kmeans prototype here is built from the very labels it would be scored
  against -- circular, and worse the smaller a class's test count is (a
  2-3-point centroid is nearly the points themselves). It exists purely for
  visual/comparative use (2D-projection anchors, browsing nearest/farthest
  crops) and must never produce a percentage, a compactness number, or a
  cleaning flag. Every function below that computes a *score* raises if
  handed a "kmeans" PrototypeSet; purely descriptive functions (composition,
  nearest/farthest, projections) accept either, and the caller is
  responsible for labeling the source in the UI so the two are never shown
  as equivalent.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MODEL_SOURCE = "model"
KMEANS_SOURCE = "kmeans"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-8, None)


@dataclass
class PrototypeSet:
    vectors: np.ndarray  # (C, K, D), L2-normalized; NaN rows for classes with no prototype
    class_labels: list  # len C, aligned with vectors' first axis
    source: str  # MODEL_SOURCE or KMEANS_SOURCE
    k: int

    def label_index(self) -> dict:
        return {label: i for i, label in enumerate(self.class_labels)}

    def require_model_source(self, purpose: str) -> None:
        if self.source != MODEL_SOURCE:
            raise ValueError(
                f"{purpose} is a scored diagnostic and must use real, imported prototypes -- "
                f"a kmeans estimate is fit on the same (test) labels it would be scored against, "
                f"which is circular. Import the model's own prototypes to unlock this."
            )


def compute_kmeans_prototypes(
    embeddings: np.ndarray, labels: np.ndarray, class_order: list, k: int, seed: int = 0
) -> PrototypeSet:
    """Per-class KMeans over the *loaded* pool (always test, in lookin) --
    an estimate, not the model's real decision-boundary prototype. See the
    module docstring for why this must stay visual-only."""
    from sklearn.cluster import KMeans

    d = embeddings.shape[1]
    c = len(class_order)
    vectors = np.full((c, k, d), np.nan, dtype=np.float32)
    for i, cls in enumerate(class_order):
        mask = labels == cls
        n = int(mask.sum())
        if n == 0:
            continue
        embs = embeddings[mask]
        k_eff = min(k, n)
        if k_eff == 1:
            centers = embs.mean(axis=0, keepdims=True)
        else:
            centers = KMeans(n_clusters=k_eff, n_init=4, random_state=seed).fit(embs).cluster_centers_
        centers = l2_normalize(centers)
        vectors[i, :k_eff] = centers
        if k_eff < k:
            vectors[i, k_eff:] = centers[-1]  # pad by repeating -- a no-op under max-similarity reductions
    return PrototypeSet(vectors=vectors, class_labels=list(class_order), source=KMEANS_SOURCE, k=k)


def class_mean_prototypes(embeddings: np.ndarray, labels: np.ndarray, class_order: list) -> PrototypeSet:
    """One prototype per class: the normalized mean of the class's
    embeddings, keyed by the *dataset's* labels. compute_kmeans_prototypes
    at K=1 gives the same vectors; this is the vectorized form, since one
    boolean mask per class is slow past a few thousand classes. Same
    caveat as any kmeans estimate: descriptive only, never scored."""
    d = embeddings.shape[1]
    index = {label: i for i, label in enumerate(class_order)}
    rows = np.array([index.get(l, -1) for l in labels.tolist()], dtype=np.int64)
    keep = rows >= 0
    sums = np.zeros((len(class_order), d), dtype=np.float64)
    np.add.at(sums, rows[keep], embeddings[keep])
    counts = np.bincount(rows[keep], minlength=len(class_order))
    vectors = np.full((len(class_order), 1, d), np.nan, dtype=np.float32)
    present = counts > 0
    vectors[present, 0] = l2_normalize(sums[present])
    return PrototypeSet(vectors=vectors, class_labels=list(class_order), source=KMEANS_SOURCE, k=1)


def relabel_prototypes(protos: PrototypeSet, label_of: dict) -> PrototypeSet:
    """`protos` with each class label passed through `label_of` (labels it
    does not mention are kept). The network names its classes its own way
    -- the score columns -- and the Data tab's class mapping is what says
    which dataset class each one is; without this a prototype is matched to
    the dataset by spelling alone. Several network classes mapped onto one
    dataset class keep all their vectors, as extra subcenters of it.

    Returns `protos` itself when nothing changes, so a caller comparing by
    identity (see PrototypeSourceRegistry.assignment) keeps its cache."""
    groups: dict = {}
    for i, label in enumerate(protos.class_labels):
        groups.setdefault(label_of.get(label, label), []).append(i)
    if all(len(ix) == 1 and protos.class_labels[ix[0]] == label for label, ix in groups.items()):
        return protos
    d = protos.vectors.shape[2]
    k = max(len(ix) for ix in groups.values()) * protos.k
    vectors = np.full((len(groups), k, d), np.nan, dtype=np.float32)
    for g, ix in enumerate(groups.values()):
        stacked = protos.vectors[ix].reshape(-1, d)
        vectors[g, : len(stacked)] = stacked
    return PrototypeSet(vectors=vectors, class_labels=list(groups), source=protos.source, k=k)


def assign_nearest_prototype(embeddings: np.ndarray, protos: PrototypeSet, batch_size: int = 4096):
    """For every embedding, the single closest prototype across ALL classes
    and all K sub-centers -- not restricted to the sample's own label. This
    generalizes arcface-suite's per-class-only subcenter assignment, and is
    what exposes cross-class contamination (a class's own subcenter quietly
    absorbing another class's samples).

    Returns (class_idx[N], k_idx[N], similarity[N]); class_idx indexes into
    protos.class_labels.
    """
    c, k, d = protos.vectors.shape
    flat = protos.vectors.reshape(c * k, d)
    valid = ~np.isnan(flat).any(axis=1)
    flat_safe = np.where(np.isnan(flat), 0.0, flat)

    n = embeddings.shape[0]
    best_class = np.zeros(n, dtype=np.int64)
    best_k = np.zeros(n, dtype=np.int64)
    best_sim = np.full(n, -np.inf, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = embeddings[start:end] @ flat_safe.T  # (b, C*K)
        sims[:, ~valid] = -np.inf
        idx = sims.argmax(axis=1)
        rows = np.arange(end - start)
        best_sim[start:end] = sims[rows, idx]
        best_class[start:end] = idx // k
        best_k[start:end] = idx % k

    return best_class, best_k, best_sim


@dataclass
class PrototypeRole:
    class_label: str
    k_idx: int
    total: int
    own_count: int
    other_counts: list  # [(label, count), ...] sorted desc, excludes own class, capped
    purity: float  # own_count / total, nan if total == 0


def prototype_composition(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    protos: PrototypeSet,
    top_other: int = 2,
    assignment: Optional[tuple] = None,
) -> dict:
    """Global nearest-prototype assignment, reduced to a per-(class, k)
    breakdown of which true classes actually land on each prototype. Keyed
    by class_label -> list of PrototypeRole (one per subcenter). Valid for
    either prototype source -- purity here is a descriptive count, not by
    itself a claim about the model (see module docstring for how the caller
    must present a kmeans-sourced result).

    ``assignment``, if given, is a precomputed (best_class, best_k) pair
    from assign_nearest_prototype -- an O(N x C x K) computation that's
    worth caching across calls (e.g. across every class a UI lets the user
    pick) rather than repeating here."""
    if assignment is not None:
        best_class, best_k = assignment
    else:
        best_class, best_k, _ = assign_nearest_prototype(embeddings, protos)
    c, k, _ = protos.vectors.shape
    out: dict = {label: [] for label in protos.class_labels}
    for ci, label in enumerate(protos.class_labels):
        for ki in range(k):
            mask = (best_class == ci) & (best_k == ki)
            total = int(mask.sum())
            if total == 0:
                out[label].append(PrototypeRole(label, ki, 0, 0, [], float("nan")))
                continue
            assigned_labels = true_labels[mask]
            own = int((assigned_labels == label).sum())
            others: dict = {}
            for l in assigned_labels[assigned_labels != label]:
                others[l] = others.get(l, 0) + 1
            other_sorted = sorted(others.items(), key=lambda kv: -kv[1])[:top_other]
            out[label].append(PrototypeRole(label, ki, total, own, other_sorted, own / total))
    return out


def _class_prototype_vectors(protos: PrototypeSet, label: str, k_idx: Optional[int]):
    """(vectors, k_indices) -- valid prototype vectors for one class
    (K', D), and the *original* subcenter index each row corresponds to
    (since invalid/NaN rows are dropped, row i is not necessarily
    subcenter i). Every valid subcenter when k_idx is None (the "whole
    class, subcenters merged" view), or exactly one subcenter when given.
    (None, None) if the class/subcenter has no valid prototype."""
    idx = protos.label_index()
    if label not in idx:
        return None, None
    proto_c = protos.vectors[idx[label]]  # (K, D)
    if k_idx is not None:
        if k_idx >= proto_c.shape[0] or np.isnan(proto_c[k_idx]).any():
            return None, None
        return proto_c[k_idx : k_idx + 1], np.array([k_idx])
    valid = ~np.isnan(proto_c).any(axis=1)
    if not valid.any():
        return None, None
    return proto_c[valid], np.where(valid)[0]


def nearest_farthest_within_class(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    uuids: np.ndarray,
    protos: PrototypeSet,
    label: str,
    k_top: int = 8,
    k_idx: Optional[int] = None,
):
    """Own-class-only similarity (unlike prototype_composition's global
    assignment) -- the "does this image look like a typical member of its
    own labeled class" question, valid for either prototype source.
    ``k_idx``, if given, restricts to that one subcenter instead of the
    class's best-matching subcenter per sample.

    Returns (nearest, farthest), each a list of (uuid, similarity,
    matched_k_idx) -- the subcenter each sample actually matched, so a
    caller merging subcenters (k_idx=None) can still show which one won.
    """
    proto_vecs, k_indices = _class_prototype_vectors(protos, label, k_idx)
    if proto_vecs is None:
        return [], []
    mask = true_labels == label
    if not mask.any():
        return [], []
    embs = embeddings[mask]
    sim_matrix = embs @ proto_vecs.T  # (n, K')
    best_local = sim_matrix.argmax(axis=1)
    rows = np.arange(len(sim_matrix))
    sims = sim_matrix[rows, best_local]
    matched_k = k_indices[best_local]
    order = np.argsort(-sims)
    ids = uuids[mask]
    nearest = list(zip(ids[order[:k_top]].tolist(), sims[order[:k_top]].tolist(), matched_k[order[:k_top]].tolist()))
    tail = order[-k_top:][::-1]
    farthest = list(zip(ids[tail].tolist(), sims[tail].tolist(), matched_k[tail].tolist()))
    return nearest, farthest


def nearest_impostors(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    uuids: np.ndarray,
    protos: PrototypeSet,
    label: str,
    k_top: int = 8,
    k_idx: Optional[int] = None,
):
    """Other classes' samples closest to this class's prototype -- the
    "who's most likely to be mistaken for this identity" question.
    ``k_idx``, if given, restricts to that one subcenter.

    Returns a list of (uuid, true_label, similarity, matched_k_idx).
    """
    proto_vecs, k_indices = _class_prototype_vectors(protos, label, k_idx)
    if proto_vecs is None:
        return []
    mask = true_labels != label
    if not mask.any():
        return []
    embs = embeddings[mask]
    sim_matrix = embs @ proto_vecs.T
    best_local = sim_matrix.argmax(axis=1)
    rows = np.arange(len(sim_matrix))
    sims = sim_matrix[rows, best_local]
    matched_k = k_indices[best_local]
    order = np.argsort(-sims)[:k_top]
    ids = uuids[mask]
    lbls = true_labels[mask]
    return list(zip(ids[order].tolist(), lbls[order].tolist(), sims[order].tolist(), matched_k[order].tolist()))


@dataclass
class ClassEmbeddingStats:
    label: str
    n: int
    compactness: float  # mean cosine sim to own prototype (best over K)
    separation: float  # cosine sim to nearest OTHER class's prototype


def per_class_stats(embeddings: np.ndarray, true_labels: np.ndarray, protos: PrototypeSet) -> list:
    protos.require_model_source("compactness/separation")
    c, k, d = protos.vectors.shape
    flat_valid_mean = np.nan_to_num(np.nanmean(protos.vectors, axis=1))  # (C, D) representative vector per class
    mean_norm = l2_normalize(flat_valid_mean)
    sim_matrix = mean_norm @ mean_norm.T
    np.fill_diagonal(sim_matrix, -np.inf)
    nearest_other_sim = sim_matrix.max(axis=1)

    idx = protos.label_index()
    # The rows grouped by true class once, then one slice per class, rather
    # than comparing every row against every label.
    codes = np.array([idx.get(l, -1) for l in true_labels], dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(c + 1))
    out = []
    for label, ci in idx.items():
        proto_c = protos.vectors[ci]
        valid = ~np.isnan(proto_c).any(axis=1)
        rows = order[bounds[ci]:bounds[ci + 1]]
        n = int(len(rows))
        if n == 0 or not valid.any():
            out.append(ClassEmbeddingStats(label, n, float("nan"), float("nan")))
            continue
        sims = (embeddings[rows] @ proto_c[valid].T).max(axis=1)
        out.append(ClassEmbeddingStats(label, n, float(sims.mean()), float(nearest_other_sim[ci])))
    return out


@dataclass
class PrototypeStats:
    class_label: str
    k_idx: int
    n_assigned: int  # samples whose globally-nearest prototype is this one (prototype_composition's role.total)
    compactness: float  # mean similarity of those *assigned* samples to this exact prototype vector
    separation: float  # similarity to the nearest OTHER prototype, any class, *excluding this class's own subcenters*
    purity: float  # share of assigned samples whose true label actually matches this prototype's class


def per_prototype_stats(
    embeddings: np.ndarray, true_labels: np.ndarray, protos: PrototypeSet, assignment: Optional[tuple] = None
) -> list:
    """The subcenter-level counterpart to per_class_stats: one row per
    individual (class, k) prototype instead of one blended row per class.
    per_class_stats reduces over subcenters (max-similarity for
    compactness, the class's mean vector for separation) specifically to
    give a single per-class summary; this doesn't reduce anything, so two
    subcenters of the same class can -- and often do -- look very
    different here even when their per_class_stats average looks fine.

    ``assignment``, if given, is a precomputed (best_class, best_k) from
    assign_nearest_prototype -- worth reusing since it's the same O(N x C x K)
    computation the role panel already caches.
    """
    protos.require_model_source("per-prototype stats")
    if assignment is not None:
        best_class, best_k = assignment
    else:
        best_class, best_k, _ = assign_nearest_prototype(embeddings, protos)

    c, k, d = protos.vectors.shape
    flat = protos.vectors.reshape(c * k, d)
    valid_flat = ~np.isnan(flat).any(axis=1)
    flat_safe = np.where(np.isnan(flat), 0.0, flat)

    sim_all = flat_safe @ flat_safe.T  # (C*K, C*K)
    class_of_flat = np.repeat(np.arange(c), k)
    exclude = (class_of_flat[:, None] == class_of_flat[None, :]) | ~valid_flat[None, :]
    sim_other_class = np.where(exclude, -np.inf, sim_all)

    out = []
    for ci in range(c):
        for ki in range(k):
            flat_idx = ci * k + ki
            if not valid_flat[flat_idx]:
                continue
            mask = (best_class == ci) & (best_k == ki)
            n_assigned = int(mask.sum())
            if n_assigned == 0:
                compactness, purity = float("nan"), float("nan")
            else:
                sims = embeddings[mask] @ flat_safe[flat_idx]
                compactness = float(sims.mean())
                purity = float((true_labels[mask] == protos.class_labels[ci]).mean())
            row = sim_other_class[flat_idx]
            separation = float(row.max()) if np.isfinite(row).any() else float("nan")
            out.append(PrototypeStats(protos.class_labels[ci], ki, n_assigned, compactness, separation, purity))
    return out


def prototype_confusion_pairs(protos: PrototypeSet, top_n: int = 15) -> list:
    """Prototype pairs (by class mean vector) with the highest cosine
    similarity -- the identities the model is most likely to mix up."""
    protos.require_model_source("prototype confusion pairs")
    mean_vecs = l2_normalize(np.nan_to_num(np.nanmean(protos.vectors, axis=1)))
    sim = mean_vecs @ mean_vecs.T
    np.fill_diagonal(sim, -np.inf)
    flat_idx = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
    seen, rows = set(), []
    for a, b in flat_idx:
        key = tuple(sorted((int(a), int(b))))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"a": protos.class_labels[a], "b": protos.class_labels[b], "sim": float(sim[a, b])})
        if len(rows) >= top_n:
            break
    return rows


def prototype_similarity_matrix(protos: PrototypeSet, class_order: Optional[list] = None):
    """Pairwise cosine similarity between every *individual* valid
    prototype (not the per-class mean used by prototype_confusion_pairs) --
    finer-grained than the confusion-pairs table, since two classes can
    share a mean-level similarity while actually being confused through
    just one particular subcenter.

    Rows/columns are grouped by class -- every subcenter of a class sits
    together -- in ``class_order`` (defaults to the PrototypeSet's own
    order; pass e.g. the worst-margin classes from per_class_stats to focus
    the matrix on whatever's actually interesting instead of everything).
    Classes/subcenters with no valid prototype are simply omitted.

    Returns (labels, matrix): labels is a list of (class_label, k_idx)
    aligned with the matrix's rows/cols.
    """
    protos.require_model_source("prototype similarity matrix")
    order = class_order if class_order is not None else protos.class_labels
    idx = protos.label_index()
    labels, vecs = [], []
    for cls in order:
        if cls not in idx:
            continue
        proto_c = protos.vectors[idx[cls]]
        for k in range(proto_c.shape[0]):
            if np.isnan(proto_c[k]).any():
                continue
            labels.append((cls, k))
            vecs.append(proto_c[k])
    if not vecs:
        return [], np.zeros((0, 0))
    mat = np.stack(vecs)
    return labels, mat @ mat.T


def prototype_assignment_matrix(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    protos: PrototypeSet,
    class_order: Optional[list] = None,
    assignment: Optional[tuple] = None,
):
    """Confusion matrix at prototype granularity: rows are true classes,
    columns are individual (class, k) prototypes, restricted to
    ``class_order`` on both axes (e.g. the same worst-margin subset
    prototype_similarity_matrix was given, so the two stay comparable).
    Cell (X, Y) is the count of test samples truly labeled X whose
    globally-nearest prototype is Y -- the same question the role panel
    answers for one class at a time, gridded across every selected class
    at once.

    ``assignment``, if given, is a precomputed (best_class, best_k) from
    assign_nearest_prototype, worth reusing since it's already cached
    elsewhere (role panel, per_prototype_stats).

    Returns (row_labels, col_labels, matrix): row_labels is a list of class
    labels, col_labels a list of (class_label, k_idx), matrix has shape
    (len(row_labels), len(col_labels)).
    """
    protos.require_model_source("prototype assignment matrix")
    if assignment is not None:
        best_class, best_k = assignment
    else:
        best_class, best_k, _ = assign_nearest_prototype(embeddings, protos)

    order = class_order if class_order is not None else protos.class_labels
    idx = protos.label_index()
    row_labels = [cls for cls in order if cls in idx]
    row_of = {cls: i for i, cls in enumerate(row_labels)}

    col_labels = []
    col_lookup = {}  # (class_idx, k_idx) -> column position
    for cls in row_labels:
        ci = idx[cls]
        proto_c = protos.vectors[ci]
        for k in range(proto_c.shape[0]):
            if np.isnan(proto_c[k]).any():
                continue
            col_lookup[(ci, k)] = len(col_labels)
            col_labels.append((cls, k))

    row_idx_of_sample = np.array([row_of.get(l, -1) for l in true_labels])
    col_idx_of_sample = np.array([col_lookup.get((int(bc), int(bk)), -1) for bc, bk in zip(best_class, best_k)])
    valid = (row_idx_of_sample >= 0) & (col_idx_of_sample >= 0)

    matrix = np.zeros((len(row_labels), len(col_labels)), dtype=np.int64)
    np.add.at(matrix, (row_idx_of_sample[valid], col_idx_of_sample[valid]), 1)
    return row_labels, col_labels, matrix


@dataclass
class CleaningFlags:
    uuid: np.ndarray
    label: np.ndarray
    own_similarity: np.ndarray
    best_other_label: np.ndarray
    best_other_similarity: np.ndarray
    flagged: np.ndarray
    reasons: list  # str per row, ";"-joined


def _max_over_subcenters(sims: np.ndarray, c: int, k: int) -> np.ndarray:
    """(b, C*K) similarities to every subcenter -> (b, C), the best subcenter
    of each class. An element-wise maximum across the K subcenters rather
    than .max(axis=2): numpy reduces a length-2 innermost axis some twenty
    times slower than it compares two arrays."""
    view = sims.reshape(len(sims), c, k)
    best = view[:, :, 0].copy()
    for ki in range(1, k):
        np.maximum(best, view[:, :, ki], out=best)
    return best


# Rows per block in flag_noisy_samples (see there).
NOISY_BLOCK = 4096


def flag_noisy_samples(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    uuids: np.ndarray,
    protos: PrototypeSet,
    strategies: tuple = ("closer_to_other_class", "percentile"),
    percentile: float = 2.0,
    other_class_margin: float = 0.0,
    min_class_size: int = 10,
) -> CleaningFlags:
    protos.require_model_source("dataset cleaning")
    c, k, d = protos.vectors.shape
    flat = protos.vectors.reshape(c * k, d)
    valid = ~np.isnan(flat).any(axis=1)
    flat_safe = np.where(np.isnan(flat), 0.0, flat)

    label_to_col = protos.label_index()
    own_col = np.array([label_to_col.get(l, -1) for l in true_labels], dtype=np.int64)
    has_own = own_col >= 0
    n = len(true_labels)
    own_similarity = np.full(n, np.nan, dtype=np.float32)
    best_other_label = np.array([None] * n, dtype=object)
    best_other_similarity = np.full(n, np.nan, dtype=np.float32)
    class_labels = np.array(protos.class_labels, dtype=object)

    # A block of rows at a time: the (N, C*K) similarity matrix is 1.6 GB at
    # a hundred thousand crops and two thousand classes of two subcenters,
    # and only two numbers per row are kept from it.
    for start in range(0, n, NOISY_BLOCK):
        stop = min(start + NOISY_BLOCK, n)
        sims = embeddings[start:stop] @ flat_safe.T
        sims[:, ~valid] = -np.inf
        sims = _max_over_subcenters(sims, c, k)  # (b, C)
        local = np.flatnonzero(has_own[start:stop])
        if not len(local):
            continue
        cols = own_col[start:stop][local]
        rows = start + local
        own_similarity[rows] = sims[local, cols]
        sims[local, cols] = -np.inf
        best_idx = sims[local].argmax(axis=1)
        best_other_label[rows] = class_labels[best_idx]
        best_other_similarity[rows] = sims[local, best_idx]

    rows = np.where(has_own)[0]
    reasons = [[] for _ in range(n)]
    if "closer_to_other_class" in strategies:
        mask = has_own & (best_other_similarity > own_similarity + other_class_margin)
        for i in np.where(mask)[0]:
            reasons[i].append("closer_to_other_class")
    if "percentile" in strategies:
        # The rows grouped by class once, then one slice per class.
        order = rows[np.argsort(own_col[rows], kind="stable")]
        bounds = np.searchsorted(own_col[order], np.arange(c + 1))
        for ci in range(c):
            members = order[bounds[ci]:bounds[ci + 1]]
            if len(members) < min_class_size:
                continue
            threshold = np.percentile(own_similarity[members], percentile)
            for i in members[own_similarity[members] <= threshold]:
                reasons[i].append("percentile_low_similarity")

    flagged = np.array([len(r) > 0 for r in reasons])
    return CleaningFlags(
        uuid=uuids, label=true_labels, own_similarity=own_similarity,
        best_other_label=best_other_label, best_other_similarity=best_other_similarity,
        flagged=flagged, reasons=[";".join(r) for r in reasons],
    )


PROJECTION_METHODS = ("pca", "tsne", "umap", "umap_circle")
# The methods that can place a point they were not fitted on. PCA is a
# linear map, so it is exact; UMAP learns an approximate one from the knn
# structure of what it was fitted on. t-SNE learns no map at all -- its
# output is a set of coordinates optimized jointly, with nothing to apply
# to a new vector afterwards -- which is why adding points to a t-SNE map
# means fitting it again, and the Embeddings tab says so rather than
# offering a choice it cannot honor.
OUT_OF_SAMPLE_METHODS = ("pca", "umap", "umap_circle")


class OutOfSampleUnsupported(RuntimeError):
    """Raised by ProjectionFit.transform for a method that cannot place new
    points (t-SNE). A distinct type, not a plain error string, because the
    UI's answer to it is a specific one: refit with the new points
    included."""


def _haversine_to_disk(raw: np.ndarray) -> np.ndarray:
    """(colatitude, longitude) in radians -> the unit disk, by azimuthal
    equidistant projection, as in UMAP's own "embedding on a sphere"
    example. Its own function because both halves of a spherical
    projection need it and have to agree: the points a fit produced, and
    any point placed onto that fit afterwards."""
    x = np.sin(raw[:, 0]) * np.cos(raw[:, 1])
    y = np.sin(raw[:, 0]) * np.sin(raw[:, 1])
    return np.stack([x, y], axis=1)


@dataclass
class ProjectionFit:
    """One fitted 2D projection: the coordinates it produced, and -- when
    the method allows it -- the fitted model itself, so that points which
    were not part of the fit can be placed onto the same map.

    Why the model is kept rather than thrown away
    ---------------------------------------------
    Adding a handful of crops to a map used to mean re-projecting
    everything, which moves every existing point: the cloud a user has
    spent a minute reading rearranges itself, and nothing about the new
    picture is comparable to the old one. Keeping the fit means the map
    holds still and the new points land in it -- the difference between
    "where does this image fall?" and "here is a different map".

    The cost is memory: a fitted UMAP holds on to the data it was fitted on
    and its knn index. That is a deliberate trade (the alternative is
    re-projecting on every overlay toggle), and it is bounded by the sample
    size the run was given.
    """
    method: str
    points: np.ndarray                      # (n, 2) the fitted points
    proto_points: Optional[np.ndarray] = None   # (m, 2) prototypes, jointly placed
    model: Optional[object] = None          # the fitted estimator, None for t-SNE

    def supports_transform(self) -> bool:
        return self.model is not None

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Place vectors that were not part of the fit, in the fit's own
        coordinates. Raises OutOfSampleUnsupported when the method has no
        such map to apply."""
        if self.model is None:
            raise OutOfSampleUnsupported(
                f"'{self.method}' cannot place points it was not fitted on -- it has to be fitted "
                f"again with them included."
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if not len(embeddings):
            return np.zeros((0, 2), dtype=float)
        out = self.model.transform(embeddings)
        if self.method == "umap_circle":
            return _haversine_to_disk(np.asarray(out, dtype=float))
        return np.asarray(out, dtype=float)


def fit_projection(
    embeddings: np.ndarray, method: str = "tsne", proto_vectors: Optional[np.ndarray] = None,
    seed: int = 0,
) -> ProjectionFit:
    """Fit a 2D projection of `embeddings` (jointly with `proto_vectors`,
    if given, so prototypes land where they belong relative to the points)
    and return it as a ProjectionFit -- the coordinates *and*, where the
    method has one, the fitted model.

    "umap_circle" runs UMAP with a *spherical* (haversine) output metric
    instead of the usual Euclidean one, then projects the result onto a
    flat disk -- ArcFace embeddings are L2-normalized (they live on a
    hypersphere, which is exactly the geometry the angular margin loss
    optimizes), so proximity here means angular closeness on that sphere
    rather than an arbitrary Euclidean layout. "umap" is the same algorithm
    with its default Euclidean output metric.
    """
    n = embeddings.shape[0]
    has_protos = proto_vectors is not None and len(proto_vectors)
    combined = np.concatenate([embeddings, proto_vectors], axis=0) if has_protos else embeddings

    model = None
    if method == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=2, random_state=seed)
        proj = model.fit_transform(combined)
    elif method == "umap":
        import umap

        model = umap.UMAP(n_components=2, random_state=seed)
        proj = model.fit_transform(combined)
    elif method == "umap_circle":
        import umap

        model = umap.UMAP(n_components=2, output_metric="haversine", random_state=seed)
        # raw columns are (colatitude, longitude) in radians; the disk
        # mapping is applied here and, identically, to anything placed onto
        # this fit later -- see ProjectionFit.transform.
        proj = _haversine_to_disk(np.asarray(model.fit_transform(combined), dtype=float))
    else:
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, (len(combined) - 1) // 3))
        # No model kept: TSNE has no transform() to keep one for.
        proj = TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity).fit_transform(combined)

    proj = np.asarray(proj, dtype=float)
    return ProjectionFit(
        method=method, points=proj[:n], proto_points=proj[n:] if has_protos else None, model=model,
    )


def project_2d(
    embeddings: np.ndarray, method: str = "tsne", proto_vectors: Optional[np.ndarray] = None, seed: int = 0
):
    """(points_2d, proto_points_2d | None) -- fit_projection for callers
    that only want the coordinates."""
    fit = fit_projection(embeddings, method=method, proto_vectors=proto_vectors, seed=seed)
    return fit.points, fit.proto_points


# ---- similarity search --------------------------------------------------

@dataclass
class Neighbour:
    uuid: str
    similarity: float
    label: Optional[str] = None


def nearest_neighbours(
    embeddings: np.ndarray, uuids: np.ndarray, query_uuid: str, k: int = 50,
    labels: Optional[np.ndarray] = None,
) -> list:
    """The k crops closest to `query_uuid` in embedding space, nearest first.

    This is the gesture nobody can do today: standing in front of a strange
    error and asking "what else looks like this to the model?". The answer
    separates the two explanations that matter -- a lone mislabelled crop,
    or a whole pocket of them the model has learned as a cluster.

    Cosine similarity, on vectors already L2-normalized at load time (see
    Experiment.load_embeddings), so it is one matrix-vector product. The
    query itself is excluded: it is trivially its own nearest neighbour and
    would waste a row.
    """
    if embeddings is None or not len(embeddings):
        return []
    uuids = np.asarray(uuids).astype(str)
    matches = np.flatnonzero(uuids == str(query_uuid))
    if not len(matches):
        return []
    return nearest_to_vector(
        embeddings, uuids, embeddings[matches[0]], k=k, labels=labels, exclude=int(matches[0]),
    )


def nearest_to_vector(
    embeddings: np.ndarray, uuids: np.ndarray, query: np.ndarray, k: int = 50,
    labels: Optional[np.ndarray] = None, exclude: Optional[int] = None,
) -> list:
    """The same search, from a vector rather than from a uuid in the pool.

    The pool being searched and the thing being searched *for* do not have
    to come from the same place: a crop loaded as a simple inference (see
    InferenceSet) has an embedding but no row in the dataset, and "what in
    the dataset looks like this?" is the whole reason to put it on the map.
    `exclude` drops one row of the pool -- the query itself, when it is in
    there, since a crop is trivially its own nearest neighbour.
    """
    if embeddings is None or not len(embeddings):
        return []
    uuids = np.asarray(uuids).astype(str)
    similarity = embeddings @ np.asarray(query, dtype=embeddings.dtype)
    limit = len(similarity)
    if exclude is not None:
        similarity = similarity.copy()
        similarity[exclude] = -np.inf
        limit -= 1

    k = min(max(int(k), 0), limit)
    if k <= 0:
        return []
    # argpartition first: at 500k embeddings a full sort costs far more than
    # the k values actually wanted.
    top = np.argpartition(-similarity, k - 1)[:k]
    top = top[np.argsort(-similarity[top])]
    return [
        Neighbour(
            uuid=str(uuids[i]), similarity=float(similarity[i]),
            label=str(labels[i]) if labels is not None else None,
        )
        for i in top if np.isfinite(similarity[i])
    ]


# ---- how much two classes overlap --------------------------------------

# How many crops of each class the pairwise similarity is measured over.
# Named rather than left as a default argument because the manual's
# "Class overlap matrix" page quotes it when it explains the numbers (see
# core/manual/pages_embeddings.py, which reads this constant rather than
# repeating the figure), and a number that drifted from the one actually
# used would be worse than no number.
OVERLAP_MAX_PER_CLASS = 400

@dataclass
class ClassOverlap:
    """Mean cosine similarity between every pair of classes, as embedded.

    Answers a question the confusion matrix cannot: *why* two classes get
    confused. A confusion says the model mixes them up; this says whether
    they actually sit on top of each other in feature space. Two classes
    that overlap heavily are a taxonomy problem (or a labelling one) --
    more training on the same definitions will not separate them.

    Computed from the crops themselves rather than from prototypes, so it
    describes the pool in front of the user and works without a prototype
    sidecar.
    """
    labels: list
    matrix: np.ndarray      # (C, C) mean pairwise cosine similarity
    counts: np.ndarray      # (C,) crops per class
    separation: np.ndarray  # (C,) own-class mean minus best other-class mean

    def top_pairs(self, n: int | None = 10) -> list:
        """The n most-overlapping distinct class pairs, worst first (all of
        them with n=None)."""
        rows, cols = np.triu_indices(len(self.labels), k=1)
        values = self.matrix[rows, cols]
        finite = np.isfinite(values)
        rows, cols, values = rows[finite], cols[finite], values[finite]
        order = np.argsort(-values, kind="stable")
        if n is not None:
            order = order[:n]
        return [(self.labels[rows[k]], self.labels[cols[k]], float(values[k])) for k in order]


def class_overlap(
    embeddings: np.ndarray, labels: np.ndarray, class_order: Optional[list] = None,
    max_per_class: int = OVERLAP_MAX_PER_CLASS, seed: int = 0,
) -> ClassOverlap:
    """Mean cosine similarity within and between classes.

    Each class is subsampled to `max_per_class` crops. The full pairwise
    computation is O(n^2) in the class size and would be minutes at
    half a million rows for an answer that a few hundred samples per class
    already gives to two decimal places -- the mean of a similarity
    distribution converges long before the sample runs out.
    """
    labels = np.asarray(labels).astype(str)
    present = list(class_order) if class_order else sorted(set(labels.tolist()))
    present = [l for l in present if (labels == l).sum() > 0]

    rng = np.random.default_rng(seed)
    sampled: dict = {}
    counts = []
    for label in present:
        idx = np.flatnonzero(labels == label)
        counts.append(len(idx))
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        sampled[label] = embeddings[idx]

    # The mean of every pairwise dot product between two sets is the dot
    # product of their sums divided by the number of pairs, so the whole
    # matrix is one (C x D) @ (D x C) product rather than C^2 blocks of
    # (400 x D) @ (D x 400) -- the same numbers, hours sooner at a few
    # thousand classes. Own-class similarity excludes each crop against
    # itself: |sum|^2 counts every pair twice plus each crop's squared norm.
    n = len(present)
    matrix = np.full((n, n), np.nan, dtype=float)
    if n:
        dim = embeddings.shape[1]
        sums = np.stack([sampled[l].sum(axis=0) if len(sampled[l]) else np.zeros(dim)
                         for l in present]).astype(np.float64)
        sizes = np.array([len(sampled[l]) for l in present], dtype=np.float64)
        self_norms = np.array([float((sampled[l].astype(np.float64) ** 2).sum()) for l in present])
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = (sums @ sums.T) / np.outer(sizes, sizes)
            own = (np.einsum("ij,ij->i", sums, sums) - self_norms) / (sizes * (sizes - 1))
        matrix[np.outer(sizes, sizes) == 0] = np.nan
        np.fill_diagonal(matrix, np.where(sizes >= 2, own, np.nan))

    others = matrix.copy()
    np.fill_diagonal(others, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a class with no finite neighbour
        best_other = np.nanmax(others, axis=1) if n else np.array([])
    diag = np.diag(matrix).copy() if n else np.full(0, np.nan)
    separation = np.where(np.isfinite(best_other) & np.isfinite(diag), diag - best_other, np.nan)

    return ClassOverlap(labels=present, matrix=matrix,
                        counts=np.array(counts), separation=separation)
