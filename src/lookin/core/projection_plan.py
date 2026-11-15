"""Which crops a projection is fitted on, and which it shows.

Two different sets, on purpose
------------------------------
Fitting a projection is the slow part, and it does not get better past a
few thousand points: PCA's axes and UMAP's layout from 3,000 crops are,
to the eye, those from 300,000. Showing crops is cheap by comparison: PCA
places a new point with one matrix product, UMAP with a nearest-neighbour
search against the points it was fitted on. So a map is fitted on a
modest *fit set* and then shows a larger *display set*, every point of
which is placed onto that fit (ProjectionFit.transform). t-SNE has no way
to place a point it was not fitted on, so with it the two sets are one.

How the fit set is drawn
------------------------
* SAMPLING_REAL (the default): at random, so each class keeps its share
  of the data -- the map shows the data as it is.
* SAMPLING_BALANCED: the same number of crops from every class (all of
  them for a class that has fewer) -- a rare class gets as much say in
  the layout as a frequent one, which is what makes it readable.
The display set is drawn at random, after the fit set, which it always
contains: every point the layout was built from is on the map.

A plan is saved in the workspace (Workspace.embedding_projection for the
Embeddings tab, inside Workspace.embedding_comparison for Compare
embeddings) and in snapshots, as plan_to_json writes it. Its keys and
SAMPLING_* values are a stored format, never reworded.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

SAMPLING_REAL = "real"
SAMPLING_BALANCED = "balanced"
SAMPLINGS = (SAMPLING_REAL, SAMPLING_BALANCED)

DEFAULT_FIT_SIZE = 3000
DEFAULT_PER_CLASS = 100
DEFAULT_DISPLAY_SIZE = 20_000
MAX_FIT_SIZE = 50_000
MAX_PER_CLASS = 5_000
# What a browser draws smoothly as points (see static/js/projection.js).
MAX_DISPLAY_SIZE = 200_000
# Points placed onto a fit per batch: progress is reported, and a Cancel
# honoured, between batches.
PLACE_BATCH = 5_000
# Methods that can place points they were not fitted on.
TRANSFORM_METHODS = ("pca", "umap", "umap_circle")


@dataclass
class ProjectionPlan:
    method: str = "pca"
    sampling: str = SAMPLING_REAL
    fit_size: int = DEFAULT_FIT_SIZE        # SAMPLING_REAL: crops in the fit set
    per_class: int = DEFAULT_PER_CLASS      # SAMPLING_BALANCED: crops per class
    display_size: int = DEFAULT_DISPLAY_SIZE
    seed: int = 0

    def places_more(self) -> bool:
        """Whether points beyond the fit set can be shown at all."""
        return self.method in TRANSFORM_METHODS


def _int(value, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(int(float(value)), high))
    except (TypeError, ValueError):
        return default


def plan_from_json(raw, method_default: str = "pca") -> ProjectionPlan:
    raw = raw if isinstance(raw, dict) else {}
    sampling = raw.get("sampling") if raw.get("sampling") in SAMPLINGS else SAMPLING_REAL
    return ProjectionPlan(
        method=str(raw.get("method") or method_default),
        sampling=sampling,
        fit_size=_int(raw.get("fit_size"), DEFAULT_FIT_SIZE, 10, MAX_FIT_SIZE),
        per_class=_int(raw.get("per_class"), DEFAULT_PER_CLASS, 1, MAX_PER_CLASS),
        display_size=_int(raw.get("display_size"), DEFAULT_DISPLAY_SIZE, 10, MAX_DISPLAY_SIZE),
        seed=_int(raw.get("seed"), 0, 0, 2**31 - 1),
    )


def plan_to_json(plan: ProjectionPlan) -> dict:
    return {"method": plan.method, "sampling": plan.sampling, "fit_size": int(plan.fit_size),
            "per_class": int(plan.per_class), "display_size": int(plan.display_size), "seed": int(plan.seed)}


def clamp(plan: ProjectionPlan) -> ProjectionPlan:
    """The plan with every number within its bounds (and read from text
    when it was typed in)."""
    return plan_from_json(vars(plan), plan.method)


@dataclass
class PlanRows:
    """The rows (positions in the pool's arrays) of the two sets."""
    fit: np.ndarray                 # sorted
    display: np.ndarray             # sorted, contains `fit`
    fit_by_class: dict = field(default_factory=dict)       # label -> crops in the fit set
    display_by_class: dict = field(default_factory=dict)   # label -> crops shown
    notes: list = field(default_factory=list)


def _by_class(labels: np.ndarray, rows: np.ndarray) -> dict:
    names, counts = np.unique(np.asarray([str(x) for x in labels[rows]], dtype=object), return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts)}


def choose_rows(labels, plan: ProjectionPlan, pool: Optional[np.ndarray] = None) -> PlanRows:
    """The fit set and the display set, drawn from `pool` (rows of
    `labels`; all of them when None) as the plan says."""
    labels = np.asarray(labels, dtype=object)
    pool = np.arange(len(labels)) if pool is None else np.asarray(pool, dtype=np.int64)
    rng = np.random.default_rng(plan.seed)
    notes = []
    if plan.sampling == SAMPLING_BALANCED:
        parts, short = [], 0
        names = np.asarray([str(x) for x in labels[pool]], dtype=object)
        for name in sorted(set(names.tolist()), key=str):
            rows = pool[names == name]
            if len(rows) > plan.per_class:
                rows = rng.choice(rows, plan.per_class, replace=False)
            else:
                short += 1
            parts.append(rows)
        fit = np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
        if short:
            notes.append(f"{short} class(es) have fewer than {plan.per_class} crops: all of theirs are used.")
    else:
        size = min(plan.fit_size, len(pool))
        fit = np.sort(rng.choice(pool, size, replace=False)) if size < len(pool) else np.sort(pool)
    if not plan.places_more():
        display = fit
        if len(pool) > len(fit):
            notes.append("t-SNE cannot place crops it was not fitted on: the map shows the fit set only.")
    else:
        rest = np.setdiff1d(pool, fit, assume_unique=False)
        extra = max(0, min(plan.display_size, len(pool)) - len(fit))
        if extra < len(rest):
            rest = rng.choice(rest, extra, replace=False)
        display = np.sort(np.concatenate([fit, rest])) if len(rest) else fit
    return PlanRows(fit=fit, display=display, fit_by_class=_by_class(labels, fit),
                    display_by_class=_by_class(labels, display), notes=notes)


def place(fit, embeddings: np.ndarray, progress: Optional[Callable[[int, int], None]] = None,
          batch: int = PLACE_BATCH) -> np.ndarray:
    """`embeddings` placed onto a fitted projection (ProjectionFit), in
    batches; `progress(done, total)` is called before each one -- where a
    Cancel can stop it (it may raise)."""
    total = len(embeddings)
    out = np.zeros((total, 2), dtype=float)
    for start in range(0, total, batch):
        if progress is not None:
            progress(start, total)
        out[start:start + batch] = fit.transform(embeddings[start:start + batch])
    return out


def project(embeddings: np.ndarray, rows: PlanRows, method: str, proto_vectors=None, seed: int = 0,
            progress: Optional[Callable[[str], None]] = None) -> tuple:
    """(display coordinates, prototype coordinates, the fit): fitted on
    rows.fit, every row of rows.display placed -- those of the fit set at
    the coordinates the fit gave them, the others through transform.
    `progress(text)` reports each step."""
    from lookin.core.embedding_analysis import fit_projection

    if progress is not None:
        progress(f"Fitting {method} on {len(rows.fit):,} crops...")
    fit = fit_projection(embeddings[rows.fit], method=method, proto_vectors=proto_vectors, seed=seed)
    coords = np.zeros((len(rows.display), 2), dtype=float)
    in_fit = np.isin(rows.display, rows.fit)
    position = np.searchsorted(rows.fit, rows.display[in_fit])
    coords[in_fit] = fit.points[position]
    others = rows.display[~in_fit]
    if len(others):
        def step(done, total):
            if progress is not None:
                progress(f"Placing {total:,} more crops on the map... {100 * done // max(total, 1)}%")
        coords[~in_fit] = place(fit, embeddings[others], step)
    return coords, fit.proto_points, fit


def describe(rows: PlanRows, plan: ProjectionPlan) -> str:
    """One line: what the map was fitted on and what it shows."""
    fit_classes = rows.fit_by_class
    smallest = min(fit_classes.items(), key=lambda kv: kv[1]) if fit_classes else None
    how = ("as in the data" if plan.sampling == SAMPLING_REAL else f"balanced, up to {plan.per_class:,} per class")
    text = (f"Fitted on {len(rows.fit):,} crops of {len(fit_classes):,} classes ({how}"
            + (f"; smallest: {smallest[0]} with {smallest[1]:,}" if smallest else "") + ")")
    shown = len(rows.display)
    return text + (f"; {shown:,} shown." if shown != len(rows.fit) else "; the same crops shown.")


def with_method(plan: ProjectionPlan, method: str) -> ProjectionPlan:
    return replace(plan, method=method)
