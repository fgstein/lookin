"""Shared context-building helper for the Embeddings and Cleaning tabs --
factored out so the merge/uuid-alignment/dedup logic (and the "sidecar
covers only some rows" warning) lives in exactly one place instead of being
copy-pasted between the two views.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def experiments_with_embeddings(state) -> list:
    return [name for name, exp in state.experiments.items() if exp.has_embeddings()]


def experiments_with_model_prototypes(state) -> list:
    return [
        name for name, exp in state.experiments.items()
        if exp.has_embeddings() and exp.model_prototypes is not None
    ]


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


class RowLookup:
    """uuid -> that row of the analysis frame as a dict, built when asked.

    Stands in for the {uuid: record} dict this used to be: the carousel and
    the similarity panel look up a page of rows at a time, while building
    the whole dict up front turned every row into a dict of every column --
    two thousand score columns included, which at a hundred thousand rows
    was half a minute and most of 12 GB. The score columns are left out of
    the records: each would be one more line of metadata under a crop.

    Duplicated uuids resolve to their last row, as the dict did (uuids are
    not guaranteed unique; see UuidOverlap).
    """

    def __init__(self, frame, uuids: pd.Series, skip_columns=()):
        skip = set(skip_columns)
        # One array per kept column rather than frame.iloc[i]: a map asks
        # for every plotted point at once, and a row lookup on a frame two
        # thousand columns wide costs a Series of two thousand values.
        self._arrays = {c: frame[c].to_numpy() for c in frame.columns if c not in skip}
        self._position = {u: i for i, u in enumerate(uuids)}

    def __contains__(self, uuid) -> bool:
        return uuid in self._position

    def __len__(self) -> int:
        return len(self._position)

    def __getitem__(self, uuid) -> dict:
        i = self._position[uuid]
        # Python scalars, as DataFrame.to_dict gave: the carousel formats
        # a value by its type, and np.float32 is not a float.
        return {c: _plain(values[i]) for c, values in self._arrays.items()}

    def get(self, uuid, default=None):
        return self[uuid] if uuid in self._position else default


def build_analysis_context(state, exp_name: str):
    """merged rows restricted to those that actually have an embedding,
    plus the aligned embeddings/labels/uuids arrays -- every analysis
    function operates on this filtered subset, index-aligned throughout.
    Returns None if the experiment has no embeddings attached.

    Built once and reused until the merged frame or the embeddings change:
    the tabs ask for it on every render, and it is not cheap at scale.
    """
    exp = state.experiments[exp_name]
    merged = state.merged(exp_name)
    cache = state.__dict__.setdefault("_analysis_context_cache", {})
    hit = cache.get(exp_name)
    if hit is not None and hit[0] is merged and hit[1] is exp.embeddings:
        return hit[2]

    uuid_col = state.dataset.columns.uuid_col
    label_col = state.dataset.columns.label_col
    embeddings, mask = exp.aligned_embeddings(merged, uuid_col)
    if embeddings is None:
        cache.pop(exp_name, None)
        return None
    # No copy when every row has an embedding -- the usual case, and the
    # frame is two thousand score columns wide at that scale.
    sub = merged.reset_index(drop=True) if mask.all() else merged[mask].reset_index(drop=True)
    sub_uuids = sub[uuid_col].astype(str)
    context = {
        "exp": exp,
        "merged": sub,
        "row_by_uuid": RowLookup(sub, sub_uuids, skip_columns=exp.score_cols),
        "embeddings": embeddings if mask.all() else embeddings[mask],
        "labels": sub[label_col].to_numpy(),
        "uuids": sub_uuids.to_numpy(),
        "uuid_col": uuid_col,
        "crop_col": state.dataset.columns.crop_col,
    }
    cache[exp_name] = (merged, exp.embeddings, context)
    return context


def coverage_note(ctx: dict, state, exp_name: str) -> str:
    """Empty string if the embeddings sidecar covers every merged row,
    otherwise a note explaining how many rows are actually usable."""
    total_rows = len(state.merged(exp_name))
    covered = len(ctx["embeddings"])
    if covered >= total_rows:
        return ""
    return f"The embeddings sidecar covers {covered:,}/{total_rows:,} rows for '{exp_name}' -- analysis below only considers those."
