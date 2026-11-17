"""Saved views of the Embeddings and Compare embeddings tabs.

A projection, and for a comparison its measures, take from seconds to
minutes to compute; looking at them again should not. A *view* is one of
those results, kept: the settings that produced it (what the tab's own
view_state holds) and what was computed -- the position of every crop on
the map, the prototypes, and for a comparison its measures. Opening a view
redraws it at once; nothing is recomputed.

What is frozen, and what is not
-------------------------------
Only what was computed: positions and measures, keyed by crop uuid. The
labels, the predictions, the colors, the display filters are read from the
data as it is when the view is opened, so a relabel since shows -- and a
view whose files changed since it was saved is said to be out of date
(file_stamp), its crops that are gone left off the map.

A PCA is kept with the positions (its components and mean, a few
kilobytes): crops can still be placed onto a saved PCA view. A UMAP model
holds its whole training set and is not kept -- a saved UMAP or t-SNE view
is positions only, and says so: placing more crops means running the
projection again.

Where
-----
Saved views live in the workspace's folder (views/: one index, one .npz
per view), so they survive a restart and anyone who opens the workspace
sees them. The tabs also keep the last HISTORY_MAX runs of the session in
memory, each one a click away from being saved.

Keys stored in the index (never reworded): TAB_EMBEDDINGS, TAB_COMPARE,
and the fields of view_to_json.
"""
from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from lookin.core import fileio

TAB_EMBEDDINGS = "embeddings"
TAB_COMPARE = "embcompare"
TABS = (TAB_EMBEDDINGS, TAB_COMPARE)
VIEWS_DIRNAME = "views"
INDEX_NAME = "views.json"
ID_PREFIX = "lv-"
# Runs of the session kept in memory, per tab.
HISTORY_MAX = 10
NAME_MAX = 120

_lock = threading.Lock()


@dataclass
class SavedView:
    id: str
    tab: str
    name: str
    created_at: str
    summary: str = ""
    settings: dict = field(default_factory=dict)
    stamp: list = field(default_factory=list)       # [[path, size, mtime], ...] of the data it was computed on
    has_model: bool = False                         # crops can be placed onto it (a PCA)
    arrays: dict = field(default_factory=dict)      # name -> ndarray; empty until load_arrays()
    saved: bool = False                             # in the workspace (else: this session's history)


def new_id() -> str:
    return ID_PREFIX + secrets.token_hex(4)


def make_view(tab: str, name: str, settings: dict, arrays: dict, stamp: list, summary: str = "") -> SavedView:
    if tab not in TABS:
        raise ValueError(f"Unknown tab {tab!r}.")
    return SavedView(id=new_id(), tab=tab, name=(name or "View")[:NAME_MAX],
                     created_at=datetime.now().isoformat(timespec="seconds"), summary=summary,
                     settings=dict(settings), stamp=list(stamp), arrays=dict(arrays),
                     has_model=any(k.startswith(MODEL_PREFIX) for k in arrays))


def view_to_json(view: SavedView) -> dict:
    return {"id": view.id, "tab": view.tab, "name": view.name, "created_at": view.created_at,
            "summary": view.summary, "settings": view.settings, "stamp": view.stamp, "has_model": view.has_model}


def view_from_json(raw: dict) -> Optional[SavedView]:
    if not isinstance(raw, dict) or raw.get("tab") not in TABS or not str(raw.get("id", "")).startswith(ID_PREFIX):
        return None
    return SavedView(id=str(raw["id"]), tab=raw["tab"], name=str(raw.get("name", ""))[:NAME_MAX],
                     created_at=str(raw.get("created_at", "")), summary=str(raw.get("summary", "")),
                     settings=raw.get("settings") if isinstance(raw.get("settings"), dict) else {},
                     stamp=raw.get("stamp") if isinstance(raw.get("stamp"), list) else [],
                     has_model=bool(raw.get("has_model")), saved=True)


# ---- the data a view was computed on ---------------------------------------------

def file_stamp(paths) -> list:
    """[[path, size, mtime], ...] of the files a result was computed from."""
    out = []
    for p in sorted({str(p) for p in paths if p}):
        try:
            st = Path(p).stat()
            out.append([p, int(st.st_size), round(float(st.st_mtime), 3)])
        except OSError:
            out.append([p, 0, 0.0])
    return out


def changed_files(view: SavedView) -> list:
    """The files the view was computed on that changed (or went) since."""
    then = {str(p): (size, mtime) for p, size, mtime in (tuple(x) for x in view.stamp if len(x) == 3)}
    now = {p: (size, mtime) for p, size, mtime in file_stamp(then)}
    return [p for p in then if now.get(p) != then[p]]


# ---- a PCA, kept with the positions ------------------------------------------------

MODEL_PREFIX = "model_"


class LinearProjection:
    """What a fitted PCA does to new points -- enough to place crops onto a
    saved view (ProjectionFit.transform), without pickling scikit-learn."""

    def __init__(self, components: np.ndarray, mean: np.ndarray):
        self.components = np.asarray(components, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mean) @ self.components.T


def model_arrays(fit) -> dict:
    """The arrays that keep a fit's model, when it is a PCA; else none."""
    model = getattr(fit, "model", None)
    if fit is None or getattr(fit, "method", "") != "pca" or model is None:
        return {}
    if isinstance(model, LinearProjection):
        return {MODEL_PREFIX + "components": model.components, MODEL_PREFIX + "mean": model.mean}
    components, mean = getattr(model, "components_", None), getattr(model, "mean_", None)
    if components is None or mean is None or np.asarray(components).shape[0] < 2:
        return {}
    return {MODEL_PREFIX + "components": np.asarray(components), MODEL_PREFIX + "mean": np.asarray(mean)}


def fit_from_arrays(method: str, points: np.ndarray, arrays: dict, proto_points=None):
    """A ProjectionFit standing for the saved one: its points, and its model
    when one was kept (a PCA) -- None otherwise, so placing more crops says
    the projection must be run again."""
    from lookin.core.embedding_analysis import ProjectionFit

    model = None
    if MODEL_PREFIX + "components" in arrays and MODEL_PREFIX + "mean" in arrays:
        model = LinearProjection(arrays[MODEL_PREFIX + "components"], arrays[MODEL_PREFIX + "mean"])
    return ProjectionFit(method=method, points=np.asarray(points), proto_points=proto_points, model=model)


# ---- the workspace's catalogue ------------------------------------------------------

def views_dir(workspace) -> Path:
    return Path(workspace.path) / VIEWS_DIRNAME


def _read_index(workspace) -> list:
    path = views_dir(workspace) / INDEX_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return []
    items = raw.get("views") if isinstance(raw, dict) else None
    return [v for v in (view_from_json(x) for x in (items or [])) if v is not None]


def _write_index(workspace, views: list) -> None:
    folder = views_dir(workspace)
    folder.mkdir(parents=True, exist_ok=True)
    fileio.atomic_write_text(folder / INDEX_NAME,
                             json.dumps({"version": 1, "views": [view_to_json(v) for v in views]}, indent=2))


def list_views(workspace, tab: str = "") -> list:
    """The workspace's saved views, newest first (arrays not loaded)."""
    if workspace is None or not getattr(workspace, "path", None):
        return []
    views = _read_index(workspace)
    if tab:
        views = [v for v in views if v.tab == tab]
    return sorted(views, key=lambda v: v.created_at, reverse=True)


def save_view(workspace, view: SavedView) -> SavedView:
    """Write `view` (with its arrays) into the workspace's catalogue."""
    if workspace is None or not getattr(workspace, "path", None):
        raise ValueError("Views are saved in a workspace of the library; this session is not one.")
    folder = views_dir(workspace)
    folder.mkdir(parents=True, exist_ok=True)
    with _lock, fileio.locked(folder / INDEX_NAME):
        tmp = folder / f".{view.id}.tmp.npz"
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in view.arrays.items()})
        tmp.replace(folder / f"{view.id}.npz")
        views = [v for v in _read_index(workspace) if v.id != view.id]
        view.saved = True
        _write_index(workspace, views + [view])
    return view


def load_arrays(workspace, view: SavedView) -> dict:
    """The arrays of a saved view (from its .npz)."""
    path = views_dir(workspace) / f"{view.id}.npz"
    if not path.is_file():
        raise ValueError(f"The saved view {view.name!r} has lost its data file.")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def find_view(workspace, view_id: str) -> SavedView:
    for v in list_views(workspace):
        if v.id == view_id:
            return v
    raise ValueError(f"No saved view {view_id!r} in this workspace.")


def rename_view(workspace, view_id: str, name: str) -> SavedView:
    name = (name or "").strip()[:NAME_MAX]
    if not name:
        raise ValueError("A view needs a name.")
    folder = views_dir(workspace)
    with _lock, fileio.locked(folder / INDEX_NAME):
        views = _read_index(workspace)
        found = next((v for v in views if v.id == view_id), None)
        if found is None:
            raise ValueError(f"No saved view {view_id!r} in this workspace.")
        found.name = name
        _write_index(workspace, views)
    return found


def delete_view(workspace, view_id: str) -> None:
    folder = views_dir(workspace)
    with _lock, fileio.locked(folder / INDEX_NAME):
        views = _read_index(workspace)
        if not any(v.id == view_id for v in views):
            raise ValueError(f"No saved view {view_id!r} in this workspace.")
        _write_index(workspace, [v for v in views if v.id != view_id])
        try:
            (folder / f"{view_id}.npz").unlink()
        except OSError:
            pass


def push_history(history: list, view: SavedView) -> list:
    """This session's runs, newest first, HISTORY_MAX at most."""
    return ([view] + [v for v in history if v.id != view.id])[:HISTORY_MAX]
