"""Which dataset CSVs exist, for the Home page and the All datasets window.

There is no dataset registry in Lookin: a dataset is a CSV on disk. So the
list is gathered from what already points at datasets --

* every workspace's dataset,
* the dataset parameter of every listed MLflow run (home_view.json's
  "dataset_param"), which may name a file not on this machine,
* the CSVs found in folders the user asked to watch,

and the facts about each file (rows, columns, classes) are only read when
someone asks for them -- a catalogue that opened every CSV just to list it
would be the slowest page of the app.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lookin.core import workspace_sources as wss
from lookin.core.prefs import config_dir

WATCH_FILE = "web_datasets.json"
MAX_SCAN_FILES = 2000
MAX_SCAN_DEPTH = 3
# Files that sit beside datasets and are not datasets: results CSVs are
# experiments, and the review sidecars belong to the dataset next to them.
NOT_A_DATASET_SUFFIXES = ("_results.csv", "_review.csv", "_review_versions.csv")

_lock = threading.Lock()
_facts: dict = {}     # path -> (mtime_ns, size, Facts)


@dataclass
class Facts:
    rows: Optional[int] = None
    columns: list = field(default_factory=list)
    classes: Optional[int] = None
    uuid_col: str = ""
    crop_col: str = ""
    label_col: str = ""
    note: str = ""
    error: str = ""


@dataclass
class DatasetEntry:
    path: str
    name: str
    folder: str
    here: bool
    size: Optional[int] = None
    mtime: Optional[float] = None
    used_by: list = field(default_factory=list)       # workspaces
    named_by: list = field(default_factory=list)      # (server name, experiment, run label)
    watched: bool = False

    def facts(self) -> Optional[Facts]:
        return cached_facts(self.path)


def _watch_path() -> Path:
    return config_dir() / WATCH_FILE


def watched_folders() -> list:
    try:
        raw = json.loads(_watch_path().read_text(encoding="utf-8"))
        return [str(f) for f in raw.get("folders", []) if isinstance(f, str)]
    except (OSError, ValueError, AttributeError):
        return []


def _save_folders(folders: list) -> bool:
    path = _watch_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"folders": folders}, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def add_watched_folder(folder: str) -> bool:
    folders = watched_folders()
    folder = str(Path(folder).expanduser())
    if folder in folders:
        return False
    return _save_folders(folders + [folder])


def remove_watched_folder(folder: str) -> bool:
    return _save_folders([f for f in watched_folders() if f != folder])


def _scan(folder: str) -> list:
    out = []
    root = Path(folder)
    if not root.is_dir():
        return out
    stack = [(root, 0)]
    while stack and len(out) < MAX_SCAN_FILES:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_dir(follow_symlinks=False):
                if depth < MAX_SCAN_DEPTH:
                    stack.append((Path(e.path), depth + 1))
            elif e.name.lower().endswith(".csv") and not e.name.lower().endswith(NOT_A_DATASET_SUFFIXES):
                out.append(e.path)
    return out


def catalog(workspaces, hub=None) -> list:
    entries: dict = {}

    def entry(path: str) -> DatasetEntry:
        key = str(path)
        if key not in entries:
            p = Path(key)
            here = False
            size = mtime = None
            try:
                st = p.stat()
                here = p.is_file()
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                pass
            entries[key] = DatasetEntry(path=key, name=p.name or key, folder=str(p.parent) if here else "",
                                        here=here, size=size, mtime=mtime)
        return entries[key]

    for ws in workspaces:
        if ws.mount.dataset_path:
            e = entry(ws.mount.dataset_path)
            if ws not in e.used_by:
                e.used_by.append(ws)
    if hub is not None:
        param = hub.config.dataset_param
        for server in list(hub.servers.values()):
            names = {x.experiment_id: x.name for x in server.experiments}
            for run in list(server.runs):
                p = wss.dataset_of_run(run, param)
                if p:
                    entry(p).named_by.append((server.name, names.get(run.experiment_id, run.experiment_id), run.label()))
    for folder in watched_folders():
        for p in _scan(folder):
            entry(p).watched = True
    return sorted(entries.values(), key=lambda d: (-(d.mtime or 0), d.name.lower()))


def cached_facts(path: str) -> Optional[Facts]:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    with _lock:
        hit = _facts.get(path)
    if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    return None


def read_facts(path: str) -> Facts:
    """Columns and roles from the head of the file, then the row count and
    the number of classes from a pass over the label column only."""
    from lookin.core.data_manager import _count_data_rows
    from lookin.core.schema import sniff_csv

    facts = Facts()
    try:
        st = Path(path).stat()
        guess = sniff_csv(path)
        facts.columns = list(guess.columns)
        facts.uuid_col, facts.crop_col, facts.label_col = guess.uuid_col or "", guess.crop_col or "", guess.label_col or ""
        facts.note = guess.explain()
        facts.rows = _count_data_rows(path)
        if facts.label_col:
            import pandas as pd
            # Only an empty cell is a missing label ("None" can be a class).
            labels = pd.read_csv(path, usecols=[facts.label_col], keep_default_na=False,
                                 na_values={facts.label_col: [""]})[facts.label_col]
            facts.classes = int(labels.nunique(dropna=True))
    except Exception as ex:  # noqa: BLE001
        facts.error = str(ex)
        return facts
    with _lock:
        _facts[path] = (st.st_mtime_ns, st.st_size, facts)
    return facts


def sample_crops(path: str, facts: Facts, n: int = 6) -> list:
    """(absolute crop path, label) for the first few rows."""
    if not facts.crop_col:
        return []
    try:
        import pandas as pd
        cols = [c for c in (facts.crop_col, facts.label_col) if c]
        df = pd.read_csv(path, usecols=cols, nrows=200)
    except Exception:  # noqa: BLE001
        return []
    base = Path(path).resolve().parent
    out = []
    for _, row in df.head(n).iterrows():
        p = Path(str(row[facts.crop_col]))
        out.append((str(p if p.is_absolute() else base / p), str(row[facts.label_col]) if facts.label_col else ""))
    return out
