"""Small machine-local preferences: where the MLflow server is, and where
a reviewer had got to in a triage queue.

Everything here is stored as JSON under the user's config directory with the
stdlib only -- no network, nothing fetched, nothing new in the wheel. These
files are machine-local by nature and are never packaged.

Nothing here is *data*. Losing any of it costs the user a re-typed URI or a
scroll back to where they were, never a decision or a result -- which is why
every read treats an unreadable file as "no preference" rather than an
error, and none of it travels with an exported workspace.

The saved-session history that used to live here is gone: the library
(core/library.py) does that job properly now, with names, tags and views
instead of a ten-entry ring buffer. An old recents.json is simply left on
disk, unread.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def config_dir() -> Path:
    """XDG config home, falling back to ~/.config."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".config"
    return root / "lookin"


# ---------------------------------------------------------------------------
# MLflow connection preference
#
# Kept here rather than inside a workspace's mount: a tracking
# URI belongs to the *machine*, not to one dataset mount, and the same URI is
# reused whichever dataset is open. Same rules as the sessions file above --
# stdlib only, machine-local, and an unreadable file just means "no
# preference", never an error.
# ---------------------------------------------------------------------------

MLFLOW_PREF_FILE = "mlflow.json"


MAX_MLFLOW_PICKS = 5


@dataclass
class MlflowPick:
    """One complete "what I loaded last time" triple. Keeping the run and
    artifact alongside the experiment is what lets the browse dialog offer a
    single-click shortcut back to a CSV: an experiment id alone would still
    leave the user hunting through a list of near-identical run names."""
    experiment_id: str = ""
    experiment_name: str = ""
    run_id: str = ""
    run_name: str = ""
    artifact_path: str = ""

    def label(self) -> str:
        return self.run_name or self.run_id[:8] or self.experiment_name


@dataclass
class MlflowPref:
    tracking_uri: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    recent: list = field(default_factory=list)  # list[MlflowPick], most recent first


def mlflow_pref_path() -> Path:
    return config_dir() / MLFLOW_PREF_FILE


def load_mlflow_pref() -> MlflowPref:
    path = mlflow_pref_path()
    if not path.is_file():
        return MlflowPref()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        _log(f"mlflow pref: ignoring unreadable {path} ({ex})")
        return MlflowPref()
    if not isinstance(raw, dict):
        return MlflowPref()
    picks = [
        MlflowPick(
            experiment_id=str(e.get("experiment_id", "")),
            experiment_name=str(e.get("experiment_name", "")),
            run_id=str(e.get("run_id", "")),
            run_name=str(e.get("run_name", "")),
            artifact_path=str(e.get("artifact_path", "")),
        )
        for e in raw.get("recent", []) or []
        if isinstance(e, dict)
    ]
    return MlflowPref(
        tracking_uri=str(raw.get("tracking_uri", "")),
        experiment_id=str(raw.get("experiment_id", "")),
        experiment_name=str(raw.get("experiment_name", "")),
        recent=picks[:MAX_MLFLOW_PICKS],
    )


def save_mlflow_pref(pref: MlflowPref) -> bool:
    path = mlflow_pref_path()
    payload = {k: v for k, v in asdict(pref).items() if k != "recent"}
    payload["recent"] = [asdict(pick) for pick in pref.recent[:MAX_MLFLOW_PICKS]]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as ex:
        _log(f"mlflow pref: could not write {path} ({ex})")
        return False


def remember_mlflow_pick(pref: MlflowPref, pick: MlflowPick) -> None:
    """Push a pick to the front of pref.recent, de-duplicated on the exact
    run + artifact it points at. Mutates `pref`; the caller saves."""
    if not pick.run_id or not pick.artifact_path:
        return
    others = [p for p in pref.recent
              if not (p.run_id == pick.run_id and p.artifact_path == pick.artifact_path)]
    pref.recent = [pick] + others[:MAX_MLFLOW_PICKS - 1]


# ---------------------------------------------------------------------------
# Triage cursor
#
# Where the reviewer had got to in a queue, per (dataset, version). A queue
# of two thousand flagged crops is not finished in one sitting, and coming
# back the next morning to start again from crop 1 -- scrolling past four
# hundred already-decided ones -- is how a cleaning pass gets abandoned.
#
# Deliberately not stored in the review sidecars: this is where the user
# was looking, not something they decided. It must never end up in an
# exported dataset, and losing it costs nothing.
# ---------------------------------------------------------------------------

TRIAGE_FILE = "triage.json"
MAX_TRIAGE_CURSORS = 50


def triage_path() -> Path:
    return config_dir() / TRIAGE_FILE


def _triage_key(dataset_path, version_id: str) -> str:
    return f"{Path(dataset_path).resolve() if dataset_path else ''}::{version_id}"


def _load_triage() -> dict:
    path = triage_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        _log(f"triage cursor: ignoring unreadable {path} ({ex})")
        return {}
    return raw if isinstance(raw, dict) else {}


def load_triage_cursor(dataset_path, version_id: str) -> str:
    """The uuid the reviewer was on, or "" if this queue is new. The *uuid*
    rather than an index: the candidate list is recomputed every time it is
    opened, so position 412 is not stable across sessions but the crop is."""
    if not version_id:
        return ""
    return str(_load_triage().get(_triage_key(dataset_path, version_id), ""))


def save_triage_cursor(dataset_path, version_id: str, uuid: str) -> bool:
    if not version_id:
        return False
    cursors = _load_triage()
    cursors[_triage_key(dataset_path, version_id)] = str(uuid)
    # Newest last; drop the oldest when there are too many, since a stale
    # cursor for a dataset nobody has opened in months costs more confusion
    # than it saves.
    if len(cursors) > MAX_TRIAGE_CURSORS:
        for key in list(cursors)[: len(cursors) - MAX_TRIAGE_CURSORS]:
            cursors.pop(key, None)
    path = triage_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(json.dumps(cursors, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as ex:
        _log(f"triage cursor: could not write {path} ({ex})")
        return False


def clear_triage_cursor(dataset_path, version_id: str) -> bool:
    cursors = _load_triage()
    if cursors.pop(_triage_key(dataset_path, version_id), None) is None:
        return False
    try:
        triage_path().write_text(json.dumps(cursors, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False
