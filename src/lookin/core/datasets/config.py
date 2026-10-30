"""Where the datasets are and how to read them: one JSON file per user,
written by the Datasets page's settings.

    ~/.config/lookin/datasets.json

    {
      "folders": ["/shared/datasets"],
      "exclude": ["*/mappings/*", "mappings/*"],
      "mlflow": [{"uri": "http://mlflow.lab:5000", "experiment": "datasets"}],
      "columns": {"id": "uuid", "image": "crop", "split": "split", "source": "source",
                  "mappings": ["mapped_class_label", "fine_label"]},
      "overrides": {"/shared/datasets/old/old.csv": {"mappings": ["label"]}}
    }

`folders` are scanned for CSVs (see catalog.py), except the ones whose path
-- relative to their folder -- matches a pattern of `exclude` (fnmatch,
where * also crosses "/"). `mlflow` names the experiments whose runs log
the datasets (see mlflow_link.py).

`columns` says which column is what, for every dataset: nothing is guessed
from the values. `mappings` are the label columns -- one dataset can carry
several labellings of the same rows, and the statistics of the labels are
computed for each of them that the file has. `overrides` replace some of
those keys for one dataset (by absolute path); a key it does not name
comes from `columns`. A column a file does not have is simply absent.

Per user, like home_view.json: two people can look at different folders.
The folders used to be the "watched folders" of the All datasets window,
kept in web_datasets.json; that file is read once, the first time this one
does not exist yet.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

from lookin.core import fileio
from lookin.core.prefs import config_dir
from lookin.core.workspace_sources import normalize_uri

CONFIG_FILE = "datasets.json"
LEGACY_WATCH_FILE = "web_datasets.json"

# Column roles: keys of `columns` and of an override. Compared in code.
ROLE_ID = "id"
ROLE_IMAGE = "image"
ROLE_SPLIT = "split"
ROLE_SOURCE = "source"
ROLE_MAPPINGS = "mappings"
SINGLE_ROLES = (ROLE_ID, ROLE_IMAGE, ROLE_SPLIT, ROLE_SOURCE)

DEFAULT_COLUMNS = {ROLE_ID: "uuid", ROLE_IMAGE: "crop", ROLE_SPLIT: "split", ROLE_SOURCE: "source",
                   ROLE_MAPPINGS: ["mapped_class_label"]}
# Label mapping tables kept beside the datasets: CSVs, but not datasets.
DEFAULT_EXCLUDE = ["*/mappings/*", "mappings/*"]


@dataclass
class MlflowSource:
    uri: str
    experiment: str

    def key(self) -> tuple:
        return (self.uri, self.experiment)


def _columns(raw, base: dict = None) -> dict:
    """A columns dict (or an override) cleaned up: known keys only, names
    stripped, mappings a list without duplicates. With `base`, keys missing
    from `raw` are left out (an override); without, they take the defaults."""
    out = {} if base is not None else {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_COLUMNS.items()}
    if not isinstance(raw, dict):
        return out
    for key in SINGLE_ROLES:
        if isinstance(raw.get(key), str):
            out[key] = raw[key].strip()
    if isinstance(raw.get(ROLE_MAPPINGS), list):
        out[ROLE_MAPPINGS] = list(dict.fromkeys(str(m).strip() for m in raw[ROLE_MAPPINGS] if str(m).strip()))
    return out


@dataclass
class DatasetsConfig:
    folders: list = field(default_factory=list)
    exclude: list = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    mlflow: list = field(default_factory=list)      # MlflowSource
    columns: dict = field(default_factory=lambda: _columns({}))
    overrides: dict = field(default_factory=dict)   # absolute path -> partial columns

    def to_dict(self) -> dict:
        return {"folders": list(self.folders), "exclude": list(self.exclude),
                "mlflow": [{"uri": m.uri, "experiment": m.experiment} for m in self.mlflow],
                "columns": self.columns, "overrides": self.overrides}

    def columns_for(self, path: str) -> dict:
        """The columns of one dataset: its override over the common ones."""
        merged = {k: (list(v) if isinstance(v, list) else v) for k, v in self.columns.items()}
        merged.update(self.overrides.get(str(path), {}))
        return merged

    def excluded(self, relative: str) -> bool:
        rel = relative.replace("\\", "/")
        return any(fnmatch.fnmatchcase(rel, pattern) for pattern in self.exclude)


def config_path() -> Path:
    return config_dir() / CONFIG_FILE


def _from_dict(raw) -> DatasetsConfig:
    cfg = DatasetsConfig()
    if not isinstance(raw, dict):
        return cfg
    for folder in raw.get("folders") or []:
        if isinstance(folder, str) and folder.strip():
            folder = str(Path(folder.strip()).expanduser())
            if folder not in cfg.folders:
                cfg.folders.append(folder)
    if isinstance(raw.get("exclude"), list):
        cfg.exclude = list(dict.fromkeys(str(p).strip() for p in raw["exclude"] if str(p).strip()))
    for entry in raw.get("mlflow") or []:
        if not isinstance(entry, dict):
            continue
        uri = normalize_uri(str(entry.get("uri") or ""))
        experiment = str(entry.get("experiment") or "").strip()
        if uri and experiment and (uri, experiment) not in {m.key() for m in cfg.mlflow}:
            cfg.mlflow.append(MlflowSource(uri, experiment))
    cfg.columns = _columns(raw.get("columns"))
    if isinstance(raw.get("overrides"), dict):
        for path, value in raw["overrides"].items():
            override = _columns(value, base={})
            if override:
                cfg.overrides[str(path)] = override
    return cfg


def load() -> DatasetsConfig:
    path = config_path()
    try:
        return _from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        return DatasetsConfig()
    try:
        legacy = json.loads((config_dir() / LEGACY_WATCH_FILE).read_text(encoding="utf-8"))
        return _from_dict({"folders": legacy.get("folders", [])})
    except (OSError, ValueError, AttributeError):
        return DatasetsConfig()


def save(cfg: DatasetsConfig) -> bool:
    try:
        fileio.atomic_write_text(config_path(), json.dumps(cfg.to_dict(), indent=2))
        return True
    except OSError:
        return False


def update(change) -> DatasetsConfig:
    """Read, change, write -- under the file's lock, so a folder added in
    another tab at the same moment is not written back out."""
    with fileio.locked(config_path()):
        cfg = load()
        change(cfg)
        save(cfg)
        return cfg


def add_folder(folder: str) -> bool:
    folder = str(Path(folder.strip()).expanduser()) if folder and folder.strip() else ""
    if not folder:
        return False
    added = []

    def change(cfg):
        if folder not in cfg.folders:
            cfg.folders.append(folder)
            added.append(folder)
    update(change)
    return bool(added)


def remove_folder(folder: str) -> None:
    update(lambda cfg: cfg.folders.remove(folder) if folder in cfg.folders else None)


def set_exclude(patterns: list) -> None:
    clean = list(dict.fromkeys(p.strip() for p in patterns if p and p.strip()))
    update(lambda cfg: setattr(cfg, "exclude", clean))


def add_mlflow(uri: str, experiment: str) -> bool:
    source = MlflowSource(normalize_uri(uri), (experiment or "").strip())
    if not source.uri or not source.experiment:
        return False
    added = []

    def change(cfg):
        if source.key() not in {m.key() for m in cfg.mlflow}:
            cfg.mlflow.append(source)
            added.append(source)
    update(change)
    return bool(added)


def remove_mlflow(uri: str, experiment: str) -> None:
    key = (normalize_uri(uri), experiment)
    update(lambda cfg: setattr(cfg, "mlflow", [m for m in cfg.mlflow if m.key() != key]))


def set_columns(columns: dict) -> None:
    """The common columns, for every dataset."""
    update(lambda cfg: setattr(cfg, "columns", _columns(columns)))


def set_override(path: str, override: dict) -> None:
    """Replace some columns for one dataset; an empty override removes it."""
    clean = _columns(override, base={})

    def change(cfg):
        if clean:
            cfg.overrides[str(path)] = clean
        else:
            cfg.overrides.pop(str(path), None)
    update(change)
