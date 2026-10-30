"""Where the datasets are: one JSON file per user, written by the Datasets
page's settings.

    ~/.config/lookin/datasets.json

    {
      "folders": ["/shared/datasets"],
      "mlflow": [{"uri": "http://mlflow.lab:5000", "experiment": "datasets"}]
    }

`folders` are scanned for CSVs (see catalog.py). `mlflow` names the
experiments whose runs log the datasets -- one CSV and a hash.txt per run
(see mlflow_link.py). Per user, like home_view.json: two people can look at
different folders.

The folders used to be the "watched folders" of the All datasets window,
kept in web_datasets.json; that file is read once, the first time this one
does not exist yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lookin.core import fileio
from lookin.core.prefs import config_dir
from lookin.core.workspace_sources import normalize_uri

CONFIG_FILE = "datasets.json"
LEGACY_WATCH_FILE = "web_datasets.json"


@dataclass
class MlflowSource:
    uri: str
    experiment: str

    def key(self) -> tuple:
        return (self.uri, self.experiment)


@dataclass
class DatasetsConfig:
    folders: list = field(default_factory=list)
    mlflow: list = field(default_factory=list)      # MlflowSource

    def to_dict(self) -> dict:
        return {"folders": list(self.folders),
                "mlflow": [{"uri": m.uri, "experiment": m.experiment} for m in self.mlflow]}


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
    for entry in raw.get("mlflow") or []:
        if not isinstance(entry, dict):
            continue
        uri = normalize_uri(str(entry.get("uri") or ""))
        experiment = str(entry.get("experiment") or "").strip()
        if uri and experiment and (uri, experiment) not in {m.key() for m in cfg.mlflow}:
            cfg.mlflow.append(MlflowSource(uri, experiment))
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
