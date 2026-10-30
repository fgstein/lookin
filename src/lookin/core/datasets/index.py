"""Everything known about the datasets, kept up to date for the whole
process: the Datasets page of every browser tab and the MCP tools read the
same index.

* The folder scan is cheap (names and sizes) and redone on demand.
* Profiles are expensive (a full read of the CSV), so they are computed by
  one worker thread, one file at a time, and cached on disk by content
  hash -- a profile survives restarts, and a CSV copied to another folder
  is not read again. Every file of the shared folders is queued when the
  page first opens, so the list fills in on its own; the dataset someone
  is looking at jumps the queue.
* The content hash of a file is remembered against its size and
  modification time, so it is recomputed only when the file changes.
* The MLflow index (mlflow_link.py) is refreshed on demand, in the
  background, and kept on disk too.

Nothing here writes to the shared folders.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

from lookin.core import fileio
from lookin.core import mlflow_cache
from lookin.core.dataset_config import CONFIG_NAME
from lookin.core.datasets import catalog, compare as cmp, config as dcfg, mlflow_link, profile as prof

FINGERPRINTS_FILE = "fingerprints.json"
PROFILES_DIR = "profiles"
MEMORY_PROFILES = 64

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_FAILED = "failed"


def _cache_dir() -> Path:
    return mlflow_cache.user_cache_dir() / "datasets"


def _toml_stamp(path: str) -> int:
    try:
        return (Path(path).parent / CONFIG_NAME).stat().st_mtime_ns
    except OSError:
        return 0


def _roles_signature(roles: dict) -> str:
    text = json.dumps({k: roles.get(k, "") for k in ("id", "image", "label", "split", "source", "root")},
                      sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class DatasetIndex:
    _instance: Optional["DatasetIndex"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "DatasetIndex":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Forget the process-wide index (tests)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._stop.set()
            cls._instance = None

    def __init__(self):
        self.lock = threading.RLock()
        self.config = dcfg.load()
        self.files: list = []
        self.by_path: dict = {}
        self.scanned_at = 0.0
        # path -> {"size", "mtime_ns", "toml", "hash", "roles"} (roles signature)
        self.fingerprints: dict = self._load_fingerprints()
        self._profiles: OrderedDict = OrderedDict()     # "<hash>-<sig>" -> profile
        self.status: dict = {}                          # path -> (state, text)
        self._queue: list = []
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._compute_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self.mlflow: dict = mlflow_link.load_index()
        self.mlflow_busy = ""
        self._listeners: dict = {}

    # ---- listeners --------------------------------------------------------

    def subscribe(self, key, fn: Callable[[], None]) -> None:
        with self.lock:
            self._listeners[key] = fn

    def unsubscribe(self, key) -> None:
        with self.lock:
            self._listeners.pop(key, None)

    def changed(self) -> None:
        with self.lock:
            fns = list(self._listeners.values())
        for fn in fns:
            try:
                fn()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    # ---- config and scan ----------------------------------------------------

    def reload_config(self) -> dcfg.DatasetsConfig:
        self.config = dcfg.load()
        return self.config

    def rescan(self) -> list:
        cfg = self.reload_config()
        files = catalog.scan(cfg.folders)
        with self.lock:
            self.files = files
            self.by_path = {f.path: f for f in files}
            self.scanned_at = time.time()
        return files

    def ensure_scanned(self, max_age: float = 30.0) -> list:
        if not self.scanned_at or time.time() - self.scanned_at > max_age:
            return self.rescan()
        return self.files

    def find(self, key: str) -> Optional[catalog.DatasetFile]:
        """A dataset by path, by path relative to its shared folder, or by
        file name (or name without .csv) when that is unique."""
        self.ensure_scanned()
        key = (key or "").strip()
        if not key:
            return None
        if key in self.by_path:
            return self.by_path[key]
        for f in self.files:
            if f.relative() == key:
                return f
        named = [f for f in self.files if f.name == key or f.stem == key]
        if len(named) == 1:
            return named[0]
        hashed = [f for f in self.files if self.hash_of(f.path) == key]
        return hashed[0] if hashed else None

    def ambiguous(self, key: str) -> list:
        return [f for f in self.files if f.name == key or f.stem == key]

    # ---- fingerprints -----------------------------------------------------

    def _fp_path(self) -> Path:
        return _cache_dir() / FINGERPRINTS_FILE

    def _load_fingerprints(self) -> dict:
        try:
            raw = json.loads(self._fp_path().read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_fingerprints(self) -> None:
        try:
            with self.lock:
                text = json.dumps(self.fingerprints)
            self._fp_path().parent.mkdir(parents=True, exist_ok=True)
            fileio.atomic_write_text(self._fp_path(), text)
        except OSError:
            pass

    def _record(self, path: str) -> Optional[dict]:
        """The fingerprint record of a file, if it still describes it."""
        f = self.by_path.get(path)
        rec = self.fingerprints.get(path)
        if not rec or f is None:
            return None
        if rec.get("size") != f.size or rec.get("mtime_ns") != f.mtime_ns or rec.get("toml") != _toml_stamp(path):
            return None
        return rec

    def hash_of(self, path: str) -> str:
        rec = self._record(path)
        return rec["hash"] if rec else ""

    # ---- profiles ---------------------------------------------------------

    def _profile_file(self, key: str) -> Path:
        return _cache_dir() / PROFILES_DIR / f"{key}.json"

    def profile_of(self, path: str) -> Optional[dict]:
        """The cached profile of a file, or None when it has to be computed."""
        rec = self._record(path)
        if not rec:
            return None
        key = f"{rec['hash']}-{rec['roles']}"
        with self.lock:
            hit = self._profiles.get(key)
            if hit is not None:
                self._profiles.move_to_end(key)
                return hit
        try:
            p = json.loads(self._profile_file(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if p.get("version") != prof.PROFILE_VERSION:
            return None
        self._remember(key, p)
        return p

    def _remember(self, key: str, p: dict) -> None:
        with self.lock:
            self._profiles[key] = p
            self._profiles.move_to_end(key)
            while len(self._profiles) > MEMORY_PROFILES:
                self._profiles.popitem(last=False)

    def compute(self, path: str, progress=None) -> dict:
        """The profile of `path`, computed now if it is not cached. Raises on
        a file that cannot be read. One computation at a time, process-wide:
        reading two large CSVs at once only doubles the memory peak."""
        cached = self.profile_of(path)
        if cached is not None:
            return cached
        with self._compute_lock:
            cached = self.profile_of(path)
            if cached is not None:
                return cached
            f = self.by_path.get(path)
            toml = _toml_stamp(path)
            roles = prof.resolve_roles(path)
            if roles.error:
                raise ValueError(roles.error)
            p = prof.compute(path, roles, progress=progress)
            sig = _roles_signature(p["roles"])
            key = f"{p['hash']}-{sig}"
            try:
                target = self._profile_file(key)
                target.parent.mkdir(parents=True, exist_ok=True)
                fileio.atomic_write_text(target, json.dumps(p))
            except OSError:
                pass
            self._remember(key, p)
            if f is not None:
                with self.lock:
                    self.fingerprints[path] = {"size": f.size, "mtime_ns": f.mtime_ns, "toml": toml,
                                               "hash": p["hash"], "roles": sig}
                self._save_fingerprints()
            return p

    # ---- the background queue -----------------------------------------------

    def request(self, path: str, first: bool = False) -> None:
        """Queue `path` for profiling (at the front with `first`)."""
        if self.profile_of(path) is not None:
            return
        with self.lock:
            state = self.status.get(path, ("", ""))[0]
            if state == STATE_RUNNING:
                return
            if path in self._queue:
                if not first:
                    return
                self._queue.remove(path)
            if first:
                self._queue.insert(0, path)
            else:
                self._queue.append(path)
            self.status[path] = (STATE_QUEUED, "Waiting to be read")
            self._start_worker()
        self._wake.set()

    def request_all(self) -> None:
        for f in self.ensure_scanned():
            if self.profile_of(f.path) is None and self.status.get(f.path, ("", ""))[0] != STATE_FAILED:
                self.request(f.path)

    def retry(self, path: str) -> None:
        with self.lock:
            self.status.pop(path, None)
        self.request(path, first=True)

    def recompute(self, path: str) -> None:
        """Forget what is known of `path` and profile it again, first."""
        with self.lock:
            rec = self.fingerprints.pop(path, None)
            if rec:
                self._profiles.pop(f"{rec['hash']}-{rec['roles']}", None)
            self.status.pop(path, None)
        self._save_fingerprints()
        self.request(path, first=True)

    def _start_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="lookin-datasets", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                path = self._queue.pop(0) if self._queue else None
                if path is not None:
                    self.status[path] = (STATE_RUNNING, "Reading")
            if path is None:
                self._wake.wait(timeout=5.0)
                self._wake.clear()
                continue
            self.changed()

            def progress(text, path=path):
                with self.lock:
                    self.status[path] = (STATE_RUNNING, text)
                self.changed()
            try:
                self.compute(path, progress=progress)
                with self.lock:
                    self.status.pop(path, None)
            except Exception as ex:  # noqa: BLE001
                with self.lock:
                    self.status[path] = (STATE_FAILED, f"{type(ex).__name__}: {ex}")
            self.changed()

    def pending(self) -> int:
        with self.lock:
            return len(self._queue) + sum(1 for s, _ in self.status.values() if s == STATE_RUNNING)

    # ---- MLflow -------------------------------------------------------------

    def refresh_mlflow(self, on_done=None) -> None:
        """Re-read the datasets experiments, in the background."""
        if self.mlflow_busy:
            return
        cfg = self.reload_config()
        self.mlflow_busy = "Listing the datasets experiments"
        self.changed()

        def work():
            index = {}
            try:
                for source in cfg.mlflow:
                    def progress(text):
                        self.mlflow_busy = text
                        self.changed()
                    index[source.key()] = mlflow_link.refresh_source(
                        source.uri, source.experiment, self.mlflow.get(source.key()), progress=progress)
                with self.lock:
                    self.mlflow = index
                try:
                    mlflow_link.save_index(index)
                except OSError:
                    pass
            finally:
                self.mlflow_busy = ""
                self.changed()
                if on_done is not None:
                    on_done()
        threading.Thread(target=work, name="lookin-datasets-mlflow", daemon=True).start()

    def mlflow_sources(self) -> dict:
        """The configured sources only: one removed from the settings is
        not matched against any more."""
        keys = {m.key() for m in self.config.mlflow}
        with self.lock:
            return {k: v for k, v in self.mlflow.items() if k in keys}

    def link(self, path: str) -> mlflow_link.Link:
        """The MLflow run of a file -- or of its main dataset, for a
        derivative that has none of its own."""
        f = self.by_path.get(path)
        if f is None:
            return mlflow_link.Link()
        sources = self.mlflow_sources()
        own = mlflow_link.match(sources, f.name, f.size, self.hash_of(path))
        if own.status or not f.parent:
            return own
        parent = self.by_path.get(f.parent)
        if parent is None:
            return own
        inherited = mlflow_link.match(sources, parent.name, parent.size, self.hash_of(parent.path))
        if inherited.run is None:
            return own
        return mlflow_link.Link(mlflow_link.LINK_DERIVED, inherited.run,
                                f"Derived from {parent.name}, whose run this is ({inherited.status}).")

    # ---- comparison ---------------------------------------------------------

    def compare(self, a: str, b: str) -> dict:
        pa, pb = self.compute(a), self.compute(b)
        return cmp.compare(a, pa, b, pb)
