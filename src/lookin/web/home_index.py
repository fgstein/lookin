"""What the Home page shows from disk, read off the request path.

The Home page used to read the disk on every redraw: every workspace's
manifest, the board, the favourites file (once per row), a stat of every
file of every workspace, the snapshots folder of each row, and every
dataset of the catalogue. Locally that is a few hundred cheap calls; with
the library on a network share it was more than a second per redraw -- and
the page redraws on every click and several times a second while MLflow
lists its runs.

So the page draws from this index instead, and the index is read on a
worker thread: the page shows at once (with "Reading the library..." the
very first time), and fills in when the read is done. It is read again
when Home is shown, after a change made from Home, when the MLflow listing
moves (the catalogue names the datasets of runs) and every REFRESH_S while
someone is looking, so a workspace a colleague saved on the shared library
still appears by itself.

One index per library and config directory, shared by every tab: ten tabs
on Home read the library once, not ten times.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from lookin.core import board as bd
from lookin.core import library as lib
from lookin.core import workspace_sources as wss
from lookin.core.prefs import config_dir

# How often the index is read again while a Home page is connected.
REFRESH_S = 5.0
# A burst of requests (the MLflow listing moving several times a second)
# makes one read, this long after the first of them.
SOON_S = 1.5


@dataclass
class WsInfo:
    """What a workspace row says that is not in its manifest."""
    summary: wss.DownloadSummary
    missing: list = field(default_factory=list)       # wss._File not on disk
    present: dict = field(default_factory=dict)       # experiment name -> its files are here
    snapshots: int = 0


def _ws_info(ws) -> WsInfo:
    files = [f for e in ws.mount.experiments for f in wss._files_of(e)]
    here = {id(f): wss._present(f) for f in files}
    missing = [f for f in files if not here[id(f)]]
    summary = wss.DownloadSummary(wss.DOWNLOAD_LOCAL)
    if files:
        size: Optional[int] = 0
        for f in missing:
            if f.size is None:
                size = None
                break
            size += f.size
        state = (wss.DOWNLOAD_ALL if not missing
                 else wss.DOWNLOAD_PARTIAL if len(missing) < len(files) else wss.DOWNLOAD_NONE)
        summary = wss.DownloadSummary(state, len(missing), size)
    present = {}
    for e in ws.mount.experiments:
        if e.from_mlflow():
            present[e.name] = all(here[id(f)] for f in files if f.experiment is e)
        else:
            try:
                present[e.name] = Path(e.path).is_file()
            except OSError:
                present[e.name] = False
    try:
        snapshots = sum(1 for entry in ws.snapshots_dir().iterdir() if entry.suffix == ".json")
    except OSError:
        snapshots = 0
    return WsInfo(summary, missing, present, snapshots)


class HomeIndex:
    _instances: dict = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls) -> "HomeIndex":
        """The index of the current library root and config directory (tests
        switch both; a real process has one of each)."""
        key = (str(lib.library_root()), str(config_dir()))
        with cls._instances_lock:
            index = cls._instances.get(key)
            if index is None:
                index = cls._instances[key] = cls(key[0])
            return index

    def __init__(self, root: str):
        self.root = root
        self.lock = threading.RLock()
        self.ready = False
        self.workspaces: list = []
        self.info: dict = {}
        self.board: dict = {"columns": []}
        self.favorites: set = set()
        self.opened: dict = {}
        self.datasets: list = []
        self._snapshot_lists: dict = {}       # ws id -> (count, [Snapshot])
        self._listeners: dict = {}
        self._listener_lock = threading.Lock()
        self._busy = False
        self._again = False
        self._soon: Optional[threading.Timer] = None
        self._done = threading.Condition(self.lock)
        self._generation = 0
        self._ticker: Optional[threading.Thread] = None

    # ---- listeners ----

    def subscribe(self, key, fn: Callable[[], None]) -> None:
        with self._listener_lock:
            self._listeners[key] = fn
        self._start_ticker()

    def unsubscribe(self, key) -> None:
        with self._listener_lock:
            self._listeners.pop(key, None)

    def _changed(self) -> None:
        with self._listener_lock:
            fns = list(self._listeners.values())
        for fn in fns:
            try:
                fn()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def _start_ticker(self) -> None:
        with self._listener_lock:
            if self._ticker is not None:
                return
            self._ticker = threading.Thread(target=self._tick, daemon=True, name="lookin-home-index")
        self._ticker.start()

    def _tick(self) -> None:
        while True:
            time.sleep(REFRESH_S)
            with self._listener_lock:
                watched = bool(self._listeners)
            if watched:
                self.refresh()

    # ---- reading ----

    def refresh(self, wait: bool = False, hub=None) -> None:
        """Read the library again on a worker thread. Coalesced: a request
        made while a read runs makes one more read after it, never several.
        `wait` blocks until a read that started after this call is done."""
        if hub is not None:
            self._hub = hub
        with self.lock:
            target = self._generation + (2 if self._busy else 1)
            if self._busy:
                self._again = True
            else:
                self._busy = True
                threading.Thread(target=self._run, daemon=True, name="lookin-home-read").start()
            if wait:
                while self._generation < target:
                    self._done.wait(30)

    def refresh_soon(self, hub=None) -> None:
        """refresh(), once, SOON_S from the first of a burst of calls."""
        if hub is not None:
            self._hub = hub
        with self.lock:
            if self._soon is not None:
                return
            self._soon = threading.Timer(SOON_S, self._fire_soon)
            self._soon.daemon = True
            self._soon.start()

    def _fire_soon(self) -> None:
        with self.lock:
            self._soon = None
        self.refresh()

    def ensure(self, hub=None) -> None:
        """A first read, if none was made or started yet."""
        if hub is not None:
            self._hub = hub
        with self.lock:
            if self.ready or self._busy:
                return
        self.refresh()

    def _run(self) -> None:
        while True:
            try:
                self._read()
            except Exception:  # noqa: BLE001 -- a read that failed keeps the last good one
                traceback.print_exc()
            with self.lock:
                self._generation += 1
                self._done.notify_all()
                if not self._again:
                    self._busy = False
                    break
                self._again = False
            self._changed()
        self._changed()

    def _read(self) -> None:
        from lookin.web import datasets as dsc

        workspaces = lib.list_workspaces(self.root)
        info = {w.id: _ws_info(w) for w in workspaces}
        board = bd.load(self.root)
        personal = lib.load_personal()
        catalog = dsc.catalog(workspaces, getattr(self, "_hub", None))
        for entry in catalog:
            entry.cached = dsc.cached_facts(entry.path) if entry.here else None
        with self.lock:
            self.workspaces = workspaces
            self.info = info
            self.board = board
            self.favorites = set(personal["favorites"])
            self.opened = dict(personal["opened"])
            self.datasets = catalog
            self.ready = True

    # ---- changes made from Home: shown at once, read again behind ----

    def put(self, ws) -> None:
        """`ws` as just written: replaces its old self in the index."""
        info = _ws_info(ws)
        with self.lock:
            self.workspaces = [ws] + [w for w in self.workspaces if w.id != ws.id]
            self.workspaces.sort(key=lambda b: b.updated_at, reverse=True)
            self.info[ws.id] = info
            self._snapshot_lists.pop(ws.id, None)
        self.refresh()

    def remove(self, workspace_id: str) -> None:
        with self.lock:
            self.workspaces = [w for w in self.workspaces if w.id != workspace_id]
            self.info.pop(workspace_id, None)
        self.refresh()

    def set_favorite(self, object_id: str, favorite: bool) -> None:
        lib.set_favorite(object_id, favorite)
        with self.lock:
            (self.favorites.add if favorite else self.favorites.discard)(object_id)

    def reload_board(self) -> dict:
        """The board after Home changed it: one small file, read now."""
        board = bd.load(self.root)
        with self.lock:
            self.board = board
        return board

    # ---- reading the index ----

    def workspace(self, workspace_id: str):
        with self.lock:
            return next((w for w in self.workspaces if w.id == workspace_id), None)

    def info_of(self, ws) -> WsInfo:
        with self.lock:
            info = self.info.get(ws.id)
        if info is None:          # put() since the last read: computed once
            info = _ws_info(ws)
            with self.lock:
                self.info[ws.id] = info
        return info

    def is_favorite(self, object_id: str) -> bool:
        return object_id in self.favorites

    def last_opened(self, object_id: str) -> str:
        return str(self.opened.get(object_id, ""))

    def snapshots(self, ws) -> list:
        """A workspace's snapshots, read again only when their count moved."""
        count = self.info_of(ws).snapshots
        with self.lock:
            hit = self._snapshot_lists.get(ws.id)
        if hit is not None and hit[0] == count:
            return hit[1]
        snaps = lib.list_snapshots(ws)
        with self.lock:
            self._snapshot_lists[ws.id] = (count, snaps)
        return snaps


def hash_dataset_later(ws, path: str, then: Optional[Callable] = None) -> None:
    """Record the fingerprint of `ws`'s dataset `path`, read on a worker
    thread: hashing reads the whole file, which for a large CSV on a network
    share is seconds the request used to wait for, the page frozen. Only
    written if the workspace still uses that file by then. `then(fresh)` is
    called with the workspace as written."""
    if not path:
        return

    def work():
        try:
            if not Path(path).is_file():
                return
            digest = lib.hash_file(path)
        except OSError:
            return

        def change(fresh):
            if fresh.mount.dataset_path != path or fresh.dataset_hash == digest:
                return False
            fresh.dataset_hash = digest
            return True
        fresh = lib.update_workspace(ws.path, change)
        if fresh is not None and then is not None:
            then(fresh)

    threading.Thread(target=work, daemon=True, name="lookin-hash").start()
