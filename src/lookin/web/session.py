"""One session per browser tab, and everything that belongs to it.

Isolation is the point. Several workspaces are explored at once, each in a
tab of its own, and nothing one tab does may reach another: each session
holds its own AppState (the loaded data), its own views (filters, charts,
selections), its own display settings, its own background jobs and its own
push channel. The only thing tabs share is what is on disk -- the library
of workspaces and the Home settings -- and that is shared on purpose.

The tab keeps its session id in sessionStorage, so a reload comes back to
the same session with everything still loaded; a duplicated tab notices
the clash and takes a new id (see static/app.js).

The attributes below deliberately mirror lookin.ui.context.AppContext
(`state`, `display_lut`, `display_roll_col`, `prototype_source`,
`view_state`, `current_workspace`, ...): the Flet-free helpers of the
original UI -- view_state.py, prototype_source.py, the crop encoders --
take a "ctx" and read exactly those, so they work unchanged on a Session.
"""
from __future__ import annotations

import json
import queue
import secrets
import threading
import time
import traceback
from collections import OrderedDict
from typing import Callable, Optional

from lookin.core.data_manager import AppState
from lookin.ui.components import LUT_NONE, ROLL_NONE, _HEAVY_WORK_LOCK
from lookin.ui.prototype_source import PrototypeSourceRegistry
from lookin.ui.view_state import ViewStateRegistry

# A session nobody has talked to for this long, with no page connected to
# its event stream, is dropped -- the tab was closed.
IDLE_TIMEOUT_S = 3 * 3600.0
REAP_EVERY_S = 300.0
CHART_CACHE_MAX = 400
DOWNLOAD_CACHE_MAX = 40

_local = threading.local()


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class Busy:
    """"Is anything still running?", answered in the top bar of every view
    of the session (the web twin of components.BusyIndicator)."""

    def __init__(self, session: "Session"):
        self.session = session
        self._jobs: list = []
        self._lock = threading.Lock()

    def begin(self, what: str) -> None:
        with self._lock:
            self._jobs.append([what, ""])
        self.session.mark("chrome")

    def detail_text(self, detail: str) -> None:
        with self._lock:
            if self._jobs:
                self._jobs[-1][1] = detail
        self.session.mark("chrome")

    def end(self) -> None:
        with self._lock:
            if self._jobs:
                self._jobs.pop()
        self.session.mark("chrome")

    @property
    def running(self) -> list:
        with self._lock:
            return [what for what, _ in self._jobs]


class Job:
    """One view's expensive rebuild, off the request thread -- the web twin
    of components.BackgroundRebuild: one job per view at a time, the
    *latest* request coalesced behind the running one rather than dropped,
    and every job behind the process-wide heavy-work lock (matplotlib and
    the CPU-bound rebuilds must not run three at once)."""

    def __init__(self, session: "Session", name: str):
        self.session = session
        self.name = name
        self._lock = threading.Lock()
        self._busy = False
        self._pending = None

    @property
    def busy(self) -> bool:
        return self._busy

    def run(self, work: Callable[[], None], on_error: Optional[Callable] = None) -> bool:
        with self._lock:
            if self._busy:
                self._pending = (work, on_error)
                return False
            self._busy = True
        threading.Thread(target=self._runner, args=(work, on_error), daemon=True).start()
        return True

    def _runner(self, work, on_error):
        while True:
            self.session.busy.begin(self.name)
            try:
                with _HEAVY_WORK_LOCK:
                    work()
            except Exception as ex:  # noqa: BLE001
                _log(f"{self.name}: rebuild failed: {ex}")
                traceback.print_exc()
                if on_error is not None:
                    try:
                        on_error(ex)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()
            finally:
                self.session.busy.end()
            with self._lock:
                if self._pending is None:
                    self._busy = False
                    return
                work, on_error = self._pending
                self._pending = None


class Session:
    def __init__(self, sid: str, app):
        self.sid = sid
        self.app = app                       # the WebApp (CLI options, store)
        self.created = time.time()
        self.last_seen = time.time()
        self.lock = threading.RLock()        # one request at a time per tab
        self.closed = False

        # ---- the AppContext half (see the module docstring) ----
        self.state = AppState()
        self.display_lut = LUT_NONE
        self.display_roll_col = ROLL_NONE
        self.prototype_source = PrototypeSourceRegistry(self.state)
        self.view_state = ViewStateRegistry(self)
        self.current_workspace = None
        self.current_snapshot = None
        self.view_state_baseline = None
        self.busy = Busy(self)
        self.viewport = (1440, 900)          # the browser window, (width, height)

        # ---- what the page shows ----
        self.mode = "home"                   # "home" | "analysis"
        self.tab = "sources"
        self.title = "Lookin"
        self.views: dict = {}
        self.dialogs: list = []
        self.booted = False

        # ---- fragments served by URL ----
        self._images: dict = {}              # key -> Path (only paths a view registered)
        self._charts: "OrderedDict[str, bytes]" = OrderedDict()
        self._downloads: "OrderedDict[str, tuple]" = OrderedDict()
        self._point_maps: "OrderedDict[str, Callable]" = OrderedDict()
        self._blob_lock = threading.Lock()

        # ---- push channel ----
        self._subscribers: list = []
        self._sub_lock = threading.Lock()
        self._pending_dirty: set = set()
        self._pending_cmds: list = []
        self._dialogs_dirty = False
        self._jobs: dict = {}
        self._listeners: list = []
        self._closers: list = []

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------

    def view(self, name: str):
        """The session's instance of a view, built on first use."""
        v = self.views.get(name)
        if v is None:
            from lookin.web.views import VIEW_CLASSES
            cls = VIEW_CLASSES[name]
            v = cls(self)
            self.views[name] = v
        return v

    @property
    def upload_view(self):
        """The Sources view -- what save_status and the workspace code call
        "upload_view", after the original UI's Data tab."""
        return self.view("sources")

    def goto(self, tab: str) -> None:
        """Switch to another analysis tab, as clicking it would (a
        confusion-matrix cell's "Open in Gallery", for instance)."""
        from lookin.web.render import AppActions
        AppActions(self).on_tab(tab=tab)

    def job(self, name: str) -> Job:
        j = self._jobs.get(name)
        if j is None:
            j = self._jobs[name] = Job(self, name)
        return j

    def on_close(self, fn: Callable[[], None]) -> None:
        self._closers.append(fn)

    def close(self) -> None:
        self.closed = True
        for fn in self._closers:
            try:
                fn()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    # ------------------------------------------------------------------
    # change notification (the notify_* half of AppContext)
    # ------------------------------------------------------------------

    def notify_change(self) -> None:
        """Data changed: every analysis view marks itself stale (they show
        an "out of date -- update" banner rather than recomputing)."""
        for name, v in list(self.views.items()):
            fn = getattr(v, "on_data_change", None)
            if fn is not None:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        self.mark("*")

    def notify_rename(self, old: str, new: str) -> None:
        self.prototype_source.rename_experiment(old, new)
        for v in list(self.views.values()):
            fn = getattr(v, "rename_experiment", None)
            if fn is not None:
                try:
                    fn(old, new)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        self.mark("*")

    def notify_review(self) -> None:
        for v in list(self.views.values()):
            fn = getattr(v, "on_review_change", None)
            if fn is not None:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        self.mark("*")

    def notify_display_change(self) -> None:
        for v in list(self.views.values()):
            fn = getattr(v, "on_display_changed", None)
            if fn is not None:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        self.mark("*")

    # ------------------------------------------------------------------
    # pushing to the page
    # ------------------------------------------------------------------

    def mark(self, *names: str) -> None:
        """These views (or "chrome", "dialogs", "*") need re-rendering. Inside
        a request they ride on its response; from a background thread they
        are pushed over the event stream and the page asks for them."""
        if getattr(_local, "session", None) is self:
            self._pending_dirty.update(names)
            return
        self.push({"dirty": list(names)})

    def cmd(self, **command) -> None:
        """A client-side command: toast, open, copy, tab, title, download."""
        if getattr(_local, "session", None) is self:
            self._pending_cmds.append(command)
            return
        self.push({"cmds": [command]})

    def toast(self, text: str, kind: str = "ok") -> None:
        if text:
            self.cmd(toast=text, kind=kind)

    def push(self, message: dict) -> None:
        data = json.dumps(message)
        with self._sub_lock:
            subs = list(self._subscribers)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, data)
            except RuntimeError:      # the loop is gone: that page is too
                self.unsubscribe((loop, q))

    def subscribe(self, loop, q) -> tuple:
        """Register an event-stream reader: an asyncio queue on `loop`,
        fed from any thread."""
        sub = (loop, q)
        with self._sub_lock:
            self._subscribers.append(sub)
        return sub

    def unsubscribe(self, q) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    @property
    def connected(self) -> bool:
        with self._sub_lock:
            return bool(self._subscribers)

    # ------------------------------------------------------------------
    # dialogs
    # ------------------------------------------------------------------

    def open_dialog(self, dialog) -> None:
        dialog.session = self
        self.dialogs = [d for d in self.dialogs if d.id != dialog.id] + [dialog]
        self.mark("dialogs")

    def close_dialog(self, dialog_or_id) -> None:
        did = dialog_or_id if isinstance(dialog_or_id, str) else dialog_or_id.id
        self.dialogs = [d for d in self.dialogs if d.id != did]
        self.mark("dialogs")

    def dialog(self, did: str):
        return next((d for d in self.dialogs if d.id == did), None)

    # ------------------------------------------------------------------
    # served blobs: crops, charts, downloads
    # ------------------------------------------------------------------

    def image_key(self, path) -> str:
        """A key the page can ask /img for. Only paths registered here are
        ever served -- the URL never carries a path."""
        sp = str(path)
        key = format(abs(hash(sp)) % (1 << 48), "x")
        with self._blob_lock:
            self._images[key] = sp
        return key

    def image_path(self, key: str) -> Optional[str]:
        with self._blob_lock:
            return self._images.get(key)

    def put_point_map(self, resolve: Callable[[int], Optional[tuple]]) -> str:
        """Register an interactive map's point -> (path, lut, turns)
        resolver; /pimg/<sid>/<key>/<i> serves through it. Only the last
        few maps are kept."""
        key = secrets.token_hex(6)
        with self._blob_lock:
            self._point_maps[key] = resolve
            while len(self._point_maps) > 4:
                self._point_maps.popitem(last=False)
        return key

    def point_map(self, key: str):
        with self._blob_lock:
            return self._point_maps.get(key)

    def put_chart(self, png: bytes, key: Optional[str] = None) -> str:
        key = key or secrets.token_hex(8)
        with self._blob_lock:
            self._charts[key] = png
            self._charts.move_to_end(key)
            while len(self._charts) > CHART_CACHE_MAX:
                self._charts.popitem(last=False)
        return key

    def chart(self, key: str) -> Optional[bytes]:
        with self._blob_lock:
            return self._charts.get(key)

    def put_download(self, data: bytes, filename: str, mime: str = "application/octet-stream") -> str:
        key = secrets.token_hex(8)
        with self._blob_lock:
            self._downloads[key] = (data, filename, mime)
            while len(self._downloads) > DOWNLOAD_CACHE_MAX:
                self._downloads.popitem(last=False)
        return key

    def download(self, key: str):
        with self._blob_lock:
            return self._downloads.get(key)


class RequestScope:
    """`with RequestScope(session):` -- changes marked inside ride on the
    response instead of the event stream."""

    def __init__(self, session: Session):
        self.session = session

    def __enter__(self):
        self.session.lock.acquire()
        _local.session = self.session
        self.session.last_seen = time.time()
        return self.session

    def __exit__(self, *exc):
        _local.session = None
        self.session.lock.release()
        return False


class SessionStore:
    def __init__(self, app):
        self.app = app
        self._sessions: dict = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True).start()

    def get(self, sid: str, create: bool = True) -> Optional[Session]:
        if not sid or len(sid) > 64 or not all(c.isalnum() or c in "-_" for c in sid):
            return None
        with self._lock:
            s = self._sessions.get(sid)
            if s is None and create:
                s = self._sessions[sid] = Session(sid, self.app)
            return s

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _reaper(self) -> None:
        while True:
            time.sleep(REAP_EVERY_S)
            now = time.time()
            with self._lock:
                stale = [sid for sid, s in self._sessions.items()
                         if not s.connected and now - s.last_seen > IDLE_TIMEOUT_S]
                dropped = [self._sessions.pop(sid) for sid in stale]
            for s in dropped:
                s.close()
