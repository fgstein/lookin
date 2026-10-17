"""What is known about the MLflow servers, shared by every tab.

The listing of a tracking server is the same whoever looks at it, so it is
fetched once per process rather than once per open tab: ten workspaces open
in ten tabs must not list the same server ten times (the original UI went
to some length to avoid exactly that, see ui/home_view.HomeView.on_shown).
What the user *does* with it -- a selection, an expanded run -- stays in
their own session.

The settings are the user's own home_view.json (core/home_config.py), the
same file `lookin ui` reads and writes: edits in either show up in both.

Every server call runs on a worker thread; a server that is slow or down
never blocks a page, and one that is down shows its last cached listing,
greyed, retried every minute.
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Optional

from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.core.home_config import HomeConfig
from lookin.ui.home_view import ServerState
from lookin.ui.mlflow_panel import UNSUCCESSFUL_STATUSES, _default_tracking_uri

CONFIG_POLL_S = 2.0
PROGRESSIVE_DRAW_S = 0.4
HOME_STALE_S = 300.0
RETRY_DOWN_S = 60.0


class MlflowHub:
    _instance: Optional["MlflowHub"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "MlflowHub":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.config = HomeConfig(fallback_uri=_default_tracking_uri())
        self.servers: dict = {}
        self.lock = threading.RLock()
        self._listeners: dict = {}
        self._listener_lock = threading.Lock()
        self.activated = False
        self.last_full_refresh = 0.0
        self.downloading: dict = {}       # workspace id -> progress text
        self._stop = threading.Event()
        self._watcher: Optional[threading.Thread] = None
        self.sync_servers()

    # ---- listeners ----

    def subscribe(self, key, fn: Callable[[], None]) -> None:
        with self._listener_lock:
            self._listeners[key] = fn

    def unsubscribe(self, key) -> None:
        with self._listener_lock:
            self._listeners.pop(key, None)

    def changed(self) -> None:
        with self._listener_lock:
            fns = list(self._listeners.values())
        for fn in fns:
            try:
                fn()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    # ---- servers ----

    def sync_servers(self) -> None:
        wanted = {s["uri"]: s["name"] for s in self.config.servers}
        with self.lock:
            for uri in list(self.servers):
                if uri not in wanted:
                    del self.servers[uri]
            for uri, name in wanted.items():
                if uri in self.servers:
                    self.servers[uri].name = name
                else:
                    self.servers[uri] = ServerState(uri=uri, name=name)

    def activate(self) -> None:
        """Start listing -- only once somebody actually looks at Home."""
        if not self.activated:
            self.activated = True

            def start():
                self._show_cached_listings()
                self.changed()
                self.refresh()

            threading.Thread(target=start, daemon=True).start()
            if self._watcher is None:
                self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
                self._watcher.start()
            return
        if time.monotonic() - self.last_full_refresh > HOME_STALE_S:
            self.refresh()

    def _show_cached_listings(self) -> None:
        for server in list(self.servers.values()):
            cached = mlflow_cache.load_listing(server.uri)
            if cached:
                server.adopt_listing(cached, mlflow_cache.load_artifacts(server.uri))
                server.keep_only(self.wanted_ids(server))

    def wanted_ids(self, server: ServerState) -> list:
        shown = set(self.config.shown_experiments(server.uri)) - set(self.config.hidden_experiments(server.uri))
        return [e.experiment_id for e in server.experiments if e.name in shown]

    def refresh(self, uri: Optional[str] = None, runs_only: bool = False, only: Optional[list] = None) -> None:
        if uri is None and only is None:
            self.last_full_refresh = time.monotonic()
        for server in list(self.servers.values()):
            if uri is None or server.uri == uri:
                server.status = "loading"
                threading.Thread(target=self._fetch_worker, args=(server, runs_only, only), daemon=True).start()
        self.changed()

    def _fetch_worker(self, server: ServerState, runs_only: bool, only: Optional[list] = None) -> None:
        server.last_try = time.monotonic()
        try:
            if not runs_only or not server.experiments:
                server.experiments = mlf.list_experiments(server.uri)
                self.config.adopt_defaults(server.uri, server.experiments)
            wanted = self.wanted_ids(server)
            with self.lock:
                server.keep_only(wanted)
            todo = [i for i in (only if only is not None else wanted) if i in wanted]
            filter_string = self.config.runs.get("filter", "")

            def one(experiment_id):
                try:
                    return experiment_id, mlf.search_runs(server.uri, [experiment_id], filter_string), None
                except mlf.MlflowError as ex:
                    return experiment_id, None, ex

            first_error = None
            last_draw = 0.0
            with ThreadPoolExecutor(max_workers=self.config.parallel_requests) as pool:
                for future in as_completed([pool.submit(one, i) for i in todo]):
                    experiment_id, result, error = future.result()
                    if error is not None:
                        first_error = first_error or error
                        continue
                    runs, truncated = result
                    with self.lock:
                        if experiment_id not in self.wanted_ids(server):
                            continue
                        server.runs_by_exp[experiment_id] = runs
                        server.truncated_exps.discard(experiment_id)
                        if truncated:
                            server.truncated_exps.add(experiment_id)
                        server.rebuild_runs()
                    if time.monotonic() - last_draw > PROGRESSIVE_DRAW_S:
                        last_draw = time.monotonic()
                        self.changed()
            if first_error is not None:
                raise first_error
            with self.lock:
                server.listed_at = datetime.now().isoformat(timespec="seconds")
                server.status, server.error, server.offline = "ok", "", False
            mlflow_cache.save_listing(server.uri, server.listing_payload())
        except mlf.MlflowError as ex:
            self._server_down(server, str(ex))
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            self._server_down(server, f"Unexpected error: {ex}")
        self.changed()

    def _server_down(self, server: ServerState, error: str) -> None:
        with self.lock:
            if "filter" in error.lower() and server.experiments and self.config.runs.get("filter"):
                server.status, server.error = "ok", f"The filter was rejected: {error}"
                return
            server.status, server.error = "down", error
            if not server.runs:
                cached = mlflow_cache.load_listing(server.uri)
                if cached:
                    server.adopt_listing(cached, mlflow_cache.load_artifacts(server.uri))
            server.offline = bool(server.runs)

    def fetch_artifacts(self, server: ServerState, run_id: str, then=None) -> None:
        entry = server.artifacts.get(run_id)
        if entry is not None and "csvs" in entry:
            if then:
                then()
            return
        if server.status == "down":
            server.artifacts[run_id] = {"error": "server unreachable"}
            self.changed()
            if then:
                then()
            return
        server.artifacts[run_id] = {"loading": True}

        def work():
            try:
                csvs, npzs = mlf.list_csv_and_npz_artifacts(server.uri, run_id)
                server.artifacts[run_id] = {"csvs": csvs, "npzs": npzs}
                mlflow_cache.save_artifacts(server.uri, server.artifacts_payload())
            except mlf.MlflowError as ex:
                server.artifacts[run_id] = {"error": str(ex)}
            self.changed()
            if then:
                then()

        threading.Thread(target=work, daemon=True).start()

    def _watch_loop(self) -> None:
        """Pick up hand edits of home_view.json, and retry servers that are down."""
        while not self._stop.wait(CONFIG_POLL_S):
            try:
                if self.config.changed_on_disk():
                    self.config.reload()
                    self.sync_servers()
                    self.refresh()
                    continue
                now = time.monotonic()
                for server in list(self.servers.values()):
                    if server.status == "down" and now - server.last_try > RETRY_DOWN_S:
                        server.status = "loading"
                        threading.Thread(target=self._fetch_worker, args=(server, False), daemon=True).start()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    # ---- what the tables show ----

    def visible_runs(self, server: ServerState, query: Optional[str] = None,
                     show_unsuccessful: Optional[bool] = None, experiment_id: Optional[str] = None) -> list:
        hidden = self.config.hidden_run_ids(server.uri)
        shown_names = set(self.config.shown_experiments(server.uri)) - set(self.config.hidden_experiments(server.uri))
        shown_ids = {e.experiment_id for e in server.experiments if e.name in shown_names}
        query = (self.config.runs.get("search", "") if query is None else query).strip().lower()
        if show_unsuccessful is None:
            show_unsuccessful = bool(self.config.runs.get("show_unsuccessful"))
        out = []
        for run in server.runs:
            if run.run_id in hidden:
                continue
            if server.experiments and run.experiment_id not in shown_ids:
                continue
            if experiment_id is not None and run.experiment_id != experiment_id:
                continue
            if not show_unsuccessful and run.status in UNSUCCESSFUL_STATUSES:
                continue
            if query and not run_matches(run, query):
                continue
            out.append(run)
        return out

    @staticmethod
    def membership(workspaces) -> tuple:
        by_run: dict = {}
        by_csv: dict = {}
        for ws in workspaces:
            for e in ws.mount.experiments:
                if not e.from_mlflow():
                    continue
                uri = wss.normalize_uri(e.mlflow_uri)
                lst = by_run.setdefault((uri, e.mlflow_run_id), [])
                if ws not in lst:
                    lst.append(ws)
                by_csv.setdefault((uri, e.mlflow_run_id, e.artifact_path), []).append(ws)
        return by_run, by_csv


def run_matches(run: mlf.Run, query: str) -> bool:
    haystack = " ".join([run.run_name, run.run_id, run.description,
                         *(f"{k}={v}" for k, v in run.params.items())]).lower()
    return all(word in haystack for word in query.split())
