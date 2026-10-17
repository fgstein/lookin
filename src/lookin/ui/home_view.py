"""The Home page: every MLflow run on the left, every workspace on the right.

This is where Lookin starts. The analysis tabs are what a workspace opens
into; Home is where the user decides which runs are worth looking at and
puts them in one.

    runs        One table across every configured MLflow server, grouped by
                experiment -- MLflow's own runs page without the charts.
                A run expands to its results CSVs, since one run routinely
                logs several (one per split, one per test set).
    workspaces  The library's workspaces, each a drop target. Dropping a
                run or a CSV on one adds it there; the download button
                beside it fetches what was added.

The link between the two is shown from both sides: a run carries a coloured
badge for every workspace it is in, and a workspace lists the runs it holds
with a link back to each one in the MLflow UI.

What is shown -- servers, experiments, hidden runs, columns, filters -- is
the user's own JSON file (core/home_config.py), read back whenever it
changes on disk.

Every server call runs on a worker thread, as in the MLflow dialog: a
tracking server is remote by definition, and one that is slow or down must
never freeze the page. When one is down, the last listing fetched from it
is shown instead, greyed, so the workspaces built from it stay usable.
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import flet as ft

from lookin.core import board as bd
from lookin.core import library as lib
from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.core.home_config import (
    METRIC_PREFIX, PARAM_PREFIX, HomeConfig, column_label, column_value,
)
from lookin.ui import browser_panel as bp
from lookin.ui.browser_panel import TEXT_HEADING, TEXT_ROW, TEXT_SMALL, TEXT_SUB, TEXT_TITLE
from lookin.ui.components import ACCENT, info_banner, status_row
from lookin.ui.mlflow_panel import (
    STATUS_COLORS, UNSUCCESSFUL_STATUSES, _default_tracking_uri, _format_time, _human_size,
)
from lookin.ui.server_browser import ServerFileBrowser

DRAG_GROUP = "lookin-results"
KEY_SEP = "\t"   # never in a URI, a run id or an artifact path

# How much of each list is drawn at once; a "Show more" button adds the next
# page. The browser builds every control it is sent before it paints
# anything, and it is by far the slow side: a Home page of 300 workspaces
# and 250 run rows was 11,000 controls and took ~20s to first appear, where
# the server needed a quarter of a second to send it. Nobody reads past the
# first screenfuls anyway -- they search.
RUN_PAGE = 100
BOARD_PAGE = 25
MAX_RUN_ROWS = RUN_PAGE   # kept for the name older comments use
COLUMN_WIDTH = 80
BADGES_WIDTH = 110
# Workspaces take two thirds of the page and runs one third: the workspaces
# are what the user builds and comes back to, the runs what they pick from.
RUNS_SHARE, WORKSPACES_SHARE = 1, 2
# The run name never gets narrower than this. On a narrow runs panel with
# many columns the table scrolls sideways instead of squeezing the names out.
RUN_NAME_MIN_WIDTH = 140
PAGE_PADDING = 32 + 15    # the page's side padding, plus the divider
# Fixed part of a run row: chevron, checkbox and status dot on the left;
# link and menu on the right; row padding and the gaps between cells.
ROW_FIXED_WIDTH = 64 + 64 + 36
# The workspace board: one fixed-width column per user-named group, side by
# side, scrolling sideways once they no longer fit (see core/board.py).
WORKSPACE_DRAG_GROUP = "lookin-workspace"
BOARD_COLUMN_WIDTH = 330
HANDLE_LINE_WIDTH = 10

CONFIG_POLL_S = 2.0       # how often the config file is checked for edits
PROGRESSIVE_DRAW_S = 0.4
HOME_STALE_S = 300.0      # coming back to Home refreshes the runs if they are older than this
MAX_EXPERIMENT_CHIPS = 8  # shown experiments drawn as chips; the rest are one "+N more" chip  # while runs stream in, the table is redrawn at most this often
RETRY_DOWN_S = 60.0       # how often an unreachable server is tried again

# Workspace badge colours. Chosen by the workspace id, so a workspace keeps
# its colour whatever else is created or deleted.
WORKSPACE_COLORS = (
    ft.Colors.PURPLE_200, ft.Colors.TEAL_200, ft.Colors.ORANGE_200, ft.Colors.PINK_200,
    ft.Colors.LIGHT_BLUE_200, ft.Colors.LIME_200, ft.Colors.AMBER_200, ft.Colors.CYAN_200,
)

DOWNLOAD_ICONS = {
    wss.DOWNLOAD_ALL: (ft.Icons.CIRCLE, ft.Colors.GREEN, "Downloaded -- everything is on this machine"),
    wss.DOWNLOAD_PARTIAL: (ft.Icons.CIRCLE, ft.Colors.AMBER, "Partly downloaded"),
    wss.DOWNLOAD_NONE: (ft.Icons.CIRCLE_OUTLINED, ft.Colors.ON_SURFACE_VARIANT, "Not downloaded yet"),
}


def workspace_color(workspace_id: str) -> str:
    return WORKSPACE_COLORS[sum(map(ord, workspace_id or "")) % len(WORKSPACE_COLORS)]


def run_key(uri: str, run_id: str) -> str:
    return KEY_SEP.join(("run", wss.normalize_uri(uri), run_id))


def csv_key(uri: str, run_id: str, path: str) -> str:
    return KEY_SEP.join(("csv", wss.normalize_uri(uri), run_id, path))


def parse_key(key: str) -> tuple:
    """("run", uri, run_id, "") or ("csv", uri, run_id, path)."""
    parts = (key or "").split(KEY_SEP)
    if len(parts) == 3 and parts[0] == "run":
        return "run", parts[1], parts[2], ""
    if len(parts) == 4 and parts[0] == "csv":
        return "csv", parts[1], parts[2], parts[3]
    return "", "", "", ""


def _dc_from_dict(cls, raw: dict):
    known = cls.__dataclass_fields__
    return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ServerState:
    """What is known about one tracking server right now.

    Runs are kept per experiment (`runs_by_exp`), because that is the unit
    they are fetched in: showing one more experiment fetches that one, and
    hiding one fetches nothing. `runs` is the flattened view the table
    draws, grouped by experiment -- see rebuild_runs.
    """
    uri: str
    name: str
    status: str = "idle"              # idle | loading | ok | down
    error: str = ""
    experiments: list = field(default_factory=list)   # [mlf.Experiment]
    runs_by_exp: dict = field(default_factory=dict)   # experiment id -> [mlf.Run]
    runs: list = field(default_factory=list)          # [mlf.Run], grouped by experiment
    truncated_exps: set = field(default_factory=set)
    listed_at: str = ""               # when the runs were last fetched
    offline: bool = False             # showing a cached listing because the server is down
    # run_id -> {"csvs": [...], "npzs": [...]} | {"error": str} | {"loading": True}
    artifacts: dict = field(default_factory=dict)
    last_try: float = 0.0

    @property
    def truncated(self) -> bool:
        return bool(self.truncated_exps)

    def experiment_by_id(self, experiment_id: str):
        return next((e for e in self.experiments if e.experiment_id == experiment_id), None)

    def run_by_id(self, run_id: str):
        return next((r for r in self.runs if r.run_id == run_id), None)

    def rebuild_runs(self) -> None:
        """Flatten runs_by_exp, one experiment after the other (by name).

        Grouped explicitly: MLflow returns runs sorted by start time across
        experiments, so drawing them in that order interleaves experiments
        and repeats the same experiment header several times down the table.
        """
        names = {e.experiment_id: e.name.lower() for e in self.experiments}
        order = sorted(self.runs_by_exp, key=lambda i: (names.get(i, "~"), i))
        self.runs = [run for experiment_id in order for run in self.runs_by_exp[experiment_id]]

    def keep_only(self, experiment_ids) -> None:
        keep = set(experiment_ids)
        for experiment_id in list(self.runs_by_exp):
            if experiment_id not in keep:
                del self.runs_by_exp[experiment_id]
        self.truncated_exps &= keep
        self.rebuild_runs()

    def listing_payload(self) -> dict:
        return {
            "listed_at": self.listed_at,
            "experiments": [asdict(e) for e in self.experiments],
            "runs": [asdict(r) for r in self.runs],
            "truncated_experiments": sorted(self.truncated_exps),
        }

    def artifacts_payload(self) -> dict:
        return {
            run_id: {"csvs": [asdict(a) for a in v["csvs"]], "npzs": [asdict(a) for a in v["npzs"]]}
            for run_id, v in self.artifacts.items() if "csvs" in v
        }

    def adopt_listing(self, payload: dict, artifacts: Optional[dict] = None) -> None:
        self.experiments = [_dc_from_dict(mlf.Experiment, e) for e in payload.get("experiments", [])
                            if isinstance(e, dict)]
        self.runs_by_exp = {}
        for raw in payload.get("runs", []):
            if isinstance(raw, dict):
                run = _dc_from_dict(mlf.Run, raw)
                self.runs_by_exp.setdefault(run.experiment_id, []).append(run)
        self.truncated_exps = set(payload.get("truncated_experiments") or [])
        self.listed_at = str(payload.get("listed_at", ""))
        self.rebuild_runs()
        # Listings written before the run-CSV lists moved to a file of their
        # own still carry them under "artifacts".
        # Stale: good to draw, not to add from -- see needs_listing.
        for run_id, v in {**(payload.get("artifacts") or {}), **(artifacts or {})}.items():
            if isinstance(v, dict) and run_id not in self.artifacts:
                self.artifacts[run_id] = {
                    "csvs": [_dc_from_dict(mlf.ArtifactFile, a) for a in v.get("csvs", [])],
                    "npzs": [_dc_from_dict(mlf.ArtifactFile, a) for a in v.get("npzs", [])],
                    "stale": True,
                }

    def mark_artifacts_stale(self) -> None:
        """What a refresh does to the run -> CSV lists: they stay drawn, and
        are listed again before anything is added from them."""
        for entry in self.artifacts.values():
            if "csvs" in entry:
                entry["stale"] = True

    def needs_listing(self, run_id: str) -> bool:
        """Whether `run_id`'s CSVs must be listed before a run is added or
        loaded from them. A list read from disk, or from before a refresh,
        is not enough while the server answers: the run may have logged its
        CSV since (it was still running), or re-logged it with another size
        -- and a workspace that recorded the old size never counts it as
        downloaded. With the server down, the old list is all there is."""
        entry = self.artifacts.get(run_id)
        if entry is None or entry.get("loading"):
            return True
        return bool(entry.get("stale")) and self.status != "down"


class HomeView:
    def __init__(self, ctx, workspace_panel, start_threads: bool = True):
        self.ctx = ctx
        self.workspace_panel = workspace_panel
        self.config = HomeConfig(fallback_uri=_default_tracking_uri())
        self.servers: dict = {}           # uri -> ServerState
        self.workspaces: list = []
        self.selected: set = set()        # run and csv keys
        self.expanded_runs: set = set()   # run keys
        self.expanded_workspaces: set = set()
        self._card_slots: dict = {}       # workspace id -> the Container its card is drawn in
        self._run_limit = RUN_PAGE        # rows of the runs table drawn (see RUN_PAGE)
        self._activated = False           # MLflow listing started (see on_shown)
        self._last_full_refresh = 0.0
        self._board_limits: dict = {}     # column id -> cards drawn in it (see BOARD_PAGE)
        self._debounce_timers: dict = {}
        self.downloading: dict = {}       # workspace id -> progress text
        self._lock = threading.RLock()
        self._start_threads = start_threads
        self._stop = threading.Event()
        self._sync_servers()

        # ---- left: runs ----
        self.servers_row = ft.Row(spacing=6, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.experiments_row = ft.Row(spacing=6, wrap=True, run_spacing=6,
                                      vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.search_field = ft.TextField(
            hint_text="Search run names and parameters", dense=True, border_radius=8,
            text_size=TEXT_SUB, expand=2, prefix_icon=ft.Icons.SEARCH,
            value=self.config.runs.get("search", ""), on_change=self._on_search_change,
        )
        self.filter_field = ft.TextField(
            hint_text="MLflow filter, e.g. metrics.val_acc > 0.9 and params.backbone = 'vit'",
            dense=True, border_radius=8, text_size=TEXT_SUB, expand=3,
            prefix_icon=ft.Icons.FILTER_ALT_OUTLINED,
            value=self.config.runs.get("filter", ""), on_submit=self._on_filter_submit,
            # Also on leaving the field: a filter typed and clicked away
            # from would otherwise sit there looking applied when it is not.
            on_blur=self._on_filter_submit,
            tooltip="MLflow's own search syntax, sent to the server. Applied on Enter or on leaving the field.",
        )
        self.unsuccessful_box = ft.Checkbox(
            label="Failed and killed", value=bool(self.config.runs.get("show_unsuccessful")),
            on_change=self._on_unsuccessful_change,
        )
        self.header_row = ft.Container()
        self.runs_list = ft.ListView(spacing=0, expand=True)
        # Given an explicit width (see _table_width) inside a sideways-
        # scrolling row, so the run names keep a readable minimum.
        self.runs_table = ft.Container(content=ft.Column([
            self.header_row,
            ft.Container(self.runs_list, expand=True, border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                         border_radius=8, padding=ft.padding.symmetric(vertical=2)),
        ], spacing=8, expand=True))
        self.runs_footer = ft.Row(spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.banner = ft.Column(spacing=6)

        # ---- right: workspaces ----
        self.workspace_filter = ft.TextField(
            hint_text="Filter workspaces", dense=True, border_radius=8, text_size=TEXT_SUB,
            prefix_icon=ft.Icons.SEARCH,
            on_change=lambda e: self._debounced("workspaces", self._render_workspaces),
        )
        self.board_row = ft.Row(spacing=10, scroll=ft.ScrollMode.AUTO, expand=True,
                                vertical_alignment=ft.CrossAxisAlignment.STRETCH)
        self.status = ft.Container()
        self.file_browser = ServerFileBrowser(ctx)
        self.root: Optional[ft.Control] = None

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------

    def build(self) -> ft.Control:
        left = ft.Column([
            ft.Row([
                ft.Text("Runs", weight=ft.FontWeight.BOLD, size=TEXT_TITLE),
                self.servers_row,
                ft.Container(expand=True),
                ft.IconButton(ft.Icons.ADD_LINK, tooltip="Add an MLflow server", on_click=self._on_add_server),
                ft.IconButton(ft.Icons.REFRESH, tooltip="Refresh every server", on_click=lambda e: self.refresh()),
                ft.IconButton(ft.Icons.VIEW_COLUMN_OUTLINED, tooltip="Choose columns",
                              on_click=self._on_columns),
                ft.IconButton(ft.Icons.DATA_OBJECT, tooltip="Edit the view settings (home_view.json)",
                              on_click=self._on_edit_config),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.banner,
            self.experiments_row,
            # Two lines rather than one: at a third of the page the filter
            # box, which holds a whole MLflow expression, needs the width.
            ft.Row([self.search_field, self.unsuccessful_box], spacing=8,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.filter_field], spacing=8),
            ft.Row([self.runs_table], scroll=ft.ScrollMode.AUTO, expand=True,
                   vertical_alignment=ft.CrossAxisAlignment.STRETCH),
            self.runs_footer,
        ], spacing=8, expand=RUNS_SHARE)

        right = ft.Container(expand=WORKSPACES_SHARE, content=ft.Column([
            ft.Row([
                ft.Text("Workspaces", weight=ft.FontWeight.BOLD, size=TEXT_TITLE),
                ft.Container(expand=True),
                ft.FilledButton("New", icon=ft.Icons.ADD, on_click=self._on_new_workspace,
                                style=ft.ButtonStyle(bgcolor=ACCENT)),
                ft.IconButton(ft.Icons.INVENTORY_2_OUTLINED, tooltip="Manage workspaces and snapshots",
                              on_click=lambda e: self.workspace_panel.open_panel()),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.workspace_filter,
            self.board_row,
            ft.Text("Drag runs or CSVs onto a workspace to add them, and a workspace onto a group "
                    "to move it.", size=TEXT_SMALL,
                    italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        ], spacing=8, expand=True))

        self.root = ft.Container(
            padding=ft.padding.only(left=16, right=16, top=10, bottom=10),
            expand=True,
            content=ft.Column([
                ft.Row([left, ft.VerticalDivider(width=1), right], spacing=14, expand=True,
                       vertical_alignment=ft.CrossAxisAlignment.STRETCH),
                self.status,
            ], spacing=6, expand=True),
        )
        self.reload_workspaces()
        self._render_all()
        if self._start_threads:
            threading.Thread(target=self._watch_loop, daemon=True).start()
        return self.root

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    def _status(self, message: str, kind: str = "success") -> None:
        self.status.content = status_row(message, kind) if message else None
        bp.safe_update(self.status)

    def _open_url(self, url: str) -> None:
        # Always a new tab: navigating this one away to MLflow would drop
        # this session and everything it has loaded.
        try:
            self.ctx.page.launch_url(url, web_window_name="_blank")
        except Exception as ex:  # noqa: BLE001
            self._status(f"Could not open {url}: {ex}", "error")

    def _sync_servers(self) -> None:
        """Make self.servers follow the config's server list."""
        wanted = {s["uri"]: s["name"] for s in self.config.servers}
        with self._lock:
            for uri in list(self.servers):
                if uri not in wanted:
                    del self.servers[uri]
            for uri, name in wanted.items():
                if uri in self.servers:
                    self.servers[uri].name = name
                else:
                    self.servers[uri] = ServerState(uri=uri, name=name)

    def reload_workspaces(self) -> None:
        self.workspaces = lib.list_workspaces()

    def on_shown(self) -> None:
        """Called by app.py each time Home comes on screen: a workspace may
        have been saved, opened or changed from the analysis tabs.

        The MLflow side only starts here, not when the page is built. Every
        workspace opens in a tab of its own, and each tab has a Home behind
        its analysis tabs: listing every shown experiment there too would
        compete with the workspace's own loading for the CPU -- measured
        tripling the read time of 300k-row CSVs -- and load the tracking
        server once more per open tab, for a page nobody is looking at.
        On a later visit it refreshes only if what it shows has gone stale.
        """
        self.reload_workspaces()
        if self._start_threads and not self._activated:
            self._activated = True
            self._render_all()

            def start():
                # Off the click: reading a cached listing of thousands of
                # runs takes a moment, and Home should be on screen first.
                self._show_cached_listings()
                self._render_all()
                self.refresh()

            threading.Thread(target=start, daemon=True).start()
            return
        if self._start_threads and time.monotonic() - self._last_full_refresh > HOME_STALE_S:
            self.refresh()
        self._render_all()

    # ------------------------------------------------------------------
    # talking to servers
    # ------------------------------------------------------------------

    def _show_cached_listings(self) -> None:
        """Draw what each server held last time, right away, and let the
        refresh replace it as it comes in. Listing every run of a busy
        server takes seconds to tens of seconds -- most of it MLflow
        building the response -- and a Home page that is blank for that
        long every time it opens reads as broken."""
        for server in self.servers.values():
            cached = mlflow_cache.load_listing(server.uri)
            if cached:
                server.adopt_listing(cached, mlflow_cache.load_artifacts(server.uri))
                server.keep_only(self._wanted_ids(server))

    def _wanted_ids(self, server: ServerState) -> list:
        shown = set(self.config.shown_experiments(server.uri)) - set(self.config.hidden_experiments(server.uri))
        return [e.experiment_id for e in server.experiments if e.name in shown]

    def refresh(self, uri: Optional[str] = None, runs_only: bool = False, only: Optional[list] = None) -> None:
        """Fetch again. `only` limits it to those experiment ids of `uri`
        -- what showing one more experiment asks for."""
        if uri is None and only is None:
            self._last_full_refresh = time.monotonic()
        for server in list(self.servers.values()):
            if uri is None or server.uri == uri:
                if not runs_only:
                    server.mark_artifacts_stale()
                server.status = "loading"
                threading.Thread(target=self._fetch_worker, args=(server, runs_only, only), daemon=True).start()
        self._render_servers()

    def _fetch_worker(self, server: ServerState, runs_only: bool, only: Optional[list] = None) -> None:
        """List the experiments (unless `runs_only`), then the runs of the
        shown ones -- or of `only` -- one request per experiment, several at
        once, drawing each experiment as soon as it arrives.

        One request per experiment rather than one for them all: they can
        run side by side, and a server with several workers answers four at
        once about three times faster than one after the other. How many is
        a setting (home_view.json, "parallel_requests"): past the server's
        own worker count, more only queue up and slow everything down.
        """
        server.last_try = time.monotonic()
        try:
            if not runs_only or not server.experiments:
                server.experiments = mlf.list_experiments(server.uri)
                self.config.adopt_defaults(server.uri, server.experiments)
            wanted = self._wanted_ids(server)
            with self._lock:
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
                    with self._lock:
                        # Checked against what is shown *now*: a refresh of
                        # a hundred experiments takes seconds, and one the
                        # user has unticked meanwhile must not come back.
                        # Same for a filter changed meanwhile: its refresh
                        # is on the way, and these runs, landing last, would
                        # replace its runs with the old filter's.
                        if (experiment_id not in self._wanted_ids(server)
                                or filter_string != self.config.runs.get("filter", "")):
                            continue
                        server.runs_by_exp[experiment_id] = runs
                        server.truncated_exps.discard(experiment_id)
                        if truncated:
                            server.truncated_exps.add(experiment_id)
                        server.rebuild_runs()
                    if time.monotonic() - last_draw > PROGRESSIVE_DRAW_S:
                        last_draw = time.monotonic()
                        self._render_experiments()
                        self._render_runs()
            if first_error is not None:
                raise first_error
            with self._lock:
                server.listed_at = datetime.now().isoformat(timespec="seconds")
                server.status, server.error, server.offline = "ok", "", False
            mlflow_cache.save_listing(server.uri, server.listing_payload())
        except mlf.MlflowError as ex:
            self._server_down(server, str(ex))
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            self._server_down(server, f"Unexpected error: {ex}")
        self._render_all()

    def _server_down(self, server: ServerState, error: str) -> None:
        """A filter the server rejects is not the server being down: that
        keeps the runs it had and says what was wrong with the filter."""
        with self._lock:
            if "filter" in error.lower() and server.experiments and self.config.runs.get("filter"):
                server.status, server.error = "ok", f"The filter was rejected: {error}"
                return
            server.status, server.error = "down", error
            if not server.runs:
                cached = mlflow_cache.load_listing(server.uri)
                if cached:
                    server.adopt_listing(cached, mlflow_cache.load_artifacts(server.uri))
            server.offline = bool(server.runs)

    def _fetch_artifacts(self, server: ServerState, run_id: str, then=None) -> None:
        entry = server.artifacts.get(run_id)
        if entry is not None and "csvs" in entry and not server.needs_listing(run_id):
            if then:
                then()
            return
        if server.status == "down":
            server.artifacts[run_id] = {"error": "server unreachable"}
            self._render_runs()
            if then:
                then()
            return
        server.artifacts[run_id] = {"loading": True}

        def work():
            try:
                csvs, npzs = mlf.list_csv_and_npz_artifacts(server.uri, run_id)
                server.artifacts[run_id] = {"csvs": csvs, "npzs": npzs}
                # Its own small file: rewriting the whole runs listing for
                # this -- tens of MB on a busy server -- took over half a
                # second per expanded run.
                mlflow_cache.save_artifacts(server.uri, server.artifacts_payload())
            except mlf.MlflowError as ex:
                server.artifacts[run_id] = {"error": str(ex)}
            self._render_runs()
            if then:
                then()

        threading.Thread(target=work, daemon=True).start()

    def _watch_loop(self) -> None:
        """Pick up hand edits of home_view.json, and retry servers that are down."""
        while not self._stop.wait(CONFIG_POLL_S):
            try:
                if not self._activated:
                    continue
                if self.config.changed_on_disk():
                    self.config.reload()
                    self._sync_servers()
                    self.search_field.value = self.config.runs.get("search", "")
                    self.filter_field.value = self.config.runs.get("filter", "")
                    self.unsuccessful_box.value = bool(self.config.runs.get("show_unsuccessful"))
                    bp.safe_update(self.search_field, self.filter_field, self.unsuccessful_box)
                    self.refresh()
                    continue
                now = time.monotonic()
                for server in list(self.servers.values()):
                    if server.status == "down" and now - server.last_try > RETRY_DOWN_S:
                        server.status = "loading"
                        threading.Thread(target=self._fetch_worker, args=(server, False), daemon=True).start()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _render_all(self) -> None:
        self._render_servers()
        self._render_experiments()
        self._render_runs()
        self._render_workspaces()

    def _render_servers(self) -> None:
        self.servers_row.controls.clear()
        for server in self.servers.values():
            color = {"ok": ft.Colors.GREEN, "down": ft.Colors.RED, "loading": ft.Colors.AMBER}.get(
                server.status, ft.Colors.ON_SURFACE_VARIANT)
            tip = server.uri + (f"\n{server.error}" if server.error else "")
            chip = ft.Container(
                content=ft.Row([
                    ft.Container(width=8, height=8, border_radius=4, bgcolor=color),
                    ft.Text(server.name, size=TEXT_SUB),
                ], spacing=6, tight=True),
                padding=ft.padding.symmetric(horizontal=10, vertical=4),
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT), border_radius=12,
            )
            self.servers_row.controls.append(ft.PopupMenuButton(
                content=chip, tooltip=tip,
                items=[
                    ft.PopupMenuItem(text="Refresh", icon=ft.Icons.REFRESH,
                                     on_click=lambda e, u=server.uri: self.refresh(u)),
                    ft.PopupMenuItem(text="Open in MLflow", icon=ft.Icons.OPEN_IN_NEW,
                                     on_click=lambda e, u=server.uri: self._open_url(u + "/")),
                    ft.PopupMenuItem(text="Remove this server", icon=ft.Icons.LINK_OFF,
                                     on_click=lambda e, u=server.uri: self._remove_server(u)),
                ],
            ))
        self.banner.controls.clear()
        if not self.servers:
            self.banner.controls.append(ft.Row([
                info_banner("No MLflow server yet. Add one to list its runs here -- workspaces work without one."),
                ft.OutlinedButton("Add a server", icon=ft.Icons.ADD_LINK, on_click=self._on_add_server),
            ], spacing=10))
        if self.config.error:
            self.banner.controls.append(info_banner(self.config.error, icon=ft.Icons.WARNING_AMBER_OUTLINED))
        for server in self.servers.values():
            if server.status == "down":
                when = f" -- showing the listing from {server.listed_at.replace('T', ' ')}" if server.offline else ""
                self.banner.controls.append(info_banner(
                    f"{server.name} is unreachable{when}. Retrying every minute.",
                    icon=ft.Icons.CLOUD_OFF))
            elif server.error:
                self.banner.controls.append(info_banner(server.error, icon=ft.Icons.WARNING_AMBER_OUTLINED))
        bp.safe_update(self.servers_row, self.banner)

    def _render_experiments(self) -> None:
        """The shown experiments as chips -- a few of them -- and a button
        to the full list.

        Every experiment used to be a chip. With a hundred on a server that
        filled the whole runs panel and pushed the runs table off screen, so
        the chips are now only what is shown (clicking one hides its runs),
        capped at MAX_EXPERIMENT_CHIPS, and choosing among all of them is
        the Experiments picker's job (see _open_experiment_picker).
        """
        row = self.experiments_row
        row.controls.clear()
        several = len(self.servers) > 1
        for server in self.servers.values():
            if not server.experiments:
                continue
            shown = set(self.config.shown_experiments(server.uri)) - set(self.config.hidden_experiments(server.uri))
            row.controls.append(ft.Text(f"{server.name}:" if several else "Experiments:", size=TEXT_SMALL,
                                        color=ft.Colors.ON_SURFACE_VARIANT))
            ordered = sorted((e for e in server.experiments if e.name in shown), key=lambda e: e.name.lower())
            for exp in ordered[:MAX_EXPERIMENT_CHIPS]:
                row.controls.append(self._experiment_chip(server, exp, True))
            if len(ordered) > MAX_EXPERIMENT_CHIPS:
                row.controls.append(ft.Container(
                    content=ft.Text(f"+{len(ordered) - MAX_EXPERIMENT_CHIPS} more", size=TEXT_SUB),
                    padding=ft.padding.symmetric(horizontal=8, vertical=3), border_radius=12,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                    tooltip="\n".join(e.name for e in ordered[MAX_EXPERIMENT_CHIPS:MAX_EXPERIMENT_CHIPS + 30]),
                    on_click=lambda e, srv=server: self._open_experiment_picker(srv),
                ))
            row.controls.append(ft.TextButton(
                f"Experiments ({len(ordered)}/{len(server.experiments)})", icon=ft.Icons.CHECKLIST,
                tooltip="Choose which experiments' runs are listed",
                on_click=lambda e, srv=server: self._open_experiment_picker(srv),
            ))
        bp.safe_update(row)

    def _open_experiment_picker(self, server: ServerState) -> None:
        """Every experiment of a server, searchable, with a tick for those
        whose runs are listed; hidden ones in a section of their own.

        Applying fetches only what was newly ticked, and drops what was
        unticked without asking the server anything (see _apply_experiments).
        """
        shown = set(self.config.shown_experiments(server.uri))
        hidden = set(self.config.hidden_experiments(server.uri))
        # Most recently updated first: that is where the runs worth looking
        # at usually are, and it matches how the first ones shown are chosen.
        experiments = sorted(server.experiments, key=lambda e: -(e.last_update_time or 0))
        boxes = {e.name: ft.Checkbox(value=e.name in shown and e.name not in hidden) for e in experiments}
        hide_flags = {e.name: e.name in hidden for e in experiments}
        query = ft.TextField(hint_text="Search experiments", dense=True, border_radius=8, text_size=TEXT_SUB,
                             prefix_icon=ft.Icons.SEARCH, autofocus=True)
        listing = ft.ListView(spacing=0, expand=True)
        count_text = ft.Text("", size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)

        def row_for(exp) -> ft.Control:
            is_hidden = hide_flags[exp.name]
            when = _format_time(exp.last_update_time)
            return ft.Row([
                boxes[exp.name] if not is_hidden else ft.Container(width=40),
                ft.Column([
                    ft.Text(exp.name, size=TEXT_ROW, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                            color=ft.Colors.ON_SURFACE_VARIANT if is_hidden else None),
                    ft.Text(f"updated {when}" if when else exp.experiment_id, size=TEXT_SMALL,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                ], spacing=0, tight=True, expand=True),
                ft.IconButton(ft.Icons.VISIBILITY_OUTLINED if is_hidden else ft.Icons.VISIBILITY_OFF_OUTLINED,
                              icon_size=16, tooltip="Show in this list again" if is_hidden else
                              "Hide from this list (and from the table)",
                              on_click=lambda e, n=exp.name: toggle_hidden(n)),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        def render(e=None):
            q = (query.value or "").strip().lower()
            matching = [x for x in experiments if not q or q in x.name.lower()]
            visible = [x for x in matching if not hide_flags[x.name]]
            hidden_rows = [x for x in matching if hide_flags[x.name]]
            listing.controls = [row_for(x) for x in visible[:bp.MAX_RENDERED_ROWS]]
            if len(visible) > bp.MAX_RENDERED_ROWS:
                listing.controls.append(bp.overflow_note(bp.MAX_RENDERED_ROWS, len(visible)))
            if hidden_rows:
                listing.controls.append(bp.group_header(f"Hidden ({len(hidden_rows)})"))
                listing.controls.extend(row_for(x) for x in hidden_rows[:bp.MAX_RENDERED_ROWS])
            count_text.value = f"{sum(1 for b in boxes.values() if b.value)} ticked of {len(experiments)}"
            bp.safe_update(listing, count_text)

        def toggle_hidden(name):
            hide_flags[name] = not hide_flags[name]
            if hide_flags[name]:
                boxes[name].value = False
            render()

        def set_all(value):
            q = (query.value or "").strip().lower()
            for x in experiments:
                if not hide_flags[x.name] and (not q or q in x.name.lower()):
                    boxes[x.name].value = value
            render()

        for box in boxes.values():
            box.on_change = lambda e: (setattr(count_text, "value", f"{sum(1 for b in boxes.values() if b.value)} "
                                                                    f"ticked of {len(experiments)}"),
                                       bp.safe_update(count_text))
        query.on_change = lambda e: self._debounced("experiment-picker", render)

        def apply(e):
            self.ctx.page.close(dialog)
            self._apply_experiments(
                server, {n for n, b in boxes.items() if b.value}, {n for n, h in hide_flags.items() if h})

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Experiments on {server.name}", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=560, height=520, content=ft.Column([
                query,
                ft.Row([count_text, ft.Container(expand=True),
                        ft.TextButton("Tick all matching", on_click=lambda e: set_all(True)),
                        ft.TextButton("Untick all matching", on_click=lambda e: set_all(False))], spacing=4),
                listing,
            ], spacing=8)),
        )
        dialog.actions = [ft.FilledButton("Apply", on_click=apply, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(dialog))]
        render()
        self.ctx.page.open(dialog)

    def _apply_experiments(self, server: ServerState, shown: set, hidden: set) -> None:
        before = set(self._wanted_ids(server))
        self.config.set_experiments(server.uri, shown, hidden)
        with self._lock:
            server.keep_only(self._wanted_ids(server))
        self._render_experiments()
        self._render_runs()
        new = [i for i in self._wanted_ids(server) if i not in before]
        if new:
            self.refresh(server.uri, runs_only=True, only=new)

    def _experiment_chip(self, server: ServerState, exp, shown: bool) -> ft.Control:
        runs = server.runs_by_exp.get(exp.experiment_id)
        count = len(runs) if shown and runs is not None else None
        label = exp.name + (f" · {count}" if count is not None else "")
        return ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.CHECK if shown else ft.Icons.ADD, size=13,
                        color=ACCENT if shown else ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(label, size=TEXT_SUB),
                ft.Container(
                    content=ft.Icon(ft.Icons.VISIBILITY_OFF_OUTLINED, size=13, color=ft.Colors.ON_SURFACE_VARIANT),
                    tooltip="Hide this experiment from the list",
                    on_click=lambda e, u=server.uri, n=exp.name: self._set_experiment_hidden(u, n, True),
                ),
            ], spacing=4, tight=True),
            padding=ft.padding.symmetric(horizontal=8, vertical=3),
            border_radius=12,
            border=ft.border.all(1, ACCENT if shown else ft.Colors.OUTLINE_VARIANT),
            bgcolor=ft.Colors.with_opacity(0.12, ACCENT) if shown else None,
            tooltip=(exp.description or exp.name) + "\nClick to show or hide its runs.",
            on_click=lambda e, u=server.uri, n=exp.name: self._toggle_experiment(u, n),
        )

    # ---- runs table ----

    def _visible_runs(self, server: ServerState) -> list:
        hidden = self.config.hidden_run_ids(server.uri)
        shown_names = set(self.config.shown_experiments(server.uri)) - set(self.config.hidden_experiments(server.uri))
        shown_ids = {e.experiment_id for e in server.experiments if e.name in shown_names}
        query = (self.search_field.value or "").strip().lower()
        show_unsuccessful = bool(self.unsuccessful_box.value)
        out = []
        for run in server.runs:
            if run.run_id in hidden:
                continue
            if server.experiments and run.experiment_id not in shown_ids:
                continue
            if not show_unsuccessful and run.status in UNSUCCESSFUL_STATUSES:
                continue
            if query and not self._run_matches(run, query):
                continue
            out.append(run)
        return out

    @staticmethod
    def _run_matches(run: mlf.Run, query: str) -> bool:
        haystack = " ".join([run.run_name, run.run_id, run.description,
                             *(f"{k}={v}" for k, v in run.params.items())]).lower()
        return all(word in haystack for word in query.split())

    def _membership(self) -> tuple:
        """Which workspaces hold each run and each CSV -- the badges."""
        by_run: dict = {}
        by_csv: dict = {}
        for ws in self.workspaces:
            for e in ws.mount.experiments:
                if not e.from_mlflow():
                    continue
                uri = wss.normalize_uri(e.mlflow_uri)
                by_run.setdefault((uri, e.mlflow_run_id), [])
                if ws not in by_run[(uri, e.mlflow_run_id)]:
                    by_run[(uri, e.mlflow_run_id)].append(ws)
                by_csv.setdefault((uri, e.mlflow_run_id, e.artifact_path), []).append(ws)
        return by_run, by_csv

    def _render_runs(self) -> None:
        with self._lock:
            self._render_runs_locked()

    def _table_width(self) -> int:
        """The runs panel's width, or wider when the columns would squeeze
        the run names below RUN_NAME_MIN_WIDTH -- then it scrolls."""
        page_width = getattr(self.ctx.page, "width", None) or 1440
        panel = (page_width - PAGE_PADDING) * RUNS_SHARE / (RUNS_SHARE + WORKSPACES_SHARE)
        columns = len(self.config.columns) * (COLUMN_WIDTH + 8) + BADGES_WIDTH + 8
        return int(max(panel, ROW_FIXED_WIDTH + columns + RUN_NAME_MIN_WIDTH))

    def on_window_resized(self) -> None:
        self._render_runs()

    def _render_runs_locked(self) -> None:
        columns = self.config.columns
        self.runs_table.width = self._table_width()
        self.header_row.content = ft.Container(
            padding=ft.padding.only(left=8, right=8),
            content=ft.Row([
                ft.Container(width=64),
                ft.Text("Run", size=TEXT_SMALL, weight=ft.FontWeight.BOLD, expand=True,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                *[ft.Text(column_label(c), size=TEXT_SMALL, weight=ft.FontWeight.BOLD, width=COLUMN_WIDTH,
                          color=ft.Colors.ON_SURFACE_VARIANT, tooltip=c["key"], no_wrap=True,
                          overflow=ft.TextOverflow.ELLIPSIS) for c in columns],
                ft.Text("Workspaces", size=TEXT_SMALL, weight=ft.FontWeight.BOLD, width=BADGES_WIDTH,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Container(width=64),
            ], spacing=8),
        )
        self.runs_list.controls.clear()
        by_run, by_csv = self._membership()
        several = len(self.servers) > 1
        rendered = total = 0
        for server in self.servers.values():
            runs = self._visible_runs(server)
            total += len(runs)
            current_exp = None
            for run in runs:
                if rendered >= self._run_limit:
                    break
                if run.experiment_id != current_exp:
                    current_exp = run.experiment_id
                    self.runs_list.controls.append(self._experiment_header(server, run.experiment_id, several))
                self.runs_list.controls.append(self._run_row(server, run, by_run))
                rendered += 1
                if run_key(server.uri, run.run_id) in self.expanded_runs:
                    self.runs_list.controls.extend(self._artifact_rows(server, run, by_csv))
        if not total:
            loading = any(s.status == "loading" for s in self.servers.values())
            self.runs_list.controls.append(ft.Container(padding=16, content=ft.Text(
                "Loading runs..." if loading else
                ("No run matches." if any(s.runs for s in self.servers.values())
                 else "No runs to show. Pick an experiment above."),
                size=TEXT_SUB, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)))
        if total > rendered:
            self.runs_list.controls.append(self._show_more(
                rendered, total, RUN_PAGE, lambda e: self._more_runs()))
        self._render_footer(total)
        bp.safe_update(self.runs_table)

    def _experiment_header(self, server: ServerState, experiment_id: str, several: bool) -> ft.Control:
        exp = server.experiment_by_id(experiment_id)
        name = exp.name if exp else experiment_id
        if several:
            name = f"{server.name} / {name}"
        return ft.Container(
            padding=ft.padding.only(left=10, top=8, bottom=2, right=8),
            content=ft.Row([
                ft.Icon(ft.Icons.SCIENCE_OUTLINED, size=14, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(name, size=TEXT_SMALL, weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.IconButton(ft.Icons.OPEN_IN_NEW, icon_size=13, tooltip="Open this experiment in MLflow",
                              style=ft.ButtonStyle(padding=0), width=22, height=22,
                              on_click=lambda e: self._open_url(mlf.experiment_ui_url(server.uri, experiment_id))),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _badges(self, workspaces: list) -> ft.Control:
        return ft.Row([
            ft.Container(
                content=ft.Text(ws.label()[:12], size=10, color=ft.Colors.BLACK, no_wrap=True),
                bgcolor=workspace_color(ws.id), border_radius=4,
                padding=ft.padding.symmetric(horizontal=5, vertical=1), tooltip=f"In workspace {ws.label()}",
            ) for ws in workspaces
        ], spacing=3, wrap=False, width=BADGES_WIDTH, scroll=ft.ScrollMode.HIDDEN)

    def _run_row(self, server: ServerState, run: mlf.Run, by_run: dict) -> ft.Control:
        key = run_key(server.uri, run.run_id)
        expanded = key in self.expanded_runs
        tooltip = "\n".join(filter(None, [
            run.run_name or run.run_id, f"{run.status} -- started {_format_time(run.start_time)}",
            f"by {run.user}" if run.user else "", run.description,
        ]))
        row = ft.Container(
            padding=ft.padding.only(left=4, right=8, top=1, bottom=1),
            opacity=0.55 if server.offline else 1.0,
            bgcolor=ft.Colors.with_opacity(0.10, ACCENT) if key in self.selected else None,
            content=ft.Row([
                ft.IconButton(ft.Icons.EXPAND_MORE if expanded else ft.Icons.CHEVRON_RIGHT, icon_size=16,
                              width=26, height=26, style=ft.ButtonStyle(padding=0),
                              tooltip="Show its CSVs", on_click=lambda e: self._toggle_run(server, run)),
                ft.Checkbox(value=key in self.selected, on_change=lambda e: self._toggle_selected(key),
                            width=24, height=24),
                ft.Container(width=8, height=8, border_radius=4,
                             bgcolor=STATUS_COLORS.get(run.status, ft.Colors.ON_SURFACE_VARIANT)),
                ft.Text(run.label(), size=TEXT_ROW, expand=True, no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS, tooltip=tooltip),
                *[ft.Text(column_value(run, c), size=TEXT_SUB, width=COLUMN_WIDTH, no_wrap=True,
                          overflow=ft.TextOverflow.ELLIPSIS, tooltip=column_value(run, c) or None)
                  for c in self.config.columns],
                self._badges(by_run.get((server.uri, run.run_id), [])),
                ft.IconButton(ft.Icons.OPEN_IN_NEW, icon_size=15, width=28, height=28,
                              style=ft.ButtonStyle(padding=0), tooltip="Open this run in MLflow",
                              on_click=lambda e: self._open_url(
                                  mlf.run_ui_url(server.uri, run.experiment_id, run.run_id))),
                ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, icon_size=16, width=28, height=28, items=[
                    ft.PopupMenuItem(text="Add to a workspace...", icon=ft.Icons.DRIVE_FILE_MOVE_OUTLINE,
                                     on_click=lambda e: self._on_add_to(self._drag_keys(key))),
                    ft.PopupMenuItem(text="Hide this run", icon=ft.Icons.VISIBILITY_OFF_OUTLINED,
                                     on_click=lambda e: self._hide_run(server, run)),
                    ft.PopupMenuItem(text="Copy run id", icon=ft.Icons.COPY,
                                     on_click=lambda e: self._copy(run.run_id)),
                ]),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )
        return self._draggable(row, key, run.label())

    def _artifact_rows(self, server: ServerState, run: mlf.Run, by_csv: dict) -> list:
        entry = server.artifacts.get(run.run_id) or {"loading": True}

        def note(text: str, color=ft.Colors.ON_SURFACE_VARIANT) -> ft.Control:
            return ft.Container(padding=ft.padding.only(left=72, top=2, bottom=4),
                                content=ft.Text(text, size=TEXT_SUB, italic=True, color=color))

        if entry.get("loading"):
            return [note("Listing its artifacts...")]
        if "error" in entry:
            return [note(f"Could not list its artifacts: {entry['error']}", ft.Colors.RED)]
        candidates = wss.candidates_for_run(server.uri, run, entry["csvs"], entry["npzs"])
        if not candidates:
            return [note("No CSV in this run's artifacts.")]
        rows = []
        for c in candidates:
            key = csv_key(server.uri, run.run_id, c.csv.path)
            sidecars = [name for name, art in (("embeddings", c.embeddings), ("prototypes", c.prototypes)) if art]
            cached = mlflow_cache.is_cached(run.run_id, c.csv.path, c.csv.size)
            row = ft.Container(
                padding=ft.padding.only(left=34, right=8, top=1, bottom=1),
                opacity=0.55 if server.offline else 1.0,
                bgcolor=ft.Colors.with_opacity(0.10, ACCENT) if key in self.selected else None,
                content=ft.Row([
                    ft.Checkbox(value=key in self.selected, on_change=lambda e, k=key: self._toggle_selected(k),
                                width=24, height=24),
                    ft.Icon(ft.Icons.TABLE_CHART_OUTLINED, size=15, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text(c.csv.path, size=TEXT_SUB, expand=True, no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS, tooltip=c.csv.path),
                    ft.Text(" + ".join(sidecars), size=TEXT_SMALL, color=ACCENT,
                            tooltip="Sidecars logged next to it -- downloaded with it") if sidecars else ft.Container(),
                    ft.Text(_human_size(c.csv.size), size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Icon(ft.Icons.CIRCLE, size=9, color=ft.Colors.GREEN,
                            tooltip="Already downloaded") if cached else ft.Container(width=9),
                    self._badges(by_csv.get((server.uri, run.run_id, c.csv.path), [])),
                    ft.IconButton(ft.Icons.OPEN_IN_NEW, icon_size=15, width=28, height=28,
                                  style=ft.ButtonStyle(padding=0), tooltip="Open this artifact in MLflow",
                                  on_click=lambda e, p=c.csv.path: self._open_url(
                                      mlf.run_ui_url(server.uri, run.experiment_id, run.run_id, p))),
                    ft.Container(width=28),
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            )
            rows.append(self._draggable(row, key, Path(c.csv.path).name))
        return rows

    def _draggable(self, content: ft.Control, key: str, label: str) -> ft.Control:
        return ft.Draggable(
            group=DRAG_GROUP, data=key, content=content,
            content_feedback=ft.Container(
                content=ft.Text(self._drag_label(key, label), size=TEXT_SUB, color=ft.Colors.ON_PRIMARY),
                bgcolor=ACCENT, border_radius=6, padding=ft.padding.symmetric(horizontal=10, vertical=5),
            ),
        )

    def _drag_label(self, key: str, label: str) -> str:
        n = len(self._drag_keys(key))
        return label if n == 1 else f"{n} items"

    def _drag_keys(self, key: str) -> list:
        """What a drag starting on `key` carries: the whole selection when
        the row is part of it, the row alone otherwise -- as in a file
        manager."""
        return sorted(self.selected) if key in self.selected else [key]

    def _render_footer(self, total: int) -> None:
        self.runs_footer.controls.clear()
        truncated = any(s.truncated for s in self.servers.values())
        bits = [f"{total:,} run(s)"]
        if truncated:
            bits.append("listing capped -- narrow it with a filter")
        hidden = len(self.config.runs.get("hidden", []))
        self.runs_footer.controls.append(ft.Text("  ·  ".join(bits), size=TEXT_SMALL,
                                                 color=ft.Colors.ON_SURFACE_VARIANT))
        if hidden:
            self.runs_footer.controls.append(ft.TextButton(
                f"Show the {hidden} hidden run(s)", on_click=lambda e: self._unhide_runs()))
        self.runs_footer.controls.append(ft.Container(expand=True))
        if self.selected:
            self.runs_footer.controls.extend([
                ft.Text(f"{len(self.selected)} selected", size=TEXT_SUB),
                ft.OutlinedButton("Add to a workspace...", icon=ft.Icons.DRIVE_FILE_MOVE_OUTLINE,
                                  on_click=lambda e: self._on_add_to(sorted(self.selected))),
                ft.TextButton("Clear", on_click=lambda e: self._clear_selection()),
            ])
        bp.safe_update(self.runs_footer)

    # ---- workspaces ----

    def _render_workspaces(self) -> None:
        """The board: a column per group, each a drop target for workspaces,
        each workspace card a drop target for runs."""
        query = (self.workspace_filter.value or "").strip().lower()

        def shown(w) -> bool:
            return not query or query in " ".join(
                [w.name, w.description, " ".join(w.tags), w.mount.dataset_path]).lower()

        board = bd.load()
        self.board_row.controls.clear()
        self._card_slots = {}
        for index, (column, items) in enumerate(bd.layout(board, self.workspaces)):
            empty_note = ("No workspace yet. Create one with New, then drag runs onto it."
                          if index == 0 and not self.workspaces else "Drop workspaces here.")
            self.board_row.controls.append(
                self._board_column(board, column, [w for w in items if shown(w)], len(items), empty_note))
        self.board_row.controls.append(ft.Container(
            content=ft.IconButton(ft.Icons.ADD, tooltip="Add a group", on_click=self._on_add_column),
            alignment=ft.alignment.top_center, width=48,
        ))
        bp.safe_update(self.board_row)

    def _show_more(self, shown: int, total: int, page: int, on_click) -> ft.Control:
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            content=ft.Row([
                ft.Text(f"{shown:,} of {total:,} shown", size=TEXT_SMALL, italic=True,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                ft.TextButton(f"Show {min(page, total - shown):,} more", on_click=on_click),
            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _more_runs(self) -> None:
        self._run_limit += RUN_PAGE
        self._render_runs()

    def _more_cards(self, column_id: str) -> None:
        self._board_limits[column_id] = self._board_limits.get(column_id, BOARD_PAGE) + BOARD_PAGE
        self._render_workspaces()

    def _board_column(self, board: dict, column: dict, items: list, total: int, empty_note: str) -> ft.Control:
        limit = self._board_limits.get(column["id"], BOARD_PAGE)
        cards = [self._card_slot(ws) for ws in items[:limit]]
        if len(items) > limit:
            cards.append(self._show_more(limit, len(items), BOARD_PAGE,
                                         lambda e: self._more_cards(column["id"])))
        if not cards:
            cards = [ft.Container(padding=10, content=ft.Text(
                empty_note if total == 0 else "Nothing here matches the filter.",
                size=TEXT_SUB, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))]
        header = ft.Row([
            ft.Text(column["name"], size=TEXT_HEADING, weight=ft.FontWeight.BOLD, expand=True, no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS, tooltip=column["name"]),
            ft.Container(content=ft.Text(str(total), size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                         padding=ft.padding.symmetric(horizontal=7, vertical=1), border_radius=8,
                         bgcolor=ft.Colors.with_opacity(0.10, ft.Colors.ON_SURFACE)),
            ft.IconButton(ft.Icons.ADD, icon_size=16, width=30, height=30, style=ft.ButtonStyle(padding=0),
                          tooltip="New workspace in this group",
                          on_click=lambda e: self._on_new_workspace(e, column)),
            ft.PopupMenuButton(icon=ft.Icons.MORE_HORIZ, icon_size=16, width=30, height=30, items=[
                ft.PopupMenuItem(text="Rename...", icon=ft.Icons.EDIT_OUTLINED,
                                 on_click=lambda e: self._on_rename_column(column)),
                ft.PopupMenuItem(text="Remove this group...", icon=ft.Icons.DELETE_OUTLINE,
                                 on_click=lambda e: self._on_remove_column(column)),
            ]),
        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        box = ft.Container(
            width=BOARD_COLUMN_WIDTH, padding=8, border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.ON_SURFACE),
            border=ft.border.all(1, ft.Colors.TRANSPARENT),
            content=ft.Column([
                header,
                ft.Column(cards, spacing=8, scroll=ft.ScrollMode.AUTO, expand=True,
                          horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            ], spacing=8, expand=True, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
        )

        def will_accept(e):
            box.border = ft.border.all(1.5, ACCENT)
            box.bgcolor = ft.Colors.with_opacity(0.10, ACCENT)
            bp.safe_update(box)

        def leave(e):
            box.border = ft.border.all(1, ft.Colors.TRANSPARENT)
            box.bgcolor = ft.Colors.with_opacity(0.04, ft.Colors.ON_SURFACE)
            bp.safe_update(box)

        return ft.DragTarget(group=WORKSPACE_DRAG_GROUP, content=box, on_will_accept=will_accept,
                             on_leave=leave, on_accept=lambda e: self._on_workspace_drop(column, e))

    def _card_slot(self, ws: lib.Workspace) -> ft.Control:
        """A card inside a slot that can be redrawn on its own.

        Expanding a card, or a download ticking forward, changes one card.
        Redrawing the whole board for it cost a fifth of a second per event
        with a few hundred workspaces -- three times a second for the length
        of a download -- and sent every card to the browser again.
        """
        slot = ft.Container(content=self._workspace_card(ws))
        self._card_slots[ws.id] = slot
        return slot

    def _refresh_card(self, workspace_id: str) -> None:
        slot = self._card_slots.get(workspace_id)
        ws = next((w for w in self.workspaces if w.id == workspace_id), None)
        if slot is None or ws is None or slot.page is None:
            self._render_workspaces()
            return
        slot.content = self._workspace_card(ws)
        bp.safe_update(slot)

    def _drag_handle(self, ws: lib.Workspace) -> ft.Control:
        """The one part of a card that drags it to another column.

        The whole card used to be draggable, and a click meant to expand it
        and a drag meant to move it started the same way -- the card kept
        moving when the user wanted its details. A handle makes the two
        different gestures in different places.

        Its drag group differs from the runs' (WORKSPACE_DRAG_GROUP vs
        DRAG_GROUP), so a run dropped on a card never moves it, and a card
        dropped on another card lands in that card's column.
        """
        return ft.Draggable(
            group=WORKSPACE_DRAG_GROUP, data=ws.id,
            content=ft.GestureDetector(
                mouse_cursor=ft.MouseCursor.GRAB,
                # Claims the tap, so clicking the handle without dragging
                # does not also expand the card underneath it.
                on_tap=lambda e: None,
                # Three short lines, drawn rather than taken from the icon
                # font: the stock menu/drag glyphs are either two lines or
                # three long ones. The padding is the grab area, larger
                # than the lines themselves.
                content=ft.Container(
                    content=ft.Column(
                        [ft.Container(width=HANDLE_LINE_WIDTH, height=2, border_radius=1,
                                      bgcolor=ft.Colors.ON_SURFACE_VARIANT) for _ in range(3)],
                        spacing=3, tight=True,
                    ),
                    padding=ft.padding.symmetric(horizontal=5, vertical=10),
                    tooltip="Drag to move this workspace to another group",
                ),
            ),
            content_feedback=ft.Container(
                content=ft.Text(ws.label(), size=TEXT_SUB, color=ft.Colors.ON_PRIMARY),
                bgcolor=ACCENT, border_radius=6, padding=ft.padding.symmetric(horizontal=10, vertical=5),
            ),
        )

    def _on_workspace_drop(self, column: dict, e) -> None:
        try:
            workspace_id = getattr(self.ctx.page.get_control(e.src_id), "data", None)
        except Exception:  # noqa: BLE001
            workspace_id = None
        if workspace_id:
            bd.move(workspace_id, column["id"], known_ids=[w.id for w in self.workspaces])
        self._render_workspaces()

    def _on_add_column(self, e) -> None:
        name = ft.TextField(label="Group name", value=bd.next_default_name(bd.load()), dense=True,
                            border_radius=8, autofocus=True, text_size=TEXT_SUB)

        def add(ev):
            self.ctx.page.close(dialog)
            bd.add_column(name.value or "")
            self._render_workspaces()

        name.on_submit = add
        dialog = ft.AlertDialog(modal=True, title=ft.Text("Add a group", weight=ft.FontWeight.BOLD),
                                content=ft.Container(width=380, content=name))
        dialog.actions = [ft.FilledButton("Add", on_click=add, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _on_rename_column(self, column: dict) -> None:
        name = ft.TextField(label="Group name", value=column["name"], dense=True, border_radius=8,
                            autofocus=True, text_size=TEXT_SUB)

        def save(ev):
            self.ctx.page.close(dialog)
            bd.rename_column(column["id"], name.value or "")
            self._render_workspaces()

        name.on_submit = save
        dialog = ft.AlertDialog(modal=True, title=ft.Text("Rename group", weight=ft.FontWeight.BOLD),
                                content=ft.Container(width=380, content=name))
        dialog.actions = [ft.FilledButton("Save", on_click=save, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _on_remove_column(self, column: dict) -> None:
        board = bd.load()
        target = bd.neighbour(board, column["id"])
        if target is None:
            self._status("The last group can't be removed -- workspaces need somewhere to be.", "warning")
            return
        count = next((len(items) for c, items in bd.layout(board, self.workspaces) if c["id"] == column["id"]), 0)
        body = (f"Its {count} workspace(s) move to \"{target['name']}\". No workspace is deleted."
                if count else "It is empty.")
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text(f"Remove the group \"{column['name']}\"?", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=420, content=ft.Text(body, size=TEXT_SUB)),
        )

        def yes(ev):
            self.ctx.page.close(dialog)
            bd.remove_column(column["id"], workspaces=self.workspaces)
            self._render_workspaces()
            self._status(f"Removed \"{column['name']}\"." + (f" {count} workspace(s) moved to "
                                                              f"\"{target['name']}\"." if count else ""),
                         "neutral")

        dialog.actions = [ft.TextButton("Remove", on_click=yes, style=ft.ButtonStyle(color=ft.Colors.RED)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _workspace_card(self, ws: lib.Workspace) -> ft.Control:
        expanded = ws.id in self.expanded_workspaces
        current = getattr(self.ctx, "current_workspace", None)
        is_open = current is not None and current.id == ws.id
        summary = wss.download_summary(ws)
        state = summary.state
        n = len(ws.mount.experiments)
        progress = self.downloading.get(ws.id)

        status_controls: list = []
        if progress is not None:
            status_controls.append(ft.ProgressRing(width=14, height=14, stroke_width=2, color=ACCENT))
        elif state in DOWNLOAD_ICONS:
            icon, color, tip = DOWNLOAD_ICONS[state]
            status_controls.append(ft.Icon(icon, size=11, color=color, tooltip=tip))
            if state != wss.DOWNLOAD_ALL:
                size = summary.missing_bytes
                what = f"{summary.missing} file(s)" + (f", {_human_size(size)}" if size else "")
                status_controls.append(ft.IconButton(
                    ft.Icons.DOWNLOAD, icon_size=17, width=30, height=30, style=ft.ButtonStyle(padding=0),
                    tooltip=f"Download what this workspace needs ({what})",
                    on_click=lambda e: self._download(ws)))

        dataset = Path(ws.mount.dataset_path).name if ws.mount.dataset_path else "no dataset yet"
        snapshots = lib.list_snapshots(ws) if expanded else None
        subtitle = f"{ws.id}  ·  {dataset}"
        header = ft.Row([
            ft.Container(width=10, height=10, border_radius=5, bgcolor=workspace_color(ws.id)),
            ft.Text(ws.label(), size=TEXT_ROW, weight=ft.FontWeight.BOLD, expand=True, no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS, tooltip=ws.description or ws.label()),
            ft.Container(content=ft.Text("open", size=10, color=ft.Colors.ON_PRIMARY), bgcolor=ACCENT,
                         border_radius=4, padding=ft.padding.symmetric(horizontal=5, vertical=1),
                         tooltip="This is the workspace loaded in the analysis tabs") if is_open else ft.Container(),
            ft.Text(f"{n} exp", size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            *status_controls,
            ft.Icon(ft.Icons.EXPAND_LESS if expanded else ft.Icons.EXPAND_MORE, size=18),
        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        body: list = [
            ft.Container(content=header, on_click=lambda e: self._toggle_workspace(ws)),
            ft.Text(progress if progress is not None else subtitle, size=TEXT_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                    tooltip=ws.mount.dataset_path or None),
        ]
        if expanded:
            body.append(ft.Divider(height=1))
            if not ws.mount.experiments:
                body.append(ft.Text("Empty -- drop runs or CSVs here.", size=TEXT_SUB, italic=True,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
            for e in ws.mount.experiments:
                body.append(self._workspace_experiment_row(ws, e))
            if snapshots:
                body.append(ft.Text("Snapshots", size=TEXT_SMALL, weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
                for snap in snapshots[:6]:
                    body.append(ft.Row([
                        ft.Icon(ft.Icons.BOOKMARK_OUTLINE, size=14, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Text(snap.label(), size=TEXT_SUB, expand=True, no_wrap=True,
                                overflow=ft.TextOverflow.ELLIPSIS, tooltip=snap.id),
                        ft.IconButton(ft.Icons.OPEN_IN_NEW, icon_size=16, width=26, height=26,
                                      style=ft.ButtonStyle(padding=0), tooltip="Open this snapshot in a new tab",
                                      on_click=lambda e, s=snap: self._open_workspace(ws, s)),
                    ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER))
                if len(snapshots) > 6:
                    body.append(ft.Text(f"... and {len(snapshots) - 6} more (Manage)", size=TEXT_SMALL,
                                        italic=True, color=ft.Colors.ON_SURFACE_VARIANT))
            body.append(ft.Row([
                ft.FilledButton("Open", icon=ft.Icons.OPEN_IN_NEW, on_click=lambda e: self._open_workspace(ws),
                                style=ft.ButtonStyle(bgcolor=ACCENT), tooltip="Open in a new tab"),
                ft.TextButton("Dataset...", icon=ft.Icons.DATASET_OUTLINED,
                              on_click=lambda e: self._on_set_dataset(ws)),
                ft.PopupMenuButton(icon=ft.Icons.MORE_HORIZ, items=[
                    ft.PopupMenuItem(text="Rename...", icon=ft.Icons.EDIT_OUTLINED,
                                     on_click=lambda e: self._on_rename(ws)),
                    ft.PopupMenuItem(text="Details, export, snapshots...", icon=ft.Icons.INVENTORY_2_OUTLINED,
                                     on_click=lambda e: self.workspace_panel.open_panel(select=ws)),
                    ft.PopupMenuItem(text="Delete...", icon=ft.Icons.DELETE_OUTLINE,
                                     on_click=lambda e: self._on_delete(ws)),
                ]),
            ], spacing=6, wrap=True))

        card = ft.Container(
            # The handle sits on the card's left edge, outside the content,
            # centred on its height.
            content=ft.Row([
                self._drag_handle(ws),
                ft.Column(body, spacing=4, tight=True, expand=True),
            ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.only(left=2, right=10, top=8, bottom=8),
            border_radius=10,
            border=ft.border.all(1.5 if is_open else 1, ACCENT if is_open else ft.Colors.OUTLINE_VARIANT),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        )

        def will_accept(e, c=card):
            c.border = ft.border.all(2, ACCENT)
            c.bgcolor = ft.Colors.with_opacity(0.14, ACCENT)
            bp.safe_update(c)

        def leave(e, c=card):
            c.border = ft.border.all(1.5 if is_open else 1, ACCENT if is_open else ft.Colors.OUTLINE_VARIANT)
            c.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGHEST
            bp.safe_update(c)

        return ft.DragTarget(
            group=DRAG_GROUP, content=card,
            on_will_accept=will_accept, on_leave=leave,
            on_accept=lambda e: self._on_drop(ws, e),
        )

    def _workspace_experiment_row(self, ws: lib.Workspace, e: lib.MountExperiment) -> ft.Control:
        if e.from_mlflow():
            downloaded = wss.is_downloaded(e)
            server = self.servers.get(wss.normalize_uri(e.mlflow_uri))
            where = f"{e.mlflow_run_name or e.mlflow_run_id[:8]} / {e.artifact_path}"
            if server is None:
                where += f"  ({e.mlflow_uri})"
            lead = ft.Icon(ft.Icons.SCIENCE_OUTLINED, size=14, color=ft.Colors.ON_SURFACE_VARIANT,
                           tooltip="From an MLflow run")
            state = ft.Icon(ft.Icons.CIRCLE if downloaded else ft.Icons.CIRCLE_OUTLINED, size=9,
                            color=ft.Colors.GREEN if downloaded else ft.Colors.ON_SURFACE_VARIANT,
                            tooltip="Downloaded" if downloaded else "Not downloaded yet")
            link = ft.IconButton(
                ft.Icons.OPEN_IN_NEW, icon_size=14, width=24, height=24, style=ft.ButtonStyle(padding=0),
                tooltip="Open this run in MLflow",
                on_click=lambda ev: self._open_url(mlf.run_ui_url(
                    e.mlflow_uri, e.mlflow_experiment_id, e.mlflow_run_id, e.artifact_path)))
        else:
            where = e.path
            lead = ft.Icon(ft.Icons.INSERT_DRIVE_FILE_OUTLINED, size=14, color=ft.Colors.ON_SURFACE_VARIANT,
                           tooltip="A local file")
            state = ft.Icon(ft.Icons.CIRCLE if Path(e.path).is_file() else ft.Icons.ERROR_OUTLINE, size=9,
                            color=ft.Colors.GREEN if Path(e.path).is_file() else ft.Colors.RED,
                            tooltip="On disk" if Path(e.path).is_file() else "Missing from disk")
            link = ft.Container(width=24)
        return ft.Row([
            lead,
            ft.Column([
                ft.Text(e.name, size=TEXT_SUB, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                ft.Text(where, size=10, color=ft.Colors.ON_SURFACE_VARIANT, no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS, tooltip=where),
            ], spacing=0, tight=True, expand=True),
            state, link,
            ft.IconButton(ft.Icons.CLOSE, icon_size=14, width=24, height=24, style=ft.ButtonStyle(padding=0),
                          tooltip="Remove from this workspace (the run and its files are not touched)",
                          on_click=lambda ev: self._remove_experiment(ws, e.name)),
        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # ------------------------------------------------------------------
    # user actions: filters and visibility
    # ------------------------------------------------------------------

    def _on_search_change(self, e) -> None:
        # Debounced: each keystroke would otherwise redraw up to
        # MAX_RUN_ROWS rows and rewrite home_view.json -- which every other
        # open tab then notices and reloads.
        def apply():
            self.config.set_run_option("search", self.search_field.value or "")
            self._run_limit = RUN_PAGE
            self._render_runs()

        self._debounced("runs", apply)

    def _on_filter_submit(self, e) -> None:
        value = (self.filter_field.value or "").strip()
        if value == self.config.runs.get("filter", "") and e is not None and e.name == "blur":
            return
        self.config.set_run_option("filter", value)
        for server in self.servers.values():
            server.error = ""
        self.refresh(runs_only=True)

    def _on_unsuccessful_change(self, e) -> None:
        self.config.set_run_option("show_unsuccessful", bool(self.unsuccessful_box.value))
        self._render_runs()

    def _toggle_experiment(self, uri: str, name: str) -> None:
        """Showing an experiment fetches its runs and nothing else; hiding
        one fetches nothing -- its runs are simply dropped from the table.
        Both used to re-fetch every shown experiment, which with thirty of
        them shown was seven seconds per click."""
        self.config.toggle_experiment(uri, name)
        self._after_experiment_change(uri, name)

    def _set_experiment_hidden(self, uri: str, name: str, hidden: bool) -> None:
        self.config.set_experiment_hidden(uri, name, hidden)
        self._after_experiment_change(uri, name)

    def _after_experiment_change(self, uri: str, name: str) -> None:
        server = self.servers.get(uri)
        if server is None:
            return
        with self._lock:
            server.keep_only(self._wanted_ids(server))
        self._render_experiments()
        self._render_runs()
        experiment = next((e for e in server.experiments if e.name == name), None)
        if experiment is not None and experiment.experiment_id in self._wanted_ids(server):
            self.refresh(uri, runs_only=True, only=[experiment.experiment_id])

    def _hide_run(self, server: ServerState, run: mlf.Run) -> None:
        self.config.set_run_hidden(server.uri, run.run_id, run.label(), True)
        self.selected.discard(run_key(server.uri, run.run_id))
        self._render_runs()
        self._status(f"Hid {run.label()}. The footer can show hidden runs again.", "neutral")

    def _unhide_runs(self) -> None:
        self.config.unhide_all_runs()
        self._render_runs()

    def _toggle_run(self, server: ServerState, run: mlf.Run) -> None:
        key = run_key(server.uri, run.run_id)
        if key in self.expanded_runs:
            self.expanded_runs.discard(key)
        else:
            self.expanded_runs.add(key)
            self._fetch_artifacts(server, run.run_id)
        self._render_runs()

    def _toggle_selected(self, key: str) -> None:
        if key in self.selected:
            self.selected.discard(key)
        else:
            self.selected.add(key)
        self._render_runs()

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._render_runs()

    def _toggle_workspace(self, ws: lib.Workspace) -> None:
        if ws.id in self.expanded_workspaces:
            self.expanded_workspaces.discard(ws.id)
        else:
            self.expanded_workspaces.add(ws.id)
        self._refresh_card(ws.id)

    def _debounced(self, name: str, fn, delay: float = bp.FILTER_DEBOUNCE_S) -> None:
        """Run `fn` once typing pauses: a filter redrawing hundreds of rows
        or cards on every keystroke is what makes typing lag."""
        timer = self._debounce_timers.get(name)
        if timer is not None:
            timer.cancel()
        timer = threading.Timer(delay, fn)
        timer.daemon = True
        self._debounce_timers[name] = timer
        timer.start()

    def _copy(self, text: str) -> None:
        from lookin.ui.components import copy_to_clipboard
        copy_to_clipboard(self.ctx, text)
        self._status(f"Copied {text}.", "neutral")

    # ------------------------------------------------------------------
    # adding runs to a workspace
    # ------------------------------------------------------------------

    def _on_drop(self, ws: lib.Workspace, e) -> None:
        try:
            src = self.ctx.page.get_control(e.src_id)
            key = getattr(src, "data", None)
        except Exception:  # noqa: BLE001
            key = None
        self._refresh_card(ws.id)   # clears the drop highlight
        if not key:
            return
        self.add_keys(ws, self._drag_keys(key))

    def _on_add_to(self, keys: list) -> None:
        """The same as a drop, for whoever would rather pick from a list."""
        if not self.workspaces:
            self._on_new_workspace(None)
            return
        dropdown = ft.Dropdown(
            label="Workspace", dense=True, border_radius=8, autofocus=True,
            options=[ft.dropdown.Option(w.id, w.label()) for w in self.workspaces],
            value=self.workspaces[0].id,
        )
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text(f"Add {len(keys)} item(s) to...", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=420, content=dropdown),
        )

        def go(e):
            self.ctx.page.close(dialog)
            ws = next((w for w in self.workspaces if w.id == dropdown.value), None)
            if ws is not None:
                self.add_keys(ws, keys)

        dialog.actions = [ft.FilledButton("Add", on_click=go, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def add_keys(self, ws: lib.Workspace, keys: list) -> None:
        """Add dragged runs and CSVs to `ws`. A run's CSVs have to be known
        first, so any run not expanded yet has its artifacts listed -- then
        this comes back round."""
        pending = []
        for key in keys:
            kind, uri, run_id, _ = parse_key(key)
            server = self.servers.get(uri)
            if server is None:
                continue
            if server.needs_listing(run_id):
                pending.append((server, run_id))
        if pending:
            self._status(f"Listing the artifacts of {len(pending)} run(s)...", "busy")
            remaining = {"n": len(pending)}

            def one_done():
                remaining["n"] -= 1
                if remaining["n"] == 0:
                    self.add_keys(ws, keys)

            for server, run_id in pending:
                server.artifacts.pop(run_id, None)
                self._fetch_artifacts(server, run_id, then=one_done)
            return

        direct, ambiguous, problems = [], [], []
        for key in keys:
            kind, uri, run_id, path = parse_key(key)
            server = self.servers.get(uri)
            run = server.run_by_id(run_id) if server else None
            if run is None:
                continue
            entry = server.artifacts.get(run_id, {})
            if "error" in entry:
                problems.append(f"{run.label()}: {entry['error']}")
                continue
            candidates = wss.candidates_for_run(uri, run, entry.get("csvs", []), entry.get("npzs", []))
            if kind == "csv":
                direct.extend(c for c in candidates if c.csv.path == path)
            elif not candidates:
                problems.append(f"{run.label()}: no CSV in its artifacts")
            elif len(candidates) == 1:
                direct.extend(candidates)
            else:
                ambiguous.extend(candidates)
        if ambiguous:
            self._pick_csvs(ws, direct, ambiguous, problems)
        else:
            self._commit_add(ws, direct, problems)

    def _pick_csvs(self, ws: lib.Workspace, direct: list, ambiguous: list, problems: list) -> None:
        """A dragged run with several CSVs: which of them?"""
        boxes = []
        rows: list = []
        current_run = None
        for c in ambiguous:
            if c.run.run_id != current_run:
                current_run = c.run.run_id
                rows.append(ft.Text(c.run.label(), weight=ft.FontWeight.BOLD, size=TEXT_ROW))
            box = ft.Checkbox(label=c.csv.path, value=False, data=c)
            boxes.append(box)
            rows.append(box)
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Which results CSVs?", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=620, height=min(420, 60 + 34 * len(rows)), content=ft.Column(
                [ft.Text("These runs logged several. Each one you tick becomes an experiment in "
                         f"{ws.label()}.", size=TEXT_SUB), *rows],
                spacing=4, scroll=ft.ScrollMode.AUTO)),
        )

        def go(e):
            self.ctx.page.close(dialog)
            self._commit_add(ws, direct + [b.data for b in boxes if b.value], problems)

        dialog.actions = [ft.FilledButton("Add", on_click=go, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _commit_add(self, ws: lib.Workspace, candidates: list, problems: list) -> None:
        # Applied to the manifest as it is on disk now, not to `ws`: this
        # page may have been drawn minutes ago, and another tab may have
        # added to the same workspace since (see library.update_workspace).
        outcome = {}

        def change(fresh):
            outcome["result"] = wss.add_candidates(fresh, candidates, self.config.dataset_param)
            return bool(outcome["result"].added)

        fresh = lib.update_workspace(ws.path, change)
        if fresh is None:
            self._status(f"Could not write {ws.label()} -- was it deleted?", "error")
            self.reload_workspaces()
            self._render_workspaces()
            return
        ws, result = fresh, outcome["result"]
        result.skipped.extend((p, "") for p in problems)
        self.selected.clear()
        self._after_workspace_change(ws)
        kind = "success" if result.added and not result.skipped else ("warning" if result.added else "error")
        message = result.message().replace(" ()", "")
        self._status(f"{ws.label()}: {message}", kind)

    def _remove_experiment(self, ws: lib.Workspace, name: str) -> None:
        fresh = lib.update_workspace(ws.path, lambda w: wss.remove_experiment(w, name))
        if fresh is not None:
            self._after_workspace_change(fresh)
            self._status(f"Removed {name} from {ws.label()}.", "neutral")

    def _after_workspace_change(self, ws: lib.Workspace) -> None:
        """Re-read the library, keep the open workspace object current, and
        repaint both halves -- the badges on the left follow the change."""
        self.expanded_workspaces.add(ws.id)
        current = getattr(self.ctx, "current_workspace", None)
        if current is not None and current.id == ws.id:
            # The workspace open in this tab: a dataset set here has to
            # reach its Data tab, not only its card.
            self.workspace_panel.follow_workspace(ws)
        self.reload_workspaces()
        self._render_runs()
        self._render_workspaces()

    # ------------------------------------------------------------------
    # downloading and opening
    # ------------------------------------------------------------------

    def _download(self, ws: lib.Workspace, then=None) -> None:
        if ws.id in self.downloading:
            return
        self.downloading[ws.id] = "Starting the download..."
        self._refresh_card(ws.id)
        if getattr(self.ctx, "busy", None) is not None:
            self.ctx.busy.begin(f"Downloading {ws.label()}")
        last = {"at": 0.0}

        def progress(index, count, name, done, total):
            now = time.monotonic()
            if now - last["at"] < 0.3:
                return
            last["at"] = now
            amount = f"{_human_size(done)} of {_human_size(total)}" if total else _human_size(done)
            self.downloading[ws.id] = f"Downloading {index}/{count}: {name} -- {amount}"
            self._refresh_card(ws.id)

        def work():
            try:
                fetched, errors = wss.download(ws, progress)
                # The files are in; record that on the manifest as it is
                # now, not on `ws`, which was read before a download that
                # may have taken minutes (see library.update_workspace).
                lib.update_workspace(ws.path, wss.point_at_cache)
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                fetched, errors = 0, [str(ex)]
            finally:
                self.downloading.pop(ws.id, None)
                if getattr(self.ctx, "busy", None) is not None:
                    self.ctx.busy.end()
            self.reload_workspaces()
            self._render_all()
            if errors:
                self._status(f"{ws.label()}: downloaded {fetched} file(s), {len(errors)} failed -- "
                             + "; ".join(errors)[:400], "error")
            else:
                self._status(f"{ws.label()}: downloaded {fetched} file(s).", "success")
                if then is not None:
                    then()

        threading.Thread(target=work, daemon=True).start()

    def _open_workspace(self, ws: lib.Workspace, snapshot=None) -> None:
        # A workspace with no dataset yet -- or one that is not on this
        # machine -- still opens: its tab lands on the Data tab, where the
        # dataset is set up, and the first one loaded there is written back
        # into the workspace (see WorkspacePanel.adopt_dataset).
        missing = wss.missing_files(ws)
        if not missing:
            self.workspace_panel.open_in_new_tab(ws, snapshot)
            return
        size = wss.missing_bytes(ws)
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Not downloaded yet", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=520, content=ft.Text(
                f"{len(missing)} file(s) of {ws.label()} are still on the MLflow server"
                + (f" ({_human_size(size)})." if size else ".")
                + " Download them first, or open the workspace without those experiments.",
                size=TEXT_SUB)),
        )

        def download_then_open(e):
            self.ctx.page.close(dialog)
            self._download(ws, then=lambda: self._offer_open(ws, snapshot))

        def open_without(e):
            # The new tab leaves out what is not downloaded by itself (see
            # WorkspacePanel.open_from_link) and says so there.
            self.ctx.page.close(dialog)
            self.workspace_panel.open_in_new_tab(ws, snapshot)

        dialog.actions = [
            ft.FilledButton("Download and open", icon=ft.Icons.DOWNLOAD, on_click=download_then_open,
                            style=ft.ButtonStyle(bgcolor=ACCENT)),
            ft.TextButton("Open without them", on_click=open_without),
            ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(dialog)),
        ]
        self.ctx.page.open(dialog)

    def _offer_open(self, ws: lib.Workspace, snapshot=None) -> None:
        """After "Download and open": the download is done, open on a click.

        Not opened directly, because a browser only lets a page open a tab
        in answer to a click it has just received -- and the click that
        asked for this was a download ago, so the tab would be silently
        blocked as a pop-up.
        """
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text("Downloaded", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=420, content=ft.Text(
                f"Everything {ws.label()} needs is on this machine.", size=TEXT_SUB)),
        )

        def go(e):
            self.ctx.page.close(dialog)
            self.workspace_panel.open_in_new_tab(ws, snapshot)

        dialog.actions = [ft.FilledButton("Open", icon=ft.Icons.OPEN_IN_NEW, on_click=go,
                                          style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Later", on_click=lambda e: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    # ------------------------------------------------------------------
    # workspace dialogs
    # ------------------------------------------------------------------

    def _dataset_field(self, value: str = "") -> tuple:
        field_ = ft.TextField(label="Dataset CSV (on this machine)", value=value, expand=True, dense=True,
                              border_radius=8, text_size=TEXT_SUB)
        browse = ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse",
                               on_click=lambda e: self.file_browser.open(
                                   on_pick=lambda p: (setattr(field_, "value", p), bp.safe_update(field_)),
                                   extensions={".csv"}, title="Which dataset CSV?"))
        return field_, ft.Row([field_, browse], spacing=6)

    def _on_new_workspace(self, e, column: Optional[dict] = None) -> None:
        """`column` is the board column the workspace goes in -- the one
        whose + was clicked. From the header's New, the first column is
        offered and the dropdown picks another."""
        board = bd.load()
        name = ft.TextField(label="Name", dense=True, border_radius=8, autofocus=True, text_size=TEXT_SUB)
        dataset, dataset_row = self._dataset_field()
        dataset.label = "Dataset CSV (optional)"
        # Keyed by column id, never by name: two columns may share a name.
        column_picker = ft.Dropdown(
            label="Group", dense=True, border_radius=8, text_size=TEXT_SUB,
            value=(column or board["columns"][0])["id"],
            options=[ft.dropdown.Option(c["id"], c["name"]) for c in board["columns"]],
            visible=len(board["columns"]) > 1,
        )
        add_selected = ft.Checkbox(label=f"Add the {len(self.selected)} selected item(s)", value=True,
                                   visible=bool(self.selected))
        error = ft.Text("", color=ft.Colors.RED, size=TEXT_SUB)
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text("New workspace", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=560, content=ft.Column([
                name, column_picker, dataset_row,
                ft.Text(f"Can be left empty: set it later in the Data tab of the open workspace, or let the "
                        f"first run added bring it (its '{self.config.dataset_param}' parameter).",
                        size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                add_selected, error,
            ], spacing=10, tight=True)),
        )

        def create(ev, open_it: bool = False):
            if not (name.value or "").strip():
                error.value = "Enter a name."
                bp.safe_update(error)
                return
            path = (dataset.value or "").strip()
            ws = lib.create_workspace(name.value.strip(), lib.Mount(dataset_path=path),
                                      hash_dataset=bool(path) and Path(path).is_file())
            self.ctx.page.close(dialog)
            if ws is None:
                self._status("Could not create the workspace.", "error")
                return
            self.expanded_workspaces.add(ws.id)
            self.reload_workspaces()
            bd.move(ws.id, column_picker.value, known_ids=[w.id for w in self.workspaces])
            self._render_workspaces()
            self._status(f"Created {ws.label()} ({ws.id}).")
            if add_selected.visible and add_selected.value and self.selected:
                self.add_keys(ws, sorted(self.selected))
            if open_it:
                # Still inside the click, which is what lets the browser
                # open the tab (see _offer_open). Runs added above that
                # finish after this land in the open tab on their own (see
                # WorkspacePanel.follow_workspace).
                self.workspace_panel.open_in_new_tab(ws)

        name.on_submit = create
        dialog.actions = [ft.FilledButton("Create and open", icon=ft.Icons.OPEN_IN_NEW,
                                          on_click=lambda ev: create(ev, open_it=True),
                                          style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.OutlinedButton("Create", on_click=create),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _on_set_dataset(self, ws: lib.Workspace) -> None:
        dataset, dataset_row = self._dataset_field(ws.mount.dataset_path)
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text(f"Dataset of {ws.label()}", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=560, content=ft.Column([
                dataset_row,
                ft.Text("Every experiment in a workspace is scored against this one dataset.",
                        size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=10, tight=True)),
        )

        def save(ev):
            path = (dataset.value or "").strip()
            digest = lib.hash_file(path) if path and Path(path).is_file() else ""
            self.ctx.page.close(dialog)

            def change(fresh):
                fresh.mount.dataset_path, fresh.dataset_hash = path, digest
                return True

            fresh = lib.update_workspace(ws.path, change)
            if fresh is not None:
                self._after_workspace_change(fresh)
                self._status(f"{fresh.label()} now uses {Path(path).name or 'no dataset'}.")

        dialog.actions = [ft.FilledButton("Save", on_click=save, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _on_rename(self, ws: lib.Workspace) -> None:
        name = ft.TextField(label="Name", value=ws.name, dense=True, border_radius=8, autofocus=True)
        dialog = ft.AlertDialog(modal=True, title=ft.Text("Rename workspace", weight=ft.FontWeight.BOLD),
                                content=ft.Container(width=420, content=name))

        def save(ev):
            new_name = (name.value or "").strip()
            self.ctx.page.close(dialog)
            if not new_name:
                return

            def change(fresh):
                fresh.name = new_name
                return True

            fresh = lib.update_workspace(ws.path, change)
            if fresh is not None:
                self._after_workspace_change(fresh)

        dialog.actions = [ft.FilledButton("Save", on_click=save, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _on_delete(self, ws: lib.Workspace) -> None:
        count = len(lib.list_snapshots(ws))
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Delete {ws.label()}" + (f" and its {count} snapshot(s)?" if count else "?"),
                          weight=ft.FontWeight.BOLD),
            content=ft.Text("The workspace directory is removed. The runs on MLflow, the dataset and the "
                            "downloaded files are not touched.", size=TEXT_SUB),
        )

        def yes(ev):
            self.ctx.page.close(dialog)
            if lib.delete_workspace(ws):
                self.reload_workspaces()
                self._render_all()
                self._status(f"Deleted {ws.label()}.", "neutral")
            else:
                self._status("Could not delete the workspace.", "error")

        dialog.actions = [ft.TextButton("Delete", on_click=yes, style=ft.ButtonStyle(color=ft.Colors.RED)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    # ------------------------------------------------------------------
    # servers, columns, config
    # ------------------------------------------------------------------

    def _on_add_server(self, e) -> None:
        uri = ft.TextField(label="Tracking URI", hint_text="http://mlflow.lab:5000", dense=True,
                           border_radius=8, autofocus=True, text_size=TEXT_SUB)
        name = ft.TextField(label="Short name (optional)", dense=True, border_radius=8, text_size=TEXT_SUB)
        error = ft.Text("", color=ft.Colors.RED, size=TEXT_SUB)
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text("Add an MLflow server", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=520, content=ft.Column([uri, name, error], spacing=10, tight=True)),
        )

        def add(ev):
            value = (uri.value or "").strip()
            if not value.startswith(("http://", "https://")):
                error.value = "Enter an http:// or https:// address."
                bp.safe_update(error)
                return
            if not self.config.add_server(value, name.value or ""):
                error.value = "That server is already in the list."
                bp.safe_update(error)
                return
            self.ctx.page.close(dialog)
            self._sync_servers()
            self.refresh(wss.normalize_uri(value))

        dialog.actions = [ft.FilledButton("Add", on_click=add, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _remove_server(self, uri: str) -> None:
        self.config.remove_server(uri)
        self._sync_servers()
        self.selected = {k for k in self.selected if parse_key(k)[1] != uri}
        self._render_all()

    def _on_columns(self, e) -> None:
        """Every parameter and metric seen in the listed runs, as checkboxes."""
        params, metrics = set(), set()
        for server in self.servers.values():
            for run in server.runs:
                params.update(run.params)
                metrics.update(run.metrics)
        chosen = [c["key"] for c in self.config.columns]
        keys = [PARAM_PREFIX + k for k in sorted(params)] + [METRIC_PREFIX + k for k in sorted(metrics)]
        keys += [k for k in chosen if k not in keys]
        boxes = [ft.Checkbox(label=k, value=k in chosen, data=k) for k in keys]
        filter_box = ft.TextField(hint_text="Filter", dense=True, border_radius=8, text_size=TEXT_SUB,
                                  prefix_icon=ft.Icons.SEARCH)
        listing = ft.Column(boxes, spacing=0, scroll=ft.ScrollMode.AUTO, expand=True)

        def on_filter(ev):
            q = (filter_box.value or "").lower()
            for b in boxes:
                b.visible = q in b.data.lower()
            bp.safe_update(listing)

        filter_box.on_change = on_filter
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text("Columns", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=520, height=480, content=ft.Column([
                filter_box,
                ft.Text("Labels and order can be changed in home_view.json.", size=TEXT_SMALL,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                listing if keys else info_banner("No runs listed yet -- nothing to choose from."),
            ], spacing=8)),
        )

        def save(ev):
            ticked = {b.data for b in boxes if b.value}
            order = [k for k in chosen if k in ticked] + [b.data for b in boxes if b.value and b.data not in chosen]
            self.config.set_columns(order)
            self.ctx.page.close(dialog)
            self._render_runs()

        dialog.actions = [ft.FilledButton("Save", on_click=save, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _on_edit_config(self, e) -> None:
        editor = ft.TextField(value=self.config.text(), multiline=True, min_lines=20, max_lines=20,
                              text_size=12, border_radius=8)
        error = ft.Text("", color=ft.Colors.RED, size=TEXT_SUB)
        dialog = ft.AlertDialog(
            modal=True, title=ft.Text("View settings", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=760, content=ft.Column([
                ft.Text(f"{self.config.path} -- this is the file itself: edit it here or in any editor, "
                        "Home follows either.", size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                        selectable=True),
                editor, error,
            ], spacing=8, tight=True)),
        )

        def save(ev):
            problem = self.config.replace_text(editor.value or "")
            if problem:
                error.value = problem
                bp.safe_update(error)
                return
            self.ctx.page.close(dialog)
            self._after_config_change()

        def reload(ev):
            self.config.reload()
            editor.value = self.config.text()
            error.value = self.config.error
            bp.safe_update(editor, error)

        dialog.actions = [ft.FilledButton("Save", on_click=save, style=ft.ButtonStyle(bgcolor=ACCENT)),
                          ft.TextButton("Reload from file", on_click=reload),
                          ft.TextButton("Cancel", on_click=lambda ev: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    def _after_config_change(self) -> None:
        self._sync_servers()
        self.search_field.value = self.config.runs.get("search", "")
        self.filter_field.value = self.config.runs.get("filter", "")
        self.unsuccessful_box.value = bool(self.config.runs.get("show_unsuccessful"))
        bp.safe_update(self.search_field, self.filter_field, self.unsuccessful_box)
        self.refresh()
