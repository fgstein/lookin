"""MLflow integration: a connection indicator (red/green dot) plus a browse
dialog for picking an experiment -> run ->
CSV artifact and loading it into the app exactly like a manually-added
experiment (see UploadView.add_experiment_from_path).

Opened from the Data tab's "Add from MLflow...": it loads one CSV into the
session on screen. Organising runs into workspaces is the Home page's job
(ui/home_view.py); this dialog is the quick way in for a one-off look. Its
connection indicator is kept up to date but no longer sits in the app bar
-- the Home page shows every server's state instead.

The dialog is laid out as four side-by-side panels: experiments, runs, CSV
artifacts, and a read-only detail panel for the selected run. That last one
exists because picking the right run out of a list of near-identical names
is the actual hard part of this dialog -- and MLflow already returns the
description, the params and the latest metrics in the *same* response as the
run list (see core/mlflow_client.list_runs), so showing them costs nothing.

Every server call the dialog makes runs on a worker thread: the tracking
server is remote by definition, and a slow one used to freeze the whole UI
with no feedback at all.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import flet as ft

from lookin.core import mlflow_client as mlf
from lookin.core import mlflow_cache
from lookin.core import prefs
from lookin.core.data_manager import sidecar_paths_for_csv
from lookin.ui.components import info_banner, status_row, ACCENT
# The generic list-browser machinery this dialog grew, now shared with the
# Workspaces window (see ui/browser_panel.py). The thin wrappers further down
# keep every call site here unchanged.
from lookin.ui import browser_panel as bp
from lookin.ui.browser_panel import (
    COLUMN_SPACING, FILTER_DEBOUNCE_S, MIN_COLUMNS_HEIGHT, MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH, TEXT_HEADING, TEXT_ROW, TEXT_SMALL, TEXT_SUB, TEXT_TITLE,
    ListRow as _ListRow,
    # Re-exported rather than used here: the dialog's own render cap is part
    # of what this module promises, and its tests read it from here.
    MAX_RENDERED_ROWS,
)

POLL_INTERVAL_S = 30.0   # how often the background thread re-tests the MLflow connection
TICK_S = 1.0             # heartbeat of that thread: elapsed counters tick this often
CONTACT_LABEL_EVERY_S = 5.0  # ... the "last contact" label only needs this

FALLBACK_TRACKING_URI = "http://127.0.0.1:5000"
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"


def _env_tracking_uri() -> str:
    """The tracking URI from the environment, or "" if there is nothing this
    panel can use.

    MLFLOW_TRACKING_URI is the variable MLflow's own client and CLI already
    read, so a machine set up for MLflow needs no configuring here. But it
    legitimately holds values this panel cannot talk to -- ``file:///...``
    for a local mlruns directory, ``databricks``, a bare path -- and the
    client here speaks the HTTP API only. Those are reported to the user
    rather than silently dialled and failed on.
    """
    value = os.environ.get(TRACKING_URI_ENV, "").strip()
    return value if value.startswith(("http://", "https://")) else ""


def _default_tracking_uri() -> str:
    return _env_tracking_uri() or FALLBACK_TRACKING_URI


# Shared with the Home page's workspaces (see core/mlflow_cache.py).
CACHE_DIR = mlflow_cache.artifacts_dir()

# ---- dialog geometry ----
# Sized from the actual page at open time rather than fixed: the app window
# is 1440x920 by default and the dialog used to claim 800x(220-tall lists),
# which is what made every run name ellipsize.
#
# The sizes below are what is left *after* the dialog's own chrome, and are
# never floored above what the page can actually give: asking for more than
# fits doesn't scroll, it collapses every expand-ing list to zero height, so
# a short browser window would show four empty boxes.

# Widths of the three list columns as a fraction of the content width (the
# details panel takes whatever is left), each with a floor so they stay
# usable rather than shrinking to a sliver on a narrow window.
COLUMN_FRACTIONS = (0.21, 0.26, 0.21)
COLUMN_MIN_WIDTHS = (170, 200, 170)

# The panels row is given an explicit height rather than left to expand:
# expanding children silently collapse to nothing when the fixed rows above
# and below them (connection, recents, load footer) already use up the
# dialog. Below MIN_COLUMNS_HEIGHT the outer column scrolls instead.
FIXED_ROWS_HEIGHT = 292

# ---- type scale ----
# Nothing below TEXT_SMALL: the panel used to mix 9, 10, 12 and 13pt, which
# is unreadable at arm's length on a high-DPI screen. The top of the scale
# came back down after that -- a dialog that is mostly lists reads as
# shouting when its rows are 14pt and its headings 15-18pt, and the extra
# line height cost about two visible rows per list.

# ---- render caps ----
# A tracking server can hold thousands of experiments, and every row here is
# a Container + Row + Column + Icon + two Texts that has to be built, then
# serialized to the client, on every keystroke in the filter box. Past a
# screenful or two nobody reads the rest -- they filter -- so only this many
# rows are built, and the list says what it is not showing.
# Params/metrics of one run, same reasoning: some runs log hundreds.
MAX_KV_ROWS = 60
# Filter boxes re-render on a timer rather than on every keystroke: typing
# "resnet" used to rebuild the whole run list six times.

STATUS_COLORS = {
    "FINISHED": ft.Colors.GREEN,
    "RUNNING": ft.Colors.BLUE,
    "SCHEDULED": ft.Colors.BLUE_GREY,
    "FAILED": ft.Colors.RED,
    "KILLED": ft.Colors.ORANGE,
}

# Runs that did not produce results, hidden by default: an experiment worth
# browsing usually has a long tail of crashed attempts, and they are never
# what the user came here to load. The checkbox says how many are hidden, so
# this is never a silent disappearance.
UNSUCCESSFUL_STATUSES = ("FAILED", "KILLED")

# Sort options. The first element of each pair is a *key* compared in
# _sorted_experiments/_sorted_runs -- only the second, the label, is display
# text (see the dropdown note in CLAUDE.md).
EXPERIMENT_SORTS = (
    ("name", "Name A-Z"),
    ("updated", "Recently updated"),
    ("created", "Recently created"),
)
RUN_SORTS = (
    ("recent", "Most recent"),
    ("name", "Name A-Z"),
    ("status", "Status"),
    ("duration", "Longest first"),
)


def _human_size(n: int | None) -> str:
    if n is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


def _format_time(ms: int | None) -> str:
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return ""


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_metric(value: float) -> str:
    """Compact but honest: integers stay integers, tiny/huge values go
    scientific, everything else keeps 4 significant digits.

    NaN and infinity are shown as-is rather than formatted. They are not
    edge cases here: a diverged loss, an AUC over a class with no positive
    sample, a metric logged before its first update -- MLflow stores those
    as NaN/Infinity and hands them back in runs/search. int() refuses both,
    so this used to raise while rendering the run details panel, which took
    down the whole selection with "cannot convert float NaN to integer".
    """
    if value != value:
        return "NaN"
    if value in (float("inf"), float("-inf")):
        return "+inf" if value > 0 else "-inf"
    magnitude = abs(value)
    if value == int(value) and magnitude < 1e9:
        return str(int(value))
    if magnitude >= 1e5 or 0 < magnitude < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _run_group(start_ms: int | None) -> str:
    """Coarse "when was this" bucket, used as a separator in the run list so
    the most recent handful is visually detached from the back catalogue."""
    if not start_ms:
        return "Undated"
    try:
        day = datetime.fromtimestamp(start_ms / 1000).date()
    except (OSError, OverflowError, ValueError):
        return "Undated"
    days = (datetime.now().date() - day).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return "This week"
    if days < 31:
        return "This month"
    return "Older"


def _format_last_contact(elapsed_s: float | None) -> str:
    if elapsed_s is None:
        return ""
    minutes = int(elapsed_s // 60)
    if minutes < 1:
        return "last contact: just now"
    if minutes < 60:
        return f"last contact: {minutes} min"
    hours = minutes // 60
    return f"last contact: {hours}h{minutes % 60:02d}"


MAX_STATUS_CHARS = 400


def _short(message: str) -> str:
    """Cap what goes into the footer's status row.

    Nothing here composes the message: it comes from whatever failed while
    reading the CSV, and pandas errors quote the offending data. A file that
    is not a CSV at all parses as one enormous column name, and the
    resulting "column not found, available: [...]" ran to tens of MB -- laid
    out as a single Text, that froze the page for good.
    """
    message = " ".join(message.split())
    if len(message) <= MAX_STATUS_CHARS:
        return message
    return message[:MAX_STATUS_CHARS] + " [...] (full message in the console)"


def _format_elapsed(seconds: float) -> str:
    """How long the thing the user is waiting for has been running. Shown
    next to every "Loading ..." line: a spinner says "something is
    happening", which is not the question after ten seconds -- "is this
    server slow or is this app wedged?" is, and only a number answers it."""
    if seconds < 1:
        return ""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def _format_transfer(done: int, total: int | None) -> str:
    """"12MB of 340MB (3%)", or just "12MB" when the server sent no length."""
    if not total:
        return _human_size(done)
    percent = min(100, int(done * 100 / total)) if total else 0
    return f"{_human_size(done)} of {_human_size(total)} ({percent}%)"


@dataclass
class _Task:
    """One thing the panel is waiting on the tracking server for.

    `texts` are the controls that spell the wait out where it is happening
    (the "Loading runs..." line inside the run list); the dialog title shows
    the innermost task as well, because the list the user is waiting on is
    often not the one they are looking at.
    """
    label: str
    started: float
    texts: list


class MlflowPanel:
    def __init__(self, ctx):
        self.ctx = ctx
        self.connected = False
        self.experiments: list[mlf.Experiment] = []
        self.runs: list[mlf.Run] = []
        self.artifacts: list[mlf.ArtifactFile] = []
        self.selected_experiment: mlf.Experiment | None = None
        self.selected_run: mlf.Run | None = None
        self.selected_artifact: mlf.ArtifactFile | None = None
        # .npz artifacts of the selected run, keyed by artifact path: used
        # both to flag which CSVs have sidecars and, after the download, to
        # fetch them -- one walk of the artifact tree serves both.
        self.npz_artifacts: dict = {}
        self.pref = prefs.load_mlflow_pref()

        # Rendered rows by key, per list: what lets a click move the
        # highlight instead of rebuilding the list (see _select_in_list).
        self._experiment_rows: dict[str, _ListRow] = {}
        self._run_rows: dict[str, _ListRow] = {}
        self._artifact_rows: dict[str, _ListRow] = {}
        self._filter_timers: dict[str, threading.Timer] = {}

        # Everything the panel is currently waiting on the server for, so
        # each of those waits can say how long it has been waiting.
        self._task_lock = threading.Lock()
        self._tasks: list[_Task] = []

        # Set before a selection is made programmatically, and consumed by
        # the worker that finishes loading that level: this is what lets one
        # click on a "recent" chip walk experiment -> run -> artifact.
        self._pending_experiment_id = self.pref.experiment_id
        self._pending_run_id = ""
        self._pending_artifact_path = ""
        self._restored = False

        # ---- app-bar indicator ----
        self.status_dot = ft.Container(width=10, height=10, border_radius=5, bgcolor=ft.Colors.RED)
        self.last_contact_text = ft.Text("", size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.indicator = ft.Container(
            content=ft.Column([
                ft.Row([self.status_dot, ft.Text("MLflow", size=13)], spacing=6),
                self.last_contact_text,
            ], spacing=0, tight=True),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=8,
            ink=True,
            tooltip="MLflow connection -- click to load an experiment from MLflow",
            on_click=self._open_panel,
        )

        # ---- background connection health check ----
        # Verifies the connection on its own, not just when the user opens
        # the dialog and clicks Connect -- a server that dies (or comes
        # back) mid-session should be reflected in the app-bar dot without
        # the user having to go check manually.
        self.last_contact: float | None = None  # time.monotonic() of the last successful contact
        self._poll_stop = threading.Event()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        # Stopped by app.py when the session closes (see stop_polling).

        # ---- connection controls ----
        # A URI the user typed here before wins: it was an explicit choice,
        # made in this app, and the environment is only the default.
        self.uri_field = ft.TextField(
            label="MLflow tracking URI",
            value=self.pref.tracking_uri or _default_tracking_uri(),
            expand=True, border_radius=8, text_size=TEXT_ROW,
            on_submit=self._on_connect,
            tooltip=f"Defaults to ${TRACKING_URI_ENV} when it holds an http(s) address, "
                    f"otherwise to {FALLBACK_TRACKING_URI}. Whatever you connect to here is "
                    f"remembered and used next time.",
        )
        self.connect_btn = ft.FilledButton("Connect", icon=ft.Icons.LINK, on_click=self._on_connect, style=ft.ButtonStyle(bgcolor=ACCENT))
        self.connection_status = ft.Container(content=status_row(self._initial_status_text(), "neutral"))

        # ---- recent picks ----
        self.recents_row = ft.Row([], spacing=8, wrap=True, run_spacing=6)
        self.recents_container = ft.Container(content=self.recents_row, visible=False)

        # ---- browse lists ----
        self.experiment_filter = ft.TextField(
            label="Filter", dense=True, border_radius=8, text_size=TEXT_SUB,
            prefix_icon=ft.Icons.SEARCH, on_change=self._on_experiment_filter_change,
        )
        self.experiment_sort = self._sort_dropdown(EXPERIMENT_SORTS, self._on_experiment_sort_change)
        self.experiments_list = ft.ListView(expand=True, spacing=2)

        self.run_filter = ft.TextField(
            label="Filter", dense=True, border_radius=8, text_size=TEXT_SUB,
            prefix_icon=ft.Icons.SEARCH, on_change=self._on_run_filter_change,
            tooltip="Matches the run name, id, status, description and parameters. "
                    "Write part of a parameter name, '=', and part of its value "
                    "(batch_size=64) to match on that parameter alone.",
        )
        self.run_sort = self._sort_dropdown(RUN_SORTS, self._on_run_sort_change)
        self.show_unsuccessful = ft.Checkbox(
            label="Show failed / killed", value=False, on_change=self._on_run_sort_change,
            label_style=ft.TextStyle(size=TEXT_SUB),
        )
        self.hidden_runs_text = ft.Text("", size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.runs_list = ft.ListView(expand=True, spacing=2)

        self.artifact_filter = ft.TextField(
            label="Filter", dense=True, border_radius=8, text_size=TEXT_SUB,
            prefix_icon=ft.Icons.SEARCH, on_change=self._on_artifact_filter_change,
        )
        self.artifacts_list = ft.ListView(expand=True, spacing=2)
        self.details_column = ft.Column([], spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)

        # ---- "the panel is waiting on the server" indicator ----
        # In the dialog title, because that is the one part of the dialog
        # that never scrolls away, and because the wait is rarely where the
        # user is looking: picking a run kicks off an artifact walk three
        # panels to the right.
        self.busy_label = ft.Text("", size=TEXT_SUB, color=ACCENT, weight=ft.FontWeight.W_500)
        self.busy_row = ft.Row([
            ft.ProgressRing(width=13, height=13, stroke_width=2, color=ACCENT),
            self.busy_label,
        ], spacing=8, visible=False, tight=True)

        # ---- load footer ----
        self.load_name_field = ft.TextField(label="Experiment name in Lookin", expand=True, border_radius=8, text_size=TEXT_ROW)
        self.load_btn = ft.FilledButton("Load into Data tab", icon=ft.Icons.DOWNLOAD_ROUNDED, on_click=self._on_load_click, disabled=True, style=ft.ButtonStyle(bgcolor=ACCENT))
        # Determinate whenever the server sends a Content-Length, which is
        # the difference between "this is 40% done" and a spinner that says
        # nothing at all while 400MB of embeddings come down.
        self.download_bar = ft.ProgressBar(
            visible=False, value=0, height=4, color=ACCENT,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST)
        self.panel_status = ft.Container(content=None)

        # Widths are assigned by _resize_to_page() every time the dialog opens.
        self._column_panels = [
            self._list_panel(1, "Experiment", [self.experiment_filter], self.experiments_list,
                             trailing=self.experiment_sort),
            self._list_panel(2, "Run", [
                self.run_filter,
                ft.Row([self.show_unsuccessful, self.hidden_runs_text], spacing=4,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], self.runs_list, trailing=self.run_sort),
            self._list_panel(3, "CSV artifact", [self.artifact_filter], self.artifacts_list),
        ]

        # The height goes on the Container, not the Row: a Row inside a
        # Column sizes itself from its children and ignores its own height,
        # so the four panels would keep collapsing. Height is assigned by
        # _resize_to_page() alongside the column widths.
        self.panels_box = ft.Container(
            height=MIN_COLUMNS_HEIGHT,
            content=ft.Row(
                [*self._column_panels, self._details_panel()],
                spacing=COLUMN_SPACING,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        )

        self.content_container = ft.Container(
            width=MIN_DIALOG_WIDTH,
            height=MIN_DIALOG_HEIGHT,
            content=ft.Column(
                [
                    ft.Row([self.uri_field, self.connect_btn], spacing=10),
                    self.connection_status,
                    self.recents_container,
                    ft.Divider(height=1),
                    self.panels_box,
                    ft.Divider(height=1),
                    ft.Row([self.load_name_field, self.load_btn], spacing=10),
                    self.download_bar,
                    self.panel_status,
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.HUB_OUTLINED, color=ACCENT, size=19),
                ft.Text("Load an experiment from MLflow", weight=ft.FontWeight.BOLD, size=TEXT_TITLE),
                ft.Container(expand=True),
                self.busy_row,
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            content=self.content_container,
            actions=[ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.dialog))],
        )
        # Fill in the empty states now, so the first open shows "Select an
        # experiment" / "Select a run" rather than three blank boxes.
        self._render_recents()
        self._render_experiments()
        self._render_runs()
        self._render_artifacts()
        self._render_details()

    # ---- layout helpers ----
    def _sort_dropdown(self, options: tuple, on_change) -> ft.Dropdown:
        return bp.sort_dropdown(options, on_change)

    def _panel_header(self, step, title, trailing=None) -> ft.Control:
        return bp.panel_header(step, title, trailing)

    def _list_panel(self, step, title: str, above: list, listview: ft.ListView,
                    width=None, trailing=None) -> ft.Control:
        return bp.list_panel(step, title, above, listview, width, trailing)

    def _details_panel(self) -> ft.Control:
        # Kept as an attribute because the panel shows the experiment before
        # a run is picked, and the heading has to say which one it is. It
        # sits *beside* details_column rather than inside it, so every
        # update of the column has to name it too or the heading goes stale
        # ("Experiment details" over a panel that is showing a run).
        self.details_title = ft.Text("Run details", weight=ft.FontWeight.BOLD, size=TEXT_HEADING)
        return bp.details_panel(self.details_title, self.details_column)

    def _resize_to_page(self):
        bp.fit_dialog_to_page(
            self.ctx.page, self.content_container, self.panels_box,
            self._column_panels, COLUMN_FRACTIONS, COLUMN_MIN_WIDTHS, FIXED_ROWS_HEIGHT,
        )

    @staticmethod
    @staticmethod
    def _scroll_into_view(listview: ft.ListView, key: str) -> None:
        bp.scroll_into_view(listview, key)

    @staticmethod
    @staticmethod
    def _safe_update(*controls) -> None:
        bp.safe_update(*controls)

    def _begin_task(self, label: str, *texts: ft.Text) -> None:
        with self._task_lock:
            self._tasks.append(_Task(label=label, started=time.monotonic(), texts=list(texts)))
        self._render_busy()

    def _end_task(self, label: str) -> None:
        with self._task_lock:
            for i, task in enumerate(self._tasks):
                if task.label == label:
                    self._tasks.pop(i)
                    break
        self._render_busy()

    def _busy(self) -> bool:
        with self._task_lock:
            return bool(self._tasks)

    def _render_busy(self) -> None:
        with self._task_lock:
            current = self._tasks[-1] if self._tasks else None
        self.busy_row.visible = current is not None
        self.busy_label.value = f"{current.label}..." if current else ""
        self._safe_update(self.busy_row)

    def _tick_tasks(self) -> None:
        """Once a second, from the poll thread: refresh every "<what>... 12s"
        counter. Only controls whose text actually changed are pushed."""
        with self._task_lock:
            tasks = list(self._tasks)
        if not tasks:
            return
        now = time.monotonic()
        dirty: list[ft.Control] = []
        for task in tasks:
            value = f"{task.label}... {_format_elapsed(now - task.started)}".rstrip()
            for text in task.texts:
                if text.value != value:
                    text.value = value
                    dirty.append(text)
        innermost = f"{tasks[-1].label}... {_format_elapsed(now - tasks[-1].started)}".rstrip()
        if self.busy_label.value != innermost:
            self.busy_label.value = innermost
            dirty.append(self.busy_label)
        self._safe_update(*dirty)

    def _show_loading(self, listview: ft.ListView, label: str) -> ft.Text:
        """Put a spinner + label in a list and hand back the label, so
        _begin_task can keep an elapsed counter running on it."""
        text = ft.Text(f"{label}...", size=TEXT_SUB, italic=True,
                       color=ft.Colors.ON_SURFACE_VARIANT, expand=True)
        listview.controls = [ft.Container(
            content=ft.Row([ft.ProgressRing(width=14, height=14, stroke_width=2, color=ACCENT), text],
                           spacing=10),
            padding=10,
        )]
        return text

    def _debounced(self, name: str, fn) -> None:
        """Run `fn` once the user stops typing. Every keystroke in a filter
        box used to rebuild and re-send a whole list."""
        timer = self._filter_timers.pop(name, None)
        if timer is not None:
            timer.cancel()
        timer = threading.Timer(FILTER_DEBOUNCE_S, fn)
        timer.daemon = True
        self._filter_timers[name] = timer
        timer.start()

    @staticmethod
    @staticmethod
    def _capped(items: list, selected_key, key_of) -> list:
        return bp.capped(items, selected_key, key_of)

    def _poll_loop(self):
        """Runs for the lifetime of the session (daemon thread, stopped by
        stop_polling when app.py sees the session close): every TICK_S it
        advances the elapsed counters of whatever the panel is waiting for, every
        CONTACT_LABEL_EVERY_S it refreshes the "last contact" label, and
        every POLL_INTERVAL_S it re-tests the connection against whatever
        URI is currently in the field."""
        elapsed_since_check = POLL_INTERVAL_S  # check right away on first tick
        elapsed_since_label = 0.0
        while not self._poll_stop.wait(TICK_S):
            self._tick_tasks()
            elapsed_since_check += TICK_S
            elapsed_since_label += TICK_S
            if elapsed_since_check >= POLL_INTERVAL_S:
                elapsed_since_check = 0.0
                # Never while the user is waiting on something: a tracking
                # server slow enough to be worth a health check is also slow
                # enough that an extra experiments/search queued behind the
                # user's own request makes their wait longer, for an answer
                # their request is about to give anyway.
                if not self._busy():
                    self._check_connection_periodic()
            if elapsed_since_label >= CONTACT_LABEL_EVERY_S:
                elapsed_since_label = 0.0
                self._refresh_last_contact_label()

    def _check_connection_periodic(self):
        uri = self.uri_field.value.strip()
        if not uri:
            return
        try:
            mlf.test_connection(uri)
            recovered = not self.connected
            self.connected = True
            self.last_contact = time.monotonic()
            self.status_dot.bgcolor = ft.Colors.GREEN
            if recovered:
                # Came back on its own (or this is the first successful
                # check) -- refresh the experiment list so the browse dialog
                # isn't stuck showing an empty/stale result from before.
                try:
                    self.experiments = mlf.list_experiments(uri)
                    self.connection_status.content = status_row(
                        f"Connected to {uri} -- {len(self.experiments)} experiment(s)", "success")
                    self._render_experiments()
                    self._restore_remembered_experiment()
                    self._safe_update(self.experiments_list, self.connection_status)
                except mlf.MlflowError:
                    pass
        except mlf.MlflowError:
            self.connected = False
            self.status_dot.bgcolor = ft.Colors.RED
        self._safe_update(self.status_dot)

    def _refresh_last_contact_label(self):
        elapsed = time.monotonic() - self.last_contact if self.last_contact is not None else None
        text = _format_last_contact(elapsed)
        if text != self.last_contact_text.value:
            self.last_contact_text.value = text
            self._safe_update(self.last_contact_text)

    # ---- connection ----
    @staticmethod
    def _initial_status_text() -> str:
        """Says why the field holds what it holds, when that is not obvious:
        an MLFLOW_TRACKING_URI this panel cannot use would otherwise look
        like the variable was simply ignored."""
        raw = os.environ.get(TRACKING_URI_ENV, "").strip()
        if raw and not _env_tracking_uri():
            return (f"Not connected -- ${TRACKING_URI_ENV} is '{raw}', which is not an "
                    f"http(s) tracking server, so it was not used.")
        return "Not connected"

    def stop_polling(self) -> None:
        self._poll_stop.set()

    def open_panel(self, e=None):
        self._open_panel(e)

    def _open_panel(self, e):
        self._resize_to_page()
        self.ctx.page.open(self.dialog)

    # ---- recent picks ----
    def _render_recents(self):
        """A row of one-click shortcuts back to what was loaded before.
        Clicking one walks the whole experiment -> run -> artifact chain, so
        the common "same run as yesterday" case never touches a list."""
        self.recents_row.controls.clear()
        picks = self.pref.recent
        self.recents_container.visible = bool(picks)
        if not picks:
            return
        self.recents_row.controls.append(ft.Text(
            "Recent:", size=TEXT_SUB, color=ft.Colors.ON_SURFACE_VARIANT))
        for pick in picks:
            self.recents_row.controls.append(ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.HISTORY, size=15, color=ACCENT),
                    ft.Text(pick.label(), size=TEXT_SUB, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                ], spacing=6, tight=True),
                padding=ft.padding.symmetric(horizontal=10, vertical=5),
                border_radius=14,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ink=True,
                tooltip=f"{pick.experiment_name} / {pick.label()} / {pick.artifact_path}",
                on_click=lambda e, pick=pick: self._open_recent(pick),
            ))

    def _open_recent(self, pick):
        if not self.experiments:
            # The chips are read from disk, so they show up before (or
            # without) a connection -- say so rather than blaming the server
            # for a missing experiment we never looked for.
            self.panel_status.content = status_row(
                "Connect to the tracking server first, then this shortcut will work.", "warning")
            self._safe_update(self.panel_status)
            return
        exp = next((x for x in self.experiments if x.experiment_id == pick.experiment_id), None)
        if exp is None:
            self.panel_status.content = status_row(
                f"'{pick.experiment_name}' is not on this server (any more) -- pick an experiment below.",
                "warning")
            self._safe_update(self.panel_status)
            return
        self._pending_run_id = pick.run_id
        self._pending_artifact_path = pick.artifact_path
        self._on_select_experiment(exp, scroll=True)

    def _remember_pick(self, artifact_path: str):
        run, exp = self.selected_run, self.selected_experiment
        if run is None or exp is None:
            return
        prefs.remember_mlflow_pick(self.pref, prefs.MlflowPick(
            experiment_id=exp.experiment_id, experiment_name=exp.name,
            run_id=run.run_id, run_name=run.run_name, artifact_path=artifact_path,
        ))
        prefs.save_mlflow_pref(self.pref)
        self._render_recents()
        self._safe_update(self.recents_container)

    def _remember_pref(self):
        self.pref.tracking_uri = self.uri_field.value.strip()
        if self.selected_experiment is not None:
            self.pref.experiment_id = self.selected_experiment.experiment_id
            self.pref.experiment_name = self.selected_experiment.name
        prefs.save_mlflow_pref(self.pref)

    def _restore_remembered_experiment(self):
        """Re-open what was loaded last session, once, and only if the user
        hasn't already picked something themselves. The most recent pick
        restores the run and the artifact too; a bare experiment id (from a
        pref file written before picks existed) restores just the list."""
        if self._restored or self.selected_experiment is not None:
            return
        self._restored = True
        if self.pref.recent:
            self._open_recent(self.pref.recent[0])
            return
        match = next((x for x in self.experiments if x.experiment_id == self._pending_experiment_id), None)
        self._pending_experiment_id = ""
        if match is not None:
            self._on_select_experiment(match, scroll=True)

    def _on_connect(self, e):
        uri = self.uri_field.value.strip()
        if not uri:
            self.connection_status.content = status_row("Enter a tracking URI", "warning")
            self._safe_update(self.connection_status)
            return

        self.connect_btn.text = "Connecting..."
        self.connect_btn.disabled = True
        self.connection_status.content = status_row(f"Contacting {uri}...", "busy")
        # The status row right below already names the URI; the list says
        # what is holding it up, and the ticker adds how long it has been.
        loading = self._show_loading(self.experiments_list, "Contacting the server")
        self._safe_update(self.connect_btn, self.connection_status, self.experiments_list)
        self._begin_task("Contacting the server", loading)
        threading.Thread(target=self._connect_worker, args=(uri,), daemon=True).start()

    def _connect_worker(self, uri: str):
        try:
            mlf.test_connection(uri)
            experiments = mlf.list_experiments(uri)
            error = None
        except mlf.MlflowError as ex:
            experiments, error = [], str(ex)
        finally:
            self._end_task("Contacting the server")

        if error is None:
            self.connected = True
            self.experiments = experiments
            self.last_contact = time.monotonic()
            self.status_dot.bgcolor = ft.Colors.GREEN
            self.connection_status.content = status_row(f"Connected -- {len(experiments)} experiment(s) found", "success")
            self._remember_pref()
            self._render_recents()
        else:
            self.connected = False
            self.experiments = []
            self.status_dot.bgcolor = ft.Colors.RED
            self.connection_status.content = status_row(
                f"{error} (tried {uri} -- is the tracking server running?)", "error")

        self.selected_experiment = None
        self.selected_run = None
        self.selected_artifact = None
        self.runs = []
        self.artifacts = []
        self._render_experiments()
        self._render_runs()
        self._render_artifacts()
        self._render_details()
        self.load_btn.disabled = True

        self.connect_btn.text = "Connect"
        self.connect_btn.disabled = False
        self._refresh_last_contact_label()
        # Targeted rather than self.dialog: updating the dialog re-serializes
        # every row of every list and the whole details panel, which on a
        # server with a few hundred experiments is most of the panel's lag.
        self._safe_update(
            self.status_dot, self.connect_btn, self.connection_status, self.recents_container,
            self.experiments_list, self.runs_list, self.hidden_runs_text, self.artifacts_list,
            self.details_title, self.details_column, self.load_btn,
        )
        if error is None:
            self._restore_remembered_experiment()

    # ---- experiments ----
    def _on_experiment_filter_change(self, e):
        self._debounced("experiments", self._on_experiment_sort_change)

    def _on_experiment_sort_change(self, e=None):
        self._render_experiments()
        self._safe_update(self.experiments_list)

    def _sorted_experiments(self) -> list:
        mode = self.experiment_sort.value
        if mode == "updated":
            return sorted(self.experiments, key=lambda x: -(x.last_update_time or x.creation_time or 0))
        if mode == "created":
            return sorted(self.experiments, key=lambda x: -(x.creation_time or 0))
        return sorted(self.experiments, key=lambda x: x.name.lower())

    def _render_experiments(self):
        query = (self.experiment_filter.value or "").strip().lower()
        self.experiments_list.controls.clear()
        self._experiment_rows.clear()
        matching = [exp for exp in self._sorted_experiments()
                    if not query or query in exp.name.lower() or query in exp.description.lower()]
        selected_id = self.selected_experiment.experiment_id if self.selected_experiment else None
        for exp in self._capped(matching, selected_id, lambda x: x.experiment_id):
            subtitle = ""
            if self.experiment_sort.value != "name":
                stamp = exp.last_update_time if self.experiment_sort.value == "updated" else exp.creation_time
                subtitle = _format_time(stamp)
            row = self._row(
                exp.name, subtitle or None, exp.experiment_id == selected_id,
                lambda e, exp=exp: self._on_select_experiment(exp),
                icon=ft.Icons.SCIENCE_OUTLINED,
                key=exp.experiment_id,
            )
            self._experiment_rows[exp.experiment_id] = row
            self.experiments_list.controls.append(row.container)
        if not matching:
            self.experiments_list.controls.append(
                self._placeholder("No experiments" if not self.experiments else "No matches"))
        elif len(matching) > len(self._experiment_rows):
            self.experiments_list.controls.append(
                self._overflow_note(len(self._experiment_rows), len(matching)))

    def _on_select_experiment(self, exp: mlf.Experiment, scroll: bool = False):
        previous = self.selected_experiment.experiment_id if self.selected_experiment else None
        self.selected_experiment = exp
        self.selected_run = None
        self.selected_artifact = None
        self.runs = []
        self.artifacts = []
        self.npz_artifacts = {}
        self.load_btn.disabled = True
        if not self._select_in_list(self._experiment_rows, previous, exp.experiment_id):
            self._render_experiments()
            self._safe_update(self.experiments_list)
        if scroll:
            self._scroll_into_view(self.experiments_list, exp.experiment_id)
        loading = self._show_loading(self.runs_list, "Loading runs")
        # The count belongs to the runs that were just thrown away; leaving
        # it up says "(43 hidden)" over a list that is loading.
        self.hidden_runs_text.value = ""
        self._render_artifacts()
        self._render_details()
        self.panel_status.content = None
        self._safe_update(self.runs_list, self.hidden_runs_text, self.artifacts_list,
                          self.details_title, self.details_column, self.load_btn,
                          self.panel_status)
        self._remember_pref()

        uri = self.uri_field.value.strip()
        self._begin_task("Loading runs", loading)
        threading.Thread(target=self._fetch_runs_worker, args=(uri, exp), daemon=True).start()

    def _fetch_runs_worker(self, uri: str, exp: mlf.Experiment):
        try:
            runs, error = mlf.list_runs(uri, exp.experiment_id), None
        except mlf.MlflowError as ex:
            runs, error = [], str(ex)
        finally:
            self._end_task("Loading runs")
        if self.selected_experiment is None or self.selected_experiment.experiment_id != exp.experiment_id:
            return  # the user moved on while this was in flight
        self.runs = runs
        self.panel_status.content = status_row(error, "error") if error else None
        self._render_runs()
        self._render_details()  # the experiment details panel counts the runs
        self._safe_update(self.runs_list, self.hidden_runs_text, self.details_title,
                          self.details_column, self.panel_status)
        self._consume_pending_run()

    def _consume_pending_run(self):
        """Second leg of a "recent" chip: select the remembered run, if it is
        still in the list. If it has gone, the artifact leg is dropped too --
        the user is left on the experiment, which is as far as we can honour."""
        pending, self._pending_run_id = self._pending_run_id, ""
        if not pending:
            return
        run = next((r for r in self.runs if r.run_id == pending), None)
        if run is None:
            self._pending_artifact_path = ""
            return
        if run.status in UNSUCCESSFUL_STATUSES and not self.show_unsuccessful.value:
            # Otherwise the restored run would be selected but invisible.
            self.show_unsuccessful.value = True
            self._render_runs()
        self._on_select_run(run, scroll=True)

    # ---- runs ----
    def _on_run_filter_change(self, e):
        self._debounced("runs", self._on_run_sort_change)

    def _on_run_sort_change(self, e=None):
        self._render_runs()
        self._safe_update(self.runs_list, self.hidden_runs_text)

    def _sorted_runs(self) -> list:
        mode = self.run_sort.value
        if mode == "name":
            return sorted(self.runs, key=lambda r: r.label().lower())
        if mode == "status":
            return sorted(self.runs, key=lambda r: (r.status, -(r.start_time or 0)))
        if mode == "duration":
            return sorted(self.runs, key=lambda r: -(r.duration_s() or 0))
        return sorted(self.runs, key=lambda r: -(r.start_time or 0))

    @staticmethod
    def _run_matches(run: mlf.Run, query: str) -> bool:
        """Substring match over everything that identifies a run, not just
        its name -- 'resnet50' or 'lr=0.001' are how people actually
        remember which run they want."""
        if not query:
            return True
        if "=" in query:
            key, _, value = query.partition("=")
            key, value = key.strip(), value.strip()
            return any(key in k.lower() and value in str(v).lower() for k, v in run.params.items())
        haystack = " ".join([
            run.label(), run.run_id, run.status, run.description,
            " ".join(f"{k}={v}" for k, v in run.params.items()),
        ]).lower()
        return query in haystack

    def _render_runs(self):
        query = (self.run_filter.value or "").strip().lower()
        show_unsuccessful = bool(self.show_unsuccessful.value)
        self.runs_list.controls.clear()

        hidden = 0
        visible = []
        for run in self._sorted_runs():
            if run.status in UNSUCCESSFUL_STATUSES and not show_unsuccessful:
                hidden += 1
                continue
            if not self._run_matches(run, query):
                continue
            visible.append(run)

        self.hidden_runs_text.value = f"({hidden} hidden)" if hidden else ""

        self._run_rows.clear()
        selected_id = self.selected_run.run_id if self.selected_run else None
        group = None
        for run in self._capped(visible, selected_id, lambda r: r.run_id):
            # Date separators only make sense while the list is in date
            # order; any other sort would scatter them through the list.
            if self.run_sort.value == "recent":
                run_group = _run_group(run.start_time)
                if run_group != group:
                    group = run_group
                    self.runs_list.controls.append(self._group_header(run_group))
            subtitle = " · ".join(x for x in (
                run.status, _format_time(run.start_time), _format_duration(run.duration_s()),
            ) if x)
            row = self._row(
                run.label(), subtitle, run.run_id == selected_id,
                lambda e, run=run: self._on_select_run(run),
                icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
                icon_color=STATUS_COLORS.get(run.status, ft.Colors.ON_SURFACE_VARIANT),
                key=run.run_id,
            )
            self._run_rows[run.run_id] = row
            self.runs_list.controls.append(row.container)

        if len(visible) > len(self._run_rows):
            self.runs_list.controls.append(self._overflow_note(len(self._run_rows), len(visible)))

        if not visible:
            if self.selected_experiment is None:
                msg = "Select an experiment"
            elif not self.runs:
                msg = "No runs"
            elif hidden and not query:
                msg = "Every run in this experiment failed or was killed"
            else:
                msg = "No matches"
            self.runs_list.controls.append(self._placeholder(msg))

    def _on_select_run(self, run: mlf.Run, scroll: bool = False):
        previous = self.selected_run.run_id if self.selected_run else None
        self.selected_run = run
        self.selected_artifact = None
        self.artifacts = []
        self.npz_artifacts = {}
        self.load_btn.disabled = True
        if not self._select_in_list(self._run_rows, previous, run.run_id):
            self._render_runs()
            self._safe_update(self.runs_list, self.hidden_runs_text)
        if scroll:
            self._scroll_into_view(self.runs_list, run.run_id)
        self._render_details()
        loading = self._show_loading(self.artifacts_list, "Looking for CSV artifacts")
        self.panel_status.content = None
        self._safe_update(self.details_title, self.details_column, self.artifacts_list,
                          self.load_btn, self.panel_status)

        uri = self.uri_field.value.strip()
        self._begin_task("Looking for CSV artifacts", loading)
        threading.Thread(target=self._fetch_artifacts_worker, args=(uri, run), daemon=True).start()

    def _fetch_artifacts_worker(self, uri: str, run: mlf.Run):
        # One walk of the artifact tree for both: the .npz files are what
        # tells the CSV rows whether embedding/prototype sidecars will come
        # along, which the user wants to know *before* loading, not after.
        try:
            (artifacts, npz), error = mlf.list_csv_and_npz_artifacts(uri, run.run_id), None
        except mlf.MlflowError as ex:
            artifacts, npz, error = [], [], str(ex)
        finally:
            self._end_task("Looking for CSV artifacts")
        if self.selected_run is None or self.selected_run.run_id != run.run_id:
            return  # the user moved on while this was in flight
        self.artifacts = artifacts
        self.npz_artifacts = {a.path: a for a in npz}
        self.panel_status.content = status_row(error, "error") if error else None
        self._render_artifacts()
        self._safe_update(self.artifacts_list, self.panel_status)
        self._consume_pending_artifact()

    def _consume_pending_artifact(self):
        """Third and last leg of a "recent" chip."""
        pending, self._pending_artifact_path = self._pending_artifact_path, ""
        if not pending:
            return
        art = next((a for a in self.artifacts if a.path == pending), None)
        if art is not None:
            self._on_select_artifact(art)

    # ---- artifacts ----
    def _on_artifact_filter_change(self, e):
        self._debounced("artifacts", self._apply_artifact_filter)

    def _apply_artifact_filter(self):
        self._render_artifacts()
        self._safe_update(self.artifacts_list)

    def _sidecars_for(self, csv_path: str) -> list:
        """Which of the by-convention sidecars this CSV actually has in the
        run (same naming rule _auto_load_embedding_sidecars applies after the
        download -- shown here so the answer is visible beforehand)."""
        embed_rel, proto_rel = sidecar_paths_for_csv(csv_path)
        return [name for name, rel in (("embeddings", embed_rel), ("prototypes", proto_rel))
                if rel.as_posix() in self.npz_artifacts]

    def _render_artifacts(self):
        query = (self.artifact_filter.value or "").strip().lower()
        self.artifacts_list.controls.clear()
        self._artifact_rows.clear()
        matching = [art for art in self.artifacts if not query or query in art.path.lower()]
        matched = len(matching)
        selected_path = self.selected_artifact.path if self.selected_artifact else None
        for art in self._capped(matching, selected_path, lambda a: a.path):
            selected = art.path == selected_path
            sidecars = self._sidecars_for(art.path)
            badge = None
            if sidecars:
                badge = ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.LAYERS_OUTLINED, size=13, color=ACCENT),
                        ft.Text(" + ".join(sidecars), size=TEXT_SMALL, color=ACCENT),
                    ], spacing=4, tight=True),
                    padding=ft.padding.symmetric(horizontal=7, vertical=2),
                    border_radius=10, border=ft.border.all(1, ACCENT),
                    tooltip="Loading this CSV will also attach these sidecars, found in the same "
                            "folder of the run's artifacts.",
                )
            row = self._row(
                art.path, _human_size(art.size), selected,
                lambda e, art=art: self._on_select_artifact(art),
                icon=ft.Icons.TABLE_CHART_OUTLINED,
                extra=badge,
                key=art.path,
            )
            self._artifact_rows[art.path] = row
            self.artifacts_list.controls.append(row.container)
        if matched > len(self._artifact_rows):
            self.artifacts_list.controls.append(
                self._overflow_note(len(self._artifact_rows), matched))
        if not matched:
            if self.selected_run is None:
                msg = "Select a run"
            elif not self.artifacts:
                msg = "No CSV artifacts in this run"
            else:
                msg = "No matches"
            self.artifacts_list.controls.append(self._placeholder(msg))

    def _on_select_artifact(self, art: mlf.ArtifactFile):
        previous = self.selected_artifact.path if self.selected_artifact else None
        self.selected_artifact = art
        if not self._select_in_list(self._artifact_rows, previous, art.path):
            self._render_artifacts()
            self._safe_update(self.artifacts_list)
        run = self.selected_run
        stem = Path(art.path).stem
        suggested = f"{run.label()}_{stem}" if run else stem
        self.load_name_field.value = suggested
        self.load_btn.disabled = False
        # The details panel shows the *run*, which has not changed -- so
        # picking a CSV costs two row updates and a text field, not a
        # rebuild of the params and metrics list.
        self._safe_update(self.load_name_field, self.load_btn)

    # ---- run details ----
    def _render_details(self):
        self.details_column.controls.clear()
        run = self.selected_run
        if run is None:
            self._render_experiment_details()
            return

        self.details_title.value = "Run details"

        status_color = STATUS_COLORS.get(run.status, ft.Colors.ON_SURFACE_VARIANT)
        self.details_column.controls.append(ft.Row([
            ft.Text(run.label(), size=TEXT_TITLE, weight=ft.FontWeight.BOLD, expand=True,
                    max_lines=2, overflow=ft.TextOverflow.ELLIPSIS, tooltip=run.label()),
            ft.Container(
                content=ft.Text(run.status, size=TEXT_SMALL, weight=ft.FontWeight.BOLD, color=status_color),
                padding=ft.padding.symmetric(horizontal=10, vertical=4),
                border_radius=12, border=ft.border.all(1, status_color),
            ),
        ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER))

        self.details_column.controls.append(ft.Row([
            ft.Text(run.run_id, size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                    font_family="monospace", expand=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            ft.IconButton(ft.Icons.CONTENT_COPY, icon_size=15, tooltip="Copy run ID",
                          on_click=lambda e, rid=run.run_id: self.ctx.page.set_clipboard(rid)),
        ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER))

        meta = [
            ("Started", _format_time(run.start_time)),
            ("Duration", _format_duration(run.duration_s()) or ("still running" if run.status == "RUNNING" else "")),
            ("User", run.user),
            ("Git commit", run.git_commit[:10] if run.git_commit else ""),
        ]
        meta_rows = [self._kv_row(k, v) for k, v in meta if v]
        if meta_rows:
            self.details_column.controls.append(ft.Column(meta_rows, spacing=6))

        self.details_column.controls.append(ft.Divider(height=1))
        self.details_column.controls.append(
            ft.Text("Description", size=TEXT_HEADING, weight=ft.FontWeight.BOLD))
        if run.description.strip():
            self.details_column.controls.append(ft.Text(
                run.description.strip(), size=TEXT_ROW, selectable=True))
        else:
            self.details_column.controls.append(ft.Text(
                "No description recorded for this run.",
                size=TEXT_SUB, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))

        if run.metrics:
            self.details_column.controls.append(self._kv_section(
                f"Metrics ({len(run.metrics)})",
                [(k, _format_metric(v)) for k, v in sorted(run.metrics.items())],
                expanded=True,
            ))
        if run.params:
            self.details_column.controls.append(self._kv_section(
                f"Parameters ({len(run.params)})",
                sorted(run.params.items()),
                expanded=False,
            ))

    def _render_experiment_details(self):
        """What the panel shows while no run is selected: either the
        experiment's own description, or the prompt to pick one."""
        exp = self.selected_experiment
        if exp is None:
            self.details_title.value = "Run details"
            self.details_column.controls.append(info_banner(
                "Select a run to see its description, parameters and metrics before loading anything.",
                ft.Icons.INFO_OUTLINE,
            ))
            return

        self.details_title.value = "Experiment details"
        self.details_column.controls.append(ft.Text(
            exp.name, size=TEXT_TITLE, weight=ft.FontWeight.BOLD,
            max_lines=2, overflow=ft.TextOverflow.ELLIPSIS, tooltip=exp.name))

        meta = [
            ("Experiment ID", exp.experiment_id),
            ("Created", _format_time(exp.creation_time)),
            ("Last updated", _format_time(exp.last_update_time)),
            ("Runs", f"{len(self.runs)}" if self.runs else ""),
        ]
        meta_rows = [self._kv_row(k, v) for k, v in meta if v]
        if meta_rows:
            self.details_column.controls.append(ft.Column(meta_rows, spacing=6))

        self.details_column.controls.append(ft.Divider(height=1))
        self.details_column.controls.append(
            ft.Text("Description", size=TEXT_HEADING, weight=ft.FontWeight.BOLD))
        if exp.description.strip():
            self.details_column.controls.append(ft.Text(
                exp.description.strip(), size=TEXT_ROW, selectable=True))
        else:
            self.details_column.controls.append(ft.Text(
                "No description recorded for this experiment.",
                size=TEXT_SUB, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))

        self.details_column.controls.append(info_banner(
            "Pick a run on the left to see its parameters and metrics.", ft.Icons.ARROW_BACK))

    def _kv_row(self, key: str, value: str) -> ft.Control:
        return ft.Row([
            ft.Text(key, size=TEXT_SUB, color=ft.Colors.ON_SURFACE_VARIANT,
                    expand=2, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, tooltip=key),
            ft.Text(value, size=TEXT_SUB, weight=ft.FontWeight.W_500, expand=3,
                    max_lines=2, overflow=ft.TextOverflow.ELLIPSIS, tooltip=value,
                    text_align=ft.TextAlign.RIGHT, selectable=True),
        ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.START)

    def _kv_section(self, title: str, items: list, expanded: bool) -> ft.Control:
        """A collapsible key/value block. A collapsed one builds nothing
        until it is opened: a run that logged 400 params paid for 400 rows
        (each two Texts) on every click in the run list, to show a tile that
        starts closed.

        Long sections are cut off at MAX_KV_ROWS with a line saying so --
        this panel is for recognising the right run, not for reading every
        value the run logged."""
        body = ft.Column(spacing=6)
        shown = items[:MAX_KV_ROWS]

        def fill():
            if body.controls:
                return
            body.controls = [self._kv_row(k, str(v)) for k, v in shown]
            if len(items) > len(shown):
                body.controls.append(ft.Text(
                    f"... and {len(items) - len(shown):,} more, not shown here",
                    size=TEXT_SMALL, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))

        def on_change(e):
            if str(e.data).lower() == "true":
                fill()
                self._safe_update(body)

        if expanded:
            fill()
        return ft.ExpansionTile(
            title=ft.Text(title, size=TEXT_HEADING, weight=ft.FontWeight.BOLD),
            initially_expanded=expanded,
            maintain_state=True,
            on_change=on_change,
            tile_padding=ft.padding.symmetric(horizontal=4),
            controls=[ft.Container(
                content=body,
                padding=ft.padding.only(left=4, right=4, bottom=10),
            )],
        )

    # ---- row widgets ----
    def _placeholder(self, message: str) -> ft.Control:
        return ft.Container(
            content=ft.Text(message, size=TEXT_SUB, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
            padding=10,
        )

    def _group_header(self, text: str) -> ft.Control:
        return bp.group_header(text)

    def _row(self, title: str, subtitle, selected: bool, on_click,
             icon=None, icon_color=None, extra=None, key=None) -> _ListRow:
        return bp.row(title, subtitle, selected, on_click, icon, icon_color, extra, key)

    def _select_in_list(self, rows: dict, previous_key, new_key) -> bool:
        return bp.select_in_list(rows, previous_key, new_key)

    def _overflow_note(self, shown: int, total: int) -> ft.Control:
        return bp.overflow_note(shown, total)

    def _on_load_click(self, e):
        if self.selected_run is None or self.selected_artifact is None:
            return
        if self.ctx.state.dataset is None:
            self.panel_status.content = status_row("Load the dataset in the Data tab first.", "warning")
            self.panel_status.update()
            return

        self.load_btn.text = "Loading..."
        self.load_btn.disabled = True
        self.panel_status.content = status_row(
            f"Downloading {Path(self.selected_artifact.path).name}...", "busy")
        self._safe_update(self.load_btn, self.panel_status)
        self._begin_task(f"Loading {Path(self.selected_artifact.path).name}")

        # On a worker like every other server call this dialog makes (see
        # the module docstring): this one downloads the CSV, both .npz
        # sidecars, reads and merges them, and re-renders every view. On the
        # event thread that is the whole app frozen for as long as it takes,
        # with a button that says "Loading..." only because nothing has had
        # a chance to repaint since.
        threading.Thread(
            target=self._load_worker,
            # The sidecar listing too, as of now: the run list stays
            # clickable during the download, and selecting another run
            # replaces self.npz_artifacts with that run's.
            args=(self.uri_field.value.strip(), self.selected_run, self.selected_artifact,
                  self.load_name_field.value.strip(), dict(self.npz_artifacts)),
            daemon=True,
        ).start()

    def _load_worker(self, uri: str, run: mlf.Run, art: mlf.ArtifactFile, typed_name: str,
                     npz_artifacts: dict | None = None):
        try:
            dest = self._download_cached(uri, run.run_id, art.path, art.size,
                                         progress=self._download_progress(Path(art.path).name))
            name = typed_name or Path(art.path).stem
            self._hide_download_bar()
            # Reading is the other half of the wait, and on a large CSV it
            # is the longer one: saying only "Downloading" leaves the user
            # watching a finished download for a minute.
            self.panel_status.content = status_row(f"Reading {Path(art.path).name}...", "busy")
            self._safe_update(self.panel_status)
            success, message = self.ctx.load_experiment_from_file(str(dest), name)
            if success:
                message += " " + self._auto_load_embedding_sidecars(uri, run, art, name, npz_artifacts)
            kind = "success" if success else "error"
            self.panel_status.content = status_row(_short(message), kind)
            if success:
                self._remember_pick(art.path)
            if success and self.ctx.go_to_data_tab:
                self.ctx.go_to_data_tab()
                self.ctx.page.close(self.dialog)
        except mlf.MlflowError as ex:
            self.panel_status.content = status_row(_short(str(ex)), "error")
        except Exception as ex:
            # Anything else (a full cache disk, a permission error on the
            # rename) would otherwise kill this thread silently and leave
            # the button stuck on "Loading..." forever -- on the event
            # thread it at least surfaced in the console.
            traceback.print_exc()
            self.panel_status.content = status_row(_short(f"Could not load: {ex}"), "error")

        self._end_task(f"Loading {Path(art.path).name}")
        self._hide_download_bar()
        self.load_btn.text = "Load into Data tab"
        self.load_btn.disabled = False
        self._safe_update(self.load_btn, self.panel_status)

    def _download_progress(self, name: str):
        """A progress callback for mlflow_client.download_artifact that
        writes the transfer into the footer.

        Throttled in time as well as in bytes (the client already reports
        only every 512KB): on a fast link that is still hundreds of
        callbacks a second, and each one is a repaint pushed to the browser.
        """
        state = {"at": 0.0}

        def report(done: int, total: int | None) -> None:
            now = time.monotonic()
            done_downloading = total is not None and done >= total
            if now - state["at"] < 0.25 and not done_downloading:
                return
            state["at"] = now
            self.download_bar.visible = True
            self.download_bar.value = (done / total) if total else None
            self.panel_status.content = status_row(
                f"Downloading {name} -- {_format_transfer(done, total)}", "busy")
            self._safe_update(self.download_bar, self.panel_status)

        return report

    def _hide_download_bar(self) -> None:
        if self.download_bar.visible:
            self.download_bar.visible = False
            self._safe_update(self.download_bar)

    @staticmethod
    def _download_cached(uri: str, run_id: str, artifact_path: str, expected_size: int | None,
                         progress=None) -> Path:
        return mlflow_cache.download_cached(uri, run_id, artifact_path, expected_size, progress=progress)

    def _auto_load_embedding_sidecars(self, uri: str, run: mlf.Run, csv_art: mlf.ArtifactFile, experiment_name: str,
                                      npz_artifacts: dict | None = None) -> str:
        """By-convention sidecar discovery, same rule as a locally-loaded
        CSV (see UploadView._auto_attach_sidecars): next to the CSV
        artifact just loaded (same directory in the run's artifact tree),
        look for '<base>_embeddings.npz' and '<base>_prototypes.npz' and, if
        the run actually logged them there, download and attach them.
        Silent if neither is present -- most runs won't have them, and
        that's not an error. Loading goes through EmbeddingsView so the same
        cache-invalidation happens as for a locally-loaded sidecar, and the
        Data tab's row fields are updated to match via
        UploadView.set_row_sidecar_fields() -- those fields stay editable
        either way, so this is only ever a starting guess, never a lock-in."""
        exp = self.ctx.state.experiments.get(experiment_name)
        if exp is None:
            return ""
        embed_rel, proto_rel = sidecar_paths_for_csv(csv_art.path)
        # The artifact tree was already walked when this run was selected
        # (see _fetch_artifacts_worker), and that walk is what put the
        # "+ embeddings + prototypes" badge on the CSV row -- so re-walking
        # it here would be a second full traversal of the run's tree, over
        # the network, to learn what the panel already knows.
        by_path = self.npz_artifacts if npz_artifacts is None else npz_artifacts
        notes = []

        embed_art = by_path.get(embed_rel.as_posix())
        if embed_art is not None:
            try:
                dest = self._download_cached(uri, run.run_id, embed_art.path, embed_art.size,
                                             progress=self._download_progress(embed_rel.name))
                # UploadView._auto_attach_sidecars guesses these same cache
                # paths and loads whatever is already on disk, so on a
                # second load of the same run the sidecar is attached
                # before this runs -- re-reading it would mean parsing the
                # largest file in the run twice for nothing.
                if exp.embeddings_path == str(dest):
                    ok, msg = True, f"{len(exp.embeddings_uuids):,} embeddings"
                else:
                    ok, msg = self.ctx.prototype_source.load_embeddings_for(experiment_name, str(dest))
                notes.append(msg if ok else f"{embed_rel.name} found but could not be loaded ({msg})")
                if self.ctx.upload_view:
                    self.ctx.upload_view.set_row_sidecar_fields(
                        experiment_name, embed_path=str(dest), embed_status=msg if ok else f"error: {msg}",
                    )
            except Exception as ex:
                notes.append(f"{embed_rel.name} found but could not be loaded ({ex})")

        proto_art = by_path.get(proto_rel.as_posix())
        if proto_art is not None:
            try:
                dest = self._download_cached(uri, run.run_id, proto_art.path, proto_art.size,
                                             progress=self._download_progress(proto_rel.name))
                if exp.model_prototypes_path == str(dest):
                    ok, msg = True, f"{len(exp.model_prototypes.class_labels):,}-class prototypes"
                else:
                    ok, msg = self.ctx.prototype_source.load_prototypes_for(experiment_name, str(dest))
                notes.append(msg if ok else f"{proto_rel.name} found but could not be loaded ({msg})")
                if self.ctx.upload_view:
                    self.ctx.upload_view.set_row_sidecar_fields(
                        experiment_name, proto_path=str(dest), proto_status=msg if ok else f"error: {msg}",
                    )
            except Exception as ex:
                notes.append(f"{proto_rel.name} found but could not be loaded ({ex})")

        return f"(also attached: {', '.join(notes)} -- see the Data tab)" if notes else ""
