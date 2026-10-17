"""Home: every workspace, grouped, with the MLflow runs and the datasets
they are made from beside them.

The workspaces table is the main thing -- grouped by the user's own groups
(core/board.py, the same board.json `lookin ui` shows as columns), each
group collapsible, filterable, regroupable by dataset or tag. A run (or one
of its CSVs) dragged onto a workspace row adds it there; a workspace dragged
onto a group header moves it. Everything the original Home did per
workspace is here: open (in a tab of its own), download what it needs, set
its dataset, rename, details/export/snapshots, delete, and remove one of its
experiments.

The MLflow side lists the latest runs across every configured server; the
full table, with its filters, columns and selection, is the All runs window
(see mlflow_dialogs.RunsDialog).
"""
from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from lookin.core import board as bd
from lookin.core import library as lib
from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.ui.home_view import csv_key, parse_key, run_key, workspace_color
from lookin.web import html as h
from lookin.web.datasets import catalog
from lookin.web.mlflow_hub import MlflowHub
from lookin.web.views.base import ConfirmDialog, Dialog, PromptDialog, View
from lookin.web.views.files import FileBrowserDialog, menu

GROUP_PAGE = 25
RECENT_RUNS = 14
SIDE_DATASETS = 12
STATUS_DOT = {"FINISHED": "var(--ok)", "RUNNING": "var(--acc)", "SCHEDULED": "var(--ghost)",
              "FAILED": "var(--err)", "KILLED": "var(--warn-dot)"}
# Group-by options: keys compared in code, labels shown (see CLAUDE.md).
GROUP_BY = (("group", "Group"), ("dataset", "Dataset"), ("tag", "Tag"), ("none", "Nothing"))
QUICK = (("pinned", "Pinned"), ("week", "This week"), ("download", "To download"))
DOWNLOAD_STATE = {
    wss.DOWNLOAD_ALL: ("var(--ok)", "On disk"),
    wss.DOWNLOAD_PARTIAL: ("var(--warn-dot)", "To download"),
    wss.DOWNLOAD_NONE: ("var(--ghost)", "Not downloaded"),
    wss.DOWNLOAD_LOCAL: ("var(--ghost)", "Local files"),
}


def _when(stamp: str) -> str:
    if not stamp:
        return ""
    try:
        t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if t.tzinfo is not None:
            t = t.astimezone().replace(tzinfo=None)
    except ValueError:
        return str(stamp)[:16]
    delta = datetime.now() - t
    if delta < timedelta(minutes=1):
        return "just now"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)} min ago"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)} h ago"
    if delta < timedelta(days=2):
        return "yesterday"
    if delta < timedelta(days=7):
        return f"{delta.days} days ago"
    return t.strftime("%b %d, %Y") if t.year != datetime.now().year else t.strftime("%b %d")


def _ms_when(ms) -> str:
    if not ms:
        return ""
    try:
        return _when(datetime.fromtimestamp(ms / 1000).isoformat())
    except (OSError, OverflowError, ValueError):
        return ""


def _snapshot_count(ws) -> int:
    try:
        return sum(1 for e in ws.snapshots_dir().iterdir() if e.suffix == ".json")
    except OSError:
        return 0


def _mlflow_link(uri: str) -> str:
    """A server chip's "Open in MLflow" -- only for an http(s) address (see
    h.safe_href): the address comes from the Home settings file."""
    href = h.safe_href(f"{uri}/")
    if not href:
        return ""
    return (f'<a href="{h.e(href)}" target="_blank" rel="noopener">{h.icon("external", 14)}'
            f'<span>Open in MLflow</span></a>')


class HomeView(View):
    name = "home"
    title = "Home"

    def __init__(self, session):
        super().__init__(session)
        self.hub = MlflowHub.get()
        self.workspaces: list = []
        self.query = ""
        self.ws_filter = ""
        self.quick: set = set()
        self.group_by = "group"
        self.collapsed: set = set()
        self.limits: dict = {}
        self.expanded: set = set()
        self.ds_filter = ""
        self._last_mark = 0.0
        self.hub.subscribe(session.sid, self._hub_changed)
        session.on_close(lambda: self.hub.unsubscribe(session.sid))
        self.reload()

    def _hub_changed(self) -> None:
        if self.s.mode == "home":
            self.s.mark("home")

    def reload(self) -> None:
        self.workspaces = lib.list_workspaces()

    def on_shown(self) -> None:
        self.reload()
        self.hub.activate()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def search_box(self) -> str:
        return h.text_input(self.query, input_=self.i("search"), type_="search",
                            placeholder="Search workspaces, runs, datasets, lk-… ids", width="100%")

    def render(self) -> str:
        if not self.hub.activated:
            self.hub.activate()
        return (f'<div class="home-grid"><main class="home-main card">{self._workspaces_table()}</main>'
                f'<aside class="home-side">{self._runs_panel()}{self._datasets_panel()}</aside></div>')

    # ---- workspaces ----

    def _matches(self, w) -> bool:
        q = (self.query + " " + self.ws_filter).strip().lower()
        hay = " ".join([w.name, w.id, w.description, " ".join(w.tags), w.mount.dataset_path,
                        " ".join(e.name for e in w.mount.experiments)]).lower()
        if q and not all(word in hay for word in q.split()):
            return False
        if "pinned" in self.quick and not lib.is_favorite(w.id):
            return False
        if "week" in self.quick:
            stamp = lib.last_opened(w.id) or w.updated_at
            try:
                t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).replace(tzinfo=None)
                if datetime.now() - t > timedelta(days=7):
                    return False
            except ValueError:
                return False
        if "download" in self.quick and wss.download_state(w) not in (wss.DOWNLOAD_PARTIAL, wss.DOWNLOAD_NONE):
            return False
        return True

    def _groups(self) -> list:
        """[(group id, name, color, [workspaces], total, is_board_column)]"""
        if self.group_by == "group":
            board = bd.load()
            out = []
            for index, (column, items) in enumerate(bd.layout(board, self.workspaces)):
                out.append((column["id"], column["name"], workspace_color(column["id"]),
                            [w for w in items if self._matches(w)], len(items), True))
            return out
        buckets: dict = {}
        for w in self.workspaces:
            if not self._matches(w):
                continue
            if self.group_by == "dataset":
                keys = [Path(w.mount.dataset_path).name if w.mount.dataset_path else "(no dataset)"]
            elif self.group_by == "tag":
                keys = list(w.tags) or ["(no tag)"]
            else:
                keys = ["All workspaces"]
            for k in keys:
                buckets.setdefault(k, []).append(w)
        return [(f"{self.group_by}:{k}", k, workspace_color(k), items, len(items), False)
                for k, items in sorted(buckets.items(), key=lambda kv: kv[0].lower())]

    def _workspaces_table(self) -> str:
        total = len(self.workspaces)
        groups = self._groups()
        chips = "".join(h.chip(label, act_attr=self.a("quick", key=key), selected=key in self.quick)
                        for key, label in QUICK)
        toolbar = (f'<div class="toolbar"><h1>Workspaces</h1>'
                   f'<span class="muted small">{total:,} · {len([g for g in groups if g[5]]) or len(groups)} groups</span>'
                   f'{h.text_input(self.ws_filter, input_=self.i("ws_filter"), type_="search", placeholder="Filter…", width="180px")}'
                   f'{chips}<span class="sp"></span>'
                   f'{h.select(GROUP_BY, self.group_by, change=self.c("group_by"), label="Group by")}'
                   f'{h.button("Collapse all", act_attr=self.a("collapse_all"), small=True)}'
                   f'{h.button("Expand all", act_attr=self.a("expand_all"), small=True)}'
                   f'{h.button("New group", act_attr=self.a("new_group"), small=True, icon_name="plus") if self.group_by == "group" else ""}'
                   f'</div>')
        head = ('<thead><tr><th style="width:22px"></th><th>Name</th><th>Dataset</th><th>Experiments</th>'
                '<th class="num">Snap.</th><th>Files</th><th style="text-align:right">Opened</th><th style="width:30px"></th></tr></thead>')
        bodies = []
        for gid, name, color, items, gtotal, is_column in groups:
            bodies.append(self._group_tbody(gid, name, color, items, gtotal, is_column))
        if not self.workspaces:
            bodies.append('<tbody><tr><td colspan="8">' + h.empty(
                "No workspace yet. Create one with New workspace, then drag runs onto it.") + '</td></tr></tbody>')
        note = ('<div class="lst-foot">Drag a run from the right onto a workspace row to add it; drag a workspace '
                '(by its handle) onto a group header to move it.</div>')
        return (f'{toolbar}<div class="tbl-wrap"><table class="tbl sticky">{head}{"".join(bodies)}</table></div>'
                f'{note}{self._id_hint()}')

    def _id_hint(self) -> str:
        q = self.query.strip()
        if not (q.startswith("lk-") or q.startswith("lkw-") or q.startswith("lkd-")):
            return ""
        if q.startswith("lk-"):
            found = lib.find_snapshot(q)
            if found:
                ws, snap = found
                return (f'<div class="lst-foot">{h.icon("bookmark", 13)} Snapshot {h.mono(snap.id)} of {h.e(ws.label())} — '
                        f'<a href="/?snapshot={h.e(quote(snap.id, safe=""))}" target="_blank" rel="noopener">open it in a new tab</a></div>')
            return f'<div class="lst-foot">No snapshot {h.mono(q)} in the library.</div>'
        ws = next((w for w in self.workspaces if w.id == q), None)
        if ws:
            return (f'<div class="lst-foot">Workspace {h.e(ws.label())} — '
                    f'<a href="/?workspace={h.e(quote(ws.id, safe=""))}" target="_blank" rel="noopener">open it in a new tab</a></div>')
        return ""

    def _group_tbody(self, gid, name, color, items, gtotal, is_column) -> str:
        collapsed = gid in self.collapsed
        summary = ""
        if collapsed:
            recent = pending = 0
            for w in items:
                stamp = lib.last_opened(w.id) or w.updated_at
                try:
                    t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).replace(tzinfo=None)
                    if datetime.now() - t <= timedelta(days=7):
                        recent += 1
                except ValueError:
                    pass
                if wss.download_state(w) in (wss.DOWNLOAD_PARTIAL, wss.DOWNLOAD_NONE):
                    pending += 1
            bits = []
            if recent:
                bits.append(f"{recent} opened this week")
            if pending:
                bits.append(f"{pending} to download")
            summary = " · ".join(bits)
        drop = (f' data-drop="home:move_ws" data-drop-kinds="ws"{h.data_args(group=gid)}'
                if is_column else "")
        gmenu = ""
        if is_column:
            gmenu = menu([("Rename…", self.a("rename_group", group=gid), "edit", False),
                          ("Remove this group…", self.a("remove_group", group=gid), "trash", True)],
                         title="Group actions")
        count_text = f"{len(items)}" if len(items) == gtotal else f"{len(items)} of {gtotal}"
        header = (f'<tr class="grp"><td colspan="8"><div class="row-nw">'
                  f'<button type="button" class="grp-btn"{self.a("toggle_group", group=gid)} aria-expanded="{"false" if collapsed else "true"}">'
                  f'<span class="tri">{"▶" if collapsed else "▼"}</span>{h.dot(color, size=8)}'
                  f'<b>{h.e(name)}</b><span class="mono small muted">{count_text}</span>'
                  f'<span class="small muted">{h.e(summary)}</span></button>'
                  f'{h.button("+ workspace", act_attr=self.a("new_workspace", group=gid if is_column else ""), kind="link", small=True)}'
                  f'{gmenu}</div></td></tr>')
        rows = []
        if not collapsed:
            limit = self.limits.get(gid, GROUP_PAGE)
            for w in items[:limit]:
                rows.append(self._ws_row(w))
            if len(items) > limit:
                rows.append(f'<tr><td colspan="8"><div class="row-nw small muted" style="padding:2px 0 2px 30px">'
                            f'{limit:,} of {len(items):,} shown {h.button(f"Show {min(GROUP_PAGE, len(items) - limit):,} more", act_attr=self.a("more", group=gid), kind="link", small=True)}</div></td></tr>')
            if not items:
                note = ("Drop workspaces here." if gtotal == 0 else "Nothing here matches the filter.")
                rows.append(f'<tr><td colspan="8"><div class="small muted" style="padding:4px 0 4px 30px;font-style:italic">{note}</div></td></tr>')
        return f'<tbody data-key="g-{h.e(gid)}"{drop}>{header}{"".join(rows)}</tbody>'

    def _ws_row(self, w) -> str:
        expanded = w.id in self.expanded
        current = self.s.current_workspace
        progress = self.hub.downloading.get(w.id)
        summary = wss.download_summary(w)
        color, label = DOWNLOAD_STATE.get(summary.state, ("var(--ghost)", ""))
        if progress is not None:
            files = f'<span class="row-nw small" style="color:var(--acc-ink)"><span class="spinner"></span>{h.e(progress)}</span>'
        else:
            dl = ""
            if summary.state not in (wss.DOWNLOAD_ALL, wss.DOWNLOAD_LOCAL):
                size = summary.missing_bytes
                what = f"{summary.missing} file(s)" + (f", {h.human_size(size)}" if size else "")
                dl = h.button(icon_name="download", act_attr=self.a("download", id=w.id), small=True, kind="ghost",
                              title=f"Download what this workspace needs ({what})")
            files = f'<span class="row-nw small" style="color:var(--muted)">{h.dot(color, size=6)}{h.e(label)}{dl}</span>'
        dataset = Path(w.mount.dataset_path).name if w.mount.dataset_path else "no dataset yet"
        exps = ", ".join(e.name for e in w.mount.experiments) or "—"
        opened = _when(lib.last_opened(w.id) or w.updated_at)
        missing = wss.missing_files(w)
        if missing:
            name_link = (f'<button type="button" class="ws-name btn-link" style="border:0;background:none;padding:0;font:inherit;font-weight:500;cursor:pointer;color:var(--ink)"'
                         f'{self.a("open_ws", id=w.id)} title="{h.e(w.description or w.label())}">{h.e(w.label())}</button>')
        else:
            name_link = (f'<a class="ws-name" href="/?workspace={h.e(quote(w.id, safe=""))}" target="_blank" rel="noopener" '
                         f'title="{h.e(w.description or "Open in a new tab")}">{h.e(w.label())}</a>')
        is_open = current is not None and current.id == w.id
        star = h.icon("star", 12, "c-warn") if lib.is_favorite(w.id) else ""
        open_badge = '<span class="badge" style="background:var(--acc);color:#fff">open</span>' if is_open else ""
        ws_menu = menu([
            ("Open in a new tab", self.a("open_ws", id=w.id), "external", False),
            ("Dataset…", self.a("set_dataset", id=w.id), "table", False),
            ("Rename…", self.a("rename", id=w.id), "edit", False),
            ("Unfavourite" if lib.is_favorite(w.id) else "Favourite", self.a("fav", id=w.id), "star", False),
            ("Details, export, snapshots…", self.a("details", id=w.id), "save", False),
            "hr",
            ("Delete…", self.a("delete", id=w.id), "trash", True),
        ], title="Workspace actions")
        tr = (f'<tr data-drop="home:drop_on_ws" data-drop-kinds="run csv"{h.data_args(id=w.id)}'
              f'{" class=sel" if expanded else ""}>'
              f'<td><span class="row-nw" style="gap:2px"><span class="handle" draggable="true" data-drag="{h.e(w.id)}" data-drag-kind="ws" '
              f'data-drag-label="{h.e(w.label())}" title="Drag to move this workspace to another group">{h.icon("drag", 13)}</span>'
              f'<button type="button" class="chev"{self.a("toggle_ws", id=w.id)} style="font-size:9px" aria-label="Show details" '
              f'aria-expanded="{"true" if expanded else "false"}">{"▼" if expanded else "▶"}</button></span></td>'
              f'<td class="nowrap"><span class="row-nw">{star}{name_link}{open_badge}</span></td>'
              f'<td class="mono small muted nowrap" title="{h.e(w.mount.dataset_path)}">{h.e(dataset)}</td>'
              f'<td class="mono small" style="max-width:280px"><span class="ell" style="display:block" title="{h.e(exps)}">{h.e(exps)}</span></td>'
              f'<td class="num small">{_snapshot_count(w)}</td>'
              f'<td class="nowrap">{files}</td>'
              f'<td class="small muted nowrap" style="text-align:right">{h.e(opened)}</td>'
              f'<td>{ws_menu}</td></tr>')
        detail = ""
        if expanded:
            detail = f'<tr class="ws-detail"><td colspan="8">{self._ws_detail(w)}</td></tr>'
        return tr + detail

    def _ws_detail(self, w) -> str:
        rows = []
        if not w.mount.experiments:
            rows.append('<div class="small muted" style="font-style:italic">Empty -- drop runs or CSVs here.</div>')
        for e in w.mount.experiments:
            if e.from_mlflow():
                downloaded = wss.is_downloaded(e)
                server = self.hub.servers.get(wss.normalize_uri(e.mlflow_uri))
                where = f"{e.mlflow_run_name or e.mlflow_run_id[:8]} / {e.artifact_path}"
                if server is None:
                    where += f"  ({e.mlflow_uri})"
                state = h.dot("var(--ok)" if downloaded else "var(--ghost)", "Downloaded" if downloaded else "Not downloaded yet", 7)
                link = h.link_button("", mlf.run_ui_url(e.mlflow_uri, e.mlflow_experiment_id, e.mlflow_run_id, e.artifact_path),
                                     icon_name="external", title="Open this run in MLflow", small=True)
                lead = h.icon("cloud", 13)
            else:
                where = e.path
                present = Path(e.path).is_file()
                state = h.dot("var(--ok)" if present else "var(--err)", "On disk" if present else "Missing from disk", 7)
                link = ""
                lead = h.icon("file", 13)
            rows.append(f'<div class="row-nw" style="min-height:24px">{lead}<span class="mono small">{h.e(e.name)}</span>'
                        f'<span class="small muted ell grow" title="{h.e(where)}">{h.e(where)}</span>{state}{link}'
                        f'{h.button(icon_name="x", act_attr=self.a("remove_exp", id=w.id, name=e.name), small=True, kind="ghost", title="Remove from this workspace (the run and its files are not touched)")}</div>')
        snaps = lib.list_snapshots(w)
        snap_html = ""
        if snaps:
            items = "".join(
                f'<div class="row-nw" style="min-height:22px">{h.icon("bookmark", 13)}'
                f'<span class="small ell grow" title="{h.e(s.id)}">{h.e(s.label())}</span>'
                f'<a class="btn btn-sm btn-ghost btn-icon" href="/?snapshot={h.e(quote(s.id, safe=""))}" target="_blank" rel="noopener" title="Open this snapshot in a new tab">{h.icon("external", 13)}</a></div>'
                for s in snaps[:6])
            more = (f'<div class="small muted" style="font-style:italic">... and {len(snaps) - 6} more '
                    f'{h.button("(Manage)", act_attr=self.a("details", id=w.id), kind="link", small=True)}</div>') if len(snaps) > 6 else ""
            snap_html = f'<div style="flex:1 1 280px;min-width:240px"><div class="subhead">Snapshots</div>{items}{more}</div>'
        return (f'<div class="ws-detail-inner"><div style="flex:2 1 420px;min-width:340px">'
                f'<div class="subhead">Experiments <span class="muted">{h.e(w.id)} · {h.e(w.mount.dataset_path or "no dataset yet")}</span></div>'
                f'{"".join(rows)}</div>{snap_html}</div>')

    # ---- runs ----

    def _runs_panel(self) -> str:
        hub = self.hub
        chips = []
        for server in hub.servers.values():
            color = {"ok": "var(--ok)", "down": "var(--err)", "loading": "var(--warn-dot)"}.get(server.status, "var(--ghost)")
            tip = server.uri + (f"\n{server.error}" if server.error else "")
            chips.append(
                f'<span class="menu-wrap"><button type="button" class="chip{" off" if server.status == "down" else ""}" data-menu="1" title="{h.e(tip)}">'
                f'{h.dot(color, size=6)}{h.e(server.name)}</button><div class="menu">'
                f'<button type="button"{self.a("refresh_server", uri=server.uri)}>{h.icon("refresh", 14)}<span>Refresh</span></button>'
                f'{_mlflow_link(server.uri)}'
                f'<button type="button" class="danger"{self.a("remove_server", uri=server.uri)}>{h.icon("unlink", 14)}<span>Remove this server</span></button>'
                f'</div></span>')
        chips.append(h.button("+ server", act_attr=self.a("add_server"), small=True, cls="chip"))
        banners = []
        if not hub.servers:
            banners.append(h.banner("No MLflow server yet. Add one to list its runs here -- workspaces work without one.", "info"))
        if hub.config.error:
            banners.append(h.banner(hub.config.error, "warn"))
        for server in hub.servers.values():
            if server.status == "down":
                when = f" -- showing the listing from {server.listed_at.replace('T', ' ')}" if server.offline else ""
                banners.append(h.banner(f"{server.name} is unreachable{when}. Retrying every minute.", "warn", icon_name="cloud-off"))
            elif server.error:
                banners.append(h.banner(server.error, "warn"))
        runs = []
        total = 0
        q = self.query.strip().lower()
        for server in list(hub.servers.values()):
            names = {e.experiment_id: e.name for e in server.experiments}
            for run in hub.visible_runs(server, query=q):
                total += 1
                runs.append((run.start_time or 0, server, run, names.get(run.experiment_id, run.experiment_id)))
        runs.sort(key=lambda x: -x[0])
        by_run, _ = hub.membership(self.workspaces)
        rows = []
        for _, server, run, exp_name in runs[:RECENT_RUNS]:
            key = run_key(server.uri, run.run_id)
            in_ws = by_run.get((server.uri, run.run_id), [])
            badge = f'<span class="small muted nowrap">in {len(in_ws)} ws</span>' if in_ws else ""
            rows.append(
                f'<div class="lst-row" draggable="true" data-drag="{h.e(key)}" data-drag-kind="run" data-drag-label="{h.e(run.label())}"'
                f'{" style=opacity:.55" if server.offline else ""} title="{h.e(run.label())} · {h.e(run.status)} · drag onto a workspace">'
                f'{h.dot(STATUS_DOT.get(run.status, "var(--ghost)"), run.status, 6)}'
                f'<span class="mono ell" style="font-size:12px">{h.e(run.label())}</span>'
                f'<span class="small muted nowrap">{h.e(exp_name)}</span><span class="sp"></span>{badge}'
                f'<span class="small muted nowrap" style="width:68px;text-align:right">{h.e(_ms_when(run.start_time))}</span></div>')
        if not rows:
            loading = any(s.status == "loading" for s in hub.servers.values())
            rows.append(h.empty("Loading runs..." if loading else ("No run matches." if any(s.runs for s in hub.servers.values())
                                                                  else "No runs to show. Pick experiments in All runs.")))
        head = (f'<div class="toolbar"><h2>MLflow runs</h2><span class="muted small">latest across servers</span>'
                f'<span class="sp"></span>{h.button(f"All runs ({total:,})", act_attr=self.a("all_runs"), small=True, cls="btn-acc")}</div>')
        server_row = f'<div class="row" style="padding:6px 10px;border-bottom:1px solid var(--line-3)">{"".join(chips)}</div>'
        ban = f'<div style="padding:6px 10px">{"".join(banners)}</div>' if banners else ""
        return (f'<section class="card">{head}{server_row}{ban}<div>{"".join(rows)}</div>'
                f'<div class="lst-foot">Drag a run onto a workspace row to add it.</div></section>')

    # ---- datasets ----

    def _datasets_panel(self) -> str:
        entries = catalog(self.workspaces, self.hub)
        q = (self.ds_filter + " " + self.query).strip().lower()
        if q:
            entries = [d for d in entries if all(w in (d.path + " " + d.name).lower() for w in q.split())]
        rows = []
        for d in entries[:SIDE_DATASETS]:
            facts = d.facts()
            rows.append(
                f'<div class="lst-row" title="{h.e(d.path)}">'
                f'<span class="mono ell grow" style="font-size:12px;{"color:var(--ghost)" if not d.here else ""}">{h.e(d.name)}</span>'
                f'<span class="mono small muted" style="width:62px;text-align:right">{h.e(f"{facts.rows:,}" if facts and facts.rows is not None else (h.human_size(d.size) if d.here else "—"))}</span>'
                f'<span class="mono small muted" style="width:36px;text-align:right">{len(d.used_by)}</span>'
                f'<span class="small muted nowrap" style="width:72px;text-align:right">{h.e(_when(datetime.fromtimestamp(d.mtime).isoformat()) if d.mtime else "not here")}</span></div>')
        if not rows:
            rows.append(h.empty("No dataset known yet. Watch a folder, or set a workspace's dataset."))
        head = (f'<div class="toolbar"><h2>Datasets</h2>'
                f'{h.text_input(self.ds_filter, input_=self.i("ds_filter"), type_="search", placeholder="Filter…", cls="grow")}'
                f'{h.button(f"All datasets ({len(entries):,})", act_attr=self.a("all_datasets"), small=True, cls="btn-acc")}</div>'
                f'<div class="colhead"><span class="grow">Name</span><span style="width:62px;text-align:right">Rows</span>'
                f'<span style="width:36px;text-align:right">Used</span><span style="width:72px;text-align:right">Modified</span></div>')
        return (f'<section class="card">{head}<div>{"".join(rows)}</div>'
                f'<div class="lst-foot">Greyed: only named by a run\'s parameters, not on this machine yet.</div></section>')

    # ------------------------------------------------------------------
    # filters and layout
    # ------------------------------------------------------------------

    def on_search(self, value: str = "", **_):
        self.query = value or ""
        self.s.mark("home")

    def on_ws_filter(self, value: str = "", **_):
        self.ws_filter = value or ""
        self.refresh()

    def on_ds_filter(self, value: str = "", **_):
        self.ds_filter = value or ""
        self.refresh()

    def on_quick(self, key: str = "", **_):
        self.quick ^= {key}
        self.refresh()

    def on_group_by(self, value: str = "group", **_):
        self.group_by = value if value in dict(GROUP_BY) else "group"
        self.collapsed = set()
        self.refresh()

    def on_toggle_group(self, group: str = "", **_):
        self.collapsed ^= {group}
        self.refresh()

    def on_collapse_all(self, **_):
        self.collapsed = {g[0] for g in self._groups()}
        self.refresh()

    def on_expand_all(self, **_):
        self.collapsed = set()
        self.refresh()

    def on_more(self, group: str = "", **_):
        self.limits[group] = self.limits.get(group, GROUP_PAGE) + GROUP_PAGE
        self.refresh()

    def on_toggle_ws(self, id: str = "", **_):  # noqa: A002
        self.expanded ^= {id}
        self.refresh()

    # ------------------------------------------------------------------
    # groups
    # ------------------------------------------------------------------

    def on_move_ws(self, group: str = "", key: str = "", kind: str = "", **_):
        if kind == "ws" and key:
            bd.move(key, group, known_ids=[w.id for w in self.workspaces])
        self.refresh()

    def on_new_group(self, **_):
        def add(values):
            bd.add_column(values.get("name", "") or "")
            self.refresh()
        self.s.open_dialog(PromptDialog("Add a group", [("name", "Group name", bd.next_default_name(bd.load()))],
                                        add, submit_label="Add"))

    def on_rename_group(self, group: str = "", **_):
        column = next((c for c in bd.load()["columns"] if c["id"] == group), None)
        if column is None:
            return

        def save(values):
            bd.rename_column(group, values.get("name", "") or "")
            self.refresh()
        self.s.open_dialog(PromptDialog("Rename group", [("name", "Group name", column["name"])], save))

    def on_remove_group(self, group: str = "", **_):
        board = bd.load()
        column = next((c for c in board["columns"] if c["id"] == group), None)
        if column is None:
            return
        target = bd.neighbour(board, group)
        if target is None:
            self.s.toast("The last group can't be removed -- workspaces need somewhere to be.", "warn")
            return
        count = next((len(items) for c, items in bd.layout(board, self.workspaces) if c["id"] == group), 0)
        body = (f"Its {count} workspace(s) move to \"{target['name']}\". No workspace is deleted." if count else "It is empty.")

        def yes():
            bd.remove_column(group, workspaces=self.workspaces)
            self.s.toast(f"Removed \"{column['name']}\"." + (f" {count} workspace(s) moved to \"{target['name']}\"." if count else ""), "neutral")
            self.refresh()
        self.s.open_dialog(ConfirmDialog(f"Remove the group \"{column['name']}\"?", body, yes, yes_label="Remove"))

    # ------------------------------------------------------------------
    # workspaces
    # ------------------------------------------------------------------

    def _ws(self, wid):
        return next((w for w in self.workspaces if w.id == wid), None)

    def on_fav(self, id: str = "", **_):  # noqa: A002
        lib.set_favorite(id, not lib.is_favorite(id))
        self.refresh()

    def on_details(self, id: str = "", **_):  # noqa: A002
        from lookin.web.views.workspaces import WorkspacesDialog
        self.s.open_dialog(WorkspacesDialog(self.s, select=self._ws(id)))

    def on_new_workspace(self, group: str = "", keys=None, **_):
        self.s.open_dialog(NewWorkspaceDialog(self, group=group, keys=list(keys or [])))

    def on_set_dataset(self, id: str = "", **_):  # noqa: A002
        ws = self._ws(id)
        if ws is not None:
            self.s.open_dialog(SetDatasetDialog(self, ws))

    def on_rename(self, id: str = "", **_):  # noqa: A002
        ws = self._ws(id)
        if ws is None:
            return

        def save(values):
            new_name = (values.get("name") or "").strip()
            if not new_name:
                return None

            def change(fresh):
                fresh.name = new_name
                return True
            fresh = lib.update_workspace(ws.path, change)
            if fresh is not None:
                self.after_workspace_change(fresh)
            return None
        self.s.open_dialog(PromptDialog("Rename workspace", [("name", "Name", ws.name)], save))

    def on_delete(self, id: str = "", **_):  # noqa: A002
        ws = self._ws(id)
        if ws is None:
            return
        count = len(lib.list_snapshots(ws))

        def yes():
            if lib.delete_workspace(ws):
                self.reload()
                self.s.toast(f"Deleted {ws.label()}.", "neutral")
            else:
                self.s.toast("Could not delete the workspace.", "error")
            self.refresh()
        self.s.open_dialog(ConfirmDialog(
            f"Delete {ws.label()}" + (f" and its {count} snapshot(s)?" if count else "?"),
            "The workspace directory is removed. The runs on MLflow, the dataset and the downloaded files are not touched.",
            yes))

    def on_remove_exp(self, id: str = "", name: str = "", **_):  # noqa: A002
        ws = self._ws(id)
        if ws is None:
            return
        fresh = lib.update_workspace(ws.path, lambda w: wss.remove_experiment(w, name))
        if fresh is not None:
            self.after_workspace_change(fresh)
            self.s.toast(f"Removed {name} from {ws.label()}.", "neutral")

    def after_workspace_change(self, ws) -> None:
        self.expanded.add(ws.id)
        current = self.s.current_workspace
        if current is not None and current.id == ws.id:
            self.s.current_workspace = ws
            self.s.mark("chrome")
        self.reload()
        self.s.mark("home", "dialogs")

    # ---- opening and downloading ----

    def on_open_ws(self, id: str = "", snapshot: str = "", **_):  # noqa: A002
        ws = self._ws(id)
        if ws is None:
            return
        snap = lib.load_snapshot(ws, snapshot) if snapshot else None
        missing = wss.missing_files(ws)
        link = f"/?snapshot={quote(snap.id, safe='')}" if snap is not None else f"/?workspace={quote(ws.id, safe='')}"
        if not missing:
            self.s.cmd(open=link)
            return
        self.s.open_dialog(MissingFilesDialog(self, ws, link, missing))

    def on_download(self, id: str = "", **_):  # noqa: A002
        ws = self._ws(id)
        if ws is not None:
            self.download(ws)

    def download(self, ws, then=None) -> None:
        hub = self.hub
        if ws.id in hub.downloading:
            return
        hub.downloading[ws.id] = "Starting the download..."
        hub.changed()
        self.s.busy.begin(f"Downloading {ws.label()}")
        last = {"at": 0.0}
        session = self.s

        def progress(index, count, name, done, total):
            now = time.monotonic()
            if now - last["at"] < 0.3:
                return
            last["at"] = now
            amount = f"{h.human_size(done)} of {h.human_size(total)}" if total else h.human_size(done)
            hub.downloading[ws.id] = f"Downloading {index}/{count}: {name} -- {amount}"
            hub.changed()

        def work():
            try:
                fetched, errors = wss.download(ws, progress)
                lib.update_workspace(ws.path, wss.point_at_cache)
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                fetched, errors = 0, [str(ex)]
            finally:
                hub.downloading.pop(ws.id, None)
                session.busy.end()
            self.reload()
            hub.changed()
            if errors:
                session.toast(f"{ws.label()}: downloaded {fetched} file(s), {len(errors)} failed -- "
                              + "; ".join(errors)[:400], "error")
            else:
                session.toast(f"{ws.label()}: downloaded {fetched} file(s).", "ok")
                if then is not None:
                    then()

        threading.Thread(target=work, daemon=True).start()

    # ---- adding runs ----

    def on_drop_on_ws(self, id: str = "", kind: str = "", key: str = "", **_):  # noqa: A002
        ws = self._ws(id)
        if ws is None or kind not in ("run", "csv") or not key:
            return
        keys = [k for k in key.split("\n") if k]
        self.add_keys(ws, keys)

    def add_keys(self, ws, keys: list, on_done=None) -> None:
        """Add runs and CSVs to `ws`. A run's CSVs have to be known first, so
        any run whose artifacts are not listed yet is listed -- then this
        comes back round (see HomeView.add_keys in the original UI)."""
        hub = self.hub
        pending = []
        for key in keys:
            kind, uri, run_id, _ = parse_key(key)
            server = hub.servers.get(uri)
            if server is None:
                continue
            entry = server.artifacts.get(run_id)
            if entry is None or entry.get("loading"):
                pending.append((server, run_id))
        if pending:
            self.s.toast(f"Listing the artifacts of {len(pending)} run(s)...", "neutral")
            remaining = {"n": len(pending)}

            def one_done():
                remaining["n"] -= 1
                if remaining["n"] == 0:
                    with self.s.lock:
                        self.add_keys(ws, keys, on_done)
                    self.s.push({"dirty": ["home", "dialogs"]})

            for server, run_id in pending:
                server.artifacts.pop(run_id, None)
                hub.fetch_artifacts(server, run_id, then=one_done)
            return
        direct, ambiguous, problems = [], [], []
        for key in keys:
            kind, uri, run_id, path = parse_key(key)
            server = hub.servers.get(uri)
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
            self.s.open_dialog(PickCsvsDialog(self, ws, direct, ambiguous, problems, on_done))
        else:
            self.commit_add(ws, direct, problems, on_done)

    def commit_add(self, ws, candidates: list, problems: list, on_done=None) -> None:
        outcome = {}

        def change(fresh):
            outcome["result"] = wss.add_candidates(fresh, candidates, self.hub.config.dataset_param)
            return bool(outcome["result"].added)

        fresh = lib.update_workspace(ws.path, change)
        if fresh is None:
            self.s.toast(f"Could not write {ws.label()} -- was it deleted?", "error")
            self.reload()
            self.refresh()
            return
        result = outcome["result"]
        result.skipped.extend((p, "") for p in problems)
        self.after_workspace_change(fresh)
        kind = "ok" if result.added and not result.skipped else ("warn" if result.added else "error")
        self.s.toast(f"{fresh.label()}: {result.message().replace(' ()', '')}", kind)
        if on_done is not None:
            on_done()

    # ---- servers and windows ----

    def on_all_runs(self, **_):
        from lookin.web.views.mlflow_dialogs import RunsDialog
        self.s.open_dialog(RunsDialog(self.s))

    def on_all_datasets(self, **_):
        from lookin.web.views.mlflow_dialogs import DatasetsDialog
        self.s.open_dialog(DatasetsDialog(self.s))

    def on_refresh_server(self, uri: str = "", **_):
        self.hub.refresh(uri)

    def on_remove_server(self, uri: str = "", **_):
        self.hub.config.remove_server(uri)
        self.hub.sync_servers()
        self.hub.changed()

    def on_add_server(self, **_):
        from lookin.web.views.mlflow_dialogs import AddServerDialog
        self.s.open_dialog(AddServerDialog(self.hub))


# ---- dialogs -----------------------------------------------------------------

class NewWorkspaceDialog(Dialog):
    width = 580
    title = "New workspace"

    def __init__(self, home: HomeView, group: str = "", keys=None):
        super().__init__()
        self.home = home
        self.group = group
        self.keys = list(keys or [])
        self.name = ""
        self.dataset = ""
        self.add_keys = bool(self.keys)
        # The group the workspace goes in: the one whose + was clicked, else
        # the first. Keyed by the board column's id, never by its name --
        # two groups may share a name.
        columns = bd.load()["columns"]
        ids = [c["id"] for c in columns]
        self.group = group if group in ids else (ids[0] if ids else "")

    def body(self) -> str:
        param = self.home.hub.config.dataset_param
        add = (h.checkbox(f"Add the {len(self.keys)} selected item(s)", self.add_keys, change=self.c("add_keys"))
               if self.keys else "")
        columns = bd.load()["columns"]
        group = (h.select([(c["id"], c["name"]) for c in columns], self.group, change=self.c("group"), label="Group")
                 if len(columns) > 1 else "")
        return (f'<div class="form-col">{h.text_input(self.name, change=self.c("name"), enter=h.on_enter(self.target, "create_enter"), label="Name", autofocus=True)}'
                f'{group}'
                f'<label class="fld"><span class="fld-l">Dataset CSV (optional)</span><div class="row-nw">'
                f'{h.text_input(self.dataset, change=self.c("dataset"), mono=True, cls="grow")}'
                f'{h.button(icon_name="folder", act_attr=self.a("browse"), title="Browse the server")}'
                f'{h.button("Datasets…", act_attr=self.a("pick"), small=False)}</div></label>'
                f'<p class="muted small">Can be left empty: set it later in the Sources tab of the open workspace, or let the '
                f"first run added bring it (its '{h.e(param)}' parameter).</p>{add}</div>")

    def footer(self) -> str:
        return (h.button("Cancel", act_attr=self.a("close"))
                + h.button("Create", act_attr=self.a("create"))
                + h.button("Create and open", act_attr=self.a("create", open_it=True), kind="primary", icon_name="external"))

    def on_group(self, value: str = "", **_):
        self.group = value or self.group

    def on_create_enter(self, value: str = "", **_):
        self.name = value or self.name
        self.on_create()

    def on_name(self, value: str = "", **_):
        self.name = value or ""

    def on_dataset(self, value: str = "", **_):
        self.dataset = (value or "").strip()

    def on_add_keys(self, value=False, **_):
        self.add_keys = bool(value)

    def on_browse(self, **_):
        def picked(p):
            self.dataset = p
            self.refresh()
        self.session.open_dialog(FileBrowserDialog(picked, self.dataset, {".csv"}, title="Which dataset CSV?"))

    def on_pick(self, **_):
        from lookin.web.views.mlflow_dialogs import DatasetsDialog

        def picked(p):
            self.dataset = p
            self.refresh()
        self.session.open_dialog(DatasetsDialog(self.session, on_pick=picked, pick_label="Use this dataset"))

    def on_create(self, open_it=False, **_):
        if not (self.name or "").strip():
            self.error = "Enter a name."
            self.refresh()
            return
        path = self.dataset
        ws = lib.create_workspace(self.name.strip(), lib.Mount(dataset_path=path),
                                  hash_dataset=bool(path) and Path(path).is_file())
        self.close()
        home = self.home
        if ws is None:
            self.session.toast("Could not create the workspace.", "error")
            return
        if self.group:
            bd.move(ws.id, self.group, known_ids=[w.id for w in lib.list_workspaces()])
        home.expanded.add(ws.id)
        home.reload()
        home.s.mark("home")
        self.session.toast(f"Created {ws.label()} ({ws.id}).", "ok")
        if self.add_keys and self.keys:
            home.add_keys(ws, self.keys)
        if open_it:
            # Runs added above that finish downloading later reach the open
            # tab on their own (see workspaces.follow_workspace).
            self.session.cmd(open=f"/?workspace={quote(ws.id, safe='')}")


class SetDatasetDialog(Dialog):
    width = 580

    def __init__(self, home: HomeView, ws):
        super().__init__(f"Dataset of {ws.label()}")
        self.home = home
        self.ws = ws
        self.dataset = ws.mount.dataset_path

    def body(self) -> str:
        return (f'<div class="row-nw">{h.text_input(self.dataset, change=self.c("dataset"), mono=True, cls="grow", label="")}'
                f'{h.button(icon_name="folder", act_attr=self.a("browse"), title="Browse the server")}'
                f'{h.button("Datasets…", act_attr=self.a("pick"))}</div>'
                '<p class="muted small" style="margin-top:6px">Every experiment in a workspace is scored against this one dataset.</p>')

    def footer(self) -> str:
        return h.button("Cancel", act_attr=self.a("close")) + h.button("Save", act_attr=self.a("save"), kind="primary")

    def on_dataset(self, value: str = "", **_):
        self.dataset = (value or "").strip()

    def on_browse(self, **_):
        def picked(p):
            self.dataset = p
            self.refresh()
        self.session.open_dialog(FileBrowserDialog(picked, self.dataset, {".csv"}, title="Which dataset CSV?"))

    def on_pick(self, **_):
        from lookin.web.views.mlflow_dialogs import DatasetsDialog

        def picked(p):
            self.dataset = p
            self.refresh()
        self.session.open_dialog(DatasetsDialog(self.session, on_pick=picked, pick_label="Use this dataset"))

    def on_save(self, **_):
        path = self.dataset
        digest = lib.hash_file(path) if path and Path(path).is_file() else ""
        self.close()

        def change(fresh):
            fresh.mount.dataset_path, fresh.dataset_hash = path, digest
            return True
        fresh = lib.update_workspace(self.ws.path, change)
        if fresh is not None:
            self.home.after_workspace_change(fresh)
            self.session.toast(f"{fresh.label()} now uses {Path(path).name or 'no dataset'}.", "ok")


class MissingFilesDialog(Dialog):
    width = 540
    title = "Not downloaded yet"

    def __init__(self, home: HomeView, ws, link: str, missing: list):
        super().__init__()
        self.home = home
        self.ws = ws
        self.link = link
        self.missing = missing

    def body(self) -> str:
        size = wss.missing_bytes(self.ws)
        return (f"<p>{len(self.missing)} file(s) of {h.e(self.ws.label())} are still on the MLflow server"
                + (f" ({h.e(h.human_size(size))})." if size else ".")
                + " Download them first, or open the workspace without those experiments.</p>")

    def footer(self) -> str:
        return (h.button("Cancel", act_attr=self.a("close"))
                + f'<a class="btn" href="{h.e(self.link)}" target="_blank" rel="noopener" data-act="{h.e(self.target)}:close">Open without them</a>'
                + h.button("Download and open", act_attr=self.a("download"), kind="primary", icon_name="download"))

    def on_download(self, **_):
        self.close()
        session = self.session

        def then():
            session.open_dialog(OpenNowDialog(self.ws, self.link))

        self.home.download(self.ws, then=then)


class OpenNowDialog(Dialog):
    """After "Download and open": a browser only opens a tab in answer to
    a click it has just received, and that click was a download ago."""
    width = 440
    title = "Downloaded"

    def __init__(self, ws, link: str):
        super().__init__()
        self.ws = ws
        self.link = link

    def body(self) -> str:
        return f"<p>Everything {h.e(self.ws.label())} needs is on this machine.</p>"

    def footer(self) -> str:
        return (h.button("Later", act_attr=self.a("close"))
                + f'<a class="btn btn-primary" href="{h.e(self.link)}" target="_blank" rel="noopener" data-act="{h.e(self.target)}:close">'
                  f'{h.icon("external", 14)}<span>Open</span></a>')


class PickCsvsDialog(Dialog):
    width = 640
    title = "Which results CSVs?"

    def __init__(self, home: HomeView, ws, direct, ambiguous, problems, on_done=None):
        super().__init__()
        self.home = home
        self.ws = ws
        self.direct = direct
        self.ambiguous = ambiguous
        self.problems = problems
        self.on_done = on_done
        self.ticked: set = set()

    def body(self) -> str:
        rows = []
        current = None
        for i, c in enumerate(self.ambiguous):
            if c.run.run_id != current:
                current = c.run.run_id
                rows.append(f'<div class="subhead">{h.e(c.run.label())}</div>')
            rows.append(h.checkbox(c.csv.path, i in self.ticked, change=self.c("tick", i=i)) + "<br>")
        return (f"<p>These runs logged several. Each one you tick becomes an experiment in {h.e(self.ws.label())}.</p>"
                f'<div style="max-height:360px;overflow-y:auto">{"".join(rows)}</div>')

    def footer(self) -> str:
        return h.button("Cancel", act_attr=self.a("close")) + h.button("Add", act_attr=self.a("add"), kind="primary")

    def on_tick(self, i=0, value=False, **_):
        if value:
            self.ticked.add(int(i))
        else:
            self.ticked.discard(int(i))

    def on_add(self, **_):
        self.close()
        chosen = [c for i, c in enumerate(self.ambiguous) if i in self.ticked]
        self.home.commit_add(self.ws, self.direct + chosen, self.problems, self.on_done)
