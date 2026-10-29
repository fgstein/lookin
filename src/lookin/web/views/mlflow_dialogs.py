"""The two big windows of Home -- All runs and All datasets -- and the small
ones around them (add a server, columns, the settings file).

All runs is the full MLflow table, Home's runs panel at full size: every
configured server, the experiments listed (and hidden) per server, the
run search, MLflow's own filter syntax, failed/killed runs on demand,
user-chosen param/metric columns, a run expanding to its results CSVs,
and a selection that goes into a workspace -- or, opened from a
workspace's Sources tab, straight into the session on screen.
"""
from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path

from lookin.core import library as lib
from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.core.home_config import METRIC_PREFIX, PARAM_PREFIX, column_label, column_value
from lookin.web.home_model import (
    SELECTED_ALL, SELECTED_NONE, SELECTED_SOME, csv_key, csv_selected, parse_key, run_key, run_selection,
    set_selected, unselect_run, workspace_color,
)
from lookin.core.mlflow_client import UNSUCCESSFUL_STATUSES
from lookin.web import datasets as dsc
from lookin.web import html as h
from lookin.web.mlflow_hub import MlflowHub
from lookin.web.views.base import ConfirmDialog, Dialog
from lookin.web.views.files import FileBrowserDialog, menu
from lookin.web.views.media import crop_img

RUN_PAGE = 100
STATUS_DOT = {"FINISHED": "var(--ok)", "RUNNING": "var(--acc)", "SCHEDULED": "var(--ghost)",
              "FAILED": "var(--err)", "KILLED": "var(--warn-dot)"}


def _fmt_ms(ms) -> str:
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return ""


def _follow_hub(dialog, session) -> None:
    """Redraw this window whenever the shared MLflow listing changes, for as
    long as it is open."""
    key = (session.sid, dialog.id)
    hub = MlflowHub.get()
    hub.subscribe(key, lambda: session.mark("dialogs") if session.dialog(dialog.id) is not None
                  else hub.unsubscribe(key))
    original_close = dialog.close

    def close():
        hub.unsubscribe(key)
        original_close()

    dialog.close = close
    dialog.on_close = lambda **_: close()


def confirm_remove_server(session, uri: str) -> None:
    """Ask, then forget an MLflow server -- from Home's server chip or from
    the All runs window. Only the view settings change: workspaces keep the
    experiments they took from it, and downloaded files stay."""
    hub = MlflowHub.get()
    server = hub.servers.get(uri)
    if server is None:
        return

    def yes():
        hub.config.remove_server(uri)
        hub.sync_servers()
        hub.changed()
        session.toast(f"Removed {server.name}.", "neutral")
        session.mark("home", "dialogs")

    session.open_dialog(ConfirmDialog(
        f"Remove the server {server.name}?",
        f"{server.uri} and its runs leave the lists. Workspaces keep the experiments they took from it, "
        "and files already downloaded stay. \"+ server\" adds it back.",
        yes, yes_label="Remove"))


def _badges(workspaces) -> str:
    return "".join(f'<span class="badge" style="background:{workspace_color(w.id)}" title="In workspace {h.e(w.label())}">'
                   f'{h.e(w.label()[:14])}</span> ' for w in workspaces)


class RunsDialog(Dialog):
    width = 1320
    uses_hub = True

    def __init__(self, session, mode: str = "home"):
        super().__init__("All MLflow runs")
        self.s = session
        self.hub = MlflowHub.get()
        self.mode = mode                      # "home" | "session"
        self.selected: set = set()
        self.expanded: set = set()
        self.server_off: set = set()
        self.exp_filter = ""
        self.only_free = False
        self.show_hidden = False
        self.limit = RUN_PAGE
        self.dest_ws = ""
        _follow_hub(self, session)
        self.hub.activate()

    def title_html(self) -> str:
        sub = "load into this session" if self.mode == "session" else ""
        return f'<span>All MLflow runs</span>{f"<span class=muted small>{h.e(sub)}</span>" if sub else ""}'

    # ---- rendering ----

    def render(self) -> str:
        html = super().render()
        return html.replace('class="dlg"', 'class="dlg dlg-big"', 1)

    def body(self) -> str:
        hub = self.hub
        cfg = hub.config
        workspaces = lib.list_workspaces()
        by_run, by_csv = hub.membership(workspaces)
        total = shown = 0
        groups = []
        columns = cfg.columns
        several = len(hub.servers) > 1
        for server in list(hub.servers.values()):
            runs = hub.visible_runs(server)
            if self.show_hidden:
                hidden = cfg.hidden_run_ids(server.uri)
                runs = runs + [r for r in server.runs if r.run_id in hidden]
            total += len(runs)
            if server.uri in self.server_off:
                continue
            if self.only_free:
                runs = [r for r in runs if not by_run.get((server.uri, r.run_id))]
            groups.append((server, runs))
        rows = []
        rendered = 0
        for server, runs in groups:
            current = None
            for run in runs:
                if rendered >= self.limit:
                    break
                if run.experiment_id != current:
                    current = run.experiment_id
                    exp = server.experiment_by_id(current)
                    name = exp.name if exp else current
                    if several:
                        name = f"{server.name} / {name}"
                    n_exp = sum(1 for r in runs if r.experiment_id == current)
                    warn = (" · server unreachable, showing the cached listing" if server.status == "down" and server.offline else "")
                    rows.append(f'<tr class="grp"><td colspan="{6 + len(columns)}"><div class="row-nw" style="height:26px">'
                                f'<b>{h.e(name)}</b><span class="small muted">{n_exp} run(s)</span>'
                                f'<span class="small" style="color:var(--err-ink)">{h.e(warn)}</span>'
                                f'{h.link_button("", mlf.experiment_ui_url(server.uri, current), icon_name="external", title="Open this experiment in MLflow", small=True)}'
                                f'</div></td></tr>')
                rows.append(self._run_row(server, run, by_run, columns))
                shown += 1
                rendered += 1
                if run_key(server.uri, run.run_id) in self.expanded:
                    rows.extend(self._artifact_rows(server, run, by_csv, columns))
        if not rows:
            loading = any(s.status == "loading" for s in hub.servers.values())
            rows.append(f'<tr><td colspan="{6 + len(columns)}">' + h.empty(
                "Loading runs..." if loading else ("No run matches." if any(s.runs for s in hub.servers.values())
                                                  else "No runs to show. Tick an experiment on the left.")) + "</td></tr>")
        more = ""
        if total > rendered and rendered >= self.limit:
            more = (f'<div class="lst-foot">{rendered:,} of {total:,} shown '
                    f'{h.button(f"Show {RUN_PAGE} more", act_attr=self.a("more"), kind="link", small=True)}</div>')
        head_cols = "".join(f'<th title="{h.e(c["key"])}">{h.e(column_label(c))}</th>' for c in columns)
        table = (f'<table class="tbl sticky compact"><thead><tr><th style="width:44px"></th><th>Run</th><th>Status</th>'
                 f'<th>Started</th>{head_cols}<th>In workspaces</th><th style="width:56px"></th></tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table>')
        truncated = any(s.truncated for s in hub.servers.values())
        self._counts = (shown, total, truncated)
        return (f'<div class="dlg-cols">{self._facets()}<div class="pane">{self._toolbar()}{table}{more}</div></div>')

    def _toolbar(self) -> str:
        cfg = self.hub.config
        return (f'<div class="toolbar" style="position:sticky;top:0;background:var(--surface);z-index:2">'
                f'{h.text_input(cfg.runs.get("search", ""), input_=self.i("search"), type_="search", placeholder="Search run names and parameters", width="260px")}'
                f'{h.text_input(cfg.runs.get("filter", ""), enter=h.on_enter(self.target, "filter"), change=self.c("filter"), placeholder="MLflow filter, e.g. metrics.val_acc > 0.9", mono=True, cls="grow", title="MLflow\'s own search syntax, sent to the server. Applied on Enter or on leaving the field.")}'
                f'{h.checkbox("Failed and killed", bool(cfg.runs.get("show_unsuccessful")), change=self.c("unsuccessful"))}'
                f'{h.button("Columns", act_attr=self.a("columns"), small=True, icon_name="columns")}'
                f'{h.button("Settings", act_attr=self.a("settings"), small=True, icon_name="code", title="Edit the view settings (home_view.json)")}'
                f'{h.button("Refresh", act_attr=self.a("refresh"), small=True, icon_name="refresh")}</div>')

    def _facets(self) -> str:
        hub = self.hub
        cfg = hub.config
        out = ['<div class="facets"><div class="facet-h first">Servers</div>']
        for server in hub.servers.values():
            color = {"ok": "var(--ok)", "down": "var(--err)", "loading": "var(--warn-dot)"}.get(server.status, "var(--ghost)")
            note = {"ok": "connected", "loading": "loading…", "down": "unreachable" + (" · cached listing" if server.offline else "")}.get(server.status, "")
            if server.error and server.status != "down":
                note = server.error[:80]
            checked = " checked" if server.uri not in self.server_off else ""
            out.append(f'<div class="row-nw" style="align-items:flex-start;gap:0;padding-right:6px">'
                       f'<label class="facet-chk grow" style="min-width:0"><input type="checkbox"{checked}{self.c("server_toggle", uri=server.uri)}>'
                       f'<span style="display:flex;flex-direction:column;min-width:0"><span class="row-nw" style="font-weight:500">'
                       f'{h.dot(color, size=6)}{h.e(server.name)}</span>'
                       f'<span class="mono small muted ell" title="{h.e(server.uri)}">{h.e(server.uri)}</span>'
                       f'<span class="small muted">{h.e(note)}</span></span></label>'
                       f'{h.button(icon_name="unlink", act_attr=self.a("remove_server", uri=server.uri), small=True, kind="ghost", title="Remove this server")}</div>')
        if not hub.servers:
            out.append('<div class="small muted" style="padding:0 12px">No server yet.</div>')
        out.append(f'<div style="padding:4px 12px">{h.button("+ Connect a server", act_attr=self.a("add_server"), small=True)}</div>')
        out.append('<div class="facet-h">MLflow experiments</div>')
        out.append(f'<div style="padding:0 12px 6px">{h.text_input(self.exp_filter, input_=self.i("exp_filter"), type_="search", placeholder="Search experiments", width="100%")}</div>')
        q = self.exp_filter.strip().lower()
        for server in hub.servers.values():
            if not server.experiments:
                continue
            shown = set(cfg.shown_experiments(server.uri))
            hidden = set(cfg.hidden_experiments(server.uri))
            exps = sorted(server.experiments, key=lambda e: -(e.last_update_time or 0))
            matching = [e for e in exps if not q or q in e.name.lower()]
            visible = [e for e in matching if e.name not in hidden]
            hidden_rows = [e for e in matching if e.name in hidden]
            ticked = sum(1 for e in exps if e.name in shown and e.name not in hidden)
            out.append(f'<div class="row-nw" style="padding:4px 12px 2px;font-size:11px;font-weight:600;color:var(--muted)">'
                       f'{h.e(server.name)} · {ticked} of {len(exps)} listed<span class="sp"></span>'
                       f'{h.button("all", act_attr=self.a("tick_all", uri=server.uri, value=True), kind="link", small=True, title="Tick all matching")}'
                       f'{h.button("none", act_attr=self.a("tick_all", uri=server.uri, value=False), kind="link", small=True, title="Untick all matching")}</div>')
            for e in visible[:200]:
                runs = server.runs_by_exp.get(e.experiment_id)
                count = f"{len(runs)}" if (e.name in shown and runs is not None) else ""
                on = " checked" if e.name in shown else ""
                out.append(f'<div class="row-nw" style="padding:0 6px 0 12px;min-height:24px">'
                           f'<label class="chk grow" style="min-width:0" title="{h.e((e.description or e.name) + chr(10) + "updated " + _fmt_ms(e.last_update_time))}">'
                           f'<input type="checkbox"{on}{self.c("exp_toggle", uri=server.uri, name=e.name)}>'
                           f'<span class="ell">{h.e(e.name)}</span></label>'
                           f'<span class="mono small muted">{count}</span>'
                           f'{h.button(icon_name="eye-off", act_attr=self.a("exp_hide", uri=server.uri, name=e.name, hidden=True), small=True, kind="ghost", title="Hide from this list (and from the table)")}</div>')
            if hidden_rows:
                out.append(f'<div class="small muted" style="padding:6px 12px 2px;font-weight:600">Hidden ({len(hidden_rows)})</div>')
                for e in hidden_rows[:200]:
                    out.append(f'<div class="row-nw" style="padding:0 6px 0 12px;min-height:24px;color:var(--faint)">'
                               f'<span class="ell grow small">{h.e(e.name)}</span>'
                               f'{h.button(icon_name="eye", act_attr=self.a("exp_hide", uri=server.uri, name=e.name, hidden=False), small=True, kind="ghost", title="Show in this list again")}</div>')
        out.append('<div class="facet-h">Other</div>')
        out.append(f'<label class="facet-chk"><input type="checkbox"{" checked" if self.only_free else ""}{self.c("only_free")}>Not in any workspace</label>')
        n_hidden = len(cfg.runs.get("hidden", []))
        out.append(f'<label class="facet-chk"><input type="checkbox"{" checked" if self.show_hidden else ""}{self.c("show_hidden")}>Show hidden runs ({n_hidden})</label>')
        if n_hidden:
            out.append(f'<div style="padding:2px 12px">{h.button(f"Unhide all {n_hidden}", act_attr=self.a("unhide_all"), small=True)}</div>')
        out.append("</div>")
        return "".join(out)

    def _run_row(self, server, run, by_run, columns) -> str:
        key = run_key(server.uri, run.run_id)
        expanded = key in self.expanded
        state = run_selection(self.selected, server, run.run_id)
        sel = state != SELECTED_NONE
        hidden = run.run_id in self.hub.config.hidden_run_ids(server.uri)
        tip = "\n".join(filter(None, [run.run_name or run.run_id, f"{run.status} -- started {_fmt_ms(run.start_time)}",
                                      f"by {run.user}" if run.user else "", run.description]))
        cols = "".join(f'<td class="mono small nowrap" title="{h.e(column_value(run, c))}">{h.e(column_value(run, c))}</td>' for c in columns)
        rmenu = menu([
            ("Add to a workspace…", self.a("add_to", keys=[key]), "send", False) if self.mode == "home"
            else ("Load into this session", self.a("load", keys=[key]), "download", False),
            ("Show this run again" if hidden else "Hide this run", self.a("hide_run", uri=server.uri, run_id=run.run_id, name=run.label(), hidden=not hidden), "eye-off", False),
            ("Copy run id", self.a("copy", text=run.run_id), "copy", False),
        ], title="Run actions")
        cls = []
        if sel:
            cls.append("sel")
        if server.offline or hidden:
            cls.append("dim")
        return (f'<tr{f" class={chr(34)}{" ".join(cls)}{chr(34)}" if cls else ""} data-key="r-{h.e(run.run_id)}">'
                f'<td class="nowrap"><input type="checkbox"{" checked" if state == SELECTED_ALL else ""}'
                f'{" data-indeterminate=1" if state == SELECTED_SOME else ""}{self.c("select", key=key)} aria-label="Select {h.e(run.label())}">'
                f'<button type="button" class="chev"{self.a("toggle_run", key=key)} style="font-size:9px" title="Show its CSVs">{"▼" if expanded else "▶"}</button></td>'
                f'<td class="mono nowrap" title="{h.e(tip)}">{h.e(run.label())}</td>'
                f'<td class="nowrap"><span class="row-nw small">{h.dot(STATUS_DOT.get(run.status, "var(--ghost)"), size=6)}{h.e(run.status.title())}</span></td>'
                f'<td class="small muted nowrap">{h.e(_fmt_ms(run.start_time))}</td>'
                f'{cols}<td class="nowrap">{_badges(by_run.get((server.uri, run.run_id), []))}</td>'
                f'<td class="nowrap">{h.link_button("", mlf.run_ui_url(server.uri, run.experiment_id, run.run_id), icon_name="external", title="Open this run in MLflow", small=True)}{rmenu}</td></tr>')

    def _artifact_rows(self, server, run, by_csv, columns) -> list:
        entry = server.artifacts.get(run.run_id) or {"loading": True}
        span = 6 + len(columns)

        def note(text, color="var(--faint)"):
            return [f'<tr><td colspan="{span}"><div class="small" style="padding:2px 0 4px 70px;font-style:italic;color:{color}">{h.e(text)}</div></td></tr>']

        if entry.get("loading"):
            return note("Listing its artifacts...")
        if "error" in entry:
            return note(f"Could not list its artifacts: {entry['error']}", "var(--err)")
        candidates = wss.candidates_for_run(server.uri, run, entry["csvs"], entry["npzs"])
        if not candidates:
            return note("No CSV in this run's artifacts.")
        out = []
        for c in candidates:
            key = csv_key(server.uri, run.run_id, c.csv.path)
            sel = csv_selected(self.selected, key)
            sidecars = [n for n, art in (("embeddings", c.embeddings), ("prototypes", c.prototypes)) if art]
            cached = mlflow_cache.is_cached(run.run_id, c.csv.path, c.csv.size)
            out.append(
                f'<tr{" class=sel" if sel else ""}><td style="padding-left:34px"><input type="checkbox"{" checked" if sel else ""}{self.c("select", key=key)}></td>'
                f'<td colspan="3"><span class="row-nw">{h.icon("table", 13)}<span class="mono small ell" title="{h.e(c.csv.path)}">{h.e(c.csv.path)}</span>'
                f'{f"<span class=small style=color:var(--acc)>+ {h.e(chr(32).join(sidecars))}</span>" if sidecars else ""}'
                f'<span class="small muted">{h.e(h.human_size(c.csv.size))}</span>'
                f'{h.dot("var(--ok)", "Already downloaded", 6) if cached else ""}</span></td>'
                f'<td colspan="{len(columns)}"></td><td>{_badges(by_csv.get((server.uri, run.run_id, c.csv.path), []))}</td>'
                f'<td>{h.link_button("", mlf.run_ui_url(server.uri, run.experiment_id, run.run_id, c.csv.path), icon_name="external", title="Open this artifact in MLflow", small=True)}</td></tr>')
        return out

    def footer(self) -> str:
        shown, total, truncated = getattr(self, "_counts", (0, 0, False))
        n = len(self.selected)
        bits = f'<span class="small muted">{total:,} run(s){" · listing capped -- narrow it with a filter" if truncated else ""}</span>'
        sel = (f'<span style="font-weight:500">{"No run selected" if not n else f"{n} selected"}</span>'
               f'{h.button("Clear", act_attr=self.a("clear"), kind="ghost", small=True) if n else ""}')
        if self.mode == "session":
            actions = h.button(f"Load {n} into this session" if n else "Load into this session",
                               act_attr=self.a("load", keys=sorted(self.selected)), kind="primary", disabled=not n,
                               icon_name="download")
        else:
            workspaces = lib.list_workspaces()
            opts = [(w.id, w.label()) for w in workspaces]
            if not self.dest_ws and opts:
                self.dest_ws = opts[0][0]
            actions = (h.select(opts, self.dest_ws, change=self.c("dest"), label="Add to") if opts else "")
            actions += h.button("Add", act_attr=self.a("add_to", keys=sorted(self.selected)), disabled=not n or not opts, cls="btn-acc")
            actions += h.button("New workspace from selection", act_attr=self.a("new_ws"), kind="primary", disabled=not n)
        note = '<span class="small muted">Adding is instant; files download when you choose to.</span>' if self.mode == "home" else ""
        return f'{bits}{sel}{note}<span class="sp"></span>{actions}{h.button("Close", act_attr=self.a("close"))}'

    # ---- actions: filters ----

    def on_search(self, value: str = "", **_):
        self.hub.config.set_run_option("search", value or "")
        self.limit = RUN_PAGE
        self.refresh()

    def on_filter(self, value: str = "", **_):
        value = (value or "").strip()
        if value == self.hub.config.runs.get("filter", ""):
            return
        self.hub.config.set_run_option("filter", value)
        for server in self.hub.servers.values():
            server.error = ""
        self.hub.refresh(runs_only=True)

    def on_unsuccessful(self, value=False, **_):
        self.hub.config.set_run_option("show_unsuccessful", bool(value))
        self.refresh()

    def on_refresh(self, **_):
        self.hub.refresh()

    def on_more(self, **_):
        self.limit += RUN_PAGE
        self.refresh()

    def on_server_toggle(self, uri: str = "", value=True, **_):
        if value:
            self.server_off.discard(uri)
        else:
            self.server_off.add(uri)
        self.refresh()

    def on_add_server(self, **_):
        self.s.open_dialog(AddServerDialog(self.hub))

    def on_remove_server(self, uri: str = "", **_):
        self.server_off.discard(uri)
        confirm_remove_server(self.s, uri)

    def on_exp_filter(self, value: str = "", **_):
        self.exp_filter = value or ""
        self.refresh()

    def _after_experiments(self, server, before: set) -> None:
        with self.hub.lock:
            server.keep_only(self.hub.wanted_ids(server))
        new = [i for i in self.hub.wanted_ids(server) if i not in before]
        if new:
            self.hub.refresh(server.uri, runs_only=True, only=new)
        self.hub.changed()

    def on_exp_toggle(self, uri: str = "", name: str = "", value=False, **_):
        server = self.hub.servers.get(uri)
        if server is None:
            return
        before = set(self.hub.wanted_ids(server))
        cfg = self.hub.config
        shown = set(cfg.shown_experiments(uri))
        hidden = set(cfg.hidden_experiments(uri))
        if value:
            shown.add(name)
        else:
            shown.discard(name)
        cfg.set_experiments(uri, shown, hidden)
        self._after_experiments(server, before)

    def on_exp_hide(self, uri: str = "", name: str = "", hidden=True, **_):
        server = self.hub.servers.get(uri)
        if server is None:
            return
        before = set(self.hub.wanted_ids(server))
        self.hub.config.set_experiment_hidden(uri, name, bool(hidden))
        self._after_experiments(server, before)

    def on_tick_all(self, uri: str = "", value=True, **_):
        server = self.hub.servers.get(uri)
        if server is None:
            return
        before = set(self.hub.wanted_ids(server))
        cfg = self.hub.config
        shown = set(cfg.shown_experiments(uri))
        hidden = set(cfg.hidden_experiments(uri))
        q = self.exp_filter.strip().lower()
        for e in server.experiments:
            if e.name in hidden or (q and q not in e.name.lower()):
                continue
            if value:
                shown.add(e.name)
            else:
                shown.discard(e.name)
        cfg.set_experiments(uri, shown, hidden)
        self._after_experiments(server, before)

    def on_only_free(self, value=False, **_):
        self.only_free = bool(value)
        self.refresh()

    def on_show_hidden(self, value=False, **_):
        self.show_hidden = bool(value)
        self.refresh()

    def on_unhide_all(self, **_):
        self.hub.config.unhide_all_runs()
        self.refresh()

    def on_columns(self, **_):
        self.s.open_dialog(ColumnsDialog(self.hub))

    def on_settings(self, **_):
        self.s.open_dialog(ConfigDialog(self.hub))

    # ---- actions: rows ----

    def on_select(self, key: str = "", value=False, **_):
        _, uri, _, _ = parse_key(key)
        set_selected(self.selected, self.hub.servers.get(uri), key, bool(value))
        self.refresh()

    def on_clear(self, **_):
        self.selected.clear()
        self.refresh()

    def on_toggle_run(self, key: str = "", **_):
        if key in self.expanded:
            self.expanded.discard(key)
        else:
            self.expanded.add(key)
            _, uri, run_id, _ = parse_key(key)
            server = self.hub.servers.get(uri)
            if server is not None:
                self.hub.fetch_artifacts(server, run_id)
        self.refresh()

    def on_hide_run(self, uri: str = "", run_id: str = "", name: str = "", hidden=True, **_):
        self.hub.config.set_run_hidden(uri, run_id, name, bool(hidden))
        if hidden:
            server = self.hub.servers.get(uri)
            if server is not None:
                unselect_run(self.selected, server, run_id)
            self.s.toast(f"Hid {name}. 'Show hidden runs' brings it back.", "neutral")
        self.refresh()

    def on_copy(self, text: str = "", **_):
        self.s.cmd(copy=text)
        self.s.toast(f"Copied {text}.", "neutral")

    def on_dest(self, value: str = "", **_):
        self.dest_ws = value

    def on_add_to(self, keys=None, **_):
        keys = list(keys or [])
        if not keys:
            return
        home = self.s.view("home")
        workspaces = lib.list_workspaces()
        if not workspaces:
            home.on_new_workspace(keys=keys)
            return
        if len(keys) == 1 and keys[0] not in self.selected:
            self.s.open_dialog(AddToDialog(self, keys))
            return
        ws = next((w for w in workspaces if w.id == self.dest_ws), workspaces[0])
        home.add_keys(ws, keys, on_done=self._added)

    def _added(self) -> None:
        self.selected.clear()
        self.refresh()

    def on_new_ws(self, **_):
        self.s.view("home").on_new_workspace(keys=sorted(self.selected))

    # ---- loading into the session (mode "session") ----

    def on_load(self, keys=None, **_):
        keys = list(keys or [])
        if not keys:
            return
        if self.s.state.dataset is None:
            self.s.toast("Load the dataset in the Sources tab first.", "warn")
            return
        hub = self.hub
        pending = []
        for key in keys:
            kind, uri, run_id, _ = parse_key(key)
            server = hub.servers.get(uri)
            if server is None:
                continue
            if server.needs_listing(run_id):
                pending.append((server, run_id))
        if pending:
            self.s.toast(f"Listing the artifacts of {len(pending)} run(s)...", "neutral")
            remaining = {"n": len(pending)}

            def one_done():
                remaining["n"] -= 1
                if remaining["n"] == 0:
                    with self.s.lock:
                        self.on_load(keys=keys)
                    self.s.push({"dirty": ["dialogs"]})

            for server, run_id in pending:
                server.artifacts.pop(run_id, None)
                hub.fetch_artifacts(server, run_id, then=one_done)
            return
        candidates, problems = [], []
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
            cands = wss.candidates_for_run(uri, run, entry.get("csvs", []), entry.get("npzs", []))
            if kind == "csv":
                candidates.extend(c for c in cands if c.csv.path == path)
            elif not cands:
                problems.append(f"{run.label()}: no CSV in its artifacts")
            else:
                candidates.extend(cands)
        for p in problems:
            self.s.toast(p, "error")
        if not candidates:
            return
        self.close()
        session = self.s

        def work():
            sources = session.upload_view
            loaded = 0
            for c in candidates:
                taken = set(session.state.experiments) | {r.name for r in sources.rows}
                name = wss.unique_name(taken, c.default_name(), c.run.run_id)
                session.busy.begin(f"Loading {Path(c.csv.path).name}")
                try:
                    dest = mlflow_cache.download_cached(c.uri, c.run.run_id, c.csv.path, c.csv.size)
                    origin = lib.MountExperiment(
                        name=name, path=str(dest), mlflow_uri=c.uri, mlflow_experiment_id=c.run.experiment_id,
                        mlflow_run_id=c.run.run_id, mlflow_run_name=c.run.run_name, artifact_path=c.csv.path,
                        artifact_size=c.csv.size)
                    ok, message = sources.add_experiment_from_path(str(dest), name, mlflow=origin)
                    if not ok:
                        session.toast(f"{name}: {message}", "error")
                        continue
                    loaded += 1
                    for art, kind in ((c.embeddings, "embed"), (c.prototypes, "proto")):
                        if art is None:
                            continue
                        try:
                            side = mlflow_cache.download_cached(c.uri, c.run.run_id, art.path, art.size)
                            exp = session.state.experiments.get(name)
                            if kind == "embed":
                                if exp is not None and exp.embeddings_path == str(side):
                                    ok2, msg = True, f"{len(exp.embeddings_uuids):,} embeddings"
                                else:
                                    ok2, msg = session.prototype_source.load_embeddings_for(name, str(side))
                                sources.set_row_sidecar_fields(name, embed_path=str(side), embed_status=msg if ok2 else f"error: {msg}")
                            else:
                                if exp is not None and exp.model_prototypes_path == str(side):
                                    ok2, msg = True, f"{len(exp.model_prototypes.class_labels):,}-class prototypes"
                                else:
                                    ok2, msg = session.prototype_source.load_prototypes_for(name, str(side))
                                sources.set_row_sidecar_fields(name, proto_path=str(side), proto_status=msg if ok2 else f"error: {msg}")
                        except Exception as ex:  # noqa: BLE001
                            session.toast(f"{Path(art.path).name} found but could not be loaded ({ex})", "warn")
                except mlf.MlflowError as ex:
                    session.toast(str(ex)[:300], "error")
                except Exception as ex:  # noqa: BLE001
                    traceback.print_exc()
                    session.toast(f"Could not load: {ex}", "error")
                finally:
                    session.busy.end()
            if loaded:
                session.notify_change()
                session.toast(f"Loaded {loaded} experiment(s) from MLflow.", "ok")
            session.push({"dirty": ["*"]})

        threading.Thread(target=work, daemon=True).start()


class AddToDialog(Dialog):
    width = 440

    def __init__(self, runs: RunsDialog, keys: list):
        super().__init__(f"Add {len(keys)} item(s) to...")
        self.runs = runs
        self.keys = keys
        self.workspaces = lib.list_workspaces()
        self.dest = self.workspaces[0].id if self.workspaces else ""

    def body(self) -> str:
        return h.select([(w.id, w.label()) for w in self.workspaces], self.dest, change=self.c("dest"), label="Workspace")

    def footer(self) -> str:
        return h.button("Cancel", act_attr=self.a("close")) + h.button("Add", act_attr=self.a("go"), kind="primary")

    def on_dest(self, value: str = "", **_):
        self.dest = value

    def on_go(self, **_):
        self.close()
        ws = next((w for w in self.workspaces if w.id == self.dest), None)
        if ws is not None:
            self.session.view("home").add_keys(ws, self.keys)


class AddServerDialog(Dialog):
    width = 520
    title = "Add an MLflow server"

    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.uri = ""
        self.name = ""

    def body(self) -> str:
        return (f'<div class="form-col">{h.text_input(self.uri, change=self.c("uri"), label="Tracking URI", placeholder="http://mlflow.lab:5000", autofocus=True, mono=True)}'
                f'{h.text_input(self.name, change=self.c("name"), label="Short name (optional)")}'
                '<p class="muted small">A server on your own network, opt-in: Lookin only talks to the servers listed here.</p></div>')

    def footer(self) -> str:
        return h.button("Cancel", act_attr=self.a("close")) + h.button("Add", act_attr=self.a("add"), kind="primary")

    def on_uri(self, value: str = "", **_):
        self.uri = (value or "").strip()

    def on_name(self, value: str = "", **_):
        self.name = value or ""

    def on_add(self, **_):
        value = self.uri
        if not value.startswith(("http://", "https://")):
            self.error = "Enter an http:// or https:// address."
            self.refresh()
            return
        if not self.hub.config.add_server(value, self.name or ""):
            self.error = "That server is already in the list."
            self.refresh()
            return
        self.close()
        self.hub.sync_servers()
        self.hub.refresh(wss.normalize_uri(value))


class ColumnsDialog(Dialog):
    width = 520
    title = "Columns"

    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        params, metrics = set(), set()
        for server in hub.servers.values():
            for run in server.runs:
                params.update(run.params)
                metrics.update(run.metrics)
        self.chosen = [c["key"] for c in hub.config.columns]
        keys = [PARAM_PREFIX + k for k in sorted(params)] + [METRIC_PREFIX + k for k in sorted(metrics)]
        keys += [k for k in self.chosen if k not in keys]
        self.keys = keys
        self.ticked = set(self.chosen)
        self.query = ""

    def body(self) -> str:
        if not self.keys:
            return h.banner("No runs listed yet -- nothing to choose from.", "info")
        q = self.query.lower()
        rows = "".join(
            f'<div>{h.checkbox(k, k in self.ticked, change=self.c("tick", key=k))}</div>'
            for k in self.keys if q in k.lower())
        return (f'{h.text_input(self.query, input_=self.i("query"), type_="search", placeholder="Filter", width="100%")}'
                '<p class="muted small" style="margin:6px 0">Labels and order can be changed in home_view.json.</p>'
                f'<div class="picker-list" style="padding:6px 10px">{rows}</div>')

    def footer(self) -> str:
        return h.button("Cancel", act_attr=self.a("close")) + h.button("Save", act_attr=self.a("save"), kind="primary")

    def on_query(self, value: str = "", **_):
        self.query = value or ""
        self.refresh()

    def on_tick(self, key: str = "", value=False, **_):
        if value:
            self.ticked.add(key)
        else:
            self.ticked.discard(key)

    def on_save(self, **_):
        order = [k for k in self.chosen if k in self.ticked] + [k for k in self.keys if k in self.ticked and k not in self.chosen]
        self.hub.config.set_columns(order)
        self.close()
        self.hub.changed()


class ConfigDialog(Dialog):
    width = 780
    title = "View settings"

    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.text = hub.config.text()

    def body(self) -> str:
        return (f'<p class="muted small">{h.e(str(self.hub.config.path))} -- this is the file itself: edit it here or in any '
                'editor, Home follows either.</p>'
                f'<form id="{self.id}-f"{self.submit("save")}>{h.textarea(self.text, name="text", rows=22)}</form>')

    def footer(self) -> str:
        return (h.button("Cancel", act_attr=self.a("close")) + h.button("Reload from file", act_attr=self.a("reload"))
                + f'<button type="submit" form="{self.id}-f" class="btn btn-primary"><span>Save</span></button>')

    def on_save(self, text: str = "", **_):
        problem = self.hub.config.replace_text(text or "")
        if problem:
            self.text = text
            self.error = problem
            self.refresh()
            return
        self.close()
        self.hub.sync_servers()
        self.hub.refresh()

    def on_reload(self, **_):
        self.hub.config.reload()
        self.text = self.hub.config.text()
        self.error = self.hub.config.error
        self.refresh()


# ---- All datasets ------------------------------------------------------------

class DatasetsDialog(Dialog):
    width = 1320
    uses_hub = True

    def __init__(self, session, on_pick=None, pick_label: str = ""):
        super().__init__("All datasets")
        self.s = session
        self.hub = MlflowHub.get()
        self.on_pick_fn = on_pick
        self.pick_label = pick_label
        self.query = ""
        self.location = None
        self.avail = {"here", "named"}
        self.usage = set()
        self.modified = "any"
        self.selected = None
        self.loading_facts = False
        self.dest_ws = ""
        _follow_hub(self, session)

    def render(self) -> str:
        return super().render().replace('class="dlg"', 'class="dlg dlg-big"', 1)

    def _entries(self):
        return dsc.catalog(lib.list_workspaces(), self.hub)

    def body(self) -> str:
        entries = self._entries()
        if self.selected is None and entries:
            self.selected = entries[0].path
        locs: dict = {}
        for d in entries:
            key = d.folder or "not on this machine"
            locs[key] = locs.get(key, 0) + 1
        for f in dsc.watched_folders():
            locs.setdefault(f, 0)
        q = self.query.strip().lower()
        now = datetime.now().timestamp()

        def keep(d):
            if q and not all(w in (d.path + " " + d.name).lower() for w in q.split()):
                return False
            if self.location is not None and (d.folder or "not on this machine") != self.location:
                return False
            if d.here and "here" not in self.avail:
                return False
            if not d.here and "named" not in self.avail:
                return False
            if self.usage == {"used"} and not d.used_by:
                return False
            if self.usage == {"unused"} and d.used_by:
                return False
            if self.modified == "week" and (not d.mtime or now - d.mtime > 7 * 86400):
                return False
            if self.modified == "month" and (not d.mtime or now - d.mtime > 31 * 86400):
                return False
            return True

        shown = [d for d in entries if keep(d)]
        self._counts = (len(shown), len(entries))
        facets = ['<div class="facets"><div class="facet-h first">Locations</div>',
                  f'<button type="button" class="facet-b{" on" if self.location is None else ""}"{self.a("loc", value="")}>'
                  f'Everywhere<span class="sp mono small muted">{len(entries)}</span></button>']
        watched = set(dsc.watched_folders())
        for loc, n in sorted(locs.items(), key=lambda kv: kv[0]):
            on = self.location == loc
            unwatch = (h.button(icon_name="x", act_attr=self.a("unwatch", folder=loc), small=True, kind="ghost", title="Stop watching this folder")
                       if loc in watched else "")
            facets.append(f'<div class="row-nw" style="padding-right:6px"><button type="button" class="facet-b{" on" if on else ""}" title="{h.e(loc)}"{self.a("loc", value=loc)}>'
                          f'<span class="mono small ell">{h.e(loc)}</span><span class="sp mono small muted">{n}</span></button>{unwatch}</div>')
        facets.append(f'<div style="padding:4px 12px">{h.button("+ Watch a folder", act_attr=self.a("watch"), small=True)}</div>')
        facets.append('<div class="facet-h">Availability</div>')
        facets.append(f'<label class="facet-chk"><input type="checkbox"{" checked" if "here" in self.avail else ""}{self.c("avail", key="here")}>On this machine</label>')
        facets.append(f'<label class="facet-chk"><input type="checkbox"{" checked" if "named" in self.avail else ""}{self.c("avail", key="named")}>Only named by a run</label>')
        facets.append('<div class="facet-h">Usage</div>')
        facets.append(f'<label class="facet-chk"><input type="checkbox"{" checked" if "used" in self.usage else ""}{self.c("usage", key="used")}>Used by a workspace</label>')
        facets.append(f'<label class="facet-chk"><input type="checkbox"{" checked" if "unused" in self.usage else ""}{self.c("usage", key="unused")}>Not used yet</label>')
        facets.append('<div class="facet-h">Modified</div>')
        for key, label in (("any", "Any time"), ("week", "This week"), ("month", "This month")):
            facets.append(f'<label class="facet-chk"><input type="radio" name="{self.id}-mod"{" checked" if self.modified == key else ""}'
                          f'{self.c("modified", key=key)}>{h.e(label)}</label>')
        facets.append("</div>")
        rows = []
        for d in shown[:500]:
            facts = d.facts()
            sel = d.path == self.selected
            mod = datetime.fromtimestamp(d.mtime).strftime("%Y-%m-%d") if d.mtime else "not here"
            used = len(d.used_by)
            rows.append(
                f'<tr{" class=sel" if sel else ""}>'
                f'<td><button type="button" class="btn-link mono" style="border:0;background:none;padding:0;cursor:pointer;font-size:12px;{"color:var(--ghost)" if not d.here else "color:var(--ink)"}"'
                f'{self.a("select", path=d.path)}>{h.e(d.name)}</button></td>'
                f'<td class="mono small muted nowrap" title="{h.e(d.folder)}">{h.e(d.folder or "not on this machine")}</td>'
                f'<td class="num">{"" if not facts or facts.rows is None else f"{facts.rows:,}"}</td>'
                f'<td class="num">{"" if not facts else len(facts.columns)}</td>'
                f'<td class="num">{"" if not facts or facts.classes is None else facts.classes}</td>'
                f'<td class="num muted">{h.e(h.human_size(d.size) if d.size is not None else "—")}</td>'
                f'<td class="small muted nowrap">{h.e(mod)}</td>'
                f'<td class="small muted nowrap" style="text-align:right">{f"{used} workspace" + ("s" if used > 1 else "") if used else "—"}</td></tr>')
        if not rows:
            rows.append('<tr><td colspan="8">' + h.empty("No dataset here. Watch a folder, or browse the server.") + "</td></tr>")
        table = ('<table class="tbl sticky compact"><thead><tr><th>Dataset</th><th>Folder</th><th class="num">Rows</th>'
                 '<th class="num">Cols</th><th class="num">Classes</th><th class="num">Size</th><th>Modified</th>'
                 f'<th style="text-align:right">Used by</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
        toolbar = (f'<div class="toolbar" style="position:sticky;top:0;background:var(--surface);z-index:2">'
                   f'{h.text_input(self.query, input_=self.i("query"), type_="search", placeholder="Search names and paths…", width="320px")}'
                   f'<span class="small muted">{len(shown):,} of {len(entries):,} shown</span><span class="sp"></span>'
                   f'{h.button("Browse server files…", act_attr=self.a("browse"), small=True, icon_name="folder")}'
                   f'{h.button("Rescan", act_attr=self.a("rescan"), small=True, icon_name="refresh")}</div>')
        current = next((d for d in entries if d.path == self.selected), None)
        return (f'<div class="dlg-cols">{"".join(facets)}<div class="pane">{toolbar}{table}</div>'
                f'<div class="side-pane">{self._preview(current)}</div></div>')

    def _preview(self, d) -> str:
        if d is None:
            return h.empty("Select a dataset.")
        mod = datetime.fromtimestamp(d.mtime).strftime("%Y-%m-%d %H:%M") if d.mtime else "not on this machine"
        head = (f'<div><div class="mono" style="font-size:14px;font-weight:600;overflow-wrap:anywhere">{h.e(d.name)}</div>'
                f'<div class="mono small muted" style="overflow-wrap:anywhere">{h.e(d.path)}</div>'
                f'<div class="row small muted" style="margin-top:4px"><span>{h.e(h.human_size(d.size) if d.size is not None else "")}</span><span>{h.e(mod)}</span></div></div>')
        body = ""
        if not d.here:
            body = h.banner("Not on this machine. A run's parameters name this path; copy the file here or point "
                            "Lookin at the folder that holds it.", "warn")
        else:
            facts = d.facts()
            if facts is None:
                if self.loading_facts:
                    body = h.spinner("Reading the file…")
                else:
                    body = (f'<div class="banner">{h.icon("info", 15)}<div class="banner-body">Columns, roles and sample crops are '
                            f'read when you ask: a large CSV is not opened just to list it.'
                            f'<div style="margin-top:6px">{h.button("Load preview", act_attr=self.a("facts", path=d.path), small=True)}</div></div></div>')
            elif facts.error:
                body = h.banner(facts.error, "error")
            else:
                roles = (f'<div class="subhead">Detected roles</div>'
                         + h.kv_grid([("id", h.mono(facts.uuid_col or "none")), ("image", h.mono(facts.crop_col or "none")),
                                      ("label", h.mono(facts.label_col or "none"))])
                         + f'<p class="muted small" style="margin-top:4px">{h.e(facts.note)}</p>')
                counts = (f'<div class="row small"><b>{facts.rows:,}</b> rows · <b>{len(facts.columns)}</b> columns'
                          f'{f" · <b>{facts.classes}</b> classes" if facts.classes is not None else ""}</div>')
                cols = "".join(f'<div class="row-nw small" style="height:20px"><span class="mono">{h.e(c)}</span>'
                               f'<span class="sp muted">{"id" if c == facts.uuid_col else "image path" if c == facts.crop_col else "label" if c == facts.label_col else ""}</span></div>'
                               for c in facts.columns[:60])
                samples = dsc.sample_crops(d.path, facts)
                thumbs = "".join(crop_img(self.s, p, size=46, alt=lbl, zoom=True, title=lbl) for p, lbl in samples)
                body = (counts + roles + f'<div class="subhead">Columns</div>{cols}'
                        + (f'<div class="subhead">Sample crops</div><div class="row" style="gap:4px">{thumbs}</div>' if thumbs else ""))
        used = ", ".join(w.label() for w in d.used_by) or "No workspace yet"
        named = "".join(f'<div class="mono small">{h.e(srv)} › {h.e(exp)} › {h.e(run)}</div>' for srv, exp, run in d.named_by[:12])
        if len(d.named_by) > 12:
            named += f'<div class="small muted">... and {len(d.named_by) - 12} more</div>'
        return (f'{head}{body}<div><div class="subhead">Used by</div><div class="small">{h.e(used)}</div></div>'
                f'<div><div class="subhead">Named by MLflow runs</div>{named or "<div class=small>none</div>"}</div>')

    def footer(self) -> str:
        sel = self.selected
        name = Path(sel).name if sel else ""
        left = f'<span class="small" style="font-weight:500">Selected: {h.mono(name) if name else "—"}</span><span class="sp"></span>'
        if self.on_pick_fn is not None:
            return left + h.button("Cancel", act_attr=self.a("close")) + h.button(
                self.pick_label or "Use this dataset", act_attr=self.a("pick"), kind="primary", disabled=not sel)
        workspaces = lib.list_workspaces()
        opts = [(w.id, w.label()) for w in workspaces]
        if not self.dest_ws and opts:
            self.dest_ws = opts[0][0]
        use = (h.select(opts, self.dest_ws, change=self.c("dest"), label="Use in")
               + h.button("Set as dataset", act_attr=self.a("set_dataset"), disabled=not sel, cls="btn-acc")) if opts else ""
        return (left + use + h.button("New workspace on it", act_attr=self.a("new_ws"), kind="primary", disabled=not sel)
                + h.button("Close", act_attr=self.a("close")))

    # ---- actions ----

    def on_query(self, value: str = "", **_):
        self.query = value or ""
        self.refresh()

    def on_loc(self, value: str = "", **_):
        self.location = value or None
        self.refresh()

    def on_avail(self, key: str = "", value=False, **_):
        (self.avail.add if value else self.avail.discard)(key)
        self.refresh()

    def on_usage(self, key: str = "", value=False, **_):
        (self.usage.add if value else self.usage.discard)(key)
        self.refresh()

    def on_modified(self, key: str = "any", **_):
        self.modified = key
        self.refresh()

    def on_select(self, path: str = "", **_):
        self.selected = path
        self.refresh()

    def on_facts(self, path: str = "", **_):
        self.loading_facts = True
        session = self.s

        def work():
            dsc.read_facts(path)
            self.loading_facts = False
            session.push({"dirty": ["dialogs", "home"]})

        threading.Thread(target=work, daemon=True).start()
        self.refresh()

    def on_watch(self, **_):
        def picked(folder):
            dsc.add_watched_folder(folder)
            self.location = folder
            self.refresh()
        self.s.open_dialog(FileBrowserDialog(picked, "", dirs_only=True, title="Watch which folder for dataset CSVs?"))

    def on_unwatch(self, folder: str = "", **_):
        dsc.remove_watched_folder(folder)
        if self.location == folder:
            self.location = None
        self.refresh()

    def on_browse(self, **_):
        def picked(path):
            self.selected = path
            if self.on_pick_fn is not None:
                self.close()
                self.on_pick_fn(path)
                return
            self.refresh()
        self.s.open_dialog(FileBrowserDialog(picked, "", {".csv"}, title="Select a dataset CSV"))

    def on_rescan(self, **_):
        self.refresh()

    def on_pick(self, **_):
        if self.selected and self.on_pick_fn is not None:
            self.close()
            self.on_pick_fn(self.selected)

    def on_dest(self, value: str = "", **_):
        self.dest_ws = value

    def on_set_dataset(self, **_):
        ws = next((w for w in lib.list_workspaces() if w.id == self.dest_ws), None)
        path = self.selected
        if ws is None or not path:
            return
        home = self.s.view("home")
        home.set_dataset(ws, path)

    def on_new_ws(self, **_):
        from lookin.web.views.home import NewWorkspaceDialog
        dlg = NewWorkspaceDialog(self.s.view("home"))
        dlg.dataset = self.selected or ""
        dlg.name = Path(self.selected).stem if self.selected else ""
        self.s.open_dialog(dlg)
