"""`?mlflow_run=<id>`: one MLflow run, ready to go in a workspace.

What TrainHub's "Lookin" button opens (see core/links.py and
core/run_links.py): Home, with this dialog on top. It finds the run on the
configured MLflow servers, lists its results CSVs, says which workspaces
already use it (each one a link), and puts it in a new workspace or an
existing one -- the same library calls as dropping the run on Home -- then
downloads its files and opens the workspace in this tab.
"""
from __future__ import annotations

import threading
import traceback
from urllib.parse import quote

from lookin.core import board as bd
from lookin.core import library as lib
from lookin.core import mlflow_client as mlf
from lookin.core import run_links
from lookin.core import workspace_sources as wss
from lookin.web import html as h
from lookin.web.views.base import Dialog

# Where the run goes -- keys compared in code, never reworded.
TARGET_NEW = "new"
TARGET_EXISTING = "existing"


def open_run_link(session, run_id: str, server: str = "", artifact: str = "") -> None:
    """Open the dialog for `run_id` -- once per tab: a reload of the same
    address must not open it again after it was used or closed."""
    handled = session.__dict__.setdefault("_run_links", set())
    key = (run_id, server, artifact)
    if key in handled:
        return
    handled.add(key)
    session.open_dialog(RunLinkDialog(session, run_id, server, artifact))


class RunLinkDialog(Dialog):
    width = 640
    title = "MLflow run to a workspace"

    def __init__(self, session, run_id: str, server: str = "", artifact: str = ""):
        super().__init__()
        self.session = session
        self.run_id = run_id.strip()
        self.server_hint = server.strip()
        self.artifact = artifact.strip()
        self.loading = True
        self.unknown_server = ""
        self.uri = ""
        self.run = None
        self.candidates: list = []
        self.used_by: list = []          # workspaces this run is already in
        self.workspaces: list = []
        self.columns: list = []          # Home groups: (board column id, name)
        self.choice = 0                  # index into candidates
        self.destination = TARGET_NEW
        self.name = ""
        self.group = ""
        self.workspace_id = ""
        self.busy = False
        threading.Thread(target=self._load, daemon=True, name="lookin-run-link").start()

    # ---- finding the run ----

    def _load(self) -> None:
        error, unknown, found = "", "", None
        try:
            uri, run = run_links.resolve_run(self.run_id, self.server_hint)
            csvs, npzs = mlf.list_csv_and_npz_artifacts(uri, run.run_id)
            found = (uri, run, wss.candidates_for_run(uri, run, csvs, npzs))
        except run_links.RunNotFound as ex:
            error, unknown = str(ex), ex.unknown_server
        except (mlf.MlflowError, ValueError) as ex:
            error = str(ex)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            error = f"Could not read the run: {ex}"
        workspaces = lib.list_workspaces()
        with self.session.lock:
            self.loading = False
            self.error, self.unknown_server = error, unknown
            self.workspaces = sorted(workspaces, key=lambda w: w.label().lower())
            if found is not None:
                self.uri, self.run, self.candidates = found
                self.used_by = [w for w in self.workspaces
                                if any(e.mlflow_run_id == self.run.run_id for e in w.mount.experiments)]
                paths = [c.csv.path for c in self.candidates]
                self.choice = paths.index(self.artifact) if self.artifact in paths else 0
                self.name = self.candidates[self.choice].default_name() if self.candidates else self.run.label()
                if not self.candidates:
                    self.error = f"Run {self.run.label()} logged no CSV: there is nothing to explore in Lookin."
            columns = bd.load()["columns"]
            self.columns = [(c["id"], c["name"]) for c in columns]
            self.group = self.columns[0][0] if self.columns else ""
            self.workspace_id = self.workspaces[0].id if self.workspaces else ""
        self.refresh()

    # ---- drawing ----

    def body(self) -> str:
        if self.loading:
            return h.spinner(f"Looking for run {self.run_id} on the MLflow servers...")
        if self.run is None:
            add = ""
            if self.unknown_server.lower().startswith(("http://", "https://")):
                add = ('<p class="small">If that server is one of yours, add it -- Lookin only talks to the '
                       'servers listed on Home -- and look again.</p>'
                       + h.button(f"Add {self.unknown_server} and retry", act_attr=self.a("add_server"), small=True))
            return f'<p>Run <span class="mono">{h.e(self.run_id)}</span> could not be opened.</p>{add}'
        run = self.run
        url = mlf.run_ui_url(self.uri, run.experiment_id, run.run_id)
        info = h.kv_grid([
            ("Run", f'<span class="mono">{h.e(run.label())}</span> '
                    + h.link_button("", url, icon_name="external", title="Open this run in MLflow", small=True)),
            ("Status", h.e(run.status or "?")),
            ("Server", f'<span class="mono">{h.e(self.uri)}</span>'),
        ])
        used = ""
        if self.used_by:
            chips = "".join(
                f'<a class="chip" href="/?workspace={h.e(quote(w.id, safe=""))}" title="Open {h.e(w.label())} in this tab">'
                f'{h.e(w.label())}</a>' for w in self.used_by)
            used = (f'<div class="subhead">Already in</div><div class="row-wrap" style="gap:4px">{chips}</div>')
        if not self.candidates:
            return info + used
        csv = ""
        if len(self.candidates) > 1:
            csv = h.select([(str(i), c.csv.path) for i, c in enumerate(self.candidates)], str(self.choice),
                           change=self.c("csv"), label="Results CSV")
        else:
            csv = (f'<div class="small"><span class="muted">Results CSV</span> '
                   f'<span class="mono">{h.e(self.candidates[0].csv.path)}</span></div>')
        targets = [(TARGET_NEW, "A new workspace")]
        if self.workspaces:
            targets.append((TARGET_EXISTING, "An existing workspace"))
        where = ('<div class="row-nw"><span class="small muted">Put it in</span>'
                 + h.segmented(targets, self.destination, act_for=lambda k: self.a("target", value=k), label="Put it in")
                 + "</div>")
        if self.destination == TARGET_NEW:
            group = (h.select(self.columns, self.group, change=self.c("group"), label="Group")
                     if len(self.columns) > 1 else "")
            fields = h.text_input(self.name, change=self.c("name"), label="Name", autofocus=True) + group
        else:
            fields = h.select([(w.id, w.label()) for w in self.workspaces], self.workspace_id,
                              change=self.c("workspace"), label="Workspace")
        return (f'{info}{used}<div class="form-col" style="margin-top:10px">{csv}{where}{fields}'
                '<p class="muted small">Its files are downloaded, then the workspace opens in this tab.</p></div>')

    def footer(self) -> str:
        if self.loading or self.run is None or not self.candidates:
            return h.button("Close", act_attr=self.a("close"))
        label = "Create and open" if self.destination == TARGET_NEW else "Add and open"
        return (h.button("Cancel", act_attr=self.a("close"))
                + h.button(label, act_attr=self.a("go"), kind="primary", disabled=self.busy))

    # ---- actions ----

    def on_csv(self, value: str = "", **_):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self.candidates):
            previous = self.candidates[self.choice].default_name()
            self.choice = index
            if not self.name.strip() or self.name == previous:
                self.name = self.candidates[index].default_name()

    def on_target(self, value: str = "", **_):
        if value in (TARGET_NEW, TARGET_EXISTING):
            self.destination = value

    def on_name(self, value: str = "", **_):
        self.name = value or ""

    def on_group(self, value: str = "", **_):
        self.group = value or self.group

    def on_workspace(self, value: str = "", **_):
        self.workspace_id = value or self.workspace_id

    def on_add_server(self, **_):
        from lookin.web.mlflow_hub import MlflowHub

        hub = MlflowHub.get()
        hub.config.add_server(self.unknown_server)
        hub.sync_servers()
        hub.refresh(wss.normalize_uri(self.unknown_server))
        self.loading, self.error, self.unknown_server = True, "", ""
        threading.Thread(target=self._load, daemon=True, name="lookin-run-link").start()

    def on_go(self, **_):
        if self.busy or not self.candidates:
            return
        candidate = self.candidates[self.choice]
        home = self.session.view("home")
        if self.destination == TARGET_NEW:
            name = self.name.strip()
            if not name:
                self.error = "Enter a name."
                self.refresh()
                return
            ws = lib.create_workspace(name, lib.Mount(), hash_dataset=False)
            if ws is None:
                self.error = "Could not create the workspace."
                self.refresh()
                return
            if self.group:
                bd.move(ws.id, self.group, known_ids=[w.id for w in self.workspaces] + [ws.id])
                home.index.reload_board()
        else:
            ws = next((w for w in self.workspaces if w.id == self.workspace_id), None)
            if ws is None:
                self.error = "Pick a workspace."
                self.refresh()
                return
        outcome = {}

        def change(fresh):
            outcome["result"] = wss.add_candidates(fresh, [candidate], home.hub.config.dataset_param)
            return bool(outcome["result"].added or outcome["result"].dataset_set)

        fresh = lib.update_workspace(ws.path, change)
        if fresh is None:
            self.error = f"Could not write {ws.label()} -- was it deleted?"
            self.refresh()
            return
        result = outcome["result"]
        if not result.added:
            # Already in it, or refused (another dataset): said, and nothing opened.
            self.error = f"{fresh.label()}: {result.message()}"
            home.after_workspace_change(fresh)
            self.refresh()
            return
        self.busy = True
        home.after_workspace_change(fresh)
        if fresh.mount.dataset_path:
            from lookin.web.home_index import hash_dataset_later
            hash_dataset_later(fresh, fresh.mount.dataset_path, home.index.put)
        self.session.toast(f"{fresh.label()}: {result.message()}", "ok")
        self.close()
        session = self.session
        home.download(fresh, then=lambda: session.cmd(navigate=f"/?workspace={quote(fresh.id, safe='')}"))
