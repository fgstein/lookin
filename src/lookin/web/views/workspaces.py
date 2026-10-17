"""The Workspaces window -- every workspace, the snapshots saved inside
them, their details -- and opening a workspace into a tab.

Same behaviour as lookin.ui.workspace_panel:

* editing a name, tags or description keeps the id (an id written into a
  report has to keep pointing at what it pointed at); saving the *view*
  again always makes a new snapshot;
* "Save as workspace" stores what is loaded; "Update" rewrites the
  workspace open in this tab, keeping its id and snapshots;
* export writes a zip on the server, import reads one and asks where the
  crops are (a workspace never carries images);
* opening is always in a tab of its own (?workspace= / ?snapshot=), so
  nothing on screen here is ever replaced.
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

from lookin.core import library as lib
from lookin.ui import save_status
from lookin.web import html as h
from lookin.web.views.base import ConfirmDialog, Dialog, MessageDialog
from lookin.web.views.files import FileBrowserDialog

WORKSPACE_SORTS = (("updated", "Recently updated"), ("name", "Name A-Z"), ("created", "Recently created"))
SNAPSHOT_SORTS = (("updated", "Recently updated"), ("name", "Name A-Z"))
MAX_ROWS = 300


def _tags_of(text: str) -> list:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _matches(query: str, *fields) -> bool:
    query = (query or "").strip().lower()
    return not query or any(query in str(f or "").lower() for f in fields)


def workspace_link(workspace, snapshot=None) -> str:
    return f"/?snapshot={snapshot.id}" if snapshot is not None else f"/?workspace={workspace.id}"


def provenance() -> dict:
    """Text to read later, never used to fetch anything."""
    try:
        from lookin.core.home_config import HomeConfig
        from lookin.ui.mlflow_panel import _default_tracking_uri
        servers = HomeConfig(fallback_uri=_default_tracking_uri()).servers
        if servers:
            return {"mlflow_tracking_uri": str(servers[0]["uri"])}
    except Exception:  # noqa: BLE001
        pass
    return {}


class WorkspacesDialog(Dialog):
    width = 1180

    def __init__(self, session, select=None, focus_save: bool = False):
        super().__init__("Workspaces")
        self.s = session
        self.workspaces: list = []
        self.snapshots: list = []
        self.selected_ws = select or session.current_workspace
        self.selected_snap = None
        self.ws_filter = ""
        self.snap_filter = ""
        self.ws_sort = "updated"
        self.snap_sort = "updated"
        self.save_name = ""
        self.include_embeddings = False
        self.status = ("", "")
        self.focus_save = focus_save
        self.reload()
        self._suggest_name()

    def title_html(self) -> str:
        return f'{h.icon("save", 17, "c-acc")}<span>Workspaces</span>'

    # ---- data ----

    def reload(self) -> None:
        self.workspaces = lib.list_workspaces()
        if self.selected_ws is not None:
            self.selected_ws = next((w for w in self.workspaces if w.id == self.selected_ws.id), None)
        self._reload_snapshots()

    def _reload_snapshots(self) -> None:
        self.snapshots = lib.list_snapshots(self.selected_ws) if self.selected_ws else []
        if self.selected_snap is not None:
            self.selected_snap = next((s for s in self.snapshots if s.id == self.selected_snap.id), None)

    def _suggest_name(self) -> None:
        if self.save_name:
            return
        dataset = self.s.state.dataset
        if dataset is not None and dataset.csv_path:
            self.save_name = Path(dataset.csv_path).stem

    def _sorted_ws(self) -> list:
        favorites = set(lib.load_personal()["favorites"])
        items = [w for w in self.workspaces if _matches(self.ws_filter, w.name, w.description, " ".join(w.tags))]
        if self.ws_sort == "name":
            items.sort(key=lambda w: w.label().lower())
        elif self.ws_sort == "created":
            items.sort(key=lambda w: w.created_at, reverse=True)
        else:
            items.sort(key=lambda w: w.updated_at, reverse=True)
        items.sort(key=lambda w: w.id not in favorites)
        return items

    def _sorted_snaps(self) -> list:
        favorites = set(lib.load_personal()["favorites"])
        items = [s for s in self.snapshots if _matches(self.snap_filter, s.name, s.description, " ".join(s.tags))]
        if self.snap_sort == "name":
            items.sort(key=lambda s: s.label().lower())
        else:
            items.sort(key=lambda s: s.updated_at, reverse=True)
        items.sort(key=lambda s: s.id not in favorites)
        return items

    # ---- rendering ----

    def body(self) -> str:
        favorites = set(lib.load_personal()["favorites"])
        ws_rows = []
        if not self.workspaces:
            ws_rows.append(h.banner("Nothing saved yet. Load a dataset, then use 'Save as workspace' below.", "info"))
        for w in self._sorted_ws()[:MAX_ROWS]:
            on = self.selected_ws is not None and w.id == self.selected_ws.id
            star = h.icon("star", 13, "c-warn") if w.id in favorites else h.icon("save", 13)
            ws_rows.append(
                f'<button type="button" class="facet-b{" on" if on else ""}" style="min-height:40px;{"opacity:.5" if not w.exists() else ""}"'
                f'{self.a("select_ws", id=w.id)}>{star}<span style="display:flex;flex-direction:column;min-width:0">'
                f'<span class="ell">{h.e(w.label())}</span><span class="muted small ell">{h.e(w.mount.summary() or w.id)}</span></span></button>')
        snap_rows = []
        if self.selected_ws is None:
            snap_rows.append(h.banner("Select a workspace.", "info"))
        elif not self.snapshots:
            snap_rows.append(h.banner("No snapshots in this workspace yet.", "info"))
        for s in self._sorted_snaps()[:MAX_ROWS]:
            on = self.selected_snap is not None and s.id == self.selected_snap.id
            star = h.icon("star", 13, "c-warn") if s.id in favorites else h.icon("bookmark", 13)
            snap_rows.append(
                f'<button type="button" class="facet-b{" on" if on else ""}" style="min-height:40px"{self.a("select_snap", id=s.id)}>'
                f'{star}<span style="display:flex;flex-direction:column;min-width:0"><span class="ell">{h.e(s.label())}</span>'
                f'<span class="muted small mono">{h.e(s.id)}</span></span></button>')

        def column(title, filt, filt_action, sort_opts, sort_val, sort_action, rows):
            return (f'<div style="flex:1 1 0;min-width:220px;display:flex;flex-direction:column;border:1px solid var(--line-2);border-radius:8px;overflow:hidden">'
                    f'<div class="toolbar"><h2>{h.e(title)}</h2>'
                    f'{h.select(sort_opts, sort_val, change=self.c(sort_action), cls="sp")}</div>'
                    f'<div style="padding:6px 8px">{h.text_input(filt, input_=self.i(filt_action), placeholder="Filter", type_="search", width="100%")}</div>'
                    f'<div style="flex:1 1 auto;overflow-y:auto;height:340px">{"".join(rows)}</div></div>')

        cols = (column("Workspaces", self.ws_filter, "ws_filter", WORKSPACE_SORTS, self.ws_sort, "ws_sort", ws_rows)
                + column("Snapshots", self.snap_filter, "snap_filter", SNAPSHOT_SORTS, self.snap_sort, "snap_sort", snap_rows)
                + f'<div style="flex:1.3 1 0;min-width:280px;border:1px solid var(--line-2);border-radius:8px;padding:10px;overflow-y:auto;height:420px">{self._details()}</div>')
        current = self.s.current_workspace
        update_btn = ""
        if current is not None and current.exists():
            st = save_status.compute(self.s)
            update_btn = h.button(f"Update '{current.label()}'", act_attr=self.a("update_ws"),
                                  disabled=not st.has_data or st.workspace_saved, icon_name="refresh",
                                  title="Already matches what is loaded." if st.workspace_saved else
                                  "Write the dataset and experiments loaded now into this workspace. Its id and its snapshots are kept.")
        save_bar = (f'<div class="row" style="margin-top:12px;padding-top:12px;border-top:1px solid var(--line-2)">'
                    f'{h.text_input(self.save_name, change=self.c("save_name"), label="Name", width="260px", autofocus=self.focus_save)}'
                    f'{h.button("Save as workspace", act_attr=self.a("save_ws"), kind="primary", icon_name="save")}'
                    f'{update_btn}'
                    f'{h.button("Save this snapshot", act_attr=self.a("save_snap"), icon_name="bookmark", title="A new snapshot in the selected workspace -- never an overwrite")}'
                    f'</div><div class="row" style="margin-top:8px">'
                    f'{h.button("Import a workspace…", act_attr=self.a("import_ws"), icon_name="upload")}'
                    f'{h.checkbox("Include embeddings in exports (much larger)", self.include_embeddings, change=self.c("include_emb"), title="Needed only to re-project or search by similarity; the saved map itself travels without them.")}'
                    f'</div>')
        text, kind = self.status
        status = h.status_line(text, kind) if text else ""
        return f'<div class="row-nw" style="align-items:stretch;gap:10px">{cols}</div>{save_bar}{status}'

    def _details(self) -> str:
        if self.selected_snap is not None:
            return self._snap_details()
        if self.selected_ws is not None:
            return self._ws_details()
        return '<div class="subhead">Details</div>' + h.banner("Select a workspace or a snapshot.", "info")

    def _meta_form(self, obj, action) -> str:
        return (f'<form class="form-col" style="margin-top:10px"{self.submit(action)}>'
                f'{h.text_input(obj.name, name="name", label="Name")}'
                f'{h.text_input(", ".join(obj.tags), name="tags", label="Tags (comma separated)")}'
                f'<label class="fld"><span class="fld-l">Description</span>{h.textarea(obj.description, name="description", rows=3, mono=False)}</label>'
                f'<div>{h.button("Save details", kind="primary", small=True, type_="submit", icon_name="save")}</div></form>')

    def _ws_details(self) -> str:
        w = self.selected_ws
        m = w.mount
        rows = [("Id", h.mono(w.id)), ("Dataset", h.e(m.dataset_path or "(none)")), ("Split", h.e(m.split_kind or "(none)")),
                ("Experiments", h.e(", ".join(e.name for e in m.experiments) or "(none)")),
                ("Saved", h.e(w.created_at)), ("Snapshots", str(len(self.snapshots)))]
        if m.row_filter_col:
            rows.append(("Row filter", h.e(f"{m.row_filter_col} = {m.row_filter_value}")))
        if m.dataset_filter:
            from lookin.core.dataset_filter import describe
            lines = describe(m.dataset_filter)
            if lines:
                rows.append(("Filter", "<br>".join(h.e(line) for line in lines)))
        for k, v in (w.provenance or {}).items():
            rows.append((str(k), h.e(str(v))))
        warn = ""
        if not w.exists():
            warn = h.banner("This workspace's directory is gone.", "error")
        elif w.dataset_changed():
            warn = h.banner("The dataset CSV has changed since this workspace was saved -- rows may have been added or removed.", "warn")
        starred = lib.is_favorite(w.id)
        actions = (f'<div class="row" style="margin-top:10px">'
                   f'<a class="btn btn-primary{" disabled" if not w.exists() else ""}" href="{h.e(workspace_link(w))}" target="_blank" rel="noopener">{h.icon("external", 14)}<span>Open</span></a>'
                   f'{h.button("Unfavourite" if starred else "Favourite", act_attr=self.a("fav", id=w.id), icon_name="star", small=True)}'
                   f'{h.button("Export…", act_attr=self.a("export_ws"), icon_name="download", small=True, disabled=not w.exists())}'
                   f'{h.button("Delete", act_attr=self.a("delete_ws"), kind="danger", small=True, icon_name="trash")}</div>')
        return (f'<div class="subhead">Workspace details</div>{warn}{h.kv_grid(rows)}{actions}'
                f'{self._meta_form(w, "save_ws_details")}')

    def _snap_details(self) -> str:
        s = self.selected_snap
        rows = [("Id", h.mono(s.id)), ("Workspace", h.e(self.selected_ws.label() if self.selected_ws else "")),
                ("Saved", h.e(s.created_at))]
        if s.description:
            rows.append(("Description", h.e(s.description)))
        warn = "" if s.understood() else h.banner(
            "This snapshot was saved by a newer version of Lookin. The data will open, but the view it describes cannot be restored.", "warn")
        starred = lib.is_favorite(s.id)
        actions = (f'<div class="row" style="margin-top:10px">'
                   f'<a class="btn btn-primary" href="{h.e(workspace_link(self.selected_ws, s))}" target="_blank" rel="noopener">{h.icon("external", 14)}<span>Open this snapshot</span></a>'
                   f'{h.button("Unfavourite" if starred else "Favourite", act_attr=self.a("fav", id=s.id), icon_name="star", small=True)}'
                   f'{h.button("Copy id", act_attr=self.a("copy", text=s.id), icon_name="copy", small=True)}'
                   f'{h.button("Delete", act_attr=self.a("delete_snap"), kind="danger", small=True, icon_name="trash")}</div>')
        return (f'<div class="subhead">Snapshot details</div>{warn}{h.kv_grid(rows)}{actions}'
                f'{self._meta_form(s, "save_snap_details")}')

    def footer(self) -> str:
        return h.button("Close", act_attr=self.a("close"))

    # ---- actions ----

    def _set_status(self, text: str, kind: str = "ok") -> None:
        self.status = (text, kind)
        self.refresh()

    def on_select_ws(self, id: str = "", **_):  # noqa: A002
        self.selected_ws = next((w for w in self.workspaces if w.id == id), None)
        self.selected_snap = None
        self._reload_snapshots()
        self.refresh()

    def on_select_snap(self, id: str = "", **_):  # noqa: A002
        self.selected_snap = next((s for s in self.snapshots if s.id == id), None)
        self.refresh()

    def on_ws_filter(self, value: str = "", **_):
        self.ws_filter = value or ""
        self.refresh()

    def on_snap_filter(self, value: str = "", **_):
        self.snap_filter = value or ""
        self.refresh()

    def on_ws_sort(self, value: str = "updated", **_):
        self.ws_sort = value
        self.refresh()

    def on_snap_sort(self, value: str = "updated", **_):
        self.snap_sort = value
        self.refresh()

    def on_save_name(self, value: str = "", **_):
        self.save_name = value or ""

    def on_include_emb(self, value=False, **_):
        self.include_embeddings = bool(value)

    def on_fav(self, id: str = "", **_):  # noqa: A002
        lib.set_favorite(id, not lib.is_favorite(id))
        self.refresh()

    def on_copy(self, text: str = "", **_):
        self.s.cmd(copy=text)
        self.s.toast(f"Copied {text}.", "neutral")

    def on_save_ws_details(self, name: str = "", tags: str = "", description: str = "", **_):
        edited = self.selected_ws
        if edited is None:
            return

        def change(fresh):
            fresh.name, fresh.tags, fresh.description = name, _tags_of(tags), description
            return True

        if lib.update_workspace(edited.path, change) is not None:
            self.status = (f"Saved details for {name or edited.label()}.", "ok")
        else:
            self.status = ("Could not write the workspace.", "error")
        self.reload()
        self.s.mark("chrome")
        self.refresh()

    def on_save_snap_details(self, name: str = "", tags: str = "", description: str = "", **_):
        snap = self.selected_snap
        if snap is None or self.selected_ws is None:
            return
        snap.name, snap.tags, snap.description = name, _tags_of(tags), description
        if lib.save_snapshot(self.selected_ws, snap):
            self.status = (f"Saved details for {snap.label()}.", "ok")
        else:
            self.status = ("Could not write the snapshot.", "error")
        self._reload_snapshots()
        self.refresh()

    def on_save_ws(self, **_):
        mount = self.s.upload_view.current_mount()
        if mount is None:
            self._set_status("Load a dataset in the Sources tab first.", "warning")
            return
        name = (self.save_name or "").strip() or Path(mount.dataset_path).stem
        workspace = lib.create_workspace(name, mount, provenance=provenance())
        if workspace is None:
            self._set_status("Could not create the workspace.", "error")
            return
        self.save_name = ""
        self.selected_ws, self.selected_snap = workspace, None
        self.s.current_workspace = workspace
        self.s.title = f"{workspace.label()} — Lookin"
        self.reload()
        save_status.mark_baseline(self.s)
        self.s.mark("chrome")
        self.s.cmd(replace_url=workspace_link(workspace))
        self._set_status(f"Saved workspace {workspace.label()} ({workspace.id}).")

    def on_update_ws(self, **_):
        workspace = self.s.current_workspace
        mount = self.s.upload_view.current_mount()
        if workspace is None:
            return
        if mount is None:
            self._set_status("Load a dataset in the Sources tab first.", "warning")
            return

        def go():
            fresh = lib.update_workspace_mount(workspace.path, mount)
            if fresh is None:
                self._set_status("Could not write the workspace.", "error")
                return
            self.s.current_workspace = fresh
            self.selected_ws, self.selected_snap = fresh, None
            self.reload()
            save_status.mark_baseline(self.s)
            self.s.mark("chrome")
            self._set_status(f"Updated workspace {fresh.label()} ({fresh.id}).")

        self.s.open_dialog(ConfirmDialog(
            f"Update {workspace.label()}?",
            "The workspace will hold the dataset and experiments loaded now, in place of what it held. "
            "Its id and its snapshots are kept; MLflow runs not downloaded here stay in it.",
            go, yes_label="Update", danger=False))

    def on_save_snap(self, **_):
        if self.selected_ws is None:
            self._set_status("Select the workspace to save this snapshot into.", "warning")
            return
        snapshot = lib.create_snapshot(self.selected_ws, self.s.view_state.capture(), name=(self.save_name or "").strip())
        if snapshot is None:
            self._set_status("Could not save the snapshot.", "error")
            return
        self.save_name = ""
        self.selected_snap = snapshot
        self.s.current_snapshot = snapshot
        self._reload_snapshots()
        save_status.mark_baseline(self.s)
        self.s.mark("chrome")
        self._set_status(f"Saved snapshot {snapshot.id} -- share that id to reopen exactly this.")

    def on_export_ws(self, **_):
        workspace = self.selected_ws
        if workspace is None:
            return

        def picked(directory: str):
            dest = Path(directory) / f"{lib.slugify(workspace.label())}-{workspace.id}.zip"
            ok, message = lib.export_workspace(workspace, dest, include_embeddings=self.include_embeddings)
            if ok:
                message = f"{message} Wrote {dest}"
            self._set_status(message, "ok" if ok else "error")

        self.s.open_dialog(FileBrowserDialog(picked, "", dirs_only=True, title="Export this workspace to which folder?"))

    def on_import_ws(self, **_):
        def picked(path: str):
            workspace, notes = lib.import_workspace(path)
            if workspace is None:
                self._set_status(" ".join(notes), "error")
                return
            self.selected_ws, self.selected_snap = workspace, None
            self.reload()
            self.refresh()
            self.s.open_dialog(ImageRootDialog(self, workspace, notes))

        self.s.open_dialog(FileBrowserDialog(picked, "", {".zip"}, title="Import which workspace?"))

    def on_delete_ws(self, **_):
        workspace = self.selected_ws
        if workspace is None:
            return
        count = len(self.snapshots)
        note = f" and its {count} snapshot(s)" if count else ""

        def go():
            if lib.delete_workspace(workspace):
                self.selected_ws, self.selected_snap = None, None
                self.reload()
                self._set_status(f"Deleted {workspace.label()}.")
            else:
                self._set_status("Could not delete the workspace.", "error")

        self.s.open_dialog(ConfirmDialog(
            f"Delete {workspace.label()}{note}?",
            "The workspace directory is removed. The dataset and results CSVs it points at are not touched.", go))

    def on_delete_snap(self, **_):
        snap = self.selected_snap
        if snap is None:
            return

        def go():
            if lib.delete_snapshot(self.selected_ws, snap.id):
                self.selected_snap = None
                self._reload_snapshots()
                self._set_status("Snapshot deleted.")
            else:
                self._set_status("Could not delete the snapshot.", "error")

        self.s.open_dialog(ConfirmDialog(f"Delete the snapshot {snap.label()}?",
                                         f"Any report quoting {snap.id} will no longer resolve.", go))


class ImageRootDialog(Dialog):
    width = 560
    title = "Where are the crops on this machine?"

    def __init__(self, parent: WorkspacesDialog, workspace, notes):
        super().__init__()
        self.parent = parent
        self.workspace = workspace
        self.notes = list(notes or [])
        self.path = ""

    def body(self) -> str:
        notes = "".join(f'<div class="muted small">- {h.e(n)}</div>' for n in self.notes)
        return ('<p>A workspace carries the dataset and the results, never the images -- so this is the one path '
                'that has to be answered here.</p>'
                f'<div class="row-nw">{h.text_input(self.path, change=self.c("path"), mono=True, cls="grow", label="")}'
                f'{h.button(icon_name="folder", act_attr=self.a("browse"), title="Browse the server")}</div>{notes}')

    def footer(self) -> str:
        return h.button("Skip", act_attr=self.a("close")) + h.button("Save", act_attr=self.a("save"), kind="primary")

    def on_path(self, value: str = "", **_):
        self.path = (value or "").strip()

    def on_browse(self, **_):
        def picked(p):
            self.path = p
            self.refresh()
        self.session.open_dialog(FileBrowserDialog(picked, self.path, dirs_only=True, title="Where are the crops?"))

    def on_save(self, **_):
        lib.rebase_image_root(self.workspace, self.path)
        self.close()
        self.parent.reload()
        self.parent._set_status(f"Imported {self.workspace.label()} ({self.workspace.id}).")


# ---- opening a workspace into this tab -------------------------------------

def open_from_link(session, workspace, snapshot=None) -> None:
    """Load a workspace (or one of its snapshots) into this tab, as asked by
    its ?workspace= / ?snapshot= address. Experiments whose files are not
    downloaded yet are left out, and said so."""
    from lookin.core import workspace_sources as wss

    missing = [e.name for e in workspace.mount.experiments if not wss.is_downloaded(e)]
    mount = wss.downloaded_only(workspace.mount) if missing else None
    session.mode = "analysis"
    session.tab = "sources"
    session.title = f"{workspace.label()} — Lookin"
    session.current_workspace = workspace
    session.current_snapshot = None
    session.mark("page")

    def work():
        try:
            session.upload_view.restore_mount(mount if mount is not None else workspace.mount)
            _adopt_loaded_mount(session, workspace)
            _remember_loadable(session, workspace)
            lib.mark_opened(workspace.id)
            with session.lock:
                if snapshot is None:
                    save_status.mark_baseline(session)
                    if missing:
                        session.open_dialog(MessageDialog(
                            "Opened without some experiments",
                            ["Not downloaded yet, so not loaded:"] + [f"- {n}" for n in missing]
                            + ["Download them from the workspace's row on the Home page, then open it again."],
                            kind="warn"))
                    session.push({"dirty": ["page", "dialogs"]})
                    return
                lib.mark_opened(snapshot.id)
                if not snapshot.understood():
                    save_status.mark_baseline(session)
                    _report_loaded(session, workspace, snapshot, [
                        "This snapshot was saved by a newer version of Lookin -- only the data was opened."], False)
                    session.push({"dirty": ["page", "dialogs"]})
                    return
                warnings = session.view_state.restore(snapshot.view_state)
                session.current_snapshot = snapshot
                save_status.mark_baseline(session)
                _report_loaded(session, workspace, snapshot, warnings, True)
            session.push({"dirty": ["page", "dialogs"]})
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            session.push({"dirty": ["page"]})

    threading.Thread(target=work, daemon=True).start()


def _report_loaded(session, workspace, snapshot, warnings: list, restored: bool) -> None:
    ok = restored and not warnings
    lines = [f'<p><b>Snapshot {h.e(snapshot.id)}{(" -- " + h.e(snapshot.name)) if snapshot.name else ""}</b></p>',
             f"<p>From workspace {h.e(workspace.label())}.</p>"]
    if warnings:
        lines.append("<p><b>Could not be applied:</b></p>")
        lines += [f'<p class="c-warn">- {h.e(w)}</p>' for w in warnings]
    if restored:
        lines.append('<p class="muted small">The tab and its settings are back. Anything still being rebuilt shows in '
                     'the top bar until it finishes.</p>')
    session.open_dialog(MessageDialog("Snapshot loaded" if ok else "Snapshot loaded, with gaps", lines,
                                      kind="ok" if ok else "warn"))


def _adopt_loaded_mount(session, workspace) -> None:
    """Record what the load filled in that the workspace left blank -- only
    when the load brought back exactly the workspace's dataset and
    experiments (see WorkspacePanel._adopt_loaded_mount)."""
    loaded = session.upload_view.current_mount()
    if loaded is None:
        return

    def change(fresh):
        same_contents = (loaded.dataset_path == fresh.mount.dataset_path
                         and {(e.name, e.path) for e in loaded.experiments}
                         == {(e.name, e.path) for e in fresh.mount.experiments})
        if not same_contents or save_status.same_mount(loaded, fresh.mount):
            return False
        fresh.mount = loaded
        return True

    fresh = lib.update_workspace(workspace.path, change)
    if fresh is not None:
        workspace.mount = fresh.mount


def adopt_dataset(session) -> None:
    """After every dataset load: a workspace opened with no usable dataset
    takes the one just loaded (see WorkspacePanel.adopt_dataset)."""
    workspace = session.current_workspace
    loaded = session.upload_view.current_mount()
    if workspace is None or loaded is None or workspace.mount.exists():
        return
    digest = lib.hash_file(loaded.dataset_path)

    def change(fresh):
        if fresh.mount.exists():
            return False
        for name in ("dataset_path", "uuid_col", "crop_col", "label_col", "image_root",
                     "row_filter_col", "row_filter_value", "split_kind", "review_version", "dataset_filter"):
            setattr(fresh.mount, name, getattr(loaded, name))
        fresh.dataset_hash = digest
        return True

    fresh = lib.update_workspace(workspace.path, change)
    if fresh is None or fresh.mount.dataset_path != loaded.dataset_path:
        return
    workspace.mount, workspace.dataset_hash = fresh.mount, fresh.dataset_hash
    _adopt_loaded_mount(session, workspace)
    save_status.mark_baseline(session)
    session.title = f"{workspace.label()} — Lookin"
    session.mark("chrome")


# ---- following the open workspace's manifest -------------------------------

# How often a tab looks at the manifest of the workspace it has open, to
# follow a change made from Home or from another tab (see follow_workspace).
WATCH_S = 3.0


def _content(workspace) -> dict:
    """A workspace as saved, less the stamp every write moves."""
    payload = lib.workspace_to_dict(workspace)
    payload.pop("updated_at", None)
    return payload


def _remember_loadable(session, workspace) -> None:
    """What of `workspace` this tab could have loaded, as (name, path):
    anything outside this set that turns up loadable later is new."""
    from lookin.core import workspace_sources as wss
    session._loadable_seen = {(e.name, e.path) for e in workspace.mount.experiments if wss.is_downloaded(e)}


def follow_workspace(session, fresh) -> bool:
    """The open workspace's manifest changed outside this tab's Sources tab
    -- on its Home card, in this tab or another one. Bring what is loaded
    along, as far as that is safe (see WorkspacePanel.follow_workspace):

    - nothing loaded and the workspace now names a dataset it did not
      before: load the whole workspace;
    - the same dataset loaded: load the experiments that have become
      loadable since this tab last looked (added from Home, or just
      downloaded) -- not one the user removed here and has not saved;
    - anything else is left alone, and the app bar's Save says they differ.

    Returns False when it could not look yet (a load is running)."""
    from lookin.core import workspace_sources as wss

    upload = session.upload_view
    previous = session.current_workspace
    if previous is None or previous.id != fresh.id:
        return True
    lock = session.__dict__.setdefault("_follow_lock", threading.Lock())
    if upload.is_loading() or not lock.acquire(blocking=False):
        return False
    try:
        loaded = upload.current_mount()
        session.current_workspace = fresh
        seen = getattr(session, "_loadable_seen", set())
        if loaded is None:
            if fresh.mount.dataset_path and fresh.mount.dataset_path != previous.mount.dataset_path:
                upload.restore_mount(wss.downloaded_only(fresh.mount))
                _adopt_loaded_mount(session, fresh)
                save_status.mark_baseline(session)
        elif loaded.dataset_path == fresh.mount.dataset_path:
            here = {e.name for e in loaded.experiments}
            new = [e for e in fresh.mount.experiments
                   if e.name not in here and (e.name, e.path) not in seen and wss.is_downloaded(e)]
            if new:
                upload.add_mount_experiments(new)
                _adopt_loaded_mount(session, fresh)
        _remember_loadable(session, fresh)
        session.mark("*")
        return True
    finally:
        lock.release()


def check_current_workspace(session) -> None:
    workspace = session.current_workspace
    if workspace is None or workspace.path is None:
        return
    try:
        stat = (Path(workspace.path) / lib.MANIFEST_NAME).stat()
    except OSError:
        return
    # The inode too: every write is a replace, so it moves on each one even
    # when two land within one mtime tick.
    seen = (workspace.id, stat.st_ino, stat.st_mtime_ns)
    if getattr(session, "_seen_manifest", None) == seen:
        return
    fresh = lib.load_workspace(workspace.path)
    if fresh is None:
        return
    if _content(fresh) == _content(workspace) or follow_workspace(session, fresh):
        session._seen_manifest = seen


def start_watching(session) -> None:
    """Follow the open workspace's manifest for the life of the session.
    A thread per tab rather than a signal between tabs: the only thing two
    tabs share is the library on disk, and one stat every few seconds is
    the whole cost."""
    def watch():
        while not session.closed:
            time.sleep(WATCH_S)
            try:
                check_current_workspace(session)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    threading.Thread(target=watch, daemon=True, name=f"lookin-watch-{session.sid[:6]}").start()
