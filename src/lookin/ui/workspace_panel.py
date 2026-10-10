"""The Workspaces window: every workspace, and the views saved inside them.

The Home page (ui/home_view.py) is where workspaces are created and filled
from MLflow; this window is where one is looked at in detail -- metadata,
snapshots, export and import -- and where the session on screen is saved.

Two levels, browsed side by side -- workspaces, then the snapshots of the
selected one -- with a detail panel on the right. The same shape as the
MLflow dialog it borrows its machinery from (see ui/browser_panel.py), one
level shallower.

    workspace    "the eval of checkpoint 42 on dataset B" -- what you organise,
              and what you delete to reclaim disk
    snapshot  "the confusion on class 3 that looked wrong" -- a bookmark in
              that workspace, and what an id points at

What this deliberately does not do
----------------------------------
No worker threads, no polling, no connection state. The MLflow panel needs
all three because a tracking server is remote and may be slow or absent;
this one lists a local directory and answers in milliseconds. Copying that
machinery would be inventing a failure mode to defend against.

It also never recomputes on open. Restoring loads the mount and puts the
controls back; every expensive rebuild stays behind the banner the user
clicks, exactly as after any other data change (see ui/view_state.py).

The copy-on-write rule
----------------------
Editing a name, a tag or a description keeps the id -- an id written into a
report months ago has to keep pointing at what it pointed at. Saving the
*view* again makes a new snapshot instead. Both halves live in
core/library.py; this panel only offers them as two different buttons.
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

import flet as ft

from lookin.core import library as lib
from lookin.ui import browser_panel as bp
from lookin.ui import save_status
from lookin.ui.browser_panel import (
    COLUMN_SPACING, MIN_COLUMNS_HEIGHT, MIN_DIALOG_HEIGHT, MIN_DIALOG_WIDTH,
    TEXT_HEADING, TEXT_SMALL, TEXT_SUB, TEXT_TITLE,
)
from lookin.ui.components import ACCENT, info_banner, saved_path_message, status_row
from lookin.ui.server_browser import ServerFileBrowser

# How often a tab looks at the manifest of the workspace it has open, to
# follow a change made from Home or from another tab (see follow_workspace).
WATCH_S = 3.0

COLUMN_FRACTIONS = (0.30, 0.30)
COLUMN_MIN_WIDTHS = (240, 240)
FIXED_ROWS_HEIGHT = 210

# Sort options. The first element of each pair is a *key* compared in
# _sorted_workspaces / _sorted_snapshots -- only the second is display text
# (see the dropdown note in CLAUDE.md).
WORKSPACE_SORTS = (
    ("updated", "Recently updated"),
    ("name", "Name A-Z"),
    ("created", "Recently created"),
)
SNAPSHOT_SORTS = (
    ("updated", "Recently updated"),
    ("name", "Name A-Z"),
)


def _content(workspace) -> dict:
    """A workspace as saved, less the stamp every write moves."""
    payload = lib.workspace_to_dict(workspace)
    payload.pop("updated_at", None)
    return payload


def _tags_of(text: str) -> list:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


class WorkspacePanel:
    def __init__(self, ctx):
        self.ctx = ctx
        self.workspaces: list = []
        self.snapshots: list = []
        self.selected_workspace = None
        self.selected_snapshot = None
        self._workspace_rows: dict = {}
        self._snapshot_rows: dict = {}
        # Following the open workspace (see follow_workspace).
        self._loadable_seen: set = set()
        self._seen_manifest = None
        self._follow_lock = threading.Lock()
        self._stop = threading.Event()

        # ---- lists ----
        self.workspace_filter = ft.TextField(
            hint_text="Filter by name, tag or description", dense=True, border_radius=8,
            text_size=TEXT_SUB, on_change=lambda e: self._render_workspaces(),
        )
        self.snapshot_filter = ft.TextField(
            hint_text="Filter snapshots", dense=True, border_radius=8,
            text_size=TEXT_SUB, on_change=lambda e: self._render_snapshots(),
        )
        self.workspace_sort = bp.sort_dropdown(WORKSPACE_SORTS, lambda e: self._render_workspaces())
        self.snapshot_sort = bp.sort_dropdown(SNAPSHOT_SORTS, lambda e: self._render_snapshots())
        self.workspaces_list = ft.ListView(spacing=1, expand=True)
        self.snapshots_list = ft.ListView(spacing=1, expand=True)

        # ---- details ----
        self.details_title = ft.Text("Details", weight=ft.FontWeight.BOLD, size=TEXT_HEADING)
        self.details_column = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO, expand=True)

        # ---- save ----
        self.save_name_field = ft.TextField(
            label="Name", expand=True, dense=True, border_radius=8, text_size=TEXT_SUB)
        self.save_workspace_btn = ft.FilledButton(
            "Save as workspace", icon=ft.Icons.INVENTORY_2_OUTLINED,
            on_click=self._on_save_workspace, style=ft.ButtonStyle(bgcolor=ACCENT))
        self.save_view_btn = ft.OutlinedButton(
            "Save this snapshot", icon=ft.Icons.BOOKMARK_ADD_OUTLINED, on_click=self._on_save_view)
        # Writes what is loaded into the workspace that is open, in place --
        # shown only while one is (see _sync_update_button).
        self.update_workspace_btn = ft.OutlinedButton(
            "Update workspace", icon=ft.Icons.SYNC, on_click=self._on_update_workspace, visible=False)
        self.panel_status = ft.Container()
        # Export writes to, and import reads from, the *server's* filesystem
        # -- the browser's own file input never hands back a real path (see
        # ui/server_browser.py).
        self.file_browser = ServerFileBrowser(ctx)
        self.include_embeddings = ft.Checkbox(
            label="Include embeddings (much larger)", value=False,
            tooltip="Needed only to re-project or search by similarity; the saved map itself travels without them.")
        self.import_btn = ft.OutlinedButton(
            "Import a workspace...", icon=ft.Icons.UNARCHIVE_OUTLINED, on_click=self._on_import)

        self._column_panels = [
            bp.list_panel(1, "Workspaces", [self.workspace_filter], self.workspaces_list,
                          trailing=self.workspace_sort),
            bp.list_panel(2, "Snapshots", [self.snapshot_filter], self.snapshots_list,
                          trailing=self.snapshot_sort),
        ]
        self.panels_box = ft.Container(
            height=MIN_COLUMNS_HEIGHT,
            content=ft.Row(
                [*self._column_panels, bp.details_panel(self.details_title, self.details_column)],
                spacing=COLUMN_SPACING, expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        )
        self.content_container = ft.Container(
            width=MIN_DIALOG_WIDTH, height=MIN_DIALOG_HEIGHT,
            content=ft.Column([
                self.panels_box,
                ft.Divider(height=1),
                ft.Row([self.save_name_field, self.save_workspace_btn, self.update_workspace_btn,
                        self.save_view_btn], spacing=10),
                ft.Row([self.import_btn, self.include_embeddings], spacing=10,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.panel_status,
            ], spacing=12, scroll=ft.ScrollMode.AUTO),
        )
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.INVENTORY_2_OUTLINED, color=ACCENT, size=19),
                ft.Text("Workspaces", weight=ft.FontWeight.BOLD, size=TEXT_TITLE),
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            content=self.content_container,
            actions=[ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.dialog))],
        )
        self._render_workspaces()
        self._render_snapshots()
        self._render_details()

    # ---- opening ----------------------------------------------------

    def open_panel(self, e=None, select=None):
        """`select` opens the window on that workspace -- what the Home
        page's "Details" does."""
        if select is not None:
            self.selected_workspace, self.selected_snapshot = select, None
        self.reload()
        bp.fit_dialog_to_page(
            self.ctx.page, self.content_container, self.panels_box,
            self._column_panels, COLUMN_FRACTIONS, COLUMN_MIN_WIDTHS, FIXED_ROWS_HEIGHT,
        )
        self._suggest_save_name()
        self.ctx.page.open(self.dialog)

    def reload(self) -> None:
        """Re-read the library from disk. Cheap -- a few hundred small JSON
        files -- and done on every open so a workspace another session saved
        (or another tab, which is a separate session with its own state) is
        there without a restart."""
        self.workspaces = lib.list_workspaces()
        if self.selected_workspace is not None:
            self.selected_workspace = next(
                (b for b in self.workspaces if b.id == self.selected_workspace.id), None)
        self._reload_snapshots()
        self._render_workspaces()
        self._render_snapshots()
        self._render_details()
        self._sync_update_button()

    def _sync_update_button(self) -> None:
        """Offered for the workspace open in this tab, and enabled only when
        what is loaded differs from what it holds -- the same test the app
        bar's workspace chip makes."""
        workspace = getattr(self.ctx, "current_workspace", None)
        btn = self.update_workspace_btn
        btn.visible = workspace is not None and workspace.exists()
        if btn.visible:
            btn.text = f"Update '{workspace.label()}'"
            status = save_status.compute(self.ctx)
            btn.disabled = not status.has_data or status.workspace_saved
            btn.tooltip = ("Already matches what is loaded." if status.workspace_saved else
                           "Write the dataset and experiments loaded now into this workspace. "
                           "Its id and its snapshots are kept.")
        bp.safe_update(btn)

    def _reload_snapshots(self) -> None:
        self.snapshots = lib.list_snapshots(self.selected_workspace) if self.selected_workspace else []
        if self.selected_snapshot is not None:
            self.selected_snapshot = next(
                (s for s in self.snapshots if s.id == self.selected_snapshot.id), None)

    def _suggest_save_name(self) -> None:
        if self.save_name_field.value:
            return
        dataset = self.ctx.state.dataset
        if dataset is not None and dataset.csv_path:
            self.save_name_field.value = Path(dataset.csv_path).stem

    def _status(self, message: str, kind: str = "ok") -> None:
        self.panel_status.content = status_row(message, kind)
        bp.safe_update(self.panel_status)

    # ---- listing ----------------------------------------------------

    @staticmethod
    def _matches(query: str, *fields) -> bool:
        query = (query or "").strip().lower()
        if not query:
            return True
        return any(query in str(f or "").lower() for f in fields)

    def _sorted_workspaces(self) -> list:
        key = self.workspace_sort.value
        favorites = set(lib.load_personal()["favorites"])
        items = [b for b in self.workspaces
                 if self._matches(self.workspace_filter.value, b.name, b.description, " ".join(b.tags))]
        if key == "name":
            items.sort(key=lambda b: b.label().lower())
        elif key == "created":
            items.sort(key=lambda b: b.created_at, reverse=True)
        else:
            items.sort(key=lambda b: b.updated_at, reverse=True)
        # Favourites first, each group keeping the chosen order.
        items.sort(key=lambda b: b.id not in favorites)
        return items

    def _sorted_snapshots(self) -> list:
        items = [s for s in self.snapshots
                 if self._matches(self.snapshot_filter.value, s.name, s.description, " ".join(s.tags))]
        if self.snapshot_sort.value == "name":
            items.sort(key=lambda s: s.label().lower())
        else:
            items.sort(key=lambda s: s.updated_at, reverse=True)
        favorites = set(lib.load_personal()["favorites"])
        items.sort(key=lambda s: s.id not in favorites)
        return items

    def _render_workspaces(self) -> None:
        self._workspace_rows.clear()
        self.workspaces_list.controls.clear()
        if not self.workspaces:
            self.workspaces_list.controls.append(info_banner(
                "Nothing saved yet. Load a dataset, then use 'Save as workspace' below."))
            bp.safe_update(self.workspaces_list)
            return
        items = self._sorted_workspaces()
        favorites = set(lib.load_personal()["favorites"])
        selected_id = self.selected_workspace.id if self.selected_workspace else None
        for workspace in bp.capped(items, selected_id, lambda b: b.id):
            # A workspace whose directory has gone is shown greyed rather than
            # hidden: knowing it existed, and where it pointed, is more use
            # than a silently shorter list (same rule as RecentsPanel).
            missing = not workspace.exists()
            entry = bp.row(
                workspace.label(),
                workspace.summary() or workspace.id,
                selected=workspace.id == selected_id,
                on_click=lambda e, b=workspace: self._on_select_workspace(b),
                icon=ft.Icons.STAR if workspace.id in favorites else ft.Icons.INVENTORY_2_OUTLINED,
                icon_color=ft.Colors.AMBER if workspace.id in favorites else (
                    ft.Colors.ON_SURFACE_VARIANT if missing else None),
                key=workspace.id,
            )
            if missing:
                entry.container.opacity = 0.5
            self._workspace_rows[workspace.id] = entry
            self.workspaces_list.controls.append(entry.container)
        if len(items) > len(self._workspace_rows):
            self.workspaces_list.controls.append(bp.overflow_note(len(self._workspace_rows), len(items)))
        bp.safe_update(self.workspaces_list)

    def _render_snapshots(self) -> None:
        self._snapshot_rows.clear()
        self.snapshots_list.controls.clear()
        if self.selected_workspace is None:
            self.snapshots_list.controls.append(info_banner("Select a workspace."))
            bp.safe_update(self.snapshots_list)
            return
        if not self.snapshots:
            self.snapshots_list.controls.append(info_banner(
                "No snapshots in this workspace yet."))
            bp.safe_update(self.snapshots_list)
            return
        favorites = set(lib.load_personal()["favorites"])
        selected_id = self.selected_snapshot.id if self.selected_snapshot else None
        for snapshot in bp.capped(self._sorted_snapshots(), selected_id, lambda s: s.id):
            entry = bp.row(
                snapshot.label(), snapshot.id,
                selected=snapshot.id == selected_id,
                on_click=lambda e, s=snapshot: self._on_select_snapshot(s),
                icon=ft.Icons.STAR if snapshot.id in favorites else ft.Icons.BOOKMARK_OUTLINE,
                icon_color=ft.Colors.AMBER if snapshot.id in favorites else None,
                key=snapshot.id,
            )
            self._snapshot_rows[snapshot.id] = entry
            self.snapshots_list.controls.append(entry.container)
        bp.safe_update(self.snapshots_list)

    def _on_select_workspace(self, workspace) -> None:
        self.selected_workspace = workspace
        self.selected_snapshot = None
        self._reload_snapshots()
        self._render_workspaces()
        self._render_snapshots()
        self._render_details()

    def _on_select_snapshot(self, snapshot) -> None:
        previous = self.selected_snapshot.id if self.selected_snapshot else None
        self.selected_snapshot = snapshot
        if not bp.select_in_list(self._snapshot_rows, previous, snapshot.id):
            self._render_snapshots()
        self._render_details()

    # ---- details and actions ---------------------------------------

    def _render_details(self) -> None:
        self.details_column.controls.clear()
        if self.selected_snapshot is not None:
            self.details_title.value = "Snapshot details"
            self._render_snapshot_details()
        elif self.selected_workspace is not None:
            self.details_title.value = "Workspace details"
            self._render_workspace_details()
        else:
            self.details_title.value = "Details"
            self.details_column.controls.append(info_banner("Select a workspace or a snapshot."))
        bp.safe_update(self.details_title, self.details_column)

    def _kv(self, key: str, value: str) -> ft.Control:
        return ft.Row([
            ft.Text(f"{key}:", size=TEXT_SUB, color=ft.Colors.ON_SURFACE_VARIANT, width=104),
            ft.Text(value, size=TEXT_SUB, selectable=True, expand=True),
        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.START)

    def _metadata_editor(self, obj, on_save) -> ft.Control:
        """Name, tags and description, with one Save button.

        Saving here keeps the id -- that is the whole point of separating a
        metadata edit from a state change (see the module docstring).
        """
        name = ft.TextField(label="Name", value=obj.name, dense=True, border_radius=8, text_size=TEXT_SUB)
        tags = ft.TextField(label="Tags (comma separated)", value=", ".join(obj.tags),
                            dense=True, border_radius=8, text_size=TEXT_SUB)
        description = ft.TextField(label="Description", value=obj.description, multiline=True,
                                   min_lines=2, max_lines=4, dense=True, border_radius=8, text_size=TEXT_SUB)

        def save(e):
            obj.name, obj.tags, obj.description = name.value or "", _tags_of(tags.value), description.value or ""
            on_save()

        return ft.Column([
            name, tags, description,
            ft.Row([ft.FilledButton("Save details", icon=ft.Icons.SAVE_OUTLINED, on_click=save,
                                    style=ft.ButtonStyle(bgcolor=ACCENT))]),
        ], spacing=8)

    def _favorite_button(self, object_id: str) -> ft.Control:
        starred = lib.is_favorite(object_id)

        def toggle(e):
            lib.set_favorite(object_id, not starred)
            self._render_workspaces()
            self._render_snapshots()
            self._render_details()

        return ft.TextButton(
            "Unfavourite" if starred else "Favourite",
            icon=ft.Icons.STAR if starred else ft.Icons.STAR_BORDER, on_click=toggle)

    def _render_workspace_details(self) -> None:
        workspace = self.selected_workspace
        mount = workspace.mount
        rows = [
            self._kv("Id", workspace.id),
            self._kv("Dataset", mount.dataset_path or "(none)"),
            self._kv("Split", mount.split_kind or "(none)"),
            self._kv("Experiments", ", ".join(e.name for e in mount.experiments) or "(none)"),
            self._kv("Saved", workspace.created_at),
            self._kv("Snapshots", str(len(self.snapshots))),
        ]
        if mount.row_filter_col:
            rows.append(self._kv("Row filter", f"{mount.row_filter_col} = {mount.row_filter_value}"))
        for key, value in (workspace.provenance or {}).items():
            rows.append(self._kv(str(key), str(value)))

        if not workspace.exists():
            rows.insert(0, info_banner("This workspace's directory is gone.", icon=ft.Icons.ERROR_OUTLINE))
        elif workspace.dataset_changed():
            # A dataset that differs from the one this was saved against
            # renders plausible, wrong numbers with no other symptom.
            rows.insert(0, info_banner(
                "The dataset CSV has changed since this workspace was saved -- rows may have been "
                "added or removed.", icon=ft.Icons.WARNING_AMBER_OUTLINED))

        self.details_column.controls.extend(rows)
        self.details_column.controls.append(ft.Divider(height=1))
        self.details_column.controls.append(ft.Row([
            ft.FilledButton("Open", icon=ft.Icons.OPEN_IN_NEW,
                            on_click=self._on_open_workspace, disabled=not workspace.exists(),
                            style=ft.ButtonStyle(bgcolor=ACCENT)),
            self._favorite_button(workspace.id),
            ft.TextButton("Export...", icon=ft.Icons.ARCHIVE_OUTLINED,
                          on_click=self._on_export, disabled=not workspace.exists()),
            ft.TextButton("Delete", icon=ft.Icons.DELETE_OUTLINE,
                          on_click=self._on_delete_workspace, style=ft.ButtonStyle(color=ft.Colors.RED)),
        ], spacing=6, wrap=True))
        self.details_column.controls.append(ft.Divider(height=1))
        self.details_column.controls.append(
            self._metadata_editor(workspace, self._save_workspace_details))

    def _render_snapshot_details(self) -> None:
        snapshot = self.selected_snapshot
        rows = [
            self._kv("Id", snapshot.id),
            self._kv("Workspace", self.selected_workspace.label() if self.selected_workspace else ""),
            self._kv("Saved", snapshot.created_at),
        ]
        if snapshot.description:
            rows.append(self._kv("Description", snapshot.description))
        if not snapshot.understood():
            rows.insert(0, info_banner(
                "This snapshot was saved by a newer version of Lookin. The data will open, but the "
                "view it describes cannot be restored.", icon=ft.Icons.WARNING_AMBER_OUTLINED))
        self.details_column.controls.extend(rows)
        self.details_column.controls.append(ft.Divider(height=1))
        self.details_column.controls.append(ft.Row([
            ft.FilledButton("Open this snapshot", icon=ft.Icons.OPEN_IN_NEW,
                            on_click=self._on_open_snapshot, style=ft.ButtonStyle(bgcolor=ACCENT)),
            self._favorite_button(snapshot.id),
            ft.TextButton("Delete", icon=ft.Icons.DELETE_OUTLINE,
                          on_click=self._on_delete_snapshot, style=ft.ButtonStyle(color=ft.Colors.RED)),
        ], spacing=6, wrap=True))
        self.details_column.controls.append(ft.Divider(height=1))
        self.details_column.controls.append(
            self._metadata_editor(snapshot, self._save_snapshot_details))

    def _save_workspace_details(self) -> None:
        # Only the three fields the editor shows are written, onto the
        # manifest as it is now: the rest of self.selected_workspace was read
        # when this window opened, and another tab may have changed it since.
        edited = self.selected_workspace

        def change(fresh):
            fresh.name, fresh.tags, fresh.description = edited.name, list(edited.tags), edited.description
            return True

        if lib.update_workspace(edited.path, change) is not None:
            self._status(f"Saved details for {edited.label()}.")
        else:
            self._status("Could not write the workspace.", "error")
        self.reload()

    def _save_snapshot_details(self) -> None:
        if lib.save_snapshot(self.selected_workspace, self.selected_snapshot):
            self._status(f"Saved details for {self.selected_snapshot.label()}.")
        else:
            self._status("Could not write the snapshot.", "error")
        self._reload_snapshots()
        self._render_snapshots()
        self._render_details()

    # ---- saving ------------------------------------------------------

    def _on_save_workspace(self, e) -> None:
        mount = self.ctx.upload_view.current_mount() if self.ctx.upload_view else None
        if mount is None:
            self._status("Load a dataset in the Data tab first.", "warning")
            return
        name = (self.save_name_field.value or "").strip() or Path(mount.dataset_path).stem
        workspace = lib.create_workspace(name, mount, provenance=self._provenance())
        if workspace is None:
            self._status("Could not create the workspace.", "error")
            return
        self.save_name_field.value = ""
        self.selected_workspace, self.selected_snapshot = workspace, None
        self.ctx.current_workspace = workspace
        self.reload()
        self._refresh_save_status(baseline=True)
        self._status(f"Saved workspace {workspace.label()} ({workspace.id}).")

    def _on_update_workspace(self, e) -> None:
        workspace = getattr(self.ctx, "current_workspace", None)
        mount = self.ctx.upload_view.current_mount() if self.ctx.upload_view else None
        if workspace is None:
            return
        if mount is None:
            self._status("Load a dataset in the Data tab first.", "warning")
            return
        self._confirm(
            f"Update {workspace.label()}?",
            "The workspace will hold the dataset and experiments loaded now, in place of what it held. "
            "Its id and its snapshots are kept; MLflow runs not downloaded here stay in it.",
            lambda: self._update_workspace(workspace, mount), yes_label="Update", danger=False)

    def _update_workspace(self, workspace, mount) -> None:
        fresh = lib.update_workspace_mount(workspace.path, mount)
        if fresh is None:
            self._status("Could not write the workspace.", "error")
            return
        self.ctx.current_workspace = fresh
        self.selected_workspace, self.selected_snapshot = fresh, None
        self.reload()
        self._refresh_save_status(baseline=True)
        self._status(f"Updated workspace {fresh.label()} ({fresh.id}).")

    def _on_save_view(self, e) -> None:
        """Save the current state as a snapshot in the selected workspace.

        Always a new snapshot, never an overwrite: a snapshot is what an id in
        a report points at, so it can never change under that id.
        """
        if self.selected_workspace is None:
            self._status("Select the workspace to save this snapshot into.", "warning")
            return
        name = (self.save_name_field.value or "").strip()
        snapshot = lib.create_snapshot(self.selected_workspace, self.ctx.view_state.capture(), name=name)
        if snapshot is None:
            self._status("Could not save the snapshot.", "error")
            return
        self.save_name_field.value = ""
        self.selected_snapshot = snapshot
        self.ctx.current_snapshot = snapshot
        self._reload_snapshots()
        self._render_snapshots()
        self._render_details()
        self._refresh_save_status(baseline=True)
        self._status(f"Saved snapshot {snapshot.id} -- share that id to reopen exactly this.")

    def _refresh_save_status(self, baseline: bool = False) -> None:
        """Repaint the app-bar chips. `baseline` also resets "changed since"
        to now, which every save and every load does."""
        if baseline:
            save_status.mark_baseline(self.ctx)
        if getattr(self.ctx, "save_status", None) is not None:
            self.ctx.save_status.refresh()

    def _provenance(self) -> dict:
        """Text to read later, never used to fetch anything: whoever opens
        this workspace may be on a different MLflow, or none at all."""
        provenance = {}
        uri = getattr(getattr(self.ctx, "mlflow_panel", None), "uri_field", None)
        if uri is not None and uri.value:
            provenance["mlflow_tracking_uri"] = str(uri.value)
        return provenance

    # ---- opening -----------------------------------------------------

    def _on_open_workspace(self, e) -> None:
        self.open_in_new_tab(self.selected_workspace)

    def _on_open_snapshot(self, e) -> None:
        self.open_in_new_tab(self.selected_workspace, self.selected_snapshot)

    def open_in_new_tab(self, workspace, snapshot=None) -> None:
        """Open a workspace -- or one of its snapshots -- in a browser tab of
        its own. How every Open button in the app opens one.

        A tab per workspace is what lets several be open at once and be
        switched between without reloading anything: every browser tab is
        an independent session with its own data, filters and rebuilt
        charts. It is also why opening never asks about unsaved work any
        more: nothing on screen here is replaced.

        The link is relative, like the image export's (see
        components.open_png_in_tab): it resolves against the page's base
        address, which is what keeps it right behind a forwarded port or a
        proxy path. `_blank` is spelled out for the same reason as there --
        navigating this tab away would lose what it has loaded.
        """
        if workspace is None:
            return
        link = f"?snapshot={snapshot.id}" if snapshot is not None else f"?workspace={workspace.id}"
        if self.dialog.page is not None:
            self.ctx.page.close(self.dialog)
        self.ctx.page.launch_url(link, web_window_name="_blank")

    def open_from_link(self, workspace, snapshot=None) -> None:
        """Load a workspace into this tab, as asked by its ?workspace= or
        ?snapshot= link (see app._open_from_url).

        Experiments whose files are not downloaded yet are left out and
        said so: the Home page offers the download before it opens a tab,
        so this only happens for a link that was bookmarked or shared.
        """
        from lookin.core import workspace_sources as wss
        missing = [e.name for e in workspace.mount.experiments if not wss.is_downloaded(e)]
        mount = wss.downloaded_only(workspace.mount) if missing else None
        self.selected_workspace, self.selected_snapshot = workspace, snapshot
        self._open(workspace, snapshot, mount)
        if missing and snapshot is None:
            dialog = ft.AlertDialog(
                modal=True,
                title=ft.Row([ft.Icon(ft.Icons.WARNING_AMBER_OUTLINED, color=ft.Colors.AMBER),
                              ft.Text("Opened without some experiments", weight=ft.FontWeight.BOLD)],
                             spacing=10),
                content=ft.Container(width=560, content=ft.Column(
                    [ft.Text("Not downloaded yet, so not loaded:", size=TEXT_SUB)]
                    + [ft.Text(f"- {name}", size=TEXT_SUB) for name in missing]
                    + [ft.Text("Download them from the workspace's card on the Home page, then open it "
                               "again.", size=TEXT_SUB)],
                    spacing=6, tight=True)),
            )
            dialog.actions = [ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(dialog))]
            self.ctx.page.open(dialog)

    def _open(self, workspace, snapshot, mount=None, switch_tab: bool = True) -> None:
        if workspace is None or self.ctx.upload_view is None:
            return
        # Only when it is actually on screen: opening a ?snapshot= link
        # comes through here with the dialog never having been opened, and
        # Flet refuses to close a dialog it has not been given ("AlertDialog
        # Control must be added to the page first"), which took the whole
        # first render down with it.
        if self.dialog.page is not None:
            self.ctx.page.close(self.dialog)
        # Onto the analysis tabs before the load starts, so its progress
        # bars are what the user watches rather than a Home page that seems
        # to have ignored the click.
        # With a tab per workspace, the browser's tab strip is how the user
        # tells them apart -- so the tab is named after what it holds.
        self.ctx.page.title = f"{workspace.label()} \u2014 Lookin"
        show_analysis = getattr(self.ctx, "show_analysis", None)
        if show_analysis is not None and switch_tab:
            show_analysis()
        # Set before the load, not after: from here on a report exported
        # from this session bookmarks its view into this workspace (see
        # MetricsView._bookmark_for_report), and the load itself reads it to
        # keep each experiment's MLflow run (see UploadView.current_mount).
        self.ctx.current_workspace = workspace
        self.ctx.current_snapshot = None
        self.ctx.upload_view.restore_mount(mount if mount is not None else workspace.mount)
        self._adopt_loaded_mount(workspace)
        self._remember_loadable(workspace)
        lib.mark_opened(workspace.id)
        if self.ctx.state.dataset is None and switch_tab:
            # No dataset yet, or not on this machine: the Data tab is where
            # it gets set up, so that is where the tab opens.
            go_to_data_tab = getattr(self.ctx, "go_to_data_tab", None)
            if go_to_data_tab is not None:
                go_to_data_tab()
        if snapshot is None:
            self._refresh_save_status(baseline=True)
            return

        lib.mark_opened(snapshot.id)
        if not snapshot.understood():
            self._refresh_save_status(baseline=True)
            self._report_loaded(workspace, snapshot, [
                "This snapshot was saved by a newer version of Lookin -- only the data was opened."],
                restored=False)
            return

        warnings = self.ctx.view_state.restore(snapshot.view_state)
        self.ctx.current_snapshot = snapshot
        self._refresh_save_status(baseline=True)
        self._report_loaded(workspace, snapshot, warnings, restored=True)

    def _adopt_loaded_mount(self, workspace) -> None:
        """Record what the load filled in that the workspace left blank.

        A workspace built on the Home page names a dataset and some CSVs and
        nothing else; the Data tab then picks the column mapping, the split
        and the sidecars sitting next to each CSV. Without writing those
        back, the workspace would differ from what is on screen the moment
        it opened, and the app bar would call a workspace nobody touched
        unsaved.

        Only when the load brought back exactly the workspace's dataset and
        experiments -- a partial load (an experiment not downloaded, a CSV
        that failed to read) must not shrink the workspace to what happened
        to succeed.
        """
        loaded = self.ctx.upload_view.current_mount() if self.ctx.upload_view else None
        if loaded is None:
            return

        def change(fresh):
            # Checked against the manifest as it is now: if another tab
            # changed the workspace during this load, what was loaded is no
            # longer the whole of it, and writing it back would undo that.
            same_contents = (
                loaded.dataset_path == fresh.mount.dataset_path
                and {(e.name, e.path) for e in loaded.experiments}
                == {(e.name, e.path) for e in fresh.mount.experiments})
            if not same_contents or save_status.same_mount(loaded, fresh.mount):
                return False
            fresh.mount = loaded
            return True

        fresh = lib.update_workspace(workspace.path, change)
        if fresh is not None:
            workspace.mount = fresh.mount

    def adopt_dataset(self) -> None:
        """Called after every dataset load in the Data tab: a workspace that
        was opened with no usable dataset takes the one just loaded.

        This is what lets a workspace be set up from the inside -- created
        on the Home page with only runs, opened, and given its dataset here.
        A workspace whose own dataset is on this machine is never touched:
        loading a different CSV into its tab is trying something else, and
        the app bar already says it is no longer the saved workspace.
        """
        workspace = getattr(self.ctx, "current_workspace", None)
        loaded = self.ctx.upload_view.current_mount() if self.ctx.upload_view else None
        # Checked on the copy in memory first, and again on disk below: this
        # runs on every dataset load, and hashing a large CSV is not free.
        if workspace is None or loaded is None or workspace.mount.exists():
            return
        digest = lib.hash_file(loaded.dataset_path)

        def change(fresh):
            if fresh.mount.exists():
                return False
            # The dataset and how it is read; the experiments stay the
            # workspace's own -- they carry their MLflow origin, which the
            # loaded rows do not know.
            for name in ("dataset_path", "uuid_col", "crop_col", "label_col", "image_root",
                         "row_filter_col", "row_filter_value", "split_kind", "review_version"):
                setattr(fresh.mount, name, getattr(loaded, name))
            fresh.dataset_hash = digest
            return True

        fresh = lib.update_workspace(workspace.path, change)
        if fresh is None or fresh.mount.dataset_path != loaded.dataset_path:
            return
        workspace.mount, workspace.dataset_hash = fresh.mount, fresh.dataset_hash
        # Now that the dataset matches, the rest of what was loaded -- the
        # sidecars found next to each CSV -- is recorded the usual way.
        self._adopt_loaded_mount(workspace)
        self._refresh_save_status(baseline=True)
        self.ctx.page.title = f"{workspace.label()} \u2014 Lookin"

    # ---- following changes made elsewhere -----------------------------

    def follow_workspace(self, fresh) -> bool:
        """The open workspace's manifest changed outside this tab's Data
        tab -- on its Home card, in this tab or another one. Bring what is
        loaded along, as far as that is safe:

        - nothing loaded and the workspace now names a dataset it did not
          name before: load the whole workspace. That is a dataset set on
          the Home page after the workspace was opened, and an empty Data
          tab next to a card that names one is what this is for.
        - the same dataset loaded: load the experiments that have become
          loadable since this tab last looked -- added from Home, or just
          downloaded. Only those: one the user removed here and has not
          saved yet is not one to bring back.

        Anything else -- a different dataset loaded here -- is left as it
        is, and the app bar says the two differ. Returns False when it could
        not look yet (a load is running), so the caller tries again.
        """
        upload_view = self.ctx.upload_view
        previous = getattr(self.ctx, "current_workspace", None)
        if upload_view is None or previous is None or previous.id != fresh.id:
            return True
        if upload_view.is_loading() or not self._follow_lock.acquire(blocking=False):
            return False
        try:
            from lookin.core import workspace_sources as wss
            loaded = upload_view.current_mount()
            self.ctx.current_workspace = fresh
            if loaded is None:
                if fresh.mount.dataset_path and fresh.mount.dataset_path != previous.mount.dataset_path:
                    self._open(fresh, None, wss.downloaded_only(fresh.mount), switch_tab=False)
                    return True
            elif loaded.dataset_path == fresh.mount.dataset_path:
                here = {e.name for e in loaded.experiments}
                new = [e for e in fresh.mount.experiments
                       if e.name not in here and (e.name, e.path) not in self._loadable_seen
                       and wss.is_downloaded(e)]
                if new:
                    upload_view.add_mount_experiments(new)
                    self._adopt_loaded_mount(fresh)
            self._remember_loadable(fresh)
            self._refresh_save_status()
            return True
        finally:
            self._follow_lock.release()

    def _remember_loadable(self, workspace) -> None:
        """What of `workspace` this tab could have loaded, as (name, path):
        anything outside this set that turns up loadable later is new."""
        from lookin.core import workspace_sources as wss
        self._loadable_seen = {(e.name, e.path) for e in workspace.mount.experiments if wss.is_downloaded(e)}

    def start_watching(self) -> None:
        """Follow the open workspace's manifest from now until stop_watching.

        A thread per tab rather than a signal between tabs: every browser
        tab is its own session, and the only thing two of them share is the
        library on disk. One stat every few seconds is the whole cost.
        """
        threading.Thread(target=self._watch, daemon=True).start()

    def stop_watching(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.wait(WATCH_S):
            try:
                self.check_current_workspace()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def check_current_workspace(self) -> None:
        workspace = getattr(self.ctx, "current_workspace", None)
        if workspace is None or workspace.path is None:
            return
        try:
            stat = (Path(workspace.path) / lib.MANIFEST_NAME).stat()
        except OSError:
            return
        # The inode too: every write is a replace (see library._atomic_write_json),
        # so it moves on each one even when two land within one mtime tick.
        seen = (workspace.id, stat.st_ino, stat.st_mtime_ns)
        if self._seen_manifest == seen:
            return
        fresh = lib.load_workspace(workspace.path)
        if fresh is None:
            return
        if _content(fresh) == _content(workspace) or self.follow_workspace(fresh):
            self._seen_manifest = seen

    # ---- the app bar's Save -------------------------------------------

    def save_current(self, e=None) -> None:
        """Write what is loaded into the workspace open in this tab -- or,
        with none open, ask for a name and make it one."""
        workspace = getattr(self.ctx, "current_workspace", None)
        mount = self.ctx.upload_view.current_mount() if self.ctx.upload_view else None
        if mount is None:
            return
        if workspace is None or not workspace.exists():
            self.open_panel()
            return
        self._update_workspace(workspace, mount)

    def _report_loaded(self, workspace, snapshot, warnings: list, restored: bool) -> None:
        """Say that the snapshot is in, and what did not make it.

        Always shown, not only on a problem. Restoring runs a dataset load
        and then several rebuilds, each reporting itself somewhere
        different -- without one clear "this is now the snapshot you asked
        for" there is no moment the user can point at and call it finished.

        A snapshot restored to nine tenths of what it was is worse than one
        not restored at all, because the user believes they are looking at
        the same thing, so anything that could not be applied is listed
        here rather than logged.
        """
        ok = restored and not warnings
        lines = [
            ft.Text(f"Snapshot {snapshot.id}" + (f" -- {snapshot.name}" if snapshot.name else ""),
                    size=TEXT_SUB, weight=ft.FontWeight.BOLD),
            ft.Text(f"From workspace {workspace.label()}.", size=TEXT_SUB),
        ]
        if warnings:
            lines.append(ft.Text("Could not be applied:", size=TEXT_SUB, weight=ft.FontWeight.BOLD))
            lines += [ft.Text(f"- {w}", size=TEXT_SUB, color=ft.Colors.AMBER) for w in warnings]
        if restored:
            lines.append(ft.Text(
                "The tab and its settings are back. Anything still being rebuilt shows in the "
                "app bar until it finishes.",
                size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE if ok else ft.Icons.WARNING_AMBER_OUTLINED,
                        color=ACCENT if ok else ft.Colors.AMBER),
                ft.Text("Snapshot loaded" if ok else "Snapshot loaded, with gaps",
                        weight=ft.FontWeight.BOLD),
            ], spacing=10),
            content=ft.Container(width=560, content=ft.Column(
                lines, spacing=6, tight=True, scroll=ft.ScrollMode.AUTO)),
        )
        dialog.actions = [ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(dialog))]
        self.ctx.page.open(dialog)

    # ---- export and import -------------------------------------------

    def _on_export(self, e) -> None:
        workspace = self.selected_workspace
        if workspace is None:
            return
        self.file_browser.open(
            on_pick=lambda directory: self._export_to(workspace, directory),
            dirs_only=True, title="Export this workspace to which folder?")

    def _export_to(self, workspace, directory: str) -> None:
        dest = Path(directory) / f"{lib.slugify(workspace.label())}-{workspace.id}.zip"
        self._status(f"Exporting to {dest.name}...", "busy")
        ok, message = lib.export_workspace(
            workspace, dest, include_embeddings=bool(self.include_embeddings.value))
        # The message names the file; the status line adds where it landed,
        # since that path is what the user has to hand to whoever the workspace
        # is for (see components.saved_path_message).
        if ok:
            message = f"{message} Wrote {saved_path_message(self.ctx, dest)}"
        self._status(message, "ok" if ok else "error")

    def _on_import(self, e) -> None:
        self.file_browser.open(
            on_pick=self._import_from, extensions={".zip"}, title="Import which workspace?")

    def _import_from(self, path: str) -> None:
        self._status(f"Importing {Path(path).name}...", "busy")
        workspace, notes = lib.import_workspace(path)
        if workspace is None:
            self._status(" ".join(notes), "error")
            return
        self.selected_workspace, self.selected_snapshot = workspace, None
        self.reload()
        # The crops are the one thing a workspace never carries, so this is the
        # question the person opening it has to answer.
        self._ask_image_root(workspace, notes)

    def _ask_image_root(self, workspace, notes: list) -> None:
        field = ft.TextField(label="Image root path", expand=True, border_radius=8, dense=True)
        browse = ft.IconButton(
            ft.Icons.FOLDER_OPEN,
            on_click=lambda e: self.file_browser.open(
                on_pick=lambda p: (setattr(field, "value", p), bp.safe_update(field)),
                dirs_only=True, title="Where are the crops?"))

        def save(e):
            lib.rebase_image_root(workspace, field.value)
            self.ctx.page.close(dialog)
            self.reload()
            self._status(f"Imported {workspace.label()} ({workspace.id}).")

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Where are the crops on this machine?", weight=ft.FontWeight.BOLD),
            content=ft.Container(width=560, content=ft.Column([
                ft.Text("A workspace carries the dataset and the results, never the images -- so this "
                        "is the one path that has to be answered here.", size=TEXT_SUB),
                ft.Row([field, browse], spacing=6),
                *[ft.Text(f"- {n}", size=TEXT_SMALL, color=ft.Colors.ON_SURFACE_VARIANT) for n in notes],
            ], spacing=10, tight=True)),
        )
        dialog.actions = [
            ft.FilledButton("Save", on_click=save, style=ft.ButtonStyle(bgcolor=ACCENT)),
            ft.TextButton("Skip", on_click=lambda e: self.ctx.page.close(dialog)),
        ]
        self.ctx.page.open(dialog)

    # ---- deleting ----------------------------------------------------

    def _on_delete_workspace(self, e) -> None:
        workspace = self.selected_workspace
        if workspace is None:
            return
        count = len(self.snapshots)
        note = f" and its {count} snapshot(s)" if count else ""
        self._confirm(
            f"Delete {workspace.label()}{note}?",
            "The workspace directory is removed. The dataset and results CSVs it points at are not touched.",
            lambda: self._delete_workspace(workspace))

    def _delete_workspace(self, workspace) -> None:
        if lib.delete_workspace(workspace):
            self.selected_workspace, self.selected_snapshot = None, None
            self.reload()
            self._status(f"Deleted {workspace.label()}.")
        else:
            self._status("Could not delete the workspace.", "error")

    def _on_delete_snapshot(self, e) -> None:
        snapshot = self.selected_snapshot
        if snapshot is None:
            return
        self._confirm(
            f"Delete the snapshot {snapshot.label()}?",
            f"Any report quoting {snapshot.id} will no longer resolve.",
            lambda: self._delete_snapshot(snapshot))

    def _delete_snapshot(self, snapshot) -> None:
        if lib.delete_snapshot(self.selected_workspace, snapshot.id):
            self.selected_snapshot = None
            self._reload_snapshots()
            self._render_snapshots()
            self._render_details()
            self._status("Snapshot deleted.")
        else:
            self._status("Could not delete the snapshot.", "error")

    def _confirm(self, title: str, body: str, on_yes, yes_label: str = "Delete", danger: bool = True) -> None:
        dialog = ft.AlertDialog(modal=True, title=ft.Text(title, weight=ft.FontWeight.BOLD),
                                content=ft.Text(body, size=TEXT_SUB))

        def yes(e):
            self.ctx.page.close(dialog)
            on_yes()

        dialog.actions = [
            ft.TextButton(yes_label, on_click=yes,
                          style=ft.ButtonStyle(color=ft.Colors.RED) if danger else None),
            ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(dialog)),
        ]
        self.ctx.page.open(dialog)
