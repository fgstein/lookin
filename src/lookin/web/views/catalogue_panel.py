"""The catalogue of views of the Embeddings and Compare embeddings tabs
(core/view_catalogue.py), as a card of the tab.

Two lists: the views saved in the workspace -- opened, renamed, deleted --
and this session's recent runs, each one click from being saved. Opening a
view redraws it without computing anything: the tab hands over its result
(`catalogue_current()`) and takes one back (`catalogue_open(view, arrays)`),
both implemented by the tab, which knows what its results are.

The tab routes its "cat" action here: `on_cat(op, **args)`.
"""
from __future__ import annotations

import traceback
from datetime import datetime

from lookin.core import view_catalogue as vc
from lookin.web import html as h
from lookin.web.views.base import ConfirmDialog, PromptDialog

MANUAL_PAGE = "saved-views"


def _when(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return stamp or ""


class CataloguePanel:
    def __init__(self, view, tab: str):
        self.view = view
        self.tab = tab
        self.history: list = []          # this session's runs, newest first (SavedView, arrays in memory)
        self.name_value = ""
        self.message = ("", "info")
        self.current_id = ""             # the view on screen, when it is one of these

    @property
    def s(self):
        return self.view.s

    def workspace(self):
        workspace = getattr(self.s, "current_workspace", None)
        return workspace if workspace is not None and workspace.exists() else None

    # ---- what the tab tells it ----

    def add_run(self, view) -> None:
        """A run just shown: the newest of the session's history."""
        if view is None:
            return
        self.history = vc.push_history(self.history, view)
        self.current_id = view.id

    # ---- actions ----

    def handle(self, op: str, args: dict):
        try:
            if op == "name":
                self.name_value = str(args.get("value", ""))
            elif op == "save":
                self._save_current()
            elif op == "save_recent":
                self._save_recent(str(args.get("id", "")))
            elif op == "open":
                self._open(str(args.get("id", "")))
            elif op == "rename":
                self._rename(str(args.get("id", "")))
            elif op == "delete":
                self._delete(str(args.get("id", "")))
        except Exception as ex:  # noqa: BLE001 -- said in the card, not a broken tab
            traceback.print_exc()
            self.message = (str(ex), "error")
        self.view.refresh()

    def _save_current(self) -> None:
        workspace = self.workspace()
        if workspace is None:
            self.message = ("Views are saved in a workspace of the library; this session is not one.", "error")
            return
        current = self.view.catalogue_current()
        if current is None:
            self.message = ("Nothing to save yet: run the projection, and save once it is on screen.", "info")
            return
        if current.saved:
            # Already in the workspace: saving it again keeps a copy, the
            # original stays as it is (rename it from the list).
            from dataclasses import replace as _replace
            current = _replace(current, id=vc.new_id(), saved=False)
        if self.name_value.strip():
            current.name = self.name_value.strip()[:vc.NAME_MAX]
        vc.save_view(workspace, current)
        self.history = [v for v in self.history if v.id != current.id]
        self.current_id = current.id
        self.name_value = ""
        self.message = (f"Saved \"{current.name}\" in this workspace.", "success")

    def _save_recent(self, view_id: str) -> None:
        workspace = self.workspace()
        found = next((v for v in self.history if v.id == view_id), None)
        if workspace is None or found is None:
            return
        vc.save_view(workspace, found)
        self.history = [v for v in self.history if v.id != view_id]
        self.message = (f"Saved \"{found.name}\" in this workspace.", "success")

    def _open(self, view_id: str) -> None:
        found = next((v for v in self.history if v.id == view_id), None)
        arrays = found.arrays if found is not None else None
        if found is None:
            workspace = self.workspace()
            if workspace is None:
                return
            found = vc.find_view(workspace, view_id)
            arrays = vc.load_arrays(workspace, found)
        self.current_id = found.id
        self.message = ("", "info")
        self.view.catalogue_open(found, arrays)

    def _rename(self, view_id: str) -> None:
        workspace = self.workspace()
        if workspace is None:
            return
        found = vc.find_view(workspace, view_id)

        def submit(values):
            try:
                vc.rename_view(workspace, view_id, values.get("name", ""))
            except ValueError as ex:
                return str(ex)
            self.view.refresh()
            return None

        self.s.open_dialog(PromptDialog("Rename the view", [("name", "Name", found.name)], submit, submit_label="Rename"))

    def _delete(self, view_id: str) -> None:
        workspace = self.workspace()
        if workspace is None:
            return
        found = vc.find_view(workspace, view_id)

        def yes():
            vc.delete_view(workspace, view_id)
            self.view.refresh()

        self.s.open_dialog(ConfirmDialog("Delete the view", f"Delete \"{found.name}\" from this workspace's views?", yes))

    # ---- the card ----

    def render(self) -> str:
        a = lambda op, **kw: h.act(self.view.name, "cat", op=op, **kw)  # noqa: E731
        workspace = self.workspace()
        has_current = self.view.catalogue_current(peek=True)
        computing = bool(getattr(self.view, "_computing", False) or getattr(self.view, "running", False))
        rows = []
        for v in vc.list_views(workspace, self.tab) if workspace is not None else []:
            changed = vc.changed_files(v)
            badges = (f'<span class="chip small c-warn" title="{h.e("Changed since: " + ", ".join(changed))}">out of date</span>'
                      if changed else "")
            badges += ("" if v.has_model else '<span class="chip small" title="A UMAP or t-SNE view keeps positions only: '
                       'adding crops to it means running the projection again">positions only</span>')
            rows.append(self._row(v, a, badges, saved=True))
        recent = [self._row(v, a, "", saved=False) for v in self.history]
        save = ""
        if workspace is not None:
            hint = ('<div class="muted small">Computing: save the view once it is on screen.</div>'
                    if computing else "")
            save = hint + (f'<div class="row-nw">{h.text_input(self.name_value, placeholder="Name of the view on screen", width=230, change=h.on_change(self.view.name, "cat", op="name"), extra=" data-force=1")}'
                    f'{h.button("Save this view", act_attr=a("save"), small=True, icon_name="bookmark", disabled=not has_current, title="Keep what is on screen -- settings, positions, measures -- in this workspace")}</div>')
        else:
            save = '<div class="muted small">Views are saved in a workspace of the library; this session is not one.</div>'
        text, kind = self.message
        msg = h.banner(text, kind) if text else ""
        body = [save, msg]
        body.append('<b class="small">Saved in this workspace</b>')
        body.append("".join(rows) or '<div class="muted small">None yet.</div>')
        if recent:
            body.append('<b class="small">Recent runs (this session)</b>')
            body.append("".join(recent))
        return h.section("Views", f'<div class="stack vcat">{"".join(b for b in body if b)}</div>', manual=MANUAL_PAGE,
                         sub="open a computed view again, at once")

    def _row(self, v, a, badges: str, saved: bool) -> str:
        on = " on" if v.id == self.current_id else ""
        open_btn = (f'<button type="button" class="vcat-open{on}"{a("open", id=v.id)} title="{h.e(v.summary)}">'
                    f'<span class="ell small">{h.e(v.name)}</span>'
                    f'<span class="muted small">{h.e(_when(v.created_at))}</span></button>')
        if saved:
            tools = (h.button(icon_name="edit", act_attr=a("rename", id=v.id), small=True, kind="ghost", title="Rename")
                     + h.button(icon_name="trash", act_attr=a("delete", id=v.id), small=True, kind="ghost", title="Delete"))
        else:
            tools = (h.button(icon_name="bookmark", act_attr=a("save_recent", id=v.id), small=True, kind="ghost",
                              title="Save in this workspace", disabled=self.workspace() is None))
        return f'<div class="row-nw vcat-row">{open_btn}{badges}<span class="grow"></span>{tools}</div>'
