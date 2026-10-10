"""Is what is on screen saved anywhere, and under what name.

Two questions, not one, because they have different answers and different
costs to get wrong:

    Is this data saved as a workspace?    -- the dataset, its mapping and its
                                          experiments. Lost by loading
                                          something else.
    Is this state saved as a snapshot? -- the tab, the filters, the
                                          projection settings. Lost by
                                          changing them.

The app bar shows both, so "I will just open this other workspace" is never a
silent way to lose an afternoon's filters.

How "saved" is decided
----------------------
A workspace is saved when one was opened or written in this session, its
directory is still there, and the mount on screen still matches the one it
holds. Loading a different dataset therefore un-saves it, which is exactly
what the user needs to be told.

A snapshot is saved when one was opened or written in this session and the
state captured right now is byte-for-byte the state it holds. That
comparison is the honest test -- anything weaker (a dirty flag each view
has to remember to set) is a flag that will be missed somewhere and will
then claim work is safe when it is not.

Cost: a comparison captures the view state, which reads control values and
allocates a few small dicts. It is cheap enough to do on every refresh, and
refreshes are driven by data changes and tab switches, not by keystrokes.
"""
from __future__ import annotations

from dataclasses import dataclass

import flet as ft

from lookin.core.library import mount_to_dict
from lookin.ui.components import ACCENT

# Fields of a mount that move on their own and say nothing about whether the
# user's work is saved: when the file was last stat'ed, and how big it was.
VOLATILE_MOUNT_FIELDS = ("saved_at", "dataset_size", "dataset_mtime")

# Longer than this and the app bar starts pushing the tabs around.
MAX_NAME_CHARS = 22


@dataclass
class SaveStatus:
    workspace_name: str = ""
    workspace_saved: bool = False
    snapshot_name: str = ""
    snapshot_saved: bool = False
    has_data: bool = False
    # Has the state moved since the last save or load? Tracked separately
    # from snapshot_saved because the two answer different questions: a
    # session that has never had a snapshot is *not* saved, but if nothing
    # has been touched since it was loaded there is also nothing to lose.
    view_changed: bool = False

    def anything_unsaved(self) -> bool:
        """True when loading something else would throw away work.

        Deliberately not "not saved". Warning every time a snapshot happens
        not to exist would fire hardest right after a workspace is saved,
        which is the one moment nothing is at risk -- and a warning that
        cries wolf is one nobody reads. So: the data is not in a workspace, or
        the settings have actually been changed since the last save or load.
        """
        return self.has_data and (not self.workspace_saved or self.view_changed)


def _comparable_mount(mount) -> dict:
    if mount is None:
        return {}
    payload = mount_to_dict(mount)
    for field in VOLATILE_MOUNT_FIELDS:
        payload.pop(field, None)
    return payload


def same_mount(a, b) -> bool:
    """Equal in everything that says whether the user's work is saved."""
    return _comparable_mount(a) == _comparable_mount(b)


def compute(ctx) -> SaveStatus:
    """What is saved right now. Never raises: this drives an indicator, and
    an indicator that can take the app down is worse than no indicator."""
    status = SaveStatus()
    try:
        upload_view = getattr(ctx, "upload_view", None)
        current_mount = upload_view.current_mount() if upload_view is not None else None
    except Exception:  # noqa: BLE001
        current_mount = None
    status.has_data = current_mount is not None

    workspace = getattr(ctx, "current_workspace", None)
    if workspace is not None:
        status.workspace_name = workspace.label()
        try:
            status.workspace_saved = bool(
                current_mount is not None
                and workspace.exists()
                and _comparable_mount(workspace.mount) == _comparable_mount(current_mount)
            )
        except Exception:  # noqa: BLE001
            status.workspace_saved = False

    try:
        current_view = ctx.view_state.capture()
    except Exception:  # noqa: BLE001
        current_view = None

    snapshot = getattr(ctx, "current_snapshot", None)
    if snapshot is not None:
        status.snapshot_name = snapshot.label()
        status.snapshot_saved = current_view is not None and snapshot.view_state == current_view

    baseline = getattr(ctx, "view_state_baseline", None)
    status.view_changed = bool(
        status.has_data and current_view is not None
        and baseline is not None and baseline != current_view
    )
    return status


def mark_baseline(ctx) -> None:
    """Remember the state as "nothing to lose from here".

    Called after a save and after a load. Cheap, and the alternative -- a
    dirty flag every view has to remember to set -- is a flag that will be
    missed somewhere and will then claim work is safe when it is not.
    """
    try:
        ctx.view_state_baseline = ctx.view_state.capture()
    except Exception:  # noqa: BLE001
        ctx.view_state_baseline = None


def _elide(name: str) -> str:
    return name if len(name) <= MAX_NAME_CHARS else name[:MAX_NAME_CHARS - 1] + "…"


class SaveStatusIndicator:
    """Two small chips in the app bar: one per question above.

    Clicking either opens the Workspaces window, because that is where both answers
    are changed. Refreshed on every data change and on every tab switch --
    not on every keystroke, so a filter typed and not yet left can read as
    saved for a moment; clicking through to the Workspaces window recomputes.
    """

    def __init__(self, ctx, on_click=None, on_save=None):
        self.ctx = ctx
        self._on_click = on_click
        self._on_save = on_save
        self.workspace_icon = ft.Icon(ft.Icons.INVENTORY_2_OUTLINED, size=15)
        self.workspace_text = ft.Text("", size=11)
        self.snapshot_icon = ft.Icon(ft.Icons.BOOKMARK_OUTLINE, size=15)
        self.snapshot_text = ft.Text("", size=11)

        self.workspace_chip = ft.Container(
            content=ft.Row([self.workspace_icon, self.workspace_text], spacing=5, tight=True),
            padding=ft.padding.symmetric(horizontal=8, vertical=4),
            border_radius=8, ink=True, on_click=self._clicked,
        )
        self.snapshot_chip = ft.Container(
            content=ft.Row([self.snapshot_icon, self.snapshot_text], spacing=5, tight=True),
            padding=ft.padding.symmetric(horizontal=8, vertical=4),
            border_radius=8, ink=True, on_click=self._clicked,
        )
        # The one-click answer to the workspace chip turning amber: shown
        # only while there is something to save, so its being there is the
        # signal. Snapshots stay a deliberate act in the Workspaces window --
        # each is an id a report may quote, not a file to keep overwriting.
        self.save_button = ft.FilledButton(
            "Save", icon=ft.Icons.SAVE_OUTLINED, visible=False, on_click=self._save_clicked,
            style=ft.ButtonStyle(bgcolor=ACCENT, padding=ft.padding.symmetric(horizontal=12)),
        )
        self.control = ft.Row([self.save_button, self.workspace_chip, self.snapshot_chip],
                              spacing=2, tight=True,
                              vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.refresh()

    def _clicked(self, e):
        if self._on_click is not None:
            self._on_click(e)

    def _save_clicked(self, e):
        if self._on_save is not None:
            self._on_save(e)

    def refresh(self) -> None:
        status = compute(self.ctx)
        self._apply(
            self.workspace_chip, self.workspace_icon, self.workspace_text,
            saved=status.workspace_saved, name=status.workspace_name, has_data=status.has_data,
            saved_word="Workspace", unsaved_word="No workspace",
            saved_tip="This data is saved as a workspace.",
            unsaved_tip="This data is not saved as a workspace -- loading something else loses the mount.",
        )
        self._apply(
            self.snapshot_chip, self.snapshot_icon, self.snapshot_text,
            saved=status.snapshot_saved, name=status.snapshot_name, has_data=status.has_data,
            saved_word="Snapshot", unsaved_word="No snapshot",
            saved_tip="What is on screen matches this snapshot.",
            unsaved_tip="The filters and settings on screen are not saved in any snapshot.",
        )
        self._apply_save_button(status)
        for control in (self.save_button, self.workspace_chip, self.snapshot_chip, self.workspace_text,
                        self.snapshot_text, self.workspace_icon, self.snapshot_icon):
            if control.page:
                control.update()

    def _apply_save_button(self, status: SaveStatus) -> None:
        button = self.save_button
        button.visible = self._on_save is not None and status.has_data and not status.workspace_saved
        if not button.visible:
            return
        if status.workspace_name:
            button.text = "Save"
            button.tooltip = (f"Write the dataset and experiments loaded now into '{status.workspace_name}'. "
                              "Its id and its snapshots are kept.")
        else:
            button.text = "Save as workspace"
            button.tooltip = "Keep the dataset and experiments loaded now as a named workspace."

    @staticmethod
    def _apply(chip, icon, text, *, saved, name, has_data, saved_word, unsaved_word,
               saved_tip, unsaved_tip) -> None:
        # Nothing loaded is neither saved nor unsaved -- saying "unsaved"
        # over an empty app would be a warning about nothing.
        if not has_data:
            icon.color = ft.Colors.ON_SURFACE_VARIANT
            text.value = "--"
            text.color = ft.Colors.ON_SURFACE_VARIANT
            chip.tooltip = "Nothing loaded yet."
            return
        if saved:
            icon.color = ACCENT
            text.value = _elide(name) if name else saved_word
            text.color = ft.Colors.ON_SURFACE
            chip.tooltip = f"{saved_tip} ({name})" if name else saved_tip
            return
        icon.color = ft.Colors.AMBER
        # A name with a dot means "this one, but changed since" -- more use
        # than a bare "unsaved", which loses which snapshot it drifted from.
        text.value = f"{_elide(name)} *" if name else unsaved_word
        text.color = ft.Colors.AMBER
        chip.tooltip = f"{unsaved_tip} (last: {name})" if name else unsaved_tip

