"""A file/folder browser dialog that lists the *server's* filesystem (the
machine running `lookin ui`) rather than the client's.

Flet's own FilePicker opens the browser's native file input, which -- browser
sandboxing, not a Flet limitation -- never hands back a real filesystem path,
only the bare filename with no directory. That's a hard wall in web view: no
client-side trick gets around it, in any browser. But `lookin ui` runs as a
Python process on the same machine whose files it's meant to load, so this
dialog just lists directories with os.scandir() from that process and lets
the user navigate to the file they want -- the resulting path is always the
complete, absolute one, which is what actually loading the file needs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import flet as ft

from lookin.ui.components import ACCENT

LIST_HEIGHT = 360


class ServerFileBrowser:
    """One shared dialog, reused for every "Browse" button in the Data tab.
    Only one can be open at a time, so a single instance per UploadView is
    enough -- open() reconfigures it (callback, extension filter, starting
    directory) for whichever field triggered it."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._on_pick: Optional[Callable[[str], None]] = None
        self._extensions: Optional[set] = None
        self._dirs_only = False
        self._current_dir: Path = Path.home()

        self.title_text = ft.Text("Select a file", weight=ft.FontWeight.BOLD)
        self.path_field = ft.TextField(
            label="Path", expand=True, border_radius=8, dense=True, on_submit=self._on_path_submit,
        )
        self.up_btn = ft.IconButton(ft.Icons.ARROW_UPWARD, tooltip="Parent folder", on_click=self._go_up)
        self.error_text = ft.Text("", size=11, color=ft.Colors.RED)
        self.list_view = ft.ListView(height=LIST_HEIGHT, spacing=1)
        self.select_dir_btn = ft.FilledButton(
            "Use this folder", icon=ft.Icons.CHECK, on_click=self._on_select_current_dir,
            visible=False, style=ft.ButtonStyle(bgcolor=ACCENT),
        )

        self.dialog = ft.AlertDialog(
            modal=True,
            title=self.title_text,
            content=ft.Container(
                width=620,
                content=ft.Column([
                    ft.Row([self.up_btn, self.path_field]),
                    self.error_text,
                    ft.Container(
                        content=self.list_view, border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                        border_radius=8, padding=4,
                    ),
                ], spacing=8, tight=True),
            ),
            actions=[
                self.select_dir_btn,
                ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(self.dialog)),
            ],
        )

    def open(
        self, on_pick: Callable[[str], None], start_path: str = "",
        extensions: Optional[set] = None, dirs_only: bool = False, title: str = "Select a file",
    ) -> None:
        """`on_pick(absolute_path)` fires once a file is clicked (or, in
        dirs_only mode, once "Use this folder" is clicked). `start_path` is
        whatever the target field already had -- a file path lands on that
        file's directory, a directory path is used as-is, anything invalid
        or blank falls back to the home directory."""
        self._on_pick = on_pick
        self._extensions = {e.lower() for e in extensions} if extensions else None
        self._dirs_only = dirs_only
        self.title_text.value = title
        self.select_dir_btn.visible = dirs_only

        start = Path(start_path).expanduser() if start_path.strip() else Path.home()
        if start.is_file():
            start = start.parent
        self._current_dir = start if start.is_dir() else Path.home()
        self._render()
        self.ctx.page.open(self.dialog)

    def _go_up(self, e):
        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._current_dir = parent
            self._render()

    def _on_path_submit(self, e):
        candidate = Path(self.path_field.value.strip()).expanduser()
        if candidate.is_dir():
            self._current_dir = candidate
            self._render()
        elif candidate.is_file() and not self._dirs_only:
            self._pick(candidate)
        else:
            self.error_text.value = f"Not a valid {'folder' if self._dirs_only else 'file or folder'}: {candidate}"
            if self.error_text.page:
                self.error_text.update()

    def _on_select_current_dir(self, e):
        self._pick(self._current_dir)

    def _pick(self, path: Path) -> None:
        callback = self._on_pick
        self.ctx.page.close(self.dialog)
        if callback:
            callback(str(path))

    def _enter_dir(self, path: Path) -> None:
        self._current_dir = path
        self._render()

    def _render(self) -> None:
        self.path_field.value = str(self._current_dir)
        self.error_text.value = ""
        self.list_view.controls.clear()

        try:
            entries = [e for e in os.scandir(self._current_dir) if not e.name.startswith(".")]
        except OSError as ex:
            self.error_text.value = f"Cannot read this folder: {ex}"
            entries = []

        dirs = sorted((e for e in entries if e.is_dir(follow_symlinks=False)), key=lambda e: e.name.lower())
        files = []
        if not self._dirs_only:
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    continue
                if self._extensions and Path(e.name).suffix.lower() not in self._extensions:
                    continue
                files.append(e)
            files.sort(key=lambda e: e.name.lower())

        for entry in dirs:
            self.list_view.controls.append(self._row(entry.name, ft.Icons.FOLDER, ACCENT, lambda e, p=Path(entry.path): self._enter_dir(p)))
        for entry in files:
            self.list_view.controls.append(self._row(entry.name, ft.Icons.INSERT_DRIVE_FILE_OUTLINED, None, lambda e, p=Path(entry.path): self._pick(p)))
        if not dirs and not files:
            self.list_view.controls.append(ft.Text("(empty)", size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))

        if self.path_field.page:
            self.path_field.update()
        if self.error_text.page:
            self.error_text.update()
        if self.list_view.page:
            self.list_view.update()

    def _row(self, name: str, icon, icon_color, on_click) -> ft.Control:
        return ft.Container(
            content=ft.Row([ft.Icon(icon, size=16, color=icon_color), ft.Text(name, size=12)], spacing=8),
            padding=ft.padding.symmetric(horizontal=8, vertical=6),
            border_radius=6,
            ink=True,
            on_click=on_click,
        )
