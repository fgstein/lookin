"""Browsing the *server's* filesystem -- the machine running `lookin web`.

A browser's own file input never hands back a real path, only a bare file
name (sandboxing, in every browser). Lookin runs on the machine whose files
it loads, so this lists directories from the server process and the picked
path is always the complete, absolute one. Same rules as
lookin.ui.server_browser.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from lookin.web import html as h
from lookin.web.views.base import Dialog


class FileBrowserDialog(Dialog):
    width = 640

    def __init__(self, on_pick: Callable[[str], None], start_path: str = "",
                 extensions: Optional[set] = None, dirs_only: bool = False, title: str = "Select a file"):
        super().__init__(title)
        self.on_pick_fn = on_pick
        self.extensions = {x.lower() for x in extensions} if extensions else None
        self.dirs_only = dirs_only
        start = Path(start_path).expanduser() if (start_path or "").strip() else Path.home()
        if start.is_file():
            start = start.parent
        self.current = start if start.is_dir() else Path.home()
        self.entries: list = []

    def _scan(self) -> tuple:
        try:
            entries = [e for e in os.scandir(self.current) if not e.name.startswith(".")]
        except OSError as ex:
            self.error = f"Cannot read this folder: {ex}"
            return [], []
        dirs = sorted((e for e in entries if e.is_dir(follow_symlinks=False)), key=lambda e: e.name.lower())
        files = []
        if not self.dirs_only:
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    continue
                if self.extensions and Path(e.name).suffix.lower() not in self.extensions:
                    continue
                files.append(e)
            files.sort(key=lambda e: e.name.lower())
        return dirs, files

    def body(self) -> str:
        dirs, files = self._scan()
        self.entries = [("d", e.path) for e in dirs] + [("f", e.path) for e in files]
        rows = []
        for i, (kind, path) in enumerate(self.entries):
            name = Path(path).name
            if kind == "d":
                rows.append(f'<button type="button" class="fb-row"{self.a("enter", i=i)}>'
                            f'{h.icon("folder", 15, "c-acc")}<span>{h.e(name)}</span></button>')
            else:
                size = ""
                try:
                    size = h.human_size(Path(path).stat().st_size)
                except OSError:
                    pass
                rows.append(f'<button type="button" class="fb-row"{self.a("pick", i=i)}>'
                            f'{h.icon("file", 15)}<span class="grow">{h.e(name)}</span>'
                            f'<span class="muted small">{h.e(size)}</span></button>')
        if not rows:
            rows.append('<div class="empty">(empty)</div>')
        path_input = h.text_input(str(self.current), name="path", enter=h.on_enter(self.target, "go"), mono=True,
                                  cls="grow", extra=' data-force="1"')
        return (f'<div class="fb-path">{h.button(icon_name="chev-up", act_attr=self.a("up"), title="Parent folder")}'
                f'{path_input}</div><div class="fb-list">{"".join(rows)}</div>')

    def footer(self) -> str:
        use = h.button("Use this folder", act_attr=self.a("use_dir"), kind="primary", icon_name="check") \
            if self.dirs_only else ""
        return h.button("Cancel", act_attr=self.a("close")) + use

    def on_up(self, **_):
        parent = self.current.parent
        if parent != self.current:
            self.current = parent
        self.error = ""
        self.refresh()

    def on_enter(self, i: int = 0, **_):
        if 0 <= i < len(self.entries):
            self.current = Path(self.entries[i][1])
        self.error = ""
        self.refresh()

    def on_go(self, value: str = "", **_):
        candidate = Path((value or "").strip()).expanduser()
        if candidate.is_dir():
            self.current = candidate
            self.error = ""
        elif candidate.is_file() and not self.dirs_only:
            self._pick(candidate)
            return
        else:
            self.error = f"Not a valid {'folder' if self.dirs_only else 'file or folder'}: {candidate}"
        self.refresh()

    def on_pick(self, i: int = 0, **_):
        if 0 <= i < len(self.entries):
            self._pick(Path(self.entries[i][1]))

    def on_use_dir(self, **_):
        self._pick(self.current)

    def _pick(self, path: Path) -> None:
        self.close()
        self.on_pick_fn(str(path))


def menu(items, *, icon_name: str = "dots", title: str = "More", right: bool = True, label: str = "",
         small: bool = True) -> str:
    """A popup menu. `items`: (label, act_attr, icon, danger) tuples, or
    "hr" for a separator, or raw HTML strings starting with "<"."""
    out = []
    for item in items:
        if item == "hr":
            out.append("<hr>")
        elif isinstance(item, str):
            out.append(item)
        else:
            lbl, act_attr, ic, danger = (list(item) + [None, None])[:4]
            out.append(f'<button type="button"{act_attr}{" class=danger" if danger else ""}>'
                       f'{h.icon(ic, 14) if ic else ""}<span>{h.e(lbl)}</span></button>')
    btn = (f'<button type="button" class="btn {"btn-sm " if small else ""}btn-ghost{" btn-icon" if not label else ""}" '
           f'data-menu="1" title="{h.e(title)}" aria-label="{h.e(title)}">{h.icon(icon_name, 14)}'
           f'{f"<span>{h.e(label)}</span>" if label else ""}</button>')
    return f'<span class="menu-wrap{" menu-r" if right else ""}">{btn}<div class="menu">{"".join(out)}</div></span>'
