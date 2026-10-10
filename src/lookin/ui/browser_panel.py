"""The pieces a hierarchy-browsing dialog is made of: a sized dialog, a
column of clickable rows, a filter box, a sort dropdown and a detail panel.

Pulled out of ui/mlflow_panel.py, which grew all of this while becoming the
app's one "browse a tree and pick something" dialog. The Workspaces window
(ui/workspace_panel.py) is the same shape one level shallower -- workspaces, then
their snapshots -- and rebuilding the row rendering, the selection handling
and the dialog sizing a second time would guarantee the two drifted apart.

Nothing here knows what is being browsed. Everything takes the controls it
should act on; no state is held.

Note on what did *not* come with it: the MLflow panel's worker threads,
connection polling and task counters stay there. They exist because a
tracking server is remote and may be slow or absent, and a dialog listing a
local directory has no such problem -- copying them would be inventing a
failure mode to defend against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import flet as ft

from lookin.ui.components import ACCENT

# ---- dialog sizing ----
DIALOG_CHROME_W = 96   # AlertDialog horizontal insets
DIALOG_CHROME_H = 150  # title row + actions row + vertical insets
MIN_DIALOG_WIDTH, MAX_DIALOG_WIDTH = 640, 1560
MIN_DIALOG_HEIGHT, MAX_DIALOG_HEIGHT = 260, 940
COLUMN_SPACING = 14
MIN_COLUMNS_HEIGHT = 300

# ---- type scale ----
# Nothing below TEXT_SMALL: a dialog that is mostly lists is unreadable at
# arm's length on a high-DPI screen once it mixes 9 and 10pt, and reads as
# shouting once its rows reach 14pt.
TEXT_SMALL = 11
TEXT_SUB = 11
TEXT_ROW = 13
TEXT_HEADING = 13
TEXT_TITLE = 15

# ---- render caps ----
# Every row is a Container + Row + Column + Icon + two Texts that has to be
# built and then serialized to the client on every keystroke in a filter
# box. Past a screenful or two nobody reads the rest -- they filter -- so
# only this many are built, and the list says what it is not showing.
MAX_RENDERED_ROWS = 120
# Filter boxes re-render on a timer rather than per keystroke: typing
# "resnet" would otherwise rebuild the whole list six times.
FILTER_DEBOUNCE_S = 0.22

SORT_DROPDOWN_WIDTH = 140


@dataclass
class ListRow:
    """One rendered row, kept so that selecting it costs two control updates
    instead of rebuilding and re-sending the entire list.

    Re-rendering was the MLflow panel's main source of lag on a real
    tracking server: clicking a run in a 200-run experiment rebuilt 200 rows
    purely to move a highlight, and did it on the UI thread.
    """
    container: ft.Container
    bar: ft.Container
    title: ft.Text
    icon: Optional[ft.Icon] = None
    icon_color: Optional[str] = None  # fixed colour; None -> follows selection

    def set_selected(self, selected: bool) -> None:
        self.container.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGHEST if selected else None
        self.bar.bgcolor = ACCENT if selected else ft.Colors.TRANSPARENT
        self.title.weight = ft.FontWeight.BOLD if selected else ft.FontWeight.W_400
        if self.icon is not None and self.icon_color is None:
            self.icon.color = ACCENT if selected else ft.Colors.ON_SURFACE_VARIANT


def safe_update(*controls) -> None:
    """update() only what Flet has actually mounted. Called from worker
    threads too, where the dialog may have been closed since."""
    for control in controls:
        if control is not None and control.page:
            control.update()


def scroll_into_view(listview: ft.ListView, key: str) -> None:
    """Bring a programmatically-selected row into view. Only for selections
    the user did not click, where the highlighted row is otherwise likely to
    be off-screen."""
    if not listview.page:
        return
    try:
        listview.scroll_to(key=key, duration=250)
    except Exception:
        pass  # never let a cosmetic scroll break the selection itself


def sort_dropdown(options: tuple, on_change) -> ft.Dropdown:
    """`options` is a tuple of (key, label). The key is compared by the
    caller's sort and must never be reworded; only the label is display
    text (see CLAUDE.md)."""
    return ft.Dropdown(
        value=options[0][0],
        options=[ft.dropdown.Option(key, label) for key, label in options],
        on_change=on_change, dense=True, text_size=TEXT_SUB, border_radius=8,
        width=SORT_DROPDOWN_WIDTH, content_padding=ft.padding.symmetric(horizontal=10, vertical=4),
        tooltip="Sort order",
    )


def panel_header(step: Optional[int], title: ft.Control,
                 trailing: Optional[ft.Control] = None) -> ft.Control:
    controls: list = []
    if step is not None:
        controls.append(ft.Container(
            content=ft.Text(str(step), size=TEXT_SMALL, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            width=21, height=21, border_radius=11, bgcolor=ACCENT, alignment=ft.alignment.center,
        ))
    controls.append(title)
    if trailing is not None:
        controls.append(ft.Container(expand=True))
        controls.append(trailing)
    return ft.Row(controls, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def list_panel(step: Optional[int], title: str, above: list, listview: ft.ListView,
               width: Optional[int] = None, trailing: Optional[ft.Control] = None) -> ft.Control:
    title_text = ft.Text(title, weight=ft.FontWeight.BOLD, size=TEXT_HEADING)
    controls: list = [panel_header(step, title_text, trailing)]
    controls.extend(above)
    controls.append(ft.Container(
        content=listview, border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=8, padding=4, expand=True,
    ))
    return ft.Container(content=ft.Column(controls, spacing=8, expand=True), width=width)


def details_panel(title_text: ft.Text, body: ft.Control) -> ft.Control:
    return ft.Container(
        content=ft.Column([
            panel_header(None, title_text),
            ft.Container(content=body, border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                         border_radius=8, padding=12, expand=True),
        ], spacing=8, expand=True),
        expand=True,
    )


def group_header(text: str) -> ft.Control:
    return ft.Container(
        content=ft.Text(text.upper(), size=TEXT_SMALL, weight=ft.FontWeight.BOLD,
                        color=ft.Colors.ON_SURFACE_VARIANT),
        padding=ft.padding.only(left=12, top=10, bottom=2),
    )


def row(title: str, subtitle: Optional[str], selected: bool, on_click,
        icon=None, icon_color=None, extra: Optional[ft.Control] = None,
        key: Optional[str] = None) -> ListRow:
    """One clickable list entry. Selection reads as an accent bar plus a
    filled background, not just a bold weight -- at this type size the
    weight alone was easy to miss.

    Returns the ListRow rather than the container, so the caller can
    register it and later flip its selection in place (see select_in_list).
    """
    title_text = ft.Text(
        title, size=TEXT_ROW,
        weight=ft.FontWeight.BOLD if selected else ft.FontWeight.W_400,
        max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, tooltip=title,
    )
    texts: list = [title_text]
    if subtitle:
        texts.append(ft.Text(subtitle, size=TEXT_SUB, color=ft.Colors.ON_SURFACE_VARIANT,
                             max_lines=1, overflow=ft.TextOverflow.ELLIPSIS))
    if extra is not None:
        texts.append(ft.Row([extra], spacing=0, tight=True))

    bar = ft.Container(width=3, border_radius=2,
                       bgcolor=ACCENT if selected else ft.Colors.TRANSPARENT)
    inner: list = [bar]
    icon_control = None
    if icon is not None:
        icon_control = ft.Icon(
            icon, size=16,
            color=icon_color or (ACCENT if selected else ft.Colors.ON_SURFACE_VARIANT))
        inner.append(icon_control)
    inner.append(ft.Column(texts, spacing=1, tight=True, expand=True))

    container = ft.Container(
        key=key,
        content=ft.Row(inner, spacing=9, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.padding.symmetric(horizontal=8, vertical=7),
        border_radius=6,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST if selected else None,
        ink=True,
        on_click=on_click,
    )
    return ListRow(container=container, bar=bar, title=title_text,
                   icon=icon_control, icon_color=icon_color)


def select_in_list(rows: dict, previous_key: Optional[str], new_key: Optional[str]) -> bool:
    """Move a highlight from one row to another without rebuilding the list.
    False when a row involved is not currently rendered (filtered out, or
    past MAX_RENDERED_ROWS), which is the caller's cue to re-render."""
    if new_key is not None and new_key not in rows:
        return False
    touched = []
    for key, selected in ((previous_key, False), (new_key, True)):
        entry = rows.get(key) if key is not None else None
        if entry is not None:
            entry.set_selected(selected)
            touched.append(entry.container)
    safe_update(*touched)
    return True


def overflow_note(shown: int, total: int) -> ft.Control:
    return ft.Container(
        content=ft.Text(f"{shown} of {total:,} shown -- type in the filter box to narrow it down",
                        size=TEXT_SMALL, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        padding=ft.padding.symmetric(horizontal=10, vertical=8),
    )


def capped(items: list, selected_key, key_of: Callable) -> list:
    """The first MAX_RENDERED_ROWS items, always including the selected one.

    A restored pick can sit well past the cap, and a selection nobody can
    see is worse than a row out of order -- so it is moved to the end of
    what is rendered rather than dropped.
    """
    if len(items) <= MAX_RENDERED_ROWS:
        return items
    head = items[:MAX_RENDERED_ROWS]
    if selected_key is None or any(key_of(x) == selected_key for x in head):
        return head
    picked = next((x for x in items if key_of(x) == selected_key), None)
    return head[:-1] + [picked] if picked is not None else head


def fit_dialog_to_page(page, content_container: ft.Container, panels_box: ft.Control,
                       panels: list, fractions: tuple, minimums: tuple,
                       fixed_rows_height: int) -> None:
    """Fit a dialog to the page it is about to open on, and split the
    content width across its panels.

    Called on every open, so resizing the browser window and reopening picks
    up the new size. The panels row is given an explicit height rather than
    left to expand: expanding children silently collapse to nothing when the
    fixed rows above and below them already use up the dialog.
    """
    width = getattr(page, "width", None) or getattr(page.window, "width", None) or 1440
    height = getattr(page, "height", None) or getattr(page.window, "height", None) or 920

    content_w = min(MAX_DIALOG_WIDTH, max(MIN_DIALOG_WIDTH, width - DIALOG_CHROME_W))
    content_h = min(MAX_DIALOG_HEIGHT, max(MIN_DIALOG_HEIGHT, height - DIALOG_CHROME_H))
    content_container.width = content_w
    content_container.height = content_h

    inner_w = content_w - COLUMN_SPACING * len(panels)
    for panel, fraction, minimum in zip(panels, fractions, minimums):
        panel.width = max(minimum, inner_w * fraction)

    panels_box.height = max(MIN_COLUMNS_HEIGHT, content_h - fixed_rows_height)
