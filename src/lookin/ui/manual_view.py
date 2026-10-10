"""The reference manual: one window, a table of contents down the side, and
a page of typeset explanation for every number this app puts on screen.

Every (i) button in the app opens *this* window at its page (see
components.info_button). One window rather than a tooltip per chart for two
reasons: a tooltip has room for a sentence, not for a definition and its
pitfalls; and a reader who came in through the F1 chart usually leaves
having read about precision, macro-averaging and support too, which only
works if the neighbouring pages are one click away.

The page content itself is data and lives in core/manual/ -- this module
only knows how to draw a block. Equations go through ui/mathtext.py.
"""
from __future__ import annotations

import re

import flet as ft

from lookin.core import manual
from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)
from lookin.ui.components import ACCENT
from lookin.ui.mathtext import (
    DISPLAY_SIZE, INK, INK_MUTED, MathRenderError, render_math,
)

BODY_SIZE = 14
BODY_COLOR = ft.Colors.ON_SURFACE
MUTED = ft.Colors.ON_SURFACE_VARIANT

# Width of the inter-word gap in prose, in pixels. Prose is laid out as a
# wrapping Row of words rather than a single Text because a word and an
# inline equation have to sit on the same line, and only a Row can hold an
# Image beside a Text (see _inline).
WORD_GAP = 4

# Added to the surrounding text size when typesetting an inline formula.
# Positive because mathtext's italics have a smaller x-height than the body
# face, so a symbol set at the prose size reads as a footnote next to it.
INLINE_LIFT = 1.5

CALLOUT_STYLE = {
    CALLOUT_PITFALL: (ft.Icons.REPORT_PROBLEM_OUTLINED, ft.Colors.ORANGE),
    CALLOUT_WARNING: (ft.Icons.ERROR_OUTLINE, ft.Colors.RED),
    CALLOUT_TIP: (ft.Icons.LIGHTBULB_OUTLINE, ft.Colors.GREEN),
    CALLOUT_EXAMPLE: (ft.Icons.CALCULATE_OUTLINED, ACCENT),
}


# ---- inline prose -------------------------------------------------------

def _math_image(tex: str, size: float, color: str) -> ft.Control:
    """One formula as an image, or its source in red if it will not typeset.

    Never raises: a manual that refuses to open because one page has a
    stray brace in it is worse than a page with one broken formula on it,
    and tests/test_manual.py typesets every formula in the manual anyway.
    """
    try:
        data, width, height = render_math(tex, size, color)
    except MathRenderError:
        return ft.Text(tex, size=size, color=ft.Colors.RED, font_family="monospace")
    return ft.Image(src_base64=data, width=width, height=height, fit=ft.ImageFit.CONTAIN)


# What a paragraph marks up: $mathematics$ and *emphasis*. Deliberately
# the only two -- a page is prose with formulas in it, not a document
# format, and every additional marker is one more character that cannot be
# written literally.
_SPAN = re.compile(r"\$([^$]*)\$|\*([^*]*)\*")


def _inline(text: str, size: float = BODY_SIZE, color=BODY_COLOR, italic: bool = False) -> ft.Control:
    """A paragraph in which $...$ spans are typeset mathematics and *...*
    spans are emphasised.

    Laid out as a wrapping Row of *units*, where a unit is everything
    between two spaces -- so "$p$." keeps its full stop welded to the
    symbol, and a line never breaks between a formula and the comma after
    it.
    """
    units: list = []
    current: list = []

    def flush():
        if not current:
            return
        units.append(current[0] if len(current) == 1 else ft.Row(
            list(current), spacing=0, tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ))
        current.clear()

    def words(chunk: str, emphasised: bool = False) -> None:
        for piece in re.split(r"(\s+)", chunk):
            if not piece:
                continue
            if piece.isspace():
                flush()
            else:
                current.append(ft.Text(piece, size=size, color=color,
                                       italic=italic or emphasised,
                                       weight=ft.FontWeight.W_500 if emphasised else None))

    position = 0
    for match in _SPAN.finditer(text):
        words(text[position:match.start()])
        formula, emphasis = match.group(1), match.group(2)
        if formula:
            current.append(_math_image(formula, size + INLINE_LIFT,
                                       INK if color is BODY_COLOR else INK_MUTED))
        elif emphasis:
            words(emphasis, emphasised=True)
        position = match.end()
    words(text[position:])
    flush()

    return ft.Row(units, wrap=True, spacing=WORD_GAP, run_spacing=3,
                  vertical_alignment=ft.CrossAxisAlignment.CENTER)


# ---- one block at a time ------------------------------------------------

def _equation_block(block: Equation) -> ft.Control:
    # Scrollable: a display equation is an image of a fixed width, and the
    # widest ones here are around 430px. In a narrow window that is wider
    # than the column, and a clipped formula is a wrong formula -- so it
    # scrolls sideways instead.
    rows: list = [ft.Row([_math_image(block.tex, DISPLAY_SIZE, INK)],
                         alignment=ft.MainAxisAlignment.CENTER,
                         scroll=ft.ScrollMode.AUTO)]
    if block.note:
        rows.append(_inline(block.note, size=12, color=MUTED, italic=True))
    if block.where:
        rows.append(ft.Container(height=2))
        for symbol, meaning in block.where:
            rows.append(ft.Row([
                ft.Container(_math_image(symbol, BODY_SIZE + INLINE_LIFT, INK_MUTED),
                             width=64, alignment=ft.alignment.center_right),
                ft.Container(_inline(meaning, size=12.5, color=MUTED), expand=True),
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER))
    return ft.Container(
        content=ft.Column(rows, spacing=6, tight=True),
        padding=ft.padding.symmetric(vertical=12, horizontal=14),
        margin=ft.margin.symmetric(vertical=4),
        border_radius=10,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
    )


def _bullets_block(block: Bullets) -> ft.Control:
    rows = []
    for number, item in enumerate(block.items, start=1):
        marker = f"{number}." if block.numbered else "•"
        rows.append(ft.Row([
            ft.Container(ft.Text(marker, size=BODY_SIZE, color=MUTED), width=22,
                         alignment=ft.alignment.top_right, padding=ft.padding.only(top=1)),
            ft.Container(_inline(item), expand=True),
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START))
    return ft.Column(rows, spacing=6, tight=True)


def _table_block(block: Table) -> ft.Control:
    weights = block.weights or tuple(1 for _ in block.headers)
    header = ft.Row([
        ft.Container(ft.Text(h, size=12, weight=ft.FontWeight.BOLD), expand=w)
        for h, w in zip(block.headers, weights)
    ], spacing=8)
    rows = [ft.Container(header, padding=ft.padding.symmetric(vertical=6, horizontal=10),
                         bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST, border_radius=6)]
    for row in block.rows:
        rows.append(ft.Container(
            ft.Row([ft.Container(_inline(str(cell), size=12.5), expand=w)
                    for cell, w in zip(row, weights)], spacing=8),
            padding=ft.padding.symmetric(vertical=5, horizontal=10),
        ))
    if block.caption:
        rows.append(_inline(block.caption, size=11.5, color=MUTED, italic=True))
    return ft.Container(ft.Column(rows, spacing=2, tight=True),
                        margin=ft.margin.symmetric(vertical=4))


def _howto_block(block: HowToRead) -> ft.Control:
    rows = [ft.Row([ft.Icon(ft.Icons.VISIBILITY_OUTLINED, size=16, color=ACCENT),
                    ft.Text(block.title, size=13, weight=ft.FontWeight.BOLD)], spacing=8)]
    for element, meaning in block.items:
        rows.append(ft.Row([
            ft.Container(_inline(element, size=12.5, color=ft.Colors.ON_SURFACE), width=190),
            ft.Container(_inline(meaning, size=12.5, color=MUTED), expand=True),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START))
    return ft.Container(
        content=ft.Column(rows, spacing=8, tight=True),
        padding=14, border_radius=10,
        margin=ft.margin.symmetric(vertical=4),
        border=ft.border.all(1, ft.Colors.with_opacity(0.35, ACCENT)),
    )


def _callout_block(block: Callout) -> ft.Control:
    icon, color = CALLOUT_STYLE.get(block.kind, CALLOUT_STYLE[CALLOUT_TIP])
    inner = [ft.Row([ft.Icon(icon, size=16, color=color),
                     ft.Text(block.title, size=13, weight=ft.FontWeight.BOLD, color=color)], spacing=8)]
    inner += [_render_block(b) for b in block.blocks]
    return ft.Container(
        content=ft.Column(inner, spacing=8, tight=True),
        padding=14, border_radius=10,
        margin=ft.margin.symmetric(vertical=6),
        bgcolor=ft.Colors.with_opacity(0.07, color),
        border=ft.border.only(left=ft.BorderSide(3, color)),
    )


def _render_block(block) -> ft.Control:
    if isinstance(block, Para):
        return _inline(block.text)
    if isinstance(block, Heading):
        return ft.Container(
            ft.Text(block.text, size=15, weight=ft.FontWeight.BOLD, color=ACCENT),
            margin=ft.margin.only(top=14, bottom=2),
        )
    if isinstance(block, Equation):
        return _equation_block(block)
    if isinstance(block, Bullets):
        return _bullets_block(block)
    if isinstance(block, Table):
        return _table_block(block)
    if isinstance(block, HowToRead):
        return _howto_block(block)
    if isinstance(block, Callout):
        return _callout_block(block)
    return ft.Text(str(block), size=BODY_SIZE)  # unreachable; kept honest


# ---- the window ---------------------------------------------------------

class ManualView:
    """The manual window, built once and hung off the AppContext.

    Holds the reading position (`self.current`) and a back stack, so
    following a "see also" link and coming back lands where it started --
    a reader who follows three links out of the F1 page should not have to
    find it again in the table of contents.
    """

    MAX_W, MAX_H = 1240, 820
    TOC_W = 272
    WINDOW_MARGIN_W, WINDOW_MARGIN_H = 140, 140

    def __init__(self, ctx):
        self.ctx = ctx
        self.current: str = manual.DEFAULT_PAGE
        self._history: list = []
        self._toc_buttons: dict = {}

        self.search_field = ft.TextField(
            hint_text="Search the manual", dense=True, border_radius=8,
            prefix_icon=ft.Icons.SEARCH, text_size=12, on_change=self._on_search,
        )
        self.toc_column = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, expand=True)
        self.body_column = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
        self.back_button = ft.IconButton(
            ft.Icons.ARROW_BACK, tooltip="Back", icon_size=18, on_click=self._on_back, disabled=True,
        )
        self.title_text = ft.Text("", size=17, weight=ft.FontWeight.BOLD)

        self.body = ft.Container(content=ft.Row([
            ft.Container(
                content=ft.Column([self.search_field, self.toc_column], spacing=8, expand=True),
                width=self.TOC_W, padding=ft.padding.only(right=8),
            ),
            ft.VerticalDivider(width=1),
            ft.Container(content=self.body_column, expand=True, padding=ft.padding.only(left=14, right=6)),
        # STRETCH, not START: the table of contents and the page both
        # scroll, and a scrolling Column needs a bounded height. Left to
        # START the two Containers would be laid out against an unbounded
        # height, which Flutter answers with a render exception and an
        # empty dialog rather than a visible error.
        ], spacing=8, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH))

        self.dialog = ft.AlertDialog(
            # modal=False so a tap outside dismisses, plus on_dismiss
            # below: Flet does not reset `open` when a dialog is dismissed
            # that way, and a stale True makes the next page.open() do
            # nothing at all -- so the second (i) button a reader pressed
            # would appear dead (see ui/similarity.py, same trap).
            modal=False,
            on_dismiss=lambda e: self.ctx.page.close(self.dialog),
            title=ft.Row([
                self.back_button,
                ft.Icon(ft.Icons.MENU_BOOK_OUTLINED, size=18, color=ACCENT),
                self.title_text,
            ], spacing=8),
            content=self.body,
            actions=[ft.TextButton("Close", on_click=self.close)],
        )
        self._build_toc()

    # -- table of contents --

    def _build_toc(self, query: str = "") -> None:
        """Draw the contents, chapter by chapter.

        A chapter has to read as a chapter and not as one more entry in a
        list of sixty: sixty page titles in one column, separated only by a
        slightly smaller grey line, is a wall. So each chapter gets a
        banded header in the accent color with its page count, and its
        pages are indented under it behind a rule -- the indent is what
        makes the grouping visible while scrolling past, when the header
        itself has already gone off the top.
        """
        self.toc_column.controls.clear()
        self._toc_buttons.clear()
        for chapter in manual.CHAPTERS:
            pages = [manual.PAGES[pid] for pid in chapter.pages if manual.matches(manual.PAGES[pid], query)]
            if not pages:
                continue
            self.toc_column.controls.append(ft.Container(
                content=ft.Row([
                    ft.Text(chapter.title.upper(), size=11, weight=ft.FontWeight.BOLD,
                            color=ACCENT, no_wrap=False, expand=True),
                    ft.Text(str(len(pages)), size=10.5, color=MUTED),
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=ft.padding.symmetric(vertical=6, horizontal=10),
                margin=ft.margin.only(top=14, bottom=4),
                border_radius=6,
                bgcolor=ft.Colors.with_opacity(0.13, ACCENT),
            ))
            entries = []
            for page in pages:
                entry = ft.Container(
                    content=ft.Text(page.title, size=12.5, no_wrap=False),
                    padding=ft.padding.symmetric(vertical=6, horizontal=10),
                    border_radius=7, ink=True,
                    on_click=lambda e, pid=page.id: self.show(pid),
                )
                self._toc_buttons[page.id] = entry
                entries.append(entry)
            # The rule down the left is the chapter's spine: it says which
            # pages belong to the header above even once it has scrolled
            # out of view.
            self.toc_column.controls.append(ft.Container(
                content=ft.Column(entries, spacing=2, tight=True),
                margin=ft.margin.only(left=6),
                padding=ft.padding.only(left=6),
                border=ft.border.only(left=ft.BorderSide(
                    2, ft.Colors.with_opacity(0.22, ACCENT))),
            ))
        self._paint_selection()

    def _paint_selection(self) -> None:
        for page_id, entry in self._toc_buttons.items():
            selected = page_id == self.current
            entry.bgcolor = ft.Colors.with_opacity(0.18, ACCENT) if selected else None
            entry.content.color = ACCENT if selected else ft.Colors.ON_SURFACE
            entry.content.weight = ft.FontWeight.BOLD if selected else ft.FontWeight.NORMAL

    def _on_search(self, e) -> None:
        self._build_toc(self.search_field.value or "")
        if self.toc_column.page:
            self.toc_column.update()

    # -- navigation --

    def _on_back(self, e) -> None:
        if self._history:
            self.show(self._history.pop(), remember=False)

    def show(self, page_id: str, remember: bool = True) -> None:
        """Draw one page. An unknown id lands on the index rather than
        raising: an (i) button pointing at a page that was renamed should
        leave the reader in the manual, not in a stack trace."""
        page = manual.PAGES.get(page_id)
        if page is None:
            page = manual.PAGES[manual.DEFAULT_PAGE]
        if remember and page.id != self.current:
            self._history.append(self.current)
        self.current = page.id
        self.back_button.disabled = not self._history
        self.title_text.value = page.title
        self.body_column.controls = self._page_controls(page)
        self._paint_selection()
        if self.dialog.page is None:
            return  # first page, drawn before the dialog is opened
        self.ctx.page.update()
        # Back to the top: the reader followed a link, and landing halfway
        # down the new page because the old one was scrolled there is
        # disorienting. Guarded because scroll_to on a Column that has not
        # been laid out yet is not an error worth losing the page over.
        try:
            self.body_column.scroll_to(offset=0, duration=10)
        except Exception:
            pass

    def _page_controls(self, page) -> list:
        controls: list = []
        if page.shown_in:
            controls.append(ft.Row([
                ft.Icon(ft.Icons.WIDGETS_OUTLINED, size=14, color=MUTED),
                ft.Text(page.shown_in, size=11.5, color=MUTED, italic=True),
            ], spacing=6))
        controls.append(ft.Container(
            content=ft.Column([
                ft.Text("IN SHORT", size=10, weight=ft.FontWeight.BOLD, color=ACCENT),
                _inline(page.summary, size=14.5),
            ], spacing=6, tight=True),
            padding=14, border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.10, ACCENT),
            margin=ft.margin.only(top=4, bottom=6),
        ))
        controls += [_render_block(block) for block in page.blocks]

        links = [pid for pid in page.see_also if pid in manual.PAGES]
        if links:
            controls.append(ft.Container(
                ft.Column([
                    ft.Text("See also", size=12, weight=ft.FontWeight.BOLD, color=MUTED),
                    ft.Row([
                        ft.TextButton(manual.PAGES[pid].title, icon=ft.Icons.ARROW_FORWARD,
                                      on_click=lambda e, target=pid: self.show(target))
                        for pid in links
                    ], wrap=True, spacing=4, run_spacing=0),
                ], spacing=4, tight=True),
                margin=ft.margin.only(top=18, bottom=8),
            ))
        return controls

    # -- opening --

    def _fit_to_window(self) -> None:
        """Same reasoning as components.ImageZoomDialog: a fixed size
        overflows a smaller window and pushes the text out of view."""
        page = self.ctx.page
        available_w = getattr(page, "width", None) or self.MAX_W
        available_h = getattr(page, "height", None) or self.MAX_H
        self.body.width = max(560, min(self.MAX_W, int(available_w) - self.WINDOW_MARGIN_W))
        self.body.height = max(360, min(self.MAX_H, int(available_h) - self.WINDOW_MARGIN_H))

    def open(self, page_id: str | None = None) -> None:
        self._history.clear()
        self.show(page_id or self.current, remember=False)
        self._fit_to_window()
        self.ctx.page.open(self.dialog)

    def close(self, e=None) -> None:
        self.ctx.page.close(self.dialog)
