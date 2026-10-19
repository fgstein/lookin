"""The reference manual behind every (i) button: a table of contents, a
search, and one typeset page at a time, with a back stack so following
"see also" links and coming back lands where the reader started.

Page content is data in core/manual/; formulas are typeset by
ui/mathtext.render_math (matplotlib, so nothing is fetched) and embedded as
PNG data URIs. Same structure as lookin.ui.manual_view.
"""
from __future__ import annotations

import re

from lookin.core import manual
from lookin.core.manual.model import (
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
    Bullets, Callout, Equation, Heading, HowToRead, Para, Table,
)
from lookin.ui.mathtext import DISPLAY_SIZE, MathRenderError, render_math
from lookin.web import html as h
from lookin.web.views.base import Dialog

INK = "#15181E"
INK_MUTED = "#545B69"
BODY_SIZE = 14
INLINE_LIFT = 1.5
_SPAN = re.compile(r"\$([^$]*)\$|\*([^*]*)\*")
CALLOUT_STYLE = {
    CALLOUT_PITFALL: ("warn", "var(--warn-dot)"),
    CALLOUT_WARNING: ("warn", "var(--err)"),
    CALLOUT_TIP: ("info", "var(--ok)"),
    CALLOUT_EXAMPLE: ("target", "var(--acc)"),
}


def math_img(tex: str, size: float, color: str = INK) -> str:
    """One formula as an image, or its source in red if it will not typeset."""
    try:
        data, width, height = render_math(tex, size, color)
    except MathRenderError:
        return f'<code style="color:var(--err)">{h.e(tex)}</code>'
    return (f'<img class="math" src="data:image/png;base64,{data}" width="{width:.0f}" height="{height:.0f}" '
            f'alt="{h.e(tex)}" style="vertical-align:middle">')


def inline(text: str, size: float = BODY_SIZE, color: str = INK) -> str:
    out = []
    pos = 0
    for m in _SPAN.finditer(text):
        out.append(h.e(text[pos:m.start()]))
        formula, emphasis = m.group(1), m.group(2)
        if formula:
            out.append(f'<span class="nowrap">{math_img(formula, size + INLINE_LIFT, color)}</span>')
        elif emphasis:
            out.append(f"<em style=\"font-weight:500\">{h.e(emphasis)}</em>")
        pos = m.end()
    out.append(h.e(text[pos:]))
    return "".join(out)


def render_block(block) -> str:
    if isinstance(block, Para):
        return f"<p>{inline(block.text)}</p>"
    if isinstance(block, Heading):
        return f'<h4 style="color:var(--acc)">{h.e(block.text)}</h4>'
    if isinstance(block, Equation):
        rows = [f'<div style="overflow-x:auto;text-align:center">{math_img(block.tex, DISPLAY_SIZE)}</div>']
        if block.note:
            rows.append(f'<div class="muted" style="font-style:italic;font-size:12px">{inline(block.note, 12, INK_MUTED)}</div>')
        for symbol, meaning in block.where:
            rows.append(f'<div class="row-nw" style="gap:10px;margin-top:4px"><span style="width:64px;text-align:right">'
                        f'{math_img(symbol, BODY_SIZE + INLINE_LIFT, INK_MUTED)}</span>'
                        f'<span class="muted" style="font-size:12.5px">{inline(meaning, 12.5, INK_MUTED)}</span></div>')
        return f'<div class="eq" style="padding:12px 14px;border-radius:10px;background:var(--surface-3)">{"".join(rows)}</div>'
    if isinstance(block, Bullets):
        tag = "ol" if block.numbered else "ul"
        return f"<{tag}>" + "".join(f"<li>{inline(item)}</li>" for item in block.items) + f"</{tag}>"
    if isinstance(block, Table):
        weights = block.weights or tuple(1 for _ in block.headers)
        total = sum(weights) or 1
        head = "".join(f'<th style="width:{100 * w / total:.0f}%">{h.e(x)}</th>' for x, w in zip(block.headers, weights))
        body = "".join("<tr>" + "".join(f"<td>{inline(str(c), 12.5)}</td>" for c in row) + "</tr>" for row in block.rows)
        cap = f'<div class="muted small" style="font-style:italic;margin-top:4px">{inline(block.caption, 11.5, INK_MUTED)}</div>' if block.caption else ""
        return f'<div style="margin:6px 0"><table class="tbl" style="font-size:12.5px"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{cap}</div>'
    if isinstance(block, HowToRead):
        rows = "".join(f'<div class="row-nw" style="align-items:flex-start;gap:12px;margin-top:6px"><span style="width:190px;flex:none">{inline(el, 12.5)}</span>'
                       f'<span class="muted" style="font-size:12.5px">{inline(meaning, 12.5, INK_MUTED)}</span></div>' for el, meaning in block.items)
        return (f'<div style="padding:12px 14px;border-radius:10px;border:1px solid rgba(37,80,200,.35);margin:6px 0">'
                f'<div class="row-nw">{h.icon("eye", 15, "c-acc")}<b>{h.e(block.title)}</b></div>{rows}</div>')
    if isinstance(block, Callout):
        ic, color = CALLOUT_STYLE.get(block.kind, CALLOUT_STYLE[CALLOUT_TIP])
        inner = "".join(render_block(b) for b in block.blocks)
        return (f'<div style="padding:12px 14px;border-radius:10px;margin:8px 0;border-left:3px solid {color};background:var(--surface-2)">'
                f'<div class="row-nw" style="color:{color};font-weight:700;margin-bottom:6px">{h.icon(ic, 15)}{h.e(block.title)}</div>{inner}</div>')
    return f"<p>{h.e(str(block))}</p>"


class ManualDialog(Dialog):
    width = 1240

    def __init__(self):
        super().__init__("Reference manual")
        self.current = manual.DEFAULT_PAGE
        self.history: list = []
        self.query = ""

    def show(self, page_id, remember: bool = True) -> None:
        page = manual.PAGES.get(page_id) or manual.PAGES[manual.DEFAULT_PAGE]
        if remember and page.id != self.current:
            self.history.append(self.current)
        self.current = page.id

    def title_html(self) -> str:
        back = h.button(icon_name="arrow-left", act_attr=self.a("back"), small=True, kind="ghost", title="Back",
                        disabled=not self.history)
        page = manual.PAGES.get(self.current) or manual.PAGES[manual.DEFAULT_PAGE]
        return f'{back}{h.icon("book", 17, "c-acc")}<span>{h.e(page.title)}</span>'

    def render(self) -> str:
        return super().render().replace('class="dlg"', 'class="dlg dlg-big"', 1).replace(
            'class="dlg-b"', 'class="dlg-b flush"', 1)

    def body(self) -> str:
        toc = [f'<div style="padding:4px 10px 8px">{h.text_input(self.query, input_=self.i("search"), type_="search", placeholder="Search the manual", width="100%")}</div>']
        for chapter in manual.CHAPTERS:
            pages = [manual.PAGES[pid] for pid in chapter.pages if manual.matches(manual.PAGES[pid], self.query)]
            if not pages:
                continue
            toc.append(f'<div class="facet-h" style="color:var(--acc)">{h.e(chapter.title)} <span class="muted small">{len(pages)}</span></div>')
            for p in pages:
                on = p.id == self.current
                toc.append(f'<button type="button" class="facet-b{" on" if on else ""}" style="padding-left:18px;min-height:26px"'
                           f'{self.a("show", page=p.id)}>{h.e(p.title)}</button>')
        page = manual.PAGES.get(self.current) or manual.PAGES[manual.DEFAULT_PAGE]
        parts = []
        if page.shown_in:
            parts.append(f'<div class="muted small" style="font-style:italic;margin-bottom:6px">{h.icon("grid", 13)} {h.e(page.shown_in)}</div>')
        parts.append(f'<div style="padding:12px 14px;border-radius:10px;background:var(--acc-soft);margin-bottom:10px">'
                     f'<div style="font-size:10px;font-weight:700;color:var(--acc)">IN SHORT</div>{inline(page.summary, 14.5)}</div>')
        parts += [render_block(b) for b in page.blocks]
        links = [pid for pid in page.see_also if pid in manual.PAGES]
        if links:
            parts.append('<div class="subhead" style="margin-top:18px">See also</div><div class="row">'
                         + "".join(h.button(manual.PAGES[pid].title, act_attr=self.a("show", page=pid), kind="link",
                                            icon_name="arrow-right", small=True) for pid in links) + "</div>")
        return (f'<div class="manual"><div class="manual-toc">{"".join(toc)}</div>'
                f'<div class="manual-page" data-key="mp-{h.e(page.id)}">{"".join(parts)}</div></div>')

    def on_show(self, page: str = "", **_):
        self.show(page)
        self.refresh()
        self.session.cmd(event="manual-top")

    def on_back(self, **_):
        if self.history:
            self.show(self.history.pop(), remember=False)
        self.refresh()

    def on_search(self, value: str = "", **_):
        self.query = value or ""
        self.refresh()


def open_manual(session, page_id=None) -> None:
    dialog = getattr(session, "_manual", None)
    if dialog is None:
        dialog = session._manual = ManualDialog()
    dialog.history.clear()
    dialog.show(page_id or dialog.current, remember=False)
    session.open_dialog(dialog)
