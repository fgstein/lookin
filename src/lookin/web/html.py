"""Building HTML on the server: escaping, attributes, and the handful of
widgets every view is made of.

Views return HTML strings. Interactivity is declared with data- attributes
that static/app.js turns into requests (see `act`, `on_change`):

    <button data-act="metrics:refresh">        click   -> POST /api/act
    <select data-change="metrics:set_metric">  change  -> {value}
    <input  data-input="home:search">          typing  -> {value}, debounced
    <form   data-submit="home:create">         submit  -> every named field

`target` is the session object that handles it ("metrics", "home",
"dlg:<id>", "app"); `action` the name of its `on_<action>` method.

Every text that reaches the page goes through `e()`. Class labels and
column names come from the user's own files and are shown verbatim, so
they are escaped, never normalized.
"""
from __future__ import annotations

import json
from html import escape as _escape
from typing import Iterable, Optional


def e(value) -> str:
    """Escape for text and attribute values. None renders as nothing."""
    if value is None:
        return ""
    return _escape(str(value), quote=True)


def attrs(**kw) -> str:
    """Attributes from keywords: `class_` -> class, `data_x` -> data-x,
    True -> bare attribute, False/None -> omitted."""
    out = []
    for key, value in kw.items():
        if value is None or value is False:
            continue
        name = key.rstrip("_").replace("_", "-")
        if value is True:
            out.append(name)
        else:
            out.append(f'{name}="{e(value)}"')
    return (" " + " ".join(out)) if out else ""


def tag(name: str, *children, **kw) -> str:
    return f"<{name}{attrs(**kw)}>{''.join(c for c in children if c)}</{name}>"


def _args(args: dict) -> str:
    return f" data-args='{e(json.dumps(args, separators=(',', ':')))}'" if args else ""


def data_args(**args) -> str:
    """The data-args attribute alone, for an element that is not itself a
    data-act (a drop target): JSON built by hand breaks on a quote."""
    return _args(args)


def safe_href(url) -> str:
    """`url` if a link may point there: http(s), or a path on this server.
    A URL read from a file (an MLflow server in a workspace someone shared,
    the Home settings) could otherwise be `javascript:...`, which a click
    runs inside Lookin's page."""
    text = str(url or "").strip()
    scheme = text.split(":", 1)[0].lower() if ":" in text.split("/", 1)[0] else ""
    if scheme in ("http", "https") or (not scheme and text.startswith("/") and not text.startswith("//")):
        return text
    return ""


def act(target: str, action: str, **args) -> str:
    """Attributes for a click that runs `target.on_<action>(**args)`."""
    return f' data-act="{e(target)}:{e(action)}"{_args(args)}'


def on_change(target: str, action: str, **args) -> str:
    return f' data-change="{e(target)}:{e(action)}"{_args(args)}'


def on_input(target: str, action: str, **args) -> str:
    """Debounced as the user types (see app.js)."""
    return f' data-input="{e(target)}:{e(action)}"{_args(args)}'


def on_enter(target: str, action: str, **args) -> str:
    return f' data-enter="{e(target)}:{e(action)}"{_args(args)}'


def on_submit(target: str, action: str, **args) -> str:
    return f' data-submit="{e(target)}:{e(action)}"{_args(args)}'


# ---- icons ---------------------------------------------------------------
# Stroke icons drawn inline, so nothing is fetched and they take the text
# colour. 24x24 viewBox, paths only.

_ICONS = {
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "eye-off": '<path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.4 0 10 7 10 7a17 17 0 0 1-3.2 4.2M6.6 6.6C3.8 8.4 2 12 2 12s3.6 7 10 7a9.7 9.7 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "chev-right": '<path d="M9 6l6 6-6 6"/>',
    "chev-down": '<path d="M6 9l6 6 6-6"/>',
    "chev-left": '<path d="M15 6l-6 6 6 6"/>',
    "chev-up": '<path d="M6 15l6-6 6 6"/>',
    "refresh": '<path d="M20 11a8 8 0 0 0-14.6-4.5L4 8M4 4v4h4M4 13a8 8 0 0 0 14.6 4.5L20 16M20 20v-4h-4"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
    "external": '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
    "dots": '<circle cx="5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="19" cy="12" r="1.3"/>',
    "dots-v": '<circle cx="12" cy="5" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="12" cy="19" r="1.3"/>',
    "download": '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
    "upload": '<path d="M12 20V9M7 14l5-5 5 5M5 4h14"/>',
    "folder": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z"/>',
    "file": '<path d="M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7z"/><path d="M14 3v4h4"/>',
    "table": '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><path d="M3.5 9.5h17M3.5 14.5h17M9.5 9.5v10"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14zM20 17v4H6.5"/>',
    "home": '<path d="M4 11l8-7 8 7v8.5a.5.5 0 0 1-.5.5H15v-6H9v6H4.5a.5.5 0 0 1-.5-.5z"/>',
    "trash": '<path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 13h9l1-13"/>',
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16z"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="1.5"/><path d="M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8"/>',
    "link": '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
    "unlink": '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7M3 3l18 18"/>',
    "warn": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.2v.3"/>',
    "info": '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.3"/>',
    "grid": '<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/><rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/>',
    "columns": '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><path d="M9.5 4.5v15M14.5 4.5v15"/>',
    "code": '<path d="M8 7l-5 5 5 5M16 7l5 5-5 5"/>',
    "bookmark": '<path d="M6.5 4h11v16l-5.5-4-5.5 4z"/>',
    "star": '<path d="M12 4l2.4 5 5.6.6-4.2 3.8 1.2 5.6L12 16.2 7 19l1.2-5.6L4 9.6 9.6 9z"/>',
    "save": '<path d="M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
    "filter": '<path d="M4 5h16l-6 7.5V19l-4-2v-4.5z"/>',
    "image": '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><circle cx="9" cy="10" r="1.8"/><path d="M20.5 16l-5-5-8 8.5"/>',
    "zoom": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5M11 8v6M8 11h6"/>',
    "lasso": '<ellipse cx="12" cy="10" rx="8.5" ry="5.5"/><path d="M6.5 14.5c-1.2 3 .8 6 3.8 5"/>',
    "pan": '<path d="M12 3v18M3 12h18M12 3l-2.5 2.5M12 3l2.5 2.5M12 21l-2.5-2.5M12 21l2.5-2.5M3 12l2.5-2.5M3 12l2.5 2.5M21 12l-2.5-2.5M21 12l-2.5 2.5"/>',
    "drag": '<path d="M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01" stroke-width="3"/>',
    "cloud": '<path d="M7 18a4.5 4.5 0 0 1-.5-9 6 6 0 0 1 11.5 1.5A3.75 3.75 0 0 1 17.5 18z"/>',
    "cloud-off": '<path d="M7 18a4.5 4.5 0 0 1-.5-9M10 5.4A6 6 0 0 1 18 10.5 3.75 3.75 0 0 1 19.6 17M3 3l18 18"/>',
    "play": '<path d="M7 5l12 7-12 7z"/>',
    "list": '<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "rotate": '<path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5"/>',
    "shuffle": '<path d="M4 7h3.5c4 0 5 10 9 10H20M4 17h3.5c1.5 0 2.5-1.4 3.3-3M14 7h6M17 4l3 3-3 3M17 14l3 3-3 3"/>',
    "arrow-left": '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "arrows-ud": '<path d="M8 4v16M4 8l4-4 4 4M16 20V4M12 16l4 4 4-4"/>',
    "broom": '<path d="M15 4l5 5M13.5 5.5l5 5L11 18l-5-5z"/><path d="M6 13l-3 7 7-3"/>',
    "layers": '<path d="M12 4l9 5-9 5-9-5z"/><path d="M3 14l9 5 9-5"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r=".8"/>',
    "chart": '<path d="M4 20V4M4 20h16M8 16v-5M12 16V8M16 16v-3"/>',
    "send": '<path d="M4 12l16-8-6 16-2.5-6.5z"/>',
}


def icon(name: str, size: int = 16, cls: str = "") -> str:
    path = _ICONS.get(name, _ICONS["info"])
    return (f'<svg class="ic {e(cls)}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{path}</svg>')


# ---- widgets -------------------------------------------------------------

def button(label: str = "", *, act_attr: str = "", kind: str = "", icon_name: str = "",
           title: str = "", disabled: bool = False, small: bool = False, cls: str = "",
           aria_label: str = "", type_: str = "button", extra: str = "") -> str:
    """kind: "" (outlined), "primary", "danger", "ghost", "link"."""
    classes = " ".join(c for c in ("btn", f"btn-{kind}" if kind else "", "btn-sm" if small else "",
                                   "btn-icon" if (icon_name and not label) else "", cls) if c)
    inner = (icon(icon_name, 14 if small else 15) if icon_name else "") + (f"<span>{e(label)}</span>" if label else "")
    return (f'<button type="{type_}" class="{classes}"{act_attr}'
            f'{attrs(title=title or None, disabled=disabled, aria_label=aria_label or (title if not label else None))}'
            f'{extra}>{inner}</button>')


def link_button(label: str, href: str, *, kind: str = "", icon_name: str = "", title: str = "",
                new_tab: bool = True, small: bool = False) -> str:
    href = safe_href(href)
    if not href:
        return ""
    classes = " ".join(c for c in ("btn", f"btn-{kind}" if kind else "", "btn-sm" if small else "",
                                   "btn-icon" if (icon_name and not label) else "") if c)
    inner = (icon(icon_name, 14) if icon_name else "") + (f"<span>{e(label)}</span>" if label else "")
    return (f'<a class="{classes}" href="{e(href)}"{attrs(title=title or None, aria_label=title if not label else None)}'
            f'{" target=\"_blank\" rel=\"noopener\"" if new_tab else ""}>{inner}</a>')


def info_btn(page_id: Optional[str], title: str = "Open the reference manual") -> str:
    """The (i) that opens the manual at `page_id` -- a key, never shown."""
    if not page_id:
        return ""
    return (f'<button type="button" class="info-btn"{act("app", "manual", page=page_id)} '
            f'title="{e(title)}" aria-label="{e(title)}">i</button>')


# Past this many characters a select's chosen label may be cut by the
# select's width cap (see .sel in app.css), so it gets a tooltip.
LONG_LABEL = 30


def select(options: Iterable, value, *, change: str = "", name: str = "", label: str = "",
           cls: str = "", disabled: bool = False, title: str = "", id_: str = "") -> str:
    """options: (key, label) pairs. The key is what code compares; only the
    label is display text (see CLAUDE.md)."""
    opts = []
    chosen = ""
    for opt in options:
        key, text = (opt if isinstance(opt, (tuple, list)) else (opt, opt))[:2]
        sel = " selected" if str(key) == str(value) else ""
        if sel:
            chosen = str(text)
        opts.append(f'<option value="{e(key)}"{sel}>{e(text)}</option>')
    if not title and len(chosen) > LONG_LABEL:
        title = chosen
    sel_attrs = attrs(name=name or None, disabled=disabled, title=title or None, id=id_ or None,
                      aria_label=label or None)
    sel_html = f'<select class="sel {e(cls)}"{change}{sel_attrs}>{"".join(opts)}</select>'
    if label:
        return f'<label class="fld"><span class="fld-l">{e(label)}</span>{sel_html}</label>'
    return sel_html


def checkbox(label: str, checked: bool, *, change: str = "", name: str = "", title: str = "",
             disabled: bool = False, cls: str = "", ellipsis: bool = False) -> str:
    """ellipsis: the label is a name of any length (an experiment): kept on
    one line, cut with an ellipsis, the full text in the tooltip."""
    if ellipsis:
        cls = f"{cls} ell".strip()
        title = title or label
    return (f'<label class="chk {e(cls)}"{attrs(title=title or None)}>'
            f'<input type="checkbox"{change}{attrs(name=name or None, checked=bool(checked), disabled=disabled)}>'
            f'<span>{e(label)}</span></label>')


def radio(name: str, options: Iterable, value, *, change: str = "") -> str:
    out = []
    for key, text in options:
        out.append(f'<label class="chk"><input type="radio" name="{e(name)}" value="{e(key)}"{change}'
                   f'{" checked" if str(key) == str(value) else ""}><span>{e(text)}</span></label>')
    return "".join(out)


def text_input(value="", *, name: str = "", placeholder: str = "", input_: str = "", enter: str = "",
               change: str = "", label: str = "", cls: str = "", type_: str = "text", width: str = "",
               title: str = "", autofocus: bool = False, mono: bool = False, id_: str = "",
               disabled: bool = False, extra: str = "") -> str:
    if isinstance(width, (int, float)) and not isinstance(width, bool):
        width = f"{int(width)}px"       # a number is pixels
    style = f' style="width:{e(width)}"' if width else ""
    classes = "inp" + (" mono" if mono else "") + (f" {cls}" if cls else "")
    field = (f'<input type="{type_}" class="{classes}"{input_}{enter}{change}{style}{extra}'
             f'{attrs(name=name or None, value=value if value is not None else "", placeholder=placeholder or None, title=title or None, autofocus=autofocus, id=id_ or None, disabled=disabled, aria_label=(label or placeholder) or None)}>')
    if label:
        return f'<label class="fld"><span class="fld-l">{e(label)}</span>{field}</label>'
    return field


def textarea(value="", *, name: str = "", rows: int = 10, cls: str = "", mono: bool = True) -> str:
    return (f'<textarea class="inp ta{" mono" if mono else ""} {e(cls)}" rows="{rows}"'
            f'{attrs(name=name or None)} spellcheck="false">{e(value)}</textarea>')


def banner(text: str, kind: str = "info", *, icon_name: str = "", actions: str = "", lines=None) -> str:
    """kind: info | warn | error | ok | stale"""
    ic = icon_name or {"warn": "warn", "error": "warn", "ok": "check", "stale": "warn"}.get(kind, "info")
    body = f"<div>{e(text)}</div>" if text else ""
    if lines:
        body += "".join(f"<div>{e(line)}</div>" for line in lines)
    return (f'<div class="banner banner-{kind}">{icon(ic, 15)}<div class="banner-body">{body}</div>'
            f'{f"<div class=banner-actions>{actions}</div>" if actions else ""}</div>')


def status_line(text: str, kind: str = "ok") -> str:
    if not text:
        return ""
    return f'<div class="status status-{kind}">{e(text)}</div>'


def name(text, cls: str = "") -> str:
    """A name the user chose -- an experiment, a run -- which can be any
    length: one line, cut with an ellipsis, the full name on hover."""
    return f'<span class="xname {e(cls)}" title="{e(text)}">{e(text)}</span>'


def dot(color: str, title: str = "", size: int = 7) -> str:
    return (f'<span class="dot" style="background:{e(color)};width:{size}px;height:{size}px"'
            f'{attrs(title=title or None)}></span>')


def section(title: str, body: str, *, manual=None, right: str = "", id_: str = "", sub: str = "",
            cls: str = "", collapsed: Optional[bool] = None, toggle: str = "") -> str:
    """A card with a bold title row, an optional (i), a one-line subtitle
    and controls on the right. `manual` is a page id (see CLAUDE.md)."""
    chev = ""
    if collapsed is not None:
        chev = (f'<button type="button" class="chev"{toggle} aria-expanded="{"false" if collapsed else "true"}">'
                f'{icon("chev-right" if collapsed else "chev-down", 14)}</button>')
    head = (f'<div class="card-h">{chev}<h2>{e(title)}</h2>{info_btn(manual)}'
            f'{f"<span class=card-sub>{e(sub)}</span>" if sub else ""}'
            f'<div class="card-r">{right}</div></div>')
    content = "" if collapsed else f'<div class="card-b">{body}</div>'
    return f'<section class="card {e(cls)}"{attrs(id=id_ or None)}>{head}{content}</section>'


def kv_grid(rows) -> str:
    return '<div class="kv">' + "".join(f'<span class="kv-k">{e(k)}</span><span class="kv-v">{v}</span>'
                                       for k, v in rows) + "</div>"


def chip(text: str, *, color: str = "", act_attr: str = "", title: str = "", selected: bool = False,
         cls: str = "") -> str:
    dot_html = dot(color) if color else ""
    tagname = "button" if act_attr else "span"
    type_attr = ' type="button"' if act_attr else ""
    return (f'<{tagname}{type_attr} class="chip{" on" if selected else ""} {e(cls)}"{act_attr}'
            f'{attrs(title=title or None)}>{dot_html}{e(text)}</{tagname}>')


def segmented(options, value, *, act_for, label: str = "") -> str:
    """A row of toggle buttons. `act_for(key)` returns the act attribute."""
    btns = "".join(
        f'<button type="button" class="seg-b{" on" if str(k) == str(value) else ""}"'
        f' aria-pressed="{"true" if str(k) == str(value) else "false"}"{act_for(k)}>{e(t)}</button>'
        for k, t in options)
    return f'<div class="seg" role="group"{attrs(aria_label=label or None)}>{btns}</div>'


def progress(text: str = "", fraction: Optional[float] = None) -> str:
    bar = ('<div class="pbar"><div style="width:%.1f%%"></div></div>' % (100 * max(0.0, min(1.0, fraction)))
           if fraction is not None else '<div class="pbar pbar-ind"><div></div></div>')
    return f'<div class="progress">{bar}<span>{e(text)}</span></div>'


def spinner(text: str = "") -> str:
    return f'<div class="spin-row"><span class="spinner"></span><span>{e(text)}</span></div>'


def empty(text: str) -> str:
    return f'<div class="empty">{e(text)}</div>'


def mono(text) -> str:
    return f'<span class="mono">{e(text)}</span>'


def human_size(n) -> str:
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""
