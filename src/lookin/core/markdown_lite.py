"""The little Markdown the Notes tab writes, both ways.

The Notes tab is edited as formatted text (a contenteditable in the page)
and kept as Markdown (core/notes.py), which is what a person reads in a
text editor, an agent reads over MCP and the Markdown report carries as is.
So the page needs Markdown turned into HTML to show it, and the HTML the
browser's editor produces turned back into Markdown to keep it.

Only what a short note uses, the way a chat message is formatted: headings,
paragraphs, bold, italic, strikethrough, inline code, links, bullet and
numbered lists (nested), quotes, a rule, code blocks -- and a *region*, a
stretch the app fills in itself and may fill again, written
``<!-- lookin:NAME -->`` ... ``<!-- /lookin:NAME -->`` in the Markdown and
``<div data-region="NAME">`` in the HTML.

No dependency on purpose (see CLAUDE.md): a Markdown library is a package
to vendor for a few dozen lines of rules. What is not listed above is kept
as its text: a table pasted from a spreadsheet comes back as its cells'
text, never as markup that was not checked.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

# Where a link may point: anything else (javascript:, data:) is kept as
# the text it was attached to.
SAFE_LINK = re.compile(r"^(https?://|mailto:|#|/)", re.IGNORECASE)
REGION_OPEN = re.compile(r"^<!--\s*lookin:([A-Za-z0-9_-]+)\s*-->$")
REGION_CLOSE = re.compile(r"^<!--\s*/lookin:([A-Za-z0-9_-]+)\s*-->$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$")
RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
FENCE = re.compile(r"^\s*```")
# Characters that would start formatting if left bare in a text node. An
# underscore inside a word (snake_case, a run name) cannot, so it stays.
_ESCAPE = re.compile(r"([\\`*\[\]~]|(?<!\w)_|_(?!\w))")
_LINE_START = re.compile(r"^(\s*)(#{1,6}\s|>|[-+*]\s|\d{1,9}[.)]\s)")


def region_open(name: str) -> str:
    return f"<!-- lookin:{name} -->"


def region_close(name: str) -> str:
    return f"<!-- /lookin:{name} -->"


# ---- Markdown -> HTML --------------------------------------------------------

def _safe_href(url: str) -> str:
    url = url.strip()
    return url if SAFE_LINK.match(url) else ""


def inline_html(text: str) -> str:
    """One line of Markdown as HTML: escaped, then the inline formatting."""
    stash: list = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    # Backslash escapes and code spans first: nothing inside them is markup.
    text = re.sub(r"\\([\\`*_\[\]~#>+\-.!()])", lambda m: keep(html.escape(m.group(1))), text)
    text = re.sub(r"`([^`]+)`", lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text)

    def link(m):
        href = _safe_href(m.group(2))
        label = inline_html(m.group(1))
        return keep(f'<a href="{html.escape(href)}">{label}</a>' if href else label)

    text = re.sub(r"\[([^\]]*)\]\(([^)\s]*)\)", link, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<![_\w])_(?!\s)(.+?)(?<!\s)_(?![_\w])", r"<em>\1</em>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def _paragraph(lines: list) -> str:
    return "<p>" + "<br>".join(inline_html(line.strip()) for line in lines) + "</p>"


def _list_html(items: list) -> str:
    """items: (level, ordered, text) in document order -> nested lists."""
    out, stack = [], []          # stack of (level, tag)
    for level, ordered, text in items:
        tag = "ol" if ordered else "ul"
        while stack and stack[-1][0] > level:
            out.append(f"</li></{stack.pop()[1]}>")
        if stack and stack[-1][0] == level and stack[-1][1] != tag:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or stack[-1][0] < level:
            out.append(f"<{tag}><li>")
            stack.append((level, tag))
        else:
            out.append("</li><li>")
        out.append(inline_html(text))
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def to_html(text: str) -> str:
    """Markdown (the subset above) as HTML."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list = []
    para: list = []
    items: list = []
    quote: list = []
    i = 0

    def flush():
        if para:
            out.append(_paragraph(para))
            para.clear()
        if items:
            out.append(_list_html(items))
            items.clear()
        if quote:
            out.append(f"<blockquote>{to_html(chr(10).join(quote))}</blockquote>")
            quote.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if FENCE.match(line):
            flush()
            code = []
            i += 1
            while i < len(lines) and not FENCE.match(lines[i]):
                code.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            i += 1
            continue
        opened, closed = REGION_OPEN.match(stripped), REGION_CLOSE.match(stripped)
        if opened or closed:
            flush()
            out.append(f'<div data-region="{html.escape(opened.group(1))}">' if opened else "</div>")
            i += 1
            continue
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith(">"):
            if para or items:
                flush()
            quote.append(re.sub(r"^\s*>\s?", "", line))
            i += 1
            continue
        if quote:
            flush()
        heading = HEADING.match(stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline_html(heading.group(2))}</h{level}>")
        elif RULE.match(line):
            flush()
            out.append("<hr>")
        elif LIST_ITEM.match(line) and not para:
            indent, marker, body = LIST_ITEM.match(line).groups()
            items.append((len(indent.replace("\t", "  ")) // 2, marker[-1] in ".)", body))
        elif items and line.startswith("  "):
            level, ordered, body = items[-1]
            items[-1] = (level, ordered, f"{body} {stripped}")
        else:
            if items:
                flush()
            para.append(line)
        i += 1
    flush()
    # Regions left open by a hand edit are closed rather than swallowing
    # the rest of the page.
    depth = sum(part.count('<div data-region=') for part in out) - out.count("</div>")
    out.extend(["</div>"] * max(0, depth))
    return "\n".join(out)


# ---- HTML -> Markdown --------------------------------------------------------

_BLOCKS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "hr",
           "section", "article", "header", "footer", "table", "tr", "br"}
_VOID = {"br", "hr", "img", "input", "meta", "link", "col", "area", "base", "wbr", "source"}
_DROP = {"script", "style", "head", "title", "template", "noscript", "button", "select", "textarea"}


class _Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str, attrs=None):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.children: list = []


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root")
        self.stack = [self.root]
        self.dropping = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP:
            self.dropping += 1
            return
        if self.dropping:
            return
        node = _Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        if tag in _DROP:
            self.dropping = max(0, self.dropping - 1)
            return
        if self.dropping or tag in _VOID:
            return
        for k in range(len(self.stack) - 1, 0, -1):
            if self.stack[k].tag == tag:
                del self.stack[k:]
                return

    def handle_data(self, data):
        if not self.dropping:
            self.stack[-1].children.append(data)


def escape(text) -> str:
    """`text` as Markdown that reads as exactly that text (a run name
    with an asterisk in it, say)."""
    return _ESCAPE.sub(r"\\\1", str(text))


_escape_text = escape


def _is_bold(node) -> bool:
    style = node.attrs.get("style", "") or ""
    return node.tag in ("strong", "b") or bool(re.search(r"font-weight:\s*(bold|[6-9]00)", style))


def _is_italic(node) -> bool:
    style = node.attrs.get("style", "") or ""
    return node.tag in ("em", "i") or "font-style: italic" in style or "font-style:italic" in style


def _inline(nodes: list) -> str:
    out = []
    for node in nodes:
        if isinstance(node, str):
            out.append(_escape_text(re.sub(r"[ \t\r\n\xa0]+", " ", node)))
            continue
        if node.tag == "br":
            out.append("\n")
            continue
        if node.tag in _BLOCKS:
            # A block inside a line (a <div> Chrome put in a list item):
            # its own line.
            inner = _inline(node.children).strip(" ")
            if inner:
                out.append("\n" + inner + "\n")
            continue
        inner = _inline(node.children)
        if not inner.strip():
            out.append(inner)
            continue
        lead = inner[: len(inner) - len(inner.lstrip(" "))]
        trail = inner[len(inner.rstrip(" ")):]
        core = inner.strip(" ")
        if node.tag == "code":
            text = "".join(c if isinstance(c, str) else _plain(c) for c in node.children)
            core = f"`{text.strip()}`"
        elif node.tag == "a":
            href = _safe_href(node.attrs.get("href", "") or "")
            core = f"[{core}]({href})" if href else core
        else:
            if node.tag in ("s", "strike", "del"):
                core = f"~~{core}~~"
            if _is_italic(node):
                core = f"*{core}*"
            if _is_bold(node):
                core = f"**{core}**"
        out.append(lead + core + trail)
    return "".join(out)


def _plain(node) -> str:
    if node.tag == "br":
        return "\n"
    return "".join(c if isinstance(c, str) else _plain(c) for c in node.children)


def _line_start_safe(text: str) -> str:
    """A paragraph line that would read as a heading or a list item is
    escaped where it starts."""
    return "\n".join(_LINE_START.sub(lambda m: m.group(1) + "\\" + m.group(2), line)
                     for line in text.split("\n"))


def _tidy(text: str) -> str:
    lines = [line.rstrip() for line in text.strip("\n").split("\n")]
    return "\n".join(lines).strip()


def _blocks(nodes: list, out: list) -> None:
    """Append each block of `nodes` to `out` as Markdown text."""
    run: list = []

    def flush():
        text = _tidy(_inline(run))
        run.clear()
        if text:
            out.append(_line_start_safe(text))

    for node in nodes:
        if isinstance(node, str) or node.tag not in _BLOCKS or node.tag == "br":
            run.append(node)
            continue
        flush()
        tag = node.tag
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = _tidy(_inline(node.children)).replace("\n", " ")
            if text:
                out.append("#" * int(tag[1]) + " " + text)
        elif tag in ("ul", "ol"):
            lines = _list_lines(node, 0)
            if lines:
                out.append("\n".join(lines))
        elif tag == "blockquote":
            inner: list = []
            _blocks(node.children, inner)
            if inner:
                out.append("\n".join("> " + line if line else ">" for line in "\n\n".join(inner).split("\n")))
        elif tag == "pre":
            out.append("```\n" + _plain(node).strip("\n") + "\n```")
        elif tag == "hr":
            out.append("---")
        elif tag in ("div", "section") and node.attrs.get("data-region"):
            name = re.sub(r"[^A-Za-z0-9_-]", "", node.attrs["data-region"])
            inner = []
            _blocks(node.children, inner)
            out.append(region_open(name))
            out.extend(inner)
            out.append(region_close(name))
        elif tag == "li":
            lines = _list_lines(_Node("ul", {}), 0, [node])
            out.append("\n".join(lines))
        elif tag == "tr":
            cells = [_tidy(_inline(c.children)) for c in node.children if not isinstance(c, str)]
            text = " · ".join(c for c in cells if c)
            if text:
                out.append(_line_start_safe(text))
        else:
            # p, div, table and the like: whatever blocks are inside, or
            # the text itself.
            if any(not isinstance(c, str) and c.tag in _BLOCKS and c.tag != "br" for c in node.children):
                _blocks(node.children, out)
            else:
                text = _tidy(_inline(node.children))
                if text:
                    out.append(_line_start_safe(text))
    flush()


def _list_lines(node, level: int, items=None) -> list:
    lines = []
    ordered = node.tag == "ol"
    number = 1
    for item in items if items is not None else node.children:
        if isinstance(item, str):
            if item.strip():
                lines.append("  " * level + "- " + _escape_text(item.strip()))
            continue
        if item.tag in ("ul", "ol"):
            lines.extend(_list_lines(item, level + 1))
            continue
        children = item.children if item.tag == "li" else [item]
        text_nodes = [c for c in children if isinstance(c, str) or c.tag not in ("ul", "ol")]
        text = _tidy(_inline(text_nodes)).replace("\n", " ")
        marker = f"{number}." if ordered else "-"
        number += 1
        lines.append("  " * level + f"{marker} {text}".rstrip())
        for child in children:
            if not isinstance(child, str) and child.tag in ("ul", "ol"):
                lines.extend(_list_lines(child, level + 1))
    return lines


def from_html(source: str) -> str:
    """HTML (what the editor produced) as Markdown (the subset above)."""
    tree = _Tree()
    tree.feed(source or "")
    tree.close()
    out: list = []
    _blocks(tree.root.children, out)
    text = "\n\n".join(out)
    # A region's markers sit on their own lines, with no blank line inside
    # needed for the round trip -- but keep one so it reads well.
    return re.sub(r"\n{3,}", "\n\n", text).strip() + ("\n" if text.strip() else "")


def for_report(text: str, demote: int = 2) -> str:
    """Notes as a report section carries them: the region markers dropped
    and every heading `demote` levels down (capped at six), so a note's
    "# Title" sits under the report's "## Notes" instead of beside it."""
    out, fenced = [], False
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            if REGION_OPEN.match(line.strip()) or REGION_CLOSE.match(line.strip()):
                continue
            heading = HEADING.match(line.strip())
            if heading:
                line = "#" * min(6, len(heading.group(1)) + demote) + " " + heading.group(2)
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def to_plain(text: str) -> str:
    """Markdown as plain lines (a slide's text box): the markers dropped,
    the formatting characters removed, list items kept as bullets."""
    tree = _Tree()
    tree.feed(to_html(text))
    tree.close()
    out: list = []
    _plain_blocks(tree.root.children, out, 0)
    return "\n".join(line for line in out).strip()


def _plain_blocks(nodes, out: list, level: int) -> None:
    for node in nodes:
        if isinstance(node, str):
            if node.strip():
                out.append(node.strip())
            continue
        if node.tag in ("ul", "ol"):
            for k, item in enumerate(c for c in node.children if not isinstance(c, str)):
                bullet = f"{k + 1}." if node.tag == "ol" else "•"
                text = " ".join(_plain(c) if not isinstance(c, str) else c for c in item.children
                                if isinstance(c, str) or c.tag not in ("ul", "ol")).strip()
                out.append("    " * level + f"{bullet} {re.sub(r'\s+', ' ', text)}")
                _plain_blocks([c for c in item.children if not isinstance(c, str) and c.tag in ("ul", "ol")],
                              out, level + 1)
        elif node.tag == "div":
            _plain_blocks(node.children, out, level)
        else:
            if node.tag == "blockquote":
                _plain_blocks(node.children, out, level)
                continue
            text = re.sub(r"[ \t]+", " ", _plain(node).replace("\xa0", " ")).strip()
            out.extend(line.strip() for line in text.split("\n") if line.strip())
