"""Small charts drawn as inline SVG and HTML: bars, histograms and shaded
tables in a card.

matplotlib is what the analysis tabs use for their figures, but a page of
a dozen small charts would be a dozen renders and images to fetch. These
are a few hundred bytes of markup each,
drawn from numbers already computed, sharp at any zoom and styled by the
page's own CSS variables -- so they cost nothing to show.
"""
from __future__ import annotations

from lookin.web import html as h

ACCENT = "var(--acc)"


def _fmt_count(n) -> str:
    return f"{int(n):,}"


def hbars(rows: list, *, limit: int = 0, total: int = 0, more_attr: str = "", value_fmt=_fmt_count) -> str:
    """A list of (label, value) as horizontal bars, with the share of
    `total` beside each. `limit` rows shown, then a "show all" line."""
    if not rows:
        return h.empty("Nothing to show.")
    top = max(v for _, v in rows) or 1
    shown = rows[:limit] if limit and len(rows) > limit else rows
    total = total or sum(v for _, v in rows) or 1
    out = []
    for label, v in shown:
        out.append(f'<div class="hb-row" title="{h.e(label)}: {h.e(value_fmt(v))} ({v / total:.1%})">'
                   f'<span class="hb-l mono ell">{h.e(label)}</span>'
                   f'<span class="hb-t"><span class="hb-b" style="width:{100 * v / top:.2f}%"></span></span>'
                   f'<span class="hb-v mono">{h.e(value_fmt(v))}</span>'
                   f'<span class="hb-p mono">{v / total:.1%}</span></div>')
    more = ""
    if len(shown) < len(rows):
        rest = rows[len(shown):]
        label = f"Show all {len(rows):,}" if more_attr else f"{len(rest):,} more"
        more = (f'<button type="button" class="hb-more"{more_attr}>{h.e(label)}</button>' if more_attr
                else f'<div class="hb-more">{h.e(label)}</div>')
    return f'<div class="hbars">{"".join(out)}{more}</div>'


def histogram(edges: list, counts: list, width: int = 220, height: int = 46) -> str:
    if not counts or not edges:
        return ""
    top = max(counts) or 1
    n = len(counts)
    bar_w = width / n
    rects = []
    for i, c in enumerate(counts):
        bh = height * c / top if c else 0
        tip = f"{edges[i]:.4g} to {edges[i + 1]:.4g}: {c:,}"
        rects.append(f'<rect x="{i * bar_w:.2f}" y="{height - bh:.2f}" width="{max(bar_w - 1, 0.5):.2f}" '
                     f'height="{bh:.2f}"><title>{h.e(tip)}</title></rect>')
    return (f'<svg class="mini-hist" width="100%" height="{height}" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none"><g fill="{ACCENT}">{"".join(rects)}</g></svg>')


def share_table(rows: list, cols: list, counts: list, *, row_limit: int = 0, more_attr: str = "",
                corner: str = "") -> str:
    """A class x value table shaded by row share: each cell is the part of
    the row's total in that column, so every row reads on its own whatever
    the class's size. Counts are in the tooltips."""
    if not rows or not cols:
        return h.empty("Nothing to show.")
    shown = rows[:row_limit] if row_limit and len(rows) > row_limit else rows
    head = "".join(f'<th class="mono ell" title="{h.e(c)}">{h.e(c)}</th>' for c in cols)
    body = []
    for i, label in enumerate(shown):
        row = counts[i]
        total = sum(row) or 1
        cells = []
        for j, n in enumerate(row):
            share = n / total
            # The accent mixed into nothing: the same shade reads on the light
            # and the dark theme, whose --acc differ.
            alpha = 8 + 85 * share if n else 0
            ink = "#fff" if share > 0.55 else "var(--ink)"
            bg = f"background:color-mix(in srgb, var(--acc) {alpha:.1f}%, transparent);" if n else ""
            cells.append(f'<td class="sh-c mono" style="{bg}color:{ink}" title="{h.e(label)} / {h.e(cols[j])}: '
                         f'{n:,} rows ({share:.1%} of the class)">{"" if not n else f"{share:.0%}"}</td>')
        body.append(f'<tr><th class="sh-r mono ell" title="{h.e(label)}">{h.e(label)}</th>'
                    f'<td class="sh-n mono">{sum(row):,}</td>{"".join(cells)}</tr>')
    more = ""
    if len(shown) < len(rows):
        more = (f'<button type="button" class="hb-more"{more_attr}>Show all {len(rows):,} classes</button>'
                if more_attr else f'<div class="hb-more">{len(rows) - len(shown):,} more classes</div>')
    return (f'<div class="sh-wrap"><table class="sh"><thead><tr><th class="sh-r">{h.e(corner)}</th>'
            f'<th class="sh-n">rows</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{more}')


def diverging_bar(delta: float, scale: float) -> str:
    """A bar left (negative) or right (positive) of a centre line, for a
    change in share; `scale` is the largest |delta| shown."""
    scale = scale or 1.0
    w = 50 * min(1.0, abs(delta) / scale)
    side = f"left:{50 - w:.2f}%" if delta < 0 else "left:50%"
    cls = "neg" if delta < 0 else "pos"
    return f'<span class="dv"><span class="dv-b {cls}" style="{side};width:{w:.2f}%"></span></span>'
