"""A self-contained HTML report of what is currently on screen.

Everything this tool computes dies when the browser tab closes. A run is
worth writing down: which dataset, which experiments, which review version,
and the numbers and charts that came out. This writes one file that can be
mailed, archived next to a training run, or opened in six months.

Self-contained on purpose, and for the same reason the rest of the app is
(see CLAUDE.md): every image is inlined as a base64 data URI and every style
is in the document, so the file opens with no network, no server and no
sibling directory. It is also why this is HTML rather than PDF -- a PDF
writer would be a new dependency, and a browser prints this to PDF anyway.

Nothing here knows about the web pages: the report is built from the same core objects
the views render, so it can be produced (and tested) without a UI.
"""
from __future__ import annotations

import base64
import html
import io
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPORT_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px; background: #f6f6f8; color: #1b1b1f;
       font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #dcdce3; }
h3 { font-size: 15px; margin: 20px 0 8px; }
.sub { color: #5c5c68; margin: 0 0 24px; }
.card { background: #fff; border: 1px solid #e2e2e9; border-radius: 12px; padding: 18px; margin: 12px 0; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
.tile { border: 1px solid #e2e2e9; border-radius: 10px; padding: 12px 14px; min-width: 150px; background: #fff; }
.tile .k { color: #5c5c68; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
.tile .v { font-size: 22px; font-weight: 700; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid #ececf1; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: #5c5c68; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
tbody tr:hover { background: #fafafc; }
.scroll { overflow-x: auto; }
figure { margin: 0; }
figure img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e2e9; background: #fff; }
figcaption { color: #5c5c68; font-size: 12px; margin-top: 6px; }
.note { color: #5c5c68; font-size: 12px; font-style: italic; }
.pos { color: #1a7f4b; } .neg { color: #b23b3b; }
.notes h3, .notes h4, .notes h5, .notes h6 { margin: 14px 0 6px; }
.notes p, .notes ul, .notes ol { margin: 0 0 8px; }
.notes blockquote { margin: 0 0 8px; padding: 2px 12px; border-left: 3px solid #dcdce3; color: #5c5c68; }
.notes code { background: #f1f1f5; border-radius: 3px; padding: 0 3px; }
footer { color: #8a8a95; font-size: 11px; margin-top: 40px; border-top: 1px solid #dcdce3; padding-top: 12px; }
@media print { body { background: #fff; padding: 0; } .card, .tile { break-inside: avoid; } }
"""


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value, digits: int = 4) -> str:
    """Numbers the way the app shows them, with NaN spelled out rather than
    printed as 'nan' -- a report is read by people who were not there."""
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if number != number:  # NaN
        return "n/a"
    if number == int(number) and abs(number) < 1e6:
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


@dataclass
class ReportSection:
    title: str
    blocks: list = field(default_factory=list)  # already-rendered HTML strings


class Report:
    """Accumulates sections, then renders one HTML document."""

    # What the file is, for a writer-agnostic caller (a browser download).
    extension = "html"
    mime = "text/html; charset=utf-8"

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.sections: list = []
        self.meta: list = []  # (key, value) shown in the header block

    # ---- content ----
    def add_meta(self, key: str, value) -> None:
        self.meta.append((key, value))

    def section(self, title: str) -> ReportSection:
        section = ReportSection(title=title)
        self.sections.append(section)
        return section

    @staticmethod
    def tiles(items: list) -> str:
        """items: [(label, value, css_class or "")]"""
        cells = "".join(
            f'<div class="tile"><div class="k">{_esc(label)}</div>'
            f'<div class="v {cls}">{value if isinstance(value, str) else _fmt(value)}</div></div>'
            for label, value, cls in items
        )
        return f'<div class="grid">{cells}</div>'

    @staticmethod
    def table(headers: list, rows: list, note: str = "") -> str:
        head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell if isinstance(cell, str) else _fmt(cell)}</td>"
                             for cell in row) + "</tr>"
            for row in rows
        )
        note_html = f'<p class="note">{_esc(note)}</p>' if note else ""
        return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>{note_html}")

    @staticmethod
    def dataframe(df, note: str = "", float_digits: int = 4) -> str:
        headers = [str(df.index.name or "")] + [str(c) for c in df.columns]
        rows = [[str(idx)] + [df.loc[idx, c] for c in df.columns] for idx in df.index]
        return Report.table(headers, rows, note)

    @staticmethod
    def figure(fig, caption: str = "", dpi: int = 130) -> str:
        """A matplotlib figure, inlined. The figure is closed here: a report
        can hold a dozen of them, and pyplot keeps every figure alive until
        it is told otherwise."""
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        caption_html = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        return (f'<figure><img alt="{_esc(caption)}" '
                f'src="data:image/png;base64,{encoded}">{caption_html}</figure>')

    @staticmethod
    def paragraph(text: str, note: bool = False) -> str:
        return f'<p class="{"note" if note else ""}">{_esc(text)}</p>'

    @staticmethod
    def notes(markdown: str) -> str:
        """The user's notes (core/notes.py), formatted: their headings sit
        under the report's own sections, the recap markers are dropped."""
        from lookin.core import markdown_lite as md

        return f'<div class="card notes">{md.to_html(md.for_report(markdown))}</div>'

    # ---- output ----
    def to_html(self) -> str:
        meta_rows = "".join(
            f"<tr><td>{_esc(k)}</td><td style='text-align:left'>{_esc(v)}</td></tr>"
            for k, v in self.meta
        )
        meta_block = (f'<div class="card"><div class="scroll"><table><tbody>{meta_rows}'
                      f"</tbody></table></div></div>") if meta_rows else ""
        body = "".join(
            f"<h2>{_esc(section.title)}</h2>" + "".join(section.blocks)
            for section in self.sections
        )
        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_esc(self.title)}</title><style>{REPORT_CSS}</style></head><body><main>'
            f'<h1>{_esc(self.title)}</h1>'
            f'<p class="sub">{_esc(self.subtitle)}</p>'
            f'{meta_block}{body}'
            f'<footer>Generated by Lookin on {generated} '
            f'({_esc(platform.node())}). Self-contained: every image is embedded, '
            f'nothing is fetched when this file is opened.</footer>'
            '</main></body></html>'
        )

    def to_bytes(self) -> bytes:
        """The whole file, for a download in the browser."""
        return self.to_html().encode("utf-8")

    def write(self, dest_path) -> Path:
        """Atomic write, same rule as every other file this app produces: a
        half-written report is worse than none."""
        from lookin.core import fileio

        dest_path = Path(dest_path)
        data = self.to_bytes()
        fileio.atomic_write(dest_path, lambda tmp: tmp.write_bytes(data))
        return dest_path
