"""A PowerPoint (.pptx) rendering of the same report `report.py` writes as HTML.

.pptx is Office Open XML (ECMA-376): a zip of plain XML that python-pptx
writes directly. No Microsoft software, no license, no network call --
same offline guarantee as the rest of the app (see CLAUDE.md).

`PptxReport` subclasses `Report` and overrides its content methods
(`section`, `table`, `figure`, `paragraph`, `tiles`, `dataframe`) so that
report_builder.py's `_add_*` functions -- which only ever call those
methods -- build a slide deck instead of an HTML document, unchanged.
Where report.py's methods return an HTML string to accumulate, this
class's build a slide immediately and return "" (harmless once appended
to `section.blocks`, which nothing here ever reads back).

A short paragraph immediately followed by a table, figure or tiles block
(e.g. `paragraph(experiment_name)` then `table(...)` in report_builder's
per-class section) is folded into that block's own slide title instead of
getting a slide of its own -- otherwise every experiment name would flash
past on a slide with nothing else on it. A "note" paragraph (a caveat,
not a label) instead attaches to the slide that was just built, the way a
footnote sits under the thing it caveats.
"""
from __future__ import annotations

import io
import platform
from datetime import datetime
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from lookin.core.report import Report, _fmt

SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)
MARGIN = Inches(0.5)
MAX_TABLE_ROWS = 14  # more than this per slide and the text is unreadable
NOTES_LINES_PER_SLIDE = 16

_MUTED = RGBColor(0x5C, 0x5C, 0x68)
_POS = RGBColor(0x1A, 0x7F, 0x4B)
_NEG = RGBColor(0xB2, 0x3B, 0x3B)


class PptxReport(Report):
    """Same content-building API as `Report`; renders slides, not HTML."""

    extension = "pptx"
    mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

    def __init__(self, title: str, subtitle: str = ""):
        super().__init__(title, subtitle)
        self._prs = Presentation()
        self._prs.slide_width = SLIDE_WIDTH
        self._prs.slide_height = SLIDE_HEIGHT
        self._title_slide_built = False
        self._current_section_title = ""
        self._pending_label = None
        self._last_slide = None
        self._footnote_boxes: dict = {}
        self._finished = False

    # ---- content (overrides Report's) ------------------------------------
    def section(self, title: str):
        self._flush_pending_label()
        # `add_meta` is always called before the first `section()` in
        # report_builder.build_report, so `self.meta` is complete by now --
        # the title slide has to be built after it, not in __init__.
        if not self._title_slide_built:
            self._build_title_slide()
            self._title_slide_built = True
        self._current_section_title = title
        self._last_slide = None
        return super().section(title)

    def tiles(self, items: list) -> str:
        label = self._consume_pending_label()
        self._last_slide = self._add_tiles_slide(label, items)
        return ""

    def table(self, headers: list, rows: list, note: str = "") -> str:
        label = self._consume_pending_label()
        self._last_slide = self._add_table_slides(label, headers, rows, note)
        return ""

    def dataframe(self, df, note: str = "", float_digits: int = 4) -> str:
        headers = [str(df.index.name or "")] + [str(c) for c in df.columns]
        rows = [[str(idx)] + [df.loc[idx, c] for c in df.columns] for idx in df.index]
        return self.table(headers, rows, note)

    def figure(self, fig, caption: str = "", dpi: int = 130) -> str:
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        label = self._consume_pending_label()
        self._last_slide = self._add_figure_slide(label, buf.getvalue(), caption)
        return ""

    def notes(self, markdown: str) -> str:
        """The user's notes as text slides, a slide per heading of theirs
        (the recap markers and the formatting characters dropped)."""
        from lookin.core import markdown_lite as md

        self._flush_pending_label()
        chunks, title, lines = [], self._current_section_title, []
        for line in md.for_report(markdown, demote=0).split("\n"):
            heading = md.HEADING.match(line.strip())
            if heading:
                if any(l.strip() for l in lines):
                    chunks.append((title, "\n".join(lines)))
                title, lines = heading.group(2), []
            else:
                lines.append(line)
        if any(l.strip() for l in lines):
            chunks.append((title, "\n".join(lines)))
        for title, body in chunks:
            text = md.to_plain(body)
            for k in range(0, max(1, len(text.split("\n"))), NOTES_LINES_PER_SLIDE):
                part = "\n".join(text.split("\n")[k:k + NOTES_LINES_PER_SLIDE])
                self._last_slide = self._add_text_slide(title, part)
        return ""

    def paragraph(self, text: str, note: bool = False) -> str:
        if note:
            if self._last_slide is not None:
                self._add_footnote(self._last_slide, text)
            else:
                self._last_slide = self._add_text_slide(self._current_section_title, text)
        else:
            self._flush_pending_label()
            self._pending_label = text
        return ""

    # ---- output -----------------------------------------------------------
    def _finish(self) -> None:
        """The last slides, once: to_bytes() and write() may both be called."""
        if self._finished:
            return
        self._finished = True
        self._flush_pending_label()
        self._add_footer_slide()

    def to_bytes(self) -> bytes:
        self._finish()
        buf = io.BytesIO()
        self._prs.save(buf)
        return buf.getvalue()

    def write(self, dest_path) -> Path:
        self._finish()

        from lookin.core import fileio

        dest_path = Path(dest_path)
        fileio.atomic_write(dest_path, lambda tmp: self._prs.save(str(tmp)))
        return dest_path

    # ---- slide building -----------------------------------------------------
    def _consume_pending_label(self):
        label = self._pending_label
        self._pending_label = None
        return label

    def _flush_pending_label(self) -> None:
        if self._pending_label is not None:
            label = self._pending_label
            self._pending_label = None
            self._last_slide = self._add_text_slide(self._current_section_title, label)

    def _add_label_box(self, slide, label: str) -> int:
        """A short label (an experiment name) or a full sentence (a caveat
        naming what follows) as plain body text under the slide's title --
        not folded into the title placeholder itself, which is one line
        tall and clips or overlaps the content below it once the text is
        longer than a word or two. Returns the top (in EMU) the slide's
        main content should start at, below this box."""
        top = int(Inches(0.95))
        box = slide.shapes.add_textbox(MARGIN, top, SLIDE_WIDTH - 2 * MARGIN, Inches(0.9))
        text_frame = box.text_frame
        text_frame.word_wrap = True
        text_frame.text = label
        self._style_run(text_frame.paragraphs[0].runs[0], size=16)
        return int(Inches(1.95))

    def _build_title_slide(self) -> None:
        # The stock template's title/subtitle placeholders are positioned for
        # its original 4:3 size; resizing the presentation to widescreen (see
        # __init__) does not move them, so both need an explicit top/height
        # here or they print on top of each other.
        slide = self._prs.slides.add_slide(self._prs.slide_layouts[0])
        title_ph = slide.shapes.title
        title_ph.text = self.title
        title_ph.left, title_ph.top = MARGIN, Inches(0.6)
        title_ph.width, title_ph.height = SLIDE_WIDTH - 2 * MARGIN, Inches(1.0)
        if len(slide.placeholders) > 1:
            subtitle_ph = slide.placeholders[1]
            subtitle_ph.text = self.subtitle
            subtitle_ph.left, subtitle_ph.top = MARGIN, Inches(1.7)
            subtitle_ph.width, subtitle_ph.height = SLIDE_WIDTH - 2 * MARGIN, Inches(0.6)
        if self.meta:
            top = Inches(2.6)
            n_rows = len(self.meta)
            height = min(int(Inches(0.42)) * n_rows, int(SLIDE_HEIGHT - top - MARGIN))
            frame = slide.shapes.add_table(n_rows, 2, MARGIN, top, SLIDE_WIDTH - 2 * MARGIN, height)
            table = frame.table
            key_width = int(Inches(3.4))
            table.columns[0].width = key_width
            table.columns[1].width = int(SLIDE_WIDTH - 2 * MARGIN) - key_width
            for i, (key, value) in enumerate(self.meta):
                self._set_cell_text(table.cell(i, 0), str(key), bold=True)
                self._set_cell_text(table.cell(i, 1), str(value))

    def _new_slide(self, title: str):
        slide = self._prs.slides.add_slide(self._prs.slide_layouts[5])  # "Title Only"
        title_ph = slide.shapes.title
        title_ph.text = title
        title_ph.left, title_ph.top = MARGIN, Inches(0.3)
        title_ph.width = SLIDE_WIDTH - 2 * MARGIN
        return slide

    def _add_table_slides(self, label, headers: list, rows: list, note: str):
        title = self._current_section_title
        chunks = [rows[i:i + MAX_TABLE_ROWS] for i in range(0, len(rows), MAX_TABLE_ROWS)] or [[]]
        slide = None
        for page, chunk in enumerate(chunks, start=1):
            page_title = title if len(chunks) == 1 else f"{title} ({page}/{len(chunks)})"
            slide = self._new_slide(page_title)
            top = self._add_label_box(slide, label) if label else int(Inches(1.1))
            self._add_table_shape(slide, headers, chunk, top)
            if page == len(chunks) and note:
                self._add_footnote(slide, note)
        return slide

    def _add_table_shape(self, slide, headers: list, rows: list, top: int) -> None:
        n_rows, n_cols = len(rows) + 1, max(len(headers), 1)
        width = int(SLIDE_WIDTH - 2 * MARGIN)
        height = min(int(Inches(0.4)) * n_rows, int(SLIDE_HEIGHT - top - MARGIN))
        frame = slide.shapes.add_table(n_rows, n_cols, MARGIN, top, width, height)
        table = frame.table
        for c, header in enumerate(headers):
            self._set_cell_text(table.cell(0, c), str(header), bold=True)
        for r, row in enumerate(rows, start=1):
            for c, value in enumerate(row):
                text = value if isinstance(value, str) else _fmt(value)
                self._set_cell_text(table.cell(r, c), text, size=10)

    def _add_figure_slide(self, label, png_bytes: bytes, caption: str):
        slide = self._new_slide(self._current_section_title)
        top = self._add_label_box(slide, label) if label else int(Inches(1.1))
        caption_h = Inches(0.4) if caption else 0
        avail_w, avail_h = int(SLIDE_WIDTH - 2 * MARGIN), int(SLIDE_HEIGHT - top - MARGIN - caption_h)
        with Image.open(io.BytesIO(png_bytes)) as img:
            px_w, px_h = img.size
        aspect = px_w / px_h
        width, height = avail_w, avail_w / aspect
        if height > avail_h:
            height, width = avail_h, avail_h * aspect
        left = int(MARGIN) + int((avail_w - width) / 2)
        img_top = int(top) + int((avail_h - height) / 2)
        slide.shapes.add_picture(io.BytesIO(png_bytes), left, img_top, width=int(width), height=int(height))
        if caption:
            box = slide.shapes.add_textbox(MARGIN, SLIDE_HEIGHT - Inches(0.5), SLIDE_WIDTH - 2 * MARGIN, Inches(0.4))
            box.text_frame.text = caption
            paragraph = box.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.CENTER
            self._style_run(paragraph.runs[0], size=11, italic=True, color=_MUTED)
        return slide

    def _add_tiles_slide(self, label, items: list):
        slide = self._new_slide(self._current_section_title)
        n = len(items)
        cols = min(4, max(n, 1))
        rows = -(-n // cols) if n else 1
        top0 = self._add_label_box(slide, label) if label else int(Inches(1.2))
        avail_w, avail_h = int(SLIDE_WIDTH - 2 * MARGIN), int(SLIDE_HEIGHT - top0 - MARGIN)
        tile_w, tile_h = avail_w / cols, min(int(Inches(1.6)), avail_h / rows)
        for idx, (item_label, value, css_class) in enumerate(items):
            r, c = divmod(idx, cols)
            left = int(MARGIN) + int(c * tile_w) + int(Inches(0.05))
            top = int(top0) + int(r * tile_h)
            box = slide.shapes.add_textbox(left, top, int(tile_w) - int(Inches(0.1)), int(tile_h) - int(Inches(0.1)))
            text_frame = box.text_frame
            text_frame.word_wrap = True
            label_p = text_frame.paragraphs[0]
            label_p.text = str(item_label)
            self._style_run(label_p.runs[0], size=11, color=_MUTED)
            value_p = text_frame.add_paragraph()
            value_p.text = value if isinstance(value, str) else _fmt(value)
            color = _POS if css_class == "pos" else _NEG if css_class == "neg" else None
            self._style_run(value_p.runs[0], size=26, bold=True, color=color)
        return slide

    def _add_text_slide(self, title: str, text: str):
        slide = self._new_slide(title)
        box = slide.shapes.add_textbox(MARGIN, Inches(1.2), SLIDE_WIDTH - 2 * MARGIN, Inches(1.0))
        box.text_frame.word_wrap = True
        box.text_frame.text = text
        for paragraph in box.text_frame.paragraphs:
            for run in paragraph.runs:
                self._style_run(run, size=14)
        return slide

    def _add_footnote(self, slide, text: str) -> None:
        box = self._footnote_boxes.get(id(slide))
        if box is None:
            box = slide.shapes.add_textbox(MARGIN, SLIDE_HEIGHT - Inches(0.9), SLIDE_WIDTH - 2 * MARGIN, Inches(0.8))
            box.text_frame.word_wrap = True
            box.text_frame.paragraphs[0].text = text
            self._footnote_boxes[id(slide)] = box
            paragraph = box.text_frame.paragraphs[0]
        else:
            paragraph = box.text_frame.add_paragraph()
            paragraph.text = text
        self._style_run(paragraph.runs[0], size=10, italic=True, color=_MUTED)

    def _add_footer_slide(self) -> None:
        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._add_text_slide(
            "Generated",
            f"Generated by Lookin on {generated} ({platform.node()}). Self-contained: every image "
            f"and table is embedded in this file, nothing is fetched when it is opened.",
        )

    @staticmethod
    def _set_cell_text(cell, text: str, size: int = 11, bold: bool = False) -> None:
        cell.text = text
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(size)
                run.font.bold = bold

    @staticmethod
    def _style_run(run, size: int = 12, bold: bool = False, italic: bool = False, color=None) -> None:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        if color is not None:
            run.font.color.rgb = color
