r"""Typesetting mathematics with no network and no browser.

The manual is a course, and a course without equations is a glossary. The
obvious way to get real equations on screen -- KaTeX or MathJax in a web
view -- is exactly the kind of thing CLAUDE.md rules out unless the whole
bundle *and* its fonts are vendored, and Flet's Markdown widget does not
typeset mathematics at all.

matplotlib is already a dependency, it ships its own math fonts (STIX, the
same face core/chart_style.py gives every chart), and its mathtext engine
is a real typesetter: fractions, sums, roots, hats, subscripts, Greek. So a
formula is drawn the way a chart is -- to a PNG, in memory, offline -- and
handed to Flet as base64.

Two consequences worth knowing before writing a page:

- The output is raster, so it is rendered at RENDER_DPI and displayed at a
  quarter of that scale, which is what keeps it sharp on a high-density
  screen. It cannot be selected or copied as text.
- mathtext is a *subset* of LaTeX. \text, \operatorname, \mathbb, \mathcal
  and \boldsymbol all typeset on the pinned matplotlib; environments do
  not -- no align, no cases, no array -- and neither does \underbrace. A
  derivation is therefore one Equation block per line, spaced with \, and
  \;. Anything unsupported raises, and is caught here rather than blanking
  a page --
  but tests/test_manual.py typesets every equation in the manual, so a
  typo fails there instead of quietly rendering as an error string.
"""
from __future__ import annotations

import base64
import io
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from lookin.core.chart_style import apply_chart_style

apply_chart_style()

# Rendered at 4x the logical size and displayed at 1/4 of it: a formula is
# small text, and small text resampled down from a high-resolution render
# is the difference between "typeset" and "a screenshot of typeset".
RENDER_DPI = 288.0
LOGICAL_DPI = 72.0
SCALE = RENDER_DPI / LOGICAL_DPI

# The two sizes a page uses, in points: a display equation on its own line,
# and a symbol sitting inside a sentence. Inline is a touch smaller than
# the prose around it (14) because mathtext italics read taller than the
# body face at the same nominal size.
DISPLAY_SIZE = 17.0
INLINE_SIZE = 13.5

# Ink color. The app is dark-themed and fixed that way (see app.py), so
# this is a constant rather than something read back from the theme -- and
# the background stays transparent, so an equation sits on whatever card
# it lands in.
INK = "#E3E6EB"
INK_MUTED = "#AEB5C0"

# Left as-is when the formula will not typeset: the raw source, so the page
# still says something and the error is obvious rather than silent.
class MathRenderError(ValueError):
    pass


@lru_cache(maxsize=1024)
def render_math(tex: str, size: float = DISPLAY_SIZE, color: str = INK) -> tuple:
    """Typeset `tex` (mathematics, without the surrounding dollars).

    Returns (base64_png, width, height) where the two lengths are in
    logical pixels, ready to hand to an ft.Image. Raises MathRenderError if
    mathtext cannot parse the formula.

    Cached: a page re-renders on every open, the manual has hundreds of
    formulas between its pages, and each one is a matplotlib figure.
    """
    # A bare Figure rather than plt.figure(): pyplot keeps every figure it
    # creates in a global registry until it is closed, and a page holds
    # dozens of formulas. Going through the canvas directly also means this
    # can run off the UI thread without touching pyplot's global state.
    figure = Figure(figsize=(0.01, 0.01))
    FigureCanvasAgg(figure)
    figure.patch.set_alpha(0.0)
    figure.text(0, 0, f"${tex}$", fontsize=size, color=color)
    buffer = io.BytesIO()
    try:
        figure.savefig(
            buffer, format="png", dpi=RENDER_DPI, transparent=True,
            bbox_inches="tight", pad_inches=0.02,
        )
    except Exception as exc:  # mathtext parse errors surface at draw time
        raise MathRenderError(f"cannot typeset {tex!r}: {exc}") from exc

    data = buffer.getvalue()
    width, height = _png_size(data)
    return base64.b64encode(data).decode("ascii"), width / SCALE, height / SCALE


def _png_size(data: bytes) -> tuple:
    """Width and height in device pixels, read straight out of the PNG IHDR
    chunk -- 16 bytes into a file we just wrote, which is cheaper and more
    honest than decoding the image again through PIL."""
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height
