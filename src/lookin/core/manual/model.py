"""The shape of a manual page: blocks of content, and the page that holds
them.

Deliberately in core/ and free of Flet, for the same reason the chart style
is (see core/chart_style.py): the text of a page is data, and the widget
that draws it is not. That split is what lets a test walk every equation in
the manual and typeset it without a window, and what would let an exported
report reuse a page's prose later.

The blocks are a small closed set rather than Markdown on purpose. Markdown
would hand us a string with `$...$` in it and no way to know which parts are
mathematics until render time, and Flet has no Markdown renderer that
typesets mathematics at all (see ui/mathtext.py for what does). A block
list says, up front, what each piece *is*.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---- inline content -----------------------------------------------------
# Prose may carry short mathematics inside a sentence ("the confidence
# p_max"), written as $...$ inside the text of a Para or a bullet. The
# renderer splits on those; the model keeps them in the string so a page
# reads like a paragraph in the source rather than a list of fragments.
MATH_DELIMITER = "$"


@dataclass(frozen=True)
class Para:
    """A paragraph of prose. May contain $inline math$."""
    text: str


@dataclass(frozen=True)
class Heading:
    """A section heading inside a page."""
    text: str


@dataclass(frozen=True)
class Equation:
    """A display equation, typeset on its own line.

    `tex` is mathematics *without* the surrounding dollars -- the renderer
    adds them. `where` is the list of (symbol, meaning) pairs that defines
    every variable the equation just introduced; a symbol is mathematics
    too, again without dollars. An equation whose variables are already
    defined on the page leaves `where` empty.

    `note` is the one line that says what the formula *means*, printed
    under it in italics: the sentence a reader needs when the algebra is
    clear but the point of it is not.
    """
    tex: str
    where: tuple = ()
    note: str = ""


@dataclass(frozen=True)
class Bullets:
    """An unordered list. Each item may contain $inline math$."""
    items: tuple
    numbered: bool = False


@dataclass(frozen=True)
class Table:
    """A small table of literal text -- a worked example's arithmetic, or a
    "this column means that" legend. Cells may contain $inline math$."""
    headers: tuple
    rows: tuple
    caption: str = ""
    # Relative column widths, one per header. Left empty the columns share
    # the width equally, which is wrong for the common "symbol, then what
    # it means" table where the first column holds three characters.
    weights: tuple = ()


@dataclass(frozen=True)
class Callout:
    """A boxed aside. `kind` picks the color and icon, and is compared in
    the renderer: "pitfall", "tip", "warning", "example".
    """
    kind: str
    title: str
    blocks: tuple


CALLOUT_PITFALL = "pitfall"
CALLOUT_TIP = "tip"
CALLOUT_WARNING = "warning"
CALLOUT_EXAMPLE = "example"
CALLOUT_KINDS = (CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING, CALLOUT_EXAMPLE)


@dataclass(frozen=True)
class HowToRead:
    """The "what am I looking at" block: one line per visual element of the
    chart or table this page explains. Items are (element, meaning) pairs.

    Its own block rather than a Table because it is the one thing a reader
    who clicked (i) on a chart came for, and it is drawn accordingly.
    """
    items: tuple
    title: str = "How to read it"


# ---- the page -----------------------------------------------------------

@dataclass(frozen=True)
class ManualPage:
    """One page of the manual.

    `id` is the anchor an (i) button points at -- a stable key, compared in
    code and never shown to the user, so it is never reworded (see
    CLAUDE.md on keys vs labels).

    `summary` is the two or three sentences at the top of the page: what
    this quantity is, in one breath, for the reader who clicked (i) to
    settle a single question and does not want the course.

    `blocks` is the course underneath it. `see_also` lists other page ids,
    rendered as links at the bottom.

    `keywords` is extra text the page search matches on -- the names a
    reader might type that do not appear in the title.
    """
    id: str
    title: str
    summary: str
    blocks: tuple = ()
    see_also: tuple = ()
    keywords: tuple = ()
    # Where in the app this page is explained from. Free text, shown under
    # the title, so a page opened from the table of contents still says
    # which tab it belongs to.
    shown_in: str = ""


@dataclass(frozen=True)
class Chapter:
    """A group of pages in the table of contents."""
    id: str
    title: str
    pages: tuple = field(default_factory=tuple)
