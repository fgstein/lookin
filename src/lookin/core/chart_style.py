"""One look for every chart the app draws, and the two helpers that put it
on an Axes.

In core rather than in ui/components.py because core/report_builder.py draws
figures too, and that half of the app deliberately does not import Flet (see
its module docstring) -- a style that only applied to the charts on screen
would give the exported report a different typeface from the tab it was
exported from.

Nothing here is fetched: see CHART_SERIF_FONTS below, and CLAUDE.md for why
that matters.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, NullLocator


def subplots(nrows: int = 1, ncols: int = 1, *, figsize=None, dpi=None, **kwargs):
    """``plt.subplots`` without pyplot's global figure registry.

    Every browser tab is a session in the same server process, and Flet runs
    event handlers on a thread pool -- so two tabs (or two handlers of one)
    can draw at the same moment. pyplot keeps every figure in one global
    registry that is not safe to touch from two threads at once, which is
    why the heavy rebuilds already queue behind one lock (see
    ui/components.BackgroundRebuild). A figure made here never enters that
    registry, so it can be drawn from any thread without that queue.

    The figure is otherwise identical: rcParams (the app style below) are
    read at creation as with pyplot, and plt.close() on it is a harmless
    no-op, so existing callers need no other change.
    """
    figure = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(figure)
    axes = figure.subplots(nrows, ncols, **kwargs)
    return figure, axes


# ---- one look for every chart in the app --------------------------------
# Every figure this app draws goes out as a PNG that ends up in a report, a
# slide or a paper, so they are typeset rather than merely plotted: a serif
# face and a grid dense enough to read a value off the axes.
#
# The font list is bounded by the offline rule (see CLAUDE.md): STIXGeneral
# and DejaVu Serif are both shipped *inside* matplotlib, so this resolves
# with no font on the machine and nothing fetched. STIXGeneral first because
# it is the Times-like face the scientific publishing world (and LaTeX's
# newtx/mathptmx) uses, which is what "looks like LaTeX" means here;
# matching mathtext to it keeps a $\mu$ in an axis label in the same
# typeface as the words beside it. Computer Modern (cmr10) would be the
# literal LaTeX face and is bundled too, but it has no minus sign glyph, so
# every negative tick label renders as a box.
CHART_SERIF_FONTS = ["STIXGeneral", "DejaVu Serif", "serif"]

# The two grid weights. Major is what values are read off; minor is there to
# subdivide it, so it has to be visible without competing with the data --
# hence the much lighter grey rather than a lower alpha, which would make a
# line under a filled bar change color with what is behind it.
GRID_MAJOR_STYLE = {"color": "#b4b9c2", "linewidth": 0.7, "linestyle": "-"}
GRID_MINOR_STYLE = {"color": "#dde0e6", "linewidth": 0.5, "linestyle": "-"}

# Which grid an axis gets. Keys, compared in code.
GRID_BOTH = "both"     # major and minor
GRID_MAJOR = "major"   # major only -- what a categorical axis gets, since
                       # there is nothing between two categories to subdivide
GRID_NONE = "none"


def apply_chart_style() -> None:
    """Set the rcParams every figure in the app inherits. Called once at
    import: the alternative, a style passed per figure, is the one that
    eventually gets forgotten on the chart nobody re-reads."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": CHART_SERIF_FONTS,
        "mathtext.fontset": "stix",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.axisbelow": True,   # data over the grid, never under it
        "axes.edgecolor": "#9aa0aa",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#3c4148",
        "ytick.color": "#3c4148",
    })


apply_chart_style()


def style_axes(ax, x: str = GRID_BOTH, y: str = GRID_BOTH) -> None:
    """Put the app's grid on one Axes.

    Per axis rather than per figure because the two sides rarely want the
    same thing: a bar chart's x axis carries categories, where a minor tick
    halfway between "north" and "south" divides nothing, while its y axis is
    a scale where the minor lines are the point.

    Minor ticks are only *added* where there are none: a log axis already
    has its own decade subdivisions, and replacing those with evenly spaced
    ones would draw a grid that lies about the scale.
    """
    ax.set_axisbelow(True)
    for name, which in (("x", x), ("y", y)):
        axis = ax.xaxis if name == "x" else ax.yaxis
        if which == GRID_NONE:
            ax.grid(False, which="both", axis=name)
            continue
        ax.grid(True, which="major", axis=name, **GRID_MAJOR_STYLE)
        if which != GRID_BOTH:
            ax.grid(False, which="minor", axis=name)
            continue
        if isinstance(axis.get_minor_locator(), NullLocator) and axis.get_scale() == "linear":
            axis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="minor", axis=name, **GRID_MINOR_STYLE)


# ---- legends outside the axes -------------------------------------------
# A legend in the corner of a plot covers data whenever there is enough of
# it to need a legend at all, and on a chart comparing experiments there
# always is. Every multi-series chart therefore puts it under the axes.
#
# Two calls rather than one, and in this order:
#
#     reserve_legend_space(fig, n_entries, ncol)   # lays the figure out
#     legend_below(ax, ncol=ncol)                  # ...then fills the gap
#
# because the legend is placed at the *figure's* bottom edge, not at the
# axes': anchoring it to the axes puts it on top of the rotated tick labels
# hanging below them, which is precisely the collision this whole
# arrangement exists to avoid. A figure legend is invisible to
# tight_layout(), so the room for it is reserved first -- and reserving it
# with tight_layout's own `rect` is what guarantees no overlap, since
# tight_layout fits the axes *and every label around them* inside what is
# left.
LEGEND_MAX_COLS = 4
LEGEND_ROW_INCHES = 0.24
# A legend taller than this much of the figure means the chart has become a
# legend with a plot attached; past it the figure is better grown than the
# axes squeezed, which is what the callers do by adding
# legend_height_inches() to their own height.
LEGEND_MAX_FRACTION = 0.45


def legend_columns(n_entries: int, max_cols: int = LEGEND_MAX_COLS) -> int:
    """How many columns a legend of `n_entries` gets under the plot. Kept
    low deliberately: entries are experiment names, and four long names on
    one line run wider than the figure."""
    return max(1, min(max_cols, n_entries))


def legend_height_inches(n_entries: int, ncol: int) -> float:
    """Figure height to add for a legend placed under the axes. One row of
    slack on top of the rows themselves, so the legend does not sit flush
    against the tick labels above it."""
    rows = -(-max(0, n_entries) // max(1, ncol))
    return (rows + 1) * LEGEND_ROW_INCHES if rows else 0.0


def reserve_legend_space(fig, n_entries: int, ncol: int) -> None:
    """Lay `fig` out leaving a strip along its bottom edge for a legend.

    Replaces the fig.tight_layout() a chart would otherwise end with -- it
    *is* that call, with the strip taken out of the rectangle the axes are
    fitted into.
    """
    height = float(fig.get_size_inches()[1]) or 1.0
    fraction = min(LEGEND_MAX_FRACTION, legend_height_inches(n_entries, ncol) / height)
    fig.tight_layout(rect=(0, fraction, 1, 1))


def legend_below(ax, handles=None, labels=None, ncol: int = 1, fontsize: int = 8,
                 title: str | None = None):
    """Draw a legend at the bottom of the figure, centred under `ax`.

    Centred under the axes rather than under the figure so that a figure
    holding two plots side by side (ROC and PR) gets one legend under each,
    each one under what it describes.

    Call after reserve_legend_space(), which is what leaves the room and
    what fixes the axes position this reads.
    """
    figure = ax.get_figure()
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    position = ax.get_position()
    kwargs = {
        "loc": "lower center", "bbox_to_anchor": ((position.x0 + position.x1) / 2, 0.0),
        "bbox_transform": figure.transFigure, "ncol": max(1, ncol), "fontsize": fontsize,
        "frameon": False, "handlelength": 1.6, "columnspacing": 1.4,
    }
    if title:
        kwargs["title"] = title
        kwargs["title_fontsize"] = fontsize
    return figure.legend(handles, labels, **kwargs)
