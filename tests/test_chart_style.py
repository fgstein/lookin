"""The look every chart in the app shares.

Two things are being protected. The font stack has to resolve to something
matplotlib ships with, because the machines this runs on have no route to
a font server (see CLAUDE.md) and a missing family silently falls back to
the default sans -- the chart still renders, so nothing fails, it just
stops looking like the rest of them.

And the grid has to follow the axis it is on: minor lines belong on a
scale, never between two categories, and never *replacing* the decade
subdivisions of a log axis with evenly spaced ones, which would draw a
grid that lies about the scale.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullLocator

from lookin.core import chart_style as cs


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def test_the_chart_font_is_one_matplotlib_ships_with():
    """STIXGeneral and DejaVu Serif are inside the matplotlib wheel, so
    this resolves on a machine with no fonts installed and nothing
    fetched."""
    bundled = {f.name for f in font_manager.fontManager.ttflist}
    assert bundled & set(cs.CHART_SERIF_FONTS)
    assert plt.rcParams["font.serif"][0] in bundled


def test_the_style_is_applied_at_import():
    assert plt.rcParams["font.family"] == ["serif"]
    # Data over the grid, never under it.
    assert plt.rcParams["axes.axisbelow"] is True


def test_both_grids_go_on_a_scale(ax):
    cs.style_axes(ax)
    assert isinstance(ax.yaxis.get_minor_locator(), AutoMinorLocator)
    assert ax.yaxis.get_gridlines()
    assert ax.xaxis.get_minor_locator() is not None


def test_a_categorical_axis_gets_no_minor_grid(ax):
    cs.style_axes(ax, x=cs.GRID_MAJOR)
    assert isinstance(ax.xaxis.get_minor_locator(), NullLocator)


def test_an_axis_can_be_left_without_a_grid(ax):
    """The counts axis of the slice chart: context on its own scale, whose
    lines would cross the metric's at meaningless angles."""
    cs.style_axes(ax, y=cs.GRID_NONE)
    assert not any(line.get_visible() for line in ax.yaxis.get_gridlines())


def test_a_log_axis_keeps_its_own_subdivisions(ax):
    ax.set_yscale("log")
    cs.style_axes(ax)
    assert isinstance(ax.yaxis.get_minor_locator(), LogLocator)


def test_a_legend_is_placed_under_the_axes_not_inside_them(ax):
    ax.plot([0, 1], [0, 1], label="experiment_0")
    cs.reserve_legend_space(ax.get_figure(), 1, 1)
    legend = cs.legend_below(ax, ncol=1)

    # On the figure rather than the axes: an axes-anchored legend sits on
    # top of the rotated tick labels hanging below them.
    assert legend in ax.get_figure().legends
    assert ax.get_legend() is None
    assert ax.get_position().y0 > 0


def test_nothing_is_drawn_for_a_chart_with_no_series(ax):
    assert cs.legend_below(ax) is None


def test_more_entries_need_more_room():
    assert cs.legend_height_inches(8, 2) > cs.legend_height_inches(2, 2)
    assert cs.legend_height_inches(0, 1) == 0.0
    # Columns are what keeps a dozen experiments from becoming a dozen rows.
    assert cs.legend_height_inches(8, 4) < cs.legend_height_inches(8, 1)


def test_a_legend_never_takes_the_whole_figure():
    figure = plt.figure(figsize=(6, 2))
    cs.reserve_legend_space(figure, 60, 1)
    # 60 entries would ask for ten inches out of two; the axes survive.
    assert figure.subplotpars.bottom <= cs.LEGEND_MAX_FRACTION
    plt.close(figure)
