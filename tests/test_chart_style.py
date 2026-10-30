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
    legend = cs.legend_below(ax, ncol=1)
    cs.layout_with_legends_below(ax.get_figure())

    # On the figure rather than the axes: an axes-anchored legend sits on
    # top of the rotated tick labels hanging below them.
    assert legend in ax.get_figure().legends
    assert ax.get_legend() is None
    assert ax.get_position().y0 > 0


def test_nothing_is_drawn_for_a_chart_with_no_series(ax):
    assert cs.legend_below(ax) is None


def _overlaps_the_axes(figure, ax, legend) -> bool:
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_tightbbox(renderer)   # the axes *and* their tick labels
    return legend.get_window_extent(renderer).y1 > axes_box.y0


def test_a_long_legend_grows_the_figure_rather_than_covering_the_plot():
    def chart(with_legend: bool):
        figure, ax = cs.subplots(figsize=(6, 4))
        for i in range(12):
            # Wrapped onto two lines each: what an estimate counting
            # entries got wrong, and printed over the plot.
            ax.plot([0, 1], [0, i], label=cs.legend_name(f"run_{i}_" + "x" * 140))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["a_long_class_name", "another_long_class_name"], rotation=45, ha="right")
        legend = cs.legend_below(ax) if with_legend else None
        cs.layout_with_legends_below(figure)
        return figure, ax, legend

    figure, ax, legend = chart(True)
    assert not _overlaps_the_axes(figure, ax, legend)
    # The plot kept the room it had without a legend: the figure grew instead.
    bare_figure, bare_ax, _ = chart(False)
    height = ax.get_position().height * figure.get_size_inches()[1]
    bare_height = bare_ax.get_position().height * bare_figure.get_size_inches()[1]
    assert abs(height - bare_height) < 0.05


def test_two_legends_side_by_side_never_touch_and_stay_in_the_figure():
    figure, axes = cs.subplots(1, 2, figsize=(4, 3))
    legends = []
    for ax in axes:
        ax.plot([0, 1], [0, 1], label=cs.legend_name("x" * 99) + " (AUC=0.900)")
        legends.append(cs.legend_below(ax, fontsize=7))
    cs.layout_with_legends_below(figure)
    renderer = figure.canvas.get_renderer()
    left, right = (legend.get_window_extent(renderer) for legend in legends)
    assert left.x1 < right.x0
    assert left.x0 >= 0 and right.x1 <= figure.get_size_inches()[0] * figure.dpi
    for ax, legend in zip(axes, legends):
        assert not _overlaps_the_axes(figure, ax, legend)


def test_a_legend_entry_is_wrapped_never_cut():
    long = ("finetuned_resnet_with_a_very_long_name_to_see_what_the_legend_does_with_it"
            "_and_then_some_more_words_to_go_past_the_wrap_width_v2")
    wrapped = cs.legend_name(long)
    assert "…" not in wrapped and wrapped.replace("\n", "") == long
    assert all(len(line) <= cs.LEGEND_NAME_WRAP for line in wrapped.split("\n"))
    assert cs.legend_name("short") == "short"


def test_the_pr_curve_is_drawn_on_the_unit_square_with_iso_f1_lines():
    import numpy as np
    from types import SimpleNamespace
    from lookin.core import charts

    curve = SimpleNamespace(fpr=np.array([0, .5, 1]), tpr=np.array([0, .8, 1]), roc_auc=.7,
                            recall=np.array([0, .3, .6]), precision=np.array([1, .7, .5]),
                            average_precision=.6)
    fig = charts.roc_pr_curves({"a": curve}, "cat", {"a": "#123456"})
    roc, pr = fig.axes[:2]
    assert pr.get_xlim() == (0, 1) and pr.get_ylim()[0] == 0 and roc.get_xlim() == (0, 1)
    grey = [line for line in pr.get_lines() if line.get_color() == charts.ISO_F1_COLOR]
    assert len(grey) == len(charts.ISO_F1_VALUES)
