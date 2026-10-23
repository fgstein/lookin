"""Every chart Lookin draws, as a function of the numbers it shows.

The tabs, the exported report and the MCP endpoint all draw through here,
so a chart an agent asks for is the very picture the tab shows -- same
binning, same colours, same captions. Each function takes plain inputs
(the core result objects, a dict of experiment name -> colour) and returns
a matplotlib Figure, never a widget or a file: the caller decides whether
it becomes an <img>, a slide or a PNG on disk (see figure_png).

Nothing here computes a metric. What a chart shows is computed in
core/metrics.py, core/calibration.py, core/slices.py and friends; this only
lays it out, which is what makes "the agent's chart and the tab's chart
disagree" impossible rather than unlikely.
"""
from __future__ import annotations

import io

import numpy as np

from lookin.core import class_focus as cf
from lookin.core import slices as sl
from lookin.core.chart_style import (
    GRID_MAJOR, GRID_NONE, legend_below, legend_columns, legend_height_inches, reserve_legend_space,
    short_name, style_axes, subplots,
)

# One series in a single-series chart (a column histogram).
BAR_COLOR = "#2550C8"
# The wrong predictions and the mean confidence, in every calibration chart.
WRONG_COLOR = "#d62728"

# Per-class metrics that count rows rather than score them: no [0, 1] axis.
COUNT_METRICS = {"support"}

# Past this many distinct values a categorical column is drawn as sorted
# sizes rather than one bar per value; the RANKED_NAMED largest and
# smallest are named under it.
RANKED_CATEGORIES_MIN = 60
RANKED_NAMED = 5
# "Fewer than this many" is the count reported for the long tail.
RANKED_SMALL = 10

# Slices drawn in a slice chart. A hundred-value categorical column is
# legitimate in a table (it scrolls) but unreadable as a chart.
MAX_CHART_SLICES = 30
# The count bars of a slice chart are background, not a series: a neutral
# grey that no experiment color can be confused with, and an axis labelled
# in the same grey so the eye reads it as the context it is.
COUNT_BAR_COLOR = "#ccd1d9"
COUNT_AXIS_COLOR = "#7b828d"

# A confusion matrix is drawn cell by cell, with its counts, up to this
# many classes; past it, the focus classes are, plus the whole-matrix
# overview (class_focus.overview_figure).
CONFUSION_CELLS_MAX = cf.FOCUS_MAX


def figure_png(fig, dpi: int = 130) -> bytes:
    """A figure as PNG bytes, white background (as every chart is shown)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


# ---- per-class metrics ----------------------------------------------------

def class_tick_labels(labels: list, per_class_by_name: dict) -> tuple:
    """"car" -> "car [126]", where 126 is that class's support (ground-truth
    row count) in the first experiment that has it. Also returns the labels
    whose support is not the same number in every experiment, so the caller
    can flag that the single count shown is not universal -- this happens
    when experiments don't share the exact same uuid overlap with the
    dataset."""
    tick_labels = []
    uneven = set()
    for label in labels:
        supports = [
            int(pc.loc[label, "support"])
            for pc in per_class_by_name.values()
            if label in pc.index and pc.loc[label, "support"] == pc.loc[label, "support"]
        ]
        if supports:
            tick_labels.append(f"{label} [{supports[0]}]")
            if len(set(supports)) > 1:
                uneven.add(label)
        else:
            tick_labels.append(label)
    return tick_labels, uneven


def per_class_bars(per_class_by_name: dict, metric: str, labels: list, colors: dict) -> tuple:
    """(figure, uneven_support): one group of bars per class, one bar per
    experiment. `per_class_by_name` maps an experiment to its per-class
    table indexed by label (ExperimentMetrics.per_class.set_index("label"))."""
    n_exp = len(per_class_by_name)
    fig_width = max(6.0, min(18.0, len(labels) * (0.5 + 0.13 * n_exp)))
    legend_cols = legend_columns(n_exp)
    fig, ax = subplots(figsize=(fig_width, 4.3 + legend_height_inches(n_exp, legend_cols)))
    width = 0.8 / max(n_exp, 1)
    x = np.arange(len(labels))
    for i, (name, per_class) in enumerate(per_class_by_name.items()):
        values = [per_class[metric].get(label, np.nan) for label in labels]
        offset = (i - (n_exp - 1) / 2) * width
        ax.bar(x + offset, values, width, label=short_name(name), color=colors.get(name))
    ax.set_xticks(x)
    tick_labels, uneven = class_tick_labels(labels, per_class_by_name)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    if metric in COUNT_METRICS:
        ax.set_ylabel(f"{metric} (rows)")
    else:
        ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class {metric} by experiment")
    style_axes(ax, x=GRID_NONE)
    reserve_legend_space(fig, n_exp, legend_cols)
    legend_below(ax, ncol=legend_cols)
    return fig, uneven


def roc_pr_curves(curves_by_name: dict, target: str, colors: dict):
    """ROC and Precision-Recall side by side, one line per experiment.
    `curves_by_name` holds drawable CurveData (no disabled_reason)."""
    n_curves = len(curves_by_name)
    legend_cols = min(2, n_curves)
    fig, (ax1, ax2) = subplots(1, 2, figsize=(11, 4.4 + legend_height_inches(n_curves, legend_cols)))
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    for name, curve in curves_by_name.items():
        color = colors.get(name)
        ax1.plot(curve.fpr, curve.tpr, color=color, label=f"{short_name(name)} (AUC={curve.roc_auc:.3f})")
        ax2.plot(curve.recall, curve.precision, color=color, label=f"{short_name(name)} (AP={curve.average_precision:.3f})")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(f"ROC curve — {target}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall curve — {target}")
    style_axes(ax1)
    style_axes(ax2)
    reserve_legend_space(fig, n_curves, legend_cols)
    for ax in (ax1, ax2):
        legend_below(ax, ncol=legend_cols, fontsize=7)
    return fig


# ---- confusion matrices ---------------------------------------------------

def confusion_matrix(confusion, title: str, focus: list | None = None):
    """The matrix cell by cell with its counts. With `focus`, only those
    classes, plus an "(other)" column counting the crops predicted as any
    class outside them -- the way a many-class matrix is shown."""
    if focus is not None:
        grid, other = cf.focus_matrix(confusion, focus)
        shown = grid.copy()
        shown["(other)"] = other
        n = len(grid)
        counts = shown.to_numpy()
        columns, rows = list(shown.columns), list(shown.index)
        scale = grid.to_numpy().max() or 1
        coloured = np.where(np.arange(n + 1) < n, counts, 0)
    else:
        counts = confusion.to_numpy()
        columns, rows = list(confusion.columns), list(confusion.index)
        scale = counts.max() or 1
        coloured = counts
    side = max(4.0, 0.42 * len(columns) + 2.2)
    fig, ax = subplots(figsize=(side, side * 0.9))
    ax.imshow(coloured, cmap="Blues")
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for i in range(len(rows)):
        for j in range(len(columns)):
            value = int(counts[i, j])
            if value:
                dark = coloured[i, j] > scale * 0.55
                ax.text(j, i, str(value), ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")
    fig.tight_layout()
    return fig


# ---- calibration ----------------------------------------------------------

def reliability_diagram(name: str, data, color: str):
    """A bar per confidence bin with its actual accuracy, the mean
    confidence per bin in red, the diagonal for perfect calibration; below,
    the bins' share of all predictions on a log scale (the same binning
    ECE/MCE are computed from). None when the scores are not probabilities
    (data.disabled_reason says why)."""
    if data.disabled_reason:
        return None
    width = 1.0 / len(data.bin_centers)
    fig, (ax1, ax2) = subplots(2, 1, figsize=(4.6, 5.4),
                               gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect calibration")
    ax1.bar(data.bin_centers, np.nan_to_num(data.bin_accuracy), width=width * 0.9,
            color=color, edgecolor="black", linewidth=0.4, label="accuracy")
    ax1.plot(data.bin_centers, data.bin_confidence, color=WRONG_COLOR, marker="o",
             markersize=3, linewidth=1.2, label="mean confidence")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"{short_name(name)} — ECE={data.ece:.3f} · MCE={data.mce:.3f}", fontsize=11)
    ax1.legend(fontsize=7, loc="upper left")

    ax2.bar(data.bin_centers, data.bin_fraction, width=width * 0.9, color=color)
    ax2.set_yscale("log")
    ax2.set_xlabel("predicted confidence")
    ax2.set_ylabel("fraction\n(log)", fontsize=9)
    ax2.set_xlim(0, 1)
    style_axes(ax1)
    style_axes(ax2)
    fig.tight_layout()
    return fig


def confidence_split_histogram(name: str, hist, color: str):
    """Where the correct and the wrong predictions sit on the confidence
    axis: two separated humps mean an operating point exists."""
    centers = (hist.edges[:-1] + hist.edges[1:]) / 2
    width = (hist.edges[1] - hist.edges[0]) * 0.9
    fig, ax = subplots(figsize=(4.6, 3.4))
    ax.bar(centers, hist.correct, width=width, color=color, label="correct")
    ax.bar(centers, hist.wrong, width=width, bottom=hist.correct, color=WRONG_COLOR, label="wrong")
    ax.set_yscale("log")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("crops (log)")
    ax.set_title(f"{short_name(name)} — confidence of right vs. wrong", fontsize=10)
    style_axes(ax)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def risk_coverage(curves_by_name: dict, colors: dict):
    """Risk-coverage, one line per experiment (calibration.RiskCoverage)."""
    n = len(curves_by_name)
    fig, ax = subplots(figsize=(7.4, 4.4 + legend_height_inches(n, legend_columns(n))))
    for name, curve in curves_by_name.items():
        if not len(curve.coverage):
            continue
        ax.plot(curve.coverage, curve.risk, color=colors.get(name),
                label=f"{short_name(name)} (AURC={curve.aurc:.4f})")
    ax.set_xlabel("Coverage -- share of crops the model is allowed to decide")
    ax.set_ylabel("Selective risk -- error rate among those")
    ax.set_title("Risk-coverage: what abstaining buys")
    ax.set_xlim(0, 1)
    style_axes(ax)
    # One entry per experiment, under the plot: the interesting part of a
    # risk-coverage curve is its right-hand end.
    reserve_legend_space(fig, n, legend_columns(n))
    legend_below(ax, ncol=legend_columns(n))
    return fig


# ---- dataset columns ------------------------------------------------------

def column_chart(column: str, series, is_numeric: bool) -> tuple:
    """(figure, notes) for one dataset column: a histogram for a numeric
    column, a bar per value for a categorical one, and past
    RANKED_CATEGORIES_MIN values the sorted sizes on a log scale with the
    extremes named in the notes."""
    fig, ax = subplots(figsize=(9.5, 3.2))
    notes = []
    if is_numeric:
        # An inf has no bin: numpy refuses the whole histogram. Drawn
        # without, and said so.
        finite = series[np.isfinite(series.to_numpy(dtype=float))]
        if len(finite) < len(series):
            notes.append(f"{len(series) - len(finite):,} infinite value(s) not drawn.")
        ax.hist(finite, bins=40, color=BAR_COLOR, edgecolor="white")
        ax.set_title(f"{column}", fontsize=12)
        ax.set_ylabel("count", fontsize=10)
        style_axes(ax)
    elif series.nunique() > RANKED_CATEGORIES_MIN:
        counts = series.value_counts().sort_values(ascending=False)
        ranks = np.arange(1, len(counts) + 1)
        ax.fill_between(ranks, counts.values, step="mid", color=BAR_COLOR, alpha=0.35)
        ax.step(ranks, counts.values, where="mid", color=BAR_COLOR)
        ax.set_yscale("log")
        ax.set_xlim(0.5, len(counts) + 0.5)
        ax.set_title(f"{column} -- {len(counts):,} values, largest first", fontsize=12)
        ax.set_xlabel("rank", fontsize=10)
        ax.set_ylabel("count (log scale)", fontsize=10)
        style_axes(ax)

        def named(part) -> str:
            return ", ".join(f"{label} ({int(n):,})" for label, n in part.items())

        notes = [f"Largest: {named(counts.head(RANKED_NAMED))}",
                 f"Smallest: {named(counts.tail(RANKED_NAMED)[::-1])}",
                 f"Median {int(counts.median()):,} per value; "
                 f"{int((counts < RANKED_SMALL).sum()):,} value(s) with fewer than {RANKED_SMALL}."]
    else:
        counts = series.value_counts().sort_values(ascending=False)
        ax.bar(counts.index.astype(str), counts.values, color=BAR_COLOR)
        ax.set_title(f"{column}", fontsize=12)
        ax.set_ylabel("count", fontsize=10)
        for label in ax.get_xticklabels():
            label.set_fontsize(9)
            if len(counts) > 12:
                label.set_rotation(45)
                label.set_ha("right")
        style_axes(ax, x=GRID_NONE)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig, notes


# ---- slices ---------------------------------------------------------------

def _counts_agree(rows_by_name: dict, names: list, values: list) -> bool:
    """Whether every experiment holds the same number of crops in every
    slice, which is the normal case -- n comes from the dataset, not from
    the model. They only diverge when the experiments join different
    subsets of it (partial uuid overlap)."""
    if not values:
        return True
    return all(len({rows_by_name[name][v].n for name in names}) == 1 for v in values)


def slice_chart(ordered_values: list, reports: dict, metric: str, metric_label: str, column: str,
                colors: dict, class_label=None) -> tuple:
    """(figure, notes): a metric per slice of `column`, one marked line per
    experiment (with its interval), over grey bars counting the crops in
    each slice -- reading a 0.42 without seeing that it rests on nine crops
    is the mistake this arrangement makes hard. `reports` maps an
    experiment to its slices.SliceReport."""
    values = ordered_values[:MAX_CHART_SLICES]
    names = list(reports)
    rows_by_name = {name: {row.value: row for row in report.rows} for name, report in reports.items()}
    n_exp = len(names)
    fig_width = max(6.0, min(18.0, len(values) * (0.5 + 0.06 * n_exp)))
    counts_agree = _counts_agree(rows_by_name, names, values)
    legend_entries = n_exp + (n_exp if not counts_agree else 1)
    legend_cols = legend_columns(legend_entries)
    fig, ax_counts = subplots(figsize=(fig_width, 4.8 + legend_height_inches(legend_entries, legend_cols)))
    ax_metric = ax_counts.twinx()
    ax_metric.yaxis.tick_left()
    ax_metric.yaxis.set_label_position("left")
    ax_counts.yaxis.tick_right()
    ax_counts.yaxis.set_label_position("right")
    x = np.arange(len(values))
    if counts_agree:
        ax_counts.bar(x, [rows_by_name[names[0]][v].n for v in values], width=0.68, color=COUNT_BAR_COLOR,
                      label="crops in slice")
    else:
        bar_width = 0.68 / max(n_exp, 1)
        for i, name in enumerate(names):
            ax_counts.bar(x + (i - (n_exp - 1) / 2) * bar_width, [rows_by_name[name][v].n for v in values],
                          bar_width, color=colors.get(name), alpha=0.28,
                          label=f"crops in slice ({short_name(name)})")
    ax_counts.set_ylabel("crops in slice", color=COUNT_AXIS_COLOR, fontsize=9)
    ax_counts.tick_params(axis="y", colors=COUNT_AXIS_COLOR, labelsize=8)
    ax_counts.set_ylim(0, max([rows_by_name[name][v].n for name in names for v in values] or [1]) * 1.15)
    for name in names:
        rows = [rows_by_name[name][v] for v in values]
        heights = [sl.metric_value(row, metric, class_label) for row in rows]
        color = colors.get(name)
        intervals = [sl.metric_interval(row, metric, class_label) for row in rows]
        if any(low == low for low, _ in intervals):
            lows = [max(0.0, hh - low) if low == low else 0.0 for hh, (low, _) in zip(heights, intervals)]
            highs = [max(0.0, high - hh) if high == high else 0.0 for hh, (_, high) in zip(heights, intervals)]
            ax_metric.errorbar(x, heights, yerr=[lows, highs], label=short_name(name), color=color, marker="o",
                               markersize=5, linewidth=1.8, capsize=3, elinewidth=0.9, ecolor=color, alpha=0.95)
        else:
            ax_metric.plot(x, heights, label=short_name(name), color=color, marker="o", markersize=5, linewidth=1.8)
        overall = sl.metric_value(reports[name].overall, metric, class_label)
        if overall == overall:
            ax_metric.axhline(overall, color=color, linestyle="--", linewidth=1, alpha=0.6)
    ax_metric.set_ylabel(metric_label)
    if metric not in sl.COUNT_METRICS and metric != sl.METRIC_MEAN_CONFIDENCE:
        ax_metric.set_ylim(0, 1.05)
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(values, rotation=45, ha="right", fontsize=8)
    ax_counts.set_xlim(-0.7, len(values) - 0.3)
    ax_metric.set_title(f"{metric_label} by {column} (dashed line: all rows)")
    style_axes(ax_counts, x=GRID_MAJOR, y=GRID_NONE)
    style_axes(ax_metric, x=GRID_NONE)
    handles = ax_metric.get_legend_handles_labels()[0] + ax_counts.get_legend_handles_labels()[0]
    labels = ax_metric.get_legend_handles_labels()[1] + ax_counts.get_legend_handles_labels()[1]
    reserve_legend_space(fig, legend_entries, legend_cols)
    legend_below(ax_metric, handles, labels, ncol=legend_cols)
    notes = ["Bars are the number of crops in each slice (right axis); the marked lines are the "
             "metric (left axis). The line follows the order chosen, so it traces a real trend "
             "only when the slices are ordered by value."]
    if not counts_agree:
        notes.append("The experiments do not cover the same rows, so each gets its own count bars.")
    if len(ordered_values) > MAX_CHART_SLICES:
        notes.append(f"Chart shows the first {MAX_CHART_SLICES} slices in this order, of "
                     f"{len(ordered_values)}.")
    return fig, notes


# ---- embeddings -----------------------------------------------------------

def projection_scatter(coords, groups, title: str, colors_by_group: dict | None = None,
                       prototypes=None, max_legend: int = 20):
    """A 2D projection as a scatter, one colour per group (the class, or
    any column). `prototypes` is an optional (coords, groups) pair drawn as
    stars over the cloud. Past `max_legend` groups the legend is left out:
    a legend of a thousand classes is a wall, not a key."""
    from lookin.core.point_coloring import categorical_colors

    coords = np.asarray(coords, dtype=float)
    groups = np.asarray([str(g) for g in groups])
    names = sorted(set(groups.tolist()))
    colors = dict(colors_by_group or {})
    missing = [n for n in names if n not in colors]
    colors.update(zip(missing, categorical_colors(len(missing))))
    fig, ax = subplots(figsize=(7.2, 6.6))
    size = max(2.0, min(14.0, 6000.0 / max(len(coords), 1)))
    for name in names:
        mask = groups == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=size, color=colors[name], label=name,
                   alpha=0.8, linewidths=0)
    if prototypes is not None:
        p_coords, p_groups = prototypes
        p_coords = np.asarray(p_coords, dtype=float)
        for (px, py), g in zip(p_coords, p_groups):
            ax.scatter([px], [py], marker="*", s=180, color=colors.get(str(g), "#333333"),
                       edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(names) <= max_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False, markerscale=2.5)
    fig.tight_layout()
    return fig
