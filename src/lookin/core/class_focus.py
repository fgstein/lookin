"""Which classes a class-by-class grid shows when there are too many for it.

A confusion matrix drawn one clickable cell per (true, predicted) pair is
the right picture for a dozen classes and an impossible one for two
thousand: four million cells is past what the UI can render, and past what
anyone can read. So above FOCUS_MAX classes the interactive grid shows a
*focus set* -- at most FOCUS_MAX classes, picked by a criterion or by hand
-- and the rest of the matrix is read through two other views: the list of
the largest off-diagonal cells (top_confused_pairs), and the whole matrix
drawn as one picture.

The other per-class views of the app (Confidence, Calibration, Compare,
Embeddings, Prototypes) use the same focus set through web/views/widgets.py's
ClassFocus, each with criteria of its own; rank_by and pairs_to_focus are
the rankings they share. Everything here works over tables the tabs have
already computed, so picking another criterion costs nothing a user would
notice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, LogNorm

from lookin.core.chart_style import subplots

# Most classes an interactive class-by-class grid shows at once.
FOCUS_MAX = 20

# Criterion keys: compared in code and stored in a snapshot, never reworded.
FOCUS_MOST_CONFUSED = "most_confused"
FOCUS_WORST_F1 = "worst_f1"
FOCUS_LARGEST_SUPPORT = "largest_support"
FOCUS_MANUAL = "manual"

# A ranking on a per-class rate (worst F1, least confident, ...) is led by
# the tiniest classes -- one crop in two wrong is an F1 of 0 -- so those
# criteria only rank classes with at least this many crops, by default.
MIN_SUPPORT_DEFAULT = 10

# Criteria particular to one tab; each tab lists the ones that fit it.
FOCUS_LEAST_CONFIDENT = "least_confident"      # Confidence: lowest mean margin
FOCUS_HIGHEST_ENTROPY = "highest_entropy"      # Confidence: highest mean entropy
FOCUS_MOST_CHANGED = "most_changed"            # Compare: most fixes plus breakages
FOCUS_MOST_OVERLAPPING = "most_overlapping"    # Embeddings: most similar class pairs
FOCUS_LEAST_SEPARATED = "least_separated"      # Embeddings: smallest own-minus-other
FOCUS_WORST_MARGIN = "worst_margin"            # Prototypes: compactness minus separation
FOCUS_LEAST_PURE = "least_pure"                # Prototypes: purity of the worst subcenter
FOCUS_LOWEST_COVERAGE = "lowest_coverage"      # Calibration: least kept at the target
FOCUS_MOST_PREDICTED = "most_predicted"        # Calibration: most predictions

# (key, label) for the Metrics tab, in the order its dropdown lists them.
FOCUS_CRITERIA = (
    (FOCUS_MOST_CONFUSED, "Most confused pairs"),
    (FOCUS_WORST_F1, "Worst F1"),
    (FOCUS_LARGEST_SUPPORT, "Largest support"),
    (FOCUS_MANUAL, "Hand-picked"),
)


def all_labels(confusions: list) -> list:
    """Every class present in any of the matrices, in first-seen order --
    the order compute_metrics gave them, which is the score-column order."""
    seen: dict = {}
    for confusion in confusions:
        for label in confusion.index:
            seen.setdefault(label, None)
    return list(seen)


def summed_confusion(confusions: list) -> pd.DataFrame:
    """The matrices added cell by cell, over the union of their classes --
    what a ranking across several experiments is taken from, so the focus
    set is one set for every experiment and their grids stay comparable."""
    labels = all_labels(confusions)
    total = pd.DataFrame(0, index=labels, columns=labels, dtype=np.int64)
    for confusion in confusions:
        total = total.add(confusion.reindex(index=labels, columns=labels, fill_value=0), fill_value=0)
    return total.astype(np.int64)


def top_confused_pairs(confusion: pd.DataFrame, limit: int | None = None,
                       include_diagonal: bool = False) -> pd.DataFrame:
    """The non-zero off-diagonal cells, largest first.

    Columns: true, predicted, count, share -- share being the fraction of
    the true class's crops that went to that prediction, so a pair of 12
    in a class of 15 is not ranked as if it were 12 in a class of 3,000.
    `include_diagonal` keeps the diagonal cells in the ranking, for a
    matrix whose diagonal is news rather than the bulk (the Compare tab's
    movements: a crop arriving on the diagonal was fixed).
    """
    columns = ["true", "predicted", "count", "share"]
    if confusion.empty:
        return pd.DataFrame(columns=columns)
    counts = confusion.to_numpy()
    off = counts.copy()
    if not include_diagonal:
        np.fill_diagonal(off, 0)
    rows, cols = np.nonzero(off)
    values = off[rows, cols]
    order = np.lexsort((cols, rows, -values))  # largest first, then matrix order
    if limit is not None:
        order = order[:limit]
    rows, cols, values = rows[order], cols[order], values[order]
    support = counts.sum(axis=1)
    index = list(confusion.index)
    columns_labels = list(confusion.columns)
    return pd.DataFrame({
        "true": [index[i] for i in rows],
        "predicted": [columns_labels[j] for j in cols],
        "count": values.astype(np.int64),
        "share": values / np.maximum(support[rows], 1),
    }, columns=columns)


def rank_classes(criterion: str, confusions: list, per_class: list, limit: int = FOCUS_MAX,
                 min_support: int = 0) -> list:
    """The focus set a criterion picks, at most `limit` classes.

    `confusions` and `per_class` hold one entry per selected experiment
    (ExperimentMetrics.confusion and .per_class). FOCUS_MANUAL picks
    nothing -- the caller keeps whatever the user chose. `min_support`
    leaves out of FOCUS_WORST_F1 the classes with fewer crops than that
    (summed over the experiments' supports, averaged: one experiment's
    crop count, not several).
    """
    if criterion == FOCUS_MOST_CONFUSED:
        return _most_confused(summed_confusion(confusions), limit)
    if criterion in (FOCUS_WORST_F1, FOCUS_LARGEST_SUPPORT):
        frames = [pc[["label", "f1", "support"]] for pc in per_class if len(pc)]
        if not frames:
            return []
        table = pd.concat(frames).groupby("label", sort=False).agg(
            f1=("f1", "mean"), support=("support", "sum"), n=("support", "size")).reset_index()
        if criterion == FOCUS_WORST_F1:
            table = table[table["support"] / table["n"] >= min_support]
            # Ties are common -- every class the model never gets right has
            # an F1 of exactly 0 -- and among those the larger class is the
            # one that costs more, so it comes first.
            table = table.sort_values(["f1", "support", "label"], ascending=[True, False, True], na_position="last")
        else:
            table = table.sort_values(["support", "label"], ascending=[False, True])
        return list(table["label"][:limit])
    return []


def _most_confused(total: pd.DataFrame, limit: int) -> list:
    """Both ends of the largest confusions, walking down the pair list
    until the set is full -- so a pair that made the cut sits in the grid
    as a cell, rather than one end of it being folded into "other"."""
    return pairs_to_focus(top_confused_pairs(total)[["true", "predicted"]].itertuples(index=False), limit)


def focus_matrix(confusion: pd.DataFrame, focus: list) -> tuple:
    """(grid, other): the focus rows by the focus columns, and per focus
    row the crops predicted as any class outside the set -- so every row
    still adds up to its class's support.

    A focus class this experiment never saw gets an all-zero row and
    column rather than being dropped, so several experiments' grids keep
    the same shape and can be read side by side.
    """
    labels = list(confusion.index)
    full = confusion.reindex(index=focus, fill_value=0)
    grid = full.reindex(columns=focus, fill_value=0).astype(np.int64)
    in_focus = [c for c in labels if c in set(focus)]
    other = (full.sum(axis=1) - full[in_focus].sum(axis=1)).astype(np.int64) if len(full) else pd.Series(dtype=np.int64)
    return grid, other


def rank_by(table: pd.DataFrame, column: str, ascending: bool, limit: int = FOCUS_MAX,
            label_col: str = "label", support_col: str | None = "support",
            min_support: int = 0) -> list:
    """The first `limit` labels of `table` sorted on `column` -- NaN last,
    ties broken by the larger support (the class a tie costs more on), then
    by label so the order never depends on the input's. Rows whose support
    is under `min_support` are left out."""
    if min_support and support_col and support_col in table.columns:
        table = table[table[support_col] >= min_support]
    by = [column] + ([support_col] if support_col and support_col in table.columns else []) + [label_col]
    order = [ascending] + ([False] if len(by) == 3 else []) + [True]
    return list(table.sort_values(by, ascending=order, na_position="last")[label_col][:limit])


def pairs_to_focus(pairs, limit: int = FOCUS_MAX) -> list:
    """Both ends of each (a, b) pair, in order, until `limit` classes --
    the same walk _most_confused does, for any list of ranked pairs."""
    picked: list = []
    for a, b in pairs:
        for label in (a, b):
            if label not in picked and len(picked) < limit:
                picked.append(label)
        if len(picked) >= limit:
            break
    return picked


# ---- the whole matrix as one picture ------------------------------------
# Here rather than in ui/ because the exported report draws it too (see
# core/report_builder.py), the same way core/chart_style.py is shared.

# Up to this many classes the overview names every row and column; past it
# the labels would overlap, so only the focus classes are named.
OVERVIEW_ALL_LABELS_MAX = 60
# Blocks across an enlarged or exported overview: at one pixel per class a
# lone confusion fades out whenever the picture is shown smaller than its
# full size, and these many blocks keep it a visible square.
ZOOM_BLOCKS = 500


def _pooled(counts: np.ndarray, block: int) -> np.ndarray:
    """Sum `block` x `block` tiles, so a thumbnail keeps a lone confusion
    as a visible dot instead of averaging it away when it is downscaled."""
    if block <= 1:
        return counts
    n = len(counts)
    padded_n = -(-n // block) * block
    padded = np.zeros((padded_n, padded_n), dtype=counts.dtype)
    padded[:n, :n] = counts
    return padded.reshape(padded_n // block, block, padded_n // block, block).sum(axis=(1, 3))


def _merged_ticks(rows: list, names: list, n: int, side: float, text_size: float) -> tuple:
    """Tick positions and names for the named classes, with classes too
    close to print apart merged into one tick: "species_1614, 1615" at
    their midpoint rather than two names drawn on top of each other. Two
    ticks collide when they are nearer than one line of text, measured in
    classes along an axis about `side` inches long."""
    min_gap = n * (text_size * 1.3 / 72) / (side * 0.8)
    groups: list = []
    for row, name in zip(rows, names):
        if groups and row - groups[-1][-1][0] < min_gap:
            groups[-1].append((row, name))
        else:
            groups.append([(row, name)])
    ticks, merged = [], []
    for group in groups:
        ticks.append(sum(r for r, _ in group) / len(group))
        first = group[0][1]
        # "species_1614, 1615": the shared prefix is written once.
        rest = [_after_common_prefix(first, name) for _, name in group[1:]]
        merged.append(", ".join([first] + rest))
    return ticks, merged


def _after_common_prefix(first: str, other: str) -> str:
    """`other` without the prefix it shares with `first`, cut at a
    separator so a word is never split ("species_1615" -> "1615")."""
    common = 0
    while common < min(len(first), len(other)) and first[common] == other[common]:
        common += 1
    cut = max((i for i, ch in enumerate(other[:common]) if not ch.isalnum()), default=-1)
    return other[cut + 1:] or other


def overview_figure(confusion, focus: list, title: str, block: int = 1):
    """Where the errors are, across the whole matrix, as one heatmap.

    Only the off-diagonal cells are coloured: the diagonal is orders of
    magnitude above any confusion and, drawn, is all the eye sees -- it is
    a thin grey guide line instead. Log-scaled, with a single crop already
    a clear blue, since most confusions at this scale are one or two crops.
    `block` > 1 sums tiles of classes (see _pooled) for a thumbnail; the
    axes stay in class positions either way, so the focus classes named on
    them line up.
    """
    counts = confusion.to_numpy().copy()
    n = len(counts)
    np.fill_diagonal(counts, 0)
    errors = _pooled(counts, block)
    side = max(6.0, min(22.0, 4.0 + n / 100))
    fig, ax = subplots(figsize=(side, side))
    extent_edge = len(errors) * block - 0.5
    vmax = max(int(errors.max()) if errors.size else 1, 2)
    cmap = ListedColormap(colormaps["Blues"](np.linspace(0.35, 1.0, 256)))
    image = ax.imshow(np.ma.masked_less_equal(errors, 0), cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax),
                      interpolation="nearest", aspect="equal",
                      extent=(-0.5, extent_edge, extent_edge, -0.5))
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="#b0b0b0", linewidth=0.6, zorder=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    labels = list(confusion.index)
    # Text grows with the figure, which is up to 22 inches wide: at a fixed
    # 7pt the names were a smudge unless zoomed all the way in.
    text_size = max(7.0, side * 0.55)
    if n <= OVERVIEW_ALL_LABELS_MAX:
        ticks = [float(i) for i in range(n)]
        names = [str(l) for l in labels]
    else:
        position = {label: i for i, label in enumerate(labels)}
        rows = sorted(position[l] for l in focus if l in position)
        # A faint line along each named class's row and column, so a name
        # on the edge can be followed to its cells in the middle.
        for r in rows:
            ax.axhline(r, color="#f0a040", linewidth=0.5, alpha=0.6, zorder=0)
            ax.axvline(r, color="#f0a040", linewidth=0.5, alpha=0.6, zorder=0)
        ticks, names = _merged_ticks(rows, [str(labels[r]) for r in rows], n, side, text_size)
    ax.set_xticks(ticks, names, rotation=90, fontsize=text_size)
    ax.set_yticks(ticks, names, fontsize=text_size)
    ax.set_xlabel("predicted", fontsize=text_size * 1.2)
    ax.set_ylabel("true", fontsize=text_size * 1.2)
    ax.set_title(title, fontsize=text_size * 1.4)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.ax.tick_params(labelsize=text_size)
    colorbar.set_label("misclassified crops (log scale)" if block <= 1
                       else f"misclassified crops per {block} x {block} block (log scale)",
                       fontsize=text_size * 1.1)
    fig.tight_layout()
    return fig
