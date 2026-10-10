"""What color each point of a projection gets, from any column of the
joined dataset x experiment frame.

Why this is not just "one color per class"
------------------------------------------
The Embeddings tab used to answer exactly one question -- "do the classes
separate?" -- because class was the only thing a point's color could carry.
But the merged frame (see AppState.merged) has always held far more than
the label: every metadata column of the dataset, plus the derived
`predicted_label`, `predicted_score` and `correct`. Painting the same
cloud by one of those turns the map into a different diagnostic without
recomputing anything: a low-confidence rim around every cluster is a
calibration problem, a cluster that is entirely one sensor is a dataset
problem, and neither is visible when color means class.

Two kinds, one contract
-----------------------
A column is either categorical (one color per value, a legend that can be
ticked off) or numeric (a viridis ramp between the column's own extremes).
The split follows the rule the rest of the app already uses (see
Dataset.column_kinds): a numeric column with only a handful of distinct
values holds labels, not a scale, and 0/1 shaded on a ramp is nonsense.

Both kinds return the same thing -- a color per point plus the [(color,
name)] legend -- so the two renderings of the map (the canvas and the
matplotlib figure) each have one code path instead of one per column kind.

Why every point also gets a *group name*
----------------------------------------
The interactive legend is not decoration, it is the display filter: ticking
an entry off hides those points in place. That needs the plotted points
labelled with the same names the legend uses, which for a numeric column
means naming the bin a point fell into. So numeric colorings get filterable
legends for free -- hiding the top decile of confidence is as useful as
hiding a class.

Stability across runs
---------------------
Colors are derived from a `reference` series -- the whole joined column --
rather than from the sampled points, so a value keeps its color when the
sample size, the class filter or the projection method changes. Comparing
two runs by eye is the whole point of the tab, and it does not survive a
palette that reshuffles itself every time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from lookin.core.data_manager import CATEGORICAL_MAX_UNIQUE, HIGH_CARDINALITY_MAX_UNIQUE

# Coloring kinds. Keys, compared in code (both renderings branch on them),
# never shown to the user as-is.
KIND_CATEGORICAL = "categorical"
KIND_NUMERIC = "numeric"

# The group a point lands in when its value is null, and the one it lands
# in when its value exists but did not make the cap below. Display strings,
# but also the names the legend filters on, so they have to be names no
# column can produce -- parentheses are not a legal number and would be a
# bizarre category.
MISSING_GROUP = "(missing)"
OTHER_GROUP = "(other)"
# Deliberately outside the qualitative palette (which is built from tab20*,
# whose grays are #7f7f7f / #c7c7c7 / #d9d9d9): these two must not be
# mistaken for a real category's color in a legend that has both.
MISSING_COLOR = "#9aa0a6"
OTHER_COLOR = "#cfd4d9"

# How many discrete steps a numeric ramp is cut into for the *canvas*
# rendering. Not a continuous ramp because a canvas Points shape carries a
# single color, so N distinct colors means N shapes -- one shape per point
# is exactly the cost the interactive map is built to avoid. The matplotlib
# rendering is given the raw values and stays continuous; eight steps is
# close enough that the two read as the same picture.
NUMERIC_BINS = 8

# Past this many distinct values a per-value legend stops being readable --
# and the palette itself runs out of colors that are still distinguishable
# as 6pt scatter points (see _build_qualitative_palette). The values past
# the cap are grouped under OTHER_GROUP rather than dropped, so the cloud
# stays whole.
MAX_CATEGORIES = 32

# A numeric column whose whole range is narrower than this is treated as
# constant: cutting it into bins would print eight identical edges.
_FLAT_RANGE = 1e-12


def _build_qualitative_palette() -> tuple:
    """~35 qualitative colors ordered darkest/most-distinct first.

    Built from matplotlib's tab20/tab20b/tab20c rather than used raw,
    because those are laid out as light/dark shade runs of the same hue:
    taken in their natural order, a 32-class legend ends up with several
    near-identical pale entries that are effectively invisible as 6pt
    scatter points on a white plot. So this walks each palette by shade
    tier -- every hue's darkest shade first, then every hue's second
    shade, ... -- which keeps adjacent category colors far apart in hue and
    pushes the washed-out shades past the MAX_CATEGORIES cap.
    """
    tab20 = plt.get_cmap("tab20").colors    # 10 hues x 2 shades (dark, light)
    tab20b = plt.get_cmap("tab20b").colors  # 5 hues x 4 shades (dark -> light)
    tab20c = plt.get_cmap("tab20c").colors
    out = []
    out.extend(tab20[0::2])  # the 10 saturated tab10-equivalent hues
    for tier in range(4):
        out.extend(tab20b[tier::4])
        out.extend(tab20c[tier::4])
    out.extend(tab20[1::2])  # palest tab20 shades, last resort only
    return tuple(out)


QUALITATIVE = _build_qualitative_palette()


def categorical_colors(n: int) -> list:
    """n distinct hex colors, stable by index (same index always maps to the
    same color) -- for per-value legends where every value needs its own
    identifiable color, as opposed to a continuous colormap."""
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for r, g, b in (QUALITATIVE[i % len(QUALITATIVE)] for i in range(n))]


def natural_sort_key(value):
    """Sort distinct values naturally: numerically when they are numbers,
    alphabetically otherwise. A mixed column (numbers with a stray
    'unknown') would raise on a plain sorted(), so numbers sort first and
    everything else follows as text.

    Numeric *strings* count as numbers, because by the time a legend is
    being ordered every value has been through group_name() -- and a
    "2, 10, 100" column read back as text sorts to "10, 100, 2", which
    reads as a bug in the legend rather than as a sort order.
    """
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return (0, float(value), "")
    text = str(value)
    try:
        number = float(text)
    except (TypeError, ValueError):
        return (1, 0.0, text)
    return (0, number, "") if np.isfinite(number) else (1, 0.0, text)


# How a float is spelled when it is being used as a *label* rather than as
# a measurement. Six significant digits, because the alternative is what
# str() gives: the dataset frame is downcast to float32 on load (halving
# the memory of a wide numeric table), and str() on a float32 prints the
# binary value in full -- an angle recorded as 34.7 comes back as
# "34.70000076293945", and a legend of those is unreadable. The cost is
# that two values agreeing to six digits share one group, which is the
# same trade-off this function already makes for text.
_FLOAT_LABEL_DIGITS = 6


def group_name(value) -> str:
    """The name a single value is filed under. Strings throughout, because
    these names are legend labels as well as group keys -- which does mean
    two values with the same text share a group, the same trade-off
    core/slices.py makes for its categorical slices."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or value is pd.NaT:
        return MISSING_GROUP
    try:
        if pd.isna(value):
            return MISSING_GROUP
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{_FLOAT_LABEL_DIGITS}g}"
    return str(value)


@dataclass
class ColorMapping:
    """One coloring of one set of plotted points.

    colors/groups are index-aligned with the points they were built from;
    legend is what the user reads, in display order, and holds only the
    groups that actually have a point on screen.
    """
    kind: str
    column: str                                  # what is being colored, for captions and titles
    colors: list = field(default_factory=list)   # (N,) hex, one per point
    legend: list = field(default_factory=list)   # [(hex, group name)]
    groups: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    # Numeric only: the raw per-point value (NaN where missing) and the
    # range the ramp spans, so the matplotlib rendering can draw a genuine
    # continuous scatter with a colorbar in the column's own units.
    values: np.ndarray | None = None
    vmin: float = 0.0
    vmax: float = 1.0
    # Filled when the cap grouped some values under OTHER_GROUP: a legend
    # silently missing half its entries is worse than no legend.
    note: str = ""
    _color_by_group: dict = field(default_factory=dict, repr=False)

    def color_of(self, value) -> str:
        """The color a value would be painted -- for the things that are
        drawn *beside* the points and have to match them, i.e. the
        prototypes. OTHER_COLOR for a value the mapping never saw, which is
        the honest answer rather than an exception."""
        return self._color_by_group.get(group_name(value), OTHER_COLOR)


def is_numeric_column(series: pd.Series) -> bool:
    """Whether a column is a scale rather than a set of labels. Same rule as
    Dataset.column_kinds: numeric dtype *and* more distinct values than a
    categorical column would have."""
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    return int(series.dropna().nunique()) > CATEGORICAL_MAX_UNIQUE


def can_be_continuous(series) -> bool:
    """Whether it even makes sense to read this column as a scale.

    Numbers can be read either way -- as a measurement or as a set of
    labels -- and which one is right is the user's call, not the dtype's
    (see default_kind). Text cannot: there is no ramp between "north" and
    "south", so no choice is offered there.
    """
    series = pd.Series(series)
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


def default_kind(series) -> str:
    """How a column is read when nobody has said otherwise.

    A guess, and one that is wrong often enough to be overridable: a float
    column with eight distinct values is a coarse measurement to one person
    (colour it on a ramp) and eight labels to another (one colour each),
    and nothing in the data says which. The guess errs towards labels,
    because a handful of distinct values shaded on a ramp is unreadable
    while a scale drawn as labels is merely ugly.
    """
    return KIND_NUMERIC if is_numeric_column(series) else KIND_CATEGORICAL


def build_mapping(values, column: str, reference=None, kind: str | None = None,
                  max_categories: int = MAX_CATEGORIES, bins: int = NUMERIC_BINS) -> ColorMapping:
    """Color the points in `values`, deciding the scheme from `reference`.

    `values` is the plotted subset; `reference` the whole column it was
    sampled from (defaults to `values`). Splitting the two is what keeps a
    value's color the same across runs -- see the module docstring.

    `kind` forces KIND_CATEGORICAL or KIND_NUMERIC instead of reading it off
    the dtype. Needed wherever the caller *knows* what the values are and
    the dtype does not say so: class labels that happen to be integers are
    still classes, and shading them on a ramp would invent an order between
    them that does not exist.
    """
    values = pd.Series(values).reset_index(drop=True)
    reference = values if reference is None else pd.Series(reference).reset_index(drop=True)
    if kind is None:
        kind = KIND_NUMERIC if is_numeric_column(reference) else KIND_CATEGORICAL
    if kind == KIND_NUMERIC:
        return _numeric_mapping(values, reference, column, bins)
    return _categorical_mapping(values, reference, column, max_categories)


def _categorical_mapping(values, reference, column: str, max_categories: int) -> ColorMapping:
    # Which values survive the cap is decided by frequency (keeping the
    # rare tail out of the legend rather than an arbitrary alphabetical
    # slice of it); the order they are then *shown* in is natural, so the
    # legend reads like a sorted list.
    counts = reference.dropna().astype(object).map(group_name).value_counts()
    kept = list(counts.index[:max_categories])
    dropped = max(0, len(counts) - len(kept))
    kept.sort(key=natural_sort_key)
    color_by_group = dict(zip(kept, categorical_colors(len(kept))))

    known = set(kept)
    groups = np.array([group_name(v) for v in values.to_numpy(dtype=object)], dtype=object)
    groups = np.array([g if g in known or g == MISSING_GROUP else OTHER_GROUP for g in groups],
                      dtype=object)
    color_by_group[MISSING_GROUP] = MISSING_COLOR
    color_by_group[OTHER_GROUP] = OTHER_COLOR
    colors = [color_by_group[g] for g in groups]

    # Only the groups with a point on screen: an empty entry is a checkbox
    # that filters nothing.
    present = set(groups.tolist())
    legend = [(color_by_group[g], g) for g in kept if g in present]
    legend += [(color_by_group[g], g) for g in (OTHER_GROUP, MISSING_GROUP) if g in present]

    note = ""
    if dropped:
        note = (f"{dropped} value(s) of '{column}' share one color: the column has more than "
                f"{max_categories} distinct values, and the rarest are grouped as {OTHER_GROUP}.")
    return ColorMapping(kind=KIND_CATEGORICAL, column=column, colors=colors, legend=legend,
                        groups=groups, note=note, _color_by_group=color_by_group)


def _format_edge(value: float) -> str:
    return f"{value:.4g}"


def _numeric_mapping(values, reference, column: str, bins: int) -> ColorMapping:
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    finite_ref = ref[np.isfinite(ref)]
    vmin = float(finite_ref.min()) if finite_ref.size else 0.0
    vmax = float(finite_ref.max()) if finite_ref.size else 1.0

    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(raw)

    bins = max(1, int(bins))
    if vmax - vmin <= _FLAT_RANGE:
        # A range too narrow to cut: float noise around a single value
        # gets past the distinct-value test above but would print eight
        # identical edges. One band is the honest answer.
        bins = 1
        idx = np.zeros(len(raw), dtype=int)
        names = [_format_edge(vmin)]
    else:
        edges = np.linspace(vmin, vmax, bins + 1)
        norm = np.clip((raw - vmin) / (vmax - vmin), 0.0, 1.0)
        idx = np.clip(np.nan_to_num(norm * bins, nan=0.0).astype(int), 0, bins - 1)
        names = [
            f"[{_format_edge(edges[b])}, {_format_edge(edges[b + 1])}"
            f"{']' if b == bins - 1 else ')'}"
            for b in range(bins)
        ]

    ramp = plt.get_cmap("viridis")
    bin_colors = [to_hex(ramp((b + 0.5) / bins)) for b in range(bins)]
    color_by_group = dict(zip(names, bin_colors))
    color_by_group[MISSING_GROUP] = MISSING_COLOR

    groups = np.where(finite, np.asarray(names, dtype=object)[idx], MISSING_GROUP).astype(object)
    colors = [color_by_group[g] for g in groups]

    present = set(groups.tolist())
    legend = [(color_by_group[n], n) for n in names if n in present]
    if MISSING_GROUP in present:
        legend.append((MISSING_COLOR, MISSING_GROUP))

    return ColorMapping(kind=KIND_NUMERIC, column=column, colors=colors, legend=legend,
                        groups=groups, values=np.where(finite, raw, np.nan),
                        vmin=vmin, vmax=vmax, _color_by_group=color_by_group)


def colorable_columns(merged: pd.DataFrame, exclude=(), score_cols=(),
                      max_unique: int = HIGH_CARDINALITY_MAX_UNIQUE) -> list:
    """The columns of a joined frame worth offering as a coloring.

    Everything is fair game except what cannot carry meaning as a color: an
    identifier or a free-text column with a distinct value per row (the same
    reason the Gallery keeps those out of its filters -- one color per row
    is a picture of nothing), a column with no value at all, and whatever
    the caller already spends elsewhere (the uuid, the crop path, and the
    label column, which the "Class" entry is).

    Ordered for the dropdown rather than alphabetically overall: the three
    derived columns first, because "was it right?" and "how sure was it?"
    are the questions this feature exists for; the raw per-class score
    columns last, because a 200-class experiment contributes 200 of them
    and they would otherwise bury every metadata column.
    """
    exclude = set(exclude)
    score_cols = [c for c in score_cols if c in merged.columns]
    derived = [c for c in ("predicted_label", "predicted_score", "correct") if c in merged.columns]

    def usable(col: str) -> bool:
        if col in exclude:
            return False
        series = merged[col]
        if not series.notna().any():
            return False
        if pd.api.types.is_numeric_dtype(series):
            return True
        return int(series.dropna().nunique()) <= max_unique

    spent = set(derived) | set(score_cols) | exclude
    metadata = sorted(c for c in merged.columns if c not in spent)
    return [c for c in derived + metadata + sorted(score_cols) if usable(c)]
