"""Which plotted points are shown, decided column by column and
independently of what the color means.

Why this is not the legend
--------------------------
The Embeddings tab has had one display filter since the map existed: the
legend, where ticking an entry off hides its points in place. It works, and
it is the fastest gesture on the tab -- but it can only ever filter on the
*one* column the color is currently coming from. "Show me the low-confidence
crops of one sensor, colored by class" is three columns and one color, and
the legend can express exactly one of them.

So the filters live here instead: any number of columns at once, none of
them tied to the coloring, and each one surviving a recolor (which rebuilds
the map, and with it the legend's own hidden set).

What a filter is
----------------
One column, and either
  * a set of values that are hidden -- for a column read as labels, and
  * a window [low, high] -- for a column read as a scale.
Which of the two follows the column's kind, and that kind is the user's to
override (see point_coloring.default_kind): a float column with eight
distinct values is a set of labels to one person and a coarse measurement
to another, and only they know which.

Rows with no value are never filtered out silently. They get their own
entry -- a "(missing)" box on a categorical filter, an `include_missing`
switch on a numeric one -- because "these crops have no reading for this
column" is a finding as often as it is an accident, and a filter that
quietly dropped them would hide it. It is also the common case rather than
an edge one now that a map can carry crops from a simple inference, which
has none of the dataset's columns.

Everything here is plain numpy/pandas over arrays of values, so the rules
above are testable without a UI -- the same split the rest of core keeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lookin.core.point_coloring import (
    KIND_CATEGORICAL, KIND_NUMERIC, MISSING_GROUP, group_name, natural_sort_key,
)

# How many values one categorical filter offers. Past this the panel stops
# being a filter and becomes a wall of checkboxes -- and a column with
# hundreds of distinct values is not what anyone filters on by ticking.
# Values past the cap are simply not offered, and points carrying them are
# never hidden by this filter (see ColumnFilter.mask): a value nobody can
# see in the panel must not be able to disappear from the map either.
MAX_FILTER_VALUES = 60


@dataclass
class ColumnFilter:
    """One column's filter, and the state of its controls.

    `values` (categorical) and `bounds` (numeric) describe what the column
    *offers* and are computed from the whole column, not from the points
    currently on screen: they must not move as other filters narrow the
    cloud, or the panel would rearrange itself under the pointer.
    """
    column: str
    kind: str
    # Categorical: every value the panel offers, in display order, and the
    # ones currently unticked.
    values: list = field(default_factory=list)
    hidden: set = field(default_factory=set)
    # How many distinct values did not make MAX_FILTER_VALUES.
    truncated: int = 0
    # Numeric: the column's own full range, and the window kept of it.
    bounds: tuple = (0.0, 1.0)
    window: tuple = (0.0, 1.0)
    # Both kinds: whether rows with no value for this column are kept.
    include_missing: bool = True
    # Whether any row at all has no value here -- the control is only worth
    # drawing when it is.
    has_missing: bool = False

    def is_active(self) -> bool:
        """Whether this filter actually hides anything. A filter that is on
        but hides nothing still costs a line of the panel, and the caption
        under the map counts on knowing the difference."""
        if self.has_missing and not self.include_missing:
            return True
        if self.kind == KIND_NUMERIC:
            return tuple(self.window) != tuple(self.bounds)
        return bool(self.hidden)

    def reset(self) -> None:
        self.hidden = set()
        self.window = self.bounds
        self.include_missing = True

    def mask(self, values) -> np.ndarray:
        """(N,) bool: which of these values this filter keeps.

        `values` are the plotted points' values for this column, in the
        order they are drawn. Anything the panel does not offer is kept --
        see MAX_FILTER_VALUES."""
        values = pd.Series(values).reset_index(drop=True)
        if not len(values):
            return np.zeros(0, dtype=bool)
        if self.kind == KIND_NUMERIC:
            numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            known = np.isfinite(numbers)
            low, high = float(self.window[0]), float(self.window[1])
            keep = known & (numbers >= low) & (numbers <= high)
            return keep | (~known & self.include_missing)
        groups = np.array([group_name(v) for v in values.to_numpy(dtype=object)], dtype=object)
        keep = ~np.isin(groups, list(self.hidden)) if self.hidden else np.ones(len(groups), dtype=bool)
        if not self.include_missing:
            keep &= groups != MISSING_GROUP
        return keep

    def describe(self) -> str:
        """One short phrase for the caption under the map -- what this
        filter is keeping, not that it exists."""
        if not self.is_active():
            return ""
        if self.kind == KIND_NUMERIC:
            text = f"{self.column} in [{self.window[0]:.4g}, {self.window[1]:.4g}]"
        else:
            kept = [v for v in self.values if v not in self.hidden]
            if len(kept) <= 3:
                text = f"{self.column} = {', '.join(kept) if kept else 'nothing'}"
            else:
                text = f"{self.column}: {len(kept)} of {len(self.values)} values"
        if self.has_missing and not self.include_missing:
            text += " (no missing)"
        return text


def build_filter(column: str, reference, kind: str, cap: int = MAX_FILTER_VALUES) -> ColumnFilter:
    """A filter over `reference` (the whole column, across every pool that
    can put a point on the map), showing everything.

    Built from the whole column rather than from the sample so that the
    boxes and the slider ends stay where they are as the sample size, the
    class picker or another filter changes what is actually drawn."""
    series = pd.Series(reference).reset_index(drop=True)
    if kind == KIND_NUMERIC:
        numbers = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite = numbers[np.isfinite(numbers)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        if hi <= lo:
            # A constant column: a slider needs two different ends, and a
            # hair of width is enough to give it one without moving the
            # value the user reads.
            hi = lo + 1e-9
        return ColumnFilter(
            column=column, kind=KIND_NUMERIC, bounds=(lo, hi), window=(lo, hi),
            has_missing=bool((~np.isfinite(numbers)).any()),
        )

    groups = series.map(group_name)
    counts = groups.value_counts()
    present = [v for v in counts.index if v != MISSING_GROUP]
    # Which values survive the cap is decided by frequency (so the rare
    # tail is what goes, not an arbitrary alphabetical slice); the order
    # they are then shown in is natural, so the panel reads like a list.
    kept = sorted(present[:cap], key=natural_sort_key)
    has_missing = MISSING_GROUP in counts.index
    return ColumnFilter(
        column=column, kind=KIND_CATEGORICAL, values=kept,
        truncated=max(0, len(present) - len(kept)), has_missing=has_missing,
    )


def combined_mask(filters, values_by_column: dict, n: int) -> np.ndarray:
    """(n,) bool: the points every filter keeps.

    AND rather than OR, because each filter answers a different question
    about the same point and a point has to satisfy all of them to be the
    thing the user asked to see. A column with no values to hand (nothing
    on the map carries it) filters nothing rather than everything -- an
    empty map is never the useful answer to a missing column.
    """
    keep = np.ones(n, dtype=bool)
    for column_filter in filters:
        values = values_by_column.get(column_filter.column)
        if values is None or not len(values):
            continue
        keep &= column_filter.mask(values)
    return keep
