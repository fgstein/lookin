"""The display filters: which points are shown, decided column by column.

The rule these are all protecting is that a filter hides what the user
asked to hide and nothing else. The two ways to get that wrong are
symmetrical and both silent: hiding a point nobody said to hide (a value
the panel never offered, a row with no reading at all), and leaving one
visible that every filter rejects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.point_coloring import KIND_CATEGORICAL, KIND_NUMERIC, MISSING_GROUP, group_name
from lookin.core.point_filters import build_filter, combined_mask


# ---- what a column offers ------------------------------------------------

def test_a_categorical_filter_offers_the_whole_column_not_the_sample():
    """The boxes are built from every value the column has, so they do not
    rearrange themselves as another filter narrows the cloud."""
    column_filter = build_filter("site", pd.Series(["north", "south", "north", "east"]),
                                 KIND_CATEGORICAL)
    assert column_filter.values == ["east", "north", "south"]
    assert not column_filter.is_active(), "a fresh filter hides nothing"
    assert column_filter.mask(["north", "south", "east"]).all()


def test_the_rarest_values_are_the_ones_that_do_not_fit():
    values = ["common"] * 10 + ["also_common"] * 8 + ["rare"] * 2 + ["rarer"]
    column_filter = build_filter("site", pd.Series(values), KIND_CATEGORICAL, cap=2)
    assert set(column_filter.values) == {"common", "also_common"}
    assert column_filter.truncated == 2


def test_a_value_the_panel_never_offered_is_never_hidden():
    """A checkbox is the only way to hide something. A value past the cap
    has none, so it stays on the map -- the alternative is points
    disappearing with nothing on screen that explains it."""
    values = ["common"] * 10 + ["rare"]
    column_filter = build_filter("site", pd.Series(values), KIND_CATEGORICAL, cap=1)
    column_filter.hidden = {"common"}
    assert list(column_filter.mask(["common", "rare"])) == [False, True]


def test_a_numeric_filter_spans_the_column_and_starts_open():
    column_filter = build_filter("score", pd.Series([0.1, 0.5, 0.9]), KIND_NUMERIC)
    assert column_filter.bounds == (0.1, 0.9)
    assert column_filter.window == column_filter.bounds
    assert not column_filter.is_active()
    assert column_filter.mask([0.1, 0.5, 0.9]).all(), "the ends are inside the window"


def test_a_constant_column_still_gives_a_slider_two_ends():
    column_filter = build_filter("flat", pd.Series([3.0, 3.0]), KIND_NUMERIC)
    low, high = column_filter.bounds
    assert high > low and round(low, 6) == 3.0


# ---- rows with no value --------------------------------------------------

def test_missing_values_are_visible_until_someone_says_otherwise():
    column_filter = build_filter("score", pd.Series([0.1, 0.9, np.nan]), KIND_NUMERIC)
    assert column_filter.has_missing
    assert list(column_filter.mask([0.5, np.nan])) == [True, True]

    column_filter.include_missing = False
    assert list(column_filter.mask([0.5, np.nan])) == [True, False]
    assert column_filter.is_active(), "dropping the missing rows is a filter like any other"


def test_a_column_nothing_is_missing_from_says_so():
    """The control only earns its line when there is something for it to
    decide."""
    assert not build_filter("score", pd.Series([0.1, 0.9]), KIND_NUMERIC).has_missing
    assert not build_filter("site", pd.Series(["a", "b"]), KIND_CATEGORICAL).has_missing


def test_missing_is_its_own_box_rather_than_one_value_among_others():
    column_filter = build_filter("site", pd.Series(["a", None, "b"]), KIND_CATEGORICAL)
    assert column_filter.values == ["a", "b"], "not listed with the real values"
    assert column_filter.has_missing
    assert list(column_filter.mask(["a", None])) == [True, True]
    column_filter.include_missing = False
    assert list(column_filter.mask(["a", None])) == [True, False]


# ---- the window and the boxes -------------------------------------------

def test_a_window_keeps_what_is_inside_it_and_its_edges():
    column_filter = build_filter("score", pd.Series([0.0, 1.0]), KIND_NUMERIC)
    column_filter.window = (0.25, 0.75)
    assert list(column_filter.mask([0.2, 0.25, 0.5, 0.75, 0.8])) == \
        [False, True, True, True, False]
    assert column_filter.is_active()


def test_a_discrete_float_is_filtered_by_the_names_its_legend_shows():
    """Same grouping as the coloring -- a value and its legend entry have
    to be the same string, or ticking one would hide the other."""
    values = np.array([34.7, 34.7, 90.0], dtype=np.float32)
    column_filter = build_filter("roll", pd.Series(values), KIND_CATEGORICAL)
    assert column_filter.values == ["34.7", "90"], "float32 noise is not a label"
    column_filter.hidden = {"34.7"}
    assert list(column_filter.mask(values)) == [False, False, True]


# ---- several at once -----------------------------------------------------

def test_every_filter_has_to_agree():
    site = build_filter("site", pd.Series(["a", "b", "c"]), KIND_CATEGORICAL)
    site.hidden = {"c"}
    score = build_filter("score", pd.Series([0.0, 1.0]), KIND_NUMERIC)
    score.window = (0.4, 1.0)

    mask = combined_mask([site, score],
                         {"site": ["a", "b", "c", "a"], "score": [0.9, 0.1, 0.9, 0.5]}, 4)
    assert list(mask) == [True, False, False, True]


def test_a_column_with_nothing_to_compare_against_filters_nothing():
    """An empty map is never the useful answer to a column that is not
    there -- another experiment, an inference unloaded."""
    site = build_filter("site", pd.Series(["a"]), KIND_CATEGORICAL)
    site.hidden = {"a"}
    assert list(combined_mask([site], {}, 3)) == [True, True, True]


def test_what_a_filter_says_it_is_keeping():
    site = build_filter("site", pd.Series(["a", "b", "c", "d"]), KIND_CATEGORICAL)
    assert site.describe() == "", "an open filter has nothing to report"
    site.hidden = {"d"}
    assert site.describe() == "site = a, b, c"
    # Past a handful, naming them all is longer than the caption it goes
    # in -- so it counts them instead.
    many = build_filter("site", pd.Series(list("abcdefgh")), KIND_CATEGORICAL)
    many.hidden = {"a"}
    assert "7 of 8 values" in many.describe()

    score = build_filter("score", pd.Series([0.0, 1.0, np.nan]), KIND_NUMERIC)
    score.window = (0.25, 0.5)
    assert score.describe() == "score in [0.25, 0.5]"
    score.include_missing = False
    assert score.describe().endswith("(no missing)")


def test_resetting_puts_everything_back():
    column_filter = build_filter("score", pd.Series([0.0, 1.0, np.nan]), KIND_NUMERIC)
    column_filter.window = (0.2, 0.3)
    column_filter.include_missing = False
    column_filter.reset()
    assert not column_filter.is_active()
    assert column_filter.mask([0.0, 1.0, np.nan]).all()
