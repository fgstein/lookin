"""What a projection point's color is allowed to mean.

The Embeddings map used to have exactly one coloring, so there was nothing
to get wrong. Now any column of the joined frame can drive it, and the ways
that goes wrong are all silent on screen: a legend that omits half its
values without saying so, a class column read as a scale because the labels
happen to be integers, a value that changes color between two runs the user
is trying to compare by eye.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.point_coloring import (
    KIND_CATEGORICAL, KIND_NUMERIC, MISSING_COLOR, MISSING_GROUP, OTHER_GROUP,
    build_mapping, colorable_columns, natural_sort_key,
)


def _labels(mapping) -> list:
    return [name for _, name in mapping.legend]


# ---- categorical --------------------------------------------------------

def test_each_value_gets_its_own_color_and_the_legend_lists_what_is_drawn():
    series = pd.Series(["north", "south", "north", "east"])
    mapping = build_mapping(series, "site")
    assert mapping.kind == KIND_CATEGORICAL
    assert _labels(mapping) == ["east", "north", "south"]
    assert len(set(mapping.colors)) == 3
    assert mapping.colors[0] == mapping.colors[2], "the same value, twice, in one color"


def test_a_value_keeps_its_color_when_the_sample_changes():
    """The tab exists to be compared against itself -- a different sample
    size or class filter that reshuffled the palette would make two runs
    of the same data look like two different datasets."""
    reference = pd.Series(["a", "b", "c", "d"] * 5)
    whole = build_mapping(reference, "site")
    part = build_mapping(reference[reference.isin(["b", "d"])], "site", reference=reference)
    assert part.color_of("b") == whole.color_of("b")
    assert part.color_of("d") == whole.color_of("d")
    assert _labels(part) == ["b", "d"], "the legend still follows what is on screen"


def test_rows_with_no_value_are_a_group_of_their_own():
    """Dropping them would leave a hole in the cloud with no explanation.
    'The column is empty exactly here' is a finding, not a gap."""
    mapping = build_mapping(pd.Series(["a", None, "b", np.nan]), "site")
    assert MISSING_GROUP in _labels(mapping)
    assert mapping.colors[1] == MISSING_COLOR
    assert mapping.colors[1] == mapping.colors[3]


def test_past_the_cap_the_rare_values_share_one_color_and_it_is_said_so():
    values = pd.Series([f"v{i}" for i in range(50)] + ["v0"] * 20)
    mapping = build_mapping(values, "session")
    assert _labels(mapping)[-1] == OTHER_GROUP
    assert len(mapping.legend) == 33, "32 named values plus the catch-all"
    assert "v0" in _labels(mapping), "the common ones are the ones kept"
    assert "session" in mapping.note and OTHER_GROUP in mapping.note


def test_a_label_column_of_integers_is_still_a_set_of_labels():
    """Forced, not inferred: shading class 1..30 on a ramp would invent an
    order between classes that does not exist."""
    labels = pd.Series(list(range(30)) * 3)
    mapping = build_mapping(labels, "class", kind=KIND_CATEGORICAL)
    assert mapping.kind == KIND_CATEGORICAL
    assert len(set(mapping.colors)) == 30


# ---- numeric ------------------------------------------------------------

def _numeric(n=200):
    return pd.Series(np.linspace(0.0, 1.0, n))


def test_a_numeric_column_becomes_a_ramp_named_in_its_own_units():
    mapping = build_mapping(_numeric(), "predicted_score")
    assert mapping.kind == KIND_NUMERIC
    assert (mapping.vmin, mapping.vmax) == (0.0, 1.0)
    assert _labels(mapping)[0].startswith("[0, 0.125")
    assert _labels(mapping)[-1].endswith("]"), "the top band has to be closed"
    assert len(set(mapping.colors)) == 8


def test_the_ramp_spans_the_whole_column_not_the_sample():
    reference = _numeric()
    low = build_mapping(reference[:20], "predicted_score", reference=reference)
    assert (low.vmin, low.vmax) == (0.0, 1.0)
    assert len(set(low.colors)) < 8, "a slice of the range uses a slice of the ramp"
    assert len(low.legend) == len(set(low.colors)), "no legend entry with nothing under it"


def test_a_numeric_column_with_holes_keeps_them_visible():
    values = pd.Series([0.0, np.nan, 1.0])
    mapping = build_mapping(values, "area", reference=_numeric())
    assert mapping.colors[1] == MISSING_COLOR
    assert np.isnan(mapping.values[1])
    assert MISSING_GROUP in _labels(mapping)


def test_a_range_too_narrow_to_cut_becomes_one_band():
    """Eight bins on a column that is one value plus float noise would
    print eight identical edges."""
    values = pd.Series([2.0 + i * 1e-15 for i in range(40)])
    mapping = build_mapping(values, "flat", kind=KIND_NUMERIC)
    assert len(mapping.legend) == 1


# ---- the legend doubles as the display filter ---------------------------

@pytest.mark.parametrize("kind", [KIND_CATEGORICAL, KIND_NUMERIC])
def test_every_point_belongs_to_a_group_the_legend_names(kind):
    """The interactive legend is checkboxes, and a checkbox that matches no
    point hides nothing -- so the two lists have to be the same list."""
    values = pd.Series(np.r_[np.linspace(0, 5, 40), [np.nan]])
    mapping = build_mapping(values, "col", kind=kind)
    assert set(mapping.groups.tolist()) == set(_labels(mapping))
    assert len(mapping.groups) == len(mapping.colors) == len(values)


def test_numbers_written_as_text_still_sort_as_numbers():
    """Every value goes through str() before it reaches a legend, and
    "2, 10, 100" sorted as text reads as a bug in the legend."""
    assert sorted(["10", "9", "100"], key=natural_sort_key) == ["9", "10", "100"]
    assert sorted(["b", "10", "a"], key=natural_sort_key) == ["10", "a", "b"]


# ---- which columns are offered at all -----------------------------------

@pytest.fixture
def joined() -> pd.DataFrame:
    return pd.DataFrame({
        "uuid": [f"u{i}" for i in range(10)],
        "crop": [f"c/{i}.png" for i in range(10)],
        "label": ["a", "b"] * 5,
        "site": ["north", "south"] * 5,
        "notes": [f"free text {i}" for i in range(10)],
        "empty": [None] * 10,
        "a": np.linspace(0, 1, 10),
        "b": np.linspace(1, 0, 10),
        "predicted_label": ["a", "b"] * 5,
        "predicted_score": np.linspace(0.5, 1, 10),
        "correct": [True, False] * 5,
    })


def test_the_derived_columns_come_first_and_the_score_columns_last(joined):
    """A 200-class experiment contributes 200 score columns; leaving them
    in alphabetical order would bury every metadata column under them."""
    offered = colorable_columns(joined, exclude={"uuid", "crop", "label"}, score_cols=["a", "b"])
    assert offered == ["predicted_label", "predicted_score", "correct", "notes", "site", "a", "b"]


def test_a_column_that_cannot_carry_a_color_is_not_offered(joined):
    """One color per row is a picture of nothing, and a column with no
    value at all is a single grey blob."""
    offered = colorable_columns(joined, exclude={"uuid", "crop", "label"},
                                score_cols=["a", "b"], max_unique=5)
    assert "notes" not in offered, "10 distinct values, cap of 5"
    assert "empty" not in offered
    assert "uuid" not in offered and "label" not in offered
