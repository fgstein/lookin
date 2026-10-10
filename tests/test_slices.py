"""Per-slice metrics.

The risks here are not in the arithmetic -- an accuracy over a boolean mask
is hard to get wrong -- they are in the slicing: bins that differ between
two experiments would compare different populations while claiming to
compare models, a dropped null slice would hide the finding it exists to
surface, and a ranking that puts a four-row slice at the top sends the user
to look at four images instead of the real regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.slices import (
    BINNING_QUANTILE, BINNING_VALUES, KIND_CATEGORICAL, KIND_NUMERIC, METRIC_ACCURACY,
    METRIC_F1, METRIC_FALSE_ALARM, METRIC_PRECISION, METRIC_RECALL, METRIC_SUPPORT,
    MISSING_SLICE, SORT_SIZE, SORT_VALUE, SORT_WORST,
    assign, build_spec, metric_interval, metric_value, slice_report, sort_rows,
    wilson_interval,
)

LABELS = ["cat", "dog"]


def _merged(labels, predictions, **columns) -> pd.DataFrame:
    """A merged frame shaped like AppState.merged builds it."""
    df = pd.DataFrame({
        "label": list(labels),
        "predicted_label": list(predictions),
        "predicted_score": np.linspace(0.6, 0.99, len(labels)),
    })
    df["correct"] = df["predicted_label"] == df["label"]
    for name, values in columns.items():
        df[name] = list(values)
    return df


# ---- building the slices ------------------------------------------------

def test_a_categorical_column_gets_one_slice_per_value():
    spec = build_spec(pd.Series(["north", "south", "north", "east"], name="site"))
    assert spec.kind == KIND_CATEGORICAL
    assert spec.names == ["east", "north", "south"]


def test_a_numeric_column_is_cut_into_quantile_bins():
    spec = build_spec(pd.Series(np.arange(100.0), name="area"), bins=4)
    assert spec.kind == KIND_NUMERIC
    assert len(spec.names) == 4
    # Quantile bins, so every bin holds roughly the same number of rows --
    # which is the whole reason they aren't equal-width.
    counts = pd.Series(assign(pd.Series(np.arange(100.0), name="area"), spec)).value_counts()
    assert counts.min() >= 24


def test_a_numeric_column_with_few_distinct_values_stays_categorical():
    """0/1/2 flags are labels written as numbers -- cutting them into
    deciles would invent nine empty bins and one holding everything."""
    spec = build_spec(pd.Series([0, 1, 0, 1, 2, 2], name="flag"), bins=10)
    assert spec.kind == KIND_CATEGORICAL
    assert spec.names == ["0", "1", "2"]


def test_repeated_values_collapse_bins_and_say_so():
    """A column that is 80% one value cannot produce ten distinct deciles;
    the table has to report that rather than silently show four bins."""
    values = pd.Series([1.0] * 80 + list(np.arange(2.0, 22.0)), name="area")
    spec = build_spec(values, bins=10, binning=BINNING_QUANTILE)
    assert len(spec.names) < 10
    assert "collapse" in spec.note


# ---- discrete columns ---------------------------------------------------

def test_a_discrete_numeric_column_gets_one_bar_per_value():
    """Twenty-five gain settings are twenty-five settings, not a scale:
    binning them would mix three of them into every bar."""
    values = pd.Series(list(range(25)) * 4, name="gain")
    spec = build_spec(values, bins=10)
    assert spec.kind == KIND_CATEGORICAL
    assert spec.names == [str(i) for i in range(25)]
    assert "one slice per value" in spec.note


def test_a_continuous_numeric_column_is_still_binned():
    values = pd.Series(np.linspace(0.0, 1.0, 500), name="area")
    spec = build_spec(values, bins=10)
    assert spec.kind == KIND_NUMERIC
    assert len(spec.names) == 10


def test_binning_can_be_forced_either_way():
    values = pd.Series(np.linspace(0.0, 1.0, 500), name="area")
    assert build_spec(values, binning=BINNING_VALUES).kind == KIND_CATEGORICAL
    flags = pd.Series([0, 1, 2, 3, 4, 5] * 20, name="flag")
    assert build_spec(flags, bins=3, binning=BINNING_QUANTILE).kind == KIND_NUMERIC


def test_a_discrete_float_column_is_named_readably():
    """The dataset frame is downcast to float32 on load, so str() on 34.7
    spells it "34.70000076293945" -- which would be the bar's label *and*
    the key rows are matched against."""
    values = pd.Series(np.array([34.7, 12.5, 34.7], dtype=np.float32), name="angle")
    spec = build_spec(values)
    assert spec.names == ["12.5", "34.7"]
    assert list(assign(values, spec)) == ["34.7", "12.5", "34.7"]


def test_every_value_lands_in_exactly_one_numeric_bin():
    """Including the maximum, which a half-open cut would push past the last
    edge and leave unassigned."""
    values = pd.Series(np.arange(50.0), name="area")
    spec = build_spec(values, bins=5)
    names = assign(values, spec)
    assert "" not in set(names)
    assert set(names) == set(spec.names)


def test_nulls_become_their_own_slice_rather_than_disappearing():
    values = pd.Series(["north", None, "south", None], name="site")
    spec = build_spec(values)
    assert spec.n_missing == 2
    assert list(assign(values, spec)).count(MISSING_SLICE) == 2


def test_the_same_spec_is_reusable_across_experiments():
    """Two experiments join to different subsets of the dataset; bins
    derived from each subset would put the same crop in '[1.2, 3.4)' for one
    and '[1.3, 3.5)' for the other, and the table would silently compare
    two different populations."""
    full = pd.Series(np.arange(100.0), name="area")
    spec = build_spec(full, bins=4)
    subset = pd.Series(np.arange(0.0, 40.0), name="area")
    names = assign(subset, spec)
    # Every name the subset produces is one the shared spec defined.
    assert set(names) <= set(spec.names)


def test_too_many_categories_are_capped_and_counted():
    values = pd.Series([f"v{i}" for i in range(30)], name="session")
    spec = build_spec(values, max_values=10)
    assert len(spec.names) == 10
    assert spec.n_values_dropped == 20


# ---- the report ---------------------------------------------------------

def test_slice_accuracy_matches_the_rows_it_covers():
    merged = _merged(
        labels=["cat", "cat", "dog", "dog"],
        predictions=["cat", "dog", "dog", "dog"],
        site=["north", "north", "south", "south"],
    )
    spec = build_spec(merged["site"])
    report = slice_report("exp", merged, "label", spec)
    by_value = {row.value: row for row in report.rows}
    assert by_value["north"].accuracy == 0.5
    assert by_value["north"].n_errors == 1
    assert by_value["south"].accuracy == 1.0
    assert report.overall.accuracy == 0.75


def test_a_slice_reports_how_many_classes_it_holds():
    """Comparing a two-class slice's accuracy to a ten-class slice's is
    comparing two different problems -- the count is what says so."""
    merged = _merged(
        labels=["cat", "cat", "dog", "bird"],
        predictions=["cat", "cat", "dog", "bird"],
        site=["north", "north", "south", "south"],
    )
    spec = build_spec(merged["site"])
    by_value = {row.value: row for row in slice_report("exp", merged, "label", spec).rows}
    assert by_value["north"].n_classes == 1
    assert by_value["south"].n_classes == 2


def test_the_missing_slice_carries_its_own_accuracy():
    merged = _merged(
        labels=["cat", "cat", "dog", "dog"],
        predictions=["cat", "cat", "cat", "cat"],
        site=["north", "north", None, None],
    )
    spec = build_spec(merged["site"])
    by_value = {row.value: row for row in slice_report("exp", merged, "label", spec).rows}
    assert by_value[MISSING_SLICE].n == 2
    assert by_value[MISSING_SLICE].accuracy == 0.0


def test_an_empty_slice_still_gets_a_row():
    """A value present in the dataset but absent from this experiment's
    joined rows is a fact worth showing -- a groupby would drop it and the
    two experiments' tables would have different rows."""
    dataset_column = pd.Series(["north", "south", "east"], name="site")
    spec = build_spec(dataset_column)
    merged = _merged(labels=["cat"], predictions=["cat"], site=["north"])
    report = slice_report("exp", merged, "label", spec)
    assert [row.value for row in report.rows] == ["east", "north", "south"]
    assert report.rows[0].n == 0


def test_a_missing_column_yields_an_empty_report_rather_than_raising():
    merged = _merged(labels=["cat"], predictions=["cat"])
    spec = build_spec(pd.Series(["north"], name="site"))
    assert slice_report("exp", merged, "label", spec).rows == []


# ---- the small-slice guard ----------------------------------------------

def test_wilson_stays_inside_zero_one_where_the_normal_interval_does_not():
    """29/30 is exactly the regime the normal approximation breaks in: it
    returns an upper bound above 1."""
    low, high = wilson_interval(29, 30)
    assert 0.0 <= low <= high <= 1.0
    assert high < 1.0


def test_a_smaller_slice_gets_a_wider_interval():
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(9, 10)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_wilson_on_an_empty_slice_is_nan_rather_than_a_crash():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_worst_first_ranking_holds_small_slices_below_measured_ones():
    merged = _merged(
        labels=["cat"] * 60 + ["dog"] * 2,
        # The big slice gets 30 of 60 right; the tiny one gets 0 of 2.
        predictions=["cat"] * 30 + ["dog"] * 30 + ["cat"] * 2,
        site=["north"] * 60 + ["tiny"] * 2,
    )
    spec = build_spec(merged["site"])
    rows = slice_report("exp", merged, "label", spec).rows
    ranked = sort_rows(rows, SORT_WORST, min_rows=30)
    assert ranked[0].value == "north"      # 0.50 over 60 rows leads
    assert ranked[-1].value == "tiny"      # 0.00 over 2 rows does not


def test_the_other_sort_orders_do_what_they_say():
    merged = _merged(
        labels=["cat"] * 6,
        predictions=["cat"] * 6,
        site=["b", "b", "b", "a", "c", "c"],
    )
    rows = slice_report("exp", merged, "label", build_spec(merged["site"])).rows
    assert [r.value for r in sort_rows(rows, SORT_VALUE)] == ["a", "b", "c"]
    assert [r.value for r in sort_rows(rows, SORT_SIZE)][0] == "b"


@pytest.mark.parametrize("bins", [2, 3, 7])
def test_bin_counts_are_honoured_when_the_data_allows(bins):
    spec = build_spec(pd.Series(np.arange(1000.0), name="area"), bins=bins)
    assert len(spec.names) == bins


# ---- metrics that depend on a class -------------------------------------

def test_per_class_counts_are_one_vs_rest_within_the_slice():
    merged = _merged(
        labels=["cat", "cat", "dog", "dog"],
        predictions=["cat", "dog", "dog", "cat"],
        site=["north", "north", "north", "north"],
    )
    spec = build_spec(merged["site"])
    row = slice_report("exp", merged, "label", spec).rows[0]
    dog = row.per_class["dog"]
    assert (dog.tp, dog.fp, dog.fn, dog.tn) == (1, 1, 1, 1)
    assert dog.support == 2 and dog.predicted == 2 and dog.negatives == 2


def test_the_false_alarm_rate_of_one_class_is_its_fpr_in_that_slice():
    """The point of the whole feature: 'FA on class 1' per sensor. Here
    every north crop that is not a dog is called one, and no south crop is."""
    merged = _merged(
        labels=["cat", "cat", "cat", "cat"],
        predictions=["dog", "dog", "cat", "cat"],
        site=["north", "north", "south", "south"],
    )
    spec = build_spec(merged["site"])
    rows = {r.value: r for r in slice_report("exp", merged, "label", spec).rows}
    assert metric_value(rows["north"], METRIC_FALSE_ALARM, "dog") == 1.0
    assert metric_value(rows["south"], METRIC_FALSE_ALARM, "dog") == 0.0
    # ...and the interval is over the two negatives it actually saw, which
    # is what says the 1.0 is not yet a measurement.
    low, high = metric_interval(rows["north"], METRIC_FALSE_ALARM, "dog")
    assert low < 0.5 and high == 1.0


def test_recall_precision_and_f1_agree_with_their_definitions():
    merged = _merged(
        labels=["cat", "cat", "dog", "dog"],
        predictions=["cat", "dog", "dog", "cat"],
        site=["north"] * 4,
    )
    row = slice_report("exp", merged, "label", build_spec(merged["site"])).rows[0]
    assert metric_value(row, METRIC_RECALL, "cat") == 0.5
    assert metric_value(row, METRIC_PRECISION, "cat") == 0.5
    assert metric_value(row, METRIC_F1, "cat") == 0.5
    assert metric_value(row, METRIC_SUPPORT, "cat") == 2.0


def test_a_class_absent_from_a_slice_reads_nan_rather_than_zero():
    """Zero would plot as a real result at the bottom of the axis and rank
    the slice as the worst one; NaN is the honest 'not measured here'."""
    merged = _merged(
        labels=["cat", "cat", "dog", "dog"],
        predictions=["cat", "cat", "dog", "dog"],
        site=["north", "north", "south", "south"],
    )
    spec = build_spec(merged["site"])
    rows = {r.value: r for r in slice_report("exp", merged, "label", spec).rows}
    assert np.isnan(metric_value(rows["north"], METRIC_RECALL, "dog"))
    assert metric_value(rows["north"], METRIC_SUPPORT, "dog") == 0.0


def test_every_slice_carries_the_same_classes():
    """A table whose columns changed from row to row could not show that a
    class vanished from a slice, which is itself a finding."""
    merged = _merged(
        labels=["cat", "cat", "dog", "dog"],
        predictions=["cat", "cat", "dog", "dog"],
        site=["north", "north", "south", "south"],
    )
    report = slice_report("exp", merged, "label", build_spec(merged["site"]))
    assert report.class_labels == ["cat", "dog"]
    for row in report.rows + [report.overall]:
        assert set(row.per_class) == {"cat", "dog"}


def test_accuracy_keeps_its_interval_through_the_shared_accessor():
    merged = _merged(
        labels=["cat"] * 4, predictions=["cat", "cat", "dog", "dog"], site=["north"] * 4,
    )
    row = slice_report("exp", merged, "label", build_spec(merged["site"])).rows[0]
    assert metric_value(row, METRIC_ACCURACY) == 0.5
    assert metric_interval(row, METRIC_ACCURACY) == (row.ci_low, row.ci_high)
