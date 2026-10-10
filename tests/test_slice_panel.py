"""The Metrics tab's slice panel, rendered without a browser.

core/slices.py is covered separately; what is checked here is the wiring
that only exists in the view: which columns it offers, that a render
produces a real table rather than an exception, that the caches follow a
rename, and that picking a column costs nothing until it is picked.

Same harness as scripts/benchmark_scaling.py: every .update() in the view
is already guarded by `if control.page:`, so a context with no page renders
the whole thing and skips the pushes.
"""
from __future__ import annotations

import flet as ft
import pytest

from lookin.core import data_manager as dm
from lookin.core import slices as sl
from lookin.ui.slice_panel import METRIC_ACCURACY, METRIC_MACRO_F1, SlicePanel


class FakeCtx:
    def __init__(self, state):
        self.state = state
        self.page = None
        self.image_zoom = None
        self.display_lut = "none"


@pytest.fixture
def panel(loaded_state):
    p = SlicePanel(FakeCtx(loaded_state))
    p.build()
    return p


def _controls(panel) -> list:
    return list(panel.body.controls)


def test_only_columns_the_app_can_build_controls_for_are_offered(panel, monkeypatch):
    """The same rule the Gallery's filters follow: a free-text column would
    produce one slice per row. The cap is lowered here because the fixture's
    'note' column only has six distinct values."""
    monkeypatch.setattr(dm, "HIGH_CARDINALITY_MAX_UNIQUE", 4)
    offered = panel._sliceable_columns()
    assert set(offered) == {"flag", "site", "area"}
    assert "note" not in offered      # high cardinality
    assert "label" not in offered     # not metadata


def test_nothing_is_computed_until_a_column_is_picked(panel):
    panel.set_experiments(["exp"])
    assert _controls(panel) == []
    assert panel._report_cache == {}


def test_picking_a_column_renders_a_table_and_a_chart(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    panel._render()
    assert len(_controls(panel)) >= 2
    assert panel._report_cache


def test_the_rendered_table_holds_one_row_per_slice_plus_the_reference(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    panel._render()
    tables = _find(panel.body, ft.DataTable)
    assert tables, "no DataTable rendered"
    # north / south / east, plus the 'all rows' reference line.
    assert len(tables[0].rows) == 4


def test_every_metric_renders(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "area"
    for metric in (METRIC_ACCURACY, METRIC_MACRO_F1):
        panel.metric_dropdown.value = metric
        panel._render()
        assert _find(panel.body, ft.DataTable)


def test_a_numeric_column_is_binned_and_the_bin_count_is_honoured(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "area"
    # The fixture's 'area' has six distinct values, so automatic binning
    # would give it one bar per value -- this is the other branch.
    panel.binning_dropdown.value = sl.BINNING_QUANTILE
    panel.bins_field.value = "3"
    panel._render()
    tables = _find(panel.body, ft.DataTable)
    assert len(tables[0].rows) == 4      # 3 bins + the reference line


def test_a_discrete_column_gets_one_row_per_value_rather_than_bins(panel):
    """Six crop areas are six values, not a scale -- the default has to show
    all six rather than cutting them into ten empty deciles."""
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "area"
    panel._render()
    tables = _find(panel.body, ft.DataTable)
    assert len(tables[0].rows) == 7      # 6 values + the reference line


def test_a_per_class_metric_renders_and_shows_its_class_picker(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    panel.metric_dropdown.value = sl.METRIC_FALSE_ALARM
    panel.class_dropdown.value = "dog"
    panel._sync_controls()
    panel._render()
    assert panel.class_dropdown.visible
    assert _find(panel.body, ft.DataTable)


def test_the_class_picker_is_hidden_for_a_metric_that_takes_no_class(panel):
    panel.set_experiments(["exp"])
    panel.metric_dropdown.value = METRIC_ACCURACY
    panel._sync_controls()
    assert not panel.class_dropdown.visible


def test_worst_first_follows_the_metric_being_shown(panel):
    """A table of false alarm rates ranked by accuracy would lead with a
    slice that is not the one being looked at."""
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    panel.metric_dropdown.value = sl.METRIC_FALSE_ALARM
    panel.class_dropdown.value = "dog"
    panel._render()
    tables = _find(panel.body, ft.DataTable)
    first_slice = tables[0].rows[0].cells[0].content.value
    # 'east' is where the one crop wrongly called a dog is (see conftest).
    assert first_slice == "east"


def test_an_out_of_range_bin_count_is_clamped_not_rejected(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "area"
    panel.bins_field.value = "9999"
    panel._render()
    assert int(panel.bins_field.value) <= 50


def test_a_nonsense_bin_count_falls_back_to_the_default(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "area"
    panel.bins_field.value = "not a number"
    panel._render()
    assert panel.bins_field.value.isdigit()


def test_invalidate_drops_the_caches(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    panel._render()
    assert panel._report_cache
    panel.invalidate()
    assert panel._report_cache == {}
    assert panel._spec_cache == {}


def test_a_rename_moves_the_cache_instead_of_recomputing(panel):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    panel._render()
    before = dict(panel._report_cache)

    panel.rename_experiment("exp", "renamed")
    assert all(key[-1] == "renamed" for key in panel._report_cache)
    assert len(panel._report_cache) == len(before)
    assert panel._selected_names == ["renamed"]


def test_a_column_that_vanished_from_the_dataset_clears_the_selection(panel, loaded_state):
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    loaded_state.dataset.df = loaded_state.dataset.df.drop(columns=["site"])
    panel.set_experiments(["exp"])
    # Flet stores a cleared Dropdown value as "", not None.
    assert not panel.column_dropdown.value


def _find(control, kind: type) -> list:
    """Every control of `kind` anywhere under `control`."""
    found = []
    stack = [control]
    while stack:
        node = stack.pop()
        if isinstance(node, kind):
            found.append(node)
        for attr in ("controls", "rows"):
            children = getattr(node, attr, None)
            if isinstance(children, list):
                stack.extend(children)
        content = getattr(node, "content", None)
        if content is not None:
            stack.append(content)
    return found


def test_a_count_metric_is_shown_as_a_count_and_a_share(panel):
    """Support minus the dataset's own support is a large negative number
    with no reading; the share of it the slice holds is the fact that makes
    the count mean something."""
    panel.set_experiments(["exp"])
    panel.column_dropdown.value = "site"
    panel.metric_dropdown.value = sl.METRIC_SUPPORT
    panel.class_dropdown.value = "cat"
    panel._render()
    table = _find(panel.body, ft.DataTable)[0]
    assert table.columns[4].label.value == "share of all rows"
    # Two cat crops in the fixture, both at 'north': whole count, whole share.
    values = {row.cells[0].content.value: (row.cells[3].content.controls[0].value,
                                           row.cells[4].content.value)
              for row in table.rows[:-1]}
    assert values["north"] == ("2", "100.0%")
    assert values["south"][0] == "0"
