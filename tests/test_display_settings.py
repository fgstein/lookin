"""The app-bar display control, rendered without a browser.

The LUT mapping itself is covered in tests/test_luts.py and the rotation
in tests/test_roll_rotation.py; what is checked here is the wiring that
only exists in the control -- which columns it offers as a roll, that
picking one reaches the session setting and tells the views to redraw, and
that a column left over from a previously loaded dataset is dropped rather
than left claiming to do something.

Same harness as tests/test_slice_panel.py: every .update() in the view is
guarded by `if control.page:`, so a context with no page renders and skips
the pushes.
"""
from __future__ import annotations

import pytest

from lookin.core.data_manager import AppState
from lookin.ui.components import LUT_MINMAX, LUT_NONE, ROLL_NONE
from lookin.ui.display_settings import (
    ROLL_NONE_KEY, DisplaySettings, roll_column_candidates,
)


class FakeCtx:
    def __init__(self, state):
        self.state = state
        self.page = None
        self.display_lut = LUT_NONE
        self.display_roll_col = ROLL_NONE
        self.display_changes = 0

    def notify_display_change(self) -> None:
        self.display_changes += 1


@pytest.fixture
def control(loaded_state):
    return DisplaySettings(FakeCtx(loaded_state))


# ---- which columns can hold a roll --------------------------------------

def test_only_numeric_non_structural_columns_are_offered(loaded_state):
    offered = roll_column_candidates(loaded_state.dataset)
    assert offered == ["flag", "area"]
    assert "site" not in offered, "a roll is a number, not a category"
    assert "crop" not in offered and "label" not in offered, "structural roles"


def test_no_dataset_offers_nothing(control):
    assert roll_column_candidates(None) == []


# ---- picking one --------------------------------------------------------

def test_picking_a_column_sets_it_and_asks_the_views_to_redraw(control):
    control._render_roll()
    control.roll_dropdown.value = "area"
    control._choose_roll_column(None)
    assert control.ctx.display_roll_col == "area"
    assert control.ctx.display_changes == 1


def test_picking_the_same_column_again_costs_nothing(control):
    control._render_roll()
    control.roll_dropdown.value = "area"
    control._choose_roll_column(None)
    control._choose_roll_column(None)
    assert control.ctx.display_changes == 1


def test_the_roll_section_is_only_the_column_picker(control):
    """There is no direction to choose: the correction turns against the
    roll, which is the only way it cancels it (see
    components.quarter_turns_for_roll)."""
    control.ctx.display_roll_col = "area"
    control._render_roll()
    assert control.roll_area.controls == [control.roll_dropdown]


def test_no_rotation_clears_the_setting(control):
    control.ctx.display_roll_col = "area"
    control._render_roll()
    control.roll_dropdown.value = ROLL_NONE_KEY
    control._choose_roll_column(None)
    assert control.ctx.display_roll_col == ROLL_NONE


# ---- following the loaded dataset ---------------------------------------

def test_a_column_from_a_previous_dataset_is_dropped(control, tmp_path, tiny_dataset_df):
    """It would silently do nothing -- a record with no such column is left
    as stored -- and a setting that claims to be on while doing nothing is
    worse than one that is off."""
    control.ctx.display_roll_col = "area"
    without_area = tiny_dataset_df.drop(columns=["area"])
    path = tmp_path / "other.csv"
    without_area.to_csv(path, index=False)
    state = AppState()
    from lookin.core.data_manager import ColumnConfig
    state.load_dataset(str(path), ColumnConfig())
    control.ctx.state = state

    control._render_roll()
    assert control.ctx.display_roll_col == ROLL_NONE


def test_no_dataset_says_so_instead_of_showing_an_empty_dropdown(control):
    control.ctx.state = AppState()
    control._render_roll()
    assert control.roll_dropdown not in control.roll_area.controls
    assert "Load a dataset" in control.roll_area.controls[0].value


# ---- the app-bar label --------------------------------------------------

def test_the_label_names_the_column_in_use(control):
    control.ctx.display_lut = LUT_MINMAX
    control._sync_label()
    assert "roll" not in control.label.value

    control.ctx.display_roll_col = "area"
    control._sync_label()
    assert "Min / max stretch" in control.label.value
    assert "roll: area" in control.label.value
