"""The crop display dialog, rendered without a browser.

The LUT mapping itself is covered in tests/test_luts.py and the rotation
in tests/test_roll_rotation.py; what is checked here is the wiring that
only exists in the dialog -- which columns it offers as a roll, that
picking one reaches the session setting and tells the views to redraw, and
that a column left over from a previously loaded dataset is dropped rather
than left claiming to do something.
"""
from __future__ import annotations

import pytest

from lookin.core.data_manager import AppState, ColumnConfig
from lookin.core.images import LUT_MINMAX, LUT_NONE, ROLL_NONE, ROLL_NONE_KEY, roll_column_candidates
from lookin.web.views.display import DisplayDialog, display_label


class FakeSession:
    def __init__(self, state):
        self.state = state
        self.display_lut = LUT_NONE
        self.display_roll_col = ROLL_NONE
        self.display_changes = 0

    def notify_display_change(self) -> None:
        self.display_changes += 1

    def mark(self, *names) -> None:
        pass


@pytest.fixture
def dialog(loaded_state):
    dlg = DisplayDialog()
    dlg.session = FakeSession(loaded_state)
    return dlg


# ---- which columns can hold a roll --------------------------------------

def test_only_numeric_non_structural_columns_are_offered(loaded_state):
    offered = roll_column_candidates(loaded_state.dataset)
    assert offered == ["flag", "area"]
    assert "site" not in offered, "a roll is a number, not a category"
    assert "crop" not in offered and "label" not in offered, "structural roles"


def test_no_dataset_offers_nothing():
    assert roll_column_candidates(None) == []


def test_the_dialog_offers_exactly_those_columns(dialog):
    html = dialog.body()
    assert 'value="area"' in html and 'value="flag"' in html
    assert 'value="site"' not in html


# ---- picking one --------------------------------------------------------

def test_picking_a_column_sets_it_and_asks_the_views_to_redraw(dialog):
    dialog.on_roll(value="area")
    assert dialog.session.display_roll_col == "area"
    assert dialog.session.display_changes == 1


def test_picking_the_same_column_again_costs_nothing(dialog):
    dialog.on_roll(value="area")
    dialog.on_roll(value="area")
    assert dialog.session.display_changes == 1


def test_no_rotation_clears_the_setting(dialog):
    dialog.session.display_roll_col = "area"
    dialog.on_roll(value=ROLL_NONE_KEY)
    assert dialog.session.display_roll_col == ROLL_NONE


def test_picking_a_lut_sets_it_once(dialog):
    dialog.on_lut(key=LUT_MINMAX)
    dialog.on_lut(key=LUT_MINMAX)
    assert dialog.session.display_lut == LUT_MINMAX
    assert dialog.session.display_changes == 1


# ---- following the loaded dataset ---------------------------------------

def test_a_column_from_a_previous_dataset_is_dropped(dialog, tmp_path, tiny_dataset_df):
    """It would silently do nothing -- a record with no such column is left
    as stored -- and a setting that claims to be on while doing nothing is
    worse than one that is off."""
    dialog.session.display_roll_col = "area"
    path = tmp_path / "other.csv"
    tiny_dataset_df.drop(columns=["area"]).to_csv(path, index=False)
    state = AppState()
    state.load_dataset(str(path), ColumnConfig())
    dialog.session.state = state

    dialog.body()
    assert dialog.session.display_roll_col == ROLL_NONE


def test_no_dataset_says_so_instead_of_showing_an_empty_dropdown(dialog):
    dialog.session.state = AppState()
    html = dialog.body()
    assert "Load a dataset" in html
    assert "Roll column" not in html


# ---- the top-bar label --------------------------------------------------

def test_the_label_names_the_column_in_use(dialog):
    session = dialog.session
    session.display_lut = LUT_MINMAX
    assert "roll" not in display_label(session)

    session.display_roll_col = "area"
    assert "min/max" in display_label(session)
    assert "roll: area" in display_label(session)
