"""The results -> dataset class mapping, past the point where every column
gets its own picker.

Past MAPPING_MAX_ALL classes the Data tab lists only the columns that match
no dataset class by name, and used to stop there: a column whose name did
match -- the wrong class, as far as the user knew -- and any unmatched one
past the first twenty could only be remapped by editing the results CSV.
Every column has to be reachable from the tab.
"""
from __future__ import annotations

import flet as ft
import pandas as pd
import pytest

from lookin.core.data_manager import AppState, ColumnConfig
from lookin.ui.class_matrix import ClassPicker
from lookin.ui.upload_view import MAPPING_MAX_ALL, UploadView

N_CLASSES = MAPPING_MAX_ALL + 15


class FakeCtx:
    def __init__(self, state):
        self.state = state
        self.changes = 0

    def notify_change(self):
        self.changes += 1


@pytest.fixture
def many(tmp_path):
    classes = [f"c{i}" for i in range(N_CLASSES)]
    dataset = pd.DataFrame({
        "uuid": [f"u{i}" for i in range(N_CLASSES)],
        "crop": [f"crops/{i}.png" for i in range(N_CLASSES)],
        "label": classes,
    })
    # Every column matches a dataset class but the last thirty, which are
    # spelled the network's way -- more unmatched columns than one screen.
    columns = classes[:-30] + [f"net_{c}" for c in classes[-30:]]
    results = pd.DataFrame({"uuid": dataset["uuid"]})
    for i, col in enumerate(columns):
        results[col] = [1.0 if j == i else 0.0 for j in range(N_CLASSES)]
    dataset.to_csv(tmp_path / "dataset.csv", index=False)
    results.to_csv(tmp_path / "results.csv", index=False)
    state = AppState()
    state.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    state.add_experiment(str(tmp_path / "results.csv"), "exp")
    view = UploadView.__new__(UploadView)
    view.ctx = FakeCtx(state)
    row = {"loaded_name": "exp", "mapping_container": ft.Column()}
    view._build_row_mapping(row)
    return view, row, state


def _parts(row):
    column = row["mapping_container"].controls[0]
    picker_box = next(c for c in column.controls if isinstance(c, ft.Column))
    mapping_row = next(c for c in column.controls if isinstance(c, ft.Row))
    return picker_box, mapping_row


def _picker(view, row) -> ClassPicker:
    picker_box, _ = _parts(row)
    search = picker_box.controls[0]
    # The picker is reached through its search field's handler.
    return search.on_change.__self__


def test_the_first_unmatched_columns_are_listed_up_front(many):
    view, row, _ = many
    _, mapping_row = _parts(row)
    assert len(mapping_row.controls) == MAPPING_MAX_ALL
    assert all(d.label.startswith("net_") for d in mapping_row.controls)


def test_a_column_matched_by_name_can_still_be_remapped(many):
    view, row, state = many
    picker = _picker(view, row)
    picker._add("c3")
    _, mapping_row = _parts(row)
    added = next(d for d in mapping_row.controls if d.label == "c3")
    assert added.value == "c3"

    added.value = "c4"
    added.on_change(type("E", (), {"control": added})())
    assert state.experiments["exp"].class_labels[3] == "c4"


def test_an_unmatched_column_past_the_first_screen_is_reachable(many):
    view, row, state = many
    last = state.experiments["exp"].score_cols[-1]
    picker = _picker(view, row)
    picker._add(last)
    _, mapping_row = _parts(row)
    assert last in [d.label for d in mapping_row.controls]


def test_opened_fields_survive_a_rebuild(many):
    view, row, _ = many
    _picker(view, row)._add("c3")
    row["mapping_container"].controls.clear()
    view._build_row_mapping(row)
    _, mapping_row = _parts(row)
    assert "c3" in [d.label for d in mapping_row.controls]


def test_an_exact_match_is_listed_before_longer_ones():
    """Only the first few matches are offered; "c2" must not be buried
    under "c12", "c20", "c21", ..."""
    picker = ClassPicker(lambda picked: None, limit=None)
    picker.set_labels([f"c{i}" for i in range(300)] + ["x"])
    picker.search.value = "c2"
    picker._refresh_suggestions()
    assert picker.suggestions.controls[0].text == "c2"
