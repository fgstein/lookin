"""The Compare tab's confusion matrices, rendered without a browser.

core/comparison.py is covered in test_comparison.py; what is checked here is
the tab's own behaviour: that switching between the net matrix and the two
gross ones redraws the grid without recomputing the comparison, that each
cell shows the right number and colour, and that clicking a cell fills the
provenance panel beside it with where that cell's crops came from and went.
"""
from __future__ import annotations

import flet as ft
import pandas as pd
import pytest

from lookin.core import comparison as cmp
from lookin.core.data_manager import AppState, ColumnConfig
from lookin.ui.compare_view import CompareView

CLASSES = ["cat", "dog", "bird"]


class FakeCtx:
    def __init__(self, state):
        self.state = state
        self.page = None
        self.image_zoom = None
        self.display_lut = "none"


def _write_results(path, uuids, predicted, confidence=0.9):
    rows = []
    for pred in predicted:
        row = {c: (1.0 - confidence) / (len(CLASSES) - 1) for c in CLASSES}
        row[pred] = confidence
        rows.append(row)
    df = pd.DataFrame(rows, columns=CLASSES)
    df.insert(0, "uuid", list(uuids))
    df.to_csv(path, index=False)


@pytest.fixture
def view(tmp_path):
    """The same six crops as test_comparison.py: one fix, two regressions,
    and a bird row whose regression the net matrix cancels out."""
    df = pd.DataFrame({
        "uuid": list("abcdef"),
        "crop": [f"crops/{u}.png" for u in "abcdef"],
        "label": ["cat", "cat", "dog", "dog", "bird", "bird"],
    })
    df.to_csv(tmp_path / "dataset.csv", index=False)
    _write_results(tmp_path / "a.csv", df["uuid"], ["cat", "cat", "cat", "dog", "cat", "bird"])
    _write_results(tmp_path / "b.csv", df["uuid"], ["cat", "dog", "dog", "dog", "dog", "cat"])

    state = AppState()
    state.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    state.add_experiment(str(tmp_path / "a.csv"), "base")
    state.add_experiment(str(tmp_path / "b.csv"), "cand")

    v = CompareView(FakeCtx(state))
    v.build()
    v.baseline_value, v.candidate_value = "base", "cand"
    v._compare_now()
    return v


def _grid(view) -> list:
    """The matrix rows currently on screen, header included."""
    grid = view._matrix_card.content.controls[2].controls[0]
    return grid.controls


def _cell_text(view, true_label: str, predicted_label: str) -> str:
    labels = list(view._diff.confusion(view.matrix_mode).index)
    row = _grid(view)[1 + labels.index(true_label)]
    # +1 again for the row-label container at the head of each row.
    return row.controls[1 + labels.index(predicted_label)].content.value


def _switch(view) -> ft.SegmentedButton:
    return view._matrix_card.content.controls[0]


def _flow_grid(view) -> list:
    """The provenance panel's rows, header included."""
    return view._flow_card.content.controls[4].controls[0].controls


def _totals(view) -> str:
    return view._flow_card.content.controls[2].value


def _panel_row(view, label: str) -> list:
    """One class's line in the panel's focused reading: label, came from,
    went to."""
    for row in _flow_grid(view)[1:]:
        if row.controls[0].content.value == label:
            return row.controls
    raise AssertionError(f"{label!r} is not in the panel: {_flow_labels(view)}")


def _to_class_flow(view):
    switch = view._flow_card.content.controls[1]
    switch.selected = {cmp.FLOW_CLASS}
    view._on_flow_mode_change(ft.ControlEvent(
        target="", name="change", data="", control=switch, page=None))


def _flow_cell(view, from_label: str, to_label: str):
    labels = _flow_labels(view)
    row = _flow_grid(view)[1 + labels.index(from_label)]
    return row.controls[1 + labels.index(to_label)]


def _flow_labels(view) -> list:
    """Read back off the row-label column, since the panel trims the
    classes no crop of this true class ever touches."""
    return [row.controls[0].content.value for row in _flow_grid(view)[1:]]


def _click(control):
    control.on_click(ft.ControlEvent(
        target="", name="click", data="", control=control, page=None))


def _switch_to(view, mode):
    _switch(view).selected = {mode}
    view._on_matrix_mode_change(ft.ControlEvent(
        target="", name="change", data="", control=_switch(view), page=None))


# ---- the three matrices -------------------------------------------------

def test_the_net_matrix_is_what_a_fresh_comparison_shows(view):
    assert view.matrix_mode == cmp.MATRIX_NET
    assert _cell_text(view, "cat", "cat") == "-1"
    assert _cell_text(view, "cat", "dog") == "+1"


def test_switching_redraws_the_grid_with_the_arrivals(view):
    _switch_to(view, cmp.MATRIX_MOVED_IN)

    assert view.matrix_mode == cmp.MATRIX_MOVED_IN
    assert _cell_text(view, "cat", "dog") == "+1"
    # Unlike the outcome matrices this replaced, the diagonal carries
    # something: a crop arriving on it is one the candidate fixed.
    assert _cell_text(view, "dog", "dog") == "+1"
    # The whole point: the net matrix cancels this cell out, and the user
    # asked to see the traffic anyway.
    assert _cell_text(view, "bird", "cat") == "+1"


def test_a_departure_is_written_as_a_subtraction(view):
    """Every number in this tab says what happened to the cell's count, so
    a departure is negative even though it is a count of crops."""
    _switch_to(view, cmp.MATRIX_MOVED_OUT)
    assert _cell_text(view, "dog", "cat") == "-1"   # the confusion the candidate cleared up
    assert _cell_text(view, "cat", "cat") == "-1"   # a crop that left the diagonal, i.e. broke
    assert _cell_text(view, "bird", "cat") == "-1"
    # Nothing moved through here, and an empty cell has no sign to carry.
    assert _cell_text(view, "cat", "bird") == "·"


def test_switching_does_not_recompute_the_comparison(view):
    joined, diff = view._joined, view._diff
    _switch_to(view, cmp.MATRIX_MOVED_OUT)
    assert view._joined is joined and view._diff is diff


def test_every_mode_offers_all_three_and_marks_the_current_one(view):
    for mode in cmp.MATRIX_VIEWS:
        view.matrix_mode = mode
        view._render_matrix(view._diff)
        switch = _switch(view)
        assert [s.value for s in switch.segments] == list(cmp.MATRIX_VIEWS)
        assert switch.selected == {mode}


# ---- colour -------------------------------------------------------------

def _cell_color(view, mode, true_label, predicted_label):
    view.matrix_mode = mode
    view._render_matrix(view._diff)
    labels = list(view._diff.confusion(mode).index)
    row = _grid(view)[1 + labels.index(true_label)]
    return row.controls[1 + labels.index(predicted_label)].bgcolor


def _is_red(color) -> bool:
    # diverging_color interpolates from a neutral grey towards brick red or
    # sea green; the sign of the news, not of the count, picks the target.
    return int(color[1:3], 16) > int(color[5:7], 16)


def _is_green(color) -> bool:
    return int(color[3:5], 16) > int(color[1:3], 16)


def test_an_arrival_is_red_off_the_diagonal_and_green_on_it(view):
    """A movement matrix is a plain count, but it still has a direction: a
    crop arriving on the true class was fixed, one arriving anywhere else
    was broken. Colouring by the count alone would paint both the same."""
    off = _cell_color(view, cmp.MATRIX_MOVED_IN, "cat", "dog")
    on = _cell_color(view, cmp.MATRIX_MOVED_IN, "dog", "dog")
    assert _is_red(off), f"expected a red cell, got {off}"
    assert _is_green(on), f"expected a green cell, got {on}"


def test_a_departure_reads_the_other_way_round(view):
    """Leaving a confusion is good news; leaving the diagonal is not."""
    off = _cell_color(view, cmp.MATRIX_MOVED_OUT, "dog", "cat")
    on = _cell_color(view, cmp.MATRIX_MOVED_OUT, "cat", "cat")
    assert _is_green(off), f"expected a green cell, got {off}"
    assert _is_red(on), f"expected a red cell, got {on}"


# ---- clicking a cell ----------------------------------------------------

def test_clicking_a_cell_selects_it_instead_of_opening_the_carousel(view, monkeypatch):
    """A cell of any of these matrices is a set of *movements*, and the
    images alone cannot say what moved where -- so the click fills the
    provenance panel and the carousel is opened from there."""
    opened = []
    monkeypatch.setattr(view, "_open_carousel",
                        lambda rows, title, focus_uuid=None: opened.append(title))

    view._on_cell_click("bird", "cat")
    assert view.selected_cell == ("bird", "cat")
    assert opened == []


def test_clicking_the_same_cell_again_puts_the_panel_away(view):
    view._on_cell_click("bird", "cat")
    view._on_cell_click("bird", "cat")
    assert view.selected_cell is None


def test_the_selection_survives_switching_matrix(view):
    """The same cell in the other matrix is the other half of the same
    question, so the panel is not thrown away on a switch."""
    view._on_cell_click("bird", "cat")
    _switch_to(view, cmp.MATRIX_MOVED_OUT)
    assert view.selected_cell == ("bird", "cat")


def test_a_selected_cell_is_outlined_in_the_grid(view):
    view._on_cell_click("bird", "cat")
    labels = list(view._diff.confusion(view.matrix_mode).index)
    row = _grid(view)[1 + labels.index("bird")]
    assert row.controls[1 + labels.index("cat")].border is not None
    assert row.controls[1 + labels.index("dog")].border is None


# ---- the provenance panel, cell by cell ---------------------------------

def test_the_panel_waits_for_a_cell_before_saying_anything(view):
    assert view.selected_cell is None
    assert "Pick a cell" in view._flow_card.content.controls[1].value


def test_the_panel_counts_only_the_crops_that_touch_the_picked_cell(view):
    """The point of the focused reading: for (bird -> cat), 'f' arrived
    from bird and 'e' left for dog, and no other crop of the bird class is
    counted -- not even the ones that moved elsewhere in the same row."""
    view._on_cell_click("bird", "cat")
    assert view.flow_mode == cmp.FLOW_CELL
    assert _flow_labels(view) == ["bird", "dog"]
    assert _panel_row(view, "bird")[1].content.value == "+1"   # came from the true class
    assert _panel_row(view, "bird")[2].content.value == "·"    # and nothing went back to it
    assert _panel_row(view, "dog")[1].content.value == "·"
    assert _panel_row(view, "dog")[2].content.value == "-1"    # left for dog
    assert "+1 moved in, -1 moved out" in _totals(view)


def test_a_class_the_cell_never_traded_with_is_not_a_line(view):
    """A cell usually exchanges crops with two or three classes; the rest
    would be a page of zeros."""
    view._on_cell_click("dog", "cat")
    assert "bird" not in _flow_labels(view)


def test_clicking_a_panel_line_opens_exactly_that_move(view, monkeypatch):
    opened = {}
    monkeypatch.setattr(view, "_open_carousel",
                        lambda rows, title, focus_uuid=None: opened.update(rows=rows, title=title))

    view._on_cell_click("bird", "cat")
    _click(_panel_row(view, "dog")[2])          # the departure to dog
    assert list(opened["rows"]["uuid"]) == ["e"]
    assert "base: cat" in opened["title"] and "cand: dog" in opened["title"]

    _click(_panel_row(view, "bird")[1])         # the arrival from bird
    assert list(opened["rows"]["uuid"]) == ["f"]


def test_an_empty_panel_count_is_not_clickable(view):
    view._on_cell_click("bird", "cat")
    assert _panel_row(view, "bird")[2].on_click is None


def test_a_cell_nothing_moved_through_says_so(view):
    view._on_cell_click("cat", "bird")
    assert "0 moved in, 0 moved out — net 0" in _totals(view)
    assert "Nothing moved through this cell" in _flow_grid(view)[1].controls[0].value


def test_the_panel_buttons_open_each_direction_on_its_own(view, monkeypatch):
    opened = {}
    monkeypatch.setattr(view, "_open_carousel",
                        lambda rows, title, focus_uuid=None: opened.update(rows=rows, title=title))

    view._on_cell_click("bird", "cat")
    buttons = view._flow_card.content.controls[5].controls
    _click(buttons[0])
    assert list(opened["rows"]["uuid"]) == ["f"], "moved in"
    _click(buttons[1])
    assert list(opened["rows"]["uuid"]) == ["e"], "moved out"


# ---- the provenance panel, whole class ----------------------------------

def test_the_class_reading_counts_every_crop_of_the_class(view):
    """The wider question, kept as a second mode: for true=bird it shows
    the crop that moved cat -> dog as well, which never touched the picked
    cell."""
    view._on_cell_click("bird", "cat")
    _to_class_flow(view)
    assert view.flow_mode == cmp.FLOW_CLASS
    assert _flow_cell(view, "bird", "cat").content.value == "+1"   # into the picked cell
    assert _flow_cell(view, "cat", "dog").content.value == "-1"    # out of it, to dog


def test_the_class_reading_leaves_an_unrelated_move_unsigned(tmp_path, view):
    """A move between two classes that are not the picked one is real, but
    it does not change that cell's count -- signing it would say it did."""
    view._on_cell_click("bird", "bird")
    _to_class_flow(view)
    # cat -> dog touches neither the bird column nor the bird row.
    assert _flow_cell(view, "cat", "dog").content.value == "1"


def test_the_class_reading_outlines_the_arrivals_and_the_departures(view):
    view._on_cell_click("bird", "cat")
    _to_class_flow(view)
    assert _flow_cell(view, "bird", "cat").border is not None   # into cat, from bird
    assert _flow_cell(view, "cat", "dog").border is not None    # out of cat, to dog
    assert _flow_cell(view, "bird", "dog").border is None       # neither


def test_the_class_reading_drops_the_classes_this_row_never_touches(view):
    """true=dog only ever involves cat and dog, so a bird row would be a
    row of zeros."""
    view._on_cell_click("dog", "dog")
    _to_class_flow(view)
    assert _flow_labels(view) == ["cat", "dog"]


def test_an_empty_cell_still_gets_its_column_in_the_class_reading(view):
    """Clicking an empty cell is how "why is nothing here" gets asked, and
    a panel that trimmed that class away would answer nothing."""
    view._on_cell_click("dog", "bird")
    _to_class_flow(view)
    assert _flow_labels(view) == ["bird", "cat", "dog"]


def test_switching_the_panel_reading_does_not_recompute_anything(view):
    view._on_cell_click("bird", "cat")
    joined, diff = view._joined, view._diff
    _to_class_flow(view)
    assert view._joined is joined and view._diff is diff


# ---- the snapshot -------------------------------------------------------

def test_the_chosen_matrix_survives_a_snapshot(view):
    view.matrix_mode = cmp.MATRIX_MOVED_IN
    state = view.view_state()
    view.matrix_mode = cmp.MATRIX_NET
    assert view.restore_view_state(state) == []
    assert view.matrix_mode == cmp.MATRIX_MOVED_IN


def test_the_selected_cell_survives_a_snapshot(view):
    view._on_cell_click("bird", "cat")
    state = view.view_state()
    view.selected_cell = None
    assert view.restore_view_state(state) == []
    assert view.selected_cell == ("bird", "cat")


def test_the_panel_reading_survives_a_snapshot(view):
    view.flow_mode = cmp.FLOW_CLASS
    state = view.view_state()
    view.flow_mode = cmp.FLOW_CELL
    assert view.restore_view_state(state) == []
    assert view.flow_mode == cmp.FLOW_CLASS


def test_a_snapshot_naming_an_unknown_matrix_falls_back(view):
    view.restore_view_state({"baseline": "base", "candidate": "cand", "matrix": "nope"})
    assert view.matrix_mode in cmp.MATRIX_VIEWS


def test_a_selected_cell_this_pair_does_not_have_is_dropped(view):
    """A snapshot can name a class the current pair never predicts, and
    indexing the flow matrix with it would raise mid-render."""
    view.selected_cell = ("tractor", "cat")
    view._render_matrix(view._diff)
    assert view.selected_cell is None
