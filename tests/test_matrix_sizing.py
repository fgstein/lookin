"""How a confusion matrix is sized from the number of classes it shows.

A binary matrix used to be four 34px cells in the corner of a wide tab --
the most common case there is, drawn smallest. The grid now has a floor:
below eight columns the cells take the room the missing ones would have
used, so the card keeps the footprint an eight-class matrix has. Above
that nothing changes, because a many-class matrix already needs its width.
"""
from __future__ import annotations

import pytest

from lookin.ui.components import (
    MATRIX_FLOOR_COLUMNS, matrix_cell_size, matrix_label_chars, matrix_text_size,
)
from lookin.ui.metrics_view import CELL_SIZE, LABEL_COL_WIDTH


def _grid_width(n_columns: int) -> int:
    """What metrics_view._build_confusion_grid computes for n classes."""
    return LABEL_COL_WIDTH + n_columns * (matrix_cell_size(n_columns, CELL_SIZE) + 2)


@pytest.mark.parametrize("n_columns", list(range(1, MATRIX_FLOOR_COLUMNS + 1)))
def test_a_small_matrix_keeps_the_footprint_of_an_eight_class_one(n_columns):
    reference = _grid_width(MATRIX_FLOOR_COLUMNS)
    assert abs(_grid_width(n_columns) - reference) <= n_columns, "within the cell rounding"


def test_two_classes_gives_big_cells_rather_than_a_small_grid():
    assert matrix_cell_size(2, CELL_SIZE) == 4 * CELL_SIZE + 6


@pytest.mark.parametrize("n_columns", [8, 9, 12, 40, 100])
def test_a_many_class_matrix_keeps_the_base_pitch(n_columns):
    """Past the floor the grid grows instead, which is what it needs: cells
    that also grew would put a hundred classes behind a scrollbar."""
    assert matrix_cell_size(n_columns, CELL_SIZE) == CELL_SIZE


def test_no_columns_at_all_is_not_a_division_by_zero():
    assert matrix_cell_size(0, CELL_SIZE) == CELL_SIZE


def test_the_text_grows_with_the_cell_and_stops():
    assert matrix_text_size(CELL_SIZE, CELL_SIZE, 10) == 10
    assert matrix_text_size(CELL_SIZE - 5, CELL_SIZE, 10) == 10, "never smaller than the base"
    assert matrix_text_size(int(1.5 * CELL_SIZE), CELL_SIZE, 10) == 15
    assert matrix_text_size(10 * CELL_SIZE, CELL_SIZE, 10) == 18, "capped"
    assert matrix_text_size(10 * CELL_SIZE, CELL_SIZE, 9, cap=14) == 14


def test_a_wider_cell_shows_more_of_the_class_label():
    narrow = matrix_label_chars(CELL_SIZE, CELL_SIZE, 4)
    wide = matrix_label_chars(matrix_cell_size(2, CELL_SIZE), CELL_SIZE, 4)
    assert narrow == 4, "the base cell is the floor it always was"
    assert wide > narrow
