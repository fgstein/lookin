"""The workspace board: user-named columns that survive whatever the
library does underneath them."""
from __future__ import annotations

import json

import pytest

from lookin.core import board as bd
from lookin.core import library as lib


@pytest.fixture
def root(tmp_path):
    return tmp_path / "library"


def _ws(*ids):
    return [lib.Workspace(id=i, name=i) for i in ids]


def _names(layout):
    return [(c["name"], [w.id for w in ws]) for c, ws in layout]


def test_a_new_board_is_one_column_called_group_1_and_keeps_its_id(root):
    board = bd.load(root)
    assert [c["name"] for c in board["columns"]] == ["group 1"]
    assert bd.load(root)["columns"][0]["id"] == board["columns"][0]["id"]
    assert bd.move("a", board["columns"][0]["id"], root=root)["columns"][0]["workspaces"] == ["a"]


def test_workspaces_the_board_does_not_know_show_in_the_first_column(root):
    board = bd.add_column("", root)
    assert _names(bd.layout(board, _ws("a", "b"))) == [("group 1", ["a", "b"]), ("group 2", [])]


def test_moving_puts_a_workspace_on_exactly_one_column(root):
    board = bd.add_column("done", root)
    second = board["columns"][1]["id"]
    bd.move("a", second, root=root)
    board = bd.move("b", second, index=0, root=root)
    assert _names(bd.layout(board, _ws("a", "b", "c"))) == [("group 1", ["c"]), ("done", ["b", "a"])]
    first = board["columns"][0]["id"]
    board = bd.move("a", first, root=root)
    assert _names(bd.layout(board, _ws("a", "b", "c"))) == [("group 1", ["a", "c"]), ("done", ["b"])]


def test_removing_a_column_sends_its_workspaces_to_the_left(root):
    bd.add_column("b", root)
    board = bd.add_column("c", root)
    b, c = board["columns"][1]["id"], board["columns"][2]["id"]
    bd.move("x", b, root=root)
    bd.move("y", c, root=root)
    board = bd.remove_column(c, root)
    assert _names(bd.layout(board, _ws("x", "y"))) == [("group 1", []), ("b", ["x", "y"])]


def test_removing_the_first_column_sends_them_right_including_unplaced_ones(root):
    board = bd.add_column("next", root)
    first, second = board["columns"][0]["id"], board["columns"][1]["id"]
    bd.move("placed", first, root=root)
    bd.move("other", second, root=root)
    board = bd.remove_column(first, root, workspaces=_ws("placed", "other", "unplaced"))
    assert _names(bd.layout(board, _ws("placed", "other", "unplaced"))) == [
        ("next", ["other", "placed", "unplaced"])]


def test_the_last_column_cannot_be_removed(root):
    board = bd.load(root)
    bd.save(board, root)
    assert len(bd.remove_column(board["columns"][0]["id"], root)["columns"]) == 1


def test_rename_and_default_names(root):
    board = bd.add_column("", root)
    board = bd.rename_column(board["columns"][0]["id"], "  To explore ", root)
    assert [c["name"] for c in board["columns"]] == ["To explore", "group 2"]
    assert bd.next_default_name(board) == "group 3"
    assert bd.rename_column(board["columns"][0]["id"], "   ", root)["columns"][0]["name"] == "To explore"


def test_ids_of_deleted_workspaces_are_dropped_on_the_next_write(root):
    board = bd.load(root)
    col = board["columns"][0]["id"]
    bd.save(board, root)
    bd.move("gone", col, root=root)
    board = bd.move("kept", col, root=root, known_ids={"kept"})
    assert board["columns"][0]["workspaces"] == ["kept"]


def test_a_hand_mangled_board_keeps_what_it_can(root):
    root.mkdir(parents=True)
    bd.board_path(root).write_text(json.dumps({"columns": [
        {"id": "col-a", "name": "A", "workspaces": ["w1", "w1"]}, "junk",
        {"id": "col-b", "workspaces": ["w1", "w2"]}]}))
    board = bd.load(root)
    assert [(c["name"], c["workspaces"]) for c in board["columns"]] == [("A", ["w1"]), ("untitled", ["w2"])]
    bd.board_path(root).write_text("{not json")
    assert [c["name"] for c in bd.load(root)["columns"]] == ["group 1"]


def test_a_workspace_nobody_moved_keeps_its_place_when_others_arrive(root):
    board = bd.add_column("b", root)
    first, second = board["columns"][0]["id"], board["columns"][1]["id"]
    known = ["old", "mover"]
    bd.move("mover", second, root=root, known_ids=known)
    board = bd.move("mover", first, root=root, known_ids=known)
    assert _names(bd.layout(board, _ws(*known)))[0] == ("group 1", ["old", "mover"])
