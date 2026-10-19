"""The workspace board: user-named columns, and which workspace sits where.

    <library-root>/board.json

    {"columns": [{"id": "col-7f3a2b", "name": "group 1",
                  "workspaces": ["lkw-dv6692", "lkd-abcdef"]}]}

Columns are whatever the user calls them -- a workspace is an
investigation, and how people sort theirs (by stage, by project, by
person) is not something to decide for them. It starts as a single
"group 1".

Kept beside the workspaces rather than in the per-user settings, because
it is about the workspaces: two people sharing a library on a mount see
the same board, the way two people on one Jira project see the same one.
And kept in one file rather than a field in each manifest, so that
renaming or removing a column is one write, not one per workspace.

The board never has to agree with the library exactly. A workspace that is
on no column (new, imported, copied in by hand) shows in the first one; an
id whose workspace is gone is dropped the next time the board is written.
Every mutation re-reads the file and writes it back under the file's lock
(core/fileio.py), so two sessions editing the board -- two tabs, or two
people on one library -- never lose each other's changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from lookin.core import fileio
from lookin.core import library as lib

BOARD_FILE = "board.json"
COLUMN_PREFIX = "col-"
DEFAULT_COLUMN_NAME = "group 1"


def board_path(root=None) -> Path:
    return (Path(root) if root is not None else lib.library_root()) / BOARD_FILE


def _default() -> dict:
    return {"columns": [{"id": lib.new_id(COLUMN_PREFIX), "name": DEFAULT_COLUMN_NAME, "workspaces": []}]}


def _normalize(raw) -> dict:
    """Tolerant, like every other file here: a hand-edited board with one
    bad column costs that column, not the board."""
    columns, seen_cols, seen_ws = [], set(), set()
    raw_columns = raw.get("columns") if isinstance(raw, dict) else None
    for c in raw_columns if isinstance(raw_columns, list) else []:
        if not isinstance(c, dict):
            continue
        col_id = str(c.get("id") or lib.new_id(COLUMN_PREFIX))
        if col_id in seen_cols:
            continue
        seen_cols.add(col_id)
        ids = []
        for ws_id in c.get("workspaces") or []:
            ws_id = str(ws_id)
            if ws_id not in seen_ws:       # a workspace is on one column only
                seen_ws.add(ws_id)
                ids.append(ws_id)
        columns.append({"id": col_id, "name": str(c.get("name") or "untitled"), "workspaces": ids})
    return {"columns": columns} if columns else _default()


def load(root=None) -> dict:
    """The board, written out the first time it is made: a default board's
    column id is random, and one that only lived in memory would get a
    different id on every read -- so a move aimed at it would miss."""
    raw = lib._read_json(board_path(root))
    board = _normalize(raw)
    if raw is None or not (raw.get("columns") if isinstance(raw, dict) else None):
        # Looked at again under the lock: two sessions starting on an empty
        # library would otherwise each write a default column of their own,
        # and a move aimed at the loser's column id would miss.
        with _locked(root):
            raw = lib._read_json(board_path(root))
            board = _normalize(raw)
            if raw is None or not (raw.get("columns") if isinstance(raw, dict) else None):
                save(board, root)
    return board


def _locked(root=None):
    return fileio.locked(board_path(root))


def save(board: dict, root=None) -> bool:
    return lib._atomic_write_json(board, board_path(root))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def layout(board: dict, workspaces: list) -> list:
    """``[(column, [workspace, ...]), ...]`` in board order.

    A workspace the board does not mention goes at the end of the first
    column; an id the library no longer has is skipped.
    """
    by_id = {w.id: w for w in workspaces}
    placed: set = set()
    out = []
    for column in board["columns"]:
        items = [by_id[i] for i in column["workspaces"] if i in by_id]
        placed.update(w.id for w in items)
        out.append((column, items))
    unplaced = [w for w in workspaces if w.id not in placed]
    if out and unplaced:
        out[0] = (out[0][0], out[0][1] + unplaced)
    return out


def column_of(board: dict, workspace_id: str) -> Optional[dict]:
    return next((c for c in board["columns"] if workspace_id in c["workspaces"]), None)


# ---------------------------------------------------------------------------
# changing -- each one re-reads, changes, writes, and returns the new board
# ---------------------------------------------------------------------------

def _column(board: dict, column_id: str) -> Optional[dict]:
    return next((c for c in board["columns"] if c["id"] == column_id), None)


def _sync(board: dict, known_ids) -> None:
    """Make the file say what the screen shows: drop ids the library no
    longer has, and write down where unplaced workspaces are displayed (end
    of the first column). Without the second half, a workspace nobody had
    moved yet would jump below the cards moved in after it, since placed
    ones are listed first."""
    if known_ids is None:
        return
    known = list(known_ids)
    keep = set(known)
    for c in board["columns"]:
        c["workspaces"] = [i for i in c["workspaces"] if i in keep]
    placed = {i for c in board["columns"] for i in c["workspaces"]}
    board["columns"][0]["workspaces"] += [i for i in known if i not in placed]


def move(workspace_id: str, column_id: str, index: Optional[int] = None, root=None,
         known_ids=None) -> dict:
    """Put a workspace on a column, at `index` (end when None)."""
    with _locked(root):
        board = load(root)
        target = _column(board, column_id)
        if target is None:
            return board
        _sync(board, known_ids)
        for c in board["columns"]:
            c["workspaces"] = [i for i in c["workspaces"] if i != workspace_id]
        position = len(target["workspaces"]) if index is None else max(0, min(index, len(target["workspaces"])))
        target["workspaces"].insert(position, workspace_id)
        save(board, root)
        return board


def place_after(workspace_id: str, after_id: str, root=None, known_ids=None) -> dict:
    """Put a workspace right below another one, on that one's column -- where
    a duplicate goes. The end of the first column when `after_id` is on
    none (it is shown there, see layout)."""
    with _locked(root):
        board = load(root)
        _sync(board, known_ids)
        for c in board["columns"]:
            c["workspaces"] = [i for i in c["workspaces"] if i != workspace_id]
        target = column_of(board, after_id)
        if target is None:
            board["columns"][0]["workspaces"].append(workspace_id)
        else:
            target["workspaces"].insert(target["workspaces"].index(after_id) + 1, workspace_id)
        save(board, root)
        return board


def add_column(name: str, root=None) -> dict:
    with _locked(root):
        board = load(root)
        board["columns"].append({"id": lib.new_id(COLUMN_PREFIX),
                                 "name": name.strip() or next_default_name(board), "workspaces": []})
        save(board, root)
        return board


def rename_column(column_id: str, name: str, root=None) -> dict:
    with _locked(root):
        board = load(root)
        column = _column(board, column_id)
        if column is not None and name.strip():
            column["name"] = name.strip()
            save(board, root)
        return board


def neighbour(board: dict, column_id: str) -> Optional[dict]:
    """Where a removed column's workspaces go: the column on its left, or
    on its right when it was the first. None when it is the only one."""
    ids = [c["id"] for c in board["columns"]]
    if column_id not in ids or len(ids) < 2:
        return None
    i = ids.index(column_id)
    return board["columns"][i - 1] if i > 0 else board["columns"][1]


def remove_column(column_id: str, root=None, workspaces: Optional[list] = None) -> dict:
    """Remove a column; its workspaces go, in order, to the end of its
    neighbour. The last column cannot be removed -- a board with no column
    has nowhere to show a workspace.

    `workspaces` is the library's list, so that the workspaces the board
    does not mention yet are written where they are shown first (see
    _sync) -- the first column holds them, and removing it has to carry
    them along.
    """
    with _locked(root):
        board = load(root)
        target = neighbour(board, column_id)
        if target is None:
            return board
        if workspaces is not None:
            _sync(board, [w.id for w in workspaces])
        column = _column(board, column_id)
        target["workspaces"] += [i for i in column["workspaces"] if i not in target["workspaces"]]
        board["columns"] = [c for c in board["columns"] if c["id"] != column_id]
        save(board, root)
        return board


def next_default_name(board: dict) -> str:
    """"group N" with the first N not already taken."""
    names = {c["name"] for c in board["columns"]}
    n = len(board["columns"]) + 1
    while f"group {n}" in names:
        n += 1
    return f"group {n}"
