"""The names of exported files.

Everything Lookin writes for a person to keep -- a chart, the Embeddings
map, a crop, a report, the notes -- is named

    lookin_<workspace>_<tab>_<what>_<YYYYMMDD-HHMMSS>.<ext>

so that a folder of exports says, file by file, which workspace and which
view each came from, and two exports of the same chart never overwrite or
shadow one another. Parts that are empty are left out; each is reduced to
letters, digits, dots, dashes and underscores, and cut, so that a long
experiment or class name cannot make a name the filesystem refuses.

The page builds the names of what it saves itself (static/app.js, the map
in static/js/projection.js) from a stem made here, adding the time of the
click: export_stem without the date, export_name with it.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

PREFIX = "lookin"
PART_MAX = 40
NAME_MAX = 180
DATE_FORMAT = "%Y%m%d-%H%M%S"


def part(text) -> str:
    """One part of a name: safe characters only, cut to PART_MAX."""
    cleaned = re.sub(r"[^A-Za-z0-9.-]+", "_", str(text or "")).strip("._-")
    return cleaned[:PART_MAX].rstrip("._-")


def export_stem(*parts) -> str:
    """lookin_<part>_<part>... without the date and the extension; empty
    parts are skipped, as is a part repeating the one before it."""
    out = [PREFIX]
    for p in parts:
        cleaned = part(p)
        if cleaned and cleaned.lower() != out[-1].lower():
            out.append(cleaned)
    return "_".join(out)[:NAME_MAX]


def export_name(*parts, ext: str = "png", when: Optional[datetime] = None) -> str:
    """The full file name: the stem, the date and time, the extension."""
    stamp = (when or datetime.now()).strftime(DATE_FORMAT)
    return f"{export_stem(*parts)}_{stamp}.{ext.lstrip('.')}"


def workspace_part(workspace=None, state=None) -> str:
    """What names the workspace in a file name: its name, else the dataset
    CSV's (a session opened on files, with no workspace)."""
    if workspace is not None:
        label = workspace.label() if hasattr(workspace, "label") else str(workspace)
        if label:
            return label
    dataset = getattr(state, "dataset", None) if state is not None else None
    csv_path = getattr(dataset, "csv_path", None) if dataset is not None else None
    if csv_path:
        from pathlib import Path
        return Path(csv_path).stem
    return ""
