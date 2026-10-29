"""A workspace's notes: free text the user writes about it, kept as
Markdown next to its manifest.

The Notes tab edits them, the reports carry them (core/report_builder.py),
and an agent reads and writes them over MCP (core/agent_tools.py,
core/agent_actions.py) -- all through here.

A new note starts with the workspace's name as its title and a short
recap of how its experiments perform. The recap is written *into* the
note, once: it is the user's text from then on, edited like the rest, and
"Regenerate" (or regenerate_recap over MCP) writes it again from the
current numbers -- after a relabeling or a new experiment, say. It sits
between two markers (markdown_lite.region_open("recap")) so that it can
be found again; a note whose markers were deleted gets the new recap
under its title instead.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lookin.core import fileio
from lookin.core import markdown_lite as md
from lookin.core.library import NOTES_NAME
RECAP = "recap"
# Under the recap of a new note: where the user's own text goes.
NOTES_HEADING = "## Notes"


def notes_path(workspace) -> Path | None:
    return Path(workspace.path) / NOTES_NAME if workspace is not None and workspace.path else None


def read(workspace) -> str | None:
    """The workspace's notes, or None when it has none yet."""
    path = notes_path(workspace)
    try:
        return path.read_text(encoding="utf-8") if path is not None and path.is_file() else None
    except OSError:
        return None


def stamp(workspace) -> float:
    """When the notes file last changed (0 when there is none): how a tab
    finds out that another tab, or an agent, wrote them."""
    path = notes_path(workspace)
    try:
        return path.stat().st_mtime if path is not None and path.is_file() else 0.0
    except OSError:
        return 0.0


def write(workspace, text: str) -> Path:
    """Save the notes, atomically and under the file's lock."""
    path = notes_path(workspace)
    if path is None:
        raise ValueError("This workspace has no directory to keep notes in.")
    text = (text or "").rstrip() + "\n"
    with fileio.locked(path):
        fileio.atomic_write_text(path, text)
    return path


# ---- the recap ---------------------------------------------------------------

def _number(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if number != number else f"{number:.{digits}f}"


def recap(state, metrics_by_name: dict, when: datetime | None = None) -> str:
    """A few lines on how each experiment does, as Markdown: what a reader
    of the note needs to place it, not the Metrics tab again.
    `metrics_by_name`: experiment name -> metrics.ExperimentMetrics, in
    the order to list them."""
    when = when or datetime.now()
    lines = ["## Performance summary", ""]
    dataset = getattr(state, "dataset", None)
    if dataset is not None:
        name = Path(dataset.csv_path).name if dataset.csv_path else "dataset"
        lines.append(f"*{when:%Y-%m-%d %H:%M} -- {md.escape(name)}, {len(dataset.df):,} rows, "
                     f"{len(dataset.labels())} classes, {md.escape(state.split_kind)} pool.*")
        lines.append("")
    if not metrics_by_name:
        lines.append("No experiment loaded yet.")
        return "\n".join(lines)
    best = max(metrics_by_name, key=lambda n: _sortable(metrics_by_name[n].macro_f1))
    for name, m in metrics_by_name.items():
        line = (f"- **{md.escape(name)}** -- accuracy {_number(m.accuracy)} · macro F1 {_number(m.macro_f1)}"
                f" · macro AP {_number(m.macro_avg_precision)} · {m.n_samples:,} rows")
        if name == best and len(metrics_by_name) > 1:
            line += " (best macro F1)"
        lines.append(line)
    return "\n".join(lines)


def _sortable(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return float("-inf") if number != number else number


def template(title: str, recap_text: str) -> str:
    """A new note: the title, the recap, and room for the user's text."""
    return (f"# {md.escape(title)}\n\n{md.region_open(RECAP)}\n\n{recap_text}\n\n{md.region_close(RECAP)}\n\n"
            f"{NOTES_HEADING}\n\n")


def with_recap(text: str, recap_text: str) -> str:
    """`text` with its recap replaced by `recap_text` -- or, if the markers
    are gone, the recap put back under the title (at the top when there is
    no title)."""
    opening, closing = md.region_open(RECAP), md.region_close(RECAP)
    block = f"{opening}\n\n{recap_text}\n\n{closing}"
    start = text.find(opening)
    end = text.find(closing, start + 1) if start >= 0 else -1
    if start >= 0 and end >= 0:
        return text[:start] + block + text[end + len(closing):]
    lines = text.split("\n")
    title = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if title is None:
        return block + "\n\n" + text.lstrip("\n")
    return "\n".join(lines[:title + 1]) + "\n\n" + block + "\n\n" + "\n".join(lines[title + 1:]).lstrip("\n")


def has_content(text: str | None) -> bool:
    """Whether a note says anything a report should carry."""
    return bool(text and md.to_plain(text).strip())
