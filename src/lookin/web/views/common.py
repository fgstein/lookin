"""Pieces several analysis tabs draw the same way:

* uuid_overlap_banner / score_diagnostics_banner -- the warnings a tab shows
  before (or instead of) its numbers; grouped_warnings -- the same for
  several experiments at once, each kind of warning said once;
* status_line -- one success / error / warning / busy line;
* saved_path_message -- the path a "wrote it there" line shows, copied to
  the clipboard of the browser the user sits at;
* save_png -- write a picture next to the user's downloads on the server;
* confused_pairs_table / overview_image -- the confusion helpers;
* matrix_cell_size / matrix_text_size / matrix_label_chars /
  short_column_labels -- how a class grid is sized and labelled.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from lookin.core.class_focus import ZOOM_BLOCKS, overview_figure
from lookin.web import html as h
from lookin.web.views.media import CHART_DPI, chart_html, fig_png

CLIPBOARD_NOTE = " (path copied to the clipboard)"


def default_save_dir() -> Path:
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.cwd()


def slug(title: str, fallback: str = "image") -> str:
    """A panel title reduced to something safe in a filename and a URL."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:60] or fallback


# ---- class grids ------------------------------------------------------------

# The number of columns a confusion matrix is sized against when it has
# fewer: a two-class matrix draws four big cells instead of four small ones
# lost in a corner of the tab. Above this the cells keep their base size and
# the grid grows, which is what a many-class matrix needs.
MATRIX_FLOOR_COLUMNS = 8


def matrix_cell_size(n_columns: int, base: int, spacing: int = 2) -> int:
    """Cell pitch for a confusion matrix with `n_columns` columns.

    At MATRIX_FLOOR_COLUMNS columns or more this is `base`, so a matrix
    with many classes is as wide as it has to be. Below it, the cells take
    the room those columns would have used: the grid keeps the footprint of
    an eight-class one rather than shrinking to a thumbnail, which is what
    a binary matrix -- the most common case there is -- used to do.
    """
    if n_columns <= 0 or n_columns >= MATRIX_FLOOR_COLUMNS:
        return base
    return int(round(MATRIX_FLOOR_COLUMNS * (base + spacing) / n_columns)) - spacing


def matrix_text_size(cell_size: int, base_cell: int, base_size: int, cap: int = 18) -> float:
    """`base_size`, grown with the cell it sits in (see matrix_cell_size).

    Counts drawn at 10px inside a 140px cell would be a grid of small
    numbers floating in colour. Capped, because past a point a bigger digit
    stops being easier to read.
    """
    if cell_size <= base_cell or base_cell <= 0:
        return base_size
    return min(cap, round(base_size * cell_size / base_cell))


def matrix_label_chars(cell_size: int, base_cell: int, minimum: int) -> int:
    """How many characters of a class label a column header keeps.

    Headers are truncated to keep the grid on its pitch. `minimum` is what
    fits at the base pitch, and a cell that grew shows proportionally more
    -- the text in it grew too (see matrix_text_size), so the ratio is the
    honest estimate. The full label is in the cell's tooltip either way.
    """
    if base_cell <= 0 or cell_size <= base_cell:
        return minimum
    return max(minimum, int(minimum * cell_size / base_cell))


# Rows of the confused-pairs table shown under each matrix.
PAIRS_SHOWN = 15


def short_column_labels(labels: list, chars: int) -> tuple:
    """(short labels, dropped prefix) for the column headers of a grid.

    Headers are a few characters wide, and class names in a big taxonomy
    usually share a long prefix ("species_0412", "species_0413"), so
    cutting each one to its first characters makes every header identical.
    The prefix every label shares is dropped instead, then each is cut to
    `chars`; the caller says which prefix went, and the full label stays in
    the cell's tooltip.
    """
    names = [str(l) for l in labels]
    prefix = os.path.commonprefix(names) if len(names) > 1 else ""
    # Never strip a label down to nothing (one label a prefix of another),
    # and cut only after a separator: "Quercus r" out of "Quercus robur"
    # and "Quercus rubra" would leave "obur" and "ubra".
    prefix = prefix[: min((len(n) for n in names), default=0) - 1] if prefix else ""
    cut = max((i for i, ch in enumerate(prefix) if not ch.isalnum()), default=-1)
    prefix = prefix[: cut + 1]
    short = [n[len(prefix):] for n in names]
    return [s if len(s) <= chars else s[: chars - 1] + "…" for s in short], prefix


def _warn_box(lines: list, icon_name: str = "warn") -> str:
    return h.banner("", "warn", icon_name=icon_name, lines=lines)


def uuid_overlap_banner(overlap, dataset_label: str = "dataset", experiment_label: str = "experiment") -> str:
    lines = []
    if overlap.dataset_only:
        lines.append(f"{overlap.dataset_only:,} {dataset_label} rows have no matching {experiment_label} row")
    if overlap.experiment_only:
        lines.append(f"{overlap.experiment_only:,} {experiment_label} rows have no matching {dataset_label} row")
    if overlap.dataset_dupes:
        lines.append(f"{dataset_label} uuid column has {overlap.dataset_dupes:,} duplicate value(s)")
    if overlap.experiment_dupes:
        lines.append(f"{experiment_label} uuid column has {overlap.experiment_dupes:,} duplicate value(s)")
    return _warn_box(lines) if lines else ""


def score_diagnostics_banner(diagnostics, experiment_label: str = "experiment") -> str:
    lines = []
    if not diagnostics.normalized:
        lines.append(
            f"score columns do not sum to 1 ({diagnostics.n_rows_not_normalized:,} of "
            f"{diagnostics.n_rows:,} rows, worst off by {diagnostics.max_row_sum_deviation:.3f}"
            + (f"; {diagnostics.n_rows_missing:,} with a missing score" if diagnostics.n_rows_missing else "")
            + ")")
        lines.append("  -> disabled: log loss, Brier score and its decomposition, ECE / MCE, reliability diagram")
    if diagnostics.constant_labels:
        named = ", ".join(f"'{label}' (constant {diagnostics.constant_values[label]:g})"
                          for label in diagnostics.constant_labels)
        lines.append(f"score column(s) with no ranking: {named}")
        lines.append("  -> disabled for those classes: ROC-AUC, average precision, ROC / PR curves. "
                     "They are also excluded from the macro and weighted averages.")
    if not lines:
        return ""
    lines[0] = f"{experiment_label}: {lines[0]}"
    return _warn_box(lines, "list")


# ---- warnings grouped across experiments ------------------------------------
# A tab comparing ten experiments that all miss the same rows used to stack
# ten copies of the same banner above its numbers. Instead each kind of
# warning is one line naming the experiments it applies to, and what each
# one says in detail is shown on request.

def overlap_warnings(overlap) -> list:
    """(kind, title, detail, consequence) of a uuid match, with no
    experiment name in them, so that the same problem reads the same."""
    out = []
    if overlap.dataset_only:
        out.append(("dataset_only", "Dataset rows with no matching prediction",
                    f"{overlap.dataset_only:,} rows", ""))
    if overlap.experiment_only:
        out.append(("experiment_only", "Predictions with no matching dataset row",
                    f"{overlap.experiment_only:,} rows", ""))
    if overlap.dataset_dupes:
        out.append(("dataset_dupes", "Duplicate ids in the dataset",
                    f"{overlap.dataset_dupes:,} duplicate value(s)", ""))
    if overlap.experiment_dupes:
        out.append(("experiment_dupes", "Duplicate ids in the results",
                    f"{overlap.experiment_dupes:,} duplicate value(s)", ""))
    return out


def diagnostics_warnings(diagnostics) -> list:
    """(kind, title, detail, consequence) of the score diagnostics."""
    out = []
    if not diagnostics.normalized:
        detail = (f"{diagnostics.n_rows_not_normalized:,} of {diagnostics.n_rows:,} rows, worst off by "
                  f"{diagnostics.max_row_sum_deviation:.3f}"
                  + (f"; {diagnostics.n_rows_missing:,} with a missing score" if diagnostics.n_rows_missing else ""))
        out.append(("not_normalized", "Score columns do not sum to 1", detail,
                    "Disabled: log loss, Brier score and its decomposition, ECE / MCE, reliability diagram."))
    if diagnostics.constant_labels:
        named = ", ".join(f"'{label}' (constant {diagnostics.constant_values[label]:g})"
                          for label in diagnostics.constant_labels)
        out.append(("constant", "Score columns with no ranking", named,
                    "Disabled for those classes: ROC-AUC, average precision, ROC / PR curves. "
                    "They are also excluded from the macro and weighted averages."))
    return out


def grouped_warnings(by_experiment: dict, expanded: bool, toggle_attr: str) -> str:
    """One banner for the warnings of every experiment shown.
    `by_experiment`: name -> list of (kind, title, detail, consequence).

    Each kind is one line, with the number of experiments it concerns;
    expanded, its detail per experiment, and experiments whose detail is
    the same share a line. `toggle_attr` is the act attribute that flips
    `expanded`."""
    kinds: dict = {}
    for name, items in by_experiment.items():
        for kind, title, detail, consequence in items:
            entry = kinds.setdefault(kind, {"title": title, "consequence": consequence, "details": {}})
            entry["details"].setdefault(detail, []).append(name)
    if not kinds:
        return ""
    n_shown = len(by_experiment)
    lines = []
    for entry in kinds.values():
        names = [n for group in entry["details"].values() for n in group]
        whom = (f"all {n_shown} experiments" if len(names) == n_shown and n_shown > 1
                else names[0] if len(names) == 1 else f"{len(names)} experiments")
        line = f'<div><b>{h.e(entry["title"])}</b> <span class="small">-- {h.e(whom)}</span></div>'
        if expanded:
            for detail, group in entry["details"].items():
                line += (f'<div class="warn-detail small"><span class="mono">{h.e(", ".join(group))}</span>: '
                         f'{h.e(detail)}</div>')
            if entry["consequence"]:
                line += f'<div class="warn-detail small"><i>{h.e(entry["consequence"])}</i></div>'
        lines.append(line)
    label = "Hide details" if expanded else "Show details"
    button = h.button(label, act_attr=toggle_attr, small=True, kind="ghost",
                      icon_name="chev-down" if not expanded else "chev-up",
                      extra=f' aria-expanded="{"true" if expanded else "false"}"')
    count = len(kinds)
    head = f'<div class="small" style="opacity:.85">{count} warning{"s" if count != 1 else ""}</div>'
    return (f'<div class="banner banner-warn warn-group">{h.icon("warn", 15)}'
            f'<div class="banner-body">{head}{"".join(lines)}</div>'
            f'<div class="banner-actions">{button}</div></div>')


STATUS_KINDS = {"success": ("check", "c-ok"), "error": ("warn", "c-err"),
                "warning": ("warn", "c-warn"), "neutral": ("info", "muted")}


def status_line(text: str, kind: str = "neutral") -> str:
    if not text:
        return ""
    if kind == "busy":
        return h.spinner(text)
    icon_name, cls = STATUS_KINDS.get(kind, STATUS_KINDS["neutral"])
    return f'<div class="status {cls}">{h.icon(icon_name, 14)}<span>{h.e(text)}</span></div>'


def saved_path_message(session, path) -> str:
    session.cmd(copy=str(path))
    return f"{path}{CLIPBOARD_NOTE}"


def save_png(session, png: bytes, title: str) -> tuple:
    """(ok, message): `png` written next to the user's downloads, on the
    machine running Lookin (where the data is)."""
    dest = default_save_dir() / f"lookin_{slug(title)}_{datetime.now():%Y%m%d-%H%M%S}.png"
    try:
        dest.write_bytes(png)
    except OSError as ex:
        return False, f"Could not save: {ex}"
    return True, f"Saved to {saved_path_message(session, dest)}"


def confused_pairs_table(pairs, n_nonzero: int, act_for) -> str:
    """class_focus.top_confused_pairs as a table whose rows open the crops
    behind them; `act_for(true_label, predicted_label)` is the click."""
    if not len(pairs):
        return '<div class="muted small">No confusion at all: every crop is on the diagonal.</div>'
    rows = "".join(
        f'<tr class="clk"{act_for(true_label, predicted_label)}><td>{h.e(true_label)}</td>'
        f'<td>{h.e(predicted_label)}</td><td class="num">{int(count):,}</td><td class="num">{share * 100:.1f}%</td></tr>'
        for true_label, predicted_label, count, share in pairs.itertuples(index=False))
    return (f'<div class="small"><b>Most confused pairs -- {len(pairs)} of {n_nonzero:,} non-zero '
            f'off-diagonal cells</b></div>'
            f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr><th>True class</th><th>Predicted as</th>'
            f'<th class="num">Crops</th><th class="num">Of true class</th></tr></thead><tbody>{rows}</tbody></table></div>')


def overview_image(session, confusion, title: str, size: int, cache: dict, zoom_act: str) -> str:
    """The whole matrix as a `size`-pixel picture, drawn once per matrix;
    `zoom_act` draws the finer, labelled one on demand (see
    overview_zoom)."""
    key = (id(confusion), size, title)
    hit = cache.get(key)
    png = hit[1] if hit is not None and hit[0] is confusion else None
    if png is None:
        block = max(1, -(-len(confusion) // size))
        fig = overview_figure(confusion, [], title, block=block)
        side_inches = fig.get_size_inches()[0]
        png = fig_png(fig, dpi=max(40, int(2 * size / side_inches)))
        cache[key] = (confusion, png)
    ckey = session.put_chart(png)
    return (f'<button type="button" class="ov-btn" title="{h.e(title)} -- click to enlarge (takes a few seconds for many classes)"{zoom_act}>'
            f'<img src="/chart/{session.sid}/{ckey}.png" width="{size}" height="{size}" alt="{h.e(title)}" '
            f'data-static="{ckey}" style="object-fit:contain"></button>')


def overview_zoom(session, confusion, focus: list, title: str) -> None:
    """The enlarged overview, blocked finely enough that a lone confusion
    stays visible at fit-to-window."""
    fine_block = max(1, -(-len(confusion) // ZOOM_BLOCKS))
    png = fig_png(overview_figure(confusion, focus, title, block=fine_block), dpi=CHART_DPI)
    ckey = session.put_chart(png)
    session.cmd(zoom=f"/chart/{session.sid}/{ckey}.png", title=title, pixelated=True)


def card(title: str, body: str, *, right: str = "", key: str = "", cls: str = "", manual=None,
         width: str = "") -> str:
    """A titled card, with the (i) button of its manual page."""
    style = f' style="width:{width}"' if width else ""
    head = (f'<div class="card-h"><h2>{h.e(title)}</h2>{h.info_btn(manual) if manual else ""}'
            f'<div class="card-r">{right}</div></div>') if title else ""
    return (f'<section class="card {h.e(cls)}"{f" data-key={chr(34)}{h.e(key)}{chr(34)}" if key else ""}{style}>'
            f'{head}<div class="card-b">{body}</div></section>')


__all__ = ["uuid_overlap_banner", "score_diagnostics_banner", "overlap_warnings", "diagnostics_warnings",
           "grouped_warnings", "status_line", "saved_path_message",
           "save_png", "confused_pairs_table", "overview_image", "overview_zoom", "card", "chart_html"]
