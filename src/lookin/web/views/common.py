"""Pieces several analysis tabs draw the same way -- the web counterparts of
the helpers in lookin.ui.components and lookin.ui.class_matrix:

* uuid_overlap_banner / score_diagnostics_banner -- the warnings a tab shows
  before (or instead of) its numbers;
* status_line -- one success / error / warning / busy line;
* saved_path_message -- the path a "wrote it there" line shows, copied to
  the clipboard of the browser the user sits at;
* save_png -- write a picture next to the user's downloads on the server;
* confused_pairs_table / overview_image -- the confusion helpers.

The pure helpers (colours, matrix sizing) are reused from
lookin.ui.components as they are.
"""
from __future__ import annotations

from datetime import datetime

from lookin.core.class_focus import ZOOM_BLOCKS, overview_figure
from lookin.ui.components import _default_save_dir
from lookin.web import html as h
from lookin.web.views.media import CHART_DPI, chart_html, fig_png

CLIPBOARD_NOTE = " (path copied to the clipboard)"


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
            f"{diagnostics.n_rows:,} rows, worst off by {diagnostics.max_row_sum_deviation:.3f})")
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
    from lookin.ui.image_export import slug
    dest = _default_save_dir() / f"lookin_{slug(title)}_{datetime.now():%Y%m%d-%H%M%S}.png"
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
    """A titled card (the original's card() + section_title())."""
    style = f' style="width:{width}"' if width else ""
    head = (f'<div class="card-h"><h2>{h.e(title)}</h2>{h.info_btn(manual) if manual else ""}'
            f'<div class="card-r">{right}</div></div>') if title else ""
    return (f'<section class="card {h.e(cls)}"{f" data-key={chr(34)}{h.e(key)}{chr(34)}" if key else ""}{style}>'
            f'{head}<div class="card-b">{body}</div></section>')


__all__ = ["uuid_overlap_banner", "score_diagnostics_banner", "status_line", "saved_path_message",
           "save_png", "confused_pairs_table", "overview_image", "overview_zoom", "card", "chart_html"]
