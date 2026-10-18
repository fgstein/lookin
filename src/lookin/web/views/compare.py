"""Compare: two experiments, head to head, crop by crop.

Same tab as lookin.ui.compare_view, as HTML. A baseline picker and a
candidate picker, and an explicit Compare click (a comparison is a full
pass over every crop of both experiments) that runs in the background.
Then, in the order the question is asked:

1. the score-line -- how many crops changed verdict, in each direction,
   and how little of that the accuracy delta shows (four tiles that open
   their crops in the carousel);
2. the confusion matrix -- *where* the change landed, as the net
   difference or as the gross arrivals and departures it nets, with the
   most changed cells listed under it;
3. next to it, the provenance panel -- for the cell picked in that matrix,
   where its crops came from and where the ones it lost went;
4. the crop lists -- the regressions and the fixes themselves, most
   confident first, and the per-class table.

Every pandas pass over the joined frame happens in the background job or
in an action handler and is cached; render() only draws from the caches.
"""
from __future__ import annotations

from lookin.core import class_focus as cf
from lookin.core import comparison as cmp
from lookin.ui.class_matrix import short_column_labels
from lookin.ui.compare_view import (
    CELL_SIZE, CHANGED_CELLS_SHOWN, COMPARE_FOCUS_CRITERIA, CROP_LIST_PAGE, FLOW_CELL_SIZE,
    FLOW_COLUMN_WIDTH, FLOW_LABEL_COL_WIDTH, FLOW_MODE_LABELS, LABEL_COL_WIDTH, MATRIX_LABELS,
    OUTCOME_LABELS, PLACEHOLDER_PANEL_WIDTH, _signed,
)
from lookin.ui.components import (
    diverging_color, matrix_cell_size, matrix_label_chars, matrix_text_size,
)
from lookin.web import html as h
from lookin.web.views.base import RebuildView
from lookin.web.views.carousel import CarouselDialog
from lookin.web.views.common import status_line
from lookin.web.views.widgets import ClassFocus

# The original's Flet colours (RED_400, GREEN_400, ON_SURFACE_VARIANT,
# ORANGE_400), as the page's CSS values.
OUTCOME_COLORS = {
    cmp.OUTCOME_REGRESSION: "#E53935",
    cmp.OUTCOME_FIX: "#2E9E4F",
    cmp.OUTCOME_BOTH_RIGHT: "var(--muted)",
    cmp.OUTCOME_BOTH_WRONG: "#EF8A00",
}
FIX_COLOR = OUTCOME_COLORS[cmp.OUTCOME_FIX]
BROKEN_COLOR = OUTCOME_COLORS[cmp.OUTCOME_REGRESSION]
# The original's dark cell for "this crop never moved" in the class-wide grid.
UNMOVED_CELL = "#232329"
# ON_SURFACE_VARIANT of the original's dark theme: the muted text on those
# dark cells (the page's own --muted would vanish on them).
ON_DARK_MUTED = "#9d9aa6"

PLACEHOLDER_NO_DATA = ("Load a dataset and at least two experiments in the 'Data' tab -- "
                       "this tab compares one against another.")
PLACEHOLDER_PICK = "Pick a baseline and a candidate above, then click Compare."
STALE_DATA = "Data has changed -- this comparison is out of date."
STALE_PAIR = "Pair changed -- click Compare to update."

# Icons the shared set does not have (Flet's TRENDING_DOWN/UP,
# CHECK_CIRCLE_OUTLINE, REMOVE_CIRCLE_OUTLINE, SWAP_HORIZ, COMPARE_ARROWS,
# CALL_SPLIT, LOGIN, LOGOUT, TOUCH_APP), drawn inline like html.icon.
_ICONS = {
    "trend-down": '<path d="M3 7l6 6 4-4 8 8M15 17h6v-6"/>',
    "trend-up": '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>',
    "check-circle": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12.5l3 3 5-6"/>',
    "remove-circle": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12h8"/>',
    "swap": '<path d="M4 8h15M15 4l4 4-4 4M20 16H5M9 12l-4 4 4 4"/>',
    "compare": '<path d="M4 8h12M12 4l4 4-4 4M20 16H8M12 12l-4 4 4 4"/>',
    "split": '<path d="M12 21v-8L6 7M6 7V11M6 7h4M12 13l6-6M18 7v4M18 7h-4"/>',
    "login": '<path d="M10 17l5-5-5-5M15 12H3M14 4h5a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-5"/>',
    "logout": '<path d="M16 17l5-5-5-5M21 12H9M10 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h5"/>',
    "touch": '<path d="M9 11V5.5a1.5 1.5 0 0 1 3 0V11M12 10.5a1.5 1.5 0 0 1 3 0V12M15 11.5a1.5 1.5 0 0 1 3 0V15a6 6 0 0 1-6 6h-1a6 6 0 0 1-5-2.7L4 15.5a1.5 1.5 0 0 1 2.5-1.6L9 16"/>',
    "burst": '<rect x="7" y="7" width="13" height="12" rx="1.5"/><path d="M4 16V5.5A1.5 1.5 0 0 1 5.5 4H17"/>',
}
OUTCOME_ICONS = {
    cmp.OUTCOME_REGRESSION: "trend-down",
    cmp.OUTCOME_FIX: "trend-up",
    cmp.OUTCOME_BOTH_RIGHT: "check-circle",
    cmp.OUTCOME_BOTH_WRONG: "remove-circle",
}


def _icon(name: str, size: int = 15, color: str = "") -> str:
    path = _ICONS.get(name)
    if path is None:
        return h.icon(name, size)
    style = f' style="color:{color}"' if color else ""
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true"{style}>{path}</svg>')


def _icon_button(label: str, icon_name: str, act_attr: str, *, kind: str = "", disabled: bool = False,
                 title: str = "", small: bool = False) -> str:
    """h.button with one of the local icons above."""
    classes = " ".join(c for c in ("btn", f"btn-{kind}" if kind else "", "btn-sm" if small else "",
                                   "btn-icon" if not label else "") if c)
    inner = _icon(icon_name, 14 if small else 15) + (f"<span>{h.e(label)}</span>" if label else "")
    return (f'<button type="button" class="{classes}"{act_attr}'
            f'{h.attrs(title=title or None, disabled=disabled, aria_label=(title if not label else None))}>'
            f'{inner}</button>')


def _hint(text: str) -> str:
    return f'<div class="cmp-hint">{_icon("touch", 13)}<span>{h.e(text)}</span></div>'


class CompareView(RebuildView):
    name = "compare"
    title = "Compare"
    stale_message = STALE_DATA

    def __init__(self, session):
        super().__init__(session)
        self.baseline_value = None
        self.candidate_value = None
        self.matrix_mode = cmp.MATRIX_NET
        self.selected_cell = None          # (true, predicted) labels, or None
        self.flow_mode = cmp.FLOW_CELL
        self.class_focus = ClassFocus(self.name, "focus", COMPARE_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the matrix and the per-class table")
        self._focus: list = []
        self.stale_text = ""               # the red banner's text, "" = hidden

        # What one Compare click produced (all replaced together).
        self._joined = None
        self._diff = None
        self._compared_pair = None
        self._labels: list = []            # the matrices' class order
        self._outcome_rows: dict = {}      # outcome key -> rows_for_outcome frame
        self._per_class_table = None
        self._changed_cells = None         # top_confused_pairs over moved in + out
        self._flow_cache: dict = {}        # (cell, mode) -> what the provenance panel draws
        self.showing_results = False       # False: the placeholder instead of the results
        self.compare_error = ""

    # ---- experiment pickers ---------------------------------------------

    def _sync_pair(self) -> None:
        """The original's _rebuild_dropdowns: keep the user's choice while
        it survives, default to the first two experiments."""
        names = list(self.state.experiments)
        for attr, default_index in (("baseline_value", 0), ("candidate_value", 1)):
            value = getattr(self, attr)
            if value not in names:
                value = names[default_index] if len(names) > default_index else None
            setattr(self, attr, value)

    def _pair_is_valid(self) -> bool:
        a, b = self.baseline_value, self.candidate_value
        loaded = self.state.experiments
        return bool(a and b and a in loaded and b in loaded)

    def _on_pair_change(self) -> None:
        if self._compared_pair and self._compared_pair != (self.baseline_value, self.candidate_value):
            self.stale_text = STALE_PAIR

    def on_baseline(self, value: str = "", **_):
        self.baseline_value = value or None
        self._on_pair_change()

    def on_candidate(self, value: str = "", **_):
        self.candidate_value = value or None
        self._on_pair_change()

    def on_swap(self, **_):
        self.baseline_value, self.candidate_value = self.candidate_value, self.baseline_value
        self._sync_pair()
        self._on_pair_change()

    # ---- lifecycle --------------------------------------------------------

    def on_shown(self) -> None:
        self._sync_pair()
        if self._diff is None or self.dirty:
            self.showing_results = False
            self.class_focus.hide()

    def on_data_change(self) -> None:
        """Flag, never recompute -- joining two merged frames is real work."""
        self.dirty = True
        self._sync_pair()
        self.stale_text = STALE_DATA

    def stale_html(self) -> str:
        if not self.stale_text or self._diff is None:
            return ""
        return h.banner(self.stale_text, "stale",
                        actions=h.button("Update now", act_attr=self.a("update"), icon_name="refresh", small=True))

    def on_update(self, **_):
        self.on_compare()

    def on_compare(self, **_):
        if not self._pair_is_valid():
            return
        self.progress_text = "Comparing..."
        self.rebuild(self._compare_now)

    def recompute_after_restore(self) -> None:
        if self._pair_is_valid():
            self.on_compare()

    def _compare_now(self) -> None:
        baseline, candidate = self.baseline_value, self.candidate_value
        try:
            self.set_progress(f"Joining {baseline} and {candidate}...")
            joined = cmp.build_comparison(self.state, baseline, candidate)
            self.set_progress("Summarising the differences...")
            diff = cmp.compare_experiments(self.state, baseline, candidate, joined=joined)
            print(f"[lookin] compare: {baseline} vs {candidate} -- {diff.n_common:,} common crops, "
                  f"{diff.counts[cmp.OUTCOME_FIX]:,} fixed, {diff.counts[cmp.OUTCOME_REGRESSION]:,} broken",
                  flush=True)
            self.set_progress("Drawing...")
            outcome_rows = {o: cmp.rows_for_outcome(joined, o)
                            for o in (cmp.OUTCOME_REGRESSION, cmp.OUTCOME_FIX,
                                      cmp.OUTCOME_BOTH_RIGHT, cmp.OUTCOME_BOTH_WRONG)}
            per_class = cmp.per_class_change(joined, diff.labels)
            moved = diff.moved_in_confusion + diff.moved_out_confusion
            changed = cf.top_confused_pairs(moved, include_diagonal=True) if not moved.empty else None
        except Exception as ex:
            print(f"[lookin] compare failed: {ex}", flush=True)
            self.compare_error = f"Error comparing experiments: {ex}"
            self.showing_results = False
            self.class_focus.hide()
            raise
        self._joined, self._diff = joined, diff
        self._compared_pair = (baseline, candidate)
        self._labels = list(diff.delta_confusion.index)
        self._outcome_rows = outcome_rows
        self._per_class_table = per_class
        self._changed_cells = changed
        self._flow_cache = {}
        self.compare_error = ""
        self._resolve_focus()
        self._check_selected_cell()
        self._flow_data()
        self.showing_results = True
        self.stale_text = ""

    # ---- focus set: which classes the matrix and table show ---------------

    def _resolve_focus(self) -> None:
        diff = self._diff
        if diff is None:
            return
        table = self._per_class_table
        changed = self._changed_cells

        def rank(criterion):
            if criterion == cf.FOCUS_MOST_CHANGED and changed is not None:
                return cf.pairs_to_focus(changed[["true", "predicted"]].itertuples(index=False))
            if criterion == cf.FOCUS_LARGEST_SUPPORT and table is not None and len(table):
                return cf.rank_by(table, "support", ascending=False)
            return []

        self._focus = self.class_focus.resolve(list(diff.delta_confusion.index), rank)

    def _on_focus_change(self) -> None:
        if self._diff is None:
            return
        self._resolve_focus()

    def on_focus(self, op: str = "", **args):
        self.class_focus.handle(op, args)

    # ---- the selected cell --------------------------------------------------

    def _label(self, index):
        try:
            i = int(index)
        except (TypeError, ValueError):
            return None
        return self._labels[i] if 0 <= i < len(self._labels) else None

    def _check_selected_cell(self) -> None:
        """A selection made against another pair (or read from a snapshot,
        where an int label came back as a string) is matched by text, and
        dropped when this pair does not have its classes."""
        if self.selected_cell is None:
            return
        by_text = {str(l): l for l in self._labels}
        resolved = tuple(by_text.get(str(l)) for l in self.selected_cell)
        self.selected_cell = resolved if all(l is not None for l in resolved) else None

    def on_cell(self, t=None, p=None, **_):
        """Select the cell (clicking it again clears the selection)."""
        tl, pl = self._label(t), self._label(p)
        if tl is None or pl is None:
            return
        picked = (tl, pl)
        self.selected_cell = None if self.selected_cell == picked else picked
        self._flow_data()

    def on_matrix(self, key: str = "", **_):
        if key in cmp.MATRIX_VIEWS:
            self.matrix_mode = key

    def on_flow(self, key: str = "", **_):
        if key in cmp.FLOW_MODES:
            self.flow_mode = key
            self._flow_data()

    def _flow_data(self):
        """What the provenance panel draws for the selected cell, computed
        once per (cell, reading)."""
        if self.selected_cell is None or self._joined is None:
            return None
        key = (self.selected_cell, self.flow_mode)
        hit = self._flow_cache.get(key)
        if hit is not None:
            return hit
        true_label, predicted_label = self.selected_cell
        diff = self._diff
        if self.flow_mode == cmp.FLOW_CELL:
            data = cmp.cell_flows(self._joined, diff.labels, true_label, predicted_label)
        else:
            flow = cmp.flow_matrix(self._joined, diff.labels, true_label)
            labels = cmp.flow_labels(flow, true_label)
            if predicted_label in flow.index and predicted_label not in labels:
                keep = set(labels) | {predicted_label}
                labels = [label for label in flow.index if label in keep]
            data = (flow, labels)
        self._flow_cache[key] = data
        return data

    # ---- rendering ----------------------------------------------------------

    def render(self) -> str:
        self._sync_pair()
        return (f'<div class="stack cmp">{self.stale_html()}{self._head_html()}'
                f'{self._results_html()}</div>')

    def _head_html(self) -> str:
        names = list(self.state.experiments)
        options = [(n, n) for n in names]
        picker = (f'<div class="row cmp-pick">'
                  f'{h.select(options, self.baseline_value or "", change=self.c("baseline"), label="Baseline (A)", cls="cmp-sel")}'
                  f'{_icon_button("", "swap", self.a("swap"), kind="ghost", title="Swap baseline and candidate")}'
                  f'{h.select(options, self.candidate_value or "", change=self.c("candidate"), label="Candidate (B)", cls="cmp-sel")}'
                  f'{_icon_button("Compare", "compare", self.a("compare"), kind="primary", disabled=not self._pair_is_valid())}'
                  f'</div>')
        progress = self.progress_html() if self.progress_text else ""
        return h.section("Compare two experiments", picker + progress + self._summary_html(),
                         manual="fixes-regressions")

    def _summary_html(self) -> str:
        if self.compare_error and not self.showing_results:
            return f'<div class="cmp-gap">{h.banner(self.compare_error, "error")}</div>'
        if not self.showing_results or self._diff is None:
            state = self.state
            message = PLACEHOLDER_NO_DATA if (state.dataset is None or len(state.experiments) < 2) else PLACEHOLDER_PICK
            return f'<div class="cmp-gap">{h.banner(message, "info")}</div>'
        diff = self._diff
        counts = diff.counts
        changed = counts[cmp.OUTCOME_REGRESSION] + counts[cmp.OUTCOME_FIX]
        headline = (f"{changed:,} crop(s) changed verdict "
                    f"({counts[cmp.OUTCOME_FIX]:,} fixed, {counts[cmp.OUTCOME_REGRESSION]:,} broken) "
                    f"— net {diff.net_change():+,}")
        accuracy_line = (f"Accuracy {diff.baseline_accuracy:.4f} → {diff.candidate_accuracy:.4f} "
                         f"({diff.accuracy_delta:+.4f})")
        tiles = "".join(self._score_tile(o, counts[o], diff.n_common)
                        for o in (cmp.OUTCOME_REGRESSION, cmp.OUTCOME_FIX,
                                  cmp.OUTCOME_BOTH_RIGHT, cmp.OUTCOME_BOTH_WRONG))
        notes = []
        if diff.n_baseline_only or diff.n_candidate_only:
            notes.append(status_line(
                f"Compared on the {diff.n_common:,} crop(s) both cover. "
                f"{diff.n_baseline_only:,} only in '{diff.baseline}', "
                f"{diff.n_candidate_only:,} only in '{diff.candidate}' -- those are excluded, "
                f"since counts over different row sets cannot be compared.", "warning"))
        if self.state.use_reviewed_labels:
            notes.append(status_line(
                "Both models are judged against the reviewed labels (the overlay is on).", "neutral"))
        return (f'<div class="cmp-sum">'
                f'<div class="cmp-pair"><b>{h.e(diff.baseline)}</b>{h.icon("arrow-right", 15, "muted")}'
                f'<b>{h.e(diff.candidate)}</b></div>'
                f'<div class="cmp-headline">{h.e(headline)}</div>'
                f'<div class="muted">{h.e(accuracy_line)}</div>'
                f'<div class="muted small"><i>The accuracy delta only shows the net; these four numbers '
                f'say what produced it.</i></div>'
                f'<div class="cmp-tiles">{tiles}</div>{"".join(notes)}</div>')

    def _score_tile(self, outcome: str, count: int, total: int) -> str:
        share = f"{100 * count / total:.1f}%" if total else "--"
        color = OUTCOME_COLORS[outcome]
        act_attr = self.a("open_outcome", outcome=outcome) if count else ""
        return (f'<button type="button" class="cmp-tile" style="border-color:{color}"{act_attr}'
                f'{h.attrs(disabled=not count, title="Open these crops in the carousel" if count else None)}>'
                f'<span class="cmp-tile-h" style="color:{color}">{_icon(OUTCOME_ICONS[outcome], 15)}'
                f'<span>{h.e(OUTCOME_LABELS[outcome])}</span></span>'
                f'<span class="cmp-tile-n">{count:,}</span>'
                f'<span class="muted small">{h.e(share)}</span></button>')

    def _results_html(self) -> str:
        if not self.showing_results or self._diff is None:
            return ""
        diff = self._diff
        return (f'{self.class_focus.render()}{self._matrix_section(diff)}'
                f'{self._lists_section(diff)}{self._per_class_section(diff)}')

    # ---- the matrix -----------------------------------------------------------

    def _matrix_improvement(self, true_label, predicted_label, value: int) -> int:
        if self.matrix_mode == cmp.MATRIX_MOVED_OUT:
            value = -value
        return cmp.cell_improvement(true_label, predicted_label, value)

    def _matrix_cell_text(self, value: int) -> str:
        if not value:
            return "·"
        return _signed(-value if self.matrix_mode == cmp.MATRIX_MOVED_OUT else value)

    def _matrix_caption(self, diff) -> str:
        if self.matrix_mode == cmp.MATRIX_MOVED_IN:
            return (f"New cases: crops '{diff.candidate}' puts in a cell '{diff.baseline}' had put "
                    "them elsewhere than. Nothing is netted here, so a cell that took in twelve "
                    "crops and lost twelve others still shows the twelve that arrived. On the "
                    "diagonal an arrival is a crop the candidate fixed; anywhere else it is one "
                    "it broke or moved from one mistake to another.")
        if self.matrix_mode == cmp.MATRIX_MOVED_OUT:
            return (f"Vanished cases: crops '{diff.baseline}' had in a cell and '{diff.candidate}' "
                    "puts elsewhere. Off the diagonal a departure is a confusion cleared up; on "
                    "the diagonal it is a crop that used to be right and is not any more.")
        return (f"Each cell is '{diff.candidate}' minus '{diff.baseline}'. Green is an improvement, "
                "red a regression -- on the diagonal more is better, everywhere else more is "
                "worse, so the colour follows the news rather than the sign. This nets arrivals "
                "against departures: switch above to see either one on its own.")

    def _matrix_cell_tooltip(self, diff, true_label, predicted_label, value: int) -> str:
        head = f"true = {true_label}\npredicted = {predicted_label}\n"
        if self.matrix_mode == cmp.MATRIX_MOVED_IN:
            return (f"{head}{value:+,d} crop(s): '{diff.candidate}' moved them into this cell\n"
                    "(click to see where they came from)")
        if self.matrix_mode == cmp.MATRIX_MOVED_OUT:
            return (f"{head}{-value:+,d} crop(s): '{diff.baseline}' had them here and "
                    f"'{diff.candidate}' moved them away\n(click to see where they went)")
        diagonal_note = ("more correct is better" if true_label == predicted_label
                         else "more of this confusion is worse")
        return f"{head}{diff.candidate} - {diff.baseline} = {value:+d}\n({diagonal_note})"

    def _caption(self, text: str) -> str:
        """The original's caption, kept whole but clamped to two lines (the
        full text is on hover): a tab carries one line under a chart."""
        return f'<div class="cmp-cap" title="{h.e(text)}">{h.e(text)}</div>'

    def _matrix_switch(self, diff) -> str:
        totals = {cmp.MATRIX_NET: diff.net_change(), cmp.MATRIX_MOVED_IN: diff.n_moved,
                  cmp.MATRIX_MOVED_OUT: diff.n_moved}
        options = [(key, f"{MATRIX_LABELS[key]} ({totals[key]:+,})" if key == cmp.MATRIX_NET
                    else f"{MATRIX_LABELS[key]} ({totals[key]:,})") for key in cmp.MATRIX_VIEWS]
        return h.segmented(options, self.matrix_mode, act_for=lambda k: self.a("matrix", key=k),
                           label="Matrix")

    def _matrix_section(self, diff) -> str:
        matrix = diff.confusion(self.matrix_mode)
        if matrix.empty:
            return ""
        all_labels = list(matrix.index)
        index_of = {l: i for i, l in enumerate(self._labels)}
        many = self.class_focus.many
        present = set(all_labels)
        labels = [l for l in self._focus if l in present] if many else all_labels
        values = matrix.loc[labels, labels].to_numpy()
        other = (matrix.loc[labels].to_numpy().sum(axis=1) - values.sum(axis=1)) if many else None
        max_abs = int(max(abs(values.min()), abs(values.max()), 1)) if values.size else 1

        n_cols = len(labels) + (1 if many else 0)
        cell = matrix_cell_size(n_cols, CELL_SIZE)
        value_size = matrix_text_size(cell, CELL_SIZE, 10)
        header_size = matrix_text_size(cell, CELL_SIZE, 9, cap=14)
        header_chars = matrix_label_chars(cell, CELL_SIZE, 5)
        row_label_size = matrix_text_size(cell, CELL_SIZE, 10, cap=14)
        short_labels, dropped_prefix = short_column_labels(labels, 2 * header_chars)

        out = [f'<div style="width:{LABEL_COL_WIDTH}px"></div>']
        for short, full in zip(short_labels, labels):
            out.append(f'<div class="cmp-colh" style="font-size:{header_size}px;height:{2.6 * header_size + 4:.0f}px"'
                       f' title="{h.e(full)}"><span>{h.e(short)}</span></div>')
        if many:
            out.append(f'<div class="cmp-colh" style="font-size:{header_size}px;height:{2.6 * header_size + 4:.0f}px"'
                       f' title="Any class not shown in this grid"><span><i>other</i></span></div>')
        for i, tl in enumerate(labels):
            out.append(f'<div class="cmp-rowh" style="font-size:{row_label_size}px" title="{h.e(tl)}">{h.e(tl)}</div>')
            for j, pl in enumerate(labels):
                value = int(values[i, j])
                improvement = self._matrix_improvement(tl, pl, value)
                picked = self.selected_cell == (tl, pl)
                color = "#ffffff" if abs(improvement) > max_abs * 0.4 else "#c9c9d1"
                out.append(
                    f'<div class="cmp-cell clk{" on" if picked else ""}" style="background:{diverging_color(improvement, max_abs)};'
                    f'color:{color};font-size:{value_size}px;height:{cell}px"'
                    f' title="{h.e(self._matrix_cell_tooltip(diff, tl, pl, value))}"'
                    f'{self.a("cell", t=index_of[tl], p=index_of[pl])}>{h.e(self._matrix_cell_text(value))}</div>')
            if many:
                value = int(other[i])
                out.append(f'<div class="cmp-cell cmp-other" style="font-size:{value_size}px;height:{cell}px"'
                           f' title="{h.e(f"true = {tl}\npredicted = any class not in this grid\n{value:+,d} in total")}">'
                           f'<i>{h.e(self._matrix_cell_text(value))}</i></div>')
        grid = (f'<div class="cmp-scroll"><div class="cmp-grid" style="grid-template-columns:'
                f'{LABEL_COL_WIDTH}px repeat({n_cols},{cell}px)">{"".join(out)}</div></div>')
        hint = _hint("Click a cell to see which cells its crops came from and went to"
                     + (f" -- column headers leave out the shared prefix '{dropped_prefix}'"
                        if dropped_prefix else ""))
        matrix_panel = (f'<div class="cmp-panel">{self._matrix_switch(diff)}'
                        f'{self._caption(self._matrix_caption(diff))}{grid}{hint}'
                        f'{self._changed_cells_html(diff)}</div>')
        body = f'<div class="cmp-wrap">{matrix_panel}{self._flow_panel(diff)}</div>'
        return h.section("Where the change landed", body, manual="delta-confusion")

    def _changed_cells_html(self, diff) -> str:
        cells = self._changed_cells
        if cells is None or not len(cells):
            return '<div class="muted small">No crop changed prediction.</div>'
        shown = cells.head(CHANGED_CELLS_SHOWN)
        index_of = {l: i for i, l in enumerate(self._labels)}
        rows = []
        for true_label, predicted_label, _traffic, _share in shown.itertuples(index=False):
            moved_in = int(diff.moved_in_confusion.loc[true_label, predicted_label])
            moved_out = int(diff.moved_out_confusion.loc[true_label, predicted_label])
            net = moved_in - moved_out
            bg = diverging_color(cmp.cell_improvement(true_label, predicted_label, net), max(abs(net), 1))
            sel = " sel" if self.selected_cell == (true_label, predicted_label) else ""
            rows.append(
                f'<tr class="clk{sel}"{self.a("cell", t=index_of[true_label], p=index_of[predicted_label])}>'
                f'<td>{h.e(true_label)}</td><td>{h.e(predicted_label)}</td>'
                f'<td class="num">{moved_in:+,d}</td><td class="num">{-moved_out:+,d}</td>'
                f'<td class="num"><span class="cmp-chip" style="background:{bg}">{net:+,d}</span></td></tr>')
        return (f'<div class="small cmp-gap"><b>Most changed cells -- {len(shown)} of {len(cells):,} cells '
                f'where a crop moved</b></div>'
                f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr><th>True class</th>'
                f'<th>Predicted as</th><th class="num">Moved in</th><th class="num">Moved out</th>'
                f'<th class="num">Net</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    # ---- provenance panel ---------------------------------------------------------

    def _flow_panel(self, diff) -> str:
        if self.selected_cell is None:
            return (f'<div class="cmp-panel" style="width:{PLACEHOLDER_PANEL_WIDTH}px">'
                    f'<div class="cmp-ptitle">Where they came from</div>'
                    f'<div class="muted small">Pick a cell on the left. This panel then shows which cells '
                    f'its crops arrived from and which ones the crops it lost went to.</div></div>')
        true_label, predicted_label = self.selected_cell
        moved_in = int(diff.moved_in_confusion.loc[true_label, predicted_label])
        moved_out = int(diff.moved_out_confusion.loc[true_label, predicted_label])
        data = self._flow_data()
        if self.flow_mode == cmp.FLOW_CELL:
            grid, caption = self._cell_flows_grid(diff, data, true_label, predicted_label)
        else:
            grid, caption = self._class_flow_grid(diff, data, true_label, predicted_label)
        switch = h.segmented([(key, FLOW_MODE_LABELS[key]) for key in cmp.FLOW_MODES], self.flow_mode,
                             act_for=lambda k: self.a("flow", key=k), label="Reading")
        buttons = (
            _icon_button(f"Open the {moved_in:,} that moved in", "login", self.a("open_movement", direction=cmp.MATRIX_MOVED_IN),
                         kind="primary", disabled=not moved_in, small=True)
            + _icon_button(f"Open the {moved_out:,} that moved out", "logout", self.a("open_movement", direction=cmp.MATRIX_MOVED_OUT),
                           kind="primary", disabled=not moved_out, small=True))
        return (f'<div class="cmp-panel">'
                f'<div class="cmp-ptitle">{_icon("split", 16, "var(--acc)")}<span>Where they came from</span>'
                f'<span class="c-acc cmp-psub">true {h.e(true_label)} / predicted {h.e(predicted_label)}</span></div>'
                f'{switch}'
                f'<div>{h.e(f"{_signed(moved_in)} moved in, {_signed(-moved_out)} moved out — net {_signed(moved_in - moved_out)}")}</div>'
                f'{self._caption(caption)}<div class="cmp-scroll">{grid}</div>'
                f'<div class="row">{buttons}</div>'
                f'{_hint("Click a cell here to open exactly that move in the carousel")}</div>')

    def _flow_cell(self, true_label, from_label, to_label, value: int, text: str, bg: str, color: str,
                   width: int, height: int, font: float, outlined: bool = False) -> str:
        idx = {l: i for i, l in enumerate(self._labels)}
        act_attr = self.a("open_flow", f=idx[from_label], t=idx[to_label]) if value else ""
        return (f'<div class="cmp-cell{" clk" if value else ""}{" cmp-out" if outlined else ""}" '
                f'style="background:{bg};color:{color};font-size:{font}px;height:{height}px;width:{width}px"'
                f' title="{h.e(self._flow_tooltip(true_label, from_label, to_label, value))}"{act_attr}>'
                f'{h.e(text)}</div>')

    def _cell_flows_grid(self, diff, table, true_label, predicted_label):
        caption = (f"Only the crops that touch this cell, by the class they came from or went to. "
                   f"Rows are '{diff.baseline}'s class for an arrival and '{diff.candidate}'s for "
                   f"a departure; a move onto {true_label} is a fix, one away from it a breakage. "
                   "Busiest first.")
        cols = f"{FLOW_LABEL_COL_WIDTH}px {FLOW_COLUMN_WIDTH}px {FLOW_COLUMN_WIDTH}px"
        out = ['<div></div>', '<div class="cmp-fh">came from</div>', '<div class="cmp-fh">went to</div>']
        if table is None or table.empty:
            return (f'<div class="cmp-grid" style="grid-template-columns:{cols}">{"".join(out)}</div>'
                    f'<div class="muted small"><i>Nothing moved through this cell.</i></div>', caption)
        max_abs = max(int(table.to_numpy().max()), 1)
        for label, row in table.iterrows():
            came, went = int(row[cmp.COL_CAME_FROM]), int(row[cmp.COL_WENT_TO])
            out.append(f'<div class="cmp-rowh" style="font-size:12px" title="{h.e(label)}">{h.e(label)}</div>')
            for from_label, to_label, value, arriving in ((label, predicted_label, came, True),
                                                          (predicted_label, label, went, False)):
                improvement = cmp.flow_improvement(true_label, from_label, to_label, value)
                text = "·" if not value else _signed(value if arriving else -value)
                out.append(self._flow_cell(true_label, from_label, to_label, value, text,
                                           diverging_color(improvement, max_abs),
                                           "#ffffff" if improvement else ON_DARK_MUTED,
                                           FLOW_COLUMN_WIDTH, FLOW_CELL_SIZE, 12))
        return f'<div class="cmp-grid" style="grid-template-columns:{cols}">{"".join(out)}</div>', caption

    def _class_flow_grid(self, diff, data, true_label, predicted_label):
        flow, labels = data
        caption = (f"Every crop whose true class is {true_label}, not only the ones that touch "
                   f"the picked cell. Rows: what '{diff.baseline}' predicted. Columns: what "
                   f"'{diff.candidate}' predicts. The outlined column is the picked cell's "
                   "arrivals and the outlined row its departures -- those carry a sign, the "
                   "other moves do not touch that cell's count. The diagonal never moved.")
        values = flow.loc[labels, labels].to_numpy()
        moves = [abs(int(values[i, j])) for i in range(len(labels)) for j in range(len(labels))
                 if labels[i] != labels[j]]
        max_abs = max(moves or [0], default=0) or 1
        cell = matrix_cell_size(len(labels), FLOW_CELL_SIZE)
        value_size = matrix_text_size(cell, FLOW_CELL_SIZE, 10)
        header_size = matrix_text_size(cell, FLOW_CELL_SIZE, 9, cap=14)
        header_chars = matrix_label_chars(cell, FLOW_CELL_SIZE, 5)
        row_label_size = matrix_text_size(cell, FLOW_CELL_SIZE, 10, cap=14)
        acc = lambda label: ' c-acc' if label == predicted_label else ''   # noqa: E731
        out = ['<div></div>']
        for label in labels:
            out.append(f'<div class="cmp-fh{acc(label)}" style="font-size:{header_size}px" title="{h.e(label)}">'
                       f'{h.e(str(label)[:header_chars])}</div>')
        for i, from_label in enumerate(labels):
            out.append(f'<div class="cmp-rowh{acc(from_label)}" style="font-size:{row_label_size}px" '
                       f'title="{h.e(from_label)}">{h.e(from_label)}</div>')
            for j, to_label in enumerate(labels):
                value = int(values[i, j])
                improvement = cmp.flow_improvement(true_label, from_label, to_label, value)
                moved = from_label != to_label
                outlined = moved and (to_label == predicted_label or from_label == predicted_label)
                color = (("#ffffff" if abs(improvement) > max_abs * 0.4 else "#c9c9d1") if moved
                         else ON_DARK_MUTED)
                out.append(self._flow_cell(
                    true_label, from_label, to_label, value,
                    self._class_flow_text(value, from_label, to_label, predicted_label, moved),
                    diverging_color(improvement, max_abs) if moved else UNMOVED_CELL, color,
                    cell, cell, value_size, outlined))
        cols = f"{FLOW_LABEL_COL_WIDTH}px repeat({len(labels)},{cell}px)"
        return f'<div class="cmp-grid" style="grid-template-columns:{cols}">{"".join(out)}</div>', caption

    @staticmethod
    def _class_flow_text(value: int, from_label, to_label, predicted_label, moved: bool) -> str:
        if not value:
            return "·"
        if moved and to_label == predicted_label:
            return _signed(value)
        if moved and from_label == predicted_label:
            return _signed(-value)
        return f"{value:,}"

    def _flow_tooltip(self, true_label, from_label, to_label, value: int) -> str:
        diff = self._diff
        head = (f"true = {true_label}\n{diff.baseline}: {from_label}\n"
                f"{diff.candidate}: {to_label}\n{value:,} crop(s)")
        if not value:
            return head
        if from_label == to_label:
            return f"{head}\n(unchanged)"
        if to_label == true_label:
            return f"{head}\n(fixed by the candidate)"
        if from_label == true_label:
            return f"{head}\n(broken by the candidate)"
        return f"{head}\n(one mistake traded for another)"

    # ---- crop lists and the per-class table -----------------------------------------

    def _lists_section(self, diff) -> str:
        body = (f'<div class="cmp-wrap">{self._outcome_panel(cmp.OUTCOME_REGRESSION)}'
                f'{self._outcome_panel(cmp.OUTCOME_FIX)}</div>')
        return h.section("What changed, crop by crop", body, manual="fixes-regressions")

    def _outcome_panel(self, outcome: str) -> str:
        rows = self._outcome_rows.get(outcome)
        n = 0 if rows is None else len(rows)
        color = OUTCOME_COLORS[outcome]
        uuid_col = self.state.dataset.columns.uuid_col if self.state.dataset is not None else None
        items = []
        if not n:
            items.append('<div class="muted small"><i>None.</i></div>')
        else:
            for row in rows.head(CROP_LIST_PAGE).to_dict("records"):
                uuid = str(row.get(uuid_col, "")) if uuid_col else ""
                items.append(
                    f'<button type="button" class="cmp-crop" title="Open this crop in the carousel"'
                    f'{self.a("open_outcome", outcome=outcome, uuid=uuid)}>'
                    f'<span class="mono cmp-uuid">{h.e(uuid)}</span>'
                    f'<span class="cmp-crop-l"><b>true {h.e(row[cmp.COL_TRUE])}</b>'
                    f'<span class="muted">·  {h.e(self._diff.baseline)}: {h.e(row[cmp.COL_PRED_A])} '
                    f'({row[cmp.COL_SCORE_A]:.2f})</span>'
                    f'<span style="color:{color}">·  {h.e(self._diff.candidate)}: {h.e(row[cmp.COL_PRED_B])} '
                    f'({row[cmp.COL_SCORE_B]:.2f})</span></span></button>')
            if n > CROP_LIST_PAGE:
                items.append(f'<div class="muted small"><i>...and {n - CROP_LIST_PAGE:,} more -- open them all below.</i></div>')
        return (f'<div class="cmp-panel cmp-list" data-key="cmp-{h.e(outcome)}">'
                f'<div class="cmp-ptitle" style="color:{color}">{_icon(OUTCOME_ICONS[outcome], 16)}'
                f'<span>{h.e(OUTCOME_LABELS[outcome])} ({n:,})</span></div>'
                f'<div class="muted small"><i>Most confident first.</i></div>'
                f'<div class="cmp-crops">{"".join(items)}</div>'
                f'<div>{_icon_button(f"Open all {n:,} in the carousel", "burst", self.a("open_outcome", outcome=outcome), kind="primary", disabled=not n, small=True)}</div>'
                f'</div>')

    def _per_class_section(self, diff) -> str:
        table = self._per_class_table
        if table is None:
            return ""
        n_all = len(table)
        if self.class_focus.many:
            table = table[table["label"].isin(set(self._focus))]
        if table.empty:
            return ""
        max_abs = int(max(abs(table["net"].min()), abs(table["net"].max()), 1))
        rows = []
        for r in table.sort_values("net").to_dict("records"):
            rows.append(
                f'<tr><td>{h.e(r["label"])}</td><td class="num muted">{int(r["support"]):,}</td>'
                f'<td class="num" style="color:{FIX_COLOR}">{int(r["fixes"]):,}</td>'
                f'<td class="num" style="color:{BROKEN_COLOR}">{int(r["regressions"]):,}</td>'
                f'<td class="num"><span class="cmp-chip" style="background:{diverging_color(int(r["net"]), max_abs)}">'
                f'{int(r["net"]):+,}</span></td></tr>')
        sub = ("Worst first. A flat net number hides a model that traded one class for another."
               + (f" {len(table)} of {n_all:,} classes, picked under 'Classes shown' above."
                  if self.class_focus.many else ""))
        body = (f'<div class="muted small cmp-gap-b"><i>{h.e(sub)}</i></div>'
                f'<div class="tbl-wrap cmp-pc"><table class="tbl compact"><thead><tr><th>Class</th>'
                f'<th class="num">Support</th><th class="num">Fixed</th><th class="num">Broken</th>'
                f'<th class="num">Net</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
        return h.section("Per class", body, manual="per-class-change")

    # ---- carousel ---------------------------------------------------------------------

    def on_open_outcome(self, outcome: str = "", uuid: str = "", **_):
        rows = self._outcome_rows.get(outcome)
        if rows is None:
            return
        self._open_carousel(rows, OUTCOME_LABELS[outcome], uuid or None)

    def on_open_movement(self, direction: str = "", **_):
        if self.selected_cell is None or self._joined is None or direction not in cmp.MOVEMENTS:
            return
        true_label, predicted_label = self.selected_cell
        rows = cmp.rows_in_movement_cell(self._joined, direction, true_label, predicted_label)
        self._open_carousel(rows, f"{MATRIX_LABELS[direction]} — true {true_label} / "
                                  f"predicted {predicted_label}")

    def on_open_flow(self, f=None, t=None, **_):
        if self.selected_cell is None or self._joined is None:
            return
        from_label, to_label = self._label(f), self._label(t)
        if from_label is None or to_label is None:
            return
        true_label = self.selected_cell[0]
        rows = cmp.rows_in_flow_cell(self._joined, true_label, from_label, to_label)
        self._open_carousel(rows, f"true {true_label} — {self._diff.baseline}: {from_label} → "
                                  f"{self._diff.candidate}: {to_label}")

    def _open_carousel(self, rows, title: str, focus_uuid=None) -> None:
        if rows is None or not len(rows):
            return
        state = self.state
        cols = state.dataset.columns
        suggestions = {str(u): str(p) for u, p in zip(rows[cols.uuid_col], rows[cmp.COL_PRED_B])}
        dialog = CarouselDialog(
            self.s, f"{title}  ·  {len(rows):,} image(s)", state.dataset, len(rows),
            lambda start, end: rows.iloc[start:end].to_dict("records"),
            cols.crop_col, cols.uuid_col, highlight_uuid=focus_uuid, review_source="compare",
            suggestions=suggestions, label_col=cmp.COL_TRUE)
        self.s.open_dialog(dialog)

    # ---- the rest of the app talking to this view -------------------------------------

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if self.baseline_value == old_name:
            self.baseline_value = new_name
        if self.candidate_value == old_name:
            self.candidate_value = new_name
        if self._compared_pair and old_name in self._compared_pair:
            self._compared_pair = tuple(new_name if n == old_name else n for n in self._compared_pair)
            if self._diff is not None:
                if self._diff.baseline == old_name:
                    self._diff.baseline = new_name
                if self._diff.candidate == old_name:
                    self._diff.candidate = new_name
        self._sync_pair()
        self.refresh()

    # ---- snapshots (see ui/view_state.py) ---------------------------------------------

    def view_state(self) -> dict:
        return {"baseline": self.baseline_value or "", "candidate": self.candidate_value or "",
                "matrix": self.matrix_mode,
                "cell_true": self.selected_cell[0] if self.selected_cell else "",
                "cell_predicted": self.selected_cell[1] if self.selected_cell else "",
                "flow": self.flow_mode,
                "focus": self.class_focus.view_state()}

    def restore_view_state(self, state: dict) -> list:
        """Put the pair back, but do not compare (recompute_after_restore
        does, when the registry asks)."""
        warnings = []
        loaded = self.state.experiments
        for key, attr in (("baseline", "baseline_value"), ("candidate", "candidate_value")):
            name = state.get(key) or ""
            if not name:
                continue
            if name in loaded:
                setattr(self, attr, name)
            else:
                warnings.append(f"Compare: experiment '{name}' is not loaded.")
        matrix = state.get("matrix")
        if matrix in cmp.MATRIX_VIEWS:
            self.matrix_mode = matrix
        cell = (state.get("cell_true") or "", state.get("cell_predicted") or "")
        self.selected_cell = cell if all(cell) else None
        flow = state.get("flow")
        if flow in cmp.FLOW_MODES:
            self.flow_mode = flow
        if self._diff is not None:
            self._check_selected_cell()
            self._flow_data()
        self.class_focus.restore_view_state(state.get("focus"))
        if self._diff is not None:
            self._resolve_focus()
        self._sync_pair()
        self.refresh()
        return warnings
