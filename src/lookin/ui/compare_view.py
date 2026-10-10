"""Compare tab: two experiments, head to head, crop by crop.

Deliberately not part of the Metrics tab. Metrics overlays *N* experiments
and answers "how good is each one"; a diff is intrinsically *A against B*
and has a direction -- what the candidate broke is not the same list as
what it fixed. Two selection models in one tab would be confusing, so this
gets its own, with a baseline picker and a candidate picker.

Three things, in the order the question is actually asked:

1. the score-line -- how many crops changed verdict, in each direction,
   and how little of that the accuracy delta shows;
2. the confusion matrix -- *where* the change landed, as the net
   difference or as the gross arrivals and departures it nets;
3. next to it, the provenance panel -- for the cell picked in that matrix,
   where its crops came from and where the ones it lost went;
4. the crop lists -- the regressions and the fixes themselves, most
   confident first, opening in the shared image carousel.

Like the Metrics tab, the (potentially multi-second) computation happens
only on an explicit click and runs off the UI thread -- see
components.BackgroundRebuild.
"""
from __future__ import annotations

import traceback

import flet as ft

from lookin.core import class_focus as cf
from lookin.core import comparison as cmp
from lookin.ui.class_matrix import ClassFocus, short_column_labels
from lookin.ui.carousel import Carousel
from lookin.ui.components import (
    info_button,
    ACCENT, BackgroundRebuild, StaleBanner, card, diverging_color, grid_card_width,
    info_banner, matrix_cell_size, matrix_label_chars, matrix_text_size, section_title,
    status_row,
)

CELL_SIZE = 38
FLOW_CELL_SIZE = 34      # the provenance matrix sits beside the main one, so its cells are tighter
LABEL_COL_WIDTH = 96
FLOW_LABEL_COL_WIDTH = 84
FLOW_COLUMN_WIDTH = 88   # "came from" / "went to" in the panel's focused reading
PLACEHOLDER_PANEL_WIDTH = 300  # the provenance panel before a cell is picked
CROP_LIST_PAGE = 24  # rows shown per outcome list before "open all in the carousel"

# Display labels for core.comparison's outcome keys. The keys themselves are
# compared in code and must not be reworded (see CLAUDE.md).
# Which classes the matrix and the per-class table show past FOCUS_MAX
# (see ClassFocus).
COMPARE_FOCUS_CRITERIA = (
    (cf.FOCUS_MOST_CHANGED, "Most changed cells"),
    (cf.FOCUS_LARGEST_SUPPORT, "Largest support"),
    (cf.FOCUS_MANUAL, "Hand-picked"),
)
# Rows of the most-changed-cells table.
CHANGED_CELLS_SHOWN = 15
CHANGED_TABLE_WIDTH = 520

OUTCOME_LABELS = {
    cmp.OUTCOME_REGRESSION: "Broken by the candidate",
    cmp.OUTCOME_FIX: "Fixed by the candidate",
    cmp.OUTCOME_BOTH_RIGHT: "Right in both",
    cmp.OUTCOME_BOTH_WRONG: "Wrong in both",
}
OUTCOME_COLORS = {
    cmp.OUTCOME_REGRESSION: ft.Colors.RED_400,
    cmp.OUTCOME_FIX: ft.Colors.GREEN_400,
    cmp.OUTCOME_BOTH_RIGHT: ft.Colors.ON_SURFACE_VARIANT,
    cmp.OUTCOME_BOTH_WRONG: ft.Colors.ORANGE_400,
}
# Display labels for the three confusion matrices, keyed by
# core.comparison's MATRIX_VIEWS keys (compared in code, see CLAUDE.md).
MATRIX_LABELS = {
    cmp.MATRIX_NET: "Net change",
    cmp.MATRIX_MOVED_IN: "Moved in",
    cmp.MATRIX_MOVED_OUT: "Moved out",
}
# Display labels for the provenance panel's two readings, keyed by
# core.comparison's FLOW_MODES keys (compared in code, see CLAUDE.md).
FLOW_MODE_LABELS = {
    cmp.FLOW_CELL: "This cell only",
    cmp.FLOW_CLASS: "The whole class",
}
OUTCOME_ICONS = {
    cmp.OUTCOME_REGRESSION: ft.Icons.TRENDING_DOWN,
    cmp.OUTCOME_FIX: ft.Icons.TRENDING_UP,
    cmp.OUTCOME_BOTH_RIGHT: ft.Icons.CHECK_CIRCLE_OUTLINE,
    cmp.OUTCOME_BOTH_WRONG: ft.Icons.REMOVE_CIRCLE_OUTLINE,
}


def _signed(value: int) -> str:
    """A count written as what it does to a cell: an arrival adds, a
    departure subtracts. Zero gets no sign, since nothing moved and "+0"
    would claim a direction the number does not have."""
    return "0" if not value else f"{value:+,d}"


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class CompareView:
    def __init__(self, ctx):
        self.ctx = ctx
        self._dirty = True
        self._joined = None            # the full per-crop comparison frame
        self._diff = None              # the ExperimentDiff summary
        self._compared_pair = None     # (baseline, candidate) the above belong to

        # Both pickers are *rebuilt*, not mutated, whenever the loaded
        # experiments change -- see _rebuild_dropdowns. The chosen names
        # live here rather than being read back off the controls, since the
        # controls are replaced underneath them.
        self.baseline_value: str | None = None
        self.candidate_value: str | None = None
        self.baseline_dropdown = self._make_dropdown("Baseline (A)", self._on_baseline_change)
        self.candidate_dropdown = self._make_dropdown("Candidate (B)", self._on_candidate_change)
        self.picker_container = ft.Container()
        self.swap_btn = ft.IconButton(
            ft.Icons.SWAP_HORIZ, tooltip="Swap baseline and candidate", on_click=self._on_swap)
        self.compare_btn = ft.FilledButton(
            "Compare", icon=ft.Icons.COMPARE_ARROWS, on_click=self._on_compare_click,
            style=ft.ButtonStyle(bgcolor=ACCENT))

        self.progress_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_banner = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2), self.progress_text],
            spacing=10, visible=False)
        self.stale_banner = StaleBanner(
            self._on_compare_click, "Data has changed -- this comparison is out of date.")
        self.rebuild_job = BackgroundRebuild(ctx, "Compare")

        self.summary_area = ft.Column(spacing=12)
        self.matrix_area = ft.Column(spacing=12)
        # Which of the three matrices is on screen. One at a time rather
        # than three side by side: past a handful of classes a single grid
        # already wants the whole tab (see components.grid_card_width), and
        # three of them stacked would push the crop lists off the screen.
        self.matrix_mode = cmp.MATRIX_NET
        # The cell the provenance panel describes, as (true, predicted), or
        # None for "nothing picked yet". Clicking a cell selects it rather
        # than opening the carousel: a cell of any of these matrices is a
        # set of *movements*, and which of them to look at is the question
        # the panel exists to answer.
        self.selected_cell = None
        # Which reading the provenance panel gives of that cell: only the
        # crops that touch it, or the whole class it sits in.
        self.flow_mode = cmp.FLOW_CELL
        # Both grid cards are sized from the window, so on_window_resized
        # has to know how wide the grid inside each one wants to be.
        self._matrix_card = None
        self._matrix_grid_width = 0
        self._flow_card = None
        self._flow_grid_width = 0
        self.lists_area = ft.Column(spacing=12)
        # Past FOCUS_MAX classes the matrix and the per-class table show a
        # focus set -- one for both, so they talk about the same classes.
        self.class_focus = ClassFocus(COMPARE_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the matrix and the per-class table")
        self._focus: list = []
        self._per_class_table = None  # cmp.per_class_change for the current diff

        self.carousel_title = ft.Text("", weight=ft.FontWeight.BOLD)
        self.carousel_content = ft.Container()
        self.carousel_dialog = ft.AlertDialog(
            # modal=False so tapping outside dismisses; on_dismiss then has
            # to sync `open` back to False, which Flet does not do itself --
            # a stale True makes the next page.open() a no-op, and the dialog
            # simply stops opening with nothing to show why.
            modal=False, title=self.carousel_title, content=self.carousel_content,
            on_dismiss=lambda e: self.ctx.page.close(self.carousel_dialog),
            actions=[ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.carousel_dialog))],
        )
        self.root = ft.Container(padding=20)

    # ---- progress -------------------------------------------------------
    def _set_progress(self, text: str):
        self.progress_banner.visible = True
        self.progress_text.value = text
        if self.progress_banner.page:
            self.progress_banner.update()

    def _clear_progress(self):
        if self.progress_banner.visible:
            self.progress_banner.visible = False
            if self.progress_banner.page:
                self.progress_banner.update()

    # ---- experiment pickers --------------------------------------------
    def _make_dropdown(self, label: str, on_change, names=(), value=None) -> ft.Dropdown:
        return ft.Dropdown(
            label=label, width=280, border_radius=8, on_change=on_change,
            options=[ft.dropdown.Option(n, n) for n in names], value=value,
        )

    def _rebuild_dropdowns(self):
        """Rebuild both pickers from what is currently loaded, preserving
        the user's choice when it survives and defaulting to the first two
        experiments so the common case is one click.

        The Dropdowns are recreated rather than having their `options`
        reassigned. Reassigning the option list of an already-mounted
        Dropdown does not reliably reach the browser, and doing it during
        ctx.notify_change() -- which ends with a full page.update() --
        makes Flet diff options it never registered and raise, an error
        that lands in the *loader's* try/except and reads as "failed to
        load experiment". Handing Flet a brand-new subtree is the operation
        it handles cleanly, and it is what the Metrics tab already does for
        its own experiment checkboxes.
        """
        names = list(self.ctx.state.experiments)
        for attr, default_index, label, handler in (
            ("baseline", 0, "Baseline (A)", self._on_baseline_change),
            ("candidate", 1, "Candidate (B)", self._on_candidate_change),
        ):
            value = getattr(self, f"{attr}_value")
            if value not in names:
                value = names[default_index] if len(names) > default_index else None
            setattr(self, f"{attr}_value", value)
            setattr(self, f"{attr}_dropdown", self._make_dropdown(label, handler, names, value))

        self.compare_btn.disabled = not self._pair_is_valid()
        self.picker_container.content = ft.Row([
            self.baseline_dropdown, self.swap_btn, self.candidate_dropdown,
            ft.Container(width=10), self.compare_btn,
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _pair_is_valid(self) -> bool:
        a, b = self.baseline_value, self.candidate_value
        return bool(a and b and a in self.ctx.state.experiments and b in self.ctx.state.experiments)

    def _on_baseline_change(self, e):
        self.baseline_value = self.baseline_dropdown.value
        self._on_pair_change(e)

    def _on_candidate_change(self, e):
        self.candidate_value = self.candidate_dropdown.value
        self._on_pair_change(e)

    def on_shown(self):
        """Called when this tab is selected (see app.py).

        Populating the two pickers has to happen *here* rather than in
        refresh(), because refresh() runs inside ctx.notify_change(), which
        itself fires in the middle of a load (see
        UploadView._auto_attach_sidecars). Replacing a Dropdown's options
        and pushing that to the browser at that moment makes Flet emit a
        remove command for options it has not registered yet, and the
        AssertionError that comes back out of flet_web surfaces as
        "failed to load experiment" with nothing else to go on.

        So refresh() only ever changes Python-side state, and the push
        happens when the user actually looks at the tab. Same rule the
        Metrics tab follows for its own experiment picker, which rebuilds
        on the Update click rather than on every data change.
        """
        self._rebuild_dropdowns()
        if self._diff is None or self._dirty:
            self._render_placeholder()
        if self.root.page:
            self.root.update()

    def _push_picker_state(self):
        """Push just the picker row -- safe from a user interaction on an
        already-mounted control, which is the only place this is called."""
        if self.picker_container.page:
            self.picker_container.update()

    def _on_pair_change(self, e):
        self.compare_btn.disabled = not self._pair_is_valid()
        # The results on screen belong to the previous pair; say so instead
        # of leaving them looking like an answer to the new question.
        if self._compared_pair and self._compared_pair != (
                self.baseline_value, self.candidate_value):
            self.stale_banner.text.value = "Pair changed -- click Compare to update."
            self.stale_banner.show()
        self._push_picker_state()

    def _on_swap(self, e):
        self.baseline_value, self.candidate_value = self.candidate_value, self.baseline_value
        self._rebuild_dropdowns()
        self._on_pair_change(e)

    # ---- lifecycle ------------------------------------------------------
    def refresh(self):
        """Data changed. Same rule as Metrics: flag, never recompute --
        joining two merged frames is real work at scale."""
        self._dirty = True
        # Rebuilding the pickers means handing the container a brand-new
        # subtree, which Flet adds cleanly on the page.update() that
        # ctx.notify_change() ends with. What must never happen here is
        # reassigning a mounted Dropdown's `options`: Flet then emits
        # remove commands for options it never registered, and the
        # AssertionError comes back out inside the *loader's* try/except,
        # reading as "failed to load experiment" with nothing to go on.
        self._rebuild_dropdowns()
        if self.root.page:
            self.stale_banner.text.value = "Data has changed -- this comparison is out of date."
            self.stale_banner.show()

    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        """The chosen pair, read off the values rather than the dropdowns:
        the controls are rebuilt whenever the loaded experiments change,
        which is exactly why baseline_value/candidate_value exist."""
        return {"baseline": self.baseline_value or "", "candidate": self.candidate_value or "",
                "matrix": self.matrix_mode,
                # Two labels rather than a composite key: a class label
                # comes from the user's CSV and could contain any separator
                # baked into one (same reason as the case-study dropdown in
                # ui/embeddings_view.py).
                "cell_true": self.selected_cell[0] if self.selected_cell else "",
                "cell_predicted": self.selected_cell[1] if self.selected_cell else "",
                "flow": self.flow_mode,
                "focus": self.class_focus.view_state()}

    def restore_view_state(self, state: dict) -> list:
        """Put the pair back, but do not compare. A comparison is a full
        pass over every crop of both experiments; the user clicks for it
        here as everywhere else."""
        warnings = []
        loaded = self.ctx.state.experiments
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
        # Kept as the user left it, but never trusted: the comparison is
        # not recomputed here, and _render_matrix drops a cell whose
        # classes this pair does not have.
        cell = (state.get("cell_true") or "", state.get("cell_predicted") or "")
        self.selected_cell = cell if all(cell) else None
        flow = state.get("flow")
        if flow in cmp.FLOW_MODES:
            self.flow_mode = flow
        self.class_focus.restore_view_state(state.get("focus"))
        self._rebuild_dropdowns()
        if self.root.page:
            self.root.update()
        return warnings

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow a rename without recomputing: the comparison already on
        screen is still valid, only its name changed."""
        if self.baseline_value == old_name:
            self.baseline_value = new_name
        if self.candidate_value == old_name:
            self.candidate_value = new_name
        if self._compared_pair and old_name in self._compared_pair:
            self._compared_pair = tuple(
                new_name if n == old_name else n for n in self._compared_pair)
            if self._diff is not None:
                if self._diff.baseline == old_name:
                    self._diff.baseline = new_name
                if self._diff.candidate == old_name:
                    self._diff.candidate = new_name
        self._rebuild_dropdowns()
        if not self._dirty and self._diff is not None:
            self._render_results()
        if self.root.page:
            self.root.update()

    def recompute_after_restore(self) -> None:
        """Re-run the head-to-head. Only when both experiments are still
        loaded -- _pair_is_valid is what restore_view_state may have left
        half-filled."""
        if self._pair_is_valid():
            self._on_compare_click(None)

    def _on_compare_click(self, e):
        if not self._pair_is_valid():
            return
        self._set_progress("Comparing...")
        self.rebuild_job.run(self._compare_now, on_error=self._on_error)

    def _on_error(self, ex: Exception):
        self._clear_progress()
        self.summary_area.controls = [
            info_banner(f"Error comparing experiments: {ex}", icon=ft.Icons.ERROR_OUTLINE)]
        self.matrix_area.controls = []
        self._matrix_card = self._flow_card = None
        self.lists_area.controls = []
        if self.root.page:
            self.root.update()

    def _compare_now(self):
        baseline, candidate = self.baseline_value, self.candidate_value
        try:
            self._set_progress(f"Joining {baseline} and {candidate}...")
            joined = cmp.build_comparison(self.ctx.state, baseline, candidate)
            self._set_progress("Summarising the differences...")
            diff = cmp.compare_experiments(self.ctx.state, baseline, candidate, joined=joined)
            self._joined, self._diff = joined, diff
            self._compared_pair = (baseline, candidate)
            _log(f"compare: {baseline} vs {candidate} -- {diff.n_common:,} common crops, "
                 f"{diff.counts[cmp.OUTCOME_FIX]:,} fixed, {diff.counts[cmp.OUTCOME_REGRESSION]:,} broken")
            self._set_progress("Drawing...")
            self._render_results()
        except Exception as ex:
            _log(f"compare failed: {ex}")
            traceback.print_exc()
            self._on_error(ex)
            return
        self._dirty = False
        self._clear_progress()
        self.stale_banner.hide()
        if self.root.page:
            self.root.update()

    # ---- rendering ------------------------------------------------------
    def _render_placeholder(self):
        state = self.ctx.state
        if state.dataset is None or len(state.experiments) < 2:
            message = ("Load a dataset and at least two experiments in the 'Data' tab -- "
                       "this tab compares one against another.")
        else:
            message = "Pick a baseline and a candidate above, then click Compare."
        self.summary_area.controls = [info_banner(message)]
        self.class_focus.hide()
        self._per_class_table = None
        self.matrix_area.controls = []
        self._matrix_card = self._flow_card = None
        self.lists_area.controls = []

    def _render_results(self):
        diff = self._diff
        if diff is None:
            self._render_placeholder()
            return
        self._render_summary(diff)
        self._per_class_table = cmp.per_class_change(self._joined, diff.labels)
        self._resolve_focus(diff)
        self._render_matrix(diff)
        self._render_lists(diff)

    # ---- focus set: which classes the matrix and table show (see core/class_focus.py) ----
    def _resolve_focus(self, diff) -> None:
        moved = diff.moved_in_confusion + diff.moved_out_confusion
        table = self._per_class_table

        def rank(criterion):
            if criterion == cf.FOCUS_MOST_CHANGED:
                cells = cf.top_confused_pairs(moved, include_diagonal=True)
                return cf.pairs_to_focus(cells[["true", "predicted"]].itertuples(index=False))
            if criterion == cf.FOCUS_LARGEST_SUPPORT and table is not None and len(table):
                return cf.rank_by(table, "support", ascending=False)
            return []

        self._focus = self.class_focus.resolve(list(diff.delta_confusion.index), rank)

    def _on_focus_change(self):
        if self._diff is None:
            return
        self._resolve_focus(self._diff)
        self._render_matrix(self._diff)
        self._render_lists(self._diff)
        for control in (self.class_focus.build(), self.matrix_area, self.lists_area):
            if control.page:
                control.update()

    def _render_summary(self, diff):
        counts = diff.counts
        changed = counts[cmp.OUTCOME_REGRESSION] + counts[cmp.OUTCOME_FIX]

        tiles = ft.Row([
            self._score_tile(outcome, counts[outcome], diff.n_common)
            for outcome in (cmp.OUTCOME_REGRESSION, cmp.OUTCOME_FIX,
                            cmp.OUTCOME_BOTH_RIGHT, cmp.OUTCOME_BOTH_WRONG)
        ], spacing=12, wrap=True, run_spacing=12)

        headline = (
            f"{changed:,} crop(s) changed verdict "
            f"({counts[cmp.OUTCOME_FIX]:,} fixed, {counts[cmp.OUTCOME_REGRESSION]:,} broken) "
            f"— net {diff.net_change():+,}"
        )
        accuracy_line = (
            f"Accuracy {diff.baseline_accuracy:.4f} → {diff.candidate_accuracy:.4f} "
            f"({diff.accuracy_delta:+.4f})"
        )

        notes = []
        if diff.n_baseline_only or diff.n_candidate_only:
            notes.append(status_row(
                f"Compared on the {diff.n_common:,} crop(s) both cover. "
                f"{diff.n_baseline_only:,} only in '{diff.baseline}', "
                f"{diff.n_candidate_only:,} only in '{diff.candidate}' -- those are excluded, "
                f"since counts over different row sets cannot be compared.",
                "warning"))
        if self.ctx.state.use_reviewed_labels:
            notes.append(status_row(
                "Both models are judged against the reviewed labels (the overlay is on).", "neutral"))

        self.summary_area.controls = [card(ft.Column([
            ft.Row([
                ft.Text(diff.baseline, weight=ft.FontWeight.BOLD, size=15),
                ft.Icon(ft.Icons.ARROW_FORWARD, size=16, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(diff.candidate, weight=ft.FontWeight.BOLD, size=15),
            ], spacing=10),
            ft.Text(headline, size=14),
            ft.Text(accuracy_line, size=13, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text(
                "The accuracy delta only shows the net; these four numbers say what produced it.",
                size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
            tiles,
            *notes,
        ], spacing=10))]

    def _score_tile(self, outcome: str, count: int, total: int) -> ft.Control:
        share = f"{100 * count / total:.1f}%" if total else "--"
        color = OUTCOME_COLORS[outcome]
        return ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(OUTCOME_ICONS[outcome], size=15, color=color),
                        ft.Text(OUTCOME_LABELS[outcome], size=11, color=color)], spacing=6),
                ft.Text(f"{count:,}", size=22, weight=ft.FontWeight.BOLD),
                ft.Text(share, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=2, tight=True),
            padding=12, border_radius=10, width=190,
            border=ft.border.all(1, color),
            ink=True, disabled=not count,
            tooltip="Open these crops in the carousel" if count else None,
            on_click=(lambda e, outcome=outcome: self._open_outcome(outcome)) if count else None,
        )

    def _matrix_improvement(self, true_label, predicted_label, value: int) -> int:
        """Signed "is this cell good news", which is what the colour shows.

        On the difference matrix the sign of the count is not the sign of
        the news, so core.comparison decides (more on the diagonal is
        better, more anywhere else is worse). The two gross matrices are
        plain counts, but they still have a direction: a crop arriving on
        the diagonal was fixed and one arriving anywhere else was broken,
        and a departure is the same statement backwards -- which is exactly
        cell_improvement applied to a movement of the right sign.
        """
        if self.matrix_mode == cmp.MATRIX_MOVED_OUT:
            value = -value
        return cmp.cell_improvement(true_label, predicted_label, value)

    def _matrix_cell_text(self, value: int) -> str:
        if not value:
            return "\u00b7"
        # Signed in all three matrices, because in all three the number says
        # what happened to this cell's count: an arrival adds to it, a
        # departure takes from it, and the difference is the two together.
        # A bare "12" is the one reading that does not say which.
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
        return (f"{head}{diff.candidate} - {diff.baseline} = {value:+d}\n({diagonal_note})")

    def _make_matrix_switch(self, diff) -> ft.Control:
        """Rebuilt on every render rather than kept and mutated -- the card
        around it is rebuilt too, and handing Flet a brand-new subtree is
        the operation it handles cleanly (same reason as _rebuild_dropdowns).
        `matrix_mode` is the source of truth, not the control."""
        # Every crop that changed prediction is one arrival and one
        # departure, so the two gross totals are the same number. Showing
        # it twice is not a bug, it is the conservation the net matrix
        # hides -- and both are usually far larger than the net.
        totals = {
            cmp.MATRIX_NET: diff.net_change(),
            cmp.MATRIX_MOVED_IN: diff.n_moved,
            cmp.MATRIX_MOVED_OUT: diff.n_moved,
        }
        return ft.SegmentedButton(
            show_selected_icon=False, allow_empty_selection=False,
            selected={self.matrix_mode},
            segments=[
                ft.Segment(
                    value=key,
                    label=ft.Text(f"{MATRIX_LABELS[key]} ({totals[key]:+,})" if key == cmp.MATRIX_NET
                                  else f"{MATRIX_LABELS[key]} ({totals[key]:,})", size=12))
                for key in cmp.MATRIX_VIEWS
            ],
            on_change=self._on_matrix_mode_change,
        )

    def _on_matrix_mode_change(self, e):
        """Only the matrices are redrawn -- the comparison itself is
        unchanged, and all three came out of the same pass. The selected
        cell survives the switch: the same cell in the other matrix is the
        other half of the same question."""
        selected = next(iter(e.control.selected), None)
        if selected not in cmp.MATRIX_VIEWS or selected == self.matrix_mode:
            return
        self.matrix_mode = selected
        if self._diff is None:
            return
        self._render_matrix(self._diff)
        if self.matrix_area.page:
            self.matrix_area.update()

    def _render_matrix(self, diff):
        matrix = diff.confusion(self.matrix_mode)
        if matrix.empty:
            self.matrix_area.controls = []
            self._matrix_card = self._flow_card = None
            return
        all_labels = list(matrix.index)
        # A selection made against another pair of experiments can name a
        # class this one never predicts; dropping it beats indexing the
        # flow matrix with a label it does not have.
        if self.selected_cell and not set(self.selected_cell) <= set(all_labels):
            self.selected_cell = None
        many = self.class_focus.many
        # Past FOCUS_MAX classes: the focus rows by the focus columns, plus
        # an "other" column per row for the cells outside the set.
        labels = [l for l in self._focus if l in set(all_labels)] if many else all_labels
        values = matrix.loc[labels, labels].to_numpy()
        other = (matrix.loc[labels].to_numpy().sum(axis=1) - values.sum(axis=1)) if many else None
        max_abs = int(max(abs(values.min()), abs(values.max()), 1)) if values.size else 1

        # Few classes give big cells rather than a small grid, same as the
        # Metrics tab's matrix (see components.matrix_cell_size).
        cell = matrix_cell_size(len(labels) + (1 if many else 0), CELL_SIZE)
        value_size = matrix_text_size(cell, CELL_SIZE, 10)
        header_size = matrix_text_size(cell, CELL_SIZE, 9, cap=14)
        header_chars = matrix_label_chars(cell, CELL_SIZE, 5)
        row_label_size = matrix_text_size(cell, CELL_SIZE, 10, cap=14)

        # Two lines of header text, without the prefix every class shares
        # (see short_column_labels) -- same as the Metrics tab's grid.
        short_labels, dropped_prefix = short_column_labels(labels, 2 * header_chars)
        header_cells = [ft.Container(width=LABEL_COL_WIDTH)] + [
            ft.Container(content=ft.Text(short, size=header_size, text_align=ft.TextAlign.CENTER, max_lines=2),
                         width=cell, height=2.6 * header_size + 4,
                         alignment=ft.alignment.bottom_center, tooltip=pl)
            for short, pl in zip(short_labels, labels)
        ]
        if many:
            header_cells.append(ft.Container(
                content=ft.Text("other", size=header_size, italic=True), width=cell,
                alignment=ft.alignment.bottom_center, tooltip="Any class not shown in this grid"))
        header = ft.Row(header_cells, spacing=2)
        rows = [header]
        for i, tl in enumerate(labels):
            controls = [ft.Container(
                content=ft.Text(tl, size=row_label_size, weight=ft.FontWeight.BOLD),
                width=LABEL_COL_WIDTH, alignment=ft.alignment.center_right,
                padding=ft.padding.only(right=8))]
            for j, pl in enumerate(labels):
                value = int(values[i, j])
                improvement = self._matrix_improvement(tl, pl, value)
                picked = self.selected_cell == (tl, pl)
                controls.append(ft.Container(
                    width=cell, height=cell, border_radius=4,
                    bgcolor=diverging_color(improvement, max_abs),
                    alignment=ft.alignment.center, ink=True,
                    border=ft.border.all(2, ACCENT) if picked else None,
                    content=ft.Text(self._matrix_cell_text(value), size=value_size,
                                    color="#ffffff" if abs(improvement) > max_abs * 0.4 else "#c9c9d1"),
                    tooltip=self._matrix_cell_tooltip(diff, tl, pl, value),
                    on_click=lambda e, tl=tl, pl=pl: self._on_cell_click(tl, pl),
                ))
            if many:
                value = int(other[i])
                controls.append(ft.Container(
                    width=cell, height=cell, border_radius=4, bgcolor=ft.Colors.SURFACE,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT), alignment=ft.alignment.center,
                    content=ft.Text(self._matrix_cell_text(value), size=value_size, italic=True),
                    tooltip=f"true = {tl}\npredicted = any class not in this grid\n{value:+,d} in total",
                ))
            rows.append(ft.Row(controls, spacing=2))

        # As wide as the grid, up to the whole tab: a fixed cap turned every
        # matrix past ~16 classes into a narrow window scrolling over its own
        # content, row labels included (see components.grid_card_width).
        grid_width = LABEL_COL_WIDTH + (len(labels) + (1 if many else 0)) * (cell + 2)
        self._matrix_grid_width = max(grid_width, CHANGED_TABLE_WIDTH)
        self._matrix_card = card(
            ft.Column([
                self._make_matrix_switch(diff),
                ft.Text(self._matrix_caption(diff), size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Row([ft.Column(rows, spacing=2)], scroll=ft.ScrollMode.AUTO),
                ft.Row([ft.Icon(ft.Icons.TOUCH_APP, size=13, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Text("Click a cell to see which cells its crops came from and went to"
                                + (f" -- column headers leave out the shared prefix '{dropped_prefix}'"
                                   if dropped_prefix else ""),
                                size=10, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)],
                       spacing=6),
                self._changed_cells_table(diff),
            ], spacing=10), width=grid_card_width(self.ctx.page, self._matrix_grid_width))
        self._flow_card = self._build_flow_card(diff)
        self.matrix_area.controls = [
            section_title("Where the change landed", icon=ft.Icons.GRID_ON, ctx=self.ctx, manual="delta-confusion"),
            # A wrapping row rather than two columns: both cards ask for the
            # width their own grid needs, and past a dozen classes the main
            # matrix alone wants the whole tab. Wrapping puts the provenance
            # panel underneath instead of squeezing two grids into half a
            # window each.
            ft.Row([self._matrix_card, self._flow_card], spacing=12, wrap=True,
                   run_spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
        ]

    def _changed_cells_table(self, diff) -> ft.Control:
        """The cells of the whole matrix where the most crops moved -- in
        or out, diagonal included, since a crop arriving on the diagonal
        is a fix. Clicking a row picks that cell, as clicking it in the
        grid does."""
        moved = diff.moved_in_confusion + diff.moved_out_confusion
        cells = cf.top_confused_pairs(moved, include_diagonal=True)
        if not len(cells):
            return ft.Text("No crop changed prediction.", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        shown = cells.head(CHANGED_CELLS_SHOWN)
        rows = []
        for true_label, predicted_label, _traffic, _share in shown.itertuples(index=False):
            moved_in = int(diff.moved_in_confusion.loc[true_label, predicted_label])
            moved_out = int(diff.moved_out_confusion.loc[true_label, predicted_label])
            net = moved_in - moved_out
            rows.append(ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(str(true_label), size=12)),
                    ft.DataCell(ft.Text(str(predicted_label), size=12)),
                    ft.DataCell(ft.Text(f"{moved_in:+,d}", size=12)),
                    ft.DataCell(ft.Text(f"{-moved_out:+,d}", size=12)),
                    ft.DataCell(ft.Container(
                        ft.Text(f"{net:+,d}", size=12, weight=ft.FontWeight.BOLD),
                        bgcolor=diverging_color(cmp.cell_improvement(true_label, predicted_label, net),
                                                max(abs(net), 1)),
                        border_radius=4, padding=ft.padding.symmetric(horizontal=6, vertical=2))),
                ],
                on_select_changed=lambda e, t=true_label, p=predicted_label: self._on_cell_click(t, p),
            ))
        table = ft.DataTable(
            columns=[ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12))
                     for h in ("True class", "Predicted as", "Moved in", "Moved out", "Net")],
            rows=rows, heading_row_height=36, data_row_min_height=32, data_row_max_height=40,
            column_spacing=20, show_checkbox_column=False,
        )
        return ft.Column([
            ft.Text(f"Most changed cells -- {len(shown)} of {len(cells):,} cells where a crop moved",
                    size=12, weight=ft.FontWeight.BOLD),
            ft.Row([table], scroll=ft.ScrollMode.AUTO),
        ], spacing=6)

    # ---- provenance panel ----------------------------------------------
    def _build_flow_card(self, diff) -> ft.Control:
        """Where the crops of the selected cell came from, and where the
        ones that left it went.

        The true class is the same under both models, so a crop can only
        move along its own row of the matrix on the left. That is what
        makes this panel per class rather than a second true-by-predicted
        grid, and it reads that row two ways (see FLOW_MODE_LABELS): only
        the crops that touch the picked cell, which is what clicking a cell
        asks, or every crop of its class, which is the wider question of
        what happened to the class as a whole.
        """
        if self.selected_cell is None:
            self._flow_grid_width = PLACEHOLDER_PANEL_WIDTH
            return card(ft.Column([
                ft.Text("Where they came from", weight=ft.FontWeight.BOLD, size=14),
                ft.Text("Pick a cell on the left. This panel then shows which cells its crops "
                        "arrived from and which ones the crops it lost went to.",
                        size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=10), width=grid_card_width(self.ctx.page, PLACEHOLDER_PANEL_WIDTH))

        true_label, predicted_label = self.selected_cell
        moved_in = int(diff.moved_in_confusion.loc[true_label, predicted_label])
        moved_out = int(diff.moved_out_confusion.loc[true_label, predicted_label])

        if self.flow_mode == cmp.FLOW_CELL:
            grid, grid_width, caption = self._cell_flows_grid(diff, true_label, predicted_label)
        else:
            grid, grid_width, caption = self._class_flow_grid(diff, true_label, predicted_label)
        self._flow_grid_width = grid_width

        return card(ft.Column([
            # The card wraps underneath the matrix once the classes stop
            # fitting side by side, so it says what it is as well as which
            # cell it belongs to.
            ft.Row([ft.Icon(ft.Icons.CALL_SPLIT, size=17, color=ACCENT),
                    ft.Text("Where they came from", weight=ft.FontWeight.BOLD, size=14),
                    ft.Text(f"true {true_label} / predicted {predicted_label}",
                            size=13, color=ACCENT)], spacing=8, wrap=True),
            self._make_flow_switch(),
            ft.Text(f"{_signed(moved_in)} moved in, {_signed(-moved_out)} moved out "
                    f"\u2014 net {_signed(moved_in - moved_out)}", size=13),
            ft.Text(caption, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Row([ft.Column(grid, spacing=2)], scroll=ft.ScrollMode.AUTO),
            ft.Row([
                ft.FilledButton(
                    f"Open the {moved_in:,} that moved in", icon=ft.Icons.LOGIN,
                    disabled=not moved_in, style=ft.ButtonStyle(bgcolor=ACCENT),
                    on_click=lambda e: self._open_movement(cmp.MATRIX_MOVED_IN)),
                ft.FilledButton(
                    f"Open the {moved_out:,} that moved out", icon=ft.Icons.LOGOUT,
                    disabled=not moved_out, style=ft.ButtonStyle(bgcolor=ACCENT),
                    on_click=lambda e: self._open_movement(cmp.MATRIX_MOVED_OUT)),
            ], spacing=8, wrap=True, run_spacing=8),
            ft.Row([ft.Icon(ft.Icons.TOUCH_APP, size=13, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("Click a cell here to open exactly that move in the carousel",
                            size=10, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)], spacing=6),
        ], spacing=10), width=grid_card_width(self.ctx.page, grid_width))

    def _make_flow_switch(self) -> ft.Control:
        """Rebuilt on every render rather than kept and mutated, same as the
        matrix switch -- `flow_mode` is the source of truth, not the
        control."""
        return ft.SegmentedButton(
            show_selected_icon=False, allow_empty_selection=False,
            selected={self.flow_mode},
            segments=[ft.Segment(value=key, label=ft.Text(FLOW_MODE_LABELS[key], size=12))
                      for key in cmp.FLOW_MODES],
            on_change=self._on_flow_mode_change,
        )

    def _on_flow_mode_change(self, e):
        """Only the panel changes -- the comparison and the matrix on the
        left are untouched, and both readings come off the same frame."""
        selected = next(iter(e.control.selected), None)
        if selected not in cmp.FLOW_MODES or selected == self.flow_mode:
            return
        self.flow_mode = selected
        if self._diff is None:
            return
        self._render_matrix(self._diff)
        if self.matrix_area.page:
            self.matrix_area.update()

    def _cell_flows_grid(self, diff, true_label, predicted_label):
        """One line per class the picked cell traded crops with: what
        arrived from it, what left for it. Only the crops that touch the
        cell are counted -- not the rest of the class.

        Two columns rather than a square, because for one cell the square
        has nothing else in it: every arrival lands in the same column and
        every departure starts from the same row, so a grid would be two
        lines of numbers and a page of zeros between them.
        """
        table = cmp.cell_flows(self._joined, diff.labels, true_label, predicted_label)
        caption = (f"Only the crops that touch this cell, by the class they came from or went to. "
                   f"Rows are '{diff.baseline}'s class for an arrival and '{diff.candidate}'s for "
                   f"a departure; a move onto {true_label} is a fix, one away from it a breakage. "
                   "Busiest first.")
        header = ft.Row([
            ft.Container(width=FLOW_LABEL_COL_WIDTH),
            ft.Container(ft.Text("came from", size=10, weight=ft.FontWeight.BOLD),
                         width=FLOW_COLUMN_WIDTH, alignment=ft.alignment.center),
            ft.Container(ft.Text("went to", size=10, weight=ft.FontWeight.BOLD),
                         width=FLOW_COLUMN_WIDTH, alignment=ft.alignment.center),
        ], spacing=2)
        if table.empty:
            return ([header, ft.Row([ft.Text("Nothing moved through this cell.", size=12,
                                             italic=True, color=ft.Colors.ON_SURFACE_VARIANT)])],
                    FLOW_LABEL_COL_WIDTH + 2 * (FLOW_COLUMN_WIDTH + 2), caption)

        max_abs = max(int(table.to_numpy().max()), 1)
        rows = [header]
        for label, row in table.iterrows():
            came, went = int(row[cmp.COL_CAME_FROM]), int(row[cmp.COL_WENT_TO])
            rows.append(ft.Row([
                ft.Container(ft.Text(str(label), size=12, weight=ft.FontWeight.BOLD),
                             width=FLOW_LABEL_COL_WIDTH, alignment=ft.alignment.center_right,
                             padding=ft.padding.only(right=8)),
                # An arrival came from `label` and landed here, so the
                # baseline is the one that said `label`; a departure is the
                # same sentence backwards.
                self._flow_count_cell(true_label, label, predicted_label, came,
                                      arriving=True, max_abs=max_abs),
                self._flow_count_cell(true_label, predicted_label, label, went,
                                      arriving=False, max_abs=max_abs),
            ], spacing=2))
        return rows, FLOW_LABEL_COL_WIDTH + 2 * (FLOW_COLUMN_WIDTH + 2), caption

    def _flow_count_cell(self, true_label, from_label, to_label, value: int,
                         arriving: bool, max_abs: int) -> ft.Control:
        improvement = cmp.flow_improvement(true_label, from_label, to_label, value)
        # An arrival adds to the picked cell and a departure takes from it,
        # which is the sign -- the colour still follows the news, so a
        # departure off the diagonal is a negative number on green.
        text = "\u00b7" if not value else _signed(value if arriving else -value)
        return ft.Container(
            width=FLOW_COLUMN_WIDTH, height=FLOW_CELL_SIZE, border_radius=4,
            bgcolor=diverging_color(improvement, max_abs),
            alignment=ft.alignment.center, ink=bool(value),
            content=ft.Text(text, size=12,
                            color="#ffffff" if improvement else ft.Colors.ON_SURFACE_VARIANT),
            tooltip=self._flow_tooltip(true_label, from_label, to_label, value),
            on_click=(lambda e: self._open_flow_cell(from_label, to_label)) if value else None,
        )

    def _class_flow_grid(self, diff, true_label, predicted_label):
        """The whole class: what the baseline predicted (rows) against what
        the candidate predicts (columns), for every crop of it."""
        flow = cmp.flow_matrix(self._joined, diff.labels, true_label)
        labels = cmp.flow_labels(flow, true_label)
        # The selected class stays in even when the trim would drop it --
        # clicking an empty cell is how "why is nothing here" gets asked,
        # and a panel with no column for it outlines nothing and answers
        # nothing. Re-ordered through flow.index so the axes keep the
        # matrix's own class order.
        if predicted_label in flow.index and predicted_label not in labels:
            keep = set(labels) | {predicted_label}
            labels = [label for label in flow.index if label in keep]
        caption = (f"Every crop whose true class is {true_label}, not only the ones that touch "
                   f"the picked cell. Rows: what '{diff.baseline}' predicted. Columns: what "
                   f"'{diff.candidate}' predicts. The outlined column is the picked cell's "
                   "arrivals and the outlined row its departures -- those carry a sign, the "
                   "other moves do not touch that cell's count. The diagonal never moved.")
        grid, width = self._flow_grid(flow, labels, true_label, predicted_label)
        return grid, width, caption

    def _flow_grid(self, flow, labels: list, true_label, predicted_label):
        """The class-wide flow grid itself, and the width it wants."""
        values = flow.loc[labels, labels].to_numpy()
        # Scaled on the moves only: the diagonal is everything that stayed
        # put and is usually an order of magnitude larger, so letting it
        # into the scale would flatten every move to the same dull colour.
        moves = [abs(int(values[i, j])) for i in range(len(labels)) for j in range(len(labels))
                 if labels[i] != labels[j]]
        max_abs = max(moves or [0], default=0) or 1

        cell = matrix_cell_size(len(labels), FLOW_CELL_SIZE)
        value_size = matrix_text_size(cell, FLOW_CELL_SIZE, 10)
        header_size = matrix_text_size(cell, FLOW_CELL_SIZE, 9, cap=14)
        header_chars = matrix_label_chars(cell, FLOW_CELL_SIZE, 5)
        row_label_size = matrix_text_size(cell, FLOW_CELL_SIZE, 10, cap=14)

        header = ft.Row(
            [ft.Container(width=FLOW_LABEL_COL_WIDTH)] + [
                ft.Container(
                    content=ft.Text(label[:header_chars], size=header_size,
                                    color=ACCENT if label == predicted_label else None),
                    width=cell, alignment=ft.alignment.center, tooltip=label)
                for label in labels
            ], spacing=2)
        rows = [header]
        for i, from_label in enumerate(labels):
            controls = [ft.Container(
                content=ft.Text(from_label, size=row_label_size, weight=ft.FontWeight.BOLD,
                                color=ACCENT if from_label == predicted_label else None),
                width=FLOW_LABEL_COL_WIDTH, alignment=ft.alignment.center_right,
                padding=ft.padding.only(right=8))]
            for j, to_label in enumerate(labels):
                value = int(values[i, j])
                improvement = cmp.flow_improvement(true_label, from_label, to_label, value)
                moved = from_label != to_label
                # The arrivals of the selected cell are its column, the
                # departures its row -- outlined so the answer to "where
                # did these come from" is one glance rather than a hunt.
                outlined = moved and (to_label == predicted_label or from_label == predicted_label)
                controls.append(ft.Container(
                    width=cell, height=cell, border_radius=4,
                    bgcolor=diverging_color(improvement, max_abs) if moved else "#232329",
                    alignment=ft.alignment.center, ink=bool(value),
                    border=ft.border.all(1, ACCENT) if outlined else None,
                    content=ft.Text(
                        self._class_flow_text(value, from_label, to_label, predicted_label, moved),
                        size=value_size,
                        color=("#ffffff" if abs(improvement) > max_abs * 0.4 else "#c9c9d1")
                        if moved else ft.Colors.ON_SURFACE_VARIANT),
                    tooltip=self._flow_tooltip(true_label, from_label, to_label, value),
                    on_click=(lambda e, f=from_label, t=to_label: self._open_flow_cell(f, t))
                    if value else None,
                ))
            rows.append(ft.Row(controls, spacing=2))
        return rows, FLOW_LABEL_COL_WIDTH + len(labels) * (cell + 2)

    def _class_flow_text(self, value: int, from_label, to_label, predicted_label,
                         moved: bool) -> str:
        """Signed for the moves that change the picked cell's count, plain
        otherwise.

        Its column is what the cell gained and its row what it lost, so
        those get a sign like every other number in this tab. A move
        between two other classes is a real move but it leaves the picked
        cell alone, and signing it would say it did not.
        """
        if not value:
            return "\u00b7"
        if moved and to_label == predicted_label:
            return _signed(value)
        if moved and from_label == predicted_label:
            return _signed(-value)
        return f"{value:,}"

    def _flow_tooltip(self, true_label, from_label, to_label, value: int) -> str:
        diff = self._diff
        head = (f"true = {true_label}\n{diff.baseline}: {from_label}\n"
                f"{diff.candidate}: {to_label}\n{value:,} crop(s)")
        # The verdict belongs to crops, so an empty cell does not get one:
        # "0 crop(s), broken by the candidate" says something happened.
        if not value:
            return head
        if from_label == to_label:
            return f"{head}\n(unchanged)"
        if to_label == true_label:
            return f"{head}\n(fixed by the candidate)"
        if from_label == true_label:
            return f"{head}\n(broken by the candidate)"
        return f"{head}\n(one mistake traded for another)"

    def on_window_resized(self) -> None:
        """Both grids are as wide as the window lets them be, so they have
        to follow the window. Only their width: nothing is recomputed."""
        for card_control, grid_width in ((self._matrix_card, self._matrix_grid_width),
                                         (self._flow_card, self._flow_grid_width)):
            if card_control is None:
                continue
            width = grid_card_width(self.ctx.page, grid_width)
            if width != card_control.width:
                card_control.width = width
                if card_control.page:
                    card_control.update()

    def _render_lists(self, diff):
        self.lists_area.controls = [
            section_title("What changed, crop by crop", icon=ft.Icons.PHOTO_LIBRARY_OUTLINED, ctx=self.ctx, manual="fixes-regressions"),
            ft.Row([
                self._outcome_card(cmp.OUTCOME_REGRESSION),
                self._outcome_card(cmp.OUTCOME_FIX),
            ], spacing=12, wrap=True, run_spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.START),
            self._per_class_card(diff),
        ]

    def _outcome_card(self, outcome: str) -> ft.Control:
        rows = cmp.rows_for_outcome(self._joined, outcome)
        color = OUTCOME_COLORS[outcome]
        cols = self.ctx.state.dataset.columns

        body: list = []
        if not len(rows):
            body.append(ft.Text("None.", size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))
        else:
            for _, row in rows.head(CROP_LIST_PAGE).iterrows():
                body.append(self._crop_row(row, outcome))
            if len(rows) > CROP_LIST_PAGE:
                body.append(ft.Text(
                    f"...and {len(rows) - CROP_LIST_PAGE:,} more -- open them all below.",
                    size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))

        return card(ft.Column([
            ft.Row([ft.Icon(OUTCOME_ICONS[outcome], size=17, color=color),
                    ft.Text(f"{OUTCOME_LABELS[outcome]} ({len(rows):,})",
                            weight=ft.FontWeight.BOLD, size=14, color=color)], spacing=8),
            ft.Text("Most confident first.",
                    size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Column(body, spacing=4, scroll=ft.ScrollMode.AUTO, height=280),
            ft.FilledButton(
                f"Open all {len(rows):,} in the carousel", icon=ft.Icons.BURST_MODE_OUTLINED,
                disabled=not len(rows), style=ft.ButtonStyle(bgcolor=ACCENT),
                on_click=lambda e, outcome=outcome: self._open_outcome(outcome)),
        ], spacing=10), width=430)

    def _crop_row(self, row, outcome: str) -> ft.Control:
        cols = self.ctx.state.dataset.columns
        uuid = str(row[cols.uuid_col])
        return ft.Container(
            content=ft.Column([
                ft.Text(uuid, size=11, font_family="monospace",
                        max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                ft.Row([
                    ft.Text(f"true {row[cmp.COL_TRUE]}", size=11, weight=ft.FontWeight.W_500),
                    ft.Text(f"·  {self._diff.baseline}: {row[cmp.COL_PRED_A]} "
                            f"({row[cmp.COL_SCORE_A]:.2f})",
                            size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text(f"·  {self._diff.candidate}: {row[cmp.COL_PRED_B]} "
                            f"({row[cmp.COL_SCORE_B]:.2f})",
                            size=11, color=OUTCOME_COLORS[outcome]),
                ], spacing=6, wrap=True),
            ], spacing=2, tight=True),
            padding=ft.padding.symmetric(horizontal=8, vertical=6),
            border_radius=6, ink=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            tooltip="Open this crop in the carousel",
            on_click=lambda e, uuid=uuid, outcome=outcome: self._open_outcome(outcome, focus_uuid=uuid),
        )

    def _per_class_card(self, diff) -> ft.Control:
        table = self._per_class_table
        if table is None:
            table = cmp.per_class_change(self._joined, diff.labels)
        n_all_classes = len(table)
        if self.class_focus.many:
            table = table[table["label"].isin(set(self._focus))]
        if table.empty:
            return ft.Container()
        header = ft.Row([
            ft.Container(ft.Text("Class", size=11, weight=ft.FontWeight.BOLD), width=140),
            ft.Container(ft.Text("Support", size=11, weight=ft.FontWeight.BOLD), width=80),
            ft.Container(ft.Text("Fixed", size=11, weight=ft.FontWeight.BOLD), width=70),
            ft.Container(ft.Text("Broken", size=11, weight=ft.FontWeight.BOLD), width=70),
            ft.Container(ft.Text("Net", size=11, weight=ft.FontWeight.BOLD), width=70),
        ], spacing=4)
        max_abs = int(max(abs(table["net"].min()), abs(table["net"].max()), 1))
        rows = [header]
        for _, r in table.sort_values("net").iterrows():
            rows.append(ft.Row([
                ft.Container(ft.Text(str(r["label"]), size=12), width=140),
                ft.Container(ft.Text(f"{int(r['support']):,}", size=12,
                                     color=ft.Colors.ON_SURFACE_VARIANT), width=80),
                ft.Container(ft.Text(f"{int(r['fixes']):,}", size=12,
                                     color=ft.Colors.GREEN_400), width=70),
                ft.Container(ft.Text(f"{int(r['regressions']):,}", size=12,
                                     color=ft.Colors.RED_400), width=70),
                ft.Container(
                    ft.Text(f"{int(r['net']):+,}", size=12, weight=ft.FontWeight.BOLD),
                    width=70, bgcolor=diverging_color(int(r["net"]), max_abs),
                    border_radius=4, padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    alignment=ft.alignment.center_left),
            ], spacing=4))
        return card(ft.Column([
            ft.Row([ft.Text("Per class", weight=ft.FontWeight.BOLD, size=14),
                    ft.Container(expand=True),
                    info_button(self.ctx, "per-class-change")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text("Worst first. A flat net number hides a model that traded one class for another."
                    + (f" {len(table)} of {n_all_classes:,} classes, picked under 'Classes shown' above."
                       if self.class_focus.many else ""),
                    size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
            *rows,
        ], spacing=6))

    # ---- carousel -------------------------------------------------------
    def _open_outcome(self, outcome: str, focus_uuid=None):
        rows = cmp.rows_for_outcome(self._joined, outcome)
        self._open_carousel(rows, OUTCOME_LABELS[outcome], focus_uuid)

    def _on_cell_click(self, true_label: str, predicted_label: str):
        """Select the cell and describe it in the provenance panel.

        This used to open the carousel straight away, which answered "which
        crops" while leaving "and where were they before" unanswerable -- a
        cell of any of these three matrices is a set of *movements*, and
        the images alone do not say what moved where. Clicking the same
        cell again clears the selection, so the panel can be put away
        without picking something else.
        """
        picked = (true_label, predicted_label)
        self.selected_cell = None if self.selected_cell == picked else picked
        if self._diff is None:
            return
        self._render_matrix(self._diff)
        if self.matrix_area.page:
            self.matrix_area.update()

    def _open_movement(self, direction: str):
        """Every crop that arrived in the selected cell, or every one that
        left it -- the two buttons under the provenance panel."""
        if self.selected_cell is None:
            return
        true_label, predicted_label = self.selected_cell
        rows = cmp.rows_in_movement_cell(self._joined, direction, true_label, predicted_label)
        self._open_carousel(
            rows, f"{MATRIX_LABELS[direction]} \u2014 true {true_label} / "
                  f"predicted {predicted_label}")

    def _open_flow_cell(self, from_label: str, to_label: str):
        """One cell of the provenance panel: the crops of this true class
        the baseline called `from_label` and the candidate calls
        `to_label`."""
        if self.selected_cell is None:
            return
        true_label = self.selected_cell[0]
        rows = cmp.rows_in_flow_cell(self._joined, true_label, from_label, to_label)
        self._open_carousel(
            rows, f"true {true_label} \u2014 {self._diff.baseline}: {from_label} \u2192 "
                  f"{self._diff.candidate}: {to_label}")

    def _open_carousel(self, rows, title: str, focus_uuid=None):
        if not len(rows):
            return
        state = self.ctx.state
        cols = state.dataset.columns
        # The candidate's own prediction pre-fills the relabel picker. On a
        # confident regression that is exactly the hypothesis worth testing:
        # the new model may be right and the label wrong, which is a
        # relabel, not a model problem.
        suggestions = {
            str(u): str(p) for u, p in zip(rows[cols.uuid_col], rows[cmp.COL_PRED_B])
        }
        # The frame is kept and sliced a page at a time, same as the Metrics
        # tab's confusion cells: a single outcome can be tens of thousands
        # of crops at scale.
        self.carousel_title.value = f"{title}  ·  {len(rows):,} image(s)"
        carousel = Carousel(
            state.dataset, len(rows),
            lambda start, end: rows.iloc[start:end].to_dict("records"),
            cols.crop_col, cols.uuid_col, ctx=self.ctx,
            highlight_uuid=focus_uuid,
            review_source="compare",
            suggestions=suggestions,
            label_col=cmp.COL_TRUE,
        )
        self.carousel_content.content = carousel.build()
        self.ctx.page.open(self.carousel_dialog)

    # ---- build ----------------------------------------------------------
    def build(self) -> ft.Control:
        self._rebuild_dropdowns()
        self._render_placeholder()
        self.root.content = ft.Column([
            section_title("Compare two experiments", icon=ft.Icons.COMPARE_ARROWS, ctx=self.ctx, manual="fixes-regressions"),
            ft.Text("What did the candidate break, and what did it fix?",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            self.stale_banner.control,
            card(self.picker_container),
            self.progress_banner,
            self.summary_area,
            self.class_focus.build(),
            self.matrix_area,
            self.lists_area,
        ], spacing=16, scroll=ft.ScrollMode.AUTO)
        return self.root
