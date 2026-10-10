"""Shared review controls: the decision bar that appears under a crop, and
the display vocabulary for statuses and reasons.

The split between this module and core/review.py is the one thing to keep
straight while editing either. core/review.py holds the *keys* -- `ok`,
`exclude`, `radiometry` -- which are compared in code and written into files
that outlive any wording. This module holds the *labels* the user reads, and
every one of them can be reworded freely without touching a single stored
file. Never let a label become a key by using it in a comparison.

The decision bar lives here rather than in any one tab because the carousel
is shared by the Gallery, the confusion-matrix drill-down, the Embeddings
case studies and Cleaning: putting the controls on the carousel card makes
relabelling reachable from all four at once, instead of stranding it in the
Cleaning tab where the crops the user is actually looking at mostly aren't.
"""
from __future__ import annotations

import flet as ft

from lookin.core.review import (
    FrozenVersionError, REVIEW_REASONS, REVIEW_STATUSES,
    STATUS_EXCLUDE, STATUS_OK, STATUS_RELABEL, STATUS_UNCERTAIN,
)
from lookin.ui.class_matrix import searchable_dropdown_kwargs
from lookin.ui.components import ACCENT

STATUS_LABELS = {
    STATUS_OK: "Keep",
    STATUS_RELABEL: "Relabel",
    STATUS_EXCLUDE: "Exclude",
    STATUS_UNCERTAIN: "Uncertain",
}

STATUS_ICONS = {
    STATUS_OK: ft.Icons.CHECK_CIRCLE_OUTLINE,
    STATUS_RELABEL: ft.Icons.LABEL_OUTLINE,
    STATUS_EXCLUDE: ft.Icons.BLOCK,
    STATUS_UNCERTAIN: ft.Icons.HELP_OUTLINE,
}

STATUS_COLORS = {
    STATUS_OK: ft.Colors.GREEN,
    STATUS_RELABEL: ACCENT,
    STATUS_EXCLUDE: ft.Colors.RED,
    STATUS_UNCERTAIN: ft.Colors.ORANGE,
}

# Single-key shortcuts for the triage panel. Chosen to be reachable without
# looking: the four decisions sit under the left hand, the class keys are
# the number row.
STATUS_SHORTCUTS = {
    " ": STATUS_OK,
    "K": STATUS_OK,
    "X": STATUS_EXCLUDE,
    "U": STATUS_UNCERTAIN,
    "R": STATUS_RELABEL,
}

REASON_LABELS = {
    "blurred": "Blurred",
    "radiometry": "Radiometry",
    "occluded": "Occluded",
    "truncated": "Truncated crop",
    "multiple_objects": "Multiple objects",
    "wrong_crop": "Wrong crop",
    "not_an_object": "Not an object",
    "duplicate": "Duplicate",
    "ambiguous": "Ambiguous",
}

NO_REASON = "__none__"


def reason_options() -> list:
    """Dropdown options for the reason vocabulary. The option *key* is the
    stored reason -- a key, never reworded -- and the text is the label."""
    return [ft.dropdown.Option(NO_REASON, "No reason")] + [
        ft.dropdown.Option(reason, REASON_LABELS.get(reason, reason)) for reason in REVIEW_REASONS
    ]


def class_options(labels: list) -> list:
    """Dropdown options for the dataset's class labels. Key and text are
    both the label itself: these values come from the user's CSV and are
    stored verbatim, so there is nothing to map."""
    return [ft.dropdown.Option(str(label)) for label in labels]


def status_chip(status: str, inherited_from: str = "") -> ft.Control:
    """A compact colored badge for one decision, marking an inherited one so
    a decision made two versions ago never looks like it was made here."""
    color = STATUS_COLORS.get(status, ft.Colors.ON_SURFACE_VARIANT)
    text = STATUS_LABELS.get(status, status)
    if inherited_from:
        text = f"{text} · from {inherited_from}"
    return ft.Container(
        content=ft.Row([
            ft.Icon(STATUS_ICONS.get(status, ft.Icons.CIRCLE_OUTLINED), size=13, color=color),
            ft.Text(text, size=11, color=color, weight=ft.FontWeight.BOLD),
        ], spacing=5, tight=True),
        padding=ft.padding.symmetric(horizontal=8, vertical=3),
        border_radius=6,
        bgcolor=ft.Colors.with_opacity(0.12, color),
    )


def decision_summary(ctx, uid) -> ft.Control:
    """The current effective decision for one crop, or a neutral marker."""
    store = getattr(ctx.state, "review", None)
    if store is None or not store.versions:
        return ft.Text("not reviewed", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
    decision = store.decision_for(str(uid))
    if decision is None:
        return ft.Text("not reviewed", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
    inherited = decision.version_id if decision.version_id != store.active_version_id else ""
    parts = [status_chip(decision.status, inherited)]
    if decision.status == STATUS_RELABEL and decision.new_label:
        parts.append(ft.Text(f"→ {decision.new_label}", size=11, weight=ft.FontWeight.BOLD))
    if decision.reason:
        parts.append(ft.Text(REASON_LABELS.get(decision.reason, decision.reason), size=11,
                             color=ft.Colors.ON_SURFACE_VARIANT))
    return ft.Row(parts, spacing=8, tight=True, wrap=True)


# The reset button sits on the same row as the two pickers, so its width is
# part of the arithmetic that keeps that row on one line.
RESET_BTN_WIDTH = 40
PICKER_GAP = 8
MIN_PICKER_WIDTH = 120


class DecisionBar:
    """Review controls for a single crop: four decision buttons, a class
    picker for relabelling and a reason picker.

    The class picker is pre-filled with whatever the evidence suggested (the
    nearest other class, or the model's prediction), so the common case is
    one click on Relabel rather than a hunt through a forty-entry dropdown.
    """

    def __init__(self, ctx, uid, current_label, suggestion: str = "", source: str = "",
                 on_decided=None, compact: bool = False, width: int | None = None):
        self.ctx = ctx
        self.uid = str(uid)
        self.current_label = current_label
        self.source = source
        self.on_decided = on_decided
        self.compact = compact
        self.width = width

        labels = ctx.state.dataset.labels() if ctx.state.dataset is not None else []
        default_class = suggestion if suggestion in [str(l) for l in labels] else (
            str(labels[0]) if labels else None
        )
        # The two pickers and the reset button share one row. Sized from the
        # space the caller actually has rather than a fixed guess: at a fixed
        # 142 each they overflowed a narrow card, the row wrapped, and the
        # reset button ended up alone on a line of its own at the bottom.
        picker_width = self._picker_width()
        self.class_dropdown = ft.Dropdown(
            label="New class", options=class_options(labels), value=default_class,
            width=picker_width, dense=True, border_radius=8,
            **searchable_dropdown_kwargs(len(labels)),
        )
        self.reason_dropdown = ft.Dropdown(
            label="Reason", options=reason_options(), value=NO_REASON,
            width=picker_width, dense=True, border_radius=8,
        )
        self.status_area = ft.Container(content=decision_summary(ctx, self.uid))
        self.root = ft.Column(spacing=8, tight=True)

    def _picker_width(self) -> int:
        """Half of what is left once the reset button and the gaps between
        the three controls are taken out."""
        if self.width is None:
            return 142 if self.compact else 175
        return int(max(MIN_PICKER_WIDTH, (self.width - RESET_BTN_WIDTH - 2 * PICKER_GAP) / 2))

    def _reason(self) -> str:
        value = self.reason_dropdown.value
        return "" if value in (None, NO_REASON) else value

    def _on_click(self, status: str):
        new_label = self.class_dropdown.value or ""
        ok, message = record_decision(
            self.ctx, [self.uid], status,
            new_label=new_label if status == STATUS_RELABEL else "",
            reason=self._reason(), source=self.source,
        )
        self.status_area.content = decision_summary(self.ctx, self.uid) if ok else ft.Text(
            message, size=11, color=ft.Colors.RED,
        )
        if self.status_area.page:
            self.status_area.update()
        if self.on_decided:
            self.on_decided(self.uid, status, ok, message)

    def _on_reset(self, e):
        store = self.ctx.state.review
        try:
            store.reset(self.uid)
            store.save()
            self.ctx.notify_review()
        except FrozenVersionError as ex:
            self.status_area.content = ft.Text(str(ex), size=11, color=ft.Colors.RED)
        else:
            self.status_area.content = decision_summary(self.ctx, self.uid)
        if self.status_area.page:
            self.status_area.update()

    def build(self) -> ft.Control:
        size = 11 if self.compact else 12
        # Tighter padding and text inside a card, so the four verdicts stay
        # on one line: they are one choice between four, and splitting them
        # across two rows reads as two groups that mean different things.
        # wrap stays on as a fallback for a card too narrow even for this.
        button_style = ft.ButtonStyle(
            padding=ft.padding.symmetric(horizontal=6 if self.compact else 10, vertical=6),
            text_style=ft.TextStyle(size=size) if self.compact else None,
        )
        buttons = [
            ft.FilledTonalButton(
                STATUS_LABELS[status], icon=STATUS_ICONS[status],
                icon_color=STATUS_COLORS.get(status),
                on_click=lambda e, s=status: self._on_click(s),
                style=button_style,
            )
            for status in REVIEW_STATUSES
        ]
        pickers = ft.Row(
            [self.class_dropdown, self.reason_dropdown,
             ft.IconButton(ft.Icons.RESTART_ALT, tooltip="Clear this decision",
                           icon_size=16, width=RESET_BTN_WIDTH, on_click=self._on_reset)],
            spacing=PICKER_GAP, wrap=False,
            vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.root.controls = [
            ft.Row(buttons, spacing=5, wrap=True, run_spacing=5),
            pickers,
            self.status_area,
        ]
        if not self.compact:
            self.root.controls.insert(0, ft.Text("Review", size=size, weight=ft.FontWeight.BOLD))
        return self.root


def record_decision(ctx, uuids, status: str, new_label: str = "", reason: str = "",
                    note: str = "", source: str = "") -> tuple:
    """Write one decision (or a whole batch) and persist immediately.

    Saving on every decision rather than on a timer or at exit is
    deliberate: the store only holds decided rows, so writing it is
    milliseconds, and a review session that loses an afternoon of judgement
    calls to a crashed browser tab is not a session anyone repeats.

    Returns (ok, message). A frozen active version comes back as a refusal
    with the fix in the message, not an exception -- it is a normal state
    (every version gets frozen the moment it is exported), and the caller
    surfaces it where the user is already looking.
    """
    store = getattr(ctx.state, "review", None)
    if store is None:
        return False, "Load a dataset before reviewing."
    uuids = [str(u) for u in uuids]
    try:
        if len(uuids) == 1:
            store.decide(uuids[0], status, new_label=new_label, reason=reason, note=note, source=source)
            message = f"{STATUS_LABELS.get(status, status)} recorded."
        else:
            batch_id = store.decide_many(uuids, status, new_label=new_label, reason=reason,
                                         note=note, source=source)
            message = f"{STATUS_LABELS.get(status, status)} applied to {len(uuids):,} crops (batch {batch_id})."
    except FrozenVersionError as ex:
        return False, str(ex)
    except ValueError as ex:
        return False, str(ex)
    store.save()
    ctx.notify_review()
    return True, message
