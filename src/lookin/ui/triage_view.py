"""Triage: one crop at a time, decided from the keyboard, auto-advancing.

The carousel is the right place to record the occasional decision while
looking at something else. It is the wrong place to work through two
thousand flagged crops: three images to a page, a dropdown per card and a
mouse round-trip per decision means nobody finishes the queue.

So this is the other mode -- a single large image, four keys, and the next
crop already on screen. Two things make it fast rather than merely
keyboard-driven:

* the relabel target is pre-filled from the evidence (the nearest other
  class, or what the models predicted), so agreeing costs one key;
* reference crops for both classes in question sit under the image, because
  "is this an X or a Y" is answered by looking at known examples of each,
  not by remembering what the class is supposed to look like.

A decision is written and saved immediately (see review_widgets.record_
decision) -- there is no "save session" step to forget, and closing the
panel mid-queue loses nothing.
"""
from __future__ import annotations

import flet as ft

from lookin.core import prefs
from lookin.core.review import (
    FrozenVersionError, REVIEW_STATUSES, STATUS_EXCLUDE, STATUS_OK, STATUS_RELABEL, STATUS_UNCERTAIN,
)
from lookin.ui.class_matrix import make_searchable
from lookin.ui.components import ACCENT, crop_display, file_to_full_base64
from lookin.ui.review_widgets import (
    NO_REASON, STATUS_ICONS, STATUS_LABELS,
    class_options, decision_summary, reason_options, record_decision,
)

IMAGE_SIZE = 380
REFERENCE_THUMB = 74
MAX_REFERENCES = 6

# Flet reports keys by name ("Space", "Arrow Right"), not by character.
KEY_TO_STATUS = {
    "Space": STATUS_OK,
    "K": STATUS_OK,
    "X": STATUS_EXCLUDE,
    "U": STATUS_UNCERTAIN,
    "R": STATUS_RELABEL,
}

SHORTCUT_HINT = (
    "Space / K keep   ·   R relabel   ·   X exclude   ·   U uncertain   ·   "
    "1-9 relabel to that class   ·   ← → move   ·   Z undo   ·   Esc close"
)


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class TriageView:
    """A modal review queue over a CandidateSet.

    One instance is created per Cleaning view and reused for every queue --
    the dialog and its keyboard handler are set up once, and open() just
    swaps in a new list.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.candidates: list = []
        self.row_by_uuid: dict = {}
        self.references: dict = {}  # class label -> [uuid, ...]
        self.source = ""
        self.index = 0
        # Index the queue was resumed at, so _render can say so once -- see
        # _resume_index. None means "started from the top".
        self._resumed_at: int | None = None
        self._history: list = []  # uuids decided in this session, for Z
        self._previous_key_handler = None

        self.title_text = ft.Text(weight=ft.FontWeight.BOLD)
        self.progress_text = ft.Text(size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_bar = ft.ProgressBar(value=0, color=ACCENT, bar_height=4, border_radius=2)
        # Given the frame's own size rather than left to its intrinsic one:
        # crops here are often tiny (a 32x32 chip is normal), and an image
        # left to size itself renders at native resolution -- too small to
        # judge, which is the one thing this panel exists for. The frame in
        # turn stays inside the dialog's width, since an image wider than the
        # slot it lands in is clipped away rather than scaled down.
        self.image = ft.Image(width=IMAGE_SIZE, height=IMAGE_SIZE, fit=ft.ImageFit.CONTAIN,
                              border_radius=10)
        self.image_frame = ft.Container(content=self.image, width=IMAGE_SIZE, height=IMAGE_SIZE,
                                        alignment=ft.alignment.center)
        self.label_row = ft.Row(spacing=10, wrap=True)
        self.detail_row = ft.Row(spacing=14, wrap=True)
        self.decision_row = ft.Container()
        self.class_dropdown = ft.Dropdown(label="Relabel to", width=210, dense=True, border_radius=8)
        self.reason_dropdown = ft.Dropdown(label="Reason", options=reason_options(), value=NO_REASON,
                                           width=190, dense=True, border_radius=8)
        self.references_column = ft.Column(spacing=6)
        self.message_text = ft.Text(size=12)

        self.content = ft.Container(width=980, height=560)
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row([self.title_text, ft.Container(expand=True), self.progress_text], spacing=12),
            content=self.content,
            actions=[
                ft.TextButton("Close", on_click=self.close),
            ],
            on_dismiss=lambda e: self._restore_keyboard(),
        )

    # ---- queue lifecycle ----
    def open(self, candidate_set, row_by_uuid: dict, references: dict | None = None) -> None:
        self.candidates = list(candidate_set.candidates)
        self.row_by_uuid = row_by_uuid
        self.references = references or {}
        self.source = candidate_set.source
        self._history = []
        self._resumed_at = None
        self.title_text.value = candidate_set.name
        self.index = self._resume_index()
        self._build_content()
        self._render()
        self._capture_keyboard()
        self.ctx.page.open(self.dialog)

    # ---- resuming ----
    def _cursor_key(self):
        """(dataset csv, review version) -- what a saved position belongs
        to. Returns (None, "") when either is missing, which disables
        resuming rather than saving a position against nothing."""
        dataset = self.ctx.state.dataset
        store = getattr(self.ctx.state, "review", None)
        version_id = getattr(store, "active_version_id", None) if store is not None else None
        if dataset is None or dataset.csv_path is None or not version_id:
            return None, ""
        return dataset.csv_path, version_id

    def _resume_index(self) -> int:
        """Where this queue was left, or 0.

        The saved cursor is a *uuid*, not a position: the candidate list is
        recomputed from scratch every time the queue is opened (different
        source, different threshold, decisions made since), so index 412 is
        not the same crop tomorrow but the crop itself is. A uuid that is no
        longer in the queue -- usually because it has since been decided --
        simply starts from the top.
        """
        dataset_path, version_id = self._cursor_key()
        if dataset_path is None:
            return 0
        remembered = prefs.load_triage_cursor(dataset_path, version_id)
        if not remembered:
            return 0
        for i, candidate in enumerate(self.candidates):
            if str(candidate.uuid) == remembered:
                _log(f"triage: resuming at {remembered} ({i + 1}/{len(self.candidates)})")
                self._resumed_at = i
                return i
        return 0

    def _save_cursor(self) -> None:
        """Called on every move and every decision. Writing a few dozen
        bytes is cheaper than the render that just happened, and the whole
        point is that closing the browser mid-queue costs nothing."""
        dataset_path, version_id = self._cursor_key()
        candidate = self._current()
        if dataset_path is None or candidate is None:
            return
        prefs.save_triage_cursor(dataset_path, version_id, str(candidate.uuid))

    def close(self, e=None) -> None:
        self._restore_keyboard()
        self.ctx.page.close(self.dialog)

    def _capture_keyboard(self) -> None:
        # The page has exactly one keyboard handler, so the previous one is
        # kept and put back on close rather than clobbered -- otherwise
        # whatever else the app binds later stops working after one triage
        # session, which is the kind of bug nobody traces back to here.
        self._previous_key_handler = self.ctx.page.on_keyboard_event
        self.ctx.page.on_keyboard_event = self._on_key

    def _restore_keyboard(self) -> None:
        self.ctx.page.on_keyboard_event = self._previous_key_handler
        self._previous_key_handler = None

    # ---- current item ----
    def _current(self):
        if 0 <= self.index < len(self.candidates):
            return self.candidates[self.index]
        return None

    def _labels(self) -> list:
        dataset = self.ctx.state.dataset
        return [str(l) for l in dataset.labels()] if dataset is not None else []

    def _reason(self) -> str:
        value = self.reason_dropdown.value
        return "" if value in (None, NO_REASON) else value

    # ---- keyboard ----
    def _on_key(self, e) -> None:
        key = getattr(e, "key", "") or ""
        if key == "Escape":
            self.close()
            return
        if key == "Arrow Right":
            self._move(1)
            return
        if key == "Arrow Left":
            self._move(-1)
            return
        if key == "Z":
            self._undo_last()
            return
        if key.isdigit() and key != "0":
            labels = self._labels()
            position = int(key) - 1
            if position < len(labels):
                self.class_dropdown.value = labels[position]
                self._decide(STATUS_RELABEL)
            return
        status = KEY_TO_STATUS.get(key)
        if status is not None:
            self._decide(status)

    # ---- decisions ----
    def _decide(self, status: str) -> None:
        candidate = self._current()
        if candidate is None:
            return
        new_label = self.class_dropdown.value or ""
        if status == STATUS_RELABEL and not new_label:
            self._message("Pick a class to relabel to.", ok=False)
            return
        ok, message = record_decision(
            self.ctx, [candidate.uuid], status,
            new_label=new_label if status == STATUS_RELABEL else "",
            reason=self._reason(), source=self.source,
        )
        if not ok:
            self._message(message, ok=False)
            return
        self._history.append(candidate.uuid)
        self._message(message, ok=True)
        self._move(1)

    def _undo_last(self) -> None:
        if not self._history:
            self._message("Nothing to undo in this session.", ok=False)
            return
        uid = self._history.pop()
        store = self.ctx.state.review
        try:
            store.reset(uid)
            store.save()
        except FrozenVersionError as ex:
            self._message(str(ex), ok=False)
            return
        self.ctx.notify_review()
        position = next((i for i, c in enumerate(self.candidates) if c.uuid == uid), None)
        if position is not None:
            self.index = position
        self._render()
        self._save_cursor()
        self._message(f"Cleared the decision on {uid}.", ok=True)

    def _move(self, step: int) -> None:
        if not self.candidates:
            return
        self.index = max(0, min(len(self.candidates) - 1, self.index + step))
        self._render()
        self._save_cursor()

    def _message(self, text: str, ok: bool) -> None:
        self.message_text.value = text
        self.message_text.color = ft.Colors.GREEN if ok else ft.Colors.RED
        if self.message_text.page:
            self.message_text.update()

    # ---- rendering ----
    def _thumb(self, uid: str) -> ft.Control | None:
        record = self.row_by_uuid.get(str(uid))
        dataset = self.ctx.state.dataset
        if record is None or dataset is None:
            return None
        crop_col = dataset.columns.crop_col
        if crop_col not in record:
            return None
        return ft.Image(
            src_base64=file_to_full_base64(dataset.resolve_crop_path(record[crop_col]),
                                           *crop_display(self.ctx, record)),
            width=REFERENCE_THUMB, height=REFERENCE_THUMB, fit=ft.ImageFit.COVER, border_radius=6,
        )

    def _reference_strip(self, label: str) -> ft.Control:
        thumbs = []
        for uid in self.references.get(str(label), [])[:MAX_REFERENCES]:
            thumb = self._thumb(uid)
            if thumb is not None:
                thumbs.append(thumb)
        if not thumbs:
            return ft.Text(f"No reference crops for {label}.", size=11, italic=True,
                           color=ft.Colors.ON_SURFACE_VARIANT)
        return ft.Row(thumbs, spacing=6, wrap=False, scroll=ft.ScrollMode.AUTO)

    def _render_references(self, candidate) -> None:
        if not self.references:
            self.references_column.controls = []
            return
        blocks = [ft.Text("Reference crops", size=12, weight=ft.FontWeight.BOLD)]
        blocks.append(ft.Text(f"labelled {candidate.label}", size=11, color=ft.Colors.ON_SURFACE_VARIANT))
        blocks.append(self._reference_strip(candidate.label))
        if candidate.suggestion and candidate.suggestion != candidate.label:
            blocks.append(ft.Text(f"suggested {candidate.suggestion}", size=11, color=ACCENT))
            blocks.append(self._reference_strip(candidate.suggestion))
        self.references_column.controls = blocks

    def _render(self) -> None:
        candidate = self._current()
        total = len(self.candidates)
        resumed = getattr(self, "_resumed_at", None)
        position = f"{min(self.index + 1, total)} / {total:,}" if total else "empty queue"
        if resumed is not None and self.index == resumed and total:
            # Otherwise opening a queue at crop 412 looks like the first 411
            # went missing.
            position += "  ·  resumed where you left off"
        self.progress_text.value = position
        self.progress_bar.value = ((self.index + 1) / total) if total else 0
        # The counter lives in the dialog's title, not in its content, so
        # updating the content alone leaves it a decision behind.
        if self.progress_text.page:
            self.progress_text.update()

        if candidate is None:
            self.image.src_base64 = None
            self.label_row.controls = [ft.Text("Queue finished.", size=14, weight=ft.FontWeight.BOLD)]
            self.detail_row.controls = []
            self.references_column.controls = []
            self.decision_row.content = None
            if self.content.page:
                self.content.update()
            return

        record = self.row_by_uuid.get(candidate.uuid)
        dataset = self.ctx.state.dataset
        if record is not None and dataset is not None:
            self.image.src_base64 = file_to_full_base64(
                dataset.resolve_crop_path(record[dataset.columns.crop_col]),
                *crop_display(self.ctx, record),
            )

        self.label_row.controls = [
            ft.Text(candidate.uuid, size=12, selectable=True, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Container(
                content=ft.Text(f"labelled {candidate.label}", size=13, weight=ft.FontWeight.BOLD),
                padding=ft.padding.symmetric(horizontal=10, vertical=4),
                border_radius=8, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            ),
        ]
        if candidate.suggestion and candidate.suggestion != candidate.label:
            self.label_row.controls.append(ft.Container(
                content=ft.Text(f"suggested {candidate.suggestion}", size=13, weight=ft.FontWeight.BOLD, color=ACCENT),
                padding=ft.padding.symmetric(horizontal=10, vertical=4),
                border_radius=8, bgcolor=ft.Colors.with_opacity(0.14, ACCENT),
            ))
        self.label_row.controls.append(decision_summary(self.ctx, candidate.uuid))

        self.detail_row.controls = [
            ft.Text(f"{key}: {value}", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
            for key, value in candidate.detail.items()
        ]

        labels = self._labels()
        self.class_dropdown.options = class_options(labels)
        make_searchable(self.class_dropdown)
        self.class_dropdown.value = (
            candidate.suggestion if candidate.suggestion in labels
            else (candidate.label if candidate.label in labels else (labels[0] if labels else None))
        )
        self.decision_row.content = ft.Row([
            ft.FilledTonalButton(
                STATUS_LABELS[status], icon=STATUS_ICONS[status],
                on_click=lambda e, s=status: self._decide(s),
            )
            for status in REVIEW_STATUSES
        ] + [
            ft.IconButton(ft.Icons.UNDO, tooltip="Undo the last decision (Z)", on_click=lambda e: self._undo_last()),
            ft.IconButton(ft.Icons.CHEVRON_LEFT, tooltip="Previous (left arrow)", on_click=lambda e: self._move(-1)),
            ft.IconButton(ft.Icons.CHEVRON_RIGHT, tooltip="Skip (right arrow)", on_click=lambda e: self._move(1)),
        ], spacing=8, wrap=True)

        self._render_references(candidate)
        if self.content.page:
            self.content.update()

    def _build_content(self) -> None:
        store = getattr(self.ctx.state, "review", None)
        version_note = ""
        if store is not None and store.active_version_id:
            version_note = f"Recording into version '{store.active_version_id}'."
            if store.is_frozen(store.active_version_id):
                version_note += " It is frozen -- create a new version before deciding anything."
        self.content.content = ft.Row([
            ft.Column([
                self.image_frame,
                self.label_row,
                self.detail_row,
            ], spacing=10, width=IMAGE_SIZE + 16, scroll=ft.ScrollMode.AUTO),
            ft.VerticalDivider(width=1),
            ft.Column([
                ft.Text(version_note, size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                self.progress_bar,
                self.decision_row,
                ft.Row([self.class_dropdown, self.reason_dropdown], spacing=10, wrap=True),
                ft.Text(SHORTCUT_HINT, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                self.message_text,
                ft.Divider(height=1),
                self.references_column,
            ], spacing=12, expand=True, scroll=ft.ScrollMode.AUTO),
        ], spacing=18, vertical_alignment=ft.CrossAxisAlignment.START)
