"""Triage: one crop at a time, decided from the keyboard, auto-advancing --
the web twin of lookin.ui.triage_view.TriageView.

Same queue, same keys (Space / K keep, R relabel, X exclude, U uncertain,
1-9 relabel to that class, left / right move, Z undo, Esc close), same
pre-filled relabel target (the suggestion, else the current label), same
reference crops for both classes in question, same resume-where-you-left-
off cursor (core.prefs, keyed by dataset CSV and review version), and every
decision written and saved at once through review_widgets.record_decision.

The keys are read by static/js/cleaning.js, which only acts while this
panel is the top dialog and focus is not in a field, and sends them as
`key` actions with the original's Flet key names ("Space", "Arrow Right").
"""
from __future__ import annotations

from lookin.core import prefs
from lookin.core.review import FrozenVersionError, REVIEW_STATUSES, STATUS_RELABEL
from lookin.ui.review_widgets import NO_REASON, STATUS_LABELS, record_decision
from lookin.ui.triage_view import (
    IMAGE_SIZE, KEY_TO_STATUS, MAX_REFERENCES, REFERENCE_THUMB, SHORTCUT_HINT,
)
from lookin.web import html as h
from lookin.web.views.base import Dialog
from lookin.web.views.carousel import STATUS_CLS, STATUS_ICON, decision_summary, reason_options
from lookin.web.views.details import details_url
from lookin.web.views.media import crop_url
from lookin.web.views.widgets import class_select


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class TriageDialog(Dialog):
    """A modal review queue over a CandidateSet."""

    width = 1000

    def __init__(self, session, candidate_set, row_by_uuid: dict, references: dict | None = None):
        super().__init__(candidate_set.name)
        self.session = session
        self.candidates = list(candidate_set.candidates)
        self.row_by_uuid = row_by_uuid
        self.references = references or {}
        self.source = candidate_set.source
        self._history: list = []  # uuids decided in this session, for Z
        self._resumed_at = None
        self.class_value = None
        self.reason_value = NO_REASON
        self.message = ("", True)
        self.index = self._resume_index()
        self._load_current()

    # ---- resuming ----
    def _cursor_key(self):
        dataset = self.session.state.dataset
        store = getattr(self.session.state, "review", None)
        version_id = getattr(store, "active_version_id", None) if store is not None else None
        if dataset is None or dataset.csv_path is None or not version_id:
            return None, ""
        return dataset.csv_path, version_id

    def _resume_index(self) -> int:
        """Where this queue was left (a uuid, not a position), or 0."""
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
        dataset_path, version_id = self._cursor_key()
        candidate = self._current()
        if dataset_path is None or candidate is None:
            return
        prefs.save_triage_cursor(dataset_path, version_id, str(candidate.uuid))

    # ---- current item ----
    def _current(self):
        if 0 <= self.index < len(self.candidates):
            return self.candidates[self.index]
        return None

    def _labels(self) -> list:
        dataset = self.session.state.dataset
        return [str(l) for l in dataset.labels()] if dataset is not None else []

    def _load_current(self) -> None:
        """The relabel target, pre-filled from the evidence for each crop."""
        candidate = self._current()
        if candidate is None:
            return
        labels = self._labels()
        self.class_value = (
            candidate.suggestion if candidate.suggestion in labels
            else (candidate.label if candidate.label in labels else (labels[0] if labels else None))
        )

    def _reason(self) -> str:
        return "" if self.reason_value in (None, NO_REASON) else self.reason_value

    # ---- keyboard ----
    def on_key(self, key: str = "", **_):
        key = str(key or "")
        if key == "Escape":
            self.close()
            return
        if key == "Arrow Right":
            self._move(1)
        elif key == "Arrow Left":
            self._move(-1)
        elif key == "Z":
            self._undo_last()
        elif key.isdigit() and key != "0":
            labels = self._labels()
            position = int(key) - 1
            if position < len(labels):
                self.class_value = labels[position]
                self._decide(STATUS_RELABEL)
        else:
            status = KEY_TO_STATUS.get(key)
            if status is not None:
                self._decide(status)
        self.refresh()

    # ---- decisions ----
    def on_decide(self, status: str = "", **_):
        self._decide(str(status))
        self.refresh()

    def _decide(self, status: str) -> None:
        candidate = self._current()
        if candidate is None:
            return
        new_label = self.class_value or ""
        if status == STATUS_RELABEL and not new_label:
            self.message = ("Pick a class to relabel to.", False)
            return
        ok, message = record_decision(
            self.session, [candidate.uuid], status,
            new_label=new_label if status == STATUS_RELABEL else "",
            reason=self._reason(), source=self.source,
        )
        if not ok:
            self.message = (message, False)
            return
        self._history.append(candidate.uuid)
        self.message = (message, True)
        self._move(1)

    def on_undo(self, **_):
        self._undo_last()
        self.refresh()

    def _undo_last(self) -> None:
        if not self._history:
            self.message = ("Nothing to undo in this session.", False)
            return
        uid = self._history.pop()
        store = self.session.state.review
        try:
            store.reset(uid)
            store.save()
        except FrozenVersionError as ex:
            self.message = (str(ex), False)
            return
        self.session.notify_review()
        position = next((i for i, c in enumerate(self.candidates) if c.uuid == uid), None)
        if position is not None:
            self.index = position
            self._load_current()
        self._save_cursor()
        self.message = (f"Cleared the decision on {uid}.", True)

    def on_move(self, step: int = 1, **_):
        self._move(int(step))
        self.refresh()

    def _move(self, step: int) -> None:
        if not self.candidates:
            return
        self.index = max(0, min(len(self.candidates) - 1, self.index + step))
        # The original re-renders the crop on every move, which resets the
        # relabel target from the evidence.
        self._load_current()
        self._save_cursor()

    def on_cls(self, value: str = "", **_):
        if value in self._labels():
            self.class_value = value
        self.refresh()

    def on_reason(self, value: str = "", **_):
        self.reason_value = value or NO_REASON
        self.refresh()

    def on_review_change(self) -> None:
        self.refresh()

    # ---- rendering ----
    def _position_text(self) -> str:
        total = len(self.candidates)
        position = f"{min(self.index + 1, total)} / {total:,}" if total else "empty queue"
        if self._resumed_at is not None and self.index == self._resumed_at and total:
            # Otherwise opening a queue at crop 412 looks like the first 411
            # went missing.
            position += "  ·  resumed where you left off"
        return position

    def title_html(self) -> str:
        return (f'<span class="ell">{h.e(self.title)}</span><span class="grow"></span>'
                f'<span class="muted small tri-pos">{h.e(self._position_text())}</span>')

    def _crop_src(self, uid):
        record = self.row_by_uuid.get(str(uid))
        dataset = self.session.state.dataset
        if record is None or dataset is None:
            return None
        crop_col = dataset.columns.crop_col
        if crop_col not in record:
            return None
        return crop_url(self.session, dataset.resolve_crop_path(record[crop_col]), record)

    def _reference_strip(self, label: str) -> str:
        thumbs = []
        for uid in self.references.get(str(label), [])[:MAX_REFERENCES]:
            src = self._crop_src(uid)
            if src is not None:
                thumbs.append(f'<img class="crop" src="{h.e(src)}" alt="{h.e(uid)}" title="{h.e(uid)}" '
                              f'data-zoom data-pixelated data-title="{h.e(uid)}" data-info="{h.e(details_url(self.session, uid))}" '
                              f'style="width:{REFERENCE_THUMB}px;height:{REFERENCE_THUMB}px;object-fit:cover">')
        if not thumbs:
            return f'<div class="muted small"><i>No reference crops for {h.e(label)}.</i></div>'
        return f'<div class="tri-refs">{"".join(thumbs)}</div>'

    def _references_html(self, candidate) -> str:
        if not self.references:
            return ""
        parts = ['<div class="cl-sub">Reference crops</div>',
                 f'<div class="muted small">labelled {h.e(candidate.label)}</div>',
                 self._reference_strip(candidate.label)]
        if candidate.suggestion and candidate.suggestion != candidate.label:
            parts.append(f'<div class="small c-acc">suggested {h.e(candidate.suggestion)}</div>')
            parts.append(self._reference_strip(candidate.suggestion))
        return "".join(parts)

    def body(self) -> str:
        candidate = self._current()
        total = len(self.candidates)
        fraction = ((self.index + 1) / total) if total else 0
        store = getattr(self.session.state, "review", None)
        version_note = ""
        if store is not None and store.active_version_id:
            version_note = f"Recording into version '{store.active_version_id}'."
            if store.is_frozen(store.active_version_id):
                version_note += " It is frozen -- create a new version before deciding anything."
        text, ok = self.message
        message = f'<div class="small {"c-ok" if ok else "c-err"}">{h.e(text)}</div>' if text else ""
        pbar = f'<div class="pbar tri-bar"><div style="width:{100 * fraction:.1f}%"></div></div>'

        if candidate is None:
            left = (f'<div class="tri-img" style="width:{IMAGE_SIZE}px;height:{IMAGE_SIZE}px"></div>'
                    f'<div class="tri-done">Queue finished.</div>')
            right = (f'<div class="muted small"><i>{h.e(version_note)}</i></div>{pbar}'
                     f'<div class="muted small">{h.e(SHORTCUT_HINT)}</div>{message}')
            return self._layout(left, right)

        src = self._crop_src(candidate.uuid)
        img = (f'<img class="crop" src="{h.e(src)}" alt="{h.e(candidate.uuid)}" data-zoom data-pixelated '
               f'data-info="{h.e(details_url(self.session, candidate.uuid))}" '
               f'data-title="{h.e(candidate.uuid)}" style="width:{IMAGE_SIZE}px;height:{IMAGE_SIZE}px;object-fit:contain">'
               if src else "")
        # The next crop is already fetched, so the queue never waits on it.
        nxt = self.candidates[self.index + 1] if self.index + 1 < total else None
        nxt_src = self._crop_src(nxt.uuid) if nxt is not None else None
        prefetch = f'<img src="{h.e(nxt_src)}" alt="" class="tri-prefetch">' if nxt_src else ""
        labels = (f'<div class="row-wrap tri-labels"><span class="muted small mono cl-sel">{h.e(candidate.uuid)}</span>'
                  f'<span class="tri-chip">labelled {h.e(candidate.label)}</span>')
        if candidate.suggestion and candidate.suggestion != candidate.label:
            labels += f'<span class="tri-chip tri-sugg">suggested {h.e(candidate.suggestion)}</span>'
        labels += f'{decision_summary(self.session, candidate.uuid)}</div>'
        details = "".join(f'<span>{h.e(key)}: {h.e(value)}</span>' for key, value in candidate.detail.items())
        left = (f'<div class="tri-img" style="width:{IMAGE_SIZE}px;height:{IMAGE_SIZE}px">{img}</div>{prefetch}'
                f'{labels}<div class="row-wrap muted small tri-detail">{details}</div>')

        buttons = "".join(
            f'<button type="button" class="btn btn-sm dec"{self.a("decide", status=st)}>'
            f'{h.icon(STATUS_ICON[st], 13, STATUS_CLS[st])}<span>{h.e(STATUS_LABELS[st])}</span></button>'
            for st in REVIEW_STATUSES)
        buttons += (h.button(icon_name="rotate", act_attr=self.a("undo"), small=True, kind="ghost",
                             title="Undo the last decision (Z)")
                    + h.button(icon_name="chev-left", act_attr=self.a("move", step=-1), small=True, kind="ghost",
                               title="Previous (left arrow)")
                    + h.button(icon_name="chev-right", act_attr=self.a("move", step=1), small=True, kind="ghost",
                               title="Skip (right arrow)"))
        pickers = (class_select(self._labels(), self.class_value, self.c("cls"), label="Relabel to", cls="tri-pick")
                   + h.select(reason_options(), self.reason_value, change=self.c("reason"), label="Reason"))
        right = (f'<div class="muted small"><i>{h.e(version_note)}</i></div>{pbar}'
                 f'<div class="row-wrap">{buttons}</div><div class="row-wrap">{pickers}</div>'
                 f'<div class="muted small">{h.e(SHORTCUT_HINT)}</div>{message}'
                 f'<hr class="cl-hr">{self._references_html(candidate)}')
        return self._layout(left, right)

    def _layout(self, left: str, right: str) -> str:
        return (f'<div class="triage" data-triage="{self.id}">'
                f'<div class="tri-left" style="width:{IMAGE_SIZE + 16}px">{left}</div>'
                f'<div class="tri-right">{right}</div></div>')
