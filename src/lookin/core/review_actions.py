"""Recording review decisions, and the words a user reads for them.

The split between this module and core/review.py is the one thing to keep
straight while editing either. core/review.py holds the *keys* -- `ok`,
`exclude`, `radiometry` -- which are compared in code and written into files
that outlive any wording. This module holds the *labels* the user reads, and
every one of them can be reworded freely without touching a single stored
file. Never let a label become a key by using it in a comparison.

record_decision is the one way a decision is written, whether it comes from
the carousel, the triage queue, the Cleaning tab or the MCP endpoint, so
every one of them saves the same way and reports the same refusals.
"""
from __future__ import annotations

from lookin.core.review import (
    FrozenVersionError, STATUS_EXCLUDE, STATUS_OK, STATUS_RELABEL, STATUS_UNCERTAIN,
)

STATUS_LABELS = {
    STATUS_OK: "Keep",
    STATUS_RELABEL: "Relabel",
    STATUS_EXCLUDE: "Exclude",
    STATUS_UNCERTAIN: "Uncertain",
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


def record_decision(session, uuids, status: str, new_label: str = "", reason: str = "",
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
    store = getattr(session.state, "review", None)
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
    try:
        store.save()
    except (OSError, ValueError) as ex:
        # A read-only or full disk (OSError), or a sidecar another session
        # or a hand edit left unparsable, which save() refuses to write over
        # (ValueError). The decision stays in memory and still pending (the
        # next save that works writes it), but the user must not be told it
        # is safe.
        session.notify_review()
        where = store.decisions_path.name if store.decisions_path is not None else "the review file"
        return False, f"Recorded here but NOT saved to {where}: {ex}"
    session.notify_review()
    return True, message
