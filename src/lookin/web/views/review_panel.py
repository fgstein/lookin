"""The review version manager and the export that consumes it, embedded at
the top of the Cleaning tab.

Three cards:

* Review version -- the active version, New version / Freeze, the counts,
  reasons and the bulk actions of the version (each undoable as a batch),
  and the overlay switch with the version the metrics are scored against;
* What changed between two versions -- diff_versions, resolved decisions;
* Export for training -- the enriched CSV (and the excluded uuids) written
  to a folder on the machine running Lookin, picked with FileBrowserDialog.

This is a component, not a view: its controls are declared as
`<owner>:<action>` with an `op` argument and the owner forwards them to
`handle(op, args)` (see views/widgets.py for the same pattern).
"""
from __future__ import annotations

import traceback
from pathlib import Path

from lookin.core.review import (
    NO_PARENT, REVIEW_STATUSES, STATUS_UNREVIEWED, diff_versions,
    export_enriched_dataset, export_excluded_uuids,
)
from lookin.core.review_actions import REASON_LABELS, STATUS_LABELS
from lookin.web import html as h
from lookin.web.views.base import Dialog
from lookin.web.views.common import saved_path_message
from lookin.web.views.files import FileBrowserDialog


# The "no pin" entry of the overlay version picker. A key, compared in code.
ACTIVE_VERSION = "__active__"

# A colour per review status, as CSS classes (green / accent / red / orange).
STATUS_TONE = {"ok": "ok", "relabel": "acc", "exclude": "err", "uncertain": "warn"}


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def count_pill(label: str, value, tone: str = "") -> str:
    text = f"{value:,}" if isinstance(value, int) else str(value)
    return (f'<span class="cl-pill"><b class="{f"t-{tone}" if tone else "muted"}">{h.e(text)}</b>'
            f'<span class="muted">{h.e(label)}</span></span>')


class NewVersionDialog(Dialog):
    """"New review version": name, parent, note."""

    width = 500

    def __init__(self, panel):
        super().__init__("New review version")
        self.panel = panel
        store = panel.store()
        self.name = store.next_version_id()
        self.parent = store.active_version_id or NO_PARENT
        self.note = ""

    def body(self) -> str:
        store = self.panel.store()
        options = [(NO_PARENT, "nothing (independent pass)")] + [(vid, vid) for vid in (store.versions if store else [])]
        return (f'<form class="stack" id="{self.id}-f"{self.submit("create")}>'
                f'<div class="muted">A version inherits every decision of its parent and stores only what it changes.</div>'
                f'<div class="row">{h.text_input(self.name, name="name", label="Version name", width="200px", autofocus=True)}'
                f'{h.select(options, self.parent, name="parent", label="Inherits from")}</div>'
                f'{h.text_input(self.note, name="note", label="What this pass is for", width="420px")}'
                f'</form>')

    def footer(self) -> str:
        return (f'<button type="submit" form="{self.id}-f" class="btn btn-primary"><span>Create</span></button>'
                + h.button("Cancel", act_attr=self.a("close")))

    def on_create(self, name: str = "", parent: str = "", note: str = "", **_):
        self.name, self.parent, self.note = name, parent, note
        store = self.panel.store()
        if store is None:
            self.close()
            return
        try:
            store.create_version(
                version_id=(name or "").strip(),
                parent=parent or NO_PARENT,
                note=(note or "").strip(),
            )
            store.save()
        except ValueError as ex:
            self.error = str(ex)
            self.refresh()
            return
        self.close()
        self.panel.s.notify_review()


class ReviewPanel:
    """Version selector + counts + batch undo + export."""

    def __init__(self, session, target: str, action: str = "rp"):
        self.s = session
        self.target = target
        self.action = action
        self.status = ("", True)            # the version card's status line
        self.diff_a = None
        self.diff_b = None
        self.diff_result = None             # ("message", kind) | VersionDiff
        self.export_checks: dict = {}       # version id -> ticked
        self.export_dir = ""
        self.export_name = ""
        self.include_reason = True
        self.drop_excluded = False
        self.export_status = ([], True)     # (lines, ok)

    # ---- helpers ----
    def store(self):
        return getattr(self.s.state, "review", None)

    def _a(self, op: str, **kw) -> str:
        return h.act(self.target, self.action, op=op, **kw)

    def _c(self, op: str, **kw) -> str:
        return h.on_change(self.target, self.action, op=op, **kw)

    @staticmethod
    def _status_label(status: str) -> str:
        return STATUS_LABELS.get(status, "not reviewed" if status == STATUS_UNREVIEWED else status)

    def _verdict_text(self, decision) -> str:
        if decision is None:
            return "not reviewed"
        label = self._status_label(decision.status)
        return f"{label} -> {decision.new_label}" if decision.new_label else label

    def _default_export_name(self) -> str:
        dataset = self.s.state.dataset
        if dataset is None or dataset.csv_path is None:
            return "dataset_reviewed.csv"
        return f"{Path(dataset.csv_path).stem}_reviewed.csv"

    # ---- keeping the pickers consistent with the store ----
    def _sync(self) -> None:
        store, dataset = self.store(), self.s.state.dataset
        if store is None or dataset is None:
            return
        names = list(store.versions)
        if store.active_version_id not in names:
            store.active_version_id = store.resume_version_id() or (names[-1] if names else None)
        # Default to the two most recent, which is the pair anyone actually
        # wants: "what did my latest pass change".
        if self.diff_a not in names:
            self.diff_a = names[-2] if len(names) > 1 else (names[0] if names else None)
        if self.diff_b not in names:
            self.diff_b = names[-1] if names else None
        for vid in names:
            self.export_checks.setdefault(vid, vid == store.active_version_id)
        if not self.export_dir and dataset.csv_path is not None:
            self.export_dir = str(Path(dataset.csv_path).parent)
        if not self.export_name:
            self.export_name = self._default_export_name()

    # ---- rendering ----
    def render(self) -> str:
        has_dataset = self.s.state.dataset is not None and self.store() is not None
        if not has_dataset:
            return h.banner("Load a dataset on the Sources tab to start reviewing -- decisions are stored next to its CSV.",
                            "info")
        self._sync()
        return (f'<div class="cl-review">{self._version_card()}'
                f'<div class="cl-review-side">{self._diff_card()}{self._export_card()}</div></div>')

    def _version_card(self) -> str:
        store = self.store()
        state = self.s.state
        names = list(store.versions)
        options = [(vid, f"{vid}  ({store.n_decisions(vid):,})" + (" · frozen" if store.versions[vid].frozen else ""))
                   for vid in names]
        active = store.versions.get(store.active_version_id or "")
        if active is None:
            info = "No version yet -- one is created the first time you record a decision."
            freeze_disabled = True
        else:
            bits = [f"parent: {active.parent}"]
            if active.created_at:
                bits.append(f"created {active.created_at}")
            if active.author:
                bits.append(f"by {active.author}")
            if active.note:
                bits.append(active.note)
            if active.frozen:
                bits.append("FROZEN" + (f" -- exported to {active.exported_to}" if active.exported_to else ""))
            info = "  ·  ".join(bits)
            freeze_disabled = active.frozen
        version_select = (h.select(options, store.active_version_id or "", change=self._c("version"),
                                   label="Active version")
                          if options else h.select([("", "")], "", label="Active version", disabled=True))
        controls = (f'<div class="row">{version_select}'
                    f'{h.button("New version", act_attr=self._a("new_version"), kind="primary", icon_name="plus", small=True)}'
                    f'{h.button("Freeze", act_attr=self._a("freeze"), icon_name="save", small=True, disabled=freeze_disabled)}'
                    f'</div><div class="muted small">{h.e(info)}</div>')

        counts = store.counts()
        reviewed = sum(counts.values())
        pills = count_pill("reviewed", reviewed, "acc") + "".join(
            count_pill(STATUS_LABELS[status].lower(), counts.get(status, 0), STATUS_TONE[status])
            for status in REVIEW_STATUSES)
        reasons = store.reason_counts()
        reason_pills = "".join(count_pill(REASON_LABELS.get(reason, reason), count)
                               for reason, count in list(reasons.items())[:10]) or \
            '<span class="muted small"><i>No reasons recorded yet.</i></span>'
        batches = store.batches()
        batch_rows = "".join(
            f'<tr><td>{count:,} × {h.e(STATUS_LABELS.get(status, status))}</td>'
            f'<td class="muted">{h.e(timestamp)}</td>'
            f'<td>{h.button("Undo", act_attr=self._a("undo_batch", batch=batch_id), icon_name="rotate", small=True, kind="ghost")}</td></tr>'
            for batch_id, status, count, timestamp in batches[:6])
        batches_html = (f'<table class="tbl compact cl-batches"><tbody>{batch_rows}</tbody></table>' if batch_rows else
                        '<span class="muted small"><i>No bulk actions in this version.</i></span>')

        # -- overlay --
        pinned = state.review_version_id if state.review_version_id in names else None
        overlay_select = h.select(
            [(ACTIVE_VERSION, "Active version (follows the picker above)")] + [(n, f"Version {n}") for n in names],
            pinned or ACTIVE_VERSION, change=self._c("overlay_version"), label="Scored against",
            disabled=not names)
        effective = state.overlay_version_id()
        if not state.use_reviewed_labels:
            overlay_note = "Off -- every metric uses the labels exactly as they were loaded."
        elif effective:
            ov_counts = store.counts(effective)
            overlay_note = (f"On -- every metric in the app is computed through version '{effective}': "
                            f"{ov_counts.get('relabel', 0):,} relabelled, {ov_counts.get('exclude', 0):,} dropped as excluded.")
        else:
            overlay_note = "On, but there is no version to resolve through yet."
        overlay = (f'<div class="row">{h.checkbox("Use reviewed labels in metrics", bool(state.use_reviewed_labels), change=self._c("overlay"), cls="sw")}'
                   f'{overlay_select}</div>'
                   f'<div class="muted small"><i>Every metric in the app follows this switch. Pin a version to keep scoring '
                   f'against it while you carry on reviewing into another.</i></div>'
                   f'<div class="muted small"><i>{h.e(overlay_note)}</i></div>')
        text, ok = self.status
        status = f'<div class="small {"c-ok" if ok else "c-err"}">{h.e(text)}</div>' if text else ""
        body = (f'{controls}<div class="row-wrap">{pills}</div>'
                f'<div class="cl-sub">Reasons</div><div class="row-wrap">{reason_pills}</div>'
                f'<div class="cl-sub">Bulk actions</div>{batches_html}'
                f'<hr class="cl-hr">{overlay}{status}')
        return h.section("Review version", f'<div class="stack cl-tight">{body}</div>', manual="review-versions",
                         sub="Decisions are recorded against a version, which inherits its parent's and stores only what it changes.")

    def _diff_card(self) -> str:
        store = self.store()
        names = list(store.versions)
        opts = [(n, n) for n in names]
        pickers = (f'<div class="row">{h.select(opts, self.diff_a or "", change=self._c("diff_a"), label="From", disabled=not names)}'
                   f'{h.icon("arrow-right", 16, "muted")}'
                   f'{h.select(opts, self.diff_b or "", change=self._c("diff_b"), label="To", disabled=not names)}'
                   f'{h.button("Compare versions", act_attr=self._a("diff"), icon_name="columns", small=True, disabled=len(names) < 2)}'
                   f'</div>')
        return h.section("What changed between two versions", f'<div class="stack cl-tight">{pickers}{self._diff_html()}</div>',
                         manual="review-versions", sub="Which decisions differ between two passes.")

    def _diff_html(self) -> str:
        diff = self.diff_result
        if diff is None:
            return ""
        if isinstance(diff, tuple):
            return h.banner(diff[0], diff[1])
        if diff.total_changes() == 0:
            return h.banner(f"'{diff.version_b}' resolves to exactly the same decisions as "
                            f"'{diff.version_a}' ({diff.unchanged:,} in common).", "ok")
        parts = [
            '<div class="row-wrap">'
            + count_pill("newly decided", len(diff.added), "ok")
            + count_pill("verdict changed", len(diff.changed), "warn")
            + count_pill("decision withdrawn", len(diff.removed), "err")
            + count_pill("unchanged", diff.unchanged) + '</div>',
            '<div class="muted small"><i>Resolved decisions, not stored ones: an inherited decision counts as present, '
            'because what matters is the verdict a training run would read.</i></div>',
        ]
        transitions = diff.transitions()
        if transitions:
            rows = "".join(
                f'<tr><td class="t-{STATUS_TONE.get(before, "")}">{h.e(self._status_label(before))}</td>'
                f'<td class="t-{STATUS_TONE.get(after, "")}">{h.e(self._status_label(after))}</td>'
                f'<td class="num">{count:,}</td></tr>'
                for (before, after), count in sorted(transitions.items(), key=lambda kv: -kv[1]))
            parts.append(f'<table class="tbl compact cl-narrow"><thead><tr><th>From</th><th>To</th>'
                         f'<th class="num">Crops</th></tr></thead><tbody>{rows}</tbody></table>')
        sample = [
            *[(uid, None, d, "newly decided") for uid, d in diff.added[:8]],
            *[(uid, a, b, "changed") for uid, a, b in diff.changed[:8]],
            *[(uid, d, None, "withdrawn") for uid, d in diff.removed[:8]],
        ]
        if sample:
            parts.append('<div class="cl-sub">Examples</div>')
            parts.append('<div class="mono small muted cl-examples">' + "".join(
                f'<div>{h.e(uid)}  ·  {h.e(kind)}: {h.e(self._verdict_text(before))} -&gt; {h.e(self._verdict_text(after))}</div>'
                for uid, before, after, kind in sample) + '</div>')
        return "".join(parts)

    def _export_card(self) -> str:
        store = self.store()
        checks = "".join(
            h.checkbox(f"{vid} — {store.n_decisions(vid):,} own decisions, {len(store.effective(vid)):,} effective",
                       bool(self.export_checks.get(vid)), change=self._c("export_check", vid=vid))
            for vid in store.versions) or '<span class="muted small"><i>Nothing to export yet.</i></span>'
        dir_row = (f'<div class="row-nw cl-dir">{h.icon("folder", 15, "muted")}'
                   f'{h.text_input(self.export_dir, label="Destination folder", change=self._c("export_dir"), mono=True, cls="grow")}'
                   f'{h.button(icon_name="folder", act_attr=self._a("browse"), title="Browse", small=True)}</div>')
        opts = (f'<div class="row">{h.text_input(self.export_name, label="File name", change=self._c("export_name"), width="260px", mono=True)}'
                f'{h.checkbox("Include the reason column", self.include_reason, change=self._c("include_reason"))}'
                f'{h.checkbox("Drop excluded rows (not round-trippable)", self.drop_excluded, change=self._c("drop_excluded"))}</div>')
        buttons = (f'<div class="row">{h.button("Export dataset", act_attr=self._a("export"), kind="primary", icon_name="download", small=True)}'
                   f'{h.button("Export excluded uuids", act_attr=self._a("export_uuids"), icon_name="list", small=True)}</div>')
        lines, ok = self.export_status
        status = "".join(f'<div class="small {"c-ok" if ok else "c-err"} cl-sel">{h.e(line)}</div>' for line in lines)
        body = f'<div class="col cl-checks">{checks}</div>{dir_row}{opts}{buttons}{status}'
        return h.section("Export for training", f'<div class="stack cl-tight">{body}</div>', manual="export-for-training",
                         sub="Your dataset CSV, back out unchanged, plus label_review_<version> and review_status_<version>.")

    # ---- actions ----
    def handle(self, op: str, args: dict) -> None:
        fn = getattr(self, f"_op_{op}", None)
        if fn is not None:
            fn(args)

    def _op_version(self, args):
        store = self.store()
        value = args.get("value")
        if store is None or not value or value not in store.versions:
            return
        store.active_version_id = value
        # Only matters while the overlay is on, but it's the same call
        # either way and the merge cache checks the version in its stamp.
        self.s.state._invalidate_merged_cache()
        self.s.notify_review()

    def _op_new_version(self, args):
        if self.store() is None:
            return
        self.s.open_dialog(NewVersionDialog(self))

    def _op_freeze(self, args):
        store = self.store()
        if store is None or not store.active_version_id:
            return
        store.freeze(store.active_version_id)
        store.save()
        self.status = (f"Version '{store.active_version_id}' is frozen. Create a new one to keep reviewing.", True)
        self.s.notify_review()

    def _op_overlay(self, args):
        self.s.state.use_reviewed_labels = bool(args.get("value"))
        self._apply_overlay_change()

    def _op_overlay_version(self, args):
        """Pin the version the metrics resolve through -- separate from the
        active version on purpose: one model scored against v1's labels
        while the review carries on into v2."""
        value = args.get("value")
        store = self.store()
        names = list(store.versions) if store is not None else []
        self.s.state.review_version_id = None if value == ACTIVE_VERSION or value not in names else value
        self._apply_overlay_change()

    def _apply_overlay_change(self) -> None:
        self.s.state._invalidate_merged_cache()
        # This one *is* a data change: every metric, curve and confusion
        # matrix in the app now answers a different question.
        self.s.notify_change()

    def _op_undo_batch(self, args):
        store = self.store()
        batch_id = str(args.get("batch", ""))
        if store is None or not batch_id:
            return
        removed = store.undo_batch(batch_id)
        store.save()
        self.status = (f"Undid {removed:,} decision(s) from batch {batch_id}.", True)
        self.s.notify_review()

    def _op_diff_a(self, args):
        self.diff_a = args.get("value") or None

    def _op_diff_b(self, args):
        self.diff_b = args.get("value") or None

    def _op_diff(self, args):
        store = self.store()
        if store is None or not self.diff_a or not self.diff_b:
            return
        if self.diff_a == self.diff_b:
            self.diff_result = ("Pick two different versions.", "info")
            return
        self.diff_result = diff_versions(store, self.diff_a, self.diff_b)

    def _op_export_check(self, args):
        vid = str(args.get("vid", ""))
        self.export_checks[vid] = bool(args.get("value"))

    def _op_export_dir(self, args):
        self.export_dir = str(args.get("value", ""))

    def _op_export_name(self, args):
        self.export_name = str(args.get("value", ""))

    def _op_include_reason(self, args):
        self.include_reason = bool(args.get("value"))

    def _op_drop_excluded(self, args):
        self.drop_excluded = bool(args.get("value"))

    def _op_browse(self, args):
        def picked(path: str) -> None:
            self.export_dir = path
            self.s.mark(self.target)
        self.s.open_dialog(FileBrowserDialog(picked, start_path=self.export_dir or "", dirs_only=True,
                                             title="Select a destination folder"))

    def _selected_export_versions(self) -> list:
        store = self.store()
        if store is None:
            return []
        return [vid for vid in store.versions if self.export_checks.get(vid)]

    def _op_export(self, args):
        store, dataset = self.store(), self.s.state.dataset
        if store is None or dataset is None or dataset.csv_path is None:
            return
        versions = self._selected_export_versions()
        if not versions:
            self.export_status = (["Pick at least one version to export."], False)
            return
        directory = (self.export_dir or "").strip() or str(Path(dataset.csv_path).parent)
        name = (self.export_name or "").strip() or self._default_export_name()
        try:
            result = export_enriched_dataset(
                store, dataset.csv_path, dataset.columns.uuid_col, dataset.columns.label_col,
                versions, Path(directory) / name,
                include_reason=self.include_reason,
                drop_excluded=self.drop_excluded,
            )
        except (OSError, ValueError) as ex:
            _log(f"export failed: {ex}")
            traceback.print_exc()
            self.export_status = ([f"Export failed: {ex}"], False)
            return
        lines = [
            f"Wrote {result.rows:,} rows to {saved_path_message(self.s, result.path)}",
            f"Added columns: {', '.join(result.columns)}",
            f"{result.relabelled:,} relabelled, {result.excluded:,} excluded.",
            f"Version(s) {', '.join(versions)} are now frozen -- create a new one to keep reviewing.",
        ] + list(result.notes)
        self.export_status = (lines, True)
        self.s.notify_review()

    def _op_export_uuids(self, args):
        store, dataset = self.store(), self.s.state.dataset
        if store is None or dataset is None or not store.active_version_id:
            return
        directory = (self.export_dir or "").strip() or str(Path(dataset.csv_path).parent)
        dest = Path(directory) / f"{Path(dataset.csv_path).stem}_excluded_{store.active_version_id}.txt"
        try:
            result = export_excluded_uuids(store, store.active_version_id, dest)
        except OSError as ex:
            self.export_status = ([f"Export failed: {ex}"], False)
            return
        self.export_status = ([f"Wrote {result.rows:,} excluded uuid(s) to {saved_path_message(self.s, result.path)}"], True)
