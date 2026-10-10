"""The review version manager and the export that consumes it -- the two
halves of "what have I decided, and how do I train on it".

Kept out of cleaning_view.py because neither is about *finding* suspicious
crops: this is bookkeeping over decisions already made, and it is the part
that has to be right for a training run to be reproducible six months
later.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import flet as ft

from lookin.core.review import (
    NO_PARENT, REVIEW_STATUSES, STATUS_UNREVIEWED, diff_versions,
    export_enriched_dataset, export_excluded_uuids,
)
from lookin.ui.components import (
    ACCENT, BackgroundRebuild, card, diverging_color, info_banner, saved_path_message,
    section_title,
)
from lookin.ui.review_widgets import REASON_LABELS, STATUS_COLORS, STATUS_LABELS


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


# The "no pin" entry of the overlay version picker. A key, compared in code.
ACTIVE_VERSION = "__active__"


def _count_pill(label: str, value, color=None) -> ft.Control:
    color = color or ft.Colors.ON_SURFACE_VARIANT
    return ft.Container(
        content=ft.Row([
            ft.Text(f"{value:,}" if isinstance(value, int) else str(value), size=13,
                    weight=ft.FontWeight.BOLD, color=color),
            ft.Text(label, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
        ], spacing=6, tight=True),
        padding=ft.padding.symmetric(horizontal=10, vertical=5),
        border_radius=8,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
    )


class ReviewPanel:
    """Version selector + counts + batch undo + export."""

    def __init__(self, ctx, server_browser=None):
        self.ctx = ctx
        self.server_browser = server_browser

        self.version_dropdown = ft.Dropdown(label="Active version", width=240, border_radius=8,
                                            on_change=self._on_version_change)
        self.new_version_btn = ft.FilledButton("New version", icon=ft.Icons.ADD,
                                               on_click=self._open_new_version, style=ft.ButtonStyle(bgcolor=ACCENT))
        self.freeze_btn = ft.OutlinedButton("Freeze", icon=ft.Icons.LOCK_OUTLINE, on_click=self._on_freeze)
        self.overlay_switch = ft.Switch(label="Use reviewed labels in metrics", value=False,
                                        on_change=self._on_overlay_toggle)
        # Which version the metrics are computed through, independently of
        # which one you are recording decisions into.
        self.overlay_version_container = ft.Container()
        self.overlay_note = ft.Text(size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)

        self.version_info = ft.Text(size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.counts_row = ft.Row(spacing=8, wrap=True)
        self.reasons_row = ft.Row(spacing=8, wrap=True)
        self.batches_column = ft.Column(spacing=4)
        self.status_text = ft.Text(size=12)

        # -- new-version dialog --
        self.name_field = ft.TextField(label="Version name", width=200, border_radius=8, dense=True)
        self.parent_dropdown = ft.Dropdown(label="Inherits from", width=200, border_radius=8, dense=True)
        self.note_field = ft.TextField(label="What this pass is for", width=420, border_radius=8, dense=True)
        self.new_version_error = ft.Text(size=11, color=ft.Colors.RED)
        self.new_version_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("New review version"),
            content=ft.Column([
                ft.Text("A version inherits every decision of its parent and stores only what it changes.",
                        size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Row([self.name_field, self.parent_dropdown], spacing=12),
                self.note_field,
                self.new_version_error,
            ], spacing=12, tight=True, width=460),
            actions=[
                ft.TextButton("Create", on_click=self._on_create_version),
                ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(self.new_version_dialog)),
            ],
        )

        # -- export --
        self.export_versions_column = ft.Column(spacing=2)
        self.export_version_checks: dict = {}
        self.export_dir_field = ft.TextField(label="Destination folder", expand=True, border_radius=8,
                                             prefix_icon=ft.Icons.FOLDER_OPEN)
        self.export_name_field = ft.TextField(label="File name", width=280, border_radius=8)
        self.include_reason_check = ft.Checkbox(label="Include the reason column", value=True)
        self.drop_excluded_check = ft.Checkbox(label="Drop excluded rows (not round-trippable)", value=False)
        self.export_btn = ft.FilledButton("Export dataset", icon=ft.Icons.DOWNLOAD_OUTLINED,
                                          on_click=self._on_export, style=ft.ButtonStyle(bgcolor=ACCENT))
        self.export_uuids_btn = ft.OutlinedButton("Export excluded uuids", icon=ft.Icons.LIST_ALT_OUTLINED,
                                                  on_click=self._on_export_uuids)
        self.export_status = ft.Column(spacing=4)

        # -- version diff --
        self.diff_a_container = ft.Container()
        self.diff_b_container = ft.Container()
        self.diff_a_value: str | None = None
        self.diff_b_value: str | None = None
        self.diff_result = ft.Column(spacing=8)
        self.diff_btn = ft.OutlinedButton("Compare versions", icon=ft.Icons.DIFFERENCE_OUTLINED,
                                          on_click=self._on_diff_click)

        self.banner_area = ft.Container()
        self.version_card = ft.Container()
        self.diff_card = ft.Container()
        self.export_card = ft.Container()
        self.root = ft.Column(spacing=14)

    # ---- version diff (B4) ----
    def _version_picker(self, label: str, attr: str) -> ft.Dropdown:
        """Rebuilt, never mutated -- see CompareView._rebuild_dropdowns for
        why reassigning a mounted Dropdown's options is not safe here."""
        store = self._store()
        names = list(store.versions) if store is not None else []
        value = getattr(self, attr)
        if value not in names:
            value = names[-1] if names else None
            setattr(self, attr, value)

        def on_change(e, attr=attr):
            setattr(self, attr, e.control.value)

        return ft.Dropdown(
            label=label, width=170, border_radius=8, dense=True, value=value,
            options=[ft.dropdown.Option(n, n) for n in names], on_change=on_change,
        )

    def _rebuild_diff_pickers(self) -> None:
        store = self._store()
        names = list(store.versions) if store is not None else []
        # Default to the two most recent, which is the pair anyone actually
        # wants: "what did my latest pass change".
        if self.diff_a_value not in names:
            self.diff_a_value = names[-2] if len(names) > 1 else (names[0] if names else None)
        if self.diff_b_value not in names:
            self.diff_b_value = names[-1] if names else None
        self.diff_a_container.content = self._version_picker("From", "diff_a_value")
        self.diff_b_container.content = self._version_picker("To", "diff_b_value")
        self.diff_btn.disabled = len(names) < 2

    def _on_diff_click(self, e):
        store = self._store()
        if store is None or not self.diff_a_value or not self.diff_b_value:
            return
        if self.diff_a_value == self.diff_b_value:
            self.diff_result.controls = [
                info_banner("Pick two different versions.", ft.Icons.INFO_OUTLINE)]
            self._safe_update(self.diff_result)
            return

        diff = diff_versions(store, self.diff_a_value, self.diff_b_value)
        self.diff_result.controls = self._render_diff(diff)
        self._safe_update(self.diff_result)

    def _render_diff(self, diff) -> list:
        if diff.total_changes() == 0:
            return [info_banner(
                f"'{diff.version_b}' resolves to exactly the same decisions as "
                f"'{diff.version_a}' ({diff.unchanged:,} in common).", ft.Icons.CHECK_CIRCLE_OUTLINE)]

        controls = [
            ft.Row([
                _count_pill("newly decided", len(diff.added), ft.Colors.GREEN_400),
                _count_pill("verdict changed", len(diff.changed), ft.Colors.ORANGE_400),
                _count_pill("decision withdrawn", len(diff.removed), ft.Colors.RED_400),
                _count_pill("unchanged", diff.unchanged),
            ], spacing=8, wrap=True),
            ft.Text(
                "Resolved decisions, not stored ones: an inherited decision counts as present, "
                "because what matters is the verdict a training run would read.",
                size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
        ]

        transitions = diff.transitions()
        if transitions:
            rows = [ft.Row([
                ft.Container(ft.Text("From", size=11, weight=ft.FontWeight.BOLD), width=110),
                ft.Container(ft.Text("To", size=11, weight=ft.FontWeight.BOLD), width=110),
                ft.Container(ft.Text("Crops", size=11, weight=ft.FontWeight.BOLD), width=70),
            ], spacing=4)]
            for (before, after), count in sorted(transitions.items(), key=lambda kv: -kv[1]):
                rows.append(ft.Row([
                    ft.Container(ft.Text(self._status_label(before), size=12,
                                         color=STATUS_COLORS.get(before)), width=110),
                    ft.Container(ft.Text(self._status_label(after), size=12,
                                         color=STATUS_COLORS.get(after)), width=110),
                    ft.Container(ft.Text(f"{count:,}", size=12), width=70),
                ], spacing=4))
            controls.append(ft.Column(rows, spacing=4))

        sample = [
            *[(uid, None, d, "newly decided") for uid, d in diff.added[:8]],
            *[(uid, a, b, "changed") for uid, a, b in diff.changed[:8]],
            *[(uid, d, None, "withdrawn") for uid, d in diff.removed[:8]],
        ]
        if sample:
            controls.append(ft.Text("Examples", size=12, weight=ft.FontWeight.BOLD))
            for uid, before, after, kind in sample:
                controls.append(ft.Text(
                    f"{uid}  ·  {kind}: "
                    f"{self._verdict_text(before)} -> {self._verdict_text(after)}",
                    size=11, font_family="monospace", color=ft.Colors.ON_SURFACE_VARIANT))
        return controls

    @staticmethod
    def _status_label(status: str) -> str:
        return STATUS_LABELS.get(status, "not reviewed" if status == STATUS_UNREVIEWED else status)

    def _verdict_text(self, decision) -> str:
        if decision is None:
            return "not reviewed"
        label = self._status_label(decision.status)
        return f"{label} -> {decision.new_label}" if decision.new_label else label

    @staticmethod
    def _safe_update(*controls) -> None:
        for control in controls:
            if control is not None and control.page:
                control.update()

    # ---- helpers ----
    def _store(self):
        return getattr(self.ctx.state, "review", None)

    def _set_status(self, message: str, ok: bool = True) -> None:
        self.status_text.value = message
        self.status_text.color = ft.Colors.GREEN if ok else ft.Colors.RED
        if self.status_text.page:
            self.status_text.update()

    # ---- versions ----
    def _on_version_change(self, e):
        store = self._store()
        if store is None or not self.version_dropdown.value:
            return
        store.active_version_id = self.version_dropdown.value
        # Only matters while the overlay is on, but it's the same call
        # either way and the merge cache checks the version in its stamp.
        self.ctx.state._invalidate_merged_cache()
        self.refresh()
        self.ctx.notify_review()

    def _open_new_version(self, e):
        store = self._store()
        if store is None:
            return
        self.name_field.value = store.next_version_id()
        options = [ft.dropdown.Option(NO_PARENT, "nothing (independent pass)")] + [
            ft.dropdown.Option(vid) for vid in store.versions
        ]
        self.parent_dropdown.options = options
        self.parent_dropdown.value = store.active_version_id or NO_PARENT
        self.note_field.value = ""
        self.new_version_error.value = ""
        self.ctx.page.open(self.new_version_dialog)

    def _on_create_version(self, e):
        store = self._store()
        if store is None:
            return
        try:
            store.create_version(
                version_id=(self.name_field.value or "").strip(),
                parent=self.parent_dropdown.value or NO_PARENT,
                note=(self.note_field.value or "").strip(),
            )
            store.save()
        except ValueError as ex:
            self.new_version_error.value = str(ex)
            if self.new_version_error.page:
                self.new_version_error.update()
            return
        self.ctx.page.close(self.new_version_dialog)
        self.refresh()
        self.ctx.notify_review()

    def _on_freeze(self, e):
        store = self._store()
        if store is None or not store.active_version_id:
            return
        store.freeze(store.active_version_id)
        store.save()
        self._set_status(f"Version '{store.active_version_id}' is frozen. Create a new one to keep reviewing.")
        self.refresh()
        self.ctx.notify_review()

    def _on_overlay_toggle(self, e):
        self.ctx.state.use_reviewed_labels = bool(self.overlay_switch.value)
        self._render_overlay_controls()
        self._safe_update(self.overlay_note, self.overlay_version_container)
        self._apply_overlay_change()

    def _on_overlay_version_change(self, e):
        """Pin the version the metrics resolve through.

        Separate from the active version on purpose: you record decisions
        into the version you are working on, but you may well want every
        model scored against v1's labels while you build v2 -- comparing
        one model against v1 and another against v3 and calling the
        difference a model improvement is the mistake this prevents.
        """
        value = e.control.value
        self.ctx.state.review_version_id = None if value == ACTIVE_VERSION else value
        self._render_overlay_controls()
        # Pushed here rather than left to the notify below: the Cleaning tab
        # does not re-render this panel on a data change, so without this
        # the picker reads "v1" while the line under it still describes v2.
        self._safe_update(self.overlay_note, self.overlay_version_container)
        self._apply_overlay_change()

    def _apply_overlay_change(self) -> None:
        self.ctx.state._invalidate_merged_cache()
        # This one *is* a data change: every metric, curve and confusion
        # matrix in the app now answers a different question, so the views
        # that cache them have to know they're stale.
        self.ctx.notify_change()

    def _render_overlay_controls(self) -> None:
        """Rebuild the version picker (never mutate a mounted Dropdown's
        options -- see CompareView._rebuild_dropdowns)."""
        store = self._store()
        names = list(store.versions) if store is not None else []
        state = self.ctx.state
        pinned = state.review_version_id if state.review_version_id in names else None

        self.overlay_version_container.content = ft.Dropdown(
            label="Scored against", width=260, border_radius=8, dense=True,
            value=pinned or ACTIVE_VERSION,
            options=[ft.dropdown.Option(ACTIVE_VERSION, "Active version (follows the picker above)")]
                    + [ft.dropdown.Option(n, f"Version {n}") for n in names],
            on_change=self._on_overlay_version_change,
            disabled=not names,
        )
        effective = state.overlay_version_id()
        if not state.use_reviewed_labels:
            self.overlay_note.value = "Off -- every metric uses the labels exactly as they were loaded."
        elif effective:
            counts = store.counts(effective) if store is not None else {}
            excluded = counts.get("exclude", 0)
            relabelled = counts.get("relabel", 0)
            self.overlay_note.value = (
                f"On -- every metric in the app is computed through version '{effective}': "
                f"{relabelled:,} relabelled, {excluded:,} dropped as excluded."
            )
        else:
            self.overlay_note.value = "On, but there is no version to resolve through yet."

    def _on_undo_batch(self, batch_id: str):
        store = self._store()
        if store is None:
            return
        removed = store.undo_batch(batch_id)
        store.save()
        self._set_status(f"Undid {removed:,} decision(s) from batch {batch_id}.")
        self.refresh()
        self.ctx.notify_review()

    # ---- export ----
    def _default_export_name(self) -> str:
        dataset = self.ctx.state.dataset
        if dataset is None or dataset.csv_path is None:
            return "dataset_reviewed.csv"
        return f"{Path(dataset.csv_path).stem}_reviewed.csv"

    def _on_browse_export_dir(self, e):
        if self.server_browser is None:
            return
        self.server_browser.open(
            lambda path: self._set_export_dir(path),
            start_path=self.export_dir_field.value or "",
            dirs_only=True, title="Select a destination folder",
        )

    def _set_export_dir(self, path: str) -> None:
        self.export_dir_field.value = path
        if self.export_dir_field.page:
            self.export_dir_field.update()

    def _selected_export_versions(self) -> list:
        store = self._store()
        if store is None:
            return []
        return [vid for vid in store.versions if self.export_version_checks.get(vid) is not None
                and self.export_version_checks[vid].value]

    def _on_export(self, e):
        store, dataset = self._store(), self.ctx.state.dataset
        if store is None or dataset is None or dataset.csv_path is None:
            return
        versions = self._selected_export_versions()
        if not versions:
            self._show_export_result("Pick at least one version to export.", ok=False)
            return
        directory = (self.export_dir_field.value or "").strip() or str(Path(dataset.csv_path).parent)
        name = (self.export_name_field.value or "").strip() or self._default_export_name()
        try:
            result = export_enriched_dataset(
                store, dataset.csv_path, dataset.columns.uuid_col, dataset.columns.label_col,
                versions, Path(directory) / name,
                include_reason=self.include_reason_check.value,
                drop_excluded=self.drop_excluded_check.value,
            )
        except (OSError, ValueError) as ex:
            _log(f"export failed: {ex}")
            traceback.print_exc()
            self._show_export_result(f"Export failed: {ex}", ok=False)
            return
        lines = [
            f"Wrote {result.rows:,} rows to {saved_path_message(self.ctx, result.path)}",
            f"Added columns: {', '.join(result.columns)}",
            f"{result.relabelled:,} relabelled, {result.excluded:,} excluded.",
            f"Version(s) {', '.join(versions)} are now frozen -- create a new one to keep reviewing.",
        ] + result.notes
        self._show_export_result("\n".join(lines), ok=True)
        self.refresh()
        self.ctx.notify_review()

    def _on_export_uuids(self, e):
        store, dataset = self._store(), self.ctx.state.dataset
        if store is None or dataset is None or not store.active_version_id:
            return
        directory = (self.export_dir_field.value or "").strip() or str(Path(dataset.csv_path).parent)
        dest = Path(directory) / f"{Path(dataset.csv_path).stem}_excluded_{store.active_version_id}.txt"
        try:
            result = export_excluded_uuids(store, store.active_version_id, dest)
        except OSError as ex:
            self._show_export_result(f"Export failed: {ex}", ok=False)
            return
        self._show_export_result(
            f"Wrote {result.rows:,} excluded uuid(s) to "
            f"{saved_path_message(self.ctx, result.path)}", ok=True)

    def _show_export_result(self, message: str, ok: bool) -> None:
        self.export_status.controls = [
            ft.Text(line, size=12, color=ft.Colors.GREEN if ok else ft.Colors.RED, selectable=True)
            for line in message.split("\n")
        ]
        if self.export_status.page:
            self.export_status.update()

    # ---- rendering ----
    def _render_versions(self) -> None:
        store = self._store()
        if store is None:
            return
        names = list(store.versions)
        self.version_dropdown.options = [
            ft.dropdown.Option(vid, f"{vid}  ({store.n_decisions(vid):,})" + (" · frozen" if store.versions[vid].frozen else ""))
            for vid in names
        ]
        if store.active_version_id not in names:
            store.active_version_id = store.resume_version_id() or (names[-1] if names else None)
        self.version_dropdown.value = store.active_version_id

        active = store.versions.get(store.active_version_id or "")
        if active is None:
            self.version_info.value = "No version yet -- one is created the first time you record a decision."
            self.freeze_btn.disabled = True
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
            self.version_info.value = "  ·  ".join(bits)
            self.freeze_btn.disabled = active.frozen

        counts = store.counts()
        reviewed = sum(counts.values())
        self.counts_row.controls = [_count_pill("reviewed", reviewed, ACCENT)] + [
            _count_pill(STATUS_LABELS[status].lower(), counts.get(status, 0), STATUS_COLORS[status])
            for status in REVIEW_STATUSES
        ]
        reasons = store.reason_counts()
        self.reasons_row.controls = (
            [_count_pill(REASON_LABELS.get(reason, reason), count) for reason, count in list(reasons.items())[:10]]
            or [ft.Text("No reasons recorded yet.", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)]
        )

        batches = store.batches()
        self.batches_column.controls = [
            ft.Row([
                ft.Text(f"{count:,} × {STATUS_LABELS.get(status, status)}", size=12),
                ft.Text(timestamp, size=11, color=ft.Colors.ON_SURFACE_VARIANT, expand=True),
                ft.TextButton("Undo", icon=ft.Icons.UNDO, on_click=lambda e, b=batch_id: self._on_undo_batch(b)),
            ], spacing=10)
            for batch_id, status, count, timestamp in batches[:6]
        ] or [ft.Text("No bulk actions in this version.", size=11, italic=True,
                      color=ft.Colors.ON_SURFACE_VARIANT)]

    def _render_export(self) -> None:
        store, dataset = self._store(), self.ctx.state.dataset
        if store is None or dataset is None:
            return
        previous = {vid: check.value for vid, check in self.export_version_checks.items()}
        self.export_version_checks = {}
        rows = []
        for vid in store.versions:
            check = ft.Checkbox(
                label=f"{vid} — {store.n_decisions(vid):,} own decisions, "
                      f"{len(store.effective(vid)):,} effective",
                value=previous.get(vid, vid == store.active_version_id),
            )
            self.export_version_checks[vid] = check
            rows.append(check)
        self.export_versions_column.controls = rows or [
            ft.Text("Nothing to export yet.", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
        ]
        if not self.export_dir_field.value and dataset.csv_path is not None:
            self.export_dir_field.value = str(Path(dataset.csv_path).parent)
        if not self.export_name_field.value:
            self.export_name_field.value = self._default_export_name()

    def refresh(self) -> None:
        has_dataset = self.ctx.state.dataset is not None and self._store() is not None
        self.banner_area.content = None if has_dataset else info_banner(
            "Load a dataset on the Data tab to start reviewing -- decisions are stored next to its CSV."
        )
        self.version_card.visible = has_dataset
        self.diff_card.visible = has_dataset
        self.export_card.visible = has_dataset
        if has_dataset:
            self._render_versions()
            self._rebuild_diff_pickers()
            self._render_export()
        self.overlay_switch.value = bool(self.ctx.state.use_reviewed_labels)
        if has_dataset:
            self._render_overlay_controls()
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        self.version_card = card(ft.Column([
            section_title("Review version", icon=ft.Icons.HISTORY_TOGGLE_OFF, ctx=self.ctx, manual="review-versions"),
            ft.Text(
                "Decisions are recorded against a version, which inherits its parent's and stores "
                "only what it changes.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([self.version_dropdown, self.new_version_btn, self.freeze_btn], spacing=12, wrap=True),
            self.version_info,
            self.counts_row,
            ft.Text("Reasons", size=12, weight=ft.FontWeight.BOLD),
            self.reasons_row,
            ft.Text("Bulk actions", size=12, weight=ft.FontWeight.BOLD),
            self.batches_column,
            ft.Divider(height=1),
            ft.Row([self.overlay_switch, self.overlay_version_container], spacing=16, wrap=True,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(
                "Every metric in the app follows this switch. Pin a version to keep scoring against "
                "it while you carry on reviewing into another.",
                size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self.overlay_note,
            self.status_text,
        ], spacing=10))

        self.diff_card = card(ft.Column([
            section_title("What changed between two versions", icon=ft.Icons.DIFFERENCE_OUTLINED, ctx=self.ctx, manual="review-versions"),
            ft.Text(
                "Which decisions differ between two passes.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([self.diff_a_container, ft.Icon(ft.Icons.ARROW_FORWARD, size=16),
                    self.diff_b_container, self.diff_btn],
                   spacing=12, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.diff_result,
        ], spacing=10))

        self.export_card = card(ft.Column([
            section_title("Export for training", icon=ft.Icons.IOS_SHARE, ctx=self.ctx, manual="export-for-training"),
            ft.Text(
                "Your dataset CSV, back out unchanged, plus label_review_<version> and "
                "review_status_<version>.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self.export_versions_column,
            ft.Row([self.export_dir_field,
                    ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=self._on_browse_export_dir)],
                   spacing=6),
            ft.Row([self.export_name_field, self.include_reason_check, self.drop_excluded_check],
                   spacing=14, wrap=True),
            ft.Row([self.export_btn, self.export_uuids_btn], spacing=12),
            self.export_status,
        ], spacing=10))

        self.root.controls = [self.banner_area, self.version_card, self.diff_card,
                              self.export_card]
        self.refresh()
        return self.root
