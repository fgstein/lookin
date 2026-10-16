"""Tab 3: filterable image gallery, a mosaic that loads as it is scrolled.

Filters: checkboxes for label/categorical/binary columns, range sliders for
numeric columns. Nothing is filtered by default -- the user adds the
specific filters they want via the picker at the top, and can remove any of
them individually. Optionally color each thumbnail's border by whether a
chosen experiment's prediction was correct, and filter by predicted label
too (used by the confusion-matrix drill-down). Clicking a thumbnail opens a
full-size detail dialog with all of its data.

The mosaic has no pages: it opens with enough rows to fill the window and
appends the next rows whenever the scroll position nears the bottom (see
_on_scroll), so the first crops show immediately and the rest are a scroll
away rather than a click on "next page" away. The user sets how many crops
sit on a row; how many rows there are is the size of the filtered set.
"""
from __future__ import annotations

import math
import threading

import flet as ft
import numpy as np
import pandas as pd

from lookin.core.class_focus import FOCUS_MAX
from lookin.ui.class_matrix import ClassPicker, picker_checks
from lookin.ui.components import crop_display, file_to_full_base64, section_title, info_banner, card, ACCENT, BackgroundRebuild, StaleBanner
from lookin.ui.carousel import Carousel

MAX_COLS = 12
DEFAULT_COLS = 6
NONE_EXPERIMENT = "__none__"
# Height a row takes beyond its thumbnail: the caption under it, the
# cell's padding and border, and the spacing to the next row.
ROW_EXTRA_HEIGHT = 44
# Load the next rows once the bottom is less than this many screens away.
PREFETCH_SCREENS = 1.0
SCROLL_EVENT_INTERVAL = 120  # ms between scroll reports, at most


class GalleryView:
    def __init__(self, ctx):
        self.ctx = ctx
        self.active_filter_cols: list[str] = []
        self.filter_controls: dict = {}  # col -> {"type": ..., ...}
        self.predicted_label_checks: dict = {}
        # The filtered (and possibly shuffled) frame the mosaic draws from,
        # the records of its first `len(_current_records)` rows -- the ones
        # in the grid so far -- and the number of crops per row.
        self._filtered: pd.DataFrame | None = None
        self._current_records: list = []
        self._cols_n = DEFAULT_COLS
        self._load_lock = threading.Lock()
        self._dirty = False
        self.progress_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_banner = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2), self.progress_text],
            spacing=10, visible=False,
        )
        self.stale_banner = StaleBanner(self._on_update_click)
        self.rebuild_job = BackgroundRebuild(ctx, "Gallery")

        self.cols_field = ft.TextField(label="images per row", value=str(DEFAULT_COLS), width=160, border_radius=8)
        self.grid_shape_label = ft.Text(f"{DEFAULT_COLS} per row", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.settings_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Mosaic settings"),
            content=ft.Row([self.cols_field], spacing=12),
            actions=[
                ft.TextButton("Apply", on_click=self._on_apply_settings),
                ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.settings_dialog)),
            ],
        )
        self.experiment_dropdown = ft.Dropdown(label="Color by experiment prediction", width=280, on_change=self._on_experiment_change)
        # 0 shows the crops in the dataset's own order; any other value is
        # the seed of a shuffle. A seed rather than a stored permutation so
        # that paging keeps one order, a snapshot reproduces it, and a
        # filter change shuffles the new set without a second click.
        self.shuffle_seed = 0
        self.shuffle_switch = ft.Switch(label="Shuffle", value=False, on_change=self._on_shuffle_toggle)
        self.reshuffle_btn = ft.IconButton(ft.Icons.SHUFFLE, tooltip="Another random order",
                                           on_click=self._on_reshuffle, visible=False)

        self.add_filter_dropdown = ft.Dropdown(label="Add a filter for column...", width=280, border_radius=8)
        self.add_filter_button = ft.FilledButton("Add", icon=ft.Icons.ADD, on_click=self._on_add_filter_click, style=ft.ButtonStyle(bgcolor=ACCENT))
        self.skipped_text = ft.Text("", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)

        self.filters_column = ft.Column(spacing=14)
        self.predicted_filter_container = ft.Column()
        self.grid_area = ft.Column(spacing=4)
        self.count_label = ft.Text()
        # Fallback for a window tall enough that the loaded rows do not
        # overflow it: with nothing to scroll, no scroll event comes.
        self.more_btn = ft.TextButton("Show more", icon=ft.Icons.EXPAND_MORE, on_click=self._load_more)
        self.footer = ft.Container(content=self.more_btn, alignment=ft.alignment.center, visible=False)
        self.right_panel: ft.Column | None = None

        self.detail_title = ft.Text(weight=ft.FontWeight.BOLD)
        self.detail_content = ft.Container()  # width comes from the carousel, which sizes itself to the page
        self.detail_dialog = ft.AlertDialog(
            # modal=False so tapping outside dismisses; on_dismiss then has
            # to sync `open` back to False, which Flet does not do itself --
            # a stale True makes the next page.open() a no-op.
            modal=False,
            title=self.detail_title,
            content=self.detail_content,
            on_dismiss=self._detail_close,
            actions=[ft.TextButton("Close", on_click=self._detail_close)],
        )

        self.root = ft.Container(padding=16)

    # ---- filter panel ----
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

    def _available_filter_columns(self):
        state = self.ctx.state
        if state.dataset is None:
            return []
        cols_cfg = state.dataset.columns
        kinds = state.dataset.column_kinds()
        all_cols = [cols_cfg.label_col] + kinds.categorical + kinds.binary + kinds.numeric
        return [c for c in all_cols if c not in self.active_filter_cols]

    def _rebuild_add_filter_dropdown(self):
        available = self._available_filter_columns()
        self.add_filter_dropdown.options = [ft.dropdown.Option(c) for c in available]
        if self.add_filter_dropdown.value not in available:
            self.add_filter_dropdown.value = available[0] if available else None

    def _on_add_filter_click(self, e):
        col = self.add_filter_dropdown.value
        if not col or col in self.active_filter_cols:
            return
        self.active_filter_cols.append(col)
        self._refresh_in_background()

    def _on_remove_filter_click(self, col: str):
        if col in self.active_filter_cols:
            self.active_filter_cols.remove(col)
        self._refresh_in_background()

    def _build_filter_control(self, col: str, is_label: bool) -> None:
        state = self.ctx.state
        df = state.dataset.df
        kinds = state.dataset.column_kinds()
        is_numeric = (not is_label) and col in kinds.numeric

        header = ft.Row(
            [ft.Text(col, weight=ft.FontWeight.BOLD, size=13),
             ft.IconButton(ft.Icons.CLOSE, icon_size=14, tooltip="Remove this filter",
                            on_click=lambda e, c=col: self._on_remove_filter_click(c))],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        if is_numeric:
            series = df[col].dropna()
            lo, hi = float(series.min()), float(series.max())
            if lo == hi:
                hi = lo + 1.0
            slider = ft.RangeSlider(
                min=lo, max=hi, start_value=lo, end_value=hi, divisions=100,
                label="{value}",
                on_change_end=self._on_filter_change,
            )
            range_text = ft.Text(f"[{lo:.2f}, {hi:.2f}]", size=11)

            def on_change(e, slider=slider, range_text=range_text):
                range_text.value = f"[{slider.start_value:.2f}, {slider.end_value:.2f}]"
                range_text.update()

            slider.on_change = on_change
            self.filter_controls[col] = {"type": "numeric", "slider": slider}
            self.filters_column.controls.append(ft.Column([header, slider, range_text], spacing=4))
        else:
            values = sorted(df[col].dropna().unique().tolist(), key=str)
            if is_label and len(values) > FOCUS_MAX:
                # One checkbox per class stops being a filter past a few
                # dozen; search and pick instead (see ClassPicker).
                picker = ClassPicker(lambda picked: self._on_filter_change(None), limit=None,
                                     label="Find a class to keep")
                value_checks = picker_checks(picker, values)
                body = picker.build()
            else:
                value_checks = {}
                for v in values:
                    value_checks[v] = ft.Checkbox(label=str(v), value=False, on_change=self._on_filter_change)
                body = ft.Row(controls=list(value_checks.values()), wrap=True)
            self.filter_controls[col] = {"type": "categorical", "checks": value_checks}
            self.filters_column.controls.append(ft.Column([header, body], spacing=4))

    def _rebuild_filters(self):
        self.filter_controls.clear()
        self.filters_column.controls.clear()
        state = self.ctx.state
        if state.dataset is None:
            # Reset everything that a previous dataset may have left behind
            # (e.g. after "Clear all data") -- previously this method never
            # ran with a dataset that had gone from set back to None, so
            # this path was untested; without it, the predicted-label
            # filter and "color by experiment" dropdown kept stale values.
            # A Dropdown's displayed text doesn't clear itself just because
            # its value is set to None -- give it an explicit "(none)"
            # option and select that, same as the "dataset loaded but no
            # experiments yet" state below.
            self.experiment_dropdown.options = [ft.dropdown.Option(key=NONE_EXPERIMENT, text="(none)")]
            self.experiment_dropdown.value = NONE_EXPERIMENT
            self.active_filter_cols = []
            self._rebuild_add_filter_dropdown()
            self._rebuild_predicted_filter()
            self._clear_progress()
            return

        cols_cfg = state.dataset.columns
        kinds = state.dataset.column_kinds()
        # A column can disappear between renders (relabeling, a fresh
        # dataset load with different columns) -- drop anything stale
        # instead of erroring on a missing column.
        valid_columns = set(state.dataset.df.columns) | {cols_cfg.label_col}
        self.active_filter_cols = [c for c in self.active_filter_cols if c in valid_columns]

        if not self.active_filter_cols:
            self.filters_column.controls.append(info_banner(
                "No filters applied yet -- pick a column above and click Add."
            ))
        else:
            self._set_progress("Building filters...")
            for col in self.active_filter_cols:
                self._set_progress(f"Building filters... {col}")
                self._build_filter_control(col, is_label=(col == cols_cfg.label_col))

        self.skipped_text.value = (
            f"Not offered above (too many unique values): {', '.join(kinds.high_cardinality)}"
            if kinds.high_cardinality else ""
        )
        self._rebuild_add_filter_dropdown()

        self.experiment_dropdown.options = [ft.dropdown.Option(key=NONE_EXPERIMENT, text="(none)")] + [
            ft.dropdown.Option(name) for name in state.experiments
        ]
        self.experiment_dropdown.value = NONE_EXPERIMENT
        self._rebuild_predicted_filter()
        self._clear_progress()

    def _rebuild_predicted_filter(self):
        self.predicted_label_checks.clear()
        self.predicted_filter_container.controls.clear()
        state = self.ctx.state
        exp_name = self.experiment_dropdown.value
        if not exp_name or exp_name == NONE_EXPERIMENT or state.dataset is None:
            return
        # What the model can predict, not what the dataset is annotated
        # with: an unannotated dataset has no labels at all, and would leave
        # this filter empty.
        experiment = state.experiments.get(exp_name)
        predictable = experiment.class_labels if experiment is not None else []
        labels = sorted(set(state.dataset.labels()) | {str(l) for l in predictable}, key=str)
        if len(labels) > FOCUS_MAX:
            picker = ClassPicker(lambda picked: self._on_filter_change(None), limit=None,
                                 label="Find a predicted class to keep")
            self.predicted_label_checks.update(picker_checks(picker, labels))
            body = picker.build()
        else:
            for v in labels:
                self.predicted_label_checks[v] = ft.Checkbox(label=str(v), value=False, on_change=self._on_filter_change)
            body = ft.Row(controls=list(self.predicted_label_checks.values()), wrap=True)
        self.predicted_filter_container.controls.append(
            ft.Column([ft.Text("Predicted label", weight=ft.FontWeight.BOLD, size=13), body])
        )

    def _on_filter_change(self, e):
        self._render_grid()
        self.root.update()
        self._to_top()

    def _on_experiment_change(self, e):
        self._rebuild_predicted_filter()
        self._render_grid()
        self.root.update()

    def _open_settings(self, e):
        self.ctx.page.open(self.settings_dialog)

    def _on_apply_settings(self, e):
        self.ctx.page.close(self.settings_dialog)
        self._render_grid()
        self.root.update()

    def _on_shuffle_toggle(self, e):
        self._set_shuffle(self._next_seed() if self.shuffle_switch.value else 0)

    def _on_reshuffle(self, e):
        self._set_shuffle(self._next_seed())

    def _next_seed(self) -> int:
        return int(np.random.default_rng().integers(1, 2**31 - 1))

    def _set_shuffle(self, seed: int) -> None:
        self.shuffle_seed = seed
        self.shuffle_switch.value = bool(seed)
        self.reshuffle_btn.visible = bool(seed)
        self._render_grid()
        if self.root.page:
            self.root.update()
        self._to_top()

    # ---- scrolling ----
    def _on_scroll(self, e):
        if e.max_scroll_extent is None or e.pixels is None:
            return
        viewport = e.viewport_dimension or self._page_height()
        if e.max_scroll_extent - e.pixels <= viewport * PREFETCH_SCREENS:
            self._load_more()

    def _load_more(self, e=None):
        if not self._load_lock.acquire(blocking=False):
            return  # scroll events come in bursts; one batch at a time
        try:
            if not self._has_more():
                return
            self._append_rows(self._batch_rows())
        finally:
            self._load_lock.release()
        if self.grid_area.page:
            self.grid_area.update()
            self.count_label.update()
            self.footer.update()

    def _to_top(self):
        if self.right_panel is not None and self.right_panel.page:
            self.right_panel.scroll_to(offset=0, duration=0)

    # ---- filtering ----
    def _filtered_df(self) -> pd.DataFrame:
        state = self.ctx.state
        df = state.dataset.df
        cols = state.dataset.columns
        mask = pd.Series(True, index=df.index)

        for col, entry in self.filter_controls.items():
            if entry["type"] == "categorical":
                selected = [v for v, cb in entry["checks"].items() if cb.value]
                if selected:
                    mask &= df[col].isin(selected)
            else:
                slider = entry["slider"]
                mask &= df[col].between(slider.start_value, slider.end_value)

        filtered = df[mask]

        exp_name = self.experiment_dropdown.value
        if exp_name and exp_name != NONE_EXPERIMENT:
            merged = state.merged(exp_name)
            merged = merged[merged[cols.uuid_col].isin(filtered[cols.uuid_col])]
            selected_pred = [v for v, cb in self.predicted_label_checks.items() if cb.value]
            if selected_pred:
                merged = merged[merged["predicted_label"].isin(selected_pred)]
            return merged
        return filtered

    # ---- grid rendering ----
    def _grid_cols(self) -> int:
        try:
            v = int(self.cols_field.value)
        except (TypeError, ValueError):
            v = DEFAULT_COLS
        return max(1, min(MAX_COLS, v))

    def _thumb_size(self) -> int:
        return max(56, min(150, 920 // self._cols_n))

    def _page_height(self) -> float:
        return float(getattr(self.ctx.page, "height", None) or 900)

    def _batch_rows(self) -> int:
        """Rows per load: a window's worth plus one, so the first load
        overflows the panel and there is something to scroll."""
        return math.ceil(self._page_height() / (self._thumb_size() + ROW_EXTRA_HEIGHT)) + 1

    def _total(self) -> int:
        return 0 if self._filtered is None else len(self._filtered)

    def _has_more(self) -> bool:
        return len(self._current_records) < self._total()

    def _render_grid(self):
        """Start the mosaic over from the top of the current filtered set."""
        self.grid_area.controls.clear()
        self._current_records = []
        self._filtered = None
        state = self.ctx.state
        if state.dataset is None:
            self.grid_area.controls.append(info_banner("Load a dataset in the 'Data' tab first."))
            self.count_label.value = ""
            self.footer.visible = False
            return

        self._cols_n = self._grid_cols()
        self.grid_shape_label.value = f"{self._cols_n} per row"
        filtered = self._filtered_df()
        if self.shuffle_seed:
            order = np.random.default_rng(self.shuffle_seed).permutation(len(filtered))
            filtered = filtered.iloc[order]
        self._filtered = filtered
        self._append_rows(self._batch_rows())

    def _append_rows(self, n_rows: int):
        """Add the next `n_rows` rows of crops under the ones already shown.
        The rows already in the grid are not touched, so the update that
        follows sends only the new ones to the browser."""
        state = self.ctx.state
        cols_cfg = state.dataset.columns
        cols_n = self._cols_n
        thumb_size = self._thumb_size()
        show_prediction = bool(self.experiment_dropdown.value) and self.experiment_dropdown.value != NONE_EXPERIMENT

        start = len(self._current_records)
        records = self._filtered.iloc[start:start + n_rows * cols_n].to_dict("records")
        self._current_records.extend(records)

        for r in range(0, len(records), cols_n):
            row_controls = []
            for i in range(start + r, start + min(r + cols_n, len(records))):
                rec = self._current_records[i]
                img_path = state.dataset.resolve_crop_path(rec[cols_cfg.crop_col])
                b64 = file_to_full_base64(img_path, *crop_display(self.ctx, rec))
                # A crop with no annotation shows no label rather than "nan",
                # and is neither right nor wrong -- so no border either.
                label = rec.get(cols_cfg.label_col)
                annotated = label is not None and not pd.isna(label) and str(label) != ""
                caption = str(label) if annotated else ""
                border_color = None
                if show_prediction:
                    if annotated:
                        border_color = ft.Colors.GREEN if rec.get("correct") else ft.Colors.RED
                    caption = f"{caption or '?'} -> {rec.get('predicted_label')}"
                cell = ft.Container(
                    width=thumb_size,
                    content=ft.Column([
                        ft.Image(src_base64=b64, width=thumb_size, height=thumb_size, fit=ft.ImageFit.COVER, border_radius=6),
                        ft.Text(caption, size=9, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                    ], spacing=2, tight=True),
                    border=ft.border.all(2, border_color) if border_color else None,
                    border_radius=6,
                    padding=3,
                    ink=True,
                    on_click=lambda e, idx=i: self._open_detail(idx),
                )
                row_controls.append(cell)
            self.grid_area.controls.append(ft.Row(controls=row_controls, spacing=6))

        shown, total = len(self._current_records), self._total()
        self.count_label.value = (
            f"{total:,} matching rows" if shown >= total
            else f"{shown:,} of {total:,} matching rows shown  ·  scroll for more"
        )
        self.footer.visible = shown < total

    # ---- detail dialog (the shared scrolling carousel) ----
    def _open_detail(self, index: int):
        state = self.ctx.state
        if not self._current_records or self._filtered is None:
            return
        cols_cfg = state.dataset.columns
        # The whole filtered set, in the mosaic's order, starting at the
        # clicked image and wrapping round -- rows are read from the frame
        # as the carousel scrolls to them, so this costs nothing up front.
        filtered, total = self._filtered, self._total()

        def provider(start: int, end: int) -> list:
            positions = [(index + i) % total for i in range(start, end)]
            return filtered.iloc[positions].to_dict("records")

        # No uuid column: nothing to highlight by, and nothing to key a
        # review decision on -- the carousel leaves both out for None.
        uuid_col = cols_cfg.uuid_col or None
        clicked_uuid = self._current_records[index].get(uuid_col) if uuid_col else None

        self.detail_title.value = f"{total:,} matching image(s), from the one clicked"
        self.detail_content.content = Carousel(
            state.dataset, total, provider, cols_cfg.crop_col, uuid_col, highlight_uuid=clicked_uuid,
            ctx=self.ctx, review_source="gallery", label_col=cols_cfg.label_col,
        ).build()
        self.ctx.page.open(self.detail_dialog)

    def _detail_close(self, e):
        self.ctx.page.close(self.detail_dialog)

    # ---- external navigation (from confusion matrix drill-down) ----
    def apply_external_filter(self, experiment_name: str, true_label: str, predicted_label: str):
        state = self.ctx.state
        cols_cfg = state.dataset.columns

        if self._dirty:
            # Data changed since this view last rebuilt (see refresh()) --
            # this is an explicit, deliberate navigation (a confusion-matrix
            # cell click), so it gets to jump the "click Update" requirement
            # and just rebuild filters now, same as clicking Update would.
            self._rebuild_filters()
            self._dirty = False
            self.stale_banner.hide()

        if cols_cfg.label_col not in self.active_filter_cols:
            self.active_filter_cols.append(cols_cfg.label_col)
            self._rebuild_filters()

        for col, entry in self.filter_controls.items():
            if entry["type"] == "categorical":
                for cb in entry["checks"].values():
                    cb.value = False
            else:
                slider = entry["slider"]
                slider.start_value = slider.min
                slider.end_value = slider.max

        label_entry = self.filter_controls.get(cols_cfg.label_col)
        if label_entry and true_label in label_entry["checks"]:
            label_entry["checks"][true_label].value = True

        self.experiment_dropdown.value = experiment_name
        self._rebuild_predicted_filter()
        if predicted_label in self.predicted_label_checks:
            self.predicted_label_checks[predicted_label].value = True

        self._render_grid()
        if self.root.page:
            self.root.update()
        self._to_top()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow an experiment rename (see AppContext.notify_rename): the
        "color by experiment prediction" dropdown holds the name as both an
        option and its selected value, and a stale value there would send
        the next render to AppState.merged() with a name that no longer
        exists. Nothing is re-scanned -- the grid is showing the same rows
        under a new label."""
        for option in self.experiment_dropdown.options or []:
            if option.key == old_name:
                option.key = new_name
                option.text = new_name
        if self.experiment_dropdown.value == old_name:
            self.experiment_dropdown.value = new_name
        if self.root.page:
            self.root.update()

    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        """The filters the user built, not the controls that draw them.

        Checkbox values are stored as strings: a value read back out of a
        CSV can come back as a numpy int where it went in as a Python one,
        and a filter that silently stopped matching would look like a
        filter that had simply been cleared.
        """
        filters = {}
        for col, entry in self.filter_controls.items():
            if entry["type"] == "categorical":
                filters[col] = {
                    "kind": "categorical",
                    "selected": [str(v) for v, cb in entry["checks"].items() if cb.value],
                }
            else:
                slider = entry["slider"]
                filters[col] = {
                    "kind": "numeric",
                    "start": float(slider.start_value),
                    "end": float(slider.end_value),
                }
        return {
            "columns": list(self.active_filter_cols),
            "filters": filters,
            "experiment": self.experiment_dropdown.value or NONE_EXPERIMENT,
            "predicted": [str(v) for v, cb in self.predicted_label_checks.items() if cb.value],
            "cols": str(self.cols_field.value or ""),
            "shuffle": int(self.shuffle_seed),
        }

    def restore_view_state(self, state: dict) -> list:
        """Rebuild the filter panel, then put the user's choices back into
        it -- the same order apply_external_filter uses, and for the same
        reason: _rebuild_filters replaces every control, so anything set
        before it would be thrown away."""
        warnings = []
        if self.ctx.state.dataset is None:
            return ["Gallery: no dataset is loaded."]

        dataset = self.ctx.state.dataset
        valid = set(dataset.df.columns) | {dataset.columns.label_col}
        wanted = [c for c in state.get("columns", []) or [] if isinstance(c, str)]
        missing = [c for c in wanted if c not in valid]
        if missing:
            warnings.append(f"Gallery: no column named {', '.join(missing)} -- that filter was dropped.")
        self.active_filter_cols = [c for c in wanted if c in valid]

        # The data has not changed under us -- this *is* the load -- so the
        # stale banner would be asking the user to rebuild what was just built.
        self._dirty = False
        self.stale_banner.hide()
        self._rebuild_filters()

        experiment = state.get("experiment") or NONE_EXPERIMENT
        if experiment != NONE_EXPERIMENT:
            if experiment in self.ctx.state.experiments:
                self.experiment_dropdown.value = experiment
                self._rebuild_predicted_filter()
            else:
                warnings.append(f"Gallery: experiment '{experiment}' is not loaded -- colouring is off.")

        saved = state.get("filters") or {}
        for col, entry in self.filter_controls.items():
            spec = saved.get(col)
            if not isinstance(spec, dict):
                continue
            if entry["type"] == "categorical" and spec.get("kind") == "categorical":
                selected = {str(v) for v in spec.get("selected", []) or []}
                for value, check in entry["checks"].items():
                    check.value = str(value) in selected
            elif entry["type"] == "numeric" and spec.get("kind") == "numeric":
                slider = entry["slider"]
                try:
                    start, end = float(spec["start"]), float(spec["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                # The column's range can have moved since -- clamp rather
                # than set a window the slider cannot represent.
                slider.start_value = max(slider.min, min(slider.max, start))
                slider.end_value = max(slider.min, min(slider.max, end))

        predicted = {str(v) for v in state.get("predicted", []) or []}
        for value, check in self.predicted_label_checks.items():
            check.value = str(value) in predicted

        # Snapshots from before the mosaic scrolled also carry "rows" and
        # "page"; there is nothing left for either to set.
        if state.get("cols"):
            self.cols_field.value = str(state["cols"])
        try:
            self.shuffle_seed = max(0, int(state.get("shuffle", 0) or 0))
        except (TypeError, ValueError):
            self.shuffle_seed = 0
        self.shuffle_switch.value = bool(self.shuffle_seed)
        self.reshuffle_btn.visible = bool(self.shuffle_seed)

        self._render_grid()
        if self.root.page:
            self.root.update()
        return warnings

    def refresh(self):
        """Called via ctx.notify_change() whenever the underlying data
        changes. Rebuilding filters (per-column unique/min/max scans) is
        real work at scale, so this never happens automatically -- it just
        flags the view as stale and shows a banner; _on_update_click() (or
        adding/removing a filter, which needs a fresh render anyway) does
        the actual rebuild. Scrolling and toggling *existing* filters stay
        instant regardless -- they just re-render the grid from whatever
        filters/data are already in front of the user."""
        self._dirty = True
        if self.root.page:
            self.stale_banner.show()

    def on_display_changed(self):
        """The display LUT changed. Only the mosaic needs redrawing -- the
        filters are unaffected, so this is a grid re-render rather than the
        full (expensive) rebuild behind the stale banner."""
        if self.root.page:
            self._render_grid()
            self.root.update()

    def _on_update_click(self, e):
        self._refresh_in_background()

    def _refresh_in_background(self):
        """Every user-triggered filter rebuild goes through here. Building a
        filter control means scanning every unique value of its column over
        the whole dataset -- the single most expensive thing measured in
        scaling.md -- and on the UI thread that froze the app outright.
        Ignored while a rebuild is already running.

        Note this is only the *filter panel*: scrolling the mosaic and toggling
        filters that already exist stay synchronous, because they are flat
        in dataset size and instant either way."""
        self._set_progress("Rebuilding filters...")
        self.rebuild_job.run(self._refresh_now, on_error=self._on_rebuild_error)

    def _on_rebuild_error(self, ex: Exception):
        self._clear_progress()
        self.grid_area.controls.clear()
        self.grid_area.controls.append(
            info_banner(f"Error rebuilding the gallery: {ex}", icon=ft.Icons.ERROR_OUTLINE))
        if self.root.page:
            self.root.update()

    def _refresh_now(self):
        self._rebuild_filters()
        self._render_grid()
        self._dirty = False
        self.stale_banner.hide()
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        controls_row = ft.Row(controls=[
            ft.IconButton(ft.Icons.SETTINGS_OUTLINED, tooltip="Mosaic settings", on_click=self._open_settings),
            self.grid_shape_label,
            ft.VerticalDivider(),
            self.experiment_dropdown,
            ft.VerticalDivider(),
            self.shuffle_switch,
            self.reshuffle_btn,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

        left_panel = card(
            ft.Column(
                [
                    section_title("Filters"),
                    ft.Text("Add a filter for each column you want to narrow down; nothing is filtered until you do.", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    self.add_filter_dropdown,
                    self.add_filter_button,
                    self.skipped_text,
                    self.filters_column,
                    self.predicted_filter_container,
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
            width=300,
        )
        self.right_panel = right_panel = ft.Column(
            controls=[
                card(ft.Column([section_title("Mosaic"), controls_row, self.count_label])),
                self.progress_banner,
                self.grid_area,
                self.footer,
            ],
            expand=True, scroll=ft.ScrollMode.AUTO, spacing=12,
            on_scroll=self._on_scroll, on_scroll_interval=SCROLL_EVENT_INTERVAL,
        )

        self.root.expand = True
        self.root.content = ft.Column(
            controls=[
                self.stale_banner.control,
                ft.Row(
                    controls=[left_panel, right_panel], expand=True, spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ],
            expand=True, spacing=10,
        )
        self._refresh_now()
        return self.root
