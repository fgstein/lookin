"""Tab 2: a dashboard grid showing distributions for columns the user picks.

Nothing renders by default -- a wide CSV can have dozens of metadata
columns, and building a full-dataset chart for every one of them on every
load is both slow and rarely what's wanted. Instead the user adds the
specific columns they care about via the picker at the top; each chart can
be removed individually.
"""
from __future__ import annotations

import flet as ft
import numpy as np
import matplotlib.pyplot as plt

from lookin.ui.components import fig_to_zoomable_image, section_title, info_banner, card, style_axes, GRID_NONE, ACCENT, BackgroundRebuild, StaleBanner
from lookin.core.chart_style import subplots

CHART_WIDTH = 860
CHART_HEIGHT = 300


# Past this many distinct values a categorical chart is drawn as sorted
# sizes rather than one bar per value (see _make_chart); the RANKED_NAMED
# largest and smallest are named under it.
RANKED_CATEGORIES_MIN = 60
RANKED_NAMED = 5
# "Fewer than this many" is the count reported for the long tail.
RANKED_SMALL = 10


class StatsView:
    def __init__(self, ctx):
        self.ctx = ctx
        self.active_columns: list[str] = []
        self.charts_area = ft.Column(spacing=14)
        self.progress_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.progress_banner = ft.Row(
            [ft.ProgressRing(width=14, height=14, stroke_width=2), self.progress_text],
            spacing=10, visible=False,
        )
        self.stale_banner = StaleBanner(self._on_update_click)
        self.rebuild_job = BackgroundRebuild(ctx, "Statistics")
        self.add_dropdown = ft.Dropdown(label="Add a chart for column...", width=320, border_radius=8)
        self.add_button = ft.FilledButton("Add", icon=ft.Icons.ADD, on_click=self._on_add_click, style=ft.ButtonStyle(bgcolor=ACCENT))
        self.skipped_text = ft.Text("", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
        self.root = ft.Container(padding=20)
        self._dirty = False

    def _make_chart(self, col: str, series, is_numeric: bool) -> ft.Control:
        fig, ax = subplots(figsize=(9.5, 3.2))
        if is_numeric:
            ax.hist(series, bins=40, color="#6C5CE7", edgecolor="white")
            ax.set_title(f"{col}", fontsize=12)
            ax.set_ylabel("count", fontsize=10)
            # Both axes are scales here -- the column's own units and a
            # count -- so both get the full grid.
            style_axes(ax)
        elif series.nunique() > RANKED_CATEGORIES_MIN:
            # One bar per value stops being readable long before two
            # thousand classes: the sizes, sorted, as one curve -- the shape
            # of the imbalance is what this chart is for -- with the
            # extremes named under it instead of on the axis.
            counts = series.value_counts().sort_values(ascending=False)
            ranks = np.arange(1, len(counts) + 1)
            ax.fill_between(ranks, counts.values, step="mid", color="#6C5CE7", alpha=0.35)
            ax.step(ranks, counts.values, where="mid", color="#6C5CE7")
            ax.set_yscale("log")
            ax.set_xlim(0.5, len(counts) + 0.5)
            ax.set_title(f"{col} -- {len(counts):,} values, largest first", fontsize=12)
            ax.set_xlabel("rank", fontsize=10)
            ax.set_ylabel("count (log scale)", fontsize=10)
            style_axes(ax)
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            chart_image = fig_to_zoomable_image(fig, self.ctx, col, width=CHART_WIDTH - 36, height=CHART_HEIGHT - 40)

            def named(part) -> str:
                return ", ".join(f"{label} ({int(n):,})" for label, n in part.items())

            notes = [
                f"Largest: {named(counts.head(RANKED_NAMED))}",
                f"Smallest: {named(counts.tail(RANKED_NAMED)[::-1])}",
                f"Median {int(counts.median()):,} per value; "
                f"{int((counts < RANKED_SMALL).sum()):,} value(s) with fewer than {RANKED_SMALL}.",
            ]
            header = ft.Row(
                [ft.Text(col, weight=ft.FontWeight.BOLD, size=13),
                 ft.IconButton(ft.Icons.CLOSE, icon_size=16, tooltip="Remove this chart",
                                on_click=lambda e, c=col: self._on_remove_click(c))],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            )
            return card(ft.Column([header, chart_image,
                                   *[ft.Text(n, size=11, color=ft.Colors.ON_SURFACE_VARIANT) for n in notes]],
                                  spacing=4))
        else:
            counts = series.value_counts().sort_values(ascending=False)
            ax.bar(counts.index.astype(str), counts.values, color="#6C5CE7")
            ax.set_title(f"{col}", fontsize=12)
            ax.set_ylabel("count", fontsize=10)
            if len(counts) > 12:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
            else:
                plt.setp(ax.get_xticklabels(), fontsize=9)
            # x is a list of categories: nothing between two of them to
            # subdivide, so only the count axis is gridded.
            style_axes(ax, x=GRID_NONE)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        chart_image = fig_to_zoomable_image(fig, self.ctx, col, width=CHART_WIDTH - 36, height=CHART_HEIGHT - 40)
        header = ft.Row(
            [ft.Text(col, weight=ft.FontWeight.BOLD, size=13),
             ft.IconButton(ft.Icons.CLOSE, icon_size=16, tooltip="Remove this chart",
                            on_click=lambda e, c=col: self._on_remove_click(c))],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )
        return card(ft.Column([header, chart_image], spacing=4))

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

    def _available_columns(self):
        """(label, is_numeric) for every column that's safe/sensible to
        chart -- excludes high-cardinality columns (see data_manager's
        HIGH_CARDINALITY_MAX_UNIQUE) and anything already showing."""
        state = self.ctx.state
        if state.dataset is None:
            return [], []
        cols_cfg = state.dataset.columns
        kinds = state.dataset.column_kinds()
        numeric_set = set(kinds.numeric)
        all_cols = [cols_cfg.label_col] + kinds.numeric + kinds.binary + kinds.categorical
        available = [c for c in all_cols if c not in self.active_columns]
        return available, numeric_set

    def _rebuild_add_dropdown(self):
        available, _ = self._available_columns()
        self.add_dropdown.options = [ft.dropdown.Option(c) for c in available]
        if self.add_dropdown.value not in available:
            self.add_dropdown.value = available[0] if available else None

    def _on_add_click(self, e):
        col = self.add_dropdown.value
        if not col or col in self.active_columns:
            return
        self.active_columns.append(col)
        self._refresh_in_background()

    def _on_remove_click(self, col: str):
        if col in self.active_columns:
            self.active_columns.remove(col)
        self._refresh_in_background()

    def _render(self):
        self.charts_area.controls.clear()
        state = self.ctx.state
        self._rebuild_add_dropdown()
        if state.dataset is None:
            self.charts_area.controls.append(info_banner("Load a dataset in the 'Data' tab first."))
            self.skipped_text.value = ""
            self._clear_progress()
            return

        df = state.dataset.df
        kinds = state.dataset.column_kinds()
        numeric_set = set(kinds.numeric)
        # A column can disappear between renders (relabeling, a fresh
        # dataset load with different columns) -- drop anything stale
        # instead of erroring on a missing column.
        valid_columns = set(df.columns) | {state.dataset.columns.label_col}
        self.active_columns = [c for c in self.active_columns if c in valid_columns]

        if not self.active_columns:
            self.charts_area.controls.append(info_banner(
                "No charts shown yet -- pick a column above and click Add."
            ))
        else:
            for i, col in enumerate(self.active_columns):
                self._set_progress(f"Rendering charts... {col} ({i + 1}/{len(self.active_columns)})")
                series = df[col].dropna()
                self.charts_area.controls.append(self._make_chart(col, series, col in numeric_set))

        self.skipped_text.value = (
            f"Not offered above (too many unique values): {', '.join(kinds.high_cardinality)}"
            if kinds.high_cardinality else ""
        )
        self._clear_progress()

    def refresh(self):
        """Called via ctx.notify_change() whenever the underlying data
        changes. Rebuilding every active chart is expensive at scale, so
        this never happens automatically -- it just flags the view as
        stale and shows a banner; _on_update_click() (or adding/removing a
        chart, which needs a fresh render anyway) does the actual work."""
        self._dirty = True
        if self.root.page:
            self.stale_banner.show()

    def _on_update_click(self, e):
        self._refresh_in_background()

    def _refresh_in_background(self):
        """Every user-triggered rebuild goes through here. Rendering one
        chart per active column is seconds of matplotlib at scale (see
        scaling.md), and doing it on the UI thread meant the progress
        banner below could never actually paint. Ignored while a rebuild is
        already running -- that banner is already saying so."""
        self._set_progress("Rendering charts...")
        self.rebuild_job.run(self._refresh_now, on_error=self._on_rebuild_error)

    def _on_rebuild_error(self, ex: Exception):
        self._clear_progress()
        self.charts_area.controls.clear()
        self.charts_area.controls.append(
            info_banner(f"Error rendering charts: {ex}", icon=ft.Icons.ERROR_OUTLINE))
        if self.root.page:
            self.root.update()

    def _refresh_now(self):
        self._render()
        self._dirty = False
        self.stale_banner.hide()
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        self.root.content = ft.Column(
            controls=[
                section_title("Column statistics", icon=ft.Icons.BAR_CHART, ctx=self.ctx, manual="column-statistics"),
                ft.Text("Add a chart for each column you want to inspect; nothing shows until you do.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                self.stale_banner.control,
                ft.Row([self.add_dropdown, self.add_button], spacing=10),
                self.skipped_text,
                self.progress_banner,
                self.charts_area,
            ],
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
        )
        self._refresh_now()
        return self.root
