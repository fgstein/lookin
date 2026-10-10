"""The app-bar control for how crops are mapped to the screen: the display
LUT, and the roll correction that turns each crop upright.

One session-wide setting rather than one per view: a LUT that applied in
the carousel but not in the gallery would mean two views of the same crop
disagreeing about what it looks like, which is exactly the confusion a
display setting is supposed to remove. The roll correction sits here for
the same reason, and because the two answer the same question -- what does
this crop look like on screen -- rather than anything about the data.

Both are *display* transforms. The file is never touched, and nothing the
tool computes -- metrics, embeddings, prototypes -- reads pixels through
them. That is worth saying out loud in the UI, because a viewer that
silently rewrote what it measured would be the worst kind of bug.
"""
from __future__ import annotations

import flet as ft
import pandas as pd

from lookin.ui.components import (
    ACCENT, LUT_DESCRIPTIONS, LUT_NONE, LUT_OPTIONS, ROLL_NONE,
)

# The dropdown entry meaning "leave every crop as stored". A key, compared
# in code -- only the label beside it is display text (see CLAUDE.md). It
# cannot be the empty string, which Flet's dropdown treats as no selection.
ROLL_NONE_KEY = "__none__"


def roll_column_candidates(dataset) -> list:
    """The numeric, non-structural columns of `dataset`, in file order.

    A roll is one number per crop, so only numeric columns are offered.
    The three structural roles are excluded: a uuid or a crop path is
    never a roll, and offering a numeric *label* column here would just be
    a way to rotate crops by their class.

    Returns nothing at all when no dataset is loaded, which is what the
    dialog shows instead of an empty dropdown that looks broken.
    """
    if dataset is None:
        return []
    reserved = {dataset.columns.uuid_col, dataset.columns.crop_col, dataset.columns.label_col}
    return [str(col) for col in dataset.df.columns
            if str(col) not in reserved and pd.api.types.is_numeric_dtype(dataset.df[col])]


class DisplaySettings:
    """A small button in the app bar that opens the LUT and roll picker."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.label = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.indicator = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.TUNE, size=17, color=ACCENT),
                ft.Column([ft.Text("Display", size=13), self.label], spacing=0, tight=True),
            ], spacing=6, tight=True),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=8, ink=True,
            tooltip="How crops are mapped to the screen -- click to change",
            on_click=self._open,
        )

        self.description = ft.Text(size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.options_column = ft.Column(spacing=2)

        # Rebuilt from the loaded dataset every time the dialog opens (see
        # _render_roll): which columns exist is not known when this is
        # built, and changes with every dataset loaded afterwards.
        self.roll_dropdown = ft.Dropdown(
            label="Roll column", dense=True, width=250, options=[],
            on_change=self._choose_roll_column,
        )
        self.roll_area = ft.Column(spacing=8, tight=True)

        self.dialog = ft.AlertDialog(
            modal=False,
            title=ft.Row([ft.Icon(ft.Icons.TUNE, color=ACCENT), ft.Text("Crop display", weight=ft.FontWeight.BOLD)],
                         spacing=10),
            # Bounded height, so the roll section below the LUTs scrolls
            # inside the dialog rather than pushing it off a short window.
            content=ft.Container(width=470, height=560, content=ft.Column([
                ft.Text(
                    "Greyscale crops often use a narrow slice of the available range, which shows as "
                    "a flat grey square. A lookup table stretches that slice across the screen.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                self.options_column,
                ft.Divider(height=1),
                self.description,
                ft.Divider(height=1),
                ft.Text("Roll correction", size=13, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "If a column of the dataset records the roll each crop was cut at, every "
                    "crop can be turned by the nearest quarter turn that cancels it, so a page "
                    "of thumbnails shares one orientation. Values are read as radians.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                self.roll_area,
                ft.Text(
                    "Only whole quarter turns are applied: a crop rolled by 0.3 rad stays as "
                    "stored, and no pixel is ever resampled.",
                    size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Divider(height=1),
                ft.Text(
                    "Display only: the files are never modified, and no metric, embedding or "
                    "prototype is computed through the LUT or the roll correction.",
                    size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=12, scroll=ft.ScrollMode.AUTO)),
            on_dismiss=lambda e: self.ctx.page.close(self.dialog),
            actions=[ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.dialog))],
        )
        self._sync_label()

    def _current(self) -> str:
        return getattr(self.ctx, "display_lut", LUT_NONE) or LUT_NONE

    def _current_roll_col(self) -> str:
        return getattr(self.ctx, "display_roll_col", ROLL_NONE) or ROLL_NONE

    def _sync_label(self) -> None:
        current = self._current()
        self.label.value = next((label for key, label in LUT_OPTIONS if key == current), current)
        roll_col = self._current_roll_col()
        if roll_col:
            # Named rather than just flagged: two datasets can both have a
            # plausible-looking angle column, and "roll: yaw_deg" is the
            # only version of this that says which one is being applied.
            self.label.value += f"  ·  roll: {roll_col}"
        self.description.value = LUT_DESCRIPTIONS.get(current, "")

    def _open(self, e) -> None:
        self._render_options()
        self._render_roll()
        self._sync_label()
        self.ctx.page.open(self.dialog)

    def _render_options(self) -> None:
        current = self._current()
        self.options_column.controls = [
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if key == current
                            else ft.Icons.RADIO_BUTTON_UNCHECKED,
                            size=16, color=ACCENT if key == current else ft.Colors.ON_SURFACE_VARIANT),
                    ft.Column([
                        ft.Text(label, size=13,
                                weight=ft.FontWeight.BOLD if key == current else ft.FontWeight.NORMAL),
                        ft.Text(LUT_DESCRIPTIONS.get(key, ""), size=11,
                                color=ft.Colors.ON_SURFACE_VARIANT),
                    ], spacing=1, tight=True, expand=True),
                ], spacing=10),
                padding=ft.padding.symmetric(horizontal=10, vertical=8),
                border_radius=8, ink=True,
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST if key == current else None,
                on_click=lambda e, key=key: self._choose(key),
            )
            for key, label in LUT_OPTIONS
        ]

    # ---- roll correction ------------------------------------------------

    def _render_roll(self) -> None:
        """Rebuild the column picker from whatever dataset is loaded now.

        A column chosen against a previous dataset is dropped here rather
        than left in place: it would silently do nothing (a record with no
        such column is left as stored), and a setting that claims to be on
        while doing nothing is worse than one that is off.
        """
        dataset = getattr(self.ctx.state, "dataset", None)
        candidates = roll_column_candidates(dataset)
        current = self._current_roll_col()
        if current and current not in candidates:
            self.ctx.display_roll_col = ROLL_NONE
            current = ROLL_NONE

        if dataset is None:
            self.roll_area.controls = [ft.Text(
                "Load a dataset to pick a roll column.",
                size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)]
            return
        if not candidates:
            self.roll_area.controls = [ft.Text(
                "This dataset has no numeric column besides its id, crop and label ones, "
                "so there is nothing that could hold a roll.",
                size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)]
            return

        self.roll_dropdown.options = [
            ft.dropdown.Option(ROLL_NONE_KEY, "No rotation (as stored)"),
            *(ft.dropdown.Option(col, col) for col in candidates),
        ]
        self.roll_dropdown.value = current or ROLL_NONE_KEY
        self.roll_area.controls = [self.roll_dropdown]

    def _choose_roll_column(self, e) -> None:
        chosen = self.roll_dropdown.value or ROLL_NONE_KEY
        column = ROLL_NONE if chosen == ROLL_NONE_KEY else chosen
        if column == self._current_roll_col():
            return
        self.ctx.display_roll_col = column
        self._after_roll_change()

    def _after_roll_change(self) -> None:
        self._render_roll()
        self._sync_label()
        # Same reasoning as _choose below: the crops on screen were encoded
        # at the old orientation, and only the views drawing them redraw.
        self.ctx.notify_display_change()
        for control in (self.roll_area, self.roll_dropdown, self.label):
            if control.page:
                control.update()

    def _choose(self, key: str) -> None:
        if key == self._current():
            return
        self.ctx.display_lut = key
        self._render_options()
        self._sync_label()
        # Every crop on screen was encoded through the old LUT, so the views
        # holding them have to re-render. This goes through notify_change
        # rather than a redraw of its own: the views already know how to
        # rebuild themselves, and which ones are showing crops is their
        # business, not this control's.
        self.ctx.notify_display_change()
        for control in (self.options_column, self.description, self.label):
            if control.page:
                control.update()
