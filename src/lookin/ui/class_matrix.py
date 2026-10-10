"""The pieces a per-class view needs once there are more classes than it
can draw one control, bar or cell per class for (see core/class_focus.py):

* ClassFocus -- which classes a tab shows: a criterion, or picked by hand,
  at most FOCUS_MAX. Hidden while every class fits.
* ClassPicker -- type to find a class among thousands and pick some. A
  dropdown listing two thousand labels is neither searchable nor cheap to
  send to the browser; this only ever shows the labels that match.
* class_dropdown -- a single-class dropdown that becomes type-to-filter
  past FOCUS_MAX options.
* confused_pairs_table -- the largest off-diagonal cells, clickable.
* overview_image -- the whole matrix as one picture, for its shape.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import flet as ft
import numpy as np
from lookin.core.class_focus import (
    FOCUS_MANUAL, FOCUS_MAX, MIN_SUPPORT_DEFAULT, ZOOM_BLOCKS, overview_figure,
)
from lookin.ui.components import ZOOM_DPI, card, fig_to_base64

# How many matching labels the picker lists under the search field.
PICKER_SUGGESTIONS = 12
# Rows of the confused-pairs table shown under each matrix.
PAIRS_SHOWN = 15


class ClassPicker:
    """A search field over class labels plus the picked ones as chips.

    `on_change(labels)` fires with the new list on every add or removal;
    the owner decides what that means (the Metrics tab switches its focus
    criterion to hand-picked). Labels are carried in closures, never
    encoded into a key -- they come from the user's CSV.
    """

    def __init__(self, on_change: Callable[[list], None], limit: int | None = FOCUS_MAX,
                 label: str = "Find a class"):
        self.on_change = on_change
        self.limit = limit
        self._labels: list = []
        self._picked: list = []
        self.search = ft.TextField(
            label=label, hint_text="Type part of a class name", width=280, dense=True,
            border_radius=8, prefix_icon=ft.Icons.SEARCH, on_change=self._on_search,
        )
        self.suggestions = ft.Row(spacing=6, run_spacing=6, wrap=True)
        self.message = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.chips = ft.Row(spacing=6, run_spacing=6, wrap=True)
        self.root = ft.Column([self.search, self.suggestions, self.message, self.chips], spacing=8)

    def build(self) -> ft.Control:
        return self.root

    def set_labels(self, labels: list) -> None:
        """Every class that can be picked."""
        self._labels = list(labels)
        self._refresh_suggestions()

    @property
    def picked(self) -> list:
        return list(self._picked)

    def set_picked(self, picked: list) -> None:
        """Show `picked` as the current selection, without firing on_change."""
        self._picked = list(picked)
        self._rebuild_chips()
        self._refresh_suggestions()

    def set_one(self, label, picked: bool) -> None:
        """Pick or unpick one label without firing on_change -- for code
        that restores a selection one value at a time (see PickedFlag)."""
        if picked and label not in self._picked:
            self._picked.append(label)
        elif not picked and label in self._picked:
            self._picked = [l for l in self._picked if l != label]
        else:
            return
        self._rebuild_chips()
        self._refresh_suggestions()

    def _on_search(self, e) -> None:
        self._refresh_suggestions()
        self._push()

    def _refresh_suggestions(self) -> None:
        query = (self.search.value or "").strip().lower()
        self.message.value = ""
        if not query:
            self.suggestions.controls = []
            return
        found = [l for l in self._labels if query in str(l).lower()]
        # Exact, then prefix, then anywhere: only the first few are listed,
        # and among two thousand numeric labels "21" would otherwise sit
        # behind "1021", "1121", ... and be impossible to pick.
        found.sort(key=lambda l: (str(l).lower() != query, not str(l).lower().startswith(query)))
        matches = [l for l in found if l not in self._picked]
        self.suggestions.controls = [
            ft.OutlinedButton(str(label), icon=ft.Icons.ADD, on_click=lambda e, l=label: self._add(l))
            for label in matches[:PICKER_SUGGESTIONS]
        ]
        if not found:
            self.message.value = "No class matches."
        elif len(matches) > PICKER_SUGGESTIONS:
            self.message.value = f"{len(matches):,} classes match -- showing the first {PICKER_SUGGESTIONS}."

    def _add(self, label) -> None:
        if self.limit is not None and len(self._picked) >= self.limit:
            self.message.value = f"At most {self.limit} classes -- remove one first."
            self._push()
            return
        self._picked.append(label)
        self._changed()

    def _remove(self, label) -> None:
        self._picked = [l for l in self._picked if l != label]
        self._changed()

    def _clear(self, e=None) -> None:
        self._picked = []
        self._changed()

    def _changed(self) -> None:
        self._rebuild_chips()
        self._refresh_suggestions()
        self._push()
        self.on_change(list(self._picked))

    def _rebuild_chips(self) -> None:
        chips = [
            ft.Chip(label=ft.Text(str(label), size=11), on_delete=lambda e, l=label: self._remove(l))
            for label in self._picked
        ]
        if self._picked:
            chips.append(ft.TextButton("Clear", icon=ft.Icons.CLEAR_ALL, on_click=self._clear))
        self.chips.controls = chips

    def _push(self) -> None:
        if self.root.page:
            self.root.update()


class PickedFlag:
    """Stands in for an ft.Checkbox of one value, backed by a ClassPicker:
    `.value` is whether the value is picked, and setting it picks or
    unpicks it. Lets code written against a {value: checkbox} dict -- the
    Gallery's filters, their snapshots, the confusion-matrix drill-down --
    work unchanged when the checkboxes become a picker."""

    def __init__(self, picker: "ClassPicker", label):
        self._picker = picker
        self._label = label

    @property
    def value(self) -> bool:
        return self._label in self._picker._picked

    @value.setter
    def value(self, picked: bool) -> None:
        self._picker.set_one(self._label, bool(picked))


def picker_checks(picker: "ClassPicker", labels: list) -> dict:
    """{label: PickedFlag} over `labels`, the picker offering all of them."""
    picker.set_labels(labels)
    return {label: PickedFlag(picker, label) for label in labels}


class ClassFocus:
    """Which classes a tab's per-class views show once there are too many:
    a criterion picks at most `limit` of them, or the user does by hand.

    One per tab, shared by everything on it that draws per class, so the
    grid, the bars and the tables of a tab all talk about the same classes.
    The owner supplies the criteria -- (key, label) pairs, FOCUS_MANUAL
    among them -- and, at each render, a `rank(criterion)` returning the
    classes that criterion picks; `on_change()` fires when the user picks
    another criterion or edits the set, and the owner redraws from what it
    already computed. While every class fits (`limit` or fewer) the control
    is hidden and every class is shown.

    `support_criteria` are the criteria that rank on a per-class rate, which
    the smallest classes would otherwise lead: for those an "at least N"
    field appears, and the owner's rank() reads `min_support` and leaves
    the smaller classes out (see class_focus.rank_by). `support_noun` is
    what N counts -- crops, or predictions for a table of predicted classes.
    """

    def __init__(self, criteria: tuple, on_change: Callable[[], None], default: str | None = None,
                 limit: int = FOCUS_MAX, what: str = "the views below",
                 support_criteria: tuple = (), support_noun: str = "crops"):
        self.criteria = tuple(criteria)
        self.limit = limit
        self.what = what
        self.on_change = on_change
        self.criterion = default or self.criteria[0][0]
        self.classes: list = []
        self.many = False
        self.support_criteria = frozenset(support_criteria)
        self.min_support = MIN_SUPPORT_DEFAULT
        self.dropdown = ft.Dropdown(
            label="Classes shown", value=self.criterion, width=240, border_radius=8,
            options=[ft.dropdown.Option(key, label) for key, label in self.criteria],
            on_change=self._on_criterion,
        )
        self.min_support_field = ft.TextField(
            label=f"At least N {support_noun}", value=str(self.min_support), width=150, dense=True,
            border_radius=8, keyboard_type=ft.KeyboardType.NUMBER,
            tooltip=f"Only classes with at least this many {support_noun} are ranked by this criterion",
            on_submit=self._on_min_support, on_blur=self._on_min_support,
        )
        self.picker = ClassPicker(self._on_picked, limit=limit)
        self.note = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        # Shown from the moment a change is made until the owner has redrawn:
        # at thousands of classes a redraw takes a second or two, and a
        # control that does nothing visible for that long reads as broken.
        self.busy = ft.Row([ft.ProgressRing(width=14, height=14, stroke_width=2),
                            ft.Text("Updating...", size=12, color=ft.Colors.ON_SURFACE_VARIANT)],
                           spacing=8, visible=False)
        self.root = card(ft.Column([
            ft.Row([self.note, self.busy], spacing=16, wrap=True),
            ft.Row([self.dropdown, self.min_support_field, self.picker.build()], spacing=16, wrap=True,
                   vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=10))
        self.root.visible = False
        self._sync_min_support_field()
        # Changes that arrive while a redraw is running (adding three
        # classes in a row) are folded into one redraw after it.
        self._lock = threading.Lock()
        self._running = False
        self._pending = False

    def _sync_min_support_field(self) -> None:
        self.min_support_field.visible = self.criterion in self.support_criteria
        self.min_support_field.value = str(self.min_support)

    def build(self) -> ft.Control:
        return self.root

    def resolve(self, labels: list, rank: Callable[[str], list]) -> list:
        """The classes to show out of `labels`, every one of them while
        they fit. Call once per render, before drawing."""
        labels = list(labels)
        self.many = len(labels) > self.limit
        self.root.visible = self.many
        if not self.many:
            return labels
        if self.criterion == FOCUS_MANUAL:
            present = set(labels)
            focus = [c for c in self.classes if c in present]
        else:
            focus = list(rank(self.criterion))[: self.limit]
        self.classes = focus
        self.dropdown.value = self.criterion
        self._sync_min_support_field()
        self.picker.set_labels(labels)
        self.picker.set_picked(focus)
        self.note.value = (
            f"{len(labels):,} classes: {len(focus)} of them are shown in {self.what} (at most {self.limit}). "
            f"Pick them by a criterion, or search and add them by hand."
        )
        return focus

    def min_support_for(self, criterion: str) -> int:
        """What a rank() should pass as its minimum support for `criterion`."""
        return self.min_support if criterion in self.support_criteria else 0

    def hide(self) -> None:
        self.many = False
        self.root.visible = False

    def _changed(self) -> None:
        """Run the owner's redraw, at most one at a time: a change made
        while one is running marks it pending, and the running one goes
        round once more when it finishes -- with whatever the latest state
        is by then."""
        with self._lock:
            if self._running:
                self._pending = True
                return
            self._running = True
        self._set_busy(True)
        try:
            while True:
                self.on_change()
                with self._lock:
                    if not self._pending:
                        self._running = False
                        break
                    self._pending = False
        finally:
            with self._lock:
                self._running = False
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.busy.visible = busy
        if self.busy.page:
            self.busy.update()

    def _on_criterion(self, e) -> None:
        self.criterion = e.control.value or self.criteria[0][0]
        self._sync_min_support_field()
        self._changed()

    def _on_min_support(self, e) -> None:
        try:
            value = max(0, int(str(self.min_support_field.value).strip()))
        except ValueError:
            value = self.min_support
        if value == self.min_support:
            self.min_support_field.value = str(value)
            if self.min_support_field.page:
                self.min_support_field.update()
            return
        self.min_support = value
        self._changed()

    def _on_picked(self, picked: list) -> None:
        self.criterion = FOCUS_MANUAL
        self.classes = list(picked)
        self._changed()

    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        return {"criterion": self.criterion, "classes": [str(c) for c in self.classes],
                "min_support": self.min_support}

    def restore_view_state(self, state) -> None:
        """Hand-picked classes that no longer exist are dropped at the next
        resolve()."""
        if not isinstance(state, dict):
            return
        if state.get("criterion") in {key for key, _ in self.criteria}:
            self.criterion = state["criterion"]
            self.dropdown.value = self.criterion
        classes = state.get("classes")
        if isinstance(classes, list):
            self.classes = [c for c in classes if isinstance(c, str)][: self.limit]
        min_support = state.get("min_support")
        if isinstance(min_support, int) and min_support >= 0:
            self.min_support = min_support
        self._sync_min_support_field()


def class_dropdown(labels: list, **kwargs) -> ft.Dropdown:
    """A dropdown over class labels, type-to-filter once there are more
    than FOCUS_MAX of them -- scrolling a menu of two thousand entries to
    find one is not a way to pick a class. Keyword arguments go to
    ft.Dropdown; `options` is built here (key and text both the label, as
    in review_widgets.class_options) unless given."""
    return ft.Dropdown(**searchable_dropdown_kwargs(len(labels)), **{
        "options": [ft.dropdown.Option(str(l)) for l in labels], **kwargs})


def searchable_dropdown_kwargs(n_options: int) -> dict:
    """What turns an existing ft.Dropdown into a type-to-filter one when it
    lists more than FOCUS_MAX options -- for dropdowns whose options are
    rebuilt in place rather than created through class_dropdown."""
    many = n_options > FOCUS_MAX
    return {"enable_filter": many, "editable": many, "menu_height": 400 if many else None}


def make_searchable(dropdown: ft.Dropdown) -> None:
    """Apply searchable_dropdown_kwargs to `dropdown` for its current options."""
    for key, value in searchable_dropdown_kwargs(len(dropdown.options or [])).items():
        setattr(dropdown, key, value)


def short_column_labels(labels: list, chars: int) -> tuple:
    """(short labels, dropped prefix) for the column headers of a grid.

    Headers are a few characters wide, and class names in a big taxonomy
    usually share a long prefix ("species_0412", "species_0413"), so
    cutting each one to its first characters makes every header identical.
    The prefix every label shares is dropped instead, then each is cut to
    `chars`; the caller says which prefix went, and the full label stays in
    the cell's tooltip.
    """
    names = [str(l) for l in labels]
    prefix = os.path.commonprefix(names) if len(names) > 1 else ""
    # Never strip a label down to nothing (one label a prefix of another),
    # and cut only after a separator: "Quercus r" out of "Quercus robur"
    # and "Quercus rubra" would leave "obur" and "ubra".
    prefix = prefix[: min((len(n) for n in names), default=0) - 1] if prefix else ""
    cut = max((i for i, ch in enumerate(prefix) if not ch.isalnum()), default=-1)
    prefix = prefix[: cut + 1]
    short = [n[len(prefix):] for n in names]
    return [s if len(s) <= chars else s[: chars - 1] + "…" for s in short], prefix


def confused_pairs_table(pairs, n_nonzero: int, on_click: Callable[[str, str], None]) -> ft.Control:
    """The rows of class_focus.top_confused_pairs as a table whose rows open
    the crops behind them, plus how many off-diagonal cells there are."""
    if not len(pairs):
        return ft.Text("No confusion at all: every crop is on the diagonal.", size=12,
                       color=ft.Colors.ON_SURFACE_VARIANT)
    rows = [
        ft.DataRow(
            cells=[
                ft.DataCell(ft.Text(str(true_label), size=12)),
                ft.DataCell(ft.Text(str(predicted_label), size=12)),
                ft.DataCell(ft.Text(f"{int(count):,}", size=12)),
                ft.DataCell(ft.Text(f"{share * 100:.1f}%", size=12)),
            ],
            on_select_changed=lambda e, t=true_label, p=predicted_label: on_click(t, p),
        )
        for true_label, predicted_label, count, share in pairs.itertuples(index=False)
    ]
    table = ft.DataTable(
        columns=[ft.DataColumn(ft.Text(h, weight=ft.FontWeight.BOLD, size=12))
                 for h in ("True class", "Predicted as", "Crops", "Of true class")],
        rows=rows, heading_row_height=36, data_row_min_height=32, data_row_max_height=40,
        column_spacing=20, show_checkbox_column=False,
    )
    return ft.Column([
        ft.Text(f"Most confused pairs -- {len(pairs)} of {n_nonzero:,} non-zero off-diagonal cells",
                size=12, weight=ft.FontWeight.BOLD),
        ft.Row([table], scroll=ft.ScrollMode.AUTO),
    ], spacing=6)


def overview_image(ctx, confusion, focus: list, title: str, size: int, cache: dict | None = None) -> ft.Control:
    """overview_figure as a `size`-pixel thumbnail that opens the zoomable
    viewer on click.

    The thumbnail does not name the focus classes -- at thumbnail size the
    names are unreadable anyway -- so it depends on the matrix alone and is
    drawn once per matrix: pass the same `cache` dict on every render and a
    change of focus costs no drawing at all. The enlarged view, drawn only
    when asked for (it takes seconds at two thousand classes), does name
    them. The thumbnail is encoded at the resolution it is shown at, twice
    it for high-density screens, with classes pooled into blocks so a lone
    confusion is not lost in the downscale.
    """
    # Keyed by the matrix object, and holding on to it: an id() alone can be
    # reused by a later matrix once this one is freed.
    key = (id(confusion), size, title)
    hit = cache.get(key) if cache is not None else None
    thumbnail = hit[1] if hit is not None and hit[0] is confusion else None
    if thumbnail is None:
        block = max(1, -(-len(confusion) // size))
        fig = overview_figure(confusion, [], title, block=block)
        side_inches = fig.get_size_inches()[0]
        thumbnail = fig_to_base64(fig, dpi=max(40, int(2 * size / side_inches)))
        if cache is not None:
            cache[key] = (confusion, thumbnail)
    image = ft.Image(src_base64=thumbnail, fit=ft.ImageFit.CONTAIN, width=size, height=size)
    zoom = getattr(ctx, "image_zoom", None) if ctx is not None else None
    if zoom is None:
        return image

    def open_zoom(e):
        # Blocked too, only more finely: at one pixel per class the viewer's
        # fit-to-window view shrinks two thousand classes into a few
        # hundred pixels and every lone confusion fades out -- the very
        # thing the thumbnail had to avoid. ZOOM_BLOCKS blocks across keep
        # each one visible at fit and a crisp square at full zoom.
        fine_block = max(1, -(-len(confusion) // ZOOM_BLOCKS))
        zoom.open(fig_to_base64(overview_figure(confusion, focus, title, block=fine_block), dpi=ZOOM_DPI), title)

    return ft.Container(content=image, ink=True, border_radius=8,
                        tooltip=f"{title} -- click to enlarge (takes a few seconds for many classes)",
                        on_click=open_zoom)
