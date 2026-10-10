"""Shared paginated mosaic of crop images. Used by the gallery's
click-to-inspect dialog, the confusion-matrix drill-down, the similarity
search, the lasso selection, Cleaning, Calibration and the slice breakdown.

The default view is the mosaic itself and nothing else: a tight grid of
crops, two pixels apart, at whatever zoom level the user picked. Metadata
and the review controls are one toggle away rather than always drawn --
they used to take three quarters of every card, which meant three crops on
screen at a time when the reason a user opens this dialog is to *look at
the images*. Deciding on a crop is what you do after you have found it.

Two toggles, both in the toolbar:

* labels -- the class badge over each crop's top-right corner, colored by
  class, on by default because it is small enough to leave on;
* details -- turns every tile into a card carrying its metadata list, the
  "Find similar" button and the decision bar. Cards are wide (a decision
  bar has two dropdowns and a reset button that must stay on one line), so
  this trades most of the grid away; that is the point of it being a mode.

Images are shown at full resolution, no server-side downscaling or lossy
re-encoding -- this is a computer-vision tool, and quietly degrading image
quality for speed would defeat the point. The trade-off instead is a
bounded page: zooming out packs more crops per screen, but never more than
MAX_TILES_PER_PAGE of them, and pages are pulled lazily via page_provider
so a match set of any size is never materialized at once.
"""
from __future__ import annotations

import math
from typing import Callable

import flet as ft
import numpy as np

from lookin.ui.components import (
    build_zoom_overlay, categorical_colors, crop_display, file_to_full_base64,
    open_png_in_tab, save_png_to_disk,
)
from lookin.ui.review_widgets import DecisionBar

# The zoom ladder, in pixels of tile edge. Steps rather than a slider
# because every step has to re-fetch and re-lay out a page of full-res
# crops: a slider would do that on every drag frame.
#
# The top steps are larger than any tile this actually draws: what stops
# the zoom is the window (see _tile_size), not the end of this tuple. A
# ladder that stopped at 400px stopped there on a tall screen too, leaving
# two rows of crops and a zoom-in button that refused to make it one.
ZOOM_STEPS = (64, 88, 120, 160, 220, 300, 400, 540, 720, 960, 1280)
DEFAULT_ZOOM = 2  # 120px -- a crop you can recognize, a screen you can scan

TILE_GAP = 2   # "very little space between the images", i.e. a mosaic
INFO_GAP = 10  # cards need to read as separate objects, tiles do not

# A card in details mode is at least this wide, whatever the zoom: the
# decision bar puts its four verdicts on one row and its two pickers plus
# the reset button on the next, and below this they wrap -- "Uncertain"
# drops onto a line of its own and reads as a different kind of answer
# than the other three.
INFO_MIN_WIDTH = 380
# ...and never wider than this, however much room is left over: two cards
# on a wide screen should not become two billboards.
INFO_MAX_WIDTH = 520
INFO_META_HEIGHT = 76   # the scrollable metadata list under the crop
# Lines in that list at most. A row of a joined frame carries one score
# column per class -- two thousand lines under every crop, and as many
# controls, at that many classes; the columns that say something about a
# crop (its metadata, the prediction) come first.
META_MAX_ROWS = 60
# AppState.merged's derived columns (keys, compared here, never reworded).
PREDICTION_KEYS = ("predicted_label", "predicted_score", "correct")
INFO_BAR_HEIGHT = 150   # the decision bar
SIMILAR_BTN_HEIGHT = 38  # the "Find similar" row, when it is offered
CARD_PADDING = 20       # a card's own padding, left + right
TOOLBAR_HEIGHT = 56     # pager, zoom and the two toggles, above the grid

# What the hosting AlertDialog takes for its own title, actions and insets.
# The mosaic is capped at the page minus this, so the grid never has its
# last row drawn below the bottom edge of the dialog.
DIALOG_CHROME_H = 190
DIALOG_CHROME_W = 90
MIN_GRID_WIDTH = 360
MIN_GRID_HEIGHT = 320
# Below this a crop is not a crop any more, so a window too short for even
# one card stops shrinking the image and lets the grid scroll instead.
MIN_TILE_SIZE = 48

# Crops are served full resolution (see components.file_to_full_base64), so
# a page is a real amount of bytes to push to the browser. Zoomed all the
# way out on a wide screen the grid would hold several hundred; this caps
# it at what the gallery tab's own mosaic already allows (12 x 12).
MAX_TILES_PER_PAGE = 144

BADGE_TEXT_SIZE = 9
# A badge is only ever a corner of a tile, so a long class name is cut
# rather than allowed to run past the crop and over its neighbours. The
# tile cannot do the cutting: it puts no width constraint on the badge, so
# an ellipsis overflow would never trigger, and Flet has no way to measure
# rendered text on the server. Hence a character budget from an estimated
# average glyph width -- approximate on purpose, and only ever wrong by a
# few pixels of a badge.
BADGE_MAX_CHARS = 12
BADGE_MIN_CHARS = 4
BADGE_CHAR_PX = 4.9  # at BADGE_TEXT_SIZE, for the app's font


def _short(label, budget: int) -> str:
    """A class label cut to what fits in a badge. Never used as a key --
    the label the code compares against is always the record's own
    value."""
    text = str(label)
    return text if len(text) <= budget else text[:max(1, budget - 3)] + "..."


class Carousel:
    def __init__(
        self,
        dataset,
        total: int,
        page_provider: Callable[[int, int], list],
        crop_col: str,
        uuid_col: str | None = None,
        highlight_uuid=None,
        ctx=None,
        review_source: str = "",
        suggestions: dict | None = None,
        label_col: str | None = None,
        chrome_height: int = DIALOG_CHROME_H,
    ):
        """total: total number of matching rows.
        page_provider(start, end) -> list[dict]: fetches just that slice,
        so callers backed by a DataFrame (e.g. a confusion-matrix match set)
        never materialize more than one page's worth of rows at a time.

        ctx: the AppContext, optional. When given, clicking a crop shows it
        full size, zoomable, in a layer over the mosaic. This is the one
        place a crop is shown without an existing click action -- the
        gallery tab's own mosaic and the embedding tab's thumbnail rows
        already use their click to open *this* view, so zoom is reached
        from here rather than by taking those over.

        review_source: the provenance string stamped on any decision made
        from this mosaic -- "gallery", "confusion_matrix", "similarity"
        and so on. Every caller passes one, because every caller is a place
        where the user is already looking at a crop and forming an opinion
        about it: a crop you can see is a crop you can relabel. It stays a
        required argument rather than defaulting, so a new call site has to
        say *where* its decisions come from; "60% of our exclusions came
        from the confusion matrix" is only answerable if nobody was allowed
        to skip it.

        suggestions: uuid -> suggested class label, used to pre-fill the
        relabel picker (the nearest other class, or a model's prediction).

        chrome_height: how much of the window the host takes for itself.
        The default covers a dialog whose whole content is this mosaic;
        a caller that puts rows of its own above it (the similarity search
        has a picker and a summary) passes a larger one, or the mosaic
        claims height that is not there and pushes the dialog off screen.
        """
        self.dataset = dataset
        self.total = total
        self.page_provider = page_provider
        self.crop_col = crop_col
        self.uuid_col = uuid_col
        self.highlight_uuid = highlight_uuid
        self.ctx = ctx
        self.review_source = review_source
        self.suggestions = suggestions or {}
        self.label_col = label_col
        self.chrome_height = chrome_height
        self.page_index = 0
        self.show_labels = True
        self.show_info = False
        # Positions into the caller's order, shuffled; None while the crops
        # are shown in that order (see _toggle_shuffle).
        self._order: np.ndarray | None = None
        self.zoom_index = DEFAULT_ZOOM
        self.zoom_index = self._initial_zoom()
        self._colors: dict | None = None
        self._zoom_b64: str | None = None
        self._zoom_title = "image"
        self.stack: ft.Stack | None = None
        self.zoom_layer, self.zoom_image = build_zoom_overlay(
            self._grid_width(), self._view_height(),
            on_back=self._hide_zoom, on_save=self._save_zoom, on_open=self._open_zoom,
        )

        self.grid = ft.Row(wrap=True, spacing=TILE_GAP, run_spacing=TILE_GAP,
                           alignment=ft.MainAxisAlignment.START)
        self.grid_box = ft.Container(width=self._grid_width(), content=self.grid)
        # The grid keeps its natural height and the area around it is the
        # one with a fixed size: a container clamped to the space available
        # clips whatever does not fit, and the scrollable column above it
        # then sees a child that already fits and offers no scrollbar --
        # which is how a zoomed-in page lost its last row entirely.
        self.scroll_area = ft.Column([self.grid_box], scroll=ft.ScrollMode.AUTO)
        self.page_label = ft.Text(size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.prev_btn = ft.IconButton(ft.Icons.CHEVRON_LEFT, tooltip="Previous page", on_click=self._prev)
        self.next_btn = ft.IconButton(ft.Icons.CHEVRON_RIGHT, tooltip="Next page", on_click=self._next)
        self.zoom_out_btn = ft.IconButton(ft.Icons.ZOOM_OUT, tooltip="Smaller images", on_click=self._zoom_out)
        self.zoom_in_btn = ft.IconButton(ft.Icons.ZOOM_IN, tooltip="Bigger images", on_click=self._zoom_in)
        self.labels_btn = ft.IconButton(on_click=self._toggle_labels)
        self.info_btn = ft.IconButton(on_click=self._toggle_info)
        self.shuffle_btn = ft.IconButton(on_click=self._toggle_shuffle)
        self._sync_toggles()
        self._render()

    # ---- what the mosaic is allowed to show --------------------------------

    def _reviewable(self) -> bool:
        return bool(
            self.review_source and self.ctx is not None
            and getattr(self.ctx.state, "review", None) is not None
            and self.uuid_col is not None
        )

    def _offers_similarity(self) -> bool:
        search = getattr(self.ctx, "similarity", None) if self.ctx is not None else None
        return bool(search is not None and self.review_source != "similarity"
                    and self.uuid_col and search.is_available())

    # ---- geometry ----------------------------------------------------------

    def _page_size(self) -> tuple:
        page = getattr(self.ctx, "page", None) if self.ctx is not None else None
        width = getattr(page, "width", None) or 1440
        height = getattr(page, "height", None) or 920
        return float(width), float(height)

    def _grid_width(self) -> int:
        page_width, _ = self._page_size()
        return int(max(MIN_GRID_WIDTH, page_width - DIALOG_CHROME_W))

    def _grid_height(self) -> int:
        """The scrollable area the tiles sit in: the dialog minus its own
        chrome, minus the toolbar above the grid."""
        _, page_height = self._page_size()
        available = max(MIN_GRID_HEIGHT, page_height - self.chrome_height)
        return int(available - TOOLBAR_HEIGHT)

    def _tile_size(self, zoom: int | None = None) -> int:
        """The crop's edge: the zoom step, bounded by what the dialog can
        actually show.

        Bounded on *both* axes. Width, or a single tile on a narrow window
        is laid out past the right edge of the dialog. Height, because
        zooming in is a request to fill the window with the crop, not to
        push it out of the bottom of it: past this point a step up would
        only move pixels off screen.

        In details mode the card's own controls are taken out of the
        height first. They are what the user turned the mode on for, and
        the crop is the compressible part -- it is still click-to-zoom at
        full resolution, and a decision bar below the bottom edge is
        indistinguishable from the feature not existing.
        """
        step = ZOOM_STEPS[self.zoom_index if zoom is None else zoom]
        by_width = self._grid_width() - TILE_GAP
        by_height = self._grid_height() - self._info_extra_height()
        return int(max(MIN_TILE_SIZE, min(step, by_width, by_height)))

    def _gap(self) -> int:
        return INFO_GAP if self.show_info else TILE_GAP

    def _info_extra_height(self) -> int:
        """Everything a card carries under the crop. Zero in mosaic mode,
        where a tile is the crop and nothing else."""
        if not self.show_info:
            return 0
        height = INFO_META_HEIGHT + CARD_PADDING
        if self._reviewable():
            height += INFO_BAR_HEIGHT
        if self._offers_similarity():
            height += SIMILAR_BTN_HEIGHT
        return height

    def _cell_size(self, zoom: int | None = None) -> tuple:
        """(width, height) of one grid cell, tile or card.

        A tile is exactly the zoom step, leftover width and all -- a mosaic
        that stretched its images to divide the row evenly would be showing
        them at a size the user did not pick. A card instead shares out
        what is left: at a fixed minimum width, two cards in a 1100px
        dialog left a third of it empty.
        """
        tile = self._tile_size(zoom)
        if not self.show_info:
            return tile, tile
        gap = self._gap()
        columns = self._columns(zoom)
        width = (self._grid_width() - gap * (columns - 1)) / columns
        width = int(max(INFO_MIN_WIDTH, min(INFO_MAX_WIDTH, width)))
        return min(width, self._grid_width()), int(tile + self._info_extra_height())

    def _columns(self, zoom: int | None = None) -> int:
        """How many cells fit across. Measured against the *minimum* a cell
        can be, so _cell_size can hand the remainder back to the cards
        without the two disagreeing about how many there are."""
        gap = self._gap()
        minimum = INFO_MIN_WIDTH if self.show_info else self._tile_size(zoom)
        return max(1, int((self._grid_width() + gap) // (minimum + gap)))

    def _rows(self, zoom: int | None = None) -> int:
        _, cell_h = self._cell_size(zoom)
        gap = self._gap()
        return max(1, int((self._grid_height() + gap) // (cell_h + gap)))

    def _tiles_per_page(self, zoom: int | None = None) -> int:
        return max(1, min(MAX_TILES_PER_PAGE, self._columns(zoom) * self._rows(zoom)))

    def _initial_zoom(self) -> int:
        """How far out the mosaic opens: far enough that everything it has
        to show is on one page, and no further.

        The size of the match set is the whole difference between the two
        things this view is asked to do. A confusion-matrix cell holding
        four crops has no reason to draw them as thumbnails in a window
        that could show them full size, and a cell holding nine hundred
        has no business trying -- one fixed default was wrong for both.
        Past the bottom of the ladder there is nothing left to give, so a
        very large set opens as dense as the mosaic goes and pages.
        """
        if self.total <= 0:
            return DEFAULT_ZOOM
        for index in reversed(range(len(ZOOM_STEPS))):
            if self._tiles_per_page(index) >= self.total:
                return index
        return 0

    def _total_pages(self) -> int:
        return max(1, math.ceil(self.total / self._tiles_per_page()))

    def _view_height(self) -> int:
        """The whole view, toolbar included -- and a constant, for a given
        window, whatever the zoom is or how many crops there are.

        It used to shrink to the tiles on the page, which made the dialog
        resize itself under the cursor on every zoom step. Worse, the
        full-size viewer is a layer over this same area (a second dialog
        cannot be stacked on the first, see build_zoom_overlay), so
        "full size" inherited that changing size and was never bigger than
        whatever the mosaic happened to need. A fixed frame is one the
        viewer can fill.
        """
        return self._grid_height() + TOOLBAR_HEIGHT

    # ---- class colors ------------------------------------------------------

    def _class_colors(self) -> dict:
        """One color per class label, assigned by the dataset's own label
        order -- the same mapping the projection's by-class legend uses, so
        a class is the same color wherever the user meets it."""
        if self._colors is None:
            labels = []
            if self.dataset is not None:
                try:
                    labels = [str(label) for label in self.dataset.labels()]
                except Exception:  # a dataset without labels is not a rendering error
                    labels = []
            self._colors = dict(zip(labels, categorical_colors(len(labels))))
        return self._colors

    def _class_color(self, label) -> str:
        return self._class_colors().get(str(label), ft.Colors.ON_SURFACE)

    # ---- rendering ---------------------------------------------------------

    def _render(self):
        per_page = self._tiles_per_page()
        start = self.page_index * per_page
        end = min(start + per_page, self.total)
        records = self._fetch(start, end) if self.total else []

        self.grid.spacing = self.grid.run_spacing = self._gap()
        self.grid_box.width = self._grid_width()
        self.scroll_area.height = self._grid_height()
        if records:
            self.grid.controls = [self._build_tile(rec) for rec in records]
        else:
            self.grid.controls = [ft.Text("No images match.", italic=True)]

        total_pages = self._total_pages()
        self.page_label.value = (
            f"{start + 1}-{end} of {self.total:,}  ·  page {self.page_index + 1}/{total_pages}"
            if self.total else "0 images"
        )
        self.prev_btn.disabled = self.page_index <= 0
        self.next_btn.disabled = self.page_index >= total_pages - 1
        self.zoom_out_btn.disabled = self.zoom_index <= 0
        # Greyed out once the step above would be clamped back to this same
        # size by the window: a button that visibly does nothing reads as
        # broken, where a disabled one reads as "this is as big as it gets".
        self.zoom_in_btn.disabled = (self.zoom_index >= len(ZOOM_STEPS) - 1
                                     or self._tile_size(self.zoom_index + 1) <= self._tile_size())

    def _badge(self, rec: dict) -> ft.Control | None:
        """The class label over the crop's top-right corner, in the class
        color. Where the row carries a prediction (a confusion-matrix cell,
        a calibration bucket, anything built from AppState.merged) the
        badge reads "truth -> prediction", each side in its own class
        color: which way a crop is wrong is the whole question there, and
        it is a wasted trip to the details toggle to find out.

        The arrow is ASCII on purpose. The app bundles its own font so it
        can run with no network (see _offline_assets.py), and that font has
        no U+2192 -- a real arrow renders as a tofu box.
        """
        if not self.show_labels or not self.label_col:
            return None
        truth = rec.get(self.label_col)
        if truth is None:
            return None
        predicted = rec.get("predicted_label")
        budget = self._badge_budget(predicted is not None)
        parts = [ft.Text(_short(truth, budget), size=BADGE_TEXT_SIZE, weight=ft.FontWeight.BOLD,
                         color=self._class_color(truth), max_lines=1)]
        if predicted is not None:
            parts.append(ft.Text(f"-> {_short(predicted, budget)}", size=BADGE_TEXT_SIZE,
                                 color=self._class_color(predicted), max_lines=1))
        return ft.Container(
            content=ft.Row(parts, spacing=3, tight=True),
            top=2, right=2,
            padding=ft.padding.symmetric(horizontal=4, vertical=1),
            border_radius=4,
            # A backdrop, because a class color legible on the app's
            # surface is not legible over an arbitrary crop.
            bgcolor=ft.Colors.with_opacity(0.78, ft.Colors.SURFACE_CONTAINER_HIGHEST),
        )

    def _badge_budget(self, with_prediction: bool) -> int:
        """How many characters each side of the badge gets, from the width
        of the tile it has to stay inside. Zoomed right out a badge is
        nearly as wide as the crop, so the names are cut hard -- the full
        text is a hover away, and the zoom is one click away."""
        usable = self._tile_size() - 12  # the badge's own padding and margins
        if with_prediction:
            usable -= 3 * BADGE_CHAR_PX  # the "-> " between the two names
            usable /= 2
        return int(max(BADGE_MIN_CHARS, min(BADGE_MAX_CHARS, usable / BADGE_CHAR_PX)))

    def _tooltip(self, rec: dict) -> str:
        """What the tile says on hover -- in mosaic mode this is the only
        place its identity is written down."""
        parts = []
        if self.label_col and rec.get(self.label_col) is not None:
            parts.append(str(rec[self.label_col]))
        if rec.get("predicted_label") is not None:
            parts.append(f"predicted {rec['predicted_label']}")
        if self.uuid_col and rec.get(self.uuid_col) is not None:
            parts.append(str(rec[self.uuid_col]))
        parts.append("click to view full size")
        return "  ·  ".join(parts)

    def _build_tile(self, rec: dict) -> ft.Control:
        img_path = self.dataset.resolve_crop_path(rec[self.crop_col])
        b64 = file_to_full_base64(img_path, *crop_display(self.ctx, rec))
        tile = self._tile_size()
        cell_w, _ = self._cell_size()
        # In details mode the card is wider than the zoom step (a decision
        # bar has a width of its own), so the crop is given the whole card
        # to sit in rather than being pinned to the left of it.
        img_w = cell_w - CARD_PADDING if self.show_info else tile
        highlighted = (self.highlight_uuid is not None and self.uuid_col is not None
                       and rec.get(self.uuid_col) == self.highlight_uuid)

        # CONTAIN, not COVER: filling the square would crop pixels out of
        # the image, and hiding part of a crop is the one thing this view
        # must never do.
        image: ft.Control = ft.Image(
            src_base64=b64, width=img_w, height=tile,
            fit=ft.ImageFit.CONTAIN, border_radius=8 if self.show_info else 2,
        )
        badge = self._badge(rec)
        if badge is not None:
            image = ft.Stack([image, badge], width=img_w, height=tile)
        if self.ctx is not None:
            zoom_title = (str(rec.get(self.uuid_col))
                          if self.uuid_col and rec.get(self.uuid_col) is not None else img_path.name)
            image = ft.Container(
                content=image, ink=True, border_radius=8 if self.show_info else 2,
                tooltip=self._tooltip(rec),
                on_click=lambda e, b=b64, t=zoom_title: self._show_zoom(b, t),
            )

        if not self.show_info:
            return ft.Container(
                width=tile, height=tile, content=image, border_radius=2,
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                border=ft.border.all(2, ft.Colors.PRIMARY) if highlighted else None,
            )
        return self._build_card(rec, image, cell_w, highlighted)

    def _build_card(self, rec: dict, image: ft.Control, cell_w: int, highlighted: bool) -> ft.Control:
        """A tile with everything on it: the metadata list, "Find similar"
        and the decision bar. Only built in details mode."""
        meta_rows = []
        items = [(k, v) for k, v in rec.items() if k != self.crop_col]
        # The prediction first: a joined frame puts it after every score
        # column, where the cap below would cut it off.
        items.sort(key=lambda kv: kv[0] not in PREDICTION_KEYS)
        for k, v in items[:META_MAX_ROWS]:
            display_v = f"{v:.4f}" if isinstance(v, float) else str(v)
            meta_rows.append(ft.Row([
                ft.Container(ft.Text(k, size=10, weight=ft.FontWeight.BOLD), width=112),
                ft.Text(display_v, size=10, selectable=True, expand=True),
            ]))
        if len(items) > META_MAX_ROWS:
            meta_rows.append(ft.Text(f"... and {len(items) - META_MAX_ROWS:,} more column(s)",
                                     size=10, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))

        card_controls = [
            image,
            ft.Column(meta_rows, spacing=3, scroll=ft.ScrollMode.AUTO, height=INFO_META_HEIGHT),
        ]
        similar_btn = self._similarity_button(rec)
        if similar_btn is not None:
            card_controls.append(similar_btn)
        if self._reviewable():
            uid = rec.get(self.uuid_col)
            label = rec.get(self.label_col) if self.label_col else None
            card_controls.append(ft.Divider(height=1))
            card_controls.append(DecisionBar(
                self.ctx, uid, label, suggestion=str(self.suggestions.get(str(uid), "")),
                source=self.review_source, compact=True,
                width=cell_w - CARD_PADDING,
            ).build())
        return ft.Container(
            width=cell_w,
            content=ft.Column(card_controls, spacing=8, tight=True),
            padding=10,
            border_radius=10,
            border=ft.border.all(3, ft.Colors.PRIMARY) if highlighted else ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        )

    def _similarity_button(self, rec: dict) -> ft.Control | None:
        """"Find similar" -- shown only where it can actually answer.

        Needs a uuid to search from and an experiment with embeddings
        attached; without either, the button would open a dialog whose only
        content is an explanation of why it is empty. It is also hidden
        inside the similarity dialog's own mosaic, where it would just
        re-ask the same question one crop over.
        """
        search = getattr(self.ctx, "similarity", None) if self.ctx is not None else None
        uid = rec.get(self.uuid_col) if self.uuid_col else None
        if search is None or uid is None or self.review_source == "similarity":
            return None
        if not search.is_available():
            return None
        return ft.TextButton(
            "Find similar", icon=ft.Icons.TRAVEL_EXPLORE,
            tooltip="The crops nearest to this one in the model's embedding space",
            on_click=lambda e, uid=str(uid): search.open(uid),
            style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=8, vertical=2)),
        )

    # ---- toolbar -----------------------------------------------------------

    def _fetch(self, start: int, end: int) -> list:
        """Rows `start`..`end` of what is shown -- the caller's slice, or
        one row at a time through the shuffle. page_provider only knows
        contiguous slices, and a page is at most MAX_TILES_PER_PAGE rows."""
        if self._order is None:
            return self.page_provider(start, end)
        records = []
        for i in self._order[start:end].tolist():
            records.extend(self.page_provider(i, i + 1))
        return records

    def _sync_toggles(self):
        shuffled = self._order is not None
        self.shuffle_btn.icon = ft.Icons.SHUFFLE_ON if shuffled else ft.Icons.SHUFFLE
        self.shuffle_btn.icon_color = ft.Colors.PRIMARY if shuffled else ft.Colors.ON_SURFACE_VARIANT
        self.shuffle_btn.tooltip = "Back to the original order" if shuffled else "Show the crops in a random order"
        self.labels_btn.icon = ft.Icons.LABEL if self.show_labels else ft.Icons.LABEL_OFF_OUTLINED
        self.labels_btn.icon_color = ft.Colors.PRIMARY if self.show_labels else ft.Colors.ON_SURFACE_VARIANT
        self.labels_btn.tooltip = "Hide the class labels" if self.show_labels else "Show the class labels"
        self.info_btn.icon = ft.Icons.INFO if self.show_info else ft.Icons.INFO_OUTLINE
        self.info_btn.icon_color = ft.Colors.PRIMARY if self.show_info else ft.Colors.ON_SURFACE_VARIANT
        self.info_btn.tooltip = (
            "Back to the plain mosaic" if self.show_info
            else "Show the metadata and the review controls on every crop"
        )

    def _keep_position(self, change) -> None:
        """Apply a change of zoom or mode without losing the user's place.

        The page holds a different number of crops afterwards, so the page
        *index* means something different too: zooming out on page 9 of 30
        used to land somewhere unrelated. The first crop currently on
        screen is the anchor, and the new page is whichever one holds it.
        """
        anchor = self.page_index * self._tiles_per_page()
        change()
        self.page_index = anchor // self._tiles_per_page()
        self._render()
        self._push_update()

    def _zoom_in(self, e):
        # Same guard as the button's disabled state: a step the window
        # would clamp straight back is not taken, so zooming out again
        # does not need the clicks back that did nothing on the way up.
        if (self.zoom_index < len(ZOOM_STEPS) - 1
                and self._tile_size(self.zoom_index + 1) > self._tile_size()):
            self._keep_position(lambda: setattr(self, "zoom_index", self.zoom_index + 1))

    def _zoom_out(self, e):
        if self.zoom_index > 0:
            self._keep_position(lambda: setattr(self, "zoom_index", self.zoom_index - 1))

    def _toggle_labels(self, e):
        self.show_labels = not self.show_labels
        self._sync_toggles()
        self._render()
        self._push_update()

    def _toggle_info(self, e):
        def change():
            self.show_info = not self.show_info
            self._sync_toggles()

        self._keep_position(change)

    def _toggle_shuffle(self, e):
        """Random order on, or back to the caller's. Each time it is turned
        on is a new draw. Back to the first page either way: the crops that
        were on screen are scattered over every page now."""
        self._order = None if self._order is not None else np.random.default_rng().permutation(self.total)
        self.page_index = 0
        self._sync_toggles()
        self._render()
        self._push_update()
        if self.shuffle_btn.page:
            self.shuffle_btn.update()

    def _prev(self, e):
        if self.page_index > 0:
            self.page_index -= 1
            self._render()
            self._push_update()

    def _next(self, e):
        if self.page_index < self._total_pages() - 1:
            self.page_index += 1
            self._render()
            self._push_update()

    def _push_update(self):
        if self.grid.page:
            self.scroll_area.update()
            self.page_label.update()
            self.prev_btn.update()
            self.next_btn.update()
            self.zoom_out_btn.update()
            self.zoom_in_btn.update()
            self.labels_btn.update()
            self.info_btn.update()

    # ---- full-size zoom, layered over the mosaic ----
    #
    # Toggling one already-mounted layer, rather than opening a second
    # dialog: the containing dialog stays open and untouched, so there is
    # nothing to tear down and rebuild (the crops are full resolution and
    # re-decoding a page of them is exactly what made the old version lag),
    # and no dialog route is closed and reopened behind Flutter's back.

    def _show_zoom(self, b64: str, title: str) -> None:
        self._zoom_b64, self._zoom_title = b64, title
        self.zoom_image.src_base64 = b64
        status = self.zoom_layer.data
        status.value = title
        status.color = ft.Colors.ON_SURFACE_VARIANT
        self.zoom_layer.visible = True
        if self.zoom_layer.page:
            self.zoom_layer.update()

    def _hide_zoom(self, e=None) -> None:
        self.zoom_layer.visible = False
        if self.zoom_layer.page:
            self.zoom_layer.update()

    def _save_zoom(self, e=None) -> None:
        if self._zoom_b64:
            self._report_zoom(*save_png_to_disk(self._zoom_b64, self._zoom_title, self.ctx))

    def _open_zoom(self, e=None) -> None:
        """The crop as a real image in a browser tab, where it can be
        copied or saved on the user's own machine rather than the server's
        (see ui/image_export.py)."""
        if self._zoom_b64:
            self._report_zoom(*open_png_in_tab(self.ctx, self._zoom_b64, self._zoom_title))

    def _report_zoom(self, ok: bool, message: str) -> None:
        status = self.zoom_layer.data
        status.value = message
        status.color = ft.Colors.GREEN if ok else ft.Colors.RED
        if status.page:
            status.update()

    def build(self) -> ft.Control:
        toolbar = ft.Row(
            [
                self.prev_btn, self.page_label, self.next_btn,
                ft.VerticalDivider(width=12),
                self.zoom_out_btn, self.zoom_in_btn,
                ft.VerticalDivider(width=12),
                self.labels_btn, self.info_btn, self.shuffle_btn,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4, wrap=False,
        )
        # The grid scrolls vertically: in details mode one card can be
        # taller than the dialog, and the alternative is a decision bar the
        # user can see but cannot reach.
        mosaic_view = ft.Column([toolbar, self.scroll_area], spacing=4)
        if self.ctx is None:
            return mosaic_view
        self.stack = ft.Stack([mosaic_view, self.zoom_layer],
                              width=self._grid_width(), height=self._view_height())
        return self.stack


def build_carousel_from_list(
    dataset, records: list, crop_col: str, uuid_col: str | None = None,
    highlight_uuid=None, ctx=None, review_source: str = "", suggestions: dict | None = None,
    label_col: str | None = None,
) -> ft.Control:
    """Convenience wrapper for callers that already have a small, fully
    materialized list of row dicts (e.g. one gallery mosaic page)."""
    carousel = Carousel(
        dataset, len(records), lambda start, end: records[start:end],
        crop_col, uuid_col, highlight_uuid, ctx, review_source, suggestions, label_col,
    )
    return carousel.build()
