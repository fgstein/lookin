"""Mosaic geometry and the crop-click routing.

Two things reported from using the app, both of which look like nothing in
a unit test and like a broken feature on screen:

* a page of tiles laid out taller than the dialog draws its last row -- and
  in details mode its decision buttons -- below the bottom edge: present in
  the layout, unreachable by the user;
* a crop shown outside the mosaic has no way to relabel it, which is how
  clicking a point in the embedding projection became the one place in the
  app where looking at a crop was a dead end -- so a tap goes to the
  mosaic, not to the bare viewer and not to a side panel.

Neither needs a browser to pin down: both are arithmetic and wiring.
"""
from __future__ import annotations

import numpy as np
import pytest

from lookin.ui.carousel import (
    BADGE_MAX_CHARS, CARD_PADDING, DEFAULT_ZOOM, DIALOG_CHROME_H, INFO_MAX_WIDTH,
    INFO_MIN_WIDTH, MAX_TILES_PER_PAGE, MIN_TILE_SIZE, TILE_GAP, TOOLBAR_HEIGHT,
    ZOOM_STEPS, Carousel,
)
from lookin.ui.review_widgets import MIN_PICKER_WIDTH, PICKER_GAP, RESET_BTN_WIDTH, DecisionBar


class FakePage:
    def __init__(self, width, height):
        self.width, self.height = width, height


class FakeDataset:
    def __init__(self, labels=("cat", "dog", "bird")):
        self._labels = list(labels)

    def labels(self):
        return self._labels


class FakeCtx:
    """Enough of an AppContext for the geometry: a page, and a review store
    so the decision bar counts towards the height of a details card."""

    def __init__(self, width=1440, height=920, reviewable=True, similarity=None):
        self.page = FakePage(width, height)
        self.similarity = similarity

        class State:
            review = object() if reviewable else None
            dataset = None

        self.state = State()


def _mosaic(width=1440, height=920, reviewable=True, similarity=None,
            zoom=DEFAULT_ZOOM, info=False, total=1000) -> Carousel:
    """A Carousel with only the fields the geometry reads -- no Flet
    controls are built, so this runs without a page. `zoom=None` leaves the
    view to pick its own opening step, the way __init__ does."""
    view = Carousel.__new__(Carousel)
    view.ctx = FakeCtx(width, height, reviewable, similarity)
    view.review_source = "test" if reviewable else ""
    view.uuid_col = "uuid"
    view.label_col = "label"
    view.dataset = FakeDataset()
    view.total = total
    view.page_index = 0
    view.show_labels = True
    view.show_info = info
    view._colors = None
    view.stack = None
    view.chrome_height = DIALOG_CHROME_H
    view.zoom_index = view._initial_zoom() if zoom is None else zoom
    return view


# ---- the default view is the mosaic ------------------------------------

def test_the_mosaic_opens_without_the_details():
    """The complaint this whole view answers: the review controls took most
    of the card, so three crops filled the dialog."""
    view = _mosaic()
    assert view.show_info is False
    assert view.show_labels is True
    assert view._cell_size() == (view._tile_size(), view._tile_size()), "a tile is the crop, nothing else"
    assert view._gap() == TILE_GAP


def test_the_mosaic_shows_far_more_crops_than_the_filmstrip_did():
    """Three at a time was the old page size."""
    assert _mosaic()._tiles_per_page() > 3


# ---- zooming -----------------------------------------------------------

def test_zooming_in_grows_the_tile_and_thins_the_page():
    """On a window big enough that no step is clamped back by it."""
    for step in range(len(ZOOM_STEPS) - 1):
        smaller = _mosaic(width=3000, height=2000, zoom=step)
        bigger = _mosaic(width=3000, height=2000, zoom=step + 1)
        assert bigger._tile_size() > smaller._tile_size()
        assert bigger._tiles_per_page() <= smaller._tiles_per_page()


def test_zooming_all_the_way_in_gets_to_a_single_row():
    """The ladder used to stop at 400px, which on a tall screen left two
    rows of crops and a zoom-in button that would not make it one."""
    for height, info in [(h, i) for h in (760, 920, 1200, 1600) for i in (False, True)]:
        view = _mosaic(width=1440, height=height, total=100, zoom=0, info=info)
        view._render = lambda: None
        view._push_update = lambda: None
        for _ in range(len(ZOOM_STEPS) * 2):
            view._zoom_in(None)
        assert view._rows() == 1, f"still {view._rows()} rows at {height}px, info={info}"


def test_a_page_is_bounded_however_far_out_it_zooms():
    """Every tile is a full-resolution crop pushed to the browser, so the
    grid is capped rather than filling a 4K screen with them."""
    for width, height in ((1440, 920), (2560, 1440), (3840, 2160)):
        view = _mosaic(width=width, height=height, zoom=0)
        assert 1 <= view._tiles_per_page() <= MAX_TILES_PER_PAGE


def test_a_tile_never_grows_wider_than_the_grid():
    view = _mosaic(width=420, zoom=len(ZOOM_STEPS) - 1)
    assert view._tile_size() <= view._grid_width() - TILE_GAP
    assert view._columns() >= 1


def test_zooming_keeps_the_crops_the_user_was_looking_at():
    """Zooming out on page 9 of 30 used to land on an unrelated page: the
    index survived and the page size under it did not."""
    view = _mosaic()
    view.page_index = 8
    anchor = view.page_index * view._tiles_per_page()
    view._render = lambda: None
    view._push_update = lambda: None

    view._zoom_out(None)
    start = view.page_index * view._tiles_per_page()
    assert start <= anchor < start + view._tiles_per_page()


# ---- the opening zoom follows the size of the match set ----------------

def test_one_crop_opens_as_large_as_the_window_allows():
    """A confusion-matrix cell with a single crop in it used to open as a
    120px thumbnail in the middle of an empty dialog."""
    assert _mosaic(total=1, zoom=None).zoom_index == len(ZOOM_STEPS) - 1


def test_a_whole_match_set_opens_on_one_page_when_it_can():
    for total in (1, 2, 6, 12, 24, 40):
        view = _mosaic(total=total, zoom=None)
        assert view._tiles_per_page() >= total, f"{total} crops did not fit one page"


def test_a_bigger_set_opens_further_out():
    steps = [_mosaic(total=total, zoom=None).zoom_index for total in (1, 6, 30, 120, 900)]
    assert steps == sorted(steps, reverse=True)
    assert steps[0] > steps[-1]


def test_a_set_too_big_for_the_ladder_opens_as_dense_as_it_goes():
    assert _mosaic(total=5000, zoom=None).zoom_index == 0


def test_an_empty_result_still_has_a_zoom():
    assert _mosaic(total=0, zoom=None).zoom_index == DEFAULT_ZOOM


# ---- zooming in never pushes the crop off the window -------------------

@pytest.mark.parametrize("height", [600, 720, 800, 920, 1080])
@pytest.mark.parametrize("info", [False, True])
def test_the_tile_is_bounded_by_the_window_not_only_by_the_ladder(height, info):
    """The report this is here for: zoomed all the way in, the crop was
    drawn past the bottom of the dialog."""
    view = _mosaic(height=height, info=info, zoom=len(ZOOM_STEPS) - 1)
    tile = view._tile_size()
    assert tile <= view._grid_width() - TILE_GAP
    assert tile <= max(MIN_TILE_SIZE, view._grid_height() - view._info_extra_height())


def test_a_whole_card_fits_the_window_unless_the_crop_is_already_minimal():
    """Details mode pays for its controls out of the crop, so the card as
    a whole stays inside the dialog."""
    for height in (720, 920, 1080):
        view = _mosaic(height=height, info=True, zoom=len(ZOOM_STEPS) - 1)
        assert view._cell_size()[1] <= view._grid_height() or view._tile_size() == MIN_TILE_SIZE


def test_zooming_in_stops_where_it_stops_doing_anything():
    """A button that visibly does nothing reads as broken -- and every
    click that did nothing on the way up would need one back on the way
    down."""
    view = _mosaic(height=600, zoom=0)
    view._render = lambda: None
    view._push_update = lambda: None
    for _ in range(len(ZOOM_STEPS) * 2):
        view._zoom_in(None)

    top = view._tile_size()
    next_step = min(view.zoom_index + 1, len(ZOOM_STEPS) - 1)
    assert view._tile_size(next_step) <= top, "it stopped before the gains did"
    assert view._tile_size(view.zoom_index - 1) < top, "and it stopped after them"


# ---- the details mode ---------------------------------------------------

def test_details_mode_widens_the_cell_enough_for_the_decision_bar():
    cell_w, _ = _mosaic(info=True, zoom=0)._cell_size()
    assert cell_w >= INFO_MIN_WIDTH


@pytest.mark.parametrize("width", [800, 1024, 1280, 1440, 1920, 2560])
def test_the_cards_share_out_the_width_the_row_does_not_use(width):
    """At a fixed minimum width, two cards in a 1100px dialog left a third
    of it empty -- and the row must still hold the number of columns the
    layout thinks it does."""
    view = _mosaic(width=width, info=True)
    cell_w, _ = view._cell_size()
    columns = view._columns()
    assert INFO_MIN_WIDTH <= cell_w <= INFO_MAX_WIDTH
    assert columns * cell_w + (columns - 1) * view._gap() <= view._grid_width() + 1


def test_details_mode_pays_for_the_controls_in_height():
    tight, detailed = _mosaic(), _mosaic(info=True)
    assert detailed._cell_size()[1] > tight._cell_size()[1]
    assert detailed._tiles_per_page() < tight._tiles_per_page()


def test_a_read_only_mosaic_spends_less_height_on_a_card():
    assert _mosaic(info=True, reviewable=False)._cell_size()[1] < _mosaic(info=True)._cell_size()[1]


def test_the_find_similar_row_is_paid_for_in_the_card():
    """It was added without being accounted for, which is what pushed the
    decision bar off the bottom in the first place."""
    class Search:
        @staticmethod
        def is_available():
            return True

    without = _mosaic(info=True)
    with_search = _mosaic(info=True, similarity=Search())
    assert with_search._cell_size()[1] > without._cell_size()[1]


# ---- nothing is drawn below the bottom edge of the dialog ---------------

@pytest.mark.parametrize("height", [600, 720, 800, 920, 1080])
@pytest.mark.parametrize("info", [False, True])
def test_the_grid_stays_inside_the_dialog(height, info):
    """The failure this is here for: at 720px the card was laid out taller
    than the dialog and the class picker was drawn below its bottom edge."""
    view = _mosaic(height=height, info=info, zoom=len(ZOOM_STEPS) - 1)
    assert view._grid_height() + TOOLBAR_HEIGHT == view._view_height()
    assert view._view_height() <= max(320, height - DIALOG_CHROME_H)


def test_the_view_keeps_one_size_whatever_it_is_showing():
    """It used to shrink to the tiles on the page, so the dialog resized
    itself under the cursor on every zoom step -- and the full-size viewer,
    which is a layer over this same area, was never bigger than whatever
    the mosaic happened to need."""
    sizes = {
        (_mosaic(total=total, zoom=zoom, info=info)._grid_width(),
         _mosaic(total=total, zoom=zoom, info=info)._view_height())
        for total in (1, 6, 900)
        for zoom in range(len(ZOOM_STEPS))
        for info in (False, True)
    }
    assert len(sizes) == 1


def test_the_full_size_viewer_covers_the_whole_view():
    """The viewer is a layer over the mosaic rather than a dialog of its
    own (see build_zoom_overlay), so anything the layer does not cover is
    mosaic showing through underneath it."""
    view = Carousel(FakeDataset(), 0, lambda start, end: [], "crop", "uuid",
                    ctx=FakeCtx(1440, 920), review_source="test", label_col="label")
    stack = view.build()

    assert (view.zoom_layer.width, view.zoom_layer.height) == (stack.width, stack.height)
    assert stack.height == view._view_height()
    assert view.zoom_image.width <= view.zoom_layer.width
    assert view.zoom_image.height <= view.zoom_layer.height


def test_a_host_with_rows_of_its_own_gets_a_shorter_view():
    """The similarity dialog puts a picker and a summary above the mosaic;
    sized to the window regardless, the mosaic would run off the bottom."""
    embedded, alone = _mosaic(), _mosaic()
    embedded.chrome_height = DIALOG_CHROME_H + 170
    assert embedded._view_height() < alone._view_height()


# ---- the class badge ----------------------------------------------------

def test_the_badge_is_the_class_label_in_its_class_color():
    view = _mosaic()
    badge = view._badge({"label": "dog"})
    row = badge.content.controls
    assert [text.value for text in row] == ["dog"]
    assert row[0].color == view._class_color("dog")


def test_the_badge_carries_the_prediction_where_there_is_one():
    """A confusion-matrix cell or a calibration bucket: which way the crop
    is wrong is the question being asked there."""
    row = _mosaic()._badge({"label": "cat", "predicted_label": "dog"}).content.controls
    assert [text.value for text in row] == ["cat", "-> dog"]
    assert row[0].color != row[1].color


def test_the_badge_arrow_stays_ascii():
    """The app bundles its own font so it can run offline, and that font
    has no U+2192: a real arrow renders as a tofu box on the crop."""
    row = _mosaic()._badge({"label": "cat", "predicted_label": "dog"}).content.controls
    assert all(ord(char) < 128 for text in row for char in text.value)


def test_a_long_class_name_is_cut_to_the_corner_it_sits_in():
    """Nothing constrains the badge's width, so an ellipsis overflow would
    never fire -- the label would just run past the crop and over its
    neighbours."""
    row = _mosaic()._badge({"label": "a-very-long-class-name"}).content.controls
    assert len(row[0].value) <= BADGE_MAX_CHARS


def test_a_smaller_tile_gets_a_shorter_badge():
    """Zoomed right out, "airplane -> automobile" is twice the width of the
    crop it belongs to."""
    record = {"label": "automobile", "predicted_label": "airplane"}
    small = _mosaic(zoom=0)._badge(record).content.controls
    large = _mosaic(zoom=len(ZOOM_STEPS) - 1)._badge(record).content.controls
    assert len(small[0].value) < len(large[0].value)


def test_a_badge_always_says_something():
    """Cut to nothing it would be a colored smudge in the corner."""
    for step in range(len(ZOOM_STEPS)):
        row = _mosaic(zoom=step)._badge({"label": "automobile", "predicted_label": "airplane"})
        assert all(len(text.value.strip()) > 1 for text in row.content.controls)


def test_two_classes_never_share_a_badge_color():
    view = _mosaic()
    colors = {view._class_color(label) for label in ("cat", "dog", "bird")}
    assert len(colors) == 3


def test_the_badge_can_be_switched_off():
    view = _mosaic()
    view.show_labels = False
    assert view._badge({"label": "dog"}) is None


def test_a_mosaic_with_no_label_column_shows_no_badge():
    view = _mosaic()
    view.label_col = None
    assert view._badge({"label": "dog"}) is None


# ---- the pickers stay on one line --------------------------------------

def test_the_two_pickers_and_the_reset_button_fit_one_row():
    """The reset button used to wrap onto a line of its own at the bottom."""
    for window in (1024, 1280, 1440, 1920):
        cell_w, _ = _mosaic(width=window, info=True)._cell_size()
        available = cell_w - CARD_PADDING
        bar = DecisionBar.__new__(DecisionBar)
        bar.width, bar.compact = available, True
        picker = bar._picker_width()
        assert picker >= MIN_PICKER_WIDTH
        assert 2 * picker + RESET_BTN_WIDTH + 2 * PICKER_GAP <= available + 1


def test_pickers_fall_back_to_a_fixed_width_with_no_hint():
    bar = DecisionBar.__new__(DecisionBar)
    bar.width, bar.compact = None, True
    assert bar._picker_width() > 0


# ---- clicking a point in the projection --------------------------------
def _projection(on_point=None, prototype_at=(500.0, 500.0)):
    """A partially-built InteractiveProjection whose hit test always lands
    on point 1, plus the list of crops handed to the full-size viewer.

    `prototype_at` is where the single prototype sits on screen; the taps
    below all happen at (0, 0), so the default puts it well out of reach
    and passing (0.0, 0.0) puts it right under the click. None stands for
    a projection drawn with no prototypes at all.
    """
    from lookin.ui.interactive_projection import InteractiveProjection

    view = InteractiveProjection.__new__(InteractiveProjection)
    view.on_point = on_point
    view.uuids = ["crop-a", "crop-b"]
    view._pick = lambda x, y: 1
    view._image_for = lambda i: "b64"

    if prototype_at is None:
        view.psx = view.psy = None
    else:
        view.psx = np.array([prototype_at[0]])
        view.psy = np.array([prototype_at[1]])

    zoomed = []

    class Zoom:
        @staticmethod
        def open(b64, title):
            zoomed.append(title)

    class Ctx:
        image_zoom = Zoom()

    view.ctx = Ctx()
    return view, zoomed


def _tap(view):
    """Drive InteractiveProjection._on_tap at the origin."""
    class Event:
        local_x = local_y = 0.0

    view._on_tap(Event())


def test_tapping_a_point_hands_the_crop_to_its_owner():
    """So it opens in the carousel, with the review controls -- rather than
    in the plain image viewer, which shows a crop and no way to decide on
    it. A side panel beside the map did this job for a while and was taken
    out again: it was too tall for the screens this runs on."""
    seen = []
    view, zoomed = _projection(on_point=seen.append)
    _tap(view)

    # The *position* of the point, not its uuid: the map can hold crops
    # from more than one pool and a uuid is only unique within one of them.
    assert seen == [1]
    assert zoomed == [], "the bare viewer is not used when there is an owner"


def test_tapping_a_prototype_opens_nothing():
    """A prototype is drawn three times the size of a point and on top of
    the cloud, so a click on one is aimed at it -- and it has no crop. The
    failure would be the carousel opening on whichever point happens to lie
    under the marker."""
    seen = []
    view, zoomed = _projection(on_point=seen.append, prototype_at=(0.0, 0.0))
    _tap(view)

    assert seen == []
    assert zoomed == []


def test_a_tap_still_lands_when_no_prototypes_are_drawn():
    """_on_tap tests the prototypes first; with none there is nothing to
    test against, and the tap must not be swallowed on the way."""
    seen = []
    view, _ = _projection(on_point=seen.append, prototype_at=None)
    _tap(view)
    assert seen == [1]


def test_tapping_falls_back_to_the_viewer_with_no_owner():
    """Better than swallowing the click."""
    view, zoomed = _projection(on_point=None)
    _tap(view)
    assert zoomed == ["crop-b"]



def test_the_projection_view_wires_the_point_click_up():
    """The regression itself: the callback existed for the lasso but not for
    a single tap, so that one path still reached the viewer."""
    import inspect

    from lookin.ui import projection_view

    source = inspect.getsource(projection_view.ProjectionView)
    assert "on_point=" in source, "InteractiveProjection is built without a point handler"
    assert 'review_source="projection"' in source, "and its carousel would have no review controls"


# ---- shuffle -----------------------------------------------------------

def test_shuffle_shows_every_crop_once_in_another_order():
    rows = [{"uuid": str(i)} for i in range(50)]
    view = _mosaic(total=len(rows))
    view.page_provider = lambda start, end: rows[start:end]
    view._order = None
    assert view._fetch(0, 50) == rows

    view._order = np.random.default_rng(1).permutation(50)
    shuffled = view._fetch(0, 25) + view._fetch(25, 50)
    assert shuffled != rows
    assert sorted(shuffled, key=lambda r: int(r["uuid"])) == rows
