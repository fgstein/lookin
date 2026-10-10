"""How the Embeddings tab hands the window to the map.

Two panes, and full screen swaps between them. All of it is arithmetic
against page.width/height plus a handful of visibility flags, and every way
it can go wrong is silent in the layout and obvious on screen:

* a map sized taller than the room above and below it is not a smaller
  map, it is a map whose bottom is cut off by the window -- the constants
  it is measured against were wrong by ~60px until the running app was
  measured, so they are pinned here;
* full screen that forgets to give the settings column back strands the
  user in a tab with no controls and nothing to press.

No Flet session is involved: build() is real, but nothing is mounted, so
this is the tree and the numbers, not a rendering.
"""
from __future__ import annotations

import flet as ft
import numpy as np
import pytest

from lookin.ui.projection_view import (
    CHROME_HEIGHT, FULLSCREEN_CHROME_HEIGHT, FULLSCREEN_PADDING, MAP_MAX,
    MAP_MAX_FULLSCREEN, MIN_PLOT_SIZE, ProjectionView,
)

# Windows worth checking: a small laptop, the common 16:9 sizes, a big
# monitor, and a tall/narrow one -- the last is the shape full screen helps
# most, because there the halved width was the binding constraint.
WINDOWS = [(1280, 800), (1440, 900), (1920, 1080), (2560, 1440), (1100, 1400)]


class FakePage:
    def __init__(self, width, height):
        self.width, self.height = width, height


class FakeCtx:
    def __init__(self, width, height):
        self.page = FakePage(width, height)

        class State:
            dataset = None

        self.state = State()


def _view(width=1440, height=900) -> ProjectionView:
    """A built tab: build() wires left_pane, map_pane and panes_row, which
    is what full screen moves around."""
    view = ProjectionView(FakeCtx(width, height))
    view.build()
    return view


def _with_a_map(view: ProjectionView) -> ProjectionView:
    """A run puts a switcher in the results pane -- that is where the full
    screen button lives, so it only exists once there is a projection to
    look at."""
    view.projection_results_container.content = view._switcher(
        ft.Text("interactive"), ft.Text("static"), [ft.Text("pan"), ft.Text("zoom")])
    return view


# ---- the map fits in the window it was measured from -------------------

@pytest.mark.parametrize("width,height", WINDOWS)
def test_the_map_fits_above_and_below(width, height):
    """The failure this is here for: the map ran off the bottom of the
    window, which looks like a scatter plot with its lower edge shaved off
    and nothing at all in the layout."""
    view = _view(width, height)
    for full in (False, True):
        view._fullscreen = full
        chrome = FULLSCREEN_CHROME_HEIGHT if full else CHROME_HEIGHT
        assert view._map_size() <= height - chrome or view._map_size() == MIN_PLOT_SIZE


@pytest.mark.parametrize("width,height", WINDOWS)
def test_the_map_fits_left_and_right(width, height):
    view = _view(width, height)
    assert view._map_size() <= width * 0.5 - 40 or view._map_size() == MIN_PLOT_SIZE
    view._fullscreen = True
    assert view._map_size() <= width - 2 * FULLSCREEN_PADDING


@pytest.mark.parametrize("width,height", WINDOWS)
def test_full_screen_never_shrinks_the_map(width, height):
    """It drops chrome and lifts the half-width bound, so it can only add
    room. A negative here would mean the button makes the map worse."""
    view = _view(width, height)
    normal = view._map_size()
    view._fullscreen = True
    assert view._map_size() >= normal


def test_full_screen_helps_most_where_the_width_was_binding():
    """On a tall window the settings column was costing the map more than
    half its size; on 16:9 the square is height-bound and the gain is only
    the chrome that full screen reclaims. Both are worth stating, because
    a big jump on 16:9 would mean the map had stopped fitting."""
    tall = _view(1100, 1400)
    narrow_gain = tall._map_size()
    tall._fullscreen = True
    assert tall._map_size() > 2 * narrow_gain

    wide = _view(1920, 1080)
    before = wide._map_size()
    wide._fullscreen = True
    assert 0 < wide._map_size() - before <= CHROME_HEIGHT - FULLSCREEN_CHROME_HEIGHT


def test_the_caps_still_bound_a_huge_screen():
    view = _view(5120, 2880)
    assert view._map_size() == MAP_MAX
    view._fullscreen = True
    assert view._map_size() == MAP_MAX_FULLSCREEN


# ---- what the toggle actually moves ------------------------------------

def test_full_screen_hides_the_settings_column():
    view = _with_a_map(_view())
    assert view.left_pane.visible

    view._set_fullscreen(True)
    assert not view.left_pane.visible, "the map would still be in half a window"
    assert view.root.padding == FULLSCREEN_PADDING
    assert view.panes_row.alignment == ft.MainAxisAlignment.CENTER
    assert view.map_pane.width == view._map_size()


def test_leaving_full_screen_puts_everything_back():
    view = _with_a_map(_view())
    before = (view.root.padding, view.panes_row.alignment, view.map_pane.width)
    view._set_fullscreen(True)
    view._set_fullscreen(False)
    assert view.left_pane.visible
    assert (view.root.padding, view.panes_row.alignment, view.map_pane.width) == before


def test_the_button_says_which_way_it_goes():
    """One button for both directions, so the icon is the only thing that
    tells the user whether pressing it enters or leaves."""
    view = _with_a_map(_view())
    assert view.fullscreen_btn.icon == ft.Icons.FULLSCREEN
    view._set_fullscreen(True)
    assert view.fullscreen_btn.icon == ft.Icons.FULLSCREEN_EXIT
    view._set_fullscreen(False)
    assert view.fullscreen_btn.icon == ft.Icons.FULLSCREEN


def test_a_rerun_keeps_the_button_pointing_the_right_way():
    """Every run builds a new switcher, and with it a new button -- which
    would come back saying "enter" while the tab is already full screen."""
    view = _with_a_map(_view())
    view._set_fullscreen(True)
    _with_a_map(view)  # what a second run does
    assert view.fullscreen_btn.icon == ft.Icons.FULLSCREEN_EXIT


# ---- one line above the map --------------------------------------------

def test_everything_above_the_map_is_one_flat_row():
    """The map is square and height-bound, so a second row of controls
    comes straight out of the picture. Flat matters as much as single: a
    nested Row is laid out as one block, so it cannot share the line, and
    with no width of its own it runs past the edge of the pane instead of
    wrapping -- which is how the end of "Reset view" went missing."""
    view = _view()
    toolbar = [ft.Text("pan"), ft.Text("zoom"), ft.Text("reset")]
    column = view._switcher(ft.Text("interactive"), ft.Text("static"), toolbar)

    header = column.controls[0]
    assert isinstance(header, ft.Row)
    assert header.wrap, "with no wrap the row is clipped at the pane edge, not folded"
    assert not any(isinstance(c, (ft.Row, ft.Column)) for c in header.controls), \
        "a nested row would take a line of its own"
    for control in toolbar:
        assert control in header.controls, "the map's controls share the tab's line"
    assert view.fullscreen_btn in header.controls

    # ...and the renderings are below it, not beside it
    assert column.controls[1].content.value == "interactive"
    assert column.controls[2].content.value == "static"


def test_there_is_no_caption_line_over_the_map():
    """"N points from M classes" used to sit here. It cost a row of the
    picture to repeat what the left column already says."""
    view = _view()
    column = view._switcher(ft.Text("interactive"), ft.Text("static"), [])
    assert len(column.controls) == 3, "header, interactive, static -- nothing else"


def test_a_restricted_projection_still_warns_somewhere():
    """The caption that went with a *class-restricted* run is not the same
    thing and is kept, beside the chips that caused it: a projection fitted
    to two classes looks exactly like one fitted to twenty."""
    import inspect

    from lookin.ui import projection_view

    source = inspect.getsource(projection_view.ProjectionView._render_projection)
    assert "self.projection_note.content" in source
    assert "not comparable" in source, "the warning itself is gone, not just moved"


def test_dropping_the_result_lets_the_settings_column_back_in():
    """The only way out of full screen is a button that lives with the
    map. Lose the map while full screen and the tab has no controls left
    on it at all -- including the ones that would run the next projection."""
    view = _with_a_map(_view())
    view._set_fullscreen(True)
    view._reset_result()
    assert not view._fullscreen
    assert view.left_pane.visible, "stranded with no way to run anything"


# ---- the toolbar belongs to the interactive rendering ------------------

def _toolbar_owner(zoomed=False, lassoed=False):
    """An InteractiveProjection built far enough to have a toolbar. Its
    buttons are the awkward part of sharing a line with the tab: half of
    them decide their own visibility, so hiding the toolbar and setting
    those has to be one decision, not two."""
    from lookin.ui.interactive_projection import InteractiveProjection

    view = InteractiveProjection.__new__(InteractiveProjection)
    view._toolbar_visible = True
    view._fit_scale = 10.0
    view._scale = 20.0 if zoomed else 10.0
    view._selected = np.array([1, 2]) if lassoed else np.array([], dtype=int)
    view._lasso_data = np.zeros((4, 2)) if lassoed else None
    view.on_select = None
    view._build_toolbar()
    return view


def test_static_hides_the_whole_toolbar():
    """Pan, Lasso and the zoom buttons drive the canvas; against a
    matplotlib PNG they do nothing at all."""
    view = _toolbar_owner(zoomed=True, lassoed=True)
    assert all(c.visible for c in view.toolbar_controls())

    view.set_toolbar_visible(False)
    assert not any(c.visible for c in view.toolbar_controls())


def test_a_hidden_toolbar_does_not_creep_back_one_button_at_a_time():
    """The failure the single resolver is here for: Reset view turns
    itself on with a zoom and the selection buttons with a lasso, so a
    toolbar hidden by "Static" would reappear in pieces over a figure that
    has nothing to do with them."""
    view = _toolbar_owner()
    view.set_toolbar_visible(False)

    view._scale = view._fit_scale * 4          # as a zoom would leave it
    view._sync_toolbar()
    assert not view.reset_view_btn.visible

    view._selected = np.array([1, 2])
    view._lasso_data = np.zeros((4, 2))
    view._set_selection_text()
    assert not view.show_selection_btn.visible
    assert not view.clear_selection_btn.visible


def test_coming_back_restores_what_the_map_had_showing():
    """And the other half: the buttons that were legitimately hidden
    before must not all switch on together on the way back."""
    view = _toolbar_owner(zoomed=True)
    view.set_toolbar_visible(False)
    view.set_toolbar_visible(True)

    assert view.mode_switch.visible and view.zoom_in_btn.visible
    assert view.reset_view_btn.visible, "the map is still zoomed"
    assert not view.show_selection_btn.visible, "there was never a selection"
    assert not view.clear_selection_btn.visible
