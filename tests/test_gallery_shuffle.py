"""The Gallery's shuffle: a random order that holds still while paging,
comes back with a snapshot, and goes away when switched off."""
from __future__ import annotations

import pytest

from lookin.ui.gallery_view import GalleryView


class FakePage:
    width, height = 1440, 900

    def run_thread(self, fn, *args):
        fn(*args)


class FakeCtx:
    def __init__(self, state):
        self.page = FakePage()
        self.state = state


@pytest.fixture
def gallery(loaded_state, monkeypatch):
    view = GalleryView(FakeCtx(loaded_state))
    # Thumbnails are not what is under test, and the crops do not exist.
    monkeypatch.setattr(view, "_refresh_in_background", lambda: None)
    view.rows_field.value, view.cols_field.value = "1", "2"
    view.build()
    return view


def _all_pages(view) -> list:
    uuids, view.page_index = [], 0
    for _ in range(3):
        view._render_grid()
        uuids += [r["uuid"] for r in view._current_records]
        view.page_index += 1
    return uuids


def test_shuffle_reorders_every_crop_and_paging_keeps_the_order(gallery):
    plain = _all_pages(gallery)
    assert plain == list("abcdef")
    gallery._set_shuffle(12345)
    shuffled = _all_pages(gallery)
    assert sorted(shuffled) == plain and shuffled != plain
    assert _all_pages(gallery) == shuffled
    gallery._set_shuffle(0)
    assert _all_pages(gallery) == plain


def test_a_snapshot_brings_the_same_order_back(gallery):
    gallery._set_shuffle(12345)
    order = _all_pages(gallery)
    saved = gallery.view_state()
    gallery._set_shuffle(0)
    gallery.restore_view_state(saved)
    assert gallery.shuffle_switch.value
    assert _all_pages(gallery) == order


def test_a_reshuffle_blacks_out_the_old_pictures_first(gallery):
    import flet as ft

    gallery._render_grid()
    images = [c for row in gallery.grid_area.controls for cell in row.controls
              for c in getattr(getattr(cell, "content", None), "controls", None) or [] if isinstance(c, ft.Image)]
    assert images
    gallery._blank_grid()
    firsts = [cell.content.controls[0] for row in gallery.grid_area.controls for cell in row.controls
              if getattr(cell, "content", None) is not None]
    assert firsts and not any(isinstance(c, ft.Image) for c in firsts)
    assert all(c.bgcolor == ft.Colors.BLACK and c.width == images[0].width for c in firsts)
