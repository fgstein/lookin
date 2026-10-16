"""The Gallery's shuffle -- a random order that holds still while the
mosaic scrolls, comes back with a snapshot, and goes away when switched
off -- and the scrolling itself."""
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
    # One crop per row and a short window: a batch is a few rows, so the
    # six crops of the fixture take more than one load.
    view.cols_field.value = "1"
    view.ctx.page.height = 300
    view.build()
    return view


def _all_pages(view) -> list:
    """Every crop, in the order the mosaic shows them, by scrolling to the
    bottom until nothing more comes."""
    view._render_grid()
    for _ in range(10):
        view._load_more()
    return [r["uuid"] for r in view._current_records]


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


def test_the_mosaic_opens_with_a_batch_and_scrolls_to_the_rest(gallery):
    gallery._render_grid()
    first = len(gallery._current_records)
    assert 0 < first < 6, "the whole set was loaded up front"
    assert gallery.footer.visible
    rows_before = list(gallery.grid_area.controls)

    class Scroll:
        pixels, max_scroll_extent, viewport_dimension = 900.0, 1000.0, 300.0

    gallery._on_scroll(Scroll())
    assert len(gallery._current_records) > first
    assert gallery.grid_area.controls[:len(rows_before)] == rows_before, "rows already shown are kept"


def test_a_scroll_far_from_the_bottom_loads_nothing(gallery):
    gallery._render_grid()
    first = len(gallery._current_records)

    class Scroll:
        pixels, max_scroll_extent, viewport_dimension = 0.0, 5000.0, 300.0

    gallery._on_scroll(Scroll())
    assert len(gallery._current_records) == first


def test_the_detail_dialog_reaches_the_whole_set_from_the_clicked_crop(gallery, monkeypatch):
    built = {}

    class FakeCarousel:
        def __init__(self, dataset, total, provider, *args, **kwargs):
            built["total"], built["provider"] = total, provider

        def build(self):
            return None

    import lookin.ui.gallery_view as gv
    monkeypatch.setattr(gv, "Carousel", FakeCarousel)
    gallery.ctx.page.open = lambda dialog: None
    gallery._render_grid()
    gallery._open_detail(1)

    assert built["total"] == 6, "not just the crops loaded so far"
    assert [r["uuid"] for r in built["provider"](0, 6)] == list("bcdefa")
