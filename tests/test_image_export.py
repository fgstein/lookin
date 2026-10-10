"""Exporting a rendered PNG as a file the browser can open.

What these pin down is the contract ui/image_export.py has with two things
that are not visible from Python: flet_web's static file handler, which
serves the assets root at the site root, and the page's `<base href>`,
which is what a relative URL is resolved against.
"""
from __future__ import annotations

import base64
import io
import os

import pytest
from PIL import Image

from lookin.ui import image_export as ie
from lookin.ui.components import open_png_in_tab


def _png_base64() -> str:
    buf = io.BytesIO()
    Image.new("L", (1, 1), 0).save(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


PNG = _png_base64()


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """Each test gets its own assets root -- the module's is a process-wide
    temp dir created once and kept for the life of the app."""
    root = tmp_path / "assets"
    (root / ie.EXPORTS_SUBDIR).mkdir(parents=True)
    monkeypatch.setattr(ie, "_root", root)
    monkeypatch.setattr(ie, "_published", [])
    return root


def test_the_url_is_relative_so_a_forwarded_port_still_resolves_it():
    """An absolute URL would have to name a host and port, and the server
    only knows its own -- not the ones the browser reached it through."""
    ok, url = ie.publish_png(PNG, "2D projection (umap)")

    assert ok
    assert not url.startswith(("/", "http://", "https://"))
    assert url.startswith(f"{ie.EXPORTS_SUBDIR}/")


def test_the_file_lands_where_the_static_handler_serves_it(_isolated_root):
    """The URL is the path under the assets root, which is what flet_web
    appends to its list of served directories."""
    _, url = ie.publish_png(PNG, "confusion matrix")

    served = _isolated_root / url
    assert served.is_file()
    assert served.read_bytes() == base64.b64decode(PNG)


def test_a_title_that_is_not_a_filename_still_produces_one():
    """Panel titles carry class labels from the user's CSV, so they can
    hold anything at all -- slashes, spaces, accents."""
    _, url = ie.publish_png(PNG, "score / classe: bélier (2%)")

    name = url.split("/")[1]
    assert name.endswith(".png")
    assert set(name) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def test_two_exports_of_the_same_panel_get_different_urls():
    """A redrawn panel exported again must not land on the browser's cached
    copy of the previous picture."""
    _, first = ie.publish_png(PNG, "same title")
    _, second = ie.publish_png(PNG, "same title")

    assert first != second


def test_old_exports_are_pruned(_isolated_root):
    """Each export is a full-resolution encoding; an afternoon of them
    would otherwise grow without bound."""
    urls = [ie.publish_png(PNG, f"chart {i}")[1] for i in range(ie.MAX_KEPT + 5)]

    kept = list((_isolated_root / ie.EXPORTS_SUBDIR).glob("*.png"))
    assert len(kept) == ie.MAX_KEPT
    # The most recent survive: it is the one just asked for that matters.
    assert (_isolated_root / urls[-1]).is_file()
    assert not (_isolated_root / urls[0]).is_file()


def test_a_corrupt_encoding_is_reported_rather_than_raised():
    """The handler runs on a button click; an exception there dies in
    Flet's event loop with nothing shown to the user."""
    ok, message = ie.publish_png("not base64 at all!!", "chart")

    assert not ok
    assert "Could not prepare" in message


class _Page:
    def __init__(self):
        self.launched = []

    def launch_url(self, url, web_window_name=None, **kwargs):
        self.launched.append((url, web_window_name))


class _Ctx:
    def __init__(self):
        self.page = _Page()


def test_the_tab_is_a_new_one():
    """Navigating the current tab away would take the loaded dataset and
    every open panel with it."""
    ctx = _Ctx()

    ok, _ = open_png_in_tab(ctx, PNG, "chart")

    assert ok
    (url, window_name), = ctx.page.launched
    assert window_name == "_blank"
    assert url.startswith(f"{ie.EXPORTS_SUBDIR}/")


def test_nothing_is_written_when_there_is_no_page_to_open_it_in():
    ok, _ = open_png_in_tab(None, PNG, "chart")

    assert not ok
    assert not list((ie.assets_root() / ie.EXPORTS_SUBDIR).glob("*.png"))


def test_a_dead_instance_leaves_no_exports_behind(tmp_path, monkeypatch):
    """atexit covers a clean exit and Ctrl-C but not a SIGTERM, so the
    directory of a killed server is only ever cleaned up by the next run."""
    monkeypatch.setattr(ie.tempfile, "gettempdir", lambda: str(tmp_path))
    # Searched for rather than hardcoded: any number picked as "surely not
    # a pid" is one this machine may well have handed out.
    gone = next(p for p in range(4_000_000, 3_999_000, -1) if not ie._pid_alive(p))
    dead = tmp_path / f"{ie.DIR_PREFIX}{gone}-abcd"
    (dead / ie.EXPORTS_SUBDIR).mkdir(parents=True)
    alive = tmp_path / f"{ie.DIR_PREFIX}{os.getpid()}-efgh"
    (alive / ie.EXPORTS_SUBDIR).mkdir(parents=True)
    unrelated = tmp_path / "someone-elses-tempdir"
    unrelated.mkdir()

    ie._sweep_orphans()

    assert not dead.exists()
    # Several instances on one machine is normal, and pulling the assets
    # out from under a running one would break every image it has open.
    assert alive.is_dir()
    assert unrelated.is_dir()
