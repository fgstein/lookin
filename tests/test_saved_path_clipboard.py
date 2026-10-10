"""The path of a file written to disk, put on the user's clipboard.

The asymmetry this exists for is the one ui/image_export.py describes: the
file is written on the machine the app runs on, and the person asking for
it is sitting at another one, reaching it over a forwarded port. The path
is therefore the only part of a save they can act on -- and retyping it out
of a status line is exactly the kind of transcription that ends up
pointing at a directory that does not exist.

Flet's clipboard carries text only, which is all a path is.
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from lookin.ui.components import (
    CLIPBOARD_NOTE, copy_to_clipboard, save_png_to_disk, saved_path_message,
)


class FakePage:
    def __init__(self, fails: bool = False):
        self.clipboard = None
        self.fails = fails

    def set_clipboard(self, text):
        if self.fails:
            raise RuntimeError("the browser refused the clipboard")
        self.clipboard = text


class FakeCtx:
    def __init__(self, page=None):
        self.page = page


def _png_base64() -> str:
    buf = io.BytesIO()
    Image.new("L", (1, 1), 0).save(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_the_path_lands_on_the_clipboard_as_text():
    page = FakePage()
    assert copy_to_clipboard(FakeCtx(page), "/data/exports/report.html")
    assert page.clipboard == "/data/exports/report.html"


def test_the_status_line_still_shows_the_path_it_copied():
    """"Copied" on its own would hide the one thing the user reads the
    line for: where the file actually went."""
    page = FakePage()
    message = saved_path_message(FakeCtx(page), "/data/exports/report.html")

    assert message.startswith("/data/exports/report.html")
    assert CLIPBOARD_NOTE in message


def test_a_browser_that_refuses_the_clipboard_does_not_fail_the_save():
    """The write already happened by the time this runs; a clipboard the
    browser would not hand over must not turn it into an error."""
    message = saved_path_message(FakeCtx(FakePage(fails=True)), "/data/out.png")

    assert message == "/data/out.png"
    assert CLIPBOARD_NOTE not in message


def test_no_page_means_no_claim_that_anything_was_copied():
    """A panel built without a context -- a test, or a script."""
    assert not copy_to_clipboard(FakeCtx(None), "/data/out.png")
    assert not copy_to_clipboard(None, "/data/out.png")
    assert saved_path_message(None, "/data/out.png") == "/data/out.png"


def test_saving_a_chart_copies_where_it_was_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("lookin.ui.components._default_save_dir", lambda: tmp_path)
    page = FakePage()

    ok, message = save_png_to_disk(_png_base64(), "Per-class f1", FakeCtx(page))

    assert ok
    written = list(tmp_path.glob("*.png"))
    assert len(written) == 1
    assert page.clipboard == str(written[0])
    assert str(written[0]) in message
