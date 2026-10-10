"""Display LUTs.

These change what the screen shows and must never change the file, so the
tests check both: that the mapping does what it says, and that the source
image is untouched. Colour handling matters too -- a per-channel stretch
would shift the white balance and make a display setting look like a
different image.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from lookin.ui.components import (
    LUT_EQUALIZE, LUT_MINMAX, LUT_NONE, LUT_OPTIONS, LUT_PERCENTILE,
    _apply_lut, file_to_full_base64,
)


def _decode(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))))


@pytest.fixture
def dim_grey(tmp_path):
    """An 8-bit greyscale crop using only a narrow slice of the range --
    exactly the case a LUT exists for."""
    arr = np.linspace(60, 90, 64 * 64).reshape(64, 64).astype(np.uint8)
    path = tmp_path / "dim.png"
    Image.fromarray(arr, mode="L").save(path)
    return path


# ---- the mappings -------------------------------------------------------

def test_minmax_uses_the_whole_range():
    arr = np.linspace(60, 90, 100).reshape(10, 10).astype(np.uint8)
    out = _apply_lut(arr, LUT_MINMAX)
    assert out.min() == 0 and out.max() == 255


def test_percentile_clips_an_outlier_instead_of_being_flattened_by_it():
    """One hot pixel is what makes a plain min/max stretch useless."""
    arr = np.linspace(100, 400, 400).reshape(20, 20).astype(np.uint16)
    arr[0, 0] = 60_000
    minmax = _apply_lut(arr, LUT_MINMAX)
    percentile = _apply_lut(arr, LUT_PERCENTILE)
    assert minmax[10, 10] < 5, "the outlier flattens everything else"
    assert percentile[10, 10] > 100, "clipping it brings the detail back"
    assert percentile.max() == 255


def test_percentile_falls_back_when_the_clip_window_collapses():
    """A near-constant image plus a couple of outliers puts both
    percentiles on the same value; stretching [lo, lo] would paint the crop
    black, which is strictly worse than showing it."""
    arr = np.full((20, 20), 100, dtype=np.uint16)
    arr[0, 0] = 60_000
    out = _apply_lut(arr, LUT_PERCENTILE)
    assert out.max() > 0, "something is rendered rather than a black square"


def test_equalisation_spreads_a_clumped_histogram():
    rng = np.random.default_rng(0)
    arr = (rng.normal(128, 8, size=(64, 64))).clip(0, 255).astype(np.uint8)
    out = _apply_lut(arr, LUT_EQUALIZE)
    assert out.std() > arr.std()


def test_a_flat_image_does_not_divide_by_zero():
    out = _apply_lut(np.full((8, 8), 42, dtype=np.uint8), LUT_MINMAX)
    assert out.shape == (8, 8) and out.dtype == np.uint8


@pytest.mark.parametrize("lut", [LUT_MINMAX, LUT_PERCENTILE, LUT_EQUALIZE])
def test_every_lut_returns_displayable_8_bit(lut):
    arr = (np.linspace(0, 65535, 256).reshape(16, 16)).astype(np.uint16)
    out = _apply_lut(arr, lut)
    assert out.dtype == np.uint8 and out.shape == arr.shape


# ---- colour and alpha ---------------------------------------------------

def test_colour_channels_are_stretched_together_not_separately():
    """A per-channel stretch would pull a blue-tinted crop towards grey --
    a display setting that silently rewrites the colours."""
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[..., 0] = 40   # dim red
    arr[..., 2] = 200  # bright blue
    out = _apply_lut(arr, LUT_MINMAX)
    assert out[..., 0].max() < out[..., 2].max(), "the blue cast survives"


def test_alpha_is_carried_through_untouched():
    """Stretching transparency would make a masked crop's background
    reappear."""
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[..., :3] = 100
    arr[..., 3] = 255
    out = _apply_lut(arr, LUT_MINMAX)
    assert out.shape == (8, 8, 4)
    assert (out[..., 3] == 255).all()


# ---- through the loader -------------------------------------------------

def test_no_lut_serves_the_file_untouched(dim_grey):
    original = dim_grey.read_bytes()
    served = base64.b64decode(file_to_full_base64(dim_grey, LUT_NONE))
    assert served == original, "a PNG with no LUT is passed through byte for byte"


def test_a_lut_changes_what_is_shown_but_not_the_file(dim_grey):
    before = dim_grey.read_bytes()
    shown = _decode(file_to_full_base64(dim_grey, LUT_PERCENTILE))
    assert shown.max() > 200, "the dim crop is stretched for display"
    assert dim_grey.read_bytes() == before, "the file on disk is untouched"


def test_the_cache_is_keyed_on_the_lut(dim_grey):
    """Switching LUT must not return the previous rendering."""
    plain = file_to_full_base64(dim_grey, LUT_NONE)
    stretched = file_to_full_base64(dim_grey, LUT_MINMAX)
    assert plain != stretched
    assert file_to_full_base64(dim_grey, LUT_NONE) == plain, "and back again"


def test_sixteen_bit_is_brought_down_even_with_no_lut(tmp_path):
    """No web image format carries 16-bit channels, so this is not a LUT
    choice -- it is the only way to show the data at all."""
    arr = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16)
    path = tmp_path / "deep.tif"
    Image.fromarray(arr, mode="I;16").save(path)
    shown = _decode(file_to_full_base64(path, LUT_NONE))
    assert shown.dtype == np.uint8
    assert shown.min() == 0 and shown.max() == 255


def test_an_unreadable_image_falls_back_instead_of_raising(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    assert file_to_full_base64(broken, LUT_MINMAX)


def test_every_offered_option_is_implemented(dim_grey):
    """The dropdown and the implementation must not drift apart."""
    for key, label in LUT_OPTIONS:
        assert label
        assert file_to_full_base64(dim_grey, key)
