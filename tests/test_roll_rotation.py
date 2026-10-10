"""Roll correction: turning each crop to the same upright quarter.

Like the LUTs beside it (tests/test_luts.py) this is a display transform,
so the tests check both halves of that: the crop on screen is turned, and
the file it came from is not. The other half of the feature is that a
quarter turn is *lossless* -- the same pixel values come out the other
side, only reindexed -- which is what makes it safe to look at through.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from lookin.ui.components import (
    LUT_MINMAX, LUT_NONE, ROLL_NONE,
    crop_display, file_to_full_base64, quarter_turns_for_roll, roll_quarter_turns,
)


def _decode(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))))


class FakeCtx:
    """Just the display attributes a view reads -- what every call site
    actually needs from the context to draw a crop."""

    def __init__(self, lut=LUT_NONE, roll_col=ROLL_NONE):
        self.display_lut = lut
        self.display_roll_col = roll_col


@pytest.fixture
def wedge(tmp_path):
    """A crop with no symmetry at all: one bright corner, so every one of
    the four orientations is distinguishable from the other three."""
    arr = np.zeros((32, 48), dtype=np.uint8)
    arr[:8, :8] = 255
    path = tmp_path / "wedge.png"
    Image.fromarray(arr, mode="L").save(path)
    return path


# ---- reading a roll angle ----------------------------------------------

HALF_PI = np.pi / 2


@pytest.mark.parametrize("angle, turns", [
    (0, 0), (HALF_PI, 3), (np.pi, 2), (3 * HALF_PI, 1), (2 * np.pi, 0),
    (-HALF_PI, 1), (5 * HALF_PI, 3),
])
def test_a_right_angle_is_cancelled_by_the_opposite_quarter_turn(angle, turns):
    """The turn undoes the roll, so it runs against it: a crop rolled by
    +pi/2 is brought back by three counter-clockwise quarters, not one."""
    assert quarter_turns_for_roll(angle) == turns


def test_the_roll_is_read_in_radians():
    """A roll column is in radians: every value it can hold sits within a
    few units of zero, which read as degrees would never turn anything."""
    assert quarter_turns_for_roll(1.5708) == 3
    assert quarter_turns_for_roll(3.1416) == 2
    assert quarter_turns_for_roll(-1.5708) == 1


def test_an_angle_snaps_to_the_nearest_quarter():
    """Either side of pi/2 is "one turn off": the correction only ever
    lands on the four upright orientations."""
    assert quarter_turns_for_roll(HALF_PI - 0.05) == 3
    assert quarter_turns_for_roll(HALF_PI + 0.05) == 3
    assert quarter_turns_for_roll(np.pi / 4 - 0.01) == 0, "less than half a quarter stays put"
    assert quarter_turns_for_roll(np.pi / 4 + 0.01) == 3


@pytest.mark.parametrize("value", [None, float("nan"), "", "north", [1]])
def test_an_unreadable_roll_leaves_the_crop_alone(value):
    """One bad cell must not stop the rest of the column from being used,
    so it means "no roll known here", not an error."""
    assert quarter_turns_for_roll(value) == 0


# ---- the session setting ------------------------------------------------

def test_no_roll_column_means_no_rotation():
    assert roll_quarter_turns(FakeCtx(), {"roll": HALF_PI}) == 0


def test_the_chosen_column_is_the_one_read():
    ctx = FakeCtx(roll_col="roll")
    assert roll_quarter_turns(ctx, {"roll": HALF_PI, "other_angle": np.pi}) == 3


def test_a_column_missing_from_the_record_is_not_an_error():
    """A setting left over from a previously loaded dataset renders the
    crop as stored rather than raising out of a drawing path."""
    assert roll_quarter_turns(FakeCtx(roll_col="roll"), {"brightness": 12}) == 0


def test_crop_display_carries_both_settings():
    ctx = FakeCtx(lut=LUT_MINMAX, roll_col="roll")
    assert crop_display(ctx, {"roll": np.pi}) == (LUT_MINMAX, 2)


def test_crop_display_survives_a_context_without_the_settings():
    assert crop_display(object(), {"roll": HALF_PI}) == (LUT_NONE, 0)
    assert crop_display(None, None) == (LUT_NONE, 0)


# ---- what ends up on screen --------------------------------------------

def test_the_crop_is_turned_by_the_requested_quarters(wedge):
    stored = _decode(file_to_full_base64(wedge, LUT_NONE, 0))
    turned = _decode(file_to_full_base64(wedge, LUT_NONE, 1))
    assert np.array_equal(turned, np.rot90(stored, 1)), "counter-clockwise, like PIL and numpy"


def test_the_correction_brings_a_rolled_crop_back_upright(wedge):
    """The whole point of the feature, in one assertion: a crop stored
    rolled by +pi/2 ends up on screen as the unrolled one. Turning with
    the roll rather than against it left every quarter roll a half turn
    out, which is what the old "turn the other way" box worked around."""
    stored = _decode(file_to_full_base64(wedge, LUT_NONE, 0))
    rolled = np.rot90(stored, 1)  # what a crop cut at a roll of +pi/2 holds
    corrected = np.rot90(rolled, quarter_turns_for_roll(HALF_PI))
    assert np.array_equal(corrected, stored)


def test_a_non_square_crop_keeps_all_of_itself(wedge):
    stored = _decode(file_to_full_base64(wedge, LUT_NONE, 0))
    turned = _decode(file_to_full_base64(wedge, LUT_NONE, 1))
    assert stored.shape == (32, 48)
    assert turned.shape == (48, 32), "the frame turns with the crop instead of cropping it"


def test_a_quarter_turn_changes_no_pixel_value(wedge):
    """The reason only whole quarters are ever applied: this is a pure
    reindexing, where a free rotation would resample the detail the tool
    exists to show."""
    stored = _decode(file_to_full_base64(wedge, LUT_NONE, 0))
    for turns in (1, 2, 3):
        turned = _decode(file_to_full_base64(wedge, LUT_NONE, turns))
        assert sorted(turned.ravel().tolist()) == sorted(stored.ravel().tolist())


def test_four_quarters_come_back_to_the_start(wedge):
    assert file_to_full_base64(wedge, LUT_NONE, 4) == file_to_full_base64(wedge, LUT_NONE, 0)


def test_the_cache_does_not_serve_one_orientation_for_another(wedge):
    """Switching the setting has to re-encode, the same way switching LUT
    does -- the cache is keyed on both."""
    upright = file_to_full_base64(wedge, LUT_NONE, 0)
    turned = file_to_full_base64(wedge, LUT_NONE, 1)
    assert upright != turned
    assert file_to_full_base64(wedge, LUT_NONE, 0) == upright, "and back again"


def test_the_lut_and_the_turn_compose(wedge):
    stretched = _decode(file_to_full_base64(wedge, LUT_MINMAX, 0))
    both = _decode(file_to_full_base64(wedge, LUT_MINMAX, 1))
    assert np.array_equal(both, np.rot90(stretched, 1))


def test_the_file_is_never_touched(wedge):
    before = wedge.read_bytes()
    file_to_full_base64(wedge, LUT_MINMAX, 3)
    assert wedge.read_bytes() == before


def _projection_over(path, record, ctx):
    """A partially-built InteractiveProjection with one point, whose crop
    is `path` -- just what _image_for reads."""
    from lookin.ui.interactive_projection import InteractiveProjection

    class Dataset:
        @staticmethod
        def resolve_crop_path(value):
            return value

    class State:
        dataset = Dataset()

    ctx.state = State()
    view = InteractiveProjection.__new__(InteractiveProjection)
    view.ctx = ctx
    view.records = [{"crop": path, **record}]
    view.crop_col = "crop"
    view._b64_by_index = {}
    return view


def test_the_projection_hover_is_turned_too(wedge):
    """The crop at the pointer in the embedding projection is a crop on
    screen like any other, so it follows the roll setting."""
    ctx = FakeCtx(roll_col="roll")
    view = _projection_over(wedge, {"roll": HALF_PI}, ctx)
    stored = _decode(file_to_full_base64(wedge, LUT_NONE, 0))
    assert np.array_equal(_decode(view._image_for(0)), np.rot90(stored, 3))


def test_the_projection_hover_follows_a_changed_setting(wedge):
    """Its per-point cache must not keep serving the crop as it looked
    before the setting changed."""
    ctx = FakeCtx(roll_col="roll")
    view = _projection_over(wedge, {"roll": HALF_PI}, ctx)
    turned = view._image_for(0)
    ctx.display_roll_col = ROLL_NONE
    assert view._image_for(0) == file_to_full_base64(wedge, LUT_NONE, 0)
    assert view._image_for(0) != turned


def test_a_16_bit_crop_can_be_turned(tmp_path):
    """16-bit data takes the re-encoding path for its own reasons; the
    turn has to survive being composed with that."""
    arr = np.linspace(0, 60_000, 24 * 24).reshape(24, 24).astype(np.uint16)
    path = tmp_path / "deep.tif"
    Image.fromarray(arr).save(path)
    upright = _decode(file_to_full_base64(path, LUT_NONE, 0))
    turned = _decode(file_to_full_base64(path, LUT_NONE, 2))
    assert np.array_equal(turned, np.rot90(upright, 2))
