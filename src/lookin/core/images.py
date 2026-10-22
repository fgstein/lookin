"""Crops as the screen shows them: the display LUTs, the roll correction,
and the decode/re-encode every crop goes through on its way to a browser,
a report or a mosaic.

Everything here is a *display* transform: it changes what is drawn, never
the file on disk, and no metric, embedding or prototype is computed through
it.
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ---- display LUTs -------------------------------------------------------
# Keys, compared in code and stored in the session's display setting -- only
# the labels beside them are display text (see CLAUDE.md).
LUT_NONE = "none"
LUT_MINMAX = "minmax"
LUT_PERCENTILE = "percentile"
LUT_EQUALIZE = "equalize"

LUT_OPTIONS = (
    (LUT_NONE, "No LUT (as stored)"),
    (LUT_MINMAX, "Min / max stretch"),
    (LUT_PERCENTILE, "Contrast (2-98%)"),
    (LUT_EQUALIZE, "Histogram equalisation"),
)
LUT_DESCRIPTIONS = {
    LUT_NONE: "Pixels exactly as they are in the file.",
    LUT_MINMAX: "Stretches the darkest pixel to black and the brightest to white. "
                "Shows everything, but one hot pixel flattens the rest.",
    LUT_PERCENTILE: "Same stretch, but clipping the darkest and brightest 2% first -- "
                    "the usual choice, since it ignores outliers.",
    LUT_EQUALIZE: "Flattens the histogram: maximum local contrast, at the cost of any "
                  "relationship between pixel value and displayed brightness.",
}
# The percentiles LUT_PERCENTILE clips to before stretching.
PERCENTILE_CLIP = (2.0, 98.0)


# ---- roll correction ----------------------------------------------------
# A dataset may carry the roll each crop was cut at. Knowing it, the viewer
# can turn every crop back to the same upright quarter, so a wall of
# thumbnails stops being a wall of arbitrarily-rotated ones.
#
# Only whole quarter turns are ever applied. A free rotation would have to
# resample the pixels -- interpolating exactly the fine detail this tool
# exists to let people look at -- and would leave triangular borders around
# each crop. A quarter turn is a pure reindexing: every pixel value comes
# out the other side unchanged.

# The "no roll column chosen" key, held in session.display_roll_col.
ROLL_NONE = ""


def quarter_turns_for_roll(value) -> int:
    """Counter-clockwise quarter turns that bring a crop rolled by `value`
    radians back upright, i.e. the nearest quarter turn that cancels it.

    The roll is always in radians: a column in degrees would have its
    values read as that many radians, which is wrong but not detectably so.

    A roll of 1.5 and one of 1.6 (either side of pi/2) both round to the
    same single turn: the correction only ever snaps to the four upright
    orientations, and what is left over stays visible in the crop, which
    is the honest outcome -- a viewer that straightened a pi/4 roll would
    be inventing an orientation the data does not support.

    Anything that is not a finite number -- a blank cell, a NaN, a string
    -- means "no roll known for this crop" and leaves it as stored. That is
    per crop, not per dataset: one unparseable cell must not stop the rest
    of the column from being used.
    """
    try:
        angle = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(angle):
        return 0
    # PIL and np.rot90 both count quarter turns counter-clockwise, and a
    # roll column counts its angle the same way, so a roll of +pi/2 is
    # cancelled by turning the negated angle -- a quarter turn clockwise.
    # Turning with the roll rather than against it is what used to leave
    # crops a half turn off, which is what the old "turn the other way"
    # box existed to undo.
    return (-int(round(angle / (np.pi / 2)))) % 4


def crop_display(session, record) -> tuple:
    """The (lut, quarter_turns) this session shows `record`'s crop through.

    One call rather than two so a view that draws crops cannot end up
    honouring the LUT while ignoring the roll. Everything is read
    defensively: a session without the display attributes (a test double)
    means "as stored", never an AttributeError from a rendering path.
    """
    lut = (getattr(session, "display_lut", LUT_NONE) or LUT_NONE) if session is not None else LUT_NONE
    return lut, roll_quarter_turns(session, record)


def roll_quarter_turns(session, record) -> int:
    """Quarter turns for one crop under the session's roll setting.

    A column that is not in `record` -- a setting left over from a
    previously loaded dataset, or a record built from a subset of the
    columns -- yields no rotation rather than an error.
    """
    if session is None or record is None:
        return 0
    column = getattr(session, "display_roll_col", ROLL_NONE) or ROLL_NONE
    if not column:
        return 0
    try:
        value = record[column]
    except (KeyError, IndexError, TypeError):
        return 0
    return quarter_turns_for_roll(value)


# The dropdown entry meaning "leave every crop as stored". A key, compared
# in code -- only the label beside it is display text (see CLAUDE.md).
ROLL_NONE_KEY = "__none__"


def roll_column_candidates(dataset) -> list:
    """The numeric, non-structural columns of `dataset`, in file order.

    A roll is one number per crop, so only numeric columns are offered.
    The three structural roles are excluded: a uuid or a crop path is
    never a roll, and offering a numeric *label* column here would just be
    a way to rotate crops by their class.

    Returns nothing at all when no dataset is loaded, which is what the
    dialog shows instead of an empty dropdown that looks broken.
    """
    if dataset is None:
        return []
    reserved = {dataset.columns.uuid_col, dataset.columns.crop_col, dataset.columns.label_col}
    return [str(col) for col in dataset.df.columns
            if str(col) not in reserved and pd.api.types.is_numeric_dtype(dataset.df[col])]


def load_and_encode_png(path: Path, lut: str = LUT_NONE, quarter_turns: int = 0) -> bytes:
    """Open any Pillow-readable image (TIFF, BMP, WEBP, ...), apply the
    display LUT and the roll correction, and re-encode to PNG bytes.

    16-bit data is always brought down to 8 bits, LUT or not, because no
    web image format carries 16-bit channels -- with no LUT selected that
    is a plain min/max stretch, which is the only way to show the data at
    all. This is the one place pixel *values* change, and only ever for
    display: nothing here writes back to the file."""
    try:
        img = Image.open(path)
    except Image.DecompressionBombError as ex:
        # Not an OSError, so it used to escape every caller's "show the
        # placeholder" handler and fail the whole request or panel.
        raise ValueError(str(ex)) from ex
    if img.mode == "CMYK" or img.mode in ("P", "PA"):
        # Palette indices and ink channels are not intensities: stretched
        # by a LUT they draw a different picture (CMYK's K read as alpha,
        # a palette as greys), and PNG cannot hold CMYK at all.
        has_alpha = img.mode == "PA" or "transparency" in img.info
        img = img.convert("RGBA" if has_alpha else "RGB")
    arr = np.array(img)

    if lut != LUT_NONE:
        arr = _apply_lut(arr, lut)
        img = Image.fromarray(arr)
    elif arr.dtype.kind in "uif" and arr.dtype != np.uint8:
        # 16-bit, and also 32-bit int or float (a mode "I"/"F" TIFF), which
        # PNG cannot store either: without this they rendered as the
        # placeholder, as if the file were broken. NaN pixels are left out
        # of the range, or they would make all of it NaN.
        finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
        arr = _stretch(np.nan_to_num(arr, nan=lo) if arr.dtype.kind == "f" else arr, lo, hi)
        img = Image.fromarray(arr)

    if quarter_turns % 4:
        # expand=True so a non-square crop keeps all of itself after an
        # odd number of turns instead of being cropped to its old frame.
        # Pillow special-cases multiples of 90 into a transpose, so this
        # moves pixels without resampling any of them.
        img = img.rotate(90 * (quarter_turns % 4), expand=True)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _stretch(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear map [lo, hi] -> [0, 255], clipping outside. A flat image
    (hi <= lo) becomes black rather than dividing by zero."""
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).round().astype(np.uint8)


def _apply_lut(arr: np.ndarray, lut: str) -> np.ndarray:
    """Map an image's intensities for display.

    The mapping is computed over *all* channels jointly rather than per
    channel. On a greyscale crop -- the case this exists for -- there is
    only one channel and the distinction is moot; on a colour one, a
    per-channel stretch would shift the white balance, turning a display
    setting into something that looks like a different image.

    An alpha channel is carried through untouched: stretching transparency
    is meaningless and would make a masked crop's background reappear.
    """
    alpha = None
    if arr.ndim == 3 and arr.shape[2] in (2, 4):
        alpha = arr[..., -1]
        arr = arr[..., :-1]

    finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
    if not finite.size:
        return np.zeros(arr.shape, dtype=np.uint8)

    if lut == LUT_MINMAX:
        out = _stretch(arr, float(finite.min()), float(finite.max()))
    elif lut == LUT_PERCENTILE:
        lo, hi = (float(v) for v in np.percentile(finite, PERCENTILE_CLIP))
        if hi <= lo:
            # The clip window collapsed: the image is near-constant apart
            # from a few outliers, and stretching [lo, lo] would paint the
            # whole crop black -- strictly worse than showing it. Fall back
            # to the full range, which at least renders something.
            lo, hi = float(finite.min()), float(finite.max())
        out = _stretch(arr, lo, hi)
    elif lut == LUT_EQUALIZE:
        out = _equalize(arr)
    else:
        out = _stretch(arr, float(finite.min()), float(finite.max()))

    if alpha is not None:
        alpha8 = alpha if alpha.dtype == np.uint8 else _stretch(alpha, 0, float(alpha.max() or 1))
        out = np.dstack([out, alpha8])
    return out


def _equalize(arr: np.ndarray) -> np.ndarray:
    """Histogram equalisation over the whole image, 256 bins.

    Done by hand rather than with PIL.ImageOps.equalize because that one
    only accepts 8-bit L/P/RGB images, and the crops this is for are
    routinely 16-bit."""
    flat = arr.astype(np.float64).ravel()
    if flat.size == 0 or flat.max() == flat.min():
        return np.zeros(arr.shape, dtype=np.uint8)
    counts, edges = np.histogram(flat, bins=256)
    cdf = np.cumsum(counts).astype(np.float64)
    cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1e-9)
    # np.interp maps each pixel through the CDF built from the bin centres.
    centres = (edges[:-1] + edges[1:]) / 2.0
    mapped = np.interp(flat, centres, cdf * 255.0)
    return mapped.reshape(arr.shape).round().clip(0, 255).astype(np.uint8)


def load_display_image(path, lut: str = LUT_NONE, quarter_turns: int = 0) -> Image.Image:
    """One crop as a PIL image, through the display LUT and the roll
    correction -- what a mosaic or a report pastes. Raises OSError or
    ValueError for a file that cannot be read, like load_and_encode_png."""
    return Image.open(io.BytesIO(load_and_encode_png(Path(path), lut or LUT_NONE, quarter_turns)))


# ---- crops served to the browser -----------------------------------------
# Shared by every session in the process -- every browser tab -- on purpose:
# the entry is a pure function of the file and the display transform, so a
# crop encoded for one tab is the right answer for all of them. The key
# carries the file's size and mtime, so an image rewritten on disk is never
# served stale, and the lock keeps the LRU safe across request threads.
_CROP_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_CROP_CACHE_MAX = 3000
_CROP_LOCK = threading.Lock()
# Formats a browser decodes natively -- served as-is, byte for byte.
# Anything else (TIFF above all) is re-encoded to PNG first.
NATIVE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
PLACEHOLDER_SVG = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" '
                   b'fill="#eceef2"/><text x="16" y="21" font-size="14" text-anchor="middle" fill="#8a93a6">?</text></svg>')


def crop_bytes(path: str, lut: str, turns: int) -> tuple:
    """(bytes, mime) for a crop: the file's own bytes when nothing has to
    change (PNG/JPEG, no LUT, no turn), else a lossless PNG re-encode
    through the display transform. A file that cannot be read gives the
    placeholder, never an error -- one broken crop must not fail a page.

    This is a computer-vision tool: crops are always served at their
    original resolution, since downscaling would hide exactly the detail
    people open them to see."""
    p = Path(path)
    try:
        st = p.stat()
        version = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return PLACEHOLDER_SVG, "image/svg+xml"
    key = f"{path}\x00{version}\x00{lut}\x00{turns}"
    with _CROP_LOCK:
        hit = _CROP_CACHE.get(key)
        if hit is not None:
            _CROP_CACHE.move_to_end(key)
            return hit
    try:
        suffix = p.suffix.lower()
        if (lut or LUT_NONE) == LUT_NONE and not turns and suffix in NATIVE_MIME:
            out = (p.read_bytes(), NATIVE_MIME[suffix])
        else:
            out = (load_and_encode_png(p, lut or LUT_NONE, turns), "image/png")
    except (OSError, ValueError) as ex:
        print(f"[lookin] could not load image {path}: {ex}", flush=True)
        return PLACEHOLDER_SVG, "image/svg+xml"
    with _CROP_LOCK:
        _CROP_CACHE[key] = out
        while len(_CROP_CACHE) > _CROP_CACHE_MAX:
            _CROP_CACHE.popitem(last=False)
    return out
