"""Small reusable helpers shared by the tab views: rendering matplotlib
figures and local image files as base64 Flet Images, and a stable color
per experiment name so comparison charts stay consistent across tabs.
"""
from __future__ import annotations

import base64
import io
import threading
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import flet as ft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# The qualitative palette lives in core/point_coloring.py, beside the code
# that decides what a projection point's color *means*: it is color
# arithmetic with no widget in it, and core must not import flet to reach
# it. Re-exported here because this is where the rest of the UI has always
# asked for a per-class color.
from lookin.core.point_coloring import QUALITATIVE, categorical_colors  # noqa: F401
# The chart style lives in core for the same reason: core/report_builder.py
# draws figures without ever importing Flet, and both halves have to agree
# on what a Lookin chart looks like. Re-exported here because this is where
# the tab views ask for everything else about a figure.
from lookin.core.chart_style import (  # noqa: F401
    CHART_SERIF_FONTS, GRID_BOTH, GRID_MAJOR, GRID_MAJOR_STYLE, GRID_MINOR_STYLE, GRID_NONE,
    LEGEND_MAX_COLS, apply_chart_style, legend_below, legend_columns, legend_height_inches,
    reserve_legend_space, style_axes,
)
from lookin.ui.image_export import publish_png, slug
from lookin.core.chart_style import subplots


_PLACEHOLDER_PNG_B64 = None

ACCENT = "#6C5CE7"


INLINE_DPI = 110  # what the chart is drawn at where it sits in the page
ZOOM_DPI = 200    # ...and again for the click-to-enlarge dialog


def _encode(fig, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fig_to_base64(fig, dpi: int = INLINE_DPI) -> str:
    b64 = _encode(fig, dpi)
    plt.close(fig)
    return b64


def fig_to_image(fig, **image_kwargs) -> ft.Image:
    b64 = fig_to_base64(fig)
    return ft.Image(src_base64=b64, fit=ft.ImageFit.CONTAIN, **image_kwargs)


def fig_to_zoomable_image(fig, ctx, title: str, **image_kwargs) -> ft.Control:
    """Same inline chart as fig_to_image, but clicking it opens the
    full-size zoomable view (see ImageZoomDialog).

    The figure is encoded twice -- once at INLINE_DPI for the page, once at
    ZOOM_DPI for the dialog -- and both encodings have to happen here,
    before the figure is closed: enlarging the inline PNG instead would just
    upscale a 110-dpi bitmap and look soft. The extra encoding is ~4x the
    bytes of the inline one, which is why it is done per figure rather than
    kept for every figure ever rendered.

    `ctx` is the AppContext; passing it in is what gives a plain helper
    access to the page-level dialog. Falls back to a non-clickable image if
    there is no context (keeps the helper usable from tests/scripts).
    """
    inline_b64 = _encode(fig, INLINE_DPI)
    zoom_b64 = _encode(fig, ZOOM_DPI)
    plt.close(fig)

    image = ft.Image(src_base64=inline_b64, fit=ft.ImageFit.CONTAIN, **image_kwargs)
    if ctx is None or getattr(ctx, "image_zoom", None) is None:
        return image
    return ft.Container(
        content=image, ink=True, border_radius=8,
        tooltip=f"{title} — click to enlarge",
        on_click=lambda e: ctx.image_zoom.open(zoom_b64, title),
    )


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

# The "no roll column chosen" key, held in ctx.display_roll_col.
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


def crop_display(ctx, record) -> tuple:
    """The (lut, quarter_turns) this session shows `record`'s crop through.

    One call rather than two so a view that draws crops cannot end up
    honouring the LUT while ignoring the roll. Everything is read
    defensively: a context without the display attributes (an older one, or
    a test double) means "as stored", never an AttributeError from a
    rendering path.
    """
    lut = (getattr(ctx, "display_lut", LUT_NONE) or LUT_NONE) if ctx is not None else LUT_NONE
    return lut, roll_quarter_turns(ctx, record)


def roll_quarter_turns(ctx, record) -> int:
    """Quarter turns for one crop under the session's roll setting.

    A column that is not in `record` -- a setting left over from a
    previously loaded dataset, or a record built from a subset of the
    columns -- yields no rotation rather than an error.
    """
    if ctx is None or record is None:
        return 0
    column = getattr(ctx, "display_roll_col", ROLL_NONE) or ROLL_NONE
    if not column:
        return 0
    try:
        value = record[column]
    except (KeyError, IndexError, TypeError):
        return 0
    return quarter_turns_for_roll(value)


# Shared by every session in the process -- every browser tab -- on purpose:
# the entry is a pure function of the file and the display transform, so a
# crop encoded for one tab is the right answer for all of them. Two things
# make sharing it safe: the key carries the file's size and mtime, so an
# image rewritten on disk is never served stale to anyone, and the lock,
# since tabs and their worker threads read and evict it concurrently and an
# OrderedDict is not safe to reorder from two threads at once.
_IMAGE_CACHE: "OrderedDict[str, str]" = OrderedDict()
_IMAGE_CACHE_MAX = 1500  # bounds memory; raw full-res images are much larger than thumbnails were
_IMAGE_CACHE_LOCK = threading.Lock()

# Formats the browser can decode natively from raw bytes -- served as-is,
# byte for byte, with zero processing. Anything else (TIFF above all,
# common for scientific/microscopy crops) can't be displayed by a browser
# at all and has to be re-encoded to PNG first, see _load_and_encode below.
_NATIVE_WEB_SUFFIXES = {".png", ".jpg", ".jpeg"}


def file_to_full_base64(path: Path, lut: str = LUT_NONE, quarter_turns: int = 0) -> str:
    """Full-resolution base64 for a crop image, cached (LRU-ish).

    This is a computer-vision tool -- silently downscaling crops would hide
    exactly the detail users need to see, so images are always served at
    their original resolution. The scale trade-off instead lives in the
    callers: the gallery mosaic and the carousel (ui/carousel.py) both
    encode a crop only when the user scrolls near it, a screenful at a
    time, rather than materializing a whole match set.

    PNG/JPEG are already something a browser can decode, so those are
    passed through completely unmodified (same bytes, no re-encoding, no
    value changes). TIFF -- along with anything else a browser can't show
    natively -- gets losslessly re-encoded to PNG (container format only,
    pixel values untouched) *unless* it's 16-bit, which browsers can't
    display at all (no 16-bit-per-channel web image format exists): those
    get a per-image min/max normalization down to 8-bit first. That's the
    one place this function ever changes pixel values, and only because
    there is no way to show 16-bit imagery in a browser otherwise.

    `lut` is the viewer's display mapping (see LUT_OPTIONS). At LUT_NONE the
    pass-through above still applies -- the bytes are served untouched. Any
    other value is a *display* transform: it changes what the screen shows,
    never the file, and the cache is keyed on it so switching LUT does not
    return a stale rendering of the previous one.

    `quarter_turns` is the roll correction (see quarter_turns_for_roll),
    counted counter-clockwise. It is a display transform in exactly the
    same sense, and it is lossless: a quarter turn moves pixels without
    resampling any of them. A non-zero one does cost the pass-through
    above, since rotated bytes are by definition not the file's bytes.
    """
    lut = lut or LUT_NONE
    quarter_turns = int(quarter_turns) % 4
    try:
        stat = path.stat()
        version = f"{stat.st_size}\x00{stat.st_mtime_ns}"
    except OSError:
        version = ""
    key = f"{path}\x00{version}\x00{lut}\x00{quarter_turns}"
    with _IMAGE_CACHE_LOCK:
        cached = _IMAGE_CACHE.get(key)
        if cached is not None:
            _IMAGE_CACHE.move_to_end(key)
            return cached

    try:
        if lut == LUT_NONE and not quarter_turns and path.suffix.lower() in _NATIVE_WEB_SUFFIXES:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        else:
            b64 = base64.b64encode(_load_and_encode_png(path, lut, quarter_turns)).decode("ascii")
    except (OSError, ValueError) as ex:
        print(f"[lookin] could not load image {path}: {ex}")
        b64 = _placeholder_base64()

    with _IMAGE_CACHE_LOCK:
        _IMAGE_CACHE[key] = b64
        while len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
            _IMAGE_CACHE.popitem(last=False)
    return b64


def _load_and_encode_png(path: Path, lut: str = LUT_NONE, quarter_turns: int = 0) -> bytes:
    """Open any Pillow-readable image (TIFF, BMP, WEBP, ...), apply the
    display LUT and the roll correction, and re-encode to PNG bytes.

    16-bit data is always brought down to 8 bits, LUT or not, because no
    web image format carries 16-bit channels -- with no LUT selected that
    is a plain min/max stretch, which is the only way to show the data at
    all. This is the one place pixel *values* change, and only ever for
    display: nothing here writes back to the file."""
    img = Image.open(path)
    arr = np.array(img)

    if lut != LUT_NONE:
        arr = _apply_lut(arr, lut)
        img = Image.fromarray(arr)
    elif arr.dtype == np.uint16:
        arr = _stretch(arr, float(arr.min()), float(arr.max()))
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


def _placeholder_base64() -> str:
    """The one figure this module can build from the UI thread while a
    background rebuild is drawing its own -- hence the shared lock."""
    global _PLACEHOLDER_PNG_B64
    if _PLACEHOLDER_PNG_B64 is None:
        with _HEAVY_WORK_LOCK:
            if _PLACEHOLDER_PNG_B64 is not None:
                return _PLACEHOLDER_PNG_B64
            fig, ax = subplots(figsize=(1, 1))
            ax.axis("off")
            ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=20)
            _PLACEHOLDER_PNG_B64 = fig_to_base64(fig, dpi=60)
    return _PLACEHOLDER_PNG_B64


def blues_color(fraction: float) -> str:
    """Interpolate from a light to a dark blue, like matplotlib's Blues cmap."""
    fraction = max(0.0, min(1.0, fraction))
    r0, g0, b0 = 224, 231, 255
    r1, g1, b1 = 30, 58, 138
    r = int(r0 + (r1 - r0) * fraction)
    g = int(g0 + (g1 - g0) * fraction)
    b = int(b0 + (b1 - b0) * fraction)
    return f"#{r:02x}{g:02x}{b:02x}"


def diverging_color(improvement: float, max_abs: float) -> str:
    """Green for better, red for worse, neutral in the middle.

    Takes an *improvement*, not a raw difference: on a difference matrix's
    diagonal more is better and everywhere else more is worse, so the sign
    of the count change is not the sign of the news (see
    core.comparison.cell_improvement). Callers convert first; this only
    picks the color.
    """
    if not max_abs:
        return "#2a2a30"
    fraction = max(-1.0, min(1.0, improvement / max_abs))
    neutral = (42, 42, 48)
    target = (34, 139, 87) if fraction >= 0 else (190, 60, 60)  # sea green / brick
    weight = abs(fraction)
    r, g, b = (int(n + (t - n) * weight) for n, t in zip(neutral, target))
    return f"#{r:02x}{g:02x}{b:02x}"


def color_for_experiment(name: str, all_names: list) -> str:
    """One color per experiment, assigned by its position in `all_names`.

    `all_names` is the user's own experiment order (AppState.experiments is
    ordered, and the Metrics tab lets that order be dragged around), so the
    palette runs top-to-bottom through the list the user arranged: the
    first experiment is always the first color, and a legend read in order
    walks the palette in order. This used to key off `sorted(all_names)`
    instead, which meant the colors were pinned to alphabetical order no
    matter how the charts were arranged -- and shuffled anyway the moment a
    newly loaded experiment sorted into the middle of the list.
    """
    if name not in all_names:
        all_names = list(all_names) + [name]
    # QUALITATIVE opens with exactly the ten tab10 colors this used to be
    # limited to, then keeps going -- so the first ten experiments look
    # unchanged, and an eleventh no longer silently reuses the first one's
    # color in every legend and confusion-matrix header.
    idx = list(all_names).index(name) % len(QUALITATIVE)
    r, g, b = QUALITATIVE[idx]
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def info_button(ctx, manual_page, tooltip: str | None = None, icon_size: int = 17) -> ft.Control:
    """The (i) that opens the reference manual at `manual_page`.

    `manual_page` is a key, never shown to the user and compared against
    the page ids in core/manual/ -- so it is never reworded (see CLAUDE.md
    on keys vs labels). A page id that no longer exists lands the reader on
    the manual's front page rather than raising, and tests/test_manual.py
    fails on it, which is where a rename is meant to be caught.

    `ctx` may be a context without a manual attached (the view tests build
    exactly that), in which case the button is still drawn and does
    nothing: a chart that refuses to render because its help is missing
    would be a strange trade.
    """
    return ft.IconButton(
        ft.Icons.INFO_OUTLINE,
        icon_size=icon_size,
        icon_color=ft.Colors.ON_SURFACE_VARIANT,
        tooltip=tooltip or "What is this? Open the manual",
        on_click=lambda e: _open_manual(ctx, manual_page),
        style=ft.ButtonStyle(padding=2),
    )


def _open_manual(ctx, manual_page) -> None:
    manual = getattr(ctx, "manual", None)
    if manual is None:
        return
    # A callable page id, resolved at click time rather than at build time:
    # the section it belongs to draws several different things depending on
    # a dropdown (the per-class metric, the candidate strategy), and the
    # right page to open is the one for what is on screen *now*.
    manual.open(manual_page() if callable(manual_page) else manual_page)


def section_title(text: str, icon: str | None = None, ctx=None, manual=None) -> ft.Control:
    """A section heading, with the (i) that explains it pinned to the right.

    The (i) hangs off the heading rather than off each chart under it
    because a section is the unit a reader asks about ("what is this
    reliability diagram?"), and one button per number would put forty of
    them on the Metrics tab.
    """
    label = ft.Text(text, size=16, weight=ft.FontWeight.BOLD)
    head = label if icon is None else ft.Row([ft.Icon(icon, size=18, color=ACCENT), label], spacing=8)
    if manual is None:
        return head
    return ft.Row([head, ft.Container(expand=True), info_button(ctx, manual)],
                  spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def info_banner(text: str, icon=None) -> ft.Container:
    icon = icon or ft.Icons.INFO_OUTLINE
    return ft.Container(
        content=ft.Row([ft.Icon(icon, size=16), ft.Text(text, size=12, italic=True, expand=True)], spacing=8),
        padding=12,
        border_radius=10,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
    )


STATUS_PALETTE = {
    "success": (ft.Colors.GREEN, ft.Icons.CHECK_CIRCLE_OUTLINE),
    "error": (ft.Colors.RED, ft.Icons.ERROR_OUTLINE),
    "warning": (ft.Colors.ORANGE, ft.Icons.WARNING_AMBER_OUTLINED),
    "neutral": (ft.Colors.ON_SURFACE_VARIANT, ft.Icons.RADIO_BUTTON_UNCHECKED),
}


def status_row(text: str, kind: str) -> ft.Control:
    # "busy" gets a spinner rather than an icon: a static glyph next to
    # "Loading..." says the same thing whether the load is running or has
    # been stuck for a minute, which is precisely the question being asked.
    if kind == "busy":
        return ft.Row([
            ft.ProgressRing(width=14, height=14, stroke_width=2, color=ACCENT),
            ft.Text(text, color=ft.Colors.ON_SURFACE_VARIANT, size=12),
        ], spacing=6)
    color, icon = STATUS_PALETTE.get(kind, STATUS_PALETTE["neutral"])
    return ft.Row([ft.Icon(icon, color=color, size=15), ft.Text(text, color=color, size=12)], spacing=6)


class BusyIndicator:
    """The app bar's "the app is working on something" light.

    Every tab has its own progress banner, and every one of them is invisible
    from the other tabs -- so a user who starts a 500k-row load on the Data
    tab and goes to look at Metrics while it reads has no way to tell
    whether the app is still loading, has finished, or has failed: Metrics
    looks exactly as it does when nothing at all is happening. This sits in
    the app bar, which every tab shares, and says what is running and how
    far it has got.

    It does not replace the in-tab progress: that says *what* is being
    read, and stays. This answers the smaller question -- is anything still
    going on? -- from wherever the user happens to be looking.

    begin/end nest, because loading a session is one dataset followed by
    several experiments and each of those announces itself: what is shown
    is the innermost task, and the light only goes out when the last one
    ends. It is called from worker threads as well as the UI thread (see
    BackgroundRebuild), so the small amount of state here is locked.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stack: list[str] = []
        self.text = ft.Text("", size=13, color=ACCENT, weight=ft.FontWeight.BOLD)
        self.detail = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.control = ft.Container(
            content=ft.Row([
                ft.ProgressRing(width=14, height=14, stroke_width=2, color=ACCENT),
                ft.Column([self.text, self.detail], spacing=0, tight=True),
            ], spacing=8),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=8,
            visible=False,
            tooltip="Lookin is busy -- the rest of the app stays usable while it works",
        )

    def begin(self, what: str) -> None:
        with self._lock:
            self._stack.append(what)
        self._render()

    def detail_text(self, detail: str) -> None:
        """Progress *within* the running task -- rows read so far, say."""
        self.detail.value = detail
        self._push()

    def end(self) -> None:
        with self._lock:
            if self._stack:
                self._stack.pop()
        self._render()

    def _render(self) -> None:
        with self._lock:
            current = self._stack[-1] if self._stack else ""
        self.control.visible = bool(current)
        self.text.value = current
        # The detail belongs to the task that just ended, so it goes with
        # it rather than lingering under the next one's name.
        self.detail.value = ""
        self._push()

    def _push(self) -> None:
        if self.control.page:
            self.control.update()


def uuid_overlap_banner(overlap, dataset_label: str = "dataset", experiment_label: str = "experiment") -> ft.Control | None:
    """A warning banner describing a dataset<->experiment uuid mismatch, or
    None if there's nothing to warn about. `overlap` is a
    lookin.core.data_manager.UuidOverlap. Shared between the Data tab
    (right after loading an experiment) and the Metrics tab (before/instead
    of computing metrics), since the same mismatch matters in both places."""
    lines = []
    if overlap.dataset_only:
        lines.append(f"{overlap.dataset_only:,} {dataset_label} rows have no matching {experiment_label} row")
    if overlap.experiment_only:
        lines.append(f"{overlap.experiment_only:,} {experiment_label} rows have no matching {dataset_label} row")
    if overlap.dataset_dupes:
        lines.append(f"{dataset_label} uuid column has {overlap.dataset_dupes:,} duplicate value(s)")
    if overlap.experiment_dupes:
        lines.append(f"{experiment_label} uuid column has {overlap.experiment_dupes:,} duplicate value(s)")
    if not lines:
        return None
    return ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.WARNING_AMBER_OUTLINED, color=ft.Colors.ORANGE, size=16),
            ft.Column([ft.Text(line, size=12, color=ft.Colors.ORANGE) for line in lines], spacing=2, expand=True),
        ], spacing=8),
        padding=10,
        border_radius=8,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
    )


def score_diagnostics_banner(diagnostics, experiment_label: str = "experiment") -> ft.Control | None:
    """A warning banner naming every metric that was disabled for one
    experiment and why, or None if nothing was disabled. `diagnostics` is a
    lookin.core.metrics.ScoreDiagnostics.

    The disabled metrics show up as "--" in the Summary table, which on its
    own is indistinguishable from "the computation failed" -- this is what
    turns that into an explanation the user can act on (renormalize the
    export, or accept that a thresholded class has no ranking)."""
    lines = []
    if not diagnostics.normalized:
        lines.append(
            f"score columns do not sum to 1 ({diagnostics.n_rows_not_normalized:,} of "
            f"{diagnostics.n_rows:,} rows, worst off by {diagnostics.max_row_sum_deviation:.3f})"
        )
        lines.append(
            "  -> disabled: log loss, Brier score and its decomposition, ECE / MCE, reliability diagram"
        )
    if diagnostics.constant_labels:
        named = ", ".join(
            f"'{label}' (constant {diagnostics.constant_values[label]:g})"
            for label in diagnostics.constant_labels
        )
        lines.append(f"score column(s) with no ranking: {named}")
        lines.append(
            "  -> disabled for those classes: ROC-AUC, average precision, ROC / PR curves. "
            "They are also excluded from the macro and weighted averages."
        )
    if not lines:
        return None
    return ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.RULE, color=ft.Colors.ORANGE, size=16),
            ft.Column(
                [ft.Text(f"{experiment_label}: {lines[0]}", size=12, color=ft.Colors.ORANGE)]
                + [ft.Text(line, size=12, color=ft.Colors.ORANGE) for line in lines[1:]],
                spacing=2, expand=True,
            ),
        ], spacing=8),
        padding=10,
        border_radius=8,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
    )


# Only one expensive rebuild runs at a time, process-wide.
#
# Two reasons, and either one alone would be enough. matplotlib's pyplot
# keeps a *global* figure registry, so two threads calling plt.subplots()
# concurrently can corrupt it -- and every heavy rebuild in this app ends in
# matplotlib. And these jobs are CPU-bound to begin with: running three at
# once would not finish any of them sooner, it would just make all three
# slower and the machine less responsive.
#
# Reentrant, because a rebuild already holding it can reach code that takes
# it again: the Gallery's rebuild renders its mosaic, and the first crop
# that fails to load builds the "?" placeholder under this same lock (see
# _placeholder_base64). With a plain Lock that thread waited on itself
# forever -- a Gallery spinning on "Rebuilding filters..." for good, and
# every other tab's rebuild queued behind it.
_HEAVY_WORK_LOCK = threading.RLock()


class BackgroundRebuild:
    """Runs one view's expensive rebuild off the Flet UI thread.

    Every one of these rebuilds already reports its own progress (see each
    view's `_set_progress`) and already sits behind an explicit "Update
    now" click -- but until this existed they all ran on the UI thread, so
    the spinner never painted, the progress text never appeared, and the
    whole app was frozen for as long as the work took (measured at ~19s for
    Metrics on a million rows, see scaling.md). The user had chosen to pay
    the cost; they just had no way to tell it was being paid.

    One job per view at a time, with the *last* request coalesced rather
    than dropped. Python cannot safely stop a running thread, so a second
    request while one is running cannot pre-empt it -- but it must not be
    thrown away either: these views mutate their own state (the list of
    active charts, the list of active filters) *before* asking for a
    rebuild, so a dropped request leaves the view showing something the
    user did not ask for, with no banner to say so. Instead the newest
    request is held and run as soon as the current one finishes, and any
    request that arrives in between replaces it -- clicking Add five times
    quickly costs two renders, not five, and the last one shows all five
    charts.

    Flet control mutations from the worker thread are fine and are what
    every one of these rebuilds does (ui/projection_view.py has worked this
    way for a while) -- it is only the *page* that must not be blocked.
    """

    def __init__(self, ctx, name: str):
        self.ctx = ctx
        self.name = name
        self._lock = threading.Lock()
        self._busy = False
        self._pending = None  # (work, on_error) to run once the current job ends

    @property
    def busy(self) -> bool:
        return self._busy

    def run(self, work, on_error=None) -> bool:
        """Ask for a rebuild. Returns True if it started immediately, False
        if it was queued behind the one already running."""
        with self._lock:
            if self._busy:
                self._pending = (work, on_error)
                return False
            self._busy = True

        self.ctx.page.run_thread(self._runner, work, on_error)
        return True

    def _runner(self, work, on_error):
        while True:
            # The app bar says what is running, so a rebuild started on one
            # tab is still visibly running from another -- the in-tab
            # progress banner only answers that question for whoever stayed
            # to watch it (see BusyIndicator).
            busy = getattr(self.ctx, "busy", None)
            if busy is not None:
                busy.begin(self.name)
            try:
                with _HEAVY_WORK_LOCK:
                    work()
            except Exception as ex:
                # A rebuild that dies silently leaves its progress banner
                # spinning forever, which is indistinguishable from a hang.
                print(f"[lookin] {self.name}: rebuild failed: {ex}", flush=True)
                traceback.print_exc()
                if on_error is not None:
                    try:
                        on_error(ex)
                    except Exception:
                        traceback.print_exc()
            finally:
                if busy is not None:
                    busy.end()
            with self._lock:
                if self._pending is None:
                    self._busy = False
                    return
                work, on_error = self._pending
                self._pending = None


class StaleBanner:
    """A persistent red 'this view is out of date' banner with an explicit
    Update button, for tabs whose rebuild is too expensive to redo on every
    data change or every tab switch (Statistics/Gallery/Metrics). The view
    marks itself dirty and calls show() instead of re-rendering; only the
    user clicking the button actually triggers the (possibly slow) rebuild,
    via the on_update callback passed at construction."""

    def __init__(self, on_update, message: str = "Data has changed -- this view is out of date."):
        self.text = ft.Text(message, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=13, expand=True)
        self.button = ft.FilledButton(
            "Update now", icon=ft.Icons.REFRESH, on_click=on_update,
            style=ft.ButtonStyle(bgcolor=ft.Colors.WHITE, color=ft.Colors.RED_700),
        )
        self.control = ft.Container(
            content=ft.Row(
                [ft.Icon(ft.Icons.WARNING_ROUNDED, color=ft.Colors.WHITE), self.text, self.button],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=14, border_radius=10, bgcolor=ft.Colors.RED_700, visible=False,
        )

    def show(self) -> None:
        self.control.visible = True
        if self.control.page:
            self.control.update()

    def hide(self) -> None:
        self.control.visible = False
        if self.control.page:
            self.control.update()


def _default_save_dir() -> Path:
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.cwd()


def save_png_to_disk(b64: str, title: str, ctx=None) -> tuple[bool, str]:
    """Write a base64 PNG next to the user's other downloads and report what
    happened. Saving is server-side on purpose: this runs on the machine the
    data is loaded from (same reasoning as ui/server_browser.py), and a
    browser-side download would mean navigating to a data: URI, which Chrome
    blocks for top-level navigation."""
    dest = _default_save_dir() / f"lookin_{slug(title)}_{datetime.now():%Y%m%d-%H%M%S}.png"
    try:
        dest.write_bytes(base64.b64decode(b64))
    except (OSError, ValueError) as ex:
        return False, f"Could not save: {ex}"
    return True, f"Saved to {saved_path_message(ctx, dest)}"


# What a status line adds once the path is on the clipboard. Appended rather
# than replacing the path: the path is still the thing the user reads to
# check *where* it went, and a message that only said "copied" would hide it.
CLIPBOARD_NOTE = " (path copied to the clipboard)"


def copy_to_clipboard(ctx, text) -> bool:
    """Put `text` on the clipboard of the browser the user is sitting at.

    Only text, and only that direction -- which is exactly enough for a
    path. The file itself lives on the machine the app runs on, normally
    reached over a forwarded port (see ui/image_export.py for the same
    asymmetry applied to images), so what the user needs after a save is the
    string to paste into their own terminal, scp or file dialog, not the
    bytes.

    Never fatal: a page that is gone, or a browser that refuses the
    clipboard because the write did not follow a click, must not turn a
    successful save into a failed one.
    """
    page = getattr(ctx, "page", None) if ctx is not None else None
    if page is None:  # a panel built without a context, i.e. a test
        return False
    try:
        page.set_clipboard(str(text))
    except Exception as ex:  # noqa: BLE001 -- see above: best effort, always
        print(f"[lookin] could not copy to the clipboard: {ex}", flush=True)
        return False
    return True


def saved_path_message(ctx, path) -> str:
    """The path a "wrote it there" status line should show, copied to the
    clipboard on the way out.

    One helper rather than a copy call at each of the half-dozen places that
    write a file, so that a new export cannot end up being the one that
    silently does not copy.
    """
    return f"{path}{CLIPBOARD_NOTE}" if copy_to_clipboard(ctx, path) else str(path)


# What the status line says once the tab has been asked for. The tab is
# opened by the browser, not by us, so "opened" is a claim we cannot check
# -- hence the second half, which is the only thing the user can act on if
# a pop-up blocker swallowed it.
OPENED_IN_TAB = ("Opened in a new tab -- right-click the image there to copy or save it. "
                 "If no tab opened, allow pop-ups for this page.")


def open_png_in_tab(ctx, b64: str, title: str) -> tuple[bool, str]:
    """Serve a base64 PNG from the app and open it in a new browser tab,
    where it is a real image the browser can copy and save (see
    ui/image_export.py for why that is the only way to reach the user's own
    clipboard).

    web_window_name is spelled out rather than left to url_launcher's
    default: the whole point is a *second* tab, and navigating the current
    one away would take the loaded dataset and every open panel with it.
    """
    page = getattr(ctx, "page", None) if ctx is not None else None
    if page is None:  # a carousel built without a context, i.e. a test
        return False, "Not attached to a page."
    ok, result = publish_png(b64, title)
    if not ok:
        return False, result
    page.launch_url(result, web_window_name="_blank")
    return True, OPENED_IN_TAB


def build_zoom_overlay(width: int, height: int, on_back, on_save, on_open) -> tuple:
    """A full-size zoomable image layer meant to be stacked *inside* an
    already-open dialog, rather than shown as a dialog of its own.

    This exists because Flet shows only one dialog at a time: opening a
    second one dismisses the first (firing its on_dismiss), and reopening
    the first from the second's close handler -- while Flutter is still
    animating the route out -- leaves the page's control tree corrupted, so
    the content behind goes blank. Overlaying inside the dialog that is
    already open avoids the route change entirely, which also makes it
    instant: nothing is torn down and rebuilt.

    Returns (container, image) so the caller can swap the image source.
    """
    image = ft.Image(fit=ft.ImageFit.CONTAIN, width=width - 24, height=height - 64)
    viewer = ft.InteractiveViewer(
        content=image, min_scale=1.0, max_scale=8.0,
        scale_enabled=True, pan_enabled=True, clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    status = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT, expand=True)
    container = ft.Container(
        visible=False, width=width, height=height, padding=8, border_radius=10,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        content=ft.Column([
            ft.Row([
                ft.TextButton("Back", icon=ft.Icons.ARROW_BACK, on_click=on_back),
                ft.Text("Scroll to zoom, drag to pan.", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                status,
                ft.TextButton("Open in a tab", icon=ft.Icons.OPEN_IN_NEW, on_click=on_open),
                ft.TextButton("Save PNG", icon=ft.Icons.DOWNLOAD_OUTLINED, on_click=on_save),
            ], spacing=10),
            viewer,
        ], spacing=4, tight=True),
    )
    container.data = status  # the caller needs it to report save results
    return container, image


class ImageZoomDialog:
    """Full-size viewer for charts, which are clicked straight from the page
    with no dialog already open. Built once and hung off the AppContext.

    Crops inside a carousel deliberately do NOT come through here -- they
    are already inside a dialog, and a second dialog cannot be stacked on
    the first (see build_zoom_overlay).

    The content sits in an ft.InteractiveViewer, which is what supplies
    scroll-to-zoom and drag-to-pan. That covers what the matplotlib
    navigation toolbar is actually used for; the toolbar itself belongs to
    matplotlib's interactive backends, and this app renders through Agg to
    static PNGs on the server, so it is not available here.
    """

    MAX_W, MAX_H = 1180, 760  # upper bound; the real size adapts to the window
    CHROME_H = 150            # dialog title + hint line + action row

    def __init__(self, ctx):
        self.ctx = ctx
        self._b64: str | None = None
        self._title = "image"

        # The image carries explicit dimensions rather than relying on
        # expand/fit: an ft.Image with no intrinsic size inside an
        # InteractiveViewer collapses to nothing and the dialog opens empty.
        self.image = ft.Image(fit=ft.ImageFit.CONTAIN, width=self.MAX_W, height=self.MAX_H - self.CHROME_H)
        self.viewer = ft.InteractiveViewer(
            content=self.image, min_scale=1.0, max_scale=8.0,
            scale_enabled=True, pan_enabled=True, clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        self.status = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT, expand=True)
        self.title_text = ft.Text(weight=ft.FontWeight.BOLD)
        self.body = ft.Container(
            content=ft.Column([
                ft.Text("Scroll to zoom, drag to pan.", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Container(content=self.viewer, expand=True, border_radius=8, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST),
            ], spacing=8, tight=False),
        )
        self.dialog = ft.AlertDialog(
            modal=False,  # tapping outside dismisses, same as Close
            title=self.title_text,
            content=self.body,
            actions=[
                self.status,
                ft.TextButton("Open in a tab", icon=ft.Icons.OPEN_IN_NEW, on_click=self._on_open),
                ft.TextButton("Save PNG", icon=ft.Icons.DOWNLOAD_OUTLINED, on_click=self._on_save),
                ft.TextButton("Close", on_click=self.close),
            ],
        )

    # Kept clear of the window edges so a comfortable strip of barrier stays
    # clickable -- with a near-full-width dialog there is nowhere left to
    # click "outside" to dismiss it.
    WINDOW_MARGIN_W, WINDOW_MARGIN_H = 220, 200

    def _fit_to_window(self) -> None:
        """Size the dialog to the browser window, capped at MAX_W/MAX_H. A
        fixed 1180x760 overflows a smaller window and pushes the image out of
        view entirely."""
        page = self.ctx.page
        avail_w = getattr(page, "width", None) or self.MAX_W
        avail_h = getattr(page, "height", None) or self.MAX_H
        w = max(320, min(self.MAX_W, int(avail_w) - self.WINDOW_MARGIN_W))
        h = max(240, min(self.MAX_H, int(avail_h) - self.WINDOW_MARGIN_H))
        self.body.width, self.body.height = w, h
        self.image.width, self.image.height = w, h - self.CHROME_H

    def open(self, b64: str, title: str = "image") -> None:
        """Only ever called with no dialog already open (charts are clicked
        from the page). Nothing to stack, nothing to restore."""
        self._b64, self._title = b64, title
        self.title_text.value = title
        self.image.src_base64 = b64
        self.status.value = ""
        self.status.color = ft.Colors.ON_SURFACE_VARIANT
        self._fit_to_window()
        self.ctx.page.open(self.dialog)

    def close(self, e=None) -> None:
        self.ctx.page.close(self.dialog)

    def _on_save(self, e) -> None:
        if self._b64:
            self._report(*save_png_to_disk(self._b64, self._title, self.ctx))

    def _on_open(self, e) -> None:
        if self._b64:
            self._report(*open_png_in_tab(self.ctx, self._b64, self._title))

    def _report(self, ok: bool, message: str) -> None:
        self.status.value = message
        self.status.color = ft.Colors.GREEN if ok else ft.Colors.RED
        if self.status.page:
            self.status.update()


# card()'s own padding, left plus right: what a card adds around the grid
# it holds (see grid_card_width).
CARD_PADDING_WIDTH = 36
# What a tab's scrolling column leaves for its cards: the root Container's
# 20px padding on each side, plus room for the page scrollbar.
TAB_CHROME_WIDTH = 56


# The number of columns a confusion matrix is sized against when it has
# fewer: a two-class matrix draws four big cells instead of four small ones
# lost in a corner of the tab. Above this the cells keep their base size and
# the grid grows, which is what a many-class matrix needs.
MATRIX_FLOOR_COLUMNS = 8


def matrix_cell_size(n_columns: int, base: int, spacing: int = 2) -> int:
    """Cell pitch for a confusion matrix with `n_columns` columns.

    At MATRIX_FLOOR_COLUMNS columns or more this is `base`, so a matrix
    with many classes is as wide as it has to be. Below it, the cells take
    the room those columns would have used: the grid keeps the footprint of
    an eight-class one rather than shrinking to a thumbnail, which is what
    a binary matrix -- the most common case there is -- used to do.
    """
    if n_columns <= 0 or n_columns >= MATRIX_FLOOR_COLUMNS:
        return base
    return int(round(MATRIX_FLOOR_COLUMNS * (base + spacing) / n_columns)) - spacing


def matrix_text_size(cell_size: int, base_cell: int, base_size: int, cap: int = 18) -> float:
    """`base_size`, grown with the cell it sits in (see matrix_cell_size).

    Counts drawn at 10px inside a 140px cell would be a grid of small
    numbers floating in colour. Capped, because past a point a bigger digit
    stops being easier to read.
    """
    if cell_size <= base_cell or base_cell <= 0:
        return base_size
    return min(cap, round(base_size * cell_size / base_cell))


def matrix_label_chars(cell_size: int, base_cell: int, minimum: int) -> int:
    """How many characters of a class label a column header keeps.

    Headers are truncated to keep the grid on its pitch. `minimum` is what
    fits at the base pitch, and a cell that grew shows proportionally more
    -- the text in it grew too (see matrix_text_size), so the ratio is the
    honest estimate. The full label is in the cell's tooltip either way.
    """
    if base_cell <= 0 or cell_size <= base_cell:
        return minimum
    return max(minimum, int(minimum * cell_size / base_cell))


def grid_card_width(page, grid_width: int, minimum: int = 240) -> int:
    """Width for a card holding a fixed-pitch grid -- a confusion matrix.

    The grid's own width, so a small matrix stays small and several of them
    sit side by side in a wrapping row, growing up to the whole tab for a
    matrix with many classes. A fixed cap did the opposite of what it was
    meant to: capping at 460px kept a row of matrices tidy for three classes
    and, past ten, showed ten columns out of a hundred behind a scrollbar,
    row labels scrolling away with them. Only a matrix wider than the window
    scrolls now, and it does so with the whole width already spent.

    page.width is None before the first frame, so the declared window size
    (see app.main) stands in.
    """
    width = getattr(page, "width", None) or 1440
    available = max(minimum, int(width) - TAB_CHROME_WIDTH)
    return int(min(grid_width + CARD_PADDING_WIDTH, available))


def card(content: ft.Control, width: int | None = None, expand=None) -> ft.Container:
    return ft.Container(
        content=content,
        width=width,
        expand=expand,
        padding=18,
        border_radius=14,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        margin=ft.margin.only(bottom=4),
    )
