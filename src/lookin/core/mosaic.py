"""A mosaic of crops as one picture -- what the Gallery shows, as a PNG.

For the places a page cannot go: an agent's answer, a slide, a message.
Each crop goes through the same display transform the browser shows it
through (core/images.py: the LUT and the roll correction), is fitted into
a square tile without being stretched, and gets an optional caption under
it and an optional coloured frame -- green for a right prediction, red for
a wrong one, as the Gallery draws them.

Pillow only, and the one font is matplotlib's own DejaVu Sans, so nothing
is fetched (see CLAUDE.md).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from lookin.core.images import LUT_NONE, load_display_image

MAX_TILES = 400
DEFAULT_TILE = 160
MIN_TILE, MAX_TILE = 48, 512
GAP = 4
FRAME = 3
BACKGROUND = (255, 255, 255)
CAPTION_COLOR = (30, 34, 41)
PLACEHOLDER_COLOR = (236, 238, 242)
# Frames, as the Gallery's outcome borders.
FRAME_RIGHT = (46, 158, 79)
FRAME_WRONG = (229, 57, 53)


@dataclass
class Tile:
    """One crop of a mosaic: where it is, how to show it, what to write
    under it, and whether the prediction on it was right (None: no frame)."""
    path: str
    caption: str = ""
    quarter_turns: int = 0
    correct: bool | None = None


def _font(size: int):
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except (OSError, ValueError, ImportError):
        return ImageFont.load_default()


def _fit(text: str, font, width: int, draw) -> str:
    """`text`, cut with an ellipsis to fit `width` pixels."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def _fitted(img, side: int):
    """`img` scaled to fit a `side`-pixel square, aspect kept. A crop
    smaller than the tile is enlarged by whole pixels (nearest neighbour):
    interpolating would invent detail that is not in the file, which is the
    one thing a crop viewer must not do. A larger one is reduced smoothly."""
    scale = min(side / img.width, side / img.height)
    if scale >= 1:
        factor = int(scale)
        return img.resize((img.width * factor, img.height * factor), Image.Resampling.NEAREST)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def render(tiles: list, columns: int = 0, tile_size: int = DEFAULT_TILE, lut: str = LUT_NONE,
           title: str = "") -> tuple:
    """(png bytes, n_unreadable): the tiles on a grid, `columns` wide
    (0: as square as the count allows). A crop that cannot be read is a
    grey tile, counted, never an error -- one broken file must not cost the
    whole picture."""
    tiles = list(tiles)[:MAX_TILES]
    if not tiles:
        raise ValueError("No crops to put in the mosaic.")
    tile_size = max(MIN_TILE, min(MAX_TILE, int(tile_size or DEFAULT_TILE)))
    columns = int(columns) if columns and int(columns) > 0 else math.ceil(math.sqrt(len(tiles)))
    columns = max(1, min(columns, len(tiles)))
    rows = math.ceil(len(tiles) / columns)

    caption_size = max(9, tile_size // 12)
    font = _font(caption_size)
    title_font = _font(caption_size + 4)
    has_captions = any(t.caption for t in tiles)
    caption_h = caption_size + 6 if has_captions else 0
    title_h = caption_size + 16 if title else 0
    cell_w, cell_h = tile_size + GAP, tile_size + caption_h + GAP
    canvas = Image.new("RGB", (columns * cell_w + GAP, rows * cell_h + GAP + title_h), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((GAP + 2, GAP + 2), _fit(title, title_font, canvas.width - 2 * GAP, draw),
                  fill=CAPTION_COLOR, font=title_font)

    unreadable = 0
    for i, tile in enumerate(tiles):
        x = GAP + (i % columns) * cell_w
        y = title_h + GAP + (i // columns) * cell_h
        inner = tile_size - 2 * FRAME if tile.correct is not None else tile_size
        offset = FRAME if tile.correct is not None else 0
        if tile.correct is not None:
            draw.rectangle([x, y, x + tile_size - 1, y + tile_size - 1],
                           fill=FRAME_RIGHT if tile.correct else FRAME_WRONG)
        try:
            img = load_display_image(tile.path, lut, tile.quarter_turns)
            img = img.convert("RGBA") if img.mode in ("LA", "PA", "RGBA") else img.convert("RGB")
            img = _fitted(img, inner)
            box = Image.new("RGB", (inner, inner), BACKGROUND)
            box.paste(img, ((inner - img.width) // 2, (inner - img.height) // 2),
                      img if img.mode == "RGBA" else None)
        except (OSError, ValueError):
            unreadable += 1
            box = Image.new("RGB", (inner, inner), PLACEHOLDER_COLOR)
            ImageDraw.Draw(box).text((inner // 2, inner // 2), "?", fill=(138, 147, 166), font=font, anchor="mm")
        canvas.paste(box, (x + offset, y + offset))
        if tile.caption:
            draw.text((x + 1, y + tile_size + 2), _fit(tile.caption, font, tile_size - 2, draw),
                      fill=CAPTION_COLOR, font=font)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), unreadable
