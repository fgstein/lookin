"""Crops and charts as URLs the page can load.

Crops: the page never sees a path. A view registers the file with its
session and gets a key; /img/<sid>/<key>?l=<lut>&t=<turns> serves it through
the session's display LUT and roll correction (see server.crop_bytes). The
LUT and the turn are in the URL, so changing either is a new URL and the
browser cannot show a stale rendering.

Charts: a matplotlib figure is rendered to PNG once, kept in the session's
chart store, and shown as an <img> at half its pixel size -- drawn at twice
the on-screen resolution, so it is sharp on any screen and the zoom view
(a click) shows it at full resolution.
"""
from __future__ import annotations

import io
import struct

from lookin.ui.components import INLINE_DPI, crop_display
from lookin.web import html as h

CHART_DPI = INLINE_DPI * 2


def crop_url(session, path, record=None) -> str:
    lut, turns = crop_display(session, record)
    key = session.image_key(path)
    return f"/img/{session.sid}/{key}?l={lut}&t={turns}"


def crop_img(session, path, record=None, *, size: int = 96, alt: str = "", cls: str = "",
             title: str = "", zoom: bool = False, act_attr: str = "", fit: str = "contain") -> str:
    url = crop_url(session, path, record)
    zoom_attr = f' data-zoom data-pixelated data-title="{h.e(alt or "crop")}"' if zoom else ""
    return (f'<img class="crop {h.e(cls)}" src="{h.e(url)}" alt="{h.e(alt)}" loading="lazy" '
            f'style="width:{size}px;height:{size}px;object-fit:{fit}"{zoom_attr}'
            f'{h.attrs(title=title or None)}{act_attr}>')


def fig_png(fig, dpi: int = CHART_DPI) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass
    return buf.getvalue()


def png_size(data: bytes) -> tuple:
    try:
        w, hgt = struct.unpack(">II", data[16:24])
        return int(w), int(hgt)
    except Exception:  # noqa: BLE001
        return 0, 0


def chart_html(session, png: bytes, title: str = "chart", *, max_width: str = "100%", scale: float = 0.5,
               key: str = "") -> str:
    """An <img> for a rendered chart; click to zoom."""
    ckey = session.put_chart(png, key or None)
    w, hgt = png_size(png)
    size = f' width="{int(w * scale)}" height="{int(hgt * scale)}"' if w else ""
    return (f'<img class="chart" src="/chart/{session.sid}/{ckey}.png"{size} alt="{h.e(title)}" '
            f'data-zoom data-title="{h.e(title)}" style="max-width:{max_width}" data-static="{ckey}">')


def fig_html(session, fig, title: str = "chart", **kw) -> str:
    return chart_html(session, fig_png(fig), title, **kw)
