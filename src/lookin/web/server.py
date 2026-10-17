"""The HTTP side: one FastAPI app serving the page, the actions, the event
stream, crops, charts and downloads.

    GET  /                 the page (a shell; the content arrives with /api/boot)
    POST /api/boot         attach this tab to its session, apply ?workspace= / ?snapshot=
    POST /api/act          run one action, answer with what it changed
    GET  /api/render       re-render named fragments (after a pushed "dirty")
    GET  /api/events       server-sent events: what changed in the background
    GET  /img/{sid}/{key}  a crop, through the session's LUT and roll
    GET  /chart/{sid}/{key}.png
    GET  /dl/{sid}/{key}   a file the session prepared (report, export)
    GET  /static/...       the CSS, the JS, the fonts -- all local

Every route that touches data takes the tab's session id, and nothing is
looked up across sessions: see session.py for why that is the whole point.
"""
from __future__ import annotations

import asyncio
import io
import json
import mimetypes
import queue
import threading
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lookin import __version__
from lookin.web.session import RequestScope, SessionStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
SSE_KEEPALIVE_S = 15.0


class WebApp:
    """Process-wide settings from the command line, and the session store."""

    def __init__(self, dataset: Optional[str] = None, experiments=None):
        self.dataset = dataset
        self.experiments = list(experiments or [])
        self.store = SessionStore(self)


# ---- crops -----------------------------------------------------------------

_CROP_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_CROP_CACHE_MAX = 3000
_CROP_LOCK = threading.Lock()
_NATIVE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_PLACEHOLDER = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" '
                b'fill="#eceef2"/><text x="16" y="21" font-size="14" text-anchor="middle" fill="#8a93a6">?</text></svg>')


def crop_bytes(path: str, lut: str, turns: int) -> tuple:
    """(bytes, mime) for a crop: the file's own bytes when nothing has to
    change (PNG/JPEG, no LUT, no turn), else a lossless PNG re-encode
    through the display transform -- the same rules as the original UI
    (see lookin.ui.components.file_to_full_base64)."""
    from lookin.ui.components import LUT_NONE, _load_and_encode_png

    p = Path(path)
    try:
        st = p.stat()
        version = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return _PLACEHOLDER, "image/svg+xml"
    key = f"{path}\x00{version}\x00{lut}\x00{turns}"
    with _CROP_LOCK:
        hit = _CROP_CACHE.get(key)
        if hit is not None:
            _CROP_CACHE.move_to_end(key)
            return hit
    try:
        suffix = p.suffix.lower()
        if (lut or LUT_NONE) == LUT_NONE and not turns and suffix in _NATIVE:
            out = (p.read_bytes(), _NATIVE[suffix])
        else:
            out = (_load_and_encode_png(p, lut or LUT_NONE, turns), "image/png")
    except (OSError, ValueError) as ex:
        print(f"[lookin] could not load image {path}: {ex}", flush=True)
        return _PLACEHOLDER, "image/svg+xml"
    with _CROP_LOCK:
        _CROP_CACHE[key] = out
        while len(_CROP_CACHE) > _CROP_CACHE_MAX:
            _CROP_CACHE.popitem(last=False)
    return out


# ---- the page ----------------------------------------------------------------

def page_html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lookin</title>
<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/static/app.css?v={__version__}">
<script src="/static/app.js?v={__version__}" defer></script>
</head>
<body>
<div id="app"><div class="boot">Loading…</div></div>
<div id="dialogs"></div>
<div id="toasts" aria-live="polite"></div>
<div id="zoom"></div>
</body>
</html>"""


def create_app(webapp: WebApp) -> FastAPI:
    from lookin.web import render

    app = FastAPI(title="Lookin", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    store = webapp.store

    def no_session():
        return JSONResponse({"error": "no session", "reload": True}, status_code=410)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(page_html(), headers={"Cache-Control": "no-store"})

    @app.post("/api/boot")
    async def boot(request: Request):
        body = await request.json()
        session = store.get(str(body.get("sid", "")))
        if session is None:
            return no_session()
        query = parse_qs(str(body.get("search", "")).lstrip("?"))
        return await asyncio.to_thread(_boot, session, query, body)

    def _viewport(session, body):
        # The browser window, for the views that size a page of crops to
        # what fits on screen (the carousel); sent with every request.
        try:
            session.viewport = (max(320, int(body.get("vw") or 0)), max(320, int(body.get("vh") or 0)))
        except (TypeError, ValueError):
            pass

    def _boot(session, query, body=None):
        if body:
            _viewport(session, body)
        with RequestScope(session):
            try:
                render.boot(session, query)
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                session.toast(f"Could not open: {ex}", "error")
            return JSONResponse(render.collect(session, full=True))

    @app.post("/api/act")
    async def act(request: Request):
        body = await request.json()
        session = store.get(str(body.get("sid", "")), create=False)
        if session is None:
            return no_session()
        return await asyncio.to_thread(_act, session, body)

    def _act(session, body):
        target = str(body.get("target", ""))
        action = str(body.get("action", ""))
        args = body.get("args") or {}
        _viewport(session, body)
        with RequestScope(session):
            try:
                render.dispatch(session, target, action, args)
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                session.toast(f"{type(ex).__name__}: {ex}", "error")
            return JSONResponse(render.collect(session))

    @app.get("/api/render")
    def render_names(sid: str, names: str = ""):
        session = store.get(sid, create=False)
        if session is None:
            return no_session()
        with RequestScope(session):
            session._pending_dirty.update(n for n in names.split(",") if n)
            return JSONResponse(render.collect(session))

    @app.get("/api/events")
    async def events(sid: str):
        # Created if need be: the page opens its stream *before* booting,
        # so nothing pushed while a workspace loads is lost.
        session = store.get(sid, create=True)
        if session is None:
            return no_session()
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        sub = session.subscribe(asyncio.get_running_loop(), q)

        async def stream():
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        data = await asyncio.wait_for(q.get(), SSE_KEEPALIVE_S)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {data}\n\n"
            finally:
                session.unsubscribe(sub)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/img/{sid}/{key}")
    def image(sid: str, key: str, l: str = "none", t: int = 0):  # noqa: E741
        session = store.get(sid, create=False)
        if session is None:
            return Response(status_code=404)
        path = session.image_path(key)
        if path is None:
            return Response(status_code=404)
        data, mime = crop_bytes(path, l, int(t) % 4)
        return Response(data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/chart/{sid}/{key}.png")
    def chart(sid: str, key: str):
        session = store.get(sid, create=False)
        data = session.chart(key) if session is not None else None
        if data is None:
            return Response(status_code=404)
        return Response(data, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/dl/{sid}/{key}")
    def download(sid: str, key: str, inline: int = 0):
        session = store.get(sid, create=False)
        item = session.download(key) if session is not None else None
        if item is None:
            return Response(status_code=404)
        data, filename, mime = item
        disposition = "inline" if inline else "attachment"
        return Response(data, media_type=mime,
                        headers={"Content-Disposition": f'{disposition}; filename="{filename}"',
                                 "Cache-Control": "private, no-store"})

    return app


def run_web(host: str = "127.0.0.1", port: int = 8560, dataset: Optional[str] = None,
            experiments=None, library_dir: Optional[str] = None) -> None:
    import uvicorn

    from lookin.core import library

    if library_dir:
        library.set_library_root(library_dir)
    webapp = WebApp(dataset=dataset, experiments=experiments)
    app = create_app(webapp)
    print(f"[lookin] web interface on http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", ws="none")
