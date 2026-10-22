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
import time
import traceback
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lookin import __version__
from lookin.core.images import crop_bytes
from lookin.web.session import RequestScope, SessionStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
SSE_KEEPALIVE_S = 15.0
# The browser window the page reports, clamped: it sizes pages of crops and
# the Embeddings map, and a request claiming a window a million pixels wide
# must not turn into a million-pixel figure.
VIEWPORT_MIN, VIEWPORT_MAX = 320, 16384


class WebApp:
    """Process-wide settings from the command line, and the session store."""

    def __init__(self, dataset: Optional[str] = None, experiments=None):
        self.dataset = dataset
        self.experiments = list(experiments or [])
        self.store = SessionStore(self)


def content_disposition(disposition: str, filename: str) -> str:
    """The header for a download named after something the user named (a
    workspace, a class): headers are latin-1, so a name in another script
    -- or with a quote or a line break in it -- would fail the response.
    An ASCII stand-in for old clients, the real name in RFC 5987 form."""
    name = str(filename)
    fallback = "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in name) or "download"
    return f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{quote(name, safe='')}"


# ---- the page ----------------------------------------------------------------

def _extra_assets() -> str:
    """One stylesheet per view under static/css/ and one script per view
    under static/js/ (the canvas of the Embeddings map), all local."""
    out = []
    for path in sorted((STATIC_DIR / "css").glob("*.css")):
        out.append(f'<link rel="stylesheet" href="/static/css/{path.name}?v={__version__}">')
    for path in sorted((STATIC_DIR / "js").glob("*.js")):
        out.append(f'<script src="/static/js/{path.name}?v={__version__}" defer></script>')
    return "\n".join(out)


# Inline and first in <head>, so the theme is set before anything is drawn:
# the saved choice (app.js, THEME_KEY), else the system's. Kept in this
# browser rather than in the library or the session -- every tab of it
# follows, and a colleague on the same library keeps theirs.
THEME_KEY = "lookin-theme"
THEME_SCRIPT = ("(function(){var t=null;try{t=localStorage.getItem(\"" + THEME_KEY + "\")}catch(e){}"
                "if(t!==\"light\"&&t!==\"dark\")t=window.matchMedia&&matchMedia(\"(prefers-color-scheme: dark)\").matches"
                "?\"dark\":\"light\";document.documentElement.setAttribute(\"data-theme\",t)})();")


def page_html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lookin</title>
<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
<script>{THEME_SCRIPT}</script>
<link rel="stylesheet" href="/static/app.css?v={__version__}">
<script src="/static/app.js?v={__version__}" defer></script>
{_extra_assets()}
</head>
<body>
<div id="app"><div class="boot">Loading…</div></div>
<div id="dialogs"></div>
<div id="toasts" aria-live="polite"></div>
<div id="zoom"></div>
</body>
</html>"""


def _mcp_parts(enable_mcp: bool):
    """(routes, lifespan) for the MCP endpoint at MCP_PATH (see
    lookin/mcp_server.py), served on the same port as the page."""
    from contextlib import asynccontextmanager

    if not enable_mcp:
        return [], None
    from mcp.server.transport_security import TransportSecuritySettings

    from lookin.mcp_server import MCP_PATH, build_server

    server = build_server()
    mcp_app = server.streamable_http_app(
        streamable_http_path=MCP_PATH, stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @asynccontextmanager
    async def lifespan(_app):
        async with server.session_manager.run():
            yield

    return list(mcp_app.routes), lifespan


def create_app(webapp: WebApp, enable_mcp: bool = False) -> FastAPI:
    from lookin.web import render

    mcp_routes, lifespan = _mcp_parts(enable_mcp)
    app = FastAPI(title="Lookin", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.router.routes.extend(mcp_routes)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    store = webapp.store

    def no_session():
        return JSONResponse({"error": "no session", "reload": True}, status_code=410)

    @app.middleware("http")
    async def refuse_cross_site_posts(request: Request, call_next):
        # Lookin listens with no login, so any web page the user visits
        # could otherwise POST to /api/act and drive it -- open a session,
        # browse the disk, delete workspaces. The answers are never readable
        # cross-site, but the side effects would happen. Two checks that
        # need no token and do not depend on the address the page was opened
        # under (a port forward or a gateway under another name are normal
        # here, see mcp_server.py):
        #  * the page's own requests are JSON, and a page on another site
        #    cannot send a JSON content type without a CORS preflight, which
        #    this server never grants -- a <form> or a no-cors fetch can only
        #    send text/plain, urlencoded or multipart;
        #  * browsers that send Fetch Metadata say where a request comes from.
        # The MCP endpoint keeps its own transport rules (it is read-only).
        if request.method == "POST" and request.url.path.startswith("/api/"):
            site = request.headers.get("sec-fetch-site", "")
            ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if (site and site not in ("same-origin", "none")) or ctype != "application/json":
                return JSONResponse({"error": "cross-site request refused"}, status_code=403)
        return await call_next(request)

    async def json_body(request: Request) -> Optional[dict]:
        """The request's JSON object, or None for anything else (malformed
        JSON, a list, a string) -- answered with a 400, not a traceback."""
        try:
            body = await request.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    def bad_request():
        return JSONResponse({"error": "expected a JSON object"}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(page_html(), headers={"Cache-Control": "no-store"})

    @app.post("/api/boot")
    async def boot(request: Request):
        body = await json_body(request)
        if body is None:
            return bad_request()
        session = store.get(str(body.get("sid", "")))
        if session is None:
            return no_session()
        query = parse_qs(str(body.get("search", "")).lstrip("?"))
        return await asyncio.to_thread(_boot, session, query, body)

    def _viewport(session, body):
        # The browser window, for the views that size a page of crops to
        # what fits on screen (the carousel); sent with every request.
        def clamp(value) -> int:
            return min(VIEWPORT_MAX, max(VIEWPORT_MIN, int(value or 0)))

        try:
            session.viewport = (clamp(body.get("vw")), clamp(body.get("vh")))
        except (TypeError, ValueError, OverflowError):   # Infinity is valid JSON to Python
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
        body = await json_body(request)
        if body is None or not isinstance(body.get("args") or {}, dict):
            return bad_request()
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

    @app.post("/api/ping")
    async def ping(request: Request):
        """A hidden tab's "still here": it holds no event stream (see
        app.js), and a session nobody is connected to is otherwise reaped
        once idle. Takes no lock -- it must answer while an action runs."""
        body = await json_body(request)
        session = store.get(str((body or {}).get("sid", "")), create=False)
        if session is None:
            return no_session()
        session.last_seen = time.time()
        return JSONResponse({"ok": True})

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

    @app.get("/pimg/{sid}/{mkey}/{i}")
    def point_image(sid: str, mkey: str, i: int):
        """The crop behind point `i` of an interactive map (see
        Session.put_point_map): the map knows thousands of points, and only
        the ones the pointer lands on are ever read."""
        session = store.get(sid, create=False)
        resolve = session.point_map(mkey) if session is not None else None
        found = resolve(int(i)) if resolve is not None else None
        if not found:
            return Response(status_code=404)
        path, lut, turns = found
        data, mime = crop_bytes(str(path), lut, int(turns) % 4)
        return Response(data, media_type=mime, headers={"Cache-Control": "private, max-age=600"})

    @app.get("/pinfo/{sid}/{mkey}/{i}")
    def point_info(sid: str, mkey: str, i: int):
        """The details panel for point `i` of an interactive map: the map's
        tiles open the full-size view directly, and know only their point."""
        from lookin.web import html as h
        from lookin.web.views.details import details_html

        session = store.get(sid, create=False)
        describe = session.point_describer(mkey) if session is not None else None
        found = describe(int(i)) if describe is not None else None
        if not found:
            return Response(status_code=404)
        uid, crop = found
        with session.lock:
            try:
                body = details_html(session, uid=uid, crop=None if uid is not None else crop)
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                body = h.banner(f"Could not read this crop: {type(ex).__name__}: {ex}", "error")
        return HTMLResponse(body, headers={"Cache-Control": "no-store"})

    @app.get("/info/{sid}")
    def crop_info(sid: str, u: Optional[str] = None, c: Optional[str] = None):
        """The details panel of the full-size view (views/details.py),
        fetched only when the user opens it."""
        from lookin.web import html as h
        from lookin.web.views.details import details_html

        session = store.get(sid, create=False)
        if session is None:
            return Response(status_code=404)
        with session.lock:
            try:
                body = details_html(session, uid=u, crop=c)
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                body = h.banner(f"Could not read this crop: {type(ex).__name__}: {ex}", "error")
        return HTMLResponse(body, headers={"Cache-Control": "no-store"})

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
                        headers={"Content-Disposition": content_disposition(disposition, filename),
                                 "Cache-Control": "private, no-store"})

    return app


def run_web(host: str = "127.0.0.1", port: int = 8560, dataset: Optional[str] = None,
            experiments=None, library_dir: Optional[str] = None, enable_mcp: bool = True) -> None:
    import uvicorn

    from lookin.core import library

    if library_dir:
        library.set_library_root(library_dir)
    webapp = WebApp(dataset=dataset, experiments=experiments)
    app = create_app(webapp, enable_mcp=enable_mcp)
    from lookin.web.render import warm_up
    warm_up()
    url = f"http://{'127.0.0.1' if host in ('', '0.0.0.0', '*') else host}:{port}"
    print(f"[lookin] web interface on {url}", flush=True)
    if enable_mcp:
        from lookin.mcp_server import MCP_PATH
        print(f"[lookin] MCP endpoint for AI agents: {url}{MCP_PATH}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", ws="none")
