"""Handing a rendered PNG to the browser as a real file.

Every picture this app shows -- matplotlib chart or dataset crop alike --
is a base64 PNG that CanvasKit paints into a `<canvas>`. There is no
`<img>` element anywhere on the page, so the browser's own "Copy image" and
"Save image as" never appear on any of them, and Flet's clipboard API
(`Page.set_clipboard`) carries text only, with no image equivalent.

Neither would help anyway: the clipboard that matters belongs to the
machine the *user* is sitting at, which is not the machine lookin runs on
-- the app is normally reached over a forwarded port.

What does work is giving the browser a URL. The PNG is written into the
directory Flet serves as its assets root (see app.run_ui) and opened in a
new tab, where it is a plain `<img>` again: right-click -> Copy image
pastes it into PowerPoint or a document, right-click -> Save image as puts
it on the user's own disk rather than the server's. That is the difference
from components.save_png_to_disk, which writes server-side and is the right
answer only when those are the same machine.

Nothing is fetched from anywhere: the file is served by the app's own web
server, from the same origin the page came from (see CLAUDE.md).
"""
from __future__ import annotations

import atexit
import base64
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from secrets import token_hex
from threading import Lock

# Exports are all that lives in the assets root today, but the whole root
# is served at the site root, so they get a directory of their own rather
# than sitting loose beside whatever else may need serving later.
EXPORTS_SUBDIR = "exports"

# The temp directory is named after the process that owns it, so a later
# run can tell an abandoned one from one still in use (see _sweep_orphans).
DIR_PREFIX = "lookin-assets-"

# How many exported PNGs stay on disk. Each is a full-resolution encoding
# (ZOOM_DPI -- a dense scatter runs to about a megabyte) and an afternoon
# of exports would otherwise accumulate without bound. A tab already open
# keeps working until its file is pruned, and 30 is well past the point
# where anyone is still going back to paste one.
MAX_KEPT = 30

_root: Path | None = None
# Published exports, oldest first. The order is kept here rather than read
# back from the directory's mtimes: a burst of exports lands inside one
# filesystem timestamp tick, and sorting on a tie is arbitrary -- which had
# _prune deleting the file the user had just asked for while keeping older
# ones. The list is authoritative because the directory is this process's
# own temp dir and nothing else ever writes to it.
_published: list[Path] = []
_lock = Lock()


def assets_root() -> Path:
    """The directory Flet serves as its assets root, created on first ask.

    Called before ft.app() rather than at the first export: flet_web checks
    the directory exists exactly once, when it configures its static file
    handler, and silently drops an assets_dir that was missing at that
    moment -- after which every export URL would 404 and, because unknown
    paths fall back to index.html, open a second copy of the app instead of
    the image.
    """
    global _root
    if _root is None:
        _sweep_orphans()
        root = Path(tempfile.mkdtemp(prefix=f"{DIR_PREFIX}{os.getpid()}-"))
        (root / EXPORTS_SUBDIR).mkdir()
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        _root = root
    return _root


def _sweep_orphans() -> None:
    """Delete the export directories of lookin processes that are gone.

    The atexit hook above covers a clean exit and a Ctrl-C, but not a
    SIGTERM -- and a server stopped that way (by a service manager, or a
    kill) would otherwise leave its exports in the temp directory forever,
    tens of megabytes at a time across restarts.

    A directory whose pid is still running is left alone: several instances
    on one machine is a normal thing to do, and taking the assets out from
    under a running one would break every image it has open. A pid that
    belongs to some other user's process reads as alive for the same
    reason it is not ours to delete.
    """
    for path in Path(tempfile.gettempdir()).glob(f"{DIR_PREFIX}*"):
        pid = path.name[len(DIR_PREFIX):].split("-")[0]
        if not pid.isdigit() or _pid_alive(int(pid)):
            continue
        shutil.rmtree(path, ignore_errors=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0 tests for the process without touching it
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: someone else's, so not ours to delete
        return True
    return True


def slug(title: str, fallback: str = "image") -> str:
    """A panel title reduced to something safe in a filename and a URL."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:60] or fallback


def publish_png(b64: str, title: str) -> tuple[bool, str]:
    """Write a base64 PNG where the app's own web server can serve it, and
    return the URL to open it at -- or (False, message) if it could not be
    written.

    The URL is relative on purpose. It resolves against the page's `<base
    href>`, which is what keeps it correct when the app is reached over a
    forwarded port or mounted under a proxy path: the server has no way to
    know the host and port the browser actually used to get here.

    Every export gets a fresh name, so reopening a panel that has been
    redrawn cannot land on a cached copy of the old picture.
    """
    name = f"{slug(title)}-{datetime.now():%H%M%S}-{token_hex(3)}.png"
    dest = assets_root() / EXPORTS_SUBDIR / name
    try:
        dest.write_bytes(base64.b64decode(b64))
    except (OSError, ValueError) as ex:  # binascii.Error is a ValueError
        return False, f"Could not prepare the image: {ex}"
    _prune(dest)
    return True, f"{EXPORTS_SUBDIR}/{name}"


def _prune(published: Path) -> None:
    """Record one export and drop all but the MAX_KEPT most recent.

    Best effort on the unlink: a file that vanished under us must never
    turn into a failed export -- the image the user asked for has already
    been written by the time this runs.
    """
    with _lock:
        _published.append(published)
        stale, _published[:] = _published[:-MAX_KEPT], _published[-MAX_KEPT:]
    for path in stale:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
