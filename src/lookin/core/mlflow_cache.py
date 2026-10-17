"""Where MLflow artifacts land on this machine, and how they get there.

One cache, shared by everything that downloads from a tracking server --
the MLflow browse dialog and the workspaces on the Home page -- so a CSV
downloaded by one is a cache hit for the other. It moved here from
ui/mlflow_panel.py when the second user appeared.

The layout is ``<cache>/<run_id>/<artifact path>``. A run id is a UUID
minted by the tracking server, so two servers do not collide in practice,
and keeping the layout unchanged keeps every file downloaded before the
move usable.

Also the *offline* half of the Home page: the last runs listing fetched
from each server is kept here, so a server that is down still shows what it
held (greyed, read-only) instead of an empty table.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from lookin.core import mlflow_client as mlf


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def user_cache_dir() -> Path:
    """XDG-style per-user cache dir -- the installed package directory may
    not be writable (site-packages, a read-only install, another user's
    home), so downloaded artifacts never get cached alongside the code."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "lookin"


def artifacts_dir() -> Path:
    return user_cache_dir() / "mlflow_artifacts"


def cached_path(run_id: str, artifact_path: str) -> Path:
    """Where `artifact_path` of `run_id` is, or will be once downloaded.

    Raises MlflowError for a run id or path that would point outside the
    cache (``..``, an absolute path): both come from a server listing or a
    workspace manifest, and neither may choose where a download is written.
    """
    if not (mlf.is_safe_relative_path(run_id) and "/" not in run_id and "\\" not in run_id
            and mlf.is_safe_relative_path(artifact_path)):
        raise mlf.MlflowError(f"Refusing artifact {artifact_path!r} of run {run_id!r}: "
                              "it would be stored outside the download cache")
    return artifacts_dir() / run_id / Path(artifact_path)


def cached_path_str(run_id: str, artifact_path: str) -> str:
    """cached_path as a plain string, built without pathlib -- for the hot
    loops that only compare paths (see workspace_sources._present). Equal
    to str(cached_path(...)) for any relative POSIX artifact path, which is
    the only kind MLflow hands out."""
    return os.path.join(str(artifacts_dir()), run_id, *artifact_path.split("/"))


def is_cached(run_id: str, artifact_path: str, expected_size=None) -> bool:
    """True when the artifact is on disk -- and, when the listing gave a
    size, is that size. A size mismatch is a stale or truncated copy, which
    is not "downloaded" however much it looks like it."""
    try:
        path = cached_path(run_id, artifact_path)
        if not path.is_file():
            return False
        return expected_size is None or path.stat().st_size == int(expected_size)
    except (OSError, ValueError, mlf.MlflowError):
        return False


def download_cached(uri: str, run_id: str, artifact_path: str, expected_size=None,
                    progress=None) -> Path:
    """Download an artifact into the cache, or reuse what is already there
    when it matches the size the run's artifact listing reported.

    Reusing matters more than it looks: pressing Load again on the same run
    used to re-download the CSV *and* both .npz sidecars every time, and an
    embeddings sidecar is the largest file in the run by a wide margin.

    The download goes to a temporary name and is renamed into place, so an
    interrupted transfer cannot leave a truncated file behind that a later
    load would happily read as a cache hit.
    """
    dest = cached_path(run_id, artifact_path)
    if expected_size is not None and is_cached(run_id, artifact_path, expected_size):
        return dest
    # Unique per thread: two tabs fetching the same artifact at once would
    # otherwise write into one .part file. Each gets its own, and the two
    # renames into place are atomic and carry the same bytes.
    tmp = dest.with_name(f"{dest.name}.part{os.getpid()}.{threading.get_ident()}")
    try:
        mlf.download_artifact(uri, run_id, artifact_path, tmp,
                              progress=progress, total_hint=expected_size)
        # The size is what is_cached and the workspace cards check a copy
        # against later. A file that does not have it -- re-logged on the
        # server since it was listed, or cut short by a server that sent no
        # length -- would sit in the cache never counting as downloaded, and
        # be fetched again on every try; better said now, once.
        got = tmp.stat().st_size
        if expected_size is not None and got != int(expected_size):
            raise mlf.MlflowError(
                f"{artifact_path} changed on the server since it was listed "
                f"({int(expected_size)} bytes then, {got} now): list the run again, "
                "or in a workspace remove the experiment and add it back")
        tmp.replace(dest)
    finally:
        # A failed or refused transfer leaves nothing behind.
        tmp.unlink(missing_ok=True)
    return dest


# ---------------------------------------------------------------------------
# last known listing, per server
# ---------------------------------------------------------------------------

def _listing_path(uri: str) -> Path:
    digest = hashlib.sha256(uri.rstrip("/").encode("utf-8")).hexdigest()[:16]
    return user_cache_dir() / "mlflow_listings" / f"{digest}.json"


def save_listing(uri: str, payload: dict) -> None:
    """Best effort: failing to write a cache must never fail a listing
    that succeeded."""
    path = _listing_path(uri)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as ex:
        _log(f"mlflow cache: could not write {path} ({ex})")


def _artifacts_path(uri: str) -> Path:
    return _listing_path(uri).with_suffix(".artifacts.json")


def save_artifacts(uri: str, payload: dict) -> None:
    """The run -> CSV lists, apart from the runs listing: they grow one run
    at a time as runs are expanded, and the runs listing is tens of MB on a
    busy server -- far too much to rewrite on every expand."""
    path = _artifacts_path(uri)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as ex:
        _log(f"mlflow cache: could not write {path} ({ex})")


def load_artifacts(uri: str) -> dict:
    path = _artifacts_path(uri)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        _log(f"mlflow cache: ignoring unreadable {path} ({ex})")
        return {}
    return raw if isinstance(raw, dict) else {}


def load_listing(uri: str):
    path = _listing_path(uri)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        _log(f"mlflow cache: ignoring unreadable {path} ({ex})")
        return None
    return raw if isinstance(raw, dict) else None
