"""What TrainHub knows of the MLflow runs Lookin's experiments come from.

TrainHub launches and monitors the trainings and inferences, Lookin explores
their results: the two meet at the MLflow run. TrainHub serves
``GET /api/links/mlflow-runs`` -- for each MLflow run id it shows somewhere
(a run it launched, a run it found in a project's experiments, a run
directory of a checkpoint source, an inference made with a checkpoint), what
it is and the address of the page that shows it. Lookin draws those as
"TrainHub" badges next to the experiments made from the same runs: Home's
workspace rows, the Sources tab, and the MCP answers.

Nothing is tagged for the other tool: the MLflow run id is the whole link,
so a run started by hand links as well as one TrainHub launched, as long as
TrainHub lists it somewhere.

Opt-in, like an MLflow server: only once a TrainHub address is set --
``lookin web --trainhub-url``, the LOOKIN_TRAINHUB_URL variable, or
"trainhub_url" in home_view.json (the TrainHub chip on Home). Its token, when
TrainHub has one (TRAINHUB_TOKEN there), goes in LOOKIN_TRAINHUB_TOKEN. The
index is read on a worker thread and kept REFRESH_S; a TrainHub that does
not answer costs a missing badge and a line in the chip's tooltip, never a
slow page.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

URL_ENV = "LOOKIN_TRAINHUB_URL"
TOKEN_ENV = "LOOKIN_TRAINHUB_TOKEN"
INDEX_PATH = "/api/links/mlflow-runs"
REFRESH_S = 30.0
# The index answers from TrainHub's caches, but its first call after a
# restart scans the checkpoint sources (over SSH), which takes a while.
TIMEOUT = httpx.Timeout(90.0, connect=5.0)

# What each kind of entry is, as the badge's tooltip says it. The kinds are
# TrainHub's (a format shared by the two tools): never reworded.
KIND_RUN = "run"
KIND_EXTERNAL = "external"
KIND_CHECKPOINT_RUN = "checkpoint_run"
KIND_INFERENCE = "inference"
KIND_LABELS = {
    KIND_RUN: "Run launched by TrainHub",
    KIND_EXTERNAL: "MLflow run shown in a TrainHub project (not launched by it)",
    KIND_CHECKPOINT_RUN: "Run directory of a TrainHub checkpoint source (the training)",
    KIND_INFERENCE: "Inference made with a checkpoint, in TrainHub's Checkpoints tab",
}

# Set by `lookin web --trainhub-url`; wins over the variable and the file.
_override = ""


def set_url(url: Optional[str]) -> None:
    global _override
    _override = (url or "").strip().rstrip("/")


def configured_url() -> str:
    """The TrainHub address, "" when none is set."""
    url = _override or os.environ.get(URL_ENV, "").strip().rstrip("/")
    if not url:
        from lookin.core.home_config import HomeConfig

        url = HomeConfig().trainhub_url
    return url if url.lower().startswith(("http://", "https://")) else ""


def url_source() -> str:
    """Where the address comes from, for the chip's tooltip."""
    if _override:
        return "--trainhub-url"
    if os.environ.get(URL_ENV, "").strip():
        return URL_ENV
    return "home_view.json"


@dataclass
class TrainhubLink:
    kind: str
    label: str
    url: str
    detail: str = ""

    def title(self) -> str:
        bits = [KIND_LABELS.get(self.kind, "TrainHub"), self.detail]
        return " -- ".join(b for b in bits if b)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "label": self.label, "url": self.url, "detail": self.detail or None}


def _safe_path(path) -> str:
    """Only an address on the TrainHub server itself ("/p/...")."""
    path = str(path or "")
    return path if path.startswith("/") and not path.startswith("//") else ""


def parse_index(payload, base_url: str) -> dict:
    """{run id: [TrainhubLink]} from TrainHub's answer; whatever does not
    have the expected shape is dropped, not raised."""
    out: dict = {}
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        return out
    for run_id, entries in runs.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = _safe_path(entry.get("path"))
            label = str(entry.get("label") or "").strip()
            if not path or not label:
                continue
            out.setdefault(str(run_id), []).append(TrainhubLink(
                kind=str(entry.get("kind") or ""), label=label, url=base_url + path,
                detail=str(entry.get("detail") or "")))
    return out


def fetch_index(base_url: str, token: str = "", transport=None) -> dict:
    """Ask TrainHub, once. Raises httpx.HTTPError or ValueError."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(base_url=base_url, timeout=TIMEOUT, headers=headers, transport=transport) as client:
        resp = client.get(INDEX_PATH)
    if resp.status_code == 401:
        raise ValueError(f"TrainHub refused the request (401): set its token in {TOKEN_ENV}.")
    if resp.status_code == 404:
        raise ValueError("This TrainHub has no run index (404): it is older than 0.19.")
    if resp.status_code != 200:
        raise ValueError(f"TrainHub answered {resp.status_code}.")
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("app") != "trainhub":
        raise ValueError("The address does not answer like a TrainHub server.")
    return parse_index(payload, base_url)


class TrainhubIndex:
    """One per process, shared by every tab: ten tabs on Home ask TrainHub
    once per REFRESH_S, not ten times."""

    _instance: Optional["TrainhubIndex"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "TrainhubIndex":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, fetch: Optional[Callable[[str, str], dict]] = None):
        self._fetch = fetch or fetch_index
        self._lock = threading.Lock()
        self._listeners: dict = {}
        self.url = ""
        self.runs: dict = {}
        self.error = ""
        self.fetched_at = 0.0
        self._fetching: Optional[threading.Thread] = None
        self._tried_at = 0.0

    # ---- listeners: the views that draw badges redraw when it moves ----

    def subscribe(self, key, fn: Callable[[], None]) -> None:
        with self._lock:
            self._listeners[key] = fn

    def unsubscribe(self, key) -> None:
        with self._lock:
            self._listeners.pop(key, None)

    def _changed(self) -> None:
        with self._lock:
            fns = list(self._listeners.values())
        for fn in fns:
            try:
                fn()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    # ---- reading ----

    def ensure(self) -> None:
        """Start a read when the last one is older than REFRESH_S. Never waits."""
        with self._lock:
            if self._fetching is not None or time.monotonic() - self._tried_at < REFRESH_S:
                return
            self._tried_at = time.monotonic()
            self._fetching = threading.Thread(target=self._read, daemon=True, name="lookin-trainhub")
            self._fetching.start()

    def refresh(self, wait: float = 0.0) -> None:
        """Read again now (the address changed, an MCP call wants it fresh);
        wait up to `wait` seconds for the answer."""
        with self._lock:
            self._tried_at = 0.0
        self.ensure()
        thread = self._fetching
        if wait and thread is not None:
            thread.join(wait)

    def _read(self) -> None:
        try:
            url = configured_url()
            if not url:
                runs, error = {}, ""
            else:
                try:
                    runs, error = self._fetch(url, os.environ.get(TOKEN_ENV, "").strip()), ""
                except (httpx.HTTPError, ValueError) as ex:
                    with self._lock:
                        runs = self.runs if self.url == url else {}
                    error = f"Could not read TrainHub at {url}: {ex}"
            with self._lock:
                moved = (url, runs, error) != (self.url, self.runs, self.error)
                self.url, self.runs, self.error = url, runs, error
                if not error:
                    self.fetched_at = time.time()
        except Exception:  # noqa: BLE001 -- a bug here must not stop the next read
            traceback.print_exc()
            moved = False
        finally:
            with self._lock:
                self._fetching = None
        if moved:
            self._changed()

    def links(self, run_id: str) -> list:
        """What TrainHub shows of `run_id`, from the last read (and a read
        started when that is old)."""
        self.ensure()
        with self._lock:
            return list(self.runs.get(run_id or "", ()))

    def status(self) -> tuple:
        """(url, error, number of runs known): what the chip on Home says."""
        self.ensure()
        with self._lock:
            return self.url, self.error, len(self.runs)
