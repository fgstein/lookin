"""Minimal REST client for an MLflow tracking server.

Talks directly to MLflow's stable HTTP API (no `mlflow` package dependency,
since that pulls in a heavy set of transitive libraries the app otherwise
doesn't need) -- just experiments, runs, artifact listing, and artifact
download, which is all the "load an experiment from MLflow" feature needs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Connect and read are budgeted separately, because they answer different
# questions. Failing to *connect* means nothing is listening: 5s is already
# generous on a LAN, and a short budget is what keeps a wrong URI from
# looking like a hung app for 15s. Once connected the server is alive and
# merely slow -- an experiments/search over a large tracking database
# routinely takes longer than the 15s this used to allow, and cutting it off
# there reported a working server as unreachable.
# Where a fresh install looks for a tracking server, and the variable that
# overrides it.
FALLBACK_TRACKING_URI = "http://127.0.0.1:5000"
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"

# Run statuses that mean the run did not finish: drawn apart in the runs
# table, since their results are partial at best.
UNSUCCESSFUL_STATUSES = ("FAILED", "KILLED")


def env_tracking_uri() -> str:
    """The tracking URI from the environment, or "" if there is nothing this
    client can use.

    MLFLOW_TRACKING_URI is the variable MLflow's own client and CLI already
    read, so a machine set up for MLflow needs no configuring here. But it
    legitimately holds values this client cannot talk to -- ``file:///...``
    for a local mlruns directory, ``databricks``, a bare path -- and the
    client here speaks the HTTP API only. Those are reported to the user
    rather than silently dialled and failed on.
    """
    value = os.environ.get(TRACKING_URI_ENV, "").strip()
    return value if value.startswith(("http://", "https://")) else ""


def default_tracking_uri() -> str:
    return env_tracking_uri() or FALLBACK_TRACKING_URI


CONNECT_TIMEOUT = 5.0
TIMEOUT = httpx.Timeout(30.0, connect=CONNECT_TIMEOUT)
# Downloads get their own, much longer read budget: a proxied-artifact
# server can take a while to start streaming a multi-hundred-MB embeddings
# sidecar, and killing that at 30s reports the server as unreachable when it
# is merely busy.
DOWNLOAD_TIMEOUT = httpx.Timeout(300.0, connect=CONNECT_TIMEOUT)
MAX_ARTIFACT_NODES = 500  # safety cap on the recursive artifact walk
# Per suffix, not shared across them: a run that logs hundreds of .npz files
# must not push its results CSVs out of the listing (and vice versa), which
# is what a single shared budget did -- silently, since the walk just stops.
MAX_FILES_PER_SUFFIX = 200


class MlflowError(Exception):
    pass


@dataclass
class Experiment:
    """As returned by ``experiments/search``, which already carries the
    description tag and both timestamps -- so ordering the list by activity,
    or previewing what an experiment is about, needs no extra request."""
    experiment_id: str
    name: str
    description: str = ""
    creation_time: int | None = None
    last_update_time: int | None = None


@dataclass
class Run:
    """Everything ``runs/search`` already hands back for one run.

    The description, params and metrics come from the *same* response as
    the id/name/status -- MLflow returns them inline -- so carrying them
    here costs no extra request, and lets the browse dialog show what a run
    actually was before the user commits to downloading its CSV.
    """
    run_id: str
    run_name: str
    status: str
    start_time: int | None
    end_time: int | None = None
    user: str = ""
    description: str = ""
    git_commit: str = ""
    source_name: str = ""
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    experiment_id: str = ""

    def label(self) -> str:
        return self.run_name or self.run_id[:8]

    def duration_s(self) -> float | None:
        """None while the run is still going (no end_time yet)."""
        if not self.start_time or not self.end_time:
            return None
        return max(0.0, (self.end_time - self.start_time) / 1000.0)


@dataclass
class ArtifactFile:
    path: str  # full relative path from the run's artifact root
    size: int | None = None


def _client(tracking_uri: str, timeout=TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=tracking_uri.rstrip("/"), timeout=timeout)


def _request(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    try:
        resp = client.request(method, path, **kwargs)
    except httpx.HTTPError as ex:
        raise MlflowError(f"Could not reach MLflow server: {ex}") from ex
    if resp.status_code != 200:
        raise MlflowError(f"MLflow server returned {resp.status_code} for {path}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as ex:
        raise MlflowError(f"Unexpected response from MLflow server at {path}") from ex
    # Every MLflow endpoint answers an object. Anything else (a proxy's JSON
    # list or string) would surface as an AttributeError further down, past
    # callers that only catch MlflowError.
    if not isinstance(data, dict):
        raise MlflowError(f"Unexpected response from MLflow server at {path}")
    return data


def test_connection(tracking_uri: str) -> None:
    """Raises MlflowError if the server can't be reached / doesn't look like MLflow."""
    with _client(tracking_uri) as client:
        _request(client, "POST", "/api/2.0/mlflow/experiments/search", json={"max_results": 1})


def list_experiments(tracking_uri: str) -> list[Experiment]:
    experiments: list[Experiment] = []
    with _client(tracking_uri) as client:
        page_token = None
        for _ in range(20):  # hard cap on pagination loops
            body = {"max_results": 1000}
            if page_token:
                body["page_token"] = page_token
            data = _request(client, "POST", "/api/2.0/mlflow/experiments/search", json=body)
            for e in data.get("experiments", []):
                tags = {t["key"]: t["value"] for t in e.get("tags", [])}
                experiments.append(Experiment(
                    experiment_id=e["experiment_id"],
                    name=e.get("name", e["experiment_id"]),
                    description=tags.get("mlflow.note.content", ""),
                    creation_time=e.get("creation_time"),
                    last_update_time=e.get("last_update_time"),
                ))
            page_token = data.get("next_page_token")
            if not page_token:
                break
    experiments.sort(key=lambda e: e.name.lower())
    return experiments


def _parse_run(r: dict) -> Run:
    info = r.get("info", {})
    block = r.get("data", {})
    tags = {t["key"]: t["value"] for t in block.get("tags", [])}
    run_name = info.get("run_name") or tags.get("mlflow.runName") or ""
    params = {p["key"]: p.get("value", "") for p in block.get("params", [])}
    # runs/search returns the *latest* value logged for each metric key,
    # which is what a "was this the good run?" glance wants.
    metrics = {}
    for m in block.get("metrics", []):
        try:
            metrics[m["key"]] = float(m.get("value"))
        except (TypeError, ValueError):
            continue
    return Run(
        run_id=info["run_id"],
        run_name=run_name,
        status=info.get("status", "?"),
        start_time=info.get("start_time"),
        end_time=info.get("end_time"),
        user=info.get("user_id") or tags.get("mlflow.user", ""),
        description=tags.get("mlflow.note.content", ""),
        git_commit=tags.get("mlflow.source.git.commit", ""),
        source_name=tags.get("mlflow.source.name", ""),
        params=params,
        metrics=metrics,
        experiment_id=str(info.get("experiment_id", "")),
    )


def get_run(tracking_uri: str, run_id: str) -> Run:
    """One run by id, wherever its experiment is."""
    with _client(tracking_uri) as client:
        data = _request(client, "GET", "/api/2.0/mlflow/runs/get", params={"run_id": run_id})
    if not isinstance(data.get("run"), dict):
        raise MlflowError(f"Unexpected response from MLflow server for run {run_id}")
    return _parse_run(data["run"])


# runs/search refuses more than 1000 per page. The page cap bounds the whole
# listing: a Home page showing 10k runs is one nobody reads -- they filter.
SEARCH_PAGE_SIZE = 1000
MAX_SEARCH_PAGES = 10


def search_runs(tracking_uri: str, experiment_ids: list, filter_string: str = "",
                max_pages: int = MAX_SEARCH_PAGES) -> tuple:
    """``(runs, truncated)`` for several experiments at once, most recent first.

    `filter_string` is MLflow's own search syntax (``metrics.acc > 0.9 and
    params.backbone = 'vit'``), handed to the server untouched: it already
    parses and indexes it, and a second parser here would disagree with it
    on some corner. A malformed filter comes back as an MlflowError carrying
    the server's message, which is the useful one.
    """
    if not experiment_ids:
        return [], False
    runs: list = []
    truncated = False
    with _client(tracking_uri) as client:
        page_token = None
        for page in range(max_pages):
            body = {
                "experiment_ids": [str(e) for e in experiment_ids],
                "max_results": SEARCH_PAGE_SIZE,
                "order_by": ["attribute.start_time DESC"],
            }
            if filter_string.strip():
                body["filter"] = filter_string.strip()
            if page_token:
                body["page_token"] = page_token
            data = _request(client, "POST", "/api/2.0/mlflow/runs/search", json=body)
            runs.extend(_parse_run(r) for r in data.get("runs", []))
            page_token = data.get("next_page_token")
            if not page_token:
                break
            if page == max_pages - 1:
                truncated = True
    return runs, truncated


def list_runs(tracking_uri: str, experiment_id: str) -> list[Run]:
    """Every run of one experiment, most recent first -- paged like
    search_runs. Reading only the first page dropped the older runs of a
    busy experiment without a word, and searching for one by name then
    answered "No matches"."""
    return search_runs(tracking_uri, [experiment_id])[0]


# ---------------------------------------------------------------------------
# links into the MLflow web UI
#
# The UI is served by the same process as the tracking API, at the same
# address -- which is what "the UI address is the tracking URI" means here.
# These are hash routes, stable since MLflow 1.x.
# ---------------------------------------------------------------------------

def experiment_ui_url(tracking_uri: str, experiment_id: str) -> str:
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}"


def run_ui_url(tracking_uri: str, experiment_id: str, run_id: str, artifact_path: str = "") -> str:
    url = f"{experiment_ui_url(tracking_uri, experiment_id)}/runs/{run_id}"
    if artifact_path:
        url += f"/artifacts/{artifact_path.lstrip('/')}"
    return url


def is_safe_relative_path(path) -> bool:
    """True for a relative path with no empty, ``.`` or ``..`` part, under
    either separator -- the only kind MLflow hands out, and the only kind
    that stays inside the download cache once joined to it (see
    mlflow_cache.cached_path). Paths also come from workspace manifests
    someone else wrote, so this is not only about the server."""
    if not isinstance(path, str) or not path or "\0" in path:
        return False
    if path.startswith(("/", "\\")) or (len(path) > 1 and path[1] == ":"):
        return False
    return all(part not in ("", ".", "..") for part in path.replace("\\", "/").split("/"))


def _walk_artifacts(tracking_uri: str, run_id: str, suffixes: tuple, max_files: int) -> dict:
    """Breadth-first walk of the run's artifact tree, collecting files by
    suffix. Capped so a run with an unexpectedly huge/deep artifact tree
    can't hang the UI.

    Takes several suffixes at once because the browse dialog wants the CSVs
    *and* the .npz sidecars sitting next to them: asking for both in one
    walk is one traversal of the tree instead of two. `max_files` applies to
    each suffix on its own -- see MAX_FILES_PER_SUFFIX -- and the walk only
    stops early once *every* suffix is full, so a lopsided run (many .npz,
    one .csv) still yields the CSV.
    """
    found = {suffix: [] for suffix in suffixes}
    with _client(tracking_uri) as client:
        queue = [""]
        visited = 0
        while queue and visited < MAX_ARTIFACT_NODES:
            if all(len(files) >= max_files for files in found.values()):
                break
            path = queue.pop(0)
            visited += 1
            params = {"run_id": run_id}
            if path:
                params["path"] = path
            data = _request(client, "GET", "/api/2.0/mlflow/artifacts/list", params=params)
            for f in data.get("files", []):
                # The path decides where a download lands in the cache: one
                # that would land outside it is not offered at all.
                if not is_safe_relative_path(f.get("path")):
                    continue
                if f.get("is_dir"):
                    queue.append(f["path"])
                    continue
                lowered = f["path"].lower()
                for suffix in suffixes:
                    if lowered.endswith(suffix):
                        if len(found[suffix]) < max_files:
                            found[suffix].append(ArtifactFile(path=f["path"], size=f.get("file_size")))
                        break
    for files in found.values():
        files.sort(key=lambda a: a.path.lower())
    return found


def list_csv_artifacts(tracking_uri: str, run_id: str) -> list[ArtifactFile]:
    return _walk_artifacts(tracking_uri, run_id, (".csv",), MAX_FILES_PER_SUFFIX)[".csv"]


def list_csv_and_npz_artifacts(tracking_uri: str, run_id: str) -> tuple:
    """``(csv_files, npz_files)`` from a single walk -- what the browse
    dialog needs to list the loadable CSVs and, at the same time, say which
    of them already have embedding/prototype sidecars logged beside them."""
    found = _walk_artifacts(tracking_uri, run_id, (".csv", ".npz"), MAX_FILES_PER_SUFFIX)
    return found[".csv"], found[".npz"]


PROGRESS_EVERY_BYTES = 512 * 1024  # how often download_artifact calls back


def download_artifact(tracking_uri: str, run_id: str, artifact_path: str, dest_path: Path,
                      progress=None, total_hint: int | None = None) -> Path:
    """Stream one artifact to `dest_path`.

    `progress`, when given, is called as ``progress(downloaded, total)`` --
    `total` being the Content-Length, or `total_hint` (the size the run's
    artifact listing reported) when the server sends no length, or None when
    neither is known. It is called at most every PROGRESS_EVERY_BYTES, so a
    caller that repaints the UI on each callback is not woken per chunk. It
    exists because "is this stuck?" is the only question a user has while an
    embeddings sidecar of a few hundred MB comes down.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with _client(tracking_uri, timeout=DOWNLOAD_TIMEOUT) as client:
        try:
            with client.stream("GET", "/get-artifact", params={"run_id": run_id, "path": artifact_path}) as resp:
                if resp.status_code != 200:
                    raise MlflowError(f"Could not download artifact (HTTP {resp.status_code})")
                total = total_hint
                length = resp.headers.get("content-length")
                if length:
                    try:
                        total = int(length)
                    except ValueError:
                        pass
                downloaded = 0
                reported = 0
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None and downloaded - reported >= PROGRESS_EVERY_BYTES:
                            reported = downloaded
                            progress(downloaded, total)
                if progress is not None:
                    progress(downloaded, total or downloaded)
        except httpx.HTTPError as ex:
            raise MlflowError(f"Could not download artifact: {ex}") from ex
    return dest_path
