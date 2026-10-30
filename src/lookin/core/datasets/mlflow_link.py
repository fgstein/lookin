"""Which MLflow run a dataset CSV came from.

The datasets of the shared folders are mostly made by scripts that log
each one to an MLflow "datasets" experiment: one run per dataset, carrying
the CSV and a hash.txt with its content hash (fingerprint.dataset_hash).
The CSV is then copied into the shared folder by hand, so nothing on disk
says which run it came from -- the hash does:

* **hash-matched** -- the file's content hash is the hash.txt of a run;
* **name-matched** -- no hash matches, but a run's CSV artifact has the same
  file name and the same size in bytes (a hash computed differently, or a
  run logged before hash.txt existed).

hash.txt is a few bytes, so every run's is downloaded (once: an artifact of
a finished run never changes) and the index -- hash and CSV artifact per
run, plus what the run says about itself -- is kept in Lookin's cache, so
the page and the MCP tools match files without asking a server anything.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

from lookin.core import fileio
from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core.datasets.fingerprint import parse_hash_file

HASH_FILE = "hash.txt"
INDEX_FILE = "mlflow_index.json"
PARALLEL_RUNS = 4

LINK_HASH = "hash-matched"
LINK_NAME = "name-matched"      # same file name and size: likely, not proven
LINK_PARENT = "parent-matched"  # no run of its own; its main dataset has one
LINK_NONE = ""


@dataclass
class DatasetRun:
    uri: str
    experiment: str
    experiment_id: str
    run_id: str
    run_name: str = ""
    status: str = ""
    start_time: Optional[int] = None
    user: str = ""
    description: str = ""
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    csv_path: str = ""            # artifact path of the CSV, "" if none
    csv_size: Optional[int] = None
    hash: str = ""                # from hash.txt, "" if none
    error: str = ""               # why the artifacts could not be read

    def label(self) -> str:
        return self.run_name or self.run_id[:8]

    def csv_name(self) -> str:
        return PurePosixPath(self.csv_path).name if self.csv_path else ""

    def ui_url(self) -> str:
        return mlf.run_ui_url(self.uri, self.experiment_id, self.run_id)


@dataclass
class SourceState:
    """One configured (server, experiment): its runs, and how the last
    refresh went."""
    uri: str
    experiment: str
    runs: list = field(default_factory=list)        # DatasetRun
    fetched_at: float = 0.0
    error: str = ""


def _index_path() -> Path:
    return mlflow_cache.user_cache_dir() / "datasets" / INDEX_FILE


_lock = threading.Lock()


def load_index() -> dict:
    """{(uri, experiment): SourceState} as last saved."""
    try:
        raw = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    known = DatasetRun.__dataclass_fields__
    for entry in raw.get("sources", []) if isinstance(raw, dict) else []:
        try:
            runs = [DatasetRun(**{k: v for k, v in r.items() if k in known}) for r in entry.get("runs", [])]
            state = SourceState(uri=entry["uri"], experiment=entry["experiment"], runs=runs,
                                fetched_at=float(entry.get("fetched_at", 0)), error=str(entry.get("error", "")))
        except (KeyError, TypeError, ValueError):
            continue
        out[(state.uri, state.experiment)] = state
    return out


def save_index(index: dict) -> None:
    payload = {"sources": [{"uri": s.uri, "experiment": s.experiment, "fetched_at": s.fetched_at,
                            "error": s.error, "runs": [asdict(r) for r in s.runs]} for s in index.values()]}
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        fileio.atomic_write_text(path, json.dumps(payload))


def _read_run(uri: str, experiment: str, run: mlf.Run, previous: Optional[DatasetRun]) -> DatasetRun:
    out = DatasetRun(uri=uri, experiment=experiment, experiment_id=run.experiment_id, run_id=run.run_id,
                     run_name=run.run_name, status=run.status, start_time=run.start_time, user=run.user,
                     description=run.description, params=dict(run.params), metrics=dict(run.metrics),
                     tags=dict(run.tags))
    # A finished run's artifacts do not change: what was read last time stands.
    if previous is not None and not previous.error and run.status != "RUNNING" and (previous.hash or previous.csv_path):
        out.csv_path, out.csv_size, out.hash = previous.csv_path, previous.csv_size, previous.hash
        return out
    try:
        found = mlf.list_artifacts_by_suffix(uri, run.run_id, (".csv", HASH_FILE))
        csvs = found[".csv"]
        if csvs:
            # One CSV per run; if a run has more, the largest is the dataset.
            csv = max(csvs, key=lambda a: a.size or 0)
            out.csv_path, out.csv_size = csv.path, csv.size
        hashes = found[HASH_FILE]
        if hashes:
            local = mlflow_cache.download_cached(uri, run.run_id, hashes[0].path, expected_size=hashes[0].size)
            out.hash = parse_hash_file(Path(local).read_text(encoding="utf-8", errors="replace"))
    except (mlf.MlflowError, OSError) as ex:
        out.error = str(ex)
    return out


def refresh_source(uri: str, experiment: str, previous: Optional[SourceState] = None,
                   progress=None) -> SourceState:
    """List the runs of one datasets experiment and read each one's CSV
    artifact and hash.txt. Raises nothing: a server that is down comes back
    as a state carrying `error` and the runs known from last time."""
    state = SourceState(uri=uri, experiment=experiment)
    before = {r.run_id: r for r in (previous.runs if previous else [])}
    try:
        experiments = mlf.list_experiments(uri)
        match = next((e for e in experiments if e.name == experiment), None)
        if match is None:
            raise mlf.MlflowError(f"No experiment named '{experiment}' on {uri}.")
        runs = mlf.list_runs(uri, match.experiment_id)
    except mlf.MlflowError as ex:
        state.error = str(ex)
        state.runs = list(before.values())
        state.fetched_at = previous.fetched_at if previous else 0.0
        return state
    done = [0]

    def one(run):
        result = _read_run(uri, experiment, run, before.get(run.run_id))
        done[0] += 1
        if progress is not None:
            progress(f"{experiment}: {done[0]}/{len(runs)} runs")
        return result

    with ThreadPoolExecutor(max_workers=PARALLEL_RUNS) as pool:
        state.runs = list(pool.map(one, runs))
    state.fetched_at = time.time()
    return state


@dataclass
class Link:
    status: str = LINK_NONE
    run: Optional[DatasetRun] = None
    reason: str = ""


def match(index: dict, name: str, size: int, content_hash: str = "") -> Link:
    """The run a file came from, by hash first, else by name and size."""
    runs = [r for s in index.values() for r in s.runs]
    if content_hash:
        for r in runs:
            if r.hash and r.hash == content_hash:
                return Link(LINK_HASH, r, "Same content hash as the run's hash.txt.")
    for r in runs:
        if r.csv_path and r.csv_name() == name and r.csv_size is not None and int(r.csv_size) == int(size):
            if not content_hash:
                why = "the file's content hash is not computed yet"
            elif r.hash:
                why = "but the file's content hash differs from the run's hash.txt"
            else:
                why = "the run has no hash.txt to check the content against"
            return Link(LINK_NAME, r, f"Same file name and size as the run's CSV; {why}.")
    return Link()
