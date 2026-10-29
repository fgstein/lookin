"""Filling a workspace from MLflow: which runs go in, and getting their
files onto this machine.

The Home page's two halves meet here. A results CSV dragged from the runs
table becomes a MountExperiment that remembers its run (see
core/library.MountExperiment), and a workspace's download button turns
those references into files in the download cache.

Adding and downloading are two steps on purpose: adding is instant and
works on a slow link, and downloading several hundred megabytes of
embeddings is something the user decides to wait for, not something a drop
does behind their back.

No UI here, and nothing that talks to a server except `download` -- the
listing of runs and artifacts is the Home page's to do and to cache.
"""
from __future__ import annotations

import os
from stat import S_ISREG
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from lookin.core import library as lib
from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core.data_manager import sidecar_paths_for_csv

# The run parameter a training run records its evaluation dataset under.
# A default only: it is a setting of the Home page (home_view.json,
# "dataset_param"), because it is a convention of one training codebase.
DEFAULT_DATASET_PARAM = "datamodule/configs/database/csv_path"

# Per-workspace download state, as the Home page shows it.
DOWNLOAD_ALL = "all"          # every MLflow file is on disk (green dot)
DOWNLOAD_PARTIAL = "partial"  # some are
DOWNLOAD_NONE = "none"        # none are
DOWNLOAD_LOCAL = "local"      # nothing in it comes from MLflow


def normalize_uri(uri: str) -> str:
    return (uri or "").strip().rstrip("/")


# ---------------------------------------------------------------------------
# what a run offers
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One results CSV of one run, ready to go into a workspace."""
    uri: str
    run: mlf.Run
    csv: mlf.ArtifactFile
    embeddings: Optional[mlf.ArtifactFile] = None
    prototypes: Optional[mlf.ArtifactFile] = None
    several_in_run: bool = False

    def default_name(self) -> str:
        """The run name alone when the run has one CSV -- that is what the
        user recognises -- and run name plus CSV stem when it has several,
        since they would otherwise all be called the same thing."""
        stem = PurePosixPath(self.csv.path).stem
        base = self.run.label()
        return f"{base}/{stem}" if self.several_in_run else base


def candidates_for_run(uri: str, run: mlf.Run, csvs: list, npzs: list) -> list:
    """The CSVs of `run` as Candidates, each with the embeddings and
    prototypes sidecars logged next to it, when there are any -- the same
    naming convention as a local CSV (see data_manager.sidecar_paths_for_csv)."""
    by_path = {a.path: a for a in npzs}
    out = []
    for csv in csvs:
        embed_rel, proto_rel = sidecar_paths_for_csv(csv.path)
        out.append(Candidate(
            uri=normalize_uri(uri), run=run, csv=csv,
            embeddings=by_path.get(embed_rel.as_posix()),
            prototypes=by_path.get(proto_rel.as_posix()),
            several_in_run=len(csvs) > 1,
        ))
    return out


def dataset_of_run(run: mlf.Run, param: str = DEFAULT_DATASET_PARAM) -> str:
    return str((run.params or {}).get(param, "") or "").strip() if param else ""


# ---------------------------------------------------------------------------
# adding
# ---------------------------------------------------------------------------

def unique_name(used, base: str, run_id: str = "") -> str:
    """`base`, or -- when another experiment already has that name -- `base`
    followed by the start of `run_id`: two MLflow runs called the same are
    still two experiments, and the run id is what tells them apart. A
    counter only when that is taken too (or there is no run id)."""
    used = set(used)
    base = base or "experiment"
    if base not in used:
        return base
    if run_id:
        base = f"{base} · {run_id[:8]}"
        if base not in used:
            return base
    name, n = f"{base} (2)", 3
    while name in used:
        name = f"{base} ({n})"
        n += 1
    return name


def _unique_name(mount: lib.Mount, base: str, run_id: str = "") -> str:
    return unique_name((e.name for e in mount.experiments), base, run_id)


def contains(workspace: lib.Workspace, uri: str, run_id: str, artifact_path: str) -> bool:
    uri = normalize_uri(uri)
    return any(
        e.mlflow_run_id == run_id and e.artifact_path == artifact_path
        and normalize_uri(e.mlflow_uri) == uri
        for e in workspace.mount.experiments
    )


@dataclass
class AddResult:
    added: list = field(default_factory=list)      # experiment names
    skipped: list = field(default_factory=list)    # (label, reason)
    dataset_set: str = ""                          # the dataset adopted from a run, if any

    def message(self) -> str:
        bits = []
        if self.added:
            bits.append(f"Added {len(self.added)} experiment(s)")
        if self.dataset_set:
            bits.append(f"dataset set to {Path(self.dataset_set).name}")
        if self.skipped:
            bits.append(f"skipped {len(self.skipped)}: "
                        + "; ".join(f"{label} ({reason})" for label, reason in self.skipped))
        return ", ".join(bits) + "." if bits else "Nothing to add."


def add_candidates(workspace: lib.Workspace, candidates: list,
                   dataset_param: str = DEFAULT_DATASET_PARAM) -> AddResult:
    """Add `candidates` to `workspace`'s mount. Does not save -- the caller
    does, once, whatever came of it.

    A workspace is one dataset. A run that records which dataset it was
    evaluated on (`dataset_param`) either gives an empty workspace its
    dataset, or is refused when it names a different one: scoring model A
    on dataset X next to model B on dataset Y renders plausible, wrong
    comparisons with no other symptom. A run that records nothing is let in
    -- that is a guess the user made, not one made for them.
    """
    result = AddResult()
    mount = workspace.mount
    for c in candidates:
        label = c.default_name()
        if contains(workspace, c.uri, c.run.run_id, c.csv.path):
            result.skipped.append((label, "already in this workspace"))
            continue
        dataset = dataset_of_run(c.run, dataset_param)
        if dataset:
            if not mount.dataset_path:
                mount.dataset_path = dataset
                result.dataset_set = dataset
            elif dataset != mount.dataset_path:
                result.skipped.append((label, f"evaluated on {Path(dataset).name}, "
                                              f"not {Path(mount.dataset_path).name}"))
                continue
        name = _unique_name(mount, label, c.run.run_id)
        experiment = lib.MountExperiment(
            name=name,
            path=str(mlflow_cache.cached_path(c.run.run_id, c.csv.path)),
            mlflow_uri=c.uri,
            mlflow_experiment_id=c.run.experiment_id,
            mlflow_run_id=c.run.run_id,
            mlflow_run_name=c.run.run_name,
            artifact_path=c.csv.path,
            artifact_size=c.csv.size,
        )
        if c.embeddings is not None:
            experiment.embeddings_artifact = c.embeddings.path
            experiment.embeddings_size = c.embeddings.size
            experiment.embeddings_path = str(mlflow_cache.cached_path(c.run.run_id, c.embeddings.path))
        if c.prototypes is not None:
            experiment.prototypes_artifact = c.prototypes.path
            experiment.prototypes_size = c.prototypes.size
            experiment.prototypes_path = str(mlflow_cache.cached_path(c.run.run_id, c.prototypes.path))
        mount.experiments.append(experiment)
        result.added.append(name)
    return result


def remove_experiment(workspace: lib.Workspace, name: str) -> bool:
    before = len(workspace.mount.experiments)
    workspace.mount.experiments = [e for e in workspace.mount.experiments if e.name != name]
    return len(workspace.mount.experiments) != before


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------

@dataclass
class _File:
    experiment: lib.MountExperiment
    kind: str             # "csv" | "embeddings" | "prototypes"
    artifact_path: str
    size: Optional[int]


def _files_of(experiment: lib.MountExperiment) -> list:
    if not experiment.from_mlflow():
        return []
    files = [_File(experiment, "csv", experiment.artifact_path, experiment.artifact_size)]
    if experiment.embeddings_artifact:
        files.append(_File(experiment, "embeddings", experiment.embeddings_artifact, experiment.embeddings_size))
    if experiment.prototypes_artifact:
        files.append(_File(experiment, "prototypes", experiment.prototypes_artifact, experiment.prototypes_size))
    return files


def _local_path(f: _File) -> str:
    e = f.experiment
    return {"csv": e.path, "embeddings": e.embeddings_path, "prototypes": e.prototypes_path}[f.kind]


def _present(f: _File) -> bool:
    """On disk where the experiment reads it from. For a file in the
    download cache the size has to match too; for one elsewhere (the data/
    of an imported workspace) being there is enough.

    Plain strings and one os.stat rather than pathlib: the Home page asks
    this for every file of every workspace on each redraw -- a few thousand
    times with a few hundred workspaces -- and building Path objects was
    most of what that redraw cost.
    """
    local = _local_path(f)
    if not local:
        return False
    try:
        stat = os.stat(local)
    except OSError:
        return False
    if not S_ISREG(stat.st_mode):
        return False
    if f.size is not None and local == mlflow_cache.cached_path_str(f.experiment.mlflow_run_id, f.artifact_path):
        return stat.st_size == int(f.size)
    return True


def is_downloaded(experiment: lib.MountExperiment) -> bool:
    """True for a local experiment, and for an MLflow one whose files are
    all here."""
    return all(_present(f) for f in _files_of(experiment))


def missing_files(workspace: lib.Workspace) -> list:
    return [f for e in workspace.mount.experiments for f in _files_of(e) if not _present(f)]


@dataclass
class DownloadSummary:
    state: str                      # one of the DOWNLOAD_* constants
    missing: int = 0                # files still on the server
    missing_bytes: Optional[int] = 0   # their total size; None when one is unknown


def download_summary(workspace: lib.Workspace) -> DownloadSummary:
    """Everything a workspace card says about downloading, in one pass over
    its files -- the card used to make three, each re-checking every file."""
    files = [f for e in workspace.mount.experiments for f in _files_of(e)]
    if not files:
        return DownloadSummary(DOWNLOAD_LOCAL)
    missing = [f for f in files if not _present(f)]
    size: Optional[int] = 0
    for f in missing:
        if f.size is None:
            size = None
            break
        size += f.size
    if not missing:
        state = DOWNLOAD_ALL
    else:
        state = DOWNLOAD_PARTIAL if len(missing) < len(files) else DOWNLOAD_NONE
    return DownloadSummary(state, len(missing), size)


def download_state(workspace: lib.Workspace) -> str:
    return download_summary(workspace).state


def missing_bytes(workspace: lib.Workspace) -> Optional[int]:
    """What the download button will fetch, or None when a size is unknown."""
    return download_summary(workspace).missing_bytes


def download(workspace: lib.Workspace,
             progress: Optional[Callable[[int, int, str, int, Optional[int]], None]] = None) -> tuple:
    """Fetch every missing MLflow file of `workspace` into the cache.

    Returns ``(fetched, errors)``. Carries on past a failed file -- one
    unreachable server must not stop the runs from another -- and reports
    each failure by experiment name.

    `progress(index, count, name, done_bytes, total_bytes)` is called as
    each file comes down, for the card's progress line.
    """
    todo = missing_files(workspace)
    fetched, errors = 0, []
    for index, f in enumerate(todo, start=1):
        e = f.experiment
        label = f"{e.name} ({f.kind})" if f.kind != "csv" else e.name

        def report(done, total, _i=index, _label=label):
            if progress is not None:
                progress(_i, len(todo), _label, done, total)

        try:
            dest = mlflow_cache.download_cached(e.mlflow_uri, e.mlflow_run_id, f.artifact_path,
                                                f.size, progress=report)
        except (mlf.MlflowError, OSError) as ex:
            errors.append(f"{label}: {ex}")
            continue
        # A file present elsewhere never gets here, so the experiment is
        # pointed at the cache: that is where this copy is.
        if f.kind == "csv":
            e.path = str(dest)
        elif f.kind == "embeddings":
            e.embeddings_path = str(dest)
        else:
            e.prototypes_path = str(dest)
        fetched += 1
    return fetched, errors


def point_at_cache(workspace: lib.Workspace) -> bool:
    """Point every MLflow experiment whose files are in the download cache
    at them. Returns whether anything changed.

    What a finished download applies to the workspace *as it is on disk
    now* (see library.update_workspace): the download itself worked from a
    copy read before it started, and an experiment added meanwhile must
    survive it.
    """
    changed = False
    for e in workspace.mount.experiments:
        for f in _files_of(e):
            # is_cached first: it is False for a path that cached_path
            # refuses (one that would point outside the cache).
            if not mlflow_cache.is_cached(e.mlflow_run_id, f.artifact_path, f.size):
                continue
            cached = mlflow_cache.cached_path(e.mlflow_run_id, f.artifact_path)
            if _local_path(f) == str(cached):
                continue
            if Path(_local_path(f)).is_file():
                continue    # a copy elsewhere (an imported workspace's data/) stays in use
            if f.kind == "csv":
                e.path = str(cached)
            elif f.kind == "embeddings":
                e.embeddings_path = str(cached)
            else:
                e.prototypes_path = str(cached)
            changed = True
    return changed


def downloaded_only(mount: lib.Mount) -> lib.Mount:
    """A copy of `mount` without the experiments that are not on disk yet --
    what "open without them" opens."""
    from dataclasses import replace
    return replace(mount, experiments=[e for e in mount.experiments if is_downloaded(e)])
