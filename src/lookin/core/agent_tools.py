"""What an agent can ask Lookin, as plain functions returning JSON-ready data.

These are the tools the MCP endpoint serves (see lookin/mcp_server.py), kept
free of any MCP import so they can be tested and called on their own. Each
returns a dict of plain values: str, int, float, bool, None, lists, dicts.

Two rules shape every answer:

- **Small.** The reader may be a local model with a few thousand tokens of
  context, so a tool returns the top N of anything long and says how many
  there were in total, never a whole table.
- **Caveats travel with the numbers.** The UI shows a banner when a dataset
  changed or an experiment barely overlaps it; a model given the number
  without the banner will repeat it with confidence. Every answer built
  from a workspace carries its `warnings`.

Loading a workspace is the slow part (reading CSVs of hundreds of thousands
of rows), so the last few loaded are kept, along with the metrics computed
on them. Calls are serialized by one lock: the MCP SDK runs each call on a
worker thread, AppState is not thread-safe, and two workspaces loading at
once would also double the memory peak.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lookin.core import comparison as cmp
from lookin.core import library as lib
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources
from lookin.core.metrics import PER_CLASS_COLUMNS, compute_metrics
from lookin.core.workspace_state import LoadedWorkspace, find_workspace, load_workspace_state

# A loaded workspace holds its whole dataset in memory. Two lets an agent
# go back and forth between workspaces without holding the library.
MAX_CACHED_WORKSPACES = 2
DEFAULT_TOP = 20
MAX_TOP = 200
# get_examples shows this many metadata columns per crop; a dataset can have
# hundreds, and the first few are usually the ones someone put there on purpose.
MAX_EXAMPLE_METADATA_COLUMNS = 8
DEFAULT_MAX_RUNS = 20
MAX_RUNS = 200

# The summary get_metrics returns, in the order the Metrics tab reads them.
SUMMARY_FIELDS = (
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "cohen_kappa",
    "macro_roc_auc", "weighted_roc_auc", "macro_avg_precision", "weighted_avg_precision",
    "log_loss", "brier_score", "ece", "mce",
)

_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()  # workspace id -> _Entry


class _Entry:
    def __init__(self, stamp: tuple, loaded: LoadedWorkspace):
        self.stamp = stamp
        self.loaded = loaded
        self.metrics: dict = {}  # experiment name -> ExperimentMetrics


def clear_cache() -> None:
    with _lock:
        _cache.clear()


# ---- JSON helpers ---------------------------------------------------------

def _clean(value):
    """A value as plain JSON: numpy scalars unwrapped, NaN and inf as None
    (JSON has neither, and a model reads null as "not available")."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 4)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _top(top) -> int:
    try:
        top = int(top)
    except (TypeError, ValueError):
        top = DEFAULT_TOP
    return max(1, min(top, MAX_TOP))


def _ms_to_iso(ms) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


# ---- workspace loading ----------------------------------------------------

def _stamp(workspace: lib.Workspace) -> tuple:
    """What has to be unchanged for a cached load to still be this workspace:
    its manifest (experiments added or removed) and its dataset file. The
    manifest's mtime and not just its `updated_at`, which only has
    one-second resolution."""
    def mtime_ns(path) -> int:
        try:
            return Path(path).stat().st_mtime_ns if path else 0
        except OSError:
            return 0
    manifest = Path(workspace.path) / lib.MANIFEST_NAME if workspace.path else None
    return workspace.updated_at, mtime_ns(manifest), mtime_ns(workspace.mount.dataset_path)


def _entry(workspace_id: str) -> _Entry:
    """Must be called with _lock held."""
    workspace = find_workspace(workspace_id)
    stamp = _stamp(workspace)
    entry = _cache.get(workspace.id)
    if entry is not None and entry.stamp == stamp:
        _cache.move_to_end(workspace.id)
        return entry
    _cache.pop(workspace.id, None)
    entry = _Entry(stamp, load_workspace_state(workspace))
    _cache[workspace.id] = entry
    while len(_cache) > MAX_CACHED_WORKSPACES:
        _cache.popitem(last=False)
    return entry


def _experiment_name(entry: _Entry, name: str) -> str:
    if name in entry.loaded.state.experiments:
        return name
    if name in entry.loaded.skipped:
        reason = next((w for w in entry.loaded.warnings if repr(name) in w), "")
        raise ValueError(f"Experiment {name!r} is in this workspace but is not loaded. {reason}".strip())
    available = ", ".join(entry.loaded.state.experiments) or "none"
    raise ValueError(f"No experiment named {name!r} in this workspace. Loaded experiments: {available}.")


def _metrics(entry: _Entry, name: str):
    if name not in entry.metrics:
        state = entry.loaded.state
        experiment = state.experiments[name]
        entry.metrics[name] = compute_metrics(name, state.merged(name), state.dataset.columns.label_col,
                                              experiment.score_cols, experiment.class_labels)
    return entry.metrics[name]


def _context(entry: _Entry) -> dict:
    """Which pool the numbers below are computed on."""
    loaded = entry.loaded
    state = loaded.state
    out = {"workspace_id": loaded.workspace.id, "workspace": loaded.workspace.label()}
    if state.dataset is not None:
        out["split"] = state.split_kind
        out["labels"] = "original"  # the review overlay is never applied here
        review = state.review
        if review is not None and review.versions:
            out["review_versions_available"] = sorted(review.versions)
    out["warnings"] = list(loaded.warnings)
    return out


# ---- tools ----------------------------------------------------------------

def list_workspaces() -> dict:
    workspaces = []
    for w in lib.list_workspaces():
        workspaces.append({
            "id": w.id,
            "name": w.name,
            "description": w.description,
            "tags": list(w.tags),
            "dataset": w.mount.label() if w.mount.dataset_path else None,
            "split": w.mount.split_kind or None,
            "experiments": [e.name for e in w.mount.experiments],
            "download_state": workspace_sources.download_state(w),
            "updated_at": w.updated_at,
        })
    return {"library": str(lib.library_root()), "count": len(workspaces), "workspaces": workspaces}


def get_workspace(workspace_id: str) -> dict:
    w = find_workspace(workspace_id)
    m = w.mount
    experiments = []
    for e in m.experiments:
        item = {"name": e.name, "downloaded": workspace_sources.is_downloaded(e),
                "has_embeddings": bool(e.embeddings_path or e.embeddings_artifact)}
        if e.from_mlflow():
            item.update(mlflow_run_id=e.mlflow_run_id, mlflow_run_name=e.mlflow_run_name,
                        mlflow_experiment_id=e.mlflow_experiment_id, mlflow_server=e.mlflow_uri,
                        artifact=e.artifact_path)
        else:
            item["results_csv"] = e.path
        experiments.append(item)
    return _clean({
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "tags": list(w.tags),
        "created_at": w.created_at,
        "updated_at": w.updated_at,
        "dataset": {
            "path": m.dataset_path or None,
            "on_this_machine": m.exists(),
            "columns": {"uuid": m.uuid_col, "crop": m.crop_col, "label": m.label_col},
            "row_filter": {"column": m.row_filter_col, "value": m.row_filter_value} if m.row_filter_col else None,
            "split": m.split_kind or None,
            "review_version": m.review_version or None,
        },
        "experiments": experiments,
        "snapshots": [{"id": s.id, "name": s.name, "updated_at": s.updated_at} for s in lib.list_snapshots(w)],
        "download_state": workspace_sources.download_state(w),
        "provenance": dict(w.provenance),
    })


def get_metrics(workspace_id: str, experiments: list | None = None) -> dict:
    with _lock:
        entry = _entry(workspace_id)
        state = entry.loaded.state
        names = [_experiment_name(entry, n) for n in experiments] if experiments else list(state.experiments)
        rows = []
        for name in names:
            m = _metrics(entry, name)
            row = {"experiment": name, "n_samples": m.n_samples, "n_classes": len(state.experiments[name].score_cols)}
            row.update({f: getattr(m, f) for f in SUMMARY_FIELDS})
            rows.append(row)
        out = _context(entry)
        if state.dataset is not None:
            out["dataset_rows"] = len(state.dataset.df)
        out["experiments"] = rows
        if entry.loaded.skipped:
            out["not_loaded"] = list(entry.loaded.skipped)
        return _clean(out)


def get_per_class(workspace_id: str, experiment: str, sort_by: str = "f1", ascending: bool = True,
                  top: int = DEFAULT_TOP) -> dict:
    if sort_by not in PER_CLASS_COLUMNS:
        raise ValueError(f"sort_by must be one of {', '.join(PER_CLASS_COLUMNS)}; got {sort_by!r}.")
    with _lock:
        entry = _entry(workspace_id)
        name = _experiment_name(entry, experiment)
        table = _metrics(entry, name).per_class
        ordered = table.sort_values(sort_by, ascending=ascending, na_position="last", kind="stable")
        out = _context(entry)
        out.update(experiment=name, n_classes=len(table), sorted_by=sort_by, ascending=bool(ascending),
                   classes=ordered.head(_top(top))[PER_CLASS_COLUMNS].to_dict(orient="records"))
        return _clean(out)


def get_confusions(workspace_id: str, experiment: str, top: int = DEFAULT_TOP) -> dict:
    with _lock:
        entry = _entry(workspace_id)
        name = _experiment_name(entry, experiment)
        confusion = _metrics(entry, name).confusion
        counts = confusion.to_numpy()
        support = counts.sum(axis=1)
        errors = counts.copy()
        np.fill_diagonal(errors, 0)
        true_idx, pred_idx = np.nonzero(errors)
        pairs = sorted(zip(true_idx, pred_idx), key=lambda p: -errors[p])
        rows = [{
            "true_label": confusion.index[t],
            "predicted_label": confusion.columns[p],
            "count": int(errors[t, p]),
            "share_of_true_class": errors[t, p] / support[t] if support[t] else None,
        } for t, p in pairs[:_top(top)]]
        out = _context(entry)
        out.update(experiment=name, n_errors=int(errors.sum()), n_samples=int(counts.sum()),
                   n_confused_pairs=len(pairs), confusions=rows)
        return _clean(out)


def compare_experiments(workspace_id: str, baseline: str, candidate: str, top: int = DEFAULT_TOP) -> dict:
    with _lock:
        entry = _entry(workspace_id)
        baseline = _experiment_name(entry, baseline)
        candidate = _experiment_name(entry, candidate)
        state = entry.loaded.state
        joined = cmp.build_comparison(state, baseline, candidate)
        diff = cmp.compare_experiments(state, baseline, candidate, joined=joined)
        per_class = cmp.per_class_change(joined, diff.labels)
        per_class = per_class.loc[per_class["net"].abs().sort_values(ascending=False, kind="stable").index]
        per_class = per_class[(per_class["fixes"] > 0) | (per_class["regressions"] > 0)]
        out = _context(entry)
        out.update(
            baseline=baseline, candidate=candidate,
            n_common=diff.n_common, n_baseline_only=diff.n_baseline_only, n_candidate_only=diff.n_candidate_only,
            baseline_accuracy=diff.baseline_accuracy, candidate_accuracy=diff.candidate_accuracy,
            accuracy_delta=diff.accuracy_delta,
            fixed=diff.counts.get(cmp.OUTCOME_FIX, 0), broken=diff.counts.get(cmp.OUTCOME_REGRESSION, 0),
            both_right=diff.counts.get(cmp.OUTCOME_BOTH_RIGHT, 0), both_wrong=diff.counts.get(cmp.OUTCOME_BOTH_WRONG, 0),
            net_change=diff.net_change(), n_prediction_changed=diff.n_moved,
            n_classes_changed=len(per_class),
            per_class_change=per_class.head(_top(top)).to_dict(orient="records"),
        )
        return _clean(out)


def get_examples(workspace_id: str, experiment: str, true_label: str | None = None,
                 predicted_label: str | None = None, errors_only: bool = True, limit: int = DEFAULT_TOP) -> dict:
    with _lock:
        entry = _entry(workspace_id)
        name = _experiment_name(entry, experiment)
        state = entry.loaded.state
        cols = state.dataset.columns
        merged = state.merged(name)
        mask = pd.Series(True, index=merged.index)
        if errors_only:
            mask &= ~merged["correct"]
        if true_label is not None:
            mask &= merged[cols.label_col] == true_label
        if predicted_label is not None:
            mask &= merged["predicted_label"] == predicted_label
        # Most confident first: a confident error is the one worth a look
        # (a mislabel, or a real blind spot), a hesitant one is just hard.
        rows = merged[mask].sort_values("predicted_score", ascending=False, kind="stable")
        metadata = state.dataset.metadata_columns()[:MAX_EXAMPLE_METADATA_COLUMNS]
        examples = [{
            "uuid": r[cols.uuid_col],
            "label": r[cols.label_col],
            "predicted_label": r["predicted_label"],
            "confidence": r["predicted_score"],
            "crop": r[cols.crop_col],
            "metadata": {c: r[c] for c in metadata if c in r},
        } for _, r in rows.head(_top(limit)).iterrows()]
        out = _context(entry)
        out.update(experiment=name, n_matching=int(mask.sum()), sorted_by="confidence (descending)",
                   examples=examples)
        return _clean(out)


def search_mlflow_runs(filter: str = "", experiment_names: list | None = None, server: str | None = None,
                       max_results: int = DEFAULT_MAX_RUNS) -> dict:
    """Runs on the MLflow servers configured on Lookin's Home page."""
    from lookin.core.home_config import HomeConfig

    config = HomeConfig()
    servers = [s["uri"] for s in config.servers]
    if server:
        uri = workspace_sources.normalize_uri(server)
        if uri not in servers:
            raise ValueError(f"MLflow server {server!r} is not configured in Lookin. Configured: "
                             f"{', '.join(servers) or 'none'}.")
        servers = [uri]
    if not servers:
        raise ValueError("No MLflow server is configured in Lookin (add one on the Home page).")
    limit = max(1, min(int(max_results or DEFAULT_MAX_RUNS), MAX_RUNS))

    # Which workspace each run is already in, the way Home's badges say it.
    in_workspaces: dict = {}
    for w in lib.list_workspaces():
        for e in w.mount.experiments:
            if e.mlflow_run_id:
                in_workspaces.setdefault(e.mlflow_run_id, set()).add(w.id)

    runs, errors, truncated = [], [], False
    for uri in servers:
        try:
            experiments = mlf.list_experiments(uri)
            if experiment_names:
                wanted = set(experiment_names)
            else:
                wanted = set(config.shown_experiments(uri)) - set(config.hidden_experiments(uri))
            chosen = {e.experiment_id: e.name for e in experiments if e.name in wanted}
            unknown = sorted(wanted - set(chosen.values()))
            if unknown:
                errors.append(f"{uri}: no experiment named {', '.join(map(repr, unknown))}")
            found, more = mlf.search_runs(uri, list(chosen), filter_string=filter or "", max_pages=1)
        except mlf.MlflowError as ex:
            errors.append(f"{uri}: {ex}")
            continue
        truncated = truncated or more
        for r in found:
            runs.append({
                "run_id": r.run_id,
                "run_name": r.run_name,
                "experiment": chosen.get(r.experiment_id, r.experiment_id),
                "server": uri,
                "status": r.status,
                "start_time": _ms_to_iso(r.start_time),
                "user": r.user,
                "params": dict(r.params),
                "metrics": dict(r.metrics),
                "in_workspaces": sorted(in_workspaces.get(r.run_id, ())),
            })
    runs.sort(key=lambda r: r["start_time"] or "", reverse=True)
    out = {"filter": filter or "", "n_found": len(runs), "truncated": truncated or len(runs) > limit,
           "runs": runs[:limit]}
    if errors:
        out["errors"] = errors
    return _clean(out)
