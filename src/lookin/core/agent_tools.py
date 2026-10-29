"""What an agent can ask Lookin, as plain functions returning JSON-ready data.

These are the reading tools the MCP endpoint serves (see lookin/mcp_server.py);
the ones that change the library are in core/agent_actions.py and the ones
that produce pictures and files in core/agent_outputs.py. All are kept free
of any MCP import so they can be tested and called on their own. Each
returns a dict of plain values: str, int, float, bool, None, lists, dicts.

Two rules shape every answer:

- **Small.** The reader may be a local model with a few thousand tokens of
  context, so a tool returns the top N of anything long and says how many
  there were in total, never a whole table.
- **Caveats travel with the numbers.** The UI shows a banner when a dataset
  changed or an experiment barely overlaps it; a model given the number
  without the banner will repeat it with confidence. Every answer built
  from a workspace carries its `warnings`.
- **A link to see it.** Every answer about a workspace carries an
  `open_url` (core/links.py) that opens the same thing in `lookin web`,
  and every crop one that opens that crop -- the number is the start of a
  look, not the end of it.

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

from lookin.core import calibration as cal
from lookin.core import candidates as cand
from lookin.core import comparison as cmp
from lookin.core import confidence as conf
from lookin.core import dataset_filter as dsf
from lookin.core import library as lib
from lookin.core import links
from lookin.core import mlflow_client as mlf
from lookin.core import slices as sl
from lookin.core import workspace_sources
from lookin.core.embedding_analysis import nearest_neighbours
from lookin.core.embedding_context import build_analysis_context
from lookin.core.metrics import (
    PER_CLASS_COLUMNS, compute_metrics, compute_reliability, diagnose_scores,
)
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


def _context(entry: _Entry, tab: str = "") -> dict:
    """Which pool the numbers below are computed on, and where to see them."""
    loaded = entry.loaded
    state = loaded.state
    out = {"workspace_id": loaded.workspace.id, "workspace": loaded.workspace.label(),
           "open_url": links.link(loaded.workspace.id, tab=tab)}
    if state.dataset is not None:
        out["split"] = state.split_kind
        out["labels"] = "original"  # the review overlay is never applied here
        review = state.review
        if review is not None and review.versions:
            out["review_versions_available"] = sorted(review.versions)
    out["warnings"] = list(loaded.warnings)
    return out


# ---- tools ----------------------------------------------------------------

def _group_names(workspaces: list) -> dict:
    """workspace id -> the name of the Home group it sits in."""
    from lookin.core import board as bd

    return {w.id: column.get("name", "") for column, items in bd.layout(bd.load(), workspaces) for w in items}


def list_workspaces() -> dict:
    all_workspaces = lib.list_workspaces()
    groups = _group_names(all_workspaces)
    workspaces = []
    for w in all_workspaces:
        workspaces.append({
            "id": w.id,
            "name": w.name,
            "group": groups.get(w.id) or None,
            "description": w.description,
            "tags": list(w.tags),
            "dataset": w.mount.label() if w.mount.dataset_path else None,
            "split": w.mount.split_kind or None,
            "experiments": [e.name for e in w.mount.experiments],
            "download_state": workspace_sources.download_state(w),
            "updated_at": w.updated_at,
            "open_url": links.link(w.id),
        })
    return {"library": str(lib.library_root()), "count": len(workspaces),
            "groups": sorted({g for g in groups.values() if g}), "workspaces": workspaces}


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
    from lookin.core.dataset_filter import describe

    return _clean({
        "id": w.id,
        "name": w.name,
        "open_url": links.link(w.id),
        "group": _group_names(lib.list_workspaces()).get(w.id) or None,
        "description": w.description,
        "tags": list(w.tags),
        "created_at": w.created_at,
        "updated_at": w.updated_at,
        "dataset": {
            "path": m.dataset_path or None,
            "on_this_machine": m.exists(),
            "columns": {"uuid": m.uuid_col, "crop": m.crop_col, "label": m.label_col},
            "image_root": m.image_root or None,
            "row_filter": {"column": m.row_filter_col, "value": m.row_filter_value} if m.row_filter_col else None,
            "filter": describe(m.dataset_filter) if m.dataset_filter else [],
            "split": m.split_kind or None,
            "review_version": m.review_version or None,
        },
        "experiments": experiments,
        "snapshots": [{"id": s.id, "name": s.name, "description": s.description, "updated_at": s.updated_at,
                       "open_url": links.link(snapshot_id=s.id)} for s in lib.list_snapshots(w)],
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
        out = _context(entry, "metrics")
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
        out = _context(entry, "metrics")
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
        out = _context(entry, "metrics")
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
        out = _context(entry, "compare")
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


# Orders get_examples and a mosaic can put crops in. Keys, compared in code.
SORT_MOST_CONFIDENT = "most_confident"
SORT_LEAST_CONFIDENT = "least_confident"
SORT_DATASET = "dataset_order"
EXAMPLE_SORTS = (SORT_MOST_CONFIDENT, SORT_LEAST_CONFIDENT, SORT_DATASET)


def select_rows(state, experiment: str = "", true_label=None, predicted_label=None, errors_only: bool = False,
                where: str = "", sort: str = SORT_MOST_CONFIDENT) -> pd.DataFrame:
    """The crops matching every condition given, in `sort` order -- the
    rows of the experiment's merged frame (with its prediction), or of the
    dataset when no experiment is named. `where` is a query in the Filter
    tab's language over the dataset's columns (core/dataset_filter.py),
    parsed and checked, never run as Python. Labels compare as text: they
    come from the user's CSV, integers on one side and strings on the other."""
    cols = state.dataset.columns
    frame = state.merged(experiment) if experiment else state.dataset.df
    if (errors_only or predicted_label is not None) and not experiment:
        raise ValueError("errors_only and predicted_label need an experiment.")
    if sort not in EXAMPLE_SORTS:
        raise ValueError(f"sort must be one of {', '.join(EXAMPLE_SORTS)}; got {sort!r}.")
    mask = pd.Series(True, index=frame.index)
    if errors_only:
        mask &= ~frame["correct"].astype(bool)
    if true_label is not None:
        mask &= frame[cols.label_col].astype(str) == str(true_label)
    if predicted_label is not None:
        mask &= frame["predicted_label"].astype(str) == str(predicted_label)
    if where:
        mask &= dsf.query_mask(frame, where).reindex(frame.index, fill_value=False)
    rows = frame[mask]
    if experiment and sort != SORT_DATASET:
        # Most confident first by default: a confident error is the one
        # worth a look (a mislabel, or a real blind spot), a hesitant one
        # is just hard.
        rows = rows.sort_values("predicted_score", ascending=sort == SORT_LEAST_CONFIDENT, kind="stable")
    return rows


def get_examples(workspace_id: str, experiment: str = "", true_label: str | None = None,
                 predicted_label: str | None = None, errors_only: bool = True, where: str = "",
                 sort: str = SORT_MOST_CONFIDENT, limit: int = DEFAULT_TOP) -> dict:
    with _lock:
        entry = _entry(workspace_id)
        name = _experiment_name(entry, experiment) if experiment else ""
        state = entry.loaded.state
        if state.dataset is None:
            raise ValueError("This workspace has no dataset loaded.")
        cols = state.dataset.columns
        rows = select_rows(state, name, true_label, predicted_label, errors_only and bool(name), where, sort)
        metadata = state.dataset.metadata_columns()[:MAX_EXAMPLE_METADATA_COLUMNS]
        workspace_id = entry.loaded.workspace.id
        examples = []
        for _, r in rows.head(_top(limit)).iterrows():
            item = {"uuid": r[cols.uuid_col] if cols.uuid_col else None, "label": r[cols.label_col]}
            if name:
                item.update(predicted_label=r["predicted_label"], confidence=r["predicted_score"])
            item["crop"] = r[cols.crop_col]
            item["metadata"] = {c: r[c] for c in metadata if c in r}
            if cols.uuid_col:
                item["open_url"] = links.link(workspace_id, crop=str(r[cols.uuid_col]), experiment=name)
            examples.append(item)
        out = _context(entry, "gallery")
        out.update(experiment=name or None, n_matching=len(rows),
                   sorted_by=sort if name else SORT_DATASET, examples=examples)
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


def list_mlflow_run_files(run_id: str, server: str | None = None) -> dict:
    """The results CSVs a run logged (and the embeddings / prototypes .npz
    beside each), as add_mlflow_run reads them."""
    uri = _server_for_run(server)
    csvs, npzs = mlf.list_csv_and_npz_artifacts(uri, run_id)
    return _clean({
        "server": uri, "run_id": run_id,
        "csvs": [{"path": a.path, "size": a.size} for a in csvs],
        "npz": [{"path": a.path, "size": a.size} for a in npzs],
    })


def _server_for_run(server: str | None) -> str:
    """`server` if it is configured, else the only configured one."""
    from lookin.core.home_config import HomeConfig

    servers = [s["uri"] for s in HomeConfig(fallback_uri=mlf.default_tracking_uri()).servers]
    if server:
        uri = workspace_sources.normalize_uri(server)
        if uri not in servers:
            raise ValueError(f"MLflow server {server!r} is not configured in Lookin. Configured: "
                             f"{', '.join(servers) or 'none'}.")
        return uri
    if len(servers) == 1:
        return servers[0]
    if not servers:
        raise ValueError("No MLflow server is configured in Lookin (add one on the Home page).")
    raise ValueError(f"Several MLflow servers are configured; name one: {', '.join(servers)}.")


# ---- the dataset -----------------------------------------------------------

MAX_CLASSES_LISTED = 50
HISTOGRAM_BINS = 20


def get_dataset_summary(workspace_id: str) -> dict:
    """What the dataset holds: its rows (before and after the workspace's
    filter), its columns by kind, and its classes."""
    with _lock:
        entry = _entry(workspace_id)
        state = entry.loaded.state
        dataset = state.dataset
        if dataset is None:
            raise ValueError("This workspace has no dataset loaded.")
        cols = dataset.columns
        kinds = dataset.column_kinds()
        counts = dataset.df[cols.label_col].value_counts()
        out = _context(entry, "statistics")
        out.update(
            dataset=str(dataset.csv_path), rows=len(dataset.df), rows_before_filter=len(dataset.full_df),
            filter=dsf.describe(state.dataset_filter) if state.dataset_filter else [],
            columns={"uuid": cols.uuid_col, "crop": cols.crop_col, "label": cols.label_col,
                     "numeric": kinds.numeric, "categorical": kinds.categorical, "binary": kinds.binary,
                     "high_cardinality": kinds.high_cardinality},
            n_classes=int(len(counts)),
            classes=[{"label": k, "count": int(v)} for k, v in counts.head(MAX_CLASSES_LISTED).items()],
            experiments=list(state.experiments),
        )
        if len(counts) > MAX_CLASSES_LISTED:
            out["classes_note"] = f"The {MAX_CLASSES_LISTED} largest of {len(counts):,} classes."
        return _clean(out)


def get_column_stats(workspace_id: str, column: str, top: int = DEFAULT_TOP) -> dict:
    """One dataset column: a numeric one's range, quantiles and histogram,
    a categorical one's most frequent values -- the numbers behind the
    Statistics tab's chart."""
    with _lock:
        entry = _entry(workspace_id)
        dataset = entry.loaded.state.dataset
        if dataset is None:
            raise ValueError("This workspace has no dataset loaded.")
        if column not in dataset.df.columns:
            raise ValueError(f"No column {column!r}. Columns: {', '.join(map(str, dataset.df.columns))}.")
        series = dataset.df[column]
        out = _context(entry, "statistics")
        out.update(column=column, rows=len(series), missing=int(series.isna().sum()),
                   n_unique=int(series.nunique(dropna=True)))
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = series.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            out["kind"] = "numeric"
            if finite.size:
                counts, edges = np.histogram(finite, bins=HISTOGRAM_BINS)
                out.update(min=finite.min(), max=finite.max(), mean=finite.mean(), std=finite.std(),
                           quantiles={f"p{q}": float(np.percentile(finite, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
                           histogram=[{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(c)}
                                      for i, c in enumerate(counts)])
            if finite.size < series.notna().sum():
                out["infinite"] = int(series.notna().sum() - finite.size)
        else:
            counts = series.value_counts()
            out["kind"] = "categorical"
            out["values"] = [{"value": k, "count": int(v)} for k, v in counts.head(_top(top)).items()]
        return _clean(out)


# ---- calibration and confidence ---------------------------------------------

def _calibration_of(entry: _Entry, name: str) -> dict:
    """What the Calibration tab computes for one experiment."""
    state = entry.loaded.state
    experiment = state.experiments[name]
    merged = state.merged(name)
    confidence = merged["predicted_score"].to_numpy(dtype=float)
    correct = merged["correct"].to_numpy().astype(bool)
    labels_for_cols = experiment.class_labels[: len(experiment.score_cols)]
    diagnostics = diagnose_scores(merged[experiment.score_cols].to_numpy(), labels_for_cols)
    index = cal.build_index(confidence, correct)
    return {
        "index": index, "curve": cal.risk_coverage(index), "diagnostics": diagnostics,
        "reliability": compute_reliability(merged, diagnostics=diagnostics),
        "histogram": cal.confidence_histogram(confidence, correct),
        "accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "mean_confidence": float(np.nanmean(confidence)) if confidence.size else float("nan"),
    }


def _point(p) -> dict:
    return {"threshold": p.threshold, "accepted": p.n_accepted, "coverage": p.coverage,
            "accuracy_accepted": p.accuracy, "errors_accepted": p.n_errors_accepted,
            "abstained": p.n_abstained, "accuracy_abstained": p.accuracy_abstained,
            "unreachable": bool(p.unreachable)}


def get_calibration(workspace_id: str, experiments: list | None = None, target_accuracy: float | None = None,
                    threshold: float | None = None, fit_temperature: bool = False) -> dict:
    """Whether the confidence can be believed (ECE, MCE, confidence gap) and
    what refusing to answer buys (AURC, excess AURC, an operating point)."""
    with _lock:
        entry = _entry(workspace_id)
        state = entry.loaded.state
        names = [_experiment_name(entry, n) for n in experiments] if experiments else list(state.experiments)
        rows = []
        for name in names:
            c = _calibration_of(entry, name)
            rel, curve = c["reliability"], c["curve"]
            row = {"experiment": name, "n": c["index"].n, "accuracy": c["accuracy"],
                   "mean_confidence": c["mean_confidence"],
                   "confidence_gap": c["mean_confidence"] - c["accuracy"],
                   "ece": None if rel.disabled_reason else rel.ece,
                   "mce": None if rel.disabled_reason else rel.mce,
                   "aurc": curve.aurc, "excess_aurc": curve.aurc - curve.optimal_aurc}
            if rel.disabled_reason:
                row["probabilistic_metrics_disabled"] = rel.disabled_reason
            if target_accuracy is not None:
                row["for_target_accuracy"] = _point(cal.threshold_for_target(c["index"], float(target_accuracy)))
            if threshold is not None:
                row["at_threshold"] = _point(cal.threshold_point(c["index"], float(threshold)))
            if fit_temperature:
                experiment = state.experiments[name]
                merged = state.merged(name)
                fit = cal.fit_temperature(merged[experiment.score_cols].to_numpy(),
                                          merged[state.dataset.columns.label_col].to_numpy(),
                                          experiment.class_labels[: len(experiment.score_cols)],
                                          y_pred=merged["predicted_label"].to_numpy())
                row["temperature"] = ({"disabled": fit.disabled_reason} if fit.disabled_reason else
                                      {"temperature": fit.temperature, "nll_before": fit.nll_before,
                                       "nll_after": fit.nll_after, "ece_before": fit.ece_before,
                                       "ece_after": fit.ece_after,
                                       "note": "Fitted on the rows it is measured on: optimistic. A "
                                               "temperature for production belongs on a held-out split."})
            rows.append(row)
        out = _context(entry, "calibration")
        out["experiments"] = rows
        return _clean(out)


def get_confidence(workspace_id: str, experiment: str) -> dict:
    """Margin (top score minus the second) and entropy, for the right and
    the wrong predictions apart -- whether the model hesitates where it errs."""
    with _lock:
        entry = _entry(workspace_id)
        name = _experiment_name(entry, experiment)
        state = entry.loaded.state
        exp = state.experiments[name]
        data = conf.compute_confidence(state.merged(name), exp.score_cols, exp.class_labels)
        df = data.df
        correct = df["correct"].astype(bool)

        def summary(column: str) -> dict:
            out = {}
            for key, part in (("correct", df.loc[correct, column]), ("wrong", df.loc[~correct, column])):
                values = part.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                out[key] = ({"n": int(values.size), "mean": values.mean(), "median": float(np.median(values)),
                             "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}
                            if values.size else {"n": 0})
            return out

        out = _context(entry, "confidence")
        out.update(experiment=name, softmax_applied=bool(data.softmax_applied),
                   margin=summary(conf.MARGIN_COL), normalized_entropy=summary(conf.NORMALIZED_ENTROPY_COL))
        return _clean(out)


# ---- slices ----------------------------------------------------------------

SLICE_METRICS = (sl.METRIC_ACCURACY, sl.METRIC_MACRO_F1, sl.METRIC_MEAN_CONFIDENCE, sl.METRIC_RECALL,
                 sl.METRIC_PRECISION, sl.METRIC_F1, sl.METRIC_FALSE_ALARM, sl.METRIC_MISS_RATE,
                 sl.METRIC_SUPPORT, sl.METRIC_PREDICTED)
SLICE_BINNINGS = (sl.BINNING_AUTO, sl.BINNING_VALUES, sl.BINNING_QUANTILE)
SLICE_SORTS = (sl.SORT_WORST, sl.SORT_VALUE, sl.SORT_SIZE)
# The metrics that are about one class, and need it named.
CLASS_SLICE_METRICS = (sl.METRIC_RECALL, sl.METRIC_PRECISION, sl.METRIC_F1, sl.METRIC_FALSE_ALARM,
                       sl.METRIC_MISS_RATE, sl.METRIC_SUPPORT, sl.METRIC_PREDICTED)


def compute_slices(state, column: str, names: list, binning: str = sl.BINNING_AUTO,
                   bins: int = sl.DEFAULT_BINS) -> tuple:
    """(spec, {experiment: SliceReport}) -- the Metrics tab's breakdown."""
    if binning not in SLICE_BINNINGS:
        raise ValueError(f"binning must be one of {', '.join(SLICE_BINNINGS)}; got {binning!r}.")
    if column not in state.dataset.df.columns:
        raise ValueError(f"No column {column!r} in the dataset.")
    spec = sl.build_spec(state.dataset.df[column], bins=max(2, min(int(bins), sl.MAX_BINS)), binning=binning)
    labels = [str(label) for label in state.dataset.labels()]
    reports = {n: sl.slice_report(n, state.merged(n), state.dataset.columns.label_col, spec, class_labels=labels)
               for n in names}
    return spec, reports


def check_slice_metric(metric: str, class_label) -> None:
    if metric not in SLICE_METRICS:
        raise ValueError(f"metric must be one of {', '.join(SLICE_METRICS)}; got {metric!r}.")
    if metric in CLASS_SLICE_METRICS and class_label is None:
        raise ValueError(f"metric {metric!r} is about one class: name it with class_label.")


def get_slices(workspace_id: str, column: str, experiments: list | None = None, metric: str = sl.METRIC_ACCURACY,
               class_label: str | None = None, binning: str = sl.BINNING_AUTO, bins: int = sl.DEFAULT_BINS,
               min_rows: int = sl.DEFAULT_MIN_ROWS, sort: str = sl.SORT_WORST, top: int = DEFAULT_TOP) -> dict:
    """Every experiment scored again on each slice of a dataset column --
    "94%, except 61% on this sensor". Each slice carries its row count and,
    for accuracy, its 95% Wilson interval; slices under `min_rows` are
    flagged and never ranked first."""
    check_slice_metric(metric, class_label)
    if sort not in SLICE_SORTS:
        raise ValueError(f"sort must be one of {', '.join(SLICE_SORTS)}; got {sort!r}.")
    with _lock:
        entry = _entry(workspace_id)
        state = entry.loaded.state
        names = [_experiment_name(entry, n) for n in experiments] if experiments else list(state.experiments)
        if not names:
            raise ValueError("No experiment is loaded in this workspace.")
        spec, reports = compute_slices(state, column, names, binning, bins)
        first = reports[names[0]]
        ordered = [row.value for row in sl.sort_rows(first.rows, sort, int(min_rows), metric=metric,
                                                      class_label=class_label)]
        rows_by_name = {n: {row.value: row for row in r.rows} for n, r in reports.items()}
        slices = []
        for value in ordered[:_top(top)]:
            item = {"slice": value, "rows": rows_by_name[names[0]][value].n,
                    "small": rows_by_name[names[0]][value].is_small(int(min_rows))}
            for n in names:
                row = rows_by_name[n].get(value)
                if row is None:
                    continue
                low, high = sl.metric_interval(row, metric, class_label)
                item[n] = {"value": sl.metric_value(row, metric, class_label),
                           "interval": [low, high] if low == low else None,
                           "errors": row.n_errors, "n_classes": row.n_classes}
            slices.append(item)
        out = _context(entry, "metrics")
        out.update(column=column, metric=metric, class_label=class_label, binning=binning, sort=sort,
                   min_rows=int(min_rows), n_slices=len(ordered),
                   overall={n: sl.metric_value(r.overall, metric, class_label) for n, r in reports.items()
                            if r.overall is not None},
                   slices=slices)
        if spec.note:
            out["note"] = spec.note
        return _clean(out)


# ---- crops worth a look ------------------------------------------------------

CANDIDATE_SOURCES = ("margin", "confident_errors", "all_errors", "consensus")


def find_cleaning_candidates(workspace_id: str, source: str = "confident_errors", experiment: str = "",
                             limit: int = DEFAULT_TOP, min_margin: float = 0.0, min_agreeing: int = 2,
                             class_label: str | None = None, match: str = cand.MATCH_LABEL) -> dict:
    """The Cleaning tab's candidate search: crops whose label is worth a
    second look, most suspicious first, each with what the evidence
    suggests instead."""
    if source not in CANDIDATE_SOURCES:
        raise ValueError(f"source must be one of {', '.join(CANDIDATE_SOURCES)}; got {source!r}.")
    if match not in (cand.MATCH_LABEL, cand.MATCH_SUGGESTION):
        raise ValueError(f"match must be {cand.MATCH_LABEL!r} or {cand.MATCH_SUGGESTION!r}.")
    with _lock:
        entry = _entry(workspace_id)
        state = entry.loaded.state
        if state.dataset is None:
            raise ValueError("This workspace has no dataset loaded.")
        cols = state.dataset.columns
        class_filter = cand.ClassFilter(class_label, match) if class_label is not None else None
        limit = _top(limit)
        if source == "consensus":
            if len(state.experiments) < 2:
                raise ValueError("Consensus needs at least two loaded experiments.")
            found = cand.by_consensus(state.dataset.df, list(state.experiments.values()), cols.uuid_col,
                                      cols.label_col, min_agreeing=int(min_agreeing), limit=limit,
                                      class_filter=class_filter)
            name = ""
        else:
            if not experiment:
                raise ValueError(f"source {source!r} needs an experiment.")
            name = _experiment_name(entry, experiment)
            exp = state.experiments[name]
            if source == "all_errors":
                found = cand.all_errors(state.dataset.df, exp, cols.uuid_col, cols.label_col,
                                        class_filter=class_filter)
            else:
                found = cand.by_margin(state.dataset.df, exp, cols.uuid_col, cols.label_col,
                                       min_margin=float(min_margin) if source == "margin" else -np.inf,
                                       limit=limit, confident_errors_only=source == "confident_errors",
                                       class_filter=class_filter)
        workspace = entry.loaded.workspace.id
        rows = [{"uuid": c.uuid, "label": c.label, "suggestion": c.suggestion or None, "score": c.score,
                 "detail": dict(c.detail), "open_url": links.link(workspace, crop=c.uuid, experiment=name)}
                for c in found.candidates[:limit]]
        out = _context(entry, "cleaning")
        out.update(source=source, name=found.name, n_found=len(found), candidates=rows,
                   review_source=found.source)
        if found.note:
            out["note"] = found.note
        if state.split_kind == "train":
            out["note"] = (out.get("note", "") + " This pool is marked train: its predictions are "
                           "in-sample.").strip()
        return _clean(out)


def find_similar(workspace_id: str, experiment: str, uuid: str, k: int = DEFAULT_TOP) -> dict:
    """The crops nearest to one crop in an experiment's embedding space, and
    how many share its label: a crop whose neighbours are mostly another
    class is mislabelled or genuinely ambiguous."""
    with _lock:
        entry = _entry(workspace_id)
        name = _experiment_name(entry, experiment)
        ensure_embeddings(entry, name)
        context = build_analysis_context(entry.loaded.state, name)
        if context is None:
            raise ValueError(f"Experiment {name!r} has no embeddings attached.")
        found = nearest_neighbours(context["embeddings"], context["uuids"], str(uuid), k=_top(k),
                                   labels=context["labels"])
        if not found and str(uuid) not in set(context["uuids"]):
            raise ValueError(f"No crop {uuid!r} with an embedding in experiment {name!r}.")
        own = context["labels"][list(context["uuids"]).index(str(uuid))]
        workspace = entry.loaded.workspace.id
        neighbours = [{"uuid": n.uuid, "label": n.label, "similarity": n.similarity,
                       "open_url": links.link(workspace, crop=n.uuid, experiment=name)} for n in found]
        same = sum(1 for n in found if str(n.label) == str(own))
        out = _context(entry, "embeddings")
        out.update(experiment=name, uuid=str(uuid), label=own, k=len(found), same_label=same,
                   neighbours=neighbours)
        return _clean(out)


def ensure_embeddings(entry: _Entry, name: str, prototypes: bool = False) -> None:
    """Attach the workspace's embeddings (and prototypes) sidecars to a
    loaded experiment the first time something needs them -- they can be
    gigabytes, and most questions never do."""
    experiment = next((e for e in entry.loaded.workspace.mount.experiments if e.name == name), None)
    exp = entry.loaded.state.experiments[name]
    if not exp.has_embeddings() and experiment is not None and experiment.embeddings_path \
            and Path(experiment.embeddings_path).is_file():
        exp.load_embeddings(experiment.embeddings_path)
    if prototypes and exp.model_prototypes is None and experiment is not None and experiment.prototypes_path \
            and Path(experiment.prototypes_path).is_file():
        exp.load_model_prototypes(experiment.prototypes_path)


# ---- review ------------------------------------------------------------------

def get_review(workspace_id: str, version: str = "") -> dict:
    """The review store of the workspace's dataset: its versions (which one
    is active, which are frozen), and what a version decided."""
    with _lock:
        entry = _entry(workspace_id)
        store = entry.loaded.state.review
        if store is None:
            raise ValueError("This workspace has no dataset loaded, so no review store.")
        chosen = version or store.active_version_id
        if chosen and chosen not in store.versions:
            raise ValueError(f"No review version {chosen!r}. Versions: {', '.join(store.versions) or 'none'}.")
        versions = [{"id": v.version_id, "parent": v.parent or None, "note": v.note, "author": v.author,
                     "created_at": v.created_at, "frozen": store.is_frozen(v.version_id),
                     "own_decisions": store.n_decisions(v.version_id)} for v in store.versions.values()]
        out = _context(entry, "cleaning")
        out.update(active_version=store.active_version_id, versions=versions)
        if chosen:
            out.update(version=chosen, counts=store.counts(chosen), reasons=store.reason_counts(chosen),
                       batches=[{"batch_id": b, "status": status, "crops": n, "at": at}
                                for b, status, n, at in store.batches(chosen)[:20]])
        return _clean(out)


def get_snapshot(snapshot_id: str) -> dict:
    """One saved view: what it is called, its workspace, the view state it
    restores (tab, filters, settings of each tab) and the link that opens it."""
    found = lib.find_snapshot(snapshot_id)
    if found is None:
        raise ValueError(f"No snapshot {snapshot_id!r} in the library.")
    workspace, snapshot = found
    return _clean({"id": snapshot.id, "name": snapshot.name, "description": snapshot.description,
                   "tags": list(snapshot.tags), "workspace_id": workspace.id, "workspace": workspace.label(),
                   "created_at": snapshot.created_at, "updated_at": snapshot.updated_at,
                   "open_url": links.link(snapshot_id=snapshot.id), "view_state": snapshot.view_state})


def get_notes(workspace_id: str) -> dict:
    """The workspace's notes (core/notes.py), as Markdown."""
    from lookin.core import notes as nt

    w = find_workspace(workspace_id)
    text = nt.read(w)
    return _clean({
        "workspace_id": w.id, "workspace": w.label(), "open_url": links.link(w.id, tab="notes"),
        "exists": text is not None, "notes": text or "",
        "warnings": [] if text is not None else ["This workspace has no notes yet: write_notes starts them, "
                                                  "or opening its Notes tab writes the title and a recap."],
    })
