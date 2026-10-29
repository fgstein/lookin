"""What an agent can have Lookin produce: charts, mosaics of crops, reports.

The third part of the MCP endpoint (see lookin/mcp_server.py), beside
core/agent_tools.py (reading) and core/agent_actions.py (changing). A chart
is drawn by core/charts.py -- the very function the tab draws it with -- and
a mosaic by core/mosaic.py through the display LUT the Gallery would use.

Pictures come back as PNG bytes for the endpoint to send inline. Files
(reports, exports, and any picture asked to be saved) are written on the
machine running Lookin: into the folder asked for, else into the exports
folder, which `lookin web` serves at /exports/<name> -- so an agent on
another machine can hand its user a link rather than a path they cannot
open.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from lookin.core import agent_tools as tools
from lookin.core import charts
from lookin.core import class_focus as cf
from lookin.core import links
from lookin.core import mosaic
from lookin.core.chart_style import HEAVY_WORK_LOCK, color_for_experiment
from lookin.core.images import LUT_NONE, LUT_OPTIONS, quarter_turns_for_roll
from lookin.core.metrics import MACRO_AVERAGE, compute_curves

EXPORTS_ROUTE = "/exports"

CHART_KINDS = (
    "per_class", "roc_pr", "confusion_matrix", "confusion_overview", "reliability", "confidence_histogram",
    "risk_coverage", "column", "slices", "projection",
)
PER_CLASS_METRICS = ("f1", "precision", "recall", "specificity", "roc_auc", "average_precision", "support")
PROJECTION_METHODS = ("pca", "umap", "tsne", "umap_circle")
MAX_PROJECTION_POINTS = 20_000
DEFAULT_PROJECTION_POINTS = 3_000
CAPTIONS = ("none", "label", "prediction", "uuid")


# ---- files -------------------------------------------------------------------

def exports_dir() -> Path:
    from lookin.core.mlflow_cache import user_cache_dir

    path = user_cache_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "lookin"


def destination(dest: str, default_name: str, beside: Path | None = None) -> Path:
    """Where to write a file: `dest` if it names a file (refused when one
    is already there -- an export never overwrites), inside `dest` if it is
    a folder, else in `beside` or the exports folder; a default name that
    is taken gets a timestamp rather than replacing anything."""
    if dest:
        path = Path(dest).expanduser()
        if path.is_dir():
            folder, name = path, default_name
        else:
            if path.exists():
                raise ValueError(f"{path} already exists -- give another name; an export never overwrites a file.")
            if not path.parent.is_dir():
                raise ValueError(f"No folder {path.parent} on the machine running Lookin.")
            return path
    else:
        folder, name = (beside if beside is not None else exports_dir()), default_name
    path = folder / _safe_name(name)
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d-%H%M%S}{path.suffix}")
    return path


def file_answer(path: Path, **extra) -> dict:
    """Where a written file is, and the link to download it when it sits
    in the exports folder."""
    path = Path(path)
    out = {"path": str(path), "size": path.stat().st_size if path.exists() else None}
    try:
        if path.resolve().parent == exports_dir().resolve():
            out["url"] = f"{links.base_url()}{EXPORTS_ROUTE}/{path.name}"
    except OSError:
        pass
    out.update({k: v for k, v in extra.items() if v is not None})
    return tools._clean(out)


def exported_file(name: str) -> Path | None:
    """A file of the exports folder by its bare name, for the web route --
    None for anything else (a path, a parent reference, a missing file)."""
    if not name or name != os.path.basename(name) or name.startswith("."):
        return None
    path = exports_dir() / name
    return path if path.is_file() else None


def _save_png(png: bytes, save_to: str, default_name: str) -> dict:
    path = destination(save_to, default_name)
    path.write_bytes(png)
    return file_answer(path)


# ---- charts ------------------------------------------------------------------

def _colors(state, names) -> dict:
    return {n: color_for_experiment(n, state.color_slots) for n in names}


def _names(entry, experiments) -> list:
    state = entry.loaded.state
    names = [tools._experiment_name(entry, n) for n in experiments] if experiments else list(state.experiments)
    if not names:
        raise ValueError("No experiment is loaded in this workspace.")
    return names


def _focus(metrics_list) -> list | None:
    """The classes a per-class chart shows once there are too many: the
    most confused, as the Metrics tab picks by default."""
    confusions = [m.confusion for m in metrics_list]
    if len(cf.all_labels(confusions)) <= cf.FOCUS_MAX:
        return None
    return cf.rank_classes(cf.FOCUS_MOST_CONFUSED, confusions, [m.per_class for m in metrics_list])


def plot_chart(workspace_id: str, chart: str, experiments: list | None = None, metric: str = "",
               target: str = "", column: str = "", classes: list | None = None, class_label: str | None = None,
               binning: str = "auto", bins: int = 10, min_rows: int = 30, method: str = "pca",
               max_points: int = DEFAULT_PROJECTION_POINTS, color_by: str = "", save_to: str = "") -> tuple:
    """(answer, png): one chart of a workspace, drawn as its tab draws it."""
    if chart not in CHART_KINDS:
        raise ValueError(f"chart must be one of {', '.join(CHART_KINDS)}; got {chart!r}.")
    with tools._lock, HEAVY_WORK_LOCK:
        entry = tools._entry(workspace_id)
        state = entry.loaded.state
        if state.dataset is None:
            raise ValueError("This workspace has no dataset loaded.")
        notes, tab = [], "metrics"
        if chart == "column":
            if column not in state.dataset.df.columns:
                raise ValueError(f"No column {column!r} in the dataset.")
            series = state.dataset.df[column]
            numeric = column in state.dataset.column_kinds().numeric
            fig, notes = charts.column_chart(column, series.dropna(), numeric)
            title, tab = f"Column {column}", "statistics"
        elif chart == "projection":
            fig, notes, title = _projection(entry, experiments, method, max_points, color_by)
            tab = "embeddings"
        else:
            names = _names(entry, experiments)
            colors = _colors(state, names)
            fig, title, tab, notes = _metric_chart(entry, chart, names, colors, metric, target, column, classes,
                                                   class_label, binning, bins, min_rows)
        png = charts.figure_png(fig)
    out = tools._context(entry, tab)
    out.update(chart=chart, title=title)
    if notes:
        out["notes"] = notes
    if save_to:
        out["saved"] = _save_png(png, save_to, f"{chart}.png")
    return tools._clean(out), png


def _metric_chart(entry, chart, names, colors, metric, target, column, classes, class_label, binning, bins,
                  min_rows) -> tuple:
    state = entry.loaded.state
    label_col = state.dataset.columns.label_col
    notes = []
    if chart == "per_class":
        metric = metric or "f1"
        if metric not in PER_CLASS_METRICS:
            raise ValueError(f"metric must be one of {', '.join(PER_CLASS_METRICS)}; got {metric!r}.")
        metrics = {n: tools._metrics(entry, n) for n in names}
        per_class = {n: m.per_class.set_index("label") for n, m in metrics.items()}
        labels = sorted(set().union(*[set(pc.index) for pc in per_class.values()]), key=str)
        focus = [c for c in classes] if classes else _focus(list(metrics.values()))
        if focus is not None:
            notes.append(f"{len(focus)} of {len(labels):,} classes shown"
                         + ("." if classes else " -- the most confused; name others with `classes`."))
            labels = [c for c in focus if c in set(labels)]
        fig, uneven = charts.per_class_bars(per_class, metric, labels, colors)
        notes.append("[n] under each class is its support.")
        return fig, f"Per-class {metric}", "metrics", notes
    if chart == "roc_pr":
        target = target or MACRO_AVERAGE
        curves = {}
        for n in names:
            exp = state.experiments[n]
            curve = compute_curves(state.merged(n), label_col, exp.score_cols, exp.class_labels, target)
            if curve.disabled_reason:
                notes.append(f"{n}: not drawn ({curve.disabled_reason}).")
            else:
                curves[n] = curve
        if not curves:
            raise ValueError(" ".join(notes) or "No curve to draw.")
        return charts.roc_pr_curves(curves, target, colors), f"ROC and PR curves -- {target}", "metrics", notes
    if chart in ("confusion_matrix", "confusion_overview"):
        if len(names) != 1:
            raise ValueError("A confusion matrix is drawn for one experiment: name exactly one.")
        m = tools._metrics(entry, names[0])
        if m.confusion.empty:
            raise ValueError(f"{names[0]} has no row matching the dataset, so no confusion matrix.")
        title = f"{names[0]} -- accuracy {m.accuracy:.4f}"
        if chart == "confusion_overview":
            block = max(1, -(-len(m.confusion) // cf.ZOOM_BLOCKS))
            return cf.overview_figure(m.confusion, list(classes or []), title, block=block), title, "metrics", notes
        focus = list(classes) if classes else _focus([m])
        if focus is not None:
            notes.append(f"{len(focus)} of {len(m.confusion):,} classes; the last column counts crops "
                         f"predicted as any other class.")
        return charts.confusion_matrix(m.confusion, title, focus=focus), title, "metrics", notes
    if chart in ("reliability", "confidence_histogram"):
        if len(names) != 1:
            raise ValueError(f"A {chart} chart is drawn for one experiment: name exactly one.")
        c = tools._calibration_of(entry, names[0])
        if chart == "reliability":
            fig = charts.reliability_diagram(names[0], c["reliability"], colors[names[0]])
            if fig is None:
                raise ValueError(f"No reliability diagram: {c['reliability'].disabled_reason}")
            return fig, f"Reliability -- {names[0]}", "calibration", notes
        return (charts.confidence_split_histogram(names[0], c["histogram"], colors[names[0]]),
                f"Confidence of right vs. wrong -- {names[0]}", "calibration", notes)
    if chart == "risk_coverage":
        curves = {n: tools._calibration_of(entry, n)["curve"] for n in names}
        return charts.risk_coverage(curves, colors), "Risk-coverage", "calibration", notes
    # slices
    if not column:
        raise ValueError("A slices chart needs the dataset column to slice by.")
    slice_metric = metric or "accuracy"
    tools.check_slice_metric(slice_metric, class_label)
    spec, reports = tools.compute_slices(state, column, names, binning, bins)
    first = reports[names[0]]
    from lookin.core import slices as sl

    ordered = [row.value for row in sl.sort_rows(first.rows, sl.SORT_WORST, int(min_rows), metric=slice_metric,
                                                  class_label=class_label)]
    if not ordered:
        raise ValueError(f"No rows to slice by {column!r}.")
    label = slice_metric.replace("_", " ") + (f" ({class_label})" if class_label is not None else "")
    fig, notes = charts.slice_chart(ordered, reports, slice_metric, label, column, colors, class_label=class_label)
    return fig, f"{label} by {column}", "metrics", notes


def _projection(entry, experiments, method: str, max_points: int, color_by: str) -> tuple:
    from lookin.core.embedding_analysis import fit_projection
    from lookin.core.embedding_context import build_analysis_context

    if method not in PROJECTION_METHODS:
        raise ValueError(f"method must be one of {', '.join(PROJECTION_METHODS)}; got {method!r}.")
    names = _names(entry, experiments)
    if len(names) != 1:
        raise ValueError("A projection is drawn for one experiment: name exactly one.")
    name = names[0]
    tools.ensure_embeddings(entry, name)
    context = build_analysis_context(entry.loaded.state, name)
    if context is None:
        raise ValueError(f"Experiment {name!r} has no embeddings attached.")
    n = len(context["embeddings"])
    cap = max(100, min(int(max_points or DEFAULT_PROJECTION_POINTS), MAX_PROJECTION_POINTS))
    rows = np.arange(n) if n <= cap else np.sort(np.random.default_rng(0).choice(n, cap, replace=False))
    if color_by and color_by not in context["merged"].columns:
        raise ValueError(f"No column {color_by!r} to colour by.")
    groups = (context["merged"][color_by].to_numpy()[rows] if color_by else context["labels"][rows])
    fit = fit_projection(context["embeddings"][rows], method=method, seed=0)
    title = f"{name} -- {method.upper()} of {len(rows):,} crops, coloured by {color_by or 'class'}"
    notes = [f"A random sample of {len(rows):,} of {n:,} crops (seed 0)."] if n > cap else []
    return charts.projection_scatter(fit.points, groups, title), notes, title


# ---- mosaics -----------------------------------------------------------------

def make_mosaic(workspace_id: str, experiment: str = "", uuids: list | None = None, true_label: str | None = None,
                predicted_label: str | None = None, errors_only: bool = False, where: str = "",
                sort: str = tools.SORT_MOST_CONFIDENT, limit: int = 48, columns: int = 0,
                tile_size: int = mosaic.DEFAULT_TILE, caption: str = "label", lut: str = "",
                roll_column: str = "", frames: bool = True, title: str = "", save_to: str = "") -> tuple:
    """(answer, png): crops of a workspace as one picture -- named by uuid,
    or picked like get_examples picks them. `lut` and `roll_column` are the
    display settings of the top bar (core/images.py)."""
    if caption not in CAPTIONS:
        raise ValueError(f"caption must be one of {', '.join(CAPTIONS)}; got {caption!r}.")
    if lut and lut not in {k for k, _ in LUT_OPTIONS}:
        raise ValueError(f"lut must be one of {', '.join(k for k, _ in LUT_OPTIONS)}; got {lut!r}.")
    limit = max(1, min(int(limit or 48), mosaic.MAX_TILES))
    with tools._lock:
        entry = tools._entry(workspace_id)
        state = entry.loaded.state
        if state.dataset is None:
            raise ValueError("This workspace has no dataset loaded.")
        name = tools._experiment_name(entry, experiment) if experiment else ""
        cols = state.dataset.columns
        if roll_column and roll_column not in state.dataset.df.columns:
            raise ValueError(f"No column {roll_column!r} to read the roll from.")
        if uuids:
            frame = state.merged(name) if name else state.dataset.df
            wanted = [str(u) for u in uuids][:limit]
            by_uuid = frame.assign(_u=frame[cols.uuid_col].astype(str)).drop_duplicates("_u").set_index("_u")
            missing = [u for u in wanted if u not in by_uuid.index]
            rows = by_uuid.loc[[u for u in wanted if u in by_uuid.index]].reset_index(drop=True)
        else:
            missing = []
            rows = tools.select_rows(state, name, true_label, predicted_label, errors_only and bool(name),
                                     where, sort).head(limit)
        if rows.empty:
            raise ValueError("No crop matches.")
        tiles = []
        for _, r in rows.iterrows():
            text = ""
            if caption == "label":
                text = str(r[cols.label_col])
            elif caption == "prediction":
                text = (f"{r[cols.label_col]} -> {r['predicted_label']} ({r['predicted_score']:.2f})" if name
                        else str(r[cols.label_col]))
            elif caption == "uuid" and cols.uuid_col:
                text = str(r[cols.uuid_col])
            tiles.append(mosaic.Tile(
                path=str(state.dataset.resolve_crop_path(r[cols.crop_col])), caption=text,
                quarter_turns=quarter_turns_for_roll(r[roll_column]) if roll_column else 0,
                correct=bool(r["correct"]) if (name and frames) else None))
        png, unreadable = mosaic.render(tiles, columns=columns, tile_size=tile_size, lut=lut or LUT_NONE,
                                        title=title)
        out = tools._context(entry, "gallery")
        out.update(experiment=name or None, n_crops=len(tiles),
                   uuids=[str(u) for u in rows[cols.uuid_col]] if cols.uuid_col else None)
        if missing:
            out["not_found"] = missing
        if unreadable:
            out["unreadable"] = unreadable
        if name and frames:
            out["frames"] = "green: predicted right, red: predicted wrong"
    if save_to:
        out["saved"] = _save_png(png, save_to, "mosaic.png")
    return tools._clean(out), png


# ---- reports -----------------------------------------------------------------

REPORT_FORMATS = ("html", "pptx", "md")


def generate_report(workspace_id: str, experiments: list | None = None, format: str = "html",
                    baseline: str = "", candidate: str = "", per_class: bool = True, confusion: bool = True,
                    dest: str = "", snapshot: bool = True, per_class_charts: bool = True,
                    notes: bool = True) -> dict:
    """The Metrics tab's report: provenance, the workspace's notes,
    summary, per-class charts and tables, confusion matrices and, with a
    baseline and a candidate, the head to head -- one self-contained HTML
    file, a PowerPoint deck or a Markdown document."""
    from lookin.core import library as lib
    from lookin.core import notes as nt
    from lookin.core import view_state as vs
    from lookin.core.report import Report
    from lookin.core.report_builder import build_report, default_report_name
    from lookin.core.report_md import MarkdownReport
    from lookin.core.report_pptx import PptxReport

    if format not in REPORT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(REPORT_FORMATS)}; got {format!r}.")
    with tools._lock, HEAVY_WORK_LOCK:
        entry = tools._entry(workspace_id)
        state = entry.loaded.state
        names = _names(entry, experiments)
        pair = None
        if baseline or candidate:
            if not (baseline and candidate):
                raise ValueError("A comparison needs both baseline and candidate.")
            pair = (tools._experiment_name(entry, baseline), tools._experiment_name(entry, candidate))
            names = list(dict.fromkeys(names + list(pair)))
        snapshot_id = ""
        if snapshot:
            # As the tab does: the report's provenance names a snapshot of
            # the view it was made from, so its figures can be reopened.
            saved = lib.create_snapshot(entry.loaded.workspace,
                                        {vs.APP: {vs.TAB: vs.SNAPSHOT_TABS.index("metrics")}, vs.VIEWS: {}},
                                        name="Report (MCP)", description="Saved when an agent exported a report.")
            snapshot_id = saved.id if saved else ""
        writer = {"html": Report, "pptx": PptxReport, "md": MarkdownReport}[format]
        report = build_report(state, names, include_per_class=per_class, include_confusion=confusion,
                              comparison_pair=pair, snapshot_id=snapshot_id, report_cls=writer,
                              include_per_class_charts=per_class_charts,
                              notes=(nt.read(entry.loaded.workspace) or "") if notes else "")
        path = destination(dest, default_report_name(state, extension=format))
        report.write(path)
    out = file_answer(path, experiments=names, format=format)
    if snapshot_id:
        out["snapshot"] = {"id": snapshot_id, "open_url": links.link(snapshot_id=snapshot_id)}
    return out
