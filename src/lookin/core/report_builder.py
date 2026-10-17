"""Assembles the HTML report from the loaded state.

Split from core/report.py -- that one is a small, dumb HTML writer with no
knowledge of this app; this one knows what a Lookin session contains and
decides what is worth writing down. Neither imports Flet, so the whole
report can be produced and tested without a UI.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from lookin.core import class_focus as cf
from lookin.core import comparison as cmp
# Imported for its side effect: it applies the app's chart style, so the
# confusion matrices in an exported report come out in the same typeface as
# the tab they were exported from. No grid is put on them -- a heatmap has a
# cell grid of its own, and rules drawn through it would be noise.
from lookin.core import chart_style  # noqa: F401
from lookin.core.metrics import PER_CLASS_COLUMNS, compute_metrics
from lookin.core.report import Report
from lookin.core.chart_style import subplots

# Rows of the most-confused-pairs table in a many-class report.
REPORT_PAIRS = 25

METRIC_ROWS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("mcc", "MCC"),
    ("cohen_kappa", "Cohen's kappa"),
    ("macro_roc_auc", "Macro ROC-AUC"),
    ("macro_avg_precision", "Macro avg. precision"),
    ("log_loss", "Log loss"),
    ("brier_score", "Brier score"),
    ("ece", "ECE"),
)


def build_report(state, experiment_names: list, include_per_class: bool = True,
                 include_confusion: bool = True, comparison_pair=None,
                 snapshot_id: str = "", report_cls: type = Report,
                 focus: list | None = None) -> Report:
    """One report over the given experiments.

    `comparison_pair` is an optional (baseline, candidate) -- when given, the
    Compare tab's head-to-head goes in too, because "which one shipped and
    why" is exactly what a report is read for later.

    `snapshot_id` is the snapshot this report was produced from (see
    core/library.py). It is written into the provenance so that the figures
    below can be *reopened*, not just read: a number six months old is worth
    much more when the panel that produced it is one paste away.

    `focus` is the classes the per-class sections show once there are more
    than class_focus.FOCUS_MAX -- the Metrics tab passes the ones picked
    there, so the report shows what was on screen. Left out, the most
    confused classes are used. A table or figure per class stops being
    readable (or, for a slide deck, printable) at a few thousand classes;
    the report then carries the focus classes, the most confused pairs and
    the whole matrix as one picture instead.

    `report_cls` picks the writer: `Report` (the default) for HTML, or
    `PptxReport` (core/report_pptx.py) for PowerPoint -- everything below
    only calls the writer's content methods, so it does not need to know
    or care which one it got.
    """
    if state.dataset is None:
        raise ValueError("No dataset loaded.")
    if not experiment_names:
        raise ValueError("Pick at least one experiment to report on.")

    dataset = state.dataset
    cols = dataset.columns
    report = report_cls(
        title="Lookin report",
        subtitle=f"{Path(dataset.csv_path).name if dataset.csv_path else 'dataset'} "
                 f"-- {len(dataset.df):,} rows, {len(dataset.labels())} classes",
    )

    _add_provenance(report, state, snapshot_id)
    metrics = {name: _metrics_for(state, name) for name in experiment_names}
    focus = _report_focus(metrics, focus)
    _add_summary(report, metrics)
    if include_per_class:
        _add_per_class(report, metrics, focus)
    if include_confusion:
        _add_confusion(report, metrics, focus)
    if comparison_pair:
        _add_comparison(report, state, *comparison_pair, focus=focus)
    return report


def _report_focus(metrics: dict, focus: list | None) -> list | None:
    """The classes per-class sections are cut to, or None for all of them."""
    confusions = [m.confusion for m in metrics.values()]
    labels = cf.all_labels(confusions)
    if len(labels) <= cf.FOCUS_MAX:
        return None
    present = set(labels)
    focus = [c for c in (focus or []) if c in present]
    return focus or cf.rank_classes(cf.FOCUS_MOST_CONFUSED, confusions,
                                    [m.per_class for m in metrics.values()])


def _focus_note(focus: list, n_all: int) -> str:
    return (f"{len(focus)} of {n_all:,} classes: the ones picked under 'Classes shown' in the Metrics "
            f"tab when the report was exported ({', '.join(str(c) for c in focus)}).")


def _add_provenance(report: Report, state, snapshot_id: str = "") -> None:
    """What this report is *about*, first and in full.

    A number without its provenance is unusable six months later: which
    file, which split, which review version, and -- the one that silently
    changes every figure below -- whether the reviewed labels were applied.
    """
    dataset = state.dataset
    report.add_meta("Dataset CSV", dataset.csv_path or "(unsaved)")
    report.add_meta("Rows loaded", f"{len(dataset.df):,}")
    if dataset.full_df is not None and dataset.full_df is not dataset.df:
        from lookin.core.dataset_filter import describe
        report.add_meta("Dataset filter", f"{len(dataset.df):,} of {len(dataset.full_df):,} rows kept: "
                        + "; ".join(describe(getattr(state, "dataset_filter", None) or {})))
    report.add_meta("Split", state.split_kind)
    if dataset.row_filter:
        report.add_meta("Row filter", f"{dataset.row_filter[0]} = {dataset.row_filter[1]}")
    if dataset.image_root:
        report.add_meta("Image root", dataset.image_root)
    report.add_meta("Experiments loaded", ", ".join(state.experiments) or "(none)")
    if snapshot_id:
        # The whole point of the id: paste it into Lookin's library and the
        # panels behind every figure below come back as they were.
        report.add_meta("Snapshot", f"{snapshot_id} -- reopen this view from Lookin's Workspaces window")

    store = getattr(state, "review", None)
    if store is not None and store.versions:
        version_id = store.active_version_id or "(none)"
        report.add_meta("Review version", version_id)
        if version_id in store.versions:
            counts = store.counts(version_id)
            report.add_meta(
                "Decisions at that version",
                ", ".join(f"{k}: {v:,}" for k, v in counts.items() if v),
            )
    report.add_meta(
        "Reviewed labels applied",
        "yes -- every figure below is on the cleaned pool" if state.use_reviewed_labels else "no",
    )
    report.add_meta("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def _metrics_for(state, name):
    experiment = state.experiments[name]
    return compute_metrics(name, state.merged(name), state.dataset.columns.label_col,
                           experiment.score_cols, experiment.class_labels)


def _add_summary(report: Report, metrics: dict) -> None:
    section = report.section("Summary")
    names = list(metrics)

    headers = ["Metric"] + list(names)
    rows = [[label] + [getattr(metrics[n], attr) for n in names] for attr, label in METRIC_ROWS]
    rows.insert(0, ["Samples"] + [metrics[n].n_samples for n in names])
    section.blocks.append(report.table(headers, rows))

    warnings = []
    for name in names:
        diagnostics = metrics[name].score_diagnostics
        if not diagnostics.normalized:
            warnings.append(
                f"'{name}': {diagnostics.n_rows_not_normalized:,} row(s) do not sum to 1, so log loss, "
                f"Brier and the calibration figures are not reported for it -- they would be numbers "
                f"without a meaning rather than bad numbers."
            )
        if diagnostics.constant_labels:
            warnings.append(
                f"'{name}': the score column(s) {', '.join(diagnostics.constant_labels)} are constant, "
                f"so nothing about those classes can be ranked; they are excluded from the macro "
                f"ROC-AUC and average precision."
            )
    for warning in warnings:
        section.blocks.append(report.paragraph(warning, note=True))


def _add_per_class(report: Report, metrics_by_name: dict, focus: list | None) -> None:
    section = report.section("Per class")
    for name, metrics in metrics_by_name.items():
        table = metrics.per_class[[c for c in PER_CLASS_COLUMNS if c in metrics.per_class.columns]]
        if focus is not None:
            n_all = len(table)
            table = table.set_index("label").reindex([c for c in focus if c in set(table["label"])]).reset_index()
            section.blocks.append(report.paragraph(_focus_note(focus, n_all), note=True))
        section.blocks.append(report.paragraph(name))
        section.blocks.append(report.table(
            [c.replace("_", " ") for c in table.columns],
            [[row[c] for c in table.columns] for _, row in table.iterrows()],
        ))


def _add_confusion(report: Report, metrics_by_name: dict, focus: list | None) -> None:
    section = report.section("Confusion matrices")
    for name, metrics in metrics_by_name.items():
        confusion = metrics.confusion
        if focus is not None:
            _add_focus_confusion(report, section, name, metrics, focus)
            continue
        size = max(4.0, 0.42 * len(confusion.columns) + 2.2)
        fig, ax = subplots(figsize=(size, size * 0.9))
        ax.imshow(confusion.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(confusion.index)), confusion.index, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name} -- accuracy {metrics.accuracy:.4f}", fontsize=10)
        maximum = confusion.to_numpy().max() or 1
        for i in range(len(confusion.index)):
            for j in range(len(confusion.columns)):
                value = int(confusion.iat[i, j])
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=7,
                            color="white" if value > maximum * 0.55 else "black")
        fig.tight_layout()
        section.blocks.append(report.figure(fig, f"Confusion matrix -- {name}"))


def _add_focus_confusion(report: Report, section, name: str, metrics, focus: list) -> None:
    """A many-class matrix, the way the Metrics tab shows it: the focus
    classes with an "other" column, the most confused pairs across every
    class, and the whole matrix as one picture of where the errors are."""
    confusion = metrics.confusion
    grid, other = cf.focus_matrix(confusion, focus)
    shown = grid.copy()
    shown["(other)"] = other
    counts = shown.to_numpy()
    n = len(grid)
    fig, ax = subplots(figsize=(0.42 * (n + 1) + 2.2, (0.42 * (n + 1) + 2.2) * 0.9))
    ax.imshow(np.where(np.arange(n + 1) < n, counts, 0), cmap="Blues")
    ax.set_xticks(range(n + 1), list(shown.columns), rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), list(shown.index), fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name} -- accuracy {metrics.accuracy:.4f}", fontsize=10)
    maximum = grid.to_numpy().max() or 1
    for i in range(n):
        for j in range(n + 1):
            value = int(counts[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=7,
                        color="white" if j < n and value > maximum * 0.55 else "black")
    fig.tight_layout()
    section.blocks.append(report.figure(fig, f"Confusion matrix -- {name}, {n} of {len(confusion):,} classes"))
    section.blocks.append(report.paragraph(
        _focus_note(focus, len(confusion)) + " The last column counts the crops predicted as any "
        "class outside them.", note=True))

    pairs = cf.top_confused_pairs(confusion)
    section.blocks.append(report.table(
        ["True class", "Predicted as", "Crops", "Of true class"],
        [[str(t), str(p), int(c), f"{share * 100:.1f}%"]
         for t, p, c, share in pairs.head(REPORT_PAIRS).itertuples(index=False)],
        note=f"Most confused pairs -- {min(len(pairs), REPORT_PAIRS)} of {len(pairs):,} non-zero "
             f"off-diagonal cells, across every class.",
    ))
    block = max(1, -(-len(confusion) // cf.ZOOM_BLOCKS))
    section.blocks.append(report.figure(
        cf.overview_figure(confusion, focus, f"{name} -- all {len(confusion):,} classes", block=block),
        f"Where the errors are, across every class -- {name}"))


def _add_comparison(report: Report, state, baseline: str, candidate: str, focus: list | None = None) -> None:
    section = report.section(f"Compare: {baseline} to {candidate}")
    joined = cmp.build_comparison(state, baseline, candidate)
    diff = cmp.compare_experiments(state, baseline, candidate, joined=joined)

    section.blocks.append(report.tiles([
        ("Broken by the candidate", diff.counts[cmp.OUTCOME_REGRESSION], "neg"),
        ("Fixed by the candidate", diff.counts[cmp.OUTCOME_FIX], "pos"),
        ("Net change", f"{diff.net_change():+,}", "pos" if diff.net_change() >= 0 else "neg"),
        ("Accuracy delta", f"{diff.accuracy_delta:+.4f}",
         "pos" if diff.accuracy_delta >= 0 else "neg"),
    ]))
    section.blocks.append(report.paragraph(
        f"Compared on the {diff.n_common:,} crop(s) both experiments cover "
        f"({diff.n_baseline_only:,} only in {baseline}, {diff.n_candidate_only:,} only in "
        f"{candidate}). The accuracy delta only shows the net: "
        f"{diff.counts[cmp.OUTCOME_REGRESSION]:,} crop(s) that used to be right are now wrong.",
        note=True))

    per_class = cmp.per_class_change(joined, diff.labels).sort_values("net")
    n_all = len(diff.labels)
    if focus is not None:
        focus = [c for c in focus if c in set(diff.labels)]
        per_class = per_class[per_class["label"].isin(set(focus))]
    section.blocks.append(report.table(
        ["Class", "Support", "Fixed", "Broken", "Net"],
        [[str(r["label"]), int(r["support"]), int(r["fixes"]), int(r["regressions"]),
          f"{int(r['net']):+,}"] for _, r in per_class.iterrows()],
        note="Worst first. A flat net number hides a model that traded one class for another.",
    ))

    delta, moved_in, moved_out = diff.delta_confusion, diff.moved_in_confusion, diff.moved_out_confusion
    if focus is not None and not delta.empty:
        # The focus classes both ways: the full squares are thousands of
        # columns wide, which neither a page nor a slide can hold.
        delta, moved_in, moved_out = (m.loc[focus, focus] for m in (delta, moved_in, moved_out))
        section.blocks.append(report.paragraph(_focus_note(focus, n_all), note=True))
    if not delta.empty:
        section.blocks.append(report.paragraph(
            f"Differential confusion matrix ({candidate} minus {baseline}), over the common crops."))
        section.blocks.append(report.dataframe(delta))
        # The difference nets arrivals against departures, so a cell that
        # took in two hundred crops and lost two hundred others reads as
        # zero -- the same reason the Compare tab offers all three (see
        # comparison.movement_confusion).
        section.blocks.append(report.paragraph(
            f"Moved in: the {diff.n_moved:,} crop(s) whose predicted class changed, counted where "
            f"{candidate} put them. Nothing is netted here."))
        section.blocks.append(report.dataframe(moved_in))
        section.blocks.append(report.paragraph(
            f"Moved out: the same crops, counted where {baseline} had put them. Cell by cell, "
            "moved in minus moved out is the differential matrix above."))
        section.blocks.append(report.dataframe(moved_out))


def default_report_name(state, extension: str = "html") -> str:
    stem = Path(state.dataset.csv_path).stem if state.dataset and state.dataset.csv_path else "lookin"
    return f"{stem}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.{extension}"
