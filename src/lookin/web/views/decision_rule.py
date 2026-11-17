"""Decision Rule: one bias per class added to an experiment's log-scores
before the argmax (core/decision_rule.py), tuned on validation rows or --
when the user says so, loudly -- on the evaluated rows themselves.

Top to bottom: the experiment; the rows the biases are tuned on, with what
is checked about them; the objective, the per-class table where classes are
counted in it or locked at 0 and every bias can be typed; what the rule
does to the evaluated rows, with the one-vs-rest F1-max and AP as the
reference; the per-class bars before and after; one class's
precision-recall as its bias sweeps (the deployable counterpart of its
one-vs-rest curve); and the confusion matrix before, after and the change.
"Apply to the workspace" makes every tab read the experiment as if its
network carried the biases (AppState.apply_decision_rule).

Everything computed runs on this view's job: reading the validation rows,
fitting, and re-evaluating after a bias is typed. The tab always works from
the scores as read (AppState.raw_merged), whatever rule is applied.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lookin.core import charts
from lookin.core import class_focus as cf
from lookin.core import decision_rule as dr
from lookin.core.chart_style import color_for_experiment
from lookin.core.metrics import ExperimentMetrics, compute_metrics
from lookin.web import html as h
from lookin.web.views.base import ConfirmDialog, RebuildView
from lookin.web.views.media import fig_html
from lookin.web.views.widgets import ClassFocus, class_select, experiment_options, rule_tag

# Per-class metrics the bars can show: the Metrics tab's.
from lookin.web.views.metrics import METRIC_OPTIONS, PERCLASS_MANUAL_PAGES  # noqa: E402

# Which classes the per-class table, bars and matrices show past FOCUS_MAX.
DECISION_FOCUS_CRITERIA = (
    (cf.FOCUS_MOST_CHANGED, "Most changed F1"),
    (cf.FOCUS_WORST_F1, "Worst F1 after"),
    (cf.FOCUS_LARGEST_SUPPORT, "Largest support"),
    (cf.FOCUS_MANUAL, "Hand-picked"),
)
# The two series of the per-class bars -- names of the legend.
BEFORE = "Plain argmax"
AFTER = "Decision rule"
BEFORE_COLOR = "#9aa0aa"
# How often a running fit may redraw its progress line.
PROGRESS_EVERY = 0.4
DASH = "—"

SUMMARY_ROWS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("mcc", "MCC"),
    ("cohen_kappa", "Cohen's kappa"),
)
REFERENCE_ROWS = (
    ("macro_f1_max", "Macro F1-max"),
    ("weighted_f1_max", "Weighted F1-max"),
    ("macro_avg_precision", "Macro AP"),
    ("weighted_avg_precision", "Weighted AP"),
)


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _fmt(value, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None and value == value else DASH


def _delta(after, before) -> str:
    if after is None or before is None or after != after or before != before:
        return DASH
    d = after - before
    cls = "c-ok" if d > 5e-4 else "c-warn" if d < -5e-4 else "muted"
    return f'<span class="{cls}">{d:+.3f}</span>'


def _objective_label(key: str) -> str:
    return dict(dr.OBJECTIVES).get(key, key)


@dataclass
class _Context:
    """What one (experiment, tuning rows) pair needs, read once: the
    evaluated rows and the tuning rows as the optimiser sees them, and the
    metrics of the plain argmax."""
    experiment: str
    score_cols: list
    labels: list
    label_col: str
    uuid_col: str
    eval_frame: Optional[pd.DataFrame] = None
    eval_problem: Optional[dr.Problem] = None
    pool: Optional[dr.TuningPool] = None
    tuning_problem: Optional[dr.Problem] = None
    check: Optional[dr.PoolCheck] = None
    before: Optional[ExperimentMetrics] = None
    problem: str = ""                       # why nothing can be tuned

    @property
    def usable(self) -> bool:
        return not self.problem and self.tuning_problem is not None and (
            self.check is None or self.check.blocking is None)


@dataclass
class _Fit:
    result: dr.TuningResult
    cross: Optional[dr.CrossFit] = None
    spec: Optional[dr.TuningSpec] = None


@dataclass
class _Out:
    """What the last evaluation drew."""
    after: Optional[ExperimentMetrics] = None
    objective_before: float = float("nan")
    objective_now: float = float("nan")
    objective_note: str = ""
    rows: list = field(default_factory=list)       # per-class table rows (dicts)
    bars: str = ""
    sweep: str = ""
    confusion: str = ""


class DecisionRuleView(RebuildView):
    name = "decision"
    title = "Decision Rule"
    stale_message = "Data has changed -- the evaluation below is out of date."

    def __init__(self, session):
        super().__init__(session)
        self.experiment: Optional[str] = None
        self.pool = dr.POOL_VAL
        self.split_column = ""
        self.split_value = dr.DEFAULT_VAL_VALUE
        self.objective = dr.OBJ_MACRO_F1
        self.excluded: set = set()      # labels left out of the objective
        self.locked: set = set()        # labels whose bias stays 0
        self.bound_text = ""
        self.biases: dict = {}          # label -> bias being looked at
        self.metric = "f1"
        self.sweep_class: Optional[str] = None
        self.ctx: Optional[_Context] = None
        self._ctx_key = None
        self.fit: Optional[_Fit] = None
        self.out = _Out()
        self.message = ""               # a problem with what was typed
        self.class_focus = ClassFocus(self.name, "focus", DECISION_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the table, the bars and the matrices",
                                      support_criteria=(cf.FOCUS_WORST_F1,))
        self._focus: list = []
        self._last_progress = 0.0
        self.built = False

    # ---- lifecycle -----------------------------------------------------

    def _sync(self) -> None:
        names = list(self.state.experiments)
        if self.experiment not in names:
            self._select_experiment(names[0] if names else None)
        if not self.split_column and self.state.dataset is not None:
            row_filter = self.state.dataset.row_filter
            self.split_column = (row_filter[0] if row_filter and row_filter[0] else dr.DEFAULT_SPLIT_COLUMN)

    def _select_experiment(self, name: Optional[str]) -> None:
        self.experiment = name
        self.excluded, self.locked = set(), set()
        self.fit = None
        self.sweep_class = None
        exp = self.state.experiments.get(name) if name else None
        rule = exp.decision_rule if exp is not None else None
        self.biases = self._biases_from_rule(exp, rule) if rule else {}

    @staticmethod
    def _biases_from_rule(exp, rule) -> dict:
        labels = exp.class_labels[: len(exp.score_cols)]
        return {str(l): float(rule["biases"].get(str(c), 0.0)) for c, l in zip(exp.score_cols, labels)}

    def can_update(self) -> bool:
        return self.state.dataset is not None and bool(self.state.experiments)

    def on_shown(self) -> None:
        if not self.built and self.can_update():
            self.built = True
            self.rebuild(self._prepare)

    def on_data_change(self) -> None:
        self.dirty = True
        self._ctx_key = None

    def on_update(self, **_):
        self._ctx_key = None
        self.built = True
        self.rebuild(self._prepare)

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if self.experiment == old_name:
            self.experiment = new_name
            if self.ctx is not None:
                self.ctx.experiment = new_name
            self._ctx_key = None if self._ctx_key is None else (new_name, *self._ctx_key[1:])
        self.refresh()

    # ---- computing -------------------------------------------------------

    def _key(self) -> tuple:
        return (self.experiment, self.pool, self.split_column, self.split_value)

    def _prepare(self) -> None:
        """Read what the chosen experiment and tuning rows need (once per
        choice), then evaluate."""
        self._sync()
        if self.experiment is None or self.state.dataset is None:
            self.ctx = None
            return
        if self._ctx_key != self._key() or self.ctx is None:
            self.ctx = self._read_context()
            self._ctx_key = self._key()
        self._evaluate()

    def _read_context(self) -> _Context:
        state = self.state
        name = self.experiment
        exp = state.experiments[name]
        cols = state.dataset.columns
        labels = list(exp.class_labels[: len(exp.score_cols)])
        ctx = _Context(name, list(exp.score_cols), labels, cols.label_col, cols.uuid_col)
        self.set_progress(f"Reading the evaluated rows of {name}...")
        ctx.eval_frame = state.raw_merged(name)
        try:
            ctx.eval_problem = dr.prepare(ctx.eval_frame, cols.label_col, exp.score_cols, labels)
        except ValueError as ex:
            ctx.problem = f"No bias can be tuned for {name}: {ex}."
        self.set_progress("Scoring the plain argmax...")
        ctx.before = compute_metrics(name, ctx.eval_frame, cols.label_col, exp.score_cols, exp.class_labels)
        if ctx.problem:
            return ctx
        if self.pool == dr.POOL_TEST:
            ctx.pool = dr.evaluation_as_tuning_pool(ctx.eval_frame)
            ctx.tuning_problem = ctx.eval_problem
            ctx.check = dr.check_pools([], [], same_pool=True)
            return ctx
        self.set_progress(f"Reading the rows where '{self.split_column}' is '{self.split_value}'...")
        ctx.pool = dr.read_tuning_pool(state, name, self.split_column, self.split_value)
        if ctx.pool.frame is None:
            return ctx
        ctx.check = dr.check_pools(ctx.pool.frame[cols.uuid_col], ctx.eval_frame[cols.uuid_col], same_pool=False)
        try:
            ctx.tuning_problem = dr.prepare(ctx.pool.frame, cols.label_col, exp.score_cols, labels,
                                            space=ctx.eval_problem.space)
        except ValueError as ex:
            ctx.problem = f"The tuning rows cannot be used: {ex}."
        return ctx

    def _bias_vector(self, labels: list) -> np.ndarray:
        return np.array([0.0 if str(l) in self.locked else float(self.biases.get(str(l), 0.0)) for l in labels])

    def _spec(self) -> dr.TuningSpec:
        labels = self.ctx.labels if self.ctx is not None else []
        classes = None
        if self.objective in dr.SUBSET_OBJECTIVES and self.excluded:
            classes = tuple(str(l) for l in labels if str(l) not in self.excluded)
        bound = None
        if self.bound_text.strip():
            try:
                bound = abs(float(self.bound_text))
            except ValueError:
                bound = None
        return dr.TuningSpec(objective=self.objective, classes=classes, locked=tuple(sorted(self.locked)),
                             max_abs_bias=bound or None)

    def _evaluate(self) -> None:
        """Everything that depends on the biases: the evaluated rows after
        the rule, the objective on the tuning rows, the table, the charts."""
        ctx = self.ctx
        out = _Out()
        if ctx is None or ctx.before is None:
            self.out = out
            return
        name = ctx.experiment
        exp = self.state.experiments[name]
        if ctx.eval_problem is not None:
            biases = self._bias_vector(ctx.labels)
            self.set_progress("Scoring the evaluated rows with the biases...")
            after_frame = dr.apply_to_frame(ctx.eval_frame, ctx.label_col, ctx.score_cols, ctx.labels,
                                            ctx.eval_problem.space, biases)
            out.after = compute_metrics(name, after_frame, ctx.label_col, ctx.score_cols, exp.class_labels)
            if ctx.tuning_problem is not None:
                try:
                    spec = self._spec()
                    zeros = np.zeros(len(ctx.labels))
                    out.objective_before = dr.objective_value(ctx.tuning_problem, spec, zeros)
                    out.objective_now = dr.objective_value(ctx.tuning_problem, spec, biases)
                except ValueError as ex:
                    out.objective_note = str(ex)
        else:
            out.after = ctx.before
        out.rows = self._table_rows(ctx, out.after)
        self._resolve_focus(out.rows)
        self.out = out
        self.set_progress("Drawing...")
        out.bars = self._bars_html(ctx, out.after)
        out.sweep = self._sweep_html(ctx)
        out.confusion = self._confusion_html(ctx, out.after)
        self.out = out

    def _table_rows(self, ctx: _Context, after: ExperimentMetrics) -> list:
        before_pc = ctx.before.per_class.set_index("label")
        after_pc = after.per_class.set_index("label")
        tuning_n = ({str(l): int(n) for l, n in zip(ctx.tuning_problem.labels, ctx.tuning_problem.n_true)}
                    if ctx.tuning_problem is not None else {})
        rows = []
        for label in ctx.labels:
            f1_b = float(before_pc["f1"].get(label, np.nan)) if label in before_pc.index else np.nan
            f1_a = float(after_pc["f1"].get(label, np.nan)) if label in after_pc.index else np.nan
            support = int(before_pc["support"].get(label, 0)) if label in before_pc.index else 0
            rows.append({"label": str(label), "n_tuning": tuning_n.get(str(label), 0), "support": support,
                         "f1_before": f1_b, "f1_after": f1_a,
                         "change": abs(f1_a - f1_b) if f1_a == f1_a and f1_b == f1_b else np.nan})
        return rows

    def _resolve_focus(self, rows: list) -> None:
        table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["label", "support", "change", "f1_after"])
        rankings = {cf.FOCUS_MOST_CHANGED: ("change", False), cf.FOCUS_WORST_F1: ("f1_after", True),
                    cf.FOCUS_LARGEST_SUPPORT: ("support", False)}
        self._focus = self.class_focus.resolve(
            table["label"].tolist(),
            lambda criterion: cf.rank_by(table, *rankings[criterion],
                                         min_support=self.class_focus.min_support_for(criterion))
            if criterion in rankings else [])

    def _on_focus_change(self) -> None:
        if self.ctx is not None:
            self.rebuild(self._evaluate, clear_stale=False)

    def on_focus(self, op: str = "", **args):
        self.class_focus.handle(op, args)

    # ---- the charts ------------------------------------------------------

    def _shown_labels(self, labels: list) -> list:
        if self.class_focus.many:
            return [l for l in self._focus if l in set(map(str, labels))]
        return [str(l) for l in labels]

    def _bars_html(self, ctx: _Context, after: ExperimentMetrics) -> str:
        metric = self.metric if self.metric in METRIC_OPTIONS else "f1"
        per_class = {BEFORE: ctx.before.per_class.set_index("label"), AFTER: after.per_class.set_index("label")}
        labels = [l for l in self._shown_labels(ctx.labels) if l in per_class[BEFORE].index]
        if not labels:
            return ""
        colors = {BEFORE: BEFORE_COLOR, AFTER: color_for_experiment(ctx.experiment, self.state.color_slots)}
        fig, _uneven = charts.per_class_bars(per_class, metric, labels, colors,
                                             title=f"Per-class {metric}, before and after the rule")
        return fig_html(self.s, fig, f"Per-class {metric}, plain argmax and decision rule")

    def _sweep_html(self, ctx: _Context) -> str:
        problem = ctx.tuning_problem
        if problem is None or not ctx.usable:
            return ""
        names = [str(l) for l in problem.labels]
        if self.sweep_class not in names:
            # The class the rule moves most, else the first one shown.
            shown = set(self._shown_labels(problem.labels))
            moved = sorted((r for r in self.out.rows if r["label"] in shown and r["change"] == r["change"]),
                           key=lambda r: -r["change"])
            self.sweep_class = moved[0]["label"] if moved and moved[0]["change"] > 0 else (
                next(iter(self._shown_labels(problem.labels)), names[0]))
        label = self.sweep_class
        biases = self._bias_vector(problem.labels)
        spec = None
        try:
            spec = self._spec()
            dr._weights(problem, spec)
        except ValueError:
            spec = None
        sweep = dr.class_sweep(problem, biases, label, spec)
        frame = ctx.pool.frame if ctx.pool is not None and ctx.pool.frame is not None else ctx.eval_frame
        column = ctx.score_cols[names.index(label)]
        truth_value = next((v for v in ctx.labels if str(v) == label), label)
        ovr = dr.ovr_curve(frame, ctx.label_col, column, truth_value)
        plain = dr.class_point(problem, np.zeros(len(names)), label)
        objective = (f"{_objective_label(self.objective)} (tuning rows)" if spec is not None else "")
        fig = charts.decision_rule_sweep(sweep, objective, ovr=ovr, plain=plain)
        return fig_html(self.s, fig, f"Bias sweep -- {label}")

    def _confusion_html(self, ctx: _Context, after: ExperimentMetrics) -> str:
        before_cm, after_cm = ctx.before.confusion, after.confusion
        if before_cm.empty:
            return ""
        focus = self._focus if self.class_focus.many else None
        figures = [
            (charts.confusion_matrix(before_cm, "Plain argmax", focus=focus), "Confusion -- plain argmax"),
            (charts.confusion_matrix(after_cm, "Decision rule", focus=focus), "Confusion -- decision rule"),
            (charts.confusion_change(before_cm, after_cm, "Change (decision rule - plain argmax)", focus=focus),
             "Confusion -- change"),
        ]
        return ('<div class="row-wrap" style="align-items:flex-start;gap:12px">'
                + "".join(f'<div>{fig_html(self.s, fig, title)}</div>' for fig, title in figures) + "</div>")

    # ---- fitting ---------------------------------------------------------

    def on_compute(self, **_):
        if self.ctx is None or not self.ctx.usable:
            return
        try:
            spec = self._spec()
            dr._weights(self.ctx.tuning_problem, spec)
        except ValueError as ex:
            self.message = str(ex)
            return
        self.message = ""
        self.rebuild(self._fit, clear_stale=False)

    def _progress(self, start: int, starts: int, passes: int, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last_progress < PROGRESS_EVERY:
            from lookin.web.session import checkpoint
            checkpoint()
            return
        self._last_progress = now
        self.set_progress(f"Fitting the biases -- starting point {start} of {starts}, pass {passes}, "
                          f"step {done + 1:,} of {total:,}...")

    def _fit(self) -> None:
        ctx = self.ctx
        if ctx is None or not ctx.usable:
            return
        spec = self._spec()
        problem = ctx.tuning_problem
        _log(f"decision rule: fitting {spec.objective} for '{ctx.experiment}' on {problem.n_rows:,} rows "
             f"({self.pool})...")
        t0 = time.monotonic()
        result = dr.fit_biases(problem, spec, progress=self._progress)
        _log(f"decision rule: {result.objective_before:.4f} -> {result.objective_after:.4f} in "
             f"{result.passes} pass(es), {time.monotonic() - t0:.1f}s")
        cross = None
        if self.pool == dr.POOL_TEST:
            self.set_progress("Measuring the optimism: fitting on one half, scoring on the other...")
            cross = dr.cross_fit(problem, spec, in_sample=result.objective_after)
        self.fit = _Fit(result, cross, spec)
        self.biases = {str(l): float(b) for l, b in zip(result.labels, result.biases)}
        self.sweep_class = None
        self._evaluate()

    # ---- actions -----------------------------------------------------------

    def on_experiment(self, value="", **_):
        if value in self.state.experiments and value != self.experiment:
            self._select_experiment(value)
            self.rebuild(self._prepare)

    def on_pool(self, value="", **_):
        if value in dr.POOL_KEYS and value != self.pool:
            self.pool = value
            self.fit = None
            self.rebuild(self._prepare)

    def on_split(self, column=None, value=None, **_):
        if column is not None:
            self.split_column = str(column).strip()
        if value is not None:
            self.split_value = str(value).strip()
        self.fit = None
        self.rebuild(self._prepare)

    def on_split_column(self, value="", **_):
        self.on_split(column=value)

    def on_split_value(self, value="", **_):
        self.on_split(value=value)

    def on_objective(self, value="", **_):
        if value in dict(dr.OBJECTIVES):
            self.objective = value
            self.rebuild(self._evaluate, clear_stale=False)

    def on_in_objective(self, label="", value=False, **_):
        if value:
            self.excluded.discard(str(label))
        else:
            self.excluded.add(str(label))
        self.rebuild(self._evaluate, clear_stale=False)

    def on_all_in_objective(self, value="all", **_):
        if value == "all":
            self.excluded = set()
        elif self.ctx is not None:
            self.excluded = {str(l) for l in self.ctx.labels}
        self.rebuild(self._evaluate, clear_stale=False)

    def on_lock(self, label="", value=False, **_):
        if value:
            self.locked.add(str(label))
            self.biases[str(label)] = 0.0
        else:
            self.locked.discard(str(label))
        self.rebuild(self._evaluate, clear_stale=False)

    def on_bias(self, label="", value="", **_):
        try:
            bias = float(str(value).strip() or 0.0)
        except ValueError:
            self.message = f"The bias of {label} must be a number."
            return
        if not np.isfinite(bias):
            return
        self.message = ""
        self.biases[str(label)] = bias
        self.rebuild(self._evaluate, clear_stale=False)

    def on_bound(self, value="", **_):
        text = str(value).strip()
        if text:
            try:
                if float(text) <= 0:
                    raise ValueError
            except ValueError:
                self.message = "The largest bias must be a positive number, or left empty for no bound."
                return
        self.message = ""
        self.bound_text = text

    def on_reset(self, **_):
        self.biases = {}
        self.fit = None
        self.rebuild(self._evaluate, clear_stale=False)

    def on_load_applied(self, **_):
        exp = self.state.experiments.get(self.experiment or "")
        if exp is not None and exp.decision_rule:
            self.biases = self._biases_from_rule(exp, exp.decision_rule)
            self.fit = None
            self.rebuild(self._evaluate, clear_stale=False)

    def on_metric(self, value="", **_):
        if value in METRIC_OPTIONS:
            self.metric = value
            if self.ctx is not None and self.out.after is not None:
                self.rebuild(lambda: setattr(self.out, "bars", self._bars_html(self.ctx, self.out.after)),
                             clear_stale=False)

    def on_sweep_class(self, value="", **_):
        self.sweep_class = str(value)
        if self.ctx is not None:
            self.rebuild(lambda: setattr(self.out, "sweep", self._sweep_html(self.ctx)), clear_stale=False)

    # ---- applying ------------------------------------------------------------

    def _record(self) -> dict:
        ctx = self.ctx
        spec = self._spec()
        return dr.make_record(
            ctx.score_cols, ctx.labels, self._bias_vector(ctx.labels), ctx.eval_problem.space, self.pool,
            split_column=self.split_column if self.pool == dr.POOL_VAL else "",
            split_value=self.split_value if self.pool == dr.POOL_VAL else "",
            n_rows=ctx.tuning_problem.n_rows if ctx.tuning_problem is not None else 0,
            objective=self.objective, classes=spec.classes, locked=spec.locked, max_abs_bias=spec.max_abs_bias,
            objective_before=self.out.objective_before, objective_after=self.out.objective_now)

    def on_apply(self, **_):
        ctx = self.ctx
        if ctx is None or ctx.eval_problem is None or not ctx.usable:
            return
        record = self._record()
        if self.pool == dr.POOL_TEST:
            self.s.open_dialog(ConfirmDialog(
                "Apply biases tuned on the evaluated rows?",
                f"These biases were fitted on the {ctx.eval_problem.n_rows:,} rows every tab evaluates "
                f"{ctx.experiment} on. Once applied, every metric of the workspace is measured on the rows the "
                f"rule was fitted to, and is optimistic. The experiment will carry a red badge saying so.",
                lambda: self._apply(record), yes_label="Apply anyway", danger=True, width=520))
            return
        self._apply(record)

    def _apply(self, record: dict) -> None:
        name = self.ctx.experiment
        try:
            self.state.apply_decision_rule(name, record)
        except ValueError as ex:
            self.s.toast(f"Not applied: {ex}", "error")
            return
        _log(f"decision rule applied to '{name}' (tuned on {record['tuned_on']})")
        self.s.toast(f"Decision rule applied to {name}: every tab now reads it with these biases.", "ok")
        self.s.notify_change()

    def on_revert(self, **_):
        name = self.experiment
        if name and name in self.state.experiments and self.state.revert_decision_rule(name):
            self.s.toast(f"{name} is read with its scores as they are again.", "ok")
            self.s.notify_change()

    def on_export(self, **_):
        ctx = self.ctx
        if ctx is None or ctx.eval_problem is None:
            return
        from lookin.core.export_names import export_name, workspace_part
        filename = export_name(workspace_part(self.s.current_workspace, self.state), ctx.experiment,
                               "decision_rule", ext="json")
        data = json.dumps(self._record(), indent=2, sort_keys=True).encode("utf-8")
        key = self.s.put_download(data, filename, "application/json")
        self.s.cmd(download=f"/dl/{self.s.sid}/{key}", filename=filename)

    # ---- rendering -------------------------------------------------------------

    def render(self) -> str:
        self._sync()
        self.on_shown()
        state = self.state
        parts = [self.stale_html()]
        if state.dataset is None or not state.experiments:
            parts.append(h.banner("Load a dataset and at least one experiment (results CSV) in the 'Sources' "
                                  "tab first.", "info"))
            return f'<div class="stack">{"".join(parts)}</div>'
        parts.append(self._head_html())
        if self.progress_text or self.error:
            parts.append(self.progress_html())
        ctx = self.ctx
        if ctx is None or ctx.before is None:
            return f'<div class="stack">{"".join(parts)}</div>'
        if ctx.problem:
            parts.append(h.banner(ctx.problem, "error"))
        parts.append(self._pool_html(ctx))
        if ctx.eval_problem is not None:
            parts.append(self._tuning_html(ctx))
            parts.append(self._summary_html(ctx))
        if self.out.bars:
            right = h.select([(m, m) for m in METRIC_OPTIONS], self.metric, change=self.c("metric"),
                             label="Per-class metric")
            parts.append(h.section("Per class, before and after", self.out.bars,
                                   manual=PERCLASS_MANUAL_PAGES.get(self.metric, "f1"), right=right,
                                   sub="On the evaluated rows."))
        if self.out.sweep:
            picker = class_select([str(l) for l in ctx.labels], self.sweep_class,
                                  change=self.c("sweep_class"), label="Class")
            parts.append(h.section("One class, every bias", self.out.sweep, manual="decision-rule-curve",
                                   right=picker, sub="On the tuning rows, the other biases as set above."))
        if self.out.confusion:
            parts.append(h.section("Confusion matrix", self.out.confusion, manual="confusion-matrix",
                                   sub="On the evaluated rows."))
        return f'<div class="stack">{"".join(parts)}</div>'

    def _head_html(self) -> str:
        state = self.state
        names = list(state.experiments)
        picker = h.select(experiment_options(state, names), self.experiment or "", change=self.c("experiment"),
                          label="Experiment")
        exp = state.experiments.get(self.experiment or "")
        status = ""
        if exp is not None and exp.decision_rule:
            rule = exp.decision_rule
            tuned = ("the evaluated rows" if dr.is_tuned_on_evaluation(rule)
                     else f"{rule.get('n_rows', 0):,} rows where '{rule.get('split_column')}' is "
                          f"'{rule.get('split_value')}'")
            status = h.banner(
                f"A decision rule is applied to {self.experiment}: every tab reads it with these biases "
                f"(tuned on {tuned}, {rule.get('created', '')}).",
                "warn" if dr.is_tuned_on_evaluation(rule) else "info",
                actions=(h.button("Show its biases", act_attr=self.a("load_applied"), small=True)
                         + h.button("Revert", act_attr=self.a("revert"), small=True, icon_name="x")))
        body = (f'<div class="row-wrap">{picker}{rule_tag(state, self.experiment) if self.experiment else ""}</div>'
                f'{status}')
        return h.section("Decision rule", body, manual="decision-rule",
                         sub="One bias per class added to the log-scores before the argmax.")

    def _pool_html(self, ctx: _Context) -> str:
        pool_pick = h.segmented(((dr.POOL_VAL, "Validation rows"), (dr.POOL_TEST, "Evaluated rows")), self.pool,
                                act_for=lambda k: self.a("pool", value=k), label="Tune on")
        controls = f'<div class="row-wrap"><b class="small">Tune on</b>{pool_pick}'
        if self.pool == dr.POOL_VAL:
            controls += (h.text_input(self.split_column, label="Rows where column", width=120,
                                      change=self.c("split_column"))
                         + h.text_input(self.split_value, label="is", width=100, change=self.c("split_value")))
        controls += "</div>"
        n_eval = ctx.eval_problem.n_rows if ctx.eval_problem is not None else len(ctx.eval_frame)
        banners = []
        pool = ctx.pool
        if self.pool == dr.POOL_TEST:
            banners.append(h.banner(
                f"Tuning on the {n_eval:,} evaluated rows themselves. Every metric below is measured on the rows "
                f"the biases are fitted to, so it is optimistic -- the more so for small classes. Use it to explore; "
                f"deploy biases tuned on validation rows.", "error"))
        elif pool is not None and pool.disabled_reason:
            banners.append(h.banner(pool.disabled_reason, "error"))
        elif pool is not None:
            if ctx.check is not None and ctx.check.blocking:
                banners.append(h.banner(ctx.check.blocking, "error"))
            else:
                banners.append(h.banner(
                    f"Tuning on {pool.n_rows:,} validation rows (where '{pool.split_column}' is "
                    f"'{pool.split_value}'); none of them is among the {n_eval:,} evaluated rows.", "ok"))
            for warning in pool.warnings:
                banners.append(h.banner(warning, "warn"))
            if ctx.check is not None and ctx.check.n_tuning_duplicates:
                banners.append(h.banner(f"{ctx.check.n_tuning_duplicates:,} uuid(s) appear more than once among "
                                        f"the tuning rows.", "warn"))
        return h.section("Tuning rows", controls + "".join(banners), manual="decision-rule-pools",
                         sub="The rows the biases are fitted to. The evaluated rows are the ones every tab uses.")

    def _tuning_html(self, ctx: _Context) -> str:
        objective = h.select(list(dr.OBJECTIVES), self.objective, change=self.c("objective"), label="Maximise")
        bound = h.text_input(self.bound_text, label="Largest |bias|", width=80, change=self.c("bound"),
                             placeholder="none", title="No bias goes past this, either way. Empty: no bound.")
        compute = h.button("Compute the best biases", act_attr=self.a("compute"), kind="primary", icon_name="chart",
                           disabled=not ctx.usable or self.running)
        reset = h.button("All biases to 0", act_attr=self.a("reset"), small=True, kind="ghost")
        subset = self.objective in dr.SUBSET_OBJECTIVES
        bulk = (h.button("Count every class", act_attr=self.a("all_in_objective", value="all"), small=True,
                         kind="ghost", disabled=not subset)
                + h.button("Count none", act_attr=self.a("all_in_objective", value="none"), small=True,
                           kind="ghost", disabled=not subset))
        controls = f'<div class="row-wrap">{objective}{bound}{compute}{reset}{bulk}</div>'
        notes = []
        if self.message:
            notes.append(h.banner(self.message, "error"))
        if not subset:
            notes.append('<div class="muted small"><i>Balanced accuracy counts every class: a mean of recalls never '
                         'pays for a false positive, so over a subset its best is to predict that subset '
                         'everywhere.</i></div>')
        out = self.out
        if out.objective_note:
            notes.append(h.banner(out.objective_note, "warn"))
        elif out.objective_now == out.objective_now:
            where = "evaluated" if self.pool == dr.POOL_TEST else "tuning"
            notes.append(f'<div class="small"><b>{h.e(_objective_label(self.objective))}</b> on the {where} rows: '
                         f'{_fmt(out.objective_before)} with every bias at 0, <b>{_fmt(out.objective_now)}</b> with '
                         f'the biases below ({_delta(out.objective_now, out.objective_before)}).</div>')
        if self.fit is not None:
            notes.append(self._fit_note(self.fit))
        collapsed = [r["label"] for r in out.rows if r["label"] in self.excluded and r["f1_before"] == r["f1_before"]
                     and r["f1_after"] == r["f1_after"] and r["f1_after"] < r["f1_before"] - 0.05]
        if collapsed:
            notes.append(h.banner(
                f"Left out of the objective, {len(collapsed)} class(es) lose more than 0.05 of F1 on the evaluated "
                f"rows: {', '.join(collapsed[:8])}{'...' if len(collapsed) > 8 else ''}.", "warn"))
        return h.section("Biases", controls + "".join(notes) + self.class_focus.render() + self._table_html(ctx),
                         manual="decision-rule", sub="Fitted, or typed by hand -- every change is scored at once.")

    def _fit_note(self, fit: _Fit) -> str:
        r = fit.result
        ref = (f" {r.reference} is kept at 0 as the reference (biases are only defined up to a common constant)."
               if r.reference else "")
        absent = f" No tuning row for: {', '.join(map(str, r.absent))}." if r.absent else ""
        conv = "" if r.converged else f" Stopped after {r.passes} passes before converging."
        text = (f'<div class="muted small"><i>Fitted on {r.n_rows:,} rows: no single class\'s bias can be moved for '
                f'the better from here (a local optimum).{h.e(ref)}{h.e(absent)}{h.e(conv)}</i></div>')
        if fit.cross is not None:
            c = fit.cross
            text += h.banner(
                f"Optimism of tuning on the evaluated rows: fitted on half of them and scored on the other half, "
                f"the objective is {c.held_out:.3f}, against {c.in_sample:.3f} fitted and scored on all of them "
                f"({c.before:.3f} with every bias at 0). The gap is how much the numbers below flatter the rule.",
                "warn")
        return text

    def _table_html(self, ctx: _Context) -> str:
        rows = self.out.rows
        if not rows:
            return ""
        shown = set(self._shown_labels(ctx.labels))
        subset = self.objective in dr.SUBSET_OBJECTIVES
        where = "evaluated" if self.pool == dr.POOL_TEST else "tuning"
        head = ("<tr><th>Class</th>"
                f'<th class="num" title="Rows of this class among the {where} rows">Tuning rows</th>'
                '<th class="num" title="Rows of this class among the evaluated rows">Evaluated rows</th>'
                '<th title="Whether this class\'s F1 counts in the objective">In objective</th>'
                '<th title="Keep this class\'s bias at 0">Lock at 0</th>'
                '<th class="num">Bias</th><th class="num">F1 before</th><th class="num">F1 after</th>'
                '<th class="num">Change</th></tr>')
        body = []
        for r in rows:
            label = r["label"]
            if label not in shown:
                continue
            locked = label in self.locked
            counted = (label not in self.excluded) or not subset
            bias = 0.0 if locked else self.biases.get(label, 0.0)
            body.append(
                f'<tr><td><b>{h.e(label)}</b></td>'
                f'<td class="num">{r["n_tuning"]:,}</td><td class="num">{r["support"]:,}</td>'
                f'<td>{h.checkbox("", counted, change=self.c("in_objective", label=label), disabled=not subset)}</td>'
                f'<td>{h.checkbox("", locked, change=self.c("lock", label=label))}</td>'
                f'<td class="num">{h.text_input(f"{bias:.4g}", width=80, change=self.c("bias", label=label), disabled=locked, mono=True)}</td>'
                f'<td class="num">{_fmt(r["f1_before"])}</td><td class="num">{_fmt(r["f1_after"])}</td>'
                f'<td class="num">{_delta(r["f1_after"], r["f1_before"])}</td></tr>')
        return (f'<div class="tbl-wrap"><table class="tbl compact"><thead>{head}</thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>'
                f'<div class="muted small"><i>F1 before and after on the evaluated rows. A bias is in log-score units: '
                f'+0.7 roughly doubles how much the class is favoured.</i></div>')

    def _summary_html(self, ctx: _Context) -> str:
        before, after = ctx.before, self.out.after
        if after is None:
            return ""
        rows = "".join(
            f'<tr><td>{h.e(label)}</td><td class="num">{_fmt(getattr(before, key))}</td>'
            f'<td class="num"><b>{_fmt(getattr(after, key))}</b></td>'
            f'<td class="num">{_delta(getattr(after, key), getattr(before, key))}</td></tr>'
            for key, label in SUMMARY_ROWS)
        table = ('<div class="tbl-wrap"><table class="tbl compact"><thead><tr><th>Metric</th>'
                 '<th class="num">Plain argmax</th><th class="num">Decision rule</th><th class="num">Change</th>'
                 f'</tr></thead><tbody>{rows}</tbody></table></div>')
        reference = "".join(
            f'<tr><td>{h.e(label)}</td><td class="num">{_fmt(getattr(before, key))}</td></tr>'
            for key, label in REFERENCE_ROWS)
        reference = ('<div class="tbl-wrap"><table class="tbl compact"><thead><tr>'
                     '<th>Reference: one-vs-rest, scores as read</th><th class="num">Value</th></tr></thead>'
                     f'<tbody>{reference}</tbody></table></div>'
                     '<div class="muted small"><i>Each class ranked on its own score, with no rule that gives every '
                     'row one class: what no decision rule reaches, shown for scale.</i></div>')
        exp = self.state.experiments.get(ctx.experiment)
        applied = exp is not None and exp.decision_rule is not None
        same = applied and np.allclose(
            self._bias_vector(ctx.labels),
            np.array([float(exp.decision_rule["biases"].get(str(c), 0.0)) for c in ctx.score_cols]), atol=1e-9)
        apply_label = "Applied" if same else ("Apply these biases instead" if applied else "Apply to the workspace")
        actions = (h.button(apply_label, act_attr=self.a("apply"), kind="primary" if not same else "",
                            icon_name="check", disabled=same or not ctx.usable or self.running,
                            title="Every tab then reads this experiment as if its network carried these biases")
                   + (h.button("Revert", act_attr=self.a("revert"), icon_name="x") if applied else "")
                   + h.button("Export JSON", act_attr=self.a("export"), icon_name="download", kind="ghost",
                              title="The biases, keyed by score column, and how they were obtained"))
        body = (f'<div class="row-wrap" style="align-items:flex-start;gap:24px">'
                f'<div style="flex:1 1 360px">{table}</div><div style="flex:1 1 300px">{reference}</div></div>')
        return h.section("On the evaluated rows", body, manual="decision-rule", right=actions,
                         sub=f"{ctx.eval_problem.n_rows:,} rows -- the ones every tab uses.")
