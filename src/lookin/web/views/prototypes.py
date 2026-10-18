"""Prototypes: prototype-space diagnostics for experiments that have a
raw-feature sidecar attached -- the web port of lookin.ui.prototypes_view.

Same panels, same computations, same texts: the sidecar summary with the
k-means estimate per experiment, the role (composition) bar of every
prototype, the neighbor explorer (nearest / farthest / impostors), and,
with the model's own prototypes, compactness & separation, the
per-prototype table, the most confused class pairs, the assignment matrix
(class -> prototype, clickable) and the raw proximity matrix.

Which prototype source is active, the k-means estimates and the global
nearest-prototype assignment live in session.prototype_source (see
lookin.ui.prototype_source), which this view subscribes to the way the
original app wires PrototypesView.refresh.

Everything the panels draw is computed in a background rebuild and cached
as HTML plus the payloads the clicks need (the uuids behind a role segment,
the assignment arrays behind a matrix cell); render() only assembles it.
"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from lookin.core import class_focus as cf
from lookin.core import embedding_analysis as ea
from lookin.core.chart_style import subplots
from lookin.ui.components import GRID_NONE, blues_color, style_axes
from lookin.ui.embedding_context import build_analysis_context, coverage_note, experiments_with_embeddings
from lookin.ui.prototypes_view import (
    MARGIN_COLOR_LIMIT,
    MAX_GRID_PROTOTYPES,
    MAX_PROTOTYPE_TABLE_ROWS,
    MAX_ROLE_BARS,
    OTHER_COLORS,
    OWN_COLOR,
    PROTOTYPE_FOCUS_CRITERIA,
    REST_COLOR,
    SIMILARITY_AXIS_LIMITS,
    THUMB,
)
from lookin.web import html as h
from lookin.web.views.base import Dialog, RebuildView
from lookin.web.views.carousel import open_carousel
from lookin.web.views.media import crop_img, fig_html
from lookin.web.views.widgets import ClassFocus

GRID_CELL_SIZE = 34
GRID_LABEL_WIDTH = 110
DEFAULT_MATRIX_N = "31"
DEFAULT_K = "2"
MATRIX_ORDERS = (
    ("margin", "Lowest margin first"),
    ("alpha", "Normal order (alphabetical)"),
)


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _fmt(value: float, spec: str) -> str:
    return format(value, spec) if value == value else "—"


class MatrixSettingsDialog(Dialog):
    """The original's "Proximity matrix settings" dialog."""

    width = 420
    title = "Proximity matrix settings"

    def __init__(self, view: "PrototypesView"):
        super().__init__()
        self.view = view

    def body(self) -> str:
        v = self.view
        return (f'<form class="form-col" id="{self.id}-f"{self.submit("apply")}>'
                f'{h.text_input(v.matrix_n, name="matrix_n", label="Classes in the matrix", width="90px", autofocus=True)}'
                f'{h.select(MATRIX_ORDERS, v.matrix_order, name="order", label="Class order")}</form>')

    def footer(self) -> str:
        return (f'<button type="submit" form="{self.id}-f" class="btn btn-primary"><span>Apply</span></button>'
                + h.button("Close", act_attr=self.a("close")))

    def on_apply(self, matrix_n: str = "", order: str = "", **_):
        self.view.matrix_n = str(matrix_n)
        if order in {k for k, _ in MATRIX_ORDERS}:
            self.view.matrix_order = order
        self.close()
        self.view.rebuild()


class PrototypesView(RebuildView):
    name = "prototypes"
    title = "Prototypes"

    def __init__(self, session):
        super().__init__(session)
        self.exp_name = ""
        self.kmeans_k: dict = {}           # exp name -> K as typed
        self.scope_value = ""              # the neighbor explorer's choice (a positional key)
        self.matrix_n = DEFAULT_MATRIX_N
        self.matrix_order = "margin"
        self.class_focus = ClassFocus(self.name, "focus", PROTOTYPE_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the panels below",
                                      support_criteria=(cf.FOCUS_WORST_MARGIN, cf.FOCUS_LEAST_PURE))
        self._focus: list = []
        self._memo: dict = {}
        self.result: dict = {}             # what compute() produced; render() draws from it
        self.cases = None                  # the neighbor explorer's last result
        self.cases_busy = False
        self.built = False
        self._own_source_change = False
        # The original app subscribes PrototypesView.refresh to the registry:
        # an attach, a source selection or an estimate redraws every panel.
        self.s.prototype_source.subscribe(self._on_source_changed)

    # ---- lifecycle -------------------------------------------------------

    def on_shown(self) -> None:
        if not self.built and self.state.dataset is not None:
            self.built = True
            self.rebuild()

    def on_data_change(self) -> None:
        # Nothing drawn yet (no sidecar until now): there is nothing to call
        # out of date, so draw it, as the original's refresh did.
        if self.built and "ctx" not in self.result:
            self.rebuild()
        else:
            self.dirty = True

    def can_update(self) -> bool:
        return self.built and self.state.dataset is not None

    def _on_source_changed(self) -> None:
        if self._own_source_change or not self.built:
            return
        self.rebuild()

    def on_display_changed(self) -> None:
        # Only the crops depend on the display settings; their URLs carry
        # the LUT and the turn, so a re-render is all it takes.
        self.refresh()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if old_name in self.kmeans_k:
            self.kmeans_k[new_name] = self.kmeans_k.pop(old_name)
        if self.exp_name == old_name:
            self.exp_name = new_name
        if self.built:
            self.rebuild()
        else:
            self.refresh()

    def _on_focus_change(self) -> None:
        self.rebuild()

    # ---- computing ---------------------------------------------------------

    def compute(self) -> None:
        state = self.state
        self.cases = None
        if state.dataset is None or not experiments_with_embeddings(state):
            self.result = {"empty": True}
            return
        names = experiments_with_embeddings(state)
        if self.exp_name not in names:
            self.exp_name = names[0]
        exp_name = self.exp_name
        self.set_progress(f"Computing prototype diagnostics for '{exp_name}'...")
        ctx = build_analysis_context(state, exp_name)
        if ctx is None:
            self.result = {"exp": exp_name}
            return
        res = {"exp": exp_name, "ctx": ctx, "note": coverage_note(ctx, state, exp_name)}
        try:
            self._resolve_focus(ctx, exp_name)
            self._compute_role(res, ctx, exp_name)
            self._compute_scope(res, ctx, exp_name)
            self.set_progress("Drawing compactness & separation...")
            self._compute_compactness(res, ctx, exp_name)
        except Exception as ex:  # noqa: BLE001
            _log(f"error rendering prototypes tab: {ex}")
            traceback.print_exc()
            res["error"] = f"Error: {ex}"
        self.result = res

    def _cached(self, kind: str, compute, *inputs):
        """compute(), remembered for these exact input objects (by identity),
        one result per kind -- as in the original."""
        hit = self._memo.get(kind)
        if hit is not None and len(hit[0]) == len(inputs) and all(a is b for a, b in zip(hit[0], inputs)):
            return hit[1]
        value = compute()
        self._memo[kind] = (inputs, value)
        return value

    def _resolve_focus(self, ctx: dict, exp_name: str) -> None:
        labels = ctx["labels"]
        classes = sorted({str(l) for l in labels})
        registry = self.s.prototype_source

        def support_table() -> pd.DataFrame:
            counts = pd.Series(labels).astype(str).value_counts()
            return pd.DataFrame({"label": counts.index, "support": counts.to_numpy()})

        def least_pure() -> list:
            active = registry.active_set(exp_name)
            if active is None:
                return []
            source_key, proto_set, _color, _note = active
            best_class, best_k = registry.assignment(ctx, exp_name, source_key, proto_set)
            composition = self._cached(
                "composition",
                lambda: ea.prototype_composition(ctx["embeddings"], labels, proto_set, assignment=(best_class, best_k)),
                ctx["embeddings"], labels, proto_set, best_class)
            table = pd.DataFrame([
                {"label": str(cls), "purity": min((r.purity for r in roles if r.total), default=np.nan),
                 "support": sum(r.total for r in roles)}
                for cls, roles in composition.items()
            ])
            return cf.rank_by(table, "purity", ascending=True,
                              min_support=self.class_focus.min_support) if len(table) else []

        def rank(criterion):
            if criterion == cf.FOCUS_WORST_MARGIN and ctx["exp"].model_prototypes is not None:
                protos = ctx["exp"].model_prototypes
                stats = self._cached("class_stats", lambda: ea.per_class_stats(ctx["embeddings"], labels, protos),
                                     ctx["embeddings"], labels, protos)
                table = pd.DataFrame({"label": [str(s.label) for s in stats],
                                      "margin": [s.compactness - s.separation for s in stats],
                                      "support": [s.n for s in stats]})
                return cf.rank_by(table, "margin", ascending=True,
                                  min_support=self.class_focus.min_support_for(criterion))
            if criterion in (cf.FOCUS_WORST_MARGIN, cf.FOCUS_LEAST_PURE):
                return least_pure()
            if criterion == cf.FOCUS_LARGEST_SUPPORT:
                return cf.rank_by(support_table(), "support", ascending=False)
            return []

        self._focus = self.class_focus.resolve(classes, rank)

    # ---- role of each prototype ---------------------------------------------

    def _compute_role(self, res: dict, ctx: dict, exp_name: str) -> None:
        registry = self.s.prototype_source
        active = registry.active_set(exp_name)
        if active is None:
            res["role_html"] = h.banner(
                "No prototypes yet for this experiment -- import the model's real prototypes, "
                "or estimate them with kmeans, in the sidecar section above.", "info")
            return
        source_key, proto_set, color, note = active
        header = f'<div class="pr-src">{h.dot(color, size=8)}<b>{h.e(note)}</b></div>'
        best_class, best_k = registry.assignment(ctx, exp_name, source_key, proto_set)
        composition = self._cached(
            "composition",
            lambda: ea.prototype_composition(ctx["embeddings"], ctx["labels"], proto_set, assignment=(best_class, best_k)),
            ctx["embeddings"], ctx["labels"], proto_set, best_class)

        role_classes = proto_set.class_labels
        if self.class_focus.many:
            present = set(proto_set.class_labels)
            role_classes = [c for c in self._focus if c in present]
        roles = [(cls, role) for cls in role_classes for role in composition.get(cls, [])]
        total_roles = len(roles)
        truncated = total_roles > MAX_ROLE_BARS
        segments: list = []   # flat: index -> (title, uuids)
        index_of = proto_set.label_index()
        bars = [self._role_bar(ctx, index_of, best_class, best_k, role, segments)
                for _, role in roles[:MAX_ROLE_BARS]]
        res["segments"] = segments
        if not bars:
            res["role_html"] = h.section(
                "Role of each prototype",
                header + '<div class="muted small"><i>No prototype to show.</i></div>', manual="prototype-roles")
            return
        if truncated:
            count_text = (
                f"The first {len(bars)} of {total_roles:,} prototypes "
                f"({proto_set.k} per class over {len(proto_set.class_labels):,} classes) -- "
                f"the rest are left out to keep this panel drawable."
            )
        elif self.class_focus.many:
            count_text = (f"The {len(bars)} prototypes of the {len(role_classes)} classes picked above "
                          f"(out of {len(proto_set.class_labels):,}).")
        else:
            count_text = f"All {len(bars)} prototypes, across every class."
        caption = (f"{count_text} Nearest-prototype assignment (cosine), "
                   "over the prototypes of every class at once. Click a segment to open the matching images.")
        res["role_html"] = h.section(
            "Role of each prototype",
            f'{header}<div class="muted small">{h.e(caption)}</div><div class="pr-bars">{"".join(bars)}</div>',
            manual="prototype-roles")

    def _role_bar(self, ctx: dict, index_of: dict, best_class, best_k, role, segments: list) -> str:
        ci = index_of[role.class_label]
        base_mask = (best_class == ci) & (best_k == role.k_idx)
        labels = ctx["labels"]
        parts = []

        def segment(count: int, color: str, tooltip: str, uuids, title: str) -> None:
            segments.append((title, uuids.tolist()))
            parts.append(f'<button type="button" class="pr-seg" style="flex-grow:{max(count, 1)};background:{color}" '
                         f'title="{h.e(tooltip)}"{self.a("segment", n=len(segments) - 1)}></button>')

        if role.total == 0:
            parts.append(f'<span class="pr-seg" style="flex-grow:1;background:{REST_COLOR}"></span>')
        else:
            own_uuids = ctx["uuids"][base_mask & (labels == role.class_label)]
            segment(role.own_count, OWN_COLOR, f"{role.class_label} (own class): {role.own_count}", own_uuids,
                    f"{role.class_label} — prototype {role.k_idx} — own class")
            shown_others = {lbl for lbl, _ in role.other_counts}
            for i, (lbl, count) in enumerate(role.other_counts):
                other_uuids = ctx["uuids"][base_mask & (labels == lbl)]
                segment(count, OTHER_COLORS[i % len(OTHER_COLORS)], f"class {lbl}: {count}", other_uuids,
                        f"{role.class_label} — prototype {role.k_idx} — class {lbl}")
            rest_count = role.total - role.own_count - sum(c for _, c in role.other_counts)
            if rest_count > 0:
                rest_mask = base_mask & (labels != role.class_label) & (~np.isin(labels, list(shown_others)))
                segment(rest_count, REST_COLOR, f"other classes: {rest_count}", ctx["uuids"][rest_mask],
                        f"{role.class_label} — prototype {role.k_idx} — other classes")
        purity_text = f"{role.purity * 100:.0f}% pure · n={role.total}" if role.total else "n=0"
        return (f'<div class="pr-bar"><div class="pr-bar-h"><b>{h.e(f"{role.class_label} — subcenter {role.k_idx}")}</b>'
                f'<span class="muted">{h.e(purity_text)}</span></div>'
                f'<div class="pr-track">{"".join(parts)}</div></div>')

    # ---- neighbor explorer -------------------------------------------------

    def _compute_scope(self, res: dict, ctx: dict, exp_name: str) -> None:
        active = self.s.prototype_source.active_set(exp_name)
        options: list = []
        scope_map: dict = {}
        if active is not None:
            _source_key, proto_set, _color, _note = active

            def add(label: str, scope: tuple) -> None:
                # Positional keys, never the class label (see CLAUDE.md).
                key = str(len(scope_map))
                options.append((key, label))
                scope_map[key] = scope

            scope_classes = proto_set.class_labels
            if self.class_focus.many:
                present = set(proto_set.class_labels)
                scope_classes = [c for c in self._focus if c in present]
            index_of = proto_set.label_index()
            for cls in scope_classes:
                ci = index_of[cls]
                valid_ks = [k for k in range(proto_set.k) if not np.isnan(proto_set.vectors[ci, k]).any()]
                if not valid_ks:
                    continue
                add(f"{cls} — whole class", (cls, None))
                if proto_set.k > 1:
                    for k_idx in valid_ks:
                        add(f"{cls} — subcenter {k_idx}", (cls, k_idx))
        if self.scope_value not in scope_map:
            self.scope_value = options[0][0] if options else ""
        res["scope_options"] = options
        res["scope_map"] = scope_map

    # ---- compactness / separation (model only) -------------------------------

    def _compute_compactness(self, res: dict, ctx: dict, exp_name: str) -> None:
        exp = ctx["exp"]
        if exp.model_prototypes is None:
            res["compact_html"] = h.banner(
                "Compactness/separation and prototype confusion pairs need real, imported prototypes "
                "(they're scored diagnostics -- a kmeans estimate on the test set can't be used here, see the note above).",
                "info")
            return
        protos = exp.model_prototypes
        stats = self._cached("class_stats", lambda: ea.per_class_stats(ctx["embeddings"], ctx["labels"], protos),
                             ctx["embeddings"], ctx["labels"], protos)
        best_class, best_k = self.s.prototype_source.assignment(ctx, exp_name, "model", protos)

        def sort_key(s):
            margin = s.compactness - s.separation
            return margin if margin == margin else float("inf")

        stats_sorted = sorted(stats, key=sort_key)
        n_all_classes = len(stats_sorted)
        if self.class_focus.many:
            keep = set(self._focus)
            stats_sorted = [st for st in stats_sorted if str(st.label) in keep]
        class_labels = [s.label for s in stats_sorted]
        compactness = [s.compactness for s in stats_sorted]
        separation = [s.separation for s in stats_sorted]
        margins = [c - sep for c, sep in zip(compactness, separation)]
        n_classes = len(class_labels)
        x = np.arange(n_classes)
        width = max(6, n_classes * 0.5)

        caption = ("One row per class, worst margin first. These charts aggregate a class's subcenters; "
                   "the per-prototype table below does not."
                   + (f" {n_classes} of {n_all_classes:,} classes, picked above." if self.class_focus.many else ""))

        fig_bars, axes = subplots(1, 2, figsize=(width, 4.2))
        axes[0].bar(x, compactness, color="#4c72b0")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("compactness")
        axes[1].bar(x, separation, color="#d62728")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("separation (nearest other class)")
        for ax in axes:
            style_axes(ax, x=GRID_NONE)
        fig_bars.tight_layout()

        fig_margin, ax_m = subplots(figsize=(width, 3.2))
        ax_m.bar(x, margins, color=["#d62728" if m < 0 else "#4c72b0" for m in margins])
        ax_m.axhline(0, color="gray", linewidth=0.8)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
        ax_m.set_ylabel("margin (compactness - separation)")
        style_axes(ax_m, x=GRID_NONE)
        fig_margin.tight_layout()

        fig_scatter = self._build_compactness_scatter(class_labels, compactness, separation, margins)

        proto_stats = self._cached(
            "prototype_stats",
            lambda: ea.per_prototype_stats(ctx["embeddings"], ctx["labels"], protos, assignment=(best_class, best_k)),
            ctx["embeddings"], ctx["labels"], protos, best_class)
        proto_stats_sorted = sorted(proto_stats, key=sort_key)
        if self.class_focus.many:
            keep = set(self._focus)
            proto_stats_sorted = [st for st in proto_stats_sorted if str(st.class_label) in keep]
        proto_table_truncated = len(proto_stats_sorted) > MAX_PROTOTYPE_TABLE_ROWS
        proto_stats_shown = proto_stats_sorted[:MAX_PROTOTYPE_TABLE_ROWS]
        rows = "".join(
            f'<tr><td>{h.e(f"{s.class_label}#{s.k_idx}")}</td><td class="num">{s.n_assigned:,}</td>'
            f'<td class="num">{_fmt(s.compactness, ".3f")}</td><td class="num">{_fmt(s.separation, ".3f")}</td>'
            f'<td class="num">{_fmt(s.compactness - s.separation, "+.3f") if s.compactness == s.compactness and s.separation == s.separation else "—"}</td>'
            f'<td class="num">{f"{s.purity * 100:.0f}%" if s.purity == s.purity else "—"}</td></tr>'
            for s in proto_stats_shown)
        table = self._table(["prototype", "n assigned", "compactness", "separation", "margin", "purity"], rows,
                            numeric_from=1)

        pairs = ea.prototype_confusion_pairs(protos, top_n=12)
        pair_rows = "".join(f'<tr><td>{h.e(p["a"])}</td><td>{h.e(p["b"])}</td><td class="num">{p["sim"]:.3f}</td></tr>'
                            for p in pairs)
        pair_table = self._table(["class A", "class B", "similarity"], pair_rows, numeric_from=2)

        try:
            matrix_n = max(2, min(int(self.matrix_n), n_classes))
        except ValueError:
            matrix_n = min(31, n_classes)
        requested_n = matrix_n
        k_per_class = max(1, protos.k)
        max_classes_in_grid = max(2, MAX_GRID_PROTOTYPES // k_per_class)
        capped_by_k = matrix_n > max_classes_in_grid
        matrix_n = min(matrix_n, max_classes_in_grid)
        show_settings = not self.class_focus.many
        if self.class_focus.many:
            matrix_n = min(len(class_labels), max_classes_in_grid)
            capped_by_k = len(class_labels) > max_classes_in_grid
            requested_n = len(class_labels)
            matrix_classes = class_labels[:matrix_n]
            order_note = "picked above, lowest margin first"
        elif self.matrix_order == "alpha":
            matrix_classes = sorted(class_labels)[:matrix_n]
            order_note = "in alphabetical order"
        else:
            matrix_classes = class_labels[:matrix_n]
            order_note = "sorted by ascending margin (most at risk first)"
        assign_rows, assign_cols, assign_matrix = ea.prototype_assignment_matrix(
            ctx["embeddings"], ctx["labels"], protos, class_order=matrix_classes, assignment=(best_class, best_k),
        )
        idx_by_label = {label: i for i, label in enumerate(protos.class_labels)}
        res["assign"] = (list(assign_rows), list(assign_cols), idx_by_label, best_class, best_k)
        assignment_grid = self._assignment_grid(assign_rows, assign_cols, assign_matrix)
        sim_labels_flat, sim_matrix = ea.prototype_similarity_matrix(protos, class_order=matrix_classes)
        similarity_grid = self._similarity_grid(sim_labels_flat, sim_matrix)

        chart_w = f"{max(600, n_classes * 55)}px"
        matrix_caption = (
            f"{matrix_n} classes {order_note}, all subcenters as columns. "
            f"Click a cell to open those images."
            + (
                f" Limited to {matrix_n} classes here (not the {requested_n} asked for): "
                f"this experiment has {k_per_class} prototypes per class, so that is already "
                f"{matrix_n * k_per_class} columns -- both matrices below are per prototype, not per class."
                if capped_by_k else ""
            ))
        proto_caption = ("One row per subcenter, not per class."
                         + (f" The {len(proto_stats_shown):,} worst margins out of {len(proto_stats_sorted):,} "
                            f"prototypes; the rest are above them and left out." if proto_table_truncated else ""))
        settings_btn = (h.button(icon_name="sliders", act_attr=self.a("matrix_settings"), small=True, kind="ghost",
                                 title="Matrix settings") if show_settings else "")
        body = (
            f'<div class="muted small">{h.e(caption)}</div>'
            f'{fig_html(self.s, fig_bars, "Compactness & separation per class", max_width=f"min(100%,{chart_w})")}'
            f'<div class="muted small">Margin per class (negative = the nearest other class\'s prototype is as close as its own)</div>'
            f'{fig_html(self.s, fig_margin, "Margin per class", max_width=f"min(100%,{chart_w})")}'
            f'{fig_html(self.s, fig_scatter, "Compactness vs separation", max_width="min(100%,460px)")}'
            f'{self._subhead("Per-prototype detail", "compactness-separation")}'
            f'<div class="muted small">{h.e(proto_caption)}</div>{table}'
            f'{self._subhead("Most confused class pairs (averaged over subcenters)", "prototype-similarity")}{pair_table}'
            f'{self._subhead("Assignment matrix (class → prototype)", "assignment-matrix", settings_btn)}'
            f'<div class="muted small">{h.e(matrix_caption)}</div>{assignment_grid}'
            f'{self._subhead("Raw proximity matrix between prototypes", "prototype-similarity")}'
            f'<div class="muted small">No click here: a similarity between two prototype vectors has no images behind it.</div>'
            f'{similarity_grid}'
        )
        res["compact_html"] = h.section("Compactness & separation (real prototypes)",
                                        f'<div class="pr-compact">{body}</div>', manual="compactness-separation")

    @staticmethod
    def _subhead(title: str, manual: str, extra: str = "") -> str:
        return f'<div class="subhead pr-sub"><span>{h.e(title)}</span>{extra}<span class="grow"></span>{h.info_btn(manual)}</div>'

    @staticmethod
    def _table(columns: list, rows: str, numeric_from: int) -> str:
        head = "".join(f'<th{" class=num" if i >= numeric_from else ""}>{h.e(c)}</th>' for i, c in enumerate(columns))
        return (f'<div class="tbl-wrap pr-tbl"><table class="tbl compact sticky"><thead><tr>{head}</tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')

    def _build_compactness_scatter(self, class_labels, compactness, separation, margins):
        """Same as the original: fixed absolute similarity scale, fixed
        colour scale centred on 0."""
        fig, ax = subplots(figsize=(5.6, 5.6))
        sc = ax.scatter(
            separation, compactness, s=22, c=margins, cmap="RdYlGn",
            vmin=-MARGIN_COLOR_LIMIT, vmax=MARGIN_COLOR_LIMIT,
            edgecolors="black", linewidths=0.3, zorder=3,
        )
        if len(class_labels) <= 30:
            for lbl, sep, comp in zip(class_labels, separation, compactness):
                ax.annotate(lbl, (sep, comp), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        lo, hi = SIMILARITY_AXIS_LIMITS
        values = list(compactness) + list(separation)
        finite = [v for v in values if v == v]
        if finite:
            lo = min(lo, min(finite) - 0.02)
            hi = max(hi, max(finite) + 0.02)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("separation (nearest other class)")
        ax.set_ylabel("compactness")
        style_axes(ax)
        fig.colorbar(sc, ax=ax, label="margin (compactness - separation)", fraction=0.046, pad=0.04, extend="both")
        fig.tight_layout()
        return fig

    def _assignment_grid(self, row_labels: list, col_labels: list, matrix: np.ndarray) -> str:
        if not row_labels or not col_labels:
            return h.banner("No valid prototype to show.", "info")
        max_val = int(matrix.max()) if matrix.size else 1
        head = [f'<span class="pr-lab" style="width:{GRID_LABEL_WIDTH}px"></span>']
        for cls, k in col_labels:
            head.append(f'<span class="pr-colh" title="{h.e(f"{cls}#{k}")}">{h.e(f"{str(cls)[:6]}#{k}")}</span>')
        lines = [f'<div class="pr-grow">{"".join(head)}</div>']
        for i, row_label in enumerate(row_labels):
            cells = [f'<span class="pr-lab pr-lab-b" style="width:{GRID_LABEL_WIDTH}px">{h.e(row_label)}</span>']
            for j, (col_cls, col_k) in enumerate(col_labels):
                count = int(matrix[i, j])
                fraction = count / max_val if max_val else 0.0
                fg = "#ffffff" if fraction > 0.55 else "#111111"
                tip = f"class = {row_label}\nprototype = {col_cls}#{col_k}\nn = {count}"
                cells.append(f'<button type="button" class="pr-cell" style="background:{blues_color(fraction)};color:{fg}" '
                             f'title="{h.e(tip)}"{self.a("cell", i=i, j=j)}>{count}</button>')
            lines.append(f'<div class="pr-grow">{"".join(cells)}</div>')
        return f'<div class="pr-grid">{"".join(lines)}</div>'

    def _similarity_grid(self, labels_flat: list, sim_matrix: np.ndarray) -> str:
        n = len(labels_flat)
        if n == 0:
            return h.banner("No valid prototype to show.", "info")
        off_diag = np.array([sim_matrix[i, j] for i in range(n) for j in range(n) if i != j])
        vmin, vmax = (float(off_diag.min()), float(off_diag.max())) if off_diag.size else (-1.0, 1.0)
        span = (vmax - vmin) or 1.0

        def gap(idx: int) -> int:
            """Extra spacing before a row/column that starts a new class's block."""
            return 4 if idx > 0 and labels_flat[idx][0] != labels_flat[idx - 1][0] else 0

        head = [f'<span class="pr-lab" style="width:{GRID_LABEL_WIDTH}px"></span>']
        for j, (cls, k) in enumerate(labels_flat):
            head.append(f'<span class="pr-colh" style="margin-left:{gap(j)}px" title="{h.e(f"{cls}#{k}")}">'
                        f'{h.e(f"{str(cls)[:6]}#{k}")}</span>')
        lines = [f'<div class="pr-grow">{"".join(head)}</div>']
        for i, (row_cls, row_k) in enumerate(labels_flat):
            cells = [f'<span class="pr-lab pr-lab-b" style="width:{GRID_LABEL_WIDTH}px;font-size:9px">'
                     f'{h.e(f"{str(row_cls)[:10]}#{row_k}")}</span>']
            for j, (col_cls, col_k) in enumerate(labels_flat):
                if i == j:
                    bg, fg, text = "#dcdcdc", "#888888", "—"
                else:
                    fraction = max(0.0, min(1.0, (sim_matrix[i, j] - vmin) / span))
                    bg = blues_color(fraction)
                    fg = "#ffffff" if fraction > 0.55 else "#111111"
                    text = f"{sim_matrix[i, j]:.2f}"
                tip = f"{row_cls}#{row_k} · {col_cls}#{col_k}\nsimilarity = {sim_matrix[i, j]:.3f}"
                cells.append(f'<span class="pr-cell pr-sim" style="background:{bg};color:{fg};margin-left:{gap(j)}px" '
                             f'title="{h.e(tip)}">{text}</span>')
            lines.append(f'<div class="pr-grow" style="margin-top:{gap(i)}px">{"".join(cells)}</div>')
        return f'<div class="pr-grid">{"".join(lines)}</div>'

    # ---- rendering -----------------------------------------------------------

    def render(self) -> str:
        state = self.state
        self.on_shown()
        intro = ('<div class="muted small">Needs an embeddings sidecar on a loaded experiment, plus the model\'s own '
                 'prototypes for the scored diagnostics -- both attach in the Data tab.</div>')
        head = h.section("Prototype-space analysis", intro + self._sidecars_html(), manual="prototypes")
        parts = [self.stale_html(), head]
        res = self.result
        if state.dataset is None or not experiments_with_embeddings(state):
            parts.append(h.banner(
                "Attach an embeddings .npz sidecar to at least one loaded experiment in the Data tab "
                "(uuid + embeddings arrays) to unlock this tab.", "info"))
            return f'<div class="stack pr">{"".join(p for p in parts if p)}</div>'
        parts.append(self.progress_html())
        if res.get("note"):
            parts.append(h.banner(res["note"], "warn"))
        parts.append(self.class_focus.render())
        names = experiments_with_embeddings(state)
        exp_select = h.select([(n, n) for n in names], self.exp_name, change=self.c("experiment"), label="Experiment")
        parts.append(f'<div class="row">{exp_select}</div>')
        if "ctx" not in res:
            if not res and not self.progress_text:
                parts.append(h.spinner("Computing..."))
            return f'<div class="stack pr">{"".join(p for p in parts if p)}</div>'
        if res.get("error"):
            parts.append(h.banner(res["error"], "error"))
            return f'<div class="stack pr">{"".join(p for p in parts if p)}</div>'
        parts.append(res.get("role_html", ""))
        parts.append(self._cases_section())
        parts.append(res.get("compact_html", ""))
        return f'<div class="stack pr">{"".join(p for p in parts if p)}</div>'

    def _sidecars_html(self) -> str:
        experiments = self.state.experiments
        if not experiments:
            return ""
        rows = []
        for name, exp in experiments.items():
            embed_line = (
                f"Embeddings: {len(exp.embeddings_uuids):,} loaded ({exp.embeddings_path})" if exp.has_embeddings()
                else "Embeddings: not loaded -- set the path in the Data tab (auto-detected next to the results CSV)"
            )
            proto_line = (
                f"Model prototypes: {len(exp.model_prototypes.class_labels):,} classes ({exp.model_prototypes_path})"
                if exp.model_prototypes is not None
                else "Model prototypes: not loaded -- set the path in the Data tab (auto-detected next to the results CSV)"
            )
            k_value = self.kmeans_k.get(name, DEFAULT_K)
            kmeans = (f'<form class="row-nw"{h.on_submit(self.name, "kmeans", exp=name)}>'
                      f'{h.text_input(k_value, name="k", label="K", width="52px", change=self.c("k", exp=name))}'
                      f'<button type="submit" class="btn btn-sm"{h.attrs(disabled=self.progress_text != "" or None)}>'
                      f'{h.icon("target", 14)}<span>Estimate with k-means</span></button></form>')
            rows.append(
                f'<tr><td><b>{h.name(name)}</b></td>'
                f'<td><div class="small muted ell">{h.e(embed_line)}</div><div class="small muted ell">{h.e(proto_line)}</div></td>'
                f'<td>{kmeans}</td>'
                f'<td class="small muted">{h.e(self.s.prototype_source.status_text(name))}</td></tr>')
        return ('<div class="muted small" style="margin-top:6px">Prototype source -- pick one, whichever you use '
                'last is the active one:</div>'
                f'<div class="tbl-wrap"><table class="tbl compact pr-side"><tbody>{"".join(rows)}</tbody></table></div>')

    def _cases_section(self) -> str:
        res = self.result
        options = res.get("scope_options", [])
        line = ("Pick a class -- as a whole, or one specific prototype -- then run the display."
                + (" Only the classes picked above are listed." if self.class_focus.many else ""))
        controls = (f'<div class="row">{h.select(options, self.scope_value, change=self.c("scope"), label="Class / prototype")}'
                    f'{h.button("Show cases", act_attr=self.a("show_cases"), kind="primary", icon_name="image", small=True, disabled=not options or self.cases_busy)}</div>')
        if not options:
            block = h.banner("No prototype available -- import real prototypes or estimate them with kmeans above.", "info")
        elif self.cases_busy:
            block = h.spinner("Finding the cases...")
        else:
            block = self._cases_html()
        return h.section("Neighbor explorer", f'<div class="muted small">{h.e(line)}</div>{controls}{block}',
                         manual="case-studies")

    def _cases_html(self) -> str:
        cases = self.cases
        if not cases:
            return ""
        if cases.get("error"):
            return h.banner(cases["error"], "error")
        out = [f'<div class="pr-cases-t">Cases for {h.e(cases["suffix"])}</div>']
        for gi, (caption, title, items) in enumerate(cases["groups"]):
            out.append(f'<div class="muted small">{h.e(caption)}</div>{self._thumb_row(cases["ctx"], gi, items)}')
        return f'<div class="pr-cases">{"".join(out)}</div>'

    def _thumb_row(self, ctx: dict, gi: int, items: list) -> str:
        dataset = self.state.dataset
        cells = []
        for ui, (u, caption) in enumerate(items):
            rec = ctx["row_by_uuid"].get(u)
            if rec is None or dataset is None:
                continue
            img_path = dataset.resolve_crop_path(rec[ctx["crop_col"]])
            img = crop_img(self.s, img_path, rec, size=THUMB, fit="cover")
            cells.append(f'<button type="button" class="pr-thumb" title="{h.e(caption)}"{self.a("thumb", g=gi, u=ui)}>'
                         f'{img}<span>{h.e(caption)}</span></button>')
        if not cells:
            return '<div class="muted small"><i>(none)</i></div>'
        return f'<div class="pr-thumbs">{"".join(cells)}</div>'

    # ---- drill-downs ---------------------------------------------------------

    def _open_carousel(self, title: str, uuids: list, ctx: dict, highlight=None) -> None:
        records = [ctx["row_by_uuid"][u] for u in uuids if u in ctx["row_by_uuid"]]
        open_carousel(self.s, title, records, ctx["crop_col"], ctx["uuid_col"], highlight_uuid=highlight,
                      review_source="prototypes", label_col=self.state.dataset.columns.label_col)

    def on_segment(self, n: int = 0, **_):
        segments = self.result.get("segments") or []
        ctx = self.result.get("ctx")
        if ctx is None or not (0 <= int(n) < len(segments)):
            return
        title, uuids = segments[int(n)]
        self._open_carousel(f"{title} · {len(uuids)} image(s)", uuids, ctx)

    def on_cell(self, i: int = 0, j: int = 0, **_):
        assign = self.result.get("assign")
        ctx = self.result.get("ctx")
        if assign is None or ctx is None:
            return
        row_labels, col_labels, idx_by_label, best_class, best_k = assign
        i, j = int(i), int(j)
        if not (0 <= i < len(row_labels) and 0 <= j < len(col_labels)):
            return
        row_label = row_labels[i]
        col_cls, col_k = col_labels[j]
        mask = (ctx["labels"] == row_label) & (best_class == idx_by_label[col_cls]) & (best_k == col_k)
        uuids = ctx["uuids"][mask].tolist()
        self._open_carousel(f"{row_label} → prototype {col_cls}#{col_k} · {len(uuids)} image(s)", uuids, ctx)

    def on_thumb(self, g: int = 0, u: int = 0, **_):
        cases = self.cases
        if not cases or cases.get("error"):
            return
        g, u = int(g), int(u)
        if not (0 <= g < len(cases["groups"])):
            return
        _caption, title, items = cases["groups"][g]
        if not (0 <= u < len(items)):
            return
        self._open_carousel(title, [x for x, _ in items], cases["ctx"], highlight=items[u][0])

    # ---- actions -------------------------------------------------------------

    def on_experiment(self, value: str = "", **_):
        self.exp_name = value
        self.rebuild()

    def on_focus(self, op: str = "", **args):
        self.class_focus.handle(op, args)

    def on_scope(self, value: str = "", **_):
        self.scope_value = value

    def on_k(self, exp: str = "", value: str = "", **_):
        self.kmeans_k[exp] = str(value)

    def on_kmeans(self, exp: str = "", k: str = "", **_):
        self.kmeans_k[exp] = str(k)
        try:
            k_int = int(str(k))
        except ValueError:
            return
        registry = self.s.prototype_source

        def work():
            self.set_progress(f"Estimating {k_int} prototypes per class with k-means for '{exp}'...")
            self._own_source_change = True
            try:
                registry.compute_kmeans(exp, k_int)
            finally:
                self._own_source_change = False
            self.compute()

        self.built = True
        self.rebuild(work)

    def on_matrix_settings(self, **_):
        self.s.open_dialog(MatrixSettingsDialog(self))

    def on_show_cases(self, **_):
        exp_name = self.exp_name
        scope_map = self.result.get("scope_map") or {}
        scope_value = self.scope_value
        if not exp_name or scope_value not in scope_map:
            return
        cls, k_idx = scope_map[scope_value]
        self.cases_busy = True

        def work():
            try:
                ctx = build_analysis_context(self.state, exp_name)
                active = self.s.prototype_source.active_set(exp_name) if ctx is not None else None
                if active is None:
                    return
                source_key, proto_set, _color, _note = active
                scope_label = "whole class" if k_idx is None else f"subcenter {k_idx}"
                suffix = f"{cls} — {source_key} — {scope_label}"
                nearest, farthest = ea.nearest_farthest_within_class(
                    ctx["embeddings"], ctx["labels"], ctx["uuids"], proto_set, cls, k_top=8, k_idx=k_idx)
                impostors = ea.nearest_impostors(
                    ctx["embeddings"], ctx["labels"], ctx["uuids"], proto_set, cls, k_top=8, k_idx=k_idx)
                self.cases = {"ctx": ctx, "suffix": suffix, "groups": [
                    ("Nearest to prototype (representative)", f"Nearest — {suffix}",
                     [(u, f"{s:.3f} · {cls}#{k}") for u, s, k in nearest]),
                    ("Farthest from prototype (atypical)", f"Farthest — {suffix}",
                     [(u, f"{s:.3f} · {cls}#{k}") for u, s, k in farthest]),
                    ("Nearest impostors (other identities)", f"Impostors — {suffix}",
                     [(u, f"{s:.3f} · {l} → {cls}#{k}") for u, l, s, k in impostors]),
                ]}
            except Exception as ex:  # noqa: BLE001
                traceback.print_exc()
                self.cases = {"error": f"Error: {ex}"}
            finally:
                self.cases_busy = False
                self.refresh()

        self.s.job(f"{self.title} cases").run(work)
