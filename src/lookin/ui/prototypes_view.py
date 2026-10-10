"""Tab: prototype-space diagnostics for experiments that have a raw-feature
sidecar attached -- prototype role/composition, nearest/farthest/impostor
case studies, and compactness/separation with the prototype confusion
matrix. Every panel here needs an active prototype source; the 2D
projection, which needs only the embeddings, lives in its own tab (see
ui/projection_view.py), and dataset cleaning in a third (see
ui/cleaning_view.py).

Functionally the same diagnostics as arcface-suite's report.html /
explorer.html, built directly into the app instead of exported as static
files, and computed against whatever the user attaches rather than
requiring a specific model or checkpoint.

Two prototype sources, never conflated (see core/embedding_analysis.py):
"model" (imported, real -- the only source used for any *scored* number)
and "kmeans" (estimated in-tool from the loaded pool, which is always the
test split -- visual/comparative use only, kept out of compactness/purity/
cleaning wherever it's used). Exactly one is *active* per experiment at a
time -- kmeans is a fallback for when no real prototypes are available, not
a second thing to compare side by side. Which one that is, and everything
cached from it, is owned by ui/prototype_source.py rather than by this
view, because the projection tab draws the active prototypes too and the
Data tab and MLflow panel both attach the sidecars that invalidate it.
"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import flet as ft

from lookin.core import embedding_analysis as ea
from lookin.ui.carousel import build_carousel_from_list
from lookin.core import class_focus as cf
from lookin.ui.class_matrix import ClassFocus
from lookin.ui.components import info_button, fig_to_zoomable_image, section_title, info_banner, card, crop_display, file_to_full_base64, blues_color, style_axes, GRID_NONE
from lookin.ui.embedding_context import build_analysis_context, coverage_note, experiments_with_embeddings
from lookin.core.chart_style import subplots

OWN_COLOR = "#3b7d5c"
OTHER_COLORS = ["#c9762f", "#b0472e", "#a04ec9"]
REST_COLOR = "#9aa0a6"
THUMB = 84
GRID_CELL_SIZE = 34
GRID_LABEL_WIDTH = 110
# Both grids below are per *prototype*, not per class: a class with K
# subcenters contributes K columns, and the proximity matrix is prototypes x
# prototypes. So "Classes in the matrix" alone does not bound their size --
# at K=8 the default 31 classes means a 248x248 grid, i.e. ~61k clickable
# cells in one update, which does not render at all (the web client stalls
# before drawing anything). The real budget is a number of prototypes per
# axis, and the class count is derived from it; the setting stays a class
# count because that is what the user picks classes by.
# Which classes the panels show past FOCUS_MAX (see ClassFocus).
PROTOTYPE_FOCUS_CRITERIA = (
    (cf.FOCUS_WORST_MARGIN, "Worst margin"),
    (cf.FOCUS_LEAST_PURE, "Least pure prototypes"),
    (cf.FOCUS_LARGEST_SUPPORT, "Largest support"),
    (cf.FOCUS_MANUAL, "Hand-picked"),
)
MAX_GRID_PROTOTYPES = 64
# One bar per (class, subcenter) in the role panel -- linear in K rather
# than quadratic, so the cap is much higher, but it still needs one: a
# thousand classes x 8 subcenters is 8,000 bars with clickable segments.
MAX_ROLE_BARS = 600
# Same shape of problem for the per-prototype table, which is sorted worst
# margin first -- so the tail is the part that can be dropped.
MAX_PROTOTYPE_TABLE_ROWS = 400
# Compactness and separation are both cosine similarities, so the
# compactness-vs-separation scatter gets a fixed, absolute scale instead of
# one fitted to whatever happens to be loaded -- otherwise a compactness
# range of 0.47-0.50 fills the whole plot height and looks like a huge
# spread, and no two experiments can be compared by eye.
SIMILARITY_AXIS_LIMITS = (0.0, 1.0)
# ...and the margin color scale is fixed and symmetric around 0, so red
# always means "negative margin" (the nearest other class is closer than
# the class's own prototype) rather than merely "the lowest value present".
# Margins beyond this are clipped, which the colorbar flags with arrows.
MARGIN_COLOR_LIMIT = 0.5


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class PrototypesView:
    def __init__(self, ctx):
        self.ctx = ctx
        # Which source is active per experiment, the kmeans estimates and
        # the nearest-prototype assignments all live in ctx.prototype_source
        # (see ui/prototype_source.py) -- this view only holds the controls
        # that display and drive them.
        self._kmeans_fields: dict = {}  # exp name -> K TextField
        self._source_status_texts: dict = {}  # exp name -> Text ("Active source: ...")
        self._case_study_scope_map: dict = {}  # dropdown option value -> (class label, k_idx | None)

        self.experiment_dropdown = ft.Dropdown(label="Experiment", width=260, on_change=self._on_experiment_change, border_radius=8)
        # No class picker: every panel below now covers all classes at once
        # (see _render_role / _rebuild_case_study_scope), so the only thing
        # left to choose is which experiment.
        self.case_study_scope_dropdown = ft.Dropdown(label="Class / prototype", width=340, border_radius=8)
        self.case_study_btn = ft.FilledButton("Show cases", icon=ft.Icons.PHOTO_LIBRARY_OUTLINED, on_click=self._on_show_case_studies)

        # Past FOCUS_MAX classes every per-class panel below -- the role
        # bars, the case-study picker, the per-class charts and both
        # prototype grids -- shows one focus set, picked here.
        self.class_focus = ClassFocus(PROTOTYPE_FOCUS_CRITERIA, self._on_focus_change,
                                      what="the panels below",
                                      support_criteria=(cf.FOCUS_WORST_MARGIN, cf.FOCUS_LEAST_PURE))
        self._focus: list = []
        # Per-class statistics, kept across renders (see _cached): changing
        # the focus set redraws the panels but recomputes none of these.
        self._memo: dict = {}

        self.sidecars_column = ft.Column(spacing=10)
        self.role_area = ft.Column(spacing=10)
        self.case_study_area = ft.Column(spacing=10)
        self.compactness_area = ft.Container()
        self.warnings_area = ft.Column(spacing=6)

        self.matrix_n_field = ft.TextField(label="Classes in the matrix", value="31", width=200, border_radius=8)
        self.matrix_order_dropdown = ft.Dropdown(
            label="Class order", value="margin", width=260, border_radius=8,
            options=[
                ft.dropdown.Option("margin", "Lowest margin first"),
                ft.dropdown.Option("alpha", "Normal order (alphabetical)"),
            ],
        )
        self.matrix_settings_btn = ft.IconButton(ft.Icons.SETTINGS_OUTLINED, tooltip="Matrix settings", on_click=self._open_matrix_settings)
        self.matrix_settings_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Proximity matrix settings"),
            content=ft.Column([self.matrix_n_field, self.matrix_order_dropdown], spacing=12, tight=True),
            actions=[
                ft.TextButton("Apply", on_click=self._on_apply_matrix_settings),
                ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.matrix_settings_dialog)),
            ],
        )

        self.carousel_title = ft.Text(weight=ft.FontWeight.BOLD)
        self.carousel_content = ft.Container()  # width comes from the carousel, which sizes itself to the page
        self.carousel_dialog = ft.AlertDialog(
            # modal=False so tapping outside dismisses; on_dismiss then has
            # to sync `open` back to False, which Flet does not do itself --
            # a stale True makes the next page.open() a no-op.
            modal=False, title=self.carousel_title, content=self.carousel_content,
            on_dismiss=lambda e: self.ctx.page.close(self.carousel_dialog),
            actions=[ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.carousel_dialog))],
        )

        self.root = ft.Container(padding=20)

    # ---- helpers -----------------------------------------------------

    def _analysis_context(self, exp_name: str):
        return build_analysis_context(self.ctx.state, exp_name)

    def _open_carousel(self, title: str, uuids: list, ctx: dict, highlight=None):
        records = [ctx["row_by_uuid"][u] for u in uuids if u in ctx["row_by_uuid"]]
        self.carousel_title.value = title
        self.carousel_content.content = build_carousel_from_list(
            self.ctx.state.dataset, records, ctx["crop_col"], ctx["uuid_col"], highlight_uuid=highlight,
            ctx=self.ctx, review_source="prototypes",
            label_col=self.ctx.state.dataset.columns.label_col,
        )
        self.ctx.page.open(self.carousel_dialog)

    # ---- sidecar loaders -----------------------------------------------

    def _rebuild_sidecar_rows(self):
        self.sidecars_column.controls.clear()
        for name, exp in self.ctx.state.experiments.items():
            embed_line = (
                f"Embeddings: {len(exp.embeddings_uuids):,} loaded ({exp.embeddings_path})" if exp.has_embeddings()
                else "Embeddings: not loaded -- set the path in the Data tab (auto-detected next to the results CSV)"
            )
            proto_line = (
                f"Model prototypes: {len(exp.model_prototypes.class_labels):,} classes ({exp.model_prototypes_path})"
                if exp.model_prototypes is not None
                else "Model prototypes: not loaded -- set the path in the Data tab (auto-detected next to the results CSV)"
            )
            kmeans_field = ft.TextField(label="K", value="2", width=70, border_radius=8, dense=True)
            self._kmeans_fields[name] = kmeans_field
            source_status = ft.Text(self.ctx.prototype_source.status_text(name), size=11, color=ft.Colors.ON_SURFACE_VARIANT)
            self._source_status_texts[name] = source_status

            self.sidecars_column.controls.append(card(ft.Column([
                ft.Text(name, weight=ft.FontWeight.BOLD),
                ft.Text(embed_line, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(proto_line, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text("Prototype source -- pick one, whichever you use last is the active one:", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Row([kmeans_field, ft.OutlinedButton("Estimate with k-means", icon=ft.Icons.SCATTER_PLOT, on_click=lambda e, n=name: self._on_compute_kmeans(n))]),
                source_status,
            ], spacing=6)))

    def _refresh_source_status(self, name: str) -> None:
        t = self._source_status_texts.get(name)
        if t is None:
            return
        t.value = self.ctx.prototype_source.status_text(name)
        if t.page:
            t.update()

    def _on_compute_kmeans(self, name: str):
        try:
            k = int(self._kmeans_fields[name].value)
        except ValueError:
            return
        # The registry runs the estimate, caches it and makes it the active
        # source; its subscribers (this view and the projection tab) are
        # what re-render afterwards, so nothing is refreshed here.
        self.ctx.prototype_source.compute_kmeans(name, k)

    def _refresh_experiment_dropdown(self):
        names = experiments_with_embeddings(self.ctx.state)
        self.experiment_dropdown.options = [ft.dropdown.Option(n) for n in names]
        if self.experiment_dropdown.value not in names:
            self.experiment_dropdown.value = names[0] if names else None

    # ---- top-level render ------------------------------------------------

    def _on_experiment_change(self, e):
        self._render()
        if self.root.page:
            self.root.update()

    def _open_matrix_settings(self, e):
        self.ctx.page.open(self.matrix_settings_dialog)

    def _on_apply_matrix_settings(self, e):
        self.ctx.page.close(self.matrix_settings_dialog)
        self._render()
        if self.root.page:
            self.root.update()

    _on_change = _on_experiment_change  # matrix_n_field and any other plain "re-render" control share this

    def _render(self):
        state = self.ctx.state
        self.warnings_area.controls.clear()
        if state.dataset is None or not experiments_with_embeddings(state):
            self.role_area.controls = [info_banner(
                "Attach an embeddings .npz sidecar to at least one loaded experiment in the Data tab "
                "(uuid + embeddings arrays) to unlock this tab."
            )]
            self.case_study_area.controls = []
            self.compactness_area.content = None
            return

        if not self.experiment_dropdown.value:
            self._refresh_experiment_dropdown()

        exp_name = self.experiment_dropdown.value
        if not exp_name:
            return
        ctx = self._analysis_context(exp_name)
        if ctx is None:
            return

        note = coverage_note(ctx, state, exp_name)
        if note:
            self.warnings_area.controls.append(info_banner(note, icon=ft.Icons.WARNING_AMBER_OUTLINED))

        try:
            self._resolve_focus(ctx, exp_name)
            self._render_role(ctx, exp_name)
            self._rebuild_case_study_scope(ctx, exp_name)
            self._render_compactness(ctx, exp_name)
        except Exception as ex:
            _log(f"error rendering embeddings tab: {ex}")
            traceback.print_exc()
            self.role_area.controls = [info_banner(f"Error: {ex}", icon=ft.Icons.ERROR_OUTLINE)]

    # ---- focus set: which classes the panels show (see core/class_focus.py) ----
    def _resolve_focus(self, ctx: dict, exp_name: str) -> None:
        labels = ctx["labels"]
        classes = sorted({str(l) for l in labels})

        def support_table() -> pd.DataFrame:
            counts = pd.Series(labels).astype(str).value_counts()
            return pd.DataFrame({"label": counts.index, "support": counts.to_numpy()})

        def least_pure() -> list:
            active = self.ctx.prototype_source.active_set(exp_name)
            if active is None:
                return []
            source_key, proto_set, _color, _note = active
            best_class, best_k = self.ctx.prototype_source.assignment(ctx, exp_name, source_key, proto_set)
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
                # Margin needs the model's own prototypes; without them the
                # closest question an estimate can answer is purity.
                return least_pure()
            if criterion == cf.FOCUS_LARGEST_SUPPORT:
                return cf.rank_by(support_table(), "support", ascending=False)
            return []

        self._focus = self.class_focus.resolve(classes, rank)

    def _cached(self, kind: str, compute, *inputs):
        """compute(), remembered for these exact input objects -- compared
        by identity, so new data always recomputes. One result per kind:
        a slot per past input would keep every superseded embedding matrix
        alive."""
        hit = self._memo.get(kind)
        if hit is not None and len(hit[0]) == len(inputs) and all(a is b for a, b in zip(hit[0], inputs)):
            return hit[1]
        value = compute()
        self._memo[kind] = (inputs, value)
        return value

    def _on_focus_change(self):
        self._render()
        if self.root.page:
            self.root.update()

    # ---- prototype role / composition ------------------------------------

    def _render_role(self, ctx: dict, exp_name: str):
        """Every prototype of every class, not one class at a time: the
        underlying nearest-prototype assignment is global and computed once
        anyway (see PrototypeSourceRegistry.assignment), so restricting
        the display to a single picked class hid the comparison the panel
        exists to support.
        Bars flow in a wrapping row, grouped and ordered by class."""
        self.role_area.controls = [ft.Row([self.experiment_dropdown], wrap=True, spacing=16)]
        active = self.ctx.prototype_source.active_set(exp_name)
        if active is None:
            self.role_area.controls.append(info_banner(
                "No prototypes yet for this experiment -- import the model's real prototypes, "
                "or estimate them with kmeans, in the sidecar section above."
            ))
            return
        source_key, proto_set, color, note = active

        header = ft.Row([ft.Container(width=8, height=8, bgcolor=color, border_radius=4), ft.Text(note, size=12, weight=ft.FontWeight.BOLD)], spacing=8)
        best_class, best_k = self.ctx.prototype_source.assignment(ctx, exp_name, source_key, proto_set)
        composition = self._cached(
            "composition",
            lambda: ea.prototype_composition(ctx["embeddings"], ctx["labels"], proto_set, assignment=(best_class, best_k)),
            ctx["embeddings"], ctx["labels"], proto_set, best_class)

        # Count first, build second: one bar per (class, subcenter) means K
        # times as many controls as there are classes, and past a few
        # hundred the web client spends longer laying them out than the
        # whole load took. Truncation keeps the class grouping intact.
        role_classes = proto_set.class_labels
        if self.class_focus.many:
            present = set(proto_set.class_labels)
            role_classes = [c for c in self._focus if c in present]
        roles = [(cls, role) for cls in role_classes for role in composition.get(cls, [])]
        total_roles = len(roles)
        truncated = total_roles > MAX_ROLE_BARS
        bars = [self._build_role_bar(ctx, exp_name, source_key, proto_set, role)
                for _, role in roles[:MAX_ROLE_BARS]]
        if not bars:
            self.role_area.controls.append(card(ft.Column([header, ft.Text("No prototype to show.", size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)], spacing=8)))
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
        self.role_area.controls.append(card(ft.Column([
            section_title("Role of each prototype", icon=ft.Icons.DONUT_LARGE, ctx=self.ctx, manual="prototype-roles"),
            header,
            ft.Text(
                f"{count_text} Nearest-prototype assignment (cosine), "
                "over the prototypes of every class at once. Click a segment to open the matching images.",
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row(bars, wrap=True, spacing=8, run_spacing=8),
        ], spacing=10)))

    def _build_role_bar(self, ctx: dict, exp_name: str, source_key: str, proto_set, role) -> ft.Control:
        ci = proto_set.label_index()[role.class_label]
        # Already computed and cached by the caller's own assignment() call,
        # so this is a lookup, not a second O(N x C x K) pass.
        best_class, best_k = self.ctx.prototype_source.assignment(ctx, exp_name, source_key, proto_set)
        base_mask = (best_class == ci) & (best_k == role.k_idx)

        segments = []
        if role.total == 0:
            segments.append(ft.Container(bgcolor=REST_COLOR, height=16, expand=1))
        else:
            own_uuids = ctx["uuids"][base_mask & (ctx["labels"] == role.class_label)]
            segments.append(self._bar_segment(role.own_count, OWN_COLOR, f"{role.class_label} (own class): {role.own_count}", own_uuids, ctx, f"{role.class_label} — prototype {role.k_idx} — own class"))
            shown_others = {lbl for lbl, _ in role.other_counts}
            for i, (lbl, count) in enumerate(role.other_counts):
                other_uuids = ctx["uuids"][base_mask & (ctx["labels"] == lbl)]
                color = OTHER_COLORS[i % len(OTHER_COLORS)]
                segments.append(self._bar_segment(count, color, f"class {lbl}: {count}", other_uuids, ctx, f"{role.class_label} — prototype {role.k_idx} — class {lbl}"))
            rest_count = role.total - role.own_count - sum(c for _, c in role.other_counts)
            if rest_count > 0:
                rest_mask = base_mask & (ctx["labels"] != role.class_label) & (~np.isin(ctx["labels"], list(shown_others)))
                rest_uuids = ctx["uuids"][rest_mask]
                segments.append(self._bar_segment(rest_count, REST_COLOR, f"other classes: {rest_count}", rest_uuids, ctx, f"{role.class_label} — prototype {role.k_idx} — other classes"))

        purity_text = f"{role.purity * 100:.0f}% pure · n={role.total}" if role.total else "n=0"
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(f"{role.class_label} — subcenter {role.k_idx}", size=12, weight=ft.FontWeight.BOLD),
                    ft.Text(purity_text, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row(segments, spacing=1, height=16),
            ], spacing=4),
            padding=8, border_radius=8, border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT), width=340,
        )

    def _bar_segment(self, count: int, color: str, tooltip: str, uuids: np.ndarray, ctx: dict, title: str) -> ft.Control:
        return ft.Container(
            bgcolor=color, expand=max(count, 1), height=16, tooltip=tooltip, ink=True,
            on_click=lambda e, u=uuids.tolist(), t=title: self._open_carousel(f"{t} · {len(u)} image(s)", u, ctx),
        )

    # ---- case studies (nearest/farthest/impostors) -----------------------
    #
    # Deliberately *not* rendered automatically on every class change: the
    # point is to let the user pick exactly which class -- and, when a
    # source has more than one subcenter, exactly which single prototype --
    # to inspect, then press the button. Auto-rendering "all sources, all
    # subcenters merged" on every render (the previous design) buried that
    # choice instead of exposing it.

    def _rebuild_case_study_scope(self, ctx: dict, exp_name: str):
        """One dropdown covering every class rather than a class picker plus
        a scope picker: each class contributes a "whole class" entry and one
        entry per valid subcenter, so picking what to inspect is a single
        choice. The map's values carry the class along with the subcenter,
        since the class is no longer held in a separate control."""
        active = self.ctx.prototype_source.active_set(exp_name)
        options = []
        self._case_study_scope_map = {}
        if active is not None:
            source_key, proto_set, color, note = active

            def add(label: str, scope: tuple) -> None:
                # Keyed by position, not by a "<class><separator><k>" string:
                # class labels come from the user's own data and could contain
                # whatever separator we picked, silently colliding.
                key = str(len(self._case_study_scope_map))
                options.append(ft.dropdown.Option(key, label))
                self._case_study_scope_map[key] = scope

            scope_classes = proto_set.class_labels
            if self.class_focus.many:
                present = set(proto_set.class_labels)
                scope_classes = [c for c in self._focus if c in present]
            index_of = proto_set.label_index()
            for cls in scope_classes:
                ci = index_of[cls]
                valid_ks = [k for k in range(proto_set.k) if not np.isnan(proto_set.vectors[ci, k]).any()]
                if not valid_ks:
                    continue  # class absent from the pool the prototypes were built from
                add(f"{cls} — whole class", (cls, None))
                if proto_set.k > 1:
                    for k_idx in valid_ks:
                        add(f"{cls} — subcenter {k_idx}", (cls, k_idx))

        self.case_study_scope_dropdown.options = options
        valid_values = {o.key for o in options}
        if self.case_study_scope_dropdown.value not in valid_values:
            self.case_study_scope_dropdown.value = options[0].key if options else None
        self.case_study_btn.disabled = not options

        self.case_study_area.controls = [card(ft.Column([
            section_title("Neighbor explorer", icon=ft.Icons.PHOTO_LIBRARY_OUTLINED, ctx=self.ctx, manual="case-studies"),
            ft.Text(
                "Pick a class -- as a whole, or one specific prototype -- then run the display."
                + (" Only the classes picked above are listed." if self.class_focus.many else ""),
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([self.case_study_scope_dropdown, self.case_study_btn], spacing=12, wrap=True),
            ft.Container(),
        ], spacing=10))]
        if not options:
            self.case_study_area.controls[0].content.controls[-1] = info_banner(
                "No prototype available -- import real prototypes or estimate them with kmeans above."
            )

    def _on_show_case_studies(self, e):
        exp_name = self.experiment_dropdown.value
        scope_value = self.case_study_scope_dropdown.value
        if not exp_name or scope_value not in self._case_study_scope_map:
            return
        ctx = self._analysis_context(exp_name)
        if ctx is None:
            return
        active = self.ctx.prototype_source.active_set(exp_name)
        if active is None:
            return
        source_key, proto_set, color, note = active
        cls, k_idx = self._case_study_scope_map[scope_value]

        scope_label = "whole class" if k_idx is None else f"subcenter {k_idx}"
        title_suffix = f"{cls} — {source_key} — {scope_label}"

        nearest, farthest = ea.nearest_farthest_within_class(ctx["embeddings"], ctx["labels"], ctx["uuids"], proto_set, cls, k_top=8, k_idx=k_idx)
        impostors = ea.nearest_impostors(ctx["embeddings"], ctx["labels"], ctx["uuids"], proto_set, cls, k_top=8, k_idx=k_idx)

        block = ft.Column([
            ft.Text(f"Cases for {title_suffix}", size=13, weight=ft.FontWeight.BOLD),
            ft.Text("Nearest to prototype (representative)", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            self._thumb_row(ctx, [u for u, _, _ in nearest], [f"{s:.3f} · {cls}#{k}" for _, s, k in nearest], f"Nearest — {title_suffix}"),
            ft.Text("Farthest from prototype (atypical)", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            self._thumb_row(ctx, [u for u, _, _ in farthest], [f"{s:.3f} · {cls}#{k}" for _, s, k in farthest], f"Farthest — {title_suffix}"),
            ft.Text("Nearest impostors (other identities)", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            self._thumb_row(ctx, [u for u, _, _, _ in impostors], [f"{s:.3f} · {l} → {cls}#{k}" for _, l, s, k in impostors], f"Impostors — {title_suffix}"),
        ], spacing=6)

        self.case_study_area.controls[0].content.controls[-1] = block
        if self.case_study_area.page:
            self.case_study_area.update()

    def _thumb_row(self, ctx: dict, uuids: list, captions: list, title: str) -> ft.Control:
        cells = []
        for u, caption in zip(uuids, captions):
            rec = ctx["row_by_uuid"].get(u)
            if rec is None:
                continue
            img_path = self.ctx.state.dataset.resolve_crop_path(rec[ctx["crop_col"]])
            b64 = file_to_full_base64(img_path, *crop_display(self.ctx, rec))
            cells.append(ft.Container(
                content=ft.Column([
                    ft.Image(src_base64=b64, width=THUMB, height=THUMB, fit=ft.ImageFit.COVER, border_radius=6),
                    ft.Text(caption, size=9, text_align=ft.TextAlign.CENTER),
                ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                width=THUMB + 8, ink=True,
                on_click=lambda e, u=u: self._open_carousel(title, uuids, ctx, highlight=u),
            ))
        if not cells:
            return ft.Text("(none)", size=10, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
        return ft.Row(cells, spacing=6, wrap=True)

    # ---- compactness / separation / confusion pairs (model only) --------

    def _render_compactness(self, ctx: dict, exp_name: str):
        exp = ctx["exp"]
        if exp.model_prototypes is None:
            self.compactness_area.content = info_banner(
                "Compactness/separation and prototype confusion pairs need real, imported prototypes "
                "(they're scored diagnostics -- a kmeans estimate on the test set can't be used here, see the note above)."
            )
            return
        protos = exp.model_prototypes

        stats = self._cached("class_stats", lambda: ea.per_class_stats(ctx["embeddings"], ctx["labels"], protos),
                             ctx["embeddings"], ctx["labels"], protos)
        # Reused below by per_prototype_stats too -- same O(N x C x K)
        # computation the role panel already caches, no need to redo it.
        best_class, best_k = self.ctx.prototype_source.assignment(ctx, exp_name, "model", protos)

        def sort_key(s):
            margin = s.compactness - s.separation
            return margin if margin == margin else float("inf")  # NaN (no support) sorts last

        stats_sorted = sorted(stats, key=sort_key)
        n_all_classes = len(stats_sorted)
        if self.class_focus.many:
            # The focus set, still worst margin first -- one bar per class
            # stops being a chart well before a thousand.
            keep = set(self._focus)
            stats_sorted = [st for st in stats_sorted if str(st.label) in keep]
        class_labels = [s.label for s in stats_sorted]
        compactness = [s.compactness for s in stats_sorted]
        separation = [s.separation for s in stats_sorted]
        margins = [c - sep for c, sep in zip(compactness, separation)]
        n_classes = len(class_labels)
        x = np.arange(n_classes)
        width = max(6, n_classes * 0.5)

        section = section_title("Compactness & separation (real prototypes)", icon=ft.Icons.SCATTER_PLOT_OUTLINED, ctx=self.ctx, manual="compactness-separation")
        caption = ft.Text(
            "One row per class, worst margin first. These charts aggregate a class's subcenters; "
            "the per-prototype table below does not."
            + (f" {n_classes} of {n_all_classes:,} classes, picked above." if self.class_focus.many else ""),
            size=11, color=ft.Colors.ON_SURFACE_VARIANT,
        )

        fig_bars, axes = subplots(1, 2, figsize=(width, 4.2))
        axes[0].bar(x, compactness, color="#4c72b0")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("compactness")
        axes[1].bar(x, separation, color="#d62728")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("separation (nearest other class)")
        # One bar per class on x, a similarity on y: only the scale gets a
        # grid, and it gets both weights since the gap between two classes
        # here is routinely a hundredth.
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

        # One row per individual (class, subcenter) prototype -- not per
        # class -- specifically because the class-level numbers above blend
        # subcenters together and can hide a subcenter that's actually
        # loose, contaminated, or badly separated while its sibling is
        # fine. Sorted the same way (ascending margin) but at the
        # granularity that actually varies.
        proto_stats = self._cached(
            "prototype_stats",
            lambda: ea.per_prototype_stats(ctx["embeddings"], ctx["labels"], protos, assignment=(best_class, best_k)),
            ctx["embeddings"], ctx["labels"], protos, best_class)

        def proto_sort_key(s):
            margin = s.compactness - s.separation
            return margin if margin == margin else float("inf")

        # Also one row per (class, subcenter), so also K times the class
        # count -- kept to the worst margins, which is what the sort is for
        # and what this table is read for.
        proto_stats_sorted = sorted(proto_stats, key=proto_sort_key)
        if self.class_focus.many:
            keep = set(self._focus)
            proto_stats_sorted = [st for st in proto_stats_sorted if str(st.class_label) in keep]
        proto_table_truncated = len(proto_stats_sorted) > MAX_PROTOTYPE_TABLE_ROWS
        proto_stats_shown = proto_stats_sorted[:MAX_PROTOTYPE_TABLE_ROWS]
        table_rows = [ft.DataRow(cells=[
            ft.DataCell(ft.Text(f"{s.class_label}#{s.k_idx}")),
            ft.DataCell(ft.Text(f"{s.n_assigned:,}")),
            ft.DataCell(ft.Text(f"{s.compactness:.3f}" if s.compactness == s.compactness else "—")),
            ft.DataCell(ft.Text(f"{s.separation:.3f}" if s.separation == s.separation else "—")),
            ft.DataCell(ft.Text(f"{(s.compactness - s.separation):+.3f}" if s.compactness == s.compactness and s.separation == s.separation else "—")),
            ft.DataCell(ft.Text(f"{s.purity * 100:.0f}%" if s.purity == s.purity else "—")),
        ]) for s in proto_stats_shown]
        table = ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c)) for c in ["prototype", "n assigned", "compactness", "separation", "margin", "purity"]],
            rows=table_rows,
        )

        pairs = ea.prototype_confusion_pairs(protos, top_n=12)
        pair_rows = [ft.DataRow(cells=[
            ft.DataCell(ft.Text(str(p["a"]))), ft.DataCell(ft.Text(str(p["b"]))), ft.DataCell(ft.Text(f"{p['sim']:.3f}")),
        ]) for p in pairs]
        pair_table = ft.DataTable(columns=[ft.DataColumn(ft.Text(c)) for c in ["class A", "class B", "similarity"]], rows=pair_rows)

        try:
            matrix_n = max(2, min(int(self.matrix_n_field.value), n_classes))
        except ValueError:
            matrix_n = min(31, n_classes)
        requested_n = matrix_n
        # Derive the real cap from the prototype budget (see
        # MAX_GRID_PROTOTYPES): the columns are subcenters, so K multiplies
        # whatever class count is asked for.
        k_per_class = max(1, protos.k)
        max_classes_in_grid = max(2, MAX_GRID_PROTOTYPES // k_per_class)
        capped_by_k = matrix_n > max_classes_in_grid
        matrix_n = min(matrix_n, max_classes_in_grid)
        # The matrix settings pick classes among all of them; past
        # FOCUS_MAX the classes picked above do, so the settings go away.
        self.matrix_settings_btn.visible = not self.class_focus.many
        if self.class_focus.many:
            # The classes picked above, not the matrix settings: those rank
            # among every class, and the panels here should agree.
            matrix_n = min(len(class_labels), max_classes_in_grid)
            capped_by_k = len(class_labels) > max_classes_in_grid
            requested_n = len(class_labels)
            matrix_classes = class_labels[:matrix_n]
            order_note = "picked above, lowest margin first"
        elif self.matrix_order_dropdown.value == "alpha":
            matrix_classes = sorted(class_labels)[:matrix_n]
            order_note = "in alphabetical order"
        else:
            matrix_classes = class_labels[:matrix_n]  # class_labels already sorted ascending by margin
            order_note = "sorted by ascending margin (most at risk first)"
        assign_row_labels, assign_col_labels, assign_matrix = ea.prototype_assignment_matrix(
            ctx["embeddings"], ctx["labels"], protos, class_order=matrix_classes, assignment=(best_class, best_k),
        )
        assignment_grid = self._build_assignment_grid(ctx, assign_row_labels, assign_col_labels, assign_matrix, best_class, best_k)

        sim_labels_flat, sim_matrix = ea.prototype_similarity_matrix(protos, class_order=matrix_classes)
        similarity_grid = self._build_similarity_grid(sim_labels_flat, sim_matrix)

        self.compactness_area.content = card(ft.Column([
            section, caption,
            fig_to_zoomable_image(fig_bars, self.ctx, "Compactness & separation per class", width=max(600, n_classes * 55), height=310),
            ft.Text("Margin per class (negative = the nearest other class's prototype is as close as its own)", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            fig_to_zoomable_image(fig_margin, self.ctx, "Margin per class", width=max(600, n_classes * 55), height=240),
            # "separation" is the similarity to the *nearest other class*, so
            fig_to_zoomable_image(fig_scatter, self.ctx, "Compactness vs separation", width=460, height=460),
            ft.Row([ft.Text("Per-prototype detail", size=12, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    info_button(self.ctx, "compactness-separation")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(
                "One row per subcenter, not per class."
                + (
                    f" The {len(proto_stats_shown):,} worst margins out of {len(proto_stats_sorted):,} "
                    f"prototypes; the rest are above them and left out."
                    if proto_table_truncated else ""
                ),
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([table], scroll=ft.ScrollMode.ALWAYS),
            ft.Row([ft.Text("Most confused class pairs (averaged over subcenters)",
                            size=12, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    info_button(self.ctx, "prototype-similarity")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([pair_table], scroll=ft.ScrollMode.ALWAYS),
            ft.Row([
                ft.Text("Assignment matrix (class → prototype)", size=12, weight=ft.FontWeight.BOLD),
                self.matrix_settings_btn,
                ft.Container(expand=True),
                info_button(self.ctx, "assignment-matrix"),
            ], spacing=4, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(
                f"{matrix_n} classes {order_note}, all subcenters as columns. "
                f"Click a cell to open those images."
                + (
                    f" Limited to {matrix_n} classes here (not the {requested_n} asked for): "
                    f"this experiment has {k_per_class} prototypes per class, so that is already "
                    f"{matrix_n * k_per_class} columns -- both matrices below are per prototype, not per class."
                    if capped_by_k else ""
                ),
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            assignment_grid,
            ft.Row([ft.Text("Raw proximity matrix between prototypes", size=12, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    info_button(self.ctx, "prototype-similarity")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(
                "No click here: a similarity between two prototype vectors has no images behind it.",
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            similarity_grid,
        ], spacing=14))

    def _build_compactness_scatter(self, class_labels, compactness, separation, margins):
        """Both axes are cosine similarities, so the scale is absolute and
        fixed (see SIMILARITY_AXIS_LIMITS) rather than fitted to the data:
        with auto-scaled axes a compactness range as narrow as 0.47-0.50
        gets stretched over the full plot height, turning negligible
        differences into dramatic-looking spread and making two experiments
        impossible to compare side by side.

        The color scale is fixed and centered on 0 for the same reason. It
        used to be `c=margins, cmap="RdYlGn"` with matplotlib's default
        auto-scaling, which normalizes to the data's own min/max -- so the
        lowest margin in the set was painted full red even when *every*
        class had a comfortably positive margin. Red now means one thing
        only: the class's nearest other class sits closer than its own
        prototype (negative margin, genuine confusion risk).
        """
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
        # Only ever widened, never narrowed: a fixed scale must not push a
        # real (if unusual) value off the edge of the plot.
        values = list(compactness) + list(separation)
        finite = [v for v in values if v == v]
        if finite:
            lo = min(lo, min(finite) - 0.02)
            hi = max(hi, max(finite) + 0.02)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")  # so "above the diagonal" reads as a 45° line
        ax.set_xlabel("separation (nearest other class)")
        ax.set_ylabel("compactness")
        # Both axes are the same similarity scale, and the point of the plot
        # is which side of the diagonal a class falls on -- so both get the
        # grid that lets that distance be read.
        style_axes(ax)
        fig.colorbar(sc, ax=ax, label="margin (compactness - separation)", fraction=0.046, pad=0.04, extend="both")
        fig.tight_layout()
        return fig

    def _build_assignment_grid(
        self, ctx: dict, row_labels: list, col_labels: list, matrix: np.ndarray, best_class: np.ndarray, best_k: np.ndarray
    ) -> ft.Control:
        """Same visual language as the Metrics tab's confusion matrix
        (blues gradient, clickable cells) -- rows are true classes, columns
        are individual prototypes. Clicking (class X, prototype Y) opens
        the images actually labeled X that this prototype attracted."""
        if not row_labels or not col_labels:
            return info_banner("No valid prototype to show.")

        idx_by_label = {label: i for i, label in enumerate(ctx["exp"].model_prototypes.class_labels)}
        max_val = int(matrix.max()) if matrix.size else 1

        header = ft.Row(
            controls=[ft.Container(width=GRID_LABEL_WIDTH)] + [
                ft.Container(
                    content=ft.Text(f"{cls[:6]}#{k}", size=8),
                    width=GRID_CELL_SIZE, alignment=ft.alignment.center, tooltip=f"{cls}#{k}",
                ) for cls, k in col_labels
            ],
            spacing=2,
        )
        body_rows = [header]
        for i, row_label in enumerate(row_labels):
            row_controls = [
                ft.Container(
                    content=ft.Text(row_label, size=10, weight=ft.FontWeight.BOLD),
                    width=GRID_LABEL_WIDTH, alignment=ft.alignment.center_right,
                    padding=ft.padding.only(right=8),
                )
            ]
            for j, (col_cls, col_k) in enumerate(col_labels):
                count = int(matrix[i, j])
                fraction = count / max_val if max_val else 0.0
                bg = blues_color(fraction)
                fg = "#ffffff" if fraction > 0.55 else "#111111"
                col_ci = idx_by_label[col_cls]
                row_controls.append(
                    ft.Container(
                        width=GRID_CELL_SIZE, height=GRID_CELL_SIZE, bgcolor=bg, border_radius=4,
                        alignment=ft.alignment.center, ink=True,
                        content=ft.Text(str(count), size=10, color=fg),
                        tooltip=f"class = {row_label}\nprototype = {col_cls}#{col_k}\nn = {count}",
                        on_click=lambda e, rl=row_label, ci=col_ci, k=col_k, cls=col_cls: self._on_assignment_cell_click(ctx, rl, ci, k, cls, best_class, best_k),
                    )
                )
            body_rows.append(ft.Row(row_controls, spacing=2))

        return ft.Row([ft.Column(body_rows, spacing=2)], scroll=ft.ScrollMode.ALWAYS)

    def _on_assignment_cell_click(self, ctx: dict, row_label: str, col_class_idx: int, col_k: int, col_cls: str, best_class: np.ndarray, best_k: np.ndarray):
        mask = (ctx["labels"] == row_label) & (best_class == col_class_idx) & (best_k == col_k)
        uuids = ctx["uuids"][mask].tolist()
        self._open_carousel(f"{row_label} → prototype {col_cls}#{col_k} · {len(uuids)} image(s)", uuids, ctx)

    def _build_similarity_grid(self, labels_flat: list, sim_matrix: np.ndarray) -> ft.Control:
        """Raw geometric proximity between prototypes (cosine similarity of
        their vectors) -- unlike the assignment matrix above, a cell here
        has no set of images behind it (it's a property of the prototype
        vectors alone, not of any actual sample), so it's display-only, no
        click. Same grid language otherwise, including class-grouping and a
        masked diagonal (always exactly 1.0, uninformative)."""
        n = len(labels_flat)
        if n == 0:
            return info_banner("No valid prototype to show.")

        off_diag = np.array([sim_matrix[i, j] for i in range(n) for j in range(n) if i != j])
        vmin, vmax = (float(off_diag.min()), float(off_diag.max())) if off_diag.size else (-1.0, 1.0)
        span = (vmax - vmin) or 1.0

        def group_gap(idx: int) -> int:
            """Extra spacing before column/row idx if it starts a new class's block."""
            return 6 if idx > 0 and labels_flat[idx][0] != labels_flat[idx - 1][0] else 2

        header_cells = [ft.Container(width=GRID_LABEL_WIDTH)]
        for j, (cls, k) in enumerate(labels_flat):
            header_cells.append(ft.Container(
                content=ft.Text(f"{cls[:6]}#{k}", size=8),
                width=GRID_CELL_SIZE, alignment=ft.alignment.center, tooltip=f"{cls}#{k}",
                margin=ft.margin.only(left=group_gap(j) - 2),
            ))
        body_rows = [ft.Row(header_cells, spacing=2)]

        for i, (row_cls, row_k) in enumerate(labels_flat):
            row_controls = [ft.Container(
                content=ft.Text(f"{row_cls[:10]}#{row_k}", size=9, weight=ft.FontWeight.BOLD),
                width=GRID_LABEL_WIDTH, alignment=ft.alignment.center_right,
                padding=ft.padding.only(right=8),
                margin=ft.margin.only(top=group_gap(i) - 2),
            )]
            for j, (col_cls, col_k) in enumerate(labels_flat):
                is_diag = i == j
                if is_diag:
                    bg, fg, text = "#dcdcdc", "#888888", "—"
                else:
                    fraction = max(0.0, min(1.0, (sim_matrix[i, j] - vmin) / span))
                    bg = blues_color(fraction)
                    fg = "#ffffff" if fraction > 0.55 else "#111111"
                    text = f"{sim_matrix[i, j]:.2f}"
                row_controls.append(ft.Container(
                    width=GRID_CELL_SIZE, height=GRID_CELL_SIZE, bgcolor=bg, border_radius=4,
                    alignment=ft.alignment.center,
                    content=ft.Text(text, size=9, color=fg),
                    tooltip=f"{row_cls}#{row_k} · {col_cls}#{col_k}\nsimilarity = {sim_matrix[i, j]:.3f}",
                    margin=ft.margin.only(left=group_gap(j) - 2, top=group_gap(i) - 2),
                ))
            body_rows.append(ft.Row(row_controls, spacing=2))

        return ft.Row([ft.Column(body_rows, spacing=2)], scroll=ft.ScrollMode.ALWAYS)

    # ---- lifecycle ---------------------------------------------------

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow an experiment rename (see AppContext.notify_rename). The
        caches that hold minutes of kmeans / nearest-prototype work are
        re-keyed by the registry itself, which AppContext calls before this;
        what is left here is this view's own name-keyed controls."""
        for store in (self._kmeans_fields, self._source_status_texts):
            if old_name in store:
                store[new_name] = store.pop(old_name)

        selected = self.experiment_dropdown.value
        self._refresh_experiment_dropdown()
        if selected == old_name:
            self.experiment_dropdown.value = new_name
        self._rebuild_sidecar_rows()
        self._render()
        if self.root.page:
            self.root.update()

    def refresh(self):
        self._rebuild_sidecar_rows()
        self._refresh_experiment_dropdown()
        self._render()
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        self.root.content = ft.Column(
            controls=[
                section_title("Prototype-space analysis", icon=ft.Icons.HUB_OUTLINED, ctx=self.ctx, manual="prototypes"),
                ft.Text(
                    "Needs an embeddings sidecar on a loaded experiment, plus the model's own "
                    "prototypes for the scored diagnostics -- both attach in the Data tab.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                self.sidecars_column,
                self.warnings_area,
                self.class_focus.build(),
                self.role_area,
                self.case_study_area,
                self.compactness_area,
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
        )
        self.refresh()
        return self.root
