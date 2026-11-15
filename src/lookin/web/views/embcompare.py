"""Compare embeddings: one model's embedding space, seen through several
datasets.

The Embeddings tab draws one experiment of this workspace. This tab draws
several experiments -- of this workspace or of any other in the library --
in one space, which is only meaningful when they come from the same model:
their imported prototypes are compared, and the comparison is refused when
they differ (core/embedding_compare.py). Typically: the same model, run on
two bases.

What is compared is saved in this workspace's manifest
(Workspace.embedding_comparison) as references -- a workspace id and an
experiment name -- never as a copy: a member from another workspace is read
from that workspace each time the comparison is run.

The map is the Embeddings tab's (static/js/projection.js, through
embeddings.map_widget): hover a point for its crop, click it for its
nearest crops in another experiment, lasso a group to see its crops, or
ring a point's nearest points on the map with the Neighbours tool. Side by
side, every experiment gets an interactive panel of the same points (its
own in color, the others in grey), all moving and selecting together.
Below, one row per experiment against the reference, every pair of
experiments on demand, per class, and the crops of the other base that the
reference does not cover or labels differently.

Keys compared in code, never reworded: the projection modes and the color
modes of core/embedding_compare.py, DISPLAY_*, and the per-class sorts
(embedding_compare.CLASS_SORTS).
"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from lookin.core import embedding_compare as ec
from lookin.core import library as lib
from lookin.core import projection_plan as pp
from lookin.core.chart_style import blues_color, color_for_experiment
from lookin.core.images import crop_display
from lookin.web import html as h
from lookin.web.views.base import NO_RENDER, Dialog, RebuildView
from lookin.web.views.carousel import Carousel, CarouselDialog
from lookin.web.views.embeddings import METHOD_OPTIONS, SAMPLING_OPTIONS, legend_card, map_widget
from lookin.web.views.media import crop_img, fig_html

DISPLAY_OVERLAY = "overlay"
DISPLAY_PANELS = "panels"
DISPLAY_OPTIONS = [(DISPLAY_OVERLAY, "Overlaid"), (DISPLAY_PANELS, "Side by side")]
MODE_OPTIONS = [(ec.PROJ_JOINT, "Fitted on all"), (ec.PROJ_REFERENCE, "Fitted on the reference")]
COLOR_OPTIONS = [(ec.COLOR_MEMBER, "Experiment"), (ec.COLOR_CLASS, "Class (true label)"),
                 (ec.COLOR_CORRECT, "Right / wrong")]
# A session with no workspace still compares its own experiments, under
# this id; such a comparison cannot be saved.
SESSION_ID = "session"
# Rows of the per-class table shown at once.
CLASS_TABLE_MAX = 60
# Crops a class's "show its crops" opens, per experiment.
CLASS_CROPS_MAX = 200
DEFAULT_NEIGHBOURS = 24
# Manual pages of the measures (ids are keys, see CLAUDE.md).
MANUAL_OVERVIEW = "embedding-comparison"
MANUAL_DOMAIN = "domain-separability"
MANUAL_COVERAGE = "embedding-coverage"
MANUAL_AGREEMENT = "cross-label-agreement"
MANUAL_CLASS = "class-shift"
MANUAL_NEIGHBOURS = "cross-neighbours"
MANUAL_PAIRWISE = "embedding-pairwise"
MANUAL_PROBE = "linear-probe"
# Where the map's Neighbours tool searches (keys): every point of the map,
# the clicked point's experiment, the other experiments, the reference.
NN_ALL = "all"
NN_SAME = "same"
NN_OTHER = "other"
NN_REFERENCE = "reference"
NN_OPTIONS = [(NN_ALL, "Every experiment"), (NN_SAME, "Its own experiment"), (NN_OTHER, "The other experiments"),
              (NN_REFERENCE, "The reference")]
# Side by side: the other experiments' points, under the panel's own.
OTHERS_GREY = "#d5d9e0"
PANEL_MIN_SIZE = 280
REFERENCE_CHIP = ' <span class="chip on small">reference</span>'


class _Crops:
    """What a carousel asks of a dataset, for crops whose records carry
    their own absolute path (Member.record) -- they may come from several
    workspaces at once."""

    def __init__(self, labels):
        self._labels = sorted({str(x) for x in labels if x}, key=str)

    def resolve_crop_path(self, path) -> Path:
        return Path(str(path))

    def labels(self) -> list:
        return self._labels


def _pct(value) -> str:
    return "--" if value is None else f"{100 * value:.1f}%"


def _num(value, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


class EmbeddingCompareView(RebuildView):
    name = "embcompare"
    title = "Compare embeddings"
    stale_job = "Compare embeddings"

    def __init__(self, session):
        super().__init__(session)
        self.refs: list = []
        self.reference = 0
        self.loaded_for = None          # the workspace id the refs were read for
        self.add_choice = ""
        self.mode = ec.PROJ_JOINT
        self.method = "pca"
        # Per experiment: which crops the projection is fitted on and which
        # it shows (core/projection_plan.py), saved with the comparison.
        self.plan = ec.default_plan()
        self.sample_value = str(self.plan.fit_size)
        self.color_by = ec.COLOR_MEMBER
        self.display = DISPLAY_OVERLAY
        self.class_sort = "coverage"
        self.stats_other = 0            # which member's per-class table is shown (index into members)
        self.nn_scope = NN_ALL
        # The last run.
        self.members: list = []
        self.checks: list = []
        self.problems: list = []
        self.refusal = ""
        self.cmap = None
        self.stats: dict = {}           # member index -> PairStats
        self.probes: dict = {}          # member index -> LinearProbe | None
        self.accuracies: dict = {}      # member index -> the model's accuracy
        self.pairwise = None            # ec.Pairwise, on demand
        self.pairwise_busy = False
        self._vectors = None            # the embedding of every point of the map, in its order
        self.map_html = ""
        self.legend_html = ""
        self.panels_html = ""
        self._map_points: list = []     # point i of the map -> (member index, row)
        self.save_note = ""

    # ---- the workspace and its saved comparison ----

    def _workspace(self):
        workspace = getattr(self.s, "current_workspace", None)
        return workspace if workspace is not None and workspace.exists() else None

    def _local_id(self) -> str:
        workspace = self._workspace()
        return workspace.id if workspace is not None else SESSION_ID

    def _local_name(self) -> str:
        workspace = self._workspace()
        return workspace.label() if workspace is not None else "This session"

    def _local(self) -> tuple:
        return self._local_id(), self._local_name(), self.state

    def _local_experiments(self) -> list:
        return [name for name, exp in self.state.experiments.items() if exp.has_embeddings()]

    def on_shown(self) -> None:
        local = self._local_id()
        if self.loaded_for == local:
            return
        self.loaded_for = local
        workspace = self._workspace()
        refs, reference = ec.comparison_from_json(workspace.embedding_comparison if workspace else {})
        if not refs and self._local_experiments():
            # Nothing saved: start from this workspace's first experiment,
            # the reference the others are added to.
            refs, reference = [ec.MemberRef(local, self._local_experiments()[0])], 0
        self.refs, self.reference = refs, reference
        plan, mode = ec.comparison_plan(workspace.embedding_comparison if workspace else {})
        if plan is not None:
            self.plan = plan
            self.method = plan.method if plan.method in ec.METHODS else self.method
            self.sample_value = str(plan.fit_size)
        if mode:
            self.mode = mode
        self._clear_results()

    def _save(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            self.save_note = "Not saved: this session is not a workspace of the library."
            return
        value = (ec.comparison_to_json(self.refs, self.reference, pp.with_method(self.plan, self.method), self.mode)
                 if len(self.refs) > 1 else {})

        def change(fresh) -> bool:
            if fresh.embedding_comparison == value:
                return False
            fresh.embedding_comparison = value
            return True

        try:
            fresh = lib.update_workspace(workspace.path, change)
        except Exception as ex:  # noqa: BLE001
            fresh = None
            traceback.print_exc()
            self.save_note = f"Could not save the comparison in the workspace: {ex}"
            return
        if fresh is None:
            self.save_note = "Could not save the comparison in the workspace."
            return
        workspace.embedding_comparison = value
        self.save_note = "Saved in this workspace." if value else ""

    def _member_label(self, ref) -> str:
        if ref.workspace_id == self._local_id():
            return f"{self._local_name()} / {ref.experiment}"
        found = next((w for w in lib.list_workspaces() if w.id == ref.workspace_id), None)
        return f"{found.label() if found else ref.workspace_id} / {ref.experiment}"

    # ---- which experiments can be added ----

    def _listed(self) -> list:
        """Every experiment with embeddings: this session's own (which may
        not be saved yet) and the library's."""
        local = self._local_id()
        rows = []
        for name in self._local_experiments():
            exp = self.state.experiments[name]
            rows.append({"workspace_id": local, "workspace": self._local_name(), "experiment": name,
                         "prototypes_path": exp.model_prototypes_path or "", "local": exp.model_prototypes})
        seen = {(r["workspace_id"], r["experiment"]) for r in rows}
        try:
            library_rows = ec.comparable_experiments()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            library_rows = []
        rows += [r for r in library_rows if (r["workspace_id"], r["experiment"]) not in seen and r["embeddings"]]
        return rows

    def _reference_prototypes(self):
        if not self.refs:
            return None
        ref = self.refs[self.reference]
        if ref.workspace_id == self._local_id() and ref.experiment in self.state.experiments:
            return self.state.experiments[ref.experiment].model_prototypes
        for row in self._listed():
            if (row["workspace_id"], row["experiment"]) == (ref.workspace_id, ref.experiment):
                return ec.read_prototypes(row.get("prototypes_path"))
        return None

    def _candidates(self) -> tuple:
        """(addable rows, how many others cannot be compared)."""
        taken = {(r.workspace_id, r.experiment) for r in self.refs}
        rows = [r for r in self._listed() if (r["workspace_id"], r["experiment"]) not in taken]
        reference = self._reference_prototypes() if self.refs else None
        same, others = [], []
        for row in rows:
            protos = row.get("local") if row.get("local") is not None else ec.read_prototypes(row.get("prototypes_path"))
            if protos is None or (reference is not None and not ec.check_prototypes(reference, protos).ok):
                others.append(row)
            else:
                same.append(row)
        return same, len(others)

    # ---- actions on the list ----

    def on_choose(self, value: str = "", **_):
        self.add_choice = value

    def on_add(self, **_):
        same, _ = self._candidates()
        row = next((r for r in same if f'{r["workspace_id"]}/{r["experiment"]}' == self.add_choice), None)
        if row is None and same and not self.add_choice:
            row = same[0]
        if row is None:
            return
        self.refs.append(ec.MemberRef(row["workspace_id"], row["experiment"]))
        self.add_choice = ""
        self._changed()

    def on_add_all(self, **_):
        same, _ = self._candidates()
        if not same:
            return
        if not self.refs:
            first = same.pop(0)
            self.refs.append(ec.MemberRef(first["workspace_id"], first["experiment"]))
            same, _ = self._candidates()
        self.refs += [ec.MemberRef(r["workspace_id"], r["experiment"]) for r in same]
        self.add_choice = ""
        self._changed()

    def on_remove(self, i: int = 0, **_):
        i = int(i)
        if not 0 <= i < len(self.refs):
            return
        del self.refs[i]
        if self.reference >= len(self.refs) or self.reference == i:
            self.reference = 0
        elif i < self.reference:
            self.reference -= 1
        self._changed()

    def on_reference(self, i: int = 0, **_):
        i = int(i)
        if 0 <= i < len(self.refs) and i != self.reference:
            self.reference = i
            self._changed()

    def _changed(self) -> None:
        self._save()
        self._clear_results()

    def _clear_results(self) -> None:
        self.members, self.checks, self.problems, self.refusal = [], [], [], ""
        self.cmap, self.stats, self.probes, self.accuracies, self.pairwise = None, {}, {}, {}, None
        self.map_html = self.legend_html = self.panels_html = ""
        self._map_points = []
        self._vectors = None
        self.error = ""

    # ---- settings ----

    def on_mode(self, value: str = "", **_):
        if value in ec.PROJECTION_MODES:
            self.mode = value

    def on_method(self, value: str = "", **_):
        if value in ec.METHODS:
            self.method = value

    def on_sample(self, value: str = "", **_):
        self.on_fit_size(value)

    def _set_plan(self, **change) -> None:
        self.plan = pp.clamp(pp.ProjectionPlan(**{**vars(self.plan), **change}))
        self.sample_value = str(self.plan.fit_size)

    def on_sampling(self, value: str = "", **_):
        if value in pp.SAMPLINGS:
            self._set_plan(sampling=value)

    def on_fit_size(self, value: str = "", **_):
        self._set_plan(fit_size=value)

    def on_per_class(self, value: str = "", **_):
        self._set_plan(per_class=value)

    def on_display_size(self, value: str = "", **_):
        self._set_plan(display_size=value)

    def on_color(self, value: str = "", **_):
        if value in ec.COLOR_MODES and value != self.color_by:
            self.color_by = value
            self._redraw()

    def on_display(self, value: str = "", **_):
        if value in (DISPLAY_OVERLAY, DISPLAY_PANELS) and value != self.display:
            self.display = value
            self._redraw()

    def on_nn_scope(self, value: str = "", **_):
        if value in dict(NN_OPTIONS):
            self.nn_scope = value

    def on_class_sort(self, value: str = "", **_):
        if value in ec.CLASS_SORTS:
            self.class_sort = value

    def on_stats_other(self, value: str = "", **_):
        try:
            i = int(value)
        except (TypeError, ValueError):
            return
        if i in self.stats:
            self.stats_other = i

    # ---- running ----

    def on_run(self, **_):
        if len(self.refs) < 2:
            self.error = "Add at least one more experiment to compare with."
            return
        if self.mode == ec.PROJ_REFERENCE and self.method not in ec.TRANSFORM_METHODS:
            self.error = ("t-SNE cannot place points it was not fitted on, so it cannot project onto the "
                          "reference: pick PCA or UMAP, or fit the projection on all experiments.")
            return
        self.rebuild(self._compute)

    def can_update(self) -> bool:
        return self.cmap is not None or bool(self.refusal)

    def on_update(self, **_):
        self.rebuild(self._compute)

    def _compute(self) -> None:
        self._clear_results()
        self.set_progress("Reading the experiments...")
        members, problems = ec.load_members(self.refs, local=self._local())
        self.problems = problems
        if problems:
            self.refusal = "Some experiments could not be read, so nothing is compared."
            return
        reference = self.reference if self.reference < len(members) else 0
        self.members = members
        self.checks = ec.check_members(members, reference)
        problem = ec.comparison_problem(members, reference)
        if problem:
            self.refusal = ("These experiments cannot be compared: their embeddings must come from the same "
                            "model, which their prototypes show.")
            return
        self.set_progress("Projecting...")
        self.plan = pp.clamp(pp.with_method(self.plan, self.method))
        self._save()            # the plan the comparison was run with is part of it
        self.cmap = ec.project_members(members, reference, mode=self.mode, plan=self.plan,
                                       progress=self.set_progress)
        how = ("as in the data" if self.plan.sampling == pp.SAMPLING_REAL
               else f"balanced, up to {self.plan.per_class:,} per class")
        fitted = sum(len(f) for f in self.cmap.fit_rows)
        shown = sum(len(r) for r in self.cmap.rows)
        self.cmap.notes.insert(0, f"Fitted on {fitted:,} crops ({how}; "
                               f"{'every experiment' if self.mode == ec.PROJ_JOINT else 'the reference only'}); "
                               f"{shown:,} shown.")
        for i, member in enumerate(members):
            self.accuracies[i] = ec.member_accuracy(member)
            if i != reference:
                self.set_progress(f"Measuring {member.name} against the reference ({i + 1}/{len(members)})...")
                self.stats[i] = ec.pair_stats(members[reference], member)
            self.set_progress(f"Linear probe on {member.name} ({i + 1}/{len(members)})...")
            self.probes[i] = ec.member_probe(member)
        self.stats_other = next(iter(self.stats), 0)
        self._draw()

    def _redraw(self) -> None:
        if self.cmap is not None:
            self.rebuild(self._draw, clear_stale=False)

    def _member_colors(self) -> dict:
        names = [m.name for m in self.members]
        return {name: color_for_experiment(name, names) for name in names}

    def _draw(self) -> None:
        from lookin.core import charts

        members, cmap = self.members, self.cmap
        colors = ec.group_colors(members, cmap, self.color_by, self._member_colors())
        data, points, owner = self._map_data(colors)
        self._map_points = points
        session = self.s

        def resolve(i: int):
            if not 0 <= i < len(points):
                return None
            m, row = points[i]
            lut, turns = crop_display(session, members[m].record(row))
            return members[m].crop_path(row), lut, turns

        title = {ec.COLOR_MEMBER: "Show experiment", ec.COLOR_CLASS: "Show class",
                 ec.COLOR_CORRECT: "Show"}[self.color_by]
        names = data["groups"]
        legend = [(colors.get(n, "#555555"), n) for n in names]
        if self.color_by == ec.COLOR_CLASS:
            legend.sort(key=lambda item: str(item[1]))
        if self.display == DISPLAY_PANELS:
            # One interactive map per experiment, of the same points in the
            # same order -- its own in color, every other in grey under
            # them -- linked: they pan, zoom and select together.
            link = f"ecmp-{len(points)}-{id(cmap)}"
            grey = OTHERS_GREY
            panels, first_key = [], ""
            for m, member in enumerate(members):
                own = owner == m
                palette = sorted(set(data["palette"]) | {grey})
                pindex = {c: k for k, c in enumerate(palette)}
                panel = dict(data, palette=palette,
                             c=[pindex[data["palette"][c]] if own[i] else pindex[grey] for i, c in enumerate(data["c"])],
                             z=own.astype(int).tolist(), k=None)
                html_, key = map_widget(self.s, self.name, panel, resolve, None, "", compact=True, link=link,
                                        min_size=PANEL_MIN_SIZE, heading=member.name)
                first_key = first_key or key
                panels.append(f'<div class="ecmp-panel">{html_}</div>')
            self.panels_html = f'<div class="ecmp-panels">{"".join(panels)}</div>'
            self.legend_html = legend_card(legend, names, first_key, title)
            self.map_html = ""
            return
        title_fig = (f"{dict(METHOD_OPTIONS).get(self.method, self.method)} -- "
                     f"{'fitted on all' if cmap.mode == ec.PROJ_JOINT else 'fitted on the reference'}")
        fig = charts.embedding_comparison_overlay(members, cmap, self.color_by, colors, title_fig)
        static_html = fig_html(self.s, fig, "Compared embeddings")
        self.map_html, dkey = map_widget(self.s, self.name, data, resolve, None, static_html)
        self.legend_html = legend_card(legend, names, dkey, title)
        self.panels_html = ""

    def _map_data(self, colors: dict) -> tuple:
        """(the map's JSON, point i -> (member, row), point i -> member) --
        the same points in the same order for the overlay and every panel."""
        members, cmap = self.members, self.cmap
        reference = self.reference if self.reference < len(members) else 0
        xs, ys, groups, kinds, hover, points = [], [], [], [], [], []
        for i, (member, rows, pts) in enumerate(zip(members, cmap.rows, cmap.points)):
            member_groups = ec.point_groups(member, rows, self.color_by)
            for j, row in enumerate(rows.tolist()):
                xs.append(float(pts[j, 0]))
                ys.append(float(pts[j, 1]))
                groups.append(member_groups[j])
                kinds.append(0 if i == reference else 1)
                label = member.labels[row] or str(member.uuids[row])[:8]
                hover.append(f"{member.name}: {label}")
                points.append((i, int(row)))
        names, gindex, g = [], {}, []
        for name in groups:
            if name not in gindex:
                gindex[name] = len(names)
                names.append(name)
            g.append(gindex[name])
        palette = sorted({colors.get(n, "#555555") for n in names})
        pindex = {c: k for k, c in enumerate(palette)}
        hl, hl_index, hi = [], {}, []
        for text in hover:
            if text not in hl_index:
                hl_index[text] = len(hl)
                hl.append(text)
            hi.append(hl_index[text])
        data = {
            "x": np.round(np.asarray(xs), 6).tolist(), "y": np.round(np.asarray(ys), 6).tolist(),
            "c": [pindex[colors.get(name, "#555555")] for name in groups], "palette": palette, "g": g,
            "groups": [str(n) for n in names], "k": kinds if len(members) > 1 else None,
            "hl": hl, "hi": hi, "protos": None,
        }
        if cmap.proto_points is not None:
            coords = np.asarray(cmap.proto_points, dtype=float)
            pcolors = [colors.get(label, "#333333") if self.color_by == ec.COLOR_CLASS else "#333333"
                       for label in cmap.proto_labels]
            data["protos"] = {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(), "color": pcolors,
                              "k": [int(k) for k in cmap.proto_ks], "label": list(cmap.proto_labels)}
        owner = np.asarray([m for m, _ in points], dtype=int)
        return data, points, owner

    # ---- the map's Neighbours tool ----

    def _map_vectors(self):
        if self._vectors is None and self._map_points:
            self._vectors = np.stack([self.members[m].embeddings[row] for m, row in self._map_points])
        return self._vectors

    def on_neighbours(self, i: int = 0, k: int = 20, map: str = "", **_):
        """Point i's k nearest points of the map in embedding space, among
        the experiments the "Neighbours searched in" setting names; sent
        back for the map to ring (projection.js)."""
        from lookin.core.embedding_analysis import nearest_rows

        vectors = self._map_vectors()
        i = int(i)
        if vectors is None or not 0 <= i < len(vectors):
            return NO_RENDER
        owner = np.asarray([m for m, _ in self._map_points])
        among = None
        if self.nn_scope == NN_SAME:
            among = owner == owner[i]
        elif self.nn_scope == NN_OTHER:
            among = owner != owner[i]
        elif self.nn_scope == NN_REFERENCE:
            among = owner == self.reference
        rows, sims = nearest_rows(vectors, i, max(1, min(int(k), 500)), among=among)
        self.s.cmd(event="map-neighbours", detail={"map": str(map), "query": i, "idx": rows.tolist(),
                                                    "sims": [round(float(x), 4) for x in sims]})
        return NO_RENDER

    # ---- every pair ----

    def on_pairwise(self, **_):
        if len(self.members) < 2 or self.pairwise_busy:
            return
        self.pairwise_busy = True
        members = list(self.members)

        def work():
            try:
                self.pairwise = ec.pairwise_stats(members, progress=self.set_progress)
            finally:
                self.pairwise_busy = False

        self.rebuild(work, clear_stale=False)

    # ---- crops behind the map ----

    def _carousel(self, title: str, records: list, header: str = "") -> None:
        labels = {r.get("label") for r in records} | {r.get("predicted_label") for r in records}
        dialog = CarouselDialog(self.s, title, _Crops(labels), len(records), lambda a, b: records[a:b], "path",
                                "uuid", label_col="label", header_html=header, outcome_borders=True, foreign=True)
        self.s.open_dialog(dialog)

    def on_point(self, i: int = 0, **_):
        i = int(i)
        if not 0 <= i < len(self._map_points) or not self.members:
            return
        m, row = self._map_points[i]
        self.s.open_dialog(CrossNeighboursDialog(self.s, self.members, m, row, self.reference))

    def on_select(self, idx=None, **_):
        picked = [self._map_points[int(i)] for i in (idx or []) if 0 <= int(i) < len(self._map_points)]
        if not picked:
            return
        records = [self.members[m].record(row) for m, row in picked]
        counts = {}
        for m, _ in picked:
            counts[self.members[m].name] = counts.get(self.members[m].name, 0) + 1
        self._carousel(f"Lasso selection -- {len(records):,} images",
                       records, header=h.e(", ".join(f"{n}: {c:,}" for n, c in counts.items())))

    def on_outside(self, i: int = 0, **_):
        stats = self.stats.get(int(i))
        if stats is None:
            return
        member = self.members[int(i)]
        records = []
        for row, sim in stats.outside:
            rec = member.record(row)
            rec["nearest_reference_similarity"] = round(sim, 4)
            records.append(rec)
        self._carousel(f"{member.name}: outside the reference -- {len(records):,} images", records,
                       header=h.e(f"Their nearest crop of the reference is less similar than "
                                  f"{stats.coverage_threshold:.3f}, the reference's own usual distance; "
                                  f"least similar first."))

    def on_disagree(self, i: int = 0, **_):
        stats = self.stats.get(int(i))
        if stats is None:
            return
        member = self.members[int(i)]
        records = []
        for row, own, theirs, share in stats.disagreements:
            rec = member.record(row)
            rec["neighbours_label"] = theirs
            rec["neighbours_share"] = round(share, 2)
            records.append(rec)
        self._carousel(f"{member.name}: labelled otherwise than the reference -- {len(records):,} images", records,
                       header=h.e(f"Most of their {ec.AGREEMENT_K} nearest crops of the reference carry another "
                                  f"label (neighbours_label); most unanimous first."))

    def on_class_crops(self, label: str = "", **_):
        records = []
        for member in self.members:
            rows = np.flatnonzero(member.labels == label)
            rows = rows[ec.sample_rows(len(rows), CLASS_CROPS_MAX)]
            records += [member.record(int(r)) for r in rows]
        self._carousel(f"Class {label} in every compared experiment -- {len(records):,} images", records)

    # ---- rendering ----

    def render(self) -> str:
        self.on_shown()
        left = [f'<div class="sec-t"><h2>One model, several datasets</h2>{h.info_btn(MANUAL_OVERVIEW)}</div>',
                self._members_card(), self._settings_card()]
        left.append(self.legend_html)
        has_map = bool(self.map_html or self.panels_html)
        split = ('<div class="emb-split" title="Drag to resize the map; drag it to the left edge, or double-click, '
                 'for a full-width map"></div>')
        right = [self.stale_html(), self.progress_html()]
        if self.refusal:
            right.append(self._refusal_html())
        elif self.map_html:
            right.append(self.map_html)
        elif self.panels_html:
            right.append(self.panels_html)
        elif not self.progress_text:
            right.append(h.banner("Add the experiments to compare, then click 'Compare'.", "info"))
        # The tables go below the map, across the page: with many
        # experiments they are wider than the map's column.
        below = [self._summary_card(), self._pairwise_card(), self._classes_card()] if self.stats else []
        display = f"{self.s.display_lut or ''}.{self.s.display_roll_col or ''}"
        return (f'<div class="stack"><div class="emb ecmp{" emb-has-map" if has_map else ""}" data-display="{h.e(display)}">'
                f'<div class="emb-left stack">{"".join(p for p in left if p)}</div>{split}'
                f'<div class="emb-right stack">{"".join(p for p in right if p)}</div></div>'
                f'{"".join(b for b in below if b)}</div>')

    def _members_card(self) -> str:
        rows = []
        checks = self.checks if len(self.checks) == len(self.refs) else []
        names = [self._member_label(r) for r in self.refs]
        colors = {n: color_for_experiment(n, names) for n in names}
        for i, ref in enumerate(self.refs):
            name = names[i]
            is_ref = i == self.reference
            badge = ('<span class="chip on small">reference</span>' if is_ref else
                     h.button("Make reference", act_attr=self.a("reference", i=i), small=True, kind="link",
                              title="Measure the others against this one, and project onto it"))
            status = ""
            if checks:
                ok = checks[i].ok
                status = (h.icon("check", 13, "c-ok") if ok else h.icon("warn", 13, "c-warn"))
                status = f'<span title="{h.e(checks[i].reason)}">{status}</span>'
            rows.append(f'<div class="row-nw ecmp-m">{h.dot(colors[name], size=10)}'
                        f'<span class="ell small" title="{h.e(name)}">{h.e(name)}</span>{status}'
                        f'<span class="grow"></span>{badge}'
                        f'{h.button(icon_name="x", act_attr=self.a("remove", i=i), small=True, kind="ghost", title="Remove from the comparison")}'
                        f'</div>')
        same, n_others = self._candidates()
        options = [(f'{r["workspace_id"]}/{r["experiment"]}', f'{r["workspace"]} / {r["experiment"]}') for r in same]
        if options and self.add_choice not in dict(options):
            self.add_choice = options[0][0]
        add = (f'<div class="row-nw">{h.select(options, self.add_choice, change=self.c("choose"), cls="grow", title="An experiment of the same model")}'
               f'{h.button("Add", act_attr=self.a("add"), kind="primary", icon_name="plus", small=True)}'
               f'{h.button(f"Add all ({len(options)})", act_attr=self.a("add_all"), small=True, title="Every experiment of the library from the same model") if len(options) > 1 else ""}</div>'
               if options else '<div class="muted small">No other experiment of the library comes from the same model.</div>')
        notes = []
        if n_others:
            notes.append(f"{n_others} other experiment(s) with embeddings come from another model, or have no "
                         f"prototypes to show they do not, and cannot be added.")
        if self.save_note:
            notes.append(self.save_note)
        body = ("".join(rows) or '<div class="muted small">Nothing to compare yet.</div>') + add + "".join(
            f'<div class="muted small">{h.e(n)}</div>' for n in notes)
        return h.section("Experiments compared", f'<div class="stack">{body}</div>')

    def _settings_card(self) -> str:
        controls = [
            h.select(MODE_OPTIONS, self.mode, change=self.c("mode"), label="Projection"),
            h.select(METHOD_OPTIONS, self.method, change=self.c("method"), label="Method"),
            h.select(SAMPLING_OPTIONS, self.plan.sampling, change=self.c("sampling"), label="Fit on",
                     title="Which crops of each experiment the projection is fitted on"),
            (h.text_input(str(self.plan.fit_size), label="Crops per experiment", width=70,
                          change=self.c("fit_size"), extra=' data-force="1"')
             if self.plan.sampling == pp.SAMPLING_REAL else
             h.text_input(str(self.plan.per_class), label="Per class and experiment", width=60,
                          change=self.c("per_class"), extra=' data-force="1"')),
            h.text_input(str(self.plan.display_size), label="Shown per experiment", width=80,
                         change=self.c("display_size"),
                         extra=' data-force="1"' + (" disabled" if self.method not in pp.TRANSFORM_METHODS else "")),
            h.info_btn("fit-and-shown"),
            h.button("Compare", act_attr=self.a("run"), kind="primary", icon_name="play",
                     disabled=len(self.refs) < 2),
        ]
        redraw = [
            h.select(COLOR_OPTIONS, self.color_by, change=self.c("color"), label="Color by"),
            h.select(DISPLAY_OPTIONS, self.display, change=self.c("display"), label="Show"),
            h.select(NN_OPTIONS, self.nn_scope, change=self.c("nn_scope"), label="Neighbours searched in",
                     title="Where the map's Neighbours tool looks for a point's nearest points"),
        ]
        note = ""
        if self.mode == ec.PROJ_REFERENCE and self.method not in ec.TRANSFORM_METHODS:
            note = '<div class="muted small">t-SNE can only be fitted on all experiments.</div>'
        notes = "".join(f'<div class="muted small">{h.e(n)}</div>' for n in (self.cmap.notes if self.cmap else []))
        err = h.banner(self.error, "error") if self.error and not self.progress_text else ""
        body = (f'<div class="row-wrap">{"".join(controls)}</div>{note}'
                f'<div class="row-wrap">{"".join(redraw)}</div>{notes}{err}')
        return f'<section class="card"><div class="card-b stack">{body}</div></section>'

    def _refusal_html(self) -> str:
        lines = list(self.problems)
        for member, check in zip(self.members, self.checks):
            if not check.ok:
                lines.append(f"{member.name}: {check.reason}")
        return h.banner(self.refusal, "error", lines=lines)

    def _summary_card(self) -> str:
        """One row per experiment -- the reference first -- so that twenty
        datasets read as one table, not twenty blocks."""
        reference = self.reference if self.reference < len(self.members) else 0
        ref_stats = next(iter(self.stats.values()), None)
        head = (f"<th>Experiment</th><th class='num'>Crops</th><th class='num'>Model accuracy</th>"
                f"<th class='num'>Probe accuracy{h.info_btn(MANUAL_PROBE)}</th>"
                f"<th class='num'>Told apart (AUC){h.info_btn(MANUAL_DOMAIN)}</th>"
                f"<th class='num'>Covered{h.info_btn(MANUAL_COVERAGE)}</th>"
                f"<th class='num'>Nearest reference similarity</th>"
                f"<th class='num'>Labels agreeing{h.info_btn(MANUAL_AGREEMENT)}</th><th></th>")
        rows = []
        for i in [reference] + [j for j in range(len(self.members)) if j != reference]:
            member = self.members[i]
            probe = self.probes.get(i)
            cells = [f"<td><span class='ell' title='{h.e(member.name)}'>{h.e(member.name)}</span>"
                     f"{REFERENCE_CHIP if i == reference else ''}</td>",
                     f"<td class='num'>{len(member):,}</td>", f"<td class='num'>{_pct(self.accuracies.get(i))}</td>",
                     f"<td class='num'>{_pct(probe.accuracy) if probe is not None else '--'}</td>"]
            stats = self.stats.get(i)
            if stats is None:
                inside = ref_stats
                cells += ["<td class='num muted'>0.5</td>",
                          f"<td class='num muted'>{100 - ec.COVERAGE_PERCENTILE:.0f}%</td>",
                          f"<td class='num muted'>{_num(inside.nn_similarity_reference) if inside else '--'}</td>",
                          f"<td class='num muted'>{_pct(inside.agreement_reference) if inside else '--'}</td>", "<td></td>"]
            else:
                auc = (f"{stats.domain_auc:.3f}" if stats.domain_auc is not None
                       else "<span class='muted'>same crops</span>" if stats.n_shared else "--")
                buttons = (h.button(f"{len(stats.outside):,} outside", act_attr=self.a("outside", i=i), small=True,
                                    kind="link", disabled=not stats.outside,
                                    title="Crops with no reference crop as close as the reference's own")
                           + h.button(f"{len(stats.disagreements):,} labelled otherwise", act_attr=self.a("disagree", i=i),
                                      small=True, kind="link", disabled=not stats.disagreements,
                                      title="Crops whose nearest reference crops mostly carry another label")
                           + h.button("classes", act_attr=self.a("stats_other", value=str(i)), small=True, kind="link",
                                      title="Show this experiment in the per-class table below"))
                cells += [f"<td class='num' title='{stats.n_shared:,} crops in both left out'>{auc}</td>",
                          f"<td class='num' title='nearest reference crop at least {stats.coverage_threshold:.3f}'>{_pct(stats.coverage)}</td>",
                          f"<td class='num'>{_num(stats.nn_similarity_other)}</td>",
                          f"<td class='num'>{_pct(stats.agreement)}</td>", f"<td class='row-nw'>{buttons}</td>"]
            rows.append(f"<tr>{''.join(cells)}</tr>")
        table = (f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{head}</tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
        note = ('<div class="muted small">The reference\'s row, in grey, is what each figure is inside the reference '
                'itself.</div>')
        warnings = [w for m in self.members for w in m.warnings]
        extra = h.banner("Note", "info", lines=warnings) if warnings else ""
        return h.section("How far apart the datasets are", table + note + extra,
                         sub="each experiment against the reference")

    def _pairwise_card(self) -> str:
        if len(self.members) < 3 and self.pairwise is None:
            return ""
        button = h.button("Measure every pair", act_attr=self.a("pairwise"), small=True, icon_name="layers",
                          disabled=self.pairwise_busy,
                          title=f"Domain AUC and coverage of every pair, on {ec.PAIRWISE_POINTS:,} crops per experiment")
        if self.pairwise is None:
            return h.section("Every pair of experiments", f'<div class="row">{button}</div>', manual=MANUAL_PAIRWISE,
                             sub="which datasets resemble which")
        pw = self.pairwise
        n = len(pw.names)

        def grid(matrix, fmt, low, high, tip):
            cells = ['<span></span>'] + [f'<span class="cm-h" style="font-size:9px" title="{h.e(nm)}">{k + 1}</span>'
                                         for k, nm in enumerate(pw.names)]
            for a in range(n):
                cells.append(f'<span class="cm-r small ell" title="{h.e(pw.names[a])}">{a + 1}. {h.e(pw.names[a])}</span>')
                for b in range(n):
                    v = matrix[a, b]
                    if not np.isfinite(v):
                        cells.append('<span class="ov-c ov-na" style="height:30px"></span>')
                        continue
                    f = min(1.0, max(0.0, (v - low) / max(high - low, 1e-9)))
                    cells.append(f'<span class="ov-c" style="height:30px;background:{blues_color(f)};'
                                 f'color:{"#ffffff" if f > 0.55 else "#111111"}" '
                                 f'title="{h.e(tip(a, b, v))}">{fmt(v)}</span>')
            return (f'<div class="cm-scroll"><div class="cm-grid" style="grid-template-columns:220px '
                    f'repeat({n}, 44px)">{"".join(cells)}</div></div>')

        auc = grid(pw.domain_auc, lambda v: f"{v:.2f}", 0.5, 1.0,
                   lambda a, b, v: f"{pw.names[a]} vs {pw.names[b]}: told apart with AUC {v:.3f}")
        finite = pw.coverage[np.isfinite(pw.coverage)]
        low = float(finite.min()) if finite.size else 0.0
        cov = grid(pw.coverage, lambda v: f"{100 * v:.0f}", low, 1.0,
                   lambda a, b, v: f"{100 * v:.1f}% of {pw.names[b]} covered by {pw.names[a]}")
        body = (f'<div class="row">{button}</div>'
                f'<b class="small">Told apart (domain AUC; darker = more different; empty: the same crops)</b>{auc}'
                f'<b class="small">Covered (%): row covers column (darker = more covered)</b>{cov}')
        return h.section("Every pair of experiments", body, manual=MANUAL_PAIRWISE,
                         sub=f"{ec.PAIRWISE_POINTS:,} crops per experiment")

    def _classes_card(self) -> str:
        if self.stats_other not in self.stats:
            return ""
        stats = self.stats[self.stats_other]
        member = self.members[self.stats_other]
        controls = []
        if len(self.stats) > 1:
            controls.append(h.select([(str(i), self.members[i].name) for i in self.stats], str(self.stats_other),
                                     change=self.c("stats_other"), label="Against the reference"))
        controls.append(h.select([(k, v[0]) for k, v in ec.CLASS_SORTS.items()], self.class_sort,
                                 change=self.c("class_sort"), label="Sort"))
        rows = ec.sorted_classes(stats.per_class, self.class_sort)
        shown = rows[:CLASS_TABLE_MAX]
        head = ("<th>Class</th><th class='num'>Crops</th><th class='num'>Centroid similarity</th>"
                "<th class='num'>Similarity to prototype</th><th class='num'>Accuracy</th>"
                "<th class='num'>Covered</th><th class='num'>Labels agreeing</th><th></th>")

        def pair(a, b, fmt):
            return f"{fmt(a)} <span class='muted'>&rarr;</span> {fmt(b)}"

        body = "".join(
            f"<tr><td title='{h.e(c.label)}'>{h.e(c.label)}</td>"
            f"<td class='num'>{pair(c.n_reference, c.n_other, lambda v: f'{v:,}')}</td>"
            f"<td class='num'>{_num(c.centroid_similarity)}</td>"
            f"<td class='num'>{pair(c.prototype_reference, c.prototype_other, _num)}</td>"
            f"<td class='num'>{pair(c.accuracy_reference, c.accuracy_other, _pct)}</td>"
            f"<td class='num'>{_pct(c.coverage)}</td><td class='num'>{_pct(c.agreement)}</td>"
            f"<td>{h.button(icon_name='grid', act_attr=self.a('class_crops', label=c.label), small=True, kind='ghost', title='This class in every compared experiment')}</td></tr>"
            for c in shown)
        more = (f'<div class="muted small"><i>{len(rows) - len(shown):,} more classes; change the sort to see '
                f'others.</i></div>' if len(rows) > len(shown) else "")
        table = (f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{head}</tr></thead>'
                 f'<tbody>{body}</tbody></table></div>{more}')
        return h.section("Per class", f'<div class="row-wrap">{"".join(controls)}</div>{table}', manual=MANUAL_CLASS,
                         sub=f"reference -> {member.name}")

    # ---- snapshots: what is compared lives in the workspace ----

    def view_state(self) -> dict:
        return {}

    def restore_view_state(self, state) -> list:
        return []

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        local = self._local_id()
        renamed = False
        for i, ref in enumerate(self.refs):
            if ref.workspace_id == local and ref.experiment == old_name:
                self.refs[i] = ec.MemberRef(local, new_name)
                renamed = True
        if renamed:
            # The manifests were already rewritten (Session.notify_rename);
            # what is in memory follows.
            workspace = self._workspace()
            if workspace is not None:
                workspace.embedding_comparison = (ec.comparison_to_json(self.refs, self.reference, self.plan, self.mode)
                                                  if len(self.refs) > 1 else {})
            self._clear_results()


class CrossNeighboursDialog(Dialog):
    """A crop of one compared experiment, and its nearest crops in another
    (or the same) -- exact, over the whole of that experiment."""

    def __init__(self, session, members: list, member: int, row: int, reference: int):
        query = members[member]
        super().__init__(f"Nearest crops of {query.uuids[row]}", width=4000)
        self.s = session
        self.members = members
        self.member, self.row = member, row
        others = [i for i in range(len(members)) if i != member]
        self.searched = reference if reference != member else (others[0] if others else member)
        self.k = DEFAULT_NEIGHBOURS
        self.carousel = None
        self.summary = ""
        self._compute()

    def _compute(self) -> None:
        query, target = self.members[self.member], self.members[self.searched]
        found = ec.neighbours_across(query, self.row, target, k=self.k)
        records = []
        for n in found:
            rec = target.record(n.index)
            rec["similarity"] = round(n.similarity, 4)
            records.append(rec)
        own = query.labels[self.row]
        same = sum(1 for n in found if own and n.label == own)
        self.summary = (f"{len(found)} nearest in {target.name}"
                        + (f"; {same}/{len(found)} share this crop's label ({own})" if own else "")
                        + (f"; similarity {found[0].similarity:.3f} down to {found[-1].similarity:.3f}." if found else "."))
        labels = {r.get("label") for r in records} | {own}
        self.carousel = Carousel(self, self.s, _Crops(labels), len(records), lambda a, b: records[a:b], "path",
                                 "uuid", highlight_uuid=None, label_col="label", chrome_height=150,
                                 outcome_borders=True, foreign=True)

    def body(self) -> str:
        query = self.members[self.member]
        rec = query.record(self.row)
        caption = " -- ".join(str(x) for x in (query.name, rec.get("label") or "no label",
                                                 f"predicted {rec['predicted_label']}" if rec.get("predicted_label") else "")
                              if x)
        img = crop_img(self.s, query.crop_path(self.row), rec, size=96, alt=str(query.uuids[self.row]), zoom=True)
        options = [(str(i), m.name + (" (the same experiment)" if i == self.member else ""))
                   for i, m in enumerate(self.members)]
        controls = (h.select(options, str(self.searched), change=self.c("target"), label="Search in")
                    + h.select([(str(k), str(k)) for k in (12, 24, 48, 96)], str(self.k), change=self.c("k"),
                               label="Neighbours"))
        head = (f'<div class="row-nw">{img}<div class="stack"><div class="small">{h.e(caption)}'
                f'{h.info_btn(MANUAL_NEIGHBOURS)}</div>'
                f'<div class="row-wrap">{controls}</div><div class="muted small">{h.e(self.summary)}</div></div></div>')
        return f'<div class="car-head">{head}</div>' + self.carousel.render()

    def on_target(self, value: str = "", **_):
        try:
            i = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= i < len(self.members):
            self.searched = i
            self._compute()
            self.refresh()

    def on_k(self, value: str = "", **_):
        try:
            self.k = max(1, min(int(value), 200))
        except (TypeError, ValueError):
            return
        self._compute()
        self.refresh()

    def on_car(self, op: str = "", **args):
        if self.carousel.handle(op, args) is NO_RENDER:
            return NO_RENDER
        self.refresh()
