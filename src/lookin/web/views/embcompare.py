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
nearest crops in another experiment, lasso a group to see its crops. Below
it, how far apart the bases are, per class, and the crops of the other
base that the reference does not cover or labels differently.

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
from lookin.core.chart_style import color_for_experiment
from lookin.core.images import crop_display
from lookin.web import html as h
from lookin.web.views.base import NO_RENDER, Dialog, RebuildView
from lookin.web.views.carousel import Carousel, CarouselDialog
from lookin.web.views.embeddings import METHOD_OPTIONS, legend_card, map_widget
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
        self.sample_value = str(ec.DEFAULT_POINTS_PER_MEMBER)
        self.color_by = ec.COLOR_MEMBER
        self.display = DISPLAY_OVERLAY
        self.class_sort = "coverage"
        self.stats_other = 0            # which member's per-class table is shown (index into members)
        # The last run.
        self.members: list = []
        self.checks: list = []
        self.problems: list = []
        self.refusal = ""
        self.cmap = None
        self.stats: dict = {}           # member index -> PairStats
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
        self._clear_results()

    def _save(self) -> None:
        workspace = self._workspace()
        if workspace is None:
            self.save_note = "Not saved: this session is not a workspace of the library."
            return
        value = ec.comparison_to_json(self.refs, self.reference) if len(self.refs) > 1 else {}

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
        self.cmap, self.stats = None, {}
        self.map_html = self.legend_html = self.panels_html = ""
        self._map_points = []
        self.error = ""

    # ---- settings ----

    def on_mode(self, value: str = "", **_):
        if value in ec.PROJECTION_MODES:
            self.mode = value

    def on_method(self, value: str = "", **_):
        if value in ec.METHODS:
            self.method = value

    def on_sample(self, value: str = "", **_):
        self.sample_value = str(value)

    def _per_member(self) -> int:
        try:
            return max(10, min(int(float(self.sample_value)), ec.MAX_POINTS_PER_MEMBER))
        except (TypeError, ValueError):
            return ec.DEFAULT_POINTS_PER_MEMBER

    def on_color(self, value: str = "", **_):
        if value in ec.COLOR_MODES and value != self.color_by:
            self.color_by = value
            self._redraw()

    def on_display(self, value: str = "", **_):
        if value in (DISPLAY_OVERLAY, DISPLAY_PANELS) and value != self.display:
            self.display = value
            self._redraw()

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
        self.cmap = ec.project_members(members, reference, mode=self.mode, method=self.method,
                                       per_member=self._per_member())
        for i, member in enumerate(members):
            if i == reference:
                continue
            self.set_progress(f"Measuring {member.name} against the reference...")
            self.stats[i] = ec.pair_stats(members[reference], member)
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
        title = (f"{dict(METHOD_OPTIONS).get(self.method, self.method)} -- "
                 f"{'fitted on all' if cmap.mode == ec.PROJ_JOINT else 'fitted on the reference'}")
        if self.display == DISPLAY_PANELS:
            fig = charts.embedding_comparison_panels(members, cmap, self.color_by, colors, title)
            self.panels_html = fig_html(self.s, fig, "Compared embeddings, side by side")
            self.map_html = self.legend_html = ""
            return
        fig = charts.embedding_comparison_overlay(members, cmap, self.color_by, colors, title)
        static_html = fig_html(self.s, fig, "Compared embeddings")
        self.map_html, self.legend_html = self._interactive(colors, static_html)
        self.panels_html = ""

    def _interactive(self, colors: dict, static_html: str) -> tuple:
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
        self._map_points = points
        session = self.s

        def resolve(i: int):
            if not 0 <= i < len(points):
                return None
            m, row = points[i]
            lut, turns = crop_display(session, members[m].record(row))
            return members[m].crop_path(row), lut, turns

        map_html, dkey = map_widget(self.s, self.name, data, resolve, None, static_html)
        title = {ec.COLOR_MEMBER: "Show experiment", ec.COLOR_CLASS: "Show class",
                 ec.COLOR_CORRECT: "Show"}[self.color_by]
        legend = [(colors.get(n, "#555555"), n) for n in names]
        if self.color_by == ec.COLOR_CLASS:
            legend.sort(key=lambda item: str(item[1]))
        return map_html, legend_card(legend, names, dkey, title)

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
            right.append(f'<section class="card"><div class="card-b">{self.panels_html}</div></section>')
        elif not self.progress_text:
            right.append(h.banner("Add the experiments to compare, then click 'Compare'.", "info"))
        if self.stats:
            right.append(self._summary_card())
            right.append(self._classes_card())
        display = f"{self.s.display_lut or ''}.{self.s.display_roll_col or ''}"
        return (f'<div class="emb ecmp{" emb-has-map" if has_map else ""}" data-display="{h.e(display)}">'
                f'<div class="emb-left stack">{"".join(p for p in left if p)}</div>{split}'
                f'<div class="emb-right stack">{"".join(p for p in right if p)}</div></div>')

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
               f'{h.button("Add", act_attr=self.a("add"), kind="primary", icon_name="plus", small=True)}</div>'
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
            h.text_input(self.sample_value, label="Crops per experiment", width=70, change=self.c("sample"),
                         extra=' data-force="1"'),
            h.button("Compare", act_attr=self.a("run"), kind="primary", icon_name="play",
                     disabled=len(self.refs) < 2),
        ]
        redraw = [
            h.select(COLOR_OPTIONS, self.color_by, change=self.c("color"), label="Color by"),
            h.select(DISPLAY_OPTIONS, self.display, change=self.c("display"), label="Show"),
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
        reference = self.members[self.reference] if self.reference < len(self.members) else self.members[0]
        blocks = []
        for i, stats in self.stats.items():
            member = self.members[i]

            def row(label, value, page, title=""):
                return (f'<span class="kv-k">{h.e(label)}{h.info_btn(page)}</span>'
                        f'<span class="kv-v"{f" title={chr(34)}{h.e(title)}{chr(34)}" if title else ""}>{value}</span>')

            auc = "--" if stats.domain_auc is None else f"{stats.domain_auc:.3f}"
            grid = "".join([
                row("Told apart (domain AUC)", auc, MANUAL_DOMAIN, "0.5: the model sees no difference; 1: always told apart"),
                row("Covered by the reference", f"{_pct(stats.coverage)} "
                    f"<span class='muted'>(nearest reference crop at least {stats.coverage_threshold:.3f})</span>",
                    MANUAL_COVERAGE),
                row("Similarity to the nearest reference crop",
                    f"{stats.nn_similarity_other:.3f} <span class='muted'>(inside the reference: "
                    f"{stats.nn_similarity_reference:.3f}, medians)</span>", MANUAL_COVERAGE),
                row("Labels agreeing with the reference", f"{_pct(stats.agreement)} "
                    f"<span class='muted'>(inside the reference: {_pct(stats.agreement_reference)})</span>",
                    MANUAL_AGREEMENT),
            ])
            counts = (f"{stats.n_other:,} crops measured against {stats.n_reference:,} of the reference"
                      + (f"; {stats.n_shared:,} crops are in both and left out of the domain test" if stats.n_shared else ""))
            buttons = (h.button(f"Outside the reference ({len(stats.outside):,})", act_attr=self.a("outside", i=i),
                                small=True, icon_name="grid", disabled=not stats.outside,
                                title="Crops of this base with no reference crop as close as the reference's own")
                       + h.button(f"Labelled otherwise ({len(stats.disagreements):,})", act_attr=self.a("disagree", i=i),
                                  small=True, icon_name="grid", disabled=not stats.disagreements,
                                  title="Crops whose nearest reference crops mostly carry another label"))
            blocks.append(f'<div class="stack"><b class="small">{h.e(member.name)} '
                          f'<span class="muted">vs {h.e(reference.name)}</span></b>'
                          f'<div class="kv">{grid}</div><div class="muted small">{h.e(counts)}</div>'
                          f'<div class="row-wrap">{buttons}</div></div>')
        warnings = [w for m in self.members for w in m.warnings]
        if warnings:
            blocks.append(h.banner("Note", "info", lines=warnings))
        return h.section("How far apart the datasets are", "".join(blocks))

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
                workspace.embedding_comparison = (ec.comparison_to_json(self.refs, self.reference)
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
