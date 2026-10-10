"""Tab: find the crops worth a human look, decide on them, and export the
result as something you can train on.

Three things happen here, in that order:

1. a *candidate source* proposes crops -- from score margins, from
   cross-model consensus, from prototype distances, from a noisy subcenter,
   from near-duplicates. They are interchangeable by design (see
   core/candidates.py): embeddings are one method, not the entry fee.
2. the crops get *reviewed* -- one at a time in the triage panel, a page at
   a time in the carousel, or in bulk for a whole subcenter.
3. the decisions get *exported* into a versioned copy of the dataset (see
   ui/review_panel.py).

Two rules the UI enforces rather than merely documents:

* scored prototype diagnostics only ever run on real, imported prototypes
  (see PrototypeSet.require_model_source) -- a kmeans estimate fitted on the
  pool it would be scored against is circular.
* bulk actions exist only on a train pool. Excluding hundreds of test crops
  in one click is how an evaluation gets quietly faked; on test, every
  decision is made on one crop at a time.
"""
from __future__ import annotations

import traceback

import flet as ft
import numpy as np

from lookin.core import candidates as cand
from lookin.core import embedding_analysis as ea
from lookin.core.data_manager import SPLIT_TRAIN
from lookin.core.review import STATUS_EXCLUDE, STATUS_RELABEL, STATUS_UNCERTAIN
from lookin.ui.carousel import build_carousel_from_list
from lookin.ui.class_matrix import searchable_dropdown_kwargs
from lookin.ui.components import ACCENT, card, diverging_color, info_banner, section_title
from lookin.ui.embedding_context import (
    build_analysis_context, coverage_note, experiments_with_embeddings, experiments_with_model_prototypes,
)
from lookin.ui.review_panel import ReviewPanel
from lookin.ui.review_widgets import NO_REASON, record_decision, reason_options
from lookin.core.comparison import IMPACT_METRICS, review_impact
from lookin.ui.server_browser import ServerFileBrowser
from lookin.ui.triage_view import TriageView

# Candidate source keys. Compared in code, so they are keys, not labels --
# the readable name lives in SOURCE_LABELS.
SOURCE_MARGIN = "margin"
SOURCE_CONFIDENT = "confident_errors"
SOURCE_ALL_ERRORS = "all_errors"
SOURCE_CONSENSUS = "consensus"
SOURCE_PROTOTYPES = "prototype_flags"
SOURCE_DUPLICATES = "near_duplicates"

SOURCE_LABELS = {
    SOURCE_MARGIN: "Score margin (results CSV only)",
    SOURCE_CONFIDENT: "Confident errors (results CSV only)",
    SOURCE_ALL_ERRORS: "All errors, most confident first (results CSV only)",
    SOURCE_CONSENSUS: "Consensus across experiments",
    SOURCE_PROTOTYPES: "Prototype flags (embeddings)",
    SOURCE_DUPLICATES: "Near-duplicates and leakage (embeddings)",
}

# Which sources cannot run without a named experiment loaded. Used to guard
# the automatic re-run when a snapshot is restored: a search whose
# experiment has gone should do nothing, not raise.
SOURCE_NEEDS_EXPERIMENT = (SOURCE_MARGIN, SOURCE_CONFIDENT, SOURCE_ALL_ERRORS)

SOURCE_REQUIREMENTS = {
    SOURCE_MARGIN: "one loaded experiment",
    SOURCE_CONFIDENT: "one loaded experiment",
    SOURCE_ALL_ERRORS: "one loaded experiment",
    SOURCE_CONSENSUS: "at least two loaded experiments",
    SOURCE_PROTOTYPES: "embeddings + the model's real prototypes",
    SOURCE_DUPLICATES: "embeddings",
}

MAX_TABLE_ROWS = 60
REFERENCE_COUNT = 6

# Class picker: "all classes" plus one positional key per class ("0", "1",
# ...) -- see _render_class_picker for why the class label is never the key.
ALL_CLASSES = "all"
MATCH_LABELS = {
    cand.MATCH_LABEL: "Annotated class",
    cand.MATCH_SUGGESTION: "Suggested class",
}


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class CleaningView:
    def __init__(self, ctx):
        self.ctx = ctx
        self.server_browser = ServerFileBrowser(ctx)
        self.review_panel = ReviewPanel(ctx, self.server_browser)
        self.triage = TriageView(ctx)

        self._candidate_set = None
        self._row_by_uuid: dict = {}
        self._references: dict = {}
        self._subcenter_rows: list = []
        self._subcenter_exp = ""  # the experiment the rows above were ranked on

        # -- candidate sources --
        self.source_dropdown = ft.Dropdown(
            label="Candidate source", width=320, border_radius=8,
            options=[ft.dropdown.Option(key, SOURCE_LABELS[key]) for key in SOURCE_LABELS],
            value=SOURCE_MARGIN, on_change=self._on_source_change,
        )
        self.experiment_dropdown = ft.Dropdown(label="Experiment", width=240, border_radius=8)
        # -- class filter: applies to every source and to the subcenters --
        self._class_choices: list = []  # position -> class label, see _render_class_picker
        self._class_scope = None  # the chosen class label, None for all classes
        self.class_container = ft.Container()
        self.match_dropdown = ft.Dropdown(
            label="Match on", width=190, border_radius=8, disabled=True,
            options=[ft.dropdown.Option(key, text) for key, text in MATCH_LABELS.items()],
            value=cand.MATCH_LABEL, on_change=self._on_match_change,
        )
        self.limit_field = ft.TextField(label="max candidates", value="300", width=140, border_radius=8)
        self.margin_field = ft.TextField(label="min margin", value="0.0", width=130, border_radius=8)
        self.percentile_field = ft.TextField(label="percentile", value="2.0", width=130, border_radius=8)
        self.agree_field = ft.TextField(label="min models agreeing", value="2", width=170, border_radius=8)
        self.similarity_field = ft.TextField(label="min similarity", value="0.98", width=140, border_radius=8)
        self.cross_split_check = ft.Checkbox(label="Only pairs across splits (leakage)", value=False)
        self.strategy_other = ft.Checkbox(label="closer to another class", value=True)
        self.strategy_percentile = ft.Checkbox(label="lowest percentile of own-class similarity", value=True)
        self.run_btn = ft.FilledButton("Find candidates", icon=ft.Icons.TRAVEL_EXPLORE,
                                       on_click=self._on_run, style=ft.ButtonStyle(bgcolor=ACCENT))
        self.params_row = ft.Row(spacing=14, wrap=True)

        # -- results --
        self.results_header = ft.Text(size=12)
        self.results_table_area = ft.Container()
        self.triage_btn = ft.FilledButton("Start triage", icon=ft.Icons.PLAY_ARROW_ROUNDED,
                                          on_click=self._on_triage, disabled=True)
        self.carousel_btn = ft.OutlinedButton("Open as filmstrip", icon=ft.Icons.VIEW_CAROUSEL_OUTLINED,
                                              on_click=self._on_carousel, disabled=True)
        self.bulk_status_dropdown = ft.Dropdown(
            label="Bulk action", width=190, border_radius=8, dense=True,
            options=[
                ft.dropdown.Option(STATUS_EXCLUDE, "Exclude all"),
                ft.dropdown.Option(STATUS_UNCERTAIN, "Mark all uncertain"),
                ft.dropdown.Option(STATUS_RELABEL, "Relabel all to the suggestion"),
            ],
            value=STATUS_EXCLUDE,
        )
        self.bulk_reason_dropdown = ft.Dropdown(label="Reason", options=reason_options(), value=NO_REASON,
                                                width=180, border_radius=8, dense=True)
        self.bulk_btn = ft.OutlinedButton("Apply to all candidates", icon=ft.Icons.PLAYLIST_ADD_CHECK,
                                          on_click=self._on_bulk, disabled=True)
        self.bulk_note = ft.Text(size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)

        # -- subcenters --
        self.subcenter_btn = ft.FilledButton("Rank subcenters", icon=ft.Icons.SCATTER_PLOT_OUTLINED,
                                             on_click=self._on_rank_subcenters)
        self.subcenter_area = ft.Container()

        # -- reports --
        self.impact_btn = ft.OutlinedButton("Measure this version", icon=ft.Icons.QUERY_STATS,
                                            on_click=self._on_impact)
        self.impact_area = ft.Container()
        self.taxonomy_btn = ft.OutlinedButton("Check the taxonomy", icon=ft.Icons.ACCOUNT_TREE_OUTLINED,
                                              on_click=self._on_taxonomy)
        self.taxonomy_area = ft.Container()

        # -- carousel dialog --
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

        # -- confirmation for bulk actions --
        self.confirm_text = ft.Text(size=12)
        self.confirm_preview = ft.Row(spacing=8, wrap=True)
        self._pending_bulk = None
        self.confirm_dialog = ft.AlertDialog(
            modal=True, title=ft.Text("Apply to every candidate?"),
            content=ft.Column([self.confirm_text, self.confirm_preview], spacing=12, tight=True, width=560),
            actions=[
                ft.TextButton("Apply", on_click=self._on_confirm_bulk),
                ft.TextButton("Cancel", on_click=lambda e: self.ctx.page.close(self.confirm_dialog)),
            ],
        )

        self.split_banner = ft.Container()
        self.warnings_area = ft.Column(spacing=6)
        self.sources_card = ft.Container()
        self.results_card = ft.Container()
        self.subcenter_card = ft.Container()
        self.reports_card = ft.Container()
        self.root = ft.Container(padding=20)

        ctx.subscribe_review(self._on_review_changed)

    # ---- shared helpers ----
    def _store(self):
        return getattr(self.ctx.state, "review", None)

    def _is_train(self) -> bool:
        return self.ctx.state.split_kind == SPLIT_TRAIN

    def _analysis_context(self, exp_name: str):
        return build_analysis_context(self.ctx.state, exp_name)

    def _dataset_records(self, uuids: list) -> dict:
        """uuid -> row dict for the dataset rows behind a candidate list.

        Built from the dataset alone rather than from a merge: a candidate
        can come from a source that never touched an experiment (duplicates,
        a subcenter), and the triage panel only needs the crop path and the
        label.
        """
        dataset = self.ctx.state.dataset
        if dataset is None:
            return {}
        wanted = set(str(u) for u in uuids)
        uuid_col = dataset.columns.uuid_col
        subset = dataset.df[dataset.df[uuid_col].astype(str).isin(wanted)]
        return {str(row[uuid_col]): row for row in subset.to_dict("records")}

    def _build_references(self, exp_name: str = "") -> dict:
        """class label -> a few uuids that best represent it.

        Nearest-to-prototype crops when real prototypes are available (those
        are genuine exemplars), otherwise the first few crops of the class,
        which is weaker but still beats showing nothing: the whole point is
        to answer "what does an X actually look like" without leaving the
        panel.
        """
        dataset = self.ctx.state.dataset
        if dataset is None:
            return {}
        references: dict = {}
        analysis = None
        proto_set = None
        if exp_name and exp_name in self.ctx.state.experiments:
            analysis = self._analysis_context(exp_name)
            exp = self.ctx.state.experiments[exp_name]
            proto_set = exp.model_prototypes
        if analysis is not None and proto_set is not None:
            for label in dataset.labels():
                nearest, _ = ea.nearest_farthest_within_class(
                    analysis["embeddings"], analysis["labels"], analysis["uuids"],
                    proto_set, str(label), k_top=REFERENCE_COUNT,
                )
                if nearest:
                    references[str(label)] = [uid for uid, _sim, _k in nearest]
        if not references:
            uuid_col, label_col = dataset.columns.uuid_col, dataset.columns.label_col
            for label, group in dataset.df.groupby(label_col, observed=True):
                references[str(label)] = [str(u) for u in group[uuid_col].head(REFERENCE_COUNT)]
        return references

    # ---- candidate sources ----
    # Which manual page the (i) on the candidate sections opens. Each
    # strategy answers a different question and has its own page; a single
    # "cleaning" page would be the one nobody reads.
    SOURCE_MANUAL_PAGES = {
        SOURCE_MARGIN: "margin-candidates",
        SOURCE_CONFIDENT: "margin-candidates",
        SOURCE_ALL_ERRORS: "label-noise",
        SOURCE_CONSENSUS: "consensus",
        SOURCE_PROTOTYPES: "embedding-flags",
        SOURCE_DUPLICATES: "near-duplicates",
    }

    def _candidate_manual_page(self) -> str:
        return self.SOURCE_MANUAL_PAGES.get(self.source_dropdown.value, "label-noise")

    def _on_source_change(self, e):
        self._render_params()
        if self.sources_card.page:
            self.sources_card.update()

    # ---- class filter ----
    def _render_class_picker(self) -> None:
        """Rebuilt rather than mutated (see CompareView._rebuild_dropdowns),
        and keyed by position like the case-study picker in embeddings_view:
        class labels come from the user's CSV, so any fixed key standing for
        "all classes" could collide with one of them."""
        dataset = self.ctx.state.dataset
        labels = dataset.labels() if dataset is not None else []
        counts = dataset.df[dataset.columns.label_col].value_counts().to_dict() if dataset is not None else {}
        if self._class_scope not in labels:
            self._class_scope = None
        self._class_choices = labels
        self.class_container.content = ft.Dropdown(
            label="Class", width=240, border_radius=8,
            **searchable_dropdown_kwargs(len(labels) + 1),
            value=ALL_CLASSES if self._class_scope is None else str(labels.index(self._class_scope)),
            options=[ft.dropdown.Option(ALL_CLASSES, "All classes")] + [
                ft.dropdown.Option(str(i), f"{label}  ({int(counts.get(label, 0)):,})")
                for i, label in enumerate(labels)
            ],
            on_change=self._on_class_change,
        )
        self.match_dropdown.disabled = self._class_scope is None

    def _class_filter(self):
        if self._class_scope is None:
            return None
        return cand.ClassFilter(label=self._class_scope, match=self.match_dropdown.value or cand.MATCH_LABEL)

    def _on_class_change(self, e):
        value = e.control.value
        if value not in (None, ALL_CLASSES) and not (str(value).isdigit() and int(value) < len(self._class_choices)):
            return  # text typed into the filter, not an option picked
        self._class_scope = None if value in (None, ALL_CLASSES) else self._class_choices[int(value)]
        self.match_dropdown.disabled = self._class_scope is None
        self._render_subcenters()
        if self.root.page:
            self.root.update()

    def _on_match_change(self, e):
        self._render_subcenters()
        if self.root.page:
            self.root.update()

    def _available_experiments(self, source: str) -> list:
        state = self.ctx.state
        if source in (SOURCE_MARGIN, SOURCE_CONFIDENT, SOURCE_ALL_ERRORS, SOURCE_CONSENSUS):
            return list(state.experiments)
        if source == SOURCE_PROTOTYPES:
            return experiments_with_model_prototypes(state)
        if source == SOURCE_DUPLICATES:
            return experiments_with_embeddings(state)
        return []

    def _render_params(self) -> None:
        source = self.source_dropdown.value
        names = self._available_experiments(source)
        self.experiment_dropdown.options = [ft.dropdown.Option(n) for n in names]
        if self.experiment_dropdown.value not in names:
            self.experiment_dropdown.value = names[0] if names else None
        self._render_class_picker()

        controls = []
        if source != SOURCE_CONSENSUS:
            controls.append(self.experiment_dropdown)
        if source == SOURCE_MARGIN:
            controls.append(self.margin_field)
        elif source == SOURCE_CONSENSUS:
            controls.append(self.agree_field)
        elif source == SOURCE_PROTOTYPES:
            controls += [self.strategy_other, self.strategy_percentile, self.percentile_field, self.margin_field]
        elif source == SOURCE_DUPLICATES:
            controls += [self.similarity_field, self.cross_split_check]
        if source != SOURCE_ALL_ERRORS:  # the whole point of that one is having no cap
            controls.append(self.limit_field)
        controls.append(self.run_btn)
        self.params_row.controls = controls

        requirement = SOURCE_REQUIREMENTS.get(source, "")
        enough = bool(names) if source != SOURCE_CONSENSUS else len(self.ctx.state.experiments) >= 2
        self.run_btn.disabled = not enough
        self.results_header.value = "" if enough else f"This source needs {requirement}."

    def _int_field(self, field, default: int) -> int:
        try:
            return int(float(field.value))
        except (TypeError, ValueError):
            return default

    def _float_field(self, field, default: float) -> float:
        try:
            return float(field.value)
        except (TypeError, ValueError):
            return default

    def recompute_after_restore(self) -> None:
        """Re-run the candidate search the snapshot set up. Guarded on the
        source's own requirements, so restoring a search whose experiment is
        no longer loaded does nothing rather than erroring."""
        state = self.ctx.state
        if state.dataset is None:
            return
        if self.source_dropdown.value in SOURCE_NEEDS_EXPERIMENT and \
                self.experiment_dropdown.value not in state.experiments:
            return
        self._on_run(None)

    def _on_run(self, e):
        state = self.ctx.state
        if state.dataset is None:
            return
        source = self.source_dropdown.value
        exp_name = self.experiment_dropdown.value
        limit = self._int_field(self.limit_field, 300)
        dataset_df = state.dataset.df
        uuid_col, label_col = state.dataset.columns.uuid_col, state.dataset.columns.label_col

        class_filter = self._class_filter()
        extra_notes = []

        try:
            if source in (SOURCE_MARGIN, SOURCE_CONFIDENT):
                experiment = state.experiments[exp_name]
                candidate_set = cand.by_margin(
                    dataset_df, experiment, uuid_col, label_col,
                    min_margin=self._float_field(self.margin_field, 0.0) if source == SOURCE_MARGIN else -np.inf,
                    limit=limit, confident_errors_only=(source == SOURCE_CONFIDENT), class_filter=class_filter,
                )
            elif source == SOURCE_ALL_ERRORS:
                candidate_set = cand.all_errors(
                    dataset_df, state.experiments[exp_name], uuid_col, label_col, class_filter=class_filter,
                )
            elif source == SOURCE_CONSENSUS:
                candidate_set = cand.by_consensus(
                    dataset_df, list(state.experiments.values()), uuid_col, label_col,
                    min_agreeing=self._int_field(self.agree_field, 2), limit=limit, class_filter=class_filter,
                )
            elif source == SOURCE_PROTOTYPES:
                candidate_set = self._run_prototype_flags(exp_name, limit, class_filter)
            elif source == SOURCE_DUPLICATES:
                if class_filter is not None and class_filter.match == cand.MATCH_SUGGESTION:
                    # Both crops of a pair carry the same label and nothing
                    # proposes another, so there is no suggested class to match.
                    class_filter = cand.ClassFilter(label=class_filter.label, match=cand.MATCH_LABEL)
                    extra_notes.append(
                        f"Near-duplicates have no suggested class, so this is filtered on the annotated "
                        f"class {class_filter.label} instead."
                    )
                candidate_set = self._run_duplicates(exp_name, limit, class_filter)
            else:
                return
        except Exception as ex:  # a bad class mapping, a missing column, a sidecar mismatch
            _log(f"candidate search failed: {ex}")
            traceback.print_exc()
            self.results_table_area.content = info_banner(f"Error: {ex}", icon=ft.Icons.ERROR_OUTLINE)
            self.results_header.value = ""
            self._update_results()
            return

        if candidate_set is None:
            return
        if class_filter is not None:
            candidate_set.name += f" -- {class_filter.describe()}"
        notes = [candidate_set.note] + extra_notes
        if self._is_train():
            # Added to the source's own note rather than replacing it: the
            # duplicates source explains a missing split column there.
            notes.insert(0, (
                "Train pool: these scores come from a model that has already seen these crops, so hard-but-"
                "correct samples rank alongside genuine annotation errors. Consensus across experiments is "
                "the strategy least affected by this."
            ))
        candidate_set.note = " ".join(n for n in notes if n)
        self._set_candidate_set(candidate_set, exp_name)

    def _run_prototype_flags(self, exp_name: str, limit: int, class_filter=None):
        state = self.ctx.state
        analysis = self._analysis_context(exp_name)
        exp = state.experiments.get(exp_name)
        if analysis is None or exp is None or exp.model_prototypes is None:
            self.results_table_area.content = info_banner(
                "Attach both an embeddings .npz sidecar and the model's real prototypes .npz to "
                "this experiment (Data tab): a kmeans estimate is not accepted for a scored "
                "diagnostic."
            )
            self._update_results()
            return None
        strategies = []
        if self.strategy_other.value:
            strategies.append("closer_to_other_class")
        if self.strategy_percentile.value:
            strategies.append("percentile")
        flags = ea.flag_noisy_samples(
            analysis["embeddings"], analysis["labels"], analysis["uuids"], exp.model_prototypes,
            strategies=tuple(strategies),
            percentile=self._float_field(self.percentile_field, 2.0),
            other_class_margin=self._float_field(self.margin_field, 0.0),
        )
        return cand.from_embedding_flags(flags, exp_name, limit=limit, class_filter=class_filter)

    def _run_duplicates(self, exp_name: str, limit: int, class_filter=None):
        state = self.ctx.state
        analysis = self._analysis_context(exp_name)
        if analysis is None:
            return None
        merged = analysis["merged"]
        split_column = None
        row_filter = state.dataset.row_filter
        for column in merged.columns:
            if str(column).lower() == "split":
                split_column = merged[column].astype(str).to_numpy()
                break
        pairs, skipped = cand.near_duplicates(
            analysis["embeddings"], analysis["uuids"], analysis["labels"], split_column,
            threshold=self._float_field(self.similarity_field, 0.98), limit=limit,
            cross_split_only=bool(self.cross_split_check.value),
            only_label=class_filter.label if class_filter is not None else None,
        )
        candidate_set = cand.duplicate_candidates(pairs)
        notes = []
        if split_column is None:
            notes.append(
                "No 'split' column in the loaded pool, so train/test leakage can't be detected -- reload the "
                "dataset without the row filter to get both splits in one pool."
            )
        elif row_filter is not None:
            notes.append(
                f"The pool is filtered to {row_filter[0]}={row_filter[1]!r}, so every pair here is within one "
                f"split. Leakage detection needs both splits loaded at once."
            )
        if skipped:
            notes.append(f"{len(skipped)} class(es) were too large to compare pairwise and were skipped.")
        candidate_set.note = " ".join(notes)
        return candidate_set

    def _set_candidate_set(self, candidate_set, exp_name: str = "") -> None:
        self._candidate_set = candidate_set
        self._references = self._build_references(exp_name)
        # The record map has to cover the reference crops as well as the
        # candidates: triage renders both from it, and a reference whose row
        # is missing silently degrades into "no reference crops", which is
        # exactly the help the panel exists to give.
        reference_uuids = [uid for uuids in self._references.values() for uid in uuids]
        self._row_by_uuid = self._dataset_records(list(candidate_set.uuids()) + reference_uuids)
        self._render_results()
        self._update_results()

    # ---- results ----
    def _render_results(self) -> None:
        candidate_set = self._candidate_set
        if candidate_set is None:
            self.results_table_area.content = None
            self.results_header.value = ""
            self.triage_btn.disabled = self.carousel_btn.disabled = self.bulk_btn.disabled = True
            return

        store = self._store()
        decided = store.effective() if store is not None else {}
        n_decided = sum(1 for c in candidate_set.candidates if c.uuid in decided)
        version_id = store.active_version_id if store is not None else None
        self.results_header.value = f"{candidate_set.name}: " + (
            f"{len(candidate_set):,} candidates · {n_decided:,} already decided in '{version_id}'"
            if version_id else
            f"{len(candidate_set):,} candidates · no review version yet -- one is created with your first decision"
        )

        columns = ["uuid", "class", "suggested", "score"] + candidate_set.columns
        rows = []
        for candidate in candidate_set.candidates[:MAX_TABLE_ROWS]:
            decision = decided.get(candidate.uuid)
            cells = [
                ft.DataCell(ft.Text(candidate.uuid[:14], size=11)),
                ft.DataCell(ft.Text(str(candidate.label), size=12)),
                ft.DataCell(ft.Text(str(candidate.suggestion or "—"), size=12,
                                    color=ACCENT if candidate.suggestion else None)),
                ft.DataCell(ft.Text(f"{candidate.score:.4f}", size=11)),
            ] + [
                ft.DataCell(ft.Text(str(candidate.detail.get(column, "")), size=11))
                for column in candidate_set.columns
            ]
            rows.append(ft.DataRow(
                cells=cells,
                color=ft.Colors.with_opacity(0.06, ACCENT) if decision is not None else None,
                on_select_changed=lambda e, u=candidate.uuid: self._open_carousel(highlight=u),
            ))

        table = ft.DataTable(
            columns=[ft.DataColumn(ft.Text(c)) for c in columns],
            rows=rows, heading_row_height=36, data_row_max_height=42,
        )
        blocks = []
        if candidate_set.note:
            blocks.append(info_banner(candidate_set.note, icon=ft.Icons.WARNING_AMBER_OUTLINED))
        blocks.append(ft.Text(
            f"Showing the first {min(len(candidate_set), MAX_TABLE_ROWS)} of {len(candidate_set):,}. "
            f"Click a row to open it in the filmstrip; use triage to work through the whole queue.",
            size=11, color=ft.Colors.ON_SURFACE_VARIANT,
        ))
        blocks.append(ft.Row([table], scroll=ft.ScrollMode.ALWAYS))
        self.results_table_area.content = ft.Column(blocks, spacing=10)

        has_candidates = len(candidate_set) > 0
        self.triage_btn.disabled = not has_candidates
        self.carousel_btn.disabled = not has_candidates
        self.bulk_btn.disabled = not (has_candidates and self._is_train())
        self.bulk_note.value = (
            "Bulk actions are available because this pool is marked as train."
            if self._is_train() else
            "Bulk actions are disabled on a test pool -- excluding many crops at once there fakes the metric. "
            "Mark the pool as train on the Data tab if you are cleaning a training set."
        )

    def _update_results(self) -> None:
        if self.results_card.page:
            self.results_card.update()

    def _on_triage(self, e):
        if self._candidate_set is None:
            return
        self.triage.open(self._candidate_set, self._row_by_uuid, self._references)

    def _on_carousel(self, e):
        self._open_carousel()

    def _open_carousel(self, highlight=None) -> None:
        candidate_set = self._candidate_set
        dataset = self.ctx.state.dataset
        if candidate_set is None or dataset is None:
            return
        records = [self._row_by_uuid[u] for u in candidate_set.uuids() if u in self._row_by_uuid]
        self.carousel_title.value = candidate_set.name
        self.carousel_content.content = build_carousel_from_list(
            dataset, records, dataset.columns.crop_col, dataset.columns.uuid_col,
            highlight_uuid=highlight, ctx=self.ctx, review_source=candidate_set.source,
            suggestions={c.uuid: c.suggestion for c in candidate_set.candidates if c.suggestion},
            label_col=dataset.columns.label_col,
        )
        self.ctx.page.open(self.carousel_dialog)

    # ---- bulk ----
    def _on_bulk(self, e):
        candidate_set = self._candidate_set
        if candidate_set is None or not self._is_train():
            return
        status = self.bulk_status_dropdown.value
        self._pending_bulk = (candidate_set, status)
        self.confirm_text.value = (
            f"{len(candidate_set):,} crops from “{candidate_set.name}” will be marked "
            f"{status} in version '{self._store().active_version_id}'. "
            f"This is undoable in one click from the review panel, as a single batch."
        )
        dataset = self.ctx.state.dataset
        preview = []
        for candidate in candidate_set.candidates[:8]:
            record = self._row_by_uuid.get(candidate.uuid)
            if record is None:
                continue
            from lookin.ui.components import crop_display, file_to_full_base64
            preview.append(ft.Image(
                src_base64=file_to_full_base64(
                    dataset.resolve_crop_path(record[dataset.columns.crop_col]),
                    *crop_display(self.ctx, record)),
                width=64, height=64, fit=ft.ImageFit.COVER, border_radius=6,
            ))
        self.confirm_preview.controls = preview
        self.ctx.page.open(self.confirm_dialog)

    def _on_confirm_bulk(self, e):
        if self._pending_bulk is None:
            return
        candidate_set, status = self._pending_bulk
        reason = "" if self.bulk_reason_dropdown.value in (None, NO_REASON) else self.bulk_reason_dropdown.value
        if status == STATUS_RELABEL:
            # Relabelling a whole set only makes sense per suggestion, so
            # the batch is grouped by target class rather than forced to one.
            by_label: dict = {}
            for candidate in candidate_set.candidates:
                if candidate.suggestion:
                    by_label.setdefault(candidate.suggestion, []).append(candidate.uuid)
            applied = 0
            for label, uuids in by_label.items():
                ok, message = record_decision(self.ctx, uuids, status, new_label=label,
                                              reason=reason, source=candidate_set.source)
                if not ok:
                    self.confirm_text.value = message
                    if self.confirm_text.page:
                        self.confirm_text.update()
                    return
                applied += len(uuids)
            message = f"Relabelled {applied:,} crops to their suggested class."
        else:
            ok, message = record_decision(self.ctx, candidate_set.uuids(), status,
                                          reason=reason, source=candidate_set.source)
            if not ok:
                self.confirm_text.value = message
                if self.confirm_text.page:
                    self.confirm_text.update()
                return
        self.ctx.page.close(self.confirm_dialog)
        self._pending_bulk = None
        self.results_header.value = message
        self._render_results()
        self._update_results()

    # ---- subcenters ----
    def _on_rank_subcenters(self, e):
        state = self.ctx.state
        exp_name = self.experiment_dropdown.value
        names = experiments_with_model_prototypes(state)
        if exp_name not in names:
            exp_name = names[0] if names else None
        if exp_name is None:
            self._subcenter_rows = []
            self.subcenter_area.content = info_banner(
                "Ranking subcenters needs an experiment with both embeddings and the model's real prototypes."
            )
            if self.subcenter_area.page:
                self.subcenter_area.update()
            return
        analysis = self._analysis_context(exp_name)
        exp = state.experiments[exp_name]
        try:
            rows = cand.noisy_subcenters(
                analysis["embeddings"], analysis["labels"], analysis["uuids"], exp.model_prototypes,
            )
        except Exception as ex:
            _log(f"subcenter ranking failed: {ex}")
            traceback.print_exc()
            self._subcenter_rows = []
            self.subcenter_area.content = info_banner(f"Error: {ex}", icon=ft.Icons.ERROR_OUTLINE)
            if self.subcenter_area.page:
                self.subcenter_area.update()
            return

        # Kept unfiltered: changing the class re-filters this list rather
        # than re-running the ranking.
        self._subcenter_rows = rows
        self._subcenter_exp = exp_name
        self._render_subcenters()
        if self.subcenter_area.page:
            self.subcenter_area.update()

    def _labels_by_uuid(self) -> dict:
        dataset = self.ctx.state.dataset
        if dataset is None:
            return {}
        uuid_col, label_col = dataset.columns.uuid_col, dataset.columns.label_col
        return dict(zip(dataset.df[uuid_col].astype(str), dataset.df[label_col].astype(str)))

    def _render_subcenters(self) -> None:
        """The ranking table, through the class filter: only subcenters that
        would still hold something once opened, with how much."""
        rows = self._subcenter_rows
        if not rows:
            return  # nothing ranked yet, or an error banner that must stay up
        exp_name = self._subcenter_exp
        class_filter = self._class_filter()
        matching: dict = {}
        if class_filter is not None:
            labels_by_uuid = self._labels_by_uuid()
            for i, row in enumerate(rows):
                n = cand.subcenter_matches(row, labels_by_uuid, class_filter)
                if n:
                    matching[i] = n
            shown = [(i, rows[i]) for i in matching]
        else:
            shown = list(enumerate(rows))

        table_rows = []
        for i, row in shown[:25]:
            cells = [
                ft.DataCell(ft.Text(f"{row.class_label}#{row.k_idx}", size=12)),
                ft.DataCell(ft.Text(f"{row.n_assigned:,}", size=12)),
                ft.DataCell(ft.Text(f"{row.purity:.3f}", size=12)),
                ft.DataCell(ft.Text(f"{row.compactness:.3f}", size=12)),
                ft.DataCell(ft.Text(f"{row.separation:.3f}", size=12)),
                ft.DataCell(ft.Text(f"{row.noise_score:.3f}", size=12, weight=ft.FontWeight.BOLD)),
            ]
            if class_filter is not None:
                cells.append(ft.DataCell(ft.Text(f"{matching[i]:,}", size=12, color=ACCENT)))
            table_rows.append(ft.DataRow(
                cells=cells, on_select_changed=lambda e, r=row, n=exp_name: self._open_subcenter(r, n)))

        columns = ["prototype", "assigned", "purity", "compactness", "separation", "noise"]
        if class_filter is not None:
            columns.append(class_filter.describe())
        blocks = [ft.Text(
            "One row per (class, subcenter) prototype, worst first. Click a row to load its "
            "samples as candidates.",
            size=11, color=ft.Colors.ON_SURFACE_VARIANT,
        )]
        if class_filter is not None:
            blocks.append(ft.Text(
                f"Class filter: {len(shown):,} of {len(rows):,} subcenters hold crops {class_filter.describe()}; "
                f"opening one lists only those crops.",
                size=11, color=ACCENT,
            ))
        if shown:
            blocks.append(ft.Row([ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c)) for c in columns],
                rows=table_rows, heading_row_height=36, data_row_max_height=40,
            )], scroll=ft.ScrollMode.ALWAYS))
        self.subcenter_area.content = ft.Column(blocks, spacing=10)

    def _open_subcenter(self, row, exp_name: str) -> None:
        dataset = self.ctx.state.dataset
        if dataset is None:
            return
        candidate_set = cand.apply_class_filter(
            cand.subcenter_candidates(row, self._labels_by_uuid()), self._class_filter())
        if self._is_train():
            candidate_set.note = (
                f"Subcenter {row.class_label}#{row.k_idx} holds {row.n_assigned:,} samples at "
                f"{row.purity:.0%} purity. Bulk actions apply to every sample listed."
            )
        self._set_candidate_set(candidate_set, exp_name)

    # ---- reports ----
    def _on_impact(self, e):
        state = self.ctx.state
        store = self._store()
        exp_name = self.experiment_dropdown.value or (list(state.experiments)[0] if state.experiments else None)
        if store is None or state.dataset is None or exp_name is None:
            self.impact_area.content = info_banner("Load a dataset and at least one experiment first.")
            if self.impact_area.page:
                self.impact_area.update()
            return
        rows = []
        for version_id in store.versions:
            impact = cand.version_impact(
                store, version_id, state.dataset.df, state.experiments[exp_name],
                state.dataset.columns.uuid_col, state.dataset.columns.label_col,
            )
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(version_id, size=12, weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{impact.reviewed:,}", size=12)),
                ft.DataCell(ft.Text(f"{impact.relabelled:,}", size=12)),
                ft.DataCell(ft.Text(f"{impact.excluded:,}", size=12)),
                ft.DataCell(ft.Text(f"{impact.accuracy_before:.4f}", size=12)),
                ft.DataCell(ft.Text(f"{impact.accuracy_after:.4f}", size=12)),
                ft.DataCell(ft.Text(f"{impact.delta:+.4f}", size=12, weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.GREEN if impact.delta >= 0 else ft.Colors.RED)),
                ft.DataCell(ft.Text(
                    "  ".join(f"{source} {share:.0%}" for source, share in impact.strategy_precision.items()),
                    size=11,
                )),
            ]))
        self.impact_area.content = ft.Column([
            ft.Text(
                f"'{exp_name}' scored before and after each version's decisions.",
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([ft.DataTable(
                columns=[ft.DataColumn(ft.Text(c)) for c in
                         ["version", "reviewed", "relabelled", "excluded", "before", "after", "delta",
                          "flag precision by source"]],
                rows=rows, heading_row_height=36, data_row_max_height=40,
            )], scroll=ft.ScrollMode.ALWAYS),
            *self._active_version_breakdown(exp_name),
        ], spacing=10)
        if self.impact_area.page:
            self.impact_area.update()

    def _active_version_breakdown(self, exp_name: str) -> list:
        """The table above answers "did accuracy move" for every version.
        This answers "how" for the active one.

        Accuracy alone hides the shape of a cleaning pass: dropping a
        confusable class's worst crops raises accuracy while leaving macro
        F1 flat, and a pass that only removes noise moves calibration more
        than it moves either. These come from the same overlay every metric
        in the app uses (see comparison.review_impact), so they agree with
        what the Metrics tab shows with the overlay switched on.
        """
        try:
            impact = review_impact(self.ctx.state, exp_name)
        except Exception as ex:
            _log(f"cleaning: could not compute the metric breakdown: {ex}")
            return []

        header = ft.Row([
            ft.Container(ft.Text("Metric", size=11, weight=ft.FontWeight.BOLD), width=170),
            ft.Container(ft.Text("Before", size=11, weight=ft.FontWeight.BOLD), width=90),
            ft.Container(ft.Text("After", size=11, weight=ft.FontWeight.BOLD), width=90),
            ft.Container(ft.Text("Change", size=11, weight=ft.FontWeight.BOLD), width=100),
        ], spacing=4)
        deltas = [abs(impact.delta(a)) for a, _ in IMPACT_METRICS if impact.delta(a) == impact.delta(a)]
        max_abs = max(deltas) if deltas else 1.0
        rows = [header]
        for attr, label in IMPACT_METRICS:
            before, after = getattr(impact.before, attr), getattr(impact.after, attr)
            delta = impact.delta(attr)
            rows.append(ft.Row([
                ft.Container(ft.Text(label, size=12), width=170),
                ft.Container(ft.Text(f"{before:.4f}" if before == before else "n/a", size=12,
                                     color=ft.Colors.ON_SURFACE_VARIANT), width=90),
                ft.Container(ft.Text(f"{after:.4f}" if after == after else "n/a", size=12), width=90),
                ft.Container(
                    ft.Text(f"{delta:+.4f}" if delta == delta else "n/a", size=12,
                            weight=ft.FontWeight.BOLD),
                    width=100, border_radius=4, alignment=ft.alignment.center_left,
                    padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    bgcolor=diverging_color(delta if delta == delta else 0.0, max_abs)),
            ], spacing=4))

        return [
            ft.Divider(height=1),
            ft.Text(f"Version '{impact.version_id}' in full", size=12, weight=ft.FontWeight.BOLD),
            ft.Text(
                f"{impact.n_excluded:,} excluded, {impact.n_relabelled:,} relabelled -- "
                f"{impact.before.n_samples:,} rows scored before, {impact.after.n_samples:,} after.",
                size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Column(rows, spacing=4),
        ]

    def _on_taxonomy(self, e):
        state = self.ctx.state
        names = experiments_with_model_prototypes(state)
        exp_name = self.experiment_dropdown.value if self.experiment_dropdown.value in names else (
            names[0] if names else None
        )
        if exp_name is None:
            self.taxonomy_area.content = info_banner(
                "Taxonomy checks read the model's real prototypes -- attach a prototypes .npz first."
            )
            if self.taxonomy_area.page:
                self.taxonomy_area.update()
            return
        exp = state.experiments[exp_name]
        analysis = self._analysis_context(exp_name)
        try:
            per_class = ea.per_class_stats(analysis["embeddings"], analysis["labels"], exp.model_prototypes)
            hints = cand.taxonomy_hints(exp.model_prototypes, per_class)
        except Exception as ex:
            self.taxonomy_area.content = info_banner(f"Error: {ex}", icon=ft.Icons.ERROR_OUTLINE)
            if self.taxonomy_area.page:
                self.taxonomy_area.update()
            return
        if not hints:
            self.taxonomy_area.content = info_banner(
                "No class pair looks like a duplicate, and no class splits into distant subcenters."
            )
        else:
            self.taxonomy_area.content = ft.Column([
                ft.Text(
                    "Where the finding is about the label set rather than about individual labels.",
                    size=11, color=ft.Colors.ON_SURFACE_VARIANT,
                )
            ] + [
                ft.Row([
                    ft.Container(
                        content=ft.Text(hint.kind.upper(), size=10, weight=ft.FontWeight.BOLD, color=ACCENT),
                        padding=ft.padding.symmetric(horizontal=8, vertical=3),
                        border_radius=6, bgcolor=ft.Colors.with_opacity(0.14, ACCENT),
                    ),
                    ft.Text(f"{hint.a}" + (f" ↔ {hint.b}" if hint.b else ""), size=12,
                            weight=ft.FontWeight.BOLD),
                    ft.Text(hint.detail, size=11, color=ft.Colors.ON_SURFACE_VARIANT, expand=True),
                ], spacing=10)
                for hint in hints[:12]
            ], spacing=8)
        if self.taxonomy_area.page:
            self.taxonomy_area.update()

    # ---- lifecycle ----
    def _on_review_changed(self) -> None:
        """A decision was recorded somewhere -- possibly from a carousel in
        another tab. Only the parts that show review state redraw; the
        candidate list itself is unchanged."""
        self.review_panel.refresh()
        if self._candidate_set is not None:
            self._render_results()
        if self.root.page:
            self.root.update()

    def _render_split_banner(self) -> None:
        state = self.ctx.state
        if state.dataset is None:
            self.split_banner.content = None
            return
        if self._is_train():
            self.split_banner.content = info_banner(
                "This pool is marked as TRAIN. Predictions on it are in-sample, so every ranking here also "
                "surfaces hard-but-correct crops; bulk actions are enabled.",
                icon=ft.Icons.MODEL_TRAINING,
            )
        else:
            self.split_banner.content = info_banner(
                f"This pool is marked as {state.split_kind}. Bulk actions are disabled -- decisions are made "
                f"one crop at a time.",
                icon=ft.Icons.FACT_CHECK_OUTLINED,
            )

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow an experiment rename (see AppContext.notify_rename): the
        experiment dropdown holds the name, and a stale selection would
        point a search at an experiment that no longer exists."""
        selected = self.experiment_dropdown.value
        self._render_params()
        if selected == old_name:
            self.experiment_dropdown.value = new_name
        if self.root.page:
            self.root.update()

    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        """The search that was set up, not its results.

        `class_scope` stores the class *label*, never the positional key the
        dropdown carries: that key is an index into the label list and
        would point at a different class the moment the dataset changes
        (see _render_class_picker).
        """
        return {
            "source": self.source_dropdown.value or "",
            "experiment": self.experiment_dropdown.value or "",
            "match": self.match_dropdown.value or cand.MATCH_LABEL,
            "class_scope": self._class_scope or "",
            "limit": str(self.limit_field.value or ""),
            "margin": str(self.margin_field.value or ""),
            "percentile": str(self.percentile_field.value or ""),
            "agree": str(self.agree_field.value or ""),
            "similarity": str(self.similarity_field.value or ""),
            "cross_split": bool(self.cross_split_check.value),
            "strategy_other": bool(self.strategy_other.value),
            "strategy_percentile": bool(self.strategy_percentile.value),
        }

    def restore_view_state(self, state: dict) -> list:
        """Put the search back and stop there. Finding candidates is a
        full scan over the dataset, and this tab has always waited for the
        user to click "Find candidates"."""
        warnings = []

        source = state.get("source")
        if source in SOURCE_LABELS:
            self.source_dropdown.value = source
        elif source:
            warnings.append(f"Cleaning: unknown candidate source '{source}'.")

        match = state.get("match")
        if match in MATCH_LABELS:
            self.match_dropdown.value = match

        experiment = state.get("experiment") or ""
        if experiment:
            if experiment in self.ctx.state.experiments:
                self.experiment_dropdown.value = experiment
            else:
                warnings.append(f"Cleaning: experiment '{experiment}' is not loaded.")

        scope = state.get("class_scope") or ""
        if scope:
            dataset = self.ctx.state.dataset
            labels = dataset.labels() if dataset is not None else []
            if scope in labels:
                self._class_scope = scope
            else:
                warnings.append(f"Cleaning: class '{scope}' is not in this dataset -- searching every class.")
                self._class_scope = None

        for key, field in (
            ("limit", self.limit_field), ("margin", self.margin_field),
            ("percentile", self.percentile_field), ("agree", self.agree_field),
            ("similarity", self.similarity_field),
        ):
            if state.get(key):
                field.value = str(state[key])

        for key, check in (
            ("cross_split", self.cross_split_check),
            ("strategy_other", self.strategy_other),
            ("strategy_percentile", self.strategy_percentile),
        ):
            if key in state:
                check.value = bool(state[key])

        self._render_class_picker()
        self._render_params()
        if self.root.page:
            self.root.update()
        return warnings

    def refresh(self):
        state = self.ctx.state
        self.warnings_area.controls.clear()
        self._render_split_banner()
        self._render_params()
        self.review_panel.refresh()
        exp_name = self.experiment_dropdown.value
        if exp_name and exp_name in state.experiments and state.experiments[exp_name].has_embeddings():
            analysis = self._analysis_context(exp_name)
            if analysis is not None:
                note = coverage_note(analysis, state, exp_name)
                if note:
                    self.warnings_area.controls.append(info_banner(note, icon=ft.Icons.WARNING_AMBER_OUTLINED))
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        self.sources_card = card(ft.Column([
            section_title("Find candidates", icon=ft.Icons.TRAVEL_EXPLORE, ctx=self.ctx, manual=self._candidate_manual_page),
            ft.Text(
                "Every source below produces the same kind of queue. The first two need nothing but a results "
                "CSV; the others add evidence when you have it.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([self.source_dropdown, self.class_container, self.match_dropdown], spacing=16, wrap=True),
            ft.Text(
                "The class filter applies to every source and to the subcenter ranking below: match "
                "on the crop's current label, or on the class the evidence proposes instead.",
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self.params_row,
        ], spacing=10))

        self.results_card = card(ft.Column([
            section_title("Candidates", icon=ft.Icons.FACT_CHECK_OUTLINED, ctx=self.ctx, manual=self._candidate_manual_page),
            self.results_header,
            ft.Row([self.triage_btn, self.carousel_btn], spacing=12),
            ft.Row([self.bulk_status_dropdown, self.bulk_reason_dropdown, self.bulk_btn], spacing=12, wrap=True),
            self.bulk_note,
            self.results_table_area,
        ], spacing=10))

        self.subcenter_card = card(ft.Column([
            section_title("Noisy subcenters", icon=ft.Icons.SCATTER_PLOT_OUTLINED, ctx=self.ctx, manual="noisy-subcenters"),
            ft.Text(
                "Which of the model's own subcenters collected the most junk.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([self.subcenter_btn], spacing=12),
            self.subcenter_area,
        ], spacing=10))

        self.reports_card = card(ft.Column([
            section_title("Did it work?", icon=ft.Icons.QUERY_STATS, ctx=self.ctx, manual="cleaning-impact"),
            ft.Text(
                "What each version changed, and whether the label set itself is the problem.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([self.impact_btn, self.taxonomy_btn], spacing=12),
            self.impact_area,
            self.taxonomy_area,
        ], spacing=10))

        self.root.content = ft.Column(
            controls=[
                section_title("Dataset cleaning", icon=ft.Icons.CLEANING_SERVICES_OUTLINED, ctx=self.ctx, manual="label-noise"),
                ft.Text(
                    "Find suspicious crops, decide on them, and export a versioned dataset you can train on.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                self.split_banner,
                self.warnings_area,
                self.review_panel.build(),
                self.sources_card,
                self.results_card,
                self.subcenter_card,
                self.reports_card,
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
        )
        self.refresh()
        return self.root
