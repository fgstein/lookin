"""Cleaning: find the crops worth a human look, decide on them, and export
the result as something you can train on -- the web twin of
lookin.ui.cleaning_view.CleaningView.

Same three steps, same controls, same texts:

1. a candidate source proposes crops (score margin, confident errors, all
   errors, consensus, prototype flags, near-duplicates), through the class
   filter that applies to every source and to the subcenter ranking;
2. the crops get reviewed -- one at a time in the triage panel
   (views/triage.py), a page at a time in the shared carousel, or in bulk
   for a whole set (train pools only);
3. the decisions get exported into a versioned copy of the dataset
   (views/review_panel.py).

Every search, ranking and report runs off the request thread, each on a job
of its own so that asking for the impact report while a search runs does
not cancel either; render() only draws what they left behind.
"""
from __future__ import annotations

import traceback

import numpy as np

from lookin.core import candidates as cand
from lookin.core import embedding_analysis as ea
from lookin.core.class_focus import FOCUS_MAX
from lookin.core.comparison import IMPACT_METRICS, review_impact
from lookin.core.data_manager import SPLIT_TRAIN
from lookin.core.review import STATUS_EXCLUDE, STATUS_RELABEL, STATUS_UNCERTAIN
from lookin.ui.cleaning_view import (
    ALL_CLASSES, MATCH_LABELS, MAX_TABLE_ROWS, REFERENCE_COUNT, SOURCE_ALL_ERRORS, SOURCE_CONFIDENT,
    SOURCE_CONSENSUS, SOURCE_DUPLICATES, SOURCE_LABELS, SOURCE_MARGIN, SOURCE_NEEDS_EXPERIMENT,
    SOURCE_PROTOTYPES, SOURCE_REQUIREMENTS,
)
from lookin.ui.cleaning_view import CleaningView as _FletCleaningView
from lookin.ui.components import diverging_color
from lookin.ui.embedding_context import (
    build_analysis_context, coverage_note, experiments_with_embeddings, experiments_with_model_prototypes,
)
from lookin.ui.review_widgets import NO_REASON, record_decision
from lookin.web import html as h
from lookin.web.session import checkpoint
from lookin.web.views.base import Dialog, StaleTracking, View
from lookin.web.views.carousel import open_carousel, reason_options
from lookin.web.views.media import crop_url
from lookin.web.views.review_panel import ReviewPanel
from lookin.web.views.triage import TriageDialog

SOURCE_MANUAL_PAGES = _FletCleaningView.SOURCE_MANUAL_PAGES

BULK_OPTIONS = [
    (STATUS_EXCLUDE, "Exclude all"),
    (STATUS_UNCERTAIN, "Mark all uncertain"),
    (STATUS_RELABEL, "Relabel all to the suggestion"),
]

# Text fields of the search, with the original's defaults. The keys are the
# snapshot keys (see view_state).
FIELD_DEFAULTS = {"limit": "300", "margin": "0.0", "percentile": "2.0", "agree": "2", "similarity": "0.98"}
FIELD_LABELS = {"limit": "max candidates", "margin": "min margin", "percentile": "percentile",
                "agree": "min models agreeing", "similarity": "min similarity"}
FIELD_WIDTHS = {"limit": "80px", "margin": "70px", "percentile": "70px", "agree": "60px", "similarity": "70px"}


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class BulkConfirmDialog(Dialog):
    """"Apply to every candidate?" -- with the first crops as a preview."""

    width = 600

    def __init__(self, view, candidate_set, status: str):
        super().__init__("Apply to every candidate?")
        self.view = view
        self.candidate_set = candidate_set
        self.status = status
        store = view._store()
        self.text = (
            f"{len(candidate_set):,} crops from “{candidate_set.name}” will be marked "
            f"{status} in version '{store.active_version_id if store is not None else None}'. "
            f"This is undoable in one click from the review panel, as a single batch."
        )

    def body(self) -> str:
        dataset = self.view.state.dataset
        preview = []
        for candidate in self.candidate_set.candidates[:8]:
            record = self.view._row_by_uuid.get(candidate.uuid)
            if record is None or dataset is None:
                continue
            url = crop_url(self.view.s, dataset.resolve_crop_path(record[dataset.columns.crop_col]), record)
            preview.append(f'<img class="crop cl-thumb" src="{h.e(url)}" alt="{h.e(candidate.uuid)}" '
                           f'style="width:64px;height:64px;object-fit:cover">')
        return (f'<div class="stack"><div class="small">{h.e(self.text)}</div>'
                f'<div class="row-wrap">{"".join(preview)}</div></div>')

    def footer(self) -> str:
        return (h.button("Apply", act_attr=self.a("apply"), kind="primary")
                + h.button("Cancel", act_attr=self.a("close")))

    def on_apply(self, **_):
        problem = self.view._apply_bulk(self.candidate_set, self.status)
        if problem:
            self.text = problem
            self.refresh()
            return
        self.close()


class CleaningView(StaleTracking, View):
    name = "cleaning"
    title = "Cleaning"
    # candidates found on data that has since changed: search again with the
    # same settings (StaleTracking)
    stale_job = "Cleaning: results"
    stale_message = "Data has changed -- these candidates were found on the previous data."
    updating_message = "Data has changed -- looking for the candidates again."
    cancelled_message = "The update was cancelled -- these candidates are from the previous data."

    def __init__(self, session):
        super().__init__(session)
        self.init_stale()
        self.review_panel = ReviewPanel(session, self.name, "rp")

        # -- the search --
        self.source = SOURCE_MARGIN
        self.experiment = None
        self.match = cand.MATCH_LABEL
        self._class_scope = None          # the chosen class label, None for all classes
        self._class_choices: list = []    # position -> class label (see _class_picker)
        self.fields = dict(FIELD_DEFAULTS)
        self.cross_split = False
        self.strategy_other = True
        self.strategy_percentile = True
        self.bulk_status = STATUS_EXCLUDE
        self.bulk_reason = NO_REASON

        # -- results, written by the background jobs --
        self._candidate_set = None
        self._row_by_uuid: dict = {}
        self._references: dict = {}
        self._results_msg = None          # (text, kind) replacing the table
        self._subcenter_rows: list = []
        self._subcenter_exp = ""
        self._subcenter_msg = None
        self._impact = None               # (exp_name, [VersionImpact], ReviewImpact | None)
        self._impact_msg = None
        self._taxonomy = None             # [TaxonomyHint]
        self._taxonomy_msg = None
        self._warnings: list = []         # the coverage note
        self._coverage_stale = True

        self.busy: dict = {}              # area -> progress text
        self.built = False

    # ---- shared helpers ----
    def _store(self):
        return getattr(self.state, "review", None)

    def _is_train(self) -> bool:
        return self.state.split_kind == SPLIT_TRAIN

    def _analysis_context(self, exp_name: str):
        return build_analysis_context(self.state, exp_name)

    def _bg(self, area: str, text: str, work) -> None:
        """Run `work` on a job of this area's own, its progress shown in
        that area of the tab."""
        self.busy[area] = text

        def run():
            try:
                work()
            finally:
                self.busy.pop(area, None)
                self.s.mark(self.name, "chrome")

        def failed(ex):
            self.busy.pop(area, None)
            self.s.toast(f"Error: {ex}", "error")
            self.s.mark(self.name)

        self.s.job(f"Cleaning: {area}").run(run, on_error=failed)
        self.s.mark(self.name)

    def _progress(self, area: str) -> str:
        text = self.busy.get(area)
        return h.spinner(text) if text else ""

    def _dataset_records(self, uuids: list) -> dict:
        """uuid -> row dict for the dataset rows behind a candidate list."""
        dataset = self.state.dataset
        if dataset is None:
            return {}
        wanted = set(str(u) for u in uuids)
        uuid_col = dataset.columns.uuid_col
        subset = dataset.df[dataset.df[uuid_col].astype(str).isin(wanted)]
        return {str(row[uuid_col]): row for row in subset.to_dict("records")}

    def _build_references(self, exp_name: str = "") -> dict:
        """class label -> a few uuids that best represent it: nearest to the
        real prototypes when there are some, else the first crops."""
        dataset = self.state.dataset
        if dataset is None:
            return {}
        references: dict = {}
        analysis = None
        proto_set = None
        if exp_name and exp_name in self.state.experiments:
            analysis = self._analysis_context(exp_name)
            proto_set = self.state.experiments[exp_name].model_prototypes
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

    # ---- lifecycle ----
    def on_shown(self) -> None:
        if not self.built:
            self.built = True
            self._refresh_coverage()

    def _refresh_coverage(self) -> None:
        """The original's refresh(): the "sidecar covers only some rows"
        note for the selected experiment -- building the analysis context is
        not cheap, so it runs in the background."""
        self._coverage_stale = False
        state = self.state
        self._sync_params()
        exp_name = self.experiment
        if state.dataset is None or not exp_name or exp_name not in state.experiments \
                or not state.experiments[exp_name].has_embeddings():
            self._warnings = []
            return

        def work():
            warnings = []
            if exp_name in state.experiments:
                analysis = self._analysis_context(exp_name)
                if analysis is not None:
                    note = coverage_note(analysis, state, exp_name)
                    if note:
                        warnings.append(note)
            self._warnings = warnings

        self._bg("context", "Reading the embeddings...", work)

    def on_data_change(self) -> None:
        # The original re-renders its controls from the new data (cheap,
        # done by render here) and recomputes the coverage note.
        self._coverage_stale = True
        if self.built and self.s.tab == self.name:
            self._refresh_coverage()
        if self._candidate_set is not None:
            self.dirty = True

    def can_update(self) -> bool:
        return self._candidate_set is not None and self.state.dataset is not None

    def on_update(self, **_):
        self.on_run()

    def on_cancelling(self) -> None:
        if "results" in self.busy:
            self.busy["results"] = self.stopping_message      # cleared by the job when it stops

    def on_review_change(self) -> None:
        """A decision was recorded somewhere -- possibly from a carousel in
        another tab. Everything that shows review state is drawn from the
        store on render, so redrawing is all there is to do."""
        self.refresh()

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        if self.experiment == old_name:
            self.experiment = new_name
        if self._subcenter_exp == old_name:
            self._subcenter_exp = new_name
        self.refresh()

    # ---- parameters ----
    def _available_experiments(self, source: str) -> list:
        state = self.state
        if source in (SOURCE_MARGIN, SOURCE_CONFIDENT, SOURCE_ALL_ERRORS, SOURCE_CONSENSUS):
            return list(state.experiments)
        if source == SOURCE_PROTOTYPES:
            return experiments_with_model_prototypes(state)
        if source == SOURCE_DUPLICATES:
            return experiments_with_embeddings(state)
        return []

    def _sync_params(self) -> list:
        """The original's _render_params: the experiment follows what the
        source can use, the class scope what the dataset holds."""
        names = self._available_experiments(self.source)
        if self.experiment not in names:
            self.experiment = names[0] if names else None
        dataset = self.state.dataset
        labels = dataset.labels() if dataset is not None else []
        if self._class_scope not in labels:
            self._class_scope = None
        self._class_choices = list(labels)
        return names

    def _enough(self, names: list) -> bool:
        return bool(names) if self.source != SOURCE_CONSENSUS else len(self.state.experiments) >= 2

    def _int_field(self, key: str, default: int) -> int:
        try:
            return int(float(self.fields.get(key)))
        except (TypeError, ValueError):
            return default

    def _float_field(self, key: str, default: float) -> float:
        try:
            return float(self.fields.get(key))
        except (TypeError, ValueError):
            return default

    def _class_filter(self):
        if self._class_scope is None:
            return None
        return cand.ClassFilter(label=self._class_scope, match=self.match or cand.MATCH_LABEL)

    def _candidate_manual_page(self) -> str:
        return SOURCE_MANUAL_PAGES.get(self.source, "label-noise")

    # ---- the search ----
    def recompute_after_restore(self) -> None:
        """Re-run the candidate search the snapshot set up, guarded on the
        source's own requirements."""
        state = self.state
        if state.dataset is None:
            return
        self._sync_params()
        if self.source in SOURCE_NEEDS_EXPERIMENT and self.experiment not in state.experiments:
            return
        self.on_run()

    def on_run(self, **_):
        state = self.state
        if state.dataset is None:
            return
        names = self._sync_params()
        if not self._enough(names):
            return
        source = self.source
        exp_name = self.experiment
        limit = self._int_field("limit", 300)
        margin = self._float_field("margin", 0.0)
        percentile = self._float_field("percentile", 2.0)
        agree = self._int_field("agree", 2)
        similarity = self._float_field("similarity", 0.98)
        cross_split = bool(self.cross_split)
        strategies = []
        if self.strategy_other:
            strategies.append("closer_to_other_class")
        if self.strategy_percentile:
            strategies.append("percentile")
        class_filter = self._class_filter()
        is_train = self._is_train()

        def work():
            self._search(source, exp_name, limit, margin, percentile, agree, similarity, cross_split,
                         tuple(strategies), class_filter, is_train)
            self.dirty = False

        self.cancelled = False
        self._bg("results", "Finding candidates...", work)

    def _search(self, source, exp_name, limit, margin, percentile, agree, similarity, cross_split,
                strategies, class_filter, is_train) -> None:
        state = self.state
        dataset_df = state.dataset.df
        uuid_col, label_col = state.dataset.columns.uuid_col, state.dataset.columns.label_col
        extra_notes = []
        try:
            if source in (SOURCE_MARGIN, SOURCE_CONFIDENT):
                experiment = state.experiments[exp_name]
                candidate_set = cand.by_margin(
                    dataset_df, experiment, uuid_col, label_col,
                    min_margin=margin if source == SOURCE_MARGIN else -np.inf,
                    limit=limit, confident_errors_only=(source == SOURCE_CONFIDENT), class_filter=class_filter,
                )
            elif source == SOURCE_ALL_ERRORS:
                candidate_set = cand.all_errors(
                    dataset_df, state.experiments[exp_name], uuid_col, label_col, class_filter=class_filter,
                )
            elif source == SOURCE_CONSENSUS:
                candidate_set = cand.by_consensus(
                    dataset_df, list(state.experiments.values()), uuid_col, label_col,
                    min_agreeing=agree, limit=limit, class_filter=class_filter,
                )
            elif source == SOURCE_PROTOTYPES:
                candidate_set = self._run_prototype_flags(exp_name, limit, strategies, percentile, margin,
                                                          class_filter)
            elif source == SOURCE_DUPLICATES:
                if class_filter is not None and class_filter.match == cand.MATCH_SUGGESTION:
                    # Both crops of a pair carry the same label and nothing
                    # proposes another, so there is no suggested class to match.
                    class_filter = cand.ClassFilter(label=class_filter.label, match=cand.MATCH_LABEL)
                    extra_notes.append(
                        f"Near-duplicates have no suggested class, so this is filtered on the annotated "
                        f"class {class_filter.label} instead."
                    )
                candidate_set = self._run_duplicates(exp_name, limit, similarity, cross_split, class_filter)
            else:
                return
        except Exception as ex:  # a bad class mapping, a missing column, a sidecar mismatch
            _log(f"candidate search failed: {ex}")
            traceback.print_exc()
            self._results_msg = (f"Error: {ex}", "error")
            return

        checkpoint()                      # a Cancel during the search lands here
        if candidate_set is None:
            return
        if class_filter is not None:
            candidate_set.name += f" -- {class_filter.describe()}"
        notes = [candidate_set.note] + extra_notes
        if is_train:
            notes.insert(0, (
                "Train pool: these scores come from a model that has already seen these crops, so hard-but-"
                "correct samples rank alongside genuine annotation errors. Consensus across experiments is "
                "the strategy least affected by this."
            ))
        candidate_set.note = " ".join(n for n in notes if n)
        self._set_candidate_set(candidate_set, exp_name)

    def _run_prototype_flags(self, exp_name, limit, strategies, percentile, margin, class_filter=None):
        state = self.state
        analysis = self._analysis_context(exp_name)
        exp = state.experiments.get(exp_name)
        if analysis is None or exp is None or exp.model_prototypes is None:
            self._results_msg = (
                "Attach both an embeddings .npz sidecar and the model's real prototypes .npz to "
                "this experiment (Data tab): a kmeans estimate is not accepted for a scored "
                "diagnostic.", "info")
            return None
        flags = ea.flag_noisy_samples(
            analysis["embeddings"], analysis["labels"], analysis["uuids"], exp.model_prototypes,
            strategies=strategies, percentile=percentile, other_class_margin=margin,
        )
        return cand.from_embedding_flags(flags, exp_name, limit=limit, class_filter=class_filter)

    def _run_duplicates(self, exp_name, limit, similarity, cross_split, class_filter=None):
        state = self.state
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
            threshold=similarity, limit=limit, cross_split_only=cross_split,
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
        references = self._build_references(exp_name)
        # The record map covers the reference crops as well as the
        # candidates: triage renders both from it.
        reference_uuids = [uid for uuids in references.values() for uid in uuids]
        rows = self._dataset_records(list(candidate_set.uuids()) + reference_uuids)
        self._references = references
        self._row_by_uuid = rows
        self._candidate_set = candidate_set
        self._results_msg = None

    # ---- results actions ----
    def on_triage(self, **_):
        if self._candidate_set is None:
            return
        self.s.open_dialog(TriageDialog(self.s, self._candidate_set, self._row_by_uuid, self._references))

    def on_carousel(self, **_):
        self._open_carousel()

    def on_row(self, i: int = 0, **_):
        candidate_set = self._candidate_set
        if candidate_set is None or not (0 <= int(i) < len(candidate_set)):
            return
        self._open_carousel(highlight=candidate_set.candidates[int(i)].uuid)

    def _open_carousel(self, highlight=None) -> None:
        candidate_set = self._candidate_set
        dataset = self.state.dataset
        if candidate_set is None or dataset is None:
            return
        records = [self._row_by_uuid[u] for u in candidate_set.uuids() if u in self._row_by_uuid]
        open_carousel(
            self.s, candidate_set.name, records, dataset.columns.crop_col, dataset.columns.uuid_col,
            highlight_uuid=highlight, review_source=candidate_set.source,
            suggestions={c.uuid: c.suggestion for c in candidate_set.candidates if c.suggestion},
            label_col=dataset.columns.label_col,
        )

    def on_bulk_status(self, value: str = "", **_):
        if value in {k for k, _ in BULK_OPTIONS}:
            self.bulk_status = value

    def on_bulk_reason(self, value: str = "", **_):
        self.bulk_reason = value or NO_REASON

    def on_bulk(self, **_):
        candidate_set = self._candidate_set
        if candidate_set is None or not self._is_train() or not len(candidate_set):
            return
        self.s.open_dialog(BulkConfirmDialog(self, candidate_set, self.bulk_status))

    def _apply_bulk(self, candidate_set, status: str):
        """Returns an error message, or None once the batch is recorded."""
        reason = "" if self.bulk_reason in (None, NO_REASON) else self.bulk_reason
        if status == STATUS_RELABEL:
            # Relabelling a whole set only makes sense per suggestion, so
            # the batch is grouped by target class rather than forced to one.
            by_label: dict = {}
            for candidate in candidate_set.candidates:
                if candidate.suggestion:
                    by_label.setdefault(candidate.suggestion, []).append(candidate.uuid)
            applied = 0
            for label, uuids in by_label.items():
                ok, message = record_decision(self.s, uuids, status, new_label=label,
                                              reason=reason, source=candidate_set.source)
                if not ok:
                    return message
                applied += len(uuids)
            message = f"Relabelled {applied:,} crops to their suggested class."
        else:
            ok, message = record_decision(self.s, candidate_set.uuids(), status,
                                          reason=reason, source=candidate_set.source)
            if not ok:
                return message
        self.s.toast(message, "ok")
        return None

    # ---- the search controls ----
    def on_source(self, value: str = "", **_):
        if value in SOURCE_LABELS:
            self.source = value
            self._sync_params()

    def on_experiment(self, value: str = "", **_):
        if value in self.state.experiments:
            self.experiment = value

    def on_class(self, value: str = "", **_):
        value = "" if value is None else str(value)
        if value == ALL_CLASSES or value == "All classes" or value == "":
            self._class_scope = None
        elif value.isdigit() and int(value) < len(self._class_choices) and len(self._class_choices) + 1 <= FOCUS_MAX:
            self._class_scope = self._class_choices[int(value)]
        else:
            # The type-to-filter field (past FOCUS_MAX classes) sends the
            # label itself; anything else is text typed, not a class picked.
            match = next((l for l in self._class_choices if str(l) == value), None)
            if match is None:
                return
            self._class_scope = match

    def on_match(self, value: str = "", **_):
        if value in MATCH_LABELS:
            self.match = value

    def on_field(self, key: str = "", value="", **_):
        if key in self.fields:
            self.fields[key] = str(value)

    def on_check(self, key: str = "", value=False, **_):
        if key in ("cross_split", "strategy_other", "strategy_percentile"):
            setattr(self, key, bool(value))

    # ---- subcenters ----
    def on_rank_subcenters(self, **_):
        state = self.state
        names = experiments_with_model_prototypes(state)
        exp_name = self.experiment if self.experiment in names else (names[0] if names else None)
        if exp_name is None:
            self._subcenter_rows = []
            self._subcenter_msg = ("Ranking subcenters needs an experiment with both embeddings and the model's "
                                   "real prototypes.", "info")
            return

        def work():
            analysis = self._analysis_context(exp_name)
            exp = state.experiments[exp_name]
            try:
                rows = cand.noisy_subcenters(
                    analysis["embeddings"], analysis["labels"], analysis["uuids"], exp.model_prototypes,
                )
            except Exception as ex:  # noqa: BLE001
                _log(f"subcenter ranking failed: {ex}")
                traceback.print_exc()
                self._subcenter_rows = []
                self._subcenter_msg = (f"Error: {ex}", "error")
                return
            # Kept unfiltered: changing the class re-filters this list
            # rather than re-running the ranking.
            self._subcenter_msg = None
            self._subcenter_exp = exp_name
            self._subcenter_rows = rows

        self._bg("subcenters", "Ranking subcenters...", work)

    def _labels_by_uuid(self) -> dict:
        dataset = self.state.dataset
        if dataset is None:
            return {}
        uuid_col, label_col = dataset.columns.uuid_col, dataset.columns.label_col
        return dict(zip(dataset.df[uuid_col].astype(str), dataset.df[label_col].astype(str)))

    def on_subcenter(self, i: int = 0, **_):
        rows = self._subcenter_rows
        if self.state.dataset is None or not (0 <= int(i) < len(rows)):
            return
        row = rows[int(i)]
        exp_name = self._subcenter_exp
        class_filter = self._class_filter()
        is_train = self._is_train()

        def work():
            candidate_set = cand.apply_class_filter(
                cand.subcenter_candidates(row, self._labels_by_uuid()), class_filter)
            if is_train:
                candidate_set.note = (
                    f"Subcenter {row.class_label}#{row.k_idx} holds {row.n_assigned:,} samples at "
                    f"{row.purity:.0%} purity. Bulk actions apply to every sample listed."
                )
            self._set_candidate_set(candidate_set, exp_name)

        self._bg("results", "Loading the subcenter...", work)

    # ---- reports ----
    def on_impact(self, **_):
        state = self.state
        store = self._store()
        exp_name = self.experiment or (list(state.experiments)[0] if state.experiments else None)
        if store is None or state.dataset is None or exp_name is None:
            self._impact = None
            self._impact_msg = ("Load a dataset and at least one experiment first.", "info")
            return

        def work():
            impacts = []
            for version_id in list(store.versions):
                impacts.append(cand.version_impact(
                    store, version_id, state.dataset.df, state.experiments[exp_name],
                    state.dataset.columns.uuid_col, state.dataset.columns.label_col,
                ))
            try:
                breakdown = review_impact(state, exp_name)
            except Exception as ex:  # noqa: BLE001
                _log(f"cleaning: could not compute the metric breakdown: {ex}")
                breakdown = None
            self._impact_msg = None
            self._impact = (exp_name, impacts, breakdown)

        self._bg("impact", "Measuring the versions...", work)

    def on_taxonomy(self, **_):
        state = self.state
        names = experiments_with_model_prototypes(state)
        exp_name = self.experiment if self.experiment in names else (names[0] if names else None)
        if exp_name is None:
            self._taxonomy = None
            self._taxonomy_msg = ("Taxonomy checks read the model's real prototypes -- attach a prototypes .npz first.",
                                  "info")
            return

        def work():
            exp = state.experiments[exp_name]
            analysis = self._analysis_context(exp_name)
            try:
                per_class = ea.per_class_stats(analysis["embeddings"], analysis["labels"], exp.model_prototypes)
                hints = cand.taxonomy_hints(exp.model_prototypes, per_class)
            except Exception as ex:  # noqa: BLE001
                self._taxonomy = None
                self._taxonomy_msg = (f"Error: {ex}", "error")
                return
            if not hints:
                self._taxonomy = None
                self._taxonomy_msg = ("No class pair looks like a duplicate, and no class splits into distant "
                                      "subcenters.", "info")
            else:
                self._taxonomy_msg = None
                self._taxonomy = hints

        self._bg("taxonomy", "Checking the taxonomy...", work)

    # ---- the review panel ----
    def on_rp(self, op: str = "", **args):
        self.review_panel.handle(op, args)

    # ---- rendering ----
    def render(self) -> str:
        if not self.built:
            self.on_shown()
        elif self._coverage_stale and self.s.tab == self.name:
            self._refresh_coverage()
        names = self._sync_params()
        return (f'<div class="stack cl">{self.stale_html()}{self._header_html()}{self.review_panel.render()}'
                f'{self._sources_html(names)}{self._results_html(names)}'
                f'<div class="grid2 cl-bottom">{self._subcenters_html()}{self._reports_html()}</div></div>')

    def _header_html(self) -> str:
        state = self.state
        parts = []
        if state.dataset is not None:
            if self._is_train():
                parts.append(h.banner(
                    "This pool is marked as TRAIN. Predictions on it are in-sample, so every ranking here also "
                    "surfaces hard-but-correct crops; bulk actions are enabled.", "warn"))
            else:
                parts.append(h.banner(
                    f"This pool is marked as {state.split_kind}. Bulk actions are disabled -- decisions are made "
                    f"one crop at a time.", "info"))
        parts += [h.banner(note, "warn") for note in self._warnings]
        parts.append(self._progress("context"))
        return h.section("Dataset cleaning", f'<div class="stack cl-tight">{"".join(parts)}</div>' if any(parts) else "",
                         manual="label-noise", cls="cl-head")

    def _class_picker(self) -> str:
        """Keyed by position, like the case-study picker in the Embeddings
        tab: class labels come from the user's CSV, so a fixed key standing
        for "all classes" could collide with one of them. Past FOCUS_MAX
        classes, a type-to-filter field over the labels instead."""
        dataset = self.state.dataset
        labels = self._class_choices
        counts = dataset.df[dataset.columns.label_col].value_counts().to_dict() if dataset is not None else {}
        if len(labels) + 1 <= FOCUS_MAX:
            options = [(ALL_CLASSES, "All classes")] + [
                (str(i), f"{label}  ({int(counts.get(label, 0)):,})") for i, label in enumerate(labels)]
            value = ALL_CLASSES if self._class_scope is None else str(labels.index(self._class_scope))
            return h.select(options, value, change=self.c("class"), label="Class")
        list_id = "cl-classes"
        options = '<option value="All classes">' + "".join(
            f'<option value="{h.e(label)}">{h.e(f"{label}  ({int(counts.get(label, 0)):,})")}</option>'
            for label in labels)
        value = "All classes" if self._class_scope is None else str(self._class_scope)
        return (f'<label class="fld"><span class="fld-l">Class</span>'
                f'<input class="inp" list="{list_id}" value="{h.e(value)}" placeholder="Type part of a class name" '
                f'style="width:220px"{self.c("class")}><datalist id="{list_id}">{options}</datalist></label>')

    def _field(self, key: str) -> str:
        return h.text_input(self.fields.get(key, ""), label=FIELD_LABELS[key], width=FIELD_WIDTHS[key],
                            change=self.c("field", key=key))

    def _sources_html(self, names: list) -> str:
        source = self.source
        exp_select = (h.select([(n, n) for n in names], self.experiment or "", change=self.c("experiment"),
                               label="Experiment")
                      if names else h.select([("", "")], "", label="Experiment", disabled=True))
        params = []
        if source != SOURCE_CONSENSUS:
            params.append(exp_select)
        if source == SOURCE_MARGIN:
            params.append(self._field("margin"))
        elif source == SOURCE_CONSENSUS:
            params.append(self._field("agree"))
        elif source == SOURCE_PROTOTYPES:
            params += [
                h.checkbox("closer to another class", self.strategy_other,
                           change=self.c("check", key="strategy_other")),
                h.checkbox("lowest percentile of own-class similarity", self.strategy_percentile,
                           change=self.c("check", key="strategy_percentile")),
                self._field("percentile"), self._field("margin"),
            ]
        elif source == SOURCE_DUPLICATES:
            params += [self._field("similarity"),
                       h.checkbox("Only pairs across splits (leakage)", self.cross_split,
                                  change=self.c("check", key="cross_split"))]
        if source != SOURCE_ALL_ERRORS:  # the whole point of that one is having no cap
            params.append(self._field("limit"))
        searching = "results" in self.busy
        params.append(h.button("Find candidates", act_attr=self.a("run"), kind="primary", icon_name="search",
                               small=True, disabled=not self._enough(names) or self.state.dataset is None or searching))
        top = (f'<div class="row">'
               f'{h.select([(k, SOURCE_LABELS[k]) for k in SOURCE_LABELS], source, change=self.c("source"), label="Candidate source")}'
               f'{self._class_picker()}'
               f'{h.select(list(MATCH_LABELS.items()), self.match, change=self.c("match"), label="Match on", disabled=self._class_scope is None)}'
               f'</div>'
               f'<div class="muted small">The class filter applies to every source and to the subcenter ranking below: '
               f'match on the crop\'s current label, or on the class the evidence proposes instead.</div>')
        body = f'<div class="stack cl-tight">{top}<div class="row cl-params">{"".join(params)}</div></div>'
        return h.section("Find candidates", body, manual=self._candidate_manual_page(),
                         sub="Every source below produces the same kind of queue. The first two need nothing but a "
                             "results CSV; the others add evidence when you have it.")

    def _results_html(self, names: list) -> str:
        candidate_set = self._candidate_set
        store = self._store()
        decided = store.effective() if store is not None else {}
        has_candidates = candidate_set is not None and len(candidate_set) > 0
        if not self._enough(names):
            header = f"This source needs {SOURCE_REQUIREMENTS.get(self.source, '')}."
        elif candidate_set is None or self._results_msg is not None:
            header = ""
        else:
            n_decided = sum(1 for c in candidate_set.candidates if c.uuid in decided)
            version_id = store.active_version_id if store is not None else None
            header = f"{candidate_set.name}: " + (
                f"{len(candidate_set):,} candidates · {n_decided:,} already decided in '{version_id}'"
                if version_id else
                f"{len(candidate_set):,} candidates · no review version yet -- one is created with your first decision"
            )
        buttons = (f'<div class="row">'
                   f'{h.button("Start triage", act_attr=self.a("triage"), kind="primary", icon_name="play", small=True, disabled=not has_candidates)}'
                   f'{h.button("Open as filmstrip", act_attr=self.a("carousel"), icon_name="grid", small=True, disabled=not has_candidates)}'
                   f'<span class="vsep"></span>'
                   f'{h.select(BULK_OPTIONS, self.bulk_status, change=self.c("bulk_status"), label="Bulk action")}'
                   f'{h.select(reason_options(), self.bulk_reason, change=self.c("bulk_reason"), label="Reason")}'
                   f'{h.button("Apply to all candidates", act_attr=self.a("bulk"), icon_name="check", small=True, disabled=not (has_candidates and self._is_train()))}'
                   f'</div>')
        bulk_note = ""
        if candidate_set is not None:
            bulk_note = ("Bulk actions are available because this pool is marked as train."
                         if self._is_train() else
                         "Bulk actions are disabled on a test pool -- excluding many crops at once there fakes the metric. "
                         "Mark the pool as train on the Data tab if you are cleaning a training set.")
        parts = [f'<div class="small cl-header">{h.e(header)}</div>' if header else "", buttons,
                 f'<div class="muted small"><i>{h.e(bulk_note)}</i></div>' if bulk_note else "",
                 self._progress("results")]
        if self._results_msg is not None:
            parts.append(h.banner(*self._results_msg))
        elif candidate_set is not None:
            parts.append(self._candidates_table(candidate_set, decided))
        return h.section("Candidates", f'<div class="stack cl-tight">{"".join(parts)}</div>',
                         manual=self._candidate_manual_page())

    def _candidates_table(self, candidate_set, decided: dict) -> str:
        blocks = []
        if candidate_set.note:
            blocks.append(h.banner(candidate_set.note, "warn"))
        blocks.append(f'<div class="muted small">Showing the first {min(len(candidate_set), MAX_TABLE_ROWS)} of '
                      f'{len(candidate_set):,}. Click a row to open it in the filmstrip; use triage to work through '
                      f'the whole queue.</div>')
        columns = ["uuid", "class", "suggested", "score"] + list(candidate_set.columns)
        head = "".join(f'<th{" class=num" if c == "score" else ""}>{h.e(c)}</th>' for c in columns)
        rows = []
        for i, candidate in enumerate(candidate_set.candidates[:MAX_TABLE_ROWS]):
            on = " cl-decided" if candidate.uuid in decided else ""
            sugg = (f'<td class="c-acc">{h.e(candidate.suggestion)}</td>' if candidate.suggestion
                    else '<td class="muted">—</td>')
            cells = (f'<td class="mono" title="{h.e(candidate.uuid)}">{h.e(candidate.uuid[:14])}</td>'
                     f'<td>{h.e(candidate.label)}</td>{sugg}<td class="num">{candidate.score:.4f}</td>'
                     + "".join(f'<td>{h.e(candidate.detail.get(column, ""))}</td>' for column in candidate_set.columns))
            rows.append(f'<tr class="clk{on}"{self.a("row", i=i)}>{cells}</tr>')
        blocks.append(f'<div class="tbl-wrap cl-scroll"><table class="tbl compact sticky"><thead><tr>{head}</tr></thead>'
                      f'<tbody>{"".join(rows)}</tbody></table></div>')
        return "".join(blocks)

    def _subcenters_html(self) -> str:
        parts = [f'<div class="row">{h.button("Rank subcenters", act_attr=self.a("rank_subcenters"), kind="primary", icon_name="target", small=True, disabled="subcenters" in self.busy)}</div>',
                 self._progress("subcenters")]
        if self._subcenter_msg is not None:
            parts.append(h.banner(*self._subcenter_msg))
        elif self._subcenter_rows:
            parts.append(self._subcenter_table())
        return h.section("Noisy subcenters", f'<div class="stack cl-tight">{"".join(parts)}</div>',
                         manual="noisy-subcenters", sub="Which of the model's own subcenters collected the most junk.")

    def _subcenter_table(self) -> str:
        """The ranking table, through the class filter: only subcenters that
        would still hold something once opened, with how much."""
        rows = self._subcenter_rows
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
        columns = ["prototype", "assigned", "purity", "compactness", "separation", "noise"]
        if class_filter is not None:
            columns.append(class_filter.describe())
        body = []
        for i, row in shown[:25]:
            cells = (f'<td>{h.e(f"{row.class_label}#{row.k_idx}")}</td><td class="num">{row.n_assigned:,}</td>'
                     f'<td class="num">{row.purity:.3f}</td><td class="num">{row.compactness:.3f}</td>'
                     f'<td class="num">{row.separation:.3f}</td><td class="num"><b>{row.noise_score:.3f}</b></td>')
            if class_filter is not None:
                cells += f'<td class="num c-acc">{matching[i]:,}</td>'
            body.append(f'<tr class="clk"{self.a("subcenter", i=i)}>{cells}</tr>')
        blocks = ['<div class="muted small">One row per (class, subcenter) prototype, worst first. Click a row to load '
                  'its samples as candidates.</div>']
        if class_filter is not None:
            blocks.append(f'<div class="small c-acc">Class filter: {len(shown):,} of {len(rows):,} subcenters hold crops '
                          f'{h.e(class_filter.describe())}; opening one lists only those crops.</div>')
        if shown:
            head = "".join(f'<th{"" if c == "prototype" else " class=num"}>{h.e(c)}</th>' for c in columns)
            blocks.append(f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{head}</tr></thead>'
                          f'<tbody>{"".join(body)}</tbody></table></div>')
        return "".join(blocks)

    def _reports_html(self) -> str:
        parts = [f'<div class="row">'
                 f'{h.button("Measure this version", act_attr=self.a("impact"), icon_name="chart", small=True, disabled="impact" in self.busy)}'
                 f'{h.button("Check the taxonomy", act_attr=self.a("taxonomy"), icon_name="layers", small=True, disabled="taxonomy" in self.busy)}'
                 f'</div>', self._progress("impact")]
        if self._impact_msg is not None:
            parts.append(h.banner(*self._impact_msg))
        elif self._impact is not None:
            parts.append(self._impact_html(*self._impact))
        parts.append(self._progress("taxonomy"))
        if self._taxonomy_msg is not None:
            parts.append(h.banner(*self._taxonomy_msg))
        elif self._taxonomy:
            parts.append(self._taxonomy_html(self._taxonomy))
        return h.section("Did it work?", f'<div class="stack cl-tight">{"".join(parts)}</div>', manual="cleaning-impact",
                         sub="What each version changed, and whether the label set itself is the problem.")

    def _impact_html(self, exp_name: str, impacts: list, breakdown) -> str:
        rows = "".join(
            f'<tr><td><b>{h.e(impact.version_id)}</b></td><td class="num">{impact.reviewed:,}</td>'
            f'<td class="num">{impact.relabelled:,}</td><td class="num">{impact.excluded:,}</td>'
            f'<td class="num">{impact.accuracy_before:.4f}</td><td class="num">{impact.accuracy_after:.4f}</td>'
            f'<td class="num {"c-ok" if impact.delta >= 0 else "c-err"}"><b>{impact.delta:+.4f}</b></td>'
            f'<td class="small">{h.e("  ".join(f"{source} {share:.0%}" for source, share in impact.strategy_precision.items()))}</td></tr>'
            for impact in impacts)
        head = "".join(f'<th{" class=num" if c not in ("version", "flag precision by source") else ""}>{c}</th>'
                       for c in ["version", "reviewed", "relabelled", "excluded", "before", "after", "delta",
                                 "flag precision by source"])
        out = [f'<div class="muted small">\'{h.e(exp_name)}\' scored before and after each version\'s decisions.</div>',
               f'<div class="tbl-wrap"><table class="tbl compact"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>']
        if breakdown is not None:
            out.append(self._breakdown_html(breakdown))
        return "".join(out)

    def _breakdown_html(self, impact) -> str:
        """How the active version moved each metric -- see the original's
        _active_version_breakdown."""
        deltas = [abs(impact.delta(a)) for a, _ in IMPACT_METRICS if impact.delta(a) == impact.delta(a)]
        max_abs = max(deltas) if deltas else 1.0
        rows = []
        for attr, label in IMPACT_METRICS:
            before, after = getattr(impact.before, attr), getattr(impact.after, attr)
            delta = impact.delta(attr)
            color = diverging_color(delta if delta == delta else 0.0, max_abs)
            rows.append(
                f'<tr><td>{h.e(label)}</td>'
                f'<td class="num muted">{f"{before:.4f}" if before == before else "n/a"}</td>'
                f'<td class="num">{f"{after:.4f}" if after == after else "n/a"}</td>'
                f'<td class="num"><b class="cl-delta" style="background:{color}">'
                f'{f"{delta:+.4f}" if delta == delta else "n/a"}</b></td></tr>')
        return (f'<hr class="cl-hr"><div class="cl-sub">Version \'{h.e(impact.version_id)}\' in full</div>'
                f'<div class="muted small"><i>{impact.n_excluded:,} excluded, {impact.n_relabelled:,} relabelled -- '
                f'{impact.before.n_samples:,} rows scored before, {impact.after.n_samples:,} after.</i></div>'
                f'<table class="tbl compact cl-narrow"><thead><tr><th>Metric</th><th class="num">Before</th>'
                f'<th class="num">After</th><th class="num">Change</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')

    def _taxonomy_html(self, hints: list) -> str:
        items = "".join(
            f'<div class="cl-hint"><span class="cl-kind">{h.e(hint.kind.upper())}</span>'
            f'<b>{h.e(hint.a)}{h.e(f" ↔ {hint.b}") if hint.b else ""}</b>'
            f'<span class="muted small">{h.e(hint.detail)}</span></div>'
            for hint in hints[:12])
        return (f'<div class="muted small">Where the finding is about the label set rather than about individual '
                f'labels.</div><div class="stack cl-hints">{items}</div>')

    # ---- snapshots (see ui/view_state.py) ----
    def view_state(self) -> dict:
        """The search that was set up, not its results. `class_scope`
        stores the class *label*, never the positional key."""
        self._sync_params()
        return {
            "source": self.source or "",
            "experiment": self.experiment or "",
            "match": self.match or cand.MATCH_LABEL,
            "class_scope": self._class_scope or "",
            "limit": str(self.fields.get("limit") or ""),
            "margin": str(self.fields.get("margin") or ""),
            "percentile": str(self.fields.get("percentile") or ""),
            "agree": str(self.fields.get("agree") or ""),
            "similarity": str(self.fields.get("similarity") or ""),
            "cross_split": bool(self.cross_split),
            "strategy_other": bool(self.strategy_other),
            "strategy_percentile": bool(self.strategy_percentile),
        }

    def restore_view_state(self, state: dict) -> list:
        warnings = []
        source = state.get("source")
        if source in SOURCE_LABELS:
            self.source = source
        elif source:
            warnings.append(f"Cleaning: unknown candidate source '{source}'.")

        match = state.get("match")
        if match in MATCH_LABELS:
            self.match = match

        experiment = state.get("experiment") or ""
        if experiment:
            if experiment in self.state.experiments:
                self.experiment = experiment
            else:
                warnings.append(f"Cleaning: experiment '{experiment}' is not loaded.")

        scope = state.get("class_scope") or ""
        if scope:
            dataset = self.state.dataset
            labels = dataset.labels() if dataset is not None else []
            if scope in labels:
                self._class_scope = scope
            else:
                warnings.append(f"Cleaning: class '{scope}' is not in this dataset -- searching every class.")
                self._class_scope = None

        for key in ("limit", "margin", "percentile", "agree", "similarity"):
            if state.get(key):
                self.fields[key] = str(state[key])
        for key in ("cross_split", "strategy_other", "strategy_percentile"):
            if key in state:
                setattr(self, key, bool(state[key]))
        self._sync_params()
        self.refresh()
        return warnings
