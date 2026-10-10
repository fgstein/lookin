"""Versioned review state.

This is the module where a silent bug is most expensive: its output is a
`label_review_vN` column that someone trains on. Nothing here raises when
it goes wrong -- it just writes the wrong label -- so the cumulative
resolution, the tombstones, the freezing and the CSV round-trip are all
pinned down explicitly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from lookin.core.review import (
    FrozenVersionError, NO_PARENT, ReviewStore,
    STATUS_EXCLUDE, STATUS_OK, STATUS_RELABEL, STATUS_UNCERTAIN, STATUS_UNREVIEWED,
    export_enriched_dataset, export_excluded_uuids, review_paths_for_csv,
)


@pytest.fixture
def store(tmp_path) -> ReviewStore:
    return ReviewStore(decisions_path=tmp_path / "d.csv", versions_path=tmp_path / "v.csv")


# ---- versions -----------------------------------------------------------

def test_review_paths_are_named_after_the_dataset(tmp_path):
    decisions, versions = review_paths_for_csv(tmp_path / "my_data.csv")
    assert decisions.name == "my_data_review.csv"
    assert versions.name == "my_data_review_versions.csv"
    assert decisions.parent == tmp_path


def test_next_version_id_skips_taken_names(store):
    store.create_version("v1")
    store.create_version("v2")
    assert store.next_version_id() == "v3"


def test_next_version_id_skips_a_user_named_collision(store):
    store.create_version("v1")
    store.create_version("v3")  # user named it v3 themselves
    assert store.next_version_id() == "v2"
    store.create_version("v2")
    assert store.next_version_id() == "v4"


@pytest.mark.parametrize("bad", ["Has Caps", "with-dash", "with space", "x" * 33, "accént"])
def test_create_version_rejects_names_that_cannot_be_a_column_suffix(store, bad):
    with pytest.raises(ValueError):
        store.create_version(bad)


def test_create_version_with_no_name_generates_one(store):
    """An empty name means "pick the next free one", not "invalid"."""
    assert store.create_version().version_id == "v1"
    assert store.create_version("").version_id == "v2"


def test_create_version_defaults_to_inheriting_from_the_active_one(store):
    store.create_version("v1")
    v2 = store.create_version("v2")
    assert v2.parent == "v1"


def test_create_version_can_start_an_independent_pass(store):
    store.create_version("v1")
    v2 = store.create_version("second_reviewer", parent=NO_PARENT)
    assert v2.parent == NO_PARENT
    assert store.chain("second_reviewer") == ["second_reviewer"]


def test_duplicate_version_name_is_refused(store):
    store.create_version("v1")
    with pytest.raises(ValueError):
        store.create_version("v1")


def test_chain_is_root_first(store):
    store.create_version("v1")
    store.create_version("v2")
    store.create_version("v3")
    assert store.chain("v3") == ["v1", "v2", "v3"]


def test_chain_survives_a_hand_edited_cycle(store):
    """A manifest edited by hand can point a version at itself; chain() must
    stop rather than spin forever."""
    store.create_version("v1")
    store.versions["v1"].parent = "v1"
    assert store.chain("v1") == ["v1"]


# ---- cumulative resolution ---------------------------------------------

def test_a_child_version_inherits_its_parents_decisions(store):
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)
    store.create_version("v2")
    store.decide("b", STATUS_OK)

    effective = store.effective("v2")
    assert set(effective) == {"a", "b"}
    assert effective["a"].status == STATUS_EXCLUDE
    assert effective["a"].version_id == "v1", "an inherited decision keeps the version that made it"
    assert effective["b"].version_id == "v2"


def test_a_child_decision_overrides_the_parents(store):
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)
    store.create_version("v2")
    store.decide("a", STATUS_OK)

    assert store.effective("v1")["a"].status == STATUS_EXCLUDE
    assert store.effective("v2")["a"].status == STATUS_OK


def test_a_child_version_only_stores_what_it_changes(store):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.decide("b", STATUS_OK)
    store.create_version("v2")
    store.decide("b", STATUS_EXCLUDE)

    assert set(store.own_decisions("v2")) == {"b"}
    assert len(store.effective("v2")) == 2


# ---- reset / tombstones -------------------------------------------------

def test_reset_drops_a_decision_made_in_this_version(store):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.reset("a")
    assert "a" not in store.effective("v1")
    assert "a" not in store.own_decisions("v1")


def test_reset_writes_a_tombstone_to_cancel_an_inherited_decision(store):
    """Without the tombstone there would be no way to undo a parent's call
    short of unfreezing and editing the parent."""
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)
    store.create_version("v2")
    store.reset("a")

    assert "a" not in store.effective("v2")
    assert store.own_decisions("v2")["a"].status == STATUS_UNREVIEWED
    assert store.effective("v1")["a"].status == STATUS_EXCLUDE, "the parent is untouched"


# ---- decision validation ------------------------------------------------

def test_relabel_without_a_target_class_is_refused(store):
    store.create_version("v1")
    with pytest.raises(ValueError):
        store.decide("a", STATUS_RELABEL)


def test_unknown_status_is_refused(store):
    store.create_version("v1")
    with pytest.raises(ValueError):
        store.decide("a", "probably_fine")


def test_uuids_are_stored_as_strings(store):
    """Crop ids come from a CSV and can parse as ints; the overlay looks
    them up by str, so they have to be stored that way."""
    store.create_version("v1")
    store.decide(12345, STATUS_OK)
    assert "12345" in store.effective("v1")


# ---- freezing -----------------------------------------------------------

def test_a_frozen_version_refuses_new_decisions(store):
    store.create_version("v1")
    store.freeze("v1")
    with pytest.raises(FrozenVersionError):
        store.decide("a", STATUS_OK)


def test_a_frozen_version_refuses_a_reset(store):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.freeze("v1")
    with pytest.raises(FrozenVersionError):
        store.reset("a")


def test_resume_picks_the_newest_unfrozen_version(store):
    store.create_version("v1")
    store.create_version("v2")
    store.freeze("v2")
    assert store.resume_version_id() == "v1"


def test_resume_returns_none_when_everything_is_frozen(store):
    store.create_version("v1")
    store.freeze("v1")
    assert store.resume_version_id() is None


def test_ensure_active_version_creates_a_child_when_all_are_frozen(store):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.freeze("v1")

    fresh = store.ensure_active_version()
    assert fresh.version_id != "v1"
    assert fresh.parent == "v1", "the new pass keeps inheriting the frozen work"
    assert store.effective(fresh.version_id)["a"].status == STATUS_OK


# ---- batches ------------------------------------------------------------

def test_a_bulk_action_is_undoable_as_one_unit(store):
    store.create_version("v1")
    store.decide("keep_me", STATUS_OK)
    batch = store.decide_many(["a", "b", "c"], STATUS_EXCLUDE)

    assert len(store.effective("v1")) == 4
    assert store.undo_batch(batch) == 3
    assert set(store.effective("v1")) == {"keep_me"}


def test_undo_batch_leaves_frozen_versions_alone(store):
    store.create_version("v1")
    batch = store.decide_many(["a", "b"], STATUS_EXCLUDE)
    store.freeze("v1")
    assert store.undo_batch(batch) == 0
    assert len(store.effective("v1")) == 2


def test_decide_many_accepts_an_iterator(store):
    """The count in the log line used to be read after the loop had already
    consumed the argument."""
    store.create_version("v1")
    store.decide_many(iter(["a", "b", "c"]), STATUS_OK)
    assert len(store.effective("v1")) == 3


# ---- counts -------------------------------------------------------------

def test_counts_are_over_the_resolved_state(store):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.decide("b", STATUS_EXCLUDE)
    store.create_version("v2")
    store.decide("a", STATUS_EXCLUDE)

    assert store.counts("v1") == {STATUS_OK: 1, STATUS_RELABEL: 0, STATUS_EXCLUDE: 1, STATUS_UNCERTAIN: 0}
    assert store.counts("v2") == {STATUS_OK: 0, STATUS_RELABEL: 0, STATUS_EXCLUDE: 2, STATUS_UNCERTAIN: 0}


# ---- persistence --------------------------------------------------------

def test_save_load_round_trip(store, tmp_path):
    store.create_version("v1", note="first pass")
    store.decide("a", STATUS_RELABEL, new_label="dog", reason="ambiguous", note="hard call")
    store.decide("b", STATUS_EXCLUDE)
    store.create_version("v2")
    store.decide("c", STATUS_UNCERTAIN)
    store.freeze("v1")
    assert store.save() is True

    reloaded = ReviewStore.load(tmp_path / "d.csv", tmp_path / "v.csv")
    assert list(reloaded.versions) == ["v1", "v2"]
    assert reloaded.versions["v1"].frozen is True
    assert reloaded.versions["v1"].note == "first pass"
    assert reloaded.versions["v2"].parent == "v1"

    decision = reloaded.effective("v2")["a"]
    assert (decision.status, decision.new_label, decision.reason) == (STATUS_RELABEL, "dog", "ambiguous")
    assert len(reloaded.effective("v2")) == 3
    assert reloaded.active_version_id == "v2", "resumes the unfrozen version"


def test_save_is_a_no_op_when_nothing_changed(store):
    store.create_version("v1")
    assert store.save() is True
    assert store.save() is False
    assert store.save(force=True) is True


def test_load_tolerates_missing_files(tmp_path):
    store = ReviewStore.load(tmp_path / "nope.csv", tmp_path / "nope_versions.csv")
    assert store.versions == {}
    assert store.active_version_id is None


def test_load_rebuilds_a_version_the_manifest_lost(store, tmp_path):
    """Work must survive a lost/truncated manifest rather than being dropped."""
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.save()
    (tmp_path / "v.csv").unlink()

    reloaded = ReviewStore.load(tmp_path / "d.csv", tmp_path / "v.csv")
    assert "v1" in reloaded.versions
    assert reloaded.effective("v1")["a"].status == STATUS_OK


def test_revision_moves_on_every_mutation(store):
    """AppState.merged folds this into its cache key."""
    store.create_version("v1")
    before = store.revision
    store.decide("a", STATUS_OK)
    assert store.revision > before


# ---- enrich / export ----------------------------------------------------

def test_enrich_fills_every_row(store, tiny_dataset_df):
    store.create_version("v1")
    store.decide("a", STATUS_RELABEL, new_label="dog", reason="ambiguous")
    store.decide("b", STATUS_EXCLUDE)

    out, notes = store.enrich(tiny_dataset_df, "uuid", "label", ["v1"])
    assert notes == []
    by_uuid = out.set_index("uuid")

    # relabelled: the corrected label
    assert by_uuid.loc["a", "label_review_v1"] == "dog"
    assert by_uuid.loc["a", "review_status_v1"] == STATUS_RELABEL
    assert by_uuid.loc["a", "review_reason_v1"] == "ambiguous"
    # excluded: keeps its own label, exclusion lives in the status column
    assert by_uuid.loc["b", "label_review_v1"] == "cat"
    assert by_uuid.loc["b", "review_status_v1"] == STATUS_EXCLUDE
    # untouched: the original label, so the column needs no fallback logic
    assert by_uuid.loc["c", "label_review_v1"] == "dog"
    assert by_uuid.loc["c", "review_status_v1"] == STATUS_UNREVIEWED
    assert list(tiny_dataset_df.columns) == [c for c in out.columns if not c.endswith("_v1")]


def test_enrich_does_not_overwrite_an_existing_column(store, tiny_dataset_df):
    df = tiny_dataset_df.assign(label_review_v1="something precious")
    store.create_version("v1")
    store.decide("a", STATUS_OK)

    out, notes = store.enrich(df, "uuid", "label", ["v1"])
    assert (out["label_review_v1"] == "something precious").all()
    assert "label_review_v1_2" in out.columns
    assert any("label_review_v1" in n for n in notes)


def test_enrich_writes_one_column_set_per_version(store, tiny_dataset_df):
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)
    store.create_version("v2")
    store.decide("a", STATUS_OK)

    out, _ = store.enrich(tiny_dataset_df, "uuid", "label", ["v1", "v2"])
    by_uuid = out.set_index("uuid")
    assert by_uuid.loc["a", "review_status_v1"] == STATUS_EXCLUDE
    assert by_uuid.loc["a", "review_status_v2"] == STATUS_OK


def test_export_reads_the_whole_csv_not_the_loaded_split(store, tmp_path, tiny_dataset_df):
    """The app usually holds one split; an export that only wrote those rows
    would hand back a truncated training file."""
    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)

    result = export_enriched_dataset(
        store, csv, "uuid", "label", ["v1"], tmp_path / "out.csv",
    )
    written = pd.read_csv(tmp_path / "out.csv")
    assert result.rows == len(tiny_dataset_df) == len(written)
    assert result.excluded == 1


def test_export_freezes_what_it_exported(store, tmp_path, tiny_dataset_df):
    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    store.create_version("v1")
    store.decide("a", STATUS_OK)

    export_enriched_dataset(store, csv, "uuid", "label", ["v1"], tmp_path / "out.csv")
    assert store.is_frozen("v1")
    assert store.versions["v1"].exported_to.endswith("out.csv")


def test_export_can_drop_excluded_rows(store, tmp_path, tiny_dataset_df):
    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)

    result = export_enriched_dataset(
        store, csv, "uuid", "label", ["v1"], tmp_path / "out.csv", drop_excluded=True,
    )
    assert result.rows == len(tiny_dataset_df) - 1


def test_export_refuses_a_missing_column(store, tmp_path, tiny_dataset_df):
    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    store.create_version("v1")
    with pytest.raises(ValueError):
        export_enriched_dataset(store, csv, "not_a_column", "label", ["v1"], tmp_path / "out.csv")


def test_export_excluded_uuids_writes_a_plain_list(store, tmp_path):
    store.create_version("v1")
    store.decide_many(["b", "a"], STATUS_EXCLUDE)
    result = export_excluded_uuids(store, "v1", tmp_path / "excluded.txt")
    assert (tmp_path / "excluded.txt").read_text().split() == ["a", "b"]
    assert result.excluded == 2


# ---- round trip through the dataset CSV ---------------------------------

def test_adopt_picks_versions_back_up_from_an_exported_csv(store, tmp_path, tiny_dataset_df):
    """Export -> train -> load that file back here: the versions it encodes
    have to come back, otherwise the enriched CSV is a dead end."""
    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    store.create_version("v1")
    store.decide("a", STATUS_RELABEL, new_label="dog", reason="ambiguous")
    store.decide("b", STATUS_EXCLUDE)
    export_enriched_dataset(store, csv, "uuid", "label", ["v1"], tmp_path / "out.csv")

    exported = pd.read_csv(tmp_path / "out.csv")
    fresh = ReviewStore()
    adopted = fresh.adopt_from_dataframe(exported, "uuid", "label")

    assert adopted == ["v1"]
    assert fresh.versions["v1"].frozen is True, "adopted work is history, not an open pass"
    effective = fresh.effective("v1")
    assert effective["a"].status == STATUS_RELABEL
    assert effective["a"].new_label == "dog"
    assert effective["b"].status == STATUS_EXCLUDE
    assert "c" not in effective, "unreviewed rows are not decisions"


def test_adopt_ignores_versions_already_known(store, tiny_dataset_df):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    df = tiny_dataset_df.assign(review_status_v1=STATUS_EXCLUDE, label_review_v1="dog")
    assert store.adopt_from_dataframe(df, "uuid", "label") == []
    assert store.effective("v1")["a"].status == STATUS_OK


def test_adopted_versions_are_chained_so_a_new_pass_inherits(tiny_dataset_df):
    df = tiny_dataset_df.assign(
        review_status_v1=STATUS_OK, label_review_v1=tiny_dataset_df["label"],
        review_status_v2=STATUS_OK, label_review_v2=tiny_dataset_df["label"],
    )
    fresh = ReviewStore()
    adopted = fresh.adopt_from_dataframe(df, "uuid", "label")
    assert adopted == ["v1", "v2"]
    assert fresh.versions["v1"].parent == NO_PARENT
    assert fresh.versions["v2"].parent == "v1"


# ---- overlay ------------------------------------------------------------

def test_overlay_matches_enrich(store, tiny_dataset_df):
    """The in-app overlay and the exported columns must never disagree --
    that is what makes 'metrics on the cleaned pool' mean the same thing as
    the file the user trains on."""
    store.create_version("v1")
    store.decide("a", STATUS_RELABEL, new_label="dog")
    store.decide("b", STATUS_EXCLUDE)
    store.decide("c", STATUS_OK)

    labels, statuses = store.overlay(
        tiny_dataset_df["uuid"].to_numpy(), tiny_dataset_df["label"].to_numpy(), "v1",
    )
    enriched, _ = store.enrich(tiny_dataset_df, "uuid", "label", ["v1"])
    assert list(labels) == list(enriched["label_review_v1"])
    assert list(statuses) == list(enriched["review_status_v1"])


# ---- an export must never eat what it was made from --------------------

def test_export_refuses_to_overwrite_the_source_dataset(store, tmp_path, tiny_dataset_df):
    """The dataset CSV is an input. The whole design keeps decisions out of
    it so it is never rewritten, and it may be read-only, shared, or the
    only copy."""
    from lookin.core.review import DestinationRefused

    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    before = csv.read_text()
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)

    with pytest.raises(DestinationRefused, match="dataset CSV"):
        export_enriched_dataset(store, csv, "uuid", "label", ["v1"], csv)
    assert csv.read_text() == before, "not a byte of the source moved"
    assert not store.is_frozen("v1"), "and the version was not frozen for an export that did not happen"


def test_export_refuses_to_overwrite_the_decisions_file(tmp_path, tiny_dataset_df):
    """One letter from the default export name: 'X_review.csv' holds the
    decisions, 'X_reviewed.csv' is the export."""
    from lookin.core.review import DestinationRefused

    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    decisions, versions = review_paths_for_csv(csv)
    store = ReviewStore(decisions_path=decisions, versions_path=versions)
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)
    store.save()
    before = decisions.read_text()

    with pytest.raises(DestinationRefused, match="review decision"):
        export_enriched_dataset(store, csv, "uuid", "label", ["v1"], decisions)
    assert decisions.read_text() == before
    assert ReviewStore.load(decisions, versions).effective("v1")["a"].status == STATUS_EXCLUDE


def test_export_refuses_to_overwrite_the_version_manifest(tmp_path, tiny_dataset_df):
    from lookin.core.review import DestinationRefused

    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    decisions, versions = review_paths_for_csv(csv)
    store = ReviewStore(decisions_path=decisions, versions_path=versions)
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.save()

    with pytest.raises(DestinationRefused, match="version manifest"):
        export_enriched_dataset(store, csv, "uuid", "label", ["v1"], versions)


def test_a_relative_path_to_the_source_is_refused_too(store, tmp_path, tiny_dataset_df, monkeypatch):
    """A string comparison would let '../dir/dataset.csv' through."""
    from lookin.core.review import DestinationRefused

    (tmp_path / "data").mkdir()
    csv = tmp_path / "data" / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    store.create_version("v1")
    store.decide("a", STATUS_OK)

    monkeypatch.chdir(tmp_path / "data")
    with pytest.raises(DestinationRefused):
        export_enriched_dataset(store, csv, "uuid", "label", ["v1"], "./full.csv")
    with pytest.raises(DestinationRefused):
        export_enriched_dataset(store, csv, "uuid", "label", ["v1"], "../data/full.csv")


def test_a_normal_destination_is_still_accepted(store, tmp_path, tiny_dataset_df):
    """The guard must not get in the way of the actual workflow, including
    re-exporting over a previous export."""
    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    store.create_version("v1")
    store.decide("a", STATUS_OK)

    dest = tmp_path / "full_reviewed.csv"
    assert export_enriched_dataset(store, csv, "uuid", "label", ["v1"], dest).rows == 6
    assert export_enriched_dataset(store, csv, "uuid", "label", ["v1"], dest).rows == 6, \
        "re-exporting over a previous export is fine"


def test_the_uuid_export_refuses_the_decisions_file(tmp_path, tiny_dataset_df):
    from lookin.core.review import DestinationRefused

    csv = tmp_path / "full.csv"
    tiny_dataset_df.to_csv(csv, index=False)
    decisions, versions = review_paths_for_csv(csv)
    store = ReviewStore(decisions_path=decisions, versions_path=versions)
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE)
    store.save()

    with pytest.raises(DestinationRefused):
        export_excluded_uuids(store, "v1", decisions)
    assert export_excluded_uuids(store, "v1", tmp_path / "excluded.txt").excluded == 1


# ---- two sessions on one dataset --------------------------------------------
# Every browser tab is its own session, and two open workspaces often share
# a dataset -- so two stores on the same sidecars must not erase each other.

def _two_stores(tmp_path):
    from lookin.core.review import ReviewStore, review_paths_for_csv
    decisions, versions = review_paths_for_csv(tmp_path / "ships.csv")
    first = ReviewStore.load(decisions, versions)
    first.ensure_active_version()
    first.save(force=True)
    return ReviewStore.load(decisions, versions), ReviewStore.load(decisions, versions), (decisions, versions)


def test_two_sessions_saving_keep_each_others_decisions(tmp_path):
    from lookin.core.review import ReviewStore, STATUS_EXCLUDE, STATUS_OK
    a, b, paths = _two_stores(tmp_path)
    a.decide("u1", STATUS_EXCLUDE)
    a.save()
    b.decide("u2", STATUS_OK)
    b.save()
    on_disk = ReviewStore.load(*paths)
    assert set(on_disk.effective()) == {"u1", "u2"}
    assert set(b.effective()) == {"u1", "u2"}       # b now sees a's work too


def test_an_undo_in_one_session_is_not_resurrected_by_the_other(tmp_path):
    from lookin.core.review import ReviewStore, STATUS_EXCLUDE, STATUS_OK
    a, b, paths = _two_stores(tmp_path)
    batch = a.decide_many(["u1", "u2"], STATUS_EXCLUDE)
    a.save()
    b.save(force=True)          # b merges: it now holds u1 and u2
    b.undo_batch(batch)
    b.save()
    a.decide("u3", STATUS_OK)
    a.save()                    # a still has u1, u2 in memory, but did not touch them since
    assert set(ReviewStore.load(*paths).effective()) == {"u3"}


def test_the_same_crop_decided_twice_keeps_the_later_save(tmp_path):
    from lookin.core.review import ReviewStore, STATUS_EXCLUDE, STATUS_OK
    a, b, paths = _two_stores(tmp_path)
    a.decide("u1", STATUS_EXCLUDE)
    a.save()
    b.decide("u1", STATUS_OK)
    b.save()
    assert ReviewStore.load(*paths).effective()["u1"].status == STATUS_OK


def test_a_version_created_in_each_session_both_survive(tmp_path):
    from lookin.core.review import ReviewStore
    a, b, paths = _two_stores(tmp_path)
    a.create_version("pass_a")
    a.save()
    b.create_version("pass_b")
    b.save()
    assert {"pass_a", "pass_b"} <= set(ReviewStore.load(*paths).versions)


def test_a_version_frozen_elsewhere_refuses_late_decisions(tmp_path):
    from lookin.core.review import ReviewStore, STATUS_EXCLUDE
    a, b, paths = _two_stores(tmp_path)
    version = a.active_version_id
    a.freeze(version)
    a.save()
    b.decide("u9", STATUS_EXCLUDE, version_id=version)   # b has not seen the freeze yet
    b.save()
    on_disk = ReviewStore.load(*paths)
    assert on_disk.is_frozen(version) and "u9" not in on_disk.own_decisions(version)
