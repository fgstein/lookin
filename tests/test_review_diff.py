"""Comparing two review versions, and measuring what a cleaning pass bought."""
from __future__ import annotations

import pandas as pd
import pytest

from lookin.core.review import (
    ReviewStore, STATUS_EXCLUDE, STATUS_OK, STATUS_RELABEL, STATUS_UNCERTAIN,
    STATUS_UNREVIEWED, diff_versions,
)


@pytest.fixture
def store(tmp_path) -> ReviewStore:
    return ReviewStore(decisions_path=tmp_path / "d.csv", versions_path=tmp_path / "v.csv")


def test_a_child_that_changed_nothing_shows_no_changes(store):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.decide("b", STATUS_EXCLUDE)
    store.create_version("v2")

    diff = diff_versions(store, "v1", "v2")
    assert diff.total_changes() == 0
    assert diff.unchanged == 2


def test_added_removed_and_changed_are_told_apart(store):
    store.create_version("v1")
    store.decide("keeps", STATUS_OK)
    store.decide("flips", STATUS_OK)
    store.decide("dropped", STATUS_EXCLUDE)
    store.create_version("v2")
    store.decide("flips", STATUS_EXCLUDE)   # changed
    store.decide("fresh", STATUS_UNCERTAIN)  # added
    store.reset("dropped")                   # removed, via a tombstone

    diff = diff_versions(store, "v1", "v2")
    assert [u for u, _ in diff.added] == ["fresh"]
    assert [u for u, _ in diff.removed] == ["dropped"]
    assert [u for u, _, _ in diff.changed] == ["flips"]
    assert diff.unchanged == 1
    assert diff.total_changes() == 3


def test_a_changed_relabel_target_counts_as_a_change(store):
    store.create_version("v1")
    store.decide("a", STATUS_RELABEL, new_label="dog")
    store.create_version("v2")
    store.decide("a", STATUS_RELABEL, new_label="cat")

    diff = diff_versions(store, "v1", "v2")
    assert len(diff.changed) == 1
    assert diff.changed[0][1].new_label == "dog"
    assert diff.changed[0][2].new_label == "cat"


def test_re_deciding_the_same_verdict_is_not_a_change(store):
    """Reason, note, author and timestamp all move on a re-decision; only
    the verdict counts, or every touched crop would look changed."""
    store.create_version("v1")
    store.decide("a", STATUS_EXCLUDE, reason="blurred")
    store.create_version("v2")
    store.decide("a", STATUS_EXCLUDE, reason="occluded", note="second look")

    diff = diff_versions(store, "v1", "v2")
    assert diff.total_changes() == 0
    assert diff.unchanged == 1


def test_transitions_name_both_ends(store):
    store.create_version("v1")
    store.decide("a", STATUS_OK)
    store.decide("b", STATUS_OK)
    store.create_version("v2")
    store.decide("a", STATUS_EXCLUDE)
    store.decide("c", STATUS_EXCLUDE)

    transitions = diff_versions(store, "v1", "v2").transitions()
    assert transitions[(STATUS_OK, STATUS_EXCLUDE)] == 1
    assert transitions[(STATUS_UNREVIEWED, STATUS_EXCLUDE)] == 1, "an added decision came from nothing"


def test_two_independent_passes_can_be_compared(store):
    """A second reviewer's blind pass has no parent in common; the diff is
    still the useful thing to look at."""
    from lookin.core.review import NO_PARENT

    store.create_version("first")
    store.decide("a", STATUS_OK)
    store.decide("b", STATUS_EXCLUDE)
    store.create_version("second", parent=NO_PARENT)
    store.decide("a", STATUS_EXCLUDE)
    store.decide("b", STATUS_EXCLUDE)

    diff = diff_versions(store, "first", "second")
    assert len(diff.changed) == 1 and diff.changed[0][0] == "a"
    assert diff.unchanged == 1


def test_the_diff_is_directional(store):
    store.create_version("v1")
    store.create_version("v2")
    store.decide("a", STATUS_OK)

    forward = diff_versions(store, "v1", "v2")
    backward = diff_versions(store, "v2", "v1")
    assert len(forward.added) == 1 and not forward.removed
    assert len(backward.removed) == 1 and not backward.added


# ---- what a cleaning pass bought ---------------------------------------

def test_review_impact_scores_the_same_experiment_both_ways(loaded_state):
    """"I cleaned 800 crops" is not a result; a metric delta is."""
    from lookin.core.comparison import review_impact

    loaded_state.review.create_version("v1")
    # 'f' is the one crop the experiment gets wrong (true bird, predicted dog)
    loaded_state.review.decide("f", STATUS_RELABEL, new_label="dog")

    impact = review_impact(loaded_state, "exp")
    assert impact.version_id == "v1"
    assert impact.n_relabelled == 1
    assert impact.after.accuracy > impact.before.accuracy
    assert impact.delta("accuracy") > 0


def test_review_impact_counts_exclusions(loaded_state):
    from lookin.core.comparison import review_impact

    loaded_state.review.create_version("v1")
    loaded_state.review.decide("f", STATUS_EXCLUDE)

    impact = review_impact(loaded_state, "exp")
    assert impact.n_excluded == 1
    assert impact.after.n_samples == impact.before.n_samples - 1


def test_review_impact_restores_the_overlay_flag(loaded_state):
    """Leaving it on would make the whole app silently show cleaned
    metrics -- the exact failure the overlay's banner exists to prevent."""
    from lookin.core.comparison import review_impact

    loaded_state.review.create_version("v1")
    loaded_state.review.decide("a", STATUS_OK)

    assert loaded_state.use_reviewed_labels is False
    review_impact(loaded_state, "exp")
    assert loaded_state.use_reviewed_labels is False

    loaded_state.use_reviewed_labels = True
    review_impact(loaded_state, "exp")
    assert loaded_state.use_reviewed_labels is True


def test_review_impact_needs_something_to_measure(loaded_state):
    from lookin.core.comparison import review_impact

    with pytest.raises(ValueError, match="decide on some crops"):
        review_impact(loaded_state, "exp")


# ---- the triage cursor -------------------------------------------------

def test_the_triage_cursor_round_trips(tmp_path, monkeypatch):
    from lookin.core import prefs

    monkeypatch.setattr(prefs, "config_dir", lambda: tmp_path)
    assert prefs.load_triage_cursor("/data/set.csv", "v1") == ""

    prefs.save_triage_cursor("/data/set.csv", "v1", "crop-42")
    assert prefs.load_triage_cursor("/data/set.csv", "v1") == "crop-42"


def test_cursors_are_per_dataset_and_per_version(tmp_path, monkeypatch):
    from lookin.core import prefs

    monkeypatch.setattr(prefs, "config_dir", lambda: tmp_path)
    prefs.save_triage_cursor("/data/a.csv", "v1", "one")
    prefs.save_triage_cursor("/data/a.csv", "v2", "two")
    prefs.save_triage_cursor("/data/b.csv", "v1", "three")

    assert prefs.load_triage_cursor("/data/a.csv", "v1") == "one"
    assert prefs.load_triage_cursor("/data/a.csv", "v2") == "two"
    assert prefs.load_triage_cursor("/data/b.csv", "v1") == "three"


def test_a_cursor_can_be_cleared(tmp_path, monkeypatch):
    from lookin.core import prefs

    monkeypatch.setattr(prefs, "config_dir", lambda: tmp_path)
    prefs.save_triage_cursor("/data/a.csv", "v1", "one")
    assert prefs.clear_triage_cursor("/data/a.csv", "v1") is True
    assert prefs.load_triage_cursor("/data/a.csv", "v1") == ""
    assert prefs.clear_triage_cursor("/data/a.csv", "v1") is False


def test_old_cursors_are_dropped_rather_than_growing_forever(tmp_path, monkeypatch):
    from lookin.core import prefs

    monkeypatch.setattr(prefs, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(prefs, "MAX_TRIAGE_CURSORS", 3)
    for i in range(6):
        prefs.save_triage_cursor(f"/data/{i}.csv", "v1", f"crop-{i}")

    assert prefs.load_triage_cursor("/data/5.csv", "v1") == "crop-5"
    assert prefs.load_triage_cursor("/data/0.csv", "v1") == "", "the oldest is gone"


def test_an_unreadable_cursor_file_is_ignored(tmp_path, monkeypatch):
    """It is a convenience; it must never be what stops a queue opening."""
    from lookin.core import prefs

    monkeypatch.setattr(prefs, "config_dir", lambda: tmp_path)
    (tmp_path / "triage.json").write_text("{not json", encoding="utf-8")
    assert prefs.load_triage_cursor("/data/a.csv", "v1") == ""


# ---- scoring against a chosen version ----------------------------------

def test_metrics_can_be_pinned_to_a_version_other_than_the_active_one(loaded_state):
    """Comparing one model against v1's labels and another against v3's,
    then calling the difference a model improvement, is the mistake this
    prevents."""
    review = loaded_state.review
    review.create_version("v1")
    review.decide("f", STATUS_RELABEL, new_label="dog")   # f: true bird, predicted dog
    review.create_version("v2")
    review.reset("f")                                      # v2 takes it back
    loaded_state.use_reviewed_labels = True

    # Active version is v2, which undoes v1's relabel.
    assert loaded_state.overlay_version_id() == "v2"
    assert bool(loaded_state.merged("exp").set_index("uuid").loc["f", "correct"]) is False

    loaded_state.review_version_id = "v1"
    assert loaded_state.overlay_version_id() == "v1"
    assert bool(loaded_state.merged("exp").set_index("uuid").loc["f", "correct"]) is True


def test_pinning_a_version_invalidates_the_merge(loaded_state):
    loaded_state.review.create_version("v1")
    loaded_state.review.decide("a", STATUS_EXCLUDE)
    loaded_state.review.create_version("v2")
    loaded_state.use_reviewed_labels = True

    first = loaded_state.merged("exp")
    loaded_state.review_version_id = "v1"
    assert loaded_state.merged("exp") is not first


def test_a_pin_to_a_version_that_no_longer_exists_falls_back(loaded_state):
    """A remembered pin must never leave the overlay resolving through
    nothing at all."""
    loaded_state.review.create_version("v1")
    loaded_state.review_version_id = "ghost"
    assert loaded_state.overlay_version_id() == "v1"


def test_clear_all_drops_the_pin(loaded_state):
    loaded_state.review.create_version("v1")
    loaded_state.review_version_id = "v1"
    loaded_state.clear_all()
    assert loaded_state.review_version_id is None
