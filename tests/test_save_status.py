"""Is what is on screen saved, and does the indicator tell the truth.

The rule under test is the one that makes the whole thing worth having:
restoring a snapshot must leave the state *equal* to that snapshot. If it
does not, the top bar reports freshly restored work as already modified,
and the warning before a load fires every time -- which is how a warning
stops being read.
"""
from __future__ import annotations

import pytest

from lookin.core.library import Mount, MountExperiment
from lookin.web import save_status
from lookin.web.render import save_chips
from lookin.web.save_status import SaveStatus, compute, mark_baseline


class FakeViewState:
    def __init__(self, state):
        self.state = state

    def capture(self):
        return self.state


class FakeUploadView:
    def __init__(self, mount):
        self._mount = mount

    def current_mount(self):
        return self._mount


class FakeWorkspace:
    def __init__(self, mount, name="eval", present=True):
        self.mount = mount
        self._name = name
        self._present = present

    def label(self):
        return self._name

    def exists(self):
        return self._present


class FakeSnapshot:
    def __init__(self, view_state, name="lk-abc123"):
        self.view_state = view_state
        self._name = name

    def label(self):
        return self._name


class FakeSession:
    def __init__(self, mount=None, view=None):
        self.upload_view = FakeUploadView(mount) if mount is not None else None
        self.view_state = FakeViewState(view if view is not None else {"app": {"tab": 0}})
        self.current_workspace = None
        self.current_snapshot = None
        self.view_state_baseline = None


@pytest.fixture
def mount(tmp_path):
    return Mount(dataset_path=str(tmp_path / "ships.csv"), split_kind="test",
                 experiments=[MountExperiment(name="baseline", path="b.csv")])


# ---- the workspace half ----------------------------------------------------

def test_a_workspace_holding_the_loaded_mount_is_saved(mount):
    session = FakeSession(mount=mount)
    session.current_workspace = FakeWorkspace(mount)

    status = compute(session)

    assert status.workspace_saved
    assert status.workspace_name == "eval"


def test_a_workspace_whose_directory_is_gone_is_not_saved(mount):
    session = FakeSession(mount=mount)
    session.current_workspace = FakeWorkspace(mount, present=False)
    assert not compute(session).workspace_saved


def test_loading_a_different_dataset_un_saves_the_workspace(mount, tmp_path):
    """Which is exactly what the user needs to be told before they open
    something else."""
    session = FakeSession(mount=mount)
    session.current_workspace = FakeWorkspace(Mount(dataset_path=str(tmp_path / "other.csv")))
    assert not compute(session).workspace_saved


def test_a_restat_of_the_same_file_does_not_un_save_it(mount):
    """size/mtime move on their own and say nothing about the user's work."""
    stale = Mount(dataset_path=mount.dataset_path, split_kind="test",
                  experiments=list(mount.experiments),
                  dataset_size=123, dataset_mtime=1.0, saved_at="yesterday")
    session = FakeSession(mount=mount)
    session.current_workspace = FakeWorkspace(stale)
    assert compute(session).workspace_saved


# ---- the snapshot half --------------------------------------------------

def test_a_snapshot_equal_to_the_state_is_saved(mount):
    view = {"app": {"tab": 6}}
    session = FakeSession(mount=mount, view=view)
    session.current_snapshot = FakeSnapshot(dict(view))
    assert compute(session).snapshot_saved


def test_a_snapshot_that_no_longer_matches_is_not(mount):
    session = FakeSession(mount=mount, view={"app": {"tab": 6}})
    session.current_snapshot = FakeSnapshot({"app": {"tab": 1}})

    status = compute(session)

    assert not status.snapshot_saved
    assert status.snapshot_name == "lk-abc123", "the name is kept, so the chip can say which"


# ---- what actually triggers the warning ---------------------------------

def test_nothing_loaded_is_never_unsaved():
    assert not compute(FakeSession()).anything_unsaved()


def test_data_in_no_workspace_is_unsaved(mount):
    assert compute(FakeSession(mount=mount)).anything_unsaved()


def test_a_saved_workspace_with_untouched_settings_does_not_warn(mount):
    """The moment right after saving is the one moment nothing is at risk.
    A warning that cries wolf is one nobody reads."""
    session = FakeSession(mount=mount, view={"app": {"tab": 6}})
    session.current_workspace = FakeWorkspace(mount)
    mark_baseline(session)

    assert not compute(session).anything_unsaved()


def test_changing_the_settings_after_a_save_does_warn(mount):
    session = FakeSession(mount=mount, view={"app": {"tab": 6}})
    session.current_workspace = FakeWorkspace(mount)
    mark_baseline(session)

    session.view_state = FakeViewState({"app": {"tab": 2}})

    status = compute(session)
    assert status.view_changed
    assert status.anything_unsaved()


def test_a_baseline_is_ignored_when_no_data_is_loaded():
    session = FakeSession()
    session.view_state_baseline = {"app": {"tab": 9}}
    assert not compute(session).view_changed


# ---- the chips in the top bar -------------------------------------------

def test_the_chips_show_both_names_when_saved(mount):
    session = FakeSession(mount=mount, view={"app": {"tab": 6}})
    session.current_workspace = FakeWorkspace(mount, name="eval ckpt42")
    session.current_snapshot = FakeSnapshot({"app": {"tab": 6}}, name="class 3")

    html = save_chips(session)

    assert "eval ckpt42</button>" in html
    assert "class 3</button>" in html


def test_a_drifted_snapshot_is_marked_with_its_name(mount):
    """More use than a bare "unsaved", which loses which one it drifted
    from."""
    session = FakeSession(mount=mount, view={"app": {"tab": 1}})
    session.current_workspace = FakeWorkspace(mount)
    session.current_snapshot = FakeSnapshot({"app": {"tab": 6}}, name="class 3")

    assert "class 3 *</button>" in save_chips(session)


def test_nothing_loaded_says_so_rather_than_unsaved():
    html = save_chips(FakeSession())
    assert "Nothing loaded yet." in html
    assert "*" not in html


def test_a_long_name_is_elided_so_the_top_bar_does_not_grow():
    assert len(save_status.elide("a" * 80)) <= save_status.MAX_NAME_CHARS
    assert save_status.elide("short") == "short"


def test_the_indicator_never_raises_on_a_broken_context():
    """It drives a chip in the top bar; one that can take the app down is
    worse than no chip."""
    class Exploding:
        def capture(self):
            raise RuntimeError("boom")

    session = FakeSession()
    session.view_state = Exploding()
    assert isinstance(compute(session), SaveStatus)
