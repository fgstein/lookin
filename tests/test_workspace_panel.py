"""The Workspaces window, rendered without a browser.

What matters is the behaviour the panel promises over the storage layer
below it: saving a view never overwrites the one an id already points at,
a metadata edit keeps the id, a workspace whose directory is gone is still
listed, and a restore that only partly applied says so.
"""
from __future__ import annotations

import pytest

from lookin.core import library as lib
from lookin.core.data_manager import AppState
from lookin.core.library import Mount
from lookin.ui import save_status
from lookin.ui.workspace_panel import WorkspacePanel


class FakePage:
    width = 1440
    height = 920

    def __init__(self):
        self.opened = []
        self.launched = []

    def open(self, dialog):
        self.opened.append(dialog)

    def launch_url(self, url, web_window_name=None):
        self.launched.append((url, web_window_name))

    def close(self, dialog):
        pass


class FakeUploadView:
    def __init__(self, session=None):
        self._session = session
        self.restored = None

    def current_mount(self):
        return self._session

    def restore_mount(self, mount):
        self.restored = mount

    def is_loading(self):
        return False

    def add_mount_experiments(self, experiments):
        self.added = list(experiments)


class FakeViewState:
    def __init__(self, state=None, warnings=None):
        self._state = state or {"app": {"tab": 6}}
        self._warnings = warnings or []
        self.restored = None

    def capture(self):
        return self._state

    def restore(self, state):
        self.restored = state
        return self._warnings


class FakeCtx:
    def __init__(self, session=None):
        self.page = FakePage()
        self.state = AppState()
        self.upload_view = FakeUploadView(session)
        self.view_state = FakeViewState()


@pytest.fixture
def mount(tmp_path):
    csv = tmp_path / "ships.csv"
    csv.write_text("uuid,crop,label\na,crops/a.png,tug\n", encoding="utf-8")
    return Mount(dataset_path=str(csv), split_kind="test")


@pytest.fixture
def panel(monkeypatch, tmp_path, mount):
    # Never touch the user's real library or preferences.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    lib.set_library_root(tmp_path / "library")
    try:
        yield WorkspacePanel(FakeCtx(session=mount))
    finally:
        lib.set_library_root(None)


# ---- saving -------------------------------------------------------------

def test_save_workspace_writes_one_and_selects_it(panel):
    panel.save_name_field.value = "Eval ckpt42"
    panel._on_save_workspace(None)

    assert panel.selected_workspace is not None
    assert panel.selected_workspace.name == "Eval ckpt42"
    assert [b.name for b in lib.list_workspaces()] == ["Eval ckpt42"]


def test_save_workspace_without_a_dataset_says_so(panel):
    panel.ctx.upload_view = FakeUploadView(session=None)
    panel._on_save_workspace(None)

    assert panel.selected_workspace is None
    assert panel.panel_status.content is not None


def test_save_view_needs_a_workspace(panel):
    panel._on_save_view(None)
    assert panel.selected_snapshot is None


def test_save_view_stores_what_the_registry_captured(panel):
    panel._on_save_workspace(None)
    panel.ctx.view_state = FakeViewState({"app": {"tab": 6}, "views": {"gallery": {"page": 2}}})

    panel._on_save_view(None)

    assert panel.selected_snapshot.view_state["views"]["gallery"] == {"page": 2}


def test_saving_a_view_twice_never_overwrites_the_first(panel):
    """An id quoted in a report has to keep showing what it showed."""
    panel._on_save_workspace(None)
    panel.ctx.view_state = FakeViewState({"app": {"tab": 1}})
    panel._on_save_view(None)
    first = panel.selected_snapshot

    panel.ctx.view_state = FakeViewState({"app": {"tab": 8}})
    panel._on_save_view(None)

    assert panel.selected_snapshot.id != first.id
    assert lib.load_snapshot(panel.selected_workspace, first.id).view_state == {"app": {"tab": 1}}


# ---- metadata edits keep the id ----------------------------------------

def test_editing_workspace_details_keeps_the_id(panel):
    panel._on_save_workspace(None)
    workspace_id = panel.selected_workspace.id

    panel.selected_workspace.name = "Renamed"
    panel.selected_workspace.tags = ["keep"]
    panel._save_workspace_details()

    assert panel.selected_workspace.id == workspace_id
    assert lib.list_workspaces()[0].name == "Renamed"
    assert lib.list_workspaces()[0].tags == ["keep"]


def test_editing_view_details_keeps_the_id_and_forks_nothing(panel):
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    snapshot_id = panel.selected_snapshot.id

    panel.selected_snapshot.name = "Class 3"
    panel._save_snapshot_details()

    assert panel.selected_snapshot.id == snapshot_id
    assert len(lib.list_snapshots(panel.selected_workspace)) == 1


# ---- listing ------------------------------------------------------------

def test_a_workspace_whose_directory_is_gone_is_still_listed(panel):
    panel._on_save_workspace(None)
    lib.delete_workspace(panel.selected_workspace)
    # The panel still holds the record; a reload from disk would drop it,
    # so this is the in-memory case the greying is for.
    panel.selected_workspace = None
    panel._render_workspaces()

    assert panel.workspaces_list.controls, "the list is never left blank without a reason"


def test_filtering_matches_name_tag_and_description(panel):
    panel.save_name_field.value = "alpha"
    panel._on_save_workspace(None)
    panel.selected_workspace.tags = ["nightly"]
    panel.selected_workspace.description = "the good one"
    panel._save_workspace_details()

    for query in ("alpha", "nightly", "good"):
        panel.workspace_filter.value = query
        assert len(panel._sorted_workspaces()) == 1, query

    panel.workspace_filter.value = "nothing like this"
    assert panel._sorted_workspaces() == []


def test_favourites_are_listed_first(panel):
    panel.save_name_field.value = "aaa"
    panel._on_save_workspace(None)
    panel.save_name_field.value = "zzz"
    panel._on_save_workspace(None)
    panel.reload()

    zzz = next(b for b in panel.workspaces if b.name == "zzz")
    lib.set_favorite(zzz.id, True)

    assert panel._sorted_workspaces()[0].name == "zzz"


# ---- opening ------------------------------------------------------------
# Open never loads into the tab it is clicked in: it opens a new browser tab
# on a ?workspace= or ?snapshot= link, and that tab loads it (open_from_link).

def test_open_launches_a_new_tab_and_leaves_this_one_alone(panel):
    panel._on_save_workspace(None)
    panel._on_open_workspace(None)

    assert panel.ctx.page.launched == [(f"?workspace={panel.selected_workspace.id}", "_blank")]
    assert panel.ctx.upload_view.restored is None, "nothing on screen here was replaced"


def test_opening_a_snapshot_launches_its_link(panel):
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    panel._on_open_snapshot(None)

    assert panel.ctx.page.launched == [(f"?snapshot={panel.selected_snapshot.id}", "_blank")]


def test_a_workspace_link_restores_the_mount_but_no_view(panel, mount):
    panel._on_save_workspace(None)
    panel.open_from_link(panel.selected_workspace)

    assert panel.ctx.upload_view.restored.dataset_path == mount.dataset_path
    assert panel.ctx.view_state.restored is None


def test_a_snapshot_link_restores_the_mount_then_the_view(panel):
    panel._on_save_workspace(None)
    panel.ctx.view_state = FakeViewState({"app": {"tab": 6}})
    panel._on_save_view(None)

    panel.open_from_link(panel.selected_workspace, panel.selected_snapshot)

    assert panel.ctx.upload_view.restored is not None
    assert panel.ctx.view_state.restored == {"app": {"tab": 6}}


def test_a_partial_restore_opens_a_dialog_saying_what_was_lost(panel):
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    panel.ctx.view_state = FakeViewState(warnings=["Gallery: experiment 'x' is not loaded."])

    panel.open_from_link(panel.selected_workspace, panel.selected_snapshot)

    assert panel.ctx.page.opened, "a silent partial restore is the failure mode this avoids"


def test_a_clean_restore_still_says_it_is_loaded(panel):
    """Restoring runs a load and then several rebuilds, each reporting
    itself somewhere different -- without one clear "done" there is no
    moment the user can point at and call it finished."""
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    panel.ctx.page.opened.clear()

    panel.open_from_link(panel.selected_workspace, panel.selected_snapshot)

    assert len(panel.ctx.page.opened) == 1


def test_a_view_from_a_newer_lookin_opens_the_data_only(panel):
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    panel.selected_snapshot.view_state_version = lib.VIEW_STATE_VERSION + 1

    panel.open_from_link(panel.selected_workspace, panel.selected_snapshot)

    assert panel.ctx.upload_view.restored is not None, "the data still opens"
    assert panel.ctx.view_state.restored is None, "the view is not half-applied"
    assert panel.ctx.page.opened


def test_a_link_leaves_out_experiments_not_downloaded_and_says_so(panel, mount, tmp_path):
    from lookin.core.library import MountExperiment
    local = tmp_path / "local.csv"
    local.write_text("uuid\n")
    mount.experiments = [
        MountExperiment(name="here", path=str(local)),
        MountExperiment(name="remote", path=str(tmp_path / "not-yet.csv"), mlflow_run_id="r1",
                        artifact_path="eval/x.csv", mlflow_uri="http://m"),
    ]
    workspace = lib.create_workspace("ws", mount, root=lib.library_root())
    panel.ctx.page.opened.clear()

    panel.open_from_link(workspace)

    assert [e.name for e in panel.ctx.upload_view.restored.experiments] == ["here"]
    assert len(panel.ctx.page.opened) == 1


def test_link_parameters_are_read_from_either_routing_style():
    from lookin.app import snapshot_id_in_route, workspace_id_in_route
    assert workspace_id_in_route("/?workspace=lkw-abc234") == "lkw-abc234"
    assert workspace_id_in_route("/#/?workspace=lkw-abc234") == "lkw-abc234"
    assert snapshot_id_in_route("/?snapshot=lk-xyz") == "lk-xyz"
    assert workspace_id_in_route("/?snapshot=lk-xyz") == ""


# ---- deleting -----------------------------------------------------------

def test_deleting_a_workspace_takes_its_views(panel):
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    workspace = panel.selected_workspace

    panel._delete_workspace(workspace)

    assert lib.list_workspaces() == []
    assert panel.selected_workspace is None


def test_deleting_a_view_leaves_the_workspace(panel):
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    snapshot = panel.selected_snapshot

    panel._delete_snapshot(snapshot)

    assert lib.list_snapshots(panel.selected_workspace) == []
    assert len(lib.list_workspaces()) == 1


def test_opening_from_a_url_works_with_the_dialog_never_opened(panel):
    """A ?snapshot= link opens a workspace without anyone having opened the
    Workspaces window first -- Flet refuses to close a dialog it was never given, and
    that used to take the whole first render down."""
    panel._on_save_workspace(None)
    panel._on_save_view(None)
    workspace, snapshot = panel.selected_workspace, panel.selected_snapshot

    fresh = WorkspacePanel(FakeCtx(session=workspace.mount))
    fresh.open_from_link(workspace, snapshot)

    assert fresh.ctx.upload_view.restored is not None
    assert fresh.ctx.view_state.restored == snapshot.view_state


# ---- unsaved work --------------------------------------------------------

def test_nothing_loaded_is_never_unsaved(panel):
    """"Unsaved" over an empty app would be a warning about nothing."""
    panel.ctx.upload_view = FakeUploadView(session=None)
    status = save_status.compute(panel.ctx)

    assert not status.has_data
    assert not status.anything_unsaved()


# ---- a workspace set up from the inside ----------------------------------

def test_a_workspace_opened_without_a_dataset_opens_on_the_data_tab(panel):
    ws = lib.create_workspace("Empty", Mount(), hash_dataset=False)
    jumps = []
    panel.ctx.go_to_data_tab = lambda: jumps.append("data")
    panel.open_from_link(ws)
    assert jumps == ["data"]


def test_the_first_dataset_loaded_in_it_is_written_back(panel, mount):
    ws = lib.create_workspace("Empty", Mount(experiments=[lib.MountExperiment(
        name="run", path="/cache/run.csv", mlflow_run_id="r1", artifact_path="eval/run.csv")]),
        hash_dataset=False)
    panel.ctx.current_workspace = ws
    panel.adopt_dataset()

    fresh = lib.load_workspace(ws.path)
    assert fresh.mount.dataset_path == mount.dataset_path
    assert fresh.dataset_hash, "hashed, so a later change to the CSV is noticed"
    [experiment] = fresh.mount.experiments
    assert experiment.mlflow_run_id == "r1", "the workspace's runs are kept, with their origin"


def test_a_workspace_with_its_own_dataset_is_never_repointed(panel, tmp_path):
    other = tmp_path / "other.csv"
    other.write_text("uuid,crop,label\n", encoding="utf-8")
    ws = lib.create_workspace("Own", Mount(dataset_path=str(other)), hash_dataset=False)
    panel.ctx.current_workspace = ws
    panel.adopt_dataset()
    assert lib.load_workspace(ws.path).mount.dataset_path == str(other)


# ---- updating the open workspace -----------------------------------------

def test_the_open_workspace_can_be_updated_in_place(panel, mount, tmp_path):
    panel._on_save_workspace(None)
    saved = panel.ctx.current_workspace
    panel.reload()
    assert panel.update_workspace_btn.visible
    assert panel.update_workspace_btn.disabled  # nothing to write yet

    changed = Mount(dataset_path=mount.dataset_path, split_kind="train")
    panel.ctx.upload_view = FakeUploadView(changed)
    panel.reload()
    assert not panel.update_workspace_btn.disabled

    panel._update_workspace(saved, changed)
    assert [w.id for w in lib.list_workspaces()] == [saved.id]
    assert lib.list_workspaces()[0].mount.split_kind == "train"
    assert save_status.compute(panel.ctx).workspace_saved


def test_no_update_is_offered_without_an_open_workspace(panel):
    panel.reload()
    assert not panel.update_workspace_btn.visible


# ---- following changes made from Home or another tab ---------------------

def test_a_dataset_set_from_home_loads_into_the_open_empty_tab(panel, mount):
    ws = lib.create_workspace("Empty", Mount(), hash_dataset=False)
    panel.ctx.upload_view = FakeUploadView(session=None)
    panel.open_from_link(ws)

    fresh = lib.update_workspace(ws.path, lambda w: setattr(w.mount, "dataset_path", mount.dataset_path) or True)
    panel.follow_workspace(fresh)

    assert panel.ctx.upload_view.restored.dataset_path == mount.dataset_path
    assert panel.ctx.current_workspace.mount.dataset_path == mount.dataset_path


def test_a_change_made_in_another_tab_is_noticed(panel, mount):
    ws = lib.create_workspace("Empty", Mount(), hash_dataset=False)
    panel.ctx.upload_view = FakeUploadView(session=None)
    panel.open_from_link(ws)
    panel.check_current_workspace()               # nothing changed: nothing loaded
    assert panel.ctx.upload_view.restored.dataset_path == ""

    lib.update_workspace(ws.path, lambda w: setattr(w.mount, "dataset_path", mount.dataset_path) or True)
    panel.check_current_workspace()
    assert panel.ctx.upload_view.restored.dataset_path == mount.dataset_path


def test_runs_added_from_home_load_and_removed_ones_stay_removed(panel, mount, tmp_path):
    kept, removed, added = (tmp_path / f"{n}.csv" for n in ("kept", "removed", "added"))
    for path in (kept, removed, added):
        path.write_text("uuid,a\n", encoding="utf-8")
    ws = lib.create_workspace("WS", Mount(dataset_path=mount.dataset_path, experiments=[
        lib.MountExperiment(name="kept", path=str(kept)),
        lib.MountExperiment(name="removed", path=str(removed))]), hash_dataset=False)
    panel.open_from_link(ws)
    # This tab has "removed" taken out, not saved yet.
    panel.ctx.upload_view = FakeUploadView(Mount(dataset_path=mount.dataset_path, experiments=[
        lib.MountExperiment(name="kept", path=str(kept))]))

    def add(w):
        w.mount.experiments.append(lib.MountExperiment(name="added", path=str(added)))
        return True

    panel.follow_workspace(lib.update_workspace(ws.path, add))
    assert [e.name for e in panel.ctx.upload_view.added] == ["added"]


def test_save_writes_the_open_workspace_in_place(panel, mount):
    ws = lib.create_workspace("WS", Mount(dataset_path=mount.dataset_path), hash_dataset=False)
    panel.ctx.current_workspace = ws
    panel.ctx.upload_view = FakeUploadView(Mount(dataset_path=mount.dataset_path, split_kind="train"))
    assert not save_status.compute(panel.ctx).workspace_saved

    panel.save_current()

    assert [w.id for w in lib.list_workspaces()] == [ws.id]
    assert lib.load_workspace(ws.path).mount.split_kind == "train"
    assert save_status.compute(panel.ctx).workspace_saved


def test_save_without_an_open_workspace_asks_for_a_name(panel):
    panel.save_current()
    assert panel.ctx.page.opened == [panel.dialog]
    assert lib.list_workspaces() == []


def test_the_save_button_shows_only_when_there_is_something_to_save(panel, mount):
    indicator = save_status.SaveStatusIndicator(panel.ctx, on_save=panel.save_current)
    assert indicator.save_button.visible and indicator.save_button.text == "Save as workspace"

    panel._on_save_workspace(None)
    indicator.refresh()
    assert not indicator.save_button.visible
