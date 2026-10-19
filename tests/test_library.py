"""The snapshot library's storage layer.

What matters here is not that a round-trip works -- it is the rules the
library promises and that the UI above it will rely on: a rename keeps the
id, a state change does not; a corrupt file is skipped rather than fatal;
favourites never travel with a workspace; and nothing outside a workspace
directory can be deleted.
"""
from __future__ import annotations

import json

import pytest

from lookin.core import library as lib
from lookin.core.library import Mount, MountExperiment


@pytest.fixture
def root(tmp_path):
    return tmp_path / "library"


@pytest.fixture
def mount(tmp_path):
    csv = tmp_path / "ships.csv"
    csv.write_text("uuid,crop,label\na,crops/a.png,tug\n", encoding="utf-8")
    return Mount(
        dataset_path=str(csv),
        label_col="label",
        image_root=str(tmp_path / "crops"),
        split_kind="test",
        experiments=[MountExperiment(name="baseline", path=str(tmp_path / "baseline_results.csv"))],
    )


@pytest.fixture
def workspace(root, mount):
    return lib.create_workspace("Eval ckpt42 on dataset B", mount, tags=["eval"], root=root)


# ---- ids ----------------------------------------------------------------

def test_ids_avoid_ambiguous_characters():
    """An id is meant to be retyped off a printed report."""
    for _ in range(50):
        body = lib.new_id()[len(lib.SNAPSHOT_PREFIX):]
        assert not set(body) & set("01ilo")


def test_ids_are_distinct_by_prefix():
    assert lib.new_id(lib.WORKSPACE_PREFIX).startswith("lkw-")
    assert lib.new_id(lib.SNAPSHOT_PREFIX).startswith("lk-")


@pytest.mark.parametrize("name,expected", [
    ("Eval ckpt42 on dataset B", "eval-ckpt42-on-dataset-b"),
    ("  spaces  ", "spaces"),
    ("accents: éà", "accents"),
    ("", "workspace"),
    ("///", "workspace"),
])
def test_slugify(name, expected):
    assert lib.slugify(name) == expected


# ---- workspaces ------------------------------------------------------------

def test_create_workspace_writes_a_manifest_and_a_snapshots_dir(workspace):
    assert workspace.exists()
    assert (workspace.path / lib.MANIFEST_NAME).is_file()
    assert workspace.snapshots_dir().is_dir()
    assert workspace.path.name.endswith(lib.WORKSPACE_SUFFIX)
    assert workspace.id in workspace.path.name, "the id is in the directory name, so a rename cannot collide"


def test_workspace_round_trips_its_mount(workspace):
    reloaded = lib.load_workspace(workspace.path)
    assert reloaded.mount.dataset_path == workspace.mount.dataset_path
    assert reloaded.mount.image_root == workspace.mount.image_root
    assert reloaded.mount.split_kind == "test"
    assert [e.name for e in reloaded.mount.experiments] == ["baseline"]
    assert reloaded.tags == ["eval"]


def test_data_dir_is_not_materialised_until_export(workspace):
    """A workspace references its CSVs where they are; only an export copies
    them in. Until then the whole workspace is a few KB."""
    assert not workspace.data_dir().exists()


def test_list_workspaces_skips_an_unreadable_manifest(root, mount, capsys):
    good = lib.create_workspace("good", mount, root=root)
    broken = root / "broken-lkd-000000.lookin"
    broken.mkdir(parents=True)
    (broken / lib.MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    listed = lib.list_workspaces(root)

    assert [b.id for b in listed] == [good.id], "one bad file must not empty the library"
    assert "ignoring unreadable" in capsys.readouterr().out


def test_list_workspaces_ignores_unrelated_directories(root, mount):
    lib.create_workspace("good", mount, root=root)
    (root / "notes").mkdir()
    assert len(lib.list_workspaces(root)) == 1


def test_list_workspaces_on_a_missing_root_is_empty(tmp_path):
    assert lib.list_workspaces(tmp_path / "nope") == []


def test_renaming_a_workspace_keeps_its_id_and_directory(workspace):
    original_id, original_path = workspace.id, workspace.path
    workspace.name = "Renamed"
    workspace.tags = ["eval", "keep"]
    assert lib.save_workspace(workspace)

    reloaded = lib.load_workspace(original_path)
    assert reloaded.id == original_id, "a rename must never move the id a report points at"
    assert reloaded.name == "Renamed"
    assert reloaded.tags == ["eval", "keep"]


def test_delete_workspace_removes_it(workspace, root):
    assert lib.delete_workspace(workspace)
    assert lib.list_workspaces(root) == []


def test_delete_refuses_a_directory_that_is_not_a_workspace(tmp_path, capsys):
    victim = tmp_path / "important"
    victim.mkdir()
    (victim / "keep.txt").write_text("do not delete me", encoding="utf-8")

    assert not lib.delete_workspace(lib.Workspace(id="lkd-aaaaaa", path=victim))

    assert (victim / "keep.txt").is_file()
    assert "refusing to delete" in capsys.readouterr().out


# ---- the dataset hash ---------------------------------------------------

def test_dataset_hash_detects_a_changed_csv(workspace, mount):
    assert workspace.dataset_hash
    assert not workspace.dataset_changed()

    with open(mount.dataset_path, "a", encoding="utf-8") as handle:
        handle.write("b,crops/b.png,barge\n")

    assert workspace.dataset_changed(), "a differing CSV renders plausible, wrong numbers"


def test_dataset_hash_is_quiet_when_it_was_never_taken(root, mount):
    workspace = lib.create_workspace("no hash", mount, root=root, hash_dataset=False)
    assert workspace.dataset_hash == ""
    assert not workspace.dataset_changed(), "no hash means no claim, not a failed check"


def test_hash_file_on_a_missing_file_is_empty(tmp_path, capsys):
    assert lib.hash_file(tmp_path / "gone.csv") == ""
    assert "could not hash" in capsys.readouterr().out


# ---- snapshots ----------------------------------------------------------

def test_snapshot_round_trip(workspace):
    state = {"tab": 6, "projection": {"method": "tsne", "sample": 1500}}
    snapshot = lib.create_snapshot(workspace, state, name="Class 3 confusion")

    reloaded = lib.load_snapshot(workspace, snapshot.id)
    assert reloaded.view_state == state
    assert reloaded.name == "Class 3 confusion"
    assert reloaded.workspace_id == workspace.id
    assert reloaded.view_state_version == lib.VIEW_STATE_VERSION


def test_a_snapshot_written_as_part_of_a_bundle_is_still_read(workspace):
    """Workspaces were called bundles, and snapshots saved then carry a
    "bundle_id" key and an "lkd-" id. Renaming must not orphan them."""
    path = workspace.snapshots_dir() / "lk-legacy.json"
    path.write_text(json.dumps({"id": "lk-legacy", "bundle_id": "lkd-abcdef", "view_state": {}}))
    assert lib.load_snapshot(workspace, "lk-legacy").workspace_id == "lkd-abcdef"


def test_snapshots_are_separate_files(workspace):
    """Ten views of one eval cost ten small files, not ten copies of the
    mount -- and two people saving at once never write the same file."""
    first = lib.create_snapshot(workspace, {"tab": 2})
    second = lib.create_snapshot(workspace, {"tab": 6})

    files = sorted(p.name for p in workspace.snapshots_dir().glob("*.json"))
    assert files == sorted([f"{first.id}.json", f"{second.id}.json"])


def test_renaming_a_snapshot_keeps_its_id(workspace):
    snapshot = lib.create_snapshot(workspace, {"tab": 2}, name="first")
    snapshot.name = "second"
    assert lib.save_snapshot(workspace, snapshot)

    assert lib.load_snapshot(workspace, snapshot.id).name == "second"
    assert len(lib.list_snapshots(workspace)) == 1, "a rename must not fork"


def test_changing_the_state_forks_a_new_snapshot(workspace):
    original = lib.create_snapshot(workspace, {"tab": 2}, name="mine", tags=["x"])
    forked = lib.fork_snapshot(workspace, original, {"tab": 6})

    assert forked.id != original.id
    assert forked.name == "mine" and forked.tags == ["x"], "metadata carries over"
    assert lib.load_snapshot(workspace, original.id).view_state == {"tab": 2}, \
        "the id in a report still shows what it showed"


def test_delete_snapshot(workspace):
    snapshot = lib.create_snapshot(workspace, {"tab": 2})
    assert lib.delete_snapshot(workspace, snapshot.id)
    assert lib.list_snapshots(workspace) == []


def test_list_snapshots_skips_a_corrupt_file(workspace):
    good = lib.create_snapshot(workspace, {"tab": 2})
    (workspace.snapshots_dir() / "lk-broken.json").write_text("{", encoding="utf-8")

    assert [s.id for s in lib.list_snapshots(workspace)] == [good.id]


def test_a_snapshot_from_a_newer_lookin_is_flagged_not_applied(workspace):
    snapshot = lib.create_snapshot(workspace, {"tab": 2})
    path = workspace.snapshots_dir() / f"{snapshot.id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["view_state_version"] = lib.VIEW_STATE_VERSION + 1
    path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = lib.load_snapshot(workspace, snapshot.id)
    assert not reloaded.understood(), "a half-applied view is worse than none"


# ---- finding one by id --------------------------------------------------

def test_find_snapshot_across_workspaces(root, mount):
    first = lib.create_workspace("one", mount, root=root)
    second = lib.create_workspace("two", mount, root=root)
    lib.create_snapshot(first, {"tab": 1})
    wanted = lib.create_snapshot(second, {"tab": 9})

    found = lib.find_snapshot(wanted.id, root=root)

    assert found is not None
    assert found[0].id == second.id
    assert found[1].view_state == {"tab": 9}


def test_find_snapshot_is_case_insensitive_and_trims(root, mount):
    workspace = lib.create_workspace("one", mount, root=root)
    snapshot = lib.create_snapshot(workspace, {"tab": 1})

    assert lib.find_snapshot(f"  {snapshot.id.upper()} ", root=root) is not None


def test_find_snapshot_missing_returns_none(root, mount):
    lib.create_workspace("one", mount, root=root)
    assert lib.find_snapshot("lk-zzzzzz", root=root) is None
    assert lib.find_snapshot("", root=root) is None


# ---- the library root ---------------------------------------------------

def test_library_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(lib.LIBRARY_ENV, str(tmp_path / "shared"))
    lib.set_library_root(None)
    assert lib.library_root() == tmp_path / "shared"


def test_set_library_root_overrides_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(lib.LIBRARY_ENV, str(tmp_path / "shared"))
    lib.set_library_root(tmp_path / "explicit")
    try:
        assert lib.library_root() == tmp_path / "explicit"
    finally:
        lib.set_library_root(None)


# ---- per-user preferences ----------------------------------------------

def test_favorites_live_outside_the_workspace(workspace, monkeypatch, tmp_path):
    """A favourite written into a shared workspace would mean two colleagues
    overwriting each other's."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    assert lib.set_favorite(workspace.id, True)
    assert lib.is_favorite(workspace.id)

    manifest = json.loads((workspace.path / lib.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert "favorites" not in manifest
    assert not lib.personal_path().is_relative_to(workspace.path)


def test_favorites_hold_workspaces_and_snapshots_together(workspace, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    snapshot = lib.create_snapshot(workspace, {"tab": 2})

    lib.set_favorite(workspace.id, True)
    lib.set_favorite(snapshot.id, True)
    assert lib.is_favorite(workspace.id) and lib.is_favorite(snapshot.id)

    lib.set_favorite(workspace.id, False)
    assert not lib.is_favorite(workspace.id)
    assert lib.is_favorite(snapshot.id), "unfavouriting one must not touch the other"


def test_mark_opened_records_a_timestamp(workspace, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert lib.last_opened(workspace.id) == ""
    lib.mark_opened(workspace.id)
    assert lib.last_opened(workspace.id)


def test_personal_file_survives_being_mangled(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    lib.personal_path().parent.mkdir(parents=True, exist_ok=True)
    lib.personal_path().write_text("not json", encoding="utf-8")

    assert lib.load_personal() == {"favorites": [], "opened": {}}
    assert "ignoring unreadable" in capsys.readouterr().out


def test_a_file_that_was_never_written_is_not_logged_as_unreadable(monkeypatch, tmp_path, capsys):
    """First run is not a fault: only a file that exists and cannot be read
    is worth a line, or the real message gets buried."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert lib.load_personal() == {"favorites": [], "opened": {}}
    assert capsys.readouterr().out == ""


def test_update_workspace_changes_the_manifest_as_it_is_now(tmp_path):
    """A page drawn before another tab added an experiment must not erase
    it by saving the copy it holds."""
    ws = lib.create_workspace("A", Mount(dataset_path="/d.csv"), root=tmp_path, hash_dataset=False)
    stale = lib.load_workspace(ws.path)
    lib.update_workspace(ws.path, lambda w: w.mount.experiments.append(MountExperiment("e1", "/e1.csv")) or True)
    stale.mount.experiments.append(MountExperiment("e2", "/e2.csv"))   # what the old page believes
    fresh = lib.update_workspace(ws.path, lambda w: w.mount.experiments.append(MountExperiment("e2", "/e2.csv")) or True)
    assert [e.name for e in fresh.mount.experiments] == ["e1", "e2"]
    assert lib.update_workspace(tmp_path / "gone.lookin", lambda w: True) is None


# ---- updating a workspace in place ---------------------------------------

def test_updating_a_workspace_keeps_its_id_directory_and_snapshots(workspace, tmp_path):
    snapshot = lib.create_snapshot(workspace, {"app": {"tab": 1}})
    other = tmp_path / "other.csv"
    other.write_text("uuid,crop,label\nb,crops/b.png,ferry\n", encoding="utf-8")
    mount = Mount(dataset_path=str(other), split_kind="train",
                  experiments=[MountExperiment(name="v2", path=str(tmp_path / "v2.csv"))])

    fresh = lib.update_workspace_mount(workspace.path, mount)

    assert fresh.id == workspace.id and fresh.path == workspace.path
    assert fresh.mount.dataset_path == str(other)
    assert [e.name for e in fresh.mount.experiments] == ["v2"]
    assert fresh.dataset_hash == lib.hash_file(str(other))
    assert [s.id for s in lib.list_snapshots(fresh)] == [snapshot.id]


def test_updating_keeps_an_mlflow_run_that_was_never_downloaded(workspace, tmp_path):
    pending = MountExperiment(name="remote", path=str(tmp_path / "cache" / "missing.csv"),
                              mlflow_run_id="r1", artifact_path="results.csv")
    lib.update_workspace(workspace.path, lambda w: w.mount.experiments.append(pending) or True)

    fresh = lib.update_workspace_mount(workspace.path, Mount(dataset_path=workspace.mount.dataset_path))

    assert [e.name for e in fresh.mount.experiments] == ["remote"]


# ---- duplicating ----------------------------------------------------------

def test_duplicate_is_a_new_workspace_with_the_same_mount(workspace, root):
    copy = lib.duplicate_workspace(workspace)
    assert copy is not None
    assert copy.id != workspace.id and copy.path != workspace.path
    assert copy.name == "Eval ckpt42 on dataset B (copy)"
    assert copy.tags == ["eval"]
    assert lib.load_workspace(copy.path).mount == workspace.mount
    assert {w.id for w in lib.list_workspaces(root)} == {workspace.id, copy.id}


def test_duplicate_takes_the_name_it_is_given(workspace):
    assert lib.duplicate_workspace(workspace, "  other  ").name == "other"


def test_duplicate_copies_snapshots_under_new_ids(workspace, root):
    snap = lib.create_snapshot(workspace, {"tab": "metrics"}, name="the confusion")
    copy = lib.duplicate_workspace(workspace)
    [copied] = lib.list_snapshots(copy)
    assert copied.id != snap.id and copied.workspace_id == copy.id
    assert (copied.name, copied.view_state, copied.created_at) == ("the confusion", {"tab": "metrics"}, snap.created_at)
    # The original's id still names the original, and only it.
    assert lib.find_snapshot(snap.id, root)[0].id == workspace.id
    assert len(lib.list_snapshots(workspace)) == 1


def test_duplicate_reads_the_manifest_as_it_is_now(workspace):
    lib.update_workspace(workspace.path, lambda w: (setattr(w, "description", "edited elsewhere"), True)[1])
    assert lib.duplicate_workspace(workspace).description == "edited elsewhere"


def test_duplicate_points_its_own_data_at_its_own_copy(workspace):
    """An imported workspace reads its CSVs from its data/; the copy must not
    break when the original is deleted."""
    data = workspace.path / lib.DATA_DIRNAME
    data.mkdir()
    (data / "results.csv").write_text("uuid,tug\na,1\n", encoding="utf-8")
    inside = str(data / "results.csv")
    lib.update_workspace(workspace.path, lambda w: (setattr(w.mount.experiments[0], "path", inside), True)[1])
    copy = lib.duplicate_workspace(workspace)
    assert copy.mount.experiments[0].path == str(copy.path / lib.DATA_DIRNAME / "results.csv")
    assert copy.mount.dataset_path == workspace.mount.dataset_path   # outside: shared
    assert lib.delete_workspace(lib.load_workspace(workspace.path))
    assert lib.load_workspace(copy.path).mount.experiments[0].path
    assert (copy.path / lib.DATA_DIRNAME / "results.csv").is_file()


def test_duplicate_of_a_deleted_workspace_is_none(workspace):
    assert lib.delete_workspace(workspace)
    assert lib.duplicate_workspace(workspace) is None


def test_an_experiment_color_is_saved_with_the_workspace(root, mount):
    mount.experiments[0].color = "#ff8800"
    ws = lib.create_workspace("colored", mount, root=root)
    assert lib.load_workspace(ws.path).mount.experiments[0].color == "#ff8800"
