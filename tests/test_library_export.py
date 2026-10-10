"""Exporting a workspace and importing it somewhere else.

This is the half that makes a workspace worth anything to a colleague, and the
half with the sharpest failure modes: a path from the sender's machine that
silently resolves to nothing, a dataset that is not quite the one the view
was saved against, and an archive that writes outside where it was told to.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from lookin.core import library as lib
from lookin.core.library import Mount, MountExperiment


@pytest.fixture
def sandbox(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "ships.csv").write_text("uuid,crop,label\na,crops/a.png,tug\n", encoding="utf-8")
    (data / "baseline_results.csv").write_text("uuid,tug,barge\na,0.9,0.1\n", encoding="utf-8")
    (data / "ships_review.csv").write_text("uuid,status\na,ok\n", encoding="utf-8")
    (data / "baseline_embeddings.npz").write_bytes(b"not really an npz, but bytes")
    return tmp_path


@pytest.fixture
def mount(sandbox):
    data = sandbox / "data"
    return Mount(
        dataset_path=str(data / "ships.csv"),
        image_root=str(sandbox / "crops_here"),
        split_kind="test",
        experiments=[MountExperiment(
            name="baseline",
            path=str(data / "baseline_results.csv"),
            embeddings_path=str(data / "baseline_embeddings.npz"),
        )],
    )


@pytest.fixture
def workspace(sandbox, mount):
    return lib.create_workspace("Eval one", mount, root=sandbox / "library", tags=["nightly"])


# ---- export -------------------------------------------------------------

def test_export_carries_the_data_but_not_the_crops(sandbox, workspace):
    dest = sandbox / "out" / "eval.zip"
    ok, message = lib.export_workspace(workspace, dest)

    assert ok, message
    with zipfile.ZipFile(dest) as archive:
        names = archive.namelist()
    assert "manifest.json" in names
    assert "data/ships.csv" in names
    assert "data/baseline_results.csv" in names
    assert "data/ships_review.csv" in names, "review decisions are part of the view"
    assert not any("crops" in n for n in names), "crops never travel -- they are what keeps this small"


def test_embeddings_are_left_out_unless_asked_for(sandbox, workspace):
    plain, with_embeddings = sandbox / "a.zip", sandbox / "b.zip"
    lib.export_workspace(workspace, plain)
    lib.export_workspace(workspace, with_embeddings, include_embeddings=True)

    assert "data/baseline_embeddings.npz" not in zipfile.ZipFile(plain).namelist()
    assert "data/baseline_embeddings.npz" in zipfile.ZipFile(with_embeddings).namelist()


def test_exported_paths_are_relative_to_the_workspace(sandbox, workspace):
    dest = sandbox / "eval.zip"
    lib.export_workspace(workspace, dest)

    with zipfile.ZipFile(dest) as archive:
        manifest = json.loads(archive.read("manifest.json"))

    assert manifest["mount"]["dataset_path"] == "data/ships.csv"
    assert manifest["mount"]["experiments"][0]["path"] == "data/baseline_results.csv"
    assert manifest["mount"]["image_root"] == "", \
        "a crop path from the sender's machine would only mislead"


def test_export_takes_the_saved_views_with_it(sandbox, workspace):
    snapshot = lib.create_snapshot(workspace, {"app": {"tab": 6}}, name="Class 3")
    dest = sandbox / "eval.zip"
    lib.export_workspace(workspace, dest)

    with zipfile.ZipFile(dest) as archive:
        assert f"snapshots/{snapshot.id}.json" in archive.namelist()


def test_export_reports_a_file_that_has_gone(sandbox, workspace, mount):
    (sandbox / "data" / "baseline_results.csv").unlink()
    ok, message = lib.export_workspace(workspace, sandbox / "eval.zip")

    assert ok, "the rest still exports"
    assert "missing" in message.lower()
    assert "baseline_results.csv" in message


def test_export_of_a_workspace_with_no_dataset_is_refused(sandbox):
    empty = lib.create_workspace("empty", Mount(), root=sandbox / "library")
    ok, message = lib.export_workspace(empty, sandbox / "x.zip")
    assert not ok and "no dataset" in message.lower()


def test_two_experiments_sharing_a_basename_do_not_overwrite(sandbox, mount):
    """Different directories, same file name -- the second silently winning
    would be a workspace that scores one model twice."""
    other = sandbox / "other"
    other.mkdir()
    (other / "baseline_results.csv").write_text("uuid,tug,barge\na,0.2,0.8\n", encoding="utf-8")
    mount.experiments.append(MountExperiment(name="second", path=str(other / "baseline_results.csv")))
    workspace = lib.create_workspace("two", mount, root=sandbox / "library")

    dest = sandbox / "two.zip"
    lib.export_workspace(workspace, dest)

    with zipfile.ZipFile(dest) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        payloads = {archive.read(e["path"]) for e in manifest["mount"]["experiments"]}
    paths = [e["path"] for e in manifest["mount"]["experiments"]]
    assert len(set(paths)) == 2, paths
    assert len(payloads) == 2, "each experiment keeps its own contents"


# ---- import -------------------------------------------------------------

@pytest.fixture
def exported(sandbox, workspace):
    lib.create_snapshot(workspace, {"app": {"tab": 6}}, name="Class 3")
    dest = sandbox / "eval.zip"
    lib.export_workspace(workspace, dest)
    return dest


def test_import_resolves_the_paths_against_what_it_extracted(sandbox, exported):
    elsewhere = sandbox / "colleague"
    imported, notes = lib.import_workspace(exported, root=elsewhere)

    assert imported is not None, notes
    dataset = imported.mount.dataset_path
    assert dataset.startswith(str(elsewhere)), dataset
    assert lib.Path(dataset).is_file()
    assert lib.Path(imported.mount.experiments[0].path).is_file()


def test_import_brings_the_saved_views(sandbox, exported):
    imported, _ = lib.import_workspace(exported, root=sandbox / "colleague")
    snapshots = lib.list_snapshots(imported)

    assert [s.name for s in snapshots] == ["Class 3"]
    assert snapshots[0].view_state == {"app": {"tab": 6}}


def test_import_gives_the_workspace_a_fresh_id(sandbox, exported, workspace):
    """The same workspace imported twice is two workspaces; reusing the sender's
    id would make one of them unreachable."""
    first, _ = lib.import_workspace(exported, root=sandbox / "colleague")
    second, _ = lib.import_workspace(exported, root=sandbox / "colleague")

    assert first.id != second.id != workspace.id
    assert len(lib.list_workspaces(sandbox / "colleague")) == 2


def test_import_leaves_the_image_root_for_the_user(sandbox, exported):
    imported, _ = lib.import_workspace(exported, root=sandbox / "colleague")
    assert imported.mount.image_root == ""

    assert lib.rebase_image_root(imported, sandbox / "my_crops")
    reloaded = lib.load_workspace(imported.path)
    assert reloaded.mount.image_root == str(sandbox / "my_crops")


def test_importing_something_that_is_not_a_workspace(sandbox):
    junk = sandbox / "junk.zip"
    with zipfile.ZipFile(junk, "w") as archive:
        archive.writestr("hello.txt", "not a workspace")

    imported, notes = lib.import_workspace(junk, root=sandbox / "colleague")
    assert imported is None
    assert any("not a Lookin workspace" in n for n in notes)


def test_importing_a_file_that_is_not_a_zip(sandbox):
    plain = sandbox / "notes.zip"
    plain.write_text("just text", encoding="utf-8")

    imported, notes = lib.import_workspace(plain, root=sandbox / "colleague")
    assert imported is None and notes


def test_an_archive_cannot_write_outside_the_workspace(sandbox, exported, capsys):
    """A workspace arrives from another machine. An entry climbing out with
    ".." would otherwise write wherever it liked."""
    tampered = sandbox / "evil.zip"
    with zipfile.ZipFile(exported) as source, zipfile.ZipFile(tampered, "w") as dest:
        for name in source.namelist():
            dest.writestr(name, source.read(name))
        dest.writestr("../../pwned.txt", "should never be written")

    elsewhere = sandbox / "colleague"
    lib.import_workspace(tampered, root=elsewhere)

    assert not (sandbox / "pwned.txt").exists()
    assert not (elsewhere.parent / "pwned.txt").exists()
    assert "refusing archive entry" in capsys.readouterr().out


def test_a_workspace_whose_dataset_is_missing_says_so(sandbox, workspace):
    (sandbox / "data" / "ships.csv").unlink()
    dest = sandbox / "eval.zip"
    lib.export_workspace(workspace, dest)

    imported, notes = lib.import_workspace(dest, root=sandbox / "colleague")

    assert imported is not None
    assert any("cannot be opened" in n for n in notes)
