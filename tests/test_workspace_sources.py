"""Filling a workspace from MLflow runs, without a server.

What is pinned here is the contract the Home page relies on: dropping a CSV
records a reference and downloads nothing, a workspace stays one dataset,
the download state is honest about stale copies, and a manifest written
before experiments knew their run still loads.
"""
from __future__ import annotations

import json

import pytest

from lookin.core import library as lib
from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss

URI = "http://mlflow.lab:5000"
PARAM = wss.DEFAULT_DATASET_PARAM


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    lib.set_library_root(tmp_path / "library")
    yield
    lib.set_library_root(None)


def _run(run_id="r1", name="vit-aug", dataset="/data/ships.csv"):
    params = {PARAM: dataset} if dataset else {}
    return mlf.Run(run_id=run_id, run_name=name, status="FINISHED", start_time=1,
                   params=params, experiment_id="7")


def _csv(path="eval/test_results.csv", size=5):
    return mlf.ArtifactFile(path=path, size=size)


def _workspace(dataset=""):
    return lib.create_workspace("ws", lib.Mount(dataset_path=dataset), hash_dataset=False)


def test_a_run_with_one_csv_is_named_after_the_run():
    [c] = wss.candidates_for_run(URI, _run(), [_csv()], [])
    assert c.default_name() == "vit-aug"


def test_several_csvs_in_a_run_are_told_apart_by_file_name():
    names = [c.default_name() for c in wss.candidates_for_run(
        URI, _run(), [_csv("a/test_results.csv"), _csv("a/val_results.csv")], [])]
    assert names == ["vit-aug/test_results", "vit-aug/val_results"]


def test_sidecars_logged_next_to_the_csv_are_picked_up():
    npzs = [mlf.ArtifactFile("eval/test_embeddings.npz", 9), mlf.ArtifactFile("other.npz", 1)]
    [c] = wss.candidates_for_run(URI, _run(), [_csv()], npzs)
    assert c.embeddings.path == "eval/test_embeddings.npz"
    assert c.prototypes is None


def test_adding_records_the_run_and_downloads_nothing():
    ws = _workspace()
    result = wss.add_candidates(ws, wss.candidates_for_run(URI + "/", _run(), [_csv()], []))
    [e] = ws.mount.experiments
    assert result.added == ["vit-aug"]
    assert (e.mlflow_uri, e.mlflow_run_id, e.artifact_path) == (URI, "r1", "eval/test_results.csv")
    assert e.path == str(mlflow_cache.cached_path("r1", "eval/test_results.csv"))
    assert not wss.is_downloaded(e)
    assert wss.download_state(ws) == wss.DOWNLOAD_NONE


def test_an_empty_workspace_takes_its_dataset_from_the_first_run():
    ws = _workspace()
    result = wss.add_candidates(ws, wss.candidates_for_run(URI, _run(), [_csv()], []), PARAM)
    assert ws.mount.dataset_path == "/data/ships.csv"
    assert result.dataset_set == "/data/ships.csv"


def test_a_run_on_another_dataset_is_refused():
    ws = _workspace("/data/ships.csv")
    other = _run("r2", "other", dataset="/data/planes.csv")
    result = wss.add_candidates(ws, wss.candidates_for_run(URI, other, [_csv()], []), PARAM)
    assert not ws.mount.experiments
    assert "planes.csv" in result.skipped[0][1]


def test_a_run_that_records_no_dataset_is_let_in():
    ws = _workspace("/data/ships.csv")
    result = wss.add_candidates(ws, wss.candidates_for_run(URI, _run(dataset=""), [_csv()], []), PARAM)
    assert result.added


def test_the_same_csv_twice_is_skipped_and_names_stay_unique():
    ws = _workspace()
    first = wss.candidates_for_run(URI, _run(), [_csv()], [])
    wss.add_candidates(ws, first)
    again = wss.add_candidates(ws, first)
    assert again.skipped and not again.added
    wss.add_candidates(ws, wss.candidates_for_run(URI, _run("r9", "vit-aug"), [_csv()], []))
    # Same run name, another run: told apart by the run id, not a counter.
    assert [e.name for e in ws.mount.experiments] == ["vit-aug", "vit-aug · r9"]
    assert wss.unique_name({"a", "a · r9"}, "a", "r9") == "a · r9 (2)"
    assert wss.unique_name({"a"}, "a") == "a (2)"


def test_download_fetches_what_is_missing_and_the_state_follows(monkeypatch):
    ws = _workspace()
    wss.add_candidates(ws, wss.candidates_for_run(URI, _run(), [_csv(size=5)], []))
    calls = []

    def fake_download(uri, run_id, artifact_path, dest, progress=None, total_hint=None):
        calls.append((uri, run_id, artifact_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"12345")

    monkeypatch.setattr(mlf, "download_artifact", fake_download)
    fetched, errors = wss.download(ws)
    assert (fetched, errors) == (1, [])
    assert calls == [(URI, "r1", "eval/test_results.csv")]
    assert wss.download_state(ws) == wss.DOWNLOAD_ALL
    assert wss.download(ws) == (0, [])   # nothing left to fetch


def test_a_cached_copy_of_the_wrong_size_is_not_downloaded():
    ws = _workspace()
    wss.add_candidates(ws, wss.candidates_for_run(URI, _run(), [_csv(size=5)], []))
    stale = mlflow_cache.cached_path("r1", "eval/test_results.csv")
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"123")
    assert wss.download_state(ws) == wss.DOWNLOAD_NONE


def test_a_failed_file_does_not_stop_the_others(monkeypatch):
    ws = _workspace()
    wss.add_candidates(ws, wss.candidates_for_run(URI, _run(), [_csv("a.csv", 1), _csv("b.csv", 1)], []))

    def flaky(uri, run_id, artifact_path, dest, progress=None, total_hint=None):
        if artifact_path == "a.csv":
            raise mlf.MlflowError("boom")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"1")

    monkeypatch.setattr(mlf, "download_artifact", flaky)
    fetched, errors = wss.download(ws)
    assert fetched == 1 and "boom" in errors[0]
    assert wss.download_state(ws) == wss.DOWNLOAD_PARTIAL
    assert [e.name for e in wss.downloaded_only(ws.mount).experiments] == ["vit-aug/b"]


def test_a_workspace_of_local_files_is_not_a_download():
    ws = _workspace()
    ws.mount.experiments.append(lib.MountExperiment(name="local", path="/x.csv"))
    assert wss.download_state(ws) == wss.DOWNLOAD_LOCAL


def test_the_mlflow_link_survives_a_save_and_a_reload():
    ws = _workspace()
    wss.add_candidates(ws, wss.candidates_for_run(URI, _run(), [_csv()], []))
    lib.save_workspace(ws)
    [e] = lib.load_workspace(ws.path).mount.experiments
    assert (e.mlflow_run_id, e.artifact_path, e.artifact_size) == ("r1", "eval/test_results.csv", 5)


def test_a_manifest_from_before_mlflow_links_still_loads(tmp_path):
    raw = {"id": "lkd-abcdef", "name": "old", "mount": {
        "dataset_path": "/d.csv", "experiments": [{"name": "e", "path": "/e.csv"}]}}
    ws = lib.workspace_from_dict(json.loads(json.dumps(raw)))
    [e] = ws.mount.experiments
    assert e.path == "/e.csv" and not e.from_mlflow() and e.artifact_size is None
