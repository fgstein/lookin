"""MLflow failure paths: what happens when the server, the run or the cache
is not what the code expected.

The server is an httpx.MockTransport behind mlflow_client._client, so the
real request, streaming and error handling run -- only the network is gone.
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

from lookin.core import library as lib
from lookin.core import mlflow_cache
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss

URI = "http://mlflow.lab:5000"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    lib.set_library_root(tmp_path / "library")
    yield
    lib.set_library_root(None)


@pytest.fixture
def server(monkeypatch):
    """Route every client the module builds to `handler(request)`."""
    holder = {}

    def install(handler):
        holder["handler"] = handler

        def client(tracking_uri, timeout=mlf.TIMEOUT):
            return httpx.Client(base_url=tracking_uri.rstrip("/"), timeout=timeout,
                                transport=httpx.MockTransport(lambda r: holder["handler"](r)))

        monkeypatch.setattr(mlf, "_client", client)

    return install


def _cache_files():
    root = mlflow_cache.user_cache_dir()
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


# ---------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------

def test_an_artifact_replaced_since_it_was_listed_is_not_cached_as_it(server):
    """The workspace recorded 5 bytes; the run's file was re-logged since
    and is now 9. Taking it used to put a file in the cache that never
    matched the recorded size again: the workspace stayed "not downloaded"
    for good, and every press of the download button fetched it anew."""
    server(lambda r: httpx.Response(200, content=b"new,data\n"))
    with pytest.raises(mlf.MlflowError, match="changed"):
        mlflow_cache.download_cached(URI, "r1", "eval/results.csv", expected_size=5)
    assert _cache_files() == []


class _Broken(httpx.SyncByteStream):
    def __iter__(self):
        yield b"uuid,label\n"
        raise httpx.ReadError("connection reset")


def test_an_interrupted_download_leaves_no_part_file(server):
    server(lambda r: httpx.Response(200, stream=_Broken()))
    with pytest.raises(mlf.MlflowError):
        mlflow_cache.download_cached(URI, "r1", "eval/results.csv", expected_size=100)
    assert _cache_files() == []


@pytest.mark.parametrize("bad", ["../../outside.csv", "ABSOLUTE", "eval/../../outside.csv"])
def test_an_artifact_path_cannot_escape_the_cache(server, bad, tmp_path):
    """The path comes from the server's listing or from a workspace
    manifest someone else wrote (an imported workspace): neither may decide
    where on this machine a download is written."""
    if bad == "ABSOLUTE":
        bad = (tmp_path / "outside.csv").as_posix()
    server(lambda r: httpx.Response(200, content=b"x"))
    with pytest.raises(mlf.MlflowError):
        mlflow_cache.download_cached(URI, "r1", bad)
    assert not mlflow_cache.is_cached("r1", bad)
    assert not (tmp_path / "outside.csv").exists()
    assert not (tmp_path / "cache" / "outside.csv").exists()


def test_a_run_id_cannot_escape_the_cache(server):
    server(lambda r: httpx.Response(200, content=b"x"))
    with pytest.raises(mlf.MlflowError):
        mlflow_cache.download_cached(URI, "..", "outside.csv")
    with pytest.raises(mlf.MlflowError):
        mlflow_cache.download_cached(URI, "", "outside.csv")


def test_a_workspace_naming_an_unsafe_artifact_reports_it_and_writes_nothing(server, tmp_path):
    server(lambda r: httpx.Response(200, content=b"x"))
    ws = lib.create_workspace("ws", lib.Mount(), hash_dataset=False)
    ws.mount.experiments.append(lib.MountExperiment(
        name="evil", path=str(tmp_path / "nowhere.csv"), mlflow_uri=URI, mlflow_run_id="r1",
        artifact_path="../../../evil.csv", artifact_size=1))
    fetched, errors = wss.download(ws)
    assert fetched == 0 and len(errors) == 1
    assert not (tmp_path / "evil.csv").exists()
    assert wss.point_at_cache(ws) is False  # and does not raise


def test_the_artifact_walk_drops_paths_that_would_escape_the_cache(server):
    server(lambda r: httpx.Response(200, json={"files": [
        {"path": "../evil.csv", "is_dir": False, "file_size": 1},
        {"path": "ok.csv", "is_dir": False, "file_size": 1},
    ]}))
    assert [a.path for a in mlf.list_csv_artifacts(URI, "r1")] == ["ok.csv"]


# ---------------------------------------------------------------------------
# odd answers
# ---------------------------------------------------------------------------

def test_json_that_is_not_an_object_is_an_mlflow_error(server):
    """A proxy answering 200 with a JSON list or string used to escape as an
    AttributeError, which the browse dialog's workers do not catch."""
    server(lambda r: httpx.Response(200, json=["not", "mlflow"]))
    with pytest.raises(mlf.MlflowError):
        mlf.list_runs(URI, "7")
    with pytest.raises(mlf.MlflowError):
        mlf.list_experiments(URI)


def test_list_runs_reads_past_the_first_page(server):
    """The browse dialog's run list used to stop at the first 200 runs, with
    nothing saying so: older runs of a busy experiment were simply absent,
    and searching for one by name answered "No matches"."""
    pages = {None: (["a", "b"], "t1"), "t1": (["c"], None)}

    def handler(request):
        body = json.loads(request.content)
        ids, token = pages[body.get("page_token")]
        out = {"runs": [{"info": {"run_id": i, "experiment_id": "7"}} for i in ids]}
        if token:
            out["next_page_token"] = token
        return httpx.Response(200, json=out)

    server(handler)
    assert [r.run_id for r in mlf.list_runs(URI, "7")] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# the Home listing (web hub; the Flet Home page shares ServerState)
# ---------------------------------------------------------------------------

@pytest.fixture
def hub():
    from lookin.web.mlflow_hub import MlflowHub
    h = MlflowHub()
    assert URI in h.servers
    return h


def test_an_artifact_listing_read_from_disk_is_listed_again_before_use(hub, monkeypatch):
    """The run -> CSV lists are saved to disk and read back on the next
    start. They used to be trusted forever: a run expanded while it was
    still RUNNING (its CSV not logged yet) said "no CSV" for good, and a
    CSV re-logged since kept its old size -- which then made the workspace
    it was added to look undownloaded forever."""
    server = hub.servers[URI]
    mlflow_cache.save_artifacts(URI, {"r1": {"csvs": [], "npzs": []}})
    server.adopt_listing({"experiments": [], "runs": []}, mlflow_cache.load_artifacts(URI))
    calls = []

    def listing(uri, run_id):
        calls.append(run_id)
        return [mlf.ArtifactFile("eval/results.csv", 9)], []

    monkeypatch.setattr(mlf, "list_csv_and_npz_artifacts", listing)
    done = []
    hub.fetch_artifacts(server, "r1", then=lambda: done.append(1))
    for _ in range(200):
        if done:
            break
        time.sleep(0.01)
    assert calls == ["r1"]
    assert [a.path for a in server.artifacts["r1"]["csvs"]] == ["eval/results.csv"]
    # Listed in this process: trusted from now on.
    hub.fetch_artifacts(server, "r1")
    assert calls == ["r1"]


def test_a_refresh_lists_expanded_runs_again(hub, monkeypatch):
    """Refresh is the user saying "show me what is there now": a CSV logged
    since the run was expanded has to turn up, without a restart."""
    server = hub.servers[URI]
    server.artifacts["r1"] = {"csvs": [], "npzs": []}
    monkeypatch.setattr(hub, "_fetch_worker", lambda *a, **k: None)
    hub.refresh(URI)
    assert server.artifacts["r1"].get("stale")


def test_runs_fetched_under_a_filter_that_changed_meanwhile_are_dropped(hub, monkeypatch):
    """Two refreshes overlap when the filter is changed while a listing is
    still coming in. The older one finishing last used to replace the
    table with runs matching the *previous* filter, with nothing to tell."""
    server = hub.servers[URI]
    server.experiments = [mlf.Experiment("7", "cls-v3")]
    hub.config.set_experiments(URI, ["cls-v3"], [])
    hub.config.runs["filter"] = "params.a = '1'"

    def search(uri, ids, filter_string=""):
        hub.config.runs["filter"] = "params.a = '2'"   # the user changed it meanwhile
        return [mlf.Run("old", "old", "FINISHED", 1, experiment_id="7")], False

    monkeypatch.setattr(mlf, "search_runs", search)
    monkeypatch.setattr(mlflow_cache, "save_listing", lambda *a: None)
    hub._fetch_worker(server, runs_only=True)
    assert [r.run_id for r in server.runs] == []
