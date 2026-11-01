"""The link with TrainHub, through the MLflow run id alone: which workspaces
use a run (GET /api/links/mlflow-runs), ?mlflow_run= putting a run in a
workspace, and TrainHub's own index drawn as badges."""
from __future__ import annotations

import threading
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from lookin.core import library as lib
from lookin.core import links
from lookin.core import mlflow_client as mlf
from lookin.core import run_links
from lookin.core import trainhub_links as th
from lookin.core import workspace_sources as wss
from lookin.core.home_config import HomeConfig, normalize
from lookin.web import render
from lookin.web.server import WebApp, create_app
from lookin.web.session import RequestScope

URI = "http://mlflow.lab:5000"
PARAM = wss.DEFAULT_DATASET_PARAM


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv(th.URL_ENV, raising=False)
    lib.set_library_root(tmp_path / "library")
    links.set_base_url("http://lookin.lab:8560")
    th.set_url(None)
    yield
    lib.set_library_root(None)
    links.set_base_url("")
    th.set_url(None)


def _run(run_id="r1", name="vit-aug", dataset="/data/ships.csv"):
    return mlf.Run(run_id=run_id, run_name=name, status="FINISHED", start_time=1,
                   params={PARAM: dataset} if dataset else {}, experiment_id="7")


def _workspace_with(run_id="r1", name="ws"):
    ws = lib.create_workspace(name, lib.Mount(), hash_dataset=False)
    candidates = wss.candidates_for_run(URI, _run(run_id), [mlf.ArtifactFile("eval/test_results.csv", 5)], [])
    return lib.update_workspace(ws.path, lambda fresh: bool(wss.add_candidates(fresh, candidates).added))


# ---- the link format ---------------------------------------------------------

def test_run_link_and_relative():
    url = links.run_link("abc", URI, "eval/x.csv")
    parts = urlsplit(url)
    assert parts.netloc == "lookin.lab:8560"
    assert parse_qs(parts.query) == {"mlflow_run": ["abc"], "mlflow_server": [URI], "artifact": ["eval/x.csv"]}
    assert links.relative(url).startswith("/?mlflow_run=abc")
    assert links.relative(links.link("lkw-1")) == "/?workspace=lkw-1"


# ---- which workspaces use a run ----------------------------------------------

def test_the_index_lists_the_workspaces_of_each_run():
    a = _workspace_with("r1", "alpha")
    b = _workspace_with("r1", "beta")
    _workspace_with("r2", "gamma")
    lib.create_workspace("local only", lib.Mount(), hash_dataset=False)
    payload = run_links.index_payload()
    assert payload["app"] == "lookin" and payload["version"]
    assert set(payload["runs"]) == {"r1", "r2"}
    r1 = payload["runs"]["r1"]
    assert [e["workspace"] for e in r1] == ["alpha", "beta"]
    assert {e["workspace_id"] for e in r1} == {a.id, b.id}
    assert r1[0]["path"] == f"/?workspace={a.id}"
    assert r1[0]["url"] == f"http://lookin.lab:8560/?workspace={a.id}"
    assert r1[0]["experiment"] == "vit-aug" and r1[0]["artifact"] == "eval/test_results.csv"
    assert set(run_links.index_payload(["r2", "nope"])["runs"]) == {"r2"}


def test_the_index_route_is_a_plain_get():
    _workspace_with("r1")
    client = TestClient(create_app(WebApp()))
    resp = client.get("/api/links/mlflow-runs", params=[("run_id", "r1"), ("run_id", "r9")])
    assert resp.status_code == 200
    assert list(resp.json()["runs"]) == ["r1"]
    assert "access-control-allow-origin" not in resp.headers       # never readable cross-site
    assert client.get("/api/links/mlflow-runs").json()["app"] == "lookin"


# ---- finding a run from its id ------------------------------------------------

def test_resolve_run_tries_the_hinted_server_first_then_the_others(monkeypatch):
    asked = []

    def get_run(uri, run_id):
        asked.append(uri)
        if uri != "http://b:5000":
            raise mlf.MlflowError("404")
        return _run(run_id)

    monkeypatch.setattr(mlf, "get_run", get_run)
    uri, run = run_links.resolve_run("r1", "http://b:5000/", servers=["http://a:5000", "http://b:5000"])
    assert (uri, run.run_id, asked) == ("http://b:5000", "r1", ["http://b:5000"])
    asked.clear()
    # A hint Lookin is not configured for is never dialled; the configured ones are.
    with pytest.raises(run_links.RunNotFound) as err:
        run_links.resolve_run("r1", "http://elsewhere:5000", servers=["http://a:5000"])
    assert asked == ["http://a:5000"]
    assert err.value.unknown_server == "http://elsewhere:5000"
    with pytest.raises(run_links.RunNotFound, match="No MLflow server"):
        run_links.resolve_run("r1", servers=[])


# ---- ?mlflow_run= -----------------------------------------------------------

def _wait(pred, timeout=10.0):
    deadline = time.time() + timeout
    while not pred():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def test_mlflow_run_link_puts_the_run_in_a_new_workspace_and_opens_it(monkeypatch, tmp_path):
    from lookin.web.views.run_link import TARGET_EXISTING, RunLinkDialog

    monkeypatch.setattr(run_links, "configured_servers", lambda: [URI])
    monkeypatch.setattr(mlf, "get_run", lambda uri, run_id: _run(run_id, dataset=str(tmp_path / "d.csv")))
    monkeypatch.setattr(mlf, "list_csv_and_npz_artifacts", lambda uri, run_id: (
        [mlf.ArtifactFile("eval/val_results.csv", 5), mlf.ArtifactFile("eval/test_results.csv", 5)], []))
    downloaded = []

    def fake_download(uri, run_id, artifact_path, dest, progress=None, total_hint=None):
        downloaded.append(artifact_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"12345")

    monkeypatch.setattr(mlf, "download_artifact", fake_download)
    already = _workspace_with("r1", "already here")
    s = WebApp().store.get("l" * 24, create=True)
    cmds = []
    s.cmd = lambda **c: cmds.append(c)
    query = {"mlflow_run": ["r1"], "mlflow_server": [URI], "artifact": ["eval/test_results.csv"]}
    with RequestScope(s):
        render.boot(s, query)
    assert s.mode == "home"
    [dialog] = s.dialogs
    assert isinstance(dialog, RunLinkDialog)
    _wait(lambda: not dialog.loading)
    with RequestScope(s):
        body = dialog.body()
    assert not dialog.error, dialog.error
    assert "already here" in body and f"/?workspace={already.id}" in body
    assert dialog.candidates[dialog.choice].csv.path == "eval/test_results.csv"   # the one asked for
    assert dialog.name == "vit-aug/test_results"
    # A reload of the same address does not open it again.
    with RequestScope(s):
        dialog.on_close()
        render.boot(s, query)
    assert s.dialogs == []
    s.open_dialog(dialog)
    with RequestScope(s):
        dialog.on_name(value="From TrainHub")
        dialog.on_go()
    assert s.dialogs == []
    _wait(lambda: any("navigate" in c for c in cmds))
    [created] = [w for w in lib.list_workspaces() if w.name == "From TrainHub"]
    assert [e.mlflow_run_id for e in created.mount.experiments] == ["r1"]
    assert created.mount.dataset_path == str(tmp_path / "d.csv")
    assert downloaded == ["eval/test_results.csv"]
    assert {"navigate": f"/?workspace={created.id}"} in cmds
    assert TARGET_EXISTING  # the other target is a key, never reworded


def test_mlflow_run_link_says_when_no_server_knows_the_run(monkeypatch):
    def nowhere(uri, run_id):
        raise mlf.MlflowError("RESOURCE_DOES_NOT_EXIST")

    monkeypatch.setattr(run_links, "configured_servers", lambda: [URI])
    monkeypatch.setattr(mlf, "get_run", nowhere)
    s = WebApp().store.get("n" * 24, create=True)
    with RequestScope(s):
        render.boot(s, {"mlflow_run": ["zz"], "mlflow_server": ["http://other:5000"]})
    [dialog] = s.dialogs
    _wait(lambda: not dialog.loading)
    with RequestScope(s):
        body, footer = dialog.body(), dialog.footer()
    assert "zz" in dialog.error and "http://other:5000" in dialog.error
    assert "add_server" in body and "go" not in footer


# ---- TrainHub's index -------------------------------------------------------

TRAINHUB_ANSWER = {
    "app": "trainhub",
    "version": "0.19.0",
    "runs": {
        "r1": [
            {"kind": "inference", "label": "Lab · run_a/last.ckpt", "detail": "Inference made with ...",
             "path": "/checkpoints/lab?run=run_a&inference=r1"},
            {"kind": "run", "label": "evil", "path": "javascript:alert(1)"},
            {"kind": "run", "label": "", "path": "/p/x"},
        ],
        "r2": "not a list",
    },
}


def _transport(status=200, payload=None, seen=None):
    def handle(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=TRAINHUB_ANSWER if payload is None else payload)
    return httpx.MockTransport(handle)


def test_trainhub_index_is_parsed_and_only_its_own_pages_kept():
    seen = []
    runs = th.fetch_index("http://trainhub.lab:8420", "tok", transport=_transport(seen=seen))
    assert list(runs) == ["r1"]
    [link] = runs["r1"]
    assert link.url == "http://trainhub.lab:8420/checkpoints/lab?run=run_a&inference=r1"
    assert link.kind == th.KIND_INFERENCE and "Checkpoints" in link.title()
    assert seen[0].url.path == th.INDEX_PATH
    assert seen[0].headers["authorization"] == "Bearer tok"


@pytest.mark.parametrize(("status", "payload", "message"), [
    (401, {}, th.TOKEN_ENV),
    (404, {}, "older than 0.19"),
    (200, {"app": "lookin"}, "does not answer like a TrainHub"),
])
def test_trainhub_index_errors_say_what_to_do(status, payload, message):
    with pytest.raises(ValueError, match=message):
        th.fetch_index("http://trainhub.lab:8420", transport=_transport(status, payload))


def test_the_trainhub_address_comes_from_the_flag_the_variable_or_the_file(monkeypatch):
    assert th.configured_url() == ""
    HomeConfig().set_trainhub_url("http://from-file:8420/")
    assert th.configured_url() == "http://from-file:8420" and th.url_source() == "home_view.json"
    monkeypatch.setenv(th.URL_ENV, "http://from-env:8420")
    assert th.configured_url() == "http://from-env:8420" and th.url_source() == th.URL_ENV
    th.set_url("http://from-flag:8420")
    assert th.configured_url() == "http://from-flag:8420" and th.url_source() == "--trainhub-url"
    assert normalize({"trainhub_url": "javascript:x"})["trainhub_url"] == ""


def test_the_index_is_read_behind_and_kept_when_trainhub_goes_away(monkeypatch):
    th.set_url("http://trainhub.lab:8420")
    answers = [{"r1": [th.TrainhubLink("run", "proj · r00001", "http://trainhub.lab:8420/p/proj?run=r00001")]}]
    gate = threading.Event()

    def fetch(url, token):
        gate.wait(5)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    index = th.TrainhubIndex(fetch=fetch)
    moved = []
    index.subscribe("t", lambda: moved.append(1))
    assert index.links("r1") == []                 # never waits: read behind
    gate.set()
    _wait(lambda: index.fetched_at)
    assert [link.label for link in index.links("r1")] == ["proj · r00001"] and moved
    answers.append(httpx.ConnectError("refused"))
    index.refresh(wait=5)
    assert "refused" in index.error
    assert [link.label for link in index.links("r1")] == ["proj · r00001"]   # last answer kept
    assert index.status()[0] == "http://trainhub.lab:8420"


def test_badges_link_to_trainhub_and_nothing_shows_without_it(monkeypatch):
    from lookin.web.views import trainhub as view

    index = th.TrainhubIndex(fetch=lambda url, token: {})
    monkeypatch.setattr(th.TrainhubIndex, "_instance", index)
    index.runs = {"r1": [th.TrainhubLink("inference", "Lab · run_a/last.ckpt",
                                         "http://trainhub.lab:8420/checkpoints/lab?run=run_a&inference=r1")],
                  "r2": [th.TrainhubLink("run", "bad", "javascript:alert(1)")]}
    index._tried_at = time.monotonic()            # no read in the test
    html = view.badges("r1")
    assert 'href="http://trainhub.lab:8420/checkpoints/lab?run=run_a&amp;inference=r1"' in html
    assert "TrainHub · Lab · run_a/last.ckpt" in html
    assert view.badges("r2") == "" and view.badges("") == "" and view.badges("r3") == ""
    assert "+ TrainHub" in view.status_chip(' data-act="x"')


def test_the_mcp_tool_answers_workspaces_and_trainhub(monkeypatch):
    from lookin.core import agent_tools

    _workspace_with("r1", "alpha")
    index = th.TrainhubIndex(fetch=lambda url, token: {})
    monkeypatch.setattr(th.TrainhubIndex, "_instance", index)
    index.runs = {"r1": [th.TrainhubLink("run", "proj · r00001", "http://t/p/proj?run=r00001")]}
    index.fetched_at = index._tried_at = time.monotonic()
    out = agent_tools.get_mlflow_run_links(["r1", "r9"])["runs"]
    assert [w["workspace"] for w in out["r1"]["workspaces"]] == ["alpha"]
    assert out["r1"]["trainhub"][0]["url"] == "http://t/p/proj?run=r00001"
    assert out["r9"]["workspaces"] == [] and out["r9"]["new_workspace_url"].endswith("/?mlflow_run=r9")
    with pytest.raises(ValueError):
        agent_tools.get_mlflow_run_links([])
