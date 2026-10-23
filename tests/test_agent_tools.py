"""What the MCP endpoint answers, checked on a two-experiment workspace.

The functions in core/agent_tools.py are tested directly; one test at the
end goes through the real ASGI app over HTTP, the way OpenCode reaches it,
to check the wiring: the route exists next to the UI, the session manager
starts, and a tool call comes back as JSON.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from lookin.core import agent_tools as tools
from lookin.core import library as lib
from lookin.core import mlflow_client as mlf
from lookin.core.library import Mount, MountExperiment


@pytest.fixture
def library_root(tmp_path, monkeypatch):
    root = tmp_path / "library"
    lib.set_library_root(root)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    tools.clear_cache()
    yield root
    lib.set_library_root(None)
    tools.clear_cache()


@pytest.fixture
def workspace(library_root, tmp_path, dataset_csv, results_csv):
    # "better" fixes crop f (bird predicted dog) and breaks crop a (cat predicted dog).
    better = pd.read_csv(results_csv)
    better.loc[better["uuid"] == "f", ["dog", "bird"]] = [0.1, 0.8]
    better.loc[better["uuid"] == "a", ["cat", "dog"]] = [0.1, 0.8]
    better_csv = tmp_path / "better_results.csv"
    better.to_csv(better_csv, index=False)
    mount = Mount(dataset_path=str(dataset_csv), split_kind="test", experiments=[
        MountExperiment(name="baseline", path=str(results_csv)),
        MountExperiment(name="better", path=str(better_csv)),
        MountExperiment(name="gone", path=str(tmp_path / "missing.csv")),
    ])
    return lib.create_workspace("Eval on tiny", mount, tags=["eval"], root=library_root)


def test_list_workspaces_names_their_experiments(workspace):
    out = tools.list_workspaces()
    assert out["count"] == 1
    listed = out["workspaces"][0]
    assert listed["id"] == workspace.id
    assert listed["experiments"] == ["baseline", "better", "gone"]
    json.dumps(out)


def test_a_workspace_is_found_by_name_as_well_as_id(workspace):
    assert tools.get_workspace("Eval on tiny")["id"] == workspace.id
    with pytest.raises(ValueError, match="No workspace"):
        tools.get_workspace("nope")


def test_get_metrics_reports_each_experiment_and_says_what_it_skipped(workspace):
    out = tools.get_metrics(workspace.id)
    by_name = {row["experiment"]: row for row in out["experiments"]}
    assert set(by_name) == {"baseline", "better"}
    assert by_name["baseline"]["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert by_name["baseline"]["n_samples"] == 6
    assert out["not_loaded"] == ["gone"]
    assert any("'gone'" in w for w in out["warnings"])
    json.dumps(out)  # NaN would make this invalid JSON for a strict client


def test_asking_for_a_skipped_experiment_explains_why(workspace):
    with pytest.raises(ValueError, match="not loaded.*not on this machine"):
        tools.get_per_class(workspace.id, "gone")


def test_per_class_puts_the_worst_class_first(workspace):
    out = tools.get_per_class(workspace.id, "baseline", top=1)
    assert out["n_classes"] == 3
    assert len(out["classes"]) == 1
    # baseline misses one of the two birds and calls it a dog.
    assert out["classes"][0]["label"] in {"bird", "dog"}
    with pytest.raises(ValueError, match="sort_by"):
        tools.get_per_class(workspace.id, "baseline", sort_by="nonsense")


def test_confusions_list_only_off_diagonal_cells(workspace):
    out = tools.get_confusions(workspace.id, "baseline")
    assert out["n_errors"] == 1
    assert out["confusions"] == [
        {"true_label": "bird", "predicted_label": "dog", "count": 1, "share_of_true_class": 0.5}]


def test_compare_counts_fixes_and_breaks(workspace):
    out = tools.compare_experiments(workspace.id, "baseline", "better")
    assert out["fixed"] == 1 and out["broken"] == 1
    assert out["net_change"] == 0
    assert out["n_common"] == 6
    assert {row["label"] for row in out["per_class_change"]} == {"bird", "cat"}


def test_examples_filter_and_put_confident_errors_first(workspace):
    out = tools.get_examples(workspace.id, "baseline")
    assert out["n_matching"] == 1
    example = out["examples"][0]
    assert (example["uuid"], example["label"], example["predicted_label"]) == ("f", "bird", "dog")
    assert "site" in example["metadata"]

    every = tools.get_examples(workspace.id, "baseline", true_label="cat", errors_only=False)
    assert [e["uuid"] for e in every["examples"]] == ["a", "b"]


def test_a_changed_workspace_is_reloaded(workspace, tmp_path, results_csv):
    assert {r["experiment"] for r in tools.get_metrics(workspace.id)["experiments"]} == {"baseline", "better"}

    def drop_better(fresh):
        fresh.mount.experiments = [e for e in fresh.mount.experiments if e.name != "better"]
        return True
    lib.update_workspace(workspace.path, drop_better)
    assert {r["experiment"] for r in tools.get_metrics(workspace.id)["experiments"]} == {"baseline"}


def test_search_mlflow_runs_uses_the_home_page_servers(workspace, monkeypatch):
    from lookin.core.home_config import HomeConfig

    config = HomeConfig(fallback_uri="http://mlflow.lan:5000")
    config.set_experiments("http://mlflow.lan:5000", shown=["arcface"], hidden=[])
    asked = {}

    def fake_search(uri, experiment_ids, filter_string="", max_pages=10):
        asked.update(uri=uri, ids=experiment_ids, filter=filter_string)
        return [mlf.Run(run_id="r1", run_name="vit-b", status="FINISHED", start_time=1_700_000_000_000,
                        params={"backbone": "vit"}, metrics={"val_acc": 0.93}, experiment_id="7")], False

    monkeypatch.setattr(mlf, "list_experiments",
                        lambda uri: [mlf.Experiment("7", "arcface"), mlf.Experiment("8", "other")])
    monkeypatch.setattr(mlf, "search_runs", fake_search)

    out = tools.search_mlflow_runs(filter="metrics.val_acc > 0.9")
    assert asked == {"uri": "http://mlflow.lan:5000", "ids": ["7"], "filter": "metrics.val_acc > 0.9"}
    assert out["runs"][0]["experiment"] == "arcface"
    assert out["runs"][0]["metrics"] == {"val_acc": 0.93}
    assert out["runs"][0]["in_workspaces"] == []


def test_search_mlflow_runs_without_a_server_says_so(library_root):
    with pytest.raises(ValueError, match="No MLflow server"):
        tools.search_mlflow_runs()


# ---- over HTTP --------------------------------------------------------------

def _rpc(client, method, params=None, id_=1):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}},
                           headers={"Accept": "application/json, text/event-stream",
                                    "MCP-Protocol-Version": "2025-11-25"})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_endpoint_serves_tools_next_to_the_ui(workspace, tmp_path):
    from starlette.testclient import TestClient

    from lookin.mcp_server import TOOLS
    from lookin.web.server import WebApp, create_app

    app = create_app(WebApp(), enable_mcp=True)
    with TestClient(app, base_url="http://lookin-server.lan:8550") as client:
        init = _rpc(client, "initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "lookin"

        listed = _rpc(client, "tools/list", id_=2)["result"]["tools"]
        assert {t["name"] for t in listed} == {t.__name__ for t in TOOLS}
        hints = {t["name"]: t["annotations"]["readOnlyHint"] for t in listed}
        assert hints["get_metrics"] and not hints["create_workspace"]

        called = _rpc(client, "tools/call", {"name": "get_metrics", "arguments": {"workspace_id": workspace.id}},
                      id_=3)["result"]
        assert not called.get("isError"), called
        assert "structuredContent" not in called  # the answer is sent once, as text
        payload = json.loads(called["content"][0]["text"])
        assert {r["experiment"] for r in payload["experiments"]} == {"baseline", "better"}

        # A mistake comes back with its reason, so the model can fix the call
        # -- not as the SDK's bare "Error executing tool get_metrics".
        failed = _rpc(client, "tools/call", {"name": "get_metrics", "arguments": {"workspace_id": "nope"}},
                      id_=4)["result"]
        assert failed["isError"]
        assert "No workspace with id or name 'nope'" in failed["content"][0]["text"]

        # The UI is still what / serves.
        assert client.get("/").status_code == 200
