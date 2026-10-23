"""The MCP endpoint beyond reading: building workspaces, saving views,
recording review decisions, and having charts, mosaics and reports made.

Each tool is called as a function, on a library in tmp_path, then checked
the way the interface would see it -- the manifest on disk, the review
sidecar, the link that opens it. A few tests go over HTTP to check what
only the wiring decides: the read-only switch, the picture content, the
exports route and the cross-site guard.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from lookin import mcp_server
from lookin.core import agent_actions as actions
from lookin.core import agent_outputs as outputs
from lookin.core import agent_tools as tools
from lookin.core import board as bd
from lookin.core import library as lib
from lookin.core import links
from lookin.core.review import ReviewStore, review_paths_for_csv

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"


@pytest.fixture
def library_root(tmp_path, monkeypatch):
    root = tmp_path / "library"
    lib.set_library_root(root)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    links.set_base_url("http://lookin.lan:8560")
    tools.clear_cache()
    yield root
    lib.set_library_root(None)
    links.set_base_url("")
    tools.clear_cache()


@pytest.fixture
def example(tmp_path):
    dest = tmp_path / "ex"
    shutil.copytree(EXAMPLE, dest)
    return dest


@pytest.fixture
def built(library_root, example):
    """A workspace made the way an agent makes one: a dataset, then two
    local results CSVs."""
    ws = actions.create_workspace("Example", dataset_path=str(example / "dataset.csv"), group="Evals")
    actions.add_experiment(ws["id"], str(example / "baseline_cnn_results.csv"))
    actions.add_experiment(ws["id"], str(example / "finetuned_resnet_results.csv"), name="resnet")
    return ws["id"]


# ---- building a workspace ---------------------------------------------------

def test_a_created_workspace_is_one_the_interface_opens(built, example):
    workspace = lib.load_workspace(next(w.path for w in lib.list_workspaces() if w.id == built))
    assert workspace.mount.dataset_path == str(example / "dataset.csv")
    assert (workspace.mount.uuid_col, workspace.mount.crop_col, workspace.mount.label_col) == ("uuid", "crop", "label")
    names = [e.name for e in workspace.mount.experiments]
    assert names == ["baseline_cnn", "resnet"]
    # The sidecars beside each CSV come with it, found by name.
    assert workspace.mount.experiments[0].embeddings_path.endswith("baseline_cnn_embeddings.npz")
    # In the group asked for, on Home.
    assert bd.column_of(bd.load(), built)["name"] == "Evals"
    metrics = tools.get_metrics(built)
    assert {r["experiment"] for r in metrics["experiments"]} == {"baseline_cnn", "resnet"}


def test_answers_carry_the_link_that_opens_them(built):
    assert tools.get_metrics(built)["open_url"] == f"http://lookin.lan:8560/?workspace={built}&tab=metrics"
    example = tools.get_examples(built, "baseline_cnn", limit=1)["examples"][0]
    assert example["open_url"] == (f"http://lookin.lan:8560/?workspace={built}&crop={example['uuid']}"
                                   f"&experiment=baseline_cnn")


def test_a_wrong_column_is_refused_not_saved(library_root, example):
    with pytest.raises(ValueError, match="no column 'nope'"):
        actions.create_workspace("Bad", dataset_path=str(example / "dataset.csv"), label_col="nope")
    assert lib.list_workspaces() == []


def test_a_dataset_that_is_not_there_is_refused(library_root, tmp_path):
    with pytest.raises(ValueError, match="No dataset CSV"):
        actions.create_workspace("Bad", dataset_path=str(tmp_path / "missing.csv"))


def test_update_changes_only_what_is_given(built):
    actions.update_workspace(built, description="weekly eval", tags=["eval"], group="Archive")
    workspace = tools.get_workspace(built)
    assert workspace["description"] == "weekly eval" and workspace["tags"] == ["eval"]
    assert workspace["group"] == "Archive"
    assert [e["name"] for e in workspace["experiments"]] == ["baseline_cnn", "resnet"]


def test_experiments_are_renamed_recoloured_and_removed(built):
    actions.rename_experiment(built, "resnet", "resnet50")
    actions.set_experiment_color(built, "resnet50", "#123ABC")
    with pytest.raises(ValueError, match="#rrggbb"):
        actions.set_experiment_color(built, "resnet50", "blue")
    workspace = lib.load_workspace(next(w.path for w in lib.list_workspaces() if w.id == built))
    assert workspace.mount.experiments[1].name == "resnet50"
    assert workspace.mount.experiments[1].color == "#123abc"
    actions.remove_experiment(built, "resnet50")
    assert [e["name"] for e in tools.get_workspace(built)["experiments"]] == ["baseline_cnn"]


def test_deleting_asks_for_the_id_twice(built):
    with pytest.raises(ValueError, match="pass its id"):
        actions.delete_workspace(built, confirm="Example")
    assert any(w.id == built for w in lib.list_workspaces())
    actions.delete_workspace(built, confirm=built)
    assert not any(w.id == built for w in lib.list_workspaces())


def test_a_duplicate_sits_next_to_its_original(built):
    copy = actions.duplicate_workspace(built, "Example (copy)")
    ids = bd.column_of(bd.load(), built)["workspaces"]
    assert ids[ids.index(built) + 1] == copy["id"]


# ---- the dataset filter -------------------------------------------------------

def test_the_filter_is_checked_saved_and_then_used_by_every_number(built):
    out = actions.set_dataset_filter(built, query="label != 'cat'")
    assert (out["rows_kept"], out["rows_total"]) == (720, 800)
    metrics = tools.get_metrics(built)
    assert metrics["dataset_rows"] == 720
    assert any("720 of 800" in w for w in metrics["warnings"])
    actions.set_dataset_filter(built)
    assert tools.get_metrics(built)["dataset_rows"] == 800


def test_a_filter_that_keeps_nothing_or_does_not_parse_is_refused(built):
    with pytest.raises(ValueError, match="keeps no row"):
        actions.set_dataset_filter(built, query="label == 'no such class'")
    with pytest.raises(ValueError):
        actions.set_dataset_filter(built, query="__import__('os')")
    assert tools.get_workspace(built)["dataset"]["filter"] == []


# ---- snapshots ----------------------------------------------------------------

def test_a_snapshot_opens_on_the_tab_it_was_given(built):
    snap = actions.create_snapshot(built, "the metrics", tab="metrics")
    assert snap["open_url"] == f"http://lookin.lan:8560/?snapshot={snap['id']}"
    saved = tools.get_snapshot(snap["id"])
    assert saved["view_state"]["app"]["tab"] == 3
    actions.update_snapshot(snap["id"], name="renamed")
    assert tools.get_snapshot(snap["id"])["name"] == "renamed"
    with pytest.raises(ValueError, match="pass its id"):
        actions.delete_snapshot(snap["id"], confirm="renamed")
    actions.delete_snapshot(snap["id"], confirm=snap["id"])
    with pytest.raises(ValueError):
        tools.get_snapshot(snap["id"])


# ---- review --------------------------------------------------------------------

def test_decisions_land_in_the_review_store_and_never_in_the_dataset(built, example):
    before = (example / "dataset.csv").read_bytes()
    uuids = [e["uuid"] for e in tools.get_examples(built, "baseline_cnn", limit=3)["examples"]]
    out = actions.record_review_decisions(built, uuids, "exclude", reason="blurred")
    assert out["counts"]["exclude"] == 3
    store = ReviewStore.load(*review_paths_for_csv(example / "dataset.csv"))
    assert set(store.effective(out["version"])) >= set(uuids)
    assert (example / "dataset.csv").read_bytes() == before
    with pytest.raises(ValueError, match="new_label"):
        actions.record_review_decisions(built, uuids, "relabel")
    with pytest.raises(ValueError, match="not in this dataset"):
        actions.record_review_decisions(built, ["no-such-uuid"], "ok")


def test_a_reviewed_export_is_a_new_file_beside_the_dataset(built, example):
    uuids = [e["uuid"] for e in tools.get_examples(built, "baseline_cnn", limit=2)["examples"]]
    version = actions.record_review_decisions(built, uuids, "exclude")["version"]
    out = actions.export_reviewed_dataset(built)
    assert Path(out["path"]).parent == example
    assert out["rows"] == 800 and out["excluded"] == 2
    assert f"review_status_{version}" in out["columns"]


# ---- pictures and files ----------------------------------------------------------

@pytest.mark.parametrize("chart", outputs.CHART_KINDS)
def test_every_chart_kind_draws(built, chart):
    kwargs = {"column": "label"} if chart in ("column", "slices") else {}
    if chart in ("confusion_matrix", "confusion_overview", "reliability", "confidence_histogram", "projection"):
        kwargs["experiments"] = ["baseline_cnn"]
    answer, png = outputs.plot_chart(built, chart, **kwargs)
    assert png.startswith(b"\x89PNG")
    assert answer["chart"] == chart and answer["open_url"].startswith("http://lookin.lan:8560/")


def test_a_chart_needing_one_experiment_says_so(built):
    with pytest.raises(ValueError, match="one experiment"):
        outputs.plot_chart(built, "confusion_matrix")


def test_a_mosaic_frames_right_and_wrong_and_names_its_crops(built):
    answer, png = outputs.make_mosaic(built, "baseline_cnn", errors_only=True, limit=6, caption="prediction")
    assert png.startswith(b"\x89PNG")
    assert answer["n_crops"] == 6 and len(answer["uuids"]) == 6
    assert "unreadable" not in answer
    by_uuid, _ = outputs.make_mosaic(built, uuids=answer["uuids"][:2] + ["nope"])
    assert by_uuid["uuids"] == answer["uuids"][:2] and by_uuid["not_found"] == ["nope"]


def test_a_report_is_written_where_it_can_be_downloaded(built):
    out = outputs.generate_report(built, baseline="baseline_cnn", candidate="resnet")
    path = Path(out["path"])
    assert path.parent == outputs.exports_dir() and path.suffix == ".html"
    assert out["url"] == f"http://lookin.lan:8560/exports/{path.name}"
    assert "Compare: baseline_cnn to resnet" in path.read_text(encoding="utf-8")
    assert tools.get_snapshot(out["snapshot"]["id"])["workspace_id"] == built


def test_an_export_never_overwrites(built, tmp_path):
    taken = tmp_path / "taken.html"
    taken.write_text("mine")
    with pytest.raises(ValueError, match="never overwrites"):
        outputs.generate_report(built, dest=str(taken))
    assert taken.read_text() == "mine"


def test_the_exports_route_serves_bare_names_only(library_root):
    (outputs.exports_dir() / "r.html").write_text("ok")
    assert outputs.exported_file("r.html") is not None
    for name in ("../r.html", "sub/r.html", ".hidden", "missing.html"):
        assert outputs.exported_file(name) is None


# ---- over HTTP --------------------------------------------------------------------

def _rpc(client, method, params=None, id_=1, headers=None):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}},
                           headers={"Accept": "application/json, text/event-stream",
                                    "MCP-Protocol-Version": "2025-11-25", **(headers or {})})
    return response


def _init(client):
    _rpc(client, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                "clientInfo": {"name": "test", "version": "0"}})


def test_a_picture_comes_back_as_an_image_beside_its_json(built):
    from starlette.testclient import TestClient

    from lookin.web.server import WebApp, create_app

    with TestClient(create_app(WebApp(), enable_mcp=True)) as client:
        _init(client)
        result = _rpc(client, "tools/call", {"name": "make_mosaic", "arguments": {
            "workspace_id": built, "experiment": "baseline_cnn", "limit": 4}}, id_=2).json()["result"]
        kinds = [c["type"] for c in result["content"]]
        assert kinds == ["text", "image"]
        assert json.loads(result["content"][0]["text"])["n_crops"] == 4
        report = json.loads(_rpc(client, "tools/call", {"name": "generate_report", "arguments": {
            "workspace_id": built}}, id_=3).json()["result"]["content"][0]["text"])
        served = client.get(f"/exports/{Path(report['path']).name}")
        assert served.status_code == 200 and b"<html" in served.content[:200].lower()


def test_read_only_serves_no_tool_that_writes(built):
    from starlette.testclient import TestClient

    from lookin.web.server import WebApp, create_app

    with TestClient(create_app(WebApp(), enable_mcp=True, mcp_read_only=True)) as client:
        _init(client)
        names = {t["name"] for t in _rpc(client, "tools/list", id_=2).json()["result"]["tools"]}
        assert names == {t.__name__ for t in mcp_server.READ_TOOLS}
        assert "create_workspace" not in names and "generate_report" not in names
        refused = _rpc(client, "tools/call", {"name": "plot_chart", "arguments": {
            "workspace_id": built, "chart": "risk_coverage", "save_to": "/tmp"}}, id_=3).json()["result"]
        assert refused["isError"] and "read-only" in refused["content"][0]["text"]
    mcp_server.build_server()  # back to the default for the other tests


def test_a_page_on_another_site_cannot_drive_the_endpoint(built):
    from starlette.testclient import TestClient

    from lookin.web.server import WebApp, create_app

    with TestClient(create_app(WebApp(), enable_mcp=True)) as client:
        response = _rpc(client, "tools/list", headers={"Sec-Fetch-Site": "cross-site"})
        assert response.status_code == 403


# ---- the links themselves ------------------------------------------------------------

def test_a_link_opens_the_workspace_on_its_tab_with_the_crop_shown(built):
    from lookin.web import render
    from lookin.web.server import WebApp
    from lookin.web.session import RequestScope

    crop = tools.get_examples(built, "baseline_cnn", limit=1)["examples"][0]["uuid"]
    session = WebApp().store.get("l" * 24, create=True)
    query = {"workspace": [built], "tab": ["calibration"], "crop": [crop], "experiment": ["baseline_cnn"]}
    with RequestScope(session):
        render.boot(session, query)
    for _ in range(600):   # the workspace opens on a worker thread
        if session.current_workspace is not None and getattr(session, "_opening", None) is None:
            break
        time.sleep(0.05)
    assert session.tab == "calibration"
    assert crop in session.dialogs[-1].title
