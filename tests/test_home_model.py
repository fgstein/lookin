"""What Home knows about a tracking server, and how runs and their CSVs are
named and selected -- without a browser or a tracking server.

Servers are filled in by hand, as the fetch worker would fill them.
"""
from __future__ import annotations

import pytest

from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.web import home_model as hv

URI = "http://mlflow.lab:5000"


@pytest.fixture
def server():
    server = hv.ServerState(uri=URI, name="lab", status="ok")
    server.experiments = [mlf.Experiment("7", "cls-v3"), mlf.Experiment("8", "sandbox")]
    server.runs_by_exp = {"7": [
        mlf.Run("r1", "vit-aug", "FINISHED", 3, experiment_id="7",
                params={wss.DEFAULT_DATASET_PARAM: "/data/ships.csv", "backbone": "vit"}),
        mlf.Run("r2", "vit-noaug", "FINISHED", 2, experiment_id="7", params={"backbone": "vit"}),
        mlf.Run("r3", "crashed", "FAILED", 1, experiment_id="7"),
    ]}
    server.rebuild_runs()
    server.artifacts["r1"] = {"csvs": [mlf.ArtifactFile("eval/test_results.csv", 4)], "npzs": []}
    server.artifacts["r2"] = {"csvs": [mlf.ArtifactFile("a/test_results.csv", 4),
                                       mlf.ArtifactFile("a/val_results.csv", 4)], "npzs": []}
    return server


def test_run_and_artifact_links_point_at_the_mlflow_ui():
    assert mlf.run_ui_url(URI + "/", "7", "r1") == f"{URI}/#/experiments/7/runs/r1"
    assert mlf.run_ui_url(URI, "7", "r1", "eval/x.csv") == f"{URI}/#/experiments/7/runs/r1/artifacts/eval/x.csv"


def test_keys_round_trip():
    assert hv.parse_key(hv.csv_key(URI + "/", "r1", "a b/c.csv")) == ("csv", URI, "r1", "a b/c.csv")
    assert hv.parse_key(hv.run_key(URI, "r1")) == ("run", URI, "r1", "")
    assert hv.parse_key("garbage") == ("", "", "", "")


def test_runs_are_grouped_by_experiment_whatever_order_they_came_in():
    server = hv.ServerState(uri=URI, name="lab", experiments=[
        mlf.Experiment("1", "zeta"), mlf.Experiment("2", "alpha")])
    server.runs_by_exp = {"1": [mlf.Run("z1", "", "FINISHED", 5, experiment_id="1")],
                          "2": [mlf.Run("a1", "", "FINISHED", 9, experiment_id="2"),
                                mlf.Run("a2", "", "FINISHED", 1, experiment_id="2")]}
    server.rebuild_runs()
    assert [r.run_id for r in server.runs] == ["a1", "a2", "z1"]


def test_the_cached_listing_round_trips_without_its_run_csv_lists():
    server = hv.ServerState(uri=URI, name="lab", experiments=[mlf.Experiment("7", "e")])
    server.runs_by_exp = {"7": [mlf.Run("r1", "n", "FINISHED", 1, experiment_id="7")]}
    server.truncated_exps = {"7"}
    server.rebuild_runs()
    server.artifacts["r1"] = {"csvs": [mlf.ArtifactFile("a.csv", 3)], "npzs": []}
    payload = server.listing_payload()
    assert "artifacts" not in payload
    again = hv.ServerState(uri=URI, name="lab")
    again.adopt_listing(payload, server.artifacts_payload())
    assert [r.run_id for r in again.runs] == ["r1"] and again.truncated
    assert again.artifacts["r1"]["csvs"][0].path == "a.csv"


def test_a_run_box_is_half_ticked_while_only_some_of_its_csvs_are(server):
    run = hv.run_key(URI, "r2")
    test, val = (hv.csv_key(URI, "r2", p) for p in ("a/test_results.csv", "a/val_results.csv"))
    selected = set()
    hv.set_selected(selected, server, test, True)
    assert hv.run_selection(selected, server, "r2") == hv.SELECTED_SOME
    hv.set_selected(selected, server, val, True)
    assert hv.run_selection(selected, server, "r2") == hv.SELECTED_ALL
    hv.set_selected(selected, server, run, False)
    assert not selected
    # Ticking the run ticks every CSV it is known to have...
    hv.set_selected(selected, server, run, True)
    assert selected == {test, val}
    hv.set_selected(selected, server, val, False)
    assert selected == {test} and hv.run_selection(selected, server, "r2") == hv.SELECTED_SOME


def test_a_run_whose_csvs_are_not_listed_is_selected_whole(server):
    selected = set()
    hv.set_selected(selected, server, hv.run_key(URI, "r3"), True)
    assert selected == {hv.run_key(URI, "r3")}
    assert hv.run_selection(selected, server, "r3") == hv.SELECTED_ALL
    # Listed later: unticking one CSV of the whole run keeps the others.
    server.artifacts["r3"] = {"csvs": [mlf.ArtifactFile("x.csv", 1), mlf.ArtifactFile("y.csv", 1)], "npzs": []}
    assert hv.csv_selected(selected, hv.csv_key(URI, "r3", "x.csv"))
    hv.set_selected(selected, server, hv.csv_key(URI, "r3", "x.csv"), False)
    assert selected == {hv.csv_key(URI, "r3", "y.csv")}


def test_a_workspace_keeps_its_badge_colour():
    assert hv.workspace_color("lkw-abc") == hv.workspace_color("lkw-abc")
    assert hv.workspace_color("lkw-abc").startswith("#")
