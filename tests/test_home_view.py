"""The Home page, rendered without a browser or a tracking server.

Servers are filled in by hand, as the fetch worker would fill them, and the
threads are never started -- what is pinned is what the page does with a
listing: which runs it shows, what a drop adds, and whether the link
between a run and its workspaces is drawn.
"""
from __future__ import annotations

import pytest

from lookin.core import library as lib
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.core.data_manager import AppState
from lookin.ui import home_view as hv

URI = "http://mlflow.lab:5000"


class FakePage:
    width, height = 1440, 920

    def __init__(self):
        self.opened, self.launched, self.controls = [], [], {}

    def open(self, dialog):
        self.opened.append(dialog)

    def close(self, dialog):
        pass

    def launch_url(self, url, web_window_name=None):
        self.launched.append((url, web_window_name))

    def get_control(self, id_):
        return self.controls[id_]


class FakePanel:
    def __init__(self):
        self.requested = []

    def open_in_new_tab(self, ws, snapshot=None):
        self.requested.append((ws, snapshot))

    def open_panel(self, e=None, select=None):
        pass

    def follow_workspace(self, ws):
        self.followed = ws
        return True


class FakeCtx:
    def __init__(self):
        self.page = FakePage()
        self.state = AppState()
        self.current_workspace = None
        self.busy = None


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    lib.set_library_root(tmp_path / "library")
    view = hv.HomeView(FakeCtx(), FakePanel(), start_threads=False)
    server = view.servers[URI]
    server.status = "ok"
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
    view.config.toggle_experiment(URI, "cls-v3")
    view.build()
    yield view
    lib.set_library_root(None)


def _names(view):
    return [r.run_name for r in view._visible_runs(view.servers[URI])]


def test_failed_runs_are_hidden_unless_asked_for(home):
    assert _names(home) == ["vit-aug", "vit-noaug"]
    home.unsuccessful_box.value = True
    assert _names(home) == ["vit-aug", "vit-noaug", "crashed"]


def test_search_matches_names_and_parameters(home):
    home.search_field.value = "noaug"
    assert _names(home) == ["vit-noaug"]
    home.search_field.value = "backbone=vit aug"
    assert _names(home) == ["vit-aug", "vit-noaug"]


def test_a_hidden_run_stays_hidden_and_is_saved(home):
    server = home.servers[URI]
    home._hide_run(server, server.runs[0])
    assert _names(home) == ["vit-noaug"]
    assert hv.HomeView(FakeCtx(), FakePanel(), start_threads=False).config.hidden_run_ids(URI) == {"r1"}


def test_dropping_a_run_with_one_csv_adds_it(home):
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home.reload_workspaces()
    home.add_keys(ws, [hv.run_key(URI, "r1")])
    [e] = lib.load_workspace(ws.path).mount.experiments
    assert (e.mlflow_run_id, e.artifact_path) == ("r1", "eval/test_results.csv")
    assert lib.load_workspace(ws.path).mount.dataset_path == "/data/ships.csv"


def test_dropping_a_run_with_several_csvs_asks_which(home):
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home.add_keys(ws, [hv.run_key(URI, "r2")])
    dialog = home.ctx.page.opened[-1]
    boxes = [c for c in dialog.content.content.controls if getattr(c, "data", None) is not None]
    boxes[1].value = True
    dialog.actions[0].on_click(None)
    [e] = lib.load_workspace(ws.path).mount.experiments
    assert e.artifact_path == "a/val_results.csv"
    assert e.name == "vit-noaug/val_results"


def test_dragging_a_selected_row_carries_the_whole_selection(home):
    a, b = hv.run_key(URI, "r1"), hv.csv_key(URI, "r2", "a/test_results.csv")
    home.selected = {a, b}
    assert home._drag_keys(a) == sorted([a, b])
    assert home._drag_keys(hv.run_key(URI, "r3")) == [hv.run_key(URI, "r3")]


def test_a_drop_reads_the_key_off_the_dragged_control(home):
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home.reload_workspaces()

    class Dragged:
        data = hv.csv_key(URI, "r2", "a/test_results.csv")

    class Event:
        src_id = "_42"

    home.ctx.page.controls["_42"] = Dragged()
    home._on_drop(ws, Event())
    assert [e.artifact_path for e in lib.load_workspace(ws.path).mount.experiments] == ["a/test_results.csv"]


def test_runs_show_which_workspaces_hold_them(home):
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home.add_keys(ws, [hv.run_key(URI, "r1")])
    by_run, by_csv = home._membership()
    assert [w.id for w in by_run[(URI, "r1")]] == [ws.id]
    assert (URI, "r1", "eval/test_results.csv") in by_csv
    assert (URI, "r2") not in by_run


def test_a_workspace_without_a_dataset_still_opens(home):
    # The dataset is set up inside the workspace, on its Data tab.
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home._open_workspace(ws)
    [(opened, _)] = home.workspace_panel.requested
    assert opened.id == ws.id


def test_opening_with_undownloaded_runs_offers_to_download(home, tmp_path):
    dataset = tmp_path / "ships.csv"
    dataset.write_text("uuid,crop,label\n")
    ws = lib.create_workspace("A", lib.Mount(dataset_path=str(dataset)), hash_dataset=False)
    home.servers[URI].runs[0].params[wss.DEFAULT_DATASET_PARAM] = str(dataset)
    home.add_keys(ws, [hv.run_key(URI, "r1")])
    ws = lib.load_workspace(ws.path)
    home._open_workspace(ws)
    dialog = home.ctx.page.opened[-1]
    dialog.actions[1].on_click(None)          # "Open without them" -- the new tab leaves them out
    [(opened, _)] = home.workspace_panel.requested
    assert opened.id == ws.id


def test_mlflow_links_never_navigate_this_tab_away(home):
    home._open_url("http://mlflow.lab:5000/#/experiments/7")
    assert home.ctx.page.launched == [("http://mlflow.lab:5000/#/experiments/7", "_blank")]



def test_run_and_artifact_links_point_at_the_mlflow_ui():
    assert mlf.run_ui_url(URI + "/", "7", "r1") == f"{URI}/#/experiments/7/runs/r1"
    assert mlf.run_ui_url(URI, "7", "r1", "eval/x.csv") == f"{URI}/#/experiments/7/runs/r1/artifacts/eval/x.csv"


def test_keys_round_trip():
    assert hv.parse_key(hv.csv_key(URI + "/", "r1", "a b/c.csv")) == ("csv", URI, "r1", "a b/c.csv")
    assert hv.parse_key(hv.run_key(URI, "r1")) == ("run", URI, "r1", "")
    assert hv.parse_key("garbage") == ("", "", "", "")


# ---- the board ------------------------------------------------------------

def test_dropping_a_workspace_on_a_column_moves_it(home):
    from lookin.core import board as bd
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home.reload_workspaces()
    second = bd.add_column("done")["columns"][1]

    class Dragged:
        data = ws.id

    class Event:
        src_id = "_7"

    home.ctx.page.controls["_7"] = Dragged()
    home._on_workspace_drop(second, Event())
    assert bd.column_of(bd.load(), ws.id)["name"] == "done"


def test_removing_a_column_asks_first_then_moves_its_workspaces(home):
    from lookin.core import board as bd
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home.reload_workspaces()
    second = bd.add_column("done")["columns"][1]
    bd.move(ws.id, second["id"])
    home._on_remove_column(second)
    assert bd.column_of(bd.load(), ws.id)["name"] == "done"      # nothing yet: a dialog is open
    dialog = home.ctx.page.opened[-1]
    assert "group 1" in dialog.content.content.value
    dialog.actions[0].on_click(None)
    assert bd.column_of(bd.load(), ws.id)["name"] == "group 1"
    assert [c["name"] for c in bd.load()["columns"]] == ["group 1"]


def _new_workspace(home, name, column=None, open_it=False):
    home._on_new_workspace(None, column)
    dialog = home.ctx.page.opened[-1]
    dialog.content.content.controls[0].value = name
    dialog.actions[0 if open_it else 1].on_click(None)
    return next(w for w in lib.list_workspaces() if w.name == name)


def test_a_workspace_created_from_a_column_goes_in_it(home):
    from lookin.core import board as bd
    second = bd.add_column("project B")["columns"][1]
    ws = _new_workspace(home, "in B", column=second)
    assert bd.column_of(bd.load(), ws.id)["name"] == "project B"
    assert not ws.mount.dataset_path, "the dataset is optional"


def test_create_and_open_opens_the_new_workspace(home):
    ws = _new_workspace(home, "fresh", open_it=True)
    [(opened, _)] = home.workspace_panel.requested
    assert opened.id == ws.id


def test_a_dataset_set_on_the_open_workspaces_card_reaches_its_tab(home, tmp_path):
    dataset = tmp_path / "ships.csv"
    dataset.write_text("uuid,crop,label\n")
    ws = lib.create_workspace("A", lib.Mount(), hash_dataset=False)
    home.ctx.current_workspace = ws
    home._on_set_dataset(ws)
    dialog = home.ctx.page.opened[-1]
    dialog.content.content.controls[0].controls[0].value = str(dataset)
    dialog.actions[0].on_click(None)
    assert home.workspace_panel.followed.mount.dataset_path == str(dataset)


def test_the_last_column_is_not_offered_for_removal(home):
    from lookin.core import board as bd
    opened = len(home.ctx.page.opened)
    home._on_remove_column(bd.load()["columns"][0])
    assert len(home.ctx.page.opened) == opened
    assert "last column" in home.status.content.controls[1].value


# ---- fetching at scale --------------------------------------------------------

@pytest.fixture
def fake_server(home, monkeypatch):
    """search_runs answering from a dict, recording what was asked."""
    calls = []
    runs = {"7": [mlf.Run("r1", "a", "FINISHED", 3, experiment_id="7")],
            "8": [mlf.Run("r8", "b", "FINISHED", 9, experiment_id="8")]}

    def search_runs(uri, ids, filter_string=""):
        calls.append(list(ids))
        return [r for i in ids for r in runs.get(i, [])], False

    monkeypatch.setattr(mlf, "search_runs", search_runs)
    monkeypatch.setattr(home, "_render_all", lambda: None)
    return calls


def test_showing_an_experiment_fetches_only_that_one(home, fake_server):
    server = home.servers[URI]
    home.config.toggle_experiment(URI, "sandbox")
    home._fetch_worker(server, True, only=["8"])
    assert fake_server == [["8"]]
    assert [r.run_id for r in server.runs] == ["r1", "r2", "r3", "r8"]   # cls-v3 kept, sandbox added


def test_hiding_an_experiment_fetches_nothing(home, fake_server, monkeypatch):
    refreshed = []
    monkeypatch.setattr(home, "refresh", lambda *a, **k: refreshed.append((a, k)))
    home._toggle_experiment(URI, "cls-v3")          # shown -> not shown
    assert refreshed == [] and fake_server == []
    assert home.servers[URI].runs == []


def test_a_full_refresh_asks_one_request_per_shown_experiment(home, fake_server):
    home.config.toggle_experiment(URI, "sandbox")
    home._fetch_worker(home.servers[URI], True)
    assert sorted(map(tuple, fake_server)) == [("7",), ("8",)]


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


def test_the_experiment_row_stays_small_however_many_experiments_there_are(home):
    server = home.servers[URI]
    server.experiments = [mlf.Experiment(str(i), f"exp-{i:03d}") for i in range(100)]
    home.config.set_experiments(URI, {e.name for e in server.experiments}, set())
    home._render_experiments()
    chips = [c for c in home.experiments_row.controls if isinstance(c, hv.ft.Container)]
    assert len(chips) == hv.MAX_EXPERIMENT_CHIPS + 1          # eight chips and "+92 more"


def test_applying_the_picker_fetches_only_what_was_newly_ticked(home, fake_server, monkeypatch):
    refreshed = []
    monkeypatch.setattr(home, "refresh", lambda uri=None, runs_only=False, only=None: refreshed.append(only))
    server = home.servers[URI]
    home._apply_experiments(server, {"cls-v3", "sandbox"}, set())
    assert refreshed == [["8"]]
    home._apply_experiments(server, {"sandbox"}, {"cls-v3"})       # untick + hide: no request
    assert refreshed == [["8"]]
    assert home.config.hidden_experiments(URI) == ["cls-v3"]
    assert server.runs_by_exp.keys() <= {"8"}


def test_the_picker_opens_with_every_experiment(home):
    home.servers[URI].experiments = [mlf.Experiment(str(i), f"exp-{i:03d}") for i in range(300)]
    home._open_experiment_picker(home.servers[URI])
    dialog = home.ctx.page.opened[-1]
    assert dialog.title.value.startswith("Experiments on")


def test_an_experiment_unticked_during_a_refresh_does_not_come_back(home, monkeypatch):
    home.config.toggle_experiment(URI, "sandbox")
    server = home.servers[URI]

    def slow_search(uri, ids, filter_string=""):
        if ids == ["8"]:
            home.config.toggle_experiment(URI, "sandbox")     # the user unticks it meanwhile
        return [mlf.Run(f"r-{ids[0]}", "x", "FINISHED", 1, experiment_id=ids[0])], False

    monkeypatch.setattr(mlf, "search_runs", slow_search)
    monkeypatch.setattr(home, "_render_all", lambda: None)
    home._fetch_worker(server, True)
    assert "8" not in server.runs_by_exp
