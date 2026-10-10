"""The Home page's settings file: hand-editable, and never clobbered."""
from __future__ import annotations

import json

import pytest

from lookin.core import home_config as hc
from lookin.core import mlflow_client as mlf


@pytest.fixture
def path(tmp_path):
    return tmp_path / "home_view.json"


def test_a_first_start_uses_the_known_tracking_uri(path):
    cfg = hc.HomeConfig(path, fallback_uri="http://mlflow.lab:5000/")
    assert cfg.servers == [{"name": "mlflow.lab", "uri": "http://mlflow.lab:5000"}]


def test_every_change_is_written_to_the_file(path):
    cfg = hc.HomeConfig(path, fallback_uri="http://a:5000")
    cfg.set_experiment_hidden("http://a:5000", "Default", True)
    cfg.set_run_hidden("http://a:5000", "r1", "bad run", True)
    cfg.set_columns(["metrics.val_acc", "params.backbone"])
    on_disk = json.loads(path.read_text())
    assert on_disk["experiments"]["http://a:5000"]["hidden"] == ["Default"]
    assert on_disk["runs"]["hidden"][0]["run_id"] == "r1"
    assert [c["key"] for c in on_disk["columns"]] == ["metrics.val_acc", "params.backbone"]


def test_a_hand_edit_is_picked_up(path):
    cfg = hc.HomeConfig(path, fallback_uri="http://a:5000")
    cfg.save()
    raw = json.loads(path.read_text())
    raw["columns"] = [{"key": "metrics.ece", "label": "ECE"}, "params.lr", {"key": "nonsense"}]
    path.write_text(json.dumps(raw))
    import os
    os.utime(path, ns=(1, 1))   # a different mtime than the one just written
    assert cfg.changed_on_disk()
    cfg.reload()
    assert [hc.column_label(c) for c in cfg.columns] == ["ECE", "lr"]


def test_a_file_that_does_not_parse_is_never_overwritten(path):
    path.write_text('{"servers": [,]}')
    cfg = hc.HomeConfig(path)
    assert cfg.error
    cfg.set_run_option("search", "vit")
    assert path.read_text() == '{"servers": [,]}'


def test_removing_every_server_is_respected(path):
    cfg = hc.HomeConfig(path, fallback_uri="http://a:5000")
    cfg.remove_server("http://a:5000")
    assert hc.HomeConfig(path, fallback_uri="http://a:5000").servers == []


def test_the_editor_refuses_invalid_json_and_writes_nothing(path):
    cfg = hc.HomeConfig(path, fallback_uri="http://a:5000")
    cfg.save()
    before = path.read_text()
    assert cfg.replace_text("{nope") .startswith("Not valid JSON")
    assert path.read_text() == before


def test_a_new_server_shows_its_most_recent_experiments(path):
    cfg = hc.HomeConfig(path)
    exps = [mlf.Experiment(str(i), f"e{i}", last_update_time=i) for i in range(6)]
    assert cfg.shown_experiments("http://a:5000") == []   # rendering before the first listing
    cfg.adopt_defaults("http://a:5000", exps)
    assert cfg.shown_experiments("http://a:5000") == ["e5", "e4", "e3"]
    cfg.toggle_experiment("http://a:5000", "e5")
    cfg.adopt_defaults("http://a:5000", exps)   # already known: left alone
    assert cfg.shown_experiments("http://a:5000") == ["e4", "e3"]


def test_hiding_an_experiment_also_unshows_it(path):
    cfg = hc.HomeConfig(path)
    cfg.toggle_experiment("http://a:5000", "e1")
    cfg.set_experiment_hidden("http://a:5000", "e1", True)
    assert cfg.shown_experiments("http://a:5000") == []


def test_choosing_columns_keeps_labels_typed_in_the_file(path):
    cfg = hc.HomeConfig(path)
    cfg.data["columns"] = [{"key": "metrics.ece", "label": "ECE"}]
    cfg.set_columns(["metrics.ece", "params.lr"])
    assert cfg.columns == [{"key": "metrics.ece", "label": "ECE"}, {"key": "params.lr"}]


def test_column_values_come_from_params_and_metrics():
    run = mlf.Run("r", "n", "FINISHED", 0, params={"lr": "3e-4"}, metrics={"acc": 0.93125, "nan": float("nan")})
    assert hc.column_value(run, {"key": "params.lr"}) == "3e-4"
    assert hc.column_value(run, {"key": "metrics.acc"}) == "0.9313"
    assert hc.column_value(run, {"key": "metrics.nan"}) == "NaN"
    assert hc.column_value(run, {"key": "metrics.missing"}) == ""


def test_a_server_with_only_the_default_experiment_is_asked_again_later(path):
    cfg = hc.HomeConfig(path)
    cfg.adopt_defaults("http://a:5000", [mlf.Experiment("0", "Default", last_update_time=9)])
    assert not cfg.knows_server("http://a:5000")
    cfg.adopt_defaults("http://a:5000", [mlf.Experiment("0", "Default", last_update_time=9),
                                         mlf.Experiment("1", "cls", last_update_time=1)])
    assert cfg.shown_experiments("http://a:5000") == ["cls"]


def test_a_change_starts_from_what_another_tab_just_wrote(path):
    first = hc.HomeConfig(path, fallback_uri="http://a:5000")
    second = hc.HomeConfig(path, fallback_uri="http://a:5000")
    first.set_run_hidden("http://a:5000", "r1", "one", True)
    second.set_run_hidden("http://a:5000", "r2", "two", True)   # second has not reloaded yet
    assert hc.HomeConfig(path).hidden_run_ids("http://a:5000") == {"r1", "r2"}


def test_parallel_requests_is_a_bounded_setting(path):
    path.write_text(json.dumps({"parallel_requests": 999}))
    assert hc.HomeConfig(path).parallel_requests == hc.MAX_PARALLEL_REQUESTS
    path.write_text(json.dumps({"parallel_requests": "x"}))
    assert hc.HomeConfig(path).parallel_requests == hc.DEFAULT_PARALLEL_REQUESTS
