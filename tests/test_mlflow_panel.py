"""Tests for ui/mlflow_panel: the formatting helpers, and the panel itself
rendered without a browser.

The helpers are what render the run list and the run details panel, so a
value one of them refuses is a run the user cannot select. The panel tests
below cover what keeps a slow tracking server from feeling like a frozen
app: the render caps, moving a selection without rebuilding a list, and the
elapsed counters on everything the panel is waiting for.
"""
from __future__ import annotations

import math

import pytest

from lookin.ui.mlflow_panel import _format_metric


@pytest.mark.parametrize("value, expected", [
    (3.0, "3"),
    (-7.0, "-7"),
    (0.0, "0"),
    (0.918273, "0.9183"),
    (1.5e-6, "1.500e-06"),
    (2.5e7 + 0.5, "2.500e+07"),  # integers stay integers, so this is not one
])
def test_format_metric_shapes_a_finite_value(value, expected):
    assert _format_metric(value) == expected


@pytest.mark.parametrize("value, expected", [
    (float("nan"), "NaN"),
    (float("inf"), "+inf"),
    (float("-inf"), "-inf"),
])
def test_format_metric_survives_a_non_finite_metric(value, expected):
    """MLflow stores and returns NaN/Infinity metrics (a diverged loss, an
    AUC over a class with no positive sample). int() refuses both, so this
    used to raise "cannot convert float NaN to integer" while rendering the
    run details -- from a worker thread, where it just killed the run
    selection with nothing shown."""
    assert _format_metric(value) == expected


def test_every_float_a_run_can_carry_formats_to_something():
    """The panel has no fallback around this call, so the contract is that
    it never raises, whatever runs/search happened to return."""
    for value in (0.0, -0.0, 1e308, -1e308, 5e-324, math.nan, math.inf, -math.inf):
        assert isinstance(_format_metric(value), str)


# ---------------------------------------------------------------------------
# The panel itself, rendered without a browser.
#
# Same harness as tests/test_slice_panel.py: every .update() in the panel is
# guarded by `if control.page:`, so a context whose page has no `page`
# attribute on its controls renders everything and skips the pushes. What is
# checked here is the machinery that keeps a slow tracking server from
# feeling like a frozen app -- the render caps, the in-place selection, and
# the "how long has this been loading" counters.
# ---------------------------------------------------------------------------
import threading

import flet as ft

from lookin.core import mlflow_client as mlf
from lookin.ui.mlflow_panel import (
    MAX_KV_ROWS, MAX_RENDERED_ROWS, MlflowPanel,
    _format_elapsed, _format_transfer,
)


class FakePage:
    """Enough page for construction: the panel stores an on_disconnect
    handler and reads width/height when the dialog opens."""
    width = 1440
    height = 920
    on_disconnect = None

    def close(self, dialog):
        pass


class FakeCtx:
    def __init__(self):
        self.page = FakePage()
        self.state = None
        self.upload_view = None
        self.go_to_data_tab = None


@pytest.fixture
def panel(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # never read the user's real prefs
    p = MlflowPanel(FakeCtx())
    p._poll_stop.set()  # the health-check thread would dial a server from a test
    yield p
    p._poll_stop.set()


def _experiments(n: int) -> list:
    return [mlf.Experiment(experiment_id=str(i), name=f"exp-{i:04d}") for i in range(n)]


def _runs(n: int) -> list:
    return [mlf.Run(run_id=f"r{i:04d}", run_name=f"run-{i:04d}", status="FINISHED",
                    start_time=1_700_000_000_000 + i, end_time=1_700_000_060_000 + i)
            for i in range(n)]


def test_a_huge_experiment_list_is_not_rendered_in_full(panel):
    """A tracking server with thousands of experiments used to build (and
    push) one row per experiment on every keystroke in the filter box."""
    panel.experiments = _experiments(2000)
    panel._render_experiments()
    assert len(panel._experiment_rows) == MAX_RENDERED_ROWS
    # ... and the list says so rather than pretending that is everything.
    note = panel.experiments_list.controls[-1].content.value
    assert "2,000" in note


def test_the_selected_row_survives_the_cap(panel):
    """A restored 'recent' pick can sit well past the cap; a selection
    nobody can see is worse than a row shown out of order."""
    panel.experiments = _experiments(2000)
    panel.selected_experiment = panel.experiments[1500]
    panel._render_experiments()
    assert "1500" in panel._experiment_rows
    assert panel._experiment_rows["1500"].bar.bgcolor is not ft.Colors.TRANSPARENT


def test_selecting_a_run_moves_the_highlight_without_rebuilding_the_list(panel):
    panel.runs = _runs(50)
    panel._render_runs()
    before = list(panel.runs_list.controls)
    first, second = panel._run_rows["r0000"], panel._run_rows["r0001"]
    first.set_selected(True)

    assert panel._select_in_list(panel._run_rows, "r0000", "r0001")
    assert panel.runs_list.controls == before  # nothing was rebuilt
    assert second.bar.bgcolor is not ft.Colors.TRANSPARENT
    assert first.bar.bgcolor is ft.Colors.TRANSPARENT


def test_selecting_a_row_that_is_not_rendered_asks_for_a_full_render(panel):
    """The caller falls back to _render_* when this returns False -- without
    it, a run filtered out of the list would be selected invisibly."""
    panel.runs = _runs(5)
    panel._render_runs()
    assert panel._select_in_list(panel._run_rows, None, "r9999") is False


def test_a_run_with_hundreds_of_params_builds_only_what_is_open(panel):
    """The params tile starts collapsed: building its rows anyway cost a
    Row of two Texts per param on every click in the run list."""
    run = _runs(1)[0]
    run.params = {f"p{i}": i for i in range(500)}
    run.metrics = {"accuracy": 0.9}
    panel.selected_run = run
    panel._render_details()

    tiles = [c for c in panel.details_column.controls if isinstance(c, ft.ExpansionTile)]
    metrics_tile, params_tile = tiles[0], tiles[1]
    assert metrics_tile.controls[0].content.controls  # expanded: built
    assert params_tile.controls[0].content.controls == []  # collapsed: empty

    params_tile.on_change(type("E", (), {"data": "true"})())
    rows = params_tile.controls[0].content.controls
    assert len(rows) == MAX_KV_ROWS + 1  # the cap, plus the line that admits it


def test_a_run_whose_metrics_are_unformattable_still_renders(panel):
    """_format_metric is called for every metric while building the details
    panel, from a worker thread, with no fallback around it."""
    run = _runs(1)[0]
    run.metrics = {"loss": float("nan"), "auc": float("inf")}
    panel.selected_run = run
    panel._render_details()
    assert panel.details_column.controls


def test_a_loading_list_says_how_long_it_has_been_loading(panel):
    """The whole point: a spinner says 'something is happening', which is
    not the question after ten seconds."""
    text = panel._show_loading(panel.runs_list, "Loading runs")
    panel._begin_task("Loading runs", text)
    assert panel.busy_row.visible

    panel._tasks[0].started -= 12.0
    panel._tick_tasks()
    assert text.value == "Loading runs... 12s"
    assert panel.busy_label.value == "Loading runs... 12s"

    panel._end_task("Loading runs")
    assert not panel.busy_row.visible
    assert not panel._busy()


def test_the_health_check_stands_aside_while_the_user_waits(panel):
    """Two requests to a slow server queue up, and the one the user is
    waiting for is the one that gets longer."""
    assert not panel._busy()
    panel._begin_task("Loading runs")
    assert panel._busy()


def test_the_filter_box_re_renders_once_the_typing_stops(panel):
    panel.experiments = _experiments(30)
    panel.experiment_filter.value = "exp-0007"
    for _ in range(8):  # eight keystrokes, one render
        panel._on_experiment_filter_change(None)
    timer = panel._filter_timers["experiments"]
    timer.join(timeout=2.0)
    assert not timer.is_alive()
    assert len(panel._experiment_rows) == 1


def test_download_progress_reads_as_a_transfer(panel):
    report = panel._download_progress("results.csv")
    report(5 * 1024 * 1024, 20 * 1024 * 1024)
    assert panel.download_bar.visible
    assert panel.download_bar.value == pytest.approx(0.25)
    panel._hide_download_bar()
    assert not panel.download_bar.visible


@pytest.mark.parametrize("done, total, expected", [
    (5 * 1024 * 1024, 20 * 1024 * 1024, "5MB of 20MB (25%)"),
    (2048, None, "2KB"),  # a server that sends no Content-Length
    (0, 0, "0B"),
])
def test_format_transfer(done, total, expected):
    assert _format_transfer(done, total) == expected


@pytest.mark.parametrize("seconds, expected", [
    (0.4, ""),  # nothing to say yet; the spinner already said it
    (9.7, "9s"),
    (75.0, "1m15s"),
])
def test_format_elapsed(seconds, expected):
    assert _format_elapsed(seconds) == expected


def test_a_stale_worker_still_clears_its_task(panel, monkeypatch):
    """The user clicking another experiment mid-flight makes the in-flight
    result stale -- but a task left on the stack would leave the dialog
    saying 'Loading runs...' forever."""
    exp = mlf.Experiment(experiment_id="1", name="one")
    monkeypatch.setattr(mlf, "list_runs", lambda uri, eid: _runs(3))
    panel._begin_task("Loading runs")
    panel.selected_experiment = mlf.Experiment(experiment_id="2", name="two")  # moved on
    panel._fetch_runs_worker("http://x", exp)
    assert not panel._busy()
    assert panel.runs == []  # and the stale result was dropped


def test_a_giant_error_message_never_reaches_the_status_row():
    """The message comes from whatever failed while reading the CSV, and
    pandas quotes the offending data: a downloaded file that is not a CSV
    parses as one enormous column name, and the resulting "available
    columns" error froze the page as a single Text control."""
    from lookin.ui.mlflow_panel import MAX_STATUS_CHARS, _short
    short = _short("Column 'uuid' not found. Available: " + "x" * 40_000_000)
    assert len(short) < MAX_STATUS_CHARS + 60
    assert short.startswith("Column 'uuid' not found.")


def test_a_normal_message_is_left_alone():
    from lookin.ui.mlflow_panel import _short
    assert _short("loaded 12,000 rows") == "loaded 12,000 rows"


def test_the_hidden_run_count_does_not_outlive_its_runs(panel, monkeypatch):
    """It is rendered outside the run list, so a targeted update that
    forgets it leaves '(43 hidden)' standing over a list that is loading a
    different experiment -- or over 'Select an experiment' after a
    reconnect."""
    panel.runs = _runs(20)
    panel.runs[0].status = "FAILED"
    panel._render_runs()
    assert panel.hidden_runs_text.value == "(1 hidden)"

    monkeypatch.setattr(mlf, "list_runs", lambda uri, eid: [])  # no server in a test
    panel._on_select_experiment(mlf.Experiment(experiment_id="9", name="another"))
    assert panel.hidden_runs_text.value == ""


def test_the_details_heading_follows_what_the_details_panel_shows(panel):
    """The heading is a sibling of the details column, not a child of it:
    an update that names only the column leaves 'Run details' standing over
    an experiment, and vice versa."""
    panel.selected_experiment = mlf.Experiment(experiment_id="1", name="one")
    panel._render_details()
    assert panel.details_title.value == "Experiment details"

    panel.selected_run = _runs(1)[0]
    panel._render_details()
    assert panel.details_title.value == "Run details"

    panel.selected_run = None
    panel.selected_experiment = None
    panel._render_details()
    assert panel.details_title.value == "Run details"


def test_sidecars_are_those_of_the_run_being_loaded_not_the_one_now_selected(panel, monkeypatch):
    """Loading runs on a worker, and the run list stays clickable meanwhile.
    The sidecars used to be looked up in the panel's *current* artifact
    listing once the CSV was in: selecting another run during a long
    download emptied it, and the loaded run's embeddings were silently not
    attached."""
    from pathlib import Path
    from types import SimpleNamespace

    run_a = mlf.Run(run_id="a", run_name="a", status="FINISHED", start_time=1)
    embed = mlf.ArtifactFile("eval/test_embeddings.npz", 20)
    panel.selected_run = run_a
    panel.selected_artifact = mlf.ArtifactFile("eval/test_results.csv", 10)
    panel.npz_artifacts = {embed.path: embed}
    attached = []
    panel.ctx.state = SimpleNamespace(
        dataset=object(), experiments={"mine": SimpleNamespace(embeddings_path="", model_prototypes_path="")})
    panel.ctx.load_experiment_from_file = lambda path, name: (True, "loaded")
    panel.ctx.prototype_source = SimpleNamespace(
        load_embeddings_for=lambda name, path: (attached.append(path) or True, "20 embeddings"))
    started = []
    monkeypatch.setattr(threading, "Thread", lambda target, args, daemon: SimpleNamespace(
        start=lambda: started.append((target, args))))
    panel.load_name_field.value = "mine"
    panel._on_load_click(None)
    [(target, args)] = started

    def download(uri, run_id, path, size, progress=None):
        if path.endswith(".csv"):
            # The user clicks another run while the CSV comes down.
            panel._on_select_run(mlf.Run(run_id="b", run_name="b", status="FINISHED", start_time=2))
        return Path("/cache") / run_id / path

    monkeypatch.setattr(panel, "_download_cached", download)
    target(*args)
    assert attached == [str(Path("/cache/a/eval/test_embeddings.npz"))]
