"""The top bar's "still running" line, and the saves that run behind it.

A dataset load starts the rebuild of the tab on screen, which begins after
the load and ends before it: ending "the latest entry" removed the rebuild's
and left "Loading dataset" up for good. And a save writes in the
background, each step in the top bar and the server log, so a slow disk is
told apart from a hang."""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest

from lookin.core import library as lib
from lookin.web import render
from lookin.web.server import WebApp
from lookin.web.session import RequestScope

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"


def _session(sid: str):
    session = WebApp().store.get(sid, create=True)
    with RequestScope(session):
        render.boot(session, {})
    return session


def _wait(condition, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_jobs_ending_out_of_order_each_remove_their_own_entry():
    s = _session("b" * 24)
    load = s.busy.begin("Loading dataset")
    rebuild = s.busy.begin("Metrics")
    s.busy.detail(load, "10 / 20 rows")
    assert s.busy.entries == [("Loading dataset", "10 / 20 rows"), ("Metrics", "")]
    s.busy.end(rebuild)
    assert s.busy.running == ["Loading dataset"]
    s.busy.end(load)
    assert s.busy.running == []
    # ending twice, or a detail after the end, brings nothing back
    s.busy.end(load)
    s.busy.detail(load, "late")
    assert s.busy.entries == []


def test_the_top_bar_says_what_is_running_and_at_which_step():
    s = _session("c" * 24)
    entry = s.busy.begin("Saving workspace")
    s.busy.detail(entry, "Hashing dataset.csv: 12 MB of 80 MB")
    s.mode = "analysis"
    top = render.render_top(s)
    assert "Saving workspace -- Hashing dataset.csv: 12 MB of 80 MB" in top
    s.busy.end(entry)
    assert 'class="busy"' not in render.render_top(s)


def test_a_dataset_load_does_not_stay_in_the_top_bar_after_a_tab_switch(tmp_path):
    """The load notifies the tab on screen, whose rebuild begins after the
    load's entry and ends whenever it ends."""
    example = tmp_path / "ex"
    shutil.copytree(EXAMPLE, example)
    s = _session("d" * 24)
    with RequestScope(s):
        render.dispatch(s, "app", "tab", {"tab": "metrics"})
    sources = s.view("sources")
    sources.set_dataset_path(str(example / "dataset.csv"))
    assert sources.load_dataset_now()
    _wait(lambda: not any(j.busy for j in list(s._jobs.values())))
    assert "Loading dataset" not in s.busy.running
    assert s.busy.running == []


def test_a_background_task_logs_its_steps_and_runs_once_at_a_time(capsys):
    s = _session("e" * 24)
    release = threading.Event()

    def work(step):
        step("Hashing")
        release.wait(10)
        step("Writing")

    assert s.background("Saving workspace", work)
    _wait(lambda: s.busy.entries == [("Saving workspace", "Hashing")])
    assert not s.background("Saving workspace", work)       # a second click waits
    release.set()
    _wait(lambda: not s.background_running("Saving workspace"))
    assert s.busy.entries == []
    log = capsys.readouterr().out
    assert "Saving workspace: started" in log
    assert "Saving workspace: Hashing --" in log
    assert "Saving workspace: Writing --" in log
    assert "Saving workspace: done in" in log


def test_a_failing_background_task_says_on_which_step(capsys):
    s = _session("f" * 24)

    def work(step):
        step("Writing manifest.json")
        raise OSError("disk full")

    assert s.background("Saving workspace", work)
    _wait(lambda: not s.background_running("Saving workspace"))
    assert s.busy.entries == []
    assert "while Writing manifest.json: disk full" in capsys.readouterr().out


@pytest.fixture
def library(tmp_path):
    lib.set_library_root(tmp_path / "lib")
    try:
        yield tmp_path
    finally:
        lib.set_library_root(None)


def test_save_writes_the_workspace_in_the_background_step_by_step(library, capsys):
    example = library / "ex"
    shutil.copytree(EXAMPLE, example)
    s = _session("g" * 24)
    sources = s.view("sources")
    sources.set_dataset_path(str(example / "dataset.csv"))
    assert sources.load_dataset_now()
    ws = lib.create_workspace("saved", lib.Mount(dataset_path=""))
    s.current_workspace = ws
    with RequestScope(s):
        render.dispatch(s, "app", "save_workspace", {})
    _wait(lambda: not s.background_running("Saving workspace"))
    assert s.current_workspace.mount.dataset_path == str(example / "dataset.csv")
    assert s.current_workspace.dataset_hash
    log = capsys.readouterr().out
    for step in ("Hashing dataset.csv", "Waiting for the lock on manifest.json",
                 "Writing manifest.json", "Saving workspace: done in"):
        assert step in log, step


def test_library_steps_reach_the_progress_callback(library):
    data = library / "data.csv"
    data.write_text("uuid,label\n" + "".join(f"{i},a\n" for i in range(1000)))
    steps = []
    ws = lib.create_workspace("w", lib.Mount(dataset_path=str(data)), progress=steps.append)
    assert ws is not None
    assert steps[0].startswith("Hashing data.csv")
    assert steps[-1] == "Writing manifest.json"
    steps.clear()
    assert lib.create_snapshot(ws, {}, progress=steps.append) is not None
    assert steps and steps[-1].startswith("Writing ")
    steps.clear()
    ok, _ = lib.export_workspace(ws, library / "out.zip", progress=steps.append)
    assert ok and any(s.startswith("Compressing data.csv") for s in steps)
    steps.clear()
    imported, _ = lib.import_workspace(library / "out.zip", progress=steps.append)
    assert imported is not None and any(s.startswith("Extracting data.csv") for s in steps)
