"""The background-rebuild helper the expensive views run through.

Not a UI test: BackgroundRebuild only needs something with a `run_thread`,
so a two-line fake page exercises the whole contract -- which matters,
because the failure modes here (a dropped request, a job that never clears
its busy flag) look like a hung view rather than an error.
"""
from __future__ import annotations

import threading
import time

import pytest

from lookin.ui.components import BackgroundRebuild

SLOW = 0.20  # long enough that a follow-up request lands while the job runs


class FakePage:
    def run_thread(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()


class FakeCtx:
    def __init__(self):
        self.page = FakePage()


@pytest.fixture
def job() -> BackgroundRebuild:
    return BackgroundRebuild(FakeCtx(), "test")


def _appender(runs: list, tag: str, delay: float = SLOW):
    def work():
        time.sleep(delay)
        runs.append(tag)
    return work


def _wait_until_idle(job: BackgroundRebuild, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while job.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not job.busy, "the job never cleared its busy flag"


def test_work_runs_off_the_calling_thread(job):
    runs = []
    caller = threading.get_ident()
    job.run(lambda: runs.append(threading.get_ident()))
    _wait_until_idle(job)
    assert runs and runs[0] != caller, "the UI thread must not be the one doing the work"


def test_a_second_request_is_queued_not_dropped(job):
    """The views mutate their own state (active charts, active filters)
    before asking for a rebuild, so a dropped request leaves the view
    showing something the user did not ask for."""
    runs = []
    assert job.run(_appender(runs, "first")) is True
    assert job.run(_appender(runs, "second")) is False, "queued behind the running one"
    _wait_until_idle(job)
    assert runs == ["first", "second"]


def test_only_the_newest_queued_request_survives(job):
    """Clicking Add five times quickly costs two renders, not five -- and
    the last one reflects every click."""
    runs = []
    job.run(_appender(runs, "first"))
    for tag in ("stale_a", "stale_b", "newest"):
        job.run(_appender(runs, tag))
    _wait_until_idle(job)
    assert runs == ["first", "newest"]


def test_a_failing_job_reports_and_recovers(job):
    """A rebuild that dies silently leaves its progress banner spinning
    forever, which is indistinguishable from a hang."""
    seen = []

    def boom():
        raise RuntimeError("no")

    job.run(boom, on_error=seen.append)
    _wait_until_idle(job)
    assert len(seen) == 1 and isinstance(seen[0], RuntimeError)

    runs = []
    job.run(_appender(runs, "after", delay=0))
    _wait_until_idle(job)
    assert runs == ["after"], "the helper is usable again after a failure"


def test_a_failing_job_still_runs_what_was_queued_behind_it(job):
    runs = []

    def boom():
        time.sleep(SLOW)
        raise RuntimeError("no")

    job.run(boom, on_error=lambda ex: None)
    job.run(_appender(runs, "queued", delay=0))
    _wait_until_idle(job)
    assert runs == ["queued"]


def test_heavy_jobs_never_overlap():
    """matplotlib's pyplot keeps a global figure registry, so two rebuilds
    drawing at once can corrupt it -- the lock is process-wide, across
    views and across browser sessions."""
    overlaps = []
    active = []
    guard = threading.Lock()

    def work():
        with guard:
            active.append(1)
            overlaps.append(len(active))
        time.sleep(0.05)
        with guard:
            active.pop()

    jobs = [BackgroundRebuild(FakeCtx(), f"view{i}") for i in range(4)]
    for j in jobs:
        j.run(work)
    for j in jobs:
        _wait_until_idle(j)
    assert max(overlaps) == 1, f"jobs overlapped: {overlaps}"


def test_a_crop_that_fails_to_load_inside_a_rebuild_does_not_hang_it(job, tmp_path):
    """The Gallery's rebuild renders its mosaic, and a crop that cannot be
    read gets the "?" placeholder -- which is built under the same lock the
    rebuild already holds. With a non-reentrant lock that thread waited on
    itself forever, spinning "Rebuilding filters..." for good."""
    from lookin.ui import components

    components._PLACEHOLDER_PNG_B64 = None  # force the placeholder to be built here
    done = threading.Event()
    job.run(lambda: (components.file_to_full_base64(tmp_path / "missing.png"), done.set()))
    assert done.wait(10), "the rebuild deadlocked on its own lock"
