"""Several Lookin sessions writing the same files.

Every browser tab of `lookin web` is a session, `lookin ui` and `lookin web`
run side by side, and two people can share one library on a mount. Each
test here forces one interleaving that used to lose somebody's write.

The cross-process tests start a real second process (spawn, as on Windows
and macOS): a lock that only works between threads of one process is
exactly the bug they look for.
"""
from __future__ import annotations

import multiprocessing
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from lookin.core import board as bd
from lookin.core import fileio
from lookin.core import library as lib
from lookin.core import review as rv
from lookin.core.review import STATUS_EXCLUDE, STATUS_OK, ReviewStore, review_paths_for_csv

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"
_SPAWN = multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# helpers run in the second process (module level, so spawn can import them)
# ---------------------------------------------------------------------------

def _child_review_decide(paths, uid, ready, go, done):
    store = ReviewStore.load(*paths)
    store.decide(uid, STATUS_OK)
    ready.set()
    go.wait(30)
    store.save()
    done.set()


def _child_add_tag(path, tag, ready, go, done):
    ready.set()
    go.wait(30)

    def change(fresh):
        fresh.tags.append(tag)
        return True

    lib.update_workspace(path, change)
    done.set()


def _start(target, *args):
    ready, go, done = _SPAWN.Event(), _SPAWN.Event(), _SPAWN.Event()
    proc = _SPAWN.Process(target=target, args=(*args, ready, go, done), daemon=True)
    proc.start()
    assert ready.wait(60), "the second process did not start"
    return proc, go, done


# ---------------------------------------------------------------------------
# review decisions: lookin ui and lookin web on one dataset
# ---------------------------------------------------------------------------

def _review_paths(tmp_path):
    paths = review_paths_for_csv(tmp_path / "ships.csv")
    first = ReviewStore.load(*paths)
    first.ensure_active_version()
    first.save(force=True)
    return paths


def test_a_review_save_in_another_process_is_not_overwritten(tmp_path, monkeypatch):
    paths = _review_paths(tmp_path)
    proc, go, done = _start(_child_review_decide, paths, "u2")
    mine = ReviewStore.load(*paths)
    mine.decide("u1", STATUS_EXCLUDE)

    real_write = rv._atomic_write_csv
    calls = []

    def write(df, path):
        # The other process saves between this one's merge check and its
        # write: exactly the window the old process-only lock left open.
        if not calls:
            go.set()
            done.wait(2.0)
        calls.append(path)
        real_write(df, path)

    monkeypatch.setattr(rv, "_atomic_write_csv", write)
    mine.save()
    assert done.wait(30)
    proc.join(30)
    assert set(ReviewStore.load(*paths).effective()) == {"u1", "u2"}


def test_a_save_landing_while_a_store_loads_is_merged_later(tmp_path, monkeypatch):
    """The signature a store compares against must describe what it read,
    not what was on disk a moment after."""
    import pandas as pd

    paths = _review_paths(tmp_path)
    other = ReviewStore.load(*paths)
    other.decide("u2", STATUS_OK)
    real_read = pd.read_csv
    state = {"fired": False}

    def read(*args, **kwargs):
        out = real_read(*args, **kwargs)
        if not state["fired"] and Path(args[0]) == Path(paths[0]):
            state["fired"] = True
            t = threading.Thread(target=other.save)
            t.start()
            t.join(1.0)
            state["thread"] = t
        return out

    monkeypatch.setattr(pd, "read_csv", read)
    mine = ReviewStore.load(*paths)
    monkeypatch.setattr(pd, "read_csv", real_read)
    state["thread"].join(10)
    mine.decide("u1", STATUS_EXCLUDE)
    mine.save()
    assert set(ReviewStore.load(*paths).effective()) == {"u1", "u2"}


def test_a_write_within_one_mtime_tick_is_still_noticed(tmp_path):
    """Two saves inside one filesystem timestamp tick, with files of the
    same size: only the inode tells them apart."""
    paths = _review_paths(tmp_path)
    seed = ReviewStore.load(*paths)
    seed.author = "x"
    seed.decide("u1", STATUS_OK)
    seed.save()
    a, b = ReviewStore.load(*paths), ReviewStore.load(*paths)
    a.author = b.author = "x"
    before = [os.stat(p) for p in paths]
    a.reset("u1")
    a.decide("u2", STATUS_OK)
    a.save()
    for p, st in zip(paths, before):
        assert os.stat(p).st_size == st.st_size
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    b.decide("u3", STATUS_EXCLUDE)
    b.save()
    assert set(ReviewStore.load(*paths).effective()) == {"u2", "u3"}


def test_a_version_named_like_one_frozen_elsewhere_is_not_unfrozen(tmp_path):
    """Two tabs each create "the next version" -- both call it v2. One
    exports (freezes) it; the other's decisions must not reopen it."""
    paths = _review_paths(tmp_path)
    a, b = ReviewStore.load(*paths), ReviewStore.load(*paths)
    va = a.create_version()
    a.decide("u1", STATUS_EXCLUDE)
    a.freeze(va.version_id)
    a.save()
    vb = b.create_version()
    assert vb.version_id == va.version_id            # neither knew of the other
    b.decide("u2", STATUS_OK)
    b.save()
    disk = ReviewStore.load(*paths)
    assert disk.is_frozen(va.version_id)
    assert set(disk.own_decisions(va.version_id)) == {"u1"}
    # b's work is not dropped either: it is in a version of its own.
    others = [v for v in disk.versions if "u2" in disk.own_decisions(v)]
    assert others and others[0] != va.version_id
    assert b.active_version_id == others[0]
    assert set(b.own_decisions()) == {"u2"}


def test_a_decision_that_cannot_be_saved_is_reported(tmp_path, monkeypatch):
    from lookin.ui.review_widgets import record_decision

    paths = _review_paths(tmp_path)
    store = ReviewStore.load(*paths)

    class Ctx:
        class state:
            review = store
        notified = 0

        def notify_review(self):
            Ctx.notified += 1

    def refuse(*_a, **_k):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(rv, "_atomic_write_csv", refuse)
    ok, message = record_decision(Ctx(), ["u1"], STATUS_OK)
    assert not ok and "not saved" in message.lower()
    monkeypatch.undo()
    # Still pending, so the next save writes it.
    assert store.save()
    assert "u1" in ReviewStore.load(*paths).effective()


# ---------------------------------------------------------------------------
# workspace manifests, the board, personal favourites
# ---------------------------------------------------------------------------

def test_a_manifest_update_in_another_process_is_not_overwritten(tmp_path):
    workspace = lib.create_workspace("w", lib.Mount(dataset_path=""), root=tmp_path, hash_dataset=False)
    proc, go, done = _start(_child_add_tag, str(workspace.path), "theirs")

    def change(fresh):
        go.set()
        done.wait(2.0)
        fresh.tags.append("mine")
        return True

    lib.update_workspace(workspace.path, change)
    assert done.wait(30)
    proc.join(30)
    assert set(lib.load_workspace(workspace.path).tags) == {"mine", "theirs"}


def _barrier_on(monkeypatch, module, name):
    """Make the first two callers of module.name meet after reading, so both
    read the same file before either writes."""
    barrier = threading.Barrier(2, timeout=1.0)
    real = getattr(module, name)

    def wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return out

    monkeypatch.setattr(module, name, wrapped)


def _together(*fns):
    threads = [threading.Thread(target=f) for f in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)


def test_two_tabs_adding_groups_at_once_keep_both(tmp_path, monkeypatch):
    bd.load(tmp_path)
    _barrier_on(monkeypatch, bd, "_normalize")
    _together(lambda: bd.add_column("left", tmp_path), lambda: bd.add_column("right", tmp_path))
    monkeypatch.undo()
    names = {c["name"] for c in bd.load(tmp_path)["columns"]}
    assert {"left", "right"} <= names


def test_two_tabs_starring_at_once_keep_both(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    lib.save_personal({"favorites": [], "opened": {}})
    _barrier_on(monkeypatch, lib, "load_personal")
    _together(lambda: lib.set_favorite("lkw-aaaaaa", True), lambda: lib.mark_opened("lkw-bbbbbb"))
    monkeypatch.undo()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    personal = lib.load_personal()
    assert personal["favorites"] == ["lkw-aaaaaa"] and "lkw-bbbbbb" in personal["opened"]


def test_image_root_answer_keeps_what_home_added_meanwhile(tmp_path):
    workspace = lib.create_workspace("w", lib.Mount(dataset_path=""), root=tmp_path, hash_dataset=False)

    def add(fresh):
        fresh.mount.experiments.append(lib.MountExperiment(name="run", path="/x/run_results.csv"))
        return True

    lib.update_workspace(workspace.path, add)       # Home, while the dialog was open
    assert lib.rebase_image_root(workspace, "/crops")
    fresh = lib.load_workspace(workspace.path)
    assert fresh.mount.image_root == "/crops"
    assert [e.name for e in fresh.mount.experiments] == ["run"]


def test_home_config_notices_a_write_within_one_mtime_tick(tmp_path):
    from lookin.core.home_config import HomeConfig

    path = tmp_path / "home_view.json"
    a, b = HomeConfig(path), HomeConfig(path)
    a.add_server("http://one:5000")
    b.reload()
    before = os.stat(path)
    a.add_server("http://two:5000")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    b.add_server("http://three:5000")
    uris = {s["uri"] for s in HomeConfig(path).servers}
    assert {"http://one:5000", "http://two:5000", "http://three:5000"} <= uris


# ---------------------------------------------------------------------------
# the helpers themselves
# ---------------------------------------------------------------------------

def test_a_failed_write_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "x.json"

    def boom(tmp):
        tmp.write_text("half")
        raise OSError(28, "No space left on device")

    with pytest.raises(OSError):
        fileio.atomic_write(target, boom)
    assert list(tmp_path.iterdir()) == []


def test_a_stale_lock_is_broken_and_a_held_one_waited_for(tmp_path, monkeypatch):
    target = tmp_path / "f.json"
    lock = fileio.lock_path(target)
    lock.write_text("crashed 1\n")
    old = time.time() - fileio.LOCK_STALE_S - 5
    os.utime(lock, (old, old))
    t0 = time.monotonic()
    with fileio.locked(target):
        assert lock.exists()
    assert not lock.exists() and time.monotonic() - t0 < 2.0

    lock.write_text("alive 2\n")                    # fresh: someone is working
    t0 = time.monotonic()
    with fileio.locked(target, timeout=0.3):
        pass
    assert time.monotonic() - t0 >= 0.3
    assert lock.exists()                            # not ours to remove
    lock.unlink()


def test_locks_nest_within_a_thread(tmp_path):
    target = tmp_path / "f.json"
    with fileio.locked(target):
        with fileio.locked(target, timeout=0.2):
            pass
        assert fileio.lock_path(target).exists()
    assert not fileio.lock_path(target).exists()


def test_a_read_only_directory_does_not_block(tmp_path):
    missing = tmp_path / "gone" / "f.json"
    with fileio.locked(missing, timeout=0.2):
        pass
    assert not (tmp_path / "gone").exists()


# ---------------------------------------------------------------------------
# lookin web: background threads and the request path on one session
# ---------------------------------------------------------------------------

def test_a_merge_computed_on_the_old_data_is_not_cached(loaded_state, monkeypatch):
    """A rebuild job merges while the Sources tab relabels the dataset: the
    job's merge, made from the old labels, must not be what the next
    caller gets from the cache."""
    state = loaded_state
    name = next(iter(state.experiments))
    real = state._apply_review_overlay

    def overlay(merged, cols):
        state.apply_label_remap({"cat": "feline"})   # lands mid-merge
        return real(merged, cols)

    monkeypatch.setattr(state, "_apply_review_overlay", overlay)
    state.merged(name)
    monkeypatch.setattr(state, "_apply_review_overlay", real)
    assert "feline" in set(state.merged(name)[state.dataset.columns.label_col])


@pytest.fixture
def example(tmp_path):
    dest = tmp_path / "ex"
    shutil.copytree(EXAMPLE, dest)
    return dest


def test_reloading_the_page_while_a_snapshot_opens_does_not_load_it_twice(example, tmp_path, monkeypatch):
    from lookin.web import render
    from lookin.web.server import WebApp
    from lookin.web.session import RequestScope
    from lookin.web.views import sources as sources_mod

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    lib.set_library_root(tmp_path / "library")
    try:
        workspace = lib.create_workspace(
            "ex", lib.Mount(dataset_path=str(example / "dataset.csv")), hash_dataset=False)
        snapshot = lib.create_snapshot(workspace, {})
        release = threading.Event()
        calls = []

        def slow_restore(self, mount):
            calls.append(mount.dataset_path)
            release.wait(10)

        monkeypatch.setattr(sources_mod.SourcesView, "restore_mount", slow_restore)
        app = WebApp()
        session = app.store.get("r" * 24, create=True)
        query = {"snapshot": [snapshot.id]}
        with RequestScope(session):
            render.boot(session, query)
        with RequestScope(session):                 # F5 while it loads
            render.boot(session, query)
        time.sleep(0.3)
        release.set()
        time.sleep(0.3)
        assert len(calls) == 1
        session.close()
    finally:
        lib.set_library_root(None)


def test_a_sidecar_broken_elsewhere_is_reported_not_overwritten(tmp_path):
    from lookin.ui.review_widgets import record_decision

    paths = _review_paths(tmp_path)
    store = ReviewStore.load(*paths)
    Path(paths[0]).write_bytes(b'version_id,uuid\n"v1,u0\n\xff\xfe')   # broken by someone else

    class Ctx:
        class state:
            review = store

        def notify_review(self):
            pass

    broken = Path(paths[0]).read_bytes()
    ok, message = record_decision(Ctx(), ["u1"], STATUS_OK)
    assert not ok and "not saved" in message.lower()
    assert Path(paths[0]).read_bytes() == broken
