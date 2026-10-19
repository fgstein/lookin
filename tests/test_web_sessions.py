"""The browser interface (`lookin web`): one session per browser tab, and
nothing shared between them; every snapshot view restores what it saved;
every tab renders."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lookin.ui import view_state as vs
from lookin.web import render
from lookin.web.server import WebApp
from lookin.web.session import RequestScope

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"


@pytest.fixture
def example(tmp_path):
    """A copy of the example data: review decisions are written next to the
    dataset CSV, and the repository's copy must stay untouched."""
    dest = tmp_path / "ex"
    shutil.copytree(EXAMPLE, dest)
    return dest


def _session(app: WebApp, sid: str):
    session = app.store.get(sid, create=True)
    with RequestScope(session):
        render.boot(session, {})
    return session


def _load(session, example: Path, experiments: list) -> None:
    sources = session.view("sources")
    sources.set_dataset_path(str(example / "dataset.csv"))
    assert sources.load_dataset_now(then_experiments=experiments)


def test_two_tabs_do_not_share_state(example):
    app = WebApp()
    a = _session(app, "a" * 24)
    b = _session(app, "b" * 24)
    _load(a, example, [("baseline", str(example / "baseline_cnn_results.csv"))])
    assert a.state.dataset is not None
    assert b.state.dataset is None
    assert list(a.state.experiments) == ["baseline"]
    _load(b, example, [("resnet", str(example / "finetuned_resnet_results.csv"))])
    assert list(a.state.experiments) == ["baseline"]
    assert list(b.state.experiments) == ["resnet"]
    assert a.view("gallery") is not b.view("gallery")
    a.view("gallery").on_shuffle(value=True)
    assert b.view("gallery").shuffle_seed == 0


def test_every_tab_renders(example):
    app = WebApp()
    s = _session(app, "c" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv")),
                       ("resnet", str(example / "finetuned_resnet_results.csv"))])
    for tab in render.TAB_KEYS:
        with RequestScope(s):
            s.goto(tab)
            html = render.render_view(s, tab)
        assert "failed to render" not in html, (tab, html[:300])


def test_snapshot_views_round_trip(example):
    app = WebApp()
    s = _session(app, "d" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv")),
                       ("resnet", str(example / "finetuned_resnet_results.csv"))])
    gallery = s.view("gallery")
    gallery._refresh_now()
    gallery.cols_value = "5"
    s.view("metrics").metric = "recall"
    captured = s.view_state.capture()
    views = captured[vs.VIEWS]
    for key in (vs.GALLERY, vs.METRICS, vs.CALIBRATION, vs.COMPARE, vs.EMBEDDINGS):
        assert key in views, key

    t = _session(app, "e" * 24)
    _load(t, example, [("baseline", str(example / "baseline_cnn_results.csv")),
                       ("resnet", str(example / "finetuned_resnet_results.csv"))])
    for key, state in views.items():
        view = t.view_state._views[key]
        assert view.restore_view_state(state) == [], key
        assert view.view_state() == state, key


def test_an_open_workspace_follows_its_manifest(example, tmp_path):
    """A dataset set on the Home card after the workspace was opened loads
    into the empty tab; a run added later loads next to what is there."""
    from lookin.core import library as lib
    from lookin.web.views.workspaces import _remember_loadable, follow_workspace

    lib.set_library_root(tmp_path / "lib")
    try:
        ws = lib.create_workspace("follow me", lib.Mount(dataset_path=""))
        s = _session(WebApp(), "g" * 24)
        s.current_workspace = ws
        _remember_loadable(s, ws)

        fresh = lib.update_workspace_mount(ws.path, lib.Mount(dataset_path=str(example / "dataset.csv")))
        assert follow_workspace(s, fresh)
        assert s.state.dataset is not None
        assert not s.state.experiments

        mount = lib.Mount(dataset_path=str(example / "dataset.csv"), experiments=[
            lib.MountExperiment(name="baseline", path=str(example / "baseline_cnn_results.csv"))])
        fresh = lib.update_workspace_mount(ws.path, mount)
        assert follow_workspace(s, fresh)
        assert list(s.state.experiments) == ["baseline"]
    finally:
        lib.set_library_root(None)


def test_gallery_loads_as_it_is_scrolled(example):
    """No pages: a first batch, then appends asked for by the page -- each
    only the new crops, and a request from a grid that has since started
    over is ignored. The mosaic is the carousel, drawn in the tab."""
    from lookin.web.views.base import NO_RENDER

    s = _session(WebApp(), "h" * 24)
    _load(s, example, [])
    gallery = s.view("gallery")
    gallery._refresh_now()
    gallery.built = True
    car = gallery.carousel
    with RequestScope(s):
        html = render.render_view(s, "gallery")
    first = car.loaded
    assert 0 < first < 800
    assert html.count('<div class="ctile') == first
    assert "scroll for more" not in html
    with RequestScope(s):
        assert gallery.on_car(op="more", gen=car.gen, loaded=first) is NO_RENDER
        cmd = s._pending_cmds[-1]["append"]
    assert cmd["start"] == first and car.loaded > first
    assert cmd["html"].count('<div class="ctile') == car.loaded - first
    assert cmd["count"] == f"{car.total:,} image(s)"
    stale = car.gen
    car.zoom_index = 0
    gallery._set_shuffle(7)
    assert gallery.carousel is not car and gallery.carousel.zoom_index == 0
    with RequestScope(s):
        assert gallery.on_car(op="more", gen=stale, loaded=gallery.carousel.loaded) is NO_RENDER
    state = gallery.view_state()
    assert "page" not in state and "rows" not in state
    # A snapshot written before the mosaic scrolled still restores.
    assert gallery.restore_view_state({**state, "rows": "4", "page": 3}) == []


def test_the_filter_reaches_every_tab_and_is_saved(example, tmp_path):
    """Applied from the Filter tab: the merge, the mount a workspace saves,
    and the snapshot all carry it; a snapshot without one removes it."""
    from lookin.core import library as lib
    from lookin.ui import view_state as vs

    s = _session(WebApp(), "i" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv"))])
    total = len(s.state.dataset.df)
    f = s.view("filter")
    with RequestScope(s):
        f.on_choose_column(value="label")
        f.on_add()
        cid = f.conditions[0]["_cid"]
        f.on_pick(cid=cid, v="cat", value=True)
        assert len(s.state.dataset.df) == total          # a draft moves nothing
        f.on_apply()
    kept = len(s.state.dataset.df)
    assert 0 < kept < total
    assert len(s.state.merged("baseline")) == kept
    mount = s.view("sources").current_mount()
    assert mount.dataset_filter["conditions"][0]["values"] == ["cat"]
    assert lib.mount_from_dict(lib.mount_to_dict(mount)).dataset_filter == mount.dataset_filter

    captured = s.view_state.capture()
    assert captured[vs.VIEWS][vs.DATASET_FILTER]["conditions"]
    # The Filter tab is saved as the tab it sits under: the index is shared with `lookin ui`.
    with RequestScope(s):
        s.goto("filter")
    assert s.view_state.capture()[vs.APP][vs.TAB] == 0

    with RequestScope(s):
        f.on_clear()
    assert len(s.state.dataset.df) == total
    empty = s.view_state.capture()
    with RequestScope(s):
        assert s.view_state.restore(captured) == []
    assert len(s.state.dataset.df) == kept
    with RequestScope(s):
        s.view_state.restore(empty)
    assert len(s.state.dataset.df) == total

    # A workspace opened with a filter applies it on load.
    t = _session(WebApp(), "j" * 24)
    with RequestScope(t):
        t.view("sources").restore_mount(mount)
    assert len(t.state.dataset.df) == kept
    assert t.view("filter").conditions[0]["values"] == ["cat"]


def test_lookin_toml_fills_the_columns_and_the_root(example):
    """Picked: lookin.toml wins over the guess. Restored: only the fields
    still at their default take it; a column it names that the file lacks
    is a warning, not a choice."""
    import pandas as pd

    from lookin.core import library as lib

    df = pd.read_csv(example / "dataset.csv").rename(columns={"uuid": "id", "label": "cls"})
    df.to_csv(example / "renamed.csv", index=False)
    (example / "lookin.toml").write_text(
        '[defaults]\nroot = "imgs"\n\n["renamed.csv"]\nid = "id"\nlabel = "cls"\n\n'
        '["*.csv"]\nlabel = "nope"\n', encoding="utf-8")
    s = _session(WebApp(), "t" * 24)
    sources = s.view("sources")
    sources.set_dataset_path(str(example / "renamed.csv"))
    assert (sources.uuid_value, sources.label_value) == ("id", "cls")
    assert sources.image_root == str(example / "imgs")
    assert "lookin.toml" in sources.config_note[0]

    sources.set_dataset_path(str(example / "dataset.csv"))   # matched by "*.csv" only
    assert sources.label_value == "label" and sources.config_note[1] == "warning"

    sources.dataset_path, sources.label_value = str(example / "renamed.csv"), "subset"
    sources.image_root = ""
    sources._sniff()                                          # as a restore does
    assert sources.label_value == "subset"                    # chosen by the workspace: kept
    assert sources.image_root == str(example / "imgs")        # empty: filled in

    (example / "lookin.toml").write_text("not = [toml", encoding="utf-8")
    sources.set_dataset_path(str(example / "renamed.csv"))
    assert sources.config_note[1] == "warning" and "not valid TOML" in sources.config_note[0]
    assert lib.Mount().uuid_col == "uuid"


def test_relabel_fields_keep_the_csv_class_names(example):
    s = _session(WebApp(), "u" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv"))])
    sources = s.view("sources")
    sources.relabel_open = True
    with RequestScope(s):
        html = render.render_view(s, "sources")
        shown = sources._relabel_shown
        i = shown.index("cat")
        sources.on_apply_relabel(**{f"r{i}": "feline"})
        html = render.render_view(s, "sources")
    assert "feline" in s.state.dataset.labels() and "cat" not in s.state.dataset.labels()
    assert sources._relabel_shown == shown                    # titled by the CSV's names still
    assert 'data-orig="cat"' in html and 'value="feline"' in html
    assert "feline" in s.state.experiments["baseline"].class_labels


def test_a_dataset_dropped_on_a_workspace_becomes_its_dataset(example, tmp_path):
    from lookin.core import library as lib
    from lookin.web.views.base import ConfirmDialog

    lib.set_library_root(tmp_path / "lib")
    try:
        empty = lib.create_workspace("empty", lib.Mount(dataset_path=""))
        used = lib.create_workspace("used", lib.Mount(dataset_path=str(example / "dataset.csv")))
        s = _session(WebApp(), "v" * 24)
        home = s.view("home")
        home.reload(wait=True)
        other = str(example / "dataset_reviewed.csv")
        with RequestScope(s):
            home.on_drop_on_ws(id=empty.id, kind="dataset", key=other)
            home.on_drop_on_ws(id=used.id, kind="dataset", key=other)
        assert lib.load_workspace(empty.path).mount.dataset_path == other
        # Replacing one asks first.
        assert lib.load_workspace(used.path).mount.dataset_path == str(example / "dataset.csv")
        assert isinstance(s.dialogs[-1], ConfirmDialog)
    finally:
        lib.set_library_root(None)


def test_a_rebuild_goes_on_when_the_tab_is_left_and_stops_on_cancel(example):
    """Leaving a tab leaves its computation running; the banner's Cancel
    stops it (at its next progress step) and then offers to run it again;
    while it runs, there is no "Update now" to queue it twice."""
    import threading
    import time

    from lookin.web.views.base import RebuildView

    app = WebApp()
    s = _session(app, "k" * 24)
    with RequestScope(s):
        render.AppActions(s).on_tab(tab="statistics")
    view = s.view("statistics")
    assert isinstance(view, RebuildView)
    started, release, steps = threading.Event(), threading.Event(), []

    def slow():
        started.set()
        release.wait(5)
        view.set_progress("step 2")        # a checkpoint: a cancelled job stops here
        steps.append("done")

    def wait_idle():
        for _ in range(200):
            if not view.running:
                return
            time.sleep(0.02)
        raise AssertionError("the job never stopped")

    view.dirty = True
    with RequestScope(s):
        view.rebuild(slow)
    assert started.wait(5)
    banner = view.stale_html()
    assert "Update now" not in banner and "Cancel" in banner      # already running
    assert "Cancel" not in view.progress_html()                   # one Cancel, in the banner

    with RequestScope(s):
        render.AppActions(s).on_tab(tab="metrics")                # leave the tab: it goes on
    release.set()
    wait_idle()
    assert steps == ["done"] and not view.dirty and not view.cancelled

    started.clear(); release.clear()
    view.dirty = True
    with RequestScope(s):
        view.rebuild(slow)
    assert started.wait(5)
    with RequestScope(s):
        view.on_cancel()
    release.set()
    wait_idle()
    assert steps == ["done"] and view.cancelled and view.dirty
    assert "resume" in view.stale_html() and "cancelled" in view.stale_html()


def test_every_tab_updates_by_itself_after_a_data_change(example):
    """Data changes while the user is elsewhere: every tab that shows
    something computed from it is marked out of date, and updates as soon
    as it is shown again -- no "Update now" to press."""
    import time

    app = WebApp()
    s = _session(app, "u" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv")),
                       ("resnet", str(example / "finetuned_resnet_results.csv"))])
    s.state.experiments["baseline"].load_embeddings(str(example / "baseline_cnn_embeddings.npz"))
    ok, message = s.prototype_source.load_prototypes_for("baseline", str(example / "baseline_cnn_prototypes.npz"))
    assert ok, message

    def idle():
        for _ in range(1500):
            if not any(j.busy for j in list(s._jobs.values())):
                return
            time.sleep(0.02)
        raise AssertionError(f"still running: {[n for n, j in s._jobs.items() if j.busy]}")

    def act(fn):
        with RequestScope(s):
            fn()
        idle()

    tabs = ["statistics", "gallery", "metrics", "calibration", "confidence", "compare",
            "prototypes", "embeddings", "cleaning"]
    # show something on every tab
    for tab in tabs:
        act(lambda: render.AppActions(s).on_tab(tab=tab))
    stats = s.view("statistics")
    act(lambda: (stats.on_choose(value=s.state.dataset.columns.label_col), stats.on_add()))
    compare = s.view("compare")
    act(lambda: (setattr(compare, "baseline_value", "baseline"), setattr(compare, "candidate_value", "resnet"),
                 compare.on_compare()))
    emb = s.view("embeddings")
    act(lambda: (setattr(emb, "experiment", "baseline"), emb.on_run()))
    cleaning = s.view("cleaning")
    act(cleaning.on_run)
    assert compare._diff is not None and emb._has_result and cleaning._candidate_set is not None

    assert "role_html" in s.view("prototypes").result         # prototypes drawn, not just "none yet"

    # a real data change: a row filter, so every cache keyed on the old rows is caught out
    act(lambda: render.AppActions(s).on_tab(tab="filter"))
    act(lambda: s.view("filter").on_query_enter(value="is_blurry == 0"))
    assert 0 < len(s.state.dataset.df) < 800
    act(lambda: render.AppActions(s).on_tab(tab="sources"))
    for tab in tabs:
        assert s.view(tab).dirty, f"{tab} was not marked out of date"
        assert "out of date" in s.view(tab).stale_html() or "previous data" in s.view(tab).stale_html(), tab

    for tab in tabs:
        act(lambda: render.AppActions(s).on_tab(tab=tab))
        view = s.view(tab)
        assert not view.dirty, f"{tab} did not update when shown"
        assert view.stale_html() == "", tab
        assert not getattr(view, "error", ""), (tab, view.error)
    assert "error" not in s.view("prototypes").result, s.view("prototypes").result.get("error")


def test_a_cancelled_update_waits_for_the_user_or_new_data(example):
    import threading
    import time

    app = WebApp()
    s = _session(app, "w" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv"))])
    stats = s.view("statistics")
    with RequestScope(s):
        render.AppActions(s).on_tab(tab="statistics")
        stats.on_choose(value=s.state.dataset.columns.label_col)
        stats.on_add()
    started, release, runs = threading.Event(), threading.Event(), []

    def slow():
        runs.append(1)
        started.set()
        release.wait(5)
        stats.set_progress("next")

    def idle():
        for _ in range(500):
            if not stats.running:
                return
            time.sleep(0.02)
        raise AssertionError("still running")

    idle()
    stats.compute = slow
    with RequestScope(s):
        render.AppActions(s).on_tab(tab="sources")
        s.notify_change()
        render.AppActions(s).on_tab(tab="statistics")        # arriving starts the update
    assert started.wait(5)
    banner = stats.stale_html()
    assert "being updated" in banner and "Cancel" in banner and "Update now" not in banner
    with RequestScope(s):
        stats.on_cancel()
    release.set()
    idle()
    assert stats.cancelled and stats.dirty and "Update now" in stats.stale_html()

    with RequestScope(s):                                     # away and back: still waits
        render.AppActions(s).on_tab(tab="sources")
        render.AppActions(s).on_tab(tab="statistics")
    assert len(runs) == 1 and not stats.running

    started.clear(); release.clear()
    with RequestScope(s):                                     # new data: updates again
        s.notify_change()
    assert started.wait(5) and len(runs) == 2 and not stats.cancelled
    release.set()
    idle()
    assert not stats.dirty


def test_a_hidden_tab_gets_its_commands_when_it_comes_back():
    """A hidden tab drops its event stream (app.js): what was pushed to it
    meanwhile is kept -- a toast, a download -- and handed to the stream it
    opens again; what only needed redrawing is not (the page redraws on
    reconnect)."""
    import asyncio

    app = WebApp()
    s = app.store.get("h" * 24, create=True)
    s.push({"dirty": ["home"]})
    s.toast("Report ready")
    s.cmd(download="/dl/x", filename="report.pptx")

    async def reconnect():
        q = asyncio.Queue()
        s.subscribe(asyncio.get_running_loop(), q)
        return [q.get_nowait() for _ in range(q.qsize())]

    got = asyncio.run(reconnect())
    assert len(got) == 2 and "Report ready" in got[0] and "report.pptx" in got[1]
    assert asyncio.run(reconnect()) == []        # handed over once


def test_a_hidden_tab_pinging_is_not_reaped_and_needs_no_lock():
    import threading
    import time

    from fastapi.testclient import TestClient

    from lookin.web.server import create_app

    app = WebApp()
    client = TestClient(create_app(app))
    s = app.store.get("p" * 24, create=True)
    s.last_seen = 0.0
    s.lock.acquire()               # an action running in that tab
    try:
        result = {}
        t = threading.Thread(target=lambda: result.setdefault(
            "r", client.post("/api/ping", json={"sid": "p" * 24})))
        t.start()
        t.join(5)
        assert result["r"].status_code == 200
    finally:
        s.lock.release()
    assert time.time() - s.last_seen < 60
    assert client.post("/api/ping", json={"sid": "q" * 24}).status_code == 410


def test_home_shows_at_once_and_fills_in_from_the_library(tmp_path):
    """Home draws from an index read off the request path: no analysis view
    is built (their imports take seconds), and a workspace saved on disk
    appears once the library is read again -- not on every redraw."""
    from lookin.core import library as lib
    from lookin.web.home_index import HomeIndex

    lib.set_library_root(tmp_path / "lib")
    try:
        first = lib.create_workspace("first", lib.Mount(dataset_path=""))
        s = _session(WebApp(), "w" * 24)
        assert not {"metrics", "embeddings", "gallery", "cleaning"} & set(s.views)
        home = s.view("home")
        home.reload(wait=True)
        with RequestScope(s):
            assert "first" in render.render_view(s, "home")
        second = lib.create_workspace("second", lib.Mount(dataset_path=""))
        with RequestScope(s):
            assert "second" not in render.render_view(s, "home")      # nothing read on a redraw
        home.reload(wait=True)
        with RequestScope(s):
            html = render.render_view(s, "home")
        assert "first" in html and "second" in html
        assert HomeIndex.get() is home.index                            # one per library
        assert {first.id, second.id} <= {w.id for w in home.workspaces}
    finally:
        lib.set_library_root(None)


def test_a_dropped_dataset_is_set_at_once_and_hashed_behind(example, tmp_path):
    import time

    from lookin.core import library as lib

    lib.set_library_root(tmp_path / "lib")
    try:
        ws = lib.create_workspace("w", lib.Mount(dataset_path=""))
        s = _session(WebApp(), "x" * 24)
        home = s.view("home")
        home.reload(wait=True)
        path = str(example / "dataset.csv")
        with RequestScope(s):
            home.on_drop_on_ws(id=ws.id, kind="dataset", key=path)
        assert lib.load_workspace(ws.path).mount.dataset_path == path
        for _ in range(200):
            if lib.load_workspace(ws.path).dataset_hash:
                break
            time.sleep(0.02)
        assert lib.load_workspace(ws.path).dataset_hash == lib.hash_file(path)
    finally:
        lib.set_library_root(None)


def test_home_redraws_at_most_once_a_second_on_its_own():
    """An MLflow listing moves several times a second: Home is marked for
    redrawing once, then once more when the second is up."""
    import time

    s = _session(WebApp(), "y" * 24)
    home = s.view("home")
    pushed = []
    s.push = lambda message: pushed.append(message)
    home._last_mark = 0.0
    for _ in range(10):
        home._hub_changed()
    assert len(pushed) == 1
    time.sleep(1.3)
    assert len(pushed) == 2


def test_a_restored_snapshot_redraws_every_restored_view(example):
    """The snapshot registry holds the views lazily (render._LazyView); a
    restore must still reach each view's recompute_after_restore, or a
    reopened snapshot shows an empty Embeddings map, Metrics, Compare..."""
    app = WebApp()
    s = _session(app, "f" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv")),
                       ("resnet", str(example / "finetuned_resnet_results.csv"))])
    captured = s.view_state.capture()

    t = _session(app, "g" * 24)
    _load(t, example, [("baseline", str(example / "baseline_cnn_results.csv")),
                       ("resnet", str(example / "finetuned_resnet_results.csv"))])
    called = []
    for name in ("embeddings", "metrics", "compare", "calibration", "cleaning"):
        t.view(name).recompute_after_restore = lambda n=name: called.append(n)
    with RequestScope(t):
        t.view_state.restore(captured)
    assert sorted(called) == ["calibration", "cleaning", "compare", "embeddings", "metrics"]


def test_embeddings_settings_cannot_fold_away_without_a_map(example):
    """The splitter folds the settings column away for a wider map; with no
    map yet, folded settings left one "Click 'Run projection'" line and no
    button to run it. The column only folds when a map is shown, and then a
    "Show settings" button brings it back."""
    s = _session(WebApp(), "s" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv"))])
    with RequestScope(s):
        s.goto("embeddings")
        html = render.render_view(s, "embeddings")
    assert "emb-has-map" not in html and "Run projection" in html
    s.view("embeddings").results_html = '<div class="pmap" data-map="k"></div>'
    with RequestScope(s):
        assert 'class="emb emb-has-map"' in render.render_view(s, "embeddings")


def test_the_embeddings_map_follows_the_display_setting(example):
    """The map is never redrawn by the page, so its crops' addresses name
    the display setting from the tab around it, and a new setting tells the
    map to fetch them again -- they used to come back from the browser's
    cache, drawn through the old LUT."""
    s = _session(WebApp(), "t" * 24)
    _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv"))])
    with RequestScope(s):
        s.goto("embeddings")
        assert 'data-display="none."' in render.render_view(s, "embeddings")
    with RequestScope(s):
        s.display_lut = "minmax"
        s.notify_display_change()
        out = render.collect(s)
    assert {"event": "display-changed"} in out.get("cmds", [])
    with RequestScope(s):
        assert 'data-display="minmax."' in render.render_view(s, "embeddings")


def test_a_workspace_duplicated_from_home_sits_below_it(tmp_path):
    from lookin.core import board as bd
    from lookin.core import library as lib

    lib.set_library_root(tmp_path / "lib")
    try:
        original = lib.create_workspace("original", lib.Mount(dataset_path=""))
        lib.create_workspace("other", lib.Mount(dataset_path=""))
        s = _session(WebApp(), "d" * 24)
        home = s.view("home")
        home.reload(wait=True)
        with RequestScope(s):
            assert "Duplicate…" in render.render_view(s, "home")
            home.on_duplicate(id=original.id)
            [dialog] = s.dialogs
            dialog.on_submit(name="the copy")
            html = render.render_view(s, "home")
        assert not s.dialogs
        copy = next(w for w in lib.list_workspaces() if w.name == "the copy")
        assert copy.id != original.id and "the copy" in html
        ids = bd.column_of(bd.load(), original.id)["workspaces"]
        assert ids[ids.index(original.id) + 1] == copy.id
    finally:
        lib.set_library_root(None)


def test_the_theme_is_set_before_the_first_paint_and_toggled_in_both_modes(example):
    from lookin.web.server import THEME_KEY, page_html

    page = page_html()
    assert page.index(THEME_KEY) < page.index("app.css")        # in <head>, before the stylesheet
    assert "http" not in page.split("<body>")[0].replace('http-equiv', '')  # nothing fetched
    s = _session(WebApp(), "t" * 24)
    with RequestScope(s):
        assert "data-theme-toggle" in render.render_top(s)
        _load(s, example, [("baseline", str(example / "baseline_cnn_results.csv"))])
        s.goto("metrics")
        assert "data-theme-toggle" in render.render_top(s)
