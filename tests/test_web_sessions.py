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
    over is ignored."""
    from lookin.web.views.base import NO_RENDER

    s = _session(WebApp(), "h" * 24)
    _load(s, example, [])
    gallery = s.view("gallery")
    gallery._refresh_now()
    first = len(gallery._records)
    assert 0 < first < 800
    with RequestScope(s):
        assert gallery.on_more(gen=gallery.gen, loaded=first) is NO_RENDER
        cmd = s._pending_cmds[-1]["append"]
    assert cmd["start"] == first and len(gallery._records) > first
    assert cmd["html"].count('class="gcell') == len(gallery._records) - first
    stale = gallery.gen
    gallery._set_shuffle(7)
    with RequestScope(s):
        before = len(gallery._records)
        assert gallery.on_more(gen=stale, loaded=before) is NO_RENDER
    assert len(gallery._records) == before
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
