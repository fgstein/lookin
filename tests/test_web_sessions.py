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
        home.reload()
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
