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
    gallery.rows_value, gallery.cols_value = "3", "5"
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
