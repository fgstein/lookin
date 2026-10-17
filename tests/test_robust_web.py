"""The browser interface's request handling: what a hostile page or a
malformed request can make the server do, and what a hostile label, column
or file name does to the page."""
from __future__ import annotations

import json
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lookin.web import html as h
from lookin.web import render
from lookin.web.server import WebApp, create_app
from lookin.web.session import RequestScope

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"
SID = "r" * 24


@pytest.fixture
def client():
    app = WebApp()
    c = TestClient(create_app(app), raise_server_exceptions=False)
    assert c.post("/api/boot", json={"sid": SID}).status_code == 200
    c.webapp = app
    return c


def _act(client, action="noop", target="app", **extra):
    return client.post("/api/act", json={"sid": SID, "target": target, "action": action, **extra})


# ---- the request side ----------------------------------------------------------

def test_cross_site_posts_are_refused(client):
    """A page on any other site can POST to localhost; a JSON content type
    and Fetch Metadata are what tell the page's own requests apart."""
    body = json.dumps({"sid": SID, "target": "app", "action": "home"}).encode()
    # What a <form> or a no-cors fetch can send.
    for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"):
        assert client.post("/api/act", content=body, headers={"content-type": ctype}).status_code == 403
    assert client.post("/api/boot", content=body, headers={"content-type": "text/plain"}).status_code == 403
    for site in ("cross-site", "same-site"):
        r = client.post("/api/act", content=body, headers={"content-type": "application/json", "sec-fetch-site": site})
        assert r.status_code == 403
    # The page itself, with or without Fetch Metadata.
    for headers in ({}, {"sec-fetch-site": "same-origin"}):
        r = client.post("/api/act", content=body, headers={"content-type": "application/json; charset=utf-8", **headers})
        assert r.status_code == 200
    assert client.get("/").status_code == 200


def test_malformed_bodies_are_a_400_not_a_traceback(client):
    for content in (b"{not json", b"[1, 2]", b'"x"', b"null"):
        r = client.post("/api/act", content=content, headers={"content-type": "application/json"})
        assert r.status_code == 400, content
        r = client.post("/api/boot", content=content, headers={"content-type": "application/json"})
        assert r.status_code == 400, content
    assert _act(client, "tab", args=["sources"]).status_code == 400
    assert _act(client, "tab", args="sources").status_code == 400


def test_an_infinite_or_huge_viewport_is_clamped(client):
    # Python's json reads Infinity; int(inf) is an OverflowError, not a ValueError.
    body = b'{"sid": "%s", "target": "app", "action": "noop", "vw": Infinity, "vh": 1e308}' % SID.encode()
    assert client.post("/api/act", content=body, headers={"content-type": "application/json"}).status_code == 200
    assert _act(client, vw=10 ** 9, vh=-5).status_code == 200
    session = client.webapp.store.get(SID, create=False)
    assert session.viewport == (16384, 320)


def test_only_methods_of_the_class_are_actions(client):
    """Dialogs keep the callback they were given as on_yes_fn, on_pick_fn,
    on_submit_fn...: named as an action, it would be called directly."""
    from lookin.web.views.base import ConfirmDialog

    session = client.webapp.store.get(SID, create=False)
    fired = []
    dialog = ConfirmDialog("Delete?", "Really?", lambda: fired.append(1))
    with RequestScope(session):
        session.open_dialog(dialog)
    r = _act(client, "yes_fn", target=f"dlg:{dialog.id}")
    assert r.status_code == 200 and not fired
    assert any("no action" in c.get("toast", "") for c in r.json()["cmds"])
    # A hook the session calls is not an action either.
    r = _act(client, "data_change", target="gallery")
    assert any("no action" in c.get("toast", "") for c in r.json()["cmds"])
    # The real ones still are.
    assert _act(client, "yes", target=f"dlg:{dialog.id}").status_code == 200
    assert fired == [1]
    assert not _act(client, "tab", args={"tab": "metrics"}).json()["cmds"]
    assert session.tab == "metrics"


def test_a_download_named_in_any_script(client):
    session = client.webapp.store.get(SID, create=False)
    key = session.put_download(b"{}", 'carte "日本"\r\n.json', "application/json")
    r = client.get(f"/dl/{SID}/{key}")
    assert r.status_code == 200
    header = r.headers["content-disposition"]
    assert "filename*=UTF-8''" in header and "%E6%97%A5" in header and "\n" not in header


def test_links_never_run_script():
    """An MLflow address comes from a file -- the Home settings, a workspace
    someone shared -- and a javascript: link runs inside Lookin's page."""
    from lookin.web.views.home import _mlflow_link

    for bad in ("javascript:alert(1)//", " JavaScript:alert(1)", "java\nscript:alert(1)", "\x01javascript:x",
                "data:text/html,<script>x</script>", "vbscript:x", "//evil.example/x", "mlflow.lab:5000"):
        assert h.safe_href(bad) == "", bad
        assert h.link_button("", bad, icon_name="external") == ""
        assert "href" not in _mlflow_link(bad)
    for good in ("http://mlflow.lab:5000/#/experiments/1", "https://x/y", "/?workspace=lkw-1"):
        assert h.safe_href(good) == good
        assert 'href="' in h.link_button("", good)


def test_drop_arguments_survive_quotes():
    attr = h.data_args(group='col "7" \\ <b>')
    p = _Collect()
    p.feed(f"<tbody{attr}></tbody>")
    assert json.loads(p.attrs["data-args"]) == {"group": 'col "7" \\ <b>'}


def test_snapshot_ids_stay_in_their_folder(tmp_path):
    """?snapshot= is read from the address bar; an id is a file name."""
    from lookin.core import library as lib

    lib.set_library_root(tmp_path / "lib")
    try:
        ws = lib.create_workspace("w", lib.Mount(dataset_path=""))
        outside = tmp_path / "victim.json"
        outside.write_text(json.dumps({"id": "lk-x", "name": "not a snapshot"}))
        rel = "../../../victim"
        assert (ws.snapshots_dir() / f"{rel}.json").resolve() == outside.resolve()
        assert lib.load_snapshot(ws, rel) is None
        assert lib.find_snapshot(rel) is None
        assert not lib.delete_snapshot(ws, rel)
        assert outside.is_file()
        snap = lib.Snapshot(id=rel, name="x")
        assert not lib.save_snapshot(ws, snap)
        assert json.loads(outside.read_text())["name"] == "not a snapshot"
    finally:
        lib.set_library_root(None)


def test_the_filter_query_survives_a_non_string(client, tmp_path):
    session = client.webapp.store.get(SID, create=False)
    shutil.copytree(EXAMPLE, tmp_path / "ex")
    sources = session.view("sources")
    sources.set_dataset_path(str(tmp_path / "ex" / "dataset.csv"))
    assert sources.load_dataset_now(then_experiments=[])
    for value in (None, 5, ["a"], {"k": 1}, True):
        _act(client, "query", target="filter", args={"value": value})
        _act(client, "example", target="filter", args={"text": value})
        with RequestScope(session):
            html = render.render_view(session, "filter")
        assert "failed to render" not in html, value


# ---- hostile names on every tab -------------------------------------------------

HOSTILE_LABELS = {"cat": "<xssa onmouseover=alert(1)>cat", "dog": "\"'><xssb onload=alert(2)>",
                  "bird": "`${x}`</script><xssc>", "frog": "javascript:alert(4)",
                  "deer": "red;background:url(http://evil/x)"}
HOSTILE_COLUMNS = {"quality_flag": "<xssd onmouseover=alert(5)>q", "subset": "sub\"'><xsse>"}
HOSTILE_EXPERIMENT = "<xssg>base\"'`"


class _Collect(HTMLParser):
    """Every injected element, handler attribute, script URL or unparsable
    data-args in a page."""

    def __init__(self):
        super().__init__()
        self.bad = []
        self.attrs = {}

    def handle_starttag(self, tag, attrs):
        if tag.startswith("xss"):
            self.bad.append(("element", tag))
        for k, v in attrs:
            self.attrs[k] = v
            if k.startswith("on") or k.startswith("xss"):
                self.bad.append(("attribute", tag, k, v))
            if k == "data-args":
                try:
                    json.loads(v)
                except ValueError:
                    self.bad.append(("data-args", v))
            if k in ("href", "src") and v and "script:" in v.lower():
                self.bad.append(("url", tag, v))
            if k == "style" and v and ("url(" in v or "evil" in v):
                self.bad.append(("style", tag, v))


@pytest.fixture
def hostile(tmp_path):
    dest = tmp_path / "hostile"
    shutil.copytree(EXAMPLE, dest)
    for p in dest.glob("*review*"):
        p.unlink()
    ds = pd.read_csv(dest / "dataset.csv")
    ds["label"] = ds["label"].replace(HOSTILE_LABELS)
    ds["subset"] = ds["subset"].astype(str) + "<xssf>"
    ds.rename(columns=HOSTILE_COLUMNS).to_csv(dest / "dataset.csv", index=False)
    for name in ("baseline_cnn_results.csv", "finetuned_resnet_results.csv"):
        r = pd.read_csv(dest / name)
        r["label"] = r["label"].replace(HOSTILE_LABELS)
        r.rename(columns=HOSTILE_LABELS).to_csv(dest / name, index=False)
    for name in ("baseline_cnn_prototypes.npz", "finetuned_resnet_prototypes.npz"):
        z = dict(np.load(dest / name))
        z["labels"] = np.array([HOSTILE_LABELS.get(str(lbl), str(lbl)) for lbl in z["labels"]])
        np.savez(dest / name, **z)
    classes = json.loads((dest / "classes.json").read_text())
    (dest / "classes.json").write_text(json.dumps([HOSTILE_LABELS.get(c, c) for c in classes]))
    (dest / "<xssh onerror=x>.csv").write_text("a\n")
    return dest


def _settle(session) -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        if not session.busy.running and not any(j.busy for j in session._jobs.values()):
            return
        time.sleep(0.05)


def _scan(where: str, html: str, found: list) -> None:
    p = _Collect()
    p.feed(html)
    found += [(where, *b) for b in p.bad]


def test_hostile_names_are_escaped_on_every_tab(hostile):
    session = WebApp().store.get("x" * 24)
    with RequestScope(session):
        render.boot(session, {})
    sources = session.view("sources")
    sources.set_dataset_path(str(hostile / "dataset.csv"))
    assert sources.load_dataset_now(then_experiments=[
        (HOSTILE_EXPERIMENT, str(hostile / "baseline_cnn_results.csv")),
        ("resnet", str(hostile / "finetuned_resnet_results.csv"))])
    _settle(session)
    found: list = []
    shown = 0
    for tab in render.TAB_KEYS:
        with RequestScope(session):
            session.goto(tab)
        _settle(session)
        run = getattr(session.view(tab), "on_run", None)
        if run is not None:
            with RequestScope(session):
                run()
            _settle(session)
        with RequestScope(session):
            out = render.collect(session, full=True)
        assert "failed to render" not in out["page"], tab
        shown += out["page"].count("&lt;xss")
        _scan(tab, out["page"], found)
    assert shown > 100        # the names did reach the page -- escaped

    # The dialogs that show labels, file names and experiment names.
    metrics = session.view("metrics")
    with RequestScope(session):
        metrics.on_cell(experiment=HOSTILE_EXPERIMENT, true=HOSTILE_LABELS["cat"], pred=HOSTILE_LABELS["dog"])
        session.view("gallery").on_open(i=0)
        from lookin.web.views.files import FileBrowserDialog
        session.open_dialog(FileBrowserDialog(lambda p: None, start_path=str(hostile)))
        session.open_dialog(__import__("lookin.web.views.workspaces", fromlist=["x"]).WorkspacesDialog(session))
        dialogs = render.render_dialogs(session)
    assert dialogs.count("&lt;xss") >= 3
    _scan("dialogs", dialogs, found)
    with RequestScope(session):
        session.mode = "home"
        _scan("home", render.collect(session, full=True)["page"], found)
    assert not found, found[:10]
