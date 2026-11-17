"""The Decision Rule tab in a real session: it tunes on the validation rows
of the dataset CSV, applies the rule to every tab, and reverts it; past
FOCUS_MAX classes it shows the focus set."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from lookin.core import decision_rule as dr
from lookin.web import render
from lookin.web.server import WebApp
from lookin.web.session import RequestScope


def _data(tmp_path, k: int, n: int = 1500):
    rng = np.random.default_rng(0)
    labels = [f"class_{i:02d}" for i in range(k)]
    prior = np.linspace(3, 1, k)
    prior /= prior.sum()
    y = rng.choice(k, n, p=prior)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), y] += 1.5
    logits += np.log(prior)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    uuids = [f"u{i}" for i in range(n)]
    split = np.where(np.arange(n) % 3 == 0, "val", "test")
    pd.DataFrame({"uuid": uuids, "crop": [f"{u}.png" for u in uuids], "label": [labels[i] for i in y],
                  "split": split}).to_csv(tmp_path / "dataset.csv", index=False)
    results = pd.DataFrame(p, columns=labels)
    results.insert(0, "uuid", uuids)
    results.to_csv(tmp_path / "res_results.csv", index=False)


def _session(tmp_path):
    app = WebApp()
    s = app.store.get("d" * 24, create=True)
    with RequestScope(s):
        render.boot(s, {})
    sources = s.view("sources")
    sources.set_dataset_path(str(tmp_path / "dataset.csv"))
    assert sources.load_dataset_now(then_experiments=[("exp", str(tmp_path / "res_results.csv"))])
    assert s.state.dataset.row_filter == ("split", "test")
    return s


def _idle(s):
    for _ in range(3000):
        if not any(j.busy for j in list(s._jobs.values())):
            return
        time.sleep(0.02)
    raise AssertionError("still running")


def _act(s, fn):
    with RequestScope(s):
        fn()
    _idle(s)


@pytest.mark.parametrize("k", [6, 25])
def test_tune_on_validation_apply_and_revert(tmp_path, k):
    _data(tmp_path, k)
    s = _session(tmp_path)
    view = s.view("decision")
    _act(s, lambda: render.AppActions(s).on_tab(tab="decision"))
    assert not view.error, view.error
    assert view.ctx.pool.n_rows == 500 and view.ctx.check.blocking is None
    assert view.class_focus.many == (k > 20)
    _act(s, view.on_compute)
    assert view.fit is not None and view.fit.result.objective_after > view.fit.result.objective_before
    with RequestScope(s):
        html = render.render_view(s, "decision")
    assert "failed to render" not in html and "Tuning on 500 validation rows" in html
    before = s.view("decision").ctx.before.macro_f1
    _act(s, view.on_apply)
    rule = s.state.experiments["exp"].decision_rule
    assert rule["tuned_on"] == dr.POOL_VAL and rule["n_rows"] == 500
    with RequestScope(s):
        assert "Biased: exp" in render.render_top(s)
    metrics = s.view("metrics")
    _act(s, lambda: render.AppActions(s).on_tab(tab="metrics"))
    assert not metrics.error
    shown = metrics._metrics_cache["exp"].macro_f1
    assert shown == pytest.approx(view.out.after.macro_f1)
    assert shown != pytest.approx(before)
    # the workspace mount carries the rule
    mount = s.view("sources").current_mount()
    assert dr.record_to_text(rule) == mount.experiments[0].decision_rule
    _act(s, lambda: render.AppActions(s).on_tab(tab="decision"))
    _act(s, view.on_revert)
    assert s.state.experiments["exp"].decision_rule is None
    with RequestScope(s):
        assert "Biased" not in render.render_top(s)


def test_tuning_on_the_evaluated_rows_is_said_and_measured(tmp_path):
    _data(tmp_path, 5)
    s = _session(tmp_path)
    view = s.view("decision")
    _act(s, lambda: render.AppActions(s).on_tab(tab="decision"))
    _act(s, lambda: view.on_pool(value=dr.POOL_TEST))
    _act(s, view.on_compute)
    assert view.fit.cross is not None
    with RequestScope(s):
        html = render.render_view(s, "decision")
    assert "Tuning on the 1,000 evaluated rows themselves" in html and "Optimism" in html
