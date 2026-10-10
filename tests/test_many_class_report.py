"""A report over thousands of classes (core/report_builder.py): the
per-class sections are cut to the focus set, so the HTML stays readable
and the slide deck stays a deck."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pptx import Presentation

from lookin.core import class_focus as cf
from lookin.core.data_manager import AppState, ColumnConfig
from lookin.core.report_builder import build_report
from lookin.core.report_pptx import PptxReport

N_CLASSES = 300


@pytest.fixture
def many_class_state(tmp_path) -> AppState:
    rng = np.random.default_rng(0)
    labels = [f"species_{i:04d}" for i in range(N_CLASSES)]
    true_idx = np.repeat(np.arange(N_CLASSES), 4)
    uuids = [f"u{i}" for i in range(len(true_idx))]
    pd.DataFrame({"uuid": uuids, "crop": "x.png", "label": [labels[i] for i in true_idx]}).to_csv(
        tmp_path / "dataset.csv", index=False)
    pred = np.where(rng.random(len(true_idx)) < 0.2, true_idx ^ 1, true_idx)
    for name in ("base", "cand"):
        scores = np.full((len(true_idx), N_CLASSES), 0.0)
        scores[np.arange(len(true_idx)), pred] = 1.0
        df = pd.DataFrame(scores, columns=labels)
        df.insert(0, "uuid", uuids)
        df.to_csv(tmp_path / f"{name}.csv", index=False)
        pred = np.where(rng.random(len(true_idx)) < 0.5, true_idx, pred)
    state = AppState()
    state.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    state.add_experiment(str(tmp_path / "base.csv"), "base")
    state.add_experiment(str(tmp_path / "cand.csv"), "cand")
    return state


def test_html_report_keeps_to_the_focus_classes(many_class_state):
    focus = ["species_0010", "species_0011", "species_0200"]
    html = build_report(many_class_state, ["base", "cand"], comparison_pair=("base", "cand"),
                        focus=focus).to_html()
    assert f"3 of {N_CLASSES:,} classes" in html
    assert "Most confused pairs" in html
    # A class outside the focus set gets no per-class row (the pairs table
    # after it covers every class on purpose).
    per_class = html.split("Per class", 1)[1].split("Confusion matrices", 1)[0]
    assert "<td>species_0010</td>" in per_class
    assert "<td>species_0100</td>" not in per_class


def test_without_a_focus_the_most_confused_classes_are_used(many_class_state):
    html = build_report(many_class_state, ["base"]).to_html()
    assert f"{cf.FOCUS_MAX} of {N_CLASSES:,} classes" in html


def test_pptx_report_stays_a_deck(tmp_path, many_class_state):
    report = build_report(many_class_state, ["base", "cand"], comparison_pair=("base", "cand"),
                          report_cls=PptxReport)
    path = tmp_path / "r.pptx"
    report.write(path)
    assert len(Presentation(str(path)).slides) < 40
