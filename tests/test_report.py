"""The self-contained HTML report.

The property that matters most is in the name: the file must open with no
network and no sibling files, because that is the difference between an
archived result and a broken link. That is asserted directly.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from lookin.core.report import Report
from lookin.core.report_builder import build_report, default_report_name
from lookin.core.review import STATUS_EXCLUDE


# ---- the writer ---------------------------------------------------------

def test_a_report_is_a_complete_html_document():
    report = Report("Title", "subtitle")
    html = report.to_html()
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html, "styles are inlined, not linked"


def test_nan_is_spelled_out_rather_than_printed_as_nan():
    """A report is read by people who were not there when it was made."""
    html = Report.table(["m"], [[float("nan")]])
    assert "n/a" in html and "nan" not in html


def test_content_is_escaped():
    report = Report("<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in report.to_html()
    assert "&lt;script&gt;" in report.to_html()


def test_a_figure_is_embedded_not_linked():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    html = Report.figure(fig, "a line")
    assert "src=\"data:image/png;base64," in html
    assert "a line" in html


def test_figures_do_not_leak(monkeypatch):
    """pyplot keeps every figure alive until told otherwise, and a report
    can hold a dozen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    for _ in range(5):
        fig, ax = plt.subplots()
        Report.figure(fig)
    assert len(plt.get_fignums()) == before


# ---- the builder --------------------------------------------------------

def test_the_report_covers_the_loaded_experiment(loaded_state):
    report = build_report(loaded_state, ["exp"])
    html = report.to_html()
    assert "Summary" in html and "Per class" in html and "Confusion matrices" in html
    assert "exp" in html
    assert "Accuracy" in html


def test_the_report_is_self_contained(loaded_state):
    """No http(s) reference anywhere, and no relative src/href -- the file
    has to open on a machine with no network and no sibling directory."""
    html = build_report(loaded_state, ["exp"]).to_html()
    assert "http://" not in html and "https://" not in html
    external = [m for m in re.findall(r'(?:src|href)="([^"]+)"', html)
                if not m.startswith("data:")]
    assert external == [], external


def test_provenance_says_whether_reviewed_labels_were_applied(loaded_state):
    """The one setting that silently changes every figure in the report."""
    assert "Reviewed labels applied" in build_report(loaded_state, ["exp"]).to_html()

    loaded_state.review.create_version("v1")
    loaded_state.review.decide("a", STATUS_EXCLUDE)
    loaded_state.use_reviewed_labels = True
    html = build_report(loaded_state, ["exp"]).to_html()
    assert "cleaned pool" in html
    assert "v1" in html


def test_the_report_names_the_source_csv_and_the_split(loaded_state):
    html = build_report(loaded_state, ["exp"]).to_html()
    assert "dataset.csv" in html
    assert "Split" in html


def test_score_problems_are_reported_rather_than_silently_dropped(tmp_path, dataset_csv):
    """A metric that is missing because it would be meaningless has to say
    so -- otherwise the reader assumes it was forgotten."""
    from lookin.core.data_manager import AppState, ColumnConfig

    pd.DataFrame({
        "uuid": list("abcdef"),
        "cat": [0.9] * 6, "dog": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "bird": [0.1] * 6,
    }).to_csv(tmp_path / "r.csv", index=False)
    state = AppState()
    state.load_dataset(str(dataset_csv), ColumnConfig())
    state.add_experiment(str(tmp_path / "r.csv"), "raw")

    html = build_report(state, ["raw"]).to_html()
    assert "do not sum to 1" in html
    assert "constant" in html


def test_the_comparison_section_is_optional_and_directional(loaded_state, results_csv):
    loaded_state.add_experiment(str(results_csv), "exp2")
    without = build_report(loaded_state, ["exp", "exp2"]).to_html()
    assert "Compare:" not in without

    with_pair = build_report(loaded_state, ["exp", "exp2"],
                             comparison_pair=("exp", "exp2")).to_html()
    assert "Compare: exp to exp2" in with_pair
    assert "used to be right are now wrong" in with_pair


def test_a_report_with_no_experiment_is_refused(loaded_state):
    with pytest.raises(ValueError):
        build_report(loaded_state, [])


def test_the_report_can_be_written_and_read_back(loaded_state, tmp_path):
    dest = build_report(loaded_state, ["exp"]).write(tmp_path / "out.html")
    assert dest.exists()
    text = dest.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert not list(tmp_path.glob(".*tmp*")), "the temp file is replaced, not left behind"


def test_the_default_name_carries_the_dataset_and_the_date(loaded_state):
    name = default_report_name(loaded_state)
    assert name.startswith("dataset_report_") and name.endswith(".html")


def test_a_snapshot_id_is_recorded_so_the_figures_can_be_reopened(loaded_state):
    """The point of the id in a report: six months later the panels behind
    every figure are one paste away, rather than gone."""
    html = build_report(loaded_state, ["exp"], snapshot_id="lk-7f3a2b").to_html()
    assert "lk-7f3a2b" in html
    assert "Snapshot" in html


def test_a_report_without_a_snapshot_says_nothing_about_one(loaded_state):
    """Not every session came from a library workspace, and an empty 'Snapshot:'
    row would read as a broken link rather than an absent one."""
    assert "Snapshot" not in build_report(loaded_state, ["exp"]).to_html()
