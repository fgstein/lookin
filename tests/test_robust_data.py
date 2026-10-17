"""Unusual but legitimate data: a missing score, a missing label, a
metadata column named like a class, a run that matches no row, a map of a
handful of points. Each used to raise or, worse, give a wrong number."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.data_manager import AppState, ColumnConfig
from lookin.core.metrics import compute_metrics, diagnose_scores


def _state(tmp_path, dataset: pd.DataFrame, results: pd.DataFrame) -> AppState:
    dataset.to_csv(tmp_path / "dataset.csv", index=False)
    results.to_csv(tmp_path / "run_results.csv", index=False)
    state = AppState()
    state.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    state.add_experiment(str(tmp_path / "run_results.csv"), "run")
    return state


def _dataset(labels) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame({"uuid": [f"u{i}" for i in range(n)], "crop": ["c.png"] * n, "label": labels,
                         "site": [["n", "s"][i % 2] for i in range(n)]})


def test_a_missing_score_is_never_the_prediction(tmp_path):
    """np.argmax returns the first NaN: a crop scored 0.9 for "b" and
    nothing for "a" was predicted "a" -- and counted right."""
    results = pd.DataFrame({"uuid": ["u0", "u1", "u2"], "a": [np.nan, 0.8, np.nan],
                            "b": [0.9, 0.2, np.nan]})
    state = _state(tmp_path, _dataset(["a", "a", "b"]), results)
    merged = state.merged("run").set_index("uuid")
    assert merged.loc["u0", "predicted_label"] == "b"
    assert merged.loc["u0", "predicted_score"] == pytest.approx(0.9)
    assert not merged.loc["u0", "correct"]
    assert merged.loc["u1", "predicted_label"] == "a"
    # No score at all: no prediction, not the first class.
    assert pd.isna(merged.loc["u2", "predicted_label"])
    assert np.isnan(merged.loc["u2", "predicted_score"])


def test_a_row_with_a_missing_score_is_not_a_distribution():
    scores = np.array([[0.4, 0.6], [np.nan, 0.9]])
    diagnostics = diagnose_scores(scores, ["a", "b"])
    assert not diagnostics.normalized
    assert diagnostics.n_rows_not_normalized == 1


def test_a_metadata_column_named_like_a_class_keeps_the_scores(tmp_path):
    """A dataset column "b" next to the score column "b": the merge kept
    the dataset's column under that name, and every metric read the
    metadata as the class's score."""
    dataset = _dataset(["a", "b", "a", "b"]).assign(b=[0.0, 0.0, 1.0, 1.0])
    results = pd.DataFrame({"uuid": ["u0", "u1", "u2", "u3"], "a": [0.9, 0.1, 0.2, 0.3],
                            "b": [0.1, 0.9, 0.8, 0.7]})
    state = _state(tmp_path, dataset, results)
    merged = state.merged("run")
    assert merged["b"].tolist() == pytest.approx([0.1, 0.9, 0.8, 0.7])
    assert merged["predicted_label"].tolist() == ["a", "b", "b", "b"]
    metrics = compute_metrics("run", merged, "label", ["a", "b"], ["a", "b"])
    assert metrics.accuracy == pytest.approx(0.75)


def test_rows_with_no_label_are_left_out_of_the_metrics(tmp_path):
    """A crop with no label is neither right nor wrong: it counted as an
    error in accuracy and precision, and its NaN broke every place that
    sorts the classes it sees (slices, the Embeddings map, prototypes)."""
    from lookin.core import slices as sl
    from lookin.ui.embedding_context import build_analysis_context

    dataset = _dataset(["a", None, "b", "b"])
    results = pd.DataFrame({"uuid": ["u0", "u1", "u2", "u3"], "a": [0.9, 0.9, 0.2, 0.3],
                            "b": [0.1, 0.1, 0.8, 0.7]})
    state = _state(tmp_path, dataset, results)
    merged = state.merged("run")
    assert "u1" not in set(merged["uuid"])
    metrics = compute_metrics("run", merged, "label", ["a", "b"], ["a", "b"])
    assert metrics.n_samples == 3
    assert metrics.accuracy == pytest.approx(1.0)
    per_class = metrics.per_class.set_index("label")
    assert per_class.loc["a", "precision"] == pytest.approx(1.0)

    report = sl.slice_report("run", merged, "label", sl.build_spec(merged["site"]))
    assert report.overall.n == 3

    np.savez(tmp_path / "emb.npz", uuid=np.array(["u0", "u1", "u2", "u3"]),
             embeddings=np.eye(4, dtype=np.float32))
    state.experiments["run"].load_embeddings(str(tmp_path / "emb.npz"))
    context = build_analysis_context(state, "run")
    assert sorted(set(context["labels"].tolist())) == ["a", "b"]


def test_report_of_a_run_that_matches_no_row(tmp_path):
    """Drawing the confusion matrix of an empty run raised, and the whole
    report export failed."""
    from lookin.core.report_builder import build_report

    results = pd.DataFrame({"uuid": ["x0", "x1"], "a": [0.9, 0.1], "b": [0.1, 0.9]})
    state = _state(tmp_path, _dataset(["a", "b"]), results)
    report = build_report(state, ["run"])
    assert report is not None


@pytest.mark.parametrize("method", ["pca", "tsne", "umap", "umap_circle"])
@pytest.mark.parametrize("n,dim", [(1, 8), (2, 8), (3, 8), (4, 8), (6, 1)])
def test_a_map_of_a_handful_of_points(method, n, dim):
    """The Filter tab can leave three crops; every method raised on a few
    points (t-SNE below six, UMAP below four, PCA in one dimension)."""
    from lookin.core.embedding_analysis import fit_projection

    points = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    fit = fit_projection(points, method=method)
    assert fit.points.shape == (n, 2)
    assert np.isfinite(fit.points).all()


def test_a_class_named_none_or_na_is_a_label(tmp_path):
    """pandas reads "None", "NA", "null"... as missing: every crop of a
    "None" background class lost its label (and with it every metric), and
    the training export wrote those labels back empty."""
    from lookin.core.review import export_enriched_dataset

    dataset = _dataset(["None", "NA", "cat", ""]).assign(note=["NA", "x", "y", "z"])
    results = pd.DataFrame({"uuid": ["u0", "u1", "u2", "u3"], "None": [0.8, 0.1, 0.1, 0.3],
                            "NA": [0.1, 0.8, 0.1, 0.3], "cat": [0.1, 0.1, 0.8, 0.4]})
    state = _state(tmp_path, dataset, results)
    assert state.dataset.df["label"].tolist()[:3] == ["None", "NA", "cat"]
    assert pd.isna(state.dataset.df["label"].iloc[3])       # an empty cell is still missing
    assert pd.isna(state.dataset.df["note"].iloc[0])        # other columns: pandas' defaults
    merged = state.merged("run")
    assert merged["correct"].tolist() == [True, True, True]

    out = export_enriched_dataset(state.review, tmp_path / "dataset.csv", "uuid", "label", [],
                                  tmp_path / "out" / "export.csv")
    exported = pd.read_csv(out.path, keep_default_na=False)
    assert exported["label"].tolist() == ["None", "NA", "cat", ""]


def test_statistics_chart_of_a_column_holding_inf(tmp_path):
    """numpy refuses to bin an inf: the histogram raised, and the whole
    Statistics tab showed an error instead of any of its charts."""
    from lookin.web import render
    from lookin.web.server import WebApp
    from lookin.web.session import RequestScope

    n = 30
    _dataset(["a", "b"] * (n // 2)).assign(
        size=[np.inf if i == 3 else float(i) for i in range(n)]).to_csv(tmp_path / "dataset.csv", index=False)
    session = WebApp().store.get("r" * 24, create=True)
    with RequestScope(session):
        render.boot(session, {})
    sources = session.view("sources")
    sources.set_dataset_path(str(tmp_path / "dataset.csv"))
    assert sources.load_dataset_now()
    stats = session.view("statistics")
    stats.active_columns = ["size"]
    stats.compute()
    assert "size" in stats.charts
    assert any("infinite" in note for note in stats.charts["size"][1])


def test_cleaning_candidates_with_missing_scores_and_labels(tmp_path):
    """The cleaning strategies read the scores themselves: a missing score
    was the suggestion (np.argmax picks NaN), a NaN margin passed the
    minimum-margin test, and a crop with no label was a consensus error."""
    from lookin.core import candidates as cand

    dataset = _dataset(["a", "a", None, "b"])
    results = pd.DataFrame({"uuid": ["u0", "u1", "u2", "u3"], "a": [np.nan, 0.1, 0.1, 0.5],
                            "b": [0.9, 0.9, 0.9, np.nan]})
    state = _state(tmp_path, dataset, results)
    state.add_experiment(str(tmp_path / "run_results.csv"), "run2")
    exp = state.experiments["run"]
    df = state.dataset.df

    margin = {c.uuid: c for c in cand.by_margin(df, exp, "uuid", "label", min_margin=0.0).candidates}
    # u0 and u3 have no score for their own class: no margin to rank by.
    assert set(margin) == {"u1"}
    assert margin["u1"].suggestion == "b"

    errors = {c.uuid: c for c in cand.all_errors(df, exp, "uuid", "label").candidates}
    # u0 scored only "b": predicted "b", wrong -- it was predicted "a", its
    # missing score, and counted right.
    assert errors["u0"].suggestion == "b"
    consensus = {c.uuid for c in cand.by_consensus(df, [exp, state.experiments["run2"]], "uuid", "label").candidates}
    assert "u2" not in consensus
