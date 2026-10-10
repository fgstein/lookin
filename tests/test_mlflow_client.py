"""Tests for the artifact walk in core/mlflow_client.

No server and no httpx: the walk's shape (how it paginates the tree and how
it budgets what it keeps) is the part that has bitten us, and that is
decided entirely by _walk_artifacts' own loop -- so the two collaborators it
has, the client factory and the request, are stubbed out.
"""
from __future__ import annotations

import contextlib

import pytest

from lookin.core import mlflow_client as mlf


@pytest.fixture
def fake_tree(monkeypatch):
    """Install an artifact tree ({dir path -> [(name, is_dir)]}) that
    _walk_artifacts will see, and return the list of listed paths so a test
    can assert on how much of the tree was walked."""
    listed: list[str] = []

    def install(tree: dict):
        @contextlib.contextmanager
        def fake_client(*args, **kwargs):
            yield object()

        def fake_request(client, method, path, **kwargs):
            rel = kwargs.get("params", {}).get("path", "")
            listed.append(rel)
            files = []
            for name, is_dir in tree.get(rel, []):
                full = f"{rel}/{name}" if rel else name
                files.append({"path": full, "is_dir": is_dir, "file_size": None if is_dir else 10})
            return {"files": files}

        monkeypatch.setattr(mlf, "_client", fake_client)
        monkeypatch.setattr(mlf, "_request", fake_request)
        return listed

    return install


def test_the_walk_descends_into_subdirectories(fake_tree):
    fake_tree({
        "": [("outputs", True), ("readme.txt", False)],
        "outputs": [("run_results.csv", False), ("run_embeddings.npz", False)],
    })
    csvs, npzs = mlf.list_csv_and_npz_artifacts("http://server", "r1")
    assert [a.path for a in csvs] == ["outputs/run_results.csv"]
    assert [a.path for a in npzs] == ["outputs/run_embeddings.npz"]


def test_a_flood_of_npz_does_not_hide_the_csv(fake_tree):
    """The two suffixes get separate budgets. They used to share one, so a
    run logging more than the cap's worth of .npz at the top level stopped
    the walk before it ever descended to the directory holding the results
    CSV -- silently, with an empty artifact list as the only symptom."""
    flood = [(f"checkpoint_{i:04d}.npz", False) for i in range(mlf.MAX_FILES_PER_SUFFIX * 2)]
    fake_tree({
        "": flood + [("outputs", True)],
        "outputs": [("run_results.csv", False)],
    })
    csvs, npzs = mlf.list_csv_and_npz_artifacts("http://server", "r1")
    assert [a.path for a in csvs] == ["outputs/run_results.csv"]
    assert len(npzs) == mlf.MAX_FILES_PER_SUFFIX


def test_the_walk_stops_once_every_suffix_is_full(fake_tree):
    """...and only then: the cap is there so a pathological tree can't hang
    the dialog, not to cut the walk short while something is still missing."""
    per_dir = mlf.MAX_FILES_PER_SUFFIX
    tree = {"": [("a", True), ("b", True), ("c", True)]}
    for d in ("a", "b", "c"):
        tree[d] = [(f"{d}{i}.csv", False) for i in range(per_dir)] + \
                  [(f"{d}{i}.npz", False) for i in range(per_dir)]
    listed = fake_tree(tree)

    csvs, npzs = mlf.list_csv_and_npz_artifacts("http://server", "r1")
    assert len(csvs) == per_dir and len(npzs) == per_dir
    # root + the first subdirectory filled both budgets; the other two are
    # never listed.
    assert listed == ["", "a"]


def test_search_runs_follows_pages_and_sends_the_filter(monkeypatch):
    bodies = []

    @contextlib.contextmanager
    def fake_client(*args, **kwargs):
        yield object()

    def fake_request(client, method, path, **kwargs):
        body = kwargs["json"]
        bodies.append(body)
        page = int(body.get("page_token", 0))
        run = {"info": {"run_id": f"r{page}", "experiment_id": "7", "status": "FINISHED"}, "data": {}}
        return {"runs": [run], **({"next_page_token": str(page + 1)} if page < 2 else {})}

    monkeypatch.setattr(mlf, "_client", fake_client)
    monkeypatch.setattr(mlf, "_request", fake_request)
    runs, truncated = mlf.search_runs("http://x", ["7", 8], "metrics.acc > 0.9")
    assert [r.run_id for r in runs] == ["r0", "r1", "r2"]
    assert runs[0].experiment_id == "7"
    assert not truncated
    assert bodies[0]["filter"] == "metrics.acc > 0.9"
    assert bodies[0]["experiment_ids"] == ["7", "8"]

    bodies.clear()
    runs, truncated = mlf.search_runs("http://x", ["7"], max_pages=2)
    assert len(runs) == 2 and truncated
    assert "filter" not in bodies[0]


def test_search_runs_with_no_experiment_asks_nothing(monkeypatch):
    monkeypatch.setattr(mlf, "_client", None)   # would fail if called
    assert mlf.search_runs("http://x", []) == ([], False)
