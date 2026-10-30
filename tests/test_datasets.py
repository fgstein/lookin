"""The dataset manager's core: the shared folder as the catalogue, the
content hash the dataset scripts log, profiles, the link to MLflow runs,
comparison, and the MCP answers built on them.

Everything runs on a tmp_path shared folder; config and cache go to
tmp_path too (XDG_CONFIG_HOME / XDG_CACHE_HOME), and no server is reached.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from lookin.core.datasets import agent, catalog, compare, config as dcfg, fingerprint, mlflow_link, profile
from lookin.core.datasets.index import DatasetIndex

CLASSES = ["cat", "dog", "bird", "None"]   # "None" is a class name, not a missing value


def _frame(n=120, seed=0, classes=CLASSES, sources=("cam1", "cam2"), splits=("train", "test")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "uuid": [f"{i:08x}-0000-4000-8000-{i:012x}" for i in range(n)],
        "crop": [f"crops/{i}.png" for i in range(n)],
        "label": [classes[i % len(classes)] for i in range(n)],
        "source": [sources[i % len(sources)] for i in range(n)],
        "split": [splits[(i // 7) % len(splits)] for i in range(n)],
        "brightness": rng.random(n),
    })


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    DatasetIndex.reset()
    yield tmp_path
    DatasetIndex.reset()


@pytest.fixture
def shared(env):
    """shared/A.csv, its derivative shared/A/A_clean.csv, and shared/B.csv."""
    root = env / "shared"
    (root / "A").mkdir(parents=True)
    (root / "A" / "crops").mkdir()
    a = _frame()
    a.to_csv(root / "A.csv", index=False)
    (root / "crops").mkdir()
    for c in a["crop"]:
        (root / c).write_bytes(b"")
    a[a["label"] != "bird"].to_csv(root / "A" / "A_clean.csv", index=False)
    _frame(60, seed=1, classes=["cat", "dog"]).to_csv(root / "B.csv", index=False)
    (root / "x_results.csv").write_text("uuid,cat\n")       # an experiment, not a dataset
    dcfg.add_folder(str(root))
    return root


# ---- the hash the scripts log ----------------------------------------------

def _scripts_hash(df: pd.DataFrame) -> str:
    """The function the dataset scripts use, as they wrote it."""
    df = df[sorted(df.columns)]
    row_hashes = pd.util.hash_pandas_object(df, index=False).sort_values()
    digest = hashlib.sha256()
    digest.update(",".join(sorted(df.columns)).encode("utf-8"))
    digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def test_the_hash_is_the_scripts_hash_and_ignores_order(tmp_path):
    df = _frame()
    df.to_csv(tmp_path / "a.csv", index=False)
    shuffled = df.sample(frac=1, random_state=3)[list(reversed(df.columns))]
    shuffled.to_csv(tmp_path / "b.csv", index=False)
    h = fingerprint.dataset_hash(fingerprint.read_for_hash(tmp_path / "a.csv"))
    assert h == _scripts_hash(pd.read_csv(tmp_path / "a.csv"))
    assert h == fingerprint.dataset_hash(fingerprint.read_for_hash(tmp_path / "b.csv"))


def test_a_hash_file_is_read_whatever_surrounds_the_hash():
    h = "a" * 64
    assert fingerprint.parse_hash_file(h + "\n") == h
    assert fingerprint.parse_hash_file(f"{h.upper()}  dataset.csv") == h
    assert fingerprint.parse_hash_file("nothing here") == ""


# ---- the catalogue -----------------------------------------------------------

def test_a_csv_in_the_folder_named_after_another_is_its_derivative(shared):
    files = {f.name: f for f in catalog.scan([str(shared)])}
    assert set(files) == {"A.csv", "A_clean.csv", "B.csv"}
    assert files["A_clean.csv"].parent == files["A.csv"].path
    assert files["A.csv"].children == [files["A_clean.csv"].path]
    assert files["B.csv"].parent is None


def test_a_missing_folder_gives_nothing_rather_than_an_error(tmp_path):
    assert catalog.scan([str(tmp_path / "gone")]) == []
    assert catalog.missing_folders([str(tmp_path / "gone")]) == [str(tmp_path / "gone")]


def test_the_config_keeps_folders_and_experiments_once(env):
    assert dcfg.add_folder("/data/shared")
    assert not dcfg.add_folder("/data/shared")
    assert dcfg.add_mlflow("http://mlflow:5000/", "datasets")
    assert not dcfg.add_mlflow("http://mlflow:5000", "datasets")
    cfg = dcfg.load()
    assert cfg.folders == ["/data/shared"]
    assert [m.key() for m in cfg.mlflow] == [("http://mlflow:5000", "datasets")]
    dcfg.remove_mlflow("http://mlflow:5000", "datasets")
    assert dcfg.load().mlflow == []


# ---- the profile -------------------------------------------------------------

def test_the_profile_counts_classes_sources_and_splits(shared):
    p = profile.compute(shared / "A.csv")
    assert p["rows"] == 120
    assert p["roles"]["source"] == "source" and p["roles"]["split"] == "split"
    counts = dict(map(tuple, p["classes"]["counts"]))
    assert counts == {"cat": 30, "dog": 30, "bird": 30, "None": 30}      # "None" kept as a class
    assert p["classes"]["missing"] == 0 and p["classes"]["imbalance"] == 1.0
    assert dict(map(tuple, p["sources"]["counts"])) == {"cam1": 60, "cam2": 60}
    table = p["class_by_source"]
    assert sum(map(sum, table["counts"])) == 120
    assert p["quality"]["crops_checked"] > 0 and p["quality"]["crops_missing"] == 0
    assert p["hash"] == _scripts_hash(pd.read_csv(shared / "A.csv"))


def test_the_issues_say_what_is_wrong(env):
    df = _frame(40)
    df.loc[0, "uuid"] = df.loc[1, "uuid"]                # a duplicated id
    df.loc[2, "label"] = ""                              # a missing label
    df["split"] = ["train"] * 40
    df.loc[df["label"] == "cat", "split"] = "test"       # only cats in test
    path = env / "bad.csv"
    df.to_csv(path, index=False)
    p = profile.compute(path)
    codes = {i["code"] for i in p["issues"]}
    assert {"duplicate_ids", "missing_labels", "crops_missing", "class_absent_from_split"} <= codes
    assert profile.health(p) == profile.SEVERITY_ERROR
    assert p["issues"][0]["severity"] == profile.SEVERITY_ERROR


def test_lookin_toml_names_the_split_and_source_columns(env):
    df = _frame(20).rename(columns={"source": "camera", "split": "fold"})
    df.to_csv(env / "d.csv", index=False)
    (env / "lookin.toml").write_text('["d.csv"]\nsource = "camera"\nsplit = "fold"\n')
    roles = profile.resolve_roles(env / "d.csv")
    assert (roles.source, roles.split) == ("camera", "fold")


# ---- MLflow ------------------------------------------------------------------

def _run(run_id, csv="A.csv", size=None, hash_=""):
    return mlflow_link.DatasetRun(uri="http://m", experiment="datasets", experiment_id="1", run_id=run_id,
                                  run_name=run_id, csv_path=f"data/{csv}", csv_size=size, hash=hash_)


def test_a_file_is_confirmed_by_hash_and_probable_by_name_and_size():
    index = {("http://m", "datasets"): mlflow_link.SourceState(
        "http://m", "datasets", runs=[_run("r1", hash_="f" * 64), _run("r2", csv="B.csv", size=10)])}
    assert mlflow_link.match(index, "anything.csv", 1, "f" * 64).status == mlflow_link.LINK_CONFIRMED
    probable = mlflow_link.match(index, "B.csv", 10, "e" * 64)
    assert probable.status == mlflow_link.LINK_PROBABLE and probable.run.run_id == "r2"
    assert mlflow_link.match(index, "B.csv", 11, "").status == mlflow_link.LINK_NONE


def test_a_derivative_inherits_its_main_datasets_run(shared):
    index = DatasetIndex.get()
    index.rescan()
    main = str((shared / "A.csv").resolve())
    derived = str((shared / "A" / "A_clean.csv").resolve())
    p = index.compute(main)
    dcfg.add_mlflow("http://m", "datasets")
    index.reload_config()
    index.mlflow = {("http://m", "datasets"): mlflow_link.SourceState(
        "http://m", "datasets", runs=[_run("r1", hash_=p["hash"])])}
    assert index.link(main).status == mlflow_link.LINK_CONFIRMED
    link = index.link(derived)
    assert link.status == mlflow_link.LINK_DERIVED and link.run.run_id == "r1"


# ---- the index -----------------------------------------------------------------

def test_a_profile_is_cached_until_the_file_changes(shared):
    index = DatasetIndex.get()
    index.rescan()
    path = str((shared / "B.csv").resolve())
    assert index.profile_of(path) is None
    first = index.compute(path)
    assert index.profile_of(path) is first
    DatasetIndex.reset()                                   # a restart: read back from disk
    again = DatasetIndex.get()
    again.rescan()
    assert again.profile_of(path)["hash"] == first["hash"]
    _frame(61, seed=1, classes=["cat", "dog"]).to_csv(path, index=False)
    again.rescan()
    assert again.profile_of(path) is None


def test_the_background_queue_profiles_every_file(shared):
    index = DatasetIndex.get()
    index.request_all()
    import time
    deadline = time.time() + 30
    while index.pending() and time.time() < deadline:
        time.sleep(0.05)
    assert all(index.profile_of(f.path) is not None for f in index.files)


# ---- comparison ----------------------------------------------------------------

def test_comparing_a_derivative_with_its_main_dataset(shared):
    index = DatasetIndex.get()
    index.rescan()
    result = index.compare(str((shared / "A.csv").resolve()), str((shared / "A" / "A_clean.csv").resolve()))
    assert result["classes"]["only_a"] == ["bird"]
    assert result["overlap"]["common"] == 90 and result["overlap"]["only_a"] == 30
    assert result["overlap"]["label_changes"] == 0
    assert 0 < result["classes"]["js_divergence"] <= 1


def test_js_divergence_bounds():
    assert compare.js_divergence({"a": 5}, {"a": 9}) == 0.0
    assert compare.js_divergence({"a": 5}, {"b": 9}) == pytest.approx(1.0)


# ---- the MCP answers -------------------------------------------------------------

def test_the_agent_finds_datasets_by_name_and_answers_in_plain_json(shared):
    listing = agent.list_datasets()
    assert listing["count"] == 3
    clean = next(d for d in listing["datasets"] if d["name"] == "A_clean.csv")
    assert clean["derived_from"] == "shared/A.csv"
    stats = agent.get_dataset_stats("A", sections=["classes", "class_by_split"], top=2)
    assert stats["classes"]["count"] == 4 and len(stats["classes"]["largest"]) == 2
    assert stats["class_by_split"]["columns"]
    diff = agent.compare_datasets("A.csv", "shared/A/A_clean.csv")
    assert diff["classes"]["only_in_a"] == ["bird"]
    rows = agent.get_dataset_rows("B", label="dog", limit=3)
    assert rows["matching"] == 30 and len(rows["rows"]) == 3
    with pytest.raises(ValueError, match="No dataset"):
        agent.get_dataset("nope")
