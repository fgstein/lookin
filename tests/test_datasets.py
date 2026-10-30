"""The dataset manager's core: the shared folder as the catalogue, the
content hash the dataset scripts log, the columns of the settings, profiles
per mapping, the link to MLflow runs, comparison, and the MCP answers built
on them.

Everything runs on a tmp_path shared folder; config and cache go to
tmp_path too (XDG_CONFIG_HOME / XDG_CACHE_HOME), and no server is reached.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import pytest

from lookin.core.datasets import agent, catalog, compare, config as dcfg, fingerprint, mlflow_link, profile
from lookin.core.datasets.index import DatasetIndex

CLASSES = ["cat", "dog", "bird", "None"]   # "None" is a class name, not a missing value
FINE = "mapped_class_label"                # the default mapping of the settings
COARSE = "coarse_label"


def _frame(n=120, seed=0, classes=CLASSES, sources=("cam1", "cam2"), splits=("train", "test")):
    rng = np.random.default_rng(seed)
    fine = [classes[i % len(classes)] for i in range(n)]
    return pd.DataFrame({
        "uuid": [f"{i:08x}-0000-4000-8000-{i:012x}" for i in range(n)],
        "crop": [f"crops/{i}.png" for i in range(n)],
        FINE: fine,
        # a second labelling, of part of the rows only
        COARSE: ["animal" if c in ("cat", "dog") else "" for c in fine],
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
    """shared/A.csv, its derivative shared/A/A_clean.csv, shared/B.csv, and
    a label mapping table that is not a dataset."""
    root = env / "shared"
    (root / "A" / "mappings").mkdir(parents=True)
    a = _frame()
    a.to_csv(root / "A.csv", index=False)
    a[a[FINE] != "bird"].to_csv(root / "A" / "A_clean.csv", index=False)
    _frame(60, seed=1, classes=["cat", "dog"]).to_csv(root / "B.csv", index=False)
    (root / "A" / "mappings" / "A_mapping.csv").write_text("fine,coarse\ncat,animal\n")
    (root / "x_results.csv").write_text("uuid,cat\n")       # an experiment, not a dataset
    dcfg.add_folder(str(root))
    dcfg.set_columns({**dcfg.DEFAULT_COLUMNS, dcfg.ROLE_MAPPINGS: [FINE, COARSE]})
    return root


def _index(root=None):
    index = DatasetIndex.get()
    index.rescan()
    return index


def _p(root, rel):
    return str((root / rel).resolve())


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
    files = {f.name: f for f in catalog.scan([str(shared)], dcfg.load().excluded)}
    assert set(files) == {"A.csv", "A_clean.csv", "B.csv"}      # no mapping table, no results CSV
    assert files["A_clean.csv"].parent == files["A.csv"].path
    assert files["A.csv"].children == [files["A_clean.csv"].path]
    assert files["B.csv"].parent is None
    assert files["A.csv"].group() == "shared" and files["A_clean.csv"].group() == "shared/A"


def test_the_exclusion_patterns_decide_what_is_a_dataset(shared):
    everything = {f.name for f in catalog.scan([str(shared)])}
    assert "A_mapping.csv" in everything
    dcfg.set_exclude(["*/mappings/*", "B.csv"])
    kept = {f.name for f in catalog.scan([str(shared)], dcfg.load().excluded)}
    assert kept == {"A.csv", "A_clean.csv"}


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
    assert cfg.columns[dcfg.ROLE_MAPPINGS] == ["mapped_class_label"]
    dcfg.remove_mlflow("http://mlflow:5000", "datasets")
    assert dcfg.load().mlflow == []


def test_an_override_replaces_only_the_keys_it_names(env):
    dcfg.set_columns({**dcfg.DEFAULT_COLUMNS, dcfg.ROLE_SOURCE: "camera"})
    dcfg.set_override("/x/old.csv", {dcfg.ROLE_MAPPINGS: ["label"]})
    cfg = dcfg.load()
    old = cfg.columns_for("/x/old.csv")
    assert old[dcfg.ROLE_MAPPINGS] == ["label"] and old[dcfg.ROLE_SOURCE] == "camera"
    assert cfg.columns_for("/x/other.csv")[dcfg.ROLE_MAPPINGS] == ["mapped_class_label"]
    dcfg.set_override("/x/old.csv", {})
    assert "/x/old.csv" not in dcfg.load().overrides


# ---- the profile -------------------------------------------------------------

def _columns(*mappings):
    return {**dcfg.DEFAULT_COLUMNS, dcfg.ROLE_MAPPINGS: list(mappings)}


def test_the_profile_has_the_instances_and_one_entry_per_mapping(shared):
    p = profile.compute(shared / "A.csv", _columns(FINE, COARSE, "absent_label"))
    assert p["rows"] == 120
    assert p["roles"][dcfg.ROLE_MAPPINGS] == [FINE, COARSE]
    assert p["roles"]["missing"] == ["absent_label"]
    fine = p["mappings"][FINE]["classes"]
    assert dict(map(tuple, fine["counts"])) == {"cat": 30, "dog": 30, "bird": 30, "None": 30}  # "None" kept
    coarse = p["mappings"][COARSE]["classes"]
    assert dict(map(tuple, coarse["counts"])) == {"animal": 60} and coarse["missing"] == 60
    assert {i["code"] for i in p["mappings"][COARSE]["issues"]} >= {"unlabelled_rows"}
    assert dict(map(tuple, p["sources"]["counts"])) == {"cam1": 60, "cam2": 60}
    assert sum(map(sum, p["mappings"][FINE]["class_by_source"]["counts"])) == 120
    assert p["hash"] == _scripts_hash(pd.read_csv(shared / "A.csv"))
    assert "crops_missing" not in p["quality"]                # the images are never touched
    assert {c["name"] for c in p["column_profiles"]} == {"source", "split", "brightness"}


def test_nothing_is_guessed_a_column_the_settings_do_not_name_is_not_a_role(env):
    df = _frame(20).rename(columns={"source": "camera"})
    df.to_csv(env / "d.csv", index=False)
    p = profile.compute(env / "d.csv", _columns(FINE))
    assert p["roles"][dcfg.ROLE_SOURCE] == "" and "source" in p["roles"]["missing"]
    assert "sources" not in p
    p = profile.compute(env / "d.csv", {**_columns(FINE), dcfg.ROLE_SOURCE: "camera"})
    assert dict(map(tuple, p["sources"]["counts"])) == {"cam1": 10, "cam2": 10}


def test_the_issues_say_what_is_wrong(env):
    df = _frame(40)
    df.loc[0, "uuid"] = df.loc[1, "uuid"]                 # a duplicated id
    df["split"] = ["train"] * 40
    df.loc[df[FINE] == "cat", "split"] = "test"           # only cats in test
    path = env / "bad.csv"
    df.to_csv(path, index=False)
    p = profile.compute(path, _columns(FINE))
    assert {i["code"] for i in p["issues"]} == {"duplicate_ids"}
    assert {i["code"] for i in p["mappings"][FINE]["issues"]} >= {"class_absent_from_split"}
    assert profile.health(p) == profile.SEVERITY_ERROR
    assert {i.get("mapping") for i in profile.all_issues(p)} == {None, FINE}


def test_a_file_with_none_of_the_mappings_says_so(env):
    _frame(10).to_csv(env / "d.csv", index=False)
    p = profile.compute(env / "d.csv", _columns("nope"))
    assert p["mappings"] == {} and {i["code"] for i in p["issues"]} == {"no_mapping"}


# ---- MLflow ------------------------------------------------------------------

def _run(run_id, csv="A.csv", size=None, hash_=""):
    return mlflow_link.DatasetRun(uri="http://m", experiment="datasets", experiment_id="1", run_id=run_id,
                                  run_name=run_id, csv_path=f"data/{csv}", csv_size=size, hash=hash_)


def test_a_file_is_matched_by_hash_else_by_name_and_size():
    index = {("http://m", "datasets"): mlflow_link.SourceState(
        "http://m", "datasets", runs=[_run("r1", hash_="f" * 64), _run("r2", csv="B.csv", size=10)])}
    assert mlflow_link.match(index, "anything.csv", 1, "f" * 64).status == mlflow_link.LINK_HASH
    by_name = mlflow_link.match(index, "B.csv", 10, "e" * 64)
    assert by_name.status == mlflow_link.LINK_NAME and by_name.run.run_id == "r2"
    assert mlflow_link.match(index, "B.csv", 11, "").status == mlflow_link.LINK_NONE


def test_a_derivative_inherits_its_main_datasets_run(shared):
    index = _index()
    main, derived = _p(shared, "A.csv"), _p(shared, "A/A_clean.csv")
    p = index.compute(main)
    dcfg.add_mlflow("http://m", "datasets")
    index.reload_config()
    index.mlflow = {("http://m", "datasets"): mlflow_link.SourceState(
        "http://m", "datasets", runs=[_run("r1", hash_=p["hash"])])}
    assert index.link(main).status == mlflow_link.LINK_HASH
    link = index.link(derived)
    assert link.status == mlflow_link.LINK_PARENT and link.run.run_id == "r1"


# ---- the index -----------------------------------------------------------------

def test_a_profile_is_cached_until_the_file_or_its_columns_change(shared):
    index = _index()
    path = _p(shared, "B.csv")
    assert index.profile_of(path) is None
    first = index.compute(path)
    assert index.profile_of(path) is first
    DatasetIndex.reset()                                   # a restart: read back from disk
    again = _index()
    assert again.profile_of(path)["hash"] == first["hash"]
    dcfg.set_override(path, {dcfg.ROLE_MAPPINGS: [FINE]})  # other columns: another profile
    again.rescan()
    assert again.profile_of(path) is None
    assert list(again.compute(path)["mappings"]) == [FINE]
    _frame(61, seed=1, classes=["cat", "dog"]).to_csv(path, index=False)
    again.rescan()
    assert again.profile_of(path) is None


def test_the_background_queue_profiles_every_file(shared):
    index = DatasetIndex.get()
    index.request_all()
    deadline = time.time() + 30
    while index.pending() and time.time() < deadline:
        time.sleep(0.05)
    assert all(index.profile_of(f.path) is not None for f in index.files)


# ---- comparison ----------------------------------------------------------------

def test_comparing_a_derivative_with_its_main_dataset_mapping_by_mapping(shared):
    index = _index()
    result = index.compare(_p(shared, "A.csv"), _p(shared, "A/A_clean.csv"))
    assert result["mappings"][FINE]["only_a"] == ["bird"]
    assert 0 < result["mappings"][FINE]["js_divergence"] <= 1
    assert result["mappings"][COARSE]["js_divergence"] == 0.0
    assert result["overlap"]["common"] == 90 and result["overlap"]["only_a"] == 30
    assert result["overlap"]["mappings"][FINE]["label_changes"] == 0


def test_label_changes_are_counted_per_mapping(env):
    a = _frame(40)
    b = a.copy()
    b.loc[:4, FINE] = "dog"
    dcfg.add_folder(str(env / "s"))
    (env / "s").mkdir()
    a.to_csv(env / "s" / "a.csv", index=False)
    b.to_csv(env / "s" / "b.csv", index=False)
    dcfg.set_columns(_columns(FINE, COARSE))
    index = _index()
    ov = index.compare(_p(env, "s/a.csv"), _p(env, "s/b.csv"))["overlap"]
    assert ov["mappings"][FINE]["label_changes"] == 4          # rows 0 to 4, of which row 1 was dog already
    assert ov["mappings"][COARSE]["label_changes"] == 0


def test_js_divergence_bounds():
    assert compare.js_divergence({"a": 5}, {"a": 9}) == 0.0
    assert compare.js_divergence({"a": 5}, {"b": 9}) == pytest.approx(1.0)


# ---- the MCP answers -------------------------------------------------------------

def test_the_agent_answers_per_mapping_in_plain_json(shared):
    listing = agent.list_datasets()
    assert listing["count"] == 3
    clean = next(d for d in listing["datasets"] if d["name"] == "A_clean.csv")
    assert clean["derived_from"] == "shared/A.csv" and clean["group"] == "shared/A"
    stats = agent.get_dataset_stats("A", mapping=FINE, sections=["classes", "class_by_split"], top=2)
    assert list(stats["mappings"]) == [FINE]
    assert stats["mappings"][FINE]["classes"]["count"] == 4
    assert len(stats["mappings"][FINE]["classes"]["largest"]) == 2
    every = agent.get_dataset_stats("A", sections=["classes"])
    assert set(every["mappings"]) == {FINE, COARSE}
    with pytest.raises(ValueError, match="no mapping"):
        agent.get_dataset_stats("A", mapping="nope")
    diff = agent.compare_datasets("A.csv", "shared/A/A_clean.csv")
    assert diff["mappings"][FINE]["only_in_a"] == ["bird"]
    rows = agent.get_dataset_rows("B", label="dog", mapping=FINE, limit=3)
    assert rows["matching"] == 30 and len(rows["rows"]) == 3
    with pytest.raises(ValueError, match="which mapping"):
        agent.get_dataset_rows("B", label="dog")
    with pytest.raises(ValueError, match="No dataset"):
        agent.get_dataset("nope")
