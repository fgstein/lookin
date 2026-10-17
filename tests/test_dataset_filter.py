"""The Filter tab's engine (core/dataset_filter.py): conditions, the query
language and what it refuses, and the filter on AppState -- applied to
every row, kept across a reload of the same file, dropped for another."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lookin.core import dataset_filter as dsf
from lookin.core.data_manager import AppState
from lookin.core.data_manager import ColumnConfig

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"


@pytest.fixture
def df():
    return pd.DataFrame({
        "label": ["cat", "dog", "cat", "bird", None],
        "width": [10, 40, 64, np.nan, 5],
        "my col": ["a", "b", "c", "d", "e"],
        "path": ["x/blurry.png", "y.png", "z.png", "w.png", "v.png"],
        "flag": [True, False, True, False, True],
        "num_txt": ["1", "2", "n/a", "4", "5"],
    })


@pytest.mark.parametrize("query, kept", [
    ("width > 32", [1, 2]),
    ("label == 'cat' and width < 20", [0]),
    ("label == 'cat' or width < 20", [0, 2, 4]),
    ("`my col` in ['a', 'c']", [0, 2]),
    ("col('my col') == 'e'", [4]),
    ("F('my col') != 'e'", [0, 1, 2, 3]),
    ("path.str.contains('blurry')", [0]),
    ("not contains(path, 'BLURRY')", [1, 2, 3, 4]),
    ("'blur' in path", [0]),
    ("label.isna()", [4]),
    ("label is not None", [0, 1, 2, 3]),
    ("width.between(5, 40)", [0, 1, 4]),
    ("5 <= width <= 40", [0, 1, 4]),
    ("num_txt > 1", [1, 3, 4]),
    ("flag", [0, 2, 4]),
    ("~flag | (width > 50)", [1, 2, 3]),
    ("width * 2 > 100", [2]),
    ("len(label) == 3", [0, 1, 2]),
    ("label.str.upper() == 'CAT'", [0, 2]),
    ("label not in ['cat', 'dog']", [3, 4]),
])
def test_queries(df, query, kept):
    assert np.flatnonzero(dsf.query_mask(df, query).to_numpy()).tolist() == kept


@pytest.mark.parametrize("query, says", [
    ("label = 'cat'", "=="),
    ("lable == 'x'", "Did you mean 'label'"),
    ("__import__('os').system('true')", "Unknown function"),
    ("width.__class__", "not available"),
    ("().__class__.__bases__", "not available"),
    ("open('/etc/passwd')", "Unknown function"),
    ("[x for x in width]", "comprehension"),
    ("width", "must be a condition"),
    ("10 ** 100 > width", "Exponents"),
    ("label.str.contains('(')", "regular expression"),
    ("width > 3 &&  width < 5", "and"),
])
def test_refused_queries_say_why(df, query, says):
    with pytest.raises(dsf.QueryError) as err:
        dsf.query_mask(df, query)
    assert says in str(err.value)


def test_conditions_and_their_query_agree(df):
    spec = dsf.normalize_spec({"conditions": [
        {"column": "label", "op": "in", "values": ["cat", "dog"]},
        {"column": "width", "op": "range", "min": 20, "max": None},
        {"column": "path", "op": "not_contains", "text": "BLUR"},
        {"column": "my col", "op": "in", "values": [], "on": True},       # not filled in: keeps all
        {"column": "flag", "op": "in", "values": ["True"], "on": False},  # off: keeps all
    ]})
    mask, steps = dsf.apply_spec(df, spec)
    assert np.flatnonzero(mask.to_numpy()).tolist() == [1, 2]
    assert len(steps) == 3
    query = dsf.query_from_conditions(spec)
    assert dsf.query_mask(df, query).tolist() == mask.tolist()


def test_values_match_as_text_whatever_the_dtype():
    """A column that went float because one cell is empty still matches "3"."""
    frame = pd.DataFrame({"k": [3.0, 4.0, np.nan]})
    cond = {"column": "k", "op": "in", "values": ["3"], "on": True}
    assert dsf.condition_mask(frame, cond).tolist() == [True, False, False]
    assert dsf.query_mask(frame, "k in ['3', 4]").tolist() == [True, True, False]


def test_malformed_spec_is_dropped_not_raised():
    spec = dsf.normalize_spec({"conditions": [{"column": "", "op": "in"}, {"op": "nope"}, "x",
                                              {"column": "a", "op": "range", "min": "abc", "max": "3"}],
                               "query": 3})
    assert spec["conditions"] == [{"column": "a", "op": "range", "on": True, "min": None, "max": 3.0}]
    assert spec["query"] == ""
    assert dsf.normalize_spec("garbage") == dsf.empty_spec()


def _load(state: AppState, csv: Path):
    state.load_dataset(str(csv), ColumnConfig(uuid_col="uuid", crop_col="crop", label_col="label"))


def test_state_filter_follows_the_file(tmp_path):
    ex = tmp_path / "ex"
    shutil.copytree(EXAMPLE, ex)
    state = AppState()
    _load(state, ex / "dataset.csv")
    total = len(state.dataset.df)
    kept, all_rows = state.set_dataset_filter({"query": "label == 'cat'"})
    assert all_rows == total and 0 < kept < total
    assert len(state.dataset.df) == kept and len(state.dataset.full_df) == total
    # A reload of the same file keeps it...
    _load(state, ex / "dataset.csv")
    assert len(state.dataset.df) == kept
    # ...a filter that no longer applies is not applied in part...
    state.set_dataset_filter({"query": "no_such_column > 1"})
    assert state.dataset_filter_error and len(state.dataset.df) == total
    # ...and another file drops it.
    state.set_dataset_filter({"query": "label == 'cat'"})
    other = tmp_path / "other.csv"
    shutil.copy(ex / "dataset.csv", other)
    _load(state, other)
    assert not state.dataset_filter and len(state.dataset.df) == total


def test_label_remap_reaches_the_rows_out_of_view(tmp_path):
    ex = tmp_path / "ex"
    shutil.copytree(EXAMPLE, ex)
    state = AppState()
    _load(state, ex / "dataset.csv")
    state.set_dataset_filter({"query": "label == 'cat'"})
    state.apply_label_remap({"dog": "puppy"})
    assert "puppy" in set(state.dataset.full_df["label"])
    state.set_dataset_filter({})
    assert "dog" not in set(state.dataset.df["label"])
