"""Shared fixtures.

Everything here builds tiny in-memory or tmp_path data: these tests cover
lookin.core, which is pure computation with no Flet dependency, so nothing
needs a browser, a server or the bundled example dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.data_manager import AppState, ColumnConfig


@pytest.fixture
def tiny_dataset_df() -> pd.DataFrame:
    """Six crops, three classes, one column of each metadata kind."""
    return pd.DataFrame({
        "uuid": ["a", "b", "c", "d", "e", "f"],
        "crop": [f"crops/{u}.png" for u in "abcdef"],
        "label": ["cat", "cat", "dog", "dog", "bird", "bird"],
        "flag": [0, 1, 0, 1, 0, 1],                      # binary
        "site": ["north", "north", "south", "south", "east", "east"],  # categorical (3 values)
        "area": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],          # numeric
        "note": [f"free text {i}" for i in range(6)],     # high cardinality (with the cap lowered)
    })


@pytest.fixture
def tiny_results_df() -> pd.DataFrame:
    """Scores for the same six uuids: rows 0-3 predicted right, 4-5 wrong."""
    return pd.DataFrame({
        "uuid": ["a", "b", "c", "d", "e", "f"],
        "cat":  [0.8, 0.7, 0.1, 0.2, 0.1, 0.1],
        "dog":  [0.1, 0.2, 0.8, 0.7, 0.1, 0.8],
        "bird": [0.1, 0.1, 0.1, 0.1, 0.8, 0.1],
        "label": ["cat", "cat", "dog", "dog", "bird", "dog"],  # ignored as a score column
    })


@pytest.fixture
def dataset_csv(tmp_path, tiny_dataset_df):
    path = tmp_path / "dataset.csv"
    tiny_dataset_df.to_csv(path, index=False)
    return path


@pytest.fixture
def results_csv(tmp_path, tiny_results_df):
    path = tmp_path / "results_exp_results.csv"
    tiny_results_df.to_csv(path, index=False)
    return path


@pytest.fixture
def loaded_state(dataset_csv, results_csv) -> AppState:
    """An AppState with the dataset and one experiment loaded."""
    state = AppState()
    state.load_dataset(str(dataset_csv), ColumnConfig())
    state.add_experiment(str(results_csv), "exp")
    return state


@pytest.fixture
def embeddings_npz(tmp_path):
    """A sidecar covering only four of the six uuids -- partial coverage is
    normal, not an error (see Experiment.aligned_embeddings)."""
    path = tmp_path / "emb.npz"
    np.savez(
        path,
        uuid=np.array(["a", "b", "c", "d"], dtype="<U8"),
        embeddings=np.eye(4, 3, dtype=np.float32),
    )
    return path
