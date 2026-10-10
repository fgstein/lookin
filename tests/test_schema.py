"""Guessing the id / crop / label columns from the file.

The cases that matter are the ones where the names are no help: the shape
of the values has to carry the guess on its own, and a column must never be
claimed for two roles.
"""
from __future__ import annotations

import uuid as uuid_module

import pandas as pd
import pytest

from lookin.core.schema import guess_schema, sniff_csv


def _uuids(n: int) -> list:
    return [str(uuid_module.uuid4()) for _ in range(n)]


@pytest.fixture
def obvious() -> pd.DataFrame:
    return pd.DataFrame({
        "uuid": _uuids(20),
        "crop": [f"crops/img_{i}.png" for i in range(20)],
        "label": ["cat", "dog"] * 10,
        "brightness": range(20),
    })


# ---- the easy case ------------------------------------------------------

def test_conventional_names_are_recognised(obvious):
    guess = guess_schema(obvious)
    assert (guess.uuid_col, guess.crop_col, guess.label_col) == ("uuid", "crop", "label")
    assert guess.is_complete()


def test_the_bundled_example_is_read_correctly():
    guess = sniff_csv("src/lookin/data/example/dataset.csv")
    assert (guess.uuid_col, guess.crop_col, guess.label_col) == ("uuid", "crop", "label")


# ---- names carrying no information -------------------------------------

def test_meaningless_column_names_still_work(obvious):
    """The whole reason the shape decides and the name only breaks ties."""
    renamed = obvious.rename(columns={"uuid": "a", "crop": "b", "label": "c", "brightness": "d"})
    guess = guess_schema(renamed)
    assert (guess.uuid_col, guess.crop_col, guess.label_col) == ("a", "b", "c")


def test_unconventional_but_plausible_names(obvious):
    renamed = obvious.rename(columns={"uuid": "sample_id", "crop": "filepath", "label": "category"})
    guess = guess_schema(renamed)
    assert (guess.uuid_col, guess.crop_col, guess.label_col) == ("sample_id", "filepath", "category")


# ---- shapes -------------------------------------------------------------

def test_a_compact_uuid_counts_as_a_uuid():
    """uuid4().hex is how plenty of exports write ids."""
    df = pd.DataFrame({
        "ident": [uuid_module.uuid4().hex for _ in range(20)],
        "p": [f"a/b_{i}.jpg" for i in range(20)],
        "cls": ["x", "y"] * 10,
    })
    assert guess_schema(df).uuid_col == "ident"


def test_a_bare_directory_path_counts_as_a_crop():
    df = pd.DataFrame({
        "uuid": _uuids(20),
        "where": [f"batch_1/sub/{i}" for i in range(20)],
        "label": ["x", "y"] * 10,
    })
    assert guess_schema(df).crop_col == "where"


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"])
def test_every_supported_image_suffix_reads_as_a_path(suffix):
    df = pd.DataFrame({
        "uuid": _uuids(10),
        "f": [f"img_{i}{suffix}" for i in range(10)],
        "label": ["x", "y"] * 5,
    })
    assert guess_schema(df).crop_col == "f"


# ---- what must NOT be picked -------------------------------------------

def test_a_constant_column_is_never_the_label():
    """It scores best on 'few distinct values' while being the one thing a
    label certainly is not."""
    df = pd.DataFrame({
        "uuid": _uuids(20),
        "crop": [f"c/{i}.png" for i in range(20)],
        "batch": ["batch_1"] * 20,
        "y_true": ["cat", "dog"] * 10,
    })
    assert guess_schema(df).label_col == "y_true"


def test_a_numeric_column_is_not_the_label():
    df = pd.DataFrame({
        "uuid": _uuids(20),
        "crop": [f"c/{i}.png" for i in range(20)],
        "score": [0, 1] * 10,
        "kind": ["cat", "dog"] * 10,
    })
    assert guess_schema(df).label_col == "kind"


def test_free_text_is_not_the_label():
    df = pd.DataFrame({
        "uuid": _uuids(20),
        "crop": [f"c/{i}.png" for i in range(20)],
        "notes": [f"annotator comment number {i}" for i in range(20)],
        "kind": ["cat", "dog"] * 10,
    })
    assert guess_schema(df).label_col == "kind"


def test_no_column_is_claimed_twice():
    """A path column is also a fine-looking id; it must not be both."""
    df = pd.DataFrame({
        "crop": [f"crops/{i}.png" for i in range(20)],
        "label": ["cat", "dog"] * 10,
    })
    guess = guess_schema(df)
    assert guess.crop_col == "crop"
    assert guess.uuid_col != "crop"
    assert guess.label_col == "label"


def test_the_id_falls_back_to_an_all_distinct_string_column():
    """No uuid-shaped column at all -- still better than nothing."""
    df = pd.DataFrame({
        "ref": [f"SPECIMEN-{i:05d}" for i in range(30)],
        "crop": [f"c/{i}.png" for i in range(30)],
        "label": ["cat", "dog"] * 15,
    })
    assert guess_schema(df).uuid_col == "ref"


def test_a_uuid_column_always_beats_the_fallback():
    df = pd.DataFrame({
        "ref": [f"SPECIMEN-{i:05d}" for i in range(30)],
        "uuid": _uuids(30),
        "crop": [f"c/{i}.png" for i in range(30)],
        "label": ["cat", "dog"] * 15,
    })
    assert guess_schema(df).uuid_col == "uuid"


# ---- degrading gracefully ----------------------------------------------

def test_a_file_with_nothing_recognisable_guesses_what_it_can():
    df = pd.DataFrame({"m1": [1.0, 2.0, 3.0], "m2": [4.0, 5.0, 6.0]})
    guess = guess_schema(df)
    assert not guess.is_complete()
    assert guess.columns == ["m1", "m2"], "every column is still offered to the user"
    assert "no crop column recognised" in guess.explain()


def test_a_missing_file_is_reported_not_raised(tmp_path):
    """This runs on whatever half-typed path is in the field."""
    guess = sniff_csv(tmp_path / "nope.csv")
    assert guess.error == "File not found."
    assert guess.columns == []
    assert guess.explain() == "File not found."


def test_a_file_that_is_not_a_csv_is_reported_not_raised(tmp_path):
    binary = tmp_path / "image.png"
    binary.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + bytes(range(256)) * 4)
    guess = sniff_csv(binary)
    assert guess.error, "an unreadable file explains itself"
    assert not guess.is_complete()


def test_only_the_first_rows_are_read(tmp_path):
    """A guess that needed the whole file would cost more than typing three
    column names."""
    df = pd.DataFrame({
        "uuid": _uuids(5000),
        "crop": [f"c/{i}.png" for i in range(5000)],
        "label": ["cat", "dog"] * 2500,
    })
    csv = tmp_path / "big.csv"
    df.to_csv(csv, index=False)
    guess = sniff_csv(csv, sample_rows=50)
    assert guess.evidence["uuid"].n_sampled == 50
    assert guess.is_complete()


def test_the_explanation_says_what_it_looked_at(obvious):
    text = guess_schema(obvious).explain()
    assert "uuid-shaped" in text and "look like paths" in text
    assert "change any of them below" in text
