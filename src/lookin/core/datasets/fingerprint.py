"""The content hash of a dataset CSV -- the same one the dataset scripts
compute and log to MLflow as hash.txt, so a CSV copied by hand into a
shared folder can be traced back to the run that produced it.

It is a hash of the *table*, not of the file: columns sorted by name, rows
sorted by their own hash, so two CSVs holding the same rows in a different
order are the same dataset. That also means it depends on how pandas reads
the file (dtypes, what counts as missing), so the CSV is read exactly the
way the scripts' own `pd.read_csv(path)` does -- with no option at all.
If the scripts change how they hash, `dataset_hash` is the one function to
change.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")


def dataset_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the dataset's content, independent of row and column order."""
    df = df[sorted(df.columns)]
    row_hashes = pd.util.hash_pandas_object(df, index=False).sort_values()

    digest = hashlib.sha256()
    digest.update(",".join(sorted(df.columns)).encode("utf-8"))
    digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def read_for_hash(path) -> pd.DataFrame:
    """The CSV as the dataset scripts read it."""
    return pd.read_csv(path)


def parse_hash_file(text: str) -> str:
    """The hash in a hash.txt: the first 64-character hex string in it, so a
    trailing newline or a `sha256sum`-style "hash  file" line both work.
    "" when there is none."""
    found = HASH_RE.search((text or "").lower())
    return found.group(0) if found else ""
