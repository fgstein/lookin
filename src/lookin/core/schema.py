"""Guessing which column is the id, which is the crop path, and which is
the label -- from the file itself rather than from its column names.

Names are a weak signal: a CSV coming out of somebody's training script
calls them anything. What is not weak is the *shape* of the values, and the
three roles have shapes that barely overlap:

* the id is a uuid -- a very specific string format;
* the crop is a path -- it has a directory separator, or an image
  extension, or both;
* the label is a short string with few distinct values.

So the shape decides and the name only breaks ties. Every guess is a
starting point: the Data tab shows them in dropdowns listing every column
in the file, and the user overrides any of them in one click.

Only the first few hundred rows are read. A guess that needs the whole file
would cost more than typing three column names, which is the thing it is
meant to save.

That sample is the head of the file, not a spread, which matters for one of
the three tests. Whether a value is uuid-shaped or path-shaped does not
depend on where it sits, so those two are unaffected. Distinct-value counts
are: a CSV sorted by class shows only the first few classes in its first
200 rows (the bundled example shows 3 of its 10). That biases the count
*downwards*, which favours the real label column rather than working
against it -- but it is why nothing here reports a class count as fact, and
why a column that is constant across the sample is refused outright.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

SAMPLE_ROWS = 200

# Canonical 8-4-4-4-12 hex, and the same thing without dashes -- both are
# what "a uuid" means in a CSV, and rejecting the second would fail on
# exports built with uuid4().hex.
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
UUID_COMPACT_RE = re.compile(r"^[0-9a-f]{32}$", re.I)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")

# Name hints, lowercase, checked as whole names first then as substrings.
# They only ever break a tie between columns that already look right.
UUID_NAMES = ("uuid", "id", "uid", "guid", "sample_id", "image_id", "crop_id", "key")
CROP_NAMES = ("crop", "path", "image", "img", "file", "filename", "filepath", "image_path", "crop_path")
LABEL_NAMES = ("label", "class", "target", "category", "gt", "ground_truth", "y", "annotation")

# Above this many distinct values, a string column is an identifier or free
# text, not a class label -- the same reasoning as
# data_manager.HIGH_CARDINALITY_MAX_UNIQUE, which caps what the UI will
# build one control per value for.
LABEL_MAX_UNIQUE = 200


@dataclass
class ColumnEvidence:
    name: str
    uuid_fraction: float = 0.0     # share of sampled values that are a uuid
    path_fraction: float = 0.0     # share that look like a file path
    n_unique: int = 0
    n_sampled: int = 0
    is_numeric: bool = False


@dataclass
class SchemaGuess:
    columns: list = field(default_factory=list)          # every column, in file order
    uuid_col: str = ""
    crop_col: str = ""
    label_col: str = ""
    evidence: dict = field(default_factory=dict)         # name -> ColumnEvidence
    error: str = ""                                      # why nothing could be read

    def is_complete(self) -> bool:
        return bool(self.uuid_col and self.crop_col and self.label_col)

    def explain(self) -> str:
        """One line saying what was guessed and on what basis -- a guess the
        user cannot see the reasoning for is a guess they have to re-check
        by hand."""
        if self.error:
            return self.error
        if not self.columns:
            return "No columns found."
        parts = []
        for role, name in (("id", self.uuid_col), ("crop", self.crop_col), ("label", self.label_col)):
            if not name:
                parts.append(f"no {role} column recognised")
                continue
            ev = self.evidence.get(name)
            if ev is None:
                parts.append(f"{role}: {name}")
            elif role == "id":
                parts.append(f"id: {name} ({ev.uuid_fraction:.0%} uuid-shaped)")
            elif role == "crop":
                parts.append(f"crop: {name} ({ev.path_fraction:.0%} look like paths)")
            else:
                parts.append(f"label: {name} ({ev.n_unique} distinct value(s) in the sample)")
        sampled = max((ev.n_sampled for ev in self.evidence.values()), default=0)
        return (f"Guessed from the first {sampled:,} row(s): " + ", ".join(parts)
                + " -- change any of them below.")


def _looks_like_path(value: str) -> bool:
    lowered = value.lower()
    return lowered.endswith(IMAGE_SUFFIXES) or "/" in value or "\\\\" in value


def _name_score(name: str, hints: tuple) -> float:
    """1.0 for an exact name, 0.5 for a name containing a hint, 0 otherwise."""
    lowered = name.strip().lower()
    if lowered in hints:
        return 1.0
    return 0.5 if any(hint in lowered for hint in hints) else 0.0


def gather_evidence(df: pd.DataFrame) -> dict:
    evidence = {}
    for column in df.columns:
        series = df[column].dropna()
        as_text = series.astype(str)
        n = len(as_text)
        ev = ColumnEvidence(
            name=str(column),
            n_unique=int(series.nunique()),
            n_sampled=n,
            is_numeric=bool(pd.api.types.is_numeric_dtype(df[column])),
        )
        if n:
            ev.uuid_fraction = float(
                as_text.map(lambda v: bool(UUID_RE.match(v) or UUID_COMPACT_RE.match(v))).mean())
            ev.path_fraction = float(as_text.map(_looks_like_path).mean())
        evidence[str(column)] = ev
    return evidence


def guess_schema(df: pd.DataFrame) -> SchemaGuess:
    """Assign the three roles from the evidence, most distinctive first.

    Order matters and is not arbitrary: a uuid column is unmistakable, so it
    is claimed first and can never be mistaken for a label afterwards; a
    path is nearly as distinctive; the label is whatever short-string column
    is left. Going the other way round would let a low-cardinality id
    column steal the label role.
    """
    guess = SchemaGuess(columns=[str(c) for c in df.columns], evidence=gather_evidence(df))
    taken: set = set()

    def claim(name: str) -> str:
        if name:
            taken.add(name)
        return name or ""

    guess.uuid_col = claim(_best(
        guess, taken,
        shape=lambda ev: ev.uuid_fraction if ev.uuid_fraction >= 0.9 else 0.0,
        names=UUID_NAMES,
        # No uuid-shaped column: fall back to the most identifier-like one
        # -- nearly all-distinct, not numeric. Path-shaped columns are
        # excluded: a column of crop paths is also all-distinct and would
        # otherwise be claimed as the id here, leaving the crop role empty
        # for the one column that unambiguously fills it.
        fallback=lambda ev: (ev.n_unique / ev.n_sampled) if ev.n_sampled and
        ev.n_unique / ev.n_sampled > 0.99 and not ev.is_numeric
        and ev.path_fraction < 0.9 else 0.0,
    ))
    guess.crop_col = claim(_best(
        guess, taken,
        shape=lambda ev: ev.path_fraction if ev.path_fraction >= 0.9 else 0.0,
        names=CROP_NAMES,
    ))
    guess.label_col = claim(_best(
        guess, taken,
        # At least two distinct values: a column that never varies across
        # the sample scores best on "few distinct values" while being the
        # one thing a label certainly is not.
        shape=lambda ev: (
            1.0 - min(ev.n_unique, LABEL_MAX_UNIQUE) / LABEL_MAX_UNIQUE
            if not ev.is_numeric and 2 <= ev.n_unique <= LABEL_MAX_UNIQUE and ev.path_fraction < 0.5
            else 0.0
        ),
        names=LABEL_NAMES,
    ))
    return guess


def _best(guess: SchemaGuess, taken: set, shape, names: tuple, fallback=None) -> str:
    """Highest (shape, name) score among the columns not already claimed.

    The shape score dominates -- the name only separates columns the values
    already agree about, which is what makes a CSV with columns called a, b,
    c work as well as one with sensible names.
    """
    best_name, best_score = "", 0.0
    for name in guess.columns:
        if name in taken:
            continue
        ev = guess.evidence[name]
        score = shape(ev)
        if score <= 0 and fallback is not None:
            score = fallback(ev) * 0.5  # a fallback never outranks a real shape match
        if score <= 0:
            continue
        score += _name_score(name, names) * 0.25
        if score > best_score:
            best_name, best_score = name, score
    return best_name


def sniff_csv(path, sample_rows: int = SAMPLE_ROWS) -> SchemaGuess:
    """Read the header and the first `sample_rows` rows and guess.

    Every failure is a returned SchemaGuess carrying `error`, never an
    exception: this runs on whatever half-typed path is in the field, and a
    traceback on every keystroke would be worse than no guess at all.
    """
    try:
        df = pd.read_csv(path, nrows=sample_rows)
    except FileNotFoundError:
        return SchemaGuess(error="File not found.")
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as ex:
        return SchemaGuess(error=f"Could not read this file: {ex}")
    if df.empty and not len(df.columns):
        return SchemaGuess(error="This file has no columns.")
    return guess_schema(df)
