"""The statistics of one dataset CSV, computed in one read and kept as plain
numbers -- a JSON-ready dict that the page draws from, the MCP tools answer
from, and the cache stores. Nothing here draws anything.

What comes first is what a dataset is judged on: how balanced its classes
are, where its rows come from (the `source` column) and how the classes are
spread across sources and across splits. Then every other column, then the
checks that say whether the file can be trusted as it is.

Reading a 500k-row CSV takes seconds, so a profile is computed once per
content hash, in the background, and read from the cache afterwards (see
index.py). Nothing here opens an image: the crop check only stats a sample
of paths.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lookin.core.dataset_config import config_for_csv
from lookin.core.datasets import fingerprint
from lookin.core.schema import sniff_csv

# Bumped whenever the shape or the meaning of a profile changes: a cached
# profile of another version is recomputed rather than misread.
PROFILE_VERSION = 2
# A class with fewer examples than this is "rare": too few to train on or
# to measure a per-class metric with any confidence.
RARE_CLASS_MIN = 10
# Above this max/min ratio between class sizes, the imbalance is flagged.
IMBALANCE_WARN = 100.0
# A class whose rows come at this share or more from one source, in a
# dataset that has several, is flagged: a model can learn the source.
SOURCE_DOMINANCE = 0.95
TOP_VALUES = 10
HIST_BINS = 20
# Above this many distinct values a text column is an identifier or free
# text, not a category (same threshold as data_manager.HIGH_CARDINALITY_MAX_UNIQUE).
CATEGORICAL_MAX_UNIQUE = 200
# Columns of a cross table past this many values: the rest are summed into
# OTHER, so a source column with thousands of values stays a readable table.
MAX_CROSS_VALUES = 30
CROP_CHECK_SAMPLE = 200
MISSING_EXAMPLES = 5
OTHER = "(other)"
EMPTY = "(empty)"

SPLIT_NAMES = ("split", "subset", "fold", "partition")
SOURCE_NAMES = ("source", "origin", "provenance", "camera", "sensor", "device", "site")
# A column named exactly like this is the label, whatever the guess from the
# values says: the guess favours the column with the fewest distinct values
# in the first rows, and a flag column (ok / blurry) beats 40 classes there.
LABEL_NAMES = ("label", "class", "target", "category")

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass
class Roles:
    """Which column is what. "" when the file has none."""
    id: str = ""
    image: str = ""
    label: str = ""
    split: str = ""
    source: str = ""
    root: str = ""        # folder crop paths are relative to; "" = the CSV's folder
    note: str = ""        # where the roles came from
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _by_name(columns: list, names: tuple) -> str:
    lowered = {c.strip().lower(): c for c in columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return ""


def resolve_roles(path) -> Roles:
    """The guess from the values (schema.sniff_csv), overridden by what a
    lookin.toml beside the CSV says -- the same rule as the Sources tab. The
    split and source columns come from lookin.toml, else by name."""
    guess = sniff_csv(path)
    if guess.error:
        return Roles(error=guess.error)
    columns = list(guess.columns)
    roles = Roles(id=guess.uuid_col or "", image=guess.crop_col or "", label=guess.label_col or "",
                  note="Guessed from the values.")
    named_label = _by_name(columns, LABEL_NAMES)
    if named_label and named_label not in (roles.id, roles.image):
        roles.label = named_label
    roles.split = _by_name(columns, SPLIT_NAMES)
    roles.source = _by_name(columns, SOURCE_NAMES)
    config = config_for_csv(path)
    if config is not None and not config.error:
        for attr, value in (("id", config.id), ("image", config.image), ("label", config.label),
                            ("split", config.split_col), ("source", config.source_col)):
            if value is not None and (not value or value in columns):
                setattr(roles, attr, value)
        if config.root:
            roles.root = config.root
        roles.note = f"From {config.source}."
    elif config is not None and config.error:
        roles.note = f"{config.source} {config.error}; guessed from the values instead."
    return roles


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------

def _as_label_strings(series: pd.Series) -> pd.Series:
    """Labels as strings, missing left missing; whole-number floats lose
    their ".0" (see data_manager._labels_as_str)."""
    values = series
    if pd.api.types.is_float_dtype(values):
        present = values.dropna()
        if len(present) and (present == present.round()).all():
            values = values.astype("Int64")
    return values.astype(str).astype(object).where(values.notna(), None)


def _read_text_columns(path, columns: list) -> pd.DataFrame:
    """Label, split and source as text, where only an empty cell is missing:
    "None" or "NA" can be a class name (see data_manager._label_na_options)."""
    wanted = [c for c in dict.fromkeys(columns) if c]
    if not wanted:
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=wanted, keep_default_na=False, na_values={c: [""] for c in wanted})
    return pd.DataFrame({c: _as_label_strings(df[c]) for c in wanted})


def _counts(series: pd.Series) -> list:
    """[[value, count], ...] largest first, ties by value."""
    counts = series.dropna().value_counts()
    return sorted(([str(k), int(v)] for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0]))


def _class_summary(labels: pd.Series) -> dict:
    counts = _counts(labels)
    sizes = np.array([n for _, n in counts], dtype=float)
    k = len(sizes)
    total = float(sizes.sum())
    entropy = 1.0
    if k > 1 and total > 0:
        p = sizes / total
        entropy = float(-(p * np.log(p)).sum() / math.log(k))
    return {
        "counts": counts,
        "n": k,
        "labelled": int(total),
        "missing": int(labels.isna().sum()),
        "imbalance": float(sizes.max() / sizes.min()) if k else None,
        "entropy": entropy if k else None,
        "rare": int((sizes < RARE_CLASS_MIN).sum()),
        "rare_min": RARE_CLASS_MIN,
        "median": float(np.median(sizes)) if k else None,
    }


def _value_summary(values: pd.Series) -> dict:
    return {"counts": _counts(values), "missing": int(values.isna().sum()), "n": int(values.nunique(dropna=True))}


def _cross(labels: pd.Series, other: pd.Series, class_order: list) -> dict:
    """Classes (rows, in `class_order`) x values of `other` (columns, largest
    first), as counts. An empty cell of `other` is a value of its own."""
    keep = labels.notna()
    rows = labels[keep]
    cols = other[keep].fillna(EMPTY)
    totals = cols.value_counts()
    order = [str(v) for v in totals.index]
    if len(order) > MAX_CROSS_VALUES:
        shown = set(order[:MAX_CROSS_VALUES - 1])
        cols = cols.where(cols.isin(shown), OTHER)
        order = order[:MAX_CROSS_VALUES - 1] + [OTHER]
    table = pd.crosstab(rows, cols).reindex(index=class_order, columns=order, fill_value=0)
    return {"rows": class_order, "cols": order, "counts": table.to_numpy(dtype=int).tolist(),
            "col_totals": [int(x) for x in table.sum(axis=0).tolist()]}


def _column_profile(name: str, s: pd.Series) -> dict:
    out = {"name": name, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
           "unique": int(s.nunique(dropna=True))}
    present = s.dropna()
    if pd.api.types.is_bool_dtype(s) or (out["unique"] <= 2 and not pd.api.types.is_float_dtype(s)):
        out["kind"] = "categorical"
    elif pd.api.types.is_numeric_dtype(s):
        values = present.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        out["kind"] = "numeric"
        out["infinite"] = int(len(values) - len(finite))
        if len(finite):
            q = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
            counts, edges = np.histogram(finite, bins=HIST_BINS)
            out.update(min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()),
                       std=float(finite.std()), quantiles=[float(x) for x in q],
                       hist={"edges": [float(x) for x in edges], "counts": [int(x) for x in counts]})
        return out
    else:
        out["kind"] = "categorical" if out["unique"] <= CATEGORICAL_MAX_UNIQUE else "text"
    counts = present.astype(str).value_counts()
    out["top"] = [[str(k), int(v)] for k, v in counts.head(TOP_VALUES).items()]
    out["other"] = int(counts.iloc[TOP_VALUES:].sum()) if len(counts) > TOP_VALUES else 0
    return out


def _crop_check(images: pd.Series, csv_path, root: str) -> dict:
    present = images.dropna().astype(str)
    present = present[present.str.len() > 0]
    if present.empty:
        return {"checked": 0, "missing": 0, "examples": []}
    n = min(CROP_CHECK_SAMPLE, len(present))
    picks = present.iloc[np.linspace(0, len(present) - 1, n).astype(int)]
    base = Path(root) if root else Path(csv_path).resolve().parent
    missing = []
    for value in picks:
        p = Path(value)
        if not p.is_absolute():
            p = base / p
        if not os.path.isfile(p):
            missing.append(value)
    return {"checked": int(n), "missing": len(missing), "examples": missing[:MISSING_EXAMPLES],
            "base": str(base)}


def _issue(severity: str, code: str, message: str, count=None) -> dict:
    return {"severity": severity, "code": code, "message": message, "count": count}


def _issues(p: dict) -> list:
    out = []
    roles, q, classes = p["roles"], p["quality"], p.get("classes")
    if not roles.get("label"):
        out.append(_issue(SEVERITY_ERROR, "no_label", "No label column found: name it in a lookin.toml beside the CSV."))
    if q.get("crops_missing"):
        out.append(_issue(SEVERITY_ERROR, "crops_missing",
                          f"{q['crops_missing']} of {q['crops_checked']} sampled crop paths do not exist "
                          f"(relative to {q.get('crops_base', 'the CSV folder')}).", q["crops_missing"]))
    if q.get("duplicate_ids"):
        out.append(_issue(SEVERITY_ERROR, "duplicate_ids",
                          f"{q['duplicate_ids']:,} rows repeat an id of another row.", q["duplicate_ids"]))
    if q.get("missing_labels"):
        out.append(_issue(SEVERITY_WARNING, "missing_labels",
                          f"{q['missing_labels']:,} rows have no label.", q["missing_labels"]))
    if q.get("duplicate_images"):
        out.append(_issue(SEVERITY_WARNING, "duplicate_images",
                          f"{q['duplicate_images']:,} rows point at a crop another row already uses.",
                          q["duplicate_images"]))
    if q.get("duplicate_rows"):
        out.append(_issue(SEVERITY_WARNING, "duplicate_rows",
                          f"{q['duplicate_rows']:,} rows are exact copies of another row.", q["duplicate_rows"]))
    if q.get("empty_images"):
        out.append(_issue(SEVERITY_WARNING, "empty_images",
                          f"{q['empty_images']:,} rows have no crop path.", q["empty_images"]))
    if classes:
        if classes["imbalance"] and classes["imbalance"] >= IMBALANCE_WARN:
            out.append(_issue(SEVERITY_WARNING, "imbalance",
                              f"The largest class is {classes['imbalance']:,.0f}x the smallest.",
                              round(classes["imbalance"], 1)))
        if classes["rare"]:
            out.append(_issue(SEVERITY_INFO, "rare_classes",
                              f"{classes['rare']} classes have fewer than {RARE_CLASS_MIN} rows.", classes["rare"]))
    split = p.get("class_by_split")
    if split:
        for j, value in enumerate(split["cols"]):
            if value in (EMPTY, OTHER):
                continue
            absent = [split["rows"][i] for i, row in enumerate(split["counts"]) if row[j] == 0]
            if absent:
                names = ", ".join(absent[:5]) + (", ..." if len(absent) > 5 else "")
                out.append(_issue(SEVERITY_WARNING, "class_absent_from_split",
                                  f"{len(absent)} classes have no row in split '{value}': {names}.", len(absent)))
    source = p.get("class_by_source")
    if source and len([c for c in source["cols"] if c != EMPTY]) > 1:
        dominated = []
        for i, row in enumerate(source["counts"]):
            total = sum(row)
            if total >= RARE_CLASS_MIN and max(row) >= SOURCE_DOMINANCE * total:
                dominated.append(source["rows"][i])
        if dominated:
            names = ", ".join(dominated[:5]) + (", ..." if len(dominated) > 5 else "")
            out.append(_issue(SEVERITY_INFO, "single_source_classes",
                              f"{len(dominated)} classes come from one source for {SOURCE_DOMINANCE:.0%} "
                              f"or more of their rows: {names}.", len(dominated)))
    order = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}
    return sorted(out, key=lambda i: order[i["severity"]])


def health(profile: dict) -> str:
    """The worst severity among the issues, or "ok"."""
    severities = {i["severity"] for i in profile.get("issues", [])}
    for s in (SEVERITY_ERROR, SEVERITY_WARNING):
        if s in severities:
            return s
    return "ok"


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------

def compute(path, roles: Roles = None, progress=None) -> dict:
    """The profile of the CSV at `path`. Raises OSError / ValueError /
    pandas' ParserError on a file that cannot be read."""
    t0 = time.monotonic()
    say = progress or (lambda text: None)
    roles = roles if roles is not None else resolve_roles(path)
    say("Reading the CSV")
    df = fingerprint.read_for_hash(path)
    say("Hashing the content")
    content_hash = fingerprint.dataset_hash(df)
    say("Counting classes")
    text = _read_text_columns(path, [roles.label, roles.split, roles.source])
    p: dict = {
        "version": PROFILE_VERSION,
        "hash": content_hash,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "roles": roles.to_dict(),
    }
    labels = text[roles.label] if roles.label else None
    if labels is not None:
        p["classes"] = _class_summary(labels)
        order = [c for c, _ in p["classes"]["counts"]]
        if roles.source:
            p["sources"] = _value_summary(text[roles.source])
            p["class_by_source"] = _cross(labels, text[roles.source], order)
        if roles.split:
            p["splits"] = _value_summary(text[roles.split])
            p["class_by_split"] = _cross(labels, text[roles.split], order)
    say("Profiling the columns")
    skip = {roles.id, roles.image, roles.label}
    p["column_profiles"] = [_column_profile(str(c), df[c]) for c in df.columns if c not in skip]
    say("Checking crops")
    quality = {"missing_labels": int(labels.isna().sum()) if labels is not None else 0,
               "duplicate_rows": int(df.duplicated().sum())}
    if roles.id:
        quality["duplicate_ids"] = int(df[roles.id].dropna().duplicated().sum())
    if roles.image:
        images = df[roles.image]
        quality["empty_images"] = int(images.isna().sum())
        quality["duplicate_images"] = int(images.dropna().duplicated().sum())
        check = _crop_check(images, path, roles.root)
        quality.update(crops_checked=check["checked"], crops_missing=check["missing"],
                       crops_missing_examples=check["examples"], crops_base=check.get("base", ""))
    p["quality"] = quality
    p["issues"] = _issues(p)
    p["computed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p["seconds"] = round(time.monotonic() - t0, 2)
    return p
