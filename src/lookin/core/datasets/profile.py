"""The statistics of one dataset CSV, computed in one read and kept as plain
numbers -- a JSON-ready dict that the page draws from, the MCP tools answer
from, and the cache stores. Nothing here draws anything.

Which column is what comes from the settings (config.py), never from the
values: the id, the image path, the split and the source column, and the
*mappings* -- the label columns. A dataset can carry several labellings of
the same rows (a fine label, a coarse one, the label of another taxonomy),
and not every row has every one: an empty cell is a row without a label in
that mapping, counted, not an error.

The profile has two halves, as the page has:

* the instances -- what does not depend on a labelling: rows, the source and
  split columns, every other column, duplicate ids, rows and images;
* one entry per mapping the file has -- its classes, their balance, and how
  they spread across sources and splits.

Reading a 500k-row CSV takes seconds, so a profile is computed once per
content hash and set of columns, in the background, and read from the cache
afterwards (see index.py). Nothing here touches the images.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from lookin.core.datasets import config as dcfg, fingerprint

# Bumped whenever the shape or the meaning of a profile changes: a cached
# profile of another version is recomputed rather than misread.
PROFILE_VERSION = 3
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
OTHER = "(other)"
EMPTY = "(empty)"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


def roles_for(columns: dict, header: list) -> dict:
    """The roles of one file: the configured column names it has, and the
    configured ones it lacks (reported, not an error)."""
    present = set(header)
    roles = {}
    missing = []
    for key in dcfg.SINGLE_ROLES:
        name = columns.get(key) or ""
        roles[key] = name if name in present else ""
        if name and name not in present:
            missing.append(name)
    configured = list(columns.get(dcfg.ROLE_MAPPINGS) or [])
    roles[dcfg.ROLE_MAPPINGS] = [m for m in configured if m in present]
    missing += [m for m in configured if m not in present]
    roles["missing"] = missing
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
    """Labels, split and source as text, where only an empty cell is missing:
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
        "entropy": round(entropy, 6) if k else None,
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


def _issue(severity: str, code: str, message: str, count=None, mapping: str = "") -> dict:
    out = {"severity": severity, "code": code, "message": message, "count": count}
    if mapping:
        out["mapping"] = mapping
    return out


def _instance_issues(p: dict) -> list:
    out = []
    roles, q = p["roles"], p["quality"]
    if not roles[dcfg.ROLE_MAPPINGS]:
        out.append(_issue(SEVERITY_WARNING, "no_mapping",
                          "None of the label columns of the settings is in this file."))
    if q.get("duplicate_ids"):
        out.append(_issue(SEVERITY_ERROR, "duplicate_ids",
                          f"{q['duplicate_ids']:,} rows repeat an id of another row.", q["duplicate_ids"]))
    if q.get("duplicate_images"):
        out.append(_issue(SEVERITY_WARNING, "duplicate_images",
                          f"{q['duplicate_images']:,} rows point at an image another row already uses.",
                          q["duplicate_images"]))
    if q.get("duplicate_rows"):
        out.append(_issue(SEVERITY_WARNING, "duplicate_rows",
                          f"{q['duplicate_rows']:,} rows are exact copies of another row.", q["duplicate_rows"]))
    if q.get("empty_images"):
        out.append(_issue(SEVERITY_WARNING, "empty_images",
                          f"{q['empty_images']:,} rows have no image path.", q["empty_images"]))
    return out


def _mapping_issues(name: str, m: dict) -> list:
    out = []
    classes = m["classes"]
    if classes["imbalance"] and classes["imbalance"] >= IMBALANCE_WARN:
        out.append(_issue(SEVERITY_WARNING, "imbalance",
                          f"The largest class is {classes['imbalance']:,.0f}x the smallest.",
                          round(classes["imbalance"], 1), name))
    if classes["rare"]:
        out.append(_issue(SEVERITY_INFO, "rare_classes",
                          f"{classes['rare']} classes have fewer than {RARE_CLASS_MIN} rows.", classes["rare"], name))
    if classes["missing"]:
        out.append(_issue(SEVERITY_INFO, "unlabelled_rows",
                          f"{classes['missing']:,} rows have no label in this mapping.", classes["missing"], name))
    split = m.get("class_by_split")
    if split:
        for j, value in enumerate(split["cols"]):
            if value in (EMPTY, OTHER):
                continue
            absent = [split["rows"][i] for i, row in enumerate(split["counts"]) if row[j] == 0]
            if absent:
                names = ", ".join(absent[:5]) + (", ..." if len(absent) > 5 else "")
                out.append(_issue(SEVERITY_WARNING, "class_absent_from_split",
                                  f"{len(absent)} classes have no row in split '{value}': {names}.", len(absent), name))
    source = m.get("class_by_source")
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
                              f"or more of their rows: {names}.", len(dominated), name))
    return out


def _by_severity(issues: list) -> list:
    order = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}
    return sorted(issues, key=lambda i: order[i["severity"]])


def all_issues(profile: dict) -> list:
    """Every issue of a profile: the instances' and every mapping's."""
    out = list(profile.get("issues", []))
    for m in profile.get("mappings", {}).values():
        out += m.get("issues", [])
    return _by_severity(out)


def health(profile: dict) -> str:
    """The worst severity among all the issues, or "ok"."""
    severities = {i["severity"] for i in all_issues(profile)}
    for s in (SEVERITY_ERROR, SEVERITY_WARNING):
        if s in severities:
            return s
    return "ok"


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------

def compute(path, columns: dict, progress=None) -> dict:
    """The profile of the CSV at `path`, read with the `columns` of the
    settings (config.DatasetsConfig.columns_for). Raises OSError /
    ValueError / pandas' ParserError on a file that cannot be read."""
    t0 = time.monotonic()
    say = progress or (lambda text: None)
    say("Reading the CSV")
    df = fingerprint.read_for_hash(path)
    say("Hashing the content")
    content_hash = fingerprint.dataset_hash(df)
    roles = roles_for(columns, [str(c) for c in df.columns])
    mappings = roles[dcfg.ROLE_MAPPINGS]
    split, source = roles[dcfg.ROLE_SPLIT], roles[dcfg.ROLE_SOURCE]
    say("Reading the labels")
    text = _read_text_columns(path, mappings + [split, source])
    p: dict = {
        "version": PROFILE_VERSION,
        "hash": content_hash,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "roles": roles,
    }
    if source:
        p["sources"] = _value_summary(text[source])
    if split:
        p["splits"] = _value_summary(text[split])
    p["mappings"] = {}
    for name in mappings:
        say(f"Counting classes of {name}")
        labels = text[name]
        m = {"classes": _class_summary(labels)}
        order = [c for c, _ in m["classes"]["counts"]]
        if source:
            m["class_by_source"] = _cross(labels, text[source], order)
        if split:
            m["class_by_split"] = _cross(labels, text[split], order)
        m["issues"] = _by_severity(_mapping_issues(name, m))
        p["mappings"][name] = m
    say("Profiling the columns")
    skip = {roles[dcfg.ROLE_ID], roles[dcfg.ROLE_IMAGE], *mappings}
    p["column_profiles"] = [_column_profile(str(c), df[c]) for c in df.columns if c not in skip]
    quality = {"duplicate_rows": int(df.duplicated().sum())}
    if roles[dcfg.ROLE_ID]:
        quality["duplicate_ids"] = int(df[roles[dcfg.ROLE_ID]].dropna().duplicated().sum())
    if roles[dcfg.ROLE_IMAGE]:
        images = df[roles[dcfg.ROLE_IMAGE]]
        quality["empty_images"] = int(images.isna().sum())
        quality["duplicate_images"] = int(images.dropna().duplicated().sum())
    p["quality"] = quality
    p["issues"] = _by_severity(_instance_issues(p))
    p["computed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p["seconds"] = round(time.monotonic() - t0, 2)
    return p
