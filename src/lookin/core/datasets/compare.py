"""Two datasets side by side: which rows they share, which classes only one
of them has, how their class and source distributions differ, and which
labels changed on the rows they share.

Any two -- a derivative and its main dataset, train and test, two versions
of one dataset. The distributions come from the two profiles; the row-level
part reads the id and label columns of both files, so it needs an id column
on each side.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from lookin.core.datasets.profile import _read_text_columns

TOP_CHANGES = 20


def js_divergence(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence (base 2, so between 0 and 1) between two
    count distributions over the union of their keys."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], dtype=float)
    b = np.array([q.get(k, 0) for k in keys], dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())
    return max(0.0, min(1.0, (kl(a, m) + kl(b, m)) / 2))


def _distribution(counts_a: list, counts_b: list) -> dict:
    a = {k: n for k, n in counts_a}
    b = {k: n for k, n in counts_b}
    total_a, total_b = sum(a.values()) or 1, sum(b.values()) or 1
    keys = sorted(set(a) | set(b), key=lambda k: (-(a.get(k, 0) + b.get(k, 0)), k))
    rows = [{"value": k, "a": a.get(k, 0), "b": b.get(k, 0),
             "share_a": a.get(k, 0) / total_a, "share_b": b.get(k, 0) / total_b} for k in keys]
    return {
        "rows": rows,
        "only_a": [k for k in keys if k in a and k not in b],
        "only_b": [k for k in keys if k in b and k not in a],
        "js_divergence": js_divergence(a, b),
    }


def _rows(a_path, a_roles: dict, b_path, b_roles: dict) -> dict:
    a_id, b_id = a_roles.get("id"), b_roles.get("id")
    if not a_id or not b_id:
        return {"error": "Row-level comparison needs an id column in both datasets."}
    a = _read_text_columns(a_path, [a_id, a_roles.get("label")])
    b = _read_text_columns(b_path, [b_id, b_roles.get("label")])
    a = a.dropna(subset=[a_id]).drop_duplicates(subset=[a_id]).set_index(a_id)
    b = b.dropna(subset=[b_id]).drop_duplicates(subset=[b_id]).set_index(b_id)
    common = a.index.intersection(b.index)
    out = {"common": int(len(common)), "only_a": int(len(a.index.difference(b.index))),
           "only_b": int(len(b.index.difference(a.index)))}
    a_label, b_label = a_roles.get("label"), b_roles.get("label")
    if a_label and b_label and len(common):
        la, lb = a.loc[common, a_label], b.loc[common, b_label]
        changed = (la.fillna("") != lb.fillna(""))
        out["label_changes"] = int(changed.sum())
        pairs = Counter(zip(la[changed].fillna("(none)"), lb[changed].fillna("(none)")))
        out["top_changes"] = [[x, y, int(n)] for (x, y), n in pairs.most_common(TOP_CHANGES)]
    return out


def compare(a_path, a_profile: dict, b_path, b_profile: dict, rows: bool = True) -> dict:
    out = {"rows_a": a_profile.get("rows"), "rows_b": b_profile.get("rows"),
           "same_content": a_profile.get("hash") == b_profile.get("hash")}
    if a_profile.get("classes") and b_profile.get("classes"):
        out["classes"] = _distribution(a_profile["classes"]["counts"], b_profile["classes"]["counts"])
    if a_profile.get("sources") and b_profile.get("sources"):
        out["sources"] = _distribution(a_profile["sources"]["counts"], b_profile["sources"]["counts"])
    if a_profile.get("splits") and b_profile.get("splits"):
        out["splits"] = _distribution(a_profile["splits"]["counts"], b_profile["splits"]["counts"])
    if rows:
        try:
            out["overlap"] = _rows(a_path, a_profile["roles"], b_path, b_profile["roles"])
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as ex:
            out["overlap"] = {"error": f"Could not read the id columns: {ex}"}
    js = out.get("classes", {}).get("js_divergence")
    if js is not None and math.isnan(js):
        out["classes"]["js_divergence"] = None
    return out
