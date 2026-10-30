"""What an agent can ask about the datasets, as plain functions returning
JSON-ready dicts -- the dataset half of the MCP endpoint (see
lookin/mcp_server.py, and core/agent_tools.py for the workspace half, whose
two rules apply here too: answers stay small, and caveats travel with the
numbers).

A dataset is named by its file name ("dataset_A.csv" or "dataset_A"), its
path from its shared folder's name ("datasets/A/A_clean.csv"), its absolute path, or its content
hash. A profile that is not cached yet is computed on the spot, which reads
the whole CSV once; nothing is ever written to the shared folders.
"""
from __future__ import annotations

from lookin.core.agent_tools import DEFAULT_TOP, MAX_TOP, _clean, _ms_to_iso, _top
from lookin.core import library as lib
from lookin.core import links
from lookin.core.datasets import mlflow_link, profile as prof
from lookin.core.datasets.index import DatasetIndex

SECTIONS = ("classes", "sources", "splits", "class_by_source", "class_by_split", "columns", "quality")
DEFAULT_SECTIONS = ("classes", "sources", "splits", "quality")
MAX_CROSS_ROWS = 50
MAX_ROWS = 50

__all__ = ["list_datasets", "get_dataset", "get_dataset_stats", "compare_datasets", "find_dataset_issues",
           "get_dataset_rows", "SECTIONS", "DEFAULT_TOP", "MAX_TOP"]


def _index() -> DatasetIndex:
    index = DatasetIndex.get()
    index.ensure_scanned()
    return index


def _resolve(key: str):
    index = _index()
    f = index.find(key)
    if f is None:
        same = index.ambiguous(key)
        if len(same) > 1:
            raise ValueError(f"'{key}' names {len(same)} datasets: "
                             + ", ".join(x.relative() for x in same) + ". Use the relative path.")
        folders = ", ".join(index.config.folders) or "none configured"
        raise ValueError(f"No dataset '{key}' in the shared folders ({folders}). Call list_datasets.")
    return index, f


def _run_info(link: mlflow_link.Link) -> dict | None:
    if link.run is None:
        return None
    r = link.run
    return {"link": link.status, "why": link.reason, "server": r.uri, "experiment": r.experiment,
            "run_id": r.run_id, "run_name": r.run_name, "started": _ms_to_iso(r.start_time), "user": r.user,
            "description": r.description, "params": r.params, "metrics": r.metrics, "tags": r.tags,
            "csv_artifact": r.csv_path}


def _used_by(path: str) -> list:
    return [{"id": w.id, "name": w.name, "open_url": links.link(w.id)}
            for w in lib.list_workspaces() if w.mount.dataset_path == path]


def list_datasets() -> dict:
    index = _index()
    out = []
    for f in index.files:
        p = index.profile_of(f.path)
        link = index.link(f.path)
        entry = {"name": f.name, "path": f.relative(), "derived_from": index.by_path[f.parent].relative()
                 if f.parent else None, "size_bytes": f.size,
                 "mlflow": link.status or None, "mlflow_run": link.run.label() if link.run else None,
                 "open_url": links.dataset_link(f.path)}
        if p is not None:
            classes = p.get("classes") or {}
            entry.update(rows=p["rows"], classes=classes.get("n"), health=prof.health(p),
                         issues=len(p.get("issues", [])))
        else:
            entry["stats"] = "not computed yet (get_dataset_stats computes them)"
        out.append(entry)
    return _clean({"folders": index.config.folders, "count": len(out), "datasets": out})


def get_dataset(dataset: str) -> dict:
    index, f = _resolve(dataset)
    p = index.profile_of(f.path)
    link = index.link(f.path)
    out = {
        "name": f.name, "path": f.path, "relative_path": f.relative(), "folder": f.root,
        "size_bytes": f.size, "content_hash": index.hash_of(f.path) or None,
        "derived_from": index.by_path[f.parent].relative() if f.parent else None,
        "derivatives": [index.by_path[c].relative() for c in f.children],
        "mlflow": _run_info(link),
        "used_by_workspaces": _used_by(f.path),
        "open_url": links.dataset_link(f.path),
    }
    if p is not None:
        out.update(rows=p["rows"], columns=p["columns"], roles=p["roles"], health=prof.health(p))
    return _clean(out)


def _cross(table: dict, top: int) -> dict:
    rows = table["rows"][:min(top, MAX_CROSS_ROWS)]
    return {"columns": table["cols"], "rows": [[r] + table["counts"][i] for i, r in enumerate(rows)],
            "rows_total": len(table["rows"])}


def get_dataset_stats(dataset: str, sections: list | None = None, top: int = DEFAULT_TOP) -> dict:
    index, f = _resolve(dataset)
    top = _top(top)
    p = index.compute(f.path)
    wanted = [s for s in (sections or DEFAULT_SECTIONS) if s in SECTIONS]
    out = {"name": f.name, "rows": p["rows"], "roles": {k: v for k, v in p["roles"].items() if k not in ("note", "error")},
           "health": prof.health(p), "warnings": [i["message"] for i in p.get("issues", [])
                                                  if i["severity"] != prof.SEVERITY_INFO],
           "open_url": links.dataset_link(f.path)}
    if "classes" in wanted and p.get("classes"):
        c = p["classes"]
        out["classes"] = {"count": c["n"], "labelled_rows": c["labelled"], "missing_labels": c["missing"],
                          "imbalance_max_over_min": c["imbalance"], "normalized_entropy": c["entropy"],
                          f"classes_under_{c['rare_min']}_rows": c["rare"],
                          "largest": c["counts"][:top], "smallest": c["counts"][::-1][:top]}
    for key in ("sources", "splits"):
        if key in wanted and p.get(key):
            out[key] = {"count": p[key]["n"], "missing": p[key]["missing"], "values": p[key]["counts"][:top]}
    for key in ("class_by_source", "class_by_split"):
        if key in wanted and p.get(key):
            out[key] = _cross(p[key], top)
    if "columns" in wanted:
        cols = []
        for c in p.get("column_profiles", []):
            item = {k: c.get(k) for k in ("name", "kind", "missing", "unique")}
            if c.get("kind") == "numeric":
                item.update({k: c.get(k) for k in ("min", "max", "mean", "std")})
            elif c.get("top"):
                item["top"] = c["top"][:5]
            cols.append(item)
        out["columns"] = cols
    if "quality" in wanted:
        out["quality"] = p.get("quality", {})
    return _clean(out)


def compare_datasets(dataset_a: str, dataset_b: str, top: int = DEFAULT_TOP) -> dict:
    index, a = _resolve(dataset_a)
    _, b = _resolve(dataset_b)
    top = _top(top)
    result = index.compare(a.path, b.path)
    out = {"a": a.relative(), "b": b.relative(), "rows_a": result["rows_a"], "rows_b": result["rows_b"],
           "same_content": result["same_content"], "open_url": links.dataset_link(a.path, b.path)}
    for key in ("classes", "sources", "splits"):
        d = result.get(key)
        if not d:
            continue
        rows = sorted(d["rows"], key=lambda r: -abs(r["share_b"] - r["share_a"]))[:top]
        out[key] = {"js_divergence": d["js_divergence"], "only_in_a": d["only_a"][:top],
                    "only_in_b": d["only_b"][:top],
                    "largest_share_changes": [{"value": r["value"], "a": r["a"], "b": r["b"],
                                               "share_a": r["share_a"], "share_b": r["share_b"]} for r in rows]}
    if "overlap" in result:
        ov = dict(result["overlap"])
        if "top_changes" in ov:
            ov["top_changes"] = [{"from": x, "to": y, "rows": n} for x, y, n in ov["top_changes"][:top]]
        out["rows_by_id"] = ov
    return _clean(out)


def find_dataset_issues(dataset: str | None = None) -> dict:
    """Issues of one dataset, or of every dataset whose stats are cached."""
    index = _index()
    if dataset:
        _, f = _resolve(dataset)
        p = index.compute(f.path)
        return _clean({"name": f.name, "health": prof.health(p), "issues": p.get("issues", [])})
    out, pending = [], []
    for f in index.files:
        p = index.profile_of(f.path)
        if p is None:
            pending.append(f.relative())
            continue
        if p.get("issues"):
            out.append({"name": f.relative(), "health": prof.health(p), "issues": p["issues"]})
    order = {prof.SEVERITY_ERROR: 0, prof.SEVERITY_WARNING: 1, "ok": 2, prof.SEVERITY_INFO: 2}
    out.sort(key=lambda d: order.get(d["health"], 3))
    return _clean({"datasets_with_issues": out, "not_computed_yet": pending,
                   "warnings": [f"{len(pending)} datasets have no stats yet; name one to compute it."] if pending else []})


def get_dataset_rows(dataset: str, label: str | None = None, source: str | None = None,
                     split: str | None = None, limit: int = DEFAULT_TOP) -> dict:
    import pandas as pd

    index, f = _resolve(dataset)
    p = index.compute(f.path)
    roles = p["roles"]
    limit = max(1, min(int(limit or DEFAULT_TOP), MAX_ROWS))
    df = pd.read_csv(f.path, keep_default_na=False, na_values=[""], dtype=str)
    mask = pd.Series(True, index=df.index)
    for value, col, what in ((label, roles.get("label"), "label"), (source, roles.get("source"), "source"),
                             (split, roles.get("split"), "split")):
        if value is None:
            continue
        if not col:
            raise ValueError(f"This dataset has no {what} column.")
        mask &= df[col] == str(value)
    picked = df[mask]
    return _clean({"name": f.name, "matching": int(len(picked)), "shown": int(min(limit, len(picked))),
                   "rows": picked.head(limit).fillna("").to_dict(orient="records")})
