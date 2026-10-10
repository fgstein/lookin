"""Scaling benchmark for lookin.

Measures how each tab's core operations scale with (a) dataset row count and
(b) metadata column count, by calling lookin's own production code directly
(AppState, StatsView, GalleryView, MetricsView) rather than reimplementing
the logic -- the numbers this produces are exactly what the real app does,
minus the browser round-trip.

Generated CSVs land OUTSIDE this repo (see --out, defaults to a sibling
directory of the repo) since the high end of the sweep is hundreds of MB --
never under src/, so the repo itself stays small to git or copy.

Usage:
    uv run python scripts/benchmark_scaling.py
    uv run python scripts/benchmark_scaling.py --out /path/to/scratch --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lookin.core.data_manager import AppState, ColumnConfig  # noqa: E402
from lookin.ui import components as ui_components  # noqa: E402
from lookin.ui.gallery_view import GalleryView  # noqa: E402
from lookin.ui.interactive_projection import InteractiveProjection  # noqa: E402
from lookin.ui.metrics_view import MetricsView  # noqa: E402
from lookin.ui.stats_view import StatsView  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EXAMPLE_CROPS_DIR = REPO_ROOT / "src" / "lookin" / "data" / "example" / "crops"

ROW_SWEEP = [0, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
ROW_SWEEP_FIXED_COLS = 20

COL_SWEEP = [0, 5, 10, 20, 35, 50]
COL_SWEEP_FIXED_ROWS = 200_000

QUICK_ROW_SWEEP = [0, 1_000, 20_000]
QUICK_COL_SWEEP = [0, 10]
QUICK_FIXED_ROWS = 20_000

# The Embeddings tab's interactive map is measured in *points on the map*
# (its "Sample size" field), not dataset rows -- a million-row dataset drawn
# at the default 1,500-point sample costs what 1,500 points cost.
POINT_SWEEP = [1_500, 10_000, 50_000, 200_000]
QUICK_POINT_SWEEP = [1_500, 50_000]


class FakeCtx:
    """Just enough of AppContext for the view classes to render without a
    live Flet page -- every .update() call in the real views is already
    guarded by `if control.page:`, so it's skipped safely here, and no
    render path touches ctx.page during a pure render (only click handlers
    that this benchmark never invokes do)."""
    def __init__(self, state):
        self.state = state
        self.page = None


def _column_spec(n_extra_cols: int):
    """Deterministic mix of column kinds, representative of typical CV
    metadata: numeric scores, binary flags, bounded categoricals."""
    spec = []
    for i in range(n_extra_cols):
        kind = i % 5
        if kind in (0, 1):
            spec.append(("numeric", f"metric_{i}"))
        elif kind == 2:
            spec.append(("binary", f"flag_{i}"))
        else:
            spec.append(("categorical", f"cat_{i}"))
    return spec


def _crop_pool() -> list:
    pool = sorted(p.name for p in EXAMPLE_CROPS_DIR.glob("*.png"))
    if not pool:
        raise RuntimeError(f"No example crop images found in {EXAMPLE_CROPS_DIR}")
    return pool


def generate_dataset_csv(path: Path, n_rows: int, n_extra_cols: int, crop_pool: list, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data = {
        "uuid": [f"{i:09d}-bench" for i in range(n_rows)],
        "crop": [crop_pool[i % len(crop_pool)] for i in range(n_rows)],
        "label": rng.choice(CLASSES, size=n_rows) if n_rows else np.array([], dtype=object),
    }
    for kind, name in _column_spec(n_extra_cols):
        if kind == "numeric":
            data[name] = rng.uniform(0, 1, size=n_rows) if n_rows else np.array([], dtype=float)
        elif kind == "binary":
            data[name] = rng.integers(0, 2, size=n_rows) if n_rows else np.array([], dtype=int)
        else:
            data[name] = rng.choice([f"v{k}" for k in range(6)], size=n_rows) if n_rows else np.array([], dtype=object)
    pd.DataFrame(data).to_csv(path, index=False)


def generate_experiment_csv(path: Path, n_rows: int, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    data = {"uuid": [f"{i:09d}-bench" for i in range(n_rows)]}
    scores = rng.dirichlet(np.ones(len(CLASSES)), size=n_rows) if n_rows else np.zeros((0, len(CLASSES)))
    for i in range(len(CLASSES)):
        data[f"class_{i + 1}"] = scores[:, i]
    pd.DataFrame(data).to_csv(path, index=False)


# ---- benchmarks (each calls real lookin production code) ----

def bench_dataset_load(csv_path: Path):
    state = AppState()
    t0 = time.perf_counter()
    state.load_dataset(str(csv_path), ColumnConfig(), progress_cb=None, image_root=str(EXAMPLE_CROPS_DIR))
    dt = time.perf_counter() - t0
    return dt, state


def bench_experiment_load(state: AppState, csv_path: Path):
    t0 = time.perf_counter()
    state.add_experiment(str(csv_path), "bench_exp", progress_cb=None)
    return time.perf_counter() - t0


def bench_stats_all_charts(state: AppState):
    ctx = FakeCtx(state)
    view = StatsView.__new__(StatsView)
    StatsView.__init__(view, ctx)
    cols_cfg = state.dataset.columns
    kinds = state.dataset.column_kinds()
    view.active_columns = [cols_cfg.label_col] + kinds.numeric + kinds.binary + kinds.categorical
    t0 = time.perf_counter()
    view._render()
    dt = time.perf_counter() - t0
    return dt, len(view.active_columns)


def bench_gallery_filters(state: AppState):
    ctx = FakeCtx(state)
    view = GalleryView.__new__(GalleryView)
    GalleryView.__init__(view, ctx)
    cols_cfg = state.dataset.columns
    kinds = state.dataset.column_kinds()
    view.active_filter_cols = [cols_cfg.label_col] + kinds.categorical + kinds.binary + kinds.numeric
    t0 = time.perf_counter()
    view._rebuild_filters()
    dt = time.perf_counter() - t0
    return dt, view


def bench_gallery_grid(view: GalleryView):
    ui_components._IMAGE_CACHE.clear()  # force cold image reads, same as a never-before-seen page
    t0 = time.perf_counter()
    view._render_grid()
    return time.perf_counter() - t0


def bench_metrics(state: AppState):
    ctx = FakeCtx(state)
    view = MetricsView.__new__(MetricsView)
    MetricsView.__init__(view, ctx)
    view._rebuild_experiment_checks()
    t0 = time.perf_counter()
    view._render()
    return time.perf_counter() - t0


def bench_projection_frame(n_points: int) -> dict:
    """One frame of the interactive projection: what a pan, a wheel zoom or
    a window resize costs, at `n_points` points on the map.

    Two numbers per frame, because they are two different bottlenecks --
    building the canvas shapes (Python and numpy) and serializing the
    coordinates to JSON (what actually goes over the websocket, and what
    the browser then has to parse). A "draft" frame is the thinned one the
    map serves while a gesture is still moving; the full frame is what
    lands once it stops. See ui/interactive_projection._RedrawPump.
    """
    rng = np.random.default_rng(0)
    view = InteractiveProjection.__new__(InteractiveProjection)
    view.points = rng.normal(size=(n_points, 2))
    colors = [f"#{i:02x}4c78" for i in rng.integers(0, len(CLASSES), n_points)]
    view.point_colors = colors
    view._color_array = np.asarray(colors)
    view.point_groups = np.asarray([CLASSES[i] for i in rng.integers(0, len(CLASSES), n_points)])
    view.prototypes = None
    view.size = 820
    view._hidden_groups = set()
    view._selected = np.array([], dtype=int)
    view._last_frame = None
    view._fit()
    view.sx, view.sy = view._to_screen(view.points[:, 0], view.points[:, 1])

    result = {"n_points": n_points}
    for kind, draft in (("full", False), ("draft", True)):
        t0 = time.perf_counter()
        shapes = view._point_shapes(draft=draft)
        result[f"{kind}_build_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        payload = json.dumps([shape.points for shape in shapes])
        result[f"{kind}_json_s"] = time.perf_counter() - t0
        result[f"{kind}_kb"] = len(payload) / 1024
        result[f"{kind}_points"] = sum(len(shape.points) for shape in shapes)
    return result


def run_variant(label: str, out_dir: Path, n_rows: int, n_cols: int, crop_pool: list, keep_csv: bool) -> dict:
    print(f"[bench] === {label}: {n_rows:,} rows x {n_cols} metadata columns ===", flush=True)
    ds_path = out_dir / f"ds_{label}.csv"
    exp_path = out_dir / f"exp_{label}.csv"

    t0 = time.perf_counter()
    generate_dataset_csv(ds_path, n_rows, n_cols, crop_pool)
    generate_experiment_csv(exp_path, n_rows)
    gen_dt = time.perf_counter() - t0
    print(f"[bench]   generated CSVs in {gen_dt:.1f}s "
          f"(dataset {ds_path.stat().st_size / 1e6:.1f} MB, experiment {exp_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    result = {"label": label, "n_rows": n_rows, "n_cols": n_cols}

    dt, state = bench_dataset_load(ds_path)
    result["dataset_load_s"] = dt
    print(f"[bench]   dataset load:      {dt:8.3f}s", flush=True)

    dt = bench_experiment_load(state, exp_path)
    result["experiment_load_s"] = dt
    print(f"[bench]   experiment load:   {dt:8.3f}s", flush=True)

    dt, n_charts = bench_stats_all_charts(state)
    result["stats_all_charts_s"] = dt
    result["stats_n_charts"] = n_charts
    print(f"[bench]   stats ({n_charts:2d} charts): {dt:8.3f}s", flush=True)

    dt, gallery_view = bench_gallery_filters(state)
    result["gallery_filters_s"] = dt
    print(f"[bench]   gallery filters:   {dt:8.3f}s", flush=True)

    dt = bench_gallery_grid(gallery_view)
    result["gallery_grid_page_s"] = dt
    print(f"[bench]   gallery grid page: {dt:8.3f}s", flush=True)

    dt = bench_metrics(state)
    result["metrics_s"] = dt
    print(f"[bench]   metrics:           {dt:8.3f}s", flush=True)

    if not keep_csv:
        ds_path.unlink(missing_ok=True)
        exp_path.unlink(missing_ok=True)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "lookin_benchmark_data",
                         help="Directory for generated CSVs and results.json (outside the repo by default)")
    parser.add_argument("--quick", action="store_true", help="Small sweep, for smoke-testing this script itself")
    parser.add_argument("--keep-csv", action="store_true", help="Don't delete generated CSVs after each variant")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    crop_pool = _crop_pool()
    print(f"[bench] output dir: {args.out}", flush=True)
    print(f"[bench] crop image pool: {len(crop_pool)} real images from {EXAMPLE_CROPS_DIR}", flush=True)

    row_sweep = QUICK_ROW_SWEEP if args.quick else ROW_SWEEP
    row_sweep_cols = 10 if args.quick else ROW_SWEEP_FIXED_COLS
    col_sweep = QUICK_COL_SWEEP if args.quick else COL_SWEEP
    col_sweep_rows = QUICK_FIXED_ROWS if args.quick else COL_SWEEP_FIXED_ROWS

    results = {"row_sweep": [], "col_sweep": [], "projection_sweep": [],
               "row_sweep_fixed_cols": row_sweep_cols, "col_sweep_fixed_rows": col_sweep_rows}
    results_path = args.out / "results.json"

    for n_rows in row_sweep:
        label = f"rows_{n_rows}"
        r = run_variant(label, args.out, n_rows, row_sweep_cols, crop_pool, args.keep_csv)
        results["row_sweep"].append(r)
        results_path.write_text(json.dumps(results, indent=2))

    for n_cols in col_sweep:
        label = f"cols_{n_cols}"
        r = run_variant(label, args.out, col_sweep_rows, n_cols, crop_pool, args.keep_csv)
        results["col_sweep"].append(r)
        results_path.write_text(json.dumps(results, indent=2))

    for n_points in (QUICK_POINT_SWEEP if args.quick else POINT_SWEEP):
        r = bench_projection_frame(n_points)
        results["projection_sweep"].append(r)
        print(f"[bench] === projection map: {n_points:,} points ===", flush=True)
        for kind in ("full", "draft"):
            print(f"[bench]   {kind:5s} frame: build {r[f'{kind}_build_s'] * 1000:6.1f}ms + "
                  f"json {r[f'{kind}_json_s'] * 1000:6.1f}ms, {r[f'{kind}_kb']:7.0f} KB "
                  f"({r[f'{kind}_points']:,} points drawn)", flush=True)
        results_path.write_text(json.dumps(results, indent=2))

    print(f"[bench] done. results written to {results_path}", flush=True)


if __name__ == "__main__":
    main()
