# Scaling benchmark

How each tab's core operation scales with dataset size (0 to 1,000,000 rows)
and metadata-column count (0 to 50 columns) -- measured by calling Lookin's
own production code directly (`AppState`, and the Statistics, Gallery and
Metrics views of a `lookin web` session) rather than a reimplementation, see
[scripts/benchmark_scaling.py](scripts/benchmark_scaling.py).

## Method

- Two independent sweeps rather than a full row x column grid (which would
  need 42 dataset variants instead of 13): a **row sweep** at a fixed 20
  metadata columns, and a **column sweep** at a fixed 200,000 rows.
- Every operation measured is the code the server runs for a request, on a
  session with no browser attached -- timings are computation and HTML
  rendering, not the network round-trip or the browser's own layout.
- Statistics and Gallery are opt-in (see below) -- nothing renders until the
  user adds a chart or filter. The numbers here are the **worst case**: every
  available column added as a chart / a filter at once. Real usage is
  whatever subset the user actually picks, so in practice these are ceilings,
  not typical timings.
- Metrics timing is one experiment's confusion matrix + per-class metrics +
  macro ROC/PR curves against the merged dataset+experiment data, and the
  HTML of the tab.
- Gallery grid timing is the mosaic's first page: filtering the frame, the
  page's HTML, then reading and encoding its first 60 crops with the crop
  cache cleared first, so it is always a cold read.
- Generated CSVs (up to ~330 MB at the high end) are **not** stored in this
  repo -- `benchmark_scaling.py` writes them to an external directory
  (default: a sibling of the repo) and deletes each variant's files right
  after using them, so re-running this leaves nothing behind to accidentally
  `git add` or copy. Only the bundled example dataset stays under `src/`.
- Machine: 4 cores, 15 GB RAM, Linux. pandas 3.0.5, numpy 2.5.1,
  scikit-learn 1.9.0, matplotlib 3.11.1. Single run, not averaged -- treat
  as order-of-magnitude, not a precise benchmark.

Reproduce with:
```bash
uv run python scripts/benchmark_scaling.py --out /some/path/outside/the/repo
```

## Row sweep (fixed at 20 metadata columns)

| Rows | Dataset load | Experiment load | Stats, all 21 charts | Gallery, all 21 filters | Gallery grid page | Metrics |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.012s | 0.003s | 3.55s | 0.020s | 0.009s | 0.64s |
| 1,000 | 0.065s | 0.039s | 5.22s | 0.034s | 0.031s | 0.36s |
| 10,000 | 0.12s | 0.053s | 4.84s | 0.12s | 0.041s | 0.42s |
| 50,000 | 0.19s | 0.11s | 5.01s | 0.64s | 0.089s | 0.59s |
| 100,000 | 0.35s | 0.20s | 4.67s | 1.09s | 0.15s | 0.86s |
| 500,000 | 1.43s | 0.96s | 6.09s | 5.07s | 0.72s | 3.61s |
| 1,000,000 | 3.86s | 2.42s | 6.91s | 10.07s | 1.41s | 6.87s |

## Column sweep (fixed at 200,000 rows)

| Metadata columns | Dataset load | Experiment load | Stats, all charts | Gallery, all filters | Gallery grid page | Metrics |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.30s | 0.54s | 0.30s | 0.014s | 0.022s | 1.22s |
| 5 | 0.35s | 0.52s | 2.29s | 0.26s | 0.091s | 1.24s |
| 10 | 0.43s | 0.54s | 2.67s | 0.63s | 0.19s | 1.82s |
| 20 | 0.62s | 0.57s | 5.25s | 2.15s | 0.34s | 1.71s |
| 35 | 0.89s | 0.49s | 8.47s | 6.54s | 0.51s | 1.85s |
| 50 | 1.69s | 0.43s | 13.56s | 12.97s | 0.80s | 2.46s |

(Experiment load doesn't depend on dataset metadata columns -- the ~0.4-0.6s
there is noise plus the fixed 200,000-row experiment CSV's own read cost.)

## Per-tab findings

**Sources tab (dataset/experiment load)** -- scales linearly with rows x
columns, as expected for CSV parsing + dtype downcasting: ~4s to load 1M
rows x 20 columns, ~1.7s for 200k rows x 50 columns. The chunked reader
reports its progress as it goes, so a long load is visibly a load.

**Statistics tab** -- the worst case (every column charted) grows with the
number of charts far more than with rows: one chart per column, each a
matplotlib figure rendered to PNG. Since charts are opt-in with nothing
shown by default, a user looking at 3 columns on a 1M-row dataset pays
roughly 3/21 of the 6.9s row-sweep number. The ~3.5s floor even at 0 rows
is matplotlib's own per-figure cost, not data-driven.

**Gallery tab** -- two very different costs:
- The mosaic's first page: the crops themselves cost the same whatever the
  dataset (60 cold reads), and what grows with rows is the pass that
  filters the frame before the page is cut -- 1.4s at 1M rows, 0.8s at
  50 columns on 200k rows. Every later page is a slice of that frame.
- The filter panel, in the worst case (every column added as a filter), is
  the single most expensive thing measured: ~10s at 1M rows and ~13s at 50
  columns on 200k rows. Building a categorical filter means scanning and
  sorting every unique value in that column across the whole dataset, so
  cost scales with rows x active filter columns. In practice this is
  bounded by how many filters the user actually opts into (default: zero),
  and by the `HIGH_CARDINALITY_MAX_UNIQUE` cap that keeps
  unbounded-cardinality columns out of the picker entirely.

**Metrics tab** -- scales with rows (~6.9s at 1M rows), and stays within
1.2-2.5s across the column sweep since metrics only touch the label and
score columns, never dataset metadata. At the high end the cost is the
dataset<->experiment merge plus the confusion matrix / per-class metrics /
ROC+PR curves over every row.

## Why none of this blocks the interface

1. **Opt-in charts/filters** (Statistics, Gallery) -- the worst-case numbers
   above require the user to deliberately add every single column; the
   default is zero charts and zero filters, i.e. near-instant.
2. **Explicit "Update now"** -- none of these costs are paid automatically
   on a tab switch or a data change: a red banner appears and the
   recompute only happens when the user clicks it.
3. **Off the request thread** -- every rebuild runs on its view's job
   (`web/session.py` `Job`): one at a time per view, the latest request
   coalesced, behind one process-wide lock so three CPU-bound rebuilds do
   not slow each other down. The page stays responsive and shows the
   progress each view reports.
4. **One busy light per session** (`Session.busy`) -- every running job
   is named in the top bar, so a user who started a 600,000-row load and
   went to look at Metrics can tell whether it is still reading, has
   finished, or has failed.
5. **One tab, one session** -- a heavy rebuild in one browser tab never
   delays another tab's requests beyond the shared lock above.

So the numbers in this document are the ceiling a user can hit by opting
into everything at once on a million-row dataset -- seconds, never
minutes, and always by explicit choice rather than an unexplained freeze.


## Home, MLflow and the workspace library

Measured 2026-09-25 against a real MLflow 3.16 server (sqlite backend, 4
workers) holding **100 experiments x 100 runs**, each run with 60 params and
15 metrics, plus a library of **300 workspaces** (4 MLflow experiments and 5
snapshots each) and a **300,000-row** dataset with four 300,000-row results
CSVs, on a 12-core machine. The scripts and generated data lived outside
the repo; the numbers are what the Home page's own code paths took.

### What was slow, and what changed

| Operation | Before | After | What changed |
|---|---:|---:|---|
| Home first paint (300 workspaces, cached 10k-run listing) | ~20 s | < 2 s | Lists are paged: 25 cards per board column and 100 run rows, then "Show more". The browser builds every control it is sent before painting, and Home had been 11,000 of them (the server needed 0.25 s to send them). |
| Show one more experiment (100 shown) | ~25 s | 0.3 s | Runs are cached per experiment; showing one fetches only it. |
| Hide an experiment | ~25 s | 0.2 s | Nothing is fetched; its runs are dropped locally. |
| List every run of 100 experiments (10k runs) | 24.7 s | 10.9 s, first rows at 0.3 s | One request per experiment, 4 at a time (`parallel_requests` in home_view.json), drawn as each arrives. |
| Open Home again | blank until listed | cached listing at once, refreshed behind it | Last listing kept per server, shown first. |
| Expand a run | 0.19 s + 0.65 s cache rewrite | 0.19 s | Run -> CSV lists cached in their own small file, not in the 34 MB runs listing. |
| Board redraw (300 workspaces) | 510 ms | 207 ms | Download state computed in one pass without pathlib; expanding a card or a download tick redraws that card only. |
| Open a 300k-row workspace in a new tab | ~12 s | ~5 s | A tab's Home talks to MLflow only once it is shown -- it had been listing every shown experiment in the background, tripling CSV read times and loading the tracking server once more per open tab. |
| 100 experiment chips | filled the runs panel | 8 chips + "Experiments" picker | Choosing experiments is a searchable picker; applying it fetches only what was newly ticked. |

### What is now bound by something else

- **MLflow itself.** A `runs/search` page of 1,000 runs with 60 params each
  is 9.7 MB and takes the server 2.4 s to build; parsing it takes Lookin
  under 0.1 s. The REST API cannot return a subset of params, so the only
  lever is concurrency: 4 parallel requests were 3.3x faster than 1 on a
  4-worker server, and 16 were slower than 4. Match `parallel_requests` to
  the server's worker count.
- **Loading a workspace** is the Sources tab's own cost (see the row sweep
  above): 0.55 s for the 300k-row dataset, ~0.25 s per 300k-row results
  CSV, 0.44 s for the first merge.
- **Downloading** a 35 MB results CSV took 0.7 s from a local server; on a
  real network it is the link that decides.

Unchanged and already fast: listing 300 workspaces (23 ms), the board file
(< 1 ms), a safe read-change-write of a manifest (0.4 ms), listing 101
experiments (24 ms), hashing a 70 MB dataset (40 ms).
