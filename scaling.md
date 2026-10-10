# Scaling benchmark

How each tab's core operation scales with dataset size (0 to 1,000,000 rows)
and metadata-column count (0 to 50 columns), plus what one frame of the
Embeddings tab's interactive map costs (1,500 to 200,000 points) -- measured
by calling lookin's own production code directly (`AppState`, `StatsView`,
`GalleryView`, `MetricsView`, `InteractiveProjection`) rather than a
reimplementation, see
[scripts/benchmark_scaling.py](scripts/benchmark_scaling.py).

## Method

- Two independent sweeps rather than a full row x column grid (which would
  need 42 dataset variants instead of 13): a **row sweep** at a fixed 20
  metadata columns, and a **column sweep** at a fixed 200,000 rows. A third,
  separate **point sweep** covers the Embeddings tab's interactive map,
  which is sized by how many points are drawn rather than by the dataset
  behind them.
- The per-frame numbers for that map are the exception to "computation
  only" below: a frame is only useful once it has reached the browser, so
  the serialized payload it puts on the websocket is measured too.
- Every operation measured is the actual code the app runs, called through a
  minimal fake context (no live browser) -- timings below are computation
  only, not network/render round-trip.
- Statistics and Gallery are opt-in (see below) -- nothing renders until the
  user adds a chart or filter. The numbers here are the **worst case**: every
  available column added as a chart / a filter at once. Real usage is
  whatever subset the user actually picks, so in practice these are ceilings,
  not typical timings.
- Metrics timing is one experiment's confusion matrix + per-class metrics +
  macro ROC/PR curves (`f1` metric, macro-average curve target) against the
  merged dataset+experiment data.
- Gallery grid timing is a single page render (default 4x6 = 24 images) with
  the thumbnail cache cleared first, so it's always a cold read -- this is
  what pagination is supposed to buy: page cost independent of dataset size.
- Generated CSVs (up to ~330 MB at the high end) are **not** stored in this
  repo -- `benchmark_scaling.py` writes them to an external directory
  (default: a sibling of the repo) and deletes each variant's files right
  after using them, so re-running this leaves nothing behind to accidentally
  `git add` or copy. Only the bundled example dataset stays under `src/`.
- Machine: 12 cores, 31 GB RAM, Linux. flet 0.28.3, pandas 3.0.5, numpy
  2.5.1, scikit-learn 1.9.0. Single run, not averaged -- treat as
  order-of-magnitude, not a precise benchmark.

Reproduce with:
```bash
uv run python scripts/benchmark_scaling.py --out /some/path/outside/the/repo
```

## Row sweep (fixed at 20 metadata columns)

| Rows | Dataset load | Experiment load | Stats, all 21 charts | Gallery, all 21 filters | Gallery grid page | Metrics |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.002s | 0.001s | 1.52s | 0.021s | 0.002s | 0.25s |
| 1,000 | 0.007s | 0.005s | 1.90s | 0.040s | 0.006s | 0.37s |
| 10,000 | 0.026s | 0.014s | 1.95s | 0.119s | 0.007s | 0.47s |
| 50,000 | 0.111s | 0.095s | 2.03s | 0.506s | 0.007s | 1.15s |
| 100,000 | 0.231s | 0.169s | 2.06s | 0.891s | 0.007s | 1.89s |
| 500,000 | 1.06s | 0.81s | 3.08s | 4.65s | 0.007s | 9.38s |
| 1,000,000 | 2.02s | 1.38s | 3.74s | 10.26s | 0.009s | 19.27s |

## Column sweep (fixed at 200,000 rows)

| Metadata columns | Dataset load | Experiment load | Stats, all charts | Gallery, all filters | Gallery grid page | Metrics |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.114s | 0.286s | 0.12s | 0.011s | 0.003s | 3.52s |
| 5 | 0.168s | 0.279s | 0.62s | 0.176s | 0.004s | 3.73s |
| 10 | 0.241s | 0.294s | 1.26s | 0.576s | 0.007s | 3.80s |
| 20 | 0.392s | 0.322s | 2.38s | 1.86s | 0.006s | 3.76s |
| 35 | 0.605s | 0.282s | 3.81s | 5.22s | 0.009s | 3.77s |
| 50 | 0.828s | 0.284s | 5.26s | 10.26s | 0.012s | 3.86s |

(Experiment load doesn't depend on dataset metadata columns -- the ~0.28-0.32s
there is noise plus the fixed 200,000-row experiment CSV's own read cost.)

## Per-tab findings

**Data tab (dataset/experiment load)** -- scales linearly with rows x
columns, as expected for CSV parsing + dtype downcasting: ~2s to load 1M
rows x 20 columns, ~0.8s for 200k rows x 50 columns. Comfortable up to at
least 1M rows; the chunked reader with progress callback (added earlier)
keeps this responsive rather than blocking silently.

**Statistics tab** -- the worst case (every column charted) grows
roughly linearly with both rows and columns, but the practical scaling
knob is *how many charts the user adds*, not dataset size: since charts
are opt-in with nothing shown by default, a user looking at 3 columns on a
1M-row dataset pays roughly 3/21 of the 3.74s row-sweep number, not the
full amount. The ~1.5s floor even at 0 rows is matplotlib's own per-figure
setup cost (one `plt.subplots()` + `savefig()` per chart), not data-driven.

**Gallery tab** -- two very different costs:
- The grid (paginated mosaic) is **flat**: 0.002-0.012s regardless of
  1 row or 1,000,000 rows, 0 columns or 50 columns. This is the point of
  pagination -- browsing is never slower on a huge dataset than on a small
  one.
- The filter panel, in the worst case (every column added as a filter), is
  the single most expensive thing measured: 10.26s at 1M rows and,
  independently, 10.26s at 50 columns on 200k rows. Building a categorical
  filter means scanning and sorting every unique value in that column
  across the whole dataset, so cost scales with rows x active filter
  columns. In practice this is bounded by how many filters the user
  actually opts into (default: zero), and by the
  `HIGH_CARDINALITY_MAX_UNIQUE` cap that already keeps unbounded-cardinality
  columns out of the picker entirely -- but a user who adds all 35+ columns
  as filters on a million-row dataset should expect several seconds, not
  milliseconds.

**Metrics tab** -- scales with rows only (~19s at 1M rows, ~0.25s at 0
rows), completely flat across the column sweep (3.5-3.9s regardless of 0
vs 50 metadata columns) since metrics only touch the label and score
columns, never dataset metadata. This is the second most expensive
operation at the high end, driven by the dataset<->experiment merge plus
sklearn's confusion matrix / per-class metrics / ROC+PR curve computation
over up to 1M rows.

**Embeddings tab, interactive map** -- the one cost that is paid *while the
user's hand is moving*, so it is measured per frame rather than per click.
Sized by points on the map (the tab's "Sample size", default 1,500), not by
dataset rows: a million-row dataset drawn at that default costs what 1,500
points cost.

| Points on the map | Full frame: build | + serialize | payload | Draft frame: build | + serialize | payload |
|---:|---:|---:|---:|---:|---:|---:|
| 1,500 | 1.1ms | 0.7ms | 23 KB | 0.6ms | 0.7ms | 23 KB |
| 10,000 | 3.0ms | 4.6ms | 156 KB | 2.9ms | 4.6ms | 156 KB |
| 50,000 | 13.4ms | 27.5ms | 781 KB | 5.6ms | 4.8ms | 156 KB |
| 200,000 | 107.6ms | 93.4ms | 3,125 KB | 11.8ms | 4.7ms | 156 KB |

A drag samples the pointer every 40ms (`DRAG_INTERVAL_MS`) whatever a frame
costs, and Flet runs every event handler on its own pool thread -- so a
50,000-point map used to start a new ~130ms redraw (measured in the running
app, not in the table above) every 40ms, each one re-serializing three
quarters of a megabyte. They overlapped, they queued, and the map fell
further behind the cursor the longer the drag lasted. Three things changed:

- the coordinates are built by numpy in one pass instead of a Python
  comprehension per point (56ms -> 13ms of the 50,000-point full frame);
- while a gesture is still moving the map draws a **draft** -- every k-th
  point, capped at `DRAFT_MAX_POINTS` -- so the cost of dragging stops
  growing with the sample size at all, and the full cloud is drawn once the
  gesture stops (`_RedrawPump`);
- a frame identical to the one already on screen is not drawn. Panning at
  the fitted zoom is clamped to no movement, so every one of those 130ms
  redraws was redrawing the same picture.

Measured in the running app at 50,000 points, before and after: a frame
mid-drag went from ~130ms to ~11ms, and the full-detail frame that lands
when the drag stops from ~130ms to ~45ms.

## Why none of this blocks the UI in practice

All five of these costs are the reason for five changes made in this
project:
1. **Opt-in charts/filters** (Statistics, Gallery) -- the worst-case numbers
   above require the user to deliberately add every single column; the
   default is zero charts and zero filters, i.e. near-instant.
2. **Explicit "Update now"** (Statistics, Gallery, Metrics) -- none of these
   costs are ever paid automatically on a tab switch or a data change
   anymore; a red banner appears and the (possibly multi-second) recompute
   only happens when the user clicks it.
3. **Off the UI thread** (Statistics, Gallery, Metrics) -- the rebuild the
   user asked for runs on a worker thread (`components.BackgroundRebuild`),
   so the rest of the app keeps responding and the per-phase progress each
   of these views already reported can finally be seen. Before this, the
   user had chosen to pay the cost but had no way to tell it was being
   paid: the whole app was frozen for the duration.
4. **Rationed frames** (Embeddings) -- the interactive map cannot put its
   cost behind a button, because the cost *is* the interaction. It instead
   draws at most one frame at a time, starts the next one no sooner than
   the last one took, and thins the cloud until the gesture ends (see the
   Embeddings section above).
5. **One busy light for the whole app** (`components.BusyIndicator`) --
   every tab's progress banner is invisible from every other tab, so a user
   who started a 600,000-row load on the Data tab and went to look at
   Metrics had no way to tell whether the app was still reading, had
   finished, or had failed: Metrics looks the same either way. Both CSV
   loads and every `BackgroundRebuild` now light a named spinner in the app
   bar, which every tab shares.

Reading a dataset or results CSV on the Data tab is still synchronous, and
an earlier version of this document claimed its progress bar therefore
could not paint. That turns out to be wrong, and it is worth writing down
why: Flet dispatches every sync event handler to a thread pool
(`Page.run_thread`), so the load already runs off the event loop and its
per-chunk `.update()` calls do reach the browser. Watched in the running
app, a 600,000-row load streams "50,000 / 600,000 rows", "150,000 /
600,000 rows", then "Finalizing", and the tabs stay clickable throughout.

What was actually missing was that none of that is visible from anywhere
except the Data tab -- hence the app-bar indicator above, rather than a
rewrite of the load. Moving the read to a worker thread would still be a
change to a contract, not just a thread: `add_experiment_from_path()`
returns its success to three callers that expect an answer straight away
(the Data tab, `--experiment` at startup, and the MLflow panel).

So the numbers in this document are the ceiling a user can hit by opting
into everything at once on a million-row dataset -- multiple seconds, never
minutes, and always by explicit choice rather than an automatic, unexplained
freeze.


## Home, MLflow and the workspace library

Measured 2026-09-25 against a real MLflow 3.16 server (sqlite backend, 4
workers) holding **100 experiments x 100 runs**, each run with 60 params and
15 metrics, plus a library of **300 workspaces** (4 MLflow experiments and 5
snapshots each) and a **300,000-row** dataset with four 300,000-row results
CSVs. Same machine as above. The scripts and generated data lived outside
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
- **Loading a workspace** is the data tab's own cost (see the row sweep
  above): 0.55 s for the 300k-row dataset, ~0.25 s per 300k-row results
  CSV, 0.44 s for the first merge.
- **Downloading** a 35 MB results CSV took 0.7 s from a local server; on a
  real network it is the link that decides.

Unchanged and already fast: listing 300 workspaces (23 ms), the board file
(< 1 ms), a safe read-change-write of a manifest (0.4 ms), listing 101
experiments (24 ms), hashing a 70 MB dataset (40 ms).
