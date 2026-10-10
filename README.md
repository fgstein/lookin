# Lookin

A visual explorer for deep-learning classification results: dataset
statistics, a filterable image gallery, and cross-experiment metrics
(confusion matrices, ROC / PR curves) -- organised in workspaces built from
your [MLflow](https://mlflow.org) runs.

## Install & run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv tool install .
lookin ui
```

This opens the web interface at `http://127.0.0.1:8550`. Use `--host` /
`--port` to change where it binds:

```bash
lookin ui --host 0.0.0.0 --port 8080
```

Point it at data on startup instead of typing/browsing for it in the UI,
with `--dataset` and (optionally, repeatable) `--experiment NAME=PATH`:

```bash
lookin ui --dataset /path/to/dataset.csv \
  --experiment baseline=/path/to/results_baseline.csv \
  --experiment finetuned=/path/to/results_finetuned.csv
```

Without installing, you can also run it directly from the project directory:

```bash
uv run lookin ui
```

## What it does

1. **Data tab** -- load a dataset CSV (one row per cropped image: an id,
   a path to the crop, a label, and any number of metadata columns;
   the three roles are guessed from the file's own values -- an id is
   uuid-shaped, a crop is a path, a label is a short string -- and each is
   a dropdown of that file's columns, so a wrong guess is one click to fix;
   crops can be PNG, JPEG, or TIFF, including 16-bit TIFF, which gets a
   per-image min/max normalization to 8-bit for display -- see the display
   LUTs below) and,
   independently, one or more results CSVs as named "experiments". A
   results CSV needs an id column plus one score column per class, named
   after the class itself (e.g. `cat, dog, bird, ...`) -- class names are
   read directly from those column headers. An optional `label` column
   (an already-computed argmax prediction) is recognized and ignored as a
   score. Column names, per-experiment class mappings, and dataset
   relabeling are all editable here. A dataset alone is enough to use
   Statistics and the Gallery; Metrics needs at least one loaded
   experiment. A third, optional kind of file loads here too -- a **simple
   inference**: crops that are *not* in the dataset (a batch of new
   acquisitions, images from another source), as a CSV with an id column
   and a path to each image, plus an embeddings `.npz`. No label is
   expected. See the Embeddings tab below.
2. **Statistics tab** -- a histogram or value-count chart for every column.
3. **Gallery tab** -- a filterable, paginated image mosaic (label,
   categorical/binary metadata checkboxes, numeric range sliders), with
   adjustable rows x columns. Click an image for a full-resolution,
   paginated carousel view with all of its metadata.
4. **Metrics tab** -- per-class precision/recall/F1, ROC and
   Precision-Recall curves, and clickable confusion matrices (a cell opens
   the image carousel for that exact true/predicted combination) --
   overlaid across every experiment you select, for direct comparison.
   Plus a **breakdown by metadata**: pick any metadata column and every
   accuracy is recomputed per slice of it -- per sensor, per site, per crop
   size -- so "the model is at 94%" becomes "94%, except 61% on this
   sensor". See below.
5. **Calibration tab** -- whether the confidence can be believed, and what
   refusing to answer buys: reliability diagrams, ECE/MCE, the
   risk-coverage curve, a threshold explorer that runs both directions, and
   temperature scaling. See below.
6. **Compare tab** -- two experiments head to head: how many crops each
   one broke and fixed, a differential confusion matrix (candidate minus
   baseline, coloured by whether the change is good news rather than by
   the sign of the count), a per-class breakdown, and the regressed and
   fixed crops themselves, most-confident first, openable in the image
   carousel. The accuracy delta only shows the net; a model can gain a
   point overall while breaking hundreds of crops it used to get right.
7. **Embeddings tab** -- a 2D projection of an experiment's embeddings,
   plus a class-overlap matrix: mean cosine similarity within and between
   classes, measured on the crops themselves. The confusion matrix says the
   model mixes two classes up; this says whether they genuinely sit on top
   of each other, which is the difference between a model that needs more
   training and a taxonomy that needs rewriting. Simple inferences loaded
   on the Data tab can be overlaid on the same map -- see below.
8. **Cleaning tab** -- find the crops worth a human look, decide on them,
   and export a versioned dataset you can train on. See below.

**Display LUTs.** Greyscale crops often use a narrow slice of the available
range and read as a flat grey square. The **Display** control in the top bar
maps them to the screen: none, min/max stretch, a 2-98% contrast stretch
(the usual choice, since it ignores outliers), or histogram equalisation.
It is one session-wide setting, so two views of the same crop never disagree
about what it looks like -- and it is display only: the files are never
modified, and no metric, embedding or prototype is computed through it. The
bundled example is greyscale for this reason.

**Roll correction.** If a column of the dataset records the roll each crop
was cut at, the same **Display** control turns every crop by the nearest
quarter turn that cancels it, so a page of thumbnails shares one
orientation instead of showing the same object four ways. Values are read
as radians, and only whole quarter turns are ever applied: a crop rolled by
0.3 rad is left as stored, and no pixel is resampled -- a quarter turn
moves pixels without touching their values. The turn always runs against
the roll, which is what cancels it. Display only, in the same sense as the
LUTs above.

**Relabelling.** Every carousel in the app carries the review controls --
the gallery, a confusion-matrix cell, the Compare tab's regression lists,
the similarity results, the embedding lasso, the prototype drill-downs and
the cleaning queue. A crop you can see is a crop you can relabel. Each
records where the decision came from, which is what makes "60% of our
exclusions came from the confusion matrix" answerable.

Anywhere a crop is shown, **Find similar** opens its nearest neighbours in
an experiment's embedding space, and reports how many of them share its
label -- a crop whose neighbours are mostly another class is either
mislabelled or genuinely ambiguous.

**Simple inferences: crops that are not in the dataset.** Load a CSV of
image paths and an embeddings `.npz` from the same model (Data tab, section
3), and the Embeddings tab can draw those crops on the same map as the
dataset's own, as squares. They carry no label and never enter a metric,
the confusion matrix, the cleaning queue or the review -- the tool has no
ground truth for them and does not pretend to. Everything about *looking*
at them works exactly as for a dataset crop: hover for the image, click or
lasso to open them in the carousel, and Find similar to ask what in the
dataset the model thinks they resemble. Whatever other columns their CSV
carries (a confidence, a sensor, even class scores) can color the map, and
that is all those columns are ever used for.

**Colouring and filtering are two different questions.** "Color by" paints
the cloud with one column; the **Filter points** menu hides points on as
many columns as you like, and the two have nothing to do with each other --
colour by confidence while hiding one sensor and every blurry crop. Pick
the columns you want to filter on and only those get a panel, so the column
never costs screen space until you ask for it. Hiding points is a redraw,
never another projection: what is left is exactly where it always was, and
a line above the map says how many went. The legend keeps its own quick
filter on the colour column, and the two combine.

**A number is a scale, or it is a set of labels.** A float column with a
handful of distinct values (a roll angle, a quantised score) is a
measurement to one person and eight categories to another, and nothing in
the data says which -- so a **Continuous / Discrete** switch sits beside
"Color by" (and on that column's filter panel) and settles it. Continuous
gives a viridis ramp between the column's own minimum and maximum and a
colorbar in its own units; discrete gives one colour per value. The choice
follows the column: it drives the colouring *and* the shape of that
column's filter (a window over a range, or a box per value).

By default the map does not move: the projection keeps the model it was
fitted on and the new crops are placed into it, so the cloud you were
reading stays where it was. Ticking **Refit the map on these points too**
fits everything together instead, which moves every point -- and with
t-SNE that is the only option, since t-SNE learns no map that can place a
point it was not fitted on. A line under the controls always says which of
the two you are looking at. Color by **Source** to see the two pools apart
at a glance.

## The (i) buttons: a built-in reference manual

Every section that shows a number, a curve or a table carries a small **(i)**
on its right. It opens one window -- the reference manual -- at the page for
what you were looking at, with the table of contents of the whole manual
down the side, so the neighbouring pages are one click away. The book icon
in the app bar opens the same window at its front page, and the search box
filters the contents.

A page starts with a two-sentence summary, then gives the definition with
the equation the app actually computes and every symbol in it named, then
"how to read it" for the chart or table it belongs to, then the ways that
particular number lies. Its pages cover the metrics (accuracy through MCC),
the curves, calibration and abstention, slices, experiment comparison, the
embedding space, prototypes and the cleaning strategies.

Because the manual holds the explanations, the tabs themselves stay short: a
section carries one line saying what it is or how to drive it, and the theory
lives one click away behind its (i).

Equations are typeset with matplotlib's mathtext engine and drawn as images,
in the same serif face as the charts -- there is no MathJax, no web view and
nothing fetched (see *Runs fully offline*). The content lives in
`src/lookin/core/manual/`, one module per chapter, as data rather than
markup; `src/lookin/ui/manual_view.py` draws it.

## Where the average comes from: slicing by metadata

An accuracy is an average, and an average over a dataset that mixes sensors,
sites, acquisition times and crop sizes hides the thing you are looking for.
The per-class table already splits that average by *label*; the **Break down
by metadata** panel at the bottom of the Metrics tab splits it by anything
else the dataset CSV carries.

Pick a column and every selected experiment is scored again on each slice of
it, side by side, ranked worst-first. The chart puts two things on two axes
because they answer each other: grey bars are how many crops each slice
holds, and a marked line per experiment is the metric, with each experiment's
overall figure as a dashed reference line. Reading a 0.42 without seeing that
it rests on nine crops is the mistake that arrangement makes hard -- and on a
numeric column ordered by value, the line is a real trend (accuracy against
crop size, say) rather than a ranking.

A categorical column gets one slice per value; a numeric one is cut
into quantile bins (equal-width bins on a skewed column put 99% of the rows
in the first one and three rows in the last, and that last bin's "40%"
would be noise read as a finding). Rows where the column is null get their
own slice rather than being dropped -- "it fails where this metadata is
absent" is an answer, not a gap. Clicking a slice opens its errors in the
usual carousel, so the crops that explain a bad slice are one click from
noticing it.

Two guard-rails, because slicing turns one number into twenty and the worst
of twenty is low even when nothing is wrong:

- every accuracy carries its **Wilson score interval**, so a twelve-point
  gap between two forty-row slices reads as what it is. Wilson rather than
  the textbook normal interval because the normal one is actively wrong in
  exactly this regime -- small n, proportion near 1, upper bounds above 1.
- slices under a **row floor** (30 by default, adjustable) are shown greyed
  and never ranked first. A four-crop slice at 50% is not the worst slice,
  it is an unmeasured one, and letting it head the table sends you to look
  at four images while a real regression sits three rows down.

Each row also reports how many classes the slice actually holds -- the
number that says whether comparing its accuracy to another slice's is
comparing like with like.

Nothing is computed until a column is picked, and picking one does not
recompute the confusion matrices or the curves.

## Calibration and abstention

The Metrics tab answers "how often is it right". The **Calibration** tab
answers the two questions that come next:

**Can the confidence be believed?** A reliability diagram per experiment
(accuracy against confidence, per bin, with the same binning the ECE/MCE
columns in the Metrics summary use), a confidence gap in one number (mean
confidence minus accuracy: positive is over-confident), and a histogram of
where the correct and the wrong predictions sit on the confidence axis --
the picture that says whether a threshold can work at all.

**What does refusing to answer buy?** Accept a prediction above a
threshold, send the rest to a human. The **risk-coverage curve** sweeps
every threshold at once: coverage on the x axis, error rate among the
accepted on the y. Its area (AURC) is a single comparable number for how
well a model knows what it doesn't know -- and since it also falls when a
model simply makes fewer errors, the table reports **excess AURC** beside
it: what this model costs above a model of the same accuracy that ranked
every one of its mistakes last. That column is the one that isolates the
ranking.

The threshold explorer runs both directions. From a threshold: how many
crops are accepted, at what accuracy, how many errors slip through, how
many land in the human queue -- and buttons opening either of those two
sets in the carousel, since "the errors this would let through" is worth
looking at before committing to a number. From a target: the most permissive
threshold that still reaches, say, 99% accuracy on what it accepts, and the
coverage that buys. Targets nothing can reach are reported as unreachable
rather than as a plausible number.

**One threshold per class** is available too, for the usual case where the
classes are not equally hard: a single global cut-off that makes the
hardest class safe throws away most of an easy majority class for nothing.

**Temperature scaling** fits the standard post-hoc fix -- one scalar divided
into the logits -- and reports the log loss and ECE before and after it.
Dividing by a positive scalar cannot move an argmax, so accuracy, the
confusion matrix and every ranking metric are unchanged; only the confidence
values move. Two caveats are shown next to the result rather than buried:
the fit reads the same rows the ECE is measured on, so the improvement is
optimistic (a temperature meant for production belongs on a held-out split),
and the fit minimises log loss, not ECE, so the tab says explicitly when the
two disagree instead of implying an improvement it did not check.

**What survives scores that are not probabilities.** A results CSV whose
rows do not sum to 1 (a sigmoid head, raw logits) disables exactly what it
should. ECE/MCE, the reliability diagram and temperature scaling read the
confidence as a probability, so they are refused with the reason shown. The
risk-coverage curve, every threshold and the accepted/abstained counts only
*rank* by confidence and then count outcomes, so they stay correct and stay
enabled -- the accuracy of what a threshold accepts is a fact about the
rows, whatever the score is called.

## Cleaning and annotation review

Reviewing produces *decisions* (`ok` / `relabel` / `exclude` / `uncertain`,
plus a reason), never edits to your dataset CSV. They are stored next to it
in `<dataset>_review.csv` and `<dataset>_review_versions.csv`, and are
grouped into named **versions**: `v2` declares `v1` as its parent, inherits
everything it decided and stores only what it changes. Exporting for
training freezes the version it exported, so the run that consumed it stays
identifiable afterwards.

**Finding candidates.** Several interchangeable sources feed one review
queue -- score margin and confident errors (results CSV only), consensus
across several experiments, prototype flags and noisy subcenters
(embeddings + the model's real prototypes), and near-duplicates, which is
also how train/test leakage shows up. Embeddings are one method here, not
the price of entry.

**Deciding.** The review controls sit on the shared image carousel, so a
crop can be relabelled from the Gallery, a confusion-matrix cell or the
Cleaning tab alike. For a long queue, the triage panel shows one crop at a
time with reference crops of both classes in question, decided from the
keyboard (`Space`/`K` keep, `R` relabel, `X` exclude, `U` uncertain, `1-9`
relabel to that class, `Z` undo). Bulk actions -- excluding a whole noisy
subcenter at once -- exist only when the pool is marked as **train** on the
Data tab, and are undoable as a single batch.

**Exporting.** The export writes your dataset CSV back out unchanged --
every row, every column, in order -- plus two columns per version:

| column | contents |
| --- | --- |
| `label_review_v2` | the corrected label; the *original* label for a row nobody reviewed, so the column is complete |
| `review_status_v2` | `unreviewed` / `ok` / `relabel` / `exclude` / `uncertain` |
| `review_reason_v2` | optional |

An export is always a **new** file: writing it over the dataset CSV it was
made from, or over either review sidecar, is refused with an error rather
than done. The dataset is an input -- possibly read-only, possibly the only
copy -- and `<dataset>_review.csv` (the decisions) is one letter away from
`<dataset>_reviewed.csv` (the default export name), which is exactly the
kind of near-miss that loses an afternoon of judgement.

Training then filters `review_status_v2 != "exclude"` and reads
`label_review_v2`. The source CSV is re-read in full, so a row filter on the
Data tab never truncates the export. Loading an exported CSV back into
Lookin picks its versions up again, so the enriched file is a complete
exchange format on its own.

**Comparing two review passes.** A version diff on the Cleaning tab shows
what changed between any two versions -- newly decided, verdict changed,
withdrawn -- with the status transitions and examples. It compares
*resolved* states, so an inherited decision counts as present: what matters
is the verdict a training run would read. Alongside it, "Did it work?"
scores an experiment with and without the review overlay, so a cleaning
pass is measured in metric deltas rather than in crops touched.

**Which version the metrics use.** The overlay switch has a version picker
beside it. By default it follows whichever version you are recording into;
pinning one keeps every metric scored against that version's labels while
you carry on reviewing into another. Comparing one model against v1's labels
and another against v3's, then calling the difference a model improvement,
is the mistake this prevents.

**Resuming.** The triage queue remembers where you were, per dataset and
per version, in `~/.config/lookin/triage.json`. It stores the crop's uuid
rather than a position, since the candidate list is rebuilt every time the
queue opens.

**Exporting a report.** The Metrics tab writes a single self-contained HTML
file: provenance (which CSV, which split, which review version, whether the
reviewed labels were applied), the summary table, the per-class tables and
the confusion matrices, with every image embedded as a data URI. It opens
with no network and no sibling files, and a browser prints it to PDF.
Selecting exactly two experiments adds the head-to-head from the Compare
tab. HTML rather than PDF on purpose: a PDF writer would be a new
dependency, which is what the rest of this project avoids.

## Home: MLflow runs and workspaces

Lookin opens on its **Home** page (the "Home" link in the app bar brings it
back at any time):

- **On the left, your MLflow runs** -- every run of every configured tracking
  server, grouped by experiment, like MLflow's own runs page without the
  charts. Pick which experiments to list, hide the ones and the runs you
  never want to see, search by name or parameter, filter server-side with
  MLflow's own syntax (`metrics.val_acc > 0.9 and params.backbone = 'vit'`),
  and choose which params and metrics are columns. A run expands to the
  results CSVs it logged; every run, experiment and CSV links to the MLflow
  UI.
- **On the right, your workspaces.** A workspace is one dataset and the
  experiments scored on it, plus the snapshots (saved views) made from it.
  Drag a run or a CSV -- or a whole selection -- onto a workspace to add
  it. A run with several CSVs asks which ones. The workspace's dataset is
  taken from the first run added (its
  `datamodule/configs/database/csv_path` parameter, configurable), and a
  run evaluated on another dataset is refused.
- **Adding is not downloading.** A workspace records which run and artifact
  each experiment is; its download button fetches them into
  `~/.cache/lookin/`, and a green dot says everything is here. Opening a
  workspace loads it into the analysis tabs.
- Each run carries a coloured badge for every workspace it is in.

What Home shows is one per-user file, `~/.config/lookin/home_view.json` --
servers, shown and hidden experiments, hidden runs, columns and their
labels, filters, the dataset parameter's name, and `parallel_requests`: how
many run listings are asked of a server at once (default 4 -- set it to
the MLflow server's worker count; see scaling.md). The UI writes it, and
edits made to it by hand (or in the `{}` editor on the page) are picked up
live. When a server is unreachable, the last listing fetched from it is
shown greyed, and workspaces keep working from what was downloaded.

Workspaces live in `~/.config/lookin/library` (or `--library` /
`$LOOKIN_LIBRARY`, e.g. a shared mount). The Workspaces window (the box icon
on Home, or the chips in the app bar) is where one is exported, imported,
renamed and its snapshots managed. Workspaces used to be called
*bundles*; existing ones load as they are.

For a one-off look without a workspace, the Data tab's **Add from MLflow...**
loads a single CSV from a run into the session on screen.

A bundled CIFAR-10-based example dataset loads by default so the tool is
usable immediately.

## Development

```bash
uv sync              # install into a local .venv, including dev deps
uv run lookin ui     # run from source
uv run pytest        # run the test suite
```

Tests cover `src/lookin/core/` -- the pure-computation layer with no Flet
dependency: the versioned review store (cumulative resolution, tombstones,
freezing, and the export/adopt round-trip), the metrics (including the
hand-written Brier decomposition and calibration error, checked against
their defining identities), the slice machinery (quantile bins that stay
comparable across experiments, the Wilson interval, the small-slice guard),
selective prediction (that the two threshold queries agree with each other,
including on tied confidences, and that temperature scaling never moves an
argmax), and the data model (column typing, the dataset<->experiment join,
and its cache invalidation).

A few tests render a view instead: the slice panel and the Calibration tab
are built against an `AppState` and a context with no Flet page, which is
enough to check the wiring those two add on top of `core/` -- every
`.update()` in a view is already guarded by `if control.page:`, so a
pageless context renders the whole thing and skips the pushes.

None of it needs a browser, a server or the bundled data -- everything is
built in `tmp_path`.

Regenerate the bundled example data (`src/lookin/data/example/`):

```bash
uv run python scripts/generate_example.py
```

Generate a larger synthetic dataset for scale testing (reuses the example
crop images under fresh uuids -- not meant to be committed):

```bash
uv run python scripts/generate_stress_test.py --rows 120000 --experiments 5
```

### Project layout

```
src/lookin/
  app.py            Flet app entry point (run_ui)
  cli.py            `lookin` command (argparse)
  core/             data model, metrics, MLflow REST client -- no UI deps
  core/manual/      the reference manual's text, one module per chapter
  ui/               one module per tab, plus shared components/carousel
  data/example/     bundled CIFAR-10-based example dataset
scripts/            one-off data-generation scripts (not shipped in the package)
```

## Scale notes

Sized for up to ~20 experiments of ~500k rows each: CSV loads are chunked
with progress feedback, numeric columns are downcast to float32/int32,
dataset<->experiment joins and computed metrics/curves are cached, and
image detail views page through full-resolution crops a few at a time
instead of loading a whole match set at once.

The Calibration tab sorts each experiment's confidences once and then
answers every threshold query -- the slider, the target search, the whole
summary table -- from that index, so the explorer is live rather than a form
with an Apply button. The index costs 8 bytes per row per selected
experiment (float32 confidence + int32 running count) and is dropped on
Update, like every other cache in the app.

## Runs fully offline

Flet's web client normally reaches out to the internet in three places, all
eliminated here:

1. It fetches its own `flet-web` package via pip if the installed version
   doesn't match `flet` exactly -- `flet` and `flet-web` are pinned to the
   exact same version in `pyproject.toml`, so that check always passes
   locally.
2. Its bundled Flutter app streams the CanvasKit (Skia/WASM) renderer from
   `gstatic.com/flutter-canvaskit` on page load -- `lookin ui` patches the
   installed Flet web assets, once, to serve a CanvasKit copy bundled
   inside this package instead. This has to be patched in *two* places:
   `flutter_bootstrap.js`'s own loader config, and a second, independent
   default read from the engine-init config inside the compiled
   `main.dart.js` itself.
3. The app's base UI font (Roboto) and CanvasKit's fallback fonts for
   other scripts (CJK, Arabic, etc.) are both fetched from
   `fonts.gstatic.com` through that same engine-init config. Roboto is
   bundled locally and served from a local `fonts/` directory; the
   (unneeded) fallback-script fonts simply 404 locally instead of
   reaching Google.

See `src/lookin/_offline_assets.py` for the implementation. You can confirm
this yourself by opening your browser's Network tab while running
`lookin ui`: every request should stay on `localhost`.
