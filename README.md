# Lookin

A visual explorer for deep-learning classification results: dataset
statistics, a filterable image gallery, and cross-experiment metrics
(confusion matrices, ROC / PR curves) -- organised in workspaces built from
your [MLflow](https://mlflow.org) runs.

## Install & run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv tool install .
lookin web
```

This serves the interface at `http://127.0.0.1:8560` -- a plain HTML page
served by the Python process, nothing fetched from the network. Use
`--host` / `--port` to change where it binds:

```bash
lookin web --host 0.0.0.0 --port 8080
```

Point it at data on startup instead of typing/browsing for it in the UI,
with `--dataset` and (optionally, repeatable) `--experiment NAME=PATH`:

```bash
lookin web --dataset /path/to/dataset.csv \
  --experiment baseline=/path/to/results_baseline.csv \
  --experiment finetuned=/path/to/results_finetuned.csv
```

Without installing, you can also run it directly from the project directory:

```bash
uv run lookin web
```

Every browser tab is its own session: open several workspaces side by side
and nothing in one affects the others. Two servers can run at once on one
machine (on two ports), and share the same library, workspaces, snapshots
and review decisions. See `src/lookin/web/README.md` for how the interface
is built.

It has a light and a dark mode: the sun / moon button in the top bar
switches between them, and this browser remembers the choice (until then it
follows the system's setting). Charts and the Embeddings map stay on white
in both, like the PNGs they are exported as.

## What it does

1. **Sources tab** -- load a dataset CSV (one row per cropped image: an id,
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
2. **Filter tab** -- remove rows from the loaded dataset by conditions on
   its columns (pick values, a numeric range, contains / regex, empty or
   not) or by a query written like a pandas one (`width >= 32 and label in
   ['cat', 'dog']`, `` `object size`.between(0.2, 0.8) ``,
   `crop.str.contains('cam2/')`). The query is parsed and checked, never
   run as Python. Once applied, every other tab works on the rows kept, the
   bar above them says how many, and the filter is saved with the workspace
   and in each snapshot.
3. **Statistics tab** -- a histogram or value-count chart for every column.
4. **Gallery tab** -- a filterable, paginated image mosaic (label,
   categorical/binary metadata checkboxes, numeric range sliders), with
   adjustable rows x columns. Click an image for a full-resolution,
   paginated carousel view with all of its metadata.
5. **Metrics tab** -- per-class precision/recall/F1, ROC and
   Precision-Recall curves, and clickable confusion matrices (a cell opens
   the image carousel for that exact true/predicted combination) --
   overlaid across every experiment you select, for direct comparison.
   Plus a **breakdown by metadata**: pick any metadata column and every
   accuracy is recomputed per slice of it -- per sensor, per site, per crop
   size -- so "the model is at 94%" becomes "94%, except 61% on this
   sensor". See below.
6. **Calibration tab** -- whether the confidence can be believed, and what
   refusing to answer buys: reliability diagrams, ECE/MCE, the
   risk-coverage curve, a threshold explorer that runs both directions, and
   temperature scaling. See below.
7. **Compare tab** -- two experiments head to head: how many crops each
   one broke and fixed, a differential confusion matrix (candidate minus
   baseline, coloured by whether the change is good news rather than by
   the sign of the count), a per-class breakdown, and the regressed and
   fixed crops themselves, most-confident first, openable in the image
   carousel. The accuracy delta only shows the net; a model can gain a
   point overall while breaking hundreds of crops it used to get right.
8. **Embeddings tab** -- a 2D projection of an experiment's embeddings,
   plus a class-overlap matrix: mean cosine similarity within and between
   classes, measured on the crops themselves. The confusion matrix says the
   model mixes two classes up; this says whether they genuinely sit on top
   of each other, which is the difference between a model that needs more
   training and a taxonomy that needs rewriting. Simple inferences loaded
   on the Sources tab can be overlaid on the same map -- see below.
9. **Cleaning tab** -- find the crops worth a human look, decide on them,
   and export a versioned dataset you can train on. See below.
10. **Notes tab** -- free text about the workspace, written like a chat
    message (formatted as you type) and kept as `notes.md` in its folder.
    A new note starts with the workspace's name and a short performance
    recap, which "Regenerate the recap" rewrites from the current numbers;
    every exported report starts with the notes.

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
image paths and an embeddings `.npz` from the same model (Sources tab, section
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
in the top bar opens the same window at its front page, and the search box
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
in the same serif face as the charts -- there is no MathJax and nothing
fetched (see *Runs fully offline*). The content lives in
`src/lookin/core/manual/`, one module per chapter, as data rather than
markup; `src/lookin/web/views/manual.py` draws it.

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
Sources tab, and are undoable as a single batch.

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
Sources tab never truncates the export. Loading an exported CSV back into
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

**Exporting a report.** The Metrics tab writes a single self-contained file
-- an HTML page, a PowerPoint deck or a Markdown document -- downloaded by
the browser (or written to a folder on the server, if you pick one):
provenance (which CSV, which split, which review version, whether the
reviewed labels were applied), the Notes tab's text, the summary table,
per-class charts and tables, and the confusion matrices, with every image
embedded. It opens with no network and no sibling files, and a browser
prints the HTML one to PDF. Selecting exactly two experiments adds the
head-to-head from the Compare tab. No PDF writer on purpose: it would be a
new dependency, which is what the rest of this project avoids.

## Home: MLflow runs and workspaces

Lookin opens on its **Home** page (the "Home" link in the top bar brings it
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
  it. A run with several CSVs asks which ones. The workspaces sit in
  groups you name (a project, a stage...); each group's **+** creates one
  in it. The dataset is optional at creation: set it on the card, in the
  Sources tab once the workspace is open, or let the first run added bring it
  (its `datamodule/configs/database/csv_path` parameter, configurable). A
  run evaluated on another dataset is refused.
- **An open workspace follows its card.** A dataset or a run added on Home
  -- in this tab or another -- shows up in the tab where the workspace is
  open within a few seconds.
- **Saving is explicit.** What you load or change in the Sources tab is not
  written into the workspace until you click **Save** in the top bar, which
  appears as soon as the two differ (or **Save as workspace** when no
  workspace is open). Snapshots of the view are saved separately, from the
  Workspaces window.
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
on Home, or the chips in the top bar) is where one is exported, imported,
renamed and its snapshots managed. Workspaces used to be called
*bundles*; existing ones load as they are.

**Duplicate...** in a workspace's menu on Home copies it -- dataset,
experiments and snapshots -- under a new id, right below the original;
changing one never changes the other.

An experiment's color is its dot's to change: click it (Metrics tab, or the
Experiments table on Sources) for ten preset colors, any other color, or *Automatic* to go back to its palette one. Every chart
follows, and the color is saved with the workspace.

For a one-off look without a workspace, the Sources tab's **Add from MLflow...**
loads a single CSV from a run into the session on screen.

A bundled CIFAR-10-based example dataset loads by default so the tool is
usable immediately.

## Datasets: the shared folders

The **Datasets** page (the button on Home's top bar) lists every dataset CSV
of the folders you point it at -- typically a team's shared, read-only
datasets folder -- grouped by the folder each one sits in, and reads each
one in the background, one at a time. Nothing is ever written there:
content hashes and statistics are cached in `~/.cache/lookin/datasets/`,
keyed by content and columns, so a file is read again only when it or its
settings change.

- **Columns, from the settings.** Nothing is guessed from the values: the
  id, image, split and source columns and the *mappings* -- the label
  columns -- are set once for every dataset in the page's settings
  (`mapped_class_label` by default), and can be overridden for one dataset
  from its page. A dataset can carry several mappings (several labellings of
  the same rows, each covering all or part of them); a column a file does
  not have is simply absent.
- **What is a dataset.** Every CSV of the folders but results CSVs, review
  files and the paths matching the exclusion patterns of the settings (by
  default `*/mappings/*` and `mappings/*`, the label mapping tables).
- **Derived datasets.** A CSV inside a folder named after another CSV
  (`train.csv` and `train/train_clean.csv`) is listed under it, as derived
  from it, and can be compared with it in one click.
- **Where a dataset comes from.** Point the page at the MLflow experiments
  that log your datasets (one run per dataset: the CSV and a `hash.txt` of
  its content hash). A CSV is then traced back to its run by that hash --
  *hash-matched* -- or, failing that, by the run's CSV having the same name
  and size -- *name-matched*. A derived CSV shows its main dataset's run --
  *parent-matched*. The run's
  params, metrics and tags are shown with the dataset.
- **Statistics, in tabs.** *Instances* holds what does not depend on a
  labelling: the source and split distributions, the checks (duplicate ids,
  rows and images, rows without an image path) and a small chart per other
  column. Then one tab per mapping: class balance (max/min ratio,
  normalized entropy, rare classes, unlabelled rows), the classes by source
  and by split, and their checks (classes absent from a split, classes that
  come from one source). The images themselves are never opened.
- **Comparing two datasets.** Ctrl+click a second dataset: sources and
  splits side by side, the rows both share through the id columns, and per
  mapping the class distributions with their Jensen-Shannon divergence and
  the labels that changed on the shared rows.

The folders, patterns, columns and experiments are kept in
`~/.config/lookin/datasets.json`.

## AI agents: the MCP endpoint

`lookin web` also serves an [MCP](https://modelcontextprotocol.io) endpoint
at `/mcp`, on the same host and port as the interface. An agent pointed at
it can do what a person does in the interface: read every number, build and
organise workspaces, review crops, and have charts, mosaics and reports
made -- "which experiment has the best macro F1, and what does it still
confuse?", "make a workspace from these three MLflow runs and send me the
report", "show me the twenty most confident errors on sensor B". It works
with any MCP client that can reach the server, such as OpenCode with a
local LLM:

```json
{ "mcp": { "lookin": { "type": "remote", "url": "http://lookin-server:8560/mcp" } } }
```

The model needs to support tool calling (and images, for the pictures).

**Reading** -- answers in compact JSON:

| tool | what it answers |
| --- | --- |
| `list_workspaces` | every workspace: id, name, group, dataset, experiments, download state |
| `get_workspace` | one workspace's dataset, filter, experiments (and their MLflow runs), snapshots |
| `get_dataset_summary`, `get_column_stats` | rows, columns by kind, classes; one column's distribution |
| `get_metrics` | the summary metrics of each experiment of a workspace |
| `get_per_class` | per-class precision, recall, F1, AUC... worst classes first |
| `get_confusions` | the most frequent (true -> predicted) mistakes |
| `compare_experiments` | crops fixed and broken by a candidate, per-class changes |
| `get_slices` | every experiment scored per slice of a metadata column, with intervals |
| `get_calibration`, `get_confidence` | ECE, AURC, an operating point for a target accuracy, temperature; margin and entropy |
| `get_examples` | individual crops, filtered by label, prediction or a query on the metadata |
| `find_cleaning_candidates`, `find_similar` | the Cleaning tab's candidate search; nearest crops in embedding space |
| `get_review`, `get_snapshot` | review versions and their counts; one saved view |
| `search_mlflow_runs`, `list_mlflow_run_files` | runs on the configured MLflow servers; the CSVs a run logged |
| `get_notes` | the workspace's notes, as Markdown |
| `list_datasets` | every dataset CSV of the shared folders: derivatives, MLflow link, rows, classes, health |
| `get_dataset` | one dataset's file, content hash, MLflow run (and how sure the link is), derivatives, workspaces |
| `get_dataset_stats` | sources, splits, columns, checks; per mapping: class balance, class x source / split tables |
| `compare_datasets` | two datasets: shared rows; per mapping, distribution changes, classes on one side only, changed labels |
| `find_dataset_issues` | the checks that failed, for one dataset or all of them |
| `get_dataset_rows` | a few rows of a dataset, filtered by label (in a mapping), source or split |

**Pictures and files** -- `plot_chart` draws any chart of the tabs (per-class
bars, ROC/PR, confusion matrix, reliability, risk-coverage, a column, a
slice breakdown, a 2D projection of the embeddings) with the very code the
tab draws it with, and `make_mosaic` puts crops on one picture, framed green
or red by an experiment's verdict; both come back as images. `generate_report`
writes the Metrics tab's HTML, PowerPoint or Markdown report.

**Changing the library** -- `create_workspace` (the dataset's columns are
guessed as on the Sources tab), `update_workspace` (and its Home group),
`duplicate_workspace`, `delete_workspace`, `rename_group`; `add_experiment`
(a local results CSV), `add_mlflow_run`, `download_workspace`,
`remove_experiment`, `rename_experiment`, `set_experiment_color`;
`set_dataset_filter` (the Filter tab); `create_snapshot`, `update_snapshot`,
`delete_snapshot`; `export_workspace`, `import_workspace`;
`record_review_decisions`, `undo_review_batch`, `create_review_version`,
`export_reviewed_dataset`; `write_notes`, `regenerate_notes_recap` (the
Notes tab). Each does what the matching button does, through
the same code, so a tab with that workspace open picks the change up within
seconds. Deleting asks for the id a second time, and those tools are marked
destructive so a client can ask its user first.

**Links.** Every answer about a workspace carries an `open_url` that opens
the same thing in the interface -- on the right tab, and for a crop, with
that crop shown (`/?workspace=lkw-...&tab=metrics`,
`...&crop=<uuid>&experiment=<name>`; the same parameters work in any link).
Files an agent has written land in `~/.cache/lookin/exports/` unless it
names another folder, and are downloadable at `/exports/<name>`. When
people reach Lookin under another address than the one it binds (a port
forward, a gateway), `--public-url http://lookin-server:8560` makes the
links use it.

Metrics are computed on the server, where the data is, on the workspace's
original labels and through its dataset filter, exactly as the tabs show
them. An experiment whose MLflow files are not downloaded is reported as
not loaded, not silently left out. Every answer carries the same warnings
the interface would show (dataset changed since the workspace was saved,
experiment barely overlapping the dataset, filtered rows). The last two
workspaces loaded stay in memory, so follow-up questions are fast.

Whoever can reach the interface can reach the endpoint, and it can do no
more than the interface: requests from a page on another site are refused
for both. `--mcp-read-only` serves the reading tools only (and no picture
is written to disk); `--no-mcp` turns the endpoint off.

## Development

```bash
uv sync              # install into a local .venv, including dev deps
uv run lookin web    # run from source
uv run pytest        # run the test suite
```

Most tests cover `src/lookin/core/` -- the computation layer, which knows
nothing of HTML: the versioned review store (cumulative resolution,
tombstones, freezing, and the export/adopt round-trip), the metrics
(including the hand-written Brier decomposition and calibration error,
checked against their defining identities), the slice machinery (quantile
bins that stay comparable across experiments, the Wilson interval, the
small-slice guard), selective prediction (that the two threshold queries
agree with each other, including on tied confidences, and that temperature
scaling never moves an argmax), and the data model (column typing, the
dataset<->experiment join, and its cache invalidation).

The rest drive the browser interface without a browser: a `lookin web`
session is a Python object, so a test opens one, calls a view's actions and
reads the HTML it renders -- every tab renders, two tabs never share state,
every snapshot view restores what it saved (`tests/test_web_sessions.py`).
The MCP tools are called both as functions and over HTTP
(`tests/test_agent_tools.py`, `tests/test_mcp_write.py`).

Everything is built in `tmp_path`; the bundled example data is copied there
before a test writes next to it.

Regenerate the bundled example data (`src/lookin/data/example/`):

```bash
uv run python scripts/generate_example.py
```

Generate a larger synthetic dataset for scale testing (reuses the example
crop images under fresh uuids -- not meant to be committed):

```bash
uv run python scripts/generate_stress_test.py --rows 120000 --experiments 5
```

Generate a load test with one real image per row -- by default 500,000
synthetic 256x256 PNGs over 40 classes (a class is a coloured shape, e.g.
`teal_star`), with metadata columns and three simulated models with
embeddings. It takes about 4 GB, refuses to start if the disk is too
small, and resumes where it stopped if interrupted. `scripts/demo.py` then
shows it as two workspaces, one reading the files in place and one from the
demo MLflow server, not downloaded yet:

```bash
uv run python scripts/generate_load_test.py --estimate-only   # disk needed, nothing written
uv run python scripts/generate_load_test.py                   # into .load_test/ (git-ignored)
uv run python scripts/demo.py
```

### Project layout

```
src/lookin/
  cli.py            `lookin` command (argparse)
  mcp_server.py     the MCP endpoint: tool descriptions for AI agents
  core/             data model, metrics, charts, MLflow REST client -- no HTML
  core/manual/      the reference manual's text, one module per chapter
  web/              `lookin web`: FastAPI server, per-tab sessions, HTML views
  web/static/       the page's CSS, JavaScript and fonts -- all local
  data/example/     bundled CIFAR-10-based example dataset
scripts/            data generation and benchmarks (not shipped in the package)
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

Lookin is used on machines with no route to the internet, so nothing it
serves comes from anywhere else:

- The page is plain HTML, CSS and JavaScript written for it, in
  `src/lookin/web/static/` -- no framework, no CDN.
- Its fonts (IBM Plex Sans and Mono, under their open licence) ship in
  `static/fonts/`; charts and equations use the serif faces bundled inside
  matplotlib.
- Charts, equations and crops are drawn on the server (matplotlib, Pillow)
  and sent as images; the Embeddings map is drawn on a canvas by
  `static/js/projection.js`.
- Nothing is installed or downloaded at runtime. The only outbound calls
  are to the MLflow servers you configure yourself.

You can confirm this by opening your browser's Network tab while running
`lookin web`: every request stays on the Lookin server.
