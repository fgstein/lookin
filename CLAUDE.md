# Conventions for this project

## The app must run with no internet access

Lookin is used on machines that have no route to the internet. Anything
that only works because a network happened to be reachable is a bug, even
when it "works on my machine".

Concretely, when adding a feature:

- **No CDN, ever.** No `<script src="https://...">`, no remote stylesheet,
  no webfont fetched at runtime, no map tiles, no remote favicon. A
  JavaScript charting library (Plotly, Bokeh, D3, deck.gl, ...) is only
  acceptable if its full bundle is vendored under `src/lookin/web/static/`
  and served locally -- the page's fonts (IBM Plex, `static/fonts/`) and
  the Embeddings map's canvas code (`static/js/projection.js`) already are.
- **No install-on-demand.** Nothing may shell out to `pip`, `uv`, or a
  package downloader at runtime; dependencies get declared in
  `pyproject.toml`, not fetched lazily.
- **No model/dataset/asset download at runtime.** Everything a feature
  needs must ship in the wheel or come from a path the user provides.
- Outbound calls to a host the *user* configured (an MLflow tracking
  server on their own network, see `core/mlflow_client.py`) are fine --
  those are explicitly opt-in and are expected to fail gracefully. The
  rule is about implicit fetches the user never asked for.

The cheapest way to check: run `lookin web` with networking cut off,
exercise the feature, and confirm in the browser's Network tab that every
request stays on the Lookin server and nothing renders blank.

## Everything is in English — code *and* interface

This project is written entirely in English, and must stay that way.

**Code**: identifiers, comments, docstrings, log messages.

**User-visible text**: every string a user can read in the running app —
labels, button text, tooltips (`title=`), section titles, status and error
messages, `h.banner` copy, table headers, dropdown *labels*, what an MCP
tool answers or describes, and everything matplotlib draws (axis labels, titles, legend entries and
legend titles, colorbar labels).

The UI used to have French text mixed in; it was fully translated to English
on 2026-09-02. Do not reintroduce French, even when the person asking writes
to you in French — answer them in their language, but write English into the
files.

## The one trap when touching display strings

Some strings are **keys**, not text. An option of `h.select` is a
`(key, label)` pair, key first and label second: the key is a technical identifier that
code compares against, the label is what the user reads. Only the label is
translatable.

Keys that are compared somewhere and must never be reworded:

- projection methods: `pca`, `tsne`, `umap`, `umap_circle`
- matrix ordering: `margin`, `alpha`
- class focus criteria (`core/class_focus.py` `FOCUS_*`): `most_confused`,
  `worst_f1`, `largest_support`, `manual`, `least_confident`,
  `highest_entropy`, `most_changed`, `most_overlapping`, `least_separated`,
  `worst_margin`, `least_pure`, `lowest_coverage`, `most_predicted` -- also
  stored in snapshots
- prototype source: `model`, `kmeans`; the Embeddings map's "Prototypes from"
  dropdown (`web/views/embeddings.py` `PROTO_FROM_*`): `model`, `dataset`
- cleaning strategies: `closer_to_other_class`, `percentile`
- candidate class filter (`core/candidates.py` `MATCH_*`): `label`, `suggestion`;
  and the candidate source keys in `web/views/cleaning.py`, e.g. `all_errors`
- dataframe columns built in `AppState.merged`: `predicted_label`,
  `predicted_score`, `correct`
- `MACRO_AVERAGE` in `core/metrics.py` — a dropdown value *and* a comparison
- `NO_UUID_KEY` in `web/views/sources.py`: the uuid picker's "None" choice
- review statuses and reasons (`core/review.py`): `ok`, `relabel`, `exclude`,
  `uncertain`, `blurred`, ... -- written into the review files and accepted
  as is by the MCP tools; their labels live in `core/review_actions.py`
- the dataset manager (`core/datasets/`): MLflow link statuses
  (`mlflow_link.LINK_*`: `confirmed`, `probable`, `derived`, also CSS
  classes), issue severities and codes in `profile.py` (`error`, `warning`,
  `info`; `duplicate_ids`, ...), the MCP `sections` of
  `agent.SECTIONS`, and the page's sort keys (`web/views/datasets.py`
  `SORTS`)
- class labels themselves come from the user's CSV: never translate or
  normalize them
- manual page ids (`core/manual/pages_*.py`, e.g. `f1`, `roc-auc`): every
  (i) button in the UI names one, and `tests/test_manual.py` fails on a
  rename that leaves a button behind

The Embeddings map's "Color by" dropdown sidesteps this by keying its column
entries positionally (`"1"`, `"2"`, ...) and keeping the column name aside
(see `_rebuild_color_options` in `web/views/embeddings.py`), because a
user's column name could otherwise collide with a key the code reserves.

## The reference manual behind the (i) buttons

Every section showing a number carries an (i) that opens
`web/views/manual.py` at one page of `core/manual/`. Adding a page is: write
it in the right `pages_*.py` module as blocks (not Markdown -- the renderer
has to know which parts are mathematics), and point an (i) at its id with
`h.section(title, body, manual="that-id")` (or `h.info_btn("that-id")`).
Where the section draws different things depending on a dropdown, resolve
the id from a `*_MANUAL_PAGES` table (see `PERCLASS_MANUAL_PAGES` in
`web/views/metrics.py`) -- `tests/test_manual.py` reads those tables too.

This is also where explanations belong. A section on a tab carries at most
one line -- what it is, or how to drive it -- and anything that defines a
number, says how it is computed or how to interpret it goes in the manual
instead of under the chart. Before deleting such a paragraph from a view,
check its page actually says what the paragraph said, and add it there if it
does not.

Equations go through `web/mathtext.py`, which typesets them with matplotlib
-- so no `\underbrace`, and no `align`/`cases`/`array` environments; a
derivation is one `Equation` block per line. `tests/test_manual.py` typesets
every formula in the manual, so a mistake fails there rather than in a
dialog nobody opened.

## The MCP endpoint does what the interface does

`lookin web` serves an MCP endpoint (`src/lookin/mcp_server.py`) through
which an agent can do anything a person can: read (`core/agent_tools.py`),
change the library (`core/agent_actions.py`), and have charts, mosaics and
reports made (`core/agent_outputs.py`). Keep the two in step:

- A computation lives in `core/`, never in a view, so a tab and a tool call
  the same function. A chart is drawn in `core/charts.py` and the view only
  puts the figure on the page -- the agent's chart is the tab's chart.
- A feature added to a tab gets its tool, or an argument on an existing
  one. A tool's description is the only documentation the model gets:
  write it for a model that has never seen Lookin.
- A tool that writes goes through the same library call as the button
  (`library.update_workspace`, `record_decision`, ...), refuses rather than
  guesses, and is left out of `--mcp-read-only` (see `READ_TOOLS`).
- Every answer about a workspace carries its `warnings` and an `open_url`
  (`core/links.py`); the link parameters are a format and never renamed.

## Versions

The version lives in one place, `src/lookin/__init__.py` (`__version__`),
and is bumped in a commit of its own ("Bump version to 5.2"). Whenever a
change adds or changes something a user can see -- a page, a tab, a tool,
a behaviour -- at least *propose* bumping the minor version before
committing or pushing it; never let such a change go out silently at the
old version. Besides telling users something changed, the version is the
cache key of the page's stylesheets and scripts (`?v=<version>`), so
without a bump a browser can keep serving the old CSS.

## Checking before you commit a text change

Both of these should come back clean:

```bash
# 1. no accented characters in any string literal
uv run python -c "
import ast, pathlib, re
acc = re.compile(r'[éèêëàâäçîïôöûùüœÉÈÊÀÂÇÎÔÛÙ]')
hits = [(p, n.lineno, n.value) for p in pathlib.Path('src/lookin').rglob('*.py')
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8')))
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and acc.search(n.value)]
print(hits or 'clean')"
```

```bash
# 2. no accented string is used as a dict key or in a comparison
uv run python -c "
import ast, pathlib, re
acc = re.compile(r'[éèêëàâäçîïôöûùüœ]')
bad = []
for p in pathlib.Path('src/lookin').rglob('*.py'):
    for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
        got = []
        if isinstance(n, ast.Compare):
            got = [n.left, *n.comparators]
        elif isinstance(n, ast.Subscript):
            got = [n.slice]
        elif isinstance(n, ast.Dict):
            got = n.keys
        bad += [(p, c.lineno, c.value) for c in got
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and acc.search(c.value)]
print(bad or 'clean')"
```

Rewording an f-string is the other easy way to break something: dropping a
`{variable}` fails silently and only shows up in the rendered UI. Render the
affected panel and read the output, don't just re-read the source.
