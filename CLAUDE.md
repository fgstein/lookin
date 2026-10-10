# Conventions for this project

## The app must run with no internet access

Lookin is used on machines that have no route to the internet. Anything
that only works because a network happened to be reachable is a bug, even
when it "works on my machine".

Concretely, when adding a feature:

- **No CDN, ever.** No `<script src="https://...">`, no remote stylesheet,
  no webfont fetched at runtime, no map tiles, no remote favicon. A
  JavaScript charting library (Plotly, Bokeh, D3, deck.gl, ...) is only
  acceptable if its full bundle is vendored under `src/lookin/data/` and
  served locally -- see `_offline_assets.py`, which already does exactly
  this for CanvasKit and Roboto because Flet fetched both from Google.
- **No install-on-demand.** Nothing may shell out to `pip`, `uv`, or a
  package downloader at runtime. This is why `flet` and `flet-web` are
  pinned to the same exact version in `pyproject.toml` (see the comment
  there); dependencies get declared, not fetched lazily.
- **No model/dataset/asset download at runtime.** Everything a feature
  needs must ship in the wheel or come from a path the user provides.
- Outbound calls to a host the *user* configured (an MLflow tracking
  server on their own network, see `core/mlflow_client.py`) are fine --
  those are explicitly opt-in and are expected to fail gracefully. The
  rule is about implicit fetches the user never asked for.

The cheapest way to check: run the UI with networking cut off, exercise the
feature, and confirm no request goes out and nothing renders blank.

## Everything is in English — code *and* interface

This project is written entirely in English, and must stay that way.

**Code**: identifiers, comments, docstrings, log messages.

**User-visible text**: every string a user can read in the running app —
Flet widget labels, button text, tooltips, section titles, status and error
messages, `info_banner` copy, `DataTable` column headers, dropdown *labels*,
and everything matplotlib draws (axis labels, titles, legend entries and
legend titles, colorbar labels).

The UI used to have French text mixed in; it was fully translated to English
on 2026-09-02. Do not reintroduce French, even when the person asking writes
to you in French — answer them in their language, but write English into the
files.

## The one trap when touching display strings

Some strings are **keys**, not text. A `ft.dropdown.Option(key, label)` takes
the key first and the label second: the key is a technical identifier that
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
  dropdown (`ui/projection_view.py` `PROTO_FROM_*`): `model`, `dataset`
- cleaning strategies: `closer_to_other_class`, `percentile`
- candidate class filter (`core/candidates.py` `MATCH_*`): `label`, `suggestion`;
  and the candidate source keys in `ui/cleaning_view.py`, e.g. `all_errors`
- dataframe columns built in `AppState.merged`: `predicted_label`,
  `predicted_score`, `correct`
- `MACRO_AVERAGE` in `core/metrics.py` — a dropdown value *and* a comparison
- `NO_UUID_KEY` in `ui/upload_view.py`: the uuid picker's "None" choice
- class labels themselves come from the user's CSV: never translate or
  normalize them
- manual page ids (`core/manual/pages_*.py`, e.g. `f1`, `roc-auc`): every
  (i) button in the UI names one, and `tests/test_manual.py` fails on a
  rename that leaves a button behind

The case-study dropdown in `ui/embeddings_view.py` sidesteps this by keying
its options positionally (`"0"`, `"1"`, ...) and keeping the class label in
`_case_study_scope_map`, because a user's class label could otherwise collide
with any separator baked into a composite key.

## The reference manual behind the (i) buttons

Every section showing a number carries an (i) that opens
`ui/manual_view.py` at one page of `core/manual/`. Adding a page is: write
it in the right `pages_*.py` module as blocks (not Markdown -- the renderer
has to know which parts are mathematics), and point an (i) at its id with
`section_title(..., ctx=self.ctx, manual="that-id")`. Where the section
draws different things depending on a dropdown, pass a method instead of a
string and resolve the id at click time (see `MetricsView._perclass_manual_page`).

This is also where explanations belong. A section on a tab carries at most
one line -- what it is, or how to drive it -- and anything that defines a
number, says how it is computed or how to interpret it goes in the manual
instead of under the chart. Before deleting such a paragraph from a view,
check its page actually says what the paragraph said, and add it there if it
does not.

Equations go through `ui/mathtext.py`, which typesets them with matplotlib
-- so no `\underbrace`, and no `align`/`cases`/`array` environments; a
derivation is one `Equation` block per line. `tests/test_manual.py` typesets
every formula in the manual, so a mistake fails there rather than in a
dialog nobody opened.

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
