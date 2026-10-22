# `lookin web` -- how the browser interface is built

`lookin web` serves Lookin as a plain HTML/CSS page driven by the server.
Everything that computes lives in `lookin.core` and knows nothing of HTML;
this package draws it, and keeps one session per browser tab.

## The moving parts

* `server.py` -- FastAPI. `/api/boot`, `/api/act` (one user action),
  `/api/render` (regions changed in the background), `/api/events` (SSE),
  `/img` (crops), `/chart` (rendered charts), `/dl` (downloads), `/static`.
* `session.py` -- one `Session` per **browser tab**. It owns an `AppState`,
  the views, the dialogs, the job queues. Two tabs never share anything but
  the process-wide MLflow hub, so several workspaces can be explored side by
  side. What core reads off a session: `state`, `display_lut`,
  `display_roll_col`, `prototype_source`, `view_state`, `notify_change()`,
  `notify_rename()`, `notify_review()`, `notify_display_change()`,
  `upload_view`.
* `render.py` -- which regions to send back, the page layout, the tabs.
* `views/` -- one module per tab, plus dialogs and shared components.
* `static/app.js` -- the whole client runtime (no framework, no CDN).

## A view

```python
class FooView(RebuildView):          # or View, for a cheap one
    name = "foo"                      # the action target, and the tab key
    title = "Foo"                     # job name, progress text

    def render(self) -> str: ...      # HTML for the tab, from self's state
    def on_bar(self, value="", **_):  # an action, from data-act="foo:bar"
        ...
```

* Render is a pure function of the view's state: keep everything the user
  chose (filters, selections, dropdown values) in attributes and draw from
  them. Never compute something expensive in `render()` -- cache what
  `compute()` produced and draw from that.
* After any action of a view, the view and the top bar are re-rendered
  automatically. From a background thread, call `self.refresh()` (or
  `self.s.mark("<view>")`) and the page fetches the new HTML.
* `RebuildView` is a view whose content is expensive to rebuild:
  `on_data_change()` marks it stale (red "Data has changed" banner with
  "Update now"), `rebuild(work)` runs `work` on the session's job for this
  view (one at a time, the latest request coalesced, behind the process-wide
  matplotlib lock), `set_progress(text)` shows a spinner line, and
  `stale_html()` / `progress_html()` draw both.
* Hooks the session calls on every view that has them:
  `on_data_change()`, `rename_experiment(old, new)`, `on_review_change()`,
  `on_display_changed()`, `on_shown()` (the tab was selected).
* Snapshot views (`gallery`, `metrics`, `calibration`, `compare`,
  `embeddings`, `cleaning`) implement `view_state() -> dict` and
  `restore_view_state(state) -> list[str]` (see `core/view_state.py`). The
  keys are a file format: a snapshot saved months ago must still restore,
  so a key is never renamed.

## HTML helpers (`web/html.py`)

`h.e()` escapes. Attributes for actions:

* `h.act(target, action, **args)` -- click; `self.a(action, **args)` in a view
* `h.on_change(...)` / `self.c(...)` -- `<select>`, checkbox, range: the
  value arrives as `value` (a bool for a checkbox)
* `h.on_input(...)` / `self.i(...)` -- debounced typing, `value`
* `h.on_enter(target, action)` -- Enter in a text field, `value`
* `h.on_submit(...)` -- a `<form>`: every named field arrives as an argument

Widgets: `h.button(label, act_attr=, kind="primary|danger|ghost|link",
icon_name=, small=, title=, disabled=)`, `h.select(options, value,
change=, label=)` (options are `(key, label)` pairs -- **keys are never
reworded**, see CLAUDE.md), `h.checkbox`, `h.radio`, `h.text_input`,
`h.textarea`, `h.banner(text, "info|warn|error|ok|stale", lines=)`,
`h.section(title, body, manual="<page id>", right=, sub=)` (a card with a
bold title and the (i) button), `h.segmented`, `h.chip`, `h.progress`,
`h.spinner`, `h.kv_grid`, `h.info_btn(page_id)`, `h.icon(name)`.

Elements with an `id` or `data-key` are matched when the page is morphed,
so give repeated cards a stable `data-key`. `data-force="1"` on an input
makes the server's value win over what the user typed; `data-static="k"`
on an element skips re-morphing it while `k` is unchanged (charts).

CSS: the shared rules are in `static/app.css` (tokens `--acc`, `--line`,
`--muted`, ...; classes `card`, `card-h`, `card-b`, `tbl`, `row`,
`row-wrap`, `stack`, `grid2`, `muted`, `small`, `mono`, `num`). A view's own
rules go in `static/css/<view>.css`, which every page loads.

## Pictures

* Charts: build the matplotlib figure (styled by `core/chart_style.py`), then
  `fig_html(self.s, fig, "title")` (`views/media.py`) -- an `<img>` drawn at
  twice the screen resolution; a click zooms it, with Open / Save.
* Crops: `crop_url(self.s, path, record)` (honours the display LUT and the
  roll correction) or `crop_img(...)`.
* The carousel: `open_carousel(session, title, records, crop_col, uuid_col,
  highlight_uuid=, review_source="<where>", suggestions=, label_col=)` for a
  list of row dicts, or `CarouselDialog(session, title, dataset, total,
  page_provider, ...)` for a lazily paged set (`views/carousel.py`).
  `header_html=` puts extra controls above the mosaic (e.g. an "Open in
  Gallery" button). The decision bar and "Find similar" come with it.

## Shared components

* `views/widgets.py` -- `class_select` (a class dropdown, type-to-filter
  past FOCUS_MAX), `ClassPicker`, `ClassFocus` (the criteria of
  `core/class_focus.py`, or a manual pick). They take the owner's target and
  an action name; the owner's `on_<action>(op, **args)` calls
  `component.handle(op, args)`.
* `views/common.py` -- `uuid_overlap_banner`, `score_diagnostics_banner`,
  `status_line`, `saved_path_message`, `save_png`, `confused_pairs_table`,
  `overview_image` / `overview_zoom`, `card`, and how a class grid is
  sized (`matrix_cell_size`, `short_column_labels`).
* `views/files.py` -- `FileBrowserDialog` (a path on the machine running
  Lookin), `menu(...)`.
* `views/base.py` -- `Dialog`, `MessageDialog`, `ConfirmDialog`,
  `PromptDialog`.

Dialogs: subclass `Dialog`, implement `body()` / `footer()`, open with
`session.open_dialog(d)`; their actions are `self.a(...)` like a view's.

Other session calls: `session.toast(text, kind)`, `session.goto(tab)`,
`session.cmd(copy=text)`, `session.cmd(download=url, filename=...)` after
`session.put_download(bytes, name, mime)`, `session.viewport` (the browser
window, `(width, height)`).

## Rules that hold everywhere

The project rules in `CLAUDE.md` apply unchanged: nothing fetched from the
network (every asset is under `static/`), English only, dropdown keys never
reworded, class labels never normalized and never used as ids, explanations
in the manual (an (i) next to each section) rather than under the chart.
