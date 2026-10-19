"""The Sources tab: what a workspace is made of -- one dataset, the
experiments scored on it, and optional simple inferences.

The web twin of lookin.ui.upload_view, with the same behaviour:

* the three roles (id, image, label) are guessed from the file's values and
  offered as dropdowns of its columns, "None (no id)" included;
* an optional row filter (on by default, split = test -- ignored when the
  file has no such column) and the pool kind (test / train / other);
* each experiment is a results CSV plus its optional embeddings and
  prototypes sidecars, auto-detected next to it by name; its name is edited
  in place and a rename relabels every view without recomputing anything;
* per experiment, the id match against the dataset and the class mapping
  (results column -> dataset label, dataset labels only);
* dataset-wide relabeling, which cascades into every experiment's mapping;
* simple inferences: crops outside the dataset, drawn on the Embeddings map.

Loads run on a worker thread and report their progress live; everything
else is immediate.
"""
from __future__ import annotations

import itertools
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

from lookin.core import library
from lookin.core.dataset_config import CONFIG_NAME, config_for_csv
from lookin.core.class_focus import FOCUS_MAX
from lookin.core.data_manager import (
    SPLIT_OTHER, SPLIT_TEST, SPLIT_TRAIN, ColumnConfig, sidecar_paths_for_csv,
)
from lookin.core.schema import sniff_csv
from lookin.ui.components import color_for_experiment
from lookin.web import html as h
from lookin.web.views.base import View
from lookin.web.views.files import FileBrowserDialog
from lookin.web.views.widgets import experiment_color_menu

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATASET = PACKAGE_ROOT / "data" / "example" / "dataset.csv"
EXAMPLE_EXPERIMENTS = [
    ("baseline_cnn", PACKAGE_ROOT / "data" / "example" / "baseline_cnn_results.csv"),
    ("finetuned_resnet", PACKAGE_ROOT / "data" / "example" / "finetuned_resnet_results.csv"),
]
MAPPING_MAX_ALL = FOCUS_MAX
# The uuid picker's "no id column" choice -- a key compared in code, never
# a column name (see CLAUDE.md).
NO_UUID_KEY = "__no_uuid__"
SPLIT_OPTIONS = ((SPLIT_TEST, "test (held out)"), (SPLIT_TRAIN, "train (used to fit)"), (SPLIT_OTHER, "other"))
PROGRESS_PUSH_S = 0.25

_ids = itertools.count(1)


def _fraction(progress: Optional[float]) -> Optional[float]:
    """A bar's fill, or None for a moving bar: nothing counted yet (the
    rows are being counted) is shown as work going on, not as 0%."""
    return progress if progress else None


class ExpRow:
    def __init__(self, name: str = "", path: str = ""):
        self.id = next(_ids)
        self.name = name
        self.path = path
        self.embed_path = ""
        self.proto_path = ""
        self.guessed_embed: Optional[str] = None
        self.guessed_proto: Optional[str] = None
        self.loaded_name: Optional[str] = None
        self.status = ("Not loaded", "neutral")
        self.embed_status = ("not loaded", "neutral")
        self.proto_status = ("not loaded", "neutral")
        self.progress: Optional[float] = None
        self.loading = False
        self.held = False
        self.open = False
        self.overlap = None
        self.mapping_query = ""
        self.mapping_picked: list = []


class InfRow:
    def __init__(self, name: str = "", path: str = "", uuid_col: str = "uuid", crop_col: str = "crop"):
        self.id = next(_ids)
        self.name = name
        self.path = path
        self.uuid_col = uuid_col
        self.crop_col = crop_col
        self.embed_path = ""
        self.image_root = ""
        self.guessed_embed: Optional[str] = None
        self.loaded_name: Optional[str] = None
        self.status = ("Not loaded", "neutral")
        self.progress: Optional[float] = None
        self.loading = False


class SourcesView(View):
    name = "sources"
    title = "Sources"

    def __init__(self, session):
        super().__init__(session)
        self.dataset_path = str(EXAMPLE_DATASET) if not session.app.dataset else session.app.dataset
        self.uuid_value = "uuid"
        self.crop_value = "crop"
        self.label_value = "label"
        self.column_options: list = []
        # One line under the columns: which lookin.toml filled them in, or
        # why it was not used.
        self.config_note = ("", "neutral")
        self.image_root = ""
        self.row_filter_enabled = True
        self.row_filter_col = "split"
        self.row_filter_value = "test"
        self.split_kind = SPLIT_TEST
        self.ds_status = ("Not loaded", "neutral")
        self.ds_progress: Optional[float] = None
        self.ds_loading = False
        self.rows: list = []
        self.inf_rows: list = []
        self.held: list = []
        self.relabel_open = False
        self.relabel_query = ""
        self.relabel_picked: list = []
        # Each class of the dataset CSV -> the name it has now. A relabeling
        # renames the values in the loaded frame, so without this the field
        # of a class renamed "cat" -> "feline" would be titled "feline" from
        # then on, and the CSV's own name would be gone from the screen.
        self.label_names: dict = {}
        self._last_push = 0.0
        if not session.app.dataset:
            for name, path in EXAMPLE_EXPERIMENTS:
                self.rows.append(ExpRow(name, str(path)))
        self._sniff(apply_config=True)

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    def _row(self, rid) -> Optional[ExpRow]:
        return next((r for r in self.rows if r.id == int(rid)), None)

    def _inf(self, rid) -> Optional[InfRow]:
        return next((r for r in self.inf_rows if r.id == int(rid)), None)

    def _push(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_push > PROGRESS_PUSH_S:
            self._last_push = now
            self.s.mark(self.name)

    def _bg(self, fn, *args) -> None:
        def run():
            try:
                fn(*args)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            finally:
                self.s.mark(self.name, "chrome")
        threading.Thread(target=run, daemon=True).start()

    def _browse(self, on_pick, start: str = "", extensions=None, dirs_only=False, title="Select a file"):
        self.s.open_dialog(FileBrowserDialog(on_pick, start, extensions, dirs_only, title))

    # ------------------------------------------------------------------
    # guessing the roles
    # ------------------------------------------------------------------

    def _sniff(self, apply_config: bool = False) -> None:
        """Read the head of the file and fill the three columns and the
        root in: from the lookin.toml next to it where it says (see
        core/dataset_config.py), else by guessing from the values.

        `apply_config`: the user just picked this file, so lookin.toml wins
        over every field. Otherwise (a workspace being restored) it only
        fills a field still at its default -- a value the workspace chose
        itself stays -- and a guess only ever replaces a column that is not
        in this file (see UploadView._sniff_dataset for the reasoning)."""
        path = (self.dataset_path or "").strip()
        self.config_note = ("", "neutral")
        if not path:
            self.column_options = []
            return
        guess = sniff_csv(path)
        self.column_options = list(guess.columns)
        config = config_for_csv(path)
        from_config = []
        if config is not None and config.error:
            self.config_note = (f"{CONFIG_NAME} next to this file {config.error} -- not used.", "warning")
        elif config is not None:
            defaults = library.Mount()
            unknown = []
            for attr, key, default in (("uuid_value", "id", defaults.uuid_col), ("crop_value", "image", defaults.crop_col),
                                       ("label_value", "label", defaults.label_col)):
                value = getattr(config, key)
                if value is None:
                    continue
                if value and value not in self.column_options:
                    unknown.append(value)
                    continue
                if apply_config or getattr(self, attr) in ("", default) or getattr(self, attr) not in self.column_options:
                    setattr(self, attr, value)
                    from_config.append(key)
            if config.root is not None and (apply_config or not self.image_root):
                self.image_root = config.root
                from_config.append("root")
            if unknown:
                self.config_note = (f"{CONFIG_NAME} names {', '.join(repr(u) for u in unknown)}, not a column of this "
                                    f"file -- guessed instead.", "warning")
            elif from_config:
                self.config_note = (f"{', '.join(k.capitalize() for k in from_config)} from the {CONFIG_NAME} "
                                    f"next to this file", "neutral")
        for attr, key, guessed in (("uuid_value", "id", guess.uuid_col), ("crop_value", "image", guess.crop_col),
                                   ("label_value", "label", guess.label_col)):
            if key in from_config:
                continue
            if attr == "uuid_value" and not self.uuid_value:
                continue
            if guessed and getattr(self, attr) not in self.column_options:
                setattr(self, attr, guessed)
        if ("id" not in from_config and self.uuid_value and self.uuid_value not in self.column_options
                and self.column_options and not guess.uuid_col):
            self.uuid_value = ""

    def set_dataset_path(self, path: str) -> None:
        self.dataset_path = path
        self._sniff(apply_config=True)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        return (f'<div class="stack">{self._dataset_card()}{self._experiments_card()}'
                f'{self._inferences_card()}</div>')

    # ---- dataset ----

    def _dataset_card(self) -> str:
        state = self.state
        text, kind = self.ds_status
        dot_color = {"success": "var(--ok)", "error": "var(--err)", "warning": "var(--warn-dot)",
                     "busy": "var(--acc)"}.get(kind, "var(--ghost)")
        status = (f'<span class="row-nw" style="font-size:12px;color:{"var(--ok)" if kind == "success" else "var(--err)" if kind == "error" else "var(--muted)"}">'
                  f'{h.dot(dot_color)}{h.e(text)}</span>')
        right = (h.button("Change dataset…", act_attr=self.a("pick_dataset"), small=True, icon_name="table")
                 + h.button("Load dataset", act_attr=self.a("load_dataset"), kind="primary", small=True,
                            icon_name="upload", disabled=self.ds_loading)
                 + h.button("Clear all data", act_attr=self.a("clear_all"), small=True, icon_name="trash"))
        names = list(dict.fromkeys(list(self.column_options) + [self.uuid_value, self.crop_value, self.label_value]))
        names = sorted((n for n in names if n), key=lambda n: (str(n).casefold(), str(n)))
        role_opts = [(n, n) for n in names]
        roles = (
            h.select([(NO_UUID_KEY, "None (no id)")] + role_opts, self.uuid_value or NO_UUID_KEY,
                     change=self.c("set_role", role="uuid_value"), label="id", cls="mono",
                     title="The crop id. Guessed from the values: a uuid has a very specific shape. "
                           "None if the file has no id: the crops and their metadata can still be browsed, "
                           "but no experiment can be matched to them.")
            + h.select(role_opts, self.crop_value, change=self.c("set_role", role="crop_value"), label="image",
                       cls="mono", title="The path to the image. Guessed from the values: a path has a "
                                         "separator or an image extension.")
            + h.select(role_opts, self.label_value, change=self.c("set_role", role="label_value"), label="label",
                       cls="mono", title="The class. Guessed from the values: a short string with few "
                                         "distinct values.")
        )
        file_field = (f'<div class="row-nw">{h.text_input(self.dataset_path, change=self.c("set_path"), mono=True, cls="grow", label="")}'
                      f'{h.button(icon_name="folder", act_attr=self.a("browse_dataset"), title="Browse the server", small=True)}</div>')
        file_meta = ""
        try:
            p = Path(self.dataset_path)
            if p.is_file():
                st = p.stat()
                file_meta = f'<span class="muted small">{h.e(h.human_size(st.st_size))}</span>'
        except OSError:
            pass
        root_field = (f'<div class="row-nw">{h.text_input(self.image_root, change=self.c("set_image_root"), mono=True, cls="grow", placeholder="the dataset CSV\'s own folder")}'
                      f'{h.button(icon_name="folder", act_attr=self.a("browse_root"), title="Browse the server", small=True)}</div>')
        filter_cols = self.column_options or ([self.row_filter_col] if self.row_filter_col else [])
        if self.row_filter_col and self.row_filter_col not in filter_cols:
            filter_cols = [self.row_filter_col] + list(filter_cols)
        if filter_cols:
            col_widget = h.select([(c, c) for c in filter_cols], self.row_filter_col,
                                  change=self.c("set_filter_col"), cls="mono", label="")
        else:
            col_widget = h.text_input(self.row_filter_col, change=self.c("set_filter_col"), mono=True, width="140px")
        filter_note = ""
        if self.row_filter_enabled and self.column_options and self.row_filter_col not in self.column_options:
            filter_note = f'<span class="muted small">no "{h.e(self.row_filter_col)}" column in this file: every row is kept</span>'
        elif state.dataset is not None and state.dataset.row_filter:
            filter_note = f'<span class="muted small">{len(state.dataset.full_df):,} rows kept</span>'
        rows_field = (f'<div class="row">{h.checkbox("Keep only rows where", self.row_filter_enabled, change=self.c("set_filter_enabled"))}'
                      f'{col_widget}<span>=</span>{h.text_input(self.row_filter_value, change=self.c("set_filter_value"), mono=True, width="140px")}'
                      f'{filter_note}</div>')
        pool = (h.segmented(SPLIT_OPTIONS, self.split_kind, act_for=lambda k: self.a("set_split", kind=k), label="Pool")
                + '<span class="muted small">predictions on a train pool are in-sample: changes what Cleaning allows</span>')
        note_text, note_kind = self.config_note
        note = (f'<div class="small {"c-warn" if note_kind == "warning" else "muted"}" style="margin-top:3px">{h.e(note_text)}</div>'
                if note_text else "")
        grid = h.kv_grid([
            ("File", file_field + file_meta),
            ("Columns", f'<div class="row">{roles}</div>' + note),
            ("Root", root_field),
            ("Rows", rows_field),
            ("Pool", f'<div class="row">{pool}</div>'),
        ])
        # Across the card, above the form: a load can take a minute on a
        # large file, and a bar tucked under the form went unnoticed.
        prog = ""
        if self.ds_loading:
            prog = f'<div class="progress-wide" style="margin-bottom:10px">{h.progress(text, _fraction(self.ds_progress))}</div>'
        extra = ""
        if state.dataset is not None:
            kinds = state.dataset.column_kinds()
            parts = [f"Numeric: {', '.join(kinds.numeric) or '—'}", f"Binary: {', '.join(kinds.binary) or '—'}",
                     f"Categorical: {', '.join(kinds.categorical) or '—'}"]
            if kinds.high_cardinality:
                parts.append("Skipped (too many unique values -- not shown in Statistics/Gallery filters): "
                             + ", ".join(kinds.high_cardinality))
            extra = (f'<div class="muted small" style="margin-top:10px">{h.e("  ·  ".join(parts))}</div>'
                     + self._relabel_block())
        return h.section("Dataset", prog + grid + extra, manual="dataset-join", right=right,
                         sub="", id_="src-dataset").replace(
            '<div class="card-r">', f'{status}<div class="card-r">', 1)

    def _label_names(self) -> dict:
        """label_names, started over from the loaded labels whenever it no
        longer describes them (a new dataset, or labels changed elsewhere)."""
        labels = [str(l) for l in self.state.dataset.labels()]
        if set(self.label_names.values()) != set(labels):
            self.label_names = {l: l for l in labels}
        return self.label_names

    def _relabel_block(self) -> str:
        names = self._label_names()
        originals = sorted(names, key=lambda l: (l.casefold(), l))
        chev = "▼" if self.relabel_open else "▶"
        head = (f'<button type="button" class="grp-btn" style="margin-top:8px;border-top:1px solid var(--line-2);padding-top:8px;height:auto"'
                f'{self.a("toggle_relabel")} aria-expanded="{"true" if self.relabel_open else "false"}">'
                f'<span class="tri">{chev}</span><b>Relabel classes</b>'
                f'<span class="muted">rename the dataset\'s label values; cascades into every experiment\'s mapping</span></button>')
        if not self.relabel_open:
            return head
        if len(originals) > MAPPING_MAX_ALL:
            q = self.relabel_query.strip().lower()
            matches = [l for l in originals if q and (q in l.lower() or q in names[l].lower())
                       and l not in self.relabel_picked][:30]
            finder = (f'<div class="row" style="margin:6px 0">{h.text_input(self.relabel_query, input_=self.i("relabel_query"), placeholder="Find a class to rename", width="260px")}'
                      + "".join(h.chip(m, act_attr=self.a("relabel_pick", label=m)) for m in matches) + "</div>")
            shown = [l for l in self.relabel_picked if l in names]
        else:
            finder = ""
            shown = originals
        # The title of each field is the class as the CSV names it, and stays
        # so; the field holds what it is called now.
        fields = "".join(
            f'<label class="fld" style="flex-direction:column;align-items:stretch;gap:2px">'
            f'<span class="fld-l mono" style="font-size:11px" title="In the dataset CSV">{h.e(l)}</span>'
            f'<input class="inp mono{" relabelled" if names[l] != l else ""}" name="r{i}" value="{h.e(names[l])}" '
            f'style="width:150px" data-orig="{h.e(l)}"></label>'
            for i, l in enumerate(shown))
        self._relabel_shown = list(shown)
        form = (f'<form{self.submit_attr("apply_relabel")} class="col" style="margin-top:6px">{finder}'
                f'<div class="row">{fields}</div>'
                f'<div>{h.button("Apply relabeling", kind="primary", small=True, icon_name="check", type_="submit")}</div></form>')
        return head + form

    def submit_attr(self, action: str) -> str:
        return h.on_submit(self.name, action)

    # ---- experiments ----

    def _origins(self) -> dict:
        ws = self.s.current_workspace
        return {(e.name, e.path): e for e in ws.mount.experiments} if ws is not None else {}

    def _experiments_card(self) -> str:
        state = self.state
        right = (h.button("+ From MLflow…", act_attr=self.a("from_mlflow"), small=True, cls="btn-acc")
                 + h.button("+ Results CSV…", act_attr=self.a("add_row"), small=True))
        if not self.rows:
            body = h.empty("No experiment yet. Add a results CSV, or pick runs from MLflow.")
            return h.section("Experiments", body, manual="score-diagnostics", right=right,
                             sub="a name edited here changes everywhere, nothing is recomputed")
        names = list(state.experiments)
        origins = self._origins()
        head = ('<thead><tr><th style="width:18px"></th><th>Name</th><th>From</th><th>Results</th>'
                '<th>Ids matched</th><th>Classes</th><th>Embeddings</th><th>Prototypes</th><th></th></tr></thead>')
        bodies = []
        for row in self.rows:
            bodies.append(self._exp_tbody(row, names, origins))
        table = f'<div class="tbl-wrap"><table class="tbl">{head}{"".join(bodies)}</table></div>'
        return h.section("Experiments", f'<div class="flush">{table}</div>', manual="score-diagnostics", right=right,
                         sub=f"{len(self.rows)} · a name edited here changes everywhere, nothing is recomputed")

    def _exp_tbody(self, row: ExpRow, names: list, origins: dict) -> str:
        state = self.state
        exp = state.experiments.get(row.loaded_name) if row.loaded_name else None
        color = color_for_experiment(row.loaded_name, self.state.color_slots) if exp is not None else "var(--ghost)"
        dot = (experiment_color_menu(self.target, row.loaded_name, color, row.loaded_name in state.experiment_colors, size=9)
               if exp is not None else h.dot(color, size=9))
        # A row with nothing loaded opens on its files -- except while it
        # waits or loads, which its bar in the row already says.
        is_open = row.open or (exp is None and not row.held and not row.loading)
        origin = origins.get((row.loaded_name, row.path)) if exp is not None else None
        if origin is None:
            origin = next((o for (n, p), o in origins.items() if p == row.path), None)
        if origin is not None and origin.from_mlflow():
            run = origin.mlflow_run_name or origin.mlflow_run_id[:8]
            src = (f'<span class="row-nw">MLflow ·<span class="xname" style="max-width:16em" title="{h.e(run)}">'
                   f'{h.e(run)}</span></span>')
        else:
            src = "Local file"
        name_input = (f'<input class="inp mono" value="{h.e(row.name)}" title="{h.e(row.name)}" '
                      f'style="width:160px;height:24px;background:var(--surface-2);text-overflow:ellipsis"'
                      f'{self.c("rename", row=row.id)}{h.on_enter(self.name, "rename", row=row.id)} aria-label="Experiment name">')
        if row.loading:
            ids_html = f'<div class="progress-cell">{h.progress(row.status[0], _fraction(row.progress))}</div>'
        elif exp is not None and row.overlap is not None:
            ov = row.overlap
            n_ds = len(state.dataset.df) if state.dataset is not None else 0
            problem = ov.dataset_only or ov.experiment_only or ov.dataset_dupes or ov.experiment_dupes
            ids_html = (f'<span class="row-nw small" style="color:{"var(--warn)" if problem else "var(--ink-2)"}">'
                        f'{h.dot("var(--warn-dot)" if problem else "var(--ok)", size=6)}{ov.matched:,} / {n_ds:,}</span>')
        else:
            text, kind = row.status
            ids_html = f'<span class="small" style="color:{"var(--err)" if kind == "error" else "var(--warn)" if kind == "warning" else "var(--faint)"}">{h.e(text)}</span>'
        classes_html = "—"
        if exp is not None:
            # The dataset's labels once, not once per class: at two thousand
            # classes that was two thousand passes over every row, on every
            # render of this tab (four seconds each).
            known = set(state.dataset.labels()) if state.dataset is not None else set()
            mapped = sum(1 for i in range(exp.n_classes()) if i < len(exp.class_labels)
                         and exp.class_labels[i] in known)
            renamed = sum(1 for i, c in enumerate(exp.score_cols) if i < len(exp.class_labels) and exp.class_labels[i] != c)
            classes_html = f"{mapped} / {exp.n_classes()}" + (f" · {renamed} renamed" if renamed else " by name")
        emb_html = "—"
        if exp is not None and exp.embeddings is not None:
            emb_html = f"{exp.embeddings.shape[1]}-d · {exp.embeddings.shape[0]:,}"
        proto_html = "—"
        if exp is not None and exp.model_prototypes is not None:
            ps = exp.model_prototypes
            proto_html = f"{len(ps.class_labels)} × {ps.k}"
        load_label = "Reload" if exp is not None else "Load"
        actions = (h.button(load_label, act_attr=self.a("load_row", row=row.id), small=True, disabled=row.loading)
                   + h.button(icon_name="x", act_attr=self.a("remove_row", row=row.id), small=True, kind="ghost",
                              title="Remove this experiment"))
        tri = "▼" if is_open else "▶"
        tr = (f'<tr{" class=sel" if row.open and exp is not None else ""}>'
              f'<td><button type="button" class="chev"{self.a("toggle_row", row=row.id)} aria-expanded="{"true" if is_open else "false"}" '
              f'aria-label="Show details" style="font-size:9px">{tri}</button></td>'
              f'<td><div class="row-nw">{dot}{name_input}</div></td>'
              f'<td class="small nowrap" style="color:var(--muted)">{src}</td>'
              f'<td class="mono small nowrap" title="{h.e(row.path)}">{h.e(Path(row.path).name if row.path else "—")}</td>'
              f'<td class="nowrap">{ids_html}</td>'
              f'<td class="small nowrap">{h.e(classes_html)}</td>'
              f'<td class="small nowrap">{h.e(emb_html)}</td>'
              f'<td class="small nowrap">{h.e(proto_html)}</td>'
              f'<td class="nowrap" style="text-align:right">{actions}</td></tr>')
        detail = ""
        if is_open:
            detail = f'<tr class="ws-detail"><td colspan="9">{self._exp_detail(row, exp, origin)}</td></tr>'
        return f'<tbody data-key="exp-{row.id}">{tr}{detail}</tbody>'

    def _exp_detail(self, row: ExpRow, exp, origin) -> str:
        def path_line(label, value, field, browse_action, load_action=None, status=None):
            st = ""
            if status is not None:
                text, kind = status
                color = {"error": "var(--err)", "warning": "var(--warn)", "success": "var(--ok)"}.get(kind, "var(--faint)")
                st = f'<span class="small" style="color:{color}">{h.e(text)}</span>'
            load = h.button("Load", act_attr=self.a(load_action, row=row.id), small=True) if load_action else ""
            return (f'<div class="row-nw" style="margin:2px 0"><span class="small" style="width:84px;font-weight:600">{h.e(label)}</span>'
                    f'{h.text_input(value, change=self.c("set_row_field", row=row.id, field=field), mono=True, cls="grow")}'
                    f'{h.button(icon_name="folder", act_attr=self.a(browse_action, row=row.id), title="Browse the server", small=True)}'
                    f'{load}{st}</div>')

        text, kind = row.status
        color = {"error": "var(--err)", "warning": "var(--warn)", "success": "var(--ok)", "busy": "var(--acc-ink)"}.get(kind, "var(--faint)")
        status = f'<div class="small" style="color:{color};margin:4px 0">{h.e(text)}</div>'
        prog = ""
        files = (f'<div class="subhead">Files</div>'
                 + path_line("Results CSV", row.path, "path", "browse_row")
                 + status + prog
                 + path_line("Embeddings", row.embed_path, "embed_path", "browse_embed", "load_embed", row.embed_status)
                 + path_line("Prototypes", row.proto_path, "proto_path", "browse_proto", "load_proto", row.proto_status)
                 + '<div class="muted small" style="margin-top:4px">Sidecars are found next to the results CSV '
                   '(&lt;name&gt;_embeddings.npz, &lt;name&gt;_prototypes.npz) and loaded on their own.</div>')
        if origin is not None and origin.from_mlflow():
            files += (f'<div class="muted small" style="margin-top:4px">MLflow: {h.e(origin.mlflow_uri)} › '
                      f'{h.e(origin.mlflow_run_name or origin.mlflow_run_id)} › {h.e(origin.artifact_path)}</div>')
        left = f'<div style="flex:1 1 420px;min-width:340px">{files}{self._id_match(row, exp)}</div>'
        right = f'<div style="flex:1 1 420px;min-width:340px">{self._mapping(row, exp)}</div>' if exp is not None else ""
        return f'<div class="ws-detail-inner">{left}{right}</div>'

    def _id_match(self, row: ExpRow, exp) -> str:
        if exp is None:
            return ""
        ov = row.overlap
        if ov is None:
            return ""
        lines = [("Dataset rows with no prediction", ov.dataset_only),
                 ("Predictions with no dataset row", ov.experiment_only),
                 ("Duplicate ids in the dataset", ov.dataset_dupes),
                 ("Duplicate ids in the results", ov.experiment_dupes)]
        rows = "".join(
            f'<div class="row-nw" style="height:20px"><span>{h.e(label)}</span>'
            f'<span class="sp mono" style="font-weight:600;color:{"var(--ok)" if not v else "var(--warn)"}">{v:,}</span></div>'
            for label, v in lines)
        return (f'<div class="subhead" style="margin-top:8px">Id match <span class="muted">dataset ↔ this experiment, '
                f'{ov.matched:,} matched</span></div>{rows}')

    def _mapping(self, row: ExpRow, exp) -> str:
        state = self.state
        if state.dataset is None:
            return ""
        dataset_labels = state.dataset.labels()
        known = set(dataset_labels)
        indices = list(range(exp.n_classes()))
        many = len(indices) > MAPPING_MAX_ALL
        n_matched = 0
        if many:
            unmatched = [i for i in indices if i >= len(exp.class_labels) or exp.class_labels[i] not in known]
            n_matched = len(indices) - len(unmatched)
            indices = unmatched
        shown = indices[:MAPPING_MAX_ALL]

        def col_of(i):
            return exp.score_cols[i] if i < len(exp.score_cols) else f"class_{i + 1}"

        if many:
            index_of = {col_of(i): i for i in range(exp.n_classes())}
            row.mapping_picked = [c for c in row.mapping_picked if c in index_of]
            order = shown + [index_of[c] for c in row.mapping_picked if index_of[c] not in shown]
        else:
            order = shown
        pred_counts = self._pred_counts(exp)
        cells = []
        for i in order:
            current = exp.class_labels[i] if i < len(exp.class_labels) else (dataset_labels[0] if dataset_labels else "")
            renamed = current != col_of(i)
            opts = [(l, l) for l in dataset_labels]
            if current not in known:
                opts = [(current, f"{current} (not in dataset)")] + opts
            sel = h.select(opts, current, change=self.c("set_mapping", row=row.id, index=i), cls="mono",
                           title=f"What results column '{col_of(i)}' counts as")
            pc = pred_counts.get(i)
            cells.append(f'<div class="row-nw" style="height:28px;border-bottom:1px solid var(--line-3)">'
                         f'<span class="mono" style="width:110px" title="{h.e(col_of(i))}"><span class="ell" style="display:block">{h.e(col_of(i))}</span></span>'
                         f'<span class="muted small">→</span>'
                         f'<span{" style=\"outline:1px solid var(--acc);border-radius:6px\"" if renamed else ""}>{sel}</span>'
                         f'<span class="sp mono small muted" title="crops this experiment predicts as this class">{"" if pc is None else f"{pc:,} pred."}</span></div>')
        notes = []
        if n_matched:
            notes.append(f"{n_matched:,} of {exp.n_classes():,} results columns match a dataset class by name and are mapped to it.")
            if not indices:
                notes.append("Every column is mapped -- nothing to decide.")
        if indices and n_matched:
            notes.append(f"{len(indices):,} column(s) name no class of this dataset -- a spelling difference, "
                         f"or simply a class with no crop here, which needs no mapping.")
        if len(indices) > len(shown):
            notes.append(f"The first {len(shown)} are listed below.")
        finder = ""
        if many:
            q = row.mapping_query.strip().lower()
            matches = [c for c in exp.score_cols if q and q in c.lower() and c not in row.mapping_picked][:30]
            finder = (f'<div class="row" style="margin:4px 0">{h.text_input(row.mapping_query, input_=self.i("mapping_query", row=row.id), placeholder="Find a results column to map", width="260px")}'
                      + "".join(h.chip(m, act_attr=self.a("mapping_pick", row=row.id, column=m)) for m in matches) + "</div>")
        note_html = "".join(f'<div class="muted small">{h.e(n)}</div>' for n in notes)
        return (f'<div class="subhead">Classes <span class="muted">each score column, and the dataset label it '
                f'counts as (dataset labels only)</span></div>{note_html}{finder}'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));column-gap:24px">{"".join(cells)}</div>'
                f'<div class="row" style="margin-top:6px">{h.button("Match by name", act_attr=self.a("match_by_name", row=row.id), small=True)}</div>')

    def _pred_counts(self, exp) -> dict:
        """How many crops this experiment predicts as each score column."""
        try:
            import numpy as np
            scores = exp.df[exp.score_cols].to_numpy()
            if not len(scores):
                return {}
            arg = np.nanargmax(np.nan_to_num(scores, nan=-np.inf), axis=1)
            counts = np.bincount(arg, minlength=len(exp.score_cols))
            return {i: int(c) for i, c in enumerate(counts)}
        except Exception:  # noqa: BLE001
            return {}

    # ---- simple inferences ----

    def _inferences_card(self) -> str:
        right = h.button("+ Add inference…", act_attr=self.a("add_inference"), small=True)
        if not self.inf_rows:
            body = ""
            return h.section("Simple inferences", body, manual="dataset-join", right=right,
                             sub="None. Unlabelled crops outside the dataset, drawn on the Embeddings map.").replace(
                '<div class="card-b"></div>', "")
        rows = []
        for r in self.inf_rows:
            text, kind = r.status
            color = {"error": "var(--err)", "warning": "var(--warn)", "success": "var(--ok)", "busy": "var(--acc-ink)"}.get(kind, "var(--faint)")
            prog = h.progress(text, _fraction(r.progress)) if r.loading else ""
            rows.append(
                f'<div data-key="inf-{r.id}" style="padding:8px 0;border-top:1px solid var(--line-2)">'
                f'<div class="row">'
                f'{h.text_input(r.name, change=self.c("set_inf_field", row=r.id, field="name"), label="Name", width="160px", mono=True)}'
                f'<div class="row-nw grow"><span class="small" style="font-weight:600">Crops CSV</span>{h.text_input(r.path, change=self.c("set_inf_field", row=r.id, field="path"), mono=True, cls="grow")}'
                f'{h.button(icon_name="folder", act_attr=self.a("browse_inf", row=r.id, field="path"), small=True, title="Browse the server")}</div>'
                f'{h.button("Load", act_attr=self.a("load_inf", row=r.id), small=True, kind="primary", disabled=r.loading)}'
                f'{h.button(icon_name="x", act_attr=self.a("remove_inf", row=r.id), small=True, kind="ghost", title="Remove")}</div>'
                f'<div class="row" style="margin-top:6px">'
                f'{h.text_input(r.uuid_col, change=self.c("set_inf_field", row=r.id, field="uuid_col"), label="id column", width="120px", mono=True)}'
                f'{h.text_input(r.crop_col, change=self.c("set_inf_field", row=r.id, field="crop_col"), label="crop path column", width="140px", mono=True)}'
                f'<div class="row-nw grow"><span class="small" style="font-weight:600">Embeddings</span>{h.text_input(r.embed_path, change=self.c("set_inf_field", row=r.id, field="embed_path"), mono=True, cls="grow")}'
                f'{h.button(icon_name="folder", act_attr=self.a("browse_inf", row=r.id, field="embed_path"), small=True, title="Browse the server")}</div>'
                f'<div class="row-nw grow"><span class="small" style="font-weight:600">Images from</span>{h.text_input(r.image_root, change=self.c("set_inf_field", row=r.id, field="image_root"), mono=True, cls="grow", placeholder="the crops CSV\'s own folder")}'
                f'{h.button(icon_name="folder", act_attr=self.a("browse_inf", row=r.id, field="image_root"), small=True, title="Browse the server")}</div></div>'
                f'<div class="small" style="color:{color};margin-top:4px">{h.e(text)}</div>{prog}</div>')
        body = "".join(rows)
        return h.section("Simple inferences", body, manual="dataset-join", right=right,
                         sub="unlabelled crops outside the dataset, run through the same model, drawn on the Embeddings map as squares")

    # ------------------------------------------------------------------
    # dataset actions
    # ------------------------------------------------------------------

    def on_set_path(self, value: str = "", **_):
        self.dataset_path = (value or "").strip()
        self._sniff(apply_config=True)

    def on_browse_dataset(self, **_):
        self._browse(self._picked_dataset, self.dataset_path, {".csv"}, title="Select dataset CSV")

    def _picked_dataset(self, path: str) -> None:
        self.dataset_path = path
        self._sniff(apply_config=True)
        self.refresh()

    def on_pick_dataset(self, **_):
        from lookin.web.views.mlflow_dialogs import DatasetsDialog

        def picked(path: str) -> None:
            self._picked_dataset(path)
            self.s.mode = "analysis"
            self.s.tab = "sources"
            self.s.mark("page")

        self.s.open_dialog(DatasetsDialog(self.s, on_pick=picked, pick_label="Use this dataset"))

    def on_set_role(self, role: str = "", value: str = "", **_):
        if role in ("uuid_value", "crop_value", "label_value"):
            setattr(self, role, "" if value == NO_UUID_KEY else value)

    def on_set_image_root(self, value: str = "", **_):
        self.image_root = (value or "").strip()

    def on_browse_root(self, **_):
        def picked(p):
            self.image_root = p
            self.refresh()
        self._browse(picked, self.image_root, dirs_only=True, title="Select image root folder")

    def on_set_filter_enabled(self, value=False, **_):
        self.row_filter_enabled = bool(value)
        self.refresh()

    def on_set_filter_col(self, value: str = "", **_):
        self.row_filter_col = (value or "").strip()
        self.refresh()

    def on_set_filter_value(self, value: str = "", **_):
        self.row_filter_value = value or ""

    def on_set_split(self, kind: str = SPLIT_TEST, **_):
        """Applied live: realising after a load that this was the train pool
        is legitimate (see UploadView._on_split_change)."""
        self.split_kind = kind if kind in (SPLIT_TEST, SPLIT_TRAIN, SPLIT_OTHER) else SPLIT_TEST
        self.state.split_kind = self.split_kind
        self.s.notify_change()

    def on_load_dataset(self, **_):
        if self.ds_loading:
            return
        self.ds_loading = True
        self.ds_status = ("Loading...", "busy")
        self.ds_progress = 0.0
        self.refresh()
        self._bg(self.load_dataset_now)

    def load_dataset_now(self, then_experiments=None) -> bool:
        """The dataset load, on whatever thread calls it. Also what
        --dataset runs at startup (then_experiments: --experiment pairs)."""
        state = self.state
        self.ds_loading = True
        self.ds_status = ("Reading the file...", "busy")
        self.ds_progress = None
        self.s.busy.begin("Loading dataset")
        loaded = False

        def progress_cb(done: int, total: int):
            self.ds_progress = done / total if total else 0
            if total and done >= total:
                # Read; what is left (labels, types, the column kinds) has
                # no count to show.
                self.ds_progress = None
                self.ds_status = (f"Preparing {total:,} rows...", "busy")
            else:
                self.ds_status = (f"Loading... {done:,} / {total:,} rows", "busy")
            self.s.busy.detail_text(f"{done:,} / {total:,} rows")
            self._push()

        try:
            columns = ColumnConfig(uuid_col=self.uuid_value, crop_col=self.crop_value or "crop",
                                   label_col=self.label_value or "label")
            row_filter = None
            if self.row_filter_enabled and self.row_filter_col.strip():
                row_filter = (self.row_filter_col.strip(), self.row_filter_value)
            state.split_kind = self.split_kind or SPLIT_TEST
            state.load_dataset(self.dataset_path.strip(), columns, progress_cb=progress_cb,
                               row_filter=row_filter, image_root=self.image_root.strip())
            message = f"Loaded {len(state.dataset.full_df):,} rows, {len(state.dataset.metadata_columns())} metadata columns"
            if state.dataset.is_filtered():
                message += f" -- {len(state.dataset.df):,} kept by the Filter tab"
            elif state.dataset_filter_error:
                message += " -- the Filter tab's filter could not be applied"
            if not columns.has_ids():
                message += " -- no uuid column, so no experiment can be loaded against it"
            self.ds_status = (message, "success")
            loaded = True
            for row in self.rows:
                if row.loaded_name and row.loaded_name not in state.experiments:
                    row.loaded_name = None
                    row.overlap = None
                    row.status = ("Unloaded: the dataset has no uuid column", "warning")
                elif row.loaded_name:
                    row.overlap = state.uuid_overlap(row.loaded_name)
            self.relabel_picked = []
            self.label_names = {}
            self.s.notify_change()
        except Exception as ex:  # noqa: BLE001
            print(f"[lookin] error loading dataset from {self.dataset_path.strip()!r}:")
            traceback.print_exc()
            self.ds_status = (f"Error: {ex}", "error")
        finally:
            self.s.busy.end()
            self.ds_loading = False
            self.ds_progress = None
            self._push(force=True)
        if state.dataset is not None and loaded:
            self._load_held()
            for name, path in then_experiments or []:
                ok, message = self.add_experiment_from_path(path, name)
                if not ok:
                    print(f"[lookin] warning: could not auto-load experiment '{name}' from {path}: {message}", flush=True)
            from lookin.web.views.workspaces import adopt_dataset
            adopt_dataset(self.s)
        return loaded

    def on_clear_all(self, **_):
        self.state.clear_all()
        self.held = []
        self.ds_status = ("Not loaded", "neutral")
        self.rows = []
        self.inf_rows = []
        self.label_names = {}
        self.s.notify_change()

    def on_toggle_relabel(self, **_):
        self.relabel_open = not self.relabel_open
        self.refresh()

    def on_relabel_query(self, value: str = "", **_):
        self.relabel_query = value or ""
        self.refresh()

    def on_relabel_pick(self, label: str = "", **_):
        if label and label not in self.relabel_picked:
            self.relabel_picked.append(label)
        self.refresh()

    def on_apply_relabel(self, **values):
        if self.state.dataset is None:
            return
        names = self._label_names()
        shown = [l for l in getattr(self, "_relabel_shown", []) if l in names]
        mapping = {}
        for i, orig in enumerate(shown):
            current = names[orig]
            new = str(values.get(f"r{i}", current) or "").strip()
            # Two classes merged earlier share a name; the first field
            # changed says what that name becomes.
            if new and new != current and current not in mapping:
                mapping[current] = new
        if mapping:
            self.state.apply_label_remap(mapping)
            self.label_names = {orig: mapping.get(cur, cur) for orig, cur in names.items()}
            self.s.notify_change()
            self.s.toast(f"Relabelled {len(mapping)} class(es).", "ok")
        self.refresh()

    # ------------------------------------------------------------------
    # experiment actions
    # ------------------------------------------------------------------

    def on_add_row(self, **_):
        row = ExpRow(f"experiment_{len(self.rows) + 1}", "")
        row.open = True
        self.rows.append(row)
        self.refresh()

        def picked(p):
            row.path = p
            self.refresh()
        self._browse(picked, "", {".csv"}, title="Select results CSV")

    def on_from_mlflow(self, **_):
        from lookin.web.views.mlflow_dialogs import RunsDialog
        self.s.open_dialog(RunsDialog(self.s, mode="session"))

    def on_toggle_row(self, row=0, **_):
        r = self._row(row)
        if r is not None:
            r.open = not r.open
        self.refresh()

    def on_remove_row(self, row=0, **_):
        r = self._row(row)
        if r is None:
            return
        if r.loaded_name:
            self.state.remove_experiment(r.loaded_name)
            self.s.notify_change()
        self.rows.remove(r)
        self.refresh()

    def on_set_row_field(self, row=0, field: str = "", value: str = "", **_):
        r = self._row(row)
        if r is not None and field in ("path", "embed_path", "proto_path"):
            setattr(r, field, (value or "").strip())

    def on_browse_row(self, row=0, **_):
        r = self._row(row)
        if r is None:
            return

        def picked(p):
            r.path = p
            self.refresh()
        self._browse(picked, r.path, {".csv"}, title="Select results CSV")

    def on_browse_embed(self, row=0, **_):
        r = self._row(row)
        if r is None:
            return

        def picked(p):
            r.embed_path = p
            self.refresh()
        self._browse(picked, r.embed_path or r.path, {".npz"}, title="Select embeddings .npz")

    def on_browse_proto(self, row=0, **_):
        r = self._row(row)
        if r is None:
            return

        def picked(p):
            r.proto_path = p
            self.refresh()
        self._browse(picked, r.proto_path or r.path, {".npz"}, title="Select model prototypes .npz")

    def on_color(self, name: str = "", value: str = "", **_):
        if self.state.set_experiment_color(name, value):
            self.s.notify_colors()

    def on_rename(self, row=0, value: str = "", **_):
        """A loaded experiment's name is a label, nothing else: renaming
        keeps every cached metric (see UploadView._on_name_committed)."""
        r = self._row(row)
        if r is None:
            return
        new_name = (value or "").strip()
        if not r.loaded_name:
            r.name = new_name
            return
        loaded = r.loaded_name
        if new_name == loaded:
            return
        if not new_name:
            r.name = loaded
            self.refresh()
            return
        try:
            self.state.rename_experiment(loaded, new_name)
        except (KeyError, ValueError) as ex:
            r.name = loaded
            r.status = (f"Not renamed: {ex}", "warning")
            self.refresh()
            return
        r.loaded_name = new_name
        r.name = new_name
        r.overlap = self.state.uuid_overlap(new_name)
        r.status = (f"Renamed to '{new_name}'", "success")
        self.s.notify_rename(loaded, new_name)

    def on_load_row(self, row=0, **_):
        r = self._row(row)
        if r is None or r.loading:
            return
        r.loading = True
        r.status = ("Loading...", "busy")
        r.progress = 0.0
        self.refresh()
        self._bg(self._load_row_sync, r)

    def _load_row_sync(self, r: ExpRow) -> bool:
        state = self.state
        if state.dataset is None:
            r.loading = False
            r.status = ("Load the dataset first", "warning")
            return False
        if not state.dataset.columns.has_ids():
            r.loading = False
            r.status = ("The dataset has no uuid column -- nothing to match these results to", "warning")
            return False
        path = r.path.strip()
        name = (r.name or "").strip() or f"experiment_{len(self.rows)}"
        if not path:
            r.loading = False
            r.status = ("Enter a CSV path", "warning")
            return False
        r.loading = True
        r.progress = None
        r.status = ("Reading the file...", "busy")
        self._push(force=True)
        self.s.busy.begin(f"Loading experiment '{name}'")

        def progress_cb(done: int, total: int):
            r.progress = done / total if total else 0
            if total and done >= total:
                r.progress = None
                r.status = (f"Matching {total:,} rows...", "busy")
            else:
                r.status = (f"Loading... {done:,} / {total:,} rows", "busy")
            self.s.busy.detail_text(f"{done:,} / {total:,} rows")
            self._push()

        try:
            if r.loaded_name and r.loaded_name != name:
                state.remove_experiment(r.loaded_name)
            exp = state.add_experiment(path, name, uuid_col=self.uuid_value or "uuid", progress_cb=progress_cb)
            r.status = ("Finalizing...", "busy")
            self.s.busy.detail_text("finalizing")
            r.loaded_name = name
            r.name = name
            msg = f"Loaded ({len(exp.df):,} rows, {exp.n_classes()} classes)"
            msg += self._auto_attach_sidecars(path, r)
            r.status = (msg, "success")
            r.overlap = state.uuid_overlap(name)
            self.s.notify_change()
            ok = True
        except Exception as ex:  # noqa: BLE001
            print(f"[lookin] error loading experiment '{name}' from {path}:")
            traceback.print_exc()
            r.loaded_name = None
            r.overlap = None
            r.status = (f"Error: {ex}", "error")
            ok = False
        finally:
            self.s.busy.end()
            r.loading = False
            r.progress = None
            self._push(force=True)
        return ok

    def _auto_attach_sidecars(self, path: str, r: ExpRow) -> str:
        embed_path, proto_path = sidecar_paths_for_csv(path)
        for key, guessed in (("embed", embed_path), ("proto", proto_path)):
            current = getattr(r, f"{key}_path").strip()
            if not current or current == getattr(r, f"guessed_{key}"):
                setattr(r, f"{key}_path", str(guessed))
            setattr(r, f"guessed_{key}", str(guessed))
        notes = []
        if embed_path.exists() and self._load_sidecar(r, "embed"):
            notes.append(r.embed_status[0])
        if proto_path.exists() and self._load_sidecar(r, "proto"):
            notes.append(r.proto_status[0])
        return f" · sidecar detected: {', '.join(notes)}" if notes else ""

    def _load_sidecar(self, r: ExpRow, kind: str) -> bool:
        name = r.loaded_name
        path = (r.embed_path if kind == "embed" else r.proto_path).strip()
        attr = "embed_status" if kind == "embed" else "proto_status"
        if not name:
            setattr(r, attr, ("load the experiment first", "warning"))
            return False
        if not path:
            setattr(r, attr, ("enter a path", "warning"))
            return False
        what = "embeddings" if kind == "embed" else "prototypes"
        self.s.busy.begin(f"Loading {what} for '{name}'")
        try:
            loader = (self.s.prototype_source.load_embeddings_for if kind == "embed"
                      else self.s.prototype_source.load_prototypes_for)
            ok, msg = loader(name, path)
        finally:
            self.s.busy.end()
        setattr(r, attr, (msg if ok else f"error: {msg}", "neutral" if ok else "error"))
        if ok:
            self.s.notify_change()
        return ok

    def on_load_embed(self, row=0, **_):
        r = self._row(row)
        if r is not None:
            self._bg(self._load_sidecar, r, "embed")

    def on_load_proto(self, row=0, **_):
        r = self._row(row)
        if r is not None:
            self._bg(self._load_sidecar, r, "proto")

    def on_set_mapping(self, row=0, index=0, value: str = "", **_):
        r = self._row(row)
        if r is None or not r.loaded_name or r.loaded_name not in self.state.experiments:
            return
        if self.state.dataset is None or value not in set(self.state.dataset.labels()):
            self.s.toast("Only a class of the dataset can be picked.", "warn")
            self.refresh()
            return
        self.state.set_class_label(r.loaded_name, int(index), value)
        self.s.notify_change()

    def on_match_by_name(self, row=0, **_):
        r = self._row(row)
        if r is None or not r.loaded_name or r.loaded_name not in self.state.experiments:
            return
        exp = self.state.experiments[r.loaded_name]
        changed = False
        for i, col in enumerate(exp.score_cols):
            if i < len(exp.class_labels) and exp.class_labels[i] != col:
                self.state.set_class_label(r.loaded_name, i, col)
                changed = True
        if changed:
            self.s.notify_change()
        self.refresh()

    def on_mapping_query(self, row=0, value: str = "", **_):
        r = self._row(row)
        if r is not None:
            r.mapping_query = value or ""
        self.refresh()

    def on_mapping_pick(self, row=0, column: str = "", **_):
        r = self._row(row)
        if r is not None and column and column not in r.mapping_picked:
            r.mapping_picked.append(column)
        self.refresh()

    def add_experiment_from_path(self, path: str, suggested_name: str, row: Optional[ExpRow] = None) -> tuple:
        """Add a row for an already-available CSV (or use `row`, one already
        listed) and load it now, on the calling thread. Used by MLflow
        loads, --experiment and restores."""
        if self.state.dataset is None:
            return False, "Load the dataset first (Sources tab)."
        if row is None:
            row = ExpRow(suggested_name, path)
            self.rows.append(row)
        ok = self._load_row_sync(row)
        return ok, ("Loaded successfully." if ok else "Failed to load -- check the Sources tab for details.")

    def set_row_sidecar_fields(self, name: str, embed_path=None, embed_status=None,
                               proto_path=None, proto_status=None) -> None:
        for r in self.rows:
            if r.loaded_name != name:
                continue
            if embed_path is not None:
                r.embed_path = embed_path
            if embed_status is not None:
                r.embed_status = (embed_status, "neutral")
            if proto_path is not None:
                r.proto_path = proto_path
            if proto_status is not None:
                r.proto_status = (proto_status, "neutral")
            self.s.mark(self.name)
            return

    # ------------------------------------------------------------------
    # simple inferences
    # ------------------------------------------------------------------

    def on_add_inference(self, **_):
        self.inf_rows.append(InfRow(f"inference_{len(self.inf_rows) + 1}", "",
                                    self.uuid_value or "uuid", self.crop_value or "crop"))
        self.refresh()

    def on_set_inf_field(self, row=0, field: str = "", value: str = "", **_):
        r = self._inf(row)
        if r is None or field not in ("name", "path", "uuid_col", "crop_col", "embed_path", "image_root"):
            return
        setattr(r, field, (value or "").strip() if field != "name" else (value or ""))
        if field == "path":
            self._guess_inference_sidecar(r)
            self.refresh()

    def _guess_inference_sidecar(self, r: InfRow) -> None:
        if not r.path:
            return
        guess = str(sidecar_paths_for_csv(r.path)[0])
        current = (r.embed_path or "").strip()
        if current and current != (r.guessed_embed or ""):
            return
        r.guessed_embed = guess
        r.embed_path = guess

    def on_browse_inf(self, row=0, field: str = "path", **_):
        r = self._inf(row)
        if r is None:
            return

        def picked(p):
            setattr(r, field, p)
            if field == "path":
                self._guess_inference_sidecar(r)
            self.refresh()
        if field == "image_root":
            self._browse(picked, r.image_root, dirs_only=True, title="Select image root folder")
        elif field == "embed_path":
            self._browse(picked, r.embed_path or r.path, {".npz"}, title="Select embeddings .npz")
        else:
            self._browse(picked, r.path, {".csv"}, title="Select crops CSV")

    def on_load_inf(self, row=0, **_):
        r = self._inf(row)
        if r is None or r.loading:
            return
        r.loading = True
        r.status = ("Loading...", "busy")
        self.refresh()
        self._bg(self._load_inf_sync, r)

    def _load_inf_sync(self, r: InfRow) -> bool:
        state = self.state
        if state.dataset is None:
            r.loading = False
            r.status = ("Load the dataset first", "warning")
            return False
        name = (r.name or "").strip()
        if not r.path or not r.embed_path:
            r.loading = False
            r.status = ("A crops CSV and an embeddings .npz are both required", "warning")
            return False
        r.progress = 0.0
        self.s.busy.begin(f"Loading inference '{name}'")

        def progress_cb(done: int, total: int):
            r.progress = done / total if total else 0
            r.status = (f"Loading... {done:,} / {total:,} rows", "busy")
            self.s.busy.detail_text(f"{done:,} / {total:,} rows")
            self._push()

        try:
            if r.loaded_name and r.loaded_name != name:
                state.remove_inference(r.loaded_name)
            inference = state.load_inference(name, r.path, r.embed_path, uuid_col=r.uuid_col or "uuid",
                                             crop_col=r.crop_col or "crop", image_root=r.image_root,
                                             progress_cb=progress_cb)
            r.loaded_name = inference.name
            kept = len(inference.df)
            message = f"Loaded {kept:,} crops with embeddings"
            if kept < inference.csv_rows:
                message += (f" ({inference.csv_rows - kept:,} of {inference.csv_rows:,} CSV rows have "
                            f"no embedding and are not shown)")
            r.status = (message, "success")
            self.s.notify_change()
            ok = True
        except Exception as ex:  # noqa: BLE001
            print(f"[lookin] error loading inference '{name}' from {r.path}:")
            traceback.print_exc()
            r.loaded_name = None
            r.status = (f"Error: {ex}", "error")
            ok = False
        finally:
            self.s.busy.end()
            r.loading = False
            r.progress = None
            self._push(force=True)
        return ok

    def on_remove_inf(self, row=0, **_):
        r = self._inf(row)
        if r is None:
            return
        if r.loaded_name:
            self.state.remove_inference(r.loaded_name)
            self.s.notify_change()
        self.inf_rows.remove(r)
        self.refresh()

    # ------------------------------------------------------------------
    # the mount: what is loaded, in the shape the library stores
    # ------------------------------------------------------------------

    def current_mount(self):
        state = self.state
        if state.dataset is None or state.dataset.csv_path is None:
            return None
        row_filter = state.dataset.row_filter or ("", "")
        store = getattr(state, "review", None)
        origins = self._origins()
        experiments = []
        for name, exp in state.experiments.items():
            path = str(getattr(exp, "source_path", "") or self._path_for_experiment(name))
            loaded = library.MountExperiment(
                name=name, path=path,
                embeddings_path=str(exp.embeddings_path or ""),
                prototypes_path=str(exp.model_prototypes_path or ""),
                color=state.experiment_colors.get(name, ""),
            )
            origin = origins.get((name, path))
            if origin is not None:
                loaded = replace(origin, embeddings_path=loaded.embeddings_path,
                                 prototypes_path=loaded.prototypes_path, color=loaded.color)
            experiments.append(loaded)
        return library.Mount(
            dataset_path=str(state.dataset.csv_path),
            uuid_col=state.dataset.columns.uuid_col,
            crop_col=state.dataset.columns.crop_col,
            label_col=state.dataset.columns.label_col,
            image_root=str(state.dataset.image_root or ""),
            row_filter_col=str(row_filter[0] or ""),
            row_filter_value=str(row_filter[1] or ""),
            split_kind=state.split_kind,
            review_version=str(getattr(store, "active_version_id", "") or ""),
            experiments=[e for e in experiments if e.path],
            dataset_filter=dict(state.dataset_filter or {}),
        )

    def _path_for_experiment(self, name: str) -> str:
        for r in self.rows:
            if r.loaded_name == name:
                return r.path.strip()
        return ""

    def restore_mount(self, mount) -> None:
        """Refill every field from a remembered mount and load it, on the
        calling thread: the dataset, then each experiment with its sidecars.
        Per-experiment failures are not fatal (see UploadView.restore_mount)."""
        self.dataset_path = mount.dataset_path
        self.uuid_value = mount.uuid_col
        self.crop_value = mount.crop_col
        self.label_value = mount.label_col
        self.image_root = mount.image_root
        self._sniff()
        self.row_filter_enabled = bool(mount.row_filter_col)
        if mount.row_filter_col:
            self.row_filter_col = mount.row_filter_col
            self.row_filter_value = mount.row_filter_value
        self.split_kind = mount.split_kind if mount.split_kind in (SPLIT_TEST, SPLIT_TRAIN, SPLIT_OTHER) else SPLIT_TEST
        # Set before the load, which applies it: the filter is part of what
        # the workspace is, not a view of it.
        self.state.set_dataset_filter(getattr(mount, "dataset_filter", None) or {},
                                      csv_path=mount.dataset_path or "")
        self.s.view("filter").adopt_applied()
        for r in list(self.rows):
            if r.loaded_name:
                self.state.remove_experiment(r.loaded_name)
        self.rows = []
        self.held = []
        # Every experiment listed from the start, each loading in its own
        # row in turn -- not rows appearing one by one as each is done.
        self._hold(mount.experiments, "Waiting for the dataset...")
        self._push(force=True)
        self.s.busy.begin(f"Restoring {mount.label()}")
        try:
            self._restore_loads(mount)
        finally:
            self.s.busy.end()

    def _restore_loads(self, mount) -> None:
        if mount.dataset_path:
            if not self.load_dataset_now() and self.state.dataset is not None:
                # A failed load leaves the *previous* dataset in place, and
                # this workspace's experiments would then be scored against
                # it: plausible numbers from the wrong pool. Its error stays
                # on screen; its experiments wait for a dataset below.
                self.state.clear_all()
                self.s.notify_change()
        else:
            self.state.clear_all()
            self.ds_status = ("This workspace has no dataset yet -- pick its CSV above and click Load dataset", "neutral")
            self.s.notify_change()
        if self.state.dataset is None:
            for r in self.rows:
                if r.held:
                    r.status = ("Loads once the dataset is loaded", "neutral")
            self._push(force=True)
            return
        store = getattr(self.state, "review", None)
        if store is not None and mount.review_version and mount.review_version in store.versions:
            store.active_version_id = mount.review_version
        held, self.held = self.held, []
        self._load_mount_experiments(held)
        self.s.notify_change()

    def add_mount_experiments(self, experiments) -> None:
        """Load a workspace's experiments into what is already loaded --
        runs added to the open workspace from its Home card (see
        workspaces.follow_workspace)."""
        if self.state.dataset is None or not experiments:
            return
        self.s.busy.begin("Loading experiments")
        try:
            self._load_mount_experiments(experiments)
        finally:
            self.s.busy.end()
        self.s.notify_change()

    def is_loading(self) -> bool:
        return bool(self.ds_loading or any(getattr(r, "loading", False) for r in self.rows))

    def _hold(self, experiments, status: str = "Loads once the dataset is loaded") -> None:
        self.held = [e for e in experiments if e.path]
        for e in self.held:
            r = ExpRow(e.name, e.path)
            r.held = True
            r.embed_path = e.embeddings_path
            r.proto_path = e.prototypes_path
            r.status = (status, "neutral")
            self.rows.append(r)

    def _load_held(self) -> None:
        held, self.held = self.held, []
        if not held:
            return
        self._load_mount_experiments(held)
        self.s.notify_change()

    def _take_held_row(self, name: str, path: str) -> Optional[ExpRow]:
        """The row waiting for this experiment, if one is: it is loaded in
        place, where the user already sees it."""
        row = next((r for r in self.rows if r.held and r.name == name and r.path == path), None)
        if row is not None:
            row.held = False
        return row

    def _load_mount_experiments(self, experiments) -> None:
        for e in experiments:
            if not e.path:
                continue
            ok, message = self.add_experiment_from_path(e.path, e.name, row=self._take_held_row(e.name, e.path))
            if not ok:
                print(f"[lookin] library: could not restore experiment {e.name!r} from {e.path}: {message}", flush=True)
                continue
            self.state.set_experiment_color(e.name, e.color)
            for path, loader, kind in (
                (e.embeddings_path, self.s.prototype_source.load_embeddings_for, "embeddings"),
                (e.prototypes_path, self.s.prototype_source.load_prototypes_for, "prototypes"),
            ):
                if not path or not Path(path).is_file():
                    continue
                ok, note = loader(e.name, path)
                row = next((r for r in self.rows if r.loaded_name == e.name), None)
                if row is not None:
                    if kind == "embeddings":
                        row.embed_path, row.embed_status = path, (note if ok else f"error: {note}", "neutral" if ok else "error")
                    else:
                        row.proto_path, row.proto_status = path, (note if ok else f"error: {note}", "neutral" if ok else "error")
                if not ok:
                    print(f"[lookin] library: could not restore {kind} for {e.name!r}: {note}", flush=True)

    def rename_experiment(self, old: str, new: str) -> None:
        for r in self.rows:
            if r.loaded_name == old:
                r.loaded_name = new
                r.name = new
