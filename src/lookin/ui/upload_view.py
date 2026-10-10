"""Tab 1: load the dataset CSV and, independently, one or more results CSVs
(each becomes a named "experiment"). Loading the dataset and loading an
experiment are separate actions with their own status indicators -- the
dataset alone is enough to explore Statistics and the Image gallery; the
Metrics tab additionally needs at least one loaded experiment.

Each experiment keeps its own class_i -> label mapping (different model
runs can use different class orderings). The dataset's own label values
can also be renamed ("relabeled"), which cascades into every loaded
experiment's mapping so predictions keep matching up.

Editing the name of an already-loaded experiment renames it everywhere
(charts, legends, dropdowns) without re-reading its CSV or discarding a
single cached metric -- see _on_name_committed.
"""
from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path

import flet as ft

from lookin.core import library
from lookin.core.class_focus import FOCUS_MAX
from lookin.ui.class_matrix import ClassPicker
from lookin.core.data_manager import (
    SPLIT_KINDS, SPLIT_OTHER, SPLIT_TEST, SPLIT_TRAIN, ColumnConfig, sidecar_paths_for_csv,
)
from lookin.ui.components import section_title, info_banner, card, ACCENT, status_row as _status_control, uuid_overlap_banner
from lookin.core.schema import sniff_csv
from lookin.ui.server_browser import ServerFileBrowser

# Bundled example data ships inside the package itself (src/lookin/data/example)
# so `lookin ui` has something to load out of the box regardless of install location.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# Up to this many classes, the class mapping shows one picker per results
# column; past it, only the columns that match no dataset class by name.
MAPPING_MAX_ALL = FOCUS_MAX
# The uuid picker's "no id column" choice. A key, compared in code (see
# the dropdown note in CLAUDE.md), and never a column name: it maps to an
# empty uuid_value, which is what ColumnConfig takes to mean "no ids".
NO_UUID_KEY = "__no_uuid__"
EXAMPLE_DATASET = PACKAGE_ROOT / "data" / "example" / "dataset.csv"
EXAMPLE_EXPERIMENTS = [
    ("baseline_cnn", PACKAGE_ROOT / "data" / "example" / "baseline_cnn_results.csv"),
    ("finetuned_resnet", PACKAGE_ROOT / "data" / "example" / "finetuned_resnet_results.csv"),
]
# The bundled example's embeddings/prototypes (baseline_cnn_embeddings.npz,
# baseline_cnn_prototypes.npz, ...) aren't special-cased -- they just follow
# the same by-convention sidecar naming that _auto_attach_sidecars() looks
# for next to *any* loaded results CSV, local or from MLflow.


class UploadView:
    def __init__(self, ctx, initial_dataset: str | None = None):
        self.ctx = ctx
        self.experiment_rows: list[dict] = []
        self._remap_fields: list[tuple] = []
        self._dataset_loading = False
        # A workspace's experiments while it has no dataset loaded -- opened
        # before its dataset was set, or with one that failed to load. They
        # load as soon as a dataset does (see _hold_experiments).
        self._held_experiments: list = []

        self.dataset_path_field = ft.TextField(
            label="Dataset CSV path", value=initial_dataset or str(EXAMPLE_DATASET), expand=True,
            border_radius=8, prefix_icon=ft.Icons.DESCRIPTION_OUTLINED,
            on_blur=self._on_dataset_path_settled, on_submit=self._on_dataset_path_settled,
        )
        # The three roles are picked from the file's own columns rather than
        # typed. The values live here rather than being read off the
        # controls, because the controls are rebuilt whenever the file
        # changes -- see _rebuild_column_pickers. An empty uuid_value is the
        # explicit "this dataset has no id column" (see NO_UUID_KEY).
        self.uuid_value = "uuid"
        self.crop_value = "crop"
        self.label_value = "label"
        self.column_options: list = []
        self.columns_container = ft.Container()
        self.schema_note = ft.Text(size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
        self.dataset_status = ft.Container(content=_status_control("Not loaded", "neutral"))
        self.dataset_progress = ft.ProgressBar(value=0, visible=False, color=ACCENT, bar_height=6, border_radius=4)

        # Optional row filter, applied to the dataset CSV right after
        # loading (e.g. keep only the "test" split) -- off entirely if the
        # column doesn't exist in a given CSV, and always overridable.
        self.row_filter_enabled = ft.Checkbox(label="Filter rows where", value=True)
        self.row_filter_col_field = ft.TextField(label="column", value="split", width=140, border_radius=8, dense=True)
        self.row_filter_equals_text = ft.Text("=", size=13)
        self.row_filter_value_field = ft.TextField(label="value", value="test", width=140, border_radius=8, dense=True)

        # Which pool this is. Not cosmetic: predictions on a train pool are
        # in-sample, so the Cleaning tab has to warn differently there, and
        # bulk exclusion is a train-only operation (on test it would just
        # be a way to fake the metric). Defaults to test because that is
        # what the row filter above defaults to.
        self.split_dropdown = ft.Dropdown(
            label="This pool is", width=170, border_radius=8, dense=True,
            options=[
                ft.dropdown.Option(SPLIT_TEST, "test (held out)"),
                ft.dropdown.Option(SPLIT_TRAIN, "train (used to fit)"),
                ft.dropdown.Option(SPLIT_OTHER, "something else"),
            ],
            value=SPLIT_TEST, on_change=self._on_split_change,
        )

        # Optional override for where relative crop paths resolve from --
        # by default they're joined against the CSV's own directory.
        self.image_root_field = ft.TextField(
            label="Image root path (optional)", value="", expand=True,
            border_radius=8, prefix_icon=ft.Icons.FOLDER_OPEN,
            hint_text="Defaults to the dataset CSV's own directory",
        )

        # Browses the *server's* filesystem (the machine running `lookin
        # ui`), not the client's -- a browser's native file input (Flet's
        # own FilePicker) never hands back a real path, only a bare
        # filename, no matter the browser (sandboxing, not a Flet
        # limitation). Since this is the same machine the loaded files
        # actually live on, this dialog always resolves to the complete,
        # absolute path. See ui/server_browser.py.
        self.server_browser = ServerFileBrowser(ctx)

        self.load_dataset_btn = ft.FilledButton(
            "Load dataset", icon=ft.Icons.UPLOAD_ROUNDED, on_click=self._on_load_dataset_click, style=ft.ButtonStyle(bgcolor=ACCENT),
        )
        self.clear_all_btn = ft.OutlinedButton(
            "Clear all data", icon=ft.Icons.DELETE_SWEEP_OUTLINED, on_click=self._on_clear_all_click,
        )

        self.experiments_column = ft.Column(spacing=16)  # visible gap between one experiment and the next
        self.inference_rows: list[dict] = []
        self.inferences_column = ft.Column(spacing=16)
        self.dataset_extra_column = ft.Column(spacing=14)

        # The bundled example rows are only useful as a "click Load to see
        # it work" default -- once the user (or --dataset/--experiment on
        # the command line) points at their own data, showing the example
        # experiment rows alongside it is just noise, not a fallback.
        if initial_dataset is None:
            for name, path in EXAMPLE_EXPERIMENTS:
                self._add_experiment_row(name, str(path))

        self.root = ft.Container(padding=20)

    # ---- experiment rows ----
    def _add_experiment_row(self, name: str = "", path: str = "") -> None:
        idx = len(self.experiment_rows)
        # Renaming a *loaded* experiment is committed on blur/submit, not on
        # every keystroke -- but the text has to be captured on_change all
        # the same. Flet's TextField applies a keystroke to its own value
        # client-side and only flushes that to the server alongside the next
        # event it sends; typing and pressing Enter in one go races that
        # flush, and the blur/submit handler then reads the *previous* name
        # and does nothing. Registering on_change makes every keystroke an
        # event, so the value below is always the text on screen.
        name_field = ft.TextField(
            label="experiment name", value=name or f"experiment_{idx+1}", width=200, border_radius=8,
            on_change=lambda e: row.__setitem__("typed_name", e.data),
            on_blur=lambda e: self._on_name_committed(row),
            on_submit=lambda e: self._on_name_committed(row),
        )
        path_field = ft.TextField(label="results CSV path", value=path, expand=True, border_radius=8, prefix_icon=ft.Icons.DESCRIPTION_OUTLINED)
        status_container = ft.Container(content=_status_control("Not loaded", "neutral"))
        progress_bar = ft.ProgressBar(value=0, visible=False, color=ACCENT, bar_height=6, border_radius=4)
        mapping_container = ft.Column(spacing=8)
        overlap_container = ft.Container()

        # Embedding-space sidecars (see core/embedding_analysis.py) live here
        # rather than on the Embeddings tab -- one place to load a results
        # CSV plus whatever's attached to it, local or from MLflow (see
        # _auto_attach_sidecars / MlflowPanel._auto_load_embedding_sidecars,
        # both of which pre-fill and load these same two fields).
        embed_field = ft.TextField(label="embeddings .npz path (optional)", value="", expand=True, border_radius=8, dense=True, prefix_icon=ft.Icons.SCATTER_PLOT_OUTLINED)
        embed_status = ft.Text("not loaded", size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        proto_field = ft.TextField(label="model prototypes .npz path (optional)", value="", expand=True, border_radius=8, dense=True, prefix_icon=ft.Icons.HUB_OUTLINED)
        proto_status = ft.Text("not loaded", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

        load_btn = ft.IconButton(ft.Icons.DOWNLOAD_ROUNDED, tooltip="Load this experiment")

        row = {
            "name_field": name_field, "path_field": path_field,
            "status_container": status_container, "progress_bar": progress_bar, "mapping_container": mapping_container,
            "overlap_container": overlap_container,
            "embed_field": embed_field, "embed_status": embed_status,
            "proto_field": proto_field, "proto_status": proto_status,
            # Latest text typed into name_field, kept by its on_change --
            # the only reliable reading of it at blur/submit time (see above).
            "typed_name": None,
            # The paths _auto_attach_sidecars() last guessed for this row.
            # A field still holding its guess is "untouched" and gets
            # re-guessed when the results CSV changes; one the user edited
            # by hand no longer matches, and is left alone.
            "guessed_embed": None, "guessed_proto": None,
            "loaded_name": None, "container": None, "loading": False, "load_btn": load_btn,
        }

        def browse(e):
            self.server_browser.open(
                on_pick=lambda p: self._set_field(path_field, p),
                start_path=path_field.value.strip(), extensions={".csv"}, title="Select results CSV",
            )

        def load_click(e):
            if row["loading"]:
                return
            self._load_experiment_row(row)

        load_btn.on_click = load_click

        def remove_click(e):
            self._remove_row(row)

        def embed_browse(e):
            self.server_browser.open(
                on_pick=lambda p: self._set_field(embed_field, p),
                start_path=embed_field.value.strip() or path_field.value.strip(),
                extensions={".npz"}, title="Select embeddings .npz",
            )

        def embed_load_click(e):
            self._load_embeddings_for_row(row)

        def proto_browse(e):
            self.server_browser.open(
                on_pick=lambda p: self._set_field(proto_field, p),
                start_path=proto_field.value.strip() or path_field.value.strip(),
                extensions={".npz"}, title="Select model prototypes .npz",
            )

        def proto_load_click(e):
            self._load_prototypes_for_row(row)

        controls_row = ft.Row(controls=[
            name_field, path_field,
            ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=browse),
            load_btn,
            ft.IconButton(ft.Icons.DELETE_OUTLINE, tooltip="Remove", on_click=remove_click),
        ])
        sidecar_col = ft.Column([
            ft.Text(
                "Embedding-space sidecars (optional) -- auto-detected next to the results CSV "
                "(<name>_embeddings.npz / <name>_prototypes.npz) and loaded on their own; "
                "edit a path and press Load to attach a different file.",
                size=11, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row([
                embed_field,
                ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=embed_browse),
                ft.FilledButton("Load embeddings", icon=ft.Icons.UPLOAD_FILE, on_click=embed_load_click),
            ]),
            embed_status,
            ft.Row([
                proto_field,
                ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=proto_browse),
                ft.FilledButton("Load prototypes", icon=ft.Icons.UPLOAD_FILE, on_click=proto_load_click),
            ]),
            proto_status,
        ], spacing=4)
        # No divider between the experiment and its own sidecars: they are
        # one experiment, and a rule there read as a boundary between
        # experiments. The boundary that does exist -- one experiment to the
        # next -- is carried by the row's outline instead, which the shared
        # SURFACE_CONTAINER_HIGHEST background alone could not show (it is
        # the same color as the enclosing card).
        row_col = ft.Column(
            [controls_row, progress_bar, status_container, overlap_container, mapping_container, sidecar_col],
            spacing=6,
        )
        row_container = ft.Container(
            content=row_col, padding=12, border_radius=10,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        )
        row["container"] = row_container

        self.experiment_rows.append(row)
        self.experiments_column.controls.append(row_container)

    # ---- simple inferences ------------------------------------------------

    def _add_inference_row(self, name: str = "", path: str = "") -> None:
        """One loadable simple inference: a CSV of crops, its embeddings
        .npz, and where its images live.

        Deliberately shaped like an experiment row and deliberately not one
        of them: an experiment is scored against the dataset's labels,
        while this is a pool of crops nobody has a truth for (see
        core/data_manager.InferenceSet). Keeping them apart on screen is
        the same separation the state keeps in memory.
        """
        idx = len(self.inference_rows)
        name_field = ft.TextField(
            label="inference name", value=name or f"inference_{idx+1}", width=200, border_radius=8,
            # Same reason as the experiment rows: a keystroke only reaches
            # the server alongside the next event, so it is captured here.
            on_change=lambda e: row.__setitem__("typed_name", e.data),
        )
        path_field = ft.TextField(
            label="crops CSV path", value=path, expand=True, border_radius=8,
            prefix_icon=ft.Icons.DESCRIPTION_OUTLINED,
        )
        uuid_field = ft.TextField(label="id column", value=self.uuid_value or "uuid", width=140,
                                  border_radius=8, dense=True)
        crop_field = ft.TextField(label="crop path column", value=self.crop_value or "crop", width=170,
                                  border_radius=8, dense=True)
        embed_field = ft.TextField(
            label="embeddings .npz path", value="", expand=True, border_radius=8, dense=True,
            prefix_icon=ft.Icons.SCATTER_PLOT_OUTLINED,
        )
        image_root_field = ft.TextField(
            label="Image root path (optional)", value="", expand=True, border_radius=8, dense=True,
            prefix_icon=ft.Icons.FOLDER_OPEN,
            hint_text="Defaults to the crops CSV's own directory",
        )
        status_container = ft.Container(content=_status_control("Not loaded", "neutral"))
        progress_bar = ft.ProgressBar(value=0, visible=False, color=ACCENT, bar_height=6, border_radius=4)
        load_btn = ft.IconButton(ft.Icons.DOWNLOAD_ROUNDED, tooltip="Load this inference")

        row = {
            "name_field": name_field, "path_field": path_field, "uuid_field": uuid_field,
            "crop_field": crop_field, "embed_field": embed_field,
            "image_root_field": image_root_field, "status_container": status_container,
            "progress_bar": progress_bar, "load_btn": load_btn, "typed_name": None,
            # The path _guess_inference_sidecar last filled in, so a field
            # the user edited by hand is never re-guessed over.
            "guessed_embed": None,
            "loaded_name": None, "container": None, "loading": False,
        }

        def browse(e):
            self.server_browser.open(
                on_pick=lambda p: self._on_inference_csv_picked(row, p),
                start_path=path_field.value.strip(), extensions={".csv"},
                title="Select crops CSV",
            )

        def embed_browse(e):
            self.server_browser.open(
                on_pick=lambda p: self._set_field(embed_field, p),
                start_path=embed_field.value.strip() or path_field.value.strip(),
                extensions={".npz"}, title="Select embeddings .npz",
            )

        def root_browse(e):
            self.server_browser.open(
                on_pick=lambda p: self._set_field(image_root_field, p),
                start_path=image_root_field.value.strip(), dirs_only=True,
                title="Select image root folder",
            )

        load_btn.on_click = lambda e: None if row["loading"] else self._load_inference_row(row)
        path_field.on_blur = lambda e: self._guess_inference_sidecar(row)
        path_field.on_submit = lambda e: self._guess_inference_sidecar(row)

        row_col = ft.Column([
            ft.Row([
                name_field, path_field,
                ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=browse),
                load_btn,
                ft.IconButton(ft.Icons.DELETE_OUTLINE, tooltip="Remove",
                              on_click=lambda e: self._remove_inference_row(row)),
            ]),
            ft.Row([uuid_field, crop_field], spacing=8, wrap=True),
            ft.Row([
                embed_field,
                ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=embed_browse),
            ]),
            ft.Row([
                image_root_field,
                ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=root_browse),
            ]),
            progress_bar,
            status_container,
        ], spacing=6)
        row["container"] = ft.Container(
            content=row_col, padding=12, border_radius=10,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        )
        self.inference_rows.append(row)
        self.inferences_column.controls.append(row["container"])

    def _on_inference_csv_picked(self, row: dict, path: str) -> None:
        self._set_field(row["path_field"], path)
        self._guess_inference_sidecar(row)

    def _guess_inference_sidecar(self, row: dict) -> None:
        """Fill in the embeddings path from the CSV's name, by the same
        convention the experiment rows use (<name>_embeddings.npz beside
        it). Only ever a starting guess: a field the user has typed into no
        longer matches the last guess and is left alone."""
        path = row["path_field"].value.strip()
        if not path:
            return
        guess = str(sidecar_paths_for_csv(path)[0])
        field = row["embed_field"]
        current = (field.value or "").strip()
        if current and current != (row["guessed_embed"] or ""):
            return
        row["guessed_embed"] = guess
        self._set_field(field, guess)

    def _load_inference_row(self, row: dict) -> bool:
        state = self.ctx.state
        if state.dataset is None:
            # Not a technical requirement of the loading itself, but the
            # map these are drawn on is an experiment's, and everything
            # that resolves a crop path goes through the dataset.
            row["status_container"].content = _status_control("Load the dataset first", "warning")
            self.root.update()
            return False
        typed = row["typed_name"]
        name = (typed if typed is not None else row["name_field"].value or "").strip()
        path = row["path_field"].value.strip()
        embed_path = row["embed_field"].value.strip()
        if not path or not embed_path:
            row["status_container"].content = _status_control(
                "A crops CSV and an embeddings .npz are both required", "warning")
            self.root.update()
            return False

        row["loading"] = True
        row["load_btn"].disabled = True
        progress_bar = row["progress_bar"]
        progress_bar.visible = True
        progress_bar.value = 0
        row["status_container"].content = _status_control("Loading...", "busy")
        for control in (progress_bar, row["status_container"], row["load_btn"]):
            control.update()
        self.ctx.busy.begin(f"Loading inference '{name}'")

        def progress_cb(done: int, total: int):
            progress_bar.value = done / total if total else 0
            row["status_container"].content = _status_control(f"Loading... {done:,} / {total:,} rows", "busy")
            self.ctx.busy.detail_text(f"{done:,} / {total:,} rows")
            progress_bar.update()
            row["status_container"].update()

        try:
            if row["loaded_name"] and row["loaded_name"] != name:
                state.remove_inference(row["loaded_name"])
            inference = state.load_inference(
                name, path, embed_path,
                uuid_col=row["uuid_field"].value.strip() or "uuid",
                crop_col=row["crop_field"].value.strip() or "crop",
                image_root=row["image_root_field"].value.strip(),
                progress_cb=progress_cb,
            )
            row["loaded_name"] = inference.name
            kept = len(inference.df)
            message = f"Loaded {kept:,} crops with embeddings"
            if kept < inference.csv_rows:
                message += (f" ({inference.csv_rows - kept:,} of {inference.csv_rows:,} CSV rows have "
                            f"no embedding and are not shown)")
            row["status_container"].content = _status_control(message, "success")
            self.ctx.notify_change()
            success = True
        except Exception as ex:
            print(f"[lookin] error loading inference '{name}' from {path}:")
            traceback.print_exc()
            row["loaded_name"] = None
            row["status_container"].content = _status_control(f"Error: {ex}", "error")
            success = False
        finally:
            self.ctx.busy.end()
        progress_bar.visible = False
        row["loading"] = False
        row["load_btn"].disabled = False
        self.root.update()
        return success

    def _remove_inference_row(self, row: dict) -> None:
        if row not in self.inference_rows:
            return
        if row["loaded_name"]:
            self.ctx.state.remove_inference(row["loaded_name"])
            self.ctx.notify_change()
        self.inference_rows.remove(row)
        self.inferences_column.controls.remove(row["container"])
        self.root.update()

    def _on_add_inference_click(self, e):
        self._add_inference_row()
        self.root.update()

    def _remove_row(self, row: dict) -> None:
        if row not in self.experiment_rows:
            return
        if row["loaded_name"]:
            self.ctx.state.remove_experiment(row["loaded_name"])
            self.ctx.notify_change()
        self.experiment_rows.remove(row)
        self.experiments_column.controls.remove(row["container"])
        self.root.update()

    def _on_name_committed(self, row: dict) -> None:
        """Commit an edit to a loaded experiment's name field as a rename.

        A name is a label, nothing else: no loaded row, score or computed
        metric depends on it. So editing it must not mean pressing Load
        again and re-reading the results CSV -- at real sizes that read is
        the slowest thing the app does, and it would also throw away every
        cached metric, curve, kmeans estimate and prototype assignment
        computed for that experiment. AppState.rename_experiment keeps all
        of it and ctx.notify_rename lets each view relabel itself in place.

        Only reachable once the row is loaded; before that the field is
        just the name the next Load will use.
        """
        loaded = row["loaded_name"]
        if not loaded:
            return
        typed = row["typed_name"]
        new_name = (typed if typed is not None else row["name_field"].value or "").strip()
        if new_name == loaded:
            return
        if not new_name:
            # An empty name would be un-selectable in every dropdown that
            # lists experiments -- put the current one back rather than
            # silently renaming to "".
            self._reset_name_field(row, loaded)
            self.root.update()
            return
        try:
            self.ctx.state.rename_experiment(loaded, new_name)
        except (KeyError, ValueError) as ex:
            self._reset_name_field(row, loaded)
            row["status_container"].content = _status_control(f"Not renamed: {ex}", "warning")
            self.root.update()
            return

        row["loaded_name"] = new_name
        # Written back explicitly so the field, the tracked text and the
        # loaded name cannot drift apart -- the stripped name is what was
        # actually applied, and it is what the field must show.
        self._reset_name_field(row, new_name)
        # The class-mapping dropdowns close over the experiment name (see
        # _build_row_mapping), so they have to be rebuilt against the new one.
        self._build_row_mapping(row)
        overlap = self.ctx.state.uuid_overlap(new_name)
        row["overlap_container"].content = uuid_overlap_banner(overlap, "dataset", new_name)
        row["status_container"].content = _status_control(f"Renamed to '{new_name}'", "success")
        self.root.update()
        self.ctx.notify_rename(loaded, new_name)

    def _reset_name_field(self, row: dict, name: str) -> None:
        row["name_field"].value = name
        row["typed_name"] = name

    def _load_experiment_row(self, row: dict) -> bool:
        state = self.ctx.state
        if state.dataset is None:
            row["status_container"].content = _status_control("Load the dataset first", "warning")
            self.root.update()
            return False
        if not state.dataset.columns.has_ids():
            row["status_container"].content = _status_control(
                "The dataset has no uuid column -- nothing to match these results to", "warning")
            self.root.update()
            return False
        path = row["path_field"].value.strip()
        typed = row["typed_name"]
        name = (typed if typed is not None else row["name_field"].value or "").strip() \
            or f"experiment_{len(self.experiment_rows)}"
        if not path:
            row["status_container"].content = _status_control("Enter a CSV path", "warning")
            self.root.update()
            return False

        row["loading"] = True
        row["load_btn"].disabled = True
        progress_bar = row["progress_bar"]
        progress_bar.visible = True
        progress_bar.value = 0
        row["status_container"].content = _status_control("Loading...", "busy")
        progress_bar.update()
        row["status_container"].update()
        row["load_btn"].update()
        # ...and in the app bar too, so this is still visible to a user who
        # goes to look at another tab while a big results CSV reads.
        self.ctx.busy.begin(f"Loading experiment '{name}'")

        def progress_cb(done: int, total: int):
            progress_bar.value = done / total if total else 0
            row["status_container"].content = _status_control(f"Loading... {done:,} / {total:,} rows", "busy")
            self.ctx.busy.detail_text(f"{done:,} / {total:,} rows")
            progress_bar.update()
            row["status_container"].update()

        try:
            if row["loaded_name"] and row["loaded_name"] != name:
                state.remove_experiment(row["loaded_name"])
            uuid_col = self.uuid_value or "uuid"
            exp = state.add_experiment(path, name, uuid_col=uuid_col, progress_cb=progress_cb)
            # The CSV is read; what is left -- downcasting, the uuid overlap
            # check, the class mapping -- is fast but not free, and saying so
            # beats a bar that sits at 100% looking stuck.
            row["status_container"].content = _status_control("Finalizing...", "busy")
            self.ctx.busy.detail_text("finalizing")
            row["status_container"].update()
            row["loaded_name"] = name
            self._reset_name_field(row, name)
            status_msg = f"Loaded ({len(exp.df):,} rows, {exp.n_classes()} classes)"
            status_msg += self._auto_attach_sidecars(path, row)
            row["status_container"].content = _status_control(status_msg, "success")
            overlap = state.uuid_overlap(name)
            row["overlap_container"].content = uuid_overlap_banner(overlap, "dataset", name)
            self._build_row_mapping(row)
            self.ctx.notify_change()
            success = True
        except Exception as ex:
            # The UI only ever shows this one status line -- print the full
            # traceback too, since that's the only way to actually debug an
            # unexpected failure (bad CSV, permissions, encoding, etc).
            print(f"[lookin] error loading experiment '{name}' from {path}:")
            traceback.print_exc()
            row["loaded_name"] = None
            row["overlap_container"].content = None
            row["status_container"].content = _status_control(f"Error: {ex}", "error")
            success = False
        finally:
            self.ctx.busy.end()
        progress_bar.visible = False
        row["loading"] = False
        row["load_btn"].disabled = False
        self.root.update()
        return success

    def _auto_attach_sidecars(self, path: str, row: dict) -> str:
        """By-convention embeddings/prototypes discovery: next to any
        loaded results CSV (e.g. .../baseline_cnn_results.csv), look for
        baseline_cnn_embeddings.npz and baseline_cnn_prototypes.npz
        in the same directory, pre-fill this row's sidecar fields with those
        guessed paths, and load whichever actually exist. Silent (beyond the
        pre-fill) if neither is present -- most CSVs won't have them, and
        that's not an error. The fields stay editable either way, so this is
        only ever a starting guess, never a lock-in. (The MLflow panel's own
        sidecar auto-attach -- see MlflowPanel._auto_load_embedding_sidecars
        -- follows the same naming convention but checks/downloads from the
        run's artifact tree instead of the local filesystem, then calls
        set_row_sidecar_fields() below to keep this row in sync.)"""
        embed_path, proto_path = sidecar_paths_for_csv(path)
        # Overwrite a field that's empty or still holding this row's previous
        # guess (so pointing the row at a different results CSV re-guesses
        # instead of keeping the old CSV's sidecars), but never one the user
        # typed or browsed to themselves.
        for key, guessed in (("embed", embed_path), ("proto", proto_path)):
            field = row[f"{key}_field"]
            current = field.value.strip()
            if not current or current == row[f"guessed_{key}"]:
                field.value = str(guessed)
            row[f"guessed_{key}"] = str(guessed)
        notes = []
        if embed_path.exists() and self._load_embeddings_for_row(row):
            notes.append(row["embed_status"].value)
        if proto_path.exists() and self._load_prototypes_for_row(row):
            notes.append(row["proto_status"].value)
        return f" · sidecar detected: {', '.join(notes)}" if notes else ""

    def _load_embeddings_for_row(self, row: dict) -> bool:
        name = row["loaded_name"]
        path = row["embed_field"].value.strip()
        status = row["embed_status"]
        if not name:
            status.value, status.color = "load the experiment first", ft.Colors.ORANGE
        elif not path:
            status.value, status.color = "enter a path", ft.Colors.ORANGE
        else:
            # An embeddings sidecar for a large run is hundreds of
            # megabytes of npz; the app bar says it is being read.
            self.ctx.busy.begin(f"Loading embeddings for '{name}'")
            try:
                ok, msg = self.ctx.prototype_source.load_embeddings_for(name, path)
            finally:
                self.ctx.busy.end()
            status.value = msg if ok else f"error: {msg}"
            status.color = ft.Colors.ON_SURFACE_VARIANT if ok else ft.Colors.RED
            if status.page:
                status.update()
            if ok:
                self.ctx.notify_change()
            return ok
        if status.page:
            status.update()
        return False

    def _load_prototypes_for_row(self, row: dict) -> bool:
        name = row["loaded_name"]
        path = row["proto_field"].value.strip()
        status = row["proto_status"]
        if not name:
            status.value, status.color = "load the experiment first", ft.Colors.ORANGE
        elif not path:
            status.value, status.color = "enter a path", ft.Colors.ORANGE
        else:
            # Smaller than an embeddings sidecar (one vector per class, not
            # per crop), but announced the same way rather than differently.
            self.ctx.busy.begin(f"Loading prototypes for '{name}'")
            try:
                ok, msg = self.ctx.prototype_source.load_prototypes_for(name, path)
            finally:
                self.ctx.busy.end()
            status.value = msg if ok else f"error: {msg}"
            status.color = ft.Colors.ON_SURFACE_VARIANT if ok else ft.Colors.RED
            if status.page:
                status.update()
            if ok:
                self.ctx.notify_change()
            return ok
        if status.page:
            status.update()
        return False

    def set_row_sidecar_fields(
        self, name: str,
        embed_path: str | None = None, embed_status: str | None = None,
        proto_path: str | None = None, proto_status: str | None = None,
    ) -> None:
        """Called by MlflowPanel after it downloads+loads a sidecar from the
        run's artifact tree, so this row's fields reflect what actually got
        loaded (the local cache path) instead of staying blank."""
        for row in self.experiment_rows:
            if row["loaded_name"] != name:
                continue
            if embed_path is not None:
                row["embed_field"].value = embed_path
            if embed_status is not None:
                row["embed_status"].value = embed_status
                row["embed_status"].color = ft.Colors.ON_SURFACE_VARIANT
            if proto_path is not None:
                row["proto_field"].value = proto_path
            if proto_status is not None:
                row["proto_status"].value = proto_status
                row["proto_status"].color = ft.Colors.ON_SURFACE_VARIANT
            if row["container"].page:
                row["container"].update()
            return

    def add_experiment_from_path(self, path: str, suggested_name: str) -> tuple[bool, str]:
        """Used by the MLflow panel and by app.py's --experiment startup
        auto-load: add a new experiment row for an already-available CSV
        and load it immediately. Returns (success, message)."""
        if self.ctx.state.dataset is None:
            return False, "Load the dataset first (Data tab)."
        self._add_experiment_row(suggested_name, path)
        row = self.experiment_rows[-1]
        # The new row's controls (progress bar, status text, ...) only
        # exist in memory until something updates an already-mounted
        # ancestor -- without this, _load_experiment_row()'s own .update()
        # calls on those controls fail with "Control must be added to the
        # page first" the moment nothing else has rendered in between (the
        # interactive add-then-click-Load flow happens to dodge this since
        # there's always a render between those two separate user actions).
        if self.root.page:
            self.root.update()
        success = self._load_experiment_row(row)
        message = "Loaded successfully." if success else "Failed to load -- check the Data tab for details."
        return success, message

    def _build_row_mapping(self, row: dict) -> None:
        row["mapping_container"].controls.clear()
        state = self.ctx.state
        name = row["loaded_name"]
        if not name or name not in state.experiments or state.dataset is None:
            return
        exp = state.experiments[name]
        dataset_labels = state.dataset.labels()
        known = set(dataset_labels)
        # One picker per score column is fine for a dozen classes and a
        # freeze for two thousand (each listing every dataset label). Class
        # names come straight from the score-column headers, so almost
        # every column is already mapped by name: past MAPPING_MAX_ALL the
        # pickers start as the columns whose label is not a dataset class,
        # which are the ones that need a decision, and a search over every
        # results column adds a picker for any other -- matched or not --
        # so no column is ever out of reach.
        indices = list(range(exp.n_classes()))
        many = len(indices) > MAPPING_MAX_ALL
        n_matched = 0
        if many:
            unmatched = [i for i in indices if i >= len(exp.class_labels) or exp.class_labels[i] not in known]
            n_matched = len(indices) - len(unmatched)
            indices = unmatched
        shown = indices[:MAPPING_MAX_ALL]

        def source_col_of(i: int) -> str:
            return exp.score_cols[i] if i < len(exp.score_cols) else f"class_{i + 1}"

        mapping_row = ft.Row(wrap=True, spacing=8)

        def dropdown_for(i: int) -> ft.Dropdown:
            current = exp.class_labels[i] if i < len(exp.class_labels) else (dataset_labels[0] if dataset_labels else "")

            def on_change(e, idx=i, exp_name=name):
                state.set_class_label(exp_name, idx, e.control.value)
                self.ctx.notify_change()

            # The dropdown's label is the results CSV's own score-column
            # name for this class (e.g. "cat") -- the value is what dataset
            # label it maps to, editable in case the two don't spell it the
            # same way. Filterable by typing, since the list is every
            # dataset class.
            return ft.Dropdown(
                label=source_col_of(i), value=current,
                options=[ft.dropdown.Option(l) for l in dataset_labels],
                width=150 if len(dataset_labels) <= MAPPING_MAX_ALL else 220,
                enable_filter=len(dataset_labels) > MAPPING_MAX_ALL,
                editable=len(dataset_labels) > MAPPING_MAX_ALL,
                on_change=on_change, border_radius=8,
            )

        chooser = []
        if many:
            # The picked columns survive a rebuild (a rename, a relabel), so
            # the fields the user opened do not vanish under them.
            index_of = {source_col_of(i): i for i in range(exp.n_classes())}
            picked = [c for c in row.get("mapping_picked", []) if c in index_of]
            row["mapping_picked"] = picked
            dropdowns: dict = {}

            def lay_out() -> None:
                order = shown + [index_of[c] for c in picked if index_of[c] not in shown]
                for i in order:
                    if i not in dropdowns:
                        dropdowns[i] = dropdown_for(i)
                mapping_row.controls = [dropdowns[i] for i in order]

            def on_picked(columns: list) -> None:
                picked[:] = columns
                lay_out()
                if mapping_row.page:
                    mapping_row.update()

            picker = ClassPicker(on_picked, limit=None, label="Find a results column to map")
            picker.set_labels(list(index_of))
            picker.set_picked(picked)
            chooser = [picker.build()]
            lay_out()
        else:
            mapping_row.controls = [dropdown_for(i) for i in shown]

        notes = []
        if n_matched:
            notes.append(f"{n_matched:,} of {exp.n_classes():,} results columns match a dataset class by name "
                         f"and are mapped to it.")
            if not indices:
                notes.append("Every column is mapped -- nothing to decide.")
        if indices and n_matched:
            notes.append(f"{len(indices):,} column(s) name no class of this dataset -- a spelling "
                         f"difference, or simply a class with no crop here, which needs no mapping.")
        if len(indices) > len(shown):
            notes.append(f"The first {len(shown)} are listed below.")
        if many:
            notes.append("Search for any results column below to change what it maps to.")
        row["mapping_container"].controls.append(ft.Column([
            ft.Text("Class mapping (results column -> dataset label)", size=12, weight=ft.FontWeight.BOLD),
            *[ft.Text(n, size=12, color=ft.Colors.ON_SURFACE_VARIANT) for n in notes],
            *chooser,
            mapping_row,
        ], spacing=6))

    # ---- server-side file browser (see ui/server_browser.py) ----
    def _on_dataset_browse(self, e):
        self.server_browser.open(
            on_pick=self._on_dataset_picked,
            start_path=self.dataset_path_field.value.strip(), extensions={".csv"}, title="Select dataset CSV",
        )

    def _on_image_root_browse(self, e):
        self.server_browser.open(
            on_pick=lambda p: self._set_field(self.image_root_field, p),
            start_path=self.image_root_field.value.strip(), dirs_only=True, title="Select image root folder",
        )

    # ---- column mapping, guessed from the file ------------------------
    def _on_dataset_picked(self, path: str) -> None:
        self._set_field(self.dataset_path_field, path)
        self._sniff_dataset()

    def _on_dataset_path_settled(self, e) -> None:
        """Sniff when the field is done being edited, not on every
        keystroke: each sniff reads 200 rows, which is nothing locally and
        not nothing over a network mount."""
        self._sniff_dataset()

    def _sniff_dataset(self) -> None:
        """Read the header and the first rows, guess the three roles, and
        rebuild the pickers.

        A guess only ever fills a role whose current value is *not* a column
        of this file. That one rule covers both cases that matter: an
        explicit choice -- one the user made, or one restored from a
        saved mount -- survives untouched as long as the file still
        has that column, and a mapping pointing at a column this file does
        not have is replaced rather than left to fail the load with "column
        not found".
        """
        path = (self.dataset_path_field.value or "").strip()
        if not path:
            # Nothing to guess from -- and a note about the previous file
            # would read as a guess about this one.
            self.column_options = []
            self.schema_note.value = ""
            self._rebuild_column_pickers()
            self._push_column_pickers()
            return

        guess = sniff_csv(path)
        self.column_options = list(guess.columns)
        for attr, guessed in (("uuid_value", guess.uuid_col),
                              ("crop_value", guess.crop_col),
                              ("label_value", guess.label_col)):
            # "No id column" (an empty uuid_value) is a choice like any
            # other and is kept.
            if attr == "uuid_value" and not self.uuid_value:
                continue
            if guessed and getattr(self, attr) not in self.column_options:
                setattr(self, attr, guessed)
        # It is also the right guess for a readable file where no column
        # looks like an id, rather than a "uuid" the file does not have.
        if (self.uuid_value and self.uuid_value not in self.column_options
                and self.column_options and not guess.uuid_col):
            self.uuid_value = ""

        self.schema_note.value = guess.explain()
        self._rebuild_column_pickers()
        self._push_column_pickers()

    def _rebuild_column_pickers(self) -> None:
        """Fresh Dropdown controls every time.

        Never a mutated `options` list on a mounted Dropdown: that does not
        reliably reach the browser and breaks outright when it happens
        during a notify (see CompareView._rebuild_dropdowns, which learned
        it the hard way).
        """
        # The current values are always offered, even when the file could
        # not be read, so the control is never empty and a restored mapping
        # stays selectable before its file has been sniffed.
        names = list(dict.fromkeys(
            list(self.column_options) + [self.uuid_value, self.crop_value, self.label_value]))
        # Alphabetical, case-insensitive: a wide CSV is far easier to scan
        # that way than in the file's own column order.
        names = sorted((n for n in names if n), key=lambda n: (str(n).casefold(), str(n)))

        def picker(label: str, attr: str, tooltip: str, allow_none: bool = False) -> ft.Dropdown:
            def on_change(e, attr=attr):
                value = e.control.value
                setattr(self, attr, "" if value == NO_UUID_KEY else value)

            options = [ft.dropdown.Option(n, n) for n in names]
            if allow_none:
                options.insert(0, ft.dropdown.Option(NO_UUID_KEY, "None (no id)"))
            return ft.Dropdown(
                label=label, width=210, border_radius=8,
                value=getattr(self, attr) or (NO_UUID_KEY if allow_none else None),
                options=options, on_change=on_change, tooltip=tooltip,
            )

        self.columns_container.content = ft.Row([
            picker("uuid column", "uuid_value",
                   "The crop id. Guessed from the values: a uuid has a very specific shape. "
                   "None if the file has no id: the crops and their metadata can still be browsed, "
                   "but no experiment can be matched to them.", allow_none=True),
            picker("crop column", "crop_value",
                   "The path to the image. Guessed from the values: a path has a separator "
                   "or an image extension."),
            picker("label column", "label_value",
                   "The class. Guessed from the values: a short string with few distinct values."),
        ], spacing=12, wrap=True)

    def _push_column_pickers(self) -> None:
        for control in (self.columns_container, self.schema_note):
            if control.page:
                control.update()

    def _set_field(self, field: ft.TextField, path: str) -> None:
        field.value = path
        if field.page:
            field.update()

    def _on_add_from_mlflow_click(self, e):
        panel = getattr(self.ctx, "mlflow_panel", None)
        if panel is not None:
            panel.open_panel(e)

    def _on_add_experiment_click(self, e):
        self._add_experiment_row()
        self.root.update()

    # ---- dataset load ----
    def _on_load_dataset_click(self, e):
        self.load_dataset_now()

    def load_dataset_now(self) -> None:
        """The actual dataset-load logic, shared by the Load button's click
        handler and by app.py's --dataset startup auto-load."""
        if self._dataset_loading:
            return
        self._dataset_loading = True
        self.load_dataset_btn.disabled = True
        state = self.ctx.state

        self.dataset_progress.visible = True
        self.dataset_progress.value = 0
        self.dataset_status.content = _status_control("Loading...", "busy")
        self.dataset_progress.update()
        self.dataset_status.update()
        self.load_dataset_btn.update()
        self.ctx.busy.begin("Loading dataset")
        loaded = False

        def progress_cb(done: int, total: int):
            self.dataset_progress.value = done / total if total else 0
            self.dataset_status.content = _status_control(f"Loading... {done:,} / {total:,} rows", "busy")
            self.ctx.busy.detail_text(f"{done:,} / {total:,} rows")
            self.dataset_progress.update()
            self.dataset_status.update()

        try:
            columns = ColumnConfig(
                uuid_col=self.uuid_value,
                crop_col=self.crop_value or "crop",
                label_col=self.label_value or "label",
            )
            row_filter = None
            if self.row_filter_enabled.value and self.row_filter_col_field.value.strip():
                row_filter = (self.row_filter_col_field.value.strip(), self.row_filter_value_field.value)
            state.split_kind = self.split_dropdown.value or SPLIT_TEST
            state.load_dataset(
                self.dataset_path_field.value.strip(), columns, progress_cb=progress_cb,
                row_filter=row_filter, image_root=self.image_root_field.value.strip(),
            )
            # The progress bar above only covers the CSV read itself -- what's
            # left (dtype downcasting, then building the preview table below)
            # is normally fast, but says so explicitly rather than leaving the
            # tab looking frozen between "100%" and the final success message.
            self.dataset_status.content = _status_control("Finalizing (processing columns, building preview)...", "busy")
            self.ctx.busy.detail_text("finalizing")
            self.dataset_status.update()
            message = f"Loaded {len(state.dataset.df):,} rows, {len(state.dataset.metadata_columns())} metadata columns"
            if not columns.has_ids():
                message += " -- no uuid column, so no experiment can be loaded against it"
            self.dataset_status.content = _status_control(message, "success")
            print("[lookin] building dataset preview...", flush=True)
            self._build_dataset_extra()
            print("[lookin] dataset preview built", flush=True)
            loaded = True
            for row in self.experiment_rows:
                if row["loaded_name"] and row["loaded_name"] not in state.experiments:
                    # Unloaded by a dataset with no ids (see AppState.load_dataset).
                    row["loaded_name"] = None
                    row["overlap_container"].content = None
                    row["mapping_container"].controls.clear()
                    row["status_container"].content = _status_control(
                        "Unloaded: the dataset has no uuid column", "warning")
                elif row["loaded_name"]:
                    self._build_row_mapping(row)
            self.ctx.notify_change()
        except Exception as ex:
            print(f"[lookin] error loading dataset from {self.dataset_path_field.value.strip()!r}:")
            traceback.print_exc()
            self.dataset_status.content = _status_control(f"Error: {ex}", "error")
        finally:
            self.ctx.busy.end()
        self.dataset_progress.visible = False
        self._dataset_loading = False
        self.load_dataset_btn.disabled = False
        self.root.update()
        if state.dataset is not None and loaded:
            self._load_held_experiments()
            # A workspace opened without its dataset takes the one loaded
            # here (see WorkspacePanel.adopt_dataset).
            panel = getattr(self.ctx, "workspace_panel", None)
            if panel is not None:
                panel.adopt_dataset()

    def _on_split_change(self, e):
        """Changing the split kind after a load is legitimate (the user
        realizes they pointed at the train rows), so this applies live
        rather than only at load time."""
        self.ctx.state.split_kind = self.split_dropdown.value or SPLIT_TEST
        self.ctx.notify_change()

    def current_mount(self):
        """The loaded mount, in the shape core/library.py stores.

        Public because a library workspace stores the same thing (see
        core/library.py): "what is loaded right now" has one definition,
        and the Workspaces window must not grow a second one that drifts.
        """
        state = self.ctx.state
        if state.dataset is None or state.dataset.csv_path is None:
            return None
        row_filter = state.dataset.row_filter or ("", "")
        store = getattr(state, "review", None)
        # An experiment loaded from a workspace keeps the MLflow run it came
        # from. The loaded Experiment does not know it, so it is taken back
        # from the workspace -- matched on name *and* path, so an experiment
        # renamed or re-pointed since is not credited with a run it is no
        # longer a copy of.
        workspace = getattr(self.ctx, "current_workspace", None)
        origins = {(e.name, e.path): e for e in workspace.mount.experiments} if workspace is not None else {}
        experiments = []
        for name, exp in state.experiments.items():
            path = str(getattr(exp, "source_path", "") or self._path_for_experiment(name))
            loaded = library.MountExperiment(
                name=name,
                path=path,
                embeddings_path=str(exp.embeddings_path or ""),
                prototypes_path=str(exp.model_prototypes_path or ""),
            )
            origin = origins.get((name, path))
            if origin is not None:
                loaded = replace(origin, embeddings_path=loaded.embeddings_path,
                                 prototypes_path=loaded.prototypes_path)
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
        )

    def _path_for_experiment(self, name: str) -> str:
        """The CSV path shown in the row for a loaded experiment -- the
        Experiment object itself doesn't keep it."""
        for row in self.experiment_rows:
            if row["loaded_name"] == name:
                return row["path_field"].value.strip()
        return ""

    def restore_mount(self, mount) -> None:
        """Refill every field from a remembered mount and load it: the
        dataset first, then each experiment with the sidecars it had.

        Failures here are per-experiment and non-fatal -- a run whose
        results CSV has since moved shouldn't stop the dataset (and every
        other experiment) from coming back.
        """
        self.dataset_path_field.value = mount.dataset_path
        # A saved mapping is an explicit past choice: it wins over
        # anything the file's shape would suggest.
        self.uuid_value = mount.uuid_col
        self.crop_value = mount.crop_col
        self.label_value = mount.label_col
        self._sniff_dataset()
        self.image_root_field.value = mount.image_root
        self.row_filter_enabled.value = bool(mount.row_filter_col)
        if mount.row_filter_col:
            self.row_filter_col_field.value = mount.row_filter_col
            self.row_filter_value_field.value = mount.row_filter_value
        self.split_dropdown.value = mount.split_kind if mount.split_kind in SPLIT_KINDS else SPLIT_TEST
        for row in list(self.experiment_rows):
            self._remove_row(row)
        if self.root.page:
            self.root.update()

        # One outer task around the whole restore: it is a dataset load
        # followed by an experiment load per row, each of which lights the
        # app-bar indicator on its own -- without this the light would blink
        # off between them and read as "finished" halfway through.
        self.ctx.busy.begin(f"Restoring {mount.label()}")
        try:
            self._restore_loads(mount)
        finally:
            self.ctx.busy.end()

    def _restore_loads(self, mount) -> None:
        """The loading half of restore_mount, once the fields are filled."""
        if mount.dataset_path:
            self.load_dataset_now()
        else:
            # Nothing to load yet: the dataset is set up here, in the Data
            # tab. Whatever this session showed before must not pass for
            # this workspace's data.
            self.ctx.state.clear_all()
            self.dataset_extra_column.controls.clear()
            self.dataset_status.content = _status_control(
                "This workspace has no dataset yet -- pick its CSV above and click Load dataset", "neutral")
            self.ctx.notify_change()
        if self.ctx.state.dataset is None:
            self._hold_experiments(mount.experiments)
            if self.root.page:
                self.root.update()
            return
        store = getattr(self.ctx.state, "review", None)
        if store is not None and mount.review_version and mount.review_version in store.versions:
            store.active_version_id = mount.review_version
        self._load_mount_experiments(mount.experiments)
        self.ctx.notify_change()
        self.root.update()

    def _hold_experiments(self, experiments: list) -> None:
        """Show a workspace's experiments as rows that wait for a dataset.

        An experiment is scored against the dataset, so none can load before
        one does -- but dropping them would leave a workspace opened without
        a dataset looking empty, and the next save would write it back
        without its runs.
        """
        self._held_experiments = [e for e in experiments if e.path]
        for experiment in self._held_experiments:
            self._add_experiment_row(experiment.name, experiment.path)
            row = self.experiment_rows[-1]
            row["held"] = True
            row["embed_field"].value = experiment.embeddings_path
            row["proto_field"].value = experiment.prototypes_path
            row["status_container"].content = _status_control("Loads once the dataset is loaded", "neutral")

    def _load_held_experiments(self) -> None:
        held, self._held_experiments = self._held_experiments, []
        if not held:
            return
        for row in [r for r in self.experiment_rows if r.get("held")]:
            self._remove_row(row)
        self._load_mount_experiments(held)
        self.ctx.notify_change()
        self.root.update()

    def _load_mount_experiments(self, experiments: list) -> None:
        for experiment in experiments:
            if not experiment.path:
                continue
            success, message = self.add_experiment_from_path(experiment.path, experiment.name)
            if not success:
                print(f"[lookin] library: could not restore experiment {experiment.name!r} "
                      f"from {experiment.path}: {message}", flush=True)
                continue
            for path, loader, kind in (
                (experiment.embeddings_path, self.ctx.prototype_source.load_embeddings_for, "embeddings"),
                (experiment.prototypes_path, self.ctx.prototype_source.load_prototypes_for, "prototypes"),
            ):
                if not path or not Path(path).is_file():
                    continue
                ok, note = loader(experiment.name, path)
                if not ok:
                    print(f"[lookin] library: could not restore {kind} for {experiment.name!r}: {note}", flush=True)

    def _on_clear_all_click(self, e):
        self.ctx.state.clear_all()
        self._held_experiments = []
        self.dataset_status.content = _status_control("Not loaded", "neutral")
        self.dataset_extra_column.controls.clear()
        for row in list(self.experiment_rows):
            self._remove_row(row)
        for row in list(self.inference_rows):
            self._remove_inference_row(row)
        self.ctx.notify_change()
        self.root.update()

    def _build_dataset_extra(self):
        self.dataset_extra_column.controls.clear()
        state = self.ctx.state
        if state.dataset is None:
            return

        df = state.dataset.df
        preview_df = df.head(8)
        columns = [ft.DataColumn(ft.Text(c, weight=ft.FontWeight.BOLD)) for c in preview_df.columns]
        rows = []
        for _, r in preview_df.iterrows():
            cells = [ft.DataCell(ft.Text(str(v)[:40])) for v in r.values]
            rows.append(ft.DataRow(cells=cells))
        table = ft.DataTable(columns=columns, rows=rows, border_radius=8, heading_row_color=ft.Colors.SURFACE_CONTAINER_HIGHEST)

        kinds = state.dataset.column_kinds()
        kinds_text = (
            f"Numeric: {', '.join(kinds.numeric) or '—'}\n"
            f"Binary: {', '.join(kinds.binary) or '—'}\n"
            f"Categorical: {', '.join(kinds.categorical) or '—'}"
        )
        if kinds.high_cardinality:
            kinds_text += (
                f"\nSkipped (too many unique values -- not shown in Statistics/Gallery filters): "
                f"{', '.join(kinds.high_cardinality)}"
            )

        preview_card = card(ft.Column([
            section_title("Dataset preview", icon=ft.Icons.TABLE_ROWS_OUTLINED),
            ft.Text("First 8 rows", size=12, italic=True),
            ft.Row(controls=[table], scroll=ft.ScrollMode.ALWAYS),
            info_banner(kinds_text),
        ], spacing=10))
        self.dataset_extra_column.controls.append(preview_card)
        self._build_remap_card()

    def _build_remap_card(self):
        state = self.ctx.state
        labels = state.dataset.labels()
        self._remap_fields = []
        fields_row = ft.Row(wrap=True, spacing=8)

        def field_for(label, labelled: bool = False) -> ft.TextField:
            # Labelled with the original name when the fields come and go
            # with a search, since the field alone no longer says which
            # class it renames once its text is edited.
            return ft.TextField(value=str(label), label=str(label) if labelled else None,
                                width=160, border_radius=8, dense=True)

        if len(labels) > MAPPING_MAX_ALL:
            # One field per class is two thousand fields; search for the
            # classes to rename instead, and get a field for each of those.
            def on_picked(picked: list) -> None:
                kept = {orig: field for orig, field in self._remap_fields}
                self._remap_fields = [(l, kept.get(l) or field_for(l, labelled=True)) for l in picked]
                fields_row.controls = [field for _, field in self._remap_fields]
                if fields_row.page:
                    fields_row.update()

            picker = ClassPicker(on_picked, limit=None, label="Find a class to rename")
            picker.set_labels(labels)
            chooser = [picker.build()]
        else:
            self._remap_fields = [(l, field_for(l)) for l in labels]
            fields_row.controls = [field for _, field in self._remap_fields]
            chooser = []
        fields = [*chooser, fields_row]
        remap_card = card(ft.Column([
            section_title("Relabel classes", icon=ft.Icons.EDIT_OUTLINED),
            ft.Text(
                "Rename raw label values to friendlier names if you'd like. Applies to the dataset "
                "and cascades into every loaded experiment's class mapping automatically.",
                size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            *fields,
            ft.TextButton("Apply relabeling", icon=ft.Icons.CHECK, on_click=self._on_apply_remap),
        ], spacing=10))
        self.dataset_extra_column.controls.append(remap_card)

    def _on_apply_remap(self, e):
        mapping = {
            orig: field.value.strip()
            for orig, field in self._remap_fields
            if field.value.strip() and field.value.strip() != orig
        }
        if mapping:
            self.ctx.state.apply_label_remap(mapping)
            self._build_dataset_extra()
            for row in self.experiment_rows:
                if row["loaded_name"]:
                    self._build_row_mapping(row)
            self.ctx.notify_change()
        self.root.update()

    def refresh(self):
        if self.root.page:
            self.root.update()

    def build(self) -> ft.Control:
        # Guess the mapping before the first render, so the pickers arrive
        # already filled in rather than empty for a beat.
        self._sniff_dataset()
        dataset_card = card(ft.Column([
            section_title("1. Load the dataset CSV", icon=ft.Icons.IMAGE_OUTLINED, ctx=self.ctx, manual="dataset-join"),
            ft.Text(
                "One row per (cropped) image: an id, a path to the crop, a label, and any number of "
                "metadata columns. The three roles below are guessed from the file's own values; "
                "each is a dropdown of this file's columns, so a wrong guess is one click to fix.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            ft.Row(controls=[self.dataset_path_field, ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=self._on_dataset_browse)]),
            self.columns_container,
            self.schema_note,
            ft.Row(controls=[self.image_root_field, ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse", on_click=self._on_image_root_browse)]),
            ft.Row(controls=[
                self.row_filter_enabled, self.row_filter_col_field, self.row_filter_equals_text, self.row_filter_value_field,
                ft.Container(width=12), self.split_dropdown,
            ], spacing=8, wrap=True),
            ft.Row(controls=[
                self.load_dataset_btn,
                self.clear_all_btn,
                self.dataset_status,
            ], spacing=16),
            self.dataset_progress,
        ], spacing=10))

        experiments_card = card(ft.Column([
            section_title("2. Load one or more results CSVs (experiments)", icon=ft.Icons.SCIENCE_OUTLINED, ctx=self.ctx, manual="score-diagnostics"),
            ft.Text(
                "Optional -- an id column plus one score column per class. Load as many as you like; "
                "they are compared side by side in the Metrics tab. A loaded experiment's name can "
                "be edited at any time: it takes effect everywhere, and nothing is recomputed.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self.experiments_column,
            ft.Row([
                ft.TextButton("Add experiment", icon=ft.Icons.ADD, on_click=self._on_add_experiment_click),
                # One CSV straight from a run into this session, without a
                # workspace -- the Home page is where runs are organised.
                ft.TextButton("Add from MLflow...", icon=ft.Icons.CLOUD_DOWNLOAD_OUTLINED,
                              on_click=self._on_add_from_mlflow_click),
            ], spacing=4),
        ], spacing=10))

        inferences_card = card(ft.Column([
            section_title("3. Load simple inferences (optional)", icon=ft.Icons.LAYERS_OUTLINED,
                          ctx=self.ctx, manual="dataset-join"),
            ft.Text(
                "Crops that are not in the dataset, run through the same model: a CSV with an id "
                "and a path, plus an embeddings .npz from that same model ('uuid' + 'embeddings' "
                "arrays). They carry no label, so they never enter a metric -- they are drawn on "
                "the Embeddings tab's map, as squares.",
                size=12, color=ft.Colors.ON_SURFACE_VARIANT,
            ),
            self.inferences_column,
            ft.TextButton("Add simple inference", icon=ft.Icons.ADD, on_click=self._on_add_inference_click),
        ], spacing=10))

        self.root.content = ft.Column(
            controls=[dataset_card, experiments_card, inferences_card,
                      self.dataset_extra_column],
            scroll=ft.ScrollMode.AUTO,
            spacing=16,
        )
        self._build_dataset_extra()
        return self.root
