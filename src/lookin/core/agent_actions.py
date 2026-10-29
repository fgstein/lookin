"""What an agent can change in Lookin: workspaces, their experiments and
snapshots, the Home groups, the dataset filter, review decisions.

The writing half of the MCP endpoint (see lookin/mcp_server.py; the reading
half is core/agent_tools.py). Every function does what the matching button
does, through the same core call, so a workspace an agent made is one the
interface opens like any other -- and a tab where that workspace is open
picks the change up within seconds, as it does for an edit made on Home.

Three rules:

- **Read-change-write on disk.** A workspace is changed with
  library.update_workspace, which re-reads the manifest under its lock: a
  copy held in memory may be older than what a colleague just saved.
- **Refuse rather than guess.** An unknown column, a dataset that is not a
  file, a run evaluated on another dataset: each is an error in words the
  agent can act on, never a half-made workspace.
- **Nothing is deleted by accident.** Deleting a workspace or a snapshot
  asks for its id a second time (`confirm`), and the MCP endpoint marks
  those tools destructive so a client can ask its user first.
"""
from __future__ import annotations

from pathlib import Path

from lookin.core import agent_tools as tools
from lookin.core import board as bd
from lookin.core import dataset_filter as dsf
from lookin.core import library as lib
from lookin.core import links
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.core.data_manager import SPLIT_KINDS, sidecar_paths_for_csv
from lookin.core.review import (
    NO_PARENT, REVIEW_REASONS, REVIEW_STATUSES, STATUS_RELABEL, STATUS_UNREVIEWED, export_enriched_dataset,
    export_excluded_uuids,
)
from lookin.core.review_actions import record_decision
from lookin.core.workspace_state import find_workspace

MAX_DECISIONS = 50_000


def _workspace_out(workspace: lib.Workspace, **extra) -> dict:
    out = {"id": workspace.id, "name": workspace.name, "open_url": links.link(workspace.id)}
    out.update({k: v for k, v in extra.items() if v is not None})
    return tools._clean(out)


def _update(workspace: lib.Workspace, change) -> lib.Workspace:
    """library.update_workspace, as an error when the workspace is gone or
    its manifest cannot be written."""
    fresh = lib.update_workspace(workspace.path, change)
    if fresh is None:
        raise ValueError(f"Workspace {workspace.id} could not be updated (gone, or its manifest is not writable).")
    return fresh


def _known_ids() -> list:
    return [w.id for w in lib.list_workspaces()]


def _group_column(group: str, create: bool = True) -> dict:
    """The Home group named `group` (case-insensitive), made when missing."""
    name = (group or "").strip()
    board = bd.load()
    for column in board["columns"]:
        if column["name"].strip().lower() == name.lower():
            return column
    if not create:
        raise ValueError(f"No group named {group!r}. Groups: {', '.join(c['name'] for c in board['columns'])}.")
    board = bd.add_column(name)
    return board["columns"][-1]


# ---- the dataset's columns -------------------------------------------------

def _resolve_roles(path: str, uuid_col=None, crop_col=None, label_col=None, image_root=None) -> tuple:
    """(roles, notes) for a dataset CSV: what was given, else what a
    lookin.toml next to it says, else what its values look like -- the
    Sources tab's order. A role naming a column the file does not have is
    refused."""
    from lookin.core.dataset_config import config_for_csv
    from lookin.core.schema import sniff_csv

    guess = sniff_csv(path)
    if guess.error:
        raise ValueError(f"Could not read {path}: {guess.error}")
    config = config_for_csv(path)
    notes = []
    roles = {}
    for role, given, from_config, guessed, default in (
        ("uuid_col", uuid_col, getattr(config, "id", None), guess.uuid_col, ""),
        ("crop_col", crop_col, getattr(config, "image", None), guess.crop_col, "crop"),
        ("label_col", label_col, getattr(config, "label", None), guess.label_col, "label"),
    ):
        if given is not None:
            value, how = given, "given"
        elif from_config is not None and (not from_config or from_config in guess.columns):
            value, how = from_config, "lookin.toml"
        elif guessed:
            value, how = guessed, "guessed from the values"
        else:
            value, how = default, "default"
        if value and value not in guess.columns:
            raise ValueError(f"{path} has no column {value!r} (for {role}). Columns: {', '.join(guess.columns)}.")
        roles[role] = value
        notes.append(f"{role}={value or '(none)'} ({how})")
    roles["image_root"] = image_root if image_root is not None else (getattr(config, "root", None) or "")
    return roles, notes


def _check_split(split_kind) -> None:
    if split_kind is not None and split_kind not in SPLIT_KINDS:
        raise ValueError(f"split_kind must be one of {', '.join(SPLIT_KINDS)}; got {split_kind!r}.")


# ---- workspaces ------------------------------------------------------------

def create_workspace(name: str, dataset_path: str = "", group: str = "", description: str = "",
                     tags: list | None = None, uuid_col: str | None = None, crop_col: str | None = None,
                     label_col: str | None = None, image_root: str | None = None,
                     row_filter_col: str = "", row_filter_value: str = "", split_kind: str = "test") -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("A workspace needs a name.")
    _check_split(split_kind)
    mount = lib.Mount(split_kind=split_kind, row_filter_col=row_filter_col or "",
                      row_filter_value=row_filter_value or "")
    notes = []
    if dataset_path:
        path = str(Path(dataset_path).expanduser())
        if not Path(path).is_file():
            raise ValueError(f"No dataset CSV at {path} on the machine running Lookin.")
        roles, notes = _resolve_roles(path, uuid_col, crop_col, label_col, image_root)
        mount.dataset_path = path
        mount.uuid_col, mount.crop_col, mount.label_col = roles["uuid_col"], roles["crop_col"], roles["label_col"]
        mount.image_root = roles["image_root"]
    workspace = lib.create_workspace(name, mount, description=description, tags=tags or [],
                                     provenance={"created_by": "mcp"}, hash_dataset=bool(dataset_path))
    if workspace is None:
        raise ValueError(f"The workspace could not be written in the library at {lib.library_root()}.")
    if group:
        bd.move(workspace.id, _group_column(group)["id"], known_ids=_known_ids())
    return _workspace_out(workspace, group=group or None, columns=notes)


def update_workspace(workspace_id: str, name: str | None = None, description: str | None = None,
                     tags: list | None = None, group: str | None = None, dataset_path: str | None = None,
                     uuid_col: str | None = None, crop_col: str | None = None, label_col: str | None = None,
                     image_root: str | None = None, row_filter_col: str | None = None,
                     row_filter_value: str | None = None, split_kind: str | None = None,
                     review_version: str | None = None) -> dict:
    """Change what is given, keep the rest -- the id, the snapshots and
    every experiment stay."""
    workspace = find_workspace(workspace_id)
    _check_split(split_kind)
    notes = []
    roles = None
    new_dataset = dataset_path is not None and dataset_path != workspace.mount.dataset_path
    path = str(Path(dataset_path).expanduser()) if dataset_path else (workspace.mount.dataset_path or "")
    if new_dataset and dataset_path and not Path(path).is_file():
        raise ValueError(f"No dataset CSV at {path} on the machine running Lookin.")
    if path and (new_dataset or any(v is not None for v in (uuid_col, crop_col, label_col))):
        current = workspace.mount
        roles, notes = _resolve_roles(
            path,
            uuid_col if uuid_col is not None else (None if new_dataset else current.uuid_col),
            crop_col if crop_col is not None else (None if new_dataset else current.crop_col),
            label_col if label_col is not None else (None if new_dataset else current.label_col),
            image_root)

    def change(fresh: lib.Workspace) -> bool:
        m = fresh.mount
        if name is not None and name.strip():
            fresh.name = name.strip()
        if description is not None:
            fresh.description = description
        if tags is not None:
            fresh.tags = [str(t) for t in tags]
        if dataset_path is not None:
            m.dataset_path = path if dataset_path else ""
            m.dataset_size, m.dataset_mtime = 0, 0.0
            if new_dataset:
                m.dataset_filter = {}
        if roles is not None:
            m.uuid_col, m.crop_col, m.label_col = roles["uuid_col"], roles["crop_col"], roles["label_col"]
            m.image_root = roles["image_root"]
        elif image_root is not None:
            m.image_root = image_root
        if row_filter_col is not None:
            m.row_filter_col = row_filter_col
        if row_filter_value is not None:
            m.row_filter_value = row_filter_value
        if split_kind is not None:
            m.split_kind = split_kind
        if review_version is not None:
            m.review_version = review_version
        return True

    fresh = _update(workspace, change)
    if group is not None and group.strip():
        bd.move(fresh.id, _group_column(group)["id"], known_ids=_known_ids())
    return _workspace_out(fresh, columns=notes or None, group=group or None)


def duplicate_workspace(workspace_id: str, name: str = "") -> dict:
    workspace = find_workspace(workspace_id)
    copy = lib.duplicate_workspace(workspace, (name or "").strip())
    if copy is None:
        raise ValueError(f"Workspace {workspace.id} could not be copied.")
    bd.place_after(copy.id, workspace.id, known_ids=_known_ids())
    return _workspace_out(copy, copied_from=workspace.id)


def delete_workspace(workspace_id: str, confirm: str) -> dict:
    workspace = find_workspace(workspace_id)
    if confirm != workspace.id:
        raise ValueError(f"To delete workspace {workspace.label()!r}, pass its id {workspace.id!r} as confirm. "
                         f"This removes it and its {len(lib.list_snapshots(workspace))} snapshot(s); the dataset "
                         f"and results files are not touched.")
    if not lib.delete_workspace(workspace):
        raise ValueError(f"Workspace {workspace.id} could not be deleted.")
    return {"deleted": workspace.id, "name": workspace.name}


def rename_group(group: str, new_name: str) -> dict:
    column = _group_column(group, create=False)
    if not new_name.strip():
        raise ValueError("A group needs a name.")
    bd.rename_column(column["id"], new_name.strip())
    return {"group": new_name.strip()}


# ---- experiments -------------------------------------------------------------

def add_experiment(workspace_id: str, results_csv: str, name: str = "", embeddings: str = "",
                   prototypes: str = "") -> dict:
    """A results CSV on the machine running Lookin, with its sidecars --
    found next to it by name when not given."""
    workspace = find_workspace(workspace_id)
    path = Path(results_csv).expanduser()
    if not path.is_file():
        raise ValueError(f"No results CSV at {path} on the machine running Lookin.")
    embed_guess, proto_guess = sidecar_paths_for_csv(path)
    embeddings = embeddings or (str(embed_guess) if Path(embed_guess).is_file() else "")
    prototypes = prototypes or (str(proto_guess) if Path(proto_guess).is_file() else "")
    for label, sidecar in (("embeddings", embeddings), ("prototypes", prototypes)):
        if sidecar and not Path(sidecar).expanduser().is_file():
            raise ValueError(f"No {label} file at {sidecar}.")
    base = (name or "").strip() or path.stem.removesuffix("_results")
    added = {}

    def change(fresh: lib.Workspace) -> bool:
        final = wss._unique_name(fresh.mount, base)
        fresh.mount.experiments.append(lib.MountExperiment(
            name=final, path=str(path), embeddings_path=str(Path(embeddings).expanduser()) if embeddings else "",
            prototypes_path=str(Path(prototypes).expanduser()) if prototypes else ""))
        added["name"] = final
        return True

    fresh = _update(workspace, change)
    return _workspace_out(fresh, added=added["name"], embeddings=embeddings or None, prototypes=prototypes or None)


def add_mlflow_run(workspace_id: str, run_id: str, server: str | None = None, csv_path: str = "",
                   name: str = "", download: bool = True) -> dict:
    """A run's results CSV (and its sidecars), recorded as a link to the run
    -- what dropping a run on a workspace on Home does -- then fetched."""
    workspace = find_workspace(workspace_id)
    uri = tools._server_for_run(server)
    csvs, npzs = mlf.list_csv_and_npz_artifacts(uri, run_id)
    if not csvs:
        raise ValueError(f"Run {run_id} logged no CSV on {uri}.")
    if csv_path:
        csvs = [a for a in csvs if a.path == csv_path]
        if not csvs:
            raise ValueError(f"Run {run_id} has no CSV {csv_path!r}; call list_mlflow_run_files to see its CSVs.")
    elif len(csvs) > 1:
        raise ValueError(f"Run {run_id} logged {len(csvs)} CSVs -- pick one with csv_path: "
                         + ", ".join(a.path for a in csvs))
    run = mlf.get_run(uri, run_id)
    candidates = wss.candidates_for_run(uri, run, csvs, npzs)
    result = {}

    def change(fresh: lib.Workspace) -> bool:
        res = wss.add_candidates(fresh, candidates)
        if res.added and name.strip():
            for e in fresh.mount.experiments:
                if e.name == res.added[0]:
                    e.name = wss._unique_name(fresh.mount, name.strip()) if name.strip() != e.name else e.name
                    res.added[0] = e.name
        result["res"] = res
        return bool(res.added or res.dataset_set)

    fresh = _update(workspace, change)
    res = result["res"]
    if not res.added:
        raise ValueError(res.message())
    out = {"added": res.added, "message": res.message()}
    if download:
        out["download"] = download_workspace(fresh.id)
    return _workspace_out(find_workspace(fresh.id), **out)


def remove_experiment(workspace_id: str, name: str) -> dict:
    workspace = find_workspace(workspace_id)
    if name not in [e.name for e in workspace.mount.experiments]:
        raise ValueError(f"No experiment {name!r} in this workspace. Experiments: "
                         f"{', '.join(e.name for e in workspace.mount.experiments) or 'none'}.")
    fresh = _update(workspace, lambda w: wss.remove_experiment(w, name))
    return _workspace_out(fresh, removed=name)


def rename_experiment(workspace_id: str, name: str, new_name: str) -> dict:
    workspace = find_workspace(workspace_id)
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("An experiment needs a name.")
    names = [e.name for e in workspace.mount.experiments]
    if name not in names:
        raise ValueError(f"No experiment {name!r} in this workspace. Experiments: {', '.join(names) or 'none'}.")
    if new_name in names and new_name != name:
        raise ValueError(f"This workspace already has an experiment named {new_name!r}.")

    def change(fresh: lib.Workspace) -> bool:
        for e in fresh.mount.experiments:
            if e.name == name:
                e.name = new_name
                return True
        return False

    return _workspace_out(_update(workspace, change), renamed={name: new_name})


def set_experiment_color(workspace_id: str, name: str, color: str = "") -> dict:
    """"#rrggbb", or "" for its palette colour -- every chart follows."""
    import re

    if color and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError(f"color must be '#rrggbb' or empty; got {color!r}.")
    workspace = find_workspace(workspace_id)

    def change(fresh: lib.Workspace) -> bool:
        for e in fresh.mount.experiments:
            if e.name == name:
                e.color = color.lower()
                return True
        raise ValueError(f"No experiment {name!r} in this workspace.")

    return _workspace_out(_update(workspace, change), experiment=name, color=color or None)


def download_workspace(workspace_id: str) -> dict:
    """Fetch every MLflow file the workspace needs into this machine's cache."""
    workspace = find_workspace(workspace_id)
    fetched, errors = wss.download(workspace)
    # The download worked on a copy; what it fetched is recorded on the
    # manifest as it is now, as Home's download button does.
    lib.update_workspace(workspace.path, wss.point_at_cache)
    fresh = find_workspace(workspace.id)
    out = {"fetched": fetched, "download_state": wss.download_state(fresh)}
    if errors:
        out["errors"] = errors
    return out


# ---- the dataset filter --------------------------------------------------------

def set_dataset_filter(workspace_id: str, query: str = "", conditions: list | None = None) -> dict:
    """The Filter tab's filter: rows kept by `conditions` (each {column, op,
    values | min/max | text}) and by `query`. Both empty clears it. Checked
    against the dataset before it is saved."""
    workspace = find_workspace(workspace_id)
    spec = dsf.normalize_spec({dsf.CONDITIONS: conditions or [], dsf.QUERY: query or "", dsf.QUERY_ON: True})
    if conditions and len(spec[dsf.CONDITIONS]) != len(conditions):
        raise ValueError(f"Some conditions could not be read. Each is {{column, op, ...}} with op one of "
                         f"{', '.join(dsf.OPS)}.")
    kept = total = None
    if dsf.is_active(spec):
        if not workspace.mount.exists():
            raise ValueError("This workspace's dataset is not on this machine, so the filter cannot be checked.")
        with tools._lock:
            entry = tools._entry(workspace.id)
            dataset = entry.loaded.state.dataset
            mask, steps = dsf.apply_spec(dataset.full_df, spec)
            kept, total = int(mask.sum()), len(dataset.full_df)
        if kept == 0:
            raise ValueError("This filter keeps no row at all -- not saved.")

    def change(fresh: lib.Workspace) -> bool:
        fresh.mount.dataset_filter = spec if dsf.is_active(spec) else {}
        return True

    fresh = _update(workspace, change)
    return _workspace_out(fresh, filter=dsf.describe(spec), rows_kept=kept, rows_total=total)


# ---- snapshots -------------------------------------------------------------------

def create_snapshot(workspace_id: str, name: str, description: str = "", tags: list | None = None,
                    tab: str = "", from_snapshot: str = "", view_state: dict | None = None) -> dict:
    """A saved view: a copy of another snapshot's state (`from_snapshot`),
    or `view_state` as get_snapshot shows it, or just the workspace on one
    tab. `tab` overrides the tab either way."""
    from lookin.core import view_state as vs

    workspace = find_workspace(workspace_id)
    if from_snapshot:
        found = lib.find_snapshot(from_snapshot)
        if found is None:
            raise ValueError(f"No snapshot {from_snapshot!r}.")
        state = dict(found[1].view_state)
    elif view_state is not None:
        if not isinstance(view_state, dict):
            raise ValueError("view_state must be an object, as get_snapshot returns it.")
        state = dict(view_state)
    else:
        state = {vs.APP: {}, vs.VIEWS: {}}
    if tab:
        if tab not in vs.SNAPSHOT_TABS:
            raise ValueError(f"tab must be one of {', '.join(vs.SNAPSHOT_TABS)}; got {tab!r}.")
        state[vs.APP] = {**(state.get(vs.APP) or {}), vs.TAB: vs.SNAPSHOT_TABS.index(tab)}
    snapshot = lib.create_snapshot(workspace, state, name=(name or "").strip(), description=description,
                                   tags=tags or [])
    if snapshot is None:
        raise ValueError("The snapshot could not be written.")
    return {"id": snapshot.id, "name": snapshot.name, "workspace_id": workspace.id,
            "open_url": links.link(snapshot_id=snapshot.id)}


def update_snapshot(snapshot_id: str, name: str | None = None, description: str | None = None,
                    tags: list | None = None) -> dict:
    """Name, description, tags -- the id and the view it restores stay."""
    found = lib.find_snapshot(snapshot_id)
    if found is None:
        raise ValueError(f"No snapshot {snapshot_id!r}.")
    workspace, snapshot = found
    if name is not None:
        snapshot.name = name.strip()
    if description is not None:
        snapshot.description = description
    if tags is not None:
        snapshot.tags = [str(t) for t in tags]
    if not lib.save_snapshot(workspace, snapshot):
        raise ValueError("The snapshot could not be written.")
    return {"id": snapshot.id, "name": snapshot.name, "open_url": links.link(snapshot_id=snapshot.id)}


def delete_snapshot(snapshot_id: str, confirm: str) -> dict:
    found = lib.find_snapshot(snapshot_id)
    if found is None:
        raise ValueError(f"No snapshot {snapshot_id!r}.")
    workspace, snapshot = found
    if confirm != snapshot.id:
        raise ValueError(f"To delete snapshot {snapshot.label()!r}, pass its id {snapshot.id!r} as confirm.")
    if not lib.delete_snapshot(workspace, snapshot.id):
        raise ValueError("The snapshot could not be deleted.")
    return {"deleted": snapshot.id}


# ---- moving a workspace between machines -------------------------------------------

def export_workspace(workspace_id: str, dest: str = "", include_embeddings: bool = False) -> dict:
    """A self-contained .zip of the workspace -- dataset, results, snapshots
    -- that import_workspace (or the Workspaces window) reads back."""
    from lookin.core import agent_outputs as outputs

    workspace = find_workspace(workspace_id)
    path = outputs.destination(dest, f"{lib.slugify(workspace.name)}-{workspace.id}.zip")
    ok, message = lib.export_workspace(workspace, path, include_embeddings=include_embeddings)
    if not ok:
        raise ValueError(message)
    return outputs.file_answer(path, message=message)


def import_workspace(zip_path: str, group: str = "") -> dict:
    path = Path(zip_path).expanduser()
    if not path.is_file():
        raise ValueError(f"No file at {path} on the machine running Lookin.")
    workspace, notes = lib.import_workspace(path)
    if workspace is None:
        raise ValueError("; ".join(notes) or "The archive could not be imported.")
    if group:
        bd.move(workspace.id, _group_column(group)["id"], known_ids=_known_ids())
    return _workspace_out(workspace, notes=list(notes))


# ---- review ----------------------------------------------------------------------

class _ReviewTarget:
    """What record_decision needs of a session: the state holding the
    store, and someone to tell. Nobody to tell here -- a tab with the same
    dataset open reads the store from disk on its next save."""

    def __init__(self, state):
        self.state = state

    def notify_review(self) -> None:
        pass


def _store(entry):
    store = entry.loaded.state.review
    if store is None:
        raise ValueError("This workspace has no dataset loaded, so no review store.")
    return store


def record_review_decisions(workspace_id: str, uuids: list, status: str, new_label: str = "", reason: str = "",
                            note: str = "") -> dict:
    """Keep, relabel, exclude or mark uncertain -- written to the review
    store next to the dataset CSV, in its active version, as the carousel's
    decision bar does. The dataset CSV itself is never touched."""
    if status not in REVIEW_STATUSES or status == STATUS_UNREVIEWED:
        raise ValueError(f"status must be one of {', '.join(s for s in REVIEW_STATUSES if s != STATUS_UNREVIEWED)}; "
                         f"got {status!r}.")
    if reason and reason not in REVIEW_REASONS:
        raise ValueError(f"reason must be one of {', '.join(REVIEW_REASONS)}, or empty; got {reason!r}.")
    if status == STATUS_RELABEL and not new_label:
        raise ValueError("A relabel needs new_label.")
    uuids = [str(u) for u in (uuids or [])]
    if not uuids:
        raise ValueError("No uuid given.")
    if len(uuids) > MAX_DECISIONS:
        raise ValueError(f"At most {MAX_DECISIONS:,} crops per call.")
    with tools._lock:
        entry = tools._entry(workspace_id)
        state = entry.loaded.state
        store = _store(entry)
        known = set(state.dataset.full_df[state.dataset.columns.uuid_col].astype(str))
        unknown = [u for u in uuids if u not in known]
        if unknown:
            raise ValueError(f"{len(unknown)} uuid(s) are not in this dataset, e.g. {unknown[:3]}.")
        store.ensure_active_version()
        ok, message = record_decision(_ReviewTarget(state), uuids, status, new_label=new_label, reason=reason,
                                      note=note, source="mcp")
        if not ok:
            raise ValueError(message)
        return {"message": message, "version": store.active_version_id, "counts": store.counts()}


def undo_review_batch(workspace_id: str, batch_id: str) -> dict:
    with tools._lock:
        entry = tools._entry(workspace_id)
        store = _store(entry)
        n = store.undo_batch(batch_id)
        if not n:
            raise ValueError(f"No batch {batch_id!r} in the active version.")
        store.save()
        return {"undone": n, "version": store.active_version_id}


def create_review_version(workspace_id: str, name: str = "", parent: str = "", note: str = "",
                          independent: bool = False) -> dict:
    """A new review version, made active. It inherits everything its parent
    decided (the active version by default); `independent` starts empty --
    a second reviewer, a blind pass."""
    with tools._lock:
        entry = tools._entry(workspace_id)
        store = _store(entry)
        version = store.create_version(name or None, parent=NO_PARENT if independent else (parent or None),
                                       note=note)
        store.save()
        return {"version": version.version_id, "parent": version.parent or None}


def export_reviewed_dataset(workspace_id: str, versions: list | None = None, dest: str = "",
                            drop_excluded: bool = False, include_reason: bool = True,
                            excluded_uuids: bool = False) -> dict:
    """The dataset CSV written out again, every row and column, plus each
    version's label/status/reason columns -- what a training run reads.
    The versions exported are frozen, so the run stays identifiable."""
    from lookin.core import agent_outputs as outputs

    with tools._lock:
        entry = tools._entry(workspace_id)
        store = _store(entry)
        dataset = entry.loaded.state.dataset
        chosen = list(versions or ([store.active_version_id] if store.active_version_id else []))
        if not chosen:
            raise ValueError("No review version to export.")
        missing = [v for v in chosen if v not in store.versions]
        if missing:
            raise ValueError(f"No review version {missing[0]!r}. Versions: {', '.join(store.versions)}.")
        default = f"{Path(dataset.csv_path).stem}_reviewed_{chosen[-1]}.csv"
        path = outputs.destination(dest, default, beside=Path(dataset.csv_path).parent)
        result = export_enriched_dataset(store, dataset.csv_path, dataset.columns.uuid_col,
                                         dataset.columns.label_col, chosen, path, include_reason=include_reason,
                                         drop_excluded=drop_excluded)
        out = outputs.file_answer(Path(result.path), versions=chosen, rows=result.rows, excluded=result.excluded,
                                  relabelled=result.relabelled, columns=result.columns, notes=list(result.notes))
        if excluded_uuids:
            extra = export_excluded_uuids(store, chosen[-1], path.with_name(path.stem + "_excluded.txt"))
            out["excluded_uuids"] = str(extra.path)
        return out


# ---- notes -------------------------------------------------------------------

NOTES_MODES = ("replace", "append")


def write_notes(workspace_id: str, text: str, mode: str = "append") -> dict:
    """The Notes tab's text, written as the editor saves it."""
    from lookin.core import notes as nt

    if mode not in NOTES_MODES:
        raise ValueError(f"mode must be one of {', '.join(NOTES_MODES)}; got {mode!r}.")
    workspace = find_workspace(workspace_id)
    current = nt.read(workspace)
    if mode == "append" and current:
        text = current.rstrip() + "\n\n" + (text or "").strip() + "\n"
    if not (text or "").strip() and mode == "replace" and current:
        raise ValueError("Refusing to replace the notes with nothing. Write some text, or append instead.")
    path = nt.write(workspace, text or "")
    return tools._clean({"workspace_id": workspace.id, "path": str(path), "mode": mode,
                         "open_url": links.link(workspace.id, tab="notes"), "warnings": []})


def regenerate_notes_recap(workspace_id: str) -> dict:
    """What the Notes tab's "Regenerate the recap" does: the performance
    summary at the top of the notes, written again from the current
    numbers (the notes are started, with the workspace's name as title,
    when it has none)."""
    from lookin.core import notes as nt

    with tools._lock:
        entry = tools._entry(workspace_id)
        state = entry.loaded.state
        if state.dataset is None:
            raise ValueError("This workspace's dataset is not loaded, so there are no numbers to sum up. "
                             + " ".join(entry.loaded.warnings))
        metrics = {name: tools._metrics(entry, name) for name in state.experiments}
        workspace = entry.loaded.workspace
        recap = nt.recap(state, metrics)
        current = nt.read(workspace)
        text = nt.with_recap(current, recap) if current is not None else nt.template(workspace.label(), recap)
        path = nt.write(workspace, text)
        out = tools._context(entry, "notes")
    out.update({"path": str(path), "recap": recap, "started": current is None})
    return tools._clean(out)
