"""Load a library workspace into an AppState with no UI.

The Sources tab restores a workspace through its own widgets (see
SourcesView.restore_mount), because there every step also fills a field and
reports progress. Anything else that needs a workspace's numbers -- the MCP
endpoint, a script -- comes through here instead, and gets the same state:
the same columns, row filter, split kind and review version the workspace
was saved with.

Nothing that goes wrong for one experiment stops the others. What could not
be loaded is reported in `warnings`, in words that can be shown as they are,
because a caller that silently drops an experiment produces a ranking that
looks complete and is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lookin.core import library as lib
from lookin.core import workspace_sources
from lookin.core.data_manager import SPLIT_KINDS, SPLIT_TEST, AppState, ColumnConfig

# Below this share of an experiment's rows matching the dataset, its metrics
# describe a small corner of the pool at best -- usually a results CSV scored
# on a different dataset.
LOW_OVERLAP_SHARE = 0.5


@dataclass
class LoadedWorkspace:
    workspace: lib.Workspace
    state: AppState
    warnings: list = field(default_factory=list)
    # Experiments listed in the workspace that are not in `state`.
    skipped: list = field(default_factory=list)


def find_workspace(key: str, root=None) -> lib.Workspace:
    """A workspace by id, or else by exact name. Raises ValueError when
    there is none, or when a name is shared by several workspaces."""
    key = (key or "").strip()
    workspaces = lib.list_workspaces(root)
    for workspace in workspaces:
        if workspace.id == key:
            return workspace
    named = [w for w in workspaces if w.name == key]
    if len(named) == 1:
        return named[0]
    if named:
        ids = ", ".join(w.id for w in named)
        raise ValueError(f"{len(named)} workspaces are named {key!r} ({ids}) -- use the id instead.")
    raise ValueError(f"No workspace with id or name {key!r}.")


def load_workspace_state(workspace: lib.Workspace) -> LoadedWorkspace:
    mount = workspace.mount
    state = AppState()
    loaded = LoadedWorkspace(workspace=workspace, state=state)
    if not mount.dataset_path:
        loaded.warnings.append("This workspace has no dataset yet, so none of its experiments can be scored.")
        loaded.skipped = [e.name for e in mount.experiments]
        return loaded
    if not mount.exists():
        loaded.warnings.append(f"The dataset CSV {mount.dataset_path} is not on this machine.")
        loaded.skipped = [e.name for e in mount.experiments]
        return loaded

    columns = ColumnConfig(uuid_col=mount.uuid_col, crop_col=mount.crop_col or "crop",
                           label_col=mount.label_col or "label")
    row_filter = (mount.row_filter_col, mount.row_filter_value) if mount.row_filter_col else None
    state.split_kind = mount.split_kind if mount.split_kind in SPLIT_KINDS else SPLIT_TEST
    state.load_dataset(mount.dataset_path, columns, row_filter=row_filter, image_root=mount.image_root)
    if workspace.dataset_changed():
        loaded.warnings.append(
            "The dataset CSV changed on disk since this workspace was saved: rows may have appeared or "
            "disappeared, so these numbers may not match earlier ones.")
    store = state.review
    if store is not None and mount.review_version and mount.review_version in store.versions:
        store.active_version_id = mount.review_version

    for experiment in mount.experiments:
        reason = _unloadable_reason(experiment)
        if reason:
            loaded.skipped.append(experiment.name)
            loaded.warnings.append(f"Experiment {experiment.name!r} was not loaded: {reason}")
            continue
        try:
            state.add_experiment(experiment.path, experiment.name, uuid_col=mount.uuid_col or "uuid")
        except Exception as ex:
            loaded.skipped.append(experiment.name)
            loaded.warnings.append(f"Experiment {experiment.name!r} could not be read: {ex}")
            continue
        overlap = state.uuid_overlap(experiment.name)
        n_rows = len(state.experiments[experiment.name].df)
        if n_rows and overlap.matched < LOW_OVERLAP_SHARE * n_rows:
            loaded.warnings.append(
                f"Only {overlap.matched:,} of experiment {experiment.name!r}'s {n_rows:,} rows match a crop "
                f"of the dataset -- it may have been scored on another dataset.")
    return loaded


def _unloadable_reason(experiment: lib.MountExperiment) -> str:
    if not experiment.path:
        return "it has no results CSV path."
    if experiment.from_mlflow() and not workspace_sources.is_downloaded(experiment):
        return "its MLflow files are not downloaded yet (use the workspace's download button on Lookin's Home page)."
    if not Path(experiment.path).is_file():
        return f"its results CSV {experiment.path} is not on this machine."
    return ""
