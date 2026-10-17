"""The snapshot library: named, taggable workspaces of work that outlive a
session, and the snapshots inside them.

Two objects, and the difference between them is the whole design:

    Workspace    one mount -- a dataset, its column mapping, its experiments
              and their sidecars -- with a name, tags and a description.
              This is "the eval of checkpoint 42 on dataset B". It is what
              you organise, and what you delete to reclaim disk.

    Snapshot  a saved *view* of that workspace: which tab, which filters,
              which projection settings. This is "the confusion on class 3
              that looked wrong". A few KB, and many of them per workspace.

Keeping them apart is what makes ten snapshots of one eval cost ten small
JSON files rather than ten copies of a multi-hundred-megabyte mount.

On disk
-------
A workspace is a *directory*, not an archive::

    <library-root>/
      eval-ckpt42-on-dsb-lkd7f3a2b.lookin/
        manifest.json          the mount, the metadata, the provenance
        data/                  materialised only by an export (see below)
        snapshots/
          lk-7f3a2b.json

A directory rather than a zip for three reasons, all of which bite
immediately if you pick the zip:

- saving a snapshot writes one small file, instead of rewriting an archive
  that may hold hundreds of megabytes of embeddings;
- snapshots cannot dangle, because they live inside the thing they
  describe;
- two people can save snapshots into a workspace on a shared mount without
  locking, because neither ever writes a file the other one writes.

`data/` stays empty during normal use: the manifest references the dataset
and results CSVs where they already are, and only an export copies them in.
A workspace is therefore a few kilobytes until the moment you actually share
it.

What is deliberately *not* here
-------------------------------
Favourites and last-opened are per *user*, not per workspace, so they live in
the config directory (see load_personal). Writing a favourite into a workspace
on a shared NAS would mean two colleagues overwriting each other's.

Same rules as core/prefs.py, for the same reasons: stdlib only, atomic
writes, and an unreadable file is skipped with a log rather than raised --
a hand-mangled manifest must never be what stops the library listing.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import re
import secrets
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from lookin.core.prefs import config_dir

WORKSPACE_SUFFIX = ".lookin"
MANIFEST_NAME = "manifest.json"
SNAPSHOTS_DIRNAME = "snapshots"
DATA_DIRNAME = "data"
PERSONAL_FILE = "personal.json"

LIBRARY_ENV = "LOOKIN_LIBRARY"

# Bumped whenever the *shape* of a snapshot's view_state changes. A snapshot
# from a future version restores its mount and says so, rather than
# half-applying a state it only partly understands -- a view restored to 90%
# of what it was is worse than one not restored at all, because the user
# believes they are looking at the same thing.
VIEW_STATE_VERSION = 1

# No 0/1/i/l/o: an id is meant to be read off a printed report and retyped.
# 31 symbols over 6 characters is ~887 million, which is far more than a
# library of a few hundred entries can collide in.
ID_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"
ID_LENGTH = 6
SNAPSHOT_PREFIX = "lk-"
WORKSPACE_PREFIX = "lkw-"
# Workspaces were called bundles until 2026-09-24 and got "lkd-" ids. Ids
# are opaque and never re-issued, so those keep working as they are; only
# new workspaces get the new prefix.
LEGACY_WORKSPACE_PREFIX = "lkd-"

# Enough of a sha256 to identify a file, short enough to read in a manifest.
# This answers "is this the same content", not "is this cryptographically
# the same file" -- the threat here is a colleague's near-identical CSV, not
# an adversary.
HASH_CHARS = 16
_HASH_CHUNK = 1 << 20


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# ids and names
# ---------------------------------------------------------------------------

def new_id(prefix: str = SNAPSHOT_PREFIX) -> str:
    """A fresh id. Random rather than sequential: ids are handed between
    machines that share no counter, and a colleague importing a workspace must
    not be able to collide with one of their own."""
    return prefix + "".join(secrets.choice(ID_ALPHABET) for _ in range(ID_LENGTH))


def slugify(name: str, max_length: int = 40) -> str:
    """A filesystem-safe stem for a workspace directory. Cosmetic only -- the
    id beside it is what identifies the workspace, which is why renaming a
    workspace never renames its directory."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug[:max_length].strip("-") or "workspace"


def workspace_dirname(name: str, workspace_id: str) -> str:
    return f"{slugify(name)}-{workspace_id}{WORKSPACE_SUFFIX}"


def hash_file(path) -> str:
    """Content hash of a file, or "" if it cannot be read.

    Read in chunks: a dataset CSV is routinely hundreds of megabytes, and
    slurping it whole to hash it would cost more memory than loading the
    frame it describes.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()[:HASH_CHARS]
    except OSError as ex:
        _log(f"library: could not hash {path} ({ex})")
        return ""


# ---------------------------------------------------------------------------
# where the library lives
# ---------------------------------------------------------------------------

def default_library_root() -> Path:
    """`$LOOKIN_LIBRARY` if set, else a directory beside the other
    machine-local state. Pointing this at a shared mount is the only change
    needed to share a workspace by id instead of by sending a file."""
    override = os.environ.get(LIBRARY_ENV, "").strip()
    return Path(override).expanduser() if override else config_dir() / "library"


_library_root: Optional[Path] = None


def library_root() -> Path:
    return _library_root if _library_root is not None else default_library_root()


def set_library_root(path) -> None:
    """Point the library elsewhere -- used by `lookin ui --library`."""
    global _library_root
    _library_root = Path(path).expanduser() if path else None


# ---------------------------------------------------------------------------
# The mount: which dataset, mapped how, with which experiments attached.
#
# This is what a workspace records, and the shape UploadView both produces and
# consumes -- so a workspace is restored by exactly the same code path as any
# other load. It lived in core/prefs.py while the Data tab's session
# history was its only consumer; that history is gone, and a "recent
# session" was never a good name for the thing a workspace is built around.
# ---------------------------------------------------------------------------

@dataclass
class MountExperiment:
    """One results CSV in a mount, and -- when it came from MLflow -- the
    run and artifact it is a copy of.

    `path` is always where the CSV is read from on this machine. For an
    MLflow experiment that is its place in the download cache, which may
    not be filled yet: adding a run to a workspace records *what* to fetch,
    and the workspace's download button is what fetches it (see
    core/workspace_sources.py). The mlflow_* fields are what makes the link
    between a workspace and its runs visible in both directions, and what
    the "open in MLflow" links are built from.
    """
    name: str = ""
    path: str = ""
    embeddings_path: str = ""
    prototypes_path: str = ""
    mlflow_uri: str = ""
    mlflow_experiment_id: str = ""
    mlflow_run_id: str = ""
    mlflow_run_name: str = ""
    artifact_path: str = ""
    artifact_size: Optional[int] = None
    embeddings_artifact: str = ""
    embeddings_size: Optional[int] = None
    prototypes_artifact: str = ""
    prototypes_size: Optional[int] = None

    def from_mlflow(self) -> bool:
        return bool(self.mlflow_run_id and self.artifact_path)


@dataclass
class Mount:
    """One remembered mount. `dataset_size`/`dataset_mtime` are kept so the
    UI can warn that the CSV changed since the session was saved: review
    decisions stay valid either way (they're uuid-keyed), but rows may have
    appeared or disappeared, which is worth knowing before relaunching a
    training run off it."""
    dataset_path: str = ""
    uuid_col: str = "uuid"
    crop_col: str = "crop"
    label_col: str = "label"
    image_root: str = ""
    row_filter_col: str = ""
    row_filter_value: str = ""
    split_kind: str = ""
    review_version: str = ""
    experiments: list = field(default_factory=list)  # list[MountExperiment]
    saved_at: str = ""
    dataset_size: int = 0
    dataset_mtime: float = 0.0
    # The Filter tab's filter (a core.dataset_filter spec; {} for none).
    # Written by `lookin web`; `lookin ui` reads and keeps it, but does not
    # apply it.
    dataset_filter: dict = field(default_factory=dict)

    def label(self) -> str:
        return Path(self.dataset_path).name or self.dataset_path

    def exists(self) -> bool:
        try:
            return bool(self.dataset_path) and Path(self.dataset_path).is_file()
        except OSError:
            return False

    def changed_on_disk(self) -> bool:
        """True if the CSV's size or mtime moved since this was saved."""
        if not self.exists() or not self.dataset_mtime:
            return False
        try:
            stat = Path(self.dataset_path).stat()
        except OSError:
            return False
        return int(stat.st_size) != int(self.dataset_size) or abs(stat.st_mtime - self.dataset_mtime) > 1.0

    def summary(self) -> str:
        bits = []
        if self.split_kind:
            bits.append(f"split: {self.split_kind}")
        if self.row_filter_col and self.row_filter_value:
            bits.append(f"{self.row_filter_col}={self.row_filter_value}")
        if self.dataset_filter:
            from lookin.core.dataset_filter import describe
            n = len(describe(self.dataset_filter))
            if n:
                bits.append(f"filtered ({n} condition{'s' if n > 1 else ''})")
        if self.experiments:
            bits.append(f"{len(self.experiments)} experiment(s)")
        if self.review_version:
            bits.append(f"review {self.review_version}")
        return "  ·  ".join(bits)


def mount_to_dict(mount: Mount) -> dict:
    """The JSON shape of one mount. Public because a mount is also what a
    library workspace's manifest stores (see core/library.py) -- the two files
    disagreeing about the shape of a mount is exactly the bug that would
    make a saved workspace unrestorable."""
    return {
        **{k: v for k, v in asdict(mount).items() if k != "experiments"},
        "experiments": [asdict(e) for e in mount.experiments],
    }


_EXPERIMENT_SIZE_FIELDS = ("artifact_size", "embeddings_size", "prototypes_size")


def _optional_int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _experiment_from_dict(raw: dict) -> MountExperiment:
    """Every field optional: a manifest written before experiments knew
    their MLflow run has none of the mlflow_* keys, and is a local-file
    experiment exactly as it always was."""
    kwargs = {}
    for name in MountExperiment.__dataclass_fields__:
        if name not in raw:
            continue
        kwargs[name] = _optional_int(raw[name]) if name in _EXPERIMENT_SIZE_FIELDS else str(raw[name] or "")
    return MountExperiment(**kwargs)


def mount_from_dict(raw: dict) -> Mount:
    experiments = [_experiment_from_dict(e) for e in raw.get("experiments", []) or [] if isinstance(e, dict)]
    known = {f for f in Mount.__dataclass_fields__ if f != "experiments"}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if not isinstance(kwargs.get("dataset_filter", {}), dict):
        kwargs.pop("dataset_filter")
    return Mount(experiments=experiments, **kwargs)


# ---------------------------------------------------------------------------
# the two records
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """One snapshot. `view_state` is opaque here on purpose: this module
    stores and versions it, the views themselves decide what goes in it
    (see each view's view_state()/restore_view_state())."""
    id: str = ""
    name: str = ""
    description: str = ""
    tags: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    view_state: dict = field(default_factory=dict)
    view_state_version: int = VIEW_STATE_VERSION
    # Which workspace this belongs to. Redundant with where the file sits, and
    # kept anyway: it is what lets a snapshot file that was copied out of
    # its workspace be recognised as belonging elsewhere rather than silently
    # applied to whatever it landed next to.
    workspace_id: str = ""

    def label(self) -> str:
        return self.name or self.id

    def understood(self) -> bool:
        """False for a snapshot written by a newer Lookin. Its mount can
        still be opened; its view must not be half-applied."""
        return self.view_state_version <= VIEW_STATE_VERSION


@dataclass
class Workspace:
    """One mount, named and organised. `path` is where it was found and is
    never serialised -- a workspace that has been moved or copied is still the
    same workspace, which is what makes sharing one work at all."""
    id: str = ""
    name: str = ""
    description: str = ""
    tags: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    mount: Mount = field(default_factory=Mount)
    # Free-form: MLflow tracking URI, run id, checkpoint, whoever made it.
    # Text to read later, never used to fetch anything -- a colleague who
    # imports this workspace is on a different MLflow, or none at all.
    provenance: dict = field(default_factory=dict)
    dataset_hash: str = ""
    path: Optional[Path] = None

    def label(self) -> str:
        return self.name or Path(self.mount.dataset_path).name or self.id

    def snapshots_dir(self) -> Path:
        return Path(self.path) / SNAPSHOTS_DIRNAME

    def data_dir(self) -> Path:
        return Path(self.path) / DATA_DIRNAME

    def exists(self) -> bool:
        try:
            return self.path is not None and (Path(self.path) / MANIFEST_NAME).is_file()
        except OSError:
            return False

    def dataset_changed(self) -> bool:
        """True when the dataset CSV on disk is no longer the one this
        workspace was built from. A mismatch renders plausible, wrong numbers
        with no other symptom, so callers must say so rather than swallow
        it."""
        if not self.dataset_hash or not self.mount.dataset_path:
            return False
        current = hash_file(self.mount.dataset_path)
        return bool(current) and current != self.dataset_hash

    def summary(self) -> str:
        bits = []
        if self.mount.split_kind:
            bits.append(f"split: {self.mount.split_kind}")
        if self.mount.experiments:
            bits.append(f"{len(self.mount.experiments)} experiment(s)")
        if self.tags:
            bits.append(", ".join(self.tags))
        return "  ·  ".join(bits)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def _atomic_write_json(payload: dict, path: Path) -> bool:
    """Temp file in the same directory + os.replace, so a crash or a full
    disk mid-write leaves the previous file intact rather than a truncated
    one. Same directory matters: os.replace is only atomic within one
    filesystem."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as ex:
        _log(f"library: could not write {path} ({ex})")
        return False


def _read_json(path: Path) -> Optional[dict]:
    """None for "not there" and for "not readable" alike -- but only the
    second is logged. A file that has simply never been written is the
    normal first-run case, and announcing it every time would bury the
    corrupt-file message that actually matters."""
    if not Path(path).is_file():
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        _log(f"library: ignoring unreadable {path} ({ex})")
        return None
    return raw if isinstance(raw, dict) else None


def _as_str_list(value) -> list:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float))]


def workspace_to_dict(workspace: Workspace) -> dict:
    return {
        "version": 1,
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "tags": list(workspace.tags),
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at,
        "mount": mount_to_dict(workspace.mount),
        "provenance": dict(workspace.provenance),
        "dataset_hash": workspace.dataset_hash,
    }


def workspace_from_dict(raw: dict, path: Optional[Path] = None) -> Workspace:
    mount_raw = raw.get("mount")
    mount = mount_from_dict(mount_raw) if isinstance(mount_raw, dict) else Mount()
    provenance = raw.get("provenance")
    return Workspace(
        id=str(raw.get("id", "")),
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        tags=_as_str_list(raw.get("tags")),
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        mount=mount,
        provenance=provenance if isinstance(provenance, dict) else {},
        dataset_hash=str(raw.get("dataset_hash", "")),
        path=Path(path) if path is not None else None,
    )


def snapshot_to_dict(snapshot: Snapshot) -> dict:
    return {
        "id": snapshot.id,
        "name": snapshot.name,
        "description": snapshot.description,
        "tags": list(snapshot.tags),
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
        "view_state": dict(snapshot.view_state),
        "view_state_version": int(snapshot.view_state_version),
        "workspace_id": snapshot.workspace_id,
    }


def snapshot_from_dict(raw: dict) -> Snapshot:
    view_state = raw.get("view_state")
    try:
        version = int(raw.get("view_state_version", VIEW_STATE_VERSION))
    except (TypeError, ValueError):
        version = VIEW_STATE_VERSION
    return Snapshot(
        id=str(raw.get("id", "")),
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        tags=_as_str_list(raw.get("tags")),
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        view_state=view_state if isinstance(view_state, dict) else {},
        view_state_version=version,
        # "bundle_id" is what snapshots were written with before workspaces
        # were called workspaces; those files are still read as they are.
        workspace_id=str(raw.get("workspace_id", raw.get("bundle_id", ""))),
    )


# ---------------------------------------------------------------------------
# workspaces
# ---------------------------------------------------------------------------

def create_workspace(
    name: str, mount: Mount, description: str = "", tags: Optional[list] = None,
    provenance: Optional[dict] = None, root=None, hash_dataset: bool = True,
) -> Optional[Workspace]:
    """Write a new workspace directory and return it, or None if it could not
    be written.

    `hash_dataset` is honoured rather than assumed: hashing a 500k-row CSV
    takes a second or two, which is fine here (the user clicked "save") and
    is not fine on every listing.
    """
    workspace_id = new_id(WORKSPACE_PREFIX)
    root_path = Path(root) if root is not None else library_root()
    path = root_path / workspace_dirname(name, workspace_id)
    stamp = _now()
    workspace = Workspace(
        id=workspace_id,
        name=name,
        description=description,
        tags=list(tags or []),
        created_at=stamp,
        updated_at=stamp,
        mount=mount,
        provenance=dict(provenance or {}),
        dataset_hash=hash_file(mount.dataset_path) if hash_dataset and mount.dataset_path else "",
        path=path,
    )
    try:
        (path / SNAPSHOTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        _log(f"library: could not create {path} ({ex})")
        return None
    return workspace if save_workspace(workspace) else None


def save_workspace(workspace: Workspace) -> bool:
    """Rewrite a workspace's manifest. Used for metadata edits, which keep the
    id and the directory -- see the copy-on-write rule: only a change of
    *state* forks a new object, a rename never does."""
    if workspace.path is None:
        return False
    workspace.updated_at = _now()
    return _atomic_write_json(workspace_to_dict(workspace), Path(workspace.path) / MANIFEST_NAME)


# Guards update_workspace's read-change-write. Every browser tab is a session
# in this process, so a process lock is what keeps two of them from both
# reading a manifest, each changing it, and the second write erasing the
# first.
_UPDATE_LOCK = threading.Lock()


def update_workspace(path, change) -> Optional[Workspace]:
    """Apply `change(workspace) -> bool` to the manifest as it is *on disk
    now*, and save it if it returns True. Returns the fresh workspace (None
    when it is gone).

    Always this rather than editing a Workspace held in memory and calling
    save_workspace: the one in memory was read when a page was drawn, and
    another tab -- or another person on a shared library -- may have added
    an experiment since. Saving the old copy would silently remove it.
    """
    with _UPDATE_LOCK:
        workspace = load_workspace(path)
        if workspace is None:
            return None
        if change(workspace):
            if not save_workspace(workspace):
                return None
        return workspace


def update_workspace_mount(path, mount: Mount) -> Optional[Workspace]:
    """Make `mount` -- what is loaded right now -- the workspace's own,
    keeping its id, its directory and its snapshots. What "Save as
    workspace" followed by deleting the old one used to do, minus losing
    every id a report quotes.

    One exception to "what is loaded": an MLflow experiment of the
    workspace whose CSV was never downloaded on this machine could not have
    been loaded, so its absence says nothing about the user removing it,
    and it is kept. Returns the fresh workspace, None if it could not be
    written."""
    digest = hash_file(mount.dataset_path) if mount.dataset_path else ""

    def change(fresh: Workspace) -> bool:
        loaded = {e.name for e in mount.experiments}
        kept = [e for e in fresh.mount.experiments
                if e.name not in loaded and e.from_mlflow() and not Path(e.path).is_file()]
        fresh.mount = replace(mount, experiments=[*mount.experiments, *kept])
        fresh.dataset_hash = digest
        return True

    return update_workspace(path, change)


def load_workspace(path) -> Optional[Workspace]:
    raw = _read_json(Path(path) / MANIFEST_NAME)
    return workspace_from_dict(raw, path=Path(path)) if raw is not None else None


def list_workspaces(root=None) -> list:
    """Every readable workspace under the root, most recently updated first.

    An unreadable manifest is skipped rather than raised: one hand-edited
    file must not be what empties the library window.
    """
    root_path = Path(root) if root is not None else library_root()
    if not root_path.is_dir():
        return []
    workspaces = []
    try:
        entries = sorted(os.scandir(root_path), key=lambda e: e.name)
    except OSError as ex:
        _log(f"library: could not list {root_path} ({ex})")
        return []
    for entry in entries:
        if not entry.is_dir() or not entry.name.endswith(WORKSPACE_SUFFIX):
            continue
        workspace = load_workspace(entry.path)
        if workspace is not None:
            workspaces.append(workspace)
    workspaces.sort(key=lambda b: b.updated_at, reverse=True)
    return workspaces


def delete_workspace(workspace: Workspace) -> bool:
    """Remove a workspace and everything in it, snapshots included.

    Guarded twice on purpose -- this is the only call in the app that
    removes a directory tree, and the path it is handed comes from a
    manifest that may have been edited by hand.
    """
    if workspace.path is None:
        return False
    path = Path(workspace.path)
    if path.suffix != WORKSPACE_SUFFIX or not (path / MANIFEST_NAME).is_file():
        _log(f"library: refusing to delete {path} -- not a workspace directory")
        return False
    try:
        shutil.rmtree(path)
        return True
    except OSError as ex:
        _log(f"library: could not delete {path} ({ex})")
        return False


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------

def create_snapshot(
    workspace: Workspace, view_state: dict, name: str = "", description: str = "",
    tags: Optional[list] = None,
) -> Optional[Snapshot]:
    stamp = _now()
    snapshot = Snapshot(
        id=new_id(SNAPSHOT_PREFIX),
        name=name,
        description=description,
        tags=list(tags or []),
        created_at=stamp,
        updated_at=stamp,
        view_state=dict(view_state),
        view_state_version=VIEW_STATE_VERSION,
        workspace_id=workspace.id,
    )
    return snapshot if save_snapshot(workspace, snapshot) else None


def fork_snapshot(workspace: Workspace, snapshot: Snapshot, view_state: dict) -> Optional[Snapshot]:
    """Re-saving a snapshot with *different state* creates a new one.

    This is what protects an id written into a report months ago: renaming
    and re-tagging keep the id (save_snapshot below), but the view the id
    points at can never change under it.
    """
    return create_snapshot(
        workspace, view_state,
        name=snapshot.name, description=snapshot.description, tags=list(snapshot.tags),
    )


def save_snapshot(workspace: Workspace, snapshot: Snapshot) -> bool:
    """Write a snapshot in place. Metadata edits go through here and keep
    the id; a state change must go through fork_snapshot instead."""
    if workspace.path is None or not _is_plain_id(snapshot.id):
        return False
    snapshot.updated_at = _now()
    snapshot.workspace_id = snapshot.workspace_id or workspace.id
    return _atomic_write_json(
        snapshot_to_dict(snapshot), workspace.snapshots_dir() / f"{snapshot.id}.json"
    )


def _is_plain_id(snapshot_id: str) -> bool:
    """A snapshot id names a file in the snapshots folder, and it arrives in
    a URL (?snapshot=) or inside a snapshot file: one with a path separator
    or a leading dot would reach another .json on the disk -- read as a
    snapshot, or deleted with one."""
    text = str(snapshot_id or "")
    return bool(text) and not text.startswith(".") and not any(c in text for c in "/\\:\x00")


def load_snapshot(workspace: Workspace, snapshot_id: str) -> Optional[Snapshot]:
    if workspace.path is None or not _is_plain_id(snapshot_id):
        return None
    raw = _read_json(workspace.snapshots_dir() / f"{snapshot_id}.json")
    return snapshot_from_dict(raw) if raw is not None else None


def list_snapshots(workspace: Workspace) -> list:
    """This workspace's snapshots, most recently updated first."""
    if workspace.path is None:
        return []
    directory = workspace.snapshots_dir()
    if not directory.is_dir():
        return []
    snapshots = []
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError as ex:
        _log(f"library: could not list {directory} ({ex})")
        return []
    for entry in entries:
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        raw = _read_json(Path(entry.path))
        if raw is not None:
            snapshots.append(snapshot_from_dict(raw))
    snapshots.sort(key=lambda s: s.updated_at, reverse=True)
    return snapshots


def delete_snapshot(workspace: Workspace, snapshot_id: str) -> bool:
    if workspace.path is None or not _is_plain_id(snapshot_id):
        return False
    try:
        (workspace.snapshots_dir() / f"{snapshot_id}.json").unlink()
        return True
    except OSError as ex:
        _log(f"library: could not delete snapshot {snapshot_id} ({ex})")
        return False


def find_snapshot(snapshot_id: str, root=None) -> Optional[tuple]:
    """(workspace, snapshot) for an id typed by the user or arriving in a URL,
    or None.

    A scan rather than an index: a few hundred small files answer in
    milliseconds, and an index is one more thing that can disagree with the
    filesystem after somebody copies a workspace in by hand.
    """
    if not snapshot_id:
        return None
    wanted = snapshot_id.strip().lower()
    for workspace in list_workspaces(root):
        snapshot = load_snapshot(workspace, wanted)
        if snapshot is not None:
            return workspace, snapshot
    return None


# ---------------------------------------------------------------------------
# per-user preferences
#
# Favourites and last-opened belong to the person, not to the workspace. A
# favourite written into a workspace on a shared mount would mean two
# colleagues silently overwriting each other's, which is why these live in
# the config directory and never travel with an export.
# ---------------------------------------------------------------------------

def personal_path() -> Path:
    return config_dir() / PERSONAL_FILE


def load_personal() -> dict:
    """{"favorites": [id, ...], "opened": {id: iso8601}}, or empty."""
    raw = _read_json(personal_path())
    if raw is None:
        return {"favorites": [], "opened": {}}
    opened = raw.get("opened")
    return {
        "favorites": _as_str_list(raw.get("favorites")),
        "opened": opened if isinstance(opened, dict) else {},
    }


def save_personal(personal: dict) -> bool:
    return _atomic_write_json(
        {"favorites": _as_str_list(personal.get("favorites")),
         "opened": personal.get("opened") if isinstance(personal.get("opened"), dict) else {}},
        personal_path(),
    )


def is_favorite(object_id: str) -> bool:
    return object_id in load_personal()["favorites"]


def set_favorite(object_id: str, favorite: bool) -> bool:
    """Works for a workspace id and a snapshot id alike -- the ids are distinct
    by prefix, so one list holds both."""
    if not object_id:
        return False
    personal = load_personal()
    favorites = [f for f in personal["favorites"] if f != object_id]
    if favorite:
        favorites.append(object_id)
    personal["favorites"] = favorites
    return save_personal(personal)


def mark_opened(object_id: str) -> bool:
    if not object_id:
        return False
    personal = load_personal()
    personal["opened"][object_id] = _now()
    return save_personal(personal)


def last_opened(object_id: str) -> str:
    return str(load_personal()["opened"].get(object_id, ""))


# ---------------------------------------------------------------------------
# export and import
#
# A workspace references its CSVs where they already are, so the library stays a
# few kilobytes per entry however many workspaces you keep. Export is the moment
# that changes: it copies every file the workspace needs into data/ and zips the
# result, because the machine that opens it has none of those paths.
#
# Crops are deliberately never included. They are the only part that is
# genuinely large, the recipient normally already has them, and an image root
# is one question on import rather than a gigabyte in a zip.
# ---------------------------------------------------------------------------

DATA_PREFIX = f"{DATA_DIRNAME}/"


def _unique_name(used: set, name: str) -> str:
    """A name not already taken inside data/. Two experiments can easily
    have results CSVs with the same basename in different directories, and
    the second one silently overwriting the first is a workspace that scores
    one model twice."""
    candidate, stem, suffix, n = name, Path(name).stem, Path(name).suffix, 1
    while candidate in used:
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    used.add(candidate)
    return candidate


def export_workspace(workspace: Workspace, dest_path, include_embeddings: bool = False) -> tuple:
    """Write a self-contained .zip of `workspace`. Returns (ok, message).

    Every path in the exported manifest is rewritten to "data/<name>", so
    the machine that imports it resolves them against the workspace it
    extracted rather than against directories it does not have.
    """
    import zipfile

    dest = Path(dest_path)
    mount = workspace.mount
    if not mount.dataset_path:
        return False, "This workspace has no dataset to export."

    staged = {}   # arcname inside data/ -> source path on disk
    missing = []
    used: set = set()

    def stage(source: str) -> str:
        """Copy-plan one file, returning the path to record in the manifest."""
        if not source:
            return ""
        path = Path(source)
        if not path.is_file():
            missing.append(str(path))
            return ""
        arcname = _unique_name(used, path.name)
        staged[arcname] = path
        return DATA_PREFIX + arcname

    exported = Mount(
        dataset_path=stage(mount.dataset_path),
        uuid_col=mount.uuid_col, crop_col=mount.crop_col, label_col=mount.label_col,
        # The image root is the recipient's to supply -- their crops are
        # somewhere else, and a path from this machine would only mislead.
        image_root="",
        row_filter_col=mount.row_filter_col, row_filter_value=mount.row_filter_value,
        split_kind=mount.split_kind, review_version=mount.review_version,
        saved_at=mount.saved_at,
    )
    for experiment in mount.experiments:
        # replace() rather than a fresh MountExperiment: the MLflow run an
        # experiment came from travels with it, so the recipient still sees
        # which run it was -- they just read the copy in data/.
        exported.experiments.append(replace(
            experiment,
            path=stage(experiment.path),
            embeddings_path=stage(experiment.embeddings_path) if include_embeddings else "",
            prototypes_path=stage(experiment.prototypes_path) if include_embeddings else "",
        ))

    # Review decisions are part of what the user was looking at.
    from lookin.core.review import review_paths_for_csv
    review_files = []
    for path in review_paths_for_csv(mount.dataset_path):
        recorded = stage(str(path))
        if recorded:
            review_files.append(recorded)

    payload = workspace_to_dict(workspace)
    payload["mount"] = mount_to_dict(exported)
    payload["portable"] = True
    payload["review_files"] = review_files
    payload["exported_at"] = _now()

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.tmp{os.getpid()}.{threading.get_ident()}")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(payload, indent=2))
            for arcname, source in staged.items():
                archive.write(source, DATA_PREFIX + arcname)
            for snapshot in list_snapshots(workspace):
                archive.writestr(
                    f"{SNAPSHOTS_DIRNAME}/{snapshot.id}.json",
                    json.dumps(snapshot_to_dict(snapshot), indent=2),
                )
        os.replace(tmp, dest)
    except (OSError, ValueError) as ex:
        _log(f"library: could not export to {dest} ({ex})")
        return False, f"Could not write {dest.name}: {ex}"

    note = f"Exported {len(staged)} file(s) to {dest.name}."
    if missing:
        note += f" {len(missing)} file(s) were missing and were left out: {', '.join(Path(m).name for m in missing)}"
    return True, note


def _safe_members(archive, workspace_dir: Path) -> list:
    """The entries of an archive that are safe to extract.

    A workspace arrives from another machine, so an absolute path or one
    climbing out with ".." would write wherever it liked. Anything that
    does not land inside the destination is refused rather than sanitised:
    a workspace that tries it is not one to repair silently.
    """
    safe = []
    root = workspace_dir.resolve()
    for member in archive.namelist():
        if member.endswith("/"):
            continue
        target = (workspace_dir / member).resolve()
        if not str(target).startswith(str(root) + os.sep):
            _log(f"library: refusing archive entry outside the workspace: {member!r}")
            continue
        safe.append(member)
    return safe


def import_workspace(zip_path, root=None) -> tuple:
    """Extract an exported workspace into the library. Returns (workspace, notes).

    The manifest's "data/..." paths are turned back into absolute ones
    pointing at what was just extracted. The image root is deliberately not
    guessed -- the caller asks for it, because only the user knows where
    their crops are (see rebase_image_root).
    """
    import zipfile

    source = Path(zip_path)
    root_path = Path(root) if root is not None else library_root()
    notes = []
    try:
        with zipfile.ZipFile(source) as archive:
            try:
                raw = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            except (KeyError, ValueError) as ex:
                return None, [f"{source.name} is not a Lookin workspace ({ex})."]
            if not isinstance(raw, dict):
                return None, [f"{source.name} is not a Lookin workspace."]

            # A fresh id: the same workspace imported twice is two workspaces, and
            # reusing the sender's id would make one of them unreachable.
            workspace_id = new_id(WORKSPACE_PREFIX)
            name = str(raw.get("name", "")) or source.stem
            workspace_dir = root_path / workspace_dirname(name, workspace_id)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            for member in _safe_members(archive, workspace_dir):
                archive.extract(member, workspace_dir)
    except (OSError, zipfile.BadZipFile) as ex:
        return None, [f"Could not read {source.name}: {ex}"]

    workspace = workspace_from_dict(raw, path=workspace_dir)
    workspace.id = workspace_id
    workspace.path = workspace_dir

    def resolve(recorded: str) -> str:
        if not recorded:
            return ""
        if recorded.startswith(DATA_PREFIX):
            resolved = workspace_dir / recorded
            if not resolved.is_file():
                notes.append(f"Missing from the workspace: {recorded}")
                return ""
            return str(resolved)
        return recorded

    workspace.mount.dataset_path = resolve(workspace.mount.dataset_path)
    for experiment in workspace.mount.experiments:
        experiment.path = resolve(experiment.path)
        experiment.embeddings_path = resolve(experiment.embeddings_path)
        experiment.prototypes_path = resolve(experiment.prototypes_path)

    if not workspace.mount.dataset_path:
        notes.append("This workspace has no dataset CSV -- it cannot be opened.")
    (workspace_dir / SNAPSHOTS_DIRNAME).mkdir(exist_ok=True)
    save_workspace(workspace)
    notes.append(f"Imported as {workspace.label()} ({workspace.id}).")
    return workspace, notes


def rebase_image_root(workspace: Workspace, image_root) -> bool:
    """Point an imported workspace's crops at where they are on this machine.

    The dataset CSV holds crop paths relative to a root that only existed
    on the machine the workspace came from; this is the one field that has to
    be answered by the person opening it.
    """
    workspace.mount.image_root = str(image_root or "")
    return save_workspace(workspace)
