"""Versioned annotation-review state: the decisions a user makes while
cleaning a dataset, kept entirely outside the dataset CSV they were made
against.

Why not a column in the dataset CSV
-----------------------------------
Reviewing produces *decisions*, not labels. A decision carries a status, a
reason, a provenance and a version; a bare label column keeps none of that.
On top of which the dataset CSV is an *input* -- possibly read-only, on a
shared mount, hundreds of thousands of rows long -- so rewriting it on every
click is both slow and destructive. Decisions live here instead, in two
small sidecar CSVs next to the dataset (see review_paths_for_csv), and the
enriched "one column per version" dataset the user actually trains on is
produced on demand by enrich() -- an output, not a mutation of the input.

Versions
--------
A version is a named, dated, parented, freezable set of decisions.
Resolution is *cumulative*: v2 declares parent=v1, and the effective
decision for a uuid at v2 is the most recent one along the root->v2 chain
(see effective()). That means a version only ever stores what it changes,
which is what makes ten cleaning passes cheap, and it lets the UI say
"inherited from v1" instead of losing that distinction the way a flat
column would.

Freezing is what makes a training run reproducible: exporting for training
freezes the version it exported, so nobody can add three more decisions
afterwards and leave the run's dataset unidentifiable. Editing a frozen
version is refused (FrozenVersionError) -- the caller is expected to offer
creating a child version instead.

Everything here is plain pandas/stdlib over uuid-keyed rows: no split is
baked in anywhere, so decisions made while only the train split was loaded
still apply when the whole CSV is read back for export.
"""
from __future__ import annotations

import re
import threading
import time
import uuid as uuid_module
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from lookin.core import fileio

# -- statuses -------------------------------------------------------------
# These are *keys*, compared in code and written into files that outlive any
# wording of the UI. The human-readable label for each lives in the view
# layer (see core/review_actions.py STATUS_LABELS) and can be reworded freely;
# these strings can never be.
STATUS_UNREVIEWED = "unreviewed"
STATUS_OK = "ok"
STATUS_RELABEL = "relabel"
STATUS_EXCLUDE = "exclude"
STATUS_UNCERTAIN = "uncertain"

# Stored statuses, in the order the UI presents them. STATUS_UNREVIEWED is
# not among them: "never looked at" is the *absence* of a decision, and is
# only ever written as an explicit tombstone by reset() to cancel a
# decision inherited from a parent version.
REVIEW_STATUSES = (STATUS_OK, STATUS_RELABEL, STATUS_EXCLUDE, STATUS_UNCERTAIN)

# -- reasons --------------------------------------------------------------
# Also keys. A closed vocabulary rather than free text on purpose: it's what
# makes "60% of our exclusions are truncated crops" answerable, which points
# at the detector upstream rather than at the annotation -- a conclusion no
# amount of free-text notes would ever surface.
REASON_BLURRED = "blurred"
REASON_RADIOMETRY = "radiometry"
REASON_OCCLUDED = "occluded"
REASON_TRUNCATED = "truncated"
REASON_MULTIPLE_OBJECTS = "multiple_objects"
REASON_WRONG_CROP = "wrong_crop"
REASON_NOT_AN_OBJECT = "not_an_object"
REASON_DUPLICATE = "duplicate"
REASON_AMBIGUOUS = "ambiguous"

REVIEW_REASONS = (
    REASON_BLURRED, REASON_RADIOMETRY, REASON_OCCLUDED, REASON_TRUNCATED,
    REASON_MULTIPLE_OBJECTS, REASON_WRONG_CROP, REASON_NOT_AN_OBJECT,
    REASON_DUPLICATE, REASON_AMBIGUOUS,
)

# -- exported column names ------------------------------------------------
# The contract with the user's training code: <prefix><version_id>. Fixed
# and predictable by default, overridable at export time for a house
# convention.
LABEL_PREFIX = "label_review_"
STATUS_PREFIX = "review_status_"
REASON_PREFIX = "review_reason_"

DECISION_COLUMNS = [
    "version_id", "uuid", "status", "new_label", "reason", "note",
    "source", "batch_id", "author", "timestamp",
]
VERSION_COLUMNS = [
    "version_id", "parent", "created_at", "author", "note", "frozen",
    "exported_at", "exported_to",
]

NO_PARENT = "none"
# A version id becomes a column *suffix* in the exported CSV, so it has to
# survive a round-trip through a CSV header and through whatever reads it
# downstream -- hence no spaces, no case, no punctuation beyond '_'.
VERSION_ID_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class FrozenVersionError(Exception):
    """Raised when a decision is written to a frozen version. Callers are
    expected to catch this and offer creating a child version instead of
    surfacing it as a failure."""


def review_paths_for_csv(csv_path) -> tuple:
    """The by-convention (decisions, versions) sidecar paths for a dataset
    CSV -- 'ships.csv' pairs with 'ships_review.csv' and
    'ships_review_versions.csv'. Same shape as
    data_manager.sidecar_paths_for_csv, and deliberately keyed on the CSV
    rather than on a split: train and test rows filtered out of the same
    CSV share one store, which is what lets a decision made on the train
    pool still be there when the whole file is read back for export.
    """
    p = Path(csv_path)
    return p.with_name(f"{p.stem}_review.csv"), p.with_name(f"{p.stem}_review_versions.csv")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_SAVE_LOCK = threading.Lock()


def _signature(*paths) -> tuple:
    """(inode, size, mtime) of each file, or None for one that is absent --
    enough to tell that another session wrote it since. The inode because
    every save is a replace, which moves it even when two saves of the same
    size land within one mtime tick (a coarse one on a network mount)."""
    out = []
    for path in paths:
        try:
            stat = Path(path).stat()
            out.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
        except OSError:
            out.append(None)
    return tuple(out)


def _read_sidecar(path: Path) -> Optional[pd.DataFrame]:
    """One sidecar as text cells, or None when there is nothing in it.

    An empty file reads as absent: it holds no work to lose. One that is
    there and cannot be parsed is refused with its name, never treated as
    empty -- the next save would write this store over it, and a sidecar
    a hand edit broke usually still holds every decision, one stray quote
    away from readable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return None
    except (UnicodeDecodeError, pd.errors.ParserError) as ex:
        raise ValueError(
            f"The review file {path} could not be read ({ex}). Fix it, or move it away to start "
            f"without its decisions, then load the dataset again."
        ) from ex


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file in the same directory + os.replace, so a crash
    (or a full disk) mid-write leaves the previous store intact rather than
    a truncated file. Same directory matters: os.replace is only atomic
    within one filesystem."""
    fileio.atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))


@dataclass
class Decision:
    """One reviewer decision about one crop, within one version."""
    uuid: str
    status: str
    version_id: str = ""
    new_label: str = ""
    reason: str = ""
    note: str = ""
    source: str = ""
    batch_id: str = ""
    author: str = ""
    timestamp: str = ""

    def effective_label(self, original_label):
        """The label this decision implies. Only `relabel` changes it: an
        excluded crop keeps its label (knowing *what* was thrown away is
        half the value of the exclusion), and so does an uncertain one."""
        if self.status == STATUS_RELABEL and self.new_label:
            return self.new_label
        return original_label


@dataclass
class ReviewVersion:
    version_id: str
    parent: str = NO_PARENT
    created_at: str = ""
    author: str = ""
    note: str = ""
    frozen: bool = False
    exported_at: str = ""
    exported_to: str = ""


@dataclass
class ReviewStore:
    """All versions and decisions for one dataset CSV.

    `revision` is bumped on every mutation and is meant to be folded into
    any cache key downstream (see AppState.merged): a decision must not by
    itself invalidate an expensive merge, but a merge computed *through*
    the review overlay has to notice when the overlay changed.

    Several stores can be live on the same sidecars at once -- every
    browser tab is its own session, and two workspaces routinely share a
    dataset. So a store remembers what *it* changed since it last read or
    wrote the files (`_changed`, `_changed_versions`), and save() merges
    those changes into whatever is on disk when another store has written
    since (see _merge_from_disk) rather than overwriting it with a stale
    picture.
    """
    versions: dict = field(default_factory=dict)  # version_id -> ReviewVersion (insertion-ordered)
    decisions: dict = field(default_factory=dict)  # version_id -> {uuid -> Decision}
    active_version_id: Optional[str] = None
    decisions_path: Optional[Path] = None
    versions_path: Optional[Path] = None
    author: str = ""
    revision: int = 0
    _dirty: bool = False
    _changed: set = field(default_factory=set, repr=False, compare=False)          # {(version_id, uuid)}
    _changed_versions: set = field(default_factory=set, repr=False, compare=False)  # {version_id}
    _created_versions: set = field(default_factory=set, repr=False, compare=False)  # {version_id}
    _disk_signature: tuple = field(default=(), repr=False, compare=False)

    # ---- versions ----
    def next_version_id(self) -> str:
        """v1, v2, v3... skipping anything already taken (a user-named
        version could well be called 'v3' already)."""
        n = 1
        while f"v{n}" in self.versions:
            n += 1
        return f"v{n}"

    def create_version(
        self, version_id: Optional[str] = None, parent: Optional[str] = None,
        note: str = "", author: Optional[str] = None, activate: bool = True,
    ) -> ReviewVersion:
        """Create a version. `parent` defaults to the current active
        version, which is what makes the common case -- keep cleaning on
        top of what's already decided -- the default; pass NO_PARENT for a
        genuinely independent pass (a second reviewer, a blind re-review)."""
        version_id = (version_id or self.next_version_id()).strip().lower()
        if not VERSION_ID_RE.match(version_id):
            raise ValueError(
                f"Invalid version name '{version_id}': use lowercase letters, digits and underscores "
                f"only (max 32 chars) -- the name becomes a CSV column suffix, e.g. "
                f"'{LABEL_PREFIX}{version_id or 'v1'}'."
            )
        if version_id in self.versions:
            raise ValueError(f"A review version named '{version_id}' already exists.")
        if parent is None:
            parent = self.active_version_id or NO_PARENT
        if parent != NO_PARENT and parent not in self.versions:
            raise ValueError(f"No review version named '{parent}' to inherit from.")
        version = ReviewVersion(
            version_id=version_id, parent=parent, created_at=_now(),
            author=author if author is not None else self.author, note=note,
        )
        self.versions[version_id] = version
        self.decisions[version_id] = {}
        if activate:
            self.active_version_id = version_id
        self._changed_versions.add(version_id)
        self._created_versions.add(version_id)
        self._touch()
        _log(f"review: created version '{version_id}' (parent: {parent})")
        return version

    def freeze(self, version_id: str, exported_to: str = "") -> None:
        """Make a version immutable. Called automatically by an export for
        training: without this, a run's dataset is unidentifiable after the
        fact."""
        version = self.versions.get(version_id)
        if version is None or version.frozen:
            return
        self.versions[version_id] = replace(
            version, frozen=True, exported_at=_now(), exported_to=exported_to or version.exported_to,
        )
        self._changed_versions.add(version_id)
        self._touch()
        _log(f"review: froze version '{version_id}'" + (f" (exported to {exported_to})" if exported_to else ""))

    def note_export(self, version_id: str, destination: str) -> None:
        """Record where a version was last exported, freezing it in the
        process -- 'frozen' says a run depends on it, 'exported_to' says
        which file that run read."""
        self.freeze(version_id, exported_to=destination)

    def chain(self, version_id: Optional[str] = None) -> list:
        """The inheritance chain for a version, root first. Cycle-safe: a
        hand-edited manifest that points a version at itself (or at a
        descendant) stops at the repeat instead of hanging."""
        version_id = version_id or self.active_version_id
        out, seen = [], set()
        while version_id and version_id != NO_PARENT and version_id in self.versions and version_id not in seen:
            seen.add(version_id)
            out.append(version_id)
            version_id = self.versions[version_id].parent
        return list(reversed(out))

    def is_frozen(self, version_id: Optional[str] = None) -> bool:
        version = self.versions.get(version_id or self.active_version_id or "")
        return bool(version and version.frozen)

    def resume_version_id(self) -> Optional[str]:
        """The version to reopen on load: the most recently created one
        that isn't frozen. An export freezes the version it exported, so an
        exported version is never resumed -- the two rules the user asked
        for collapse into this single test. None means "they're all frozen,
        the caller should create a child"."""
        for version_id in reversed(list(self.versions)):
            if not self.versions[version_id].frozen:
                return version_id
        return None

    def ensure_active_version(self) -> ReviewVersion:
        """The version decisions land in, creating one if the store is
        empty or if every existing version is frozen."""
        if self.active_version_id and not self.is_frozen(self.active_version_id):
            return self.versions[self.active_version_id]
        resumed = self.resume_version_id()
        if resumed is not None:
            self.active_version_id = resumed
            return self.versions[resumed]
        parent = list(self.versions)[-1] if self.versions else NO_PARENT
        return self.create_version(parent=parent)

    # ---- decisions ----
    def effective(self, version_id: Optional[str] = None) -> dict:
        """uuid -> Decision, resolved along the root->version chain: a
        later version's decision wins, and an explicit `unreviewed`
        tombstone (written by reset) removes the uuid entirely.

        The returned Decision keeps its own version_id, so a caller can
        tell an inherited decision from one made in this version by
        comparing the two -- no second lookup structure needed.
        """
        out: dict = {}
        for vid in self.chain(version_id):
            for uid, decision in self.decisions.get(vid, {}).items():
                if decision.status == STATUS_UNREVIEWED:
                    out.pop(uid, None)
                else:
                    out[uid] = decision
        return out

    def own_decisions(self, version_id: Optional[str] = None) -> dict:
        """Only what this version itself stores -- what would be lost if it
        were deleted, as opposed to what it inherits."""
        return dict(self.decisions.get(version_id or self.active_version_id or "", {}))

    def decision_for(self, uid: str, version_id: Optional[str] = None) -> Optional[Decision]:
        return self.effective(version_id).get(str(uid))

    def decide(
        self, uid, status: str, new_label: str = "", reason: str = "", note: str = "",
        source: str = "", batch_id: str = "", version_id: Optional[str] = None,
    ) -> Decision:
        """Record one decision in the active (or given) version.

        A frozen active version raises rather than quietly rolling onto a
        fresh one: creating a version is a decision with consequences (it
        is what the next export will be named after), so it belongs to the
        user, via ensure_active_version, not to a side effect of clicking
        'exclude'.
        """
        version_id = version_id or self.active_version_id or self.ensure_active_version().version_id
        if self.is_frozen(version_id):
            raise FrozenVersionError(
                f"Version '{version_id}' is frozen because it was exported for training. "
                f"Create a new version to keep reviewing -- editing this one would make the "
                f"training run that used it unreproducible."
            )
        if status not in REVIEW_STATUSES and status != STATUS_UNREVIEWED:
            raise ValueError(f"Unknown review status '{status}'. Expected one of {REVIEW_STATUSES}.")
        if status == STATUS_RELABEL and not new_label:
            raise ValueError("A 'relabel' decision needs the new class label.")
        decision = Decision(
            uuid=str(uid), status=status, version_id=version_id,
            new_label=str(new_label or ""), reason=reason or "", note=note or "",
            source=source or "", batch_id=batch_id or "",
            author=self.author, timestamp=_now(),
        )
        self.decisions.setdefault(version_id, {})[str(uid)] = decision
        self._changed.add((version_id, str(uid)))
        self._touch()
        return decision

    def decide_many(
        self, uuids, status: str, new_label: str = "", reason: str = "", note: str = "",
        source: str = "", version_id: Optional[str] = None,
    ) -> str:
        """Apply one decision to many crops at once and return the batch id
        that ties them together. Bulk actions are the main way to lose work
        in this tool (a mis-clicked subcenter is hundreds of crops), so
        every one of them is undoable as a unit -- see undo_batch."""
        # Materialized up front: `uuids` is allowed to be any iterable (a
        # generator over a filtered candidate set, say), and it is both
        # looped over and counted below.
        uuids = list(uuids)
        batch_id = uuid_module.uuid4().hex[:12]
        for uid in uuids:
            self.decide(
                uid, status, new_label=new_label, reason=reason, note=note,
                source=source, batch_id=batch_id, version_id=version_id,
            )
        _log(f"review: batch {batch_id} -- {status} on {len(uuids):,} samples")
        return batch_id

    def undo_batch(self, batch_id: str) -> int:
        """Drop every decision made by one bulk action, across all
        non-frozen versions. Returns how many were removed."""
        removed = 0
        for version_id, by_uuid in self.decisions.items():
            if self.is_frozen(version_id):
                continue
            for uid in [u for u, d in by_uuid.items() if d.batch_id == batch_id]:
                del by_uuid[uid]
                self._changed.add((version_id, uid))
                removed += 1
        if removed:
            self._touch()
            _log(f"review: undid batch {batch_id} ({removed:,} decisions)")
        return removed

    def reset(self, uid, version_id: Optional[str] = None) -> None:
        """Back to genuinely-unreviewed. Drops this version's own decision;
        if a parent version still decides the uuid, writes an explicit
        `unreviewed` tombstone so the inherited decision stops applying
        too -- otherwise there would be no way to undo a parent's call
        without editing (and unfreezing) the parent."""
        version_id = version_id or self.active_version_id or self.ensure_active_version().version_id
        if self.is_frozen(version_id):
            raise FrozenVersionError(f"Version '{version_id}' is frozen.")
        self.decisions.setdefault(version_id, {}).pop(str(uid), None)
        self._changed.add((version_id, str(uid)))
        parent_chain = self.chain(version_id)[:-1]
        inherited = any(str(uid) in self.decisions.get(vid, {}) for vid in parent_chain)
        if inherited:
            self.decide(uid, STATUS_UNREVIEWED, source="reset", version_id=version_id)
        else:
            self._touch()

    # ---- summaries ----
    def counts(self, version_id: Optional[str] = None) -> dict:
        """status -> count for the resolved state of a version."""
        out = {status: 0 for status in REVIEW_STATUSES}
        for decision in self.effective(version_id).values():
            out[decision.status] = out.get(decision.status, 0) + 1
        return out

    def reason_counts(self, version_id: Optional[str] = None) -> dict:
        out: dict = {}
        for decision in self.effective(version_id).values():
            if decision.reason:
                out[decision.reason] = out.get(decision.reason, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def n_decisions(self, version_id: str) -> int:
        return len(self.decisions.get(version_id, {}))

    def batches(self, version_id: Optional[str] = None) -> list:
        """(batch_id, status, count, timestamp) for the bulk actions of one
        version, most recent first -- the undo list."""
        out: dict = {}
        for decision in self.decisions.get(version_id or self.active_version_id or "", {}).values():
            if not decision.batch_id:
                continue
            entry = out.setdefault(decision.batch_id, [decision.batch_id, decision.status, 0, decision.timestamp])
            entry[2] += 1
            entry[3] = max(entry[3], decision.timestamp)
        return sorted((tuple(v) for v in out.values()), key=lambda r: r[3], reverse=True)

    # ---- persistence ----
    def _touch(self) -> None:
        self.revision += 1
        self._dirty = True

    def to_frames(self) -> tuple:
        decision_rows = [
            {
                "version_id": d.version_id, "uuid": d.uuid, "status": d.status,
                "new_label": d.new_label, "reason": d.reason, "note": d.note,
                "source": d.source, "batch_id": d.batch_id, "author": d.author,
                "timestamp": d.timestamp,
            }
            for version_id in self.versions
            for d in self.decisions.get(version_id, {}).values()
        ]
        version_rows = [
            {
                "version_id": v.version_id, "parent": v.parent, "created_at": v.created_at,
                "author": v.author, "note": v.note, "frozen": bool(v.frozen),
                "exported_at": v.exported_at, "exported_to": v.exported_to,
            }
            for v in self.versions.values()
        ]
        return (
            pd.DataFrame(decision_rows, columns=DECISION_COLUMNS),
            pd.DataFrame(version_rows, columns=VERSION_COLUMNS),
        )

    def save(self, decisions_path=None, versions_path=None, force: bool = False) -> bool:
        """Persist both sidecars. No-op when nothing changed since the last
        save unless `force`. Returns whether anything was written."""
        decisions_path = Path(decisions_path or self.decisions_path or "")
        versions_path = Path(versions_path or self.versions_path or "")
        if not decisions_path.name or not versions_path.name:
            return False
        if not self._dirty and not force:
            return False
        t0 = time.monotonic()
        # Read-merge-write as one step: two tabs saving at the same instant
        # would otherwise both merge against the same file and the second
        # write would drop the first. The process lock covers the tabs of
        # one `lookin web`; the lock file covers a second server beside it, and
        # a colleague reviewing the same dataset on a shared mount.
        with _SAVE_LOCK, fileio.locked(decisions_path):
            if self._disk_signature and _signature(decisions_path, versions_path) != self._disk_signature:
                self._merge_from_disk(decisions_path, versions_path)
            decisions_df, versions_df = self.to_frames()
            _atomic_write_csv(decisions_df, decisions_path)
            _atomic_write_csv(versions_df, versions_path)
            self._disk_signature = _signature(decisions_path, versions_path)
        self.decisions_path, self.versions_path = decisions_path, versions_path
        self._dirty = False
        self._changed.clear()
        self._changed_versions.clear()
        self._created_versions.clear()
        _log(f"review: saved {len(decisions_df):,} decisions / {len(versions_df)} versions "
             f"to {decisions_path.name} ({time.monotonic() - t0:.2f}s)")
        return True

    def _merge_from_disk(self, decisions_path: Path, versions_path: Path) -> None:
        """Another store wrote the sidecars since this one last read them:
        take what is on disk, and apply only what this store changed on top.

        Decisions are merged per (version, uuid) -- this store's change wins
        where both touched the same crop, as in any last-writer-wins edit --
        and so are versions (created or frozen here). A version this store
        has that the file lacks is kept whole: the file cannot delete one.

        The one thing not carried over is a decision made here in a version
        the other store has since *frozen*: freezing means a training run
        was exported from it, and adding to it afterwards is exactly what
        freezing exists to prevent. Those are dropped, and logged.
        """
        disk = ReviewStore.load(decisions_path, versions_path, author=self.author)
        refused = 0
        renamed = self._rename_colliding_versions(disk)
        # Versions first: one this store created, froze or adopted replaces
        # the disk's copy, and one the disk has never seen comes with all
        # its decisions (an adopted version fills its decisions in bulk).
        for version_id, version in self.versions.items():
            if version_id in self._changed_versions or version_id not in disk.versions:
                if version_id not in disk.versions:
                    disk.decisions[version_id] = dict(self.decisions.get(version_id, {}))
                elif disk.versions[version_id].frozen and not version.frozen:
                    # Freezing only ever goes one way: a run was exported
                    # from it, and no merge may reopen it.
                    continue
                disk.versions[version_id] = version
        for version_id, uid in self._changed:
            if disk.is_frozen(version_id) and version_id not in self._changed_versions:
                refused += 1
                continue
            mine = self.decisions.get(version_id, {}).get(uid)
            by_uuid = disk.decisions.setdefault(version_id, {})
            if mine is None:
                by_uuid.pop(uid, None)
            else:
                by_uuid[uid] = mine
        self.versions, self.decisions = disk.versions, disk.decisions
        if self.active_version_id not in self.versions:
            self.active_version_id = self.resume_version_id()
        # What is displayed was computed from the old picture; bumping the
        # revision is what makes every overlay downstream recompute.
        self.revision += 1
        _log(f"review: merged with changes another session saved to {decisions_path.name}"
             + "".join(f"; version '{old}' was created elsewhere too, so this one is now '{new}'"
                       for old, new in renamed)
             + (f"; {refused} decision(s) dropped -- their version was frozen meanwhile" if refused else ""))

    def _rename_colliding_versions(self, disk: "ReviewStore") -> list:
        """A version this store created whose id another session created as
        well, and has since frozen -- two tabs each making "the next
        version" both call it v2. Merging the two would put this store's
        decisions into a version a training run was exported from, so this
        store's one moves to a fresh id instead, decisions and all. Returns
        [(old id, new id)].

        Only against a *frozen* namesake: two unfrozen ones are the same
        pass worked on from two tabs, which merges decision by decision like
        any other version."""
        renamed = []
        for old in [v for v in self.versions if v in self._created_versions]:
            theirs = disk.versions.get(old)
            if theirs is None or not theirs.frozen:
                continue
            n = 1
            while f"v{n}" in self.versions or f"v{n}" in disk.versions:
                n += 1
            new = f"v{n}"
            self.versions = {(new if k == old else k): (replace(v, version_id=new) if k == old else v)
                             for k, v in self.versions.items()}
            for vid, version in list(self.versions.items()):
                if version.parent == old and vid in self._created_versions:
                    self.versions[vid] = replace(version, parent=new)
            moved = self.decisions.pop(old, {})
            self.decisions[new] = {u: replace(d, version_id=new) for u, d in moved.items()}
            self._changed = {((new if vid == old else vid), u) for vid, u in self._changed}
            for tracked in (self._changed_versions, self._created_versions):
                if old in tracked:
                    tracked.discard(old)
                    tracked.add(new)
            if self.active_version_id == old:
                self.active_version_id = new
            renamed.append((old, new))
        return renamed

    @classmethod
    def load(cls, decisions_path, versions_path, author: str = "") -> "ReviewStore":
        """Read both sidecars, tolerating either being absent (a dataset
        that has never been reviewed) or empty.

        Under the save lock, so the two files are read as one save left
        them -- not the new decisions with the old version manifest."""
        with fileio.locked(decisions_path):
            return cls._load(decisions_path, versions_path, author)

    @classmethod
    def _load(cls, decisions_path, versions_path, author: str = "") -> "ReviewStore":
        store = cls(
            decisions_path=Path(decisions_path), versions_path=Path(versions_path), author=author,
        )
        versions_path, decisions_path = Path(versions_path), Path(decisions_path)
        # Taken before reading, not after: a save landing in between then
        # shows as a change at the next save (and is merged), instead of
        # being taken for what this store read.
        signature = _signature(decisions_path, versions_path)
        df = _read_sidecar(versions_path)
        if df is not None:
            for _, row in df.iterrows():
                version_id = str(row.get("version_id", "")).strip()
                if not version_id:
                    continue
                store.versions[version_id] = ReviewVersion(
                    version_id=version_id,
                    parent=str(row.get("parent") or NO_PARENT),
                    created_at=str(row.get("created_at", "")),
                    author=str(row.get("author", "")),
                    note=str(row.get("note", "")),
                    frozen=str(row.get("frozen", "")).strip().lower() in {"true", "1", "yes"},
                    exported_at=str(row.get("exported_at", "")),
                    exported_to=str(row.get("exported_to", "")),
                )
                store.decisions[version_id] = {}
        df = _read_sidecar(decisions_path)
        if df is not None:
            for _, row in df.iterrows():
                version_id = str(row.get("version_id", "")).strip()
                uid = str(row.get("uuid", "")).strip()
                if not version_id or not uid:
                    continue
                if version_id not in store.versions:
                    # A decisions file naming a version the manifest lost:
                    # keep the work, rebuild a placeholder version for it.
                    store.versions[version_id] = ReviewVersion(version_id=version_id, created_at=_now())
                    store.decisions[version_id] = {}
                store.decisions.setdefault(version_id, {})[uid] = Decision(
                    # Spelled as a hand edit or a spreadsheet may have left it:
                    # "Exclude " must still exclude, not count as some
                    # unknown status the overlay lets through.
                    uuid=uid, status=str(row.get("status", STATUS_OK)).strip().lower(), version_id=version_id,
                    new_label=str(row.get("new_label", "")), reason=str(row.get("reason", "")),
                    note=str(row.get("note", "")), source=str(row.get("source", "")),
                    batch_id=str(row.get("batch_id", "")), author=str(row.get("author", "")),
                    timestamp=str(row.get("timestamp", "")),
                )
        store.active_version_id = store.resume_version_id()
        store._dirty = False
        store._disk_signature = signature
        if store.versions:
            _log(f"review: loaded {sum(len(d) for d in store.decisions.values()):,} decisions across "
                 f"{len(store.versions)} version(s) from {decisions_path.name}")
        return store

    # ---- dataset round-trip ----
    def column_names(self, version_id: str, label_prefix: str = LABEL_PREFIX,
                     status_prefix: str = STATUS_PREFIX, reason_prefix: str = REASON_PREFIX) -> tuple:
        return f"{label_prefix}{version_id}", f"{status_prefix}{version_id}", f"{reason_prefix}{version_id}"

    def adopt_from_dataframe(self, df: pd.DataFrame, uuid_col: str, label_col: str) -> list:
        """Round-trip: pick up `label_review_*` / `review_status_*` columns
        found in a loaded dataset and turn them back into versions.

        This is what makes the enriched CSV a complete exchange format --
        load back the file you trained on and keep going from there, even
        if the sidecars were never copied along with it. Versions already
        present in the store are left alone (the sidecar is authoritative
        for those; it has provenance the columns can't carry).

        Returns the ids of the versions actually adopted.
        """
        adopted = []
        for column in df.columns:
            if not str(column).startswith(STATUS_PREFIX):
                continue
            version_id = str(column)[len(STATUS_PREFIX):]
            if not version_id or version_id in self.versions:
                continue
            label_col_v = f"{LABEL_PREFIX}{version_id}"
            reason_col_v = f"{REASON_PREFIX}{version_id}"
            self.versions[version_id] = ReviewVersion(
                version_id=version_id, parent=NO_PARENT, created_at=_now(),
                note=f"adopted from column '{column}'", frozen=True,
            )
            self.decisions[version_id] = {}
            statuses = df[column].astype(str)
            uuids = df[uuid_col].astype(str)
            labels = df[label_col_v].astype(str) if label_col_v in df.columns else df[label_col].astype(str)
            originals = df[label_col].astype(str)
            reasons = df[reason_col_v].astype(str) if reason_col_v in df.columns else None
            for i in range(len(df)):
                status = statuses.iat[i].strip()
                if status not in REVIEW_STATUSES:
                    continue
                new_label = labels.iat[i]
                self.decisions[version_id][uuids.iat[i]] = Decision(
                    uuid=uuids.iat[i], status=status, version_id=version_id,
                    new_label=new_label if status == STATUS_RELABEL and new_label != originals.iat[i] else "",
                    reason=reasons.iat[i] if reasons is not None else "",
                    source="adopted_from_dataset_csv", timestamp=_now(),
                )
            adopted.append(version_id)
            self._changed_versions.add(version_id)
            _log(f"review: adopted version '{version_id}' from dataset columns "
                 f"({len(self.decisions[version_id]):,} decisions)")
        if adopted:
            # Chain the adopted versions so a new pass inherits from the
            # most recent one rather than starting from nothing.
            previous = NO_PARENT
            for version_id in adopted:
                self.versions[version_id] = replace(self.versions[version_id], parent=previous)
                previous = version_id
            self._touch()
        return adopted

    def enrich(
        self, df: pd.DataFrame, uuid_col: str, label_col: str, version_ids: list,
        include_reason: bool = True, label_prefix: str = LABEL_PREFIX,
        status_prefix: str = STATUS_PREFIX, reason_prefix: str = REASON_PREFIX,
    ) -> tuple:
        """`df` plus two (or three) columns per requested version -- the
        export shape, and the thing the user's training code reads:

            label_review_v2   corrected label; the *original* label for a
                              row nobody reviewed, so the column is
                              complete and needs no fallback logic
            review_status_v2  unreviewed / ok / relabel / exclude / uncertain
            review_reason_v2  optional

        Training then filters `review_status_v2 != "exclude"` and takes
        `label_review_v2`. Exclusion deliberately lives in its own column
        rather than as a sentinel *value* of the label column: a sentinel
        could collide with a real class name (class labels come from the
        user's CSV and are never normalized), and it would throw away the
        label of the crop being excluded.

        Returns (enriched_df, notes) -- notes describes any column that had
        to be renamed to avoid overwriting something already in `df`.
        """
        from lookin.core.data_manager import _uuids_as_str

        out = df.copy()
        notes = []
        # Looked up the way the loaded dataset spelled them, which is how
        # every decision is keyed (see data_manager._uuids_as_str).
        uuids = _uuids_as_str(out[uuid_col]).astype(str)
        originals = out[label_col]

        for version_id in version_ids:
            resolved = self.effective(version_id)
            label_name, status_name, reason_name = self.column_names(
                version_id, label_prefix, status_prefix, reason_prefix,
            )
            wanted = [label_name, status_name] + ([reason_name] if include_reason else [])
            renamed = {}
            for name in wanted:
                final = name
                suffix = 2
                while final in out.columns:
                    final = f"{name}_{suffix}"
                    suffix += 1
                if final != name:
                    notes.append(f"'{name}' already exists in the dataset -- wrote '{final}' instead.")
                renamed[name] = final

            statuses, labels, reasons = [], [], []
            for i in range(len(out)):
                decision = resolved.get(uuids.iat[i])
                if decision is None:
                    statuses.append(STATUS_UNREVIEWED)
                    labels.append(originals.iat[i])
                    reasons.append("")
                else:
                    statuses.append(decision.status)
                    labels.append(decision.effective_label(originals.iat[i]))
                    reasons.append(decision.reason)
            out[renamed[label_name]] = labels
            out[renamed[status_name]] = statuses
            if include_reason:
                out[renamed[reason_name]] = reasons
        return out, notes

    def uuids_by_status(self, status: str, version_id: Optional[str] = None) -> list:
        return [uid for uid, d in self.effective(version_id).items() if d.status == status]

    def overlay(self, uuids, original_labels, version_id: Optional[str] = None) -> tuple:
        """Vectorized (effective_label, status) for an array of uuids --
        the in-app counterpart to enrich(), used to recompute metrics
        through the review overlay without materializing an export."""
        resolved = self.effective(version_id)
        labels, statuses = [], []
        for uid, original in zip(uuids, original_labels):
            decision = resolved.get(str(uid))
            if decision is None:
                labels.append(original)
                statuses.append(STATUS_UNREVIEWED)
            else:
                labels.append(decision.effective_label(original))
                statuses.append(decision.status)
        return labels, statuses


# ---- comparing two versions --------------------------------------------

@dataclass
class VersionDiff:
    """What changed between two review versions.

    Asked when a cleaning pass makes the model *worse*: v3 inherits from v2,
    the metrics dropped, and the only way to find out why is to see which
    decisions differ. Comparing resolved states rather than stored ones, so
    an inherited decision counts as present -- what matters is the effective
    verdict a training run would read, not which version happens to hold it.
    """
    version_a: str
    version_b: str
    added: list = field(default_factory=list)      # (uuid, decision_b)
    removed: list = field(default_factory=list)    # (uuid, decision_a)
    changed: list = field(default_factory=list)    # (uuid, decision_a, decision_b)
    unchanged: int = 0

    def total_changes(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)

    def transitions(self) -> dict:
        """(status_a, status_b) -> count, with STATUS_UNREVIEWED standing in
        for "no decision on that side" -- which is what an added or removed
        decision is."""
        out: dict = {}
        for _, decision in self.added:
            key = (STATUS_UNREVIEWED, decision.status)
            out[key] = out.get(key, 0) + 1
        for _, decision in self.removed:
            key = (decision.status, STATUS_UNREVIEWED)
            out[key] = out.get(key, 0) + 1
        for _, a, b in self.changed:
            key = (a.status, b.status)
            out[key] = out.get(key, 0) + 1
        return out


def _same_verdict(a: Decision, b: Decision) -> bool:
    """Equal where it counts: the status and, for a relabel, the class. The
    reason, note, author and timestamp differ on any re-decision and would
    make every touched crop look changed."""
    return a.status == b.status and a.new_label == b.new_label


def diff_versions(store: "ReviewStore", version_a: str, version_b: str) -> VersionDiff:
    resolved_a = store.effective(version_a)
    resolved_b = store.effective(version_b)
    diff = VersionDiff(version_a=version_a, version_b=version_b)

    for uid, decision_b in sorted(resolved_b.items()):
        decision_a = resolved_a.get(uid)
        if decision_a is None:
            diff.added.append((uid, decision_b))
        elif _same_verdict(decision_a, decision_b):
            diff.unchanged += 1
        else:
            diff.changed.append((uid, decision_a, decision_b))
    for uid, decision_a in sorted(resolved_a.items()):
        if uid not in resolved_b:
            diff.removed.append((uid, decision_a))
    return diff


# ---- exports ------------------------------------------------------------

class DestinationRefused(ValueError):
    """The export was aimed at a file it must not write over."""


def protected_paths_for(dataset_csv_path) -> dict:
    """The files an export must never write over, and what each one is.

    Three of them, and losing any is worse than not exporting at all:

    * the dataset CSV itself. It is an *input* -- the whole design keeps
      decisions out of it precisely so it is never rewritten (see this
      module's opening note), and it may well be read-only, shared, or the
      only copy.
    * the two review sidecars. Writing an enriched dataset over the
      decisions file destroys every judgement ever recorded against this
      CSV -- and it is one letter away from the default export name, since
      'X_review.csv' holds the decisions and 'X_reviewed.csv' is the
      export.
    """
    source = Path(dataset_csv_path).expanduser().resolve()
    decisions, versions = review_paths_for_csv(source)
    return {
        source: "the dataset CSV this export was made from",
        decisions.resolve(): "the file holding every review decision for this dataset",
        versions.resolve(): "the review version manifest for this dataset",
    }


def check_export_destination(dest_path, dataset_csv_path) -> Path:
    """Resolve `dest_path` and refuse it if it names a protected file.

    Resolved first, so '../data/dataset.csv', a relative path and a symlink
    all compare equal to the real thing rather than slipping past a string
    comparison.
    """
    dest = Path(dest_path).expanduser().resolve()
    protected = protected_paths_for(dataset_csv_path)
    what = protected.get(dest)
    if what is not None:
        raise DestinationRefused(
            f"Refusing to write the export to {dest.name}: that is {what}. "
            f"Pick another name -- an export is a new file, never a replacement "
            f"for what it was made from."
        )
    return dest



@dataclass
class ExportResult:
    path: Path
    rows: int
    columns: list
    excluded: int
    relabelled: int
    notes: list


def export_enriched_dataset(
    store: ReviewStore,
    dataset_csv_path,
    uuid_col: str,
    label_col: str,
    version_ids: list,
    dest_path,
    include_reason: bool = True,
    drop_excluded: bool = False,
    freeze: bool = True,
    label_prefix: str = LABEL_PREFIX,
    status_prefix: str = STATUS_PREFIX,
    reason_prefix: str = REASON_PREFIX,
) -> ExportResult:
    """Write the user's own dataset CSV back out, unchanged, plus the review
    columns for each requested version.

    The source CSV is re-read here *in full* rather than taken from the
    loaded DataFrame, and that is the whole point: the app is usually
    holding one split (the Sources tab's row filter), so exporting the loaded
    frame would hand back a training file with every other split silently
    missing. Decisions are uuid-keyed, so applying them to the complete
    file costs nothing.

    `drop_excluded` is off by default: shipping every row and letting the
    training code filter `review_status_* != "exclude"` keeps the export
    round-trippable (see ReviewStore.adopt_from_dataframe) and keeps the
    excluded crops inspectable. Turning it on produces a ready-to-train
    file at the cost of both.

    Exporting freezes the versions it wrote, so the run that consumes this
    file stays identifiable afterwards.
    """
    # Checked before a single row is read: the failure this prevents is
    # silent and total, so it must happen before anything else can.
    dest_path = check_export_destination(dest_path, dataset_csv_path)
    t0 = time.monotonic()
    # Every cell as the text it is in the file: "the user's own CSV back
    # out, unchanged" means an id '0012' stays '0012', a '1.50' stays
    # '1.50' and an empty cell stays empty, rather than whatever pandas'
    # type guessing turns them into on the way through.
    df = pd.read_csv(dataset_csv_path, dtype=str, keep_default_na=False)
    _log(f"export: re-read {len(df):,} rows from {dataset_csv_path} ({time.monotonic() - t0:.1f}s)")

    for required in (uuid_col, label_col):
        if required not in df.columns:
            raise ValueError(f"Column '{required}' not found in {dataset_csv_path}. Available: {list(df.columns)}")

    enriched, notes = store.enrich(
        df, uuid_col, label_col, version_ids, include_reason=include_reason,
        label_prefix=label_prefix, status_prefix=status_prefix, reason_prefix=reason_prefix,
    )

    excluded = relabelled = 0
    for version_id in version_ids:
        counts = store.counts(version_id)
        excluded = max(excluded, counts.get(STATUS_EXCLUDE, 0))
        relabelled = max(relabelled, counts.get(STATUS_RELABEL, 0))

    if drop_excluded and version_ids:
        status_column = f"{status_prefix}{version_ids[-1]}"
        if status_column in enriched.columns:
            before = len(enriched)
            enriched = enriched[enriched[status_column] != STATUS_EXCLUDE].reset_index(drop=True)
            notes.append(f"Dropped {before - len(enriched):,} row(s) excluded by '{version_ids[-1]}'.")

    fileio.atomic_write(dest_path, lambda tmp: enriched.to_csv(tmp, index=False))
    _log(f"export: wrote {len(enriched):,} rows to {dest_path}")

    if freeze:
        for version_id in version_ids:
            store.note_export(version_id, str(dest_path))
        store.save()

    added = [c for c in enriched.columns if c not in df.columns]
    return ExportResult(
        path=dest_path, rows=len(enriched), columns=added,
        excluded=excluded, relabelled=relabelled, notes=notes,
    )


def export_excluded_uuids(store: ReviewStore, version_id: str, dest_path) -> ExportResult:
    """A plain newline-separated list of the uuids excluded at a version --
    for a pipeline that would rather not parse a CSV to drop a few thousand
    crops."""
    dest_path = Path(dest_path).expanduser().resolve()
    # This one does not know the dataset, but it does know its own store's
    # files, and those are the ones whose loss is unrecoverable.
    for path, what in (
        (store.decisions_path, "the file holding every review decision for this dataset"),
        (store.versions_path, "the review version manifest for this dataset"),
    ):
        if path is not None and Path(path).expanduser().resolve() == dest_path:
            raise DestinationRefused(
                f"Refusing to write the export to {dest_path.name}: that is {what}."
            )
    uuids = sorted(store.uuids_by_status(STATUS_EXCLUDE, version_id))
    fileio.atomic_write_text(dest_path, "\n".join(uuids) + ("\n" if uuids else ""))
    _log(f"export: wrote {len(uuids):,} excluded uuids to {dest_path}")
    return ExportResult(path=dest_path, rows=len(uuids), columns=[], excluded=len(uuids), relabelled=0, notes=[])
