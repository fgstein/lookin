"""Is what is on screen saved anywhere, and under what name.

Two questions, not one, because they have different answers and different
costs to get wrong:

    Is this data saved as a workspace?    -- the dataset, its mapping and its
                                          experiments. Lost by loading
                                          something else.
    Is this state saved as a snapshot? -- the tab, the filters, the
                                          projection settings. Lost by
                                          changing them.

The top bar shows both, so "I will just open this other workspace" is never a
silent way to lose an afternoon's filters.

How "saved" is decided
----------------------
A workspace is saved when one was opened or written in this session, its
directory is still there, and the mount on screen still matches the one it
holds. Loading a different dataset therefore un-saves it, which is exactly
what the user needs to be told.

A snapshot is saved when one was opened or written in this session and the
state captured right now is byte-for-byte the state it holds. That
comparison is the honest test -- anything weaker (a dirty flag each view
has to remember to set) is a flag that will be missed somewhere and will
then claim work is safe when it is not.

Cost: a comparison captures the view state, which reads control values and
allocates a few small dicts. It is cheap enough to do on every refresh, and
refreshes are driven by data changes and tab switches, not by keystrokes.
"""
from __future__ import annotations

from dataclasses import dataclass

from lookin.core.library import mount_to_dict

# Fields of a mount that move on their own and say nothing about whether the
# user's work is saved: when the file was last stat'ed, and how big it was.
VOLATILE_MOUNT_FIELDS = ("saved_at", "dataset_size", "dataset_mtime")


@dataclass
class SaveStatus:
    workspace_name: str = ""
    workspace_saved: bool = False
    snapshot_name: str = ""
    snapshot_saved: bool = False
    has_data: bool = False
    # Has the state moved since the last save or load? Tracked separately
    # from snapshot_saved because the two answer different questions: a
    # session that has never had a snapshot is *not* saved, but if nothing
    # has been touched since it was loaded there is also nothing to lose.
    view_changed: bool = False

    def anything_unsaved(self) -> bool:
        """True when loading something else would throw away work.

        Deliberately not "not saved". Warning every time a snapshot happens
        not to exist would fire hardest right after a workspace is saved,
        which is the one moment nothing is at risk -- and a warning that
        cries wolf is one nobody reads. So: the data is not in a workspace, or
        the settings have actually been changed since the last save or load.
        """
        return self.has_data and (not self.workspace_saved or self.view_changed)


def _comparable_mount(mount) -> dict:
    if mount is None:
        return {}
    payload = mount_to_dict(mount)
    for field in VOLATILE_MOUNT_FIELDS:
        payload.pop(field, None)
    return payload


def same_mount(a, b) -> bool:
    """Equal in everything that says whether the user's work is saved."""
    return _comparable_mount(a) == _comparable_mount(b)


def compute(session) -> SaveStatus:
    """What is saved right now. Never raises: this drives an indicator, and
    an indicator that can take the app down is worse than no indicator."""
    status = SaveStatus()
    try:
        upload_view = getattr(session, "upload_view", None)
        current_mount = upload_view.current_mount() if upload_view is not None else None
    except Exception:  # noqa: BLE001
        current_mount = None
    status.has_data = current_mount is not None

    workspace = getattr(session, "current_workspace", None)
    if workspace is not None:
        status.workspace_name = workspace.label()
        try:
            status.workspace_saved = bool(
                current_mount is not None
                and workspace.exists()
                and _comparable_mount(workspace.mount) == _comparable_mount(current_mount)
            )
        except Exception:  # noqa: BLE001
            status.workspace_saved = False

    try:
        current_view = session.view_state.capture()
    except Exception:  # noqa: BLE001
        current_view = None

    snapshot = getattr(session, "current_snapshot", None)
    if snapshot is not None:
        status.snapshot_name = snapshot.label()
        status.snapshot_saved = current_view is not None and snapshot.view_state == current_view

    baseline = getattr(session, "view_state_baseline", None)
    status.view_changed = bool(
        status.has_data and current_view is not None
        and baseline is not None and baseline != current_view
    )
    return status


def mark_baseline(session) -> None:
    """Remember the state as "nothing to lose from here".

    Called after a save and after a load. Cheap, and the alternative -- a
    dirty flag every view has to remember to set -- is a flag that will be
    missed somewhere and will then claim work is safe when it is not.
    """
    try:
        session.view_state_baseline = session.view_state.capture()
    except Exception:  # noqa: BLE001
        session.view_state_baseline = None


# Longer than this and the top bar starts pushing the tabs around.
MAX_NAME_CHARS = 22


def elide(name: str) -> str:
    return name if len(name) <= MAX_NAME_CHARS else name[:MAX_NAME_CHARS - 1] + "…"
