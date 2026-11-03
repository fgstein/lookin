"""Experiment groups: several experiments drawn as one in every chart.

The usual case is the checkpoints of one training run -- ``run-A/ckpt_10``,
``run-A/ckpt_20``, ``run-A/ckpt_30`` -- that the user wants to see as "run A"
next to "run B", without three colors and three legend entries per run.

What a group changes, and what it does not
------------------------------------------
A group is a *display* layer, nothing else:

- in a chart, its members take the group's color and the group's name, and
  the legend has one entry for the group (see chart_style.legend_below);
- the curves, bars and points themselves do not move: every member is still
  computed and drawn exactly as it was, only painted and named differently
  (tests/test_experiment_groups.py checks the drawn data is identical);
- tables, pickers, dropdowns, the MCP tools and the snapshots keep naming
  each experiment by its own name, which stays unique -- a group never
  renames its members.

Showing the groups is one switch (`ExperimentGroups.shown`): off, every
chart is exactly what it would be with no group at all, and nothing about
the groups is lost.

Where a group lives
-------------------
In a workspace's manifest: ``Mount.experiment_groups`` holds each group's
id, name and color, ``MountExperiment.group`` the id of the group an
experiment belongs to, and ``Mount.show_experiment_groups`` the switch.
A group is keyed by an id rather than its name, so renaming it touches no
member. In a session the same three things live on AppState.groups.

Rules
-----
- A group name is never empty and is unique among the groups.
- A group with no member left disappears (`assign`, `forget_experiment`).
- A member's own color is kept, not erased: it comes back as soon as the
  experiment leaves the group or the groups are hidden.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Iterable, Optional

_HEX_COLOR = re.compile(r"#?([0-9a-fA-F]{6})")
GROUP_ID_PREFIX = "eg-"


def normalize_color(value) -> Optional[str]:
    """"#rrggbb" (lower case) for a color, "" for none, None for anything
    else -- a color arrives from a browser's color input or a hand-edited
    manifest, and ends up in an HTML attribute and in matplotlib."""
    text = str(value or "").strip()
    if not text:
        return ""
    m = _HEX_COLOR.fullmatch(text)
    return f"#{m.group(1).lower()}" if m else None


def new_group_id(taken: Iterable[str] = ()) -> str:
    taken = set(taken)
    while True:
        gid = GROUP_ID_PREFIX + secrets.token_hex(3)
        if gid not in taken:
            return gid


@dataclass
class ExperimentGroup:
    id: str
    name: str
    color: str = ""      # "#rrggbb", or "" for a palette color

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "color": self.color}


class ExperimentGroups:
    """The groups of one set of experiments, and which experiment is in
    which. Experiments are referred to by name; `rename_experiment` and
    `forget_experiment` keep the membership in step with the experiments."""

    def __init__(self):
        self.groups: dict = {}        # id -> ExperimentGroup, in creation order
        self.member_of: dict = {}     # experiment name -> group id
        self.shown: bool = True

    # ---- reading ----

    def get(self, gid: str) -> Optional[ExperimentGroup]:
        return self.groups.get(gid or "")

    def by_name(self, name: str) -> Optional[ExperimentGroup]:
        name = (name or "").strip()
        return next((g for g in self.groups.values() if g.name == name), None)

    def resolve(self, key: str) -> ExperimentGroup:
        """A group by id, or else by name. Raises ValueError when there is
        none -- what a tool call or a form needs, since guessing would put
        experiments in the wrong group."""
        group = self.get(key) or self.by_name(key)
        if group is None:
            known = ", ".join(repr(g.name) for g in self.groups.values()) or "none"
            raise ValueError(f"No experiment group {key!r}. Groups: {known}.")
        return group

    def group_of(self, name: str) -> Optional[ExperimentGroup]:
        """The group `name` belongs to, shown or not."""
        return self.groups.get(self.member_of.get(name, ""))

    def shown_group_of(self, name: str) -> Optional[ExperimentGroup]:
        """The group `name` is drawn as: its group while the groups are
        shown, else None."""
        return self.group_of(name) if self.shown else None

    def members(self, gid: str, order: Iterable[str]) -> list:
        """The members of `gid`, in the order of `order` (the experiment
        order of the session or the mount)."""
        return [n for n in order if self.member_of.get(n) == gid]

    def in_use(self, order: Iterable[str]) -> list:
        """The groups with at least one member among `order`, ordered by
        their first member."""
        out = []
        for name in order:
            group = self.group_of(name)
            if group is not None and group not in out:
                out.append(group)
        return out

    def chart_labels(self, names: Iterable[str]) -> Optional[dict]:
        """name -> what a chart calls it: its group's name for a member of a
        shown group, its own name otherwise. None when none of `names` is
        drawn as a group, which is what tells a chart to draw exactly as it
        always did (see charts.py)."""
        names = list(names)
        labels = {}
        grouped = False
        for name in names:
            group = self.shown_group_of(name)
            grouped = grouped or group is not None
            labels[name] = group.name if group is not None else name
        return labels if grouped else None

    # ---- changing ----

    def _check_name(self, name: str, gid: str = "") -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("An experiment group needs a name.")
        other = self.by_name(name)
        if other is not None and other.id != gid:
            raise ValueError(f"There is already an experiment group named {name!r}.")
        return name

    def unique_name(self, base: str) -> str:
        """`base`, or `base (2)`, `base (3)`... -- the first no group has."""
        base = (base or "").strip() or "group"
        if self.by_name(base) is None:
            return base
        n = 2
        while self.by_name(f"{base} ({n})") is not None:
            n += 1
        return f"{base} ({n})"

    def create(self, name: str, members: Iterable[str] = (), color: str = "",
               gid: str = "") -> ExperimentGroup:
        """A new group holding `members` (taken out of any group they were
        in). Raises ValueError on an empty or taken name, or a bad color."""
        name = self._check_name(name)
        clean = normalize_color(color)
        if clean is None:
            raise ValueError(f"A group color must be '#rrggbb' or empty; got {color!r}.")
        gid = gid or new_group_id(self.groups)
        group = ExperimentGroup(gid, name, clean)
        self.groups[gid] = group
        for member in members:
            self.assign(member, gid)
        return group

    def rename(self, gid: str, name: str) -> bool:
        group = self.get(gid)
        if group is None:
            raise ValueError(f"No experiment group {gid!r}.")
        name = self._check_name(name, gid)
        if name == group.name:
            return False
        group.name = name
        return True

    def set_color(self, gid: str, color: str) -> bool:
        group = self.get(gid)
        if group is None:
            raise ValueError(f"No experiment group {gid!r}.")
        clean = normalize_color(color)
        if clean is None:
            raise ValueError(f"A group color must be '#rrggbb' or empty; got {color!r}.")
        if clean == group.color:
            return False
        group.color = clean
        return True

    def assign(self, name: str, gid: str) -> bool:
        """Put experiment `name` in group `gid`, or in none with "". A group
        left with no member is dropped. Returns whether anything changed."""
        gid = gid or ""
        if gid and gid not in self.groups:
            raise ValueError(f"No experiment group {gid!r}.")
        before = self.member_of.get(name, "")
        if before == gid:
            return False
        if gid:
            self.member_of[name] = gid
        else:
            self.member_of.pop(name, None)
        self._drop_if_empty(before)
        return True

    def delete(self, gid: str) -> bool:
        """Remove the group; its members stay, ungrouped."""
        if gid not in self.groups:
            return False
        del self.groups[gid]
        self.member_of = {n: g for n, g in self.member_of.items() if g != gid}
        return True

    def forget_experiment(self, name: str) -> None:
        gid = self.member_of.pop(name, "")
        self._drop_if_empty(gid)

    def rename_experiment(self, old: str, new: str) -> None:
        if old in self.member_of:
            self.member_of[new] = self.member_of.pop(old)

    def set_shown(self, shown: bool) -> bool:
        shown = bool(shown)
        if shown == self.shown:
            return False
        self.shown = shown
        return True

    def clear(self) -> None:
        self.groups.clear()
        self.member_of.clear()
        self.shown = True

    def _drop_if_empty(self, gid: str) -> None:
        if gid and gid in self.groups and gid not in self.member_of.values():
            del self.groups[gid]

    # ---- to and from a mount ----

    def adopt(self, definitions, members: dict, shown: bool = True) -> None:
        """Replace everything with what a mount holds: `definitions` as
        stored in Mount.experiment_groups, `members` name -> group id.
        Tolerant like the rest of the manifest: a definition that cannot be
        read is skipped, and so is a membership naming no known group."""
        self.clear()
        self.shown = bool(shown)
        self.add_definitions(definitions)
        for name, gid in (members or {}).items():
            if gid in self.groups:
                self.member_of[name] = gid

    def add_definitions(self, definitions) -> None:
        """Learn the groups in `definitions` that are not known yet (by id),
        keeping the ones already here as they are. A name already taken by
        another group gets a suffix rather than being refused: this reads a
        file, and must not lose a group over it."""
        for raw in definitions or []:
            if not isinstance(raw, dict):
                continue
            gid = str(raw.get("id") or "").strip()
            name = str(raw.get("name") or "").strip()
            if not gid or not name or gid in self.groups:
                continue
            color = normalize_color(raw.get("color")) or ""
            self.groups[gid] = ExperimentGroup(gid, self.unique_name(name), color)

    def definitions(self, order: Iterable[str]) -> list:
        """The groups as Mount.experiment_groups stores them: those with a
        member among `order`, in the order of their first member."""
        return [g.to_dict() for g in self.in_use(order)]


def from_mount(mount) -> ExperimentGroups:
    groups = ExperimentGroups()
    groups.adopt(getattr(mount, "experiment_groups", None) or [],
                 {e.name: e.group for e in mount.experiments if getattr(e, "group", "")},
                 getattr(mount, "show_experiment_groups", True))
    return groups


def write_to_mount(groups: ExperimentGroups, mount) -> None:
    """Store `groups` back into `mount`: the definitions in use, each
    experiment's group id, and the switch."""
    names = [e.name for e in mount.experiments]
    for e in mount.experiments:
        group = groups.group_of(e.name)
        e.group = group.id if group is not None else ""
    mount.experiment_groups = groups.definitions(names)
    mount.show_experiment_groups = bool(groups.shown)


# ---- grouping experiments as they are added from MLflow ----------------------

# How the runs dialog groups what it adds. Keys, compared in code and stored
# in home_view.json ("runs": {"grouping": ...}).
GROUPING_BY_RUN = "by_run"   # the checkpoints of one run together (default)
GROUPING_NONE = "none"       # no group
GROUPING_ONE = "one"         # everything added at once, in one new group
GROUPINGS = (GROUPING_BY_RUN, GROUPING_NONE, GROUPING_ONE)


def group_added(groups: ExperimentGroups, added: list, grouping: str, name: str = "") -> list:
    """Group experiments just added, as the runs dialog was asked to.

    `added` is a list of (experiment name, run id, run label). With
    GROUPING_BY_RUN, the experiments of one run *added together* make one
    group named after the run -- only when there are at least two of them,
    and never by joining a group that already exists: an experiment added
    later stays ungrouped until the user puts it in a group. With
    GROUPING_ONE, all of `added` go into one new group called `name` (the
    first run's label when empty). Returns the groups created."""
    created = []
    if grouping == GROUPING_ONE and added:
        base = (name or "").strip() or added[0][2] or "group"
        created.append(groups.create(groups.unique_name(base), [n for n, _, _ in added]))
    elif grouping == GROUPING_BY_RUN:
        by_run: dict = {}
        for exp_name, run_id, run_label in added:
            by_run.setdefault(run_id, (run_label, []))[1].append(exp_name)
        for run_id, (run_label, names) in by_run.items():
            if run_id and len(names) >= 2:
                created.append(groups.create(groups.unique_name(run_label or run_id[:8]), names))
    return created
