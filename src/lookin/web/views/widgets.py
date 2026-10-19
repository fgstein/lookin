"""Controls several tabs share: picking classes among thousands.

Same behaviour as lookin.ui.class_matrix:

* class_select -- a single-class picker, type-to-filter past FOCUS_MAX
  options (a native list over two thousand labels is not a way to find one);
* ClassPicker -- a search field over class labels plus the picked ones as
  chips, at most `limit` of them;
* ClassFocus -- which classes a tab's per-class views show once there are
  more than FOCUS_MAX: a criterion, or picked by hand.

These are components, not handlers: each is given the owner's target and
an action name, declares its controls as `<target>:<action>` with an `op`
argument, and the owner's `on_<action>(op, **args)` calls `handle(op, args)`. Class labels are never put into a key or an id -- they
travel as values in the action arguments.
"""
from __future__ import annotations

import secrets
from typing import Callable

from lookin.core.class_focus import FOCUS_MANUAL, FOCUS_MAX, MIN_SUPPORT_DEFAULT
from lookin.web import html as h

# How many matching labels the picker lists under the search field.
PICKER_SUGGESTIONS = 12


def class_select(labels, value, change: str, label: str = "", cls: str = "", title: str = "",
                 width: int = 0) -> str:
    """A dropdown over class labels; past FOCUS_MAX of them a text field
    with the labels as completions (the browser filters them as you type).
    Key and text are both the label itself, as in review_widgets.class_options."""
    labels = [str(l) for l in labels]
    if len(labels) <= FOCUS_MAX:
        return h.select([(l, l) for l in labels], "" if value is None else str(value), change=change,
                        label=label, cls=cls, title=title)
    list_id = "cl" + secrets.token_hex(4)
    options = "".join(f'<option value="{h.e(l)}">' for l in labels)
    style = f' style="width:{width}px"' if width else ""
    field = (f'<input class="inp {h.e(cls)}" list="{list_id}" value="{h.e("" if value is None else value)}" '
             f'placeholder="Type part of a class name"{style}{h.attrs(title=title or None)}{change}>'
             f'<datalist id="{list_id}">{options}</datalist>')
    if label:
        return f'<label class="fld"><span class="fld-l">{h.e(label)}</span>{field}</label>'
    return field


class ClassPicker:
    """`on_change(labels)` fires with the new list on every add or removal."""

    def __init__(self, target: str, action: str, on_change: Callable[[list], None],
                 limit=FOCUS_MAX, label: str = "Find a class"):
        self.target = target
        self.action = action
        self.on_change = on_change
        self.limit = limit
        self.label = label
        self.labels: list = []
        self.picked: list = []
        self.query = ""
        self.message = ""

    def set_labels(self, labels) -> None:
        self.labels = list(labels)

    def set_picked(self, picked) -> None:
        self.picked = list(picked)

    def _a(self, op: str, **kw) -> str:
        return h.act(self.target, self.action, op=op, **kw)

    def matches(self) -> tuple:
        query = self.query.strip().lower()
        if not query:
            return [], ""
        found = [l for l in self.labels if query in str(l).lower()]
        # Exact, then prefix, then anywhere: among two thousand numeric
        # labels "21" would otherwise sit behind "1021", "1121", ...
        found.sort(key=lambda l: (str(l).lower() != query, not str(l).lower().startswith(query)))
        matches = [l for l in found if l not in self.picked]
        if not found:
            return [], "No class matches."
        if len(matches) > PICKER_SUGGESTIONS:
            return matches[:PICKER_SUGGESTIONS], f"{len(matches):,} classes match -- showing the first {PICKER_SUGGESTIONS}."
        return matches, ""

    def render(self) -> str:
        matches, note = self.matches()
        search = h.text_input(self.query, placeholder="Type part of a class name", label=self.label,
                              input_=h.on_input(self.target, self.action, op="search"), width=260)
        sugg = "".join(h.button(str(l), act_attr=self._a("add", label=str(l)), icon_name="plus", small=True)
                       for l in matches)
        msg = self.message or note
        chips = "".join(
            f'<span class="chip chip-x">{h.e(l)}<button type="button" class="chip-del" title="Remove"'
            f'{self._a("remove", label=str(l))}>{h.icon("x", 11)}</button></span>' for l in self.picked)
        if self.picked:
            chips += h.button("Clear", act_attr=self._a("clear"), kind="link", small=True)
        return (f'<div class="picker">{search}'
                f'{f"<div class=row-wrap>{sugg}</div>" if sugg else ""}'
                f'{f"<div class=\"muted small\">{h.e(msg)}</div>" if msg else ""}'
                f'{f"<div class=row-wrap>{chips}</div>" if chips else ""}</div>')

    def handle(self, op: str, args: dict) -> None:
        if op == "search":
            self.query = str(args.get("value", ""))
            self.message = ""
            return
        label = args.get("label")
        if op == "add":
            label = self._resolve(label)
            if label is None:
                return
            if self.limit is not None and len(self.picked) >= self.limit:
                self.message = f"At most {self.limit} classes -- remove one first."
                return
            if label not in self.picked:
                self.picked.append(label)
        elif op == "remove":
            self.picked = [l for l in self.picked if str(l) != str(label)]
        elif op == "clear":
            self.picked = []
        else:
            return
        self.message = ""
        self.on_change(list(self.picked))

    def _resolve(self, label):
        """The label object the action names (labels may be ints in the CSV)."""
        for l in self.labels:
            if str(l) == str(label):
                return l
        return None


class ClassFocus:
    """Which classes a tab's per-class views show once there are too many.
    See lookin.ui.class_matrix.ClassFocus: same criteria, same snapshot
    shape (`criterion`, `classes`, `min_support`)."""

    def __init__(self, target: str, action: str, criteria: tuple,
                 on_change: Callable[[], None], default=None, limit: int = FOCUS_MAX,
                 what: str = "the views below", support_criteria: tuple = (), support_noun: str = "crops"):
        self.target = target
        self.action = action
        self.criteria = tuple(criteria)
        self.limit = limit
        self.what = what
        self.on_change = on_change
        self.criterion = default or self.criteria[0][0]
        self.classes: list = []
        self.many = False
        self.n_labels = 0
        self.support_criteria = frozenset(support_criteria)
        self.support_noun = support_noun
        self.min_support = MIN_SUPPORT_DEFAULT
        self.picker = ClassPicker(target, action, self._on_picked, limit=limit)

    def resolve(self, labels, rank: Callable[[str], list]) -> list:
        labels = list(labels)
        self.n_labels = len(labels)
        self.many = len(labels) > self.limit
        if not self.many:
            return labels
        if self.criterion == FOCUS_MANUAL:
            present = set(labels)
            focus = [c for c in self.classes if c in present]
        else:
            focus = list(rank(self.criterion))[: self.limit]
        self.classes = focus
        self.picker.set_labels(labels)
        self.picker.set_picked(focus)
        return focus

    def min_support_for(self, criterion: str) -> int:
        return self.min_support if criterion in self.support_criteria else 0

    def hide(self) -> None:
        self.many = False

    def render(self) -> str:
        if not self.many:
            return ""
        note = (f"{self.n_labels:,} classes: {len(self.classes)} of them are shown in {self.what} "
                f"(at most {self.limit}). Pick them by a criterion, or search and add them by hand.")
        crit = h.select(list(self.criteria), self.criterion, change=self._c("criterion"), label="Classes shown")
        support = ""
        if self.criterion in self.support_criteria:
            support = h.text_input(str(self.min_support), label=f"At least N {self.support_noun}", width=110,
                                   change=self._c("min_support"),
                                   title=f"Only classes with at least this many {self.support_noun} are ranked by this criterion")
        return (f'<div class="card"><div class="card-b"><div class="muted small">{h.e(note)}</div>'
                f'<div class="row-wrap" style="align-items:flex-start;margin-top:6px">{crit}{support}'
                f'{self.picker.render()}</div></div></div>')

    def _c(self, op: str) -> str:
        return h.on_change(self.target, self.action, op=op)

    def handle(self, op: str, args: dict) -> None:
        """Criterion and minimum support here; search/add/remove/clear go to
        the picker (both share the owner's action)."""
        if op == "criterion":
            value = args.get("value")
            self.criterion = value if value in {k for k, _ in self.criteria} else self.criteria[0][0]
            self.on_change()
        elif op == "min_support":
            try:
                value = max(0, int(str(args.get("value", "")).strip()))
            except ValueError:
                return
            if value != self.min_support:
                self.min_support = value
                self.on_change()
        else:
            self.picker.handle(op, args)

    def _on_picked(self, picked: list) -> None:
        self.criterion = FOCUS_MANUAL
        self.classes = list(picked)
        self.on_change()

    # ---- snapshots ----
    def view_state(self) -> dict:
        return {"criterion": self.criterion, "classes": [str(c) for c in self.classes],
                "min_support": self.min_support}

    def restore_view_state(self, state) -> None:
        if not isinstance(state, dict):
            return
        if state.get("criterion") in {key for key, _ in self.criteria}:
            self.criterion = state["criterion"]
        classes = state.get("classes")
        if isinstance(classes, list):
            self.classes = [c for c in classes if isinstance(c, str)][: self.limit]
        min_support = state.get("min_support")
        if isinstance(min_support, int) and min_support >= 0:
            self.min_support = min_support


def experiment_color_menu(target: str, name: str, color: str, custom: bool, size: int = 10) -> str:
    """An experiment's color dot that opens its color menu: ten preset
    swatches, "Other color..." (the browser's own picker -- nothing
    fetched), and "Automatic" to go back to its palette color. Each choice
    runs `target.on_color(name=, value=)`, value "" for Automatic."""
    from lookin.ui.components import EXPERIMENT_PRESET_COLORS

    swatches = "".join(
        f'<button type="button" class="swatch{" on" if c == color else ""}" style="background:{c}"'
        f'{h.act(target, "color", name=name, value=c)} title="{c}" aria-label="Color {c}"></button>'
        for c in EXPERIMENT_PRESET_COLORS)
    picker = (f'<label class="swatch-other">{h.icon("sliders", 14)}<span>Other color…</span>'
              f'<input type="color" value="{h.e(color)}"{h.on_change(target, "color", name=name)}></label>')
    auto = (f'<hr><button type="button"{h.act(target, "color", name=name, value="")}>'
            f'{h.icon("refresh", 14)}<span>Automatic</span></button>') if custom else ""
    return (f'<span class="menu-wrap"><button type="button" class="color-btn" data-menu="1" '
            f'title="Color of {h.e(name)} -- click to change" aria-label="Color of {h.e(name)}">{h.dot(color, size=size)}</button>'
            f'<div class="menu menu-fixed color-menu"><div class="swatches">{swatches}</div>{picker}{auto}</div></span>')
