"""What every view and dialog of the web UI is made of.

A view is one tab (or Home): it owns the state of what the user chose
there, renders itself to HTML, and handles the actions its HTML declares
(`data-act="<name>:<action>"` -> `on_<action>(**args)`).

A dialog is the same thing in a modal layer, addressed as `dlg:<id>`.
"""
from __future__ import annotations

import secrets
import traceback
from typing import Callable, Optional

from lookin.web import html as h


class Handler:
    target = ""

    def handle(self, action: str, args: dict):
        fn = getattr(self, f"on_{action}", None)
        if fn is None or not callable(fn):
            raise KeyError(f"{self.target}: no action '{action}'")
        return fn(**(args or {}))


class View(Handler):
    name = ""
    title = ""

    def __init__(self, session):
        self.s = session

    @property
    def target(self) -> str:   # type: ignore[override]
        return self.name

    @property
    def state(self):
        return self.s.state

    def render(self) -> str:
        raise NotImplementedError

    def a(self, action: str, **args) -> str:
        """Shorthand for this view's act attribute."""
        return h.act(self.name, action, **args)

    def c(self, action: str, **args) -> str:
        return h.on_change(self.name, action, **args)

    def i(self, action: str, **args) -> str:
        return h.on_input(self.name, action, **args)

    def refresh(self) -> None:
        self.s.mark(self.name)


class Dialog(Handler):
    """A modal. Subclasses implement body() and footer(); actions they
    declare with self.a(...) come back to their on_* methods."""

    width = 520
    title = ""

    def __init__(self, title: str = "", width: Optional[int] = None):
        self.id = "d" + secrets.token_hex(5)
        self.session = None
        if title:
            self.title = title
        if width:
            self.width = width
        self.error = ""

    @property
    def target(self) -> str:   # type: ignore[override]
        return f"dlg:{self.id}"

    def a(self, action: str, **args) -> str:
        return h.act(self.target, action, **args)

    def c(self, action: str, **args) -> str:
        return h.on_change(self.target, action, **args)

    def i(self, action: str, **args) -> str:
        return h.on_input(self.target, action, **args)

    def submit(self, action: str, **args) -> str:
        return h.on_submit(self.target, action, **args)

    def body(self) -> str:
        return ""

    def footer(self) -> str:
        return h.button("Close", act_attr=self.a("close"))

    def title_html(self) -> str:
        return h.e(self.title)

    def render(self) -> str:
        err = f'<div class="dlg-err">{h.e(self.error)}</div>' if self.error else ""
        foot = self.footer()
        return (f'<div class="dlg-back" data-dlg="{self.id}">'
                f'<div class="dlg" role="dialog" aria-modal="true" aria-labelledby="{self.id}-t" '
                f'style="width:min({self.width}px,calc(100vw - 32px))">'
                f'<div class="dlg-h"><h2 id="{self.id}-t">{self.title_html()}</h2>'
                f'{h.button(icon_name="x", act_attr=self.a("close"), kind="ghost", title="Close", cls="dlg-x")}</div>'
                f'<div class="dlg-b">{self.body()}{err}</div>'
                f'{f"<div class=dlg-f>{foot}</div>" if foot else ""}</div></div>')

    def on_close(self, **_):
        self.session.close_dialog(self)

    def close(self) -> None:
        self.session.close_dialog(self)

    def refresh(self) -> None:
        if self.session is not None:
            self.session.mark("dialogs")


class MessageDialog(Dialog):
    def __init__(self, title: str, lines, width: int = 560, kind: str = ""):
        super().__init__(title, width)
        self.lines = [lines] if isinstance(lines, str) else list(lines)
        self.kind = kind

    def title_html(self) -> str:
        ic = {"warn": h.icon("warn", 18, "c-warn"), "ok": h.icon("check", 18, "c-ok")}.get(self.kind, "")
        return f"{ic}<span>{h.e(self.title)}</span>"

    def body(self) -> str:
        return "".join(line if line.startswith("<") else f"<p>{h.e(line)}</p>" for line in self.lines)


class ConfirmDialog(Dialog):
    def __init__(self, title: str, text: str, on_yes: Callable[[], None], yes_label: str = "Delete",
                 danger: bool = True, width: int = 460, extra_html: str = ""):
        super().__init__(title, width)
        self.text = text
        self.on_yes_fn = on_yes
        self.yes_label = yes_label
        self.danger = danger
        self.extra_html = extra_html

    def body(self) -> str:
        return f"<p>{h.e(self.text)}</p>{self.extra_html}"

    def footer(self) -> str:
        return (h.button("Cancel", act_attr=self.a("close"))
                + h.button(self.yes_label, act_attr=self.a("yes"), kind="danger" if self.danger else "primary"))

    def on_yes(self, **_):
        self.close()
        try:
            self.on_yes_fn()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            self.session.toast(f"Failed: {ex}", "error")


class PromptDialog(Dialog):
    """One or more text fields and a submit button. `fields` are
    (name, label, value) triples; `on_submit(values) -> error or None`."""

    def __init__(self, title: str, fields, on_submit: Callable[[dict], Optional[str]],
                 submit_label: str = "Save", note: str = "", width: int = 460):
        super().__init__(title, width)
        self.fields = list(fields)
        self.on_submit_fn = on_submit
        self.submit_label = submit_label
        self.note = note

    def body(self) -> str:
        inputs = "".join(
            h.text_input(value, name=name, label=label, autofocus=(i == 0))
            for i, (name, label, value) in enumerate(self.fields))
        note = f'<p class="muted small">{h.e(self.note)}</p>' if self.note else ""
        return f'<form class="form-col" id="{self.id}-f"{self.submit("submit")}>{inputs}{note}</form>'

    def footer(self) -> str:
        return (h.button("Cancel", act_attr=self.a("close"))
                + f'<button type="submit" form="{self.id}-f" class="btn btn-primary"><span>{h.e(self.submit_label)}</span></button>')

    def on_submit(self, **values):
        problem = self.on_submit_fn(values)
        if problem:
            self.error = problem
            self.refresh()
            return
        self.close()
