"""What every view and dialog of the web UI is made of.

A view is one tab (or Home): it owns the state of what the user chose
there, renders itself to HTML, and handles the actions its HTML declares
(`data-act="<name>:<action>"` -> `on_<action>(**args)`).

A dialog is the same thing in a modal layer, addressed as `dlg:<id>`.
"""
from __future__ import annotations

import secrets
import traceback
import types
from typing import Callable, Optional

from lookin.web import html as h


# What an action returns when it has already said everything it changed
# through commands (an infinite list's next batch goes out as an `append`):
# the view is then not re-rendered, which for a long list would resend every
# item already on screen.
NO_RENDER = object()


# on_* methods the session calls, not the page: a client naming one of them
# as an action would run a hook out of turn.
HOOKS = frozenset({"on_shown", "on_data_change", "on_review_change", "on_display_changed",
                   "on_prototype_source_changed"})


class Handler:
    target = ""

    def handle(self, action: str, args: dict):
        # Only a method of the class is an action. Several handlers keep the
        # callback they were given in an attribute named like one
        # (on_pick_fn, on_yes_fn, on_submit_fn, on_done...), and a client
        # sending "pick_fn" would call it directly with arguments of its own
        # choosing, past whatever checks the real action makes.
        name = f"on_{action}"
        if name in HOOKS or not isinstance(getattr(type(self), name, None), types.FunctionType):
            raise KeyError(f"{self.target}: no action '{action}'")
        if not isinstance(args or {}, dict):
            raise TypeError(f"{self.target}: the arguments of '{action}' must be an object")
        return getattr(self, name)(**(args or {}))


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


class StaleTracking:
    """What a tab does when the data under it changes -- the same in every
    tab that shows something computed from the data:

    * on_data_change() marks it `dirty` (when it shows something the change
      makes out of date);
    * arriving on the tab (AppActions.on_tab), or the data changing while it
      is the tab on screen (Session.notify_change), calls on_arrival(),
      which starts the update by itself;
    * while that runs, the red banner says so and offers "Cancel" -- never
      "Update now", which would only queue the same work again behind it;
    * Cancel stops the work at its next checkpoint (session.checkpoint);
      the banner then offers "Update now". A cancelled tab is not updated
      by itself again until the data changes again.

    Leaving the tab does not stop anything. A class using this sets
    `stale_job` (the session job its update runs on), keeps `dirty` and
    `cancelled`, and implements on_update(); can_update() says when there is
    something to update at all."""

    stale_message = "Data has changed -- this view is out of date."
    updating_message = "Data has changed -- this view is being updated."
    stopping_message = "Stopping at the next step..."
    cancelled_message = "The update was cancelled -- this view may be incomplete."
    stale_job = ""

    def init_stale(self) -> None:
        self.dirty = False
        self.cancelled = False       # the user pressed Cancel: wait for them, or for new data

    def job(self):
        job = self.s.job(self.stale_job or self.title or self.name)
        job.watchers.add(self.name)
        return job

    @property
    def running(self) -> bool:
        return self.job().busy

    def can_update(self) -> bool:
        return True

    def on_arrival(self) -> None:
        if self.dirty and not self.cancelled and not self.running and self.can_update():
            self.on_update()

    def cancel_button(self) -> str:
        # not "ghost": the banner's buttons are white on its red
        return h.button("Cancel", act_attr=self.a("cancel"), small=True, icon_name="x",
                        title="Stop this computation")

    def stale_html(self) -> str:
        if self.running:
            if not self.dirty or self.job().cancelling:
                return ""
            return h.banner(self.updating_message, "stale", actions=self.cancel_button())
        if self.cancelled:
            return h.banner(self.cancelled_message, "stale",
                            actions=h.button("Update now", act_attr=self.a("resume"), icon_name="refresh", small=True))
        if not self.dirty or not self.can_update():
            return ""
        return h.banner(self.stale_message, "stale",
                        actions=h.button("Update now", act_attr=self.a("update"), icon_name="refresh", small=True))

    def cancel_work(self) -> None:
        """The user's Cancel: stop the update; the banner then offers to run
        it again."""
        if not self.job().cancel():
            return
        self.cancelled = True
        self.on_cancelling()
        self.s.mark(self.name, "chrome")

    def on_cancelling(self) -> None:
        """Hook: what to show until the work reaches its next checkpoint."""

    def resume(self) -> None:
        self.cancelled = False
        self.on_update()

    def on_cancel(self, **_):
        self.cancel_work()

    def on_resume(self, **_):
        self.resume()


class RebuildView(StaleTracking, View):
    """A view whose content is expensive to rebuild (charts over the whole
    dataset), on the session's job queue for this view, with its progress
    shown in the view and in the top bar (see components.StaleBanner and
    BackgroundRebuild in the original UI). What a data change does is
    StaleTracking's; a redraw that is not a data update (a dropdown) gets
    its Cancel on the progress line."""

    def __init__(self, session):
        super().__init__(session)
        self.init_stale()
        self.progress_text = ""
        self.error = ""
        self._last = None            # (work, clear_stale) of the latest rebuild()

    def on_data_change(self) -> None:
        self.dirty = True

    def progress_html(self) -> str:
        if self.progress_text:
            # the red banner carries the Cancel while it is up
            cancel = (self.cancel_button()
                      if self.running and not self.dirty and not self.job().cancelling else "")
            return f'<div class="row-nw">{h.spinner(self.progress_text)}{cancel}</div>'
        if self.error:
            return h.banner(self.error, "error")
        return ""

    def set_progress(self, text: str) -> None:
        from lookin.web.session import checkpoint
        checkpoint()
        self.progress_text = text
        self.s.mark(self.name)

    def rebuild(self, work=None, clear_stale: bool = True) -> None:
        """Run `work` (default: self.compute) on this view's job queue.
        `clear_stale=False` for a redraw from what is already computed (a
        dropdown change): it must not hide the stale banner, since the data
        behind it was not refreshed."""
        work = work or self.compute
        self._last = (work, clear_stale)
        if not self.progress_text or self.progress_text == self.stopping_message:
            self.progress_text = "Updating..."
        self.error = ""
        self.cancelled = False

        def run():
            try:
                work()
                if clear_stale:
                    self.dirty = False
            finally:
                self.progress_text = ""
                self.s.mark(self.name, "chrome")

        def failed(ex):
            self.error = f"Error: {ex}"
            self.progress_text = ""
            self.s.mark(self.name)

        self.job().run(run, on_error=failed)
        self.s.mark(self.name)

    def compute(self) -> None:
        pass

    def on_update(self, **_):
        self.rebuild()

    def on_cancelling(self) -> None:
        self.progress_text = self.stopping_message      # cleared by the job when it stops

    def resume(self) -> None:
        """Run again what was cancelled: the data update when the data is
        what changed, else the redraw that was stopped."""
        self.cancelled = False
        if self.dirty or self._last is None:
            self.on_update()
        else:
            self.rebuild(*self._last)


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
