"""TrainHub badges, and the dialog that sets TrainHub's address.

A badge is a link to the TrainHub page that shows one MLflow run -- the
inference or training an experiment comes from -- read from TrainHub's run
index (core/trainhub_links.py). No TrainHub configured, or one that does not
answer: no badge, nothing else changes.
"""
from __future__ import annotations

from lookin.core import trainhub_links as th
from lookin.web import html as h
from lookin.web.views.base import Dialog


def badges(run_id: str) -> str:
    """One chip per place TrainHub shows `run_id`, each opening it there."""
    if not run_id:
        return ""
    out = []
    for link in th.TrainhubIndex.get().links(run_id):
        href = h.safe_href(link.url)
        if not href:
            continue
        out.append(f'<a class="chip th-chip" href="{h.e(href)}" target="_blank" rel="noopener" '
                   f'title="{h.e(link.title())} -- open in TrainHub">{h.icon("external", 11)}'
                   f'<span class="ell">TrainHub · {h.e(link.label)}</span></a>')
    return "".join(out)


def status_chip(act_attr: str) -> str:
    """Home's TrainHub chip: whether it answers, and a click to set it."""
    url, error, known = th.TrainhubIndex.get().status()
    if not url:
        return h.button("+ TrainHub", act_attr=act_attr, small=True, cls="chip",
                        title="Link Lookin to a TrainHub server: badges on the experiments it launched or lists")
    color = "var(--err)" if error else "var(--ok)"
    tip = f"TrainHub: {url} ({th.url_source()})\n" + (error or f"{known} MLflow run(s) known to TrainHub")
    return (f'<button type="button" class="chip{" off" if error else ""}"{act_attr} title="{h.e(tip)}">'
            f'{h.dot(color, size=6)}TrainHub</button>')


class TrainhubDialog(Dialog):
    width = 540
    title = "TrainHub"

    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.url = hub.config.trainhub_url
        self.overridden = th.url_source() != "home_view.json"

    def body(self) -> str:
        current, error, known = th.TrainhubIndex.get().status()
        note = ""
        if self.overridden:
            note = h.banner(f"The address in use comes from {th.url_source()} ({current}), which wins over the one "
                            "saved here.", "info")
        state = ""
        if current:
            state = h.banner(error, "warn") if error else h.banner(f"Connected: TrainHub knows {known} MLflow run(s).", "ok")
        return (f'<div class="form-col">{note}{state}'
                f'{h.text_input(self.url, change=self.c("url"), label="TrainHub address", placeholder="http://trainhub.lab:8420", autofocus=True, mono=True)}'
                f'<p class="muted small">Experiments made from an MLflow run TrainHub launched or lists (its projects, '
                f'checkpoints and inferences) get a badge that opens it there. A server on your own network, opt-in; '
                f'if TrainHub has a token, set it in {th.TOKEN_ENV}. Empty to unlink.</p></div>')

    def footer(self) -> str:
        return h.button("Cancel", act_attr=self.a("close")) + h.button("Save", act_attr=self.a("save"), kind="primary")

    def on_url(self, value: str = "", **_):
        self.url = (value or "").strip()

    def on_save(self, **_):
        if self.url and not self.url.lower().startswith(("http://", "https://")):
            self.error = "Enter an http:// or https:// address, or leave it empty."
            self.refresh()
            return
        self.hub.config.set_trainhub_url(self.url)
        self.close()
        th.TrainhubIndex.get().refresh()
