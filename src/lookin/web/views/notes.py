"""Notes: what the user has to say about a workspace, written as they would
write a message -- formatted text, no Markdown to type -- and kept as
Markdown in the workspace's folder (core/notes.py).

* a new note starts with the workspace's name and a short recap of its
  experiments' numbers, written once into the text (core/notes.recap);
  Regenerate writes the recap again from the current numbers;
* the text is edited in place in the page (static/js/notes.js) and saved
  as the user types; the page's editor is left alone by every re-render
  (data-keep) and only replaced when the text changed from elsewhere: a
  regeneration, another tab, an agent -- `rev` counts those;
* with no workspace open the note lives in this tab only, and goes into
  the workspace once the session is saved as one;
* it downloads as a .md file, and the reports carry it (Metrics tab).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from lookin.core import markdown_lite as md
from lookin.core import notes as nt
from lookin.core.metrics import compute_metrics
from lookin.web import html as h
from lookin.web.views.base import View

# The editor's toolbar: (command for notes.js, icon or label, tooltip).
# Commands are keys read by notes.js, never shown.
TOOLBAR = (
    ("bold", "B", "Bold (Ctrl+B)"),
    ("italic", "I", "Italic (Ctrl+I)"),
    ("strikeThrough", "S", "Strikethrough"),
    ("|", "", ""),
    ("h1", "H1", "Title"),
    ("h2", "H2", "Heading"),
    ("h3", "H3", "Subheading"),
    ("p", "¶", "Plain paragraph"),
    ("|", "", ""),
    ("insertUnorderedList", "•", "Bullet list (or type \"- \" at the start of a line)"),
    ("insertOrderedList", "1.", "Numbered list (or type \"1. \" at the start of a line)"),
    ("blockquote", "❝", "Quote"),
    ("code", "</>", "Inline code"),
    ("link", "link", "Link"),
    ("|", "", ""),
    ("removeFormat", "x", "Clear the formatting of the selection"),
)


class NotesView(View):
    name = "notes"
    title = "Notes"

    def __init__(self, session):
        super().__init__(session)
        self.text = ""                 # the note, as Markdown
        self.rev = 0                   # bumped when the editor must take `text` again
        self.loaded_for = None         # the workspace id `text` was read from, "" for none
        self._stamp = 0.0              # the notes file's mtime when last read or written here
        self.saved_at = ""
        self.status = ("", "")
        self.busy = ""
        # Whether this note has been started (from a file, or the template
        # written): an empty note the user cleared stays empty.
        self.started = False

    # ---- where the note lives ----

    def _workspace(self):
        ws = self.s.current_workspace
        return ws if ws is not None and ws.exists() else None

    def _sync(self) -> None:
        """Bring `text` in line with where the note is kept: the workspace
        opened (or changed) since, a file another tab or an agent wrote, a
        note written before this session was saved as a workspace."""
        ws = self._workspace()
        wid = ws.id if ws is not None else ""
        if wid != self.loaded_for:
            if ws is not None:
                saved = nt.read(ws)
                if saved is None and self.loaded_for == "" and self.started:
                    # Written in an unsaved session, which just became this
                    # workspace: the note goes with it.
                    self._persist()
                else:
                    self.text = saved or ""
                    self.started = saved is not None
                    self._stamp = nt.stamp(ws)
                    self.rev += 1
            elif self.loaded_for:
                self.text = ""
                self.started = False
                self.rev += 1
            self.loaded_for = wid
            return
        if ws is not None and nt.stamp(ws) != self._stamp:
            saved = nt.read(ws)
            self._stamp = nt.stamp(ws)
            if saved is not None and saved != self.text:
                self.text = saved
                self.rev += 1
                self.status = ("Updated from the saved file (another tab or an agent changed it).", "neutral")

    def changed_elsewhere(self) -> bool:
        ws = self._workspace()
        return ws is not None and ws.id == self.loaded_for and nt.stamp(ws) != self._stamp

    def _persist(self) -> None:
        ws = self._workspace()
        if ws is not None:
            nt.write(ws, self.text)
            self._stamp = nt.stamp(ws)
        self.saved_at = datetime.now().strftime("%H:%M:%S")

    def current_text(self) -> str:
        """The note as it stands, for a report."""
        self._sync()
        return self.text

    # ---- the recap ----

    def _metrics_by_name(self) -> dict:
        state = self.state
        out = {}
        if state.dataset is None:
            return out
        for name, exp in list(state.experiments.items()):
            out[name] = compute_metrics(name, state.merged(name), state.dataset.columns.label_col,
                                        exp.score_cols, exp.class_labels)
        return out

    def _title(self) -> str:
        ws = self.s.current_workspace
        if ws is not None:
            return ws.label()
        dataset = self.state.dataset
        return f"Notes on {Path(dataset.csv_path).name}" if dataset is not None and dataset.csv_path else "Notes"

    def _start(self) -> None:
        """A new note: title and recap, once the workspace has loaded."""
        self._fill_recap(new=True)

    def _fill_recap(self, new: bool) -> None:
        if self.s.job("Notes").busy:
            return
        self.busy = "Computing the recap..." if not new else "Writing the recap..."

        def work():
            recap = nt.recap(self.state, self._metrics_by_name())
            self.text = nt.template(self._title(), recap) if new else nt.with_recap(self.text, recap)
            self.started = True
            self.rev += 1
            self._persist()
            self.busy = ""
            self.status = ("Recap written from the current numbers." if not new else "", "success")
            self.s.mark(self.name)

        def failed(ex):
            self.busy = ""
            self.status = (f"Could not compute the recap: {ex}", "error")
            self.s.mark(self.name)

        self.s.job("Notes").run(work, on_error=failed)

    # ---- rendering ----

    def render(self) -> str:
        self._sync()
        loading = self.s.upload_view.is_loading()
        if not self.started and not self.busy:
            if loading:
                return h.section("Notes", h.spinner("Waiting for the workspace to finish loading..."),
                                 manual="notes")
            self._start()
        ws = self._workspace()
        where = ("Saved in the workspace as notes.md" if ws is not None
                 else "Kept in this tab only -- saved with the workspace once this session is saved as one")
        right = (h.button("Regenerate the recap", act_attr=' data-notes-act="regenerate"', small=True,
                          icon_name="refresh", disabled=bool(self.busy) or loading,
                          title="Write the performance summary again from the current numbers")
                 + h.button("Download .md", act_attr=' data-notes-act="download"', small=True, icon_name="download",
                            title="This note as a Markdown file"))
        tools = []
        for command, label, tip in TOOLBAR:
            if command == "|":
                tools.append('<span class="nt-sep"></span>')
                continue
            inner = h.icon("link", 14) if label == "link" else h.e(label)
            tools.append(f'<button type="button" class="nt-tool nt-{h.e(command)}" data-notes-cmd="{h.e(command)}" '
                         f'title="{h.e(tip)}" aria-label="{h.e(tip)}">{inner}</button>')
        editor = (f'<div class="notes-editor" contenteditable="true" spellcheck="true" data-keep data-key="notes-editor" '
                  f'data-rev="{self.rev}" data-html="{h.e(md.to_html(self.text))}" role="textbox" aria-multiline="true" '
                  f'aria-label="Notes"></div>')
        text, kind = self.status
        saved = f"Saved {self.saved_at}" if self.saved_at else ""
        foot = (f'<div class="row small muted" data-key="notes-foot">{h.spinner(self.busy) if self.busy else ""}'
                f'<span class="grow">{h.e(text)}</span><span>{h.e(saved)}</span></div>')
        body = f'<div class="nt-bar" role="toolbar" aria-label="Formatting">{"".join(tools)}</div>{editor}{foot}'
        return h.section("Notes", body, manual="notes", right=right, sub=where)

    # ---- actions ----

    def _take(self, html_text, rev) -> bool:
        """The editor's content, unless it predates a text the editor has
        not shown yet (a save racing a regeneration)."""
        try:
            if int(rev) != self.rev:
                return False
        except (TypeError, ValueError):
            return False
        if self.changed_elsewhere():
            # Written by another tab or an agent since this tab read it:
            # theirs is shown, rather than overwritten by an editor that
            # never saw it.
            self._sync()
            return False
        text = md.from_html(str(html_text or ""))
        if text != self.text:
            self.text = text
            self._persist()
            self.status = ("", "")
        return True

    def on_save(self, html: str = "", rev=0, **_):
        self._take(html, rev)

    def on_regenerate(self, html: str = "", rev=0, **_):
        self._take(html, rev)
        if self.state.dataset is None:
            self.status = ("Load a dataset first: the recap is about its experiments.", "warning")
            return
        self._fill_recap(new=not self.started)

    def on_download(self, html: str = "", rev=0, **_):
        self._take(html, rev)
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", self._title()).strip("_") or "notes"
        key = self.s.put_download(self.text.encode("utf-8"), f"{name}_notes.md", "text/markdown; charset=utf-8")
        self.s.cmd(download=f"/dl/{self.s.sid}/{key}", filename=f"{name}_notes.md")
