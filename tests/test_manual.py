"""The reference manual: its content, and the (i) buttons pointing into it.

Three things are worth a test here, and none of them is prose quality.

A page id is a *key* (see CLAUDE.md): (i) buttons all over the UI compare
against it, so a renamed page silently sends a reader to the front page
instead. Every id an (i) button names is checked to exist.

An equation is a string that only fails at render time, inside a dialog, on
the one page nobody opened during testing. They are all typeset here
instead -- which also pins the mathtext subset the pages may use.

And the English-only rule applies to the manual exactly as it does to the
rest of the UI, so the accent check runs over its text too.
"""
from __future__ import annotations

import ast
import html
import pathlib
import re

import pytest

from lookin.core import manual
from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, Para, Table,
)
from lookin.web.mathtext import render_math
from lookin.web.views.manual import ManualDialog

ACCENTED = re.compile(r"[À-ɏ]")


def _blocks(blocks):
    """Every block on a page, flattening the ones nested in a callout."""
    for block in blocks:
        if isinstance(block, Callout):
            yield from _blocks(block.blocks)
        else:
            yield block


def _texts(block) -> list:
    """Every user-visible string in one block."""
    if isinstance(block, (Para, Heading)):
        return [block.text]
    if isinstance(block, Bullets):
        return list(block.items)
    if isinstance(block, Table):
        return list(block.headers) + [str(c) for row in block.rows for c in row] + [block.caption]
    if isinstance(block, HowToRead):
        return [block.title] + [s for pair in block.items for s in pair]
    if isinstance(block, Equation):
        return [block.note] + [meaning for _, meaning in block.where]
    return []


def _page_texts(page) -> list:
    out = [page.title, page.summary, page.shown_in]
    for block in _blocks(page.blocks):
        out += _texts(block)
        if isinstance(block, Callout):  # unreachable via _blocks, kept for clarity
            out.append(block.title)
    for block in page.blocks:
        if isinstance(block, Callout):
            out.append(block.title)
    return out


ALL_PAGES = list(manual.PAGES.values())
PAGE_IDS = [page.id for page in ALL_PAGES]


def test_every_page_is_in_exactly_one_chapter():
    listed = [pid for chapter in manual.CHAPTERS for pid in chapter.pages]
    assert sorted(listed) == sorted(PAGE_IDS)
    assert len(listed) == len(set(listed))


def test_default_page_exists():
    assert manual.DEFAULT_PAGE in manual.PAGES


@pytest.mark.parametrize("page", ALL_PAGES, ids=PAGE_IDS)
def test_see_also_points_somewhere(page):
    for target in page.see_also:
        assert target in manual.PAGES, f"{page.id} links to unknown page {target!r}"
    assert page.id not in page.see_also


@pytest.mark.parametrize("page", ALL_PAGES, ids=PAGE_IDS)
def test_every_formula_typesets(page):
    """Both the display equations and the $inline$ spans inside prose."""
    for block in _blocks(page.blocks):
        if isinstance(block, Equation):
            render_math(block.tex)
            for symbol, _ in block.where:
                render_math(symbol)
        for text in _texts(block):
            parts = text.split("$")
            assert len(parts) % 2 == 1, f"unbalanced $ in {page.id}: {text[:60]}"
            for formula in parts[1::2]:
                render_math(formula)


@pytest.mark.parametrize("page", ALL_PAGES, ids=PAGE_IDS)
def test_no_accented_text(page):
    """English only, same rule as the rest of the UI (see CLAUDE.md)."""
    for text in _page_texts(page):
        assert not ACCENTED.search(text), f"accented character in {page.id}: {text[:60]}"


@pytest.mark.parametrize("page", ALL_PAGES, ids=PAGE_IDS)
def test_page_has_a_summary_and_a_body(page):
    assert len(page.summary) > 80, f"{page.id} has no real summary"
    assert page.blocks, f"{page.id} has no body"


UI_SOURCE = sorted(pathlib.Path("src/lookin/web").rglob("*.py"))


def _pages_named_in(source: str) -> set:
    """Every manual page id a view names, read out of its syntax tree.

    Four forms, because a view names a page in four places: the literal
    on a section (`manual="f1"`) or an (i) button (`info_btn("f1")`), the
    lookup table a dropdown resolves through (`PERCLASS_MANUAL_PAGES`, and
    the fallback of a `.get()` on it), and the resolver method that reads
    it (`manual_page`, whose fallback id appears nowhere else).
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == "manual":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                found.add(node.value.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            if name == "info_btn" or (name == "get" and isinstance(receiver, ast.Name)
                                      and receiver.id.endswith("_MANUAL_PAGES")):
                args = node.args if name == "info_btn" else node.args[1:]
                found |= {a.value for a in args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
        elif isinstance(node, ast.FunctionDef) and node.name.endswith("manual_page"):
            body = node.body[1:] if ast.get_docstring(node) else node.body  # skip the docstring
            for statement in body:
                found |= {n.value for n in ast.walk(statement)
                          if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name.endswith("_MANUAL_PAGES") for name in names):
                found |= {v.value for v in node.value.values
                          if isinstance(v, ast.Constant) and isinstance(v.value, str)}
    return found


def _referenced_pages() -> dict:
    """page id -> the files naming it, over the whole UI."""
    found: dict = {}
    for path in UI_SOURCE:
        if path.name in ("manual.py", "html.py"):
            continue
        for page_id in _pages_named_in(path.read_text(encoding="utf-8")):
            found.setdefault(page_id, []).append(path.name)
    return found


def test_every_info_button_points_at_a_real_page():
    """The reason a page id is a key and never reworded: renaming one here
    turns an (i) button somewhere into a link to the front page."""
    referenced = _referenced_pages()
    assert referenced, "no (i) button found in the UI -- the scan is broken, not the app"
    unknown = {pid: files for pid, files in referenced.items() if pid not in manual.PAGES}
    assert not unknown, f"(i) buttons point at pages that do not exist: {unknown}"


def test_every_dynamic_page_resolver_exists():
    """`manual=self._something(...)` has to name a method of the same class.

    A typo or a rename is not an import error -- it raises the first time
    that section is *drawn*, which is inside a background rebuild, which
    turns into an empty tab and a line in a log.
    """
    for path in UI_SOURCE:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            for inner in ast.walk(node):
                if not (isinstance(inner, ast.keyword) and inner.arg == "manual"):
                    continue
                value = inner.value.func if isinstance(inner.value, ast.Call) else inner.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
                        and value.value.id == "self":
                    assert value.attr in methods, \
                        f"{path.name}: {node.name} has no {value.attr}()"


def test_the_manual_is_reachable_from_the_tabs_that_need_it():
    """Every tab that shows a number should explain at least one of them."""
    referenced = _referenced_pages()
    naming = {name for files in referenced.values() for name in files}
    for view in ("metrics.py", "calibration.py", "confidence.py", "compare.py", "embeddings.py",
                 "prototypes.py", "cleaning.py", "slice_panel.py", "stats.py", "review_panel.py",
                 "sources.py", "filter.py"):
        assert view in naming, f"{view} has no (i) button"


def test_the_summary_table_names_real_pages():
    """Its last column points the reader at a page by *title*, so a page
    renamed without touching that table would send them looking for a
    heading that no longer exists."""
    titles = {page.title for page in ALL_PAGES}
    table = next(b for b in manual.PAGES["summary-table"].blocks
                 if isinstance(b, Table) and b.headers[-1] == "Page")
    for row in table.rows:
        assert row[-1] in titles, f"Summary table points at {row[-1]!r}, which is not a page title"


def test_search_matches_title_and_summary():
    assert manual.matches(manual.PAGES["f1"], "harmonic")
    assert manual.matches(manual.PAGES["f1"], "F1")
    assert not manual.matches(manual.PAGES["f1"], "umap")
    assert manual.matches(manual.PAGES["f1"], "")  # empty query keeps everything


def test_view_renders_every_page():
    """A page that only fails when drawn fails here, not in front of a user."""
    dialog = ManualDialog()
    for page_id in PAGE_IDS:
        dialog.show(page_id)
        assert f'data-key="mp-{page_id}"' in dialog.body()
        assert html.escape(manual.PAGES[page_id].title) in dialog.title_html()


def test_unknown_page_lands_on_the_index():
    dialog = ManualDialog()
    dialog.show("no-such-page")
    assert dialog.current == manual.DEFAULT_PAGE


def test_back_walks_the_history():
    dialog = ManualDialog()
    dialog.show("f1", remember=False)
    dialog.show("precision-recall")
    assert dialog.current == "precision-recall"
    dialog.show(dialog.history.pop(), remember=False)
    assert dialog.current == "f1"
    assert not dialog.history


def test_search_narrows_the_table_of_contents():
    dialog = ManualDialog()
    full = dialog.body().count('data-act="dlg:')
    dialog.query = "calibration"
    narrowed = dialog.body().count('data-act="dlg:')
    assert 0 < narrowed < full
