"""The manual: every page, in the order the table of contents shows them.

Content lives one module per chapter (pages_*.py); this module only stitches
them together and answers the two questions the UI asks -- "what page is
this (i) button pointing at?" and "which pages match what was typed in the
search box?".

Adding a page is: write it in the right pages_* module, list it in that
module's CHAPTER, and point an (i) button at its id. tests/test_manual.py
then checks that it is reachable, that its equations typeset and that no
(i) button in the app points at a page that does not exist.
"""
from __future__ import annotations

from lookin.core.manual.model import Chapter, ManualPage
from lookin.core.manual import (
    pages_overview,
    pages_foundations,
    pages_decision,
    pages_ranking,
    pages_probabilistic,
    pages_confidence,
    pages_abstention,
    pages_slices,
    pages_embeddings,
    pages_prototypes,
    pages_cleaning,
    pages_comparison,
    pages_dataset,
)

# Order matters: this is the table of contents, top to bottom. Foundations
# first because every later chapter leans on the confusion matrix, and the
# chapters after it follow the order of the tabs they explain.
_MODULES = (
    pages_overview,
    pages_dataset,
    pages_foundations,
    pages_decision,
    pages_ranking,
    pages_probabilistic,
    pages_confidence,
    pages_abstention,
    pages_slices,
    pages_comparison,
    pages_embeddings,
    pages_prototypes,
    pages_cleaning,
)

# The page an (i) button lands on when its own page is missing, and what
# the manual opens at from the app bar.
DEFAULT_PAGE = "index"


def _collect() -> tuple:
    chapters = []
    pages: dict = {}
    for module in _MODULES:
        chapter_id, chapter_title = module.CHAPTER
        ids = []
        for page in module.PAGES:
            if page.id in pages:
                raise ValueError(f"duplicate manual page id: {page.id}")
            pages[page.id] = page
            ids.append(page.id)
        chapters.append(Chapter(id=chapter_id, title=chapter_title, pages=tuple(ids)))
    return tuple(chapters), pages


CHAPTERS, PAGES = _collect()


def chapter_of(page_id: str) -> Chapter | None:
    for chapter in CHAPTERS:
        if page_id in chapter.pages:
            return chapter
    return None


def matches(page: ManualPage, query: str) -> bool:
    """Whether a page should stay in the table of contents while `query` is
    typed in the search box. Matched against the title, the summary and the
    page's own keywords -- the summary included on purpose, since a reader
    searching for "over-confident" is looking for a sentence, not a title."""
    query = (query or "").strip().lower()
    if not query:
        return True
    haystack = " ".join([page.title, page.summary, " ".join(page.keywords), page.id]).lower()
    return all(word in haystack for word in query.split())
