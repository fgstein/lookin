"""Exported files are named lookin_<workspace>_<tab>_<what>_<date>.<ext>."""
from __future__ import annotations

import re
from datetime import datetime

from lookin.core.export_names import export_name, export_stem, part, workspace_part


def test_a_name_says_where_and_when():
    when = datetime(2026, 11, 15, 9, 5, 7)
    assert export_name("Clean crops", "Metrics", "F1 per class", ext="png", when=when) == \
        "lookin_Clean_crops_Metrics_F1_per_class_20261115-090507.png"


def test_empty_and_repeated_parts_are_dropped():
    assert export_stem("ws", "", None, "Gallery", "gallery", "mosaic") == "lookin_ws_Gallery_mosaic"


def test_parts_are_safe_and_short():
    assert part("a/b\\\\c: d*?") == "a_b_c_d"
    assert len(part("x" * 500)) == 40
    assert re.fullmatch(r"[A-Za-z0-9._-]+", export_name("é w/s", "tab", "1:2", ext="md"))


def test_the_workspace_part_falls_back_to_the_dataset(loaded_state):
    class Ws:
        def label(self):
            return "My workspace"
    assert workspace_part(Ws(), loaded_state) == "My workspace"
    assert workspace_part(None, loaded_state) == "dataset"
    assert workspace_part(None, None) == ""


def test_the_page_carries_the_stem_and_charts_their_own(loaded_state):
    from lookin.web import render
    from lookin.web.server import WebApp
    from lookin.web.session import RequestScope
    from lookin.web.views.media import chart_html

    s = WebApp().store.get("z" * 24, create=True)
    with RequestScope(s):
        render.boot(s, {})
    s.state = loaded_state
    s.mode, s.tab = "analysis", "metrics"
    top = render.render_top(s)
    assert 'id="export-stem" data-stem="lookin_dataset_Metrics"' in top
    html = chart_html(s, b"\x89PNG\r\n\x1a\n" + b"\0" * 24, "F1 per class")
    assert 'data-export="lookin_dataset_Metrics_F1_per_class"' in html
