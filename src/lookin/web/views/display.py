"""How crops are mapped to the screen: the display LUT and the roll
correction. One session-wide setting (a LUT that applied in the carousel
but not in the gallery would mean two views of the same crop disagreeing),
display only -- no file is touched and nothing is computed through it.
Same behaviour as lookin.ui.display_settings.
"""
from __future__ import annotations

from lookin.ui.components import LUT_DESCRIPTIONS, LUT_NONE, LUT_OPTIONS, ROLL_NONE
from lookin.ui.display_settings import ROLL_NONE_KEY, roll_column_candidates
from lookin.web import html as h
from lookin.web.views.base import Dialog


def display_label(session) -> str:
    current = session.display_lut or LUT_NONE
    label = {key: text for key, text in LUT_OPTIONS}.get(current, current)
    short = {"none": "as stored", "minmax": "min/max", "percentile": "2–98% stretch", "equalize": "equalised"}.get(current, label)
    if session.display_roll_col:
        short += f" · roll: {session.display_roll_col}"
    return short


class DisplayDialog(Dialog):
    width = 500
    title = "Crop display"

    def body(self) -> str:
        s = self.session
        current = s.display_lut or LUT_NONE
        opts = []
        for key, label in LUT_OPTIONS:
            on = key == current
            opts.append(
                f'<button type="button" class="facet-b{" on" if on else ""}" style="min-height:44px;border-radius:6px;padding:4px 10px"'
                f'{self.a("lut", key=key)}><span style="width:16px">{"●" if on else "○"}</span>'
                f'<span style="display:flex;flex-direction:column"><span style="font-weight:{600 if on else 400}">{h.e(label)}</span>'
                f'<span class="muted small">{h.e(LUT_DESCRIPTIONS.get(key, ""))}</span></span></button>')
        dataset = getattr(s.state, "dataset", None)
        candidates = roll_column_candidates(dataset)
        roll = s.display_roll_col or ROLL_NONE
        if roll and roll not in candidates:
            s.display_roll_col = ROLL_NONE
            roll = ROLL_NONE
        if dataset is None:
            roll_html = '<p class="muted"><i>Load a dataset to pick a roll column.</i></p>'
        elif not candidates:
            roll_html = ('<p class="muted"><i>This dataset has no numeric column besides its id, crop and label ones, '
                         'so there is nothing that could hold a roll.</i></p>')
        else:
            roll_html = h.select([(ROLL_NONE_KEY, "No rotation (as stored)")] + [(c, c) for c in candidates],
                                 roll or ROLL_NONE_KEY, change=self.c("roll"), label="Roll column", cls="mono")
        return (
            '<p class="muted">Greyscale crops often use a narrow slice of the available range, which shows as a flat '
            'grey square. A lookup table stretches that slice across the screen.</p>'
            f'<div class="col" style="gap:2px">{"".join(opts)}</div>'
            '<div class="subhead" style="margin-top:14px">Roll correction</div>'
            '<p class="muted">If a column of the dataset records the roll each crop was cut at, every crop can be turned '
            'by the nearest quarter turn that cancels it, so a page of thumbnails shares one orientation. Values are '
            'read as radians.</p>'
            f'{roll_html}'
            '<p class="muted small" style="margin-top:8px"><i>Only whole quarter turns are applied: a crop rolled by 0.3 rad '
            'stays as stored, and no pixel is ever resampled.</i></p>'
            '<p class="muted small" style="border-top:1px solid var(--line-2);padding-top:8px"><i>Display only: the files are '
            'never modified, and no metric, embedding or prototype is computed through the LUT or the roll correction.</i></p>')

    def on_lut(self, key: str = LUT_NONE, **_):
        if key == (self.session.display_lut or LUT_NONE):
            return
        self.session.display_lut = key
        self.session.notify_display_change()
        self.session.mark("chrome", "dialogs")

    def on_roll(self, value: str = ROLL_NONE_KEY, **_):
        column = ROLL_NONE if value == ROLL_NONE_KEY else value
        if column == (self.session.display_roll_col or ROLL_NONE):
            return
        self.session.display_roll_col = column
        self.session.notify_display_change()
        self.session.mark("chrome", "dialogs")
