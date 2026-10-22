"""The shared paginated mosaic of crops, and what hangs off it: the review
decision bar and the "Find similar" search.

* a tight grid of crops, two pixels apart, at a zoom step from ZOOM_STEPS,
  that loads as it is scrolled: it opens with a screen of crops plus a row
  and appends the next screenful whenever the bottom comes within a screen
  (an infinite list, see app.js). Rows come lazily through
  `page_provider(start, end)`, so a set of any size is never materialized,
  and only the new tiles of each batch cross the wire;
* opens zoomed out just far enough for the whole set to fit on one screen;
* three toggles: class badges ("truth -> prediction" where the row has a
  prediction), details (metadata, Find similar, decision bar on every
  crop), shuffle;
* changing zoom or mode keeps the crop at the top of the view in view;
* a click shows the crop full size (the page's own zoom layer, with Open
  and Save on the user's machine).

The geometry is computed from the browser window the page reports with
every request (Session.viewport).
"""
from __future__ import annotations

import math
import secrets
import traceback
from typing import Callable, Optional

import numpy as np

from lookin.core.embedding_analysis import nearest_neighbours, nearest_to_vector
from lookin.core.point_coloring import categorical_colors
from lookin.core.review import FrozenVersionError, REVIEW_STATUSES, STATUS_RELABEL
from lookin.core.embedding_context import build_analysis_context
from lookin.core.review_actions import NO_REASON, REASON_LABELS, STATUS_LABELS, record_decision
from lookin.core.review import REVIEW_REASONS
from lookin.web import html as h
from lookin.web.views.base import NO_RENDER, Dialog
from lookin.web.views.details import details_url
from lookin.web.views.media import crop_url
from lookin.web.views.widgets import class_select


# The zoom ladder, in pixels of tile edge. Steps rather than a slider
# because every step has to re-fetch and re-lay out a page of full-res
# crops: a slider would do that on every drag frame.
#
# The top steps are larger than any tile this actually draws: what stops
# the zoom is the window (see _tile_size), not the end of this tuple. A
# ladder that stopped at 400px stopped there on a tall screen too, leaving
# two rows of crops and a zoom-in button that refused to make it one.
ZOOM_STEPS = (64, 88, 120, 160, 220, 300, 400, 540, 720, 960, 1280)
DEFAULT_ZOOM = 2  # 120px -- a crop you can recognize, a screen you can scan
TILE_GAP = 2   # "very little space between the images", i.e. a mosaic
INFO_GAP = 10  # cards need to read as separate objects, tiles do not

# A card in details mode is at least this wide, whatever the zoom: the
# decision bar puts its four verdicts on one row and its two pickers plus
# the reset button on the next, and below this they wrap -- "Uncertain"
# drops onto a line of its own and reads as a different kind of answer
# than the other three.
INFO_MIN_WIDTH = 380

# ...and never wider than this, however much room is left over: two cards
# on a wide screen should not become two billboards.
INFO_MAX_WIDTH = 520

# Lines in that list at most. A row of a joined frame carries one score
# column per class -- two thousand lines under every crop, and as many
# controls, at that many classes; the columns that say something about a
# crop (its metadata, the prediction) come first.
META_MAX_ROWS = 60

# AppState.merged's derived columns (keys, compared here, never reworded).
PREDICTION_KEYS = ("predicted_label", "predicted_score", "correct")

# Below this a crop is not a crop any more, so a window too short for even
# one card stops shrinking the image and lets the grid scroll instead.
MIN_TILE_SIZE = 48

# A badge is only ever a corner of a tile, so a long class name is cut
# rather than allowed to run past the crop and over its neighbours. The
# tile cannot do the cutting: it puts no width constraint on the badge, so
# an ellipsis overflow would never trigger, and the server has no way to
# measure rendered text. Hence a character budget from an estimated
# average glyph width -- approximate on purpose, and only ever wrong by a
# few pixels of a badge.
BADGE_MAX_CHARS = 12
BADGE_MIN_CHARS = 4
BADGE_CHAR_PX = 4.9  # at BADGE_TEXT_SIZE, for the app's font


def _short(label, budget: int) -> str:
    """A class label cut to what fits in a badge. Never used as a key --
    the label the code compares against is always the record's own
    value."""
    text = str(label)
    return text if len(text) <= budget else text[:max(1, budget - 3)] + "..."


DEFAULT_K = 48
K_CHOICES = (12, 24, 48, 96)

# What the dialog takes around the mosaic: its title bar, footer, padding
# and the toolbar over the grid.
DIALOG_CHROME_H = 172
DIALOG_CHROME_W = 64
MIN_GRID_WIDTH = 360
MIN_GRID_HEIGHT = 240
# What a card carries under its crop in details mode (metadata list,
# Find similar, the decision bar) -- the CSS below sizes them the same.
INFO_META_HEIGHT = 76
INFO_BAR_HEIGHT = 118
SIMILAR_BTN_HEIGHT = 30
CARD_PADDING = 16
STATUS_ICON = {"ok": "check", "relabel": "edit", "exclude": "x", "uncertain": "info"}
STATUS_CLS = {"ok": "c-ok", "relabel": "c-acc", "exclude": "c-err", "uncertain": "c-warn"}


# ---- review vocabulary, rendered -------------------------------------------

def status_chip(status: str, inherited_from: str = "") -> str:
    text = STATUS_LABELS.get(status, status)
    if inherited_from:
        text = f"{text} - from {inherited_from}"
    return (f'<span class="st st-{h.e(status)}">{h.icon(STATUS_ICON.get(status, "info"), 12)}'
            f'<span>{h.e(text)}</span></span>')


def decision_summary(session, uid) -> str:
    store = getattr(session.state, "review", None)
    if store is None or not store.versions:
        return '<span class="muted small"><i>not reviewed</i></span>'
    decision = store.decision_for(str(uid))
    if decision is None:
        return '<span class="muted small"><i>not reviewed</i></span>'
    inherited = decision.version_id if decision.version_id != store.active_version_id else ""
    parts = [status_chip(decision.status, inherited)]
    if decision.status == STATUS_RELABEL and decision.new_label:
        parts.append(f'<b class="small">-&gt; {h.e(decision.new_label)}</b>')
    if decision.reason:
        parts.append(f'<span class="muted small">{h.e(REASON_LABELS.get(decision.reason, decision.reason))}</span>')
    return f'<span class="row-wrap" style="gap:6px">{"".join(parts)}</span>'


def reason_options() -> list:
    return [(NO_REASON, "No reason")] + [(r, REASON_LABELS.get(r, r)) for r in REVIEW_REASONS]


class DecisionBar:
    """Four verdicts, a class picker for relabelling and a reason picker,
    for one crop. Its picker values live in `picks` (owned by the caller,
    keyed by uuid) so they survive re-renders."""

    def __init__(self, session, target: str, action: str, picks: dict, messages: dict):
        self.s = session
        self.target = target
        self.action = action
        self.picks = picks
        self.messages = messages

    def default_class(self, suggestion: str):
        labels = [str(l) for l in self.s.state.dataset.labels()] if self.s.state.dataset is not None else []
        if suggestion in labels:
            return suggestion
        return labels[0] if labels else None

    def render(self, uid, suggestion: str = "", compact: bool = True) -> str:
        uid = str(uid)
        labels = self.s.state.dataset.labels() if self.s.state.dataset is not None else []
        cls, reason = self.picks.get(uid, (self.default_class(suggestion), NO_REASON))
        a = lambda op, **kw: h.act(self.target, self.action, op=op, uid=uid, **kw)   # noqa: E731
        c = lambda op: h.on_change(self.target, self.action, op=op, uid=uid)       # noqa: E731
        buttons = "".join(
            f'<button type="button" class="btn btn-sm dec dec-{h.e(st)}"{a("decide", status=st)}>'
            f'{h.icon(STATUS_ICON[st], 13, STATUS_CLS[st])}<span>{h.e(STATUS_LABELS[st])}</span></button>'
            for st in REVIEW_STATUSES)
        pickers = (class_select(labels, cls, c("class"), title="New class", cls="dec-pick")
                   + h.select(reason_options(), reason, change=c("reason"), title="Reason", cls="dec-pick")
                   + h.button(icon_name="rotate", act_attr=a("reset"), small=True, kind="ghost",
                              title="Clear this decision"))
        msg = self.messages.get(uid)
        status = (f'<span class="c-err small">{h.e(msg)}</span>' if msg else decision_summary(self.s, uid))
        head = "" if compact else '<div class="small"><b>Review</b></div>'
        return (f'<div class="decbar">{head}<div class="dec-row">{buttons}</div>'
                f'<div class="dec-row">{pickers}</div><div class="dec-st">{status}</div></div>')

    def handle(self, op: str, args: dict, source: str) -> None:
        uid = str(args.get("uid", ""))
        if not uid:
            return
        cls, reason = self.picks.get(uid, (self.default_class(""), NO_REASON))
        if op == "class":
            self.picks[uid] = (args.get("value") or cls, reason)
        elif op == "reason":
            self.picks[uid] = (cls, args.get("value") or NO_REASON)
        elif op == "decide":
            status = str(args.get("status", ""))
            ok, message = record_decision(
                self.s, [uid], status,
                new_label=(cls or "") if status == STATUS_RELABEL else "",
                reason="" if reason in (None, NO_REASON) else reason, source=source)
            self.messages.pop(uid, None)
            if not ok:
                self.messages[uid] = message
        elif op == "reset":
            store = self.s.state.review
            self.messages.pop(uid, None)
            try:
                store.reset(uid)
                store.save()
                self.s.notify_review()
            except FrozenVersionError as ex:
                self.messages[uid] = str(ex)


# ---- the mosaic ---------------------------------------------------------------

class Carousel:
    """A component: `owner` is the dialog (or the view) it sits in, whose
    `on_car(op, ...)` forwards here. `chrome_height` is what the host takes
    above the mosaic on top of a plain dialog's chrome; a host that is not
    a dialog gives its own `chrome` -- (width, height) of the window that
    is not the mosaic.

    `shuffle_toggle`: offer the shuffle button (a host with its own shuffle
    turns it off). `outcome_borders`: frame each crop green or red by
    whether the prediction it carries ("correct") was right."""

    def __init__(self, owner, session, dataset, total: int, page_provider: Callable[[int, int], list],
                 crop_col: str, uuid_col: Optional[str] = None, highlight_uuid=None,
                 review_source: str = "", suggestions: Optional[dict] = None,
                 label_col: Optional[str] = None, chrome_height: int = 0, action: str = "car",
                 chrome: Optional[tuple] = None, shuffle_toggle: bool = True, outcome_borders: bool = False,
                 list_id: str = ""):
        self.owner = owner
        self.s = session
        self.dataset = dataset
        self.total = total
        self.page_provider = page_provider
        self.crop_col = crop_col
        self.uuid_col = uuid_col
        self.highlight_uuid = highlight_uuid
        self.review_source = review_source
        self.suggestions = suggestions or {}
        self.label_col = label_col
        self.extra_chrome = chrome_height
        self.chrome = chrome or (DIALOG_CHROME_W, DIALOG_CHROME_H)
        self.shuffle_toggle = shuffle_toggle
        self.outcome_borders = outcome_borders
        self.action = action
        self.loaded = 0                 # crops in the grid so far
        self.gen = 0                    # bumped whenever the grid starts over
        self.list_id = list_id or f"car-{secrets.token_hex(4)}"
        self.show_labels = True
        self.show_info = False
        self._order: Optional[np.ndarray] = None
        self._colors: Optional[dict] = None
        self.picks: dict = {}
        self.messages: dict = {}
        self.bar = DecisionBar(session, owner.target, action, self.picks, self.messages)
        self.zoom_index = DEFAULT_ZOOM
        self.zoom_index = self._initial_zoom()

    # ---- what the mosaic is allowed to show ----

    def _reviewable(self) -> bool:
        return bool(self.review_source and getattr(self.s.state, "review", None) is not None
                    and self.uuid_col is not None)

    def _offers_similarity(self) -> bool:
        return bool(self.review_source != "similarity" and self.uuid_col and similarity_available(self.s))

    # ---- geometry ----

    def _grid_width(self) -> int:
        return int(max(MIN_GRID_WIDTH, self.s.viewport[0] - self.chrome[0]))

    def _grid_height(self) -> int:
        return int(max(MIN_GRID_HEIGHT, self.s.viewport[1] - self.chrome[1] - self.extra_chrome))

    def _info_extra_height(self) -> int:
        if not self.show_info:
            return 0
        height = INFO_META_HEIGHT + CARD_PADDING
        if self._reviewable():
            height += INFO_BAR_HEIGHT
        if self._offers_similarity():
            height += SIMILAR_BTN_HEIGHT
        return height

    def _tile_size(self, zoom: Optional[int] = None) -> int:
        step = ZOOM_STEPS[self.zoom_index if zoom is None else zoom]
        by_width = self._grid_width() - TILE_GAP
        by_height = self._grid_height() - self._info_extra_height()
        return int(max(MIN_TILE_SIZE, min(step, by_width, by_height)))

    def _gap(self) -> int:
        return INFO_GAP if self.show_info else TILE_GAP

    def _columns(self, zoom: Optional[int] = None) -> int:
        gap = self._gap()
        minimum = INFO_MIN_WIDTH if self.show_info else self._tile_size(zoom)
        return max(1, int((self._grid_width() + gap) // (minimum + gap)))

    def _cell_size(self, zoom: Optional[int] = None) -> tuple:
        tile = self._tile_size(zoom)
        if not self.show_info:
            return tile, tile
        gap = self._gap()
        columns = self._columns(zoom)
        width = (self._grid_width() - gap * (columns - 1)) / columns
        width = int(max(INFO_MIN_WIDTH, min(INFO_MAX_WIDTH, width)))
        return min(width, self._grid_width()), int(tile + self._info_extra_height())

    def _rows(self, zoom: Optional[int] = None) -> int:
        _, cell_h = self._cell_size(zoom)
        gap = self._gap()
        return max(1, int((self._grid_height() + gap) // (cell_h + gap)))

    def _tiles_per_screen(self, zoom: Optional[int] = None) -> int:
        return max(1, self._columns(zoom) * self._rows(zoom))

    def _batch_size(self) -> int:
        """A screen plus a row, in whole rows: the extra row makes the first
        batch overflow the area, so there is something to scroll."""
        return self._columns() * (self._rows() + 1)

    def _initial_zoom(self) -> int:
        if self.total <= 0:
            return DEFAULT_ZOOM
        for index in reversed(range(len(ZOOM_STEPS))):
            if self._tiles_per_screen(index) >= self.total:
                return index
        return 0

    # ---- class colors ----

    def _class_color(self, label) -> str:
        if self._colors is None:
            labels = []
            if self.dataset is not None:
                try:
                    labels = [str(l) for l in self.dataset.labels()]
                except Exception:  # noqa: BLE001
                    labels = []
            self._colors = dict(zip(labels, categorical_colors(len(labels))))
        return self._colors.get(str(label), "#15181E")

    # ---- rendering ----

    def _fetch(self, start: int, end: int) -> list:
        if self._order is None:
            return self.page_provider(start, end)
        records = []
        for i in self._order[start:end].tolist():
            records.extend(self.page_provider(i, i + 1))
        return records

    def _badge_budget(self, with_prediction: bool) -> int:
        usable = self._tile_size() - 12
        if with_prediction:
            usable -= 3 * BADGE_CHAR_PX
            usable /= 2
        return int(max(BADGE_MIN_CHARS, min(BADGE_MAX_CHARS, usable / BADGE_CHAR_PX)))

    def _badge(self, rec: dict) -> str:
        if not self.show_labels or not self.label_col:
            return ""
        truth = rec.get(self.label_col)
        if truth is None:
            return ""
        predicted = rec.get("predicted_label")
        budget = self._badge_budget(predicted is not None)
        out = f'<b style="color:{self._class_color(truth)}">{h.e(_short(truth, budget))}</b>'
        if predicted is not None:
            out += f' <span style="color:{self._class_color(predicted)}">-&gt; {h.e(_short(predicted, budget))}</span>'
        return f'<span class="cbadge">{out}</span>'

    def _tooltip(self, rec: dict) -> str:
        parts = []
        if self.label_col and rec.get(self.label_col) is not None:
            parts.append(str(rec[self.label_col]))
        if rec.get("predicted_label") is not None:
            parts.append(f"predicted {rec['predicted_label']}")
        if self.uuid_col and rec.get(self.uuid_col) is not None:
            parts.append(str(rec[self.uuid_col]))
        parts.append("click to view full size")
        return "  -  ".join(parts)

    def _info_url(self, rec: dict) -> str:
        """Where the full-size view's (i) panel reads this crop's details."""
        uid = rec.get(self.uuid_col) if self.uuid_col else None
        return details_url(self.s, uid=uid, crop=None if uid is not None else rec.get(self.crop_col))

    def _tile(self, rec: dict, i: int) -> str:
        img_path = self.dataset.resolve_crop_path(rec[self.crop_col])
        url = crop_url(self.s, img_path, rec)
        tile = self._tile_size()
        cell_w, _ = self._cell_size()
        img_w = cell_w - CARD_PADDING if self.show_info else tile
        highlighted = (self.highlight_uuid is not None and self.uuid_col is not None
                       and rec.get(self.uuid_col) == self.highlight_uuid)
        title = (str(rec.get(self.uuid_col)) if self.uuid_col and rec.get(self.uuid_col) is not None
                 else img_path.name)
        outcome = ""
        if self.outcome_borders and "correct" in rec:
            truth = rec.get(self.label_col) if self.label_col else None
            if truth is not None and str(truth) != "" and truth == truth:
                outcome = " ok" if rec.get("correct") else " bad"
        img = (f'<span class="ctile-i" style="width:{img_w}px;height:{tile}px">'
               f'<img class="crop" src="{h.e(url)}" alt="{h.e(title)}" loading="lazy" data-zoom data-pixelated '
               f'data-title="{h.e(title)}" data-info="{h.e(self._info_url(rec))}" title="{h.e(self._tooltip(rec))}" '
               f'style="width:{img_w}px;height:{tile}px">{self._badge(rec)}</span>')
        if not self.show_info:
            return (f'<div class="ctile{" hl" if highlighted else ""}{outcome}" data-i="{i}" '
                    f'style="width:{tile}px;height:{tile}px">{img}</div>')
        return self._card(rec, img, cell_w, highlighted, i, outcome)

    def _card(self, rec: dict, img: str, cell_w: int, highlighted: bool, i: int, outcome: str = "") -> str:
        items = [(k, v) for k, v in rec.items() if k != self.crop_col]
        items.sort(key=lambda kv: kv[0] not in PREDICTION_KEYS)
        meta = "".join(
            f'<div class="meta-r"><b>{h.e(k)}</b><span>{h.e(f"{v:.4f}" if isinstance(v, float) else str(v))}</span></div>'
            for k, v in items[:META_MAX_ROWS])
        if len(items) > META_MAX_ROWS:
            meta += f'<div class="muted small"><i>... and {len(items) - META_MAX_ROWS:,} more column(s)</i></div>'
        parts = [img, f'<div class="meta" style="height:{INFO_META_HEIGHT}px">{meta}</div>']
        uid = rec.get(self.uuid_col) if self.uuid_col else None
        if self._offers_similarity() and uid is not None:
            parts.append(h.button("Find similar", act_attr=h.act(self.owner.target, self.action, op="similar", uid=str(uid)),
                                  icon_name="search", small=True, kind="ghost",
                                  title="The crops nearest to this one in the model's embedding space"))
        if self._reviewable():
            parts.append(self.bar.render(uid, str(self.suggestions.get(str(uid), ""))))
        return (f'<div class="ccard{" hl" if highlighted else ""}{outcome}" data-i="{i}" style="width:{cell_w}px">'
                f'{"".join(parts)}</div>')

    def _tiles_html(self, start: int, end: int) -> str:
        try:
            records = self._fetch(start, end)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            self.messages["__page__"] = f"Could not read these crops: {ex}"
            return ""
        return "".join(self._tile(rec, start + k) for k, rec in enumerate(records))

    def _count_text(self) -> str:
        # The size of the set, not how much of it is loaded yet: a text that
        # changed length as the grid filled in moved every button beside it.
        return f"{self.total:,} image(s)" if self.total else "0 images"

    def render(self, lead_html: str = "") -> str:
        """`lead_html`: the host's own controls, at the start of the toolbar."""
        if not self.loaded and self.total:
            self.loaded = min(self.total, self._batch_size())
        a = lambda op, **kw: h.act(self.owner.target, self.action, op=op, **kw)  # noqa: E731
        anchor = f' data-anchor="{self.list_id}"'
        zoom_in_off = (self.zoom_index >= len(ZOOM_STEPS) - 1
                       or self._tile_size(self.zoom_index + 1) <= self._tile_size())
        shuffled = self._order is not None
        tog = lambda on, name, tip, op: h.button(icon_name=name, act_attr=a(op) + anchor, small=True,  # noqa: E731
                                                 kind="ghost", title=tip, cls="on" if on else "")
        shuffle = (tog(shuffled, "shuffle", "Back to the original order" if shuffled else "Show the crops in a random order", "shuffle")
                   if self.shuffle_toggle else "")
        lead = f'{lead_html}<span class="vsep"></span>' if lead_html else ""
        toolbar = (f'<div class="car-tb">{lead}'
                   f'{h.button("-", act_attr=a("zoom_out") + anchor, small=True, title="Smaller images", disabled=self.zoom_index <= 0)}'
                   f'{h.button("+", act_attr=a("zoom_in") + anchor, small=True, title="Bigger images", disabled=zoom_in_off)}'
                   f'<span class="vsep"></span>'
                   f'{tog(self.show_labels, "bookmark", "Hide the class labels" if self.show_labels else "Show the class labels", "labels")}'
                   f'{tog(self.show_info, "info", "Back to the plain mosaic" if self.show_info else "Show the metadata and the review controls on every crop", "info")}'
                   f'{shuffle}'
                   f'{h.button(icon_name="chev-up", act_attr=a("top"), small=True, kind="ghost", title="Back to the top")}'
                   f'<span class="grow"></span>'
                   f'<span class="muted small nowrap" id="{self.list_id}-count">{h.e(self._count_text())}</span>'
                   f'</div>')
        tiles = self._tiles_html(0, self.loaded) if self.total else ""
        err = self.messages.pop("__page__", "")
        if not tiles:
            body = f'<div class="empty"><i>{h.e(err or "No images match.")}</i></div>'
        else:
            more = self.loaded < self.total
            body = (f'<div class="car-list" id="{self.list_id}" data-gen="{self.gen}" style="gap:{self._gap()}px">{tiles}</div>'
                    f'<div class="more" data-more="{h.e(self.owner.target)}:{self.action}" data-op="more" '
                    f'data-args=\'{{"op": "more"}}\' data-list="{self.list_id}"{"" if more else " hidden"}>'
                    f'{h.button("Show more", act_attr=a("more", gen=self.gen, loaded=self.loaded), small=True, kind="ghost", icon_name="chev-down")}</div>')
        grid = (f'<div class="car-grid" style="height:{self._grid_height()}px;width:{self._grid_width()}px">'
                f'{body}</div>')
        return f'<div class="car">{toolbar}{grid}</div>'

    # ---- actions ----

    def _restart_at(self, anchor: int) -> None:
        """Start the grid over with crop `anchor` loaded and scrolled to the
        top of the view -- what a zoom step or a mode toggle does, since a
        row then holds a different number of crops and the old scroll
        offset means nothing."""
        columns = self._columns()
        anchor = max(0, min(int(anchor or 0), max(0, self.total - 1)))
        row_start = (anchor // columns) * columns
        self.loaded = min(self.total, row_start + self._batch_size())
        self.gen += 1
        self.s.cmd(scroll_item={"list": self.list_id, "index": row_start})

    def handle(self, op: str, args: dict):
        anchor = args.get("anchor", 0)
        if op == "more":
            if str(args.get("gen")) != str(self.gen):
                return NO_RENDER
            if int(args.get("loaded", -1)) != self.loaded:
                return None
            start, end = self.loaded, min(self.total, self.loaded + self._batch_size())
            html = self._tiles_html(start, end)
            self.loaded = end
            self.s.cmd(append={"to": self.list_id, "gen": str(self.gen), "start": start, "html": html,
                               "count": self._count_text(), "done": self.loaded >= self.total})
            return NO_RENDER
        if op == "top":
            self.s.cmd(scroll_item={"list": self.list_id, "index": 0})
            return NO_RENDER
        if op == "zoom_in":
            if (self.zoom_index < len(ZOOM_STEPS) - 1
                    and self._tile_size(self.zoom_index + 1) > self._tile_size()):
                self.zoom_index += 1
                self._restart_at(anchor)
        elif op == "zoom_out":
            if self.zoom_index > 0:
                self.zoom_index -= 1
                self._restart_at(anchor)
        elif op == "labels":
            self.show_labels = not self.show_labels
            self._restart_at(anchor)
        elif op == "info":
            self.show_info = not self.show_info
            self._restart_at(anchor)
        elif op == "shuffle":
            self._order = None if self._order is not None else np.random.default_rng().permutation(self.total)
            self._restart_at(0)
        elif op == "similar":
            open_similarity(self.s, str(args.get("uid", "")))
        elif op in ("decide", "reset", "class", "reason"):
            self.bar.handle(op, args, self.review_source)
        return None


class CarouselDialog(Dialog):
    """A dialog whose whole content is one mosaic."""

    def __init__(self, session, title: str, dataset, total: int, page_provider, crop_col: str,
                 uuid_col=None, highlight_uuid=None, review_source: str = "", suggestions=None,
                 label_col=None, header_html: str = "", chrome_height: int = 0):
        super().__init__(title, width=4000)
        self.session = session
        self.header_html = header_html
        self.carousel = Carousel(self, session, dataset, total, page_provider, crop_col, uuid_col,
                                 highlight_uuid, review_source, suggestions, label_col,
                                 chrome_height=chrome_height + (40 if header_html else 0))

    def body(self) -> str:
        head = f'<div class="car-head">{self.header_html}</div>' if self.header_html else ""
        return head + self.carousel.render()

    def on_car(self, op: str = "", **args):
        if self.carousel.handle(op, args) is NO_RENDER:
            return NO_RENDER
        self.refresh()

    def on_review_change(self) -> None:
        self.refresh()


def open_carousel(session, title: str, records: list, crop_col: str, uuid_col=None, highlight_uuid=None,
                  review_source: str = "", suggestions=None, label_col=None, header_html: str = "") -> CarouselDialog:
    """build_carousel_from_list, as a dialog: a small, fully materialized
    list of row dicts."""
    dialog = CarouselDialog(session, title, session.state.dataset, len(records),
                            lambda start, end: records[start:end], crop_col, uuid_col, highlight_uuid,
                            review_source, suggestions, label_col, header_html=header_html)
    session.open_dialog(dialog)
    return dialog


# ---- similarity search ---------------------------------------------------------

def experiments_with_embeddings(session) -> list:
    return [name for name, exp in session.state.experiments.items() if exp.has_embeddings()]


def similarity_available(session) -> bool:
    return bool(session.state.dataset is not None and experiments_with_embeddings(session))


def open_similarity(session, query_uuid: str) -> None:
    if not query_uuid:
        return
    previous = getattr(session, "_similarity_experiment", None)
    session.open_dialog(SimilarityDialog(session, query_uuid, previous))


class SimilarityDialog(Dialog):
    def __init__(self, session, query_uuid: str, experiment=None):
        super().__init__(f"Crops most like {query_uuid}", width=4000)
        self.session = session
        self.query = str(query_uuid)
        names = experiments_with_embeddings(session)
        self.experiment = experiment if experiment in names else (names[0] if names else None)
        self.k = DEFAULT_K
        self.summary = ""
        self.message = ("", "info")
        self.carousel: Optional[Carousel] = None
        self._compute()

    def title_html(self) -> str:
        return (f'{h.icon("search", 16, "c-acc")}<span>{h.e(self.title)}</span>'
                f'<span class="grow"></span>{h.info_btn("nearest-neighbours")}')

    def _compute(self) -> None:
        s = self.session
        self.carousel = None
        self.summary = ""
        self.message = ("", "info")
        if not self.experiment:
            self.message = ("No loaded experiment has an embeddings sidecar attached -- add one on the Sources tab.", "info")
            return
        try:
            context = build_analysis_context(s.state, self.experiment)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            self.message = (f"Could not read the embeddings: {ex}", "error")
            return
        if context is None:
            self.message = (f"'{self.experiment}' has no embeddings attached.", "info")
            return
        inference = None
        if self.query not in context["row_by_uuid"]:
            inference = s.state.inference_for_uuid(self.query)
        if inference is not None:
            neighbours = nearest_to_vector(context["embeddings"], context["uuids"],
                                           inference.vector_for(self.query), k=self.k, labels=context["labels"])
        else:
            neighbours = nearest_neighbours(context["embeddings"], context["uuids"], self.query,
                                            k=self.k, labels=context["labels"])
        if not neighbours:
            self.message = (f"{self.query} has no embedding in '{self.experiment}' -- a sidecar "
                            f"covering only part of the dataset is normal, not an error.", "info")
            return
        row_by_uuid = context["row_by_uuid"]
        label_col = s.state.dataset.columns.label_col
        query_label = row_by_uuid.get(self.query, {}).get(label_col, "?")
        if inference is not None:
            classes = {n.label for n in neighbours if n.label is not None}
            top = neighbours[0]
            self.summary = (
                f"{len(neighbours)} nearest in '{self.experiment}', to a crop from the "
                f"'{inference.name}' inference (not in the dataset, so it has no label to compare "
                f"against). Nearest is {top.label} at {top.similarity:.3f}, down to "
                f"{neighbours[-1].similarity:.3f} across {len(classes)} class(es).")
        else:
            same = sum(1 for n in neighbours if n.label == str(query_label))
            self.summary = (
                f"{len(neighbours)} nearest in '{self.experiment}'. "
                f"{same}/{len(neighbours)} share this crop's label ({query_label}); "
                f"similarity {neighbours[0].similarity:.3f} down to {neighbours[-1].similarity:.3f}. "
                f"A crop whose neighbours are mostly another class is either mislabelled or "
                f"genuinely ambiguous.")
        records = []
        for neighbour in neighbours:
            record = dict(row_by_uuid.get(neighbour.uuid, {}))
            if not record:
                continue
            record["similarity"] = round(neighbour.similarity, 4)
            records.append(record)
        if not records:
            self.message = ("The neighbours are not in the loaded pool.", "info")
            return
        self.carousel = Carousel(self, s, s.state.dataset, len(records), lambda a, b: records[a:b],
                                 context["crop_col"], context["uuid_col"], highlight_uuid=self.query,
                                 review_source="similarity", label_col=label_col, chrome_height=70)

    def body(self) -> str:
        names = experiments_with_embeddings(self.session)
        pickers = (f'<div class="row">'
                   f'{h.select([(n, n) for n in names], self.experiment or "", change=self.c("experiment"), label="Embedding space", title="Similarity is measured in this run's embedding space -- two models disagree about what looks alike.")}'
                   f'{h.select([(str(k), f"{k} crops") for k in K_CHOICES], str(self.k), change=self.c("k"), label="How many")}'
                   f'</div>')
        summary = f'<div class="muted small" style="margin:6px 0">{h.e(self.summary)}</div>' if self.summary else ""
        text, kind = self.message
        msg = h.banner(text, kind) if text else ""
        return pickers + summary + msg + (self.carousel.render() if self.carousel else "")

    def on_experiment(self, value: str = "", **_):
        self.experiment = value or None
        self.session._similarity_experiment = self.experiment
        self._compute()
        self.refresh()

    def on_k(self, value: str = "", **_):
        try:
            self.k = int(value)
        except (TypeError, ValueError):
            self.k = DEFAULT_K
        self._compute()
        self.refresh()

    def on_car(self, op: str = "", **args):
        if self.carousel is not None and self.carousel.handle(op, args) is NO_RENDER:
            return NO_RENDER
        self.refresh()

    def on_review_change(self) -> None:
        self.refresh()
