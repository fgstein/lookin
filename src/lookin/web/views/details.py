"""The details panel of the full-size view: a crop's dataset row, its
review decision and what every loaded experiment predicted for it.

A zoomable crop carries `data-info="<details_url>"`; the (i) button of the
zoom layer (static/app.js, openZoom) fetches that URL only when the panel
is opened, and /info (server.py) answers with `details_html`. Nothing is
computed for a crop nobody asks about.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlencode

from lookin.ui.components import color_for_experiment
from lookin.web import html as h


def details_url(session, uid=None, crop=None) -> str:
    """The URL of a crop's details, by uuid when it has one, else by the
    value of its crop column. Empty when there is neither."""
    if uid is not None and str(uid) != "":
        return f"/info/{session.sid}?{urlencode({'u': str(uid)})}"
    if crop is not None and str(crop) != "":
        return f"/info/{session.sid}?{urlencode({'c': str(crop)})}"
    return ""


def _value(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return '<span class="muted">-</span>'
    if isinstance(v, float):
        return h.e(f"{v:.4g}")
    return h.e(str(v))


def _prediction(entry: dict, color: str) -> str:
    name = entry["experiment"]
    head = f'<div class="zi-exp">{h.dot(color, size=10)}<b class="xname" title="{h.e(name)}">{h.e(name)}</b></div>'
    if not entry["found"]:
        return f'<div class="zi-pred">{head}<span class="muted small">No score for this crop</span></div>'
    correct = entry["correct"]
    mark = "" if correct is None else (
        f'<span class="zi-ok">{h.icon("check", 12)} correct</span>' if correct
        else f'<span class="zi-bad">{h.icon("x", 12)} wrong</span>')
    score = "" if entry["score"] is None else f' <span class="mono small">{entry["score"]:.3f}</span>'
    predicted = "-" if entry["predicted"] is None else str(entry["predicted"])
    lines = [f'<div class="zi-line"><span class="muted small">Predicted</span>'
             f'<b>{h.e(predicted)}</b>{score}{mark}</div>']
    if entry["truth_score"] is not None and not correct:
        lines.append(f'<div class="zi-line"><span class="muted small">True class score</span>'
                     f'<span class="mono small">{entry["truth_score"]:.3f}</span></div>')
    if len(entry["top"]) > 1:
        top = ", ".join(f"{label} {score:.3f}" for label, score in entry["top"])
        lines.append(f'<div class="zi-line"><span class="muted small">Top {len(entry["top"])}</span>'
                     f'<span class="small">{h.e(top)}</span></div>')
    return f'<div class="zi-pred">{head}{"".join(lines)}</div>'


def details_html(session, uid: Optional[str] = None, crop: Optional[str] = None) -> str:
    state = session.state
    info = state.crop_details(uid=uid, crop=crop)
    if info is None:
        return h.banner("This crop is not in the loaded dataset.", "info")
    parts = []
    label = info["label"]
    title = info["uuid"] or (str(crop) if crop else "crop")
    sub = f'<div class="small">True class: <b>{_value(label)}</b></div>' if label is not None else ""
    if info["reference"] is not None and str(info["reference"]) != str(label):
        sub += f'<div class="small">Reviewed class: <b>{_value(info["reference"])}</b></div>'
    parts.append(f'<div class="zi-head"><b class="mono">{h.e(title)}</b>{sub}</div>')

    if info["predictions"]:
        slots = state.color_slots
        preds = "".join(_prediction(p, color_for_experiment(p["experiment"], slots)) for p in info["predictions"])
        parts.append(f'<div class="zi-sec"><div class="zi-t">Predictions</div>{preds}</div>')

    decision = info["decision"]
    if decision is not None:
        rows = [("Status", h.e(decision.status))]
        rows += [(k, h.e(v)) for k, v in (("New label", decision.new_label), ("Reason", decision.reason),
                                          ("Note", decision.note), ("By", decision.author),
                                          ("On", decision.timestamp)) if v]
        parts.append(f'<div class="zi-sec"><div class="zi-t">Review</div>{h.kv_grid(rows)}</div>')

    meta = h.kv_grid([(k, _value(v)) for k, v in info["metadata"]])
    parts.append(f'<div class="zi-sec"><div class="zi-t">Metadata</div>{meta}</div>')
    return "".join(parts)
