"""Addresses of things in `lookin web`, for a person to open.

    /?workspace=lkw-...                      a workspace, in a tab of its own
    /?snapshot=lk-...                        a saved view, restored
    ...&tab=metrics                          then this tab
    ...&crop=<uuid>[&experiment=<name>]      then this crop, in the viewer,
                                             with that experiment's prediction

Built here rather than in the web package because the MCP endpoint hands
them out too: an agent's "42 cats taken for dogs" is worth more with the
crops one click away. The parameter names are a format -- links get pasted
into reports and tickets -- so they are never renamed.
"""
from __future__ import annotations

from urllib.parse import urlencode

WORKSPACE = "workspace"
SNAPSHOT = "snapshot"
TAB = "tab"
CROP = "crop"
EXPERIMENT = "experiment"

# The server's own address, as the person reading a link reaches it. Set
# once at startup (`lookin web --public-url`, or the address it binds);
# empty, links are relative ("/?workspace=...").
_base_url = ""


def set_base_url(url: str) -> None:
    global _base_url
    _base_url = (url or "").rstrip("/")


def base_url() -> str:
    return _base_url


def link(workspace_id: str = "", snapshot_id: str = "", tab: str = "", crop: str = "",
         experiment: str = "") -> str:
    """The address that opens a workspace (or a snapshot of it), optionally
    on one tab and with one crop open. A snapshot carries its workspace, so
    `workspace_id` is left out when `snapshot_id` is given."""
    params = [(SNAPSHOT, snapshot_id)] if snapshot_id else [(WORKSPACE, workspace_id)]
    if tab:
        params.append((TAB, tab))
    if crop:
        params.append((CROP, str(crop)))
        if experiment:
            params.append((EXPERIMENT, experiment))
    return f"{_base_url}/?{urlencode(params)}"
