"""Putting a session on the page: boot, action dispatch, and what goes back.

The page has a few regions, each with a stable id the runtime morphs:

    chrome-nav    the workspace sidebar (analysis mode)
    chrome-top    the bar above the current view
    view-<name>   the current view's body
    dialogs       the modal layer

`collect()` answers every request with the regions the request changed
(see Session.mark) and the client commands it queued (toasts, new tabs...).
"""
from __future__ import annotations

import traceback

from lookin import __version__
from lookin.web import html as h
from lookin.web.views.base import Handler

# The analysis tabs, in the order the sidebar shows them.
TABS = (
    ("sources", "Sources", "Data"),
    ("filter", "Filter", "Data"),
    ("statistics", "Statistics", "Data"),
    ("gallery", "Gallery", "Data"),
    ("metrics", "Metrics", "Evaluate"),
    ("calibration", "Calibration", "Evaluate"),
    ("confidence", "Confidence", "Evaluate"),
    ("compare", "Compare", "Evaluate"),
    ("embeddings", "Embeddings", "Representations"),
    ("prototypes", "Prototypes", "Representations"),
    ("cleaning", "Cleaning", "Curate"),
)
TAB_KEYS = [t[0] for t in TABS]
TAB_TITLES = {t[0]: t[1] for t in TABS}

# A snapshot records the selected tab *by index*, in the order the original
# UI numbered its tabs (see lookin.ui.view_state) -- a file format shared
# with `lookin ui`, so it does not move when a tab is added here. A tab the
# original UI does not have is saved as the one it sits under.
SNAPSHOT_TABS = ("sources", "statistics", "gallery", "metrics", "calibration", "confidence",
                 "compare", "embeddings", "prototypes", "cleaning")
_SNAPSHOT_ALIASES = {"filter": "sources"}


def tab_index(session) -> int:
    tab = _SNAPSHOT_ALIASES.get(session.tab, session.tab)
    return SNAPSHOT_TABS.index(tab) if tab in SNAPSHOT_TABS else 0


def set_tab_index(session, index: int) -> None:
    if 0 <= index < len(SNAPSHOT_TABS):
        session.mode = "analysis"
        session.tab = SNAPSHOT_TABS[index]
        session.mark("page")


def current_view_name(session) -> str:
    return "home" if session.mode == "home" else session.tab


# ---- boot -------------------------------------------------------------------

def _wire(session) -> None:
    """Register every view's state with the snapshot registry -- the same
    keys and the same order as the original UI (see app.py there)."""
    from lookin.ui import view_state as vs

    # First: it changes which rows every view below restores against.
    session.view_state.register(vs.DATASET_FILTER, session.view("filter"))
    session.view_state.register(vs.GALLERY, session.view("gallery"))
    session.view_state.register(vs.METRICS, session.view("metrics"))
    session.view_state.register(vs.CALIBRATION, session.view("calibration"))
    session.view_state.register(vs.COMPARE, session.view("compare"))
    session.view_state.register(vs.EMBEDDINGS, session.view("embeddings"))
    session.view_state.register(vs.CLEANING, session.view("cleaning"))
    session.view_state.set_tab_accessors(lambda: tab_index(session), lambda i: set_tab_index(session, i))


def boot(session, query: dict) -> None:
    """First contact of a tab with its session -- or a reload of it.

    A reload finds the session as it was and only redraws it. A new session
    gets what the command line asked for (--dataset / --experiment), then
    whatever the address asks for (?workspace= / ?snapshot=).
    """
    from lookin.core import library as lib

    session.mark("page")
    if session.booted:
        _route(session, query)
        return
    session.booted = True
    _wire(session)
    from lookin.web.views.workspaces import start_watching
    start_watching(session)
    sources = session.view("sources")
    app = session.app
    workspace_id = (query.get("workspace") or [""])[0]
    snapshot_id = (query.get("snapshot") or [""])[0]
    if app.dataset:
        session.mode = "analysis"
        session.tab = "sources"
        sources.set_dataset_path(app.dataset)
        sources.load_dataset_now(then_experiments=list(app.experiments))
    if not app.dataset and not workspace_id and not snapshot_id:
        session.mode = "home"
    _route(session, query)


def _route(session, query: dict) -> None:
    """?snapshot=lk-... or ?workspace=lkw-... -- the id someone pasted, or
    how every Open button opens a workspace: in a tab of its own."""
    from lookin.core import library as lib

    snapshot_id = (query.get("snapshot") or [""])[0].strip()
    workspace_id = (query.get("workspace") or [""])[0].strip()
    if not snapshot_id and not workspace_id:
        return
    # Already showing it: a reload of an open workspace must not reload it.
    current = session.current_workspace
    if snapshot_id and session.current_snapshot is not None and session.current_snapshot.id == snapshot_id:
        return
    if workspace_id and not snapshot_id and current is not None and current.id == workspace_id:
        return
    # Still opening it (a reload during a long load): see open_from_link.
    opening = getattr(session, "_opening", None)
    if opening is not None and opening[1] == snapshot_id and (snapshot_id or opening[0] == workspace_id):
        return
    from lookin.web.views.workspaces import open_from_link

    if snapshot_id:
        found = lib.find_snapshot(snapshot_id)
        if found is None:
            session.toast(f"No snapshot '{snapshot_id}' in the library at {lib.library_root()}.", "error")
            return
        workspace, snapshot = found
        open_from_link(session, workspace, snapshot)
        return
    workspace = next((w for w in lib.list_workspaces() if w.id == workspace_id), None)
    if workspace is None:
        session.toast(f"No workspace '{workspace_id}' in the library.", "error")
        return
    open_from_link(session, workspace, None)


# ---- dispatch ---------------------------------------------------------------

def dispatch(session, target: str, action: str, args: dict) -> None:
    from lookin.web.views.base import NO_RENDER

    if target == "app":
        AppActions(session).handle(action, args)
        return
    if target.startswith("dlg:"):
        dialog = session.dialog(target[4:])
        if dialog is None:
            session.mark("dialogs")
            return
        if dialog.handle(action, args) is not NO_RENDER:
            session.mark("dialogs", "chrome")
        return
    from lookin.web.views import VIEW_CLASSES

    if target not in VIEW_CLASSES:
        raise KeyError(f"unknown target {target}")
    if session.view(target).handle(action, args) is NO_RENDER:
        return
    # Whatever a view's action changed is in its own region -- and in the
    # save chips, which compare the view's state with the saved one.
    session.mark(target, "chrome")


# ---- what goes back --------------------------------------------------------

def collect(session, full: bool = False) -> dict:
    dirty = set(session._pending_dirty)
    session._pending_dirty.clear()
    cmds = list(session._pending_cmds)
    session._pending_cmds.clear()
    out: dict = {"html": {}, "cmds": cmds, "view": current_view_name(session),
                 "mode": session.mode, "tab": session.tab}
    if full or "page" in dirty:
        out["page"] = render_page(session)
        out["html"]["dialogs"] = render_dialogs(session)
        out["title"] = session.title
        return out
    name = current_view_name(session)
    if "*" in dirty or name in dirty:
        out["html"][f"view-{name}"] = render_view(session, name)
    if "*" in dirty or "chrome" in dirty or name in dirty:
        if session.mode == "analysis":
            out["html"]["chrome-nav"] = render_nav(session)
        out["html"]["chrome-top"] = render_top(session)
    if "*" in dirty or "dialogs" in dirty:
        out["html"]["dialogs"] = render_dialogs(session)
    out["title"] = session.title
    return out


def render_view(session, name: str) -> str:
    try:
        return session.view(name).render()
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        return h.banner(f"This view failed to render: {type(ex).__name__}: {ex}", "error")


def render_dialogs(session) -> str:
    parts = []
    for d in session.dialogs:
        try:
            parts.append(d.render())
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            parts.append(f'<div class="dlg-back"><div class="dlg">{h.banner(str(ex), "error")}</div></div>')
    return "".join(parts)


def render_page(session) -> str:
    name = current_view_name(session)
    body = render_view(session, name)
    if session.mode == "home":
        return (f'<div class="home-page"><header id="chrome-top" class="topbar">{render_top(session)}</header>'
                f'<div id="view-home" class="view">{body}</div></div>')
    return (f'<div class="shell"><nav id="chrome-nav" class="nav" aria-label="Workspace">{render_nav(session)}</nav>'
            f'<div class="main"><header id="chrome-top" class="topbar">{render_top(session)}</header>'
            f'<div id="view-{h.e(name)}" class="view" data-view="{h.e(name)}">{body}</div></div></div>')


def _brand() -> str:
    return (f'<span class="brand">{h.icon("eye", 18, "c-acc")}<b>Lookin</b>'
            f'<span class="ver mono">v{h.e(__version__)}</span></span>')


def render_nav(session) -> str:
    from lookin.core import board as bd

    ws = session.current_workspace
    if ws is not None:
        group = ""
        try:
            column = bd.column_of(bd.load(), ws.id)
            group = column["name"] if column else ""
        except Exception:  # noqa: BLE001
            group = ""
        ws_block = (f'<button type="button" class="ws-block"{h.act("app", "workspaces")} title="Workspaces and snapshots">'
                    f'<span class="ws-g">{h.e(group or "Workspace")}</span>'
                    f'<span class="ws-n">{h.e(ws.label())}</span><span class="ws-i mono">{h.e(ws.id)}</span></button>')
    else:
        ws_block = (f'<button type="button" class="ws-block"{h.act("app", "workspaces")} title="Workspaces and snapshots">'
                    f'<span class="ws-g">Not saved</span><span class="ws-n">Unsaved session</span>'
                    f'<span class="ws-i">Save it as a workspace…</span></button>')
    items, group = [], None
    cleaning_badge = ""
    try:
        cleaning_badge = session.view("cleaning").nav_badge() if "cleaning" in session.views else ""
    except Exception:  # noqa: BLE001
        cleaning_badge = ""
    for key, title, grp in TABS:
        if grp != group:
            items.append(f'<div class="nav-h{" first" if group is None else ""}">{h.e(grp)}</div>')
            group = grp
        on = session.mode == "analysis" and session.tab == key
        badge = f'<span class="nav-badge">{h.e(cleaning_badge)}</span>' if key == "cleaning" and cleaning_badge else ""
        if key == "filter":
            badge = filter_badge(session)
        items.append(f'<a href="?tab={key}" class="nav-a{" on" if on else ""}" data-tab="{key}"'
                     f'{" aria-current=page" if on else ""}>{h.e(title)}{badge}</a>')
    from lookin.web.views.display import display_label
    foot = (f'<div class="nav-foot">'
            f'<button type="button" class="nav-a"{h.act("app", "display")}>Display<span class="nav-v">{h.e(display_label(session))}</span></button>'
            f'<button type="button" class="nav-a"{h.act("app", "manual", page="")}>Reference manual</button>'
            f'<span class="nav-ver mono">v{h.e(__version__)}</span></div>')
    return (f'<a href="/" class="nav-brand" data-home="1">{_brand()}<span class="nav-home">← Home</span></a>'
            f'{ws_block}{"".join(items)}{foot}')


def render_top(session) -> str:
    busy = session.busy.running
    busy_html = (f'<span class="busy" title="{h.e(", ".join(busy))}"><span class="spinner"></span>'
                 f'{h.e(busy[-1])}{f" (+{len(busy) - 1})" if len(busy) > 1 else ""}</span>') if busy else ""
    if session.mode == "home":
        return (f'{_brand()}<div class="top-search">{session.view("home").search_box()}</div>'
                f'<div class="top-r">{busy_html}'
                f'{h.button("Manual", act_attr=h.act("app", "manual", page=""), icon_name="book", small=True)}'
                f'{h.button("New workspace", act_attr=h.act("home", "new_workspace"), kind="primary", small=True)}</div>')
    title = TAB_TITLES.get(session.tab, "")
    return (f'<h1>{h.e(title)}</h1>'
            f'<div class="top-r">{busy_html}{filter_chip(session)}{save_chips(session)}'
            f'{h.button("Save snapshot", act_attr=h.act("app", "save_snapshot"), small=True, icon_name="bookmark")}</div>')


def filter_badge(session) -> str:
    """The Filter tab's nav badge: how many rows are in view, while a filter
    removes some."""
    state = session.state
    if state.dataset is None:
        return ""
    if state.dataset_filter_error:
        return '<span class="nav-badge warn">!</span>'
    if not state.dataset.is_filtered():
        return ""
    return f'<span class="nav-badge info">{len(state.dataset.df):,}</span>'


def filter_chip(session) -> str:
    """Shown on every tab while the Filter tab removes rows: every number
    below is on fewer rows than the file has, and that must not be
    forgotten two tabs later."""
    from lookin.core.dataset_filter import describe

    state = session.state
    if state.dataset is None:
        return ""
    go = h.act("app", "tab", tab="filter")
    if state.dataset_filter_error:
        return (f'<button type="button" class="sv warn"{go} title="{h.e(state.dataset_filter_error)}">'
                f'{h.icon("filter", 13)}Filter not applied</button>')
    if not state.dataset.is_filtered():
        return ""
    tip = "Rows kept by the Filter tab:\n" + "\n".join(describe(state.dataset_filter))
    return (f'<button type="button" class="sv filt"{go} title="{h.e(tip)}">{h.icon("filter", 13)}'
            f'{len(state.dataset.df):,} / {len(state.dataset.full_df):,} rows</button>')


def save_chips(session) -> str:
    from lookin.ui import save_status

    st = save_status.compute(session)

    def chip(saved, name, saved_word, unsaved_word, saved_tip, unsaved_tip, icon_name):
        if not st.has_data:
            return f'<button type="button" class="sv"{h.act("app", "workspaces")} title="Nothing loaded yet.">{h.icon(icon_name, 13)}--</button>'
        if saved:
            label = save_status._elide(name) if name else saved_word
            tip = f"{saved_tip} ({name})" if name else saved_tip
            return f'<button type="button" class="sv ok"{h.act("app", "workspaces")} title="{h.e(tip)}">{h.icon(icon_name, 13)}{h.e(label)}</button>'
        label = f"{save_status._elide(name)} *" if name else unsaved_word
        tip = f"{unsaved_tip} (last: {name})" if name else unsaved_tip
        return f'<button type="button" class="sv warn"{h.act("app", "workspaces")} title="{h.e(tip)}">{h.icon(icon_name, 13)}{h.e(label)}</button>'

    # The one-click answer to the workspace chip turning amber: shown only
    # while there is something to save. Snapshots stay a deliberate act in
    # the Workspaces window -- each is an id a report may quote.
    save = ""
    if st.has_data and not st.workspace_saved:
        if st.workspace_name:
            save = h.button("Save", act_attr=h.act("app", "save_workspace"), kind="primary", small=True, icon_name="save",
                            title=f"Write the dataset and experiments loaded now into '{st.workspace_name}'. "
                                  "Its id and its snapshots are kept.")
        else:
            save = h.button("Save as workspace", act_attr=h.act("app", "save_workspace"), kind="primary", small=True,
                            icon_name="save", title="Keep the dataset and experiments loaded now as a named workspace.")
    return save + (chip(st.workspace_saved, st.workspace_name, "Workspace", "No workspace",
                 "This data is saved as a workspace.",
                 "This data is not saved as a workspace -- loading something else loses the mount.", "save")
            + chip(st.snapshot_saved, st.snapshot_name, "Snapshot", "No snapshot",
                   "What is on screen matches this snapshot.",
                   "The filters and settings on screen are not saved in any snapshot.", "bookmark"))


# ---- app-level actions -----------------------------------------------------

class AppActions(Handler):
    """The "app" target -- dispatched like a view's (see Handler.handle)."""

    target = "app"

    def __init__(self, session):
        self.s = session

    def on_tab(self, tab: str = "sources", **_):
        if tab not in TAB_KEYS:
            return
        self.s.mode = "analysis"
        previous = self.s.tab
        self.s.tab = tab
        view = self.s.view(tab)
        shown = getattr(view, "on_shown", None)
        if shown is not None:
            shown()
        # data changed since this tab was drawn: update it now (StaleTracking)
        arrival = getattr(view, "on_arrival", None)
        if arrival is not None:
            arrival()
        self.s.mark("page" if previous != tab else tab)

    def on_home(self, **_):
        self.s.mode = "home"
        home = self.s.view("home")
        home.on_shown()
        self.s.mark("page")

    def on_analysis(self, **_):
        self.s.mode = "analysis"
        self.s.mark("page")

    def on_manual(self, page: str = "", **_):
        from lookin.web.views.manual import open_manual
        open_manual(self.s, page or None)

    def on_display(self, **_):
        from lookin.web.views.display import DisplayDialog
        self.s.open_dialog(DisplayDialog())

    def on_workspaces(self, **_):
        from lookin.web.views.workspaces import WorkspacesDialog
        self.s.open_dialog(WorkspacesDialog(self.s))

    def on_save_workspace(self, **_):
        """The app bar's Save: write what is loaded into the workspace open
        in this tab -- or, with none open, make it one."""
        from lookin.core import library as lib
        from lookin.ui import save_status
        from lookin.web.views.workspaces import WorkspacesDialog

        workspace = self.s.current_workspace
        mount = self.s.upload_view.current_mount()
        if mount is None:
            return
        if workspace is None or not workspace.exists():
            self.s.open_dialog(WorkspacesDialog(self.s, focus_save=True))
            return
        fresh = lib.update_workspace_mount(workspace.path, mount)
        if fresh is None:
            self.s.toast("Could not write the workspace.", "error")
            return
        self.s.current_workspace = fresh
        save_status.mark_baseline(self.s)
        self.s.toast(f"Updated workspace {fresh.label()} ({fresh.id}).", "ok")
        self.s.mark("chrome")

    def on_save_snapshot(self, **_):
        from lookin.web.views.workspaces import WorkspacesDialog
        self.s.open_dialog(WorkspacesDialog(self.s, focus_save=True))

    def on_noop(self, **_):
        return None
