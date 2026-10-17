"""Entry point: a Flet web app for visualizing deep-learning classification
results -- load a dataset CSV (crops + labels + metadata) and one or more
results CSVs (per-class scores), then explore statistics, a filterable
image mosaic, and metrics comparisons across experiments.
"""
from functools import partial
from urllib.parse import parse_qs, urlparse

import flet as ft

from lookin import __version__
from lookin._offline_assets import ensure_offline_assets
from lookin.ui.image_export import assets_root
from lookin.ui.context import AppContext
from lookin.ui.upload_view import UploadView
from lookin.ui.stats_view import StatsView
from lookin.ui.gallery_view import GalleryView
from lookin.ui.metrics_view import MetricsView
from lookin.ui.calibration_view import CalibrationView
from lookin.ui.confidence_view import ConfidenceView
from lookin.ui.compare_view import CompareView
from lookin.ui.prototypes_view import PrototypesView
from lookin.ui.projection_view import ProjectionView
from lookin.ui.cleaning_view import CleaningView
from lookin.ui.mlflow_panel import MlflowPanel
from lookin.ui.workspace_panel import WorkspacePanel
from lookin.ui.home_view import HomeView
from lookin.ui.save_status import SaveStatusIndicator
from lookin.ui import view_state as vs
from lookin.core import library
from lookin.ui.display_settings import DisplaySettings
from lookin.ui.components import ACCENT

DATA_TAB_INDEX = 0
GALLERY_TAB_INDEX = 2   # Data, Statistics, Gallery, ...
COMPARE_TAB_INDEX = 6   # ..., Metrics, Calibration, Confidence, Compare, ...


def main(page: ft.Page, dataset: str | None = None, experiments: list[tuple[str, str]] | None = None):
    page.title = "Lookin — Classification Results Viewer"
    page.padding = 0
    page.window.width = 1440
    page.window.height = 920
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed=ACCENT, use_material3=True)
    page.dark_theme = ft.Theme(color_scheme_seed=ACCENT, use_material3=True)
    page.bgcolor = ft.Colors.SURFACE

    ctx = AppContext(page)

    upload_view = UploadView(ctx, initial_dataset=dataset)
    stats_view = StatsView(ctx)
    gallery_view = GalleryView(ctx)
    metrics_view = MetricsView(ctx)
    calibration_view = CalibrationView(ctx)
    confidence_view = ConfidenceView(ctx)
    compare_view = CompareView(ctx)
    prototypes_view = PrototypesView(ctx)
    projection_view = ProjectionView(ctx)
    cleaning_view = CleaningView(ctx)

    upload_content = upload_view.build()
    stats_content = stats_view.build()
    gallery_content = gallery_view.build()
    metrics_content = metrics_view.build()
    calibration_content = calibration_view.build()
    confidence_content = confidence_view.build()
    compare_content = compare_view.build()
    prototypes_content = prototypes_view.build()
    projection_content = projection_view.build()
    cleaning_content = cleaning_view.build()

    ctx.subscribe(stats_view.refresh)
    ctx.subscribe(gallery_view.refresh)
    ctx.subscribe(metrics_view.refresh)
    ctx.subscribe(calibration_view.refresh)
    ctx.subscribe(confidence_view.refresh)
    ctx.subscribe(compare_view.refresh)
    ctx.subscribe(prototypes_view.refresh)
    ctx.subscribe(projection_view.refresh)
    ctx.subscribe(cleaning_view.refresh)

    # Attaching a sidecar or switching prototype source is not a data change
    # -- no loaded row moved -- but it does change what these two draw, so
    # they follow the registry that owns it (see ui/prototype_source.py).
    # The Embeddings tab sizes its map from the window, and the Metrics and
    # Compare tabs size their confusion matrices from it, so all three have
    # to be told when that changes -- through the context rather than by
    # taking page.on_resized, which only one view could own.
    ctx.subscribe_resize(projection_view.on_window_resized)
    ctx.subscribe_resize(metrics_view.on_window_resized)
    ctx.subscribe_resize(compare_view.on_window_resized)
    page.on_resized = lambda e: ctx.notify_resize()

    ctx.prototype_source.subscribe(prototypes_view.refresh)
    ctx.prototype_source.subscribe(projection_view.on_prototype_source_changed)

    # Changing the display LUT re-encodes every crop but changes no data,
    # so only the views that actually draw crops redraw (see
    # ctx.notify_display_change).
    ctx.subscribe_display(gallery_view.on_display_changed)
    ctx.subscribe_display(prototypes_view.refresh)

    # Renaming a loaded experiment is a relabel, not a data change -- these
    # views hold the name in caches, dropdown values and checkboxes, and
    # follow it in place rather than recomputing (see ctx.notify_rename).
    ctx.subscribe_rename(gallery_view.rename_experiment)
    ctx.subscribe_rename(metrics_view.rename_experiment)
    ctx.subscribe_rename(calibration_view.rename_experiment)
    ctx.subscribe_rename(confidence_view.rename_experiment)
    ctx.subscribe_rename(compare_view.rename_experiment)
    ctx.subscribe_rename(prototypes_view.rename_experiment)
    ctx.subscribe_rename(projection_view.rename_experiment)
    ctx.subscribe_rename(cleaning_view.rename_experiment)

    tab_specs = [
        ("Data", ft.Icons.UPLOAD_FILE, upload_content),
        ("Statistics", ft.Icons.BAR_CHART, stats_content),
        ("Gallery", ft.Icons.GRID_VIEW, gallery_content),
        ("Metrics", ft.Icons.INSIGHTS, metrics_content),
        ("Calibration", ft.Icons.RULE_OUTLINED, calibration_content),
        ("Confidence", ft.Icons.PERCENT_OUTLINED, confidence_content),
        ("Compare", ft.Icons.COMPARE_ARROWS, compare_content),
        ("Embeddings", ft.Icons.BUBBLE_CHART, projection_content),
        ("Prototypes", ft.Icons.HUB_OUTLINED, prototypes_content),
        ("Cleaning", ft.Icons.CLEANING_SERVICES_OUTLINED, cleaning_content),
    ]
    # The tab bar carries no content: given one, ft.Tabs puts it in a
    # Flutter TabBarView, which switches tab on any horizontal swipe -- a
    # two-finger trackpad gesture meant for a chart or a wide table would
    # land the user on the next tab. Flet 0.28 has no option to turn that
    # off, so the panes sit under the bar instead, one visible at a time
    # (see show_pane). Hidden rather than swapped out: a pane stays on the
    # page, so a view updating itself while its tab is not shown still works.
    tabs = ft.Tabs(
        selected_index=0,
        tabs=[ft.Tab(text=text, icon=icon) for text, icon, _ in tab_specs],
    )
    panes = [ft.Container(content, expand=True, visible=index == 0)
             for index, (_, _, content) in enumerate(tab_specs)]
    analysis = ft.Column([tabs, *panes], spacing=0, expand=1)

    # Views that build controls from the loaded experiments populate them
    # when their tab is opened rather than on every data change: a data
    # change can land in the middle of a load, and pushing a rebuilt
    # control subtree at that moment breaks Flet (see
    # CompareView.on_shown).
    on_tab_shown = {COMPARE_TAB_INDEX: compare_view.on_shown}

    def _on_tab_change(e):
        changed = show_pane(panes, tabs.selected_index)
        handler = on_tab_shown.get(tabs.selected_index)
        if handler is not None:
            handler()
        if ctx.save_status is not None:
            ctx.save_status.refresh()
        for pane in changed:
            pane.update()

    tabs.on_change = _on_tab_change

    def go_to_gallery(experiment_name: str, true_label: str, predicted_label: str):
        gallery_view.apply_external_filter(experiment_name, true_label, predicted_label)
        tabs.selected_index = GALLERY_TAB_INDEX
        show_pane(panes, GALLERY_TAB_INDEX)
        page.update()

    def go_to_data_tab():
        show_analysis()
        tabs.selected_index = DATA_TAB_INDEX
        show_pane(panes, DATA_TAB_INDEX)
        page.update()

    # What the user is looking at, so a snapshot can put it back (see
    # ui/view_state.py). Registered here rather than by each view, for the
    # same reason the jumps above are wired here: this is the one place
    # that has every view and the tab bar at once.
    ctx.view_state.register(vs.GALLERY, gallery_view)
    ctx.view_state.register(vs.METRICS, metrics_view)
    ctx.view_state.register(vs.CALIBRATION, calibration_view)
    ctx.view_state.register(vs.COMPARE, compare_view)
    ctx.view_state.register(vs.EMBEDDINGS, projection_view)
    ctx.view_state.register(vs.CLEANING, cleaning_view)

    tab_setter = make_tab_setter(tabs, panes, page, on_tab_shown)

    def set_tab_from_snapshot(index: int) -> None:
        # A snapshot is a view of the analysis tabs, so restoring one has to
        # bring them on screen -- it can be opened from the Home page.
        show_analysis()
        tab_setter(index)

    ctx.view_state.set_tab_accessors(lambda: tabs.selected_index, set_tab_from_snapshot)

    ctx.go_to_gallery = go_to_gallery
    ctx.go_to_data_tab = go_to_data_tab
    ctx.load_experiment_from_file = upload_view.add_experiment_from_path
    ctx.upload_view = upload_view

    mlflow_panel = MlflowPanel(ctx)
    # Reachable from the Workspaces window, which records the tracking URI
    # in a workspace's provenance -- as text to read later, never to fetch
    # with -- and from the Data tab's "Add from MLflow...".
    ctx.mlflow_panel = mlflow_panel
    workspace_panel = WorkspacePanel(ctx)
    # The Data tab hands it every dataset it loads, so a workspace opened
    # without one can take it (see WorkspacePanel.adopt_dataset).
    ctx.workspace_panel = workspace_panel

    # ---- Home, and the analysis tabs a workspace opens into ----
    # Two pages sharing the window rather than Home being a tab: a snapshot
    # records the selected tab by index, and inserting a tab in front would
    # have re-pointed every snapshot saved so far at the wrong one.
    home_view = HomeView(ctx, workspace_panel)
    home_content = home_view.build()
    ctx.subscribe_resize(home_view.on_window_resized)
    breadcrumb = ft.Row(spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def render_breadcrumb():
        on_home = home_content.visible
        workspace = getattr(ctx, "current_workspace", None)
        crumbs = [ft.TextButton(
            "Home", icon=ft.Icons.HOME_OUTLINED, on_click=lambda e: show_home(),
            style=ft.ButtonStyle(color=ACCENT if on_home else ft.Colors.ON_SURFACE_VARIANT),
            tooltip="Your MLflow runs and workspaces")]
        if not on_home or workspace is not None or ctx.state.dataset is not None:
            label = workspace.label() if workspace is not None else "Unsaved session"
            crumbs += [ft.Icon(ft.Icons.CHEVRON_RIGHT, size=16, color=ft.Colors.ON_SURFACE_VARIANT),
                       ft.TextButton(label, on_click=lambda e: show_analysis(),
                                     style=ft.ButtonStyle(color=None if not on_home else ft.Colors.ON_SURFACE_VARIANT),
                                     tooltip="Back to the analysis tabs")]
        breadcrumb.controls = crumbs
        if breadcrumb.page:
            breadcrumb.update()

    def show_home():
        home_content.visible, analysis.visible = True, False
        home_view.on_shown()
        render_breadcrumb()
        page.update()

    def show_analysis():
        if analysis.visible:
            return
        home_content.visible, analysis.visible = False, True
        render_breadcrumb()
        page.update()

    ctx.show_home = show_home
    ctx.show_analysis = show_analysis
    ctx.subscribe(render_breadcrumb)
    # "Is what I am looking at saved?" -- answered in the app bar, because
    # the moment it matters is the moment the user is about to open
    # something else (see ui/save_status.py).
    save_status = SaveStatusIndicator(ctx, on_click=workspace_panel.open_panel,
                                      on_save=workspace_panel.save_current)
    ctx.save_status = save_status
    ctx.subscribe(save_status.refresh)
    display_settings = DisplaySettings(ctx)
    # The same window every (i) button in the app opens, reachable without
    # first finding a chart to ask about -- the table of contents *is* the
    # manual, so opening it at the front page is opening the whole thing
    # (see ui/manual_view.py).
    manual_button = ft.IconButton(
        ft.Icons.MENU_BOOK_OUTLINED, tooltip="Reference manual",
        on_click=lambda e: ctx.manual.open(),
    )

    page.appbar = ft.AppBar(
        title=ft.Row([
            ft.Icon(ft.Icons.VISIBILITY_OUTLINED, color=ACCENT),
            ft.Text("Lookin", weight=ft.FontWeight.BOLD),
            # The running version, in the one place every tab shares. A
            # report or a snapshot can outlive the build that made it, so
            # "which Lookin is this?" has to be answerable without a
            # terminal -- this app is normally reached over a forwarded
            # port, by someone who did not start it.
            ft.Text(f"v{__version__}", size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                    tooltip=f"Lookin version {__version__}"),
            ft.Container(width=12),
            breadcrumb,
        ], spacing=10),
        center_title=False,
        bgcolor=ft.Colors.SURFACE,
        toolbar_height=56,
        actions=[ctx.busy.control, save_status.control, display_settings.indicator,
                 manual_button, ft.Container(width=8)],
    )
    # Home first, unless the launch already says what to look at: a
    # --dataset on the command line, or a ?snapshot= link.
    route = getattr(page, "route", "")
    starts_on_home = not dataset and not snapshot_id_in_route(route) and not workspace_id_in_route(route)
    home_content.visible, analysis.visible = starts_on_home, not starts_on_home
    page.add(home_content, analysis)
    if starts_on_home:
        # Starts the MLflow listing -- which a tab opened on a workspace
        # skips until its Home is actually visited (see HomeView.on_shown).
        home_view.on_shown()
    render_breadcrumb()

    # Background threads end with the *session*, not with the connection.
    # on_disconnect fires on any websocket hiccup -- a laptop waking up, a
    # gateway dropping an idle forward -- and the browser then reconnects
    # to the same session, whose threads would by then be gone for good:
    # a Home page that no longer notices edits or retries servers, and an
    # MLflow dot frozen on its last colour. on_close is the session ending.
    def on_close(e):
        home_view.stop()
        mlflow_panel.stop_polling()
        workspace_panel.stop_watching()

    page.on_disconnect = None
    page.on_close = on_close
    # A dataset set on the workspace's Home card -- here or in another tab
    # -- reaches its Data tab (see WorkspacePanel.follow_workspace).
    workspace_panel.start_watching()

    # --dataset / --experiment (see cli.py): load after the tab content is
    # already on the page, so the browser shows the UI immediately and the
    # normal progress bars/status text stream in live, same as a manual
    # click -- rather than blocking the very first render on a possibly
    # slow CSV read.
    if dataset:
        upload_view.load_dataset_now()
        for name, path in experiments or []:
            success, message = upload_view.add_experiment_from_path(path, name)
            if not success:
                print(f"[lookin] warning: could not auto-load experiment '{name}' from {path}: {message}")

    # ?snapshot=lk-7f3a2b -- the id someone pasted from a report or a chat
    # message -- or ?workspace=lkw-dv6692, which is how every Open button
    # opens a workspace: in a tab of its own (see WorkspacePanel
    # .open_in_new_tab). Loaded after the first render for the same reason
    # --dataset is: a large CSV takes tens of seconds, and the UI has to be
    # on screen with its progress bars before that starts.
    _open_from_url(page, workspace_panel)

    # A link typed or pasted into a tab that already runs Lookin does not
    # start a new session: the Flet client keeps its session id in the tab's
    # sessionStorage and reconnects to it, reporting the new address as a
    # route change. That is still someone asking for that workspace, here.
    def on_route_change(e):
        if snapshot_id_in_route(page.route) or workspace_id_in_route(page.route):
            _open_from_url(page, workspace_panel)

    page.on_route_change = on_route_change


def show_pane(panes: list, index: int) -> list:
    """Make the pane under tab `index` the one visible (see main), and
    return the panes whose visibility changed.

    Those are all a tab click has to push. page.update() would get there
    too, but it diffs every control of all ten tabs to find two flags --
    tens of milliseconds added to every click, on a switch the TabBarView
    used to do without asking the server at all.
    """
    changed = []
    for position, pane in enumerate(panes):
        if pane.visible != (position == index):
            pane.visible = position == index
            changed.append(pane)
    return changed


def make_tab_setter(tabs: ft.Tabs, panes: list, page: ft.Page, on_tab_shown: dict):
    """Build the "show this tab" function used when a snapshot is restored.

    Two things assigning selected_index does not do on its own, both of
    which made a restored view open on the wrong tab:

    * Flet does not fire on_change for a value set from Python, so the
      handler a tab needs when it is opened has to be called by hand --
      without it, a view saved on the Compare tab reopens there with empty
      pickers (see CompareView.on_shown).
    * The control has to be pushed to the client, or the switch happens
      only in the model and the browser keeps showing the old tab. This is
      the same page.update() that go_to_gallery does.
    """
    def set_tab(index: int) -> None:
        if not (0 <= index < len(tabs.tabs)):
            return
        tabs.selected_index = index
        show_pane(panes, index)
        handler = on_tab_shown.get(index)
        if handler is not None:
            handler()
        page.update()

    return set_tab


def route_param(route: str | None, name: str) -> str:
    """One query parameter of a Flet route, or "".

    page.route is parsed rather than page.query: QueryString is documented
    as internal to Page, and it only knows the URL after something has
    called it -- which at first render has not happened, so it reads an
    empty query and the link silently does nothing.

    Both routing strategies are handled: "path" gives "/?snapshot=x" and
    "hash" gives "/#/?snapshot=x", so the fragment is stripped first.
    """
    route = (route or "").split("#", 1)[-1] if "#" in (route or "") else (route or "")
    query = parse_qs(urlparse(route).query)
    return (query.get(name) or [""])[0].strip()


def snapshot_id_in_route(route: str | None) -> str:
    return route_param(route, "snapshot")


def workspace_id_in_route(route: str | None) -> str:
    return route_param(route, "workspace")


def _open_from_url(page: ft.Page, workspace_panel) -> None:
    """Open the snapshot or workspace named in the page's URL, if any.

    Never fatal: a stale or mistyped id leaves the app exactly as it would
    have started, with a line in the log. Someone opening a link should
    land in Lookin either way.
    """
    route = getattr(page, "route", "")
    snapshot_id = snapshot_id_in_route(route)
    if snapshot_id:
        found = library.find_snapshot(snapshot_id)
        if found is None:
            print(f"[lookin] no snapshot {snapshot_id!r} in the library at {library.library_root()}", flush=True)
            return
        workspace, snapshot = found
        workspace_panel.open_from_link(workspace, snapshot)
        return
    workspace_id = workspace_id_in_route(route)
    if workspace_id:
        workspace = next((w for w in library.list_workspaces() if w.id == workspace_id), None)
        if workspace is None:
            print(f"[lookin] no workspace {workspace_id!r} in the library at {library.library_root()}", flush=True)
            return
        workspace_panel.open_from_link(workspace)


def run_ui(
    host: str = "127.0.0.1", port: int = 8550,
    dataset: str | None = None, experiments: list[tuple[str, str]] | None = None,
    library_dir: str | None = None, enable_mcp: bool = True,
) -> None:
    """Launch the Flet web app -- called by `lookin ui` (see cli.py).

    The bundled Flet web client is a CanvasKit-only build (no HTML-renderer
    variant is compiled into it), and CanvasKit normally streams its
    Skia/WASM runtime from Google's CDN on page load. ensure_offline_assets()
    patches the installed Flet web assets, once, to serve a locally bundled
    copy of CanvasKit instead -- see lookin/_offline_assets.py for why.
    """
    ensure_offline_assets()
    # Where saved workspaces live. Set once here rather than read per call, so
    # every session of this server agrees -- including a --library pointed
    # at a shared mount, which is all it takes to open a colleague's workspace
    # by id instead of by sending them a file.
    if library_dir:
        library.set_library_root(library_dir)
    target = partial(main, dataset=dataset, experiments=experiments)
    # Served here rather than through ft.app() so the same server can carry
    # the MCP endpoint next to the UI (see lookin/mcp_server.py).
    import uvicorn
    from flet.utils import is_linux_server, open_in_browser

    from lookin.mcp_server import MCP_PATH, build_asgi_app

    # assets_dir is what lets a chart be opened as a real PNG in a browser
    # tab (see ui/image_export.py). It has to exist before this call and be
    # absolute, on both counts because flet_web checks once and silently
    # ignores a directory that fails either test.
    app = build_asgi_app(target, assets_dir=str(assets_root()), enable_mcp=enable_mcp)
    url = f"http://{'127.0.0.1' if host in ('', '0.0.0.0', '*') else host}:{port}"
    print(f"[lookin] web interface: {url}", flush=True)
    if enable_mcp:
        print(f"[lookin] MCP endpoint for AI agents: {url}{MCP_PATH}", flush=True)
    # What ft.app() did for a web view: open a tab, except on a headless
    # Linux server where there is no browser to open.
    if not is_linux_server():
        open_in_browser(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_ui()
