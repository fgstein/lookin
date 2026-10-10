"""Glue between the Flet page and the data model: holds the AppState and
lets each tab register a refresh callback that fires whenever data changes
(a dataset or experiment is loaded/removed, or the class mapping edited).
"""
from __future__ import annotations

from typing import Callable, Optional

import flet as ft

from lookin.core.data_manager import AppState
from lookin.ui.components import LUT_NONE, ROLL_NONE, BusyIndicator, ImageZoomDialog
from lookin.ui.prototype_source import PrototypeSourceRegistry


class AppContext:
    def __init__(self, page: ft.Page):
        self.page = page
        self.state = AppState()
        self._listeners: list[Callable[[], None]] = []
        # Renaming an experiment is deliberately *not* a data change (see
        # notify_rename): its own listener list, so views can relabel
        # themselves without being marked stale.
        self._rename_listeners: list[Callable[[str, str], None]] = []
        # A review decision changes almost nothing about the loaded data --
        # it doesn't invalidate a merge unless the review overlay is on
        # (see AppState._review_stamp), and it must not mark Statistics,
        # Gallery or Metrics stale, or reviewing one crop would ask the
        # user to recompute minutes of charts. Its own listener list, so
        # only the views that actually show review state redraw.
        self._review_listeners: list[Callable[[], None]] = []
        # The window size is a layout input for any view that sizes a
        # fixed viewport from it (see ui/projection_view.py). Its own
        # listener list rather than each view claiming page.on_resized,
        # which only one of them could hold.
        self._resize_listeners: list[Callable[[], None]] = []
        # Changing the display LUT invalidates every crop already encoded on
        # screen, but changes no data -- see notify_display_change.
        self._display_listeners: list[Callable[[], None]] = []
        # Which prototype source is active per experiment, what has been
        # computed from it, and the sidecar loading that invalidates both.
        # Shared rather than owned by a view: the Prototypes tab picks the
        # source, the Embeddings tab's projection draws it, and the Data tab
        # and MLflow panel both attach sidecars (see ui/prototype_source.py).
        self.prototype_source = PrototypeSourceRegistry(self.state)
        # One full-size zoomable viewer shared by every chart and thumbnail
        # in the app -- see components.ImageZoomDialog and
        # components.fig_to_zoomable_image.
        self.image_zoom = ImageZoomDialog(self)
        # Every (i) button in the app opens this one window (see
        # ui/manual_view.py and components.info_button). Shared for the same
        # reason the viewer above is: the same explanation of F1 is reached
        # from the Metrics tab, the Slices panel and the Compare tab, and
        # three copies of it would be three pages to keep in step.
        from lookin.ui.manual_view import ManualView
        self.manual = ManualView(self)
        # "Is anything still loading?", answered from every tab. Shared for
        # the same reason as the viewer above: the CSV reads on the Data tab
        # and the rebuilds behind every "Update now" all light the same
        # app-bar indicator (see components.BusyIndicator).
        self.busy = BusyIndicator()
        # "What else looks like this to the model?" -- reachable from every
        # crop the app shows, so it lives here rather than in a tab (see
        # ui/similarity.py).
        from lookin.ui.similarity import SimilaritySearch
        self.similarity = SimilaritySearch(self)
        # How crops are mapped to the screen. A display setting, never a
        # change to any file -- and a session-wide one, because a LUT that
        # applied in the carousel but not in the gallery would mean two
        # views of the same crop disagreeing about what it looks like.
        self.display_lut = LUT_NONE
        # Which dataset column carries each crop's roll, which decides the
        # quarter turn that cancels it (see components.quarter_turns_for_roll).
        # Session-wide for the same reason as the LUT above, and empty by
        # default: no dataset is required to have such a column, and a
        # rotation nobody asked for is a worse default than none.
        self.display_roll_col = ROLL_NONE
        # What the user is looking at, captured and put back -- the other
        # half of a saved workspace, which only records what was *loaded*
        # (see ui/view_state.py and core/library.py). Populated by app.py
        # once every view exists, for the same reason the jumps below are.
        from lookin.ui.view_state import ViewStateRegistry
        self.view_state = ViewStateRegistry(self)
        # Which library workspace this session was opened from (or last saved
        # to), so that exporting a report can bookmark the view it was
        # produced from and stamp that id into it. None until there is one.
        self.current_workspace = None
        # The snapshot this session was opened from, or last saved as. What
        # the app bar compares the current state against to say whether
        # anything is unsaved (see ui/save_status.py).
        self.current_snapshot = None
        # The state as it was at the last save or load -- what "has
        # anything changed since?" is measured against (ui/save_status.py).
        self.view_state_baseline = None
        # Set by app.py: the app-bar chips that answer "is this saved?".
        self.save_status = None
        # Set by app.py: the MLflow panel, read only for the tracking URI a
        # workspace records as provenance -- text to read later, never used to
        # fetch anything.
        self.mlflow_panel = None
        # Wired up by app.py once every view exists, so tabs can jump to
        # each other (e.g. a confusion-matrix cell opening the gallery, or
        # the MLflow panel switching to the Data tab after a load).
        self.go_to_gallery: Optional[Callable[[str, str, str], None]] = None
        self.go_to_data_tab: Optional[Callable[[], None]] = None
        self.load_experiment_from_file: Optional[Callable[[str, str], "tuple[bool, str]"]] = None
        # Set by app.py once every view exists, so the MLflow panel can
        # keep the Data tab's sidecar fields in sync after an auto-attach.
        self.upload_view = None

    def subscribe(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def subscribe_rename(self, callback: Callable[[str, str], None]) -> None:
        """Register a view's (old_name, new_name) handler -- see notify_rename."""
        self._rename_listeners.append(callback)

    def notify_rename(self, old_name: str, new_name: str) -> None:
        """An experiment was renamed. Deliberately separate from
        notify_change(): the loaded rows, scores and computed metrics are
        all bit-for-bit the same, only the label changed, so marking every
        view stale would ask the user to re-run minutes of computation to
        fix a typo. Each view instead remaps its own name-keyed state
        (caches, dropdown values, checkboxes) in place and re-renders from
        what it already has.
        """
        # The registry is re-keyed before any view runs: a view's rename
        # handler re-renders from it, so it has to already answer to the new
        # name by then.
        self.prototype_source.rename_experiment(old_name, new_name)
        for callback in self._rename_listeners:
            callback(old_name, new_name)
        self.page.update()

    def subscribe_review(self, callback: Callable[[], None]) -> None:
        """Register a view's "a review decision was made" handler -- see
        notify_review."""
        self._review_listeners.append(callback)

    def notify_review(self) -> None:
        for callback in self._review_listeners:
            callback()
        self.page.update()

    def subscribe_display(self, callback: Callable[[], None]) -> None:
        """Register a view's "the display LUT changed" handler."""
        self._display_listeners.append(callback)

    def notify_display_change(self) -> None:
        """The LUT or the roll correction changed, so every crop already
        encoded on screen is stale. Deliberately its own listener list: no
        row moved and no number changed, so marking Statistics or Metrics
        dirty would ask the user to recompute minutes of charts because
        they adjusted a contrast setting or straightened a thumbnail.
        """
        for callback in self._display_listeners:
            callback()
        self.page.update()

    def subscribe_resize(self, callback: Callable[[], None]) -> None:
        """Register a view's "the window changed size" handler."""
        self._resize_listeners.append(callback)

    def notify_resize(self) -> None:
        for callback in self._resize_listeners:
            callback()
        self.page.update()

    def notify_change(self) -> None:
        # Rebuilding Statistics/Gallery/Metrics is expensive at scale, and
        # neither "on every data change" nor "the moment you switch tabs"
        # scales well once there's real data behind them -- each of those
        # views instead just marks itself stale here and shows its own
        # "data changed -- click to update" banner, and only actually
        # rebuilds when the user clicks it (see each view's refresh()).
        for callback in self._listeners:
            callback()
        self.page.update()
