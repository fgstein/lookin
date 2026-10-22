"""What the user was looking at, captured and put back.

A workspace records which data was loaded (see core/library.py). This records
the rest: which tab, which filters, which projection settings, which pair
was being compared. Together they are what makes "open lk-7f3a2b" show the
same thing six months later, or on a colleague's machine.

Why a registry rather than one big method
-----------------------------------------
Each view owns the shape of its own state, because each view is the only
place that knows which of its attributes are a *choice* the user made and
which are derived. So every view implements two methods:

    view_state() -> dict                  what the user chose
    restore_view_state(dict) -> list[str] apply it, and say what could not be

and this module only decides the order and collects the warnings.

The two rules that are not optional
-----------------------------------
**Nothing here may raise.** A snapshot arriving from another machine
describes data that may have moved, been relabelled or been reloaded with
different columns. A view that cannot apply its state skips it; a view that
throws is caught here. Failing to restore a filter must never be what stops
a dataset opening.

**A partial restore has to say so.** A view restored to 90% of what it was
is worse than one not restored at all, because the user believes they are
looking at the same thing. Every method returns the list of what it could
not apply, and the caller shows it.

Restoring recomputes
--------------------
Everywhere else in the app an expensive rebuild waits behind a banner the
user clicks, because a data change is not a request to spend minutes (see
Session.notify_change). Opening a snapshot *is* that request: it is an
explicit "show me this again", and a restored Embeddings tab that comes back
as an empty map under a "Run projection" button is not the view that was
saved. The filters it carries would also have nothing to act on.

So restore() puts the controls back, switches to the tab, and then asks each
restored view to recompute through recompute_after_restore(). Those runs go
through each view's own background job (RebuildView.rebuild), so they run off the
request thread and report progress exactly as a manual click would -- several at
once is fine, each view has its own job queue.

Keys
----
Everything written here is a technical key stored in files that outlive any
wording of the UI, and compared on the way back in. They are never
translated and never reworded (see CLAUDE.md). One in particular is a trap:
the Embeddings tab's colour dropdown is keyed *positionally* ("0", "1", ...)
with the real column name held aside, so what goes in a snapshot is the
column name -- a stored "3" would point at whatever happened to be third
next time.
"""
from __future__ import annotations

from typing import Callable, Optional

# Which view a block of state belongs to.
GALLERY = "gallery"
EMBEDDINGS = "embeddings"
METRICS = "metrics"
CALIBRATION = "calibration"
COMPARE = "compare"
CLEANING = "cleaning"
# The Filter tab.
DATASET_FILTER = "dataset_filter"

# The app-wide block, and the keys inside it.
APP = "app"
VIEWS = "views"
TAB = "tab"
DISPLAY_LUT = "display_lut"
DISPLAY_ROLL_COL = "display_roll_col"
USE_REVIEWED_LABELS = "use_reviewed_labels"
REVIEW_VERSION_ID = "review_version_id"


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class ViewStateRegistry:
    """Collects the view state of a session, and puts one back.

    Lives on the session so that anything can reach it, and is populated
    once every view of the session exists.
    """

    def __init__(self, session):
        self.session = session
        self._views: dict = {}
        self._get_tab: Optional[Callable[[], int]] = None
        self._set_tab: Optional[Callable[[int], None]] = None

    def register(self, key: str, view) -> None:
        self._views[key] = view

    def set_tab_accessors(self, getter, setter) -> None:
        """How to read and write the selected tab. Passed in rather than
        reaching into the page layout, which the session owns."""
        self._get_tab, self._set_tab = getter, setter

    # ---- capture --------------------------------------------------------

    def capture(self) -> dict:
        views = {}
        for key, view in self._views.items():
            try:
                state = view.view_state()
            except Exception as ex:  # noqa: BLE001 -- see the module docstring
                _log(f"view state: could not capture '{key}' ({ex})")
                continue
            if state:
                views[key] = state
        return {APP: self._capture_app(), VIEWS: views}

    def _capture_app(self) -> dict:
        session, state = self.session, self.session.state
        return {
            TAB: int(self._get_tab()) if self._get_tab else 0,
            DISPLAY_LUT: session.display_lut,
            DISPLAY_ROLL_COL: session.display_roll_col,
            USE_REVIEWED_LABELS: bool(state.use_reviewed_labels),
            REVIEW_VERSION_ID: state.review_version_id or "",
        }

    # ---- restore --------------------------------------------------------

    def restore(self, state: dict) -> list:
        """Apply a captured state. Returns what could not be applied, in
        the order a user would read it."""
        if not isinstance(state, dict):
            return ["The snapshot could not be read."]
        warnings = list(self._restore_app(state.get(APP) or {}))
        # Absent means "nothing was saved"; anything that is not a mapping
        # means the file is malformed, and the two must not be conflated --
        # `or {}` did exactly that, reporting a corrupt view as an empty one.
        views = state.get(VIEWS)
        if views is None:
            views = {}
        if not isinstance(views, dict):
            return warnings + ["The snapshot could not be read."]
        restored = []
        for key, view in self._views.items():
            saved = views.get(key)
            if not isinstance(saved, dict):
                continue
            try:
                warnings.extend(view.restore_view_state(saved) or [])
            except Exception as ex:  # noqa: BLE001 -- see the module docstring
                _log(f"view state: could not restore '{key}' ({ex})")
                warnings.append(f"{key}: could not be restored ({ex})")
                continue
            restored.append(key)

        # The tab goes last, deliberately. Every view above re-renders its
        # own subtree as it restores, and the data load that precedes all of
        # this ends in a full page update -- switching first meant the tab
        # was set on the model and then never pushed, so a view saved on the
        # Embeddings tab reopened on whichever tab the app happened to start
        # on. Landing on the right tab is most of what "the same view" means.
        self._restore_tab(state.get(APP) or {})

        # ... and only then the expensive part, so the user is already
        # looking at the right tab while it computes.
        self._recompute(restored)
        return warnings

    def _recompute(self, keys: list) -> None:
        """Ask each restored view to rebuild what it was showing.

        A view that cannot (its experiment is gone, its embeddings were
        never attached) declines on its own. A view that raises is logged
        and skipped rather than allowed to take the others with it -- the
        same rule as restoring.
        """
        for key in keys:
            view = self._views.get(key)
            recompute = getattr(view, "recompute_after_restore", None)
            if recompute is None:
                continue
            try:
                recompute()
            except Exception as ex:  # noqa: BLE001 -- see the module docstring
                _log(f"view state: could not recompute '{key}' ({ex})")

    def _restore_tab(self, app: dict) -> None:
        tab = app.get(TAB)
        if self._set_tab is not None and isinstance(tab, int) and not isinstance(tab, bool):
            self._set_tab(tab)

    def _restore_app(self, app: dict) -> list:
        """Display settings and the review overlay first: they change what
        every view below draws, so applying them afterwards would leave the
        first views rendered against the wrong ones."""
        warnings = []
        session, state = self.session, self.session.state

        lut = app.get(DISPLAY_LUT)
        if isinstance(lut, str) and lut:
            session.display_lut = lut
        roll_col = app.get(DISPLAY_ROLL_COL)
        if isinstance(roll_col, str) and roll_col:
            session.display_roll_col = roll_col

        # The review overlay changes every number in the app, so a snapshot
        # that cannot restore the version it was taken at must not silently
        # score against a different one.
        version_id = str(app.get(REVIEW_VERSION_ID, "") or "")
        store = getattr(state, "review", None)
        if version_id:
            if store is not None and version_id in getattr(store, "versions", {}):
                state.review_version_id = version_id
            else:
                warnings.append(
                    f"Review version '{version_id}' is not in this dataset's review store -- "
                    f"metrics are on the active version instead."
                )
        wanted_overlay = bool(app.get(USE_REVIEWED_LABELS, False))
        if wanted_overlay and store is None:
            warnings.append("The reviewed-label overlay was on when this was saved, but there is no review store.")
        else:
            state.use_reviewed_labels = wanted_overlay

        return warnings
