"""The view-state registry: what a saved view captures, and what it does
when it cannot be put back.

The rules under test are the two the module promises. A view that raises
must not stop a dataset opening, and a view that only partly applies must
say so -- a view restored to nine tenths of what it was is worse than one
not restored at all, because the user believes they are looking at the
same thing.
"""
from __future__ import annotations

import pytest

from lookin.core.data_manager import AppState
from lookin.ui import view_state as vs
from lookin.ui.view_state import ViewStateRegistry


class FakeView:
    def __init__(self, state=None, warnings=None, raises=False):
        self._state = state if state is not None else {"a": 1}
        self._warnings = warnings or []
        self._raises = raises
        self.restored = None

    def view_state(self):
        if self._raises:
            raise RuntimeError("boom")
        return self._state

    def restore_view_state(self, state):
        if self._raises:
            raise RuntimeError("boom")
        self.restored = state
        return self._warnings


class FakeCtx:
    def __init__(self):
        self.state = AppState()
        self.display_lut = "none"
        self.display_roll_col = ""


@pytest.fixture
def registry():
    ctx = FakeCtx()
    reg = ViewStateRegistry(ctx)
    reg.ctx = ctx
    return reg


# ---- capture ------------------------------------------------------------

def test_capture_collects_every_registered_view(registry):
    registry.register(vs.GALLERY, FakeView({"columns": ["label"]}))
    registry.register(vs.METRICS, FakeView({"metric": "f1"}))

    captured = registry.capture()

    assert captured[vs.VIEWS][vs.GALLERY] == {"columns": ["label"]}
    assert captured[vs.VIEWS][vs.METRICS] == {"metric": "f1"}


def test_capture_records_the_tab_and_the_display_settings(registry):
    registry.set_tab_accessors(lambda: 6, lambda i: None)
    registry.ctx.display_lut = "viridis"
    registry.ctx.display_roll_col = "roll"

    app = registry.capture()[vs.APP]

    assert app[vs.TAB] == 6
    assert app[vs.DISPLAY_LUT] == "viridis"
    assert app[vs.DISPLAY_ROLL_COL] == "roll"


def test_capture_skips_a_view_that_raises(registry, capsys):
    """One broken view must not cost the user the whole saved view."""
    registry.register(vs.GALLERY, FakeView({"columns": []}))
    registry.register(vs.METRICS, FakeView(raises=True))

    captured = registry.capture()

    assert vs.GALLERY in captured[vs.VIEWS]
    assert vs.METRICS not in captured[vs.VIEWS]
    assert "could not capture" in capsys.readouterr().out


def test_capture_drops_a_view_with_nothing_to_say(registry):
    registry.register(vs.GALLERY, FakeView({}))
    assert registry.capture()[vs.VIEWS] == {}


# ---- restore ------------------------------------------------------------

def test_restore_hands_each_view_its_own_block(registry):
    gallery, metrics = FakeView(), FakeView()
    registry.register(vs.GALLERY, gallery)
    registry.register(vs.METRICS, metrics)

    registry.restore({vs.VIEWS: {vs.GALLERY: {"columns": ["site"]}, vs.METRICS: {"metric": "f1"}}})

    assert gallery.restored == {"columns": ["site"]}
    assert metrics.restored == {"metric": "f1"}


def test_restore_collects_what_could_not_be_applied(registry):
    registry.register(vs.GALLERY, FakeView(warnings=["Gallery: experiment 'x' is not loaded."]))
    warnings = registry.restore({vs.VIEWS: {vs.GALLERY: {}}})
    assert warnings == ["Gallery: experiment 'x' is not loaded."]


def test_a_view_that_raises_is_reported_not_propagated(registry):
    """Opening the data must survive a view that cannot restore itself."""
    good = FakeView()
    registry.register(vs.METRICS, FakeView(raises=True))
    registry.register(vs.GALLERY, good)

    warnings = registry.restore({vs.VIEWS: {vs.METRICS: {}, vs.GALLERY: {"columns": []}}})

    assert any("could not be restored" in w for w in warnings)
    assert good.restored == {"columns": []}, "the other views still restore"


def test_restore_ignores_a_block_that_is_not_a_mapping(registry):
    view = FakeView()
    registry.register(vs.GALLERY, view)
    assert registry.restore({vs.VIEWS: {vs.GALLERY: "nonsense"}}) == []
    assert view.restored is None


def test_restore_of_something_unreadable_says_so(registry):
    assert registry.restore(None)
    assert registry.restore({vs.VIEWS: []})


def test_restore_puts_the_tab_back(registry):
    seen = []
    registry.set_tab_accessors(lambda: 0, seen.append)
    registry.restore({vs.APP: {vs.TAB: 4}})
    assert seen == [4]


def test_the_tab_is_switched_after_every_view_has_restored(registry):
    """Landing on the tab you were on is most of what "the same view"
    means, and the views re-render their own subtrees as they restore --
    switching first left the tab set on the model and never shown."""
    order = []

    class Recorder(FakeView):
        def restore_view_state(self, state):
            order.append("view")
            return []

    registry.register(vs.GALLERY, Recorder())
    registry.set_tab_accessors(lambda: 0, lambda i: order.append(f"tab:{i}"))

    registry.restore({vs.APP: {vs.TAB: 6}, vs.VIEWS: {vs.GALLERY: {}}})

    assert order == ["view", "tab:6"]


def test_the_tab_survives_a_view_that_raises(registry):
    """A broken view must not cost the user the tab they were on."""
    seen = []
    registry.register(vs.METRICS, FakeView(raises=True))
    registry.set_tab_accessors(lambda: 0, seen.append)

    registry.restore({vs.APP: {vs.TAB: 6}, vs.VIEWS: {vs.METRICS: {}}})

    assert seen == [6]


@pytest.mark.parametrize("value", [None, "6", 2.5, True])
def test_a_tab_that_is_not_an_index_is_ignored(registry, value):
    """True is an int in Python, and would silently select tab 1."""
    seen = []
    registry.set_tab_accessors(lambda: 0, seen.append)
    registry.restore({vs.APP: {vs.TAB: value}})
    assert seen == []


def test_the_tab_is_captured(registry):
    registry.set_tab_accessors(lambda: 6, lambda i: None)
    assert registry.capture()[vs.APP][vs.TAB] == 6


def test_restore_applies_display_settings(registry):
    registry.restore({vs.APP: {vs.DISPLAY_LUT: "magma", vs.DISPLAY_ROLL_COL: "roll",
                               "display_roll_reverse": True}})
    assert registry.ctx.display_lut == "magma"
    assert registry.ctx.display_roll_col == "roll"
    assert not hasattr(registry.ctx, "display_roll_reverse"), (
        "a workspace saved when the direction was still a setting restores "
        "everything else and drops that key")


# ---- the review overlay, which changes every number in the app ----------

def test_a_missing_review_version_is_reported_not_silently_ignored(registry):
    """Scoring against a different cleaning pass than the one a view was
    saved at is exactly the mistake the version pin exists to prevent."""
    warnings = registry.restore({vs.APP: {vs.REVIEW_VERSION_ID: "v3"}})

    assert any("v3" in w for w in warnings)
    assert registry.ctx.state.review_version_id is None


def test_a_known_review_version_is_pinned(registry):
    class FakeStore:
        versions = {"v2": object()}

    registry.ctx.state.review = FakeStore()
    warnings = registry.restore({vs.APP: {vs.REVIEW_VERSION_ID: "v2"}})

    assert registry.ctx.state.review_version_id == "v2"
    assert warnings == []


def test_the_overlay_is_not_turned_on_without_a_review_store(registry):
    warnings = registry.restore({vs.APP: {vs.USE_REVIEWED_LABELS: True}})

    assert registry.ctx.state.use_reviewed_labels is False
    assert any("review store" in w for w in warnings)


def test_the_overlay_is_restored_when_there_is_a_store(registry):
    class FakeStore:
        versions = {}

    registry.ctx.state.review = FakeStore()
    registry.restore({vs.APP: {vs.USE_REVIEWED_LABELS: True}})
    assert registry.ctx.state.use_reviewed_labels is True


def test_display_settings_are_applied_before_the_views(registry):
    """A view rendered against the old LUT would be drawing the wrong
    thing, so the app-wide block has to land first."""
    seen = {}

    class Recorder(FakeView):
        def restore_view_state(self, state):
            seen["lut"] = registry.ctx.display_lut
            return []

    registry.register(vs.GALLERY, Recorder())
    registry.restore({vs.APP: {vs.DISPLAY_LUT: "magma"}, vs.VIEWS: {vs.GALLERY: {}}})
    assert seen["lut"] == "magma"


# ---- the tab setter app.py hands the registry ---------------------------

class FakeTabs:
    def __init__(self, count=9):
        self.tabs = list(range(count))
        self.selected_index = 0


class FakePane:
    def __init__(self, visible=False):
        self.visible = visible


def fake_panes(count=9):
    return [FakePane(visible=index == 0) for index in range(count)]


class FakePage:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1


def test_the_tab_setter_pushes_the_switch_to_the_client():
    """Setting selected_index alone changes the model and nothing else --
    the browser keeps showing the tab it was on."""
    from lookin.app import make_tab_setter

    tabs, page = FakeTabs(), FakePage()
    make_tab_setter(tabs, fake_panes(), page, {})(6)

    assert tabs.selected_index == 6
    assert page.updates == 1, "without this the switch never reaches the browser"


def test_the_tab_setter_shows_only_the_selected_pane():
    """The tab bar carries no content (see app.main): selecting a tab
    without showing its pane would leave the old tab on screen."""
    from lookin.app import make_tab_setter

    panes = fake_panes()
    make_tab_setter(FakeTabs(), panes, FakePage(), {})(6)

    assert [pane.visible for pane in panes] == [index == 6 for index in range(9)]


def test_the_tab_setter_runs_the_handler_a_tab_needs_when_opened():
    """Flet does not fire on_change for a value set from Python, so a view
    saved on the Compare tab would reopen there with empty pickers."""
    from lookin.app import make_tab_setter

    shown = []
    make_tab_setter(FakeTabs(), fake_panes(), FakePage(), {5: lambda: shown.append(5)})(5)

    assert shown == [5]


def test_the_tab_setter_ignores_an_index_that_does_not_exist():
    """A snapshot from a version with more tabs must not select nothing."""
    from lookin.app import make_tab_setter

    tabs, page = FakeTabs(count=9), FakePage()
    setter = make_tab_setter(tabs, fake_panes(), page, {})
    for bad in (-1, 9, 99):
        setter(bad)

    assert tabs.selected_index == 0
    assert page.updates == 0


# ---- recomputing what the view was showing ------------------------------

class RecomputingView(FakeView):
    def __init__(self, *args, recompute_raises=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.recomputed = 0
        self._recompute_raises = recompute_raises

    def recompute_after_restore(self):
        if self._recompute_raises:
            raise RuntimeError("boom")
        self.recomputed += 1


def test_a_restored_view_is_asked_to_recompute(registry):
    """Opening a saved view is an explicit "show me this again" -- an empty
    map under a Run button is not the view that was saved."""
    view = RecomputingView()
    registry.register(vs.EMBEDDINGS, view)

    registry.restore({vs.VIEWS: {vs.EMBEDDINGS: {"method": "tsne"}}})

    assert view.recomputed == 1


def test_a_view_with_no_saved_state_is_not_recomputed(registry):
    """Restoring one tab must not start every expensive rebuild in the app."""
    view = RecomputingView()
    registry.register(vs.EMBEDDINGS, view)
    registry.register(vs.METRICS, RecomputingView())

    registry.restore({vs.VIEWS: {vs.EMBEDDINGS: {}}})

    assert view.recomputed == 1
    assert registry._views[vs.METRICS].recomputed == 0


def test_a_view_that_failed_to_restore_is_not_recomputed(registry):
    """Recomputing a view whose state did not apply would show a result
    the user never asked for."""
    view = RecomputingView(raises=True)
    registry.register(vs.EMBEDDINGS, view)

    registry.restore({vs.VIEWS: {vs.EMBEDDINGS: {}}})

    assert view.recomputed == 0


def test_a_recompute_that_raises_does_not_stop_the_others(registry):
    good = RecomputingView()
    registry.register(vs.EMBEDDINGS, RecomputingView(recompute_raises=True))
    registry.register(vs.METRICS, good)

    registry.restore({vs.VIEWS: {vs.EMBEDDINGS: {}, vs.METRICS: {}}})

    assert good.recomputed == 1


def test_the_tab_is_shown_before_anything_recomputes(registry):
    """The user should be looking at the right tab while it computes, not
    after."""
    order = []

    class Ordered(RecomputingView):
        def recompute_after_restore(self):
            order.append("recompute")

    registry.register(vs.EMBEDDINGS, Ordered())
    registry.set_tab_accessors(lambda: 0, lambda i: order.append("tab"))

    registry.restore({vs.APP: {vs.TAB: 6}, vs.VIEWS: {vs.EMBEDDINGS: {}}})

    assert order == ["tab", "recompute"]


def test_a_view_with_no_recompute_hook_is_fine(registry):
    """The Gallery has nothing expensive to rebuild -- restoring its
    filters already redrew it."""
    registry.register(vs.GALLERY, FakeView())
    assert registry.restore({vs.VIEWS: {vs.GALLERY: {}}}) == []
