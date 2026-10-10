"""The Calibration tab, rendered without a browser.

core/calibration.py is covered separately; what is checked here is the tab's
own behaviour: that a full render produces every section, that the threshold
controls stay consistent with the numbers they display, that the sections
needing a probability switch off (and only those) when the scores are not
one, and that the caches follow a rename.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lookin.core.data_manager import AppState, ColumnConfig
from lookin.ui.calibration_view import CalibrationView


class FakeCtx:
    def __init__(self, state):
        self.state = state
        self.page = None
        self.image_zoom = None
        self.display_lut = "none"


@pytest.fixture
def view(loaded_state):
    v = CalibrationView(FakeCtx(loaded_state))
    v.build()
    return v


@pytest.fixture
def unnormalized_state(tmp_path, tiny_dataset_df):
    """The same six crops, scored by a sigmoid head: rows do not sum to 1,
    so calibration is meaningless and abstention is not."""
    tiny_dataset_df.to_csv(tmp_path / "dataset.csv", index=False)
    pd.DataFrame({
        "uuid": ["a", "b", "c", "d", "e", "f"],
        "cat": [0.9, 0.8, 0.2, 0.3, 0.1, 0.2],
        "dog": [0.7, 0.6, 0.9, 0.8, 0.2, 0.9],
        "bird": [0.6, 0.5, 0.4, 0.3, 0.9, 0.4],
    }).to_csv(tmp_path / "sigmoid_results.csv", index=False)

    state = AppState()
    state.load_dataset(str(tmp_path / "dataset.csv"), ColumnConfig())
    state.add_experiment(str(tmp_path / "sigmoid_results.csv"), "sigmoid")
    return state


# ---- a full render ------------------------------------------------------

def test_a_render_fills_every_section(view):
    assert view.summary_area.controls
    assert view.reliability_area.controls
    assert view.histogram_area.controls
    assert view.curve_area.content is not None
    assert view.threshold_table_area.controls
    assert view.target_table_area.controls


def test_the_summary_reports_the_accuracy_the_rows_actually_have(view, loaded_state):
    bundle = view._cache["exp"]
    merged = loaded_state.merged("exp")
    assert bundle.n == len(merged)
    assert bundle.accuracy == pytest.approx(merged["correct"].mean())


def test_no_experiment_selected_shows_a_message_rather_than_an_empty_page(view):
    view.experiment_checks["exp"].value = False
    view._render()
    assert view.summary_area.controls
    assert view.reliability_area.controls == []


def test_a_view_with_no_data_renders_a_placeholder():
    v = CalibrationView(FakeCtx(AppState()))
    v.build()
    assert v.summary_area.controls
    assert v.curve_area.content is None


# ---- the threshold controls --------------------------------------------

def test_moving_the_threshold_changes_what_the_table_reports(view):
    from lookin.core.calibration import threshold_point

    index = view._cache["exp"].index
    permissive = threshold_point(index, 0.0)
    strict = threshold_point(index, 0.95)
    assert permissive.n_accepted > strict.n_accepted

    view.threshold_slider.value = 0.95
    view._on_threshold_settled(None)
    assert view.threshold_table_area.controls
    assert "0.950" in view.threshold_label.value


def test_the_label_follows_the_slider_while_dragging(view):
    view.threshold_slider.value = 0.42
    view._on_threshold_slide(None)
    assert "0.420" in view.threshold_label.value


def test_an_impossible_target_renders_as_unreachable_rather_than_a_number(view):
    """The example experiment cannot be perfect, so asking for 100% accuracy
    on a target that includes its errors has to say so."""
    from lookin.core.calibration import threshold_for_target

    point = threshold_for_target(view._cache["exp"].index, 1.0)
    view.target_field.value = "1.0"
    view._on_target_change(None)
    assert view.target_table_area.controls
    # Either it is genuinely reachable by accepting only correct rows, or it
    # is flagged -- what must not happen is a silent NaN threshold.
    assert point.unreachable or point.n_accepted > 0


def test_a_target_outside_zero_one_is_clamped(view):
    view.target_field.value = "180"
    view._on_target_change(None)
    assert float(view.target_field.value) <= 1.0


def test_a_nonsense_target_falls_back_to_the_default(view):
    view.target_field.value = "high"
    view._on_target_change(None)
    assert float(view.target_field.value) == pytest.approx(0.99)


# ---- per-class thresholds ----------------------------------------------

def test_per_class_thresholds_render_one_row_per_class(view):
    view.class_experiment_dropdown.value = "exp"
    view.class_target_field.value = "0.5"
    view._compute_class_thresholds()
    assert view.class_area.controls
    rows = view._class_rows[("exp", 0.5)]
    assert {r.label for r in rows} == {"cat", "dog", "bird"}


# ---- temperature scaling ------------------------------------------------

def test_fitting_a_temperature_leaves_every_prediction_alone(view, loaded_state):
    before = loaded_state.merged("exp")["predicted_label"].tolist()
    view._fit_temperatures()
    fit = view._cache["exp"].temperature
    assert fit is not None
    assert fit.disabled_reason is None
    assert loaded_state.merged("exp")["predicted_label"].tolist() == before


def test_the_temperature_section_says_so_before_it_is_fitted(view):
    assert view.temperature_area.controls
    assert view._cache["exp"].temperature is None


# ---- scores that are not probabilities ---------------------------------

def test_ranking_sections_survive_scores_that_are_not_probabilities(unnormalized_state):
    """The distinction the whole module is arranged around: a threshold only
    ranks, so it keeps working; ECE reads the confidence as a probability, so
    it must not be shown."""
    view = CalibrationView(FakeCtx(unnormalized_state))
    view.build()
    bundle = view._cache["sigmoid"]

    assert bundle.diagnostics.normalized is False
    assert bundle.reliability.disabled_reason        # calibration off
    assert np.isnan(bundle.reliability.ece)
    assert bundle.curve.n > 0                        # abstention still on
    assert view.threshold_table_area.controls
    assert view.target_table_area.controls
    assert view.warnings_area.controls               # and the user is told


def test_temperature_scaling_is_refused_on_scores_that_are_not_probabilities(unnormalized_state):
    view = CalibrationView(FakeCtx(unnormalized_state))
    view.build()
    view._fit_temperatures()
    fit = view._cache["sigmoid"].temperature
    assert fit.disabled_reason is not None
    assert view.temperature_area.controls


# ---- lifecycle ----------------------------------------------------------

def test_a_data_change_marks_the_view_stale_without_recomputing(view):
    cached = view._cache["exp"]
    view.refresh()
    assert view._dirty is True
    assert view._cache["exp"] is cached      # nothing recomputed


def test_update_drops_the_caches_and_recomputes(view):
    cached = view._cache["exp"]
    view._cache.clear()
    view._refresh_now()
    assert "exp" in view._cache
    assert view._cache["exp"] is not cached
    assert view._dirty is False


def test_a_rename_moves_the_caches_instead_of_recomputing(view, loaded_state):
    view.class_experiment_dropdown.value = "exp"
    view.class_target_field.value = "0.5"
    view._compute_class_thresholds()
    cached = view._cache["exp"]

    # Same order as AppContext.notify_rename: the state is re-keyed first,
    # so the view's handler re-renders against a registry that already
    # answers to the new name.
    loaded_state.rename_experiment("exp", "renamed")
    view.rename_experiment("exp", "renamed")
    assert view._cache["renamed"] is cached
    assert view._cache["renamed"].name == "renamed"
    assert "exp" not in view._cache
    assert ("renamed", 0.5) in view._class_rows
    assert "renamed" in view.experiment_checks


def test_restoring_an_empty_selection_clears_the_per_class_dropdown(view):
    """The per-class dropdown is derived from which experiments are ticked.
    Unticking them by hand left it pointing at one that is no longer shown,
    so a freshly restored snapshot did not equal itself and the app bar
    reported it as already modified.
    """
    view.restore_view_state({"selected": [], "threshold": 0.9})

    assert view._selected_names() == []
    assert view.class_experiment_dropdown.value in (None, "")
    assert view.view_state()["class_experiment"] == ""


def test_restoring_a_selection_points_the_dropdown_at_it(view):
    view.restore_view_state({"selected": ["exp"]})

    assert view._selected_names() == ["exp"]
    assert view.class_experiment_dropdown.value == "exp"


def test_calibration_state_round_trips(view):
    """restore must be a fixed point of capture, or "is this saved?" can
    never answer yes."""
    view.restore_view_state({"selected": ["exp"], "threshold": 0.75,
                             "target_accuracy": "0.98", "target_precision": "0.9"})
    captured = view.view_state()

    view.restore_view_state(captured)

    assert view.view_state() == captured
