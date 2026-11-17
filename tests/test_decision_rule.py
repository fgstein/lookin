"""core/decision_rule.py: one bias per class before the argmax.

The sweep that finds a class's best bias in one sort is checked against
setting the bias and recomputing everything; the fit against a brute-force
search where one exists; the metrics after the rule against compute_metrics
on the biased scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, recall_score

from lookin.core import decision_rule as dr
from lookin.core.data_manager import AppState, ColumnConfig
from lookin.core.metrics import compute_metrics


def _frame(seed: int, n: int, k: int, rounded: bool = False, missing: bool = False):
    rng = np.random.default_rng(seed)
    labels = [f"c{i}" for i in range(k)]
    true = rng.choice(k, n, p=np.r_[0.5, np.full(k - 1, 0.5 / (k - 1))])
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), true] += 1.0
    scores = np.exp(logits)
    scores /= scores.sum(axis=1, keepdims=True)
    if rounded:
        scores = np.round(scores, 1) + 1e-9
        scores /= scores.sum(axis=1, keepdims=True)
    scores = scores.astype(np.float32)
    if missing:  # one score missing, the rest of the row still a distribution
        scores[0, 1] = np.nan
        scores[0] /= np.nansum(scores[0])
    df = pd.DataFrame(scores, columns=labels)
    df["label"] = [labels[i] for i in true]
    df["predicted_label"] = [labels[i] for i in np.nan_to_num(scores, nan=-1).argmax(axis=1)]
    return df, labels


def _problem(seed=0, n=400, k=5, **kw):
    df, labels = _frame(seed, n, k, **kw)
    return dr.prepare(df, "label", labels, labels), df, labels


def _brute_objective(problem, spec, biases):
    pred = dr.predict(problem.logs, biases)
    y = problem.y
    labels = list(range(len(problem.labels)))
    weights = dr._weights(problem, spec)
    used = [j for j in labels if weights.w[j] > 0]
    if spec.objective == dr.OBJ_BALANCED_ACCURACY:
        return float(np.mean(recall_score(y, pred, labels=used, average=None, zero_division=0)))
    per_class = f1_score(y, pred, labels=used, average=None, zero_division=0)
    return float((weights.w[used] * per_class).sum())


@pytest.mark.parametrize("objective", [dr.OBJ_MACRO_F1, dr.OBJ_WEIGHTED_F1, dr.OBJ_BALANCED_ACCURACY])
@pytest.mark.parametrize("rounded", [False, True])
def test_the_sweep_is_setting_the_bias_and_recomputing(objective, rounded):
    problem, _df, labels = _problem(1, 300, 4, rounded=rounded)
    spec = dr.TuningSpec(objective=objective)
    biases = np.array([0.0, 0.3, -0.2, 0.5])
    for c in range(4):
        sweep = dr.class_sweep(problem, biases, labels[c], spec)
        for j, value in enumerate(sweep.bias):
            trial = biases.copy()
            trial[c] = value
            assert sweep.objective[j] == pytest.approx(_brute_objective(problem, spec, trial), abs=1e-12)
            pred = dr.predict(problem.logs, trial)
            assert sweep.precision[j] * ((pred == c).sum()) == pytest.approx(((pred == c) & (problem.y == c)).sum()) \
                or (pred == c).sum() == 0


def test_the_current_interval_is_the_current_rule():
    problem, _df, labels = _problem(2, 300, 4)
    biases = np.array([0.0, 0.4, -0.3, 0.1])
    pred = dr.predict(problem.logs, biases)
    for c, label in enumerate(labels):
        sweep = dr.class_sweep(problem, biases, label)
        tp = ((pred == c) & (problem.y == c)).sum()
        assert sweep.recall[sweep.current] == pytest.approx(tp / (problem.y == c).sum())
        assert sweep.precision[sweep.current] == pytest.approx(tp / (pred == c).sum())


@pytest.mark.parametrize("objective", [dr.OBJ_MACRO_F1, dr.OBJ_BALANCED_ACCURACY])
def test_two_classes_reach_the_global_optimum(objective):
    problem, _df, _labels = _problem(3, 500, 2)
    spec = dr.TuningSpec(objective=objective)
    result = dr.fit_biases(problem, spec)
    grid = np.linspace(-6, 6, 4001)
    best = max(_brute_objective(problem, spec, np.array([0.0, b])) for b in grid)
    assert result.objective_after == pytest.approx(best, abs=1e-12)
    ref = problem.labels.index(result.reference)
    assert ref == int(np.argmax(problem.n_true)) and result.biases[ref] == 0.0


@pytest.mark.parametrize("rounded", [False, True])
def test_the_fit_only_improves_and_reports_what_it_reaches(rounded):
    problem, _df, _labels = _problem(4, 600, 6, rounded=rounded, missing=True)
    spec = dr.TuningSpec()
    result = dr.fit_biases(problem, spec)
    assert result.objective_after >= result.objective_before
    assert result.objective_after > result.objective_before  # the minority classes gain
    assert result.objective_before == pytest.approx(_brute_objective(problem, spec, np.zeros(6)), abs=1e-12)
    assert result.objective_after == pytest.approx(_brute_objective(problem, spec, result.biases), abs=1e-12)
    assert result.converged
    # No single class's bias can be moved for the better: a coordinate-wise optimum.
    for label in problem.labels[1:]:
        sweep = dr.class_sweep(problem, result.biases, label, spec)
        assert np.nanmax(sweep.objective) <= result.objective_after + 1e-12


def test_locked_classes_and_the_bound_hold():
    problem, _df, labels = _problem(5, 500, 5)
    spec = dr.TuningSpec(locked=("c2", "c3"), max_abs_bias=0.25)
    result = dr.fit_biases(problem, spec)
    assert result.reference is None
    assert result.biases[2] == 0.0 and result.biases[3] == 0.0
    assert np.all(np.abs(result.biases) <= 0.25)
    assert result.objective_after == pytest.approx(_brute_objective(problem, spec, result.biases), abs=1e-12)


def test_a_subset_objective_counts_only_its_classes():
    problem, _df, _labels = _problem(6, 500, 5)
    spec = dr.TuningSpec(classes=("c1", "c4"))
    result = dr.fit_biases(problem, spec)
    assert result.counted == ["c1", "c4"]
    assert result.objective_after == pytest.approx(_brute_objective(problem, spec, result.biases), abs=1e-12)
    with pytest.raises(ValueError, match="balanced accuracy"):
        dr.fit_biases(problem, dr.TuningSpec(objective=dr.OBJ_BALANCED_ACCURACY, classes=("c1",)))
    with pytest.raises(ValueError, match="no score column"):
        dr.fit_biases(problem, dr.TuningSpec(classes=("nope",)))


def test_a_class_with_no_row_is_left_out_of_the_objective():
    df, labels = _frame(7, 300, 4)
    df = df[df["label"] != "c3"].reset_index(drop=True)
    problem = dr.prepare(df, "label", labels, labels)
    result = dr.fit_biases(problem, dr.TuningSpec())
    assert "c3" not in result.counted
    result = dr.fit_biases(problem, dr.TuningSpec(classes=("c1", "c3")))
    assert result.counted == ["c1"] and result.absent == ["c3"]


def test_the_metrics_after_the_rule_are_compute_metrics_on_the_biased_scores():
    problem, df, labels = _problem(8, 600, 4)
    result = dr.fit_biases(problem, dr.TuningSpec())
    after = dr.apply_to_frame(df, "label", labels, labels, problem.space, result.biases)
    assert np.allclose(after[labels].sum(axis=1), 1.0, atol=1e-5)
    expected = [labels[i] for i in dr.predict(problem.logs, result.biases)]
    assert list(after["predicted_label"]) == expected
    m = compute_metrics("x", after, "label", labels, labels)
    assert m.macro_f1 == pytest.approx(result.objective_after, abs=1e-9)


def test_logits_are_shifted_and_sigmoids_refused():
    df, labels = _frame(9, 200, 3)
    logits = np.log(df[labels].to_numpy()) * 2.0 + 1.0
    assert dr.score_space(logits)[0] == dr.SPACE_LOGIT
    shifted = dr.biased_scores(logits, dr.SPACE_LOGIT, np.array([0.0, 1.0, -1.0]))
    assert np.allclose(shifted, logits + [0.0, 1.0, -1.0], atol=1e-5)
    sigmoid = df[labels].to_numpy() * 0.7
    space, reason = dr.score_space(sigmoid)
    assert space is None and "sum to 1" in reason
    df[labels] = sigmoid
    with pytest.raises(ValueError, match="sum to 1"):
        dr.prepare(df, "label", labels, labels)


def test_two_columns_for_one_class_are_refused():
    df, labels = _frame(10, 100, 3)
    with pytest.raises(ValueError, match="several score columns"):
        dr.prepare(df, "label", labels, ["c0", "c1", "c1"])


def test_cross_fit_measures_the_optimism():
    problem, _df, _labels = _problem(11, 600, 5)
    fit = dr.cross_fit(problem, dr.TuningSpec())
    assert len(fit.folds) == 2 and sum(f[1] for f in fit.folds) == problem.n_rows
    assert fit.in_sample >= fit.before


def test_check_pools_blocks_shared_rows_unless_chosen():
    assert dr.check_pools(["a", "b"], ["c", "d"], same_pool=False).blocking is None
    check = dr.check_pools(["a", "b", "b"], ["b", "c"], same_pool=False)
    assert check.n_shared == 1 and check.n_tuning_duplicates == 1 and check.blocking
    assert dr.check_pools(["a", "b"], ["a", "b"], same_pool=True).blocking is None


# ---- the validation pool read from the dataset CSV ----------------------------

def _state(tmp_path, val_predicted: int):
    df, labels = _frame(12, 300, 3)
    uuids = [f"u{i}" for i in range(len(df))]
    split = ["test"] * 200 + ["val"] * 100
    pd.DataFrame({"uuid": uuids, "crop": [f"{u}.png" for u in uuids], "label": df["label"],
                  "split": split}).to_csv(tmp_path / "ds.csv", index=False)
    results = df[labels].copy()
    results.insert(0, "uuid", uuids)
    results = results.iloc[: 200 + val_predicted]
    results.to_csv(tmp_path / "res.csv", index=False)
    state = AppState()
    state.load_dataset(str(tmp_path / "ds.csv"), ColumnConfig(), row_filter=("split", "test"))
    state.add_experiment(str(tmp_path / "res.csv"), "exp")
    return state


def test_the_validation_rows_are_read_apart_from_the_loaded_pool(tmp_path):
    state = _state(tmp_path, val_predicted=100)
    assert len(state.merged("exp")) == 200
    pool = dr.read_tuning_pool(state, "exp")
    assert pool.disabled_reason is None and pool.n_rows == 100 and pool.n_without_prediction == 0
    check = dr.check_pools(pool.frame["uuid"], state.merged("exp")["uuid"], same_pool=False)
    assert check.n_shared == 0 and check.blocking is None
    assert set(pool.frame["predicted_label"]) <= {"c0", "c1", "c2"}


def test_missing_validation_predictions_are_reported(tmp_path):
    state = _state(tmp_path, val_predicted=40)
    pool = dr.read_tuning_pool(state, "exp")
    assert pool.n_rows == 40 and pool.n_without_prediction == 60 and pool.warnings
    state = _state(tmp_path, val_predicted=0)
    pool = dr.read_tuning_pool(state, "exp")
    assert pool.frame is None and "Run the model on those rows" in pool.disabled_reason
    assert "no column 'fold'" in dr.read_tuning_pool(state, "exp", split_column="fold").disabled_reason
    assert "no row where" in dr.read_tuning_pool(state, "exp", split_value="dev").disabled_reason


def test_an_applied_rule_is_read_everywhere_and_reverted(tmp_path):
    state = _state(tmp_path, val_predicted=100)
    exp = state.experiments["exp"]
    labels = exp.score_cols
    raw = state.merged("exp")
    pool = dr.read_tuning_pool(state, "exp")
    problem = dr.prepare(pool.frame, "label", labels, labels)
    result = dr.fit_biases(problem, dr.TuningSpec())
    rule = dr.make_record(labels, labels, result.biases, problem.space, dr.POOL_VAL, "split", "val",
                          problem.n_rows, dr.OBJ_MACRO_F1, objective_before=result.objective_before,
                          objective_after=result.objective_after)
    state.apply_decision_rule("exp", rule)
    biased = state.merged("exp")
    expected = dr.apply_to_frame(raw, "label", labels, labels, problem.space, result.biases)
    assert list(biased["predicted_label"]) == list(expected["predicted_label"])
    assert np.allclose(biased[labels].to_numpy(), expected[labels].to_numpy(), atol=1e-6)
    # The tab still reads the scores as they were...
    assert np.allclose(state.raw_merged("exp")[labels].to_numpy(), raw[labels].to_numpy())
    # ...and a second rule replaces the first rather than stacking on it.
    state.apply_decision_rule("exp", rule)
    assert np.allclose(state.merged("exp")[labels].to_numpy(), expected[labels].to_numpy(), atol=1e-6)
    assert state.revert_decision_rule("exp")
    assert exp.decision_rule is None and exp.raw_df is None
    assert list(state.merged("exp")["predicted_label"]) == list(raw["predicted_label"])


def test_a_rule_for_other_scores_is_refused(tmp_path):
    state = _state(tmp_path, val_predicted=100)
    labels = state.experiments["exp"].score_cols
    rule = dr.make_record(labels, labels, [0.0, 0.1, 0.2], dr.SPACE_LOGIT, dr.POOL_VAL)
    with pytest.raises(ValueError, match="read as"):
        state.apply_decision_rule("exp", rule)
    rule = dr.make_record(["nope"], ["nope"], [0.1], dr.SPACE_LOG_PROB, dr.POOL_VAL)
    with pytest.raises(ValueError, match="does not have"):
        state.apply_decision_rule("exp", rule)
    assert state.experiments["exp"].decision_rule is None


def test_validation_predictions_are_not_a_mismatch(tmp_path):
    state = _state(tmp_path, val_predicted=100)
    overlap = state.uuid_overlap("exp")
    assert overlap.experiment_only == 0 and overlap.other_split == 100


def test_the_tolerance_is_the_metrics_one():
    from lookin.core.metrics import ROW_SUM_TOLERANCE
    assert dr.SPACE_TOLERANCE == ROW_SUM_TOLERANCE


def test_a_workspace_reopens_with_its_rule(tmp_path):
    from lookin.core import library as lib
    from lookin.core.workspace_state import load_workspace_state

    def load_mount_into(_state, mount):
        return load_workspace_state(lib.Workspace(id="lkw-test", mount=mount))

    state = _state(tmp_path, val_predicted=100)
    labels = state.experiments["exp"].score_cols
    rule = dr.make_record(labels, labels, [0.0, 0.4, -0.2], dr.SPACE_LOG_PROB, dr.POOL_TEST)
    mount = lib.Mount(dataset_path=str(tmp_path / "ds.csv"), uuid_col="uuid", crop_col="crop", label_col="label",
                      row_filter_col="split", row_filter_value="test",
                      experiments=[lib.MountExperiment(name="exp", path=str(tmp_path / "res.csv"),
                                                       decision_rule=dr.record_to_text(rule))])
    raw = lib.mount_from_dict(lib.mount_to_dict(mount))
    assert raw.experiments[0].decision_rule == dr.record_to_text(rule)
    loaded = load_mount_into(None, raw)
    assert not loaded.warnings
    assert dr.is_tuned_on_evaluation(loaded.state.experiments["exp"].decision_rule)
    broken = lib.MountExperiment(name="exp", path=str(tmp_path / "res.csv"), decision_rule='{"biases": {"zz": 1}}')
    loaded = load_mount_into(None, lib.Mount(dataset_path=str(tmp_path / "ds.csv"), uuid_col="uuid",
                                              crop_col="crop", label_col="label", experiments=[broken]))
    assert loaded.state.experiments["exp"].decision_rule is None and "not applied" in loaded.warnings[-1]
