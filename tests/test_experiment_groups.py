"""Experiment groups (core/experiment_groups.py): several experiments drawn
as one in the charts -- one color, one legend entry, the group's name.

The promise checked hardest here is the one the feature is built on: a
group changes how a chart is *painted and named*, never what it draws.
Every chart that takes the groups is drawn twice, with and without them,
and the lines, bars and error bars must be the same to the last float.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

from lookin.core import agent_actions as actions
from lookin.core import agent_outputs as outputs
from lookin.core import agent_tools as tools
from lookin.core import calibration as cal
from lookin.core import charts
from lookin.core import experiment_groups as eg
from lookin.core import library as lib
from lookin.core import links
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss
from lookin.core.chart_style import color_for_experiment
from lookin.core.data_manager import AppState, ColumnConfig
from lookin.core.metrics import MACRO_AVERAGE, compute_curves, compute_metrics, compute_reliability

matplotlib.use("Agg")

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"
CKPT_1, CKPT_2, OTHER = "run-A/ckpt_1", "run-A/ckpt_2", "run-B"


# ---- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def state() -> AppState:
    """The example dataset with three experiments: two checkpoints of one
    run and another run (the same files, under three names)."""
    s = AppState()
    s.load_dataset(str(EXAMPLE / "dataset.csv"), ColumnConfig())
    s.add_experiment(str(EXAMPLE / "baseline_cnn_results.csv"), CKPT_1)
    s.add_experiment(str(EXAMPLE / "finetuned_resnet_results.csv"), CKPT_2)
    s.add_experiment(str(EXAMPLE / "baseline_cnn_results.csv"), OTHER)
    return s


@pytest.fixture
def grouped(state):
    """`state` with the two checkpoints in group "run A", shown."""
    state.groups.clear()
    state.groups.create("run A", [CKPT_1, CKPT_2])
    yield state
    state.groups.clear()


def _merged_inputs(state):
    label_col = state.dataset.columns.label_col
    metrics, curves, risk, reliability = {}, {}, {}, {}
    for name, exp in state.experiments.items():
        merged = state.merged(name)
        metrics[name] = compute_metrics(name, merged, label_col, exp.score_cols, exp.class_labels)
        curves[name] = compute_curves(merged, label_col, exp.score_cols, exp.class_labels, MACRO_AVERAGE)
        confidence = merged["predicted_score"].to_numpy(dtype=float)
        correct = merged["correct"].to_numpy().astype(bool)
        risk[name] = cal.risk_coverage(cal.build_index(confidence, correct))
        reliability[name] = compute_reliability(merged)
    return metrics, curves, risk, reliability


@pytest.fixture(scope="module")
def inputs(state):
    return _merged_inputs(state)


def _floats(values) -> list:
    """Nested lists of floats, NaN as None -- NaN is never equal to itself."""
    array = np.asarray(values, dtype=float).round(12)
    return np.where(np.isnan(array), None, array).tolist()


def _drawn(fig) -> list:
    """Everything a figure draws, without its colors or its names: the
    points of every line, the place and size of every bar, the segments of
    every error bar -- axis by axis."""
    out = []
    for ax in fig.axes:
        lines = [_floats(line.get_xydata()) for line in ax.get_lines()]
        bars = [_floats([p.get_x(), p.get_y(), p.get_width(), p.get_height()])
                for p in ax.patches if isinstance(p, Rectangle)]
        segments = [[_floats(s) for s in c.get_segments()]
                    for c in ax.collections if isinstance(c, LineCollection)]
        out.append({"lines": lines, "bars": bars, "segments": segments, "xlim": ax.get_xlim(),
                    "ylim": ax.get_ylim()})
    return out


def _legend_labels(fig) -> list:
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    return [[t.get_text() for t in legend.get_texts()] for legend in legends]


def _both(state, draw):
    """`draw(colors, labels)` with the groups hidden, then shown."""
    state.groups.set_shown(False)
    names = list(state.experiments)
    plain = draw({n: color_for_experiment(n, state.color_slots) for n in names}, state.chart_labels(names))
    state.groups.set_shown(True)
    with_groups = draw({n: color_for_experiment(n, state.color_slots) for n in names}, state.chart_labels(names))
    return plain, with_groups


# ---- the charts draw the same thing, grouped or not --------------------------

def test_per_class_bars_draw_the_same_bars(grouped, inputs):
    metrics = inputs[0]
    per_class = {n: m.per_class.set_index("label") for n, m in metrics.items()}
    labels = sorted(set().union(*[set(pc.index) for pc in per_class.values()]))
    plain, with_groups = _both(grouped, lambda colors, names: charts.per_class_bars(
        per_class, "f1", labels, colors, names=names)[0])
    assert _drawn(plain) == _drawn(with_groups)
    assert _legend_labels(plain) == [[CKPT_1, CKPT_2, OTHER]]
    assert _legend_labels(with_groups) == [["run A", OTHER]]


def test_roc_pr_curves_draw_the_same_curves_and_drop_the_numbers(grouped, inputs):
    curves = inputs[1]
    plain, with_groups = _both(grouped, lambda colors, labels: charts.roc_pr_curves(
        curves, MACRO_AVERAGE, colors, labels))
    assert _drawn(plain) == _drawn(with_groups)
    roc, pr = _legend_labels(plain)
    assert len(roc) == 3 and all("AUC=" in t for t in roc) and all("AP=" in t for t in pr)
    assert _legend_labels(with_groups) == [["run A", OTHER], ["run A", OTHER]]


def test_risk_coverage_draws_the_same_curves_and_drops_the_aurc(grouped, inputs):
    risk = inputs[2]
    plain, with_groups = _both(grouped, lambda colors, labels: charts.risk_coverage(risk, colors, labels))
    assert _drawn(plain) == _drawn(with_groups)
    assert all("AURC=" in t for t in _legend_labels(plain)[0])
    assert _legend_labels(with_groups) == [["run A", OTHER]]


def test_slice_chart_draws_the_same_lines_and_bars(grouped):
    names = list(grouped.experiments)
    _spec, reports = tools.compute_slices(grouped, "subset", names)
    values = [row.value for row in reports[names[0]].rows]
    plain, with_groups = _both(grouped, lambda colors, labels: charts.slice_chart(
        values, reports, "accuracy", "accuracy", "subset", colors, labels=labels)[0])
    assert _drawn(plain) == _drawn(with_groups)
    entries = _legend_labels(with_groups)[0]
    assert entries.count("run A") == 1 and OTHER in entries


def test_one_experiment_charts_draw_the_same_and_take_the_group_name(grouped, inputs):
    reliability = inputs[3][CKPT_1]
    plain, with_groups = _both(grouped, lambda colors, labels: charts.reliability_diagram(
        CKPT_1, reliability, colors[CKPT_1], label=(labels or {}).get(CKPT_1)))
    assert _drawn(plain) == _drawn(with_groups)
    assert plain.axes[0].get_title().startswith(CKPT_1)
    assert with_groups.axes[0].get_title().startswith("run A")


def test_with_no_group_shown_a_chart_is_what_it_always_was(state, inputs):
    """Hidden groups, or none at all: the chart gets no labels, and is the
    chart Lookin drew before groups existed."""
    state.groups.clear()
    curves = inputs[1]
    names = list(state.experiments)
    assert state.chart_labels(names) is None
    colors = {n: color_for_experiment(n, state.color_slots) for n in names}
    before = charts.roc_pr_curves(curves, MACRO_AVERAGE, colors)
    state.groups.create("run A", [CKPT_1, CKPT_2])
    state.groups.set_shown(False)
    try:
        assert state.chart_labels(names) is None
        after = charts.roc_pr_curves(curves, MACRO_AVERAGE,
                                     {n: color_for_experiment(n, state.color_slots) for n in names},
                                     state.chart_labels(names))
        assert _drawn(before) == _drawn(after)
        assert _legend_labels(before) == _legend_labels(after)
        assert [line.get_color() for line in before.axes[0].get_lines()] == \
               [line.get_color() for line in after.axes[0].get_lines()]
    finally:
        state.groups.clear()


# ---- colors ------------------------------------------------------------------

def test_members_share_the_color_their_first_member_had(grouped):
    grouped.groups.set_shown(False)
    own = {n: color_for_experiment(n, grouped.color_slots) for n in grouped.experiments}
    grouped.groups.set_shown(True)
    shown = {n: color_for_experiment(n, grouped.color_slots) for n in grouped.experiments}
    assert shown[CKPT_1] == shown[CKPT_2] == own[CKPT_1]
    assert shown[OTHER] == own[OTHER] != shown[CKPT_1]


def test_a_members_own_color_is_kept_underneath(grouped):
    grouped.set_experiment_color(CKPT_2, "#123456")
    try:
        assert color_for_experiment(CKPT_2, grouped.color_slots) != "#123456"
        grouped.groups.set_shown(False)
        assert color_for_experiment(CKPT_2, grouped.color_slots) == "#123456"
        grouped.groups.set_shown(True)
        grouped.groups.assign(CKPT_2, "")
        assert color_for_experiment(CKPT_2, grouped.color_slots) == "#123456"
    finally:
        grouped.set_experiment_color(CKPT_2, "")


def test_a_group_color_paints_every_member(grouped):
    gid = grouped.groups.by_name("run A").id
    grouped.groups.set_color(gid, "#AA0011")
    assert color_for_experiment(CKPT_1, grouped.color_slots) == "#aa0011"
    assert color_for_experiment(CKPT_2, grouped.color_slots) == "#aa0011"
    assert grouped.colors_with_groups(False)[CKPT_1] != "#aa0011"


def test_a_group_never_takes_an_ungrouped_experiments_color():
    s = AppState()
    s.experiments = {n: object() for n in ("a", "b", "c")}
    slots = s.own_color_slots()
    # "a" is its first member (in experiment order): its slot is free, so
    # the group takes it ...
    s.groups.create("g", ["c", "a"])
    colors = s.color_slots
    assert colors["a"] == colors["c"] == slots["a"]
    assert colors["b"] == slots["b"]
    # ... and when "a" leaves, holding that slot again, the group moves.
    s.groups.assign("a", "")
    colors = s.color_slots
    assert colors["a"] == slots["a"] and colors["b"] == slots["b"]
    assert colors["c"] not in (colors["a"], colors["b"])


def test_renaming_and_removing_an_experiment_carry_its_group(grouped):
    grouped.rename_experiment(CKPT_2, "run-A/last")
    try:
        assert grouped.groups.group_of("run-A/last").name == "run A"
    finally:
        grouped.rename_experiment("run-A/last", CKPT_2)


# ---- the model -----------------------------------------------------------------

def test_group_rules():
    groups = eg.ExperimentGroups()
    a = groups.create("A", ["x", "y"])
    with pytest.raises(ValueError):
        groups.create("A", ["z"])
    with pytest.raises(ValueError):
        groups.create("  ", ["z"])
    with pytest.raises(ValueError):
        groups.create("B", ["z"], color="red")
    b = groups.create("B", ["y", "z"])            # y moves from A to B
    assert groups.members(a.id, "xyz") == ["x"] and groups.members(b.id, "xyz") == ["y", "z"]
    groups.assign("x", "")                       # A is left empty: gone
    assert groups.get(a.id) is None
    with pytest.raises(ValueError):
        groups.rename(b.id, "")
    assert groups.rename(b.id, "B2") and groups.resolve("B2") is b and groups.resolve(b.id) is b
    with pytest.raises(ValueError):
        groups.resolve("nope")
    assert groups.chart_labels(["x", "y", "z"]) == {"x": "x", "y": "B2", "z": "B2"}
    assert groups.chart_labels(["x"]) is None
    groups.set_shown(False)
    assert groups.chart_labels(["x", "y", "z"]) is None
    assert groups.delete(b.id) and not groups.member_of


def test_group_added_by_run_only_groups_what_was_added_together():
    groups = eg.ExperimentGroups()
    made = eg.group_added(groups, [("r1/c1", "r1", "run 1"), ("r1/c2", "r1", "run 1"), ("r2", "r2", "run 2")],
                          eg.GROUPING_BY_RUN)
    assert [g.name for g in made] == ["run 1"]
    assert groups.group_of("r2") is None
    # A later checkpoint of run 1 joins nothing by itself.
    assert eg.group_added(groups, [("r1/c3", "r1", "run 1")], eg.GROUPING_BY_RUN) == []
    assert groups.group_of("r1/c3") is None
    # The same run added again makes a new group, under a name of its own.
    made = eg.group_added(groups, [("r1/c4", "r1", "run 1"), ("r1/c5", "r1", "run 1")], eg.GROUPING_BY_RUN)
    assert [g.name for g in made] == ["run 1 (2)"]
    assert eg.group_added(groups, [("a", "r3", "run 3"), ("b", "r3", "run 3")], eg.GROUPING_NONE) == []
    made = eg.group_added(groups, [("a", "r3", "run 3"), ("b", "r4", "run 4")], eg.GROUPING_ONE, "mine")
    assert [g.name for g in made] == ["mine"] and groups.members(made[0].id, ["a", "b"]) == ["a", "b"]


def test_groups_round_trip_through_a_manifest():
    mount = lib.Mount(dataset_path="d.csv", experiments=[lib.MountExperiment(name=n, path=f"{n}.csv")
                                                          for n in ("a", "b", "c")])
    groups = eg.from_mount(mount)
    groups.create("G", ["a", "c"], color="#00FF00")
    groups.set_shown(False)
    eg.write_to_mount(groups, mount)
    again = lib.mount_from_dict(lib.mount_to_dict(mount))
    back = eg.from_mount(again)
    assert back.shown is False
    assert [(g.name, g.color) for g in back.in_use("abc")] == [("G", "#00ff00")]
    assert back.members(back.by_name("G").id, "abc") == ["a", "c"]


def test_a_manifest_from_before_groups_opens_with_none():
    raw = {"dataset_path": "d.csv", "experiments": [{"name": "a", "path": "a.csv"}]}
    mount = lib.mount_from_dict(raw)
    assert mount.experiment_groups == [] and mount.show_experiment_groups is True
    assert mount.experiments[0].group == ""
    assert not eg.from_mount(mount).groups


def test_add_candidates_groups_the_checkpoints_of_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    run = mlf.Run(run_id="abcdef0123", run_name="train-A", status="FINISHED", start_time=0)
    csvs = [mlf.ArtifactFile(path=f"eval/ckpt_{i}.csv", size=10) for i in (1, 2)]
    candidates = wss.candidates_for_run("http://mlflow", run, csvs, [])
    workspace = lib.Workspace(id="w", name="w", mount=lib.Mount())
    result = wss.add_candidates(workspace, candidates, grouping=eg.GROUPING_BY_RUN)
    assert result.groups == ["train-A"]
    groups = eg.from_mount(workspace.mount)
    assert groups.members(groups.by_name("train-A").id, [e.name for e in workspace.mount.experiments]) == \
        ["train-A/ckpt_1", "train-A/ckpt_2"]
    # Nothing grouped when asked not to.
    workspace = lib.Workspace(id="w2", name="w2", mount=lib.Mount())
    assert wss.add_candidates(workspace, candidates).groups == []
    assert workspace.mount.experiment_groups == []


# ---- the MCP tools -------------------------------------------------------------

@pytest.fixture
def workspace_id(tmp_path, monkeypatch):
    lib.set_library_root(tmp_path / "library")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    links.set_base_url("http://lookin.lan:8560")
    tools.clear_cache()
    example = tmp_path / "ex"
    shutil.copytree(EXAMPLE, example)
    ws = actions.create_workspace("Groups", dataset_path=str(example / "dataset.csv"))
    actions.add_experiment(ws["id"], str(example / "baseline_cnn_results.csv"), name=CKPT_1)
    actions.add_experiment(ws["id"], str(example / "finetuned_resnet_results.csv"), name=CKPT_2, group="run A")
    actions.add_experiment(ws["id"], str(example / "baseline_cnn_results.csv"), name=OTHER)
    yield ws["id"]
    lib.set_library_root(None)
    links.set_base_url("")
    tools.clear_cache()


def test_mcp_tools_manage_groups_like_the_interface(workspace_id):
    out = actions.set_experiment_group(workspace_id, CKPT_1, "run A")
    assert out["experiment_groups"] == [{"id": out["experiment_groups"][0]["id"], "name": "run A",
                                         "color": "", "experiments": [CKPT_1, CKPT_2]}]
    assert out["open_url"]
    with pytest.raises(ValueError):
        actions.set_experiment_group(workspace_id, CKPT_1, "no such group")
    with pytest.raises(ValueError):
        actions.create_experiment_group(workspace_id, "run A", [OTHER])
    actions.rename_experiment_group(workspace_id, "run A", "Run A")
    actions.set_experiment_group_color(workspace_id, "Run A", "#112233")
    out = actions.show_experiment_groups(workspace_id, False)
    assert out["experiment_groups_shown"] is False
    described = tools.get_workspace(workspace_id)
    assert described["experiment_groups"][0]["name"] == "Run A"
    assert described["experiment_groups"][0]["color"] == "#112233"
    assert {e["name"]: e.get("experiment_group") for e in described["experiments"]} == \
        {CKPT_1: "Run A", CKPT_2: "Run A", OTHER: None}
    # Removing a member, then dissolving: the experiments stay.
    out = actions.delete_experiment_group(workspace_id, "Run A")
    assert out["ungrouped"] == [CKPT_1, CKPT_2] and out["experiment_groups"] == []
    assert [e["name"] for e in tools.get_workspace(workspace_id)["experiments"]] == [CKPT_1, CKPT_2, OTHER]


def test_a_removed_member_leaves_no_empty_group(workspace_id):
    actions.remove_experiment(workspace_id, CKPT_2)
    assert tools.get_workspace(workspace_id)["experiment_groups"] == []


def test_plot_chart_follows_the_switch_or_the_argument(workspace_id):
    actions.set_experiment_group(workspace_id, CKPT_1, "run A")
    answer, _png = outputs.plot_chart(workspace_id, "roc_pr")
    assert any("AUC and AP" in n for n in answer.get("notes", []))
    answer, _png = outputs.plot_chart(workspace_id, "roc_pr", groups=False)
    assert not any("AUC and AP" in n for n in answer.get("notes", []))
    actions.show_experiment_groups(workspace_id, False)
    answer, _png = outputs.plot_chart(workspace_id, "roc_pr")
    assert not any("AUC and AP" in n for n in answer.get("notes", []))


def test_unique_legend_entries_keeps_same_name_different_color():
    from lookin.core.chart_style import unique_legend_entries

    lines = [SimpleNamespace(get_color=lambda c=c: c) for c in ("#ff0000", "#ff0000", "#00ff00")]
    handles, labels = unique_legend_entries(lines, ["g", "g", "g"])
    assert labels == ["g", "g"] and [to_hex(h.get_color()) for h in handles] == ["#ff0000", "#00ff00"]
