"""Compare embeddings: one model's space seen through several datasets.

Built on the bundled example: two workspaces over two different halves of
its dataset (the clean crops, and the blurry or occluded ones), both scored
by the same model -- the same embeddings and prototypes files -- and a
second model in the second workspace, which must be refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lookin.core import embedding_compare as ec
from lookin.core import library as lib
from lookin.core.library import Mount, MountExperiment

EXAMPLE = Path(__file__).resolve().parents[1] / "src" / "lookin" / "data" / "example"


def _experiment(name: str, stem: str) -> MountExperiment:
    return MountExperiment(name=name, path=str(EXAMPLE / f"{stem}_results.csv"),
                           embeddings_path=str(EXAMPLE / f"{stem}_embeddings.npz"),
                           prototypes_path=str(EXAMPLE / f"{stem}_prototypes.npz"))


def build_two_bases(tmp_path: Path, root: Path) -> tuple:
    """(clean workspace, degraded workspace): the example dataset split in
    two, each half its own CSV, its crops resolved from the example."""
    df = pd.read_csv(EXAMPLE / "dataset.csv")
    clean, degraded = tmp_path / "clean.csv", tmp_path / "degraded.csv"
    df[df["quality_flag"] == "ok"].to_csv(clean, index=False)
    df[df["quality_flag"] != "ok"].to_csv(degraded, index=False)
    a = lib.create_workspace("Clean base", Mount(dataset_path=str(clean), image_root=str(EXAMPLE), split_kind="test",
                                                 experiments=[_experiment("cnn", "baseline_cnn")]), root=root)
    b = lib.create_workspace("Degraded base", Mount(
        dataset_path=str(degraded), image_root=str(EXAMPLE), split_kind="test",
        experiments=[_experiment("cnn", "baseline_cnn"), _experiment("resnet", "finetuned_resnet")]), root=root)
    return a, b


@pytest.fixture
def library_root(tmp_path, monkeypatch):
    root = tmp_path / "library"
    lib.set_library_root(root)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    ec.clear_cache()
    yield root
    lib.set_library_root(None)
    ec.clear_cache()


@pytest.fixture
def bases(tmp_path, library_root):
    return build_two_bases(tmp_path, library_root)


def test_a_member_is_read_from_its_own_workspace(bases):
    a, b = bases
    member = ec.load_member(ec.MemberRef(b.id, "cnn"))
    assert member.name == "Degraded base / cnn"
    assert len(member) == 199
    assert member.embeddings.shape == (199, 64)
    assert member.prototypes is not None
    assert set(member.labels.tolist()) <= {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                                           "horse", "ship", "truck"}
    record = member.record(0)
    assert record["experiment"] == member.name and Path(record["path"]).is_file()
    assert "predicted_label" in record and record["quality_flag"] != "ok"


def test_the_same_model_is_allowed_and_another_is_refused(bases):
    a, b = bases
    same, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    assert ec.comparison_problem(same, 0) == ""
    other, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "resnet")])
    problem = ec.comparison_problem(other, 0)
    assert "prototypes differ" in problem


def test_missing_prototypes_refuse_the_comparison(bases):
    a, b = bases
    members, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    members[1].prototypes = None
    assert "no model prototypes" in ec.comparison_problem(members, 0)


def test_prototypes_are_matched_by_label_not_by_position(bases):
    a, _ = bases
    member = ec.load_member(ec.MemberRef(a.id, "cnn"))
    protos = member.prototypes
    shuffled = ec.PrototypeSet(vectors=protos.vectors[::-1].copy(), class_labels=protos.class_labels[::-1],
                               source=protos.source, k=protos.k)
    assert ec.check_prototypes(protos, shuffled).ok


def test_a_member_that_cannot_be_read_is_reported_not_raised(bases):
    a, _ = bases
    members, problems = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(a.id, "nope")])
    assert len(members) == 1 and "nope" in problems[0]


@pytest.mark.parametrize("mode,method", [(ec.PROJ_JOINT, "pca"), (ec.PROJ_REFERENCE, "pca"),
                                         (ec.PROJ_JOINT, "tsne")])
def test_both_projection_modes_place_every_member(bases, mode, method):
    a, b = bases
    members, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    cmap = ec.project_members(members, 0, mode=mode, method=method, per_member=150)
    assert [len(p) for p in cmap.points] == [150, 150]
    assert all(np.isfinite(p).all() for p in cmap.points)
    assert cmap.proto_points is not None and len(cmap.proto_points) == len(cmap.proto_labels) == 20


def test_the_reference_projection_is_the_reference_alone(bases):
    """Projected onto the reference, the reference's own points do not
    depend on what else is compared."""
    a, b = bases
    members, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    both = ec.project_members(members, 0, mode=ec.PROJ_REFERENCE, method="pca", per_member=150)
    alone = ec.project_members(members[:1], 0, mode=ec.PROJ_REFERENCE, method="pca", per_member=150)
    assert np.allclose(both.points[0], alone.points[0])


def test_tsne_cannot_project_onto_a_reference(bases):
    a, b = bases
    members, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    with pytest.raises(ValueError, match="t-SNE"):
        ec.project_members(members, 0, mode=ec.PROJ_REFERENCE, method="tsne")


def test_neighbours_across_search_the_other_member(bases):
    a, b = bases
    members, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    found = ec.neighbours_across(members[0], 3, members[1], k=5)
    assert len(found) == 5
    assert all(n.uuid in set(members[1].uuids.tolist()) for n in found)
    assert [n.similarity for n in found] == sorted((n.similarity for n in found), reverse=True)
    own = ec.neighbours_across(members[0], 3, members[0], k=5)
    assert members[0].uuids[3] not in {n.uuid for n in own}


def test_nearest_k_matches_a_full_sort():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(300, 8)).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    sims, idx = ec.nearest_k(pool[:40], pool, 5, exclude_self=True, max_cells=1000)
    full = pool[:40] @ pool.T
    np.fill_diagonal(full[:, :40], -np.inf)
    assert np.array_equal(idx, np.argsort(-full, axis=1)[:, :5])
    assert np.allclose(sims, np.sort(full, axis=1)[:, ::-1][:, :5])


def test_a_base_against_itself_is_covered(bases):
    a, _ = bases
    member = ec.load_member(ec.MemberRef(a.id, "cnn"))
    stats = ec.pair_stats(member, member)
    # Every crop is in both: nothing is left to tell the bases apart with.
    assert stats.n_shared == 601 and stats.domain_auc is None
    assert stats.coverage > 0.9
    assert stats.agreement == pytest.approx(1.0) or stats.agreement >= stats.agreement_reference


def test_pair_stats_measure_the_other_base(bases):
    a, b = bases
    members, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    stats = ec.pair_stats(members[0], members[1])
    assert stats.n_other == 199 and stats.n_shared == 0
    assert 0.0 <= stats.coverage <= 1.0 and 0.0 <= stats.domain_auc <= 1.0
    assert stats.agreement is not None and stats.agreement_reference is not None
    assert {c.label for c in stats.per_class} == set(members[1].labels.tolist()) | set(members[0].labels.tolist())
    for row, own, theirs, share in stats.disagreements:
        assert members[1].labels[row] == own != theirs and 0 < share <= 1
    for row, sim in stats.outside:
        assert sim < stats.coverage_threshold
    worst = ec.sorted_classes(stats.per_class, "coverage")
    assert worst[0].coverage <= worst[-1].coverage


def test_the_comparison_is_saved_as_references_in_the_manifest(bases):
    a, b = bases
    refs = [ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")]

    def change(fresh):
        fresh.embedding_comparison = ec.comparison_to_json(refs, 1)
        return True

    lib.update_workspace(a.path, change)
    raw = json.loads((Path(a.path) / lib.MANIFEST_NAME).read_text())
    assert raw["embedding_comparison"] == {"members": [{"workspace": a.id, "experiment": "cnn"},
                                                       {"workspace": b.id, "experiment": "cnn"}], "reference": 1}
    again = lib.load_workspace(a.path)
    assert ec.comparison_from_json(again.embedding_comparison) == (refs, 1)
    # A copy compares its own experiment, and still the other workspace's.
    copy = lib.duplicate_workspace(again, "Clean copy")
    copied, _ = ec.comparison_from_json(copy.embedding_comparison)
    assert copied == [ec.MemberRef(copy.id, "cnn"), ec.MemberRef(b.id, "cnn")]


def test_comparable_experiments_lists_the_library(bases):
    listed = ec.comparable_experiments()
    assert {(x["workspace"], x["experiment"]) for x in listed} == {
        ("Clean base", "cnn"), ("Degraded base", "cnn"), ("Degraded base", "resnet")}
    assert all(x["embeddings"] and x["prototypes"] for x in listed)


# ---- the MCP tools -------------------------------------------------------------

def test_the_mcp_tool_measures_the_saved_comparison(bases):
    from lookin.core import agent_actions as actions
    from lookin.core import agent_tools as tools

    a, b = bases
    members = [{"workspace": "Clean base", "experiment": "cnn"}, {"workspace": b.id, "experiment": "cnn"}]
    saved = actions.set_embedding_comparison(a.id, members, 0)
    assert "tab=embcompare" in saved["open_url"]
    assert tools.get_workspace(a.id)["embedding_comparison"]["members"][1] == {"workspace": b.id, "experiment": "cnn"}
    out = tools.compare_embedding_spaces(a.id, top=3)
    json.dumps(out)
    assert out["compared"] is True
    assert out["reference"] == "Clean base / cnn"
    pair = out["against_reference"][0]
    assert pair["member"] == "Degraded base / cnn" and pair["crops_measured"] == 199
    assert len(pair["classes"]) == 3 and pair["classes_total"] == 10
    assert all("open_url" in c for c in pair["outside_reference"] + pair["label_disagreements"])


def test_the_mcp_tools_refuse_two_models(bases):
    from lookin.core import agent_actions as actions
    from lookin.core import agent_tools as tools

    a, b = bases
    members = [{"workspace": a.id, "experiment": "cnn"}, {"workspace": b.id, "experiment": "resnet"}]
    out = tools.compare_embedding_spaces(a.id, members)
    assert out["compared"] is False and "prototypes differ" in out["refused"]
    assert out["members"][1]["same_model_as_reference"] is False
    with pytest.raises(ValueError, match="Not saved"):
        actions.set_embedding_comparison(a.id, members, 0)
    uuid = str(ec.load_member(ec.MemberRef(a.id, "cnn")).uuids[0])
    with pytest.raises(ValueError, match="Not compared"):
        tools.find_neighbours_across(a.id, "cnn", uuid, b.id, "resnet")


def test_the_mcp_neighbours_come_from_the_other_dataset(bases):
    from lookin.core import agent_tools as tools

    a, b = bases
    uuid = str(ec.load_member(ec.MemberRef(a.id, "cnn")).uuids[0])
    out = tools.find_neighbours_across(a.id, "cnn", uuid, "Degraded base", "cnn", k=5)
    assert out["k"] == 5 and out["searched_in"] == "Degraded base / cnn"
    other = set(ec.load_member(ec.MemberRef(b.id, "cnn")).uuids.tolist())
    assert all(n["uuid"] in other and b.id in n["open_url"] for n in out["neighbours"])


def test_the_mcp_picture_is_the_tab_chart(bases):
    from lookin.core import agent_outputs as outputs

    a, b = bases
    members = [{"workspace": a.id, "experiment": "cnn"}, {"workspace": b.id, "experiment": "cnn"}]
    for display in ("overlay", "panels"):
        answer, png = outputs.plot_embedding_comparison(a.id, members, 1, mode="reference", color_by="class",
                                                        display=display, max_points=100)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert answer["reference"] == "Degraded base / cnn" and answer["points_per_member"] == [100, 100]


def test_a_rename_follows_into_every_comparison(bases):
    from lookin.core import agent_actions as actions

    a, b = bases
    actions.set_embedding_comparison(a.id, [{"workspace": a.id, "experiment": "cnn"},
                                            {"workspace": b.id, "experiment": "cnn"}], 0)
    actions.rename_experiment(b.id, "cnn", "cnn-v1")
    refs, _ = ec.comparison_from_json(lib.load_workspace(a.path).embedding_comparison)
    assert refs[1] == ec.MemberRef(b.id, "cnn-v1")


# ---- the tab ---------------------------------------------------------------------

def _tab_session(workspace):
    from lookin.core import workspace_state as ws
    from lookin.web import render
    from lookin.web.server import WebApp
    from lookin.web.session import RequestScope

    session = WebApp().store.get("x" * 24, create=True)
    with RequestScope(session):
        render.boot(session, {})
    loaded = ws.load_workspace_state(workspace)
    for name in loaded.state.experiments:
        ws.attach_sidecars(workspace, loaded.state, name, prototypes=True)
    session.state = loaded.state
    session.current_workspace = workspace
    session.mode, session.tab = "analysis", "embcompare"
    return session


def test_the_tab_compares_and_saves(bases):
    from lookin.web import render
    from lookin.web.session import RequestScope

    a, b = bases
    s = _tab_session(a)
    view = s.view("embcompare")
    with RequestScope(s):
        html = render.render_view(s, "embcompare")
    # Nothing saved yet: this workspace's experiment is the reference, the
    # other base's same model is offered, its other model is not.
    assert view.refs == [ec.MemberRef(a.id, "cnn")]
    assert "Degraded base / cnn" in html and "Degraded base / resnet" not in html
    view.on_choose(value=f"{b.id}/cnn")
    view.on_add()
    assert ec.comparison_from_json(lib.load_workspace(a.path).embedding_comparison)[0] == view.refs
    view._compute()
    assert not view.refusal and set(view.stats) == {1}
    with RequestScope(s):
        html = render.render_view(s, "embcompare")
    for text in ("How far apart the datasets are", "Per class", 'data-target="embcompare"', " outside",
                 "Probe accuracy", 'data-pm-mode="nn"'):
        assert text in html, text
    assert set(view.probes) == {0, 1} and view.probes[0].accuracy > 0.3
    # A click on the map: the crop's nearest crops in the other base.
    with RequestScope(s):
        view.on_point(i=0)
        dialog = s.dialogs[-1]
        body = dialog.body()
    assert "Nearest crops" in dialog.title and "Search in" in body
    # The map's Neighbours tool: the nearest points of the map, among the
    # other experiment's, sent back for the map to ring.
    view.on_nn_scope(value="other")
    with RequestScope(s):
        view.on_neighbours(i=0, k=5, map="m1")
        command = s._pending_cmds[-1]
    detail = command["detail"]
    assert command["event"] == "map-neighbours" and detail["map"] == "m1" and detail["query"] == 0
    assert len(detail["idx"]) == 5 and detail["sims"] == sorted(detail["sims"], reverse=True)
    assert all(view._map_points[j][0] != view._map_points[0][0] for j in detail["idx"])
    # Side by side, coloured by class: a redraw, no new projection -- one
    # interactive map per experiment, linked, of the same points.
    cmap = view.cmap
    view.on_color(value=ec.COLOR_CLASS)
    view.display = "panels"
    view._draw()
    assert view.cmap is cmap and view.panels_html and not view.map_html
    assert view.panels_html.count('class="pmap pmap-compact"') == 2
    assert view.panels_html.count("data-link=") == 2
    # Every pair, on demand.
    view.pairwise = ec.pairwise_stats(view.members)
    assert view.pairwise.domain_auc.shape == (2, 2)
    with RequestScope(s):
        html = render.render_view(s, "embcompare")
    assert "Every pair of experiments" in html


def test_the_tab_refuses_a_saved_comparison_of_two_models(bases):
    from lookin.web import render
    from lookin.web.session import RequestScope

    a, b = bases

    def change(fresh):
        fresh.embedding_comparison = ec.comparison_to_json([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "resnet")], 0)
        return True

    s = _tab_session(lib.update_workspace(a.path, change))
    view = s.view("embcompare")
    view.on_shown()
    view._compute()
    assert view.refusal and view.cmap is None
    with RequestScope(s):
        html = render.render_view(s, "embcompare")
    assert "prototypes differ" in html


def test_add_all_takes_every_experiment_of_the_same_model(bases, tmp_path):
    a, b = bases
    c = lib.create_workspace("Third base", lib.Mount(
        dataset_path=b.mount.dataset_path, image_root=str(EXAMPLE), experiments=[_experiment("cnn", "baseline_cnn")]),
        root=lib.library_root())
    s = _tab_session(a)
    view = s.view("embcompare")
    view.on_shown()
    view.on_add_all()
    assert view.refs == [ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn"), ec.MemberRef(c.id, "cnn")]


def test_pairwise_matrices():
    rng = np.random.default_rng(0)

    def member(name, shift, n=300):
        x = rng.normal(size=(n, 16))
        x[:, 0] += shift
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return ec.Member(ref=ec.MemberRef("w", name), workspace_name="w", embeddings=x.astype(np.float32),
                         uuids=np.asarray([f"{name}{i}" for i in range(n)]), labels=np.asarray([""] * n, dtype=object),
                         predicted=None, correct=None, crops=np.asarray([""] * n), image_root=Path("."))

    near_a, near_b, far = member("a", 4.0), member("b", 4.0), member("c", -4.0)
    pw = ec.pairwise_stats([near_a, near_b, far])
    assert pw.domain_auc[0, 1] < 0.7 < pw.domain_auc[0, 2]
    assert pw.coverage[0, 1] > pw.coverage[0, 2]
    assert np.isnan(pw.domain_auc[0, 0])


def test_the_linear_probe_recovers_separable_classes():
    from lookin.core.embedding_analysis import linear_probe

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(6, 32)) * 3
    labels = np.repeat(np.arange(6), 40)
    x = centers[labels] + rng.normal(size=(len(labels), 32))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    names = np.asarray([f"c{i}" for i in labels], dtype=object)
    names[:3] = "rare"            # three crops: too few to cross-validate
    probe = linear_probe(x, names, predicted=names)
    assert probe.accuracy > 0.9 and probe.model_accuracy == 1.0
    assert probe.skipped_classes == ["rare"] and probe.n_classes == 6
    assert {c.label for c in probe.per_class} == {f"c{i}" for i in range(6)}


def test_the_embeddings_tab_neighbours_and_probe():
    """The Embeddings tab's map: its Neighbours tool and the probe section."""
    from lookin.web import render
    from lookin.web.server import WebApp
    from lookin.web.session import RequestScope

    session = WebApp().store.get("y" * 24, create=True)
    with RequestScope(session):
        render.boot(session, {})
    sources = session.view("sources")
    sources.set_dataset_path(str(EXAMPLE / "dataset.csv"))
    assert sources.load_dataset_now(then_experiments=[("cnn", str(EXAMPLE / "baseline_cnn_results.csv"))])
    session.state.experiments["cnn"].load_embeddings(str(EXAMPLE / "baseline_cnn_embeddings.npz"))
    view = session.view("embeddings")
    view._map_vector_parts = (session.state.experiments["cnn"].embeddings, np.arange(50), [], None)
    view._map_vectors_cache = None
    with RequestScope(session):
        view.on_neighbours(i=3, k=7, map="mk")
        command = session._pending_cmds[-1]
    assert command["event"] == "map-neighbours" and len(command["detail"]["idx"]) == 7
    assert 3 not in command["detail"]["idx"]
    from lookin.web.views.embeddings import probe_html
    from lookin.core.embedding_analysis import linear_probe
    from lookin.core.embedding_context import build_analysis_context

    ctx = build_analysis_context(session.state, "cnn")
    html = probe_html(linear_probe(ctx["embeddings"], ctx["labels"], ctx["merged"]["predicted_label"].to_numpy()),
                      "cnn")
    assert "Probe accuracy" in html and "The model itself" in html


def test_the_mcp_probe_and_every_pair(bases):
    from lookin.core import agent_tools as tools

    a, b = bases
    out = tools.get_class_separability(a.id, "cnn", top=3)
    json.dumps(out)
    assert 0.3 < out["probe_accuracy"] <= 1 and out["model_accuracy"] is not None
    assert len(out["worst_classes"]) == 3 and out["classes"] == 10
    members = [{"workspace": a.id, "experiment": "cnn"}, {"workspace": b.id, "experiment": "cnn"}]
    out = tools.compare_embedding_spaces(a.id, members, pairwise=True, probe=True)
    json.dumps(out)
    assert len(out["pairwise"]["domain_auc"]) == 2 and out["members"][1]["probe_accuracy"] is not None


# ---- which crops a projection is fitted on, and which it shows -------------------

def test_the_fit_set_keeps_the_real_distribution_or_balances_it():
    from lookin.core import projection_plan as pp

    labels = np.asarray(["big"] * 900 + ["small"] * 100, dtype=object)
    real = pp.choose_rows(labels, pp.ProjectionPlan(fit_size=200, display_size=600))
    assert len(real.fit) == 200 and 140 < real.fit_by_class["big"] < 200
    assert len(real.display) == 600 and set(real.fit) <= set(real.display)
    balanced = pp.choose_rows(labels, pp.ProjectionPlan(sampling=pp.SAMPLING_BALANCED, per_class=150,
                                                        display_size=1000))
    assert balanced.fit_by_class == {"big": 150, "small": 100}
    assert "fewer than 150" in balanced.notes[0]
    assert len(balanced.display) == 1000
    tsne = pp.choose_rows(labels, pp.ProjectionPlan(method="tsne", fit_size=100, display_size=1000))
    assert np.array_equal(tsne.fit, tsne.display) and "t-SNE" in tsne.notes[0]


def test_crops_placed_onto_a_fit_land_where_the_fit_would_put_them():
    from lookin.core import projection_plan as pp

    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 12)).astype(np.float32)
    labels = np.asarray(["a"] * 500, dtype=object)
    rows = pp.choose_rows(labels, pp.ProjectionPlan(fit_size=100, display_size=500))
    coords, _, fit = pp.project(x, rows, "pca")
    assert coords.shape == (500, 2)
    assert np.allclose(coords, fit.transform(x[rows.display]), atol=1e-5)


def test_plans_round_trip_and_are_clamped():
    from lookin.core import projection_plan as pp

    plan = pp.plan_from_json({"method": "umap", "sampling": "balanced", "per_class": "50", "display_size": 10**9})
    assert plan.per_class == 50 and plan.display_size == pp.MAX_DISPLAY_SIZE
    assert pp.plan_from_json(pp.plan_to_json(plan)) == plan
    assert pp.plan_from_json({"sampling": "nonsense"}).sampling == pp.SAMPLING_REAL


def test_a_comparison_follows_its_plan_and_saves_it(bases):
    from lookin.core import projection_plan as pp

    a, b = bases
    members, _ = ec.load_members([ec.MemberRef(a.id, "cnn"), ec.MemberRef(b.id, "cnn")])
    plan = pp.ProjectionPlan(sampling=pp.SAMPLING_BALANCED, per_class=5, display_size=400)
    cmap = ec.project_members(members, 0, mode=ec.PROJ_REFERENCE, plan=plan)
    assert len(cmap.fit_rows[0]) == 50 and len(cmap.fit_rows[1]) == 0
    assert [len(r) for r in cmap.rows] == [400, 199]
    s = _tab_session(a)
    view = s.view("embcompare")
    view.on_shown()
    view.on_choose(value=f"{b.id}/cnn")
    view.on_add()
    view.on_sampling(value=pp.SAMPLING_BALANCED)
    view.on_per_class(value="7")
    view._compute()
    saved_plan, _ = ec.comparison_plan(lib.load_workspace(a.path).embedding_comparison)
    assert saved_plan.sampling == pp.SAMPLING_BALANCED and saved_plan.per_class == 7
    assert "Fitted on 140 crops" in view.cmap.notes[0]


def test_the_embeddings_tab_saves_its_plan_in_the_workspace(bases):
    from lookin.core import projection_plan as pp

    a, _ = bases
    s = _tab_session(a)
    s.tab = "embeddings"
    view = s.view("embeddings")
    view.on_shown()
    view.on_sampling(value=pp.SAMPLING_BALANCED)
    view.on_per_class(value="12")
    view.plan = pp.clamp(pp.with_method(view.plan, view.method))
    view._save_plan()
    saved = pp.plan_from_json(lib.load_workspace(a.path).embedding_projection)
    assert saved.sampling == pp.SAMPLING_BALANCED and saved.per_class == 12
    # Another session on the same workspace starts from it.
    t = _tab_session(lib.load_workspace(a.path))
    other = t.view("embeddings")
    other.on_shown()
    assert other.plan.per_class == 12
