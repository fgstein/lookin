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
    for text in ("How far apart the datasets are", "Per class", 'data-target="embcompare"', "Outside the reference"):
        assert text in html, text
    # A click on the map: the crop's nearest crops in the other base.
    with RequestScope(s):
        view.on_point(i=0)
        dialog = s.dialogs[-1]
        body = dialog.body()
    assert "Nearest crops" in dialog.title and "Search in" in body
    # Side by side, coloured by class: a redraw, no new projection.
    cmap = view.cmap
    view.on_color(value=ec.COLOR_CLASS)
    view.display = "panels"
    view._draw()
    assert view.cmap is cmap and view.panels_html and not view.map_html


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
