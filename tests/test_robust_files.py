"""Files on disk that are missing, malformed, or not what they were: each
must end in a clear message or a correct number, never a crash of the page
or a silently wrong result."""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from lookin.core import library as lib
from lookin.core.data_manager import AppState, ColumnConfig, read_embeddings_npz
from lookin.core.review import STATUS_EXCLUDE, ReviewStore, export_enriched_dataset, review_paths_for_csv

EXAMPLE = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---- ids -------------------------------------------------------------------

def test_zero_padded_ids_keep_their_zeros_and_match_the_embeddings(tmp_path):
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label\n0012,a.png,cat\n0013,b.png,dog\n")
    res = _write(tmp_path / "r.csv", "uuid,cat,dog\n0012,0.9,0.1\n0013,0.2,0.8\n")
    np.savez(tmp_path / "e.npz", uuid=np.array(["0012", "0013"]), embeddings=np.eye(2, 3))
    state = AppState()
    state.load_dataset(str(ds), ColumnConfig())
    exp = state.add_experiment(str(res), "e")
    exp.load_embeddings(str(tmp_path / "e.npz"))
    merged = state.merged("e")
    assert merged["uuid"].tolist() == ["0012", "0013"]
    _, mask = exp.aligned_embeddings(merged, "uuid")
    assert mask.all()


def test_a_non_numeric_id_in_the_results_does_not_break_the_join(tmp_path):
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label\n12,a.png,cat\n13,b.png,dog\n")
    res = _write(tmp_path / "r.csv", "uuid,cat,dog\n12,0.9,0.1\nabc,0.2,0.8\n")
    state = AppState()
    state.load_dataset(str(ds), ColumnConfig())
    state.add_experiment(str(res), "e")
    assert state.merged("e")["uuid"].tolist() == ["12"]


def test_float_shaped_ids_match_whole_number_ids(tmp_path):
    """An int id column with a gap in it comes back from pandas as '12.0'."""
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label\n12,a.png,cat\n14,c.png,dog\n")
    res = _write(tmp_path / "r.csv", "uuid,cat,dog\n12.0,0.9,0.1\n14.0,0.2,0.8\n")
    np.savez(tmp_path / "e.npz", uuid=np.array([12.0, 14.0]), embeddings=np.eye(2, 3))
    state = AppState()
    state.load_dataset(str(ds), ColumnConfig())
    exp = state.add_experiment(str(res), "e")
    exp.load_embeddings(str(tmp_path / "e.npz"))
    merged = state.merged("e")
    assert len(merged) == 2
    assert exp.aligned_embeddings(merged, "uuid")[1].all()


def test_export_writes_the_dataset_back_unchanged(tmp_path):
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label,size\n0012,a.png,cat,1.50\n0013,b.png,dog,\n")
    state = AppState()
    state.load_dataset(str(ds), ColumnConfig())
    store = state.review
    store.ensure_active_version()
    store.decide("0012", STATUS_EXCLUDE)
    out = tmp_path / "out" / "train.csv"
    export_enriched_dataset(store, ds, "uuid", "label", [store.active_version_id], out, freeze=False)
    written = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert written["uuid"].tolist() == ["0012", "0013"]
    assert written["size"].tolist() == ["1.50", ""]
    assert written[f"review_status_{store.active_version_id}"].tolist() == [STATUS_EXCLUDE, "unreviewed"]


# ---- CSV files -------------------------------------------------------------

def test_a_non_utf8_csv_says_so(tmp_path):
    path = tmp_path / "ds.csv"
    path.write_bytes("uuid,crop,label\n1,a.png,chat\xe9\n".encode("latin-1"))
    with pytest.raises(ValueError, match="not UTF-8"):
        AppState().load_dataset(str(path), ColumnConfig())


@pytest.mark.parametrize("which", ["dataset", "results"])
def test_an_empty_csv_says_so(tmp_path, which):
    empty = _write(tmp_path / "empty.csv", "")
    state = AppState()
    if which == "dataset":
        with pytest.raises(ValueError, match="is empty"):
            state.load_dataset(str(empty), ColumnConfig())
    else:
        state.load_dataset(str(_write(tmp_path / "ds.csv", "uuid,crop,label\n1,a,cat\n")), ColumnConfig())
        with pytest.raises(ValueError, match="is empty"):
            state.add_experiment(str(empty), "e")


def test_a_class_named_like_a_metadata_column_reads_its_scores(tmp_path):
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label,dog\n1,a.png,cat,5\n2,b.png,dog,5\n")
    res = _write(tmp_path / "r.csv", "uuid,cat,dog\n1,0.9,0.1\n2,0.2,0.8\n")
    state = AppState()
    state.load_dataset(str(ds), ColumnConfig())
    state.add_experiment(str(res), "e")
    merged = state.merged("e")
    assert merged["predicted_label"].tolist() == ["cat", "dog"]
    assert merged["dog_dataset"].tolist() == [5, 5]


def test_a_class_named_like_the_crop_column_is_refused_clearly(tmp_path):
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label\n1,a.png,crop\n")
    res = _write(tmp_path / "r.csv", "uuid,crop,weed\n1,0.9,0.1\n")
    state = AppState()
    state.load_dataset(str(ds), ColumnConfig())
    with pytest.raises(ValueError, match="also the dataset's crop column"):
        state.add_experiment(str(res), "e")


# ---- .npz sidecars ---------------------------------------------------------

def test_a_file_that_is_not_an_npz_is_not_called_a_pickle(tmp_path):
    bad = tmp_path / "e.npz"
    bad.write_bytes(b"not a zip at all")
    with pytest.raises(ValueError, match="not a valid .npz") as info:
        read_embeddings_npz(str(bad))
    assert "pickle" not in str(info.value)
    np.savez(tmp_path / "good.npz", uuid=np.array(["a"]), embeddings=np.ones((1, 3)))
    raw = (tmp_path / "good.npz").read_bytes()
    bad.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ValueError, match="not a valid .npz"):
        read_embeddings_npz(str(bad))


def test_one_dimensional_embeddings_are_refused(tmp_path):
    np.savez(tmp_path / "e.npz", uuid=np.array(["a", "b"]), embeddings=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="2-D"):
        read_embeddings_npz(str(tmp_path / "e.npz"))


def test_non_finite_embeddings_are_dropped(tmp_path):
    np.savez(tmp_path / "e.npz", uuid=np.array(["a", "b", "c"]),
             embeddings=np.array([[np.nan, 1.0], [3.0, 4.0], [np.inf, 1.0]]))
    uuids, vectors = read_embeddings_npz(str(tmp_path / "e.npz"))
    assert uuids.tolist() == ["b"]
    assert np.isfinite(vectors).all()


# ---- review sidecars -------------------------------------------------------

def test_an_empty_review_sidecar_reads_as_none(tmp_path):
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label\n1,a,cat\n")
    decisions, versions = review_paths_for_csv(ds)
    decisions.write_text("")
    versions.write_text("\n")
    state = AppState()
    state.load_dataset(str(ds), ColumnConfig())
    assert state.review is not None and not state.review.versions


def test_an_unreadable_review_sidecar_is_named_not_overwritten(tmp_path):
    ds = _write(tmp_path / "ds.csv", "uuid,crop,label\n1,a,cat\n")
    decisions, _ = review_paths_for_csv(ds)
    broken = 'version_id,uuid,status\nv1,"1,exclude\n'
    decisions.write_text(broken)
    with pytest.raises(ValueError, match="review file"):
        AppState().load_dataset(str(ds), ColumnConfig())
    assert decisions.read_text() == broken


def test_a_hand_typed_status_still_counts(tmp_path):
    decisions = _write(tmp_path / "d_review.csv", "version_id,uuid,status\nv1,1,Exclude \n")
    store = ReviewStore.load(decisions, tmp_path / "d_review_versions.csv")
    assert store.uuids_by_status(STATUS_EXCLUDE, "v1") == ["1"]


# ---- library manifests -----------------------------------------------------

def test_a_mistyped_manifest_does_not_break_its_listing(tmp_path):
    csv = _write(tmp_path / "ds.csv", "uuid,crop,label\n1,a,cat\n")
    for raw in ({"dataset_path": 5},
                {"dataset_path": str(csv), "dataset_size": "big", "dataset_mtime": 5},
                {"dataset_path": str(csv), "dataset_size": 3, "dataset_mtime": "yesterday"}):
        mount = lib.mount_from_dict(raw)
        mount.exists()
        mount.changed_on_disk()
        mount.summary()


# ---- images ----------------------------------------------------------------

def _decoded(png: bytes):
    return Image.open(io.BytesIO(png))


@pytest.mark.parametrize("lut", ["none", "minmax"])
def test_cmyk_and_palette_images_render_as_colour(tmp_path, lut):
    from lookin.core.images import LUT_MINMAX, LUT_NONE, load_and_encode_png

    rgb = (np.arange(48).reshape(4, 4, 3) * 5).astype(np.uint8)
    Image.fromarray(rgb).convert("CMYK").save(tmp_path / "cmyk.tif")
    Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=4).save(tmp_path / "pal.tif")
    chosen = LUT_NONE if lut == "none" else LUT_MINMAX
    for name in ("cmyk.tif", "pal.tif"):
        assert _decoded(load_and_encode_png(tmp_path / name, chosen)).mode == "RGB"


def test_a_float_tiff_renders(tmp_path):
    from lookin.core.images import load_and_encode_png

    data = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    data[0, 0] = np.nan
    Image.fromarray(data, mode="F").save(tmp_path / "f.tif")
    out = np.array(_decoded(load_and_encode_png(tmp_path / "f.tif")))
    assert out.dtype == np.uint8 and out.max() == 255


def test_a_decompression_bomb_gets_the_placeholder(tmp_path, monkeypatch):
    from lookin.core.images import PLACEHOLDER_SVG, crop_bytes

    Image.fromarray(np.zeros((100, 100), np.uint8)).save(tmp_path / "bomb.tif")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    assert crop_bytes(str(tmp_path / "bomb.tif"), "", 0) == (PLACEHOLDER_SVG, "image/svg+xml")
    assert crop_bytes(str(tmp_path / "bomb.tif"), "minmax", 0) == (PLACEHOLDER_SVG, "image/svg+xml")


# ---- opening a workspace whose dataset is gone ------------------------------

def test_a_workspace_whose_dataset_is_gone_is_not_scored_against_the_previous_one(tmp_path):
    from lookin.web import render
    from lookin.web.server import WebApp
    from lookin.web.session import RequestScope

    example = tmp_path / "ex"
    shutil.copytree(EXAMPLE, example)
    app = WebApp()
    session = app.store.get("r" * 24, create=True)
    with RequestScope(session):
        render.boot(session, {})
    sources = session.view("sources")
    sources.set_dataset_path(str(example / "dataset.csv"))
    assert sources.load_dataset_now(then_experiments=[("baseline", str(example / "baseline_cnn_results.csv"))])
    mount = lib.Mount(dataset_path=str(tmp_path / "moved" / "dataset.csv"), experiments=[
        lib.MountExperiment(name="resnet", path=str(example / "finetuned_resnet_results.csv"))])
    sources.restore_mount(mount)
    assert session.state.dataset is None
    assert not session.state.experiments
    assert sources.ds_status[1] == "error"
    assert [r.name for r in sources.rows if r.held] == ["resnet"]
