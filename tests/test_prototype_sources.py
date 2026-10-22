"""Where the Embeddings map's prototypes come from.

The network's imported prototypes are named the network's way (its score
columns); the dataset's classes are named the dataset's way. They used to
be matched by spelling alone and the map always drew the network's set --
so a relabelled dataset, or one whose classes the network does not have,
got prototypes for some classes or none. Two things fix that: the network's
set goes through the Sources tab's class mapping, and the map can draw
prototypes built from the dataset's own labels instead, which it does by
default whenever the network's do not cover every class.
"""
from __future__ import annotations

import numpy as np
import pytest

from lookin.core import embedding_analysis as ea
from lookin.core.prototype_source import PrototypeSourceRegistry
from lookin.web.server import WebApp
from lookin.web.views.embeddings import PROTO_FROM_DATASET, PROTO_FROM_MODEL, EmbeddingsView


# ---- core ----------------------------------------------------------------

def test_class_means_are_the_k1_kmeans_estimate():
    rng = np.random.default_rng(0)
    embeddings = ea.l2_normalize(rng.normal(size=(40, 5)).astype(np.float32))
    labels = np.array(["a", "b", "c", "d"] * 10)
    order = ["a", "b", "c", "d", "absent"]
    means = ea.class_mean_prototypes(embeddings, labels, order)
    kmeans = ea.compute_kmeans_prototypes(embeddings, labels, order, k=1)
    assert means.class_labels == order
    np.testing.assert_allclose(means.vectors[:4], kmeans.vectors[:4], atol=1e-5)
    assert np.isnan(means.vectors[4]).all()


def _protos(labels, k=1, d=3):
    vectors = np.arange(len(labels) * k * d, dtype=np.float32).reshape(len(labels), k, d) + 1
    return ea.PrototypeSet(vectors=ea.l2_normalize(vectors), class_labels=list(labels),
                           source=ea.MODEL_SOURCE, k=k)


def test_relabel_is_the_identity_when_nothing_changes():
    protos = _protos(["cat", "dog"])
    assert ea.relabel_prototypes(protos, {"cat": "cat"}) is protos


def test_relabel_renames_through_the_mapping():
    protos = _protos(["0", "1"])
    out = ea.relabel_prototypes(protos, {"0": "cat", "1": "dog"})
    assert out.class_labels == ["cat", "dog"]
    np.testing.assert_array_equal(out.vectors, protos.vectors)


def test_two_network_classes_on_one_dataset_class_become_its_subcenters():
    protos = _protos(["tabby", "siamese", "dog"])
    out = ea.relabel_prototypes(protos, {"tabby": "cat", "siamese": "cat"})
    assert out.class_labels == ["cat", "dog"]
    assert out.k == 2
    np.testing.assert_array_equal(out.vectors[0], protos.vectors[[0, 1], 0])
    np.testing.assert_array_equal(out.vectors[1, 0], protos.vectors[2, 0])
    assert np.isnan(out.vectors[1, 1]).all()


# ---- the registry --------------------------------------------------------

@pytest.fixture
def state(loaded_state, embeddings_npz, tmp_path):
    exp = loaded_state.experiments["exp"]
    exp.load_embeddings(str(embeddings_npz))
    return loaded_state


def _attach(state, tmp_path, labels):
    path = tmp_path / "protos.npz"
    np.savez(path, labels=np.array(labels), prototypes=np.eye(len(labels), 3, dtype=np.float32))
    state.experiments["exp"].load_model_prototypes(str(path))


def test_the_network_set_follows_the_class_mapping(state, tmp_path):
    _attach(state, tmp_path, ["cat", "dog", "bird"])
    registry = PrototypeSourceRegistry(state)
    assert registry.model_set_for_dataset("exp").class_labels == ["cat", "dog", "bird"]

    state.apply_label_remap({"cat": "feline"})
    renamed = registry.model_set_for_dataset("exp")
    assert renamed.class_labels == ["feline", "dog", "bird"]
    # Same object while nothing changes, so cached assignments stay valid.
    assert registry.model_set_for_dataset("exp") is renamed


# ---- the map -------------------------------------------------------------

CLASSES = ["cat", "dog"]   # the classes of the crops with an embedding


def _view(state) -> EmbeddingsView:
    session = WebApp().store.get("p" * 24, create=True)
    session.state = state
    session.prototype_source = PrototypeSourceRegistry(state)
    view = EmbeddingsView(session)
    view._sync_proto_source("exp", CLASSES)
    return view


def test_with_no_network_prototypes_none_are_drawn_unless_asked(state):
    view = _view(state)
    assert [key for key, _ in view._proto_source_options] == [PROTO_FROM_DATASET]
    assert view._proto_source_for("exp", CLASSES) is None

    view._proto_source_choice["exp"] = PROTO_FROM_DATASET
    assert view._proto_source_for("exp", CLASSES) == PROTO_FROM_DATASET


def test_a_network_that_covers_every_class_is_the_default(state, tmp_path):
    _attach(state, tmp_path, ["cat", "dog", "bird"])
    view = _view(state)
    assert view._proto_source_for("exp", CLASSES) == PROTO_FROM_MODEL
    assert not view._proto_source_note


def test_a_network_missing_a_dataset_class_defaults_to_the_dataset(state, tmp_path):
    _attach(state, tmp_path, ["cat", "bird"])
    view = _view(state)
    assert view._proto_source_for("exp", CLASSES) == PROTO_FROM_DATASET
    assert "1 of the dataset's 2 classes" in view._proto_source_note


def test_the_network_can_still_be_picked(state, tmp_path):
    _attach(state, tmp_path, ["cat", "bird"])
    view = _view(state)
    view._proto_source_choice["exp"] = PROTO_FROM_MODEL
    assert view._proto_source_for("exp", CLASSES) == PROTO_FROM_MODEL
