"""Which prototype source is active for each experiment, and everything
derived from that choice -- owned in one place instead of by a view.

Two prototype sources exist per experiment and exactly one is *active* at a
time: "model" (imported alongside the results, the only source any scored
number is allowed to come from) and "kmeans" (estimated in-tool from the
loaded pool, indicative only). See core/embedding_analysis.py for the
distinction; what lives here is only the bookkeeping of which one is
selected and what has already been computed from it.

This used to be private state on the Embeddings view, which worked exactly
as long as a single view needed it. It no longer is: the Prototypes tab
picks the source and reports on it, the Embeddings tab's 2D projection
draws the active prototypes over the point cloud, and the Data tab and the
MLflow panel both attach sidecars -- an attach that has to invalidate what
was computed from the previous ones. Rather than have one view reach into
another's attributes, all four talk to this registry.

It also owns the sidecar loading itself, for the same reason: loading is
what invalidates the caches, so the code that does one does the other, and
every caller gets identical behaviour whether the load came from the Data
tab, the MLflow panel or anywhere else.

Nothing here builds or touches controls -- views subscribe() and re-render
themselves.
"""
from __future__ import annotations

import time
from typing import Callable

from lookin.core import embedding_analysis as ea
from lookin.ui.embedding_context import build_analysis_context

# The two sources are colored consistently wherever either is drawn (the
# role bars, the projection overlay), so the colors belong with the source
# definition rather than with one view's palette.
MODEL_COLOR = "#2f5fa8"
KMEANS_COLOR = "#8a5cb5"


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class PrototypeSourceRegistry:
    def __init__(self, state):
        self.state = state
        # exp name -> ("model", None) | ("kmeans", k) -- whichever of the
        # two sources was selected last. Absent means "default to model if
        # one is loaded, else the most recent kmeans estimate if any".
        self._active: dict = {}
        # (exp name, k) -> PrototypeSet, so re-selecting a K that was
        # already estimated is free.
        self._kmeans: dict = {}
        # (exp name, source key) -> (best_class, best_k, proto_set). Global
        # nearest-prototype assignment is O(N x C x K) and independent of
        # whatever class a view happens to be showing, so it is computed
        # once per (experiment, source) and reused.
        self._assignments: dict = {}
        # exp name -> (source PrototypeSet, mapping key, relabeled set):
        # the model's prototypes under the dataset's class names, kept so
        # the same object comes back while neither side changes.
        self._relabeled: dict = {}
        self._listeners: list = []

    # ---- subscription --------------------------------------------------

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Called after anything that changes what a view would draw: a
        sidecar attached, a source selected, an estimate computed."""
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in self._listeners:
            callback()

    # ---- the active source ---------------------------------------------

    def resolve(self, name: str):
        """('model', None) | ('kmeans', k) | (None, None)."""
        chosen = self._active.get(name)
        if chosen is not None:
            return chosen
        exp = self.state.experiments.get(name)
        if exp is not None and exp.model_prototypes is not None:
            return "model", None
        ks = sorted((k for (n, k) in self._kmeans if n == name), reverse=True)
        if ks:
            return "kmeans", ks[0]
        return None, None

    def status_text(self, name: str) -> str:
        source, k = self.resolve(name)
        if source == "model":
            return "Active source: model prototypes (imported)"
        if source == "kmeans":
            return f"Active source: k-means (K={k}, estimated on the test set -- indicative)"
        return "Active source: none"

    def active_set(self, name: str):
        """(source_key, PrototypeSet, color, note) for the active source, or
        None when neither a real nor an estimated prototype exists yet."""
        source, k = self.resolve(name)
        if source == "model":
            exp = self.state.experiments.get(name)
            if exp is not None and exp.model_prototypes is not None:
                return "model", exp.model_prototypes, MODEL_COLOR, "model prototypes (imported)"
            return None
        if source == "kmeans":
            proto = self._kmeans.get((name, k))
            if proto is not None:
                return "kmeans", proto, KMEANS_COLOR, f"estimated with k-means, K={k} — test set, indicative"
        return None

    def model_set_for_dataset(self, name: str):
        """The model's imported prototypes with each network class renamed
        to the dataset class the Data tab's class mapping assigns it, or
        None when none are loaded. The sidecar's labels are the network's
        own class names, i.e. the score columns; matched to the dataset by
        spelling alone, a relabelled class or a network trained under other
        names would get no prototype at all."""
        exp = self.state.experiments.get(name)
        if exp is None or exp.model_prototypes is None:
            return None
        label_of = dict(zip(exp.score_cols, exp.class_labels))
        key = tuple(sorted(label_of.items()))
        cached = self._relabeled.get(name)
        if cached is not None and cached[0] is exp.model_prototypes and cached[1] == key:
            return cached[2]
        relabeled = ea.relabel_prototypes(exp.model_prototypes, label_of)
        self._relabeled[name] = (exp.model_prototypes, key, relabeled)
        return relabeled

    def dataset_set(self, name: str, embeddings, labels):
        """(PrototypeSet, note) built from the dataset's own labels: the
        k-means estimate when one is the active source, otherwise one class
        mean per class. Unlike the model's set it always covers every class
        the dataset has, which is the point when the two disagree."""
        source, k = self.resolve(name)
        if source == "kmeans" and (name, k) in self._kmeans:
            return self._kmeans[(name, k)], f"k-means on the dataset labels (K={k}), indicative"
        class_order = sorted(set(labels.tolist()))
        return ea.class_mean_prototypes(embeddings, labels, class_order), "dataset class means, indicative"

    def assignment(self, ctx: dict, name: str, source_key: str, proto_set):
        """(best_class, best_k) per row, computed once per (experiment,
        source). `proto_set` is compared by identity, not equality: a fresh
        PrototypeSet object means the source was reloaded or re-estimated,
        so the cached assignment no longer describes it."""
        cache_key = (name, source_key)
        cached = self._assignments.get(cache_key)
        if cached is None or cached[2] is not proto_set:
            t0 = time.monotonic()
            best_class, best_k, _ = ea.assign_nearest_prototype(ctx["embeddings"], proto_set)
            _log(f"prototype assignment for '{name}'/{source_key} computed in {time.monotonic() - t0:.1f}s")
            self._assignments[cache_key] = (best_class, best_k, proto_set)
            return best_class, best_k
        return cached[0], cached[1]

    # ---- mutations ------------------------------------------------------

    def load_embeddings_for(self, name: str, path: str) -> tuple[bool, str]:
        """Attach an embeddings .npz sidecar, dropping whatever was derived
        from the previous embeddings. Called from the Data tab's per-row
        action and from the MLflow panel's sidecar auto-attach, so both
        paths share the exact same invalidation. Returns (success, message)
        -- message is either the loaded count or the error text."""
        exp = self.state.experiments.get(name)
        if exp is None:
            return False, "experiment not loaded"
        try:
            n = exp.load_embeddings(path)
        except Exception as ex:
            return False, str(ex)
        # A kmeans estimate is a function of the embeddings it was computed
        # from, so new embeddings invalidate it outright rather than
        # silently keeping a stale one selected.
        for key in [k for k in self._kmeans if k[0] == name]:
            del self._kmeans[key]
        if self._active.get(name, (None, None))[0] == "kmeans":
            self._active.pop(name, None)
        self.invalidate_assignments(name)
        self._notify()
        return True, f"{n:,} embeddings"

    def load_prototypes_for(self, name: str, path: str) -> tuple[bool, str]:
        """Attach a model-prototypes .npz sidecar and make it the active
        source, same sharing rationale as load_embeddings_for()."""
        exp = self.state.experiments.get(name)
        if exp is None:
            return False, "experiment not loaded"
        try:
            proto = exp.load_model_prototypes(path)
        except Exception as ex:
            return False, str(ex)
        self._active[name] = ("model", None)
        self.invalidate_assignments(name)
        self._notify()
        return True, f"{len(proto.class_labels):,}-class prototypes"

    def compute_kmeans(self, name: str, k: int) -> tuple[bool, str]:
        """Estimate K prototypes per class from the loaded pool and make
        them the active source."""
        ctx = build_analysis_context(self.state, name)
        if ctx is None:
            return False, "no embeddings loaded for this experiment"
        class_order = sorted(set(ctx["labels"].tolist()))
        t0 = time.monotonic()
        self._kmeans[(name, k)] = ea.compute_kmeans_prototypes(ctx["embeddings"], ctx["labels"], class_order, k)
        _log(f"kmeans estimate for '{name}' K={k} computed in {time.monotonic() - t0:.1f}s")
        self._active[name] = ("kmeans", k)
        self.invalidate_assignments(name)
        self._notify()
        return True, f"K={k}"

    def invalidate_assignments(self, name: str) -> None:
        for key in [k for k in self._assignments if k[0] == name]:
            del self._assignments[key]

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """Follow an experiment rename. Every entry here is keyed by
        experiment name, and two of these caches hold minutes of kmeans /
        nearest-prototype work -- re-keying them is what keeps renaming an
        experiment from throwing that away."""
        if old_name in self._active:
            self._active[new_name] = self._active.pop(old_name)
        for store in (self._kmeans, self._assignments):
            for key in [k for k in store if k[0] == old_name]:
                store[(new_name, key[1])] = store.pop(key)
        if old_name in self._relabeled:
            self._relabeled[new_name] = self._relabeled.pop(old_name)
