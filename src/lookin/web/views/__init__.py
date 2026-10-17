"""The views of the web UI, by name (the name is also the action target)."""
from __future__ import annotations

import importlib

from lookin.web import html as h
from lookin.web.views.base import View

# name -> "module:Class". Imported on first use, so a view's heavy imports
# (sklearn, umap) are only paid by the session that opens it.
_SPECS = {
    "home": "lookin.web.views.home:HomeView",
    "sources": "lookin.web.views.sources:SourcesView",
    "statistics": "lookin.web.views.stats:StatsView",
    "gallery": "lookin.web.views.gallery:GalleryView",
    "metrics": "lookin.web.views.metrics:MetricsView",
    "calibration": "lookin.web.views.calibration:CalibrationView",
    "confidence": "lookin.web.views.confidence:ConfidenceView",
    "compare": "lookin.web.views.compare:CompareView",
    "embeddings": "lookin.web.views.embeddings:EmbeddingsView",
    "prototypes": "lookin.web.views.prototypes:PrototypesView",
    "cleaning": "lookin.web.views.cleaning:CleaningView",
}


class _Placeholder(View):
    """Stands in for a view whose module is not there yet."""

    def render(self) -> str:
        return h.banner(f"The {self.name} tab is not available yet in this build.", "info")

    def view_state(self) -> dict:
        return {}

    def restore_view_state(self, state) -> list:
        return []


class _Registry(dict):
    def __missing__(self, name):
        spec = _SPECS[name]
        module_name, cls_name = spec.split(":")
        try:
            cls = getattr(importlib.import_module(module_name), cls_name)
        except ModuleNotFoundError as ex:
            if ex.name != module_name:
                raise
            cls = type(cls_name, (_Placeholder,), {"name": name})
        self[name] = cls
        return cls

    def __contains__(self, name):
        return name in _SPECS


VIEW_CLASSES = _Registry()
