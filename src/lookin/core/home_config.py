"""What the Home page shows: one JSON file per user, meant to be edited.

    ~/.config/lookin/home_view.json

The Home page writes it whenever the user shows, hides or filters something,
and reads it back whenever it changes on disk -- so editing it by hand (or
from the in-app editor) and clicking in the UI are two ways of changing the
same thing. Per user, never per workspace: which runs someone wants to see
is theirs, and a shared file would have colleagues hiding each other's.

The shape, with every key optional::

    {
      "servers": [{"name": "lab", "uri": "http://mlflow.lab:5000"}],
      "dataset_param": "datamodule/configs/database/csv_path",
      "experiments": {
        "http://mlflow.lab:5000": {"shown": ["cls-v3"], "hidden": ["Default"]}
      },
      "runs": {
        "hidden": [{"uri": "...", "run_id": "...", "name": "..."}],
        "show_unsuccessful": false,
        "search": "",
        "filter": ""
      },
      "columns": [{"key": "params.backbone", "label": "backbone"},
                  {"key": "metrics.val_acc"}],
      "parallel_requests": 4
    }

Experiments are named, not numbered: a name is what someone editing this
file by hand can read, and MLflow keeps names unique per server. Hidden
runs carry their name for the same reason, although only the id is used.

Same rules as core/prefs.py: stdlib only, atomic writes, and a file that
cannot be read is reported and replaced by defaults rather than raised --
with one difference: a file that does not *parse* is never overwritten
behind the user's back, since that would silently throw away their edit.
"""
from __future__ import annotations

import functools
import json
import threading
from pathlib import Path
from typing import Optional

from lookin.core import fileio
from lookin.core.prefs import config_dir
from lookin.core.workspace_sources import DEFAULT_DATASET_PARAM, normalize_uri

HOME_VIEW_FILE = "home_view.json"
VERSION = 1

# Column keys are MLflow's own namespaces, the same ones its search syntax
# uses -- so a column key can be pasted into the filter box as it is.
PARAM_PREFIX = "params."
METRIC_PREFIX = "metrics."
# The first time a server is seen, this many of its most recently updated
# experiments are shown. Showing none is an empty table that looks broken;
# showing all can be thousands of runs nobody asked for.
DEFAULT_SHOWN_EXPERIMENTS = 3
# How many runs/search requests go to one server at once. MLflow's server
# spends seconds building a page of runs with many params, and answers
# several at once only up to its own worker count (4 by default) -- past it,
# more requests just queue and slow each other down.
DEFAULT_PARALLEL_REQUESTS = 4
MAX_PARALLEL_REQUESTS = 16
MLFLOW_DEFAULT_EXPERIMENT = "Default"


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def home_view_path() -> Path:
    return config_dir() / HOME_VIEW_FILE


def default_config(tracking_uri: str = "") -> dict:
    servers = [{"name": _server_name(tracking_uri), "uri": normalize_uri(tracking_uri)}] if tracking_uri else []
    return {
        "version": VERSION,
        "servers": servers,
        "dataset_param": DEFAULT_DATASET_PARAM,
        "experiments": {},
        "runs": {"hidden": [], "show_unsuccessful": False, "search": "", "filter": ""},
        "columns": [],
        "parallel_requests": DEFAULT_PARALLEL_REQUESTS,
    }


def _server_name(uri: str) -> str:
    """A short label for a server: its host, without the scheme and port."""
    host = normalize_uri(uri).split("://", 1)[-1].split("/", 1)[0]
    return host.split(":", 1)[0] or uri


def _str_list(value) -> list:
    return [str(v) for v in value if isinstance(v, (str, int, float))] if isinstance(value, list) else []


def normalize(raw) -> dict:
    """Coerce whatever was read into the full shape, dropping what cannot be
    understood. Tolerant on purpose: a hand-edited file with one wrong type
    in it should cost that one setting, not all of them."""
    cfg = default_config()
    if not isinstance(raw, dict):
        return cfg
    servers = []
    for s in raw.get("servers") or []:
        if isinstance(s, dict) and str(s.get("uri", "")).strip():
            uri = normalize_uri(str(s["uri"]))
            if uri not in {x["uri"] for x in servers}:
                servers.append({"name": str(s.get("name") or _server_name(uri)), "uri": uri})
    cfg["servers"] = servers
    if isinstance(raw.get("dataset_param"), str):
        cfg["dataset_param"] = raw["dataset_param"].strip()
    experiments = raw.get("experiments")
    if isinstance(experiments, dict):
        for uri, entry in experiments.items():
            if isinstance(entry, dict):
                cfg["experiments"][normalize_uri(str(uri))] = {
                    "shown": _str_list(entry.get("shown")),
                    "hidden": _str_list(entry.get("hidden")),
                }
    runs = raw.get("runs")
    if isinstance(runs, dict):
        hidden = []
        for h in runs.get("hidden") or []:
            if isinstance(h, dict) and h.get("run_id"):
                hidden.append({"uri": normalize_uri(str(h.get("uri", ""))), "run_id": str(h["run_id"]),
                               "name": str(h.get("name", ""))})
        cfg["runs"]["hidden"] = hidden
        cfg["runs"]["show_unsuccessful"] = bool(runs.get("show_unsuccessful", False))
        for key in ("search", "filter"):
            if isinstance(runs.get(key), str):
                cfg["runs"][key] = runs[key]
    columns = []
    for c in raw.get("columns") or []:
        if isinstance(c, str):
            c = {"key": c}
        if isinstance(c, dict) and str(c.get("key", "")).startswith((PARAM_PREFIX, METRIC_PREFIX)):
            entry = {"key": str(c["key"])}
            if c.get("label"):
                entry["label"] = str(c["label"])
            columns.append(entry)
    cfg["columns"] = columns
    try:
        cfg["parallel_requests"] = max(1, min(MAX_PARALLEL_REQUESTS, int(raw.get("parallel_requests",
                                                                             DEFAULT_PARALLEL_REQUESTS))))
    except (TypeError, ValueError):
        pass
    return cfg


def _mutator(method):
    """Every change starts from the file as it is now, under one lock.

    Each browser tab holds its own HomeConfig on the same file. The Home
    page notices edits within a couple of seconds (see HomeView._watch_loop),
    but a change made inside that window would otherwise save this tab's
    older copy over what another tab -- or a hand edit -- just wrote.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # The lock file as well: `lookin ui` and `lookin web` run by one
        # person share this file from two processes.
        with _FILE_LOCK, fileio.locked(self.path):
            if self.changed_on_disk():
                self.reload()
            return method(self, *args, **kwargs)
    return wrapper


# One for the process: every tab's HomeConfig writes the same file.
_FILE_LOCK = threading.RLock()


class HomeConfig:
    """The file, held in memory, with change detection.

    Every mutator saves immediately: the file is the source of truth, and a
    setting that only lived in memory until some later save would be lost
    to the next hand edit of the file.
    """

    def __init__(self, path: Optional[Path] = None, fallback_uri: str = ""):
        self.path = Path(path) if path is not None else home_view_path()
        self.fallback_uri = fallback_uri
        self.data = default_config(fallback_uri)
        self.error = ""       # why the file on disk could not be used, if it could not
        self._mtime = None
        self.reload()

    # ---- file -----------------------------------------------------------

    def _disk_mtime(self):
        """What identifies one write of the file: (inode, mtime). The inode
        because every save is a replace, which moves it even when two
        saves land within one mtime tick -- and then a tab that compared
        mtimes alone saved its stale copy over the other's change."""
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_ino, stat.st_mtime_ns)

    def changed_on_disk(self) -> bool:
        return self._disk_mtime() != self._mtime

    def reload(self) -> None:
        self._mtime = self._disk_mtime()
        if self._mtime is None:
            self.data, self.error = default_config(self.fallback_uri), ""
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            # Kept in memory as it was, and the file left alone: this is
            # almost always a hand edit with a missing comma, and rewriting
            # the file with defaults would throw that edit away.
            self.error = f"{self.path.name} could not be read ({ex}); using the previous settings."
            _log(f"home view: {self.error}")
            return
        self.data, self.error = normalize(raw), ""
        # Only when the key is absent: an empty list is someone having
        # removed every server, and putting one back would undo that.
        if isinstance(raw, dict) and "servers" not in raw and self.fallback_uri:
            self.data["servers"] = default_config(self.fallback_uri)["servers"]

    def save(self) -> bool:
        if self.error:
            # Never overwrite a file the user is halfway through editing.
            return False
        try:
            fileio.atomic_write_text(self.path, json.dumps(self.data, indent=2))
        except OSError as ex:
            _log(f"home view: could not write {self.path} ({ex})")
            return False
        self._mtime = self._disk_mtime()
        return True

    def text(self) -> str:
        return json.dumps(self.data, indent=2)

    def replace_text(self, text: str) -> str:
        """Replace the whole config from the in-app editor. Returns "" on
        success, else what is wrong -- and then nothing is written."""
        try:
            raw = json.loads(text)
        except ValueError as ex:
            return f"Not valid JSON: {ex}"
        if not isinstance(raw, dict):
            return "The file must hold a JSON object."
        self.data, self.error = normalize(raw), ""
        return "" if self.save() else f"Could not write {self.path}."

    # ---- servers --------------------------------------------------------

    @property
    def servers(self) -> list:
        return self.data["servers"]

    @_mutator
    def add_server(self, uri: str, name: str = "") -> bool:
        uri = normalize_uri(uri)
        if not uri or uri in {s["uri"] for s in self.servers}:
            return False
        self.servers.append({"name": name.strip() or _server_name(uri), "uri": uri})
        return self.save()

    @_mutator
    def remove_server(self, uri: str) -> None:
        uri = normalize_uri(uri)
        self.data["servers"] = [s for s in self.servers if s["uri"] != uri]
        self.data["experiments"].pop(uri, None)
        self.save()

    # ---- experiments ----------------------------------------------------

    def _exp_entry(self, uri: str) -> dict:
        return self.data["experiments"].setdefault(normalize_uri(uri), {"shown": [], "hidden": []})

    def knows_server(self, uri: str) -> bool:
        return normalize_uri(uri) in self.data["experiments"]

    # Reads never create the server's entry: its absence is what tells
    # adopt_defaults the server has never been listed.
    def shown_experiments(self, uri: str) -> list:
        return list(self.data["experiments"].get(normalize_uri(uri), {}).get("shown", []))

    def hidden_experiments(self, uri: str) -> list:
        return list(self.data["experiments"].get(normalize_uri(uri), {}).get("hidden", []))

    @_mutator
    def adopt_defaults(self, uri: str, experiments: list) -> None:
        """The first time a server is listed, show its few most recently
        updated experiments (see DEFAULT_SHOWN_EXPERIMENTS)."""
        if self.knows_server(uri):
            return
        # MLflow's own "Default" experiment is where runs land that nobody
        # filed, and it exists on every server, empty or not. A server that
        # has nothing else yet is not adopted at all -- adopting it would
        # lock in "Default" and hide every experiment created afterwards.
        candidates = [e for e in experiments if e.name != MLFLOW_DEFAULT_EXPERIMENT]
        if not candidates:
            return
        recent = sorted(candidates, key=lambda e: e.last_update_time or 0, reverse=True)
        self._exp_entry(uri)["shown"] = [e.name for e in recent[:DEFAULT_SHOWN_EXPERIMENTS]]
        self.save()

    @_mutator
    def toggle_experiment(self, uri: str, name: str) -> None:
        entry = self._exp_entry(uri)
        if name in entry["shown"]:
            entry["shown"].remove(name)
        else:
            entry["shown"].append(name)
        self.save()

    @_mutator
    def set_experiments(self, uri: str, shown, hidden) -> None:
        """Replace which experiments are shown and hidden, in one write --
        what the experiment picker applies. Hidden wins over shown."""
        entry = self._exp_entry(uri)
        entry["hidden"] = sorted(set(hidden))
        entry["shown"] = sorted(set(shown) - set(hidden))
        self.save()

    @_mutator
    def set_experiment_hidden(self, uri: str, name: str, hidden: bool) -> None:
        entry = self._exp_entry(uri)
        entry["hidden"] = [n for n in entry["hidden"] if n != name]
        if hidden:
            entry["hidden"].append(name)
            entry["shown"] = [n for n in entry["shown"] if n != name]
        self.save()

    # ---- runs -----------------------------------------------------------

    @property
    def runs(self) -> dict:
        return self.data["runs"]

    def hidden_run_ids(self, uri: str) -> set:
        uri = normalize_uri(uri)
        return {h["run_id"] for h in self.runs["hidden"] if h["uri"] in ("", uri)}

    @_mutator
    def set_run_hidden(self, uri: str, run_id: str, name: str, hidden: bool) -> None:
        uri = normalize_uri(uri)
        self.runs["hidden"] = [h for h in self.runs["hidden"]
                               if not (h["run_id"] == run_id and h["uri"] in ("", uri))]
        if hidden:
            self.runs["hidden"].append({"uri": uri, "run_id": run_id, "name": name})
        self.save()

    @_mutator
    def unhide_all_runs(self) -> None:
        self.runs["hidden"] = []
        self.save()

    @_mutator
    def set_run_option(self, key: str, value) -> None:
        if self.runs.get(key) != value:
            self.runs[key] = value
            self.save()

    # ---- columns and dataset --------------------------------------------

    @property
    def columns(self) -> list:
        return self.data["columns"]

    @_mutator
    def set_columns(self, keys: list) -> None:
        """Keep the label of a column that stays, so choosing columns in
        the UI never throws away a label typed into the file."""
        labels = {c["key"]: c.get("label") for c in self.columns}
        self.data["columns"] = [{"key": k, **({"label": labels[k]} if labels.get(k) else {})} for k in keys]
        self.save()

    @property
    def parallel_requests(self) -> int:
        return int(self.data.get("parallel_requests", DEFAULT_PARALLEL_REQUESTS))

    @property
    def dataset_param(self) -> str:
        return self.data.get("dataset_param", DEFAULT_DATASET_PARAM)


def column_label(column: dict) -> str:
    """The header for a column: its label, else its key without the
    namespace, else the key."""
    if column.get("label"):
        return column["label"]
    key = column["key"]
    for prefix in (PARAM_PREFIX, METRIC_PREFIX):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def column_value(run, column: dict) -> str:
    key = column["key"]
    if key.startswith(PARAM_PREFIX):
        return str(run.params.get(key[len(PARAM_PREFIX):], ""))
    if key.startswith(METRIC_PREFIX):
        value = run.metrics.get(key[len(METRIC_PREFIX):])
        return "" if value is None else format_metric(value)
    return ""


def format_metric(value: float) -> str:
    if value != value:
        return "NaN"
    if value in (float("inf"), float("-inf")):
        return "+inf" if value > 0 else "-inf"
    magnitude = abs(value)
    if value == int(value) and magnitude < 1e9:
        return str(int(value))
    if magnitude >= 1e5 or 0 < magnitude < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4g}"

