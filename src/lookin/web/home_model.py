"""What Home knows about the MLflow servers, and how a run or one of its
CSVs is named while it is being selected and dragged.

Shared by Home (views/home.py), the runs dialog (views/mlflow_dialogs.py)
and the process-wide hub that keeps the listings fresh (mlflow_hub.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss

KEY_SEP = "\t"   # never in a URI, a run id or an artifact path

# Workspace badge colours (Material 200 tones). Chosen by the workspace id,
# so a workspace keeps its colour whatever else is created or deleted.
WORKSPACE_COLORS = (
    "#CE93D8", "#80CBC4", "#FFCC80", "#F48FB1", "#81D4FA", "#E6EE9C", "#FFE082", "#80DEEA",
)


def workspace_color(workspace_id: str) -> str:
    return WORKSPACE_COLORS[sum(map(ord, workspace_id or "")) % len(WORKSPACE_COLORS)]


def run_key(uri: str, run_id: str) -> str:
    return KEY_SEP.join(("run", wss.normalize_uri(uri), run_id))


def csv_key(uri: str, run_id: str, path: str) -> str:
    return KEY_SEP.join(("csv", wss.normalize_uri(uri), run_id, path))


def parse_key(key: str) -> tuple:
    """("run", uri, run_id, "") or ("csv", uri, run_id, path)."""
    parts = (key or "").split(KEY_SEP)
    if len(parts) == 3 and parts[0] == "run":
        return "run", parts[1], parts[2], ""
    if len(parts) == 4 and parts[0] == "csv":
        return "csv", parts[1], parts[2], parts[3]
    return "", "", "", ""


# ---- selecting a run, or some of its CSVs ---------------------------------
#
# A run's box stands for all its CSVs. Before they are listed, ticking it
# selects the run key (adding it asks which CSV when it has several, see
# views/home.HomeView.add_keys); once they are listed, it ticks each of
# them, and the run's box shows "half ticked" while only some are. Shared by Home and the
# runs dialog (views/mlflow_dialogs.RunsDialog), whose servers are the same
# ServerState.

SELECTED_ALL, SELECTED_SOME, SELECTED_NONE = "all", "some", "none"


def run_csv_keys(server, run_id: str):
    """The keys of a run's CSVs, or None while they are not listed."""
    run = server.run_by_id(run_id) if server is not None else None
    entry = server.artifacts.get(run_id) if server is not None else None
    if run is None or not entry or "csvs" not in entry:
        return None
    candidates = wss.candidates_for_run(server.uri, run, entry["csvs"], entry.get("npzs", []))
    return [csv_key(server.uri, run_id, c.csv.path) for c in candidates]


def run_selection(selected: set, server, run_id: str) -> str:
    """How much of this run is selected: SELECTED_ALL, _SOME or _NONE."""
    if run_key(server.uri, run_id) in selected:
        return SELECTED_ALL
    keys = run_csv_keys(server, run_id) or []
    n = sum(1 for k in keys if k in selected)
    return SELECTED_ALL if keys and n == len(keys) else (SELECTED_SOME if n else SELECTED_NONE)


def csv_selected(selected: set, key: str) -> bool:
    _, uri, run_id, _ = parse_key(key)
    return key in selected or run_key(uri, run_id) in selected


def set_selected(selected: set, server, key: str, value: bool) -> None:
    """Tick (value=True) or untick a run's box or one of its CSVs', in place.
    A run box that is half ticked ticks every CSV."""
    kind, uri, run_id, _ = parse_key(key)
    whole = run_key(uri, run_id)
    csvs = set(run_csv_keys(server, run_id) or [])
    if kind == "run":
        selected -= csvs | {whole}
        if value:
            selected |= csvs or {whole}
    elif value:
        selected.add(key)
    else:
        if whole in selected:
            # The whole run was ticked: untick this one CSV of it.
            selected.discard(whole)
            selected |= csvs
        selected.discard(key)


def unselect_run(selected: set, server, run_id: str) -> None:
    selected -= {run_key(server.uri, run_id), *(run_csv_keys(server, run_id) or [])}


def _dc_from_dict(cls, raw: dict):
    known = cls.__dataclass_fields__
    return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ServerState:
    """What is known about one tracking server right now.

    Runs are kept per experiment (`runs_by_exp`), because that is the unit
    they are fetched in: showing one more experiment fetches that one, and
    hiding one fetches nothing. `runs` is the flattened view the table
    draws, grouped by experiment -- see rebuild_runs.
    """
    uri: str
    name: str
    status: str = "idle"              # idle | loading | ok | down
    error: str = ""
    experiments: list = field(default_factory=list)   # [mlf.Experiment]
    runs_by_exp: dict = field(default_factory=dict)   # experiment id -> [mlf.Run]
    runs: list = field(default_factory=list)          # [mlf.Run], grouped by experiment
    truncated_exps: set = field(default_factory=set)
    listed_at: str = ""               # when the runs were last fetched
    offline: bool = False             # showing a cached listing because the server is down
    # run_id -> {"csvs": [...], "npzs": [...]} | {"error": str} | {"loading": True}
    artifacts: dict = field(default_factory=dict)
    last_try: float = 0.0

    @property
    def truncated(self) -> bool:
        return bool(self.truncated_exps)

    def experiment_by_id(self, experiment_id: str):
        return next((e for e in self.experiments if e.experiment_id == experiment_id), None)

    def run_by_id(self, run_id: str):
        return next((r for r in self.runs if r.run_id == run_id), None)

    def rebuild_runs(self) -> None:
        """Flatten runs_by_exp, one experiment after the other (by name).

        Grouped explicitly: MLflow returns runs sorted by start time across
        experiments, so drawing them in that order interleaves experiments
        and repeats the same experiment header several times down the table.
        """
        names = {e.experiment_id: e.name.lower() for e in self.experiments}
        order = sorted(self.runs_by_exp, key=lambda i: (names.get(i, "~"), i))
        self.runs = [run for experiment_id in order for run in self.runs_by_exp[experiment_id]]

    def keep_only(self, experiment_ids) -> None:
        keep = set(experiment_ids)
        for experiment_id in list(self.runs_by_exp):
            if experiment_id not in keep:
                del self.runs_by_exp[experiment_id]
        self.truncated_exps &= keep
        self.rebuild_runs()

    def listing_payload(self) -> dict:
        return {
            "listed_at": self.listed_at,
            "experiments": [asdict(e) for e in self.experiments],
            "runs": [asdict(r) for r in self.runs],
            "truncated_experiments": sorted(self.truncated_exps),
        }

    def artifacts_payload(self) -> dict:
        return {
            run_id: {"csvs": [asdict(a) for a in v["csvs"]], "npzs": [asdict(a) for a in v["npzs"]]}
            for run_id, v in self.artifacts.items() if "csvs" in v
        }

    def adopt_listing(self, payload: dict, artifacts: Optional[dict] = None) -> None:
        self.experiments = [_dc_from_dict(mlf.Experiment, e) for e in payload.get("experiments", [])
                            if isinstance(e, dict)]
        self.runs_by_exp = {}
        for raw in payload.get("runs", []):
            if isinstance(raw, dict):
                run = _dc_from_dict(mlf.Run, raw)
                self.runs_by_exp.setdefault(run.experiment_id, []).append(run)
        self.truncated_exps = set(payload.get("truncated_experiments") or [])
        self.listed_at = str(payload.get("listed_at", ""))
        self.rebuild_runs()
        # Listings written before the run-CSV lists moved to a file of their
        # own still carry them under "artifacts".
        # Stale: good to draw, not to add from -- see needs_listing.
        for run_id, v in {**(payload.get("artifacts") or {}), **(artifacts or {})}.items():
            if isinstance(v, dict) and run_id not in self.artifacts:
                self.artifacts[run_id] = {
                    "csvs": [_dc_from_dict(mlf.ArtifactFile, a) for a in v.get("csvs", [])],
                    "npzs": [_dc_from_dict(mlf.ArtifactFile, a) for a in v.get("npzs", [])],
                    "stale": True,
                }

    def mark_artifacts_stale(self) -> None:
        """What a refresh does to the run -> CSV lists: they stay drawn, and
        are listed again before anything is added from them."""
        for entry in self.artifacts.values():
            if "csvs" in entry:
                entry["stale"] = True

    def needs_listing(self, run_id: str) -> bool:
        """Whether `run_id`'s CSVs must be listed before a run is added or
        loaded from them. A list read from disk, or from before a refresh,
        is not enough while the server answers: the run may have logged its
        CSV since (it was still running), or re-logged it with another size
        -- and a workspace that recorded the old size never counts it as
        downloaded. With the server down, the old list is all there is."""
        entry = self.artifacts.get(run_id)
        if entry is None or entry.get("loading"):
            return True
        return bool(entry.get("stale")) and self.status != "down"
