"""Which workspaces use an MLflow run, and finding a run from its id alone.

The MLflow run id is the whole link between Lookin and other tools that
show MLflow runs -- TrainHub above all, which launches the trainings and the
inferences whose results Lookin explores. Nothing is tagged on either side:
a run is the same run in both because its id is, whoever started it.

* `workspaces_by_run` is what `GET /api/links/mlflow-runs` answers (see
  web/server.py): for each run id, the workspaces it is an experiment of,
  with the address that opens each one. TrainHub draws them as badges next
  to its runs.
* `resolve_run` finds a run on the configured MLflow servers from its id and
  an optional hint of which server it is on -- what `?mlflow_run=` (see
  core/links.py) needs, since the sender's address of a server need not be
  Lookin's (a hostname here, an IP there).
"""
from __future__ import annotations

from typing import Iterable, Optional

from lookin.core import links
from lookin.core import mlflow_client as mlf
from lookin.core import workspace_sources as wss

# The answer's "app" field: a caller checking it is talking to Lookin.
APP = "lookin"


def workspaces_by_run(workspaces: Iterable, groups: Optional[dict] = None,
                      run_ids: Optional[Iterable[str]] = None) -> dict:
    """{run id: [one entry per experiment of a workspace made from that run]}.

    `groups` maps a workspace id to its Home group's name. `run_ids`, when
    given, keeps only those runs. Every entry carries `path` (the link
    without this server's address) next to `url`: the caller may reach
    Lookin under another name than the one it binds.
    """
    wanted = {r for r in run_ids if r} if run_ids is not None else None
    groups = groups or {}
    out: dict = {}
    for w in workspaces:
        for e in w.mount.experiments:
            if not e.mlflow_run_id or (wanted is not None and e.mlflow_run_id not in wanted):
                continue
            url = links.link(w.id)
            out.setdefault(e.mlflow_run_id, []).append({
                "workspace_id": w.id,
                "workspace": w.label(),
                "group": groups.get(w.id) or None,
                "experiment": e.name,
                "artifact": e.artifact_path or None,
                "server": e.mlflow_uri or None,
                "path": links.relative(url),
                "url": url,
            })
    for entries in out.values():
        entries.sort(key=lambda x: (x["workspace"].lower(), x["experiment"].lower()))
    return out


def index_payload(run_ids: Optional[Iterable[str]] = None) -> dict:
    """The whole answer of `GET /api/links/mlflow-runs`, read from the
    library as it is on disk now. The shape is a format other tools read:
    keys are added, never renamed."""
    from lookin import __version__
    from lookin.core import board as bd
    from lookin.core import library as lib

    workspaces = lib.list_workspaces()
    groups = {w.id: column.get("name", "") for column, items in bd.layout(bd.load(), workspaces) for w in items}
    return {
        "app": APP,
        "version": __version__,
        "base_url": links.base_url() or None,
        "runs": workspaces_by_run(workspaces, groups, run_ids),
    }


def configured_servers() -> list:
    from lookin.core.home_config import HomeConfig

    return [wss.normalize_uri(s["uri"]) for s in HomeConfig(fallback_uri=mlf.default_tracking_uri()).servers]


class RunNotFound(ValueError):
    """No configured server knows the run. `unknown_server` is the hint when
    it is a server Lookin is not configured for -- adding it may be the fix."""

    def __init__(self, message: str, unknown_server: str = ""):
        super().__init__(message)
        self.unknown_server = unknown_server


def resolve_run(run_id: str, server: str = "", servers: Optional[list] = None) -> tuple:
    """``(tracking uri, mlf.Run)`` for a run id.

    Only servers configured in Lookin are asked (an address that arrived in
    a link is never dialled on its own): the hinted one first when it is one
    of them, then the others -- a run id is unique, so the first that knows
    it has it."""
    run_id = (run_id or "").strip()
    if not run_id:
        raise ValueError("No MLflow run id.")
    configured = configured_servers() if servers is None else [wss.normalize_uri(s) for s in servers]
    hint = wss.normalize_uri(server)
    unknown = hint if hint and hint not in configured else ""
    if not configured:
        raise RunNotFound("No MLflow server is configured in Lookin (add one on the Home page).", unknown)
    order = ([hint] if hint in configured else []) + [u for u in configured if u != hint]
    errors = []
    for uri in order:
        try:
            return uri, mlf.get_run(uri, run_id)
        except mlf.MlflowError as ex:
            errors.append(f"{uri}: {ex}")
    message = f"No MLflow server configured in Lookin knows run {run_id}"
    if unknown:
        message += f" -- it was sent as being on {server}, which is not one of them"
    raise RunNotFound(message + ". " + "; ".join(errors), unknown)
