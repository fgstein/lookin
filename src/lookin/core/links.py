"""Addresses of things in `lookin web`, for a person to open.

    /?workspace=lkw-...                      a workspace, in a tab of its own
    /?snapshot=lk-...                        a saved view, restored
    ...&tab=metrics                          then this tab
    ...&crop=<uuid>[&experiment=<name>]      then this crop, in the viewer,
                                             with that experiment's prediction
    /?view=datasets&dataset=<path>           the Datasets page, on one dataset
    ...&compare=<path>                       compared with another
    /?mlflow_run=<run id>                    Home, with "New workspace from this
    ...&mlflow_server=<tracking uri>         run" open: what TrainHub's "Lookin"
    ...&artifact=<csv path>                  button opens (see core/run_links.py)

Built here rather than in the web package because the MCP endpoint hands
them out too: an agent's "42 cats taken for dogs" is worth more with the
crops one click away. The parameter names are a format -- links get pasted
into reports and tickets -- so they are never renamed.
"""
from __future__ import annotations

from urllib.parse import urlencode

WORKSPACE = "workspace"
SNAPSHOT = "snapshot"
TAB = "tab"
CROP = "crop"
EXPERIMENT = "experiment"
VIEW = "view"
VIEW_DATASETS = "datasets"
DATASET = "dataset"
COMPARE = "compare"
MLFLOW_RUN = "mlflow_run"
MLFLOW_SERVER = "mlflow_server"
ARTIFACT = "artifact"

# The server's own address, as the person reading a link reaches it. Set
# once at startup (`lookin web --public-url`, or the address it binds);
# empty, links are relative ("/?workspace=...").
_base_url = ""


def set_base_url(url: str) -> None:
    global _base_url
    _base_url = (url or "").rstrip("/")


def base_url() -> str:
    return _base_url


def link(workspace_id: str = "", snapshot_id: str = "", tab: str = "", crop: str = "",
         experiment: str = "") -> str:
    """The address that opens a workspace (or a snapshot of it), optionally
    on one tab and with one crop open. A snapshot carries its workspace, so
    `workspace_id` is left out when `snapshot_id` is given."""
    params = [(SNAPSHOT, snapshot_id)] if snapshot_id else [(WORKSPACE, workspace_id)]
    if tab:
        params.append((TAB, tab))
    if crop:
        params.append((CROP, str(crop)))
        if experiment:
            params.append((EXPERIMENT, experiment))
    return f"{_base_url}/?{urlencode(params)}"


def dataset_link(dataset: str = "", compare: str = "") -> str:
    """The address of the Datasets page, on one dataset (its absolute path),
    and compared with a second one when `compare` is given."""
    params = [(VIEW, VIEW_DATASETS)]
    if dataset:
        params.append((DATASET, dataset))
        if compare:
            params.append((COMPARE, compare))
    return f"{_base_url}/?{urlencode(params)}"


def run_link(run_id: str, server: str = "", artifact: str = "") -> str:
    """The address that offers to put one MLflow run in a workspace: a new
    one, or one of the library's. `server` is a hint (the tracking URI as
    the sender knows it), `artifact` the results CSV to preselect."""
    params = [(MLFLOW_RUN, run_id)]
    if server:
        params.append((MLFLOW_SERVER, server))
    if artifact:
        params.append((ARTIFACT, artifact))
    return f"{_base_url}/?{urlencode(params)}"


def relative(url: str) -> str:
    """A link of this module without the server's address ("/?..."): what
    another tool gets, to put behind the address *it* reaches Lookin at."""
    return url[len(_base_url):] if _base_url and url.startswith(_base_url) else url
