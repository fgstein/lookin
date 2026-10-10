"""The MCP endpoint: Lookin's results, readable by an AI agent.

`lookin ui` serves it at /mcp, on the same host and port as the web
interface, so whatever can reach the UI can reach it -- an agent such as
OpenCode is pointed at http://<host>:<port>/mcp as a "remote" MCP server.

Everything an agent can ask is in core/agent_tools.py; this module only
describes those functions to the model and puts them on the HTTP route.
The descriptions are written for a model that has never seen Lookin: they
are the only documentation it gets.

Transport: MCP's Streamable HTTP, stateless and answering in plain JSON.
Every tool is a single request and a single answer, so there is no session
to keep and no event stream to hold open.

Host-header checking is off. The SDK turns it on for a server bound to
localhost and would then refuse every request that arrives through a port
forward or a gateway under another name -- which is exactly how this server
is reached. The endpoint is read-only and exposes nothing the UI on the same
port does not.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Annotated

from pydantic import Field

from lookin import __version__
from lookin.core import agent_tools

MCP_PATH = "/mcp"

INSTRUCTIONS = """\
Lookin is a visual explorer for image classification results. Its data is organised in workspaces:
one workspace = one labelled dataset plus the experiments (model runs, each a results CSV of class
scores) evaluated on it.

Typical use: call list_workspaces, then get_metrics on a workspace to compare its experiments, then
drill down with get_per_class, get_confusions, get_examples or compare_experiments.

Metrics are only comparable within one workspace: two workspaces usually have different datasets.
Every answer carries a "warnings" list -- report those to the user along with the numbers.
search_mlflow_runs looks up training runs on the MLflow servers Lookin is configured with.
"""

WorkspaceId = Annotated[str, Field(description="Workspace id (e.g. 'lkw-3kq9xd') or exact workspace name.")]
ExperimentName = Annotated[str, Field(description="Experiment name, as listed by get_workspace or get_metrics.")]
Top = Annotated[int, Field(description="Maximum number of rows to return.", ge=1, le=agent_tools.MAX_TOP)]


def _run(tool, *args) -> str:
    """Call a tool function and answer in compact JSON.

    Compact because, left to itself, the SDK indents a returned dict, which
    roughly doubles what a model with a small context has to read.

    The tool functions raise ValueError (and MlflowError) for every failure
    they expect -- an unknown workspace, an experiment that is not
    downloaded -- with a message meant to be read. The SDK only passes a
    message on to the model for a ToolError: anything else counts as a crash
    and the model sees "Error executing tool <name>" and nothing more, which
    leaves it unable to correct the call.
    """
    from mcp.server.mcpserver.exceptions import ToolError

    from lookin.core.mlflow_client import MlflowError

    try:
        result = tool(*args)
    except (ValueError, MlflowError) as ex:
        raise ToolError(str(ex)) from ex
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


def list_workspaces() -> str:
    """List every workspace in Lookin's library: id, name, tags, dataset, the names of its
    experiments, and whether their files are downloaded. Start here."""
    return _run(agent_tools.list_workspaces)


def get_workspace(workspace_id: WorkspaceId) -> str:
    """Describe one workspace without computing anything: its dataset (path, columns, split, row
    filter, review version), its experiments (with their MLflow run when they come from one, and
    whether they are downloaded), and its saved snapshots."""
    return _run(agent_tools.get_workspace, workspace_id)


def get_metrics(
    workspace_id: WorkspaceId,
    experiments: Annotated[list[str] | None, Field(
        description="Experiment names to include. Omit for every experiment of the workspace.")] = None,
) -> str:
    """Summary metrics of each experiment of a workspace, one row per experiment: accuracy,
    balanced accuracy, macro and weighted F1, MCC, Cohen's kappa, ROC AUC and average precision
    (macro and weighted), log loss, Brier score, ECE and MCE. Use it to find the best experiment.
    The first call on a workspace loads its CSVs and can take a while."""
    return _run(agent_tools.get_metrics, workspace_id, experiments)


def get_per_class(
    workspace_id: WorkspaceId,
    experiment: ExperimentName,
    sort_by: Annotated[str, Field(description="Column to sort by: one of " + ", ".join(
        agent_tools.PER_CLASS_COLUMNS) + ".")] = "f1",
    ascending: Annotated[bool, Field(
        description="True puts the lowest values first (the worst classes, when sorting by a score).")] = True,
    top: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """Per-class metrics of one experiment: precision, recall, specificity, F1, support, ROC AUC and
    average precision for each class. By default the classes with the lowest F1 come first."""
    return _run(agent_tools.get_per_class, workspace_id, experiment, sort_by, ascending, top)


def get_confusions(workspace_id: WorkspaceId, experiment: ExperimentName, top: Top = agent_tools.DEFAULT_TOP) -> str:
    """The most frequent mistakes of one experiment: (true label -> predicted label) pairs with
    their count and the share of the true class they represent, most frequent first. This is the
    confusion matrix without its diagonal, reduced to its largest cells."""
    return _run(agent_tools.get_confusions, workspace_id, experiment, top)


def compare_experiments(
    workspace_id: WorkspaceId,
    baseline: Annotated[str, Field(description="Name of the reference experiment.")],
    candidate: Annotated[str, Field(description="Name of the experiment compared against the baseline.")],
    top: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """Compare two experiments of the same workspace crop by crop: how many crops the candidate
    fixed (baseline wrong, candidate right) and broke (baseline right, candidate wrong), both
    accuracies on their common crops, and the classes that changed most."""
    return _run(agent_tools.compare_experiments, workspace_id, baseline, candidate, top)


def get_examples(
    workspace_id: WorkspaceId,
    experiment: ExperimentName,
    true_label: Annotated[str | None, Field(description="Only crops with this ground-truth label.")] = None,
    predicted_label: Annotated[str | None, Field(description="Only crops predicted as this label.")] = None,
    errors_only: Annotated[bool, Field(description="Only misclassified crops.")] = True,
    limit: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """Individual crops of one experiment, most confident first: uuid, label, prediction,
    confidence, crop path and a few metadata columns. By default only the errors -- a confident
    error is often a labelling mistake. Combine true_label and predicted_label to look inside one
    cell of the confusion matrix."""
    return _run(agent_tools.get_examples, workspace_id, experiment, true_label, predicted_label, errors_only, limit)


def search_mlflow_runs(
    filter: Annotated[str, Field(description=(
        "MLflow search filter, e.g. \"metrics.val_acc > 0.9 and params.backbone = 'vit'\". "
        "Empty for every run."))] = "",
    experiment_names: Annotated[list[str] | None, Field(description=(
        "MLflow experiment names to search. Omit for the experiments shown on Lookin's Home page."))] = None,
    server: Annotated[str | None, Field(description=(
        "Tracking URI of one MLflow server. Omit for every server configured in Lookin."))] = None,
    max_results: Annotated[int, Field(description="Maximum number of runs to return.", ge=1,
                                      le=agent_tools.MAX_RUNS)] = agent_tools.DEFAULT_MAX_RUNS,
) -> str:
    """Search training runs on the MLflow servers configured in Lookin, most recent first: run id
    and name, experiment, status, start time, params, metrics, and the Lookin workspaces each run is
    already part of."""
    return _run(agent_tools.search_mlflow_runs, filter, experiment_names, server, max_results)


TOOLS = (list_workspaces, get_workspace, get_metrics, get_per_class, get_confusions,
         compare_experiments, get_examples, search_mlflow_runs)


def build_server():
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer("lookin", title="Lookin", version=__version__, instructions=INSTRUCTIONS,
                       log_level="WARNING")
    for tool in TOOLS:
        # Nothing here writes anything. search_mlflow_runs is the one tool
        # that reaches outside Lookin, to servers the user configured.
        # structured_output=False: for a str return the SDK would otherwise
        # send the same JSON a second time, as {"result": "<the text>"}.
        server.add_tool(tool, structured_output=False, annotations=ToolAnnotations(
            read_only_hint=True, open_world_hint=tool is search_mlflow_runs))
    return server


def build_asgi_app(session_handler, assets_dir: str, enable_mcp: bool = True):
    """The Flet web UI at /, plus the MCP endpoint at MCP_PATH.

    This assembles what flet's own `ft.app(view=WEB_BROWSER)` would serve
    (flet_web.fastapi.serve_fastapi_web_app), with one more route in front
    of the UI's catch-all mount. The MCP session manager needs a task group
    for the life of the server; Flet's FastAPI subclass runs its startup and
    shutdown hooks inside one lifespan task, which is what lets the group be
    entered in one hook and left in the other.
    """
    import flet as ft
    import flet_web.fastapi as flet_fastapi

    stack = AsyncExitStack()
    routes = []
    if enable_mcp:
        from mcp.server.transport_security import TransportSecuritySettings

        server = build_server()
        mcp_app = server.streamable_http_app(
            streamable_http_path=MCP_PATH, stateless_http=True, json_response=True,
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        routes = list(mcp_app.routes)

        async def start_mcp():
            await stack.enter_async_context(server.session_manager.run())
    else:
        async def start_mcp():
            pass

    app = flet_fastapi.FastAPI(on_startup=[start_mcp], on_shutdown=[stack.aclose],
                               docs_url=None, redoc_url=None, openapi_url=None)
    app.router.routes.extend(routes)
    app.mount("/", flet_fastapi.app(
        session_handler, assets_dir=assets_dir, web_renderer=ft.WebRenderer.CANVAS_KIT,
        route_url_strategy="path", use_color_emoji=False,
    ))
    return app
