"""The MCP endpoint: Lookin, drivable by an AI agent.

`lookin web` serves it at /mcp, on the same host and port as the interface,
so whatever can reach the interface can reach it -- an agent such as
OpenCode is pointed at http://<host>:<port>/mcp as a "remote" MCP server.

An agent can do what a person does in the interface: read every number,
build and organise workspaces, add experiments and MLflow runs, save views,
record review decisions, export a dataset for training, and have charts,
mosaics of crops and reports made. The work is done in core/ --
agent_tools.py (reading), agent_actions.py (changing), agent_outputs.py
(pictures and files); this module only describes those functions to the
model and puts them on the HTTP route. The descriptions are written for a
model that has never seen Lookin: they are the only documentation it gets.

Transport: MCP's Streamable HTTP, stateless and answering in plain JSON.
Every tool is a single request and a single answer, so there is no session
to keep and no event stream to hold open. A chart or a mosaic comes back as
an image beside its JSON.

Host-header checking is off. The SDK turns it on for a server bound to
localhost and would then refuse every request that arrives through a port
forward or a gateway under another name -- which is exactly how this server
is reached. The endpoint exposes nothing the interface on the same port
does not, and refuses cross-site requests the way the interface does (see
web/server.py). `lookin web --mcp-read-only` serves the reading tools only;
`--no-mcp` turns the endpoint off.
"""
from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field

from lookin import __version__
from lookin.core import agent_actions as actions
from lookin.core import agent_outputs as outputs
from lookin.core import agent_tools
from lookin.core.datasets import agent as dataset_tools

MCP_PATH = "/mcp"

INSTRUCTIONS = """\
Lookin is a visual explorer for image classification results. Its data is organised in workspaces:
one workspace = one labelled dataset (a CSV with one row per image crop: an id, the crop's path, a
label, metadata columns) plus the experiments (model runs, each a results CSV of class scores)
evaluated on it. Workspaces sit in named groups on Lookin's Home page.

Typical reading: list_workspaces, then get_metrics on a workspace to compare its experiments, then
drill down with get_per_class, get_confusions, get_slices, get_calibration, compare_experiments,
get_examples or find_cleaning_candidates. plot_chart and make_mosaic return pictures; generate_report
writes a whole report. Typical building: create_workspace (with its dataset), then add_experiment
(a local results CSV) or search_mlflow_runs + add_mlflow_run.

Datasets on their own: the CSVs of the team's shared dataset folders, whether or not a workspace uses
them. list_datasets, then get_dataset (file, the MLflow run it came from, derivatives) and
get_dataset_stats (class balance, sources, splits, checks); compare_datasets puts two side by side.

Metrics are only comparable within one workspace: two workspaces usually have different datasets.
Every answer carries a "warnings" list -- report those to the user along with the numbers -- and an
"open_url" that opens the same thing in Lookin's interface; give it to the user. Crops carry their own
open_url. Files are written on the machine running Lookin; a "url" next to a path downloads it.
Class labels, column names and ids are case-sensitive and used exactly as Lookin returns them.
"""

WorkspaceId = Annotated[str, Field(description="Workspace id (e.g. 'lkw-3kq9xd') or exact workspace name.")]
ExperimentName = Annotated[str, Field(description="Experiment name, as listed by get_workspace or get_metrics.")]
Experiments = Annotated[list[str] | None, Field(
    description="Experiment names to include. Omit for every experiment of the workspace.")]
Top = Annotated[int, Field(description="Maximum number of rows to return.", ge=1, le=agent_tools.MAX_TOP)]
Where = Annotated[str, Field(description=(
    "Optional condition on the dataset's columns, written like a pandas query: "
    "\"width >= 32 and site in ['north', 'west']\", \"`object size`.between(0.2, 0.8)\", "
    "\"crop.str.contains('cam2/')\". Backticks around a column name with spaces."))]
SaveTo = Annotated[str, Field(description=(
    "Optional path (a file or a folder) on the machine running Lookin to also write the PNG to. "
    "Omit to only receive the image."))]
Dest = Annotated[str, Field(description=(
    "Optional path (a file or a folder) on the machine running Lookin. Omit to write into Lookin's "
    "exports folder, which the answer's url downloads from."))]
Confirm = Annotated[str, Field(description="The same id again, to confirm the deletion.")]

# Set by build_server(read_only=True): the picture tools then refuse save_to.
_read_only = False


def _call(tool, *args):
    """Call a tool function; its failures become a message the model reads.

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
        return tool(*args)
    except (ValueError, KeyError, MlflowError, OSError) as ex:
        raise ToolError(str(ex)) from ex


def _json(result) -> str:
    """Compact, because left to itself the SDK indents a returned dict,
    which roughly doubles what a model with a small context has to read."""
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


def _run(tool, *args) -> str:
    return _json(_call(tool, *args))


def _run_picture(tool, *args) -> list:
    from mcp.server.mcpserver import Image

    answer, png = _call(tool, *args)
    return [_json(answer), Image(data=png, format="png")]


def _no_save_in_read_only(save_to: str) -> None:
    if save_to and _read_only:
        from mcp.server.mcpserver.exceptions import ToolError
        raise ToolError("This Lookin serves its MCP endpoint read-only: nothing can be written, save_to included.")


# ---- reading ---------------------------------------------------------------

def list_workspaces() -> str:
    """List every workspace in Lookin's library: id, name, group, tags, dataset, the names of its
    experiments, whether their files are downloaded, and the link that opens it. Start here."""
    return _run(agent_tools.list_workspaces)


def get_workspace(workspace_id: WorkspaceId) -> str:
    """Describe one workspace without computing anything: its dataset (path, columns, split, row
    filter, dataset filter, review version), its experiments (with their MLflow run when they come from
    one, and whether they are downloaded), and its saved snapshots with their links."""
    return _run(agent_tools.get_workspace, workspace_id)


def get_dataset_summary(workspace_id: WorkspaceId) -> str:
    """What a workspace's dataset holds: row count (before and after its filter), its columns grouped
    by kind (numeric, categorical, binary), and its classes with their counts, largest first."""
    return _run(agent_tools.get_dataset_summary, workspace_id)


def get_column_stats(
    workspace_id: WorkspaceId,
    column: Annotated[str, Field(description="A column of the dataset, as get_dataset_summary lists it.")],
    top: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """One dataset column: for a numeric one its range, mean, quantiles and a 20-bin histogram; for
    a categorical one its most frequent values with their counts; missing values in both."""
    return _run(agent_tools.get_column_stats, workspace_id, column, top)


def get_metrics(workspace_id: WorkspaceId, experiments: Experiments = None) -> str:
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


def get_slices(
    workspace_id: WorkspaceId,
    column: Annotated[str, Field(description="Dataset column to slice by (a sensor, a site, a crop size...).")],
    experiments: Experiments = None,
    metric: Annotated[str, Field(description="One of " + ", ".join(agent_tools.SLICE_METRICS) + ". "
                                 "The per-class ones (" + ", ".join(agent_tools.CLASS_SLICE_METRICS)
                                 + ") need class_label.")] = "accuracy",
    class_label: Annotated[str | None, Field(description="The class a per-class metric is about.")] = None,
    binning: Annotated[str, Field(description="auto (numeric columns with many values in quantile bins, "
                                  "the rest one slice per value), values or quantile.")] = "auto",
    bins: Annotated[int, Field(description="Number of quantile bins.", ge=2, le=50)] = 10,
    min_rows: Annotated[int, Field(description="Slices with fewer rows are flagged 'small' and never ranked "
                                   "first.", ge=0)] = 30,
    sort: Annotated[str, Field(description="worst (worst metric first), value (column order) or size.")] = "worst",
    top: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """Score every experiment again on each slice of a dataset column: "94% overall, except 61% on
    this sensor". Each slice has its row count, number of classes and, for accuracy, a 95% Wilson
    interval -- a low number on a handful of rows is noise, say so when reporting it."""
    return _run(agent_tools.get_slices, workspace_id, column, experiments, metric, class_label, binning, bins,
                min_rows, sort, top)


def get_calibration(
    workspace_id: WorkspaceId,
    experiments: Experiments = None,
    target_accuracy: Annotated[float | None, Field(description=(
        "Optional: also find the most permissive confidence threshold whose accepted predictions reach "
        "this accuracy (e.g. 0.99), and the share of crops it keeps."), ge=0, le=1)] = None,
    threshold: Annotated[float | None, Field(description=(
        "Optional: what accepting only predictions above this confidence keeps, and at what accuracy."),
        ge=0, le=1)] = None,
    fit_temperature: Annotated[bool, Field(description=(
        "Also fit temperature scaling and report log loss and ECE before and after (slower)."))] = False,
) -> str:
    """Whether a model's confidence can be believed and what refusing to answer buys: ECE, MCE, the
    confidence gap (mean confidence minus accuracy; positive is over-confident), AURC and excess AURC
    (area under the risk-coverage curve, and what it costs above a perfect ranking of the errors)."""
    return _run(agent_tools.get_calibration, workspace_id, experiments, target_accuracy, threshold,
                fit_temperature)


def get_confidence(workspace_id: WorkspaceId, experiment: ExperimentName) -> str:
    """Margin (top score minus second score) and normalised entropy of one experiment's predictions,
    for the right and the wrong ones apart: does the model hesitate where it errs?"""
    return _run(agent_tools.get_confidence, workspace_id, experiment)


def get_examples(
    workspace_id: WorkspaceId,
    experiment: Annotated[str, Field(description=(
        "Experiment whose predictions to show. Empty for dataset rows only."))] = "",
    true_label: Annotated[str | None, Field(description="Only crops with this ground-truth label.")] = None,
    predicted_label: Annotated[str | None, Field(description="Only crops predicted as this label.")] = None,
    errors_only: Annotated[bool, Field(description="Only misclassified crops (needs an experiment).")] = True,
    where: Where = "",
    sort: Annotated[str, Field(description="most_confident, least_confident or dataset_order.")] = "most_confident",
    limit: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """Individual crops: uuid, label, prediction, confidence, crop path, a few metadata columns and a
    link that opens the crop in Lookin. By default an experiment's errors, most confident first -- a
    confident error is often a labelling mistake. Combine true_label and predicted_label to look inside
    one cell of the confusion matrix."""
    return _run(agent_tools.get_examples, workspace_id, experiment, true_label, predicted_label, errors_only,
                where, sort, limit)


def find_cleaning_candidates(
    workspace_id: WorkspaceId,
    source: Annotated[str, Field(description=(
        "confident_errors (wrong and confident), margin (scores prefer another class than the label by "
        "at least min_margin), all_errors (every error, most confident first) or consensus (several "
        "experiments agree on the same other class)."))] = "confident_errors",
    experiment: Annotated[str, Field(description="The experiment to read (every source but consensus).")] = "",
    limit: Top = agent_tools.DEFAULT_TOP,
    min_margin: Annotated[float, Field(description="For source=margin.")] = 0.0,
    min_agreeing: Annotated[int, Field(description="For source=consensus: experiments that must agree.",
                                       ge=2)] = 2,
    class_label: Annotated[str | None, Field(description="Only candidates of this class.")] = None,
    match: Annotated[str, Field(description="label (the crop is annotated class_label) or suggestion "
                                "(the evidence suggests class_label).")] = "label",
) -> str:
    """Crops whose annotation deserves a second look, most suspicious first, each with what the
    evidence suggests instead -- the Cleaning tab's search. Decide on them with
    record_review_decisions."""
    return _run(agent_tools.find_cleaning_candidates, workspace_id, source, experiment, limit, min_margin,
                min_agreeing, class_label, match)


def find_similar(
    workspace_id: WorkspaceId,
    experiment: Annotated[str, Field(description="An experiment with embeddings.")],
    uuid: Annotated[str, Field(description="The crop to search from.")],
    k: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """The crops nearest to one crop in an experiment's embedding space (cosine similarity), and how
    many share its label. A crop whose neighbours are mostly another class is mislabelled or genuinely
    ambiguous."""
    return _run(agent_tools.find_similar, workspace_id, experiment, uuid, k)


def get_review(
    workspace_id: WorkspaceId,
    version: Annotated[str, Field(description="A review version. Empty for the active one.")] = "",
) -> str:
    """The review of a workspace's dataset: its versions (parent, frozen or not, how many decisions of
    its own), the active one, and for one version its counts by status and reason and its batches."""
    return _run(agent_tools.get_review, workspace_id, version)


def get_snapshot(snapshot_id: Annotated[str, Field(description="A snapshot id (e.g. 'lk-7f3a2b').")]) -> str:
    """One saved view of a workspace: name, description, the view state it restores (tab, filters and
    settings of each tab) and the link that opens it."""
    return _run(agent_tools.get_snapshot, snapshot_id)


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


def list_mlflow_run_files(
    run_id: Annotated[str, Field(description="An MLflow run id.")],
    server: Annotated[str | None, Field(description="Tracking URI; omit when one server is configured.")] = None,
) -> str:
    """The results CSVs one MLflow run logged, and the embeddings / prototypes .npz beside them --
    what add_mlflow_run can add."""
    return _run(agent_tools.list_mlflow_run_files, run_id, server)


# ---- pictures and files ----------------------------------------------------

def plot_chart(
    workspace_id: WorkspaceId,
    chart: Annotated[Literal[outputs.CHART_KINDS], Field(description=(
        "per_class (a metric per class, one bar per experiment), roc_pr (ROC and precision-recall "
        "curves), confusion_matrix (one experiment, counts in the cells), confusion_overview (where the "
        "errors are across a many-class matrix), reliability and confidence_histogram (one experiment's "
        "calibration), risk_coverage, column (one dataset column's distribution), slices (a metric per "
        "slice of a column) or projection (a 2D map of one experiment's embeddings)."))],
    experiments: Experiments = None,
    metric: Annotated[str, Field(description=(
        "per_class: f1, precision, recall, specificity, roc_auc, average_precision or support (default f1). "
        "slices: as get_slices (default accuracy)."))] = "",
    target: Annotated[str, Field(description="roc_pr: a class label, or empty for the macro-average.")] = "",
    column: Annotated[str, Field(description="column and slices: the dataset column.")] = "",
    classes: Annotated[list[str] | None, Field(description=(
        "per_class and confusion_matrix: the classes to show. Omitted past 20 classes, the most confused "
        "are shown."))] = None,
    class_label: Annotated[str | None, Field(description="slices: the class a per-class metric is about.")] = None,
    binning: Annotated[str, Field(description="slices: auto, values or quantile.")] = "auto",
    bins: Annotated[int, Field(description="slices: quantile bins.", ge=2, le=50)] = 10,
    min_rows: Annotated[int, Field(description="slices: the small-slice floor.", ge=0)] = 30,
    method: Annotated[str, Field(description="projection: pca, umap, tsne or umap_circle.")] = "pca",
    max_points: Annotated[int, Field(description="projection: crops drawn (sampled beyond).", ge=100,
                                     le=outputs.MAX_PROJECTION_POINTS)] = outputs.DEFAULT_PROJECTION_POINTS,
    color_by: Annotated[str, Field(description="projection: a dataset column to colour by; empty for the "
                                   "class.")] = "",
    save_to: SaveTo = "",
) -> list:
    """Draw one chart of a workspace exactly as its tab in Lookin draws it, and return it as a PNG
    image, with a JSON note (title, caveats, link to the tab)."""
    _no_save_in_read_only(save_to)
    return _run_picture(outputs.plot_chart, workspace_id, chart, experiments, metric, target, column, classes,
                        class_label, binning, bins, min_rows, method, max_points, color_by, save_to)


def make_mosaic(
    workspace_id: WorkspaceId,
    experiment: Annotated[str, Field(description=(
        "Experiment whose predictions frame the crops (green right, red wrong) and caption them. "
        "Empty for dataset crops only."))] = "",
    uuids: Annotated[list[str] | None, Field(description="Exactly these crops, in this order.")] = None,
    true_label: Annotated[str | None, Field(description="Only crops with this ground-truth label.")] = None,
    predicted_label: Annotated[str | None, Field(description="Only crops predicted as this label.")] = None,
    errors_only: Annotated[bool, Field(description="Only misclassified crops (needs an experiment).")] = False,
    where: Where = "",
    sort: Annotated[str, Field(description="most_confident, least_confident or dataset_order.")] = "most_confident",
    limit: Annotated[int, Field(description="Crops in the mosaic.", ge=1, le=400)] = 48,
    columns: Annotated[int, Field(description="Crops per row; 0 for a square-ish grid.", ge=0, le=40)] = 0,
    tile_size: Annotated[int, Field(description="Tile edge in pixels.", ge=48, le=512)] = 160,
    caption: Annotated[str, Field(description="Under each crop: none, label, prediction "
                                  "(label -> prediction (confidence)) or uuid.")] = "label",
    lut: Annotated[str, Field(description=(
        "Display mapping for dim (greyscale, 16-bit) crops: none, minmax, percentile (2-98% stretch, the "
        "usual choice) or equalize. Display only; files are never changed."))] = "",
    roll_column: Annotated[str, Field(description=(
        "A dataset column holding each crop's roll in radians: crops are turned upright by whole quarter "
        "turns."))] = "",
    title: Annotated[str, Field(description="A title over the mosaic.")] = "",
    save_to: SaveTo = "",
) -> list:
    """A mosaic of crops as one PNG image -- the Gallery as a picture: named by uuid, or picked like
    get_examples picks them (a confusion cell, the errors of a class, a metadata condition)."""
    _no_save_in_read_only(save_to)
    return _run_picture(outputs.make_mosaic, workspace_id, experiment, uuids, true_label, predicted_label,
                        errors_only, where, sort, limit, columns, tile_size, caption, lut, roll_column, True,
                        title, save_to)


def generate_report(
    workspace_id: WorkspaceId,
    experiments: Experiments = None,
    format: Annotated[Literal["html", "pptx", "md"], Field(description=(
        "html (one self-contained web page), pptx (a slide deck) or md (one Markdown file, charts embedded "
        "as images)."))] = "html",
    baseline: Annotated[str, Field(description="With candidate: add the head-to-head comparison.")] = "",
    candidate: Annotated[str, Field(description="With baseline: add the head-to-head comparison.")] = "",
    per_class: Annotated[bool, Field(description="Include the per-class tables.")] = True,
    confusion: Annotated[bool, Field(description="Include the confusion matrices.")] = True,
    dest: Dest = "",
    per_class_charts: Annotated[bool, Field(description=(
        "Include bar charts of per-class F1, precision, recall and average precision, one bar per experiment."
    ))] = True,
    notes: Annotated[bool, Field(description="Include the workspace's notes (see get_notes), first.")] = True,
) -> str:
    """Write the Metrics tab's report: provenance (dataset, split, review version), the workspace's notes,
    summary metrics, per-class charts and tables, confusion matrices and optionally a comparison. A
    snapshot of the workspace is saved with it so its figures can be reopened. Returns the file's path
    and download url."""
    return _run(outputs.generate_report, workspace_id, experiments, format, baseline, candidate, per_class,
                confusion, dest, True, per_class_charts, notes)


def get_notes(workspace_id: WorkspaceId) -> str:
    """Read the workspace's notes: free text its users wrote about it (decisions, caveats, what was tried),
    as Markdown. A note usually starts with the workspace's name as a title and a "Performance summary"
    written by Lookin between <!-- lookin:recap --> markers. Read them before drawing conclusions about a
    workspace: they often say what its numbers cannot."""
    return _run(agent_tools.get_notes, workspace_id)


# ---- changing the library ---------------------------------------------------

def create_workspace(
    name: Annotated[str, Field(description="The workspace's name.")],
    dataset_path: Annotated[str, Field(description=(
        "The dataset CSV, a path on the machine running Lookin. Can be set later."))] = "",
    group: Annotated[str, Field(description="Home group to put it in (created if new).")] = "",
    description: str = "",
    tags: list[str] | None = None,
    uuid_col: Annotated[str | None, Field(description="Id column. Omit to guess it from the values.")] = None,
    crop_col: Annotated[str | None, Field(description="Crop path column. Omit to guess.")] = None,
    label_col: Annotated[str | None, Field(description="Label column. Omit to guess.")] = None,
    image_root: Annotated[str | None, Field(description="Folder relative crop paths start from.")] = None,
    row_filter_col: Annotated[str, Field(description="Keep only rows where this column...")] = "",
    row_filter_value: Annotated[str, Field(description="...equals this value (e.g. split = test).")] = "",
    split_kind: Annotated[str, Field(description="test, train or other: what the pool is.")] = "test",
) -> str:
    """Create a workspace. The dataset's id, crop and label columns are guessed from its values (or
    read from a lookin.toml beside it) unless given; the answer says which was used."""
    return _run(actions.create_workspace, name, dataset_path, group, description, tags, uuid_col, crop_col,
                label_col, image_root, row_filter_col, row_filter_value, split_kind)


def update_workspace(
    workspace_id: WorkspaceId,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    group: Annotated[str | None, Field(description="Move it to this Home group (created if new).")] = None,
    dataset_path: Annotated[str | None, Field(description="Another dataset CSV (its filter is cleared).")] = None,
    uuid_col: str | None = None,
    crop_col: str | None = None,
    label_col: str | None = None,
    image_root: str | None = None,
    row_filter_col: str | None = None,
    row_filter_value: str | None = None,
    split_kind: Annotated[str | None, Field(description="test, train or other.")] = None,
    review_version: Annotated[str | None, Field(description="The review version the workspace records.")] = None,
) -> str:
    """Change a workspace: only what is given changes. Its id, snapshots and experiments stay."""
    return _run(actions.update_workspace, workspace_id, name, description, tags, group, dataset_path, uuid_col,
                crop_col, label_col, image_root, row_filter_col, row_filter_value, split_kind, review_version)


def duplicate_workspace(workspace_id: WorkspaceId, name: str = "") -> str:
    """Copy a workspace -- dataset, experiments, snapshots -- under a new id, next to the original."""
    return _run(actions.duplicate_workspace, workspace_id, name)


def delete_workspace(workspace_id: WorkspaceId, confirm: Confirm) -> str:
    """Delete a workspace and its snapshots from the library. The dataset and results files themselves
    are not touched. Ask the user first."""
    return _run(actions.delete_workspace, workspace_id, confirm)


def rename_group(group: Annotated[str, Field(description="A Home group's current name.")], new_name: str) -> str:
    """Rename a group of the Home page."""
    return _run(actions.rename_group, group, new_name)


def add_experiment(
    workspace_id: WorkspaceId,
    results_csv: Annotated[str, Field(description=(
        "A results CSV on the machine running Lookin: an id column plus one score column per class, named "
        "after the class."))],
    name: Annotated[str, Field(description="Its name in the workspace; default from the file name.")] = "",
    embeddings: Annotated[str, Field(description="Optional embeddings .npz (found beside the CSV by name "
                                     "otherwise).")] = "",
    prototypes: Annotated[str, Field(description="Optional prototypes .npz (found beside the CSV by name "
                                     "otherwise).")] = "",
) -> str:
    """Add a local results CSV to a workspace as an experiment."""
    return _run(actions.add_experiment, workspace_id, results_csv, name, embeddings, prototypes)


def add_mlflow_run(
    workspace_id: WorkspaceId,
    run_id: Annotated[str, Field(description="The MLflow run id.")],
    server: Annotated[str | None, Field(description="Tracking URI; omit when one server is configured.")] = None,
    csv_path: Annotated[str, Field(description="Which CSV of the run, when it logged several.")] = "",
    name: str = "",
    download: Annotated[bool, Field(description="Fetch its files now.")] = True,
) -> str:
    """Add an MLflow run's results CSV (and its embeddings/prototypes) to a workspace, as dropping the
    run on it on Lookin's Home page does. A run that records another dataset than the workspace's is
    refused; an empty workspace takes the run's dataset."""
    return _run(actions.add_mlflow_run, workspace_id, run_id, server, csv_path, name, download)


def remove_experiment(workspace_id: WorkspaceId, name: ExperimentName) -> str:
    """Take an experiment out of a workspace. Its files are not deleted."""
    return _run(actions.remove_experiment, workspace_id, name)


def rename_experiment(workspace_id: WorkspaceId, name: ExperimentName, new_name: str) -> str:
    """Rename an experiment of a workspace."""
    return _run(actions.rename_experiment, workspace_id, name, new_name)


def set_experiment_color(
    workspace_id: WorkspaceId,
    name: ExperimentName,
    color: Annotated[str, Field(description="'#rrggbb', or empty for its palette colour.")] = "",
) -> str:
    """The colour an experiment is drawn in, in every chart."""
    return _run(actions.set_experiment_color, workspace_id, name, color)


def download_workspace(workspace_id: WorkspaceId) -> str:
    """Fetch every MLflow file a workspace needs onto the machine running Lookin."""
    return _run(actions.download_workspace, workspace_id)


def set_dataset_filter(
    workspace_id: WorkspaceId,
    query: Where = "",
    conditions: Annotated[list[dict] | None, Field(description=(
        "Conditions, all of which a row must meet: {\"column\": c, \"op\": op, ...} with op in, not_in "
        "(+ \"values\": [...]), range (+ \"min\", \"max\"), contains, not_contains, regex (+ \"text\"), "
        "missing or present."))] = None,
) -> str:
    """The workspace's dataset filter (the Filter tab): every number, chart and tab then works on the
    rows it keeps. Both empty clears it. Checked against the dataset before it is saved."""
    return _run(actions.set_dataset_filter, workspace_id, query, conditions)


def create_snapshot(
    workspace_id: WorkspaceId,
    name: str,
    description: str = "",
    tags: list[str] | None = None,
    tab: Annotated[str, Field(description=(
        "The tab it opens on: sources, statistics, gallery, metrics, calibration, confidence, compare, "
        "embeddings, prototypes or cleaning."))] = "",
    from_snapshot: Annotated[str, Field(description="Copy this snapshot's view.")] = "",
    view_state: Annotated[dict | None, Field(description="A view state as get_snapshot returns it.")] = None,
) -> str:
    """Save a view of a workspace as a snapshot -- an id and a link anyone with the library can open."""
    return _run(actions.create_snapshot, workspace_id, name, description, tags, tab, from_snapshot, view_state)


def update_snapshot(snapshot_id: str, name: str | None = None, description: str | None = None,
                    tags: list[str] | None = None) -> str:
    """Rename a snapshot or change its description or tags; its id and view stay."""
    return _run(actions.update_snapshot, snapshot_id, name, description, tags)


def delete_snapshot(snapshot_id: str, confirm: Confirm) -> str:
    """Delete a snapshot. Its link stops working. Ask the user first."""
    return _run(actions.delete_snapshot, snapshot_id, confirm)


def export_workspace(
    workspace_id: WorkspaceId,
    dest: Dest = "",
    include_embeddings: Annotated[bool, Field(description="Include embeddings (much larger).")] = False,
) -> str:
    """Write a self-contained .zip of a workspace -- its dataset, results and snapshots -- for another
    machine (import_workspace reads it back)."""
    return _run(actions.export_workspace, workspace_id, dest, include_embeddings)


def import_workspace(
    zip_path: Annotated[str, Field(description="An exported workspace .zip on the machine running Lookin.")],
    group: str = "",
) -> str:
    """Import an exported workspace into the library."""
    return _run(actions.import_workspace, zip_path, group)


def record_review_decisions(
    workspace_id: WorkspaceId,
    uuids: Annotated[list[str], Field(description="The crops decided on.")],
    status: Annotated[str, Field(description="ok (keep), relabel, exclude or uncertain.")],
    new_label: Annotated[str, Field(description="For relabel: the corrected class.")] = "",
    reason: Annotated[str, Field(description=(
        "Optional: blurred, radiometry, occluded, truncated, multiple_objects, wrong_crop, not_an_object, "
        "duplicate or ambiguous."))] = "",
    note: str = "",
) -> str:
    """Record review decisions on crops, in the active review version, as the decision bar under a crop
    does. Saved at once beside the dataset CSV, which itself is never modified. Several uuids form one
    batch, undoable with undo_review_batch."""
    return _run(actions.record_review_decisions, workspace_id, uuids, status, new_label, reason, note)


def undo_review_batch(workspace_id: WorkspaceId, batch_id: str) -> str:
    """Undo one batch of decisions (get_review lists them)."""
    return _run(actions.undo_review_batch, workspace_id, batch_id)


def create_review_version(
    workspace_id: WorkspaceId,
    name: Annotated[str, Field(description="Lowercase letters, digits, underscores (e.g. v3). Empty for "
                               "the next one.")] = "",
    parent: Annotated[str, Field(description="The version it inherits from; default the active one.")] = "",
    note: str = "",
    independent: Annotated[bool, Field(description="Start empty instead (a second reviewer).")] = False,
) -> str:
    """Start a new review version, made active: it inherits what its parent decided and records only
    what it changes."""
    return _run(actions.create_review_version, workspace_id, name, parent, note, independent)


def write_notes(
    workspace_id: WorkspaceId,
    text: Annotated[str, Field(description=(
        "Markdown: headings (#), **bold**, *italic*, lists (- or 1.), [links](https://...), `code`, > quotes."))],
    mode: Annotated[Literal["append", "replace"], Field(description=(
        "append: add the text at the end of the notes (the default -- keeps what people wrote). "
        "replace: the text becomes the whole notes; read them with get_notes first and keep what should "
        "stay, including the <!-- lookin:recap --> markers."))] = "append",
) -> str:
    """Write the workspace's notes -- what its Notes tab shows and what every report it exports starts
    with. Say who wrote what you add (e.g. a heading "Agent findings, <date>")."""
    return _run(actions.write_notes, workspace_id, text, mode)


def regenerate_notes_recap(workspace_id: WorkspaceId) -> str:
    """Rewrite the "Performance summary" at the top of the workspace's notes from the current numbers
    (accuracy, macro F1, macro AP and rows per experiment) -- after experiments were added or labels
    reviewed. Only the part between the <!-- lookin:recap --> markers changes; a workspace with no notes
    gets them started, titled with its name. Returns the recap written."""
    return _run(actions.regenerate_notes_recap, workspace_id)


def export_reviewed_dataset(
    workspace_id: WorkspaceId,
    versions: Annotated[list[str] | None, Field(description="Review versions to add columns for; default "
                                                "the active one.")] = None,
    dest: Annotated[str, Field(description="A path for the CSV; default beside the dataset.")] = "",
    drop_excluded: Annotated[bool, Field(description="Leave the excluded rows out.")] = False,
    include_reason: bool = True,
    excluded_uuids: Annotated[bool, Field(description="Also write the excluded uuids to a .txt.")] = False,
) -> str:
    """Write the dataset CSV out again -- every row and column, unchanged -- plus label_review_<v>,
    review_status_<v> and review_reason_<v> columns per version: what a training run reads (filter
    status != exclude, read the reviewed label). The versions exported are frozen."""
    return _run(actions.export_reviewed_dataset, workspace_id, versions, dest, drop_excluded, include_reason,
                excluded_uuids)


DatasetName = Annotated[str, Field(description=(
    "Dataset file name ('train.csv' or 'train'), its path from its shared folder's name "
    "('datasets/train/train_clean.csv'), its absolute path, or its content hash."))]


# ---- datasets: the shared folders ---------------------------------------------

def list_datasets() -> str:
    """List every dataset CSV of the shared dataset folders (independently of any workspace): name,
    path, the main dataset it is derived from (if any), whether an MLflow run is linked to it, and --
    when its statistics are already computed -- rows, classes and health (ok / warning / error)."""
    return _run(dataset_tools.list_datasets)


def get_dataset(dataset: DatasetName) -> str:
    """Describe one dataset without reading it: path, size, content hash, the main dataset it
    derives from and its own derivatives, the MLflow run it came from (with how sure the link is:
    confirmed by hash, probable by name and size, or derived from its main dataset) and that run's
    params, metrics and tags, and the workspaces that use it."""
    return _run(dataset_tools.get_dataset, dataset)


def get_dataset_stats(
    dataset: DatasetName,
    sections: Annotated[list[str] | None, Field(description=(
        "Parts to return, among: " + ", ".join(dataset_tools.SECTIONS) + ". "
        "Omit for classes, sources, splits and quality."))] = None,
    top: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """Statistics of one dataset: class balance (count, max/min ratio, normalized entropy, rare
    classes, largest and smallest classes), the distribution of its source and split columns, the
    class x source and class x split tables, every other column, and the checks (duplicate ids,
    missing labels, missing crop files...). The first call on a dataset reads the whole CSV."""
    return _run(dataset_tools.get_dataset_stats, dataset, sections, top)


def compare_datasets(
    dataset_a: Annotated[str, Field(description="The reference dataset (same naming as get_dataset).")],
    dataset_b: Annotated[str, Field(description="The dataset compared against it.")],
    top: Top = agent_tools.DEFAULT_TOP,
) -> str:
    """Compare two datasets: row counts, classes (and sources, splits) only one of them has, the
    Jensen-Shannon divergence of their distributions and the largest changes of share, and through
    their id columns how many rows they share and which labels changed on the shared rows."""
    return _run(dataset_tools.compare_datasets, dataset_a, dataset_b, top)


def find_dataset_issues(
    dataset: Annotated[str | None, Field(description=(
        "One dataset to check. Omit to list the issues of every dataset whose statistics are computed."))] = None,
) -> str:
    """The problems found in a dataset, worst first: missing crop files, duplicate ids, rows without
    a label, duplicate rows, heavy imbalance, classes absent from a split, classes that come from a
    single source."""
    return _run(dataset_tools.find_dataset_issues, dataset)


def get_dataset_rows(
    dataset: DatasetName,
    label: Annotated[str | None, Field(description="Only rows with this label.")] = None,
    source: Annotated[str | None, Field(description="Only rows with this value in the source column.")] = None,
    split: Annotated[str | None, Field(description="Only rows with this value in the split column.")] = None,
    limit: Annotated[int, Field(description="Maximum number of rows to return.", ge=1,
                                le=dataset_tools.MAX_ROWS)] = agent_tools.DEFAULT_TOP,
) -> str:
    """A few rows of a dataset, every column as text, optionally filtered by label, source and
    split, with how many rows match in total."""
    return _run(dataset_tools.get_dataset_rows, dataset, label, source, split, limit)


READ_TOOLS = (
    list_workspaces, get_workspace, get_dataset_summary, get_column_stats, get_metrics, get_per_class,
    get_confusions, compare_experiments, get_slices, get_calibration, get_confidence, get_examples,
    find_cleaning_candidates, find_similar, get_review, get_snapshot, search_mlflow_runs, list_mlflow_run_files,
    plot_chart, make_mosaic, get_notes,
    list_datasets, get_dataset, get_dataset_stats, compare_datasets, find_dataset_issues, get_dataset_rows,
)
WRITE_TOOLS = (
    generate_report, create_workspace, update_workspace, duplicate_workspace, delete_workspace, rename_group,
    add_experiment, add_mlflow_run, remove_experiment, rename_experiment, set_experiment_color,
    download_workspace, set_dataset_filter, create_snapshot, update_snapshot, delete_snapshot,
    export_workspace, import_workspace, record_review_decisions, undo_review_batch, create_review_version,
    export_reviewed_dataset, write_notes, regenerate_notes_recap,
)
TOOLS = READ_TOOLS + WRITE_TOOLS
DESTRUCTIVE_TOOLS = (delete_workspace, delete_snapshot, remove_experiment, undo_review_batch)
# The tools that reach outside Lookin, to the MLflow servers the user configured.
OPEN_WORLD_TOOLS = (search_mlflow_runs, list_mlflow_run_files, add_mlflow_run, download_workspace)


def build_server(read_only: bool = False):
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    global _read_only
    _read_only = read_only
    server = MCPServer("lookin", title="Lookin", version=__version__, instructions=INSTRUCTIONS,
                       log_level="WARNING")
    for tool in READ_TOOLS if read_only else TOOLS:
        reads = tool in READ_TOOLS
        # structured_output=False: for a str return the SDK would otherwise
        # send the same JSON a second time, as {"result": "<the text>"}.
        server.add_tool(tool, structured_output=False, annotations=ToolAnnotations(
            read_only_hint=reads, destructive_hint=tool in DESTRUCTIVE_TOOLS if not reads else None,
            open_world_hint=tool in OPEN_WORLD_TOOLS))
    return server
