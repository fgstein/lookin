"""Run Lookin as a self-contained demo: a fake MLflow server, example
workspaces, and the UI -- all in one process, with no internet access.

Everything the demo writes (Home settings, workspaces, download cache) goes
under `.demo/` in the repository rather than ~/.config and ~/.cache, so it
never touches a real setup. The state is seeded on the first run and kept
afterwards, so what is changed in the UI survives a restart; `--reset`
starts over.

What gets seeded:

- Home is pointed at the demo MLflow server, with its experiments shown and
  a few param/metric columns.
- "CIFAR-10 models": the two bundled models, added from MLflow and
  downloaded (green dot).
- "Ensemble study": a run that logged two results CSVs, added but not
  downloaded -- press its download button.
- "Stress sweep": four runs of the stress-test sweep, two of them
  downloaded (partial), when the stress-test data has been generated.
- "Local example": the bundled example from local files, no MLflow.
- "Load test (local files)" and "Load test (MLflow)": the 500k-image load
  test, when scripts/generate_load_test.py has been run. The local one reads
  the generated files in place; the MLflow one is not downloaded, so its
  download button exercises fetching the big CSVs. These are added on the
  first start after the data appears, even if the demo was seeded before.

Usage:
    uv run python scripts/demo.py [--port 8560] [--mlflow-port 5055] [--host 127.0.0.1] [--reset]
    uv run python scripts/demo.py --load-test-dir /big/disk/load_test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO / ".demo"


def _isolate() -> None:
    # Must happen before anything reads config_dir()/user_cache_dir(); both
    # read these variables on every call, so setting them here is enough.
    os.environ["XDG_CONFIG_HOME"] = str(DEMO_DIR / "config")
    os.environ["XDG_CACHE_HOME"] = str(DEMO_DIR / "cache")
    os.environ.pop("LOOKIN_LIBRARY", None)


def _check_ports_free(host: str, ports) -> None:
    # Checked before --reset wipes .demo/: a demo that is already running
    # should not cost the state and then fail on the bind anyway.
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                sys.exit(f"[demo] port {port} on {host} is already in use -- is a demo already running? "
                         "Stop it first, or pick other ports with --port / --mlflow-port.")


def _seed(uri: str) -> None:
    from demo_mlflow_server import EXAMPLE, STRESS

    from lookin.core import home_config
    from lookin.core import library as lib
    from lookin.core import mlflow_client as mlf
    from lookin.core import prefs
    from lookin.core import workspace_sources as ws

    experiments = {e.name: e for e in mlf.list_experiments(uri)}
    shown = [name for name in ("cifar10-classifiers", "cifar10-stress-sweep", "load-test") if name in experiments]
    config = home_config.default_config(uri)
    config["servers"] = [{"name": "demo", "uri": uri}]
    config["experiments"] = {uri: {"shown": shown, "hidden": ["Default"]}}
    config["columns"] = [{"key": "params.backbone", "label": "backbone"},
                         {"key": "params.lr", "label": "lr"},
                         {"key": "metrics.val_acc", "label": "val acc"}]
    path = home_config.home_view_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    prefs.save_mlflow_pref(prefs.MlflowPref(tracking_uri=uri))

    runs, _ = mlf.search_runs(uri, [experiments[name].experiment_id for name in shown])
    by_name = {r.run_name: r for r in runs}

    def candidates(*names):
        out = []
        for name in names:
            run = by_name[name]
            csvs, npzs = mlf.list_csv_and_npz_artifacts(uri, run.run_id)
            out += ws.candidates_for_run(uri, run, csvs, npzs)
        return out

    def make(name, description, names, download=None, tags=()):
        workspace = lib.create_workspace(name, lib.Mount(), description=description, tags=list(tags),
                                         provenance={"mlflow": uri})
        result = ws.add_candidates(workspace, candidates(*names))
        if workspace.mount.dataset_path:
            workspace.dataset_hash = lib.hash_file(workspace.mount.dataset_path)
        if download:
            # Download only some experiments' files, to show a partial state.
            keep = workspace.mount.experiments
            workspace.mount.experiments = [e for e in keep if e.mlflow_run_name in download]
            ws.download(workspace)
            workspace.mount.experiments = keep
        lib.save_workspace(workspace)
        print(f"[demo] workspace '{name}': {result.message()}", flush=True)

    make("CIFAR-10 models", "The two bundled models, straight from MLflow.",
         ["baseline_cnn", "finetuned_resnet"], download={"baseline_cnn", "finetuned_resnet"}, tags=["cifar10"])
    make("Ensemble study", "A run that logged two results CSVs. Not downloaded yet.",
         ["ensemble_cnn_resnet"], tags=["cifar10", "ensemble"])
    if "cifar10-stress-sweep" in experiments:
        make("Stress sweep", "Four runs of the 10k-row sweep; two are downloaded.",
             ["sweep_01", "sweep_02", "sweep_03", "sweep_04"], download={"sweep_01", "sweep_02"}, tags=["sweep"])

    local = lib.Mount(dataset_path=str(EXAMPLE / "dataset.csv"), experiments=[
        lib.MountExperiment(name=stem, path=str(EXAMPLE / f"{stem}_results.csv"),
                            embeddings_path=str(EXAMPLE / f"{stem}_embeddings.npz"),
                            prototypes_path=str(EXAMPLE / f"{stem}_prototypes.npz"))
        for stem in ("baseline_cnn", "finetuned_resnet")])
    lib.create_workspace("Local example", local, description="The bundled example, from local files only.",
                         tags=["local"])
    print("[demo] workspace 'Local example': 2 local experiments", flush=True)
    if not STRESS.is_dir():
        print("[demo] no stress-test data: run scripts/generate_stress_test.py for the sweep", flush=True)


def _seed_load_test(uri: str) -> None:
    from demo_mlflow_server import LOAD_TEST, load_test_manifest

    from lookin.core import home_config
    from lookin.core import library as lib
    from lookin.core import mlflow_client as mlf
    from lookin.core import workspace_sources as ws

    load = load_test_manifest()
    if load is None:
        return
    if not load.get("images_complete"):
        print("[demo] the load test's images are not all drawn yet: missing crops show as broken", flush=True)
    settings = load["settings"]
    summary = f"{settings['rows']:,} images of {settings['size']}px, {len(load['classes'])} classes"

    local = lib.Mount(dataset_path=str(LOAD_TEST / load["dataset"]), experiments=[
        lib.MountExperiment(name=e["name"], path=str(LOAD_TEST / e["results"]),
                            embeddings_path=str(LOAD_TEST / e["embeddings"]) if "embeddings" in e else "",
                            prototypes_path=str(LOAD_TEST / e["prototypes"]) if "prototypes" in e else "")
        for e in load["experiments"]])
    lib.create_workspace("Load test (local files)", local, tags=["load-test"],
                         description=f"{summary}, read in place from {LOAD_TEST}.")
    print(f"[demo] workspace 'Load test (local files)': {len(local.experiments)} experiments", flush=True)

    # Shown on Home, also when the demo was first seeded before the data existed.
    path = home_config.home_view_path()
    config = json.loads(path.read_text(encoding="utf-8"))
    entry = config.setdefault("experiments", {}).setdefault(uri, {"shown": [], "hidden": []})
    if "load-test" not in entry["shown"]:
        entry["shown"].append("load-test")
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    experiment = next(e for e in mlf.list_experiments(uri) if e.name == "load-test")
    runs, _ = mlf.search_runs(uri, [experiment.experiment_id])
    candidates = []
    for run in runs:
        csvs, npzs = mlf.list_csv_and_npz_artifacts(uri, run.run_id)
        candidates += ws.candidates_for_run(uri, run, csvs, npzs)
    workspace = lib.create_workspace("Load test (MLflow)", lib.Mount(), tags=["load-test"],
                                     description=f"{summary}, from MLflow. Not downloaded yet.",
                                     provenance={"mlflow": uri})
    result = ws.add_candidates(workspace, candidates)
    if workspace.mount.dataset_path:
        workspace.dataset_hash = lib.hash_file(workspace.mount.dataset_path)
    lib.save_workspace(workspace)
    print(f"[demo] workspace 'Load test (MLflow)': {result.message()}", flush=True)


def _seed_datasets(uri: str) -> None:
    """The Datasets page: the example folder (and the load test's) as shared
    folders, and the demo server's "datasets" experiment as their origin."""
    from demo_mlflow_server import EXAMPLE, LOAD_TEST

    from lookin.core.datasets import config as dcfg

    for folder in (EXAMPLE, LOAD_TEST):
        if (folder / "dataset.csv").is_file():
            dcfg.add_folder(str(folder))
    dcfg.add_mlflow(uri, "datasets")
    # The demo's datasets name their columns their own way: one mapping
    # (label), the split in subset, the source in camera (load test only).
    dcfg.set_columns({**dcfg.DEFAULT_COLUMNS, dcfg.ROLE_MAPPINGS: ["label"],
                      dcfg.ROLE_SPLIT: "subset", dcfg.ROLE_SOURCE: "camera"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="Host both servers bind to.")
    parser.add_argument("--port", type=int, default=None, help="Lookin port (default: 8560).")
    parser.add_argument("--mlflow-port", type=int, default=5055, help="Demo MLflow server port.")
    parser.add_argument("--reset", action="store_true", help="Throw away .demo/ and seed it again.")
    parser.add_argument("--state-dir", default=None, metavar="DIR",
                        help="Where the demo keeps its state (default: .demo/); lets two demos run side by side.")
    parser.add_argument("--load-test-dir", default=None, metavar="DIR",
                        help="Where scripts/generate_load_test.py wrote its data (default: .load_test/).")
    args = parser.parse_args()
    global DEMO_DIR
    if args.state_dir:
        DEMO_DIR = Path(args.state_dir).resolve()
    port = args.port or 8560

    _check_ports_free(args.host, (args.mlflow_port, port))
    if args.reset and DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    _isolate()
    if args.load_test_dir:
        os.environ["LOOKIN_LOAD_TEST_DIR"] = str(Path(args.load_test_dir).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import demo_mlflow_server

    demo_mlflow_server.start(args.host, args.mlflow_port)
    uri = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.mlflow_port}"
    marker = DEMO_DIR / "seeded"
    if not marker.exists():
        _seed(uri)
        marker.write_text(uri, encoding="utf-8")
    load_marker = DEMO_DIR / "seeded-load-test"
    if not load_marker.exists() and demo_mlflow_server.load_test_manifest() is not None:
        _seed_load_test(uri)
        load_marker.write_text(uri, encoding="utf-8")
    datasets_marker = DEMO_DIR / "seeded-datasets"
    if not datasets_marker.exists():
        _seed_datasets(uri)
        datasets_marker.write_text(uri, encoding="utf-8")
    print(f"[demo] state in {DEMO_DIR}", flush=True)

    from lookin.web.server import run_web
    run_web(host=args.host, port=port)


if __name__ == "__main__":
    main()
