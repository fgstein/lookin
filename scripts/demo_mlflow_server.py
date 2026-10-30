"""A fake MLflow tracking server for demos, built from the bundled data.

Lookin only speaks a handful of MLflow's REST endpoints (see
core/mlflow_client.py), so this answers exactly those -- experiments/search,
runs/search, artifacts/list and /get-artifact -- from an in-memory set of
experiments and runs whose artifacts are the CSVs and .npz files under
src/lookin/data/. Standard library plus pandas only: no `mlflow` package,
nothing fetched, so the demo works on a machine with no internet access.

What it serves, chosen to exercise the Home page:

- cifar10-classifiers: the two bundled models (each with embeddings and
  prototypes sidecars), an ensemble run that logged two results CSVs, a run
  evaluated on a *different* dataset (a workspace refuses it), a failed run
  and a still-running one with no artifacts.
- cifar10-stress-sweep: one run per results CSV of the stress-test dataset,
  when it has been generated (scripts/generate_stress_test.py).
- load-test: one run per simulated model of the 500k-image load test, when
  it has been generated (scripts/generate_load_test.py; its directory is
  .load_test/ unless LOOKIN_LOAD_TEST_DIR says otherwise).
- Default: empty, as on a real server.

`filter` supports MLflow's common subset: clauses joined by `and`, each
`metrics.|params.|tags.|attributes.<key> <op> <value>` with =, !=, <, <=, >,
>=, LIKE and ILIKE. Anything else is refused with a 400, as MLflow would.

The MLflow UI links Lookin draws (`/#/experiments/...`) open a bare page
that shows the experiment or run as JSON.

Usage:
    uv run python scripts/demo_mlflow_server.py [--host 127.0.0.1] [--port 5055]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "src" / "lookin" / "data"
EXAMPLE = DATA / "example"
STRESS = DATA / "stress_test"
LOAD_TEST = Path(os.environ.get("LOOKIN_LOAD_TEST_DIR") or DATA.parents[2] / ".load_test")
DATASET_PARAM = "datamodule/configs/database/csv_path"
DEFAULT_PORT = 5055


@dataclass
class FakeRun:
    name: str
    experiment_id: str
    status: str = "FINISHED"
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    description: str = ""
    # artifact path -> file on disk, or bytes built in memory
    artifacts: dict = field(default_factory=dict)
    start_time: int = 0
    end_time: int | None = None
    user: str = "demo"

    @property
    def run_id(self) -> str:
        return hashlib.md5(f"{self.experiment_id}/{self.name}".encode()).hexdigest()

    def to_json(self) -> dict:
        tags = [{"key": "mlflow.runName", "value": self.name},
                {"key": "mlflow.user", "value": self.user},
                {"key": "mlflow.source.name", "value": "train.py"},
                {"key": "mlflow.source.git.commit", "value": hashlib.sha1(self.name.encode()).hexdigest()}]
        if self.description:
            tags.append({"key": "mlflow.note.content", "value": self.description})
        if any(path.endswith(".csv") for path in self.artifacts):
            tags.append({"key": "lookin.results", "value": "true"})
        info = {
            "run_id": self.run_id, "run_uuid": self.run_id, "run_name": self.name,
            "experiment_id": self.experiment_id, "user_id": self.user, "status": self.status,
            "start_time": self.start_time, "artifact_uri": f"mlflow-artifacts:/{self.experiment_id}/{self.run_id}/artifacts",
            "lifecycle_stage": "active",
        }
        if self.end_time is not None:
            info["end_time"] = self.end_time
        return {"info": info, "data": {
            "params": [{"key": k, "value": str(v)} for k, v in self.params.items()],
            "metrics": [{"key": k, "value": v, "timestamp": self.end_time or self.start_time, "step": 0}
                        for k, v in self.metrics.items()],
            "tags": tags,
        }}

    def tags(self) -> dict:
        return {t["key"]: t["value"] for t in self.to_json()["data"]["tags"]}


@dataclass
class FakeExperiment:
    experiment_id: str
    name: str
    description: str = ""
    runs: list = field(default_factory=list)

    def to_json(self) -> dict:
        times = [r.end_time or r.start_time for r in self.runs] or [_NOW - 30 * 86_400_000]
        tags = [{"key": "mlflow.note.content", "value": self.description}] if self.description else []
        return {"experiment_id": self.experiment_id, "name": self.name, "lifecycle_stage": "active",
                "artifact_location": f"mlflow-artifacts:/{self.experiment_id}",
                "creation_time": min(times) - 3_600_000, "last_update_time": max(times), "tags": tags}


_NOW = int(time.time() * 1000)
_HOUR = 3_600_000


def _accuracy(results: pd.DataFrame, labels: pd.Series) -> float:
    classes = [c for c in results.columns if c not in ("uuid", "label")]
    predicted = results[classes].to_numpy().argmax(axis=1)
    truth = results["uuid"].map(labels)
    return float(np.mean(np.asarray(classes)[predicted] == truth.to_numpy()))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, float_format="%.5f")
    return buf.getvalue().encode("utf-8")


def build_experiments() -> list:
    labels = pd.read_csv(EXAMPLE / "dataset.csv").set_index("uuid")["label"]
    baseline = pd.read_csv(EXAMPLE / "baseline_cnn_results.csv")
    resnet = pd.read_csv(EXAMPLE / "finetuned_resnet_results.csv")
    classes = [c for c in baseline.columns if c not in ("uuid", "label")]
    merged = baseline.set_index("uuid")[classes].add(resnet.set_index("uuid")[classes]).div(2.0)
    ensemble = merged.reset_index()

    dataset = str(EXAMPLE / "dataset.csv")
    cifar = FakeExperiment("1", "cifar10-classifiers",
                           "Image classifiers trained on the CIFAR-10 subset bundled with Lookin.")

    def model_run(name, stem, frame, hours_ago, **params):
        artifacts = {f"eval/{stem}_results.csv": EXAMPLE / f"{stem}_results.csv",
                     f"eval/{stem}_embeddings.npz": EXAMPLE / f"{stem}_embeddings.npz",
                     f"eval/{stem}_prototypes.npz": EXAMPLE / f"{stem}_prototypes.npz",
                     "model/config.yaml": f"name: {name}\n".encode()}
        start = _NOW - hours_ago * _HOUR
        return FakeRun(name=name, experiment_id=cifar.experiment_id, params={DATASET_PARAM: dataset, **params},
                       metrics={"val_acc": round(_accuracy(frame, labels), 4), "val_loss": round(1.9 - 1.5 * _accuracy(frame, labels), 4)},
                       artifacts=artifacts, start_time=start, end_time=start + 2 * _HOUR)

    cifar.runs = [
        model_run("baseline_cnn", "baseline_cnn", baseline, 72, backbone="cnn-small", lr="0.001", epochs="30"),
        model_run("finetuned_resnet", "finetuned_resnet", resnet, 48, backbone="resnet18", lr="0.0003", epochs="12"),
        FakeRun(name="ensemble_cnn_resnet", experiment_id=cifar.experiment_id,
                description="Average of baseline_cnn and finetuned_resnet. Logs two results CSVs.",
                params={DATASET_PARAM: dataset, "backbone": "ensemble", "members": "baseline_cnn,finetuned_resnet"},
                metrics={"val_acc": round(_accuracy(ensemble, labels), 4)},
                artifacts={"eval/ensemble_results.csv": _csv_bytes(ensemble),
                           "eval/members/finetuned_resnet_results.csv": EXAMPLE / "finetuned_resnet_results.csv"},
                start_time=_NOW - 24 * _HOUR, end_time=_NOW - 23 * _HOUR),
        FakeRun(name="baseline_cnn_on_reviewed", experiment_id=cifar.experiment_id,
                description="Evaluated on the reviewed dataset: a workspace built on dataset.csv refuses it.",
                params={DATASET_PARAM: str(EXAMPLE / "dataset_reviewed.csv"), "backbone": "cnn-small", "lr": "0.001"},
                metrics={"val_acc": round(_accuracy(baseline, labels), 4)},
                artifacts={"eval/baseline_cnn_results.csv": EXAMPLE / "baseline_cnn_results.csv"},
                start_time=_NOW - 12 * _HOUR, end_time=_NOW - 11 * _HOUR),
        FakeRun(name="vit_tiny", experiment_id=cifar.experiment_id, status="FAILED",
                description="Crashed at epoch 3 (out of memory).",
                params={DATASET_PARAM: dataset, "backbone": "vit-tiny", "lr": "0.0005"},
                start_time=_NOW - 6 * _HOUR, end_time=_NOW - 5 * _HOUR),
        FakeRun(name="convnext_nano", experiment_id=cifar.experiment_id, status="RUNNING",
                params={DATASET_PARAM: dataset, "backbone": "convnext-nano", "lr": "0.0004"},
                metrics={"train_loss": 0.83}, start_time=_NOW - 1 * _HOUR),
    ]

    experiments = [FakeExperiment("0", "Default"), cifar]

    if (STRESS / "dataset.csv").is_file():
        stress_labels = pd.read_csv(STRESS / "dataset.csv", usecols=["uuid", "label"]).set_index("uuid")["label"]
        sweep = FakeExperiment("2", "cifar10-stress-sweep", "Hyper-parameter sweep on the 10k-row stress-test dataset.")
        backbones = ["resnet18", "resnet34", "efficientnet-b0"]
        csvs = sorted(STRESS.glob("results_exp_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        for i, path in enumerate(csvs):
            n = int(path.stem.rsplit("_", 1)[1])
            start = _NOW - (200 - 6 * i) * _HOUR
            sweep.runs.append(FakeRun(
                name=f"sweep_{n:02d}", experiment_id=sweep.experiment_id,
                params={DATASET_PARAM: str(STRESS / "dataset.csv"), "backbone": backbones[n % 3],
                        "lr": f"{[1e-3, 3e-4, 1e-4][n % 3]:g}", "seed": str(n)},
                metrics={"val_acc": round(_accuracy(pd.read_csv(path), stress_labels), 4)},
                artifacts={f"predictions/{path.name}": path},
                start_time=start, end_time=start + 3 * _HOUR))
        experiments.append(sweep)

    load = load_test_manifest()
    if load is not None:
        bench = FakeExperiment("3", "load-test", f"{load['settings']['rows']:,} synthetic "
                               f"{load['settings']['size']}px images over {len(load['classes'])} classes.")
        dataset = str(LOAD_TEST / load["dataset"])
        for i, entry in enumerate(load["experiments"]):
            artifacts = {f"eval/{entry['results']}": LOAD_TEST / entry["results"]}
            for key in ("embeddings", "prototypes"):
                if key in entry:
                    artifacts[f"eval/{entry[key]}"] = LOAD_TEST / entry[key]
            start = _NOW - (96 - 20 * i) * _HOUR
            bench.runs.append(FakeRun(
                name=entry["name"], experiment_id=bench.experiment_id,
                params={DATASET_PARAM: dataset, "backbone": entry["backbone"], "image_size": str(load["settings"]["size"])},
                metrics={"val_acc": entry["val_acc"]}, artifacts=artifacts,
                start_time=start, end_time=start + 9 * _HOUR))
        experiments.append(bench)
    experiments.append(_datasets_experiment())
    return experiments


def _datasets_experiment() -> "FakeExperiment":
    """The runs that produced the datasets: one CSV and a hash.txt each, as
    the dataset scripts log them -- what the Datasets page traces a CSV of
    the shared folders back to. The load test's run has no hash.txt (hashing
    500k rows at every start would be slow), so it shows as "name-matched"."""
    from lookin.core.datasets.fingerprint import dataset_hash, read_for_hash

    datasets = FakeExperiment("4", "datasets", "One run per dataset: the CSV and the hash of its content.")
    example = EXAMPLE / "dataset.csv"
    start = _NOW - 400 * _HOUR
    datasets.runs.append(FakeRun(
        name="cifar10_example", experiment_id=datasets.experiment_id,
        description="800 crops of CIFAR-10, ten classes, three acquisition batches.",
        params={"source_db": "cifar10", "rows": "800", "classes": "10", "builder": "make_dataset.py"},
        artifacts={"data/dataset.csv": example, "hash.txt": (dataset_hash(read_for_hash(example)) + "\n").encode()},
        start_time=start, end_time=start + _HOUR // 4))
    if (LOAD_TEST / "dataset.csv").is_file():
        start = _NOW - 300 * _HOUR
        datasets.runs.append(FakeRun(
            name="load_test_500k", experiment_id=datasets.experiment_id,
            description="Synthetic load test, logged before hash.txt existed.",
            params={"source_db": "synthetic", "builder": "generate_load_test.py"},
            artifacts={"data/dataset.csv": LOAD_TEST / "dataset.csv"},
            start_time=start, end_time=start + _HOUR))
    return datasets


def load_test_manifest() -> dict | None:
    """The load test's manifest.json, or None when it has not been generated."""
    path = LOAD_TEST / "manifest.json"
    if not path.is_file():
        return None
    load = json.loads(path.read_text(encoding="utf-8"))
    return load if load.get("experiments") else None


# ---------------------------------------------------------------------------
# filter syntax
# ---------------------------------------------------------------------------

_CLAUSE = re.compile(
    r"""^\s*(?P<ns>metrics|metric|params|param|tags|tag|attributes|attribute|attr|run)\.
        (?P<key>`[^`]+`|"[^"]+"|[\w./-]+)\s*
        (?P<op>!=|>=|<=|=|>|<|\bILIKE\b|\bLIKE\b)\s*
        (?P<value>'[^']*'|"[^"]*"|[-+\d.eE]+)\s*$""",
    re.IGNORECASE | re.VERBOSE)


class FilterError(ValueError):
    pass


def _parse_filter(text: str) -> list:
    if not text.strip():
        return []
    clauses = []
    for part in re.split(r"\s+and\s+", text.strip(), flags=re.IGNORECASE):
        m = _CLAUSE.match(part)
        if not m:
            raise FilterError(f"Invalid clause(s) in filter string: '{part.strip()}'")
        key = m["key"].strip('`"')
        value = m["value"]
        if value[0] in "'\"":
            value = value[1:-1]
        else:
            try:
                value = float(value)
            except ValueError:
                raise FilterError(f"Invalid value in filter string: '{m['value']}'")
        clauses.append((m["ns"].lower(), key, m["op"].upper(), value))
    return clauses


def _like(pattern: str, value: str, flags=0) -> bool:
    regex = "^" + re.escape(pattern).replace("%", ".*").replace("_", ".") + "$"
    return re.match(regex, value, flags) is not None


def _matches(run: FakeRun, clauses: list) -> bool:
    for ns, key, op, want in clauses:
        if ns.startswith("metric"):
            have = run.metrics.get(key)
            if have is None or isinstance(want, str):
                return False
        elif ns.startswith("param"):
            have = run.params.get(key)
        elif ns.startswith("tag"):
            have = run.tags().get(key)
        else:
            have = {"run_name": run.name, "status": run.status, "run_id": run.run_id,
                    "user_id": run.user}.get(key)
        if have is None:
            return False
        if op == "LIKE":
            ok = _like(str(want), str(have))
        elif op == "ILIKE":
            ok = _like(str(want), str(have), re.IGNORECASE)
        else:
            a, b = (have, want) if isinstance(want, float) else (str(have), str(want))
            ok = {"=": a == b, "!=": a != b, ">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

UI_PAGE = """<!doctype html><meta charset="utf-8"><title>Demo MLflow</title>
<style>body{font:14px system-ui,sans-serif;margin:24px;max-width:960px}pre{background:#f4f4f4;padding:12px;overflow:auto}</style>
<h1>Demo MLflow server</h1><p>A stand-in for the MLflow UI: this shows what the link points at.</p>
<pre id="out">loading...</pre>
<script>
async function show(){
  const m = location.hash.match(/experiments\\/([^/]+)(?:\\/runs\\/([^/]+))?/);
  const out = document.getElementById('out');
  const post = (p, b) => fetch(p, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(b)}).then(r => r.json());
  if (!m) { out.textContent = JSON.stringify(await post('/api/2.0/mlflow/experiments/search', {}), null, 2); return; }
  const runs = await post('/api/2.0/mlflow/runs/search', {experiment_ids:[m[1]], max_results:1000});
  const found = m[2] ? (runs.runs || []).filter(r => r.info.run_id === m[2]) : runs.runs;
  out.textContent = JSON.stringify(found, null, 2);
}
window.addEventListener('hashchange', show); show();
</script>"""


class DemoMlflow:
    def __init__(self, experiments: list):
        self.experiments = {e.experiment_id: e for e in experiments}
        self.runs = {r.run_id: r for e in experiments for r in e.runs}

    def artifact_size(self, run: FakeRun, path: str) -> int:
        source = run.artifacts[path]
        return len(source) if isinstance(source, bytes) else Path(source).stat().st_size

    def list_artifacts(self, run: FakeRun, path: str) -> list:
        prefix = f"{path.strip('/')}/" if path.strip("/") else ""
        entries = {}
        for full in run.artifacts:
            if not full.startswith(prefix):
                continue
            head = full[len(prefix):].split("/", 1)
            child = prefix + head[0]
            if len(head) > 1:
                entries[child] = {"path": child, "is_dir": True}
            else:
                entries[child] = {"path": child, "is_dir": False, "file_size": self.artifact_size(run, full)}
        return sorted(entries.values(), key=lambda e: e["path"])


def make_handler(state: DemoMlflow):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[demo-mlflow] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"))

        def _error(self, code: int, error_code: str, message: str) -> None:
            self._json({"error_code": error_code, "message": message}, code)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return {}

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self._send(200, UI_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if url.path == "/health":
                return self._send(200, b"OK", "text/plain")
            run = state.runs.get(query.get("run_id", ""))
            if url.path in ("/api/2.0/mlflow/artifacts/list", "/ajax-api/2.0/mlflow/artifacts/list"):
                if run is None:
                    return self._error(404, "RESOURCE_DOES_NOT_EXIST", f"Run '{query.get('run_id')}' not found")
                return self._json({"root_uri": f"mlflow-artifacts:/{run.experiment_id}/{run.run_id}/artifacts",
                                   "files": state.list_artifacts(run, query.get("path", ""))})
            if url.path == "/get-artifact":
                source = run.artifacts.get(query.get("path", "")) if run else None
                if source is None:
                    return self._error(404, "RESOURCE_DOES_NOT_EXIST", "Artifact not found")
                if isinstance(source, bytes):
                    return self._send(200, source, "application/octet-stream")
                # Streamed: the load test's results CSVs are hundreds of MB each.
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(Path(source).stat().st_size))
                self.end_headers()
                with open(source, "rb") as handle:
                    shutil.copyfileobj(handle, self.wfile, 1 << 20)
                return None
            if url.path == "/api/2.0/mlflow/runs/get":
                if run is None:
                    return self._error(404, "RESOURCE_DOES_NOT_EXIST", "Run not found")
                return self._json({"run": run.to_json()})
            return self._error(404, "ENDPOINT_NOT_FOUND", f"No endpoint {url.path}")

        def do_POST(self):
            path = urlparse(self.path).path.replace("/ajax-api/", "/api/")
            body = self._body()
            if path == "/api/2.0/mlflow/experiments/search":
                experiments = [e.to_json() for e in state.experiments.values()]
                limit = int(body.get("max_results") or 1000)
                return self._json({"experiments": experiments[:limit]})
            if path == "/api/2.0/mlflow/runs/search":
                try:
                    clauses = _parse_filter(str(body.get("filter") or ""))
                except FilterError as ex:
                    return self._error(400, "INVALID_PARAMETER_VALUE", str(ex))
                ids = [str(i) for i in body.get("experiment_ids") or []]
                runs = [r for i in ids if i in state.experiments for r in state.experiments[i].runs]
                runs = [r for r in runs if _matches(r, clauses)]
                runs.sort(key=lambda r: r.start_time, reverse=True)
                limit = int(body.get("max_results") or 1000)
                offset = int(body.get("page_token") or 0)
                page = runs[offset:offset + limit]
                payload = {"runs": [r.to_json() for r in page]}
                if offset + limit < len(runs):
                    payload["next_page_token"] = str(offset + limit)
                return self._json(payload)
            return self._error(404, "ENDPOINT_NOT_FOUND", f"No endpoint {path}")

    return Handler


def start(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Build the data and serve it on a daemon thread; returns the server."""
    state = DemoMlflow(build_experiments())
    server = ThreadingHTTPServer((host, port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True, name="demo-mlflow").start()
    total = sum(len(e.runs) for e in state.experiments.values())
    print(f"[demo-mlflow] serving {len(state.experiments)} experiments, {total} runs "
          f"on http://{host}:{port}", flush=True)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = start(args.host, args.port)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
