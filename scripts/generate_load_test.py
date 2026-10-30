"""Builds a load-test dataset: 500k distinct 256x256 images over 40 classes,
a dataset.csv with metadata columns, and a few results CSVs (with embeddings
and prototypes sidecars) simulating models of increasing quality.

Unlike generate_stress_test.py, which points every row at the same small
pool of example crops, every row here has its own image file on disk, so
the Gallery, the carousel and the thumbnail cache are measured against a
realistic number of cold reads.

The images are synthetic and readable at a glance: a class is a shape
(circle, square, triangle, ...) in a colour (red, orange, ...), so
"teal_star" is a teal star. Classes sharing a shape or a colour are the ones
the simulated models confuse. The dataset is imbalanced (a long tail of rare
classes), and the metadata columns are tied to how each image was drawn:
`is_blurry` images are blurred, `low_light` ones are dark, `occluded` ones
have a grey patch over the object.

Nothing is fetched: everything is drawn locally with Pillow and numpy.

Everything is derived from --seed, so an interrupted run picks up where it
stopped: the tables are rebuilt identically and images already on disk are
skipped. Before drawing, a sample is rendered to estimate the disk space
needed, and the run refuses to start if it does not fit.

Output (default: .load_test/ at the repository root, git-ignored):
    dataset.csv                  uuid, crop, label + metadata columns
    crops/<ab>/<uuid>.png        one image per row, sharded by uuid prefix
    <model>_results.csv          one score column per class
    <model>_embeddings.npz       uuid + embeddings (unless --no-embeddings)
    <model>_prototypes.npz       two prototypes per class
    manifest.json                what was generated; read by the demo

Usage:
    uv run python scripts/generate_load_test.py [--rows 500000] [--classes 40] [--size 256]
        [--experiments 3] [--format png|jpg] [--workers N] [--out DIR] [--estimate-only]

Then: uv run python scripts/demo.py --web   (adds the "Load test" workspaces)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import uuid as uuid_lib
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / ".load_test"
MANIFEST = "manifest.json"

SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star", "ring", "hexagon"]
COLORS = {
    "red": (214, 48, 49), "orange": (230, 126, 34), "yellow": (241, 196, 15), "green": (39, 174, 96),
    "teal": (22, 160, 133), "blue": (41, 98, 255), "purple": (142, 68, 173), "pink": (232, 67, 147),
}
QUALITY = ["ok", "low_light", "occluded"]
QUALITY_P = [0.86, 0.08, 0.06]
MODELS = [
    # name, backbone, bonus on the true class, share of rows pushed towards a look-alike class
    ("resnet18_baseline", "resnet18", 2.6, 0.30),
    ("efficientnet_b2", "efficientnet-b2", 3.4, 0.22),
    ("vit_small_finetuned", "vit-small", 4.2, 0.15),
    ("convnext_tiny", "convnext-tiny", 3.8, 0.18),
    ("resnet50_distilled", "resnet50", 3.1, 0.25),
]
CHUNK = 50_000  # rows per block when building score tables, bounds memory


def class_names(n: int) -> list[str]:
    if n > len(SHAPES) * len(COLORS):
        sys.exit(f"--classes is at most {len(SHAPES) * len(COLORS)} ({len(SHAPES)} shapes x {len(COLORS)} colours)")
    colors = list(COLORS)
    return [f"{colors[i // len(SHAPES)]}_{SHAPES[i % len(SHAPES)]}" for i in range(n)]


# ---------------------------------------------------------------------------
# Rows: everything about an image, decided up front from the seed
# ---------------------------------------------------------------------------

def build_rows(n: int, classes: list[str], size: int, ext: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_classes = len(classes)
    # Long tail: the most common class is ~9x the rarest at 40 classes.
    weights = 1.0 / np.arange(1, n_classes + 1) ** 0.6
    order = rng.permutation(n_classes)
    label_idx = rng.choice(n_classes, size=n, p=(weights / weights.sum())[np.argsort(order)])

    uuids = [str(uuid_lib.UUID(bytes=bytes(b), version=4)) for b in rng.integers(0, 256, size=(n, 16), dtype=np.uint8)]
    quality = rng.choice(len(QUALITY), size=n, p=QUALITY_P)
    is_blurry = rng.random(n) < 0.07
    object_size = np.clip(rng.normal(0.45, 0.14, n), 0.12, 0.85)
    background = rng.uniform(70, 200, n)
    contrast = rng.uniform(0.55, 1.0, n)
    light = np.where(quality == QUALITY.index("low_light"), rng.uniform(0.18, 0.4, n), rng.uniform(0.85, 1.1, n))
    brightness = background * light
    # How sure the annotator was: lower on hard images, as in real labelling.
    hard = is_blurry | (quality != 0) | (object_size < 0.22)
    annotator = np.clip(rng.normal(np.where(hard, 0.72, 0.93), 0.06), 0.3, 1.0)
    capture = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 600 * 86_400, n), unit="s")

    return pd.DataFrame({
        "uuid": uuids,
        "crop": [f"crops/{u[:2]}/{u}.{ext}" for u in uuids],
        "label": np.asarray(classes)[label_idx],
        "width": np.full(n, size, dtype=np.int32),
        "height": np.full(n, size, dtype=np.int32),
        "brightness": brightness.round(2),
        "contrast": (contrast * 100).round(2),
        "object_size_pct": (object_size * 100).round(1),
        "quality_flag": np.asarray(QUALITY)[quality],
        "is_blurry": is_blurry.astype(np.int8),
        "annotator_confidence": annotator.round(3),
        "camera": [f"cam_{i:02d}" for i in rng.integers(1, 13, n)],
        "subset": rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.15, 0.15]),
        "capture_date": capture.strftime("%Y-%m-%d"),
        # Drawing parameters, not written to dataset.csv.
        "_label_idx": label_idx,
        "_cx": rng.uniform(0.3, 0.7, n),
        "_cy": rng.uniform(0.3, 0.7, n),
        "_angle": rng.uniform(0, 2 * math.pi, n),
        "_background": background,
        "_light": light,
        "_contrast": contrast,
        "_size": object_size,
        "_tint": rng.integers(0, 3, n),
        "_gradient": rng.integers(0, 4, n),
        "_occluded": quality == QUALITY.index("occluded"),
        "_ox": rng.uniform(0.2, 0.6, n),
        "_oy": rng.uniform(0.2, 0.6, n),
    })


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

_RAMPS: dict[int, list[np.ndarray]] = {}


def _ramps(size: int) -> list[np.ndarray]:
    if size not in _RAMPS:
        ramp = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        horizontal = np.broadcast_to(ramp, (size, size))
        diagonal = (horizontal + horizontal.T) / 2
        _RAMPS[size] = [horizontal, horizontal.T, diagonal, diagonal[:, ::-1]]
    return _RAMPS[size]


def _polygon(shape: str, cx: float, cy: float, r: float, angle: float) -> list[tuple[float, float]]:
    if shape == "square":
        h = r / math.sqrt(2)
        return _rotate([(h, h), (-h, h), (-h, -h), (h, -h)], cx, cy, angle)
    if shape == "triangle":
        return _regular(3, cx, cy, r, angle)
    if shape == "diamond":
        return _rotate([(0, -r), (0.6 * r, 0), (0, r), (-0.6 * r, 0)], cx, cy, angle)
    if shape == "hexagon":
        return _regular(6, cx, cy, r, angle)
    if shape == "star":
        pts = []
        for k in range(10):
            rad = r if k % 2 == 0 else 0.45 * r
            a = k * math.pi / 5
            pts.append((rad * math.sin(a), -rad * math.cos(a)))
        return _rotate(pts, cx, cy, angle)
    if shape == "cross":
        t = 0.32 * r
        pts = [(-t, -r), (t, -r), (t, -t), (r, -t), (r, t), (t, t), (t, r), (-t, r), (-t, t), (-r, t), (-r, -t), (-t, -t)]
        return _rotate(pts, cx, cy, angle)
    raise ValueError(shape)


def _regular(k: int, cx: float, cy: float, r: float, angle: float) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(angle + 2 * math.pi * i / k), cy + r * math.sin(angle + 2 * math.pi * i / k)) for i in range(k)]


def _rotate(pts, cx, cy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def render(row: dict, classes: list[str], size: int) -> Image.Image:
    name = classes[row["_label_idx"]]
    color_name, shape = name.split("_", 1)
    base = np.asarray(COLORS[color_name], dtype=np.float32)

    # Background: a grey with a slight tint and a soft gradient.
    tint = np.array([[1.0, 0.97, 0.94], [0.94, 0.97, 1.0], [0.97, 1.0, 0.95]], dtype=np.float32)[row["_tint"]]
    bg = row["_background"] + 28.0 * _ramps(size)[row["_gradient"]]
    image = np.clip(bg[..., None] * tint, 0, 255).astype(np.uint8)
    img = Image.fromarray(image, "RGB")

    draw = ImageDraw.Draw(img)
    cx, cy = row["_cx"] * size, row["_cy"] * size
    r = row["_size"] * size / 2
    # Lower contrast pulls the object's colour towards the background.
    k = row["_contrast"]
    fill = tuple(int(v) for v in base * k + row["_background"] * (1 - k))
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "ring":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=max(2, int(r * 0.3)))
    else:
        draw.polygon(_polygon(shape, cx, cy, r, row["_angle"]), fill=fill)

    if row["_occluded"]:
        x0, y0 = row["_ox"] * size, row["_oy"] * size
        draw.rectangle([x0, y0, x0 + 0.35 * size, y0 + 0.3 * size], fill=(128, 128, 128))
    if row["is_blurry"]:
        img = img.filter(ImageFilter.GaussianBlur(radius=size / 64))
    if row["_light"] != 1.0:
        img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) * row["_light"], 0, 255).astype(np.uint8), "RGB")
    return img


def _save(img: Image.Image, target, fmt: str) -> None:
    if fmt == "jpg":
        img.save(target, format="JPEG", quality=90)
    else:
        img.save(target, format="PNG", compress_level=6)


def _render_chunk(job) -> tuple[int, int]:
    out, rows, classes, size, fmt = job
    written = skipped = 0
    made = set()
    for row in rows:
        target = out / row["crop"]
        if target.exists():
            skipped += 1
            continue
        if target.parent not in made:
            target.parent.mkdir(parents=True, exist_ok=True)
            made.add(target.parent)
        tmp = target.with_name(target.name + ".tmp")
        _save(render(row, classes, size), tmp, fmt)
        os.replace(tmp, target)  # an interrupted write never leaves a half image behind
        written += 1
    return written, skipped


def estimate_bytes(rows: pd.DataFrame, classes: list[str], size: int, fmt: str, sample: int = 300) -> float:
    picks = rows.sample(n=min(sample, len(rows)), random_state=1).to_dict("records")
    total = 0
    for row in picks:
        buf = BytesIO()
        _save(render(row, classes, size), buf, fmt)
        total += buf.tell()
    return total / len(picks)


# ---------------------------------------------------------------------------
# Simulated models
# ---------------------------------------------------------------------------

def write_model(out: Path, rows: pd.DataFrame, classes: list[str], model: tuple, index: int, seed: int,
                embeddings: bool, dim: int) -> dict:
    name, backbone, bonus, lookalike_share = model
    rng = np.random.default_rng(seed + 1000 + index)
    n, n_classes = len(rows), len(classes)
    shape_of = np.arange(n_classes) % len(SHAPES)
    color_of = np.arange(n_classes) // len(SHAPES)
    # Look-alikes: same shape, other colour (mostly) or same colour, other shape.
    lookalikes = [np.flatnonzero(((shape_of == shape_of[c]) | (color_of == color_of[c])) & (np.arange(n_classes) != c))
                  for c in range(n_classes)]
    difficulty = rng.uniform(-0.6, 0.6, n_classes)  # some classes are harder for this model

    # Class centres in embedding space; a row sits between its true and predicted class.
    centres = rng.normal(0, 1, (n_classes, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    centres *= 4.0

    truth_all = rows["_label_idx"].to_numpy()
    hard_all = ((rows["is_blurry"].to_numpy() == 1) | (rows["quality_flag"].to_numpy() != "ok")).astype(np.float32)
    results_path = out / f"{name}_results.csv"
    tmp = results_path.with_name(results_path.name + ".tmp")
    emb = np.empty((n, dim), dtype=np.float32) if embeddings else None
    correct = 0
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        truth = truth_all[start:stop]
        m = stop - start
        logits = rng.normal(0, 1.0, (m, n_classes)).astype(np.float32)
        logits[np.arange(m), truth] += bonus + difficulty[truth] - 1.4 * hard_all[start:stop]
        pushed = rng.random(m) < lookalike_share
        for i in np.flatnonzero(pushed):
            other = rng.choice(lookalikes[truth[i]])
            logits[i, other] += rng.normal(bonus * 0.85, 0.6)
        logits -= logits.max(axis=1, keepdims=True)
        scores = np.exp(logits)
        scores /= scores.sum(axis=1, keepdims=True)
        predicted = scores.argmax(axis=1)
        correct += int((predicted == truth).sum())

        frame = pd.DataFrame(scores, columns=classes)
        frame.insert(0, "uuid", rows["uuid"].to_numpy()[start:stop])
        frame.to_csv(tmp, mode="w" if start == 0 else "a", header=start == 0, index=False, float_format="%.5f")

        if emb is not None:
            mix = rng.uniform(0.0, 0.35, (m, 1)).astype(np.float32) + np.where(predicted != truth, 0.5, 0.0)[:, None]
            emb[start:stop] = ((1 - mix) * centres[truth] + mix * centres[predicted]
                               + rng.normal(0, 1.0, (m, dim)).astype(np.float32))
        print(f"  {name}: {stop:,}/{n:,} rows", end="\r", flush=True)
    os.replace(tmp, results_path)
    entry = {"name": name, "backbone": backbone, "results": results_path.name,
             "val_acc": round(correct / n, 4)}
    if emb is not None:
        emb_path = out / f"{name}_embeddings.npz"
        np.savez(emb_path, uuid=rows["uuid"].to_numpy().astype("<U36"), embeddings=emb)
        offsets = rng.normal(0, 0.8, (n_classes, 2, dim)).astype(np.float32)
        proto_path = out / f"{name}_prototypes.npz"
        np.savez(proto_path, labels=np.asarray(classes, dtype="<U64"), prototypes=centres[:, None, :] + offsets)
        entry.update(embeddings=emb_path.name, prototypes=proto_path.name)
    print(f"  {name}: {n:,} rows, accuracy {entry['val_acc']:.3f}      ", flush=True)
    return entry


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--classes", type=int, default=40)
    parser.add_argument("--size", type=int, default=256, help="Image side in pixels.")
    parser.add_argument("--experiments", type=int, default=3, help=f"Simulated models, 1 to {len(MODELS)}.")
    parser.add_argument("--format", choices=["png", "jpg"], default="png")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip the embeddings/prototypes sidecars.")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output directory (default: .load_test/ in the repository). Must not be under src/.")
    parser.add_argument("--estimate-only", action="store_true", help="Print the disk estimate and stop.")
    parser.add_argument("--skip-space-check", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    if (REPO / "src") in out.parents or out == REPO / "src":
        sys.exit("--out must not be under src/: the package would ship gigabytes of test images")
    if not 1 <= args.experiments <= len(MODELS):
        sys.exit(f"--experiments must be between 1 and {len(MODELS)}")
    classes = class_names(args.classes)
    settings = {"rows": args.rows, "classes": args.classes, "size": args.size, "format": args.format,
                "seed": args.seed}

    manifest_path = out / MANIFEST
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("settings") != settings:
            sys.exit(f"{out} already holds a load test built with {previous.get('settings')}; "
                     "pick another --out or delete it first")

    t0 = time.time()
    print(f"Building {args.rows:,} rows over {len(classes)} classes...", flush=True)
    rows = build_rows(args.rows, classes, args.size, args.format, args.seed)
    counts = rows["label"].value_counts()
    print(f"  class sizes: {counts.min():,} to {counts.max():,} rows ({time.time() - t0:.1f}s)", flush=True)

    per_image = estimate_bytes(rows, classes, args.size, args.format)
    # On a resumed run, only the images not drawn yet need space (sampled, not counted).
    sample = rows["crop"].iloc[:: max(1, args.rows // 2000)]
    missing = round(args.rows * np.mean([not (out / c).exists() for c in sample])) if out.exists() else args.rows
    tables = args.rows * (40 + 9 * len(classes)) * args.experiments + args.rows * 220
    if not args.no_embeddings:
        tables += args.experiments * args.rows * (args.embedding_dim * 4 + 144)
    # Files are rounded up to whole 4 KiB blocks on disk.
    need = missing * max(4096, math.ceil(per_image / 4096) * 4096) + tables
    existing = next(p for p in (out, *out.parents) if p.exists())
    free = shutil.disk_usage(existing).free
    print(f"  ~{per_image / 1024:.1f} KiB per {args.format.upper()} image; {missing:,} images to draw; "
          f"needs ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free on {out}", flush=True)
    if args.estimate_only:
        return
    if need > free * 0.95 and not args.skip_space_check:
        sys.exit("Not enough disk space: use --out on a larger disk, --format jpg, or fewer --rows")
    out.mkdir(parents=True, exist_ok=True)

    public = [c for c in rows.columns if not c.startswith("_")]
    dataset_path = out / "dataset.csv"
    if not dataset_path.is_file():
        rows[public].to_csv(dataset_path.with_suffix(".tmp"), index=False)
        os.replace(dataset_path.with_suffix(".tmp"), dataset_path)
        print(f"Wrote {dataset_path}", flush=True)

    experiments = []
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    done = {e["name"]: e for e in previous.get("experiments", [])}
    for i, model in enumerate(MODELS[: args.experiments]):
        entry = done.get(model[0])
        wants = not args.no_embeddings
        if entry and (out / entry["results"]).is_file() and (not wants or "embeddings" in entry):
            experiments.append(entry)
            continue
        experiments.append(write_model(out, rows, classes, model, i, args.seed, wants, args.embedding_dim))
        _write_manifest(manifest_path, settings, classes, experiments, images_complete=False)
    _write_manifest(manifest_path, settings, classes, experiments, images_complete=False)

    print(f"Drawing images with {args.workers} workers...", flush=True)
    draw = rows[["crop", "is_blurry"] + [c for c in rows.columns if c.startswith("_")]]
    total = len(draw)
    step = 1000
    # Chunks are turned into dicts one at a time: 500k rows as dicts at once is over a GB.
    jobs = ((out, draw.iloc[i:i + step].to_dict("records"), classes, args.size, args.format)
            for i in range(0, total, step))
    written = skipped = 0
    t1 = time.time()
    with Pool(args.workers) as pool:
        for w, s in pool.imap_unordered(_render_chunk, jobs):
            written += w
            skipped += s
            done_n = written + skipped
            rate = written / max(time.time() - t1, 1e-6)
            eta = (total - done_n) / rate if rate else 0
            print(f"  {done_n:,}/{total:,} images ({skipped:,} already there), "
                  f"{rate:,.0f}/s, ETA {eta / 60:.0f} min   ", end="\r", flush=True)
    print()
    _write_manifest(manifest_path, settings, classes, experiments, images_complete=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min: {out}", flush=True)


def _write_manifest(path: Path, settings: dict, classes: list, experiments: list, images_complete: bool) -> None:
    payload = {"settings": settings, "classes": classes, "dataset": "dataset.csv",
               "images_complete": images_complete, "experiments": experiments}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
