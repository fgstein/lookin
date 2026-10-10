"""
Builds the example dataset used to demo the tool: a subset of CIFAR-10
exported as greyscale cropped PNGs, a dataset.csv with synthetic metadata columns,
and two results CSVs simulating two different model "experiments".

This script needs pyarrow in addition to the app's normal requirements
(pandas, numpy, Pillow) -- pyarrow is only needed to read the CIFAR-10
parquet file, not by the app itself. Run it once to (re)populate
src/lookin/data/example/.

Usage:
    uv run python scripts/generate_example.py [--source path/to/cifar10_test.parquet]

If --source is omitted, the script downloads the CIFAR-10 "test" split
parquet file (~23MB) from the uoft-cs/cifar10 dataset on Hugging Face.
"""
import argparse
import io
import json
import urllib.request
import uuid as uuid_lib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HF_URL = (
    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
    "plain_text/test-00000-of-00001.parquet"
)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Classes that are commonly confused with each other, used to make the
# simulated model scores (and the resulting confusion matrix) look
# realistic instead of uniformly random.
CONFUSION_PAIRS = [
    ("cat", "dog"),
    ("automobile", "truck"),
    ("airplane", "bird"),
    ("deer", "horse"),
    ("ship", "airplane"),
]

PER_CLASS = 80
RNG_SEED = 0


def download_source(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {HF_URL} -> {dest}")
        urllib.request.urlretrieve(HF_URL, dest)
    return dest


def load_cifar_rows(parquet_path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    return rows


def simulate_scores(rng, true_idx: int, n_classes: int, bonus: float, confusion_boost: dict) -> np.ndarray:
    """Simulate a plausible softmax score vector for one sample."""
    logits = rng.normal(0, 1.0, size=n_classes)
    logits[true_idx] += bonus
    for other_idx, boost in confusion_boost.get(true_idx, {}).items():
        logits[other_idx] += rng.normal(boost, 0.4)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def build_confusion_boost():
    boost = {}
    for a, b in CONFUSION_PAIRS:
        ia, ib = CLASSES.index(a), CLASSES.index(b)
        boost.setdefault(ia, {})[ib] = 1.1
        boost.setdefault(ib, {})[ia] = 1.1
    return boost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data" / "example"
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(__file__).parent / ".cache"
    source = args.source or download_source(cache_dir / "cifar10_test.parquet")

    print("Reading parquet...")
    rows = load_cifar_rows(source)

    by_class = {i: [] for i in range(len(CLASSES))}
    for row in rows:
        by_class[row["label"]].append(row)

    rng = np.random.default_rng(RNG_SEED)
    confusion_boost = build_confusion_boost()

    records = []
    score_rows_baseline = []
    score_rows_finetuned = []

    subsets = ["batch_1", "batch_2", "batch_3"]
    quality_flags = ["ok", "blurry", "occluded"]
    quality_weights = [0.75, 0.15, 0.10]

    for class_idx, class_name in enumerate(CLASSES):
        candidates = by_class[class_idx]
        chosen = rng.choice(len(candidates), size=min(PER_CLASS, len(candidates)), replace=False)
        for i in chosen:
            row = candidates[int(i)]
            img_bytes = row["img"]["bytes"]
            # Greyscale on purpose: this example stands in for the kind of
            # imagery the tool is built for (microscopy, radiometry, remote
            # sensing), which is single-channel far more often than not --
            # and it is what makes the display LUTs demonstrable, since a
            # colour crop hides how narrow a range greyscale data can use.
            img = Image.open(io.BytesIO(img_bytes)).convert("L")
            arr = np.asarray(img).astype(np.float32)

            item_uuid = str(uuid_lib.uuid4())
            crop_name = f"{item_uuid}.png"
            img.save(crops_dir / crop_name)

            brightness = float(arr.mean())
            contrast = float(arr.std())
            object_size_pct = float(np.clip(rng.normal(65, 20), 5, 100))
            quality_flag = str(rng.choice(quality_flags, p=quality_weights))
            is_blurry = 1 if quality_flag == "blurry" else 0
            annotator_confidence = float(np.clip(rng.normal(0.85, 0.12), 0.3, 1.0))
            subset = str(rng.choice(subsets))

            records.append({
                "uuid": item_uuid,
                "crop": f"crops/{crop_name}",
                "label": class_name,
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "object_size_pct": round(object_size_pct, 1),
                "quality_flag": quality_flag,
                "is_blurry": is_blurry,
                "annotator_confidence": round(annotator_confidence, 3),
                "subset": subset,
            })

            baseline_scores = simulate_scores(rng, class_idx, len(CLASSES), bonus=2.2, confusion_boost=confusion_boost)
            finetuned_scores = simulate_scores(rng, class_idx, len(CLASSES), bonus=3.0, confusion_boost=confusion_boost)

            # Score columns are named after the actual classes (not a
            # generic "class_1, class_2, ..."), matching real results CSVs
            # -- lookin reads class names directly off these column headers.
            # The extra 'label' column (the argmax already applied) is
            # common in real exports too and must be ignored as a score.
            score_rows_baseline.append({
                "uuid": item_uuid,
                **{cls: round(float(s), 5) for cls, s in zip(CLASSES, baseline_scores)},
                "label": CLASSES[int(np.argmax(baseline_scores))],
            })
            score_rows_finetuned.append({
                "uuid": item_uuid,
                **{cls: round(float(s), 5) for cls, s in zip(CLASSES, finetuned_scores)},
                "label": CLASSES[int(np.argmax(finetuned_scores))],
            })

    dataset_df = pd.DataFrame.from_records(records)
    baseline_df = pd.DataFrame.from_records(score_rows_baseline)
    finetuned_df = pd.DataFrame.from_records(score_rows_finetuned)

    dataset_df.to_csv(out_dir / "dataset.csv", index=False)
    baseline_df.to_csv(out_dir / "baseline_cnn_results.csv", index=False)
    finetuned_df.to_csv(out_dir / "finetuned_resnet_results.csv", index=False)

    with open(out_dir / "classes.json", "w") as f:
        json.dump(CLASSES, f, indent=2)

    print(f"Wrote {len(dataset_df)} rows to {out_dir/'dataset.csv'}")
    print(f"Wrote {len(baseline_df)} rows to {out_dir/'baseline_cnn_results.csv'}")
    print(f"Wrote {len(finetuned_df)} rows to {out_dir/'finetuned_resnet_results.csv'}")
    print(f"Crops saved under {crops_dir}")


if __name__ == "__main__":
    main()
