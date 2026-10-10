"""Builds a larger synthetic dataset (reusing the existing example crops,
duplicated under fresh uuids) purely to functionally validate the
scalability work: chunked CSV loading with progress, dtype downcasting,
merge caching, and thumbnail caching/downscaling.

Not a literal 500k-row x 20-experiment fixture (that would take too long to
generate/store here) -- this exercises the same code paths at a smaller but
still meaningfully large scale (default 120k rows x 5 experiments).

Usage:
    uv run python scripts/generate_stress_test.py [--rows 120000] [--experiments 5]
"""
import argparse
import uuid as uuid_lib
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_ROOT = Path(__file__).resolve().parent.parent / "src" / "lookin" / "data"
EXAMPLE_DIR = _DATA_ROOT / "example"
OUT_DIR = _DATA_ROOT / "stress_test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--experiments", type=int, default=5)
    args = parser.parse_args()

    base_df = pd.read_csv(EXAMPLE_DIR / "dataset.csv")
    classes = sorted(base_df["label"].unique())
    n_classes = len(classes)
    base_n = len(base_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, base_n, size=args.rows)
    uuids = [str(uuid_lib.uuid4()) for _ in range(args.rows)]

    dataset_df = base_df.iloc[idx].reset_index(drop=True).copy()
    dataset_df["uuid"] = uuids
    # crop paths still point at the small shared pool of real image files on
    # disk -- this is what exercises the thumbnail cache under heavy reuse.
    dataset_df["crop"] = [f"../example/{c}" for c in dataset_df["crop"]]
    dataset_df.to_csv(OUT_DIR / "dataset.csv", index=False)
    print(f"Wrote {len(dataset_df):,} rows to {OUT_DIR / 'dataset.csv'}")

    true_idx = dataset_df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    for exp_i in range(args.experiments):
        bonus = rng.uniform(1.5, 3.2)
        logits = rng.normal(0, 1.0, size=(args.rows, n_classes)).astype(np.float32)
        logits[np.arange(args.rows), true_idx] += bonus
        logits -= logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        exp_scores /= exp_scores.sum(axis=1, keepdims=True)

        # Score columns are named after the classes themselves, which is
        # what AppState.add_experiment reads a results CSV's class names
        # from -- generic class_1..class_N headers would leave every
        # experiment with class labels that match no dataset label, and the
        # Metrics tab (the main thing a many-experiment fixture is for)
        # would only be able to report that the two share no classes.
        cols = {classes[i]: exp_scores[:, i].round(5) for i in range(n_classes)}
        exp_df = pd.DataFrame({"uuid": uuids, **cols})
        path = OUT_DIR / f"results_exp_{exp_i+1}.csv"
        exp_df.to_csv(path, index=False)
        print(f"Wrote {len(exp_df):,} rows to {path}")


if __name__ == "__main__":
    main()
