"""Bridge between the ArcFace training code and lookin's Embeddings tab.

Generates the three artifacts lookin needs for one experiment, from an
already-trained ArcFaceNet and an evaluation DataLoader:

    {name}_results.csv       -- uuid + one calibrated probability column per
                                 class + "label" (argmax), for lookin's Data tab
    {name}_embeddings.npz    -- uuid (<U..), embeddings (N, D) float32,
                                 for lookin's Embeddings tab
    {name}_prototypes.npz    -- labels (<U.., C), prototypes (C, D) float32
                                 (this head has no sub-centers, so K=1 --
                                 lookin accepts a plain (C, D) array for that)

This module does not import from or modify the training code above --
plug it in either by calling export_lookin_experiment() yourself after
training, or (to get it "for free" whenever the script runs) by adding
LookinExportCallback to your existing Trainer's callbacks= list. Either way
touches zero lines of the training module.

Replace the import below with wherever ArcFaceNet/LitArcFace actually live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

# TODO: point this at the actual module the ArcFace classes live in.
from your_arcface_module import ArcFaceNet, LitArcFace  # noqa: F401


@torch.no_grad()
def collect_embeddings_and_scores(net: ArcFaceNet, loader: DataLoader, device: str = "cuda"):
    """One forward pass over `loader` (must NOT be shuffled -- see
    export_lookin_experiment). Uses predict_logits(), so tau_post/temperature
    calibration is applied if fit_calibration() has already been run;
    harmless (tau_post=0, temperature=1) if it hasn't.

    Returns:
        embeddings : (N, D) float32, L2-normalized (same space ArcFaceHead.cosine() uses).
        probs      : (N, C) float32, softmax of the calibrated logits.
        labels     : (N,) int64, true labels -- returned only so a caller can
                     sanity-check loader/uuid alignment (see below).
    """
    net.eval().to(device)
    embs, probs, labels = [], [], []
    for images, y in loader:
        emb = net.embed(images.to(device))
        logits = net.head.predict_logits(emb)
        embs.append(F.normalize(emb, dim=1).float().cpu())
        probs.append(logits.softmax(dim=1).float().cpu())
        labels.append(y.cpu())
    return torch.cat(embs).numpy(), torch.cat(probs).numpy(), torch.cat(labels).numpy()


def export_lookin_experiment(
    net: ArcFaceNet,
    loader: DataLoader,
    uuids: Sequence[str],
    class_names: Sequence[str],
    output_dir: str,
    experiment_name: str,
    device: str = "cuda",
) -> dict:
    """Write the three lookin artifacts for one experiment.

    `uuids` must be the exact same length and iteration order as `loader`
    (build it with shuffle=False, e.g. `uuids = eval_dataset.df["uuid"].tolist()`
    or whatever your Dataset exposes) -- there is no way to detect a
    misaligned-but-same-length uuid list from here, it would just silently
    tag every embedding with the wrong id.

    `class_names[i]` must be the class whose integer label is `i` in this
    model, and should match the label spelling used in lookin's dataset CSV
    (lookin's Data tab can remap labels afterwards if it doesn't, but
    matching up front avoids that step).
    """
    if len(uuids) == 0:
        raise ValueError("uuids is empty")
    if net.head.num_classes != len(class_names):
        raise ValueError(f"net has {net.head.num_classes} classes but got {len(class_names)} class_names")

    embeddings, probs, _ = collect_embeddings_and_scores(net, loader, device=device)
    if len(embeddings) != len(uuids):
        raise ValueError(
            f"loader produced {len(embeddings)} rows but {len(uuids)} uuids were given -- "
            f"they must describe the exact same, non-shuffled dataset"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    uuids_arr = np.array(uuids, dtype="<U64")
    class_names_arr = np.array(class_names, dtype="<U64")

    # -- results CSV: uuid + one probability column per class + argmax label
    df = pd.DataFrame(probs, columns=list(class_names))
    df.insert(0, "uuid", uuids_arr)
    df["label"] = class_names_arr[probs.argmax(axis=1)]
    csv_path = out_dir / f"{experiment_name}_results.csv"
    df.to_csv(csv_path, index=False)

    # -- embeddings sidecar
    embeddings_path = out_dir / f"{experiment_name}_embeddings.npz"
    np.savez(embeddings_path, uuid=uuids_arr, embeddings=embeddings.astype(np.float32))

    # -- real prototypes sidecar: ArcFaceHead.W is the (C, D) class-weight
    # matrix, normalized the same way ArcFaceHead.cosine() normalizes it.
    # No sub-centers in this head, so this is K=1.
    W = F.normalize(net.head.W.detach().cpu(), dim=1).numpy().astype(np.float32)
    prototypes_path = out_dir / f"{experiment_name}_prototypes.npz"
    np.savez(prototypes_path, labels=class_names_arr, prototypes=W)

    return {"results_csv": csv_path, "embeddings": embeddings_path, "prototypes": prototypes_path}


class LookinExportCallback(L.Callback):
    """Add this to your existing Trainer's callbacks -- no change to
    LitArcFace/ArcFaceNet needed. Runs the export once training finishes,
    for every (loader, uuids) pair given, e.g.:

        trainer = L.Trainer(
            ...,
            callbacks=[LookinExportCallback(
                loaders={"val": val_loader, "test": test_loader},
                uuids={"val": val_uuids, "test": test_uuids},
                class_names=class_names,
                output_dir="lookin_export",
            )],
        )
        trainer.fit(lit_model)
        # lookin_export/val_*, lookin_export/test_* now exist

    `loaders`/`uuids` are keyed by the same experiment name (shown as-is in
    lookin) and must have exactly the same keys.
    """

    def __init__(
        self,
        loaders: Mapping[str, DataLoader],
        uuids: Mapping[str, Sequence[str]],
        class_names: Sequence[str],
        output_dir: str,
        device: str | None = None,
    ):
        super().__init__()
        if loaders.keys() != uuids.keys():
            raise ValueError(f"loaders and uuids must share the same keys, got {sorted(loaders)} vs {sorted(uuids)}")
        self.loaders = loaders
        self.uuids = uuids
        self.class_names = class_names
        self.output_dir = output_dir
        self.device = device

    def on_fit_end(self, trainer, pl_module: LitArcFace) -> None:
        device = self.device or str(pl_module.device)
        for name, loader in self.loaders.items():
            paths = export_lookin_experiment(
                pl_module.net, loader, self.uuids[name], self.class_names,
                self.output_dir, experiment_name=name, device=device,
            )
            print(f"[lookin_export] wrote {name}: {paths}")
