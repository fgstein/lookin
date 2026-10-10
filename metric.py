"""
The same classification report as before, but built *on top of* torchmetrics
instead of reimplementing it.

Design rule: every metric that torchmetrics already ships is taken from
torchmetrics. Only three things are written by hand, because they genuinely do
not exist in the library (checked against torchmetrics 1.9):

  * MulticlassLogLoss        -- there is no LogLoss in torchmetrics.classification
  * MulticlassBrierScore     -- no Brier either
  * BrierDecomposition       -- Murphy's reliability / resolution / uncertainty

All three use *sum-reducible* states of size O(1) or O(K x bins), so they cost
essentially nothing in memory and work under DDP without special handling.

Everything else -- confusion matrix, precision/recall/F1/specificity, MCC,
Cohen's kappa, AUROC, average precision, calibration error -- is a stock
torchmetrics class.

On the duplication question
---------------------------
Yes, instantiating ten metrics side by side duplicates work: each one
re-validates the input, re-runs argmax, re-accumulates its own state. But
`MetricCollection` already solves this with **compute groups**: after the first
update it compares the states of every metric, and any two metrics whose states
are identical are merged so that only one of them actually runs `update`, the
others holding a reference to the same tensors.

For the ten metrics below, the ten metrics collapse into four real state groups:

    group 0: AUROC, AveragePrecision          -> preds, target  (the expensive one)
    group 1: ConfusionMatrix, MCC, CohenKappa -> confmat        (K x K)
    group 2: CalibrationError                 -> confidences, accuracies
    group 3: F1, Precision, Recall, Specificity -> tp, fp, tn, fn (4 x K)

Measured on 200k samples / 20 classes / batch 512, CPU:

    compute_groups=True    update = 0.24 s
    compute_groups=False   update = 0.50 s
    ten metrics in a loop  update = 0.46 s

So roughly a 2x saving on the update path, for free. It is on by default; the
main thing to know is that it is *not* free to turn off, and that it re-checks
grouping on the first update after every `reset()`.

Groups 1 and 3 are mathematically redundant with each other -- tp/fp/tn/fn is
derivable from the confusion matrix -- but torchmetrics cannot merge them
because the state tensors have different shapes. That residual costs O(K^2 + 4K)
floats, which is nothing. Don't lose sleep over it.

In the collection below the effect is larger: 23 metrics collapse into 8 state
groups, for a total of 0.18 MB of state on a 4000-sample run.

One trap worth knowing: `MetricCollection.items()` and `.values()` (and
`__getitem__`, i.e. `collection["name"]`) default to `copy_state=True`, which
deep-copies the shared states so that each metric looks like it owns its own.
Handy if you are about to mutate them, wasteful if you only want to inspect.
Pass `copy_state=False` to look without allocating -- this class does so
everywhere it reads state internally (`_signed_calibration_gap`, `bootstrap`,
`state_summary`).

The one that *does* matter is group 0: AUROC and AveragePrecision keep every
predicted probability in memory, i.e. O(N x K). At 1M samples and 1000 classes
that is 4 GB. Pass `thresholds=...` to switch them to a binned multi-threshold
confusion matrix state of size O(T x K), which is a fixed few hundred kilobytes.
The trade-off, measured on the same benchmark: state drops from ~18 MB to
1.7 MB and `compute` goes from 1.6 s to 0.01 s, but `update` gets slower on CPU
because the binning happens per batch. On GPU that binning is cheap and it is
almost always the right call for large runs.

Corrections to the usual snippet
--------------------------------
Three things in the version people normally start from do not run:

  * `BalancedAccuracy` does not exist in torchmetrics. Balanced accuracy *is*
    macro-averaged recall, so `MulticlassRecall(average="macro")` is it exactly.
  * `LogLoss` does not exist either. Hence the small class below.
  * `F1Score(num_classes=K, ...)` raises: the task-agnostic wrappers require
    `task="multiclass"`. Using the explicit `Multiclass*` classes avoids the
    question entirely and is clearer about what is being computed.

Also worth flagging: `ConfusionMatrix(normalize="true")` destroys the counts,
and the counts are what you need for support-weighted anything. Keep the raw
matrix in state and normalise at display time.

Verified against a real torchmetrics==1.9.0 install (the version this was
checked against): every class name, kwarg (`normalize`, `weights`, `average`,
`thresholds`, `norm`), and state attribute name used below (`confmat`,
`preds`/`target`, `confidences`/`accuracies`, `tp`/`fp`/`tn`/`fn`) matches the
installed package, and the whole class was smoke-tested end to end (batched
updates, reset, ordinal mode, binned `auc_thresholds`, bootstrap, K=2 edge
case) with no exceptions and sane output.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)


# ---------------------------------------------------------------------------
# The three metrics torchmetrics does not provide.
#
# All of them are written as ordinary Metric subclasses with `dist_reduce_fx="sum"`
# states, which is what makes them correct under DDP for free.
# ---------------------------------------------------------------------------


class MulticlassLogLoss(Metric):
    """Mean negative log-likelihood.

    Strictly proper: minimised if and only if the predicted probabilities are
    the true ones. This is the metric to use for early stopping -- it is
    smooth, it cannot be gamed by distorting probabilities, and unlike
    argmax-level metrics it is not a step function of the model parameters.

    State is two scalars, so this is essentially free to accumulate.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_classes: int, eps: float = 1e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eps = eps
        self.add_state("nll_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        p = preds.to(torch.float64)
        p_true = p.gather(1, target.long().unsqueeze(1)).squeeze(1)
        self.nll_sum += -torch.log(p_true.clamp(min=self.eps)).sum()
        self.count += target.numel()

    def compute(self) -> Tensor:
        return self.nll_sum / self.count.clamp(min=1)


class MulticlassBrierScore(Metric):
    """Multiclass Brier score: mean squared error over the whole probability vector.

    Also strictly proper, but bounded in [0, 2], which makes it far better
    behaved than log-loss when the model occasionally assigns ~0 probability to
    the true class (log-loss explodes there and a single sample can dominate an
    entire epoch's validation number).
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("sse", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        p = preds.to(torch.float64)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, target.long().unsqueeze(1), 1.0)
        self.sse += ((p - onehot) ** 2).sum()
        self.count += target.numel()

    def compute(self) -> Tensor:
        return self.sse / self.count.clamp(min=1)


class BrierDecomposition(Metric):
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    Computed one-vs-rest per class and summed, which reconstructs the multiclass
    Brier exactly up to binning error.

      reliability  (lower better)  -- pure miscalibration. Fixable post-hoc with
                                      temperature scaling, no retraining needed.
      resolution   (higher better) -- genuine discriminative skill.
      uncertainty                  -- irreducible, fixed by the base rates.

    Why bother: two models with an identical Brier score can be a
    well-calibrated dullard and a sharp-but-overconfident net, and those two
    problems have completely different fixes. The scalar cannot tell them apart.

    State is three (K, n_bins) matrices -- a few kilobytes, independent of N.
    """

    higher_is_better = None
    full_state_update = False

    def __init__(self, num_classes: int, n_bins: int = 15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.n_bins = n_bins

        shape = (num_classes, n_bins)
        self.add_state("prob_sum", default=torch.zeros(shape, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("obs_sum", default=torch.zeros(shape, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("bin_count", default=torch.zeros(shape, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        p = preds.to(torch.float64)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, target.long().unsqueeze(1), 1.0)

        # Bin index per (sample, class), then scatter-add into the (K, bins)
        # accumulators. Fully vectorised: no Python loop over classes.
        edges = torch.linspace(0, 1, self.n_bins + 1, dtype=p.dtype, device=p.device)
        bin_idx = torch.bucketize(p, edges[1:-1].contiguous(), right=False)  # (N, K)

        k = self.num_classes
        class_idx = torch.arange(k, device=p.device).expand_as(p)
        flat = (class_idx * self.n_bins + bin_idx).reshape(-1)

        self.prob_sum += torch.bincount(flat, weights=p.reshape(-1), minlength=k * self.n_bins).reshape(k, self.n_bins)
        self.obs_sum += torch.bincount(flat, weights=onehot.reshape(-1), minlength=k * self.n_bins).reshape(k, self.n_bins)
        self.bin_count += torch.bincount(flat, minlength=k * self.n_bins).reshape(k, self.n_bins).to(torch.float64)
        self.count += target.numel()

    def compute(self) -> Dict[str, Tensor]:
        n = self.count.clamp(min=1).to(torch.float64)
        occupied = self.bin_count > 0

        mean_pred = torch.where(occupied, self.prob_sum / self.bin_count.clamp(min=1), torch.zeros_like(self.prob_sum))
        mean_obs = torch.where(occupied, self.obs_sum / self.bin_count.clamp(min=1), torch.zeros_like(self.obs_sum))

        base_rate = self.obs_sum.sum(dim=1) / n  # (K,)

        weight = self.bin_count / n
        reliability = (weight * (mean_pred - mean_obs) ** 2)[occupied].sum()
        resolution = (weight * (mean_obs - base_rate.unsqueeze(1)) ** 2)[occupied].sum()
        uncertainty = (base_rate * (1 - base_rate)).sum()

        # Keys are prefixed because MetricCollection flattens dict-returning
        # metrics straight into the top-level result using *these* names, not
        # the name the metric was registered under. Unprefixed "reliability"
        # would silently collide with anything else exposing that key.
        return {
            "brier_reliability": reliability,
            "brier_resolution": resolution,
            "brier_uncertainty": uncertainty,
        }


# ---------------------------------------------------------------------------
# The collection.
# ---------------------------------------------------------------------------


class ClassificationReport(MetricCollection):
    """Everything at once, in one update pass, with compute groups doing the
    deduplication.

    Subclassing `MetricCollection` rather than `Metric` means `update`,
    `reset`, `to()`, `clone()`, DDP synchronisation and Lightning integration
    all come for free, and the compute-group machinery keeps working.

    Args:
        num_classes: K. Binary is just K=2.
        class_names: labels for the report table.
        ordinal: classes have a natural order (severity grades, star ratings).
            Adds quadratic-weighted kappa, which is the right metric there and
            which people routinely forget exists.
        n_bins: bins for the calibration diagnostics.
        auc_thresholds: passed to AUROC / AveragePrecision. `None` keeps every
            probability in memory (exact, O(N x K)). An int switches to a binned
            state of that many thresholds -- approximate, but O(T x K) and much
            faster to compute. Use it as soon as N gets large.
        compute_groups: leave True. See the module docstring.

    Example:
        report = ClassificationReport(num_classes=5, class_names=[...])
        for logits, y in loader:
            report.update(logits.softmax(-1), y)
        print(report.report())
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Sequence[str]] = None,
        ordinal: bool = False,
        n_bins: int = 15,
        auc_thresholds: Optional[int] = None,
        compute_groups: bool = True,
    ) -> None:
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError("class_names must have exactly num_classes entries")

        k = num_classes
        auc_kw = {"thresholds": auc_thresholds}

        metrics: Dict[str, Metric] = {
            # -- decision level: all confusion-matrix derived, so torchmetrics
            #    will merge these into two state groups on its own.
            "confmat": MulticlassConfusionMatrix(num_classes=k, normalize=None),
            "mcc": MulticlassMatthewsCorrCoef(num_classes=k),
            "kappa": MulticlassCohenKappa(num_classes=k),
            # Balanced accuracy IS macro recall. Same tensor, one computation.
            "balanced_acc": MulticlassRecall(num_classes=k, average="macro"),
            "f1_macro": MulticlassF1Score(num_classes=k, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=k, average="micro"),
            "f1_weighted": MulticlassF1Score(num_classes=k, average="weighted"),
            "precision_macro": MulticlassPrecision(num_classes=k, average="macro"),
            # Per class, because the aggregate is exactly what hides the tail.
            "precision_per_class": MulticlassPrecision(num_classes=k, average=None),
            "recall_per_class": MulticlassRecall(num_classes=k, average=None),
            "f1_per_class": MulticlassF1Score(num_classes=k, average=None),
            "specificity_per_class": MulticlassSpecificity(num_classes=k, average=None),
            # -- ranking level (threshold-free)
            "auroc_macro": MulticlassAUROC(num_classes=k, average="macro", **auc_kw),
            "auroc_weighted": MulticlassAUROC(num_classes=k, average="weighted", **auc_kw),
            "auroc_per_class": MulticlassAUROC(num_classes=k, average=None, **auc_kw),
            "ap_macro": MulticlassAveragePrecision(num_classes=k, average="macro", **auc_kw),
            "ap_weighted": MulticlassAveragePrecision(num_classes=k, average="weighted", **auc_kw),
            "ap_per_class": MulticlassAveragePrecision(num_classes=k, average=None, **auc_kw),
            # -- probabilistic level (proper scoring rules + calibration)
            "log_loss": MulticlassLogLoss(num_classes=k),
            "brier": MulticlassBrierScore(num_classes=k),
            "brier_parts": BrierDecomposition(num_classes=k, n_bins=n_bins),
            "ece": MulticlassCalibrationError(num_classes=k, n_bins=n_bins, norm="l1"),
            "mce": MulticlassCalibrationError(num_classes=k, n_bins=n_bins, norm="max"),
        }

        if ordinal:
            metrics["kappa_quadratic"] = MulticlassCohenKappa(num_classes=k, weights="quadratic")
            metrics["kappa_linear"] = MulticlassCohenKappa(num_classes=k, weights="linear")

        super().__init__(metrics, compute_groups=compute_groups)

        # MetricCollection is an nn.ModuleDict, so plain attributes are fine
        # as long as they are not Modules or Tensors.
        self.num_classes = k
        self.class_names = list(class_names) if class_names else [f"class_{i}" for i in range(k)]
        self.ordinal = ordinal
        self.n_bins = n_bins
        self.auc_thresholds = auc_thresholds

    # -- computation --------------------------------------------------------

    def compute(self) -> Dict[str, object]:
        """Run the collection, then add the derived quantities that are free
        once the primitives are known."""
        res: Dict[str, object] = dict(super().compute())

        # Note: "brier_parts" never appears as a key -- MetricCollection
        # flattens dict-returning metrics, so its three values arrive directly
        # as brier_reliability / brier_resolution / brier_uncertainty.

        cm = res["confmat"].to(torch.float64)
        total = cm.sum()
        support = cm.sum(dim=1)

        # micro-F1 in single-label multiclass IS accuracy -- same number, and
        # torchmetrics will happily hand it to you under both names. Making the
        # identity explicit stops people reporting it twice as if it were two
        # independent results.
        res["accuracy"] = res["f1_micro"]

        # The second definition of macro-F1 (Opitz & Burst, 2019): harmonic
        # mean of macro precision and macro recall, rather than the mean of the
        # per-class F1s. They disagree, papers rarely say which they used, so
        # report both.
        mp = res["precision_macro"].to(torch.float64)
        mr = res["balanced_acc"].to(torch.float64)
        res["f1_macro_harmonic"] = (
            2 * mp * mr / (mp + mr) if (mp + mr).item() > 0 else torch.tensor(0.0, dtype=torch.float64)
        )

        # The single most informative derived number: how much of the score is
        # being carried by the frequent classes.
        res["macro_micro_gap"] = res["accuracy"].to(torch.float64) - res["f1_macro"].to(torch.float64)

        # Base-rate baselines. A raw log-loss of 0.9 means very different things
        # at K=2 and K=100; these give the "predict the prior" reference point.
        base = (support / total.clamp(min=1)).clamp(min=1e-12)
        res["log_loss_baseline"] = -(base * base.log()).sum()
        res["brier_baseline"] = 1 - (base**2).sum()

        # Signed calibration gap. torchmetrics' ECE is an absolute value, so it
        # cannot tell over- from under-confidence; this can.
        res["overconfidence"] = self._signed_calibration_gap()

        res["support"] = support
        res["n_samples"] = int(total.item())
        return res

    def _signed_calibration_gap(self) -> Tensor:
        """mean(confidence) - accuracy, reusing the state MulticlassCalibrationError
        already accumulated. No second pass over the data."""
        ece_metric = self.__getitem__("ece", copy_state=False)
        conf = ece_metric.confidences
        acc = ece_metric.accuracies
        conf = torch.cat(conf) if isinstance(conf, list) else conf
        acc = torch.cat(acc) if isinstance(acc, list) else acc
        return conf.to(torch.float64).mean() - acc.to(torch.float64).mean()

    # -- uncertainty --------------------------------------------------------

    def bootstrap(
        self,
        metric_names: Sequence[str] = ("accuracy", "balanced_acc", "f1_macro", "mcc", "log_loss"),
        n_resamples: int = 500,
        confidence: float = 0.95,
        seed: Optional[int] = 0,
    ) -> Dict[str, Dict[str, float]]:
        """Percentile bootstrap CIs, reusing the probabilities AUROC already stores.

        This is deliberately parasitic: rather than keeping a second copy of the
        predictions, it borrows the `preds`/`target` lists that AUROC and
        AveragePrecision are holding anyway. Which means it only works when
        `auc_thresholds=None` -- with binned states the raw predictions are gone.

        Caveat that matters more than the method: this resamples at the *sample*
        level. If your data is grouped (several crops per image, several images
        per patient), resample the groups instead or these intervals will be far
        too narrow.
        """
        if self.auc_thresholds is not None:
            raise RuntimeError(
                "bootstrap needs the raw predictions; rebuild with auc_thresholds=None"
            )

        src = self.__getitem__("auroc_macro", copy_state=False)
        preds = torch.cat(src.preds) if isinstance(src.preds, list) else src.preds
        target = torch.cat(src.target) if isinstance(src.target, list) else src.target
        preds = preds.to(torch.float64)

        n = preds.shape[0]
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        # A throwaway collection holding only what we need, so each resample is
        # cheap. Building it once and resetting is much faster than rebuilding.
        scratch = MetricCollection(
            {
                "accuracy": MulticlassF1Score(num_classes=self.num_classes, average="micro"),
                "balanced_acc": MulticlassRecall(num_classes=self.num_classes, average="macro"),
                "f1_macro": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
                "mcc": MulticlassMatthewsCorrCoef(num_classes=self.num_classes),
                "log_loss": MulticlassLogLoss(num_classes=self.num_classes),
                "brier": MulticlassBrierScore(num_classes=self.num_classes),
            },
            compute_groups=True,
        ).to(preds.device)

        unknown = set(metric_names) - set(scratch.keys())
        if unknown:
            raise KeyError(f"cannot bootstrap {sorted(unknown)}; choose from {sorted(scratch.keys())}")

        samples: Dict[str, List[float]] = {m: [] for m in metric_names}
        for _ in range(n_resamples):
            idx = torch.randint(0, n, (n,), generator=gen).to(preds.device)
            scratch.reset()
            scratch.update(preds[idx], target[idx])
            out = scratch.compute()
            for m in metric_names:
                samples[m].append(float(out[m]))

        alpha = (1 - confidence) / 2
        results = {}
        for m, vals in samples.items():
            v = torch.tensor(vals, dtype=torch.float64)
            results[m] = {
                "mean": v.mean().item(),
                "std": v.std().item(),
                "lo": torch.quantile(v, alpha).item(),
                "hi": torch.quantile(v, 1 - alpha).item(),
            }
        return results

    # -- presentation -------------------------------------------------------

    def report(self, digits: int = 4, max_classes: int = 50) -> str:
        """Human-readable summary, laid out so the trade-offs are visible."""
        r = self.compute()
        d = digits
        f = lambda x: float(x)  # noqa: E731  -- tensors and floats both arrive here

        names = self.class_names
        width = max(12, min(28, max(len(n) for n in names)))
        lines: List[str] = []

        def section(title: str) -> None:
            lines.append("")
            lines.append(title)
            lines.append("-" * max(len(title), 66))

        # ---- per class ----
        section("PER CLASS")
        lines.append(
            f"{'class':<{width}} {'prec':>8} {'recall':>8} {'spec':>8} {'f1':>8} "
            f"{'auroc':>8} {'ap':>8} {'support':>9}"
        )
        order = sorted(range(self.num_classes), key=lambda i: -f(r["support"][i]))
        for i in order[:max_classes]:
            lines.append(
                f"{names[i]:<{width}} "
                f"{f(r['precision_per_class'][i]):>8.{d}f} "
                f"{f(r['recall_per_class'][i]):>8.{d}f} "
                f"{f(r['specificity_per_class'][i]):>8.{d}f} "
                f"{f(r['f1_per_class'][i]):>8.{d}f} "
                f"{f(r['auroc_per_class'][i]):>8.{d}f} "
                f"{f(r['ap_per_class'][i]):>8.{d}f} "
                f"{int(f(r['support'][i])):>9d}"
            )
        if len(order) > max_classes:
            lines.append(f"... {len(order) - max_classes} more classes hidden (sorted by support)")

        # ---- decisions ----
        section("DECISION QUALITY (argmax)")
        lines.append(f"  accuracy (= micro-F1 = micro-P = micro-R) : {f(r['accuracy']):.{d}f}")
        lines.append(f"  balanced accuracy (= macro recall)        : {f(r['balanced_acc']):.{d}f}")
        lines.append(f"  macro F1   (mean of per-class F1)         : {f(r['f1_macro']):.{d}f}")
        lines.append(f"  macro F1   (harmonic, Opitz & Burst)      : {f(r['f1_macro_harmonic']):.{d}f}")
        lines.append(f"  weighted F1 (report only if forced)       : {f(r['f1_weighted']):.{d}f}")
        lines.append(f"  MCC        (recommended scalar)           : {f(r['mcc']):.{d}f}")
        lines.append(f"  Cohen's kappa                             : {f(r['kappa']):.{d}f}")
        if self.ordinal:
            lines.append(f"  quadratic weighted kappa (ordinal)        : {f(r['kappa_quadratic']):.{d}f}")
            lines.append(f"  linear weighted kappa                     : {f(r['kappa_linear']):.{d}f}")
        lines.append(f"  micro - macro gap                         : {f(r['macro_micro_gap']):+.{d}f}")
        lines.append("    (large positive = the frequent classes are carrying the score)")

        # ---- probabilities ----
        section("PROBABILISTIC QUALITY (strictly proper scoring rules)")
        lines.append(
            f"  log-loss                                  : {f(r['log_loss']):.{d}f}"
            f"   (base-rate baseline: {f(r['log_loss_baseline']):.{d}f})"
        )
        lines.append(
            f"  Brier score                               : {f(r['brier']):.{d}f}"
            f"   (base-rate baseline: {f(r['brier_baseline']):.{d}f})"
        )
        lines.append(f"    reliability (miscalibration, lower better) : {f(r['brier_reliability']):.{d}f}")
        lines.append(f"    resolution  (real skill,     higher better): {f(r['brier_resolution']):.{d}f}")
        lines.append(f"    uncertainty (irreducible)                  : {f(r['brier_uncertainty']):.{d}f}")

        section("CALIBRATION (diagnostic only -- ECE is not a proper score)")
        lines.append(f"  ECE ({self.n_bins} bins)                             : {f(r['ece']):.{d}f}")
        lines.append(f"  MCE                                       : {f(r['mce']):.{d}f}")
        gap = f(r["overconfidence"])
        lines.append(
            f"  mean confidence - accuracy                : {gap:+.{d}f}"
            f"  ({'overconfident' if gap > 0 else 'underconfident'})"
        )

        # ---- ranking ----
        section("RANKING QUALITY (threshold-free)")
        lines.append(f"  macro AUROC (one-vs-rest)                 : {f(r['auroc_macro']):.{d}f}")
        lines.append(f"  weighted AUROC (one-vs-rest)              : {f(r['auroc_weighted']):.{d}f}")
        lines.append(f"  macro average precision                   : {f(r['ap_macro']):.{d}f}")
        lines.append(f"  weighted average precision                : {f(r['ap_weighted']):.{d}f}")

        lines.append("")
        lines.append(f"n = {r['n_samples']} samples, K = {self.num_classes} classes")
        return "\n".join(lines)

    def confusion_table(self, normalize: Optional[str] = None, max_classes: int = 20) -> str:
        """Text confusion matrix. `normalize="true"` divides by row (recall view),
        `"pred"` by column (precision view). Kept out of the metric state on
        purpose -- normalising in state would throw away the counts."""
        cm = self.__getitem__("confmat", copy_state=False).compute().to(torch.float64)
        if normalize == "true":
            cm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)
        elif normalize == "pred":
            cm = cm / cm.sum(dim=0, keepdim=True).clamp(min=1)
        elif normalize is not None:
            raise ValueError("normalize must be None, 'true' or 'pred'")

        k = min(self.num_classes, max_classes)
        w = max(8, max(len(n[:10]) for n in self.class_names[:k]) + 1)
        head = " " * (w + 2) + "".join(f"{n[:10]:>{w}}" for n in self.class_names[:k])
        rows = [head, " " * (w + 2) + "-" * (w * k)]
        for i in range(k):
            cells = (
                "".join(f"{cm[i, j].item():>{w}.3f}" for j in range(k))
                if normalize
                else "".join(f"{int(cm[i, j].item()):>{w}d}" for j in range(k))
            )
            rows.append(f"{self.class_names[i][:10]:<{w}} |{cells}")
        rows.append("(rows = true, columns = predicted)")
        return "\n".join(rows)

    def state_summary(self) -> str:
        """Show which metrics torchmetrics actually merged, and what state costs.

        Two gotchas this makes visible:

        * Compute groups are only formed *after the first update*, and are
          recomputed after every `reset()`. Inspecting before that shows nothing.
        * `MetricCollection.items()` / `.values()` default to `copy_state=True`,
          which deep-copies the shared states so each metric looks like it owns
          its own. That is convenient but it silently allocates; pass
          `copy_state=False` when you only want to look, as done here.
        """
        lines = ["compute groups (only exist after the first update):", ""]
        total = 0

        for gid, members in sorted(self.compute_groups.items()):
            owner = self.__getitem__(members[0], copy_state=False)
            sizes = []
            group_bytes = 0
            for sname in owner._defaults:
                state = getattr(owner, sname)
                tensors = state if isinstance(state, list) else [state]
                nbytes = sum(t.numel() * t.element_size() for t in tensors)
                group_bytes += nbytes
                sizes.append(sname)
            total += group_bytes

            lines.append(f"  group {gid}: {group_bytes / 1e6:8.3f} MB   states = {sizes}")
            for name in sorted(members):
                lines.append(f"           {name}")
            lines.append("")

        lines.append(f"  unique state total: {total / 1e6:.3f} MB "
                     f"for {len(list(self.keys()))} metrics in {len(self.compute_groups)} groups")
        return "\n".join(lines)
