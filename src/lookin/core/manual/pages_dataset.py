"""Chapter: what was loaded, before any model is judged."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("dataset", "The dataset itself")

PAGES = (
    ManualPage(
        id="dataset-join",
        title="How the dataset and the results are joined",
        summary=(
            "The dataset CSV has one row per crop with its true label; each results CSV has one row "
            "per crop with one score column per class. They are joined on the crop id, and everything "
            "in the app is computed on the *intersection* -- so the first number to check is how many "
            "rows survived that join."
        ),
        shown_in="Data tab, and the banner above every metric",
        blocks=(
            Heading("What can be loaded"),
            Table(
                headers=("File", "Holds", "Needed for"),
                rows=(
                    ("Dataset CSV", "one row per crop: an id, a path to the image, a label, and any number of metadata columns (numeric, binary or categorical)",
                     "everything; Statistics and the Gallery need nothing else"),
                    ("Results CSV (an *experiment*)", "one row per crop: the same id, plus one score column per class, named after the class itself",
                     "every metric, curve and confusion matrix"),
                    ("Embeddings sidecar (.npz)", "a $uuid$ array and an $embeddings$ array from the same model",
                     "the Embeddings tab, similarity search and the embedding cleaning strategies"),
                    ("Prototypes sidecar (.npz)", "the model's own class vectors",
                     "the Prototypes tab's scored diagnostics"),
                    ("Simple inferences", "crops that are *not* in the dataset, with an id, a path and an embeddings sidecar; no label",
                     "overlaying new acquisitions on the map"),
                ),
                weights=(2, 5, 4),
            ),
            Para(
                "The three roles in the dataset CSV (id, crop, label) are guessed from the file's own "
                "values and every guess is a dropdown of that file's columns, so a wrong one is one "
                "click to fix. Several results CSVs can be loaded at once: that is what makes the "
                "Metrics, Compare and consensus views possible."
            ),
            Heading("The join"),
            Bullets((
                "The id column identifies a crop. It must appear in both files and match exactly, as text.",
                "The label column, in the dataset CSV, carries the truth.",
                "The score columns, in the results CSV, are read positionally against the class list: the first score column is the first class, and so on.",
                "Rows present in only one of the two files are dropped from the merged view.",
            )),
            Callout(CALLOUT_TIP, "Crops with no ground truth", (
                Para(
                    "A simple inference carries no label, so it never enters a metric, a confusion "
                    "matrix, the cleaning queue or the review. Any other column it happens to carry "
                    "-- a confidence, a sensor, even class scores -- is kept and can colour the map, "
                    "and nothing more. Its embeddings must come from the same model as the "
                    "experiment it is overlaid on, or the two clouds are not in the same space."
                ),
            )),
            Callout(CALLOUT_WARNING, "The overlap banner", (
                Para(
                    "When the intersection is smaller than either file, the app says so above the "
                    "metrics. A low overlap is almost always one of four things: the wrong id column, "
                    "ids that differ by a path prefix or an extension, a results file from a different "
                    "split, or a dataset filter still active from an earlier question. All four "
                    "produce metrics computed on a silently different population."
                ),
            )),
            Heading("Filters and splits"),
            Para(
                "The row filter on the Data tab (for example *split = test*) applies to everything "
                "downstream: every metric, every curve, every slice. This is what makes it possible to "
                "evaluate on a held-out split, and also what makes two numbers incomparable when the "
                "filter changed between them. If a number moved and nothing else did, check the filter "
                "before anything else."
            ),
            Callout(CALLOUT_TIP, "Class labels come from your CSV", (
                Para(
                    "The class labels are read from the data and never translated, normalised or "
                    "re-ordered. Two files that spell the same class differently -- a capital letter, "
                    "a trailing space -- produce two classes."
                ),
            )),
        ),
        see_also=("predictions", "score-diagnostics", "column-statistics", "support-and-imbalance"),
        keywords=("join", "merge", "uuid", "overlap", "split", "filter", "id column"),
    ),

    ManualPage(
        id="score-diagnostics",
        title="Score diagnostics: are these probabilities?",
        summary=(
            "When a results CSV is loaded, the app checks the score columns for the two conditions "
            "that quietly break metrics: rows that do not sum to 1, and columns that never vary. "
            "Whatever fails, the metrics that depend on it are switched off rather than computed from "
            "numbers that do not support them."
        ),
        shown_in="The banner above the Metrics, Calibration and Compare tabs",
        blocks=(
            Heading("Check 1: does each row sum to 1?"),
            Para(
                "A probabilistic metric -- log loss, Brier, ECE, MCE, the reliability diagram, "
                "temperature scaling -- reads the scores as a distribution over the classes. A sigmoid "
                "head, a set of independent one-vs-rest detectors, or a raw logit dump are all "
                "legitimate exports whose rows do not sum to 1, and on those, those metrics have no "
                "meaning at all."
            ),
            Equation(
                r"\left| \sum_{c=1}^{C} p_{i,c} - 1 \right| \leq \varepsilon \quad \mathrm{for\ every\ row\ } i",
                where=((r"\varepsilon", "a tolerance of $10^{-3}$, generous because scores are stored as 32-bit floats"),),
                note="When this fails, the app disables the probability-level metrics and says so. The decision-level and ranking-level ones are unaffected.",
            ),
            Heading("Check 2: does each score column vary?"),
            Para(
                "A constant column carries no ranking. The motivating case is a threshold encoded as a "
                "fixed value -- a background column filled with 0.3, say, so that argmax reproduces "
                "\"predict background when the maximum score is below 0.3\". Every decision-level "
                "metric on such an export is perfectly correct, and every *ranking* metric on that "
                "column is a non-result: ROC-AUC comes back as exactly 0.5 and average precision as "
                "exactly the class's prevalence, whatever the model does."
            ),
            Para(
                "The app detects those columns, excludes them from the curves and from the macro "
                "averages, and names them under the chart rather than letting 0.5 be read as a "
                "measurement."
            ),
            Callout(CALLOUT_PITFALL, "A disabled metric is information", (
                Para(
                    "An empty calibration section does not mean the model is uncalibrated -- it means "
                    "the export does not contain probabilities. If you need those metrics, the fix is "
                    "upstream: export a softmax rather than logits, or a full distribution rather than "
                    "per-class sigmoids."
                ),
            )),
        ),
        see_also=("predictions", "log-loss", "calibration-error", "one-vs-rest", "roc-auc"),
        keywords=("diagnostics", "sums to one", "constant column", "sigmoid", "logits", "disabled"),
    ),

    ManualPage(
        id="column-statistics",
        title="Column statistics",
        summary=(
            "One chart per dataset column: the distribution of its values, and how the classes sit "
            "inside it. This is model-free -- it describes the data you are about to judge a model on, "
            "and most surprises in an evaluation are visible here first."
        ),
        shown_in="Statistics tab",
        blocks=(
            Heading("What is drawn"),
            Bullets((
                "*A categorical column* becomes a bar per value, ordered by count.",
                "*A numeric column* becomes a histogram.",
                "*The label column* becomes the class distribution -- the supports that set the scale of every metric in the app.",
                "Missing values are shown as their own category rather than dropped.",
            )),
            Heading("What to look for before reading any metric"),
            Table(
                headers=("Pattern", "What it usually means"),
                rows=(
                    ("A very long tail of rare classes", "macro averages will be dominated by classes with a handful of crops"),
                    ("One class holding most of the rows", "accuracy is nearly meaningless; read balanced accuracy and MCC"),
                    ("A metadata value holding almost nothing", "any slice on it will be noise -- see how much to trust a rate"),
                    ("A bimodal numeric column", "two populations are mixed; slicing on it will probably separate two different problems"),
                    ("A class present in the dataset but absent from the scores", "a class-mapping mismatch: the score columns are read positionally"),
                ),
                weights=(2, 5),
            ),
            Callout(CALLOUT_TIP, "The cheapest sanity check there is", (
                Para(
                    "Comparing the class distribution of your evaluation pool against the distribution "
                    "you expect in production tells you, before any model is involved, whether the "
                    "prevalence-dependent metrics (accuracy, precision, average precision) will "
                    "transfer at all."
                ),
            )),
        ),
        see_also=("support-and-imbalance", "slice-breakdown", "dataset-join", "rates-and-uncertainty"),
        keywords=("statistics", "histogram", "distribution", "column", "class balance"),
    ),
)
