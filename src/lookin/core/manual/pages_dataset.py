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
        shown_in="Sources tab, and the banner above every metric",
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
            Heading("Saying it once: lookin.toml"),
            Para(
                "A file named lookin.toml in the folder of the dataset CSVs fills the Columns and the "
                "Root in whenever one of them is picked, instead of the guess. It holds one table per "
                "CSV, titled with its file name in quotes, for instance [\"train.csv\"], or with a "
                "pattern such as [\"*_test.csv\"] (an exact name wins over a pattern). An optional "
                "[defaults] table applies to every CSV of the folder; a CSV's own table overrides it, "
                "key by key."
            ),
            Table(
                headers=("Key", "What it sets"),
                rows=(
                    ("id", "the id column; an empty text (id = \"\") says the file has none"),
                    ("image", "the column holding the path to each crop"),
                    ("label", "the column holding the true class"),
                    ("root", "the folder those paths are relative to; itself relative to lookin.toml's folder unless absolute"),
                ),
                weights=(1, 5),
            ),
            Para(
                "Opening a workspace keeps the columns it was saved with; lookin.toml only fills the "
                "ones still at their default. A column it names that is not in the file is ignored, "
                "with a warning under the columns, and guessed instead."
            ),
            Heading("The join"),
            Bullets((
                "The id column identifies a crop. It must appear in both files and match exactly, as text.",
                "The label column, in the dataset CSV, carries the truth.",
                "The score columns, in the results CSV, are read positionally against the class list: the first score column is the first class, and so on.",
                "Rows present in only one of the two files are dropped from the merged view.",
                "A dataset row with an empty label has no truth to be scored against, so it is left out of the merged view too -- of every metric, slice and map. Only an empty cell is missing: a class called None or NA is a class.",
                "An empty score cell is never the prediction: a crop is predicted as its highest score that exists, and a crop with no score at all has no prediction (and counts as wrong). A row with a missing score is not a probability distribution, so log loss, Brier and ECE are disabled.",
                "A dataset column named like a class is left out of the merged view: that class's scores take its place.",
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
                "The row filter on the Sources tab (for example *split = test*) applies to everything "
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
        keywords=("join", "merge", "uuid", "overlap", "split", "filter", "id column", "toml", "lookin.toml", "root"),
    ),

    ManualPage(
        id="dataset-filter",
        title="Filtering the dataset",
        summary=(
            "The Filter tab removes rows from the loaded dataset. Every other tab -- statistics, "
            "gallery, metrics, curves, embeddings, cleaning -- then works on the rows kept, exactly as "
            "if the file had only those rows. The file itself is never modified."
        ),
        shown_in="Filter tab, and the row count in the bar above every tab while a filter is applied",
        blocks=(
            Heading("Conditions and the query"),
            Para(
                "A filter is a list of conditions on columns, plus an optional query. A row is kept "
                "when it satisfies every condition that is on, and the query. A condition that is "
                "turned off, or not filled in yet (no value picked, no bound typed), keeps every row."
            ),
            Table(
                headers=("Condition", "Keeps a row when"),
                rows=(
                    ("is one of / is not one of", "its value, read as text, is (is not) one of the values picked. A row with no value is never one of them."),
                    ("is between", "its value, read as a number, is at least the lower bound and at most the upper one; either bound can be left empty. A row with no value, or a value that is not a number, is removed."),
                    ("contains / does not contain", "its text contains (does not contain) what was typed, ignoring case."),
                    ("matches regex", "a regular expression matches somewhere in its text (Python syntax, case sensitive)."),
                    ("is empty / is not empty", "it has no value (a blank cell counts as empty)."),
                ),
                weights=(2, 6),
            ),
            Para(
                "Values are matched as text, and a whole number is written without its decimals: a "
                "column that reads 3.0 because one of its cells is empty still matches a 3 picked "
                "earlier. A CSV has no types, and the same column can read as numbers in one version "
                "of a file and as text in the next."
            ),
            Heading("Draft, then apply"),
            Para(
                "Everything changed in the tab is a draft: the tab counts the rows it keeps, says how "
                "many each condition removes on its own, and previews the rows kept and removed -- "
                "but no other tab moves until Apply. Applying changes the population every number is "
                "computed on, so every other tab then shows its out-of-date banner."
            ),
            Callout(CALLOUT_WARNING, "A filtered metric is a different metric", (
                Para(
                    "While a filter is applied, the bar above every tab shows how many rows are kept. "
                    "Two numbers computed under two different filters are not comparable, and a model "
                    "that looks better on a filtered pool may only be looking at easier crops."
                ),
            )),
            Heading("Where it is kept"),
            Bullets((
                "In the workspace, with the dataset: opening the workspace applies it again, and saving it records the filter applied at that moment.",
                "In each snapshot: reopening one puts back the filter it was taken under, including no filter at all.",
                "Reloading the same CSV keeps the filter; loading a different file drops it, since its columns may not exist there.",
                "A filter that can no longer be applied (a column it names is gone) is not applied in part: every row stays in view and the bar says the filter was not applied.",
            )),
            Callout(CALLOUT_TIP, "Two filters, two places", (
                Para(
                    "The row filter on the Sources tab (keep only rows where a column equals a value) "
                    "is applied while the file is read: rows it drops are never loaded. The Filter "
                    "tab works on what was loaded, and can be changed without reading the file again."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Experiments are not filtered, they are joined", (
                Para(
                    "Results files are never filtered themselves: they are joined with the rows kept. "
                    "A result whose row was removed by the filter is not counted as a result with no "
                    "matching dataset row in the overlap banner -- it has one, out of view."
                ),
            )),
        ),
        see_also=("dataset-filter-query", "dataset-join", "column-statistics"),
        keywords=("filter", "remove rows", "subset", "condition", "where", "keep rows"),
    ),

    ManualPage(
        id="dataset-filter-query",
        title="Writing a filter query",
        summary=(
            "A query is a formula over the dataset's columns that is true for the rows to keep, "
            "written like a pandas query: comparisons, and / or / not, lists, and a few functions. "
            "It is checked as it is typed, and Enter applies the filter."
        ),
        shown_in="Filter tab, Query section",
        blocks=(
            Heading("What can be written"),
            Table(
                headers=("Write", "Keeps the rows where"),
                rows=(
                    ("width >= 32", "width is at least 32"),
                    ("label == 'cat'", "the label is cat (text goes in quotes, single or double)"),
                    ("label != 'cat' and width < 10", "both hold; 'or' and 'not' work the same way, and parentheses group"),
                    ("label in ['cat', 'dog']", "the label is one of the list; 'not in' for the opposite"),
                    ("`object size` > 0.5", "a column whose name has spaces or symbols goes between backticks"),
                    ("col('object size') > 0.5", "the same, written as a function -- F('object size') also works"),
                    ("width.between(32, 64)", "32 <= width <= 64 -- which can also be written as is"),
                    ("crop.str.contains('blur')", "the text contains blur (a regular expression; case=False to ignore case)"),
                    ("'blur' in crop", "the same, as a plain substring"),
                    ("crop.str.startswith('cam2/')", "the text starts with cam2/; also endswith, match, fullmatch"),
                    ("annotator.isna()", "the column has no value; notna() for the opposite"),
                    ("abs(score - 0.5) < 0.1", "arithmetic between columns and numbers: + - / % and powers"),
                    ("len(label) > 3", "functions: len, abs, lower, upper, num, isna, notna, contains, matches, startswith, endswith, between"),
                ),
                weights=(4, 6),
            ),
            Para(
                "Symbols work too: & for and, | for or, ~ for not. They come from pandas, where each "
                "comparison then has to be in parentheses: (width > 3) & (height > 3)."
            ),
            Heading("Numbers and text"),
            Bullets((
                "A number compared with a text column reads that column as numbers: a cell that is not a number does not match.",
                "A text compared with a numeric column: equality compares the text of the values, an ordering needs the text to be a number.",
                "A cell with no value matches no comparison, except != and 'is None'.",
            )),
            Callout(CALLOUT_WARNING, "Only a formula", (
                Para(
                    "The query is not run as Python: it is read and checked piece by piece, and "
                    "anything that is not in the list above (an attribute, an import, an assignment, "
                    "a function not listed) is refused with a message saying what to write instead."
                ),
            )),
        ),
        see_also=("dataset-filter",),
        keywords=("query", "formula", "expression", "pandas", "fiftyone", "regex", "syntax"),
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
