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
                    ("split", "the column the Datasets page crosses the classes with as splits (default: a column named split)"),
                    ("source", "the column it crosses them with as sources (default: a column named source)"),
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
    ManualPage(
        id="dataset-class-balance",
        title="Class balance of a dataset",
        summary=(
            "How the rows of a dataset are shared out between its classes, summed up in three numbers: "
            "the ratio of the largest class to the smallest, the normalized entropy of the class "
            "distribution, and how many classes are too small to learn or measure."
        ),
        shown_in="Datasets page, Class balance",
        blocks=(
            Heading("The three numbers"),
            Equation(r"\mathrm{imbalance} = \frac{n_{\max}}{n_{\min}}", where=(
                (r"n_{\max}", "the number of rows of the largest class"),
                (r"n_{\min}", "the number of rows of the smallest class"),
            )),
            Para(
                "1 means every class has the same number of rows. It only looks at the two extremes: "
                "one tiny class is enough to make it large, which is often exactly what needs seeing. "
                "The page flags it from 100 up."
            ),
            Equation(r"H_{\mathrm{norm}} = -\frac{1}{\ln K} \sum_{k=1}^{K} p_k \ln p_k", where=(
                ("K", "the number of classes"),
                ("p_k", "the share of the labelled rows that belong to class $k$"),
            )),
            Para(
                "The normalized entropy looks at every class at once: 1 for a perfectly even split, "
                "towards 0 as a few classes take everything. Two datasets with the same imbalance ratio "
                "can have very different entropies -- one small outlier class barely moves it, a long "
                "tail of small classes does."
            ),
            Para(
                "Rare classes are those with fewer than 10 rows: too few to train on, and too few for "
                "any per-class metric measured on them to mean much (see how much to trust a rate)."
            ),
            HowToRead(items=(
                ("Bars", "one per class, largest first; the length is the class's number of rows"),
                ("Share", "the class's part of the labelled rows"),
                ("Missing labels", "rows with an empty label cell; they belong to no class"),
            )),
            Callout(CALLOUT_PITFALL, "Only an empty cell is a missing label", (
                Para("A class called None or NA is a class: it is counted, not dropped."),
            )),
        ),
        see_also=("support-and-imbalance", "dataset-groups", "rates-and-uncertainty"),
        keywords=("imbalance", "entropy", "rare classes", "distribution", "balance"),
    ),
    ManualPage(
        id="dataset-groups",
        title="Classes across sources and splits",
        summary=(
            "Where each class's rows come from (the source column) and which split they are in (the "
            "split column), as a table with one row per class. It shows a class that exists in only one "
            "split, or that comes almost entirely from one source -- both make a model look better or "
            "worse than it is."
        ),
        shown_in="Datasets page, Sources, Classes by source, Classes by split",
        blocks=(
            Heading("Which columns"),
            Para(
                "The source and split columns are named in the lookin.toml beside the CSV (keys source "
                "and split, see how the dataset and the results are joined). Without it, a column named "
                "source (or origin) and a column named split (or subset, fold) are used."
            ),
            HowToRead(items=(
                ("A row", "one class, largest first"),
                ("A column", "one value of the source (or split) column, largest first; past 30 values the rest are summed into (other)"),
                ("A cell's shade", "the share of the class's rows that have that value: the row sums to 100%"),
                ("A cell's number", "that share; hover for the number of rows"),
                ("(empty)", "rows whose source (or split) cell is empty"),
            )),
            Heading("What is flagged"),
            Bullets((
                "A class with no row at all in one of the splits: it cannot be learned (absent from train) or measured (absent from test) there.",
                "A class with 10 rows or more of which 95% or more come from one source, in a dataset with several sources: a model can learn to recognise the source instead of the class.",
            )),
        ),
        see_also=("dataset-class-balance", "slice-breakdown"),
        keywords=("source", "split", "cross table", "heatmap", "leakage", "provenance"),
    ),
    ManualPage(
        id="dataset-quality",
        title="Dataset checks",
        summary=(
            "What is checked in every dataset CSV, without opening a single image: ids that repeat, "
            "rows without a label, crops that two rows share, exact duplicate rows, and whether the crop "
            "paths actually exist."
        ),
        shown_in="Datasets page, Checks",
        blocks=(
            Table(
                headers=("Check", "Severity", "Why it matters"),
                rows=(
                    ("No label column found", "error", "nothing can be counted per class; name it in lookin.toml"),
                    ("Crop paths that do not exist", "error", "checked on 200 rows spread over the file, relative to the root in lookin.toml or to the CSV's folder"),
                    ("Repeated ids", "error", "the join with the results matches one id to several rows"),
                    ("Rows without a label", "warning", "left out of every metric"),
                    ("Two rows on one crop", "warning", "the same image counted twice, possibly under two labels"),
                    ("Exact duplicate rows", "warning", "usually a concatenation done twice"),
                    ("Rows without a crop path", "warning", "nothing to show in the Gallery"),
                    ("Large imbalance, rare classes", "warning, info", "see class balance of a dataset"),
                    ("Class absent from a split, single-source classes", "warning, info", "see classes across sources and splits"),
                ),
                weights=(3, 1, 5),
            ),
            Para(
                "The dot beside each dataset in the list is the worst of its checks: red for an error, "
                "amber for a warning, green when nothing is flagged."
            ),
        ),
        see_also=("dataset-class-balance", "dataset-groups", "dataset-join"),
        keywords=("duplicates", "missing", "checks", "quality", "health"),
    ),
    ManualPage(
        id="dataset-mlflow-link",
        title="Where a dataset comes from",
        summary=(
            "The shared folders hold copies of datasets logged to MLflow, one run per dataset with the "
            "CSV and a hash.txt holding the hash of its content. A CSV is traced back to its run by that "
            "hash; when no hash matches, a run whose CSV has the same name and size is shown as probable."
        ),
        shown_in="Datasets page, the MLflow column and the run of a dataset",
        blocks=(
            Heading("The content hash"),
            Para(
                "The hash is computed on the table, not on the file: the columns are sorted by name, each "
                "row is hashed, the row hashes are sorted, and SHA-256 is taken over the column names and "
                "the sorted row hashes. Two CSVs holding the same rows in another order therefore have "
                "the same hash. Lookin reads the CSV with pandas' default options to compute it, as the "
                "dataset scripts do."
            ),
            Table(
                headers=("Link", "Meaning"),
                rows=(
                    ("confirmed", "the file's content hash is the hash.txt of a run"),
                    ("probable", "no hash matches, but a run's CSV has the same file name and the same size in bytes"),
                    ("derived", "the file sits in the folder named after a main dataset and has no run of its own; the run shown is its main dataset's"),
                    ("none", "no run of the datasets experiments matches"),
                ),
                weights=(1, 5),
            ),
            Para(
                "The datasets experiments are set in the page's settings. Their runs are listed and each "
                "hash.txt downloaded when Refresh is pressed; the listing is kept, so the links still "
                "show when the server cannot be reached."
            ),
            Callout(CALLOUT_WARNING, "A probable link is a guess", (
                Para(
                    "The same name and size can be a different file. If many datasets stay probable "
                    "after their hash is computed, the hash in hash.txt is probably computed differently "
                    "from Lookin's."
                ),
            )),
        ),
        see_also=("dataset-comparison",),
        keywords=("mlflow", "hash", "provenance", "run", "derived", "lineage"),
    ),
    ManualPage(
        id="dataset-comparison",
        title="Comparing two datasets",
        summary=(
            "Any two datasets side by side: their class (and source, and split) distributions, the "
            "classes only one of them has, and -- through their id columns -- how many rows they share "
            "and which labels changed on the shared rows."
        ),
        shown_in="Datasets page, comparison (Ctrl+click a second dataset)",
        blocks=(
            Heading("How different are the distributions"),
            Equation(r"\mathrm{JS}(P, Q) = \frac{1}{2} \mathrm{KL}(P \,\|\, M) + \frac{1}{2} \mathrm{KL}(Q \,\|\, M)", where=(
                ("P, Q", "the class distributions of the two datasets (shares of their labelled rows)"),
                ("M", "their average, $(P + Q) / 2$"),
            )),
            Equation(r"\mathrm{KL}(P \,\|\, M) = \sum_k p_k \log_2 \frac{p_k}{m_k}", where=()),
            Para(
                "The Jensen-Shannon divergence is 0 for identical distributions and 1 for distributions "
                "with no class in common. Unlike the KL divergence it is symmetric and stays finite when "
                "a class exists on one side only. Below 0.01 the two distributions are practically the "
                "same; a derivative that removes one class out of ten typically lands around 0.05."
            ),
            HowToRead(items=(
                ("A", "the dataset selected first; B the one Ctrl+clicked"),
                ("Share A, share B", "the class's part of each dataset's labelled rows"),
                ("Change", "share B minus share A, in percentage points"),
                ("Rows by id", "rows whose id is in both, in A only, in B only; then the labels that differ between A and B on the shared ids, most frequent first"),
            )),
        ),
        see_also=("dataset-class-balance", "dataset-mlflow-link"),
        keywords=("compare", "diff", "divergence", "jensen-shannon", "overlap", "versions"),
    ),
)
