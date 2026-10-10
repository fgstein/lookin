"""Chapter: the objects every other chapter is built on."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("foundations", "Foundations")

PAGES = (
    ManualPage(
        id="predictions",
        title="From scores to a prediction",
        summary=(
            "A results CSV gives one score per class per crop. The app turns that row of scores into "
            "one predicted class (the largest entry), one confidence (the value of that entry), and one "
            "boolean (was it right?). Almost every number in the app is a different way of summarising "
            "those three columns."
        ),
        shown_in="Built on load, used by every tab",
        blocks=(
            Heading("The three derived columns"),
            Para(
                "Each row $i$ of a results CSV is a score vector $p_i = (p_{i,1}, \\ldots, p_{i,C})$, one "
                "entry per class. The app derives three columns from it, and they are the ones the rest "
                "of the app reads:"
            ),
            Equation(
                r"\hat{y}_i = \arg\max_{c} \; p_{i,c}",
                where=((r"\hat{y}_i", "the predicted class: the class with the largest score"),),
                note="predicted_label -- what the model would answer if forced to answer.",
            ),
            Equation(
                r"\hat{p}_i = \max_{c} \; p_{i,c}",
                where=((r"\hat{p}_i", "the confidence: the value of that largest score"),),
                note="predicted_score -- how strongly it answered, on its own scale.",
            ),
            Equation(
                r"\mathrm{correct}_i = \mathbb{1}[\, \hat{y}_i = y_i \,]",
                where=((r"y_i", "the true class, read from the label column of the dataset CSV"),),
                note="correct -- the column accuracy and every selective metric are counted from.",
            ),
            Heading("What argmax throws away"),
            Para(
                "The argmax is a *decision*, and it is deliberately lossy. A crop scored "
                "$(0.51, 0.49)$ and a crop scored $(0.99, 0.01)$ produce the same predicted class and "
                "contribute identically to accuracy, precision, recall, F1 and the confusion matrix. "
                "Everything that distinguishes them lives in the confidence, which is why the app "
                "carries it as its own column and why three whole chapters of this manual are about "
                "what to do with it."
            ),
            Callout(CALLOUT_PITFALL, "Ties and degenerate score columns", (
                Para(
                    "With hard 0/1 scores (a rounded export, a one-hot head) many rows tie, and the "
                    "argmax silently picks the first such column. The confidence column is then 1.0 "
                    "everywhere, every calibration number becomes meaningless, and every abstention "
                    "threshold either keeps everything or nothing."
                ),
                Para(
                    "The app detects this family of problems when a results CSV is loaded and says so "
                    "in a banner -- see the page on score diagnostics."
                ),
            )),
            Heading("Why the scores are not automatically probabilities"),
            Para(
                "Nothing forces the columns of a results CSV to sum to 1: a sigmoid head, a set of "
                "one-vs-rest detectors or a raw logit dump are all legitimate exports, and all three "
                "produce rows that do not. The decision-level and ranking-level metrics do not care "
                "-- they only use the argmax and the order. The probabilistic ones (log loss, Brier, "
                "ECE) *do* care, and the app switches them off rather than printing a number that "
                "means nothing."
            ),
        ),
        see_also=("confusion-matrix", "score-diagnostics", "metric-levels", "calibration-error"),
        keywords=("argmax", "predicted label", "confidence", "predicted_score", "correct", "softmax"),
    ),

    ManualPage(
        id="confusion-matrix",
        title="The confusion matrix",
        summary=(
            "One cell per (true class, predicted class) pair, counting the crops that fell there. "
            "Every decision-level metric in the app -- accuracy, precision, recall, specificity, F1, "
            "MCC, kappa -- is a different arithmetic summary of this one table, so reading it directly "
            "tells you things no single number can."
        ),
        shown_in="Metrics tab, Compare tab",
        blocks=(
            Heading("Definition"),
            Equation(
                r"M_{a,b} = \sum_{i=1}^{N} \mathbb{1}[\, y_i = a \,] \cdot \mathbb{1}[\, \hat{y}_i = b \,]",
                where=(
                    (r"M_{a,b}", "crops whose true class is $a$ and whose predicted class is $b$"),
                    (r"a", "the row: the truth"),
                    (r"b", "the column: what the model said"),
                ),
                note="Rows are the truth, columns are the prediction. The diagonal is what went right.",
            ),
            Para(
                "A row of the matrix sums to the *support* of that class -- how many crops of it exist. "
                "A column sums to how many crops the model *called* that class. The two are equal only "
                "for a model that makes no mistakes."
            ),
            HowToRead((
                ("A diagonal cell", "correct predictions for that class"),
                ("An off-diagonal cell $M_{a,b}$", "crops of $a$ the model called $b$ -- a confusion, with a direction"),
                ("A hot row", "one true class scattered across many predictions: the model cannot recognise it"),
                ("A hot column", "one predicted class absorbing many true classes: the model over-uses it"),
                ("An asymmetric pair", "$M_{a,b} \\gg M_{b,a}$ means the confusion runs one way only, which is a labelling or a prior problem more often than a feature problem"),
            )),
            Heading("Per-class counts"),
            Para(
                "Fix one class $c$ and collapse everything else into \"not $c$\". The matrix becomes "
                "four numbers, and those four are what every per-class metric is built from:"
            ),
            Table(
                headers=("Count", "Definition", "In words"),
                rows=(
                    ("$\\mathrm{TP}_c$", "$M_{c,c}$", "crops of $c$, called $c$"),
                    ("$\\mathrm{FP}_c$", "$\\sum_{a \\neq c} M_{a,c}$", "crops of another class, called $c$ -- false alarms"),
                    ("$\\mathrm{FN}_c$", "$\\sum_{b \\neq c} M_{c,b}$", "crops of $c$, called something else -- misses"),
                    ("$\\mathrm{TN}_c$", "everything else", "correctly left alone"),
                ),
                weights=(1, 2, 3),
            ),
            Callout(CALLOUT_TIP, "Normalising a matrix", (
                Para(
                    "Raw counts make a large class dominate the picture. Dividing each row by its "
                    "support turns the matrix into per-class recall (the diagonal) and the rate at "
                    "which each class leaks into each other class (the rest), which is the version to "
                    "read when the classes are of very different sizes. Dividing by the *column* "
                    "totals instead gives per-class precision on the diagonal -- a different question, "
                    "so never read one as the other."
                ),
            )),
            Callout(CALLOUT_PITFALL, "A clean-looking matrix on an imbalanced dataset", (
                Para(
                    "With 95% of the crops in one class, a model that predicts that class everywhere "
                    "has a matrix that is 95% diagonal by mass, and an accuracy of 0.95. The rows for "
                    "the other classes are entirely off-diagonal and contain almost no crops, so they "
                    "are invisible unless the matrix is normalised by row."
                ),
            )),
            Heading("With many classes"),
            Para(
                "Up to 20 classes, the Metrics tab draws the whole matrix as a grid of clickable cells. "
                "Past that, a grid of every class stops being something anyone can read (two thousand "
                "classes is four million cells), and the tab shows the matrix three ways instead."
            ),
            HowToRead((
                ("The grid", "at most 20 classes -- the *focus set* -- as both rows and columns, the same set for every selected experiment so their grids line up"),
                ("The \"other\" column", "per row, the crops of that class predicted as any class outside the focus set, so a row still sums to the class's support"),
                ("Most confused pairs", "the largest off-diagonal cells of the whole matrix, largest first; *of true class* is the cell divided by its row's support"),
                ("The full matrix", "every class as one picture, errors only -- see below"),
            ), title="The three views"),
            Table(
                headers=("Focus set", "Which classes"),
                rows=(
                    ("Most confused pairs", "both classes of the largest off-diagonal cells, summed over the selected experiments, down the list until 20 are picked -- so each of those pairs is a cell of the grid"),
                    ("Worst F1", "lowest F1, averaged over the selected experiments; among equal F1 (typically 0) the larger class first"),
                    ("Largest support", "most crops, summed over the selected experiments"),
                    ("Hand-picked", "whatever was searched for and added; picking a class switches to this"),
                ),
                weights=(1, 3),
            ),
            Para(
                "The full-matrix picture colours only the off-diagonal cells, on a log scale where a "
                "single crop is already a clear blue: the diagonal is orders of magnitude larger and, "
                "drawn, would be all there is to see, so it is a thin grey line instead. The thumbnail "
                "sums blocks of neighbouring classes so that a lone confusion survives being shrunk; the "
                "enlarged view is one pixel per class and can be zoomed into. Its axes name only the "
                "focus classes."
            ),
            Heading("The same selection on every tab"),
            Para(
                "Everything else that draws one bar, row or cell per class works the same way past "
                "20 classes: a *Classes shown* control picks at most 20, by a criterion that fits the "
                "tab or by hand, and every per-class view on that tab shows those. The Metrics tab's "
                "selection covers its confusion grids and its per-class chart, and is also what an "
                "exported report shows per class."
            ),
            Table(
                headers=("Tab", "What follows the selection", "Criteria besides hand-picked"),
                rows=(
                    ("Metrics", "confusion grids, per-class chart, exported report", "most confused pairs, worst F1, largest support"),
                    ("Confidence", "per-class confidence chart", "least confident, highest entropy, largest support"),
                    ("Calibration", "per-class thresholds table", "lowest coverage, most predictions"),
                    ("Compare", "the matrix and the per-class table", "most changed cells, largest support"),
                    ("Embeddings", "class overlap grid", "most overlapping pairs, least separated, largest support"),
                    ("Prototypes", "role bars, case studies, per-class charts, both prototype grids", "worst margin, least pure prototypes, largest support"),
                ),
                weights=(1, 2, 2),
            ),
            Para(
                "Controls that pick a single class (the ROC/PR class, a relabel target) become "
                "type-to-filter lists instead, and the Gallery's class filters and the Embeddings "
                "tab's class restriction become a search field with no limit on how many are picked."
            ),
            Callout(CALLOUT_PITFALL, "Rankings and small classes", (
                Para(
                    "With thousands of classes, many have a handful of crops, and a class of three crops "
                    "the model always misses has an F1 of exactly 0 -- the same as a class of three "
                    "hundred. Left alone, such a ranking is a list of the smallest classes. So every "
                    "criterion that ranks on a per-class rate (worst F1, least confident, lowest "
                    "coverage, least separated, worst margin, least pure) only ranks classes with at "
                    "least N crops -- 10 unless changed in the field beside the criterion -- and breaks "
                    "ties by the larger support. A class under N can still be added by hand."
                ),
            )),
        ),
        see_also=("accuracy", "precision-recall", "support-and-imbalance", "delta-confusion"),
        keywords=("confusion", "matrix", "true positive", "false positive", "TP", "FP", "FN", "TN"),
    ),

    ManualPage(
        id="support-and-imbalance",
        title="Support, prevalence and imbalance",
        summary=(
            "Support is the number of crops that truly belong to a class; prevalence is that count as "
            "a share of the dataset. Both are properties of *your evaluation set*, not of the model -- "
            "and they silently set the scale of accuracy, of precision and of average precision, which "
            "is why a metric measured on one pool does not transfer to another."
        ),
        shown_in="Metrics tab (per-class table), Statistics tab, Slices panel",
        blocks=(
            Equation(
                r"n_c = \sum_{i=1}^{N} \mathbb{1}[\, y_i = c \,], \qquad \pi_c = \frac{n_c}{N}",
                where=(
                    (r"n_c", "the support of class $c$: how many crops truly are $c$"),
                    (r"\pi_c", "the prevalence of $c$: its share of the pool"),
                ),
            ),
            Heading("Why it is printed beside every per-class metric"),
            Para(
                "A recall of 0.60 computed on 5 crops and a recall of 0.60 computed on 5000 are the "
                "same number and not the same measurement. The first is three crops going one way "
                "instead of two; the second is a property of the model. Support is the column that "
                "tells the two apart, and it is why the app never shows a per-class table without it."
            ),
            Heading("What imbalance does to each metric"),
            Table(
                headers=("Metric", "Depends on prevalence?", "Consequence"),
                rows=(
                    ("Accuracy", "strongly", "a majority-class-only model scores $\\max_c \\pi_c$"),
                    ("Recall, specificity", "no", "computed within one class, so the mix does not enter"),
                    ("Precision", "yes", "the same detector is more precise on a pool where its class is common"),
                    ("ROC-AUC", "no", "built from TPR and FPR, both within-class rates"),
                    ("Average precision", "yes", "its floor is the prevalence $\\pi_c$ itself"),
                    ("Macro averages", "no, by construction", "every class counts once, however rare"),
                ),
                weights=(2, 2, 5),
            ),
            Callout(CALLOUT_PITFALL, "The number that moves when nothing changed", (
                Para(
                    "Filter the Data tab to a different split, or compare a validation pool against a "
                    "production sample, and the prevalence-dependent metrics move on their own. A "
                    "precision that fell from 0.9 to 0.6 between two pools may be the same detector "
                    "meeting a rarer class. Check the support column before reading it as a regression."
                ),
            )),
            Callout(CALLOUT_TIP, "Reporting under imbalance", (
                Bullets((
                    "Quote balanced accuracy or macro F1 next to accuracy, never accuracy alone.",
                    "Quote the per-class table, not just the average -- an average over 30 classes hides the three that fail.",
                    "For a rare class, prefer the precision-recall curve to the ROC curve.",
                    "State the prevalence of the pool alongside any precision figure, or it cannot be compared with anyone else's.",
                )),
            )),
        ),
        see_also=("confusion-matrix", "accuracy", "balanced-accuracy", "averaging", "pr-auc"),
        keywords=("support", "prevalence", "imbalance", "class balance", "rare class", "n"),
    ),

    ManualPage(
        id="averaging",
        title="Macro, weighted and micro averages",
        summary=(
            "A per-class metric has to be collapsed into one number somehow, and the three standard "
            "ways answer three different questions. Macro gives every class one vote; weighted gives "
            "every crop one vote; micro pools the counts before dividing. On a balanced dataset they "
            "agree; on an imbalanced one they can differ by tens of points."
        ),
        shown_in="Metrics tab (Summary table), ROC / PR curves, Slices panel",
        blocks=(
            Equation(
                r"\mathrm{macro} = \frac{1}{C} \sum_{c=1}^{C} m_c",
                where=((r"m_c", "the metric computed for class $c$ alone"),),
                note="Every class counts once, whether it holds 4 crops or 40000.",
            ),
            Equation(
                r"\mathrm{weighted} = \sum_{c=1}^{C} \frac{n_c}{N} \, m_c",
                note="Every class counts in proportion to its support, so a big class dominates.",
            ),
            Equation(
                r"\mathrm{micro} = \frac{\sum_c \mathrm{TP}_c}{\sum_c (\mathrm{TP}_c + \mathrm{FP}_c)}",
                note="Counts are pooled first, then divided once. In single-label classification this equals accuracy.",
            ),
            Heading("Which one to use"),
            Bullets((
                "*Macro* when every class matters equally -- a rare defect class is the point of the system, not a rounding error. This is the default in this app, and the one to quote when the rare classes are the interesting ones.",
                "*Weighted* when you want a number that describes the pool as it is: it answers \"how well does this do on a crop drawn at random from this dataset\".",
                "*Micro* when you want one global count-based rate and are content for it to be accuracy under another name.",
            )),
            Callout(CALLOUT_EXAMPLE, "The gap, on a small example", (
                Para(
                    "Two classes: A with 990 crops and F1 of 0.99, B with 10 crops and F1 of 0.20."
                ),
                Equation(r"\mathrm{macro\ F}_1 = \frac{0.99 + 0.20}{2} = 0.595"),
                Equation(r"\mathrm{weighted\ F}_1 = 0.99 \times \frac{990}{1000} + 0.20 \times \frac{10}{1000} = 0.982"),
                Para(
                    "Both are correct. The weighted figure says the system is excellent on a random "
                    "crop; the macro figure says one of the two classes does not work. Reporting only "
                    "the second number is how a broken rare class ships."
                ),
            )),
            Heading("Macro averaging of curves"),
            Para(
                "For ROC and precision-recall curves the app macro-averages the *curves*, not just "
                "their areas: every class's curve is interpolated onto a common axis and the curves "
                "are averaged point by point. The reported area is then the area under that averaged "
                "curve. A class whose score column carries no ranking at all is excluded rather than "
                "counted as 0.5, and the app names the classes it dropped under the chart."
            ),
            Callout(CALLOUT_PITFALL, "Macro averages and missing classes", (
                Para(
                    "A macro average over classes that are not all present in the pool is an average "
                    "over a different set of classes than you think. When a filter leaves a class with "
                    "no crops, that class contributes nothing and the divisor changes -- which makes "
                    "two macro F1 figures computed on two filtered views not directly comparable."
                ),
            )),
        ),
        see_also=("f1", "support-and-imbalance", "roc-auc", "pr-auc", "balanced-accuracy"),
        keywords=("macro", "micro", "weighted", "average", "averaging"),
    ),

    ManualPage(
        id="rates-and-uncertainty",
        title="How much to trust a rate",
        summary=(
            "Every rate in this app -- accuracy, recall, precision, a slice's score -- is a count "
            "divided by a count, and therefore carries sampling noise that shrinks like "
            "$1/\\sqrt{n}$. The app shows a 95% Wilson interval wherever the denominator is small "
            "enough for that to matter. Two numbers whose intervals overlap are not two results."
        ),
        shown_in="Slices panel, and implicitly everywhere a rate is shown",
        blocks=(
            Heading("The interval, and why this one"),
            Para(
                "For $k$ successes out of $n$, the naive interval $\\hat{p} \\pm z \\sqrt{\\hat{p}(1-\\hat{p})/n}$ "
                "breaks exactly where evaluation data lives: near 0 or 1, and at small $n$. At 29 "
                "correct out of 30 it returns an upper bound above 1. The app uses the Wilson score "
                "interval instead, which stays inside $[0,1]$ and degrades gracefully:"
            ),
            Equation(
                r"\mathrm{centre} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \qquad "
                r"\mathrm{half} = \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}",
                where=(
                    (r"\hat{p} = k/n", "the observed rate"),
                    (r"n", "the denominator -- the number of crops the rate was computed over"),
                    (r"z = 1.96", "the 95% normal quantile"),
                ),
                note="The interval is centre plus or minus half, clipped to [0, 1].",
            ),
            Heading("Orders of magnitude worth memorising"),
            Table(
                headers=("$n$", "Half-width near $\\hat{p}=0.9$", "Meaning"),
                rows=(
                    ("30", "about $\\pm 0.10$", "0.83 and 0.97 are the same measurement"),
                    ("100", "about $\\pm 0.06$", "a 5-point difference is noise"),
                    ("1000", "about $\\pm 0.02$", "a 2-point difference is borderline"),
                    ("10000", "about $\\pm 0.006$", "half a point is real"),
                ),
                weights=(1, 2, 4),
                caption="Indicative values; the exact width depends on $\\hat{p}$ as well as $n$.",
            ),
            Callout(CALLOUT_PITFALL, "The worst of twenty numbers", (
                Para(
                    "Slicing one metric into twenty subgroups and reading off the worst is a selection, "
                    "not a measurement: the minimum of twenty noisy numbers is low even when every "
                    "subgroup behaves identically. This is why the Slices panel can hold slices under "
                    "a minimum size out of the ranking, and why the interval is drawn on every bar."
                ),
            )),
            Callout(CALLOUT_TIP, "Comparing two models", (
                Para(
                    "The interval above is for *one* rate. Two models scored on the *same* crops are "
                    "much more comparable than their separate intervals suggest, because the shared "
                    "crops cancel: what matters then is the count of crops they disagree on, which is "
                    "exactly what the Compare tab counts as fixes and regressions."
                ),
            )),
        ),
        see_also=("slice-breakdown", "accuracy", "fixes-regressions"),
        keywords=("confidence interval", "wilson", "error bar", "uncertainty", "sample size", "noise"),
    ),
)
