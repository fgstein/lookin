"""Chapter: what the order of the scores is worth, independent of any threshold."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("ranking", "Ranking and curves")

PAGES = (
    ManualPage(
        id="one-vs-rest",
        title="One-vs-rest, and how a curve is built",
        summary=(
            "ROC and precision-recall curves are defined for a yes/no question, so a multiclass "
            "problem is turned into $C$ separate yes/no questions: for each class, is this crop that "
            "class or not, ranked by that class's score column. Everything on the two curve pages "
            "rests on this reduction."
        ),
        shown_in="Metrics tab (ROC & PR curves), class dropdown",
        blocks=(
            Heading("The reduction"),
            Para(
                "Pick a target class $c$. Relabel every crop as positive if $y_i = c$ and negative "
                "otherwise, and score each crop by $p_{i,c}$ -- the column of the results CSV for that "
                "class, *not* the confidence and *not* the argmax."
            ),
            Equation(
                r"y_i^{(c)} = \mathbb{1}[\, y_i = c \,], \qquad s_i^{(c)} = p_{i,c}",
                where=(
                    (r"y_i^{(c)}", "the binary truth for the target class"),
                    (r"s_i^{(c)}", "the score the curve ranks by"),
                ),
            ),
            Heading("Sweeping the threshold"),
            Para(
                "Sort the crops by $s^{(c)}$, highest first, and walk down the list. At each position "
                "you have accepted everything above it as positive: count the true and false positives "
                "so far, and you have one point of the curve. The whole curve is that walk, which is "
                "why it shows every possible threshold at once and needs none to be chosen."
            ),
            Bullets((
                "Each step down the ranking either adds a true positive (the curve moves up on a ROC plot) or a false positive (it moves right).",
                "A perfect ranking puts every positive above every negative, so the curve goes straight up and then straight across.",
                "A random ranking mixes them evenly, so the curve follows the diagonal.",
            ), numbered=True),
            Heading("What the app does with C curves"),
            Para(
                "The class dropdown above the charts selects one target class, or the macro average. "
                "For the macro average, every class's curve is interpolated onto a common axis and "
                "averaged point by point, and the reported area is the area under that averaged curve."
            ),
            Callout(CALLOUT_WARNING, "A class whose score column does not vary", (
                Para(
                    "If a class's score column is constant -- a threshold encoded as a fixed value, a "
                    "detector that emits no score for that class -- there is no ranking to evaluate. "
                    "ROC-AUC would come out at exactly 0.5 and average precision at exactly the "
                    "prevalence, both of which look like results and are not. The app detects this, "
                    "refuses to draw that curve, and names the excluded classes under the chart."
                ),
            )),
            Callout(CALLOUT_PITFALL, "The curve does not describe the deployed system", (
                Para(
                    "A one-vs-rest curve ranks by one class's score in isolation. The deployed model "
                    "takes an argmax across all $C$ columns, which is a different rule. A class can "
                    "have an excellent ROC-AUC and still never be predicted, because another class's "
                    "column is always higher. When the two disagree, the confusion matrix is the one "
                    "describing what ships."
                ),
            )),
        ),
        see_also=("roc-auc", "pr-auc", "score-diagnostics", "averaging"),
        keywords=("one vs rest", "ovr", "binarize", "target class", "curve"),
    ),

    ManualPage(
        id="roc-auc",
        title="ROC curve and ROC-AUC",
        summary=(
            "The ROC curve plots the true positive rate against the false positive rate as the "
            "threshold sweeps from strict to permissive. Its area (AUC) has an exact meaning: the "
            "probability that a randomly chosen positive is scored above a randomly chosen negative. "
            "It is insensitive to class imbalance, which is both its strength and its trap."
        ),
        shown_in="Metrics tab (ROC & PR curves, Summary columns \"Macro ROC-AUC\" and \"Weighted ROC-AUC\")",
        blocks=(
            Equation(
                r"\mathrm{TPR}(t) = \frac{\mathrm{TP}(t)}{\mathrm{TP}(t) + \mathrm{FN}(t)}, \qquad "
                r"\mathrm{FPR}(t) = \frac{\mathrm{FP}(t)}{\mathrm{FP}(t) + \mathrm{TN}(t)}",
                where=(
                    (r"t", "the score threshold being swept"),
                    (r"\mathrm{TPR}", "true positive rate -- the recall of the target class"),
                    (r"\mathrm{FPR}", "false positive rate -- one minus the specificity"),
                ),
                note="The curve is the set of points (FPR(t), TPR(t)) for every t.",
            ),
            Heading("What the area means"),
            Equation(
                r"\mathrm{AUC} = \int_0^1 \mathrm{TPR} \; d(\mathrm{FPR}) "
                r"= \Pr\!\left[\, s(X^{+}) > s(X^{-}) \,\right]",
                where=(
                    (r"X^{+}", "a positive crop drawn at random"),
                    (r"X^{-}", "a negative crop drawn at random"),
                ),
                note="AUC is exactly the probability that the model ranks a random positive above a random negative.",
            ),
            Para(
                "That identity is worth holding on to: AUC is a statement about *ordering pairs*, not "
                "about any decision the model makes. It is also the Mann-Whitney U statistic, "
                "normalised."
            ),
            HowToRead((
                ("The diagonal", "a model with no ranking ability at all: $\\mathrm{AUC} = 0.5$"),
                ("The top-left corner", "the perfect ranking: every positive above every negative, $\\mathrm{AUC} = 1$"),
                ("A curve below the diagonal", "the score is anti-correlated -- usually an inverted sign or a swapped column"),
                ("The steep first part", "how many positives you get before the first false alarms, which is the region a strict threshold operates in"),
                ("Two curves crossing", "neither model dominates: which is better depends on the operating point you will actually use"),
            )),
            Table(
                headers=("AUC", "Reading"),
                rows=(
                    ("0.5", "no information in the ranking"),
                    ("0.7", "weak: a positive outranks a negative 7 times out of 10"),
                    ("0.9", "strong"),
                    ("0.99", "check for leakage before celebrating -- see the near-duplicates page"),
                ),
                weights=(1, 5),
            ),
            Callout(CALLOUT_PITFALL, "The imbalance trap", (
                Para(
                    "FPR has *all* the negatives in its denominator. When the negatives outnumber the "
                    "positives 100 to 1, a thousand false alarms move FPR by a hundredth, so the curve "
                    "still looks excellent while the precision of the detector is catastrophic. This is "
                    "the single most common way a rare-class model is oversold."
                ),
                Para(
                    "For a rare target class, read the precision-recall curve instead: its denominator "
                    "is what you actually raise, not what exists."
                ),
            )),
            Callout(CALLOUT_TIP, "AUC does not pick a threshold", (
                Para(
                    "A high AUC says a good threshold *exists*. It does not say where it is, and it "
                    "says nothing about whether the score values are believable. The Calibration tab "
                    "answers both questions; AUC answers neither."
                ),
            )),
        ),
        see_also=("pr-auc", "one-vs-rest", "precision-recall", "calibration-error", "risk-coverage"),
        keywords=("roc", "auc", "tpr", "fpr", "area under curve", "sensitivity"),
    ),

    ManualPage(
        id="pr-auc",
        title="Precision-Recall curve and average precision",
        summary=(
            "The precision-recall curve plots precision against recall as the threshold sweeps. Its "
            "summary, average precision (AP), is the one ranking metric that stays honest when the "
            "target class is rare -- because its denominator is what the model raises, not what "
            "exists."
        ),
        shown_in="Metrics tab (ROC & PR curves, Summary columns \"Macro AP\" and \"Weighted AP\")",
        blocks=(
            Heading("The curve"),
            Para(
                "At each threshold $t$, plot the point $(\\mathrm{recall}(t), \\mathrm{precision}(t))$. "
                "Walking from a strict threshold to a permissive one moves right (recall can only "
                "grow) while precision moves up and down as true and false positives arrive."
            ),
            Heading("Average precision"),
            Equation(
                r"\mathrm{AP} = \sum_{k} \left( R_k - R_{k-1} \right) P_k",
                where=(
                    (r"P_k", "precision at the $k$-th threshold"),
                    (r"R_k", "recall at the $k$-th threshold"),
                ),
                note="A step-wise area under the curve: precision averaged over the recalls actually reached. No interpolation, which is what distinguishes AP from the older 11-point interpolated average precision.",
            ),
            Heading("The baseline moves"),
            Para(
                "A model with no ranking ability has $\\mathrm{AP} = \\pi_c$, the prevalence of the "
                "target class. There is no fixed \"0.5 is chance\" line as there is for ROC: an AP of "
                "0.30 is excellent for a class with 2% prevalence and terrible for a class with 50%. "
                "*Always quote the prevalence next to an AP.*"
            ),
            Equation(
                r"\mathrm{AP}_{\mathrm{trivial}} = \pi_c = \frac{n_c}{N}",
                note="The horizontal line a PR curve for a useless model follows.",
            ),
            HowToRead((
                ("The left end", "precision at the very top of the ranking -- the crops the model is most sure about"),
                ("A cliff near the left", "the top of the ranking is already polluted: no strict threshold will give you a clean set"),
                ("The right end", "precision at full recall, which tends to the prevalence"),
                ("Area under the curve", "average precision, comparable only between models scored on the same pool"),
            )),
            Callout(CALLOUT_EXAMPLE, "ROC and PR on the same detector", (
                Para(
                    "1000 crops, 20 positives. A detector ranks 15 positives in its top 100. "
                    "TPR is 0.75 and FPR is $85/980 \\approx 0.087$: the ROC point looks excellent. "
                    "Precision at that same threshold is $15/100 = 0.15$: five out of six alarms are "
                    "wrong. Same model, same threshold, two very different pictures -- and the second "
                    "is the one the person triaging the alarms will experience."
                ),
            )),
            Callout(CALLOUT_TIP, "Which curve to report", (
                Bullets((
                    "Positives are rare, or false alarms are the cost: precision-recall.",
                    "Both classes are common, and you want a metric that survives a change of class mix: ROC.",
                    "Comparing across pools with different prevalences: ROC-AUC, because AP shifts with the prevalence even for an identical model.",
                )),
            )),
        ),
        see_also=("roc-auc", "precision-recall", "one-vs-rest", "support-and-imbalance"),
        keywords=("precision recall curve", "average precision", "ap", "auprc", "pr curve"),
    ),
)
