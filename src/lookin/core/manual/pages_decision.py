"""Chapter: everything computed from the confusion matrix alone."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("decision", "Decision-level metrics")

PAGES = (
    ManualPage(
        id="accuracy",
        title="Accuracy",
        summary=(
            "The share of crops the model got right. The most quoted metric and the most misleading "
            "one: it weights every crop equally, which means a class holding 2% of the pool can be "
            "entirely broken while accuracy moves by two points."
        ),
        shown_in="Metrics tab (Summary), Slices panel, Cleaning tab",
        blocks=(
            Equation(
                r"\mathrm{accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\, \hat{y}_i = y_i \,] "
                r"= \frac{\sum_c M_{c,c}}{N}",
                where=(
                    (r"N", "crops scored"),
                    (r"M_{c,c}", "the diagonal of the confusion matrix"),
                ),
                note="The trace of the confusion matrix over its total.",
            ),
            Heading("What it is good for"),
            Para(
                "Accuracy is the right headline when the classes are roughly balanced *and* every "
                "error costs the same. It is also the only metric on this list that a non-specialist "
                "reads correctly without explanation, which is worth something -- as long as it is not "
                "the only number in the report."
            ),
            Heading("The baseline it must beat"),
            Para(
                "Before reading an accuracy, compute what a model that has learned nothing would get. "
                "Always predicting the largest class scores the prevalence of that class:"
            ),
            Equation(
                r"\mathrm{accuracy}_{\mathrm{trivial}} = \max_c \pi_c",
                note="An accuracy near this value is a model that has learned nothing useful.",
            ),
            Callout(CALLOUT_EXAMPLE, "The 95% model that detects nothing", (
                Para(
                    "1000 crops, 950 background and 50 defects. A model that answers \"background\" "
                    "every time has accuracy 0.95, recall 0 on defects, and is worthless. Balanced "
                    "accuracy for it is 0.5, macro F1 is 0.49, and MCC is 0 -- all three say what "
                    "accuracy hides."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Accuracy across two different pools", (
                Para(
                    "Accuracy depends on the mix of classes in the pool it was measured on. Two "
                    "accuracies measured on two different splits, or on a filtered view of the Data "
                    "tab, are not comparable unless the class mix is the same. Balanced accuracy is "
                    "the version that survives this."
                ),
            )),
        ),
        see_also=("balanced-accuracy", "confusion-matrix", "support-and-imbalance", "mcc", "rates-and-uncertainty"),
        keywords=("accuracy", "correct", "hit rate", "top-1"),
    ),

    ManualPage(
        id="precision-recall",
        title="Precision, recall and specificity",
        summary=(
            "Three rates computed within one class. Recall asks what share of the real ones you found; "
            "precision asks what share of your alarms were real; specificity asks what share of the "
            "others you correctly left alone. They trade against each other, and which one matters is "
            "a property of your application, not of the model."
        ),
        shown_in="Metrics tab (per-class table and bars), Calibration tab, Slices panel",
        blocks=(
            Equation(
                r"\mathrm{precision}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c}",
                where=(
                    (r"\mathrm{TP}_c", "crops of $c$ called $c$"),
                    (r"\mathrm{FP}_c", "crops of another class called $c$"),
                ),
                note="Of everything the model called c, how much really was c. The denominator is a column of the confusion matrix.",
            ),
            Equation(
                r"\mathrm{recall}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c} = \frac{\mathrm{TP}_c}{n_c}",
                where=((r"\mathrm{FN}_c", "crops of $c$ called something else"),),
                note="Of everything that really was c, how much the model found. The denominator is a row: the support.",
            ),
            Equation(
                r"\mathrm{specificity}_c = \frac{\mathrm{TN}_c}{\mathrm{TN}_c + \mathrm{FP}_c}",
                note="Of everything that was not c, how much the model correctly left alone. Equal to 1 minus the false alarm rate.",
            ),
            Heading("Reading them together"),
            Table(
                headers=("Symptom", "Reading"),
                rows=(
                    ("High recall, low precision", "the model over-calls this class: it catches the real ones and much else besides"),
                    ("High precision, low recall", "the model is timid: when it commits it is right, but it misses most of them"),
                    ("Both low", "the class is not learned, or its definition overlaps another class"),
                    ("High specificity, low precision", "the class is rare -- see the pitfall below"),
                ),
                weights=(2, 5),
            ),
            Callout(CALLOUT_PITFALL, "Specificity flatters a rare class", (
                Para(
                    "With 50 defects among 1000 crops, a detector raising 100 false alarms still has "
                    "specificity $850/950 \\approx 0.89$, because the negatives are so numerous that "
                    "100 of them barely register. Precision, on the same numbers, is at most "
                    "$50/150 \\approx 0.33$. When the class is rare, precision is the honest number "
                    "and specificity is nearly free."
                ),
            )),
            Heading("The trade-off is a threshold, not a property"),
            Para(
                "For a model that emits scores, precision and recall are two ends of one dial. Accept "
                "predictions only above a high confidence and precision rises while recall falls; lower "
                "the bar and the reverse happens. This is why the pair should be read at a stated "
                "operating point, and why the precision-recall curve -- which shows every operating "
                "point at once -- is the threshold-free version of this page."
            ),
            Callout(CALLOUT_TIP, "Choosing which to optimise", (
                Bullets((
                    "A missed case is expensive (screening, safety, defect escape): optimise recall, accept the false alarms, and use abstention to keep them reviewable.",
                    "A false alarm is expensive (an automated action, an alert a human must triage): optimise precision, and set a per-class threshold for it.",
                    "You cannot state which is worse: report both, plus the curve, and let whoever owns the cost decide.",
                )),
            )),
        ),
        see_also=("f1", "confusion-matrix", "pr-auc", "per-class-thresholds", "slice-metrics"),
        keywords=("precision", "recall", "sensitivity", "specificity", "TPR", "FPR", "false alarm", "miss"),
    ),

    ManualPage(
        id="f1",
        title="F1 score",
        summary=(
            "The harmonic mean of precision and recall: one number that is only high when *both* are "
            "high. It is the standard summary of a per-class detector, and its main limitation is that "
            "it ignores true negatives entirely and fixes the precision/recall trade-off at an "
            "arbitrary 50/50."
        ),
        shown_in="Metrics tab (per-class bars, Summary), Compare tab, Slices panel",
        blocks=(
            Equation(
                r"\mathrm{F}_1 = 2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}"
                r"{\mathrm{precision} + \mathrm{recall}}",
                note="The harmonic mean of the two rates.",
            ),
            Para(
                "Written directly in counts -- which is how the app computes it, because this form "
                "stays defined when a class is absent from a slice:"
            ),
            Equation(
                r"\mathrm{F}_1 = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}",
                note="No division happens before the last one, so a class with no predictions does not produce a NaN halfway through.",
            ),
            Heading("Why harmonic and not arithmetic"),
            Para(
                "The harmonic mean is dragged down by the smaller of the two values, which is exactly "
                "the behaviour wanted: a detector with precision 1.0 and recall 0.02 is not \"51% "
                "good\". The arithmetic mean would say 0.51; F1 says 0.039."
            ),
            Table(
                headers=("Precision", "Recall", "Arithmetic mean", "$\\mathrm{F}_1$"),
                rows=(
                    ("1.00", "0.02", "0.51", "0.039"),
                    ("0.70", "0.70", "0.70", "0.700"),
                    ("0.90", "0.50", "0.70", "0.643"),
                ),
            ),
            Heading("The general form"),
            Para(
                "F1 is the $\\beta = 1$ case of a family that lets you say which side matters more. "
                "The app reports F1 only, but the family is worth knowing when someone asks for "
                "\"recall-weighted\":"
            ),
            Equation(
                r"\mathrm{F}_{\beta} = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}"
                r"{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}",
                where=((r"\beta", "how many times more recall is worth than precision"),),
                note="Beta = 2 weights recall higher; beta = 0.5 weights precision higher.",
            ),
            Callout(CALLOUT_PITFALL, "F1 ignores true negatives", (
                Para(
                    "None of TP, FP and FN is a count of crops correctly left alone, so F1 cannot see "
                    "them. Two models that differ only in how many easy negatives they get right have "
                    "the same F1. When the negatives are the point, use MCC or balanced accuracy "
                    "instead -- both use all four cells."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Macro F1 is not the F1 of the macro averages", (
                Para(
                    "The app averages the per-class F1 values. Computing F1 from the macro-averaged "
                    "precision and the macro-averaged recall instead gives a different, higher number. "
                    "Both appear in the literature; they are not interchangeable, and the difference "
                    "grows with imbalance."
                ),
            )),
        ),
        see_also=("precision-recall", "averaging", "mcc", "balanced-accuracy", "pr-auc"),
        keywords=("f1", "f-score", "harmonic mean", "fbeta", "dice"),
    ),

    ManualPage(
        id="balanced-accuracy",
        title="Balanced accuracy",
        summary=(
            "The mean of the per-class recalls -- accuracy computed as if every class were the same "
            "size. It is the one-line answer to \"accuracy is high but is the rare class working?\", "
            "and its trivial baseline is always $1/C$, whatever the class mix."
        ),
        shown_in="Metrics tab (Summary, column \"Balanced Acc\"), Cleaning tab",
        blocks=(
            Equation(
                r"\mathrm{balanced\ accuracy} = \frac{1}{C} \sum_{c=1}^{C} \mathrm{recall}_c "
                r"= \frac{1}{C} \sum_{c=1}^{C} \frac{M_{c,c}}{n_c}",
                where=(
                    (r"C", "number of classes present"),
                    (r"n_c", "support of class $c$"),
                ),
                note="Macro-averaged recall. Each class contributes one term, whatever its size.",
            ),
            Heading("Why it is the imbalance-proof accuracy"),
            Para(
                "Accuracy weights a class by how common it is; balanced accuracy re-weights every crop "
                "by $1/n_c$ so that each class contributes equally. A model that ignores a rare class "
                "entirely loses a full $1/C$ of the score, however few crops that costs it."
            ),
            Table(
                headers=("Model on 950 background / 50 defects", "Accuracy", "Balanced accuracy"),
                rows=(
                    ("always background", "0.950", "0.500"),
                    ("finds half the defects, no false alarms", "0.975", "0.750"),
                    ("finds all defects, 50 false alarms", "0.950", "0.974"),
                ),
                weights=(4, 2, 2),
                caption="The third row is the model you want and the one accuracy cannot distinguish from the first.",
            ),
            Callout(CALLOUT_PITFALL, "It says nothing about precision", (
                Para(
                    "Balanced accuracy is built from recalls only. A model that fires on everything has "
                    "perfect recall on every class and a balanced accuracy of 1.0 while being useless. "
                    "It is safe as a headline only next to a precision-aware number -- macro F1 or MCC."
                ),
            )),
            Callout(CALLOUT_TIP, "Relation to other names", (
                Para(
                    "In the two-class case balanced accuracy is the mean of sensitivity and "
                    "specificity, and equals the ROC curve's Youden index $J$ shifted and halved: "
                    "$\\mathrm{balanced\\ accuracy} = (J+1)/2$."
                ),
            )),
        ),
        see_also=("accuracy", "precision-recall", "averaging", "mcc", "support-and-imbalance"),
        keywords=("balanced accuracy", "macro recall", "imbalance", "bacc"),
    ),

    ManualPage(
        id="mcc",
        title="Matthews correlation coefficient (MCC)",
        summary=(
            "A correlation coefficient between the truth and the prediction, ranging from -1 to +1. It "
            "is the only single number in the Summary table that uses all four cells of the confusion "
            "matrix and stays honest under any class imbalance: 0 means \"no better than guessing\", "
            "whatever the mix."
        ),
        shown_in="Metrics tab (Summary, column \"MCC\"), Cleaning tab",
        blocks=(
            Heading("Two classes"),
            Equation(
                r"\mathrm{MCC} = \frac{\mathrm{TP} \cdot \mathrm{TN} - \mathrm{FP} \cdot \mathrm{FN}}"
                r"{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}",
                note="Exactly the Pearson correlation between the 0/1 truth vector and the 0/1 prediction vector.",
            ),
            Heading("Many classes"),
            Para(
                "The app computes the multiclass generalisation directly from the confusion matrix "
                "$M$, which reduces to the formula above when $C = 2$:"
            ),
            Equation(
                r"\mathrm{MCC} = \frac{N \sum_c M_{c,c} - \sum_c t_c q_c}"
                r"{\sqrt{\left(N^2 - \sum_c q_c^2\right)\left(N^2 - \sum_c t_c^2\right)}}",
                where=(
                    (r"t_c", "crops whose true class is $c$ (a row sum, the support)"),
                    (r"q_c", "crops predicted $c$ (a column sum)"),
                ),
            ),
            HowToRead((
                ("$\\mathrm{MCC} = 1$", "perfect agreement"),
                ("$\\mathrm{MCC} = 0$", "no association: the prediction carries no information about the truth"),
                ("$\\mathrm{MCC} < 0$", "systematic disagreement -- often a permuted or inverted label mapping, worth checking before anything else"),
                ("$\\mathrm{MCC} \\approx 0.3$", "weak but real; on an imbalanced problem this can coexist with an accuracy of 0.95"),
            )),
            Callout(CALLOUT_TIP, "Why it is the honest default", (
                Para(
                    "The always-majority model scores accuracy $\\max_c \\pi_c$ (high), F1 on the rare "
                    "class 0, and MCC exactly 0. No re-weighting, no baseline to remember: the "
                    "uninformative model sits at zero by construction. That property is what makes MCC "
                    "the metric to reach for when comparing models across datasets with different "
                    "class mixes."
                ),
            )),
            Callout(CALLOUT_PITFALL, "It is not a percentage", (
                Para(
                    "An MCC of 0.6 is not \"60% right\". It is a correlation, and its scale is not "
                    "linear in anything a stakeholder recognises. Report it next to a rate they can "
                    "read, not instead of one."
                ),
            )),
        ),
        see_also=("cohen-kappa", "accuracy", "balanced-accuracy", "confusion-matrix"),
        keywords=("mcc", "matthews", "phi coefficient", "correlation"),
    ),

    ManualPage(
        id="cohen-kappa",
        title="Cohen's kappa",
        summary=(
            "Accuracy with the part explainable by chance subtracted out, so that 0 means \"no better "
            "than a model guessing with the same class frequencies\" and 1 means perfect. It comes "
            "from inter-annotator agreement, which is also where it is most useful here."
        ),
        shown_in="Metrics tab (Summary, column \"Cohen's kappa\")",
        blocks=(
            Equation(
                r"\kappa = \frac{p_o - p_e}{1 - p_e}",
                where=(
                    (r"p_o", "observed agreement: the accuracy"),
                    (r"p_e", "agreement expected by chance, from the marginals"),
                ),
            ),
            Equation(
                r"p_e = \sum_{c=1}^{C} \frac{t_c}{N} \cdot \frac{q_c}{N}",
                where=(
                    (r"t_c", "crops whose true class is $c$"),
                    (r"q_c", "crops predicted $c$"),
                ),
                note="The accuracy a model would reach by drawing predictions at random with its own observed frequencies.",
            ),
            Heading("Reading it"),
            Para(
                "The denominator is how much room there was above chance, so kappa is \"the share of "
                "the achievable improvement that was actually achieved\". On a heavily imbalanced "
                "problem $p_e$ is close to 1, the denominator is tiny, and kappa becomes a much harsher "
                "-- and much more informative -- number than accuracy."
            ),
            Table(
                headers=("$\\kappa$", "Conventional reading"),
                rows=(
                    ("below 0", "worse than chance"),
                    ("0.00 to 0.20", "slight"),
                    ("0.21 to 0.40", "fair"),
                    ("0.41 to 0.60", "moderate"),
                    ("0.61 to 0.80", "substantial"),
                    ("0.81 to 1.00", "almost perfect"),
                ),
                weights=(1, 4),
                caption="Landis and Koch's bands -- a convention, not a law, and not comparable across datasets with different class mixes.",
            ),
            Callout(CALLOUT_TIP, "The use that matters here", (
                Para(
                    "Kappa was designed to compare two *annotators*. If you have a second set of labels "
                    "for the same crops -- a re-annotation pass, a second reviewer, the review versions "
                    "of the Cleaning tab -- kappa between those two label sets tells you the ceiling "
                    "any model can reach on this data. A model whose kappa against the labels matches "
                    "the humans' kappa against each other has learned everything the annotation "
                    "contains."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Kappa and MCC disagree on purpose", (
                Para(
                    "Kappa corrects for chance using the marginals of *this* model's predictions, so a "
                    "model that changes how often it predicts each class changes its own baseline. MCC "
                    "has no such moving reference. When the two disagree sharply, the model's predicted "
                    "class frequencies are far from the true ones -- which is itself the finding."
                ),
            )),
        ),
        see_also=("mcc", "accuracy", "confusion-matrix", "cleaning-impact"),
        keywords=("kappa", "cohen", "chance agreement", "annotator", "inter-rater"),
    ),
)
