"""Chapter: changing the decision itself -- one bias per class before the argmax."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("decision-rule", "Decision rule")

PAGES = (
    ManualPage(
        id="decision-rule",
        title="One bias per class before the argmax",
        summary=(
            "A multiclass model decides by taking the class with the highest score. Adding a bias to each "
            "class's log-score before that argmax is the multiclass form of moving a threshold: every crop "
            "still gets exactly one class, so -- unlike a one-vs-rest threshold per class -- the rule can be "
            "deployed. The Decision Rule tab finds the biases that maximise a metric, or lets you type them."
        ),
        shown_in="Decision Rule tab",
        blocks=(
            Heading("The rule"),
            Equation(
                r"\hat{y} = \arg\max_c \; \left( \log p_c + b_c \right)",
                where=(
                    (r"p_c", "the score of class $c$ -- a probability when the rows sum to 1"),
                    (r"b_c", "the bias of class $c$, in log-score units"),
                ),
                note="When the scores are raw logits, the bias is added to the logit itself. Scores in [0, 1] that do not sum to 1 (independent sigmoids) are refused.",
            ),
            Para(
                "It is the same as weighting each probability: $\\arg\\max_c\\, w_c\\, p_c$ with "
                "$w_c = e^{b_c}$. A bias of $+0.7$ roughly doubles how much class $c$ is favoured; "
                "$-0.7$ halves it. Every bias at 0 is the plain argmax."
            ),
            Heading("Why a bias on the log-scores"),
            Para(
                "It is exactly what the same network would do with $b$ added to the bias of its last layer: "
                "its scores would become $\\mathrm{softmax}(\\log p + b)$, i.e. $p_c\\, e^{b_c}$ renormalised. "
                "That is what *Apply to the workspace* writes, so every tab reads the experiment as if it had "
                "been trained that way -- confidence, calibration and ranking metrics included."
            ),
            Para(
                "For F1, the theoretically best rule weighs *and* shifts each calibrated probability "
                "($\\arg\\max_c\\, a_c p_c - d_c$). No network outputs that, so it could not be applied as if "
                "native; the bias on the log-scores is the part of it a network can carry."
            ),
            Heading("Only defined up to a constant"),
            Para(
                "Adding the same number to every bias changes no decision. One class is therefore kept at 0: "
                "any class you lock at 0, or else the class with the most tuning rows. Only differences "
                "between biases mean something."
            ),
            Heading("What is maximised"),
            Table(
                headers=("Objective", "Counts", "Subset of classes"),
                rows=(
                    ("Macro F1", "every counted class once", "yes: tick *In objective*"),
                    ("Weighted F1", "every counted class by its rows", "yes"),
                    ("Balanced accuracy", "the mean recall of every class", "no -- see below"),
                ),
                weights=(3, 4, 3),
            ),
            Para(
                "Leaving a class out of an F1 objective says you do not need it to perform: the fit may then "
                "give it away, and the tab flags it when its F1 drops. Balanced accuracy is never restricted: a "
                "recall never pays for a false positive, so over a subset of classes its best is to predict "
                "those classes everywhere."
            ),
            Heading("How the biases are found"),
            Para(
                "One class at a time, the others fixed: a crop goes to class $c$ exactly when $b_c$ exceeds "
                "$\\max_{k \\neq c}(\\log p_k + b_k) - \\log p_c$. Sorting those thresholds gives the objective "
                "for every value of $b_c$ at once, and the best one is kept. Passes over the classes repeat "
                "until no single bias can be moved for the better."
            ),
            Callout(CALLOUT_WARNING, "A local optimum", (
                Para(
                    "Each step is exact and the objective never decreases, but the end point is one where no "
                    "*single* bias helps -- not necessarily the best set of biases overall. With many classes, "
                    "a second Compute from the biases found can still move a little."
                ),
            )),
            HowToRead((
                ("Tuning rows", "how many rows of the class the biases were fitted on -- under a few dozen, its bias is a guess"),
                ("In objective", "whether the class's F1 counts in what is maximised"),
                ("Lock at 0", "the class's bias stays 0; the others still move around it"),
                ("Bias", "typed by hand or fitted; every change is scored at once"),
                ("F1 before / after", "on the evaluated rows, plain argmax and with the biases"),
            )),
            Callout(CALLOUT_TIP, "Reading the result", (
                Bullets((
                    "Large gains on small classes and small losses on a big one is the usual shape: the argmax favours classes the training set had plenty of.",
                    "A gain on the tuning rows that does not survive on the evaluated rows is overfitting: the classes involved have too few rows to set a bias.",
                    "Compare with the one-vs-rest F1-max: it is the ceiling of each class's score taken alone, which no rule giving every crop one class can reach.",
                )),
            )),
        ),
        see_also=("decision-rule-pools", "decision-rule-curve", "f1-max", "f1", "per-class-thresholds"),
        keywords=("bias", "logit adjustment", "class prior", "weighted argmax", "decision rule", "threshold"),
    ),

    ManualPage(
        id="decision-rule-pools",
        title="Tuning rows and evaluated rows",
        summary=(
            "Biases fitted on some rows and measured on the same rows look better than they are. The "
            "Decision Rule tab tunes on the validation rows of the dataset CSV by default and measures on the "
            "evaluated rows -- the ones every other tab uses -- after checking that no row is in both."
        ),
        shown_in="Decision Rule tab (\"Tuning rows\")",
        blocks=(
            Heading("Where the validation rows come from"),
            Para(
                "The dataset CSV is normally loaded through a row filter (Sources: keep only rows where "
                "*split* is *test*), so the other splits are not in memory. The tab reads the rows where the "
                "chosen column has the chosen value (*split* = *val* by default) from the same file, and takes "
                "the experiment's predictions on them from its results CSV. The other tabs never see them."
            ),
            Bullets((
                "No prediction for any validation row: the results CSV covers only the evaluated rows. Run the model on the validation rows and add them to the same results CSV.",
                "Predictions for some of them: the biases are tuned on those that have one, and the tab says how many are missing.",
                "A uuid both among the validation rows and the evaluated rows: refused, since the rule would then be scored on rows it was fitted to.",
            )),
            Heading("Tuning on the evaluated rows"),
            Para(
                "Allowed, because it is useful to explore what a rule can do, but every number measured "
                "afterwards is optimistic. The tab says so in red, and after a fit estimates by how much: it "
                "fits on one half of the rows (class by class), scores the other half, and the other way round."
            ),
            Equation(
                r"\mathrm{optimism} \approx J_{\mathrm{in\ sample}} - J_{\mathrm{held\ out}}",
                where=((r"J", "the objective, e.g. the macro F1"),),
                note="The held-out value is what to expect of these biases on new rows.",
            ),
            Callout(CALLOUT_PITFALL, "Applied after tuning on the evaluated rows", (
                Para(
                    "Applying such a rule to the workspace makes every tab show numbers measured on the rows "
                    "the rule was fitted to. The experiment then carries a red *tuned on test* badge everywhere, "
                    "and the top bar says so on every tab."
                ),
            )),
        ),
        see_also=("decision-rule", "rates-and-uncertainty"),
        keywords=("validation", "val split", "test split", "overfitting", "optimism", "cross-fitting", "held out"),
    ),

    ManualPage(
        id="decision-rule-curve",
        title="One class, every bias: the deployable precision-recall curve",
        summary=(
            "With every other bias fixed, sweeping one class's bias from very negative to very positive moves "
            "its precision and recall along a curve -- the precision-recall curve of a rule that gives every "
            "crop one class. Drawn over the one-vs-rest curve of the same class, it shows what the "
            "single-class constraint costs."
        ),
        shown_in="Decision Rule tab (\"One class, every bias\")",
        blocks=(
            Heading("The two curves"),
            Para(
                "The dashed curve ranks the crops by the class's own score alone, as the Metrics tab does: a "
                "crop can be counted for this class and for another at the same time. The solid curve is the "
                "argmax with this class's bias swept: a crop goes to the class only when the class beats every "
                "other one, biases included."
            ),
            Equation(
                r"c \;\mathrm{wins\ crop}\ i \;\Leftrightarrow\; b_c > \max_{k \neq c} (\log p_{ik} + b_k) - \log p_{ic}",
                note="Each crop has one such threshold: sorting them draws the whole solid curve.",
            ),
            HowToRead((
                ("The solid curve", "every operating point this class can be moved to, the other biases as set"),
                ("The dashed curve", "one-vs-rest on the class's own score: a reference no single-label rule reaches"),
                ("The open circle", "the plain argmax, every bias at 0"),
                ("The filled dot", "the biases as they are now"),
                ("The grey curves", "iso-F1 lines: the highest one the solid curve touches is the best F1 this bias can give"),
                ("Right panel", "the class's F1, and the objective, against its bias -- a flat top means the bias is robust, a spike that it is fragile"),
            )),
            Callout(CALLOUT_EXAMPLE, "Reading the gap", (
                Para(
                    "When the solid curve sits close to the dashed one, the other classes rarely compete for "
                    "this class's crops and the bias can reach almost what the score allows. A wide gap means "
                    "its crops are disputed with other classes: moving its bias trades its errors against theirs."
                ),
            )),
        ),
        see_also=("decision-rule", "pr-auc", "f1-max", "one-vs-rest"),
        keywords=("sweep", "bias curve", "argmax curve", "operating point"),
    ),
)
