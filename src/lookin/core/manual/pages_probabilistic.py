"""Chapter: taking the scores literally, as claims about the world."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("probabilistic", "Scores as probabilities")

PAGES = (
    ManualPage(
        id="log-loss",
        title="Log loss (cross-entropy)",
        summary=(
            "The negative log of the probability the model assigned to the true class, averaged over "
            "the crops. It is the loss most classifiers are trained on, it is unbounded above, and a "
            "single confident mistake can dominate it -- which is exactly its point."
        ),
        shown_in="Metrics tab (Summary, column \"Log loss\"), Calibration tab (before/after temperature)",
        blocks=(
            Equation(
                r"\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log p_{i, y_i}",
                where=(
                    (r"p_{i, y_i}", "the score the model gave to the crop's *true* class"),
                    (r"N", "crops scored"),
                ),
                note="Lower is better. A perfect, perfectly confident model scores 0.",
            ),
            Heading("What each value costs"),
            Table(
                headers=("$p_{i,y_i}$", "Contribution $-\\log p$", "Reading"),
                rows=(
                    ("0.99", "0.01", "confident and right: almost free"),
                    ("0.50", "0.69", "hesitant"),
                    ("0.10", "2.30", "wrong, and admitted it"),
                    ("0.01", "4.61", "confidently wrong: costs as much as 460 correct-and-sure crops"),
                    ("0.00", "infinite", "clipped in practice, but the intent is unbounded punishment"),
                ),
                weights=(1, 2, 4),
            ),
            Heading("The baseline"),
            Para(
                "A model that outputs the uniform distribution scores $\\log C$: about 0.69 for two "
                "classes, 2.30 for ten, 3.40 for thirty. A log loss above that is worse than knowing "
                "nothing, which happens more often than one would like -- it means the model is "
                "confidently wrong more often than it is confidently right."
            ),
            Equation(r"\mathcal{L}_{\mathrm{uniform}} = \log C"),
            Heading("Why it is called a proper scoring rule"),
            Para(
                "A scoring rule is *proper* when the way to minimise it is to report your true "
                "beliefs. Log loss is strictly proper: any distortion of an honest probability -- "
                "sharpening it to look decisive, flattening it to look cautious -- increases the "
                "expected loss. That is what makes it a calibration-sensitive metric and not just an "
                "accuracy-sensitive one."
            ),
            Callout(CALLOUT_PITFALL, "One row can own the number", (
                Para(
                    "Because the penalty grows without bound as $p \\to 0$, a handful of crops "
                    "scored near zero on their true class can set the whole average. If log loss "
                    "moved sharply while accuracy did not, look for those rows rather than for a "
                    "global change -- a mislabelled crop that the model confidently contradicts is "
                    "the usual cause, and the Cleaning tab's margin strategy finds exactly those."
                ),
            )),
            Callout(CALLOUT_WARNING, "It needs a real distribution", (
                Para(
                    "Log loss only means anything if the row of scores is a probability distribution "
                    "over the classes. With a sigmoid head or raw logits, the app disables it rather "
                    "than printing a number computed from something that is not a probability."
                ),
            )),
        ),
        see_also=("brier-score", "calibration-error", "temperature-scaling", "score-diagnostics"),
        keywords=("log loss", "cross entropy", "nll", "negative log likelihood", "proper scoring rule"),
    ),

    ManualPage(
        id="brier-score",
        title="Brier score",
        summary=(
            "The mean squared distance between the score vector and the one-hot truth. Like log loss "
            "it is a proper scoring rule, but it is bounded and it punishes a confident mistake "
            "quadratically rather than infinitely -- which makes it the stabler of the two on real "
            "exports."
        ),
        shown_in="Metrics tab (Summary, column \"Brier\")",
        blocks=(
            Equation(
                r"\mathrm{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left( p_{i,c} - o_{i,c} \right)^2",
                where=(
                    (r"o_{i,c}", "1 if the true class of crop $i$ is $c$, else 0 (the one-hot truth)"),
                    (r"p_{i,c}", "the score given to class $c$"),
                ),
                note="Lower is better. 0 is perfect; the worst possible value is 2 in this multiclass form.",
            ),
            Heading("How it differs from log loss"),
            Table(
                headers=("", "Log loss", "Brier"),
                rows=(
                    ("Penalty at $p \\to 0$ on the truth", "unbounded", "at most 2"),
                    ("Dominated by outliers", "yes", "no"),
                    ("Decomposes into calibration terms", "not as directly", "yes -- see the next page"),
                    ("Sensitive to the other classes' scores", "no, only the true class", "yes, all $C$ entries"),
                ),
                weights=(3, 2, 2),
            ),
            Para(
                "That last row matters more than it looks. Log loss only reads the entry for the true "
                "class; Brier reads the whole row. A model that puts 0.6 on the truth and 0.4 on one "
                "wrong class scores worse on Brier than one that puts 0.6 on the truth and spreads "
                "0.4 over ten classes, while log loss cannot tell them apart."
            ),
            Callout(CALLOUT_TIP, "Which to quote", (
                Para(
                    "Quote both, as the Summary table does. Agreement between them means the ranking "
                    "and the calibration are both moving in the same direction; disagreement usually "
                    "means a small number of extreme rows are driving log loss, which is worth "
                    "finding rather than averaging away."
                ),
            )),
        ),
        see_also=("brier-decomposition", "log-loss", "calibration-error", "reliability-diagram"),
        keywords=("brier", "quadratic score", "mean squared error", "proper scoring rule"),
    ),

    ManualPage(
        id="brier-decomposition",
        title="Brier decomposition: reliability, resolution, uncertainty",
        summary=(
            "Murphy's decomposition splits the Brier score into three interpretable parts: how wrong "
            "the announced probabilities are (reliability, lower is better), how much the model "
            "discriminates between crops (resolution, higher is better), and how hard the problem is "
            "in the first place (uncertainty, fixed by the data)."
        ),
        shown_in="Metrics tab (Summary, columns \"Brier reliability / resolution / uncertainty\")",
        blocks=(
            Equation(
                r"\mathrm{BS} = \mathrm{reliability} - \mathrm{resolution} + \mathrm{uncertainty}",
                note="An exact identity, once the scores are binned. The app bins into 15 equal-width bins, per class, one-vs-rest.",
            ),
            Heading("The three terms"),
            Equation(
                r"\mathrm{reliability} = \sum_{b} \frac{n_b}{N} \left( \bar{p}_b - \bar{o}_b \right)^2",
                where=(
                    (r"n_b", "crops in bin $b$"),
                    (r"\bar{p}_b", "mean score announced in bin $b$"),
                    (r"\bar{o}_b", "observed frequency of the class in bin $b$"),
                ),
                note="How far each bin's announced probability sits from what actually happened. Zero for a perfectly calibrated model. Lower is better.",
            ),
            Equation(
                r"\mathrm{resolution} = \sum_{b} \frac{n_b}{N} \left( \bar{o}_b - \bar{o} \right)^2",
                where=((r"\bar{o}", "the base rate: the overall frequency of the class"),),
                note="How far the bins' outcomes spread away from the base rate -- how much the model actually separates crops. Higher is better.",
            ),
            Equation(
                r"\mathrm{uncertainty} = \bar{o} \, (1 - \bar{o})",
                note="The variance of the outcome itself. A property of the dataset, not of the model: no model can change it.",
            ),
            HowToRead((
                ("Reliability near 0", "the announced probabilities can be taken at face value"),
                ("Reliability large", "the model is mis-calibrated -- temperature scaling is the cheap fix"),
                ("Resolution near 0", "the model says the same thing about every crop; it carries no information, however well calibrated"),
                ("Uncertainty", "the Brier score of always predicting the base rate -- the bar the model has to clear"),
            )),
            Callout(CALLOUT_TIP, "The diagnosis this enables", (
                Para(
                    "A bad Brier score has two possible causes and the decomposition tells them apart. "
                    "High reliability with decent resolution: the model knows things but announces them "
                    "wrongly -- calibrate it, and everything improves without retraining. Low "
                    "resolution: the model does not discriminate, and no post-hoc fix will help. Only "
                    "the second is a reason to go back to training."
                ),
            )),
            Callout(CALLOUT_PITFALL, "The numbers depend on the binning", (
                Para(
                    "Reliability and resolution are computed over bins, so both shift a little with "
                    "the number of bins (15 here). The identity always holds, and comparisons between "
                    "experiments in this app are always like-for-like, but do not compare these three "
                    "numbers against a paper that binned differently."
                ),
            )),
        ),
        see_also=("brier-score", "calibration-error", "reliability-diagram", "temperature-scaling"),
        keywords=("murphy", "decomposition", "reliability", "resolution", "uncertainty", "sharpness"),
    ),

    ManualPage(
        id="calibration-error",
        title="Calibration error: ECE and MCE",
        summary=(
            "Of the crops a model calls 0.80, are 80% of them right? ECE measures the average gap "
            "between confidence and accuracy across confidence bins, weighted by how many crops fall "
            "in each; MCE reports the worst bin instead. Both are 0 for a perfectly calibrated model, "
            "and both say nothing about whether the model is any good."
        ),
        shown_in="Metrics tab (Summary, columns \"ECE\" and \"MCE\"), Calibration tab",
        blocks=(
            Heading("The binning"),
            Para(
                "Take the top-label confidence $\\hat{p}_i$ of every crop and drop it into one of $B$ "
                "equal-width bins over $[0, 1]$ (the app uses $B = 15$). Within each bin, compare what "
                "the model claimed to what happened:"
            ),
            Equation(
                r"\mathrm{conf}(b) = \frac{1}{n_b} \sum_{i \in b} \hat{p}_i, \qquad "
                r"\mathrm{acc}(b) = \frac{1}{n_b} \sum_{i \in b} \mathbb{1}[\, \hat{y}_i = y_i \,]",
                where=(
                    (r"n_b", "crops whose confidence falls in bin $b$"),
                    (r"\mathrm{conf}(b)", "the average confidence claimed in that bin"),
                    (r"\mathrm{acc}(b)", "the share of that bin that was actually right"),
                ),
            ),
            Heading("The two summaries"),
            Equation(
                r"\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \mathrm{acc}(b) - \mathrm{conf}(b) \right|",
                note="Expected calibration error: the average gap, weighted by how many crops sit in each bin.",
            ),
            Equation(
                r"\mathrm{MCE} = \max_{b} \left| \mathrm{acc}(b) - \mathrm{conf}(b) \right|",
                note="Maximum calibration error: the worst bin, however few crops it holds.",
            ),
            Para(
                "ECE is the number to quote for the average behaviour; MCE is the number to check "
                "before setting a threshold, because a threshold operates *inside one bin*, and an ECE "
                "of 0.02 is no comfort if the bin you plan to cut at is off by 0.30."
            ),
            HowToRead((
                ("$\\mathrm{conf}(b) > \\mathrm{acc}(b)$", "over-confident in that bin -- the usual direction for a modern network"),
                ("$\\mathrm{conf}(b) < \\mathrm{acc}(b)$", "under-confident -- common after heavy augmentation or label smoothing"),
                ("ECE small, MCE large", "well calibrated on average, badly calibrated somewhere specific; find that bin before trusting a threshold"),
                ("Both near 0 with poor accuracy", "an honest model that knows it is bad -- calibration is not quality"),
            )),
            Callout(CALLOUT_PITFALL, "ECE can be gamed and cannot be compared across setups", (
                Bullets((
                    "It depends on the number of bins: more bins usually means a larger ECE. Fifteen here; a paper quoting ten is not comparable.",
                    "Equal-width bins leave the top bin holding most of the crops on a confident model, which hides structure exactly where thresholds are set.",
                    "A model that predicts a constant confidence equal to its accuracy has ECE 0 and no discrimination whatsoever.",
                )),
            )),
            Callout(CALLOUT_WARNING, "Top-label only", (
                Para(
                    "The ECE computed here looks at the predicted class's confidence, not at the whole "
                    "score vector. A model can be well calibrated on its top label and badly calibrated "
                    "on the rest of the distribution -- which matters when something downstream reads "
                    "more than the first entry. The Brier decomposition is the all-classes view."
                ),
            )),
        ),
        see_also=("reliability-diagram", "temperature-scaling", "brier-decomposition", "confidence-histogram", "log-loss"),
        keywords=("ece", "mce", "calibration", "over-confident", "confidence gap", "bins"),
    ),

    ManualPage(
        id="reliability-diagram",
        title="Reliability diagram",
        summary=(
            "The picture ECE summarises: confidence on the x axis, observed accuracy on the y axis, "
            "one point or bar per confidence bin. A perfectly calibrated model lies on the diagonal. "
            "Where it departs, and in which direction, is what the single ECE number cannot tell you."
        ),
        shown_in="Calibration tab (\"Is the confidence believable?\")",
        blocks=(
            Heading("Construction"),
            Para(
                "Same bins as ECE. For each bin, plot $\\mathrm{conf}(b)$ against $\\mathrm{acc}(b)$. "
                "The diagonal $y = x$ is the locus of perfect calibration: on it, a bin's crops are "
                "right exactly as often as they claimed."
            ),
            HowToRead((
                ("The bars", "the observed accuracy of the crops in that confidence bin"),
                ("The red line", "the mean confidence those same crops claimed"),
                ("The dashed diagonal", "perfect calibration -- bar and line meet on it"),
                ("A bar below the line", "over-confidence in that bin: the model claims more than it delivers"),
                ("A bar above the line", "under-confidence: it is right more often than it says"),
                ("The vertical gap between them", "that bin's contribution to ECE, before weighting by how many crops it holds"),
                ("The lower panel", "what share of all predictions fell in each bin, on a log scale -- how much each bar above is worth"),
                ("A bin with almost no crops", "noise: its accuracy is a rate over a handful of rows, and the panel below says so"),
                ("A sagging right-hand end", "the classic modern-network shape: the most confident crops are the worst calibrated"),
            )),
            Callout(CALLOUT_TIP, "Always read it with the confidence histogram", (
                Para(
                    "A reliability diagram gives every bin the same visual weight, but the bins do not "
                    "hold the same number of crops -- on a confident model, 80% of the data can sit in "
                    "the last two bins. A dramatic-looking gap in a bin holding twelve crops is not a "
                    "calibration problem; it is twelve crops."
                ),
            )),
            Callout(CALLOUT_PITFALL, "It cannot diagnose the fix by itself", (
                Para(
                    "A curve that sags uniformly below the diagonal is what temperature scaling "
                    "corrects almost exactly. A curve that crosses the diagonal, or that is flat, is "
                    "not: no single temperature can fix a model whose confidence is over-stated in one "
                    "region and under-stated in another. Check the after-curve, not just the fitted "
                    "temperature."
                ),
            )),
        ),
        see_also=("calibration-error", "confidence-histogram", "temperature-scaling", "brier-decomposition"),
        keywords=("reliability", "diagram", "calibration plot", "diagonal", "over-confidence"),
    ),

    ManualPage(
        id="confidence-histogram",
        title="Confidence histogram",
        summary=(
            "The distribution of the confidence, split into the crops that turned out right and the "
            "crops that turned out wrong. It answers a question no summary number does: is there a "
            "threshold that separates them at all?"
        ),
        shown_in="Calibration tab (\"Is the confidence believable?\")",
        blocks=(
            Para(
                "Each bar is a count of crops whose confidence $\\hat{p}_i$ fell in that bin, stacked "
                "or coloured by whether the prediction was correct. Nothing is averaged, so nothing is "
                "hidden."
            ),
            HowToRead((
                ("Two separated humps", "the confidence carries real information: a threshold between them buys accuracy cheaply"),
                ("One merged hump", "no threshold will help, however good the ECE looks -- the errors are as confident as the successes"),
                ("A spike at 1.0", "saturated scores. Check for rounded exports or a one-hot head before reading any calibration number"),
                ("Wrong crops piling up at high confidence", "confidently wrong: the expensive kind of error, and the population the Cleaning tab's margin strategy ranks"),
                ("Everything squeezed into a narrow band", "the model barely varies its confidence; abstention has nothing to work with"),
            )),
            Callout(CALLOUT_TIP, "This is the chart that decides whether abstention is viable", (
                Para(
                    "Selective prediction -- accepting only confident crops and sending the rest to a "
                    "human -- only pays when the wrong crops sit lower than the right ones. This "
                    "histogram shows that directly, and the risk-coverage curve puts a number on it."
                ),
            )),
        ),
        see_also=("reliability-diagram", "risk-coverage", "confidence-threshold", "calibration-error"),
        keywords=("histogram", "confidence distribution", "correct", "wrong", "separation"),
    ),

    ManualPage(
        id="temperature-scaling",
        title="Temperature scaling",
        summary=(
            "One scalar $T$ divided into the logits before the softmax. It is the standard post-hoc "
            "calibration fix: it cannot change which class is predicted -- so accuracy, the confusion "
            "matrix and every ranking metric are untouched -- and it usually removes most of the "
            "calibration error at the cost of fitting a single parameter."
        ),
        shown_in="Calibration tab (\"Fix the confidence: temperature scaling\")",
        blocks=(
            Equation(
                r"p_{i,c}(T) = \frac{\exp\!\left( z_{i,c} / T \right)}{\sum_{k=1}^{C} \exp\!\left( z_{i,k} / T \right)}",
                where=(
                    (r"z_{i,c}", "the logit for class $c$, recovered as $\\log p_{i,c}$ up to the constant a softmax ignores"),
                    (r"T > 0", "the temperature"),
                ),
                note="T > 1 flattens the distribution (less confident); T < 1 sharpens it (more confident); T = 1 changes nothing.",
            ),
            Heading("Why the argmax cannot move"),
            Para(
                "Dividing every logit of a row by the same positive number is a monotone "
                "transformation of that row: the largest logit stays the largest. So $\\hat{y}_i$ is "
                "unchanged for every crop, and with it accuracy, precision, recall, F1, the confusion "
                "matrix, ROC-AUC and average precision. *Only the confidence values move.* That is the "
                "whole appeal: there is no accuracy to trade away."
            ),
            Heading("How the temperature is fitted"),
            Para(
                "The app searches for the $T$ that minimises the negative log-likelihood on the loaded "
                "rows:"
            ),
            Equation(
                r"T^{\star} = \arg\min_{T} \; -\frac{1}{N} \sum_{i=1}^{N} \log p_{i, y_i}(T)",
                note="NLL, not ECE: NLL is a smooth proper scoring rule, ECE is a binned proxy with a flat gradient almost everywhere.",
            ),
            Para(
                "The fit reads at most 200000 rows, sampled with a fixed seed so the same data always "
                "gives the same temperature; the before/after numbers are then computed on every row."
            ),
            HowToRead((
                ("$T > 1$", "the model was over-confident -- by far the most common case"),
                ("$T < 1$", "the model was under-confident; often a sign of label smoothing or heavy augmentation"),
                ("$T \\approx 1$", "already calibrated: there is nothing here to win"),
                ("NLL after < NLL before", "the fit worked, by the criterion it optimised"),
                ("ECE after > ECE before", "possible and worth noticing -- see the pitfall below"),
            )),
            Callout(CALLOUT_WARNING, "Fitting and measuring on the same rows is optimistic", (
                Para(
                    "A temperature fitted on the very rows its ECE is then measured on flatters "
                    "itself. With one parameter the optimism is small, but it is not zero. A "
                    "temperature meant for production belongs on a held-out split -- in this app, that "
                    "means loading that split and reading the value fitted there."
                ),
            )),
            Callout(CALLOUT_PITFALL, "NLL improved, ECE got worse", (
                Para(
                    "The two criteria usually agree and occasionally do not: a model whose confidence "
                    "carries little information about correctness can have its NLL improved and its "
                    "ECE degraded by the same $T$. This is why the app shows both numbers before and "
                    "after rather than only the fitted temperature -- a tool that reported $T$ alone "
                    "would be claiming an improvement it had not checked."
                ),
            )),
            Callout(CALLOUT_TIP, "What it cannot fix", (
                Para(
                    "One global scalar cannot repair a model that is over-confident on one class and "
                    "under-confident on another, and it cannot create discrimination that is not "
                    "there. If the after-diagram still departs from the diagonal, the next steps are "
                    "per-class thresholds or vector scaling, not a better $T$."
                ),
            )),
        ),
        see_also=("calibration-error", "reliability-diagram", "log-loss", "per-class-thresholds"),
        keywords=("temperature", "scaling", "guo", "post-hoc", "softmax", "logits", "calibrate"),
    ),
)
