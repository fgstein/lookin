"""Chapter: choosing where to cut, and what abstaining buys."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("abstention", "Operating points and abstention")

PAGES = (
    ManualPage(
        id="confidence-threshold",
        title="Accepting on confidence: coverage and selective accuracy",
        summary=(
            "Accept every prediction whose confidence is at least $t$ and send the rest to a human. "
            "Coverage is the share you automated; selective accuracy is how often the automated part "
            "was right. Raising $t$ always trades one for the other, and the whole point is to find "
            "where that trade is worth taking."
        ),
        shown_in="Calibration tab (\"Pick an operating point\")",
        blocks=(
            Equation(
                r"A(t) = \{\, i \;:\; \hat{p}_i \geq t \,\}",
                where=((r"A(t)", "the accepted set at threshold $t$"),),
                note="Accepted at or above the threshold, not strictly above -- with hard 0/1 scores the two differ by the whole dataset.",
            ),
            Equation(
                r"\mathrm{coverage}(t) = \frac{|A(t)|}{N}, \qquad "
                r"\mathrm{accuracy}(t) = \frac{1}{|A(t)|} \sum_{i \in A(t)} \mathbb{1}[\, \hat{y}_i = y_i \,]",
                note="Coverage is how much work the model took off your hands; selective accuracy is how well it did that work.",
            ),
            Heading("The four numbers the table shows"),
            Table(
                headers=("Column", "Meaning", "Who cares"),
                rows=(
                    ("Accepted", "$|A(t)|$ -- crops the model handles alone", "throughput"),
                    ("Coverage", "accepted over total", "the share of the job automated"),
                    ("Accuracy on accepted", "selective accuracy", "the quality of what ships unchecked"),
                    ("Errors accepted", "mistakes that escape review", "the risk you are accepting"),
                    ("Abstained", "crops sent to a human", "the review workload"),
                    ("Accuracy on abstained", "how right the model was on what it refused", "whether the refusals were justified"),
                ),
                weights=(2, 4, 3),
            ),
            Para(
                "That last row is the one people forget. If the model is 95% right on the crops it "
                "*refused*, the abstention is not selecting hard cases -- it is wasting reviewer time, "
                "and the threshold is set too high."
            ),
            Callout(CALLOUT_PITFALL, "Selective accuracy is not monotone in the threshold", (
                Para(
                    "It usually rises as the threshold rises, but not always: a run of correct "
                    "predictions part-way down the ranking can lift it back up after a dip. This is "
                    "why the app searches for the *most permissive* threshold that meets a target "
                    "rather than the first one it finds -- stopping at the first dip would hand back "
                    "an operating point that automates far less than the model supports."
                ),
            )),
            Callout(CALLOUT_WARNING, "A threshold is only as good as the calibration", (
                Para(
                    "The cut is made on the confidence value, so everything on the calibration pages "
                    "applies here. If the model is over-confident, a threshold of 0.9 accepts crops "
                    "whose real accuracy is 0.75 -- the number in the table is measured, so it will "
                    "say so, but only for *this* pool. Check the reliability diagram before "
                    "transferring a threshold to new data."
                ),
            )),
        ),
        see_also=("risk-coverage", "target-accuracy", "per-class-thresholds", "confidence-histogram", "reliability-diagram"),
        keywords=("threshold", "coverage", "abstention", "selective", "accept", "reject", "operating point"),
    ),

    ManualPage(
        id="risk-coverage",
        title="Risk-coverage curve and AURC",
        summary=(
            "Sweep the threshold from \"accept only the single most confident crop\" to \"accept "
            "everything\" and plot the error rate of the accepted set against its size. The area under "
            "that curve (AURC) is one number for how well the confidence ranks the model's own "
            "mistakes -- lower is better."
        ),
        shown_in="Calibration tab (\"What does abstaining buy?\")",
        blocks=(
            Equation(
                r"\mathrm{risk}(k) = 1 - \frac{1}{k} \sum_{i=1}^{k} \mathbb{1}[\, \hat{y}_{(i)} = y_{(i)} \,], \qquad "
                r"\mathrm{coverage}(k) = \frac{k}{N}",
                where=(
                    (r"(i)", "the $i$-th crop in decreasing order of confidence"),
                    (r"k", "how many of the most confident crops are accepted"),
                ),
                note="Risk is the selective error rate: one minus the selective accuracy.",
            ),
            Equation(
                r"\mathrm{AURC} = \frac{1}{N} \sum_{k=1}^{N} \mathrm{risk}(k)",
                note="The mean risk over every possible coverage. Averaged over all k, not over the drawn points, so the number does not depend on how many points were kept for the chart.",
            ),
            Heading("The floor it is measured against"),
            Para(
                "A model with more errors has a higher AURC through no fault of its ranking, so the "
                "raw number cannot be compared across models of different accuracy. The app therefore "
                "also reports the *optimal* AURC: what the same model would score if it ranked "
                "perfectly, putting every one of its mistakes last."
            ),
            Equation(
                r"\mathrm{AURC}^{\star} = \frac{1}{N} \sum_{k=1}^{N} \left( 1 - \frac{\min(k, N - E)}{k} \right)",
                where=((r"E", "the model's total number of errors"),),
                note="Risk stays at 0 until coverage reaches the accuracy, then climbs. The gap between AURC and this floor is what the confidence ranking costs you.",
            ),
            Equation(
                r"\mathrm{excess\ AURC} = \mathrm{AURC} - \mathrm{AURC}^{\star}",
                note="What the confidence ranking costs, with the model's own error count divided out. This is the number to compare across models of different accuracy.",
            ),
            HowToRead((
                ("The left end", "risk among the most confident crops -- near 0 for a usable confidence"),
                ("The right end", "risk at full coverage, which is exactly $1 - \\mathrm{accuracy}$"),
                ("A flat start", "a large block of crops can be automated at near-zero risk: abstention pays"),
                ("A curve that rises immediately", "errors are spread across the confidence range; no threshold will buy much"),
                ("The gap to the optimal curve", "how much is lost by the ranking rather than by the errors themselves"),
            )),
            Callout(CALLOUT_TIP, "What this measures that ECE does not", (
                Para(
                    "AURC only cares about the *order* of the confidences, not their values. A model "
                    "with a terrible ECE can have an excellent AURC: its numbers are wrong but its "
                    "ranking is right, and a threshold chosen empirically on this curve will work. "
                    "The reverse also happens. Read them together: ECE for whether the numbers can be "
                    "quoted, AURC for whether a cut-off can be found."
                ),
            )),
        ),
        see_also=("confidence-threshold", "target-accuracy", "calibration-summary", "confidence-histogram"),
        keywords=("risk coverage", "aurc", "selective prediction", "rejection", "area under risk"),
    ),

    ManualPage(
        id="target-accuracy",
        title="Working back from a target accuracy",
        summary=(
            "Instead of picking a threshold and reading off the accuracy, state the accuracy the "
            "automated part must reach and let the app find the most permissive threshold that "
            "delivers it. The answer is the largest coverage compatible with your quality bar."
        ),
        shown_in="Calibration tab (\"Pick an operating point\", target accuracy field)",
        blocks=(
            Equation(
                r"t^{\star} = \max \left\{\, t \;:\; \mathrm{accuracy}(t) \geq \alpha \,\right\}",
                where=(
                    (r"\alpha", "the target accuracy on the accepted crops"),
                    (r"t^{\star}", "the threshold that meets it while automating the most"),
                ),
                note="Read as: among all thresholds that reach the bar, take the one that accepts the most crops.",
            ),
            Heading("Two details that matter in practice"),
            Bullets((
                "*The most permissive, not the first.* Selective accuracy can dip and recover as the threshold falls, so the app scans the whole ranking and takes the largest coverage that still meets the target.",
                "*Only thresholds that can actually be expressed.* A threshold cannot split two crops that share the same confidence, so cut points inside a run of equal values are skipped. With hard 0/1 scores that is most of the data, not a corner case.",
            )),
            Heading("When the target cannot be met"),
            Para(
                "If no threshold reaches $\\alpha$ -- not even accepting the single most confident "
                "crop -- the app says so rather than returning the strictest threshold it tried. That "
                "outcome is informative: it means the confidence ranking does not support the quality "
                "bar you set, and the options are to lower the bar, to calibrate, or to improve the "
                "model."
            ),
            Callout(CALLOUT_PITFALL, "The threshold is fitted on the rows you are looking at", (
                Para(
                    "The threshold that just reaches 0.98 here is, by construction, the one that "
                    "*barely* reaches it on this pool. On fresh data it will land slightly below. "
                    "Leave margin, and where the stake is real, set the threshold on one split and "
                    "verify it on another."
                ),
            )),
        ),
        see_also=("confidence-threshold", "risk-coverage", "per-class-thresholds", "rates-and-uncertainty"),
        keywords=("target accuracy", "threshold search", "coverage reached", "unreachable"),
    ),

    ManualPage(
        id="per-class-thresholds",
        title="One threshold per class",
        summary=(
            "A single global cut-off is the wrong control whenever the classes are not equally hard or "
            "equally costly. Here each predicted class gets the most permissive threshold that still "
            "reaches a target *precision* on its own predictions -- so the rare, dangerous class can be "
            "made safe without throwing away the easy majority class."
        ),
        shown_in="Calibration tab (\"Pick an operating point\", per-class table)",
        blocks=(
            Heading("Why precision and not accuracy"),
            Para(
                "The population being thresholded is \"the crops the model called $c$\". Within that "
                "population, a crop is right exactly when its true class is $c$ -- which is the "
                "definition of the precision of class $c$. So the per-class target is a precision "
                "target, and the coverage reported beside it is within the class, not over the dataset."
            ),
            Equation(
                r"t_c^{\star} = \max \left\{\, t \;:\; \frac{|\{ i : \hat{y}_i = c, \; \hat{p}_i \geq t, \; y_i = c \}|}"
                r"{|\{ i : \hat{y}_i = c, \; \hat{p}_i \geq t \}|} \geq \alpha \,\right\}",
                where=(
                    (r"\alpha", "the target precision, the same for every class"),
                    (r"t_c^{\star}", "that class's own accept threshold"),
                ),
            ),
            HowToRead((
                ("Predicted class", "the class the row is about -- the rows the model *called* this, not the rows that *are* this"),
                ("Predictions", "how many crops the model assigned to this class"),
                ("Threshold", "the confidence cut-off found for it; a high value means this class needs to be sure"),
                ("Kept", "predictions surviving the cut"),
                ("Coverage in class", "kept over predicted -- the share of this class's work that stays automated"),
                ("Precision", "measured precision among the kept ones; at or just above the target when it was reachable"),
            )),
            Callout(CALLOUT_TIP, "The shape of the result is the finding", (
                Para(
                    "A class whose threshold comes back near 1.0 with a coverage of 5% is a class the "
                    "model cannot be trusted on: you have just learned that it should be routed to a "
                    "human wholesale. A class whose threshold is 0.3 with a coverage of 99% is fully "
                    "automatable. A single global threshold would have averaged those two facts into "
                    "one unhelpful number."
                ),
            )),
            Callout(CALLOUT_WARNING, "Small classes give unstable thresholds", (
                Para(
                    "A threshold chosen to meet a precision target on 40 predictions is fitted to "
                    "roughly a dozen errors and will not hold on new data. Check the \"Predictions\" "
                    "column before acting on a class's threshold, and treat anything under a few "
                    "hundred rows as indicative -- see the page on how much to trust a rate."
                ),
            )),
        ),
        see_also=("confidence-threshold", "precision-recall", "target-accuracy", "rates-and-uncertainty"),
        keywords=("per class threshold", "target precision", "class specific", "routing"),
    ),

    ManualPage(
        id="calibration-summary",
        title="The Calibration summary, column by column",
        summary=(
            "One row per experiment: what its confidence claims, what it delivers, and how well it "
            "ranks its own mistakes. Three of these columns exist nowhere else in the app -- the "
            "confidence gap, and the two AURC figures the excess is measured between."
        ),
        shown_in="Calibration tab (\"Summary\")",
        blocks=(
            Table(
                headers=("Column", "What it is", "Read it as"),
                rows=(
                    ("Accuracy", "share of crops predicted correctly", "the level the confidence should be announcing"),
                    ("Mean confidence", "$\\frac{1}{N}\\sum_i \\hat{p}_i$", "what the model claims, on average"),
                    ("Confidence gap", "mean confidence minus accuracy", "positive: over-confident. Negative: it undersells itself"),
                    ("ECE", "average confidence gap per bin", "the same question, bin by bin instead of globally"),
                    ("MCE", "the worst bin's gap", "check this before setting a threshold inside that bin"),
                    ("AURC", "area under the risk-coverage curve", "lower is better, but it also falls when a model simply makes fewer errors"),
                    ("Best possible AURC", "the same model with its errors ranked last", "the floor this model's accuracy allows"),
                    ("Excess AURC", "AURC minus that floor", "what the *ranking* costs -- the column that isolates the confidence"),
                ),
                weights=(2, 3, 5),
            ),
            Heading("Why the gap and the ECE can disagree"),
            Para(
                "The confidence gap is a single average over the whole pool, so an over-confident "
                "region and an under-confident one cancel: a model can have a gap of 0.00 and a "
                "large ECE. The reverse cannot happen -- a gap far from zero guarantees a large ECE "
                "-- which makes the gap a fast screen and the ECE the real measurement."
            ),
            Callout(CALLOUT_PITFALL, "A blank probabilistic column is not a zero", (
                Para(
                    "ECE and MCE need the scores to be a distribution over the classes. When they "
                    "are not, those cells are empty rather than wrong. The accuracy, the gap and "
                    "the AURC columns still mean what they say: they only use the top score and its "
                    "order."
                ),
            )),
        ),
        see_also=("calibration-error", "risk-coverage", "reliability-diagram", "score-diagnostics"),
        keywords=("summary", "confidence gap", "excess aurc", "mean confidence", "columns"),
    ),
)
