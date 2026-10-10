"""Chapter: how sure the model was, independent of whether it was right."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("confidence", "Per-prediction confidence")

PAGES = (
    ManualPage(
        id="margin-entropy",
        title="Margin and entropy: two measures of certainty",
        summary=(
            "Margin is the gap between the top two scores; entropy is how spread out the whole score "
            "row is. Both describe how sure the model was about one prediction, with no reference to "
            "the ground truth at all -- unlike every number on the Metrics tab, these are defined even "
            "for a crop with no label."
        ),
        shown_in="Confidence tab (Summary, distributions, per-class breakdown, scatter)",
        blocks=(
            Equation(
                r"m_i = s_{i,(1)} - s_{i,(2)}",
                where=(
                    (r"s_{i,(1)}", "the crop's highest raw score"),
                    (r"s_{i,(2)}", "its second-highest raw score"),
                ),
                note="A clean win scores close to the full scale; a near-tie scores close to 0, whatever the scale of the scores themselves.",
            ),
            Equation(
                r"H_i = -\sum_{c=1}^{C} p_{i,c} \log p_{i,c}, \qquad \tilde{H}_i = \frac{H_i}{\log C}",
                where=(
                    (r"p_i", "the crop's score row, taken as a distribution over the $C$ classes"),
                    (r"\tilde{H}_i", "entropy divided by its maximum, so it lands in $[0, 1]$"),
                ),
                note="0 means the model put all its mass on one class; 1 means it spread it out evenly over every class, whatever $C$ is.",
            ),
            Heading("Why entropy sometimes goes through a softmax first"),
            Para(
                "Entropy is only defined for a probability distribution, so a row that does not already "
                "sum to 1 is passed through softmax first. Margin needs no such step -- it only compares "
                "the top two raw scores, so it is unaffected either way. Whether that happened for the "
                "experiment on screen is reported in the banner above the charts; see the page on score "
                "diagnostics for what \"already a distribution\" means."
            ),
            Callout(CALLOUT_WARNING, "Softmax on already-hard scores invents structure", (
                Para(
                    "A hard 0/1 top-1 export has exactly one non-zero score, so its entropy is always 0 "
                    "regardless of the softmax step -- but a row of near-ties on an unnormalized scale "
                    "gets flattened toward uniform by softmax in a way that depends on that scale, not on "
                    "the model's real spread. Read entropy on such an experiment as a ranking, not as a "
                    "calibrated quantity."
                ),
            )),
            Heading("The Summary row"),
            Table(
                headers=("Column", "Meaning"),
                rows=(
                    ("N", "predictions with a defined margin and entropy"),
                    ("Accuracy", "share of those predictions that were correct"),
                    ("Mean / median margin", "central tendency of $m_i$ over the experiment"),
                    ("Mean / median entropy (norm)", "central tendency of $\\tilde{H}_i$ over the experiment"),
                ),
                weights=(2, 4),
            ),
            Heading("Confidence distributions"),
            HowToRead((
                ("Each histogram", "the density of margin, or of normalized entropy, split by whether the prediction was correct"),
                ("Errors clustering at low margin / high entropy", "the expected shape -- the model hedges on the crops it gets wrong"),
                ("Errors spread across the whole range", "the confidence measure is not tracking correctness for this experiment -- see 'Confidently wrong'"),
            )),
            Heading("Per-class confidence"),
            HowToRead((
                ("Each bar", "the true class's mean margin (top panel) or mean entropy (bottom panel), over every crop of that class"),
                ("Class order", "sorted by mean margin, least confident first, the same order in both panels"),
                ("A class with fine accuracy but low margin", "the model is right about it but not by much -- worth a lower threshold on that class specifically"),
            )),
            Heading("Margin vs. entropy scatter"),
            HowToRead((
                ("Each point", "one prediction, plotted at its own margin and normalized entropy, colored by correctness"),
                ("Points near the bottom-right", "a clean top-1 win: high margin, low entropy"),
                ("Points near the top-left", "spread thin over several classes: low margin, high entropy"),
                ("High entropy at a middling margin", "a near-tie spread across *several* classes rather than a clean two-way tie -- the two measures do not have to agree"),
            )),
        ),
        see_also=("score-diagnostics", "confidently-wrong", "confidence-threshold", "risk-coverage"),
        keywords=("margin", "entropy", "softmax", "confidence", "certainty", "top1", "top2"),
    ),

    ManualPage(
        id="confidently-wrong",
        title="Confidently wrong",
        summary=(
            "The predictions that are both incorrect and among the model's own most confident, by "
            "margin or by entropy. A low-confidence error at least leaves a trace a threshold would "
            "catch; a confidently-wrong one would have sailed through it, which is what makes this the "
            "failure mode worth reading row by row."
        ),
        shown_in="Confidence tab (\"Confidently wrong\")",
        blocks=(
            Equation(
                r"W(q) = \{\, i \;:\; \hat{y}_i \neq y_i, \;\; \mathrm{rank}_i(\mathrm{measure}) \leq q \,\}",
                where=(
                    (r"q", "the chosen top percentile, e.g. top 25%"),
                    (r"\mathrm{measure}", "margin (high is confident) or normalized entropy (low is confident)"),
                ),
                note="Both incorrect and inside the most-confident slice of the whole experiment.",
            ),
            Heading("The threshold is set on everyone, not just the errors"),
            Para(
                "The cut-off for \"top $q$% most confident\" is taken from the margin or entropy of "
                "*every* prediction in the experiment, correct or not, and only then intersected with "
                "the wrong ones. That is what keeps \"top 25% most confident\" meaning the same slice of "
                "confidence whether the experiment is 1% wrong or 40% wrong -- the alternative, ranking "
                "the errors only among themselves, would quietly change definition as accuracy changed."
            ),
            Heading("Reading the table"),
            Table(
                headers=("Column", "Meaning"),
                rows=(
                    ("true label / predicted", "the confusion pair -- what the crop actually was and what the model called it"),
                    ("count", "how many confidently-wrong crops fall on that pair"),
                    ("mean margin / mean entropy", "how sure the model was on that pair specifically, among the confidently-wrong ones"),
                ),
                weights=(3, 5),
            ),
            Para(
                "The open-in-new icon on each row jumps to that exact true/predicted pair in the "
                "Gallery, which is the fastest way from \"this pair is common\" to \"here is why\"."
            ),
            Callout(CALLOUT_TIP, "A pair that dominates this table is a lead, not a verdict", (
                Para(
                    "Two classes the model confuses confidently and often are usually either genuinely "
                    "close (worth a note, not a fix) or mislabelled in the source data (worth a pass "
                    "through the Cleaning tab's margin strategy, which flags the same kind of row from "
                    "the other direction)."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Small counts do not make a pattern", (
                Para(
                    "A pair with 2 confidently-wrong crops out of a class of 3,000 is noise until more "
                    "data says otherwise. Read the count column before generalising from a row."
                ),
            )),
        ),
        see_also=("margin-entropy", "confidence-threshold", "margin-candidates"),
        keywords=("confidently wrong", "top percentile", "most confident errors", "confusion pair"),
    ),
)
