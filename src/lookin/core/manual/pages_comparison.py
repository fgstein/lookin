"""Chapter: two experiments, the same crops."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("comparison", "Comparing experiments")

PAGES = (
    ManualPage(
        id="fixes-regressions",
        title="Fixes, regressions and the net",
        summary=(
            "Two experiments scored on the same crops split every crop into four buckets: both right, "
            "both wrong, fixed by the candidate, broken by the candidate. The difference in accuracy "
            "is exactly (fixes - regressions) / N -- and the two counts separately say far more than "
            "their difference."
        ),
        shown_in="Compare tab (\"Compare two experiments\")",
        blocks=(
            Heading("The four outcomes"),
            Table(
                headers=("Outcome", "Baseline", "Candidate", "Read as"),
                rows=(
                    ("Both right", "correct", "correct", "the shared floor"),
                    ("Fix", "wrong", "correct", "what the candidate bought"),
                    ("Regression", "correct", "wrong", "what it cost"),
                    ("Both wrong", "wrong", "wrong", "what neither model can do"),
                ),
                weights=(2, 2, 2, 3),
            ),
            Equation(
                r"\mathrm{accuracy}_{B} - \mathrm{accuracy}_{A} = \frac{F - R}{N}",
                where=(
                    (r"F", "fixes: crops the baseline got wrong and the candidate got right"),
                    (r"R", "regressions: crops the baseline got right and the candidate got wrong"),
                    (r"N", "crops scored by both"),
                ),
                note="The headline delta, decomposed into the two movements that produced it.",
            ),
            Heading("Why the two counts matter more than the net"),
            Para(
                "A net of +40 built from 45 fixes and 5 regressions is a better model. The same net "
                "built from 400 fixes and 360 regressions is a *different* model that happens to score "
                "slightly higher -- it has changed its behaviour on 760 crops, and nothing guarantees "
                "the trade holds on the next batch. The second case is also far more fragile: a "
                "re-run with a different seed can flip the sign."
            ),
            Callout(CALLOUT_TIP, "This is a paired comparison, and that is why it is sensitive", (
                Para(
                    "Comparing two accuracies computed separately throws away the fact that the two "
                    "models saw the same crops. Counting fixes and regressions keeps it: the crops "
                    "both models handle identically cancel out, and what remains is exactly the "
                    "evidence about the difference. It is the same logic as McNemar's test, whose "
                    "statistic is built from $F$ and $R$ alone."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Only the crops both experiments scored", (
                Para(
                    "The comparison joins on the crop id, so a crop missing from either results CSV "
                    "is absent from all four buckets. If the two experiments were run on different "
                    "subsets, the $N$ here is the intersection, and it is not the $N$ either "
                    "experiment's own accuracy was computed over."
                ),
            )),
        ),
        see_also=("delta-confusion", "per-class-change", "rates-and-uncertainty", "slice-comparison"),
        keywords=("fix", "regression", "net", "paired", "mcnemar", "diff", "baseline", "candidate"),
    ),

    ManualPage(
        id="delta-confusion",
        title="The delta confusion matrix",
        summary=(
            "The candidate's confusion matrix minus the baseline's, cell by cell. It shows where the "
            "predictions moved rather than how many moved: which confusions were repaired, which were "
            "created, and whether an improvement on one class was paid for by another."
        ),
        shown_in="Compare tab (\"Where the change landed\")",
        blocks=(
            Equation(
                r"\Delta_{a,b} = M^{B}_{a,b} - M^{A}_{a,b}",
                where=(
                    (r"M^{A}", "the baseline's confusion matrix"),
                    (r"M^{B}", "the candidate's"),
                    (r"a, b", "true class, predicted class"),
                ),
            ),
            HowToRead((
                ("A positive diagonal cell", "the candidate gets more of this class right"),
                ("A negative diagonal cell", "it gets fewer right -- a class-level regression"),
                ("A negative off-diagonal cell", "a confusion that was repaired: fewer $a$ called $b$"),
                ("A positive off-diagonal cell", "a confusion that was created"),
                ("A row summing to zero", "the class's support is unchanged, as it must be -- only the distribution across columns moved"),
                ("Colour", "green for an improvement, red for a deterioration, whichever side of the diagonal the cell is on"),
            )),
            Para(
                "The three views above the matrix answer three different questions: the *net* view "
                "shows $\\Delta$ as above; the *fixes* view shows only the crops the candidate "
                "repaired, in the cell they used to occupy; the *regressions* view shows only the "
                "crops it broke, in the cell they moved to."
            ),
            Callout(CALLOUT_TIP, "The pattern to look for", (
                Para(
                    "A change concentrated in one symmetric pair of cells -- fewer $a$ called $b$, "
                    "fewer $b$ called $a$ -- is a genuine improvement in separating two classes. A "
                    "change concentrated in one *column* is the model deciding to predict that class "
                    "more often, which lifts its recall and lowers its precision and may or may not "
                    "be what you wanted."
                ),
            )),
        ),
        see_also=("fixes-regressions", "confusion-matrix", "per-class-change"),
        keywords=("delta", "difference matrix", "confusion delta", "net", "where changed"),
    ),

    ManualPage(
        id="per-class-change",
        title="Per-class fixes and regressions",
        summary=(
            "The same fix/regression counts, broken down by the crop's true class. A flat net can hide "
            "a model that traded one class for another, and this is the table that shows it."
        ),
        shown_in="Compare tab (per-class change table)",
        blocks=(
            Table(
                headers=("Column", "Meaning"),
                rows=(
                    ("support", "crops of this true class in the compared set"),
                    ("fixes", "crops of this class the candidate repaired"),
                    ("regressions", "crops of this class the candidate broke"),
                    ("net", "fixes minus regressions, for this class"),
                ),
                weights=(1, 4),
            ),
            Para(
                "Dividing the net by the support gives the change in that class's recall -- which is "
                "usually the number worth quoting, because 12 fixes on a class of 30 and 12 fixes on "
                "a class of 3000 are not the same event."
            ),
            Callout(CALLOUT_PITFALL, "The zero-sum trade", (
                Para(
                    "Two classes that are confusable often move together and in opposite directions: "
                    "the candidate leans towards one, gaining there and losing here. The overall net "
                    "is near zero and the table shows two large, opposite entries. Whether that is an "
                    "improvement depends entirely on which of the two classes costs more to get wrong "
                    "-- a question no metric in this app can answer for you."
                ),
            )),
        ),
        see_also=("fixes-regressions", "delta-confusion", "support-and-imbalance", "averaging"),
        keywords=("per class", "change", "trade", "class level regression"),
    ),
)
