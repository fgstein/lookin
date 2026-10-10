"""Chapter: the same metric, cut by a metadata column."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("slices", "Slices and subgroups")

PAGES = (
    ManualPage(
        id="slice-breakdown",
        title="Breaking a metric down by metadata",
        summary=(
            "One overall number describes the average crop, and nobody deploys on the average crop. "
            "Slicing recomputes the same metric within each value of a metadata column -- a sensor, a "
            "site, a time of day, a size bucket -- and shows where the model actually fails."
        ),
        shown_in="Slices panel (\"Break down by metadata\")",
        blocks=(
            Heading("How a column becomes slices"),
            Para(
                "Each row of the dataset is assigned to exactly one slice, from the column you pick:"
            ),
            Bullets((
                "*A categorical column* gives one slice per distinct value.",
                "*A numeric column with few distinct values* gives one slice per value too. A camera index, a gain setting or a count of objects is a label that happens to be written as a number, and cutting it into deciles would produce bins that mix settings you wanted told apart.",
                "*A continuous numeric column* is cut into bins, by quantile or by equal width.",
                "*Missing values* get their own slice rather than being dropped, because \"the metric on the rows with no value\" is often the most interesting row in the table.",
            )),
            Para(
                "The cut is computed once and reused for every experiment, so two experiments compared "
                "side by side are always compared on identical slices -- a binning recomputed per "
                "experiment would silently move the boundaries."
            ),
            HowToRead((
                ("One row", "one slice: its name, its size $n$, and the metric within it"),
                ("The $n$ column", "the denominator. Read it before reading the metric -- see the pitfall below"),
                ("The error bar", "the 95% Wilson interval for that slice's rate"),
                ("The reference row", "the same metric over the whole pool, for comparison"),
                ("The delta column", "slice minus reference: how far this subgroup is from the average"),
            )),
            Callout(CALLOUT_PITFALL, "The worst slice is usually the smallest one", (
                Para(
                    "Slicing turns one number into twenty, and the minimum of twenty noisy numbers is "
                    "low even when every subgroup behaves identically. With 30 rows in a slice, an "
                    "accuracy of 0.73 and one of 0.93 are the same measurement."
                ),
                Para(
                    "This is why the panel has a *minimum rows* control: slices below it are held out "
                    "of the \"worst first\" ranking instead of being allowed to top a list read as a "
                    "list of problems. Set it to something you would actually act on."
                ),
            )),
            Callout(CALLOUT_TIP, "What to slice by first", (
                Bullets((
                    "Anything that changed between training and deployment: sensor, site, firmware, season.",
                    "Anything the annotation process depended on: annotator, batch, review pass.",
                    "Any physical property that plausibly changes difficulty: size, distance, exposure, occlusion.",
                    "The split column itself, as a sanity check that the pools are comparable.",
                )),
            )),
        ),
        see_also=("slice-metrics", "rates-and-uncertainty", "accuracy", "column-statistics"),
        keywords=("slice", "subgroup", "breakdown", "metadata", "binning", "cohort"),
    ),

    ManualPage(
        id="slice-metrics",
        title="Which metric to slice by",
        summary=(
            "Accuracy over a slice answers \"does the model work here\". It does not answer \"does the "
            "model still confuse A with B here\", which is usually the real question. The per-class "
            "metrics -- recall, precision, false alarm rate, miss rate -- answer that, one class at a "
            "time."
        ),
        shown_in="Slices panel (Metric and Class dropdowns)",
        blocks=(
            Heading("The metrics available, and their denominators"),
            Table(
                headers=("Metric", "Formula", "Denominator"),
                rows=(
                    ("Accuracy", "$(n - \\mathrm{errors}) / n$", "every crop in the slice"),
                    ("Mean confidence", "$\\frac{1}{n}\\sum_i \\hat{p}_i$", "every crop in the slice"),
                    ("Recall (class $c$)", "$\\mathrm{TP}_c / n_c$", "crops of $c$ in the slice"),
                    ("Precision (class $c$)", "$\\mathrm{TP}_c / q_c$", "crops the model called $c$ in the slice"),
                    ("F1 (class $c$)", "$2\\mathrm{TP}_c / (2\\mathrm{TP}_c + \\mathrm{FP}_c + \\mathrm{FN}_c)$", "not a proportion -- no interval is drawn"),
                    ("False alarm (class $c$)", "$\\mathrm{FP}_c / (n - n_c)$", "crops that are *not* $c$"),
                    ("Miss rate (class $c$)", "$\\mathrm{FN}_c / n_c$", "crops of $c$ -- exactly $1 - \\mathrm{recall}$"),
                    ("Support / Predicted", "$n_c$ and $q_c$", "counts, not rates"),
                ),
                weights=(2, 3, 3),
            ),
            Para(
                "For the false alarm rate and the miss rate, a *high* value is the bad news. The panel "
                "knows this, so \"worst first\" still puts the problems at the top."
            ),
            Callout(CALLOUT_EXAMPLE, "Why the per-class view is the one that finds things", (
                Para(
                    "A defect class holding 5% of the rows triples its false alarm rate on one sensor. "
                    "Overall accuracy on that sensor moves by a point and a half -- indistinguishable "
                    "from noise in a table of twenty slices. The false alarm rate for that one class, "
                    "sliced by sensor, shows a bar three times the others."
                ),
            )),
            Callout(CALLOUT_WARNING, "An undefined metric is blank, not zero", (
                Para(
                    "Precision needs at least one prediction of the class in the slice; recall needs "
                    "at least one true crop of it. When the denominator is empty the app leaves the "
                    "cell empty rather than writing 0, which would plot as a real result at the bottom "
                    "of the axis and read as a catastrophic failure."
                ),
            )),
        ),
        see_also=("slice-breakdown", "precision-recall", "rates-and-uncertainty", "confusion-matrix"),
        keywords=("false alarm", "miss rate", "per class", "metric", "slice metric", "fpr"),
    ),

    ManualPage(
        id="slice-comparison",
        title="Comparing experiments across slices",
        summary=(
            "Two experiments sliced by the same column, side by side, is how you tell a real "
            "improvement from a redistribution. A model that gains two points overall by winning on "
            "the easy site and losing on the hard one is not an improvement in any sense that matters."
        ),
        shown_in="Slices panel (with several experiments selected)",
        blocks=(
            Para(
                "The slices are computed once from the dataset column and reused for every experiment, "
                "so each row of the table compares the same crops. The delta column is therefore a "
                "paired difference on identical data, which is far more sensitive than comparing two "
                "independent rates."
            ),
            HowToRead((
                ("A slice where both models agree", "the subgroup is not where the difference lives"),
                ("A slice with a large positive delta", "the candidate genuinely helps here"),
                ("A large negative delta on a small slice", "check $n$ first; then check whether that slice is the one that matters"),
                ("Deltas of opposite signs across slices", "a redistribution, not an improvement -- the overall number is hiding a trade"),
                ("A uniform delta everywhere", "a real, general improvement; this is the pattern you want"),
            )),
            Callout(CALLOUT_TIP, "The crop-level version of this question", (
                Para(
                    "The Compare tab asks the same question at the level of individual crops -- how "
                    "many did the candidate fix, how many did it break, and which ones. Slices tell "
                    "you *where* the change landed; the Compare tab lets you look at *what* changed."
                ),
            )),
        ),
        see_also=("slice-breakdown", "fixes-regressions", "per-class-change", "rates-and-uncertainty"),
        keywords=("compare slices", "delta", "regression", "subgroup comparison"),
    ),
)
