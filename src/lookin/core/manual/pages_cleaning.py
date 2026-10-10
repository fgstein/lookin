"""Chapter: finding label noise, and measuring what fixing it bought."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("cleaning", "Dataset cleaning")

PAGES = (
    ManualPage(
        id="label-noise",
        title="Label noise, and how to hunt it",
        summary=(
            "Some share of every annotated dataset is wrong. On a mature model, a large part of the "
            "remaining error is annotation error rather than model error -- and the only way to know "
            "which is to look. This tab ranks crops by how likely they are to be mislabelled, so that "
            "looking is affordable."
        ),
        shown_in="Cleaning tab",
        blocks=(
            Heading("Why it is worth the hours"),
            Para(
                "A mislabelled crop costs three times: it teaches the model something false, it "
                "counts as an error in every metric even when the model was right, and it corrupts "
                "whatever threshold or comparison you derive from those metrics. On a model that is "
                "already good, the last few points of measured accuracy are often entirely annotation "
                "noise."
            ),
            Heading("The strategies, and what each one needs"),
            Table(
                headers=("Strategy", "Needs", "Best at"),
                rows=(
                    ("All errors", "one results CSV", "a starting point; no ranking, just every mistake"),
                    ("Score margin", "one results CSV", "confident disagreements with the annotation"),
                    ("Consensus", "several results CSVs", "annotation errors, with the strongest evidence of the lot"),
                    ("Embedding flags", "an embeddings sidecar", "crops sitting in another class's territory, no scores needed"),
                    ("Noisy subcenters", "prototypes and embeddings", "whole pockets of contamination at once"),
                    ("Near-duplicates", "an embeddings sidecar", "duplicates and train/test leakage"),
                ),
                weights=(2, 2, 4),
            ),
            Callout(CALLOUT_TIP, "A workable order", (
                Bullets((
                    "Run near-duplicates first: leakage invalidates every other number, and finding it late means redoing the work.",
                    "Then consensus if you have several experiments -- it is the highest-precision signal available.",
                    "Then score margin, which needs only one experiment and finds the confident disagreements.",
                    "Then the embedding strategies, which find what the scores cannot: pockets the model learned *as* a cluster.",
                    "Measure with \"Did it work?\" before deciding whether a second pass is worth it.",
                ), numbered=True),
            )),
            Callout(CALLOUT_WARNING, "Reviewing only what the model got wrong biases the dataset", (
                Para(
                    "Every strategy here surfaces crops the model disagrees with. Correcting only "
                    "those makes the dataset agree with the model a little more than the truth "
                    "warrants, and the measured accuracy improves for a reason that is partly "
                    "circular. Keep a held-out split that is never cleaned this way, or review a "
                    "random sample alongside the ranked one -- that sample also tells you the true "
                    "noise rate."
                ),
            )),
        ),
        see_also=("margin-candidates", "consensus", "noisy-subcenters", "near-duplicates", "cleaning-impact"),
        keywords=("label noise", "cleaning", "mislabel", "annotation error", "candidates", "audit"),
    ),

    ManualPage(
        id="margin-candidates",
        title="Score margin",
        summary=(
            "Rank the crops by how much the model's scores prefer some other class over the one the "
            "crop is labelled with. A large positive margin on a confident model is the single best "
            "cheap signal of a label error, and it needs nothing but one results CSV."
        ),
        shown_in="Cleaning tab (Candidate source = score margin, or confident errors)",
        blocks=(
            Equation(
                r"\mathrm{margin}_i = \max_{c \neq y_i} p_{i,c} \; - \; p_{i, y_i}",
                where=(
                    (r"y_i", "the class the crop is *annotated* with"),
                    (r"p_{i,c}", "the score the model gives class $c$"),
                ),
                note="Positive means the model prefers another class to the annotation. The larger, the stronger the disagreement.",
            ),
            HowToRead((
                ("own score", "what the model gave the annotated class"),
                ("best other", "the class it preferred instead -- the suggested relabel"),
                ("other score", "what it gave that class"),
                ("margin", "the gap, and the ranking key"),
            )),
            Heading("Margin, not just wrongness"),
            Para(
                "A crop can be wrong by a hair or wrong by a mile. \"All errors\" lists both without "
                "distinction; the margin sorts them, and the top of that list is where mislabels "
                "concentrate. The *confident errors* variant additionally keeps only crops the model "
                "actually got wrong -- a narrower, higher-precision version of the same question."
            ),
            Callout(CALLOUT_PITFALL, "It finds the model's blind spots too", (
                Para(
                    "A high margin means the model disagrees confidently. That is evidence about the "
                    "annotation *only* to the extent the model is trustworthy on that class. A class "
                    "the model never learned will produce confident disagreements that are simply "
                    "wrong -- check the class's own metrics before trusting its candidates, and prefer "
                    "consensus when several experiments are available."
                ),
            )),
            Callout(CALLOUT_WARNING, "On a training pool, memorisation hides the errors", (
                Para(
                    "A model evaluated on crops it was trained on has memorised many of its "
                    "mislabelled examples and scores them confidently correct, so their margin is "
                    "negative and they never surface. On a training pool, the consensus and embedding "
                    "strategies are the ones that still work."
                ),
            )),
        ),
        see_also=("consensus", "label-noise", "predictions", "confidence-histogram"),
        keywords=("margin", "score margin", "confident errors", "suggestion", "disagreement"),
    ),

    ManualPage(
        id="consensus",
        title="Consensus across experiments",
        summary=(
            "Crops that several independently-trained models get wrong *in the same way*. One model "
            "disagreeing with an annotation is a model error as often as an annotation error; four "
            "models independently predicting the same other class is a much stronger claim."
        ),
        shown_in="Cleaning tab (Candidate source = consensus)",
        blocks=(
            Heading("The rule"),
            Para(
                "For each crop, collect the prediction of every loaded experiment. Keep the crops "
                "where at least $m$ models are wrong, and rank by how many of them agree on the *same* "
                "replacement class:"
            ),
            Equation(
                r"\mathrm{agreement}_i = \max_{c \neq y_i} \; \left| \{\, e \;:\; \hat{y}^{(e)}_i = c \,\} \right|",
                where=(
                    (r"e", "one loaded experiment"),
                    (r"\hat{y}^{(e)}_i", "that experiment's prediction for crop $i$"),
                    (r"m", "the \"min models agreeing\" setting"),
                ),
                note="Ranked by agreement first, mean confidence as the tie-break: five models agreeing weakly beats two agreeing loudly.",
            ),
            HowToRead((
                ("models wrong", "how many of the loaded experiments disagreed with the annotation"),
                ("agreeing on", "the class they converged on -- the suggested relabel"),
                ("agreement", "how many agreed on it"),
                ("unanimous", "whether *every* experiment agreed; these are the strongest candidates in the app"),
            )),
            Callout(CALLOUT_TIP, "Why this is the strongest signal available", (
                Para(
                    "Unlike every other strategy here, its evidence does not come from one model's "
                    "memory of the crop. Independently trained models make independent mistakes; they "
                    "do not independently make the *same* mistake unless something in the data is "
                    "telling them to. That is what makes it the strategy to reach for on a training "
                    "pool, where a single model's confidence proves nothing."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Independence is an assumption, not a fact", (
                Para(
                    "Models that share an architecture, a pretrained backbone, or -- worst -- a "
                    "training set, are not independent. Their agreement then reflects a shared bias as "
                    "readily as a labelling error. Consensus across genuinely different runs is "
                    "evidence; consensus across three checkpoints of one run is one model counted "
                    "three times."
                ),
            )),
            Callout(CALLOUT_TIP, "Wide disagreement is a different finding", (
                Para(
                    "Models wrong in *different* directions do not mean a mislabel; they mean a "
                    "genuinely ambiguous crop. Those are reported in the detail column rather than "
                    "ranked up, because the right action for them is usually to exclude the crop, not "
                    "to relabel it."
                ),
            )),
        ),
        see_also=("margin-candidates", "label-noise", "fixes-regressions"),
        keywords=("consensus", "ensemble", "agreement", "unanimous", "vote", "several models"),
    ),

    ManualPage(
        id="embedding-flags",
        title="Flags from the embedding",
        summary=(
            "Two score-free tests, run in embedding space against the model's prototypes: is this crop "
            "closer to another class than to its own, and is it in the bottom few percent of its own "
            "class's similarity? Both find crops the scores can miss entirely."
        ),
        shown_in="Cleaning tab (Candidate source = embedding flags)",
        blocks=(
            Heading("Strategy 1 -- closer to another class"),
            Equation(
                r"\max_{c \neq y_i, \, k} \; \tilde{u}_i \cdot P_{c,k} \;\; > \;\; "
                r"\max_{k} \; \tilde{u}_i \cdot P_{y_i, k} \; + \; \delta",
                where=(
                    (r"P_{c,k}", "the prototypes of class $c$"),
                    (r"\delta", "the \"min margin\" setting, 0 by default"),
                ),
                note="The crop sits nearer to another class's prototype than to its own label's. Raising the margin makes the test stricter.",
            ),
            Heading("Strategy 2 -- lowest percentile within the class"),
            Para(
                "For each class, compute every crop's similarity to its own prototype and flag the "
                "bottom $q$ percent. This catches the crops that are not claimed by another class but "
                "are simply nowhere near their own -- corrupted crops, wrong modality, or a second "
                "population under one label."
            ),
            Equation(
                r"\max_{k} \; \tilde{u}_i \cdot P_{y_i, k} \;\; \leq \;\; Q_q\!\left( \{\, \mathrm{own\ similarity} \;:\; y_j = y_i \,\} \right)",
                where=((r"Q_q", "the $q$-th percentile within the crop's own class"),),
                note="A relative test, per class: a class that is diffuse overall is not punished for it.",
            ),
            Callout(CALLOUT_TIP, "Both strategies can run at once", (
                Para(
                    "A crop flagged by both is a strong candidate: it is far from its own class *and* "
                    "claimed by another. The reason column names which tests fired."
                ),
            )),
            Callout(CALLOUT_WARNING, "Needs the model's own prototypes", (
                Para(
                    "These tests are scored diagnostics, so they require imported prototypes. Run "
                    "against k-means centroids fitted on the same labels they would judge, they would "
                    "be measuring their own fit -- the app refuses rather than producing a flattering "
                    "number."
                ),
            )),
            Callout(CALLOUT_PITFALL, "The percentile always flags something", (
                Para(
                    "By construction, the bottom 2% of every class is 2% of every class -- even a "
                    "perfectly clean dataset produces a full list. Read the similarity values, not "
                    "just the membership, and stop when the crops stop looking wrong."
                ),
            )),
        ),
        see_also=("compactness-separation", "prototypes", "noisy-subcenters", "nearest-neighbours"),
        keywords=("embedding flags", "percentile", "closer to other class", "own similarity", "flagged"),
    ),

    ManualPage(
        id="noisy-subcenters",
        title="Noisy subcenters",
        summary=(
            "Rank each prototype by how much junk it appears to have collected, and carry its members "
            "along so the ranking is directly actionable. Where the other strategies find crops one at "
            "a time, this one finds whole pockets of contamination in a single row."
        ),
        shown_in="Cleaning tab (\"Noisy subcenters\")",
        blocks=(
            Equation(
                r"\mathrm{noise}(c, k) = \left( 1 - \mathrm{purity} \right) + "
                r"\max\!\left( 0, \; \mathrm{separation} - \mathrm{compactness} \right)",
                where=(
                    (r"\mathrm{purity}", "share of the crops assigned here that really are class $c$"),
                    (r"\mathrm{compactness}", "mean similarity of those crops to this prototype"),
                    (r"\mathrm{separation}", "similarity to the nearest prototype of another class"),
                ),
                note="Blends the three ways a subcenter can be suspect. Higher is worse; the table is sorted by it.",
            ),
            Heading("What each term contributes"),
            Bullets((
                "*Low purity* -- the subcenter is absorbing other classes' crops. This term is the share it absorbed.",
                "*Separation above compactness* -- another class's prototype is closer to this one than its own crops are. A negative margin, clipped at zero so a healthy subcenter contributes nothing.",
                "A near-empty subcenter cannot be scored on purity and is reported by its emptiness instead: it is usually a degenerate centre that captured whatever fit nowhere else.",
            )),
            Callout(CALLOUT_TIP, "Why pockets matter more than single crops", (
                Para(
                    "A subcenter holding 300 crops with a purity of 0.6 is 120 suspect crops that "
                    "arrived together and that almost certainly share a cause -- one annotation batch, "
                    "one site, one confusable sub-type. Reviewing them as a group is faster than "
                    "reviewing 120 unrelated crops, and the fix usually generalises beyond the crops "
                    "you looked at."
                ),
            )),
            Callout(CALLOUT_PITFALL, "A low-purity subcenter is not always mislabelled data", (
                Para(
                    "It can equally be a genuinely hard region where two classes really do look alike. "
                    "The members list is there so you can tell: open a dozen crops before acting. If "
                    "they look correctly labelled, the finding is a taxonomy problem, not a labelling "
                    "one -- see the prototype similarity page."
                ),
            )),
        ),
        see_also=("compactness-separation", "prototype-roles", "embedding-flags", "prototype-similarity"),
        keywords=("noisy", "subcenter", "noise score", "pocket", "contamination", "purity"),
    ),

    ManualPage(
        id="near-duplicates",
        title="Near-duplicates and leakage",
        summary=(
            "Pairs of crops whose embeddings are nearly identical. Duplicates inflate a class's "
            "apparent support; a duplicate pair straddling train and test is leakage -- the finding "
            "that quietly invalidates an entire evaluation."
        ),
        shown_in="Cleaning tab (Candidate source = near-duplicates)",
        blocks=(
            Equation(
                r"\left\{\, (i, j) \;:\; y_i = y_j, \;\; \tilde{u}_i \cdot \tilde{u}_j \geq \tau \,\right\}",
                where=((r"\tau", "the similarity threshold, 0.98 by default"),),
                note="Pairs are searched within a class. A duplicate carrying two different labels is a labelling conflict, which the margin strategies already surface.",
            ),
            Heading("Why it is blocked by class"),
            Para(
                "Comparing every crop against every other is quadratic in the whole pool; comparing "
                "within each class is quadratic in the class sizes, which is dramatically cheaper and "
                "loses nothing that matters here."
            ),
            Heading("The leakage check"),
            Para(
                "With a split column available, the *cross-split only* option keeps just the pairs "
                "whose two crops are in different splits. Each such pair is a crop the model saw in "
                "training and is now being scored on. Even a small number of them can lift a reported "
                "accuracy by several points -- and the effect is worst on exactly the rare classes "
                "everyone quotes."
            ),
            Callout(CALLOUT_WARNING, "Run this before believing an unusually good number", (
                Para(
                    "A ROC-AUC of 0.999 or a jump of several points from a small change deserve this "
                    "check before they deserve a celebration. Leakage is the most common explanation "
                    "for both, and it is the cheapest one to rule out."
                ),
            )),
            Callout(CALLOUT_TIP, "What to do with a duplicate pair", (
                Bullets((
                    "Within one split: usually drop one of the two, unless the duplication reflects a real frequency you want the model to learn.",
                    "Across splits: remove the crop from the *test* side, never from the training side, or the evaluation stops being comparable to previous ones.",
                    "Either way, record the decision as a review entry so the next run inherits it.",
                )),
            )),
        ),
        see_also=("nearest-neighbours", "embedding-space", "label-noise", "dataset-join"),
        keywords=("duplicate", "leakage", "train test", "similarity threshold", "copy"),
    ),

    ManualPage(
        id="review-versions",
        title="Review versions",
        summary=(
            "A review version is a named set of decisions -- relabel, exclude, confirm, uncertain -- "
            "kept separately from the dataset CSV. Versions can inherit from each other, so a second "
            "pass builds on the first instead of replacing it, and any version can be switched on or "
            "off to see its effect."
        ),
        shown_in="Cleaning tab (\"Review version\", \"What changed between two versions\")",
        blocks=(
            Heading("Why decisions are not written into the dataset"),
            Bullets((
                "A decision is reversible; an edited CSV is not.",
                "Two reviewers, or two passes with different criteria, can be compared only if they are stored separately.",
                "The before/after comparison that says whether the pass was worth anything requires both states to exist at once.",
            )),
            Para(
                "A version inherits its parent's decisions and stores only what it changes, so a "
                "second pass is a short list of differences rather than a copy. Exporting for "
                "training *freezes* a version: it stops accepting decisions, so the run trained "
                "from it stays identifiable afterwards."
            ),
            Heading("The four decisions"),
            Table(
                headers=("Decision", "Effect when the overlay is on"),
                rows=(
                    ("Relabel", "the crop's true class becomes the new one, for every metric"),
                    ("Exclude", "the crop is dropped from the pool entirely"),
                    ("Confirm", "the annotation is recorded as checked and correct; nothing changes numerically"),
                    ("Uncertain", "recorded as looked at and unresolved; nothing changes numerically"),
                ),
                weights=(1, 4),
            ),
            Para(
                "*Confirm* is not a no-op in practice: it is what lets the next pass skip crops that "
                "have already been examined, and what makes the strategy precision figures meaningful "
                "-- a flag that was checked and found correct is evidence about the strategy."
            ),
            Callout(CALLOUT_TIP, "The overlay switch is the whole point", (
                Para(
                    "With the overlay off you see the dataset as annotated; with it on you see it as "
                    "reviewed. Every metric in the app follows that switch, so \"what did this pass "
                    "buy?\" is answered by flipping it -- which is exactly what the impact table "
                    "automates."
                ),
                Para(
                    "A version can also be *pinned*: the metrics keep being scored against that one "
                    "while you carry on reviewing into another, so the number under you does not "
                    "move while you work."
                ),
            )),
        ),
        see_also=("cleaning-impact", "label-noise", "export-for-training"),
        keywords=("review", "version", "relabel", "exclude", "overlay", "decisions", "pass"),
    ),

    ManualPage(
        id="cleaning-impact",
        title="Did it work? Measuring a cleaning pass",
        summary=(
            "\"I cleaned 800 crops\" is not a result; \"+2.1 points of macro F1\" is. The impact table "
            "scores the same experiment twice -- once on the raw labels, once through the review "
            "overlay -- and also reports, per strategy, what share of its flags became a real "
            "correction."
        ),
        shown_in="Cleaning tab (\"Did it work?\")",
        blocks=(
            Equation(
                r"\Delta m = m_{\mathrm{after}} - m_{\mathrm{before}}",
                where=((r"m", "any metric: accuracy, balanced accuracy, macro F1, weighted F1, MCC, macro ROC-AUC"),),
                note="The same experiment, the same scores, two different label sets. Nothing about the model changed.",
            ),
            Heading("Strategy precision"),
            Equation(
                r"\mathrm{precision}(s) = \frac{\left| \mathrm{flags\ of\ } s \mathrm{\ that\ became\ a\ relabel\ or\ an\ exclusion} \right|}"
                r"{\left| \mathrm{flags\ of\ } s \mathrm{\ that\ were\ reviewed} \right|}",
                where=((r"s", "one candidate strategy"),),
                note="What share of what a strategy proposed turned out to be a real correction rather than a confirmation. This is what tells you which strategy to keep running and which to drop.",
            ),
            Callout(CALLOUT_WARNING, "This measures the labels, not the model", (
                Para(
                    "The accuracy went up because the labels changed, not because the model improved. "
                    "The right reading is: *this much of the measured error was annotation error*. It "
                    "is a statement about the dataset and about how much headroom the metric had -- "
                    "not a model improvement, and not something to report as one."
                ),
            )),
            Callout(CALLOUT_PITFALL, "The optimism of reviewing only flagged crops", (
                Para(
                    "The reviewed crops were selected because a model disagreed with them, so "
                    "correcting them moves the metrics more than correcting a random sample of the "
                    "same size would. The delta is real, but it is an upper bound on what a full "
                    "re-annotation would give -- and part of it is the dataset being pulled towards "
                    "the model."
                ),
            )),
            Callout(CALLOUT_TIP, "When to stop cleaning", (
                Para(
                    "Watch the strategy precision across passes. When a strategy's flags stop turning "
                    "into corrections, the easy noise is gone; when the accuracy delta of a pass "
                    "approaches the width of the confidence interval on that accuracy, the pass is no "
                    "longer measurable and the effort belongs elsewhere."
                ),
            )),
        ),
        see_also=("review-versions", "label-noise", "rates-and-uncertainty", "fixes-regressions"),
        keywords=("impact", "did it work", "before after", "delta", "strategy precision", "worth it"),
    ),

    ManualPage(
        id="export-for-training",
        title="Exporting a reviewed dataset",
        summary=(
            "A review version can be written back out as a CSV with the corrected labels applied, for "
            "the next training run. The export is deliberately round-trippable, and it refuses to "
            "overwrite the dataset it was derived from."
        ),
        shown_in="Cleaning tab (\"Export for training\")",
        blocks=(
            Heading("What the file holds"),
            Para(
                "Your dataset CSV, back out unchanged -- every row, every column, in order -- plus "
                "two columns per review version: the reviewed label and the decision's status. A "
                "training pipeline filters the status column to drop the exclusions and reads the "
                "reviewed label instead of the original one. The source CSV is re-read in full, so "
                "a row filter left on the Data tab cannot truncate the export."
            ),
            Bullets((
                "*Relabelled crops* carry their new class in the label column.",
                "*Excluded crops* are kept by default, marked, so the file still describes the whole pool and can be re-imported without losing the decision. Dropping them is available and is not round-trippable.",
                "*The reason column* records why each decision was made -- worth keeping, since the next reviewer is usually you, months later.",
            )),
            Callout(CALLOUT_WARNING, "It will not write over its own source", (
                Para(
                    "An export aimed at the dataset CSV it was derived from, or at one of its "
                    "sidecars, is refused. The decisions live in the review store; overwriting the "
                    "source would destroy the only copy of the original annotation and make the "
                    "before/after comparison impossible forever."
                ),
            )),
            Callout(CALLOUT_TIP, "What to keep with the file", (
                Para(
                    "Record which review version produced it and which experiments were loaded when "
                    "the candidates were generated. A cleaned dataset with no provenance is a dataset "
                    "nobody can audit -- and the first question about a surprising result is always "
                    "\"which labels was this trained on?\"."
                ),
            )),
        ),
        see_also=("review-versions", "cleaning-impact", "dataset-join"),
        keywords=("export", "training", "csv", "round trip", "provenance", "write"),
    ),
)
