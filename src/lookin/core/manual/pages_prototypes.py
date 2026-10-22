"""Chapter: the class centroids a metric-learning model carries."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("prototypes", "Prototypes and subcenters")

PAGES = (
    ManualPage(
        id="prototypes",
        title="What a prototype is",
        summary=(
            "A prototype is a single vector standing for a whole class in embedding space -- the "
            "weight vector of a metric-learning head, or a centroid estimated from the data. A crop is "
            "classified by which prototype it is closest to, so the geometry of the prototypes *is* "
            "the decision rule."
        ),
        shown_in="Prototypes tab, Embeddings tab (prototype overlay)",
        blocks=(
            Heading("Where they come from"),
            Table(
                headers=("Source", "What it is", "What it can be used for"),
                rows=(
                    ("model", "the head's own weight vectors, imported from a sidecar", "everything, including the scored diagnostics"),
                    ("kmeans", "per-class k-means centroids computed here, over the loaded pool", "looking, and only looking"),
                ),
                weights=(1, 3, 3),
            ),
            Callout(CALLOUT_WARNING, "Why kmeans prototypes are visual-only", (
                Para(
                    "A k-means centroid is fitted on the same crops, with the same labels, that a "
                    "diagnostic would then score it against. Purity computed that way measures how "
                    "well k-means clustered, not how well the model learned -- it is circular, and it "
                    "flatters. The app therefore refuses the scored diagnostics (compactness, "
                    "separation, noise ranking) on a kmeans source and offers them only on imported "
                    "prototypes."
                ),
            )),
            Heading("Network or dataset, on the Embeddings map"),
            Para(
                "The network names its classes its own way -- its score columns -- and its prototypes "
                "carry those names. Before they are drawn, each is renamed to the dataset class the "
                "Sources tab's class mapping assigns its column, so a relabelled class keeps its "
                "prototype. Several network classes mapped onto one dataset class become subcenters "
                "of it."
            ),
            Para(
                "The map can instead draw prototypes built from the dataset's own labels: the mean "
                "embedding of each class, or the k-means estimate when the Prototypes tab made one "
                "active. These cover every class the dataset has, whatever the network was trained "
                "on, which is why the map uses them by default whenever the network's prototypes "
                "miss a class. Like any kmeans source, they are for looking only."
            ),
            Heading("Subcenters"),
            Para(
                "A class may carry $K$ prototypes rather than one. That is how a head copes with a "
                "class that has several distinct appearances: one label, several centres. The app "
                "writes them as *class#k* -- so *cat#0* and *cat#1* are two subcenters of "
                "the same class -- and most of the tables in this chapter have one row per subcenter, "
                "because two subcenters of the same class routinely behave very differently."
            ),
            Heading("Assignment"),
            Equation(
                r"(\hat{c}_i, \hat{k}_i) = \arg\max_{c, k} \; \tilde{u}_i \cdot P_{c,k}",
                where=(
                    (r"P_{c,k}", "the $k$-th prototype of class $c$, L2-normalised"),
                    (r"\tilde{u}_i", "the normalised embedding of crop $i$"),
                ),
                note="Every crop is assigned to its single closest prototype across ALL classes -- not restricted to its own label. That is what exposes one class's subcenter quietly absorbing another class's crops.",
            ),
        ),
        see_also=("embedding-space", "compactness-separation", "prototype-roles", "prototype-similarity"),
        keywords=("prototype", "centroid", "subcenter", "arcface", "head", "kmeans", "source"),
    ),

    ManualPage(
        id="compactness-separation",
        title="Compactness, separation and margin",
        summary=(
            "Compactness is how tightly a class's crops cluster around its own prototype; separation "
            "is how close that prototype sits to the nearest *other* class's. Their difference is the "
            "margin -- the headroom the model has on that class, and the single most useful number in "
            "this tab."
        ),
        shown_in="Prototypes tab (\"Compactness & separation\"), Cleaning tab (noisy subcenters)",
        blocks=(
            Equation(
                r"\mathrm{compactness}(c, k) = \frac{1}{|A_{c,k}|} \sum_{i \in A_{c,k}} \tilde{u}_i \cdot P_{c,k}",
                where=(
                    (r"A_{c,k}", "the crops assigned to this prototype"),
                    (r"P_{c,k}", "the prototype vector itself"),
                ),
                note="Mean cosine similarity of the assigned crops to their prototype. Higher is tighter.",
            ),
            Equation(
                r"\mathrm{separation}(c, k) = \max_{c' \neq c, \; k'} \; P_{c,k} \cdot P_{c',k'}",
                note="Similarity to the nearest prototype of a DIFFERENT class -- the class's own other subcenters are excluded, since a class being close to itself is not a risk. Lower is safer.",
            ),
            Equation(
                r"\mathrm{margin}(c, k) = \mathrm{compactness}(c, k) - \mathrm{separation}(c, k)",
                note="Positive: the class's own crops sit closer to it than the nearest rival prototype does. Negative: the rival prototype is closer than the class's own crops -- the class is being swallowed.",
            ),
            HowToRead((
                ("High compactness, low separation", "a healthy class: tight, and alone"),
                ("Low compactness", "the crops assigned here are not alike; one label over several appearances, or a polluted subcenter"),
                ("High separation", "another class's prototype is sitting on top of this one -- the pair to look at in the similarity matrix"),
                ("Margin near zero", "a class with no headroom: small changes in the input flip its predictions"),
                ("Margin negative", "structurally broken -- expect this class's errors to be systematic, not random"),
                ("Purity", "the share of the crops assigned here that really belong to this class"),
            )),
            Heading("The compactness-versus-separation scatter"),
            Para(
                "One point per class, separation on the x axis, compactness on the y, both on the "
                "same cosine scale so the diagonal is where the two are equal. The colour is the "
                "margin, on a fixed scale rather than the data's own range -- red means a genuinely "
                "negative margin, not merely the lowest in this set."
            ),
            HowToRead((
                ("Above the diagonal", "positive margin: the class's crops are closer to it than any rival prototype is"),
                ("Below the diagonal", "negative margin: confusion risk, whatever the class's other numbers say"),
                ("Top left", "the healthy corner -- tight, and far from everything else"),
                ("Bottom left", "diffuse but alone: the class is not learned well, but nothing is stealing it"),
                ("Top right", "tight but stuck to another class: confusion likely, and it will be systematic"),
                ("The bar chart beside it", "the same margins sorted ascending, so the classes most at risk come first"),
            )),
            Heading("Per class or per subcenter"),
            Para(
                "The per-*class* view reduces over subcenters -- compactness takes the best matching "
                "subcenter per crop, separation uses the class's mean vector -- to give one row per "
                "class. The per-*subcenter* view reduces nothing. A class whose averaged row looks "
                "fine can hold one healthy subcenter and one that is pure noise, and only the second "
                "view shows it."
            ),
            Callout(CALLOUT_PITFALL, "These are similarities, not distances", (
                Para(
                    "Everything here is a cosine similarity, so *higher means closer*. A high "
                    "separation is bad news, not good. The margin column is signed precisely so that "
                    "one consistent rule applies: bigger is better."
                ),
            )),
        ),
        see_also=("prototypes", "prototype-roles", "noisy-subcenters", "class-overlap"),
        keywords=("compactness", "separation", "margin", "purity", "tight", "headroom"),
    ),

    ManualPage(
        id="prototype-roles",
        title="The role of each prototype",
        summary=(
            "For each prototype, which true classes actually land on it. A prototype is supposed to "
            "attract its own class; what it attracts *besides* its own class is where the model's "
            "confusions are built."
        ),
        shown_in="Prototypes tab (\"Role of each prototype\")",
        blocks=(
            Para(
                "Every crop is assigned to its globally closest prototype. Grouping those assignments "
                "by prototype gives, for each one: how many crops it attracted in total, how many of "
                "them were its own class, and which other classes made up the rest."
            ),
            Equation(
                r"\mathrm{purity}(c, k) = \frac{\left| \{ i \in A_{c,k} \;:\; y_i = c \} \right|}{|A_{c,k}|}",
                where=((r"A_{c,k}", "the crops assigned to this prototype"),),
                note="The share of what a prototype attracted that genuinely belongs to it.",
            ),
            HowToRead((
                ("Total assigned", "how much of the pool this prototype is responsible for"),
                ("Own", "of those, how many really are this class"),
                ("Other classes listed", "the specific confusions this prototype creates, largest first"),
                ("Purity near 1", "the prototype does its job and attracts nothing else"),
                ("Purity well below 1", "it is absorbing another class -- read which one, it is named"),
                ("Total near zero", "a dead prototype: a subcenter the training never used, or one that only captured outliers"),
            )),
            Callout(CALLOUT_TIP, "The asymmetry is the finding", (
                Para(
                    "If *dog#1* attracts 200 cats but *cat#0* attracts no dogs, the confusion runs one "
                    "way. That is usually a prior or a boundary problem rather than a representation "
                    "one, and it is often fixable without retraining -- by a per-class threshold, for "
                    "instance."
                ),
            )),
            Callout(CALLOUT_WARNING, "On a kmeans source this is descriptive only", (
                Para(
                    "With kmeans prototypes, these counts describe how k-means partitioned the pool. "
                    "They are still useful for looking -- a centroid that mixes two classes is worth "
                    "seeing -- but they are not a statement about the model."
                ),
            )),
        ),
        see_also=("prototypes", "compactness-separation", "noisy-subcenters", "confusion-matrix"),
        keywords=("role", "composition", "purity", "assigned", "attracts", "contamination"),
    ),

    ManualPage(
        id="prototype-similarity",
        title="Prototype similarity matrix",
        summary=(
            "The cosine similarity between every pair of prototypes. Where the class overlap matrix "
            "measures how the *crops* of two classes sit relative to each other, this measures how the "
            "model's own *definitions* of them sit -- which is the decision rule itself."
        ),
        shown_in="Prototypes tab (similarity matrix and top confusion pairs)",
        blocks=(
            Equation(
                r"G_{(c,k), (c',k')} = P_{c,k} \cdot P_{c',k'}",
                note="A Gram matrix of the L2-normalised prototypes. The diagonal is 1 by construction and carries no information.",
            ),
            Heading("Ordering the classes"),
            Para(
                "With many classes the matrix only becomes readable once the rows are ordered by "
                "something meaningful. Two orders are offered:"
            ),
            Bullets((
                "*margin* -- classes with the least headroom first, so the rows that matter are at the top.",
                "*alpha* -- alphabetical, for finding a specific class rather than the worst one.",
            )),
            HowToRead((
                ("A bright off-diagonal cell", "two prototypes the model has placed almost on top of each other"),
                ("A bright block", "a family of classes the model treats as one thing -- a taxonomy problem, not a labelling one"),
                ("A bright cell within one class's subcenters", "normal: subcenters of a class are supposed to be related"),
                ("A dark matrix overall", "well-separated prototypes -- the geometry is healthy whatever the metrics say"),
                ("The top pairs list", "the same information, ranked, for when the matrix is too large to scan"),
            )),
            Callout(CALLOUT_TIP, "Taxonomy hints", (
                Para(
                    "Two readings of this matrix are automated. A pair of prototypes above 0.8 "
                    "similarity suggests a *merge*: the model sees one class where the taxonomy claims "
                    "two. A class whose own subcenters fall below 0.5 similarity to each other "
                    "suggests a *split*: one label over two visually distinct populations."
                ),
                Para(
                    "When *most* classes look splittable, the app says so rather than listing them all "
                    "-- that is not a statement about any class, it means $K$ is higher than this "
                    "dataset supports and the extra centres are splitting noise."
                ),
            )),
        ),
        see_also=("prototypes", "class-overlap", "compactness-separation", "assignment-matrix"),
        keywords=("similarity matrix", "gram", "confusion pairs", "taxonomy", "merge", "split", "ordering"),
    ),

    ManualPage(
        id="assignment-matrix",
        title="Assignment matrix",
        summary=(
            "Rows are true classes, columns are prototypes, and each cell counts the crops of that "
            "class that landed on that prototype. It is a confusion matrix of the embedding space -- "
            "computed with no classifier head involved, at subcenter resolution."
        ),
        shown_in="Prototypes tab (assignment matrix)",
        blocks=(
            Equation(
                r"A_{a, (c,k)} = \sum_{i=1}^{N} \mathbb{1}\!\left[\, y_i = a \,\right] \cdot "
                r"\mathbb{1}\!\left[\, (\hat{c}_i, \hat{k}_i) = (c, k) \,\right]",
                where=(
                    (r"a", "the row: the crop's true class"),
                    (r"(c,k)", "the column: the prototype it was assigned to"),
                ),
            ),
            HowToRead((
                ("Mass on the own-class columns", "the embedding places this class where it belongs"),
                ("A row split across two subcenters of its own class", "normal and healthy -- that is what subcenters are for"),
                ("A row leaking into another class's column", "the embedding, not the head, is confusing them"),
                ("A column collecting many rows", "a prototype acting as a sink -- often a degenerate centre that captured whatever fit nowhere else"),
                ("An empty column", "an unused prototype"),
            )),
            Callout(CALLOUT_TIP, "Compare it against the confusion matrix", (
                Para(
                    "This matrix and the Metrics tab's confusion matrix answer the same question at "
                    "two different depths. Where they agree, the confusion is built into the "
                    "representation and retraining the head will not help. Where the confusion matrix "
                    "is worse than this one, the embedding separates the classes and the head is "
                    "losing it -- a much cheaper problem to fix."
                ),
            )),
        ),
        see_also=("prototype-roles", "confusion-matrix", "prototypes", "prototype-similarity"),
        keywords=("assignment", "matrix", "sink", "leak", "subcenter", "nearest prototype"),
    ),

    ManualPage(
        id="case-studies",
        title="Case studies: nearest and farthest, impostors",
        summary=(
            "For one class or one subcenter: the crops that sit closest to its prototype, the ones "
            "that sit farthest from it while still being labelled with it, and the crops of *other* "
            "classes that sit closest to it. Three lists that turn a suspicious number into something "
            "you can look at."
        ),
        shown_in="Prototypes tab (\"Neighbor explorer\", case study dropdown)",
        blocks=(
            Table(
                headers=("List", "What it holds", "What it is for"),
                rows=(
                    ("Nearest", "own-class crops with the highest similarity to the prototype", "what the model thinks this class *is* -- a definition you can inspect"),
                    ("Farthest", "own-class crops with the lowest similarity", "mislabels, edge cases and the second population hiding under one label"),
                    ("Impostors", "crops of other classes closest to this prototype", "exactly the crops that will be predicted as this class -- the confusions, before they happen"),
                ),
                weights=(1, 3, 4),
            ),
            Callout(CALLOUT_TIP, "Read the farthest list first", (
                Para(
                    "The nearest list confirms what you already believe. The farthest list is where "
                    "the findings are: a mislabelled crop, a crop of a different modality, a class "
                    "whose definition drifted between annotation batches. It is the cheapest "
                    "label-noise audit in the app, and it needs no scores at all."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Low similarity is not proof of a bad label", (
                Para(
                    "A crop can be correctly labelled and still sit far from its prototype: unusual "
                    "pose, heavy occlusion, a rare but legitimate variant. Look before relabelling, "
                    "and use the nearest-neighbour view to see whether it is alone or part of a "
                    "pocket."
                ),
            )),
        ),
        see_also=("nearest-neighbours", "compactness-separation", "label-noise", "prototype-roles"),
        keywords=("case study", "nearest", "farthest", "impostor", "explorer", "examples"),
    ),
)
