"""Chapter: what the model's feature space looks like."""
from __future__ import annotations

from lookin.core.embedding_analysis import (
    OVERLAP_MAX_PER_CLASS, PROBE_FOLDS, PROBE_MAX_PER_CLASS, PROBE_MAX_POINTS,
)
from lookin.core.embedding_compare import (
    AGREEMENT_K, COVERAGE_PERCENTILE, DOMAIN_FOLDS, KNN_POOL_MAX, PAIRWISE_FOLDS, PAIRWISE_POINTS,
    PROTOTYPE_TOLERANCE, STATS_MAX_POINTS,
)
from lookin.core.projection_plan import DEFAULT_DISPLAY_SIZE, DEFAULT_FIT_SIZE, MAX_DISPLAY_SIZE
from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("embeddings", "The embedding space")

PAGES = (
    ManualPage(
        id="embedding-space",
        title="Embeddings and cosine similarity",
        summary=(
            "An embedding is the vector a model computes for a crop just before the classifier head: "
            "its internal description of what it saw. Two crops the model considers alike have vectors "
            "pointing in a similar direction, and the standard measure of that is the cosine "
            "similarity."
        ),
        shown_in="Embeddings tab, Prototypes tab, Cleaning tab",
        blocks=(
            Equation(
                r"\mathrm{sim}(u, v) = \frac{u \cdot v}{\|u\| \, \|v\|} = \cos \theta",
                where=(
                    (r"u, v", "two embedding vectors"),
                    (r"\theta", "the angle between them"),
                ),
                note="Ranges from -1 (opposite) through 0 (unrelated directions) to 1 (identical direction).",
            ),
            Heading("Why the norm is divided out"),
            Para(
                "Only the *direction* of an embedding is meaningful for most metric-learning heads: "
                "the length encodes things like contrast or sharpness that the model was never asked "
                "to make semantic. Dividing by the norm removes it. When the vectors are already "
                "L2-normalised -- which ArcFace and the other angular-margin heads enforce, because "
                "they optimise on the unit hypersphere -- the cosine similarity is just the dot "
                "product, which is why the app normalises once on load and then uses plain matrix "
                "products everywhere."
            ),
            Equation(
                r"\tilde{u} = \frac{u}{\|u\|}, \qquad \mathrm{sim}(u, v) = \tilde{u} \cdot \tilde{v}",
                note="What the app stores, and what every similarity in the Embeddings, Prototypes and Cleaning tabs is computed from.",
            ),
            Heading("What similarity values mean in practice"),
            Table(
                headers=("Cosine similarity", "Typical reading on a trained model"),
                rows=(
                    ("above 0.98", "near-duplicate crops -- possibly the same object, possibly leakage"),
                    ("0.8 to 0.95", "same class, same conditions"),
                    ("0.5 to 0.8", "same class, different conditions, or two confusable classes"),
                    ("around 0", "unrelated"),
                ),
                weights=(2, 5),
                caption="Indicative only: the scale depends on the training objective. Read differences, not absolute values.",
            ),
            Callout(CALLOUT_WARNING, "Embeddings must come from the model you are judging", (
                Para(
                    "The app reads embeddings from a sidecar file attached to an experiment. If the "
                    "sidecar came from a different checkpoint than the scores, every page in this "
                    "chapter describes one model while the Metrics tab describes another -- and "
                    "nothing in the numbers will look wrong."
                ),
            )),
        ),
        see_also=("projection-2d", "nearest-neighbours", "class-overlap", "prototypes"),
        keywords=("embedding", "feature", "cosine", "similarity", "l2", "normalize", "vector"),
    ),

    ManualPage(
        id="projection-2d",
        title="Reading a 2D projection",
        summary=(
            "An embedding has hundreds of dimensions; a screen has two. Every projection method "
            "therefore throws almost everything away, and each one chooses *what* to keep. Knowing "
            "which choice was made is the difference between reading the map and inventing it."
        ),
        shown_in="Embeddings tab (\"2D projection of the embeddings\")",
        blocks=(
            Heading("What a projection is for"),
            Para(
                "The map is a navigation tool, not a measurement. It is excellent at showing you "
                "*where to look*: a knot of one class inside another, an isolated island, a smear "
                "between two clusters. Every quantitative claim about the geometry should be checked "
                "with a number -- the class overlap matrix, the prototype statistics, or a nearest "
                "neighbour query."
            ),
            HowToRead((
                ("A tight, isolated cluster", "the model has a clean, consistent representation of that group"),
                ("Two clusters touching", "confusable classes -- check the overlap matrix for the actual similarity"),
                ("A point deep inside another class's cluster", "either a mislabelled crop or a genuinely ambiguous one; open it"),
                ("A long smear", "a continuum -- usually a nuisance variable (distance, exposure, time) the model encodes"),
                ("Distance between two far-apart clusters", "*not* meaningful in t-SNE or UMAP -- see the pitfalls"),
            )),
            Heading("Sampling, and why the map holds still"),
            Para(
                "Fitting a projection is superlinear in the number of points, so the tab fits on a "
                "sample (the sample size field). The fitted model is then *kept*, so that crops added "
                "later -- an overlay, a query, a second experiment -- can be placed onto the same map "
                "instead of forcing a refit that would rearrange every existing point. PCA and UMAP "
                "can place new points; t-SNE cannot, and the tab says so rather than offering a choice "
                "it cannot honour."
            ),
            Callout(CALLOUT_PITFALL, "Four things a projection does not tell you", (
                Bullets((
                    "*Cluster sizes are not meaningful* in t-SNE and UMAP: both expand sparse regions and compress dense ones.",
                    "*Distances between clusters are not meaningful* in t-SNE: only local neighbourhoods are preserved.",
                    "*Apparent clusters can be artefacts* of the hyperparameters -- t-SNE will produce clusters from pure noise at some perplexity.",
                    "*The axes mean nothing at all* in t-SNE and UMAP; in PCA they are the two directions of greatest variance, which is a real but weak claim.",
                )),
            )),
            Callout(CALLOUT_PITFALL, "Points hide one another", (
                Para(
                    "Where clusters overlap, whichever group is drawn last covers the others, and a "
                    "small class can vanish entirely under a large one. The button beside each "
                    "legend entry draws that group on top of the rest; bring a second one forward "
                    "and it goes on top of the first. Hovering over overlapping points picks the "
                    "one that is drawn on top."
                ),
            )),
            Callout(CALLOUT_TIP, "Use two methods", (
                Para(
                    "A structure that appears in both PCA and UMAP is almost certainly in the data. A "
                    "structure that appears in only one is a property of that method. Switching the "
                    "dropdown is the cheapest check there is."
                ),
            )),
        ),
        see_also=("pca", "tsne", "umap", "class-overlap", "nearest-neighbours"),
        keywords=("projection", "2d", "map", "scatter", "dimensionality reduction", "sample", "draw order", "on top"),
    ),

    ManualPage(
        id="pca",
        title="PCA",
        summary=(
            "Principal component analysis: the linear projection onto the two directions along which "
            "the data varies most. It is exact, deterministic and fast, it preserves global structure "
            "-- and it will happily show a featureless blob when the structure that matters is not "
            "linear."
        ),
        shown_in="Embeddings tab (Projection = pca)",
        blocks=(
            Heading("What it computes"),
            Para(
                "Centre the data, then find the orthogonal directions of maximum variance. The first "
                "two are the axes of the plot:"
            ),
            Equation(
                r"w_1 = \arg\max_{\|w\|=1} \; \mathrm{Var}\!\left( X w \right), \qquad "
                r"w_2 = \arg\max_{\|w\|=1, \; w \perp w_1} \; \mathrm{Var}\!\left( X w \right)",
                where=(
                    (r"X", "the centred matrix of embeddings, one row per crop"),
                    (r"w_1, w_2", "the first two principal directions -- the plot's x and y axes"),
                ),
                note="Equivalently: the two leading eigenvectors of the covariance matrix.",
            ),
            Table(
                headers=("Property", "PCA"),
                rows=(
                    ("Deterministic", "yes -- same input, same picture, always"),
                    ("Preserves global distances", "yes, to the extent the data is linear"),
                    ("Preserves local neighbourhoods", "poorly"),
                    ("Places new points", "yes, exactly: it is a fixed linear map"),
                    ("Speed", "the fastest of the three"),
                    ("Hyperparameters", "none that matter here"),
                ),
                weights=(2, 3),
            ),
            Callout(CALLOUT_TIP, "Start here", (
                Para(
                    "PCA is the honest first look: nothing in the picture is invented by the method. "
                    "If the classes separate in PCA, they are linearly separable in the embedding "
                    "space, which is a strong and useful statement. If they do not, that is a question "
                    "for t-SNE or UMAP, not a verdict."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Two components of a high-dimensional space is very little", (
                Para(
                    "A 512-dimensional embedding whose variance is spread evenly leaves the first two "
                    "components explaining a few percent of it. The resulting blob is not evidence "
                    "that the model failed -- it is evidence that its structure is not concentrated in "
                    "two linear directions."
                ),
            )),
        ),
        see_also=("projection-2d", "tsne", "umap", "embedding-space"),
        keywords=("pca", "principal component", "linear", "variance", "eigenvector"),
    ),

    ManualPage(
        id="tsne",
        title="t-SNE",
        summary=(
            "t-SNE places points so that *neighbourhoods* are preserved: crops that are close in the "
            "embedding stay close on screen. It produces the most readable cluster pictures of the "
            "three methods, and it is the one whose global layout must never be read as a "
            "measurement."
        ),
        shown_in="Embeddings tab (Projection = tsne)",
        blocks=(
            Heading("What it optimises"),
            Para(
                "Pairwise similarities in the original space are turned into a probability "
                "distribution $P$ over pairs, the same is done in the 2D map to give $Q$, and the "
                "layout is moved until the two distributions agree:"
            ),
            Equation(
                r"\mathrm{KL}(P \,\|\, Q) = \sum_{i \neq j} p_{ij} \, \log \frac{p_{ij}}{q_{ij}}",
                where=(
                    (r"p_{ij}", "how likely crop $j$ is a neighbour of crop $i$, in the embedding"),
                    (r"q_{ij}", "the same, in the 2D map -- measured with a heavy-tailed Student-t kernel"),
                ),
                note="Minimised by gradient descent. The heavy tail is what lets distant clusters push apart without penalty, which is precisely why between-cluster distances mean nothing.",
            ),
            Heading("Perplexity"),
            Para(
                "The one hyperparameter that matters: roughly, how many neighbours each point tries to "
                "keep. The app sets it from the sample size (capped at 30, floored at 5), so it "
                "adapts to small selections rather than failing on them."
            ),
            Table(
                headers=("Property", "t-SNE"),
                rows=(
                    ("Deterministic", "no -- fixed seed here, so the same run reproduces, but a different sample gives a different picture"),
                    ("Preserves local neighbourhoods", "yes, this is what it optimises"),
                    ("Preserves global distances", "no. Between-cluster distance carries no information"),
                    ("Places new points", "no -- it learns no map, so adding points means refitting"),
                    ("Speed", "the slowest of the three"),
                ),
                weights=(2, 5),
            ),
            Callout(CALLOUT_PITFALL, "t-SNE will show you clusters that are not there", (
                Para(
                    "Run on pure noise at a small perplexity, t-SNE produces convincing, well-separated "
                    "clusters. Cluster *shape*, cluster *size* and the *space between* clusters are all "
                    "artefacts of the optimisation. What is trustworthy is: these points are "
                    "neighbours of each other."
                ),
            )),
        ),
        see_also=("projection-2d", "umap", "pca", "class-overlap"),
        keywords=("tsne", "t-sne", "perplexity", "kullback", "neighbourhood", "manifold"),
    ),

    ManualPage(
        id="umap",
        title="UMAP, and the spherical variant",
        summary=(
            "UMAP builds a neighbourhood graph of the embedding and lays that graph out in 2D. It "
            "keeps more global structure than t-SNE, runs faster, and -- unlike t-SNE -- can place "
            "crops it was not fitted on. The circle variant lays the graph out on a sphere instead of "
            "a plane, which suits L2-normalised embeddings."
        ),
        shown_in="Embeddings tab (Projection = umap or umap_circle)",
        blocks=(
            Heading("The idea"),
            Bullets((
                "Build a weighted k-nearest-neighbour graph of the embeddings, with each point's local distance scale normalised so that dense and sparse regions are treated alike.",
                "Define the same kind of graph over a candidate 2D layout.",
                "Move the layout until the two graphs agree, by minimising the cross-entropy between their edge weights -- attracting connected points, repelling sampled non-neighbours.",
            ), numbered=True),
            Equation(
                r"\mathcal{C} = \sum_{(i,j)} \left[\, v_{ij} \log \frac{v_{ij}}{w_{ij}} "
                r"+ (1 - v_{ij}) \log \frac{1 - v_{ij}}{1 - w_{ij}} \,\right]",
                where=(
                    (r"v_{ij}", "edge weight in the high-dimensional neighbourhood graph"),
                    (r"w_{ij}", "edge weight in the 2D layout"),
                ),
                note="The fuzzy-set cross-entropy UMAP minimises. The second term is the repulsion t-SNE's KL divergence lacks, and it is what preserves more of the global arrangement.",
            ),
            Heading("The circle variant"),
            Para(
                "*umap_circle* runs the same algorithm with a spherical (haversine) output metric, "
                "then maps the sphere onto a flat disk by azimuthal equidistant projection. It exists "
                "because L2-normalised embeddings genuinely live on a hypersphere -- that is the "
                "geometry an angular-margin loss optimises -- so a spherical layout matches the data's "
                "own shape. Proximity on the disk means angular closeness, and the rim is as "
                "meaningful as the centre."
            ),
            Table(
                headers=("Property", "UMAP"),
                rows=(
                    ("Deterministic", "with a fixed seed, yes"),
                    ("Preserves local neighbourhoods", "yes"),
                    ("Preserves global structure", "better than t-SNE, worse than PCA"),
                    ("Places new points", "yes, approximately -- learned from the fitted graph"),
                    ("Speed", "much faster than t-SNE on large samples"),
                ),
                weights=(2, 4),
            ),
            Callout(CALLOUT_PITFALL, "Still not a measurement", (
                Para(
                    "UMAP preserves more of the global arrangement than t-SNE, which makes it more "
                    "tempting to over-read. Cluster sizes and inter-cluster gaps remain artefacts of "
                    "the layout. Use the class overlap matrix for any claim of the form \"these two "
                    "classes are close\"."
                ),
            )),
        ),
        see_also=("projection-2d", "tsne", "pca", "class-overlap", "embedding-space"),
        keywords=("umap", "haversine", "sphere", "circle", "graph", "neighbour"),
    ),

    ManualPage(
        id="class-overlap",
        title="Class overlap matrix",
        summary=(
            "The mean cosine similarity between the crops of every pair of classes, as the model "
            "embeds them. It answers a question the confusion matrix cannot: the confusion says the "
            "model mixes two classes up; this says whether they actually sit on top of each other in "
            "feature space."
        ),
        shown_in="Embeddings tab (\"How much the classes overlap\")",
        blocks=(
            Equation(
                r"S_{a,b} = \frac{1}{|A| \, |B|} \sum_{i \in A} \sum_{j \in B} \tilde{u}_i \cdot \tilde{u}_j "
                r"\quad (a \neq b)",
                where=(
                    (r"A, B", "the sampled crops of classes $a$ and $b$"),
                    (r"\tilde{u}_i", "the L2-normalised embedding of crop $i$"),
                ),
                note="Mean pairwise cosine similarity between two classes.",
            ),
            Equation(
                r"S_{a,a} = \frac{1}{|A| \left( |A| - 1 \right)} \sum_{i \in A} \sum_{j \in A, \, j \neq i} "
                r"\tilde{u}_i \cdot \tilde{u}_j",
                note="On the diagonal, each crop's similarity to itself (exactly 1) is excluded -- including it would inflate every diagonal towards a class of one.",
            ),
            Equation(
                r"\mathrm{separation}_a = S_{a,a} - \max_{b \neq a} S_{a,b}",
                note="How much more a class looks like itself than like its closest neighbour. Near zero or negative is a class with no identity of its own.",
            ),
            HowToRead((
                ("A high diagonal value", "a tight, coherent class"),
                ("A low diagonal value", "the class is internally diverse -- several visual modes under one label"),
                ("A high off-diagonal pair", "two classes the model represents almost identically"),
                ("Separation near zero", "this class is not distinguishable from its neighbour in feature space"),
                ("The top pairs list", "the most-overlapping class pairs, worst first -- the taxonomy problems, ranked"),
            )),
            Heading("Sampling"),
            Para(
                "The full computation is quadratic in the class size, so each class is subsampled to "
                f"at most {OVERLAP_MAX_PER_CLASS} crops. The mean of a similarity distribution "
                "converges long before the sample runs out, so the numbers are stable to about two "
                "decimals -- but they are estimates, and a difference in the third decimal is not a "
                "finding. The sample is seeded, so re-running on the same rows gives the same numbers."
            ),
            Callout(CALLOUT_TIP, "What to do with a high-overlap pair", (
                Para(
                    "High overlap means more training on the same definitions will not separate them. "
                    "The realistic options are: merge the two classes, re-annotate the boundary "
                    "between them, or add a feature that distinguishes them. The Prototypes tab's "
                    "taxonomy hints make the same suggestion from the prototype side."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Overlap is not confusion", (
                Para(
                    "Two classes can overlap heavily in the embedding and still be separated by the "
                    "classifier head, or sit far apart and still be confused because of a prior or a "
                    "threshold. Read this matrix *next to* the confusion matrix: agreement means a "
                    "representation problem, disagreement means a head or calibration problem."
                ),
            )),
        ),
        see_also=("confusion-matrix", "embedding-space", "prototype-similarity", "compactness-separation"),
        keywords=("overlap", "class similarity", "separation", "taxonomy", "confusable"),
    ),

    ManualPage(
        id="nearest-neighbours",
        title="Nearest neighbours",
        summary=(
            "The $k$ crops whose embeddings are closest to a chosen crop, nearest first. It answers "
            "\"what else looks like this to the model?\", which is the question that separates a lone "
            "mislabelled crop from a whole pocket of them the model has learned as a cluster."
        ),
        shown_in=("Embeddings and Compare embeddings tabs (the map's Neighbours tool), Prototypes tab "
                  "(Neighbor explorer), similarity search from any crop"),
        blocks=(
            Equation(
                r"\mathcal{N}_k(q) = \mathrm{top}\text{-}k_{\; i \neq q} \; \left( \tilde{u}_q \cdot \tilde{u}_i \right)",
                where=(
                    (r"q", "the query crop"),
                    (r"k", "how many neighbours to return"),
                ),
                note="Ranked by cosine similarity, highest first. The query itself is excluded.",
            ),
            HowToRead((
                ("Neighbours all of the query's class", "the crop sits comfortably inside its class"),
                ("Neighbours mostly of one *other* class", "strong evidence of a mislabel -- the model is not confused, it disagrees"),
                ("Neighbours of many different classes", "the crop sits in a no-man's-land: genuinely ambiguous, or corrupted"),
                ("Similarities above 0.98", "near-duplicates; check the split column for leakage"),
                ("Uniformly low similarities", "an outlier the model has no representation for"),
            )),
            Para(
                "On the interactive maps, the Neighbours tool does this search for a clicked point among the "
                "points of the map, and rings them where they sit. The search is in the embedding space, "
                "not on the screen: a neighbour drawn far from the point is a place where the projection "
                "tore apart what the model keeps together."
            ),
            Callout(CALLOUT_TIP, "The diagnostic gesture", (
                Para(
                    "Standing in front of a strange error, the useful next move is almost never "
                    "another metric -- it is this. If the neighbours are all correctly labelled crops "
                    "of the predicted class, the annotation is the problem. If they are a mixed bag, "
                    "the crop is hard. If they are a tight group all sharing the same wrong label, you "
                    "have found a systematic annotation error worth a whole cleaning pass."
                ),
            )),
        ),
        see_also=("embedding-space", "near-duplicates", "margin-candidates", "class-overlap"),
        keywords=("neighbour", "neighbor", "knn", "similar", "search", "lookalike"),
    ),

    ManualPage(
        id="embedding-comparison",
        title="One model, several datasets",
        summary=(
            "The Compare embeddings tab draws the crops of several experiments in one embedding space. "
            "It is meant for one model run on different datasets -- two sites, two sensors, last year's "
            "base and this year's -- and answers where the datasets sit relative to each other, as the "
            "model sees them."
        ),
        shown_in="Compare embeddings tab",
        blocks=(
            Heading("Only the same model can be compared"),
            Para(
                "An embedding is a coordinate in a space the model built for itself. Two models build two "
                "spaces, and a point at the same place in both means nothing: a map of their union would "
                "show the difference between the models, not between the datasets. Nothing in the vectors "
                "says which model made them -- the same number of dimensions proves nothing -- so the "
                "model's own prototypes, imported with each experiment, serve as its fingerprint."
            ),
            Equation(
                r"\min_{c, j} \; \tilde{p}^{A}_{c,j} \cdot \tilde{p}^{B}_{c,j} \;\geq\; 1 - \epsilon",
                where=(
                    (r"\tilde{p}^{A}_{c,j}", "prototype $j$ of class $c$ in experiment A, matched by class label"),
                    (r"\epsilon", f"the tolerance, {PROTOTYPE_TOLERANCE:g}: rounding, not another checkpoint"),
                ),
                note="Every prototype must be the same on both sides, or the comparison is refused.",
            ),
            Para(
                "An experiment with no imported prototypes cannot prove it comes from the same model and "
                "cannot be added. Prototypes estimated by k-means are never accepted: they are fit on each "
                "dataset's own crops, so they differ between datasets by construction."
            ),
            Heading("Two ways to project"),
            Table(
                headers=("Projection", "What it is", "Use it to"),
                rows=(
                    ("Fitted on all", "one projection fitted on the same number of crops from each "
                     "experiment", "see every dataset on an equal footing"),
                    ("Fitted on the reference", "fitted on the reference alone; the others are placed onto "
                     "that map", "ask where the new data lands in the reference's space -- the map does not "
                     "move when an experiment is added"),
                ),
                weights=(2, 4, 4),
                caption="t-SNE cannot place points it was not fitted on, so it is only offered fitted on all.",
            ),
            Para(
                "Shown overlaid, the reference's points are discs and the others' squares, whatever the "
                "colours mean; side by side, every experiment gets an interactive map of the same points, its "
                "own in colour over the others in grey. The panels move, zoom and select together, so a lasso "
                "or a point's neighbours show in every one of them. The stars are the model's prototypes. A "
                "projection is still a projection: read the measures below the map before concluding that "
                "two clouds overlap."
            ),
            Callout(CALLOUT_TIP, "What is saved", (
                Para(
                    "The list of compared experiments is saved in the workspace as references -- a workspace "
                    "and an experiment name -- never as a copy: an experiment of another workspace is read "
                    "from that workspace each time. Renaming it follows; deleting that workspace leaves a "
                    "reference that says it can no longer be read."
                ),
            )),
        ),
        see_also=("domain-separability", "embedding-coverage", "cross-label-agreement", "class-shift",
                  "cross-neighbours", "embedding-pairwise", "linear-probe", "projection-2d", "prototypes"),
        keywords=("compare", "dataset", "domain", "shift", "drift", "several", "base", "reference"),
    ),

    ManualPage(
        id="domain-separability",
        title="Domain separability",
        summary=(
            "How well a linear classifier tells the crops of two datasets apart from their embeddings "
            "alone. If it cannot, the model sees the two datasets as the same kind of data; if it can, "
            "something systematic -- a sensor, a site, a season, a preprocessing step -- reaches the "
            "model's representation."
        ),
        shown_in="Compare embeddings tab (\"Told apart (domain AUC)\")",
        blocks=(
            Para(
                f"A logistic regression is trained to predict which dataset a crop comes from, on equal "
                f"samples of each side (at most {STATS_MAX_POINTS:,} crops per side), and scored by "
                f"{DOMAIN_FOLDS}-fold cross-validation: every crop is scored by a model that never saw it. "
                f"The score is the ROC AUC of those held-out predictions."
            ),
            Equation(
                r"\mathrm{AUC} = P\left( s(x_B) > s(x_A) \right)",
                where=((r"s(x)", "the held-out score of a crop for \"comes from B\""),),
                note="The chance that a random crop of B is scored above a random crop of A.",
            ),
            HowToRead((
                ("About 0.5", "indistinguishable: the model represents both datasets alike"),
                ("0.6 to 0.8", "a visible shift; check which classes it concentrates in"),
                ("Above 0.9", "two different worlds to the model -- expect its accuracy not to transfer"),
            )),
            Callout(CALLOUT_PITFALL, "Crops in both datasets are left out", (
                Para(
                    "A crop present in both (the same uuid) is the same vector on both sides; in a held-out "
                    "fold its twin, labelled with the other side, would push the score below 0.5. Those crops "
                    "are left out of this test and counted next to it."
                ),
            )),
            Para(
                "A high AUC is not bad news by itself: it says the datasets differ, not that the model "
                "fails on one. Accuracy per class in each dataset, in the table below the map, says "
                "whether the difference costs anything."
            ),
        ),
        see_also=("embedding-comparison", "embedding-coverage", "roc-auc"),
        keywords=("domain", "classifier", "auc", "shift", "separable", "drift"),
    ),

    ManualPage(
        id="embedding-coverage",
        title="Coverage by the reference",
        summary=(
            "The share of another dataset's crops that have a crop of the reference about as close as "
            "the reference's crops usually are to each other. An uncovered crop lands where the reference "
            "has nothing: whatever the model learned from data like the reference says little about it."
        ),
        shown_in="Compare embeddings tab (\"Covered by the reference\", \"Outside the reference\")",
        blocks=(
            Para(
                f"For each crop of the reference, its nearest other crop of the reference gives a similarity "
                f"$s_{{\\mathrm{{nn}}}}$. The threshold $\\tau$ is the {COVERAGE_PERCENTILE:g}th percentile of "
                f"those: all but the reference's own outliers are at least that close to a neighbour."
            ),
            Equation(
                r"\tau = \mathrm{percentile}_{%g}\left( s_{\mathrm{nn}}(a) : a \in A \right)" % COVERAGE_PERCENTILE,
                where=((r"A", "the reference's crops"),),
            ),
            Equation(
                r"\mathrm{coverage} = \frac{1}{|B|} \sum_{b \in B} \mathbf{1}\left[ \max_{a \in A} \tilde{u}_b \cdot \tilde{u}_a \geq \tau \right]",
                where=((r"B", "the other dataset's crops"),),
                note="By construction, about 95% of the reference covers itself.",
            ),
            HowToRead((
                ("Close to 95%", "the other dataset lives where the reference does"),
                ("Much lower", "a part of it lands where the reference has no crop: open \"Outside the reference\""),
                ("Low for a few classes only", "those classes look different in the other dataset"),
            )),
            Para(
                f"Measured on samples: at most {STATS_MAX_POINTS:,} crops of each side, searched among at "
                f"most {KNN_POOL_MAX:,} of the reference. The median similarity to the nearest reference "
                f"crop, shown beside it for both sides, says the same thing without a threshold."
            ),
        ),
        see_also=("embedding-comparison", "nearest-neighbours", "domain-separability"),
        keywords=("coverage", "outside", "novel", "out of distribution", "ood", "nearest"),
    ),

    ManualPage(
        id="cross-label-agreement",
        title="Label agreement with the reference",
        summary=(
            "For each crop of another dataset, whether most of its nearest crops in the reference carry "
            "its label. Measured beside the same figure inside the reference, it shows whether two "
            "datasets were labelled the same way."
        ),
        shown_in="Compare embeddings tab (\"Labels agreeing with the reference\", \"Labelled otherwise\")",
        blocks=(
            Equation(
                r"\mathrm{agreement} = \frac{1}{|B|} \sum_{b \in B} \mathbf{1}\left[ \mathrm{mode}\left( y_a : a \in \mathcal{N}^{A}_{k}(b) \right) = y_b \right]",
                where=(
                    (r"\mathcal{N}^{A}_{k}(b)", f"the k = {AGREEMENT_K} crops of the reference nearest to b"),
                    (r"y", "a crop's label"),
                ),
            ),
            Para(
                "The same is measured for the reference's own crops (each against its other crops). That "
                "is the baseline: classes overlap, so even a perfectly labelled dataset does not agree with "
                "its neighbours every time."
            ),
            HowToRead((
                ("Both figures close", "the two datasets follow the same labelling"),
                ("The other dataset clearly lower", "its labels disagree with what the reference calls the "
                 "same-looking crops: another convention, another annotator, or label noise"),
                ("Low for one class", "that class is defined differently -- open \"Labelled otherwise\""),
            )),
            Callout(CALLOUT_PITFALL, "Not an accuracy", (
                Para(
                    "This uses the labels of both datasets and the geometry of the model, never its "
                    "predictions. A crop can be predicted right and still disagree with its neighbours, and "
                    "the other way round."
                ),
            )),
        ),
        see_also=("embedding-comparison", "nearest-neighbours", "class-shift"),
        keywords=("label", "agreement", "convention", "annotation", "knn", "noise"),
    ),

    ManualPage(
        id="class-shift",
        title="Per class, from one dataset to the other",
        summary=(
            "The Compare embeddings table: for each class, how its crops moved from the reference to the "
            "other dataset in the model's space, and what it cost. Sort it by the column that worries you."
        ),
        shown_in="Compare embeddings tab (\"Per class\")",
        blocks=(
            Table(
                headers=("Column", "What it is"),
                rows=(
                    ("Crops", "the class's crops measured in the reference, then in the other dataset"),
                    ("Centroid similarity", "cosine similarity between the mean embedding of the class in "
                     "each dataset: 1 when the class sits at the same place"),
                    ("Similarity to prototype", "mean similarity of the class's crops to its nearest prototype "
                     "of the model, in each dataset: lower means less typical for the model"),
                    ("Accuracy", "the share of the class's crops predicted right, in each dataset"),
                    ("Covered", "coverage by the reference, for this class's crops of the other dataset"),
                    ("Labels agreeing", "label agreement with the reference, for this class"),
                ),
                weights=(2, 5),
            ),
            Equation(
                r"\cos\left( \bar{u}^{A}_{c}, \bar{u}^{B}_{c} \right), \qquad \bar{u}^{A}_{c} = \frac{1}{|A_c|} \sum_{a \in A_c} \tilde{u}_a",
                where=((r"A_c", "the crops of class $c$ in the reference"),),
                note="The centroid similarity.",
            ),
            HowToRead((
                ("Low centroid similarity, accuracy held", "the class looks different but the model copes"),
                ("Lower similarity to prototype and accuracy drop", "the class is atypical in the other "
                 "dataset and the model pays for it: more such data in training"),
                ("Labels agreeing low, accuracy drop", "probably a labelling difference rather than a model one"),
            )),
            Para(
                "All of it is measured on the samples described in \"Coverage by the reference\"; a class "
                "with a handful of crops in either dataset gives noisy figures."
            ),
        ),
        see_also=("embedding-comparison", "embedding-coverage", "cross-label-agreement", "prototypes"),
        keywords=("class", "centroid", "shift", "prototype", "accuracy", "per class"),
    ),

    ManualPage(
        id="cross-neighbours",
        title="Nearest crops in another dataset",
        summary=(
            "Clicking a crop on the Compare embeddings map lists its nearest crops in another compared "
            "experiment -- the same search as \"Find similar\", across datasets. It shows what the "
            "reference contains that the model considers alike, and what it was labelled there."
        ),
        shown_in="Compare embeddings tab (a click on the map)",
        blocks=(
            Equation(
                r"\mathcal{N}^{B}_k(q) = \mathrm{top}\text{-}k_{\; b \in B} \; \left( \tilde{u}_q \cdot \tilde{u}_b \right)",
                where=((r"B", "the experiment searched, over all of its crops"),),
                note="Exact, not sampled; the crop itself is left out when it is searched in its own experiment.",
            ),
            HowToRead((
                ("Neighbours share the crop's label", "the two datasets agree about this kind of crop"),
                ("Neighbours mostly carry another label", "a labelling difference between the datasets, or a mislabel"),
                ("Low similarities only", "the other dataset has nothing like it"),
            )),
        ),
        see_also=("nearest-neighbours", "embedding-comparison", "cross-label-agreement"),
        keywords=("neighbour", "neighbor", "similar", "across", "other dataset", "knn"),
    ),

    ManualPage(
        id="linear-probe",
        title="Linear probe",
        summary=(
            "A linear classifier trained on the embeddings alone, to recover the labels. Where the overlap "
            "matrix says how alike classes are on average, the probe says whether a straight boundary can "
            "tell them apart -- which is what a classification head on these embeddings has to do."
        ),
        shown_in="Embeddings tab (\"How separable the classes are\"), Compare embeddings tab (\"Probe accuracy\")",
        blocks=(
            Para(
                f"A multinomial logistic regression, on standardized embeddings, is trained and scored by "
                f"{PROBE_FOLDS}-fold stratified cross-validation: every crop is predicted by a probe that never "
                f"saw it. Classes are sampled to at most {PROBE_MAX_PER_CLASS} crops each and "
                f"{PROBE_MAX_POINTS:,} in all; a class with fewer than {PROBE_FOLDS} crops is left out."
            ),
            Equation(
                r"\hat{y}(x) = \arg\max_{c} \; w_c \cdot z(x) + b_c",
                where=(
                    (r"z(x)", "the crop's embedding, standardized"),
                    (r"w_c, b_c", "the probe's weights for class $c$, fitted on the other folds"),
                ),
            ),
            HowToRead((
                ("Probe close to the model", "the embeddings are as separable as the head makes them: what the "
                 "model gets wrong, it gets wrong in its representation"),
                ("Probe well above the model", "the embeddings hold what is needed and the head does not use it "
                 "-- a retrained head, or calibration, could gain"),
                ("Probe well below the model", "the classes are not linearly separable in this space (the "
                 "head relies on several prototypes per class, or on a non-linear head)"),
                ("One class with a low probe F1", "look at its row of the overlap matrix and at its neighbours"),
            )),
            Para(
                "In Compare embeddings, a probe is trained on each experiment's own crops: the same model, "
                "but a dataset whose classes it represents less separably shows a lower probe accuracy."
            ),
            Callout(CALLOUT_PITFALL, "The probe is trained on the evaluated pool", (
                Para(
                    "It is cross-validated, so no crop is scored by a probe that saw it -- but it is trained on "
                    "the same test split the model is evaluated on. It measures the geometry of the "
                    "embeddings, not a model you could deploy."
                ),
            )),
        ),
        see_also=("class-overlap", "embedding-space", "domain-separability", "embedding-comparison"),
        keywords=("probe", "linear", "logistic", "separable", "separability", "classifier"),
    ),

    ManualPage(
        id="embedding-pairwise",
        title="Every pair of datasets",
        summary=(
            "With many experiments compared, the two matrices of every pair: how well each pair of datasets "
            "can be told apart, and how much of each one another covers. They show which datasets resemble "
            "which -- groups of sites, a sensor that stands apart -- rather than each one against the "
            "reference only."
        ),
        shown_in="Compare embeddings tab (\"Every pair of experiments\")",
        blocks=(
            Para(
                f"The same two measures as against the reference -- see \"Domain separability\" and \"Coverage "
                f"by the reference\" -- on smaller samples, since there are $n(n-1)/2$ pairs: "
                f"{PAIRWISE_POINTS:,} crops per experiment, and a {PAIRWISE_FOLDS}-fold cross-validated "
                f"classifier per pair."
            ),
            Table(
                headers=("Matrix", "Cell (row, column)", "Reads"),
                rows=(
                    ("Told apart", "domain AUC of the two datasets", "symmetric; 0.5 = the same to the model"),
                    ("Covered", "share of the column's crops with a crop of the row about as close as the "
                     "row's crops are to each other", "not symmetric: a large, varied dataset covers a "
                     "narrow one, not the other way round"),
                ),
                weights=(2, 4, 4),
            ),
            HowToRead((
                ("A block of low AUCs", "those datasets look alike to the model -- one group"),
                ("A row of high AUCs", "that dataset stands apart from all the others"),
                ("Row covers column, not the reverse", "the row's dataset contains the column's, and more"),
            )),
        ),
        see_also=("domain-separability", "embedding-coverage", "embedding-comparison"),
        keywords=("pairwise", "matrix", "every pair", "many", "datasets", "similarity"),
    ),

    ManualPage(
        id="fit-and-shown",
        title="Crops fitted, crops shown",
        summary=(
            "A projection is fitted on some crops and shows others: the fit is the slow part and gains "
            "little past a few thousand crops, while placing more crops on a fitted map is cheap. So the "
            "map is fitted on a modest set, then shows many more, each placed where the fit puts it."
        ),
        shown_in="Embeddings and Compare embeddings tabs (\"Fit on\", \"Shown\")",
        blocks=(
            Table(
                headers=("Setting", "What it does"),
                rows=(
                    ("Fit on: crops as in the data", f"the fit set is drawn at random -- {DEFAULT_FIT_SIZE:,} by "
                     "default -- so each class keeps its share; the map shows the data as it is"),
                    ("Fit on: balanced per class", "the same number of crops from every class (all of them for "
                     "a smaller class): a rare class weighs as much in the layout as a frequent one"),
                    ("Shown", f"how many crops the map draws -- {DEFAULT_DISPLAY_SIZE:,} by default, at most "
                     f"{MAX_DISPLAY_SIZE:,}. They always include the fit set; the others are drawn at random "
                     "and placed onto the fit"),
                ),
                weights=(3, 6),
            ),
            Table(
                headers=("Method", "Placing a crop that was not fitted"),
                rows=(
                    ("PCA", "exact and nearly free: one matrix product"),
                    ("UMAP", "an estimate from the fitted crops nearest to it -- fast, though slower than PCA; "
                     "a crop unlike any fitted one is pulled towards the structure the fit learned"),
                    ("t-SNE", "impossible: the map shows the fit set only"),
                ),
                weights=(2, 7),
            ),
            Callout(CALLOUT_TIP, "Balance the fit, not the map", (
                Para(
                    "Balancing changes which crops decide the layout, not what is shown: with a balanced fit "
                    "and a large display, rare classes get room on the map and frequent ones still appear in "
                    "their real numbers. The choice is saved in the workspace."
                ),
            )),
            Callout(CALLOUT_PITFALL, "Placed crops can look more familiar than they are", (
                Para(
                    "With UMAP, a crop placed onto a fit lands next to the fitted crops it resembles most, "
                    "even when it resembles none of them much. To see where new data really sits, fit on it "
                    "too -- or read the nearest-neighbour similarities rather than the map."
                ),
            )),
        ),
        see_also=("projection-2d", "pca", "umap", "tsne", "embedding-comparison"),
        keywords=("fit", "sample", "shown", "display", "balanced", "transform", "points"),
    ),
)
