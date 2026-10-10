"""Chapter: what the model's feature space looks like."""
from __future__ import annotations

from lookin.core.embedding_analysis import OVERLAP_MAX_PER_CLASS
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
        shown_in="Embeddings tab, Prototypes tab (Neighbor explorer), similarity search from any crop",
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
)
