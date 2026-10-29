"""Chapter: the manual's own front page, and the map of the metric zoo."""
from __future__ import annotations

from lookin.core.manual.model import (
    Bullets, Callout, Equation, Heading, HowToRead, ManualPage, Para, Table,
    CALLOUT_EXAMPLE, CALLOUT_PITFALL, CALLOUT_TIP, CALLOUT_WARNING,
)

CHAPTER = ("overview", "Start here")

PAGES = (
    ManualPage(
        id="index",
        title="About this manual",
        summary=(
            "Every chart, table and number in the app has an (i) button beside it, and every one of "
            "those buttons opens this window at the page that defines what you are looking at. The "
            "table of contents on the left is the whole manual: you can read it as a course, or jump "
            "in wherever a number surprised you."
        ),
        blocks=(
            Heading("How a page is laid out"),
            Bullets((
                "*In short* at the top: two or three sentences. Enough to settle a single question and leave.",
                "Then the definition, with the equation the app actually computes and every symbol in it named.",
                "Then *How to read it*, which maps each visual element of the chart or table to what it means.",
                "Then the pitfalls: the ways this particular number lies, and what to check instead.",
            )),
            Heading("Conventions used throughout"),
            Table(
                headers=("Symbol", "Meaning"),
                rows=(
                    ("$N$", "number of rows (crops) being scored"),
                    ("$C$", "number of classes"),
                    ("$y_i$", "the true class of row $i$ -- the label in the dataset CSV"),
                    ("$\\hat{y}_i$", "the predicted class of row $i$"),
                    ("$p_i \\in [0,1]^C$", "the score vector of row $i$, one entry per class"),
                    ("$p_{i,c}$", "the score row $i$ gives to class $c$"),
                    ("$\\hat{p}_i$", "the confidence: the largest entry of $p_i$"),
                    ("$\\mathbb{1}[A]$", "1 when $A$ is true, 0 otherwise"),
                ),
            ),
            Para(
                "A note on vocabulary: this app calls one row a *crop*, because that is what it was "
                "built to look at, but nothing here assumes images. A row is whatever your dataset CSV "
                "has one line per."
            ),
            Callout(CALLOUT_TIP, "Where the numbers come from", (
                Para(
                    "The app never trains anything and never calls out to a network. Every number in "
                    "every tab is computed, in this process, from the two CSVs you loaded: the dataset "
                    "CSV (one row per crop, with its true label and metadata) and one results CSV per "
                    "experiment (one row per crop, with one score column per class)."
                ),
            )),
        ),
        see_also=("metric-levels", "predictions", "confusion-matrix"),
        keywords=("help", "manual", "documentation", "notation", "symbols"),
    ),

    ManualPage(
        id="metric-levels",
        title="The three levels of a metric",
        summary=(
            "Classification metrics come in three families, and mixing them up is the single most "
            "common way to misread an evaluation. Decision metrics judge the class the model picked; "
            "ranking metrics judge the order it put the crops in; probabilistic metrics judge the "
            "numbers themselves. A model can be excellent at one level and useless at another."
        ),
        blocks=(
            Heading("Level 1 -- the decision"),
            Para(
                "Throw away the scores, keep only the predicted class $\\hat{y}_i = \\arg\\max_c p_{i,c}$, "
                "and compare it to $y_i$. Everything computed from the confusion matrix lives here: "
                "accuracy, precision, recall, specificity, F1, balanced accuracy, MCC, Cohen's kappa."
            ),
            Para(
                "These are the numbers that describe the system you would actually ship if you shipped "
                "the argmax. They are also blind to *how close* a decision was: a crop scored 0.51 and "
                "a crop scored 0.99 count exactly the same."
            ),
            Heading("Level 2 -- the ranking"),
            Para(
                "Keep the scores, but only their order. If you sorted every crop by its score for class "
                "$c$, would the true members of $c$ come first? ROC-AUC and average precision answer "
                "that, and they do it at every threshold at once, which is why they need no threshold."
            ),
            Para(
                "These are the numbers to look at when the deployed threshold is not settled yet, or "
                "when the class balance at deployment will differ from the balance in this evaluation."
            ),
            Heading("Level 3 -- the probabilities"),
            Para(
                "Take the scores literally, as claims about the world: of the crops this model calls "
                "0.80, are 80% of them right? Log loss, the Brier score, ECE and MCE live here. This is "
                "the level that matters as soon as a downstream decision uses the score itself -- an "
                "abstention threshold, a triage queue, a fusion with another sensor."
            ),
            Table(
                headers=("Level", "Uses", "Blind to", "Examples"),
                rows=(
                    ("Decision", "$\\hat{y}$ only", "how confident, and how close", "accuracy, F1, MCC"),
                    ("Ranking", "the order of $p$", "the actual values, and the threshold", "ROC-AUC, AP"),
                    ("Probabilistic", "the values of $p$", "nothing -- strictest level", "log loss, Brier, ECE"),
                ),
            ),
            Callout(CALLOUT_PITFALL, "The classic contradiction", (
                Para(
                    "A model with an excellent ROC-AUC and a terrible ECE is a model that ranks crops "
                    "correctly but announces the wrong numbers -- perfectly usable after temperature "
                    "scaling, and dangerous before it. A model with a good accuracy and a poor ROC-AUC "
                    "is the reverse: the chosen threshold happens to work, and nothing else will."
                ),
                Para(
                    "This is why the Summary table in the Metrics tab shows all three levels side by "
                    "side rather than a headline number."
                ),
            )),
            Heading("Which one should you report?"),
            Bullets((
                "Reporting to someone who will deploy the argmax: a decision metric, plus the confusion matrix.",
                "Comparing two training runs: a ranking metric, because it does not depend on where the threshold landed.",
                "Feeding the score into anything downstream: a probabilistic metric, and a reliability diagram.",
                "Under class imbalance: never accuracy alone -- see the page on support and imbalance.",
            )),
        ),
        see_also=("accuracy", "roc-auc", "log-loss", "calibration-error", "summary-table"),
        keywords=("levels", "families", "taxonomy", "which metric", "choose"),
    ),

    ManualPage(
        id="summary-table",
        title="The Summary table, column by column",
        summary=(
            "One row per experiment, and every column is a different summary of the same predictions. "
            "They are grouped by level: decision metrics first, then ranking, then probabilistic. "
            "Reading across a row tells you far more than any one of them."
        ),
        shown_in="Metrics tab (\"Summary\")",
        blocks=(
            Table(
                headers=("Column", "Level", "Reading", "Page"),
                rows=(
                    ("N", "--", "crops scored, after the join and any filter", "How the dataset and the results are joined"),
                    ("Accuracy", "decision", "share right; compare against $\\max_c \\pi_c$", "Accuracy"),
                    ("Balanced Acc", "decision", "macro-averaged recall; chance is $1/C$", "Balanced accuracy"),
                    ("Macro F1", "decision", "every class counts once", "F1 score"),
                    ("Weighted F1", "decision", "every crop counts once", "F1 score"),
                    ("MCC", "decision", "correlation; 0 is uninformative whatever the imbalance", "Matthews correlation coefficient (MCC)"),
                    ("Cohen's kappa", "decision", "accuracy above chance", "Cohen's kappa"),
                    ("Macro / Weighted ROC-AUC", "ranking", "quality of the ranking, threshold-free", "ROC curve and ROC-AUC"),
                    ("Macro / Weighted AP", "ranking", "ranking quality for rare classes; floor is the prevalence", "Precision-Recall curve and average precision"),
                    ("Log loss", "probabilistic", "lower better; $\\log C$ is the uniform baseline", "Log loss (cross-entropy)"),
                    ("Brier", "probabilistic", "lower better; bounded", "Brier score"),
                    ("Brier reliability / resolution / uncertainty", "probabilistic", "the three parts of Brier", "Brier decomposition: reliability, resolution, uncertainty"),
                    ("ECE / MCE", "probabilistic", "average and worst confidence gap", "Calibration error: ECE and MCE"),
                ),
                weights=(3, 2, 5, 3),
                caption="The last column names this metric's page in the table of contents on the left.",
            ),
            Heading("Reading a row"),
            Bullets((
                "Start with the decision metrics and the confusion matrix: that is the system as it would ship.",
                "Then the ranking metrics: if they are much better than the decision ones, the threshold or the head is what is losing the performance, not the features.",
                "Then the probabilistic ones: they decide whether the confidence can be used for anything downstream.",
                "A large gap between macro and weighted versions of the same metric is imbalance, and it is telling you to read the per-class table.",
            )),
            Callout(CALLOUT_WARNING, "A blank cell is not a zero", (
                Para(
                    "Probabilistic columns are left empty when the scores are not a probability "
                    "distribution, and ranking columns when a class's score column does not vary. "
                    "Both cases are explained in the banner above the table -- see the score "
                    "diagnostics page."
                ),
            )),
        ),
        see_also=("metric-levels", "accuracy", "roc-auc", "calibration-error", "score-diagnostics"),
        keywords=("summary", "table", "columns", "overview", "metrics tab"),
    ),

    ManualPage(
        id="report-export",
        title="Exporting a report",
        summary=(
            "One self-contained file holding your notes, the numbers, the charts and the provenance: "
            "which CSVs were loaded, which filter was active, which review version was in effect. It "
            "comes as an HTML page, a PowerPoint deck or a Markdown document, and opens with no server "
            "and no network."
        ),
        shown_in="Metrics tab (\"Export a report\")",
        blocks=(
            Para(
                "Everything is embedded in the file -- the charts as images, the tables as text -- so the "
                "report keeps working when moved, mailed or archived. Nothing in it is fetched when it is "
                "opened, and a browser prints the HTML one to PDF."
            ),
            Heading("Where it goes"),
            Bullets((
                "*This computer*, the default: the file is downloaded by the browser you are using, like any download.",
                "*A folder on the server*: written on the machine running Lookin, in the folder you pick -- what you want when that machine is where the reports are archived.",
            )),
            Heading("What it contains"),
            Bullets((
                "The provenance block: which dataset CSV, which split, which review version, and whether the reviewed labels were applied.",
                "The Notes tab's text, first, when there is any: what the person who made the report had to say about it.",
                "The summary table, as it stands for the experiments selected.",
                "Per-class charts: F1, precision, recall and average precision per class, one bar per experiment.",
                "The per-class tables and the confusion matrices.",
                "With exactly two experiments selected, the head-to-head from the Compare tab.",
            )),
            Para(
                "Each part can be left out with its tick box. The Markdown document embeds its charts as "
                "images inside the file; most Markdown viewers show them, a few wikis show their caption "
                "instead."
            ),
            Heading("Why the provenance block matters"),
            Para(
                "A metric without its context cannot be compared against anything. The report records "
                "the dataset file, the experiments, the active row filter, the review version and the "
                "app version, because the most common question about a six-month-old number is not "
                "what it was but what it was measured on."
            ),
            Callout(CALLOUT_TIP, "Pair it with a snapshot", (
                Para(
                    "A report is a frozen picture; a saved workspace is the live session it came from. "
                    "Exporting a report from a bookmarked workspace stamps the workspace id into it, so the "
                    "reader can reopen the exact view that produced the figure rather than trying to "
                    "reconstruct it."
                ),
            )),
        ),
        see_also=("notes", "summary-table", "dataset-join", "review-versions"),
        keywords=("report", "export", "html", "pptx", "powerpoint", "markdown", "download", "provenance",
                  "share", "archive"),
    ),

    ManualPage(
        id="notes",
        title="Notes",
        summary=(
            "A page of free text per workspace: what was decided, what was tried, what the numbers do "
            "not say. It is written like a message -- formatted as you type, no markup to learn -- "
            "kept with the workspace, and put first in every report exported from it."
        ),
        shown_in="Notes tab",
        blocks=(
            Heading("What a new note starts with"),
            Para(
                "The workspace's name as its title, then a *Performance summary*: for each experiment its "
                "accuracy, macro F1, macro average precision and number of rows, with the best macro F1 "
                "marked. The summary is written into the note once, when the note is created; from then "
                "on it is your text like the rest, and you can edit it or delete it."
            ),
            Para(
                "*Regenerate the recap* writes the summary again from the current numbers -- after a "
                "relabeling, a new experiment, a change of filter -- and only that part: the rest of the "
                "note is left as it is. The summary sits in a lightly marked block for that reason; if "
                "you delete the block, the next regeneration puts a new one under the title."
            ),
            Heading("Writing"),
            Bullets((
                "The toolbar: bold, italic, strikethrough, a title and two heading levels, bullet and numbered lists, a quote, inline code, a link.",
                "As in a chat box, typing \"- \" or \"1. \" at the start of a line starts a list, and \"# \", \"## \", \"### \" a heading.",
                "Pasted text arrives as plain text, without the formatting of the page it came from.",
                "The note is saved as you type, a moment after the last key, and when you leave the editor.",
            )),
            Heading("Where it is kept"),
            Para(
                "In the workspace's folder, as a Markdown file named notes.md -- readable in any text editor, "
                "carried by a duplicate and by an export of the workspace. In a session not saved as a "
                "workspace yet, the note lives in that browser tab and goes into the workspace when you "
                "save the session as one. *Download .md* gives you the file."
            ),
            Callout(CALLOUT_TIP, "Agents read it too", (
                Para(
                    "An AI agent connected to Lookin's MCP endpoint reads the notes (get_notes) before it "
                    "draws conclusions about a workspace, and can add its own findings to them "
                    "(write_notes). Two tabs, or a tab and an agent, writing at the same moment: the last "
                    "save wins, and a tab showing the note picks up a change made elsewhere within a few "
                    "seconds."
                ),
            )),
        ),
        see_also=("report-export", "summary-table"),
        keywords=("notes", "comments", "write", "markdown", "recap", "summary", "notion"),
    ),
)
