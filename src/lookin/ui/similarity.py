"""Find the crops nearest to one crop in embedding space.

The gesture this adds is the one anybody makes in front of a strange error:
"what else looks like this to the model?". The answer separates the two
explanations that matter -- a lone mislabelled crop, or a whole pocket of
them the model has learned as a cluster, which is a very different problem.

It hangs off the shared image carousel rather than living in a tab of its
own, because that is where a user is already looking at a crop and already
forming an opinion about it -- the same reasoning that put the review
decision bar there. One instance per session, owned by the AppContext, so
the Gallery, a confusion-matrix cell, the Cleaning queue and the Embeddings
tab all reach the same thing.
"""
from __future__ import annotations

import traceback

import flet as ft

from lookin.core.embedding_analysis import nearest_neighbours, nearest_to_vector
from lookin.ui.carousel import DIALOG_CHROME_H, Carousel
from lookin.ui.components import info_button, ACCENT, card, info_banner
from lookin.ui.embedding_context import build_analysis_context

DEFAULT_K = 48
K_CHOICES = (12, 24, 48, 96)


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


class SimilaritySearch:
    def __init__(self, ctx):
        self.ctx = ctx
        self._query_uuid: str | None = None
        self._k = DEFAULT_K

        self.title = ft.Text("", weight=ft.FontWeight.BOLD)
        self.summary = ft.Text(size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.experiment_container = ft.Container()
        self.experiment_value: str | None = None
        self.k_dropdown = ft.Dropdown(
            label="How many", width=130, border_radius=8, dense=True, value=str(DEFAULT_K),
            options=[ft.dropdown.Option(str(k), f"{k} crops") for k in K_CHOICES],
            on_change=self._on_k_change,
        )
        self.body = ft.Container()
        self.dialog = ft.AlertDialog(
            # modal=False + on_dismiss below: Flet does not reset `open`
            # when a dialog is dismissed by tapping outside, and a stale
            # True makes the next page.open() silently do nothing.
            modal=False,
            title=ft.Row([ft.Icon(ft.Icons.TRAVEL_EXPLORE, color=ACCENT), self.title,
                          ft.Container(expand=True), info_button(ctx, "nearest-neighbours")],
                         spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            content=ft.Container(width=980, content=ft.Column([
                ft.Row([self.experiment_container, self.k_dropdown], spacing=12,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.summary,
                self.body,
            ], spacing=12, tight=True)),
            on_dismiss=lambda e: self.ctx.page.close(self.dialog),
            actions=[ft.TextButton("Close", on_click=lambda e: self.ctx.page.close(self.dialog))],
        )

    # ---- availability ----
    def experiments_with_embeddings(self) -> list:
        return [name for name, exp in self.ctx.state.experiments.items() if exp.has_embeddings()]

    def is_available(self) -> bool:
        """Whether the button is worth showing at all. Without an embedding
        sidecar there is no space to be near in, so offering the search
        would be offering a dead end."""
        return bool(self.ctx.state.dataset is not None and self.experiments_with_embeddings())

    # ---- opening ----
    def open(self, query_uuid: str) -> None:
        self._query_uuid = str(query_uuid)
        names = self.experiments_with_embeddings()
        if self.experiment_value not in names:
            self.experiment_value = names[0] if names else None
        self._rebuild_experiment_picker(names)
        self.title.value = f"Crops most like {self._query_uuid}"
        self._render()
        self.ctx.page.open(self.dialog)

    def _rebuild_experiment_picker(self, names: list) -> None:
        self.experiment_container.content = ft.Dropdown(
            label="Embedding space", width=260, border_radius=8, dense=True,
            value=self.experiment_value,
            options=[ft.dropdown.Option(n, n) for n in names],
            on_change=self._on_experiment_change,
            # Which experiment's embeddings is a real choice, not a detail:
            # two runs embed the same crop differently, and "what looks
            # similar" is a property of the model, not of the dataset.
            tooltip="Similarity is measured in this run's embedding space -- two models "
                    "disagree about what looks alike.",
        )

    def _on_experiment_change(self, e):
        self.experiment_value = e.control.value
        self._render()
        self._push()

    def _on_k_change(self, e):
        try:
            self._k = int(e.control.value)
        except (TypeError, ValueError):
            self._k = DEFAULT_K
        self._render()
        self._push()

    def _push(self):
        for control in (self.body, self.summary, self.experiment_container):
            if control.page:
                control.update()

    # ---- results ----
    def _render(self) -> None:
        if not self.experiment_value or not self._query_uuid:
            self.body.content = info_banner(
                "No loaded experiment has an embeddings sidecar attached -- add one on the Data tab.")
            self.summary.value = ""
            return
        try:
            context = build_analysis_context(self.ctx.state, self.experiment_value)
        except Exception as ex:
            _log(f"similarity: could not build the embedding context: {ex}")
            traceback.print_exc()
            self.body.content = info_banner(f"Could not read the embeddings: {ex}",
                                            ft.Icons.ERROR_OUTLINE)
            return
        if context is None:
            self.body.content = info_banner(
                f"'{self.experiment_value}' has no embeddings attached.")
            self.summary.value = ""
            return

        # The query does not have to be in the pool being searched: a crop
        # loaded as a simple inference has an embedding but no row in the
        # dataset, and "what in the dataset looks like this?" is most of
        # why it is on the map at all.
        inference = None
        if self._query_uuid not in context["row_by_uuid"]:
            inference = self.ctx.state.inference_for_uuid(self._query_uuid)
        if inference is not None:
            neighbours = nearest_to_vector(
                context["embeddings"], context["uuids"], inference.vector_for(self._query_uuid),
                k=self._k, labels=context["labels"],
            )
        else:
            neighbours = nearest_neighbours(
                context["embeddings"], context["uuids"], self._query_uuid,
                k=self._k, labels=context["labels"],
            )
        if not neighbours:
            self.body.content = info_banner(
                f"{self._query_uuid} has no embedding in '{self.experiment_value}' -- a sidecar "
                f"covering only part of the dataset is normal, not an error.")
            self.summary.value = ""
            return

        row_by_uuid = context["row_by_uuid"]
        query_row = row_by_uuid.get(self._query_uuid, {})
        label_col = self.ctx.state.dataset.columns.label_col
        query_label = query_row.get(label_col, "?")

        if inference is not None:
            # No label to agree or disagree with -- that is what a simple
            # inference is. What the neighbours are is the answer instead:
            # a crop whose nearest dataset crops are all one class is one
            # the model reads as that class, whatever it actually is.
            classes = {n.label for n in neighbours if n.label is not None}
            top = neighbours[0]
            self.summary.value = (
                f"{len(neighbours)} nearest in '{self.experiment_value}', to a crop from the "
                f"'{inference.name}' inference (not in the dataset, so it has no label to compare "
                f"against). Nearest is {top.label} at {top.similarity:.3f}, down to "
                f"{neighbours[-1].similarity:.3f} across {len(classes)} class(es)."
            )
        else:
            # How many of the neighbours share the query's label. A crop
            # surrounded by its own class is unremarkable; one surrounded by
            # another class is either mislabelled or genuinely ambiguous, and
            # that ratio is the fastest way to tell which.
            same = sum(1 for n in neighbours if n.label == str(query_label))
            self.summary.value = (
                f"{len(neighbours)} nearest in '{self.experiment_value}'. "
                f"{same}/{len(neighbours)} share this crop's label ({query_label}); "
                f"similarity {neighbours[0].similarity:.3f} down to {neighbours[-1].similarity:.3f}. "
                f"A crop whose neighbours are mostly another class is either mislabelled or "
                f"genuinely ambiguous."
            )

        records = []
        for neighbour in neighbours:
            record = dict(row_by_uuid.get(neighbour.uuid, {}))
            if not record:
                continue
            record["similarity"] = round(neighbour.similarity, 4)
            records.append(record)
        if not records:
            self.body.content = info_banner("The neighbours are not in the loaded pool.")
            return

        carousel = Carousel(
            self.ctx.state.dataset, len(records),
            lambda start, end: records[start:end],
            context["crop_col"], context["uuid_col"], ctx=self.ctx,
            highlight_uuid=self._query_uuid,
            review_source="similarity",
            label_col=label_col,
            # This dialog is not only the mosaic: the embedding-space
            # picker, the "how many" dropdown and the summary sentence sit
            # above it, and the mosaic sizes itself to the window rather
            # than to the box it was handed. Without this it claims height
            # that is not there and the dialog runs off the bottom.
            chrome_height=DIALOG_CHROME_H + 170,
        )
        self.body.content = card(carousel.build())
