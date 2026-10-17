"""Statistics: a chart per column the user picks (nothing by default -- a
wide CSV has dozens of columns, and charting all of them on every load is
slow and rarely wanted). Same charts as lookin.ui.stats_view: a histogram
for a numeric column, a bar per value for a categorical one, and past
RANKED_CATEGORIES_MIN values the sorted sizes on a log scale with the
extremes named under it.
"""
from __future__ import annotations

import numpy as np

from lookin.core.chart_style import subplots
from lookin.ui.components import GRID_NONE, style_axes
from lookin.ui.stats_view import RANKED_CATEGORIES_MIN, RANKED_NAMED, RANKED_SMALL
from lookin.web import html as h
from lookin.web.views.base import RebuildView
from lookin.web.views.media import fig_html

BAR_COLOR = "#2550C8"


class StatsView(RebuildView):
    name = "statistics"
    title = "Statistics"

    def __init__(self, session):
        super().__init__(session)
        self.active_columns: list = []
        self.charts: dict = {}          # column -> (html, notes)
        self.add_choice = ""

    def subtitle(self) -> str:
        return "A chart per column you pick; nothing shows until you add one."

    def _available(self):
        state = self.state
        if state.dataset is None:
            return []
        kinds = state.dataset.column_kinds()
        cols = [state.dataset.columns.label_col] + kinds.numeric + kinds.binary + kinds.categorical
        return [c for c in cols if c not in self.active_columns]

    def render(self) -> str:
        state = self.state
        if state.dataset is None:
            return h.section("Column statistics", h.banner("Load a dataset in the Sources tab first.", "info"),
                             manual="column-statistics")
        available = self._available()
        if self.add_choice not in available:
            self.add_choice = available[0] if available else ""
        kinds = state.dataset.column_kinds()
        controls = (f'<div class="row">{h.select([(c, c) for c in available], self.add_choice, change=self.c("choose"), label="Add a chart for column", cls="mono")}'
                    f'{h.button("Add", act_attr=self.a("add"), kind="primary", icon_name="plus", disabled=not available)}'
                    f'</div>')
        skipped = (f'<div class="muted small" style="margin-top:6px">Not offered above (too many unique values): '
                   f'{h.e(", ".join(kinds.high_cardinality))}</div>') if kinds.high_cardinality else ""
        head = h.section("Column statistics", controls + skipped, manual="column-statistics",
                         sub="add a chart for each column you want to inspect")
        cards = []
        if not self.active_columns:
            cards.append(h.banner("No charts shown yet -- pick a column above and click Add.", "info"))
        for col in self.active_columns:
            item = self.charts.get(col)
            body = h.spinner("Rendering...") if item is None else item[0] + "".join(
                f'<div class="muted small">{h.e(n)}</div>' for n in item[1])
            cards.append(f'<section class="card" data-key="st-{h.e(col)}"><div class="card-h"><h2 class="mono">{h.e(col)}</h2>'
                         f'<div class="card-r">{h.button(icon_name="x", act_attr=self.a("remove", column=col), small=True, kind="ghost", title="Remove this chart")}</div></div>'
                         f'<div class="card-b">{body}</div></section>')
        grid = f'<div class="grid2" style="grid-template-columns:repeat(auto-fit,minmax(560px,1fr))">{"".join(cards)}</div>'
        return f'<div class="stack">{self.stale_html()}{head}{self.progress_html()}{grid}</div>'

    # ---- building ----

    def _chart(self, col: str, series, is_numeric: bool) -> tuple:
        fig, ax = subplots(figsize=(9.5, 3.2))
        notes = []
        if is_numeric:
            # An inf has no bin: numpy refuses the whole histogram, and the
            # tab's every chart with it. Drawn without, and said so.
            finite = series[np.isfinite(series.to_numpy(dtype=float))]
            if len(finite) < len(series):
                notes.append(f"{len(series) - len(finite):,} infinite value(s) not drawn.")
            ax.hist(finite, bins=40, color=BAR_COLOR, edgecolor="white")
            ax.set_title(f"{col}", fontsize=12)
            ax.set_ylabel("count", fontsize=10)
            style_axes(ax)
        elif series.nunique() > RANKED_CATEGORIES_MIN:
            counts = series.value_counts().sort_values(ascending=False)
            ranks = np.arange(1, len(counts) + 1)
            ax.fill_between(ranks, counts.values, step="mid", color=BAR_COLOR, alpha=0.35)
            ax.step(ranks, counts.values, where="mid", color=BAR_COLOR)
            ax.set_yscale("log")
            ax.set_xlim(0.5, len(counts) + 0.5)
            ax.set_title(f"{col} -- {len(counts):,} values, largest first", fontsize=12)
            ax.set_xlabel("rank", fontsize=10)
            ax.set_ylabel("count (log scale)", fontsize=10)
            style_axes(ax)

            def named(part) -> str:
                return ", ".join(f"{label} ({int(n):,})" for label, n in part.items())

            notes = [f"Largest: {named(counts.head(RANKED_NAMED))}",
                     f"Smallest: {named(counts.tail(RANKED_NAMED)[::-1])}",
                     f"Median {int(counts.median()):,} per value; "
                     f"{int((counts < RANKED_SMALL).sum()):,} value(s) with fewer than {RANKED_SMALL}."]
        else:
            counts = series.value_counts().sort_values(ascending=False)
            ax.bar(counts.index.astype(str), counts.values, color=BAR_COLOR)
            ax.set_title(f"{col}", fontsize=12)
            ax.set_ylabel("count", fontsize=10)
            for label in ax.get_xticklabels():
                label.set_fontsize(9)
                if len(counts) > 12:
                    label.set_rotation(45)
                    label.set_ha("right")
            style_axes(ax, x=GRID_NONE)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        return fig_html(self.s, fig, col), notes

    def compute(self) -> None:
        state = self.state
        if state.dataset is None:
            self.charts = {}
            return
        df = state.dataset.df
        numeric = set(state.dataset.column_kinds().numeric)
        valid = set(df.columns) | {state.dataset.columns.label_col}
        self.active_columns = [c for c in self.active_columns if c in valid]
        charts = {}
        for i, col in enumerate(self.active_columns):
            self.set_progress(f"Rendering charts... {col} ({i + 1}/{len(self.active_columns)})")
            charts[col] = self._chart(col, df[col].dropna(), col in numeric)
            self.charts = {**self.charts, col: charts[col]}
        self.charts = charts

    # ---- actions ----

    def on_choose(self, value: str = "", **_):
        self.add_choice = value

    def on_add(self, **_):
        col = self.add_choice
        if col and col not in self.active_columns:
            self.active_columns.append(col)
            self.rebuild()

    def on_remove(self, column: str = "", **_):
        if column in self.active_columns:
            self.active_columns.remove(column)
            self.charts.pop(column, None)
        self.refresh()
