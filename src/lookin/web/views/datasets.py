"""The Datasets page: every CSV of the shared folders, what is in it, where
it comes from, and any two side by side.

The list on the left is the shared folders as they are, grouped by the
folder each dataset sits in, a main dataset with its derivatives under it --
each with its rows, its mappings (the label columns it has), the worst of
its checks and its MLflow link.

The right side shows the one selected in tabs: "Instances" for what does
not depend on a labelling (sources, splits, checks, every other column),
then one tab per mapping with its classes. Ctrl+click a second dataset to
compare the two, with the same tabs.

Which column is what is never guessed: it comes from the settings, common
to every dataset and overridable for one (core/datasets/config.py).
Everything comes from core/datasets/index.py, shared by every tab of the
process: a profile computed for one tab is there for all of them, and the
page redraws itself as the background worker fills profiles in.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lookin.core import library as lib
from lookin.core.datasets import config as dcfg, compare as dcmp, mlflow_link, profile as prof
from lookin.core.datasets.index import STATE_FAILED, DatasetIndex
from lookin.web import html as h
from lookin.web import minicharts as mc
from lookin.web.views.base import Dialog, View

SORTS = (("name", "Name"), ("modified", "Last modified"), ("size", "Size"), ("rows", "Rows"))
GROUP_BY = (("folder", "Folder"), ("none", "No groups"))
# The first tab of a dataset; every other tab is a mapping, keyed by its column name.
TAB_INSTANCES = "__instances__"
CLASS_ROWS = 25
CROSS_ROWS = 20
COMPARE_ROWS = 30
PARAMS_SHOWN = 14

LINK_LABELS = {
    mlflow_link.LINK_HASH: "hash-matched",
    mlflow_link.LINK_NAME: "name-matched",
    mlflow_link.LINK_PARENT: "parent-matched",
}
HEALTH_COLORS = {"error": "var(--err)", "warning": "var(--warn-dot)", "ok": "var(--ok)"}
HEALTH_LABELS = {"error": "Errors in the checks", "warning": "Warnings in the checks", "ok": "No issue found"}
SEVERITY_ICONS = {"error": ("warn", "c-err"), "warning": ("warn", "c-warn"), "info": ("info", "muted")}
ROLE_LABELS = ((dcfg.ROLE_ID, "Id column"), (dcfg.ROLE_IMAGE, "Image column"),
               (dcfg.ROLE_SPLIT, "Split column"), (dcfg.ROLE_SOURCE, "Source column"))


def _when(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""


def _pct(x) -> str:
    return "" if x is None else f"{x:.1%}"


class DatasetsView(View):
    name = "datasets"
    title = "Datasets"

    def __init__(self, session):
        super().__init__(session)
        self.selected = ""
        self.compare_with = ""
        self.query = ""
        self.sort = "name"
        self.group_by = "folder"
        self.collapsed: set = set()       # folder groups folded away
        self.tab = TAB_INSTANCES
        self.show_all: set = set()        # cards expanded past their row limit
        self.overlaps: dict = {}          # (a, b) -> row-level comparison, or {"running": True}
        self._subscribed = False

    @property
    def index(self) -> DatasetIndex:
        return DatasetIndex.get()

    # ---- lifecycle -------------------------------------------------------

    def on_shown(self) -> None:
        index = self.index
        if not self._subscribed:
            key = (self.s.sid, "datasets")
            index.subscribe(key, lambda: self.s.mark("datasets") if self.s.mode == "datasets" else None)
            self.s.on_close(lambda: index.unsubscribe(key))
            self._subscribed = True
        index.rescan()
        index.request_all()
        if self.selected:
            index.request(self.selected, first=True)
        never = [m for m in index.config.mlflow if index.mlflow.get(m.key()) is None]
        if never:
            index.refresh_mlflow()

    # ---- rendering -------------------------------------------------------

    def render(self) -> str:
        index = self.index
        if not index.config.folders:
            return self._first_run()
        if self.selected and self.selected not in index.by_path:
            self.selected = ""
        if self.compare_with and self.compare_with not in index.by_path:
            self.compare_with = ""
        if self.compare_with and self.selected:
            detail = self._compare()
        elif self.selected:
            detail = self._dataset(index.by_path[self.selected])
        else:
            detail = h.section("Datasets", h.empty("Pick a dataset on the left. Ctrl+click a second one to compare the two."))
        return (f'<div class="ds-page"><aside class="ds-side">{self._list()}</aside>'
                f'<div class="ds-main stack">{detail}</div></div>')

    def _first_run(self) -> str:
        body = (h.empty("No shared folder yet. Lookin lists every dataset CSV in the folders you point it at, "
                        "and reads their statistics in the background -- nothing is ever written there.")
                + f'<div class="row" style="padding:0 14px 14px">'
                  f'{h.button("Settings…", act_attr=self.a("settings"), kind="primary", icon_name="sliders")}</div>')
        return f'<div class="ds-page one">{h.section("Datasets", body, manual="dataset-mlflow-link")}</div>'

    # ---- the list ----

    def _matches(self, f) -> bool:
        q = self.query.strip().lower()
        return not q or all(w in f.relative().lower() for w in q.split())

    def _sort_key(self, f):
        p = self.index.profile_of(f.path)
        if self.sort == "modified":
            return (-f.mtime, f.name.lower())
        if self.sort == "size":
            return (-f.size, f.name.lower())
        if self.sort == "rows":
            return (-(p["rows"] if p else -1), f.name.lower())
        return (f.name.lower(), f.path)

    def _label(self, f) -> str:
        """The file name -- with its folder in front when another dataset
        has the same name and the list is not grouped by folder."""
        if self.group_by != "folder" and f.name in getattr(self, "_same_name", ()):
            return f"{Path(f.path).parent.name}/{f.name}"
        return f.name

    def _list(self) -> str:
        index = self.index
        by_path = index.by_path
        names = [f.name for f in index.files]
        self._same_name = {n for n in names if names.count(n) > 1}
        mains = sorted((f for f in index.files if not f.parent or f.parent not in by_path), key=self._sort_key)
        blocks: dict = {}        # group -> rows html
        shown = 0
        for f in mains:
            kids = [by_path[c] for c in f.children if c in by_path]
            kids_shown = [k for k in kids if self._matches(k) or self._matches(f)]
            if not self._matches(f) and not kids_shown:
                continue
            group = f.group() if self.group_by == "folder" else ""
            rows = blocks.setdefault(group, [])
            rows.append(self._row(f, child=False))
            rows += [self._row(k, child=True) for k in sorted(kids_shown, key=self._sort_key)]
            shown += 1 + len(kids_shown)
        body = []
        for group in sorted(blocks):
            if group:
                folded = group in self.collapsed and not self.query.strip()
                count = sum(1 for r in blocks[group])
                body.append(f'<button type="button" class="ds-group"{self.a("fold", group=group)} '
                            f'aria-expanded="{"false" if folded else "true"}" title="{h.e(group)}">'
                            f'{h.icon("chev-right" if folded else "chev-down", 13)}{h.icon("folder", 13)}'
                            f'<span class="mono ell">{h.e(group)}</span><span class="muted small">{count}</span></button>')
                if folded:
                    continue
            body += blocks[group]
        pending = index.pending()
        status = [f"{len(index.files):,} datasets"]
        if shown != len(index.files):
            status.append(f"{shown:,} shown")
        if pending:
            status.append(f"reading {pending:,}")
        notes = ""
        missing = [f for f in index.config.folders if not Path(f).expanduser().is_dir()]
        if missing:
            notes += h.banner("Cannot read: " + ", ".join(missing), "warn")
        if index.mlflow_busy:
            notes += h.spinner(index.mlflow_busy)
        for s in index.mlflow_sources().values():
            if s.error:
                notes += h.banner(f"MLflow {s.experiment}: {s.error}", "warn")
        tools = (f'<div class="ds-tools">'
                 f'{h.text_input(self.query, input_=self.i("query"), placeholder="Search datasets", cls="grow", id_="ds-q")}'
                 f'{h.select(GROUP_BY, self.group_by, change=self.c("group_by"), title="Group by")}'
                 f'{h.select(SORTS, self.sort, change=self.c("sort"), title="Sort")}'
                 f'{h.button(icon_name="refresh", act_attr=self.a("refresh"), small=True, kind="ghost", title="Scan the folders again and refresh the MLflow runs")}'
                 f'{h.button(icon_name="sliders", act_attr=self.a("settings"), small=True, kind="ghost", title="Folders, columns and MLflow experiments")}'
                 f'</div>')
        head = ('<div class="ds-lh"><span></span><span>Dataset</span><span class="r">Rows</span>'
                '<span>Mappings</span><span>MLflow</span></div>')
        empty = h.empty("No dataset CSV in the shared folders." if not self.query else "No match.")
        foot = '<div class="lst-foot">Click to open, Ctrl+click a second dataset to compare.</div>'
        return (f'<section class="card ds-list"><div class="card-h"><h2>Datasets</h2>{h.info_btn("dataset-mlflow-link")}'
                f'<span class="card-sub">{h.e(" · ".join(status))}</span></div>{tools}{notes}'
                f'<div class="ds-rows">{head}{"".join(body) or empty}</div>{foot}</section>')

    def _row(self, f, child: bool) -> str:
        index = self.index
        p = index.profile_of(f.path)
        state = index.status.get(f.path)
        link = index.link(f.path)
        if p is not None:
            health = prof.health(p)
            dot = h.dot(HEALTH_COLORS[health], HEALTH_LABELS[health])
            maps = p.get("mappings", {})
            mappings = "".join(f'<span class="ds-map mono" title="{h.e(name)}: {m["classes"]["n"]:,} classes">{h.e(name)}</span>'
                               for name, m in maps.items()) or '<span class="muted small">none</span>'
            nrows = f"{p['rows']:,}"
        else:
            failed = state and state[0] == STATE_FAILED
            dot = h.dot("var(--err)" if failed else "var(--line)", state[1] if state else "Not read yet")
            mappings = '<span class="muted small">failed</span>' if failed else '<span class="spinner sm"></span>' if state else ""
            nrows = ""
        badge = ""
        if link.status:
            badge = (f'<span class="ds-link {link.status}" title="{h.e(link.reason)} Run {h.e(link.run.label())}.">'
                     f'{h.e(LINK_LABELS[link.status])}</span>')
        cls = "ds-row" + (" child" if child else "") + (" on" if f.path == self.selected else "") \
            + (" cmp" if f.path == self.compare_with else "")
        return (f'<button type="button" class="{cls}"{self.a("select", path=f.path)} data-key="ds-{h.e(f.path)}" '
                f'title="{h.e(f.path)}">{dot}<span class="ds-n mono ell">{"└ " if child else ""}{h.e(self._label(f))}</span>'
                f'<span class="r mono">{nrows}</span><span class="ds-maps">{mappings}</span><span>{badge}</span></button>')

    # ---- one dataset ----

    def _columns_html(self, f, p) -> str:
        index = self.index
        overridden = f.path in index.config.overrides
        if p is not None:
            r = p["roles"]
            bits = [f"{label.split()[0].lower()} <b class=mono>{h.e(r[key])}</b>" for key, label in ROLE_LABELS if r.get(key)]
            bits.append("mappings " + (", ".join(f"<b class=mono>{h.e(m)}</b>" for m in r[dcfg.ROLE_MAPPINGS]) or "none"))
            text = " · ".join(bits)
            if r.get("missing"):
                text += (f' <span class="muted small">-- not in this file: '
                         f'{h.e(", ".join(r["missing"]))}</span>')
        else:
            cols = index.columns_for(f.path)
            text = " · ".join(f"{k} <b class=mono>{h.e(v)}</b>" for k, v in cols.items() if isinstance(v, str) and v)
        origin = ('<span class="chip on" title="This dataset has columns of its own">overridden</span>' if overridden
                  else '<span class="muted small">from the common settings</span>')
        edit = h.button("Edit…", act_attr=self.a("edit_columns", path=f.path), small=True, kind="link")
        return f'{text} {origin} {edit} {h.info_btn("dataset-columns")}'

    def _header(self, f, p) -> str:
        index = self.index
        link = index.link(f.path)
        content_hash = index.hash_of(f.path)
        rows = [
            ("Path", f'<span class="mono small" style="overflow-wrap:anywhere">{h.e(f.path)}</span> '
                     f'<button type="button" class="link-btn" data-copy="{h.e(f.path)}" title="Copy the path">{h.icon("copy", 13)}</button>'),
            ("File", f'{h.e(h.human_size(f.size))} · modified {h.e(_when(f.mtime))}'),
        ]
        if content_hash:
            rows.append(("Content hash", f'<span class="mono small" title="{h.e(content_hash)}">{h.e(content_hash[:16])}…</span> '
                                         f'<button type="button" class="link-btn" data-copy="{h.e(content_hash)}" title="Copy the hash">{h.icon("copy", 13)}</button>'))
        if p is not None:
            rows.append(("Rows", f'{p["rows"]:,} rows · {len(p["columns"])} columns'))
        rows.append(("Columns", self._columns_html(f, p)))
        if f.parent and f.parent in index.by_path:
            parent = index.by_path[f.parent]
            rows.append(("Derived from", f'<a href="#"{self.a("select", path=parent.path)} class="mono">{h.e(parent.name)}</a> '
                                         f'{h.button("Compare", act_attr=self.a("compare", a=parent.path, b=f.path), small=True, kind="link")}'))
        if f.children:
            kids = ", ".join(f'<a href="#"{self.a("select", path=c)} class="mono">{h.e(index.by_path[c].name)}</a>'
                             for c in f.children if c in index.by_path)
            rows.append(("Derivatives", kids))
        rows.append(("MLflow", self._mlflow_html(link)))
        used = [w for w in lib.list_workspaces() if w.mount.dataset_path == f.path]
        if used:
            rows.append(("Workspaces", " ".join(
                f'<a class="chip" href="/?workspace={h.e(w.id)}" target="_blank" rel="noopener" title="Open in a new tab">'
                f'{h.e(w.label())}</a>' for w in used)))
        others = [(x.path, x.relative()) for x in index.files if x.path != f.path]
        actions = (f'{h.button("New workspace", act_attr=self.a("new_workspace", path=f.path), small=True, icon_name="plus", kind="primary")}'
                   + (h.select([("", "Compare with…")] + others, "", change=self.c("compare_pick"), title="Compare with another dataset")
                      if others else "")
                   + h.button("Recompute", act_attr=self.a("recompute", path=f.path), small=True, icon_name="refresh",
                              title="Read the file again"))
        return h.section(f.name, h.kv_grid(rows) + self._params_html(link), right=actions, sub=f.relative())

    def _mlflow_html(self, link) -> str:
        if link.run is None:
            if not self.index.config.mlflow:
                return f'<span class="muted">No datasets experiment set.</span> {h.button("Settings…", act_attr=self.a("settings"), small=True, kind="link")}'
            return '<span class="muted">No run of the datasets experiments matches this file.</span>'
        r = link.run
        url = h.safe_href(r.ui_url())
        started = datetime.fromtimestamp(r.start_time / 1000).strftime("%Y-%m-%d %H:%M") if r.start_time else ""
        bits = [f'<span class="ds-link {link.status}" title="{h.e(link.reason)}">{h.e(LINK_LABELS[link.status])}</span>',
                f'<a href="{url}" target="_blank" rel="noopener" class="mono">{h.e(r.label())} {h.icon("external", 12)}</a>' if url
                else f'<span class="mono">{h.e(r.label())}</span>',
                f'<span class="muted small">{h.e(r.experiment)} · {h.e(started)}{" · " + h.e(r.user) if r.user else ""}</span>',
                h.info_btn("dataset-mlflow-link")]
        desc = f'<div class="small" style="margin-top:3px">{h.e(r.description)}</div>' if r.description else ""
        return " ".join(bits) + desc

    def _params_html(self, link) -> str:
        if link.run is None:
            return ""
        r = link.run
        items = [("param", k, v) for k, v in sorted(r.params.items())] + \
                [("metric", k, f"{v:.6g}") for k, v in sorted(r.metrics.items())] + \
                [("tag", k, v) for k, v in sorted(r.tags.items())]
        if not items:
            return ""
        key = "params"
        shown = items if key in self.show_all else items[:PARAMS_SHOWN]
        body = "".join(f'<tr><td class="muted small">{h.e(kind)}</td><td class="mono small">{h.e(k)}</td>'
                       f'<td class="mono small" style="overflow-wrap:anywhere">{h.e(v)}</td></tr>' for kind, k, v in shown)
        more = ""
        if len(shown) < len(items):
            more = f'<button type="button" class="hb-more"{self.a("show_all", card=key)}>Show all {len(items)}</button>'
        return (f'<details class="ds-params"{" open" if key in self.show_all else ""}><summary>Run parameters, metrics and tags '
                f'<span class="muted">({len(items)})</span></summary><table class="tbl ds-ptbl">{body}</table>{more}</details>')

    def _tabs(self, mappings: list, counts: dict) -> str:
        """Instances, then one tab per mapping; the current one falls back to
        Instances when this dataset has no such mapping."""
        if self.tab != TAB_INSTANCES and self.tab not in mappings:
            self.tab = TAB_INSTANCES
        options = [(TAB_INSTANCES, "Instances")] + [
            (m, f"{m} ({counts[m]:,} classes)" if m in counts else m) for m in mappings]
        return f'<div class="ds-tabs">{h.segmented(options, self.tab, act_for=lambda k: self.a("tab", tab=k), label="Part of the dataset")}</div>'

    def _dataset(self, f) -> str:
        index = self.index
        p = index.profile_of(f.path)
        parts = [self._header(f, p)]
        if p is None:
            state = index.status.get(f.path)
            if state and state[0] == STATE_FAILED:
                parts.append(h.banner(f"Could not read this file: {state[1]}", "error",
                                      actions=h.button("Retry", act_attr=self.a("retry", path=f.path), small=True)))
            else:
                parts.append(h.section("Statistics", h.spinner(state[1] if state else "Waiting to be read")))
            return "".join(parts)
        maps = p.get("mappings", {})
        parts.append(self._tabs(list(maps), {m: v["classes"]["n"] for m, v in maps.items()}))
        parent = index.by_path.get(f.parent) if f.parent else None
        pp = index.profile_of(parent.path) if parent is not None else None
        if self.tab == TAB_INSTANCES:
            if parent is not None:
                parts.append(self._vs_parent_rows(f, p, parent, pp))
            if p.get("sources"):
                parts.append(self._values_card("Sources", p["sources"], p["roles"][dcfg.ROLE_SOURCE], "dataset-groups"))
            if p.get("splits"):
                parts.append(self._values_card("Splits", p["splits"], p["roles"][dcfg.ROLE_SPLIT], "dataset-groups"))
            if not p.get("sources") and not p.get("splits"):
                parts.append(h.section("Sources and splits", h.empty(
                    "This file has neither the source nor the split column of the settings."), manual="dataset-groups"))
            parts.append(self._issues_card(p.get("issues", []), p, "Checks"))
            parts.append(self._columns_card(p))
        else:
            name = self.tab
            m = maps[name]
            parts.append(self._classes_card(name, m))
            if parent is not None:
                parts.append(self._vs_parent_mapping(name, f, p, parent, pp))
            if m.get("class_by_source"):
                parts.append(self._cross_card("Classes by source", m["class_by_source"], f"by_source-{name}",
                                              p["roles"][dcfg.ROLE_SOURCE]))
            if m.get("class_by_split"):
                parts.append(self._cross_card("Classes by split", m["class_by_split"], f"by_split-{name}",
                                              p["roles"][dcfg.ROLE_SPLIT]))
            parts.append(self._issues_card(m.get("issues", []), p, f"Checks of {name}"))
        return "".join(parts)

    def _kpis(self, items) -> str:
        return '<div class="ds-kpis">' + "".join(
            f'<div class="ds-kpi" title="{h.e(tip)}"><span class="ds-kv mono">{h.e(v)}</span><span class="ds-kl">{h.e(k)}</span></div>'
            for k, v, tip in items) + "</div>"

    def _classes_card(self, name, m) -> str:
        c = m["classes"]
        kpis = self._kpis([
            ("classes", f"{c['n']:,}", "Distinct labels"),
            ("max / min", f"{c['imbalance']:,.1f}x" if c["imbalance"] else "–", "Rows of the largest class over rows of the smallest"),
            ("entropy", f"{c['entropy']:.3f}" if c["entropy"] is not None else "–", "Normalized entropy: 1 = perfectly even"),
            (f"under {c['rare_min']} rows", f"{c['rare']:,}", "Classes too small to learn or measure"),
            ("median class", f"{c['median']:,.0f}" if c["median"] is not None else "–", "Rows of the median class"),
            ("no label", f"{c['missing']:,}", "Rows with an empty cell in this mapping"),
        ])
        key = f"classes-{name}"
        bars = mc.hbars([(k, n) for k, n in c["counts"]], limit=0 if key in self.show_all else CLASS_ROWS,
                        total=c["labelled"], more_attr=self.a("show_all", card=key))
        return h.section("Class balance", kpis + bars, manual="dataset-class-balance",
                         sub=f"{name} · {c['labelled']:,} labelled rows")

    def _values_card(self, title, summary, column, manual) -> str:
        key = f"values-{title}"
        total = sum(n for _, n in summary["counts"]) + summary["missing"]
        rows = [(k, n) for k, n in summary["counts"]] + ([(prof.EMPTY, summary["missing"])] if summary["missing"] else [])
        bars = mc.hbars(rows, limit=0 if key in self.show_all else CLASS_ROWS, total=total,
                        more_attr=self.a("show_all", card=key))
        return h.section(title, bars, manual=manual, sub=f"column {column} · {summary['n']:,} values")

    def _cross_card(self, title, table, key, column) -> str:
        body = mc.share_table(table["rows"], table["cols"], table["counts"],
                              row_limit=0 if key in self.show_all else CROSS_ROWS,
                              more_attr=self.a("show_all", card=key), corner="class")
        return h.section(title, body, manual="dataset-groups", sub=f"share of each class's rows, by {column}")

    def _issues_card(self, issues, p, title) -> str:
        if not issues:
            body = h.status_line("Nothing flagged.", "ok")
        else:
            body = '<div class="ds-issues">' + "".join(
                f'<div class="ds-issue">{h.icon(SEVERITY_ICONS[i["severity"]][0], 14, SEVERITY_ICONS[i["severity"]][1])}'
                f'<span>{h.e(i["message"])}</span></div>' for i in issues) + "</div>"
        return h.section(title, body, manual="dataset-quality",
                         sub=f"computed {p.get('computed_at', '')[:16].replace('T', ' ')} in {p.get('seconds', 0):.1f}s")

    def _columns_card(self, p) -> str:
        cards = []
        rows = p["rows"] or 1
        for c in p.get("column_profiles", []):
            head = (f'<div class="ds-ch"><b class="mono ell" title="{h.e(c["name"])}">{h.e(c["name"])}</b>'
                    f'<span class="muted small">{h.e(c["kind"])}</span></div>')
            facts = f'{c["unique"]:,} distinct · {c["missing"] / rows:.1%} empty'
            if c["kind"] == "numeric" and c.get("hist"):
                chart = mc.histogram(c["hist"]["edges"], c["hist"]["counts"])
                facts += f' · {c["min"]:.4g} to {c["max"]:.4g}, mean {c["mean"]:.4g}'
            elif c.get("top"):
                chart = mc.hbars([(k, n) for k, n in c["top"][:5]], total=rows - c["missing"])
            else:
                chart = ""
            cards.append(f'<div class="ds-col">{head}<div class="muted small">{h.e(facts)}</div>{chart}</div>')
        body = f'<div class="ds-cols">{"".join(cards)}</div>' if cards else h.empty("No other column.")
        return h.section("Other columns", body, manual="column-statistics", sub=f"{len(cards)} columns")

    def _vs_parent_rows(self, f, p, parent, pp) -> str:
        right = h.button("Full comparison", act_attr=self.a("compare", a=parent.path, b=f.path), small=True)
        if pp is None:
            return h.section(f"Compared with {parent.name}", h.spinner("Reading the main dataset"),
                             manual="dataset-comparison", right=right)
        delta = p["rows"] - pp["rows"]
        items = [("rows", f"{delta:+,}", f"{pp['rows']:,} in {parent.name}, {p['rows']:,} here"),
                 ("rows kept", _pct(p["rows"] / pp["rows"]) if pp["rows"] else "–", "")]
        return h.section(f"Compared with {parent.name}", self._kpis(items), manual="dataset-comparison", right=right)

    def _vs_parent_mapping(self, name, f, p, parent, pp) -> str:
        right = h.button("Full comparison", act_attr=self.a("compare", a=parent.path, b=f.path), small=True)
        title = f"Compared with {parent.name}"
        if pp is None:
            return h.section(title, h.spinner("Reading the main dataset"), manual="dataset-comparison", right=right)
        if name not in pp.get("mappings", {}):
            return h.section(title, h.empty(f"{parent.name} has no {name} column."), manual="dataset-comparison", right=right)
        d = dcmp.compare(parent.path, pp, f.path, p, rows=False)["mappings"][name]
        items = [("classes lost", f"{len(d['only_a']):,}", ", ".join(d["only_a"][:10])),
                 ("classes added", f"{len(d['only_b']):,}", ", ".join(d["only_b"][:10])),
                 ("JS divergence", f"{d['js_divergence']:.3f}" if d["js_divergence"] is not None else "–",
                  "Of the class distributions: 0 = the same shares")]
        lost = ""
        if d["only_a"]:
            lost = '<div class="small" style="margin-top:6px">Classes no longer present: ' + ", ".join(
                f'<span class="mono">{h.e(x)}</span>' for x in d["only_a"][:20]) + "</div>"
        return h.section(title, self._kpis(items) + lost, manual="dataset-comparison", right=right)

    # ---- two datasets ----

    def _compare(self) -> str:
        index = self.index
        a, b = index.by_path[self.selected], index.by_path[self.compare_with]
        pa, pb = index.profile_of(a.path), index.profile_of(b.path)
        right = h.button("Close comparison", act_attr=self.a("uncompare"), small=True, icon_name="x")
        title = h.section("Comparison", f'<div class="ds-ab"><span class="ds-tag a">A</span><span class="mono">{h.e(a.relative())}</span>'
                          f'<span class="ds-tag b">B</span><span class="mono">{h.e(b.relative())}</span>'
                          f'{h.button("Swap", act_attr=self.a("swap"), small=True, icon_name="arrows-ud", kind="ghost")}</div>',
                          manual="dataset-comparison", right=right)
        if pa is None or pb is None:
            return title + h.section("Reading", h.spinner("Reading " + " and ".join(x.name for x, px in ((a, pa), (b, pb)) if px is None)))
        d = dcmp.compare(a.path, pa, b.path, pb, rows=False)
        common = list(d["mappings"])
        parts = [title, self._tabs(common, {})]
        overlap = self._overlap(a, b)
        if self.tab == TAB_INSTANCES:
            items = [("rows A", f"{pa['rows']:,}", ""), ("rows B", f"{pb['rows']:,}", ""),
                     ("difference", f"{pb['rows'] - pa['rows']:+,}", "B minus A")]
            summary = self._kpis(items)
            if d["same_content"]:
                summary += h.banner("Same content: the two files hold the same rows.", "ok")
            for side, names in (("A", d["mappings_only_a"]), ("B", d["mappings_only_b"])):
                if names:
                    summary += (f'<div class="small">Mappings only in {side}: '
                                + ", ".join(f'<span class="mono">{h.e(n)}</span>' for n in names) + "</div>")
            parts.append(h.section("Summary", summary, manual="dataset-comparison"))
            for key, label, one in (("sources", "Sources", "source"), ("splits", "Splits", "split")):
                if d.get(key):
                    parts.append(self._dist_card(label, one, d[key], f"cmp-{key}"))
            parts.append(self._overlap_card(overlap))
        else:
            name = self.tab
            parts.append(self._dist_card(f"Classes of {name}", "class", d["mappings"][name], f"cmp-{name}"))
            parts.append(self._changes_card(name, overlap))
        return "".join(parts)

    def _dist_card(self, title, one, d, key) -> str:
        rows = sorted(d["rows"], key=lambda r: -abs(r["share_b"] - r["share_a"]))
        shown = rows if key in self.show_all else rows[:COMPARE_ROWS]
        scale = max((abs(r["share_b"] - r["share_a"]) for r in rows), default=0) or 1
        body = "".join(
            f'<tr><td class="mono ell" title="{h.e(r["value"])}">{h.e(r["value"])}</td>'
            f'<td class="r mono">{r["a"]:,}</td><td class="r mono muted">{r["share_a"]:.1%}</td>'
            f'<td class="r mono">{r["b"]:,}</td><td class="r mono muted">{r["share_b"]:.1%}</td>'
            f'<td class="r mono">{100 * (r["share_b"] - r["share_a"]):+.1f}</td>'
            f'<td style="width:140px">{mc.diverging_bar(r["share_b"] - r["share_a"], scale)}</td></tr>' for r in shown)
        table = (f'<div class="ds-tblw"><table class="tbl ds-cmp"><thead><tr><th>{h.e(one)}</th>'
                 f'<th class="r">A</th><th class="r">share A</th><th class="r">B</th><th class="r">share B</th>'
                 f'<th class="r">change (pts)</th><th></th></tr></thead><tbody>{body}</tbody></table></div>')
        if len(shown) < len(rows):
            table += f'<button type="button" class="hb-more"{self.a("show_all", card=key)}>Show all {len(rows):,}</button>'
        only = ""
        for side, values in (("A", d["only_a"]), ("B", d["only_b"])):
            if values:
                only += (f'<div class="small" style="margin-top:6px">Only in {side}: '
                         + ", ".join(f'<span class="mono">{h.e(v)}</span>' for v in values[:30])
                         + (f" and {len(values) - 30:,} more" if len(values) > 30 else "") + "</div>")
        js = d.get("js_divergence")
        sub = "largest changes of share first" + (f" · JS divergence {js:.3f}" if js is not None else "")
        return h.section(title, table + only, manual="dataset-comparison", sub=sub)

    def _overlap(self, a, b) -> dict:
        key = (a.path, b.path)
        ov = self.overlaps.get(key)
        if ov is None:
            self._start_overlap(a.path, b.path)
            ov = {"running": True}
        return ov

    def _overlap_card(self, ov) -> str:
        if ov.get("running"):
            body = h.spinner("Reading the id and label columns of both files")
        elif ov.get("error"):
            body = h.banner(ov["error"], "info")
        else:
            body = self._kpis([("in both", f"{ov['common']:,}", "Rows whose id is in A and in B"),
                               ("only in A", f"{ov['only_a']:,}", ""), ("only in B", f"{ov['only_b']:,}", "")])
        return h.section("Rows by id", body, manual="dataset-comparison")

    def _changes_card(self, name, ov) -> str:
        if ov.get("running"):
            body = h.spinner("Reading the id and label columns of both files")
        elif ov.get("error"):
            body = h.banner(ov["error"], "info")
        else:
            m = ov.get("mappings", {}).get(name, {"label_changes": 0, "top_changes": []})
            body = self._kpis([("rows in both", f"{ov['common']:,}", "Rows whose id is in A and in B"),
                               ("labels changed", f"{m['label_changes']:,}", f"Among the rows in both, in {name}")])
            if m["top_changes"]:
                body += ('<table class="tbl" style="margin-top:8px"><thead><tr><th>label in A</th><th>label in B</th>'
                         '<th class="r">rows</th></tr></thead><tbody>' + "".join(
                             f'<tr><td class="mono">{h.e(x)}</td><td class="mono">{h.e(y)}</td><td class="r mono">{n:,}</td></tr>'
                             for x, y, n in m["top_changes"]) + "</tbody></table>")
        return h.section("Labels on the shared rows", body, manual="dataset-comparison", sub=name)

    def _start_overlap(self, a: str, b: str) -> None:
        key = (a, b)
        self.overlaps[key] = {"running": True}
        index = self.index

        def work():
            try:
                pa, pb = index.compute(a), index.compute(b)
                result = dcmp._rows(a, pa["roles"], b, pb["roles"])
            except Exception as ex:  # noqa: BLE001
                result = {"error": f"Could not compare the rows: {ex}"}
            self.overlaps[key] = result
            self.refresh()
        self.s.job("Datasets comparison").run(work)

    # ---- actions -----------------------------------------------------------

    def open_dataset(self, path: str, compare: str = "") -> None:
        """What a link opens (core/links.dataset_link): one dataset, or two
        compared."""
        index = self.index
        index.ensure_scanned()
        if path in index.by_path:
            self.on_select(path=path)
            if compare in index.by_path and compare != path:
                self.on_compare(a=path, b=compare)

    def on_select(self, path: str = "", ctrl: bool = False, **_):
        if path not in self.index.by_path:
            return
        if ctrl and self.selected and path != self.selected:
            self.compare_with = "" if path == self.compare_with else path
            self.index.request(path, first=True)
            return
        self.selected = path
        self.compare_with = ""
        self.index.request(path, first=True)

    def on_compare(self, a: str = "", b: str = "", **_):
        if a in self.index.by_path and b in self.index.by_path and a != b:
            self.selected, self.compare_with = a, b
            self.index.request(a, first=True)
            self.index.request(b, first=True)

    def on_compare_pick(self, value: str = "", **_):
        if value and self.selected:
            self.on_compare(a=self.selected, b=value)

    def on_uncompare(self, **_):
        self.compare_with = ""

    def on_swap(self, **_):
        if self.compare_with:
            self.selected, self.compare_with = self.compare_with, self.selected

    def on_tab(self, tab: str = "", **_):
        self.tab = tab or TAB_INSTANCES

    def on_fold(self, group: str = "", **_):
        if group in self.collapsed:
            self.collapsed.discard(group)
        else:
            self.collapsed.add(group)

    def on_query(self, value: str = "", **_):
        self.query = value or ""

    def on_sort(self, value: str = "", **_):
        if value in dict(SORTS):
            self.sort = value

    def on_group_by(self, value: str = "", **_):
        if value in dict(GROUP_BY):
            self.group_by = value

    def on_show_all(self, card: str = "", **_):
        if card in self.show_all:
            self.show_all.discard(card)
        else:
            self.show_all.add(card)

    def on_refresh(self, **_):
        index = self.index
        index.rescan()
        index.request_all()
        if index.config.mlflow:
            index.refresh_mlflow()
        self.overlaps.clear()

    def on_retry(self, path: str = "", **_):
        self.index.retry(path)

    def on_recompute(self, path: str = "", **_):
        self.index.recompute(path)
        self.overlaps = {k: v for k, v in self.overlaps.items() if path not in k}

    def on_settings(self, **_):
        self.s.open_dialog(DatasetsSettingsDialog(self))

    def on_edit_columns(self, path: str = "", **_):
        if path in self.index.by_path:
            self.s.open_dialog(ColumnsDialog(self, path))

    def on_new_workspace(self, path: str = "", **_):
        from lookin.web.views.home import NewWorkspaceDialog

        dialog = NewWorkspaceDialog(self.s.view("home"))
        dialog.dataset = path
        dialog.name = Path(path).stem
        self.s.open_dialog(dialog)

    def settings_changed(self, mlflow: bool = False) -> None:
        """After the settings changed: the folders, the columns or the
        experiments. A profile is keyed by the columns it was read with, so
        the ones a change affects are simply not found and are read again."""
        index = self.index
        index.rescan()
        index.request_all()
        if self.selected:
            index.request(self.selected, first=True)
        if mlflow:
            index.refresh_mlflow()
        self.overlaps.clear()
        self.refresh()


def _columns_form(columns: dict) -> str:
    fields = "".join(h.text_input(columns.get(key, ""), name=key, label=label, mono=True, width="180px")
                     for key, label in ROLE_LABELS)
    mappings = "\n".join(columns.get(dcfg.ROLE_MAPPINGS) or [])
    return (f'<div class="row">{fields}</div>'
            f'<label class="ds-fld" style="margin-top:6px"><span class="muted small">Mappings: the label columns, one per line</span>'
            f'{h.textarea(mappings, name=dcfg.ROLE_MAPPINGS, rows=3)}</label>')


def _columns_from_form(args: dict) -> dict:
    out = {key: str(args.get(key) or "").strip() for key in dcfg.SINGLE_ROLES}
    out[dcfg.ROLE_MAPPINGS] = [line.strip() for line in str(args.get(dcfg.ROLE_MAPPINGS) or "").splitlines() if line.strip()]
    return out


class ColumnsDialog(Dialog):
    """The columns of one dataset, over the common ones."""

    width = 720

    def __init__(self, view: DatasetsView, path: str):
        super().__init__(f"Columns of {Path(path).name}")
        self.view = view
        self.path = path

    def body(self) -> str:
        cfg = dcfg.load()
        overridden = self.path in cfg.overrides
        note = ("This dataset has columns of its own; the others use the common settings."
                if overridden else "This dataset uses the common settings. Saving here gives it columns of its own.")
        return (f'<form class="stack" id="{self.id}-f"{self.submit("save")}>'
                f'<p class="muted small">{h.e(note)} An empty field means the file has no such column.</p>'
                f'{_columns_form(cfg.columns_for(self.path))}</form>')

    def footer(self) -> str:
        reset = (h.button("Use the common settings", act_attr=self.a("reset"))
                 if self.path in dcfg.load().overrides else "")
        return (f'{reset}<button type="submit" form="{self.id}-f" class="btn btn-primary"><span>Save for this dataset</span></button>'
                + h.button("Cancel", act_attr=self.a("close")))

    def on_save(self, **args):
        cfg = dcfg.load()
        wanted = _columns_from_form(args)
        # Only what differs from the common settings is kept: a later change
        # of the common settings still reaches the keys this one leaves alone.
        override = {k: v for k, v in wanted.items() if v != cfg.columns.get(k)}
        dcfg.set_override(self.path, override)
        self.close()
        self.view.settings_changed()

    def on_reset(self, **_):
        dcfg.set_override(self.path, {})
        self.close()
        self.view.settings_changed()


class DatasetsSettingsDialog(Dialog):
    """The shared folders, what to leave out of them, the columns of every
    dataset, and the MLflow experiments that hold datasets -- written to
    ~/.config/lookin/datasets.json (core/datasets/config.py)."""

    width = 760
    title = "Datasets settings"

    def __init__(self, view: DatasetsView):
        super().__init__()
        self.view = view
        self.folder = ""
        from lookin.web.mlflow_hub import MlflowHub
        servers = list(MlflowHub.get().servers.values())
        self.server = servers[0].uri if servers else ""
        self.experiment = ""

    def _servers(self) -> list:
        from lookin.web.mlflow_hub import MlflowHub
        return list(MlflowHub.get().servers.values())

    def body(self) -> str:
        cfg = dcfg.load()
        folders = "".join(
            f'<div class="lst-row"><span class="mono grow ell" title="{h.e(f)}">{h.e(f)}</span>'
            f'{"" if Path(f).expanduser().is_dir() else "<span class=c-err small>not found</span>"}'
            f'{h.button(icon_name="trash", act_attr=self.a("remove_folder", folder=f), small=True, kind="ghost", title="Stop listing this folder")}</div>'
            for f in cfg.folders) or h.empty("No folder yet.")
        add_folder = (f'<div class="row-nw" style="margin-top:6px">'
                      f'{h.text_input(self.folder, change=self.c("folder"), enter=h.on_enter(self.target, "add_folder"), placeholder="/path/to/shared/datasets", mono=True, cls="grow", extra=" data-force=\"1\"")}'
                      f'{h.button("Browse…", act_attr=self.a("browse"))}'
                      f'{h.button("Add", act_attr=self.a("add_folder"), kind="primary")}</div>')
        exclude = (f'<form class="stack" style="margin-top:8px"{self.submit("save_exclude")}>'
                   f'<label class="ds-fld"><span class="muted small">Leave out the CSVs whose path in the folder matches '
                   f'(one pattern per line, * matches anything, slashes included)</span>'
                   f'{h.textarea(chr(10).join(cfg.exclude), name="patterns", rows=3)}</label>'
                   f'<div>{h.button("Save the patterns", small=True, type_="submit")}</div></form>')
        columns = (f'<form class="stack"{self.submit("save_columns")}>'
                   f'<p class="muted small">Read as they are in every dataset -- nothing is guessed. A dataset '
                   f'without one of them simply lacks it; one dataset can be given columns of its own from its page.</p>'
                   f'{_columns_form(cfg.columns)}'
                   f'<div>{h.button("Save the columns", small=True, type_="submit", kind="primary")}</div></form>')
        sources = "".join(
            f'<div class="lst-row"><span class="mono grow ell">{h.e(m.experiment)}</span>'
            f'<span class="muted small mono ell">{h.e(m.uri)}</span>'
            f'{h.button(icon_name="trash", act_attr=self.a("remove_mlflow", uri=m.uri, experiment=m.experiment), small=True, kind="ghost", title="Stop matching against this experiment")}</div>'
            for m in cfg.mlflow) or h.empty("No datasets experiment yet.")
        servers = self._servers()
        if servers:
            server_pick = h.select([(s.uri, f"{s.name} ({s.uri})") for s in servers], self.server,
                                   change=self.c("server"), title="MLflow server")
        else:
            server_pick = h.text_input(self.server, change=self.c("server"), placeholder="http://mlflow:5000", mono=True)
        known = next((s for s in servers if s.uri == self.server), None)
        names = sorted({e.name for e in known.experiments}) if known and known.experiments else []
        if names:
            if self.experiment not in names:
                self.experiment = names[0]
            exp_pick = h.select([(n, n) for n in names], self.experiment, change=self.c("experiment"), title="Experiment")
        else:
            exp_pick = h.text_input(self.experiment, change=self.c("experiment"), placeholder="experiment name", mono=True)
        add_mlflow = (f'<div class="row-nw" style="margin-top:6px">{server_pick}{exp_pick}'
                      f'{h.button("Add", act_attr=self.a("add_mlflow"), kind="primary")}</div>')
        return (f'<div class="subhead">Shared folders</div>'
                f'<p class="muted small">Every CSV under these folders is listed (four levels deep). A CSV in a folder '
                f'named after another CSV is listed as derived from it. Nothing is written there.</p>'
                f'<div class="lst">{folders}</div>{add_folder}{exclude}'
                f'<div class="subhead" style="margin-top:16px">Columns of every dataset {h.info_btn("dataset-columns")}</div>{columns}'
                f'<div class="subhead" style="margin-top:16px">MLflow experiments holding datasets</div>'
                f'<p class="muted small">One run per dataset, with the CSV and a hash.txt of its content hash. '
                f'Servers come from the Home page.</p>'
                f'<div class="lst">{sources}</div>{add_mlflow}')

    def footer(self) -> str:
        return h.button("Done", act_attr=self.a("close"), kind="primary")

    def on_folder(self, value: str = "", **_):
        self.folder = value or ""

    def on_add_folder(self, value: str = "", **_):
        folder = (value or self.folder or "").strip()
        if not folder:
            return
        if not Path(folder).expanduser().is_dir():
            self.error = f"Not a folder on the machine running Lookin: {folder}"
            return
        self.error = ""
        dcfg.add_folder(folder)
        self.folder = ""
        self.view.settings_changed()

    def on_browse(self, **_):
        from lookin.web.views.files import FileBrowserDialog

        def picked(path):
            self.folder = path
            self.on_add_folder(value=path)
            self.session.mark("dialogs")
        self.session.open_dialog(FileBrowserDialog(picked, self.folder, dirs_only=True, title="Which shared folder?"))

    def on_remove_folder(self, folder: str = "", **_):
        dcfg.remove_folder(folder)
        self.view.settings_changed()

    def on_save_exclude(self, patterns: str = "", **_):
        dcfg.set_exclude((patterns or "").splitlines())
        self.session.toast("Patterns saved.", "ok")
        self.view.settings_changed()

    def on_save_columns(self, **args):
        dcfg.set_columns(_columns_from_form(args))
        self.session.toast("Columns saved: the datasets are read again with them.", "ok")
        self.view.settings_changed()

    def on_server(self, value: str = "", **_):
        self.server = (value or "").strip()
        self.experiment = ""

    def on_experiment(self, value: str = "", **_):
        self.experiment = (value or "").strip()

    def on_add_mlflow(self, **_):
        if not self.server or not self.experiment:
            self.error = "Pick a server and an experiment."
            return
        self.error = ""
        dcfg.add_mlflow(self.server, self.experiment)
        self.view.settings_changed(mlflow=True)

    def on_remove_mlflow(self, uri: str = "", experiment: str = "", **_):
        dcfg.remove_mlflow(uri, experiment)
        self.view.settings_changed()
