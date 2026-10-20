"""`stbench eval compare A B ...`: results side by side, with paired differences (docs/eval.md).

Each argument is a results document or the run folder that holds one; the first is the base of the comparison.
Per layout each result's headline, its `now` and `memory` columns, each ability (with its number of questions) and each
family are shown, and each later result's difference from the base over the questions both answered, paired on
(case, qid), with a 95% CI from a bootstrap over scripts (`score_stats.paired_difference`). The headline is the mean of
the two columns, which count equally, each the mean of its cells by their weights, an ability in a column (a diagnostic
ability counted in none; a (job, ability, column) cell under 20 questions in the dataset weighing n / 20: the `thin`
cells the results record, or those of `--cells-from DEVROOT` for all of them); the headline's and the columns'
differences are those statistics of the differences over the questions both answered. A cell line shows the cell's
counted questions (a thin cell's left out) and pools their paired differences; a diagnostic cell, with none counted,
shows all its questions. An ability asked in both columns has a line for each (`clearance, now`). Every family line
pools its own questions. A second table gives the same for
the critical answerable rows, the situational answerable ones and the unknown ones apart, so a difference that is on
the critical rows is not lost in the headline among the others; a line under it says whether each later result's
critical difference is shown: the lower bound of its interval is above 0 (the rule declared for the conditioning
comparison, a correct brief against a neutral one). When the results differ in `during` the off arm holds no during
rows, so the pairs are the end questions alone, and every mean shown is over the end questions too (the tables say
so): D2 covers end answers only.

Refused (exit 2, saying why), before anything is printed: a document that is not a results document; a case two
results scored on different compiles (qids are random per compile, so two compiles share no question: `same_compile`);
a result sharing no case with the first; and results that differ in more than one lever group, `agent` (system and
reasoning and the memory and compaction limit it ran with), `channel`, `brief`, `during` or `text_only` (VISION rule
13: one axis at a time), so a difference is the effect of one thing: a memory row and the reference row of the same
model and effort differ in one lever, the agent, and so do two rows of one memory that differ in the compaction limit (a
row with a limit is named `compacts at N tokens` and never stands in for one without).
"""

from __future__ import annotations

import string
from collections.abc import Callable
from pathlib import Path

from . import ability_counts
from .cell_weights import Thin, thin_from
from .eval_report import HEADLINE_RULE, NOT_COMPARABLE, cell, interval, lever_name, sandbox_name, signed, system_name
from .eval_results import load
from .records import SCOPES
from .score_stats import aggregate, paired_difference

MARK = "*** {} ***"
RELEVANCE_ROWS = (("answerable, critical", "critical"), ("answerable, situational", "situational"),
                  ("unknown", "unknown"))   # a line of the relevance table, and the `relevance` it reads
MAX_RESULTS = len(string.ascii_uppercase)
LEVER_GROUPS = {"agent": ("system", "reasoning", "memory", "compact_at"), "channel": ("channel",), "brief": ("brief",),
                "during": ("during",), "text_only": ("text_only",)}


class Refused(ValueError):
    """The results cannot be compared."""


def _differing(a: dict, b: dict) -> list[str]:
    """The lever groups in which two results differ."""
    return [group for group, keys in LEVER_GROUPS.items() if any(a.get(k) != b.get(k) for k in keys)]


def same_compile(paths: list[str], docs: list[dict]) -> None:
    """Refused unless the results were scored on one compile. Qids are random per compile, so for each case that two
    results both hold rows of, their qid sets must meet (a `during` off arm holds a subset of the on arm's, so equal
    sets are not asked). A case with rows in only one result, or none, is skipped."""
    first: dict[str, tuple[str, set[str]]] = {}   # case id -> (the file that first held its rows, their qids)
    for path, doc in zip(paths, docs, strict=True):
        qids: dict[str, set[str]] = {}
        for row in doc["rows"]:
            qids.setdefault(row["case"], set()).add(row["qid"])
        for case, found in qids.items():
            there = first.setdefault(case, (path, found))
            if not there[1] & found:
                raise Refused(f"{path} and {there[0]} were scored on different compiles: case {case} shares no "
                              "question (qids are random per compile). Compare results made on the same compile.")


def load_all(paths: list[str]) -> tuple[list[dict], list[str]]:
    """The validated results and the names the tables call them (A, B, ...). Refused as the module says."""
    if not 2 <= len(paths) <= MAX_RESULTS:
        raise Refused(f"compare takes between 2 and {MAX_RESULTS} results, not {len(paths)}")
    try:
        docs = [load(p) for p in paths]
    except ValueError as error:
        raise Refused(str(error)) from None
    same_compile(paths, docs)
    ids = [{c["id"] for c in doc["cases"]} for doc in docs]
    for path, doc, found in zip(paths[1:], docs[1:], ids[1:], strict=True):
        if not found & ids[0]:
            raise Refused(f"{path} shares no case with {paths[0]}, so nothing can be paired")
        if len(groups := _differing(docs[0], doc)) > 1:
            raise Refused(f"{path} and {paths[0]} differ in more than one lever ({', '.join(groups)}): a comparison "
                          "changes one axis at a time (VISION rule 13), so run the arms that differ in one")
    return docs, list(string.ascii_uppercase[:len(docs)])


def _rows(doc: dict, layout: str, end_only: bool) -> list[dict]:
    return [r for r in doc["rows"] if r["layout"] == layout and (r["when"] == "end" or not end_only)]


def _paired(rows: list[dict], base: list[dict], where: dict | None, thin: Thin, mode: str = "pooled"
            ) -> dict | None:
    """`rows` minus `base` over the questions both answered (paired on (case, qid)) that `where` keeps; None when there
    are none. The `mode` of the statistic: `headline` (the headline's of the differences: the difference of the two
    headlines, or of the two columns where `where` names a scope), `counted` (pooled over the questions the means count)
    or `pooled` (all the questions)."""
    try:
        return paired_difference(rows, base, where=where, headline=mode == "headline", counted=mode == "counted",
                                 thin=thin)
    except ValueError:   # no rows, or no question in common
        return None


def _difference(rows: list[dict], base: list[dict], where: dict | None, thin: Thin, mode: str) -> str:
    d = _paired(rows, base, where, thin, mode)
    return "-" if d is None else f"{signed(d['mean'])} {interval(d['ci'])} over {d['n']}"


def _thin(doc: dict, layout: str) -> dict:
    """The (job, ability, column) cells below weight 1 that the result's aggregate of `layout` weighed, with their
    weights."""
    agg = doc["layouts"].get(layout)
    return thin_from(agg["thin"]) if agg else {}


def _grid(names: list[str], aggs: list[dict | None], rows: list[list[dict]], head: str,
          lines: list[tuple[str, Callable[[dict], dict | None], dict | None, str]], thin: Thin) -> list[str]:
    """A table: a column a result with its difference from the base. A line is (its name, the statistic of an aggregate,
    the `where` of its pairs, the `mode` of its difference: `_paired`'s)."""
    columns = [head]
    for name in names:
        columns += [name] + ([f"{name} - {names[0]}"] if name != names[0] else [])
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for label, stat_of, where, mode in lines:
        cells = [label]
        for i, agg in enumerate(aggs):
            stat = None if agg is None else stat_of(agg)
            cells.append(cell(stat) if stat and stat["mean"] is not None else "-")
            if i:
                cells.append(_difference(rows[i], rows[0], where, thin, mode) if stat and aggs[0] else "-")
        out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def _shown(names: list[str], rows: list[list[dict]], thin: Thin) -> list[str]:
    """For each later result, whether its difference from the base on the critical answerable rows is shown: its
    interval's lower bound is above 0 (the rule declared in DECISIONS, "0.5.0 declared")."""
    out = []
    for name, found in zip(names[1:], rows[1:], strict=True):
        d = _paired(found, rows[0], {"relevance": "critical"}, thin)
        head = f"- {name} - {names[0]} on the critical answerable rows:"
        if d is None or d["ci"] is None:
            out.append(f"{head} not shown ({'no pairs' if d is None else 'one script, so no interval'})")
        else:
            out.append(f"{head} {'shown' if d['ci'][0] > 0 else 'not shown'} (lower bound {signed(d['ci'][0])}, "
                       f"over {d['n']} pairs)")
    return out


def _cells_of(entry: dict) -> dict:
    """An ability's cells by scope; an older result holds none, and its ability stands as one (key None)."""
    return entry.get("cells") or {None: entry}


def _cell_lines(aggs: list[dict | None]) -> list[tuple[str, Callable[[dict], dict | None], dict, str]]:
    """A line a cell (an ability in a column) of any result: named by the ability, and by its column where the ability
    has several; its statistic the mean of its counted questions, or of all of them where none is counted; its pairs
    the cell's rows (the ability's, for an older result), pooled over the counted ones where the base counts
    any."""
    found: dict[str, dict] = {}   # ability -> scope -> the base's cell (the first result that has it)
    for agg in filter(None, aggs):
        for ability, entry in agg["by"]["ability"].items():
            for scope, cell_ in _cells_of(entry).items():
                found.setdefault(ability, {}).setdefault(scope, cell_)
    lines = []
    for ability, cells in found.items():
        for scope, base in cells.items():
            kept = f", counted {base['counted']}" if 0 < base["counted"] < base["n"] else ""
            name = f"{ability}, {scope}" if len(cells) > 1 else ability
            label = f"{name} (n {base['n']}{kept}{', diagnostic' if base['diagnostic'] else ''})"
            where = {"ability": ability} | ({} if scope is None else {"scope": scope})

            def stat(agg: dict, ability=ability, scope=scope) -> dict | None:
                entry = agg["by"]["ability"].get(ability)
                return None if entry is None else _cell_stat(_cells_of(entry).get(scope))
            lines.append((label, stat, where, "counted" if base["counted"] else "pooled"))
    return lines


def _cell_stat(entry: dict | None) -> dict | None:
    """What a cell line shows: the mean of its counted questions, or of all of them where none is counted."""
    if entry is None or not entry["counted"]:
        return entry
    return {"mean": entry["counted_mean"], "ci": entry["counted_ci"]}


def _layout(docs: list[dict], names: list[str], layout: str, end_only: bool, cells: dict | None = None
            ) -> list[str]:
    """The tables of a layout: the headline, the two columns, a line a cell (an ability in a column) and a line a
    family, then the headline of each relevance class (the critical answerable rows apart from the situational and the
    unknown ones) with the paired difference on each."""
    rows = [_rows(doc, layout, end_only) for doc in docs]
    thin_of = (lambda doc: _thin(doc, layout)) if cells is None else (lambda doc: cells)
    thin = thin_of(docs[0])
    aggs = [aggregate(r, thin=thin_of(doc)) if r else None for r, doc in zip(rows, docs, strict=True)]
    families = list(dict.fromkeys(f for a in aggs if a for f in a["by"]["family"]))
    lines = [f"## {layout}" + (": end questions only" if end_only else ""), ""]
    by_family = [("**headline**", lambda a: a["headline"], None, "headline")]
    by_family += [(f"`{scope}`", lambda a, scope=scope: a["columns"].get(scope), {"scope": scope}, "headline")
                  for scope in SCOPES if any(a and scope in a["columns"] for a in aggs)]
    by_family += _cell_lines(aggs)
    by_family += [(family, lambda a, family=family: a["by"]["family"].get(family), {"family": family}, "pooled")
                  for family in families]
    lines += _grid(names, aggs, rows, "family", by_family, thin)
    by_relevance = [(label, lambda a, c=c: a["by"]["relevance"].get(c), {"relevance": c}, "pooled")
                    for label, c in RELEVANCE_ROWS]
    lines += _grid(names, aggs, rows, "relevance", by_relevance, thin)
    return lines + _shown(names, rows, thin) + [""]


def render(docs: list[dict], names: list[str], cells: dict | None = None) -> str:
    """The comparison as markdown: who is who, the notes of each result, then a table per layout. `cells` are the
    cells below weight 1 of the dataset to take the weights from (`ability_counts.thin_rule`), for every result in place
    of those the results record."""
    end_only = len({d["during"] for d in docs}) > 1
    layouts = sorted({name for d in docs for name, agg in d["layouts"].items() if agg})
    lines = ["# SpatiotemporalBench: comparison", ""]
    lines += [MARK.format(f"{n}: {note}") for n, d in zip(names, docs, strict=True) for note in d["notes"]]
    lines += ["## Results", ""]
    lines += [f"- {n} = {system_name(d)}, {lever_name(d)}, sandbox {sandbox_name(d)}, "
              f"{sum(c['status'] == 'complete' for c in d['cases'])} of {len(d['cases'])} cases finished"
              for n, d in zip(names, docs, strict=True)] + [""]
    lines += [f"Differences are each column minus {names[0]}, over the questions both answered, paired by (case, "
              "question), with a 95% CI from a bootstrap over scripts and the number of pairs after it. Means are the "
              "reported score.", ""]
    if end_only:
        lines += ["The results differ in `during`: the off arm asked no during question, so every mean and every "
                  "difference below is over the end questions only.", ""]
    lines += [NOT_COMPARABLE, "", HEADLINE_RULE, ""]
    for layout in layouts:
        lines += _layout(docs, names, layout, end_only, cells)
    return "\n".join(lines).rstrip("\n") + "\n"


def compare(paths: list[str], out: Path | None = None, cells_from: Path | None = None) -> str:
    """The markdown comparison of the results at `paths`; also written to `out` when given. With `cells_from`, a
    dataset (the dev root), the cells and their weights are its (`--cells-from`), not the results'. Refused when they
    cannot be compared."""
    docs, names = load_all(paths)
    if cells_from is not None:
        try:
            ability_counts.require_cases(cells_from)
        except ValueError as error:
            raise Refused(str(error)) from error
    text = render(docs, names, None if cells_from is None else ability_counts.thin_rule(cells_from)[0])
    if out is not None:
        Path(out).write_text(text, encoding="utf-8", newline="\n")
    return text
