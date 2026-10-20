"""`results.md`: the human-readable form of a results document (docs/eval.md).

The notes come first (partial, unsandboxed, remote memory, a usage report with problems, the budget's stop), then the
facts of the run, then one block per layout (a layout is a world of its own: its means are never pooled with another's):
the headline (the mean of the `now` and `memory` columns, which count equally, each the mean of its cells by their
weights: an ability in a column) with its 95% interval, the same score above the floor (less the unknown share, what
answering `unknown` throughout scores), the two columns with their counts, the secondary means (the cells' mean, the
counted rows pooled) and the own-path split beside it, a line per cell (an ability in a column, so clearance has two)
with its count and its weight in the headline (a diagnostic ability, `prediction`, flagged; the (job, ability, column)
cells under 20 questions named with their weights n / 20, and the cells the means count listed), a line per family and
per case length, the answerable questions by relevance (critical, situational) beside the unknown ones, the outcome
counts with the confidently-wrong and abstention rates, the credit bands by ability beside the unpenalised credit and
the abstention bound, the mean by kind (harm, help, situation, reuse: the credit's split beside the ability one, in no
cell of the headline), the mean by the age of the evidence (`records.AGE_BANDS`, overall and per job), the share of
the harm rows answered that are catastrophes (overall and per job, with the counts), the abstention (0.11.0: recall,
precision and the `unknown` rate on answerable questions, overall, per job and per family, with Wilson intervals:
`score_abstain`), and what the layout's finished cases cost in tokens, dollars and wall time. Everything is ASCII, so
it prints on any console.

Every mean is of the reported score (0.10.0): the penalised score `2s - 1`, less a catastrophe's weight where an answer
would have ended in harm (`score.reported`). A result of an earlier build holds no catastrophe, no age band and no
`score_gate`, and its report simply lacks those two sections.

Two diagnostics stand beside the reported headline, never in it (0.7.0, FD-0.7.0 review, change 11), both read from the
rows alone and counted as the headline is (its thin cells, its bootstrap over scripts):
- the unpenalised credit: the headline's statistic over the credit `s` of each answer (no `s - (1 - s)k`), so what the
  penalty costs a system shows beside what its answers earned;
- the abstention bound (`bound_score`): the score the same answers would get had the system answered `unknown` exactly
  where an answerable question's answer scored below 0, that score becoming 0. An unknown question's score is kept: a
  wrong answer there stays negative, so the bound never credits the right answer itself.
The credit bands (`credit_band`) split each ability's questions by the credit `s`, the tiers of the credit's four
quarters: full (`s` = 1), within need (at least 3/4), right thing (at least 1/2), direction (at least 1/4), miss and
`unknown`.
The helpers `signed`, `interval`, `rate` and `cell` format a statistic for `eval_compare`, `cmd_pilot` and
`eval_sensitivity` too.
"""

from __future__ import annotations

from collections.abc import Callable

from .abilities import MIN_ABILITY_N
from .case import LENGTHS
from .cell_weights import thin_from
from .records import AGE_BANDS, SCOPES
from .score import EPS
from .score_abstain import groups
from .score_stats import ability, aggregate

HEADLINE_RULE = ("The headline is the mean of the `now` and `memory` columns, each weighing the largest weight of its "
                 "cells: the two count equally whatever their numbers of questions (VISION rule 4), unless every cell "
                 "of one is thin. A column is the mean of its cells by their weights, a cell being an ability in that "
                 "column (an ability asked in both has a cell in each), each pooled over its own questions. A cell's "
                 f"weight is min(1, n / {MIN_ABILITY_N}) for its n questions on a job in the dataset: a cell of "
                 f"{MIN_ABILITY_N} or more counts in full and a thinner one by its share. The score above the floor "
                 "is weighed the same way, and every other line pools its own questions. A diagnostic ability is "
                 "shown and counted in no mean.")
NOT_COMPARABLE = ("Each layout is scored apart from the others (its own map, kinds and jobs), and no number pools "
                  "layouts. The mean is the reported score: a wrong answer costs more than `unknown`, which scores "
                  "0, so it can be negative, and an answer that would have ended in harm (a catastrophe) scores "
                  "minus its weight.")
BOUND_RULE = ("The abstention bound is the score the same answers would get had the system answered `unknown` exactly "
              "where an answerable question's answer scored below 0: that score becomes 0. A score on an unknown "
              "question is kept. The unpenalised credit is the headline's statistic over the credit of each answer "
              "(1 for a right `unknown`, 0 for any other answer on an unknown question), with no penalty. A credit "
              "band is the credit `s` of an answer, the tiers of its four quarters: full is 1, within need at least "
              "3/4, right thing at least 1/2, direction at least 1/4, a miss less; `unknown` is every answer of "
              "`unknown`. A share is of the ability's questions.")
AGE_ROWS = (*AGE_BANDS, "none")   # the evidence-age bands in report order, a row with no band last
BAND_FLOORS = (("full", 1.0), ("within need", 0.75), ("right thing", 0.5), ("direction", 0.25))   # the quarters' sums
BANDS = (*(name for name, _ in BAND_FLOORS), "miss", "unknown")
TOKEN_NAMES = (("tokens_in", "in"), ("tokens_cached_in", "of them read from a cache"),
               ("tokens_cache_write_in", "written to one"), ("tokens_out", "out"))


def count(n: int, noun: str) -> str:
    """`1 case`, `3 cases`."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def signed(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def interval(ci: list | tuple | None) -> str:
    return "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def rate(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def cell(stat: dict) -> str:
    """`+0.123 [+0.050, +0.190]` for a `{mean, ci}`."""
    return f"{signed(stat['mean'])} {interval(stat['ci'])}"


def sandbox_name(doc: dict) -> str:
    return {"docker": "docker (the verified mode)", "none": "none (UNSANDBOXED: not a verified score)",
            "in-process": "in-process (the benchmark's own code)"}[doc["sandbox"]]


def _tokens(count: int) -> str:
    """A token count as the reports write it: `50k` for a round thousand, else the number."""
    return f"{count // 1000}k" if count % 1000 == 0 else str(count)


def system_name(doc: dict) -> str:
    """The system of a results document as the reports name it: `codex:<model>`, `with memory <name>` when the agent
    kept one and `compacts at 50k tokens` when Codex compacted its thread at a limit of its own (`memory` and
    `compact_at`, absent from a reference row)."""
    name = doc["system"] + (f" with memory {doc['memory']}" if doc.get("memory") else "")
    return name if doc.get("compact_at") is None else f"{name}, compacts at {_tokens(doc['compact_at'])} tokens"


def lever_name(doc: dict) -> str:
    """The run's options in a line: `rgb_pose channel, correct brief, during on`."""
    return (f"{doc['channel']} channel, {doc['brief']} brief, during {'on' if doc['during'] else 'off'}"
            + (", text only (no image)" if doc["text_only"] else ""))


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _stat_row(name: str, stat: dict) -> list[str]:
    return [name, str(stat["n"]), signed(stat["mean"]), interval(stat["ci"])]


def _length_order(length: str) -> tuple[int, str]:
    """Short to long (`case.LENGTHS`' order), a length it does not know last."""
    return (list(LENGTHS).index(length) if length in LENGTHS else len(LENGTHS), length)


def _counted(agg: dict) -> int:
    """The questions the means count: those of the two columns."""
    return sum(column["n"] for column in agg["columns"].values())


def bound_score(row: dict) -> float:
    """The reported score of the row had its answer been `unknown` when an answerable question's answer scored below
    0: `max(0, score)` there (0 is what `unknown` scores on an answerable question; a catastrophe, which scores below 0,
    becomes 0). A score on an unknown question is kept: a wrong answer there stays negative."""
    return max(0.0, row["score"]) if row["slice"] == "answerable" else row["score"]


def credit_band(row: dict) -> str:
    """The band of the row's answer in `BANDS`: `unknown` when the answer was `unknown` (the outcome `abstained`, or
    `correct` on an unknown question), else by the credit `s`: `full` at 1, `within need` from 3/4, `right thing` from
    1/2, `direction` from 1/4 (an answer on the right side and nothing more), `miss` below."""
    if row["outcome"] == "abstained" or (row["slice"] == "unknown" and row["outcome"] == "correct"):
        return "unknown"
    s = row["s"]
    return next((band for band, floor in BAND_FLOORS if s >= floor - EPS), "miss")


def restated(rows: list[dict], agg: dict, value: Callable[[dict], float]) -> dict:
    """`{mean, ci}` of the layout's headline over the same `rows` with each row's `score` replaced by `value(row)`,
    counted as the layout's own headline was (`agg`, its aggregate: the same thin cells): the same statistic."""
    return aggregate([{**r, "score": value(r)} for r in rows], thin=thin_from(agg["thin"]),
                     thin_ignored=agg["thin_ignored"])["headline"]


def _floor_row(agg: dict) -> list[str]:
    """The score above the floor, with the unknown share it is taken from (the score of answering `unknown` always),
    weighted as the headline is: both are the same linear statistic, so the share is their difference."""
    means = agg["headline"]["mean"], agg["above_floor"]["mean"]
    share = "n/a" if None in means else rate(means[0] - means[1])
    return _stat_row(f"above the floor (the headline less the unknown share, {share})",
                     {**agg["above_floor"], "n": _counted(agg)})


def _relevance_rows(by: dict) -> list[list[str]]:
    """The answerable questions (the `slice`), then those of them that are critical and situational, then the unknown
    ones; a class no question falls in is left out."""
    found = [("answerable", by["slice"].get("answerable")), ("answerable, critical", by["relevance"].get("critical")),
             ("answerable, situational", by["relevance"].get("situational")),
             ("unknown", by["slice"].get("unknown"))]
    return [_stat_row(name, stat) for name, stat in found if stat]


def _ability_rows(agg: dict) -> list[list[str]]:
    """A line a cell (an ability in a column): its column, its questions and those counted in the means, its weight in
    the headline (`-` where none counts), its mean over all its questions with its interval, and its mean over the
    counted ones alone (`-` where none is); a diagnostic one says so. An older result holds no cells: its
    ability stands as one, in the scopes it was asked in."""
    out = []
    for name, stat in agg["by"]["ability"].items():
        for scope, cell_ in (stat.get("cells") or {stat["scope"]: {**stat, "weight": None}}).items():
            flag = " (diagnostic: counted in no mean)" if cell_["diagnostic"] else ""
            kept = signed(cell_["counted_mean"]) if cell_["counted"] else "-"
            weight = "-" if not cell_["weight"] else rate(cell_["weight"])
            out.append([name + flag, scope, str(cell_["n"]), str(cell_["counted"]), weight, signed(cell_["mean"]), kept,
                        interval(cell_["ci"])])
    return out


def _kind_rows(agg: dict) -> list[list[str]]:
    """A line a kind: its questions, its mean over all of them with its interval; a kind the means count none of (every
    family of it of a diagnostic ability, `abilities.DIAGNOSTIC`; none is since 0.11.0, when help became `open_heading`
    and `way_to`, both counted) says so, as an ability does."""
    return [_stat_row(name + (" (diagnostic: counted in no mean)" if stat["diagnostic"] else ""), stat)
            for name, stat in agg["by"]["kind"].items()]


def _age_rows(agg: dict) -> list[list[str]]:
    """A line a band of the evidence's age (`AGE_ROWS`) over every job of the layout, then a line a (job, band): the
    questions and their mean reported score with its interval (the job's, without one). Empty for an aggregate of an
    earlier build."""
    overall = agg["by"].get("age_band") or {}
    rows = [_stat_row(band, overall[band]) for band in AGE_ROWS if band in overall]
    for job, bands in (agg.get("job_age_band") or {}).items():
        rows += [_stat_row(f"{job}: {band}", bands[band]) for band in AGE_ROWS if band in bands]
    return rows


def _catastrophe_lines(agg: dict) -> list[str]:
    """The catastrophe rate (the share of the harm rows answered that are catastrophes, with the counts), overall and
    per job; empty for an aggregate of an earlier build."""
    found = agg.get("catastrophe_rate")
    if not found:
        return []

    def line(name: str, stat: dict) -> str:
        return (f"- {name}: {rate(stat['rate'])} {interval(stat['ci'])}, {count(stat['n'], 'catastrophe')} of "
                f"{count(stat['of'], 'harm row')} answered")

    return ["### Catastrophes", "", line("all jobs", found), *(line(job, stat) for job, stat in found["jobs"].items()),
            ""]


def _cells_line(agg: dict) -> list[str]:
    """The cells the means count, by column: `now: distance, clearance; memory: memory, relevance, clearance`."""
    found = {scope: [name for name, stat in agg["by"]["ability"].items()
                     if (stat.get("cells") or {}).get(scope, {}).get("counted")] for scope in SCOPES}
    listed = "; ".join(f"{scope}: {', '.join(names)}" for scope, names in found.items() if names)
    return [f"The means count these cells, an ability in a column: {listed}.", ""] if listed else []


def _thin_line(agg: dict) -> list[str]:
    """The (job, ability, column) cells under 20 questions, which the means weigh at n / 20 of a cell (0: counted in
    none), or a sentence that the rule was not applied."""
    if agg["thin_ignored"]:
        return ["Every ability is under 20 questions on its job in this dataset, so the under-20 rule was not applied "
                "and this headline is not the release statistic.", ""]
    if agg["thin"]:
        named = ", ".join(f"{' '.join(names[:2])}" + (f" ({names[2]})" if len(names) > 2 else "") + f" {weight:.2f}"
                          for names, weight in thin_from(agg["thin"]).items())
        return [f"Under 20 questions in the dataset, so counted at part weight in the means (0.00 counts in none): "
                f"{named}.", ""]
    return []


def _splits(agg: dict, rows: list[dict]) -> list[str]:
    """The headline, the score above the floor, the unpenalised credit and the abstention bound of the layout's `rows`,
    the two columns, the secondary means and the own-path split, then each ability, family and length and the
    answerable questions by relevance."""
    by = agg["by"]
    top = [["headline", str(_counted(agg)), signed(agg["headline"]["mean"]), interval(agg["headline"]["ci"])],
           _floor_row(agg),
           _stat_row("unpenalised credit (the headline's statistic over the credit of each answer)",
                     {**restated(rows, agg, lambda r: r["s"]), "n": _counted(agg)}),
           _stat_row("abstention bound (the headline had every answerable answer scoring below 0 been `unknown`)",
                     {**restated(rows, agg, bound_score), "n": _counted(agg)})]
    top += [_stat_row(scope, agg["columns"][scope]) for scope in SCOPES if scope in agg["columns"]]
    top += [_stat_row("cells mean (the counted cells by their weights)", {**agg["abilities_mean"], "n": _counted(agg)}),
            _stat_row("pooled (the counted questions pooled)", {**agg["pooled"], "n": _counted(agg)})]
    for name, dim, value in (("own path (the answer rests on the robot's own log)", "own_path", "True"),
                             ("not own path", "own_path", "False")):
        if value in by[dim]:
            top.append(_stat_row(name, by[dim][value]))
    head = ["", "n", "mean", "95% CI"]
    lines = _table(head, top)
    lines += ["### Per ability", "", *_table(["ability", "column", "n", "counted", "weight", "mean", "counted mean",
                                             "95% CI"], _ability_rows(agg)), *_cells_line(agg), *_thin_line(agg)]
    lines += ["### Per kind", "", *_table(["kind", "n", "mean", "95% CI"], _kind_rows(agg))]
    if ages := _age_rows(agg):
        lines += ["### Per age of the evidence", "", *_table(["age band", "n", "mean", "95% CI"], ages)]
    for title, dim in (("Per family", "family"), ("Per length", "length")):
        keys = sorted(by[dim], key=_length_order) if dim == "length" else list(by[dim])
        lines += [f"### {title}", "", *_table([dim, "n", "mean", "95% CI"], [_stat_row(k, by[dim][k]) for k in keys])]
    return lines + ["### Per relevance", "", *_table(["slice", "n", "mean", "95% CI"], _relevance_rows(by))]


def _outcomes(agg: dict) -> list[str]:
    c, w, a = agg["counts"], agg["confidently_wrong_rate"], agg["abstention"]
    return [*_catastrophe_lines(agg), "### Outcomes", "",
            f"- {count(agg['n'], 'question')} over {count(agg['cases'], 'case')} of {count(agg['scripts'], 'script')}: "
            f"correct {c['correct']}, partial {c['partial']}, wrong {c['wrong']}, abstained {c['abstained']}",
            f"- confidently wrong rate (wrong over all questions): {rate(w['rate'])} {interval(w['ci'])}",
            f"- abstention precision (of {a['answered_unknown']} `unknown` answers, the share on unknown questions): "
            f"{rate(a['precision'])} {interval(a['ci_precision'])}",
            f"- abstention recall (of {a['unknown_gold']} unknown questions, the share answered `unknown`): "
            f"{rate(a['recall'])} {interval(a['ci_recall'])}", ""]


def _abstention(rows: list[dict]) -> list[str]:
    """The layout's abstention overall, by job and by family (`score_abstain`): recall (of the `unknown` golds, the
    share answered `unknown`), precision (of the `unknown` answers, the share on an `unknown` gold) and the `unknown`
    rate on answerable questions, each with its Wilson interval and counts."""
    table = [[label, s.recall.text(), s.precision.text(), s.over.text()] for label, s in groups(rows)]
    words = ("Recall is the share of the `unknown` golds answered `unknown`, precision the share of the `unknown` "
             "answers that were on an `unknown` gold, and the last column the share of the answerable questions "
             "answered `unknown`. Intervals are Wilson, 95%.")
    return ["### Abstention", "", words, "", *_table(["", "recall", "precision", "`unknown` on answerable"], table)]


def _shares(rows: list[dict], agg: dict) -> list[str]:
    """The credit bands of each ability's questions (`BANDS`, as shares), with the ability's mean reported score (all
    its questions, as `by.ability` pools them), its mean unpenalised credit and its mean under the abstention bound."""
    table = []
    for name in sorted({ability(r) for r in rows}):
        mine = [r for r in rows if ability(r) == name]
        flag = " (diagnostic)" if agg["by"]["ability"][name]["diagnostic"] else ""
        bands = [credit_band(r) for r in mine]
        table.append([name + flag, str(len(mine)), *(rate(bands.count(b) / len(mine)) for b in BANDS),
                      signed(sum(r["score"] for r in mine) / len(mine)), signed(sum(r["s"] for r in mine) / len(mine)),
                      signed(sum(bound_score(r) for r in mine) / len(mine))])
    head = ["ability", "n", *BANDS, "mean", "credit", "abstention bound"]
    return ["### Credit bands by ability", "", *_table(head, table)]


def _resources(done: list[dict]) -> list[str]:
    """Tokens, cost and wall time over the layout's finished cases; a total is `n/a` when any case's is unknown."""
    tokens = []
    for key, name in TOKEN_NAMES:
        counts = [c["tokens"][key] for c in done]
        tokens.append(f"{name} {'n/a' if None in counts else f'{sum(counts):,}'}")
    costs = [c["cost_usd"] for c in done]
    wall = sum(sum(c["wall_s"].values()) for c in done)
    cost = "n/a" if None in costs else f"${sum(costs):.4f} (${sum(costs) / len(done):.4f} a case)"
    return ["### Resources", "", f"- tokens: {', '.join(tokens)}", f"- cost: {cost}",
            f"- wall time: {wall:.1f} s in all, {wall / len(done):.1f} s a case", ""]


def _layout(doc: dict, name: str, agg: dict | None) -> list[str]:
    here = [c for c in doc["cases"] if c["layout"] == name]
    done = [c for c in here if c["status"] == "complete"]
    lines = [f"## Layout {name}: {len(done)} of {count(len(here), 'case')} finished", ""]
    if agg is None:
        return [*lines, "No case of this layout finished, so there is nothing to score.", ""]
    rows = [r for r in doc["rows"] if r["layout"] == name]
    if len(rows) != agg["n"]:
        raise ValueError(f"layout {name}: the document holds {len(rows)} rows but its aggregate counts {agg['n']}")
    return [*lines, *_splits(agg, rows), *_outcomes(agg), *_abstention(rows), *_shares(rows, agg), *_resources(done)]


def render(doc: dict) -> str:
    """The `results.md` text of a validated results document (what `eval_results.load` returns)."""
    done = [c for c in doc["cases"] if c["status"] == "complete"]
    lines = [f"*** {note} ***" for note in doc["notes"]]
    system = system_name(doc)
    cost = "n/a" if doc["cost_usd"] is None else f"${doc['cost_usd']:.4f}"
    lines += [f"# SpatiotemporalBench results: {system}", "", f"- system: {system}", f"- {lever_name(doc)}",
              f"- sandbox: {sandbox_name(doc)}", f"- verified: {'yes' if doc['verified'] else 'NO'}"]
    if doc["reasoning"]:
        lines.append(f"- reasoning effort: {doc['reasoning']}")
    lines.append(f"- cases: {len(done)} of {len(doc['cases'])} finished")
    lines += [f"- cost of the cases that ran: {cost}"
              + (f" (cap ${doc['max_usd']:.2f})" if doc["max_usd"] is not None else ""), ""]
    if unfinished := [c for c in doc["cases"] if c["status"] != "complete"]:
        lines += ["## Cases not finished", ""]
        lines += [f"- {c['layout']} {c['id']}: {c['status']}" + (f": {c['error']}" if c["error"] else "")
                  for c in unfinished]
        lines.append("")
    if not any(doc["layouts"].values()):
        return "\n".join([*lines, "No case finished, so there is nothing to score.", ""])
    for name, agg in doc["layouts"].items():
        lines += _layout(doc, name, agg)
    return "\n".join([*lines, NOT_COMPARABLE, "", HEADLINE_RULE, "", BOUND_RULE, "",
                      *(f"*** {note} ***" for note in doc["notes"]), ""])
