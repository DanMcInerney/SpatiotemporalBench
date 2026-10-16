"""The numbers a `stbench check` report shows beside the gates: whether each gate could fail (plan §4.8, §4.14).

A gate that cannot fail proves nothing, so for each (layout, job) group the report says, per family: how many questions
each case asks, how many golds are unknown, how many empty where the answer is a set, how often each prior system's
first key finds a record in its pool (`hit_rates`) and how often any of its levels above the bare template does
(`reach_rates`: a prior none of whose levels finds a record falls to the template whatever the question and makes that
system's shortcut gate pass for nothing; a first key that misses where a looser level hits is a prior at work, as every
pair family's exact keys are), and how well `current_view` does on the `now` rows (it is their reference, so under
`NOW_MEAN` the family's `now` tag is wrong); and for the group, the share of questions that are `now` (D7, declared
before data), each length's share of answerable end questions that are of a temporal family (`temporal_shares`: change
over time is claimed only where it is at least 0.2 and one a case). Truth side: it reads gold.

`broken_instruments` reads what a gate cannot say of itself (VISION rule 5: a gate that a broken shortcut passes proves
nothing): a prior whose reach rate is 0 (no level above the template finds a record), and a shortcut that learns from
other scripts yet scores well under abstaining, which is a prior that memorises where it should generalise. The second
needs evidence, as the evidence verdict does (`gates_evidence`): the mean is more than `FLOOR_MARGIN` under the floor
and the interval of the pooled score, from the shortcuts gate's `pooled`, lies under it too; with fewer than two
scripts there is no interval and nothing is proved. The rule is fixed before the dev render. The reach rule spares one
case: a family whose reference is `other_case` (a fact of the layout) on a held-out layout, where the priors learn from
public layouts only and no public layout asks the family, so their pools are empty by rule (`suite_priors`) and the
emptiness is the design; the group's numbers say so (`ABSTAINS`).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping

from . import answer_keys, baselines, gates, map_files, suite, suite_balance, suite_priors
from .case import LENGTHS, CasePaths
from .score_stats import gate_score
from .stream import CaseRun

ABSTAINS = "no public layout asks this family: its priors abstain by design"   # why a reach rate of 0 is no fault
PRIOR_SYSTEMS = (gates.BLIND, gates.PATH_LOG, gates.CURRENT_PRIOR)   # the systems `answer_keys.levels` keys
NOW_MEAN = 0.95                                                       # `current_view`'s least mean on `now` rows
FLOOR_MARGIN = gates.SHORTCUT_MARGIN                                  # a learned shortcut this far under it is broken
SET_TYPES = ("set", "positions", "cells")                             # the answer types a gold may leave empty


def _rates(cases: list[CasePaths], done: Mapping[tuple[str, str], CaseRun]
           ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """`(hit_rates, reach_rates)`, each `{family: {system: share}}`, over the questions of every case against the
    records of the other scripts' (`suite_priors.prior_records`). A question hits when its first key is some record's;
    it reaches when any of its levels above the bare template `(template,)` is a level of some record."""
    hit: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    reach: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        own = suite_priors.case_records(case, suite_priors.nm_answers(case, done))
        pool = suite_priors.prior_records(case, cases, done, nm=True)
        for system in PRIOR_SYSTEMS:
            keyed = [answer_keys.levels(system, p) for p in pool]
            first = {levels[0] for levels in keyed}
            above = {level for levels, p in zip(keyed, pool, strict=True)
                     for level in levels if level != (p["template"],)}
            for rec in own:
                levels, family = answer_keys.levels(system, rec), rec["gold"]["family"]
                hit[family][system].append(levels[0] in first)
                reach[family][system].append(any(level in above for level in levels if level != (rec["template"],)))
    return tuple({family: {system: statistics.fmean(flags) for system, flags in systems.items()}
                  for family, systems in rates.items()} for rates in (hit, reach))


def hit_rates(cases: list[CasePaths], done: Mapping[tuple[str, str], CaseRun]) -> dict[str, dict[str, float]]:
    """`{family: {system: share}}`: of the questions of the family, the share whose first key (`answer_keys.levels`)
    some record of its prior (other scripts' questions of the group, `suite_priors.prior_records`) shares."""
    return _rates(cases, done)[0]


def reach_rates(cases: list[CasePaths], done: Mapping[tuple[str, str], CaseRun]) -> dict[str, dict[str, float]]:
    """`{family: {system: share}}`: of the questions of the family, the share for which some level of
    `answer_keys.levels(system, rec)` above the bare template `(template,)` is a level of some record of its prior. A
    rate of 0 is a prior that never finds a record (`broken_instruments`)."""
    return _rates(cases, done)[1]


def _family(golds: list[dict], cases: int, hits: dict[str, float], reach: dict[str, float], now: list[float]) -> dict:
    sets = [g for g in golds if g["gold"]["type"] in SET_TYPES]
    references = {g["reference"] for g in golds}
    return {"questions": len(golds), "per_case": len(golds) / cases,
            "unknown_share": sum(g["slice"] == "unknown" for g in golds) / len(golds),
            "empty_set_share": sum(not g["gold"]["items"] for g in sets) / len(sets) if sets else None,
            "reference": references.pop() if len(references) == 1 else None,
            "class_shares": suite_balance.class_shares(golds[0]["family"], golds),
            "hit_rates": hits, "reach_rates": reach, "current_view_now_mean": statistics.fmean(now) if now else None}


def abstains_by_design(layout: str, family: Mapping) -> bool:
    """Whether the family's priors have no pool by rule on this layout: its reference is `other_case` (a fact of the
    layout, learned from public layouts only: `suite_priors`) and the layout is held out, so none asks it."""
    return layout in map_files.HELD_OUT and family.get("reference") == gates.OTHER


def temporal_shares(cases: list[CasePaths], temporal: frozenset[str]) -> dict[str, dict]:
    """For each length of `cases` (in `case.LENGTHS` order): `{cases, answerable, temporal, share, per_case}`, how many
    cases, how many of their end questions have an answerable gold (any slice but `unknown`), how many of those are of
    a family in `temporal`, that share (None without an answerable end question) and the temporal ones a case. A
    change over time is claimed only where the share is at least 0.2 and there is at least one a case (FD-0.5.0 §5)."""
    counts: dict[str, dict] = {}
    for case in cases:
        length = suite.case_info(case)["length"]
        c = counts.setdefault(length, {"cases": 0, "answerable": 0, "temporal": 0})
        c["cases"] += 1
        when = {q["qid"]: q["when"] for q in suite.read_jsonl(case.questions)}
        for g in suite.read_jsonl(case.gold):
            if when[g["qid"]] == "end" and g["slice"] != "unknown":
                c["answerable"] += 1
                c["temporal"] += g["family"] in temporal
    return {length: {**c, "share": c["temporal"] / c["answerable"] if c["answerable"] else None,
                     "per_case": c["temporal"] / c["cases"]}
            for length in LENGTHS if (c := counts.get(length))}


def group_numbers(cases: list[CasePaths], done: Mapping[tuple[str, str], CaseRun], rows: Mapping[str, list[dict]],
                  temporal: frozenset[str] = frozenset()) -> dict:
    """The report's numbers for one group of cases (one layout, one job): `cases`, `scripts` (the sorted
    ids of the cases' scripts), `now_share` (the share of golds whose scope is `now`), `temporal` (`temporal_shares` of
    the families in `temporal`, by length) and `families`, per family `{questions, per_case, unknown_share,
    empty_set_share (None unless its golds are sets), reference (the golds' common one, else None), class_shares
    (`suite_balance.class_shares`: the asked classes against the draw), hit_rates, reach_rates,
    current_view_now_mean (None without `now` rows)}`, and a `note` (`ABSTAINS`) where the family's priors have no pool
    by rule (`abstains_by_design`). `done` holds the runs of `NM_SYSTEM` (and the rest) over `cases`; `rows` the group's
    scored rows by system, of which only `current_view`'s are read."""
    golds: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        for g in suite.read_jsonl(case.gold):
            golds[g["family"]].append(g)
    hits, reach = _rates(cases, done)
    now = defaultdict(list)
    for r in rows[gates.CURRENT]:
        if r["scope"] == "now":
            now[r["family"]].append(gate_score(r))
    total = sum(map(len, golds.values()))
    families = {family: _family(gs, len(cases), hits.get(family, {}), reach.get(family, {}), now[family])
                for family, gs in sorted(golds.items())}
    layout = suite.case_info(cases[0])["layout"]
    for numbers in families.values():
        if abstains_by_design(layout, numbers):
            numbers["note"] = ABSTAINS
    return {"cases": len(cases), "scripts": sorted({c.script.id for c in cases}),
            "now_share": sum(g["scope"] == "now" for gs in golds.values() for g in gs) / total,
            "temporal": temporal_shares(cases, temporal), "families": families}


def learned_shortcuts() -> frozenset[str]:
    """The shortcuts of the shortcut gate that answer from other scripts' records alone: the baselines that take a
    prior (`Baseline.needs_prior`) and the form adversary, under the held-out rule. Their in-pool twins are not among
    them: the pool's own winner overfits by design and is read by the gates as the stricter score, never as a fault.
    `other_case` is not one: it answers from another case's stream, and an agent that trusts a wrong stream rightly
    scores under abstaining. Nor is `current_view_prior`, which no gate lists among its shortcuts (`memory_needed`
    reads it): it is not a reading VISION rules 4-5 require (the current view alone is), so its deficit does not make
    the required gate vacuous."""
    return frozenset({name for name, b in baselines.BASELINES.items() if b.needs_prior and name in gates.SHORTCUTS}
                     | {gates.FORM})


def broken_instruments(evaluated: dict, numbers: Mapping[tuple[str, str], dict]) -> list[str]:
    """A line for each instrument of a gate that cannot do its job, so that the gate it feeds proves nothing: a prior
    system none of whose levels above the bare template finds a record of any question of a family (its reach rate in
    `numbers` is 0; a first key that misses where a looser level hits is a prior at work, not a broken one), and a
    learned shortcut (`learned_shortcuts`) whose mean on a family, under the held-out rule, is more than `FLOOR_MARGIN`
    under `all_unknown`'s and whose interval (the shortcut gate's `pooled`, over scripts) is under it as well: with no
    interval, fewer than two scripts, it proves nothing. A shortcut that the shortcut gate leaves out of a family (the
    reference of an own-path family, or one with no rows) is not judged there; the in-pool twins and
    `current_view_prior` are not judged on their score (`learned_shortcuts`), and the reach rule still reads
    `current_view_prior`'s keys. `evaluated` is `gates.evaluate`'s report; `numbers` maps (layout, job) to
    `group_numbers`."""
    learned, lines = learned_shortcuts(), []
    for r in evaluated["rows"]:
        where, gate = f"{r['layout']} {r['job']} {r['family']}", r["gates"]["shortcuts"]
        scored, floor = gate.get("scores", {}), gate.get("unknown_share")
        for name, own in gate.get("pooled", {}).items():
            mean, ci = own["mean"], own["ci"]
            if name in learned and mean < floor - FLOOR_MARGIN - gates.TOL and ci is not None and ci[1] < floor:
                lines.append(f"BROKEN {where}: {name} scores {mean:.3f} (95% interval {ci[0]:.3f} to {ci[1]:.3f}), "
                             f"{floor - mean:.3f} under all_unknown ({floor:.3f}); a shortcut that learns from other "
                             "scripts and does worse than abstaining overfits, and the gate it passes proves nothing.")
        family = numbers[r["layout"], r["job"]]["families"][r["family"]]
        if abstains_by_design(r["layout"], family):
            continue   # the reach rule judges priors that could find a record; here none can (`ABSTAINS`)
        for name, rate in family["reach_rates"].items():
            if rate == 0 and answer_keys.pooled(name, r["family"]) and (name in scored or name not in gates.SHORTCUTS):
                lines.append(f"BROKEN {where}: no level of {name} above the template finds a record of any question "
                             "(reach rate 0); its gate is vacuous.")
    return lines
