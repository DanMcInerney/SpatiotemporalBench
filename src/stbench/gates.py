"""The validity gates (V0 §11, plan §4.5, §4.6), per layout, job and family, on the penalised score.

Every read of a score here is of the row's `score_gate`, the penalised score `2s - 1` (`score_stats.GATE`), never of its
`score`, the reported score a catastrophe's weight lowers (0.10.0, review B4): no harm weight changes a verdict.

The thresholds are fixed here before any data. A family ships only if every gate that applies to it passes; a failing
family is fixed or dropped and the result is written down.

| gate          | rule                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------ |
| ceiling       | oracle >= 0.95                                                                             |
| knowable      | oracle - memory_table: upper 95% bound <= 0.02 (n/a where the gold and systems share a rule) |
| memory_needed | on `memory` rows: current_view and current_view_prior each <= all_unknown + 0.05           |
| shortcuts     | blind, path_log, other_case, questionnaire_blind, form each <= all_unknown + 0.10          |
| separation    | memory_table - best shortcut >= 0.40, lower 95% bound > 0.30                               |
| time_needed   | on a temporal family: max(latest_table, first_table) <= all_unknown + 0.10                 |

Every gate gives a verdict three times. `pass` is the full verdict, the interval where a gate has one; `point` is the
verdict on the point estimates alone. Only `knowable` and `separation` differ between them: a gate that needs an
interval fails `pass` without one (fewer than two scripts) and can still hold `point`. M1's 20 one-task scripts cannot
power an interval, so M1 reads `passed_point`; M2's 30 cases must pass `passed` (recommendations §2.3, item 4). The
third, `evidence`, is read by `gates_evidence` in `memory_needed`, `shortcuts` and `time_needed`: a system fails there
only when its gain over `all_unknown` is more than the margin, in the group, in a length tier or in a slice, with an
interval above 0: a tilt that chance explains passes, and a gain on one slice that the pooled mean hides fails.
`ceiling`, `knowable` and `separation` keep their point verdict as their `evidence`. A shortcut-type gate's `cells`
say where each system stands and its `pooled` gives each system's own score and interval; `passed_evidence` is every
applicable gate's `evidence` over every group.

A gate that does not apply, or has no rows to read, is `n/a` (`pass`, `point` and `evidence` are `None`): it never
counts as a pass or a fail. `knowable` is `n/a` on a family in `shared_rule` (the gold and the systems call one rule, a
`grid` or `adjacency` function, so there is no knowing to measure); `memory_needed` is `n/a` in a group with no `memory`
row, which is a family of `now` rows; `time_needed` is `n/a` on a family that is not in `temporal`
(`FamilySpec.temporal`: it asks about change over time). A system with no rows in a group never raises: a gate that
reads it checks the systems that have rows, so a failure is never hidden, and without a failure its verdict is `n/a`
with a `note` naming what is missing, since a pass proven on part of the systems is not a pass.

A family's reference is the shortcut that answers it by construction, so it is left out of that family's shortcut gate
and of its best shortcut: the rows say so (every row's `reference`), and no family is named here. An own-path family
(`reference` is `path_log`) is one that `path_log`, which holds the pose log and the public map, answers: on every
layout. A layout-constant family (`reference` is `other_case`) is one whose answer is the same in every case of its
layout and is asked only of a layout the agent has never seen (VISION rule 5), where any case's stream carries the
layout: that is the ability, so on a layout of `map_files.HELD_OUT` `other_case` is left out in the same way. On a
public layout such a family keeps `other_case` as a gated shortcut.

A declared reference that is no shortcut is the other kind: `kind_range`, the reference of `ego_now` (the thing's true
bearing at its kind's typical range), is not among `SHORTCUTS` and so is never read by the `shortcuts` gate, but in a
group whose every oracle row names it `separation` counts it among the systems the table is compared against, so that
distance is measured above it. A group that names it and holds no rows for it, or whose rows all abstained (a small
build whose prior is under the support floor everywhere), has `separation` `n/a` with a note, never a pass.

A temporal family asks what a thing did between two moments, which only a system that keeps time can answer. The two
notebooks that cannot are the oracle's table with one sighting a track and no event table: `latest_table` keeps the
latest, `first_table` the first, so a family that asks a moved thing at its earlier state only is answered by the second
and one that asks its later state only by the first. `time_needed` holds the larger of their two scores to the shortcut
margin, and on those families both are also among the shortcuts `separation` compares the table against (not of
`shortcuts`, which reads the other scripts' learned pools).

A structure family (0.8.0: `clear_path`, how far the body could move before it meets anything) asks about the fixed
structure around the robot, which a system holding the place's layout answers without the sensors. Its reference is
`map_reader` (`baselines_map`: the public map's raycast, then the pose-keyed pool of other scripts' golds), a reference
and never a shortcut: on a group whose oracle rows all name it, the shortcut gate and `separation` leave out
`map_reader` and `other_case` on every layout, held out or not (the layout is the ability), and the group reports
`map_reader`'s score as `kind_range`'s. Everything else stays gated: `blind`, `form`, `questionnaire_blind`, `path_log`
(which answers a structure family from the robot's own log alone, no map and no pool: `answer_keys.levels`) and, on
`memory` rows, `current_view` and `current_view_prior` without a pose key. Only a family whose spec declares the
reference is a structure family; a gate's failure never makes one.

The unknown share is the score of `all_unknown` (1 on unknown golds, 0 elsewhere). `form` is the best single-feature
rule over the other scripts' questions and gold (E2's form adversary): not a system, so `check` supplies its rows and
the suite runs only the `BASELINE_NAMES`.

A learned shortcut (`LEARNED`) keeps its best response by a rule that must hold on a script left out of its pool, which
abstains on a small pool near the boundary where a fair learner would answer. `check` therefore also runs each under the
pool's own winner (`in_pool_name`, `baselines.in_pool_twins`), and the `shortcuts` and `memory_needed` gates read the
larger score of the two, being the worst of all of them. A twin with no rows in a group is not read. `separation` and
the broken-instrument rule (`suite_report`) read the held-out rule alone.

A layout is gated on its own, and so is a job: a shortcut that works on one must fail its gate there, and a verdict
pooled over both could hide it. The groups are the `(layout, job, family)` of the oracle's rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from . import gates_evidence, map_files
from .score_stats import GATE, N_BOOT, gate_score, paired_difference, pooled

CEILING, KNOWABLE_CI, SHORTCUT_MARGIN, MEMORY_MARGIN, SEPARATION, SEPARATION_CI = 0.95, 0.02, 0.10, 0.05, 0.40, 0.30
ORACLE, TABLE, CURRENT, CURRENT_PRIOR = "oracle", "memory_table", "current_view", "current_view_prior"
PATH_LOG, BLIND, OTHER = "path_log", "blind", "other_case"
QBLIND, UNKNOWN, FORM, LATEST = "questionnaire_blind", "all_unknown", "form", "latest_table"
FIRST = "first_table"
KIND_RANGE = "kind_range"   # `ego_now`'s declared reference: no shortcut, a rival `separation` reads where declared
MAP_READER = "map_reader"   # a structure family's declared reference: no shortcut, and no rival of `separation` either
NOTEBOOKS = (LATEST, FIRST)   # the time-blind notebooks: read on temporal families only
BASELINE_NAMES = (ORACLE, TABLE, CURRENT, CURRENT_PRIOR, PATH_LOG, BLIND, OTHER, QBLIND, UNKNOWN, KIND_RANGE,
                  MAP_READER, *NOTEBOOKS)  # E1's, 0.6.0's reference and 0.8.0's
SHORTCUTS = (BLIND, PATH_LOG, OTHER, QBLIND, FORM)
MEMORY_NEEDED = (CURRENT, CURRENT_PRIOR)
LEARNED = (CURRENT_PRIOR, PATH_LOG, BLIND, QBLIND, FORM)   # the systems that learn from other scripts' records
IN_POOL = "_in_pool"   # the suffix of a learned shortcut's twin under the in-pool rule
GATES = ("ceiling", "knowable", "memory_needed", "shortcuts", "separation", "time_needed")
TOL = 1e-9   # a comparison within this of its threshold is on it

Rows = dict[str, list[dict]]


def in_pool_name(name: str) -> str:
    """The name of the twin of the learned shortcut `name` (`LEARNED`) that keeps its pool's own best response."""
    return name + IN_POOL


def _with_twins(names: Iterable[str], rows: Rows) -> list[str]:
    """`names` and, after them, the in-pool twin of each learned one that has rows."""
    names = list(names)
    return [*names, *(twin for n in names if n in LEARNED and rows.get(twin := in_pool_name(n)))]


def _verdict(holds: bool, absent: list[str]) -> bool | None:
    """A failure stands whatever is missing; a pass proven on part of the systems is not a pass (`n/a`)."""
    return None if holds and absent else holds


def _gate(ok: bool, point: bool, absent: list[str], value, threshold, *, evidence: bool | None = None,
          **extra) -> dict:
    """A gate's result: `pass` (the full verdict), `point` (the point estimates alone) and `evidence` (the evidence
    reading; the point verdict for a gate that has none), the number behind them and its threshold. `absent` names the
    systems the gate should have read and found no rows for."""
    if absent:
        extra["note"] = f"no rows for {', '.join(absent)}"
    shown = point if evidence is None else evidence
    return {"pass": _verdict(ok, absent), "point": _verdict(point, absent), "evidence": _verdict(shown, absent),
            "value": value, "threshold": threshold, **extra}


def _na(threshold, note: str) -> dict:
    return {"pass": None, "point": None, "evidence": None, "value": None, "threshold": threshold, "note": note}


def _evidenced(read: dict) -> dict:
    """The keys a shortcut-type gate takes from `gates_evidence.read`."""
    note = {"evidence_note": read["note"]} if read["note"] else {}
    return {"evidence": read["ok"], "cells": read["cells"], "pooled": read["pooled"], **note}


def _mean(rows: list[dict]) -> float:
    return sum(gate_score(r) for r in rows) / len(rows)


def _ceiling(oracle: float) -> dict:
    ok = oracle >= CEILING - TOL
    return _gate(ok, ok, [], oracle, CEILING)


def _knowable(rows: Rows, family: str, shared_rule: frozenset[str], n_boot: int, seed: int) -> dict:
    if family in shared_rule:
        return _na(KNOWABLE_CI, "the gold and the systems call one rule: there is nothing to know")
    if not rows[TABLE]:
        return _na(KNOWABLE_CI, f"no rows for {TABLE}")
    gap = paired_difference(rows[ORACLE], rows[TABLE], n_boot=n_boot, seed=seed, key=GATE)
    return _gate(gap["ci"] is not None and gap["ci"][1] <= KNOWABLE_CI + TOL, gap["mean"] <= KNOWABLE_CI + TOL, [],
                 gap["mean"], KNOWABLE_CI, ci=gap["ci"])


def _memory_needed(rows: Rows, n_boot: int, seed: int) -> dict:
    read = _with_twins(MEMORY_NEEDED, rows)
    memory = {n: [r for r in rows[n] if r["scope"] == "memory"] for n in (UNKNOWN, *read)}
    if not any(r["scope"] == "memory" for r in rows[ORACLE]):
        return _na(MEMORY_MARGIN, "no memory rows")
    if not memory[UNKNOWN]:
        return _na(MEMORY_MARGIN, f"no memory rows for {UNKNOWN}")
    present = [n for n in read if memory[n]]
    if not present:
        return _na(MEMORY_MARGIN, f"no memory rows for {', '.join(MEMORY_NEEDED)}")
    unknown = _mean(memory[UNKNOWN])
    over = {n: _mean(memory[n]) - unknown for n in present}
    worst = max(over, key=over.get)
    ok = over[worst] <= MEMORY_MARGIN + TOL
    seen = gates_evidence.read({n: rows[n] for n in present}, rows[UNKNOWN], bar=MEMORY_MARGIN + TOL, scope="memory",
                               n_boot=n_boot, seed=seed)
    return _gate(ok, ok, [n for n in MEMORY_NEEDED if n not in present], over[worst], MEMORY_MARGIN, worst=worst,
                 unknown_share=unknown, **_evidenced(seen))


def _shortcuts(rows: Rows, names: list[str], n_boot: int, seed: int) -> dict:
    if not rows[UNKNOWN]:
        return _na(SHORTCUT_MARGIN, f"no rows for {UNKNOWN}")
    present = [n for n in names if rows[n]]
    if not present:
        return _na(SHORTCUT_MARGIN, f"no rows for {', '.join(names)}")
    unknown = _mean(rows[UNKNOWN])
    means = {n: _mean(rows[n]) for n in present}
    worst = max(means, key=means.get)
    ok = means[worst] - unknown <= SHORTCUT_MARGIN + TOL
    seen = gates_evidence.read({n: rows[n] for n in present}, rows[UNKNOWN], bar=SHORTCUT_MARGIN + TOL, n_boot=n_boot,
                               seed=seed)
    return _gate(ok, ok, [n for n in names if n not in present], means[worst] - unknown, SHORTCUT_MARGIN, worst=worst,
                 unknown_share=unknown, scores=means, **_evidenced(seen))


def _time_needed(rows: Rows, family: str, temporal: frozenset[str], n_boot: int, seed: int) -> dict:
    """The larger score of the two notebooks (`NOTEBOOKS`) against `all_unknown` on a temporal family: the shortcut
    margin, read as `_shortcuts` reads it (so with its keys, `worst` naming the notebook that failed). Any other family:
    `n/a` and no note, which is the case of nearly every family."""
    if family not in temporal:
        return {"pass": None, "point": None, "evidence": None, "value": None, "threshold": SHORTCUT_MARGIN}
    return _shortcuts(rows, list(NOTEBOOKS), n_boot, seed)


def _separation(rows: Rows, names: list[str], n_boot: int, seed: int) -> dict:
    present = [n for n in names if rows[n]]
    if not rows[TABLE]:
        return _na(SEPARATION, f"no rows for {TABLE}")
    if not present:
        return _na(SEPARATION, f"no rows for {', '.join(names)}")
    best = max(present, key=lambda n: _mean(rows[n]))
    gap = paired_difference(rows[TABLE], rows[best], n_boot=n_boot, seed=seed, key=GATE)
    point = gap["mean"] >= SEPARATION - TOL
    ok = point and gap["ci"] is not None and gap["ci"][0] > SEPARATION_CI + TOL
    return _gate(ok, point, [n for n in names if n not in present], gap["mean"], SEPARATION, ci=gap["ci"], best=best,
                 lower_ci_threshold=SEPARATION_CI)


def _references(layout: str, oracle: list[dict]) -> frozenset[str]:
    """The systems that are the group's reference, read from the oracle's rows: `path_log` when every row names it,
    `other_case` when every row names it and the layout is held out, `kind_range` when every row names it, and
    `map_reader` (with `other_case`, on every layout) when every row names `map_reader`. `path_log` and `other_case` are
    shortcuts left out of the group's shortcut gate; `kind_range` was never one (it is not in `SHORTCUTS`) and is read
    by `separation` as a rival instead; `map_reader` is no shortcut and no rival: the family is measured beside it
    (module docstring)."""
    excusable = (PATH_LOG, OTHER, KIND_RANGE) if layout in map_files.HELD_OUT else (PATH_LOG, KIND_RANGE)
    found = frozenset(n for n in excusable if all(r["reference"] == n for r in oracle))
    return found | {MAP_READER, OTHER} if all(r["reference"] == MAP_READER for r in oracle) else found


def _scope(oracle: list[dict]) -> str:
    """`now` or `memory` when every oracle row of the group has that scope, else `mixed`."""
    found = {r["scope"] for r in oracle}
    return found.pop() if len(found) == 1 else "mixed"


def _group(key: tuple[str, str, str], split: dict[str, dict[tuple, list[dict]]], shared_rule: frozenset[str],
           temporal: frozenset[str], n_boot: int, seed: int) -> dict:
    layout, job, family = key
    rows = {name: split.get(name, {}).get(key, []) for name in (*BASELINE_NAMES, FORM)}
    rows |= {twin: split[twin].get(key, []) for n in LEARNED if (twin := in_pool_name(n)) in split}
    references = _references(layout, rows[ORACLE])
    names = [n for n in SHORTCUTS if n not in references]
    rivals = [*names, *NOTEBOOKS] if family in temporal else [*names]   # what `separation` compares the table against
    if KIND_RANGE in references:
        rivals.append(KIND_RANGE)   # the family's declared reference: it is measured above it, though it is no shortcut
    mute = KIND_RANGE in references and bool(rows[KIND_RANGE]) and all(
        r["outcome"] == "abstained" for r in rows[KIND_RANGE])   # a reference that answered nothing is no rival
    gates = {
        "ceiling": _ceiling(_mean(rows[ORACLE])),
        "knowable": _knowable(rows, family, shared_rule, n_boot, seed),
        "memory_needed": _memory_needed(rows, n_boot, seed),
        "shortcuts": _shortcuts(rows, _with_twins(names, rows), n_boot, seed),
        "separation": _separation({**rows, KIND_RANGE: []} if mute else rows, rivals, n_boot, seed),
        "time_needed": _time_needed(rows, family, temporal, n_boot, seed),
    }
    if mute:
        gates["separation"]["note"] = f"{KIND_RANGE} answered no question of the family (every row abstained)"
    scores = {n: pooled(rs, n_boot=n_boot, seed=seed, key=GATE) for n, rs in rows.items() if rs}
    read = [n for n in rows if (n not in NOTEBOOKS or family in temporal)   # the notebooks: a temporal family's only
            and (n not in (KIND_RANGE, MAP_READER) or n in references)]     # a declared reference: its family's only
    return {
        "layout": layout, "job": job, "family": family, "scope": _scope(rows[ORACLE]),
        "scripts": len({r["script"] for r in rows[ORACLE]}), "n": len(rows[ORACLE]),
        "missing": [n for n in read if not rows[n]],
        "scores": {n: {"mean": s["mean"], "ci": s["ci"]} for n, s in scores.items()}, "gates": gates,
        "pass": all(g["pass"] for g in gates.values() if g["pass"] is not None),
        "point": all(g["point"] for g in gates.values() if g["point"] is not None),
        "evidence": all(g["evidence"] for g in gates.values() if g["evidence"] is not None),
    }


def evaluate(results: Rows, *, shared_rule: frozenset[str] = frozenset(), temporal: frozenset[str] = frozenset(),
             n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Gate every (layout, job, family) of the oracle's rows. `results` maps each system name (`BASELINE_NAMES`, `FORM`
    and the in-pool twin of each learned one, `in_pool_name`) to its result rows (`score.row` output) over the same
    questions; a name it lacks, or a system with no rows in a group, is read as no rows (see the module docstring).
    `shared_rule` names the families whose gold and systems call one rule, for which `knowable` is `n/a`, and `temporal`
    the families that ask about change over time (`FamilySpec.temporal`), which `time_needed` reads and whose
    `separation` counts the two notebooks among the shortcuts. Returns
    `{rows, passed, passed_point, passed_evidence}`; each of `rows` is `{layout, job, family, scope, scripts, n,
    missing, scores, gates, pass, point, evidence}`, `scope` that of the oracle's rows (`now`, `memory` or `mixed`),
    `scores[system]` is `{mean, ci}`, and `missing` lists the systems with no rows in the group. `passed` is every
    applicable gate's `pass` over every group, `passed_point` its `point` and `passed_evidence` its `evidence`; all
    three are False with no group."""
    if not results.get(ORACLE):
        raise ValueError(f"the gates are grouped by the {ORACLE!r} rows and there are none")
    split: dict[str, dict[tuple, list[dict]]] = {}
    for name, rows in results.items():
        for r in rows:
            split.setdefault(name, {}).setdefault((r["layout"], r["job"], r["family"]), []).append(r)
    groups = [_group(key, split, shared_rule, temporal, n_boot, seed) for key in sorted(split[ORACLE])]
    return {"rows": groups, "passed": all(g["pass"] for g in groups) and bool(groups),
            "passed_point": all(g["point"] for g in groups) and bool(groups),
            "passed_evidence": all(g["evidence"] for g in groups) and bool(groups)}


def _num(x) -> str:
    return "n/a" if x is None else f"{x:+.2f}" if abs(x) < 10 else f"{x:.1f}"


def _ci(c) -> str:
    return "" if c is None else f" [{c[0]:+.2f},{c[1]:+.2f}]"


def _cell(g: dict) -> str:
    if g["pass"] is None:
        return "n/a"
    status = "ok" if g["pass"] else "point" if g["point"] else "FAIL"
    return f"{status:<5} {_num(g['value'])} vs {g['threshold']}{_ci(g.get('ci'))}"


def _line(cells: list[str], widths: tuple[int, ...]) -> str:
    return "  ".join(f"{c:<{w}}" for c, w in zip(cells, widths, strict=True)).rstrip()


def format_table(report: dict, balance_lines: Sequence[str] = (), also_holds: bool = True) -> str:
    """A fixed-width table: one line per (layout, job, family) with its scope (`now`, `memory` or `mixed`: the oracle's
    rows), scripts and questions, each gate's verdict (`ok`, `point` for a pass on the point estimates alone, `FAIL` or
    `n/a`) with its value and threshold, the group's evidence verdict (`evid`, `ok` or `FAIL`) and its verdict; then a
    line for each gate that carries a note (why it is `n/a`, or what it did not read), one `EVID` line for each cell in
    which a system counts (`counting`), then `balance_lines` (`check` gives the balance rule's, which are not a gate),
    the evidence verdict and the report's verdict. The evidence verdict line passes only when `also_holds` too (`check`
    says whether the balance rule and the instruments held), so it is the check's, not the gates' alone."""
    names = ("layout", "job", "family", "scope", "scripts", "n", *GATES, "evid", "verdict")
    table = [[r["layout"], r["job"], r["family"], r["scope"], str(r["scripts"]), str(r["n"]),
              *(_cell(r["gates"][g]) for g in GATES), "ok" if r["evidence"] else "FAIL",
              "pass" if r["pass"] else "point" if r["point"] else "FAIL"]
             for r in report["rows"]]
    widths = tuple(max(len(row[i]) for row in [list(names), *table]) for i in range(len(names)))
    lines = [_line(list(names), widths), *(_line(row, widths) for row in table)]
    lines += [f"{r['layout']}/{r['job']}/{r['family']}: {g} {gate['note']}" for r in report["rows"]
              for g, gate in r["gates"].items() if "note" in gate]
    lines += [f"{r['layout']}/{r['job']}/{r['family']}: {g} evidence: {gate['evidence_note']}" for r in report["rows"]
              for g, gate in r["gates"].items() if "evidence_note" in gate]
    lines += [f"EVID {layout} {job} {family} {gate} [{gates_evidence.label(cell['cell'])}]: "
              f"{gates_evidence.sentence(cell)}" for layout, job, family, gate, cell in counting(report)]
    lines += balance_lines
    lines.append("EVIDENCE VERDICT PASSES" if report["passed_evidence"] and also_holds else "EVIDENCE VERDICT FAILS")
    verdict = ("ALL GATES PASS" if report["passed"] else "POINT GATES PASS, INTERVAL VERDICTS FAIL"
               if report["passed_point"] else "GATES FAIL")
    return "\n".join([*lines, verdict])


def failing(report: dict) -> Iterable[tuple[str, str, str, str]]:
    """(layout, job, family, gate) of every gate whose full verdict is a fail (an `n/a` gate is not one)."""
    for r in report["rows"]:
        for name, g in r["gates"].items():
            if g["pass"] is False:
                yield r["layout"], r["job"], r["family"], name


def counting(report: dict) -> Iterable[tuple[str, str, str, str, dict]]:
    """(layout, job, family, gate, cell) of every cell in which a system counts under the evidence reading: its gain
    over `all_unknown` is more than the gate's margin with an interval above 0. `cell` is a `gates_evidence.read` cell,
    the system that counts there named in `worst`."""
    for r in report["rows"]:
        for name, g in r["gates"].items():
            for cell in g.get("cells", ()):
                if cell["counts"]:
                    yield r["layout"], r["job"], r["family"], name, cell
