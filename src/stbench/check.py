"""`stbench check`: the validity gates over a dataset's compiled cases (VISION rule 5, plan §4.8, §4.14).

    stbench check --data DATASET [--out REPORT.json] [--workers N] [--verdict point|interval|evidence] [--verbose]
                  [--select KEY=VALUE ...] [--cases ID,...] [--answers DIR] [--reuse DIR]

Every scripted baseline (`baselines.BASELINES`) runs over every compiled case on the gate channel, with the options
`suite_priors.options_for` gives it from the cases of the case's own (layout, job) group (all the groups go through one
pool, so none waits for another's last case); when a system keys on another's answers (`prior_nm`) the runs go in two
waves, `NM_SYSTEM` first. So does the twin of each learned shortcut under the in-pool rule
(`baselines.in_pool_twins`), which the `shortcuts` and `memory_needed` gates read beside it, the larger score deciding
(`gates`). The form adversary (`form_check`, not a system: it needs every script's gold) adds its rows as the `form`
shortcut, and its twin. `gates.evaluate` then gates each (layout, job, family) and the command says plainly which gate
failed and by how much. A family ships only when its gates pass, with thresholds fixed before data: a
family that fails is fixed or dropped, never forced. The registry's temporal families (`FamilySpec.temporal`) go to the
gates, which hold `latest_table` and `first_table` (baselines like the rest) to `time_needed` on them, and the
printout gives each job's temporal share of the answerable end questions by length (`suite_report.temporal_shares`) and,
beside each family's `oracle` and `memory_table` means, the score of the reference it declares (`kind_range`'s beside
`ego_now`, `map_reader`'s beside `clear_path`).

`--verdict` is the reading the exit code follows. `interval` (the default) wants every gate to
pass on its 95% interval where it has one; `point` wants the point estimates alone, which is what M1's twenty one-task
scripts can power (recommendations §2.3 item 4); `evidence` reads the shortcut-type gates (`memory_needed`,
`shortcuts`, `time_needed`) as `gates_evidence` does, a system failing only where its gain over `all_unknown` is over
the margin with an interval above 0 (pooled, in a length tier or in a slice), and the other gates on their point
estimates, and it adds the balance rule (`suite_balance`, docs/balance.md): a family fails on a job when its held-out
best constant beats `all_unknown` with an interval above 0 (a `BALANCE FAIL` line; a group too thin to read is `thin`
and cannot fail; every family gets a `BALANCE` line in the printout under any verdict, before the table's `EVIDENCE
VERDICT` line, which counts the balance rule and the instruments, and the report's `balance` holds the entries); its
failure lines name the cell. The report (`DATASET/check_report.json`, or `--out`) holds the three
verdicts and, per group, `suite_report`'s numbers: questions per case, the `now` share, unknown
shares, the asked class shares against the draw, each prior system's first-level hit rate and reach rate,
`current_view` on `now` rows and every system's mean.
Only the cases with questions and gold are checked; a compiled case that asks nothing (a window in which no question
could be drawn) is skipped, and the report and the printout count them per group (`asked_nothing`: the yield the gates'
power rests on). A group that cannot feed `other_case` (no other script has a case of its length) refuses the check
(exit 2) before anything runs, and so does any scripted run that failed a question or ended early (these systems are
the instrument: their failure is a bug, not a score). The verdict fails, with a `BROKEN` line, when an instrument
cannot do its job (`suite_report.broken_instruments`: a prior none of whose levels above the bare template finds a
record of any question, a learned shortcut far under `all_unknown`, its interval under it too): a gate passed by a
broken shortcut proves nothing. A released dataset (it has a
`release.json`) keeps its own report unless `--out` is given. Exit code: 0 when the verdict passes, 1 when it fails,
2 when refused.

Scoped (0.11). `--select KEY=VALUE` (layout, robot, job or length, as `stbench eval --select`: `case_select`) and
`--cases ID,...` check only those cases, so a change to one job is re-checked in minutes. The groups and cases left are
run, gated and printed as the whole check would run them: a system's options look only at the cases of its own group, so
a selection by layout, robot or job (whole groups) gives each group the gates, scores and verdict of the full check. A
selection by length or case id narrows a group itself: its priors and `other_case` are built from the cases left, so
that reading is its own. The printout begins and ends with a `SCOPED CHECK` line saying what was checked, the report
holds a `scope`, and the dataset's `check_report.json` is never written by a scoped check (the export reads it for the
whole dataset): only to `--out`.
`--answers DIR` writes each scripted system's answers (`check_answers`); `--reuse DIR` reloads those of the systems
whose answers do not pass through the scorer (`Baseline.reads_scorer` false) instead of running them, and runs the
rest: a check after a scorer change and no other. The printout and the report name the systems reloaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from . import (
    answer_keys,
    baselines,
    case_select,
    check_answers,
    gates,
    gates_evidence,
    suite,
    suite_balance,
    suite_priors,
    suite_report,
)
from .case import CasePaths, depth_move_problem, read_index
from .form_check import Asked, features_of, form_adversary
from .questions_text import FAMILIES
from .stream import CaseRun
from .stream_protocol import GATE_CHANNEL

VERDICTS = ("interval", "point", "evidence")
MAX_REFUSED = 5   # failed runs a refusal names; the rest are counted
FAILS = {   # a failed gate, in words: the gate's own {value}, {threshold}, {worst} and {best}, and {ci} and {lower}
    "ceiling": "the oracle scores {value:.3f}; the gate wants at least {threshold}",
    "knowable": "oracle minus memory_table is {value:+.3f}{ci}; the gate allows at most {threshold}",
    "memory_needed": "{worst} beats all_unknown by {value:+.3f} on memory rows; the gate allows at most {threshold}",
    "shortcuts": "{worst} beats all_unknown by {value:+.3f}; the gate allows at most {threshold}",
    "separation": "memory_table beats {best} by {value:+.3f}{ci}; the gate wants at least {threshold}{lower}",
    "time_needed": "{worst}, a notebook that keeps one place for each thing, beats all_unknown by {value:+.3f}; the "
                   "gate allows at most {threshold}",
}
EVIDENCE_WORDS = {   # what a failing cell of the evidence reading adds to `gates_evidence.sentence`, by gate
    "memory_needed": " on memory rows",
    "shortcuts": "",
    "time_needed": ", a notebook that keeps one place for each thing",
}


def _asks(case: CasePaths) -> bool:
    """Whether the case's `questions.jsonl` holds a question row."""
    return any(line.strip() for line in case.questions.read_text(encoding="utf-8").splitlines())


def compiled_cases(dataset: Path) -> tuple[list[CasePaths], list[CasePaths]]:
    """The dataset's compiled cases (questions and gold on disk), in id order, as (those that ask at least one
    question, those that ask nothing: the compiler found no question to draw in their window)."""
    found = (CasePaths.from_id(dataset, cid) for cid, e in read_index(dataset).items() if e.get("kind") == "case")
    asking: list[CasePaths] = []
    silent: list[CasePaths] = []
    for case in found:
        if case.questions.is_file() and case.gold.is_file():
            (asking if _asks(case) else silent).append(case)
    return asking, silent


def groups(cases: list[CasePaths]) -> dict[tuple[str, str], list[CasePaths]]:
    """The cases by (layout, job), each group in id order: a shortcut that works on one layout or job must fail its
    gate there, so none is gated pooled."""
    found: dict[tuple[str, str], list[CasePaths]] = defaultdict(list)
    for case in cases:
        info = suite.case_info(case)
        found[info["layout"], info["job"]].append(case)
    return dict(sorted(found.items()))


def yields(asking: list[CasePaths], silent: list[CasePaths]) -> list[dict]:
    """For each (layout, job) with a compiled case: `{layout, job, compiled, asked_nothing}`, how many compiled cases it
    has and how many of them ask nothing and so are not checked. A group of nothing but such cases is listed too: it has
    no gate row, and the yield is what its absence says."""
    asked, nothing = groups(asking), groups(silent)
    return [{"layout": layout, "job": job,
             "compiled": len(asked.get((layout, job), [])) + len(nothing.get((layout, job), [])),
             "asked_nothing": len(nothing.get((layout, job), []))} for layout, job in sorted({*asked, *nothing})]


def asked_questions(cases: list[CasePaths]) -> list[Asked]:
    """Every question of the cases as delivered, with its gold and the features the form adversary may use."""
    out = []
    for case in cases:
        info = suite.case_info(case)
        golds = {g["qid"]: g for g in suite.read_jsonl(case.gold)}
        for place, q in enumerate(suite.read_jsonl(case.questions)):
            step = suite_priors.ask_step(case, q)
            features = features_of(q, k_in_task=step.k_in_task, place=place, phase=step.row["status"]["phase"])
            out.append(Asked(case.id, case.script.id, q, golds[q["qid"]], info, features, step.clock))
    return out


def _waves(names: list[str]) -> list[list[str]] | None:
    """`[[NM_SYSTEM], the rest]` when some system keys on `NM_SYSTEM`'s answers, else one wave."""
    if not any(b.prior_nm for b in baselines.BASELINES.values()):
        return None
    return [[baselines.NM_SYSTEM], [n for n in names if n != baselines.NM_SYSTEM]]


def _evidence_lines(r: dict, name: str, g: dict) -> list[str]:
    """A line for each cell in which a system counts in a shortcut-type gate: the cell, the system, its gain over
    `all_unknown` and the interval of that gain."""
    return [f"FAIL {r['layout']} {r['job']} {r['family']} {name} [{gates_evidence.label(cell['cell'])}]: "
            f"{gates_evidence.sentence(cell)}{EVIDENCE_WORDS[name]}; the gate counts a gain over {g['threshold']} "
            "whose interval is above 0." for cell in g.get("cells", ()) if cell["counts"]]


def outcome(evaluated: dict, verdict: str, broken: Sequence[str] = (), balance: Sequence[dict] = ()
            ) -> tuple[bool, list[str]]:
    """Whether the gates pass by `verdict`, and a line for each gate that fails by it: the family, the gate and by how
    much, naming the system that did it. Under `interval` a gate that holds on the point estimate and fails on its
    interval says so. Under `evidence` a shortcut-type gate gets a line for each cell in which a system counts, naming
    the cell (all rows, a length, a slice), the system and its gain with the interval. `balance`
    (`suite_balance.read`'s entries) is read under `evidence` alone: an entry that fails (a constant answer shown to
    beat `all_unknown`) fails the verdict and gets a `BALANCE FAIL` line after the gates' lines. `broken`
    (`suite_report.broken_instruments`) fails the verdict whatever the gates say, its lines after theirs: a gate that a
    broken instrument passes proves nothing."""
    key = verdict if verdict in ("point", "evidence") else "pass"
    lines = []
    for r in evaluated["rows"]:
        for name, g in r["gates"].items():
            if g[key] is not False:
                continue
            if key == "evidence" and (cells := _evidence_lines(r, name, g)):
                lines += cells
                continue
            ci = "" if g.get("ci") is None else f" (95% interval {g['ci'][0]:+.3f} to {g['ci'][1]:+.3f})"
            bound = g.get("lower_ci_threshold")
            lower = "" if bound is None else f" and the interval's lower bound above {bound}"
            why = FAILS[name].format(**{**g, "ci": ci, "lower": lower})
            held = " The point estimate holds; the interval does not." if g["point"] else ""
            lines.append(f"FAIL {r['layout']} {r['job']} {r['family']} {name}: {why}.{held}")
    passed = evaluated[{"point": "passed_point", "evidence": "passed_evidence"}.get(key, "passed")]
    unbalanced = [suite_balance.failure(b) for b in balance if b["fails"]] if key == "evidence" else []
    return passed and not broken and not unbalanced, [*lines, *unbalanced, *broken]


def _failed_runs(cases: list[CasePaths], runs: dict[str, list[CaseRun]]) -> list[str]:
    """What went wrong in each scripted run that failed a question or ended before its case did, by system then case:
    `<system> on case <id>: <status>; <n> x <reason>, ...`."""
    out = []
    for name, rs in runs.items():
        for case, run in zip(cases, rs, strict=True):
            why = ([] if run.status == "complete" else [run.status]) + [
                f"{n} x {reason}" for reason, n in sorted(run.failures.items())]
            if why:
                out.append(f"{name} on case {case.id}: {'; '.join(why)}")
    return out


def _group_ids(cases: list[CasePaths]) -> dict[str, list[str]]:
    """Each case's id to the ids of the cases of its (layout, job) group among `cases`."""
    return {case.id: [c.id for c in group] for group in groups(cases).values() for case in group}


def _reloaded(names: list[str], cases: list[CasePaths], reuse: Path | None
              ) -> dict[tuple[str, str], CaseRun]:
    """The stored runs (`check_answers`) of every system among `names` that does not read the scorer
    (`Baseline.reads_scorer`) and has a whole, current file for every one of `cases`, by (system, case id); a system
    with any file missing or stale is left out and so run. A system that looks at other cases (`needs_prior`,
    `other_stream`) needs the stored group to be the group now (`check_answers`)."""
    if reuse is None:
        return {}
    out: dict[tuple[str, str], CaseRun] = {}
    held = _group_ids(cases)
    for name in names:
        b = baselines.lookup(name)
        found = None if b.reads_scorer else check_answers.load_system(
            reuse, name, cases, held if b.needs_prior or b.other_stream else None)
        out |= {(name, cid): run for cid, run in (found or {}).items()}
    return out


def _check_runs(cases: list[CasePaths], args: argparse.Namespace, log: Callable[[str], None]
                ) -> tuple[dict, dict, list[str]]:
    """Every system over the cases of every group at once (a system's options look only at the cases of its own layout
    and job, so a group's runs are what they would be alone, and the pool is not drained between groups):
    (`{system: scored rows}`, the runs by (system, case id), the systems `--reuse` reloaded instead of running). A
    reloaded system (`_reloaded`) stands in for its run where a later wave's options look for the earlier waves'; every
    run is written under `--answers`. ValueError, naming the first `MAX_REFUSED` runs of the first group that has any,
    when a run failed a question or ended early: a scripted system that fails scores at the floor, and a shortcut gate
    that wants it low would pass for nothing."""
    names = [*baselines.BASELINES, *baselines.in_pool_twins()]
    waves = _waves(names) or [names]
    order = [name for wave in waves for name in wave]   # the order a fresh check's runs come in, reused or not
    kept = _reloaded(order, cases, args.reuse)
    reused = [name for name in order if (name, cases[0].id) in kept]

    def options(name: str, case: CasePaths, every: list[CasePaths], earlier) -> suite.RunOptions:
        return suite_priors.options_for(name, case, every, {**kept, **earlier})

    todo = [[name for name in wave if name not in reused] for wave in waves]
    runs = suite.run_suite(cases, [n for wave in todo for n in wave], GATE_CHANNEL, options=options,
                           waves=[wave for wave in todo if wave], workers=args.workers, log=log)
    runs = {name: runs[name] if name in runs else [kept[name, case.id] for case in cases] for name in order}
    at = {case.id: i for i, case in enumerate(cases)}
    for group in groups(cases).values():
        if failed := _failed_runs(group, {name: [rs[at[c.id]] for c in group] for name, rs in runs.items()}):
            more = f" (and {len(failed) - MAX_REFUSED} more)" if len(failed) > MAX_REFUSED else ""
            raise ValueError("a scripted system failed, which is a bug in the instrument and not a score: "
                             + "; ".join(failed[:MAX_REFUSED]) + more)
    held = _group_ids(cases)
    for name, rs in runs.items():
        if args.answers is not None and not (name in reused and args.answers == args.reuse):   # it is already there
            for case, run in zip(cases, rs, strict=True):
                check_answers.save(args.answers, name, case, run, held[case.id])
    rows = {name: [row for case, run in zip(cases, rs, strict=True) for row in suite.score_rows(case, run)]
            for name, rs in runs.items()}
    done = {(name, case.id): run for name, rs in runs.items() for case, run in zip(cases, rs, strict=True)}
    return rows, done, reused


def _shares(shares: dict) -> str:
    """A family's asked class shares in words, against its draw: `answerable 0.80, unknown 0.20 (the draw: unknown
    0.20)`; pools too where they differ from the classes (`yes 0.55, no 0.45`)."""
    text = ", ".join(f"{c} {x:.2f}" for c, x in shares["classes"].items())
    if shares["pools"] and set(shares["pools"]) != set(shares["classes"]):
        text += "; pools " + ", ".join(f"{p} {x:.2f}" for p, x in shares["pools"].items())
    draw = shares["draw"]
    if draw is not None:
        text += f" (the draw: {draw['pool']} {'at most ' if draw['bound'] == '<=' else ''}{draw['share']:.2f})"
    return text


def _lines(layout: str, job: str, group: dict, form: dict, scores: dict) -> list[str]:
    """The printout of one group's numbers; `scores` are the gates' per-system means of each of its families."""
    out = [f"{layout} {job}: {group['cases']} cases of {len(group['scripts'])} scripts, now share "
           f"{group['now_share']:.2f}"]
    out.append("  temporal share of answerable end questions: " + "; ".join(
        f"{length} {'n/a' if t['share'] is None else format(t['share'], '.2f')} ({t['temporal']} of {t['answerable']}, "
        f"{t['per_case']:.2f} a case over {t['cases']} cases)" for length, t in group["temporal"].items()))
    for family, f in group["families"].items():
        empty = "" if f["empty_set_share"] is None else f", empty sets {f['empty_set_share']:.2f}"
        out.append(f"  {family}: {f['questions']} questions ({f['per_case']:.2f} a case), unknown share "
                   f"{f['unknown_share']:.2f}{empty}")
        out.append("    first-level hits: " + ", ".join(f"{s} {h:.2f}" for s, h in f["hit_rates"].items()))
        never = f" ({f['note']})" if f.get("note") else " (never: its gate is vacuous)"
        own = " (own log, no pool)"
        out.append("    reach (a level above the template): " + ", ".join(
            f"{s} {r:.2f}" + ("" if r else never if answer_keys.pooled(s, family) else own)
            for s, r in f["reach_rates"].items()))
        if f["class_shares"]["classes"]:
            out.append("    class shares: " + _shares(f["class_shares"]))
        now = f["current_view_now_mean"]
        if now is not None:
            wrong = f" (under {suite_report.NOW_MEAN}: the now tag is wrong)" if now < suite_report.NOW_MEAN else ""
            out.append(f"    current_view on now rows: {now:.3f}{wrong}")
        shown = (gates.ORACLE, gates.TABLE, f["reference"])   # the family's declared reference, beside its table
        out.append("    means: " + ", ".join(f"{s} {scores[family][s]['mean']:.3f}" for s in shown
                                            if s in scores[family]))
        a = form[family]
        over = a["mean"] - a["unknown_share"]
        named = f"best feature {a['feature']}" if a["feature"] else "no feature beats it; the best"
        out.append(f"    form adversary against the unknown share: {named} scores {over:+.3f}")
    return out


def _scope_line(scope: dict) -> str:
    return (f"SCOPED CHECK ({scope['select']}): {len(scope['groups'])} of {scope['of_groups']} groups and "
            f"{scope['cases']} of {scope['of_cases']} cases that ask were checked; the verdict reads those and is not "
            "the release check's")


def _check(args: argparse.Namespace) -> int:
    began = time.time()
    cases, silent = compiled_cases(args.data)
    if not cases:
        none = f" ({len(silent)} compiled cases ask nothing)" if silent else ""
        print(f"stbench check: no case with questions and gold in {args.data}{none}; compile some first (`stbench "
              "build`, or `stbench compile`)", file=sys.stderr)
        return 2
    scope = None
    if args.cases is not None or args.select:   # scoped: only these groups and cases, gated as they would be alone
        try:
            picks = case_select.parse(args.select)
            kept = case_select.narrow(cases, args.cases, picks, args.data)
        except case_select.Bad as e:
            print(f"stbench check: {e}", file=sys.stderr)
            return 2
        scope = {"select": case_select.describe(args.cases, picks),
                 "groups": [{"layout": layout, "job": job} for layout, job in groups(kept)],
                 "of_groups": len(groups(cases)), "cases": len(kept), "of_cases": len(cases)}
        cases, silent = kept, case_select.keep(silent, args.cases, picks)
    if moved := depth_move_problem(c.script for c in cases):
        print(f"stbench check: {moved}", file=sys.stderr)
        return 2
    log = (lambda m: print(m, file=sys.stderr)) if args.verbose else (lambda m: None)
    results: dict[str, list[dict]] = defaultdict(list)
    temporal = frozenset(f.name for f in FAMILIES.values() if f.temporal)   # `time_needed` reads these
    numbers, forms, balance = {}, {}, []
    try:
        yielded = yields(cases, silent)
        every_row, every_run, reused = _check_runs(cases, args, log)
        for (layout, job), group in groups(cases).items():
            ids = {case.id for case in group}
            rows = {name: [row for row in rs if row["case"] in ids] for name, rs in every_row.items()}
            done = {key: run for key, run in every_run.items() if key[1] in ids}
            asked = asked_questions(group)
            form_rows, forms[layout, job] = form_adversary(asked)
            twin_rows, _ = form_adversary(asked, in_pool=True)
            for name, rs in {**rows, gates.FORM: form_rows, gates.in_pool_name(gates.FORM): twin_rows}.items():
                results[name] += rs
            numbers[layout, job] = suite_report.group_numbers(group, done, rows, temporal)
            balance += suite_balance.read(asked)
    except ValueError as e:   # a case the systems cannot be given (no `other_case` to feed, a run's refusal)
        print(f"stbench check: {e}", file=sys.stderr)
        return 2
    evaluated = gates.evaluate(results, shared_rule=frozenset(f.name for f in FAMILIES.values() if f.mapper_gold),
                               temporal=temporal)
    broken = suite_report.broken_instruments(evaluated, numbers)
    passed, failures = outcome(evaluated, args.verdict, broken, balance)
    balanced = not any(b["fails"] for b in balance)
    said = _scope_line(scope) if scope else None
    if said:
        print(said)
    if reused:
        print(f"REUSED the stored answers of {', '.join(reused)} from {args.reuse}; the other systems ran")
    print(gates.format_table(evaluated, [suite_balance.line(b) for b in balance], balanced and not broken))
    for y in yielded:
        if y["asked_nothing"]:
            print(f"{y['layout']} {y['job']}: {y['asked_nothing']} of {y['compiled']} compiled cases ask nothing and "
                  "are not checked")
    scores = {(r["layout"], r["job"], r["family"]): r["scores"] for r in evaluated["rows"]}
    for (layout, job), group in numbers.items():
        family_scores = {family: scores[layout, job, family] for family in group["families"]}
        print("\n".join(_lines(layout, job, group, forms[layout, job], family_scores)))
    report = {"verdict": args.verdict, "passed": passed,
              "passed_point": evaluated["passed_point"] and not broken,
              "passed_interval": evaluated["passed"] and not broken,
              "passed_evidence": evaluated["passed_evidence"] and not broken and balanced, "broken": broken,
              "balance": balance,
              "asked_nothing": yielded,
              "groups": [{"layout": layout, "job": job, **group, "form_adversary": forms[layout, job]}
                         for (layout, job), group in numbers.items()],
              "gates": evaluated["rows"], "wall_seconds": round(time.time() - began, 1)}
    if scope:
        report["scope"] = scope
    if reused:
        report["reused"] = reused
    if scope and not args.out:   # a scoped report is never the dataset's: `export` reads DATASET/check_report.json
        print(f"report not written: a scoped check leaves {args.data / 'check_report.json'}, the whole dataset's, "
              "alone; pass --out FILE to keep this report", file=sys.stderr)
    elif not args.out and (args.data / "release.json").is_file():   # the release document marks a released dataset
        print(f"report not written: {args.data} is a released dataset (it has a release.json), which a check leaves "
              "as released; pass --out FILE to keep this report", file=sys.stderr)
    else:
        out = args.out or args.data / "check_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {out}")
    if said:
        print(said)   # again at the end: the table above scrolls
    if failures:
        print("\n".join(failures), file=sys.stderr)
    return 0 if passed else 1


def _ids(text: str) -> list[str] | None:
    return text.split(",") if text else None


def _workers(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be at least 1")
    return value


def register(sub) -> None:
    p = sub.add_parser("check", help="run every scripted baseline over the compiled cases and print the validity gates")
    p.add_argument("--data", required=True, type=Path, help="dataset directory")
    p.add_argument("--out", type=Path, help="JSON report (default DATASET/check_report.json; a released dataset's is "
                   "kept unless --out is given)")
    p.add_argument("--workers", type=_workers, default=4,
                   help="processes for the runs (default %(default)s; 1 runs them in this process)")
    p.add_argument("--verdict", choices=VERDICTS, default="interval",
                   help="what the exit code follows: every gate passing on its 95%% interval (interval), on the point "
                   "estimates alone (point), or the shortcut gates on the evidence of a gain over all_unknown in the "
                   "group, a length or a slice, and the balance rule (a family fails on a job when its held-out "
                   "best constant beats all_unknown with an interval above 0) (evidence: the one `stbench export` "
                   "requires); default %(default)s")
    p.add_argument("--cases", type=_ids, metavar="ID,...",
                   help="check only these case ids (compiled cases that ask a question): a scoped check")
    p.add_argument("--select", action="append", default=[], metavar="KEY=VALUE",
                   help="check only cases whose layout, robot, job or length is VALUE (repeat a key for either value; "
                   "different keys must all hold), as `stbench eval --select`: a scoped check")
    p.add_argument("--answers", type=Path, metavar="DIR",
                   help="write each scripted system's answers to DIR/<system>/<case>.json, for --reuse")
    p.add_argument("--reuse", type=Path, metavar="DIR",
                   help="reload from DIR (what --answers wrote) the answers of the systems that do not read the scorer "
                   "instead of running them: for a check after a change to the scorer and to nothing else")
    p.add_argument("--verbose", action="store_true", help="log each run to stderr")
    p.set_defaults(func=_check)
