"""`stbench pilot report`: the pilot's measures and the release's script count, from results documents
(docs/eval.md, "The pilot report"; FD-M6-M7 §2.2 and §2.4).

    stbench pilot report [--runs R [R ...]] [--leak ON OFF]... [--brief BASE OTHER]... [--release PLAN.json]
                         [--out FILE.md]

Each R, ON, OFF, BASE and OTHER is a results document or the run folder holding one. The report is ASCII markdown,
a function of those documents alone (bootstrap seeds are fixed), with a section for each option given:
- `--runs`: the pilot rows, each during on with the correct brief. Per agent (system, memory, compaction limit and
  reasoning effort) and job: the scripts, the cases and questions a case by length, sigma (the SD of the reported score
  over questions, and per family), the design effect and rho (the Kish ICC) once there are 10 scripts, the headline
  (the mean of the `now` and `memory` columns, which count equally, each the equal mean of its abilities, with the
  diagnostic pairs the documents record left out) and its interval with those columns, how the cases ended, and per
  case-run by length the tokens, wall time and notional dollars;
- `--leak ON OFF`: D2's check, in two readings (`pilot_leak`). A pair differs in `during` alone; d = on - off of the
  penalised score (`score_gate`, so no catastrophe weight enters the verdict) over the end questions, paired by (case,
  question), with the script bootstrap's 95% interval. The verdict (`power.leak_verdict`, per job and pooled) reads the
  pairs whose gold is answerable; the pairs whose gold is `unknown` are an abstention shift, reported apart with no
  verdict, and each (job, family) has its rows;
- `--brief BASE OTHER`: BASE is the correct brief and OTHER the swapped or the neutral one, differing in `brief` alone.
  BASE - OTHER per job on the task-critical rows and on the rest, with the cases by length;
- `--release PLAN.json`: the count rule of `power` over the `--runs` (a job without pilot rows takes the plan's
  fallback sigma, and a ratio of questions it names), the budgets and the predicted half-width at each K.

Refused (exit 2, saying why, nothing printed or written): no option, a document that is not a results document, a
`--runs` document with during off or another brief, runs scored on different compiles (`eval_compare.same_compile`), a
case two runs of one agent both hold, a pair that differs in more than its one lever or that compare would refuse
(`eval_compare.load_all`), leak pairs of different agents, no end question in common, and a plan that lacks a number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import pilot_leak, power
from .case import LENGTHS
from .cell_weights import thin_from
from .eval_compare import LEVER_GROUPS, load_all, same_compile
from .eval_compare import Refused as CompareRefused
from .eval_report import cell, count, interval, lever_name, rate, signed, system_name
from .eval_results import load
from .score_stats import aggregate, paired_difference

MIN_SCRIPTS = 10    # scripts a design effect is estimated from
RHO_FLOOR = 0.05    # the release's rho is never below it
BRIEFS = ("swapped", "neutral")
TOKEN_KEYS = ("tokens_in", "tokens_cached_in", "tokens_out")


class Refused(ValueError):
    """The report cannot be made."""


@dataclass
class Group:
    """What one agent did on one job over every `--runs` document: its score rows and the records of its cases."""

    agent: str
    job: str
    rows: list[dict] = field(default_factory=list)
    cases: list[dict] = field(default_factory=list)
    thin: dict = field(default_factory=dict)   # the cells below weight 1 the documents weighed, with their weights

    @property
    def scripts(self) -> int:
        return len({r["script"] for r in self.rows})


def _agent(doc: dict) -> str:
    return system_name(doc) + (f" {doc['reasoning']}" if doc["reasoning"] else "")


def _load(path: str | Path) -> dict:
    try:
        return load(path)
    except ValueError as error:
        raise Refused(str(error)) from None


def _differs(a: dict, b: dict) -> list[str]:
    return [group for group, keys in LEVER_GROUPS.items() if any(a.get(k) != b.get(k) for k in keys)]


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _length_order(length: str) -> tuple[int, str]:
    return (list(LENGTHS).index(length) if length in LENGTHS else len(LENGTHS), length)


def _sizes(rows) -> dict[str, tuple[str, int]]:
    """case id -> (its length, the questions it asked) over score rows (one row a question)."""
    found: dict[str, list] = {}
    for r in rows:
        found.setdefault(r["case"], [r["length"], 0])[1] += 1
    return {case: (length, n) for case, (length, n) in found.items()}


def _by_length(sizes: dict[str, tuple[str, int]]) -> dict[str, tuple[int, float]]:
    """length -> (cases, mean questions a case)."""
    out = {}
    for length in sorted({length for length, _ in sizes.values()}, key=_length_order):
        asked = [n for here, n in sizes.values() if here == length]
        out[length] = (len(asked), sum(asked) / len(asked))
    return out


def _questions_per_script(sizes: dict[str, tuple[str, int]]) -> float:
    """m: the questions of one release script, q_short + q_medium + q_long over the cases. A length with fewer than 3
    cases takes q_short (the mean of every case when there is no short one), which is the conservative choice."""
    by = _by_length(sizes)
    short = by["short"][1] if "short" in by else sum(n for _, n in sizes.values()) / len(sizes)
    total = 0.0
    for length in LENGTHS:
        cases, q = by.get(length, (0, short))
        total += q if cases >= 3 else short
    return total


def _rho(g: Group) -> tuple[float, float, float] | None:
    """(design effect, rho, mean questions a script) of a group with 10 scripts and varying scores, else None."""
    if g.scripts < MIN_SCRIPTS or (de := power.design_effect(g.rows)) is None:
        return None
    m_bar = len(g.rows) / g.scripts
    return (de, power.icc(de, m_bar), m_bar) if m_bar > 1 else None


# -- the documents -----------------------------------------------------------------------------------------------
def _runs(paths: list[str]) -> dict[tuple[str, str], Group]:
    """The groups of the `--runs` documents by (agent, job). Refused unless each is during on with the correct brief."""
    docs = [_load(p) for p in paths]
    for path, doc in zip(paths, docs, strict=True):
        if not doc["during"] or doc["brief"] != "correct":
            raise Refused(f"{path} is a run with {lever_name(doc)}: --runs takes the pilot rows (during on, correct "
                          "brief); give the arms to --leak or --brief")
    try:
        same_compile(paths, docs)
    except CompareRefused as error:
        raise Refused(str(error)) from None
    groups: dict[tuple[str, str], Group] = {}
    agent_thin: dict[str, dict] = {}   # agent -> the cells below weight 1 its documents weighed, with their weights
    held: dict[tuple[str, str], int] = {}   # (agent, case id) -> the position of the document that holds it
    for i, (path, doc) in enumerate(zip(paths, docs, strict=True)):
        for case in doc["cases"]:
            if (first := held.setdefault((_agent(doc), case["id"]), i)) != i:
                raise Refused(f"{path} and {paths[first]} both hold case {case['id']} of {_agent(doc)}: give one run "
                              "a case")
            groups.setdefault((_agent(doc), case["job"]), Group(_agent(doc), case["job"])).cases.append(case)
        for row in doc["rows"]:
            groups.setdefault((_agent(doc), row["job"]), Group(_agent(doc), row["job"])).rows.append(row)
        mine = agent_thin.setdefault(_agent(doc), {})
        for agg in doc["layouts"].values():
            if agg:
                mine.update(thin_from(agg["thin"]))
    for group in groups.values():
        group.thin = agent_thin[group.agent]
    return groups


def _pair(paths: list[str], lever: str) -> list[dict]:
    """The two results of a pair, which must differ in `lever` alone (and be comparable: `eval_compare`)."""
    try:
        docs, _ = load_all(list(paths))
    except CompareRefused as error:
        raise Refused(str(error)) from None
    if (differing := _differs(*docs)) != [lever]:
        raise Refused(f"{paths[0]} and {paths[1]} differ in {', '.join(differing) or 'nothing'}, not in {lever} alone")
    return docs


# -- --runs ------------------------------------------------------------------------------------------------------
def _resources(g: Group) -> list[str]:
    """Tokens, wall time and notional dollars of a group's case-runs (a case whose usage was reported), per length."""
    ran = [c for c in g.cases if c["tokens"]["tokens_in"] is not None]
    if not ran:
        return ["No case-run reported its tokens.", ""]

    def mean(values: list, digits: int = 0) -> str:
        known = [v for v in values if v is not None]
        return f"{sum(known) / len(known):,.{digits}f}" if known else "n/a"

    rows = []
    for name, here in [*((length, [c for c in ran if c["length"] == length])
                         for length in sorted({c["length"] for c in ran}, key=_length_order)), ("all", ran)]:
        rows.append([name, str(len(here)), *(mean([c["tokens"][key] for c in here]) for key in TOKEN_KEYS),
                     mean([sum(c["wall_s"].values()) for c in here], 1), mean([c["cost_usd"] for c in here], 4)])
    total = {key: sum(c["tokens"][key] or 0 for c in ran) for key in TOKEN_KEYS}
    unpriced = sum(c["cost_usd"] is None for c in ran)
    return ["Per case-run (mean):", "",
            *_table(["length", "case-runs", "tokens in", "of them cached", "tokens out", "wall s", "USD"], rows),
            f"Over the {count(len(ran), 'case-run')}: {total['tokens_in']:,} tokens in ({total['tokens_cached_in']:,} "
            f"cached), {total['tokens_out']:,} out"
            + (f"; {count(unpriced, 'case')} could not be priced" if unpriced else "") + ".", ""]


def _group(g: Group) -> list[str]:
    layouts = "+".join(sorted({c["layout"] for c in g.cases}))
    lines = [f"### {g.agent}: {g.job} ({layouts})", "",
             "- cases: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(c["status"] for c in g.cases).items()))]
    if not g.rows:
        return [*lines, "- no case finished, so nothing is scored", "", *_resources(g)]
    sizes = _sizes(g.rows)
    agg, sd = aggregate(g.rows, thin=g.thin), power.score_sd(g.rows)
    scopes = ", ".join(f"{k} {cell(v)} over {v['n']}" for k, v in agg["columns"].items())
    lines += [f"- {count(g.scripts, 'script')}, {count(len(sizes), 'scored case')}, {count(len(g.rows), 'question')}",
              f"- headline: {cell(agg['headline'])}", f"- columns: {scopes}",
              "- sigma: " + ("n/a" if sd is None else f"{sd:.3f}") + " (the SD of the score over the questions)"]
    if (found := _rho(g)) is not None:
        lines.append(f"- design effect {found[0]:.3f}; rho {found[1]:.3f} (the Kish ICC, over {found[2]:.1f} "
                     "questions a script)")
    else:
        lines.append(f"- design effect and rho: need {MIN_SCRIPTS} scripts and scores that vary (this has "
                     f"{g.scripts})")
    by = _by_length(sizes)
    lines += ["", *_table(["length", "cases", "questions a case"],
                          [[k, str(n), f"{q:.1f}"] for k, (n, q) in by.items()])]
    families = sorted({r["family"] for r in g.rows})
    lines += _table(["family", "questions", "sigma"], [
        [f, str(len(here := [r for r in g.rows if r["family"] == f])),
         "n/a" if (s := power.score_sd(here)) is None else f"{s:.3f}"] for f in families])
    return lines + _resources(g)


def _runs_section(groups: dict[tuple[str, str], Group]) -> list[str]:
    lines = ["## Pilot rows", "", "Means are the reported score; intervals are 95% percentile bootstraps over scripts "
             "(`score_stats`); the headline is the mean of the `now` and `memory` columns, which count equally; "
             "sigma is a sample SD over questions. A case-run is one case run once by one agent.", ""]
    for key in sorted(groups):
        lines += _group(groups[key])
    return lines


# -- --leak ------------------------------------------------------------------------------------------------------
def _leak_section(pairs: list[list[dict]]) -> list[str]:
    for on, off in pairs:
        if not on["during"] or off["during"]:
            raise Refused(f"a --leak pair is ON then OFF: the first has during {'on' if on['during'] else 'off'} and "
                          f"the second during {'on' if off['during'] else 'off'}")
    if len(agents := sorted({_agent(on) for on, _ in pairs})) > 1:
        raise Refused(f"the --leak pairs are of different agents ({', '.join(agents)}): D2 pools one agent over "
                      "the jobs")
    on = [r for pair in pairs for r in pair[0]["rows"] if r["when"] == "end"]
    off = [r for pair in pairs for r in pair[1]["rows"] if r["when"] == "end"]
    try:
        return pilot_leak.section(agents[0], on, off)
    except pilot_leak.Bad as error:
        raise Refused(str(error)) from None


# -- --brief -----------------------------------------------------------------------------------------------------
def _brief_section(pairs: list[list[dict]]) -> list[str]:
    lines = ["## The brief swap", "", "BASE - OTHER (the correct brief minus the other), paired by (case, question), "
             "over the task-critical rows and over the rest, with 95% bootstrap intervals over scripts. Descriptive: "
             "no gate.", ""]
    for base, other in pairs:
        if base["brief"] != "correct" or other["brief"] not in BRIEFS:
            raise Refused(f"a --brief pair is BASE (correct) then OTHER ({' or '.join(BRIEFS)}), not "
                          f"{base['brief']} then {other['brief']}")
        shared = {r["case"] for r in other["rows"]}
        lines += [f"### correct minus {other['brief']}: {_agent(base)}", ""]
        table, shorts = [], []
        for job in sorted({r["job"] for r in base["rows"]}):
            mine = [r for r in base["rows"] if r["job"] == job and r["case"] in shared]
            by = _by_length(_sizes(mine))
            shorts.append(by.get("short", (0, 0))[0] * 2 > sum(n for n, _ in by.values()))
            for critical in (True, False):
                try:
                    d = paired_difference(base["rows"], other["rows"], where={"job": job, "critical": critical})
                    found = [str(d["n"]), str(d["scripts"]), signed(d["mean"]), interval(d["ci"])]
                except ValueError:
                    found = ["-", "-", "n/a", "n/a"]
                table.append([job, "critical" if critical else "the rest", *found,
                              ", ".join(f"{k} {n}" for k, (n, _) in by.items())])
        lines += _table(["job", "rows", "pairs", "scripts", "BASE - OTHER", "95% CI", "cases by length"], table)
        if shorts and all(shorts):
            lines += ["Most of these cases are short, so the table says little about a brief's effect over a long "
                      "horizon.", ""]
    return lines


# -- --release ---------------------------------------------------------------------------------------------------
def _number(source, key: str, where: str, *, whole: bool = False, zero: bool = False) -> float:
    value = source.get(key) if isinstance(source, dict) else None
    ok = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and (
        value >= 0 if zero else value > 0)
    if not ok or (whole and value != int(value)):
        raise Refused(f"{where}: {key!r} must be a {'whole ' if whole else ''}{'non-negative' if zero else 'positive'}"
                      f" number, not {value!r}")
    return value


def _plan(path: Path) -> dict:
    """The validated plan: `target` (the half-width), `k_min`, `render_hours`, `points`, `fallback` {sigma, rho} and
    `jobs`, a dict by job of `frames_per_script`, `seconds_per_frame`, `points_per_script` and, for a job without pilot
    rows, `questions_per_script`."""
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refused(f"{path}: cannot be read as JSON: {error}") from None
    where = f"plan {path}"
    if not isinstance(plan, dict):
        raise Refused(f"{where}: must be a JSON object")
    for key in ("target", "render_hours", "points"):
        _number(plan, key, where)
    _number(plan, "k_min", where, whole=True)
    _number(plan.get("fallback"), "sigma", where + " fallback", zero=True)
    _number(plan["fallback"], "rho", where + " fallback", zero=True)
    if not isinstance(plan.get("jobs"), dict) or not plan["jobs"]:
        raise Refused(f"{where}: 'jobs' must name at least one job")
    for job, entry in plan["jobs"].items():
        for key in ("frames_per_script", "seconds_per_frame", "points_per_script"):
            _number(entry, key, f"{where} job {job}", zero=True)
        if "questions_per_script" in entry:
            _number(entry, "questions_per_script", f"{where} job {job}")
    return plan


def _release_section(plan: dict, groups: dict[tuple[str, str], Group]) -> list[str]:
    found = [r for g in groups.values() if (r := _rho(g)) is not None]
    rho = max(RHO_FLOOR, max(r[1] for r in found)) if found else plan["fallback"]["rho"]
    rho_note = (f"the largest of {count(len(found), 'agent-job pair')} with {MIN_SCRIPTS} or more scripts, at least "
                f"{RHO_FLOOR}" if found else "the plan's fallback: no agent-job pair has enough scripts")
    inputs = {}
    for job, entry in sorted(plan["jobs"].items()):
        mine = [g for g in groups.values() if g.job == job]
        sds = [sd for g in mine if (sd := power.score_sd(g.rows)) is not None]
        sizes = _sizes({(r["case"], r["qid"]): r for g in mine for r in g.rows}.values())
        if not sizes and "questions_per_script" not in entry:
            raise Refused(f"job {job} has no pilot rows and the plan gives no 'questions_per_script' for it")
        inputs[job] = (max(sds) if sds else plan["fallback"]["sigma"],
                       _questions_per_script(sizes) if sizes else entry["questions_per_script"],
                       "pilot" if sds else "fallback")
    targets = {job: power.scripts_needed(s, m, rho, plan["target"]) for job, (s, m, _) in inputs.items()}
    jobs = plan["jobs"]
    seconds = {job: jobs[job]["frames_per_script"] * jobs[job]["seconds_per_frame"] for job in jobs}
    points = {job: jobs[job]["points_per_script"] for job in jobs}
    alloc = power.allocate(targets, seconds, points, plan["render_hours"] * 3600, plan["points"], int(plan["k_min"]))
    rows = []
    for job, (sigma, m, source) in inputs.items():
        k = alloc.k[job]
        rows.append([job, f"{sigma:.3f} ({source})", f"{m:.1f}", str(targets[job]), str(k),
                     f"{power.half_width(sigma, m, rho, k):.3f}", f"{k * jobs[job]['frames_per_script']:,.0f}",
                     f"{k * seconds[job] / 3600:.1f}", f"{k * points[job]:.1f}"])
    hours = sum(alloc.k[j] * seconds[j] for j in jobs) / 3600
    used = sum(alloc.k[j] * points[j] for j in jobs)
    rows.append(["total", "", "", str(sum(targets.values())), str(sum(alloc.k.values())), "",
                 f"{sum(alloc.k[j] * jobs[j]['frames_per_script'] for j in jobs):,.0f}", f"{hours:.1f}", f"{used:.1f}"])
    lines = ["## The release's script count", "",
             f"K_target = ceil((1.96 sigma / h)^2 (1 + (m - 1) rho) / m) with h = {plan['target']} (the 95% half-width "
             "of a pooled mean of an agent's questions); K = max(k_min, floor(f K_target)) for the largest f <= 1 "
             "within both budgets.",
             "", f"- rho {rho:.3f}: {rho_note}", f"- k_min {int(plan['k_min'])}; render budget "
             f"{plan['render_hours']} GPU-hours; quota budget {plan['points']} points", "",
             *_table(["job", "sigma", "m", "K target", "K", "half-width at K", "frames", "GPU-hours", "points"], rows),
             f"f = {rate(alloc.f)}; render {hours:.1f} of {plan['render_hours']} GPU-hours, quota {used:.1f} of "
             f"{plan['points']} points.", ""]
    if alloc.over_budget:
        lines += [f"*** OVER BUDGET: even {int(plan['k_min'])} scripts a job overrun a budget; K is k_min "
                  "everywhere ***", ""]
    return lines


# -- the command -------------------------------------------------------------------------------------------------
def build(runs: list[str], leaks: list[list[str]], briefs: list[list[str]], release: Path | None) -> str:
    """The report as markdown. Refused as the module says, before any section is made."""
    if not (runs or leaks or briefs or release):
        raise Refused("give at least one of --runs, --leak, --brief or --release")
    groups = _runs(runs) if runs else {}
    leak_pairs = [_pair(p, "during") for p in leaks]
    brief_pairs = [_pair(p, "brief") for p in briefs]
    plan = _plan(release) if release else None
    lines = ["# SpatiotemporalBench: pilot report", ""]
    if groups:
        lines += _runs_section(groups)
    if leak_pairs:
        lines += _leak_section(leak_pairs)
    if brief_pairs:
        lines += _brief_section(brief_pairs)
    if plan:
        lines += _release_section(plan, groups)
    return "\n".join(lines).rstrip("\n") + "\n"


def _report(args: argparse.Namespace) -> int:
    try:
        text = build(args.runs, args.leak, args.brief, args.release)
    except ValueError as error:   # Refused, or a document that repeats a question
        print(f"stbench pilot report: {error}", file=sys.stderr)
        return 2
    if args.out is not None:
        try:
            args.out.write_text(text, encoding="utf-8", newline="\n")
        except OSError as error:
            print(f"stbench pilot report: cannot write {args.out}: {error}", file=sys.stderr)
            return 2
    print(text, end="")
    return 0


DESCRIPTION = """\
Report the pilot from results documents (a results.json or its run folder), as ASCII markdown. One section for each
option given:
  --runs     the pilot rows (during on, correct brief): per agent and job the scripts, cases and questions a case by
             length, sigma (also per family), the design effect and rho (from 10 scripts), the headline and its 95%
             interval, how the cases ended, and per case-run the tokens, wall time and notional USD by length.
  --leak     D2: ON and OFF differ in `during` alone; d = on - off over the end questions, with the script bootstrap's
             interval. The verdict (pass, inconclusive, fail against the 0.05 margin) reads the answerable pairs, per
             job and pooled; the `unknown`-gold pairs are an abstention shift, and each family has its rows.
  --brief    the brief swap: BASE (correct) minus OTHER (swapped or neutral), per job, on the task-critical rows and
             on the rest.
  --release  the plan's script count: K_target from sigma, m and rho, cut to the render and quota budgets, with the
             predicted half-width. See docs/eval.md. Exit 2 when the inputs cannot be reported."""


def register(sub) -> None:
    p = sub.add_parser("pilot", help="the pilot: `pilot report` (the M6 measures and the release's script count)")
    r = p.add_subparsers(dest="pilot_command", required=True).add_parser(
        "report", help="report the pilot's results: sigma, design effect and rho, D2's leak check, the brief swap and "
                       "the release's script count", description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    r.add_argument("--runs", nargs="+", default=[], metavar="RESULTS",
                   help="the pilot rows: results.json files or run folders, each during on with the correct brief")
    r.add_argument("--leak", nargs=2, action="append", default=[], metavar=("ON", "OFF"),
                   help="D2: a during on and a during off result of one agent (repeat for each job)")
    r.add_argument("--brief", nargs=2, action="append", default=[], metavar=("BASE", "OTHER"),
                   help="the brief swap: the correct brief's result and the swapped or neutral one (repeat)")
    r.add_argument("--release", type=Path, metavar="PLAN.json",
                   help="the release's count: the plan with the target, k_min, budgets, fallback and the jobs' frames")
    r.add_argument("--out", type=Path, metavar="FILE.md", help="also write the report to this file")
    r.set_defaults(func=_report)
