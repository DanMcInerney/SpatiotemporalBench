"""`stbench pilot report` (`cmd_pilot.py`, FD-M6-M7 §2.2 and §2.4): the pilot's measures, D2's leak check, the brief
swap and the release's script count, from hand-written results documents. Every expected number is worked out by
hand in the comment above it (the module `power` has its own tests); the documents are built here, with no run
behind them."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from stbench.cli import main
from stbench.score_stats import aggregate

SOL = ("codex:gpt-6.1-sol", "medium")
LUNA = ("codex:gpt-6-luna", "xhigh")
SOL_NAME = "codex:gpt-6.1-sol medium"
NO_TOKENS = dict.fromkeys(("tokens_in", "tokens_cached_in", "tokens_cache_write_in", "tokens_out"))


def case(cid: str, script: str, job: str, length: str, *, status: str = "complete", tokens=None, wall: float = 10.0,
        cost: float = 0.1, layout: str = "apartment_a") -> dict:
    """A case record of a results document; `tokens` is (in, cached, out), None for a case that never ran."""
    held = NO_TOKENS if tokens is None else dict(zip(("tokens_in", "tokens_cached_in", "tokens_out"), tokens,
                                                     strict=True), tokens_cache_write_in=0)
    return {"id": cid, "script": script, "layout": layout, "robot": "robot", "job": job, "length": length,
            "status": status, "failures": {}, "wall_s": {} if tokens is None else {"m": wall}, "state_bytes": {},
            "usage": None, "usage_problems": [], "tokens": held, "remote_memory": False,
            "cost_usd": None if tokens is None else cost, "cost_note": None, "error": None}


def row(cid: str, script: str, qid: str, score: float, *, job: str, length: str = "short", when: str = "end",
        family: str = "f_a", critical: bool = False, layout: str = "apartment_a", scope: str = "memory",
        gate: float | None = None) -> dict:
    """A result row; `gate` is its `score_gate` (and the row a catastrophe, whose `score` replaces the gate), else it
    has none and `score` is both, as a row of an earlier build."""
    return {"case": cid, "script": script, "qid": qid, "family": family, "when": when, "scope": scope,
            "reference": None, "critical": critical, "slice": "answerable", "answer_class": None, "range_m": None,
            "s": 1.0,
            "score": float(score),
            **({} if gate is None else {"score_gate": float(gate), "catastrophe": True}),
            "outcome": "correct", "layout": layout, "robot": "robot", "job": job, "length": length}


def doc(agent: tuple[str, str], cases: list[dict], rows: list[dict], *, during: bool = True, brief: str = "correct",
        channel: str = "rgb_pose") -> dict:
    return {"system": agent[0], "channel": channel, "brief": brief,
            "during": during, "sandbox": "none", "reasoning": agent[1], "text_only": False,
            "verified": False, "partial": False, "max_usd": None, "cost_usd": None, "cases": cases, "rows": rows,
            "layouts": {}, "notes": []}


def save(tmp_path: Path, name: str, document: dict) -> str:
    """The run folder `tmp_path/name` holding the document as its `results.json`."""
    folder = tmp_path / name
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps(document), encoding="utf-8")
    return str(folder)


def run(*argv) -> int:
    return main(["pilot", "report", *(str(a) for a in argv)])


def report(capsys, *argv) -> str:
    assert run(*argv) == 0
    return capsys.readouterr().out


def section(md: str, heading: str) -> list[str]:
    """The lines of the part of `md` that starts at the line `heading` and ends at the next heading."""
    lines = md.splitlines()
    start = lines.index(heading) + 1
    stop = next((i for i in range(start, len(lines)) if lines[i].startswith("#")), len(lines))
    return lines[start:stop]


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip("|").split("|")]


def pilot_doc(agent, scripts: dict[str, list[float]], job: str = "vacuum") -> dict:
    """A run with one short case a script, a question for each score of the script."""
    cases = [case(f"c-{s}", s, job, "short", tokens=(1000, 100, 100)) for s in scripts]
    rows = [row(f"c-{s}", s, f"q{j}", x, job=job) for s, xs in scripts.items() for j, x in enumerate(xs)]
    return doc(agent, cases, rows)


SOL_SCORES = {**{f"s{i}": [1, 1, 1, 0] for i in range(5)}, **{f"s{i}": [1, 0, 0, 0] for i in range(5, 10)}}
LUNA_SCORES = {**{f"s{i}": [1, 1, 1, 0.5] for i in range(5)}, **{f"s{i}": [0.5, 0, 0, 0] for i in range(5, 10)}}


def plan_file(tmp_path: Path, name: str = "plan.json", **changes) -> Path:
    """A release plan: a 0.10 target, 6 scripts at least, budgets of 40 GPU-hours and 100 points, the fallback
    sigma 0.8 and rho 0.05, and two jobs a script of which takes an hour to render."""
    plan = {"target": 0.1, "k_min": 6, "render_hours": 40, "points": 100, "fallback": {"sigma": 0.8, "rho": 0.05},
            "jobs": {"a": {"frames_per_script": 1000, "seconds_per_frame": 3.6, "points_per_script": 1.0,
                           "questions_per_script": 20},
                     "b": {"frames_per_script": 500, "seconds_per_frame": 7.2, "points_per_script": 0.5,
                           "questions_per_script": 30}}, **changes}
    path = tmp_path / name
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_the_pilot_rows_by_hand(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.2): per agent and job the report gives the scripts, the cases and questions a case by
    length, sigma (overall and per family), the headline, how the cases ended and, per case-run by length, the tokens,
    wall time and dollars; two documents of one agent on two jobs make two sections. By hand, one vacuum job, five
    scored cases (questions 3, 4, 3, 2, 3 over three scripts) and two that scored nothing:
    - scores c1 1 1 0; c2 1 0 0 1; c3 0 0 1; c4 1 0; c5 0 1 0: seven ones in 15, pooled mean 7/15 = 0.467, sigma =
      sqrt(15/14 x 7/15 x 8/15) = sqrt(4/15) = 0.516; family f_a has six rows (four ones), sigma sqrt(6/5 x 4/6 x 2/6)
      = 0.516, f_b seven (two ones), sqrt(7/6 x 2/7 x 5/7) = 0.488, and c4's two rows are the families n_a (one) and n_b
      (zero), one row each, sigma n/a; the two rows of c4 are `now` (one hit: 0.5, as in
      every resample that holds its script), the other 13 `memory` (six hits: 6/13 = 0.462); n_a and n_b are two
      abilities of `now`, f_a and f_b two of `memory`, each equal inside its column: `now` (1 + 0)/2 = 1/2, `memory`
      f_a 4/6 and f_b 2/7, (2/3 + 2/7)/2 =
      10/21, so the headline, the mean of the two columns, is (1/2 + 10/21)/2 = 41/84 = 0.488 (the `columns` line
      shows these two, `now` 1/2 over 2 and `memory` 10/21 = 0.476 over 13; pooled over everything it would be 0.467);
    - lengths: short c1 c3 c5 of 3 questions, medium c4 of 2, long c2 of 4;
    - case-runs (in, cached, out, wall s, USD): c1 1000 400 100 10 .1; c3 2000 600 300 20 .2; c5 3000 800 200 30 .3;
      a short case that ended on its context 4000 1000 500 50 .5 (four short: means 2,500 700 275 27.5 0.275); medium
      c4 5000 2000 400 40 .4; long c2 9000 3000 900 60 .9; all six: 24,000 in, 7,800 cached, 2,400 out, means 4,000
      1,300 400 35.0 0.4; a pending case ran nothing and has no case-run.
    Regression caught: a headline that pools the columns' questions instead of weighing the columns and the abilities
    in them equally, a
    case-run counted for a case that never ran, one agent's two jobs merged, or the design effect estimated from three
    scripts."""
    cases = [case("c1", "s1", "vacuum", "short", tokens=(1000, 400, 100), wall=10, cost=0.1),
             case("c2", "s1", "vacuum", "long", tokens=(9000, 3000, 900), wall=60, cost=0.9),
             case("c3", "s2", "vacuum", "short", tokens=(2000, 600, 300), wall=20, cost=0.2),
             case("c4", "s2", "vacuum", "medium", tokens=(5000, 2000, 400), wall=40, cost=0.4),
             case("c5", "s3", "vacuum", "short", tokens=(3000, 800, 200), wall=30, cost=0.3),
             case("c6", "s3", "vacuum", "short", status="ended: context", tokens=(4000, 1000, 500), wall=50, cost=0.5),
             case("c7", "s3", "vacuum", "short", status="pending")]
    scores = [("c1", "s1", "short", "f_a", 1), ("c1", "s1", "short", "f_a", 1), ("c1", "s1", "short", "f_b", 0),
              ("c2", "s1", "long", "f_a", 1), ("c2", "s1", "long", "f_b", 0), ("c2", "s1", "long", "f_b", 0),
              ("c2", "s1", "long", "f_a", 1), ("c3", "s2", "short", "f_a", 0), ("c3", "s2", "short", "f_b", 0),
              ("c3", "s2", "short", "f_b", 1), ("c4", "s2", "medium", "n_a", 1), ("c4", "s2", "medium", "n_b", 0),
              ("c5", "s3", "short", "f_a", 0), ("c5", "s3", "short", "f_b", 1), ("c5", "s3", "short", "f_b", 0)]
    rows = [row(c, s, f"q{i}", x, job="vacuum", length=length, family=f, scope="now" if c == "c4" else "memory")
            for i, (c, s, length, f, x) in enumerate(scores)]
    other = doc(SOL, [case("c8", "s8", "amr", "short", layout="warehouse_a", tokens=(10, 0, 1))],
                [row("c8", "s8", "q0", 1, job="amr", layout="warehouse_a"),
                 row("c8", "s8", "q1", 0, job="amr", layout="warehouse_a")])
    md = report(capsys, "--runs", save(tmp_path, "vacuum", doc(SOL, cases, rows)), save(tmp_path, "amr", other))
    assert md.index(f"### {SOL_NAME}: amr (warehouse_a)") < md.index(f"### {SOL_NAME}: vacuum (apartment_a)")
    lines = section(md, f"### {SOL_NAME}: vacuum (apartment_a)")
    assert "- cases: complete 5, ended: context 1, pending 1" in lines
    assert "- 3 scripts, 5 scored cases, 15 questions" in lines
    text = "\n".join(lines)
    assert re.search(r"^- headline: \+0\.488 \[[-+]\d\.\d{3}, [-+]\d\.\d{3}\]$", text, re.M)
    scope = r"^- columns: now \+0\.500 \[\+0\.500, \+0\.500\] over 2, memory \+0\.476 \[[^\]]+\] over 13$"
    assert re.search(scope, text, re.M)
    assert "- sigma: 0.516 (the SD of the score over the questions)" in lines
    assert "- design effect and rho: need 10 scripts and scores that vary (this has 3)" in lines
    assert {"| short | 3 | 3.0 |", "| medium | 1 | 2.0 |", "| long | 1 | 4.0 |", "| f_a | 6 | 0.516 |",
            "| n_a | 1 | n/a |", "| n_b | 1 | n/a |",
            "| f_b | 7 | 0.488 |"} <= set(lines)
    assert {"| short | 4 | 2,500 | 700 | 275 | 27.5 | 0.2750 |", "| medium | 1 | 5,000 | 2,000 | 400 | 40.0 | 0.4000 |",
            "| long | 1 | 9,000 | 3,000 | 900 | 60.0 | 0.9000 |",
            "| all | 6 | 4,000 | 1,300 | 400 | 35.0 | 0.4000 |"} <= set(lines)
    assert "Over the 6 case-runs: 24,000 tokens in (7,800 cached), 2,400 out." in lines


def test_the_design_effect_and_rho_need_ten_scripts(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.4): the design effect and the Kish ICC are reported from 10 scripts, per agent and job,
    from the cluster variance over the naive one. By hand (ten scripts of four questions: five of 1 1 1 0 and five of
    1 0 0 0 for the first agent, 1 1 1 0.5 and 0.5 0 0 0 for the second): the first has cluster variance 1/144, naive
    1/156, DE 13/12 = 1.083, rho (1/12)/3 = 0.028, sigma sqrt(10/39) = 0.506; the second has cluster variance
    10/9 x 22.5 / 1600 = 1/64, naive 7.5 / 39 / 40 = 1/208, DE 3.25, rho 2.25/3 = 0.75, sigma sqrt(7.5/39) = 0.439. With
    nine scripts neither is estimated. Regression caught: a design effect from too few scripts, or one that clusters
    by question and so reads 1."""
    both = [save(tmp_path, "sol", pilot_doc(SOL, SOL_SCORES)), save(tmp_path, "luna", pilot_doc(LUNA, LUNA_SCORES))]
    md = report(capsys, "--runs", *both)
    sol = section(md, f"### {SOL_NAME}: vacuum (apartment_a)")
    luna = section(md, "### codex:gpt-6-luna xhigh: vacuum (apartment_a)")
    tail = "(the Kish ICC, over 4.0 questions a script)"
    sigma = "- sigma: {} (the SD of the score over the questions)"
    assert f"- design effect 1.083; rho 0.028 {tail}" in sol and sigma.format("0.506") in sol
    assert f"- design effect 3.250; rho 0.750 {tail}" in luna and sigma.format("0.439") in luna
    nine = {s: xs for s, xs in SOL_SCORES.items() if s != "s9"}
    md = report(capsys, "--runs", save(tmp_path, "nine", pilot_doc(SOL, nine)))
    assert "- design effect and rho: need 10 scripts and scores that vary (this has 9)" in md


def leak_pair(tmp_path, tag: str, job: str, layout: str, cases: dict, *, agent=SOL, stray: bool = False):
    """(ON folder, OFF folder) for a job. `cases` is {case: (script, [(qid, on score, off score)])}, the end questions
    of each; every ON case also asks a during question `d1` that scored 1. With `stray` the OFF document also holds a
    during row `d1` for the first case (scoring 0), which a sound off arm never writes."""
    records = [case(c, s, job, "short", layout=layout) for c, (s, _) in cases.items()]
    on, off = [], []
    for c, (s, ends) in cases.items():
        on.append(row(c, s, "d1", 1, job=job, when="during", layout=layout))
        for qid, a, b in ends:
            on.append(row(c, s, qid, a, job=job, layout=layout))
            off.append(row(c, s, qid, b, job=job, layout=layout))
    if stray:
        off.append(row(next(iter(cases)), cases[next(iter(cases))][0], "d1", 0, job=job, when="during", layout=layout))
    return (save(tmp_path, f"{tag}-on", doc(agent, records, on)),
            save(tmp_path, f"{tag}-off", doc(agent, records, off, during=False)))


# vacuum: pair differences (on - off) c1 +1, 0 and c2 0, +1 over scripts s1 and s2
VACUUM = {"c1": ("s1", [("e1", 1, 0), ("e2", 0, 0)]), "c2": ("s2", [("e1", 0, 0), ("e2", 1, 0)])}
# amr: c3 0, -1 and c4 0, -1 over scripts s3 and s4
AMR = {"c3": ("s3", [("e1", 0.5, 0.5), ("e2", -0.5, 0.5)]), "c4": ("s4", [("e1", 0, 0), ("e2", 0, 1)])}


def test_the_leak_check_pairs_end_questions_per_job_and_pooled(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.2, D2): d is the mean of on - off over the end questions, paired by (case, question), per
    job and pooled over every pair, with the script bootstrap's interval and `leak_verdict`; during rows are never
    paired. By hand: vacuum pairs +1, 0, 0, +1 over two scripts (script totals +1 and +1, so every resample has mean
    0.5): d +0.500 [+0.500, +0.500], above the 0.05 margin, fail. amr pairs 0, -1, 0, -1 over scripts of total -1 and
    -1: d -0.500 [-0.500, -0.500], pass. Pooled: eight pairs, script totals +1, +1, -1, -1, so a resample of the four
    scripts has mean (sum of four +-1)/8 in {-0.5, -0.25, 0, 0.25, 0.5}, and 0.5 comes up with probability 1/16 (125 of
    2000 draws, far over the 2.5% tail): d +0.000, interval [-0.500, +0.500], inconclusive. The vacuum's off document
    holds a stray during row for c1 (the on arm scored it 1, the off 0): were it paired, vacuum would have 5 pairs.
    Regression
    caught: pairs that include a during question, a pooled d that is a mean of the jobs' means, or an interval that
    ignores the scripts."""
    leak = [*leak_pair(tmp_path, "v", "vacuum", "apartment_a", VACUUM, stray=True),
            *leak_pair(tmp_path, "a", "amr", "warehouse_a", AMR)]
    md = report(capsys, "--leak", *leak[:2], "--leak", *leak[2:])
    lines = md.splitlines()
    assert "| vacuum (apartment_a) | 4 | 2 | +0.500 | [+0.500, +0.500] | fail |" in lines
    assert "| amr (warehouse_a) | 4 | 2 | -0.500 | [-0.500, -0.500] | pass |" in lines
    assert "| pooled | 8 | 4 | +0.000 | [-0.500, +0.500] | inconclusive |" in lines
    verdict = next(x for x in lines if x.startswith("Pooled verdict:"))
    assert "**inconclusive**" in verdict and "cannot exclude a leak of the upper bound (+0.500)" in verdict
    assert f"Agent {SOL_NAME}." in md


def test_the_leak_verdict_names_what_to_do(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.2, D2): a pooled pass says the during questions ship; a pooled fail names the jobs whose
    own d is above the margin, to drop before the release compile. By hand: identical on and off scores give every pair
    0, so d +0.000 [+0.000, +0.000] and the upper bound is at or below 0.05, pass; the vacuum pair alone (d +0.5) fails
    and names vacuum. Regression caught: a verdict that does not say which job's families to drop."""
    same = {"c1": ("s1", [("e1", 1, 1), ("e2", 0, 0)]), "c2": ("s2", [("e1", 0.5, 0.5)])}
    md = report(capsys, "--leak", *leak_pair(tmp_path, "same", "vacuum", "apartment_a", same))
    assert "| pooled | 3 | 2 | +0.000 | [+0.000, +0.000] | pass |" in md.splitlines()
    assert "Pooled verdict: **pass**: the during questions ship." in md.splitlines()
    md = report(capsys, "--leak", *leak_pair(tmp_path, "bad", "vacuum", "apartment_a", VACUUM))
    assert "| pooled | 4 | 2 | +0.500 | [+0.500, +0.500] | fail |" in md.splitlines()
    assert any(x.startswith("Pooled verdict: **fail**") and "(vacuum)" in x for x in md.splitlines())


def test_a_catastrophe_row_changes_no_leak_verdict(tmp_path, capsys):
    """Contract (the final review's S4; `score_stats` "Two scores"): D2 reads the penalised score (`score_gate`), never
    the reported one, so the weight of a catastrophe cannot move its verdict. By hand: two scripts, each with an end
    question both arms answer alike (1) and one both answer wrong (gate -1); in the off arm the wrong answer is a
    catastrophe (reported score -2, gate -1) and in the on arm it is not (-1, -1). Every gate difference is 0, so d is
    +0.000 [+0.000, +0.000] and the verdict pass, per job and pooled; read as the reported score the wrong pair would be
    +1, d +0.500 and fail. Regression caught: a leak check that reads the reported score, which calls a harm weight a
    leak (or hides one, in the other arm)."""
    cases = {"c1": "s1", "c2": "s2"}
    records = [case(c, s, "vacuum", "short") for c, s in cases.items()]
    on, off = [], []
    for c, s in cases.items():
        on += [row(c, s, "e1", 1, job="vacuum"), row(c, s, "e2", -1, job="vacuum")]
        off += [row(c, s, "e1", 1, job="vacuum"), row(c, s, "e2", -2, job="vacuum", gate=-1)]
    md = report(capsys, "--leak", save(tmp_path, "on", doc(SOL, records, on)),
                save(tmp_path, "off", doc(SOL, records, off, during=False)))
    lines = md.splitlines()
    assert "| vacuum (apartment_a) | 4 | 2 | +0.000 | [+0.000, +0.000] | pass |" in lines
    assert "| pooled | 4 | 2 | +0.000 | [+0.000, +0.000] | pass |" in lines
    assert "Pooled verdict: **pass**: the during questions ship." in lines


def brief_pair(tmp_path, other_brief: str = "swapped", length: str = "short"):
    """(BASE, OTHER) folders: the amr job, three cases of the given length (and one of another when short) of three
    scripts. Per question (BASE score, OTHER score): c1 critical (1, 0) and (1, 1), other (0, 0.5); c2 critical (1, 0),
    other (0.5, 0); c3 critical (0, 0.5), other (1, 1)."""
    plan = {"c1": ("s1", length, [(True, 1, 0), (True, 1, 1), (False, 0, 0.5)]),
            "c2": ("s2", length, [(True, 1, 0), (False, 0.5, 0)]),
            "c3": ("s3", "medium" if length == "short" else length, [(True, 0, 0.5), (False, 1, 1)])}
    records = [case(c, s, "amr", ln, layout="warehouse_a") for c, (s, ln, _) in plan.items()]
    base, other = [], []
    for c, (s, ln, qs) in plan.items():
        for i, (critical, a, b) in enumerate(qs):
            kw = {"job": "amr", "length": ln, "critical": critical, "layout": "warehouse_a"}
            base.append(row(c, s, f"q{i}", a, **kw))
            other.append(row(c, s, f"q{i}", b, **kw))
    return (save(tmp_path, "base", doc(SOL, records, base)),
            save(tmp_path, other_brief, doc(SOL, records, other, brief=other_brief)))


def test_the_brief_swap_splits_critical_rows_from_the_rest(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.2): BASE (correct) minus OTHER per job, on the task-critical rows and on the rest, with the
    pairs, the scripts and the cases by length; the report says when most cases are short. By hand (`brief_pair`): the
    critical differences are c1 +1, 0; c2 +1; c3 -0.5, so four pairs over three scripts with mean 1.5/4 = +0.375; the
    rest are c1 -0.5, c2 +0.5, c3 0: three pairs, mean +0.000. Two cases are short and one medium, so short is the
    majority. With all three cases long there is no such note. Regression caught: critical and other rows pooled into
    one number, a pair of rows from different questions, or a note that is always printed."""
    base, swapped = brief_pair(tmp_path)
    md = report(capsys, "--brief", base, swapped)
    assert "### correct minus swapped: " + SOL_NAME in md
    critical = cells(next(x for x in md.splitlines() if x.startswith("| amr | critical")))
    rest = cells(next(x for x in md.splitlines() if x.startswith("| amr | the rest")))
    assert [critical[2], critical[3], critical[4], critical[6]] == ["4", "3", "+0.375", "short 2, medium 1"]
    assert [rest[2], rest[3], rest[4], rest[6]] == ["3", "3", "+0.000", "short 2, medium 1"]
    assert all(re.fullmatch(r"\[[-+]\d\.\d{3}, [-+]\d\.\d{3}\]", x[5]) for x in (critical, rest))
    assert "Most of these cases are short" in md
    (tmp_path / "long").mkdir()
    base, neutral = brief_pair(tmp_path / "long", "neutral", "long")
    md = report(capsys, "--brief", base, neutral)
    assert "### correct minus neutral: " + SOL_NAME in md and "Most of these cases are short" not in md


def test_the_release_count_from_a_plan_alone(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.4 and its fallback): with no pilot rows each job takes the plan's sigma and rho and its
    own `questions_per_script`, and the count rule and the budgets give K, the predicted half-width and the totals. By
    hand (sigma 0.8, rho 0.05, h 0.10; a hour of render a script each):
    - job a, m 20: 245.85 x (1 + 19 x 0.05)/20 = 245.85 x 0.0975 = 23.97, K_target 24; job b, m 30: 245.85 x
      (1 + 29 x 0.05)/30 = 245.85 x 0.08167 = 20.08, K_target 21; 45 scripts is 45 h against 40;
    - f = 19/21 = 0.905 is the largest that fits: a gets floor(24 x 19/21) = 21 and b 19, 40 h exactly (the next step,
      f 22/24, gives 22 + 19 = 41 h);
    - half-widths 1.5679712 sqrt(0.0975/21) = 0.107 and 1.5679712 sqrt(0.08167/19) = 0.103;
    - frames 21 x 1000 = 21,000 and 19 x 500 = 9,500, GPU-hours 21.0 and 19.0, points 21 x 1.0 = 21.0 and 19 x 0.5 =
      9.5.
    Regression caught: a count not cut to the budget, the render hours and the quota not both checked, or a half-width
    reported at the target instead of at K."""
    md = report(capsys, "--release", plan_file(tmp_path))
    lines = md.splitlines()
    assert "| a | 0.800 (fallback) | 20.0 | 24 | 21 | 0.107 | 21,000 | 21.0 | 21.0 |" in lines
    assert "| b | 0.800 (fallback) | 30.0 | 21 | 19 | 0.103 | 9,500 | 19.0 | 9.5 |" in lines
    assert "| total |  |  | 45 | 40 |  | 30,500 | 40.0 | 30.5 |" in lines
    assert "f = 0.905; render 40.0 of 40 GPU-hours, quota 30.5 of 100 points." in lines
    assert "- rho 0.050: the plan's fallback: no agent-job pair has enough scripts" in lines
    md = report(capsys, "--release", plan_file(tmp_path, render_hours=5))
    assert "*** OVER BUDGET" in md and "| total |  |  | 45 | 12 |" in md


def test_the_release_count_takes_the_largest_sigma_and_rho_of_the_pilot(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.4): sigma is the larger of the agents' SDs, rho the largest Kish ICC over the agent-job
    pairs with 10 scripts, m the questions of one script, and a job without pilot rows takes the plan's fallback sigma.
    By hand (the two agents of `test_the_design_effect_and_rho_need_ten_scripts`: sigma 0.506 and 0.439, rho 0.028 and
    0.750; ten short cases of four questions, so m = 4 + 4 + 4 = 12 as a missing length takes the short mean):
    - vacuum: (1.959964 x 0.50637 / 0.1)^2 = 98.50, times (1 + 11 x 0.75)/12 = 0.7708 is 75.93, K_target 76, with the
      half-width at 76 of 0.09995 = 0.100;
    - amr has no pilot rows: sigma 0.95, m 8 from the plan, rho 0.75: 346.69 x (1 + 7 x 0.75)/8 = 346.69 x 0.78125 =
      270.85, K_target 271.
    Regression caught: the smaller sigma or rho taken, one agent's numbers used for both, or a fallback sigma applied to
    a job that has rows."""
    both = [save(tmp_path, "sol", pilot_doc(SOL, SOL_SCORES)), save(tmp_path, "luna", pilot_doc(LUNA, LUNA_SCORES))]
    plan = plan_file(tmp_path, render_hours=1e6, points=1e6, fallback={"sigma": 0.95, "rho": 0.1}, jobs={
        "vacuum": {"frames_per_script": 100, "seconds_per_frame": 1, "points_per_script": 1},
        "amr": {"frames_per_script": 50, "seconds_per_frame": 1, "points_per_script": 1, "questions_per_script": 8}})
    lines = report(capsys, "--runs", *both, "--release", plan).splitlines()
    assert "- rho 0.750: the largest of 2 agent-job pairs with 10 or more scripts, at least 0.05" in lines
    assert any(x.startswith("| vacuum | 0.506 (pilot) | 12.0 | 76 | 76 | 0.100 |") for x in lines)
    assert any(x.startswith("| amr | 0.950 (fallback) | 8.0 | 271 | 271 | 0.100 |") for x in lines)


def test_rho_is_floored_and_a_thin_length_takes_the_short_mean(tmp_path, capsys):
    """Contract (FD-M6-M7 §2.4): rho is never below 0.05, and a length with fewer than 3 dev cases takes the mean
    questions of a short case. By hand: the first agent alone has rho 0.028, floored to 0.05, so K_target =
    98.50 x (1 + 11 x 0.05)/12 = 98.50 x 0.12917 = 12.72, 13, with the half-width 0.099 at K 13. Eight cases of eight
    scripts: three short of 2, 4 and 6 questions (mean 4), two medium of 10 (fewer than 3: takes 4, not 10), three long
    of 6, 8 and 10 (mean 8): m = 4 + 4 + 8 = 16. Regression caught: a rho below the floor, or a medium length of two
    cases that dominates m."""
    plan = plan_file(tmp_path, render_hours=1e6, points=1e6, jobs={
        "vacuum": {"frames_per_script": 100, "seconds_per_frame": 1, "points_per_script": 1}})
    lines = report(capsys, "--runs", save(tmp_path, "sol", pilot_doc(SOL, SOL_SCORES)), "--release", plan).splitlines()
    assert "- rho 0.050: the largest of 1 agent-job pair with 10 or more scripts, at least 0.05" in lines
    assert any(x.startswith("| vacuum | 0.506 (pilot) | 12.0 | 13 | 13 | 0.099 |") for x in lines)
    sizes = [("short", 2), ("short", 4), ("short", 6), ("medium", 10), ("medium", 10), ("long", 6), ("long", 8),
             ("long", 10)]
    cases = [case(f"c{i}", f"s{i}", "vacuum", length) for i, (length, _) in enumerate(sizes)]
    rows = [row(f"c{i}", f"s{i}", f"q{j}", j % 2, job="vacuum", length=length) for i, (length, n) in enumerate(sizes)
            for j in range(n)]
    lines = report(capsys, "--runs", save(tmp_path, "thin", doc(SOL, cases, rows)), "--release", plan).splitlines()
    assert cells(next(x for x in lines if x.startswith("| vacuum |")))[2] == "16.0"


def test_the_report_is_ascii_deterministic_and_written_to_out(tmp_path, capsys):
    """Contract (FD-M6-M7 §3, lane B): the report is ASCII markdown that is a function of the documents alone, printed
    and, with `--out`, written to the file byte for byte, every section together. Regression caught: a non-ASCII
    character that breaks a Windows console, a bootstrap that is not seeded (two runs differ), or an `--out` that
    differs from what is printed."""
    base, swapped = brief_pair(tmp_path)
    leak = leak_pair(tmp_path, "v", "vacuum", "apartment_a", VACUUM)
    argv = ["--runs", save(tmp_path, "sol", pilot_doc(SOL, SOL_SCORES)), "--leak", *leak, "--brief", base, swapped,
            "--release", plan_file(tmp_path)]
    first = report(capsys, *argv, "--out", tmp_path / "pilot.md")
    assert first.isascii() and first == report(capsys, *argv)
    assert (tmp_path / "pilot.md").read_text(encoding="utf-8") == first
    assert [x for x in first.splitlines() if x.startswith("## ")] == [
        "## Pilot rows", "## D2: does the during answer leak into the end answers", "## The brief swap",
        "## The release's script count"]


def test_the_report_refuses_what_it_cannot_report(tmp_path, capsys):
    """Contract (FD-M6-M7 §3, lane B): exit 2, saying why, with nothing printed or written, for: no option; a `--runs`
    document with during off, or with another brief; a document that is not a results document; two `--runs`
    scored on different compiles (a case they share holds no qid in common); the same case twice; a leak
    pair that differs in `during` and another lever (compare's refusal), in another lever alone, or is given off
    first, or is of two agents; a brief pair out of order or differing in `during`; and a plan that lacks a number or
    gives a job with no pilot rows no `questions_per_script`. Regression caught: a pilot row that mixes arms (D2's
    cause measured as the effect), a pooled leak over two agents, or a traceback where a refusal belongs."""
    records = [case("c1", "s1", "vacuum", "short")]
    rows_ = [row("c1", "s1", "e1", 1, job="vacuum")]
    on = save(tmp_path, "on", doc(SOL, records, rows_))
    off = save(tmp_path, "off", doc(SOL, records, rows_, during=False))
    swapped = save(tmp_path, "swapped", doc(SOL, records, rows_, brief="swapped"))
    detections = save(tmp_path, "detections", doc(SOL, records, rows_, during=False, channel="detections"))
    luna_on = save(tmp_path, "luna-on", doc(LUNA, records, rows_))
    luna_off = save(tmp_path, "luna-off", doc(LUNA, records, rows_, during=False))
    broken = save(tmp_path, "broken", {"system": "codex:gpt-6.1-sol"})
    recompiled = save(tmp_path, "recompiled", doc(LUNA, records, [row("c1", "s1", "e9", 1, job="vacuum")]))
    bad_plan = plan_file(tmp_path, "bad.json", target=0)
    thin_plan = plan_file(tmp_path, "thin.json", jobs={"x": dict.fromkeys(("frames_per_script", "seconds_per_frame",
                                                                "points_per_script"), 1)})
    out = tmp_path / "never.md"
    for argv, message in (
            ([], "give at least one of"),
            (["--runs", off], "during off"),
            (["--runs", swapped], "swapped brief"),
            (["--runs", broken], "is not a results document"),
            (["--runs", on, recompiled], "were scored on different compiles"),
            (["--runs", on, on], "both hold case c1"),
            (["--leak", on, detections], "differ in more than one lever (channel, during)"),
            (["--leak", on, swapped], "differ in brief, not in during alone"),
            (["--leak", off, on], "a --leak pair is ON then OFF"),
            (["--leak", on, off, "--leak", luna_on, luna_off], "different agents"),
            (["--brief", swapped, on], "a --brief pair is BASE (correct) then OTHER"),
            (["--brief", on, off], "differ in during, not in brief alone"),
            (["--release", bad_plan], "'target' must be a positive number"),
            (["--release", thin_plan], "job x has no pilot rows and the plan gives no 'questions_per_script'")):
        assert run(*argv, "--out", out) == 2, argv
        shown = capsys.readouterr()
        assert shown.out == "" and message in shown.err, (argv, shown.err)
    assert not out.exists()
    assert run("--release", plan_file(tmp_path), "--out", tmp_path / "no-such-folder" / "x.md") == 2   # unwritable
    shown = capsys.readouterr()
    assert shown.out == "" and "cannot write" in shown.err


def test_help_says_what_the_report_covers(capsys):
    """Contract (FD-M6-M7 §3, lane B): `stbench pilot report --help` names each option and what it reports, so the
    command is usable without this module's source. Regression caught: a help that lists flags alone."""
    with pytest.raises(SystemExit) as stop:
        main(["pilot", "report", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for word in ("--runs", "--leak", "--brief", "--release", "--out", "design effect", "D2", "swapped or neutral"):
        assert word in text, word


def test_the_pilot_headline_leaves_out_the_pairs_the_results_record_as_thin(tmp_path, capsys):
    """Contract (review of lane A1): the pilot report's headline is the one `results.json` holds: it passes the union of
    the documents' `layouts[*].thin` pairs to `aggregate` and prints the `columns`. By hand, two scripts, the vacuum:
    `f_a` scores [1, 0] and [1, 0] (1/2) and `f_b` [1, 1] twice (1), all `memory`; with the pair (vacuum, f_b) thin the
    `memory` column is f_a alone, 1/2 over 4 questions, and the headline +0.500; without the pair it would be
    (1/2 + 1)/2 = 0.750. Regression caught: a report that disagrees with the results it is made from because its own
    aggregate counts the diagnostic pairs."""
    rows = [row(f"c-{sid}", sid, f"{sid}-{i}", x, job="vacuum", family=f) for sid in ("s1", "s2")
            for i, (f, x) in enumerate((("f_a", 1), ("f_a", 0), ("f_b", 1), ("f_b", 1)))]
    cases = [case(f"c-{sid}", sid, "vacuum", "short", tokens=(1000, 100, 100)) for sid in ("s1", "s2")]
    thin = frozenset({("vacuum", "f_b")})
    document = {**doc(SOL, cases, rows), "layouts": {"apartment_a": aggregate(rows, n_boot=20, thin=thin)}}
    md = report(capsys, "--runs", save(tmp_path, "thin", document))
    lines = section(md, f"### {SOL_NAME}: vacuum (apartment_a)")
    assert any(line.startswith("- headline: +0.500 ") for line in lines)
    assert any(line.startswith("- columns: memory +0.500 ") and line.endswith(" over 4") for line in lines)
    assert aggregate(rows, n_boot=20)["headline"]["mean"] == 0.75
