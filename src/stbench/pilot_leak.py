"""D2's leak check, in its two readings (`stbench pilot report --leak ON OFF`, FD-0.11 §3.6).

d = on - off of the penalised score (`score_gate`: no catastrophe weight enters it) over the end questions, paired by
(case, question), with the script bootstrap's 95% interval.

- **The leak** is read over the pairs whose gold is answerable (`slice` answerable): did asking during the job change
  what the agent remembers of a thing it saw? `power.leak_verdict` (margin 0.05) reads that pooled d alone, and the
  jobs' answerable d name the jobs whose during families go if it fails. In 0.9.0 these 171 pairs gave d +0.001
  [-0.065, +0.066].
- **The abstention shift** is read over the pairs whose gold is `unknown`: the right answer is "never seen", and asking
  during the job changed how willing the agent was to say so (0.9.0: +0.364 over 22 pairs, two vacuum pairs among them).
  That is calibration and not recall, so it has a d and an interval and no verdict. The row of all pairs is the 0.9.0
  report's reading, kept so the two can be set side by side.
- **By family** shows both readings for each (job, family), so a family whose own d rises (the vacuum's cords, which its
  during `tangle_ahead` questions also cover) is seen where it is.

This module reads result rows only, so it knows nothing of the truth.
"""

from __future__ import annotations

from . import power
from .eval_report import interval, signed
from .score_stats import GATE, paired_difference

MARGIN = 0.05   # D2's margin
ANSWERABLE, UNKNOWN = "answerable", "unknown"


class Bad(ValueError):
    """The pairs cannot be read."""


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _layouts(rows: list[dict], job: str) -> str:
    return "+".join(sorted({r["layout"] for r in rows if r["job"] == job}))


def _d(on: list[dict], off: list[dict], **where) -> dict | None:
    """The paired difference over the rows `where` keeps, None when no pair is left."""
    try:
        return paired_difference(on, off, where=where, key=GATE)
    except ValueError:
        return None


def _cells(d: dict | None) -> list[str]:
    """`pairs, scripts, d, 95% CI` of a difference (dashes for none)."""
    if d is None:
        return ["-", "-", "n/a", "n/a"]
    return [str(d["n"]), str(d["scripts"]), signed(d["mean"]), interval(d["ci"])]


def _jobs(on: list[dict]) -> list[str]:
    return sorted({r["job"] for r in on})


def section(agent: str, on: list[dict], off: list[dict]) -> list[str]:
    """The markdown of the check from the end rows of the ON and the OFF arm. Bad when the arms share no end question or
    no answerable one: the verdict reads those alone."""
    try:
        paired_difference(on, off, key=GATE)
    except ValueError as error:
        raise Bad(f"the --leak pairs: {error}") from None
    if (total := _d(on, off, slice=ANSWERABLE)) is None:
        raise Bad("the --leak pairs hold no end question with an answerable gold, which the verdict reads")
    rows = []
    for job in _jobs(on):
        d = _d(on, off, job=job, slice=ANSWERABLE)
        label = f"{job} ({_layouts(on, job)})"
        rows.append([label, *_cells(d), "n/a" if d is None else power.leak_verdict(d["mean"], d["ci"], MARGIN)])
    verdict = power.leak_verdict(total["mean"], total["ci"], MARGIN)
    rows.append(["pooled", *_cells(total), verdict])
    over = [row[0].split(" ")[0] for row in rows[:-1] if row[5] == "fail"]
    then = {"pass": "the during questions ship.",
            "inconclusive": "the during questions ship; publish d, the interval and n, and state that the check cannot "
                            f"exclude a leak of the upper bound ({signed(total['ci'][1]) if total['ci'] else 'n/a'}).",
            "fail": "drop the during families of every job whose own answerable d is above the margin ("
                    + (", ".join(over) or "none") + ") before the release compile."}[verdict]
    lines = ["## D2: does the during answer leak into the end answers", "",
             f"Agent {agent}. d = the mean of (on - off) of the penalised score (no catastrophe weight) over the end "
             "questions, paired by (case, question), with a 95% bootstrap interval over scripts. The off arm asked no "
             "during question, so during rows are never paired. The verdict reads the pairs whose gold is answerable "
             "(a thing the agent saw): asking during the job could leak into what it recalls of those. The margin is "
             f"{MARGIN}: pass when the upper bound is at or below it, fail when d is above it, else inconclusive. This "
             "is the one pool over layouts the design asks for (D2).", "",
             *_table(["job", "pairs", "scripts", "d", "95% CI", "verdict"], rows),
             f"Pooled verdict: **{verdict}**: {then}", ""]
    return lines + _shift(on, off) + _families(on, off)


def _shift(on: list[dict], off: list[dict]) -> list[str]:
    """The abstention shift: pairs whose gold is `unknown`, per job and pooled, and all pairs for the 0.9.0 reading."""
    rows = [[f"{job} ({_layouts(on, job)})", *_cells(_d(on, off, job=job, slice=UNKNOWN))] for job in _jobs(on)]
    rows += [["pooled", *_cells(_d(on, off, slice=UNKNOWN))],
             ["all pairs (the reading 0.9.0 published)", *_cells(_d(on, off))]]
    return ["### Abstention shift", "",
            "The pairs whose gold is `unknown` (a thing never seen, or none like it here): d is how much more often "
            "the on arm said so, or less often guessed. It is calibration, not recall, and has no verdict.", "",
            *_table(["job", "pairs", "scripts", "d", "95% CI"], rows)]


def _families(on: list[dict], off: list[dict]) -> list[str]:
    """d of each (job, family) over its answerable pairs and over its `unknown` ones."""
    rows = []
    for job in _jobs(on):
        for family in sorted({r["family"] for r in on if r["job"] == job}):
            for gold in (ANSWERABLE, UNKNOWN):
                if (d := _d(on, off, job=job, family=family, slice=gold)) is not None:
                    rows.append([job, family, gold, *_cells(d)])
    return ["### By family", "", *_table(["job", "family", "gold", "pairs", "scripts", "d", "95% CI"], rows)]
