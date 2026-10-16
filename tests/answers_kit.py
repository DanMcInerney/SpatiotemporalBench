"""Helpers for the tests of the answer side across tasks: a window of several tasks to compile and to answer.

`compiled_case(dataset, cid, script, job, a, b)` writes tasks a..b of a synthetic script as a case of the fixed id `cid`
(its length by task count, `case.LENGTHS`) and compiles it with `job` (`job_registry.get` patched), as
`baseline_kit.compiled_case` does for `vacuum/1`'s one task: every family lane's oracle test builds on it, over
`family_kit`'s scripts. `three_tasks` is a hand case of the other kind, whose questions and gold rows are written by
hand (as `stream_kit.case_on` does), so what a table or `path_log` answers about a named task is checked against values
worked out here and not against the compiler. `scored` is its answers' penalised score.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from run_kit import golds
from space_kit import BAG_SIZE, SpaceScript, vacuum_run
from stbench import case as case_mod
from stbench import job_registry, questions, score
from stbench.case import CasePaths
from stbench.job_registry import JobSpec
from stream_kit import case_on, end_q, gold_row

BAG, BOX = "bag", "box"
FINISHED = "when you finished sweeping"   # `space_kit.JOB`'s wording: the last `cleaning` step of a task


def length_of(tasks: int) -> str:
    """The length `case.LENGTHS` gives a window of `tasks` tasks."""
    return next(name for name, (lo, hi) in case_mod.LENGTHS.items() if lo <= tasks <= hi)


def compiled_case(dataset: Path, cid: str, script: SpaceScript, job: JobSpec, a: int, b: int) -> CasePaths:
    """Tasks a..b of `script` as a case of id `cid` and its length, its questions and gold compiled with `job`."""
    with mock.patch.object(case_mod, "claim_id", lambda dataset, entry: cid):
        case = script.write_case(dataset, a, b, length_of(b - a + 1))
    with mock.patch.object(job_registry, "get", lambda job_id: job):
        questions.write(case)
    return case


def three_task_script(seed: int = 1) -> SpaceScript:
    """`space_kit.vacuum_run` of three tasks (28 keyframes each, 7200 s apart; no keyframe has a render, nothing here
    asks about space) with two things, each recognisable (1,500 px) in the keyframes named:
    - a red bag at (2, 1) seen at keyframe 10 (task 0), moved in the gap to (5, 1) and seen at 38 (task 1), moved again
      to (1, 2) and seen at 66 (task 2);
    - a blue box at (3.5, 2) seen at keyframe 40 (task 1) only."""
    s = vacuum_run(3, seed=seed, depth_ks=set())
    s.item("E0001", BAG, {"color": "red"}, [(0, 7000, (2.0, 1.0, 0.0)), (7000, 14100, (5.0, 1.0, 0.0)),
                                            (14100, None, (1.0, 2.0, 0.0))], size=BAG_SIZE)
    s.see([10, 38, 66], "E0001", 1500)
    s.item("E0002", BOX, {"color": "blue"}, [(0, None, (3.5, 2.0, 0.0))])
    s.see(40, "E0002", 1500)
    return s


def where(n: int, kind: str, color: str, task: int | None) -> dict:
    """An `object_where` question about `FINISHED` in the case task `task` (None: the window's only task)."""
    return end_q(n, slots={"kind": kind, "attributes": {"color": color}, "event": FINISHED, "task": task},
                 needs=("see", "pose", "status"))


# What the hand case asks and the answer each has by hand: the latest sighting at or before the last `cleaning` step of
# the named task (keyframe 18, 46 and 74 of the case), of any task: the bag at (2, 1), (5, 1) and (1, 2); the box
# unseen in task 0, at (3.5, 2) from task 1 on.
ASKED = [(1, BAG, "red", 0, (2.0, 1.0)), (2, BAG, "red", 1, (5.0, 1.0)), (3, BAG, "red", 2, (1.0, 2.0)),
         (4, BOX, "blue", 1, (3.5, 2.0))]


def three_tasks(dataset: Path) -> tuple[CasePaths, list[dict], list[dict]]:
    """The hand 3-task case (a medium window of `three_task_script`) with `ASKED`'s questions and the gold rows worked
    out above, as `(case, questions, golds)`."""
    qs = [where(n, kind, color, task) for n, kind, color, task, _ in ASKED]
    case = case_on(dataset, three_task_script(), qs, a=0, b=2, length="medium")
    rows = [gold_row(q, gold={"type": "position", "xy": list(xy)}) for q, (*_, xy) in zip(qs, ASKED, strict=True)]
    golds(case, rows=rows)
    return case, qs, rows


def scored(rows: list[dict], answers: dict[str, dict]) -> dict[str, float]:
    """The penalised score of each answer against the gold row of its qid."""
    return {g["qid"]: score.penalised(answers[g["qid"]], g) for g in rows}

