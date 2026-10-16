"""What the tests of the robot's own-situation families (`questions_self`) share, on top of `family_kit`: a task long
enough to hold a step 60 s back, and a job that asks `where_was_i` at decision points of a kind of its own.

`LEGS` is one task of 96 keyframes, 1.5 s apart (144 s): undocking 0-2, transit 3-6, cleaning 7-86, transit 87-90,
docking 91-93, docked 94-95. `self_script(n)` is `n` such tasks, 7200 s apart, with no depth (`depth_ks` empty: no
family here reads a free-space cell), the base of step `o` of task `t` at (0.5 + 0.05 o, 0.5 + 0.1 t) so that no two
steps of a task share a position (an answer from the wrong step is told from the right one), and a decision point of
kind `mid_task` at each offset of `DECISIONS`. At offsets 27, 40 and 70 (t = 40.5, 60 and 105 s into the task) the
stream is cut after each (and after every 8 steps, so the look windows of a one-task window are 0-7, 8-15, 16-23, 24-27,
28-35, 36-40, 41-48, 49-56, 57-64, 65-70 and on). `JOB` asks `where_was_i` at `mid_task`; no registered job does until
the coordinator maps `vacuum/2`. `world` is a window of the script as the compiler's `World`, with `family_kit`'s hand
map."""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from answers_kit import length_of
from space_kit import JOB as SPACE_JOB
from space_kit import SpaceScript
from space_kit import record as space_record
from stbench.questions_world import Inputs, World

LEGS = [("kitchen", 3, "undocking"), ("kitchen", 4, "transit"), ("living room", 80, "cleaning"),
        ("living room", 4, "transit"), ("kitchen", 3, "docking"), ("kitchen", 2, "docked")]
TASK_FRAMES = sum(leg[1] for leg in LEGS)  # 96
KIND = "mid_task"
DECISIONS = (27, 40, 70)  # offsets in a task
JOB = replace(SPACE_JOB, decisions={KIND: ("where_was_i",)})


def self_script(n_tasks: int = 1, decisions: tuple[int, ...] = DECISIONS, **fields) -> SpaceScript:
    """`n_tasks` tasks of `LEGS`, a `mid_task` decision at each offset of `decisions` in each; `fields` replace
    `SpaceScript`'s (`poses`, `yaws`, `boxes`, `seed` ...)."""
    drive = {t * TASK_FRAMES + o: (round(0.5 + 0.05 * o, 3), round(0.5 + 0.1 * t, 3))
             for t in range(n_tasks) for o in range(TASK_FRAMES)}
    setup = {"tasks": [list(LEGS) for _ in range(n_tasks)], "depth_ks": set(), "poses": drive,
             "decisions": [(t * TASK_FRAMES + k, KIND) for t in range(n_tasks) for k in decisions]}
    return SpaceScript(**{**setup, **fields})


def world(script: SpaceScript, cid: str = "c0", a: int = 0, b: int | None = None, inp: Inputs | None = None) -> World:
    """The window of tasks a..b of the script (b defaults to a; its length by task count) for case id `cid`. Pass `inp`
    (`fk.inputs(script, JOB)`) to build many windows of one script without rebuilding its timeline."""
    b = a if b is None else b
    return World(inp or fk.inputs(script, JOB), space_record(script, a=a, b=b, cid=cid, length=length_of(b - a + 1)))
