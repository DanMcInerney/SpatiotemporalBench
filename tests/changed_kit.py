"""What the `changed` tests share, on top of `family_kit`: a living room toured in every task, things that stand, move,
vanish and appear in the gaps between tasks, and a job that asks `changed` and `object_where`.

The world is `family_kit.multi_run`'s: tasks of 28 keyframes, 7200 s apart, whose `cleaning` leg is a look-around at
the living room's centre (6, 1.5) (`HUB`), heading 45 degrees more at every keyframe. Four of those keyframes are
rendered, the ones that face east, north, west and south (`VIEWS`: offset 7, 9, 11 and 13 in a task; ray-cast depth over
the floor, as `space_kit` renders it); no other keyframe has depth, so the kitchen is never observed and never asked
about, and the living room is observed to about 73% by the four views, wedges at the diagonals left out (the places
the tests use are in the wedges the views cover).

A thing's fate is a list of places, one a task (`put`): it is placed from the gap before its task begins (`gap_start`,
100 s before it), seen (`px` 1,500, `space_kit.BAG_SIZE` unless told) at the keyframe that faces it and, seen standing
on the floor, drawn into the depth of that task's four views as a box. A thing up on a table is seen but not drawn,
nor is one no frame shows, and no table is drawn: the kit models what the floor shows. A person (`person`) is seen and
not drawn. `hand_inputs` swaps the renders for hand free-space states (no render cost) where a test needs only how much
of the room was observed, not where.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import family_kit as fk
from space_kit import BAG_SIZE, SpaceScript, box
from space_kit import JOB as SPACE_JOB
from space_kit import record as space_record
from stbench import case as case_mod
from stbench.job_registry import JobSpec
from stbench.questions_world import Inputs, World, cached_states
from synth import CENTRE_XY, RECOG, TASK_GAP_S

ROOM = "living room"  # the room the tasks look around: the second of `synth.RECTS`, x 4..8, y 0..3
HUB = CENTRE_XY[1]  # where the robot stands through the look-around: (6, 1.5)
VIEWS = {0: 7, 90: 9, 180: 11, 270: 13}  # heading (degrees) -> the offset in a task of the rendered keyframe facing it
TASK_FRAMES = fk.TASK_FRAMES
ROOM_CELLS = [(u, v) for u in range(40, 80) for v in range(30)]  # the living room's 1,200 inside fine cells
JOB: JobSpec = replace(SPACE_JOB, decisions={"before_doorway": (), "room_done": ()}, ends=("object_where", "changed"))
Spot = tuple[float, ...] | None  # (x, y) on the floor, (x, y, z) up on a table, None: not there
SHOES = (0.3, 0.2, 0.12)


def length_of(tasks: int) -> str:
    """The length `case.LENGTHS` gives a window of `tasks` tasks."""
    return next(name for name, (lo, hi) in case_mod.LENGTHS.items() if lo <= tasks <= hi)


def gap_start(task: int) -> float:
    """When task `task`'s world begins: 0 for the first, else 100 s before the task starts (inside the gap)."""
    return 0.0 if task == 0 else task * TASK_GAP_S - 100.0


def view_of(xy: Sequence[float]) -> int:
    """The offset in a task of the rendered keyframe that faces the place (the nearest of the four headings)."""
    bearing = math.degrees(math.atan2(xy[1] - HUB[1], xy[0] - HUB[0])) % 360
    return VIEWS[min(VIEWS, key=lambda yaw: min(abs(bearing - yaw), 360 - abs(bearing - yaw)))]


def script(n_tasks: int, *, seed: int = 1) -> SpaceScript:
    """`n_tasks` tasks of the vacuum with the four views rendered in each and nothing drawn in them."""
    views = {t * TASK_FRAMES + off for t in range(n_tasks) for off in VIEWS.values()}
    return fk.multi_run(n_tasks, seed=seed, poses={}, yaws={}, boxes=[], frame_boxes={}, depth_ks=views)


def put(s: SpaceScript, eid: str, kind: str, attrs: dict, stands: Sequence[Spot], *, size=BAG_SIZE,
        mobility: str = "carried", seen: Sequence[int] | None = None, px: int = RECOG) -> str:
    """The thing with the fate `stands` (its place in task 0, 1, ...; the last holds for the tasks after it). It is
    seen, in the tasks `seen` (default: each task it stands in), at the keyframe that faces its place; where it is seen
    standing on the floor it is drawn as a box of its size into the four views of the task."""
    stands = ([*stands, *[stands[-1]] * len(s.tasks)])[:len(s.tasks)]
    where = []
    for t, spot in enumerate(stands):
        if spot is not None:
            end = gap_start(t + 1) if t + 1 < len(stands) else None
            where.append((gap_start(t), end, (spot[0], spot[1], spot[2] if len(spot) > 2 else 0.0)))
    s.item(eid, kind, attrs, where, size=size, mobility=mobility)
    for t in [t for t, spot in enumerate(stands) if spot is not None] if seen is None else seen:
        spot = stands[t]
        s.see(t * TASK_FRAMES + view_of(spot), eid, px)
        if len(spot) == 2 or spot[2] == 0.0:
            for off in VIEWS.values():
                s.frame_boxes.setdefault(t * TASK_FRAMES + off, []).append(
                    box(spot[0] - size[0] / 2, spot[0] + size[0] / 2, spot[1] - size[1] / 2, spot[1] + size[1] / 2,
                        0.0, size[2]))
    return eid


def person(s: SpaceScript, eid: str, walks: Mapping[int, Sequence[tuple[float, float]]]) -> str:
    """A resident seen at the places `walks[task]` in each task, at the keyframes that face them (so the places of one
    task face different views), standing at each until the next."""
    steps = sorted((t * TASK_FRAMES + view_of(xy), xy) for t, route in walks.items() if t < len(s.tasks)
                   for xy in route)
    if len({k for k, _ in steps}) != len(steps):
        raise ValueError(f"two of {eid}'s places face the same keyframe: {steps}")
    where = [(s.t_of(k), s.t_of(steps[n + 1][0]) if n + 1 < len(steps) else None, (*xy, 0.0))
             for n, (k, xy) in enumerate(steps)]
    s.item(eid, "person", {}, where, mobility="mover", size=(0.5, 0.5, 1.7))
    s.see([k for k, _ in steps], eid, RECOG)
    return eid


def hand_inputs(s: SpaceScript, observed: Mapping[int, float], job: JobSpec = JOB, state: str = "free") -> Inputs:
    """`family_kit.inputs` with hand free-space states: in task t the first `observed[t]` of the living room's 1,200
    inside fine cells are in `state` at every keyframe of the task; a task not in `observed` observes nothing."""
    def states_of(ks):
        return {k: dict.fromkeys(ROOM_CELLS[:round(observed.get(k // TASK_FRAMES, 0.0) * len(ROOM_CELLS))], state)
                for k in ks}

    return replace(fk.inputs(s, job), states_of=cached_states(states_of))


def world(s: SpaceScript, a: int = 0, b: int | None = None, *, cid: str = "c0", job: JobSpec = JOB,
          inp: Inputs | None = None) -> World:
    """The window of tasks a..b of the script (b defaults to its last task) as the compiler's `World`, over `inp`
    (default: the script's renders) and `job`."""
    b = len(s.tasks) - 1 if b is None else b
    return World(replace(inp or fk.inputs(s, job), job=job),
                 space_record(s, a=a, b=b, cid=cid, length=length_of(b - a + 1)))


def by_hand(n_tasks: int = 3, *, skip: Sequence[str] = (), extra: Callable[[SpaceScript], None] | None = None
            ) -> SpaceScript:
    """The world of the by-hand gold, in the living room (places in metres), three tasks or more (the third state of
    each thing holds for the tasks after), without the things named in `skip` and with whatever `extra(script)` adds.
    Between tasks 0 and 2:
    - E0001 a red bag at (7.4, 0.8) in tasks 0 and 1, at (4.6, 1.2) from task 2: moved (an event at (4.6, 1.2));
    - E0002 a blue box (0.3 m) at (6.2, 2.4) in tasks 0 and 1, gone in task 2: removed (an event at (6.2, 2.4));
    - E0003 a yellow toy (0.2 m) appears at (7.8, 2.4) in task 2: added, next to a place the resident was seen;
    - E0004 a green bag on a table at (4.4, 2.6) up to task 1, on the floor at (4.6, 2.0) in task 2: added there;
    - E0005 a red mug on a table, moved 0.6 m along it: nothing;
    - E0006 and E0007 black shoes, the second moved from (6.6, 0.4) to (7.8, 0.4): identical, don't-care places;
    - P0001 a resident, seen at (5.4, 1.6) in task 0 and at (7.4, 2.4) and (6.2, 0.9) in task 2: don't-care places."""
    s = script(n_tasks)

    def add(eid: str, *args, **kwargs) -> None:
        if eid not in skip:
            put(s, eid, *args, **kwargs)

    add("E0001", "bag", {"color": "red"}, [(7.4, 0.8), (7.4, 0.8), (4.6, 1.2)])
    add("E0002", "box", {"color": "blue"}, [(6.2, 2.4), (6.2, 2.4), None], size=(0.3, 0.3, 0.3))
    add("E0003", "toy", {"color": "yellow"}, [None, None, (7.8, 2.4)], size=(0.2, 0.2, 0.1))
    add("E0004", "bag", {"color": "green"}, [(4.4, 2.6, 0.75), (4.4, 2.6, 0.75), (4.6, 2.0)])
    add("E0005", "mug", {"color": "red"}, [(6.9, 2.6, 0.75), (6.9, 2.6, 0.75), (7.5, 2.6, 0.75)],
        size=(0.08, 0.08, 0.1))
    add("E0006", "shoes", {"color": "black"}, [(5.8, 0.4)], size=SHOES)
    add("E0007", "shoes", {"color": "black"}, [(6.6, 0.4), (6.6, 0.4), (7.8, 0.4)], size=SHOES)
    if "P0001" not in skip:
        person(s, "P0001", {0: [(5.4, 1.6)], 2: [(7.4, 2.4), (6.2, 0.9)]})
    if extra is not None:
        extra(s)
    return s
