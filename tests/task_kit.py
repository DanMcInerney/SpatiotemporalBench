"""What the `task_things` tests share, on `family_kit` and `space_kit`: a job that asks the family, tasks whose docking
leg runs along a line, a hand map with a doorway and a rack, and a way to put a thing in a task and say how it showed.

A task is `LEGS` (16 keyframes): cleaning at 1-7 (keyframe 0 is the start phase), docking at 8-10, docked at 11-15, in
the kitchen or the living room (`script(room=...)`: the room decides where the robot stands and so which region a
`rooms` anchor holds). The stream is cut every 8 steps and after the task's last, so the look windows are 0-7 and 8-15:
the window's last look window is keyframes 8 to 15, and a thing sighted at 1-7 only is out of it. The docking leg is the
way the robot "drove": at (0, 1), (20, 1) and (40, 1) at keyframes 8, 9 and 10 (`PATH_POSES`), a segment along y = 1
for x from 0 to 40 (task t's keyframes are 16 t later). Things stand 3 m apart along it so that no don't-care circle
(`job.match_m` = 0.5 m, plus a large thing's widening) reaches another thing.

`PATH` is the predicate "within 1 m of the way you docked" (band 0.25 m, so an item's nearest point is within 0.75 m,
a don't-care place within 1.25 m); `put` places a thing of a size, kind and mobility (default a carried thing of the
bag's size), a heading, and the keyframes it is recognised at (1,500 px); `gold` is the family's gold function over a
window of the script.
"""

from __future__ import annotations

import math
from dataclasses import replace

import family_kit as fk
from space_kit import JOB as SPACE_JOB
from space_kit import SpaceScript
from space_kit import record as space_record
from stbench import case as case_mod
from stbench.geometry import quat_z
from stbench.job_registry import JobSpec
from stbench.mapspec import Furniture, MapSpec, Opening, Surface, Wall
from stbench.questions_select import Moment
from stbench.questions_task import TaskCand, task_things
from stbench.questions_world import Inputs, World
from stbench.task_predicate import TaskThings
from stbench.timeline import Entity, Placement
from synth import RECOG

LEGS = [("kitchen", 8, "cleaning"), ("kitchen", 3, "docking"), ("kitchen", 5, "docked")]
TASK_FRAMES = 16
PATH_POSES = {8: (0.0, 1.0), 9: (20.0, 1.0), 10: (40.0, 1.0)}   # the docking leg of task 0, along y = 1
LAST = 15   # a task's last keyframe (task 0)
PATH = TaskThings("within 1 m of the way you docked", "path", frozenset({"docking"}), reach_m=1.0, band_m=0.25,
                  floor=True)
POINT = TaskThings("within 2 m of where you cleaned", "point", frozenset({"cleaning"}), reach_m=2.0, band_m=0.5)
ROOMS = TaskThings("on the floor of the rooms you cleaned", "rooms", frozenset({"cleaning"}), floor=True)
BAG = (0.4, 0.4, 0.3)   # a small thing: widening 0; nearest point 0.2 m from its centre along an axis
BIG = (1.0, 1.0, 0.5)   # a large thing: widening 0.25 m; nearest point 0.5 m from its centre along an axis


def y(d: float, size: float = 0.4) -> float:
    """The y of a thing of side `size` whose nearest point is `d` from the line y = 1."""
    return 1.0 + size / 2 + d


def job(pred: TaskThings | None = PATH) -> JobSpec:
    """`space_kit.JOB` asking `task_things` at the end and nothing during."""
    return replace(SPACE_JOB, decisions={}, ends=("task_things",), task_things=pred)


def script(n_tasks: int = 1, *, room: str = "kitchen", path: dict | None = None, **fields) -> SpaceScript:
    """`n_tasks` tasks of `LEGS` in `room`; the docking leg runs along `path` (default `PATH_POSES`, for each task)."""
    legs = [(room, n, phase) for _, n, phase in LEGS]
    way = PATH_POSES if path is None else path
    poses = {t * TASK_FRAMES + k: xy for t in range(n_tasks) for k, xy in way.items()}
    return SpaceScript(tasks=[list(legs) for _ in range(n_tasks)], decisions=[], depth_ks=set(),
                       **{"poses": poses, **fields})


def put(s: SpaceScript, eid: str, xy: tuple[float, float], size=BAG, *, kind: str = "bag", mobility: str = "carried",
        z: float = 0.0, yaw_deg: float = 0.0, at=(3,), px: int = RECOG, attrs: dict | None = None,
        stands: tuple | None = None) -> str:
    """A thing standing still at `xy` (`z` above the floor, turned `yaw_deg`), recognised at the keyframes `at`; `attrs`
    default to a colour of its own id so that no two things are twins; `stands` replaces the placement with
    ((t0, t1, (x, y, z)), ...) for a thing that moves."""
    s.entities.append(Entity(eid, kind, {"color": eid} if attrs is None else attrs, mobility, tuple(size)))
    where = stands or ((0, None, (*xy, z)),)
    s.placements += [Placement(eid, t0, t1, tuple(pos), quat_z(math.radians(yaw_deg)), None) for t0, t1, pos in where]
    if at:
        s.see(list(at), eid, px)
    return eid


def room_map(s: SpaceScript, *, stock: bool = False) -> MapSpec:
    """The hand map (`family_kit.hand_map`: the kitchen x 0-4 and the living room x 4-8, both y 0-3) with a wall
    between them at x = 4 and a door in it at y = 1.5, 0.9 m wide, swinging left: its way (`zone_shapes.doorway_way`,
    by hand) is x 4.0 to 4.35, y 1.15 to 1.85. With `stock`, one pallet rack (1.1 deep, 2.7 long, 2.6 high, one bay of
    two levels, no surface) centred at (6, 14) with a free-floor mat: its two stock pallets stand at (6, 13.325) and
    (6, 14.675); the mat is a surface at floor height on a rug."""
    base = fk.hand_map(s)
    wall = Wall("W1", (4.0, 0.0), (4.0, 3.0), 0.1, 2.5, False)
    door = Opening("D1", "door", "W1", 1.5, 0.9, 2.0, "a", "left", True, False, ("R1", "R2"), (4.0, 1.5), math.pi / 2,
                   "R1")
    furniture: tuple[Furniture, ...] = ()
    if stock:
        rack = Furniture("F1", "rack", "pallet_rack", "fixed", (6.0, 14.0), (1.1, 2.7, 2.6), 0.0, "grey", True, ())
        furniture = (rack,
                     Furniture("F2", "mat", "mat", "fixed", (2.0, 5.0), (1.0, 1.0, 0.02), 0.0, "grey", False,
                               (Surface("S1", "floor mat", (0.0, 0.0, 1.0, 1.0), 0.01),)))
    return replace(base, walls=(wall,), openings=(door,), furniture=furniture)


def world(s: SpaceScript, pred: TaskThings | None = PATH, *, a: int = 0, b: int | None = None, m: MapSpec | None = None,
          cid: str = "c0") -> World:
    """The window of tasks a..b of the script as the compiler's `World`, the job asking `pred`, over the hand map `m`
    (default `family_kit.hand_map`)."""
    b = a if b is None else b
    inp: Inputs = fk.inputs(s, job(pred))
    length = next(name for name, (lo, hi) in case_mod.LENGTHS.items() if lo <= b - a + 1 <= hi)
    return World(replace(inp, map=m or inp.map), space_record(s, a=a, b=b, cid=cid, length=length))


def gold(s: SpaceScript, pred: TaskThings | None = PATH, **kw) -> list[Moment]:
    """`task_things`' moments over the window."""
    return task_things(world(s, pred, **kw))


def cand(s: SpaceScript, pred: TaskThings | None = PATH, **kw) -> TaskCand:
    """The one candidate of a one-task window."""
    (moment,) = gold(s, pred, **kw)
    (c,) = moment.answerable
    return c


def places(c: TaskCand) -> tuple[list, list]:
    """(the items' places, the don't-care places) of a candidate's gold, as written."""
    return [i["xy"] for i in c.gold["items"]], [(r["xy"], r["radius"]) for r in c.gold.get("dont_care", [])]
