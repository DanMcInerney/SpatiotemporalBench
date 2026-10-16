"""What the `ego_now` and `ego_mem` tests share: the vacuum's one-task script (`space_kit.vacuum_run`) with the robot
put at a look point, and things placed by hand in the camera's frame, so every point and offset is worked out here.

The task has 28 keyframes cut into the look windows 0-5, 6-13, 14-18, 19-21 and 22-27; the look point is `K` = 13, the
end of the second. There the base is at (3.0, 1.0) heading north (90 degrees), so the camera, mounted 0.13 m ahead and
0.08 m up, is at (3.0, 1.13, 0.08): a point (x, y) is `y - 1.13` metres ahead of it and `3.0 - x` to its left. Keyframe
2 (the third undocking step) has the base at the kitchen's centre (2.0, 1.5), heading north too, so the camera is at
(2.0, 1.63, 0.08): the vantage `ego_mem` things are seen from, before the look window that starts at 6.

`EGO_JOB` is `space_kit.JOB` with a clearance of `CLEARANCE_M` (2.0 m), loose enough that a thing 3 to 5 m from the
8 cm camera passes the resolvability screen (`nearest_part.resolvable`); a test of the screen sets its own.

`put` adds a carried thing (or a `light` one, `mobility="light"`) of a colour word standing at (x, y) and seen
recognisably (1,500 px, its faithful colour) at the keyframes given, with the points of its pixels there: the eight
corners of its box unless `pts` gives them (a list of (x, y, z), or a function of the keyframe). `scene` is the script
with the look point set; `world` its window; `cut` the same window ended at the look point, the stream as it was when
the question was asked.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

import family_kit as fk
from space_kit import JOB, SpaceScript, record, vacuum_run
from stbench.questions_world import World
from synth import RECOG

K = 13
CAMERA = (3.0, 1.13, 0.08)
SMALL = (0.1, 0.1, 0.4)   # a thing 0.4 m tall that widens nothing (`rules.widening`)
CLEARANCE_M = 2.0
EGO_JOB = replace(JOB, clearance=CLEARANCE_M)


def scene(n_tasks: int = 1, **fields) -> SpaceScript:
    """The vacuum's `n_tasks`-task script with the base at (3.0, 1.0) heading north at keyframe `K`; `fields` replace
    `SpaceScript`'s (more `poses`, `yaws`, `decisions`, `status`)."""
    poses, yaws = {K: (3.0, 1.0), **fields.pop("poses", {})}, {K: math.pi / 2, **fields.pop("yaws", {})}
    return vacuum_run(n_tasks, poses=poses, yaws=yaws, depth_ks=set(), **fields)


def corners(xy: tuple[float, float], size: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    """The eight corners of an axis-aligned box standing on the floor with its base centre at `xy`."""
    return [(xy[0] + sx * size[0] / 2, xy[1] + sy * size[1] / 2, z) for sx in (-1, 1) for sy in (-1, 1)
            for z in (0.0, size[2])]


def put(s: SpaceScript, eid: str, kind: str, colour: str, xy: tuple[float, float], seen=(), *, px: int = RECOG,
        size: tuple[float, float, float] = SMALL, mobility: str = "carried", rgb=None, where=None, pts=None) -> str:
    """The entity `eid` standing at `xy` (or at the placements `where`), seen with `px` pixels at the keyframes `seen`
    (rendered in `rgb`, its faithful colour when not given), the points of its pixels there `pts` (a list of (x, y, z),
    or a function of the keyframe; by default the corners of its box where it stands at that keyframe's time)."""
    s.item(eid, kind, {"color": colour}, where or [(0, None, (*xy, 0.0))], mobility=mobility, size=size)
    if seen:
        s.see(list(seen), eid, px, rgb=rgb)
        for k in [seen] if isinstance(seen, int) else seen:
            s.parts[k, eid] = np.array(pts(k) if callable(pts) else pts if pts is not None else
                                       corners(_at(s, eid, k, xy), size), dtype=float)
    return eid


def _at(s: SpaceScript, eid: str, k: int, xy: tuple[float, float]) -> tuple[float, float]:
    """Where the entity stands at keyframe `k`: its placement then (the one `put` was given when it has one)."""
    placed = s.timeline().placement_of(eid, s.timeline().keyframe(k).t)
    return xy if placed is None else (float(placed.pos[0]), float(placed.pos[1]))


def world(s: SpaceScript, **kw) -> World:
    return fk.world(s, job=EGO_JOB, **kw)


def cut(s: SpaceScript, k: int = K) -> World:
    """The window of `s`'s first task ended at keyframe `k` (the look point): nothing after it was delivered."""
    inp = fk.inputs(s, EGO_JOB)
    return World(inp, replace(record(s), k1=k + 1))


def told(w: World, *ids: str) -> World:
    """`w` where the robot's own log tells about `ids`: the third `target` of its first task (the helper's carried
    thing, which the subgoal names)."""
    ego = w.inp.tl.ego
    tasks = [replace(ego.tasks[0], targets=("a", "b", *ids)), *ego.tasks[1:]]
    return World(replace(w.inp, tl=replace(w.inp.tl, ego=replace(ego, tasks=tasks))), w.rec)


def named(moment) -> dict[str, dict]:
    """A moment's answerable candidates by entity id: the gold they carry."""
    return {c.entities[0]: c for c in moment.answerable} if moment else {}
