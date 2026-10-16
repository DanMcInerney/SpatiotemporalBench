"""What the tests of the ground corridor (`clear_ground`, `corridor_screens`, `table_clear`) share, on top of
`space_kit`: the tote AMR (0.8 x 0.58 m, climb 0.01 m, C 0.45 m) driving +x on a flat floor, with ray-cast depth from
its own camera (`rear_kit.CAM`) and a frame that says which box each pixel shows (`Inputs.frame_of`), so a stopper can
be named.

`drive_run` is one task of `FRAMES` keyframes in the phase `transit` (the first is the start phase), the base at `BASE`
(0.6, 0) heading +x throughout, so its front edge is at x 1.0 (`FRONT_X`): a box whose face is at x 3.01 is 2.01 m ahead
of it, in the sub-cell [3.00, 3.02), and the corridor stops at 3.01 - 1.0 - 0.01 = 2.0 m. The stream is cut after
keyframes 7 and 15 and the task's last, 19; keyframe `LAST` (19) ends the last look window (keyframes 16 to 19), the
look point the tests ask at. `depth_ks` names the keyframes with a render (all of the look window by default);
`frame_boxes` puts a box in one keyframe only.

An `Owned` box is a `space_kit` box with the entity id of the thing it stands for ("" for structure). `world` gives the
compiler's `World` of a script, with `frame_of` built from the owned boxes: a pixel belongs to the first box that holds
it (to 3 mm, the depth's rounding), else to no entity (the floor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from rear_kit import AMR, CAM, RearScript
from rear_kit import JOB as AMR_JOB
from space_kit import JOB as SPACE_JOB
from space_kit import Box, box, record
from stbench import episode as ep
from stbench import grid
from stbench.geometry import quat_z
from stbench.job_registry import JobSpec
from stbench.pixels import Frame
from stbench.questions_world import Inputs, World
from stbench.timeline import Entity, Placement
from synth import WAREHOUSE

FRAMES, LAST = 20, 19
BASE = (0.6, 0.0)
FRONT_X = BASE[0] + AMR.body.length_m / 2   # 1.0
TOLERANCE_M = 0.003

JOB = replace(AMR_JOB, decisions={}, looks=("clear_path",), clear_phases=frozenset({"transit"}), clearance=0.45,
              assignments={"clean_room": "Its current assignment is to dock at the {room}."})


@dataclass
class Owned:
    """A box, the id of the entity it stands for (empty: structure) and the keyframes it labels."""

    at: Box
    who: str = ""
    frames: tuple[int, ...] | None = None   # the keyframes it labels (None: all)


def wall(face_x: float, owner: str = "") -> Owned:
    """A 0.2 m deep, 2 m tall wall across the whole corridor with its near face at `face_x`."""
    return Owned(box(face_x, face_x + 0.2, -3.0, 3.0, 0.0, 2.0), owner)


def drive_run(*, depth_ks=range(16, 20), **fields) -> RearScript:
    """One task of `FRAMES` keyframes at `BASE` heading +x; `fields` replace `SpaceScript`'s (`boxes`, `frame_boxes`,
    `seed`, `status` ...). The boxes are plain `space_kit` boxes: the owners are given to `world`."""
    setup = {"tasks": [[("dock", FRAMES, "transit")]], "stage": WAREHOUSE,
             "poses": dict.fromkeys(range(FRAMES), BASE), "yaws": dict.fromkeys(range(FRAMES), 0.0),
             "depth_ks": set(depth_ks)}
    return RearScript(**{**setup, **fields})


def thing(script: RearScript, eid: str, size: tuple[float, float, float], at: tuple[float, float, float],
          mobility: str = "carried", under_m: float = 0.0) -> None:
    """An entity of the whole script, its base centre at `at`, in the timeline (the height screen reads it)."""
    script.entities.append(Entity(eid, "pallet", {"color": "grey"}, mobility, size, under_m=under_m))
    script.placements.append(Placement(eid, 0.0, None, at, quat_z(0.0), None))


def seen_from(script: RearScript, frames, base: tuple[float, float], yaw: float) -> RearScript:
    """The keyframes `frames` of the script stand at `base` heading `yaw` (radians) instead of at `BASE` heading +x, so
    their camera sees the other side of the place; the script is changed in place and returned."""
    for k in frames:
        script.poses[k], script.yaws[k] = base, yaw
    return script


PI = math.pi
# A closed room round the base (0.6, 0): walls across the whole corridor, each 0.2 m deep, the near faces 2.01 m ahead
# of the front edge (x 3.01), 2.41 behind the rear edge (x -2.21), 3.01 to the left of the front edge on the turned
# heading (y 3.01) and 2.11 to the right (y -2.51). The sub-cell holding each face has its centre on it, so the
# corridors stop at 2.0, 2.4, 2.6 and 2.1 m (a face on a sub-cell's edge falls in the next one: 0.01 m more).
ROOM = {"ahead": box(3.01, 3.21, -3, 3, 0, 2), "back": box(-2.41, -2.21, -3, 3, 0, 2),
        "left": box(-3, 3, 3.01, 3.21, 0, 2), "right": box(-3, 3, -2.71, -2.51, 0, 2)}
LOOKS = {3, 7, 11}   # the keyframes of `room_run` that look back, left and right; the look window (16-19) looks ahead


def room_run(walls=tuple(ROOM), **fields) -> RearScript:
    """`drive_run` in the closed room (`ROOM`'s walls named in `walls`) whose camera looked all round before the point:
    keyframes 0 to 3 stand at (2.0, 0) facing -x (the floor behind the base from x 1.07 on and the wall at -2.21), 4 to
    7 at (0.6, -1.5) facing +y, 8 to 11 at (0.6, 1.5) facing -y, and 12 to 19 at `BASE` facing +x, the look window's
    among them. Only `LOOKS` and the point (`LAST`, the look window's last) have a render (the rest would only slow the
    test)."""
    script = drive_run(boxes=[ROOM[name] for name in walls], depth_ks=LOOKS | {LAST}, **fields)
    seen_from(script, range(0, 4), (2.0, 0.0), PI)
    seen_from(script, range(4, 8), (0.6, -1.5), PI / 2)
    seen_from(script, range(8, 12), (0.6, 1.5), -PI / 2)
    return script


def world(script: RearScript, owned: list[Owned] | None = None, job: JobSpec = JOB, cid: str = "c0") -> World:
    """The window of the script's only task as the compiler's `World`, with `frame_of` that labels each pixel by the
    first of `owned` that holds it. Without `owned`, `frame_of` is absent. `cid` is the case id (the direction a
    job with several draws hashes it)."""
    tl, cameras, done = script.timeline(), script.cameras(), {}

    def states_of(ks):
        for k in ks:
            if k not in done:
                done[k] = grid.observe(grid.points(script.depth_of(k), CAM, *cameras[k]), tl.keyframe(k).base[2],
                                       grid.band(AMR))
        return {k: done[k] for k in ks}

    def frame_of(k):
        pts = grid.points(script.depth_of(k), CAM, *cameras[k])
        owner = np.full(len(pts), "", dtype="<U8")
        for o in owned:
            if o.frames is not None and k not in o.frames:
                continue
            lo, hi = np.asarray(o.at[0]) - TOLERANCE_M, np.asarray(o.at[1]) + TOLERANCE_M
            owner[np.all((pts >= lo) & (pts <= hi), axis=1) & (owner == "")] = o.who
        return Frame(pts, owner)

    inp = Inputs(tl, CAM, AMR, job, ep.pose_rows(tl), script.visibility(), states_of,
                 frame_of=None if owned is None else frame_of)
    return World(inp, record(script, cid=cid))


__all__ = ["BASE", "FRAMES", "FRONT_X", "JOB", "LAST", "LOOKS", "PI", "ROOM", "SPACE_JOB", "Owned", "drive_run",
           "room_run", "seen_from", "thing", "wall", "world"]
