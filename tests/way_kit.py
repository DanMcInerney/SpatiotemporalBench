"""What the tests of `way_to` share: hand scenes of fine cells, and the compiler's window (`way_world`) of the tote
AMR's `clear_kit` script with the home helper's body and the helper's numbers (clearance C 0.15 m, re-find radius R 1.0
m, so the room to move T is 0.34 + 0.15 = 0.49 m) over states set by hand.

- `door_scene()`: the west room x 0 to 3.0, y 0 to 3.0 and the east room x 3.2 to 6.2, y 0 to 3.0, both free; between
  them a wall x 3.0 to 3.2 (occupied) with a door y 2.0 to 2.8 (free, 0.8 m wide). A heading from the west room toward
  the east is blocked by the wall; the way is through the door.
- `corridor()`: free floor x -2.0 to 6.0, y 0 to 0.8, open at both ends: a straight line whose distances are exact
  multiples of 0.1 m.
Cells are fine cells (0.1 m): a cell (u, v) is [0.1u, 0.1u + 0.1) x [0.1v, 0.1v + 0.1).
"""

from __future__ import annotations

from dataclasses import replace

from clear_kit import FRAMES, LAST, drive_run, thing
from clear_kit import JOB as CLEAR_JOB
from rear_kit import CAM
from space_kit import record
from stbench import episode as ep
from stbench.geometry import quat_z
from stbench.questions_world import Inputs, World
from stbench.robots import ROBOTS
from stbench.timeline import Placement

HELPER = ROBOTS["helper"]
REACH, ROOM = 1.0, 0.49   # R and T of the helper's job
Cells = dict[tuple[int, int], str]
JOB = replace(CLEAR_JOB, looks=("way_to",), look_phases={"way_to": frozenset({"transit"})}, clearance=0.15,
              reacquire=REACH)


def block(x0: float, x1: float, y0: float, y1: float) -> set[tuple[int, int]]:
    """The fine cells of the rectangle x0 to x1, y0 to y1 metres (the edges on multiples of 0.1)."""
    return {(u, v) for u in range(round(x0 / 0.1), round(x1 / 0.1)) for v in range(round(y0 / 0.1), round(y1 / 0.1))}


def door_scene(door: bool = True) -> Cells:
    """The module docstring's two rooms and the wall between them; `door=False` leaves the door cells unseen."""
    cells: Cells = dict.fromkeys(block(0, 3.0, 0, 3.0) | block(3.2, 6.2, 0, 3.0), "free")
    if door:
        cells |= dict.fromkeys(block(3.0, 3.2, 2.0, 2.8), "free")
    return cells | dict.fromkeys(block(3.0, 3.2, 0, 2.0) | block(3.0, 3.2, 2.8, 3.0), "occupied")


def corridor() -> Cells:
    return dict.fromkeys(block(-2.0, 6.0, 0, 0.8), "free")


def latest(cells: Cells, k: int = 0) -> dict[tuple[int, int], tuple[str, int]]:
    """The cells as `grid.latest` gives them when keyframe `k` saw all of them."""
    return {c: (s, k) for c, s in cells.items()}


def way_world(by_k: dict[int, Cells], poses: dict[int, tuple[float, float, float]] | None = None,
              job=JOB, thing_at: tuple[float, float] = (5.0, 0.5),
              moved: tuple[int, tuple[float, float]] | None = None) -> World:
    """The compiler's window of `FRAMES` keyframes in `transit`: the base at the poses `poses` gives ((x, y, yaw in
    radians) by keyframe; default (2.0, 0.5) heading +x throughout), the thing `T1` standing at `thing_at` (and, with
    `moved` = (keyframe, place), at that place from that keyframe on), and the free-space cells of keyframe `k`
    `by_k[k]` (none for a keyframe not named). No depth, no frames: the gold reads the states alone. The look point is
    `LAST`."""
    poses = poses or dict.fromkeys(range(FRAMES), (2.0, 0.5, 0.0))
    script = drive_run(poses={k: p[:2] for k, p in poses.items()}, yaws={k: p[2] for k, p in poses.items()},
                       depth_ks=())
    thing(script, "T1", (0.4, 0.4, 0.4), (*thing_at, 0.0))
    if moved is not None:
        t = script.timeline().keyframe(moved[0]).t
        script.placements[-1] = replace(script.placements[-1], t1=t)
        script.placements.append(Placement("T1", t, None, (*moved[1], 0.0), quat_z(0.0), None))
    tl = script.timeline()

    def states_of(ks):
        return {k: by_k.get(k, {}) for k in ks}

    return World(Inputs(tl, CAM, HELPER, job, ep.pose_rows(tl), script.visibility(), states_of), record(script))


__all__ = ["HELPER", "JOB", "LAST", "REACH", "ROOM", "block", "corridor", "door_scene", "latest", "way_world"]
