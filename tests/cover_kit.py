"""What the `uncovered` tests share, on top of `family_kit`: a 3 x 2 m living room R2 (x 5..8, y 0.5..2.5: the cells i
10..15, j 1..4, 24 cells wholly inside it), a kitchen R1 (x 0..4.5, so a base at x 4.75 stands in a doorway), a map with
one fixed cabinet and one light chair, hand-set free-space states instead of depth, and a script whose brush cleans
where the test says.

The plan's room is in fine cells (u = floor(x / 0.1), v = floor(y / 0.1); cell (i, j) holds u 5i..5i+4 and v 5j..5j+4).
`cleaning_views` gives two views of the floor: keyframe 9 sees all of u 50..79, v 5..24 free except what is written
below, and keyframe 16 sees u 70..79, v 20..24 again:
- a fixed cabinet at (5.25, 2.25), 0.3 x 0.3 m (fine cells u 51..53, v 21..23, occupied), in cell (10,4);
- a chair leg at fine cell (62,17), in (12,3); clutter at (57,22) and (58,22), in (11,4), at (63,21), in (12,4), and at
  (77,17), in (15,3); a box whose front (66..68, 15) stands in (13,3), and the floor behind it, which no frame saw:
  (74,17..19) of (14,3) and the whole of (13,4);
- the map's light chair stood at (7.25, 2.25), in cell (14,4), and left before the room was seen: that floor is free.

`plan_run`'s brush cleans three strips of the room, at y 0.65, 0.95 and 1.25 (the disc reaches the fine rows v 5..13),
at keyframes 7 to 12; it is off from keyframe 13, and the base stands at the room's centre (6, 1.5) at the decision
(keyframe 18).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from family_kit import multi_run
from space_kit import JOB, SpaceScript
from space_kit import inputs as space_inputs
from space_kit import record as space_record
from stbench.job_registry import JobSpec
from stbench.mapspec import Furniture, MapSpec, Region
from stbench.questions_world import World

ROOM = ((5.0, 0.5), (8.0, 0.5), (8.0, 2.5), (5.0, 2.5))
INSIDE = {(i, j) for i in range(10, 16) for j in range(1, 5)}  # the 24 cells wholly inside R2
# a room whose edges lie on the 0.5 m grid is touched from outside by one more ring of cells (i 9..16, j 0..5)
RING = {(i, j) for i in range(9, 17) for j in range(0, 6)} - INSIDE
UNCOVERED_JOB: JobSpec = replace(JOB, decisions={"before_doorway": (), "room_done": ("uncovered",)})


def cover_map() -> MapSpec:
    """The kitchen R1, the living room R2 and, in R2, a fixed cabinet (5.25, 2.25) and a light chair (7.25, 2.25)."""
    regions = (Region("R1", "kitchen", ((0.0, 0.0), (4.5, 0.0), (4.5, 3.0), (0.0, 3.0)), (2.0, 1.5), 0, 0.0, 2.5,
                      "kitchen"),
               Region("R2", "living room", ROOM, (6.5, 1.5), 0, 0.0, 2.5, "living room"))
    furniture = (Furniture("F01", "cabinet", "cabinet", "fixed", (5.25, 2.25), (0.3, 0.3, 0.5), 0.0, "oak", True, (),
                           0.0),
                 Furniture("L01", "chair", "chair", "light", (7.25, 2.25), (0.4, 0.4, 0.9), 0.0, "oak", True, (), 0.0))
    return MapSpec("apartment_a", (0.0, 0.0, 9.0, 6.2), 0.1, 2.5, regions, (), (), (), furniture, ())


def box(u0: int, u1: int, v0: int, v1: int) -> list[tuple[int, int]]:
    """The fine cells u0..u1 x v0..v1."""
    return [(u, v) for u in range(u0, u1 + 1) for v in range(v0, v1 + 1)]


def cleaning_views() -> dict[int, dict]:
    """`CLEANING`'s two views: {keyframe: {fine cell: state}}."""
    seen = {f: "free" for f in box(50, 79, 5, 24)}
    for f in [(74, 17), (74, 18), (74, 19), *box(65, 69, 20, 24)]:
        del seen[f]
    for f in [(62, 17), (57, 22), (58, 22), (63, 21), (66, 15), (67, 15), (68, 15), (77, 17), *box(51, 53, 21, 23)]:
        seen[f] = "occupied"
    return {9: seen, 16: dict.fromkeys(box(70, 79, 20, 24), "free")}


def strips(k0: int, points: Sequence[tuple[float, float]]) -> tuple[dict, dict]:
    """(poses, status) of consecutive keyframes from `k0` at the points with the brush on."""
    poses = {k0 + i: p for i, p in enumerate(points)}
    return poses, {k: {"brush": True} for k in poses}


def cleaning_run(n_tasks: int, *runs: tuple[int, Sequence[tuple[float, float]]], **fields) -> SpaceScript:
    """`n_tasks` tasks of the vacuum whose brush is on exactly at the keyframes of the `(k0, points)` runs, the base at
    those points there (and at a room's centre elsewhere, the kitchen's or the living room's by the leg)."""
    poses: dict = {}
    status: dict = {}
    for k0, points in runs:
        p, s = strips(k0, points)
        poses.update(p)
        status.update(s)
    return multi_run(n_tasks, **{"poses": poses, "status": status, **fields})


def plan_run(**fields) -> SpaceScript:
    """`CLEANING`'s run: strips at y 0.65, 0.95 and 1.25 along x 5.05 to 7.95 (keyframes 7 to 12)."""
    points = [(5.05, 0.65), (7.95, 0.65), (7.95, 0.95), (5.05, 0.95), (5.05, 1.25), (7.95, 1.25)]
    return cleaning_run(1, (7, points), **fields)


def world(s: SpaceScript, views: dict[int, dict], a: int = 0, b: int | None = None, cid: str = "c0",
          length: str = "short", job: JobSpec = UNCOVERED_JOB, robot=None) -> World:
    """The window of tasks a..b of the script as the compiler's `World`, with `cover_map`, the free-space cells of
    `views` (keyframe -> {fine cell: state}; a keyframe not in it saw nothing) and, when given, another robot."""
    inp = replace(space_inputs(s, job), map=cover_map(), states_of=lambda ks: {k: views.get(k, {}) for k in ks})
    if robot is not None:
        inp = replace(inp, robot=robot)
    return World(inp, space_record(s, a=a, b=b, cid=cid, length=length))
