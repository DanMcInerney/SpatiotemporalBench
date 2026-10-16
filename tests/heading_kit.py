"""What the tests of `open_heading` share: hand scenes of fine-cell states, the two bodies they are read with, the base
the profile is taken at, and the compiler's window (`heading_world`) of `clear_kit`'s tote AMR over states set by hand.

- `scene()`: free floor seen to 6 m round the base, the north-east quarter beyond (1.1, 1.1) unseen, two walls and a
  pole as obstacle sub-cells (`test_clear_profile.py`).
- `wall_cells()`: free floor west of a wall whose sub-cells have their centres at x 1.01 across every row.
- `room(half)`: free floor to `half` metres on every side, walled all round at `half` + 0.01 m.
"""

from __future__ import annotations

from dataclasses import replace

from clear_kit import FRAMES, JOB, LAST, drive_run
from rear_kit import CAM
from space_kit import record
from stbench import episode as ep
from stbench import grid
from stbench.questions_world import Inputs, World
from stbench.robots import ROBOTS

AMR, VACUUM = ROBOTS["amr"], ROBOTS["vacuum"]
BASE, YAW = (0.0, 0.0), 0.0
Cells = dict[tuple[int, int], object]


def sub(x: float) -> int:
    """The index of the sub-cell whose centre is `x` (a coordinate on an odd centimetre: 2.01, 0.25 ...)."""
    return round(x / grid.SUB_M - 0.5)


def occupied(spots) -> Cells:
    """The fine cells holding the sub-cells (a, b) of `spots`, each an `Occupied` of its own."""
    stands: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for a, b in spots:
        stands.setdefault((a // 5, b // 5), set()).add((a, b))
    return {c: grid.Occupied(subs) for c, subs in stands.items()}


def scene() -> dict[tuple[int, int], tuple[str, int]]:
    """The hand scene (the module docstring) as `grid.latest` gives it, every cell seen at keyframe 0."""
    cells: Cells = {(u, v): "free" for u in range(-60, 60) for v in range(-60, 60) if not (u > 10 and v > 10)}
    spots = [(sub(2.01), b) for b in range(-50, 50)]   # the wall ahead, y -1.0 to 1.0
    spots += [(a, sub(3.01)) for a in range(-50, 50)]   # the wall to the left, x -1.0 to 1.0
    spots += [(sub(0.25), sub(-2.01))]   # the pole to the right
    cells.update(occupied(spots))
    return {c: (s, 0) for c, s in cells.items()}


def wall_cells() -> Cells:
    """Free floor at u -60 .. 9, v -60 .. 59 and a wall in the fine cells u = 10 (sub-cells a = 50: centres x 1.01)."""
    cells: Cells = {(u, v): "free" for u in range(-60, 10) for v in range(-60, 60)}
    cells.update(occupied((50, b) for b in range(-300, 300)))
    return cells


def room(half: float) -> Cells:
    """A square room round the origin: free floor to `half` m on every side (a multiple of 0.1) and a wall of sub-cells
    with centres at `half` + 0.01 m on each of the four sides."""
    n = round(half / grid.FINE_M)
    cells: Cells = {(u, v): "free" for u in range(-n, n) for v in range(-n, n)}
    edges = (sub(half + 0.01), sub(-half - 0.01))
    spots = [(a, e) for a in range(-5 * n - 5, 5 * n + 5) for e in edges]
    cells.update(occupied(spots + [(b, a) for a, b in spots]))
    return cells


HEADING_JOB = replace(JOB, looks=("open_heading",))


def heading_world(by_k: dict[int, Cells], job=HEADING_JOB) -> World:
    """The compiler's window of `clear_kit`'s tote AMR standing at the origin heading +x for `FRAMES` keyframes in
    `transit`, the look point `LAST` ending the look window of keyframes 16 to 19, with the free-space cells of keyframe
    `k` `by_k[k]` (none for a keyframe not named). No depth, no frames: the gold reads the states alone, and a stopper
    has to be stubbed."""
    script = drive_run(poses=dict.fromkeys(range(FRAMES), BASE), depth_ks=())
    tl = script.timeline()

    def states_of(ks):
        return {k: by_k.get(k, {}) for k in ks}

    inp = Inputs(tl, CAM, AMR, job, ep.pose_rows(tl), script.visibility(), states_of)
    return World(inp, record(script))


__all__ = ["AMR", "BASE", "HEADING_JOB", "LAST", "VACUUM", "YAW", "heading_world", "occupied", "room", "scene", "sub",
           "wall_cells"]
