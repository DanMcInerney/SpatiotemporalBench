"""What a flier's depth shows of a place's walls, and where its corridor meets one (0.8.0, docs/families/clear_path.md).

A flier's `clear_path` is the body's corridor against the map's wall boxes at the base's height: nothing else of the
place reaches the height it flies at (`JobSpec.fly_floor_m`). The corridor is exact, from the public map. The question
is whether the robot **saw** the wall it meets, so the gold answers `unknown` where it never saw that wall at all: a
reader of the pictures cannot place a wall it never saw. A wall some frame showed anywhere along its face counts as seen
all along: a building face here is one flat plane, so the stretch the body meets is inferred from the metres shown.

- **The wall record** (`bins_of`, `bins`): for a keyframe, the (wall id, bin) pairs its depth saw. A depth point is on a
  wall when it lies more than `MIN_HEIGHT_M` above the floor (so the street below does not count), within `NEAR_M` of
  the wall's box across it and within `END_M` of its ends along it; the bin is its 1 m stretch along the wall (`BIN_M`).
- **Where the corridor meets a wall** (`hit`): the near edge of the first wall box in the corridor, from the body's edge
  (`grid.front_edge`: the base plus half the body's length along the heading, or along the opposite for "back"), and
  the wall `HIT_DEPTH_M` past it (inside the box).
- **Saw it** (`saw`): the keyframes, among those given, that saw any bin of that wall. Only the first wall the corridor
  meets counts: the body would touch it before any wall beyond, so a seen wall behind an unseen one does not answer.

The gold (`clear_fly`) reads this over the window's keyframes up to the ask, from the script's own depth
(`Inputs.depth_points_of`); a table system (`table_clear_fly`) reads it over the depth its mapper was sent
(`table.Mapper`), so the oracle reproduces the gold. The module is neutral: numpy, `grid`, `mapspec`, `robots` and the
raycast; `bins` reads its inputs by attribute and imports no truth module.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

import numpy as np

from . import grid
from .mapspec import MapSpec, Wall, load_map
from .robots import Robot

BIN_M = 1.0           # the wall record's stretch along a wall, metres
MIN_HEIGHT_M = 0.5    # a depth point is a wall's only above this, so the street under a flier is none
NEAR_M = 0.35         # ... and within this of the wall's box across it
END_M = 0.5           # ... and within this of its ends along it
HIT_DEPTH_M = 0.06    # the wall is read this far past the first obstacle's near edge (inside the box)
CAP_M = 60.0          # the farthest a flier's corridor asks about: a corridor that runs on is open air, never certified

Bin = tuple[str, int]


class DepthInputs(Protocol):
    """What `bins` reads of a compile's inputs (`questions_world.Inputs`): the layout's public map or the timeline that
    names it, and the keyframe's depth as map-frame points."""

    map: MapSpec | None
    tl: Any
    depth_points_of: Callable[[int], np.ndarray] | None


def bins_of(points: np.ndarray, walls: Iterable[Wall]) -> frozenset[Bin]:
    """The (wall id, bin) pairs the map-frame `points` ((N, 3)) saw: those more than `MIN_HEIGHT_M` up, within `NEAR_M`
    of a wall's box across it and `END_M` of its ends along it, binned by `floor(along / BIN_M)` along the wall from its
    `a` end."""
    high = points[points[:, 2] > MIN_HEIGHT_M][:, :2]
    found: set[Bin] = set()
    if not len(high):
        return frozenset()
    for wall in walls:
        ux, uy = wall.direction
        d = high - np.array(wall.a)
        along, across = d[:, 0] * ux + d[:, 1] * uy, d[:, 1] * ux - d[:, 0] * uy
        on = (np.abs(across) <= wall.thickness / 2 + NEAR_M) & (along >= -END_M) & (along <= wall.length + END_M)
        found.update((wall.id, int(b)) for b in np.unique(np.floor(along[on] / BIN_M)))
    return frozenset(found)


# The records of the `Inputs` whose keyframes were read: a depth image is read once per compile, whichever asks.
_RECORDS: dict[int, dict[int, frozenset[Bin]]] = {}


def bins(inp: DepthInputs, ks: Iterable[int]) -> dict[int, frozenset[Bin]]:
    """The wall record of each keyframe `ks` (script numbering), from `inp.depth_points_of` over the layout's walls.
    Computed once a keyframe and `Inputs` (as `Inputs.states_of` is), so a keyframe outside the window the asker
    reads is never read. ValueError for inputs built without the depth points."""
    if inp.depth_points_of is None:
        raise ValueError("these inputs carry no depth points (Inputs.depth_points_of), so no wall can be read")
    ks = list(ks)
    if id(inp) not in _RECORDS:
        _RECORDS[id(inp)] = {}
        weakref.finalize(inp, _RECORDS.pop, id(inp), None)
    done = _RECORDS[id(inp)]
    if missing := [k for k in dict.fromkeys(ks) if k not in done]:
        walls = (inp.map if inp.map is not None else load_map(inp.tl.episode.map)).walls
        for k in missing:
            done[k] = bins_of(inp.depth_points_of(k), walls)
    return {k: done[k] for k in ks}


def wall_at(walls: Iterable[Wall], xy: Sequence[float]) -> str:
    """The id of the wall whose box lies nearest the point (across the wall, then beyond its ends)."""
    best: tuple[float, str] | None = None
    for wall in walls:
        ux, uy = wall.direction
        dx, dy = xy[0] - wall.a[0], xy[1] - wall.a[1]
        along, across = dx * ux + dy * uy, dy * ux - dx * uy
        gap = max(0.0, abs(across) - wall.thickness / 2) + max(0.0, -along, along - wall.length)
        if best is None or gap < best[0]:
            best = (gap, wall.id)
    if best is None:
        raise ValueError("the map has no wall")
    return best[1]


def hit(m: MapSpec, robot: Robot, base: Sequence[float], yaw: float, direction: str, cap: float
        ) -> tuple[float, str] | None:
    """Where the body's corridor from its edge meets a wall: how far it could fly (metres, the raycast of the map's
    wall boxes, 3 decimals) and the id of the wall it stops at. `base` is the delivered (x, y), `yaw` the delivered
    heading in radians, `direction` "ahead" or "back" (the heading plus pi). None when nothing stands in `cap` metres:
    open air."""
    heading = yaw + (math.pi if direction == "back" else 0.0)
    start = grid.front_edge(robot, base, heading)
    far = grid.raycast(m.wall_pieces(), start, heading, robot.body, cap)
    if far >= cap - grid.EPS:
        return None
    reach = far + HIT_DEPTH_M
    return far, wall_at(m.walls, (start[0] + reach * math.cos(heading), start[1] + reach * math.sin(heading)))


def saw(record: dict[int, frozenset[Bin]], ks: Iterable[int], wall: str) -> list[int]:
    """The keyframes among `ks` whose record holds some bin of the wall, in `ks` order."""
    return [k for k in ks if k in record and any(w == wall for w, _ in record[k])]
