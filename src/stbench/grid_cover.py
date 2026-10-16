"""The coverage rule `uncovered` reads: the fine cells a robot's tool swept, the obstacles a robot cannot drive over
whatever any frame saw, and the cells of a room that were seen as open floor and not cleaned (plan §5.1, design §8.2).

`brush_swath` is the robot's own path through the tool (what it cleaned); `blocked` is the layout's fixed floor
obstacles, taken from the public map, never from a light piece (world processes move those unseen); `cover` sorts the
0.5 m cells of a room into the cells to name (`positive`), the cells nobody can judge (`dont_care`) and, by leaving them
out, the negatives. The gold and the table systems' answer both call these, over the same `grid.latest` states, so they
agree on every cell. A cell judged here needs the depth: the swath alone cannot say whether a cell was open floor.

This module is neutral (numpy through `grid`, `geometry`, `robots`; the map only as a type): it reads no truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import grid
from .geometry import Point, Rect, dist_to_polygon, rects_overlap, segments_cross, under_obstacles
from .robots import Robot, Tool, passes_under

if TYPE_CHECKING:
    from .mapspec import MapSpec

TOOL = "brush"  # the robot's tool whose footprint cleans, and the status key that switches it on
TOUCH_M = 1e-6  # a cell and a blocked rectangle this close, or closer, touch: the cell's closed square is tested
Cell = tuple[int, int]
Pose = tuple[float, float, float, bool]  # base x, y, yaw (radians), the tool's on-switch


def brush_of(robot: Robot) -> Tool:
    """The robot's tool named `brush`; ValueError when it has none (the family cannot be asked of such a robot). Its
    on-switch is the status key of the same name."""
    for tool in robot.tools:
        if tool.name == TOOL:
            return tool
    raise ValueError(f"the robot {robot.id!r} has no tool named {TOOL!r}: uncovered cannot be asked of it")


def brush_swath(poses: Sequence[Pose], tool: Tool) -> set[Cell]:
    """The fine cells the tool covers. Over each run of consecutive poses `(x, y, yaw, on)` with `on` true, the tool's
    centre is the base plus `tool.offset_m` turned by the yaw, and `grid.swath` covers it with the tool's shape and
    size: a disc by the cells within its radius of the straight segments between the centres, a rectangle as
    `grid.swath` does; a run of one pose covers its own footprint. A pose with `on` false ends a run, so the path
    between two runs (transit, a charging gap) covers nothing."""
    out: set[Cell] = set()
    run: list[tuple[float, float, float]] = []
    ox, oy = tool.offset_m
    for x, y, yaw, on in [*poses, (0.0, 0.0, 0.0, False)]:  # the last, off pose closes the final run
        if on:
            c, s = math.cos(yaw), math.sin(yaw)
            run.append((x + c * ox - s * oy, y + s * ox + c * oy, yaw))
        elif run:
            out |= grid.swath(run, tool.shape, tool.size_m[0], tool.size_m[1])
            run = []
    return out


def blocked(m: MapSpec, robot: Robot) -> list[Rect]:
    """What blocks the floor of this layout for this robot, whatever any frame showed: the map's wall pieces and each
    fixed piece's obstacles (`geometry.under_obstacles`, its four leg squares, when the robot passes under it,
    `robots.passes_under`, else its whole footprint). Rugs and mats never; light pieces never, since the world moves
    them where no frame sees. A function of the public map and the robot alone."""
    out = list(m.wall_pieces())
    for f in m.furniture:
        if f.mobility == "fixed" and f.blocks:
            out += under_obstacles(f.footprint(), f.under_m) if passes_under(robot, f.under_m) else [f.footprint()]
    return out


# -- the room's cells ----------------------------------------------------------------------------------------------
def _square(c: Cell) -> list[Point]:
    """The corners of the cell's square, counter-clockwise from its south-west corner."""
    x0, y0, x1, y1 = (c[0] * grid.CELL_M, c[1] * grid.CELL_M, (c[0] + 1) * grid.CELL_M, (c[1] + 1) * grid.CELL_M)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _in(p: Point, room: Sequence[Point]) -> bool:
    """The point lies in the room, its boundary included (to 1e-9)."""
    return dist_to_polygon(p[0], p[1], list(room)) <= grid.EPS


def _meets(quad: list[Point], room: Sequence[Point]) -> bool:
    """The closed square meets the closed polygon: a corner of one lies in the other, or their outlines cross."""
    (x0, y0), (x1, y1) = quad[0], quad[2]
    in_square = [x0 - grid.EPS <= x <= x1 + grid.EPS and y0 - grid.EPS <= y <= y1 + grid.EPS for x, y in room]
    if any(in_square) or any(_in(p, room) for p in quad):
        return True
    edges = list(zip(room, [*room[1:], room[0]], strict=True))
    sides = list(zip(quad, [*quad[1:], quad[0]], strict=True))
    return any(segments_cross(a, b, p, q) for a, b in edges for p, q in sides)


def _room_cells(room: Sequence[Point]) -> tuple[list[Cell], list[Cell]]:
    """(cells wholly inside the room, cells that meet it without lying wholly inside), each sorted: a cell is the closed
    square of `grid.cell`'s convention; it lies wholly inside when every corner is in the room, the boundary counting
    as inside."""
    xs, ys = [p[0] for p in room], [p[1] for p in room]
    inside: list[Cell] = []
    part: list[Cell] = []
    for i in range(math.floor(min(xs) / grid.CELL_M) - 1, math.floor(max(xs) / grid.CELL_M) + 2):
        for j in range(math.floor(min(ys) / grid.CELL_M) - 1, math.floor(max(ys) / grid.CELL_M) + 2):
            quad = _square((i, j))
            if all(_in(p, room) for p in quad):
                inside.append((i, j))
            elif _meets(quad, room):
                part.append((i, j))
    return inside, part


def _touches(c: Cell, rects: list[Rect]) -> bool:
    """The cell's closed square touches (or overlaps) a rectangle, to `TOUCH_M`."""
    cx, cy = (c[0] + 0.5) * grid.CELL_M, (c[1] + 0.5) * grid.CELL_M
    square = Rect(cx, cy, grid.CELL_M, grid.CELL_M, 0.0)
    reach = grid.CELL_M * math.sqrt(2) / 2 + TOUCH_M
    near = [r for r in rects if math.hypot(r.cx - cx, r.cy - cy) <= reach + math.hypot(r.sx, r.sy) / 2]
    return any(rects_overlap(square, r, TOUCH_M) for r in near)


def open_cells(room: Sequence[Point], swept: set[Cell], blocked: list[Rect]) -> list[Cell]:
    """The cells that could be positive: wholly inside the room, not touching a blocked rectangle and with no fine cell
    in `swept`. It needs no depth, so it bounds what `cover` can name from above, and it is the answer a log of the
    robot's own path alone would give."""
    return [c for c in _room_cells(room)[0]
            if not _touches(c, blocked) and not any(f in swept for f in grid.fine_cells_of(c))]


def _unseen_clean(c: Cell, states: dict[Cell, tuple[str, int]], swept: set[Cell]) -> str:
    """`positive`, `dont_care` or `negative` for a cell that is inside the room and clear of every blocked rectangle."""
    fine = grid.fine_cells_of(c)
    n_swept = sum(f in swept for f in fine)
    if n_swept:
        return "negative" if n_swept == len(fine) else "dont_care"
    held = [states.get(f, ("", 0))[0] for f in fine]
    if any(s == "occupied" for s in held):
        return "negative"
    free = held.count("free")
    return "positive" if free == len(fine) else "negative" if free == 0 else "dont_care"


def cover(states: dict[Cell, tuple[str, int]], swept: set[Cell], room: Sequence[Point], blocked: list[Rect]
          ) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """(the cells to name, the don't-care cells, the evidence keyframes) of a room, from `states` (`grid.latest`'s
    output), the fine cells `swept` (`brush_swath`), the room's polygon and the `blocked` rectangles (`blocked`). A
    cell is `[i, j]` (`grid.cell`), the lists sorted. Over each 0.5 m cell whose closed square meets the room:
    1. a corner of its square outside the room (the boundary is inside): don't-care;
    2. its closed square touches a blocked rectangle: don't-care;
    3. of its 25 fine cells, some but not all in `swept`: don't-care; all: negative;
    4. none swept: negative when any fine cell is `occupied` (something stands in it) or none was observed; **positive**
       when all 25 are `free`; else (some unobserved, the rest `free`) don't-care.
    Every cell not listed is a negative, a cell outside the room included. The evidence is the sorted distinct keyframes
    of the latest states of the positive cells' fine cells, or, with no positive cell, of every observed fine cell of
    the cells wholly inside the room (what showed the room held no more open floor)."""
    inside, part = _room_cells(room)
    positive: list[Cell] = []
    dont_care: list[Cell] = list(part)
    for c in inside:
        kind = "dont_care" if _touches(c, blocked) else _unseen_clean(c, states, swept)
        if kind == "positive":
            positive.append(c)
        elif kind == "dont_care":
            dont_care.append(c)
    shown = positive or inside
    evidence = {states[f][1] for c in shown for f in grid.fine_cells_of(c) if f in states}
    return [list(c) for c in sorted(positive)], [list(c) for c in sorted(dont_care)], sorted(evidence)
