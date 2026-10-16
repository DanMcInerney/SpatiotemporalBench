"""`grid_cover` (neutral): the brush's swath, the layout's blocked rectangles and the rule that sorts a room's cells.
Every expected value is worked out by hand from the cell and fine-cell arithmetic written beside it."""

from __future__ import annotations

import math

from stbench import grid_cover
from stbench.geometry import Rect
from stbench.mapspec import Furniture, MapSpec, Wall
from stbench.robots import ROBOTS, Tool

BRUSH = Tool("brush", "disc", size_m=(0.25, 0.25), offset_m=(0.0, 0.0))  # radius 0.125 m: reaches fine cells 0.1 m away


def fine(i: int, j: int, n: int = 25) -> list[tuple[int, int]]:
    """The first `n` fine cells of cell (i, j), by hand: u runs over 5i..5i+4 and v over 5j..5j+4, u fastest."""
    return [(5 * i + du, 5 * j + dv) for dv in range(5) for du in range(5)][:n]


def lay(states: dict, cell: tuple[int, int], kind: str, k: int, n: int = 25) -> None:
    """The first `n` fine cells of `cell` observed as `kind` at keyframe `k`."""
    states.update({f: (kind, k) for f in fine(*cell, n)})


def test_cover_by_hand():
    """Contract (plan §5.1, design §8.2): over a room's 0.5 m cells, the cells to name are those wholly inside the
    room, clear of every blocked rectangle, with none of their 25 fine cells swept and all 25 seen free; a cell nobody
    can judge is don't-care; every other cell is a negative (not listed). Hand-worked on the room x 0.05..2.85, y
    0.05..1.55 (cells i 0..5, j 0..3 meet it; i 1..4, j 1..2 are wholly inside: 8 cells, the other 16 are cut by an
    edge) with a blocked piece touching (3,2) along its top edge at y 1.5 and one touching (4,2) along its east edge
    at x 2.5: (1,1) 25 free, unswept: positive; (2,1) all 25 swept: negative; (3,1) 12 swept: don't-care; (4,1) one
    occupied fine cell: negative; (1,2) all 25 unobserved: negative; (2,2) 10 unobserved and 15 free: don't-care;
    (3,2) 25 free but touching a piece: don't-care; (4,2) all 25 swept and touching a piece: don't-care (touching
    comes before the swath). The cell (5,1), cut by the wall line x 2.85 and all free, is don't-care, and (7,1)
    outside the room, all free, is not listed. The evidence of the one positive cell is the keyframes of its fine
    cells, 7 and 9; with it made occupied, every observed fine cell of the eight inside cells: 3, 4, 5, 6, 7 and 9
    (not the cut cell's 8). Regression caught: a cell perception could not judge entering the gold, or one it could
    left out."""
    room = [(0.05, 0.05), (2.85, 0.05), (2.85, 1.55), (0.05, 1.55)]
    blocked = [Rect(1.8, 1.6, 0.2, 0.2, 0.0), Rect(2.6, 1.25, 0.2, 0.4, 0.0)]  # x 1.7..1.9, y 1.5..1.7; x 2.5..2.7
    states: dict = {}
    lay(states, (1, 1), "free", 9)
    states.update({f: ("free", 7) for f in fine(1, 1, 15)})  # the first three rows (v 5..7) seen at 7, the rest at 9
    for cell in ((2, 1), (3, 1), (4, 1), (3, 2), (4, 2)):
        lay(states, cell, "free", 6 if cell == (3, 2) else 3)
    states[(22, 7)] = ("occupied", 5)  # one fine cell of (4,1)
    lay(states, (2, 2), "free", 4, 15)  # (2,2): rows v 10..12 seen, v 13 and 14 not
    lay(states, (5, 1), "free", 8)
    lay(states, (7, 1), "free", 8)
    swept = {*fine(2, 1), *fine(3, 1, 12), *fine(4, 2)}
    positive, dont_care, evidence = grid_cover.cover(states, swept, room, blocked)
    cut = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 3], [2, 0], [2, 3], [3, 0], [3, 3], [4, 0], [4, 3],
           [5, 0], [5, 1], [5, 2], [5, 3]]
    assert (positive, evidence) == ([[1, 1]], [7, 9])
    assert dont_care == sorted([*cut, [3, 1], [2, 2], [3, 2], [4, 2]])
    states[(7, 7)] = ("occupied", 9)  # a fine cell of (1,1) now occupied: no cell is named
    positive, dont_care, evidence = grid_cover.cover(states, swept, room, blocked)
    assert (positive, evidence) == ([], [3, 4, 5, 6, 7, 9])
    assert dont_care == sorted([*cut, [3, 1], [2, 2], [3, 2], [4, 2]])


def test_brush_swath_by_hand():
    """Contract: the swath is what the brush covers while it is on: `grid.swath` of the tool's centre (the base plus
    the tool's offset turned by the yaw) over each run of consecutive brush-on poses, a run of one pose covering its
    own footprint, and a brush-off pose ending a run. Hand-worked with the 0.25 m disc (radius 0.125, so the fine
    cells whose centres lie within 0.125 m: a pass at y = 0.55 reaches the rows with centres 0.45, 0.55 and 0.65, v 4
    to 6, along its length, and one more column at each end, in the row of its axis only: centres 0.1 m beyond the
    ends are 0.1 m away, and those one row off 0.14 m). Two passes along x 0.55 to 1.45, one at y 0.55 and one at y
    1.55 after a brush-off pose between them at (1.45, 1.05), are u 5..14 x v 4..6 with (4,5) and (15,5), and the same
    shifted to rows v 14..16: no fine cell between them, which a swath counting the transit would fill. A tool offset
    0.3 m ahead turned to face north puts the disc at (1.0, 1.3) and covers the four fine cells (u 9..10, v 12..13)
    whose centres are 0.07 m away; a rectangle 0.4 by 0.2 centred 0.1 m ahead covers the eight cells with centres in x
    0.9..1.3 and y 0.9..1.1. Regression caught: transit (brush off) counted as cleaning, or an offset applied
    unturned."""
    def pass_cells(row_v: int) -> set[tuple[int, int]]:
        return {(u, v) for u in range(5, 15) for v in range(row_v - 1, row_v + 2)} | {(4, row_v), (15, row_v)}

    poses = [(0.55, 0.55, 0.0, True), (1.45, 0.55, 0.0, True), (1.45, 1.05, 0.0, False), (1.45, 1.55, 0.0, True),
             (0.55, 1.55, 0.0, True)]
    assert grid_cover.brush_swath(poses, BRUSH) == pass_cells(5) | pass_cells(15)
    assert grid_cover.brush_swath([], BRUSH) == set()
    assert grid_cover.brush_swath([(1.0, 1.0, 0.0, False)] * 3, BRUSH) == set()
    ahead = Tool("brush", "disc", size_m=(0.25, 0.25), offset_m=(0.3, 0.0))
    north = [(0.0, 0.0, 0.0, False), (1.0, 1.0, math.pi / 2, True), (4.0, 4.0, 0.0, False)]
    assert grid_cover.brush_swath(north, ahead) == {(9, 12), (10, 12), (9, 13), (10, 13)}
    bar = Tool("bar", "rect", size_m=(0.4, 0.2), offset_m=(0.1, 0.0))
    assert grid_cover.brush_swath([(1.0, 1.0, 0.0, True)], bar) == {(u, v) for u in range(9, 13) for v in (9, 10)}


def test_blocked_by_hand():
    """Contract: `blocked` is the layout's floor obstacles for a robot from the public map alone: the wall pieces, and
    each fixed piece that blocks: its four 0.15 m leg squares when the robot passes under it (its underside at least
    the body's height plus the under margin: 0.10 m for the vacuum, 1.46 m for the helper), else its whole footprint.
    A rug (blocks False) never, and a light piece never. Hand-worked on one 3 m wall (a piece 3.1 m long, 0.1 m thick,
    its ends a half thickness past the wall's) and four pieces: a cabinet at (1, 1) 0.6 x 0.4 solid; a table at (2,
    1), 1.0 x 0.6, underside 0.64 m (legs at (2 +- 0.425, 1 +- 0.225) for the vacuum, the whole footprint for the
    helper); a sofa at (3, 1), 0.9 x 2.0, underside 0.08 m (under the vacuum's 0.10: the whole footprint); a rug and a
    chair, light. Regression caught: a light piece's footprint or a rug in the gold's don't-care (a world process
    moves them unseen), or a table's legs read as its whole top."""
    def piece(name: str, kind: str, mobility: str, centre, size, blocks: bool, under: float) -> Furniture:
        return Furniture(name, kind, kind, mobility, centre, size, 0.0, "oak", blocks, (), under)

    wall = Wall("W1", (0.0, 0.0), (3.0, 0.0), 0.1, 2.5, True)
    furniture = (piece("F1", "cabinet", "fixed", (1.0, 1.0), (0.6, 0.4, 1.8), True, 0.0),
                 piece("F2", "table", "fixed", (2.0, 1.0), (1.0, 0.6, 0.75), True, 0.64),
                 piece("F3", "sofa", "fixed", (3.0, 1.0), (0.9, 2.0, 0.85), True, 0.08),
                 piece("F4", "rug", "fixed", (1.0, 2.0), (2.0, 1.4, 0.01), False, 0.0),
                 piece("L1", "chair", "light", (2.0, 2.0), (0.4, 0.4, 0.9), True, 0.0))
    m = MapSpec("hand", (0.0, 0.0, 4.0, 3.0), 0.1, 2.5, (), (wall,), (), (), furniture, ())

    def rounded(rects: list[Rect]) -> list[tuple[float, ...]]:
        return sorted(tuple(round(v, 6) for v in r) for r in rects)

    shared = [(1.5, 0.0, 3.1, 0.1, 0.0), (1.0, 1.0, 0.6, 0.4, 0.0), (3.0, 1.0, 0.9, 2.0, 0.0)]
    legs = [(2.425, 1.225, 0.15, 0.15, 0.0), (2.425, 0.775, 0.15, 0.15, 0.0), (1.575, 1.225, 0.15, 0.15, 0.0),
            (1.575, 0.775, 0.15, 0.15, 0.0)]
    assert rounded(grid_cover.blocked(m, ROBOTS["vacuum"])) == sorted([*shared, *legs])
    assert rounded(grid_cover.blocked(m, ROBOTS["helper"])) == sorted([*shared, (2.0, 1.0, 1.0, 0.6, 0.0)])
