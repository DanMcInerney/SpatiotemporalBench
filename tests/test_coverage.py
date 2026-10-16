"""How a floor-cleaning robot covers a room (`coverage.perimeter`, `coverage.strips`), on hand-made rooms with a
hand-computed free space, and the brush's reach over the apartment's generated scripts."""

from __future__ import annotations

import math

import numpy as np
import pytest

import job_kit
from stbench import coverage
from stbench.geometry import Rect, dist_to_segment, distance
from stbench.mapspec import parse_map
from stbench.route import FreeSpace

RES, CLEAR = 0.05, 0.2  # the grid's cell, and the clearance every free cell keeps from a wall or piece
WALL = {"thickness": 0.1, "height": 2.5, "exterior": True}


def room(width: float = 4.0, height: float = 3.0):
    """A `width` x `height` m room, walls 0.1 m thick (faces at 0.05 and `width` - 0.05), no door, no furniture."""
    return parse_map({
        "name": "room", "bounds": [0, 0, width, height], "wall_thickness": 0.1,
        "wall_height": 2.5,
        "regions": [{"id": "R1", "name": "room", "anchor": [width / 2, height / 2], "level": 0, "z0": 0.0, "z1": 2.5,
                     "polygon": [[0.05, 0.05], [width - 0.05, 0.05], [width - 0.05, height - 0.05],
                                 [0.05, height - 0.05]]}],
        "walls": [{"id": "W01", "a": [0, 0], "b": [width, 0], **WALL},
                  {"id": "W02", "a": [width, 0], "b": [width, height], **WALL},
                  {"id": "W03", "a": [0, height], "b": [width, height], **WALL},
                  {"id": "W04", "a": [0, 0], "b": [0, height], **WALL}],
        "openings": [], "windows": [], "furniture": [], "ceiling_lights": [{"region": "R1", "pos": [2, 1.5, 2.45]}],
    })


def free_space(m, *pieces: Rect) -> FreeSpace:
    return FreeSpace(m, list(pieces), CLEAR, RES)


def length(poly) -> float:
    return sum(distance(a, b) for a, b in zip(poly, poly[1:], strict=False))


def test_the_edge_pass_is_a_closed_counter_clockwise_loop_along_the_wall():
    """Contract: `perimeter` is a closed polyline, counter-clockwise, round the room's free floor a little inside its
    edge, starting where the entry is nearest. Hand case: the walls' faces are at 0.05 and 3.95 (x), 0.05 and 2.95 (y);
    the free cells keep 0.2 m from them (centres from 0.275 to 3.725 and 0.275 to 2.725) and the pass runs one cell
    (`EDGE_INSET_M`) inside that, a rectangle from 0.325 to 3.675 by 0.325 to 2.675: 3.35 x 2.35 m, 11.4 m round,
    starting beside an entry at (0.4, 0.4). Regression caught: a loop that is open, runs the wrong way, or does not
    follow the wall."""
    m = room()
    loop = coverage.perimeter(free_space(m), m.regions[0], (0.4, 0.4))
    assert loop[0] == loop[-1] and distance(loop[0], (0.4, 0.4)) < 0.1
    assert coverage._area(loop[:-1]) == pytest.approx(3.35 * 2.35, abs=0.1)  # counter-clockwise: positive
    assert length(loop) == pytest.approx(2 * (3.35 + 2.35), abs=0.2)
    on_a_side = [min(abs(x - 0.325), abs(x - 3.675), abs(y - 0.325), abs(y - 2.675)) < 0.03 for x, y in loop]
    assert all(on_a_side)
    assert len(loop) <= 7  # the straight sides are not stairs of cells: a start, four corners, the start again


def test_the_edge_pass_goes_round_a_piece_against_the_wall_and_not_round_one_in_the_open():
    """Contract: the loop is the outer contour of the free floor: a piece standing free of the walls is a hole in it and
    leaves it alone; one against a wall is part of the edge and the loop goes round it. Hand case: a 0.6 m box in the
    middle of the room adds nothing to the 11.4 m loop (to 0.2 m); the same box against the south wall (centre y 0.35)
    makes the loop go round its inflated shape, a bay 1.1 m wide (the box, 0.2 m each side, the inset) and 0.575 m deep
    above the loop's 0.325 m line: 1.15 m longer, less its rounded corners, so more than 0.9 m. Regression caught: a
    pass that circles furniture it should not, or that skips the furniture a robot would follow."""
    m = room()
    open_floor = length(coverage.perimeter(free_space(m), m.regions[0], (0.4, 0.4)))
    middle = length(coverage.perimeter(free_space(m, Rect(2.0, 1.5, 0.6, 0.6, 0.0)), m.regions[0], (0.4, 0.4)))
    bay = length(coverage.perimeter(free_space(m, Rect(2.0, 0.35, 0.6, 0.6, 0.0)), m.regions[0], (0.4, 0.4)))
    assert middle == pytest.approx(open_floor, abs=0.2)
    assert bay > open_floor + 0.9


def test_strips_run_along_the_longer_side_a_pitch_apart_back_and_forth():
    """Contract (plan §4.3): strips are parallel to the longer side of the room, `spacing` apart, each run the other way
    round from the last, joined by short hops. Hand case: in a 4 x 3 m room (free floor x 0.25 to 3.75, y 0.25 to 2.75)
    with strips 0.2 m apart from y = 0.15, the lines at y = 0.35, 0.55 ... 2.55 are the twelve that lie in free floor:
    twelve runs of about 3.5 m along x joined by eleven hops of 0.2 m, 44 m in all, the first run beginning at the
    corner nearest the start. Regression caught: strips across the shorter side (twice the turns), strips that do not
    alternate, or a gap between strips wider than the brush."""
    m = room()
    free = free_space(m)
    poly = coverage.strips(free, m.regions[0], (2.0, 1.5), 0.2, (0.3, 0.35))
    runs = [(a, b) for a, b in zip(poly, poly[1:], strict=False) if abs(a[1] - b[1]) < 1e-6 and abs(a[0] - b[0]) > 3.0]
    hops = [(a, b) for a, b in zip(poly, poly[1:], strict=False) if abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) > 0.05]
    assert len(runs) == 12 and len(hops) == 11
    assert [round(a[1], 2) for a, _ in runs] == pytest.approx([0.35 + 0.2 * k for k in range(12)], abs=0.011)
    assert all(abs(distance(a, b) - 0.2) < 0.011 for a, b in hops)
    assert [a[0] < b[0] for a, b in runs] == [k % 2 == 0 for k in range(12)]  # east, then west, then east ...
    assert length(poly) == pytest.approx(12 * 3.5 + 11 * 0.2, abs=0.6)


def test_the_edge_pass_and_the_strips_leave_no_free_cell_out_of_the_brushs_reach():
    """Contract: strips are cut into the runs of free floor they cross, so a piece in the way leaves the strips on
    either side of it; with the edge pass they keep to the free floor and leave no free cell farther than the brush's
    radius (0.125 m: it is 0.25 m across) from the path. Hand case: a 0.6 m box in the middle of the 4 x 3 m room
    blocks, with the clearance's 0.2 m round it, x 1.4 to 2.6 and y 0.9 to 2.1; strips 0.2 m apart leave 0.1 m between
    them, and the edge pass reaches the 0.05 m band by the walls that the outermost strip (0.175 m off it) does not.
    Regression caught: a strip through a piece, or free floor behind it, or by a wall, never visited."""
    m = room()
    free = free_space(m, Rect(2.0, 1.5, 0.6, 0.6, 0.0))
    loop = coverage.perimeter(free, m.regions[0], (0.4, 0.4))
    poly = loop + coverage.strips(free, m.regions[0], (0.4, 0.4), 0.2, loop[-1])[1:]
    assert all(free.line_clear(a, b) for a, b in zip(poly, poly[1:], strict=False))
    cx, cy = free.cells_xy(coverage.reachable_floor(free, m.regions[0], (0.4, 0.4)))
    segments = list(zip(poly, poly[1:], strict=False))
    gap = [min(dist_to_segment(x, y, a, b) for a, b in segments) for x, y in zip(cx, cy, strict=True)]
    assert max(gap) <= 0.125
    assert not free.is_free((2.0, 1.5))  # the box's own cell is blocked


def test_runs_on_neighbouring_strips_chain_by_their_overlap():
    """Contract: a run joins the chain of the previous strip whose last run overlaps it most, and no chain gets two runs
    of one strip; a run that overlaps nothing by `MIN_OVERLAP_M` (0.2 m) starts a chain; a strip with no run ends every
    chain. Hand case, runs as (start, end) along the strip: strip 0 is one run (0, 3); strip 1 is two, (0, 1.2) and
    (1.8, 3), each overlapping it; strip 2 is one run (0, 3), overlapping the two equally; strip 3 is a run (2.9, 3.5),
    which overlaps (0, 3) by 0.1 m only; strip 4 is empty; strip 5 is a run (0, 1). The chains are: strip 0, the left
    run of strip 1 and strip 2; the right run of strip 1; the run of strip 3; the run of strip 5. Regression caught: a
    chain that visits two runs of one strip (a zigzag across an obstacle), or one that continues past nothing."""
    def run(u0, u1):
        return coverage._Run(u0, u1, (u0, 0.0), (u1, 0.0))

    lines = [[run(0, 3)], [run(0, 1.2), run(1.8, 3)], [run(0, 3)], [run(2.9, 3.5)], [], [run(0, 1)]]
    chains = coverage._chains(lines)
    assert [[(r.u0, r.u1) for r in c] for c in chains] == [
        [(0, 3), (0, 1.2), (0, 3)], [(1.8, 3)], [(2.9, 3.5)], [(0, 1)]]


def test_there_is_no_strip_when_no_floor_is_reachable():
    """Contract: `strips` answers just the start when no run fits. Hand case: a room the entry is not free in has no
    reachable floor. Regression caught: strips drawn on floor the robot cannot get to."""
    m = room()
    free = free_space(m, Rect(2.0, 1.5, 6.0, 6.0, 0.0))  # a piece covering the whole room
    assert coverage.strips(free, m.regions[0], (2.0, 1.5), 0.2, (2.0, 1.5)) == [(2.0, 1.5)]


# -- the brush's reach over the generated scripts ----------------------------------------------------------------
def brush_coverage(seed: int) -> float:
    """The share of the room's fine cells (0.1 m) whose centres the base can stand on that lie within the brush's
    radius of the path its keyframes draw, for the seed's script."""
    world, tl = job_kit.generated(seed)
    m = job_kit.apartment()
    room_ = m.region(tl.ego.tasks[0].targets[0])
    brush = job_kit.robot().tools[0].size_m[0] / 2
    free = job_kit.free_space(world)
    base = coverage.reachable_floor(free, room_, job_kit.OUTSIDE_THE_DOCK)
    u = np.arange(math.floor(min(p[0] for p in room_.polygon) * 10), math.ceil(max(p[0] for p in room_.polygon) * 10))
    v = np.arange(math.floor(min(p[1] for p in room_.polygon) * 10), math.ceil(max(p[1] for p in room_.polygon) * 10))
    gx, gy = np.meshgrid((u + 0.5) / 10, (v + 0.5) / 10)
    ix = np.clip(((gx - free.x0) / free.res).astype(int), 0, free.nx - 1)
    iy = np.clip(((gy - free.y0) / free.res).astype(int), 0, free.ny - 1)
    stands = base[iy, ix]
    pts = np.array([kf.base[:2] for kf in tl.ego.keyframes])
    reach = np.full(gx.shape, np.inf)
    for a, b in zip(pts, pts[1:], strict=False):
        ab = b - a
        t = np.clip(((gx - a[0]) * ab[0] + (gy - a[1]) * ab[1]) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        reach = np.minimum(reach, np.hypot(gx - (a[0] + t * ab[0]), gy - (a[1] + t * ab[1])))
    assert stands.sum() > 50  # a room with floor to clean, so the share below is not a share of nothing
    return float((reach[stands] <= brush).mean())


@pytest.mark.parametrize("room_name", job_kit.ROOMS)
def test_coverage_reaches_most_free_floor(room_name):
    """Contract (plan §4.3, design §4): the brush (a disc 0.25 m across at the base) covers at least 90% of the room's
    fine cells that the base can stand on, over the first three seeds that draw the room, with stand-in clutter on the
    floor. Regression caught: strips that skip half a room, an edge pass that wanders off, or free floor behind a piece
    that the sweep never reaches (an unrealistic vacuum)."""
    for seed in job_kit.seeds_for(room_name):
        assert brush_coverage(seed) >= 0.90, (room_name, seed)
