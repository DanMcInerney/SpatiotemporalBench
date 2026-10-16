"""A round body's own corridor (`grid.reach`, `grid.stop`, `grid.corridor_clear`, `grid.raycast`; final review B2): a
disc of radius r moving straight touches an obstacle at lateral offset y after `a + r - sqrt(r^2 - y^2)`, where `a` is
the obstacle's distance ahead of the disc's front edge, not after `a`. Hand scenes on the vacuum (0.34 m disc, r 0.17)
from a front edge at the origin heading +x with `blind` 0, the floor to the cap free and obstacle points seen at
keyframe 2; `BOX`, the same size as a box, is the rectangle rule they differ from. An obstacle point lies in a 0.02 m
sub-cell and counts at the sub-cell's centre: x 0.5 in [0.50, 0.52), centre 0.51; y 0.15 in [0.14, 0.16), centre 0.15;
y 0.0 in [0.00, 0.02), centre 0.01."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from stbench import grid
from stbench.geometry import Rect
from stbench.robots import ROBOTS

VACUUM = ROBOTS["vacuum"]
BOX = dataclasses.replace(VACUUM, body=dataclasses.replace(VACUUM.body, shape="rect"))
BAND = grid.band(VACUUM)
R = 0.17
MM = 0.0011   # a gold is rounded to the millimetre


def scene(*points: tuple[float, float], cap: float = 1.0):
    """The latest states of a floor seen free to `cap` (keyframe 1, x -0.3 to `cap`, y -0.5 to 0.5) and the obstacle
    points (x, y, 0.05 high) seen at keyframe 2."""
    free = {(u, v): "free" for u in range(-3, round(cap * 10)) for v in range(-5, 5)}
    pts = np.array([(x, y, 0.05) for x, y in points]).reshape(-1, 3)
    return grid.latest({1: free, 2: grid.observe(pts, 0.0, BAND)})


def gold(robot, *points: tuple[float, float], cap: float = 1.0) -> dict:
    return grid.corridor_clear(scene(*points, cap=cap), robot, (0.0, 0.0), 0.0, cap, 0.0)[0]


def test_reach_is_the_discs_travel_to_a_point_and_the_boxs_along():
    """Contract: `reach(shape, half, along, across)` is how far a body whose front edge is at the origin moves before
    it first touches the point. Hand values for r 0.17: a point 0.5 ahead on the axis: 0.5; 0.5 ahead and 0.15 to the
    side: 0.5 + 0.17 - sqrt(0.17^2 - 0.15^2) = 0.5 + 0.17 - 0.08 = 0.59; at the disc's widest point (0.17): 0.5 + 0.17 =
    0.67; just outside (0.18): never (None), unless `widen` takes it at the widest point; a point beside the front arc,
    0.05 behind the edge and 0.15 to the side: -0.05 + 0.17 - 0.08 = 0.04 (touched as the disc moves on). The box
    touches every point of its width after `along` alone. Regression caught: the strip rule used for a disc (0.5
    whatever the side)."""
    assert grid.reach("disc", R, 0.5, 0.0) == pytest.approx(0.5, abs=1e-3)
    assert grid.reach("disc", R, 0.5, 0.15) == pytest.approx(0.59)
    assert grid.reach("disc", R, 0.5, 0.17) == pytest.approx(0.67)
    assert grid.reach("disc", R, 0.5, -0.17) == pytest.approx(0.67)
    assert grid.reach("disc", R, 0.5, 0.18) is None
    assert grid.reach("disc", R, 0.5, 0.18, widen=0.02) == pytest.approx(0.67)
    assert grid.reach("disc", R, -0.05, 0.15) == pytest.approx(0.04)
    assert grid.reach("rect", R, 0.5, 0.15) == 0.5 and grid.reach("rect", R, 0.5, 0.18) is None


@pytest.mark.parametrize("y, want", [(0.0, 0.5), (0.15, 0.59), (0.17, 0.67)])
def test_the_disc_stops_after_its_own_travel_to_the_obstacle(y, want):
    """Contract: the gold is the disc's travel to the nearest obstacle sub-cell's centre, less half a sub-cell: an
    obstacle converging on the side (the vacuum along a wall) stops it later than the strip says. Hand values for a
    point at x 0.5 (centre 0.51): on the axis (centre y 0.01) 0.51 + 0.17 - sqrt(0.17^2 - 0.01^2) = 0.5103, less 0.01:
    0.50; at y 0.15 (centre 0.15) 0.51 + 0.17 - 0.08 = 0.60, less 0.01: 0.59; at y 0.17 (the centre on the disc's side)
    0.51 + 0.17 = 0.68, less 0.01: 0.67. The box says 0.50 for all three. Regression caught: the review's case, the
    vacuum beside a wall converging on its side (gold 0.018 against 0.129)."""
    assert gold(VACUUM, (0.5, y)) == {"type": "number", "value": pytest.approx(want, abs=MM)}
    assert gold(BOX, (0.5, y)) == {"type": "number", "value": 0.5}


def test_an_obstacle_beside_the_front_arc_stops_the_disc_as_it_moves_on():
    """Contract: a sub-cell behind the front edge but outside the disc, beside its front arc, is touched as the disc
    moves ahead, so it stops the corridor (the strip begins at the edge and finds nothing there). Hand case: the point
    (-0.05, 0.15), centre (-0.05, 0.15): 0.12 ahead of the disc's centre (at x -0.17) and 0.15 to the side, 0.192 from
    it, outside the disc (r 0.17); touched after -0.05 + 0.17 - 0.08 = 0.04, so the gold is 0.03. The box finds nothing
    and runs to the cap (1.0). Regression caught: the corridor taken to start at the front edge for a disc."""
    assert gold(VACUUM, (-0.05, 0.15)) == {"type": "number", "value": pytest.approx(0.03, abs=MM)}
    assert gold(BOX, (-0.05, 0.15)) == {"type": "number", "value": 1.0}


def test_an_obstacle_outside_the_discs_width_or_inside_its_body_stops_nothing():
    """Contract: only a sub-cell the disc touches stops it. Hand cases: a point 0.19 to the side (centre 0.19, 0.02
    outside r 0.17) at any distance; a point in the disc's own place, (-0.1, 0.0) (centre (-0.09, 0.01), travel -0.09 +
    0.17 - 0.17 < 0: already under the body). Both leave the corridor free: the cap. Regression caught: a door frame
    beside the doorway stopping a round body, or its own footprint read as an obstacle."""
    assert gold(VACUUM, (0.5, 0.19)) == {"type": "number", "value": 1.0}
    assert gold(VACUUM, (-0.1, 0.0)) == {"type": "number", "value": 1.0}


def test_the_stop_is_the_obstacle_the_disc_touches_first_not_the_nearest_ahead():
    """Contract: of two obstacles the disc takes the one it touches first. Hand case: a point on the axis at x 0.62
    (centre (0.63, 0.01), touched after 0.63 + 0.17 - 0.1697 = 0.6303) and one at (0.5, 0.15) (centre (0.51, 0.15),
    touched after 0.60): the side one is nearer ahead (0.51 against 0.63) and touched first, so the gold is 0.59. With
    the side one at x 0.58 (centre 0.59, touched after 0.59 + 0.17 - 0.08 = 0.68) the axis one is first: 0.62.
    Regression caught: the nearest along taken as the stop, or the nearest to the axis."""
    assert gold(VACUUM, (0.62, 0.0), (0.5, 0.15))["value"] == pytest.approx(0.59, abs=MM)
    assert gold(VACUUM, (0.62, 0.0), (0.58, 0.15))["value"] == pytest.approx(0.62, abs=MM)


def test_a_disc_counts_the_cells_it_sweeps_and_not_the_rectangles_far_corners():
    """Contract: the corridor was seen empty only where the body sweeps. Hand case, cap 1.0 and blind 0: the floor free
    at u 1 to 9 in four rows (v -2 to 1, centres y -0.15, -0.05, 0.05, 0.15) and absent at u 0; the disc sweeps the
    rows at +-0.05 to the cap (centres 0.05 to 0.95: 20 cells) and the rows at +-0.15 only where d + 0.17 -
    sqrt(0.17^2 - 0.15^2) = d + 0.09 reaches the cap, d <= 0.91 (u 0 to 8: 18 cells): 38 counted, the box's 40. Missing
    (9, -2), (9, 1) (the rectangle's far corners, past the arc) and (1..3, -2): the disc has 31 of its 38 (81.6%): a
    number; the box 31 of 40 (77.5%): unknown. Regression caught: the rectangle's corners counted against a disc that
    never sweeps them."""
    states = {(u, v): ("free", 1) for u in range(1, 10) for v in range(-2, 2)}
    for gone in ((9, -2), (9, 1), (1, -2), (2, -2), (3, -2)):
        del states[gone]
    assert grid.corridor_clear(states, VACUUM, (0.0, 0.0), 0.0, 1.0, 0.0)[0] == {"type": "number", "value": 1.0}
    assert grid.corridor_clear(states, BOX, (0.0, 0.0), 0.0, 1.0, 0.0)[0] == {"type": "unknown"}


def test_the_raycast_is_the_golds_on_a_perfect_sensor_for_a_disc():
    """Contract: `grid.raycast` gives what the gold gives over the cells a perfect sensor would make of the rects, for a
    disc as for a box. Hand case: a wall piece x 1.0 to 1.1, y 0.1 to 1.0 beside the axis, the disc from the origin
    heading +x, cap 3: the nearest touched sub-cell is the one centred (1.01, 0.11): 1.01 + 0.17 - sqrt(0.17^2 - 0.11^2)
    = 1.18 - 0.1296 = 1.0504, less 0.01: 1.04 (the strip says 1.00). The same wall as a point in every sub-cell of the
    rect, seen by the gold's function, agrees. Regression caught: a map reader or a flier's wall corridor that keeps
    the strip for a round body while the gold does not."""
    wall = Rect(1.05, 0.55, 0.1, 0.9, 0.0)
    assert grid.raycast([wall], (0.0, 0.0), 0.0, VACUUM.body, 3.0) == pytest.approx(1.04, abs=MM)
    assert grid.raycast([wall], (0.0, 0.0), 0.0, BOX.body, 3.0) == pytest.approx(1.0, abs=MM)
    points = [(1.01 + 0.02 * i, 0.11 + 0.02 * j) for i in range(5) for j in range(45)]
    seen = grid.corridor_clear(scene(*points, cap=3.0), VACUUM, (0.0, 0.0), 0.0, 3.0, 0.0)[0]
    assert seen == {"type": "number", "value": pytest.approx(1.04, abs=MM)}


def test_a_wall_slanting_across_the_path_is_met_where_the_arc_reaches_it():
    """Contract: a straight wall converging on the body is first touched where the disc's arc meets it, not where the
    strip's corner does (the vacuum's wall-following: 8 of 29 golds 5 to 12 cm short). Hand case: the face of a wall
    4 m long and 0.1 m thick is the line y = x - 0.3 (45 degrees, through (0.3, 0)), its solid side below it; the
    robot's centre (c, 0) is 0.17 from the face when |c - 0.3| / sqrt(2) = 0.17, c = 0.3 - 0.2404 = 0.0596, and its
    front edge is 0.17 ahead of its centre's start at -0.17: it has moved 0.0596 + 0.17 = 0.2296. The strip's first
    point is the face at y = -0.17, x = 0.13, along 0.13. Both within a sub-cell (the gold is read at the 0.02 m
    sub-cell's centre, less 0.01). Regression caught: a round body given the strip's 0.12 for a wall that stops it
    after 0.23."""
    s = math.sqrt(0.5)
    wall = Rect(0.3 + 0.05 * s, -0.05 * s, 4.0, 0.1, math.radians(45))   # the face line, 0.05 m deeper at its centre
    assert grid.raycast([wall], (0.0, 0.0), 0.0, VACUUM.body, 3.0) == pytest.approx(0.2296, abs=0.03)
    assert grid.raycast([wall], (0.0, 0.0), 0.0, BOX.body, 3.0) == pytest.approx(0.12, abs=0.03)
