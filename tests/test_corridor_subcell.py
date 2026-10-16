"""The corridor's stop is read at the obstacle's own 2 cm sub-cell (`grid.corridor_clear`, `grid.stop`), not at the near
edge of its 0.1 m fine cell. Hand scenes on the vacuum (0.34 m disc, half-width 0.17) from a start at the origin heading
+x with `blind` 0, the floor to the cap free and one obstacle point at the given distance, seen at keyframe 2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stbench import grid
from stbench.robots import ROBOTS

VACUUM = ROBOTS["vacuum"]
BAND = grid.band(VACUUM)


def scene(*points: tuple[float, float, float], cap: float = 1.0, heading_cells=None):
    """The latest states of a floor seen free to `cap` (keyframe 1) and the obstacle `points` (x, y, height above the
    floor at z 0) seen at keyframe 2: `observe` of the points puts each in its 0.1 m fine cell and its 0.02 m
    sub-cell."""
    free = {(u, v): "free" for u in range(0, round(cap * 10)) for v in range(-3, 3)}
    return grid.latest({1: free, 2: grid.observe(np.array(points).reshape(-1, 3), 0.0, BAND)})


@pytest.mark.parametrize("x, want", [(0.53, 0.52), (0.57, 0.56), (0.5, 0.5), (0.521, 0.52)])
def test_the_stop_is_the_obstacles_own_sub_cell(x, want):
    """Contract (review D1): an obstacle point at `x` along the corridor stops it at the centre of its sub-cell less
    half a sub-cell. Hand values: x 0.53 lies in the sub-cell [0.52, 0.54), centre 0.53, so 0.52; x 0.57 in [0.56,
    0.58), centre 0.57, so 0.56; x 0.50 in [0.50, 0.52), centre 0.51, so 0.50; x 0.521 in [0.52, 0.54), so 0.52. The 0.1
    m cell [0.5, 0.6) holds all four and its near edge, which 0.1 m cells gave, is 0.5: 1 cm to 6 cm short of the
    obstacle. Regression caught: the quantisation to the fine cell (a median 4 cm short, more than C/2 on a third of the
    vacuum's golds), or a sub-cell read off by one."""
    answer, evidence = grid.corridor_clear(scene((x, 0.0, 0.05)), VACUUM, (0.0, 0.0), 0.0, 1.0, 0.0)
    assert answer == {"type": "number", "value": want} and evidence == [1, 2]


def test_the_stop_names_the_fine_cell_that_holds_it():
    """Contract: `grid.stop` gives the sub-cell's distance and its fine cell, and `corridor_clear` counts the cells
    nearer than that cell. Hand values: the point (0.57, 0.0) is in sub-cell (28, 0), fine cell (5, 0), centred 0.57
    along and 0.01 left; the vacuum's disc touches it after 0.57 + 0.17 - sqrt(0.17^2 - 0.01^2) = 0.5703 (`stop.along`
    is that distance). Regression caught: the stopping cell taken from another sub-cell, which would put
    the stopper screen's frame (the keyframe that last observed it) on the wrong cell."""
    found = grid.stop(scene((0.57, 0.0, 0.05)), VACUUM, (0.0, 0.0), 0.0, 1.0)
    assert found is not None and found.cell == (5, 0) and found.along == pytest.approx(0.5703, abs=1e-4)
    assert grid.stop(scene(), VACUUM, (0.0, 0.0), 0.0, 1.0) is None


def test_a_point_outside_the_width_stops_nothing_and_a_bare_cell_stands_for_its_near_edge():
    """Contract: only a sub-cell whose centre lies in the body's width stops the corridor; a hand-built bare "occupied"
    cell stands for all 25 of its sub-cells, so its stop is its near sub-cell row's centre less half a sub-cell, which
    is the cell's near edge. Hand values: the point (0.57, 0.19) is in the sub-cell centred y 0.19, 0.02 outside the
    0.17 half-width, so the cap 1.0; the bare cell (5, 0) gives 0.5. Regression caught: a door frame beside the body
    stopping it, or a bare cell that no longer stops where its edge is."""
    assert grid.corridor_clear(scene((0.57, 0.19, 0.05)), VACUUM, (0.0, 0.0), 0.0, 1.0, 0.0)[0] == {
        "type": "number", "value": 1.0}
    bare = {**scene(), (5, 0): ("occupied", 2)}
    assert grid.corridor_clear(bare, VACUUM, (0.0, 0.0), 0.0, 1.0, 0.0)[0] == {"type": "number", "value": 0.5}


def test_the_stop_is_judged_along_the_heading():
    """Contract: the stop is the along-distance of the sub-cell's centre, whatever the heading. Hand case: heading 90
    degrees (north) from the origin, the point (0.0, 0.53) is in the sub-cell centred y 0.53: 0.52, the same as east.
    Regression caught: a stop measured along x only."""
    free = {(u, v): "free" for v in range(0, 10) for u in range(-3, 3)}
    states = grid.latest({1: free, 2: grid.observe(np.array([(0.0, 0.53, 0.05)]), 0.0, BAND)})
    answer = grid.corridor_clear(states, VACUUM, (0.0, 0.0), math.pi / 2, 1.0, 0.0)[0]
    assert answer == {"type": "number", "value": 0.52}
