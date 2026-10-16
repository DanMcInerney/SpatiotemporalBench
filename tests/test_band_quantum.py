"""The 3 mm floor allowance (`grid.DEPTH_QUANTUM_M`): a surface within 3 mm of the robot's climb is floor to the
free-space rule, for every robot. Hand-worked on a floor at z 0 and points in one fine cell."""

from __future__ import annotations

import numpy as np
import pytest

from stbench import grid
from stbench.robots import ROBOTS

HEIGHTS_MM = {"vacuum": 20, "helper": 10, "amr": 10, "drone": 0}   # each robot's climb, millimetres, by hand


@pytest.mark.parametrize("robot", sorted(ROBOTS))
def test_three_millimetres_over_the_climb_is_floor_and_four_stops(robot):
    """Contract (review D3, the coordinator's call): points at the climb plus 3 mm make a fine cell `free` (four of them
    are floor), points at the climb plus 4 mm make it `occupied`, for each of `ROBOTS`. Hand values (climb in mm: vacuum
    20, helper 10, AMR 10, drone 0): the band's floor is 23, 13, 13 and 3 mm. Regression caught: a rug as high as the
    helper's climb stopping the helper (depth rounding lifts it a hair over), or an allowance wide enough to drive over
    a 1.5 cm cable (a band floor of 2 cm or more for the AMR)."""
    robot_ = ROBOTS[robot]
    assert round(robot_.body.climb_m * 1000) == HEIGHTS_MM[robot]
    climb = HEIGHTS_MM[robot] / 1000
    floor = grid.observe(np.array([(0.05, 0.05, climb + 0.003)] * 4), 0.0, grid.band(robot_))
    stops = grid.observe(np.array([(0.05, 0.05, climb + 0.004)] * 4), 0.0, grid.band(robot_))
    assert floor == {(0, 0): "free"} and stops == {(0, 0): "occupied"}


def test_a_rug_as_high_as_the_helpers_climb_does_not_stop_its_corridor():
    """Contract: the helper (climb 1.0 cm) drives over a 1.0 cm rug whose surface the depth image reads a few
    millimetres high, and is stopped by a 1.4 cm one. Hand scene: the helper's corridor (0.4 m wide) from the origin
    along +x, cap 1, floor points everywhere and a rug's top at z 0.012 across x 0.3 to 0.5 (twelve points in the cells
    u 3 and 4): the answer is the cap, 1.0; the same strip at 0.014 stops it at the first sub-cell of the strip, centred
    0.31: 0.30. Regression caught: the rug that stopped the helper at every home it crossed."""
    robot = ROBOTS["helper"]
    cells = {(u, v) for u in range(0, 10) for v in range(-3, 3)}

    def answer(z: float):
        pts = [(0.05 + 0.1 * u + 0.025 * i, 0.05 * j - 0.05, 0.0)
               for u in range(10) for i in range(2) for j in range(2)]
        pts += [(0.31 + 0.02 * i, 0.0, z) for i in range(10)]
        states = grid.latest({1: {c: "free" for c in cells}, 2: grid.observe(np.array(pts), 0.0, grid.band(robot))})
        return grid.corridor_clear(states, robot, (0.0, 0.0), 0.0, 1.0, 0.0)[0]

    assert answer(0.012) == {"type": "number", "value": 1.0}
    assert answer(0.014) == {"type": "number", "value": 0.3}
