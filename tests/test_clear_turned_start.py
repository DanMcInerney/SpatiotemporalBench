"""Where the corridor of each move starts and what it counts (0.9.0, `clear_move`, `clear_ground.moment`), on
`clear_kit`'s tote AMR: 0.8 m long, 0.58 m wide (a box), base (0.6, 0) heading +x at the point, so its front edge is at
x 1.0, its rear edge at x 0.2 and its camera 0.38 m ahead of the base. The closed room (`room_run`) has its walls 2.01 m
ahead of the front edge, 2.41 behind the rear edge, 3.01 to the left and 2.11 to the right of the turned front edge."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from clear_kit import AMR, JOB, LAST, room_run, world
from stbench import clear_ground, clear_move
from stbench.robots import ROBOTS, blind_m

CAMERA = (0.98, 0.0)   # the AMR's camera at the point: 0.38 m ahead of the base


def gold_of(direction: str):
    m = clear_ground.moment(world(room_run(), [], replace(JOB, clear_directions=(direction,))), LAST)
    (cand,) = [*m.answerable, *m.unknown]
    return cand


@pytest.mark.parametrize("direction, value, start", [
    ("ahead", 2.0, (1.0, 0.0)), ("back", 2.4, (0.2, 0.0)), ("left", 2.6, (0.6, 0.4)), ("right", 2.1, (0.6, -0.4))])
def test_each_move_is_measured_from_its_own_edge(direction, value, start):
    """Contract (brief 4.1): the corridor starts at the body's edge for the move: the front edge ahead (x 1.0), the rear
    edge back with no turn (x 0.2, running west), and the front edge after turning in place by plus or minus 90 degrees
    (left: (0.6, 0.4) running north; right: (0.6, -0.4) running south). Hand values in the closed room: ahead the face
    at 3.01 is 2.01 from x 1.0 (2.00); back the face at -2.21 is 2.41 from x 0.2 (2.40); left the face at y 3.01 is 2.61
    from y 0.4 (2.60); right the face at y -2.51 is 2.11 from y -0.4 (2.10). `range_m` is the gold plus the distance
    from the camera (0.98, 0) to the start: 0.02, 0.78, 0.5517 and 0.5517. Regression caught: a corridor measured from
    the base (2.4, 2.8, 3.0, 2.5) or from the camera, or a turned corridor started at the rear edge or on the old
    heading."""
    c = gold_of(direction)
    assert c.gold == {"type": "number", "value": value}
    assert c.slots["direction"] == direction
    assert c.range_m == pytest.approx(value + math.dist(CAMERA, start), abs=1e-4)


def test_the_moved_start_and_heading_by_hand():
    """Contract: `clear_move.move` is the edge and the heading of each move for the delivered pose, turned with it. Hand
    values on the AMR (half length 0.4): at (0.6, 0) heading +x ahead (1.0, 0), back (0.2, 0) heading pi, left (0.6,
    0.4) heading pi/2, right (0.6, -0.4) heading -pi/2; the same base heading +y (pi/2): ahead (0.6, 0.4), back (0.6,
    -0.4), left (0.2, 0), right (1.0, 0). Regression caught: the turn applied the wrong way round (left and right
    swapped) or to the base instead of the heading."""
    north = {"ahead": (0.6, 0.4), "back": (0.6, -0.4), "left": (0.2, 0.0), "right": (1.0, 0.0)}
    for yaw, want in ((0.0, {"ahead": (1.0, 0.0), "back": (0.2, 0.0), "left": (0.6, 0.4), "right": (0.6, -0.4)}),
                       (math.pi / 2, north)):
        for direction, edge in want.items():
            start, heading = clear_move.move(AMR, (0.6, 0.0), yaw, direction)
            assert start == pytest.approx(edge, abs=1e-9) and heading == yaw + clear_move.TURN[direction]
    turns = {"ahead": 0.0, "back": math.pi, "left": math.pi / 2, "right": -math.pi / 2}
    assert turns == clear_move.TURN
    vacuum = ROBOTS["vacuum"]   # a disc: its edge is a radius (0.17) from the base on any heading
    assert clear_move.move(vacuum, (0.0, 0.0), 0.0, "left")[0] == pytest.approx((0.0, 0.17), abs=1e-9)


def test_only_a_corridor_ahead_is_blind_to_the_near_floor():
    """Contract (brief 4.1): ahead keeps 0.8.0's blind floor, `robots.blind_m` (0.53 m for the AMR: the camera, 0.5 m up
    and 10 degrees down, sees the floor from 0.5 / tan(32.2 + 10 degrees) - 0.02 = 0.53 m past the front edge); back,
    left and right count every cell they sweep. Regression caught: a blind patch dropped from a turned corridor that the
    robot never saw, or the near floor of an ahead corridor counted unseen."""
    assert clear_move.blind(AMR, "ahead") == blind_m(AMR) == pytest.approx(0.53, abs=0.01)
    assert [clear_move.blind(AMR, d) for d in ("back", "left", "right")] == [0.0, 0.0, 0.0]
