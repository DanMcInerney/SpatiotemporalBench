"""Where a `clear_path` corridor starts and which way it runs, for every move a robot can ask about (0.9.0,
docs/families/clear_path.md): the one place the compiler's gold, the table systems, `path_log` and the map reader all
get their start and heading from, so none of them can read another edge.

The robot moves straight along a heading, and the corridor starts at the body's edge for that move:
- **ahead:** the front edge, on the delivered heading;
- **back:** the rear edge, on the heading's opposite, with no turn;
- **left, right:** the robot first turns in place by plus or minus 90 degrees (left is positive, as an offset's left
  is), so the corridor starts at the front edge on the turned heading.

`grid.front_edge` of the moved heading is all of them: the base is the middle of the body, and the front edge on a
heading is half the body's length from it along that heading. The camera and the base are never the start.

A corridor ahead is blind to the floor the camera cannot see in front of the body (`robots.blind_m`); every other move
counts every cell it sweeps (the robot saw that floor before it moved on, or it did not). The drawn direction is a hash
of the case and the point, which reads nothing of the world. The module is neutral: `grid`, `robots` and `seeding`
alone, so a table system or a baseline may import it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import grid
from .robots import Robot, blind_m
from .seeding import stable_hash

TURN = {"ahead": 0.0, "back": math.pi, "left": math.pi / 2, "right": -math.pi / 2}   # the heading's turn, radians
TURNED = ("left", "right")   # the moves that first turn the body in place


def heading_of(yaw: float, direction: str) -> float:
    """The heading the corridor runs along: the delivered `yaw` (radians) turned for `direction`. KeyError for a
    direction that is not one of `TURN`."""
    return yaw + TURN[direction]


def move(robot: Robot, base: Sequence[float], yaw: float, direction: str) -> tuple[tuple[float, float], float]:
    """`(start, heading)` of the corridor `direction` from the base (x, y) at the delivered `yaw` (radians): the body's
    edge for that move and the heading it runs along."""
    heading = heading_of(yaw, direction)
    return grid.front_edge(robot, base, heading), heading


def blind(robot: Robot, direction: str) -> float:
    """The floor not counted at the start of the corridor: `robots.blind_m` ahead, 0 for every other move."""
    return blind_m(robot) if direction == "ahead" else 0.0


def draw(case: str, k: int, directions: Sequence[str]) -> str:
    """The direction asked at look point `k` of `case`: `stable_hash(case, k, "direction") % len(directions)` over the
    job's declared directions, each equally. It reads nothing of the world, so the direction tells nothing of what is
    there."""
    return directions[stable_hash(case, k, "direction") % len(directions)]
