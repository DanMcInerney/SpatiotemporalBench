"""The reach bound of `way_to` (`route_field.max_outcome`, final review S5): a best heading whose outcome no drive can
make is a dead band, not a gold. Hand scenes of fine cells (`way_kit`) with the home helper (0.34 x 0.33 m, half width
0.165 m, so `distances` reaches 0.265 m past the floor); every expected value is worked out in its docstring."""

from __future__ import annotations

import math

import pytest

from stbench import grid, route_field
from way_kit import HELPER, ROOM, block, corridor, door_scene, latest

HALF = HELPER.body.width_m / 2
SLACK = HALF + grid.FINE_M   # how far `distances` reaches past the walkable floor


def walled(gap_cells: int) -> dict:
    """Two rooms (west x 0 to 3.0, east x 3.2 to 6.2, y 0 to 3.0) joined by a hall above them (y 3.0 to 4.0, x 0 to
    6.2), and the wall between them (x 3.0 to 3.2, y 0 to 3.0) cut by a gap of `gap_cells` fine cells from y 1.0."""
    free = block(0, 3.0, 0, 3.0) | block(3.2, 6.2, 0, 3.0) | block(0, 6.2, 3.0, 4.0)
    gap, wall = block(3.0, 3.2, 1.0, 1.0 + gap_cells / 10), block(3.0, 3.2, 0, 3.0)
    return dict.fromkeys(free | gap, "free") | dict.fromkeys(wall - gap, "occupied")


def asked(gap_cells: int):
    """`route` from the middle of the gap's row, 0.4 m before the wall, to the east room's (5.0, y), goal 0.4 m."""
    y = 1.0 + gap_cells / 20
    return route_field.route(latest(walled(gap_cells)), [[(0, 2.6, y, 0.0)]], HELPER, (2.6, y), 0.0, (5.0, y), 0.4,
                             1.0, ROOM)


def test_the_bound_is_the_octile_excess_per_metre_and_the_snapped_ends_over_the_shortest_drive():
    """Contract (final review S5): `max_outcome` is `reach * (L + (L sqrt 2 FINE + (L - 1) slack) / least)` with L =
    sqrt(4 - 2 sqrt 2) the 8-connected field's worst excess over the straight line (the worst angle is atan(sqrt 2 -
    1) = 22.5 degrees: cos + (sqrt 2 - 1) sin = 1.0824) and `least` the shortest drive an outcome is told over,
    `min(reach, room)`. By hand, the helper (R = 1 m, T = 0.49 m, FINE 0.1, slack 0.265): L = 1.082392;
    (1.082392 * 0.141421 + 0.082392 * 0.265) / 0.49 = (0.153074 + 0.021834) / 0.49 = 0.356956; the bound is 1.43935
    per reach. A reach of 2 m doubles the progress told over and keeps the per-metre part with the same least drive:
    2 * (1.082392 + 0.356956) = 2.87870. A reach under the room, 0.3 m, is told over 0.3 m: 0.3 * (1.082392 + 0.174908
    / 0.3) = 0.49963. Regression caught: a bound that forgets the metric's excess (1.0 per metre drops honest rows),
    one that forgets the snapped ends (a short drive), or one divided by the room when the reach is shorter."""
    assert route_field.LATTICE == pytest.approx(math.sqrt(4 - 2 * math.sqrt(2))) == pytest.approx(1.082392, abs=1e-6)
    assert route_field.max_outcome(1.0, 0.49, SLACK) == pytest.approx(1.43935, abs=1e-5)
    assert route_field.max_outcome(2.0, 0.49, SLACK) == pytest.approx(2.87870, abs=1e-5)
    assert route_field.max_outcome(0.3, 0.49, SLACK) == pytest.approx(0.49963, abs=1e-5)


def test_a_gap_the_erosion_closes_and_the_body_fits_is_a_dead_band_not_a_gold():
    """Contract (final review S5): where the eroded floor and the body's corridor disagree, the question is not asked.
    Hand scene `walled`: the wall's gap of 4 cells (0.4 m) holds the helper (0.33 m) but not its erosion (a cell is
    walkable only with 2 cells free on each side of an edge-on cell: 5 cells), so the gap is closed in the field
    (`geo` at x 3.0 and 3.2 is infinite) and the base at (2.6, 1.2) reaches the east room only by the hall: at least the
    straight lines (2.6, 1.2) to (3.1, 3.2) to (4.6, 1.2), 2.06 + 2.50 = 4.56 m. Driving east 1.0 m, the corridor is
    clear (a run of 1.0 m) and arrives at x 3.6 in the east room, 1.0 m from the goal set (x 4.6): a drive that would
    bring the robot at least 4.56 - 1.0 = 3.56 m nearer in 1.0 m, over the bound of 1.44, so no question. With a gap
    of 5 cells (0.5 m) the erosion leaves the gap open: the base is 2.0 m from the goal set (x 2.6 to 4.6), and driving
    east 1.0 m arrives 1.0 m from it: an outcome of exactly 1.0, under the bound, and asked. Regression caught: a gold
    whose best heading is judged against a number no drive can make (the 2.99 of case 2df70cca6f753c43), or a bound
    that drops an honest straight run."""
    states = latest(walled(4))
    foot, centre = route_field.swept([[(0, 2.6, 1.2, 0.0)]], HELPER)
    walk = route_field.walkable(states, foot, centre, HALF)
    field = route_field.distances(walk, route_field.goal_cells(walk, (5.0, 1.2), 0.4) & walk.mask, SLACK)
    assert math.isinf(field.at(3.0, 1.2)) and math.isinf(field.at(3.2, 1.2))
    assert field.at(2.6, 1.2) >= 4.56 and field.near_at(3.6, 1.2) == pytest.approx(1.0)
    assert field.at(2.6, 1.2) - field.near_at(3.6, 1.2) > 2 * route_field.max_outcome(1.0, ROOM, SLACK) > 2.8
    assert asked(4) is None
    open_gap = asked(5)
    assert open_gap is not None and open_gap.best == 0 and open_gap.profile[0] == pytest.approx(1.0)
    assert open_gap.distance_m == pytest.approx(2.0)


@pytest.mark.parametrize("scene", [door_scene(), corridor()])
def test_an_honest_way_stays_under_the_bound(scene):
    """Contract (final review S5): the bound sits above every outcome a consistent field gives, so the hand scenes of
    `test_route_field` still ask. Hand cases: `door_scene` (the best heading near 60 degrees, an outcome of at least
    0.95 and at most the bound 1.44 by the metric's 1.0824 at the worst angle) and `corridor` (exactly 1.0). Regression
    caught: a bound set so tight that the straight run, or the diagonal toward a door, is a dead band."""
    base = (2.0, 0.5) if scene == door_scene() else (1.05, 0.45)
    thing = (5.0, 0.5) if scene == door_scene() else (5.05, 0.45)
    goal = 0.4 if scene == door_scene() else 0.05
    found = route_field.route(latest(scene), [[(0, *base, 0.0)]], HELPER, base, 0.0, thing, goal, 1.0, ROOM)
    assert found is not None and 0.95 <= found.profile[found.best] <= route_field.max_outcome(1.0, ROOM, SLACK)
