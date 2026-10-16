"""The way to a remembered thing (`route_field`): the walkable floor, its distance field, and the outcome of every
heading. Hand scenes of fine cells (`way_kit`) with the home helper's body (0.34 x 0.33 m, front edge 0.17 m ahead of
its base, half width 0.165 m). Every expected value is worked out in its docstring."""

from __future__ import annotations

import math

import numpy as np

from stbench import route_field
from stbench.route_field import Walk
from way_kit import HELPER, REACH, ROOM, block, corridor, door_scene, latest

POSE = (0, 2.0, 0.5, 0.0)   # the helper at (2.0, 0.5) heading +x, in the west room of `door_scene`
THING = (5.0, 0.5)          # in the east room, behind the wall


def way(states, runs=(POSE,), base=(2.0, 0.5), thing=THING, goal=0.4, reach=REACH, room=ROOM):
    return route_field.route(latest(states), [list(runs)], HELPER, base, 0.0, thing, goal, reach, room)


def test_the_best_way_goes_through_the_door_not_at_the_thing():
    """Contract: the best heading follows the seen-free route, not the straight line to the thing. Hand scene
    (`door_scene`): the helper at (2.0, 0.5) facing +x, the thing at (5.0, 0.5) behind a wall whose only gap is the door
    at x 3.0 to 3.2, y 2.0 to 2.8. The door's middle (3.1, 2.4) lies 1.1 m east and 1.9 m north of the base: bearing
    atan2(1.9, 1.1) = 60 degrees, so the best heading is within 15 degrees of it (the 8-connected field leans toward the
    diagonal) and brings the robot a full metre nearer for a metre driven (at least 0.95). The heading at the thing, 0,
    runs 0.83 m to the wall (the sub-cell at 3.01, less the front edge 2.17 and half a sub-cell) and the door lies 2.2 m
    away: 0.83 m east shortens the way to the door by 2.20 - 1.92 = 0.28 m, an outcome of 0.28 / 0.83 = 0.33 (under
    0.5); the heading away from the door, 180, loses ground (below 0). Regression caught: a gold taken as the bearing to
    the thing (a heading through the wall), the straight-line distance for the field, or a field that ignores the
    wall."""
    found = way(door_scene())
    assert found is not None and len(found.profile) == 360
    assert 45 <= found.best <= 75 and found.profile[found.best] >= 0.95
    assert found.profile[0] < 0.5 and found.profile[found.best] - found.profile[0] > 0.5
    assert found.profile[180] < 0.0
    assert found.distance_m > 4.0   # the route is round by the door: more than the 3.0 m through the wall


def test_space_never_seen_is_not_a_route():
    """Contract (truth rule 2): a claim that floor is empty needs floor seen empty, so a route needs every cell of it
    seen free or driven. Hand cases on `door_scene`: with the door's cells unseen, the east room is cut off from the
    west and the question is not asked (None); with the east room unseen, the thing has no floor near it (the goal set
    within 0.4 m of (5.0, 0.5) holds no walkable cell) and it is not asked either. The same unseen door is a route once
    the robot has driven through it: poses from (5.0, 2.4) through (3.1, 2.4) and (2.0, 2.4) to the base, the footprint
    0.33 m wide along y 2.4 certifying the door's cells at keyframes 1 (the segment into the door) and the rooms' cells
    at 0. Regression caught: a gold through unseen cells, or driven floor not counted as seen."""
    assert way(door_scene(door=False)) is None
    east_unseen = {c: s for c, s in door_scene().items() if c[0] < 32}
    assert way(east_unseen) is None
    drove = [(0, 5.0, 2.4, math.pi), (1, 3.1, 2.4, math.pi), (2, 2.0, 2.4, math.pi), (3, 2.0, 0.5, 0.0)]
    found = way(door_scene(door=False), runs=drove)
    assert found is not None and 1 in found.ks and set(found.ks) <= {0, 1, 2}
    assert 45 <= found.best <= 75


def test_a_heading_without_room_to_move_is_null_and_so_is_one_into_floor_never_seen():
    """Contract: a heading whose run is under the room to move T, or that runs into floor never seen empty, has no
    outcome (null). Hand scene `door_scene`, the helper at (2.0, 0.5) facing +x: the run east is 0.83 m (see the first
    test), so it is a heading with room for T = 0.49 or 0.8 and null for T = 0.9; at (2.7, 0.5) the front edge is at
    2.87 and the run 3.01 - 2.87 - 0.01 = 0.13 m, null for the helper's T of 0.49. South (270) leaves the rooms: y below
    0 was never seen, so the corridor is not seen empty and the heading is null at any T. Regression caught: a heading
    into a wall with an outcome, a comparison of the run against T the wrong way, or floor never seen counted as
    empty."""
    assert way(door_scene(), room=0.8).profile[0] is not None
    assert way(door_scene(), room=0.9).profile[0] is None
    near = way(door_scene(), runs=[(0, 2.7, 0.5, 0.0)], base=(2.7, 0.5))
    assert near.profile[0] is None
    assert way(door_scene()).profile[270] is None


def test_an_outcome_is_the_metres_nearer_one_reach_of_driving_brings_the_robot():
    """Contract: the outcome of a heading is `reach_m * (geo(base) - geo(arrival)) / m`, m the run capped at the reach.
    Hand scene `corridor` (free x -2 to 6, y 0 to 0.8; the body keeps 0.25 m off the sides: walkable rows 2 to 5), the
    helper at (1.05, 0.45), the thing at (5.05, 0.45) with a goal of 0.05 m (one cell), so the base is 40 cells, 4.0 m,
    from the goal: driving east 1 m arrives at x 2.05, 30 cells, 3.0 m: outcome 1.0; driving west arrives at x 0.05,
    50 cells, 5.0 m: -1.0; north the corridor ends within 0.18 m of the body (the floor is seen to y 0.8 only), so it
    is not seen empty: null. A reach of 0.5 arrives at 1.55, 35 cells: 0.5 * (4.0 - 3.5) / 0.5 = 0.5. Headings 1 and
    -1 degree arrive in the same cell as 0 and so tie with it; the best is the one nearest ahead: 0. Regression caught:
    an outcome in the wrong unit (cells, or not per reach), the sign of progress, or a tie broken away from the robot's
    own heading."""
    def along(reach):
        return route_field.route(latest(corridor()), [[(0, 1.05, 0.45, 0.0)]], HELPER, (1.05, 0.45), 0.0, (5.05, 0.45),
                                 0.05, reach, ROOM)

    found, half = along(1.0), along(0.5)
    assert found is not None and found.distance_m == 4.0
    assert found.profile[0] == 1.0 and found.profile[180] == -1.0 and found.profile[90] is None
    assert found.profile[1] == found.profile[359] == 1.0 and found.best == 0
    assert half is not None and half.profile[0] == 0.5


def test_a_robot_within_a_reach_of_the_goal_is_asked_nothing():
    """Contract: the question is not asked when the robot is nearer the goal than one reach (driving a reach would pass
    it: `geo(base) < R`). Hand scene `corridor`, goal the cell at (5.05, 0.45): a base at x 4.55 is 5 cells, 0.5 m away
    (None); at x 4.05 it is 10 cells, 1.0 m, on the line, and is asked. Regression caught: a question about a thing the
    robot has all but reached (its outcomes capped by the goal), or the boundary excluded."""
    for x, asked in ((4.55, False), (4.05, True)):
        found = route_field.route(latest(corridor()), [[(0, x, 0.45, 0.0)]], HELPER, (x, 0.45), 0.0, (5.05, 0.45), 0.05,
                                  1.0, ROOM)
        assert (found is not None) is asked, x


def test_the_floor_is_eroded_by_half_the_body_so_a_gap_narrower_than_it_is_closed():
    """Contract: a cell is walkable only if the body fits on it: every cell the body touches standing on its centre is
    known. Half width 0.165 m reaches 2 cells out to the sides of an edge-on cell (the nearest point of a cell 2 away is
    0.15 m off) and 1 diagonally (0.158 m to the cell (2, 1) away, 0.212 to (2, 2)). Hand cases: a free block of 12 x 12
    cells keeps its inner 8 x 8 (cells 2 to 9); between two such blocks a gap of 3 cells (0.3 m, under the body's 0.33
    m) leaves no walkable cell in it, and a gap of 8 cells (0.8 m, a door) leaves 4 rows. Regression caught: a route
    claimed through a gap the body cannot pass, or an erosion by the cell instead of the body."""
    known = block(0, 1.2, 0, 1.2)
    walk = route_field.walkable(latest(dict.fromkeys(known, "free")), {}, {}, HELPER.body.width_m / 2)
    inner = {(u, v) for u in range(-5, 20) for v in range(-5, 20) if walk.has((u, v))}
    assert inner == block(0.2, 1.0, 0.2, 1.0)
    for rows, walkable in ((3, False), (8, True)):
        both = block(0, 1.0, 0, 2.0) | block(1.4, 2.4, 0, 2.0) | block(1.0, 1.4, 0.4, 0.4 + rows / 10)
        walk = route_field.walkable(latest(dict.fromkeys(both, "free")), {}, {}, HELPER.body.width_m / 2)
        door = [v for v in range(0, 20) if walk.has((12, v))]
        assert (len(door) == rows - 4) if walkable else not door


def test_the_field_is_the_octile_distance_in_metres_and_a_diagonal_does_not_cut_a_corner():
    """Contract: `geo` is the 8-connected shortest distance from the goal cells, 0.1 m a side step and 0.1 sqrt 2 a
    diagonal one, and a diagonal step needs both cells it passes beside. Hand cases on a 5 x 5 open mask with the goal
    at (0, 0): (3, 0) is 0.3, (3, 3) three diagonals 0.4243, (3, 1) two sides and a diagonal 0.3414. With the cell
    (1, 0) blocked in a 3 x 3 mask the diagonal from (0, 0) to (1, 1) is not allowed: (1, 1) is 0.2 (up and across),
    not 0.1414. A cell with no walkable neighbour is infinite. `near` reaches 0.2 m past the floor: with the walkable
    cells u 3 to 5 in a row and the goal at 3, u 5 is 0.2, u 6 is 0.2 + 0.1, u 7 is 0.2 + 0.2 and u 8 is out of reach.
    Regression caught: a distance in cells, a four-connected field, a path through a corner, or an arrival with no
    value against an obstacle."""
    pad = np.zeros((11, 11), dtype=bool)
    pad[3:8, 3:8] = True
    goal = np.zeros_like(pad)
    goal[3, 3] = True
    f = route_field.distances(Walk(0, 0, pad), goal, 0.0)
    assert math.isclose(f.geo[6, 3], 0.3) and math.isclose(f.geo[6, 6], 0.3 * math.sqrt(2))
    assert math.isclose(f.geo[6, 4], 0.2 + 0.1 * math.sqrt(2))
    cut = np.zeros((7, 7), dtype=bool)
    cut[2:5, 2:5] = True
    cut[3, 2] = False
    goal = np.zeros_like(cut)
    goal[2, 2] = True
    f = route_field.distances(Walk(0, 0, cut), goal, 0.0)
    assert math.isclose(f.geo[3, 3], 0.2) and math.isinf(f.geo[3, 2])
    row = np.zeros((12, 5), dtype=bool)
    row[3:6, 2] = True
    goal = np.zeros_like(row)
    goal[3, 2] = True
    f = route_field.distances(Walk(0, 0, row), goal, 0.2)
    assert [round(float(f.near[u, 2]), 3) for u in (5, 6, 7, 8)] == [0.2, 0.3, 0.4, math.inf]


def test_the_best_index_is_the_nearest_ahead_and_the_left_one_on_a_tie():
    """Contract: of the headings within a hair of the greatest outcome the one nearest the robot's own wins, the left
    one when two are as near, none when no heading has an outcome. Hand profile of 8 headings (45 degrees apart): the
    top 1.0 is at 2 (two steps left) and 6 (two steps right): 2. Regression caught: a tie broken to the right, to the
    farthest heading, or a null counted as the best."""
    assert route_field.best_index([0.2, None, 1.0, 0.3, 0.1, 0.3, 1.0, 0.5]) == 2
    assert route_field.best_index([None, 0.5, None, 0.5]) == 1
    assert route_field.best_index([None, None]) is None


def test_driven_floor_is_cut_where_the_robot_is_put_elsewhere():
    """Contract: the floor the body swept is the poses joined by straight segments within a task, never across two (the
    robot is put elsewhere between them). Hand case: task 0 drives (1.0, 1.0) to (1.5, 1.0), task 1 (5.0, 1.0) to (5.5,
    1.0): the fine cell at (3.0, 1.0) between them is in no footprint and no centre line, the cell at (1.2, 1.0) is in
    both, first certified by keyframe 1 (the pose that arrived there). `runs_of` splits the delivered rows at a change
    of `status.task`, in radians. Regression caught: a segment drawn from one task's end to the next one's start."""
    runs = [[(0, 1.0, 1.0, 0.0), (1, 1.5, 1.0, 0.0)], [(2, 5.0, 1.0, 0.0), (3, 5.5, 1.0, 0.0)]]
    footprint, centre = route_field.swept(runs, HELPER)
    assert (30, 10) not in footprint and (30, 10) not in centre
    assert footprint[(12, 10)] == 1 and centre[(12, 10)] == 1 and footprint[(52, 10)] == 3
    rows = [{"k": k, "pose": {"pos": [float(k), 0.0, 0.0], "yaw_deg": 90.0}, "status": {"task": t}}
            for k, t in ((0, 0), (1, 0), (2, 1))]
    assert route_field.runs_of(rows) == [[(0, 0.0, 0.0, math.pi / 2), (1, 1.0, 0.0, math.pi / 2)],
                                         [(2, 2.0, 0.0, math.pi / 2)]]
