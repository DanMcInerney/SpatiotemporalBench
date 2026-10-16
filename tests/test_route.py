"""The floor planner a job plans on (V0 §4): the free-space grid, A*, clearance, nudging, the obstacles a path never
enters and the polyline resampler. The tours of the body-free camera are gone; each job plans its own path."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from stbench import route
from stbench.geometry import Rect, distance, nearest_on_rect, points_in_polygon, segments_cross
from stbench.mapspec import check_map, parse_map
from stbench.robots import ROBOTS, over_m
from stbench.route import FreeSpace, blocking_rects

CLEARANCE = 0.25  # what these tests keep from every wall and piece (a world process's, `changes.WORLD_CLEARANCE_M`)


@pytest.fixture(scope="module")
def free(apt):
    return FreeSpace(apt, blocking_rects(apt), CLEARANCE)


def test_planner_goes_through_the_bedroom_door(apt, free):
    """Contract: A* from the hall to the bedroom passes the doorway: the polyline crosses the door line of D4 and is
    under 2.5 m long. Regression caught: a grid that lets a path cut through a wall (it never meets the door line), or
    that closes the doorway, so the path takes a long way round."""
    poly = free.path((5.4, 3.3), (7.3, 3.6))
    d4 = next(o for o in apt.openings if o.id == "D4")
    line = apt.door_line(d4)
    assert any(segments_cross(a, b, line[0], line[1]) for a, b in zip(poly, poly[1:], strict=False))
    assert sum(distance(a, b) for a, b in zip(poly, poly[1:], strict=False)) < 2.5


def test_planner_goes_straight_when_nothing_is_in_the_way(free):
    """Contract: in the empty hall the smoothed path is the straight 4 m segment, two vertices. Regression caught: a
    smoother that leaves the grid's staircase in the path, or a path that bends on free floor."""
    poly = free.path((5.4, 1.0), (5.4, 5.0))
    assert len(poly) == 2 and distance(poly[0], poly[1]) == 4.0


def test_planner_detours_around_furniture_keeping_clearance(free):
    """Contract: from west to east of the living-room coffee table (x 1.85-2.95, y 4.1-4.7) the path bends around it
    (more than 2.9 m, not the straight 2.4 m) and every vertex keeps 0.25 m from it. Regression caught: a path
    through the table (furniture not blocking the grid), or one that grazes it (the clearance not applied to
    furniture)."""
    poly = free.path((1.2, 4.4), (3.6, 4.4))
    table = Rect(2.4, 4.4, 0.6, 1.1, math.pi / 2)
    assert sum(distance(a, b) for a, b in zip(poly, poly[1:], strict=False)) > 2.4 + 0.5
    assert all(nearest_on_rect(table, *p)[1] >= 0.25 - 1e-9 for p in poly)


def test_the_clearance_is_the_callers_not_a_default(apt):
    """Contract: the grid keeps the clearance it is given from every wall and obstacle, so a robot asks for its own
    (`body.clearance_m` plus its radius) and a world process for the world's. Hand case: the hall's west wall face is
    x = 4.65, so the cell centred at x = 5.05 is 0.4 m from it: free at a 0.35 m clearance, not at a 0.45 m one.
    Regression caught: a default clearance shared by every robot (the vacuum's 0.19 m and the drone's 3 m planned
    alike)."""
    walls = blocking_rects(apt)
    assert FreeSpace(apt, walls, 0.35).is_free((5.05, 3.0))
    assert not FreeSpace(apt, walls, 0.45).is_free((5.05, 3.0))


def test_planner_refuses_blocked_or_outside_goals(free):
    """Contract: a goal inside the clearance of a wall, or outside the apartment, has no path (None). Regression
    caught: a planner that returns a path to a place the body cannot stand, so a robot is sent into a wall or off the
    map."""
    assert free.path((5.4, 1.0), (4.7, 1.0)) is None  # 0.05 m from the hall wall
    assert free.path((5.4, 1.0), (-2.0, 1.0)) is None


def test_nudge_restores_the_clearance(free):
    """Contract: a point 0.1 m from the kitchen east wall face (x = 4.55) is pushed to at least 0.25 m, moving about
    0.15 m, and a point already clear stays where it is. Regression caught: a nudge that leaves the point inside the
    clearance, jumps far from where it was, or moves a point that was already free."""
    moved = free.nudge((4.45, 1.0))
    d, _ = free.clearance(moved)
    assert d >= 0.25 and distance(moved, (4.45, 1.0)) < 0.2
    assert free.nudge((4.0, 1.0)) == (4.0, 1.0)


def test_each_room_is_reachable_from_the_hall(apt, free):
    """Contract: every room's anchor is reachable from the hall and at least 85% of each room's free floor is (small
    pockets behind furniture may be cut off). Regression caught: a map edit, or a larger clearance, that walls off a
    room or most of it, so no route reaches it."""
    reach = free.reachable_from(apt.region("R1").anchor)
    cx, cy = free.cells_xy(free.free)
    rx, ry = free.cells_xy(reach)
    for r in apt.regions:
        poly = [list(p) for p in r.polygon]
        assert reach[free.cell(r.anchor)[1], free.cell(r.anchor)[0]], r.name
        assert points_in_polygon(rx, ry, poly).sum() >= 0.85 * points_in_polygon(cx, cy, poly).sum(), r.name


def test_the_polygon_mask_is_the_regions_test_at_each_cells_centre(apt, free):
    """Contract: `FreeSpace.polygon_mask(polygon)` is True at the cells whose centres (`cells_xy`) `points_in_polygon`
    puts inside, for each room of the apartment, and each room holds free floor. Regression caught: grid axes
    offset by half a cell or swapped from the centres the rest of `FreeSpace` uses, so a room's mask would sit beside
    its free cells."""
    cx, cy = free.cells_xy(np.ones_like(free.free))
    for r in apt.regions:
        mask = free.polygon_mask(r.polygon)
        assert mask.shape == free.free.shape
        assert np.array_equal(mask.ravel(), points_in_polygon(cx, cy, [list(p) for p in r.polygon])), r.name
        assert (mask & free.free).any(), r.name


# -- what a flier passes over (generator 0.7.0, plan-city §2) --------------------------------------------------------
def slab_street():
    """Four 6 x 10 m regions in a row (24 x 10 m), walled round by 25 m walls, with a 1 x 6 m slab 0.15 m high standing
    across the second region at x 8.5-9.5, y 2-8: a sidewalk on a street."""
    walls = [("W1", (0, -0.3), (24, -0.3)), ("W2", (24.3, 0), (24.3, 10)), ("W3", (24, 10.3), (0, 10.3)),
             ("W4", (-0.3, 10), (-0.3, 0))]
    return parse_map({
        "name": "slab_street", "bounds": [-0.6, -0.6, 24.6, 10.6], "wall_thickness": 0.6,
        "wall_height": 25.0,
        "regions": [{"id": f"R{i + 1}", "name": f"block {i + 1}", "level": 0, "z0": 0.0, "z1": 25.0,
                     "polygon": [[6 * i, 0], [6 * i + 6, 0], [6 * i + 6, 10], [6 * i, 10]],
                     "anchor": [6 * i + 3.5, 5]} for i in range(4)],
        "walls": [{"id": i, "a": list(a), "b": list(b), "thickness": 0.6, "height": 25.0, "exterior": True}
                  for i, a, b in walls],
        "openings": [], "windows": [], "ceiling_lights": [],
        "furniture": [{"id": "F1", "name": "slab", "kind": "slab", "mobility": "fixed", "center": [9, 5],
                       "size": [1.0, 6.0, 0.15], "yaw": 0.0, "color": "grey", "blocks": True, "surfaces": []}],
    })


SLAB = Rect(9.0, 5.0, 1.0, 6.0, 0.0)


def test_blocking_rects_lets_a_flier_pass_over_what_is_lower():
    """Contract (plan-city §2): `blocking_rects(m, light, over)` keeps a blocking fixed piece only when its top is at
    `over` or above (a flier's `robots.over_m`, 9 m for the drone at 12 m; `robots.flies_over`), and the given light
    footprints as they come; `over` None keeps every blocking piece, as a ground robot does. Hand case: the 0.15 m
    slab is an obstacle for the ground and none for the drone, a 12 m pillar is one for both, and at the boundary a
    9.0 m piece obstructs and an 8.99 m one is flown over. Regression caught: the height rule dropped (a drone that
    routes round every sidewalk of the city), applied to a ground robot (a robot that walks over the furniture), or a
    boundary that differs from the validator's (a route through a piece that `validate` then fails)."""
    m = slab_street()
    assert check_map(m) == []
    pillar = replace(m.furniture[0], id="F2", center=(21.0, 5.0), size=(1.0, 1.0, 12.0))
    tall = replace(m, furniture=(*m.furniture, pillar))
    assert blocking_rects(m) == [SLAB] and blocking_rects(m, None, None) == [SLAB]
    assert over_m(ROBOTS["drone"], 12.0) == 9.0 and over_m(ROBOTS["vacuum"], 0.0) is None
    assert blocking_rects(m, None, over_m(ROBOTS["drone"], 12.0)) == []
    assert blocking_rects(tall, None, 9.0) == [pillar.footprint()]
    assert blocking_rects(tall) == [SLAB, pillar.footprint()]
    edge = replace(m.furniture[0], id="F3", center=(23.0, 5.0), size=(1.0, 1.0, 9.0))
    below = replace(m.furniture[0], id="F4", center=(25.0, 5.0), size=(1.0, 1.0, 8.99))
    assert blocking_rects(replace(m, furniture=(*m.furniture, edge, below)), None, 9.0) == [edge.footprint()]
    light = {"E0009": Rect(2.0, 2.0, 0.5, 0.5, 0.0)}
    assert blocking_rects(m, light, 9.0) == list(light.values())


def test_a_flier_plans_over_a_slab_a_ground_robot_goes_round():
    """Contract: on the same map the slab's 0.15 m height decides the grid: a ground robot's free space has no cell on
    the slab and a drone's (given `blocking_rects` with its `over`) has. Hand case: the slab covers x 8.5-9.5, y 2-8;
    its centre is blocked at 0.25 m clearance on the ground and free for the drone at 3 m clearance (the walls are 5 m
    away). Regression caught: a drone grid that treats the sidewalk as a wall."""
    m = slab_street()
    ground = FreeSpace(m, blocking_rects(m), CLEARANCE)
    flier = FreeSpace(m, blocking_rects(m, None, over_m(ROBOTS["drone"], 12.0)), 3.0)
    assert not ground.is_free((9.0, 5.0)) and flier.is_free((9.0, 5.0))
    assert ground.path((3.5, 5.0), (15.5, 5.0)) is not None and flier.path((3.5, 5.0), (15.5, 5.0)) == [
        (3.5, 5.0), (15.5, 5.0)]


def test_resample_places_a_keyframe_every_spacing_and_ends_on_the_path():
    """Contract (`route._resample`): a straight 20 m path at 4.5 m spacing gives keyframes at 4.5, 9, 13.5, 18 and the
    end (20), each facing along the path; at 0.75 m a 3 m path gives four. Regression caught: a spacing that is not the
    caller's, or a last hop dropped."""
    poses = route._resample([(0.0, 0.0), (20.0, 0.0)], 4.5)
    assert [round(p[0], 6) for p, _ in poses] == [4.5, 9.0, 13.5, 18.0, 20.0]
    assert all(p[1] == 0.0 and yaw == 0.0 for p, yaw in poses)
    assert [round(p[0], 6) for p, _ in route._resample([(0.0, 0.0), (3.0, 0.0)], 0.75)] == [0.75, 1.5, 2.25, 3.0]
