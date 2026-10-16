"""A ground robot's own obstacles and free space (`route.robot_obstacles`, `route.robot_free_space`): what `validate`'s
body rule keeps a base clear of, as the planner reads it. Every coordinate is worked out by hand from the apartment's
map (`maps/apartment_a.json`) and the vacuum's numbers (a 0.17 m disc keeping 0.02 m, climbing 0.02 m, 0.09 m high)."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest

from stbench.generate import build_world
from stbench.geometry import Rect, dist_to_rect
from stbench.mapspec import load_map
from stbench.robots import ROBOTS
from stbench.route import FreeSpace, robot_free_space, robot_obstacles

VACUUM = ROBOTS["vacuum"]


@cache
def apartment():
    return load_map("apartment_a")


@cache
def world(seed: int = 1):
    return build_world(apartment(), seed)


def has(rects: list[Rect], want: Rect) -> bool:
    return any(all(abs(a - b) < 1e-6 for a, b in zip(r, want, strict=True)) for r in rects)


def test_what_the_vacuum_passes_under_is_four_legs_and_what_it_cannot_is_whole():
    """Contract (plan §2.2, the under-furniture rule): the vacuum's obstacles are the four 0.15 m leg squares of a piece
    with an underside it passes under (0.10 m or more: the dining table's 0.64), the whole footprint of one it cannot
    (the sofa's 0.08), and nothing for a rug. Hand case: the dining table F07 at (2.1, 1.9), 0.8 x 1.4 m turned a
    quarter, has leg squares centred 0.325 m along its depth and 0.625 m along its width from the centre, at x 1.475
    and 2.725, y 1.575 and 2.225; the sofa is its own rectangle; the living rug at (2.4, 4.6) is no obstacle.
    Regression caught: a planner that drives through a table's legs or under a sofa, or round a rug."""
    obstacles = robot_obstacles(apartment(), world(), VACUUM)
    legs = [Rect(x, y, 0.15, 0.15, math.pi / 2) for x in (1.475, 2.725) for y in (1.575, 2.225)]
    assert all(has(obstacles, leg) for leg in legs)
    assert not has(obstacles, Rect(2.1, 1.9, 0.8, 1.4, math.pi / 2))  # not the table whole
    assert has(obstacles, Rect(2.4, 5.7, 0.9, 2.0, -math.pi / 2))  # the sofa
    assert not any(abs(r.cx - 2.4) < 1e-6 and abs(r.cy - 4.6) < 1e-6 for r in obstacles)  # the rug


def test_the_dock_and_the_door_leaves_are_obstacles_in_the_state_they_have():
    """Contract: the dock is an obstacle (the scripted undock and dock leave and reach it; the planner starts and ends
    in front of it), and each door's leaf is one in its state at the time asked. Hand cases: the dock at (4.71, 5.05),
    0.12 x 0.28; the shut entrance D0 (wall W01, centre 5.4, 0.95 wide) is a leaf across its opening, centred (5.4, 0);
    the kitchen door D1 (wall W05, centre 1.9, 0.85 wide, hinged at its south jamb, swinging west) is open at 0, a leaf
    from its hinge (4.6, 1.475) west to (3.75, 1.475), and shut from 5 s, a leaf across the opening centred (4.6, 1.9).
    Regression caught: a dock the planner drives through, or a leaf read in the wrong state (the open door of the
    kitchen vanishing, the shut entrance letting the vacuum out)."""
    m, w = apartment(), build_world(apartment(), 1)  # its own world: the test changes a door
    w.ledger.set_state(w.door_entity["D1"], "open", 5.0, False)
    thick = 0.04
    at0, at6 = robot_obstacles(m, w, VACUUM, 0.0), robot_obstacles(m, w, VACUUM, 6.0)
    assert has(at0, Rect(4.71, 5.05, 0.12, 0.28, 0.0))
    assert has(at0, Rect(5.4, 0.0, 0.95, thick, math.pi))
    assert has(at0, Rect(4.175, 1.475, 0.85, thick, math.pi)) and not has(at0, Rect(4.6, 1.9, 0.85, thick, math.pi / 2))
    assert has(at6, Rect(4.6, 1.9, 0.85, thick, math.pi / 2)) and not has(at6, Rect(4.175, 1.475, 0.85, thick, math.pi))


def test_items_on_the_floor_obstruct_above_the_climb_height_and_nothing_else_does():
    """Contract (the mobility table): a carried item or a piece of clutter obstructs when it is on the floor (its base
    under `FLOOR_ITEM_Z`, 0.1 m) and taller than the robot climbs (0.02 m): not a flat one, not one exactly 0.02 m high,
    not one on a table. Hand cases (clutter put in by hand): a 0.05 m box on the floor, a 0.01 m flat item, a 0.02 m
    item and a 0.05 m box 0.75 m up; and every carried item of the world by its own base and height. Regression caught:
    a vacuum that drives over a toy or stops for a coin, or a mug on a table that blocks the floor under it."""
    m, w = apartment(), build_world(apartment(), 1)
    w.clutter |= {"C1": (Rect(1.0, 1.0, 0.1, 0.1, 0.0), 0.0, 0.05), "C2": (Rect(1.5, 1.0, 0.1, 0.1, 0.0), 0.0, 0.01),
                  "C3": (Rect(2.0, 1.0, 0.1, 0.1, 0.0), 0.0, 0.02), "C4": (Rect(2.5, 1.0, 0.1, 0.1, 0.0), 0.75, 0.80)}
    obstacles = robot_obstacles(m, w, VACUUM)
    assert [has(obstacles, w.clutter[k][0]) for k in ("C1", "C2", "C3", "C4")] == [True, False, False, False]
    floor, raised = 0, 0
    for c in w.carried.values():
        (x, y, z), (sx, sy, sz) = c.spot.pos, c.spec.size
        on_floor = z < 0.1 and sz > 0.02
        assert has(obstacles, Rect(x, y, sx, sy, c.spot.yaw)) == on_floor, c.entity
        floor, raised = floor + on_floor, raised + (z >= 0.1)
    assert floor > 0 and raised > 0  # the world has both, so the loop above is not vacuous


def test_a_free_cell_keeps_the_bodys_clearance_over_its_whole_area():
    """Contract: a free cell of `robot_free_space` certifies every point of it, not only its centre: the grid's
    clearance is the body radius plus its clearance (0.17 + 0.02) plus the cell's half-diagonal (0.025 x sqrt 2 / 2 =
    0.017678). Hand cases: the hall's west wall face is x = 4.65, so the cell centred at x = 4.8625 (0.2125 m from it)
    is free and the one at 4.8375 (0.1875 m) is not; and every corner of every free cell within 0.3 m of a wall or
    piece is at least 0.19 m from it. Regression caught: a clearance that holds at cell centres only (a path that
    clips a corner of a leg), or a half-diagonal left out."""
    m, w = apartment(), world()
    free = robot_free_space(m, w, VACUUM)
    assert free.min_clearance == pytest.approx(0.207678, abs=1e-6)
    assert free.is_free((4.8625, 3.0)) and not free.is_free((4.8375, 3.0))
    ys, xs = np.nonzero(free.free)
    cx, cy = free.x0 + (xs + 0.5) * free.res, free.y0 + (ys + 0.5) * free.res
    near = np.min([dist_to_rect(cx, cy, r) for r in free.rects], axis=0) < 0.3
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        d = np.min([dist_to_rect(cx[near] + dx * free.res / 2, cy[near] + dy * free.res / 2, r) for r in free.rects],
                   axis=0)
        assert d.min() >= 0.19 - 1e-9


def test_a_rectangular_body_is_planned_as_its_circumscribed_disc():
    """Contract: a robot whose body is a rectangle (the tote robot, 0.80 x 0.58 m, keeping 0.10 m) is planned as the
    disc that holds it. Hand case: hypot(0.80, 0.58) / 2 = 0.494065, plus 0.10 and the half-diagonal 0.017678, is
    0.611743. Regression caught: a rectangle planned as a disc of its length (a path through a gap its width fits and
    its corners do not)."""
    free = robot_free_space(apartment(), world(), ROBOTS["amr"])
    assert free.min_clearance == pytest.approx(0.611743, abs=1e-6)


def test_a_shut_door_closes_the_way_and_an_open_one_does_not():
    """Contract: the free space follows the door's state: the bedroom's only door D4 (wall W06, centre 3.3, 0.85 wide)
    lets the vacuum from the hall to the bed's side when open and not when shut. Regression caught: a planner that
    drives through a shut door, or that cannot pass an open one."""
    m, hall, bedroom = apartment(), (5.4, 3.3), (7.0, 3.3)
    assert robot_free_space(m, world(), VACUUM).path(hall, bedroom, strict=True) is not None
    shut = build_world(m, 1)
    shut.ledger.set_state(shut.door_entity["D4"], "open", 0.0, False)
    assert robot_free_space(m, shut, VACUUM).path(hall, bedroom, strict=True) is None


def test_a_strict_path_never_clips_the_corner_of_a_blocked_cell():
    """Contract: `line_clear` samples a tenth of a cell and `line_free` a half, so a diagonal chord that clips only the
    corner of a blocked cell passes the second and fails the first; a `strict` path is smoothed with the first. Hand
    case: a 0.1 m square obstacle (the cell [1.0, 1.1] squared on a 0.1 m grid); the chord x + y = 2.18 from (0.85,
    1.33) to (1.33, 0.85) crosses its corner over x 1.08 to 1.1 (0.028 m long). Regression caught: a smoothed path that
    cuts corners by up to a cell, closer to an obstacle than the body's clearance."""
    free = FreeSpace(apartment(), [Rect(1.05, 1.05, 0.1, 0.1, 0.0)], 1e-9, res=0.1)
    a, b = (0.85, 1.33), (1.33, 0.85)
    assert free.line_free(a, b) and not free.line_clear(a, b)
    assert free.path(a, b) == [a, b]
    strict = free.path(a, b, strict=True)
    assert strict[0] == a and strict[-1] == b and len(strict) > 2
    assert all(free.line_clear(p, q) for p, q in zip(strict, strict[1:], strict=False))
