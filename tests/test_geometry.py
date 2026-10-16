"""The 2D geometry every module shares: distances to outlines, the under-furniture rule and the rotation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stbench.geometry import (
    LEG_SQUARE_M,
    Rect,
    dist_to_polygon,
    dist_to_rect,
    dist_to_segment,
    forward,
    grid_in_polygon,
    pitch_of,
    points_in_polygon,
    quat_yaw_pitch,
    quat_z,
    rect_distance,
    under_obstacles,
    yaw_of,
)

ROOM = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
# polygons over a 0.25 m grid (cell centres 0.125 + 0.25 k, 24 x 16 cells: 6 x 4 m), the vertices on cell centres where
# they can be, for `test_the_cropped_grid_mask_is_the_whole_grid_test_cut_to_the_polygons_box`
GRID_POLYGONS = {
    "concave L": [[0.625, 0.625], [4.125, 0.625], [4.125, 1.875], [2.375, 1.875], [2.375, 3.375], [0.625, 3.375]],
    "touching the grid's corner and edges": [[0.125, 0.125], [5.875, 0.125], [5.875, 3.875], [0.125, 3.875]],
    "partly outside the grid": [[-1.0, -0.5], [2.5, -0.5], [2.5, 1.0], [7.0, 2.0], [3.0, 5.0], [-1.0, 4.5]],
    "a triangle over the top edge": [[2.0, 3.875], [4.0, 3.875], [3.0, 5.0]],
    "wholly outside the grid": [[7.0, 7.0], [9.0, 7.0], [8.0, 9.0]],
    "thinner than a cell": [[1.0, 1.01], [3.0, 1.01], [3.0, 1.02], [1.0, 1.02]],
}


def test_distance_to_a_segment_is_to_its_nearest_point_whether_on_it_or_beyond_an_end():
    """Contract: the distance from a point to the closed segment (0,0)-(4,0) is the perpendicular distance when the
    point projects onto the segment (3 above x = 2 is 3), the distance to the nearer end when it projects beyond one
    ((7, 4) is 5 from (4, 0); (-3, 4) is 5 from (0, 0)), and the distance to the point itself for a segment of no length
    ((3, 4) from (0, 0) is 5). Regression caught: the distance to the infinite line (0 for (7, 0)) or to the nearer
    endpoint only."""
    a, b = (0.0, 0.0), (4.0, 0.0)
    assert dist_to_segment(2, 3, a, b) == 3.0 and dist_to_segment(7, 4, a, b) == 5.0
    assert dist_to_segment(-3, 4, a, b) == 5.0 and dist_to_segment(7, 0, a, b) == 3.0
    assert dist_to_segment(3, 4, a, a) == 5.0


@pytest.mark.parametrize(("x", "y", "expected"), [
    (1.0, 1.0, 0.0),                # inside: 0, not the distance to the wall
    (2.0, -0.5, 0.5),               # below the bottom edge
    (4.19, 1.5, 0.19),              # right of the right edge
    (5.0, 4.0, math.sqrt(2)),       # off the (4, 3) corner: the corner is the nearest point
    (-1.0, -1.0, math.sqrt(2)),     # off the (0, 0) corner
])
def test_distance_to_a_polygon_is_zero_inside_and_the_distance_to_the_outline_outside(x, y, expected):
    """Contract: `dist_to_polygon` is 0 for a point inside (`point_in_polygon`) and else the distance to the nearest
    edge or corner of the 4 x 3 room. Regression caught: the distance to the edge for a point inside, or to a corner
    for a point beside an edge."""
    assert dist_to_polygon(x, y, ROOM) == pytest.approx(expected)


def test_rotations_by_hand():
    """Contract (the helpers moved here from `timeline`, unchanged): qz(90 deg) = [cos 45, 0, 0, sin 45] and `yaw_of`
    recovers an angle; yaw 0 with pitch +90 deg looks straight up, yaw 90 deg with pitch 0 looks north, and a pitch up
    is positive. Regression caught: the helpers changing in the move (a flipped pitch sign, a yaw read from the wrong
    axis), which would turn every camera and every heading."""
    w, x, y, z = quat_z(math.pi / 2)
    assert (round(w, 6), x, y, round(z, 6)) == (0.707107, 0.0, 0.0, 0.707107)
    assert yaw_of(quat_z(2.0)) == pytest.approx(2.0, abs=1e-5)
    fx, fy, fz = forward(quat_yaw_pitch(0.0, math.pi / 2))
    assert (round(fx, 5), round(fy, 5), round(fz, 5)) == (0.0, 0.0, 1.0)
    fx, fy, fz = forward(quat_yaw_pitch(math.pi / 2, 0.0))
    assert (round(fx, 5), round(fy, 5), round(fz, 5)) == (0.0, 1.0, 0.0)
    rot = quat_yaw_pitch(0.7, math.radians(-15))
    assert yaw_of(rot) == pytest.approx(0.7, abs=1e-5)
    assert pitch_of(rot) == pytest.approx(math.radians(-15), abs=1e-5)


def test_legs_obstruct_and_undersides_do_not():
    """Contract (plan §2.2, "Under-furniture"): a piece with an open underside (under_m > 0) obstructs at every height
    only by four `LEG_SQUARE_M` squares in its corners, inside it, at its yaw; a piece solid to the floor (under_m 0)
    is its whole rect. Hand case: a 1.4 x 0.8 table gives squares of 0.15 m centred (+-0.625, +-0.325) (0.7 - 0.075,
    0.4 - 0.075); a 0.17 m disc at its middle is 0.604 m from the nearest leg (the corner (0.55, 0.25): the square root
    of 0.3025 + 0.0625), so it clears by 0.604 - 0.17; one at (0.6, 0.3) is in a leg. Regression caught: a planner
    driving through table legs, or a table's middle blocked."""
    table = Rect(0.0, 0.0, 1.4, 0.8, 0.0)
    legs = under_obstacles(table, 0.64)
    assert LEG_SQUARE_M == 0.15
    assert sorted((round(r.cx, 6), round(r.cy, 6)) for r in legs) == [
        (-0.625, -0.325), (-0.625, 0.325), (0.625, -0.325), (0.625, 0.325)]
    assert {(r.sx, r.sy, r.yaw) for r in legs} == {(0.15, 0.15, 0.0)}

    def to_legs(x: float, y: float) -> float:
        return min(float(dist_to_rect(np.array([x]), np.array([y]), leg)[0]) for leg in legs)

    assert to_legs(0.0, 0.0) == pytest.approx(math.sqrt(0.365), abs=1e-9)
    assert to_legs(0.0, 0.0) - 0.17 > 0.43  # 0.604 - 0.17 = 0.434
    assert to_legs(0.6, 0.3) == 0.0
    assert under_obstacles(table, 0.0) == [table]


def test_legs_turn_with_the_piece_and_stay_inside_it():
    """Contract: the leg squares are in the piece's own frame, so a table turned a quarter has them turned with it, and
    a piece shorter than a leg square on one side gets a shorter leg there, never one poking out. Hand case: the 1.4 x
    0.8 table at yaw 90 deg at (1, 2) has its legs at (1 -+ 0.325, 2 +- 0.625); a 0.10 x 0.8 piece has legs 0.10 wide.
    Regression caught: legs left on the map axes under a turned piece (a robot blocked in the open, or let through a
    leg)."""
    turned = under_obstacles(Rect(1.0, 2.0, 1.4, 0.8, math.pi / 2), 0.5)
    assert sorted((round(r.cx, 6), round(r.cy, 6)) for r in turned) == [
        (0.675, 1.375), (0.675, 2.625), (1.325, 1.375), (1.325, 2.625)]
    thin = under_obstacles(Rect(0.0, 0.0, 0.10, 0.8, 0.0), 0.5)
    assert {(r.sx, r.sy) for r in thin} == {(0.10, 0.15)}
    assert max(abs(r.cx) + r.sx / 2 for r in thin) == pytest.approx(0.05)
    assert max(abs(r.cy) + r.sy / 2 for r in thin) == pytest.approx(0.4)


def test_rect_distance_by_hand():
    """Contract: `rect_distance` is the least distance between two oriented rectangles, 0 when they overlap or touch.
    Hand cases: unit squares centred at (0, 0) and (2, 0) are 1.0 apart; the second turned 45 degrees (a diamond whose
    near corner is sqrt(2)/2 from its centre) is 1.5 - sqrt(2)/2 = 0.7929; overlapping squares, a plus-shaped crossing
    (no corner of either inside the other) and touching ones are 0. Regression caught: a body rule that only
    prefilters (an overlap test with a margin per axis) or that misses a crossing."""
    unit = Rect(0.0, 0.0, 1.0, 1.0, 0.0)
    assert rect_distance(unit, Rect(2.0, 0.0, 1.0, 1.0, 0.0)) == pytest.approx(1.0)
    assert rect_distance(unit, Rect(2.0, 0.0, 1.0, 1.0, math.pi / 4)) == pytest.approx(1.5 - math.sqrt(2) / 2)
    assert rect_distance(unit, Rect(0.5, 0.5, 1.0, 1.0, 0.0)) == 0.0
    assert rect_distance(Rect(0.0, 0.0, 3.0, 0.2, 0.0), Rect(0.0, 0.0, 0.2, 3.0, 0.0)) == 0.0
    assert rect_distance(unit, Rect(1.0, 0.0, 1.0, 1.0, 0.0)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", GRID_POLYGONS)
def test_the_cropped_grid_mask_is_the_whole_grid_test_cut_to_the_polygons_box(name):
    """Contract: `grid_in_polygon(xs, ys, poly)` is the mask `points_in_polygon` gives over the whole meshgrid, though
    it tests only the cells in the polygon's bounding box: for a concave polygon, one touching the grid's corner and
    edges, one partly or wholly outside the grid, and one thinner than a cell (an empty mask). Regression caught: a
    crop that drops the cells on the box's edge (a vertex on a cell centre, or the polygon along the grid's last row),
    or a box that clips the grid wrongly where the polygon hangs over it."""
    xs, ys = 0.125 + 0.25 * np.arange(24), 0.125 + 0.25 * np.arange(16)
    gx, gy = np.meshgrid(xs, ys)
    poly = GRID_POLYGONS[name]
    cropped = grid_in_polygon(xs, ys, poly)
    assert cropped.shape == gx.shape
    assert np.array_equal(cropped, points_in_polygon(gx, gy, poly))
    assert cropped.any() == (name not in ("wholly outside the grid", "thinner than a cell"))


def test_the_cropped_grid_mask_keeps_the_cells_on_the_low_edge_of_the_box_and_drops_the_high_ones():
    """Contract (`points_in_polygon`'s ray rule): over the integer grid 0..4, the square from (1, 1) to (3, 3) holds the
    cells (1, 1), (2, 1), (1, 2) and (2, 2): a cell on its low edges is inside, one on its high edges outside, and the
    cropped mask says the same. Regression caught: a box cut half-open the wrong way round, which would lose the cells
    on the low edges or take in those on the high ones."""
    axis = np.arange(5.0)
    mask = grid_in_polygon(axis, axis, [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
    assert sorted(zip(*(a.tolist() for a in np.nonzero(mask)), strict=True)) == [(1, 1), (1, 2), (2, 1), (2, 2)]
