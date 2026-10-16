"""The camera's ground frame (`ego_frame.py`, 0.6.0): offsets, the slant range and the range bands. Expected values are
worked by hand."""

from __future__ import annotations

import math

import pytest

from stbench.ego_frame import band, from_offset, slant_m, to_offset


def test_ahead_is_along_the_heading_and_left_is_positive():
    """Contract: `to_offset` is `(ahead, left)` of a map point from the camera, the heading counter-clockwise from +x.
    By hand: at yaw 0 a point 3 m east and 1 m north is 3 ahead and 1 left; at yaw 90 (heading north) a point 3 m north
    is 3 ahead and 0 left, and one 2 m west is 0 ahead and 2 left (west is the left of north); at yaw 180 the same
    east point is 3 behind and 1 to the right; at yaw -90 (south) a point 2 m west is 2 to the right. Regression
    caught: a sign flip of the left axis or a swap of the axes (a thing on the right scored as on the left), or the
    heading read clockwise."""
    assert to_offset((5.0, 3.0), (2.0, 2.0), 0.0) == pytest.approx((3.0, 1.0))
    assert to_offset((2.0, 5.0), (2.0, 2.0), 90.0) == pytest.approx((3.0, 0.0))
    assert to_offset((0.0, 2.0), (2.0, 2.0), 90.0) == pytest.approx((0.0, 2.0))
    assert to_offset((5.0, 3.0), (2.0, 2.0), 180.0) == pytest.approx((-3.0, -1.0))
    assert to_offset((0.0, 2.0), (2.0, 2.0), -90.0) == pytest.approx((0.0, -2.0))
    assert to_offset((1.0, 1.0), (0.0, 0.0), 45.0) == pytest.approx((math.sqrt(2), 0.0))   # straight along the heading


@pytest.mark.parametrize("yaw", [0.0, 30.0, 90.0, 135.0, -170.0, 360.0])
def test_from_offset_inverts_to_offset(yaw):
    """Contract: `from_offset(to_offset(p, c, yaw), c, yaw) == p`, and the other way, at any heading. Regression caught:
    the two directions using different rotations (the sign of the sine terms), which agrees at yaw 0 and 180 and nowhere
    else."""
    point, cam = (7.5, -2.25), (1.0, 4.0)
    assert from_offset(to_offset(point, cam, yaw), cam, yaw) == pytest.approx(point)
    assert to_offset(from_offset((2.0, -3.0), cam, yaw), cam, yaw) == pytest.approx((2.0, -3.0))
    assert from_offset((3.0, 0.0), cam, 90.0) == pytest.approx((1.0, 7.0))   # 3 m ahead of a camera heading north


def test_slant_is_the_straight_line_distance_and_ignores_the_offset_frame():
    """Contract: `slant_m` is the 3-D distance between two (x, y, z) points. By hand: (0, 0, 0) to (3, 4, 12) is 13; a
    camera 15 m up and a thing 8 m along the floor and 6 m across is hypot(10, 15) = 18.03 from it. Regression caught:
    the horizontal range used where the slant belongs (the drone's rungs would be 40% too tight)."""
    assert slant_m((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)) == 13.0
    assert slant_m((5.0, 5.0, 15.0), (13.0, 11.0, 0.0)) == pytest.approx(math.hypot(10.0, 15.0))
    assert slant_m((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == 0.0


def test_the_range_bands_split_at_six_and_sixteen_units():
    """Contract (reports only): `band` is `near` under 6 place units, `far` from 16, `mid` between, whatever the unit.
    By hand: unit 0.5 m (a warehouse) edges at 3 m and 8 m; unit 1 m (the city) at 6 m and 16 m; a range exactly on an
    edge belongs to the upper band. Regression caught: an edge in metres instead of units (the same band for a home and
    a city), or an edge that goes to the lower band."""
    assert [band(r, 0.5) for r in (0.0, 2.99, 3.0, 7.99, 8.0, 40.0)] == ["near", "near", "mid", "mid", "far", "far"]
    assert [band(r, 1.0) for r in (5.99, 6.0, 15.99, 16.0)] == ["near", "mid", "mid", "far"]
    assert band(1.4, 0.25) == "near" and band(1.5, 0.25) == "mid" and band(4.0, 0.25) == "far"
