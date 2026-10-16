"""`World.camera_at`, `middle` and `slant` (0.6.0): the camera of a keyframe as the pose row delivers it, and the
range from it to a thing. Worked by hand on the vacuum, whose camera is mounted 0.13 m ahead of the base and 0.08 m
above it."""

from __future__ import annotations

import math

import pytest

from family_kit import camera_of, multi_run, world
from stbench.rules import relative_ladder

# Keyframe 4 (task 0, a transit leg): the base at (3.0, 1.0) heading north (90 degrees), so the camera is 0.13 m north
# of it at (3.0, 1.13), 0.08 m up. A thing 0.4 m tall stands at (3.0, 4.13): its middle is 0.2 m up, 3.0 m ahead of the
# camera along the floor and 0.12 m above it.
K, THING_XY, SIZE = 4, (3.0, 4.13), (0.1, 0.1, 0.4)


@pytest.fixture(scope="module")
def w():
    s = multi_run(1, poses={K: (3.0, 1.0)}, yaws={K: math.pi / 2})
    s.item("T", "bag", {"color": "red"}, [(0, None, (*THING_XY, 0.0))], size=SIZE)
    s.item("W", "bag", {"color": "blue"}, [(0, None, (1.0, 1.13, 0.0))], size=SIZE)    # 2 m west of the camera
    s.item("L", "bag", {"color": "green"}, [(300.0, None, (5.0, 1.0, 0.0))], size=SIZE)   # exists only from t = 300 s
    return world(s)


def test_the_camera_is_the_delivered_row_not_the_base(w):
    """Contract: `camera_at(k)` is the pose row's camera position and the base's heading in degrees. By hand: the base
    at (3.0, 1.0, 0.0) heading 90 degrees puts the camera at (3.0, 1.13, 0.08). Regression caught: the base used as the
    camera (every range 0.13 m and every height 0.08 m off, and on a drone, 12 m)."""
    pos, yaw = w.camera_at(K)
    assert pos == pytest.approx((3.0, 1.13, 0.08), abs=1e-3) and yaw == pytest.approx(90.0)
    assert (pos, yaw) == camera_of(w, K) and w.base(K) == (3.0, 1.0)


def test_the_middle_of_a_thing_is_half_its_height_above_its_base(w):
    """Contract: `middle(e, k)` is the placement's base centre and half the entity's height above its base, None when it
    does not exist then. By hand: a 0.4 m thing at (3.0, 4.13) standing on the floor has its middle at (3.0, 4.13,
    0.2); one that exists only from t = 300 s has none at keyframe 4. Regression caught: the base height returned for
    the middle (a drone's range would lose half the thing), or a thing placed before it exists."""
    assert w.middle(w.tl.entity("T"), K) == pytest.approx((3.0, 4.13, 0.2))
    assert w.middle(w.tl.entity("L"), K) is None


def test_slant_runs_from_the_camera_not_the_base(w):
    """Contract: `slant(e, k)` is the 3-D distance from the delivered camera to the thing's middle, None when it does
    not exist. By hand: 3.0 m along the floor and 0.12 m up is sqrt(9 + 0.0144) = 3.00240 m; the same thing from the
    base (0.13 m farther, 0.2 m up) would be 3.13 m horizontally. The rungs it feeds are relative_ladder(3.0024) =
    0.30024, 0.75060 and 1.50120. Regression caught: the range measured from the base (a drone's off by its whole
    mount) or the horizontal distance, and so the wrong rungs."""
    slant = w.slant(w.tl.entity("T"), K)
    assert slant == pytest.approx(math.sqrt(9.0 + 0.12 ** 2), abs=2e-3)
    assert abs(slant - math.hypot(3.13, 0.2)) > 0.1   # the base-to-middle range is a different number
    assert relative_ladder(slant) == pytest.approx([0.30024, 0.75060, 1.50120], abs=3e-4)
    assert w.slant(w.tl.entity("L"), K) is None
