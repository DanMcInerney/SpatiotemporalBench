"""The arithmetic the gold and the table systems share for the three harm families (`approach_rule`, `hazard_rule`).
Hand-worked on small hulls and points; the constants are the brief's (FD-questions-0.10 section 2) and each has its
source in the module."""

from __future__ import annotations

import math

import pytest

from stbench import approach_rule as ar
from stbench import hazard_rule as hr
from stbench.catalog import CATALOGS
from stbench.catalog_clutter import HOME_CLUTTER

SQUARE = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]


def test_the_centroid_is_the_area_centroid_or_the_mean_of_a_hull_with_no_area():
    """Contract (`approach_rule.centroid`): a hull's floor centroid is its area centroid, whichever way its vertices
    run, and the mean of the vertices when it has no area (a floor point, a segment). By hand: the square 0-2 by 0-2 has
    (1, 1); the triangle (0, 0), (3, 0), (0, 3) has (1, 1); a segment (0, 0)-(4, 2) has (2, 1); a point is itself.
    Regression caught: the mean of the vertices of a long hull (the triangle's would be the same, a rectangle with a
    collinear vertex's would not), or a crash on a degenerate hull."""
    assert ar.centroid(SQUARE) == pytest.approx((1.0, 1.0)) and ar.centroid(SQUARE[::-1]) == pytest.approx((1.0, 1.0))
    assert ar.centroid([[0, 0], [3, 0], [0, 3]]) == pytest.approx((1.0, 1.0))
    assert ar.centroid([[0, 0], [4, 2]]) == (2.0, 1.0) and ar.centroid([[3.3, 3.13]]) == (3.3, 3.13)
    assert ar.centroid([[0, 0], [1, 0], [3, 0], [4, 0]]) == (2.0, 0.0)
    with pytest.raises(ValueError):
        ar.centroid([])


def test_the_deadband_is_a_speed_under_0_3_m_s_and_a_missing_sighting_is_still():
    """Contract: a person's velocity is the centroid's displacement over the delivered seconds, zero under `V_STILL` =
    0.3 m/s and with no earlier sighting. By hand: 0.5 m in 2 s is 0.25 m/s, still; 0.7 m in 2 s is 0.35 m/s, kept; a
    moved hull over 0 s has no velocity. Regression caught: a deadband that swallows a walker or that never applies."""
    here = [[x + 1.0, y] for x, y in SQUARE]
    assert ar.hull_velocity(None, here, 2.0) == (0.0, 0.0) and ar.hull_velocity(SQUARE, here, 0.0) == (0.0, 0.0)
    assert ar.hull_velocity(SQUARE, [[x + 0.5, y] for x, y in SQUARE], 2.0) == (0.0, 0.0)
    assert ar.hull_velocity(SQUARE, [[x + 0.7, y] for x, y in SQUARE], 2.0) == pytest.approx((0.35, 0.0))


def test_the_earlier_sighting_is_the_latest_within_four_seconds_and_before_the_ask():
    """Contract (`VEL_S` = 4): of the earlier sightings, the one with the least positive age of at most 4 s. By hand:
    ages 4.0 (frame 3), 2.0 (frame 5), 0.0 (frame 6, the ask itself) and 4.5 (frame 7): frame 5; with only the 4.0 and
    the 4.5 one: frame 3 (the bound is inclusive); with none in range: None. Regression caught: the oldest sighting, the
    ask's own frame as its own earlier sighting, or a bound that is exclusive."""
    assert ar.latest_within([(3, 4.0), (5, 2.0), (6, 0.0), (7, 4.5)]) == 5
    assert ar.latest_within([(3, 4.0), (7, 4.5)]) == 3
    assert ar.latest_within([(7, 4.5)]) is None and ar.latest_within([]) is None


def test_the_clock_difference_is_the_delivered_one_and_ignores_the_local_clock_change():
    """Contract: seconds between two delivered clocks are the difference of the naive times, so a change of the local
    clock (a spring-forward night) is no part of it. By hand: 08:00:18 to 08:00:20 is 2 s; 01:59:00 to 03:00:00 on
    2026-03-08, a night the US clocks jump, is 3,660 s as written. Regression caught: a difference taken through a time
    zone."""
    assert ar.elapsed_s("2026-03-02T08:00:18", "2026-03-02T08:00:20") == 2.0
    assert ar.elapsed_s("2026-03-08T01:59:00", "2026-03-08T03:00:00") == 3660.0


def test_the_least_distance_over_the_horizon_is_sampled_every_tenth_of_a_second_and_capped():
    """Contract (`HORIZON_S` = 5, `STEP_S` = 0.1, `CAP_M` = 10): the least distance from the camera's floor point to the
    hull over [0, 5], both carried at constant velocity. By hand: a camera closing on a still point 5 m ahead at 1 m/s
    is 0 m at 5 s; at 0.5 m/s 2.5 m; the same point receding at 1 m/s stays 5 m (the least is the start); a point 20 m
    away is the cap; a camera on the hull is 0; a point 0.3 m beside the path with the gap closing at 0.7 m/s passes at
    tau = 2.9 s with the gap -0.03: hypot(0.3, 0.03) = 0.3015 (2.8 s: 0.3027). Regression caught: a horizon that ends
    early, a cap that is not applied, or the relative motion taken the wrong way round."""
    point = [[5.0, 0.0]]
    assert ar.closest_m((0, 0), (1.0, 0), point, (0, 0)) == pytest.approx(0.0, abs=1e-9)
    assert ar.closest_m((0, 0), (0.5, 0), point, (0, 0)) == pytest.approx(2.5)
    assert ar.closest_m((0, 0), (0, 0), point, (1.0, 0)) == 5.0
    assert ar.closest_m((0, 0), (0, 0), [[20.0, 0.0]], (0, 0)) == 10.0
    assert ar.closest_m((1.0, 1.0), (0, 0), SQUARE, (0, 0)) == 0.0
    assert ar.closest_m((0.3, 0.0), (0.0, 0.3), [[0.0, 2.0]], (0.0, -0.4)) == pytest.approx(math.hypot(0.3, 0.03))


def test_the_zone_is_4_m_and_the_reach_1_m_within_100_degrees_of_the_heading():
    """Contract (`ZONE_M`, `REACH_M`, `ARC_DEG`): a hull is in the zone when a point of it is within 4 m of the centre
    (inclusive), and in reach when its nearest point is within 1 m of the camera's floor point and at most 100 degrees
    from the heading (inclusive), whatever the heading when the camera point is on the hull. By hand: the square 0-2 has
    its nearest point to (-4, 1) at (0, 1), 4 m; to (-4.01, 1) 4.01 m. A point 1 m away at 100 degrees to the left of
    east is in reach, at 100.5 degrees or 1.01 m not; the hull holding the camera point is in reach facing any way.
    Regression caught: a zone or an arc that is exclusive at its edge, or a reach taken from the base."""
    assert hr.in_zone(SQUARE, (-4.0, 1.0)) and not hr.in_zone(SQUARE, (-4.01, 1.0))
    cam = (0.0, 0.0)

    def at(deg, d=1.0):
        return [[d * math.cos(math.radians(deg)), d * math.sin(math.radians(deg))]]

    assert hr.in_reach(at(100.0), cam, 0.0) and not hr.in_reach(at(100.5), cam, 0.0)
    assert hr.in_reach(at(-100.0), cam, 0.0) and hr.in_reach(at(60.0, 0.99), cam, 0.0)
    assert not hr.in_reach(at(60.0, 1.01), cam, 0.0)
    assert hr.in_reach(at(190.0), cam, math.radians(90.0)) and not hr.in_reach(at(191.0), cam, math.radians(90.0))
    assert all(hr.in_reach(SQUARE, (1.0, 1.0), math.radians(h)) for h in (0, 90, 180, 270))


def test_the_kinds_are_cords_and_cloth_that_exist_and_the_spot_kinds_are_left_out_of_the_gold():
    """Contract (FD-questions-0.10 S4): the kinds a brush catches are cords (extension cord, cable bundle, power strip)
    and cloth (bath mat, pedestal mat, clothes pile, the tracked towel), all kinds of the home catalog; the three that
    recur at one spot from script to script are `SPOT_KINDS` and out of the gold's `TANGLE_KINDS`, which is the rest.
    There is no sock: the catalog has none. Regression caught: a kind that the catalog never makes (a gold that never
    fires), a layout prior in the gold, or a kind in both lists."""
    clutter = {k.name for k in HOME_CLUTTER}
    assert set(hr.CATCH_KINDS) - {"towel"} <= clutter and "towel" in CATALOGS["home"].kinds
    assert "sock" not in clutter | set(CATALOGS["home"].kinds) | set(hr.CATCH_KINDS)
    assert set(hr.SPOT_KINDS) | set(hr.TANGLE_KINDS) == set(hr.CATCH_KINDS)
    assert not set(hr.SPOT_KINDS) & set(hr.TANGLE_KINDS)
