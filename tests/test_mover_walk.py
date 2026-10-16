"""The Walker (`mover_walk`, the resident's walk model, ported from the T3 movers): its legs, its starts and its
waypoints, on the apartment's free floor with a person-sized clearance and a kind declared here by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stbench.catalog import MoverKind
from stbench.mover_walk import Walk, Walker
from stbench.route import FreeSpace, blocking_rects

KIND = MoverKind("walker", (0.30, 0.50, 1.70), (), (0.8, 1.2), (20.0, 120.0), 0.7, 0.8)
LIVING_ROOM_FLOOR = (4.0, 4.4)  # a free cell of the living room, away from the furniture (hand-read from the map)


@pytest.fixture(scope="module")
def free(apt):
    return FreeSpace(apt, blocking_rects(apt), 0.30, 0.05)


def walker(free, seed: int, kind: MoverKind = KIND) -> Walker:
    return Walker(kind, free, np.random.default_rng(seed))


def test_a_walk_waits_out_its_dwell_then_follows_its_path_at_its_speed(free):
    """Contract (`Walker.advance`): a leg is a dwell, then the path at the leg's one speed, then the next leg. Hand
    case: from (1, 1) with 2 s of dwell left, a path to (4, 1) and 0.5 m/s, 1 s and another 1 s leave it standing at
    (1, 1) with 0 s of dwell, 4 s more take it 2 m east to (3, 1) heading east, and 4 s more reach (4, 1) after 2 s and
    spend the other 2 s of that call in the next leg's dwell, which a kind draws from 20-120 s (so 18-118 s are left),
    with that leg's path already drawn.
    Regression caught: a speed read per keyframe instead of per second, a dwell skipped or counted twice, or a call that
    drops the time left when a leg ends."""
    w = walker(free, 1)
    st = Walk((1.0, 1.0), 0.0, 2.0, ((4.0, 1.0),), 0.5)
    st = w.advance(w.advance(st, 1.0), 1.0)
    assert st.pos == (1.0, 1.0) and st.dwell == pytest.approx(0.0)
    st = w.advance(st, 4.0)
    assert st.pos == pytest.approx((3.0, 1.0)) and st.yaw == pytest.approx(0.0)
    st = w.advance(st, 4.0)
    assert st.pos == pytest.approx((4.0, 1.0)) and 18.0 <= st.dwell <= 118.0 and len(st.path) >= 1


def test_each_leg_draws_its_dwell_and_its_speed_from_the_kinds_ranges(free):
    """Contract (`Walker._leg`; `MoverKind.speed` and `dwell`): a leg starting from rest draws a dwell in 20-120 s and
    a speed in 0.8-1.2 m/s, uniformly: over 200 legs every one is inside its range and the draws spread over most of
    it (both ends within 10% of their range's ends). Regression caught: a range read as one constant, as seconds per
    keyframe, or a speed drawn from the dwell's range."""
    w, rest = walker(free, 2), Walk(LIVING_ROOM_FLOOR, 0.0, 0.0, (), 0.0)
    dwells, speeds = [], []
    for _ in range(200):
        st = w.advance(rest, 0.5)
        dwells.append(st.dwell + 0.5)
        speeds.append(st.speed)
    assert 20.0 <= min(dwells) < 30.0 and 110.0 < max(dwells) <= 120.0
    assert 0.8 <= min(speeds) < 0.84 and 1.16 < max(speeds) <= 1.2


def test_advancing_in_pieces_is_advancing_at_once(free):
    """Contract (`Walker.advance`: the state is a value, so a caller that steps keyframe by keyframe walks the same
    walk): 300 s taken as one call, as 7 s calls and as 1 s calls, from walkers of one seed, end in one pose. Regression
    caught: a leg's progress lost at a call's end (a mover that stalls or jumps at every keyframe), or a draw made
    per call instead of per leg."""
    ends = []
    for dt in (300.0, 7.0, 1.0):
        w = walker(free, 3)
        st = Walk(LIVING_ROOM_FLOOR, 0.0, 0.0, (), 0.0)
        for _ in range(round(300 / dt)):
            st = w.advance(st, dt)
        ends.append(st)
    for st in ends[1:]:
        assert st.pos == pytest.approx(ends[0].pos, abs=1e-6) and st.yaw == pytest.approx(ends[0].yaw)
    assert math.dist(ends[0].pos, LIVING_ROOM_FLOOR) > 0.5  # and it did walk somewhere


def test_a_start_keeps_its_distance_from_each_keepout_and_none_when_no_cell_is_far_enough(free):
    """Contract (`Walker.start`): the start is a reachable free cell at least the given distance from every keepout
    point, and None when no cell is that far. Hand case: 4 m from the middle of the apartment (4.5, 3.1) and 1.5 m from
    the hall's dock end (5.0, 5.0), 200 starts all are; a keepout of 100 m leaves none. Regression caught: a start in
    the robot's first pose (the resident beside the dock), or a missing start taken for a cell."""
    w = walker(free, 4)
    for _ in range(200):
        st = w.start([((4.5, 3.1), 4.0), ((5.0, 5.0), 1.5)])
        assert math.dist(st.pos, (4.5, 3.1)) >= 4.0 and math.dist(st.pos, (5.0, 5.0)) >= 1.5
        assert free.is_free(st.pos)
    assert w.start([((4.5, 3.1), 100.0)]) is None


def test_waypoints_are_in_the_room_the_mover_stands_in_with_the_kinds_probability_and_never_in_a_doorway(free, apt):
    """Contract (`MoverKind.same_room`; `Walker._waypoint`): from a spot in the living room, the next waypoint is in the
    living room with probability 0.7 (600 draws: within 4 standard errors, sqrt(0.7 x 0.3 / 600) = 0.019) and otherwise
    in another room, never in a doorway (no room). Regression caught: the two cases swapped (a share of 0.3, a resident
    who leaves every room at once), or a waypoint on a doorway cell (a resident standing in a closing door)."""
    w, here = walker(free, 5), apt.region_of(LIVING_ROOM_FLOOR)
    drawn = [w._waypoint(LIVING_ROOM_FLOOR) for _ in range(600)]
    rooms = [apt.region_of(p) for p in drawn]
    assert None not in rooms
    share = sum(r == here for r in rooms) / 600
    assert abs(share - 0.7) < 4 * math.sqrt(0.7 * 0.3 / 600)
