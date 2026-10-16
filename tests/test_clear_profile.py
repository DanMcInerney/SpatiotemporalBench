"""`clear_profile.profile`, the corridor's run at every heading turned to in place (`open_heading`'s profile), on a hand
scene of fine-cell states (the `space_kit` conventions: 0.1 m fine cells, 0.02 m sub-cells, a cell's state `free`, or an
`Occupied` naming its obstacle sub-cells). The base is (0, 0) heading +x, free floor seen to 6 m all round except the
north-east quarter beyond (1.1, 1.1); a wall 2 cm deep with its centre line at x 2.01 across y -1..1; one at y 3.01
across x -1..1; and a pole in the sub-cell at (0.25, -2.01). The tote AMR is a box 0.80 x 0.58 m (front edge at 0.4 m,
half width 0.29, cap 5 m), the vacuum a disc 0.34 m across (front edge at 0.17 m, cap 3 m)."""

from __future__ import annotations

import math
import time

import pytest

from heading_kit import AMR, BASE, VACUUM, YAW, scene, sub
from stbench import clear_profile, grid


@pytest.mark.parametrize(("robot", "cap", "ahead", "left", "back", "right"), [
    (AMR, 5.0, 1.6, 2.6, 5.0, 1.6),
    (VACUUM, 3.0, 1.83, 2.83, 3.0, 3.0),
])
def test_the_profile_holds_the_hand_run_at_each_side_and_none_where_unseen(robot, cap, ahead, left, back, right):
    """Contract: the profile's run at a heading is how far the body, turned to it, goes from its edge before the nearest
    obstacle sub-cell its width sweeps, to half a sub-cell short of that sub-cell's centre, the cap when nothing is
    swept; a heading whose corridor was not seen empty is None. Hand values, edge at 0.4 m (AMR) or 0.17 m (vacuum):
    ahead the wall's centre line is at 2.01 m: 2.01 - 0.4 - 0.01 = 1.6 (the disc: 2.01 - 0.17 - 0.01 = 1.83); left the
    wall at 3.01 m: 2.6 (2.83); behind nothing stands, so the cap; right the pole at 0.25 m off the axis is swept by the
    box (half width 0.29) at 1.6 and missed by the disc (half width 0.17): the cap. At 45 degrees the corridor runs into
    the unseen quarter and is None. Regression caught: a run from the base or the camera, a width that stops on what
    stands outside the body, a corridor over unseen floor called clear."""
    runs = clear_profile.profile(scene(), robot, BASE, YAW, cap)
    assert len(runs) == 360
    assert (runs[0], runs[90], runs[180], runs[270]) == (ahead, left, back, right)
    assert runs[45] is None


@pytest.mark.parametrize("robot", [AMR, VACUUM])
def test_the_profile_equals_the_corridor_gold_at_every_sampled_heading(robot):
    """Contract: the vectorised profile is `grid.corridor_clear` (the gold of a `clear_path` question turned to that
    heading, no blind band) from the turned body's edge, to the last digit and with `unknown` as None, at the headings
    sampled here (the axes, the diagonals, the pole's and the walls' edges, an odd one) and on another base and yaw.
    Regression caught: a profile that drifts from the gold it is meant to reproduce (a window that misses a sub-cell,
    a tie broken the other way, a rounding), so the table's answer would not be the gold."""
    cap = 5.0 if robot is AMR else 3.0
    states = scene()
    for base, yaw in ((BASE, YAW), ((0.37, -0.52), 0.9)):
        runs = clear_profile.profile(states, robot, base, yaw, cap)
        for i in (0, 1, 8, 9, 17, 44, 45, 46, 89, 90, 91, 133, 179, 180, 181, 227, 268, 270, 271, 300, 342, 359):
            heading = yaw + math.radians(i)
            gold = grid.corridor_clear(states, robot, grid.front_edge(robot, base, heading), heading, cap, 0.0)[0]
            assert runs[i] == (gold["value"] if gold["type"] == "number" else None), (i, base)


def test_a_coarser_step_samples_the_same_profile():
    """Contract: `step_deg` only thins the headings: the profile at 5 degrees is the 1 degree profile's every fifth
    entry, 72 of them, and a step that does not divide 360 is refused. Regression caught: a step that shifts the
    headings (an offset or a rounding), or a profile of the wrong length that the scorer would index out of range."""
    states = scene()
    fine = clear_profile.profile(states, AMR, BASE, YAW, 5.0, 1)
    assert clear_profile.profile(states, AMR, BASE, YAW, 5.0, 5) == fine[::5]
    with pytest.raises(ValueError):
        clear_profile.profile(states, AMR, BASE, YAW, 5.0, 7)


def test_a_profile_over_nothing_seen_is_all_none():
    """Contract: with no cell seen no corridor was seen empty: every heading is None (and `best_index` finds none).
    Regression caught: an empty view called clear at the cap."""
    runs = clear_profile.profile({}, VACUUM, BASE, YAW, 3.0)
    assert runs == [None] * 360 and clear_profile.best_index(runs) is None


def test_a_profile_is_one_pass_over_the_cells_near_the_base():
    """Contract (speed, plan H1): a point's 360 headings are one pass over the cells near the base, tens of
    milliseconds on the largest scene (the AMR to a 5 m cap over 14,400 seen cells), not 360 corridors. The bound, a
    second, is thirty times the measured time so a loaded runner passes, and a tenth of what looping
    `grid.corridor_clear` over the headings takes. Regression caught: a profile that loops the scalar gold."""
    states = scene()
    t0 = time.perf_counter()
    clear_profile.profile(states, AMR, BASE, YAW, 5.0)
    assert time.perf_counter() - t0 < 1.0


def test_the_best_heading_is_the_nearest_ahead_of_the_longest_runs():
    """Contract: `best_index` is the heading with the longest run, and among those within the sub-cell of it the one
    nearest the robot's own, the left one when two are as near. Hand profile at 90 degrees a step (4 headings): runs
    (1.0, 2.0, 1.99, None): the longest is 2.0 at 90 and 1.99 at 180 is within 0.02 of it; 90 is nearer ahead than 180.
    (2.0, 0.5, 2.0, 0.5): ahead is the best. (0.5, 2.0, 0.5, 2.0): left (90) wins over right (270) on a tie. Regression
    caught: a best heading that is the farthest ahead or the first by index, so a table answer turns more than it must
    and the gold changes with the order of the list."""
    best = clear_profile.best_index
    assert best([1.0, 2.0, 1.99, None]) == 1
    assert best([2.0, 0.5, 2.0, 0.5]) == 0
    assert best([0.5, 2.0, 0.5, 2.0]) == 1


def test_a_profile_index_maps_to_the_headings_degrees():
    """Contract: the degrees of a profile index lie in (-180, 180], left positive: the gold and the baselines answer a
    best index as these degrees (the scorer owns the way back, `score_graded.regret_credit`, which rounds an answered
    heading to the step). Hand values: at 1 degree index 181 is -179, index 180 is 180 (not -180), 270 is -90 and 0 is
    0; at 5 degrees (72 bins) index 54 is 270 degrees, -90, and index 36 is 180. Regression caught: a half-turn written
    as -180 where the scorer reads 180, or a heading to the right answered to the left."""
    assert clear_profile.deg_of(181, 1) == -179 and clear_profile.deg_of(180, 1) == 180
    assert clear_profile.deg_of(270, 1) == -90 and clear_profile.deg_of(0, 1) == 0
    assert clear_profile.deg_of(54, 5) == -90 and clear_profile.deg_of(36, 5) == 180


def test_a_box_turning_in_place_is_blocked_by_what_its_corner_sweeps():
    """Contract: a box body turning in place sweeps the disc of its half-diagonal (the AMR's 0.494 m); an obstacle
    sub-cell inside that disc and outside the footprint blocks the turn, one inside the footprint (what the body
    already holds) or beyond the disc does not, and a disc body is never blocked. Hand values, base (0, 0), heading +x,
    the footprint |x| <= 0.4, |y| <= 0.29: a sub-cell at (0.31, 0.35) is 0.468 m out and beside the footprint: blocks;
    (0.43, 0.21) is 0.479 m out and ahead of it: blocks; (0.21, 0.21) is inside the footprint; (0.61, 0.61) is beyond
    the disc. Regression caught: a turn made through a wall beside the robot, or one refused for what the body
    already contains."""
    def blocked(x: float, y: float, robot=AMR) -> bool:
        cell = (sub(x) // 5, sub(y) // 5)
        return clear_profile.turn_swept({cell: (grid.Occupied([(sub(x), sub(y))]), 0)}, robot, BASE, YAW)

    assert blocked(0.31, 0.35) and blocked(0.43, 0.21)
    assert not blocked(0.21, 0.21) and not blocked(0.61, 0.61)
    assert not blocked(0.31, 0.35, VACUUM)
