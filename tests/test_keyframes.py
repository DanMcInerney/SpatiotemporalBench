"""`keyframes.frames`: motion primitives to keyframes under a robot's frame policy. Every expectation is worked out by
hand from the vacuum's policy: a frame each 1.0 m, 45 degrees or 5 s, none closer than 1.0 s."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from stbench.keyframes import Frame, Hold, Mark, Move, Tilt, Turn, frames
from stbench.robots import ROBOTS

POLICY = ROBOTS["vacuum"].frames  # move_m 1.0, turn_deg 45.0, still_s 5.0, min_dt_s 1.0
FLIER = ROBOTS["drone"].frames  # move_m 3.0, turn_deg 30.0, still_s 3.0, min_dt_s 1.0
ORIGIN = (0.0, 0.0)


def times(fs) -> list[float]:
    return [f.t for f in fs]


def test_a_straight_run_gets_a_frame_each_metre_and_one_at_its_end():
    """Contract (plan §4.3): a frame at the start, then one after each metre of travel, and one at the end of the
    phase's run. Hand case: 2.5 m at 0.25 m/s is 10 s, with frames at 0 s, after 1 m (4 s), after 2 m (8 s) and at the
    end (10 s). Regression caught: a gap of more than 1 m without a frame, or no frame where the run ends."""
    fs = frames([Move(ORIGIN, (2.5, 0.0), 0.0, 0.25, "run")], POLICY)
    assert times(fs) == [0.0, 4.0, 8.0, 10.0]
    assert [f.xy for f in fs] == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.5, 0.0)]
    assert {f.phase for f in fs} == {"run"} and {f.yaw for f in fs} == {0.0}


def test_a_turn_in_place_gets_a_frame_each_45_degrees():
    """Contract: turning is sampled each 45 degrees, in place, and the turn's end is a frame. Hand case: 180 degrees at
    45 degrees a second is 4 s, with frames at 0 s and each second after, at 0, 45, 90, 135 and 180 degrees, the base
    standing still. Regression caught: a gap of more than 45 degrees without a frame, or a turn that ends between
    frames (a path vertex that is no keyframe)."""
    fs = frames([Turn(ORIGIN, 0.0, math.pi, 45.0, "turn")], POLICY)
    assert times(fs) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [f.yaw for f in fs] == pytest.approx([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi])
    assert {f.xy for f in fs} == {ORIGIN}


def test_a_turn_takes_the_short_way_round():
    """Contract: a turn goes the short way, so from -135 to +135 degrees is 90 degrees clockwise, not 270 the other way.
    Hand case: 90 degrees at 45 degrees a second is 2 s, with frames at 0, 1 and 2 s (headings -135, 180 and 135
    degrees). Regression caught: a turn the long way round, which takes 6 s."""
    fs = frames([Turn(ORIGIN, math.radians(-135), math.radians(135), 45.0, "turn")], POLICY)
    assert times(fs) == [0.0, 1.0, 2.0]
    assert [f.yaw for f in fs] == pytest.approx([math.radians(-135), math.pi, math.radians(135)])


def test_a_required_frame_too_soon_waits_until_a_second_has_passed():
    """Contract (map X-H3): no two frames of a task are closer than `min_dt_s`; a required frame that would be, the
    robot holds still until then. Hand cases: a mark 0.4 s after the start lands at 1.0 s, and carries the decision; a
    30 degree turn takes 0.667 s, so its end is at 1.0 s and a mark on it merges into that frame. Regression caught: two
    frames under a second apart, or a mark that makes a second frame at the same pose."""
    fs = frames([Hold(ORIGIN, 0.0, 0.4, "wait"), Mark("decide")], POLICY)
    assert times(fs) == [0.0, 1.0] and [f.decision for f in fs] == [None, "decide"]
    fs = frames([Turn(ORIGIN, 0.0, math.radians(30), 45.0, "turn"), Mark("decide")], POLICY)
    assert times(fs) == [0.0, 1.0] and [f.decision for f in fs] == [None, "decide"]


def test_a_phase_ends_in_a_frame_and_the_next_phase_follows_a_second_later():
    """Contract (M1's question 2): the last moment of each phase is a frame ("when you left the dock" resolves), and a
    frame of the next phase cannot share its instant, so it is a second later. Hand case: 0.5 m at 0.25 m/s of phase "a"
    ends at 2 s; a zero-time hold of phase "b" is a frame at 3 s. Regression caught: a phase with no frame of its own
    (an event that resolves to nothing), or two phases in one frame."""
    fs = frames([Move(ORIGIN, (0.5, 0.0), 0.0, 0.25, "a"), Hold((0.5, 0.0), 0.0, 0.0, "b")], POLICY)
    assert times(fs) == [0.0, 2.0, 3.0] and [f.phase for f in fs] == ["a", "a", "b"]


def test_standing_still_gets_a_frame_every_five_seconds():
    """Contract: with neither a metre nor 45 degrees, a frame comes when 5 s have passed. Hand case: a 12 s hold has
    frames at 0, 5, 10 and 12 s. Regression caught: a gap of more than 5 s without a frame."""
    assert times(frames([Hold(ORIGIN, 0.0, 12.0, "wait")], POLICY)) == [0.0, 5.0, 10.0, 12.0]


def test_a_reversing_move_keeps_its_heading_and_a_mark_keeps_its_place():
    """Contract: a `Move`'s yaw is its heading, which is opposite its motion when the robot reverses; a `Mark` is a
    frame where the robot is. Hand case: reversing 2 m west at 0.5 m/s with the heading kept east, marked half way, has
    frames at 0 s (x 2), 2 s (x 1, the metre and the mark in one) and 4 s (x 0), all heading 0. Regression caught: a
    reverse that turns the robot round, or a mark that moves it or makes a second frame."""
    prims = [Move((2.0, 0.0), (1.0, 0.0), 0.0, 0.5, "back"), Mark("half"),
             Move((1.0, 0.0), (0.0, 0.0), 0.0, 0.5, "back")]
    fs = frames(prims, POLICY)
    assert [(f.t, f.xy, f.yaw, f.decision) for f in fs] == [
        (0.0, (2.0, 0.0), 0.0, None), (2.0, (1.0, 0.0), 0.0, "half"), (4.0, (0.0, 0.0), 0.0, None)]


def test_nothing_happens_for_a_move_of_no_length_or_a_turn_of_no_angle():
    """Contract: a move of no length and a turn of no angle are no primitives (no frame, no time). Regression caught: a
    zero-angle vertex of a path costing a keyframe and a pause."""
    prims = [Hold(ORIGIN, 0.0, 0.0, "a"), Move(ORIGIN, ORIGIN, 0.0, 1.0, "a"), Turn(ORIGIN, 1.0, 1.0, 45.0, "a")]
    assert times(frames(prims, POLICY)) == [0.0]


def test_a_mark_with_nothing_before_it_is_refused():
    """Contract: a `Mark` takes its phase from the primitive before it. Regression caught: a mark first in the list,
    which would be a frame of no phase."""
    with pytest.raises(ValueError, match="Mark"):
        frames([Mark("decide")], POLICY)


def test_frames_start_at_the_given_time():
    """Contract: `t0` is the clock of the first frame. Regression caught: a second task's frames starting at 0."""
    assert times(frames([Move(ORIGIN, (2.0, 0.0), 0.0, 1.0, "a")], POLICY, t0=100.0)) == [100.0, 101.0, 102.0]


def test_old_primitives_sample_as_before():
    """Contract (M3-Y): a primitive that names no height and no pitch is a ground robot's, so every existing list of
    primitives keeps its frames and each frame stands at height
    0 with no pitch of its own. Hand case under the vacuum's policy: a 1.5 m run at 1 m/s marked at its end, then a 90
    degree turn at 45 degrees a second, is five frames: 0 s, 1.0 s (the metre), 2.0 s (the run's end, the mark's, held a
    second after the last frame), 3.0 s (the turn's 45 degree step) and 4.0 s (its end). Regression caught: a default
    height or pitch that moves a ground robot's frames, or a flier's fields leaking into a ground robot's."""
    prims = [Move(ORIGIN, (1.5, 0.0), 0.0, 1.0, "run"), Mark("m"), Turn((1.5, 0.0), 0.0, math.pi / 2, 45.0, "turn")]
    fs = frames(prims, POLICY)
    assert fs == [Frame(0.0, (0.0, 0.0), 0.0, "run", None), Frame(1.0, (1.0, 0.0), 0.0, "run", None),
                  Frame(2.0, (1.5, 0.0), 0.0, "run", "m"), Frame(3.0, (1.5, 0.0), round(math.pi / 4, 6), "turn", None),
                  Frame(4.0, (1.5, 0.0), round(math.pi / 2, 6), "turn", None)]
    assert {f.z for f in fs} == {0.0} and {f.pitch for f in fs} == {None}


def test_a_climb_is_framed_by_its_3d_length():
    """Contract: a `Move` counts its 3-D length, so a vertical climb of no ground distance is a move, framed every
    `move_m` of climb, and a slope counts its hypotenuse. Hand cases under a flier's policy (3 m, 30 degrees, 3 s, 1 s):
    0 to 15 m at 3 m/s is 5 s with a frame each second, at 0, 3, 6, 9, 12 and 15 m, the heading and the pitch held; a
    3-4-5 slope (4 m along, 3 m up, 1 m/s) is 5 s of travel with frames at 0, 3 and 5 s, x 0, 2.4 and 4, z 0, 1.8 and 3.
    Regression caught: a vertical move dropped as empty, or a climb framed by its ground distance (no frame at all)."""
    up = frames([Move((5.0, 5.0), (5.0, 5.0), 0.7, 3.0, "up", z=(0.0, 15.0), pitch=-0.5)], FLIER)
    assert times(up) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [f.z for f in up] == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    assert {f.xy for f in up} == {(5.0, 5.0)} and {f.yaw for f in up} == {0.7} and {f.pitch for f in up} == {-0.5}
    slope = frames([Move(ORIGIN, (4.0, 0.0), 0.0, 1.0, "s", z=(0.0, 3.0))], FLIER)
    assert [(f.t, f.xy[0], f.z) for f in slope] == [(0.0, 0.0, 0.0), (3.0, 2.4, 1.8), (5.0, 4.0, 3.0)]


def test_a_tilt_is_framed_every_turn_deg_of_pitch():
    """Contract: the camera's pitch is framed as the heading is, each `turn_deg` turned since the last frame, never
    sooner than `min_dt_s`, and the tilt's end is a frame. Hand cases for a 90 degree tilt (0 to -90) at 60 degrees a
    second, which takes 1.5 s: with a half-second `min_dt_s`, frames at 0, 0.5, 1.0 and 1.5 s at 0, -30, -60 and -90
    degrees; under the flier's one second, frames at 0 and 1.0 s (-60 degrees) and the end at 2.0 s (-90: the robot
    holds still from 1.5 s). Regression caught: a gimbal that turns between frames unseen, or one the policy cannot
    see at all."""
    tilt = Tilt(ORIGIN, 0.0, 15.0, 0.0, -math.pi / 2, 60.0, "look")
    fine = frames([tilt], replace(FLIER, min_dt_s=0.5))
    assert times(fine) == [0.0, 0.5, 1.0, 1.5]
    assert [f.pitch for f in fine] == pytest.approx([0.0, -math.pi / 6, -math.pi / 3, -math.pi / 2], abs=1e-6)
    coarse = frames([tilt], FLIER)
    assert times(coarse) == [0.0, 1.0, 2.0]
    assert [f.pitch for f in coarse] == pytest.approx([0.0, -math.pi / 3, -math.pi / 2], abs=1e-6)
    assert {f.z for f in coarse} == {15.0} and {f.xy for f in coarse} == {ORIGIN}
