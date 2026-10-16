"""The robot's own velocity, which every delivered step carries (`case.base_velocity`, `case.case_rows`, the stream's
`velocity`; final review S3): the controller's odometry, computed by the harness from the delivered pose rows of the
current task and the true script times. Hand-built poses on `Synth`'s two tasks of four keyframes, 1.5 s apart (task 1
starts 7200 s after task 0), so every number is worked out in the test."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stbench import case as case_mod
from stbench import stream
from stbench import timeline as T
from stbench.case import ScriptPaths, base_velocity, case_rows, read_case
from stbench.visibility import load_pose
from stream_kit import case_on, end_q
from synth import HOME, Synth

KITCHEN = HOME.rooms[0]
# Keyframes 0-3 are task 0 and 4-7 task 1. The base moves 0.3 m east, then 0.6 m north, then stands, then the next
# task starts far away (9, 9) and moves 0.6 m south: at 1.5 s a keyframe, 0.2 m/s east, 0.4 m/s north, 0, (the new
# task's first frame), -0.4 m/s north.
POSES = {0: (1.0, 2.0), 1: (1.3, 2.0), 2: (1.3, 2.6), 3: (1.3, 2.6), 4: (9.0, 9.0), 5: (9.0, 8.4), 6: (9.0, 8.4),
         7: (9.0, 8.4)}


def scene() -> Synth:
    return Synth(tasks=[[(KITCHEN, 4)], [(KITCHEN, 4)]], poses=POSES)


def test_a_step_carries_the_bases_displacement_from_the_frame_before_over_the_true_time_and_zero_at_a_task_start(
        tmp_path):
    """Contract (final review S3): a case row's `velocity` is the base's displacement from the previous delivered frame
    of the same task to this frame, over the true time between them, in metres a second in the map frame, and 0 at a
    task's first frame, even where the previous script frame (the last of the task before) is 8 m away. By hand on
    `POSES`: keyframe 1 is 0.3 m east in 1.5 s, (0.2, 0.0); 2 is 0.6 m north, (0.0, 0.4); 3 stands, (0.0, 0.0); 4 is
    task 1's first, (0.0, 0.0) whatever the move from (1.3, 2.6) to (9, 9); 5 is 0.6 m south, (0.0, -0.4); 0 is task 0's
    first. A window of task 1 alone starts at its first frame and gives the same rows; the rows carry no `-0.0`.
    Regression caught: a velocity differenced across the gap between tasks (a speed of 8 m in 7,198.5 s, or a window
    that depends on what came before it), one taken from the frame after, or a velocity computed over the rounded
    clock."""
    sid = scene().write(tmp_path).id
    tl = T.load(ScriptPaths(tmp_path, sid).timeline)
    both = case_mod.write_case(tmp_path, sid, 0, 1, "medium")
    rows = case_rows(read_case(both), tl, load_pose(both.script.pose))
    assert [r["velocity"] for r in rows] == [[0.0, 0.0], [0.2, 0.0], [0.0, 0.4], [0.0, 0.0],
                                             [0.0, 0.0], [0.0, -0.4], [0.0, 0.0], [0.0, 0.0]]
    assert all(str(v) != "-0.0" for r in rows for v in r["velocity"])
    second = case_mod.write_case(tmp_path, sid, 1, 1, "short")
    alone = case_rows(read_case(second), tl, load_pose(second.script.pose))
    assert [r["velocity"] for r in alone] == [r["velocity"] for r in rows[4:]]


def test_the_velocity_is_over_the_true_script_time_and_not_the_rounded_clock():
    """Contract (final review S3): the divisor is the keyframes' true time difference (`Timeline.keyframe(k).t`), not
    the difference of the tenth-rounded clocks that are delivered. By hand: two frames 0.14 s apart (clocks 0.0 and 0.1
    after rounding) with the base 0.07 m east: 0.5 m/s, where the rounded clocks would say 0.7. Rounded to the
    millimetre a second: 0.07 m in 0.3 s is 0.2333 m/s, delivered as 0.233; a base that moved -1e-7 m is 0.0, never
    -0.0. Regression caught: a velocity over the delivered clocks (wrong by up to a half tenth of a second over a step
    1-5 s apart: 3% at the AMR's fastest frames) or one that keeps a float's tail."""
    times = {0: 0.0, 1: 0.14, 2: 0.44, 3: 0.44 + 7500.0}
    tl = SimpleNamespace(keyframe=lambda k: SimpleNamespace(t=times[k]))
    row = {"status": {"task": 0}}
    rows = {0: {**row, "pose": {"pos": [0.0, 0.0, 0.0]}}, 1: {**row, "pose": {"pos": [0.07, 0.0, 0.0]}},
            2: {**row, "pose": {"pos": [0.14, 0.0, 0.0]}}, 3: {**row, "pose": {"pos": [0.14, -0.0000001, 0.0]}}}
    assert base_velocity(tl, rows, 1) == [0.5, 0.0]
    assert base_velocity(tl, rows, 2) == [0.233, 0.0]
    assert base_velocity(tl, rows, 3) == [0.0, 0.0] and str(base_velocity(tl, rows, 3)[1]) == "0.0"
    assert base_velocity(tl, rows, 0) == [0.0, 0.0]


@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_every_delivered_step_on_a_channel_that_sends_the_pose_carries_the_velocity(tmp_path, channel):
    """Contract (final review S3): a step on a channel that sends `pose` carries `velocity` beside `pose` and `camera`,
    the same rows `case_rows` makes (the numbers the gold of `closest_approach` reads), for the whole window, in case
    numbering. By hand on `POSES`: the eight steps of the two tasks are (0, 0), (0.2, 0), (0, 0.4), (0, 0), (0, 0),
    (0, -0.4), (0, 0), (0, 0). Regression caught: a step without it on a channel (the table systems read it
    strictly), or a stream whose velocity differs from the gold's."""
    case = case_on(tmp_path, scene(), [end_q(1)], a=0, b=1, length="medium")
    steps = stream.delivered(case, channel).steps
    assert [s["velocity"] for s in steps] == [[0.0, 0.0], [0.2, 0.0], [0.0, 0.4], [0.0, 0.0],
                                              [0.0, 0.0], [0.0, -0.4], [0.0, 0.0], [0.0, 0.0]]
    assert all({"pose", "camera", "velocity"} <= set(s) for s in steps)
