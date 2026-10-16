"""The compiler's recent view and end view (`World.recent`, `World.end_window`, 0.10.0): the last 8 frames of the task a
look point is in, and the last 8 frames of the case. Hand values on `looks_kit`'s tasks (the vacuum's of 28 frames,
`long_task`)."""

from __future__ import annotations

import pytest

from looks_kit import long_task, look_job, vacuum_run, world


@pytest.mark.parametrize("k, want", [(13, (6, 14)), (4, (0, 5)), (27, (20, 28)), (30, (28, 31)), (40, (33, 41))])
def test_the_recent_view_is_the_last_eight_frames_of_the_points_task(k, want):
    """Contract (FD-questions-0.10 §3): `World.recent(k)` is the last 8 frames of the task that holds `k`, up to and
    including `k`, and never reaches into the task before. Hand-worked on a window over the vacuum's two tasks (frames
    0-27 and 28-55): frame 13 sees 6 to 13; frame 4, early in its task, 0 to 4; frame 27 the task's last eight, 20 to
    27; frame 30, in the second task, starts at the task's own first frame, 28 to 30, not at 23; frame 40 sees 33 to 40.
    Regression caught: a view that slides back across the task boundary (a thing of the first task called recent in the
    second), or one that stops at a decision window's cut."""
    w = world(vacuum_run(2), length="medium", b=1, job=look_job())
    assert w.recent(k) == range(*want)


@pytest.mark.parametrize("frames, want", [(5, (0, 5)), (8, (0, 8)), (9, (1, 9)), (60, (52, 60))])
def test_the_end_view_is_the_last_eight_frames_of_the_case(frames, want):
    """Contract: `World.end_window` is the case window's last 8 frames, or all of them when the window is shorter,
    whatever the decision points cut. Hand values: a task of 5 frames holds 0 to 4, of 8 frames 0 to 7, of 9 frames 1
    to 8, of 60 frames 52 to 59; a window over the vacuum's two tasks (56 frames, its first task cut at 5, 18 and 21)
    holds 48 to 55. Regression caught: the end view following the last decision cut (4 frames, or 1), so that a thing
    seen 5 frames before the end counts as out of view."""
    assert world(long_task(frames), job=look_job()).end_window == range(*want)
    two = world(vacuum_run(2), length="medium", b=1, job=look_job())
    assert two.end_window == range(48, 56)
