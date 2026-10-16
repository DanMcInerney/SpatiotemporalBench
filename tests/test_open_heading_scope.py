"""The scope of the `open_heading` gold: `now` exactly when the look point's own frame answers it (frame k's best
heading, the same function over the cells of that keyframe alone, which `current_view` gives, has no regret beyond the
row's resolution on the window's profile), else `memory`. On `heading_kit`'s tote AMR and `wall_cells()` (free floor
west of a wall at x 1.01: the best heading is 83 degrees left, the first to run to the 5 m cap), the look point keyframe
19, which ends the look window of keyframes 16 to 19 (the window decides where a point may fall, no longer the
scope)."""

from __future__ import annotations

import pytest

from heading_kit import LAST, heading_world, wall_cells
from stbench import questions_heading


def scope(by_k: dict) -> tuple[str, float]:
    """The scope of the gold asked at the look point, and its heading."""
    (c,) = questions_heading.open_heading(heading_world(by_k), LAST).answerable
    return c.scope, c.gold["deg_left"]


def cells(keep) -> dict:
    """`wall_cells()` kept to the fine cells (u, v) that `keep` names."""
    return {c: s for c, s in wall_cells().items() if keep(*c)}


def test_the_frame_that_sees_everything_answers_it():
    """Contract: the scope is `now` when frame k's own profile gives a best heading with no regret beyond the
    resolution (0.02 m) on the window's. Hand case: the whole scene seen at keyframe 19: its best heading is the gold's,
    83.0, regret 0. Regression caught: a `memory` row the current view answers."""
    assert scope({LAST: wall_cells()}) == ("now", 83.0)


def test_a_frame_that_sees_nothing_cannot():
    """Contract: with the scene seen at keyframe 5 and nothing at keyframe 19 the frame has no heading at all:
    `memory`. Hand case as the gold's test. Regression caught: a frame with no best heading counted as one that
    answers."""
    assert scope({5: wall_cells()}) == ("memory", 83.0)


@pytest.mark.parametrize("seen_at", [18, 17, 16])
def test_a_scene_only_an_earlier_frame_of_the_look_window_saw_is_memory(seen_at):
    """Contract: the scope reads frame k alone. Hand case: the whole scene (best heading 83.0, the gold's) seen at one
    keyframe of the look window 16-19 other than 19, and nothing at 19: the gold is the same 83.0, but the point's own
    frame sees no heading, so `memory`, where the old rule, which read the look window's cells, called it `now`.
    Regression caught: the scope reading the look window again (a scene a system was sent one to three messages ago
    called `now`)."""
    assert scope({seen_at: wall_cells()}) == ("memory", 83.0)


def test_a_frame_that_sees_a_different_heading_as_good_still_answers_it():
    """Contract: the test is the regret of the frame's best, not that it is the gold's heading. Hand case: the whole
    scene at keyframe 5; at keyframe 19 only the floor west of x 0.3 (fine cells u <= 2, every row), where the corridors
    of the headings from 88 degrees on, to the west and behind, run to the cap with nothing in the way: the frame's
    best is 88, not the gold's 83, and the window's run there is the cap too: regret 0, `now`. Regression caught: a
    scope that demands the same heading, so `memory` where any best heading will do."""
    assert scope({5: wall_cells(), LAST: cells(lambda u, v: u <= 2)}) == ("now", 83.0)


def test_a_frame_whose_best_is_short_of_the_window_best_cannot():
    """Contract: where the frame's best heading runs far short of the window's best, the row is `memory`. Hand case: the
    whole scene at keyframe 5; at keyframe 19 only the sector ahead of the base (fine cells u 0 .. 10, v -20 .. 19, 2 m
    either side), which sees the wall and the headings that meet it inside the sector: the frame's best heading (70
    degrees) runs 1.75 m against the window's 5 m, so the regret is far over the resolution: `memory`. Regression
    caught: the `memory` rows that a current view that sees the wall but not the opening would be credited with."""
    assert scope({5: wall_cells(), LAST: cells(lambda u, v: 0 <= u <= 10 and -20 <= v <= 19)}) == ("memory", 83.0)
