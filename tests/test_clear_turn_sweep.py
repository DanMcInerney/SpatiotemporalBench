"""The turn sweep (0.9.0, `clear_profile.turn_swept`, `clear_ground.moment`): a question that begins "if you turned in
place" is asked only where the body could turn. A box sweeps the disc of its half-diagonal about its middle (the tote
AMR: 0.8 x 0.58 m, hypot(0.4, 0.29) = 0.494 m); a seen occupied sub-cell in that disc and outside the footprint is a
corner's collision. A disc turns inside its own footprint."""

from __future__ import annotations

from dataclasses import replace

import pytest

from clear_kit import AMR, JOB, LAST, room_run, world
from space_kit import box
from stbench import clear_ground, clear_profile
from stbench.robots import ROBOTS


def occupied(*fine: tuple[int, int]) -> dict:
    """Cells of a seen place: each given fine cell wholly occupied (a bare `occupied` stands for all 25 sub-cells)."""
    return {c: ("occupied", 0) for c in fine}


@pytest.mark.parametrize("cell, yaw, blocked", [
    ((0, 3), 0.0, True),          # y 0.3 to 0.4: every centre is over 0.29 to the side, within 0.494 of the middle
    ((0, 8), 0.0, False),         # y 0.8 to 0.9: past the disc (0.81)
    ((0, 1), 0.0, False),         # y 0.1 to 0.2: inside the footprint
    ((4, 2), 0.0, True),          # (0.41, 0.21) is 0.46 from the middle, 1 cm past the front edge: a corner's way
    ((5, 0), 0.0, False),         # (0.51, 0.01) is 0.51 from the middle: past the disc
    ((0, 3), 3.141592653589793 / 2, False),   # the same cell is ahead of a body heading north: inside the footprint
    ((3, 0), 3.141592653589793 / 2, True),    # and x 0.3 to 0.4 is now to its side
])
def test_a_box_cannot_turn_where_a_seen_obstacle_stands_in_its_sweep_outside_its_footprint(cell, yaw, blocked):
    """Contract (brief 4.1, the turn sweep): `turn_swept` is true exactly when an occupied sub-cell (0.02 m) has its
    centre within the half-diagonal of the middle of the body, and outside its footprint at the delivered heading. Hand
    values for the AMR (half length 0.4, half width 0.29, half-diagonal 0.494) at the origin, per cell: see the table's
    comments; the footprint turns with the heading, the sweep does not. Regression caught: the footprint read at the
    wrong heading, the body's own cells counted as an obstacle, or an obstacle beyond the sweep refusing the turn."""
    assert clear_profile.turn_swept(occupied(cell), AMR, (0.0, 0.0), yaw) is blocked


def test_a_disc_turns_in_its_own_footprint_and_a_free_or_unseen_cell_never_blocks():
    """Contract: the vacuum's disc sweeps nothing outside its footprint, so an obstacle against its side never blocks; a
    free cell and an unseen one block no box either. Regression caught: a disc refused a turn it can make, or the sweep
    reading cells that are not obstacles."""
    vacuum = ROBOTS["vacuum"]
    assert not clear_profile.turn_swept(occupied((0, 2), (3, 0)), vacuum, (0.0, 0.0), 0.0)
    assert not clear_profile.turn_swept({(0, 3): ("free", 0)}, AMR, (0.0, 0.0), 0.0)
    assert not clear_profile.turn_swept({}, AMR, (0.0, 0.0), 0.0)


def console_run(x: float):
    """The closed room with a console, 0.2 m wide and 0.5 m tall, from `x` to `x + 0.2` along the robot's left side
    (y 0.31 to 0.5), seen in keyframes 4 to 7 (from (0.6, -1.5) facing north)."""
    script = room_run(("ahead", "left"))
    script.boxes.append(box(x, x + 0.2, 0.31, 0.5, 0.0, 0.5))
    return script


def test_a_console_beside_the_body_asks_no_turned_corridor_and_one_further_off_does():
    """Contract (brief 4.1, plan G1): a point where a seen obstacle stands in the box body's sweep outside its footprint
    asks no left or right corridor. Hand scene on the AMR at (0.6, 0) heading +x, a console against its left side (x 0.6
    to 0.8, y 0.31 to 0.5: its sub-cell centres are 0.32 to 0.5 from the middle, over the 0.29 half-width): nothing is
    asked left, and the same console 0.7 m further east (x 1.3 to 1.5, whose nearest sub-cell centre (1.31, 0.32) is 0.7
    m from the middle, past the sweep) leaves the corridor asked, 2.6 m to the wall at y 3.01. The ahead corridor is
    asked either way. Regression caught: a turned corridor asked where the corners could not make the turn, or one
    refused for an obstacle that is not in the way."""
    left = replace(JOB, clear_directions=("left",))
    assert clear_ground.moment(world(console_run(0.6), [], left), LAST) is None
    m = clear_ground.moment(world(console_run(1.3), [], left), LAST)
    (cand,) = [*m.answerable, *m.unknown]
    assert cand.gold == {"type": "number", "value": 2.6}
    ahead = replace(JOB, clear_directions=("ahead",))
    assert clear_ground.moment(world(console_run(0.6), [], ahead), LAST) is not None


def test_the_sweep_reads_what_the_robot_saw_not_the_world():
    """Contract (brief 4.1, validity): an obstacle that no frame showed cannot stop a turn from being asked, because the
    asking would tell the agent it is there. Hand scene: the console of the last test in the frames of a camera that
    looked at it (keyframe 7, from the south) refuses the turn; with no depth in that frame (only the look window,
    facing east, which cannot see the console against the side) the turn is asked. Regression caught: a sweep read from
    the timeline or the map, which would leak where unseen obstacles stand."""
    left = replace(JOB, clear_directions=("left",))
    script = console_run(0.6)
    script.depth_ks = {LAST}   # keyframe 7 is not rendered: the console was never seen
    m = clear_ground.moment(world(script, [], left), LAST)
    assert m is not None and [*m.answerable, *m.unknown][0].gold["type"] == "unknown"
