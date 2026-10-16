"""The window a `clear_path` gold reads (0.9.0, `clear_ground.moment`): the latest cells of every keyframe of the case
up to the look point, not the point's frame alone. `clear_kit`'s tote AMR at `BASE` (0.6, 0) heading +x at the point
(keyframe 19, the look window of keyframes 16 to 19); `room_run`'s camera looked back from (2.0, 0) in keyframes 0 to 3
and saw the wall at x -2.21, 2.41 m behind the rear edge (x 0.2), whose corridor stops at 2.41 - 0.01 = 2.40 m."""

from __future__ import annotations

from dataclasses import replace

from clear_kit import JOB, LAST, PI, drive_run, room_run, seen_from, wall, world
from space_kit import box
from stbench import clear_ground

BACK = replace(JOB, clear_directions=("back",))
AHEAD = replace(JOB, clear_directions=("ahead",))
OBSTACLE = box(-1.31, -0.81, -0.2, 0.2, 0.0, 0.5)   # a crate in the back corridor, its near face at x -0.81


def only(w, k=LAST):
    m = clear_ground.moment(w, k)
    (cand,) = [*m.answerable, *m.unknown]
    return cand


def test_a_wall_seen_long_before_the_point_stops_the_back_corridor():
    """Contract (brief 4.1, truth rule 2): the gold reads what the robot last saw of each cell over the whole window up
    to the point, so a corridor behind the robot is a number from frames that were sent 16 or more keyframes earlier
    (the point's frame faces the other way and shows none of it). Hand scene: the wall at x -2.21 seen in keyframes 0 to
    3 only: the back corridor from the rear edge (0.2) is 2.41 to the sub-cell centre -2.21, so 2.40. Its evidence is
    those frames, none of the look window. Regression caught: a gold from the point's frame alone (`unknown`: it has no
    cell of the corridor), or from the base (2.8)."""
    c = only(world(room_run(), [], BACK))
    assert c.gold == {"type": "number", "value": 2.4} and c.slice == "answerable"
    assert c.evidence_k and all(k < 16 for k in c.evidence_k)


def crate_run(relook: bool):
    """`room_run` with a crate in the back corridor in keyframes 0 to 11 only; with `relook` keyframe 17 (in the look
    window) stands at (2.0, 0) facing -x again, where the crate is gone."""
    script = room_run(frame_boxes={k: [OBSTACLE] for k in range(12)})
    if relook:
        seen_from(script, [17], (2.0, 0.0), PI)
        script.depth_ks.add(17)
    return script


def test_a_cell_seen_again_later_takes_the_later_state():
    """Contract (truth rule 2): a cell's state is the last frame that observed it, so a later look overrides the frames
    before it. Hand scene: a crate in the back corridor with its near face at x -0.81 (the sub-cell centre -0.81, 1.01
    from the rear edge at 0.2: stop 1.00) is in keyframes 0 to 11; the wall is at 2.40. Seen as it was (no later look
    back) the back gold is 1.00, the crate, scope `memory`; when keyframe 17 of the look window looks back again, the
    crate is gone and the floor under it is seen free, so the gold is the wall's 2.40 (the point's own frame, 19, faces
    forward and shows none of the corridor: scope `memory` still). Regression caught: a gold that keeps the first
    sighting of a cell, or one that lets a later state not replace an older one."""
    kept = only(world(crate_run(relook=False), [], BACK))
    assert (kept.gold, kept.scope) == ({"type": "number", "value": 1.0}, "memory")
    seen = only(world(crate_run(relook=True), [], BACK))
    assert (seen.gold, seen.scope) == ({"type": "number", "value": 2.4}, "memory")


def test_nothing_after_the_point_is_read():
    """Contract: the gold names the keyframes up to the look point and no later one. Hand scene: the ahead corridor
    with nothing in it in keyframes 0 to 15 and a wall at x 3.01 in keyframes 16 to 19 only. The look window of keyframe
    15 (keyframes 8 to 15) saw an empty floor to the cap: 5.0; at keyframe 19 the wall is seen: 2.0. Regression caught:
    a gold at keyframe 15 that reads the wall of keyframes 16 to 19, a frame the system had not yet been sent."""
    script = drive_run(frame_boxes={k: [wall(3.01).at] for k in range(16, 20)}, depth_ks={15, 19})
    w = world(script, [], AHEAD)
    assert only(w, 15).gold == {"type": "number", "value": 5.0}
    assert only(w, LAST).gold == {"type": "number", "value": 2.0}

