"""The flier's `clear_path` gold (0.8.0 lane L2: `clear_fly.moment`), worked by hand on `fly_kit`'s two walls.

Two tasks of 10 keyframes (0-9 and 10-19), the base at (0, 0, 12) heading +x unless a test moves it, in the phase
`approach` after each task's first step. A case over the second task is the window 10..19, its stream cut after 17
and 19: the look window of look point 19 is 18-19, of 17 is 10-17. `WE`'s near face is 19.7 m from the origin, so the
corridor ahead from the body's edge (x 0.65) is 19.05 m, in the bin of the base's y; the hash is patched where a
test needs a direction.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from fly_kit import JOB, MAP, face, flight, fly_inputs, fly_world
from space_kit import record
from stbench import clear_fly, clear_move
from stbench.mapspec import Furniture, Wall
from stbench.questions_world import World

ASK = 19    # the second task's last keyframe: a look window end whose window is 18-19
WF = Wall("WF", (30.0, -10.0), (30.0, 10.0), 0.6, 30.0, True)   # a third wall, 10 m behind `WE`: box x 29.7..30.3


def moment(points, k=ASK, direction=0, script=None, job=JOB, hash_parts=None, m=None, **flight_fields):
    """The look point `k` of a case over the script's second task, the direction hash patched to `direction` (0 ahead,
    1 back); `hash_parts` collects the parts it was asked for; `m` is another place than `fly_kit`'s two walls."""
    script = script or flight(**flight_fields)
    if m is None:
        w = fly_world(script, points, a=1, job=job)
    else:
        w = World(fly_inputs(script, points, job, m), record(script, a=1))

    def stable_hash(*parts):
        if hash_parts is not None:
            hash_parts.append(parts)
        return direction

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clear_move, "stable_hash", stable_hash)
        return clear_fly.moment(w, k)


def test_a_wall_no_window_frame_saw_is_unknown_even_if_an_earlier_script_frame_did():
    """Contract (plan L2, review 2.4): the knowable rule reads the window's keyframes up to the ask, never the
    script's whole record. Hand case: keyframe 3 (the first task, before the window starts at 10) shows `WE` at
    bin 10, the very stretch the corridor ahead meets at the ask (keyframe 19), and no window frame shows it:
    the gold is `unknown` (the `unknown` pool, class `unknown`, scope `memory`), not 19.05. Regression caught:
    a script-wide record that leaks frames the case never sent."""
    m = moment({3: face(19.7, [0.0])})
    assert m.answerable == [] and len(m.unknown) == 1
    (c,) = m.unknown
    assert (c.gold, c.slice, c.answer_class, c.scope, c.evidence_k, c.range_m) == (
        {"type": "unknown"}, "unknown", "unknown", "memory", (), None)
    assert (c.family, c.decision_k, c.subjects, c.critical) == ("clear_path", ASK, (f"w:{ASK}",), True)


def test_a_wall_a_window_frame_saw_is_answerable_with_that_frame_as_the_evidence():
    """Contract: the wall the corridor stops at, seen by some window keyframe up to the ask (any stretch of it), gives
    the distance with the latest such keyframe as the evidence. Hand cases: keyframe 12 shows `WE` at bin 11 (the
    corridor meets bin 10): answerable 19.05, evidence (12,); keyframes 12 and 15 both show it: the latest, (15,);
    keyframe 12 shows `WW`, another wall, alone: `unknown`. Regression caught: the earliest frame as the evidence
    (a `now` row cited from a stale frame), or another wall's frame counted as seeing this one."""
    (c,) = moment({12: face(19.7, [1.0])}).answerable
    assert (c.gold, c.slice, c.answer_class, c.evidence_k) == ({"type": "number", "value": 19.05}, "answerable",
                                                               "answerable", (12,))
    assert moment({12: face(19.7, [1.0]), 15: face(19.7, [0.0])}).answerable[0].evidence_k == (15,)
    assert moment({12: face(-14.7, [0.0])}).unknown[0].gold == {"type": "unknown"}


def test_a_wall_seen_only_along_its_face_is_answered_but_a_seen_wall_behind_an_unseen_one_is_not():
    """Contract (final review S1): "anything you have seen" has a number when the wall the body meets was seen
    anywhere along its face, and the body touches the first wall it meets, so a seen wall beyond an unseen one
    leaves the gold `unknown`. The review's case in hand numbers: the corridor ahead from (0, 0) meets `WE`
    (19.05 m, at y 0, bin 10) and, 10 m behind it, `WF` (29.05 m). Keyframe 12 shows `WE` at y 5 (bin 15, five
    metres along) and keyframe 15 shows `WF` at y 0: answerable 19.05, evidence (12,) (not the frame of the wall
    behind), scope `memory`. With keyframe 12 showing nothing, `WF` seen alone: `unknown` (the first wall met is
    unseen), scope `memory`. With nothing shown at all: `unknown`. Regression caught: the old bin-and-neighbour
    rule (`unknown` for the first case), or the gold taken to the first seen wall past an unseen one (29.05 for the
    second, a number the body could not reach)."""
    place = replace(MAP, walls=(*MAP.walls, WF))
    along, behind = face(19.7, [5.0]), face(29.7, [0.0])
    (c,) = moment({12: along, 15: behind}, m=place).answerable
    assert (c.gold, c.evidence_k, c.scope) == ({"type": "number", "value": 19.05}, (12,), "memory")
    (u,) = moment({15: behind}, m=place).unknown
    assert (u.gold, u.scope) == ({"type": "unknown"}, "memory")
    assert moment({}, m=place).unknown


def test_no_frame_after_the_ask_is_read():
    """Contract: the gold names the window's keyframes up to the look point and no later one. Hand case: the case's
    window is 10..19, the ask at 17 (look window 10-17), and keyframe 18 (after the ask) shows the wall: `unknown`; with
    the ask at 19 the same frame is the evidence. Regression caught: a record over the whole window, which would
    answer a question from frames the system had not yet been sent."""
    points = {18: face(19.7, [0.0])}
    assert moment(points, k=17).unknown[0].gold == {"type": "unknown"}
    assert moment(points, k=19).answerable[0].evidence_k == (18,)


def test_the_scope_is_now_exactly_when_the_look_points_own_frame_saw_the_wall():
    """Contract (spec scope `evidence`, `questions_world.scope_of`): the scope is `now` exactly when the evidence
    keyframe is the look point's own frame, the latest one a system holds when it is asked, else `memory`: a frame of
    the same look window (18-19) is already an earlier frame. Hand cases at 19: seen by frame 19 alone gives `now`;
    by 18 alone, 12 alone, or 12 and 18 (the latest, 18, is the evidence) gives `memory`; by 18 and 19, the latest, 19,
    is the evidence and the scope `now`. Regression caught: the scope reading the look window again (frame 18 called
    `now`, a corridor that needs the previous message), or a `now` row answered from memory."""
    assert moment({19: face(19.7, [0.0])}).answerable[0].scope == "now"
    assert moment({18: face(19.7, [0.0])}).answerable[0].scope == "memory"
    assert moment({12: face(19.7, [0.0])}).answerable[0].scope == "memory"
    before = moment({12: face(19.7, [0.0]), 18: face(19.7, [0.0])}).answerable[0]
    assert (before.evidence_k, before.scope) == ((18,), "memory")
    both = moment({18: face(19.7, [0.0]), 19: face(19.7, [0.0])}).answerable[0]
    assert (both.evidence_k, both.scope) == ((19,), "now")


def test_a_corridor_into_open_air_is_not_asked():
    """Contract (plan L2, review 2.4): a corridor with no wall within the 60 m cap is never certified, so it is a dead
    band, whatever the frames show. Hand cases: heading +y from (0, 0) nothing stands in the way; from (-80, 0)
    heading +x the first wall box is `WW`'s, its face at x -15.3, 64.05 m from the body's edge at x -79.35, past
    60 m. Regression caught: an uncertifiable "60" asked of the drone."""
    seen = {j: np.vstack([face(19.7, [0.0]), face(-14.7, [0.0])]) for j in range(10, 20)}
    assert moment(seen, script=flight(yaws={ASK: math.pi / 2})) is None
    assert moment(seen, script=flight(yaws={ASK: 0.0}, poses={ASK: (-80.0, 0.0)})) is None


def test_below_the_fly_floor_nothing_is_asked():
    """Contract (plan L2, review 2.4, final review S2): the corridor is asked only when the base is at least
    `JobSpec.fly_floor_m` up (the kit's 8.35 m: a street lamp's 7.95 m head, the body's half height and a margin), so
    the landed drone (0.27 m), the one beside the street trees' crowns (7.01 m) and the one at a lamp's head (7.3 m)
    are never asked; one at exactly the floor is. A job with no floor asks below it too, where the camera reads the
    wall's foot to within C/2 (1.1 m up: 1.02 m; the tie of the next test). Regression caught: the
    landed drone's corridor, whose gold (walls only) is wrong beside benches, planters, people and cars, or a floor
    that lets a lamp in the way."""
    seen = {ASK: face(19.7, [0.0])}
    for z in (0.27, 7.01, 7.3, 8.34):
        assert moment(seen, base_z=z) is None, z
    assert moment(seen, base_z=8.35).answerable[0].gold == {"type": "number", "value": 19.05}
    assert moment(seen, base_z=1.1, job=replace(JOB, fly_floor_m=None)).answerable


def test_the_phase_must_be_one_of_the_jobs_clear_phases():
    """Contract: `clear_path` is asked only in the job's `clear_phases`. Hand case: keyframe 19 is in `approach`; a
    job whose phases are {"inspect"} asks nothing there, one whose phases hold `approach` does, and the first
    keyframe of a task (the start phase, `preflight`) is not in {"approach"}. Regression caught: a corridor
    asked in a phase the job does not declare (a status-answered phase)."""
    seen = {ASK: face(19.7, [0.0])}
    assert fly_world(flight(), seen, a=1).rows[10]["status"]["phase"] == "preflight"
    assert moment(seen, job=replace(JOB, clear_phases=frozenset({"inspect", "preflight"}))) is None
    assert moment(seen, job=replace(JOB, clear_phases=frozenset({"approach", "inspect"}))).answerable


def test_the_gold_is_the_raycast_from_the_edge_of_the_body_not_the_base():
    """Contract (plan L2): the corridor starts at the body's edge, the base plus the 0.65 m radius along the heading
    or its opposite. Hand case: the base at (3, 2), heading +x: the edge is at x 3.65 and `WE`'s face at 19.7
    is 16.05 m on (from the base 16.7, from the camera the same as the base's, the mount being under the base);
    back, the edge is at x 2.35 and `WW`'s face at -14.7 is 17.05 m on, and a frame that shows only `WE`, the
    wall the other way, leaves back `unknown`. Regression caught: a gold from the base or the camera (16.7, 17.7),
    or the heading's opposite not used for back."""
    at = {ASK: (3.0, 2.0)}
    ahead = moment({ASK: face(19.7, [2.0])}, poses=at)
    assert ahead.answerable[0].gold == {"type": "number", "value": 16.05}
    back = moment({ASK: face(-14.7, [2.0])}, direction=1, poses=at)
    assert back.answerable[0].gold == {"type": "number", "value": 17.05}
    assert moment({ASK: face(19.7, [2.0])}, direction=1, poses=at).unknown   # only `WE`, the wall behind the body, seen


def test_the_direction_is_the_hash_of_the_case_the_point_and_the_word_direction():
    """Contract (review 2.3, the declared direction): the direction is `stable_hash(case id, k, "direction") % 2`,
    ahead for 0 and back for 1, and nothing else decides it (the same inputs give the same direction whatever
    the frames show). Hand-worked with the hash patched: parts ("c0", 19, "direction"); an even hash asks ahead
    (slots direction "ahead", gold 19.05), an odd one back ("back", 14.05). Regression caught: a direction
    chosen from what the frames show (a question asked back only where a wall is known), which would leak the
    world."""
    parts: list = []
    ahead = moment({ASK: face(19.7, [0.0])}, hash_parts=parts)
    assert parts == [("c0", ASK, "direction")]
    assert ahead.answerable[0].slots == {"mode": "fly", "direction": "ahead", "cap": 60.0}
    back = moment({ASK: face(-14.7, [0.0])}, direction=1)
    assert back.answerable[0].slots == {"mode": "fly", "direction": "back", "cap": 60.0}
    assert back.answerable[0].gold == {"type": "number", "value": 14.05}


def test_the_row_is_graded_on_the_jobs_clearance_with_a_pixel_at_the_walls_foot():
    """Contract (plan G1): the candidate is credited `graded` on the job's clearance (the drone's C is 2.4 m, the
    harmful side the longer answer, the threshold the body's 1.3 m plus C, 3.7 m) with the resolution of one pixel at
    the wall's foot from the evidence frame's camera, at least 0.02 m; its `range_m` is the gold plus the distance from
    the camera to the corridor's start (the drone's camera is under its base, 0.65 m from the edge), an `unknown` has
    none and the least resolution, and the point is named `w:<k>`. Hand-worked: the camera at (0, 0, 11.9) (0.1 m under
    the base), the wall's foot at (19.7, 0) on the ground (the sidewalk is elsewhere), so d 19.7 and h 11.9: radial
    (19.7^2 + 11.9^2) / (11.9 x 381.36) = 0.11673, lateral sqrt(19.7^2 + 11.9^2) / 381.36 = 0.06035, together 0.1314 m;
    19.05 + 0.65 = 19.7. Regression caught: the distance report's band set from the wrong range, a resolution from the
    base or the wrong frame, or a resolution under the sub-cell."""
    (c,) = moment({ASK: face(19.7, [0.0])}, direction=0).answerable
    assert c.tolerances == {"resolution_m": pytest.approx(0.1314, abs=1e-4), "need_m": 2.4, "harm": "over",
                            "threshold_m": 3.7}
    assert c.range_m == 19.7 and c.entities == () and c.subjects == (f"w:{ASK}",)
    assert moment({}, direction=0).unknown[0].tolerances == {"resolution_m": 0.02, "need_m": 2.4, "harm": "over",
                                                            "threshold_m": 3.7}


def test_the_resolution_is_read_from_the_evidence_frame_and_the_ground_under_the_wall():
    """Contract: the pixel is the evidence keyframe's (the latest that saw the wall), at the ground under the point the
    corridor meets the wall. Hand cases: the wall seen at keyframe 12 from (0, 0, 12) and again at 15 from the base 5 m
    further on (so the camera is 5 m nearer the foot): evidence 15, d 14.7, h 11.9: (14.7^2 + 11.9^2) / (11.9 x 381.36)
    = 0.07874 radial, 0.04950 lateral, 0.0930 m. A plaza 0.15 m tall under the foot lifts the ground, so h = 11.75:
    d 19.7: (19.7^2 + 11.75^2) / (11.75 x 381.36) = 0.11817, with lateral 0.0602, 0.1326 m. Regression caught: the
    resolution of the first sighting, or the flat ground under a raised foot."""
    points = {12: face(19.7, [0.0]), 15: face(19.7, [0.0])}
    (c,) = moment(points, direction=0, poses={15: (5.0, 0.0), ASK: (5.0, 0.0)}).answerable
    assert (c.evidence_k, c.gold) == ((15,), {"type": "number", "value": 14.05})
    assert c.tolerances["resolution_m"] == pytest.approx(0.0930, abs=1e-3)
    raised = replace(MAP, furniture=(Furniture("F9", "plaza", "plaza", "fixed", (19.7, 0.0), (10.0, 10.0, 0.15), 0.0,
                                               "grey", True, ()),))
    (r,) = moment({ASK: face(19.7, [0.0])}, direction=0, m=raised).answerable
    assert r.tolerances["resolution_m"] == pytest.approx(0.1326, abs=1e-3)


def test_a_wall_foot_the_evidence_camera_reads_coarser_than_half_the_clearance_is_not_asked():
    """Contract (the corridor tie, `corridor_screens.error_tied`): an answerable corridor is asked only where one pixel
    at the wall's foot, from the evidence frame's camera, is at most C/2 = 1.2 m, as a ground corridor's stopper must
    be; a camera level with the foot's ground has no bound and is never asked. Hand cases, the wall `WE` 19.7 m on
    seen by the look point's own frame, the camera 0.1 m under the base (no fly floor): base 1.1 m up, h 1.0:
    radial (19.7^2 + 1.0^2) / (1.0 x 381.36) = 1.0203 and lateral 19.7 / 381.36 = 0.0517 give 1.0216 m, asked; base
    0.8 m up, h 0.7: 1.4557 and 0.0517 give 1.457 m, not asked; base 0.27 m up (the drone on its nest, h 0.17): 5.99 m,
    not asked;
    the same with a 0.18 m plaza under the wall (h 0.01, within `CONTACT_M`): no bound, not asked; a wall no frame saw
    stays an `unknown` row from the same low base. Regression caught: a drone that looks from the ground before
    takeoff asked a corridor its camera reads to hundreds of metres (a gold of 44.9 m whose full-credit band ran from
    22.5 m to 89.7 m), or a camera level with the ground read as the finest resolution."""
    low = replace(JOB, fly_floor_m=None)
    (c,) = moment({ASK: face(19.7, [0.0])}, job=low, base_z=1.1).answerable
    assert c.tolerances["resolution_m"] == pytest.approx(1.0216, abs=1e-3)
    assert moment({ASK: face(19.7, [0.0])}, job=low, base_z=0.8) is None
    assert moment({ASK: face(19.7, [0.0])}, job=low, base_z=0.27) is None
    plaza = replace(MAP, furniture=(Furniture("F9", "plaza", "plaza", "fixed", (19.7, 0.0), (10.0, 10.0, 0.18), 0.0,
                                              "grey", True, ()),))
    assert moment({ASK: face(19.7, [0.0])}, job=low, base_z=0.27, m=plaza) is None
    script = flight(base_z=0.27)
    w = fly_world(script, {ASK: face(19.7, [0.0])}, a=1, job=low)
    assert clear_fly.foot_rho(w, ASK, (19.7, 0.0)) == pytest.approx(5.99, abs=1e-2)
    on_plaza = World(fly_inputs(script, job=low, m=plaza), record(script, a=1))
    assert clear_fly.foot_rho(on_plaza, ASK, (19.7, 0.0)) is None
    (u,) = moment({}, job=low, base_z=0.27).unknown
    assert u.gold == {"type": "unknown"}


def test_a_look_point_may_be_any_frame():
    """Contract (0.10.0, one frame per `observe`): `k` is any frame of the window. Hand case: keyframe 15 of the window
    10..19 lies inside the look window 10-17 and ends none; `WE` shown by frame 15 alone gives 19.05 at 15, evidence
    (15,), scope `now`, point 15. Regression caught: a question refused at a frame that ends no decision window."""
    (c,) = moment({15: face(19.7, [0.0])}, k=15).answerable
    assert (c.gold, c.evidence_k, c.scope, c.decision_k) == ({"type": "number", "value": 19.05}, (15,), "now", 15)
