"""`rear_clear` (questions_space): how far the AMR could reverse straight back, as it last saw the space behind it.

All on `rear_kit`: one task of the tote AMR docking at a station facing +x, its base at (7.0, 4.98), rear edge at
x 6.6, the reverse corridor 0.58 m wide (y 4.69 to 5.27) running west from the rear edge; depth is ray-cast from the
AMR's own camera at millimetre resolution (`space_kit.render_depth`), so every expected value is worked out from the
scene by hand. The cap is 5 m (`questions_words.cap_of`).
"""

from __future__ import annotations

import json

from rear_kit import JOB, NEAR_BOX, asked, beside, lane_run, line_run, look_back, world
from space_kit import box, use_job
from stbench import questions
from stbench.questions_space import rear_clear

LADDER = {"ladder": [0.25, 0.5, 1.0, 2.0]}


def number(value: float) -> dict:
    return {"type": "number", "value": value}


def test_rear_clear_by_hand():
    """Contract: `rear_clear` is the straight corridor behind the docked body, from its rear edge along the delivered
    heading plus pi, over the latest cells of the window's keyframes, and its evidence is the keyframes that saw it.
    Hand-worked from `rear_kit.lane_run`: the robot came north along the lane x = 6.0, turned east at T (6.0, 4.98) and
    drove 1 m into the bay; docked at (7.0, 4.98) its rear edge is at x 6.6, so the corridor runs west from 6.6 to
    1.6. `NEAR_BOX`, 0.5 m tall, has its east face at x 4.49, in the sub-cell [4.48, 4.50), whose centre is 2.11 m
    behind the rear edge: the answer is 2.11 less half a sub-cell, 2.1. Frame 3
    (the base at (6.0, 2.0) heading north, a view from the side) is the only look at it; frame 7, at T facing the
    station, renders too and sees none of the corridor, so the evidence is (3,), which lies outside the decision's
    look window (offsets 16 to 20): scope `memory`. The record: cap 5 as the only slot, decision 20, the decision's time
    its only subject, critical, the distance ladder. Regression caught: the corridor taken from the arrival path (the
    lane runs north: nothing is there), from the base (2.5) or from the front edge or the camera (2.9), the wrong
    direction, or evidence read as every rendered frame."""
    (c,) = asked(lane_run(boxes=[NEAR_BOX], depth_ks={3, 7}))
    assert (c.family, c.slots, c.gold, c.slice) == ("rear_clear", {"cap": 5.0}, number(2.1), "answerable")
    assert (c.evidence_k, c.scope, c.decision_k, c.subjects, c.critical, c.entities, c.tolerances) == (
        (3,), "memory", 20, ("w:20",), True, (), LADDER)


def test_rear_clear_uses_only_the_window():
    """Contract: the corridor is the space as the window saw it; what was seen before the window's first step is no
    evidence (design §5). Hand-worked on two identical tasks of `lane_run` whose only render is frame 3 of the first
    (the lane view of `NEAR_BOX`): the decision of the second task (keyframe 43) in a window of the second task alone
    has no cell seen, so the answer is `unknown` with no evidence, scope `memory` (the slice `unknown`); in a window of
    both tasks it is the box's 2.1 from frame 3, so the box was there to be seen. Regression caught: a corridor read
    over the whole script, which hands a system evidence from before its case."""
    script = lane_run(2, boxes=[NEAR_BOX], depth_ks={3})
    alone = rear_clear(world(script, a=1, b=1), 43)
    assert (alone.gold, alone.slice, alone.evidence_k, alone.scope) == ({"type": "unknown"}, "unknown", (), "memory")
    both = rear_clear(world(script, a=0, b=1), 43)
    assert (both.gold, both.slice, both.evidence_k, both.scope) == (number(2.1), "answerable", (3,), "memory")


def test_rear_clear_reads_only_up_to_the_decision():
    """Contract: the corridor is the space as the robot last saw it by the decision: a look after it has not been sent
    yet and is no evidence. Hand-worked on `line_run` with the keyframe after the decision (offset 21, the first
    `undocking` step) turned round at the dock to look down the empty corridor and the only render: the decision at
    offset 20 has seen nothing, `unknown` with no evidence (the same look one step earlier, at offset 19, gives 5.0:
    the control that the render sees the corridor). Regression caught: a corridor read over the window's end, which
    hands the gold a view from the future (design §8.1)."""
    (before,) = asked(look_back(line_run(depth_ks={21}), 21))
    assert (before.gold, before.slice, before.evidence_k) == ({"type": "unknown"}, "unknown", ())
    seen = rear_clear(world(look_back(line_run(depth_ks={19}), 19)), 20)
    assert (seen.gold, seen.evidence_k) == (number(5.0), (19,))


def test_the_cells_behind_the_rear_edge_count_unseen():
    """Contract: no blind band is subtracted behind the body (the camera faces forward, so the cells there were seen on
    the way in or not at all): a cell nobody saw counts against the 80% rule however near the rear edge it lies.
    Hand-worked on `line_run` with offset 16 turned round at the dock (the base (7.0, 4.98), facing west) as the only
    render: its camera at x 6.62 sees the floor from 0.53 m beyond its front edge (x 6.6), so the cells with centres
    from x 6.07 to 6.6 are unseen. Empty, the corridor counts 300 cells (50 columns of 6), of which the 5 columns
    nearest the edge and the farthest one are unseen: 264 are free (88%), the answer is 5.0. With a 0.5 m box whose
    east face stands at x 5.07, 1.53 m behind the rear edge (the cell [5.0, 5.1)), the counted cells are the 15
    columns from x 5.15 to 6.55, a third of them unseen: `unknown` (with the blind band taken off, the near columns
    would be dropped and the answer 1.5). Regression caught: a blind band subtracted behind the body, which answers a
    distance over cells nobody saw."""
    empty = rear_clear(world(look_back(line_run(depth_ks={16}), 16)), 20)
    assert (empty.gold, empty.evidence_k) == (number(5.0), (16,))
    boxed = look_back(line_run(boxes=[box(4.57, 5.07, 4.73, 5.23, 0.0, 0.5)], depth_ks={16}), 16)
    gold = rear_clear(world(boxed), 20)
    assert (gold.gold, gold.slice, gold.evidence_k) == ({"type": "unknown"}, "unknown", (16,))


def test_rear_clear_is_memory_when_only_an_earlier_frame_saw_it():
    """Contract: the scope is `now` exactly when every keyframe that observed the corridor is the decision's own frame
    (`questions_world.scope_of`: a system gets one frame at a time), `memory` otherwise, a frame of the same look window
    included. Hand-worked on `rear_kit.line_run`, an empty floor, where the robot waits at (1.0, 4.98) facing the
    station and drives in 1.5 m a frame from offset 16: the cuts fall after offsets 7, 15, 20 (the decision) and 22, so
    the decision's look window is 16 to 20. Rendered at 16 alone, the camera at the far end of the corridor sees all 5 m
    of it (90% of the cells are floor): 5.0, evidence (16,), scope `memory` (frame 16 is in the decision's look window,
    but it is not frame 20); rendered at 10 alone: the same gold, evidence (10,), `memory`; rendered at both, the later
    look decides every cell, so the evidence is (16,) and the scope `memory` still. (The camera faces forward, so frame
    20 itself never sees the corridor behind: the `now` side is `test_compile`'s `scope_of` case.) Regression caught:
    the scope reading the look window again (16 called `now`), or an old look winning over a later one."""
    got = [rear_clear(world(line_run(depth_ks=ks)), 20) for ks in ({16}, {10}, {10, 16})]
    assert [(c.gold, c.evidence_k, c.scope) for c in got] == [
        (number(5.0), (16,), "memory"), (number(5.0), (10,), "memory"), (number(5.0), (16,), "memory")]


def test_rear_clear_hidden_changes():
    """Contract: the gold is what the depth recorded, never the world: a thing that stands where the robot last saw
    floor changes nothing, one a frame saw moves the gold to it. Hand-worked on `lane_run` with `NEAR_BOX` seen from
    frame 3 (gold 2.1, evidence (3,)): a pallet that arrives at t 6 s, after that frame (1.5 s a keyframe: 4.5 s), 0.5
    m wide where the corridor passes at x 4.97 to 5.47, and is in no depth image: the gold is still 2.1. The same
    pallet in the scene frame 3 renders: its east face at x 5.49 is in the sub-cell [5.48, 5.50), centre 1.11 m from
    the rear edge, less 0.01: 1.1. Regression caught: a free-space gold read from entity footprints or placements, which
    answers for a world no sensor recorded."""
    pallet = box(4.97, 5.49, 4.73, 5.23, 0.0, 0.5)
    script = lane_run(boxes=[NEAR_BOX], depth_ks={3})
    script.item("E0001", "pallet", {"color": "white"}, [(6.0, None, (5.22, 4.98, 0.0))], size=(0.5, 0.5, 0.5))
    (unseen,) = asked(script)
    assert (unseen.gold, unseen.evidence_k) == (number(2.1), (3,))
    (seen,) = asked(lane_run(boxes=[NEAR_BOX, pallet], depth_ks={3}))
    assert (seen.gold, seen.evidence_k) == (number(1.1), (3,))


def test_a_floor_marking_never_stops_the_corridor():
    """Contract: paint is floor to the AMR (M3-maps §1: markings are at most 0.005 m, under its 0.01 m climb even after
    the depth's millimetre rounding), so a marking across the corridor does not stop it and the answer runs on to the
    cap. Hand-worked on `rear_kit.line_run`, frames 16 (the base at (1.0, 4.98)) and 18 (at (4.0, 4.98)) rendered: a
    stripe 0.16 m deep (x 5.03 to 5.19, 1.41 m behind the rear edge) wider than the corridor, 0.005 m high, gives 5.0.
    The control, the same stripe 0.05 m high, is seen from the west by its top and stops the corridor at the sub-cell
    centred 5.19: 6.6 - 5.19 - 0.01 = 1.4, so the stripe is in the render and the corridor passes through it. Why 0.005
    and not 0.01 m (the AMR's climb, which a bay's old 0.01 m stripe equalled): the depth is read to a millimetre, the
    rounding lifts the stripe's points a hair above 0.01 m, and the corridor stops at it (measured here: 1.4 at 0.01 m
    too; the plan review's M2). Regression caught: a painted lane or bay read as an obstacle at every undock."""
    def gold(height: float):
        return rear_clear(world(line_run(boxes=[box(5.03, 5.19, 3.5, 6.5, 0.0, height)], depth_ks={16, 18})), 20).gold

    assert gold(0.005) == number(5.0)
    assert gold(0.05) == number(1.4)


def test_the_corridor_is_the_bodys_width():
    """Contract: only what stands within the body's width (half 0.29 m) stops the corridor, judged at the 0.02 m
    sub-cell, not the 0.1 m cell. Hand-worked on `lane_run` with `NEAR_BOX` (gold 2.1) and a 0.5 m box beside the
    corridor, x 4.97 to 5.49, whose south side is `offset` north of the axis (y 4.98): the fine cell [5.2, 5.3) of y has
    its centre 0.27 m off the axis, inside the half-width, and holds both sides. At 0.31 (0.02 outside) the box's side
    lies in sub-cell [5.28, 5.30), centre 0.31 off the axis: the gold stays 2.1. At 0.27 (0.02 inside) its sub-cell
    centre is 0.27 off and the box stops the corridor at its east end, the sub-cell centred 5.49: 6.6 - 5.49 - 0.01 =
    1.1. Regression caught: the corridor's 0.1 m cells counting what stands beside the body (M1's door-frame error),
    which stops the first at 1.1 too, or the width ignored."""
    def gold(offset: float):
        return rear_clear(world(lane_run(boxes=[NEAR_BOX, beside(offset)], depth_ks={3})), 20).gold

    assert gold(0.31) == number(2.1)
    assert gold(0.27) == number(1.1)


def test_rear_clear_compiles_from_the_written_case(tmp_path, monkeypatch):
    """Contract: the compile from a case's files (the depth PNGs, `camera.json`, the pose rows the stream delivers)
    makes the same question and gold row as the in-memory one, and the rows pass the harness's validators: text
    rendered from the slot alone, template `rear_clear`, `needs` depth and pose, scope and evidence in the gold row,
    the corridor's heading read from the delivered `yaw_deg`. Hand values as the by-hand test: 2.1, evidence [3],
    `memory`; with the delivered yaw of the decision's row turned to 180 degrees the corridor runs east from the
    front edge, over cells nothing saw: `unknown`. Regression caught: a gold the written files do not reproduce (the
    oracle's), or a row the validators refuse."""
    use_job(monkeypatch, JOB)
    case = lane_run(boxes=[NEAR_BOX], depth_ks={3}).write_case(tmp_path)
    (q,), (g,) = questions.compile_case(case)
    assert (q["template"], q["decision_k"], q["slots"], q["needs"]) == (
        "rear_clear", 20, {"cap": 5.0}, ["see", "pose"])
    assert q["text"] == ("If you reversed straight back now, how far could you go before reaching anything you have "
                         "seen? Answer 5 if nothing is within 5 m.")
    assert (g["gold"], g["slice"], g["scope"], g["evidence_k"], g["subjects"], g["credit"]) == (
        number(2.1), "answerable", "memory", [3], ["w:20"], "ladder")
    rows = [json.loads(line) for line in case.script.pose.read_text(encoding="utf-8").splitlines()]
    rows[20]["pose"]["yaw_deg"] = 180.0
    case.script.pose.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert questions.compile_case(case)[1][0]["gold"] == {"type": "unknown"}
