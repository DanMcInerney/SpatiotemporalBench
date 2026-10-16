"""`path_ahead` (questions_space): the during family's gold from the stream's look window, at the delivered pose."""

from __future__ import annotations

import json
from dataclasses import replace

from space_kit import (
    FRONT_EDGE_ORIGIN,
    box,
    inputs,
    record,
    use_job,
    vacuum_run,
)
from stbench import freespace, grid, questionnaire
from stbench.case import read_case
from stbench.questions_words import cap_of
from stbench.questions_world import World, load_inputs
from stbench.robots import ROBOTS, blind_m

WALL_ONLY = [box(2.21, 2.43, -3.0, 3.0, 0.0, 2.0)]
BOX_AHEAD = box(1.41, 1.73, -0.15, 0.15, 0.0, 0.3)


def asked(script) -> list:
    """The during questions of a one-task run: with `vacuum_run`'s decisions that is one, at keyframe 21."""
    return questionnaire.during(World(inputs(script), record(script)))


def facing(*ks):
    """Poses and headings that put keyframes `ks` where `vacuum_run` puts its decision's look window."""
    return {"poses": dict.fromkeys(ks, FRONT_EDGE_ORIGIN), "yaws": dict.fromkeys(ks, 0.0)}


def test_the_cap_stays_inside_the_depth_range():
    """Contract: the question's cap is 5 m or a metre inside the robot's depth range, so a corridor answered `cap` was
    seen to its end. Hand values from the robots' ranges: the vacuum 4 - 1 = 3, the AMR min(5, 9) = 5, the drone
    min(5, 599) = 5. Regression caught: a cap past `far_m`, which asks about space no pixel can show."""
    assert [cap_of(ROBOTS[r]) for r in ("vacuum", "amr", "drone")] == [3.0, 5.0, 5.0]


def test_a_decision_asks_path_ahead_with_the_gold_of_the_rendered_scene():
    """Contract: the question at a decision point is `path_ahead` with the cap as its only slot, the look window's
    ray-cast depth as its gold and the whole record the harness reads. Hand-worked from the scene (the front edge at x
    0, a wall whose face is at x 2.21, so its sub-cell is [2.20, 2.22), centre 2.21, less 0.01): 2.2 with a clear floor
    before it; with a 0.3 m box at x 1.41 (sub-cell centre 1.41, less 0.01) 1.4. Both golds are answerable, critical,
    scope-free of subjects but the decision's own time, and carry the distance ladder on the row. Regression caught: a
    corridor from the camera (1.44 and 2.24 instead), or a gold that misses an obstacle the depth shows."""
    clear = asked(vacuum_run())
    assert len(clear) == 1
    c = clear[0]
    assert (c.family, c.slots, c.decision_k, c.gold, c.slice) == ("path_ahead", {"cap": 3.0}, 21,
                                                                  {"type": "number", "value": 2.2}, "answerable")
    assert (c.subjects, c.critical, c.entities, c.tolerances) == (("w:21",), True, (),
                                                                  {"ladder": [0.25, 0.5, 1.0, 2.0]})
    blocked = asked(vacuum_run(boxes=[*WALL_ONLY, BOX_AHEAD]))
    assert blocked[0].gold == {"type": "number", "value": 1.4}


def test_path_ahead_reads_only_the_stream_look_window():
    """Contract: a `now` gold reads the look window the stream ends at the decision point and no other: keyframes 19 to
    21 (the stream is cut after the decision at 21, and the look window before it ended at 18). Hand-worked on the clear
    scene (gold 2.2): the corridor rendered only at keyframes 14 to 18, before the look window, leaves the look window
    with nothing seen: unknown, no evidence; rendered only at keyframe 20, inside it, the answer is 2.2 with evidence
    [20]; rendered at 18 and 20 the evidence is still [20] alone. Regression caught: a `now` gold from other look
    windows than the stream's (map D-20), which an agent that was sent only the look window could not reproduce."""
    before = {"depth_ks": set(range(14, 19)), **facing(*range(14, 22))}
    assert [(c.gold, c.slice, c.evidence_k) for c in asked(vacuum_run(**before))] == [({"type": "unknown"},
                                                                                       "unknown", ())]
    inside = asked(vacuum_run(depth_ks={20}, **facing(*range(14, 22))))
    assert [(c.gold, c.evidence_k) for c in inside] == [({"type": "number", "value": 2.2}, (20,))]
    both = asked(vacuum_run(depth_ks={18, 20}, **facing(*range(14, 22))))
    assert [(c.gold, c.evidence_k) for c in both] == [({"type": "number", "value": 2.2}, (20,))]


def test_the_latest_frame_of_the_look_window_decides_a_cell():
    """Contract: where two keyframes of the look window observed a cell, the later state stands (a thing that left, or
    arrived, between them). Hand-worked: the 0.3 m box at x 1.41 is rendered at keyframe 19 and gone at 20 and 21: the
    corridor is clear again and the gold 2.2, evidence 21 (the cells' latest frames); rendered at 21 only it is 1.4.
    Regression caught: the first sighting or an occupied cell winning over a later free one."""
    gone = asked(vacuum_run(frame_boxes={19: [BOX_AHEAD]}))[0]
    assert (gone.gold, gone.evidence_k) == ({"type": "number", "value": 2.2}, (21,))
    arrived = asked(vacuum_run(frame_boxes={21: [BOX_AHEAD]}))[0]
    assert arrived.gold == {"type": "number", "value": 1.4}


def test_a_corridor_that_was_not_seen_is_asked_with_the_gold_unknown():
    """Contract: a drawn point whose rule fails is asked anyway with gold `unknown` (nothing is skipped), and its
    evidence is what little was seen. Hand-worked: the look window rendered with no wall and an empty floor, for a
    vacuum whose depth range asks 4 m (0.08 m camera: the floor is seen to about 2.3 m, 23 columns of 0.1 m from the
    front edge at x 0) counts 154 cells of the disc's sweep (u 1 to 39, four rows, less the last column's two at |y|
    0.15, past the cap) of which 92 are free (59.7%, under the 80% rule): unknown, with the evidence [21] of the cells
    that were seen (the 3 m cap of the real vacuum sees 80.7% and answers 3: `test_freespace`). Regression caught: an
    unobserved corridor read as empty (map X-H5) or a drawn point dropped, which makes the asked points depend on the
    world."""
    vacuum = ROBOTS["vacuum"]
    script = vacuum_run(boxes=[])
    wide = replace(inputs(script), robot=replace(vacuum, depth=replace(vacuum.depth, far_m=5.0)))   # cap 4 m
    open_floor = questionnaire.during(World(wide, record(script)))
    assert [(c.gold, c.slice) for c in open_floor] == [({"type": "unknown"}, "unknown")]
    assert open_floor[0].evidence_k == (21,)


def test_the_gold_reads_the_delivered_pose(tmp_path, monkeypatch):
    """Contract: the gold's heading and base are the delivered pose row's (`sensors/pose.jsonl`, `yaw_deg` rounded to
    3 places), never the timeline's own floats, so the oracle, which reads the stream, reproduces it. Hand-worked from
    the written script: the row of keyframe 21 says the robot heads 90 degrees (north) where the timeline says 0; the
    corridor then runs north from (-0.17, 0.17), over cells the camera facing east never saw, so the gold is unknown,
    where the timeline's heading gives 2.2. Regression caught: an oracle that cannot reproduce its gold because the
    compile read the timeline."""
    use_job(monkeypatch)
    script = vacuum_run()
    case = script.write_case(tmp_path)
    assert questionnaire.during(World(load_inputs(case), read_case(case)))[0].gold == {"type": "number", "value": 2.2}
    rows = [json.loads(line) for line in case.script.pose.read_text(encoding="utf-8").splitlines()]
    rows[21]["pose"]["yaw_deg"] = 90.0
    case.script.pose.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert questionnaire.during(World(load_inputs(case), read_case(case)))[0].gold == {"type": "unknown"}


def test_the_look_window_states_come_from_the_scripts_files(tmp_path):
    """Contract: the gold's cells are `freespace.script_states` of the written keyframes: for the decision's look window
    the same fine cells the in-memory rule gives, and `grid.latest` over them is what `path_ahead` read. Regression
    caught: the compile and the free-space reader disagreeing on a pixel, so a system that maps from the stream misses
    the gold."""
    script = vacuum_run()
    paths = script.write(tmp_path)
    from_files = freespace.script_states(paths, [19, 20, 21])
    assert from_files == inputs(script).states_of([19, 20, 21])
    answer, evidence = grid.path_ahead(grid.latest(from_files), ROBOTS["vacuum"], FRONT_EDGE_ORIGIN, 0.0, 3.0)
    assert (answer, evidence) == ({"type": "number", "value": 2.2}, [21])
    assert all(from_files[21][(22, v)] == "occupied" for v in range(-2, 2))
    robot = ROBOTS["vacuum"]  # `path_ahead` is `corridor_clear` from the front edge, past the camera's blind distance
    assert grid.corridor_clear(grid.latest(from_files), robot, grid.front_edge(robot, FRONT_EDGE_ORIGIN, 0.0), 0.0,
                               3.0, blind_m(robot)) == (answer, evidence)


def test_states_are_computed_once(tmp_path, monkeypatch):
    """Contract: a keyframe's free-space cells are computed once per compile, whichever families read them: two reads
    of the decision's look window (keyframes 19 to 21) by `latest_cells`, the during question and the candidates trigger
    one `freespace.script_states` call for those three keyframes, and a read of 14 and 19 computes only 14. Regression
    caught: a long-window compile reading each depth image once per question."""
    use_job(monkeypatch)
    case = vacuum_run().write_case(tmp_path)
    calls, real = [], freespace.script_states
    monkeypatch.setattr(freespace, "script_states", lambda paths, ks: calls.append(list(ks)) or real(paths, ks))
    w = World(load_inputs(case), read_case(case))
    first = w.latest_cells(w.window_ks(21))
    assert w.latest_cells(range(19, 22)) == first
    assert questionnaire.during(w)[0].gold == {"type": "number", "value": 2.2}
    assert [c.family for c in questionnaire.candidates(w)].count("path_ahead") == 1  # decision 21: 5 asks no family
    assert calls == [[19, 20, 21]]
    w.latest_cells([14, 19])
    assert calls[1:] == [[14]]
