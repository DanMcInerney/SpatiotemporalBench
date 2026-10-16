"""`where_was_i` (questions_self): the robot's base at a clock a minute or more back, by hand on `self_kit`'s tasks of
96 keyframes (1.5 s apart) with decision points at offsets 27, 40 and 70 of a task (t = 40.5, 60 and 105 s). The
stream cuts a look window after every 8 steps and after each decision point and task end. The draws are recomputed here
with `seeding.stable_hash`; every expected position and clock is worked out from the script."""

from __future__ import annotations

import pytest

import family_kit as fk
import synth
from self_kit import JOB, TASK_FRAMES, self_script, world
from space_kit import box
from space_kit import record as space_record
from stbench import case, questions
from stbench.questions_self import MIN_AGO_S, where_was_i
from stbench.seeding import stable_hash, stable_hex
from synth import clock_of

LADDER = {"ladder": [0.25, 0.5, 1.0, 2.0]}


def ids(tag: str, n: int = 50) -> list[str]:
    return [stable_hex(tag, i, n=16) for i in range(n)]


def test_where_was_i_reads_the_base():
    """Contract: the gold is the delivered base's (x, y) at the named step, 4 decimals, and the question holds exactly
    the row the compiler writes: slot `clock`, evidence the step, no entity, the distance ladder, subjects the
    decision's and the step's time, answerable, critical. Hand-worked: the step 0 base moved to (1.234567, 0.765432)
    with the camera 0.13 m ahead of it (the vacuum's mount; heading 0) at (1.364567, 0.765432): at the decision 40 (60
    s on, the one step old enough) the gold is [1.2346, 0.7654], never the camera's [1.3646, 0.7654]; at the decision
    70 the 31 steps 0 to 30 are old enough, each with a base of its own, and the drawn one's is the gold. Regression
    caught: the camera's pose used for the robot's position (map X-H20), or a base read from the timeline's floats and
    not the delivered row."""
    bases = {0: (1.234567, 0.765432), **{s: (round(0.5 + 0.1 * s, 3), 0.75) for s in range(1, 31)}}
    w = world(self_script(poses=bases))
    assert w.rows[0]["camera"]["pos"][:2] == pytest.approx([1.364567, 0.765432])
    c = where_was_i(w, 40)
    assert (c.slots, c.gold, c.evidence_k) == (
        {"clock": clock_of(0)}, {"type": "position", "xy": [1.2346, 0.7654]}, (0,))
    s = stable_hash("c0", 70, "where_was_i") % 31
    c = where_was_i(w, 70)
    assert c.evidence_k == (s,)
    assert c.gold == {"type": "position", "xy": [round(bases[s][0], 4), round(bases[s][1], 4)]}
    assert (c.family, c.slice, c.entities, c.tolerances, c.decision_k, c.subjects, c.critical) == (
        "where_was_i", "answerable", (), LADDER, 70, ("w:70", f"w:{s}"), True)


def test_the_named_step_is_old_enough():
    """Contract: the named step lies at least 60 s before the decision; with none the point is a dead band (None).
    Hand-worked for 50 case ids: decision 27 (40.5 s into the window) has no step 60 s back: None; decision 40 (exactly
    60 s) names step 0 and no other; decision 70 draws from the steps 0 to 30 (105 - 1.5 s >= 60; its look window,
    65-70, is clear of them), and the draws spread (15 or more distinct steps of 31). Regression caught: a boundary (60
    s) moved either way, or a clock too recent to need a memory."""
    script = self_script()
    inp = fk.inputs(script, JOB)
    drawn_at_70 = set()
    for cid in ids("old"):
        w = world(script, cid, inp=inp)
        assert MIN_AGO_S == 60.0 and where_was_i(w, 27) is None
        assert where_was_i(w, 40).evidence_k == (0,)
        (s,) = where_was_i(w, 70).evidence_k
        assert 0 <= s <= 30
        drawn_at_70.add(s)
    assert len(drawn_at_70) >= 15


def test_a_step_inside_the_decisions_look_window_is_not_named(monkeypatch):
    """Contract: the named step lies before the look window the stream ends at the decision, even when it is 60 s back
    (what the system was just sent, which the current view answers). Hand-worked with the keyframes 30 s apart and one
    decision at 12: the stream sends 0-7 and then 8-12, and the steps 0 to 10 are 60 s or more back (30 s * (12 - s)
    >= 60), but only 0 to 7 are before the look window: for 50 ids the drawn step is among those, and some of the ids
    would have drawn 8, 9 or 10 (by the hash's draw over the 11 steps) without the rule. Regression caught: a past clock
    inside the current look window."""
    monkeypatch.setattr(synth, "DT", 30.0)
    script = self_script(decisions=(12,))
    inp = fk.inputs(script, JOB)
    assert [script.timeline().keyframe(k).t for k in (0, 10, 12)] == [0.0, 300.0, 360.0]
    pooled = ids("batch")
    assert any(stable_hash(cid, 12, "where_was_i") % 11 >= 8 for cid in pooled)
    for cid in pooled:
        assert where_was_i(world(script, cid, inp=inp), 12).evidence_k[0] <= 7


def test_the_clock_is_a_delivered_clock():
    """Contract: the slot is the case clock the stream delivered at the named step (`case.case_rows`: the case's start
    clock plus the time since the window's first step), and the question text renders it. Hand-worked on tasks 1 and 2
    of a 3-task script (a window that starts 7200 s into the script), decision 262 (offset 70 of the last task, look
    window 257-262): steps 96 to 222 are old enough, the whole first task and offsets 0 to 30 of the second. For 50 ids
    the slot is START plus the case time (1.5 s a step of the first task, 7200 s plus that in the second, to the tenth
    of a second), equals the stream's row clock, and the text is "Where were you at <clock>?"; both tasks are named by
    some id.
    Regression caught: a clock the stream never sent (script time, 10:00 and later, leaking), or a text that differs
    from its slot."""
    script = self_script(3)
    inp = fk.inputs(script, JOB)
    named_tasks = set()
    for cid in ids("clock"):
        w = world(script, cid, a=1, b=2, inp=inp)
        c = where_was_i(w, 262)
        (s,) = c.evidence_k
        assert s == 96 + stable_hash(cid, 262, "where_was_i") % 127   # steps 96-222: 127 of them
        task, offset = divmod(s, TASK_FRAMES)
        named_tasks.add(task)
        expected = clock_of(1.5 * offset + 7200.0 * (task - 1))
        assert c.slots == {"clock": expected}
        assert case.case_rows(w.rec, w.tl, w.inp.poses)[s - w.rec.k0]["clock"] == expected
        assert questions.records([c])[0][0]["text"] == f"Where were you at {expected}?"
    assert named_tasks == {1, 2}


def test_where_was_i_draw_ignores_the_world():
    """Contract: the named step depends on the case id, the decision and the window's time alone, never on the world.
    Hand-worked: for 50 ids the same script with things that move and are seen, a wall in the depth, another seed and
    every base moved elsewhere draws the same step at decisions 40 and 70 (the gold follows the moved base: (3, 2) at
    every step). Regression caught: the clock asked revealing the world."""
    bare = self_script()
    busy = self_script(seed=9, depth_ks=None, boxes=[box(2.0, 2.2, -3.0, 3.0, 0.0, 2.0)],
                       poses={k: (3.0, 2.0) for k in range(TASK_FRAMES)})
    busy.item("E0001", "bag", {"color": "red"}, [(0, 30.0, (1.0, 1.0, 0.0)), (30.0, None, (2.0, 1.0, 0.0))])
    busy.see([5, 40, 60], "E0001", 1500)
    inp, busy_inp = fk.inputs(bare, JOB), fk.inputs(busy, JOB)
    for cid in ids("world"):
        for k in (40, 70):
            a, b = where_was_i(world(bare, cid, inp=inp), k), where_was_i(world(busy, cid, inp=busy_inp), k)
            assert (a.evidence_k, a.slots) == (b.evidence_k, b.slots)
            assert b.gold == {"type": "position", "xy": [3.0, 2.0]}


def test_nothing_before_the_window_is_named():
    """Contract: only steps of the window can be named, since the stream delivers no clock before the case's first step
    (design section 5). Hand-worked on a 2-task script with a decision at offset 27 of each task: a window of the second
    task alone, at its decision 123 (40.5 s in), has no step 60 s back and the point is not asked, though the first
    task lies 7200 s earlier; a window of both tasks asks it and names a step of the first task (0 to 95) for 50 ids,
    and its first decision, 27, is not asked. Regression caught: a clock from before the case, which no system was
    sent."""
    script = self_script(2, decisions=(27,))
    inp = fk.inputs(script, JOB)
    assert where_was_i(world(script, a=1, inp=inp), 123) is None
    for cid in ids("before"):
        both = world(script, cid, a=0, b=1, inp=inp)
        assert where_was_i(both, 27) is None
        assert 0 <= where_was_i(both, 123).evidence_k[0] < TASK_FRAMES


def test_a_job_asks_where_was_i_as_a_memory_own_path_position():
    """Contract: a job that names the family at a decision kind gets its rows as `compile` writes them, valid and typed:
    during, a position answer scored on the ladder with metric `xy`, scope `memory`, reference `path_log`, needs pose,
    no region. Hand-worked: decisions 27 and 70 of a one-task window are both drawn (43 steps apart); 27 is a dead band
    and not asked, 70 is asked and draws from the steps 0 to 30, its clock START plus 1.5 s a step and its gold the
    script's base for the step, (0.5 + 0.05 s, 0.5). Regression caught: a family declared with the wrong scope,
    reference or credit, a dead band asked with a substitute, or a row the validators refuse."""
    script = self_script(decisions=(27, 70))
    qs, gs = questions.compile_inputs(fk.inputs(script, JOB), space_record(script, cid="c0"))
    s = stable_hash("c0", 70, "where_was_i") % 31
    clock = clock_of(1.5 * s)
    assert [(q["when"], q["decision_k"], q["template"], q["answer_type"], q["needs"]) for q in qs] == [
        ("during", 70, "where_was_i", "position", ["pose"])]
    assert qs[0]["slots"] == {"clock": clock} and qs[0]["text"] == f"Where were you at {clock}?"
    g = gs[0]
    assert (g["family"], g["slice"], g["credit"], g["region"], g["tolerances"], g["metric"], g["scope"], g["reference"],
            g["critical"], g["subjects"], g["evidence_k"], g["entities"]) == (
        "where_was_i", "answerable", "ladder", None, LADDER, "xy", "memory", "path_log", True, ["w:70", f"w:{s}"], [s],
        [])
    assert g["gold"] == {"type": "position", "xy": [round(0.5 + 0.05 * s, 4), 0.5]}
