"""`person_where` (questions_change): where a person was as the robot last saw them, by hand on `change_kit`'s tasks of
`vacuum_run` (28 keyframes each; events at offsets 2, 7, 18 and 26 of a task; the window's last look window is the last
task's offsets 22 to 27)."""

from __future__ import annotations

from dataclasses import replace

import change_kit as ck
from change_kit import person
from space_kit import vacuum_run
from stbench import questionnaire
from stbench.questions_change import movers, person_where
from stbench.questions_text import COINS, FAMILIES, SLICED
from stbench.seeding import stable_hex

ONLY = replace(ck.CHANGE_JOB, ends=("person_where",))


def asked(w) -> dict[str, list[tuple]]:
    """Each moment of `person_where` that asks something: (entity, gold xy, evidence) of its candidates."""
    return {m.key: [(c.entities[0], tuple(c.gold["xy"]), c.evidence_k) for c in m.answerable]
            for m in person_where(w) if m.answerable}


def resident_script():
    """Three tasks. P1 a person (no attributes) at (1, 1) through task 0, (3, 2) through task 1 and (5, 1) in task 2
    (moves in the gaps), seen at keyframes 1 and 29 only; P2 and P3 two identical yellow tote robots (movers of one
    description), both seen at keyframe 1."""
    s = vacuum_run(3)
    person(s, "P1", [(0.0, (1.0, 1.0)), (ck.GAP_S, (3.0, 2.0)), (ck.GAP2_S, (5.0, 1.0))], [1, 29])
    person(s, "P2", [(0.0, (6.0, 1.0))], [1], kind="tote_robot", colour="yellow")
    person(s, "P3", [(0.0, (6.0, 2.0))], [1], kind="tote_robot", colour="yellow")
    return s


def test_person_where_asks_a_unique_mover_seen_in_the_moments_own_task():
    """Contract: `person_where` asks a mover whose description is the only one of its kind in the world, at a moment of
    a task in which its last recognisable sighting at or before the moment lies; the gold is the placement at that
    sighting, its evidence; the unknown pool is empty (no slice). Hand values on `resident_script`: P1 is asked at
    every event of task 0 (gold (1, 1), evidence keyframe 1) and of task 1 (gold (3, 2), evidence 29: keyframe 29 is
    task 1's offset 1, before its events at 30, 35, 46 and 54), and at no event of task 2, where its last sighting
    (29) lies in the task before; P2 and P3, two movers of one description, are never asked, though seen. Regression
    caught: a gold read across a gap (the world changes in the gap, so an earlier task's sighting is not the person's
    place now), a person named by a description another mover shares, or a pool that holds a slice that was not
    drawn."""
    w = ck.window(resident_script(), 0, 2)
    assert [e.id for e in movers(w)] == ["P1"]
    want = {f"{task}|{wording}": [("P1", xy, (k,))] for task, xy, k in ((0, (1.0, 1.0), 1), (1, (3.0, 2.0), 29))
            for wording in ck.WORDINGS}
    assert asked(w) == want
    assert len(person_where(w)) == 12
    for moment in person_where(w):
        assert moment.unknown == []
        assert all(c.slice == "answerable" and c.family == "person_where" for c in moment.answerable)


def test_a_person_seen_moving_is_asked_only_where_the_robot_knew_the_place():
    """Contract: the place of a person is steady up to the moment (`World.steady`): every sighting from the last
    recognisable one to the moment, a glimpse or a dark one too, lies within 0.15 m of it; a moment that fails is not
    asked, and a later recognisable sighting makes the place that sighting's. Hand-worked on one task, a person at
    (1, 1) who walks to (1, 2) at t 6 s (keyframe 4), seen recognisably at 1 (1, 1), as a 500 px glimpse at 5 (1, 2) and
    recognisably at 9 (1, 2): 'when you set off' (keyframe 2) is asked with (1, 1) (nothing else seen by then); 'when
    you began sweeping' (7) is not (the glimpse at 5 is a metre from the last recognisable place); 'when you finished
    sweeping' (18) and 'when you got home' (26) are asked with (1, 2), evidence 9. Regression caught: a gold the
    robot could not know (a person seen going elsewhere since), or a recognisable sighting that does not replace the
    last."""
    s = vacuum_run()
    person(s, "P1", [(0.0, (1.0, 1.0)), (6.0, (1.0, 2.0))], [1, 9])
    s.see(5, "P1", 500)
    w = ck.window(s)
    assert asked(w) == {f"0|{ck.SET_OFF}": [("P1", (1.0, 1.0), (1,))], f"0|{ck.FINISHED}": [("P1", (1.0, 2.0), (9,))],
                        f"0|{ck.HOME}": [("P1", (1.0, 2.0), (9,))]}


def test_a_person_in_the_last_look_window_is_not_asked():
    """Contract: a mover with any visibility row in the window's last look window is never asked (the current view would
    answer), at any pixel count. Hand-worked: the person of the steady test, seen recognisably at keyframe 1, is asked
    at all four events of the one task (gold (1, 1), the last look window being keyframes 22 to 27); with a 100 px
    glimpse at keyframe 24 it is asked at none. Regression caught: a memory question the current view answers (VISION
    rule 4)."""
    s = vacuum_run()
    person(s, "P1", [(0.0, (1.0, 1.0))], [1])
    assert sorted(asked(ck.window(s))) == sorted(f"0|{wording}" for wording in ck.WORDINGS)
    s.see(24, "P1", 100)
    assert asked(ck.window(s)) == {}


def test_one_question_a_case_with_no_slice_and_the_text():
    """Contract: `person_where` is a temporal family, neither sliced nor a coin; the person is a subject, so the
    questionnaire asks one question about them a case whatever the moments; the text names the person by description:
    'Where was the person, as you last saw them, when you began sweeping in your second task?'. Hand-worked on
    `resident_script` over 30 case ids with a job asking only `person_where`: exactly one question, about P1, whose gold
    is P1's place at the moment it names (evidence 1 or 29), and a window of one task has the slot `task` None and no
    task phrase. Regression caught: the same person asked twice in a case, a slice the family does not have, or a text
    that names a task of a window of one."""
    assert FAMILIES["person_where"].temporal and "person_where" not in SLICED and "person_where" not in COINS
    assert FAMILIES["person_where"].slots == ("kind", "attributes", "event", "task")
    for cid in (stable_hex("one person", i, n=16) for i in range(30)):
        got = questionnaire.end(ck.window(resident_script(), 0, 2, cid, job=ONLY))
        (c,) = got
        assert c.subjects[0] == "e:P1" and c.entities == ("P1",)
        task = c.slots["task"]
        assert c.gold == {"type": "position", "xy": [1.0, 1.0] if task == 0 else [3.0, 2.0]}
        assert c.evidence_k == ((1,) if task == 0 else (29,))
    (c,) = [c for m in person_where(ck.window(resident_script(), 0, 2)) if m.key == f"1|{ck.BEGAN}"
            for c in m.answerable]
    assert FAMILIES["person_where"].render(c.slots) == (
        "Where was the person, as you last saw them, when you began sweeping in your second task?")
    s = vacuum_run()
    person(s, "P1", [(0.0, (1.0, 1.0))], [1])
    (alone,) = [c for m in person_where(ck.window(s)) if m.key == f"0|{ck.SET_OFF}" for c in m.answerable]
    assert alone.slots["task"] is None
    assert FAMILIES["person_where"].render(alone.slots) == (
        "Where was the person, as you last saw them, when you set off?")


def test_the_ladder_is_widened_for_a_large_person():
    """Contract: a `person_where` row carries the distance ladder plus `rules.widening` of the person's own box (a
    mover's kind is not in the item catalog). Hand values: a box 0.5 x 0.5 does not widen it (0.25, 0.5, 1, 2); a box
    1.2 x 0.6 widens it by max(0, 1.2 / 2 - 0.25) = 0.35 (0.6, 0.85, 1.35, 2.35). Regression caught: the catalog's size
    asked for a kind it lacks (a KeyError), or a ladder that ignores the person's size."""
    s = vacuum_run()
    person(s, "P1", [(0.0, (1.0, 1.0))], [1])
    (c,) = [c for m in person_where(ck.window(s)) if m.key == f"0|{ck.SET_OFF}" for c in m.answerable]
    assert c.tolerances == {"ladder": [0.25, 0.5, 1.0, 2.0]}
    s = vacuum_run()
    person(s, "P1", [(0.0, (1.0, 1.0))], [1])
    s.entities[0] = replace(s.entities[0], size=(1.2, 0.6, 1.7))
    (c,) = [c for m in person_where(ck.window(s)) if m.key == f"0|{ck.SET_OFF}" for c in m.answerable]
    assert c.tolerances == {"ladder": [0.6, 0.85, 1.35, 2.35]}
