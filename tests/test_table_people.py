"""The table systems' and `path_log`'s answers to `closest_person` and `person_mem` (`table_people`,
`baselines_path_people`), and that the oracle and `memory_table` reproduce the golds over a compiled case. Hand-worked:
the vacuum's camera is mounted 0.13 m ahead of its base, so a base at (2.0, 1.5) heading 90 degrees has its camera at
(2.0, 1.63) and one at (2.0, 1.0) at (2.0, 1.13); a person is a 0.34 x 0.56 m footprint."""

from __future__ import annotations

import pytest

import person_kit as pk
from baseline_kit import EVENTS, det, offset, play, question, scores, started, step
from ego_kit import K
from stbench import episode as ep
from stbench import questions, questions_people, table_answers, table_people
from stbench.baselines import MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.person_seen import PERSON_KIND
from stbench.table import Store

A = [[1.83, 3.35], [2.17, 3.35], [2.17, 3.91], [1.83, 3.91]]   # a person at (2.0, 3.63)
B = [[1.83, 2.35], [2.17, 2.35], [2.17, 2.91], [1.83, 2.91]]   # at (2.0, 2.63)
NEAR = [[1.83, 1.9], [2.17, 1.9], [2.17, 2.46], [1.83, 2.46]]  # at (2.0, 2.18)


def person(key, x, y, hull):
    return {**det(key, "person", "", x, y, hull=hull), "attributes": {}}


def ask(store, template, slots, k):
    answer_type = "number" if template == "closest_person" else "offset"
    return table_answers.answer(store, question(template, slots, answer_type=answer_type), {"now": store.steps[k]})


def fed():
    """Task 0: step 0 (camera (2.0, 1.63)) sees a person at A, step 1 (camera (2.0, 1.13)) at B, and a bag with a hull
    and a person without one (not evidence); task 1: step 2 (camera (2.0, 1.63)) sees a person at NEAR."""
    store = Store()
    store.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    store.add_step(step(0, 2.0, 1.5, 90.0, detections=[person("a", 2.0, 3.63, A)]))
    store.add_step(step(1, 2.0, 1.0, 90.0, detections=[
        person("b", 2.0, 2.63, B), person("c", 5.0, 5.0, None), det("t", "bag", "red", 2.0, 1.8, hull=NEAR)]))
    store.begin_task(1, "2026-03-02T09:00:00", 3600.0, "task 1")
    store.add_step(step(2, 2.0, 1.5, 90.0, detections=[person("d", 2.0, 2.18, NEAR)]))
    store.add_step(step(3, 3.0, 1.0, 90.0, detections=[]))
    return store


def test_closest_person_is_the_least_camera_to_hull_distance_over_the_tasks_steps():
    """Contract (`table_people`): the least floor distance from each step's own camera to the nearest point of the hulls
    of person tracks at the task's steps, at most 10, to 4 decimals. By hand: task 0 has the hull A from camera (2.0,
    1.63), 1.72 m, and B from (2.0, 1.13), 1.22 m; a person detected without a hull and a bag with one count for
    nothing. Task 1 has NEAR from (2.0, 1.63): 0.27 m. A task nobody was seen in is 10. Regression caught: a hull of
    another task, a hull measured from the wrong camera, a thing counted as a person, or no cap."""
    store = fed()
    assert ask(store, "closest_person", {"task": 0}, 3) == {"type": "number", "value": 1.22}
    assert ask(store, "closest_person", {"task": 1}, 3) == {"type": "number", "value": 0.27}
    assert ask(store, "closest_person", {"task": 5}, 3) == {"type": "number", "value": 10.0}


def test_person_mem_is_the_latest_hulls_nearest_point_in_the_ask_steps_camera_frame():
    """Contract: the point of the latest hulled person sighting nearest the camera of the step the ask is about, in that
    camera's frame; `unknown` with none. By hand: the latest hull before step 3 is NEAR (step 2); from the camera (3.0,
    1.13) heading north its nearest point is (2.17, 1.9): 0.77 ahead, 0.83 left. With only step 0's A the point is
    (2.17, 3.35): 2.22 ahead, 0.83 left. Regression caught: the first sighting read, the sighting's camera used for the
    frame, or a bag taken for the person."""
    assert ask(fed(), "person_mem", {}, 3) == offset(0.77, 0.83)
    store = Store()
    store.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    store.add_step(step(0, 2.0, 1.5, 90.0, detections=[person("a", 2.0, 3.63, A), det("t", "bag", "red", 2.0, 1.8)]))
    store.add_step(step(1, 3.0, 1.0, 90.0, detections=[]))
    assert ask(store, "person_mem", {}, 1) == offset(2.22, 0.83)
    empty = Store()
    empty.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    empty.add_step(step(0, 3.0, 1.0, 90.0, detections=[det("t", "bag", "red", 2.0, 1.8, hull=NEAR)]))
    assert ask(empty, "person_mem", {}, 0) == {"type": "unknown"}


def test_the_table_and_the_gold_agree_on_the_kind_and_the_cap():
    """Contract: the table (baseline side, which may not import the truth) keeps its own copies of the catalogs' kind
    word and the 10 m cap; they are the gold's. Regression caught: one side renamed or re-capped, the ceiling gate
    failing on every `closest_person`."""
    assert table_people.PERSON == PERSON_KIND and table_people.CAP_M == questions_people.CAP_M == 10.0


def test_path_log_answers_from_other_scripts_alone_and_abstains_without_them(tmp_path):
    """Contract (`baselines_path_people`): `path_log` reads no picture, so it has nothing of its own to say about where
    people were; with no prior it is `unknown` for both families, and the families are registered for it (the
    completeness test's gap). Regression caught: a family complete for the table systems and unreachable for the
    shortcut that gates it."""
    system = started(PathLog, tmp_path, kinds=("kitchen",), events=EVENTS, prior=[])
    system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
    system.observe([step(0, 3.0, 1.0, 90.0)])
    asks = [question("closest_person", {"task": 0}, "q1", "number"), question("person_mem", {}, "q2", "offset")]
    assert system.ask("end", {"k": 0, "clock": ""}, asks) == {"q1": {"type": "unknown"}, "q2": {"type": "unknown"}}


def people_case(tmp_path, s):
    """`s` as a case with the people questions written by hand from the gold functions, as `ego_kit`'s tests do (a
    Synth script has no instance images to compile from): the `closest_person` of its first task, and the `person_mem`
    candidates (answerable or unknown) of the look points 13 and 27."""
    w = pk.world(s, pk.one_person())
    cands = [c for m in questions_people.closest_person(w) for c in m.answerable]
    for k in (K, 27):
        moment = questions_people.person_mem(w, k)
        cands += [*moment.answerable, *moment.unknown]
    case = s.write_case(tmp_path)
    qs, gs = questions.records(cands)
    ep.write_jsonl(case.questions, qs, sort_keys=True)
    ep.write_jsonl(case.gold, gs)
    return case, gs


@pytest.mark.parametrize("seen, golds", [
    ([2, 3, 4], [("closest_person", "person"), ("person_mem", "answerable"), ("person_mem", "answerable")]),
    ([18], [("closest_person", "person"), ("person_mem", "unknown"), ("person_mem", "answerable")])])
def test_the_oracle_and_memory_table_reproduce_the_people_golds(tmp_path, seen, golds):
    """Contract (the ceiling gate, FD-task-questions review, section 3.6): over a compiled case the oracle and
    `memory_table` score 1 on every `closest_person` and `person_mem` question, because the oracle channel's person
    hulls are the golds' evidence sightings; the rows pass the record validators with the declared classes. Hand
    cases: a person at (2.0, 3.63) seen at keyframes 2, 3 and 4, the one at 3 with the box cut by the frame's top
    (evidence all the same): the nearest approach and the place as last seen at the look points 13 and 27; and one seen
    only at keyframe 18, after the look point 13 (unknown there) and before the recent view of 27 (frames 20 to 27).
    Regression caught: the oracle channel following the floor's all-edge rule (the top-cut sighting's hull missing, so
    the table's distance differs from the gold's), or any drift between the gold's evidence and the channel's."""
    s = pk.scene()
    pk.put_person(s, "A", (2.0, 3.63), seen, bbox=[100, 0, 140, 300] if len(seen) == 1 else None)
    if len(seen) > 1:
        s.see([3], "A", 1500, bbox=[100, 0, 140, 300])
    case, rows_ = people_case(tmp_path, s)
    assert [(g["family"], g["answer_class"]) for g in rows_] == golds
    for cls in (Oracle, MemoryTable):
        run, _ = play(case, cls, events=EVENTS)
        assert run.status == "complete" and run.failures == {}
        assert set(scores(case, run).values()) == {1.0}, cls.__name__
