"""`closest_person` (end, `memory`): how near a person came to the robot in a task, as the frames show them. Hand-worked
on `person_kit`'s scene: at keyframe 2 the camera is at (2.0, 1.63) heading north, at keyframe 13 at (2.0, 1.13) (the
base moved to (2.0, 1.0)), at keyframe 27 it is elsewhere; a person is a 0.34 x 0.56 m footprint, so its near edge is
0.28 m before its centre when it faces north. The job's clearance is 2.0 m (rungs 1.0, 2.0, 4.0; C/2 = 1.0 m)."""

from __future__ import annotations

import pytest

import person_kit as pk
from ego_kit import CLEARANCE_M, K, put
from stbench import questions_people
from stbench.questions_people import closest_person
from stbench.seeding import stable_hash


def scene(n_tasks=1):
    return pk.scene(n_tasks, poses={K: (2.0, 1.0)})


def only(w):
    """The one candidate the case asks, with the moment's key."""
    (moment,) = closest_person(w)
    (cand,) = moment.answerable
    assert moment.unknown == []
    return moment.key, cand


def case_with(coin_kept: bool, first: str | None = None, tasks=("0", "1")) -> str:
    """A case id whose declared 'no one' coin (`stable_hash(case, "closest_person", "empty") % 100 < 50`) is kept or
    not and, with `first`, whose hash order of the tasks puts `first` first."""
    for n in range(500):
        cid = f"c{n}"
        order = min(tasks, key=lambda t: stable_hash(cid, "closest_person", t))
        if (stable_hash(cid, "closest_person", "empty") % 100 < 50) is coin_kept and first in (None, order):
            return cid
    raise AssertionError("no case id found")


def test_the_gold_is_the_least_distance_over_the_evidence_sightings_each_from_its_own_camera():
    """Contract (FD-task-questions review, section 3.3): the gold is the least, over the task's keyframes and its
    evidence sightings, of the floor distance from that keyframe's camera floor point to the nearest point of the
    sighting's hull. By hand: the person stands at (2.0, 3.63) until 10 s and at (2.0, 2.63) after. Seen at keyframe 2
    (3.0 s, camera (2.0, 1.63)) its near edge is y = 3.35, 1.72 m; seen at keyframe 13 (19.5 s, camera (2.0, 1.13)) its
    near edge is y = 2.35, 1.22 m. The gold is 1.22, the class `person`, the evidence the two keyframes, the subject the
    task's keyframes, the need C (2.0) with the harmful side `over`, the threshold the body's 0.34 m plus C and the
    resolution of keyframe 13's camera, 0.0491 (the resolution tests work it out). Regression caught: the gold measured
    from the end pose or from one fixed camera (the base's camera at keyframe 27 would give 0.76 m, the first sighting
    1.72 m), a sighting measured from another frame's camera, or a need that is not C."""
    s = scene()
    pk.put_person(s, "A", [(0, 10.0, (2.0, 3.63, 0.0)), (10.0, None, (2.0, 2.63, 0.0))], [2, K])
    put(s, "T", "bag", "red", (2.0, 1.8), [K])   # a thing is no person, however near
    key, cand = only(pk.world(s))
    assert key == "0" and cand.family == "closest_person" and cand.slots == {"task": 0}
    assert cand.gold == {"type": "number", "value": 1.22} and cand.answer_class == "person"
    assert cand.evidence_k == (2, K) and cand.subjects == ("w:0-27",) and cand.entities == ()
    assert cand.tolerances == {"resolution_m": pytest.approx(0.0491, abs=1e-4), "need_m": 2.0, "harm": "over",
                               "threshold_m": pytest.approx(2.34)}
    assert cand.slice == "answerable" and cand.critical


def test_any_person_counts_and_a_sighting_that_is_not_evidence_adds_nothing():
    """Contract: no identity is needed; the nearest person of any, in any evidence sighting, sets the gold, and a
    sighting that is not evidence (here a box on the bottom edge) is not read. By hand at keyframe 13 (camera (2.0,
    1.13)): A at (3.0, 3.13) has its near corner (2.83, 2.85), 1.9098 m; B at (2.0, 2.63) has (2.0, 2.35), 1.22 m; C
    at (2.0, 1.9), with its box on the bottom edge, is 0.49 m but no evidence, and lies 0.73 m nearer than the gold,
    under C/2 = 1.0 m, so the task stays. Regression caught: the first person's gold, the identity of the nearest
    asked for, or a cut sighting measured as if the feet were seen."""
    s = scene()
    pk.put_person(s, "A", (3.0, 3.13), [K])
    pk.put_person(s, "B", (2.0, 2.63), [K])
    pk.put_person(s, "C", (2.0, 1.9), [K], bbox=[100, 100, 140, 480])
    assert only(pk.world(s))[1].gold == {"type": "number", "value": 1.22}


def test_a_nearer_person_the_rule_does_not_count_drops_the_task():
    """Contract (the truth's footprint screen): when a row of a person that is not an evidence sighting lies, by the
    truth's footprint from that frame's camera floor point, nearer than the gold by more than C/2 = 1.0 m, the task is
    not asked: the frames showed someone nearer than the gold would say. By hand: A seen at keyframe 2 gives 1.72 m. A
    second person seen at 500 px at keyframe 13 at (2.0, 2.0) has its footprint y 1.72-2.28, 0.59 m from the camera
    (2.0, 1.13): 1.13 m nearer, so the task is dropped; at (2.0, 2.5) it is 1.09 m away, 0.63 m nearer, and the task
    stays. Regression caught: a gold farther than what the robot saw, or a screen that drops on any nearer row."""
    def case(y):
        s = scene()
        pk.put_person(s, "A", (2.0, 3.63), [2])
        pk.put_person(s, "B", (2.0, y), [K], px=500)
        return closest_person(pk.world(s))
    assert case(2.0) == []
    (moment,) = case(2.5)
    assert moment.answerable[0].gold == {"type": "number", "value": 1.72}


def test_no_one_is_asked_only_where_no_row_of_256_px_lies_in_the_cap_and_by_the_coin():
    """Contract: with no evidence sighting nearer than 10 m the gold is 10 ("no one"), class `no_one`, asked only when
    no person row of 256 px or more lies within 10 m by the truth's footprint, and then kept by the declared coin
    (`stable_hash(case, "closest_person", "empty") % 100 < 50`). By hand from the camera (2.0, 1.63) at keyframe 2: a
    person at (2.0, 8.63) is 6.72 m away, at (2.0, 13.0) 11.09 m. A world with nobody asks on a kept coin and not on
    the other; a 300 px person at 6.72 m forbids it; 255 px at 6.72 m or 300 px at 11.09 m does not. The evidence is the
    task's last keyframe. Regression caught: a 'no one' while a person is plainly in range (the answer 10 asked of a
    robot that saw someone), the coin read the other way, or an answer 10 on every empty task."""
    kept, dropped = case_with(True, tasks=("0",)), case_with(False, tasks=("0",))

    def asked(cid, **person):
        s = scene()
        if person:
            pk.put_person(s, "A", person.pop("at"), [2], **person)
        found = closest_person(pk.world(s, cid=cid))
        return [m.answerable[0] for m in found]

    (cand,) = asked(kept)
    assert cand.gold == {"type": "number", "value": 10.0} and cand.answer_class == "no_one"
    assert cand.evidence_k == (27,) and cand.slots == {"task": 0} and cand.subjects == ("w:0-27",)
    assert asked(dropped) == []
    assert asked(kept, at=(2.0, 8.63), px=300) == []
    assert len(asked(kept, at=(2.0, 8.63), px=255)) == 1
    assert len(asked(kept, at=(2.0, 13.0), px=300)) == 1


def test_one_question_a_case_about_the_task_the_hash_puts_first_that_passes():
    """Contract (the medium end budget): a window of two tasks asks at most one `closest_person`, about the task that
    comes first in `stable_hash(case, "closest_person", task)` order among those that pass. By hand: task 0's person
    is seen at keyframe 2 at (2.0, 4.63), 2.72 m from the camera (2.0, 1.63); task 1's at keyframe 30 at (2.0, 3.63),
    1.72 m. The moment is the first task of the hash order, its gold that task's, whichever that is (two case ids give
    the two orders); when only task 1 holds anyone and the coin drops task 0's 'no one', task 1's is asked though the
    hash puts task 0 first. Regression caught: a question a task (3.5 end questions a case against 2.88), or a task
    chosen by its place in the window and not by the hash."""
    both = scene(2)
    pk.put_person(both, "A", (2.0, 4.63), [2])
    pk.put_person(both, "B", (2.0, 3.63), [30])
    for first, gold in (("0", 2.72), ("1", 1.72)):
        key, cand = only(pk.world(both, a=0, b=1, cid=case_with(True, first)))
        assert (key, cand.slots, cand.gold["value"]) == (first, {"task": int(first)}, gold)
    late = scene(2)
    pk.put_person(late, "B", (2.0, 3.63), [30])
    key, cand = only(pk.world(late, a=0, b=1, cid=case_with(False, "0")))
    assert (key, cand.gold["value"], cand.subjects) == ("1", 1.72, ("w:28-55",))


def test_the_words_are_the_plans():
    """Contract (FD-task-questions-impl, section 3): the text names the task by its ordinal and states the cap, and
    `person_mem`'s names no one. Regression caught: a reworded question the prompt's credit sentence no longer matches
    (the units, the cap and the 'no one' answer are what the agent needs)."""
    assert questions_people.SPECS[0].render({"task": 0}) == (
        "During your first task, how close did a person come to you, as you saw them? Answer the smallest distance in "
        "metres along the floor from the point below your camera to the nearest point of a person; answer 10 if you "
        "saw no one within 10 m.")
    assert questions_people.SPECS[0].render({"task": 2}).startswith("During your third task,")
    assert questions_people.SPECS[1].render({}) == (
        "As you last saw them, where is the point of the person you saw that is nearest to your camera now, measured "
        "along the floor? Answer in metres ahead of the camera and to its left.")
    assert CLEARANCE_M == 2.0   # the rungs above are C's
