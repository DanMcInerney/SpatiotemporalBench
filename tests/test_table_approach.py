"""The table systems' and `path_log`'s answers to `closest_approach` (`table_approach`, `baselines_path_approach`), and
that the oracle reproduces the gold over a compiled case. Hand-worked on a vacuum that drives north: its camera is
mounted 0.13 m ahead of its base, so a base at (3.0, 1.0) heading 90 degrees has its camera at (3.0, 1.13) and one at
(3.0, 0.7) at (3.0, 0.83); `baseline_kit.step`s are one second apart, so the robot moves 0.3 m/s, which the second
step delivers as its `velocity` (0.0, 0.3), the first step being a task's first (0.0, 0.0)."""

from __future__ import annotations

import approach_kit as ak
import person_kit as pk
from baseline_kit import EVENTS, det, play, question, started, step
from ego_kit import K
from stbench import episode as ep
from stbench import questions, questions_approach, table_answers
from stbench.baselines import MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.table import Store

WALKER = ([[3.3, 3.93]], [[3.3, 3.13]])   # a person who is one floor point: at step 0 and at step 1, 0.8 m south
FAR = [[6.0, 5.0]]   # a second person, standing: from (3.0, 1.13) it is hypot(3.0, 3.87) = 4.9 m away


def person(key, hull):
    x, y = (sum(p[i] for p in hull) / len(hull) for i in (0, 1))
    return {**det(key, "person", "", x, y, hull=hull), "attributes": {}}


def fed(identity=True, hulls=WALKER, second=(False, False)):
    """A task of two steps one second apart: the vacuum at (3.0, 0.7) and then (3.0, 1.0), heading north, and a person
    who is a floor point at the first hull and then the second (None: not seen); `second` (True: at both steps) says at
    which step another person stands at `FAR`."""
    store = Store()
    store.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    second = (second, second) if isinstance(second, bool) else second
    for k, y in enumerate((0.7, 1.0)):
        dets = [person("a", hulls[k])] if hulls[k] else []
        dets += [person("b", FAR)] if second[k] else []
        who = {"a": "P1", "b": "P2"}
        store.add_step(step(k, 3.0, y, 90.0, detections=dets, velocity=[0.0, 0.3 * k],
                            **({"identity": who} if identity else {})))
    return store


def ask(store, k=1):
    return table_answers.answer(store, question("closest_approach", {}, answer_type="number"), {"now": store.steps[k]})


def test_the_table_pairs_the_one_person_of_each_frame_and_holds_two_people_still():
    """Contract (`table_approach`, the gold's identity-free rule): the person is the one hull of the ask's step, their
    velocity is `approach_rule.pair_velocity` with the one hull of the latest earlier step within 4 s of the ask, the
    robot's is the ask's step's own delivered `velocity` (here (0.0, 0.3)), and the answer is `approach_rule.closest_m`.
    By hand: the point walks
    0.8 m south in 1 s (0.8 m/s) and the camera floor point moves 0.3 m/s north from (3.0, 0.83) to (3.0, 1.13): 2.0 m
    ahead and 0.3 m east, closing 1.1 m/s: at 1.8 s the gap is 0.02, hypot(0.3, 0.02) = 0.3007, the least of the
    samples (1.9 s: 0.3132, 1.7 s: 0.3269). The pairing reads the hulls the steps hold, not the tracks they were filed
    under: the same answer with perfect identity or none. With another person in either step the pairing is ambiguous
    (the gold asks no such point) and the person is taken as still: 0.5831 (hypot(0.3, 0.5) at 5 s). Regression caught:
    a pairing that follows a track, so that a system's way of filing detections changes the answer; a walker paired
    across two people; or a crash on the ambiguous case."""
    assert ask(fed()) == ask(fed(identity=False)) == {"type": "number", "value": 0.3007}
    for second in (True, (True, False), (False, True)):
        assert ask(fed(second=second)) == ask(fed(identity=False, second=second)) == {"type": "number", "value": 0.5831}


def test_the_cap_is_the_answer_with_no_hull_in_the_ask_frame_and_a_person_without_one_counts_for_nothing():
    """Contract: no person hull at the ask step is "no one", 10; a person detected without a floor hull is not evidence
    and counts for nothing, as for the gold; a hull an earlier step held is not today's person. Regression caught: a
    distance for a person who is not there, or a glimpse taken for evidence."""
    assert ask(fed(hulls=(WALKER[0], None))) == {"type": "number", "value": 10.0}
    assert ask(fed(hulls=(None, None))) == {"type": "number", "value": 10.0}
    glimpse = Store()
    glimpse.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    glimpse.add_step(step(0, 3.0, 1.0, 90.0, detections=[{**person("a", FAR), "floor_hull": None}],
                          velocity=[0.0, 0.0]))
    assert ask(glimpse, 0) == {"type": "number", "value": 10.0}


def test_a_system_with_only_the_latest_frame_holds_the_person_still_and_moves_with_the_delivered_velocity():
    """Contract (final review S3): `current_view`'s store holds the latest frame alone, so there is no earlier hull (the
    person is taken as still) but the step still carries the robot's own velocity (the controller's odometry), which it
    moves with. By hand: the point (3.3, 3.13) from the camera floor point (3.0, 1.13) at 0.3 m/s north: after 5 s the
    point is 0.3 m beside and 0.5 m ahead, hypot(0.3, 0.5) = 0.5831, the least; with the robot delivered as still it is
    hypot(0.3, 2.0) = 2.0224 at every instant. Regression caught: a velocity the step carries but the table ignores
    (the memoryless reader answering as if the robot stood), or a motion invented from nothing."""
    def latest(velocity):
        store = Store()
        store.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
        store.add_step(step(1, 3.0, 1.0, 90.0, detections=[person("a", WALKER[1])], velocity=velocity))
        return store
    assert ask(latest([0.0, 0.3]), 1) == {"type": "number", "value": 0.5831}
    assert ask(latest([0.0, 0.0]), 1) == {"type": "number", "value": 2.0224}


def case_of(path, s):
    """`s` as a case with the `closest_approach` of the look point written by hand from the gold function."""
    cand = questions_approach.candidate(pk.world(s), K)
    case = s.write_case(path)
    qs, gs = questions.records([cand])
    ep.write_jsonl(case.questions, qs, sort_keys=True)
    ep.write_jsonl(case.gold, gs)
    return case, qs[0]["qid"], gs[0]


def test_the_oracle_and_memory_table_reproduce_the_gold_of_a_still_person_and_a_walker(tmp_path):
    """Contract (the ceiling and `knowable` gates): over a compiled case both the oracle, whose tracks have perfect
    identity, and `memory_table` (tracks by description and position, 0.5 m) answer the gold exactly, for a still person
    and for a walker, also when a second person was seen in a frame the pairing does not read: the channel's person
    hulls are the gold's evidence sightings and the table pairs the lone hull of each frame, not a track. By hand: the
    walker goes 0.8 m in 1.5 s (0.5333 m/s), closing 0.8333 m/s with the robot's 0.3: the 2.0 m gap is gone at 2.4 s and
    the 0.3 m beside it is left: 0.3. A second person B is an evidence sighting at keyframe 5 only (`crowd`), or a 230
    px detection with no hull at keyframes 12 and 13 (`glimpsed`), so `memory_table` holds two person tracks that share
    a description when the walker, 0.8 m from its last place, is next seen: it could not re-identify them, and it does
    not need to. Regression caught: the oracle channel drifting from the gold's
    evidence; the table and the gold reading the clock or the poses differently; or the old table that lost a walker
    when a second person had been about."""
    still = ak.scene()
    pk.put_person(still, "A", (3.0, 3.63), [12, K])
    cases = [("still", still, 0.72, "now")]
    walker = ak.scene()
    ak.walker(walker, "A", (3.3, 3.93), (3.3, 3.13))
    cases.append(("walker", walker, 0.3, "memory"))
    crowd = ak.scene()
    ak.walker(crowd, "A", (3.3, 3.93), (3.3, 3.13))
    pk.put_person(crowd, "B", (6.0, 3.0), [5])
    assert questions_approach.person_hull(pk.cut(crowd, K), "B", 5) is not None   # B is evidence, in a frame skipped
    cases.append(("crowd", crowd, 0.3, "memory"))
    glimpsed = ak.scene()
    ak.walker(glimpsed, "A", (3.3, 3.93), (3.3, 3.13))
    pk.put_person(glimpsed, "B", (6.0, 5.0), [12, K], px=230)   # detected from 215 px, no hull, under the 256 px screen
    cases.append(("glimpsed", glimpsed, 0.3, "memory"))
    for name, s, value, scope in cases:
        case, qid, row = case_of(tmp_path / name, s)
        assert row["gold"] == {"type": "number", "value": value} and row["scope"] == scope, name
        for cls in (Oracle, MemoryTable):
            assert play(case, cls, events=EVENTS)[0].answers[qid] == row["gold"], (name, cls.__name__)


def test_path_log_has_no_picture_and_abstains_without_other_scripts(tmp_path):
    """Contract (`baselines_path_approach`): `path_log` reads no picture and a person's walk is no fact of the robot's
    path, so with no prior it answers `unknown`, and the family is registered for it. Regression caught: a family
    complete for the table systems and unreachable for the shortcut that gates it."""
    system = started(PathLog, tmp_path, kinds=("kitchen",), events=EVENTS, prior=[])
    system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
    system.observe([step(0, 3.0, 1.0, 90.0)])
    asks = [question("closest_approach", {}, "q1", "number")]
    assert system.ask("during", {"k": 0, "clock": ""}, asks) == {"q1": {"type": "unknown"}}
