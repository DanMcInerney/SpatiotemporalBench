"""The table systems' and `path_log`'s answers to `approach_mem` (`table_approach`, `baselines_path_approach`), and that
the oracle and `memory_table` reproduce the gold over a compiled case while `current_view` cannot. Hand-worked on a
vacuum that drives north at 0.3 m/s: a step is one second after the one before, the base is at (3.0, 0.7 + 0.3 k) and
its camera 0.13 m ahead of it, so at step 2 the camera floor point is (3.0, 1.43) and the step delivers the velocity
(0.0, 0.3). A person is one floor point, so the arithmetic is the point of each test."""

from __future__ import annotations

import approach_kit as ak
import person_kit as pk
from baseline_kit import EVENTS, det, play, question, started, step
from ego_kit import K
from stbench import table_answers
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.table import Store

NEAR = [[3.3, 3.13]]   # a person 1.70 m ahead of the camera at step 2 and 0.3 m east of it
NEARER = [[3.3, 3.43]]   # the same person 0.3 m farther north
FAR = [[6.0, 5.0]]   # a second person, 3.0 m east and 3.57 m ahead of the camera at step 2


def person(key, hull):
    x, y = (sum(p[i] for p in hull) / len(hull) for i in (0, 1))
    return {**det(key, "person", "", x, y, hull=hull), "attributes": {}}


def fed(seen, *, steps=3, second_task_at=None, identity=True) -> Store:
    """Steps 0 to `steps - 1` of a vacuum driving north, the people `seen` (step -> [(key, floor point)]) detected, a
    second task beginning at step `second_task_at`; `identity` says whether the channel names each key's person."""
    store = Store()
    store.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    for k in range(steps):
        if k == second_task_at:
            store.begin_task(1, f"2026-03-02T08:00:{k:02d}", 0.0, "task 1")
        dets = [person(key, hull) for key, hull in seen.get(k, [])]
        who = {key: f"P-{key}" for key, _ in seen.get(k, [])}
        store.add_step(step(k, 3.0, 0.7 + 0.3 * k, 90.0, detections=dets, velocity=[0.0, 0.3 if k else 0.0],
                            **({"identity": who} if identity else {})))
    return store


def ask(store, k=2):
    return table_answers.answer(store, question("approach_mem", {}, answer_type="number"), {"now": store.steps[k]})


def test_the_table_takes_the_person_last_seen_to_stand_where_they_were_seen():
    """Contract (`table_approach.approach_mem`, FD-0.11-hard 3.2): the answer is `closest_m` from the ask's camera floor
    point, moved at the ask's step's own delivered velocity, to the hull of the latest earlier step of the task that
    holds one, the person held still. By hand: the camera floor point is (3.0, 1.43 + 0.3 tau) at step 2, so after 5 s
    it is at (3.0, 2.93). The person seen at step 0 at (3.3, 3.13) is then 0.2 m ahead and 0.3 m east: hypot(0.3, 0.2) =
    0.3606, the least (the gap closes the whole 5 s). Seen at step 1 at (3.3, 3.43) as well, the latest sighting is the
    step 1's: 0.5 m ahead, hypot(0.3, 0.5) = 0.5831; the earlier sighting does not count and the person is not carried
    on at the 0.3 m/s the two sightings would give. A person B at (6.0, 5.0) seen at step 1 only, after A at step 0,
    is the person last seen: from (3.0, 2.93), hypot(3.0, 2.07) = 3.6448, not A's 0.3606 (the nearer is not the last).
    Regression caught: the nearest sighting of the task instead of the latest, a person carried along a velocity the
    sensors do not give, or the robot's own velocity ignored (hypot(0.3, 1.7) = 1.7263 at rest)."""
    assert ask(fed({0: [("a", NEAR)]})) == {"type": "number", "value": 0.3606}
    assert ask(fed({0: [("a", NEAR)], 1: [("a", NEARER)]})) == {"type": "number", "value": 0.5831}
    assert ask(fed({0: [("a", NEAR)], 1: [("b", FAR)]})) == {"type": "number", "value": 3.6448}


def test_the_cap_is_the_answer_when_the_person_is_in_view_or_none_or_two_were_last_seen():
    """Contract: the table answers 10 when the ask's step holds a hull (the person is in view: `closest_approach`'s
    question), when no step of the task before it does (none seen, or the only sighting is in the task before), and when
    the latest sighting holds two hulls (the sensors do not say which was last seen), whether or not the channel names
    the people: the same answer with perfect identity and with none. Regression caught: a distance for a person the
    reader cannot name, a track's identity deciding the person, or a sighting of an earlier task read."""
    cap = {"type": "number", "value": 10.0}
    assert ask(fed({0: [("a", NEAR)], 2: [("a", NEAR)]})) == cap   # in view at the ask
    assert ask(fed({})) == cap   # no one seen
    assert ask(fed({0: [("a", NEAR)]}, steps=4, second_task_at=1), 3) == cap   # seen in the task before this one
    for identity in (True, False):
        assert ask(fed({1: [("a", NEAR), ("b", FAR)]}, identity=identity)) == cap, identity
        assert ask(fed({0: [("a", NEAR)], 1: [("a", NEAR), ("b", FAR)]}, identity=identity)) == cap, identity


def test_a_system_with_only_the_latest_frame_has_no_one_last_seen_and_answers_the_cap():
    """Contract (the memory-needed gate): `current_view`'s store holds the latest frame alone, so there is no earlier
    step and no last-seen person: it answers 10, the harmful side of every near gold. Regression caught: a latest-frame
    reader that reaches the memory the family exists to need."""
    latest = Store()
    latest.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    latest.add_step(step(2, 3.0, 1.3, 90.0, detections=[person("a", NEAR)], velocity=[0.0, 0.3]))
    assert ask(latest) == {"type": "number", "value": 10.0}


def test_the_oracle_and_memory_table_reproduce_the_gold_and_current_view_cannot(tmp_path):
    """Contract (the ceiling and `memory_needed` gates): over a compiled case both the oracle (perfect identity) and
    `memory_table` (tracks by description and position) answer the gold exactly, and `current_view` answers 10. By hand
    (the kit's world: the person seen at keyframe 10 at (3.5, 2.0) and out of view since, then 0.5 m east): the gold is
    0.33, the camera floor point passing 0.33 m west of the footprint (`test_approach_mem`); a second person B who is an
    evidence sighting at keyframe 5 only (`crowd`), or a 230 px row with no hull at keyframes 12 and 13 (`glimpsed`,
    under the 256 px screen), changes neither the gold nor the table's answer, because the last sighting is the latest
    one and the table reads hulls by step. Regression caught: the oracle channel drifting from the gold's evidence, the
    table and the gold reading the clock, the poses or the last sighting differently (a ceiling under 0.95), or
    `current_view` reaching the gold."""
    moved = [(0, 17.0, (*ak.AT, 0.0)), (17.0, None, (4.0, 2.0, 0.0))]
    worlds = {}
    base = ak.scene()
    pk.put_person(base, "A", moved, [ak.J])
    worlds["base"] = base
    crowd = ak.scene()
    pk.put_person(crowd, "A", moved, [ak.J])
    pk.put_person(crowd, "B", (6.0, 3.0), [5])
    worlds["crowd"] = crowd
    glimpsed = ak.scene()
    pk.put_person(glimpsed, "A", moved, [ak.J])
    pk.put_person(glimpsed, "B", (6.0, 5.0), [12, K], px=230)
    worlds["glimpsed"] = glimpsed
    for name, s in worlds.items():
        case, qid, row = ak.case_of(tmp_path / name, s)
        assert row["gold"] == {"type": "number", "value": 0.33} and row["scope"] == "memory", name
        for cls in (Oracle, MemoryTable):
            assert play(case, cls, events=EVENTS)[0].answers[qid] == row["gold"], (name, cls.__name__)
        assert play(case, CurrentView, events=EVENTS)[0].answers[qid] == {"type": "number", "value": 10.0}, name


def test_path_log_has_no_picture_and_abstains_without_other_scripts(tmp_path):
    """Contract (`baselines_path_approach`): `path_log` reads no picture and where the person last seen stood is no fact
    of the robot's path, so with no prior it answers `unknown`, and the family is registered for it. Regression caught:
    a family complete for the table systems and unreachable for the shortcut that gates it."""
    system = started(PathLog, tmp_path, kinds=("kitchen",), events=EVENTS, prior=[])
    system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
    system.observe([step(0, 3.0, 1.0, 90.0)])
    asks = [question("approach_mem", {}, "q1", "number")]
    assert system.ask("during", {"k": 0, "clock": ""}, asks) == {"q1": {"type": "unknown"}}
