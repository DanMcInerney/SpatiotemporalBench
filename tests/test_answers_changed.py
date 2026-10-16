"""`changed` answered by the table systems and `path_log` (`table_changed`, `baselines_path_changed`): the oracle over
`changed_kit`'s living room, and the table's reading of tracks by hand. Expected values are worked out in the
docstrings."""

from __future__ import annotations

from unittest import mock

import pytest

import changed_kit as ck
import family_kit as fk
from answers_kit import compiled_case, scored
from baseline_kit import ROBOT, det, gold, question, record, rows, started, step
from stbench import questions_world, table_answers
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.stream import run_case
from stbench.table import Store

CID = "94c06e32acdb2064"  # a draw that asks `changed` beside `object_where` in `changed_kit.by_hand`'s window
NONE = {"type": "positions", "items": []}
LIVING = {"room": "living room", "tasks": [0, 2]}


def positions(*places: tuple[float, float]) -> dict:
    return {"type": "positions", "items": [{"xy": list(p)} for p in places]}


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """`changed_kit.by_hand`'s window of tasks 0 to 2 as a case (id `CID`, rendered views), compiled over the hand map
    (the bundled `apartment_a` the script names has rooms elsewhere): `(case, map)`."""
    s = ck.by_hand()
    hand_map = fk.hand_map(s)
    with mock.patch.object(questions_world, "load_map", lambda name: hand_map):
        return compiled_case(tmp_path_factory.mktemp("changed"), CID, s, ck.JOB, 0, 2), hand_map


def test_the_oracle_answers_changed(case):
    """Contract (E3e §3): through `run_case` on `detections`, over the case compiled from `changed_kit.by_hand` (the
    hand gold of tasks 0 and 2: events (4.6, 1.2), (4.6, 2.0) and (6.2, 2.4), eight don't-care places; a resident, a
    mug on a table, a toy added beside the resident, identical black shoes one of which moved), both `oracle` and
    `memory_table` score 1 on every question, `changed` among them, answering exactly those three events (the shoes are
    twins to both, the resident a mover, the table's mug uncounted). Regression caught: the table's events differing
    from the gold's: the floor rule, the twin rule, the mover rule, the room test or the place test read differently."""
    case, hand_map = case
    qs, golds = rows(case)
    [ask] = [g for g in golds if g["family"] == "changed"]
    assert ask["gold"]["items"] == positions((4.6, 1.2), (4.6, 2.0), (6.2, 2.4))["items"]
    for cls in (Oracle, MemoryTable):
        run = run_case(lambda state, work, cls=cls: cls(state, work, events=ck.JOB.events, map=hand_map), case,
                       channel="detections")
        assert run.status == "complete" and run.failures == {}, cls.__name__
        assert scored(golds, run.answers) == dict.fromkeys((q["qid"] for q in qs), 1.0), cls.__name__
        assert run.answers[ask["qid"]] == positions((4.6, 1.2), (4.6, 2.0), (6.2, 2.4)), cls.__name__


# -- a table by hand ------------------------------------------------------------------------------------------------
TASKS = 3
BAG, BOX, CRATE, SHOES = (0.4, 0.3, 0.3), (0.3, 0.3, 0.3), (0.5, 0.4, 0.3), (0.3, 0.2, 0.12)  # radii 0.15 to 0.2
PERSON, MUG = (0.5, 0.5, 1.7), (0.08, 0.08, 0.1)


def free_block(xy: tuple[float, float], half: float = 0.4) -> dict[tuple[int, int], str]:
    """The fine cells within `half` of the place, all free: what a step that saw the floor there holds."""
    n = round(2 * half / 0.1)
    u0, v0 = round((xy[0] - half) / 0.1), round((xy[1] - half) / 0.1)
    return {(u, v): "free" for u in range(u0, u0 + n) for v in range(v0, v0 + n)}


def detection(key: str, kind: str, color: str, spot: tuple[float, ...], size: tuple[float, float, float]) -> dict:
    """A detection of a box of `size` whose base stands at (x, y[, z]): its centre half the height above the base."""
    base = spot[2] if len(spot) > 2 else 0.0
    return {"d": key, "kind": kind, "attributes": {"color": color}, "bbox_px": [0, 0, 40, 40],
            "center_cam": [0.0, 0.0, 1.0], "center_map": [spot[0], spot[1], base + size[2] / 2],
            "extent_m": list(size)}


def thing(ident: str, kind: str, color: str, size, *stands) -> tuple:
    """One thing for `hand_store`: where it stands in each task, (x, y) on the floor, (x, y, z) up on a table, None
    where no frame detects it, or a list of those, a detection to each of the task's two steps."""
    return ident, kind, color, size, stands


def hand_store(things: list[tuple], free: dict[int, list[tuple[float, float]]] | None = None) -> Store:
    """A store of three tasks of two steps each (k = 2t and 2t + 1 in task t, the robot at (6, 1.5) on the floor), each
    thing detected with its perfect identity at the steps its stands name; the first step of task t holds the fine
    cells `free_block` makes of each place in `free[t]`."""
    store = Store()
    for t in range(TASKS):
        store.begin_task(t, f"2026-03-02T0{8 + t}:00:00", 0.0, "")
        for k in (2 * t, 2 * t + 1):
            dets, ident = [], {}
            for eid, kind, color, size, stands in things:
                spots = stands[t] if isinstance(stands[t], list) else [stands[t]]
                if k - 2 * t < len(spots) and spots[k - 2 * t] is not None:
                    dets.append(detection(f"{eid}-{k}", kind, color, spots[k - 2 * t], size))
                    ident[f"{eid}-{k}"] = eid
            store.add_step({**step(k, 6.0, 1.5), "detections": dets, "identity": ident})
        store.cells[2 * t] = {c: s for xy in (free or {}).get(t, []) for c, s in free_block(xy).items()}
    return store


def ask(store: Store, tasks=(0, 2), room: str = "living room", hand_map="default") -> dict:
    """The table's `changed` answer for the room and the two tasks, through the registry, as the vacuum."""
    hand_map = fk.hand_map(ck.script(TASKS)) if hand_map == "default" else hand_map
    q = {"template": "changed", "slots": {"room": room, "tasks": list(tasks)}}
    return table_answers.answer(store, q, {"robot": ROBOT, "map": hand_map, "events": None})


def test_the_table_by_hand():
    """Contract (E3e §1, the review's m3): the table's events from tracks, tasks 0 and 2 of a 3-task log, in the living
    room (x 4 to 8, y 0 to 3), by the gold's rules over what the table holds. Each thing is detected at one step of a
    task, where it stands:
    - a red bag E1 at (7.4, 0.8) in tasks 0 and 1, at (4.6, 1.2) in task 2: moved, an event at (4.6, 1.2);
    - a blue box E2 at (6.2, 2.4) in tasks 0 and 1, not detected in task 2, whose place task 2's cells hold free:
      removed, an event at (6.2, 2.4); a crate E6 gone from (5.2, 0.4) in the same way, whose place task 2 never
      mapped: nothing (an unseen place says nothing, truth rule 2);
    - a green bag E3 on a table (base 0.75 m up) at (4.6, 2.0) in tasks 0 and 1, on the floor at (4.6, 2.0) in task 2,
      whose floor task 0's cells hold free: the table sightings do not count, so it is added, an event at (4.6, 2.0)
      (counted, it would not have moved);
    - a red mug E5 on a table moved 0.6 m along it: nothing (counted, a move to (7.5, 2.6));
    - a resident P1 at (5.4, 1.6) in task 0, at (7.4, 2.4) and then (6.2, 0.9) in task 2: 1.2 m apart in one task, so a
      mover, whose places are don't-care and no change at (6.2, 0.9) (a thing, it would be a move there); a second
      resident P2 seen once in task 0 at (7.8, 2.8) and once in task 2 at (4.2, 0.2), who walked (7.0, 2.8) to
      (7.6, 2.0) in task 1: a mover from that task, so no change at (4.2, 0.2) either.
    So the answer is (4.6, 1.2), (4.6, 2.0), (6.2, 2.4); the same things asked of the kitchen change nothing there,
    and a table given no map says so. Regression caught: a mover reported as a change, the floor rule read differently
    from the gold's (a thing on a table counted), or a removal where the place was never seen empty."""
    store = hand_store([
        thing("E1", "bag", "red", BAG, (7.4, 0.8), (7.4, 0.8), (4.6, 1.2)),
        thing("E2", "box", "blue", BOX, (6.2, 2.4), (6.2, 2.4), None),
        thing("E3", "bag", "green", BAG, (4.6, 2.0, 0.75), (4.6, 2.0, 0.75), (4.6, 2.0)),
        thing("E5", "mug", "red", MUG, (6.9, 2.6, 0.75), (6.9, 2.6, 0.75), (7.5, 2.6, 0.75)),
        thing("E6", "crate", "brown", CRATE, (5.2, 0.4), (5.2, 0.4), None),
        thing("P1", "person", "none", PERSON, (5.4, 1.6), None, [(7.4, 2.4), (6.2, 0.9)]),
        thing("P2", "person", "grey", PERSON, (7.8, 2.8), [(7.0, 2.8), (7.6, 2.0)], (4.2, 0.2)),
    ], free={0: [(4.6, 2.0)], 2: [(6.2, 2.4)]})
    assert ask(store) == positions((4.6, 1.2), (4.6, 2.0), (6.2, 2.4))
    assert ask(store, room="kitchen") == NONE
    with pytest.raises(ValueError, match="public map"):
        ask(store, hand_map=None)


def test_identical_tracks_are_dont_care():
    """Contract (E3e §1, the review's m4): a track whose description another track of the store shares gives only
    don't-care places, wherever the other was detected: two tracks of black shoes, one at (5.8, 0.4) throughout, the
    other at (6.6, 0.4) until it is at (7.8, 0.4) in task 2 (moved 1.2 m): no event (the table cannot say which
    shoe moved), and none when the first shoe was detected in task 1 alone; with the first never detected, the one
    track is no twin and its move stands: an event at (7.8, 0.4). Regression caught: an event the table cannot tell
    from a twin, or a twin no frame showed changing the answer (the gold reads the second case the same way)."""
    moved = thing("E8", "shoes", "black", SHOES, (6.6, 0.4), (6.6, 0.4), (7.8, 0.4))
    for twin in (thing("E7", "shoes", "black", SHOES, (5.8, 0.4), (5.8, 0.4), (5.8, 0.4)),
                 thing("E7", "shoes", "black", SHOES, None, (5.8, 0.4), None)):
        assert ask(hand_store([twin, moved])) == NONE
    assert ask(hand_store([moved])) == positions((7.8, 0.4))


def test_a_pair_without_two_passes_lists_nothing():
    """Contract (E3e §1, E3a's `other_case`): a pair that names one task twice (`other_case` maps a task past its
    partner's last onto it) has no second pass to differ from, and a task the store never held has no cells to
    certify a removal: the answer is the empty list, not a raise, where tasks 0 and 2 of the same store give the
    moved bag's event. Regression caught: `other_case` failing a whole check on a partner with fewer tasks, or an
    event from a place nobody mapped."""
    store = hand_store([thing("E1", "bag", "red", BAG, (7.4, 0.8), (7.4, 0.8), (4.6, 1.2))])
    assert ask(store) == positions((4.6, 1.2))
    assert [ask(store, tasks=pair) for pair in ((2, 2), (0, 0), (0, 7), (7, 2))] == [NONE] * 4


def test_current_view_lists_nothing(tmp_path):
    """Contract (E3e §1): `current_view`, which keeps one look window and no task message, answers `changed` with the
    empty list, however the stream moved things: three tasks of two steps, a red bag at (7.4, 0.8) until task 2 and at
    (4.6, 1.2) there, are an event at (4.6, 1.2) to `memory_table` and nothing to `current_view`. Regression caught:
    a memoryless system given the gold's answer, or one that raises on a store with no tasks."""
    systems = {cls: started(cls, tmp_path / cls.__name__, events=ck.JOB.events, map=fk.hand_map(ck.script(TASKS)))
               for cls in (MemoryTable, CurrentView)}
    for t in range(TASKS):
        for system in systems.values():
            system.task(t, f"2026-03-02T0{8 + t}:00:00", 0.0, "")
            xy = (4.6, 1.2) if t == 2 else (7.4, 0.8)
            system.observe([step(2 * t + i, 6.0, 1.5, detections=[det("a", "bag", "red", *xy)]) for i in (0, 1)])
    q = question("changed", LIVING, answer_type="positions")
    got = {cls: system.ask("end", {"k": 5, "clock": ""}, [q])["q1"] for cls, system in systems.items()}
    assert got == {MemoryTable: positions((4.6, 1.2)), CurrentView: NONE}


def changed_record(room: str, tasks: list[int], value: dict, script: str) -> dict:
    """A prior record of another script's `changed` question about `room`, its brief naming the room, with the gold
    `value` (`positions`, credit `f1`)."""
    q = question("changed", {"room": room, "tasks": tasks}, answer_type="positions")
    row = {**gold(value), "family": "changed", "credit": "f1", "metric": None, "tolerances": {"match_m": 0.5}}
    return record(q, value, script=script, rooms=(room,)) | {"gold": row}


def test_path_log_answers_from_its_pool(tmp_path):
    """Contract (E3e §2): `path_log` answers `changed` with the best response over the golds of other scripts' records
    at the first `path_log` level that holds enough of them (the room and the two tasks, then the room alone, the rooms
    its briefs named beside each), `unknown` with no record. Hand case, a brief naming the living room: ten scripts'
    records of the living room for tasks 0 and 2, nine of them empty and one a change at (4.6, 1.2), are answered
    with the empty list (a held-out mean of 0.8, against 0 for abstaining and -0.8 for the place, and a resample of ten
    scripts holds the odd one five times or more, a gain of 0 or under, in 0.16% of the draws); the question about
    tasks 1 and 3, which none of them names, falls to the room alone, where twenty more records for tasks 0 and 3
    each name that place (thirty records, 21 for the place and 9 empty: the place is +1 on 21 and -1 on 9, the empty
    list the reverse, and the place's gain is sure, its interval over thirty scripts above 0): the place; with no
    record at all, `unknown`. Regression caught: a task pair that splits the pool until it is empty, an answer from a
    pool that is no better than abstaining, or a prior never read."""
    one = positions((4.6, 1.2))
    prior = [changed_record("living room", [0, 2], v, f"s{i}") for i, v in enumerate([NONE] * 9 + [one])]
    prior += [changed_record("living room", [0, 3], one, f"t{i}") for i in range(20)]
    got = []
    for records in (prior, []):
        system = started(PathLog, tmp_path / str(len(records)), map=fk.hand_map(ck.script(TASKS)),
                         events=ck.JOB.events, prior=records)
        system.task(0, "2026-03-02T08:00:00", 0.0, "Its current assignment is to clean the living room.")
        system.observe([step(0, 6.0, 1.5)])
        asked = [question("changed", {"room": "living room", "tasks": tasks}, answer_type="positions")
                 for tasks in ([0, 2], [1, 3])]
        got += [system.ask("end", {"k": 0, "clock": ""}, [q])["q1"] for q in asked]
    assert [got[0], got[2], got[3]] == [NONE, {"type": "unknown"}, {"type": "unknown"}]
    assert got[1]["type"] == "positions" and [i["xy"] for i in got[1]["items"]] == [pytest.approx([4.6, 1.2])]
