"""The answerers of `task_things` (`table_task` for the table systems, `baselines_path_task` for `path_log`): a store
fed by hand for the table's own rules, the oracle and `memory_table` through `run_case` on a compiled world for the
gold's, and `path_log`'s map claim by hand. The way is `task_kit`'s: the docking leg along y = 1, within 1 m (band 0.25
m)."""

from __future__ import annotations

import pytest

import task_kit as tk
from answers_kit import compiled_case, scored
from baseline_kit import EVENTS, ROBOT, det, question, rows, started, step
from stbench import mapspec
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.stream import run_case
from stbench.table import Store
from stbench.table_task import task_things
from stbench.task_predicate import TaskThings
from task_kit import BAG, PATH, put, y

APARTMENT = mapspec.load_map("apartment_a")   # the public map the scripted systems are given: it names a bed as fixed
SLOTS = {"phrase": PATH.phrase, "task": 0}


def listed(answer: dict) -> list[list[float]]:
    """The places of a `positions` answer, as written."""
    assert answer["type"] == "positions"
    return [i["xy"] for i in answer["items"]]


def seen(key: str, kind: str, x: float, yc: float, size=BAG, z: float = 0.0) -> dict:
    """A detection of a thing `size` big at (x, yc) standing `z` above the floor: its box centre is half its height
    up."""
    return {**det(key, kind, key, x, yc, z + size[2] / 2), "extent_m": list(size)}


def fed(tasks: list[list[list[dict]]]) -> Store:
    """A store fed task by task (case indices 0, 1, ...): each task's steps are keyframes 6 t to 6 t + 5, cleaning at
    0-2 and docking at 3-5, the docking base at (0, 1), (20, 1) and (40, 1); the detections of step j of a task are
    `tasks[t][j]` (a list of detections)."""
    store, k = Store(), 0
    for t, frames in enumerate(tasks):
        store.begin_task(t, "2026-03-02T08:00:00", 0.0, "brief")
        for j, dets in enumerate(frames):
            x, yb = (0.0, 0.0) if j < 3 else ((0.0, 1.0), (20.0, 1.0), (40.0, 1.0))[j - 3]
            store.add_step(step(k, x, yb, phase="cleaning" if j < 3 else "docking", detections=dets))
            k += 1
    return store


def ask(store: Store, task: int = 0, pred: TaskThings | None = PATH, public_map=APARTMENT) -> dict:
    return task_things(store, {"phrase": PATH.phrase, "task": task},
                       {"task_things": pred, "map": public_map, "robot": ROBOT})


def test_the_table_lists_the_tracks_that_reach_the_predicate_at_their_last_place_in_the_task():
    """Contract (plan §4 A4): from a store fed by hand the table lists, for the task asked, each track with a sighting
    in it that is a movable thing, stands on the floor and whose detected footprint is within the reach (1 m), at its
    last place in that task. Worked out: task 0 sees a bag 0.5 m from the way at (2, 1.7), another 0.9 m off at (5,
    2.1), one 1.1 m off (no), one on a 0.75 m table (no: off the floor), a bed (a fixed kind of the map, 1.0 m off: no),
    and a box seen in task 0 at (8, 1.7) and then at (9, 1.7), and in task 1 at (11, 1.7): task 0 lists (9, 1.7), its
    last place there, and task 1 (11, 1.7), and the bag at (2, 1.7) seen in task 0 only is not task 1's. A task the
    store never held, and `current_view`'s store with no task message, list nothing. Regression caught: the table
    listing a fixed piece, an off-floor thing or one past the reach, a track at its first place or a task's sighting
    from another task."""
    t0 = [[], [seen("a", "bag", 2, y(0.5)), seen("b", "bag", 5, y(0.9)), seen("c", "bag", 5.0, y(1.1)),
              seen("d", "bag", 14, y(0.5), z=0.75), seen("bed", "bed", 17, 2.0, (1.6, 2.0, 0.5)),
              seen("m", "box", 8, 1.7)], [seen("m", "box", 9, 1.7)], [], [], []]
    t1 = [[], [seen("m", "box", 11, 1.7)], [], [], [], []]
    store = fed([t0, t1])
    assert listed(ask(store, 0)) == [[2.0, 1.7], [5.0, 2.1], [9.0, 1.7]]
    assert listed(ask(store, 1)) == [[11.0, 1.7]]
    assert listed(ask(store, 7)) == []
    assert listed(task_things(Store(), SLOTS, {"task_things": PATH, "map": APARTMENT, "robot": ROBOT})) == []


def test_the_table_lists_only_the_predicates_item_kinds():
    """Contract (0.11.0, FD-0.11-design 3.4.3; the vacuum's list of what its brush could catch): a table lists a track
    of a kind the predicate names in `item_kinds` and no other, and never one of `dont_care_kinds` (the gold makes it a
    don't-care place, so a table that left it out is not charged), exactly as the gold's items; with no `item_kinds`
    every movable kind is listed. By hand, three things 0.5 m from the way seen in task 0: a cord at (2, 1.7), a bag at
    (5, 1.7) and a bath mat at (8, 1.7); the predicate names the cord and the mat as item kinds and the mat as
    don't-care, so only the cord is listed; the open predicate lists all three. Regression caught: a table that lists
    kinds the question did not ask for (a false place for every shoe it saw) or the recurring kinds (a layout prior
    answered from the picture)."""
    t0 = [[], [seen("c", "extension_cord", 2, y(0.5)), seen("b", "bag", 5, y(0.5)), seen("m", "bath_mat", 8, y(0.5))],
          [], [], [], []]
    store = fed([t0])
    named = TaskThings(PATH.phrase, "path", PATH.phases, reach_m=1.0, band_m=0.25, floor=True,
                       item_kinds=frozenset({"extension_cord", "bath_mat"}), dont_care_kinds=frozenset({"bath_mat"}))
    assert listed(ask(store, 0, named)) == [[2.0, 1.7]]
    assert listed(ask(store, 0)) == [[2.0, 1.7], [5.0, 1.7], [8.0, 1.7]]


def test_the_table_refuses_what_it_cannot_answer_with():
    """Contract: a system given no predicate or no public map raises rather than answering the empty list silently (an
    empty answer would pass for a task with nothing in it). Regression caught: a table built without the job's predicate
    that answers every task `[]`, which would score on the empty golds."""
    store = fed([[[], [], [], [], [], []]])
    with pytest.raises(ValueError, match="predicate"):
        ask(store, pred=None)
    with pytest.raises(ValueError, match="map"):
        ask(store, public_map=None)


@pytest.fixture(scope="module")
def world_case(tmp_path_factory):
    """A world with a thing of every kind the gold treats differently, 0.5 m from the way (3 m apart), as the case "c0":
    A a bag (the one item, at (2, 1.7)), B one 1.0 m off (the band), a chair, a person, a twin pair (the other far off),
    a shelf-top bag (0.75 m up), a bed (a fixed kind 1.0 m off), LATE a bag a row of the last look window shows, MOVED a
    bag that moves and is then only glimpsed, and GLIMPSE one never detected. `(case, question rows, gold rows)`."""
    s = tk.script()
    yy = y(0.5)
    put(s, "A", (2, yy))
    put(s, "B", (5, y(1.0)))
    put(s, "CHAIR", (8, yy), (0.4, 0.4, 0.9), kind="chair", mobility="light")
    put(s, "WALKER", (11, yy), (0.5, 0.4, 1.7), kind="person", mobility="mover", attrs={})
    put(s, "T1", (14, yy), attrs={"color": "red"})
    put(s, "T2", (14, 5.0), attrs={"color": "red"})
    put(s, "SHELF", (17, yy), z=0.75)
    put(s, "BED", (20, 3.0), (1.6, 2.0, 0.5), kind="bed", mobility="fixed")
    put(s, "LATE", (23, yy), at=(3, 9))
    put(s, "MOVED", (0, 0), stands=((0, 7.0, (26.0, yy, 0.0)), (7.0, None, (27.0, yy, 0.0))), at=(3,))
    s.see(6, "MOVED", 100)
    put(s, "GLIMPSE", (30, yy), px=100)
    case = compiled_case(tmp_path_factory.mktemp("task"), "c0", s, tk.job(), 0, 0)
    return (case, *rows(case))


def played(system_class, case, **opts):
    """`run_case` of `case` on `system_class` over the `detections` channel, given the public map and the predicate."""
    opts = {"map": APARTMENT, "task_things": PATH, **opts}
    run = run_case(lambda state, work: system_class(state, work, events=EVENTS, **opts), case, channel="detections")
    assert run.status == "complete" and run.failures == {}, system_class.__name__
    return run


def test_the_oracle_and_memory_table_reproduce_the_gold(world_case):
    """Contract (plan §4 A4, the ceiling): through `run_case` on `detections`, the oracle and `memory_table` score 1 on
    the question: the gold lists A alone; they list A and, as the scorer drops them, every place the gold makes
    don't-care that a detection reaches: B in the band, the chair, the person, the twin, the last-look window bag and
    the moved bag at its recognised place; never the shelf bag (off the floor), the bed (a fixed kind), the twin's
    far partner, or the thing only glimpsed. Regression caught: the table's rule differing from the gold's beyond the
    don't-care places (a false positive is a miss, the oracle under the ceiling)."""
    case, qs, golds = world_case
    assert [q["template"] for q in qs] == ["task_things"] and golds[0]["gold"]["items"] == [{"xy": [2.0, 1.7]}]
    held = [[2.0, 1.7], [5.0, 2.2], [8.0, 1.7], [11.0, 1.7], [14.0, 1.7], [23.0, 1.7], [26.0, 1.7]]
    for cls in (Oracle, MemoryTable):
        run = played(cls, case)
        assert listed(run.answers[qs[0]["qid"]]) == held, cls.__name__
        assert scored(golds, run.answers) == {qs[0]["qid"]: 1.0}, cls.__name__


def test_current_view_holds_no_task_and_lists_nothing(world_case):
    """Contract (plan §4 A4, `memory_needed`): a memoryless responder, whose store is one look window and holds no task
    message, answers the empty list, which scores under 1 on a gold with an item. Regression caught: a current view that
    reconstructs the task's anchor from a look window, which would make the memory gate vacuous."""
    case, qs, golds = world_case
    run = played(CurrentView, case)
    assert listed(run.answers[qs[0]["qid"]]) == []
    assert scored(golds, run.answers)[qs[0]["qid"]] < 1.0


def log_with(tmp_path, public_map, pred=PATH) -> PathLog:
    """A `path_log` fed one task whose docking leg runs along y = 5 (the same leg at 0, 20 and 40 m)."""
    pl = started(PathLog, tmp_path, map=public_map, events=dict(EVENTS), task_things=pred)
    pl.task(0, "2026-03-02T08:00:00", 0.0, "brief")
    pl.observe([step(0, 0.0, 0.0, phase="cleaning"), step(1, 0.0, 5.0, phase="docking"),
                step(2, 20.0, 5.0, phase="docking"), step(3, 40.0, 5.0, phase="docking")])
    return pl


def test_path_log_claims_the_floor_surfaces_of_the_map_within_the_reach(tmp_path):
    """Contract (plan §4 A4): `path_log` rebuilds the anchor from its own logged steps and claims the centres of the
    public map's floor-level surfaces within the reach. On the hand map's rack-and-mat variant a floor mat's surface (1
    x 1 m, centred (2, 5), 0.01 m up) lies on the way y = 5, so the claim is [[2, 5]]; the rack has no surface; a map
    with no mat in reach, no map and no predicate each make no claim. With no prior the answer is `unknown`, whatever
    the claim (the held-out rule keeps it only against a pool). Regression caught: a claim from a surface up on a table,
    one past the reach, or an answer from the claim alone with nothing to say it beats abstaining."""
    from stbench.baselines_path_task import map_claim

    s = tk.script()
    pl = log_with(tmp_path, tk.room_map(s, stock=True))
    assert map_claim(pl, SLOTS) == {"type": "positions", "items": [{"xy": [2.0, 5.0]}]}
    q = question("task_things", SLOTS, answer_type="positions")
    assert pl.answerers["task_things"](q, pl.ask_record(q, {"k": 3, "clock": ""}, when="end")) == {"type": "unknown"}
    far = log_with(tmp_path / "far", tk.room_map(s))   # no mat on this map
    assert map_claim(far, SLOTS) is None
    assert map_claim(log_with(tmp_path / "nomap", None), SLOTS) is None
    assert map_claim(log_with(tmp_path / "nopred", tk.room_map(s, stock=True), pred=None), SLOTS) is None
