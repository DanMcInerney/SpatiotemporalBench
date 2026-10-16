"""The robot's own log tells about a thing only from the moment it does (`World.told(upto=k)`), so a question asked
during the job names what the log had not yet told, and an end question, asked after the window, names nothing it told
(`World.told()`). Hand-built worlds on `space_kit`'s run (28 keyframes a task at 1.5 s, task 1 starting at 7,200 s; the
decision points of a task at keyframes 5 (`before_room`), 18 (`room_done`) and 21 (`before_doorway`), 28 later in task
1) over the hand map's table (x 1.4..2.6, y 2.7..3.3)."""

from __future__ import annotations

from dataclasses import replace

from ego_kit import SMALL, K, named, put, scene
from shape_kit import TABLE
from space_kit import vacuum_run
from stbench.questions_ego import ego_mem, ego_now
from stbench.questions_world import World
from test_world_told import AT_TABLE, TABLE_RULE, forever, job_of, world_on
from test_world_told import put as put_thing


def carrying(w: World, targets: dict[int, tuple[str, ...]]) -> World:
    """The same window over a timeline whose tasks carry these `targets` (by task index)."""
    tasks = [replace(t, targets=targets.get(t.index, t.targets)) for t in w.tl.ego.tasks]
    return World(replace(w.inp, tl=replace(w.inp.tl, ego=replace(w.inp.tl.ego, tasks=tasks))), w.rec)


def test_told_by_a_keyframe_holds_the_tasks_begun_and_the_decisions_so_far():
    """Contract: `told(upto=k)` is the carried thing of each task of the window that has begun (`Task.k0 <= k`: its
    subgoal names the thing from the first step) and the things standing in the zone of each zoned decision at or before
    `k`; `told()` is all of them. Hand-worked on two tasks (decisions at keyframes 5, 21, 33 and 49, both kinds zoned
    to the table; the zone is read at the decision's time, 7.5 s, 31.5 s, 7,207.5 s and 7,231.5 s):
    - E0001 on the table throughout: told from keyframe 5;
    - E0002 on the table from 7,230 s on: told from keyframe 49 alone;
    - E0003 carried by task 1, E0004 carried by task 0, both away from the table: told from keyframes 28 and 0;
    - E0005 on the table only from 100 s to 120 s, at no decision: never told.
    So `told(upto)` is {E0004} at 0 and 4, then with E0001 from 5 to 27, with E0003 from 28 to 48, with E0002 at 49 and
    after, as `told()` is; a window of task 1 alone holds E0003 from 28, E0001 from 33 and never E0004. Regression
    caught: the whole window read at an earlier question (a thing a later blocker or task tells, unnameable at a point
    that could not know it), a task counted before it begins, or a decision counted before it falls."""
    s = vacuum_run(2)
    away = (0.5, 1.5, 0.0)
    put_thing(s, "E0001", "bag", "red", forever(AT_TABLE))
    put_thing(s, "E0002", "bag", "blue", [(0, 7230.0, away), (7230.0, None, AT_TABLE)])
    put_thing(s, "E0003", "toy", "pink", forever(away))
    put_thing(s, "E0004", "toy", "yellow", forever(away))
    put_thing(s, "E0005", "box", "brown", [(0, 100.0, away), (100.0, 120.0, AT_TABLE), (120.0, None, away)])
    targets = {0: ("R1", "R2", "E0004"), 1: ("R1", "R2", "E0003")}
    job = job_of(TABLE_RULE)
    w = carrying(world_on(s, job, TABLE, a=0, b=1), targets)
    expected = {0: {"E0004"}, 4: {"E0004"}, 5: {"E0004", "E0001"}, 27: {"E0004", "E0001"},
                28: {"E0004", "E0001", "E0003"}, 48: {"E0004", "E0001", "E0003"},
                49: {"E0004", "E0001", "E0003", "E0002"}}
    assert {k: w.told(k) for k in expected} == expected
    assert w.told() == w.told(55) == expected[49]
    second = carrying(world_on(s, job, TABLE, a=1, b=1), targets)
    assert [second.told(k) for k in (28, 32, 33, 55)] == [{"E0003"}, {"E0003"}, {"E0003", "E0001"}, {"E0003", "E0001",
                                                                                                   "E0002"}]


def test_a_look_names_what_the_log_has_not_yet_told():
    """Contract: `ego_now` and `ego_mem` at the look point name a thing the log tells about only later and leave out one
    it has already told. Hand-worked at keyframe 13 (t 19.5 s) with the table zoned for `before_room` (keyframe 5,
    7.5 s) and `room_done` (keyframe 18, 27 s), and a second task of the window carrying E0007:
    - X, a red bag on the table throughout, seen at 2 and 13, and X2, a pink toy there too, seen at 2 only: standing in
      the zone at keyframe 5, told, so neither is named;
    - Y, a blue bag on the table only from 26 s to 28 s, seen at 2 and 13, and Y2, a yellow toy there then, seen at 2:
      told only by keyframe 18, so `ego_now` names Y and `ego_mem` names Y2;
    - the unseen green bag U on the table throughout (told) is no unknown row; the unseen brown box V there from 26 s to
      28 s is one (the log has not told it yet);
    - E0007, a green toy in the open seen at 13, carried by the second task, which has not begun: named by `ego_now`
      too; carried by the first task instead, it is not.
    Regression caught: `told()` of the whole window read at the look (Y, Y2, V and E0007 all hidden by a blocker or a
    task that come after the question)."""
    s = scene(2)
    table = [(0, None, AT_TABLE)]
    briefly = [(0, 26.0, (0.5, 1.5, 0.0)), (26.0, 28.0, AT_TABLE), (28.0, None, (0.5, 1.5, 0.0))]
    put(s, "X", "bag", "red", (0, 0), [2, K], where=table)
    put(s, "X2", "toy", "pink", (0, 0), [2], where=table)
    put(s, "Y", "bag", "blue", (0, 0), [2, K], where=briefly)
    put(s, "Y2", "toy", "yellow", (0, 0), [2], where=briefly)
    put(s, "U", "bag", "green", (0, 0), where=table, size=SMALL)
    put(s, "V", "box", "brown", (0, 0), where=briefly, size=SMALL)
    put(s, "E0007", "toy", "green", (6.0, 3.0), [K])
    job = job_of(TABLE_RULE, zoned=("before_room", "room_done"))
    later = carrying(world_on(s, job, TABLE, a=0, b=1), {1: ("R1", "R2", "E0007")})
    now, mem = ego_now(later, K), ego_mem(later, K)
    assert list(named(now)) == ["Y", "E0007"]
    assert list(named(mem)) == ["Y2"] and {c.subjects[0] for c in mem.unknown} >= {"e:V"}
    assert not {"e:U"} & {c.subjects[0] for c in mem.unknown}
    first = carrying(world_on(s, job, TABLE, a=0, b=1), {0: ("R1", "R2", "E0007")})
    assert list(named(ego_now(first, K))) == ["Y"]
