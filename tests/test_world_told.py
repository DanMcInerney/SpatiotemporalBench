"""What the robot's own log tells about is never asked at the end (`World.told`, `World.things`): the thing a task
carries and the blockers that stood in a decision's zone. Hand-built worlds on `space_kit`'s run (28 keyframes a task,
1.5 s apart; decision points at keyframes 5 and 21 of kinds `before_room` and `before_doorway` and 18 of kind
`room_done`; the base at the kitchen's centre (2, 1.5) facing north at 5, and at (-0.17, 0) facing east at 21) over the
hand map's rooms with the pieces of `shape_kit`."""

from __future__ import annotations

from dataclasses import replace

import change_kit as ck
import family_kit as fk
from change_kit import thing
from shape_kit import APARTMENT, TABLE
from space_kit import BAG_SIZE, JOB, vacuum_run
from space_kit import record as space_record
from stbench.questions_change import object_was
from stbench.questions_where import askable, pools
from stbench.questions_world import World
from stbench.zones import DOORWAY, SPOT, ZoneRule

TABLE_RULE = ZoneRule("the table ahead of you", "table", height_m=0.1)
SPOT_RULE = ZoneRule("the spot in front of you where you are putting it down", SPOT, height_m=0.01, top_m=0.5)
WAY_RULE = ZoneRule("the doorway ahead of you", DOORWAY, height_m=0.02)
AT_TABLE = (2.0, 2.9, 0.75)   # on the table `shape_kit.TABLE`'s top (x 1.4..2.6, y 2.7..3.3) and in the kitchen


def job_of(rule: ZoneRule, zoned=("before_room", "before_doorway")):
    """`JOB` asking no family at any decision (0.6.0: `zone_clear` is retired) with `rule` the zone of the decision
    kinds `zoned`; `room_done` (keyframe 18) has a zone only when named."""
    return replace(JOB, decisions={"before_room": (), "before_doorway": (), "room_done": ()},
                   zones=dict.fromkeys(zoned, rule), clearance=2.0)


def world_on(s, job, *pieces, piece_map=None, a=0, b=None) -> World:
    """The window of the script as a `World` of `job` over the hand map plus `pieces`, or over `piece_map`."""
    inp = fk.inputs(s, job)
    return World(replace(inp, map=piece_map or replace(inp.map, furniture=tuple(pieces))), space_record(s, a=a, b=b))


def put(s, eid, kind, colour, where, *, mobility="carried", seen=()):
    """A thing (a bag of the catalog's size, any other kind a 0.1 m cube) with placements `where`, sighted at `seen`."""
    s.item(eid, kind, {} if colour is None else {"color": colour}, where, mobility=mobility,
           size=BAG_SIZE if kind == "bag" else (0.1, 0.1, 0.1))
    if seen:
        s.see(list(seen), eid, 1500)
    return eid


def forever(xyz):
    return [(0, None, xyz)]


def test_the_things_that_stand_in_a_decisions_zone_are_told():
    """Contract (0.6.0, plan section 1.2): `World.told` holds every thing (a light piece, a carried thing, clutter or a
    mover; not a fixed piece) whose footprint overlaps the zone of a decision of the window whose kind has a zone in
    `JobSpec.zones`, at that decision's time, the zone being the one the coin sets (nearest piece of the rule's kind to
    the base), whether or not a family is asked there: no decision of these jobs asks one, yet the response and the
    subgoal still tell the zone was blocked. Hand-worked with the table zone x 1.45..2.55, y 2.75..3.25 (the table's
    footprint less 0.05 m) and decisions at t 7.5 s (keyframe 5) of `before_room`, 31.5 s (21) of `before_doorway` and
    27 s (18) of `room_done`, the job asking no family at any of them:
    - told: a bag standing on the table, a light chair, a toy and a person there too, and a pair of shoes there only at
      31.5 s (the second decision), though not at the first;
    - not told: a box 1 m from the table, a bag on it only between 10 and 20 s (no decision then), a toy there only at
      27 s (`room_done` has no zone while it is not named) and a fixed bench in the zone.
    With `room_done` zoned instead of the other two, the toy at 27 s is told and the shoes (still away then) are not;
    with no zone at all nothing is told. Regression caught: a blocker asked at the end because the kind that coined it
    asks nothing (the retired `zone_clear` was the key before), a zone read at another time than the decision, or a
    decision of a kind with no zone counted."""
    s = vacuum_run()
    put(s, "E0001", "bag", "red", forever(AT_TABLE))
    s.item("E0002", "chair", {"color": "oak"}, forever((2.0, 2.9, 0.0)), mobility="light", size=(0.4, 0.4, 0.9))
    put(s, "E0003", "toy", "pink", forever((2.2, 2.9, 0.75)))
    put(s, "E0004", "person", None, forever((1.8, 2.9, 0.0)), mobility="mover")
    put(s, "E0005", "shoes", "black", [(0, 30.0, (0.5, 1.0, 0.0)), (30.0, None, AT_TABLE)])
    put(s, "E0006", "box", "brown", forever((2.0, 2.0, 0.0)))
    away = (0.5, 1.5, 0.0)
    put(s, "E0007", "bag", "blue", [(0, 10.0, away), (10.0, 20.0, AT_TABLE), (20.0, None, away)])
    put(s, "E0008", "toy", "yellow", [(0, 26.0, away), (26.0, 28.0, AT_TABLE), (28.0, None, away)])
    s.item("E0009", "bench", {"color": "oak"}, forever((2.0, 2.9, 0.0)), mobility="fixed", size=(0.5, 0.5, 0.5))
    w = world_on(s, job_of(TABLE_RULE), TABLE)
    assert all(not families for families in w.job.decisions.values())   # nothing asks, yet the blockers are told
    assert w.told() == {"E0001", "E0002", "E0003", "E0004", "E0005"}
    assert world_on(s, job_of(TABLE_RULE, zoned=()), TABLE).told() == set()
    assert world_on(s, job_of(TABLE_RULE, zoned=("room_done",)), TABLE).told() == {
        "E0001", "E0002", "E0003", "E0004", "E0008"}   # the decision at 27 s counts when its kind has a zone


def test_a_told_thing_is_in_no_pool():
    """Contract: a thing the log tells about is in neither of `object_where`'s pools and is not among the askable
    things, whether it was seen (answerable) or not (unknown); the same world with a job that asks no `zone_clear` asks
    it. Hand-worked on a red bag standing on the table (recognisable at keyframe 1) and a blue bag there too that no
    frame shows, beside a green bag elsewhere: with `zone_clear` at `before_doorway` only the green bag is answerable
    and nothing is unknown from the table; without it the red bag is answerable and the blue one unknown. Regression
    caught: the blocker the robot just reacted to asked as a memory question (the response and the subgoal answer
    it)."""
    s = vacuum_run()
    put(s, "E0001", "bag", "red", forever(AT_TABLE), seen=[1])
    put(s, "E0002", "bag", "blue", forever((2.2, 2.9, 0.75)))
    put(s, "E0003", "bag", "green", forever((1.0, 1.0, 0.0)), seen=[1])
    told = world_on(s, job_of(TABLE_RULE), TABLE)
    answerable, unknown = pools(told)
    assert [c.entities for c in answerable] == [("E0003",)]
    assert not {"e:E0001", "e:E0002"} & {c.subjects[0] for c in unknown}
    assert {e.id for e in askable(told)} == {"E0003"}
    plain = world_on(s, replace(JOB, zones={}), TABLE)
    answerable, unknown = pools(plain)
    assert [c.entities for c in answerable] == [("E0001",), ("E0003",)]
    assert "e:E0002" in {c.subjects[0] for c in unknown}


def test_the_surface_window_a_robot_faces_is_the_zone_of_a_spot_rule():
    """Contract: for a `SPOT` rule the zone is the window of the surface the base faces (`zones.zone_of` with the
    delivered heading), so the things told are those on that window, and a decision at which nothing is faced tells
    none. Hand-worked from (2, 2) facing north at the synthetic table, whose window is x 1.8..2.2, y 2.78..3.22: a bag
    on the middle of the table top and a toy at x 2.15 are told, a toy at x 1.5 (on the table, off the window) is not;
    the decision at keyframe 21 faces east from (-0.17, 0) and no surface, so it tells none (and raises nothing).
    Regression caught: the window taken from the base's nearest piece (the heading ignored), or a decision with no
    surface in front of it failing the compile."""
    s = vacuum_run()
    s.poses[5] = (2.0, 2.0)
    put(s, "E0001", "bag", "red", forever(AT_TABLE))
    put(s, "E0002", "toy", "pink", forever((2.15, 2.9, 0.75)))
    put(s, "E0003", "toy", "yellow", forever((1.5, 2.9, 0.75)))
    put(s, "E0004", "toy", "green", [(30.0, None, (-0.17, 0.0, 0.0))])   # in front of nothing at keyframe 21
    assert world_on(s, job_of(SPOT_RULE), TABLE).told() == {"E0001", "E0002"}


def test_a_doorways_way_is_the_zone_of_a_doorway_rule():
    """Contract: for a `DOORWAY` rule the zone is the nearest interior opening's way, so a thing standing in the way is
    told. Hand-worked on `apartment_a` with the base at (4.2, 1.9), nearest to D1 whose way is x 4.6..4.95, y
    1.575..2.225: a bag at (4.775, 1.9) is told, a toy at (4.2, 1.9) (the robot's own place, before the door) and a box
    at (6.0, 1.3) (in D3's way, not the nearest) are not. Regression caught: the doorway zone left out of the told set
    (the vacuum's blockers, which the robot's skip tells about)."""
    s = vacuum_run()
    s.poses[5] = (4.2, 1.9)
    put(s, "E0001", "bag", "red", forever((4.775, 1.9, 0.0)))
    put(s, "E0002", "toy", "pink", forever((4.2, 1.9, 0.0)))
    put(s, "E0003", "box", "brown", forever((6.0, 1.3, 0.0)))
    assert world_on(s, job_of(WAY_RULE), piece_map=APARTMENT).told() == {"E0001"}


def with_targets(w: World, targets: dict[int, tuple[str, ...]]) -> World:
    """The same window over a timeline whose tasks carry these `targets` (by task index)."""
    tasks = [replace(t, targets=targets.get(t.index, t.targets)) for t in w.tl.ego.tasks]
    return World(replace(w.inp, tl=replace(w.tl, ego=replace(w.tl.ego, tasks=tasks))), w.rec)


def test_the_thing_a_task_carries_is_told_for_the_windows_tasks_only():
    """Contract: the entity named by the third of a window task's `targets` is told (the helper's carried thing: its
    subgoal names it), and only a task of the window counts; a third target that is no entity of the world (a region
    id) and a task with fewer than three targets tell nothing. Hand-worked on three tasks with a seen red bag E0001, a
    seen green bag E0003 and an unseen blue bag E0002, task 0 carrying E0001 (targets R1, R2, E0001), task 1 carrying
    nothing (two targets) and task 2 carrying E0002 and, as a fourth target, R9: the windows of task 0, tasks 0-1, task
    1, tasks 1-2 and tasks 0-2 tell {E0001}, {E0001}, {}, {E0002} and both; E0001 is in no pool of a window that carries
    it (E0003 alone is answerable) and E0002, unseen, is not unknown in a window that carries it. Regression caught: a
    carried thing asked back its own place (the log answers it), or a task outside the window telling about a thing."""
    s = vacuum_run(3)
    put(s, "E0001", "bag", "red", forever((1.0, 1.0, 0.0)), seen=[1])
    put(s, "E0002", "bag", "blue", forever((3.0, 1.0, 0.0)))
    put(s, "E0003", "bag", "green", forever((2.0, 1.0, 0.0)), seen=[1])
    targets = {0: ("R1", "R2", "E0001"), 1: ("R1", "R2"), 2: ("R1", "R2", "E0002", "R9")}

    def window(a, b):
        return with_targets(fk.world(s, a, b, "c1", "medium" if a != b else "short"), targets)

    assert [window(a, b).told() for a, b in ((0, 0), (0, 1), (1, 1), (1, 2), (0, 2))] == [
        {"E0001"}, {"E0001"}, set(), {"E0002"}, {"E0001", "E0002"}]
    answerable, unknown = pools(window(0, 0))
    assert [c.entities for c in answerable] == [("E0003",)] and "e:E0002" in {c.subjects[0] for c in unknown}
    assert "e:E0002" not in {c.subjects[0] for c in pools(window(1, 2))[1]}
    third = with_targets(fk.world(s, 0, 0), {0: ("R1", "R2", "R9")})
    assert third.told() == set() and {e.id for e in third.things()} >= {"E0001", "E0003"}


def test_object_was_asks_no_told_thing():
    """Contract: `object_was` asks no thing the log tells about, as `object_where` does not. Hand-worked on two bags
    (red and black) that both move 1.5 m in the gap after task 0 and are seen at keyframes 1 and 29 from the camera at
    (2.0919, 1.5919): the red one is 1.24 m and 0.72 m away, so both its states are asked (last rungs 0.62 and 0.36 m);
    the black one is 2.97 m away in task 0 (last rung 1.49 m, passed) and 4.45 m in task 1 (last rung 2.2 m, not): its
    earlier state only. Asked, the earlier pool holds both and the later the red one; with the second task carrying the
    black bag as its third target only the red one, at both states. Regression caught: a moved blocker or carried thing
    asked about its earlier place, which the robot's own log gives."""
    s = vacuum_run(2)
    thing(s, "E0001", "bag", "red", [(1.0, 1.0), (2.5, 1.0)], [1, 29])
    thing(s, "E0002", "bag", "black", [(5.0, 1.0), (6.5, 1.0)], [1, 29])
    w = ck.window(s, 0, 1)
    early, late = object_was(w)[0].answerable, object_was(w)[0].unknown
    assert {c.entities[0] for c in early} == {"E0001", "E0002"} and {c.entities[0] for c in late} == {"E0001"}
    carrying = with_targets(w, {1: ("R1", "R2", "E0002")})
    early, late = object_was(carrying)[0].answerable, object_was(carrying)[0].unknown
    assert {c.entities[0]: c.evidence_k for c in early} == {"E0001": (1,)}
    assert {c.entities[0]: c.evidence_k for c in late} == {"E0001": (29,)}
