"""The rules `object_where` shares with the change families (`questions_where`): `answerable_at`, `unknown_refs` and
`critical`, by hand on `space_kit`'s run with things (events at keyframes 2, 7, 18 and 26; the end look window is
keyframes 22 to 27), and on the three-task run with a moved bag. `test_questions_where.py` holds the pools' own
contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

import family_kit as fk
from space_kit import JOB, RELEVANT_KINDS, vacuum_run
from stbench.job_registry import Relevance
from stbench.questions_select import Cand
from stbench.questions_where import LastSeen, Ref, answerable_at, critical, never_seen, pools, unknown_refs
from test_questions_where import BAG, moved_bag, thing_world


def thing(w, eid):
    return next(e for e in w.tl.entities if e.id == eid)


def test_answerable_at_is_the_raw_same_task_rule():
    """Contract: `answerable_at(w, (task, k_e), thing)` is the thing's last recognisable sighting at a window keyframe
    at or before `k_e` as (x, y, keyframe), when it lies in the moment's own task; None when there is none by `k_e` or
    the last lies in an earlier task. Hand-worked on three tasks (28 keyframes each) with a red bag at (1, 1) in task 0
    and (2, 1) after, recognisable at 1 and 45: at task 0's 'began sweeping' (keyframe 7) it is (1, 1) from 1; at
    keyframe 0, before any sighting, and at task 1's 'began sweeping' (35: its last sighting by then is 1, in task 0)
    and at task 2's (63: 45, in task 1) None; at task 1's 'finished sweeping' (46) it is (2, 1) from 45. Regression
    caught: a sighting across a gap read as the state after it (the world review's M7), or the first sighting taken for
    the last."""
    w = fk.world(moved_bag([1, 45]), 0, 2, "c1", "medium")
    bag = thing(w, "E0001")
    assert answerable_at(w, (0, 7), bag) == LastSeen((1.0, 1.0), 1)
    assert answerable_at(w, (0, 0), bag) is None
    assert answerable_at(w, (1, 35), bag) is None and answerable_at(w, (2, 63), bag) is None
    assert answerable_at(w, (1, 46), bag) == LastSeen((2.0, 1.0), 45)


def test_answerable_at_leaves_the_end_look_window_and_staleness_to_the_family():
    """Contract: `answerable_at` is only the rule above, so the family keeps its own exclusions: a thing sighted again
    in the window's last look window is still answerable_at (and `pools` leaves it out), and a place a later sighting
    shows left (`World.stale`) is still its answer at the earlier event. Hand-worked: a red bag recognisable at
    keyframes 1 and 25 (the end look window is 22 to 27) answers (1, 1) from 25 at 'got home' (26) and is in no pool; on
    three tasks a bag at (1, 1) at keyframe 1 and (2, 1) at 29, 1 m away, answers (1, 1) at task 0's 'began sweeping'
    (7) though its place is stale then. Regression caught: a family's exclusion moved into the shared rule, where every
    family would inherit it (object_was asks the moved things that `object_where`'s stale and arrived exclusions
    drop)."""
    s = vacuum_run()
    s.item("E0001", "bag", {"color": "red"}, [(0, None, (1.0, 1.0, 0.0))], size=BAG)
    s.see([1, 25], "E0001", 1500)
    w = fk.world(s)
    assert answerable_at(w, (0, 26), thing(w, "E0001")) == LastSeen((1.0, 1.0), 25)
    answerable, unknown = pools(w)
    assert answerable == [] and not any("e:E0001" in c.subjects for c in unknown)
    w = fk.world(moved_bag([1, 29]), 0, 2, "c1", "medium")
    bag = thing(w, "E0001")
    assert answerable_at(w, (0, 7), bag) == LastSeen((1.0, 1.0), 1)
    assert w.stale(bag, 1, (1.0, 1.0))


def test_unknown_refs_list_the_unseen_things_then_the_absent_descriptions():
    """Contract: `unknown_refs(w, task, k_e)` names, in order, the world's own relevant things that no frame of the
    window shows and that exist at `k_e` (world-entity order), then the descriptions that match nothing in this world
    (`never_seen`, in its order); a thing with any pixel in the window, in the end look window or not, and a thing that
    does not exist yet are not among them. Hand values on `things_run`: at 'set off' (keyframe 2) E0005 (a blue bag no
    frame shows) alone; at 'got home' (26) E0005 and then E0006 (a white box that exists from keyframe 20); never E0002
    (500 px), E0003 (dark) or E0007 (in the end look window), which have pixels. Regression caught: a seen thing offered
    as unknown (truth rule 2), a thing that does not exist yet offered, or the absent descriptions ahead of the real."""
    w = thing_world()
    absent = [Ref(kind, dict(attrs), None) for kind, attrs in never_seen(w)]
    assert absent and len(absent) >= 10
    assert unknown_refs(w, 0, 2) == [Ref("bag", {"color": "blue"}, "E0005"), *absent]
    assert unknown_refs(w, 0, 26) == [Ref("bag", {"color": "blue"}, "E0005"), Ref("box", {"color": "white"}, "E0006"),
                                      *absent]


def relevance_job(regions, floor=False):
    """`space_kit.JOB` whose relevance (kinds bag, shoes, toy and box in every task) names these region kinds."""
    return replace(JOB, relevance=lambda plan: Relevance(RELEVANT_KINDS, frozenset(regions), frozenset(), floor))


def test_critical_by_kind_region_and_floor():
    """Contract: `critical(w, task, thing, place)` is true on an answerable row (with its last sighting) when the
    thing's kind is one of the task's relevance kinds, its place lies in a region whose kind the task works in
    (`Relevance.regions`; the map's rooms here: kitchen x 0-4, living room x 4-8, y 0-3) and, where the job sets
    `Relevance.floor`, the thing stood on the floor the robot drives on at that sighting; on an unknown row (no place)
    it is true by kind alone. Hand values (the job's kinds are bag, shoes, toy and box; the vacuum climbs 0.02 m): a
    red bag at (2, 1) with the kitchen and living room asked is critical; a red mug there is not (kind), with or
    without a place; the bag at (9, 1), in no room, is not (place); with only the living room asked the bag at (2, 1)
    is not and at (5, 1) is; the bag stood on a table at height 0.75 m is critical without the floor flag and not with
    it, while the bag on the floor is critical with it; an unknown row of the bag is critical in every one of those
    jobs. Regression caught: every row critical whatever the thing, place or height (a table-top box asked of a
    vacuum), criticality read from the room's name where the job names kinds, or the unknown rows never critical."""
    s = vacuum_run()
    s.item("E0001", "bag", {"color": "red"}, [(0, None, (2.0, 1.0, 0.0))], size=BAG)
    s.item("E0002", "bag", {"color": "blue"}, [(0, None, (1.0, 1.0, 0.75))], size=BAG)
    s.item("E0003", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.0))])
    s.see(1, "E0001", 1500)
    s.see(1, "E0002", 1500)
    s.see(1, "E0003", 1500)
    both, living = relevance_job({"kitchen", "living room"}), relevance_job({"living room"})
    floor = relevance_job({"kitchen", "living room"}, floor=True)
    bag, table_bag = Ref("bag", {"color": "red"}, "E0001"), Ref("bag", {"color": "blue"}, "E0002")
    mug = Ref("mug", {"color": "red"}, "E0003")
    at = {"kitchen": LastSeen((2.0, 1.0), 1), "outside": LastSeen((9.0, 1.0), 1),
          "living room": LastSeen((5.0, 1.0), 1), "table": LastSeen((1.0, 1.0), 1)}
    w = fk.world(s, job=both)
    assert critical(w, 0, bag, at["kitchen"]) and critical(w, 0, bag, None)
    assert not critical(w, 0, mug, at["kitchen"]) and not critical(w, 0, mug, None)
    assert not critical(w, 0, bag, at["outside"])
    only = fk.world(s, job=living)
    assert not critical(only, 0, bag, at["kitchen"]) and critical(only, 0, bag, at["living room"])
    assert critical(only, 0, bag, None)
    assert critical(w, 0, table_bag, at["table"])
    with_floor = fk.world(s, job=floor)
    assert not critical(with_floor, 0, table_bag, at["table"]) and critical(with_floor, 0, bag, at["kitchen"])
    assert critical(with_floor, 0, table_bag, None)


def test_the_pools_rows_are_whole():
    """Contract: `object_where`'s candidates are exactly their rows (slots, gold, slice, class, evidence, entities,
    tolerances, range, subjects and criticality), answerable first in world order and the unknown pool's unseen things
    before the absent descriptions, so the shared rules compose to what is asked. Hand-worked on `things_run`: E0001
    (2, 1) from keyframe 10, E0004 (3.5, 1) from 12; unknown E0005, E0006, then the absent descriptions, whose subject
    is the description; the one subject of a row is the thing (no named time). The answerable rows' class is
    `answerable` and their range the nearest slant (3.9534 m for E0001, seen from the camera at keyframe 10; 2.4426 m
    for E0004) and the job's re-find radius R (0.5 m, whatever the range) as the need, with the resolution of the last
    sighting's pixel (the resolution tests work it out) and no room; the unknown rows' class is
    `unknown`, with no range, the same need and the floor resolution 0.01. Regression caught: a refactor
    that moves a gold, an evidence keyframe, a subject, a class or the order of the pools."""
    w = thing_world()
    answerable, unknown = pools(w)
    nothing = {"resolution_m": 0.01, "need_m": 0.5, "harm": None}

    def row(kind, color, gold, evidence, entity, subject, tolerances=None, range_m=None):
        return Cand("object_where", {"kind": kind, "attributes": {"color": color}}, gold,
                    "answerable" if evidence else "unknown", evidence, (entity,) if evidence else (),
                    nothing if tolerances is None else tolerances, subjects=(subject,), critical=True,
                    answer_class="answerable" if evidence else "unknown", range_m=range_m)

    red, brown = answerable
    assert red.range_m == pytest.approx(3.9534, abs=1e-4) and brown.range_m == pytest.approx(2.4426, abs=1e-4)
    assert red.tolerances == {"resolution_m": pytest.approx(0.5124, abs=1e-4), "need_m": 0.5, "harm": None}
    assert brown.tolerances == {"resolution_m": pytest.approx(0.1958, abs=1e-4), "need_m": 0.5, "harm": None}
    assert answerable == [
        row("bag", "red", {"type": "position", "xy": [2.0, 1.0]}, (10,), "E0001", "e:E0001", red.tolerances,
            red.range_m),
        row("box", "brown", {"type": "position", "xy": [3.5, 1.0]}, (12,), "E0004", "e:E0004", brown.tolerances,
            brown.range_m)]
    assert unknown[:2] == [row("bag", "blue", {"type": "unknown"}, (), None, "e:E0005"),
                           row("box", "white", {"type": "unknown"}, (), None, "e:E0006")]
    kind, attrs = never_seen(w)[0]
    first = unknown[2]
    assert (first.slots["kind"], first.slots["attributes"], first.entities) == (kind, dict(attrs), ())
    assert first.subjects == (f'd:{kind}/{{"color":"{dict(attrs)["color"]}"}}',)
    assert len(unknown) == 2 + len(never_seen(w))
