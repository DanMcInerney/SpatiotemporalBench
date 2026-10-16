"""`task_things` (questions_task): the gold by hand on `task_kit`'s world. A task's docking leg runs along y = 1 from
x = 0 to 40 (`PATH`: within 1 m, band 0.25 m, so an item's nearest point is within 0.75 m of the line, a don't-care
place's within 1.25 m); a small thing (0.4 m) has its nearest point 0.2 m nearer than its centre, a large one (1 m)
0.5 m; a don't-care circle has radius 0.5 m plus the thing's widening (0 for 0.4 m, 0.25 m for 1 m)."""

from __future__ import annotations

import pytest

import family_kit as fk
import task_kit as tk
from stbench import questions, questions_task, score
from stbench.questions_text import CAPPED, FAMILIES, RELAX
from stbench.questions_world import World
from stbench.records import check_gold
from stbench.task_predicate import TaskThings
from task_kit import BIG, LAST, PATH, cand, gold, places, put, y


def test_an_item_is_inside_the_footprints_reach_and_the_band_is_dont_care():
    """Contract (plan §4 A4): the gold lists the carried things whose footprint's nearest point is within reach less
    band of the way, makes a don't-care circle of those within the band, and ignores the rest; the distance is to the
    footprint, not the centre. By hand: A (a 0.4 m bag) 0.5 m away is an item at (2, 1.7); B 1.0 m away is don't-care; C
    1.5 m away is nothing; D, a 1 m thing centred 1.6 m off (nearest point 1.1 m: the band; by its centre it would be
    nothing) is don't-care with radius 0.75; E, a 1 m thing centred 0.8 m off (nearest point 0.3 m: inside; by its
    centre it would be in the band) is an item. Evidence is the items' sightings (3 and 5) and the task's last keyframe
    (15); `match_m` is the job's 0.5 m plus the largest item widening (E's 0.25 m), and `resolutions_m` the pixel at
    each item's place from the camera of its sighting, in the gold's order: A is 0.14 m from the camera of keyframe 3
    (0.08 m up), 0.0009 m, held at 0.01; E is 12.1 m from the camera of keyframe 5, 4.798 (the resolution tests work
    them out). Regression caught: the centre's distance used for a large thing, no band, or a band on one side."""
    s = tk.script()
    put(s, "A", (2, y(0.5)))
    put(s, "B", (5, y(1.0)))
    put(s, "C", (8, y(1.5)))
    put(s, "D", (11, y(1.1, 1.0)), BIG)
    put(s, "E", (14, y(0.3, 1.0)), BIG, at=(5,))
    c = cand(s)
    assert places(c) == ([[2.0, 1.7], [14.0, 1.8]], [([5.0, 2.2], 0.5), ([11.0, 2.6], 0.75)])
    assert (c.evidence_k, c.answer_class) == ((3, 5, LAST), "items")
    assert c.tolerances == {"match_m": 0.75, "resolutions_m": [0.01, pytest.approx(4.798, abs=1e-3)], "fp_weight": 0.5}
    assert (c.family, c.slots, c.slice, c.subjects, c.entities) == (
        "task_things", {"phrase": PATH.phrase, "task": 0}, "answerable", (), ())


def test_each_dont_care_rule_makes_a_place_not_an_item(monkeypatch):
    """Contract (plan §4 A4): a thing inside the reach is still not an item, and is a don't-care place wherever it
    stood, when it is light furniture, a person, told, a twin, in a row of the window's last look window (keyframes
    8-15), only glimpsed (100 px), never seen, moved within the task (a glimpse elsewhere after its last recognisable
    place), or a hair off the floor (0.05 m up, over the robot's 0.02 m climb but under it plus 0.05 m). All stand 0.5 m
    from the way, 3 m apart; the control, plain, is the one item, at (32, 1.7). The twin's partner (the same
    description, far off) is nothing, and the moved thing leaves two places. A shelf bag (0.75 m up) is clearly off the
    floor and is nothing at all, not a place. Regression caught: any rule dropped (a table that cannot tell the thing
    from its twin, or the notebook of the last look window, would be charged for it), a don't-care place written where
    the thing is not outside the reach, or one for a thing clearly off the floor (free space for a point next to a real
    item)."""
    s = tk.script()
    yy = y(0.5)
    put(s, "CHAIR", (1, yy), (0.4, 0.4, 0.9), kind="chair", mobility="light")
    put(s, "WALKER", (4, yy), (0.5, 0.4, 1.7), kind="person", mobility="mover", attrs={})
    put(s, "TOLD", (7, yy))
    put(s, "TWIN1", (10, yy), attrs={"color": "red"})
    put(s, "TWIN2", (10, 5.0), attrs={"color": "red"})
    put(s, "LATE", (13, yy), at=(3, 9))
    put(s, "GLIMPSE", (16, yy), px=100)
    put(s, "UNSEEN", (19, yy), at=())
    put(s, "MOVED", (0, 0), stands=((0, 7.0, (25.0, yy, 0.0)), (7.0, None, (26.0, yy, 0.0))), at=(3,))
    s.see(6, "MOVED", 100)
    put(s, "CURB", (29, yy), z=0.05)
    put(s, "SHELF", (30.5, yy), z=0.75)
    put(s, "CONTROL", (32, yy))
    monkeypatch.setattr(World, "told", lambda self: frozenset({"TOLD"}))
    items, care = places(cand(s))
    assert items == [[32.0, 1.7]]
    assert [xy for xy, _ in care] == [[x, 1.7] for x in (1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 25.0, 26.0, 29.0)]
    assert {r for _, r in care} == {0.5}


def test_a_predicate_with_a_minimum_height_and_dont_care_kinds():
    """Contract (plan §4 A4; 0.7.0, the dev pilot's log review): a thing no taller than `min_height_m` is never an item
    but a don't-care place (the prompt's "thing" counts a remote, so naming one must cost nothing), and a kind in
    `dont_care_kinds` is never an item and is a don't-care place. Minimum 0.02 m, kind `bag` don't-care, all 0.5 m from
    the way: a 0.01 m sheet is a don't-care place, a 0.03 m box is an item, a bag is don't-care; a sheet 3 m from the
    way, outside the predicate, is nothing. Regression caught: a rug asked for as a thing on the floor, a thin thing
    dropped silently (a model that listed the remote lost credit), a thin thing's circle drawn outside the predicate,
    or a kind that was declared never-an-item listed."""
    s = tk.script()
    put(s, "SHEET", (2, y(0.5)), (0.4, 0.4, 0.01), kind="sheet")
    put(s, "BOX", (5, y(0.5)), (0.4, 0.4, 0.03), kind="box")
    put(s, "BAG", (8, y(0.5)))
    put(s, "FAR", (11, y(3.0)), (0.4, 0.4, 0.01), kind="sheet")
    pred = TaskThings(PATH.phrase, "path", PATH.phases, reach_m=1.0, band_m=0.25, floor=True, min_height_m=0.02,
                      dont_care_kinds=frozenset({"bag"}))
    assert places(cand(s, pred)) == ([[5.0, 1.7]], [([2.0, 1.7], 0.5), ([8.0, 1.7], 0.5)])


def test_a_listed_thin_thing_costs_nothing_and_an_unseen_one_swallows_like_any_other():
    """Contract (0.7.0, the dev pilot's log review; the swallow of `questions_task`): naming a thin thing (a remote,
    1 cm) is neither right nor wrong, and its circle swallows exactly when any don't-care circle does (a thing no frame
    shows is sensor-free). By hand, a bag item at (2, 1.7) and a remote 0.8 m beside it at (2.8, 1.7): seen, the
    remote's circle matches nothing and swallows nothing, so the gold lists the bag and an answer of the bag and the
    remote scores 1.0 (the remote's point is dropped); unseen, the circle is sensor-free and reaches the bag (0.8 m
    inside 0.5 + 0.5 m), so the gold lists nothing and the empty answer scores 1.0. Regression caught: the thin thing
    counted wrong when named (the pilot's remotes), or its circle exempt from the swallow."""
    pred = TaskThings(PATH.phrase, "path", PATH.phases, reach_m=1.0, band_m=0.25, floor=True, min_height_m=0.02)
    both = {"type": "positions", "items": [{"xy": [2.0, 1.7]}, {"xy": [2.8, 1.7]}]}
    empty = {"type": "positions", "items": []}
    for seen in (True, False):
        s = tk.script()
        put(s, "BAG1", (2, y(0.5)))
        put(s, "REMOTE", (2.8, y(0.5)), (0.15, 0.04, 0.01), kind="remote", at=(3,) if seen else ())
        c = cand(s, pred)
        if seen:
            assert places(c) == ([[2.0, 1.7]], [([2.8, 1.7], 0.5)])
            assert score_of(c, both) == 1.0
        else:
            assert places(c)[0] == [] and c.answer_class == "empty"
            assert score_of(c, empty) == 1.0


def spray(c) -> dict:
    """The answer of a system that knows where the gold's circles are and no more: a point inside each."""
    return {"type": "positions", "items": [{"xy": [x + 0.1, yv]} for (x, yv), _ in places(c)[1]]}


def score_of(c, answer: dict) -> float:
    """The penalised score of an answer against the candidate's gold row."""
    (_, (g,)) = questions.records([c])
    return score.penalised(answer, g)


@pytest.mark.parametrize("free", ["light", "told", "unseen", "stock"])
def test_an_item_within_a_sensor_free_circle_and_the_match_radius_is_swallowed(free, monkeypatch):
    """Contract (plan §4 A4, the swallow): an answer point inside a don't-care circle that matches nothing is dropped,
    not counted wrong, so a system that knows where a circle is without the sensors (the map's stock and light
    furniture, the log's told things, a thing no frame shows) could place points in it, next to likely items, for free.
    An item within the circle's radius plus the question's `match_m` (0.5 + 0.5 m) of such a place is therefore
    don't-care. By hand: a bag item at (2, 1.7) and a sensor-free place 0.8 m beside it, at (2.8, 1.7) (outside the 0.5
    m circle, inside the 1.0 m ring): a chair, a told bag, a bag no frame shows; the stock case has a pallet at (6,
    14.675) (circle 0.83 m) and the item at (7.1, 14.7), 1.1 m off. The gold lists nothing, a system that sprays one
    point in each circle scores exactly what the empty answer does (gold empty: 1.0), and a point outside every circle
    is wrong. Regression caught: the swallow reaching only the circle's own radius, which left a ring of free space
    around every sensor-free place."""
    if free == "stock":
        s = tk.script(path={8: (0.0, 14.0), 9: (20.0, 14.0), 10: (40.0, 14.0)})
        put(s, "BAG1", (7.1, 14.7))
        c = cand(s, m=tk.room_map(s, stock=True))
        item = [7.1, 14.7]
    else:
        s = tk.script()
        put(s, "BAG1", (2, y(0.5)))
        if free == "light":
            put(s, "OTHER", (2.8, y(0.5)), (0.4, 0.4, 0.9), kind="chair", mobility="light")
        else:
            put(s, "OTHER", (2.8, y(0.5)), at=(3,) if free == "told" else ())
        if free == "told":
            monkeypatch.setattr(World, "told", lambda self: frozenset({"OTHER"}))
        c = cand(s)
        item = [2.0, 1.7]
    assert places(c)[0] == [] and c.answer_class == "empty"
    assert score_of(c, spray(c)) == score_of(c, {"type": "positions", "items": []}) == 1.0
    assert score_of(c, {"type": "positions", "items": [{"xy": item}]}) == 1.0   # the swallowed item is a circle too
    assert score_of(c, {"type": "positions", "items": [{"xy": [30.0, 30.0]}]}) < 1.0   # a point elsewhere is wrong


def test_a_circle_the_sensors_show_swallows_nothing():
    """Contract (plan §4 A4): the circles of what a system cannot place without seeing it (a person, a seen twin, a
    thing in the last look window) swallow no item, since a point put there would need the sighting. A bag item at
    (2, 1.7) with a person 0.4 m beside it at (2.4, 1.7), a seen twin 0.4 m the other side at (1.6, 1.7) (its partner
    far off) and a bag in the last look window at (2.0, 2.1): each circle reaches the item, and the item is still an
    item. Regression caught: every circle swallowing, which costs half the items of a dense job for no reason."""
    s = tk.script()
    put(s, "BAG1", (2, y(0.5)))
    put(s, "WALKER", (2.4, y(0.5)), (0.5, 0.4, 1.7), kind="person", mobility="mover", attrs={})
    put(s, "T1", (1.6, y(0.5)), attrs={"color": "red"})
    put(s, "T2", (5, 5.0), attrs={"color": "red"})
    put(s, "LATE", (2.0, y(0.9)), at=(3, 9))
    assert places(cand(s))[0] == [[2.0, 1.7]]


def test_an_items_earlier_places_in_the_task_are_dont_care():
    """Contract (plan §4 A4): the question never says "last", so a thing recognised in one place and then, later in the
    same task, in another is an item at the later place and a don't-care place at the earlier one (a reader who names
    where it was is not wrong). A bag at (2, 1.7) until t = 7 s (keyframe 4), then at (3.2, 1.7), recognised at
    keyframes 3 and 6: the item is (3.2, 1.7), the don't-care place (2, 1.7), the two 1.2 m apart. Regression caught:
    the earlier place dropped (a right answer there charged as a false positive), or listed as an item."""
    s = tk.script()
    put(s, "BAG1", (0, 0), stands=((0, 7.0, (2.0, y(0.5), 0.0)), (7.0, None, (3.2, y(0.5), 0.0))), at=(3, 6))
    assert places(cand(s)) == ([[3.2, 1.7]], [([2.0, 1.7], 0.5)])


def test_a_long_thing_turned_is_judged_by_both_footprints():
    """Contract (plan §4 A4): a thing whose placed footprint is inside the reach but whose detected footprint (extents,
    no heading) is in the band is don't-care, not an item, since the oracle sees only the detection. A 1.2 x 0.4 m
    thing turned 90 degrees centred (5, 2.0): placed it spans y 1.4 to 2.6, 0.4 m from the way (inside the 0.75 m);
    detected unturned it spans y 1.8 to 2.2, 0.8 m (the band). Regression caught: the gold judging the placed footprint
    only, listing an item the table sees as in the band. (The circle's radius is 0.5 m plus the 1.2 m thing's widening,
    0.35 m.)"""
    s = tk.script()
    put(s, "LONG", (5, 2.0), (1.2, 0.4, 0.3), yaw_deg=90.0)
    assert places(cand(s)) == ([], [([5.0, 2.0], 0.85)])


def test_a_point_anchor_and_a_rooms_anchor():
    """Contract (plan §4 A4): a `point` anchor measures from the median base over the phase, and a `rooms` anchor lists
    the footprints wholly inside the rooms cleaned and off their doorway ways, with a way and a room's edge don't-care.
    Point (within 2 m, band 0.5 m of (2, 1.5), where the kitchen's cleaning steps stand): a bag 1.0 m off is an item,
    2.0 m off don't-care, 3.0 m off nothing. Rooms (the living room cleaned): a bag wholly inside, at (6.5, 0.8), is an
    item; one on the door's way, (4.2, 1.5), and one across the room's edge, (4.0, 0.5), are don't-care; one in the
    kitchen, (2, 1), is nothing. Regression caught: the anchor's median or rooms wrong, or a thing in a doorway
    counted as in the room."""
    s = tk.script()
    put(s, "NEAR", (2.0, 1.5 + 0.2 + 1.0))
    put(s, "EDGE", (-0.2, 1.5))
    put(s, "FAR", (-1.2, 1.5))
    assert places(cand(s, tk.POINT)) == ([[2.0, 2.7]], [([-0.2, 1.5], 0.5)])
    s = tk.script(room="living room")
    put(s, "INSIDE", (6.5, 0.8))
    put(s, "ONWAY", (4.2, 1.5))
    put(s, "ACROSS", (4.0, 0.5))
    put(s, "KITCHEN", (2.0, 1.0))
    assert places(cand(s, tk.ROOMS, m=tk.room_map(s))) == ([[6.5, 0.8]], [([4.0, 0.5], 0.5), ([4.2, 1.5], 0.5)])


def test_the_racks_floor_stock_is_dont_care_within_the_predicate():
    """Contract (plan §4 A4): the floor-level stock of the map's racks (scenery, no entity) is a don't-care place
    wherever its pallet is not outside the predicate, with the radius of the job's `match_m` plus a pallet's widening
    (0.5 + 0.33 m). On a map with one rack at (6, 14) whose two stock pallets stand at (6, 13.325) and (6, 14.675)
    (worked out in `task_kit.room_map`), a way along y = 14 has both pallets inside; a way along y = 30 has neither.
    Regression caught: stock a reader names as a thing on the floor charged as a false positive, or stock don't-care
    everywhere on the map."""
    near = tk.script(path={8: (0.0, 14.0), 9: (20.0, 14.0), 10: (40.0, 14.0)})
    c = cand(near, m=tk.room_map(near, stock=True))
    assert places(c) == ([], [([6.0, 13.325], 0.83), ([6.0, 14.675], 0.83)])
    far = tk.script(path={8: (0.0, 30.0), 9: (20.0, 30.0), 10: (40.0, 30.0)})
    assert places(cand(far, m=tk.room_map(far, stock=True))) == ([], [])


def test_an_empty_gold_has_the_tasks_last_keyframe_as_evidence_and_passes_check_gold():
    """Contract (plan §4 A4): a task whose predicate selects nothing still has a gold, the empty list, with the task's
    last keyframe (15) as its evidence, so `records.check_gold` (an answerable gold needs evidence) accepts the row; its
    class is `empty` and it has no subjects or entities. Compiled through the questionnaire to a row. Regression
    caught: an empty gold with no evidence, which the compile would refuse, or a gold dropped for being empty."""
    s = tk.script()
    put(s, "FARAWAY", (5, y(3.0)))
    inp = fk.inputs(s, tk.job())
    qs, gs = questions.compile_inputs(inp, tk.space_record(s))
    (q,), (g,) = qs, gs
    assert check_gold(g) == []
    assert (g["gold"], g["evidence_k"], g["answer_class"], g["slice"], g["subjects"], g["entities"]) == (
        {"type": "positions", "items": []}, [LAST], "empty", "answerable", [], [])
    assert g["tolerances"] == {"match_m": 0.5, "resolutions_m": [], "fp_weight": 0.5}
    assert (g["credit"], g["scope"], g["critical"]) == ("graded_set", "memory", True)
    assert q["text"] == ("Where were the things within 1 m of the way you docked in your first task, which you "
                         "would have to steer around next time? Answer in map coordinates. " + questions_task.COUNTED)
    assert q["text"] == FAMILIES["task_things"].render(q["slots"])   # render purity: the text is its slots' rendering


def test_relax_makes_the_named_items_dont_care_and_recomputes_the_class():
    """Contract (plan §3, §4 A4): `RELAX` turns the items of the entities a look question names into don't-care places,
    keeping the rest, and rebuilds the gold, the evidence, the matching radius and the class, the swallow included. X at
    (2, 1.7) seen at 3, Y 0.4 m beside it at (2.4, 1.7) seen at 4 and a large Z at (14, 1.8) seen at 5: all items,
    evidence 3, 4, 5, 15, match 0.75. Relaxing Z leaves X and Y, evidence 3, 4, 15, match 0.5; relaxing X makes X
    don't-care and leaves Y and Z (a relaxed item's circle swallows nothing); relaxing all three leaves `empty` with
    evidence 15.
    A name that is no item changes nothing. `CAPPED` holds the family. Regression caught: a named thing still listed
    (asked twice), the class not recomputed (an emptied gold counted as `items` in the empty cap), or evidence of
    a sighting that is no longer an item's."""
    s = tk.script()
    put(s, "X", (2, y(0.5)))
    put(s, "Y", (2.4, y(0.5)), at=(4,))
    put(s, "Z", (14, y(0.3, 1.0)), BIG, at=(5,))
    c = cand(s)
    relax = RELAX["task_things"]
    assert "task_things" in CAPPED
    assert (places(c)[0], c.evidence_k, c.tolerances["match_m"]) == ([[2.0, 1.7], [2.4, 1.7], [14.0, 1.8]],
                                                                      (3, 4, 5, LAST), 0.75)
    z = relax(c, {"Z"})
    assert (places(z)[0], z.evidence_k, z.tolerances["match_m"], z.answer_class) == (
        [[2.0, 1.7], [2.4, 1.7]], (3, 4, LAST), 0.5, "items")
    assert places(z)[1] == [([14.0, 1.8], 0.75)]
    x = relax(c, {"X"})   # X's circle is not sensor-free: it swallows no neighbour (Y, 0.4 m off, stays an item)
    assert (places(x)[0], x.evidence_k, x.answer_class) == ([[2.4, 1.7], [14.0, 1.8]], (4, 5, LAST), "items")
    assert places(x)[1] == [([2.0, 1.7], 0.5)]
    none = relax(c, {"X", "Y", "Z"})
    assert (places(none)[0], none.evidence_k, none.answer_class, none.tolerances) == (
        [], (LAST,), "empty", {"match_m": 0.5, "resolutions_m": [], "fp_weight": 0.5})
    assert relax(c, {"nobody"}).gold == c.gold


def test_every_task_of_the_window_is_a_moment_and_a_task_without_an_anchor_is_none():
    """Contract (plan §4 A4): one moment for each task of the window, keyed by its case index, whose slots carry that
    index; a thing sighted in one task only is an item there and, standing on in the other where no frame recognises it,
    a don't-care place (A is an item in task 0 and B a don't-care place there, and the reverse in task 1); a task with
    no step of the predicate's phases, and a job with no predicate, give no moment. By
    hand over two tasks: A at (2, 1.7) seen at keyframe 3, B at (8, 1.7) seen at keyframe 19 (task 1's 3rd). Regression
    caught: a window's tasks answered with the last task's anchor, or the unseen standing thing listed in a task that
    never recognised it."""
    s = tk.script(2)
    put(s, "A", (2, y(0.5)))
    put(s, "B", (8, y(0.5)), at=(19,))
    first, second = gold(s, a=0, b=1)
    assert (first.key, second.key) == ("0", "1")
    (c0,), (c1,) = first.answerable, second.answerable
    assert [c.slots["task"] for c in (c0, c1)] == [0, 1]
    assert places(c0) == ([[2.0, 1.7]], [([8.0, 1.7], 0.5)])
    assert places(c1) == ([[8.0, 1.7]], [([2.0, 1.7], 0.5)])
    assert gold(s, TaskThings("p", "path", frozenset({"charging"}))) == []
    assert gold(s, None) == []


@pytest.mark.parametrize("override", [{"dont_care_kinds": frozenset({"bag"})}, {"min_height_m": 0.5}])
def test_the_predicate_the_job_declares_is_the_one_the_gold_reads(override):
    """Contract: the gold reads `JobSpec.task_things` of the world's job and nothing else, so changing the job's
    predicate changes the gold. The same bag 0.5 m from the way: an item under `PATH`, no longer one when its kind is
    don't-care or the minimum height passes it. Regression caught: a predicate cached or defaulted behind the job."""
    s = tk.script()
    put(s, "A", (2, y(0.5)))
    assert places(cand(s))[0] == [[2.0, 1.7]]
    changed = TaskThings(PATH.phrase, "path", PATH.phases, reach_m=1.0, band_m=0.25, floor=True, **override)
    assert places(cand(s, changed))[0] == []


def test_only_the_predicates_item_kinds_are_items_the_recurring_kinds_are_dont_care_and_the_rest_nothing():
    """Contract (0.11.0, FD-0.11-design 3.4.3 and 4.5; the vacuum's list of what its brush could catch): when the
    predicate names `item_kinds`, a carried thing or clutter of one of them inside the reach is an item, one of
    `dont_care_kinds` (the kinds that recur at one spot) is a don't-care place, and one of any other kind is nothing at
    all, no item and no place (the question does not ask for it, so a reader who lists it is charged a false place: the
    words name the kinds, `test_catch_words`); light furniture and people stay don't-care by their mobility whatever
    their kind. By hand, all 0.5 m from the way and 3 m apart: a cord at (2, 1.7) is the one item; a mat (a don't-care
    kind) at (5, 1.7) and a chair (light) at (8, 1.7) are don't-care places of radius 0.5; a bag (any other kind) at
    (11, 1.7) is not in the gold. With no `item_kinds` the same world lists the cord and the bag and makes the mat and
    the chair don't-care. Regression caught: a kind the question never named listed as an item, a kind it names left
    out of the gold, a recurring kind an item (the layout prior of 0.9.0's `path_log_in_pool`), or
    the chair a list item or nothing (it is the prompt's 'not counted either way')."""
    s = tk.script()
    put(s, "CORD", (2, y(0.5)), kind="extension_cord")
    put(s, "MAT", (5, y(0.5)), kind="bath_mat")
    put(s, "CHAIR", (8, y(0.5)), (0.4, 0.4, 0.9), kind="chair", mobility="light")
    put(s, "BAG", (11, y(0.5)))
    kinds = TaskThings(PATH.phrase, "path", PATH.phases, reach_m=1.0, band_m=0.25, floor=True,
                       item_kinds=frozenset({"extension_cord", "bath_mat"}), dont_care_kinds=frozenset({"bath_mat"}))
    assert places(cand(s, kinds)) == ([[2.0, 1.7]], [([5.0, 1.7], 0.5), ([8.0, 1.7], 0.5)])
    open_kinds = TaskThings(PATH.phrase, "path", PATH.phases, reach_m=1.0, band_m=0.25, floor=True,
                            dont_care_kinds=frozenset({"bath_mat"}))
    assert places(cand(s, open_kinds)) == ([[2.0, 1.7], [11.0, 1.7]], [([5.0, 1.7], 0.5), ([8.0, 1.7], 0.5)])
