"""The table systems' answer to `way_to` (`table_way`): the gold's own rule over what the system's mapper and tracker
made of everything it was sent up to the ask. The scene is the robot's drive through the door: `way_kit`'s two rooms and
wall with the door's cells never seen, the robot driving from the east room (5.0, 2.4) west through the door (3.1, 2.4)
to the west room and down to (2.0, 0.5), where it faces south (-90 degrees); `T1` stands at (5.0, 0.5) in the east room,
seen from the first pose (keyframe 3) and from nowhere after, so the way to it is round by the door the robot has
driven. By hand: the door's middle (3.1, 2.4) lies 1.1 m east and 1.9 m north of the base, absolute bearing 60 degrees,
which is 150 degrees to the left of the way it faces: the answer is within 15 degrees of that."""

from __future__ import annotations

import math

from stbench import questions_way, table_way
from stbench.questions_select import Cand, Moment
from stbench.table import Store
from way_kit import FRAMES, HELPER, LAST, block, door_scene, way_world

PI = math.pi
# keyframes 0-4 at the east room's (5.0, 2.4), 5-9 west along y 2.4 through the door, 10 turns south, 11-14 south to the
# base, 15-19 there
DRIVE = ([(5.0, 2.4, PI)] * 5 + [(4.4, 2.4, PI), (3.8, 2.4, PI), (3.1, 2.4, PI), (2.5, 2.4, PI), (2.0, 2.4, PI)]
         + [(2.0, 2.4, -PI / 2), (2.0, 2.0, -PI / 2), (2.0, 1.6, -PI / 2), (2.0, 1.2, -PI / 2), (2.0, 0.9, -PI / 2)]
         + [(2.0, 0.5, -PI / 2)] * 5)
SEEN_AT = 3
NAME = {"kind": "pallet", "attributes": {"color": "grey"}}
HULL = [[4.8, 0.3], [5.2, 0.3], [5.2, 0.7], [4.8, 0.7]]


def cells_by_k():
    """The east room and the wall seen at keyframe 3, the west room at 9 (the door's cells at none)."""
    scene = door_scene(door=False)
    return {SEEN_AT: {c: s for c, s in scene.items() if c[0] >= 30}, 9: {c: s for c, s in scene.items() if c[0] < 30}}


def gold_at(monkeypatch, k):
    """The gold's candidate at the look point `k` (`T1` the one subject, last seen at keyframe 3), or None."""
    thing = Cand("ego_mem", dict(NAME), {"type": "offset", "ahead_m": 1.0, "left_m": 0.0}, "answerable", (SEEN_AT,),
                 ("T1",), {}, decision_k=k, subjects=("e:T1",), answer_class="answerable")
    unknown = Cand("ego_mem", dict(NAME), {"type": "unknown"}, "unknown", (), (), {}, decision_k=k,
                   subjects=("e:U1",), answer_class="unknown")
    monkeypatch.setattr(questions_way, "ego_mem", lambda w, point: Moment(str(point), [thing], [unknown]))
    w = way_world(cells_by_k(), dict(enumerate(DRIVE)))
    m = questions_way.way_to(w, k)
    return w, (m.answerable[0] if m else None)


def store_at(w, upto, seen=True, late=None):
    """What a table system holds after the steps up to keyframe `upto`: those steps of the window, the cells its mapper
    made of them (`late`, if given, are cells of keyframe `upto`) and `T1` as the detection of keyframe 3 (the floor
    hull and middle of a recognisable sighting)."""
    store, by_k = Store(), cells_by_k()
    for j in range(w.rec.k0, upto + 1):
        step = dict(w.rows[j])
        if seen and j == SEEN_AT:
            step["detections"] = [{"d": "d0", "kind": "pallet", "attributes": {"color": "grey"}, "extent_m": [0.4] * 3,
                                   "center_map": [5.0, 0.5, 0.2], "floor_hull": HULL}]
        store.add_step(step)
    store.cells = {j: c for j, c in by_k.items() if j <= upto} | ({upto: late} if late else {})
    return store


def ask(w, k, upto=None, **kw):
    """The table's answer asked at keyframe `k` of a store that holds the stream up to `upto` (default `k`)."""
    store = store_at(w, k if upto is None else upto, **kw)
    slots = {**NAME, "reach_m": 1.0, "room_m": 0.49}
    return table_way.way_to(store, slots, {"now": store.steps[k], "robot": HELPER})


def test_the_table_answers_the_gold_heading_from_what_it_was_sent(monkeypatch):
    """Contract (the ceiling): the table, holding the cells of the steps up to the ask, their poses and `T1`'s track,
    answers the gold's own heading: the best of the outcomes `route_field.route` gives, at the ask's delivered base and
    heading, toward the middle of the thing at its latest recognisable sighting. Hand value (module docstring): the
    heading is the door's, 150 degrees left give or take 15, round by the door the robot drove through (its cells were
    never seen: they are certified by the poses). Asked while the robot is at the door (keyframe 7, at (3.1, 2.4)
    facing west) the route is the rest of the way and gold and table still agree. Regression caught: a table reading
    other cells or poses than the gold (the frame, the camera), a thing's place from another sighting, a heading in
    another frame, which would fail the ceiling gate for the wrong reason."""
    for k in (LAST, 12):
        w, gold = gold_at(monkeypatch, k)
        assert gold is not None, k
        assert ask(w, k) == gold.gold, k
    w, gold = gold_at(monkeypatch, LAST)
    assert 135 <= gold.gold["deg_left"] <= 165


def test_the_table_has_no_answer_where_the_gold_has_none_and_the_current_view_has_none_of_the_thing(monkeypatch):
    """Contract: `unknown` where there is nothing to answer from. A store with no track of the thing (a current view
    that holds the latest frame alone, a thing never seen) has no place to go to; so does the same scene with the door
    never driven (the east room is cut off), where the gold is a dead band too. The gold is never `unknown`, so a table
    that answers it loses nothing it was owed. Regression caught: a default heading made up from an empty track, or a
    route through floor never seen."""
    w, gold = gold_at(monkeypatch, LAST)
    assert ask(w, LAST, seen=False) == {"type": "unknown"}
    cut = way_world(cells_by_k(), dict.fromkeys(range(FRAMES), (2.0, 0.5, -PI / 2)))
    store = store_at(cut, LAST)
    store.cells = {j: c for j, c in store.cells.items()}
    answer = table_way.way_to(store, {**NAME, "reach_m": 1.0, "room_m": 0.49},
                              {"now": store.steps[LAST], "robot": HELPER})
    assert answer == {"type": "unknown"} and questions_way.way_to(cut, LAST) is None


def test_the_table_reads_only_the_steps_and_cells_up_to_the_ask(monkeypatch):
    """Contract: a table system answers from the cells and steps up to the ask and none after it (truth rule 2; the ask
    is made before the later steps exist). Hand case: at keyframe 4 the robot is still at (5.0, 2.4) in the east room,
    1.9 m north of the thing and 1.0 m (the goal's radius R) from the floor within R of it, so the way is straight
    south: one metre of driving south lands in that floor and brings the robot the whole metre nearer, an outcome of
    1.0, the greatest (90 degrees to the left of the way it faces, west); the gold and the table agree on the heading
    that begins the plateau of such headings. A store that also holds what keyframe 19 saw (the east room all occupied,
    the thing's side blocked) answers the ask at 4 as before, and the ask at 19 `unknown`. Regression caught: a store
    read to its end, which a stream could never have given."""
    w, gold = gold_at(monkeypatch, 4)
    assert gold is not None and ask(w, 4) == gold.gold
    assert gold.tolerances["profile_m"][90] == 1.0 == max(v for v in gold.tolerances["profile_m"] if v is not None)
    walled = dict.fromkeys(block(3.2, 6.2, 0, 3.0), "occupied")
    assert ask(w, 4, upto=LAST, late=walled) == gold.gold
    assert ask(w, LAST, late=walled) == {"type": "unknown"}
