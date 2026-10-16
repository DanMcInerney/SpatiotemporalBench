"""`out_of_view`'s don't-care places for the warehouse racks' floor-level stock and for the things the robot's own log
tells about, and its `needs` (0.5.0-control lane C4; log review defect 2: the AMR's racks hold carton pallets on the
floor that fit the question's words, yet the gold neither listed them nor gave them a place). Worked by hand on
`view_kit`'s run: the decision at keyframe 21, the base (-0.17, 0), radius 3 m (so places within 3.5 m of the base are
written), chairs seen at keyframe 3. The invented rack is the warehouse's row (4 bays of 2.7 m) centred at (-0.17, 2.0)
and turned a quarter circle (yaw pi/2, local y pointing to -x), with a surface at the floor of bay 1 only."""

from __future__ import annotations

import math
from dataclasses import replace

import family_kit as fk
import view_kit as vk
from space_kit import record, vacuum_run
from stbench import questions, questions_view, score
from stbench.mapspec import Furniture, Surface
from stbench.questions_view import out_of_view
from stbench.questions_world import World
from stbench.stream_protocol import provides
from view_kit import K, thing

STOCK = [[-2.195, 2.0], [-0.845, 2.0]]  # the pallets of bay 2, the only floor stock within 3.5 m of the base (bays 0
# and 3 are 3.9 m and more away; bay 1 has a floor surface and its stock is above the floor)


def rack_map(s):
    """The script's hand map with one rack row: a surface at the floor of bay 1, stock in every other bay-level."""
    surface = Surface("S1", "rack", (0.0, -1.35, 1.08, 2.6), 0.0)
    rack = Furniture("F1", "pallet rack", "pallet_rack", "fixed", (-0.17, 2.0), (1.1, 10.8, 3.9), math.pi / 2, "grey",
                     True, (surface,))
    return replace(fk.hand_map(s), furniture=(rack,))


def world(s, racks: bool = True) -> World:
    """The window of the script's one task, on the rack map or on the hand map without it."""
    inp = fk.inputs(s, vk.VIEW_JOB)
    return World(replace(inp, map=rack_map(s)) if racks else inp, record(s, cid="c1"))


def care(gold: dict) -> list[list]:
    return [[*r["xy"], r["radius"]] for r in gold.get("dont_care", [])]


def run_with_chair():
    """E1, a chair seen at keyframe 3 at (0.83, 0), 1.0 m ahead: an item at any radius."""
    s = vacuum_run()
    thing(s, "E1", (0.83, 0.0))
    return s


def test_floor_stock_is_a_dont_care_place_within_range():
    """Contract: the gold has a don't-care place, radius 0.83 m (0.5 m plus the widening of a pallet), at each
    floor-level stock pallet of the map's racks that lies within radius + 0.5 m of the base, and none beyond it, in a
    bay with a floor surface or above the floor; the items and the matching radius are the chair's alone. By hand: bay 2
    (middle at local y 1.35) has pallets at 0.675 and 2.025, so x = -0.17 - y = -0.845 and -2.195 on y 2.0, 2.11 m and
    2.85 m from the base; bays 0 and 3 are 3.9 m and more from the base (not written); bay 1 has a floor surface (stock
    only above). Without the racks the gold is the chair alone. Regression caught: floor stock left out (a reader who
    names a stock pallet is charged for a right answer, review defect 2), written beyond the range, or written for a bay
    the map gives a surface."""
    c = out_of_view(world(run_with_chair()), K)
    assert c.gold["items"] == [{"xy": [0.83, 0.0]}] and care(c.gold) == [[*xy, 0.83] for xy in STOCK]
    assert c.tolerances == {"match_m": 0.5} and c.evidence_k == (3, K) and c.subjects == (f"w:{K}", "e:E1")
    bare = out_of_view(world(run_with_chair(), racks=False), K)
    assert bare.gold == {"type": "positions", "items": [{"xy": [0.83, 0.0]}]}


def test_an_item_next_to_stock_joins_it():
    """Contract: the swallow rule applies to the stock: an item within a stock place's radius becomes a don't-care place
    (0.5 m) and is not listed, as beside any other don't-care place, while a chair further off stays an item. By hand:
    E2 at (-0.845, 1.2) is 0.8 m from the stock pallet at (-0.845, 2.0) (inside 0.83 m) and 1.38 m from the base:
    don't-care, no item; E1 at (0.83, 0) is 2.6 m from the nearest pallet: the item. Regression caught: an item listed
    inside a don't-care place, so that the gold differs from every other don't-care place's."""
    s = run_with_chair()
    thing(s, "E2", (-0.845, 1.2))
    c = out_of_view(world(s), K)
    assert c.gold["items"] == [{"xy": [0.83, 0.0]}]
    assert care(c.gold) == [[-2.195, 2.0, 0.83], [-0.845, 1.2, 0.5], [-0.845, 2.0, 0.83]]


def test_a_literal_reader_who_names_the_stock_scores_as_one_who_does_not():
    """Contract: through the compiler and the scorer, an answer that also names a stock pallet (at its slot, or 0.6 m
    off it) scores what the same answer without it scores, and a stray place outside every don't-care radius still
    costs. By hand with the chair E1 as the one item: [E1] and [E1, stock (-0.845, 2.0)], [E1, stock (-0.245, 2.0)] and
    [E1, stock (-2.195, 2.3)] all score F1 1 (the stock answers match nothing and lie inside a radius, so they are
    dropped); [E1, (1.5, 2.0)], 2.3 m from the nearest pallet, is a false positive: 2*1/(2+1) = 2/3. With no item at all
    the empty list and the stock answer both score 1; on the map without racks the stock answer scores 0 against the
    empty gold and 2/3 beside the chair, which was the defect. Regression caught: a reader charged for what the
    question's words ask for."""

    def row(s, racks=True):
        inp = fk.inputs(s, vk.VIEW_JOB)
        inp = replace(inp, map=rack_map(s)) if racks else inp
        _, golds = questions.compile_inputs(inp, record(s, cid="c1"))
        (g,) = (g for g in golds if g["family"] == "out_of_view")
        return g

    def answer(*xys):
        return {"type": "positions", "items": [{"xy": list(xy)} for xy in xys]}

    chair = (0.83, 0.0)
    g = row(run_with_chair())
    assert [score.credit(answer(chair, *more), g) for more in
            ((), ((-0.845, 2.0),), ((-0.245, 2.0),), ((-2.195, 2.3),))] == [1.0] * 4
    assert score.credit(answer(chair, (1.5, 2.0)), g) == 2 / 3
    old = row(run_with_chair(), racks=False)
    assert score.credit(answer(chair, (-0.845, 2.0)), old) == 2 / 3
    empty_s = vacuum_run()
    assert [score.credit(answer(*xys), row(empty_s)) for xys in ((), ((-0.845, 2.0),))] == [1.0, 1.0]
    assert score.credit(answer((-0.845, 2.0)), row(empty_s, racks=False)) == 0.0


def test_told_things_are_never_items_and_are_dont_care_places(monkeypatch):
    """Contract: a thing the robot's own log tells about (`World.told()`) is never an item, and every place it was
    sighted at before the look window is a don't-care place, as for a thing with a twin; the rest of the gold is
    unchanged. By hand: the chair E1 (seen at (0.83, 0) at keyframe 3, an item without it) told: no item, a don't-care
    place at (0.83, 0) of radius 0.5, the decision as the only evidence and the time the only subject; an untold second
    chair E2 at (-0.17, -1.0) stays the item. Regression caught: a blocker or a carried thing the log names listed as
    something the robot must remember."""
    s = run_with_chair()
    thing(s, "E2", (-0.17, -1.0))
    plain = out_of_view(world(s, racks=False), K)
    assert plain.gold["items"] == [{"xy": [-0.17, -1.0]}, {"xy": [0.83, 0.0]}] and "dont_care" not in plain.gold
    monkeypatch.setattr(World, "told", lambda self: frozenset({"E1"}))
    c = out_of_view(world(s, racks=False), K)
    assert c.gold == {"type": "positions", "items": [{"xy": [-0.17, -1.0]}],
                      "dont_care": [{"xy": [0.83, 0.0], "radius": 0.5}]}
    assert c.evidence_k == (3, K) and c.subjects == (f"w:{K}", "e:E2")


def test_the_family_needs_no_depth():
    """Contract (FD-0.5.0-control-impl section 11): `out_of_view` needs the camera and the pose, so the default
    `rgb_pose` channel keeps it (`stream._questions` drops a question whose `needs` the channel does not provide); the
    gold may read the depth the camera does not need. Regression caught: `depth` back in `needs`, which would leave the
    AMR's during questions out of every default run."""
    (spec,) = questions_view.SPECS
    assert spec.needs == ("see", "pose") and set(spec.needs) <= provides("rgb_pose")
