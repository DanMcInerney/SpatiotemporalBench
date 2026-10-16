"""The answers a best response tries (plan §4.6): every answer type's candidates over a pool of gold rows. Expected
candidates are worked out by hand in each test."""

from __future__ import annotations

import pytest

from score_kit import SQUARE, UNKNOWN, cells, clock, count, items, num, off, places, pos, rec
from stbench.answer_candidates import MAX_CANDIDATES, candidates, interior_point
from stbench.answers import TYPES, Malformed, parse_answer
from stbench.score import credit

HALL, KITCHEN = [[0, 0], [2, 0], [2, 2], [0, 2]], [[1, 0], [4, 0], [4, 2], [1, 2]]
BEDROOM = [[5, 0], [7, 0], [7, 2], [5, 2]]
ROUTE = {"type": "route", "regions": {"hall": HALL, "kitchen": KITCHEN, "bedroom": BEDROOM},
         "valid": [["hall", "kitchen", "bedroom"], ["hall", "bedroom"]]}
YES, NO = {"type": "yesno", "value": True}, {"type": "yesno", "value": False}


def pool(*golds, **kw):
    return [rec(g, qid=f"q{i}", **kw) for i, g in enumerate(golds)]


def test_candidates_cover_every_type():
    """Contract (map SG-H10): `candidates` handles every type of `answers.TYPES`, with `unknown` first, then the type's
    summaries, then the pool's own values, and raises on any other type; every candidate it offers is a well-formed
    answer. Worked per type:
    - position: the centroid (2, 2) and the median (2, 0) of (0,0), (4,0), (2,6), then each point;
    - offset: the mean (2, 1) and the median (0, 0) of (0, 0), (0, 0), (6, 3), then each distinct offset;
    - number: the median 4 of 1, 3, 5, 7, then each value; time: the median clock (08:00:05 between 08:00:00 and
      08:00:10), then each; yesno and count: each value held, once; set: the empty set;
    - positions: the empty list, then the densest places: (0,0), (0.1,0) and (0,0.2) are one place at their mean
      (0.1/3, 0.2/3) and (3,0) with (3,0.1) another at (3, 0.05): [first], then [first, second];
    - heading: the four turns 0, 90, 180 and -90, the median (-30, 100, 40 give 40), then each gold's own
      (`test_answer_candidates_heading.py`);
    - cells: the empty set, then the cells in at least 1/4, 1/2 and 3/4 of four golds, (0,0) in all of them, (1,0) in
      two, (2,0) and (5,5) in one: {(0,0), (1,0), (2,0), (5,5)}, {(0,0), (1,0)}, {(0,0)}.

    Regression caught: a shortcut gate passing because `blind` never tried an answer of a type, or a candidate the
    scorer reads as malformed."""
    assert candidates(pool(pos(0, 0), pos(4, 0), pos(2, 6))) == [
        UNKNOWN, pos(2, 2), pos(2, 0), pos(0, 0), pos(4, 0), pos(2, 6)]
    assert candidates(pool(off(0, 0), off(0, 0), off(6, 3))) == [UNKNOWN, off(2, 1), off(0, 0), off(6, 3)]
    assert candidates(pool(num(1.0), num(3.0), num(5.0), num(7.0))) == [
        UNKNOWN, num(4.0), num(1.0), num(3.0), num(5.0), num(7.0)]
    start = "2026-05-14T08:00:00"
    assert candidates(pool(clock(start), clock("2026-05-14T08:00:10"))) == [
        UNKNOWN, clock("2026-05-14T08:00:05"), clock(start), clock("2026-05-14T08:00:10")]
    assert candidates(pool(YES, NO, YES)) == [UNKNOWN, YES, NO]
    assert candidates(pool(count(3), count(3), count(5))) == [UNKNOWN, count(3), count(5)]
    assert candidates(pool(items((0, 0, "added")))) == [UNKNOWN, {"type": "set", "items": []}]
    found = candidates(pool(places((0, 0), (3, 0)), places((0.1, 0)), places((0, 0.2), (3, 0.1)),
                            **{"tolerances": {"match_m": 0.5}}))
    assert found[:2] == [UNKNOWN, places()] and len(found) == 4
    first = [0.1 / 3, 0.2 / 3]
    assert found[2]["items"][0]["xy"] == pytest.approx(first)
    assert [v for item in found[3]["items"] for v in item["xy"]] == pytest.approx([*first, 3.0, 0.05])
    four = pool(cells((0, 0), (1, 0)), cells((0, 0), (1, 0), (2, 0)), cells((0, 0)), cells((0, 0), (5, 5)))
    assert candidates(four) == [UNKNOWN, cells(), cells((0, 0), (1, 0), (2, 0), (5, 5)), cells((0, 0), (1, 0)),
                                cells((0, 0))]
    covered = {"position": pool(pos(0, 0)), "offset": pool(off(1, 1)), "time": pool(clock(start)),
               "yesno": pool(YES), "count": pool(count(1)), "number": pool(num(1.0)), "set": pool(items()),
               "positions": pool(places((0, 0))), "cells": pool(cells((0, 0))), "route": pool(ROUTE, credit="route"),
               "heading": pool({"type": "heading", "deg_left": 90.0})}
    assert set(covered) == set(TYPES)
    for kind, golds in covered.items():
        offered = candidates(golds)
        assert offered[0] == UNKNOWN and len(offered) > 1, kind
        assert not any(isinstance(parse_answer(c), Malformed) for c in offered), kind
    with pytest.raises(ValueError, match="teleport"):
        candidates(pool({"type": "teleport"}))


def test_a_region_gold_gets_an_interior_point_per_region_that_scores_one():
    """Contract: for `region` golds the candidates are one interior point of each distinct region, not the gold points
    (which the pool names no better than any point in the room): two golds on a 4 m square and one on a 2 m by 2 m room
    give the squares' centres (2, 2) and (11, 1) once each, and each scores 1 on its own gold, erosion 0.2 included.
    In an L-shaped room (arms 2 m wide) whose centroid, (5/3, 5/3), is 0.47 m from the reflex corner, the point found is
    at least 1 m from every wall (a circle of up to 1.17 m fits), where the centroid would fail a 1 m erosion.
    Regression caught: a best response to a room question that cannot
    stand inside the room, so its shortcut gate is never tested."""
    room = [[10, 0], [12, 0], [12, 2], [10, 2]]
    golds = [rec(pos(1, 1), qid="a", credit="region", region=SQUARE, tolerances={"erode_m": 0.2}),
             rec(pos(3, 3), qid="b", credit="region", region=SQUARE, tolerances={"erode_m": 0.2}),
             rec(pos(11, 1), qid="c", credit="region", region=room, tolerances={"erode_m": 0.2})]
    assert candidates(golds) == [UNKNOWN, pos(2, 2), pos(11, 1)]
    assert [credit(c, g) for c, g in zip(candidates(golds)[1:], (golds[0], golds[2]), strict=True)] == [1.0, 1.0]
    ell = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
    point = interior_point(ell)
    assert credit({"type": "position", "xy": list(point)}, rec(pos(1, 1), credit="region", region=ell,
                                                                tolerances={"erode_m": 1.0})) == 1.0
    assert interior_point([(0, 0), (4, 0), (4, 4), (0, 4)]) == (2.0, 2.0)


def test_every_valid_route_is_offered_as_interior_points_that_score_one():
    """Contract: for a route gold each valid sequence is offered as an interior point of each of its regions in order
    (hall (1, 1), kitchen (2.5, 1), bedroom (6, 1)), and each scores 1 on the gold: two valid sequences, two routes;
    the same gold twice in a pool gives them once. Regression caught: a route shortcut that never tries a route in the
    rooms the gold names, so `blind` could not find the answer its priors would give."""
    golds = pool(ROUTE, ROUTE, credit="route")
    found = candidates(golds)
    assert found == [UNKNOWN, {"type": "route", "points": [[1, 1], [2.5, 1], [6, 1]]},
                     {"type": "route", "points": [[1, 1], [6, 1]]}]
    assert [credit(c, golds[0]) for c in found[1:]] == [1.0, 1.0]


def test_unknown_golds_add_nothing_and_the_list_is_capped_with_summaries_first():
    """Contract: an unknown gold contributes no candidate (`unknown` is offered anyway, first); a pool of 100 distinct
    positions gives 60 candidates, `unknown` first and the centroid and median next (summaries are never the ones cut),
    and no candidate twice. Regression caught: the cap cutting the summaries, or a pool of unknowns offering nothing
    but a crash."""
    only_unknown = [rec(UNKNOWN, slice="unknown", qid=f"u{i}") for i in range(3)]
    assert candidates(only_unknown) == [UNKNOWN] and candidates([]) == [UNKNOWN]
    wide = pool(*(pos(i, 2 * i) for i in range(100)))
    found = candidates(wide)
    assert len(found) == MAX_CANDIDATES == 60 and found[0] == UNKNOWN
    assert found[1] == pos(49.5, 99.0) and found[2] == pos(0, 0)  # the centroid and median are one answer
    assert len({repr(c) for c in found}) == 60
