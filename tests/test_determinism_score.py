"""The scorer's output in a second process (`determinism_kit.py`: nothing is recorded, the two are compared).

`scorer_output` below is a credit-only battery of the credits added to the first scorer (`cells`, `route`, `region`,
`relative`), the aggregate of a fixed set of rows (the base battery scored through `score.row`) and the gates' report on
fixed synthetic rows, which hold a family naming `other_case` as its reference on a held-out layout and on a public
one, so the output covers both branches. The test checks that its digest is the same in a second process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

from determinism_kit import digest
from score_kit import SQUARE, result_row
from stbench import gates, score
from stbench.map_files import HELD_OUT


def _pos(x, y, z=None):
    return {"type": "position", **({"xyz": [x, y, z]} if z is not None else {"xy": [x, y]})}


def _at(seconds):
    return {"type": "time", "clock": (datetime(2026, 5, 14, 8, 37, 0) + timedelta(seconds=seconds)).isoformat()}


def _count(v):
    return {"type": "count", "value": v}


def _num(v):
    return {"type": "number", "value": v}


def _yes(v):
    return {"type": "yesno", "value": v}


def _set(*items, dont_care=()):
    return {"type": "set", "items": [{"xy": [x, y], "change": c} for x, y, c in items],
            **({"dont_care": [{"xy": [x, y], "radius": r} for x, y, r in dont_care]} if dont_care else {})}


def _places(*points, dont_care=()):
    """A `positions` answer from (x, y) or (x, y, z) points."""
    return {"type": "positions", "items": [{"xy": list(p)} if len(p) == 2 else {"xyz": list(p)} for p in points],
            **({"dont_care": [{"xy": [x, y], "radius": r} for x, y, r in dont_care]} if dont_care else {})}


def _gold(gold, family, *, slice="answerable", metric="xy", tolerances=None, **extra):
    return {"qid": "", "family": family, "gold": gold, "slice": slice, "evidence_k": [3], "entities": [],
            "tolerances": tolerances or {}, "metric": metric, **extra}


UNKNOWN = {"type": "unknown"}
AT_ORIGIN, AT_NOON = _gold(_pos(0, 0), "where_last"), _gold(_at(0), "when_last")
THREE, ONE, NO_COUNT = _gold(_count(3), "count"), _gold(_count(1), "count"), _gold(_count(0), "count")
FOUR, TEN = _gold(_count(4), "count"), _gold(_count(10), "count")
TWO_CHANGES = _gold(_set((0, 0, "added"), (3, 0, "removed")), "changed")
WITH_REGION = _gold(_set((0, 0, "moved"), dont_care=[(5, 5, 0.5)]), "changed")
NEVER_SEEN = _gold(UNKNOWN, "where_last", slice="unknown")
TWO_PLACES = _gold(_places((0, 0), (3, 0)), "inventory")
PLACE_REGION = _gold(_places((0, 0), dont_care=[(5, 5, 0.5)]), "inventory")

BASE_BATTERY = (
    [(_pos(e, 0), AT_ORIGIN) for e in (0, 0.25, 0.3, 0.6, 1.5, 3)]
    + [(_at(s), AT_NOON) for s in (0, 3, 5, 20, 60, 120)]
    + [(_count(n), THREE) for n in range(1, 7)]
    + [(_count(1), ONE), (_count(2), ONE), (_count(0), NO_COUNT), (_count(1), NO_COUNT)]
    + [(_count(3), FOUR), (_count(9), TEN), (_count(8), TEN), (_count(12), TEN)]
    + [(_pos(0.3, 0, 0.4), _gold(_pos(0, 0, 0), "where_last", metric="xyz")),
       (_pos(0.3, 0), _gold(_pos(0, 0, 0), "where_last", metric="xyz")),
       (_pos(3, 0), _gold(_pos(0, 0), "where_last", tolerances={"ladder": [0.5, 1.0, 2.0, 4.0]}))]
    + [(_num(v), _gold(_num(3.0), "ego_distance")) for v in (3.0, 3.25, 3.9, 1.5)]
    + [(_num(1.5), _gold(_num(3.0), "ego_distance", tolerances={"ladder": [1.0, 2.0]}))]
    + [(_yes(True), _gold(_yes(True), "moved_between")), (_yes(False), _gold(_yes(True), "moved_between"))]
    + [(_set((0, 0, "added"), (3, 0, "removed")), TWO_CHANGES),
       (_set((0.1, 0, "added"), (3, 0.1, "removed"), (8, 8, "moved")), TWO_CHANGES),
       (_set((0, 0, "removed"), (3, 0, "added")), TWO_CHANGES),
       (_set((0.5, 0, "added"), (3, 0, "removed")), TWO_CHANGES),
       (_set((0.51, 0, "added"), (3, 0, "removed")), TWO_CHANGES),
       (_set((0.9, 0, "added")), _gold(_set((0, 0, "added")), "changed", tolerances={"match_m": 1.0})),
       (_set((0, 0, "moved"), (5.3, 5, "added")), WITH_REGION), (_set((0, 0, "moved"), (6, 5, "added")), WITH_REGION),
       (_set(), _gold(_set(), "changed")), (_set((1, 1, "added")), _gold(_set(), "changed")),
       (_set(), TWO_CHANGES)]
    + [(UNKNOWN, AT_ORIGIN), (UNKNOWN, NEVER_SEEN), (_pos(0, 0), NEVER_SEEN), (_at(0), AT_ORIGIN),
       ({"type": "bogus"}, AT_ORIGIN), ({"type": "bogus"}, NEVER_SEEN), (None, AT_ORIGIN), ("0,0", AT_ORIGIN)]
)
# A list of places scored like a set without change types (exact, partial, a repeat, the radius
# and heights, a wider radius, a don't-care region, empty against empty and against not, another type either way,
# `unknown`, and an item that carries a `change`)
POSITIONS_CASES = [
    (_places((0, 0), (3, 0)), TWO_PLACES), (_places((0.1, 0), (8, 8)), TWO_PLACES),
    (_places((0, 0), (0, 0), (3, 0)), TWO_PLACES), (_places((0.5, 0, 9), (3.5, 0, 9)), TWO_PLACES),
    (_places((0.51, 0), (3, 0.51)), TWO_PLACES),
    (_places((0.9, 0)), _gold(_places((0, 0)), "inventory", tolerances={"match_m": 1.0})),
    (_places((0, 0), (5.3, 5)), PLACE_REGION), (_places((0, 0), (6, 5)), PLACE_REGION),
    (_places(), _gold(_places(), "inventory")), (_places((1, 1)), _gold(_places(), "inventory")),
    (_places(), TWO_PLACES), (_set((0, 0, "added"), (3, 0, "added")), TWO_PLACES), (_places((0, 0)), TWO_CHANGES),
    (UNKNOWN, TWO_PLACES), ({"type": "positions", "items": [{"xy": [0, 0], "change": "added"}]}, TWO_PLACES),
]


def with_qids(battery):
    return [(a, {**g, "qid": f"q{i}"}) for i, (a, g) in enumerate(battery)]


BATTERY = with_qids(BASE_BATTERY + POSITIONS_CASES)

# Credits scored by the name each gold row gives: cells (exact, partial, empty either way, a don't-care
# cell, negative cells), route (the rooms in order, a doorway point, the wrong order, a repeat, nothing), region
# (inside, near the wall, on the edge, outside), relative (exact, 25% off, exactly 50% off, a negative estimate)
ROUTE = {"type": "route", "regions": {"hall": [[0, 0], [2, 0], [2, 2], [0, 2]],
                                      "kitchen": [[1, 0], [4, 0], [4, 2], [1, 2]],
                                      "bedroom": [[5, 0], [7, 0], [7, 2], [5, 2]]},
         "valid": [["hall", "kitchen", "bedroom"], ["hall", "bedroom"]]}


def _cells(*cells, dont_care=()):
    return {"type": "cells", "items": [list(c) for c in cells],
            **({"dont_care": [list(c) for c in dont_care]} if dont_care else {})}


def _route(*points):
    return {"type": "route", "points": [list(p) for p in points]}


def _row(gold, family, credit, tolerances=None, region=None):
    return _gold(gold, family, credit=credit, region=region, tolerances=tolerances or {})


CELLS_GOLD = _row(_cells((0, 0), (1, 0), (0, 1), dont_care=[(5, 5)]), "room_cells", "f1")
ROUTE_GOLD = _row(ROUTE, "route_back", "route")
REGION_GOLD = _row(_pos(2, 2), "drop_zone", "region", {"erode_m": 0.2}, SQUARE)
RELATIVE_GOLD = _row(_num(2.0), "distance", "relative")
NEW_CREDITS = [
    (_cells((0, 0), (1, 0), (0, 1)), CELLS_GOLD), (_cells((0, 0), (1, 0), (9, 9)), CELLS_GOLD),
    (_cells((0, 0), (1, 0), (0, 1), (5, 5)), CELLS_GOLD), (_cells(), CELLS_GOLD), (_cells((-1, -1)), CELLS_GOLD),
    (_cells(), _row(_cells(), "room_cells", "f1")), (_cells((0, 0)), _row(_cells(), "room_cells", "f1")),
    (_route((0.5, 1), (3, 1), (6, 1)), ROUTE_GOLD), (_route((0.5, 1), (4.5, 1), (6, 1)), ROUTE_GOLD),
    (_route((6, 1), (0.5, 1)), ROUTE_GOLD), (_route((0.5, 1), (0.6, 1), (6, 1)), ROUTE_GOLD),
    (_route((4.5, 1)), ROUTE_GOLD), (_route((1.5, 1), (3, 1), (6, 1)), ROUTE_GOLD),
    (_pos(2, 2), REGION_GOLD), (_pos(0.1, 2), REGION_GOLD), (_pos(0.2, 2), REGION_GOLD), (_pos(-0.1, 2), REGION_GOLD),
    (_pos(2, 2, 7), REGION_GOLD), (_pos(2, 2), _row(_pos(2, 2), "drop_zone", "region", {"erode_m": 0}, SQUARE)),
    (_pos(0, 2), _row(_pos(2, 2), "drop_zone", "region", {"erode_m": 0}, SQUARE)),
    (_num(2.0), RELATIVE_GOLD), (_num(2.5), RELATIVE_GOLD), (_num(3.0), RELATIVE_GOLD), (_num(-1.0), RELATIVE_GOLD),
    (_num(10.5), _row(_num(10.0), "distance", "relative")), (UNKNOWN, RELATIVE_GOLD), (_pos(0, 0), CELLS_GOLD),
]


def credits_of(battery) -> list:
    return [[score.credit(a, g), score.penalised(a, g), score.penalised(a, g, 0.75), score.outcome(a, g)]
            for a, g in battery]


# Rows for the aggregate: the battery scored through `score.row`, the case, script, layout, job, `when` and the
# tags of each row assigned by a fixed rule of its index.
def aggregate_rows() -> list[dict]:
    rows = []
    for i, (answer, gold) in enumerate(BATTERY):
        tagged = {**gold, "scope": "now" if i % 2 else "memory", "reference": "path_log" if i % 5 == 0 else None,
                  "critical": i % 3 == 0}
        info = {"case": f"c{i % 6}", "script": f"s{i % 3}", "layout": "L1" if i % 4 else "L2", "robot": "vacuum",
                "job": "vacuum/1", "length": "short" if i % 2 else "long"}
        rows.append(score.row(answer, tagged, when="during" if i % 2 else "end", info=info))
    return rows


# Rows for the gates: 12 scripts of one case each, five families of five questions (the fifth unknown), the ten
# systems. A system answers a fixed share of the answerable questions right and of the unknown ones `unknown`, by a
# fixed pattern. `where_last` is a clean family that passes every gate; `when_last` has a memory-less system that does
# too well (memory_needed fails) and a table that is not quite perfect (knowable fails); `ego_distance` has a blind
# shortcut that works (shortcuts and separation fail) and is an own-path family, where path_log is the reference;
# `fixed_rule` is a family whose gold and systems share a rule (knowable is n/a); `layout_fact` is a layout-constant
# family, where other_case is the reference and answers it: its scripts are on a held-out layout and a public one in
# turn, so other_case is left out of the gates on the first and fails shortcuts and separation on the second.
SHARE = {"oracle": (100, 100), "memory_table": (100, 100), "current_view": (5, 50), "current_view_prior": (25, 50),
         "path_log": (45, 50), "blind": (30, 50), "other_case": (20, 50), "questionnaire_blind": (10, 50),
         "all_unknown": (0, 100), "form": (35, 50)}
INSTEAD = {("memory_table", "when_last"): (96, 100), ("current_view", "when_last"): (80, 50),
           ("blind", "ego_distance"): (85, 50), ("memory_table", "fixed_rule"): (50, 100),
           ("other_case", "layout_fact"): (100, 100)}
FAMILIES = ("where_last", "when_last", "ego_distance", "fixed_rule", "layout_fact")
REFERENCE = {"ego_distance": "path_log", "layout_fact": "other_case"}   # the family's reference shortcut, if it has one


def layout_of(e: int, family: str) -> str:
    """The layout of script `e`: L1 for the first eight and L2 for the rest, except in the layout-constant family,
    where the even scripts are on a held-out layout and the odd ones on L1."""
    if family == "layout_fact":
        return HELD_OUT[0] if e % 2 == 0 else "L1"
    return "L1" if e < 8 else "L2"


def gate_rows() -> dict[str, list[dict]]:
    out = {}
    for n, name in enumerate(SHARE):
        rows = []
        for f, family in enumerate(FAMILIES):
            answerable, unknown = INSTEAD.get((name, family), SHARE[name])
            for e in range(12):
                for q in range(5):
                    is_unknown = q == 4
                    right = (e * 37 + q * 7 + f * 11 + n * 13) % 100 < (unknown if is_unknown else answerable)
                    abstained = name == "all_unknown" and not is_unknown
                    rows.append(result_row(
                        f"e{e}", f"e{e}-{family}-{q}", 1.0 if right else 0.0 if abstained else -1.0,
                        "correct" if right else "abstained" if abstained else "wrong", script=f"e{e}", family=family,
                        when="during" if q == 0 else "end", scope="now" if q == 0 else "memory",
                        reference=REFERENCE.get(family),
                        slice_="unknown" if is_unknown else "answerable", layout=layout_of(e, family)))
        out[name] = rows
    return out


def scorer_output() -> dict:
    return {"new_credits": credits_of(NEW_CREDITS), "aggregate": score.aggregate(aggregate_rows(), 200, 0),
            "gates": gates.evaluate(gate_rows(), shared_rule=frozenset({"fixed_rule"}), n_boot=200, seed=0)}


def test_the_scorer_output_is_deterministic():
    """Contract: the scorer's output (a credit-only battery of the four newer credits, the aggregate of a fixed set of
    rows and the gates' report on fixed synthetic rows, among them a family whose reference is `other_case`, on a
    held-out layout and on a public one) is the same in a second process (another `PYTHONHASHSEED`). Regression caught:
    a result that depends on dict or set order or on the process (the aggregate's bootstrap included), so no two runs
    of the scorer agree."""
    here = digest(scorer_output())
    there = subprocess.run([sys.executable, __file__], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout
    assert json.loads(there) == here


if __name__ == "__main__":  # the second process of the test above
    print(json.dumps(digest(scorer_output())))
