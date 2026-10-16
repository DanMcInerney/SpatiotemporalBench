"""The `tolerances` each credit's gold row holds (`records.CREDIT_KEYS`, 0.9.0): the graded credits hold several keys,
each of which `check_gold` judges, so the scorer never reads a default from a row."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest

from stbench.records import BALANCE_POOLS, CREDIT_KEYS, CREDITS, FamilySpec, check_family, check_gold, pool_of

GRADED_OFFSET = {"resolution_m": 0.05, "need_m": 0.15, "harm": "radial"}
GRADED_NUMBER = {"resolution_m": 0.02, "need_m": 0.15, "harm": "over", "threshold_m": 0.49}
GRADED_POSITION = {"resolution_m": 0.02, "need_m": 0.5, "harm": None}
GRADED_SET = {"match_m": 0.5, "resolutions_m": [0.02, 0.05], "fp_weight": 0.5}
REGRET = {"profile_m": [1.5] * 72, "step_deg": 5, "resolution_m": 0.02, "need_m": 0.15, "threshold_m": 0.49}
ROUTE = {"profile_m": [0.5, -0.2, None, 0.3] * 18, "step_deg": 5, "step_m": 1.0, "resolution_m": 0.1, "need_m": 0.15}
# (credit, gold, tolerances) of one valid row each
ROWS = {
    ("graded", "offset"): ("graded", {"type": "offset", "ahead_m": 1.0, "left_m": 0.5}, GRADED_OFFSET),
    ("graded", "number"): ("graded", {"type": "number", "value": 1.4}, GRADED_NUMBER),
    ("graded", "position"): ("graded", {"type": "position", "xy": [3.2, 1.5]}, GRADED_POSITION),
    ("graded_set", "positions"): ("graded_set", {"type": "positions", "items": [{"xy": [1, 2]}, {"xy": [3, 2]}]},
                                  GRADED_SET),
    ("regret", "heading"): ("regret", {"type": "heading", "deg_left": 30.0}, REGRET),
    ("route", "heading"): ("route", {"type": "heading", "deg_left": 30.0}, ROUTE),
}


def row(credit: str, gold: dict, tolerances: dict, **over) -> dict:
    """A gold row of an `object_where`-like family scored by `credit`."""
    base = {"qid": "fedcba9876543210", "family": "object_where", "gold": gold, "slice": "answerable",
            "answer_class": None, "range_m": None, "credit": credit, "region": None, "tolerances": tolerances,
            "metric": "xy" if gold["type"] == "position" else None, "scope": "memory", "reference": None,
            "critical": False, "subjects": ["e:E0042"], "evidence_k": [12], "entities": ["E0042"]}
    return {**base, **over}


def valid(key: tuple[str, str]) -> dict:
    credit, gold, tolerances = ROWS[key]
    return row(credit, copy.deepcopy(gold), copy.deepcopy(tolerances))


def test_the_schema_names_the_keys_by_hand():
    """Contract (plan I0): `graded` holds the resolution, the need and the harmful side, and for a number the threshold;
    a position holds no room (the direction quarter of a place is `e <= R` alone); `graded_set` the match radius, one
    resolution per gold item and the false item's weight; `regret` the profile, its step and the best run's resolution,
    need and threshold; the old credits keep their one key. Regression caught: a key added to a credit without the
    validators learning it, or a position row that still carries the room the scorer no longer reads."""
    assert CREDIT_KEYS["graded"] == {"offset": ("resolution_m", "need_m", "harm"),
                                     "number": ("resolution_m", "need_m", "harm", "threshold_m"),
                                     "position": ("resolution_m", "need_m", "harm")}
    assert CREDIT_KEYS["graded_set"] == {"positions": ("match_m", "resolutions_m", "fp_weight")}
    assert CREDIT_KEYS["regret"] == {"heading": ("profile_m", "step_deg", "resolution_m", "need_m", "threshold_m")}
    assert CREDIT_KEYS["route"] == {"route": (),
                                    "heading": ("profile_m", "step_deg", "step_m", "resolution_m", "need_m")}
    assert CREDIT_KEYS["ladder"]["number"] == ("ladder",) and CREDIT_KEYS["f1"]["cells"] == ()
    assert CREDITS[-3:] == ("graded", "graded_set", "regret")


@pytest.mark.parametrize("key", list(ROWS), ids=lambda k: "/".join(k))
def test_a_row_with_exactly_its_keys_is_valid(key):
    """Contract: a gold row of each graded credit with exactly the keys the schema names, valid values, passes.
    Regression caught: validators that refuse the rows the compiler writes (every refusal below would then pass
    vacuously)."""
    assert check_gold(valid(key)) == []


@pytest.mark.parametrize("key", list(ROWS), ids=lambda k: "/".join(k))
def test_a_missing_or_an_extra_key_is_refused_naming_it(key):
    """Contract: a row with a key missing or one more than its credit reads is refused, and the message names the key.
    Regression caught: a gold row the scorer cannot read, or reads a default from."""
    for name in CREDIT_KEYS[key[0]][key[1]]:
        gold_row = valid(key)
        del gold_row["tolerances"][name]
        assert any(name in p for p in check_gold(gold_row)), name
    gold_row = valid(key)
    gold_row["tolerances"]["radius_m"] = 1.0
    assert any("radius_m" in p for p in check_gold(gold_row))
    gold_row["tolerances"] = {"ladder": [0.1, 0.5]}   # another credit's key, with the right ones missing
    assert any("tolerances" in p for p in check_gold(gold_row))


@pytest.mark.parametrize("name", ["resolution_m", "need_m", "threshold_m", "step_m"])
@pytest.mark.parametrize("bad", [0, -0.1, math.nan, math.inf, True, "0.1", None])
def test_a_non_positive_number_is_refused_naming_its_key(name, bad):
    """Contract: the resolution, the need and the threshold are numbers above 0 (a need of 0 divides, a resolution of
    0 is an exact answer no camera gives). Regression caught: a zero or negative need or resolution reaching the
    scorer's logarithm."""
    gold_row = valid(("graded", "number"))
    gold_row["tolerances"][name] = bad
    assert any(f"tolerances.{name}" in p for p in check_gold(gold_row))
    for key in (("regret", "heading"), ("route", "heading")):
        if name in CREDIT_KEYS[key[0]][key[1]]:
            gold_row = valid(key)
            gold_row["tolerances"][name] = bad
            assert any(f"tolerances.{name}" in p for p in check_gold(gold_row)), key


def test_the_harmful_side_fits_the_golds_type():
    """Contract (brief §3.3): a number is harmful when answered long (`over`), an offset when placed far (`radial`), a
    place never; `null` means no harmful side on any type. Regression caught: a harm rule of the wrong sign for a type,
    or a harmful place."""
    for key, harms in ((("graded", "number"), ("over", None)), (("graded", "offset"), ("radial", None)),
                       (("graded", "position"), (None,))):
        for harm in ("over", "radial", None, "under"):
            gold_row = valid(key)
            gold_row["tolerances"]["harm"] = harm
            assert (check_gold(gold_row) == []) is (harm in harms), (key, harm)


def test_a_position_row_that_still_holds_a_room_is_refused():
    """Contract (0.9.0 review B2): a graded position row holds exactly `resolution_m`, `need_m` and `harm`; a `room`,
    which the scorer no longer reads, is an extra key and is refused naming it, so no row of an older compile is read
    as if its room still earned a quarter. Regression caught: a room left on the rows of a recompiled dataset."""
    gold_row = valid(("graded", "position"))
    gold_row["tolerances"]["room"] = [[0, 0], [4, 0], [4, 3], [0, 3]]
    assert any("room" in p for p in check_gold(gold_row))


def test_regrets_profile_matches_its_step():
    """Contract: `profile_m` holds `360 / step_deg` entries (numbers of at least 0 or nulls for a heading never seen)
    and `step_deg` divides 360. Hand cases: 72 entries at 5 degrees, 360 at 1, 36 at 10; 360 entries at 5 is wrong, so
    is 71, 7 degrees, 0, 2.5 and a profile of strings. Regression caught: a profile the scorer would index out of
    range, or read at the wrong heading."""
    for step, n, ok in ((5, 72, True), (1, 360, True), (10, 36, True), (5, 360, False), (5, 71, False),
                        (7, 51, False), (0, 72, False), (2.5, 144, False), (-5, 72, False)):
        gold_row = valid(("regret", "heading"))
        gold_row["tolerances"] |= {"step_deg": step, "profile_m": [1.0] * n}
        assert (check_gold(gold_row) == []) is ok, (step, n)
    gold_row = valid(("regret", "heading"))
    gold_row["tolerances"]["profile_m"] = [None] * 71 + [2.0]
    assert check_gold(gold_row) == []   # a heading never seen is a null
    gold_row["tolerances"]["profile_m"] = [-1.0] + [1.0] * 71
    assert any("profile_m" in p for p in check_gold(gold_row))
    gold_row["tolerances"]["profile_m"] = ["1"] * 72
    assert any("profile_m" in p for p in check_gold(gold_row))


def test_a_routes_outcomes_are_any_number_or_null_and_a_regrets_runs_are_not_negative():
    """Contract (FD-0.11 §3.4.2, §4.2): `route`'s `profile_m` holds each heading's outcome, how much nearer a step of
    driving brings the robot, which is negative for a heading that leads away (-0.2 is a valid entry) and null for a
    heading with no room or into space never seen; it holds `360 / step_deg` entries. `regret`'s holds runs, so a
    negative entry is refused there. Hand cases: 72 entries at 5 degrees pass on both credits when none is negative;
    one entry of -0.2 passes `route` and fails `regret`; 71 entries fail `route`. Regression caught: a route row whose
    outcome the validator calls a run, which would drop every heading that leads away from the thing."""
    route_row, regret_row = valid(("route", "heading")), valid(("regret", "heading"))
    assert check_gold(route_row) == [] and -0.2 in route_row["tolerances"]["profile_m"]
    regret_row["tolerances"]["profile_m"] = [-0.2] + [1.0] * 71
    assert any("profile_m" in p and "at least 0" in p for p in check_gold(regret_row))
    route_row["tolerances"]["profile_m"] = [None] + [-0.2] * 69 + [0.4]   # 71 entries
    assert any("profile_m" in p and "72 entries" in p for p in check_gold(route_row))
    route_row["tolerances"]["profile_m"] = [None] + [-0.2] * 69 + [0.4, 0.0]   # 72 entries
    assert check_gold(route_row) == []


def test_a_graded_sets_resolutions_are_one_per_gold_item():
    """Contract: `resolutions_m` holds a positive number for each item of the gold list, in order, and `fp_weight` is
    above 0 and at most 1. Hand cases: a two-item gold with two resolutions passes, with one or three is refused; an
    empty gold with an empty list passes. Regression caught: a resolution missing for an item, which the scorer would
    index past."""
    for resolutions, ok in (([0.02, 0.05], True), ([0.02], False), ([0.02, 0.05, 0.1], False), ([0.02, 0.0], False)):
        gold_row = valid(("graded_set", "positions"))
        gold_row["tolerances"]["resolutions_m"] = resolutions
        assert (check_gold(gold_row) == []) is ok, resolutions
    for weight, ok in ((0.5, True), (1, True), (0, False), (1.5, False), (-0.5, False)):
        gold_row = valid(("graded_set", "positions"))
        gold_row["tolerances"]["fp_weight"] = weight
        assert (check_gold(gold_row) == []) is ok, weight
    empty = valid(("graded_set", "positions"))
    empty["gold"], empty["tolerances"]["resolutions_m"] = {"type": "positions", "items": []}, []
    assert check_gold(empty) == []


def test_an_old_credit_keeps_its_one_key_and_an_unknown_gold_needs_none():
    """Contract: `ladder` still holds exactly `ladder` (a graded key beside it is refused), and an `unknown` gold, which
    has no answer to score, passes whatever graded credit its family names with empty tolerances. Regression caught: the
    schema loosened for the old credits, or the unknown rows of the graded families refused."""
    ladder = row("ladder", {"type": "number", "value": 1.4}, {"ladder": [0.25, 0.5, 1, 2]})
    assert check_gold(ladder) == []
    ladder["tolerances"]["need_m"] = 0.15
    assert any("need_m" in p for p in check_gold(ladder))
    for credit in ("graded", "graded_set", "regret"):
        unknown = row(credit, {"type": "unknown"}, {}, slice="unknown")
        assert check_gold(unknown) == [], credit


def test_a_graded_credit_fits_only_its_gold_types():
    """Contract: `graded` scores an offset, a number or a position, `graded_set` positions, `regret` a heading; a family
    that names a pair outside it is refused. Regression caught: a family declared `regret` on a number, whose every gold
    the scorer would raise on."""
    def spec(**over) -> FamilySpec:
        base = FamilySpec("fam", "during", "number", "ladder", ("see", "pose"), "now", None, False, (), str)
        return replace(base, **over)

    for credit, kinds in (("graded", {"offset", "number", "position"}), ("graded_set", {"positions"}),
                          ("regret", {"heading"}), ("route", {"route", "heading"})):
        for kind in ("offset", "number", "position", "positions", "heading", "set", "count"):
            problems = check_family(spec(credit=credit, answer_type=kind))
            assert (problems == []) is (kind in kinds), (credit, kind)


def test_way_to_balances_answerable_against_unknown_as_ego_mem_does():
    """Contract (FD-0.11 §4.2): `way_to`, a named thing recalled, draws the `unknown` slice as `ego_mem` does, so it has
    the two pools `answerable` and `unknown` that the balance rule reads. Regression caught: a sliced family with no
    pool, whose constant answers the balance rule could never test."""
    assert BALANCE_POOLS["way_to"] == BALANCE_POOLS["ego_mem"]
    assert pool_of("way_to", "answerable") == "answerable" and pool_of("way_to", "unknown") == "unknown"


def test_open_heading_has_one_balance_pool():
    """Contract (plan I0): `open_heading` declares one pool, `answerable`: it asks no `unknown`, so there is nothing to
    balance between. Regression caught: a second pool declared for a family whose draw has one class."""
    assert BALANCE_POOLS["open_heading"] == {"answerable": {"answerable"}}
    assert pool_of("open_heading", "answerable") == "answerable" and pool_of("open_heading", "unknown") is None
