"""`object_was` (questions_change): where a moved thing was during a named task, at its earlier or its later state, by
hand on `change_draw_kit`'s bags (two tasks of `vacuum_run`, 28 keyframes each; the window's last look window is
keyframes 50 to 55; see the kit for the ranges and rungs worked out by hand)."""

from __future__ import annotations

from collections import Counter

import pytest

import change_draw_kit as dk
import change_kit as ck
from change_draw_kit import moved_bags, only
from change_kit import thing
from space_kit import vacuum_run
from stbench import questionnaire
from stbench.questions_change import EARLIER, GOLDS, LATER, object_was
from stbench.questions_text import COINS, SLICED
from stbench.seeding import stable_hex


def pools(w) -> tuple[list, list]:
    """The earlier and the later candidates of a draw (every draw holds the same two pools)."""
    first = object_was(w)[0]
    return first.answerable, first.unknown


def by_state(w) -> dict[str, list[tuple]]:
    """The candidates of each pool as (entity, task asked, gold xy, evidence), sorted."""
    return {state: sorted((c.entities[0], c.slots["task"], tuple(c.gold["xy"]), c.evidence_k) for c in pool)
            for state, pool in zip((EARLIER, LATER), pools(w), strict=True)}


def test_a_moved_thing_is_a_candidate_at_the_state_whose_last_rung_its_move_passes():
    """Contract: `object_was` offers a thing at its earlier state (task i) when a place it has in a later task is beyond
    the last rung of task i's relative ladder from its place in i, and at its later state (task j) when an earlier
    task's place is beyond task j's last rung: the gap is read against the asked state's rung, not a fixed distance.
    The gold is the place in the named task and its evidence the last recognisable sighting there. Hand values on
    `moved_bags` (kit docstring): E0001 moved 2.00 m, last rung 1.5005 m in task 0 and 2.5003 m in task 1: earlier only;
    E0002 3.00 m, rungs 0.5015 and 2.0004: both; E0003 1.51 m, rungs 1.5005 and 0.7460: both; E0004 1.50 m, rungs 1.5005
    (not beyond) and 0.7510: later only. Regression caught: a fixed move threshold (1 m) in place of the asked state's
    rung (E0001 would be asked at its later state, E0004 at its earlier), a gap read against the other task's rung, or
    the wrong gold place."""
    w = ck.window(moved_bags(), 0, 1, job=only("object_was"))
    at0 = {"E0001": (5.13, 1.5), "E0002": (3.13, 1.5), "E0003": (5.13, 1.5)}
    at1 = {"E0002": (6.13, 1.5), "E0003": (3.62, 1.5), "E0004": (3.63, 1.5)}
    assert by_state(w) == {EARLIER: [(e, 0, xy, (3,)) for e, xy in at0.items()],
                           LATER: [(e, 1, xy, (31,)) for e, xy in at1.items()]}


def test_a_candidate_is_a_full_row_with_the_slant_it_was_seen_at():
    """Contract: an `object_was` candidate carries slots `kind`, `attributes` and `task` only, the position gold, the
    class of its pool (`earlier` or `later`), `range_m` the nearest recognisable slant at which the place was seen and
    the relative ladder of that range, the thing as its one subject (no named time: a task ordinal is no subject) and
    the last sighting as its evidence. Hand values: E0002 asked at the earlier state in task 0 is 1.0030 m from the
    camera, rungs 0.1003, 0.25075, 0.5015; at the later state in task 1 it is 4.0007 m, rungs 0.40007, 1.000175,
    2.00035; a bag's catalog size does not widen them. Regression caught: the job's fixed ladder on a memory question,
    a range taken from the other state's place, a named time among the subjects, or a class that does not name the
    pool."""
    w = ck.window(moved_bags(), 0, 1, job=only("object_was"))
    early, late = pools(w)
    (a,) = [c for c in early if c.entities == ("E0002",)]
    (b,) = [c for c in late if c.entities == ("E0002",)]
    assert a.slots == {"kind": "bag", "attributes": {"color": "blue"}, "task": 0}
    assert b.slots == {"kind": "bag", "attributes": {"color": "blue"}, "task": 1}
    assert (a.answer_class, b.answer_class, a.slice, a.subjects, a.critical) == (EARLIER, LATER, "answerable",
                                                                                 ("e:E0002",), True)
    assert a.range_m == pytest.approx(dk.SLANT[1.0], abs=1e-6) and b.range_m == pytest.approx(dk.SLANT[4.0], abs=1e-6)
    assert a.tolerances["ladder"] == pytest.approx(dk.RUNGS[1.0], abs=2e-6)
    assert b.tolerances["ladder"] == pytest.approx(dk.RUNGS[4.0], abs=2e-6)
    assert {c.family for c in early + late} == {"object_was"} and all(c.gold["type"] == "position" for c in early)


def test_a_large_kind_widens_the_rungs_and_the_gap_it_needs(monkeypatch):
    """Contract: the kind's widening (`rules.widening`) is added to every rung, the last one included, and so to the gap
    a move must pass. Hand-worked with the bag's catalog size replaced by 1.0 x 0.4 x 0.5 (widening 1.0 / 2 - 0.25 =
    0.25 m): E0004's rungs are 0.4002, 0.6255, 1.001 in task 1 (its last rung 0.7510 + 0.25), E0001's last rung
    in task 0 is 1.7505 m, which its 2.00 m move passes, and E0003's 1.51 m move no longer passes task 0's (1.7505 m):
    E0003 is asked at the later state only (task 1's widened rung 0.9960 m). Regression caught: a gap compared with the
    unwidened rung (E0003 and E0004 earlier), or rungs widened twice."""
    for module in ("questions_where", "questions_change"):
        monkeypatch.setattr(f"stbench.{module}.item_size", lambda kind: (1.0, 0.4, 0.5))
    w = ck.window(moved_bags(), 0, 1, job=only("object_was"))
    got = by_state(w)
    assert [e for e, *_ in got[EARLIER]] == ["E0001", "E0002"]
    assert [e for e, *_ in got[LATER]] == ["E0002", "E0003", "E0004"]
    (c,) = [c for c in pools(w)[1] if c.entities == ("E0004",)]
    assert c.tolerances["ladder"] == pytest.approx([0.4002, 0.6255, 1.001], abs=2e-6)


def test_a_move_exactly_at_the_last_rung_is_not_beyond_it(monkeypatch):
    """Contract: the gap must be strictly beyond the asked state's last rung, since an answer at the rung still earns a
    third of the credit and so a wrong-state answer would not score the penalty. Hand-worked with the ladder replaced by
    one whose last rung is 1.5 m on `moved_bags`: E0004 moved exactly 1.50 m and is asked at neither state, E0001
    (2.00 m), E0002 (3.00 m) and E0003 (1.51 m) at both. Regression caught: a gap compared with `>=` (a thing at the
    rung asked, whose other place scores a third of the credit)."""
    monkeypatch.setattr("stbench.questions_change.relative_ladder", lambda r, widen=0.0: [0.1, 0.2, 1.5])
    w = ck.window(moved_bags(), 0, 1, job=only("object_was"))
    early, late = pools(w)
    assert sorted(c.entities[0] for c in early) == ["E0001", "E0002", "E0003"]
    assert sorted(c.entities[0] for c in late) == ["E0001", "E0002", "E0003"]


def test_a_draw_is_offered_for_each_thing_the_smaller_pool_holds_and_names_no_task():
    """Contract: the moments are `draw:<j>`, as many as the smaller pool has things, each holding both pools whole, so a
    window of two tasks can ask the family. Hand values: `moved_bags` has three things in each pool, so `draw:0`,
    `draw:1`, `draw:2`; with E0001 alone moved (earlier pool only) there is no draw, with E0001 and E0004's moves only
    (one thing each way) there is one. Regression caught: moments keyed by task (a two-task window names the first
    task only at its earlier state and the second only at its later, so nothing would be asked), or a draw per candidate
    (a pool that depletes before the draws end, which tilts the share of the asked states)."""
    w = ck.window(moved_bags(), 0, 1, job=only("object_was"))
    assert [m.key for m in object_was(w)] == ["draw:0", "draw:1", "draw:2"]
    assert all((m.answerable, m.unknown) == pools(w) for m in object_was(w))
    s = vacuum_run(2)
    thing(s, "E0001", "bag", "red", [(5.13, 1.5), (7.13, 1.5)], dk.SEEN)
    assert object_was(ck.window(s, 0, 1, job=only("object_was"))) == []
    thing(s, "E0004", "bag", "purple", [(5.13, 1.5), (3.63, 1.5)], dk.SEEN)
    assert [m.key for m in object_was(ck.window(s, 0, 1, job=only("object_was")))] == ["draw:0"]


def test_a_two_task_window_asks_each_draw_in_the_pool_its_coin_gives():
    """Contract: with a window of two tasks (where the old moments, one a task, could ask nothing: the first task holds
    only earlier states and the second only later ones, both pools required), every draw is asked once, in the pool the
    hand-written coin gives it (`stable_hash(case id, family, key, "slice") % 100 < 50`: the later pool), whatever the
    case. Hand-worked on `moved_bags` over 12 case ids: three questions each, the earlier ones in task 0 and the later
    ones in task 1, and the number of later questions the coin's count over `draw:0` to `draw:2`. Regression caught: a
    coin family not registered as one (the family would draw the 20% unknown slice), draws that block each other through
    the subject rule, or a pool taken that the coin did not give."""
    for n in range(12):
        cid = stable_hex("was", n, n=16)
        asked = questionnaire.end(ck.window(moved_bags(), 0, 1, cid=cid, job=only("object_was")))
        late = sum(dk.hand_coin(cid, "object_was", f"draw:{j}") for j in range(3))
        assert len(asked) == 3 and len({c.entities for c in asked}) == 3, cid
        assert Counter(c.answer_class for c in asked) == {k: v for k, v in
                                                          ((EARLIER, 3 - late), (LATER, late)) if v}, cid
        assert all(c.slots["task"] == (0 if c.answer_class == EARLIER else 1) for c in asked)


def test_the_asked_state_is_a_fair_coin_over_two_thousand_cases():
    """Contract: the state of an `object_was` draw is the questionnaire's coin, so over 2,000 synthetic case ids the
    later share of `draw:0` is within 0.03 of one half (the binomial standard deviation is 0.011), and the family is a
    coin family, not a sliced one (whose unknown share would be 0.20). Hand bound, not a computed share. Regression
    caught: `object_was` left out of `COINS` (a 20% second pool, so the earlier state is asked four times in five and a
    notebook of first sightings scores), or a draw key that decides the coin by the case alone."""
    assert "object_was" in COINS and "object_was" not in SLICED and "object_was" in GOLDS
    ids = [stable_hex("share", i, n=16) for i in range(2000)]
    share = sum(questionnaire.unknown_slice(cid, "object_was", "draw:0") for cid in ids) / len(ids)
    assert abs(share - 0.5) <= 0.03


def test_a_thing_in_a_task_between_two_places_is_asked_at_both_states():
    """Contract: in a window of three tasks a thing has an earlier candidate in each task that some later place is
    beyond the rung of, and a later candidate in each task that some earlier place is beyond the rung of, so a middle
    task can be both. Hand values: a red bag at x = 5.13, 7.13 and 3.13 (y 1.5) in tasks 0, 1 and 2, seen from the
    camera at x = 2.13 at keyframes 3 (offset 3 of task 0) and 59 (task 2) and from the one at x = 6.13 at keyframe 35
    (offset 7 of task 1; 1.0 m from 7.13). Ranges 3.0010, 1.0030 and 1.0030 m, last rungs 1.5005, 0.5015 and 0.5015 m;
    the places are 2.00 m (tasks 0-1), 4.00 m (1-2) and 2.00 m (0-2) apart, all beyond: earlier in tasks 0 and 1, later
    in tasks 1 and 2. Regression caught: a middle task left out of either pool, or a task compared only with its
    neighbour."""
    s = vacuum_run(3)
    thing(s, "E0001", "bag", "red", [(5.13, 1.5), (7.13, 1.5), (3.13, 1.5)], [3, 35, 59])
    w = ck.window(s, 0, 2, job=only("object_was"))
    early, late = pools(w)
    assert [(c.slots["task"], c.range_m) for c in early] == [(0, pytest.approx(3.001, abs=1e-6)),
                                                              (1, pytest.approx(1.0, abs=0.01))]
    assert sorted(c.slots["task"] for c in late) == [1, 2]
    assert {c.gold["xy"][0] for c in early} == {5.13, 7.13} and {c.gold["xy"][0] for c in late} == {7.13, 3.13}


def test_only_recognisable_places_in_two_tasks_make_a_thing_moved():
    """Contract: a place counts when it is a recognisable sighting, and a thing in the window's last look window is
    never asked. Hand-worked on a blue bag that moves 3.00 m (asked at both states, so every window has a draw) and a
    red bag 5.13 -> 7.13 (a 2.00 m move that passes task 0's last rung, 1.5005 m) whose task-1 sighting is varied:
    recognisable at keyframe 31, then the red bag is asked at its earlier state; a 500 px glimpse only, or dark
    (luminance 0.005) only: no place in task 1, so no move; recognisable and also in the last look window (keyframe 52):
    not asked at all. Regression caught: a glimpse or a dark frame taken as the place that makes a thing moved, or a
    thing in view asked."""
    def red_asked(sightings) -> bool:
        s = vacuum_run(2)
        thing(s, "E0002", "bag", "blue", [(3.13, 1.5), (6.13, 1.5)], dk.SEEN)
        thing(s, "E0001", "bag", "red", [(5.13, 1.5), (7.13, 1.5)], [3])
        for k, px, lum in sightings:
            s.see(k, "E0001", px, lum=lum)
        return "E0001" in {c.entities[0] for c in pools(ck.window(s, 0, 1, job=only("object_was")))[0]}

    assert red_asked([(31, 1500, None)])
    assert not red_asked([(31, 500, None)]) and not red_asked([(31, 1500, ck.DARK)])
    assert not red_asked([(31, 1500, None), (52, 200, None)])


def test_a_place_that_a_later_sighting_in_the_same_task_contradicts_is_no_place():
    """Contract: a thing's place in a task is its last recognisable sighting only when every sighting from it to the
    task's end stays within 0.15 m (`World.steady`), so the memory table, which reads the latest sighting, holds the
    place the gold reads. Hand-worked on a red bag that moves at t = 10 s (inside task 0, whose keyframes run to t =
    40.5 s) from x = 5.13 to 7.13 and is recognisable at keyframe 3 (the old place) and at 31 (task 1, the new one),
    beside the blue bag that moves 3.00 m between the tasks: with no other sighting the red bag's place in task 0 is
    (5.13, 1.5) and it is asked at its earlier state; with a 500 px glimpse of it at keyframe 20, 2.00 m away at the new
    place, task 0 has no place (the glimpse contradicts the sighting), so the red bag has one place and is asked at no
    state. Regression caught: a gold the latest-sighting table cannot reproduce (the oracle under the ceiling), or a
    glimpse ignored."""
    def red_asked(glimpse: bool) -> bool:
        s = vacuum_run(2)
        thing(s, "E0002", "bag", "blue", [(3.13, 1.5), (6.13, 1.5)], dk.SEEN)
        thing(s, "E0001", "bag", "red", [(5.13, 1.5), (7.13, 1.5)], [3, 31], gaps=(10.0,))
        if glimpse:
            s.see(20, "E0001", 500)
        return "E0001" in {c.entities[0] for c in pools(ck.window(s, 0, 1, job=only("object_was")))[0]}

    assert red_asked(False) and not red_asked(True)
