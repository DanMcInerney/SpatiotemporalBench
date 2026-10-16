"""`object_where` and the things that move (questions_where, questions_world): a thing that moved in the window is left
to the change families, whose question names its earlier state; `object_where` asks neither a place that is stale nor
the latest state of a thing that moved, while `answerable_at` stays the raw same-task rule they share. Hand-worked on
`test_questions_where.moved_bag` (three tasks of 28 keyframes; a red bag at (1, 1) and, from the gap, at (to, 1))."""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from space_kit import record, vacuum_run
from stbench import questionnaire
from stbench.questions_where import LastSeen, answerable_at, moments, pools
from stbench.questions_world import World
from test_questions_where import BAG, moved_bag


def bag_of(w):
    return next(e for e in w.tl.entities if e.id == "E0001")


def asked(w) -> bool:
    """The bag is among `object_where`'s answerable candidates."""
    answerable, _ = pools(w)
    return any("e:E0001" in c.subjects for c in answerable)


def test_a_moved_thing_is_not_asked_in_either_state():
    """Contract: `object_where` asks no thing that moved in the window by `STALE_M` (1 m) and was seen at both places:
    not at its earlier state, whose place a later recognisable sighting shows out of date (`World.stale`; that question
    is `object_was`'s), and not at its latest state after the move (`World.arrived`), which a notebook of latest
    sightings answers; `answerable_at` still gives both raw. Hand-worked on the bag at (1, 1) in task 0 and (2, 1) from
    the gap, recognisable at keyframes 1, 29 and 57: at 'began sweeping' of task 0 (7) the raw answer is (1, 1) from 1,
    stale, and of task 1 (35) (2, 1) from 29, a place it arrived at; its last recognisable sighting is 57, at (2, 1),
    which it arrived at, so it is not asked. The same bag moved 0.8 m (to (1.8, 1)) is asked, and so is one that stays.
    Regression caught: a moved thing spent on a question a notebook answers, which leaves the change families (one
    question to a thing) no thing that moved."""
    w = fk.world(moved_bag([1, 29, 57]), 0, 2, "c1", "medium")
    bag = bag_of(w)
    assert answerable_at(w, (0, 7), bag) == LastSeen((1.0, 1.0), 1)
    assert answerable_at(w, (1, 35), bag) == LastSeen((2.0, 1.0), 29)
    assert w.stale(bag, 1, (1.0, 1.0)) and w.arrived(bag, 57, (2.0, 1.0))
    assert not asked(w)
    assert asked(fk.world(moved_bag([1, 29, 57], to=1.8), 0, 2, "c1", "medium"))
    still = vacuum_run()
    still.item("E0001", "bag", {"color": "red"}, [(0, None, (1.0, 1.0, 0.0))], size=BAG)
    still.see([1, 10], "E0001", 1500)
    assert asked(fk.world(still))


def test_a_thing_seen_only_before_it_moved_is_asked_where_it_was_last_seen():
    """Contract: a thing whose recognisable sightings are all at one place is asked there, whatever happened to it
    after: nothing unobserved changes a gold (truth rule 2), so a bag that moved 1.5 m in the gap after its last
    recognisable sighting is asked at the place it was seen. Hand-worked on the three-task window with the bag moved to
    (2.5, 1) and recognisable at keyframe 1 only: the gold is (1, 1) from 1. Regression caught: a moved thing dropped
    because of a move nobody saw, which tells the question's form from the world."""
    (c,), _ = pools(fk.world(moved_bag([1], to=2.5), 0, 2, "c1", "medium"))
    assert (c.gold, c.evidence_k) == ({"type": "position", "xy": [1.0, 1.0]}, (1,))


def test_a_large_thing_must_move_further_to_be_left_to_the_change_families(monkeypatch):
    """Contract: how far a thing must move to be left out of `object_where` is `STALE_M` plus the kind's widening, the
    same distance `World.stale` uses. Hand-worked with the bag a kind 1.0 x 0.4 x 0.5 m (widening 0.25, so 1.25 m): the
    bag moved 1 m is still asked, moved 1.25 m it is not. Regression caught: a large thing called moved by the extent
    of its own surface."""
    monkeypatch.setattr("stbench.questions_world.item_size", lambda kind: (1.0, 0.4, 0.5))
    for to, named in ((2.0, True), (2.25, False)):
        w = fk.world(moved_bag([1, 29, 57], to=to), 0, 2, "c1", "medium")
        assert asked(w) == named


def test_no_end_question_names_a_moved_thing():
    """Contract: through the questionnaire a thing that moved in the window is named by no `object_where` candidate and
    no end question, whatever the case id, while the same window with the bag moved 0.8 m names it. Hand-worked on the
    three-task window over 20 case ids: the candidates and the asked questions of the 1 m bag never carry `e:E0001`;
    those of the 0.8 m bag do, in some case. Regression caught: a moved thing back in the questionnaire by a path
    that skips `pools`."""
    for to, named in ((2.0, False), (1.8, True)):
        s = moved_bag([1, 29, 57], to=to)
        inp, rec = fk.inputs(s), record(s, a=0, b=2, length="medium")
        found = 0
        for i in range(20):
            w = World(inp, replace(rec, case=f"{i:016x}"))
            in_candidates = any("e:E0001" in c.subjects for m in moments(w) for c in m.answerable + m.unknown)
            in_questions = any("e:E0001" in c.subjects for c in questionnaire.end(w))
            assert in_candidates == named
            found += in_questions
        assert (found > 0) == named


def test_a_move_seen_only_in_unrecognisable_frames_leaves_the_thing_unasked():
    """Contract: a thing with a later sighting, of any pixel count, 0.15 m (`MOVED_NO_M`) or more from its last
    recognisable place is not asked, since the memory table reads the latest sighting and would hold it elsewhere than
    the gold (the 13 m the AMR's pallets moved in release-b's gaps, glimpsed there and never recognisably seen). By hand
    on the three-task window with the bag recognisable at keyframe 1 only, at (1, 1), and moved to (to, 1) in the gap: a
    500 px glimpse at keyframe 29 at the new place leaves it unasked when it moved 1.5 m or 0.15 m, and asked (gold (1,
    1) from 1) when it moved 0.1 m; the same bag with no glimpse is asked, moved or not. Regression caught: a gold the
    oracle's own table cannot reproduce (the 6% of release-b's answerable rows, 12% of the AMR's, that a later glimpse
    puts elsewhere), which sinks the ceiling gate."""
    def asked_with(to, glimpse):
        s = moved_bag([1], to=to)
        if glimpse:
            s.see(29, "E0001", 500)
        return asked(fk.world(s, 0, 2, "c1", "medium"))

    assert not asked_with(2.5, glimpse=True) and not asked_with(1.15, glimpse=True)
    assert asked_with(1.1, glimpse=True) and asked_with(1.0, glimpse=True) and asked_with(2.5, glimpse=False)
