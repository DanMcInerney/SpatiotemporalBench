"""The look points' slot draw (`questions_looks.slot_family`, `questionnaire.looks`, 0.8.0 lane L0), worked by hand on
the vacuum's task (candidate look frames 4 to 26, up to 4 points: `looks_kit`) with fake look families registered for
one test."""

from __future__ import annotations

import pytest

from family_kit import fake_cand, look_moment
from looks_kit import FRAMES, hand_points, look_job, register_look, vacuum_run, world
from stbench import questionnaire, questions_looks
from stbench.seeding import stable_hash, stable_hex


def ids(tag: str, n: int) -> list[str]:
    return [stable_hex(tag, i, n=16) for i in range(n)]


def gold_of(name: str, dead=lambda k: False):
    """A look family with one answerable candidate a point (a subject of its own), none where `dead(k)`."""
    def look(w, k):
        if dead(k):
            return None
        return look_moment(k, [fake_cand(name, f"k{k}", (f"r:{name}{k}",), decision_k=k)])
    return look


def hand_slot(cid: str, k: int, i: int, shares) -> str | None:
    """The draw by hand: the family whose cumulative percent range holds the slot's hash, None past the total."""
    h, top = stable_hash(cid, k, f"slot{i}") % 100, 0
    for family, percent in shares:
        top += percent
        if h < top:
            return family
    return None


def asked(w) -> list[tuple[str, int]]:
    return [(c.family, c.decision_k) for c in questionnaire.looks(w, [])]


def test_the_cumulative_ranges_of_a_slot_by_hand(monkeypatch):
    """Contract (plan L0, the slot draw): a slot's family is the one whose cumulative percent range holds the hash,
    and none when the hash falls past the total. Hand-worked on the slot (a 30, b 30) with the hash stubbed to the
    edges: 0 and 29 are a, 30 and 59 are b, 60 and 99 are none. Regression caught: ranges that overlap or leave a
    gap, an off-by-one at a boundary, or the rest of a slot read as its last family."""
    shares = (("a", 30), ("b", 30))
    for value, want in ((0, "a"), (29, "a"), (30, "b"), (59, "b"), (60, None), (99, None)):
        monkeypatch.setattr(questions_looks, "stable_hash", lambda *parts, v=value: 7 * 100 + v)
        assert questions_looks.slot_family("c", 1, 0, shares) == want, value


def test_the_slot_hash_is_keyed_on_the_case_the_point_and_the_slot():
    """Contract: the hash is `stable_hash(case id, k, f"slot{i}")`, so a family is drawn per (case, point, slot), the
    two slots of one point are drawn independently, and the same case redraws the same family. Hand-worked: over 200
    case ids and five look frames, `slot_family` equals the hand draw on a (a 50, b 50) slot, in both slots,
    and the two slots disagree at some point. Regression caught: a draw keyed on the point alone (every case draws
    alike), or both slots sharing one hash (they would always agree)."""
    shares = (("a", 50), ("b", 50))
    disagree = 0
    for cid in ids("keyed", 200):
        for k in FRAMES[:5]:
            drawn = [questions_looks.slot_family(cid, k, i, shares) for i in (0, 1)]
            assert drawn == [hand_slot(cid, k, i, shares) for i in (0, 1)]
            disagree += drawn[0] != drawn[1]
    assert disagree > 0


def test_the_drawn_family_does_not_depend_on_what_is_in_view(monkeypatch):
    """Contract (plan L0, design §8.5): which family a point draws is the hash's, never the world's. Hand-worked on the
    slot (a 50, b 50): in a world where both families have a candidate everywhere, the asked list is the points in
    order with the hand-drawn family at each; in a world where `a` has a candidate at odd frames only and `b` at every
    frame but the multiples of 3, the asked list is that list less the points where the drawn family has none (no
    other family takes its place). Regression caught: a draw keyed on a candidate or a gold (the family would follow
    what is in view), or the draw made over the families that have a candidate."""
    shares = (("a", 50), ("b", 50))
    job = look_job("a", "b", look_shares=(shares,))
    for cid in ids("view", 40):
        want = [(hand_slot(cid, k, 0, shares), k) for k in hand_points(cid)]
        register_look(monkeypatch, gold_of("a"), "a")
        register_look(monkeypatch, gold_of("b"), "b")
        assert asked(world(vacuum_run(), cid, job=job)) == [(f, k) for f, k in want if f is not None]
        register_look(monkeypatch, gold_of("a", dead=lambda k: k % 2 == 0), "a")
        register_look(monkeypatch, gold_of("b", dead=lambda k: k % 3 == 0), "b")
        dead = {"a": lambda k: k % 2 == 0, "b": lambda k: k % 3 == 0}
        assert asked(world(vacuum_run(), cid, job=job)) == [(f, k) for f, k in want
                                                             if f is not None and not dead[f](k)]


def test_a_table_of_none_asks_exactly_0_7_0s_looks(monkeypatch):
    """Contract (plan L0): a job with `look_shares=None` asks what 0.7.0 asked: every family of `looks`, in the job's
    order, at each look point, and the explicit table that spells the same (one slot a family at 100) asks the same
    questions (qids aside: same keys, families and points). Hand-worked on `fake_b` then `fake_a` over the vacuum's
    task with `fake_a` dead at the even frames. Regression caught: a default table that draws (a family dropped at
    some points), reorders the families, or changes the questions."""
    register_look(monkeypatch, gold_of("fake_a", dead=lambda k: k % 2 == 0), "fake_a")
    register_look(monkeypatch, gold_of("fake_b"), "fake_b")
    explicit = ((("fake_b", 100),), (("fake_a", 100),))
    for cid in ids("default", 30):
        plain = world(vacuum_run(), cid, job=look_job("fake_b", "fake_a"))
        spelled = world(vacuum_run(), cid, job=look_job("fake_b", "fake_a", look_shares=explicit))
        want = [(f, k) for k in hand_points(cid) for f in ("fake_b", "fake_a")
                if not (f == "fake_a" and k % 2 == 0)]
        assert asked(plain) == want
        assert [c.key for c in questionnaire.looks(plain, [])] == [c.key for c in questionnaire.looks(spelled, [])]
    assert look_job("fake_b", "fake_a").look_slots == explicit


def test_a_point_asks_at_most_one_question_a_slot(monkeypatch):
    """Contract (plan L0): a slot asks one question at a point, whichever family it draws, and a point asks at most
    as many questions as the table has slots. Hand-worked: three families each with a candidate everywhere, a table of
    two slots, (a 40, b 30, c 30) and (a 50, b 50): over 60 case ids no point is asked more than twice, each point's
    questions are those of its slots' hand draws (a family asked twice at a point only when both slots drew it, which
    the subject rule then refuses: its second candidate would name the same subject), and the point's questions are
    in slot order. Regression caught: every family of a slot asking, or a slot asking twice."""
    slots = ((("a", 40), ("b", 30), ("c", 30)), (("a", 50), ("b", 50)))
    for name in "abc":
        register_look(monkeypatch, gold_of(name), name)
    for cid in ids("once", 60):
        w = world(vacuum_run(), cid, job=look_job("a", "b", "c", look_shares=slots))
        got = asked(w)
        for k in hand_points(cid):
            drawn = [hand_slot(cid, k, i, shares) for i, shares in enumerate(slots)]
            want = [f for f in drawn if f is not None]
            if len(want) == 2 and want[0] == want[1]:
                want = want[:1]   # the second draw names the same subject r:<name><k>: refused
            assert [f for f, kk in got if kk == k] == want


def test_a_table_of_three_slots_asks_a_question_a_slot_in_slot_order(monkeypatch):
    """Contract (FD-questions-0.10 §3): the slot machinery takes any number of slots, so a table of three (the harm
    families, the situation and memory families, the rest) draws one family a slot at a point, each slot by its own
    hash (`slot0`, `slot1`, `slot2`), and asks in slot order. Hand-worked on three families with a candidate everywhere,
    each with a subject of its own per point: the table (a 100) / (b 100) / (c 100) asks a, b, c at every point; the
    table (a 50, b 50) / (b 40, c 40) / (a 30, b 30, c 30) asks, at each point, the families of the three hand draws in
    slot order, a family drawn twice once (its second candidate names the same subject, which the subject rule
    refuses).
    Regression caught: a third slot never drawn, drawn from slot 1's hash (it would repeat slot 1's family), or asked
    out of slot order."""
    for name in "abc":
        register_look(monkeypatch, gold_of(name), name)
    straight = ((("a", 100),), (("b", 100),), (("c", 100),))
    mixed = ((("a", 50), ("b", 50)), (("b", 40), ("c", 40)), (("a", 30), ("b", 30), ("c", 30)))
    third_differs = 0
    for cid in ids("three", 40):
        assert asked(world(vacuum_run(), cid, job=look_job("a", "b", "c", look_shares=straight))) == [
            (f, k) for k in hand_points(cid) for f in "abc"]
        got = asked(world(vacuum_run(), cid, job=look_job("a", "b", "c", look_shares=mixed)))
        for k in hand_points(cid):
            drawn = [hand_slot(cid, k, i, shares) for i, shares in enumerate(mixed)]
            assert [f for f, kk in got if kk == k] == list(dict.fromkeys(f for f in drawn if f is not None))
            third_differs += drawn[2] is not None and drawn[2] != drawn[1]
    assert third_differs > 0


def test_a_drawn_family_with_no_candidate_asks_nothing_and_the_slot_does_not_refill(monkeypatch):
    """Contract (plan L0): when the drawn family has no candidate at the point the slot asks nothing: the slot's other
    families are not tried. Hand-worked on the slot (a 50, b 50) with `a` never having a candidate: the asked list is
    exactly the points whose hash draws `b`, and at the points that draw `a` nothing is asked although `b` has a
    candidate there. Regression caught: refilling dead bands from the slot's other families, which raises the counts
    and lets the asking reveal what is in view."""
    shares = (("a", 50), ("b", 50))
    register_look(monkeypatch, lambda w, k: None, "a")
    register_look(monkeypatch, gold_of("b"), "b")
    refused = 0
    for cid in ids("refill", 60):
        got = asked(world(vacuum_run(), cid, job=look_job("a", "b", look_shares=(shares,))))
        points = hand_points(cid)
        assert got == [("b", k) for k in points if hand_slot(cid, k, 0, shares) == "b"]
        refused += sum(hand_slot(cid, k, 0, shares) == "a" for k in points)
    assert refused > 0


def test_a_hash_past_the_slots_total_asks_nothing(monkeypatch):
    """Contract (plan L0): the rest of a slot asks nothing: a hash at or past the total of the percents draws no
    family, and the last family does not take the rest. Hand-worked on the slot (a 30, b 30), both with a candidate
    everywhere: the asked list is the points whose hash is under 60, a for under 30 and b for 30 to 59; over 60 ids
    some points fall past the total. Regression caught: a rest share read as the last family's, which would ask `b` for
    70% of the points."""
    shares = (("a", 30), ("b", 30))
    register_look(monkeypatch, gold_of("a"), "a")
    register_look(monkeypatch, gold_of("b"), "b")
    past = 0
    for cid in ids("rest", 60):
        got = asked(world(vacuum_run(), cid, job=look_job("a", "b", look_shares=(shares,))))
        points = hand_points(cid)
        assert got == [(f, k) for k in points if (f := hand_slot(cid, k, 0, shares)) is not None]
        past += sum(stable_hash(cid, k, "slot0") % 100 >= 60 for k in points)
    assert past > 0


def test_every_family_of_a_default_table_is_drawn_whatever_the_hash(monkeypatch):
    """Contract (plan L0): the default table is one slot a family at 100, so the hash (always under 100) draws that
    family at every point of every case. Hand-worked: two families, 200 case ids, a candidate everywhere: both are
    asked at every look point. Regression caught: a default with a share under 100 (some points would ask nothing)."""
    register_look(monkeypatch, gold_of("a"), "a")
    register_look(monkeypatch, gold_of("b"), "b")
    for cid in ids("full", 200):
        got = asked(world(vacuum_run(), cid, job=look_job("a", "b")))
        assert got == [(f, k) for k in hand_points(cid) for f in "ab"]


def test_a_family_not_in_the_registry_is_refused_at_the_draw():
    """Contract: a job that asks a family at look points the registry lacks fails at the draw, naming the job and the
    family (unchanged from 0.7.0), whatever its slot table. Regression caught: the slot draw reaching `GOLDS` before the
    family is checked, an obscure KeyError."""
    for shares in (None, ((("not_a_family", 100),),)):
        w = world(vacuum_run(), job=look_job("not_a_family", look_shares=shares))
        with pytest.raises(ValueError, match="not_a_family"):
            questionnaire.looks(w, [])
