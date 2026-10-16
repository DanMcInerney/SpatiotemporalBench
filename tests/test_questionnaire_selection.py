"""The selection order of the end questionnaire (`questionnaire`): which of the questions a case could ask it does ask
when the pools offer more than the subject rule allows. A pool offers its task-critical candidates first, then a thing
of the world before an absent description (a case asks at most one), and a round visits the temporal families first;
the ask order stays the candidates' hash. Fake families on `test_questionnaire`'s run (see its helpers)."""

from __future__ import annotations

from dataclasses import replace

from family_kit import fake_cand, fake_spec, register
from space_kit import vacuum_run
from stbench import questionnaire
from stbench.questions_select import Moment
from stbench.seeding import stable_hash, stable_hex
from test_questionnaire import draws_unknown, end_job, hand_asked, moments_of, world


def test_a_pool_is_offered_critical_first_and_a_thing_of_the_world_before_a_decoy(monkeypatch):
    """Contract: a moment's pool offers its candidates in selection order, the task-critical ones first, then a thing of
    the world before an absent description (a `d:` subject), then by hash; the question is the first the subject rule
    allows, so a critical candidate is asked before a situational one that the hash prefers, and a real unseen thing
    before a decoy. Hand-worked on one sliced moment whose answerable pool holds a situational candidate and a critical
    one and whose unknown pool holds a decoy and a real thing, over 40 case ids: the question is the critical one when
    the moment draws answerable and the real one when it draws unknown, whatever the hash order of each pair (both
    orders occur); with the real thing's subject taken by an earlier question the decoy is offered. Regression caught:
    the hash picking a decoy while an unseen thing of the world is free, or a situational candidate asked while a
    critical one is free."""
    def moment(real: str):
        low = replace(fake_cand("fake_s", "low", ("r:low",)), critical=False)
        pool = [fake_cand("fake_s", "absent", ("d:x/1",), unknown=True),
                fake_cand("fake_s", "real", (real,), unknown=True)]
        return Moment("m", [low, fake_cand("fake_s", "high", ("r:high",))], pool)

    ids = [stable_hex("selection", i, n=16) for i in range(40)]
    register(monkeypatch, fake_spec("fake_s"), lambda w: [moment("e:U")], sliced=True)
    for pool in (moment("e:U").answerable, moment("e:U").unknown):  # the hash alone would put either first
        assert {stable_hash(cid, pool[0].key) < stable_hash(cid, pool[1].key) for cid in ids} == {True, False}
    asked = set()
    for cid in ids:
        (got,) = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_s")))
        want = "real" if draws_unknown(cid, "fake_s", "m") else "high"
        assert got.slots["n"] == want
        asked.add(want)
    assert asked == {"real", "high"}
    taken = fake_cand("fake_t", "taken", ("e:U",))
    register(monkeypatch, replace(fake_spec("fake_t"), temporal=True), lambda w: [Moment("t", [taken], [])])
    for cid in ids:  # fake_t is temporal, so visited first: it takes e:U before the sliced moment is tried
        got = {c.family: c.slots["n"] for c in questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_t",
                                                                                                        "fake_s")))}
        assert got == {"fake_t": "taken", "fake_s": "absent" if draws_unknown(cid, "fake_s", "m") else "high"}


def test_a_case_asks_at_most_one_absent_description(monkeypatch):
    """Contract: two end questions never both name an absent description, whatever family or moment: the moment that
    would ask the second finds no unknown candidate the rule allows, and a sliced moment needs both pools, so it is
    skipped like any with an emptied pool. Hand-worked on a sliced family of five moments whose unknown candidates are
    each their own decoy and whose answerable candidates are free, over 80 case ids: per case the asked questions are
    the hand draw's prefix up to and including the first moment that draws the unknown slice (the one decoy a case
    asks), each in its drawn slice, and no case asks two `d:` subjects. Regression caught: a case of several decoys."""
    keys = [f"m{i}" for i in range(5)]
    register(monkeypatch, fake_spec("fake_s"), lambda w: [
        Moment(k, [fake_cand("fake_s", k, (f"r:{k}",))], [fake_cand("fake_s", f"{k}u", (f"d:kind/{k}",), unknown=True)])
        for k in keys], sliced=True)
    for cid in (stable_hex("one decoy", i, n=16) for i in range(80)):
        got = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_s")))
        assert {c.slots["n"]: c.slice for c in got} == hand_asked(cid, "unknown", keys)
        assert sum(x.startswith("d:") for c in got for x in c.subjects) <= 1


def test_temporal_families_are_visited_first_in_each_round(monkeypatch):
    """Contract: in each round of the end questionnaire the families whose spec says `temporal` take their turn before
    the others, then in `stable_hash(case id, family, "family")` order, so a case that the cap cuts short keeps the
    change-over-time question; the ask order is still the candidates' hash. Hand-worked on a plain and a temporal fake
    family of one moment each, with room for one end question (`MAX_END` 1) over 40 case ids among which the family hash
    alone would visit the plain family first in some: every case asks the temporal one; with room for two it asks both.
    Regression caught: the temporal families squeezed out by hash order under the cap, which leaves a case with no
    question about change."""
    register(monkeypatch, fake_spec("fake_n"), moments_of("fake_n", ["m0"]))
    register(monkeypatch, replace(fake_spec("fake_t"), temporal=True), moments_of("fake_t", ["m0"]))
    ids = [stable_hex("temporal", i, n=16) for i in range(40)]
    plain_first = [cid for cid in ids if stable_hash(cid, "fake_n", "family") < stable_hash(cid, "fake_t", "family")]
    assert 0 < len(plain_first) < len(ids)
    monkeypatch.setitem(questionnaire.MAX_END, "short", 1)
    for cid in ids:
        (got,) = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_n", "fake_t")))
        assert got.family == "fake_t"
    monkeypatch.setitem(questionnaire.MAX_END, "short", 2)
    assert {c.family for c in questionnaire.end(world(vacuum_run(), ids[0], job=end_job("fake_n", "fake_t")))} == {
        "fake_n", "fake_t"}
