"""The look draw and the order of the questionnaire (`questionnaire.looks`, `ask`, 0.6.0 lane A2): worked by hand on
the vacuum's task (candidate look frames 4 to 26, up to 4 points: `looks_kit`) with fake families registered for one
test."""

from __future__ import annotations

from dataclasses import replace

import pytest

from family_kit import fake_cand, fake_spec, look_moment, register
from looks_kit import FRAMES, hand_points, look_job, register_look, vacuum_run, world, worlds
from space_kit import JOB, things_run
from stbench import questionnaire, questions_looks
from stbench.questionnaire import UNKNOWN_PERCENT
from stbench.questions_select import Moment
from stbench.seeding import stable_hash, stable_hex
from stbench.task_predicate import TaskThings


def ids(tag: str, n: int) -> list[str]:
    return [stable_hex(tag, i, n=16) for i in range(n)]


def asks(w):
    return [(c.family, c.decision_k) for c in questionnaire.looks(w, [])]


def test_each_look_family_asks_once_a_point_in_the_jobs_order(monkeypatch):
    """Contract: at each look point (keyframe order) each family of `JobSpec.looks`, in the job's order, asks at most
    one question; a family with no moment there (a dead band, None) or no candidate asks nothing, and nothing takes its
    place. Hand-worked on two look families over the vacuum's task: `fake_b` (declared first in the job) and `fake_a`
    (the registry's order is the other way), `fake_a` dead at the even frames (no moment) and `fake_b` at the multiples
    of 3 (a moment with no candidate): for every case id the asked list is the points in order with, at each, `fake_b`
    then `fake_a`, less those dead places. Regression caught: families ordered by the registry rather than the job,
    two questions of one family at a point, or a dead band replaced by the other family's candidate."""
    def gold(name, dead_none=lambda k: False, dead_empty=lambda k: False):
        def look(w, k):
            if dead_none(k):
                return None
            cands = [] if dead_empty(k) else [fake_cand(name, k, (f"r:{name}{k}",), decision_k=k)]
            return look_moment(k, cands)
        return look

    register_look(monkeypatch, gold("fake_a", dead_none=lambda k: k % 2 == 0), "fake_a")
    register_look(monkeypatch, gold("fake_b", dead_empty=lambda k: k % 3 == 0), "fake_b")
    for cid in ids("order", 30):
        w = world(vacuum_run(), cid, job=look_job("fake_b", "fake_a"))
        want = [(f, k) for k in hand_points(cid) for f in ("fake_b", "fake_a")
                if not (f == "fake_a" and k % 2 == 0) and not (f == "fake_b" and k % 3 == 0)]
        assert asks(w) == want


def test_a_reserved_subject_is_never_named_by_a_look_and_no_entity_twice(monkeypatch):
    """Contract (design §2, the reservation order): the end questions' subjects are reserved, so no look names one, and
    no entity is named twice across the looks. Hand-worked: an end family asks the thing e:A; a look family offers e:A,
    e:B and e:C at every point (the vacuum has four points); the looks name e:B and e:C once each and never e:A, at the
    first two points in keyframe order, and the others find nothing allowed (a dead band, not a repeat), whichever of
    B and C the hash puts first. Regression caught: looks drawn before the end questions or over none (a look names
    the reserved thing e:A), or an entity named by two looks."""
    register(monkeypatch, fake_spec("fake_end"), lambda w: [Moment("m", [fake_cand("fake_end", "m", ("e:A",))], [])])

    def gold(w, k):
        return look_moment(k, [fake_cand("fake_look", f"{k}{x}", (f"e:{x}",), decision_k=k) for x in "ABC"])

    register_look(monkeypatch, gold)
    job = look_job(ends=("fake_end",))
    firsts = set()
    for cid in ids("reserved", 40):
        w = world(vacuum_run(), cid, job=job)
        taken = questionnaire.ask(w)
        end = [c for c in taken if c.decision_k is None]
        looks = [c for c in taken if c.decision_k is not None]
        assert [c.family for c in end] == ["fake_end"]
        points = hand_points(cid)
        assert [c.decision_k for c in looks] == points[:2]
        named = [s for c in looks for s in c.subjects]
        assert sorted(named) == ["e:B", "e:C"]
        firsts.add(named[0])
    assert firsts == {"e:B", "e:C"}


def test_a_look_sees_the_end_draw_not_the_other_way_about(monkeypatch):
    """Contract: the end questions are drawn first and unchanged by the looks, and no look names a subject they hold.
    Hand-worked: `object_where` over the run with things, 20 case ids, and a look family that offers every thing of the
    run (`e:E0001` to `e:E0011`) beside a free subject at each point: the end rows of the full ask equal
    `questionnaire.end` of the same world, no look names a thing an end question names (the looks fall to the free
    subject or another thing), and over the ids some look is asked and some thing the end asks is among those offered.
    Regression caught: looks drawn first (an end question then loses the thing a look took, or a look takes a reserved
    thing), so the end questions depend on what a look found."""
    def gold(w, k):
        things = [fake_cand("fake_look", f"{k}{n}", (f"e:E{n:04d}",), decision_k=k) for n in range(1, 12)]
        return look_moment(k, [*things, fake_cand("fake_look", f"{k}f", (f"r:free{k}",), decision_k=k)])

    register_look(monkeypatch, gold)
    asked = reserved = 0
    for cid in ids("end first", 20):
        w = world(things_run(), cid, job=look_job())
        rows = questionnaire.ask(w)
        end = [c for c in rows if c.decision_k is None]
        looks = [c for c in rows if c.decision_k is not None]
        assert [c.key for c in end] == [c.key for c in questionnaire.end(w)]
        held = {s for c in end for s in c.subjects}
        assert not held & {s for c in looks for s in c.subjects}
        asked += len(looks)
        reserved += bool(held & {f"e:E{n:04d}" for n in range(1, 12)})
    assert asked > 0 and reserved > 0


def test_a_look_family_mapped_to_a_decision_kind_is_refused(monkeypatch):
    """Contract: a family in `LOOKS` named in `JobSpec.decisions` fails at once, naming the family and the kind (its
    gold gives a `Moment`, which the decision draw would hand to `records` as a candidate). Hand-worked: `fake_look`
    mapped to `before_doorway`, where the vacuum has a decision point: `during`, `drawn`, `candidates` and `ask` all
    raise. Regression caught: an obscure failure in the record writer, or a look family silently skipped."""
    register_look(monkeypatch)
    w = world(vacuum_run(), job=look_job(decisions={"before_doorway": ("fake_look",)}))
    for call in (questionnaire.during, questionnaire.drawn, questionnaire.candidates, questionnaire.ask):
        with pytest.raises(ValueError, match="fake_look.*before_doorway"):
            call(w)


def test_what_an_end_question_names_is_dont_care_in_the_relaxed_family_too(monkeypatch):
    """Contract (coordinator, F9): `RELAX` is given the things the end questions name as well as those the looks name,
    so a thing asked about at the end is don't-care in `task_things`. Hand-worked: `fake_set`'s candidate is "items"
    over e:A; the looks name only e:Z, and an end family `fake_end` asks about e:A: `RELAX` makes the set empty and
    keep 0 drops it; without `fake_end` (nothing names e:A) it stays and is kept. Regression caught: relaxing by
    the looks alone, so a thing the end asks about is also scored as an item of `task_things`."""
    job = fake_set(monkeypatch, keep=0, named_by_look=False)
    register(monkeypatch, fake_spec("fake_end"), lambda w: [Moment("m", [fake_cand("fake_end", "m", ("e:A",))], [])])
    for cid in ids("end names", 10):
        both = questionnaire.ask(world(vacuum_run(), cid, job=replace(job, ends=("fake_set", "fake_end"))))
        assert {c.family for c in both if c.decision_k is None} == {"fake_end"}
        alone = questionnaire.ask(world(vacuum_run(), cid, job=job))
        assert [c.answer_class for c in alone if c.family == "fake_set"] == ["items"]


def draws_unknown(cid: str, family: str, key: str) -> bool:
    """The slice draw by hand: a moment of a sliced family is the unknown slice for a fifth of the (case, family, key)
    triples (the look point's key is `str(k)`)."""
    return stable_hash(cid, family, key, "slice") % 100 < 20


def sliced_gold(w, k):
    """A sliced look family: an answerable and an unknown candidate at every point, each with a subject of its own."""
    return look_moment(k, [fake_cand("fake_mem", f"k{k}", (f"r:mem{k}",), decision_k=k, answer_class="answerable")],
                       [fake_cand("fake_mem", f"u{k}", (f"r:mem{k}/u",), unknown=True, decision_k=k,
                                  answer_class="unknown")])


def test_the_unknown_share_of_a_sliced_look_family_is_the_declared_fifth(monkeypatch):
    """Contract (plan §4 A2, class draws): a sliced look family's question at a point is from its unknown pool for
    `UNKNOWN_PERCENT` of the (case, family, point) triples, drawn before and independent of whether the point is asked,
    so over cases the asked share is a fifth: within 0.03 of 0.20 over 2,000 synthetic case ids (four or five points
    each, about 9,700 questions: the draw's SD is 0.004; the share would be 0.5 were only the unknown pool searched).
    Hand-worked per id: every point of the vacuum's draw asks (the pools' subjects are free), in the slice the hand
    draw of (case, family, str(k)) gives. Regression caught: the slice read from the wrong key (the same slice at every
    point of a case), a point skipped by its slice, or a pool searched alone."""
    register_look(monkeypatch, sliced_gold, "fake_mem", sliced=True)
    assert UNKNOWN_PERCENT == 20
    script, asked, unknown, drawn = vacuum_run(), 0, 0, 0
    for cid, w in zip(cids := ids("share", 2000), worlds(script, cids, job=look_job("fake_mem")), strict=True):
        got = questionnaire.looks(w, [])
        assert [c.slice == "unknown" for c in got] == [draws_unknown(cid, "fake_mem", str(k))
                                                       for k in hand_points(cid)]
        asked += len(got)
        drawn += len(hand_points(cid))
        unknown += sum(c.slice == "unknown" for c in got)
    assert asked == drawn and abs(unknown / asked - 0.20) <= 0.03


def test_a_sliced_look_point_needs_both_pools(monkeypatch):
    """Contract: a point of a sliced look family is asked only when it has a candidate allowed in each pool, whatever
    slice it would draw (so a point's being asked never predicts its slice). Hand-worked: the unknown pool empty at the
    frames that are multiples of 5 and the answerable pool empty at the multiples of 7: neither is asked, whichever
    slice the hand draw gives them, and the other points are. Regression caught: the drawn pool searched alone."""
    def gold(w, k):
        m = sliced_gold(w, k)
        return Moment(m.key, [] if k % 7 == 0 else m.answerable, [] if k % 5 == 0 else m.unknown)

    register_look(monkeypatch, gold, "fake_mem", sliced=True)
    for cid in ids("pools", 20):
        got = questionnaire.looks(world(vacuum_run(), cid, job=look_job("fake_mem")), [])
        assert [c.decision_k for c in got] == [k for k in hand_points(cid) if k % 5 and k % 7]


def fake_set(monkeypatch, *, keep: int, named_by_look: bool):
    """An end family `fake_set` in `CAPPED` and `RELAX`: one moment, a candidate of class "items" over the thing e:A
    (slot `ids`), which `RELAX` makes "empty" when a look named every one of its things. A look family offers e:A (or,
    with `named_by_look` False, e:Z) at every point. The job keeps `keep` percent of the empty."""
    spec = fake_spec("fake_set")

    def relax(c, named):
        empty = set(c.slots["ids"]) <= named
        return replace(c, answer_class="empty" if empty else "items")

    register(monkeypatch, spec, lambda w: [Moment("m", [replace(
        fake_cand("fake_set", 1, answer_class="items"), slots={"n": 1, "ids": ["A"]})], [])], capped=True,
        relax=relax)
    thing = "A" if named_by_look else "Z"
    register_look(monkeypatch, lambda w, k: look_moment(k, [fake_cand("fake_look", k, (f"e:{thing}",), decision_k=k)]))
    predicate = TaskThings("near", "point", frozenset({"cleaning"}), empty_keep_percent=keep)
    return look_job(ends=("fake_set",), task_things=predicate)


@pytest.mark.parametrize("named_by_look, keep, kept", [
    (True, 0, False),     # a look named e:A: RELAX made the set empty, the cap (keep 0) drops it
    (True, 100, True),    # keep 100: every empty is kept
    (False, 0, True),     # no look named it: the set is items, which the cap never touches
])
def test_relax_runs_before_capped_and_the_cap_reads_the_class_after_it(monkeypatch, named_by_look, keep, kept):
    """Contract (plan §4 A2, the order): `RELAX` makes the entities the looks name don't-care in an end family's
    candidate and recomputes its `answer_class`, and only then `CAPPED` thins the empty ones by the job's
    `empty_keep_percent`. Hand-worked on one candidate of class "items" over e:A: when a look names e:A, `RELAX` makes
    it "empty" and keep 0 drops it, keep 100 keeps it; when the looks name another thing it stays "items" and keep 0
    keeps it. Regression caught: the cap read before `RELAX` (the candidate still "items": kept when it should go),
    `RELAX` skipped, or the cap applied to items."""
    job = fake_set(monkeypatch, keep=keep, named_by_look=named_by_look)
    for cid in ids("relax", 10):
        taken = questionnaire.ask(world(vacuum_run(), cid, job=job))
        sets = [c for c in taken if c.family == "fake_set"]
        assert len(sets) == kept
        if kept:
            assert sets[0].answer_class == ("empty" if named_by_look else "items")


def test_the_cap_keeps_an_empty_by_its_hash_under_the_jobs_percent(monkeypatch):
    """Contract (plan §2 call 6): an empty question of a `CAPPED` family is kept iff `stable_hash(case id, family, key,
    "empty") % 100 < job.task_things.empty_keep_percent`, per job and no global. Hand-worked: with the look naming e:A
    (so the set is empty) and keep 50, over 200 case ids each is kept exactly when the hand hash is under 50, and
    some are kept and some dropped; a job with no `task_things` thins nothing. Regression caught: a keep percent not
    read from the job, a draw that is not the hash (the same cases kept at every percent), or a cap on a job that sets
    none."""
    job = fake_set(monkeypatch, keep=50, named_by_look=True)
    kept = 0
    for cid in ids("keep", 200):
        w = world(vacuum_run(), cid, job=job)
        got = [c for c in questionnaire.ask(w) if c.family == "fake_set"]
        key = questionnaire.end(w)[0].key
        assert bool(got) == (stable_hash(cid, "fake_set", key, "empty") % 100 < 50)
        kept += bool(got)
    assert 60 < kept < 140
    bare = replace(job, task_things=None)
    assert all(any(c.family == "fake_set" for c in questionnaire.ask(world(vacuum_run(), cid, job=bare)))
               for cid in ids("bare", 20))


def test_a_capped_candidate_without_a_class_is_refused(monkeypatch):
    """Contract (A2's review, 0.6.0): the thinning reads each `CAPPED` candidate's `answer_class` ("empty" or not), so a
    capped family that sets none is refused, naming it, whether or not the job has a `task_things` to thin by. Hand-
    worked: one candidate of `fake_set` with no class, asked by a job with a predicate and by one without; a classed
    candidate of the same family draws. Regression caught: an unclassed candidate passing the cap as "not empty"
    (a family whose empties are never thinned, unseen)."""
    spec = fake_spec("fake_set")
    predicate = TaskThings("near", "point", frozenset({"cleaning"}))
    job = replace(look_job(ends=("fake_set",), task_things=predicate), looks=())
    for cand, refused in ((fake_cand("fake_set", 1), True), (fake_cand("fake_set", 1, answer_class="items"), False)):
        register(monkeypatch, spec, lambda w, cand=cand: [Moment("m", [cand], [])], capped=True)
        for j in (job, replace(job, task_things=None)):
            if refused:
                with pytest.raises(ValueError, match="capped"):
                    questionnaire.ask(world(vacuum_run(), job=j))
            else:
                assert [c.family for c in questionnaire.ask(world(vacuum_run(), job=j))] == ["fake_set"]


def test_the_ask_order_is_the_during_rows_by_keyframe_then_the_end_rows(monkeypatch):
    """Contract: the rows are the decision points' and the look points' in keyframe order (a decision point before a
    look point at the same keyframe, and the looks of one point in the job's order), then the end rows in hash order.
    Hand-worked on the vacuum with `path_ahead` asked at its decision point 21 and the look family at the hand-drawn
    points: for each id the during rows are sorted by keyframe, and where 21 is a look point the `path_ahead` row comes
    first.
    Regression caught: looks listed after the decisions, or an order the harness's ask order does not follow."""
    register_look(monkeypatch)
    job = look_job(decisions={"before_doorway": ("path_ahead",)})
    seen_both = False
    for w in worlds(things_run(), ids("rows", 40), job=job):
        rows = questionnaire.ask(w)
        during = [c for c in rows if c.decision_k is not None]
        assert rows[:len(during)] == during and len(during) < len(rows)
        assert [c.decision_k for c in during] == sorted(c.decision_k for c in during)
        at_21 = [c.family for c in during if c.decision_k == 21]
        assert at_21 in ([], ["path_ahead"], ["fake_look"], ["path_ahead", "fake_look"])
        seen_both |= at_21 == ["path_ahead", "fake_look"]
    assert seen_both


def test_a_job_without_looks_asks_what_it_asked_before(monkeypatch):
    """Contract: for a job with no look family and no `RELAX`/`CAPPED` family `ask` is the decision draw followed by
    the end draw, row for row (the old compile). Regression caught: the new order changing the questions of the jobs
    that map no look."""
    for w in worlds(vacuum_run(), ids("old", 10)):
        assert [c.key for c in questionnaire.ask(w)] == [c.key for c in questionnaire.during(w) + questionnaire.end(w)]


def test_the_candidates_hold_the_look_families_at_every_candidate_point(monkeypatch):
    """Contract: `candidates` (what the hidden-change checks compare) holds each look family at every candidate point,
    drawn or not. Hand-worked: all 23 candidate frames of the vacuum's task (4 to 26), once each, with the pools'
    candidates of a sliced family, whatever the case id. Regression caught: only the drawn points compared (a change at
    an undrawn point goes unseen)."""
    register_look(monkeypatch, sliced_gold, "fake_mem", sliced=True)
    got = [c.decision_k for c in questionnaire.candidates(world(vacuum_run(), job=look_job("fake_mem")))]
    assert sorted(got) == sorted(FRAMES * 2) and questions_looks.candidate_points(
        world(vacuum_run(), job=look_job("fake_mem"))) == FRAMES


def test_a_candidate_at_the_wrong_keyframe_is_refused(monkeypatch):
    """Contract: a look question carries the look point as its `decision_k` (the harness asks after it), and a job's
    look family that is not a look family of the registry is refused naming it. Regression caught: a question asked
    after the wrong step, or a job naming a family the draw cannot ask at a point."""
    register_look(monkeypatch, lambda w, k: look_moment(k, [fake_cand("fake_look", k, (f"r:{k}",), decision_k=k + 1)]))
    with pytest.raises(ValueError, match="decision_k"):
        questionnaire.looks(world(vacuum_run(), job=look_job()), [])
    with pytest.raises(ValueError, match="nosuch"):
        questionnaire.looks(world(vacuum_run(), job=look_job("nosuch")), [])
    with pytest.raises(ValueError, match="path_ahead"):
        questionnaire.looks(world(vacuum_run(), job=look_job("path_ahead")), [])


@pytest.mark.parametrize("family, good, bad", [
    ("same_place", ("yes", "moved"), ("moved", "yes")),
    ("same_place", ("yes", "vacated"), ("vacated", "yes")),
    ("object_was", ("earlier", "later"), ("later", "earlier")),
    ("object_where", ("answerable", "unknown"), ("unknown", "answerable")),
])
def test_a_candidate_whose_class_does_not_fit_its_pool_is_refused(monkeypatch, family, good, bad):
    """Contract (plan §3): in a two-pool family the answerable pick is of the family's first pool (`records.pool_of`:
    yes, earlier, answerable) and the unknown pick of its second (no, later, unknown); a candidate of the other side is
    refused, naming it. Hand-worked on a coin family (`same_place`: yes / moved or vacated), a coin of states
    (`object_was`: earlier / later) and a sliced family (`object_where`): the right sides draw, the swapped sides raise,
    and so do candidates with no class (every family with pools sets one; an unclassed candidate would pass unseen).
    Regression caught: a coin's sides swapped, or a class left off a candidate (the balance would be read between
    the wrong classes, or not at all, without a sound)."""
    def gold(classes):
        def moments(w):
            return [Moment("m", [fake_cand(family, "a", ("r:a",), answer_class=classes[0])],
                           [fake_cand(family, "b", ("r:b",), unknown=True, answer_class=classes[1])])]
        return moments

    sliced = family == "object_where"
    register(monkeypatch, fake_spec(family), gold(good), sliced=sliced, coin=not sliced)
    assert len(questionnaire.end(world(vacuum_run(), job=replace(JOB, ends=(family,))))) == 1
    register(monkeypatch, fake_spec(family), gold(bad), sliced=sliced, coin=not sliced)
    with pytest.raises(ValueError, match="pool"):
        questionnaire.end(world(vacuum_run(), job=replace(JOB, ends=(family,))))
    register(monkeypatch, fake_spec(family), gold((None, None)), sliced=sliced, coin=not sliced)
    with pytest.raises(ValueError, match="no answer_class"):
        questionnaire.end(world(vacuum_run(), job=replace(JOB, ends=(family,))))
