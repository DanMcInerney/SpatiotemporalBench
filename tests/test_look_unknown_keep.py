"""The unknown keep of a look family (`JobSpec.look_unknown_keep`, `questionnaire.looks`, 0.8.0 lane L0), worked by hand
on the vacuum's task (candidate look frames 4 to 26, up to 4 points: `looks_kit`) with fake look families registered
for one test."""

from __future__ import annotations

from family_kit import fake_cand, look_moment
from looks_kit import hand_points, look_job, register_look, vacuum_run, world
from stbench import questionnaire
from stbench.seeding import stable_hash, stable_hex


def ids(tag: str, n: int) -> list[str]:
    return [stable_hex(tag, i, n=16) for i in range(n)]


def mixed_gold(name: str):
    """A look family with no slice: at the frames that are multiples of 3 its only candidate is an `unknown` one (in the
    moment's `unknown` pool), at the others an answerable one (in `answerable`); each names a subject of its own."""
    def look(w, k):
        if k % 3 == 0:
            return look_moment(k, [], [fake_cand(name, f"u{k}", (f"r:{name}u{k}",), unknown=True, decision_k=k,
                                                 answer_class="unknown")])
        return look_moment(k, [fake_cand(name, f"a{k}", (f"r:{name}a{k}",), decision_k=k, answer_class="answerable")])
    return look


def classes(w) -> list[tuple[str, int, str | None]]:
    return [(c.family, c.decision_k, c.answer_class) for c in questionnaire.looks(w, [])]


def hand_keep(cid: str, family: str, k: int, keep: int) -> bool:
    """The keep by hand: an unknown question of the family at the point stays when its hash is under the percent."""
    return stable_hash(cid, family, str(k), "unknown") % 100 < keep


def test_an_unsliced_look_family_asks_its_unknown_candidates(monkeypatch):
    """Contract (plan L0, `person_mem`'s and `clear_path`'s unknown golds): a point of a look family that is not sliced
    is asked from whichever pool holds its candidate, so a point with only an `unknown` candidate is asked. Hand-worked:
    with no keep share, every point of `mixed_gold` is asked, the multiples of 3 with the class `unknown` and the
    others `answerable`. Regression caught: `_turn` reading the answerable pool alone for an unpaired family, which
    would never ask an unknown gold."""
    register_look(monkeypatch, mixed_gold("fake_clear"), "fake_clear")
    for cid in ids("pools", 20):
        got = classes(world(vacuum_run(), cid, job=look_job("fake_clear")))
        assert got == [("fake_clear", k, "unknown" if k % 3 == 0 else "answerable") for k in hand_points(cid)]


def test_the_keep_thins_the_unknown_golds_of_its_family_by_the_hash(monkeypatch):
    """Contract (plan L0): an `unknown` look question of a family in `look_unknown_keep` is kept exactly when
    `stable_hash(case id, family, str(k), "unknown") % 100` is under the percent; answerable questions are never
    thinned, and a family the table does not name keeps all. Hand-worked: two families with `mixed_gold` and a keep of
    40 for `fake_a` alone, over 100 case ids: `fake_a` keeps its unknown points by the hand hash (some kept, some
    dropped), all its answerable ones, and `fake_b` all of both. Regression caught: thinning answerable rows,
    thinning by anything but the hash, or a keep that spills onto a family it does not name."""
    for name in ("fake_a", "fake_b"):
        register_look(monkeypatch, mixed_gold(name), name)
    kept = dropped = 0
    for cid in ids("keep", 100):
        job = look_job("fake_a", "fake_b", look_unknown_keep={"fake_a": 40})
        got = classes(world(vacuum_run(), cid, job=job))
        want = []
        for k in hand_points(cid):
            for family in ("fake_a", "fake_b"):
                unknown = k % 3 == 0
                if family == "fake_a" and unknown and not hand_keep(cid, family, k, 40):
                    dropped += 1
                    continue
                kept += family == "fake_a" and unknown
                want.append((family, k, "unknown" if unknown else "answerable"))
        assert got == want
    assert kept > 0 and dropped > 0


def test_a_keep_of_zero_drops_every_unknown_and_a_hundred_keeps_every_one(monkeypatch):
    """Contract (plan L0): the percent is a share of the unknown golds kept: 0 keeps none and 100 keeps all, so the
    ends do not depend on the hash. Hand-worked on `mixed_gold` over 20 ids: keep 0 leaves exactly the answerable
    points (those that are no multiple of 3), keep 100 the full list. Regression caught: a strict comparison that keeps
    the hash value 0 at keep 0, or drops the hash value 99 at keep 100."""
    register_look(monkeypatch, mixed_gold("fake_clear"), "fake_clear")
    for cid in ids("ends", 20):
        points = hand_points(cid)
        zero = classes(world(vacuum_run(), cid, job=look_job("fake_clear", look_unknown_keep={"fake_clear": 0})))
        full = classes(world(vacuum_run(), cid, job=look_job("fake_clear", look_unknown_keep={"fake_clear": 100})))
        assert zero == [("fake_clear", k, "answerable") for k in points if k % 3 != 0]
        assert full == [("fake_clear", k, "unknown" if k % 3 == 0 else "answerable") for k in points]


def test_a_dropped_unknown_reserves_nothing(monkeypatch):
    """Contract: an unknown question the keep drops is not asked, so it names no subject: a later slot may name it.
    Hand-worked: slot one draws `fake_a` (100) with only an `unknown` candidate over `r:shared<k>` and keep 0, slot two
    draws `fake_b` (100) over the same subject (a subject a point): every point asks `fake_b`
    alone. Regression caught: the thinning run before the subject rule, which would let a question that is never asked
    block another."""
    def unknown_gold(w, k):
        return look_moment(k, [], [fake_cand("fake_a", k, (f"r:shared{k}",), unknown=True, decision_k=k,
                                             answer_class="unknown")])

    register_look(monkeypatch, unknown_gold, "fake_a")
    register_look(monkeypatch, lambda w, k: look_moment(k, [fake_cand("fake_b", k, (f"r:shared{k}",), decision_k=k)]),
                  "fake_b")
    for cid in ids("reserve", 10):
        job = look_job("fake_a", "fake_b", look_unknown_keep={"fake_a": 0})
        assert [(f, k) for f, k, _ in classes(world(vacuum_run(), cid, job=job))] == [
            ("fake_b", k) for k in hand_points(cid)]
