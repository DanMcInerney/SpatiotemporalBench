"""`COINS` (`questions_text`, `questionnaire`): an end family whose moments are drawn by a 50% coin between their two
pools, the yes candidates in `Moment.answerable` and the no candidates in `Moment.unknown`, by hand on `space_kit`'s run
and fake families. The draw is the sliced draw (`stable_hash(case id, family, moment key, "slice")`) at 50 percent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from family_kit import fake_cand, fake_spec, register
from space_kit import vacuum_run
from stbench import questionnaire, questions_text
from stbench.questions_select import Moment
from stbench.questions_text import _load
from stbench.seeding import stable_hash, stable_hex
from test_questionnaire import end_job, world


def draws_no(cid: str, family: str, key: str) -> bool:
    """The coin, written out by hand: a moment of a coin family takes its no pool for half of the (case id, family,
    moment key) triples."""
    return stable_hash(cid, family, key, "slice") % 100 < 50


def two_pools(family: str, keys, blocked: str | None = None):
    """A gold function for a coin family: one moment for each key, each with a yes and a no candidate. In the `blocked`
    pool ("yes" or "no") every moment's candidate names the same subject, so the first moment asked uses it up; the
    other pool's candidates name subjects of their own."""
    def moment(k):
        yes = ("e:shared",) if blocked == "yes" else (f"r:{k}",)
        no = ("r:shared/absent",) if blocked == "no" else (f"r:{k}/no",)
        return Moment(k, [fake_cand(family, k, yes)], [fake_cand(family, f"{k}n", no, unknown=True)])

    return lambda w: [moment(k) for k in keys]


def test_the_coin_is_fair_over_moments(monkeypatch):
    """Contract: the pool a coin family's moment takes is drawn per (case id, family, moment key) by the sliced draw at
    50 percent: over 1,000 synthetic keys the yes share lies in [0.40, 0.60] and each draw is the hand draw; a sliced
    family keeps its 20 percent (the draw reads the family's own rule). Regression caught: a coin that shares the sliced
    family's 20 percent (a yes/no answered `no` four times in five, so an all-`no` agent scores), or one that moves the
    declared share of a sliced family."""
    register(monkeypatch, fake_spec("fake_c"), two_pools("fake_c", []), coin=True)
    keys = [stable_hex("coin key", i, n=16) for i in range(1000)]
    drawn = [questionnaire.unknown_slice("c0", "fake_c", k) for k in keys]
    assert drawn == [draws_no("c0", "fake_c", k) for k in keys]
    assert 0.40 <= 1 - sum(drawn) / len(keys) <= 0.60
    sliced = [questionnaire.unknown_slice("c0", "object_where", k) for k in keys]
    assert 0.15 <= sum(sliced) / len(keys) <= 0.25


def test_a_coin_moment_is_asked_only_when_both_pools_allow_a_candidate(monkeypatch):
    """Contract: a moment of a coin family is asked only when it has a yes and a no candidate the subject rule allows,
    whatever side the coin lands on, else whether it is asked would tell its answer. Hand-worked over 40 case ids, among
    which the hand draw lands the yes-only moment on both sides: of three moments, `both` (a yes and a no candidate),
    `yes_only` and `no_only`, only `both` is ever asked, in the pool its draw gives. Regression caught: a coin family
    handled as an unsliced one (the yes-only moment always asked, so its answer is always yes), or `or` for `and`."""
    moments = [Moment("both", [fake_cand("fake_c", "both", ("r:both",))],
                      [fake_cand("fake_c", "bothn", ("r:both/no",), unknown=True)]),
               Moment("yes_only", [fake_cand("fake_c", "yes_only", ("r:yes",))], []),
               Moment("no_only", [], [fake_cand("fake_c", "no_onlyn", ("r:no/no",), unknown=True)])]
    register(monkeypatch, fake_spec("fake_c"), lambda w: moments, coin=True)
    ids = [stable_hex("both pools", i, n=16) for i in range(40)]
    assert {draws_no(cid, "fake_c", "yes_only") for cid in ids} == {True, False}
    assert {draws_no(cid, "fake_c", "no_only") for cid in ids} == {True, False}
    for cid in ids:
        asked = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_c")))
        want = "bothn" if draws_no(cid, "fake_c", "both") else "both"
        assert [c.slots["n"] for c in asked] == [want]


def test_every_moment_of_two_full_pools_is_asked_in_the_pool_its_coin_gives(monkeypatch):
    """Contract: with a yes and a no candidate free at every moment, each moment is asked once, in the pool the hand
    draw gives it, and the yes share of the asked questions lies in [0.40, 0.60]. Hand-worked on four moments over 200
    case ids. Regression caught: a coin that depends on the moment's position or the case, which makes one answer
    likelier in some form."""
    keys = ["m0", "m1", "m2", "m3"]
    register(monkeypatch, fake_spec("fake_c"), two_pools("fake_c", keys), coin=True)
    asked = no = 0
    for cid in (stable_hex("coin share", i, n=16) for i in range(200)):
        got = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_c")))
        assert {c.slots["n"][:2]: c.slice == "unknown" for c in got} == {k: draws_no(cid, "fake_c", k) for k in keys}
        asked += len(got)
        no += sum(c.slice == "unknown" for c in got)
    assert asked == 4 * 200 and 0.40 <= 1 - no / asked <= 0.60


def hand_asked(cid: str, blocked: str, keys) -> dict[str, str]:
    """What `two_pools("fake_c", keys, blocked)` asks, worked by hand: the moments in `stable_hash(case id, "fake_c",
    key, "moment")` order, each asked in the pool it draws until one draws the `blocked` pool, which takes the shared
    subject; every moment after it finds that pool without an allowed candidate, so none is asked. Keyed by the
    question's slot, valued by the answer's side."""
    asked = {}
    for k in sorted(keys, key=lambda k: stable_hash(cid, "fake_c", k, "moment")):
        no = draws_no(cid, "fake_c", k)
        asked[f"{k}n" if no else k] = "no" if no else "yes"
        if no == (blocked == "no"):
            break
    return asked


@pytest.mark.parametrize("blocked", ["yes", "no"])
def test_a_pool_the_subject_rule_emptied_leaves_the_coin_fair(monkeypatch, blocked):
    """Contract: a moment with no candidate the subject rule allows in either pool is skipped whatever the coin gives
    (the other pool's candidate is free), so over cases the asked yes and no are equally many (the skew: the
    pool that runs out, the answerable one, was asked less, so the other side of a form was asked more). Hand-worked on
    five moments whose candidates in the `blocked` pool all name one subject: per case the asked questions are the
    hand draw's prefix up to and including the first moment that draws the blocked pool, and over 400 case ids the
    asked no share lies in [0.40, 0.60], where a search of the drawn pool alone gives about 0.7 (or 0.3).
    Regression caught: a moment skipped only when its drawn pool is empty."""
    keys = [f"m{i}" for i in range(5)]
    register(monkeypatch, fake_spec("fake_c"), two_pools("fake_c", keys, blocked), coin=True)
    asked = no = 0
    for cid in (stable_hex("coin skew", i, n=16) for i in range(400)):
        got = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_c")))
        assert {c.slots["n"]: "no" if c.slice == "unknown" else "yes" for c in got} == hand_asked(cid, blocked, keys)
        asked += len(got)
        no += sum(c.slice == "unknown" for c in got)
    assert 0.40 <= no / asked <= 0.60


def fake_module(specs, golds, sliced=frozenset(), coins=frozenset()):
    return SimpleNamespace(SPECS=tuple(specs), GOLDS=golds, SLICED=sliced, COINS=coins)


def test_a_family_cannot_be_both_sliced_and_coined(monkeypatch):
    """Contract: `questions_text._load` returns `COINS` as its fourth part, from the `COINS` of each module that has one
    (a module without it has none); a family named in both `SLICED` and `COINS` raises, naming it, as does a `COINS`
    name that is no end family of its module (a stray name, a during family). The live registry's two sets are
    disjoint. Hand-worked on fake modules. Regression caught: one family drawn by two rules at once, or a mistyped
    `COINS` name that silently draws nothing."""
    mods: dict = {}
    monkeypatch.setattr(questions_text, "_import", mods.get)
    ghost, other = fake_spec("ghost"), fake_spec("other", when="during", scope="now")
    mods["m1"] = fake_module([ghost], {"ghost": print}, coins=frozenset({"ghost"}))
    found = _load(("m1",))
    assert (list(found.families), found.sliced, found.coins) == (["ghost"], frozenset(), frozenset({"ghost"}))
    mods["m1"] = SimpleNamespace(SPECS=(ghost,), GOLDS={"ghost": print})
    assert _load(("m1",)).coins == frozenset()
    mods["m1"] = fake_module([ghost], {"ghost": print}, sliced=frozenset({"ghost"}), coins=frozenset({"ghost"}))
    with pytest.raises(ValueError, match=r"ghost.*both SLICED and COINS"):
        _load(("m1",))
    for name, module in {
            "nothere": fake_module([ghost], {"ghost": print}, coins=frozenset({"nothere"})),  # a stray COINS name
            "other": fake_module([other], {"other": print}, coins=frozenset({"other"})),  # COINS on a during family
    }.items():
        mods["m1"] = module
        with pytest.raises(ValueError, match=f"{name}.*COINS"):
            _load(("m1",))
    assert questions_text.COINS.isdisjoint(questions_text.SLICED)
    assert questionnaire.COINS is questions_text.COINS
