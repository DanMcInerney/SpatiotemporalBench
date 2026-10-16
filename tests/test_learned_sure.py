"""The bar every learned answer meets (`blind.learned_answer`) and `path_log`'s pooled ego fallback: a learned answer
is kept only where its held-out gain over abstaining has an interval over the scripts above 0. Hand golds, every
expected value worked out in the test."""

from __future__ import annotations

from baseline_kit import EVENTS, at, offset, question, record, started, step
from stbench.baselines_path import PathLog
from stbench.blind import Blind, Pool, held_out_gain_low, held_out_mean, learned_answer

SLOTS = {"kind": "bag", "attributes": {"color": "red"}}
ORIGIN = {"type": "position", "xy": [0.0, 0.0]}
UNKNOWN = {"type": "unknown"}


def golds_at(*places: tuple[float, float]) -> list[dict]:
    """One record per place, each its own script, the asked question about a position (the gold of a place is credited
    in full at the place and not at 10 m)."""
    q = question("object_where", {"kind": "mug", "attributes": {}})
    return [record(q, at(*p), script=f"s{n}") for n, p in enumerate(places)]


def test_a_rule_that_wins_on_average_but_loses_on_a_script_is_not_kept():
    """Contract (0.6.0 repair, the AMR's `task_things` BROKEN line): `learned_answer` keeps a best response only where
    the lower end of its held-out gain over abstaining, from a bootstrap over the scripts, is above 0; a positive
    held-out mean alone is not enough. Hand case: four scripts, three with a thing at (0, 0), one at (10, 10). Leaving
    out one of the three, the other three hold two at (0, 0) and one far: (0, 0) is the best response and scores 1 on
    the one left out. Leaving out the odd one, (0, 0) is the best response and scores -1 on it (credit 0, penalised -1).
    The held-out mean is (1 + 1 + 1 - 1) / 4 = 0.5 above abstaining's 0, but the scripts' gains are 1, 1, 1 and -1; a
    resample of four holds the odd script three times or more with probability 13/256, 5% of the resamples, whose means
    are -0.5 or -1, so the interval's lower end is negative and the answer is `unknown`. Four scripts that all agree
    gain 1 each and are kept. A pool of an `in_pool` prior is not asked: it answers (0, 0). Regression caught: a
    learned answer kept on its average held-out gain, which would answer where the draw of scripts alone could explain
    the gain, or a bound taken from the point estimate; or the in-pool twin held to the bar, which is in-sample by
    design."""
    mixed = golds_at((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (10.0, 10.0))
    assert held_out_mean(mixed) == 0.5 and held_out_gain_low(mixed) < 0
    assert learned_answer(mixed) == UNKNOWN
    assert learned_answer(Pool(mixed, in_pool=True)) == ORIGIN
    agree = golds_at((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    assert held_out_gain_low(agree) == 1.0
    assert learned_answer(agree) == ORIGIN


def test_blind_abstains_where_its_held_out_gain_is_negative_or_uncertain_and_answers_where_it_is_sure(tmp_path):
    """Contract (0.6.0 repair): `blind`'s learned answer is `unknown` unless its held-out gain over abstaining has an
    interval above 0. Hand pools of one asked question's key, `object_where` golds of one script each, the thing at
    (0, 0) or at (10, 10). (a) Three at (0, 0), two at (10, 10): in the pool (0, 0) scores (3 - 2) / 5 = 0.2; left out
    of a (0, 0) script the other four split 2-2, (0, 0) and `unknown` tie and `unknown` wins (0); left out of a (10, 10)
    script the other four are 3-1, (0, 0) scores -1: the held-out mean is -0.4, a negative gain. (b) Four at (0, 0) and
    one at (10, 10): the held-out mean is (4 - 1) / 5 = +0.6, but the scripts' gains are 1, 1, 1, 1 and -1, and a
    resample of five holds the odd script three times or more with probability 5.8%, a mean of -0.2 or under, so the
    interval is not above 0: uncertain. (c) Nine at (0, 0) and one at (10, 10): (9 - 1) / 10 = +0.8, and a resample of
    ten holds the odd script five times or more (a gain of 0 or under) with probability 0.16%: sure, so (0, 0).
    Regression caught: `blind` answering on a positive or in-pool mean, which scored under `all_unknown` on the
    warehouse's `task_things` (`BROKEN`)."""
    q = question("object_where", {"kind": "mug", "attributes": {}})

    def asked(name, places):
        prior = [record(q, at(*p), script=f"s{n}") for n, p in enumerate(places)]
        return started(Blind, tmp_path / name, prior=prior).ask("end", {"k": 0, "clock": ""}, [q])["q1"]

    near, far = (0.0, 0.0), (10.0, 10.0)
    assert asked("negative", [near] * 3 + [far] * 2) == UNKNOWN
    assert asked("uncertain", [near] * 4 + [far]) == UNKNOWN
    assert asked("sure", [near] * 9 + [far]) == ORIGIN


def test_path_log_ego_now_fallback_abstains_where_its_held_out_gain_is_not_sure(tmp_path):
    """Contract (0.6.0 repair, the drone's `ego_now` BROKEN line): `path_log`'s pooled fallback
    (`baselines_path_ego.pooled`, reached when no dense place the camera faces wins) answers a learned offset only where
    the held-out gain over abstaining is sure. Hand case: six scripts' records of `ego_now` of one kind at one phase and
    base, five with the thing 2.0 ahead and 0.5 left of the camera and one 8.0 ahead and 3.0 right; the ask's camera
    faces away from the place they put on the map, so only the pool is left. Leaving out the odd one the best response
    (2.0, 0.5) scores -1 on it, leaving out one of the five it scores 1: the held-out mean (5 - 1) / 6 is above 0, and
    the plain rule would answer (2.0, 0.5), but a resample of six draws the odd script three times or more with
    probability 6.2%, a mean of 0 or under, so the interval's lower end is not above 0 and the fallback is `unknown`.
    With all six agreeing it answers. Regression caught: the fallback keeping an offset on the average gain of a few
    scripts, which scored -0.12 on the drone's `ego_now` (release-c)."""
    now = question("ego_now", SLOTS, answer_type="offset")

    def run(path, offsets):
        recs = [record(now, offset(*o), script=f"s{i}", when="during", phase="cleaning", base=(0.0, 0.0), yaw=0.0)
                for i, o in enumerate(offsets)]
        system = started(PathLog, path, kinds=("kitchen",), events=EVENTS, prior=recs)
        system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
        system.observe([step(0, 30.0, 30.0, 90.0, phase="cleaning")])
        return system.ask("during", {"k": 0, "clock": ""}, [now])["q1"]

    five = [(2.0, 0.5)] * 5
    assert run(tmp_path / "odd", [*five, (8.0, -3.0)]) == UNKNOWN
    assert run(tmp_path / "agree", [*five, (2.0, 0.5)]) == offset(2.0, 0.5)
