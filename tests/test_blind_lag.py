"""A time gold is learned as a lag behind the clock of its own ask (`blind.Pool`, `learned_answer`): the shortcuts that
learn from other scripts' records answer a time question on the asking case's own day. Hand-made records on other days;
every expected answer and score is worked out in the test."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from baseline_kit import UNKNOWN, at, question, record, started, where
from stbench.blind import (
    EPOCH_CLOCK,
    Blind,
    Pool,
    Prior,
    QuestionnaireBlind,
    at_clock,
    lagged,
    learned_answer,
    pool_at,
    to_lag,
)
from stbench.form_check import Asked, form_adversary
from stbench.score import penalised
from stream_kit import end_q, gold_row

SEEN = question("when_seen", {"kind": "mug", "attributes": {"color": "red"}}, answer_type="time")
LADDER = {"ladder": [15.0, 30.0, 60.0, 120.0]}   # `questions_things.TIME_LADDER_S`: the minute scale
ASK = "2026-07-04T10:30:00"                      # where the asking case is asked: its last step's clock
DAYS = (3, 40, 100, 220, 300)                    # the days of the year the five other scripts ran on


def time(clock: datetime | str) -> dict:
    return {"type": "time", "clock": clock if isinstance(clock, str) else clock.isoformat()}


def asked_on(day: int, lag_s: float) -> tuple[str, dict]:
    """The clock a script that ran on the `day`-th day after 1 January 2026 was asked at (09:00:00), and its gold,
    `lag_s` seconds behind it."""
    ask = datetime(2026, 1, 1) + timedelta(days=day, hours=9)
    return ask.isoformat(), time(ask - timedelta(seconds=lag_s))


def gold_of(value: dict) -> dict:
    """A `when_seen` gold row: the time ladder on the value."""
    row = record(SEEN, value, script="g")["gold"]
    return {**row, "family": "when_seen", "tolerances": LADDER}


def prior_at(*lags: float) -> list[dict]:
    """Records of `SEEN`, one a lag, each from a script of its own on another day, with the clock it was asked at."""
    out = []
    for i, (day, lag) in enumerate(zip(DAYS, lags, strict=True)):
        ask, gold = asked_on(day, lag)
        out.append({**record(SEEN, gold, script=f"s{i}", clock=ask), "gold": gold_of(gold)})
    return out


def test_golds_the_same_time_behind_their_asks_are_learned_on_any_day(tmp_path):
    """Contract (VISION rule 5: the shortcut is the best answer learned from other cases): a time gold is a clock on
    its own case's day, so `blind` learns it as the seconds it lies behind the clock its question was asked at, and
    answers the asking case's clock less that lag. Hand case: five scripts on five other days, each asked at 09:00:00
    of its day and each gold 15 s behind: the lag 15 scores +1 on every one of them and on a script left out, so it is
    kept, and the asking case, asked at 10:30:00 on 4 July 2026, is answered 10:29:45, which is its own gold (15 s
    behind) and scores +1. The same records read as clocks (a question that holds no ask clock, as before) put every
    gold on another day from every other, so each candidate scores +1 on itself and -1 on the other four, a mean of
    -.6 against `unknown`'s 0: `unknown`, and 0 on the asking case's gold. Regression caught: a time shortcut that
    compares clocks across days and so never earns credit, which left the sensor-free rule 'the clock 15 s before the
    window's end' to beat the gates' shortcuts."""
    prior = prior_at(15, 15, 15, 15, 15)
    truth = gold_of(time("2026-07-04T10:29:45"))
    answer = started(Blind, tmp_path, prior=prior).ask("end", {"k": 0, "clock": ASK}, [SEEN])["q1"]
    assert answer == time("2026-07-04T10:29:45") and penalised(answer, truth) == 1.0
    as_clocks = learned_answer(pool_at("blind", {**SEEN, "when": "end"}, prior))   # the record holds no `clock`
    assert as_clocks == UNKNOWN and penalised(as_clocks, truth) == 0.0


def test_an_extra_time_candidate_is_judged_as_a_lag_too():
    """Contract (`path_log`'s own-log clock candidate; plan §4.6): the asker's own time candidates are clocks of its
    own day, so they move with the pool (`to_lag`), are judged as lags and answered as the clock they are. Hand pool,
    five scripts on other days whose golds lie 50, 30, 10, -10 and -30 s behind their asks. The ladder is 15, 30, 60,
    120 s and a score is 2s - 1 for the share s of its rungs met. The median lag, 10 s (10:29:50), misses by 40, 20, 0,
    20 and 40 s: 2, 3, 4, 3 and 2 rungs of four, so 0, +.5, +1, +.5 and 0, a mean of .4. The candidate 20 s behind
    (10:29:40), its own, misses by 30, 10, 10, 30 and 50 s: 3, 4, 4, 3 and 2 rungs, so +.5, +1, +1, +.5 and 0, a mean of
    .6, and wins. Read as a clock of another day it would score -1 on every gold and never be chosen. Regression
    caught: the own-log candidate judged in absolute clock, so that it never beat `unknown` and `path_log` never
    answered from its log."""
    prior = prior_at(50, 30, 10, -10, -30)
    own = time("2026-07-04T10:29:40")
    assert learned_answer(Pool(prior, ASK), [own]) == own
    assert learned_answer(Pool(prior), [own]) == UNKNOWN   # as clocks, nothing is credited


def test_lags_move_onto_and_off_the_epoch_and_are_refused_without_a_clock():
    """Contract: `to_lag` and `at_clock` are inverses on a time answer and leave any other as it stands; `lagged`
    moves a time gold by its record's own clock and raises for a record that holds none. Hand values: a gold 15 s
    behind 09:00:00 of 4 January 2026 is 15 s behind the epoch, 1 January 2000 (1999-12-31T23:59:45); moved onto
    10:30:00 on 4 July it is 10:29:45 again. Regression caught: a time gold read against nothing, which goes back to
    the clocks of other days."""
    ask, gold = asked_on(3, 15)
    assert to_lag(gold, ask) == time("1999-12-31T23:59:45")
    assert at_clock(to_lag(gold, ask), ASK) == time("2026-07-04T10:29:45")
    yes = {"type": "yesno", "value": True}
    assert to_lag(yes, ask) == at_clock(yes, ASK) == yes and at_clock(gold, None) == gold
    row = {**record(SEEN, gold, script="s", clock=ask), "gold": gold_of(gold)}
    assert lagged(row)["gold"]["gold"] == time("1999-12-31T23:59:45")
    with pytest.raises(ValueError, match="time gold and no `clock`"):
        lagged({**row, "clock": ""})


def test_questionnaire_blind_learns_the_lag_on_the_asking_cases_day(tmp_path):
    """Contract: `questionnaire_blind` learns a time gold as `blind` does and answers it on the asking case's clock,
    in the question it is asked and in the leave-one-script-out means that choose its feature. Hand case: the five
    scripts above (every gold 15 s behind its ask), asked at 10:30:00 on 4 July. A script left out leaves four golds,
    under the support floor of five, so every feature scores `unknown`'s 0 and the first stands; the five records share
    its value, and the lag 15 answers 10:29:45. Regression caught: a lag put on the wrong clock, or on none, in the
    questionnaire's cached answers."""
    system = started(QuestionnaireBlind, tmp_path, prior=prior_at(15, 15, 15, 15, 15))
    assert system.ask("end", {"k": 0, "clock": ASK}, [SEEN])["q1"] == time("2026-07-04T10:29:45")


@pytest.mark.parametrize("how", ["missing", "empty"])
def test_questionnaire_blind_refuses_a_time_record_with_no_clock_in_its_feature_means(tmp_path, how):
    """Contract: a prior record with a time gold and no clock (the key missing, or empty) is refused wherever the
    learner reads it, as `lagged` refuses it in a pool: the leave-one-script-out means that choose the feature read
    every record of the template, so a record that no pool ever holds still scores, against an unshifted gold, unless it
    is refused. Hand case: the five scripts above, each with no sibling question (feature `count` (0,)), and a sixth on
    day 330 whose end ask has one sibling (`count` (1,)), so it shares no feature value with the five: each of the five
    finds the other four, under the support floor of five, and the sixth finds none, so every mean is `unknown`'s 0 and
    the first feature, `count`, stands; the five records that share its value (0,) have every gold 15 s behind its ask,
    asked at 10:30:00: 10:29:45. With the sixth's clock that is the answer; without it, ValueError, from the mean of the
    sixth alone. Regression caught: a feature chosen on a score read against a gold that was never put on the asker's
    clock, as a prior with one malformed record would silently do."""
    ask, gold = asked_on(330, 15)
    sibling = {"template": "when_seen", "slots": SEEN["slots"]}
    sixth = {**record(SEEN, gold, script="s5", clock=ask, siblings=(sibling,)), "gold": gold_of(gold)}
    prior = prior_at(15, 15, 15, 15, 15)
    assert (started(QuestionnaireBlind, tmp_path / "ok", prior=[*prior, sixth])
            .ask("end", {"k": 0, "clock": ASK}, [SEEN])["q1"] == time("2026-07-04T10:29:45"))
    broken = {k: v for k, v in sixth.items() if k != "clock"} if how == "missing" else {**sixth, "clock": ""}
    with pytest.raises(ValueError, match="time gold and no `clock`"):
        started(QuestionnaireBlind, tmp_path / "bad", prior=[*prior, broken]).ask("end", {"k": 0, "clock": ASK}, [SEEN])


def test_an_empty_ask_clock_over_time_golds_is_refused_and_no_clock_reads_clocks(tmp_path):
    """Contract: the clock a pool carries is the asking case's, and only a missing one reads time golds as clocks (the
    documented mode above); an empty one is not a clock, so over time golds it is refused rather than counted as present
    and failing later inside the date parser. A pool of other golds does not read the clock and is unaffected. Hand
    case: the five scripts above (every gold 15 s behind its ask) asked with an empty clock: ValueError naming it, from
    `learned_answer` and from `Blind`; asked with none, `unknown` as in the first test; five records of a position gold
    at (3.2, 1.5) answer it with an empty clock and with none alike. Regression caught: an empty clock taken as present
    for the lag, whose error says nothing of the ask."""
    prior = prior_at(15, 15, 15, 15, 15)
    with pytest.raises(ValueError, match="empty `clock`"):
        learned_answer(Pool(prior, ""))
    with pytest.raises(ValueError, match="empty `clock`"):
        started(Blind, tmp_path, prior=prior).ask("end", {"k": 0, "clock": ""}, [SEEN])
    assert learned_answer(Pool(prior)) == UNKNOWN
    spots = [record(where("mug", "red", "placed"), at(3.2, 1.5), script=f"s{i}") for i in range(5)]
    assert learned_answer(Pool(spots, "")) == learned_answer(Pool(spots)) == at(3.2, 1.5)


def test_the_form_adversary_learns_the_lag_on_each_questions_own_day():
    """Contract: the form adversary pools the golds of other scripts' questions that share a feature's value and learns
    a time gold as a lag behind its ask (`Asked.clock`). Hand case: six scripts each asked one `when_seen` on a day of
    its own, its gold 15 s behind its ask: the feature `place` (0 for all) pools the other five, whose lag 15 answers
    each question exactly, a mean of 1.0 against an unknown share of 0, so it is named; read as clocks of other days it
    would answer `unknown` for 0. With `in_pool` the pool's own winner answers the same. A question with no clock is
    refused. Regression caught: an adversary blind to time, whose gate then passes for a form that predicts the
    answer."""
    q = end_q(1, template="when_seen", answer_type="time", slots=SEEN["slots"])
    info = {"case": "c", "script": "s", "layout": "apartment_a", "robot": "vacuum", "job": "vacuum/1",
            "length": "short"}
    group = []
    for i, day in enumerate((3, 40, 100, 220, 300, 330)):
        ask, gold = asked_on(day, 15)
        row = gold_row(q, gold=gold, tolerances=LADDER, metric=None, family="when_seen")
        group.append(Asked(f"c{i}", f"s{i}", q, row, {**info, "script": f"s{i}"}, {"place": 0}, ask))
    for kwargs in ({}, {"in_pool": True}):
        rows, table = form_adversary(group, **kwargs)
        assert [r["score"] for r in rows] == [1.0] * 6
        assert (table["when_seen"]["feature"], table["when_seen"]["mean"]) == ("place", 1.0)
    with pytest.raises(ValueError, match="time gold and no `clock`"):
        form_adversary([replace(a, clock=None) for a in group])


def test_a_pool_carries_the_asks_clock_and_the_priors_rule():
    """Contract: `pool_at` gives the records its key found with the clock of the question's record and the prior's
    `in_pool`, so every system that answers through `learned_answer` learns time as a lag without being told twice.
    Hand case: five records of one key asked at 09:00:00 of five days, a question with the clock 10:30:00; a prior that
    is `in_pool` says so. A question with no clock gives a pool with none; under the support floor, an empty pool that
    still holds both. Regression caught: a system that builds its pool by hand and so drops the clock or the rule."""
    prior = prior_at(15, 15, 15, 15, 15)
    rec = {**SEEN, "when": "end", "clock": ASK}
    pool = pool_at("blind", rec, Prior(prior, in_pool=True))
    assert (list(pool), pool.clock, pool.in_pool, EPOCH_CLOCK) == (prior, ASK, True, "2000-01-01T00:00:00")
    plain = pool_at("blind", {**rec, "clock": None}, prior)
    assert (plain.clock, plain.in_pool) == (None, False)
    thin = pool_at("blind", rec, Prior(prior[:2], in_pool=True))
    assert (list(thin), thin.clock, thin.in_pool) == ([], ASK, True)
