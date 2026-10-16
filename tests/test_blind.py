"""`blind`: the best response to other scripts' golds at the first key level that has enough (V0 §10, plan §4.6), kept
only when it holds on a script it did not learn from (`learned_answer`). Pools are hand-made records
(`baseline_kit.record`); every expected answer is worked out in the test."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from baseline_kit import UNKNOWN, at, number, record, started, step, where
from stbench.blind import (
    Blind,
    Pool,
    Prior,
    QuestionnaireBlind,
    best_response,
    held_out_mean,
    learned_answer,
    pool_at,
)

EVENT = "when you started cleaning"
SRC = Path(__file__).resolve().parents[1] / "src" / "stbench"


def golds(*values):
    return [record(where("mug", "red", EVENT), v, script=f"s{i}")["gold"] for i, v in enumerate(values)]


def test_blind_gives_the_answer_with_the_best_mean_penalised_score_over_other_golds():
    """Contract (V0 §10): the best response maximises the mean penalised score over the pool, on the 0.25, 0.5, 1, 2
    ladder. Distances 2.2, 2.2, 3.0: answering 2.2 scores (1 + 1 + 0) / 3 (a miss by 0.8 earns the 1 and 2 rungs, half,
    so 0), answering 3.0 (0 + 0 + 1) / 3, `unknown` 0, so 2.2. Places (1, 1) and (5, 5) in one pool: either scores (1 -
    1) / 2 = 0, a tie that `unknown`, first, wins. Regression caught: a shortcut baseline that is not the best response,
    or one that guesses where guessing cannot pay."""
    assert best_response(golds(number(2.2), number(2.2), number(3.0))) == number(2.2)
    assert best_response(golds(at(1.0, 1.0), at(5.0, 5.0))) == UNKNOWN
    assert best_response(golds(UNKNOWN, UNKNOWN, at(1.0, 1.0))) == UNKNOWN   # +1 twice against 0 once


def lookup(kind, color, event=EVENT, template="object_where"):
    """What a lookup holds of a question: its template, slots and `when`."""
    q = where(kind, color, event)
    return {"template": template, "slots": q["slots"], "when": "end"}


def many(kind, color, x, scripts, event=EVENT):
    """One record of the thing at (x, x) for each script named, asked about at `event`."""
    return [record(where(kind, color, event), at(x, x), script=s) for s in scripts]


def test_the_pool_relaxes_from_exact_slots_to_the_template():
    """Contract (`answer_keys.levels("blind", ...)`): the pool is the records that share the question's key at the
    first level that has enough of them: all slots, then kind and event (attributes dropped), then the kind, then the
    template. The prior: five red mugs (five scripts, at 1), four blue mugs (four scripts, at 2), five white mugs (two
    scripts, at 3), one red bowl (at 4), all at one event. A red mug shares the first level with its five records. A
    blue mug's first level holds four golds, a white mug's five golds from two scripts: both fall to kind and event, the
    14 mugs. A mug of another event finds the mugs at the kind (event dropped) level. A glass finds the whole template,
    15 records; another template, nothing. Regression caught: `blind` with nothing to say about a description it has
    never been asked, or one that matches too wide a pool first."""
    prior = [*many("mug", "red", 1.0, [f"r{i}" for i in range(5)]),
             *many("mug", "blue", 2.0, [f"b{i}" for i in range(4)]),
             *many("mug", "white", 3.0, ["w0", "w1", "w0", "w1", "w0"]), *many("bowl", "red", 4.0, ["c0"])]

    def found(**kw):
        return [p["gold"]["gold"]["xy"][0] for p in pool_at("blind", lookup(**kw), prior)]

    mugs = [1.0] * 5 + [2.0] * 4 + [3.0] * 5
    assert found(kind="mug", color="red") == [1.0] * 5
    assert found(kind="mug", color="blue") == mugs
    assert found(kind="mug", color="white") == mugs
    assert found(kind="mug", color="green") == mugs
    assert found(kind="mug", color="green", event="when you got home") == mugs
    assert found(kind="glass", color="red") == [*mugs, 4.0]
    assert found(kind="glass", color="red", template="path_ahead") == []


def test_a_level_with_too_few_records_is_no_pool_and_blind_abstains(tmp_path):
    """Contract (VISION rule 5, review F4): a prior answers only from a level whose records come from at least three
    scripts and number at least five golds, else `unknown`: an answer learned from one or two worlds scores under
    abstaining on the next (on the look-dev scripts `blind` scored -0.29 against `all_unknown`'s +0.29, from singleton
    pools). Hand pools of one question: five golds from three scripts are a pool; four golds from four scripts, and five
    from two, are not (nothing coarser holds more), nor is an empty prior. `blind` over a lone exact record gives
    `unknown`, where it gave that record's gold before, and over five records from five scripts at the question's own
    key gives their gold. Regression caught: a baseline that answers from a one-record pool, and so passes its
    shortcut gate by scoring under the floor."""
    q = lookup("mug", "red")

    def pool(*scripts):
        return pool_at("blind", q, many("mug", "red", 1.0, scripts))

    assert len(pool("s0", "s1", "s2", "s0", "s1")) == 5
    assert pool("s0", "s1", "s2", "s3") == []
    assert pool("s0", "s1", "s0", "s1", "s0") == []
    assert pool_at("blind", q, []) == []
    ask = where("mug", "red", EVENT)
    now = {"k": 0, "clock": ""}
    lone = started(Blind, tmp_path / "lone", prior=many("mug", "red", 1.0, ["s0"]))
    five = started(Blind, tmp_path / "five", prior=many("mug", "red", 1.0, [f"s{i}" for i in range(5)]))
    assert lone.ask("end", now, [ask]) == {"q1": UNKNOWN}
    assert five.ask("end", now, [ask]) == {"q1": at(1.0, 1.0)}


def test_blind_ignores_observations_and_the_clock_slots(tmp_path):
    """Contract: `blind` never uses a stream (observe changes nothing) and a slot named `clock` or `clocks` is no
    part of the key (clocks never repeat across cases). Hand case: five records from five scripts whose slots hold one
    clock and a question that holds another get the pool of all five: the distance 2.2 each, so 2.2, before and after
    an observe. Regression caught: a blind baseline that overfits clocks or keeps state."""
    q = {**where("mug", "red", EVENT), "template": "when_last"}
    q["slots"] = {**q["slots"], "clock": "2026-03-02T08:00:00"}
    prior = [record(q, number(2.2), script=f"s{i}") for i in range(5)]
    ask = {**q, "slots": {**q["slots"], "clock": "2026-05-14T09:30:00"}}
    system = started(Blind, tmp_path, prior=prior)
    assert system.ask("end", {"k": 0, "clock": ""}, [ask]) == {"q1": number(2.2)}
    system.observe([step(0, 1.0, 1.0)])
    assert system.ask("end", {"k": 0, "clock": ""}, [ask]) == {"q1": number(2.2)}
    assert system.ask_record(ask, {"k": 0, "clock": ""}, when="end") == {
        "template": "when_last", "slots": ask["slots"], "when": "end", "clock": ""}
    assert started(Blind, tmp_path / "none", prior=[]).ask("end", {"k": 0, "clock": ""}, [ask]) == {"q1": UNKNOWN}


YES, NO = {"type": "yesno", "value": True}, {"type": "yesno", "value": False}


def votes(*values: bool, scripts: tuple[str, ...] = ()) -> list[dict]:
    """Prior records of yes/no golds (exact credit, so a right answer is +1 and a wrong one -1) of the question `where
    ("mug", "red", EVENT)`, one a value, from the scripts named (default one script each)."""
    names, out = scripts or tuple(f"s{i}" for i in range(len(values))), []
    for value, script in zip(values, names, strict=True):
        r = record(where("mug", "red", EVENT), YES if value else NO, script=script)
        out.append({**r, "gold": {**r["gold"], "credit": "exact", "tolerances": {}, "metric": None}})
    return out


def test_a_split_pool_wins_in_the_pool_and_loses_on_the_script_it_left_out():
    """Contract (VISION rule 5: the shortcut is the best answer learned from other cases, which has to beat abstaining
    on the next case): `learned_answer` keeps the best response only when the same rule, leaving each script out and
    answering its golds from the others', beats `unknown` there. Hand pool, five scripts' yes/no golds: yes, yes, yes,
    no, no. In the pool `yes` scores (3 - 2) / 5 = +0.2 against `unknown`'s 0, so `best_response` answers yes. Leaving
    out a yes script the other four split 2-2: yes and `unknown` tie at 0 and `unknown`, first, wins, so 0 on that
    gold, three times; leaving out a no script the other four are 3-1 for yes, which is -1 on the left-out gold, twice:
    (0 + 0 + 0 - 1 - 1) / 5 = -0.4, under abstaining's 0, so `unknown`. Regression caught: a learned shortcut that
    answers with the winner of its own pool, which on the thin pools of the dev sets scored 0.1 to 0.7 under
    all-`unknown` (the `BROKEN` instrument lines)."""
    pool = votes(True, True, True, False, False)
    assert best_response([p["gold"] for p in pool]) == YES
    assert held_out_mean(pool) == pytest.approx(-0.4)
    assert learned_answer(pool) == UNKNOWN


def test_a_clear_majority_holds_on_the_script_left_out_and_is_answered():
    """Contract: a pool whose best response generalises is answered, so a shortcut that has something to learn from
    other scripts still learns it (a leak the shortcut gate must catch). Hand pool, ten scripts: yes nine times, no
    once. Leaving out a yes script the other nine are 8-1 for yes (+1, nine times); leaving out the no script they are
    9-0 (-1, once): (9 - 1) / 10 = +0.8 above abstaining's 0. A resample of ten scripts holds the no script five times
    or more (a gain of 0 or less) with probability 0.16%, so the interval's lower end is above 0 and the answer is yes.
    A pool of five yes is +1 and yes. Regression caught: a held-out rule so strict that it abstains on every pool, which
    would leave the shortcut gates vacuous."""
    pool = votes(*[True] * 9, False)
    assert held_out_mean(pool) == pytest.approx(0.8)
    assert learned_answer(pool) == YES
    assert learned_answer(votes(True, True, True, True, True)) == YES


def test_the_golds_of_one_script_are_held_out_together():
    """Contract: a script is one world, so its golds are left out together; they cannot vouch for each other. Hand pool,
    four scripts and six golds: script a holds three yes (the same fact asked three ways), b one yes, c and d one no
    each. In the pool yes scores (4 - 2) / 6 = +1/3. Leaving out a the others are yes, no, no: `no` is +1/3 and wins, -1
    on each of a's three golds; leaving out b they are 3-2 for yes (+1); leaving out c or d they are 4-1 for yes (-1
    each): (-3 + 1 - 1 - 1) / 6 = -2/3, so `unknown`. Left out one gold at a time, a's other two golds would vouch for
    the third and the sum would be (3 + 1 - 1 - 1) / 6 = +1/3, an answer. Regression caught: a held-out round that
    leaves out a gold and not its script, so that the repeats of one world pass for a fact of the world."""
    pool = votes(True, True, True, True, False, False, scripts=("a", "a", "a", "b", "c", "d"))
    assert best_response([p["gold"] for p in pool]) == YES
    assert held_out_mean(pool) == pytest.approx(-2 / 3)
    assert learned_answer(pool) == UNKNOWN


def test_blind_answers_through_the_held_out_rule(tmp_path):
    """Contract: `blind` over a pool is `learned_answer` of it, not its best response. Hand pools of the asked
    question's own key: yes, yes, yes, no, no from five scripts is the 3-2 split above, so `unknown`; yes nine times and
    no once is the clear majority above, so yes. Regression caught: `blind` still answering from the in-pool winner."""
    ask, now = where("mug", "red", EVENT), {"k": 0, "clock": ""}
    split = started(Blind, tmp_path / "split", prior=votes(True, True, True, False, False))
    clear = started(Blind, tmp_path / "clear", prior=votes(*[True] * 9, False))
    assert split.ask("end", now, [ask]) == {"q1": UNKNOWN}
    assert clear.ask("end", now, [ask]) == {"q1": YES}


def test_an_in_pool_prior_keeps_the_pools_own_winner(tmp_path):
    """Contract (VISION rule 5): the held-out round of `learned_answer` abstains where a fair learner would answer, so
    `check` also scores each learned shortcut under the pool's own winner (`Prior(in_pool=True)`, `Pool.in_pool`) and a
    gate reads the larger score. Hand pool, the 3-2 split above (yes, yes, yes, no, no from five scripts): in the pool
    `yes` scores +.2 and the held-out rule's -.4 abstains, so `learned_answer` of the plain pool is `unknown` and of the
    `in_pool` one `yes`, `best_response`'s; `blind` and `questionnaire_blind` over the same records in a plain prior
    abstain and in an `in_pool` prior answer yes. An empty pool is `unknown` under either. Regression caught: a twin
    that applies the same rule as the original, which makes the gate's larger-of-two score vacuous."""
    pool = votes(True, True, True, False, False)
    assert best_response([p["gold"] for p in pool]) == YES
    assert learned_answer(Pool(pool)) == UNKNOWN and learned_answer(Pool(pool, in_pool=True)) == YES
    assert learned_answer(Pool((), in_pool=True)) == UNKNOWN
    ask, now = where("mug", "red", EVENT), {"k": 0, "clock": ""}
    held = started(Blind, tmp_path / "held", prior=Prior(pool))
    twin = started(Blind, tmp_path / "twin", prior=Prior(pool, in_pool=True))
    assert held.ask("end", now, [ask]) == {"q1": UNKNOWN} and twin.ask("end", now, [ask]) == {"q1": YES}
    for rule, want in ((False, UNKNOWN), (True, YES)):   # `questionnaire_blind` reads the prior's rule as well
        system = started(QuestionnaireBlind, tmp_path / f"q{rule}", prior=Prior(pool, in_pool=rule))
        assert system.ask("end", now, [ask]) == {"q1": want}


def test_every_learned_shortcut_answers_through_learned_answer():
    """Contract: the baselines that learn from other scripts' records and the form adversary answer through
    `blind.learned_answer`; none calls the in-pool `best_response`, which only `blind` itself may (it is the rule's
    inner search). Regression caught: a new answerer, or an old one edited, that picks its pool's winner directly, and
    so leaves the shortcut gate to a shortcut that overfits the pool."""
    callers, learners = [], []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.attr if isinstance(n, ast.Attribute) else n.id if isinstance(n, ast.Name) else n.name
                 for n in ast.walk(tree) if isinstance(n, ast.Attribute | ast.Name | ast.alias)}
        if path.name != "blind.py" and "best_response" in names:
            callers.append(path.name)
        if "learned_answer" in names:
            learners.append(path.name)
    assert callers == []
    assert {"baselines.py", "baselines_path.py", "form_check.py"} <= set(learners)
