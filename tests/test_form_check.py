"""The form adversary (`form_check`, VISION rule 6): the features of a delivered question and the best single-feature
rule per family, scored on hand-made questions and gold, with no file read."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench import score_stats
from stbench.form_check import Asked, features_of, form_adversary
from stream_kit import UNKNOWN, during_q, end_q, gold_row

SCRIPTS = ("s1", "s2", "s3", "s4", "s5", "s6", "s7")   # seven worlds: a pool is the others' six


def at(x: float, y: float) -> dict:
    return {"type": "position", "xy": [x, y]}


def number(v: float) -> dict:
    return {"type": "number", "value": v}


def asked(script: str, question: dict, gold: dict, **features) -> Asked:
    """A question of the script's case, with the gold value (`unknown` when it is) and the features given."""
    unknown = {"slice": "unknown"} if gold == UNKNOWN else {}
    info = {"case": f"c-{script}", "script": script, "layout": "apartment_a", "robot": "vacuum", "job": "vacuum/1",
            "length": "short"}
    return Asked(f"c-{script}", script, question, gold_row(question, gold=gold, **unknown), info, features)


def test_the_features_of_a_delivered_question_by_hand():
    """Contract (plan §4.8): a question's features are what it is delivered with and where the ask is: its text's
    length in twelves of characters, answer type and `when`; the case's steps since its task began at the step asked
    about, its index in the questionnaire and that step's phase; and a walk over its slots, `slot:<name>` for a plain
    value, `slot:<name>.<key>` for each key of an object, a JSON string for a list. Hand values: 47 characters are 3
    twelves. Nothing of the qid, the template or the `needs` is a feature. Regression caught: a feature that names the
    qid or the template (a random id that a rule could split on) or one that is not hashable."""
    q = end_q(7, text="x" * 47, slots={"kind": "mug", "attributes": {"color": "red"}, "event": "when you left the dock",
                                       "cap": 3, "clocks": ["08:00", "09:00"]})
    assert features_of(q, k_in_task=12, place=2, phase="cleaning") == {
        "text_length": 3, "answer_type": "position", "when": "end", "k_in_task": 12, "place": 2, "phase": "cleaning",
        "slot:kind": "mug", "slot:attributes.color": "red", "slot:event": "when you left the dock", "slot:cap": 3,
        "slot:clocks": '["08:00", "09:00"]'}


def test_the_form_adversary_learns_from_other_scripts_only():
    """Contract (VISION rule 6, map X-H14): each question is answered from the gold of the questions of OTHER scripts
    that share a feature's value (under the support floor: five golds of three scripts), and a feature is named only
    when its rule beats all-`unknown`. Hand values, six scripts of two questions each, so that the five others' golds
    make a pool. `path_ahead`: the first question of each script has the gold 1.4 and the second an unknown gold; the
    feature `place` (0 or 1) carries that across scripts, so its rule answers every question right (mean 1.0 against an
    unknown share of 0.5) and is named; `tag`, the script's own id, shares no value with another script, so its rule
    answers `unknown` everywhere (0.5, the unknown share). `object_where`: both questions of a script have the same
    position, a different one in each script, so the feature `tag` would give it away if the adversary read its own
    script's gold, and `place` has nothing across scripts that beats `unknown`: the table's feature is None at 0.0, and
    no row scores above that. Regression caught: the adversary learning from the case it
    scores (its own script's questions in the pool), or naming a feature that predicts nothing."""
    six = SCRIPTS[:6]
    ahead = [asked(s, during_q(n, 12), gold, tag=s, place=place)
             for s in six for n, (place, gold) in enumerate([(0, number(1.4)), (1, UNKNOWN)])]
    where = [asked(s, end_q(n, slots={"kind": "mug"}), at(10.0 * i, 10.0 * i), tag=s, place=place)
             for i, s in enumerate(six, 1) for n, place in enumerate((0, 1))]
    rows, table = form_adversary(ahead + where)
    assert table["path_ahead"] == {"feature": "place", "mean": 1.0, "unknown_share": 0.5, "n": 12,
                                   "means": {"tag": 0.5, "place": 1.0}}
    assert table["object_where"] == {"feature": None, "mean": 0.0, "unknown_share": 0.0, "n": 12,
                                     "means": {"tag": 0.0, "place": 0.0}}
    by_family = {f: [r["score"] for r in rows if r["family"] == f] for f in table}
    assert by_family == {"path_ahead": [1.0] * 12, "object_where": [0.0] * 12}
    assert all(set(r) == set(score_stats.ROW_KEYS) for r in rows)
    assert {r["script"] for r in rows} == set(six)


def test_a_group_of_one_script_gives_the_adversary_nothing_to_learn_from():
    """Contract (map X-H14): with every question from one script no pool is ever non-empty, so every feature's rule is
    all-`unknown` and none is named, whatever the golds repeat. Regression caught: a pool that includes the script's own
    questions, which makes a repeated gold look like a predictable one."""
    one = [asked("s1", end_q(n, slots={"kind": "mug"}), at(1, 1), tag="s1", place=n) for n in range(3)]
    rows, table = form_adversary(one)
    assert table["object_where"]["feature"] is None and table["object_where"]["mean"] == 0.0
    assert [r["score"] for r in rows] == [0.0] * 3


def test_a_question_without_a_feature_has_the_value_none():
    """Contract: a feature some questions lack (an attribute only some have) is a feature whose value is None for the
    others, so the rule on it pools them with each other and never with a question that has a value. Hand values: six
    scripts ask a question with `colour` red and a seventh asks without; the golds of the six red ones are the same
    place and the colourless one's is another, far off. The feature's rule answers the six red ones with the other
    five red ones' place (score 1, five golds of five scripts) and the colourless one with `unknown` (no other lacks
    it: 0): mean 6/7 against an unknown share of 0, so it is named. Regression caught: a missing feature raising, or
    a question without it pooled with questions that have one."""
    red = [asked(s, end_q(n, slots={"kind": "mug"}), at(1, 1), colour="red") for n, s in enumerate(SCRIPTS[:6])]
    plain = [asked(SCRIPTS[6], end_q(9, slots={"kind": "mug"}), at(30, 30))]
    _, table = form_adversary(red + plain)
    assert table["object_where"]["feature"] == "colour"
    assert table["object_where"]["mean"] == pytest.approx(6 / 7)


def test_the_form_adversary_answers_a_pool_only_when_it_holds_on_the_script_left_out():
    """Contract (`blind.learned_answer`, VISION rule 5): the adversary is a learned shortcut like the others, so a pool
    is answered only when its best response beats abstaining on a script it did not learn from. Hand case: six scripts
    ask one question each, every question of one form, with the yes/no golds yes, yes, yes, no, no, no (exact credit).
    A yes script's pool is the five others, 2 yes and 3 no, so in the pool `no` wins (+0.2) and is wrong on the
    question (-1); a no script's is 3 yes and 2 no, `yes` wins and is wrong. Held out, a pool of 2 yes and 3 no
    answers `no` when a yes script is left out (3-1, -1, twice) and ties at 0 when a no script is (2-2, `unknown`,
    three times): (-2 + 0) / 5 = -0.4, and the mirror pool the same, so every pool is `unknown` and every row is 0,
    where the in-pool winner scored -1 on every one. Regression caught: a form adversary that answers with its pool's
    winner, which on a thin pool is worse than abstaining and can only be called a leak by luck."""
    golds = [{"type": "yesno", "value": v} for v in (True, True, True, False, False, False)]
    votes = []
    for n, (script, gold) in enumerate(zip(SCRIPTS[:6], golds, strict=True)):
        a = asked(script, end_q(n, slots={"kind": "mug"}), at(1, 1), f=1)
        votes.append(replace(a, gold=gold_row(a.question, gold=gold, credit="exact", tolerances={}, metric=None)))
    rows, table = form_adversary(votes)
    assert [r["score"] for r in rows] == [0.0] * 6
    assert table["object_where"]["feature"] is None and table["object_where"]["mean"] == 0.0
