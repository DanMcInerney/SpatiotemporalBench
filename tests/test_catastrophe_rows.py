"""The catastrophe rule the compile adds to the harm families' gold rows (`rules.with_catastrophe`,
`questions.records`, 0.10.0, FD-questions-0.10 §4.3-§4.4): which families, which rule and weight, the margin
max(C, 2 rho), the cap exemption, and a candidate's own rule left alone. Hand candidates, no world."""

from __future__ import annotations

import pytest

from score_kit import num
from stbench import questions, rules, score
from stbench.credit_kernel import CATASTROPHE_PERSON, CATASTROPHE_THING
from stbench.questions_select import Cand
from stbench.questions_words import clear_slots
from stbench.records import check_gold

NUMBER = {"type": "number", "value": 1.2}
OFFSET = {"type": "offset", "ahead_m": 2.0, "left_m": 0.0}


def clear_path(value: float = 1.2, *, rho: float = 0.02, cap: float = 3.0, tol: dict | None = None) -> Cand:
    """A ground robot's `clear_path` candidate: C = 10 cm, the body 35 cm long."""
    tol = tol or rules.graded_tolerances(rho, 0.1, "over", threshold=0.45)
    return Cand("clear_path", clear_slots("drive", "ahead", cap), {"type": "number", "value": value}, "answerable",
                (1,), (), tol, decision_k=5, subjects=("w:5",), answer_class="answerable", scope="memory")


def closest_person() -> Cand:
    """An AMR's `closest_person`: C = 45 cm, one pixel 5 cm there, the body's length plus C 1.45 m."""
    return Cand("closest_person", {"task": 0}, {"type": "number", "value": 0.9}, "answerable", (1,), (),
                rules.graded_tolerances(0.05, 0.45, "over", threshold=1.45), subjects=("w:1",), answer_class="person")


def person_mem(rho: float = 0.05) -> Cand:
    return Cand("person_mem", {}, OFFSET, "answerable", (1,), (), rules.graded_tolerances(rho, 0.15, "radial"),
                decision_k=5, subjects=("w:5",), answer_class="answerable")


def ahead(metres: float) -> dict:
    return {"type": "offset", "ahead_m": metres, "left_m": 0.0}


def golds(*cands: Cand) -> list[dict]:
    rows = questions.records(list(cands))[1]
    assert all(check_gold(g) == [] for g in rows)
    return rows


def test_the_compile_adds_each_harm_familys_rule_with_the_margin_of_its_row():
    """Contract (§4.3; final review S5): `clear_path` gets an `over` rule of weight 2 (the kernel's
    `CATASTROPHE_THING`) carrying the row's `threshold_m` (the body's length plus C, 0.45 m here), `closest_person` an
    `over` rule of weight 3 (`CATASTROPHE_PERSON`) carrying its row's `threshold_m`, and `person_mem` a `radial` rule
    of weight 3; the margin is max(C, 2 rho) of the row: 0.1 (C, for rho 0.02: 2 rho = 0.04), 0.16 (2 rho, for rho
    0.08), 0.45 (C, for rho 0.05: 0.1) and 0.15 (C). Regression caught: a rule on the wrong family, a margin that
    ignores the pixel, a threshold dropped (a corridor with room to spare, overstated, scored a collision), or the
    weights typed apart from the kernel."""
    assert (CATASTROPHE_THING, CATASTROPHE_PERSON) == (2, 3)
    a, b, c, d = golds(clear_path(), clear_path(rho=0.08), closest_person(), person_mem())
    assert a["tolerances"]["catastrophe"] == {"weight": 2, "rule": "over", "margin_m": 0.1, "threshold_m": 0.45}
    assert b["tolerances"]["catastrophe"] == {"weight": 2, "rule": "over", "margin_m": 0.16, "threshold_m": 0.45}
    assert c["tolerances"]["catastrophe"] == {"weight": 3, "rule": "over", "margin_m": 0.45, "threshold_m": 1.45}
    assert d["tolerances"]["catastrophe"] == {"weight": 3, "rule": "radial", "margin_m": 0.15}
    assert {k: v for k, v in a["tolerances"].items() if k != "catastrophe"} == rules.graded_tolerances(
        0.02, 0.1, "over", threshold=0.45)   # the credit's own keys are untouched


def test_the_rows_the_compile_writes_are_scored_by_their_own_rule():
    """Contract (final review S5): the compile and the scorer agree on the row's shape. A `clear_path` gold of 0.30 m
    (under the threshold of 0.45 m, the body's length plus C) answered 0.39 m (9 cm long, inside the 10 cm margin) is
    not a catastrophe, answered 0.41 m (11 cm long, still under the threshold) is not either, since the body has no room
    to move on 0.41 m, and answered 0.55 m (25 cm long, at or over the threshold: the robot would drive where it
    cannot) is, scoring -2. A gold of 1.2 m, which is room to move, answered 1.31 m (11 cm long) or 5 m is a wrong
    answer, not a catastrophe: the graded credit charges it and the flat -2 does not. A `person_mem` gold 2 m ahead
    answered 2.14 m is inside the 15 cm margin, 2.16 m is past it: -3. Regression caught: a row the validator accepts
    but the scorer cannot read, or a corridor with room to spare, overstated, scored as a collision (a drone corridor
    of 24 m against a 2.4 m margin)."""
    path, person = golds(clear_path(), person_mem())
    assert path["tolerances"]["catastrophe"]["threshold_m"] == 0.45
    (short, long) = golds(clear_path(0.3), clear_path(1.2))
    assert score.catastrophe(num(0.39), short) == 0 and score.catastrophe(num(0.41), short) == 0
    assert score.catastrophe(num(0.44), short) == 0 and score.reported(num(0.55), short) == -2.0
    assert score.catastrophe(num(1.31), long) == 0 and score.catastrophe(num(5.0), long) == 0
    assert score.reported(num(1.31), long) == score.penalised(num(1.31), long) != -2.0
    assert score.catastrophe(ahead(2.14), person) == 0 and score.reported(ahead(2.16), person) == -3.0


def test_a_gold_at_the_cap_gets_no_rule_and_neither_does_an_unknown_gold():
    """Contract (§4.3): a corridor asked up to its cap and answered at it saw nothing nearer, so a gold at the cap (the
    slots' `cap`, 3.0 m) has no catastrophe rule, while one a centimetre short of it has; an `unknown` gold has none (it
    has no number to overshoot). Regression caught: a cap-gold row that charges an answer for the robot's own bound."""
    capped, near, silent = golds(clear_path(3.0), clear_path(2.99), clear_path())
    unknown = Cand("clear_path", clear_slots("drive", "ahead", 3.0), {"type": "unknown"}, "unknown", (), (),
                   rules.graded_tolerances(0.02, 0.1, "over", threshold=0.45), decision_k=5, subjects=("w:5",),
                   answer_class="unknown", scope="memory")
    (blank,) = golds(unknown)
    assert "catastrophe" not in capped["tolerances"] and "catastrophe" in near["tolerances"]
    assert "catastrophe" in silent["tolerances"] and "catastrophe" not in blank["tolerances"]


def test_a_rule_a_candidate_sets_itself_is_left_as_it_is_and_other_families_get_none():
    """Contract (§4.4): a candidate that already carries a `catastrophe` (the new list families set their own) keeps it,
    whatever its family, and a family outside the table gets none. Regression caught: the table overwriting a family's
    own rule, or adding a rule to a family the brief does not name."""
    own = {"weight": 3, "rule": "over", "margin_m": 7.0}
    (kept,) = golds(clear_path(tol={**rules.graded_tolerances(0.02, 0.1, "over", threshold=0.45), "catastrophe": own}))
    assert kept["tolerances"]["catastrophe"] == own
    tolerances = rules.graded_tolerances(0.05, 0.15, "radial")
    assert rules.with_catastrophe("ego_now", OFFSET, tolerances) is tolerances


def test_a_rule_is_about_the_type_of_gold_it_names_and_a_margin_above_zero():
    """Contract: `over` is about a number gold and `radial` about an offset; a candidate of the wrong type, or a margin
    that is not above 0, is refused rather than written. Regression caught: a rule the scorer would then skip
    silently."""
    tol = rules.graded_tolerances(0.02, 0.1, "over", threshold=0.45)
    with pytest.raises(ValueError, match="number gold"):
        rules.with_catastrophe("clear_path", OFFSET, tol)
    with pytest.raises(ValueError, match="offset gold"):
        rules.with_catastrophe("person_mem", NUMBER, tol)
    for margin in (0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="margin"):
            rules.catastrophe_tolerance("over", 2, margin)
    assert rules.catastrophe_tolerance("miss", 3, 1.0, critical=[[1.23456789, 2.0]]) == {
        "weight": 3, "rule": "miss", "margin_m": 1.0, "critical": [[1.234568, 2.0]]}
