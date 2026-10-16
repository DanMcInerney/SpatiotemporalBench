"""The numbers the credit section gives for the way to a remembered thing and for the room the body needs (final review
S1, S2 of 0.11): each is read back out of the words and handed to the scorer, so the prompt cannot drift from what
`score_graded` does with the same row. Expected values are worked by hand."""

from __future__ import annotations

import re

import pytest

from score_kit import rec
from stbench import grid
from stbench.llm_credit import credit_sentence
from stbench.score import credit, penalised

# the row `questions_way` writes: the outcome of each of 4 headings, 90 degrees apart (the best, a shortfall of the
# margin, a shortfall of exactly one cell, and a heading with no room), and the tolerances from the helper's numbers
STEP_M = 1.0


def metres(text: str) -> float:
    """"20 cm" or "1.2 m" as metres."""
    number, unit = re.fullmatch(r"([\d.]+) (cm|m)", text).groups()
    return float(number) / (100 if unit == "cm" else 1)


def way_row(c: float, profile: list[float | None]) -> dict:
    return rec({"type": "heading", "deg_left": 0.0}, credit="route", family="way_to",
               tolerances={"profile_m": profile, "step_deg": 90, "step_m": STEP_M, "resolution_m": grid.FINE_M,
                           "need_m": c})


def heading(deg: float) -> dict:
    return {"type": "heading", "deg_left": deg}


@pytest.mark.parametrize(("c", "margin"), [(0.1, "20 cm"), (0.15, "20 cm"), (0.45, "45 cm")])
def test_the_way_to_bullets_state_the_margin_and_the_exactness_the_scorer_applies(c, margin):
    """Contract (final review S1): the margin of `way_to` is the largest of C, a tenth of R and twice the field's cell,
    and its quarter 4 is whole within one cell, as the words say, read out of the words and handed to the scorer. By
    hand with R = 1 m and a cell of 10 cm: C = 10 cm and 15 cm give 20 cm (twice the cell wins), C = 45 cm gives 45 cm.
    A heading short of the best by exactly the printed margin has quarters 1, 1, 1 and 0: s = 3/4, a score of 0.5 (the
    sentence says an answer off by the margin scores 0.5); one short by exactly the printed exactness has all four
    quarters: 1; one a millimetre beyond it has less than 1. Regression caught: a margin typed as C (the 0.11.0 words
    said 15 cm where the scorer used 20), a quarter 4 stated as the camera's pixel, or the two drifting apart."""
    sentence = credit_sentence(0.5, c, STEP_M, families=("way_to",), kinds=("heading",))
    said = re.search(r"Its margin is ([\d. cm]+): the largest of", sentence)
    exact = re.search(r"Quarter 4 is whole within ([\d. cm]+) of the best heading's progress", sentence)
    assert said and said[1] == margin and exact and exact[1] == "10 cm"
    margin_m, exact_m = metres(said[1]), metres(exact[1])
    assert exact_m == grid.FINE_M
    best = 2.0   # a profile of 4 headings: the best at 0, then the two shortfalls and a heading without room
    row = way_row(c, [best, best - margin_m, best - exact_m - 0.001, None])
    assert credit(heading(0), row) == pytest.approx(1.0)
    assert penalised(heading(90), row) == pytest.approx(0.5)
    assert 0.5 < penalised(heading(180), row) < 1.0
    row = way_row(c, [best, best - exact_m, None, None])
    assert penalised(heading(90), row) == pytest.approx(1.0)


def test_a_shortfall_of_15_cm_scores_what_the_bullets_imply():
    """Contract (final review S1, by hand): on the helper (C = 15 cm, R = 1 m, cell 10 cm) the bullets say a margin of
    20 cm, quarter 4 whole within 10 cm and falling smoothly to nothing at the margin, and the right direction for any
    heading with room. A heading short of the best by 15 cm: quarters 1 and 2 and 3 whole (15 cm is within the margin),
    quarter 4 = 1 - ln(15 / 10) / ln(20 / 10) = 1 - 0.58496 = 0.41504; s = (1 + 1 + 1 + 0.41504) / 4 = 0.85376 and the
    penalised score 2s - 1 = 0.70752. (The 0.11.0 words, a margin of 15 cm, would imply 0.5.) Regression caught: a
    scorer or a sentence that moves the margin, the exactness or the logarithmic falloff of quarter 4."""
    sentence = credit_sentence(0.5, 0.15, STEP_M, families=("way_to",), kinds=("heading",))
    assert "Its margin is 20 cm: the largest of 15 cm, a tenth of 1 m and twice the 10 cm cell" in sentence
    assert ("Quarter 4 is whole within 10 cm of the best heading's progress, falling smoothly to nothing at the "
            "margin") in sentence
    row = way_row(0.15, [1.0, 0.85, None, None])
    assert penalised(heading(90), row) == pytest.approx(0.70752, abs=1e-5)


def test_any_heading_with_room_has_the_direction_quarter_and_the_room_is_the_length_plus_c():
    """Contract (final review S1, S2): the way-to bullet says any heading with the room the body needs has the right
    direction, and that room is the body's length plus C however far the best run is (the margin of the far field does
    not widen it). The scorer's `regret` threshold is that: the helper (0.34 m long, C = 15 cm) has T = 0.49. By hand,
    on a best run of 5.0 m (a margin of 50 cm, which would make the room 0.84 m if the words said 'the margin'), a run
    of 0.60 m has room to move and earns the direction alone: s = 1/4 (the regret 4.4 m is past four margins); a run of
    0.45 m has none: s = 0. For the way to a thing, a heading that makes no progress (a shortfall of 1.0 m, past four
    margins of 20 cm) has the direction alone as well: s = 1/4. Regression caught: a room stated as the length plus the
    far field's margin, or a heading with room judged as a wrong answer."""
    sentence = credit_sentence(0.5, 0.15, STEP_M, headings=True, families=("way_to", "open_heading"),
                               kinds=("heading", "number"))
    assert "has the right direction, even one that leads away from the thing" in sentence
    assert sentence.count("the robot body's length plus 15 cm") == 2   # the distance's and the way's
    assert "plus the margin" not in sentence
    runs = {"profile_m": [5.0, 0.6, 0.45, 0.0], "step_deg": 90, "resolution_m": 0.1, "need_m": 0.15,
            "threshold_m": 0.49}
    row = rec(heading(0), credit="regret", family="open_heading", tolerances=runs)
    assert credit(heading(90), row) == pytest.approx(0.25) and credit(heading(180), row) == pytest.approx(0.0)
    away = way_row(0.15, [1.0, 0.0, None, None])
    assert credit(heading(90), away) == pytest.approx(0.25)
