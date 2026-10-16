"""The `heading` answer type (`answers`, 0.9.0): `{"type": "heading", "deg_left": x}`, `x` a finite number from -180 to
180, degrees to the left of the way the robot faces."""

from __future__ import annotations

import math

import pytest

from stbench.answers import PROMPT_SHAPES, TYPES, Heading, Malformed, Unknown, parse_answer, parse_gold, to_json


@pytest.mark.parametrize("deg", [0, 30, -90.5, 180, -180, 179.999])
def test_a_heading_in_range_parses_and_round_trips(deg):
    """Contract: a finite number from -180 to 180 inclusive, int or float, is a heading, and `to_json` gives back the
    object it was read from. Regression caught: a heading the question asks for refused (the end of the range), or a
    parser that changes the number."""
    answer = parse_answer({"type": "heading", "deg_left": deg}, "heading")
    assert answer == Heading(float(deg))
    assert parse_answer(to_json(answer), "heading") == answer


@pytest.mark.parametrize("obj", [
    {"type": "heading", "deg_left": 180.001}, {"type": "heading", "deg_left": -181},
    {"type": "heading", "deg_left": "30"}, {"type": "heading", "deg_left": True},
    {"type": "heading", "deg_left": None}, {"type": "heading", "deg_left": math.nan},
    {"type": "heading", "deg_left": math.inf}, {"type": "heading", "deg_left": -math.inf},
    {"type": "heading"}, {"type": "heading", "deg_left": 10, "extra": 1}, {"type": "heading", "deg": 10},
    {"type": "heading", "deg_left": [10]}])
def test_a_bad_heading_is_malformed(obj):
    """Contract (the parser never raises): a heading outside -180 to 180, a string, a boolean, NaN or infinity, a
    missing or extra key is Malformed, which the scorer credits with 0. Regression caught: an agent's bad heading scored
    as an answer (a wrapped angle, a number read out of a string)."""
    assert isinstance(parse_answer(obj, "heading"), Malformed)


def test_a_heading_answers_only_a_heading_question_and_unknown_always():
    """Contract: with the question's `answer_type`, a heading answer to a number question (and a number to a heading
    question) is Malformed with the mismatch, and `unknown` is accepted for a heading question. Regression caught: a
    heading scored against a number gold, or `unknown` refused where the family allows it."""
    assert isinstance(parse_answer({"type": "heading", "deg_left": 10}, "number"), Malformed)
    assert isinstance(parse_answer({"type": "number", "value": 10}, "heading"), Malformed)
    assert parse_answer({"type": "unknown"}, "heading") == Unknown()


def test_the_prompt_lists_the_heading_and_a_gold_heading_parses():
    """Contract: `TYPES` and `PROMPT_SHAPES` name `heading` (the prompt and the agent's `submit_answer` read them, so
    no other file needs it) and a gold of the type parses with `parse_gold`. Regression caught: a type the families may
    name that the prompt never shows an agent."""
    assert "heading" in TYPES and set(PROMPT_SHAPES) == {*TYPES, "unknown"}
    assert '"deg_left"' in PROMPT_SHAPES["heading"]
    assert parse_gold({"type": "heading", "deg_left": -45}) == Heading(-45.0)
