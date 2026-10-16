"""The `offset` answer (`answers.py`, 0.6.0): a place in the camera's ground frame, metres ahead and to the left."""

from __future__ import annotations

import pytest

from stbench.answers import PROMPT_SHAPES, Malformed, Offset, Position, Unknown, answer_type_of, parse_answer, to_json


def test_an_offset_round_trips_and_keeps_its_signs():
    """Contract: `{"type": "offset", "ahead_m": a, "left_m": l}` parses to `Offset(a, l)` with ints widened to floats,
    negative numbers (behind, right) kept, and `to_json` writes the same object back. Regression caught: the two axes
    swapped or the sign dropped on the way through (a thing behind the camera scored as one ahead of it)."""
    obj = {"type": "offset", "ahead_m": -2, "left_m": 3.5}
    assert parse_answer(obj) == Offset(-2.0, 3.5) and answer_type_of(Offset(0.0, 0.0)) == "offset"
    assert to_json(Offset(-2.0, 3.5)) == {"type": "offset", "ahead_m": -2.0, "left_m": 3.5}
    assert parse_answer(to_json(parse_answer(obj))) == Offset(-2.0, 3.5)


@pytest.mark.parametrize("obj", [
    {"type": "offset"},
    {"type": "offset", "ahead_m": 1.0},
    {"type": "offset", "left_m": 1.0},
    {"type": "offset", "ahead_m": 1.0, "left_m": 1.0, "up_m": 0.0},
    {"type": "offset", "ahead_m": 1.0, "left_m": 1.0, "xy": [0, 0]},
    {"type": "offset", "ahead_m": float("nan"), "left_m": 0.0},
    {"type": "offset", "ahead_m": 0.0, "left_m": float("inf")},
    {"type": "offset", "ahead_m": "1", "left_m": 0.0},
    {"type": "offset", "ahead_m": True, "left_m": 0.0},
    {"type": "offset", "ahead_m": None, "left_m": 0.0},
    {"type": "offset", "ahead_m": [1.0], "left_m": 0.0},
    {"type": "offset", "forward_m": 1.0, "left_m": 0.0},
])
def test_an_offset_with_a_missing_extra_or_non_finite_key_is_malformed(obj):
    """Contract: an offset has exactly `ahead_m` and `left_m`, both finite JSON numbers (a bool is none). Regression
    caught: a lenient parser crediting a typo (`forward_m`), a half-answer, a NaN or a quoted number, each of which
    would score as a real place."""
    parsed = parse_answer(obj)
    assert isinstance(parsed, Malformed) and parsed.reason


def test_an_offset_is_not_a_position_and_a_position_is_not_an_offset():
    """Contract: a question of answer type `offset` refuses a position (and the reverse) as a type mismatch, whatever
    numbers it holds; `unknown` stays allowed on both. Regression caught: a map-frame position taken for a camera-frame
    offset (or the reverse), which would score a different place by the same numbers."""
    offset, position = {"type": "offset", "ahead_m": 1.0, "left_m": 2.0}, {"type": "position", "xy": [1.0, 2.0]}
    assert parse_answer(position, "offset") == Malformed(
        "answer type 'position' does not match the question's answer_type 'offset'")
    assert parse_answer(offset, "position") == Malformed(
        "answer type 'offset' does not match the question's answer_type 'position'")
    assert parse_answer(offset, "offset") == Offset(1.0, 2.0) and parse_answer(position, "position") == Position(
        (1.0, 2.0))
    assert parse_answer({"type": "unknown"}, "offset") == Unknown()


def test_the_prompt_shape_names_the_axes_the_parser_reads_and_their_signs():
    """Contract: the shape the model is shown for an offset names its two keys and says where positive and negative
    point (ahead of the camera, to its left; behind or to the right). Regression caught: a shape that drifts from the
    parser, or leaves the sign of the left axis to be guessed."""
    shape = PROMPT_SHAPES["offset"]
    assert '"type": "offset"' in shape and '"ahead_m"' in shape and '"left_m"' in shape
    assert "to its left" in shape and "behind it or to its right" in shape
