"""What the reference agent is told of a heading answer (`llm_render`, `llm_credit.HEADING_TARGET`; the answer type is
`answers`', lane I0): the shape it is shown must be one the scorer's parser accepts, in the range the prompt states."""

from __future__ import annotations

import json
import re

import pytest

from stbench.answers import PROMPT_SHAPES, TYPES, Malformed, answer_type_of, parse_answer
from stbench.llm_credit import HEADING_TARGET
from stbench.llm_render import ask_message, labelled, system_prompt
from test_llm_render import public


def example(degrees) -> dict:
    """The shape's own JSON object with its placeholder replaced by `degrees`."""
    shape = PROMPT_SHAPES["heading"]
    text = re.search(r"\{.*?\}", shape).group(0)
    return json.loads(re.sub(r'("deg_left":\s*)\w+', lambda m: f"{m.group(1)}{json.dumps(degrees)}", text))


def test_the_shown_heading_shape_parses_in_the_range_the_prompt_states():
    """Contract (plan P2): the heading shape the prompt shows is a JSON object the parser reads as a heading, and the
    range the heading definition states, -180 to 180 degrees to the left, is the range the parser accepts; just beyond
    it, a word and a missing field are malformed. Regression caught: an answer shape the scorer refuses (every heading
    scored as malformed), or a prompt that states a wider range than the parser takes."""
    assert "heading" in TYPES and "from -180 to 180" in HEADING_TARGET
    for degrees in (0, 30.5, -90, 180, -180):
        assert answer_type_of(parse_answer(example(degrees))) == "heading"
    for bad in (180.5, -181, "left"):
        assert isinstance(parse_answer(example(bad)), Malformed)
    assert isinstance(parse_answer({"type": "heading"}), Malformed)


def test_a_heading_question_is_shown_with_its_type_and_shape_and_the_prompt_lists_the_shape(tmp_path):
    """Contract (plan P2): an ask shows a heading question's text, its answer type `heading` and the shape of
    `PROMPT_SHAPES`, and the system prompt lists the shape among the kinds of answer, so the agent knows the form
    before the first question. Regression caught: a heading question asked in a shape the prompt never taught."""
    questions = labelled([{"qid": "x", "answer_type": "heading",
                           "text": "Which way would let your body go farthest?"}])
    message = ask_message("during", {"k": 3, "clock": "c"}, questions)
    assert f"Answer type: heading. answer = {PROMPT_SHAPES['heading']}" in message
    assert f"- heading: {PROMPT_SHAPES['heading']}" in system_prompt(public(tmp_path), "rgb_pose")


@pytest.mark.parametrize("headings", [False, True])
def test_the_heading_definition_is_in_the_prompt_only_for_a_job_that_asks_one(tmp_path, headings):
    """Contract (plan P2): with the job's numbers and `headings`, the prompt holds `HEADING_TARGET` once, after the
    corridor's definition; without it, not at all. Regression caught: a definition shown to a job that never asks a
    heading, or left out of one that does."""
    prompt = system_prompt(public(tmp_path), "rgb_pose", match_m=0.5, clearance_m=0.1, reacquire_m=1.0,
                           headings=headings)
    assert prompt.count(HEADING_TARGET) == int(headings)
    if headings:
        assert prompt.index("Straight back, count the distance its rear edge moves.") < prompt.index(HEADING_TARGET)
