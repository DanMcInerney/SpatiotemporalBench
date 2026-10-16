"""How an agent's `submit_answer` call is read (`agent_tools.read_submission`), on hand-written arguments: every
expected answer, reason and raw value below is written out."""

from __future__ import annotations

import pytest

from stbench.agent_tools import NEXT, SUBMIT, TOOLS, read_submission

COUNT3 = {"type": "count", "value": 3}
TIME = {"type": "time", "clock": "2026-03-02 08:00:03"}
YESNO = {"type": "yesno", "value": True}
NEGATIVE = {"type": "count", "value": -1}
UNKNOWN = {"type": "unknown"}
NOT_JSON = "the arguments of submit_answer are not valid JSON"
NOT_OBJECT = "the arguments of submit_answer must be a JSON object"
NO_ANSWER = "missing required argument: answer"
CASES = [
    pytest.param({"answer": COUNT3}, "count", (COUNT3, "", COUNT3), id="wrapped-count"),
    pytest.param({"answer": TIME}, "time", ({"type": "time", "clock": "2026-03-02T08:00:03"}, "", TIME),
                 id="time-normalised"),
    pytest.param(COUNT3, "count", (COUNT3, "", COUNT3), id="answer-without-the-wrapper"),
    pytest.param({"answer": UNKNOWN}, "position", (UNKNOWN, "", UNKNOWN), id="unknown-on-any-type"),
    pytest.param('{"answer": {"type": "count", "value": 3}}', "count", (COUNT3, "", COUNT3), id="arguments-as-text"),
    pytest.param({"answer": YESNO}, "count",
                 (None, "answer type 'yesno' does not match the question's answer_type 'count'", YESNO),
                 id="wrong-type"),
    pytest.param("{not json", "count", (None, NOT_JSON, "{not json"), id="text-that-is-not-json"),
    pytest.param({"answer": NEGATIVE}, "count", (None, "value must be a non-negative integer, got -1", NEGATIVE),
                 id="negative-count"),
    pytest.param("[1, 2]", "count", (None, NOT_OBJECT, [1, 2]), id="not-an-object"),
    pytest.param({"foo": 1}, "count", (None, NO_ANSWER, {"foo": 1}), id="no-answer-and-no-type"),
]


@pytest.mark.parametrize(("arguments", "answer_type", "expected"), CASES)
def test_a_submission_is_the_answer_the_parser_accepts_or_a_reason_and_the_raw_arguments(
        arguments, answer_type, expected):
    """Contract (V0 §8): `read_submission(arguments, answer_type)` returns `(answer, why not, raw)`. The answer is
    `arguments["answer"]` (or the arguments themselves when they carry a `type` and no `answer`; JSON text is parsed
    first), accepted by `answers.parse_answer` for the question's type and returned in its normal JSON form (a space in
    a clock becomes a T); `unknown` is an answer to any type. What is unusable (another type, text that is not JSON, a
    value the parser refuses, arguments that are not an object or hold no answer) gives `None` and the reason, and
    `raw` is always what the model submitted, unwrapped. Regression caught: a route that reads a shape the other
    refuses, a normalisation lost, or a refusal with no reason for the repair message to carry."""
    assert read_submission(arguments, answer_type) == expected


def test_the_reference_agents_tools_are_next_frame_and_submit_answer_and_no_more():
    """Contract (DECISIONS 2026-10-10, "The reference agent remembers with its own context"; the agent is offered
    no tool to help it in its task; FD-frame-pull §3.3): the tool list of the reference agent is exactly `next_frame`
    (the sensor feed, first, with no arguments and a neutral description that says nothing of memory or of what to
    keep) and `submit_answer` (the answer channel), which takes a required label and a required answer; it holds no
    memory tool and no pointing tool. Regression caught: a memory tool (`memory_*`) or a perception tool (`locate`)
    given to the headline agent again, which would measure the scaffold's memory and not the agent's own, or a
    `next_frame` whose description coaches."""
    assert [t["function"]["name"] for t in TOOLS] == ["next_frame", "submit_answer"] and TOOLS == (NEXT, SUBMIT)
    assert NEXT["function"]["parameters"] == {"type": "object", "properties": {}}
    assert NEXT["function"]["description"] == (
        "Get what the robot senses next: a task message, one camera frame, questions to answer, or word that the job "
        "is over. Call it when you are ready for the next.")
    assert SUBMIT["function"]["parameters"]["required"] == ["label", "answer"]
    assert set(SUBMIT["function"]["parameters"]["properties"]) == {"label", "answer"}
