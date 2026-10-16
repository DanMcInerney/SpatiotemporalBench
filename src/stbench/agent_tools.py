"""What every agent of the benchmark shares of the tool loop: the `next_frame` and `submit_answer` tools, the tool
list of the reference agent, the texts a refused or repeated submission is answered with, the error that stops a run
whose context is full, and how a submission is read.

The reference agent's tools are `next_frame` and `submit_answer`, on every channel. `next_frame` is the robot's sensor
feed (what comes next: a task message, one frame, the questions of an ask, or word that the job is over), as
`submit_answer` is the answer channel; neither helps with the task. The agent remembers with its own context, so it is
offered no memory tool, and it estimates positions from what it is sent, so it is offered no pointing tool (DECISIONS,
2026-10-10; `llm_memory` and `llm_locate` stay as modules and nothing offers them). A submission is parsed by
`answers.parse_answer`, never by a model. A system asked several questions at once (an ask) labels them Q1..Qn, and
`submit_answer` names the question it answers by that label."""

from __future__ import annotations

import json

from .answers import Malformed, parse_answer, to_json

NEXT_NAME = "next_frame"
NEXT = {"type": "function", "function": {
    "name": NEXT_NAME,
    "description": "Get what the robot senses next: a task message, one camera frame, questions to answer, or word "
                   "that the job is over. Call it when you are ready for the next.",
    "parameters": {"type": "object", "properties": {}}}}
SUBMIT_NAME = "submit_answer"
SUBMIT = {"type": "function", "function": {
    "name": SUBMIT_NAME,
    "description": "Give your answer to one of the questions you were just asked. Call it once for each question, "
                   "with the question's label (Q1, Q2, ...) and the answer object in one of the shapes the question "
                   "shows.",
    "parameters": {"type": "object", "properties": {
        "label": {"type": "string", "description": "The question's label, such as Q1."},
        "answer": {"type": "object", "description": "The answer: its \"type\" and the fields of that type."}},
        "required": ["label", "answer"]}}}
NO_QUESTION = "error: no question is pending"
ALREADY = "error: the answer was already submitted"
GAVE_UP = "error: the answer was rejected twice; the question is over"
NUDGE = "Call submit_answer now; use unknown if you cannot know."
TOOLS = (NEXT, SUBMIT)  # the reference agent's tools, in every exec of every channel


def unanswered(labels: list[str]) -> str:
    """What `next_frame` answers while questions of the ask are unanswered: the labels left, and the nudge."""
    return f"error: you have not answered {', '.join(labels)}. {NUDGE}"


class ContextTooLong(RuntimeError):
    """The model's context is full, though Codex's own compaction is on: the case ends, and the reason is counted."""


def read_submission(arguments, answer_type: str) -> tuple[dict | None, str, object]:
    """(answer as JSON, why not, raw): the parsed and normalised answer, or None with the reason it is unusable. `raw`
    is what the model submitted. The answer is `arguments["answer"]`; arguments that are themselves an answer (a
    `type` key and no `answer`) are taken as one."""
    raw = arguments
    if isinstance(arguments, str):
        try:
            raw = json.loads(arguments)
        except (ValueError, RecursionError):
            return None, "the arguments of submit_answer are not valid JSON", arguments
    if not isinstance(raw, dict):
        return None, "the arguments of submit_answer must be a JSON object", raw
    if "answer" in raw:
        raw = raw["answer"]
    elif "type" not in raw:
        return None, "missing required argument: answer", raw
    parsed = parse_answer(raw, answer_type)
    if isinstance(parsed, Malformed):
        return None, parsed.reason, raw
    return to_json(parsed), "", raw
