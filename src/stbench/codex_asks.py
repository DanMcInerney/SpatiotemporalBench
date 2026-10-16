"""The answers of one ask: what `submit_answer` does with a call, and what `next_frame` does while questions are open.

An ask has several questions, labelled Q1..Qn, and the model answers each with `submit_answer(label, answer)`. The table
knows each question only by its label and `answer_type` (`labels`, `{label: answer_type}`). A submission is read by
`agent_tools.read_submission` for its label's type: a bad one gets one repair (the reply repeats the type's shape,
`answers.PROMPT_SHAPES`) and a second ends that label as malformed; a valid one ends it (a later call says `ALREADY`).

`table`, `{label: {"status", "answer" | "reason", "bad", "raw"}}`, is how the system learns the answers: `status` is
`submitted` (with `answer`), `pending` (one rejected try, with `raw`) or `failed` (with `reason` `malformed` and `raw`,
or `no_submit`); a label nothing was submitted for has no entry. The `next_frame` rule (`pull`): while labels are
unsubmitted the first call answers an error naming them (the ask's one nudge, kept in `nudged`) and the second fails
them `no_submit` and goes through. `premature` counts the calls that found labels open.
"""

from __future__ import annotations

import json

from .agent_tools import ALREADY, GAVE_UP, read_submission, unanswered
from .answers import PROMPT_SHAPES

DONE = ("submitted", "failed")  # a label in one of these statuses takes no more submissions


def _split(arguments) -> tuple[object, object]:
    """(the label, the rest) of a `submit_answer` call's arguments, which may be an object or its JSON text. The rest
    is what `read_submission` reads; a label that cannot be found is None."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            return None, arguments
    if not isinstance(arguments, dict):
        return None, arguments
    return arguments.get("label"), {k: v for k, v in arguments.items() if k != "label"}


class AskTable:
    """The state of one ask, from the moment the model is shown it to the pull that ends it."""

    def __init__(self, labels: dict[str, str]):
        self.labels = dict(labels)
        self.table: dict[str, dict] = {}
        self.nudged: list[str] = []  # the labels the first premature `next_frame` named
        self.premature = 0

    def left(self) -> list[str]:
        """The labels not yet answered or failed, in order."""
        return [label for label in self.labels if self.table.get(label, {}).get("status") not in DONE]

    def pull(self) -> str | None:
        """What `next_frame` answers while labels are open: the error naming them (the first call), else None and
        those labels failed `no_submit` (the second and later: the call goes through). None when none are open."""
        left = self.left()
        if not left:
            return None
        self.premature += 1
        if not self.nudged:
            self.nudged = left
            return unanswered(left)
        for label in left:
            bad = self.table.get(label, {}).get("bad", 0)
            self.table[label] = {"status": "failed", "reason": "no_submit", "bad": bad}
        return None

    def submit(self, arguments) -> str:
        """The reply to one `submit_answer` call: `ok`, or `error: ...` text (the label error, `ALREADY`, `GAVE_UP`, or
        a repair with the shape of the label's type)."""
        label, rest = _split(arguments)
        if not isinstance(label, str) or label not in self.labels:
            return (f"error: label must be one of {', '.join(self.labels)}: the number of the question you answer, "
                    f"as in submit_answer(label=\"{next(iter(self.labels))}\", answer={{...}})")
        held = self.table.get(label, {})
        bad = held.get("bad", 0)
        if held.get("status") in DONE:
            return GAVE_UP if held["status"] == "failed" else ALREADY
        answer, why, raw = read_submission(rest, self.labels[label])
        if answer is not None:
            self.table[label] = {"status": "submitted", "answer": answer, "bad": bad}
            return "ok"
        bad += 1
        if bad > 1:
            self.table[label] = {"status": "failed", "reason": "malformed", "raw": raw, "bad": bad}
            return GAVE_UP
        self.table[label] = {"status": "pending", "raw": raw, "bad": bad}
        shape = PROMPT_SHAPES[self.labels[label]]
        return f"error: {why}. Call submit_answer again with label {label} and answer = {shape}"
