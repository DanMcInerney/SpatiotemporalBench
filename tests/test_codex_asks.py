"""The answers of one ask (`codex_asks`): what `submit_answer` does with a call and what `next_frame` does while
questions are open. No model, no `codex`, no process.

Expected values are written by hand: the table's contents follow the module docstring.
"""

from __future__ import annotations

import json

from stbench.agent_tools import ALREADY, GAVE_UP, unanswered
from stbench.answers import PROMPT_SHAPES
from stbench.codex_asks import AskTable


def submit(table: AskTable, label, answer) -> str:
    return table.submit({"label": label, "answer": answer})


def test_each_label_of_an_ask_is_read_for_its_own_type_and_ends_with_its_own_answer():
    """Contract (plan §4.9, map HA-H13): the ask's questions are known by label and answer type; a submission is read
    by the code every agent shares (`read_submission`) for its label's type, written to the ask's table
    `{label: {status, answer, bad}}` with the rejected tries it took, and replied to with `ok`; a second submission
    for a label is refused as already made and changes nothing, while the other label is still open. Regression
    caught: one answer file per ask holding only the last submission, an answer read for another label's type, or a
    later call that overwrites an answer."""
    table = AskTable({"Q1": "count", "Q2": "position"})
    replies = [submit(table, "Q2", {"type": "position", "xy": [1.0, 2.0]}), submit(table, "Q1", {"type": "count",
               "value": 3}), submit(table, "Q1", {"type": "count", "value": 9}),
               submit(table, "Q2", {"type": "count", "value": 1})]
    assert replies == ["ok", "ok", ALREADY, ALREADY]
    assert table.table == {"Q2": {"status": "submitted", "answer": {"type": "position", "xy": [1.0, 2.0]}, "bad": 0},
                           "Q1": {"status": "submitted", "answer": {"type": "count", "value": 3}, "bad": 0}}
    assert table.left() == []


def test_a_submission_names_a_label_of_the_ask():
    """Contract: `submit_answer` needs the label of one of the ask's questions; a missing label, an unknown one and a
    label that is not text are each refused with the labels that exist, and count as no try of any question (nothing
    is written). The arguments may be given as JSON text. Regression caught: a submission with no label taken for the
    first question, or one that spends a label's one repair."""
    table = AskTable({"Q1": "count", "Q2": "count"})
    count = {"type": "count", "value": 2}
    refusals = [table.submit({"answer": count}), submit(table, "Q3", count), submit(table, 7, count)]
    for reply in refusals:
        assert reply.startswith("error: label must be one of Q1, Q2")
    assert table.submit(json.dumps({"label": "Q2", "answer": count})) == "ok"
    assert table.table == {"Q2": {"status": "submitted", "answer": count, "bad": 0}}


def test_one_bad_submission_a_label_gets_a_repair_and_a_second_is_malformed():
    """Contract (plan §4.9: one repair per label): the first invalid submission for a label is answered with the
    reason, the label and the shape of its type (`PROMPT_SHAPES`), and the label stays open; a valid one after it is
    accepted with `bad` 1 (the system calls it repaired); a second invalid one ends the label as `failed` with reason
    `malformed` and what was submitted, and a later call is refused. The repairs of two labels are counted apart.
    Regression caught: no repair round, a repair that never ends, a failure recorded without its raw submission, or a
    count of bad tries shared by the labels of an ask."""
    table = AskTable({"Q1": "count", "Q2": "position"})
    first = submit(table, "Q1", {"type": "count", "value": -1})
    good = submit(table, "Q1", {"type": "count", "value": 2})
    wrong = submit(table, "Q2", {"type": "count", "value": 1})
    again = submit(table, "Q2", {"type": "count", "value": 1})
    late = submit(table, "Q2", {"type": "position", "xy": [1.0, 2.0]})
    assert first.startswith("error: ") and first.endswith(
        f"Call submit_answer again with label Q1 and answer = {PROMPT_SHAPES['count']}")
    assert good == "ok"
    assert wrong.endswith(f"Call submit_answer again with label Q2 and answer = {PROMPT_SHAPES['position']}")
    assert again == GAVE_UP == "error: the answer was rejected twice; the question is over" and late == again
    assert table.table["Q1"] == {"status": "submitted", "answer": {"type": "count", "value": 2}, "bad": 1}
    q2 = table.table["Q2"]
    assert (q2["status"], q2["reason"], q2["bad"], q2["raw"]) == ("failed", "malformed", 2,
                                                                  {"type": "count", "value": 1})


def test_the_first_next_frame_with_labels_open_is_the_nudge_and_the_second_fails_them():
    """Contract (FD-frame-pull §3.3): while labels are unsubmitted the first `next_frame` answers an error naming
    exactly the labels left (the ask's one nudge, kept in `nudged`) and the second fails those labels `no_submit` and
    lets the call through; a label already answered, and one failed `malformed`, count as done and are not named; a
    label with one rejected try and nothing after it fails `no_submit`, keeping its bad try. `premature` counts both
    calls. Regression caught: a nudge that names answered labels, a second call that is nudged again (a model that never
    answers would loop on the harness), or a malformed label failed twice."""
    table = AskTable({"Q1": "count", "Q2": "count", "Q3": "count", "Q4": "count"})
    submit(table, "Q1", {"type": "count", "value": 1})  # answered
    submit(table, "Q2", {"type": "count", "value": -1})
    submit(table, "Q2", {"type": "count", "value": -1})  # malformed: over
    submit(table, "Q3", {"type": "count", "value": -1})  # one bad try, then nothing
    assert table.left() == ["Q3", "Q4"]
    assert table.pull() == unanswered(["Q3", "Q4"]) and table.nudged == ["Q3", "Q4"] and table.premature == 1
    assert table.pull() is None and table.premature == 2 and table.left() == []
    assert table.table["Q3"] == {"status": "failed", "reason": "no_submit", "bad": 1}
    assert table.table["Q4"] == {"status": "failed", "reason": "no_submit", "bad": 0}
    assert table.table["Q2"]["reason"] == "malformed" and table.table["Q1"]["status"] == "submitted"


def test_a_next_frame_with_every_label_done_goes_through_and_is_not_premature():
    """Contract: when every label is answered or failed, `next_frame` is not held back and counts as no premature call.
    Regression caught: a nudge for an ask that is complete, which would stall every case at its first ask."""
    table = AskTable({"Q1": "count"})
    submit(table, "Q1", {"type": "count", "value": 1})
    assert table.pull() is None and table.premature == 0 and table.nudged == []
