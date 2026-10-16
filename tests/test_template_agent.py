"""`examples/template_agent`: the answer shapes it offers a user who fills its TODO with `ANSWERS[answer_type]`."""

from __future__ import annotations

import importlib.util

import pytest

from local_kit import TEMPLATE
from stbench.answers import TYPES, Malformed, parse_answer


def load():
    spec = importlib.util.spec_from_file_location("template_agent_under_test", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("answer_type", TYPES)
def test_the_template_has_a_valid_answer_shape_for_every_type_a_question_names(answer_type):
    """Contract (review of the Perceive branch, finding 5): `ANSWERS[question["answer_type"]]`, which the template tells
    its user to return, exists for every type a question's `answer_type` may name (`answers.TYPES`: the six of Recall,
    `positions`, `cells` and `route`) and is accepted by the strict parser as an answer to such a question (so the list
    shapes, a `set` of events and `positions` of places, are each the right one). Regression caught: a template written
    before a type existed, whose `ANSWERS` raised KeyError on a `cells` or `route` question, or a shape the parser
    scores 0 (an extra key, a `set` item without its change, a `cells` item that is a float pair)."""
    shape = load().ANSWERS[answer_type]
    assert shape["type"] == answer_type
    assert not isinstance(parse_answer(shape, answer_type), Malformed)
