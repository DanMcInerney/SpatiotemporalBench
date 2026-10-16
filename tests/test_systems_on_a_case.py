"""Every scripted system over compiled synthetic cases through `stream.run_case`, on the gate channel: the oracle is
exact, and every system answers every question it is asked."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, IDS, at, compiled_case, number, play, record, rows, scores
from stbench import gates, mapspec
from stbench.answers import Malformed, parse_answer
from stbench.baselines import BASELINES, Oracle


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    return {cid: compiled_case(tmp_path_factory.mktemp(cid), cid) for cid in IDS}


def test_the_oracle_answers_every_family_exactly(cases):
    """Contract (plan §4.7, the ceiling): through `run_case` on `detections`, the oracle's penalised score is 1 on every
    question of a compiled case, `unknown` ones included. The world holds a thing seen only at 500 px, one only dark,
    one recognisable only after the event a question names (the compiler asks none of them), a thing no frame shows
    (`unknown`) and one that moves. Hand values: `path_ahead` from keyframe 21 is 2.2 (the wall's face is 2.21 m from
    the front edge, its sub-cell centre less 0.01), the brown box's place is (3.5, 1) (keyframe 12). The three cases
    draw other questions and events, so the oracle meets an `unknown` gold, an answerable thing at each event and the
    corridor. Regression caught: an answerer missing or wrong, a seen thing given an `unknown` gold, or a thing in the
    table that the gold skipped, each seen only as a ceiling failure."""
    seen: dict[str, set] = {"slice": set(), "family": set()}
    for case in cases.values():
        qs, gs = rows(case)
        run, _ = play(case, Oracle, events=EVENTS)
        assert run.status == "complete" and run.failures == {}
        assert scores(case, run) == dict.fromkeys((q["qid"] for q in qs), 1.0)
        seen["slice"] |= {g["slice"] for g in gs}
        seen["family"] |= {g["family"] for g in gs}
    assert seen == {"slice": {"answerable", "unknown"}, "family": {"path_ahead", "object_where"}}
    qs, gs = rows(cases[IDS[0]])
    answers = play(cases[IDS[0]], Oracle, events=EVENTS)[0].answers
    assert [answers[g["qid"]] for g in gs if g["family"] == "path_ahead"] == [number(2.2)]
    assert [answers[g["qid"]] for g in gs if g["gold"] == at(3.5, 1.0)] == [at(3.5, 1.0)]


def test_every_system_answers_every_question(cases):
    """Contract (map X-H15): each system of `BASELINES`, built with exactly the options its flags say (`prior`, `map`,
    `events`), completes the case on the gate channel with no failed answer and a well-formed answer to every question
    (`other_case` over a stream and an ask map of its own); the registry's keys are `gates.BASELINE_NAMES`.
    Regression caught: a gate keyed on a system with no rows, a system that raises on a question or leaves one out, or a
    flag naming an option the class does not take."""
    assert set(BASELINES) == set(gates.BASELINE_NAMES)
    case = cases[IDS[0]]
    qs, _ = rows(case)
    prior = [record(q, value, script=f"s{i}", when=q["when"], phase="transit", base=(-0.17, 0.0), rooms=("kitchen",),
                    siblings=[{"template": o["template"], "slots": o["slots"]} for o in qs if o is not q and
                              o["when"] == "end"] if q["when"] == "end" else ())
             for q in qs for i, value in enumerate([number(2.2), number(1.0)] if q["when"] == "during"
                                                   else [at(2.0, 1.0), at(3.5, 1.0), {"type": "unknown"}])]
    for name, b in BASELINES.items():
        opts = {key: value for key, value, wanted in (("prior", prior, b.needs_prior),
                                                      ("map", mapspec.load_map("apartment_a"), b.needs_map),
                                                      ("events", EVENTS, b.needs_events)) if wanted}
        run, _ = play(case, b.cls, **opts, **({"stream_from": case, "ask_map": {21: 21}} if b.other_stream else {}))
        assert run.status == "complete" and run.failures == {}, name
        assert set(run.answers) == {q["qid"] for q in qs}, name
        assert not [a for a in run.answers.values() if isinstance(parse_answer(a), Malformed)], name
