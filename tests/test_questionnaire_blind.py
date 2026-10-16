"""`questionnaire_blind`: the best single feature of an end ask's other questions, chosen leaving one script out.
Hand-made records; every expected answer and score is worked out in the test."""

from __future__ import annotations

from baseline_kit import UNKNOWN, at, record, started, where
from stbench.blind import QuestionnaireBlind, features
from stbench.form_check import form_adversary
from stream_kit import end_q
from test_form_check import asked

START = "when you started cleaning"
P1, P2 = (1.5, 1.5), (6.5, 2.5)


def other(kind: str) -> dict:
    """A sibling question as a record holds it: `{template, slots}`."""
    q = where(kind, "red", START)
    return {"template": q["template"], "slots": q["slots"]}


def tagged(kind: str, tag: str) -> dict:
    """A sibling question as a record holds it, with one more slot, `id`: a value that no two scripts share."""
    sibling = other(kind)
    return {**sibling, "slots": {**sibling["slots"], "id": tag}}


def prior():
    """Twelve other scripts, each asked the brown box and one more thing: in six the other thing was a bag and the
    box was at P1, in six it was shoes and the box was at P2. Each script's other thing carries its own `id`."""
    q = where("box", "brown", START)
    return [record(q, at(*place), script=f"{kind}{i}", siblings=[tagged(kind, f"{kind}{i}")])
            for kind, place in (("bag", P1), ("shoes", P2)) for i in range(6)]


def test_features_of_the_other_questions_by_hand():
    """Contract (plan §4.7): the features of an ask's other questions are `count` (how many), `templates` (their
    sorted templates) and, for each slot name any holds, `slot:<name>` (the sorted `json.dumps` of their values).
    Two others, a red bag asked about when you started cleaning and a `path_ahead` of cap 3, give count 2, the
    templates `object_where` and `path_ahead`, and the slots they hold (`attributes`, `event`, `kind`, `cap`), the
    values sorted as text. Regression caught: a feature that differs between the harness's records and a system's
    ask."""
    cap = {"template": "path_ahead", "slots": {"cap": 3.0}}
    assert features([other("bag"), cap]) == {
        "count": (2,), "templates": ("object_where", "path_ahead"), "slot:attributes": ('{"color": "red"}',),
        "slot:cap": ("3.0",), "slot:event": ('"when you started cleaning"',), "slot:kind": ('"bag"',)}
    assert features([]) == {"count": (0,), "templates": ()}


def test_questionnaire_blind_answers_during_questions_unknown(tmp_path):
    """Contract (plan §4.7): every during question is `unknown`; an end question is answered from the other
    scripts' records of its template that share the value of the feature with the best leave-one-script-out mean
    (the first on a tie), `unknown` when none do. Hand case: the prior's kind of the other question decides the
    box's place, and `slot:kind` is the one feature that carries it. Leaving a bag script out, the other eleven golds
    are five P1 and six P2 for every other feature, whose best response P2 (1/11) scores -1 on P1; for `slot:kind`
    the five other bag scripts give P1, the five golds of five scripts the floor wants, and +1; so `slot:kind` has
    mean 1 and the rest -1, bar `slot:id`, whose value each script holds alone: no other script shares it, so
    `unknown`, 0. Held in the sample, though, `slot:id` scores 1 as well (each script's own record is its pool) and
    sorts first, so a choice that does not leave the script out takes it, and then finds no script that shares the
    asked id. Asked beside a bag the box is at P1; beside shoes at
    P2; beside a toy nothing shares the value, so `unknown`; asked alone every feature is shared by none. Regression
    caught: a gate keyed on a system with no rows in a family, or a feature chosen by anything but its held-out
    score (one that memorises each script then picks the feature that finds nothing). The prior holds six scripts of
    each kind, so that leaving one out still leaves the five golds the support floor wants."""
    system = started(QuestionnaireBlind, tmp_path, prior=prior())
    box = where("box", "brown", START, qid="box")
    fresh = {kind: {**where(kind, "red", START, qid="other"), "slots": {**other(kind)["slots"], "id": "fresh"}}
             for kind in ("bag", "shoes", "toy")}
    beside = {kind: [box, sibling] for kind, sibling in fresh.items()}
    now = {"k": 5, "clock": ""}
    assert system.ask("end", now, beside["bag"])["box"] == at(*P1)
    assert system.ask("end", now, beside["shoes"])["box"] == at(*P2)
    assert system.ask("end", now, beside["toy"])["box"] == UNKNOWN
    assert system.ask("end", now, [box]) == {"box": UNKNOWN}
    assert system.ask("during", now, beside["bag"]) == {"box": UNKNOWN, "other": UNKNOWN}
    assert system.chosen("object_where", [tagged("bag", "fresh")]) == "slot:kind"
    assert system.chosen("object_where", []) == "count"   # alone: count and templates tie at -1, the first wins
    assert system.chosen("path_ahead", [tagged("bag", "fresh")]) is None   # no record of that template


def test_questionnaire_blind_and_form_need_support(tmp_path):
    """Contract (VISION rule 5, `blind.supported`): a prior answers from a feature's records only when they hold at
    least `MIN_POOL_GOLDS` (5) golds from `MIN_POOL_SCRIPTS` (3) scripts, else `unknown`; `questionnaire_blind` reads
    the floor in its answer and in its leave-one-script-out score, the form adversary in each feature's pool. Hand case,
    `questionnaire_blind`: a box asked beside a bag, every record of that kind at P1 (the leave-one-out score is 0 for
    every feature, so the first, `count`, is chosen): two records of one script share the value and answer `unknown`,
    five records of three scripts answer P1. The form adversary, one feature every question holds and every gold at P1:
    questions of the scripts x, y, y give each a pool of other scripts' of one script or one question, so `unknown` and
    0 for all three; x, y, y, z, z, w give x and w five golds of three scripts (score 1) and y and z four (0).
    Regression caught: a one-record pool answering, or scoring under abstaining on the next script (the floor left out
    of the two systems that learn from other scripts' golds)."""
    box = where("box", "brown", START, qid="box")
    beside = [box, where("bag", "red", START, qid="other")]

    def answer(*scripts: str) -> dict:
        records = [record(box, at(*P1), script=s, siblings=[other("bag")]) for s in scripts]
        system = started(QuestionnaireBlind, tmp_path / "-".join(scripts), prior=records)
        return system.ask("end", {"k": 0, "clock": ""}, beside)["box"]

    assert answer("s0", "s0") == UNKNOWN
    assert answer("s0", "s0", "s1", "s1", "s2") == at(*P1)

    def form(*scripts: str) -> list[float]:
        """The scores of the rule on a feature all questions share, one question a name, every gold at P1."""
        questions = [asked(s, end_q(n, slots={"kind": "mug"}), at(*P1), f=1) for n, s in enumerate(scripts)]
        return [r["score"] for r in form_adversary(questions)[0]]

    assert form("x", "y", "y") == [0.0, 0.0, 0.0]
    assert form("x", "y", "y", "z", "z", "w") == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
