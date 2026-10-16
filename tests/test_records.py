"""The question and gold rows and the family spec (`records.py`, plan §2.5): the validators every reader of the
compiler's files calls."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from stbench.answers import TYPES
from stbench.records import (
    ANSWER_CLASSES,
    BALANCE_POOLS,
    CREDITS,
    FamilySpec,
    check_family,
    check_gold,
    check_question,
    pool_of,
)

QID = "0123456789abcdef"
END_QID = "fedcba9876543210"
SQUARE = [[0, 0], [4, 0], [4, 3], [0, 3]]
# One gold of each answer type (`answers.parse_gold`'s shapes), and for each credit the types it scores with the one
# tolerance each reads (plan §2.6), written by hand: the table the validators must agree with.
GOLDS = {"position": {"type": "position", "xy": [3.2, 1.5]},
         "offset": {"type": "offset", "ahead_m": 3.5, "left_m": -1.0},
         "time": {"type": "time", "clock": "2026-03-01T08:15:00"},
         "yesno": {"type": "yesno", "value": True}, "count": {"type": "count", "value": 3},
         "number": {"type": "number", "value": 1.4},
         "set": {"type": "set", "items": [{"xy": [1.0, 2.0], "change": "added"}]},
         "positions": {"type": "positions", "items": [{"xy": [1.0, 2.0]}]},
         "cells": {"type": "cells", "items": [[0, 1]]},
         "route": {"type": "route", "regions": {"kitchen": SQUARE, "hall": [[4, 0], [6, 0], [6, 3], [4, 3]]},
                   "valid": [["kitchen", "hall"]]},
         "heading": {"type": "heading", "deg_left": 30.0}}
SCORES = {"ladder": {"position": {"ladder": [0.5, 1]}, "offset": {"ladder": [0.1, 0.25, 0.5]},
                     "number": {"ladder": [0.25, 0.5, 1, 2]}, "time": {"ladder": [3, 10, 30, 90]}},
          "region": {"position": {"erode_m": 0.2}}, "relative": {"number": {}}, "mra": {"count": {}},
          "exact": {"yesno": {}}, "f1": {"set": {"match_m": 0.5}, "positions": {"match_m": 0.5}, "cells": {}},
          "route": {"route": {}, "heading": {"profile_m": [0.5, -0.2, None, 0.3], "step_deg": 90, "step_m": 1.0,
                                             "resolution_m": 0.1, "need_m": 0.15}},
          "graded": {"offset": {"resolution_m": 0.05, "need_m": 0.15, "harm": "radial"},
                     "number": {"resolution_m": 0.02, "need_m": 0.15, "harm": "over", "threshold_m": 0.49},
                     "position": {"resolution_m": 0.02, "need_m": 0.5, "harm": None}},
          "graded_set": {"positions": {"match_m": 0.5, "resolutions_m": [0.02], "fp_weight": 0.5}},
          "regret": {"heading": {"profile_m": [1.5] * 360, "step_deg": 1, "resolution_m": 0.02, "need_m": 0.15,
                                 "threshold_m": 0.49}}}


def path_ahead(**over) -> dict:
    """A hand-written during question: how far the robot can drive straight on, asked after keyframe 40."""
    row = {"qid": QID, "when": "during", "decision_k": 40, "text": "How far can you drive straight ahead?",
           "template": "path_ahead", "slots": {}, "answer_type": "number", "needs": ["depth", "pose"]}
    return {**row, **over}


def path_ahead_gold(**over) -> dict:
    row = {"qid": QID, "family": "path_ahead", "gold": {"type": "number", "value": 1.4}, "slice": "answerable",
           "answer_class": None, "range_m": None, "credit": "ladder", "region": None,
           "tolerances": {"ladder": [0.25, 0.5, 1, 2]}, "metric": None, "scope": "now", "reference": None,
           "critical": True,
           "subjects": ["w:40"], "evidence_k": [38, 39, 40], "entities": []}
    return {**row, **over}


def object_where(**over) -> dict:
    """A hand-written end question about one entity, seen once at keyframe 12."""
    row = {"qid": END_QID, "when": "end", "decision_k": None, "text": "Where was the red mug last seen?",
           "template": "object_where", "slots": {"kind": "mug", "attributes": {"color": "red"}},
           "answer_type": "position", "needs": ["detections", "pose"]}
    return {**row, **over}


def object_where_gold(**over) -> dict:
    row = {"qid": END_QID, "family": "object_where", "gold": {"type": "position", "xy": [3.2, 1.5]},
           "slice": "answerable", "answer_class": None, "range_m": None, "credit": "ladder", "region": None,
           "tolerances": {"ladder": [0.25, 0.5, 1, 2]}, "metric": "xy", "scope": "memory", "reference": None,
           "critical": False, "subjects": ["e:E0042"], "evidence_k": [12], "entities": ["E0042"]}
    return {**row, **over}


def scored_gold(credit: str, kind: str, tolerances: dict) -> dict:
    """A gold row whose gold is a `kind` answer, scored by `credit` with these tolerances (the other keys as in
    `object_where_gold`; a region credit gets a polygon and a position gold its metric)."""
    return object_where_gold(gold=GOLDS[kind], credit=credit, tolerances=tolerances,
                             region=SQUARE if credit == "region" else None, metric="xy" if kind == "position" else None)


def path_ahead_family(**over) -> FamilySpec:
    """The plan's `path_ahead` entry (§4.5's table): during, a number on the distance ladder, depth and pose, now, its
    gold from the shared free-space rule; `render` is a stand-in (render purity is the compiler's check)."""
    spec = FamilySpec(name="path_ahead", when="during", answer_type="number", credit="ladder", needs=("depth", "pose"),
                      scope="now", reference=None, mapper_gold=True, slots=("cap",), render=lambda slots: "text")
    return replace(spec, **over)


def object_where_family(**over) -> FamilySpec:
    """The plan's `object_where` entry: end, a position on the ladder, see, pose and status, memory, three slots."""
    spec = FamilySpec(name="object_where", when="end", answer_type="position", credit="ladder",
                      needs=("see", "pose", "status"), scope="memory", reference=None, mapper_gold=False,
                      slots=("kind", "attributes", "event"), render=lambda slots: "text")
    return replace(spec, **over)


def test_a_valid_question_and_gold_pass():
    """Contract: a hand-written during row and an end row, each with its gold, give the validators nothing to say.

    Regression caught: validators that refuse everything, which would make every refusal test below pass vacuously."""
    assert check_question(path_ahead()) == []
    assert check_gold(path_ahead_gold()) == []
    assert check_question(object_where()) == []
    assert check_gold(object_where_gold()) == []


@pytest.mark.parametrize("bad", [["now"], ["depth", "memory"], ["now", "memory"]])
def test_needs_never_hold_a_scope(bad):
    """Contract (map X-H1): `now` and `memory` are the gold row's `scope`, so `needs` holding either is refused, and the
    message names `scope`; a real capability next to it stays accepted. Regression caught: a scope in `needs`, which no
    channel's capabilities hold, so every such question would be filtered out of every channel silently."""
    problems = check_question(path_ahead(needs=bad))
    assert problems and all("scope" in p for p in problems)
    assert check_question(path_ahead(needs=["depth"])) == []


@pytest.mark.parametrize("bad", [["lidar"], ["depth", "depth"], "depth", [3]])
def test_needs_are_distinct_capabilities(bad):
    """Contract: `needs` is a list of distinct strings, each one of `stream_protocol.CAPABILITIES`. Regression caught: a
    misspelt capability (`lidar`) or a repeat passing as a need no channel can meet, or a bare string read as a list."""
    assert check_question(path_ahead(needs=bad))


def test_region_credit_needs_a_region():
    """Contract: credit `region` needs `region` to be a polygon of three or more [x, y] points, and any other credit
    needs it null. Regression caught: a region gold that scores every answer 0 (no polygon to be inside), or a polygon
    on a ladder gold that nothing reads."""
    square = [[0, 0], [4, 0], [4, 3], [0, 3]]
    scored = {"credit": "region", "tolerances": {"erode_m": 0.2}}
    assert check_gold(object_where_gold(region=square, **scored)) == []
    assert any("region" in p for p in check_gold(object_where_gold(region=None, **scored)))
    assert any("region" in p for p in check_gold(object_where_gold(region=square[:2], **scored)))
    assert any("region" in p for p in check_gold(object_where_gold(region=[[0, 0], [4, 0], [4]], **scored)))
    assert any("region" in p for p in check_gold(object_where_gold(region=square)))  # credit ladder


@pytest.mark.parametrize(("when", "k", "ok"), [("during", 40, True), ("during", 0, True), ("end", None, True),
                                              ("end", 40, False), ("during", None, False), ("during", True, False),
                                              ("during", 4.0, False), ("during", "40", False)])
def test_decision_k_matches_when(when, k, ok):
    """Contract: `decision_k` is an integer (not a bool) exactly on a `during` row and null on an `end` row.
    Regression caught: an end question with a decision point, asked mid-stream, or a during question with none, never
    asked (a bool `True` read as keyframe 1 would ask it at the wrong step)."""
    assert (check_question(path_ahead(when=when, decision_k=k)) == []) is ok


def test_unknown_slice_matches_the_gold():
    """Contract: slice `unknown` goes with the gold `{"type": "unknown"}` and no other slice does. Regression caught:
    today's scorer error (a record whose slice disagrees with its gold) moved to read time, where it names the row."""
    unknown = {"type": "unknown"}
    assert check_gold(object_where_gold(gold=unknown, slice="unknown", metric=None)) == []
    assert any("slice" in p for p in check_gold(object_where_gold(slice="unknown")))
    assert any("slice" in p for p in check_gold(object_where_gold(gold=unknown)))
    assert any("slice" in p for p in check_gold(object_where_gold(slice="maybe")))


@pytest.mark.parametrize(("check", "row"), [(check_question, path_ahead()), (check_gold, path_ahead_gold())])
def test_unknown_keys_are_refused(check, row):
    """Contract: a row holds exactly its schema's keys: one extra, or one missing, is refused by its validator, and a
    row that is no object is too. Regression caught: a gold-side key slipping into the delivered question file, or a
    row that lost a key reaching a reader that indexes it."""
    assert check(row) == []
    assert check({**row, "ask_after_k": 41})
    assert check({k: v for k, v in row.items() if k != "qid"})
    assert check([row])


def test_entities_are_subjects():
    """Contract: every id in `entities` is also a subject, as `e:<id>`. Regression caught: an under-recorded subject,
    an entity the question depends on that the subject rule (no two end questions share one) never sees."""
    assert check_gold(object_where_gold()) == []
    assert any("E0042" in p for p in check_gold(object_where_gold(subjects=["w:40"])))
    assert any("E0042" in p for p in check_gold(object_where_gold(subjects=["e:E0007"])))


@pytest.mark.parametrize("subject", ["e:E0042", "d:mug/color=red", "d:mug/", "r:R3", "w:7", "w:3-9", "w:5-5", "a:10"])
def test_a_subject_is_a_thing_a_region_a_time_or_a_sighting(subject):
    """Contract: a subject is `e:<id>`, `d:<kind>/<attributes>`, `r:<region id>`, `w:<k>` or `w:<k0>-<k1>`, or (0.11.0,
    `approach_mem`) `a:<k>`, the sighting of a person at keyframe k, who is named by no entity. Regression caught:
    free text in `subjects`, which the subject rule would compare as an unrelated string, or `a:<k>` refused (the
    family's rows could not be compiled)."""
    assert check_gold(path_ahead_gold(subjects=[subject])) == []


@pytest.mark.parametrize("subject", ["E0042", "x:1", "e:", "d:mug", "r:", "w:", "w:-3", "w:a", "w:9-3", "w:1-2-3", "a:",
                                     "a:x", "a:3-9", 7])
def test_anything_else_is_not_a_subject(subject):
    """Contract: a string outside those forms, a reversed time interval, a sighting that is not one keyframe and a
    non-string are refused. Regression caught: a subject the rule cannot read (`w:9-3` names no time), accepted and
    compared as text."""
    assert any("subject" in p for p in check_gold(path_ahead_gold(subjects=[subject])))


def test_evidence_keyframes_are_sorted_and_entities_line_up():
    """Contract: `evidence_k` is sorted integers (a repeat is allowed: two entities seen in one keyframe) and `entities`
    is empty or parallel to it. Regression caught: an unsorted list or a bool keyframe, or an `entities` list one id
    short, which would pair an entity with the wrong keyframe."""
    two = {"subjects": ["e:E1", "e:E2"], "entities": ["E1", "E2"]}
    assert check_gold(path_ahead_gold(evidence_k=[5, 5], **two)) == []
    assert check_gold(path_ahead_gold(evidence_k=[5, 3], **two))
    assert check_gold(path_ahead_gold(evidence_k=[5, True], **two))
    assert check_gold(path_ahead_gold(evidence_k=[5.0, 6], **two))
    assert any("entities" in p for p in check_gold(path_ahead_gold(evidence_k=[5, 6, 7], **two)))


def test_tolerances_hold_nothing_or_exactly_one_valid_entry():
    """Contract: `tolerances` is `{}` or exactly one of a strictly increasing ladder of positive numbers, `match_m` or
    `erode_m` of at least 0, each tried on a credit that reads it (so only its own shape is in question). Regression
    caught: a ladder that is empty (the scorer divides by its length), unordered or not positive, two tolerances for one
    credit, or a negative radius that no answer could meet."""
    for credit, kind, ok in (("exact", "yesno", {}), ("ladder", "number", {"ladder": [0.1, 0.5]}),
                             ("f1", "positions", {"match_m": 0}), ("f1", "positions", {"match_m": 0.5}),
                             ("region", "position", {"erode_m": 0.2}), ("region", "position", {"erode_m": 0})):
        assert check_gold(scored_gold(credit, kind, ok)) == [], ok
    for credit, kind, bad in (*(("ladder", "number", {"ladder": v}) for v in ([], [1, 0.5], [1, 1], [0, 1], 1, [True])),
                              ("f1", "positions", {"match_m": -0.1}), ("region", "position", {"erode_m": "0.2"}),
                              ("f1", "positions", {"match_m": 1, "erode_m": 1}),
                              ("ladder", "number", {"radius_m": 1}), ("ladder", "number", []),
                              ("ladder", "number", None)):
        assert any("tolerances" in p for p in check_gold(scored_gold(credit, kind, bad))), bad


def test_a_position_gold_names_its_metric():
    """Contract: `metric` is null, `xy` or `xyz`, and never null on an answerable `position` gold (a number gold has
    none). Regression caught: a position gold with no metric, which the scorer would have to guess the dimension of."""
    assert check_gold(object_where_gold(metric="xyz")) == []
    assert any("metric" in p for p in check_gold(object_where_gold(metric=None)))
    assert any("metric" in p for p in check_gold(object_where_gold(metric="xz")))


@pytest.mark.parametrize(("key", "bad"), [("scope", "evidence"), ("scope", None), ("reference", "baseline"),
                                          ("critical", 1), ("credit", "mean"), ("family", None)])
def test_the_tags_and_the_credit_come_from_their_lists(key, bad):
    """Contract: `scope` is `now` or `memory` (the family-only `evidence` is resolved before the row is written),
    `reference` null or `path_log`, `critical` a real bool, `credit` one of `CREDITS`, and `family` a
    non-empty string. Regression caught: a tag outside its list reaching the reports as a column of its own."""
    assert any(key in p for p in check_gold(path_ahead_gold(**{key: bad})))


@pytest.mark.parametrize(("key", "bad"), [("qid", "0123456789ABCDEF"), ("qid", "0123456789abcde"), ("when", "after"),
                                          ("text", ""), ("template", None), ("slots", []), ("slots", {1: "a"}),
                                          ("answer_type", "clock")])
def test_a_question_field_has_its_own_type(key, bad):
    """Contract: `qid` is 16 lowercase hex, `when` one of `WHEN`, `text` and `template` non-empty strings, `slots` an
    object with string keys and `answer_type` one of `answers.TYPES`. Regression caught: a qid that is not the random
    hex the harness joins on, or an `answer_type` the parser would raise on at the first answer."""
    assert any(key in p for p in check_question(path_ahead(**{key: bad})))


@pytest.mark.parametrize("bad", ["0123456789ABCDEF", "0123456789abcde", "0123456789abcdef0", "g123456789abcdef", "",
                                 None, 1234567890123456])
def test_a_gold_qid_has_its_questions_form(bad):
    """Contract: a gold row's `qid` is the question's id (the harness scores an answer against the gold of the qid it
    answered), so it has the question's form, 16 lowercase hex characters, and both validators refuse the same values.
    Regression caught: a gold whose qid is free text or upper-case hex, which no question carries, so the scorer would
    find no gold for it or join it to the wrong question."""
    assert any("qid" in p for p in check_question(path_ahead(qid=bad)))
    assert any("qid" in p for p in check_gold(path_ahead_gold(qid=bad)))


def test_an_answerable_gold_names_its_evidence():
    """Contract (VISION truth rule 2): a gold of slice `answerable` has a non-empty `evidence_k`; an `unknown` gold may
    have none (nothing was seen). Regression caught: an answerable gold with no evidence passing, so a gold nothing the
    robot saw supports would be scored against the agent."""
    assert any("evidence_k" in p for p in check_gold(path_ahead_gold(evidence_k=[], entities=[])))
    assert any("evidence_k" in p for p in check_gold(object_where_gold(evidence_k=[], entities=[])))
    unknown = {"gold": {"type": "unknown"}, "slice": "unknown", "metric": None}
    assert check_gold(object_where_gold(evidence_k=[], entities=[], **unknown)) == []
    assert check_gold(object_where_gold(**unknown)) == []


def test_each_credit_scores_only_its_gold_types():
    """Contract (plan §2.6): `ladder` scores position, number and time; `region` a position; `relative` a number; `mra`
    a count; `exact` a yesno; `f1` a set, positions or cells; `route` a route; a credit on any other gold type is
    refused, and so is every combination outside that table. Regression caught: `route` on a position gold, or `exact`
    on a number, which the scorer (it dispatches on `credit`) would raise on, or score 0, at the first answer."""
    for credit, kinds in SCORES.items():
        for kind, tolerances in kinds.items():
            assert check_gold(scored_gold(credit, kind, tolerances)) == [], (credit, kind)
    for credit, kind in product(CREDITS, GOLDS):
        if kind not in SCORES[credit]:
            assert any("scores" in p for p in check_gold(scored_gold(credit, kind, {}))), (credit, kind)


def test_each_credit_needs_its_tolerance():
    """Contract (plan §2.5, §2.6): `ladder` needs `tolerances.ladder`, `region` `erode_m`, `f1` on a set or positions
    `match_m`, and every other credit, `f1` on cells included, takes `{}`; a missing key, another key or a stray one
    is refused and the message names `tolerances`. Regression caught: a ladder gold with `{}` (`score._ladder` falls
    back to its default thresholds without a word), a region gold with no erosion, or a tolerance on a credit that never
    reads it."""
    for credit, kinds in SCORES.items():
        for kind, tolerances in kinds.items():
            key = next(iter(tolerances), None)
            wrong = {"erode_m": 0.2} if key != "erode_m" else {"match_m": 0.5}
            for bad in ({} if key else {"ladder": [1]}, wrong if key else {"match_m": 0.5}):
                problems = check_gold(scored_gold(credit, kind, bad))
                assert any("tolerances" in p for p in problems), (credit, kind, bad)


def test_a_relative_gold_is_above_zero():
    """Contract (`score._relative_credit`): the relative credit divides by the gold, so a number gold scored by it is
    above 0; the other number credit, `ladder`, takes any number (0 m is a valid distance). Regression caught: a
    relative gold of 0 or below passing `check_gold`, so the compile writes it and the scorer raises on the first answer
    instead of the compile refusing the row; or the rule moved onto `ladder`, which would refuse a robot at the wall."""
    for value in (1.4, 0.001):
        assert check_gold(scored_gold("relative", "number", {}) | {"gold": {"type": "number", "value": value}}) == []
    for value in (0, 0.0, -1.4):
        gold = {"type": "number", "value": value}
        problems = check_gold(scored_gold("relative", "number", {}) | {"gold": gold})
        assert any("relative" in p and "above 0" in p for p in problems), value
        ladder = scored_gold("ladder", "number", {"ladder": [0.25, 0.5, 1, 2]}) | {"gold": gold}
        assert check_gold(ladder) == [], value


def test_an_unknown_gold_is_not_judged_by_its_credit():
    """Contract: an `unknown` gold has no answer to score, so its credit may be any of `CREDITS` with no tolerance
    (nothing reads one). Regression caught: the credit-to-type rule applied to the `unknown` type, which would refuse
    every unknown row of every family."""
    unknown = {"gold": {"type": "unknown"}, "slice": "unknown", "metric": None}
    for credit in CREDITS:
        row = object_where_gold(credit=credit, tolerances={}, region=SQUARE if credit == "region" else None, **unknown)
        assert check_gold(row) == [], credit


def test_a_valid_family_spec_passes():
    """Contract: the plan's two M1 entries (§4.5's table) give `check_family` nothing to say, and so does a family with
    no slots and each `scope` a during family may hold. Regression caught: a validator that refuses every spec, which
    would make each refusal test below pass vacuously."""
    assert check_family(path_ahead_family()) == []
    assert check_family(object_where_family()) == []
    assert check_family(path_ahead_family(slots=(), reference="path_log")) == []
    for scope in ("now", "memory", "evidence"):
        assert check_family(path_ahead_family(scope=scope)) == [], scope
    assert path_ahead_family().temporal is False  # a family is not temporal unless its module says so
    assert check_family(path_ahead_family(temporal=True)) == []


@pytest.mark.parametrize(("field", "bad"), [("name", ""), ("name", None), ("when", "after"), ("answer_type", "clock"),
                                            ("credit", "mean"), ("scope", "later"), ("scope", None),
                                            ("reference", "baseline"), ("mapper_gold", 1), ("slots", ["cap"]),
                                            ("slots", ("cap", "")), ("slots", ("cap", 3)), ("render", "text"),
                                            ("temporal", 1), ("temporal", None)])
def test_a_family_field_comes_from_its_list(field, bad):
    """Contract: a family's `name` is a non-empty string, `when` one of `WHEN`, `answer_type` one of `answers.TYPES`,
    `credit` one of `CREDITS`, `scope` one of `FAMILY_SCOPES`, `reference` null or `path_log`, `mapper_gold` and
    `temporal` real bools, `slots` a tuple of non-empty names and `render` callable; each violation is named by its
    field. Regression caught: a family the registry accepts whose tag no report column, scorer branch or renderer can
    read."""
    assert any(field in p for p in check_family(path_ahead_family(**{field: bad}))), bad


@pytest.mark.parametrize("bad", [("now",), ("depth", "memory"), ("memory",)])
def test_a_family_needs_never_hold_a_scope(bad):
    """Contract (map X-H1): a family's `needs` never holds `now` or `memory`, which are `scope`s, and the message names
    `scope`; a real capability next to one stays accepted. Regression caught: a scope in a family's `needs`, which
    every question of the family would inherit, so each would be filtered out of every channel silently."""
    problems = check_family(path_ahead_family(needs=bad))
    assert problems and all("scope" in p for p in problems)
    assert check_family(path_ahead_family(needs=("depth",))) == []


@pytest.mark.parametrize("bad", [("lidar",), ("depth", "depth"), ["depth"], "depth", (3,)])
def test_a_family_needs_are_distinct_capabilities(bad):
    """Contract: a family's `needs` is a tuple of distinct strings, each one of `stream_protocol.CAPABILITIES`.
    Regression caught: a misspelt capability (`lidar`) or a repeat passing as a need no channel can meet, or a list or
    bare string standing in for the tuple (a bare string would read as its letters)."""
    assert any("needs" in p for p in check_family(path_ahead_family(needs=bad)))


def test_a_family_slot_names_are_distinct():
    """Contract: a family's slot names are distinct. Regression caught: two slots of one name, so the second
    overwrites the first in `slots` and the question's wording silently drops a choice."""
    assert any("slots" in p for p in check_family(path_ahead_family(slots=("cap", "cap"))))
    assert check_family(path_ahead_family(slots=("cap", "event"))) == []


def test_an_end_family_is_memory():
    """Contract (plan §2.5, Scope): every end question is `memory`, so an end family's scope is `memory`; a during
    family may be `now`, `memory` or `evidence`. Regression caught: an end family tagged `now` or `evidence`, whose
    questions would land in the wrong scope column and be gated as if the current view could answer them."""
    for scope in ("now", "evidence"):
        assert any("scope" in p for p in check_family(object_where_family(scope=scope))), scope


def test_a_gold_row_declares_its_answer_class_and_range():
    """Contract (0.6.0): a gold row has `answer_class` (null, or one of `ANSWER_CLASSES`) and `range_m` (null, or a
    number of at least 0): a row without either key, a class outside the list, a class `answerable` or `unknown` that
    is not the row's slice, a class outside its family's balance pools (`object_where` classed "yes", `same_place`
    classed "later", a family with no pools classed at all), and a negative, non-numeric or boolean range are each
    refused, naming the key. Regression caught: unclassed rows pooled silently (a row written without the key), a
    typo in a class that no balance pool would ever read, a class that a draw's relaxation recomputed into another
    family's pool, or a range the distance report would divide by."""
    assert check_gold(object_where_gold(answer_class="answerable", range_m=2.5)) == []
    assert check_gold(object_where_gold(range_m=0)) == []
    coin = {"family": "same_place", "gold": {"type": "yesno", "value": True}, "credit": "exact", "tolerances": {},
            "metric": None}
    assert check_gold(object_where_gold(**coin, answer_class="yes")) == []
    assert check_gold(object_where_gold(**coin, answer_class="vacated")) == []
    for family, bad in (("object_where", "yes"), ("same_place", "later"), ("same_place", "answerable"),
                        ("ego_now", "yes")):   # a real class, but not one of the family's pools
        (problem,) = check_gold(object_where_gold(**{**coin, "family": family}, answer_class=bad))
        assert "answer_class" in problem and "no balance pool" in problem and family in problem, (family, bad)
    for key in ("answer_class", "range_m"):
        row = object_where_gold()
        del row[key]
        assert any(key in p for p in check_gold(row)), key
    for bad in ("Yes", "stayed", 3, ""):
        assert any("answer_class" in p for p in check_gold(object_where_gold(answer_class=bad))), bad
    for bad in (-0.1, "2", True, float("nan")):
        assert any("range_m" in p for p in check_gold(object_where_gold(range_m=bad))), bad
    assert any("answer_class" in p for p in check_gold(object_where_gold(answer_class="unknown")))   # slice answerable
    unknown = {"gold": {"type": "unknown"}, "slice": "unknown", "metric": None}
    assert check_gold(object_where_gold(**unknown, answer_class="unknown")) == []
    assert any("answer_class" in p for p in check_gold(object_where_gold(**unknown, answer_class="answerable")))


def test_the_balance_pools_partition_each_familys_classes():
    """Contract (0.6.0): `BALANCE_POOLS` names, per classed family, the pools a balance is read between and the classes
    of each, by hand: `same_place` yes against moved and vacated together, `object_was` earlier against later, the
    sliced families answerable against unknown, `task_things` empty against items. A class is in exactly one pool of its
    family, every class named is a declared one, and `pool_of` finds the pool of a row's class (None for a family with
    none, for no class, and for a class the family does not use). Regression caught: a pool that holds a class twice
    (a row counted on both sides), a typo that `gates_evidence` would never match, or `moved` and `vacated` read as
    two sides."""
    assert BALANCE_POOLS["same_place"] == {"yes": {"yes"}, "no": {"moved", "vacated"}}
    assert pool_of("same_place", "moved") == pool_of("same_place", "vacated") == "no"
    assert pool_of("same_place", "yes") == "yes" and pool_of("same_place", "earlier") is None
    assert pool_of("object_was", "later") == "later" and pool_of("task_things", "empty") == "empty"
    assert pool_of("ego_now", None) is None and pool_of("object_where", None) is None
    for family, pools in BALANCE_POOLS.items():
        classes = [c for members in pools.values() for c in members]
        assert len(classes) == len(set(classes)) and set(classes) <= set(ANSWER_CLASSES), family
        assert len(pools) == (1 if family == "open_heading" else 2), family


def test_a_family_credit_fits_its_answer_type():
    """Contract (plan §2.6): a family's `credit` scores its `answer_type` (the same table the gold rows are held to);
    every (credit, type) outside it is refused and the message names `credit`. Regression caught: a family declared
    `exact` on a number or `route` on a position, whose every gold the scorer would raise on or score 0."""
    for credit, kind in product(CREDITS, TYPES):
        problems = check_family(path_ahead_family(credit=credit, answer_type=kind))
        assert (problems == []) is (kind in SCORES[credit]), (credit, kind)
        assert all("credit" in p for p in problems), (credit, kind)
