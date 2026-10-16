"""The change families as the compiler writes them (`questions_change`): the registry, valid question and gold rows, and
the hidden-change checks (VISION rule 4), by hand on one script of three tasks of `vacuum_run` (28 keyframes each) that
every family can ask about."""

from __future__ import annotations

import pytest

import change_kit as ck
import family_kit as fk
from change_kit import person, thing
from space_kit import BAG_SIZE, record, vacuum_run
from stbench import questionnaire, questions, questions_text
from stbench.questions_swap import gold_changes_under_swap
from stbench.records import check_family
from stbench.seeding import stable_hex

FAMILIES = ("object_was", "same_place", "person_where")


def mixed_script(*, hidden: bool = False, twins: bool = False, blue_x: float = 3.0):
    """Three tasks with, all seen at keyframes 1 (task 0) and 29 (task 1) only:
    - E0001 a red bag at (1, 1) and from the first gap at (2.5, 1): stale at task 0's events (`object_was` gold (1, 1)),
      and a no for `same_place` across tasks 0 and 1;
    - E0002 a blue bag at (3, 1), from the first gap at (`blue_x`, 1): a yes for `same_place` across tasks 0 and 1;
    - P1 a person at (1, 2) in task 0 and (3, 2) from the first gap: `person_where` golds (1, 2) and (3, 2);
    - E0005 an orange bag at (4, 2) that never moves: the second yes of `same_place` (a moment needs a yes and a no).
    With `hidden`, things the robot never saw happen too: all three move on in the second gap (E0001 to (6, 3), E0002 to
    (6, 1), P1 to (6, 3)), and E0002 stands at (5, 3) from t 3 s to 12 s of task 0, after its sighting at keyframe 1.
    With `twins`, two identical grey bags E0003 and E0004 and two identical yellow tote robots P2 and P3 are seen at 1
    and 29, each pair exchanging places in the first gap."""
    s = vacuum_run(3)
    if hidden:
        thing(s, "E0001", "bag", "red", [(1.0, 1.0), (2.5, 1.0), (6.0, 3.0)], [1, 29])
        s.item("E0002", "bag", {"color": "blue"},
               [(0.0, 3.0, (3.0, 1.0, 0.0)), (3.0, 12.0, (5.0, 3.0, 0.0)), (12.0, ck.GAP_S, (3.0, 1.0, 0.0)),
                (ck.GAP_S, ck.GAP2_S, (blue_x, 1.0, 0.0)), (ck.GAP2_S, None, (6.0, 1.0, 0.0))], size=BAG_SIZE)
        s.see([1, 29], "E0002", 1500)
        person(s, "P1", [(0.0, (1.0, 2.0)), (ck.GAP_S, (3.0, 2.0)), (ck.GAP2_S, (6.0, 3.0))], [1, 29])
    else:
        thing(s, "E0001", "bag", "red", [(1.0, 1.0), (2.5, 1.0)], [1, 29])
        thing(s, "E0002", "bag", "blue", [(3.0, 1.0), (blue_x, 1.0)], [1, 29])
        person(s, "P1", [(0.0, (1.0, 2.0)), (ck.GAP_S, (3.0, 2.0))], [1, 29])
    thing(s, "E0005", "bag", "orange", [(4.0, 2.0), (4.0, 2.0)], [1, 29])
    if twins:
        thing(s, "E0003", "bag", "grey", [(1.0, 3.0), (4.0, 3.0)], [1, 29])
        thing(s, "E0004", "bag", "grey", [(4.0, 3.0), (1.0, 3.0)], [1, 29])
        person(s, "P2", [(0.0, (6.0, 1.0)), (ck.GAP_S, (7.0, 1.0))], [1, 29], kind="tote_robot", colour="yellow")
        person(s, "P3", [(0.0, (7.0, 2.0)), (ck.GAP_S, (6.0, 2.0))], [1, 29], kind="tote_robot", colour="yellow")
    return s


def test_the_registry_holds_the_three_families():
    """Contract: the registry holds `object_was`, `same_place` and `person_where` after the earlier families, each a
    valid end, `memory`, temporal spec with a gold function in the compiler's one dict; `object_was` and `same_place`
    are coins and none is sliced. Regression caught: a family registered without its gold function or the temporal
    flag (the `time_needed` gate would not read it), or one that draws two ways."""
    assert [f for f in questions_text.FAMILIES if f in FAMILIES] == list(FAMILIES)   # in the registry's order
    for name in FAMILIES:
        spec = questions_text.FAMILIES[name]
        assert check_family(spec) == [] and spec.when == "end" and spec.scope == "memory" and spec.temporal
        assert questions.GOLDS[name] is questionnaire.GOLDS[name]
    assert questions_text.COINS & set(FAMILIES) == {"object_was", "same_place"}
    assert not questions_text.SLICED & set(FAMILIES)


def test_the_compiled_rows_are_valid_and_hold_the_hand_golds():
    """Contract: a case that asks the three families compiles to question and gold rows that pass `records.check_*` (the
    compile raises otherwise), the text of each question the family's rendering of its slots, the template its name and
    the answer type `position` or `yesno`. Hand values on `mixed_script`: an `object_was` question names the red bag, at
    its earlier state in task 0 (class `earlier`, gold (1, 1), evidence keyframe 1) or at its later in task 1 (class
    `later`, gold (2.5, 1), evidence 29), with `range_m` the slant at that sighting (the camera at keyframes 1 and 29 is
    at (2.0919, 1.5919, 0.08): 1.244 m from (1, 1, 0.157), 0.723 m from (2.5, 1, 0.157)); a `same_place` question is
    about tasks 0 and 1, its evidence keyframes 1 and 29, yes for the blue and the orange bag (class `yes`), no for the
    red (class `moved`: 1.5 m); a `person_where` question names the person, gold (1, 2) with evidence 1 in task 0 and
    (3, 2) with evidence 29 in task 1. Over 20 case ids all three families are asked. Regression caught: a row the
    harness cannot read, a gold the hand does not give, a class or range left off the row, or a family the questionnaire
    never asks."""
    seen = set()
    for cid in (stable_hex("rows", i, n=16) for i in range(20)):
        qs, gs = fk.compiled(mixed_script(), 0, 2, cid, "medium", ck.CHANGE_JOB)
        for q, g in zip(qs, gs, strict=True):
            assert q["qid"] == g["qid"] and g["family"] == q["template"] and q["template"] in FAMILIES
            assert q["text"] == questions_text.FAMILIES[q["template"]].render(q["slots"])
            seen.add(q["template"])
            slots = q["slots"]
            if q["template"] == "same_place":
                blue = slots["attributes"]["color"] != "red"   # the blue and the orange bag are the yes
                assert q["answer_type"] == "yesno" and slots["tasks"] == [0, 1] and g["evidence_k"] == [1, 29]
                assert g["gold"] == {"type": "yesno", "value": blue} and g["range_m"] is None
                assert g["answer_class"] == ("yes" if blue else "moved")
                assert g["subjects"] in (["e:E0002"], ["e:E0005"], ["e:E0001"])
            elif q["template"] == "object_was":
                early = slots["task"] == 0
                assert q["answer_type"] == "position" and slots["attributes"] == {"color": "red"}
                assert g["gold"] == {"type": "position", "xy": [1.0, 1.0] if early else [2.5, 1.0]}
                assert g["evidence_k"] == ([1] if early else [29])
                assert g["answer_class"] == ("earlier" if early else "later")
                assert g["range_m"] == pytest.approx(1.2444 if early else 0.7231, abs=1e-3)
                assert g["tolerances"]["ladder"][2] == pytest.approx(g["range_m"] / 2, abs=1e-6)
                assert q["text"] == (f"Where was the red bag during your {'first' if early else 'second'} task? "
                                     "Answer in map coordinates.")
            else:
                assert q["answer_type"] == "position" and slots["kind"] == "person" and slots["attributes"] == {}
                xy, evidence = ([1.0, 2.0], [1]) if slots["task"] == 0 else ([3.0, 2.0], [29])
                assert g["gold"] == {"type": "position", "xy": xy} and g["evidence_k"] == evidence
    assert seen == set(FAMILIES)


def test_hidden_changes_move_no_gold():
    """Contract (VISION rule 4): what happens where the robot cannot see changes no candidate question or gold of the
    three families, and two identical things that exchange places change none. Hand-worked on `mixed_script`: the same
    script with the red bag, the blue bag and the person moved on in the second gap (after their last sightings) and the
    blue bag at (5, 3) for a stretch of task 0 after its sighting has exactly the candidate golds of the plain one; two
    identical grey bags and two identical tote robots, each pair seen in two places, are never asked and swapped from t
    100 s on change nothing; the blue bag seen 0.5 m away in task 1 (a seen shift) does change its `same_place` gold
    from yes to no, so the comparison is not vacuous. Regression caught: a gold that reads the timeline where it should
    read what was seen, or a twin that is asked."""
    base = ck.golds(ck.window(mixed_script(), 0, 2))
    assert {key.split('"')[1] for key in base} == set(FAMILIES)   # the keys name their family: all three are compared
    assert ck.golds(ck.window(mixed_script(hidden=True), 0, 2)) == base
    s = mixed_script(twins=True)
    assert ck.golds(ck.window(s, 0, 2)) == base
    inp, rec = fk.inputs(s, ck.CHANGE_JOB), record(s, a=0, b=2, cid="c1", length="medium")
    assert gold_changes_under_swap(inp, rec, "E0003", "E0004", 100.0) == []
    assert gold_changes_under_swap(inp, rec, "P2", "P3", 100.0) == []
    moved = ck.golds(ck.window(mixed_script(blue_x=3.5), 0, 2))
    flips = [key for key in base if key in moved and base[key] != moved[key]]
    assert flips and all('"same_place"' in key for key in flips)
