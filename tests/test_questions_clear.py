"""The `clear_path` family's spec, words and dispatch (`questions_clear`, `questions_words.clear_slots`, 0.9.0 lane G1),
and the registry entries the new families need (`records.BALANCE_POOLS`)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stbench import clear_fly, clear_ground, questions_clear
from stbench.questions_text import FAMILIES, GOLDS, LOOKS, SLICED
from stbench.questions_words import clear_slots
from stbench.records import BALANCE_POOLS, REFERENCES, check_family, pool_of


def test_the_clear_path_spec_is_the_designed_one():
    """Contract (plan L0, G1): `clear_path` is a during family, a `number` on the `graded` credit (0.9.0), needing the
    picture and the pose, scoped by evidence, with the reference `map_reader`, a mapper gold and the slots mode,
    direction and cap;
    it is in the registry, asked at look points (`LOOKS`) and never sliced; the spec passes `check_family`. Hand-written
    from the plan. Regression caught: a spec that disagrees with the design (a `now` scope, which would pool a
    memory-scope drone row into the `now` column), a missing `map_reader` in the references, or a family drawn as a
    sliced one."""
    spec = FAMILIES["clear_path"]
    assert (spec.when, spec.answer_type, spec.credit, spec.needs, spec.scope, spec.reference, spec.mapper_gold,
            spec.slots) == ("during", "number", "graded", ("see", "pose"), "evidence", "map_reader", True,
                            ("mode", "direction", "cap"))
    assert check_family(spec) == [] and "map_reader" in REFERENCES
    assert "clear_path" in LOOKS and "clear_path" not in SLICED and GOLDS["clear_path"] is questions_clear.clear_path


def test_the_words_of_each_direction_by_hand():
    """Contract (brief 4.1, plan 3): the ground text of each direction is the brief's, one render per direction: "If you
    drove straight ahead now, how far could your body go before it touched anything? Answer from what you have seen, in
    metres from the front of your body; answer {cap} if nothing is within {cap} m."; back "If you reversed straight back
    now" and "from the back of your body"; left and right "If you turned in place to face your {left | right} and then
    drove straight" and "from the front of your body"; and the flier's, 0.8.0's: "If you flew straight {ahead | back}
    now at this height, how far could your body go before it touched anything you have seen? Answer in metres from the
    edge of your body." The text is the spec's render of the slots alone, and the flier's states no cap. Regression
    caught: a wording that drifts from the design, a cap not in the ground text, a cap in the flier's, or a turned
    corridor measured from an edge the words do not name."""
    render = FAMILIES["clear_path"].render
    tail = ("how far could your body go before it touched anything? Answer from what you have seen, in metres from the "
            "{} of your body; answer 5 if nothing is within 5 m.")
    assert render(clear_slots("drive", "ahead", 5.0)) == f"If you drove straight ahead now, {tail.format('front')}"
    assert render(clear_slots("drive", "back", 5.0)) == f"If you reversed straight back now, {tail.format('back')}"
    assert render(clear_slots("drive", "left", 5.0)) == (
        f"If you turned in place to face your left and then drove straight, {tail.format('front')}")
    assert render(clear_slots("drive", "right", 5.0)) == (
        f"If you turned in place to face your right and then drove straight, {tail.format('front')}")
    assert render(clear_slots("drive", "ahead", 3.0)).endswith("answer 3 if nothing is within 3 m.")
    assert render(clear_slots("fly", "ahead", 60.0)) == (
        "If you flew straight ahead now at this height, how far could your body go before it touched anything you "
        "have seen? Answer in metres from the edge of your body.")
    assert render(clear_slots("fly", "back", 60.0)).startswith("If you flew straight back now at this height, ")
    assert "60" not in render(clear_slots("fly", "ahead", 60.0))


def test_the_slots_name_a_mode_and_a_direction_or_are_refused():
    """Contract: `clear_slots` builds exactly the slots the spec names and refuses a mode or direction outside the two
    vocabularies (a ground robot's are ahead, back, left and right since 0.9.0; a flier's ahead and back). Regression
    caught: a typo'd slot value rendering as a ground question, a sideways flier, or a slot set that disagrees with the
    spec's names."""
    assert tuple(clear_slots("fly", "back", 60.0)) == FAMILIES["clear_path"].slots
    assert clear_slots("drive", "left", 3.0)["direction"] == "left"
    for mode, direction in (("walk", "ahead"), ("drive", "up"), ("fly", "left"), ("Drive", "ahead")):
        with pytest.raises(ValueError, match=mode):
            clear_slots(mode, direction, 5.0)


def test_the_gold_is_the_fliers_for_a_robot_that_flies_and_the_ground_robots_otherwise(monkeypatch):
    """Contract (plan L0): `GOLDS["clear_path"](w, k)` is `clear_fly.moment` for a robot whose `flies` is true and
    `clear_ground.moment` otherwise, called with the same `(w, k)`. Hand-worked with the two stubbed to return their
    own name. Regression caught: the dispatch keyed on the job or the robot's name (a special case), or a flier given
    the ground corridor."""
    monkeypatch.setattr(clear_fly, "moment", lambda w, k: ("fly", k))
    monkeypatch.setattr(clear_ground, "moment", lambda w, k: ("ground", k))
    flier = SimpleNamespace(robot=SimpleNamespace(flies=True))
    walker = SimpleNamespace(robot=SimpleNamespace(flies=False))
    assert GOLDS["clear_path"](flier, 7) == ("fly", 7) and GOLDS["clear_path"](walker, 9) == ("ground", 9)


def test_the_new_families_balance_between_their_declared_pools():
    """Contract (plan L0): `clear_path` and `person_mem` balance answerable against unknown, and `closest_person` a
    person against no one; `pool_of` finds a row's pool by its class. Hand-written. Regression caught: a pool missing
    (the suite's balance gate would not read the family), or a class in the wrong pool."""
    assert BALANCE_POOLS["clear_path"] == {"answerable": {"answerable"}, "unknown": {"unknown"}}
    assert BALANCE_POOLS["person_mem"] == {"answerable": {"answerable"}, "unknown": {"unknown"}}
    assert BALANCE_POOLS["closest_person"] == {"person": {"person"}, "no_one": {"no_one"}}
    assert pool_of("closest_person", "no_one") == "no_one" and pool_of("clear_path", "unknown") == "unknown"
    assert pool_of("closest_person", "answerable") is None
