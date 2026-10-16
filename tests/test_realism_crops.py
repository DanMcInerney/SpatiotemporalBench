"""The crop check of `stbench realism sheet | report` (`realism_crops`): the two fixed prompts, how an answer is read
and how a crop is scored by the forced choice (first) and by free naming (second). By hand on answers made here. Free:
dicts, no model and no Blender."""

from __future__ import annotations

from stbench import realism_crops as K

ASKABLE = {"mug": ["red", "blue"], "shoes": ["black", "white"], "toy": ["yellow", "red"]}
KEY = [{"crop": "a.png", "class": "clutter:door_stop", "kind": None, "colour": None},
       {"crop": "b.png", "class": "clutter:power_strip", "kind": None, "colour": None},
       {"crop": "c.png", "class": "askable:mug", "kind": "mug", "colour": "red"},
       {"crop": "d.png", "class": "askable:mug", "kind": "mug", "colour": "blue"},
       {"crop": "e.png", "class": "askable:mug", "kind": "mug", "colour": "red"},
       {"crop": "f.png", "class": "askable:shoes", "kind": "shoes", "colour": "black"},
       {"crop": "g.png", "class": "dressing:radiator", "kind": None, "colour": None},
       {"crop": "h.png", "class": "askable:toy", "kind": "toy", "colour": "yellow"}]


def test_a_crop_is_confused_when_its_answer_names_an_askable_kind_in_one_of_its_colours():
    """Contract (`score`, `read_answer`): an answer reads `<colour> <object>`, the colour through the fixed table (navy
    is blue); a crop is confused when the object names an askable kind (the name or a synonym) in a colour that kind's
    tracked things have and the crop is not that kind's own; an askable kind's crop is right when the answer names its
    own kind in its colour. By hand with mugs red or blue, shoes black or white and toys yellow or red: a door stop
    answered 'black shoe' is confused (a black shoe is a tracked thing); a power strip answered 'green shoe' is not (no
    green shoe); mug crops answered 'red mug', 'navy mug' (blue) and 'red cup' (a synonym) are right for mugs that are
    red, blue and red; a black shoes crop answered 'dark boots' is not right (the colour is not black); a yellow toy
    answered 'yellow rubber duck' is right (a rubber duck is a toy, `scenery.SYNONYMS`). Regression caught: a clutter
    object read as a tracked thing and never noticed, a colour word the table lacks, a synonym that does not count, or
    the toy the crop check calls wrong when it is right (a duck)."""
    answers = {"a.png": "black shoe", "b.png": "green shoe", "c.png": "red mug", "d.png": "navy mug",
               "e.png": "red cup", "f.png": "dark boots", "h.png": "Yellow rubber duck"}
    out = K.score(answers, KEY, ASKABLE)
    assert out["answered"] == 7
    assert out["confused"] == [{"crop": "a.png", "class": "clutter:door_stop", "named": "shoes", "colour": "black"}]
    assert out["per_kind"] == {"mug": {"crops": 3, "right": 3, "rate": 1.0},
                               "shoes": {"crops": 1, "right": 0, "rate": 0.0},
                               "toy": {"crops": 1, "right": 1, "rate": 1.0}}
    assert K.read_answer("Navy blue mug") == ("blue", "blue mug") and K.read_answer("mug") == ("", "mug")


def test_the_forced_choice_and_free_naming_are_scored_apart_and_a_plain_answer_is_a_free_one():
    """Contract (`crop_scores`, `split_answers`): an answers file maps a crop to `{"forced": ..., "free": ...}` and each
    protocol is scored on its own; a plain string is a free answer, a protocol with no answer at all is None. By hand:
    the forced choice names a door stop 'black shoes' (confused) where free naming says 'grey rubber wedge' (not), and
    the forced 'none' for a radiator is no kind at all; a mug crop is right forced ('red mug') and wrong free ('red
    thing'). Regression caught: one protocol's answers scored as the other's, or the first protocol dropped when the
    file has only the second (the first round's file of strings)."""
    answers = {"a.png": {"forced": "black shoes", "free": "grey rubber wedge"},
               "g.png": {"forced": "none", "free": "white radiator"},
               "c.png": {"forced": "red mug", "free": "red thing"}}
    out = K.crop_scores(answers, KEY, ASKABLE)
    assert [c["named"] for c in out["forced"]["confused"]] == ["shoes"] and out["free"]["confused"] == []
    assert out["forced"]["per_kind"] == {"mug": {"crops": 1, "right": 1, "rate": 1.0}}
    assert out["free"]["per_kind"] == {"mug": {"crops": 1, "right": 0, "rate": 0.0}}
    plain = K.crop_scores({"a.png": "black shoe"}, KEY, ASKABLE)
    assert plain["forced"] is None and len(plain["free"]["confused"]) == 1
    assert K.PROTOCOLS == ("forced", "free")  # the forced choice first


def test_the_forced_prompt_lists_the_kinds_in_order():
    """Contract (`prompts`): the forced prompt lists the askable kinds in order with underscores as spaces, and the
    free one is the fixed text. Regression caught: a kind a question can name missing from the list the reviewer must
    choose among."""
    p = K.prompts(["laundry_basket", "toy", "bag"])
    assert p["kinds"] == ["bag", "laundry_basket", "toy"]
    assert "bag, laundry basket, toy" in p["forced"] and p["free"] == K.FREE_PROMPT
