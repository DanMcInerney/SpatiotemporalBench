"""What the reference agent's prompt says of a list of places, and where each list family says what it counts (0.6.0
lane R3; 0.10.0 lane J). How a list is credited (0.9.0: per item, `llm_credit`) is pinned in
`test_llm_credit_sentence.py`; the wording is hand-written."""

from __future__ import annotations

from stbench import questions_task
from stbench.answers import PROMPT_SHAPES
from stbench.catalog_clutter import HOME_CLUTTER
from stbench.questions_text import FAMILIES, LOOKS


def test_the_shape_is_the_lists_form_and_counts_nothing():
    """Contract (0.10.0): the `positions` shape is the same for every list family, so it states the list's form (an item
    is a map position in metres, `xyz` also works, an empty list means none) and says nothing of what is counted: not
    what a thing is, not that people or chairs are left out, since the lists of people and of cables share it.
    Regression caught: the shape that served `task_things` alone and told a reader of `drop_zone_people` that people are
    not counted either way."""
    shape = PROMPT_SHAPES["positions"]
    for words in ('{"type": "positions", "items": [{"xy": [x, y]}]}', "a map position in metres", '"xyz": [x, y, z]',
                  "an empty list means there are none"):
        assert words in shape
    for words in ("thing", "people", "chairs", "counted", "pair", "stack", "lamp", "furniture", "mat"):
        assert words not in shape


def test_task_things_says_what_a_thing_is_and_leaves_loose_objects_in():
    """Contract (0.6.0 R3, review D2; moved from the shape in 0.10.0): the question's own text says a thing is an object
    that is not part of the room or its fixed furniture and fittings (a docking station, a charger, a door, a table, a
    shelf, a rack, the room's own floor mat), that a pair or a stack is one thing and that people and chairs are not
    counted either way; and it names floor lamps, fans and heaters as things, which are clutter in the gold
    (`questions_task.ITEMS`), so the boundary does not push an agent to skip a loose object (29 to 33% of the home's
    items). Regression caught: a text that never says fixed pieces are no thing (an agent lists the helper's dock and
    floor mats as false positives), one worded so as to make agents skip the lamps, fans and heaters the gold counts, or
    a move that left the rule out of the question when it left the shape."""
    text = FAMILIES["task_things"].render({"phrase": "within 2 m of the way you drove", "task": 0})
    for words in ("A thing is an object that is not part of the room or its fixed furniture and fittings",
                  "(a docking station, a charger, a door, a table, a shelf, a rack or the room's own floor mat)",
                  "a pair or a stack is one thing", "a floor lamp, a fan or a heater is a thing",
                  "people and chairs are not counted either way"):
        assert words in text
    assert text.endswith(questions_task.COUNTED)
    kinds = {k.name for k in HOME_CLUTTER}
    assert {"floor_lamp", "floor_fan", "space_heater"} <= kinds and "clutter" in questions_task.ITEMS


def test_each_list_family_says_what_it_counts_in_its_own_text():
    """Contract (0.10.0): every family whose answer is `positions` names its subject in the question itself, since the
    shape no longer does. By hand: `drop_zone_people` asks for the people within 4 m of the point below the drone,
    `tangle_ahead` for anything on the floor within 1 m that the brush could catch, and neither says that people are not
    counted (the first asks for exactly them). Regression caught: a list family whose question leaves the reader to
    guess its subject from a shape shared with the others."""
    assert {FAMILIES[f].answer_type for f in ("drop_zone_people", "tangle_ahead", "task_things")} == {"positions"}
    assert {"drop_zone_people", "tangle_ahead"} <= LOOKS
    zone = FAMILIES["drop_zone_people"].render({})
    assert "where are the people within 4 m of the point below you" in zone and "not counted" not in zone
    tangle = FAMILIES["tangle_ahead"].render({})
    assert "List anything lying on the floor within 1 m of you" in tangle and "that your brush could catch" in tangle
    assert "not counted" not in tangle
