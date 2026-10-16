"""`map_reader`'s answerers are found through a registry (`baselines_map.MAP_MODULES`, 0.9.0), one module a structure
family. I0 moved 0.8.0's `clear_path` answer into `baselines_map_clear` unchanged; its hand numbers are
`test_baselines_map`'s (the vacuum at (0.5, 1.7) heading +x on `test_path_log.hand_map`)."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, UNKNOWN, number, question, started, step
from stbench import answer_registry
from stbench.baselines_map import MAP_ANSWERERS, MAP_MODULES, MapReader, map_registry
from test_baselines_map import ask, clear
from test_path_log import hand_map


def test_the_registry_holds_the_two_structure_families():
    """Contract: the modules of `MAP_MODULES` define `clear_path`'s answerer and `open_heading`'s, each family once.
    Regression caught: a module left out of the list, so its family answers `unknown` silently."""
    assert MAP_MODULES == ("baselines_map_clear", "baselines_map_heading")
    assert set(MAP_ANSWERERS) == {"clear_path", "open_heading"}


def test_clear_paths_map_answer_is_the_same_through_the_registry(tmp_path):
    """Contract (plan I0): the move of `clear_path`'s answer into `baselines_map_clear` changed nothing. Hand case (the
    test_baselines_map one): the vacuum's front edge is 0.17 m ahead of its base, so ahead the table's leg is 1.33 m on
    and behind it the wall's inner face 0.27 m on; the answer given through the system equals the registry entry's own,
    and a template no module answers is `unknown`. Regression caught: the reference reading another pose or heading
    after the move, or answering a family it has no answerer for."""
    steps = [step(0, 0.5, 1.7, phase="transit")]
    for direction, want in (("ahead", 1.33), ("back", 0.27)):
        reader = started(MapReader, tmp_path / direction, map=hand_map(), events=EVENTS)
        q = clear(direction)
        got = ask(reader, steps, q)
        assert got == number(want), direction
        rec = reader.ask_record(q, {"k": 0, "clock": ""}, when="during")
        assert MAP_ANSWERERS["clear_path"](reader, q, rec, reader.log.steps[0]) == got
    other = started(MapReader, tmp_path / "other", map=hand_map(), events=EVENTS)
    assert ask(other, steps, question("path_ahead", {"cap": 3.0}, "q2", "number")) == UNKNOWN


def test_a_template_two_modules_answer_raises_naming_it():
    """Contract (`answer_registry.collect`): a family answered twice is refused, naming it and the module. Regression
    caught: the second of two modules silently replacing the first."""
    with pytest.raises(ValueError, match="clear_path.*answered twice"):
        map_registry(("baselines_map_clear", "baselines_map_clear"))
    assert answer_registry.collect(("baselines_map_clear",), "MAP_ANSWERERS").keys() == {"clear_path"}
