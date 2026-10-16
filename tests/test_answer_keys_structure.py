"""The keys of the prior systems on a structure family (`answer_keys`, 0.8.0): a pose key would let a gated system learn
the layout from other scripts, so `path_log` and `current_view_prior` have none on a template whose spec declares
`map_reader`, and the reference itself keys on the pose. Hand records; the pose key of base (1.5, 2.7), yaw 0 is
`3,5,0` (0.5 m cells: floor(1.5 / 0.5) = 3, floor(2.7 / 0.5) = 5; heading bin 0)."""

from __future__ import annotations

from baseline_kit import number, question, record
from stbench import answer_keys
from stbench.gates import MAP_READER
from stbench.questions_text import FAMILIES

POSE = "3,5,0"
STRUCTURE = question("clear_path", {"mode": "ground", "direction": "ahead", "cap": 3.0}, "q1", "number")
OTHER = question("path_ahead", {"cap": 3.0}, "q1", "number")


def rec(q: dict) -> dict:
    return {**record(q, number(2.0), script="s", phase="transit", base=(1.5, 2.7), yaw=0.0, when="during",
                     nm=number(2.0)), "subgoal": "clean the kitchen"}


def posed(levels: list[tuple]) -> list[bool]:
    return [POSE in level for level in levels]


def test_the_structure_templates_are_the_families_that_declare_the_map_reader():
    """Contract: a baseline may not import the specs, so `STRUCTURE_TEMPLATES` repeats which families declare
    `map_reader` as their reference; this is the tie. Regression caught: a family declared a structure family in its
    spec and not here (its gated priors would learn the layout), or listed here and not declared."""
    declared = {name for name, f in FAMILIES.items() if f.reference == MAP_READER}
    assert declared == set(answer_keys.STRUCTURE_TEMPLATES) == {"clear_path", "open_heading"}


def test_a_structure_template_has_no_pose_level_in_path_log_or_current_view_prior():
    """Contract (review §2.3, final review nit): by hand, a `clear_path` record at phase `transit`, subgoal "clean the
    kitchen" and `nm` 2.0 (the `>=2` bin) keys `path_log` on (template) alone (it answers the family from its own log
    and pools nothing, so a phase or subgoal level would be dead) and `current_view_prior` on (template, phase, bin),
    (template, bin), (template): none with the pose. A template that is no structure family keeps its pose-keyed levels
    in both: four each, the pose in the first two of `path_log`'s and in the first of the prior's. Regression caught: a
    layout learner inside a gated system, the pose dropped from every family, or levels no answerer reads."""
    path_log = answer_keys.levels("path_log", rec(STRUCTURE))
    assert path_log == [("clear_path",)]
    prior = answer_keys.levels("current_view_prior", rec(STRUCTURE))
    assert prior == [("clear_path", "transit", ">=2"), ("clear_path", ">=2"), ("clear_path",)]
    assert posed(answer_keys.levels("path_log", rec(OTHER))) == [True, True, False, False]
    assert posed(answer_keys.levels("current_view_prior", rec(OTHER))) == [True, False, False, False]


def test_path_log_pools_nothing_on_a_structure_family():
    """Contract (final review, nit): `answer_keys.pooled(system, template)` is False for `path_log` on a template of
    `STRUCTURE_TEMPLATES` (its answerer reads the robot's own log, never a pool) and True for every other pair, so the
    check's reach rule does not name `path_log` for a reach rate of 0 there. Regression caught: dead levels kept only so
    that an instrument that has nothing to find reads non-zero, or the rule exempting a prior that does pool."""
    assert not answer_keys.pooled("path_log", "clear_path")
    assert all(answer_keys.pooled(system, "clear_path") for system in ("blind", "current_view_prior", "map_reader"))
    assert all(answer_keys.pooled(system, "path_ahead") for system in ("blind", "path_log", "current_view_prior"))


def test_the_map_reader_keys_on_the_pose_whatever_the_template():
    """Contract: the reference is meant to hold the layout, so its levels are `path_log`'s with the pose: four, the
    first two with `3,5,0`, the first also with the subgoal. Regression caught: a reference that cannot learn the layout
    its score is published to show."""
    levels = answer_keys.levels(MAP_READER, rec(STRUCTURE))
    assert levels == [("clear_path", "transit", POSE, "clean the kitchen"), ("clear_path", "transit", POSE),
                      ("clear_path", "transit"), ("clear_path",)]
