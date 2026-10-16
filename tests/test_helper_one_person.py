"""The helper's one resident (`job_helper_checks`, code `job_one_person`): `person_mem` says 'the person', so a helper
world holds at most one. Hand scripts: the synthetic home's one-task script with people put in by hand."""

from __future__ import annotations

import person_kit as pk
from stbench import job_helper_checks
from stbench.mapspec import load_map


def codes(*people):
    s = pk.scene()
    for i, xy in enumerate(people):
        pk.put_person(s, f"P{i}", xy)
    return [(p.code, p.message) for p in job_helper_checks.checks(s.timeline(), load_map("apartment_a"))]


def test_a_helper_world_with_two_people_fails_its_checks_naming_the_script():
    """Contract (FD-task-questions review, section 3.1): the helper's home holds one resident or none; a script with two
    people fails `checks` with `job_one_person`, naming the map and seed and the people, so 'the person' of
    `person_mem` never has two candidates. One person and none pass. Regression caught: a generator change that adds a
    second resident (or a warehouse's pickers moved into a helper world) and quietly makes `person_mem` ambiguous."""
    assert codes() == [] and codes((2.0, 3.0)) == []
    (code, message), = codes((2.0, 3.0), (5.0, 3.0))
    assert code == "job_one_person" and "apartment_a seed 1" in message and "P0, P1" in message
