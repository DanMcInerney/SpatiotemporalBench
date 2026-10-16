"""The declaration of a job's `task_things` predicate (`task_predicate.TaskThings`, 0.6.0) and its place in the job."""

from __future__ import annotations

import dataclasses

import pytest

from space_kit import JOB
from stbench import job_registry
from stbench.task_predicate import TaskThings


def path(**over) -> TaskThings:
    base = {"phrase": "on the floor within 1.2 m of the way you went", "anchor": "path",
            "phases": frozenset({"navigating", "carrying"}), "until": "placing", "reach_m": 1.2, "band_m": 0.4,
            "floor": True}
    return TaskThings(**{**base, **over})


def test_a_predicate_holds_what_the_plan_names_and_defaults_to_the_open_case():
    """Contract (plan §3): a `TaskThings` is frozen and carries the phrase, the anchor, the phases, `until`, the reach
    and band, `floor`, `min_height_m`, `dont_care_kinds`, `item_kinds` (default empty: every kind may be an item;
    0.11.0) and `empty_keep_percent` (default 100, keep every empty). By hand on the helper's. Regression caught: a
    field renamed so the tables and the question disagree, or a default that thins the empties without the
    coordinator's say."""
    p = path()
    assert (p.anchor, p.until, p.reach_m, p.band_m, p.floor) == ("path", "placing", 1.2, 0.4, True)
    assert (p.min_height_m, p.dont_care_kinds, p.item_kinds, p.empty_keep_percent) == (
        0.0, frozenset(), frozenset(), 100)
    assert [f.name for f in dataclasses.fields(TaskThings)] == [
        "phrase", "anchor", "phases", "until", "reach_m", "band_m", "floor", "min_height_m", "dont_care_kinds",
        "item_kinds", "empty_keep_percent"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.reach_m = 2.0   # type: ignore[misc]
    assert hash(p) == hash(path())   # frozen with hashable fields: it travels in a job and pickles to a worker


@pytest.mark.parametrize("over, why", [
    ({"anchor": "corridor"}, "anchor"),
    ({"phrase": ""}, "phrase"),
    ({"phases": frozenset()}, "phases"),
    ({"anchor": "point", "until": "placing"}, "until"),
    ({"reach_m": -1.0}, "reach_m"),
    ({"band_m": -0.1}, "band_m"),
    ({"empty_keep_percent": 101}, "empty_keep_percent"),
    ({"empty_keep_percent": -1}, "empty_keep_percent"),
    ({"empty_keep_percent": 50.5}, "empty_keep_percent"),
    ({"empty_keep_percent": True}, "empty_keep_percent"),
])
def test_a_malformed_predicate_is_refused_naming_the_field(over, why):
    """Contract: an anchor outside `path`, `rooms`, `point`, an empty phrase or phase set, an `until` on a non-path
    anchor, a negative distance and a keep percent that is not a whole number from 0 to 100 are refused at
    construction, naming the field. Regression caught: a job declared with a typo in its anchor that would silently
    anchor nothing, so every `task_things` gold of the job would be empty."""
    with pytest.raises(ValueError, match=why):
        path(**over)


def test_the_job_spec_carries_looks_and_the_predicate_and_derives_its_units():
    """Contract (plan §3): a `JobSpec` has `looks` (default empty) and `task_things` (default None), `unit_m` is the
    first rung of its ladder (0.25 m in a home, 0.5 in the warehouse, 1.0 in the city) and `match_m` twice that. By
    hand over the registered jobs: vacuum/2 and helper/1 0.25 and 0.5, amr/1 0.5 and 1.0, drone/1 1.0 and 2.0.
    Regression caught: a unit read from the wrong rung, or a default that makes a job ask look questions it never
    declared."""
    expected = {"vacuum/2": (0.25, 0.5), "helper/1": (0.25, 0.5), "amr/1": (0.5, 1.0), "drone/1": (1.0, 2.0)}
    for job in job_registry.all_jobs():
        if job.id in expected:
            assert (job.unit_m, job.match_m) == expected[job.id], job.id
        assert job.unit_m == job.ladder[0] and job.match_m == 2 * job.unit_m
    assert set(expected) <= {job.id for job in job_registry.all_jobs()}
    assert JOB.looks == () and JOB.task_things is None   # a job that declares neither asks none
