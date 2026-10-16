"""Whose floor the registered jobs' relevance counts (0.5.0, DECISIONS "0.5.0 declared") and that no job asks
`where_was_i`. `Relevance.floor` marks a robot that works the floor, which the critical rule reads (FD-0.5.0-impl §2,
"Critical"). What the jobs ask is `test_jobs_ask_060`'s."""

from __future__ import annotations

from stbench.job_registry import all_jobs, get
from stbench.mapspec import load_map
from stbench.questions_text import FAMILIES

MAP_OF = {"home": "apartment_a", "warehouse": "warehouse_a", "delivery": "downtown_a"}   # a plan's map, by domain
# The floor of each job whose robot is the one that ships (`vacuum/1` is the frozen first version: not read).
FLOOR = {"vacuum/2": True, "amr/1": True, "helper/1": False, "drone/1": False}


def test_no_job_asks_where_was_i():
    """Contract (DECISIONS, "0.5.0 declared": `where_was_i` is dropped, a lookup of the frame header's own pose): no
    registered job lists it at any decision kind, in its looks or in its ends; the family stays in the registry, so
    the drop is the jobs' and no family spec is edited. Regression caught: a job asking a family it cannot supply."""
    jobs = all_jobs()
    assert {job.id for job in jobs} >= set(FLOOR)   # the registry holds the four jobs, so the loop is not vacuous
    for job in jobs:
        asked = {f for families in job.decisions.values() for f in families} | set(job.looks) | set(job.ends or ())
        assert "where_was_i" not in asked, job.id
    assert "where_was_i" in FAMILIES


def test_the_floor_is_set_on_the_vacuum_and_the_amr_only():
    """Contract (FD-0.5.0-impl §2, "Critical"): the relevance of every plan of `vacuum/2` and `amr/1` (the robots that
    work the floor) has `floor` set and that of every plan of the helper and the drone (it lifts things from furniture;
    it flies) has not. Regression caught: floor set on the wrong robots (the helper's critical things would then be
    only those on the floor, and the drone's none), or never set on the floor robots (a thing on a table would read as
    critical for a robot that cannot reach it)."""
    for job_id, floor in FLOOR.items():
        job = get(job_id)
        plans = job.plan(load_map(MAP_OF[job.domain]), 1, 3)
        assert plans, job_id
        assert {job.relevance(plan).floor for plan in plans} == {floor}, job_id
