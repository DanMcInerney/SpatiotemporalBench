"""The fields the 0.5.0-control work adds to a job's declarative part: `JobSpec.subgoals`, `JobSpec.ladder` and
`Relevance.near` / `near_m`. Each defaults to what every job did before, so a job that sets none is unchanged; the
lanes that read them are tested with their own."""

from __future__ import annotations

from dataclasses import replace

from stbench import score
from stbench.geometry import Point
from stbench.job_registry import JobSpec, Relevance, get
from stbench.questions_words import DISTANCE_LADDER
from stbench.timeline_ego import TaskPlan

VACUUM = get("vacuum/1")


def bare() -> JobSpec:
    """A job built with the required fields only, as a job written before these fields was."""
    return JobSpec(id="bare/1", robot="vacuum", domain="home", task_kinds=("clean_room",), decisions={}, events={},
                   plan=VACUUM.plan, script=VACUUM.script, relevance=VACUUM.relevance, checks=VACUUM.checks,
                   anchors=frozenset(), assignments={})


def test_a_job_that_sets_no_control_field_is_unchanged():
    """Contract: a `JobSpec` built without `subgoals` and `ladder` states no subgoal (`subgoals` is None, so no pose row
    gets a `subgoal` key) and keeps the distance credit's rungs every gold row has carried (0.25, 0.5, 1 and 2 m:
    `questions_words.DISTANCE_LADDER`, which is the rungs the scorer falls back on when a row names none); a `Relevance`
    built without `near` and `near_m` has no near rule (`()` and None, so every place of its regions is critical as it
    was); `vacuum/1`, which a later lane must leave as it is, has all three defaults, and so has a job copied from it.
    Regression caught: a default that changes behaviour (a subgoal text for a job that never stated one, a ladder of
    other rungs, a near radius of zero that makes every place not critical), and a field added without a default, which
    breaks every job built by keyword."""
    spec = bare()
    assert spec.subgoals is None and spec.ladder == DISTANCE_LADDER == score.DISTANCE_LADDER == (0.25, 0.5, 1.0, 2.0)
    for job in (VACUUM, replace(VACUUM, id="other/1")):
        assert job.subgoals is None and job.ladder == DISTANCE_LADDER
    rel = Relevance(frozenset({"bag"}), frozenset({"kitchen"}), frozenset({"cleaning"}))
    assert rel.near == () and rel.near_m is None
    assert VACUUM.relevance(TaskPlan("clean_room", {"room": "bedroom"})).near_m is None


def test_the_control_fields_take_what_a_job_declares():
    """Contract: a job may set `subgoals` to a function of the timeline and the map, a `ladder` of its own rungs, and a
    relevance a `near` list of points with `near_m`; they are stored as given (the dataclasses are frozen, so a job's
    declaration is what its lanes read). Regression caught: a field that is renamed or typed so that a job lane's
    declaration (the drone's `near`/`near_m`, every job's `ladder`) cannot be built."""
    def subgoals(tl, m):
        return ["flying to the drop point"] * len(tl.ego.keyframes)

    spec = replace(bare(), subgoals=subgoals, ladder=(1.0, 2.0, 4.0, 8.0))
    assert spec.subgoals is subgoals and spec.ladder == (1.0, 2.0, 4.0, 8.0)
    point: Point = (3.0, 4.0)
    rel = Relevance(frozenset({"car"}), frozenset(), frozenset(), near=(point,), near_m=12.0)
    assert rel.near == (point,) and rel.near_m == 12.0
    assert {rel: 1}[Relevance(frozenset({"car"}), frozenset(), frozenset(), near=(point,), near_m=12.0)] == 1
