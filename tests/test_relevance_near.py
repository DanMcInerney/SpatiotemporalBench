"""The near rule of the relevance list (`Relevance.near`, `Relevance.near_m`; `questions_where.critical`): a place is
critical only within `near_m` of one of the points, in a job that sets the rule (the drone's: the ground its camera
sees from the drop point). Hand-built worlds on `space_kit`'s run over the hand map (kitchen x 0-4, living room x 4-8,
y 0-3), the bags sighted recognisably at keyframe 1."""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from space_kit import BAG_SIZE, JOB, RELEVANT_KINDS, vacuum_run
from stbench.job_registry import Relevance
from stbench.questions_where import LastSeen, Ref, critical, pools

ROOMS = frozenset({"kitchen", "living room"})
DISTANCES = (("red", 2.0), ("blue", 3.4), ("green", 3.5), ("brown", 3.6), ("orange", 6.0))   # x of a bag; y is 1


def bags():
    """Five bags in a row at y 1, x as `DISTANCES`, E0001 to E0005, and an unseen yellow one at (7, 2) (E0006)."""
    s = vacuum_run()
    for i, (colour, x) in enumerate(DISTANCES, 1):
        s.item(f"E000{i}", "bag", {"color": colour}, [(0, None, (x, 1.0, 0.0))], size=BAG_SIZE)
        s.see(1, f"E000{i}", 1500)
    s.item("E0006", "bag", {"color": "yellow"}, [(0, None, (7.0, 2.0, 0.0))], size=BAG_SIZE)
    return s


def job_near(points, near_m):
    """`JOB` whose relevance (the four kinds, the two rooms) has this near rule."""
    return replace(JOB, relevance=lambda plan: Relevance(RELEVANT_KINDS, ROOMS, frozenset(), near=points,
                                                         near_m=near_m))


def asked(job):
    """The entities `object_where` may ask answerably in the window of the bags, under `job`."""
    return [c.entities[0] for c in pools(fk.world(bags(), job=job))[0]]


def test_a_place_is_critical_only_within_near_m_of_one_of_the_points():
    """Contract: where a job sets `Relevance.near_m`, an answerable row is critical only when its place lies within
    that distance of one of `Relevance.near` (the distance itself counts), as well as in a room the task works in; an
    unknown row is critical by kind alone, since no place is known. Hand values, the points (2, 1) and 1.5 m: the bags
    at x 2.0, 3.4 and 3.5 (exactly 1.5 m) are within it and those at 3.6 and 6.0 are not; with the points (2, 1) and
    (6, 1) the bag at 6.0 is within the second; the unseen yellow bag, far from both, is still an unknown candidate.
    Regression caught: the whole room critical though the camera saw a few metres of it, so a question asks about a
    thing the robot could not have looked at, or a distance measured to the wrong end (a strict inequality drops the
    1.5 m bag)."""
    assert asked(job_near(((2.0, 1.0),), 1.5)) == ["E0001", "E0002", "E0003"]
    assert asked(job_near(((2.0, 1.0), (6.0, 1.0)), 1.5)) == ["E0001", "E0002", "E0003", "E0005"]
    w = fk.world(bags(), job=job_near(((2.0, 1.0),), 1.5))
    assert not critical(w, 0, Ref("bag", {"color": "orange"}, "E0005"), LastSeen((6.0, 1.0), 1))
    assert critical(w, 0, Ref("bag", {"color": "orange"}, "E0005"), None)
    answerable, unknown = pools(w)
    assert all(c.critical for c in answerable + unknown) and "e:E0006" in {c.subjects[0] for c in unknown}


def test_a_job_without_the_rule_keeps_every_place_in_its_rooms():
    """Contract: `near_m` None is no rule whatever `near` holds, and a rule with no point makes no place critical, as
    "within near_m of one of no points" is nowhere. Hand-worked on the five bags in the hand map's rooms: with no rule
    all five are answerable; with `near_m` None and a point still all five; with `near_m` 100 and no point none is.
    Regression caught: a job that sets nothing (the home, the warehouse, the vacuum) losing its questions to a rule it
    did not ask for."""
    every = ["E0001", "E0002", "E0003", "E0004", "E0005"]
    assert asked(job_near((), None)) == every
    assert asked(job_near(((2.0, 1.0),), None)) == every
    assert asked(job_near((), 100.0)) == []


def test_the_windows_tasks_are_joined():
    """Contract: a thing is critical in a window when it is critical under any of its tasks (kinds, rooms and the near
    rule all read per task), so a window of two tasks that work different ground asks the things of both, and a window
    of one task only its own. Hand-worked on two tasks of the vacuum whose relevance names the kitchen and the point
    (2, 1) for the first and the living room and (6, 1) for the second, 1.5 m: a bag at (2, 1) and one at (6, 1), each
    seen in its own task, are both answerable in the window of the two tasks and each alone in the window of its own
    task; a bag at (7.8, 1) in the living room, 1.8 m from the second task's point, is in none. Regression caught: the
    last task's relevance read alone, which drops everything the earlier flights and rounds worked on."""
    def relevance(plan):
        kitchen = plan.params["room"] == "kitchen"
        return Relevance(RELEVANT_KINDS, frozenset({"kitchen" if kitchen else "living room"}), frozenset(),
                         near=((2.0, 1.0) if kitchen else (6.0, 1.0),), near_m=1.5)

    s = vacuum_run(2)
    s.tasks[1] = [("living room" if leg[0] == "kitchen" else "kitchen", *leg[1:]) for leg in s.tasks[1]]
    for eid, colour, x, seen in (("E0001", "red", 2.0, 1), ("E0002", "blue", 6.0, 29), ("E0003", "green", 7.8, 29)):
        s.item(eid, "bag", {"color": colour}, [(0, None, (x, 1.0, 0.0))], size=BAG_SIZE)
        s.see(seen, eid, 1500)
    job = replace(JOB, relevance=relevance)
    assert [c.entities for c in pools(fk.world(s, 0, 1, "c1", "medium", job))[0]] == [("E0001",), ("E0002",)]
    assert [c.entities for c in pools(fk.world(s, 0, 0, "c1", "short", job))[0]] == [("E0001",)]
    assert [c.entities for c in pools(fk.world(s, 1, 1, "c1", "short", job))[0]] == [("E0002",)]
