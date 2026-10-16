"""What a job must declare to ask the 0.8.0 clearance families (DECISIONS, "0.8.0 declared"; review §2.4, §2.5, §3.1):
properties of the registered jobs, never the declared numbers (the numbers are the review's, in the job files)."""

from __future__ import annotations

from dressing_kit import render_module
from stbench import job_registry
from stbench.robots import ROBOTS


def asking(family: str):
    return [job for job in job_registry.all_jobs() if family in job.looks]


def test_a_job_that_asks_clear_path_declares_the_phases_it_drives_in():
    """Contract (review §2.5): `clear_path` is asked only in the phases in which the robot drives, because in a phase
    where it waits its own status answers the corridor (AMR at a station +0.55, helper waiting +1.00, vacuum docked
    +1.00). So a job that asks it declares `clear_phases`, every one a phase of its robot's status vocabulary (a typo
    would silently ask nothing), none the robot's start phase (where it rests), and a job that does not ask it declares
    none. Regression caught: a corridor asked while docked, waiting or landed; a phase name that matches no status."""
    for job in job_registry.all_jobs():
        phases = set(ROBOTS[job.robot].status.phases)
        if "clear_path" not in job.looks:
            assert not job.clear_phases, job.id
            continue
        assert job.clear_phases and job.clear_phases <= phases, job.id
        if not ROBOTS[job.robot].flies:
            assert ROBOTS[job.robot].status.start["phase"] not in job.clear_phases, job.id


def test_a_flier_that_asks_clear_path_declares_a_fly_floor_and_every_phase():
    """Contract (review §2.4): a flier's corridor is asked at the heights above its place's fixed structure and
    nowhere else (landed on the nest, or hovering beside the street trees' crowns, a straight flight meets benches and
    crowns the gold does not hold), so a flier that asks it declares `fly_floor_m`, and since the floor is what decides
    it every phase of its status is a clear phase; a ground job declares no floor. Regression caught: the landed
    drone's corridor (36 of 41 land points were at 0.27 m), or a phase left out of the flier's list that would drop the
    descent's and the inspection's points a floor would keep."""
    for job in job_registry.all_jobs():
        robot = ROBOTS[job.robot]
        if robot.flies and "clear_path" in job.looks:
            assert job.fly_floor_m is not None and job.fly_floor_m > 0, job.id
            assert job.clear_phases == frozenset(robot.status.phases), job.id
        else:
            assert job.fly_floor_m is None, job.id


def test_the_drones_fly_floor_clears_the_street_lamps_the_render_stands_in_the_place():
    """Contract (final review S2): the flier's gold knows only the map's walls, so its floor must clear whatever else
    the render fixes in the place at flying height: the street lamps `scenery_layout` stands every 28 m in the kerb
    strip of every sidewalk, 7.95 m to the top of the head. The base is the body's middle, so its underside is half the
    body's height below it. Hand-worked: 7.95 + 0.30 / 2 = 8.10 m must not exceed the floor. Regression caught: the
    floor left at the tree crowns' 7.3 m, where a drone's body meets a lamp head that the walls-only gold ignores."""
    lamp = render_module("scenery_layout").POST_HEIGHT_M["lamp"]
    fliers = [job for job in asking("clear_path") if ROBOTS[job.robot].flies]
    assert fliers
    for job in fliers:
        assert job.fly_floor_m >= lamp + ROBOTS[job.robot].body.height_m / 2, job.id


def test_the_vacuum_asks_its_corridor_only_while_it_cleans():
    """Contract (final review S3): the vacuum leaves its dock on a fixed route, so in `transit` the same poses and golds
    recur in different scripts and a held-out constant per phase scored +0.103 over `all_unknown` on those rows, past
    the 0.10 form margin; `transit` is not a clear phase of `vacuum/2`, `cleaning` is. Regression caught: the corridor
    asked on the route out of the dock again (a layout fact read through the status)."""
    assert job_registry.get("vacuum/2").clear_phases == frozenset({"cleaning"})


def test_a_job_that_asks_person_mem_declares_one_person():
    """Contract (review §3.1): `person_mem` asks about "the person", which names one only in a world that holds at most
    one, so a job that asks it declares `one_person` (its checks then refuse a world with two). Regression caught:
    "the person" asked with two candidates in the world, or a job asking it with the guarantee undeclared."""
    for job in asking("person_mem"):
        assert job.one_person, job.id
