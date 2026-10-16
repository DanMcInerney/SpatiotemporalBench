"""The job registry: jobs are found by versioned id, lazily, and a job's declarative part is what the compiler, the
harness and `validate` read."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from stbench import job_registry, job_vacuum, job_vacuum2
from stbench.job_registry import JobSpec, Relevance, all_jobs, get
from stbench.mapspec import load_map
from stbench.questions_text import FAMILIES
from stbench.robots import ROBOTS
from stbench.timeline_ego import TaskPlan
from stbench.zones import ZoneRule


def test_events_name_the_jobs_own_phases():
    """Contract: every event rule of every registered job names phases of its own robot's vocabulary only. Regression
    caught: an event that resolves to nothing, or a job borrowing another robot's phases."""
    jobs = all_jobs()
    assert jobs  # the registry holds a job, so the loop below is not vacuous
    for job in jobs:
        phases = set(ROBOTS[job.robot].status.phases)
        for wording, rule in job.events.items():
            assert {rule.phase, rule.after_phase or rule.phase} <= phases, (job.id, wording)


def test_jobs_are_found_lazily():
    """Contract: `get("vacuum/1")` finds the `SPEC` of `stbench.job_vacuum` and `get("vacuum/2")` that of
    `stbench.job_vacuum2` by their ids over `JOB_MODULES`; `all_jobs()` begins with those two, lists each job that is
    written once, in `JOB_MODULES` order (vacuum, vacuum 2, helper, AMR, drone), and `get` finds each by its id; a job
    nobody has written is a KeyError naming it. Which of the M3 jobs are written is not asserted: each lane lands its
    own. Regression caught: a registry that needs every job written before any can be read, one that finds a job by a
    module name it no longer has, or one that loses, repeats or reorders a job."""
    assert get("vacuum/1") is job_vacuum.SPEC and get("vacuum/2") is job_vacuum2.SPEC
    order = ["vacuum/1", "vacuum/2", "helper/1", "amr/1", "drone/1"]  # JOB_MODULES, by hand
    jobs = all_jobs()
    ids = [j.id for j in jobs]
    assert ids[:2] == ["vacuum/1", "vacuum/2"]
    assert ids == [i for i in order if i in ids]  # each once, in module order, none outside the five
    assert all(get(j.id) is j for j in jobs)
    for missing in ("vacuum/3", "nothing/1"):
        with pytest.raises(KeyError, match=missing):
            get(missing)


def test_two_versions_of_a_job_live_side_by_side(tmp_path, monkeypatch):
    """Contract (M2, `vacuum/2` beside a frozen `vacuum/1`): with two modules whose `SPEC`s are `fake/1` and `fake/2`,
    `get` returns each by its id and `all_jobs` lists both in `JOB_MODULES` order. Regression caught: a registry that
    maps one robot (or one module) to one version, so a new version could only replace the old."""
    for module, version in (("job_fake_one", 1), ("job_fake_two", 2)):
        lines = ["from dataclasses import replace", "from stbench.job_vacuum import SPEC as VACUUM",
                 f'SPEC = replace(VACUUM, id="fake/{version}")']
        (tmp_path / f"{module}.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(job_registry, "JOB_MODULES", ("job_fake_one", "job_unwritten_stub", "job_fake_two"))
    assert get("fake/1").id == "fake/1" and get("fake/2").id == "fake/2"
    assert [j.id for j in all_jobs()] == ["fake/1", "fake/2"]
    with pytest.raises(KeyError, match=r"fake/3"):
        get("fake/3")


def test_job_fields_default_to_today():
    """Contract (M2): a `JobSpec` built without `continues` and `ends` has `continues == frozenset()` (no task kind
    carries on an earlier one) and `ends is None` (the questionnaire's default end families), and so has `vacuum/1`.
    A `Relevance` built without `floor` has `floor` False (no job asks for floor-standing things until it says so).
    Regression caught: a field added without a default, which breaks every job built by keyword."""
    vacuum = get("vacuum/1")
    bare = JobSpec(id="bare/1", robot="vacuum", domain="home", task_kinds=("clean_room",), decisions={}, events={},
                   plan=vacuum.plan, script=vacuum.script, relevance=vacuum.relevance, checks=vacuum.checks,
                   anchors=frozenset(), assignments={})
    for spec in (bare, vacuum, replace(vacuum, id="other/1")):
        assert spec.continues == frozenset() and spec.ends is None
    assert Relevance(frozenset({"bag"}), frozenset({"kitchen"}), frozenset({"cleaning"})).floor is False


def test_a_job_declares_its_zones():
    """Contract (M3): a `JobSpec` built without `zones` and `slot_zones` has `zones == {}` and `slot_zones == ()` (a
    default of its own, not one shared between specs), `vacuum/1` declares none and `vacuum/2` the doorway ahead at
    `before_room` (0.5.0-control), every registered job's two
    fields hold a dict of `ZoneRule`s and a tuple of them, and a `ZoneRule` is frozen and hashable with the defaults
    the zone rule documents (no plural words, margin 0, a band starting 0.3 m up). Regression caught: a field added
    without a default (every job built by keyword breaks), a mutable default shared by all jobs, or a zone rule that
    cannot key a table system's lookup by its words."""
    vacuum = get("vacuum/1")
    fields = dict(id="bare/1", robot="vacuum", domain="home", task_kinds=("clean_room",), decisions={}, events={},
                  plan=vacuum.plan, script=vacuum.script, relevance=vacuum.relevance, checks=vacuum.checks,
                  anchors=frozenset(), assignments={})
    one, two = JobSpec(**fields), JobSpec(**fields)
    assert one.zones == {} and one.slot_zones == ()
    assert one.zones is not two.zones
    for job in all_jobs():
        assert isinstance(job.zones, dict) and isinstance(job.slot_zones, tuple), job.id
        assert all(isinstance(r, ZoneRule) for r in (*job.zones.values(), *job.slot_zones)), job.id
    assert get("vacuum/1").zones == {}
    assert get("vacuum/2").zones == {"before_room": ZoneRule("the doorway ahead of you", "doorway", height_m=0.02)}
    assert get("vacuum/1").slot_zones == get("vacuum/2").slot_zones == ()
    rule = ZoneRule("the drop zone", "drop_zone")
    assert (rule.piece_words, rule.margin_m, rule.height_m) == ("", 0.0, 0.3)
    assert {rule: 1}[ZoneRule("the drop zone", "drop_zone")] == 1
    with pytest.raises(FrozenInstanceError):
        rule.margin_m = 1.0  # type: ignore[misc]


def test_a_broken_job_module_is_not_a_missing_one(tmp_path, monkeypatch):
    """Contract: `all_jobs` and `get` skip a module that does not exist and re-raise any other import error.
    Regression caught: a job module with a typo in an import hidden as 'not written yet', so its job silently
    vanishes from the compiler's table."""
    (tmp_path / "job_broken_stub.py").write_text("import module_nobody_has\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(job_registry, "JOB_MODULES", ("stbench.job_vacuum", "stbench.job_unwritten"))
    assert [j.id for j in all_jobs()] == ["vacuum/1"]
    monkeypatch.setattr(job_registry, "JOB_MODULES", ("job_broken_stub",))
    with pytest.raises(ModuleNotFoundError, match="module_nobody_has"):
        all_jobs()
    with pytest.raises(ModuleNotFoundError, match="module_nobody_has"):
        get("nothing/1")


def test_the_vacuums_declarative_part():
    """Contract (plan §4.3, DECISIONS "M1's review"): `vacuum/1` is the vacuum's job in the home, with one task kind,
    two decision kinds that ask no family in M1 (`path_ahead` left the doorway until M2's sites; the decision points
    stay, so the job keeps its id and its scripts), the dock as its anchor, the assignment's words, and a relevance
    that names the task's room and the hall and the phases cleaning, transit and docking; its plan, script and checks
    are tested in `test_job_vacuum.py`. Regression caught: a declarative part that drifts from the plan lanes B and C
    were built on, or a family asked at the doorway again before its gold is re-gated."""
    job = get("vacuum/1")
    assert (job.robot, job.domain, job.task_kinds) == ("vacuum", "home", ("clean_room",))
    assert job.decisions == {"before_doorway": (), "room_done": ()}
    assert job.anchors == frozenset({"dock"}) and job.swaps == ()
    assert job.assignments == {"clean_room": "Its current assignment is to clean the {room}."}
    rel = job.relevance(TaskPlan("clean_room", {"room": "bedroom"}))
    assert rel.regions == frozenset({"bedroom", "hall"})
    assert rel.phases == frozenset({"cleaning", "transit", "docking"})
    assert rel.kinds == frozenset({"shoes", "bag", "toy", "box", "chair", "stool", "laundry_basket"})
    plan = job.plan(load_map("apartment_a"), 1, 1)  # lane A's: one task, in a room other than the hall
    assert [p.kind for p in plan] == ["clean_room"] and plan[0].params["room"] != "hall"


def test_the_vacuums_second_job_asks_its_mapped_families():
    """Contract (M2, DECISIONS "M2's calls": which families a job asks is the compiler's definition): `vacuum/2` is the
    vacuum's job in the home with two task kinds, `clean_rooms` and `resume`, five decision kinds (the families it
    asks are `test_jobs_ask_060`'s); its four events are `vacuum/1`'s, its anchor the dock, a `resume` the one kind a
    window may not start at, a brief text for each kind, and its relevance the plan's rooms and the hall with
    `vacuum/1`'s floor kinds, on the robot's floor. Regression caught: a window that can start at a resume, or a kind
    with no assignment text."""
    job = get("vacuum/2")
    assert (job.robot, job.domain, job.task_kinds) == ("vacuum", "home", ("clean_rooms", "resume"))
    assert set(job.decisions) == {"before_room", "before_doorway", "room_done", "cleaning_stopped", "before_backout"}
    assert job.ladder == (0.25, 0.5, 1.0, 2.0)
    assert job.subgoals is not None and get("vacuum/1").subgoals is None
    assert job.continues == frozenset({"resume"}) and job.swaps == ()
    assert job.anchors == frozenset({"dock"}) and job.events == get("vacuum/1").events
    assert job.assignments == {"clean_rooms": "Its current assignment is to clean the {rooms}.",
                               "resume": "Its current assignment is to finish cleaning the {rooms}."}
    rel = job.relevance(TaskPlan("clean_rooms", {"rooms": "kitchen, then the hall", "kinds": ["kitchen", "hall"]}))
    assert rel.regions == frozenset({"kitchen", "hall"}) and rel.phases == frozenset({"cleaning", "transit", "docking"})
    assert rel.floor
    assert rel.kinds == get("vacuum/1").relevance(TaskPlan("clean_room", {"room": "kitchen"})).kinds
    assert get("vacuum/1").continues == frozenset()


def test_the_frozen_vacuum_asks_nothing_new_in_060():
    """Contract (0.6.0 changes the shipped jobs, not the frozen first version): `vacuum/1` keeps its M1 declaration, no
    decision kind asking a family, no looks, no `task_things` and the questionnaire's default end families (`ends`
    None), while `same_place`, which 0.5.0 took out of every job and 0.6.0 returns to the four shipped ones, stays a
    registered temporal family. Regression caught: the frozen job picking up the new questions (its scripts and renders
    would no longer answer for the version that made them)."""
    frozen = get("vacuum/1")
    assert not any(frozen.decisions.values()) and frozen.looks == () and frozen.task_things is None
    assert frozen.ends is None
    assert "same_place" in FAMILIES and FAMILIES["same_place"].temporal
