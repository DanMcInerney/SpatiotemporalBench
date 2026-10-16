"""What the four shipped jobs ask at look points and in what order (FD-questions-0.10 §3, lane J): the hazard families
wired to each job, the slots a point draws harm first, and the status phases the hazard families are asked in. The
shares themselves are pinned, by job, in `test_jobs_decisions.py`; this file holds the contracts they must keep."""

from __future__ import annotations

import pytest

from stbench import job_registry
from stbench.abilities import FAMILY_KIND
from stbench.questions_text import LOOKS
from stbench.robots import ROBOTS

HAZARDS = {"amr/1": {"closest_approach", "approach_mem"}, "helper/1": {"closest_approach", "approach_mem"},
           "vacuum/2": {"tangle_ahead"},
           "drone/1": set()}   # drop_zone_people left the drone: 2 rows over the dev root (final review S2)
PHASES = {"vacuum/2": {"tangle_ahead": {"cleaning", "transit"}},
          "helper/1": {"way_to": {"navigating", "carrying"}}}   # 0.11: where the helper drives, as its `clear_path`
SLOTS = {"amr/1": 3, "helper/1": 3, "vacuum/2": 3, "drone/1": 2}   # the drone has no third family to ask
# the harm families a later slot may hold: `person_mem` is a memory family (slot 2 of the helper, which has the only
# person to remember; it is no longer asked), `drop_zone_people` took a slot of its own when it was drawn, and
# `approach_mem` (0.11) is the memory twin of slot 1's `closest_approach`, drawn in slot 3 from the share that slot
# left free (FD-0.11-hard 3.2)
LATER_HARM = {"person_mem", "drop_zone_people", "approach_mem"}


def shipped() -> list:
    return [job for job in job_registry.all_jobs() if job.looks]


def test_each_job_asks_the_hazard_families_of_its_place():
    """Contract (FD-questions-0.10 §2): `closest_approach` is asked of the AMR and the helper (the people their planners
    slow for), `tangle_ahead` of the vacuum (the cable its brush could catch), `drop_zone_people` of no job (the drone
    asked it until the final review's S2: 2 rows over 36 dev cases, under `abilities.MIN_ABILITY_N`, with no readable
    gate; its family code and page stay), and no job asks a hazard family of another place; `closest_person` stays in
    the AMR's end (S6). Regression caught: a hazard family wired to the wrong robot (a floor distance from a drone's
    15 m, a brush's cable for a tote robot) or left unwired, `drop_zone_people` back on the drone without a dataset
    that has the supply for it, and the end question lost with the new family."""
    new = {"closest_approach", "approach_mem", "tangle_ahead", "drop_zone_people"}
    assert {job.id: set(job.looks) & new for job in shipped()} == HAZARDS
    assert "closest_person" in job_registry.get("amr/1").ends


@pytest.mark.parametrize("job", shipped(), ids=lambda job: job.id)
def test_a_point_draws_its_slots_harm_first_and_no_slot_exceeds_100(job):
    """Contract (FD-questions-0.10 §3): a job's look table has three slots that sum to at most 100 each (the drone,
    which asks two families, two); slot 1 holds the job's harm families alone, so a point's first question is the harm
    one wherever the job has supply for it; the later slots hold no other harm family but the two that cannot share
    slot 1 (`person_mem`, a memory family, and the diluted `drop_zone_people`), and every family of `looks` is drawn in
    some slot. By hand on the declared tables, from the kinds of `abilities.FAMILY_KIND`. Regression caught: a table
    with a slot lost (a family lost), a slot over 100 (a family never drawn, which the registry refuses), a situation
    question drawn ahead of the harm one, or a harm family drawn after the situation questions that the order was to
    put behind it."""
    slots = job.look_slots
    assert len(slots) == SLOTS[job.id] and all(sum(percent for _, percent in slot) <= 100 for slot in slots)
    first = [family for family, _ in slots[0]]
    assert first and {FAMILY_KIND[f] for f in first} == {"harm"}
    later = {family for slot in slots[1:] for family, _ in slot}
    assert {f for f in later if FAMILY_KIND[f] == "harm"} <= LATER_HARM
    assert {family for slot in slots for family, _ in slot} == set(job.looks)


@pytest.mark.parametrize("job_id", sorted(PHASES))
def test_the_hazard_families_are_asked_in_the_phases_their_job_declares(job_id):
    """Contract (`JobSpec.look_phases`): the vacuum asks `tangle_ahead` in `cleaning` and `transit` (the brush running
    or the drive to the next strip), and the helper asks `way_to` (0.11) in `navigating` and `carrying`, where it
    drives, and each phase is one of its robot's. A job names no family it does not ask. The other jobs declare none
    (the drone's `drop_zone_people` phases left with the family, final review S2). Regression caught: a phase typed
    apart from the robot's vocabulary (the family would never be asked), a family asked docked or in flight elsewhere,
    or a phase rule lost from the job file when the families were wired."""
    job = job_registry.get(job_id)
    assert {family: set(phases) for family, phases in job.look_phases.items()} == PHASES[job_id]
    assert all(set(phases) <= set(ROBOTS[job.robot].status.phases) for phases in job.look_phases.values())
    assert set(job.look_phases) <= set(job.looks)
    assert all(not other.look_phases for other in shipped() if other.id not in PHASES)


def test_the_drone_asks_no_drop_zone_people_and_the_family_stays_for_a_dataset_with_supply():
    """Contract (final review S2, DECISIONS 2026-10-14): the drone's looks, slot table and phase rules name no
    `drop_zone_people` (2 rows over the 36 dev cases, under `abilities.MIN_ABILITY_N`, in the drone's counted clearance
    `now` cell and with no per-family diagnostic), while the family itself is still registered with its gold, so a
    dataset with more supply can ask it again by wiring one job. By hand from the drone's file: two families, two
    slots. Regression caught: the family still drawn on the drone (two rows that no gate can read), or deleted with it
    (the code and page are kept)."""
    drone = job_registry.get("drone/1")
    assert drone.looks == ("clear_path", "ego_mem")
    assert {family for slot in drone.look_slots for family, _ in slot} == {"clear_path", "ego_mem"}
    assert not drone.look_phases
    assert "drop_zone_people" in FAMILY_KIND and "drop_zone_people" in LOOKS
