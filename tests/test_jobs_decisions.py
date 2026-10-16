"""What the four shipped jobs declare for the 0.9.0 decision questions (FD-decision-questions §6 and impl §0): the
moves a job's corridor may ask, the phases it and the heading are asked in, and the slot table the points draw from.
The table is written out by hand, so a share moved without a decision to move it fails here by job."""

from __future__ import annotations

import pytest

from stbench import job_registry
from stbench.robots import ROBOTS

# (look_shares, look_unknown_keep, clear_directions) by job: the 0.10.0 tables (lane J), three slots a point, harm
# first; each job file says how its shares follow from the fills the lanes measured. A later cut moves this pin with
# its decision in DECISIONS, never with a score.
TABLES = {
    "helper/1": ((("clear_path", 50), ("closest_approach", 50)),
                 (("ego_mem", 65), ("way_to", 35)),   # person_mem dropped (0.10.0); way_to takes the empty 35 (0.11)
                 (("ego_now", 20), ("open_heading", 15), ("approach_mem", 30))),   # approach_mem: 30 of the 65 (0.11)
    "vacuum/2": ((("clear_path", 50), ("tangle_ahead", 50)), (("ego_mem", 100),), (("open_heading", 20),)),
    "drone/1": ((("clear_path", 60),), (("ego_mem", 70),)),   # drop_zone_people dropped (final review S2)
    "amr/1": ((("closest_approach", 100),), (("ego_mem", 100),), (("ego_now", 50), ("approach_mem", 50))),   # 0.11
}
KEEPS = {"helper/1": {"clear_path": 35}, "vacuum/2": {}, "drone/1": {"clear_path": 75}, "amr/1": {}}
DIRECTIONS = {"helper/1": ("ahead", "back", "left", "right"), "vacuum/2": ("ahead", "back", "left"),
              "drone/1": ("ahead", "back"), "amr/1": ("ahead",)}


@pytest.mark.parametrize("job_id", sorted(TABLES))
def test_each_shipped_job_declares_the_briefs_table(job_id):
    """Contract (FD-questions-0.10 §3, lane J): the slot shares, the `unknown` keeps and the corridor's moves of each
    job are the final table (helper: `clear_path` 50 / `closest_approach` 50, then `ego_mem` 65 / `way_to` 35 (0.11: the
    35 that slot drew nothing on), then `ego_now` 20 / `open_heading` 15 / `approach_mem` 30 (0.11: the 30 of the 65
    that slot drew nothing on); vacuum: `clear_path` 50 / `tangle_ahead` 50,
    `ego_mem` 100, `open_heading` 20; drone: `clear_path` 60 and `ego_mem` 70, two slots since `drop_zone_people` left
    (final review S2); AMR: `closest_approach`, `ego_mem` and `ego_now`, each 100, and since 0.11 slot 3 is `ego_now`
    50 / `approach_mem` 50). Regression caught: a share moved by
    hand, a direction the brief refused (the vacuum's right, a flier's sideways) or the helper's `clear_path` `unknown`
    keep lost."""
    job = job_registry.get(job_id)
    assert job.look_shares == TABLES[job_id] and dict(job.look_unknown_keep) == KEEPS[job_id], job_id
    assert job.clear_directions == DIRECTIONS[job_id], job_id


def test_a_job_that_asks_a_corridor_or_a_heading_declares_directions_its_body_allows_and_clear_phases():
    """Contract (impl §2 J1): a job that asks `clear_path` declares directions its body takes (a ground robot's among
    ahead, back, left and right, a flier's among ahead and back) and `clear_phases`; a job that asks `open_heading`
    (which turns in place, so a ground job) declares `clear_phases` as well, since a heading is asked only where the
    robot drives. Regression caught: a corridor or a heading asked while docked, or a sideways flier."""
    for job in job_registry.all_jobs():
        if "open_heading" in job.looks:
            assert not ROBOTS[job.robot].flies and job.clear_phases, job.id
        if "clear_path" in job.looks:
            allowed = ("ahead", "back") if ROBOTS[job.robot].flies else ("ahead", "back", "left", "right")
            assert job.clear_phases and set(job.clear_directions) <= set(allowed), job.id
