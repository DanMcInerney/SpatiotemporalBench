"""The path of the AMR's `task_things` (0.6.0 lane R3): the whole drive of a task, whether or not its last station was
docked at. The script is `amr_kit.generated(101, 3)`, whose second task is a replenish whose last visit is skipped (a
drum in the second packing table's bay: `test_job_amr_checks`); its steps are the delivered ones (a base position and a
phase), so the anchor is what the gold and the tables both rebuild."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import amr_kit
from stbench import job_amr
from stbench.task_predicate import anchor_of

pytestmark = pytest.mark.xdist_group(name="test_amr_task_path")


def delivered(tl, task) -> list[dict]:
    return [{"pose": {"pos": list(kf.base)}, "status": {"phase": kf.phase}}
            for kf in tl.ego.keyframes[task.k0:task.k1]]


def test_a_skipped_last_stop_is_still_on_the_anchor_path():
    """Contract (0.6.0 R3, review D1): the AMR's `task_things` path is every step of the task, so the drive to a final
    stop that is blocked and skipped (the robot reaches T, waits, drives on) is part of "the way you drove". Hand
    values from the fixture: seed 101's task 1 skips its last visit (its third `before_station` answers "skip"), so its
    last `at_station` step is an earlier visit's and the skipped stop's T comes after it. The anchor is one unbroken
    chain, a segment between each two steps that follow, and it reaches T. Regression caught: a path cut at the last
    docked station (the old `until="at_station"`), which left out the drive to a skipped final stop, often 13 to 14 m,
    where real drums and pallets stood and were charged as false positives."""
    tl = amr_kit.generated(101, 3)
    task = tl.ego.tasks[1]
    ((_, dock, hold),) = amr_kit.skipped_visits(tl, task)
    assert amr_kit.asked(tl, 1, 2).response == "skip" and tuple(hold[0].base[:2]) == tuple(dock.t)
    steps = delivered(tl, task)
    last_dock = max(i for i, s in enumerate(steps) if s["status"]["phase"] == "at_station")
    assert last_dock < hold[0].k - task.k0   # the skipped stop's T is driven to after the last docked station

    anchor = anchor_of(job_amr.SPEC.task_things, steps)
    assert len(anchor.segments) == len(steps) - 1
    assert any(math.dist(end, dock.t) < 1e-6 for seg in anchor.segments for end in seg)

    old = replace(job_amr.SPEC.task_things, phases=frozenset({"transit", "approach", "at_station", "undocking"}),
                  until="at_station")   # what the path was: cut at the last docked station
    cut = anchor_of(old, steps)
    assert len(cut.segments) < len(anchor.segments)
    assert not any(math.dist(end, dock.t) < 1e-6 for seg in cut.segments for end in seg)
