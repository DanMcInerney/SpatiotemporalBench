"""`amr/1`'s own `validate` rules (`job_amr_checks`) for the skip (0.5.0-control): each rule flags a script that breaks
it, on a script of `amr_kit.generated` (one xdist group of its own). Seed 101's first task is a replenish of three
stations whose first visit is skipped (a drum in the bay of the shelving unit F11) and whose other two dock; its second
task is a replenish whose last visit is skipped (the second packing table)."""

from __future__ import annotations

from dataclasses import replace

import pytest

import amr_kit
from stbench.job_amr_checks import checks
from stbench.timeline_ego import DecisionPoint, Ego

pytestmark = pytest.mark.xdist_group(name="test_job_amr_checks")

M = amr_kit.M


@pytest.fixture(scope="module")
def script():
    tl = amr_kit.generated(101, 3)
    assert [amr_kit.asked(tl, 0, j).response for j in range(3)] == ["skip", "go", "go"]
    assert amr_kit.asked(tl, 1, 2).response == "skip"
    return tl


def codes(tl) -> set[str]:
    return {p.code for p in checks(tl, M)}


def test_a_generated_script_passes_its_own_rules(script):
    """Contract: the job's rules find nothing in a script the job generated, skips, docked visits and a skipped last
    stop included. Regression caught: a rule that refuses what the job writes (every seed would then fail to
    generate)."""
    assert checks(script, M) == []


@pytest.mark.parametrize(("edit", "code"), [
    ("a skip with a clear bay", "job_response"),
    ("a go with a drum in the bay", "job_response"),
    ("a response that is neither", "job_response"),
    ("a before_station that answers nothing", "job_response"),
    ("a response on a before_undock", "job_response"),
    ("a skip that docks", "job_phase_order"),
    ("a dock that waits", "job_phase_order"),
    ("an undock after a skip", "job_decisions"),
    ("loaded dropped after a skipped last stop", "job_loaded"),
    ("loaded kept after a docked last stop", "job_loaded"),
    ("loaded true before the first hold", "job_loaded"),
    ("targets that are not the plan's", "job_targets"),
])
def test_the_rules_for_the_skip_have_teeth(script, edit, code):
    """Contract (0.5.0-control): `checks` flags a script in which a `before_station` answers anything but "go" or
    "skip", "skip" with a clear bay or "go" with a drum across the docking line (the robot must answer the world),
    another decision that carries a response, a visit whose phases are not its response's (a skip that approaches, a
    dock that only waits), a `before_undock` for a skipped visit, `loaded` that is not the rule's (dropped after a
    skipped last stop, kept after a docked one, true before the first hold) or targets that are not the plan's.
    Regression caught: a rule of the job that no script can break."""
    tl = script
    kfs, decisions, tasks = list(tl.ego.keyframes), list(tl.ego.decisions), list(tl.ego.tasks)
    index = {d.k: i for i, d in enumerate(decisions)}

    def answer(i: int, j: int, response):
        d = amr_kit.asked(tl, i, j)
        decisions[index[d.k]] = replace(d, response=response)
        return d

    if edit == "a skip with a clear bay":
        answer(0, 1, "skip")
    elif edit == "a go with a drum in the bay":
        answer(0, 0, "go")
    elif edit == "a response that is neither":
        answer(0, 1, "maybe")
    elif edit == "a before_station that answers nothing":
        answer(0, 1, None)
    elif edit == "a response on a before_undock":
        d = next(d for d in decisions if d.kind == "before_undock")
        decisions[index[d.k]] = replace(d, response="go")
    elif edit == "a skip that docks":
        run = amr_kit.visits(tl, tl.ego.tasks[0])[0][3]
        for kf in run[1:]:
            kfs[kf.k] = replace(kf, phase="approach")
    elif edit == "a dock that waits":
        run = amr_kit.visits(tl, tl.ego.tasks[0])[1][3]
        for kf in run:
            if kf.phase == "approach":
                kfs[kf.k] = replace(kf, phase="waiting")
    elif edit == "an undock after a skip":
        run = amr_kit.visits(tl, tl.ego.tasks[0])[0][3]
        decisions.append(DecisionPoint(run[-1].k, "before_undock"))
        decisions.sort(key=lambda d: d.k)
    elif edit == "loaded dropped after a skipped last stop":
        last = kfs[tl.ego.tasks[1].k1 - 1]
        kfs[last.k] = replace(last, status={**last.status, "loaded": False})
    elif edit == "loaded kept after a docked last stop":
        last = kfs[tl.ego.tasks[0].k1 - 1]
        kfs[last.k] = replace(last, status={**last.status, "loaded": True})
    elif edit == "loaded true before the first hold":
        kf = kfs[tl.ego.tasks[0].k0 + 5]
        kfs[kf.k] = replace(kf, status={**kf.status, "loaded": True})
    else:
        tasks[0] = replace(tasks[0], targets=("F09",))
    broken = replace(tl, ego=Ego(kfs, tasks, decisions))
    assert code in codes(broken)
