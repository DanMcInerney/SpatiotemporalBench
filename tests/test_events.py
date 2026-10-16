"""Events: a named moment of a task, resolved from the delivered steps alone."""

from __future__ import annotations

import pytest

from stbench.events import EventRule, resolve
from stbench.job_registry import get

# One vacuum task as the stream delivers it: dicts of `k` and `status` only (no decision point, no time, no pose).
PHASES = ["docked", "undocking", "undocking", "transit", "cleaning", "cleaning", "cleaning", "transit", "docking",
          "docking", "docked"]
STEPS = [{"k": 10 + i, "status": {"phase": phase, "battery_pct": 90}} for i, phase in enumerate(PHASES)]


def test_events_resolve_from_the_delivered_steps():
    """Contract: for each of `vacuum/1`'s four events, `resolve` over a task's delivered steps (keyframes 10-20: docked,
    undocking x2, transit, cleaning x3, transit, docking x2, docked) returns the one step its wording names. Hand
    values: the last undocking step is k 12, the first cleaning step k 14, the last cleaning step k 16, and the first
    docked step after the first docking one is k 20 (not the docked step that opens the task). A rule over a phase the
    task never has gives None. Regression caught: an event a system cannot resolve from what it is sent, so a scripted
    baseline would need a special case or the truth."""
    events = get("vacuum/1").events
    assert {wording: resolve(rule, STEPS) for wording, rule in events.items()} == {
        "when you left the dock": 12, "when you started cleaning": 14, "when you finished cleaning": 16,
        "when you got back to the dock": 20}
    assert resolve(EventRule("first", "paused"), STEPS) is None


def test_after_phase_needs_its_phase():
    """Contract: `after_phase` limits a rule to the steps after the task's first step in that phase, and a task without
    one has no such moment. Regression caught: an event 'after docking' that falls back to the first docked step of
    the task, which is its start."""
    rule = EventRule("first", "docked", after_phase="docking")
    assert resolve(rule, STEPS) == 20
    assert resolve(rule, STEPS[:8]) is None  # never docks
    assert resolve(rule, STEPS[:9]) is None  # starts docking, never gets back
    assert resolve(EventRule("last", "docked"), STEPS) == 20
    assert resolve(EventRule("first", "docked"), STEPS) == 10


# A task that recharges and resumes (the vacuum's sources cite iRobot's recharge and resume): it docks twice, in two
# separate runs of "docking" steps, with the charge between them.
RECHARGE = ["docked", "undocking", "cleaning", "docking", "docking", "docked", "undocking", "cleaning", "docking",
            "docking", "docked"]
RECHARGE_STEPS = [{"k": 40 + i, "status": {"phase": phase}} for i, phase in enumerate(RECHARGE)]


def test_after_phase_anchors_on_the_first_run_of_its_phase():
    """Contract (`EventRule.after_phase`): a task holding two separate runs of the `after_phase` phase anchors on the
    task's FIRST step in it, so "the first docked step after docking" is the charge between the runs. Hand values
    (k 40 to 50): the first docking step is k 43, so the first docked step after it is k 45 (the charge, not the k 50
    that ends the task) and the first undocking step after it is k 46; anchored on the last docking step (k 49) there
    would be no undocking step after it at all. Regression caught: an anchor on the last occurrence of the phase, which
    would move "when you got back to the dock" of a recharge-and-resume task to the end of its second run."""
    assert resolve(EventRule("first", "docked", after_phase="docking"), RECHARGE_STEPS) == 45
    assert resolve(EventRule("first", "undocking", after_phase="docking"), RECHARGE_STEPS) == 46
    assert resolve(EventRule("last", "docked", after_phase="docking"), RECHARGE_STEPS) == 50


def test_after_phase_excludes_its_own_step():
    """Contract: the steps that match are strictly after the anchor step, so a rule over the anchor's own phase skips
    the anchor. Hand values: with `phase == after_phase == "docking"` the first docking step (k 43) is the anchor, so
    the first matching step is the next one, k 44, and the last is the second run's second step, k 49; a task that ends
    on its anchor has none. Regression caught: an inclusive anchor, where the anchor step itself (k 43) would match and
    'after docking' would mean 'when docking began'."""
    assert resolve(EventRule("first", "docking", after_phase="docking"), RECHARGE_STEPS) == 44
    assert resolve(EventRule("last", "docking", after_phase="docking"), RECHARGE_STEPS) == 49
    assert resolve(EventRule("first", "docking", after_phase="docking"), RECHARGE_STEPS[:4]) is None


def test_an_event_rule_picks_first_or_last_and_nothing_else():
    """Contract: a rule that picks neither is a mistake in a job module, reported instead of resolving to some step, and
    a task with no steps resolves to None. Regression caught: a misspelt `pick` read as 'last', so a baseline and the
    compiler would agree on a wrong step."""
    with pytest.raises(ValueError, match="first.*last"):
        resolve(EventRule("middle", "cleaning"), STEPS)
    assert resolve(EventRule("last", "cleaning"), []) is None
