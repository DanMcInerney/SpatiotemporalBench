"""Named moments of a task ("when you left the dock"), resolved from what a system is sent and nothing else.

A job declares each moment as an `EventRule` over a task's phases (`job_registry.JobSpec.events`). `resolve` finds the
matching step in the steps the stream protocol delivers, so the compiler (which golds the question) and a scripted
baseline (which answers it) resolve one wording to one step with one rule. This module imports nothing from the package:
it names no robot's phases, and a rule over a decision point is impossible because a decision point is never delivered
as such: a step has no field for one, and since a system gets one frame per `observe`, no message cut shows where a
decision point falls either (until 0.10.0 a short batch showed which step a decision point follows; that revealed
nothing about the world). The question compiler's look windows (`look_windows`) are cut after every decision point, but
a system never sees them. A question's presence can show that a frame is one of the robot's decision moments, because
asks come first at those frames (`JobSpec.look_decisions`); the robot's own status and subgoal already show that, and
the plan, never the world, decides where they fall, so a rule over a decision point still has nothing to read.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventRule:  # a named moment of a task, resolved from the delivered steps alone (their phases)
    pick: str  # "first" or "last" of the task's steps that match
    phase: str  # steps in this phase match
    after_phase: str | None = None  # only steps after the task's first step in this phase


def resolve(rule: EventRule, steps: list[dict]) -> int | None:
    """Over one task's steps as the stream protocol delivers them (each has `k` and `status.phase`): the matching step's
    `k`, or None when no step matches (the phase never occurs, or `after_phase` never does)."""
    if rule.pick not in ("first", "last"):
        raise ValueError(f"an event rule picks the 'first' or the 'last' step, not {rule.pick!r}")
    phases = [s["status"]["phase"] for s in steps]
    start = 0
    if rule.after_phase is not None:
        if rule.after_phase not in phases:
            return None
        start = phases.index(rule.after_phase) + 1
    found = [s["k"] for s, phase in zip(steps[start:], phases[start:], strict=True) if phase == rule.phase]
    if not found:
        return None
    return found[0] if rule.pick == "first" else found[-1]
