"""`drone/1`'s subgoal: the line of text that says what the drone is doing now, as part of its flight.

`subgoals(tl, m)` gives one text for each keyframe, from the timeline alone. It is the robot's own knowledge, not the
world's: it names the flight's own drop point (the plan's `address`, which the assignment states) and the phase the
drone is in, and nothing else. It is causal: the text at keyframe `k` reads the flight's plan, the phases up to and
including `k`, the decision at `k` (its kind only) and the response of the decisions strictly before `k`, so it does not
change when the timeline is cut after `k`, nor when the coin of a decision at or after `k` falls the other way. The one
response there is, the abort at `before_lower`, therefore shows from the first keyframe of the hover on, never in the
text at `before_lower` itself (the last keyframe of the descent), which is the same whatever the drop zone held.

The `inspect` at 15 m and the one at the lowering height (an inspection's hover, or an aborted delivery's) are told
apart by whether a `descend` came before; the reel-in (`climb_out` at the lowering height) and the climb by the base's
height, the hover's being the one at `before_lower`. A delivery whose response was `abort` flies the inspection's
phases and says why it holds. A plan with no `address` (a hand-built one) gets the same texts without "at <address>".
"""

from __future__ import annotations

from .job_drone_plan import ABORT, BEFORE_LOWER, DELIVER
from .mapspec import MapSpec
from .timeline import Timeline
from .timeline_ego import Keyframe, TaskPlan

LEVEL_TOL_M = 1e-6  # a keyframe is at the hover's height when it is at most this far above it (`tethers`' rule)
PHASE_TEXT = {  # a phase's text; {at} is " at <address>" (the drop point's address), empty for a plan with none
    "takeoff": "taking off from the nest",
    "approach": "flying to the drop point{at}",
    "descend": "descending over the drop point{at}",
    "lower": "lowering the package{at}",
    "release": "releasing the package",
    "land": "landing on the nest",
}
PREPARING = {True: "preparing to deliver a package to the drop point{at}",
             False: "preparing to inspect the drop point{at}"}
INSPECTING = "inspecting the drop point{at} from above"
INSPECTING_LOW = "inspecting the drop point{at} from close above"
HOLDING = "holding over the drop point{at}: it is not clear to lower"
REELING = "reeling the tether in"
CLIMBING = "climbing away from the drop point"
RETURNING = "flying back to the nest"
WITH_PACKAGE = " with the package"


def subgoals(tl: Timeline, m: MapSpec) -> list[str]:
    """One text for each keyframe of the timeline (`JobSpec.subgoals`); `m` is the signature's, the plan's `address`
    being the zone's name already. A timeline cut after a keyframe gives the same text there."""
    out: list[str] = []
    for task in tl.ego.tasks:
        decisions = {d.k: d for d in tl.ego.decisions if task.k0 <= d.k < task.k1}
        descended, aborted, hover_z = False, False, None
        for kf in tl.ego.keyframes[task.k0:task.k1]:
            out.append(_text(kf, task.plan, descended, aborted, hover_z))
            descended = descended or kf.phase == "descend"
            if (d := decisions.get(kf.k)) is not None and d.kind == BEFORE_LOWER:
                hover_z, aborted = kf.base[2], d.response == ABORT  # read from the next keyframe on
    return out


def _text(kf: Keyframe, plan: TaskPlan, descended: bool, aborted: bool, hover_z: float | None) -> str:
    """The text of one keyframe: `descended` (a `descend` keyframe came before it), `aborted` (the response of the
    flight's `before_lower`, which came before it) and `hover_z` (the base height at that decision, None before it)."""
    address, phase = plan.params.get("address"), kf.phase
    at = f" at {address}" if address else ""  # a hand-built plan may state none: the text then names no address
    if phase == "preflight":
        return PREPARING[plan.kind == DELIVER].format(at=at)
    if phase == "inspect":
        return (HOLDING if aborted else INSPECTING_LOW if descended else INSPECTING).format(at=at)
    if phase == "climb_out":
        reeling = not aborted and hover_z is not None and kf.base[2] <= hover_z + LEVEL_TOL_M
        return REELING if reeling else CLIMBING
    if phase == "return":
        return RETURNING + (WITH_PACKAGE if aborted else "")
    return PHASE_TEXT[phase].format(at=at)

