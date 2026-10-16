"""`vacuum/2`'s subgoal: the text of what the robot is doing at each keyframe, as part of its task
(FD-0.5.0-control-impl §2). It is the robot's own knowledge: the task's rooms, as the brief names them, and where the
robot is in them.

The text of a keyframe reads the phases up to it, the kinds of the decisions up to and including it and the responses of
the decisions strictly before it, never a response at or after it: a coin decided at a `before_room` has no hand in
the text there or earlier (so the `before_room` text is the same whichever way the coin falls), and the next keyframe
tells what the robot then did.
- Undocking and the docked keyframes before it: "leaving the dock to clean the kitchen, then the hall" (a resume: "to
  finish cleaning the kitchen").
- Transit to the room `j` of `n`: "driving to the kitchen to clean it (room 1 of 2)" (a resume: "to finish cleaning
  it"; no count for one room).
- Cleaning: "cleaning the kitchen (room 1 of 2)" (a resume: "finishing the kitchen").
- After a skip, the rest of that transit, up to the next doorway decision: "leaving the kitchen uncleaned: its doorway
  is blocked".
- Transit once every room is handled: "returning to the dock"; then "docking", and "docked".
"""

from __future__ import annotations

from .job_vacuum2_plan import BEFORE_DOORWAY, BEFORE_ROOM, RESUME, SKIP
from .mapspec import MapSpec
from .timeline import Timeline
from .timeline_ego import Task


def subgoals(tl: Timeline, m: MapSpec) -> list[str]:
    """One text per keyframe, from the timeline alone and causal (the module docstring says what each reads)."""
    return [text for task in tl.ego.tasks for text in _task(tl, task)]


def _task(tl: Timeline, task: Task) -> list[str]:
    """The texts of the task's keyframes (the keyframes there are: a script cut mid-task has fewer than `k1 - k0`)."""
    rooms = [str(kind) for kind in task.plan.params["kinds"]]
    resume = task.plan.kind == RESUME
    by_k = {d.k: d for d in tl.ego.decisions if task.k0 <= d.k < task.k1}
    out, done, seen, skipped, before = [], 0, set(), None, None  # rooms handled; phases seen; the room just skipped
    for kf in tl.ego.keyframes[task.k0:task.k1]:
        if before == "cleaning" and kf.phase != "cleaning":
            done += 1
        last, here = by_k.get(kf.k - 1), by_k.get(kf.k)  # a decision at the keyframe before knows its response now
        if last is not None and last.kind == BEFORE_ROOM and last.response == SKIP:
            skipped, done = rooms[done], done + 1
        elif kf.phase != "transit" or (here is not None and here.kind in (BEFORE_ROOM, BEFORE_DOORWAY)):
            skipped = None
        out.append(_text(kf.phase, rooms, resume, done, skipped, seen))
        seen.add(kf.phase)
        before = kf.phase
    return out


def _text(phase: str, rooms: list[str], resume: bool, done: int, skipped: str | None, seen: set[str]) -> str:
    n, words = len(rooms), ", then the ".join(rooms)
    count = f" (room {done + 1} of {n})" if n > 1 else ""
    if phase in ("docked", "undocking") and "docking" not in seen:
        return f"leaving the dock to {'finish cleaning' if resume else 'clean'} the {words}"
    if phase == "docked":
        return "docked"
    if phase == "docking":
        return "docking"
    if phase == "cleaning":
        return f"finishing the {rooms[done]}" if resume else f"cleaning the {rooms[done]}{count}"
    if skipped is not None:
        return f"leaving the {skipped} uncleaned: its doorway is blocked"
    if done < n:
        return f"driving to the {rooms[done]} to {'finish cleaning' if resume else 'clean'} it{count}"
    return "returning to the dock"
