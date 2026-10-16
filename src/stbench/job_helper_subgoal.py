"""`helper/1`'s subgoal: what the helper is doing at each keyframe, as the robot itself would say it
(FD-0.5.0-control-impl §2), from the timeline alone.

A keyframe's text reads the task's kind, its thing and its two pieces (`Task.targets`, which the job wrote before the
task began: `job_helper_carry.targets`), the task's phases up to and including that keyframe, and the response of the
`before_place` decision strictly before it. It reads no coin and nothing after the keyframe: a text before the decision
is the same whichever way the coin fell, and after it the helper says what it did.

- waiting, before it has left: "waiting at its spot to {move or bring} the {thing} from the {source} to the {target}"
- navigating, before it has placed: "going to the {source} to pick up the {thing}"
- grasping: "picking up the {thing} from the {source}"
- carrying: "carrying the {thing} from the {source} to the {target}"
- placing: "putting the {thing} down on the {target}"; after an `elsewhere`: "putting the {thing} down on a free
  spot, since the spot on the {target} is occupied"
- navigating, after it has placed: "going back to its spot"
- waiting, after it has placed: "waiting at its spot"

The thing is named as a person would (`descriptions.phrase`), the pieces by the map's name, with an ordinal when a name
repeats on the map ("sofa 2").
"""

from __future__ import annotations

from collections import Counter

from . import descriptions
from .job_helper_plan import BEFORE_PLACE, TIDY
from .mapspec import MapSpec
from .timeline import Timeline
from .timeline_ego import Task


def subgoals(tl: Timeline, m: MapSpec) -> list[str]:
    """One text a keyframe, causal (the module docstring). `m` is the job's signature; the names come from the
    timeline's own furniture entities, which are the map's pieces."""
    names = _names(tl)
    after = {d.k: d.response for d in tl.ego.decisions if d.kind == BEFORE_PLACE}
    out = [""] * tl.n_keyframes
    for task in tl.ego.tasks:
        thing, source, target, verb = _task(tl, task, names)
        seen: set[str] = set()
        answer: str | None = None
        for k in range(task.k0, min(task.k1, tl.n_keyframes)):
            phase = tl.keyframe(k).phase
            seen.add(phase)
            placed = "placing" in seen
            if phase == "waiting":
                out[k] = "waiting at its spot" if placed else (
                    f"waiting at its spot to {verb} the {thing} from the {source} to the {target}")
            elif phase == "navigating":
                out[k] = "going back to its spot" if placed else f"going to the {source} to pick up the {thing}"
            elif phase == "grasping":
                out[k] = f"picking up the {thing} from the {source}"
            elif phase == "carrying":
                out[k] = f"carrying the {thing} from the {source} to the {target}"
            elif answer == "elsewhere":
                out[k] = f"putting the {thing} down on a free spot, since the spot on the {target} is occupied"
            else:
                out[k] = f"putting the {thing} down on the {target}"
            if after.get(k) is not None:
                answer = after[k]  # known from the next keyframe on
    return out


def _task(tl: Timeline, task: Task, names: dict[str, str]) -> tuple[str, str, str, str]:
    """(the thing's words, the source piece's, the target piece's, the verb) of a task."""
    if len(task.targets) < 5:
        raise ValueError(f"task {task.index} of a helper script names no thing and no pieces: {list(task.targets)}")
    eid, source, target = task.targets[2:5]
    thing = tl.entity(eid)
    return (descriptions.phrase(thing.kind, thing.attributes), names[source], names[target],
            "move" if task.plan.kind == TIDY else "bring")


def _names(tl: Timeline) -> dict[str, str]:
    """The name of each fixed piece of furniture by entity id: its name on the map, with its place among the pieces of
    that name (in the map's order) when the name repeats."""
    pieces = [e for e in tl.entities if e.mobility == "fixed" and "name" in e.attributes]
    count = Counter(str(e.attributes["name"]) for e in pieces)
    seen: Counter[str] = Counter()
    out = {}
    for e in pieces:
        name = str(e.attributes["name"])
        seen[name] += 1
        out[e.id] = f"{name} {seen[name]}" if count[name] > 1 else name
    return out
