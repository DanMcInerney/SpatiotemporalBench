"""`amr/1`'s subgoal: the text of what the robot is doing at each keyframe, as part of its task (`JobSpec.subgoals`).

It is the robot's own knowledge: the task's plan and the map's names (`phrase`: the furniture's name, with an ordinal
when the name repeats on the map, "packing table 2"; a pick's aisle by its region's name), and nothing it has not been
told. It reads the timeline alone and is **causal**: one forward scan over a task's keyframes whose state is the phases
and decisions of the keyframes so far. A keyframe's text is written before its own decision is counted, so it never
depends on a coin decided at or after it; the response to a `before_station` shows only in the keyframes after it, by
the phases they are in (docking, or waiting at T and skipping), which are what the response made.

Texts, by the run of phases the keyframe is in (`{stop}` the station's phrase, `{purpose}` from the task's kind and the
station's kind):
- `waiting` at the start, `undocking` off the charger: "leaving the charger";
- `transit` to the n-th station: "driving to {stop} to {purpose} (stop n of N)", where the purpose is "pick up totes"
  or "drop off totes" (a replenish's shelving and tables, a pick's table), "collect empty totes" or "return the empty
  totes" (a collect's tables and shelving); to a pick face: "driving into the {aisle} to pick (pick i of n)"; after the
  last stop: "returning to the charger";
- `approach` to a station: "docking at {stop}"; `at_station`: "loading at {stop}" (picking up or collecting) or
  "unloading at {stop}"; at a pick face: "waiting at the pick face while it is loaded (pick i of n)"; `undocking` off a
  station: "reversing out of {stop}";
- `waiting` at T after a `before_station`, which is the skip: "waiting at {stop}: its bay is blocked"; the transit that
  follows begins "skipping {stop}; " before its own text;
- `approach` onto the charger: "returning to the charger"; the closing `waiting`: "charging".

A plan that lists no stations (a hand-built one, as the test kits make) is read with generic words, never a KeyError:
the station is "the station", the leg "driving to the next stop" with no count, the aisle "the aisle"."""

from __future__ import annotations

from .job_amr_plan import BEFORE_STATION, COLLECT, PICK, SHELVING
from .mapspec import Furniture, MapSpec
from .timeline import Timeline
from .timeline_ego import Task

LEAVING, RETURNING, CHARGING = "leaving the charger", "returning to the charger", "charging"
THE_STATION, NEXT_STOP, THE_AISLE = "the station", "driving to the next stop", "aisle"  # a plan that names none
LOADING_PURPOSES = ("pick up totes", "collect empty totes")  # the others unload


def _label(m: MapSpec, f: Furniture) -> str:
    """The piece's name, with its ordinal among the map's pieces of that name when there are several."""
    same = [g.id for g in m.furniture if g.name == f.name]
    return f.name if len(same) == 1 else f"{f.name} {same.index(f.id) + 1}"


def phrase(m: MapSpec, f: Furniture) -> str:
    """How the subgoal names a station: "the shelving unit", "packing table 2" (an ordinal stands for the article)."""
    label = _label(m, f)
    return label if label[-1].isdigit() else f"the {label}"


def purpose(kind: str, station: Furniture) -> str:
    """What the robot does at a station of a task of this kind."""
    shelf = station.kind == SHELVING
    if kind == COLLECT:
        return "return the empty totes" if shelf else "collect empty totes"
    return "pick up totes" if shelf else "drop off totes"


def _task_texts(tl: Timeline, m: MapSpec, task: Task, marks: dict[int, str]) -> list[str]:
    """The texts of the task's keyframes (those the timeline holds), one forward scan."""
    plan, by_id = task.plan, m.furniture_by_id()
    stations = [by_id[s] for s in plan.params.get("stations", ()) if s in by_id]
    faces = plan.params.get("faces", []) if plan.kind == PICK else []
    aisle = next((r.name for r in m.regions if r.id == plan.params.get("aisle")), THE_AISLE)
    named = bool(stations)  # the plan names its stops: otherwise the texts are generic
    out: list[str] = []
    prev, prev_mark, text = None, None, ""
    seen, done, in_face, blocked = 0, 0, False, False  # before_station decisions and pick-face holds before this one
    for k in range(task.k0, min(task.k1, tl.n_keyframes)):
        phase = tl.keyframe(k).phase
        if phase != prev:
            if prev == "at_station" and in_face:
                done += 1  # a pick-face hold has just ended
            at = stations[seen - 1] if 0 < seen <= len(stations) else None  # the visit under way (after its decision)
            name = phrase(m, at) if at is not None else THE_STATION
            lead = f"skipping {name}; " if blocked else ""  # the leg after a skip
            blocked = False
            if phase == "waiting" and prev is None:
                text = LEAVING
            elif phase == "waiting" and prev == "approach":
                text = CHARGING
            elif phase == "waiting":  # at T, after the ask: the bay is blocked and the robot skips the visit
                text, blocked = f"waiting at {name}: its bay is blocked", True
            elif phase == "undocking":
                text = LEAVING if prev in (None, "waiting") else f"reversing out of {name}"
            elif phase == "transit":
                if done < len(faces):
                    text = f"driving into the {aisle} to pick (pick {done + 1} of {len(faces)})"
                elif seen < len(stations):
                    where = stations[seen]
                    text = (f"driving to {phrase(m, where)} to {purpose(plan.kind, where)} "
                            f"(stop {seen + 1} of {len(stations)})")
                else:
                    text = RETURNING if named else NEXT_STOP
                text = lead + text
            elif phase == "approach":
                text = f"docking at {name}" if prev_mark == BEFORE_STATION else RETURNING
            else:  # at_station
                in_face = done < len(faces)
                if in_face:
                    text = f"waiting at the pick face while it is loaded (pick {done + 1} of {len(faces)})"
                else:
                    verb = "working" if at is None else (
                        "loading" if purpose(plan.kind, at) in LOADING_PURPOSES else "unloading")
                    text = f"{verb} at {name}"
        out.append(text)
        prev, prev_mark = phase, marks.get(k)
        if prev_mark == BEFORE_STATION:
            seen += 1  # counted after the keyframe's own text, so a decision's text never depends on its own coin
    return out


def subgoals(tl: Timeline, m: MapSpec) -> list[str]:
    """The subgoal text of every keyframe of the timeline, in order."""
    marks = {d.k: d.kind for d in tl.ego.decisions}
    out = [""] * tl.n_keyframes
    for task in tl.ego.tasks:
        for k, text in enumerate(_task_texts(tl, m, task, marks), start=task.k0):
            out[k] = text
    return out
