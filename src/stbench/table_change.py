"""The table systems' answers to `object_was`, `same_place` and `person_where` (a module of
`table_answers.ANSWER_MODULES`; `questions_change` is the gold), each from the sightings of the tracks a description
names (`Store.matching`) and the free space the mapper kept.

A sighting is `[k, x, y, z]`. Time is the tasks': with the job's event table (`ctx["events"]`, which only a system that
keeps time is given) a thing's places are its sightings inside the task a question names (the steps of the `task`
message, `Log.task_steps`); a sighting of another task is never read, and a task the stream never had gives `unknown`.
With no event table (`latest_table`, `first_table`, `current_view`: a notebook that keeps no time) the task is not read
and the sightings are all the store holds (one a track for the notebooks: the latest, or the first):
- `object_was`: the latest sighting of the thing in the named task, `{"type": "position", "xy"}` (4 places, as
  `object_where` gives one); `unknown` when nothing matches. A notebook gives where the thing was last (or, for
  `first_table`, first) seen, whatever task is named: right at one state of a moved thing and wrong at the other.
- `same_place`: the place in the first task is the latest sighting there, the place in the second the earliest there.
  Yes when they are under `SAME_M` apart, `SAME_M` being the middle of the gap the gold leaves between a yes (under
  0.15 m) and a no (0.45 m and more). With no sighting in the second task the table says no when its mapper saw the
  first place empty in that task (`grid.place_free` over the cells of the task's steps and the thing's half-width) and
  the robot's band can show the thing gone (`can_vacate`: the robot drives, the thing stood on the floor and is taller
  than the band's floor), else `unknown`; with none in the first, `unknown`. A notebook has one place for the thing, so
  both places are it: yes whenever it holds the thing, never no, which is what makes `time_needed` read `2y - 1` for a
  yes share `y`.
- `person_where`: where the person was at the moment of the question's `event` in its task (`events.resolve` over that
  task's steps, as the gold resolves it): the latest sighting inside the task at or before it.
The gold asks only what the memory reads as the gold does (a place steady through the task, a first place nothing
contradicts, a place seen empty), so the oracle reads the same sightings and scores 1.
"""

from __future__ import annotations

import math

from . import events as events_mod
from . import grid
from .answer_candidates import UNKNOWN
from .rules import MOVED_NO_M, MOVED_YES_M
from .table import Answerer, Store, Track

SAME_M = (MOVED_NO_M + MOVED_YES_M) / 2

Seen = tuple[list[float], Track]   # a sighting [k, x, y, z] and the track that holds it


def in_task(store: Store, thing: dict, ctx: dict, task: int | None, until: float = math.inf) -> list[Seen]:
    """The sightings of the tracks `thing` (`{"kind", "attributes"}`) names that lie in the task at or before step
    `until`; with no event table, every sighting in the store."""
    seen = [(s, t) for t in store.matching(thing["kind"], thing["attributes"]) for s in t.sightings]
    if ctx["events"] is None:
        return seen
    steps = store.task_steps(task)
    return [(s, t) for s, t in seen if steps and steps[0]["k"] <= s[0] <= min(steps[-1]["k"], until)]


def _xy(s: list[float]) -> tuple[float, float]:
    return round(s[1], 4), round(s[2], 4)


def place_at(store: Store, thing: dict, ctx: dict, event: str, task: int | None) -> tuple[float, float] | None:
    """Where the thing stood at the moment `event` names in the case task `task` (None: the last task logged): the xy, 4
    places, of the latest sighting of its tracks that lies in the task at or before the moment; with no event table the
    latest sighting in the store. None when there is none or the moment never occurs."""
    until = math.inf
    if ctx["events"] is not None:
        if event not in ctx["events"]:
            raise ValueError(f"the job's events do not name {event!r}; they are {sorted(ctx['events'])}")
        until = events_mod.resolve(ctx["events"][event], store.task_steps(task))
        if until is None:
            return None
    seen = in_task(store, thing, ctx, task, until)
    return _xy(max(seen, key=lambda s: s[0][0])[0]) if seen else None


def where_then(store: Store, slots: dict, ctx: dict) -> dict:
    """The person's place at the named moment of the named task, as the gold reads it; `unknown` with none."""
    xy = place_at(store, slots, ctx, slots["event"], slots.get("task"))
    return dict(UNKNOWN) if xy is None else {"type": "position", "xy": list(xy)}


def was_during(store: Store, slots: dict, ctx: dict) -> dict:
    """The thing's place in the named task: its latest sighting there; `unknown` with none."""
    seen = in_task(store, slots, ctx, slots["task"])
    if not seen:
        return dict(UNKNOWN)
    return {"type": "position", "xy": list(_xy(max(seen, key=lambda s: s[0][0])[0]))}


def can_vacate(store: Store, track: Track, ctx: dict, sighting: list[float]) -> bool:
    """Whether the cells of the robot's band can show the thing gone, as the gold reads it
    (`questions_change._can_vacate`): the robot drives (`ctx["robot"]`), the sighting's thing stood on the floor the
    robot's base was on at that step (its base, the middle less half its height, within `climb_m`), and its top is above
    the band."""
    robot = ctx.get("robot")
    if robot is None or robot.flies:
        return False
    k, height = int(sighting[0]), track.size[2]
    base = sighting[3] - height / 2
    on_floor = abs(base - store.steps[k]["pose"]["pos"][2]) <= robot.body.climb_m + grid.EPS
    return on_floor and height > grid.band(robot)[0]


def seen_empty(store: Store, track: Track, task: int, xy: tuple[float, float]) -> bool:
    """The place `xy` was seen empty in the cells the mapper kept of the task's steps, over the thing's half-width. A
    system with no mapper, or a task with no steps, has seen nothing."""
    states = store.states_of([s["k"] for s in store.task_steps(task)])
    return grid.place_free(states, xy, min(track.size[0], track.size[1]) / 2)


def same_place(store: Store, slots: dict, ctx: dict) -> dict:
    """Yes when the thing's last place in the first task and its first in the second are under `SAME_M` apart; no when
    they are further, or when the second task never sighted it and its mapper saw the first place empty; `unknown`
    otherwise."""
    i, j = slots["tasks"]
    before, after = in_task(store, slots, ctx, i), in_task(store, slots, ctx, j)
    if not before:
        return dict(UNKNOWN)
    if ctx["events"] is None:   # a notebook keeps no time: one place, at both ends
        return {"type": "yesno", "value": True}
    last, track = max(before, key=lambda b: b[0][0])
    place = _xy(last)
    if after:
        return {"type": "yesno", "value": math.dist(place, _xy(min(after, key=lambda a: a[0][0])[0])) < SAME_M}
    if can_vacate(store, track, ctx, last) and seen_empty(store, track, j, place):
        return {"type": "yesno", "value": False}
    return dict(UNKNOWN)


ANSWERERS: dict[str, Answerer] = {"object_was": was_during, "same_place": same_place, "person_where": where_then}
