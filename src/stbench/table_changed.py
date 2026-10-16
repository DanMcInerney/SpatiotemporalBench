"""`changed` for the table systems: what a store's tracks say of where each thing stood in two tasks (`ANSWERERS`).

The room is the region of the case's public map (`ctx["map"]`) of the kind `slots["room"]` names; `slots["tasks"]` are
the two passes' case indices (`other_case`'s already moved onto its stream). Each track is read as `questions_changed`
reads an entity, from what the table holds, into one `change_events.Seen`, and `change_events.events` and `swallow`,
which the gold calls too, give the answer:
- **a sighting counts only when the thing stood on the floor then:** its box bottom (the track's centre height less half
  its height) within the robot's `climb_m` of the base height delivered at that step, as `World.on_floor` reads the
  placement. `a` and `b` are the box centre's xy at the last counted sighting in each task (`Log.task_of`);
- **a track whose description another track of the store shares** gives only unsure places, its counted sightings':
  the table tells things apart by description, so it cannot say which of two identical things moved (a thing no frame
  detected has no track, so it is never a twin here);
- **a mover**, a track with counted sightings `MOVED_YES_M` apart within one task of the log (the world acts between
  tasks, so only a person moves within one), gives only unsure places.
"Removed" and "added" are certified by `grid.place_free` over the cells of the other task's logged steps. A task the
store never held, or a pair naming one task twice, has no second pass to differ from: the answer is the empty list, as
`current_view`'s, which holds one frame and no task message, is. The table cannot read a glimpse that was detected (it
holds no pixel counts) or know a detected piece is fixed furniture; the gold reads the first, in a task where it is all
the robot saw of the thing, as unsure and never asks of the second.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from itertools import combinations

from . import change_events, grid
from .change_events import MOVED_YES_M, Seen, Xy
from .geometry import point_in_polygon
from .rules import MATCH_M
from .table import Answerer, Store, Track

# The channel gives a box's centre height and its height to 0.1 mm each, so a bottom is known to 0.2 mm: a thing on the
# floor is on it for a robot that climbs nothing (the drone's `climb_m` is 0).
BOTTOM_ROUNDING_M = 5e-4


def floor_sightings(store: Store, track: Track, climb_m: float) -> dict[int, list[Xy]]:
    """The box centre's xy at each sighting of the track that counts, by the case task holding its step (a step before
    any task message is no task's), in step order: the box stood on the floor the robot drives on, its bottom within
    `climb_m` of the base height delivered there."""
    found: dict[int, list[Xy]] = {}
    reach = climb_m + grid.EPS + BOTTOM_ROUNDING_M
    for k, x, y, z in sorted(track.sightings, key=lambda s: s[0]):
        task, bottom = store.task_of(k), z - track.size[2] / 2
        if task is not None and abs(bottom - store.steps[k]["pose"]["pos"][2]) <= reach:
            found.setdefault(task, []).append((x, y))
    return found


def read(store: Store, track: Track, i: int, j: int, climb_m: float, twin: bool) -> Seen | None:
    """The track in tasks i and j as `change_events` takes it; None when it has no counted sighting in either. It is a
    mover when it moved within any task of the log, not only these two: the table knows that of a person from the
    task in which it walked."""
    floor = floor_sightings(store, track, climb_m)
    places = sorted({p for t in (i, j) for p in floor.get(t, ())})
    if not places:
        return None
    mover = any(math.dist(p, q) >= MOVED_YES_M - grid.EPS for ps in floor.values() for p, q in combinations(set(ps), 2))
    a, b = (floor[t][-1] if t in floor else None for t in (i, j))
    return Seen(a, b, min(track.size[0], track.size[1]) / 2, tuple(places) if twin or mover else ())


def free_in(store: Store, task: int) -> Callable[[Xy, float], bool]:
    """`free(place, radius)`: the place was seen empty in the cells of the task's logged steps (`grid.place_free`)."""
    states = store.states_of([s["k"] for s in store.task_steps(task)])
    return lambda p, radius: grid.place_free(states, p, radius)


def changed(store: Store, slots: dict, ctx: dict) -> dict:
    """The places in the room where a thing was added, moved to or removed from between the two tasks (`positions`)."""
    if ctx.get("map") is None:
        raise ValueError("changed names a room: the system needs the case layout's public map (the option 'map')")
    rooms = [r for r in ctx["map"].regions if r.kind == slots["room"]]
    if len(rooms) != 1:
        raise ValueError(f"the map has {len(rooms)} rooms of the kind {slots['room']!r}; changed names a room by its "
                         f"kind, which must be unique in the layout")
    poly = list(rooms[0].polygon)
    i, j = slots["tasks"]
    twins = Counter(t.key for t in store.tracks if t.sightings)
    seens = [s for t in store.tracks if t.sightings
             if (s := read(store, t, i, j, ctx["robot"].body.climb_m, twins[t.key] > 1)) is not None]
    found, care = change_events.events(seens, lambda p: point_in_polygon(p[0], p[1], poly), free_in(store, i),
                                       free_in(store, j))
    kept, _ = change_events.swallow(found, care, MATCH_M)
    return {"type": "positions", "items": [{"xy": list(p)} for p in kept]}


ANSWERERS: dict[str, Answerer] = {"changed": changed}
