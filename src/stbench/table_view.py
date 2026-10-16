"""The table systems' answer to `out_of_view` (a module of `table_answers.ANSWER_MODULES`; `questions_view`: the gold).

The family asks where, as last seen, the movable things taller than `height` standing on the floor within `radius` of
the robot are, that no row of the steps just observed shows (`ctx["latest"]`: the frame under the harness, the gold's
look window when a caller observes one). A table keeps exactly that: each track's last sighting. Of the tracks of one
description (kind and attributes) only the most recently sighted counts (those sighted last at the same step all do),
then, for a track with a sighting before them (`b0`, their first step) and none in them, its last such sighting
`(k, x, y, z)` (the box centre) is listed when:
1. **it stood on the floor then:** the box bottom, `z - size[2] / 2`, lies within the robot's `climb_m` of the delivered
   base height at step `k` (the gold's `on_floor`, read from what the table holds);
2. **its height and distance fit:** `size[2]` is over `slots["height"]` and the place is within `slots["radius"]` of the
   ask's delivered base;
3. **the robot has not seen it gone:** the cells the logged steps after `k` and before `b0` observed do not certify its
   place empty (`grid.place_free`, radius half the shorter side of the box): the robot saw the thing was no longer there
   (the plan review's m5).
Why the newest track of a description: a thing of a unique description is one thing and stands in one place, so an older
track of its description is a ghost (a thing moved more than `MATCH_M` in a gap, and drawn at its new place by a table
that keeps a track a place, as `Store._associate` does when the description already has two tracks); the gold lists the
thing at its last seen place only (VISION rule 2: the table could know). With a twin the gold lists no item and every
place of either is don't-care, so the newest-track rule costs nothing there.
A track whose kind is the kind of a fixed piece of the public map (`ctx["map"]`'s furniture that is not light, and its
doors) is never listed: the question asks for movable things, and the map names the fixed ones (the smoke build's oracle
listed the bed, the wardrobe, the desk and the hall console; the helper's look-dev check, a door in the hall: the
timeline's doors are fixed entities the gold never lists, but a door is no furniture, so the map's openings name them).
A table built without the map raises.
The answer is `positions` (4 decimals, sorted). What a track cannot tell apart (the height and radius bands, a person's
track, a thing in view now that was only glimpsed, a glimpse after a sighting, twins, and a place seen empty since) is a
don't-care place in the gold. `current_view`'s store holds the latest frame alone, so every track has a sighting in it
and its answer is the empty list. Reads no picture: the log's poses, the tracks and the cells the mapper made.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

from . import grid
from .table import Answerer, Store, Track


def _seen_gone(track: Track, k: int, xy: tuple[float, float], latest: Callable[[], dict]) -> bool:
    """The steps after the sighting at `k` and before the observed steps saw the place empty: `grid.place_free` over the
    cells whose latest observation before them is after `k` (which is `grid.latest` over those steps alone, the
    steps `Store.states_of` would be given), the radius half the shorter side of the box. `latest()` is
    `Store.latest_states` up to the step before them."""
    since = {cell: v for cell, v in latest().items() if v[1] > k}
    return grid.place_free(since, xy, min(track.size[0], track.size[1]) / 2)


def out_of_view(store: Store, slots: dict, ctx: dict) -> dict:
    """The places of the newest tracks of their descriptions that stood on the floor within `slots["radius"]` of the
    ask's base, taller than `slots["height"]`, last seen before the observed steps and not in them, leaving out a place
    the robot saw empty since."""
    if ctx.get("map") is None:
        raise ValueError("out_of_view needs the public map: a table built without one cannot tell a fixed piece")
    fixed = ({f.kind for f in ctx["map"].furniture if f.mobility != "light"}   # the map's fixed pieces' kinds:
             | {o.type for o in ctx["map"].doors()})                           # its furniture's and its doors'
    here = ctx["now"]["pose"]["pos"][:2]
    observed = set(ctx["latest"])
    b0 = min(observed)
    climb = ctx["robot"].body.climb_m
    latest = cache(lambda: store.latest_states(b0 - 1))   # computed once, when a place is first tested
    newest: dict[tuple, int] = {}   # description -> the step it was last sighted at, by any of its tracks
    for t in store.tracks:
        newest[t.key] = max(newest.get(t.key, t.last[0]), t.last[0])
    places = []
    for t in store.tracks:
        if t.kind in fixed or t.last[0] < newest[t.key] or any(s[0] in observed for s in t.sightings):
            continue
        before = [s for s in t.sightings if s[0] < b0]
        if not before:
            continue
        k, x, y, z = before[-1]
        on_floor = abs(z - t.size[2] / 2 - store.steps[k]["pose"]["pos"][2]) <= climb + grid.EPS
        if not (on_floor and t.size[2] > slots["height"] and math.dist(here, (x, y)) <= slots["radius"] + grid.EPS):
            continue
        if not _seen_gone(t, k, (x, y), latest):
            places.append([round(x, 4), round(y, 4)])
    return {"type": "positions", "items": [{"xy": p} for p in sorted(places)]}


ANSWERERS: dict[str, Answerer] = {"out_of_view": out_of_view}
