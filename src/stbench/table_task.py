"""The table systems' answer to `task_things` (a module of `table_answers.ANSWER_MODULES`; `questions_task`: the gold).

The question asks for the things a job's predicate selects in one task of the window (`slots["task"]`, the case index;
`ctx["task_things"]`, the job's `task_predicate.TaskThings`). A table holds exactly what it needs: the task's steps (the
phases and base positions the anchor is made of: `task_predicate.anchor_of`, the code the gold calls) and each track's
sightings. A track is listed, at its last sighting in the task, when:
1. **it is a movable thing:** its kind is no fixed piece of the public map (`ctx["map"]`: the furniture that is not
   light, and its doors and arches, which the timeline holds as entities), not in `dont_care_kinds`, and, when the
   predicate names `item_kinds` (the vacuum's list of what its brush could catch), one of them;
2. **it is tall enough:** its extent over `min_height_m`;
3. **it stood on the floor** when the predicate says `floor`: the box bottom, the detected centre less half the height,
   within the robot's `climb_m` of the base height delivered at that step (`task_predicate.on_floor`, the gold's rule);
4. **it reaches the predicate:** the footprint a detection gives (the extents about the centre, with no heading)
   `task_predicate.reaches` the anchor.
A track with no sighting in the task, and a task the store never held (a `current_view` store holds one frame, so it
holds none of an earlier task), give nothing; a task with no anchor gives the empty list. What the table cannot tell
apart (a twin, a person, light furniture, a thing in the end look window or only glimpsed, one near the band's edge) the
gold makes a don't-care place, which the scorer drops. Reads no picture: the log's steps and the tracks.
"""

from __future__ import annotations

from .answers import MAX_SET_ITEMS
from .table import Answerer, Store
from .task_predicate import anchor_of, detected_rect, on_floor, reaches

# The channel gives a box's centre height and its height to 0.1 mm each, so a bottom is known to 0.2 mm (the same
# slack `table_changed.floor_sightings` allows).
BOTTOM_ROUNDING_M = 5e-4


def fixed_kinds(public_map) -> set[str]:
    """The kinds of the map's fixed furniture and of its openings: what a question about movable things never lists."""
    return ({f.kind for f in public_map.furniture if f.mobility != "light"} | {o.type for o in public_map.doors()})


def task_things(store: Store, slots: dict, ctx: dict) -> dict:
    """The last places, in the task `slots["task"]`, of the tracks that reach the job's predicate (the module
    docstring), 4 decimals, sorted. ValueError for a job with no predicate or a table built without the public map."""
    pred = ctx.get("task_things")
    if pred is None:
        raise ValueError("task_things needs the job's predicate: the system's option 'task_things' is None")
    if ctx.get("map") is None:
        raise ValueError("task_things needs the public map: a table built without one cannot tell a fixed piece")
    steps = store.task_steps(slots["task"])
    anchor = anchor_of(pred, steps, ctx["map"])
    places = []
    if anchor is not None:
        fixed, in_task = fixed_kinds(ctx["map"]) | pred.dont_care_kinds, {s["k"] for s in steps}
        climb = ctx["robot"].body.climb_m + BOTTOM_ROUNDING_M
        for t in store.tracks:
            seen = [s for s in t.sightings if s[0] in in_task]
            if not seen or t.kind in fixed or t.size[2] <= pred.min_height_m:
                continue
            if pred.item_kinds and t.kind not in pred.item_kinds:
                continue
            k, x, y, z = max(seen, key=lambda s: s[0])
            base_z = store.steps[k]["pose"]["pos"][2]
            if pred.floor and not on_floor(z - t.size[2] / 2, base_z, climb):
                continue
            if reaches(pred, anchor, detected_rect((x, y), t.size)):
                places.append([round(x, 4), round(y, 4)])
    return {"type": "positions", "items": [{"xy": p} for p in sorted(places)[:MAX_SET_ITEMS]]}


ANSWERERS: dict[str, Answerer] = {"task_things": task_things}
