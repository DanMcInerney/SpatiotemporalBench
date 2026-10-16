"""The table systems' answers to `closest_approach` and `approach_mem` (a module of `table_answers.ANSWER_MODULES`; the
golds are `questions_approach`), from the floor hulls the tracks keep of the people they detected (`Track.hulls`: the
detection's `floor_hull`, which `perception.py` sets for a person exactly when the sighting is evidence), the delivered
poses and the delivered clocks.

The rule is the gold's and is identity-free, because the sensors give no identity: it reads the hulls each step holds
(of every person track, whichever track a detection was filed under), never a track's own history. The person is the
hull of the ask's step nearest the ask's camera floor point; their velocity is `approach_rule.pair_velocity` between
that hull and the hull of the latest earlier step of the task, within `approach_rule.VEL_S` of the ask, that holds one
(the person is still when there is none, and when the ask's step or that one holds more than one hull: the pairing is
ambiguous, and the gold asks no such point). The robot's velocity is the ask's step's own `velocity` (the controller's
odometry every delivered step carries). The answer is `approach_rule.closest_m`, to 4 decimals, the cap with no person
hull in the ask's step. So every system reproduces the gold, `oracle` and `memory_table` alike, whichever way it files a
person's detections into tracks. `current_view`'s store holds the latest frame alone: no earlier step, so it answers
the held-still gold with the robot's own velocity, which is the gold of a person who stands and not that of one who
walks (the memory-needed gate; the `now` rows are the former).

**`approach_mem`** (0.11.0) is the same arithmetic for the person last seen, taken to stand where they were seen
(`approach_rule.last_seen`, which reads the hulls each step holds and no track, as above): the least predicted distance
from the ask's camera floor point, at the ask's step's own `velocity`, to the hull of the latest step of the task before
the ask that holds one, when it holds one only. It is the cap when the ask's step holds a hull (the person is in view,
which is `closest_approach`'s) or there is no last-seen person (none seen, or two at the latest sighting). So
`memory_table` and the oracle reproduce the gold on every asked row, and `current_view`'s store, which holds the latest
frame alone and so no earlier step, answers the cap: the harmful side of every near gold (the memory-needed gate).
"""

from __future__ import annotations

import math

from . import approach_rule as rule
from .nearest_part import nearest_on_hull
from .table import Answerer, Store
from .table_people import PERSON


def _hulls_by_step(store: Store) -> dict[int, list]:
    """The floor hull of every person detection with one, by step: what each frame holds, whatever the track."""
    held: dict[int, list] = {}
    for t in store.tracks:
        if t.kind == PERSON:
            for k, (_, hull) in t.hulls.items():
                held.setdefault(k, []).append(hull)
    return held


def closest_approach(store: Store, slots: dict, ctx: dict) -> dict:
    """The least predicted distance of the person in the ask's step (the module docstring), at most
    `approach_rule.CAP_M`."""
    now = ctx["now"]
    k, cam = now["k"], now["camera"]["pos"][:2]
    held = _hulls_by_step(store)
    if not held.get(k):
        return {"type": "number", "value": rule.CAP_M}
    hull = min(held[k], key=lambda h: (round(math.dist(nearest_on_hull(h, cam), cam), 9), h))
    in_task = {s["k"] for s in store.task_steps(store.task_of(k))}
    ages = [(j, rule.elapsed_s(store.steps[j]["clock"], now["clock"])) for j in held
            if j < k and j in in_task and j in store.steps]
    j = rule.latest_within(ages)
    walking = rule.pair_velocity(held[k], None if j is None else held[j], 0.0 if j is None else dict(ages)[j])
    if walking is None:   # an ambiguous pairing: the person is taken as still
        walking = (0.0, 0.0)
    return {"type": "number", "value": round(rule.closest_m(cam, tuple(now["velocity"]), hull, walking), 4)}


def approach_mem(store: Store, slots: dict, ctx: dict) -> dict:
    """The least predicted distance of the person last seen, taken to stand where they were seen (the module docstring),
    at most `approach_rule.CAP_M`."""
    now = ctx["now"]
    k, cam = now["k"], now["camera"]["pos"][:2]
    held = _hulls_by_step(store)
    in_task = [s["k"] for s in store.task_steps(store.task_of(k))]
    last = None if held.get(k) else rule.last_seen(held, k, in_task)
    if last is None:
        return {"type": "number", "value": rule.CAP_M}
    return {"type": "number", "value": round(rule.closest_m(cam, tuple(now["velocity"]), last[1], (0.0, 0.0)), 4)}


ANSWERERS: dict[str, Answerer] = {"closest_approach": closest_approach, "approach_mem": approach_mem}
