"""`path_log`'s answer to `adjacent` (a module of `baselines_path.PATH_MODULES`; `questions_adjacent` is the gold).

"Give a point in the room that connects the kitchen to the bedroom." The family is asked only of a layout the agent has
never seen, and only of rooms whose identity the camera alone gives: no brief names them, the job's relevance does not
list them, the robot did not start in them, and the connector is neither where it started nor where it ended.
`path_log` reads no picture and holds no geometry of a held-out layout (the region kinds only:
`suite_priors.options_for`), so it can place none of those rooms and abstains: the best response over its pool of other
scripts' golds (`answer_keys.levels("path_log", ...)`), which is empty by rule on a held-out layout
(`suite_priors.prior_records`: a fact of the layout is learned from public layouts only, none of which asks it), so
`unknown`.

The route is still here, so that a leak in the moment rule (a pair a brief names, asked) would show as `path_log`
scoring. A room it can place is one a brief names, and where it worked is the logged base positions of the first task
whose brief names that kind and no other room (`answer_keys.rooms_named` over the layout's kinds). Given both rooms
placed, the room that connects them is where its routes to the two part: the last position of its route to the first
that is still on its route to the second (within `SHARED_M` of one of its positions), taken in order. A route to the
room it started in is not placed: its kind is a prior of the public layouts ("the dock stands in the hall"), which
`path_log` holds no job knowledge for, and the pool cannot give. The claim is an extra candidate of the best response,
and the answer with no pool.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .answer_candidates import UNKNOWN
from .answer_keys import rooms_named
from .blind import learned_answer, pool_at

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog

SHARED_M = 0.5   # a position this near one of the other route's is on that route: two routes sample one path unevenly
Xy = tuple[float, float]


def route_to(pl: PathLog, kind: str) -> list[Xy]:
    """The base positions, in order, of the steps of the first task whose brief names `kind` and no other room of the
    layout; empty when no brief does."""
    for task in pl.log.tasks:
        if rooms_named([task.brief], pl.kinds) == (kind,):
            return [(s["pose"]["pos"][0], s["pose"]["pos"][1]) for s in pl.log.task_steps(task.index)]
    return []


def parting(first: list[Xy], second: list[Xy]) -> Xy | None:
    """The last position of `first`, in order, before its first position that is off `second`'s route (not within
    `SHARED_M` of any of its positions); None when the first is already off it."""
    last = None
    for p in first:
        if not any(math.dist(p, q) <= SHARED_M for q in second):
            break
        last = p
    return last


def claim(pl: PathLog, a: str, b: str) -> dict | None:
    """Where its routes to the two rooms part, as a position answer; None when a brief names neither room alone or
    the routes never share a position."""
    if not (first := route_to(pl, a)) or not (second := route_to(pl, b)) or (at := parting(first, second)) is None:
        return None
    return {"type": "position", "xy": [round(at[0], 4), round(at[1], 4)]}


def adjacent(pl: PathLog, question: dict, rec: dict) -> dict:
    """The pool's best response with the claim as an extra candidate; the claim alone, else `unknown`, with no pool."""
    slots = question["slots"]
    found, pool = claim(pl, slots["a"], slots["b"]), pool_at("path_log", rec, pl.prior)
    if pool:
        return learned_answer(pool, [] if found is None else [found])
    return dict(UNKNOWN) if found is None else found


PATH_ANSWERERS: dict[str, PathAnswerer] = {"adjacent": adjacent}
