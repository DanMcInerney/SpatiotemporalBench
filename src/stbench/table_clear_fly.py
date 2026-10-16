"""The table systems' answer to a flier's `clear_path` (the gold: `clear_fly`; dispatched by `table_clear`).

A table system holds what the gold reads: the wall bins its mapper recorded of every depth image it was sent
(`Store.wall_bins`, `wall_seen.bins_of`) and the delivered pose of the step the ask is about. It meets the corridor with
the one function the gold calls (`wall_seen.hit`, over the public map `ctx["map"]`) and answers the distance when a step
up to the ask saw the wall it stops at (some stretch of it), else `unknown`, as the gold does: a corridor into open
air, or a wall no step it was sent showed, is `unknown`. So `oracle` and `memory_table` give the gold, and
`current_view`, whose store holds the latest frame alone, answers only a wall that frame showed. A system built without
the map answers `unknown`.

It is a module of its own, not part of `clear_fly`, because the gold's module reads the compiler's world (truth side)
and a table system may not import it.
"""

from __future__ import annotations

import math

from . import wall_seen
from .answer_candidates import UNKNOWN
from .table import Store


def answer(store: Store, slots: dict, ctx: dict) -> dict:
    """The distance from the body's edge to the wall the corridor ahead (or back) meets at the ask's delivered pose,
    when some step of the store up to the ask saw that wall; else `unknown`."""
    now = ctx["now"]
    if ctx["map"] is None:
        return dict(UNKNOWN)
    found = wall_seen.hit(ctx["map"], ctx["robot"], now["pose"]["pos"][:2], math.radians(now["pose"]["yaw_deg"]),
                          slots["direction"], slots["cap"])
    if found is None or not wall_seen.saw(store.wall_bins, [k for k in store.wall_bins if k <= now["k"]], found[1]):
        return dict(UNKNOWN)
    return {"type": "number", "value": found[0]}
