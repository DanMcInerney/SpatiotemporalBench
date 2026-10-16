"""The table systems' answer to `uncovered` (a module of `table_answers.ANSWER_MODULES`; `questions_cover` is the gold).

The family asks which cells of the room the robot is leaving it saw as open floor and did not clean. A table system
holds both halves: the free-space states its mapper made of every depth image, and its log of every delivered pose and
brush status. It calls the one rule the gold calls, `grid_cover.cover`, over the same three things, each read from what
the system was sent:
- the room: the public map's region that holds the ask's delivered base (`ctx["map"]`, `MapSpec.region_of`); a base in
  no room is a question the harness never asks, answered `unknown`;
- the fine cells its robot's `brush` tool swept (`grid_cover.brush_swath` over every logged step up to the ask, so an
  earlier task's sweeps of the room count);
- the states as last seen (`Store.latest_states` up to the ask) and the layout's blocked rectangles
  (`grid_cover.blocked`).

`oracle` and `memory_table` therefore give the gold to the cell; `current_view`'s store holds the latest frame alone, so
its swath and states are the frame's and it names the cells swept before the frame as open (the memoryless answer the
memory-needed gate holds under the floor). Nothing here reads a truth module or a picture: the depth reaches this code
only as the mapper's states.
"""

from __future__ import annotations

import math

from . import grid_cover
from .answer_candidates import UNKNOWN
from .table import Answerer, Store


def uncovered(store: Store, slots: dict, ctx: dict) -> dict:
    """The cells of the room the base stands in that the table saw as open floor and did not sweep, as `cells`;
    `unknown` when the base stands in no room. The question has no slots."""
    if ctx["map"] is None:
        raise ValueError("uncovered names a room: the table needs the case layout's public map (the option 'map')")
    now = ctx["now"]
    room = ctx["map"].region_of(now["pose"]["pos"][:2])
    if room is None:
        return dict(UNKNOWN)
    poses = [(*s["pose"]["pos"][:2], math.radians(s["pose"]["yaw_deg"]), bool(s["status"][grid_cover.TOOL]))
             for s in store.steps_until(now["k"])]
    positive, _dont_care, _evidence = grid_cover.cover(
        store.latest_states(now["k"]), grid_cover.brush_swath(poses, grid_cover.brush_of(ctx["robot"])),
        ctx["map"].region(room).polygon, grid_cover.blocked(ctx["map"], ctx["robot"]))
    return {"type": "cells", "items": positive}


ANSWERERS: dict[str, Answerer] = {"uncovered": uncovered}
