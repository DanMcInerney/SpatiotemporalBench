"""The table systems' answer to `clear_path` (a module of `table_answers.ANSWER_MODULES`; the gold: `questions_clear`).

A ground robot's answer is the gold's: `grid.corridor_clear` over the latest cells of every step the store holds up to
the ask (the window's cells as last seen, which is all the gold reads), from the body's edge for the asked move
(`clear_move.move`: the front edge ahead, the rear edge back, the front edge on the heading turned by 90 degrees left or
right) at the ask's delivered pose, `slots["cap"]` long, the cells counted beyond `clear_move.blind` (the floor the
camera cannot see ahead of the body; none for another move). A table holds what the gold reads (the mapper's states of
every depth image it was sent and the delivered pose), so `oracle` and `memory_table` reproduce the gold; a system that
keeps only the latest frame (`current_view`) holds the cells of that frame alone, so it gives the gold exactly where the
frame shows the corridor (a `now` row) and `unknown` or another number where the corridor was seen earlier (a `memory`
row). A flier's answer is `table_clear_fly.answer` (the wall record and the public map). Nothing here reads a truth
module or a picture.
"""

from __future__ import annotations

import math

from . import clear_move, grid, table_clear_fly
from .table import Answerer, Store


def clear_path(store: Store, slots: dict, ctx: dict) -> dict:
    """A flier's answer is `table_clear_fly.answer`; a ground robot's is `grid.corridor_clear` over the store's latest
    cells up to the ask, from the edge of the asked move along its heading."""
    if slots["mode"] == "fly":
        return table_clear_fly.answer(store, slots, ctx)
    pose, robot = ctx["now"]["pose"], ctx["robot"]
    start, heading = clear_move.move(robot, pose["pos"][:2], math.radians(pose["yaw_deg"]), slots["direction"])
    return grid.corridor_clear(store.latest_states(ctx["now"]["k"]), robot, start, heading, slots["cap"],
                               clear_move.blind(robot, slots["direction"]))[0]


ANSWERERS: dict[str, Answerer] = {"clear_path": clear_path}
