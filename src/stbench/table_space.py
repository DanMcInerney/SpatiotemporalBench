"""The table systems' answer to `rear_clear` (a module of `table_answers.ANSWER_MODULES`; the gold is in
`questions_space`).

The family asks how far the robot could reverse straight back, as it last saw the space behind it. A table system holds
what the gold reads: the free-space states its mapper made of every depth image it was sent, and the delivered pose of
the step the ask is about. It calls the one rule the gold calls, `grid.corridor_clear`, over the same cells, from the
rear edge (`grid.front_edge` along the heading plus pi) with no blind band: a cell was seen on the way in or it was
not, so one nobody saw counts against the 80% rule however near the rear edge it lies.

`oracle` and `memory_table` therefore give the gold; `current_view`'s store holds the latest frame alone, so a corridor
seen on the way in and not since is `unknown` to it (the memoryless answer the memory-needed gate holds under the
floor). Nothing here reads a truth module or a picture: the depth reaches this code only as the mapper's states.
"""

from __future__ import annotations

import math

from . import grid
from .table import Answerer, Store


def rear_clear(store: Store, slots: dict, ctx: dict) -> dict:
    """`grid.corridor_clear` over every logged step's latest cells up to the ask, from the rear edge of the body at the
    ask's delivered pose along its heading plus pi, `slots["cap"]` long, no blind band."""
    now = ctx["now"]
    heading = math.radians(now["pose"]["yaw_deg"]) + math.pi
    start = grid.front_edge(ctx["robot"], now["pose"]["pos"][:2], heading)
    return grid.corridor_clear(store.latest_states(now["k"]), ctx["robot"], start, heading, slots["cap"], 0.0)[0]


ANSWERERS: dict[str, Answerer] = {"rear_clear": rear_clear}
