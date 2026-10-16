"""The table systems' answer to `open_heading` (a module of `table_answers.ANSWER_MODULES`; the gold:
`questions_heading`).

A table system holds what the gold reads: the free-space states its mapper made of every depth image it was sent and
the delivered pose of the step the ask is about. It calls the one rule the gold calls: `clear_profile.profile` over
the latest cells of every logged step up to the ask (`Store.latest_states`), at the delivered base and heading,
`slots["cap"]` long, and answers the best heading (`clear_profile.best_index`: the one nearest ahead within the
sub-cell of the longest run), `unknown` when no heading was seen. So `oracle` and `memory_table` give the gold's
heading, and `current_view`, whose store holds the latest frame alone, gives the best heading of that frame: the
memoryless answer the scope's `memory` rows are held to. Nothing here reads a truth module or a picture."""

from __future__ import annotations

import math

from . import clear_profile
from .answer_candidates import UNKNOWN
from .table import Answerer, Store


def open_heading(store: Store, slots: dict, ctx: dict) -> dict:
    """The best heading of the profile over the store's cells up to the ask, at the ask's delivered pose; `unknown`
    when no corridor was seen empty."""
    now, step = ctx["now"], clear_profile.STEP_DEG
    runs = clear_profile.profile(store.latest_states(now["k"]), ctx["robot"], now["pose"]["pos"][:2],
                                 math.radians(now["pose"]["yaw_deg"]), slots["cap"], step)
    if (best := clear_profile.best_index(runs)) is None:
        return dict(UNKNOWN)
    return {"type": "heading", "deg_left": float(clear_profile.deg_of(best, step))}


ANSWERERS: dict[str, Answerer] = {"open_heading": open_heading}
