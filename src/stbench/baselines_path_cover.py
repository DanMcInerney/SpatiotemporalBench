"""`path_log`'s answer to `uncovered` (a module of `baselines_path.PATH_MODULES`; `questions_cover` is the gold).

`path_log` keeps the pose and brush status of every step it was sent and holds the public map, so it knows where its
own brush went and what the room is: its claim is every cell of the room wholly inside the polygon, clear of the
layout's blocked rectangles, with no fine cell in its own `grid_cover.brush_swath` (`grid_cover.open_cells`), as cells.
It reads no depth, so it cannot tell open floor from a chair leg, clutter or an area behind a box no frame saw: the
family's gold names a cell only when the depth showed all 25 of its fine cells free, so this claim stays near the
family's own-path shortcut (plan §5.1, N11) and must score under what `memory_table` does. The answer is the best
response over its pool, the gold rows of other scripts' questions at the first of `answer_keys.levels("path_log", ...)`
that is supported, with the claim as an extra candidate; the claim alone with no pool. A base in no room (a question
the harness never asks) is `unknown`. On a held-out layout it holds no map, so there is no room to claim cells of: the
answer is the pool's best response alone, `unknown` with no pool.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import grid_cover
from .answer_candidates import UNKNOWN
from .blind import learned_answer, pool_at

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def claim_of(pl: PathLog, room: str) -> dict:
    """The cells of the room that its own brush did not touch."""
    poses = [(*s["pose"]["pos"][:2], math.radians(s["pose"]["yaw_deg"]), bool(s["status"][grid_cover.TOOL]))
             for _, s in sorted(pl.log.steps.items())]   # every step logged: the ask comes right after the decision's
    blocked = pl.cached("cover_blocked", lambda: grid_cover.blocked(pl.map, pl.model))
    swept = grid_cover.brush_swath(poses, grid_cover.brush_of(pl.model))
    open_cells = grid_cover.open_cells(pl.map.region(room).polygon, swept, blocked)
    return {"type": "cells", "items": [list(c) for c in open_cells]}


def uncovered(pl: PathLog, question: dict, rec: dict) -> dict:
    """The cells of the room the ask's base stands in that its own brush did not touch, best-responded over the pool."""
    room = None if pl.map is None else pl.map.region_of(rec["base"])
    if pl.map is not None and room is None:
        return dict(UNKNOWN)
    claim = None if room is None else claim_of(pl, room)
    pool = pool_at("path_log", rec, pl.prior)
    if pool:
        return learned_answer(pool, [] if claim is None else [claim])
    return dict(UNKNOWN) if claim is None else claim


PATH_ANSWERERS: dict[str, PathAnswerer] = {"uncovered": uncovered}
