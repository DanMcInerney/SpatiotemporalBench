"""`map_reader`'s answer to `open_heading` (a module of `baselines_map.MAP_MODULES`; the gold is `questions_heading`):
the best heading of the public map's raycast profile from the delivered pose, answered as it is; on a held-out layout,
the pose-keyed pool of other scripts' golds (docs/families/open_heading.md).

It is the family's reference, never a shortcut: a system that holds the place's layout knows which way has room
without the sensors, and the check prints its score beside the family's. For each heading `MAP_STEP_DEG` apart from
the delivered one, turned in place, the run is `grid.raycast` of the map's obstacles
(`baselines_path_space.obstacles`) from the turned body's edge, `slots["cap"]` long; the answer is the best heading
(`clear_profile.best_index`, the nearest ahead within a sub-cell of the longest). The ray is the case's own, so the pool
adds nothing to it: `blind.learned_answer` with the ray as a candidate scores it on other cases' golds, where it has no
reason to hold, and abstained wherever the pool was coarse, which is nearly always (a pose recurs in few scripts), so
the reference was `unknown` (the 0.9.0 final review, S1). It answers as `baselines_map_clear.clear_path`
does. On a held-out layout the reader holds no map: the pool's learned answer alone, `unknown` with no pool."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import clear_profile
from .baselines_path_space import obstacles
from .blind import learned_answer, pool_at
from .grid import front_edge, raycast

if TYPE_CHECKING:
    from .baselines_map import MapReader

MAP_STEP_DEG = 1   # the gold's profile is a run a degree; a coarser sweep misses narrow openings


def raycast_profile(reader: MapReader, base, yaw: float, cap: float) -> list[float]:
    """The map's run at each heading `MAP_STEP_DEG` apart from `yaw` (radians), turned in place at `base`: the
    raycast of the obstacles from the turned body's edge."""
    body = reader.model.body
    pieces = reader.cached("obstacles", lambda: obstacles(reader.map, reader.model))
    heads = [yaw + math.radians(i * MAP_STEP_DEG) for i in range(clear_profile.bins_of(MAP_STEP_DEG))]
    return [raycast(pieces, front_edge(reader.model, base, h), h, body, cap) for h in heads]


def open_heading(reader: MapReader, question: dict, rec: dict, step: dict) -> dict:
    """The best heading of the map's raycast profile at the pose of the ask (`rec`'s base and yaw), answered as it is
    when the reader holds the map; on a held-out layout the pose pool's learned answer, `unknown` with no pool."""
    if reader.map is None:   # a held-out layout has no map to cast a ray through: the pool alone
        return learned_answer(pool_at("map_reader", rec, reader.prior))
    runs = raycast_profile(reader, rec["base"], math.radians(rec["yaw"]), question["slots"]["cap"])
    return {"type": "heading", "deg_left": float(clear_profile.deg_of(clear_profile.best_index(runs), MAP_STEP_DEG))}


MAP_ANSWERERS = {"open_heading": open_heading}
