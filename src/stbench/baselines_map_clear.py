"""`map_reader`'s answer to `clear_path` (a module of `baselines_map.MAP_MODULES`; the gold is `questions_clear`): the
public map's raycast, answered as it is; on a held-out layout, the pose-keyed pool of other scripts' golds.
- **Ground robot.** The public map's obstacles (`baselines_path_space.obstacles`: wall pieces, fixed pieces that block
  and stand at least `FLOOR_COVERING_M` tall, door leaves) raycast along the corridor from the body's edge for the asked
  move (`clear_move.move`: the front edge ahead, the rear edge back, the front edge on the heading turned by 90 degrees
  left or right) at the pose the ask's step delivered (`raycast`, with the body's sweep and the sub-cell stop). The
  answer is the raycast as it is. A pool of other scripts' golds adds nothing to it (measured: the pose pool never beat
  the ray), and `blind.learned_answer` with the ray as a candidate abstained wherever the pool was coarse (as it nearly
  always is, since a pose recurs in few scripts): the ray is the case's own, and a held-out round scores it on other
  cases' golds, where it has no reason to hold.
- **Flier.** The same, against the walls taller than the base's height at the ask's step (a flier's body meets a wall
  only if the wall reaches it) and nothing else the map holds: no ground piece reaches a flier that stays above the
  fly floor.
On a held-out layout the reader holds no map (`Baseline.models_agent`): the pose-keyed pool of other scripts' golds
(`answer_keys.levels("map_reader", ...)`), answered through `blind.learned_answer` (kept only where its gain over
abstaining holds on a script left out), `unknown` with no pool.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

from . import clear_move
from .baselines_path_space import obstacles
from .blind import learned_answer, pool_at
from .geometry import Rect
from .grid import raycast
from .mapspec import MapSpec

if TYPE_CHECKING:
    from .baselines_map import MapReader


def walls_at(m: MapSpec, z: float) -> list[Rect]:
    """The map's wall boxes (openings cut out) of the walls taller than `z`."""
    return dataclasses.replace(m, walls=tuple(w for w in m.walls if w.height > z)).wall_pieces()


def clear_path(reader: MapReader, question: dict, rec: dict, step: dict) -> dict:
    """How far the corridor of `question` (its `direction`, ahead, back, left or right, and `cap`) from the body's edge
    at the pose of `step` runs: the map's raycast when the reader holds the map, else the pool's learned answer."""
    slots = question["slots"]
    if reader.map is None:   # a held-out layout has no map to cast a ray through: the pool alone
        return learned_answer(pool_at("map_reader", rec, reader.prior))
    start, heading = clear_move.move(reader.model, rec["base"], math.radians(rec["yaw"]), slots["direction"])
    if reader.model.flies:
        pieces = walls_at(reader.map, step["pose"]["pos"][2])
    else:
        pieces = reader.cached("obstacles", lambda: obstacles(reader.map, reader.model))
    return {"type": "number", "value": raycast(pieces, start, heading, reader.model.body, slots["cap"])}


MAP_ANSWERERS = {"clear_path": clear_path}
