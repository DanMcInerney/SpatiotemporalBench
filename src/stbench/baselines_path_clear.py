"""`path_log`'s answer to `clear_path` (a module of `baselines_path.PATH_MODULES`; the gold is `questions_clear`).

`clear_path` is a structure family: what a map would answer is `baselines_map.MapReader`'s, a reference. `path_log` is a
gated shortcut and answers from the robot's own log alone, with no map and no pool of other scripts' golds (a pose-keyed
pool would be a layout learner inside a gated system):
- **Its own path.** The fine cells its body swept between consecutive delivered base poses (`grid.swath`) count as seen
  free, and a detour is evidence (`baselines_path_space.detour`): the corridor from the body's edge for the asked move
  (`clear_move.move`: the front edge ahead, the rear edge back, the front edge on the heading turned by 90 degrees for a
  left or a right) along the moved heading, where the first cell is swept, a later unswept one is the thing the robot
  went around, and a swept one beyond it shows it went around: the answer is the distance to the unswept cell
  (`own_answer`). It is the same rule in every direction, over the robot's own swath alone.
- Else `unknown`.
That rule is kept only where it beats abstaining on other scripts. A detour is no proof of an obstacle (the helper turns
and backs away from nothing, and the rule answered it at a loss of 0.20), so it answers only where
`baselines_path_own.kept` holds for it, else `unknown`: the bar every own-log rule of `path_log` stands by.
`baselines_path_own` builds the answer, from this module's rule (`own_answer`), for the system at the ask and for the
record's `own`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import baselines_path_own, clear_move, grid
from .answer_candidates import UNKNOWN
from .baselines_path_space import detour
from .robots import Robot

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer

TEMPLATE = "clear_path"


def own_answer(model: Robot, poses: Sequence[tuple[float, float, float]], slots: dict, base: tuple[float, float],
               yaw_deg: float) -> dict:
    """The robot's own detour along the corridor of the ask at `base` heading `yaw_deg`, from the edge of the asked move
    (`slots["direction"]`, `clear_move.move`: turned for a corridor `back`, `left` or `right`) along its heading, up to
    `slots["cap"]`: the distance to the thing it went around, else `unknown`. `poses` are the delivered base poses
    (x, y, yaw in degrees) of every step it was sent up to the ask."""
    start, heading = clear_move.move(model, base, math.radians(yaw_deg), slots["direction"])
    body = model.body
    swept = grid.swath([(x, y, math.radians(yaw)) for x, y, yaw in poses], body.shape, body.length_m, body.width_m)
    far = detour(swept, start, heading, slots["cap"])
    return dict(UNKNOWN) if far is None else {"type": "number", "value": far}


OWN_RULES = {TEMPLATE: own_answer}
PATH_ANSWERERS: dict[str, PathAnswerer] = {TEMPLATE: baselines_path_own.answer}
