"""`path_log`'s answer to `open_heading` (a module of `baselines_path.PATH_MODULES`; the gold is `questions_heading`).

`open_heading` is a structure family: what a map would answer is `baselines_map.MapReader`'s, a reference. `path_log` is
a gated shortcut and answers from the robot's own log alone, with no map and no pool of other scripts' golds (a
pose-keyed pool would be a layout learner inside a gated system):
- **Its own path.** The fine cells its body swept between consecutive delivered base poses (`grid.swath`) count as
  seen free. For each heading, `LOG_STEP_DEG` apart from the delivered one, the run is how far the swept cells go
  unbroken from the turned body's edge along it (the first fine cell must be swept; capped at `slots["cap"]`). The
  answer is the heading of the longest such run, the one nearest ahead of equals (`clear_profile.best_index`): where
  the robot has already driven in a straight line is a place it fitted (`own_answer`).
- Else `unknown`, when no heading's first cell was swept.
That rule is kept only where it beats abstaining on other scripts. The heading it has driven is no proof of the best one
(the credit is the regret against the best heading, and the longest run it drove may be short beside an opening it never
drove), so it answers only where `baselines_path_own.kept` holds for it, else `unknown`: the bar every own-log rule of
`path_log` stands by. `baselines_path_own` builds the answer, from this module's rule (`own_answer`), for the system at
the ask and for the record's `own`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import baselines_path_own, clear_profile, grid
from .answer_candidates import UNKNOWN
from .robots import Robot

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer

TEMPLATE = "open_heading"
LOG_STEP_DEG = 5   # the headings tried: a straight run over the swept cells is told to a fine cell, not a degree


def own_run(swept: set[tuple[int, int]], start: tuple[float, float], heading: float, cap: float) -> float:
    """How far along `heading` from `start` the swept fine cells run unbroken, sampled at each cell's middle: 0 when the
    first is not swept, at most `cap`, to the millimetre."""
    s, run = grid.FINE_M / 2, 0.0
    while s < cap:
        if grid.fine_cell(start[0] + s * math.cos(heading), start[1] + s * math.sin(heading)) not in swept:
            break
        run, s = s + grid.FINE_M / 2, s + grid.FINE_M
    return round(min(run, cap), 3)


def own_answer(model: Robot, poses: Sequence[tuple[float, float, float]], slots: dict, base: tuple[float, float],
               yaw_deg: float) -> dict:
    """The heading of the longest straight run over the robot's own swept cells from `base` heading `yaw_deg`, capped at
    `slots["cap"]`, the nearest ahead of equals; `unknown` when it has no run. `poses` are the delivered base poses (x,
    y, yaw in degrees) of every step it was sent up to the ask."""
    yaw, body = math.radians(yaw_deg), model.body
    swept = grid.swath([(x, y, math.radians(deg)) for x, y, deg in poses], body.shape, body.length_m, body.width_m)
    n = clear_profile.bins_of(LOG_STEP_DEG)
    runs = [own_run(swept, grid.front_edge(model, base, h), h, slots["cap"])
            for h in (yaw + math.radians(i * LOG_STEP_DEG) for i in range(n))]
    best = clear_profile.best_index(runs)
    if best is None or runs[best] <= 0.0:
        return dict(UNKNOWN)
    return {"type": "heading", "deg_left": float(clear_profile.deg_of(best, LOG_STEP_DEG))}


OWN_RULES = {TEMPLATE: own_answer}
PATH_ANSWERERS: dict[str, PathAnswerer] = {TEMPLATE: baselines_path_own.answer}
