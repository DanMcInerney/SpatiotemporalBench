"""`path_log`'s answer to `way_to` (a module of `baselines_path.PATH_MODULES`; the gold is `questions_way`).

`way_to` is a memory family: a thing seen at least a recent view ago, often behind the robot, so the shortcut to fear is
"turn back the way you came". `path_log` answers from the robot's own delivered poses alone, no map, no picture and no
pool of other scripts' golds: the heading from its base at the ask toward the base it was sent `LOOK_WINDOW` steps
earlier (the length of the recent view the thing is not in), `unknown` when it was sent no such step or has not moved
from it. That rule is kept only where it beats abstaining on other scripts (`baselines_path_own.kept`), else `unknown`:
the bar every own-log rule of `path_log` stands by. `baselines_path_own` builds the answer, from this module's rule
(`own_answer`), for the system at the ask and for the record's `own`; the in-pool twin keeps the rule unasked.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import baselines_path_own, grid
from .answer_candidates import UNKNOWN
from .look_windows import LOOK_WINDOW
from .robots import Robot

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer

TEMPLATE = "way_to"


def own_answer(model: Robot, poses: Sequence[tuple[float, float, float]], slots: dict, base: tuple[float, float],
               yaw_deg: float) -> dict:
    """The heading, in degrees to the left of `yaw_deg`, from `base` toward the base pose `LOOK_WINDOW` steps before the
    ask (`poses` are the delivered base poses (x, y, yaw in degrees) of every step it was sent up to the ask, the ask's
    last); `unknown` with fewer steps, or when the base is within a fine cell of it."""
    if len(poses) <= LOOK_WINDOW:
        return dict(UNKNOWN)
    x, y, _ = poses[-1 - LOOK_WINDOW]
    if math.dist((x, y), base) < grid.FINE_M:
        return dict(UNKNOWN)
    deg = (math.degrees(math.atan2(y - base[1], x - base[0])) - yaw_deg + 180.0) % 360.0 - 180.0
    return {"type": "heading", "deg_left": 180.0 if deg == -180.0 else round(deg, 3)}


OWN_RULES = {TEMPLATE: own_answer}
PATH_ANSWERERS: dict[str, PathAnswerer] = {TEMPLATE: baselines_path_own.answer}
