"""`kind_range`: `ego_now`'s declared reference (FD-0.6.0 §2.1, VISION rule 5), the typical range of each kind put on
the bearing the current view gives.

`ego_now` asks where a thing in view is from the camera, as an offset along the floor. Once the thing is found in the
frame its bearing is nearly free, so what the family adds is the range. `kind_range` is the system that has the bearing
and no range: it takes the bearing from `current_view`'s own answer (the table's unique detection in the ask step, in
that step's camera frame) and puts the thing at the typical floor range of its kind, the median over the `ego_now`
golds of the other scripts' records. It never reads the question's true range. Its `ego_now` score is a reference the
family is reported above, and not a floor that holds everywhere (release-c, the Phase A read: -0.009 on the
AMR, +0.04 on the drone, +0.30 on the helper, -0.129 on the vacuum), so distance is reported above the larger of it and
`all_unknown` per job (`eval_distance`; DECISIONS, "Before the binding read", 3). It is a reference and not a
shortcut: `gates.SHORTCUTS` does not list it, `separation` reads it as a rival of the family that declares it
(`reference: kind_range`), and it has no in-pool twin.

The median is over any gold of the kind: there is no support floor and no fallback to other kinds' ranges (a fallback
scored under `all_unknown` where kinds are thinly held, and a reference must be the strongest held-out answer of its
kind). A kind no other script has a gold of is `unknown`. Every other template is `unknown`. This module holds the
lookup and the placing and `baselines.KindRange` the system, which subclasses `CurrentView`.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable

from .answer_candidates import UNKNOWN
from .answers import Offset, parse_answer

TEMPLATE = "ego_now"   # the one template `kind_range` answers; the slot `kind` names the thing's kind


def floor_range(gold: dict) -> float | None:
    """The floor distance in metres, `hypot(ahead, left)`, of an offset gold (a record's `gold["gold"]`); None for any
    other answer (an `unknown`, a position)."""
    match parse_answer(gold):
        case Offset(ahead_m, left_m):
            return math.hypot(ahead_m, left_m)
    return None


class Ranges:
    """The typical floor range of each kind, from other scripts' `ego_now` records (`opts["prior"]`): the median over
    any gold of the kind, none for a kind without one."""

    def __init__(self, prior: Iterable[dict]):
        by_kind: dict[str | None, list[float]] = defaultdict(list)
        for rec in prior:
            if rec["template"] == TEMPLATE and (r := floor_range(rec["gold"]["gold"])) is not None:
                by_kind[rec["slots"].get("kind")].append(r)
        self.kinds = {kind: statistics.median(found) for kind, found in by_kind.items()}

    def of(self, kind: str | None) -> float | None:
        """The typical range of `kind`: the median of its golds, None when there is none."""
        return self.kinds.get(kind)


def placed(seen: dict, r: float | None) -> dict:
    """The offset `seen` (what `current_view` answered: an offset, or `unknown`) moved along its own bearing to the
    floor range `r`. `unknown` when `seen` is not an offset or there is no `r`. A thing exactly at the camera's ground
    point has no bearing and is put straight ahead."""
    match parse_answer(seen):
        case Offset(ahead_m, left_m) if r is not None:
            d = math.hypot(ahead_m, left_m)
            ahead, left = (r, 0.0) if d == 0 else (ahead_m * r / d, left_m * r / d)
            return {"type": "offset", "ahead_m": round(ahead, 4), "left_m": round(left, 4)}
    return dict(UNKNOWN)
