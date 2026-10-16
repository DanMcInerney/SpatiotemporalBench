"""`path_log`'s answers to `drop_zone_people` and `tangle_ahead` (a module of `baselines_path.PATH_MODULES`; the gold is
`questions_hazard`).

It reads no picture, so it never saw a person or a cable. What its own log says of the first is nothing (pedestrians
walk where they like, whatever the drone does). The second is a hazard of the floor: the layout's kinds that recur at
one spot are left out of the gold for that reason (`hazard_rule.SPOT_KINDS`), so the places other scripts' golds list
recur at random, and a pool of them is a guess. So, like every question about what a camera shows, it answers only what
other scripts' records of the template say, at the first level of `answer_keys.levels("path_log", ...)` that holds
enough of them, by `blind.learned_answer`: a best response is kept only where its held-out gain over abstaining has its
interval above 0, else `unknown`. It is never weaker than `blind`: everything `blind` knows is in `path_log`'s levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .baselines_path_people import pooled

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer

PATH_ANSWERERS: dict[str, PathAnswerer] = {"drop_zone_people": pooled, "tangle_ahead": pooled}
