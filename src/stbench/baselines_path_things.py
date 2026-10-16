"""`path_log`'s answers to `when_seen`, `order` and `distance_between` (a module of `baselines_path.PATH_MODULES`).

Each is `blind`'s pool with the briefs' rooms as the key's first levels (`blind.pool_at` over `answer_keys.levels`
("path_log", ...): for `when_seen` the description, then its kind, then the template; for the two pair families the
exact pair, the kind pair, the template), answered by `blind.learned_answer`, with what the log and the public map add
as extra candidates, and `unknown` when no level holds enough records to learn from, as `object_where`. It reads no
picture, so it never knows which things were seen, only where its camera pointed:
- `when_seen`: the clock of its last logged step whose camera faced the kind's receptacle in the public map
  (`PathLog._spot`, `places.Viewpoints` over its camera blocks). A case's clock starts on a day drawn per case
  (`case.start_clock`), so the pool's golds are learned as lags behind the clock of their own ask (`blind.Pool`), and
  this candidate is judged the same way, as the lag of its clock behind the ask's: it is scored over the pool beside
  the pool's median lag and each lag, and answered when it wins on a script left out.
- `distance_between`: the distance between the two kinds' receptacle spots (floor plane), when they differ.
- `order`: none. Its two things are of one kind and share their receptacles, so the map says nothing of which was
  seen first.
The extra candidates are scored over the pool like the rest, so over it the answer is never worse than `blind`'s.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .blind import learned_answer, pool_at
from .places import Viewpoints

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def respond(pl: PathLog, rec: dict, extra: list[dict]) -> dict:
    """The learned answer over the pool the record's key finds, with the `extra` candidates; `unknown` with no pool."""
    return learned_answer(pool_at("path_log", rec, pl.prior), extra)


def last_faced(pl: PathLog, spot: tuple[float, float, float]) -> dict | None:
    """The last logged step (by `k`) whose camera faced the place, or None. Facing a place at some step from `i` on only
    ever gets harder as `i` rises, so the last such step is found by halving."""
    steps = [pl.log.steps[k] for k in sorted(pl.log.steps)]

    def faced(i: int) -> bool:
        return Viewpoints([s["camera"] for s in steps[i:]], pl.cam).sees(spot)

    if not steps or not faced(0):
        return None
    lo, hi = 0, len(steps) - 1   # faced(lo) holds; the answer is the largest such index
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if faced(mid) else (lo, mid - 1)
    return steps[lo]


def when_seen(pl: PathLog, question: dict, rec: dict) -> dict:
    """The pool's best response, with the clock of the last step that faced the kind's receptacle as a candidate."""
    kind = question["slots"]["kind"]
    spot = pl._spot(kind)
    step = pl.cached(f"faced {kind} {len(pl.log.steps)}", lambda: last_faced(pl, spot) if spot else None)
    return respond(pl, rec, [{"type": "time", "clock": step["clock"]}] if step else [])


def order(pl: PathLog, question: dict, rec: dict) -> dict:
    """The pool's best response alone."""
    return respond(pl, rec, [])


def distance_between(pl: PathLog, question: dict, rec: dict) -> dict:
    """The pool's best response, with the distance between the two kinds' receptacle spots as a candidate."""
    a, b = (pl._spot(question["slots"][name]["kind"]) for name in ("first", "second"))
    apart = round(math.dist(a[:2], b[:2]), 3) if a and b else 0.0
    return respond(pl, rec, [{"type": "number", "value": apart}] if apart > 0 else [])


PATH_ANSWERERS: dict[str, PathAnswerer] = {"when_seen": when_seen, "order": order, "distance_between": distance_between}
