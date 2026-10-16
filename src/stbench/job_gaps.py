"""One gap law for every job: the unobserved gaps between a script's tasks, from the seed alone.

`vacuum/2` drew its gaps alone (M2); the daytime light rule binds every job (every case clock ends by 17:00 under the
one fixed light rig), so every job draws its gaps here, with its own id (its own stream) and the longest its task may
take (`task_max_s`). The draw is the map's, the seed's and the job's: never the world's, so a world cannot move a
task's time.
"""

from __future__ import annotations

import math

from .seeding import make_rng

GAP_S = (600.0, 10800.0)  # each gap log-uniform from 10 minutes to 3 hours, before the span budget
SPAN_S = 7 * 3600.0  # a script spans at most this, first keyframe to last (every case clock ends by 17:00)


def gaps(map_name: str, seed: int, job_id: str, n_tasks: int, task_max_s: float,
         gap_s: tuple[float, float] = GAP_S, span_s: float = SPAN_S) -> list[float]:
    """The n - 1 unobserved gaps between a script's tasks, in seconds, from `make_rng(map_name, seed, job_id, "gaps")`
    alone (the seed's, never the world's): each log-uniform in `gap_s`; when their sum passes the span budget `B =
    span_s - n * task_max_s` each is shrunk toward the least, `g = lo + (g - lo) * (B - lo (n - 1)) / (sum - lo
    (n - 1))`, so the script spans at most `span_s` with every task at its longest."""
    rng = make_rng(map_name, seed, job_id, "gaps")
    lo, hi = gap_s
    drawn = [math.exp(float(rng.uniform(math.log(lo), math.log(hi)))) for _ in range(max(n_tasks - 1, 0))]
    budget = span_s - n_tasks * task_max_s
    floor = lo * len(drawn)
    if sum(drawn) > budget:
        scale = (budget - floor) / (sum(drawn) - floor)
        drawn = [lo + (g - lo) * scale for g in drawn]
    return drawn
