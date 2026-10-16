"""The own-log screen of the heading families (0.11, FD-0.11-design §3.3): a point is a dead band where the robot's own
log already points the best way. Off by default; a family turns it on with `FamilySpec.own_log_screen`, after a free
check showed `path_log` (or its in-pool twin) over the shortcut line on its job, and the dev root is recompiled.

`path_log` answers a heading family from the robot's own delivered poses alone (`baselines_path_own.own_answer`: the
longest straight run it drove for `open_heading`, the way back to where it was eight steps ago for `way_to`). The screen
asks what that unbarred rule would score on the point's own row, under the row's own credit, and drops the point when it
would score 0 or more (`score.penalised`), that is, when the robot's own path already points within about twice the
need of the best way. This mirrors `World.told`, which never asks what the robot's log tells about. It reads the
poses and the candidate's gold, never a picture, and takes nothing from the points it keeps: the rule is the shortcut's
own, fixed before data, and a threshold is never moved to pass a gate (design §3.3).
"""

from __future__ import annotations

from .questions_select import Cand
from .questions_world import World


def told_by_log(w: World, cand: Cand, k: int, credit: str) -> bool:
    """Whether the robot's own log answers the candidate at the look point `k`: `path_log`'s own-log heading for the
    question (from the delivered poses of every step up to `k`), scored on the row the candidate would make under
    `credit`, is 0 or more. False when the log gives no heading (`unknown`)."""
    # imported here: the registry loads the family modules, which import this one, and these load the registry back
    from . import baselines_path_own, score

    own = baselines_path_own.own_answer(w.robot, cand.family, cand.slots, [w.rows[j] for j in sorted(w.rows) if j <= k],
                                        w.base(k), w.rows[k]["pose"]["yaw_deg"])
    if own["type"] == "unknown":
        return False
    row = {"qid": "screen", "gold": cand.gold, "slice": cand.slice, "credit": credit, "tolerances": cand.tolerances}
    return score.penalised(own, row) >= 0.0
