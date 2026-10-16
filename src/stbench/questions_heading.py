"""`open_heading` (0.9.0, docs/families/open_heading.md): which way, turned to in place, would let the robot's body go
farthest before it touched anything. A decision, not a measurement: the answer is the action and it is scored by what
it gives up, its regret against the best heading (`credit` `regret`, the profile written on the gold row).

**The gold.** The profile of the window (`clear_profile.profile`): for each heading at 1 degree steps, turned in place,
`grid.corridor_clear` over the latest cells of every window keyframe up to the look point, from the turned body's edge,
every cell counted, None where the corridor was not seen empty (truth rule 2: what the robot last saw of each cell, so
a heading it never looked at is not an opening). The best heading (`clear_profile.best_index`) is the one nearest ahead
among those within the corridor's 2 cm sub-cell of the longest run. A heading into space never seen runs 0 in the
scoring. The gold reads the window up to the point and no frame after it, and the question names nothing (no subject,
no time), so a point may share a look with a corridor or a memory question.

**What is asked** (`LOOKS`: look points, never sliced, and no `unknown` gold: a point where no heading was seen asks
nothing). The point is in a phase of `JobSpec.clear_phases` (where the robot drives, as `clear_path`), a box body's
turn sweeps nothing it saw (`clear_profile.turn_swept`), and the best run is one a camera can read to the job's need:
it runs to the cap, or its corridor passes the screens of `corridor_screens` (the stopper tied and visible in the frame
that last observed it, the body 1 cm narrower or wider, no height within 5 mm of the band's edge) with the stopper's
one-pixel error and tie gap at most C/2. The row's `resolution_m` is the best run's own, as `clear_path`'s: the larger
of the 2 cm sub-cell, the contact's one-pixel error and the gap.

**The scope** is by evidence: `now` when frame `k`'s own best heading (the same function over the cells of that one
keyframe, the latest frame a system holds when it is asked, which `current_view` answers) has no regret beyond the
resolution on the window's profile, else `memory`: an answerable `memory` row is one the current view cannot answer.
A look point is any frame of the window; it decides only where the question is asked, never the scope.

The cap is the robot's (`questions_words.cap_of`) and is a slot, as for `clear_path`, because a table system needs it;
the words do not state it (a heading that runs to the cap ties with every other that does).

**The own-log screen** (0.11, `heading_screen`; off unless `FamilySpec.own_log_screen`): a point where the robot's own
log already points the best way is a dead band, the fallback declared for the own-log shortcut.
"""

from __future__ import annotations

import math

from . import clear_profile, corridor_screens, grid, heading_screen
from .questions_select import Cand, Moment
from .questions_words import cap_of
from .questions_world import World
from .records import FamilySpec

RHO_MIN_M = grid.SUB_M   # the corridor is told to its 2 cm sub-cell: no run is resolved finer


def _render(slots: dict) -> str:
    return ("If you had to turn in place and drive straight away from here, which way, of those you have seen, would "
            "let your body go farthest before it touched anything? Answer the heading in degrees to the left of the "
            "way you face now, from -180 to 180 (to your right is negative).")


def resolution(w: World, k: int, states: dict, start: tuple[float, float], heading: float, gold: dict,
               cap: float) -> float | None:
    """The best run's resolution, in metres, or None when its corridor fails the screens. A run to the cap needs none
    (`RHO_MIN_M`). Else the height and side screens pass, and the stopper is tied and visible with its one-pixel error
    and tie gap at most C/2; the resolution is the larger of those and `RHO_MIN_M`. ValueError when the inputs carry no
    frame to read the stopper in (`Inputs.frame_of`)."""
    if gold["value"] >= cap - grid.EPS:
        return RHO_MIN_M
    if not (corridor_screens.height_clear(w, k, start, heading, gold)
            and corridor_screens.side_stable(w, start, heading, gold, states, 0.0)):
        return None
    stopper, half = corridor_screens.stopper(w, start, heading, gold, states), w.job.clearance_m / 2
    error, gap = stopper.error_m or 0.0, stopper.gap_m or 0.0
    if not (stopper.tied and stopper.visible and error <= half + grid.EPS and gap <= half + grid.EPS):
        return None
    return max(RHO_MIN_M, error, gap)


def open_heading(w: World, k: int) -> Moment | None:
    """The look point `k`'s moment: the best heading's `answerable` candidate, or None (a dead band) when the phase at
    `k` is not in `JobSpec.clear_phases`, the robot flies, a box body's turn sweeps something it saw, no heading was
    seen, or the best run fails the screens. `k` is any frame of the window."""
    robot = w.robot
    if robot.flies or w.rows[k]["status"]["phase"] not in w.job.clear_phases:
        return None
    cap, base, yaw = cap_of(robot), w.base(k), w.yaw(k)
    states = w.latest_cells([j for j in range(w.rec.k0, k + 1) if j in w.rows])
    if clear_profile.turn_swept(states, robot, base, yaw):
        return None
    step = clear_profile.STEP_DEG
    runs = clear_profile.profile(states, robot, base, yaw, cap, step)
    if (best := clear_profile.best_index(runs)) is None:
        return None
    heading = yaw + math.radians(best * step)
    start = grid.front_edge(robot, base, heading)
    gold, evidence = grid.corridor_clear(states, robot, start, heading, cap, 0.0)
    if gold["type"] != "number" or (rho := resolution(w, k, states, start, heading, gold, cap)) is None:
        return None
    longest = max(r for r in runs if r is not None)
    now = clear_profile.best_index(clear_profile.profile(w.latest_cells([k]), robot, base, yaw, cap, step))
    seen_now = now is not None and longest - (runs[now] or 0.0) <= rho + grid.EPS
    tolerances = {"profile_m": runs, "step_deg": step, "resolution_m": round(rho, 6), "need_m": w.job.clearance_m,
                  "threshold_m": round(robot.body.length_m + w.job.clearance_m, 6)}
    cand = Cand("open_heading", {"cap": cap}, {"type": "heading", "deg_left": float(clear_profile.deg_of(best, step))},
                "answerable", tuple(evidence), (), tolerances, decision_k=k, subjects=(), critical=True,
                answer_class="answerable", scope="now" if seen_now else "memory")
    if SPECS[0].own_log_screen and heading_screen.told_by_log(w, cand, k, SPECS[0].credit):
        return None
    return Moment(str(k), [cand], [])


SPECS = (FamilySpec("open_heading", "during", "heading", "regret", ("see", "pose"), "evidence", "map_reader", True,
                    ("cap",), _render),)
GOLDS = {"open_heading": open_heading}
LOOKS = frozenset({"open_heading"})
