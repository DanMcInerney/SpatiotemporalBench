"""The ground robot's `clear_path` gold (0.9.0, docs/families/clear_path.md): how far its body could move straight in a
direction its job declares, from what it has seen of the floor up to the look point.

**The move** (`clear_move`): the direction is drawn from the case and the point over `JobSpec.clear_directions` (ahead,
back, left, right). The corridor starts at the body's edge for that move: the front edge ahead, the rear edge back with
no turn, the front edge on the heading turned in place by plus or minus 90 degrees (left, right). Ahead keeps 0.8.0's
blind floor (`robots.blind_m`); every other move counts every cell it sweeps.

**The gold** is `grid.corridor_clear` over the latest free-space cells of every window keyframe up to the look point
(what the robot last saw of each cell: truth rule 2), not the latest frame alone, `questions_words.cap_of` long: the one
function a system's mapper calls too, over the same depth, so the oracle reproduces it. The stop is read at the
obstacle's own 2 cm sub-cell and the floor is anything within 3 mm of the climb (`grid`). A corridor not seen empty
(under 80% of the counted cells free) is `unknown`.

**The turn sweep** (`clear_profile.turn_swept`): a box body turning in place sweeps the disc of its half-diagonal; where
a seen obstacle stands in it outside the footprint, the point asks no left or right corridor.

**The scope** is `now` when the gold is a number and the corridor over the cells of frame `k` alone
(`World.latest_cells` of that one keyframe: the frame a system holds when it is asked, since it gets one frame at a
time) gives the same number within the row's resolution; else `memory` (an `unknown` gold is `memory`: the agent must
hold what it has and has not seen). So an answerable `memory` row is exactly one the current view cannot answer. A
look point is any frame of the window; it decides only where the question is asked, never the scope.

**Asked only where the robot drives** (`JobSpec.clear_phases`): in a phase in which it sits (a station, a dock, a wait)
the status already answers, and a held-out constant scores (review 2.5). A job that names none asks none.

**Asked only where a pixel-perfect reading earns the full credit** (`corridor_screens`, on the moved start and heading):
an answerable gold passes the stopper, side and height screens, or the point asks nothing. An `unknown` gold needs no
screen (it names no stop). A point offers one candidate, answerable or unknown, never both.

**The credit** is `graded` (`credit_kernel`): the job's clearance C is the need, the gold answered longer is the harmful
side, and the resolution `rho` is the larger of the grid's 2 cm sub-cell, the stopper's floor-contact pixel error and
the gap from the stop to that contact (`corridor_screens.screen`).
"""

from __future__ import annotations

import math

from . import clear_move, clear_profile, corridor_screens, grid, rules
from .questions_select import Cand, Moment
from .questions_words import cap_of, clear_slots
from .questions_world import World

EPS = 1e-9


def sent_ks(w: World, k: int) -> list[int]:
    """The keyframes of the case's window up to look point `k`, in order: everything the system has been sent."""
    return [j for j in range(w.rec.k0, k + 1) if j in w.rows]


def moment(w: World, k: int) -> Moment | None:
    """The look point `k`'s moment of a ground robot: its `answerable` gold if it passes the screens, or its `unknown`
    one where the corridor was not seen empty; None when the phase at `k` is not in `JobSpec.clear_phases`, when the
    turn sweep refuses a left or right, or when the screens refuse the gold. `k` is any frame of the window."""
    if w.rows[k]["status"]["phase"] not in w.job.clear_phases:
        return None
    robot, cap, yaw = w.robot, cap_of(w.robot), w.yaw(k)
    direction = clear_move.draw(w.rec.case, k, w.job.clear_directions)
    start, heading = clear_move.move(robot, w.base(k), yaw, direction)
    blind, states = clear_move.blind(robot, direction), w.latest_cells(sent_ks(w, k))
    if direction in clear_move.TURNED and clear_profile.turn_swept(states, robot, w.base(k), yaw):
        return None
    gold, evidence = grid.corridor_clear(states, robot, start, heading, cap, blind)
    unknown = gold["type"] == "unknown"
    rho = grid.SUB_M if unknown else corridor_screens.screen(w, k, start, heading, gold, states, blind)
    if rho is None:
        return None
    need = w.job.clearance_m
    tol = rules.graded_tolerances(rho, need, "over", threshold=robot.body.length_m + need)
    now = not unknown and frame_agrees(w, k, (start, heading, cap, blind), gold, rho)
    cam = w.camera_at(k)[0]
    cand = Cand("clear_path", clear_slots("drive", direction, cap), gold, "unknown" if unknown else "answerable",
                tuple(evidence), (), tol, decision_k=k, subjects=(f"w:{k}",), critical=True,
                answer_class="unknown" if unknown else "answerable",
                range_m=None if unknown else round(gold["value"] + math.dist(cam[:2], start), 4),
                scope="now" if now else "memory")
    return Moment(str(k), [] if unknown else [cand], [cand] if unknown else [])


def frame_agrees(w: World, k: int, corridor: tuple, gold: dict, rho: float) -> bool:
    """Whether the corridor `(start, heading, cap, blind)` over the cells of frame `k` alone, the latest frame, is a
    number within `rho` of the window's `gold`: the current view answers it, so the row is `now`."""
    other = grid.corridor_clear(w.latest_cells([k]), w.robot, *corridor)[0]
    return other["type"] == "number" and abs(other["value"] - gold["value"]) <= rho + EPS
