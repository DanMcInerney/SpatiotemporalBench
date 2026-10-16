"""The flier's `clear_path` gold (0.8.0, docs/families/clear_path.md): how far the drone's body could fly straight
ahead or back at its height before it met a wall, as the pictures it was sent show. The family's spec and dispatch are
`questions_clear`; a table system's answer is `table_clear_fly`, over the same functions (`wall_seen`).

**The corridor.** The body is a disc 1.3 m across. Its corridor runs from its edge (the delivered base plus the body's
radius along the heading, or along the heading's opposite) against `MapSpec.wall_pieces()`, the walls the place's
fliers meet, capped at `wall_seen.CAP_M` (60 m). Nothing else of the place reaches the height it flies at, which is why
the question is asked only above the declared floor. A corridor that reaches the cap runs on into open air, which no
frame certifies empty: it is not asked.

**When it is asked.** At a look point (any frame `k`), when the delivered phase is one of the job's
`clear_phases` and the base is at least `JobSpec.fly_floor_m` up (None: no floor), so the drone is never asked while
landed on its nest or beside the street trees' crowns (the floor is the tallest crown, the body's half height and a
margin, declared by the job). The direction is `clear_move.draw` over `JobSpec.clear_directions`, which a flier declares
as ahead, back: `stable_hash(case id, k, "direction") % 2`. The hash reads nothing of the world, so the direction asked
tells nothing of what is there.

**The classes.** The wall the corridor stops at (the first it meets: the body touches it before any beyond) is **seen**
when some keyframe of the window up to `k` (never one before the case's first, which the system was not sent) shows
some stretch of it (`wall_seen`: depth over 0.5 m within 0.35 m of the wall's box), however far along its face from
where the body meets it: a building face is one flat plane, so the stretch met is inferred from the metres shown. Then
the gold is the distance, `answerable`, and the evidence is the latest such keyframe, so the scope is `now` exactly when
that keyframe is `k` itself (the latest frame saw the wall hit: a system gets one frame at a time) and `memory`
otherwise. When no such keyframe exists the gold is `unknown`, even if a wall beyond it was seen: a reader of the
pictures cannot place a wall it never saw, and the body would touch that one first. The job thins those
(`JobSpec.look_unknown_keep`).

The credit is `graded` (0.9.0): the job's clearance C is the need, the gold answered longer is the harmful side, and the
resolution `rho` is one pixel at the wall's foot, from the evidence frame's camera to the point where the corridor meets
the wall (ground from the map's fixed pieces under it, `zones.GROUND_MAX_M` tall at most), at least 0.02 m. `range_m` is
the gold plus the distance from the camera to the corridor's start (the body's edge).
"""

from __future__ import annotations

import math

from . import clear_move, rules, wall_seen
from .geometry import point_in_rect
from .mapspec import MapSpec
from .questions_select import Cand, Moment
from .questions_words import clear_slots
from .questions_world import World, scope_of
from .zones import GROUND_MAX_M

MIN_RHO_M = 0.02   # the flier's resolution is at least the ground corridor's 2 cm sub-cell


def direction_of(w: World, k: int) -> str:
    """The direction asked at look point `k`: `clear_move.draw` over `JobSpec.clear_directions` (a flier's are ahead,
    back, so `stable_hash(case id, k, "direction") % 2`)."""
    return clear_move.draw(w.rec.case, k, w.job.clear_directions)


def ground_under(m: MapSpec, xy: tuple[float, float]) -> float:
    """The height of the ground under the map point: the top of the tallest fixed piece of at most
    `zones.GROUND_MAX_M` whose footprint holds it (a plaza's top); 0 when there is none."""
    return max((f.size[2] for f in m.furniture if f.mobility == "fixed" and f.size[2] <= GROUND_MAX_M
                and point_in_rect(f.footprint(), xy[0], xy[1])), default=0.0)


def foot_rho(w: World, j: int, foot: tuple[float, float]) -> float:
    """The resolution of a gold that rests on the wall met at `foot`: one pixel at its foot (`rules.resolution_m`) from
    the camera of keyframe `j`, the evidence, at least `MIN_RHO_M`."""
    one = rules.resolution_m(w.rows[j]["camera"]["pos"], foot, ground_under(w.map, foot), w.cam.fx, w.cam.fy)
    return MIN_RHO_M if one is None else max(MIN_RHO_M, one)


def moment(w: World, k: int) -> Moment | None:
    """The look point `k`'s moment of a flier: None when the base is under `JobSpec.fly_floor_m`, when the phase is not
    in `JobSpec.clear_phases`, or when the corridor ends in open air (the cap); `answerable` when a window keyframe up
    to `k` saw the wall hit (any stretch of it), else `unknown`. `k` is any frame of the window."""
    row, job = w.rows[k], w.job
    if row["status"]["phase"] not in job.clear_phases:
        return None
    if job.fly_floor_m is not None and row["pose"]["pos"][2] < job.fly_floor_m:
        return None
    direction = direction_of(w, k)
    if (found := wall_seen.hit(w.map, w.robot, w.base(k), w.yaw(k), direction, wall_seen.CAP_M)) is None:
        return None
    far, where = found
    window = [j for j in range(w.rec.k0, k + 1) if j in w.rows]
    seen = wall_seen.saw(wall_seen.bins(w.inp, window), window, where)
    slots, need = clear_slots("fly", direction, wall_seen.CAP_M), job.clearance_m
    threshold = w.robot.body.length_m + need
    if not seen:
        tol = rules.graded_tolerances(MIN_RHO_M, need, "over", threshold=threshold)
        unknown = Cand("clear_path", slots, {"type": "unknown"}, "unknown", (), (), tol, decision_k=k,
                       subjects=(f"w:{k}",), answer_class="unknown", scope=scope_of(w, k, ()))
        return Moment(str(k), [], [unknown])
    evidence = [seen[-1]]
    edge, heading = clear_move.move(w.robot, w.base(k), w.yaw(k), direction)
    foot = (edge[0] + far * math.cos(heading), edge[1] + far * math.sin(heading))
    tol = rules.graded_tolerances(foot_rho(w, evidence[0], foot), need, "over", threshold=threshold)
    cand = Cand("clear_path", slots, {"type": "number", "value": far}, "answerable", tuple(evidence), (), tol,
                decision_k=k, subjects=(f"w:{k}",), answer_class="answerable",
                range_m=round(far + math.dist(w.camera_at(k)[0][:2], edge), 4), scope=scope_of(w, k, evidence))
    return Moment(str(k), [cand], [])
