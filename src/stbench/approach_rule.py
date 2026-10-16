"""`closest_approach`'s rule (0.10.0, docs/families/closest_approach.md): how near the person nearest the robot would
come to it in the next few seconds, if both kept moving as they are.

One owner of the arithmetic, called by the gold (`questions_approach`) and by the table systems (`table_approach`), so
the two cannot differ by a constant. It reads a person's floor hulls (what the oracle channel sends of an evidence
sighting), the robot's own delivered velocity and the delivered clocks, and nothing else. It is neutral:
`nearest_part` and the standard library.

**The motion.** Both move at constant velocity for `HORIZON_S`. The robot's velocity is the one every delivered step
carries (`velocity`, `case.base_velocity`: the controller's odometry), so it is read from the ask's own step and asks
no frame before it. The person's is the displacement of their hull's floor centroid (`centroid`) between the latest
earlier frame within `VEL_S` before the ask that holds an evidence sighting of anyone and the ask's, over the clock
difference; it is zero when there is no such frame or the speed is under `V_STILL` (`hull_velocity`). The clock is the
delivered one, to the tenth of a second (`elapsed_s`;
`case.format_clock`): a second-rounded clock moved the gold by more than C on 6% of person rows.

**The pairing is identity-free** (`pair_velocity`). A detection's key names nothing and nothing delivered says that two
detections of two frames are one person, so a reader pairs the sightings by what the frames hold. The two frames' hulls
pair only when the pairing is unambiguous: exactly one evidence person in the ask's frame and, when there is an earlier
frame, exactly one in it. With two or more in either, the reader cannot tell a walker from two people standing, and the
gold, which would pair them by the truth's identity, would claim knowledge the sensors lack: `pair_velocity` is None,
the gold asks no such point and a table holds the person still.

**The distance** (`closest_m`) is the least, over `[0, HORIZON_S]` sampled every `STEP_S`, of the floor distance from
the robot's camera floor point to the person's hull, the camera point and the hull each carried along their velocity,
capped at `CAP_M`.

**The person last seen** (`last_seen`, 0.11.0, `approach_mem`) is `closest_approach` for the person now out of view: of
the earlier steps of the task that hold a person's hull, the latest, and its hull when it holds exactly one. It is
identity-free for the reason `pair_velocity` is: nothing delivered says that a person of one step is a person of
another, so a step that holds two hulls names no one and the reader, like the gold, has no last-seen person. The
person is then taken to stand where they were last seen (`closest_m` with a velocity of zero).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from .nearest_part import nearest_on_hull

Xy = tuple[float, float]
Hull = Sequence[Sequence[float]]

# Each constant and its source (FD-questions-0.10 section 2.1; review B1):
# at the AMR's 1.0-1.5 m/s five seconds is 5-7 m, the distance its planner re-plans over; a walking person (1.2-1.4 m/s)
# covers 6-7 m
HORIZON_S = 5.0
STEP_S = 0.1      # the sampling of the horizon: at a closing speed of 3 m/s the hull moves 0.3 m between samples
VEL_S = 4.0       # the AMR's frames come 1-5 s apart, so 4 s holds most consecutive sightings of one person
V_STILL = 0.3     # a standing person's hull centroid shifts by p90 0.045-0.088 m between frames 1-4 s apart (under 0.1
#                   m/s, review B1); a walking pace is 1.2-1.4 m/s. Slower than this is standing
CAP_M = 10.0      # `closest_person`'s cap: "answer 10 if no one would come within 10 m"


def elapsed_s(earlier: str, later: str) -> float:
    """Seconds from one delivered clock (ISO, `YYYY-MM-DDTHH:MM:SS.s`) to a later one. Naive datetimes are subtracted,
    so a change of the local clock is no part of it."""
    return (datetime.fromisoformat(later) - datetime.fromisoformat(earlier)).total_seconds()


def centroid(hull: Hull) -> Xy:
    """The floor centroid of a hull (counter-clockwise vertices, `nearest_part.floor_hull`): its area centroid; the mean
    of its vertices when it has no area (one or two vertices, or all collinear). ValueError for an empty hull."""
    if not hull:
        raise ValueError("an empty hull has no centroid")
    n = len(hull)
    area2 = cx = cy = 0.0
    for i in range(n):
        (x0, y0), (x1, y1) = hull[i], hull[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if n < 3 or abs(area2) < 1e-12:
        return (sum(p[0] for p in hull) / n, sum(p[1] for p in hull) / n)
    return (cx / (3 * area2), cy / (3 * area2))


def latest_within(ages: Iterable[tuple[int, float]]) -> int | None:
    """Of the earlier frames that hold an evidence sighting of anyone, as `(frame, seconds before the ask)`, the latest
    one at most `VEL_S` before the ask and not simultaneous with it; None when there is none."""
    held = [(age, k) for k, age in ages if 0 < age <= VEL_S]
    return min(held)[1] if held else None


def hull_velocity(earlier: Hull | None, now: Hull, dt: float) -> Xy:
    """The person's velocity, metres a second: their hull centroid's displacement from the `earlier` sighting, `dt`
    seconds before, to `now`, divided by `dt`; (0, 0) with no earlier sighting or a speed under `V_STILL`."""
    if earlier is None or dt <= 0:
        return (0.0, 0.0)
    (x0, y0), (x1, y1) = centroid(earlier), centroid(now)
    v = ((x1 - x0) / dt, (y1 - y0) / dt)
    return (0.0, 0.0) if math.hypot(*v) < V_STILL else v


def pair_velocity(now: Sequence[Hull], earlier: Sequence[Hull] | None, dt: float) -> Xy | None:
    """The velocity of the person in the ask's frame, from the hulls of every evidence person in that frame (`now`) and
    in the earlier frame used (`earlier`, `dt` seconds before; None when no frame within `VEL_S` holds one): (0, 0) with
    no earlier frame (nothing says they move), `hull_velocity` of the two lone hulls, and None when the pairing is
    ambiguous, which is when `now` or `earlier` does not hold exactly one hull. The sensors give no identity, so two
    people in a frame cannot be told from one person who moved."""
    if len(now) != 1 or (earlier is not None and len(earlier) != 1):
        return None
    return (0.0, 0.0) if earlier is None else hull_velocity(earlier[0], now[0], dt)


def last_seen(held: Mapping[int, Sequence[Hull]], k: int, steps: Iterable[int]) -> tuple[int, Hull] | None:
    """The person last seen before step `k`, as `(step, hull)`: `held` maps a step to the hulls of the evidence people
    it holds (a step with none may be absent or empty), `steps` are the steps of `k`'s task. Of those before `k` that
    hold a hull, the latest is `step`, and `hull` is its one; None when no step before `k` holds one, or when the
    latest holds more than one (the sensors give no identity, so a reader cannot say which person was seen last)."""
    seen = [s for s in steps if s < k and held.get(s)]
    if not seen:
        return None
    j = max(seen)
    return (j, held[j][0]) if len(held[j]) == 1 else None


def closest_m(cam: Xy, cam_v: Xy, hull: Hull, hull_v: Xy) -> float:
    """The least floor distance, over the horizon sampled every `STEP_S`, from the camera floor point `cam` moving at
    `cam_v` to `hull` moving at `hull_v`, at most `CAP_M`. 0 when the point is on the hull."""
    best = CAP_M
    for i in range(round(HORIZON_S / STEP_S) + 1):
        tau = i * STEP_S
        here = (cam[0] + cam_v[0] * tau, cam[1] + cam_v[1] * tau)
        moved = [[x + hull_v[0] * tau, y + hull_v[1] * tau] for x, y in hull]
        near = nearest_on_hull(moved, here)
        best = min(best, math.dist(near, here))
    return best
