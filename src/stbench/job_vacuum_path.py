"""The vacuum job's path helpers: polylines to motion primitives, and where a path crosses a room's openings.

Split from `job_vacuum.py` (which owns the job) so that module stays about the job. Nothing here names a phase or a
decision kind: the job passes them in.
"""

from __future__ import annotations

import math

from .geometry import Point, distance, segments_cross, wrap_angle
from .keyframes import Hold, Mark, Move, Turn
from .mapspec import MapSpec, Region

MIN_TURN_RAD = math.radians(1.0)  # a vertex that turns the heading less than this is not turned in place
SAME_HEADING_RAD = 1e-9  # a vertex that turns less than this is no vertex at all (float noise on a straight line)


class Driver:
    """Turns polylines into primitives: at each vertex a turn in place to the next segment's heading (when it differs
    by more than `MIN_TURN_RAD`), then a straight move. The heading carries over from one polyline to the next.

    With `exact` the straight lines between the frames are the polylines themselves: a vertex that turns the heading
    by less than `MIN_TURN_RAD` (so no turn is made there) is a frame too, and a decision at the start of a polyline's
    first segment is a frame of that segment's phase even when no turn precedes it (else it would carry the phase of
    the run before). `vacuum/1` drives without; `vacuum/2`'s swaths and decisions need it."""

    def __init__(self, yaw: float, turn_dps: float, exact: bool = False):
        self.yaw, self.turn_dps, self.exact = yaw, turn_dps, exact
        self.prims: list[Move | Turn | Hold | Mark] = []

    def turn(self, at: Point, yaw: float, phase: str) -> bool:
        """Turn to `yaw` in place when it differs by more than `MIN_TURN_RAD`; whether a turn was made."""
        turned = abs(wrap_angle(yaw - self.yaw)) > MIN_TURN_RAD
        if turned:
            self.prims.append(Turn(at, self.yaw, yaw, self.turn_dps, phase))
        self.yaw = yaw
        return turned

    def begin(self, at: Point, phase: str) -> None:
        """A required frame at `at` in `phase`, the first of a run that starts there: the phase's first frame is where
        it starts, not a metre on (`keyframes.frames` frames only the end of the phase before)."""
        self.prims += [Hold(at, self.yaw, 0.0, phase), Mark()]

    def go(self, poly: list[Point], speed: float, phase: str, mark: tuple[int, float, str] | None = None) -> None:
        """Drive the polyline. `mark` is (segment index, metres from that segment's start, decision): a `Mark` there,
        after the turn into the segment when it is at the segment's start."""
        for i, (a, b) in enumerate(zip(poly, poly[1:], strict=False)):
            heading = math.atan2(b[1] - a[1], b[0] - a[0])
            bent = abs(wrap_angle(heading - self.yaw)) > SAME_HEADING_RAD
            turned = self.turn(a, heading, phase)
            if self.exact and i and bent and not turned:
                self.prims.append(Mark())
            if mark is None or mark[0] != i:
                self.prims.append(Move(a, b, heading, speed, phase))
                continue
            s = mark[1] / distance(a, b)
            at = (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
            if self.exact and s <= 0 and not i and not turned:
                self.prims.append(Hold(a, heading, 0.0, phase))
            self.prims += ([Move(a, at, heading, speed, phase)] if s > 0 else []) + [Mark(mark[2])]
            self.prims.append(Move(at, b, heading, speed, phase))


def crossing(m: MapSpec, room: Region, poly: list[Point], standoff: float, last: bool) -> tuple[int, float]:
    """Where a decision before the room's doorway sits on a path that crosses one of the room's openings (`m.door_line`
    of those whose `between` names the room): (segment index, metres from that segment's start), `standoff` before the
    crossing along its segment, or at the segment's start when that is closer. `last` takes the last crossing along the
    path (coming in), else the first (going out). Raises ValueError when the path crosses none."""
    lines = [m.door_line(o) for o in m.openings if room.id in o.between]
    hits = sorted((i, _along(a, b, p, r)) for i, (a, b) in enumerate(zip(poly, poly[1:], strict=False))
                  for p, r in lines if segments_cross(a, b, p, r))
    if not hits:
        raise ValueError(f"the path crosses none of the {room.name}'s openings")
    i, s = hits[-1] if last else hits[0]
    return i, max(0.0, s - standoff)


def _along(a: Point, b: Point, p: Point, r: Point) -> float:
    """Metres from `a` along a-b to where it crosses the line p-r (the segments are known to cross)."""
    (dx, dy), (ex, ey) = (b[0] - a[0], b[1] - a[1]), (r[0] - p[0], r[1] - p[1])
    t = ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / (dx * ey - dy * ex)
    return t * distance(a, b)
