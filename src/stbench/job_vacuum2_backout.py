"""`vacuum/2`'s back-outs: a camera-equipped vacuum cleaning under a table among chair legs drives in, then backs out
along its way in rather than turning amid the legs, and the floor behind it (the reverse corridor: the robot's width,
3 m) is what lane C2's `rear_clear` asks about.

A **site** is a fixed piece of one of `SITE_KINDS` whose open underside the vacuum passes under (`robots.passes_under`)
and that stands in a room; the room is then a site room, and a `clean_rooms` task's cleaning of it begins with exactly
one back-out, before the edge pass (none in a resumed room: the stopped task has swept its corridor). The manoeuvre:
from where it is the robot goes to an **approach point** A outside the piece's footprint, turns onto the **approach
line** (through the piece's centre, along one of its two axes, either way, offset across it), drives in at least
`BACKOUT_MIN_M` and at most to the footprint's middle, to the **far point** F, where a `before_backout` decision stands
(facing in), and reverses to A with its heading kept. The approach line is the first valid one of a fixed list: the
axes, both directions, the offsets of `OFFSETS_M`, each at the approach distances of `APPROACHES_M` (how far outside
the footprint's edge A stands); valid when A is free and reachable and the drive-in keeps the body's clearance (the
free space has it already). The drive-in goes as far as a draw says between the least and the farthest the floor stays
clear, which is the footprint's middle or `STANDOFF_M` short of what it would meet before that (the robot sees it: it
has no contact sensor); the lines are tried so once, and where none has such a drive-in (a crowded floor under the
table) once more with no standoff, the drive-in ending where the floor stops being clear. The world moves
where a back-out falls, never whether it exists, so a room with no valid line is an error, not a task without its
back-out.

`SITE_KINDS` is empty: the feasibility gate (docs/jobs.md, "Back-outs") failed on `apartment_a`, so no plan carries a
`before_backout` and the kind stays declared. The mechanism is kept whole, for the day a place gives the gate a site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Point
from .mapspec import Furniture, MapSpec, Region
from .robots import Robot, passes_under
from .route import FreeSpace

BACKOUT_MIN_M = 0.3  # the least the robot drives in
DEPTH_STEP_M = 0.025  # the drive-in's reach is found in steps of this
STANDOFF_M = 0.25  # the drive-in stops this far short of the first thing ahead (the camera sees it)
OFFSETS_M = (0.0, 0.1, -0.1, 0.2, -0.2)  # across the piece's axis through its centre
APPROACHES_M = (0.25, 0.4, 0.6, 0.9, 1.2)  # how far outside the footprint's edge the approach point stands
SITE_KINDS: tuple[str, ...] = ()  # the fixed pieces the vacuum backs out of; none until a place passes the gate
CORRIDOR_M = 3.0  # the reverse corridor's length: `rear_clear`'s cap, a metre inside the depth range


@dataclass(frozen=True)
class Backout:
    approach: Point  # A: outside the footprint, where the robot turns onto the line and where the reverse ends
    far: Point  # F: where the drive-in ends; the `before_backout` decision
    yaw: float  # the heading in, along the line, radians


def site_of(m: MapSpec, room: Region, robot: Robot) -> Furniture | None:
    """The piece the vacuum backs out of in the room: the first (by id) fixed piece of `SITE_KINDS` that the robot
    passes under and whose centre lies in the room; None when the room has none (it is not a site room)."""
    found = sorted((f for f in m.furniture if f.mobility == "fixed" and f.kind in SITE_KINDS
                    and f.under_m > 0 and passes_under(robot, f.under_m) and m.region_of(f.center) == room.id),
                   key=lambda f: f.id)
    return found[0] if found else None


def lines(piece: Furniture) -> list[tuple[Point, float, float]]:
    """The approach lines of a piece, in the order they are tried: (a point on the line at the piece's centre, the
    heading in, the footprint's half extent along the line). Each of the piece's two axes in turn, each way, at each
    offset across; the first axis is the footprint's own x."""
    rect = piece.footprint()
    out = []
    for axis, half in ((rect.yaw, rect.sx / 2), (rect.yaw + math.pi / 2, rect.sy / 2)):
        for turn in (0.0, math.pi):
            yaw = axis + turn
            for off in OFFSETS_M:
                across = (-math.sin(axis) * off, math.cos(axis) * off)
                out.append(((rect.cx + across[0], rect.cy + across[1]), yaw, half))
    return out


def plan(free: FreeSpace, piece: Furniture, arrival: Point, depth: float) -> Backout | None:
    """The back-out at `piece` for a robot at `arrival` on `free`: the first valid approach line and distance, the
    drive-in `depth` (0 to 1) of the way from `BACKOUT_MIN_M` to the farthest it may go; None when no line is valid."""
    reach = free.reachable_from(arrival)
    for standoff in (STANDOFF_M, 0.0):  # stopping short of what it sees where it can, else up to it
        for centre, yaw, half in lines(piece):
            d = (math.cos(yaw), math.sin(yaw))
            for outside in APPROACHES_M:
                start = half + outside  # A is this far before the footprint's middle along the line
                a = _along(centre, d, -start)
                ix, iy = free.cell(a)
                if start < BACKOUT_MIN_M or not free.free[iy, ix] or not reach[iy, ix]:
                    continue
                if not free.line_clear(a, _along(a, d, BACKOUT_MIN_M)):
                    continue
                clear = BACKOUT_MIN_M  # the farthest in the floor stays clear, to the middle and a standoff past it
                for i in range(1, int((start + standoff - BACKOUT_MIN_M) / DEPTH_STEP_M + 1e-9) + 1):
                    if not free.line_clear(a, _along(a, d, BACKOUT_MIN_M + i * DEPTH_STEP_M)):
                        break
                    clear = BACKOUT_MIN_M + i * DEPTH_STEP_M
                most = min(start, clear - standoff)  # the robot stops short of what it would meet, or at the middle
                if most >= BACKOUT_MIN_M - 1e-9:
                    return Backout(a, _along(a, d, BACKOUT_MIN_M + depth * (most - BACKOUT_MIN_M)), yaw)
    return None


def _along(p: Point, d: Point, s: float) -> Point:
    """The point `s` metres from `p` along the unit vector `d`."""
    return (float(p[0] + d[0] * s), float(p[1] + d[1] * s))
