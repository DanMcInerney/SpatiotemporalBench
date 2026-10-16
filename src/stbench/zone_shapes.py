"""The shapes of the zones that are not a map piece's footprint: an interior opening's way, and the window of the
surface the robot faces. `zones` builds `Zone`s from them. They come from the public map and the robot's own pose, never
from the world, so the gold (`questions_zone`) and a table system (`table_zone`) find the same place. This module is
neutral: it imports only the map and the geometry.

**A way** (`doorway_way`) is the strip the robot passes over to cross an interior opening, as a rectangle with its local
x along the wall. Across the opening it stops `JAMB_M` (0.1 m) short of each jamb: the jamb is wall, a door's open leaf
stands within 0.04 m of its hinge jamb, and with a 0.05 m inset the hinge-corner cell of one door read blocked in 9 of
10 empty views (FD-0.5.0-control-impl §1). Along the way it runs from `SWING_IN_M` (0.05 m) inside the wall face on a
door's swing side, which leaves the open leaf out, to `WAY_M` (0.3 m) beyond the other face; an arch has no swing side
and runs `WAY_M` beyond both faces.

**A window** (`surface_window`) is where a thing held at the robot's base would be put down: the first blocking fixed
piece the heading ray meets within `REACH_M` (1.5 m), its largest surface (the first of equal areas), and a rectangle
on it `LENGTH_M` (0.4 m) long along the surface's long side, centred on the ray's hit point (moved inward until it
fits), across the surface's whole other side. The surface is inset `INSET_M` (0.03 m) first, so that what only touches
its edge is not on it. `inset_surface` is that inset surface whole: every window of a piece lies inside it, which is
what lets `zones.zones_of` keep every cell a window can hold.
"""

from __future__ import annotations

import math

from .geometry import Point, Rect, to_local, to_world
from .mapspec import Furniture, MapSpec, Opening, Surface, surface_world_rect

JAMB_M = 0.1  # a way stops this far from each jamb
SWING_IN_M = 0.05  # on a door's swing side a way starts this far inside the wall face
WAY_M = 0.3  # a way runs this far beyond the wall face (an arch: beyond both; a door: the side it does not swing to)
REACH_M = 1.5  # how far ahead of the base the heading ray looks for a piece
LENGTH_M = 0.4  # a window's length along the surface's long side
INSET_M = 0.03  # a surface's inset


def doorway_way(m: MapSpec, o: Opening) -> Rect:
    """The way through the opening: `o.width - 2 * JAMB_M` across, from `SWING_IN_M` inside the swing-side face of a
    door (`o.swing`: "left" is the wall's left looking from its `a` end) to `WAY_M` past the other face, centred on
    the opening across; an arch runs `WAY_M` past both faces. ValueError when the jamb inset leaves no width."""
    wall = m.wall(o.wall)
    half = wall.thickness / 2
    across = o.width - 2 * JAMB_M
    if across <= 0:
        raise ValueError(f"the opening {o.id} is {o.width:g} m wide: inset {JAMB_M:g} m from each jamb it has no way")
    if o.type == "door":
        side = 1.0 if o.swing == "left" else -1.0  # the sign of the wall's normal on the swing side
        swing_edge, far_edge = half - SWING_IN_M, -(half + WAY_M)  # along the swing side's outward direction
    else:
        side, swing_edge, far_edge = 1.0, half + WAY_M, -(half + WAY_M)
    shift = side * (swing_edge + far_edge) / 2  # the way's centre from the wall's, along the wall's normal
    nx, ny = wall.normal
    return Rect(o.center[0] + nx * shift, o.center[1] + ny * shift, across, swing_edge - far_edge, o.yaw)


def blocking_pieces(m: MapSpec) -> list[Furniture]:
    """The fixed pieces that stand in the way of a robot (a rug or a mat blocks nothing, a light piece moves), in map
    order."""
    return [f for f in m.furniture if f.mobility == "fixed" and f.blocks]


def largest_surface(piece: Furniture) -> Surface | None:
    """The piece's surface of the greatest area, the first of equal areas; None for a piece with no surface."""
    return max(piece.surfaces, key=lambda s: s.rect[2] * s.rect[3], default=None)


def inset_surface(piece: Furniture, surface: Surface) -> Rect:
    """The surface on the map, pulled in `INSET_M` on every side. ValueError names a surface the inset leaves with no
    area (a map bug)."""
    r = surface_world_rect(piece.center, piece.yaw, surface)
    inner = Rect(r.cx, r.cy, r.sx - 2 * INSET_M, r.sy - 2 * INSET_M, r.yaw)
    if min(inner.sx, inner.sy) <= 0:
        raise ValueError(f"the surface {surface.id} of {piece.id} ({piece.name}) is {r.sx:g} x {r.sy:g} m: inset "
                         f"{INSET_M:g} m it has no area")
    return inner


def _entry(rect: Rect, origin: Point, yaw: float) -> float | None:
    """How far along the ray from `origin` heading `yaw` it enters the rectangle (0 when it starts inside); None when
    the ray misses it."""
    ox, oy = to_local(rect, *origin)
    dx, dy = math.cos(yaw - rect.yaw), math.sin(yaw - rect.yaw)  # the heading in the rectangle's frame
    near, far = 0.0, math.inf
    for o, d, half in ((ox, dx, rect.sx / 2), (oy, dy, rect.sy / 2)):
        if abs(d) < 1e-12:
            if abs(o) > half:
                return None
            continue
        t0, t1 = (-half - o) / d, (half - o) / d
        near, far = max(near, min(t0, t1)), min(far, max(t0, t1))
    return near if near <= far else None


def surface_window(m: MapSpec, base_xy: Point, yaw: float) -> tuple[Furniture, Surface, Rect] | None:
    """The piece, surface and window a thing held at `base_xy` heading `yaw` would be put down on: see the module
    docstring. None when no blocking piece lies within `REACH_M` along the heading, or the piece first met has no
    surface (a fridge in front of the robot is not a table behind it)."""
    best: tuple[float, Furniture] | None = None
    for f in blocking_pieces(m):
        t = _entry(f.footprint(), base_xy, yaw)
        if t is not None and t <= REACH_M and (best is None or t < best[0] - 1e-9):
            best = (t, f)
    if best is None or (surface := largest_surface(best[1])) is None:
        return None
    t, piece = best
    inner = inset_surface(piece, surface)
    hit = (base_xy[0] + t * math.cos(yaw), base_xy[1] + t * math.sin(yaw))
    hx, hy = to_local(inner, *hit)
    if inner.sx >= inner.sy:  # the long side runs along the surface's local x
        length = min(LENGTH_M, inner.sx)
        local, size = (_within(hx, inner.sx, length), 0.0), (length, inner.sy)
    else:
        length = min(LENGTH_M, inner.sy)
        local, size = (0.0, _within(hy, inner.sy, length)), (inner.sx, length)
    cx, cy = to_world(inner, *local)
    return piece, surface, Rect(cx, cy, size[0], size[1], inner.yaw)


def _within(centre: float, extent: float, length: float) -> float:
    """`centre` moved inward until a span of `length` about it lies inside [-extent / 2, extent / 2]."""
    slack = (extent - length) / 2
    return max(-slack, min(slack, centre))
