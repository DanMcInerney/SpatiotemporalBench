# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Where the city goes beyond an outdoor map's edge, and where its street furniture stands: as plain geometry.

Plain Python (no `bpy`), so the layout can be tested without Blender: `scenery.py` draws what this returns. The map's
streets end at open region edges (a stretch of edge that no wall covers and no other region shares); past each one the
layout continues the street as static scenery, `REACH_M` metres out, so that the farthest the drone can see (a ray 2.8
degrees below the horizon from 12 m up meets the ground at 245 m) is still city and the planar depth stays well under
the 655 m a centimetre depth image holds:
- a road, and the map's own sidewalks carried on at their width and their kerb side;
- bays marked on from a parking lane that ends near the edge, from `BARE_M` out;
- two rows of buildings along the street, the first lot of each copying the map wall beside it (its style and height),
  the rest drawn from the styles the map uses.
Street furniture (lamp posts, signal poles, hydrants and, along the continued streets only, street trees) stands only in
a sidewalk's kerb strip, the first `KERB_M` from the kerb, which no receptacle surface covers: a map sidewalk's surface
starts that far in. The map's own street trees, benches and planters are the dressing's (`dressing_layout_downtown`).
The centre line and lane dashes of a continued street carry on from the map's (`street_marks.extension`) as paint.
The carriageway of a
continued street is bare (a road and painted bays, never an object) and, for `BARE_M` past the edge, unmarked as
well: the movers' free grid keeps only a mover's centre in a region, so a bus or taxi body may reach past the edge onto
it. The layout never makes an entity, and nothing it returns touches a receptacle surface or the inside of a region
except those strips. Streets and sidewalks are axis-aligned (every street of the downtown is). Scenery is seeded by
`SEED` alone, so a scene is the same whichever episode it is built for.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import NamedTuple

import delivery_keepout
import street_marks

from stbench.geometry import Point, Rect, to_world

REACH_M = 250.0
KERB_M = 1.0  # the kerb strip: the first metre of a sidewalk, from the kerb in
SEED = 31
DEPTH_M = 24.0  # how deep the buildings of a row are, unless the row opposite is closer
BAY_M = 3.0  # lots are a whole number of bays wide
LOT_M = (15.0, 36.0)
STOREY_M, GROUND_M = 3.6, 4.5  # a building is 4.5 m plus 4 to 11 storeys (18.9 to 44.1 m), as the map's own are
OPEN_MIN_M = 2.0  # an uncovered stretch of a region edge shorter than this is a gap at a corner, not a street end
BAY_PITCH_M, BAY_NEAR_M = 7.0, 10.0  # a lane that ends within this of the edge is continued, bays this far apart
BARE_M = 6.0  # the carriageway is bare this far past a street end: a mover's body reaches 4.4 m past the edge it is in
LAMP_PITCH_M, LAMP_PHASE_M = 28.0, 10.0
POST_HALF_M = {"lamp": 0.15, "signal": 0.2, "hydrant": 0.18, "tree": 0.4}
# how far a post reaches from its pole (a lamp's arm, a signal's mast arm, a crown's radius) and how tall it is
# (`scenery.py`'s builders), for the keep-out of a flier's delivery pieces: a disc about the pole stands for the arm
POST_REACH_M = {"lamp": 1.7, "signal": 4.5, "hydrant": 0.18, "tree": 1.5}
POST_HEIGHT_M = {"lamp": 7.95, "signal": 5.2, "hydrant": 0.72, "tree": 6.7}
HYDRANT_PITCH_M = 120.0
TREE_PITCH_M, TREE_APART_M = 17.0, 3.5  # a street tree every so far along a continued street, this far from any post
# A street tree (the dressing's and the scenery's alike): the pit and stem, then the crown above 3.4 m, over the walk.
# Sizes are (across the sidewalk towards the kerb, along it, up); the crown's middle is `TREE_CROWN_SHIFT_M` across from
# the trunk's (towards the building: a crown 2 m wide centred over the kerb strip would hang over the road).
TREE_TRUNK, TREE_CROWN, TREE_CROWN_SHIFT_M = (0.8, 1.0, 3.25), (2.0, 2.6, 3.2), -0.6
TOL = 0.05


class Frame(NamedTuple):
    """A street end: `o` a point of the open edge's start, `t` the unit vector along the edge, `n` the unit vector away
    from the map. `s` is the distance out of the edge and `c` the distance along it."""

    o: Point
    t: Point
    n: Point
    width: float

    def pt(self, s: float, c: float) -> Point:
        return (self.o[0] + self.n[0] * s + self.t[0] * c, self.o[1] + self.n[1] * s + self.t[1] * c)

    def local(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.o[0], y - self.o[1]
        return dx * self.n[0] + dy * self.n[1], dx * self.t[0] + dy * self.t[1]

    def rect(self, s0: float, s1: float, c0: float, c1: float) -> Rect:
        (xa, ya), (xb, yb) = self.pt(s0, c0), self.pt(s1, c1)
        return _rect(min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb))

    def span(self, r: Rect) -> tuple[float, float, float, float]:
        """(s0, s1, c0, c1) of an axis-aligned rectangle."""
        x0, x1, y0, y1 = aabb(r)
        (sa, ca), (sb, cb) = self.local(x0, y0), self.local(x1, y1)
        return min(sa, sb), max(sa, sb), min(ca, cb), max(ca, cb)


@dataclass(frozen=True)
class Walk:
    """A sidewalk slab; `front` is the yaw of the direction its kerb faces."""

    rect: Rect
    front: float


@dataclass(frozen=True)
class Building:
    """A building lot: `rect` its footprint, `face` the yaw of the direction its street face looks."""

    rect: Rect
    height: float
    style: str
    face: float


@dataclass(frozen=True)
class Post:
    kind: str  # lamp | signal | hydrant | tree
    x: float
    y: float
    yaw: float  # the direction a lamp's or signal's arm reaches (towards the road)

    @property
    def rect(self) -> Rect:
        h = POST_HALF_M[self.kind]
        return Rect(self.x, self.y, 2 * h, 2 * h, 0.0)

    @property
    def box(self) -> delivery_keepout.Box:
        """The space the post fills with its arm or crown, as a box (xmin, ymin, zmin, xmax, ymax, zmax)."""
        r = POST_REACH_M[self.kind]
        return (self.x - r, self.y - r, 0.0, self.x + r, self.y + r, POST_HEIGHT_M[self.kind])


@dataclass
class Scenery:
    arms: list[Frame] = field(default_factory=list)
    roads: list[Rect] = field(default_factory=list)
    walks: list[Walk] = field(default_factory=list)
    bays: list[Rect] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    posts: list[Post] = field(default_factory=list)  # in the map's kerb strips and in the scenery's
    lines: list[street_marks.Mark] = field(default_factory=list)  # centre lines and dashes carried on past a street end
    strips: list[Rect] = field(default_factory=list)  # the kerb strips of the map's sidewalks and of the scenery's

    def footprints(self) -> list[tuple[str, Rect]]:
        """Every footprint of the layout and what it is, for the test that none lies on a surface or in a region."""
        return ([("road", r) for r in self.roads] + [("walk", w.rect) for w in self.walks]
                + [("bay", r) for r in self.bays] + [("building", b.rect) for b in self.buildings]
                + [(p.kind, p.rect) for p in self.posts])


def _rect(x0: float, x1: float, y0: float, y1: float) -> Rect:
    return Rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0)


def aabb(r: Rect) -> tuple[float, float, float, float]:
    """x0, x1, y0, y1 of a rectangle whose yaw is a multiple of 90 degrees."""
    swap = abs(math.sin(r.yaw)) > 0.5
    hx, hy = (r.sy, r.sx) if swap else (r.sx, r.sy)
    return r.cx - hx / 2, r.cx + hx / 2, r.cy - hy / 2, r.cy + hy / 2


def axis_aligned(yaw: float) -> bool:
    return abs(math.sin(2 * yaw)) < 1e-6


# -- the streets that end at the map's edge ---------------------------------------------------------------------------
def _uncovered(length: float, covered: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out, at = [], 0.0
    for a, b in sorted(covered):
        if a - at > TOL:
            out.append((at, min(a, length)))
        at = max(at, b)
    if length - at > TOL:
        out.append((at, length))
    return [(a, b) for a, b in out if b - a >= OPEN_MIN_M]


def _run(p: Point, t: Point, x: float, y: float) -> tuple[float, float]:
    """(distance along t from p, signed distance to the left of t) of a point."""
    return (x - p[0]) * t[0] + (y - p[1]) * t[1], -(x - p[0]) * t[1] + (y - p[1]) * t[0]


def open_edges(m) -> list[Frame]:
    """The street ends of an outdoor map: the stretches of an axis-aligned region edge that no wall covers and no other
    region shares, each as a `Frame` (`n` pointing away from the region)."""
    arms: list[Frame] = []
    for r in m.regions:
        poly = [tuple(p) for p in r.polygon]
        for p, q in zip(poly, poly[1:] + poly[:1], strict=True):
            dx, dy = q[0] - p[0], q[1] - p[1]
            length = math.hypot(dx, dy)
            if length < OPEN_MIN_M or min(abs(dx), abs(dy)) > 1e-6:
                continue
            t, n = (dx / length, dy / length), (dy / length, -dx / length)  # the interior is to the left of p -> q
            covered: list[tuple[float, float]] = []
            for o in m.regions:  # the stretch another region has an edge on
                if o is r:
                    continue
                ring = [tuple(v) for v in o.polygon]
                for u, v in zip(ring, ring[1:] + ring[:1], strict=True):
                    (s0, c0), (s1, c1) = _run(p, t, *u), _run(p, t, *v)
                    if abs(c0) < TOL and abs(c1) < TOL:
                        covered.append((max(0.0, min(s0, s1)), min(length, max(s0, s1))))
            for w in m.walls:  # the stretch a wall slab lies against, just outside the edge
                (s0, c0), (s1, c1) = _run(p, t, *w.a), _run(p, t, *w.b)
                h = w.thickness / 2
                if abs(c0 - c1) < TOL and -h - TOL <= c0 <= -h + TOL:  # to the right of t, its face on the edge
                    covered.append((min(s0, s1) - h, max(s0, s1) + h))
            for a, b in _uncovered(length, covered):
                arms.append(Frame((p[0] + t[0] * a, p[1] + t[1] * a), t, n, b - a))
    return arms


# -- kerb strips and the posts in them ---------------------------------------------------------------------------------
def kerb_strip(rect: Rect, front: float) -> Rect:
    """The first `KERB_M` from the kerb of an axis-aligned sidewalk footprint whose kerb faces along `front` (a yaw)."""
    x0, x1, y0, y1 = aabb(rect)
    fx, fy = round(math.cos(front)), round(math.sin(front))
    if fx:
        return _rect(x1 - KERB_M, x1, y0, y1) if fx > 0 else _rect(x0, x0 + KERB_M, y0, y1)
    return _rect(x0, x1, y1 - KERB_M, y1) if fy > 0 else _rect(x0, x1, y0, y0 + KERB_M)


def _strip_posts(strips: list[tuple[Rect, float]], bounds, rng: random.Random, ends: bool, seed: int = SEED
                 ) -> list[Post]:
    """Lamp posts every `LAMP_PITCH_M` along each strip (staggered from one side of a street to the other), a signal
    pole at each end of a strip that is not at the map's edge (`ends`: the map's sidewalks only) and a hydrant here and
    there. A lamp near a signal pole is dropped. Along the scenery's strips (not `ends`) a street tree stands every
    `TREE_PITCH_M` or so, clear of the other posts (its own random numbers: the rest of the layout does not move)."""
    lamps: list[Post] = []
    signals: list[Post] = []
    hydrants: list[Post] = []
    for strip, front in strips:
        x0, x1, y0, y1 = aabb(strip)
        east_west = (x1 - x0) > (y1 - y0)
        lo, hi = (x0, x1) if east_west else (y0, y1)
        mid = ((y0 + y1) / 2, (x0 + x1) / 2)[0 if east_west else 1]  # the strip's centre line across its length

        def at(c: float, off: float = 0.0, east_west=east_west, mid=mid) -> Point:
            return (c, mid + off) if east_west else (mid + off, c)

        stagger = LAMP_PITCH_M / 2 if round(math.cos(front)) + round(math.sin(front)) > 0 else 0.0
        c = LAMP_PHASE_M + stagger + LAMP_PITCH_M * math.ceil((lo + 1.0 - LAMP_PHASE_M - stagger) / LAMP_PITCH_M)
        while c < hi - 1.0:
            lamps.append(Post("lamp", *at(c), front))
            c += LAMP_PITCH_M
        if ends:
            for end, inward in ((lo, 1.0), (hi, -1.0)):
                ex, ey = at(end)
                if not (bounds[0] + 1.0 < ex < bounds[2] - 1.0 and bounds[1] + 1.0 < ey < bounds[3] - 1.0):
                    continue  # at the map's own edge: the scenery's street carries on from here
                x, y = at(end + inward * 0.5)
                if all(math.dist((x, y), (s.x, s.y)) > 3.0 for s in signals):
                    signals.append(Post("signal", x, y, front))
        for _ in range(int((hi - lo) // HYDRANT_PITCH_M)):
            hydrants.append(Post("hydrant", *at(rng.uniform(lo + 4.0, hi - 4.0), 0.15), front))
    lamps = [p for p in lamps if all(math.dist((p.x, p.y), (s.x, s.y)) > 4.0 for s in signals)]
    hydrants = [p for p in hydrants if all(math.dist((p.x, p.y), (q.x, q.y)) > 3.0 for q in (*lamps, *signals))]
    trees: list[Post] = []
    if not ends:
        trng, others = random.Random(seed + 101), [*lamps, *signals, *hydrants]
        for strip, front in strips:
            x0, x1, y0, y1 = aabb(strip)
            east_west = (x1 - x0) > (y1 - y0)
            lo, hi = (x0, x1) if east_west else (y0, y1)
            mid = (y0 + y1) / 2 if east_west else (x0 + x1) / 2
            c = lo + 6.0 + trng.uniform(0.0, TREE_PITCH_M)
            while c < hi - 6.0:
                x, y = (c, mid) if east_west else (mid, c)
                if all(math.dist((x, y), (q.x, q.y)) > TREE_APART_M for q in others):
                    trees.append(Post("tree", x, y, front))
                c += TREE_PITCH_M + trng.uniform(-3.0, 3.0)
    return lamps + signals + hydrants + trees


# -- the layout --------------------------------------------------------------------------------------------------
def _lots(rng: random.Random, length: float) -> list[float]:
    """Widths (whole bays) that add up to `length`."""
    out, left = [], length
    while left > 1e-6:
        w = min(left, BAY_M * rng.randint(int(LOT_M[0] / BAY_M), int(LOT_M[1] / BAY_M)))
        if left - w < 12.0:
            w = left
        out.append(w)
        left -= w
    return out


def _beside(m, arm: Frame, c_line: float, outward: float):
    """The map wall that a row of buildings on the line `c` of an arm (`outward` +1 or -1: the side it lies on)
    continues: parallel to the street, its slab on that side of the line and its end at the edge. (style, height) or
    None."""
    for w in m.walls:
        if not w.facade:
            continue
        (sa, ca), (sb, cb) = arm.local(*w.a), arm.local(*w.b)
        h = w.thickness / 2
        if abs(ca - cb) < TOL and abs(ca - (c_line + outward * h)) < TOL and max(sa, sb) >= -h - TOL \
                and min(sa, sb) <= 0.0:
            return w.facade, w.height
    return None


def _row(arm: Frame, c_line: float, outward: float, depth: float, first, styles, rng, reach: float, start: float
         ) -> list[Building]:
    """One row of lots along an arm, its street face on the line `c_line`, `outward` (+1 or -1) away from the street."""
    fx, fy = (-outward * arm.t[0], -outward * arm.t[1])  # the street is on the other side of the face
    face = math.atan2(fy, fx)
    out: list[Building] = []
    s = start
    for i, w in enumerate(_lots(rng, reach - start)):
        style, height = first if (i == 0 and first) else (rng.choice(styles), GROUND_M + STOREY_M * rng.randint(4, 11))
        c0, c1 = sorted((c_line, c_line + outward * depth))
        out.append(Building(arm.rect(s, s + w, c0, c1), height, style, face))
        s += w
    return out


def _bays(m, arm: Frame, reach: float) -> list[Rect]:
    """Bays marked on from each parking lane that runs along the arm and ends within `BAY_NEAR_M` of its edge, at the
    lane's own pitch."""
    out: list[Rect] = []
    for f in m.furniture:
        if f.kind != "parking_lane" or not f.surfaces or not axis_aligned(f.yaw):
            continue
        s0, s1, c0, c1 = arm.span(f.footprint())
        if s1 - s0 < c1 - c0 or c0 < -TOL or c1 > arm.width + TOL or not 0.0 <= -s1 <= BAY_NEAR_M:
            continue
        frame = Rect(f.center[0], f.center[1], 0.0, 0.0, f.yaw)
        centres = [arm.local(*to_world(frame, s.rect[0], s.rect[1])) for s in f.surfaces]
        last_s, c_mid = max(centres)[0], sum(c for _, c in centres) / len(centres)
        across, along = f.surfaces[0].rect[2], f.surfaces[0].rect[3]  # the lane's local x runs across it, y along it
        k = 1
        while (centre := last_s + BAY_PITCH_M * k) + along / 2 <= reach:
            if centre - along / 2 >= BARE_M:  # none inside the map (the map's to draw) or where a mover may reach
                out.append(arm.rect(centre - along / 2, centre + along / 2, c_mid - across / 2, c_mid + across / 2))
            k += 1
    return out


def arm_lines(streets: list[street_marks.Street], arm: Frame, reach: float) -> list[street_marks.Mark]:
    """The centre line and lane dashes of the street that ends at this arm's edge, carried on `reach` metres."""
    axis = "x" if abs(arm.n[0]) > 0.5 else "y"
    edge = arm.o[0] if axis == "x" else arm.o[1]
    sign = arm.n[0] + arm.n[1]  # the arm points up or down its axis
    across = [arm.pt(0.0, 0.0)[1 if axis == "x" else 0], arm.pt(0.0, arm.width)[1 if axis == "x" else 0]]
    for s in streets:
        if s.axis != axis or not min(across) - TOL <= s.mid <= max(across) + TOL:
            continue
        if sign > 0 and abs(s.hi - edge) < TOL:
            return street_marks.extension(s, "hi", s.hi, s.hi + reach)
        if sign < 0 and abs(s.lo - edge) < TOL:
            return street_marks.extension(s, "lo", s.lo - reach, s.lo)
    return []


def layout(m, seed: int = SEED, reach: float = REACH_M) -> Scenery:
    """The scenery of an outdoor map (an empty layout for an indoor one)."""
    lay, rng = Scenery(), random.Random(seed)
    if m.outdoor is None:
        return lay
    lay.arms = open_edges(m)
    streets = street_marks.streets(street_marks.view(m))
    for arm in lay.arms:
        lay.lines += arm_lines(streets, arm, reach)
    styles = sorted({w.facade for w in m.walls if w.facade})
    start = max((w.thickness for w in m.walls), default=0.0) / 2  # past the end of the wall slabs
    map_walks = [Walk(f.footprint(), f.yaw) for f in m.furniture if f.kind == "sidewalk" and axis_aligned(f.yaw)]
    for arm in lay.arms:
        lay.roads.append(arm.rect(0.0, reach, 0.0, arm.width))
        for wk in map_walks:  # a sidewalk that runs up to this edge carries on out of it
            s0, s1, c0, c1 = arm.span(wk.rect)
            if abs(s1) < TOL and c0 >= -TOL and c1 <= arm.width + TOL:
                lay.walks.append(Walk(arm.rect(0.0, reach, c0, c1), wk.front))
        lay.bays += _bays(m, arm, reach)
    rows = []  # (arm index, face line, outward direction, the wall beside it, the room before the row opposite)
    for i, arm in enumerate(lay.arms):
        for c_line, outward in ((0.0, -1.0), (arm.width, 1.0)):
            rows.append([i, c_line, outward, _beside(m, arm, c_line, outward), DEPTH_M])
    for a in rows:  # two rows that face each other across a block meet in the middle
        for b in rows:
            fa, fb = lay.arms[a[0]], lay.arms[b[0]]
            if a is b or fa.n != fb.n or a[2] != 1.0 or b[2] != -1.0:
                continue
            gap = fa.local(*fb.pt(0.0, b[1]))[1] - a[1]
            if 0.0 < gap < 2 * DEPTH_M:
                a[4] = b[4] = min(a[4], b[4], gap / 2)
    for i, c_line, outward, first, depth in rows:
        lay.buildings += _row(lay.arms[i], c_line, outward, depth, first, styles, rng, reach, start)
    strips = [(kerb_strip(w.rect, w.front), w.front) for w in map_walks + lay.walks]
    lay.strips = [s for s, _ in strips]
    keep = delivery_keepout.keep_out([(f.kind, f.center, f.size) for f in m.furniture])
    posts = (_strip_posts(strips[:len(map_walks)], m.bounds, rng, ends=True)
             + _strip_posts(strips[len(map_walks):], m.bounds, rng, ends=False))
    lay.posts = [p for p in posts if not keep.meets(p.box)]  # none within a flier's reach of a drop zone or a nest
    return lay
