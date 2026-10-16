"""The drop zones' states: the world made to say a flight's coin, and the ground a zone lies on.

A drop zone is a marking on the ground under it (`M3-maps`): no surface, so the world never draws an item onto one by
itself, and an item stands in it only where a surface (a plaza pad, a loading zone) lies under it. Before a flight a
drop enforces the coin both ways, with the world's own things and its own laws:
- **blocked**: an item whose receptacles include the kind of the surface under the zone (`plaza`: food carts and
  portable toilets; `loading zone`: vans, box trucks, skips, dumpsters) moves onto that surface, squared up, covering
  the zone materially (`material`), colliding with no other item (`placement.sample_box`); a zone some item already
  covers materially is left as it is;
- **clear**: every item within `CLEAR_M` of the zone moves out by its start's law (`world_processes_home.move_item`);
  one that finds no place that way (a 7.3 m box truck fits only the two 14.8 m loading zones, and stands on the middle
  of whichever it is in) is given one by moving the at most `MAX_BLOCKERS` items that stand in a spot it fits, the same
  way.
The zone is the piece's footprint inset by `EDGE_M`, the rule the zone families read (`M3-zones-C`); an item "stands in"
it when it is taller than `STAND_M` above the zone's ground and its footprint overlaps it. Every item of the downtown is
taller than `STAND_M`. Nothing here reads a keyframe: the drone never reacts to what it sees (design §8.5).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .catalog import Kind, for_map
from .changes import World
from .geometry import Point, Rect, corners, rects_overlap, to_world
from .mapspec import Furniture, MapSpec
from .placement import EDGE_MARGIN, ITEM_MARGIN, SQUARE_YAW, Slot, Spot, build_slots, occupied_entry, sample_box
from .world_processes_home import move_item
from .zones import EDGE_M, ground_z

STAND_M = 0.3  # an item stands in a zone when it is taller than this above the zone's ground
MIN_SHARE = 0.3  # an item covers a zone materially when it covers this share of its own footprint or of the zone
CLEAR_M = 0.05  # a clear zone has no item within this of it (a millimetre-depth sub-cell is 2 cm)
TRIES = 40  # items and spots a drop tries before it counts the coin as not enforced
SPOT_TRIES = 25  # spots one try draws on the surface
MAX_BLOCKERS = 2  # items a clear drop moves to make room for one that finds none
HOME_STEP_M = 0.1  # `has_home` looks for a place this far apart


def holds(rect: Rect, p: Point) -> bool:
    """Whether the point lies on the rectangle."""
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    dx, dy = p[0] - rect.cx, p[1] - rect.cy
    return abs(c * dx + s * dy) <= rect.sx / 2 + 1e-9 and abs(-s * dx + c * dy) <= rect.sy / 2 + 1e-9


def zone_rect(piece: Furniture) -> Rect:
    """The zone: the piece's footprint inset by `EDGE_M` on every side."""
    return Rect(piece.center[0], piece.center[1], piece.size[0] - 2 * EDGE_M, piece.size[1] - 2 * EDGE_M, piece.yaw)


def surface_under(w: World, piece: Furniture) -> Slot | None:
    """The receptacle surface the piece's centre lies on, the first in map order (the plaza pad, the loading zone), or
    None when it lies on bare ground."""
    return next((s for s in w.slots if holds(s.rect, piece.center)), None)


def settable(m: MapSpec, piece: Furniture, zones: list[Furniture]) -> bool:
    """Whether the world can always make this zone either way, from the map and its catalog alone: an item can be put
    onto it (a surface lies under it), and every kind of the catalog that fits that surface has a home clear of every
    drop zone of the map (`has_home`), so a clear zone never asks the world to put a thing nowhere. A 7.3 m box truck
    fits only the two loading zones, and cannot stand on one without covering the middle of it, where the map's street
    zones lie: they are not settable."""
    slots = build_slots(m, {f.id: f.id for f in m.furniture})
    under = next((s for s in slots if holds(s.rect, piece.center)), None)
    if under is None:
        return False
    polygons = [zone_rect(z) for z in zones]
    return all(has_home(kind, slots, polygons) for kind in for_map(m).kinds.values() if fits(kind, under))


def fits(kind: Kind, slot: Slot) -> bool:
    """Whether a box of the kind stands on the slot, turned a quarter turn to it either way."""
    return any(slot.rect.sx - EDGE_MARGIN * 2 >= dx and slot.rect.sy - EDGE_MARGIN * 2 >= dy
               for dx, dy in (kind.size[:2], kind.size[1::-1]))


def has_home(kind: Kind, slots: list[Slot], polygons: list[Rect]) -> bool:
    """Whether a box of the kind can stand on some slot, a quarter turn to it, clear (`CLEAR_M`) of every polygon."""
    for slot in slots:
        for turn, (dx, dy) in ((0.0, kind.size[:2]), (math.pi / 2, kind.size[1::-1])):
            hx, hy = slot.rect.sx / 2 - dx / 2 - EDGE_MARGIN, slot.rect.sy / 2 - dy / 2 - EDGE_MARGIN
            if hx < 0 or hy < 0:
                continue
            reach = math.hypot(slot.rect.sx, slot.rect.sy) / 2 + math.hypot(*kind.size[:2]) / 2 + CLEAR_M
            if all(math.hypot(slot.rect.cx - z.cx, slot.rect.cy - z.cy) > reach + math.hypot(z.sx, z.sy) / 2
                   for z in polygons):
                return True  # no polygon is within reach of any place on the slot
            n = max(round(2 * hx / HOME_STEP_M), 1)  # places along the slot, a step apart, flush to either end
            for lx in (-hx + 2 * hx * i / n for i in range(n + 1)):
                for ly in (-hy, 0.0, hy):
                    box = Rect(*to_world(slot.rect, lx, ly), kind.size[0], kind.size[1], slot.rect.yaw + turn)
                    if not any(rects_overlap(box, z, CLEAR_M) for z in polygons):
                        return True
    return False


def overlap_area(a: Rect, b: Rect) -> float:
    """The area the two rectangles share (Sutherland-Hodgman: each edge of `b` clips `a`; both are convex)."""
    poly = corners(a)
    edge = corners(b)
    for p, q in zip(edge, edge[1:] + edge[:1], strict=True):
        clipped = []
        for u, v in zip(poly, poly[1:] + poly[:1], strict=True):  # the edge u -> v
            du, dv = _side(p, q, u), _side(p, q, v)
            if (du >= 0) != (dv >= 0):
                f = du / (du - dv)
                clipped.append((u[0] + (v[0] - u[0]) * f, u[1] + (v[1] - u[1]) * f))
            if dv >= 0:
                clipped.append(v)
        poly = clipped
        if not poly:
            return 0.0
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True))) / 2


def _side(p: Point, q: Point, r: Point) -> float:
    """Positive when `r` lies left of the directed edge p-q (counter-clockwise polygons: the inside)."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def material(rect: Rect, zone: Rect) -> bool:
    """Whether an item's footprint covers the zone materially: `MIN_SHARE` of the item's own area, or of the zone's if
    that is less (a 17 m2 truck on the middle of a 7 m2 zone covers all of it)."""
    return overlap_area(rect, zone) >= MIN_SHARE * min(rect.sx * rect.sy, zone.sx * zone.sy)


def standing(w: World, zone: Rect, ground: float, margin: float = 0.0) -> dict[str, Rect]:
    """The footprints of the items standing in or within `margin` of the zone, by id in the world's order."""
    out = {}
    for eid, c in w.carried.items():
        rect, _, z1 = occupied_entry(c.spec, c.spot.pos, c.spot.yaw)
        if z1 - ground > STAND_M and rects_overlap(rect, zone, margin):
            out[eid] = rect
    return out


def enforce(w: World, m: MapSpec, rng: np.random.Generator, piece: Furniture, blocked: bool, t: float) -> bool:
    """Make the zone of `piece` blocked or clear from `t` on, as the module docstring says. Whether it is so now:
    False for a blocked coin when no item or spot fits in `TRIES` tries, or the zone lies on no surface, and for a clear
    coin when an item near it finds no place."""
    zone, ground = zone_rect(piece), ground_z(m, piece)
    if blocked:
        if any(material(rect, zone) for rect in standing(w, zone, ground).values()):
            return True
        slot = surface_under(w, piece)
        return slot is not None and _put_in(w, rng, slot, zone, t)
    return all([_move_out(w, m, rng, eid, zone, t) for eid in standing(w, zone, ground, CLEAR_M)])


def _yaws(rng: np.random.Generator, slot: Slot, square: bool) -> Callable[[], float]:
    """The yaw draw of an item on the slot: a quarter turn to the surface give or take `SQUARE_YAW` for a `square` kind
    (the vehicles park along the kerb), else any."""
    if not square:
        return lambda: float(rng.uniform(-math.pi, math.pi))
    return lambda: slot.rect.yaw + int(rng.integers(0, 4)) * math.pi / 2 + float(rng.uniform(-SQUARE_YAW, SQUARE_YAW))


def _put(w: World, eid: str, slot: Slot, spot: tuple[float, float, float], t: float) -> None:
    pos = (spot[0], spot[1], slot.z)
    w.ledger.place(eid, t, pos, spot[2], slot.furniture_entity)
    w.carried[eid].spot = Spot(slot, pos, spot[2])


def _put_in(w: World, rng: np.random.Generator, slot: Slot, zone: Rect, t: float) -> bool:
    """One of the world's items that belongs on the slot's kind of surface onto the slot, over the zone."""
    fits = [eid for eid, c in w.carried.items() if slot.kind in w.catalog.kinds[c.spec.kind].receptacles]
    for _ in range(TRIES if fits else 0):
        eid = fits[int(rng.integers(len(fits)))]
        c = w.carried[eid]
        found = sample_box(rng, c.spec.size, slot.rect, slot.z, w.without(eid).occupied(),
                           _yaws(rng, slot, w.catalog.kinds[c.spec.kind].square), SPOT_TRIES,
                           ok=lambda box: material(box, zone))
        if found is not None:
            _put(w, eid, slot, found, t)
            return True
    return False


def _outside(w: World, eid: str, zone: Rect) -> bool:
    c = w.carried[eid]
    return not rects_overlap(occupied_entry(c.spec, c.spot.pos, c.spot.yaw)[0], zone, CLEAR_M)


def _move_out(w: World, m: MapSpec, rng: np.random.Generator, eid: str, zone: Rect, t: float) -> bool:
    """The item to a new place by the start's law, drawn again while it still lies within `CLEAR_M` of the zone; when
    the law finds it none, to a place made for it (`_make_room`)."""
    for _ in range(TRIES):
        if not move_item(w, m, rng, eid, t):
            return _make_room(w, m, rng, eid, zone, t)
        if _outside(w, eid, zone):
            return True
    return False


def _make_room(w: World, m: MapSpec, rng: np.random.Generator, eid: str, zone: Rect, t: float) -> bool:
    """A spot on some surface clear of the zone, where at most `MAX_BLOCKERS` other items stand, each of which moves
    elsewhere by the start's law (and out of the spot and the zone's reach), then the item onto the spot."""
    c = w.carried[eid]
    kind = w.catalog.kinds[c.spec.kind]
    for _ in range(TRIES):
        slot = w.slots[int(rng.integers(len(w.slots)))]
        found = sample_box(rng, c.spec.size, slot.rect, slot.z, [], _yaws(rng, slot, kind.square), SPOT_TRIES,
                           ok=lambda box: not rects_overlap(box, zone, CLEAR_M))
        if found is None:
            continue
        spot = Rect(found[0], found[1], c.spec.size[0], c.spec.size[1], found[2])
        in_the_way = _in_the_way(w, eid, spot, slot.z, slot.z + c.spec.size[2])
        if len(in_the_way) <= MAX_BLOCKERS and all(_displace(w, m, rng, b, spot, zone, t) for b in in_the_way):
            _put(w, eid, slot, found, t)
            return True
    return False


def _in_the_way(w: World, eid: str, spot: Rect, z0: float, z1: float) -> list[str]:
    """The other items whose box, from `z0` to `z1`, is within `ITEM_MARGIN` of a box standing on `spot`."""
    out = []
    for b, other in w.carried.items():
        rect, bottom, top = occupied_entry(other.spec, other.spot.pos, other.spot.yaw)
        if b != eid and z0 < top and bottom < z1 and rects_overlap(spot, rect, ITEM_MARGIN):
            out.append(b)
    return out


def _displace(w: World, m: MapSpec, rng: np.random.Generator, eid: str, spot: Rect, zone: Rect, t: float) -> bool:
    """The item to a new place by the start's law that is neither on `spot` nor within reach of the zone."""
    for _ in range(TRIES):
        if not move_item(w, m, rng, eid, t):
            return False
        c = w.carried[eid]
        if _outside(w, eid, zone) and not rects_overlap(occupied_entry(c.spec, c.spot.pos, c.spot.yaw)[0], spot,
                                                         ITEM_MARGIN):
            return True
    return False
