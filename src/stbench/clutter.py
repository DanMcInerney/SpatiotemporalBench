"""Clutter, the world content no question names: the toasters, books and cables
of a lived-in apartment, and the shoes, bag, toy and box that real floors hold instead of a mat.

`generate_script` calls `place` after `build_world` and before it plans a path, so the planner meets what lies on the
floor from the start. Clutter is world content with mobility `clutter` (`timeline.MOBILITY`): an entity of the timeline,
the visibility log and the gold, that obstructs a ground robot like a carried item on the floor, collides with carried
things, is never asked about and is recorded in `World.clutter`, so a later move cannot land inside it. It is static
within a task; the world's processes between tasks (`world_processes_home`) move it by `move_floor` and drop a carried
thing by `floor_spot`, on the same floor rules.

`place` takes two streams from the rng it is given (`rng.spawn(2)`): one draws what the world holds, the other which of
the tracked things of `Catalog.floor_kinds` are left on the floor, so neither shifts the other. It does three things on
one spot finder (`placement.sample_box`):
- about half of each floor kind's tracked things move off their receptacles to open floor within 0.8 m of a piece of
  one of its anchor kinds (else their room's open floor), with a seeded yaw;
- floor clutter goes on the open floor `dressing_boxes.open_floor` keeps (at least 0.12 m from walls and blocking
  footprints, so no wall dressing can meet it), or under a piece with an open underside, off its leg squares and low
  enough to clear it;
- surface clutter goes on the back strip of a surface (the rear quarter of a metre, or half its depth, or as deep as the
  kind's own depth needs, at most four fifths of the surface; four fifths of a surface half a metre deep or more),
  clear of everything there. A kind says where it stands in
  `ClutterKind.places`: a surface kind of the map, the top of a piece that has none (`on:stove`: the pots on its back
  burners) or the front of a piece (`front:fridge`: sheets of paper held to the door, standing on its face).
Nothing goes in a doorway, a door's swing, the band `DOOR_BAND_M` from each wall face across a door's width, the dock's
apron (`mapspec.DOCK_APRON_*`), or on a wall, a blocking footprint, light furniture at its map place, a carried thing or
other clutter (a later move also keeps off light furniture where it stands then). An entity's size is its model's
measured box (`asset_manifest`), and its colour its model's own or, for a tinted model, one of its kind's.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np

from . import asset_manifest
from .catalog import FLOOR, ClutterKind, place_of
from .catalog_clutter import FRONT, ON, floor_beside
from .changes import WORLD_CLEARANCE_M, World
from .clutter_reach import FloorGuard
from .dressing_boxes import WALL_PROUD_MAX_M, open_floor
from .geometry import Rect, corners, rect_distance, rect_inside_rect, rects_overlap, to_world, under_obstacles
from .mapspec import DOCK_APRON_FRONT_M, DOCK_APRON_SIDE_M, MapSpec
from .placement import Occupied, Slot, Spot, occupied_entry, sample_box
from .timeline import Entity

DOOR_BAND_M = 0.5  # nothing stands this far from a door's wall faces, across the door's width
FLOOR_COUNT = (20, 45)  # floor clutter a script (design §3.1: about 0.5-1.2 a m² of the ~39 m² of free floor)
SURFACE_COUNT = (65, 150)  # surface clutter a script: what a real home's counters, shelves and desks hold
SURFACE_DRAWN = (100, 150)  # the range `place` draws it from, inside SURFACE_COUNT
ENTITY_COUNT = (160, 250)  # entities in the world, `build_world`'s included (3-4.5 a m² of the 55 m²)
STRIP_M = 0.25  # the back strip of a surface: this deep, or half the surface's depth, or as deep as a kind needs
STRIP_SHARE = 0.8  # but never more than this share of the surface's depth: the front stays clear
BIG_SURFACE_M = 0.5  # a surface this deep or more (a counter, a table, a desk, the sofa, the bed) is open to most of it
FRONT_SEAT_M = 0.009  # a thick thing hung on a face has its centre this far out at most (`validate`: on the piece)
FRONT_MARGIN_M = 0.04  # a sheet on a door keeps this far from the door's edges and from the next sheet
FRONT_TRIES = 60  # draws a face gets for one sheet: they are cheap and a crowded door needs them
FRONT_LOW_M = 0.8  # the lowest a sheet is held on a door, and the highest is this far under its top
DROP_SHARE = 0.5  # of the tracked things of a floor kind that are left on the floor
DROP_REACH_M = 0.8  # how far from its anchor piece a dropped thing may lie
FLAT_M = 0.02  # a thing this low lies under a robot's climb: a mat, a placemat, a sheet of paper
FLAT_INSET_M = 0.03  # a flat thing keeps this far from a wall or a blocking piece: dressing is never as low as a mat
BESIDE_M = 0.35  # how far from the piece a floor kind that stands beside one may lie (a mat at the toilet)
UNDER_WEIGHT = 2.0  # a piece's underside draws this many times its area among a floor kind's candidate places
YAW_SPREAD = 0.3  # radians a thing on a surface strays from its surface's own turn (less, the tighter the fit)
LIGHT_MARGIN_M = 0.05  # clear of light furniture at its map place, which a gap's process may later move
UNDER_WALL_M = 0.03  # a thing under a piece stays this far from a wall: a baseboard (at most 0.015 m proud) misses it
GAP_M = 0.004  # clutter sits this close to its neighbours and to the edge of its place: a real home's things touch
UNDER_SLACK_M = 0.005  # a thing under a piece stands at least this far under its open underside
DROP_TRIES, FLOOR_TRIES, SURFACE_TRIES = 6, 8, 24  # candidate anchors, then draws a place gets before the next one


@dataclass(frozen=True)
class _Model:
    id: str
    size: tuple[float, float, float]
    colour: str | None  # an untinted model's own; None where the entity's colour tints it


@cache
def _models(ids: tuple[str, ...]) -> tuple[_Model, ...]:
    """The measured boxes and colours of a clutter kind's models, from the packaged manifest."""
    m = asset_manifest.packaged()
    out = []
    for i in ids:
        a = m.asset(i)
        if a.native_bbox_m is None:
            raise asset_manifest.AssetError(f"{i}: its box is not recorded in the manifest, so no entity can be sized")
        out.append(_Model(i, tuple(a.native_bbox_m), a.colour))
    return tuple(out)


class _Site:
    """What `place` may and may not use on one map: its open floor (as rectangles), the pieces with an open underside,
    the keep-outs, the light furniture's places (at their map places, which a gap's process may put a piece back to,
    and where they stand) and the floor guard, the doors' leaves in their state at `t`."""

    def __init__(self, world: World, m: MapSpec, t: float = 0.0):
        self.m = m
        self.rooms = {r.id: r.name for r in m.regions}
        self.floor = [Rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0) for x0, y0, x1, y1 in open_floor(m)]
        self.flat = [Rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0)
                     for x0, y0, x1, y1 in open_floor(m, FLAT_INSET_M)]  # where a mat may lie: it is under any dressing
        self.under = [f for f in m.furniture if f.mobility == "fixed" and f.blocks and f.under_m > 0]
        self.walls = m.wall_pieces()
        self.solids = [*self.walls, *(f.footprint() for f in m.furniture if f.mobility == "fixed" and f.blocks)]
        self.keepout = self._keepouts()
        self.light = [f.footprint() for f in m.furniture if f.mobility == "light"] + list(world.light_rects.values())
        self.guard = FloorGuard(world, t)
        tokens = {place_of(p)[0] for k in world.catalog.clutter for p in k.places}
        self.tops = [slot for t in sorted(tokens) if t.startswith(ON) for slot in self._pieces(world, t[len(ON):], t)]
        self.fronts = {t: self._pieces(world, t[len(FRONT):], t) for t in sorted(tokens) if t.startswith(FRONT)}
        self.front_used: dict[str, list[tuple[float, float, float, float]]] = {}

    def _pieces(self, world: World, piece_kind: str, token: str) -> list[Slot]:
        """One slot per fixed piece of the kind: its footprint and the height of its top, for a clutter place that is a
        piece (`on:stove`, `front:fridge`) and no surface of the map. The slot's surface id is the piece's own."""
        pieces = [f for f in self.m.furniture if f.mobility == "fixed" and f.kind == piece_kind]
        ids = [e.id for e in world.entities if e.mobility == "fixed" and e.kind == piece_kind]
        return [Slot(eid, f.id, token, f.footprint(), f.size[2], self.m.region_of(f.center))
                for f, eid in zip(pieces, ids, strict=True)]

    def _keepouts(self) -> list[Rect]:
        """Every doorway's band, every door's swing square (a superset of its quarter disc) and the dock's apron."""
        m, out = self.m, []
        for o in m.openings:
            out.append(Rect(*o.center, o.width, m.wall(o.wall).thickness + 2 * DOOR_BAND_M, o.yaw))
            if o.type == "door":
                _, closed = m.door_leaf(o, False)
                _, tip = m.door_leaf(o, True)
                out.append(Rect((closed[0] + tip[0]) / 2, (closed[1] + tip[1]) / 2, o.width, o.width, o.yaw))
        for f in (f for f in m.furniture if f.kind == "dock"):
            fp = f.footprint()
            out.append(Rect(*to_world(fp, DOCK_APRON_FRONT_M / 2, 0.0), fp.sx + DOCK_APRON_FRONT_M,
                            fp.sy + 2 * DOCK_APRON_SIDE_M, fp.yaw))
        return out

    def room(self, x: float, y: float) -> str | None:
        region = self.m.region_of((x, y))
        return None if region is None else self.rooms[region]

    @staticmethod
    def apart(box: Rect, gap: float, solids: list[Rect]) -> bool:
        """At least `gap` from every one of the rectangles (their true distance: `open_floor` rounds the corners of a
        footprint, so a cell can lie a little nearer than its inset)."""
        reach = math.hypot(box.sx, box.sy) / 2 + gap
        return all(math.hypot(box.cx - s.cx, box.cy - s.cy) > reach + math.hypot(s.sx, s.sy) / 2
                   or rect_distance(box, s) >= gap for s in solids)

    def clear(self, box: Rect, height: float) -> bool:
        """Off every keep-out and every light piece, and no cut through a planned robot's free floor."""
        return (not any(rects_overlap(box, k) for k in self.keepout)
                and not any(rects_overlap(box, light, LIGHT_MARGIN_M) for light in self.light)
                and self.guard.allows(box, height))


def _nearby(occ: Occupied, region: Rect, z0: float, z1: float) -> Occupied:
    """What of `occ` can touch a box inside `region` standing from `z0` to `z1`: the rest never needs testing."""
    reach = math.hypot(region.sx, region.sy) / 2 + 0.05
    return [o for o in occ if o[1] < z1 and z0 < o[2]
            and math.hypot(o[0].cx - region.cx, o[0].cy - region.cy) <= reach + math.hypot(o[0].sx, o[0].sy) / 2]


def _weighted(rng: np.random.Generator, items: list, weights: list[float]):
    """The items in a drawn order without replacement, each draw weighted."""
    pool, w = list(items), list(weights)
    while pool:
        i = int(rng.choice(len(pool), p=np.array(w) / sum(w)))
        w.pop(i)
        yield pool.pop(i)


def _yaw_around(rng: np.random.Generator, centre: float, spread: float, tries: int) -> Callable[[], float]:
    """A yaw draw that strays from `centre` by less each time it is asked: the first tries are loose, the last nearly
    square, so a tight place is still filled."""
    asked = [0]

    def draw() -> float:
        scale = max(0.0, 1.0 - asked[0] / tries)
        asked[0] += 1
        return centre + float(rng.uniform(-spread, spread)) * scale

    return draw


# -- floor things -------------------------------------------------------------------------------------------------
def _in_cell(site: _Site, rng: np.random.Generator, size, occ: Occupied, cell: Rect,
             ok: Callable[[Rect], bool] | None = None, inset: float | None = WALL_PROUD_MAX_M):
    """A spot in one open floor cell, with a free yaw, at least `inset` from every wall and blocking footprint (none for
    a cell that is a piece's own footprint), clear of the keep-outs, light furniture and the floor's connection, and of
    what `ok` refuses."""
    def accepted(box: Rect) -> bool:
        return ((inset is None or site.apart(box, inset, site.solids)) and (ok is None or ok(box))
                and site.clear(box, size[2]))

    near = _nearby(occ, cell, 0.0, size[2])
    return sample_box(rng, size, cell, 0.0, near, lambda: float(rng.uniform(-math.pi, math.pi)), FLOOR_TRIES, accepted,
                      GAP_M, GAP_M)


def _under_piece(site: _Site, rng: np.random.Generator, size, occ: Occupied, piece):
    """A spot under a piece with an open underside: inside its footprint and off its leg squares."""
    legs = under_obstacles(piece.footprint(), piece.under_m)
    return _in_cell(site, rng, size, occ, piece.footprint(), lambda box: site.apart(box, UNDER_WALL_M, site.walls)
                    and not any(rects_overlap(box, g) for g in legs), None)


def _on_floor(site: _Site, rng: np.random.Generator, size, occ: Occupied, wanted: Callable[[Rect], bool],
              ok: Callable[[Rect], bool] | None = None, flat: bool = False):
    """A spot on the open floor cells `wanted` allows, drawn by area; None if none holds the thing. A `flat` thing (a
    mat, lower than a robot climbs) may lie nearer a wall or a piece, `FLAT_INSET_M` from it."""
    cells = [r for r in (site.flat if flat else site.floor) if wanted(r)]
    for cell in _weighted(rng, cells, [c.sx * c.sy for c in cells]) if cells else ():
        if (found := _in_cell(site, rng, size, occ, cell, ok, FLAT_INSET_M if flat else WALL_PROUD_MAX_M)) is not None:
            return found
    return None


def _in_room(site: _Site, room: str | None, also: Callable[[Rect], bool]) -> Callable[[Rect], bool]:
    """Whether a cell lies in the room and passes `also`."""
    return lambda r: site.room(r.cx, r.cy) == room and also(r)


def floor_slot(m: MapSpec, x: float, y: float) -> Slot:
    """Where a dropped thing is: the open floor of its room, a surface with no piece (`furniture_entity` empty)."""
    return Slot("", "floor", FLOOR, Rect(x, y, 0.0, 0.0, 0.0), 0.0, m.region_of((x, y)))


def _within(piece: Rect, reach: float = DROP_REACH_M) -> Callable[[Rect], bool]:
    """Whether a rectangle lies within `reach` (default `DROP_REACH_M`) of the piece."""
    return lambda r: rect_distance(r, piece) <= reach


def _drop_near(world: World, site: _Site, rng: np.random.Generator, size, anchors: tuple[str, ...], occ: Occupied,
               also: Callable[[Rect], bool] | None = None):
    """A spot on the open floor of an anchor piece's room within `DROP_REACH_M` of it (and where `also` allows), over a
    few drawn pieces."""
    pieces = [f for f in world.m.furniture if f.mobility == "fixed" and f.kind in anchors]
    for _ in range(DROP_TRIES if pieces else 0):
        fp = pieces[int(rng.integers(len(pieces)))].footprint()
        near, room = _within(fp), site.room(fp.cx, fp.cy)
        ok = near if also is None else (lambda box, near=near: near(box) and also(box))
        if (found := _on_floor(site, rng, size, occ, _in_room(site, room, near), ok)) is not None:
            return found
    return None


def _drop_floor_things(world: World, site: _Site, rng: np.random.Generator) -> None:
    """Each carried thing of a floor kind is, independently, left on the floor with probability `DROP_SHARE`: within
    `DROP_REACH_M` of a piece of one of its anchor kinds, else on its own room's open floor. It keeps its entity and is
    placed again from t 0 with a seeded yaw and no support; one that finds no place stays where it was."""
    anchors = dict(world.catalog.floor_kinds)
    for c in list(world.carried.values()):
        if c.spec.kind not in anchors or rng.random() >= DROP_SHARE:
            continue
        size = c.spec.size
        others = [occupied_entry(x.spec, x.spot.pos, x.spot.yaw) for x in world.carried.values() if x is not c]
        occ = others + list(world.clutter.values())
        found = _drop_near(world, site, rng, size, anchors[c.spec.kind], occ)
        if found is None and c.spot.slot.region is not None:
            room = site.rooms[c.spot.slot.region]
            found = _on_floor(site, rng, size, occ, _in_room(site, room, lambda r: True))
        if found is not None:
            x, y, yaw = found
            c.spot = Spot(floor_slot(site.m, x, y), (x, y, 0.0), yaw)
            world.ledger.place(c.entity, 0.0, (x, y, 0.0), yaw, None)
            site.guard.block(Rect(x, y, size[0], size[1], yaw), size[2])


# -- clutter ------------------------------------------------------------------------------------------------------
def _matches(token: str, slot: Slot, site: _Site) -> bool:
    where, room = place_of(token)
    return slot.kind == where and (room is None or (slot.region is not None and site.rooms[slot.region] == room))


def _strip(slot: Slot, need: float = 0.0) -> Rect:
    """The back of a surface in its piece's frame (x is the depth, the front toward +x): `STRIP_SHARE` of the depth of
    a surface of `BIG_SURFACE_M` or more, else the rear `STRIP_M` or half its depth, whichever is less, or as deep as
    `need` (a kind's own depth) asks, at most `STRIP_SHARE` of the depth."""
    sx = slot.rect.sx
    depth = STRIP_SHARE * sx if sx >= BIG_SURFACE_M else min(STRIP_SHARE * sx, max(min(STRIP_M, sx / 2), need))
    cx, cy = to_world(slot.rect, -slot.rect.sx / 2 + depth / 2, 0.0)
    return Rect(cx, cy, depth, slot.rect.sy, slot.rect.yaw)


def _on_surface(world: World, site: _Site, rng: np.random.Generator, kind: ClutterKind, size, occ: Occupied):
    slots = [s for s in (*world.slots, *site.tops) if any(_matches(p, s, site) for p in kind.places)]
    for slot in _weighted(rng, slots, [s.area for s in slots]) if slots else ():
        yaw, strip = _yaw_around(rng, slot.rect.yaw, YAW_SPREAD, SURFACE_TRIES), _strip(slot, size[0] + 0.02)
        near = _nearby(occ, strip, slot.z, slot.z + size[2])
        found = sample_box(rng, size, strip, slot.z, near, yaw, SURFACE_TRIES, None, GAP_M, GAP_M)
        if found is not None:
            return found[0], found[1], slot.z, found[2], slot.furniture_entity
    return None


def _on_front(site: _Site, rng: np.random.Generator, kind: ClutterKind, size):
    """A spot on the front face of a piece the kind's places name (`front:fridge`): a sheet stands on its edge facing
    out (its depth, x, is its thickness), its back 0.5 mm into the face so that it lies on it (a thick thing, a coat,
    sits into the face by what its half depth has over `FRONT_SEAT_M`: `validate` keeps its centre on the piece),
    between `FRONT_LOW_M` and a hand under the top, clear of the face's edges and of the sheets already there."""
    pieces = [s for t in kind.places for s in site.fronts.get(place_of(t)[0], ())]
    for slot in _weighted(rng, pieces, [s.rect.sy * s.z for s in pieces]) if pieces else ():
        used = site.front_used.setdefault(slot.furniture_entity, [])
        for _ in range(FRONT_TRIES):
            half = slot.rect.sy / 2 - size[1] / 2 - FRONT_MARGIN_M
            u = float(rng.uniform(-half, half))
            z = float(rng.uniform(FRONT_LOW_M, slot.z - size[2] - FRONT_MARGIN_M))
            box = (u - size[1] / 2 - FRONT_MARGIN_M / 2, u + size[1] / 2 + FRONT_MARGIN_M / 2, z - FRONT_MARGIN_M / 2,
                   z + size[2] + FRONT_MARGIN_M / 2)
            if any(box[0] < o[1] and o[0] < box[1] and box[2] < o[3] and o[2] < box[3] for o in used):
                continue
            used.append(box)
            x, y = to_world(slot.rect, slot.rect.sx / 2 + min(size[0] / 2, FRONT_SEAT_M) - 0.0005, u)
            return x, y, z, slot.rect.yaw, slot.furniture_entity
    return None


def _on_the_floor(site: _Site, rng: np.random.Generator, kind: ClutterKind, size, occ: Occupied):
    """A floor spot in a room the kind stands in: an open floor cell, or the underside of a piece that clears the
    thing, each drawn by its area (an underside `UNDER_WEIGHT` times over)."""
    rooms = {place_of(p)[1] for p in kind.places}
    if beside := {floor_beside(p) for p in kind.places} - {None}:
        return _on_floor_beside(site, rng, size, occ, [f.footprint() for f in site.m.furniture
                                                       if f.mobility == "fixed" and f.kind in beside], rooms)

    def inside(x: float, y: float) -> bool:
        return None in rooms or site.room(x, y) in rooms

    cells = [r for r in site.floor if inside(r.cx, r.cy)]
    under = [f for f in site.under if size[2] <= f.under_m - UNDER_SLACK_M and inside(*f.center)]
    places = [*cells, *under]
    weights = [*(c.sx * c.sy for c in cells), *(UNDER_WEIGHT * f.size[0] * f.size[1] for f in under)]
    for p in _weighted(rng, places, weights) if places else ():
        found = _in_cell(site, rng, size, occ, p) if isinstance(p, Rect) else _under_piece(site, rng, size, occ, p)
        if found is not None:
            return found[0], found[1], 0.0, found[2], None
    return None


def _on_floor_beside(site: _Site, rng: np.random.Generator, size, occ: Occupied, pieces: list[Rect],
                     rooms: set[str | None]):
    """A floor spot within `BESIDE_M` of one of the pieces (drawn in turn), in a room the kind stands in."""
    for i in rng.permutation(len(pieces)):
        near, room = _within(pieces[int(i)], BESIDE_M), site.room(pieces[int(i)].cx, pieces[int(i)].cy)
        if None not in rooms and room not in rooms:
            continue
        if (found := _on_floor(site, rng, size, occ, _in_room(site, room, near), near, flat=True)) is not None:
            return found[0], found[1], 0.0, found[2], None
    return None


def _place_one(world: World, site: _Site, rng: np.random.Generator, kind: ClutterKind, occ: Occupied) -> bool:
    """One entity of the kind, on a spot of its places: a model drawn, and if no place holds it the others in turn (a
    calendar that the fridge's door has no room for, a small photo that it has); False if none does."""
    models = _models(kind.models)
    for i in rng.permutation(len(models)):
        model = models[int(i)]
        spot = (_on_the_floor(site, rng, kind, model.size, occ) if kind.on_floor
                else _on_front(site, rng, kind, model.size) if kind.on_front
                else _on_surface(world, site, rng, kind, model.size, occ))
        if spot is not None:
            break
    else:
        return False
    x, y, z, yaw, on = spot
    eid = world.new_entity_id()
    colour = model.colour or str(rng.choice(kind.colors))
    world.entities.append(Entity(eid, kind.name, {"color": colour}, "clutter", model.size, asset=model.id))
    world.ledger.place(eid, 0.0, (x, y, z), yaw, on)
    world.clutter[eid] = entry = (Rect(x, y, model.size[0], model.size[1], yaw), z, z + model.size[2])
    occ.append(entry)
    if on is None:
        site.guard.block(entry[0], model.size[2])
    return True


def _allocate(rng: np.random.Generator, kinds: list[ClutterKind], target: int) -> dict[str, int]:
    """Counts per kind within each kind's range, summing to `target` where the ranges allow: every kind at its least,
    then one more at a time to a kind drawn by how much room its range has left."""
    counts = {k.name: k.count[0] for k in kinds}
    while sum(counts.values()) < target:
        open_ = [k for k in kinds if counts[k.name] < k.count[1]]
        if not open_:
            break
        room = np.array([k.count[1] - counts[k.name] for k in open_], dtype=float)
        counts[open_[int(rng.choice(len(open_), p=room / room.sum()))].name] += 1
    return counts


def _fill(world: World, site: _Site, rng: np.random.Generator, kinds: list[ClutterKind], target: int,
          occ: Occupied) -> None:
    """Place `target` clutter entities of these kinds, each kind its drawn count; what a kind has no room for is made up
    by other kinds with count to spare."""
    counts, placed = _allocate(rng, kinds, target), {k.name: 0 for k in kinds}
    def order(k: ClutterKind) -> tuple:  # what a home must hold, then the flat things on a surface (they lie under
        flat = k.size[2] <= FLAT_M and not k.on_floor  # the rest; a mat is laid first, it takes a room's open floor),
        return k.count[0] == 0, flat, len(k.places), -k.size[0] * k.size[1]  # then the scarcest place, big

    for k in sorted(kinds, key=order):
        for _ in range(counts[k.name]):
            placed[k.name] += _place_one(world, site, rng, k, occ)
    for _ in range(3 * max(0, target - sum(placed.values())) + 10):
        spare = [k for k in kinds if placed[k.name] < k.count[1]]
        if sum(placed.values()) >= target or not spare:
            break
        k = spare[int(rng.integers(len(spare)))]
        placed[k.name] += _place_one(world, site, rng, k, occ)


def place(world: World, m: MapSpec, rng: np.random.Generator) -> None:
    """Place the world's clutter: first the tracked things left on the floor (`rng`'s second stream), then 20-45 floor
    clutter pieces and enough surface clutter to bring the world's entities to 160-250 (its first), each entity with its
    placement in the ledger at t 0 and its footprint in `World.clutter`. A catalog with no clutter places none."""
    clutter_rng, floor_rng = rng.spawn(2)
    site = _Site(world, m)
    _drop_floor_things(world, site, floor_rng)
    kinds = list(world.catalog.clutter)
    if not kinds:
        return
    occ = world.occupied()
    _fill(world, site, clutter_rng, [k for k in kinds if k.on_floor],
          int(clutter_rng.integers(FLOOR_COUNT[0], FLOOR_COUNT[1] + 1)), occ)
    lo = max(SURFACE_DRAWN[0], ENTITY_COUNT[0] - len(world.entities))
    hi = max(lo, min(SURFACE_DRAWN[1], ENTITY_COUNT[1] - len(world.entities)))
    _fill(world, site, clutter_rng, [k for k in kinds if not k.on_floor], int(clutter_rng.integers(lo, hi + 1)), occ)


# -- a later move: the world's processes between tasks ---------------------------------------------------------------
def _clipped(cells: list[Rect], area: Rect) -> list[Rect]:
    """The open floor cells cut to the bounding box of `area` (a turned area is kept exact by the caller's `ok`)."""
    xs, ys = zip(*corners(area), strict=True)
    cut = [(max(c.cx - c.sx / 2, min(xs)), max(c.cy - c.sy / 2, min(ys)), min(c.cx + c.sx / 2, max(xs)),
            min(c.cy + c.sy / 2, max(ys))) for c in cells]
    return [Rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0) for x0, y0, x1, y1 in cut if x1 > x0 and y1 > y0]


def floor_spot(world: World, m: MapSpec, rng: np.random.Generator, size: tuple[float, float, float],
               area: Rect | None = None, avoid: Sequence[Rect] = (), t: float = 0.0,
               anchors: tuple[str, ...] = ()) -> tuple[float, float, float] | None:
    """(x, y, yaw) on open floor for a thing of `size` by `place`'s floor rules at time `t`: off every keep-out, every
    light piece (at its map place and where it stands in `world`) and every thing in `world`, clear of the doors in
    their state at `t`, and no cut through a planned robot's free floor (`FloorGuard`). With `area` the thing lies
    wholly inside it (a corridor stretch); with `anchors` (furniture kinds) within `DROP_REACH_M` of one of those
    pieces (a floor kind's drop); clear by `WORLD_CLEARANCE_M` of every rectangle in `avoid` (swept bodies). None when
    no spot holds it. Nothing is written: a thing that moves is first taken out of `world` (`World.without`), so it
    never blocks its own new place."""
    site, occ = _Site(world, m, t), world.occupied()
    if area is not None:
        site.floor = _clipped(site.floor, area)

    def ok(box: Rect) -> bool:
        return ((area is None or rect_inside_rect(box, area))
                and all(rect_distance(box, a) >= WORLD_CLEARANCE_M for a in avoid))

    if anchors:
        return _drop_near(world, site, rng, size, anchors, occ, ok)
    return _on_floor(site, rng, size, occ, lambda r: True, ok)


def move_floor(world: World, m: MapSpec, rng: np.random.Generator, eid: str, t: float) -> bool:
    """One floor clutter piece placed again from `t` on, by its kind's places (`ClutterKind.places`) and the floor
    rules of `floor_spot`, with its own model and size, in the world as it stands without it (it has left its old
    place at `t`): its ledger placement and `World.clutter` entry are replaced. False, and nothing changed, when no
    spot holds it."""
    entity = next(e for e in world.entities if e.id == eid)
    kind = world.catalog.clutter_kind(entity.kind)
    if not kind.on_floor:
        raise ValueError(f"{eid} is a {kind.name}, which is not floor clutter")
    rest = world.without(eid)
    spot = _on_the_floor(_Site(rest, m, t), rng, kind, entity.size, rest.occupied())
    if spot is None:
        return False
    x, y, z, yaw, _ = spot
    world.ledger.place(eid, t, (x, y, z), yaw, None)
    world.clutter[eid] = (Rect(x, y, entity.size[0], entity.size[1], yaw), z, z + entity.size[2])
    return True
