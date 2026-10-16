"""Moves that stay on surfaces: what a surface coin (`helper/1`'s place spot) may do to the world without touching the
floor, so no robot's free space and no planned path changes.

`clutter.move_floor` places a floor clutter piece again by its kind's places and the floor rules; `move_surface` is its
counterpart for the clutter that stands on a table, a counter or a shelf (`ClutterKind.places` names a surface kind of
the map), by the same spot finder (`clutter._on_surface`) in the world as it stands without the piece. `move_to_surface`
moves a carried item by the world's own law (`world_processes_home.move_item`: the start's split, a thing of a floor
kind left on the floor with `clutter.DROP_SHARE`, else a receptacle) and keeps the result only when the item ends on a
raised surface: a floor drop is refused and drawn again, so the floor is as it was. A caller may refuse a spot too (one
that is still where the coin wants it empty). `standing` lists what stands in a column over a surface: the carried items
and clutter whose footprint meets a rectangle and whose body reaches into a band above the surface. `free_spot` is the
scan for a place for a box on a surface (nearest an aim point, no draw), and `slide_surface` slides a clutter piece to
the nearest such place.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from . import clutter
from .catalog import FLOOR_ITEM_Z
from .changes import Carried, World
from .geometry import Point, Rect, rects_overlap, to_local, to_world
from .ledger import Ledger
from .mapspec import MapSpec
from .placement import EDGE_MARGIN, ITEM_MARGIN, Occupied, Slot, Spot, free_of, occupied_entry
from .world_processes_home import move_item

TRIES = 12  # spots a thing is drawn again for when one is refused
STEP_M = 0.02  # `free_spot` looks at spots this far apart
SURFACE_Z_TOL = 0.01  # a thing stands on a surface when its base is this near the surface's height

Band = tuple[float, float]  # heights above the surface that block: (taller than, base below)


def standing(w: World, rect: Rect, ground: float, band: Band, margin: float = 0.0) -> list[str]:
    """The carried items and clutter pieces, carried first, in the world's order, whose footprint is within `margin` of
    `rect` and which reach into the band above `ground`: the top higher than `ground + band[0]` and the base lower than
    `ground + band[1]`."""
    low, high = ground + band[0], ground + band[1]
    out = []
    for eid, c in w.carried.items():
        box, z0, z1 = occupied_entry(c.spec, c.spot.pos, c.spot.yaw)
        if z1 > low and z0 < high and rects_overlap(box, rect, margin):
            out.append(eid)
    for eid, (box, z0, z1) in w.clutter.items():
        if z1 > low and z0 < high and rects_overlap(box, rect, margin):
            out.append(eid)
    return out


def move_surface(world: World, m: MapSpec, rng: np.random.Generator, eid: str, t: float,
                 refuse: Callable[[Rect, float, float], bool] | None = None) -> bool:
    """One surface clutter piece placed again from `t` on, by its kind's places, with its own model and size, in the
    world as it stands without it (it has left its old place at `t`): its ledger placement and `World.clutter` entry
    are replaced. `refuse(footprint, z0, z1)` turns a spot down and the piece is drawn another, up to `TRIES` times.
    False, and nothing changed, when no spot holds it. ValueError for a clutter piece that does not stand on a
    surface."""
    entity = next(e for e in world.entities if e.id == eid)
    kind = world.catalog.clutter_kind(entity.kind)
    if kind.on_floor or kind.on_front:
        raise ValueError(f"{eid} is a {kind.name}, which does not stand on a surface")
    rest = world.without(eid)
    site = clutter._Site(rest, m, t)  # noqa: SLF001  (clutter.py is at its size bound: the finder stays where it is)
    occ = rest.occupied()
    for _ in range(TRIES):
        spot = clutter._on_surface(rest, site, rng, kind, entity.size, occ)  # noqa: SLF001
        if spot is None:
            return False
        x, y, z, yaw, on = spot
        box = Rect(x, y, entity.size[0], entity.size[1], yaw)
        if refuse is None or not refuse(box, z, z + entity.size[2]):
            world.ledger.place(eid, t, (x, y, z), yaw, on)
            world.clutter[eid] = (box, z, z + entity.size[2])
            return True
    return False


def move_to_surface(w: World, m: MapSpec, rng: np.random.Generator, eid: str, t: float,
                    refuse: Callable[[Spot], bool] | None = None) -> bool:
    """A carried item to a new spot on a raised surface from `t` on, by the world's own law (`move_item`) tried on a
    copy of the world: a result on the floor, or one `refuse` turns down, is thrown away and the item drawn again, up
    to `TRIES` times, so only a spot on a surface is written. False, and nothing changed, when none holds it."""
    for _ in range(TRIES):
        copied = {k: Carried(c.entity, c.spec, c.spot) for k, c in w.carried.items()}
        scratch = replace(w, ledger=Ledger(), carried=copied)
        if not move_item(scratch, m, rng, eid, t):
            return False
        spot = scratch.carried[eid].spot
        if spot.pos[2] >= FLOOR_ITEM_Z and spot.slot.furniture_entity and (refuse is None or not refuse(spot)):
            w.ledger.place(eid, t, spot.pos, spot.yaw, spot.slot.furniture_entity)
            w.carried[eid].spot = spot
            return True
    return False


def free_spot(rest: Occupied, size: tuple[float, float, float], slot: Slot, aim: Point, keep_out: Rect | None = None,
              margin: float = 0.0) -> tuple[float, float, float] | None:
    """(x, y, yaw) of the spot on the slot nearest `aim` where a box of `size` lies wholly on the surface, `EDGE_MARGIN`
    from its edges and a quarter turn to it (the longer side along the surface's first), clear by `ITEM_MARGIN` of
    everything in `rest`, and not within `margin` of `keep_out`. The aim point itself (moved inside) comes first, then a
    grid `STEP_M` apart; the nearest wins, the first turn on a tie. None when no spot holds the box."""
    near = [o for o in rest if rects_overlap(o[0], slot.rect, ITEM_MARGIN)]
    turns = (0.0, math.pi / 2) if (size[0] >= size[1]) == (slot.rect.sx >= slot.rect.sy) else (math.pi / 2, 0.0)
    found: list[tuple[float, int, float, float, float]] = []  # (distance, turn order, x, y, yaw)
    for order, turn in enumerate(turns):
        ex, ey = (size[0], size[1]) if turn == 0.0 else (size[1], size[0])
        hx, hy = slot.rect.sx / 2 - ex / 2 - EDGE_MARGIN, slot.rect.sy / 2 - ey / 2 - EDGE_MARGIN
        if hx < 0 or hy < 0:
            continue
        nx, ny = max(round(2 * hx / STEP_M), 1), max(round(2 * hy / STEP_M), 1)
        xs = [-hx + 2 * hx * i / nx for i in range(nx + 1)]
        ys = [-hy + 2 * hy * i / ny for i in range(ny + 1)]
        lx0, ly0 = to_local(slot.rect, *aim)
        centre = (max(-hx, min(hx, lx0)), max(-hy, min(hy, ly0)))
        for lx, ly in [centre] + [(a, b) for a in xs for b in ys]:
            x, y = to_world(slot.rect, lx, ly)
            found.append((math.hypot(x - aim[0], y - aim[1]), order, x, y, slot.rect.yaw + turn))
    for _, _, x, y, yaw in sorted(found, key=lambda f: f[:2]):
        box = Rect(x, y, size[0], size[1], yaw)
        if keep_out is not None and rects_overlap(box, keep_out, margin):
            continue
        if free_of(box, slot.z, slot.z + size[2], near, ITEM_MARGIN):
            return x, y, yaw
    return None


def slide_surface(w: World, eid: str, t: float, slot: Slot, aim: Point, keep_out: Rect, margin: float) -> bool:
    """A clutter piece that stands on the slot slid from `t` on to the free spot nearest `aim` that is not within
    `margin` of `keep_out` (`free_spot`, in the world as it stands without the piece), turned a quarter turn to the
    surface. False, and nothing changed, when it stands on another surface or no spot holds it."""
    box, z0, z1 = w.clutter[eid]
    if abs(z0 - slot.z) > SURFACE_Z_TOL:
        return False
    found = free_spot(w.without(eid).occupied(), (box.sx, box.sy, z1 - z0), slot, aim, keep_out, margin)
    if found is None:
        return False
    x, y, yaw = found
    w.ledger.place(eid, t, (x, y, slot.z), yaw, slot.furniture_entity)
    w.clutter[eid] = (Rect(x, y, box.sx, box.sy, yaw), slot.z, slot.z + (z1 - z0))
    return True
