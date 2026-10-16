"""Carried items: sampling the set (identical sets and lookalikes) and putting items on receptacle surfaces.

The kinds, their colours and receptacle priors, the kinds of identical sets and the number of items come from a
`Catalog` (`catalog.for_map(m)`); every function that reads one takes `catalog=` and defaults to the home catalog, so
the apartment's draws are the same calls on the same values as before the warehouse. A kind the catalog flags `square`
(the warehouse's stock) is turned a quarter turn at a time to its surface, `SQUARE_YAW` either way; the other kinds
are turned any way.
`sample_box` is the one spot finder (a box of a size inside a rectangle, clear of what stands there): the carried
items' `_try_slot` is a call of it, and so are the clutter's, the floor's and the surfaces' (`clutter.py`), whose gaps
and yaws are their own; `free_of` is its clearance test.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .catalog import HOME, Catalog, item_size
from .geometry import Point, Rect, rects_overlap, to_world
from .mapspec import MapSpec, surface_world_rect

SQRT2 = math.sqrt(2.0)
ITEM_MARGIN = 0.015  # clear space between neighbouring items
EDGE_MARGIN = 0.01  # clear space between an item and the edge of its surface
TYPICAL_SHARE = 0.7
MIN_MOVE = 0.6  # a moved item lands at least this far (in the floor plan) from where it was
N_SETS = (2, 3)  # identical sets per world, and members per set (inclusive ranges)
SET_SIZE = (2, 4)
SET_YAW = 0.12  # radians: the members of a set sit nearly square to their surface's edge
SQUARE_YAW = 0.04  # radians: how far a `square` kind (stock on racks) strays from a quarter turn to its surface
SET_SLACK = 0.005  # extra gap between members, so a block never sits exactly on the overlap margin


@dataclass(frozen=True)
class ItemSpec:
    kind: str
    attributes: dict[str, str]

    @property
    def key(self) -> tuple:
        return (self.kind, tuple(sorted(self.attributes.items())))

    @property
    def size(self) -> tuple[float, float, float]:
        return item_size(self.kind)


@dataclass(frozen=True)
class Slot:
    """A receptacle surface in the map frame."""

    furniture_entity: str
    surface_id: str
    kind: str
    rect: Rect
    z: float
    region: str | None

    @property
    def area(self) -> float:
        return self.rect.sx * self.rect.sy


@dataclass(frozen=True)
class Spot:
    """An item's place: `pos` is the base centre (x, y, surface height)."""

    slot: Slot
    pos: tuple[float, float, float]
    yaw: float


def build_slots(m: MapSpec, entity_of_furniture: dict[str, str]) -> list[Slot]:
    """Every receptacle surface of every fixed furniture piece, in map order."""
    slots = []
    for f in m.furniture:
        if f.mobility != "fixed":
            continue
        for s in f.surfaces:
            rect = surface_world_rect(f.center, f.yaw, s)
            slots.append(Slot(entity_of_furniture[f.id], s.id, s.kind, rect, s.z, m.region_of((rect.cx, rect.cy))))
    return slots


# -- the item set ------------------------------------------------------------------------------------------------
def _random_attributes(rng: np.random.Generator, kind: str, catalog: Catalog = HOME) -> dict[str, str]:
    return {"color": str(rng.choice(catalog.kinds[kind].colors))}


def _siblings(rng: np.random.Generator, spec: ItemSpec, count: int, used: set, catalog: Catalog = HOME
              ) -> list[ItemSpec] | None:
    """`count` items that each differ from `spec` in its one attribute, the colour: None when the kind does not have
    that many other colours that are not in `used` (a kind of one colour has no lookalike)."""
    colours = [c for c in catalog.kinds[spec.kind].colors if c != spec.attributes["color"]]
    out: list[ItemSpec] = []
    for i in rng.permutation(len(colours)):
        sib = ItemSpec(spec.kind, {**spec.attributes, "color": colours[int(i)]})
        if sib.key not in used and all(sib.key != o.key for o in out):
            out.append(sib)
        if len(out) == count:
            return out
    return None


@dataclass(frozen=True)
class ItemPlan:
    """The sampled items: identical sets (each placed together, first) and every other item."""

    sets: tuple[tuple[ItemSpec, ...], ...]
    others: tuple[ItemSpec, ...]

    @property
    def items(self) -> list[ItemSpec]:
        return [spec for group in self.sets for spec in group] + list(self.others)


def sample_plan(rng: np.random.Generator, n: int | None = None, catalog: Catalog = HOME) -> ItemPlan:
    """30-40 items (the catalog's `n_items`): 2-3 identical sets of 2-4 items (each of a different kind from the
    catalog's `set_kinds`), 2-3 groups differing in colour (the one attribute), and singles.

    Every other (kind, attributes) combination is unique, so a description is ambiguous only inside an identical set or
    a lookalike group. The members of a set are kept together (`place_set`). A kind of one colour (the plant) is
    never part of a lookalike group, and the world holds at most one of it.
    """
    n = int(rng.integers(catalog.n_items[0], catalog.n_items[1] + 1)) if n is None else n
    kinds = list(catalog.kinds)
    used: set = set()
    sets: list[tuple[ItemSpec, ...]] = []
    others: list[ItemSpec] = []

    def fresh_spec() -> ItemSpec:
        for _ in range(100):
            kind = str(rng.choice(kinds))
            spec = ItemSpec(kind, _random_attributes(rng, kind, catalog))
            if spec.key not in used:
                return spec
        raise RuntimeError("catalog too small for a fresh item")

    for i in rng.permutation(len(catalog.set_kinds))[: int(rng.integers(N_SETS[0], N_SETS[1] + 1))]:
        kind = catalog.set_kinds[int(i)]
        spec = ItemSpec(kind, _random_attributes(rng, kind, catalog))
        used.add(spec.key)
        sets.append((spec,) * int(rng.integers(SET_SIZE[0], SET_SIZE[1] + 1)))
    for _ in range(int(rng.integers(2, 4))):  # groups that differ in one attribute
        for _attempt in range(50):
            spec = fresh_spec()
            sibs = _siblings(rng, spec, int(rng.integers(1, 3)), used | {spec.key}, catalog)
            if sibs:
                break
        else:
            raise RuntimeError("could not build a one-attribute group")
        for s in [spec, *sibs]:
            used.add(s.key)
            others.append(s)
    while sum(map(len, sets)) + len(others) < n:
        spec = fresh_spec()
        used.add(spec.key)
        others.append(spec)
    return ItemPlan(tuple(sets), tuple(others[int(i)] for i in rng.permutation(len(others))))


def sample_items(rng: np.random.Generator, n: int | None = None, catalog: Catalog = HOME) -> list[ItemSpec]:
    """The plan's items as one list (the sets first), for callers that need what exists and not how it is grouped."""
    return sample_plan(rng, n, catalog).items


# -- placing on surfaces -----------------------------------------------------------------------------------------
Occupied = list[tuple[Rect, float, float]]  # footprint, z bottom, z top


def item_rect(spec: ItemSpec, x: float, y: float, yaw: float) -> Rect:
    sz = spec.size
    return Rect(x, y, sz[0], sz[1], yaw)


def occupied_entry(spec: ItemSpec, pos: tuple[float, float, float], yaw: float) -> tuple[Rect, float, float]:
    sz = spec.size
    return (item_rect(spec, pos[0], pos[1], yaw), pos[2], pos[2] + sz[2])


def free_of(rect: Rect, z_lo: float, z_hi: float, occupied: Occupied, margin: float = ITEM_MARGIN) -> bool:
    """Whether a box (footprint `rect`, from `z_lo` to `z_hi`) is clear, by `margin`, of everything in `occupied` that
    shares some of its height. A box can only be within `margin` of another along both axes of the first when their
    centres are less than sqrt(2) of their circumradii and the margin apart: the cheap test that skips the rest."""
    reach = math.hypot(rect.sx, rect.sy) / 2 + margin
    return not any(z_lo < hi and lo < z_hi and math.hypot(rect.cx - r.cx, rect.cy - r.cy) < SQRT2 * (
        reach + math.hypot(r.sx, r.sy) / 2) and rects_overlap(rect, r, margin) for r, lo, hi in occupied)


def _yaw(rng: np.random.Generator, slot: Slot, square: bool) -> float:
    """An item's turn on `slot`: any angle, or for a `square` kind one of the four quarter turns to the surface, give
    or take `SQUARE_YAW`."""
    if not square:
        return float(rng.uniform(-math.pi, math.pi))
    return slot.rect.yaw + int(rng.integers(0, 4)) * math.pi / 2 + float(rng.uniform(-SQUARE_YAW, SQUARE_YAW))


def sample_box(rng: np.random.Generator, size: tuple[float, float, float], rect: Rect, z: float, occupied: Occupied,
               yaw_draw: Callable[[], float], tries: int = 25, ok: Callable[[Rect], bool] | None = None,
               edge: float = EDGE_MARGIN, gap: float = ITEM_MARGIN) -> tuple[float, float, float] | None:
    """The one spot finder: (x, y, yaw) of a box of `size` standing at height `z` wholly inside `rect` (`edge` from its
    edges) and clear of `occupied` by `gap` (`free_of`), or None after `tries` draws. Each try draws a yaw (`yaw_draw`),
    skips it when the turned box is too big for `rect`, then draws the centre uniformly where it fits; `ok` may refuse
    the resulting footprint without drawing anything, so it never shifts a stream."""
    for _ in range(tries):
        yaw = yaw_draw()
        a = yaw - rect.yaw
        hx = (size[0] * abs(math.cos(a)) + size[1] * abs(math.sin(a))) / 2 + edge
        hy = (size[0] * abs(math.sin(a)) + size[1] * abs(math.cos(a))) / 2 + edge
        if hx > rect.sx / 2 or hy > rect.sy / 2:
            continue
        lx = float(rng.uniform(-(rect.sx / 2 - hx), rect.sx / 2 - hx))
        ly = float(rng.uniform(-(rect.sy / 2 - hy), rect.sy / 2 - hy))
        x, y = to_world(rect, lx, ly)
        box = Rect(x, y, size[0], size[1], yaw)
        if free_of(box, z, z + size[2], occupied, gap) and (ok is None or ok(box)):
            return x, y, yaw
    return None


def _try_slot(rng: np.random.Generator, spec: ItemSpec, slot: Slot, occupied: Occupied, tries: int = 25,
              square: bool = False) -> Spot | None:
    found = sample_box(rng, spec.size, slot.rect, slot.z, occupied, lambda: _yaw(rng, slot, square), tries)
    return None if found is None else Spot(slot, (found[0], found[1], slot.z), found[2])


def _slot_order(rng: np.random.Generator, slots: list[Slot], typical_kinds: set[str]) -> Iterator[Slot]:
    """The surfaces in the order to try them: 70% of the time the typical kinds first, 30% the others first (the rest
    follow); each draw is area-weighted, without replacement."""
    want_typical = bool(rng.random() < TYPICAL_SHARE)
    for pool in (
        [s for s in slots if (s.kind in typical_kinds) == want_typical],
        [s for s in slots if (s.kind in typical_kinds) != want_typical],
    ):
        while pool:
            weights = np.array([s.area for s in pool])
            yield pool.pop(int(rng.choice(len(pool), p=weights / weights.sum())))


def find_spot(
    rng: np.random.Generator,
    spec: ItemSpec,
    slots: list[Slot],
    occupied: Occupied,
    *,
    avoid: Point | None = None,
    avoid_slot: Slot | None = None,
    min_move: float = MIN_MOVE,
    catalog: Catalog = HOME,
) -> Spot | None:
    """A free spot for the item: 70% on a typical receptacle kind, 30% elsewhere (falling back if that is full).

    With `avoid`/`avoid_slot` the spot must differ from the item's current surface and lie >= min_move away from it.
    """
    avoid_key = None if avoid_slot is None else (avoid_slot.furniture_entity, avoid_slot.surface_id)
    usable = [s for s in slots if (s.furniture_entity, s.surface_id) != avoid_key]
    kind = catalog.kinds[spec.kind]
    for slot in _slot_order(rng, usable, set(kind.receptacles)):
        for _ in range(2):
            spot = _try_slot(rng, spec, slot, occupied, square=kind.square)
            if spot is None:
                continue
            if avoid is not None and math.hypot(spot.pos[0] - avoid[0], spot.pos[1] - avoid[1]) < min_move:
                continue
            return spot
    return None


# -- identical sets ------------------------------------------------------------------------------------------------
def _block(rng: np.random.Generator, specs: tuple[ItemSpec, ...], slot: Slot, occupied: Occupied,
           yaw_slack: float = SET_YAW) -> list[Spot] | None:
    """All the members on `slot`: side by side in a row (or two rows, for three or more), `ITEM_MARGIN` apart, all
    turned the same way (a quarter turn to the surface, `yaw_slack` either way). None if they do not fit or something
    is in the way. Laid out in the surface's own frame."""
    n, sz = len(specs), specs[0].size
    gap = ITEM_MARGIN + SET_SLACK
    size = (slot.rect.sx, slot.rect.sy)
    axes = (0, 1) if size[0] >= size[1] else (1, 0)  # the longer side first
    layouts = [(rows, axis) for rows in (1, 2) if rows == 1 or n >= 3 for axis in axes]
    for _ in range(2):
        a = float(rng.choice((0.0, math.pi / 2))) + float(rng.uniform(-yaw_slack, yaw_slack))  # yaw within the surface
        ext = (sz[0] * abs(math.cos(a)) + sz[1] * abs(math.sin(a)),  # one member's extents along the surface's x and y
               sz[0] * abs(math.sin(a)) + sz[1] * abs(math.cos(a)))
        for rows, axis in layouts:  # `axis` is the direction of a row; 0 is the surface's x
            cols = math.ceil(n / rows)
            e_row, e_across = ext[axis], ext[1 - axis]
            span_row, span_across = cols * e_row + (cols - 1) * gap, rows * e_across + (rows - 1) * gap
            room_row = size[axis] - 2 * EDGE_MARGIN - span_row
            room_across = size[1 - axis] - 2 * EDGE_MARGIN - span_across
            if room_row < 0 or room_across < 0:
                continue
            for _try in range(4):
                o_row = float(rng.uniform(-room_row / 2, room_row / 2))
                o_across = float(rng.uniform(-room_across / 2, room_across / 2))
                spots = []
                for j in range(n):
                    r, c = divmod(j, cols)
                    along = o_row - span_row / 2 + e_row / 2 + c * (e_row + gap)
                    across = o_across - span_across / 2 + e_across / 2 + r * (e_across + gap)
                    x, y = to_world(slot.rect, *((along, across) if axis == 0 else (across, along)))
                    spots.append(Spot(slot, (x, y, slot.z), slot.rect.yaw + a))
                if all(free_of(item_rect(specs[0], s.pos[0], s.pos[1], s.yaw), slot.z, slot.z + sz[2], occupied)
                       for s in spots):
                    return spots
    return None


def _spread(rng: np.random.Generator, specs: tuple[ItemSpec, ...], slots: list[Slot], occupied: Occupied,
            catalog: Catalog = HOME) -> list[Spot | None]:
    """Each member, one after another, on any of `slots` (None where one finds no room)."""
    taken, out = list(occupied), []
    for spec in specs:
        spot = find_spot(rng, spec, slots, taken, catalog=catalog)
        out.append(spot)
        if spot is not None:
            taken.append(occupied_entry(spec, spot.pos, spot.yaw))
    return out


def place_set(rng: np.random.Generator, specs: tuple[ItemSpec, ...], slots: list[Slot], occupied: Occupied,
              catalog: Catalog = HOME) -> list[Spot | None]:
    """Spots for the members of an identical set, one per spec (None where an item found no room).

    First the whole set in one block on one surface (`_block`), the surface drawn as `find_spot` draws one (typical
    kinds 70% of the time) among those where the set fits. Failing that the members spread over the surfaces of one
    piece of furniture, and failing that each goes where it can. `occupied` is not changed.
    """
    kind = catalog.kinds[specs[0].kind]
    for slot in _slot_order(rng, slots, set(kind.receptacles)):
        if (spots := _block(rng, specs, slot, occupied, SQUARE_YAW if kind.square else SET_YAW)) is not None:
            return spots
    pieces: dict[str, list[Slot]] = {}
    for s in slots:
        pieces.setdefault(s.furniture_entity, []).append(s)
    pool = [p for p in pieces.values() if len(p) > 1]
    while pool:
        weights = np.array([sum(s.area for s in p) for p in pool])
        spots = _spread(rng, specs, pool.pop(int(rng.choice(len(pool), p=weights / weights.sum()))), occupied, catalog)
        if None not in spots:
            return spots
    return _spread(rng, specs, slots, occupied, catalog)
