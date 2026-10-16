"""World changes inside a gap between tasks: item moves, furniture shifts, adds and removes (V0's, which M2's gaps no
longer call: `world_processes` draws a gap's changes by half-lives, all at one instant), and `World`, what
`generate.build_world` returns and every process changes.

Every change is a new placement interval that starts at a time strictly inside the gap, so the ego never sees a change
happen. Counts follow V0 §4: Poisson(2) carried-item moves per room, 0-2 light-furniture shifts, and about one add or
remove. Moves and shifts displace by at least 0.6 m. An added item never matches the description (kind and attributes)
of any item already in the world. Doors do not change in M1: they stay open throughout. What is added and what is
removed come from the world's catalog (`catalog.for_map`): a delivery box and a bag or the box in the apartment, a
pallet and a pallet, box or tote in the warehouse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .catalog import Catalog, for_map
from .geometry import Rect, corners, dist_to_rect, point_in_polygon, rects_overlap
from .ledger import Ledger
from .mapspec import INTERIOR_TOL, MapSpec
from .placement import ItemSpec, Slot, Spot, find_spot, occupied_entry
from .route import blocking_rects
from .timeline import Entity

# One clearance for every world process (furniture shifts, movers, dressing zones); a robot's own planning reads its
# `body.clearance_m`. It is the value the retired profiles kept, so no world is redrawn.
WORLD_CLEARANCE_M = 0.25
MOVES_PER_ROOM = 2.0
SHIFT_RANGE = (0.6, 1.0)
EDGE_BUFFER = 30.0  # seconds kept clear at both ends of the gap
FURNITURE_MARGIN = 0.05
KEYFRAME_MARGIN = 0.05  # extra distance a shifted piece keeps from every later camera position


@dataclass
class Carried:
    entity: str
    spec: ItemSpec
    spot: Spot


@dataclass
class World:
    """The mutable state of the world at the end of the latest gap, plus the ledger recording how it got there."""

    m: MapSpec
    ledger: Ledger
    slots: list[Slot]
    entities: list[Entity]
    carried: dict[str, Carried] = field(default_factory=dict)
    clutter: dict[str, tuple[Rect, float, float]] = field(default_factory=dict)  # entity id -> `occupied_entry`
    light_rects: dict[str, Rect] = field(default_factory=dict)
    door_entity: dict[str, str] = field(default_factory=dict)  # map door id -> entity id
    used_keys: set = field(default_factory=set)  # (kind, attributes) of every item that ever existed
    _next: int = 1

    @property
    def catalog(self) -> Catalog:
        return for_map(self.m)

    def new_entity_id(self) -> str:
        eid = f"E{self._next:04d}"
        self._next += 1
        return eid

    def occupied(self) -> list[tuple[Rect, float, float]]:
        """Where carried items and clutter stand now, so a later move cannot land inside either."""
        return [occupied_entry(c.spec, c.spot.pos, c.spot.yaw) for c in self.carried.values()] + list(
            self.clutter.values())

    def without(self, eid: str) -> World:
        """The world as it stands with one carried item, clutter piece or light piece not in it (the ledger shared, the
        rest as it stands): what a thing that moves finds at the instant it leaves, so it never blocks its own new
        place."""
        return replace(self, carried={k: v for k, v in self.carried.items() if k != eid},
                       clutter={k: v for k, v in self.clutter.items() if k != eid},
                       light_rects={k: v for k, v in self.light_rects.items() if k != eid})


def when_in_gap(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A whole second strictly inside the gap (lo, hi), at least EDGE_BUFFER from both ends."""
    return float(rng.integers(int(lo + EDGE_BUFFER), int(hi - EDGE_BUFFER) + 1))


# -- light furniture ---------------------------------------------------------------------------------------------
def _shift_valid(w: World, eid: str, new: Rect, others: list[Rect], later_xy: np.ndarray) -> bool:
    m = w.m
    old = w.light_rects[eid]
    region = next((r for r in m.regions if point_in_polygon(old.cx, old.cy, list(r.polygon))), None)
    if region is None or not all(point_in_polygon(x, y, list(region.polygon)) for x, y in corners(new, INTERIOR_TOL)):
        return False
    if any(rects_overlap(new, r, FURNITURE_MARGIN) for r in blocking_rects(m) + others):
        return False
    for o in (o for o in m.openings if o.interior):
        c, radius = m.door_keepout(o)
        if float(dist_to_rect(np.array([c[0]]), np.array([c[1]]), new)[0]) < radius:
            return False
    if len(later_xy):
        nearest = float(dist_to_rect(later_xy[:, 0], later_xy[:, 1], new).min())
        return nearest >= WORLD_CLEARANCE_M + KEYFRAME_MARGIN
    return True


def shift_light_furniture(w: World, rng: np.random.Generator, lo: float, hi: float, later_xy: np.ndarray) -> int:
    """0-2 pieces of light furniture slide 0.6-1.0 m (and may turn), staying in their room, off the doors, and clear
    of every camera position in the visits that follow (`later_xy`, an (n, 2) array). The count is drawn on every map;
    a map with no light furniture left to move (the downtown's) shifts nothing and draws nothing more."""
    done = 0
    ids = list(w.light_rects)
    busy: list[Rect] = []  # where shifted pieces stood before: kept clear so no two moves can collide mid-gap
    moved: set[str] = set()
    for _ in range(int(rng.integers(0, 3))):
        free_ids = [i for i in ids if i not in moved]
        if not free_ids:
            break
        eid = free_ids[int(rng.integers(0, len(free_ids)))]
        old = w.light_rects[eid]
        for _attempt in range(60):
            ang = float(rng.uniform(-math.pi, math.pi))
            dist = float(rng.uniform(*SHIFT_RANGE))
            dyaw = float(rng.uniform(-0.8, 0.8)) if rng.random() < 0.5 else 0.0
            new = Rect(old.cx + dist * math.cos(ang), old.cy + dist * math.sin(ang), old.sx, old.sy, old.yaw + dyaw)
            others = [r for i, r in w.light_rects.items() if i != eid] + busy
            if _shift_valid(w, eid, new, others, later_xy):
                w.ledger.place(eid, when_in_gap(rng, lo, hi), (new.cx, new.cy, 0.0), new.yaw, None)
                w.light_rects[eid] = new
                busy.append(old)
                moved.add(eid)
                done += 1
                break
    return done


# -- carried items -----------------------------------------------------------------------------------------------
def move_carried(w: World, rng: np.random.Generator, lo: float, hi: float, occupied: list) -> set[str]:
    """Per room, about Poisson(2) items move to another surface (at least 0.6 m away), sometimes in another room.

    `occupied` starts as w.occupied() and collects every footprint used at any time in the gap (old and new spots),
    so a later change in the same gap can never land on an item that has not left yet.
    """
    moved: set[str] = set()
    for region in w.m.regions:
        n = int(rng.poisson(MOVES_PER_ROOM))
        cands = [c for c in w.carried.values() if c.spot.slot.region == region.id and c.entity not in moved]
        for idx in rng.permutation(len(cands))[:n]:
            c = cands[int(idx)]
            spot = find_spot(rng, c.spec, w.slots, occupied, avoid=(c.spot.pos[0], c.spot.pos[1]),
                             avoid_slot=c.spot.slot, catalog=w.catalog)
            if spot is None:
                continue
            w.ledger.place(c.entity, when_in_gap(rng, lo, hi), spot.pos, spot.yaw, spot.slot.furniture_entity)
            c.spot = spot
            occupied.append(occupied_entry(c.spec, spot.pos, spot.yaw))
            moved.add(c.entity)
    return moved


def _fresh_delivery_spec(w: World) -> ItemSpec | None:
    """A delivery (the catalog's kind) whose description no item in the world has had: the first colour that is free."""
    for color in w.catalog.delivery_colors:
        spec = ItemSpec(w.catalog.delivery_kind, {"color": color})
        if spec.key not in w.used_keys:
            return spec
    return None


def add_or_remove(w: World, rng: np.random.Generator, lo: float, hi: float, touched: set[str], occupied: list) -> int:
    """About one add or remove per gap: a delivery appears, or an item of a leaving kind goes."""
    done = 0
    for _ in range(int(rng.choice([0, 1, 1, 2]))):
        if rng.random() < 0.5:
            spec = _fresh_delivery_spec(w)
            spot = None if spec is None else find_spot(rng, spec, w.slots, occupied, catalog=w.catalog)
            if spec is None or spot is None:
                continue
            occupied.append(occupied_entry(spec, spot.pos, spot.yaw))
            eid = w.new_entity_id()
            w.entities.append(Entity(eid, spec.kind, dict(spec.attributes), "carried", spec.size))
            w.ledger.place(eid, when_in_gap(rng, lo, hi), spot.pos, spot.yaw, spot.slot.furniture_entity)
            w.carried[eid] = Carried(eid, spec, spot)
            w.used_keys.add(spec.key)
            touched.add(eid)
            done += 1
        else:
            cands = [c for c in w.carried.values() if c.entity not in touched]
            leaving = [c for c in cands if c.spec.kind in w.catalog.leaving_kinds] or cands
            if not leaving:
                continue
            c = leaving[int(rng.integers(0, len(leaving)))]
            w.ledger.remove(c.entity, when_in_gap(rng, lo, hi))
            del w.carried[c.entity]
            touched.add(c.entity)
            done += 1
    return done
