"""The zone rule: a place the job's next manoeuvre uses, found on the public map by its piece kind.

A job declares its zones in `JobSpec.zones` (a decision kind or an event wording -> `ZoneRule`) and `JobSpec.slot_zones`
(the rules `slots_empty` asks about over a whole window). A zone is a piece's footprint on the map (never a surface, so
nothing new becomes a receptacle), grown by `margin_m` and then inset by `EDGE_M`, on the ground the piece lies on; it
comes from the map and the robot's pose alone, never from the world. This module is neutral (a table system reads it
too): the geometry (`zones_of`, `zone_of`, `zone_cells`) and the state of a zone over free-space cells (`zone_state`,
`coverage`) are the one rule the gold (`questions_zone`) and the table systems (`table_zone`) both call.

**The inset.** A bay touches its station's face, and a station's top and boards span its whole footprint at 0.47 to
2.0 m: the points of its face fall in sub-cells centred up to 0.01 m inside the bay (the 0.02 m sub-cell's centre, the
depth's millimetre rounding), so an uninset bay seen empty held 100 blocking sub-cells (a fixed piece read as an
occupant). Two sub-cells of inset keep every piece that only touches a zone out of it.

**The sources.** `ZoneRule.piece` is a furniture kind, or `DOORWAY` (the way through every interior opening,
`zone_shapes.doorway_way`, on the floor) or `SPOT` (the window of the surface the base faces,
`zone_shapes.surface_window`, on that surface: it needs the base's heading, `zone_of`'s `yaw`). Those shapes carry
their own insets (`JAMB_M`, `INSET_M`), so `EDGE_M` is not taken off them, and `margin_m` still grows them. For
`SPOT`, `zones_of` lists each blocking fixed piece's whole inset largest surface, which holds every window `zone_of`
can return, so `band_cells` keeps every cell a window can hold.

**The states.** A zone is judged on its own band: `observe` with the zone's ground (`base_z`) as the floor and
`(height_m, top_m)` above it, so a marking on a 0.15 m plaza is judged from the plaza's top and a flier's own band is
never read. `zone_state` takes `grid.latest`'s output over such cells.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from . import grid, zone_shapes
from .geometry import Point, Rect, corners, point_in_rect
from .mapspec import Furniture, MapSpec

Cell = tuple[int, int]
Band = tuple[float, float]


TOP_M = 3.5  # the band's top above the zone's ground (a 3.3 m box truck's roof is inside it: M3-Y)
DOORWAY = "doorway"  # a `ZoneRule.piece` that is no furniture kind: the way through every interior opening
SPOT = "spot"  # likewise: the window of the surface the base faces (`zone_of` needs its `yaw`)


@dataclass(frozen=True)
class ZoneRule:
    """A place the job's next manoeuvre uses, found on the public map by its piece kind."""

    words: str  # how a question names it: "the drop zone", "the bay ahead of you"
    piece: str  # the map furniture kind whose footprint is the zone: "drop_zone", "bay"; or DOORWAY, SPOT
    piece_words: str = ""  # the pieces in the plural, for slots_empty: "staging lanes"
    margin_m: float = 0.0  # the footprint grown by this much on every side
    height_m: float = 0.3  # "clear of anything taller than": the band starts this far above the zone's ground
    top_m: float = TOP_M  # the band ends this far above it (a thing put down on a surface needs far less than 3.5 m)


ZONE_SHARE = 0.8  # the share of a zone's fine cells that must be seen free before it is called clear
GROUND_MAX_M = 0.2  # a fixed piece no taller than this is ground: a zone lies on the top of such a piece
EDGE_M = 0.05  # every zone is its piece's footprint inset by this much (two sub-cells): see the module docstring


@dataclass(frozen=True)
class Zone:
    piece: str  # the piece's map id
    polygon: tuple[Point, ...]  # its footprint grown by `rule.margin_m - EDGE_M`, counter-clockwise (a rectangle)
    base_z: float  # its ground: `ground_z` (a way: the floor; a window: its surface's height)
    band: Band  # (rule.height_m, rule.top_m): the heights above `base_z` that block it


def ground_z(m: MapSpec, piece: Furniture) -> float:
    """The top of the tallest other fixed piece of at most `GROUND_MAX_M` whose footprint holds the piece's centre (a
    marking on a plaza is judged from the plaza's top); 0 when there is none. A taller piece is no ground."""
    return max((f.size[2] for f in m.furniture
                if f.id != piece.id and f.mobility == "fixed" and f.size[2] <= GROUND_MAX_M
                and point_in_rect(f.footprint(), piece.center[0], piece.center[1])), default=0.0)


def _shaped(piece: str, rect: Rect, base_z: float, rule: ZoneRule) -> Zone:
    """The zone on a rectangle that already carries its own inset (a way, a surface or its window): the rule's margin
    grows it. ValueError when the margin leaves it no area."""
    if min(rect.sx, rect.sy) / 2 + rule.margin_m <= 0:
        raise ValueError(f"the {rule.piece} zone of {piece} is {rect.sx:g} x {rect.sy:g} m: grown by "
                         f"{rule.margin_m:g} m it has no area")
    return Zone(piece, tuple(corners(rect, -rule.margin_m)), base_z, (rule.height_m, rule.top_m))


def zones_of(m: MapSpec, rule: ZoneRule) -> list[Zone]:
    """The zone of every piece of kind `rule.piece`, in map order. ValueError names a piece the inset leaves with no
    area (a map bug). `DOORWAY`: the way of every interior opening, in map order, named by the opening's id, on the
    floor. `SPOT`: the whole inset largest surface of every blocking fixed piece that has one, named by the piece's id,
    at the surface's height: it holds every window `zone_of` returns for the piece, so `band_cells` loses none."""
    if rule.piece == DOORWAY:
        return [_shaped(o.id, zone_shapes.doorway_way(m, o), 0.0, rule) for o in m.openings if o.interior]
    if rule.piece == SPOT:
        return [_shaped(f.id, zone_shapes.inset_surface(f, s), s.z, rule)
                for f in zone_shapes.blocking_pieces(m) if (s := zone_shapes.largest_surface(f)) is not None]
    out = []
    for f in m.furniture:
        if f.kind != rule.piece:
            continue
        grow = rule.margin_m - EDGE_M
        if min(f.size[0], f.size[1]) / 2 + grow <= 0:
            raise ValueError(f"the {f.kind} {f.id} ({f.name}) is {f.size[0]:g} x {f.size[1]:g} m: grown by "
                             f"{grow:g} m it has no area")
        out.append(Zone(f.id, tuple(corners(f.footprint(), -grow)), ground_z(m, f), (rule.height_m, rule.top_m)))
    return out


def centroid(zone: Zone) -> Point:
    """The zone's centre (the mean of its four corners: it is a rectangle), to 6 decimals."""
    n = len(zone.polygon)
    return (round(sum(x for x, _ in zone.polygon) / n, 6), round(sum(y for _, y in zone.polygon) / n, 6))


def zone_of(m: MapSpec, rule: ZoneRule, near_xy, yaw: float | None = None) -> Zone | None:
    """The zone whose centre is nearest `near_xy`, the first in map order on a tie (1e-9); None when the map has no
    piece of the kind. It is chosen from the public map and the robot's own pose, so a table system finds the same.
    `SPOT` has no nearest piece: its zone is the window of the surface the base at `near_xy` faces along `yaw`
    (`zone_shapes.surface_window`), None when there is none, and ValueError without a `yaw`."""
    if rule.piece == SPOT:
        if yaw is None:
            raise ValueError("a spot zone is the window of the surface the base faces: zone_of needs its yaw")
        if (found := zone_shapes.surface_window(m, near_xy, yaw)) is None:
            return None
        piece, surface, window = found
        return _shaped(piece.id, window, surface.z, rule)
    best, best_d = None, math.inf
    for z in zones_of(m, rule):
        d = math.dist(centroid(z), near_xy)
        if d < best_d - 1e-9:
            best, best_d = z, d
    return best


# -- the cells --------------------------------------------------------------------------------------------------
def _inside(polygon: tuple[Point, ...], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which of the points lie in the convex counter-clockwise polygon (a point within 1e-9 of an edge is inside, as
    for the cells)."""
    inside = np.ones(x.shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        inside &= (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) >= -grid.EPS
    return inside


def _box(polygon: tuple[Point, ...], pad: int) -> tuple[np.ndarray, np.ndarray]:
    """The fine cells (u, v as flat arrays) of the polygon's bounding box, `pad` cells wider on every side."""
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    us = np.arange(math.floor(min(xs) / grid.FINE_M) - pad, math.floor(max(xs) / grid.FINE_M) + pad + 1)
    vs = np.arange(math.floor(min(ys) / grid.FINE_M) - pad, math.floor(max(ys) / grid.FINE_M) + pad + 1)
    u, v = np.meshgrid(us, vs, indexing="ij")
    return u.ravel(), v.ravel()


def zone_cells(zone: Zone) -> list[Cell]:
    """The fine cells whose centres lie in the zone's polygon (a centre on an edge, to 1e-9, is in), sorted."""
    u, v = _box(zone.polygon, 1)
    keep = _inside(zone.polygon, (u + 0.5) * grid.FINE_M, (v + 0.5) * grid.FINE_M)
    return sorted(zip(u[keep].tolist(), v[keep].tolist(), strict=True))


def band_cells(m: MapSpec, rules: Iterable[ZoneRule]) -> dict[tuple[float, Band], frozenset[Cell]]:
    """The fine cells of every zone of the rules, by the (`base_z`, `band`) its states are observed with: all a zone
    band's states need keep (a drone frame observes tens of thousands of cells that no zone reads)."""
    out: dict[tuple[float, Band], set[Cell]] = {}
    for rule in rules:
        for z in zones_of(m, rule):
            out.setdefault((z.base_z, z.band), set()).update(zone_cells(z))
    return {key: frozenset(cells) for key, cells in out.items()}


# -- the states -------------------------------------------------------------------------------------------------
def _blocking(states: Mapping[Cell, tuple[str, int]], zone: Zone) -> list[Cell]:
    """The occupied fine cells that hold an obstacle sub-cell (`grid.Occupied.subs`, 0.02 m) whose centre lies in the
    polygon, so a thing beside the zone, whose points fall in an edge cell, does not block it (the corridor's rule)."""
    out = []
    for u, v in zip(*(a.tolist() for a in _box(zone.polygon, 1)), strict=True):
        state = states.get((u, v), ("", 0))[0]
        if state != "occupied":
            continue
        subs = np.array(list(grid.sub_cells_of((u, v), state)), dtype=float).reshape(-1, 2)
        if _inside(zone.polygon, (subs[:, 0] + 0.5) * grid.SUB_M, (subs[:, 1] + 0.5) * grid.SUB_M).any():
            out.append((u, v))
    return out


def zone_state(states: Mapping[Cell, tuple[str, int]], zone: Zone) -> tuple[str, list[int]]:
    """`states` is `grid.latest`'s output over cells observed with the zone's floor and band. "blocked" when an
    occupied cell holds an obstacle sub-cell whose centre lies in the polygon; else "clear" when at least `ZONE_SHARE`
    of the zone's cells are `free`; else "unknown" (a zone with no cell centre is never clear). The evidence is the
    sorted distinct keyframes of the blocking cells (blocked), or of every observed zone cell (else)."""
    if blocking := _blocking(states, zone):
        return "blocked", sorted({states[c][1] for c in blocking})
    cells = zone_cells(zone)
    free = sum(states.get(c, ("", 0))[0] == "free" for c in cells)
    seen = sorted({states[c][1] for c in cells if c in states})
    return ("clear" if cells and free >= ZONE_SHARE * len(cells) - grid.EPS else "unknown"), seen


def coverage(states: Mapping[Cell, tuple[str, int]], zone: Zone) -> float:
    """The share of the zone's cells observed in any state (0 for a zone with no cell)."""
    cells = zone_cells(zone)
    return sum(c in states for c in cells) / len(cells) if cells else 0.0


def in_view(states: Mapping[Cell, tuple[str, int]], zone: Zone) -> bool:
    """At least `ZONE_SHARE` of the zone's cells were observed, in any state: the zone was in view. `zone_clear` says
    "right now" exactly then, whatever the zone held, so the wording never predicts the answer."""
    return coverage(states, zone) >= ZONE_SHARE - grid.EPS
