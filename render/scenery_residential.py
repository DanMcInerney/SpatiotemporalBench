# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The suburb beyond an outdoor residential map's edge: the street carried on past its open ends, and rows of houses,
a row of taller blocks and trees beyond its sides, out to `scenery_layout.REACH_M`, all on lawn (the ground plane under
and round the map is the look's `GROUND`). Nothing in it is an entity (`pass_index` 0) and nothing is inside the map's
bounds.

`plan(m)` is plain Python (no `bpy`, so the layout is tested without Blender); `build(m, ground, mats)` draws it and
imports `bpy` and the draw modules only when it runs. `build_scene` calls `scenery_<SCENERY>.build` with the plane's
material and the facade materials (neither is used: the street is asphalt of its own and nothing has a facade shader)
and reads `scenery.layout.REACH_M` of any scenery module, so `layout` below is `scenery_layout`, the downtown's.
- **The street's ends**: each open edge of a `street` region (`scenery_layout.open_edges`; the plots' back edges are
  open too, and are no street) continues as a carriageway (the width the map's sidewalks and verges leave bare), the
  map's sidewalks carried on at their width and kerb side, and two rows of houses on each side of it, the first
  `SETBACK_M` from the street.
- **The sides**: every side of the bounds with no street end has rows of houses at `ROW_START_M` and a row of blocks at
  `BLOCK_ROW_M`, so the horizon is a suburb out to a quarter of a kilometre, not a field.
- **Low detail**: a house is `procedural_house.house` without its windows, door, plinth and gutters, four distinct ones
  (sizes and colours) instanced; blocks the same, two; trees are one of four, a little larger or smaller.
A building stands only where it meets nothing placed before it (street ends first), the road or the bounds. Seeded by
`SEED` alone, so a scene is the same whichever episode it is built for.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import scenery_layout as layout
from scenery_layout import Frame, Walk

from stbench.geometry import Point, Rect

SEED = 47
TOL = 0.05
GAP_M = 14.0  # along a continued street: the first lot's distance from the map's end (the plots end there)
SETBACK_M = 9.0  # a street-end house's front, from the street region's edge
LOT_M, BLOCK_LOT_M = (12.0, 19.0), (30.0, 45.0)  # a lot's width along its row, for a house and for a block
ROW_START_M = (26.0, 62.0, 100.0, 205.0)  # rows of houses beyond a side, their front lines' distance from its edge
BLOCK_ROW_M = 150.0  # and the row of blocks
STREET_ROW_M = (SETBACK_M, SETBACK_M + 40.0)  # rows of houses beside a continued street
HOUSES = ((9.0, 8.0, 8.4, "beige"), (10.0, 9.0, 8.8, "white"), (8.5, 8.0, 8.2, "red"), (11.0, 9.5, 8.9, "grey"))
BLOCKS = ((14.0, 36.0, 15.9, "red"), (13.0, 28.0, 12.9, "beige"))  # (depth, width, height, colour)
TABLE = {"house": HOUSES, "block": BLOCKS}
TREE_VARIANTS, TREE_P, TREE_CLEAR_M = 4, 0.75, 4.0  # distinct trees; a lot's chance of one; the box that keeps it clear
CURB_H = 0.15  # the sidewalk's height, the map's own


@dataclass(frozen=True)
class Site:
    """A building: its kind ("house" or "block"), which prototype of the kind (`variant`, an index of `TABLE[kind]`),
    its `centre`, `face` (the yaw its front looks along, a quarter turn) and `size` (depth, width, height as its
    builder takes it)."""

    kind: str
    variant: int
    centre: Point
    face: float
    size: tuple[float, float, float]

    @property
    def rect(self) -> Rect:
        """The footprint as an axis-aligned rectangle."""
        turned = abs(math.sin(self.face)) > 0.5
        sx, sy = (self.size[1], self.size[0]) if turned else (self.size[0], self.size[1])
        return Rect(self.centre[0], self.centre[1], sx, sy, 0.0)


@dataclass(frozen=True)
class Tree:
    x: float
    y: float
    variant: int
    scale: float


@dataclass
class Plan:
    roads: list[Rect] = field(default_factory=list)
    walks: list[Walk] = field(default_factory=list)
    sites: list[Site] = field(default_factory=list)
    trees: list[Tree] = field(default_factory=list)
    arms: list[Frame] = field(default_factory=list)


def _street_arms(m) -> list[Frame]:
    """The open region edges that are a street's end: the region just inside the middle of the edge is a `street`."""
    out = []
    for arm in layout.open_edges(m):
        region = m.region_of(arm.pt(-0.1, arm.width / 2))
        if region is not None and m.region(region).kind == "street":
            out.append(arm)
    return out


def _runs_out(arm: Frame, rect: Rect) -> tuple[float, float] | None:
    """The stretch across the arm that a piece covers, when it reaches the arm's edge from inside (else None)."""
    s0, s1, c0, c1 = arm.span(rect)
    return (c0, c1) if abs(s1) < TOL and c0 >= -TOL and c1 <= arm.width + TOL else None


def _carriageway(m, arm: Frame) -> list[tuple[float, float]]:
    """The stretches of the street's width, across, that no sidewalk or verge reaching the edge covers."""
    covered = sorted(c for f in m.furniture if f.kind in ("sidewalk", "verge") and layout.axis_aligned(f.yaw)
                     if (c := _runs_out(arm, f.footprint())) is not None)
    out, at = [], 0.0
    for a, b in covered:
        if a - at > 1.0:
            out.append((at, a))
        at = max(at, b)
    return out + ([(at, arm.width)] if arm.width - at > 1.0 else [])


def _sides(bounds) -> list[Frame]:
    """The four sides of the bounds as frames (`n` outward, `c` along the side)."""
    x0, y0, x1, y1 = bounds
    return [Frame((x0, y0), (1.0, 0.0), (0.0, -1.0), x1 - x0), Frame((x0, y1), (1.0, 0.0), (0.0, 1.0), x1 - x0),
            Frame((x0, y0), (0.0, 1.0), (-1.0, 0.0), y1 - y0), Frame((x1, y0), (0.0, 1.0), (1.0, 0.0), y1 - y0)]


def _meets(a: Rect, b: Rect, pad: float) -> bool:
    ax0, ax1, ay0, ay1 = layout.aabb(a)
    bx0, bx1, by0, by1 = layout.aabb(b)
    return ax0 < bx1 + pad and bx0 < ax1 + pad and ay0 < by1 + pad and by0 < ay1 + pad


class _Lots:
    """The rows being laid out: what is placed so far, the bounds nothing may enter and the draws."""

    def __init__(self, plan: Plan, bounds, rng: random.Random):
        x0, y0, x1, y1 = bounds
        self.plan, self.rng = plan, rng
        self.keep_out = Rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0 + 4.0, y1 - y0 + 4.0, 0.0)

    def free(self, rect: Rect, pad: float = 3.0) -> bool:
        """Whether a footprint is outside the bounds (and 2 m more) and meets no building, road or sidewalk."""
        return not (_meets(rect, self.keep_out, 0.0) or any(_meets(rect, s.rect, pad) for s in self.plan.sites)
                    or any(_meets(rect, r, 1.0) for r in [*self.plan.roads, *(w.rect for w in self.plan.walks)]))

    def on_tree(self, rect: Rect) -> bool:
        return any(_meets(rect, Rect(t.x, t.y, TREE_CLEAR_M, TREE_CLEAR_M, 0.0), 0.0) for t in self.plan.trees)

    def row(self, frame: Frame, s_front: float, c0: float, c1: float, toward: float, kind: str = "house") -> None:
        """Lots along `frame` from `c0` to `c1` across, whose front line is `s_front` out of the edge; each building's
        front looks along `toward` (a yaw). A building stands in each lot, centred across, with its back to the lot's
        far end, and a tree (a house's lot) in front of it."""
        wide = BLOCK_LOT_M if kind == "block" else LOT_M
        c = c0
        while c < c1:
            lot = self.rng.uniform(*wide)
            variant = self.rng.randrange(len(TABLE[kind]))
            depth, width, height, _ = TABLE[kind][variant]
            site = Site(kind, variant, (0.0, 0.0), toward, (depth, width, height))
            x0, x1, y0, y1 = layout.aabb(site.rect)
            along_c = abs(frame.t[0]) * (x1 - x0) + abs(frame.t[1]) * (y1 - y0)
            along_s = abs(frame.n[0]) * (x1 - x0) + abs(frame.n[1]) * (y1 - y0)
            jitter = self.rng.uniform(0.0, 3.0)
            if min(lot, c1 - c) >= along_c + 2.0:
                centre = frame.pt(s_front + along_s / 2 + jitter, c + min(lot, c1 - c) / 2)
                site = Site(kind, variant, centre, toward, (depth, width, height))
                if self.free(site.rect) and not self.on_tree(site.rect):
                    self.plan.sites.append(site)
                    if kind == "house" and self.rng.random() < TREE_P:
                        self.tree(frame, s_front - 3.5, c + self.rng.uniform(2.0, lot - 2.0))
            c += lot

    def tree(self, frame: Frame, s: float, c: float) -> None:
        x, y = frame.pt(max(s, 2.0), c)
        scale = self.rng.uniform(0.8, 1.3)
        variant = self.rng.randrange(TREE_VARIANTS)
        if self.free(Rect(x, y, TREE_CLEAR_M, TREE_CLEAR_M, 0.0), 1.0):
            self.plan.trees.append(Tree(x, y, variant, scale))


def plan(m, seed: int = SEED, reach: float = layout.REACH_M) -> Plan:
    """The scenery of an outdoor map (an empty plan for an indoor one)."""
    out, rng = Plan(), random.Random(seed)
    if m.outdoor is None:
        return out
    out.arms = _street_arms(m)
    walks = [Walk(f.footprint(), f.yaw) for f in m.furniture if f.kind == "sidewalk" and layout.axis_aligned(f.yaw)]
    for arm in out.arms:
        out.roads += [arm.rect(0.0, reach, a, b) for a, b in _carriageway(m, arm)]
        out.walks += [Walk(arm.rect(0.0, reach, *c), wk.front) for wk in walks if (c := _runs_out(arm, wk.rect))]
    lots = _Lots(out, m.bounds, rng)
    for arm in out.arms:  # the street is on the +t side of the lots at c < 0 and on the -t side of those at c > width
        t = arm.t
        for edge, away, toward in ((0.0, (-t[0], -t[1]), (t[0], t[1])), (arm.width, t, (-t[0], -t[1]))):
            side = Frame(arm.pt(0.0, edge), arm.n, away, reach)  # `s` out of the street, `c` along it
            for k, front in enumerate(STREET_ROW_M):
                yaw = math.atan2(toward[1], toward[0]) + (0.0 if k == 0 else math.pi)
                lots.row(side, front, GAP_M, reach, yaw)
    for side in _sides(m.bounds):
        if any(arm.n == side.n for arm in out.arms):
            continue
        for k, start in enumerate(ROW_START_M):  # fronts alternate, back to back across the gardens
            sign = -1.0 if k % 2 == 0 else 1.0
            lots.row(side, start, -reach, side.width + reach, math.atan2(sign * side.n[1], sign * side.n[0]))
        lots.row(side, BLOCK_ROW_M, -reach, side.width + reach, math.atan2(-side.n[1], -side.n[0]), "block")
    return out


# -- drawing ---------------------------------------------------------------------------------------------------------
def build(m, ground, mats) -> Plan:
    """Draw the scenery of an outdoor map; returns the plan it drew."""
    lay = plan(m)
    _ground(lay)
    _buildings(lay)
    _trees(lay)
    return lay


def _ground(lay: Plan) -> None:
    """The carriageways and the sidewalks (with their kerbs) carried on past the map's street ends, one mesh."""
    if not (lay.roads or lay.walks):
        return
    import bpy
    import materials_street
    from scene_mesh import MeshBuilder, cube, rect

    mb = MeshBuilder("scenery_ground")
    road = mb.slot("asphalt", bpy.data.materials.get("shader:asphalt") or materials_street.asphalt())
    top, side = mb.slot("sidewalk", materials_street.sidewalk()), mb.slot("kerb", materials_street.kerb())
    for r in lay.roads:
        x0, x1, y0, y1 = layout.aabb(r)
        rect(mb.bm, x0, x1, y0, y1, 0.0, road)
    for w in lay.walks:
        x0, x1, y0, y1 = layout.aabb(w.rect)
        for f in cube(mb.bm, x0, x1, y0, y1, 0.0, CURB_H, side):
            if f.calc_center_median().z > CURB_H - 1e-3:
                f.material_index = top
    mb.finish()


def _buildings(lay: Plan) -> None:
    """Each building an object of one of the prototypes' meshes (made once each), at its centre, turned to its face."""
    import bpy
    import procedural_house
    import shapes
    from materials_residential import Surfaces

    resolve, protos = Surfaces().resolver(), {}
    for i, s in enumerate(lay.sites):
        if (s.kind, s.variant) not in protos:
            make = procedural_house.house if s.kind == "house" else procedural_house.block
            parts = make(s.size, TABLE[s.kind][s.variant][3], detailed=False)
            protos[s.kind, s.variant] = shapes.entity_mesh(f"scenery_proto_{s.kind}{s.variant}", parts, resolve)
        ob = bpy.data.objects.new(f"scenery_{s.kind}_{i}", protos[s.kind, s.variant])
        bpy.context.scene.collection.objects.link(ob)
        ob.location, ob.rotation_euler = (s.centre[0], s.centre[1], 0.0), (0.0, 0.0, s.face)


def _tree_prototype(k: int, bark, leaf):
    """A tree of the scenery: a trunk and a crown of leaf lumps over it, 8 m high; `k` seeds the crown's shape."""
    import street_tree
    from mathutils import Matrix
    from meshkit import lathe, placed
    from scene_mesh import MeshBuilder

    mb = MeshBuilder(f"scenery_proto_tree{k}")
    b, lf = mb.slot("bark", bark), mb.slot("leaf", leaf)
    lathe(mb.bm, [(0.0, 0.0), (0.22, 0.0), (0.12, 1.0), (0.09, 3.4), (0.0, 3.4)], b, segments=10)
    placed(mb.bm, Matrix.Translation((0.0, 0.0, 3.2)),
           lambda: street_tree.crown(mb.bm, (4.4, 4.4, 5.2), b, lf, random.Random(2000 + k), snug=False))
    return mb.mesh()


def _trees(lay: Plan) -> None:
    if not lay.trees:
        return
    import bpy
    import materials_street

    bark = bpy.data.materials.get("bark") or materials_street.bark()
    leaf = bpy.data.materials.get("foliage") or materials_street.foliage()
    protos = [_tree_prototype(k, bark, leaf) for k in range(TREE_VARIANTS)]
    for i, t in enumerate(lay.trees):
        ob = bpy.data.objects.new(f"scenery_tree_{i}", protos[t.variant])
        bpy.context.scene.collection.objects.link(ob)
        ob.location, ob.scale = (t.x, t.y, 0.0), (t.scale,) * 3
