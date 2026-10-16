# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's Phase 1 dressing: what a real block has on its street that
the map does not say, as `Dressing` boxes.

Plain Python (no `bpy`). `layout(spec)` takes the downtown map's spec and returns every dressing box of the map, in a
fixed order; its random choices are seeded by the map's name. Nothing here is an entity or a receptacle, and nothing
stands where a vehicle, a mover or a question's object can be:
- **paint**, flush with the road (no more than 2 cm), in the carriageways and off every parking and loading surface:
  the double yellow centre lines, white lane dashes, zebra crosswalks across the side streets' mouths and beside the
  square's, and a stop bar before each (`street_marks.py` decides where);
- **iron**, flush: manhole covers in the travel lanes and storm drains in the gutter beside the kerb;
- **the kerb strip**, the first metre of each sidewalk from the kerb (the strip `scenery_layout` reserves, which no
  surface covers and where its lamp posts, signal poles and hydrants stand): street trees in a cast-iron pit (a
  trunk to 3.4 m, then a crown above it, over the sidewalk only: no tracked thing is taller than 3.3 m, so the crown
  hides nothing a question can ask about from above 3.4 m), benches facing the building line, planters and bike racks.
They are stationed one after another along each strip in a seeded order, clear of the posts.
- **the square**, over its paving's footprint (no receptacle volume: a pad's, up to 2.5 m): along each of its edges that
  a building wall closes, trees in pits every 6 m with a bench between each pair, backs to the wall and facing the
  square, and a planter at each corner. The edge that opens onto the street, and the pads, stay clear. A recipe is
`proc:<builder>[:<text>]` (`dressing_build_downtown.BUILDERS`); the text of the paint is its style and axis. A piece's
size is (depth, width, height) as in `dressing_spec`; a bench faces the building, the rest face the kerb.

Nothing that stands is left where the drone hovers or lifts off (`delivery_keepout`, lane M3-RY): a tree (stem and
crown together), bench, planter or rack that meets the column of a drop zone or a nest, or a nest's yard, is not drawn.
The stationing runs as if it were, and the unit is dropped after, so every other piece stays where it was. A nest's
yard is the bare ground round it, which the pieces of paving round it leave: it gets a white line (`lane_marking`, style
`bar`, flush) inside each edge that a piece of paving makes, so it reads as a paved bay and not as a gap in the paving.
"""

from __future__ import annotations

import hashlib
import math
import random

import delivery_keepout
import scenery_layout
import street_marks
from dressing_spec import Dressing
from street_marks import Mark

from stbench.geometry import Rect, to_local, to_world
from stbench.mapspec import parse_map

GROUND_KINDS = ("sidewalk", "plaza")  # the raised ground pieces the strip's and the square's pieces stand on
WALK_Z = 0.15  # the top of a sidewalk: what the kerb strip's pieces stand on
PAINT_H = 0.012  # paint is a plate this thick: flush, under the 2 cm the guard allows
IRON_H = 0.014
STRIP_X = 1.5  # the kerb strip's middle, in a sidewalk's own frame (x towards the kerb): its metre is 1.0..2.0
CROWN_X = STRIP_X + scenery_layout.TREE_CROWN_SHIFT_M  # the crown's middle: over the sidewalk, not past the kerb (2.0)
TRUNK = scenery_layout.TREE_TRUNK  # the pit and stem; the crown (`_crown`) starts above
BENCH, PLANTER, RACK = (0.55, 1.8, 0.85), (0.7, 1.4, 0.95), (0.6, 1.9, 0.82)
MANHOLE, DRAIN = (0.7, 0.7, IRON_H), (0.5, 0.9, IRON_H)  # a drain faces the kerb: its long side runs along it
HALF_ALONG = {"street_tree": 1.7, "bench": 1.3, "planter": 1.0, "bike_rack": 1.2}  # room an item takes along the strip
WEIGHTS = (("street_tree", 0.5), ("bench", 0.15), ("planter", 0.2), ("bike_rack", 0.15))
POST_CLEAR_M = {"lamp": 1.0, "signal": 1.8, "hydrant": 1.1}  # an item keeps this far from a post's centre, and its half
STRIP_END_M = 4.0  # none within this of the end of a strip (the signal poles stand there)
GAP_M = (4.0, 9.0)  # between one item's room and the next's
MANHOLE_PITCH_M, DRAIN_PITCH_M = 27.0, 22.0
Rect4 = tuple[float, float, float, float]


def layout(spec: dict) -> list[Dressing]:
    """Every Phase 1 dressing box of the downtown map. Deterministic: the same spec gives the same list. Raises
    ValueError for a map of another catalog, or when a piece of paint would land on a parking or loading surface or on
    the lane or zone that holds it (a flat plate over an entity would hide its pixels from the index pass)."""
    if spec.get("catalog") != "downtown":
        raise ValueError(f"map {spec.get('name')!r} is not a downtown map (catalog {spec.get('catalog')!r})")
    rng = random.Random(int.from_bytes(hashlib.sha256(spec["name"].encode()).digest()[:8], "big"))
    surfaces = _surfaces(spec)  # the receptacles; with the footprints of the lanes and zones that carry them (entities)
    surfaces += [street_marks.footprint(f) for f in spec["furniture"]
                 if f["surfaces"] and f["kind"] not in GROUND_KINDS]
    keep = keep_out(spec)
    marks = street_marks.marks(spec) + _yard_marks(keep)
    out = _paint(marks, surfaces)
    out += _iron(spec, marks, surfaces, rng)
    out += _kerb_strips(spec, rng, keep)
    out += _squares(spec, surfaces, rng, keep)
    return out


# -- what the drone's pieces keep clear -------------------------------------------------------------------------------
YARD_LINE_M, YARD_INSET_M = 0.15, 0.1  # a yard's boundary line: its width, and its distance from the paving's face


def _walls(spec: dict) -> list[Rect4]:
    """The slab of every wall piece, as (xmin, xmax, ymin, ymax)."""
    out = []
    for w in spec["walls"]:
        (ax, ay), (bx, by), h = w["a"], w["b"], w["thickness"] / 2
        out.append((min(ax, bx) - h, max(ax, bx) + h, min(ay, by) - h, max(ay, by) + h))
    return out


def keep_out(spec: dict) -> delivery_keepout.KeepOut:
    """The columns of the map's drop zones and nests and the yard of each nest (`delivery_keepout`), the yard bounded by
    the raised ground pieces and the walls."""
    pieces = [(f["kind"], tuple(f["center"]), tuple(f["size"])) for f in spec["furniture"]]
    grounds = [street_marks.footprint(f) for f in spec["furniture"] if f["kind"] in GROUND_KINDS]
    return delivery_keepout.keep_out(pieces, grounds, _walls(spec))


def _yard_marks(keep: delivery_keepout.KeepOut) -> list[Mark]:
    """A white line `YARD_LINE_M` wide, `YARD_INSET_M` in from each edge of a yard that a piece of paving makes (not a
    wall): the lines along y span the yard's width, the lines along x the length between them, so no two overlap."""
    out: list[Mark] = []
    for y in keep.yards:
        x0, x1, y0, y1 = y.rect
        w, i, sides = YARD_LINE_M, YARD_INSET_M, y.ground_sides
        ex0, ex1 = x0 + (i if "-x" in sides else 0.0), x1 - (i if "+x" in sides else 0.0)
        ey0, ey1 = y0 + (i + w if "-y" in sides else 0.0), y1 - (i + w if "+y" in sides else 0.0)
        out += [Mark("lane_marking", "bar", r, "x") for s, r in (("+y", (ex0, ex1, y1 - i - w, y1 - i)),
                                                                   ("-y", (ex0, ex1, y0 + i, y0 + i + w)))
                if s in sides]
        out += [Mark("lane_marking", "bar", r, "y") for s, r in (("+x", (x1 - i - w, x1 - i, ey0, ey1)),
                                                                   ("-x", (x0 + i, x0 + i + w, ey0, ey1)))
                if s in sides]
    return out


# -- the map's receptacles ------------------------------------------------------------------------------------------
def _aabb(rect: Rect) -> Rect4:
    hx, hy = (rect.sx / 2, rect.sy / 2) if abs(math.cos(rect.yaw)) > 0.5 else (rect.sy / 2, rect.sx / 2)
    return rect.cx - hx, rect.cx + hx, rect.cy - hy, rect.cy + hy


def _surfaces(spec: dict) -> list[Rect4]:
    """The world rectangle of every receptacle surface of the map's furniture (the pieces are quarter turns)."""
    out = []
    for f in spec["furniture"]:
        frame = Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"])
        for s in f["surfaces"]:
            cx, cy = to_world(frame, s["rect"][0], s["rect"][1])
            out.append(_aabb(Rect(cx, cy, s["rect"][2], s["rect"][3], f["yaw"])))
    return out


def _meets(a: Rect4, others: list[Rect4], margin: float = 0.0) -> bool:
    return any(a[0] < b[1] + margin and b[0] < a[1] + margin and a[2] < b[3] + margin and b[2] < a[3] + margin
               for b in others)


# -- paint -----------------------------------------------------------------------------------------------------------
def _paint(marks: list[Mark], surfaces: list[Rect4]) -> list[Dressing]:
    out = []
    for m in marks:
        if _meets(m.rect, surfaces):
            raise ValueError(f"a {m.kind} at {tuple(round(v, 2) for v in m.rect)} lies on a receptacle surface")
        x0, x1, y0, y1 = m.rect
        out.append(Dressing(m.kind, f"proc:paint:{m.style}:{m.along}", ((x0 + x1) / 2, (y0 + y1) / 2, 0.0), 0.0,
                            (x1 - x0, y1 - y0, PAINT_H)))
    return out


# -- iron: manholes and drains ----------------------------------------------------------------------------------------
def _iron(spec: dict, marks: list[Mark], surfaces: list[Rect4], rng: random.Random) -> list[Dressing]:
    """Manholes along the travel lanes (a lane's middle, a little off it) and drains in the gutter beside each kerb,
    wherever no mark, receptacle surface or earlier piece is within a metre."""
    taken = [m.rect for m in marks]
    out: list[Dressing] = []

    def put(cls: str, recipe: str, cx: float, cy: float, size, yaw: float) -> bool:
        r = Rect(cx, cy, size[0], size[1], yaw)
        box = _aabb(r)
        if _meets(box, surfaces, 0.3) or _meets(box, taken, 1.0):
            return False
        taken.append(box)
        out.append(Dressing(cls, recipe, (cx, cy, 0.0), math.degrees(yaw), size))
        return True

    for s in street_marks.streets(spec):
        n = s.lanes
        width = (s.c_hi - s.c_lo - 2 * s.park) / n
        t = s.lo + 14.0 + rng.uniform(0.0, 10.0)
        while t < s.hi - 8.0:
            for attempt in range(6):
                c = s.c_lo + s.park + width * (rng.randrange(n) + 0.5) + rng.uniform(-0.6, 0.6)
                x, y = (t + attempt * 2.5, c) if s.axis == "x" else (c, t + attempt * 2.5)
                if put("manhole_cover", "proc:manhole", x, y, MANHOLE, 0.0):
                    break
            t += MANHOLE_PITCH_M + rng.uniform(-8.0, 8.0)
    for f in spec["furniture"]:
        if f["kind"] != "sidewalk":
            continue
        frame = Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"])
        half, length = f["size"][0] / 2, f["size"][1]
        t = -length / 2 + 6.0 + rng.uniform(0.0, DRAIN_PITCH_M / 2)
        while t < length / 2 - 4.0:
            for attempt in range(8):  # the gutter inlet's long side lies along the kerb, 0.3 m out from its face
                cx, cy = to_world(frame, half + 0.3, t + attempt * 1.5)
                if put("storm_drain", "proc:storm_drain", cx, cy, DRAIN, f["yaw"]):
                    break
            t += DRAIN_PITCH_M + rng.uniform(-6.0, 6.0)
    return out


# -- the kerb strip ------------------------------------------------------------------------------------------------
def _kerb_strips(spec: dict, rng: random.Random, keep: delivery_keepout.KeepOut) -> list[Dressing]:
    """Street trees, benches, planters and bike racks stationed along each sidewalk's kerb strip."""
    lay = scenery_layout.layout(parse_map(spec))
    out: list[Dressing] = []
    for f in spec["furniture"]:
        if f["kind"] != "sidewalk":
            continue
        frame = Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"])
        length = f["size"][1]
        blocked = []  # (centre along, half room) of every post that stands in this strip
        for p in lay.posts:
            lx, ly = to_local(frame, p.x, p.y)
            if abs(lx - STRIP_X) <= 0.6 and abs(ly) <= length / 2:
                blocked.append((ly, POST_CLEAR_M[p.kind]))
        t = -length / 2 + STRIP_END_M + rng.uniform(0.0, 4.0)
        while True:
            kind = _pick(rng)
            half = HALF_ALONG[kind]
            t += half
            for _ in range(8):  # push the item past any post it would stand against
                hit = [(c, r) for c, r in blocked if abs(c - t) < half + r]
                if not hit:
                    break
                t = max(c + r for c, r in hit) + half
            if t + half > length / 2 - STRIP_END_M:
                break
            out += keep.clear(_item(kind, frame, f["yaw"], t, rng))
            t += half + rng.uniform(*GAP_M)
    return out


# -- the square ------------------------------------------------------------------------------------------------------
TREE_PITCH_SQ_M, TREE_INSET_M, BENCH_INSET_M, PLANTER_INSET_M = 6.0, 2.3, 0.9, 1.0
PAD_CLEAR_M = 1.0  # a square's item keeps this far from any pad (receptacle surface)


def _walled(edge: Rect4, walls: list[Rect4]) -> bool:
    """Whether a building wall lies against this edge of a footprint (a thin rectangle: the edge itself)."""
    return any(edge[0] < w[1] + 0.05 and w[0] < edge[1] + 0.05 and edge[2] < w[3] + 0.05 and w[2] < edge[3] + 0.05
               and min(edge[1], w[1]) - max(edge[0], w[0]) + min(edge[3], w[3]) - max(edge[2], w[2]) > 1.0
               for w in walls)


def _crown(x: float, y: float, yaw: float, rng: random.Random) -> Dressing:
    """The crown of the tree whose stem stands at (x, y): a size within a tenth or so of
    `scenery_layout.TREE_CROWN` and one of six shapes (the recipe's text), so that no two trees are alike; it is
    centred on (x, y)."""
    size = (round(rng.uniform(1.8, 2.1), 2), round(rng.uniform(2.3, 3.0), 2), round(rng.uniform(2.9, 3.5), 2))
    return Dressing("tree_crown", f"proc:tree_crown:{rng.randrange(6)}", (x, y, WALK_Z + TRUNK[2]), yaw, size)


def _squares(spec: dict, surfaces: list[Rect4], rng: random.Random, keep: delivery_keepout.KeepOut
             ) -> list[Dressing]:
    """Trees, benches and planters along the walled edges of each square (a `plaza` piece)."""
    walls = _walls(spec)
    out: list[Dressing] = []
    for f in spec["furniture"]:
        if f["kind"] != "plaza":
            continue
        x0, x1, y0, y1 = street_marks.footprint(f)
        # each edge as a thin rectangle, the direction into the square, and the span along the edge
        for edge, (dx, dy), lo, hi in (((x0, x1, y1, y1), (0.0, -1.0), x0, x1), ((x0, x1, y0, y0), (0.0, 1.0), x0, x1),
                                       ((x1, x1, y0, y1), (-1.0, 0.0), y0, y1), ((x0, x0, y0, y1), (1.0, 0.0), y0, y1)):
            if not _walled(edge, walls):
                continue
            yaw = math.degrees(math.atan2(dy, dx))

            def at(along: float, inset: float, size, cls: str, recipe: str, dx=dx, dy=dy, edge=edge, yaw=yaw):
                """The piece `inset` in from the edge at `along` it (x for a horizontal edge, y for a vertical one),
                facing into the square; None where it would stand within a metre of a pad."""
                x, y = (along, edge[2] + dy * inset) if dy else (edge[0] + dx * inset, along)
                box = _aabb(Rect(x, y, size[0], size[1], math.radians(yaw)))
                return None if _meets(box, surfaces, PAD_CLEAR_M) else Dressing(cls, recipe, (x, y, WALK_Z), yaw, size)

            a = lo + 3.5
            while a <= hi - 3.5 + 1e-9:
                if (tree := at(a, TREE_INSET_M, TRUNK, "street_tree", "proc:street_tree")) is not None:
                    px, py, _ = tree.pos
                    out += keep.clear([tree, _crown(px, py, yaw, rng)])
                bench_at = a + TREE_PITCH_SQ_M / 2
                bench = at(bench_at, BENCH_INSET_M, BENCH, "bench", "proc:bench") if bench_at <= hi - 3.5 else None
                if bench is not None:
                    out += keep.clear([bench])
                a += TREE_PITCH_SQ_M
            if dy:  # a planter at each end of a long edge (the short edges' ends are these corners too)
                out += [d for p in (at(lo + 1.2, PLANTER_INSET_M, PLANTER, "planter", "proc:planter"),
                                    at(hi - 1.2, PLANTER_INSET_M, PLANTER, "planter", "proc:planter"))
                        if p is not None for d in keep.clear([p])]
    return out


def _pick(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for kind, w in WEIGHTS:
        acc += w
        if x < acc:
            return kind
    return WEIGHTS[-1][0]


def _item(kind: str, frame: Rect, yaw: float, t: float, rng: random.Random) -> list[Dressing]:
    """The dressing of one item at `t` along a sidewalk (its own frame: x to the kerb, y along): a tree is its trunk and
    its crown, the rest one piece."""
    deg = math.degrees(yaw)

    def at(x: float, size, cls: str, recipe: str, z: float = WALK_Z, turn: float = 0.0) -> Dressing:
        px, py = to_world(frame, x, t)
        return Dressing(cls, recipe, (px, py, z), deg + turn, size)

    if kind == "street_tree":
        crown = _crown(*to_world(frame, CROWN_X, t), deg, rng)
        return [at(STRIP_X, TRUNK, "street_tree", "proc:street_tree"), crown]
    size = {"bench": BENCH, "planter": PLANTER, "bike_rack": RACK}[kind]
    return [at(STRIP_X, size, kind, f"proc:{kind}", turn=180.0 if kind == "bench" else 0.0)]
