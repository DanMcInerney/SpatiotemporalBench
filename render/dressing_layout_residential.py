# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The residential Phase 1 dressing (M4-suburb §4): only what a drone sees from 7 to 15 m, as `Dressing` boxes.

Plain Python (no `bpy`). `layout(spec)` takes a residential map's spec and returns every dressing box of the map, in a
fixed order; its random choices are seeded by the map's name. Nothing here is an entity or a receptacle. Five classes
(`CLASSES`, which the registry `scenery.DRESSING` lists for the catalog) and nothing else: a wheelie bin would read as a
dumpster, and mailboxes, house numbers, garden furniture, benches and bike racks are invisible from 7 m up.
- **street trees and lamps on verges** (`verge` pieces, blocking, 0.15 m): along each verge's length, a tree every
  `TREE_PITCH_M` and a lamp post between them, beside the kerb, the lamp's arm reaching towards the street;
- **shrubs and a few trees on garden beds** (`garden_bed`, blocking, 0.3 m): shrubs along a bed, a tree in some;
- **a playground on a green**: a bed whose centre lies in a `green` region of 30 m2 or more carries swings, a slide and
  a climbing frame in a row or two.

A tree is two pieces, a stem (`tree_stem`: a trunk and its first limbs, `STEM_W_M` across, from the piece it stands on
up to `STRUCTURE_FROM_M`) and a canopy (`tree_canopy`) from there up: the guard allows dressing only over a blocking
footprint, and a crown wider than a verge or a bed is over open ground, where only the structure band (from
`STRUCTURE_FROM_M` to the walls' height) is allowed. So a map's walls must stand more than `CANOPY_H_M[0]` above it.
**The drone's lowering rule:** at the lowering point everything within 3.65 m of the drone's centre must stand below its
altitude less 3 m, so no piece of 3.9 m or more (stem, canopy, lamp) stands within `CLEAR_M` of a drop zone's centre or
the nest's. Canopies keep off houses, blocks and one another.
**What a map needs for every class to appear** (`tests/test_dressing.py` asks it of each bundled map): a `verge` at
least 0.6 m across and a `garden_bed` in a `garden` (the stems stand on them, 0.5 m square, each with 4 m to spare from
a drop zone and the nest and room for a canopy 3.4 to 5.0 m square clear of houses and blocks, inside the bounds), a
`garden_bed` of 30 m2 or more whose centre lies in a `green` region (the playground), and walls 8.2 m high or more.

Deterministic: the same spec gives the same list. A recipe is `proc:<builder>[:<variant>]`
(`dressing_build_residential.BUILDERS`); a piece's size is (depth, width, height) as in `dressing_spec`; a lamp faces
the street, the rest face +x.
"""

from __future__ import annotations

import hashlib
import math
import random

from dressing_spec import Dressing

from stbench.dressing_zones import STRUCTURE_FROM_M

CLASSES = (  # (name, note): the registry's rows for this catalog, in the order the layout emits them
    ("tree_stem", "the trunk and first limbs of a street or garden tree, up to 5 m, on a verge or a garden bed"),
    ("tree_canopy", "a tree's leaf mass from 5 m up, over its stem; never over a house, a block or a drop zone"),
    ("lamp_post", "a street lamp on a verge, 5 m, its arm towards the carriageway"),
    ("shrub", "a low clump of leaf (0.7 to 1.3 m) on a garden bed"),
    ("playground", "swings, a slide or a climbing frame on a bed in a green, up to 2.4 m"),
)
STEM_W_M = 0.5  # a stem's footprint
CANOPY_W_M, CANOPY_H_M = (3.4, 5.0), (3.2, 5.0)  # a canopy's width (both ways) and height
TREE_PITCH_M = (13.0, 19.0)  # along a verge, between one tree and the next
LAMP_W_M, LAMP_H_M, LAMP_Z = (0.9, 0.3), 5.0, 0.0  # a lamp post's box: across the verge, along it, and its height
END_M = 3.0  # none within this of the end of a verge
CLEAR_M = 4.0  # a tall piece's distance from a drop zone's or the nest's centre (the drone's 3.65 m and a margin)
TALL_M = 3.9  # a piece whose top is higher than this is subject to CLEAR_M
BED_TREE_P, SHRUB_PITCH_M = 0.4, (1.5, 2.3)
SHRUB_W_M, SHRUB_H_M = (0.8, 1.4), (0.7, 1.3)
SHRUB_VARIANTS, CANOPY_VARIANTS = 4, 6
PLAY_MIN_M2, PLAY_MARGIN_M, PLAY_GAP_M = 30.0, 0.3, 1.2
PLAY_SET = (("swing_set", (2.0, 3.6, 2.4)), ("slide", (1.1, 3.6, 1.9)), ("climbing_frame", (2.2, 2.2, 2.2)))
Box2 = tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


def _aabb(f: dict) -> Box2:
    """The footprint of a furniture piece of the spec (a quarter turn) as an axis-aligned box."""
    cx, cy = f["center"]
    sx, sy = f["size"][0] / 2, f["size"][1] / 2
    if abs(math.cos(f["yaw"])) < 0.5:
        sx, sy = sy, sx
    return cx - sx, cx + sx, cy - sy, cy + sy


def _meets(a: Box2, b: Box2, margin: float = 0.0) -> bool:
    return a[0] < b[1] + margin and b[0] < a[1] + margin and a[2] < b[3] + margin and b[2] < a[3] + margin


def _dist(p: tuple[float, float], b: Box2) -> float:
    """The distance from a point to a box (0 inside it)."""
    return math.hypot(max(b[0] - p[0], 0.0, p[0] - b[1]), max(b[2] - p[1], 0.0, p[1] - b[3]))


class _Site:
    """What a map has that a tall piece keeps away from, and the canopies placed so far."""

    def __init__(self, spec: dict):
        furniture = spec["furniture"]
        self.points = [tuple(f["center"]) for f in furniture if f["kind"] in ("nest", "drop_zone")]
        self.buildings = [_aabb(f) for f in furniture if f["kind"] in ("house", "block")]
        x0, y0, x1, y1 = spec["bounds"]
        self.bounds: Box2 = (x0 + 0.3, x1 - 0.3, y0 + 0.3, y1 - 0.3)
        self.canopies: list[Box2] = []
        self.ceiling = spec["wall_height"] + 0.02

    def clear(self, box: Box2) -> bool:
        """Whether a tall piece's footprint is `CLEAR_M` from every drop zone's centre and the nest's."""
        return all(_dist(p, box) >= CLEAR_M for p in self.points)

    def canopy_ok(self, box: Box2) -> bool:
        inside = self.bounds[0] <= box[0] and box[1] <= self.bounds[1] and self.bounds[2] <= box[2] \
            and box[3] <= self.bounds[3]
        return inside and self.clear(box) and not any(_meets(box, b, 0.3) for b in self.buildings) \
            and not any(_meets(box, c, 0.5) for c in self.canopies)


def layout(spec: dict) -> list[Dressing]:
    """Every Phase 1 dressing box of a residential map. Raises ValueError for a map of another catalog or one whose
    walls are too low to hold a canopy over its stem (`STRUCTURE_FROM_M` plus the lowest canopy)."""
    if spec.get("catalog") != "residential":
        raise ValueError(f"map {spec.get('name')!r} is not a residential map (catalog {spec.get('catalog')!r})")
    if spec["wall_height"] < STRUCTURE_FROM_M + CANOPY_H_M[0]:
        raise ValueError(f"wall_height {spec['wall_height']} m leaves no room for a canopy over {STRUCTURE_FROM_M} m")
    rng = random.Random(int.from_bytes(hashlib.sha256(spec["name"].encode()).digest()[:8], "big"))
    site, street = _Site(spec), [_aabb_region(r) for r in spec["regions"] if _kind(r) == "street"]
    greens = [_aabb_region(r) for r in spec["regions"] if _kind(r) == "green"]
    out: list[Dressing] = []
    for f in spec["furniture"]:
        if f["kind"] == "verge":
            out += _verge(f, street, site, rng)
    for f in spec["furniture"]:
        if f["kind"] == "garden_bed":
            box = _aabb(f)
            centre = ((box[0] + box[1]) / 2, (box[2] + box[3]) / 2)
            play = any(g[0] <= centre[0] <= g[1] and g[2] <= centre[1] <= g[3] for g in greens)
            out += _playground(f, box, rng) if play and _area(box) >= PLAY_MIN_M2 else _bed(f, box, site, rng)
    order = {c[0]: i for i, c in enumerate(CLASSES)}
    return sorted(out, key=lambda d: order[d.cls])  # stable: the classes in the registry's order, each in map order


def _kind(region: dict) -> str:
    return str(region.get("kind", region["name"]))


def _aabb_region(region: dict) -> Box2:
    xs, ys = [p[0] for p in region["polygon"]], [p[1] for p in region["polygon"]]
    return min(xs), max(xs), min(ys), max(ys)


def _area(b: Box2) -> float:
    return (b[1] - b[0]) * (b[3] - b[2])


def _tree(x: float, y: float, z: float, site: _Site, rng: random.Random, limit: Box2) -> list[Dressing]:
    """A stem at (x, y) standing on a piece whose top is `z` and a canopy over it, or nothing: when the canopy would
    meet a house, a canopy, a drop zone's or the nest's clearance, or leave the map, or the stem would."""
    w, d = (round(rng.uniform(*CANOPY_W_M), 2) for _ in range(2))
    h = round(min(rng.uniform(*CANOPY_H_M), site.ceiling - STRUCTURE_FROM_M), 2)
    stem = (x - STEM_W_M / 2, x + STEM_W_M / 2, y - STEM_W_M / 2, y + STEM_W_M / 2)
    canopy = (x - w / 2, x + w / 2, y - d / 2, y + d / 2)
    variant = rng.randrange(CANOPY_VARIANTS)  # drawn whether or not the tree stands, so one refusal moves nothing else
    inside = limit[0] <= stem[0] and stem[1] <= limit[1] and limit[2] <= stem[2] and stem[3] <= limit[3]
    if not inside or not site.canopy_ok(canopy):
        return []
    site.canopies.append(canopy)
    return [Dressing("tree_stem", "proc:tree_stem", (x, y, z), 0.0, (STEM_W_M, STEM_W_M, STRUCTURE_FROM_M - z)),
            Dressing("tree_canopy", f"proc:tree_canopy:{variant}", (x, y, STRUCTURE_FROM_M), 0.0, (w, d, h))]


# -- verges ------------------------------------------------------------------------------------------------------
def _verge(f: dict, street: list[Box2], site: _Site, rng: random.Random) -> list[Dressing]:
    """Trees and lamp posts along a verge, beside the kerb. The long axis is along the street; the lamp faces it."""
    x0, x1, y0, y1 = box = _aabb(f)
    along_x = (x1 - x0) >= (y1 - y0)
    lo, hi = (x0, x1) if along_x else (y0, y1)
    mid = (y0 + y1) / 2 if along_x else (x0 + x1) / 2
    across = (y1 - y0) if along_x else (x1 - x0)
    z = f["size"][2]
    centre = street[0] if street else box
    toward = ((centre[2] + centre[3]) / 2 - mid) if along_x else ((centre[0] + centre[1]) / 2 - mid)
    face = (0.0 if toward >= 0 else 180.0) if not along_x else (90.0 if toward >= 0 else -90.0)
    lamp_w = round(min(LAMP_W_M[0], across - 0.1), 2)
    out: list[Dressing] = []
    t = lo + END_M + rng.uniform(0.0, 4.0)
    while t < hi - END_M:
        xy = (t, mid) if along_x else (mid, t)
        out += _tree(*xy, z, site, rng, box)
        pitch = rng.uniform(*TREE_PITCH_M)
        tl = t + pitch / 2
        lxy = (tl, mid) if along_x else (mid, tl)
        post = Dressing("lamp_post", "proc:lamp_post", (*lxy, z), face, (lamp_w, LAMP_W_M[1], LAMP_H_M))
        if lamp_w >= 0.5 and tl < hi - END_M and site.clear((post.box[0], post.box[3], post.box[1], post.box[4])):
            out.append(post)
        t += pitch
    return out


# -- garden beds -------------------------------------------------------------------------------------------------
def _bed(f: dict, box: Box2, site: _Site, rng: random.Random) -> list[Dressing]:
    """Shrubs along a garden bed and, in some, a tree."""
    x0, x1, y0, y1 = box
    along_x = (x1 - x0) >= (y1 - y0)
    lo, hi = (x0, x1) if along_x else (y0, y1)
    mid, across = ((y0 + y1) / 2, y1 - y0) if along_x else ((x0 + x1) / 2, x1 - x0)
    z = f["size"][2]
    out: list[Dressing] = []
    taken: Box2 | None = None
    if min(x1 - x0, y1 - y0) >= 1.6 and hi - lo >= 3.0 and rng.random() < BED_TREE_P:
        t = (lo + hi) / 2 if across >= 2.0 else lo + 1.0 + rng.choice((0.0, hi - lo - 2.0))
        xy = (t, mid) if along_x else (mid, t)
        tree = _tree(*xy, z, site, rng, box)
        out += tree
        taken = (xy[0] - 1.0, xy[0] + 1.0, xy[1] - 1.0, xy[1] + 1.0) if tree else None
    t = lo + 0.8
    while across >= 0.7 and t < hi - 0.8:
        w, h = round(rng.uniform(*SHRUB_W_M), 2), round(rng.uniform(*SHRUB_H_M), 2)
        depth = round(min(w, across - 0.1), 2)
        xy = (t, mid) if along_x else (mid, t)
        variant = rng.randrange(SHRUB_VARIANTS)
        size = (w, depth, h) if along_x else (depth, w, h)
        clump = (xy[0] - size[0] / 2, xy[0] + size[0] / 2, xy[1] - size[1] / 2, xy[1] + size[1] / 2)
        if taken is None or not _meets(clump, taken):
            out.append(Dressing("shrub", f"proc:shrub:{variant}", (*xy, z), 0.0, size))
        t += rng.uniform(*SHRUB_PITCH_M)
    return out


# -- the playground ----------------------------------------------------------------------------------------------
def _playground(f: dict, box: Box2, rng: random.Random) -> list[Dressing]:
    """Swings, a slide and a climbing frame in one or two rows along the long side of a bed in a green, each turned
    (a quarter) to fit the bed's width, each `PLAY_MARGIN_M` inside it."""
    x0, x1, y0, y1 = box
    along_x = (x1 - x0) >= (y1 - y0)
    lo, hi = (x0, x1) if along_x else (y0, y1)
    width = (y1 - y0) if along_x else (x1 - x0)
    mid = (y0 + y1) / 2 if along_x else (x0 + x1) / 2
    rows = [mid] if width < 6.0 else [mid - width / 4, mid + width / 4]
    z, out, k = f["size"][2], [], rng.randrange(len(PLAY_SET))
    for row in rows:
        t = lo + PLAY_MARGIN_M
        while True:
            name, (depth, length, height) = PLAY_SET[k % len(PLAY_SET)]
            k += 1
            yaw = 90.0 if along_x else 0.0  # a piece's y (its long side) runs along the bed's long axis
            if t + length > hi - PLAY_MARGIN_M or depth > width / len(rows) - 2 * PLAY_MARGIN_M:
                break
            xy = (t + length / 2, row) if along_x else (row, t + length / 2)
            out.append(Dressing("playground", f"proc:{name}", (*xy, z), yaw, (depth, length, height)))
            t += length + PLAY_GAP_M
    return out
