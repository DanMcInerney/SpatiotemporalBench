# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_dressing_build_warehouse_blender.py`: calls the warehouse's builders
(`dressing_build_warehouse.BUILDERS`) on every Dressing of `warehouse_a`'s layout, the way `dressing.draw` does (an
asset piece is its class's stand-in: no library), twice, and writes what they made as one JSON document: per piece the
box of its mesh in its own frame and in the map frame (placed by the Dressing's yaw and position, which is the
dispatcher's job and done here by hand), its material slots, how its text and barcode are made, and how much of its box
its yellow parts cover seen from each side. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_dressing_warehouse_driver.py -- \
        --render DIR --src DIR --out RESULT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from importlib import resources
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import dressing  # noqa: E402
import dressing_build_warehouse as W  # noqa: E402
import dressing_layout_warehouse as L  # noqa: E402
import materials  # noqa: E402
import numpy as np  # noqa: E402

TEXT_CLASSES = ("beam_label", "aisle_sign", "door_number")
GRID = 64  # the resolution of the coverage raster


def clear() -> None:
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.collections, bpy.data.node_groups):
        for x in list(coll):
            coll.remove(x)
    materials.use_map(None)  # forgets the named materials, which are freed with the file's data


def kind_of(mat) -> str:
    return mat.name.split(":")[-1].split(".")[0] if mat else ""


def vertices(me) -> np.ndarray:
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def slot_polys(me, name: str) -> list:
    slots = [i for i, m in enumerate(me.materials) if kind_of(m) == name]
    return [p for p in me.polygons if p.material_index in slots]


def islands(polys) -> list[set[int]]:
    """The connected groups of vertices of a set of polygons."""
    parent: dict[int, int] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in polys:
        first = find(p.vertices[0])
        for v in p.vertices[1:]:
            parent[find(v)] = first
    groups: dict[int, set[int]] = {}
    for v in list(parent):
        groups.setdefault(find(v), set()).add(v)
    return list(groups.values())


def extent(co: np.ndarray, ids) -> list[float]:
    pts = co[sorted(ids)]
    return [round(float(v), 6) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def coverage(me, name: str, size) -> list[float]:
    """The share of the Dressing's box (seen along x, along y and along z) that the polygons of material `name` cover:
    their triangles rasterised on a grid."""
    co = vertices(me)
    lo = np.array([-size[0] / 2, -size[1] / 2, 0.0])
    span = np.array(size)
    tris = []
    for p in slot_polys(me, name):
        v = list(p.vertices)
        tris += [(v[0], v[i], v[i + 1]) for i in range(1, len(v) - 1)]
    out = []
    for axes in ((1, 2), (0, 2), (0, 1)):
        hit = np.zeros((GRID, GRID), dtype=bool)
        gx, gy = np.meshgrid((np.arange(GRID) + 0.5) / GRID, (np.arange(GRID) + 0.5) / GRID, indexing="ij")
        for a, b, c in tris:
            pa, pb, pc = (((co[i][list(axes)] - lo[list(axes)]) / span[list(axes)]) for i in (a, b, c))
            den = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
            if abs(den) < 1e-12:
                continue
            w1 = ((pb[1] - pc[1]) * (gx - pc[0]) + (pc[0] - pb[0]) * (gy - pc[1])) / den
            w2 = ((pc[1] - pa[1]) * (gx - pc[0]) + (pa[0] - pc[0]) * (gy - pc[1])) / den
            hit |= (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        out.append(round(float(hit.mean()), 4) if tris else 0.0)
    return out


def digest(objs) -> str:
    doc = []
    for ob in objs:
        me = ob.data
        idx = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get("material_index", idx)
        doc.append([ob.name, len(me.vertices), len(me.polygons), hashlib.sha256(np.round(vertices(me), 5).tobytes())
                    .hexdigest(), hashlib.sha256(idx.tobytes()).hexdigest(), [kind_of(m) for m in me.materials],
                    ob.get("stbench_text")])
    return hashlib.sha256(json.dumps(doc).encode()).hexdigest()


def build_all(pieces) -> tuple[list[list], list]:
    clear()
    ctx = dressing.Ctx()
    made = []
    for d in pieces:
        kind, key, text = dressing.parse(d.recipe)
        objs = W.BUILDERS[key if kind == "proc" else d.cls](d, ctx, text)
        made.append(objs)
    return made, [ob for objs in made for ob in objs]


def report(d, objs) -> dict:
    (ob,) = objs if len(objs) == 1 else (None,)
    row = {"cls": d.cls, "recipe": d.recipe, "objects": len(objs)}
    if ob is None:
        return row
    me = ob.data
    co = vertices(me)
    yaw = math.radians(d.yaw_deg)
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    world = co @ rot.T + np.array(d.pos)
    row.update(pass_index=ob.pass_index, identity=bool(np.allclose(np.array(ob.matrix_world), np.eye(4))),
               local=[round(float(v), 6) for v in (*co.min(axis=0), *co.max(axis=0))],
               world=[round(float(v), 6) for v in (*world.min(axis=0), *world.max(axis=0))],
               materials=[kind_of(m) for m in me.materials], polys=len(me.polygons),
               text_property=ob.get("stbench_text"), mesh_text_property=me.get("stbench_text"),
               slot_polys=[sum(1 for p in me.polygons if p.material_index == i) for i in range(len(me.materials))])
    for lamp in ("led_emitter_cool", "led_emitter"):
        if lamp in row["materials"]:
            row["lens"] = extent(co, {v for p in slot_polys(me, lamp) for v in p.vertices})
    if "safety_yellow" in row["materials"]:
        row["yellow_coverage"] = coverage(me, "safety_yellow", d.size)
    if d.cls in TEXT_CLASSES:
        ink = islands(slot_polys(me, "rubber_black"))
        row["ink_islands"] = len(ink)
        row["ink"] = [extent(co, g) for g in ink]
        row["plate"] = extent(co, {v for p in slot_polys(me, "label_paper") for v in p.vertices})
        if d.cls == "beam_label":
            bars = sorted((extent(co, g) for g in islands(slot_polys(me, "black_plastic"))), key=lambda e: e[1])
            row["bars"] = [[e[1], e[4]] for e in bars]
    return row


def main() -> None:
    spec = json.loads(resources.files("stbench.maps").joinpath("warehouse_a.json").read_bytes())
    pieces = L.layout(spec)
    made, flat = build_all(pieces)
    rows = [report(d, objs) for d, objs in zip(pieces, made, strict=True)]
    first = digest(flat)
    made2, flat2 = build_all(pieces)
    result = {"rows": rows, "digest": [first, digest(flat2)],
              "linked": sorted(o.name for o in bpy.data.objects if o.users_collection)}
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(f"STBENCH-WAREHOUSE {len(rows)} pieces", flush=True)


main()
