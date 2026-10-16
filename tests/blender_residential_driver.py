# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_residential_blender.py`: builds the residential kit's pieces, dressing and far scenery
with no render (so no GPU and no asset library) and writes what the scene holds as one JSON document. Not a pytest
module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_residential_driver.py -- \
        --render DIR --src DIR --spec KIT.json --rays RAYS.json --out RESULT.json

- **pieces**: each request of RAYS (a kind, a size, a colour, rays as [origin, direction]) is built alone by
  `furniture_residential.BUILDERS` and `shapes.entity_mesh` with the residential resolver (`Surfaces`, over the
  city's, as `build_scene` builds a piece), at the origin; each ray's first hit is its location, or null;
- **materials**: every key stem `materials_residential` makes, built once;
- **dressing**: `dressing.build(spec, "residential", Ctx())`, each object's name, pass index, collections and mesh in
  the piece's own frame;
- **scenery**: `scenery_residential.build(map, None, None)`, each object's name and world box.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--spec", "--rays", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import dressing  # noqa: E402
import furniture_city  # noqa: E402
import furniture_residential  # noqa: E402
import materials_residential  # noqa: E402
import numpy as np  # noqa: E402
import scenery_residential  # noqa: E402
import shapes  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

from stbench.mapspec import parse_map  # noqa: E402


def clear() -> None:
    """An empty file: the objects, meshes, materials, collections, images and node groups of a build before are gone."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.collections, bpy.data.node_groups,
                 bpy.data.images):
        for x in list(coll):
            coll.remove(x)
    shapes.reset()


def pieces(requests: list[dict]) -> list[dict]:
    """The first hit of each ray on each requested piece, built as `build_scene` builds a fixed piece of the kit."""
    out = []
    for req in requests:
        city = furniture_city.Surfaces()
        resolve = furniture_residential.Surfaces(city).resolver()
        parts = furniture_residential.BUILDERS[req["kind"]](tuple(req["size"]), req["color"], ())
        me = shapes.entity_mesh(req["kind"], parts, resolve)
        tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        hits = []
        for origin, direction in req["rays"]:
            hit = tree.ray_cast(Vector(origin), Vector(direction), 100.0)[0]
            hits.append(None if hit is None else [round(c, 6) for c in hit])
        out.append({"kind": req["kind"], "size": req["size"], "hits": hits, "parts": len(parts),
                    "materials": [m.name for m in me.materials]})
    return out


def materials() -> dict[str, int]:
    """Every material of `materials_residential`, built: its name and how many nodes its tree has."""
    made = {}
    for name, make in materials_residential.FACTORIES.items():
        for arg in {"shingle": list(materials_residential.ROOFS), "siding": ["beige", "blue"], "render": ["white"],
                    "brick": ["red", "brown"]}.get(name, [None]):
            m = make(arg)
            made[f"{name}:{arg}" if arg else name] = len(m.node_tree.nodes)
    return made


def local_box(ob) -> list[float]:
    """The mesh in the piece's own frame (x out of the wall, y across, z up): the world points relative to the object's
    place, turned back by its yaw."""
    me = ob.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    w = np.array(ob.matrix_world)
    rel = co.reshape(-1, 3) @ w[:3, :3].T + w[:3, 3] - np.array(ob.location)
    c, s = math.cos(-ob.rotation_euler.z), math.sin(-ob.rotation_euler.z)
    pts = rel @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return [round(float(v), 6) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def world_box(ob) -> list[float]:
    me = ob.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    w = np.array(ob.matrix_world)
    pts = co.reshape(-1, 3) @ w[:3, :3].T + w[:3, 3]
    return [round(float(v), 4) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def dress(spec: dict) -> list[dict]:
    clear()
    objs = dressing.build(spec, "residential", dressing.Ctx())
    bpy.context.view_layer.update()
    return [{"name": ob.name, "pass_index": ob.pass_index, "collections": [c.name for c in ob.users_collection],
             "local": local_box(ob), "loc": [round(v, 6) for v in ob.location], "rot_z": round(ob.rotation_euler.z, 6),
             "verts": len(ob.data.vertices), "materials": [m.name if m else None for m in ob.data.materials]}
            for ob in objs]


def scenery(spec: dict) -> list[dict]:
    clear()
    scenery_residential.build(parse_map(spec), None, None)
    bpy.context.view_layer.update()
    return [{"name": ob.name, "pass_index": ob.pass_index, "box": world_box(ob), "verts": len(ob.data.vertices)}
            for ob in bpy.data.objects]


bpy.ops.wm.read_factory_settings(use_empty=True)
shapes.reset()
spec = json.loads(args.spec.read_text(encoding="utf-8"))
result = {"pieces": pieces(json.loads(args.rays.read_text(encoding="utf-8")))}
result["materials"] = materials()
result["dressing"] = dress(spec)
result["scenery"] = scenery(spec)
args.out.write_text(json.dumps(result), encoding="utf-8")
print(f"STBENCH-RESIDENTIAL {len(result['dressing'])} dressing, {len(result['scenery'])} scenery objects", flush=True)
