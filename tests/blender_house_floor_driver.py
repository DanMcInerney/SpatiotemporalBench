# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_house_floor_blender.py`: builds the scene of an episode on `house_floor` from the
asset library (no render, so no GPU), casts the rays the test names against the whole scene (`Scene.ray_cast`: the first
thing each meets, whatever object it is) and prints one JSON document with the lamps and the dressing the scene holds,
the stairs' object and what each ray hit. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_house_floor_driver.py -- --render DIR --src DIR
        --root EPISODE --private PRIVATE --manifest M.json --library L.blend --rays RAYS.json --out RESULT.json

`RAYS.json` is a list of `{"name", "origin": [x, y, z], "direction": [x, y, z]}` in the map frame; the result's
`hits` has one `{"hit", "object", "point"}` per name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--root", "--private", "--manifest", "--library", "--rays", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import build_scene  # noqa: E402

from stbench import asset_manifest  # noqa: E402
from stbench.mapspec import load_map  # noqa: E402
from stbench.timeline import load  # noqa: E402


def main() -> None:
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    hfov = json.loads((args.root / "sensors" / "camera.json").read_text())["hfov_deg"]
    scene = build_scene.build(tl, m, hfov, (asset_manifest.load(args.manifest), args.library))
    scene.apply_time(0.0)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hits = {}
    for ray in json.loads(args.rays.read_text()):
        hit, point, _normal, _index, ob, _matrix = bpy.context.scene.ray_cast(
            depsgraph, Vector(ray["origin"]), Vector(ray["direction"]))
        hits[ray["name"]] = {"hit": bool(hit), "object": ob.name if hit else None,
                             "point": [round(v, 4) for v in point] if hit else None}
    stairs = next(scene.objects[e.id] for e in tl.entities if e.kind == "stairs")
    result = {"stairs": stairs.name, "lamps": scene.lighting["lamps"],
              "dressing": sorted(o.name for o in bpy.data.objects if o.name.startswith("dressing_")), "hits": hits}
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


main()
