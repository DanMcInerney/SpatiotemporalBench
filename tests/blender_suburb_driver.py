# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_suburb_blender.py`: builds a suburb episode's scene from the asset library (no render,
so no GPU) and writes what it holds as one JSON document. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_suburb_driver.py -- \
        --render DIR --src DIR --root EPISODE --private PRIVATE --library L.blend --manifest M.json --out RESULT.json

The result: every object's name; the material names of the ground and of each region's floor; each far-scenery object's
world box (`scenery_*`); the objects with a pass index (the entities, nothing else); the dressing objects by class; the
camera's clip distance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--root", "--private", "--library", "--manifest", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import build_scene  # noqa: E402
from mathutils import Vector  # noqa: E402

from stbench import asset_manifest  # noqa: E402
from stbench.camera import Camera  # noqa: E402
from stbench.mapspec import load_map  # noqa: E402
from stbench.timeline import load  # noqa: E402


def world_box(ob) -> list[float]:
    """The object's bounding box in the map frame: x0, y0, z0, x1, y1, z1."""
    corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo, hi = [min(c[i] for c in corners) for i in range(3)], [max(c[i] for c in corners) for i in range(3)]
    return [round(v, 4) for v in (*lo, *hi)]


def material_names(ob) -> list[str | None]:
    return [None if s.material is None else s.material.name for s in ob.material_slots]


def main() -> None:
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    hfov = json.loads((args.root / "sensors" / "camera.json").read_text())["hfov_deg"]
    Camera.load(args.root / "sensors" / "camera.json")
    scene = build_scene.build(tl, m, hfov, (asset_manifest.load(args.manifest), args.library))
    bpy.context.view_layer.update()  # the objects' `matrix_world` from their locations (else each box is at the origin)
    objects = list(bpy.data.objects)
    result = {
        "objects": sorted(o.name for o in objects),
        "ground": material_names(bpy.data.objects["ground"]),
        "floors": {o.name: material_names(o) for o in objects if o.name.startswith("floor_")},
        "scenery": [{"name": o.name, "box": world_box(o), "pass_index": o.pass_index}
                    for o in objects if o.name.startswith("scenery")],
        "indexed": sorted([o.pass_index, o.name] for o in objects if o.pass_index),
        "instance_map": {str(k): v for k, v in scene.instance_map.items()},
        "dressing": sorted(o.name.removeprefix("dressing_").rsplit("_", 1)[0]
                           for o in objects if o.name.startswith("dressing_")),
        "clip_end": scene.camera.data.clip_end,
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


main()
