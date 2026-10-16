# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_delivery_pieces_blender.py`: builds the downtown's scene of a script (the placeholder
drawings of its entities; no dressing, no render, so no GPU and no download) and writes what the test compares as one
JSON document: the world box and the material names of every drone nest, drop zone and plaza piece, and the node types
of every paving material. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_delivery_driver.py -- \
        --render DIR --src DIR --private SCRIPT_PRIVATE --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--private", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import build_scene  # noqa: E402
import numpy as np  # noqa: E402
import shapes  # noqa: E402

from stbench.mapspec import load_map  # noqa: E402
from stbench.timeline import load  # noqa: E402

KINDS = ("nest", "drop_zone", "plaza")


def world_box(ob) -> list[float]:
    """The box of an object's mesh in the map frame (xmin, ymin, zmin, xmax, ymax, zmax), to a tenth of a millimetre."""
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    w = np.array(ob.matrix_world)
    pts = co.reshape(-1, 3) @ w[:3, :3].T + w[:3, 3]
    return [round(float(v), 4) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def main() -> None:
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    shapes.reset()
    scene = build_scene.build(tl, m, 80.0, None)
    scene.apply_time(0.0)
    bpy.context.view_layer.update()  # the objects' world matrices, after their transforms were set
    out: dict = {"pieces": {}, "paving": {}}
    for e in tl.entities:
        if e.kind in KINDS:
            ob = scene.objects[e.id]
            out["pieces"][e.id] = {"kind": e.kind, "box": world_box(ob),
                                   "materials": [mat.name for mat in ob.data.materials]}
    for mat in bpy.data.materials:
        if mat.name.startswith("shader:paving"):
            out["paving"][mat.name] = sorted({n.bl_idname for n in mat.node_tree.nodes})
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"STBENCH-DELIVERY {len(out['pieces'])} pieces, {len(out['paving'])} paving materials", flush=True)


main()
