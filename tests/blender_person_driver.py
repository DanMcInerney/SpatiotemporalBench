# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_person_render_blender.py`: builds the resident's entity from a library the way
`build_scene` does (`Appearance.make_object`, the entity's size the model's own box) and prints one JSON document that
the test reads: the object's count and scale, its extent in the world, the slots it draws with and where its eyes are.
Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_person_driver.py -- \
        --render DIR --src DIR --manifest M --library L.blend --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--manifest", "--library", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import asset_entities as AE  # noqa: E402

from stbench import asset_manifest  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
manifest = asset_manifest.load(args.manifest)
model = manifest.asset("resident")
app = AE.Appearance(manifest, args.library, [model.id])
ob = app.make_object(model.id, "person", "grey", tuple(model.native_bbox_m), "person_1", 1)
co = np.empty(len(ob.data.vertices) * 3)
ob.data.vertices.foreach_get("co", co)
co = (co.reshape(-1, 3) * np.array(ob.scale)) + np.array(ob.location)
slots = [slot.material.name for slot in ob.material_slots]
eyes = next(i for i, name in enumerate(slots) if name.endswith(":eyes"))
polygons = ob.data.polygons
eye_vertices = sorted({v for p in polygons if p.material_index == eyes for v in p.vertices})
result = {
    "objects": len([o for o in bpy.data.objects if o.type == "MESH"]),
    "scale": list(ob.scale), "pass_index": ob.pass_index, "slots": slots,
    "min": co.min(axis=0).tolist(), "max": co.max(axis=0).tolist(),
    "eyes_centroid": co[eye_vertices].mean(axis=0).tolist(),
    "slot_polygons": [sum(1 for p in polygons if p.material_index == i) for i in range(len(slots))],
}
args.out.write_text(json.dumps(result), encoding="utf-8")
