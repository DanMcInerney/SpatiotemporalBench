# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_asset_dressing_blender.py`: what `DressingAssets` (render/asset_dressing.py) hands a
dressing builder from a library, as one JSON document. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_dressing_asset_driver.py -- \
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

from asset_dressing import DressingAssets  # noqa: E402

from stbench import asset_manifest  # noqa: E402


def extent(ob) -> list:
    """The (min, max) of an object's mesh vertices."""
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return [co.min(axis=0).tolist(), co.max(axis=0).tolist()]


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    m = asset_manifest.load(args.manifest)
    dressing = DressingAssets(m, args.library)
    first, second = dressing.asset("dress_rest"), dressing.asset("dress_rest")
    result = {
        "in_library": sorted(dressing.in_library),
        "has": {i: dressing.has(i) for i in ("dress_rest", "dress_open", "dress_drop", "proc_glass", "nope")},
        "absent_is_none": dressing.asset("nope") is None and dressing.asset("proc_glass") is None,
        "native": {i: [round(v, 5) for v in dressing.native(i)] for i in ("dress_rest", "dress_open", "dress_drop")},
        "shared_mesh": first.data is second.data and first is not second,
        "pass_index": first.pass_index, "scale": list(first.scale), "collections": len(first.users_collection),
        "materials": {i: [s.material.name for s in dressing.asset(i).material_slots]
                      for i in ("dress_rest", "dress_drop")},
        "polygons": {i: len(dressing.asset(i).data.polygons) for i in ("dress_rest", "dress_open", "dress_drop")},
        "mesh_names": sorted(me.name for me in bpy.data.meshes), "extent": {"dress_rest": extent(first)},
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


main()
