# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_furniture_under_blender.py`: builds each blocking piece of a map alone from
`render/furniture.py` at the origin (as `build_scene` builds it) and casts rays straight up from 1 cm below the floor
at a grid over its footprint, 2.5 cm in from every edge and every 5 cm after that, in the piece's own frame. Prints
the height of the first thing each ray meets (null: nothing) as one JSON document. No render, no GPU. Not a pytest
module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_under_driver.py -- \
        --render DIR --src DIR --map apartment_a --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
ap.add_argument("--map", default="apartment_a")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import furniture  # noqa: E402
import shapes  # noqa: E402

from stbench.mapspec import load_map  # noqa: E402

EDGE_M, STEP_M, FROM_Z = 0.025, 0.05, -0.01  # the first ray's distance from an edge, the spacing, the rays' start


def grid(side: float) -> list[float]:
    """Local coordinates along one axis of a piece `side` long: EDGE_M from the first edge, every STEP_M, up to EDGE_M
    short of the last."""
    n = int((side - 2 * EDGE_M) / STEP_M + 1e-9)
    return [-side / 2 + EDGE_M + STEP_M * i for i in range(n + 1)]


def first_hits(f) -> list[list]:
    """`[x, y, z]` per ray: where it starts and the height it first meets the piece at (null: it meets nothing)."""
    zs = tuple(sorted(s.z for s in f.surfaces))
    me = shapes.entity_mesh(f.id, furniture.BUILDERS[f.kind](tuple(f.size), f.color, zs))
    tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
    out = []
    for x in grid(f.size[0]):
        for y in grid(f.size[1]):
            hit = tree.ray_cast(Vector((x, y, FROM_Z)), Vector((0.0, 0.0, 1.0)), 10.0)[0]
            out.append([round(x, 6), round(y, 6), None if hit is None else round(hit.z, 6)])
    return out


bpy.ops.wm.read_factory_settings(use_empty=True)
shapes.reset()
rows = [{"id": f.id, "kind": f.kind, "size": list(f.size), "points": first_hits(f)}
        for f in load_map(args.map).furniture if f.blocks]
args.out.write_text(json.dumps(rows), encoding="utf-8")
