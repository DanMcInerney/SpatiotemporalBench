# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_warehouse_pieces_blender.py`: builds the warehouse's charger, a bay and a staging lane
at the sizes of `warehouse_a` the way the scene builds a piece (`furniture_warehouse.BUILDERS`, then `entity_mesh`) and
writes the extent of each mesh in the piece's own frame (x to its front, origin at the base centre) as one JSON
document. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_warehouse_pieces_driver.py -- \
        --render DIR --src DIR --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import furniture_warehouse as FW  # noqa: E402
import shapes  # noqa: E402
from shapes import entity_mesh  # noqa: E402

from stbench.mapspec import load_map  # noqa: E402

KINDS = ("charger", "bay", "staging_lane")


def extent(kind: str) -> dict:
    """The size the map gives the first piece of the kind, and the box [x0, x1, y0, y1, z0, z1] its mesh fills."""
    shapes.reset()
    f = next(f for f in load_map("warehouse_a").furniture if f.kind == kind)
    parts = FW.BUILDERS[kind](tuple(f.size), f.color, tuple((*s.rect, s.z) for s in f.surfaces))
    mesh = entity_mesh(f"E_{kind}", parts)
    axes = list(zip(*(tuple(v.co) for v in mesh.vertices), strict=True))
    return {"size": list(f.size), "box": [v for a in axes for v in (min(a), max(a))],
            "colours": sorted({p.mat for p in parts})}


args.out.write_text(json.dumps({kind: extent(kind) for kind in KINDS}), encoding="utf-8")
