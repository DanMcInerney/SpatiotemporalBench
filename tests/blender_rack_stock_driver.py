# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_rack_stock_blender.py`: builds the warehouse's six racks the way the scene does (as
objects named `pallet_rack_N`, in the map's order) and writes what their stock holds as one JSON document. Not a pytest
module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_rack_stock_driver.py -- \
        --render DIR --src DIR --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import furniture_warehouse as FW  # noqa: E402
import shapes  # noqa: E402
from shapes import entity_mesh, link  # noqa: E402

from stbench.mapspec import load_map  # noqa: E402

STOCK_MATERIALS = ("beige", "oak", "white", "black", "dark")  # the colours only stock is drawn in


def box_of(p) -> list[float]:
    """The axis-aligned box [x0, x1, y0, y1, z0, z1] of a box part turned about z."""
    c, s = math.cos(p.rot[2]), math.sin(p.rot[2])
    hx, hy = (abs(c) * p.dims[0] + abs(s) * p.dims[1]) / 2, (abs(s) * p.dims[0] + abs(c) * p.dims[1]) / 2
    return [p.loc[0] - hx, p.loc[0] + hx, p.loc[1] - hy, p.loc[1] + hy, p.loc[2] - p.dims[2] / 2,
            p.loc[2] + p.dims[2] / 2]


def build_racks() -> dict:
    """The racks of warehouse_a built in the map's order, each linked as an object so the next knows its place."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials):
        for x in list(coll):
            coll.remove(x)
    shapes.reset()
    out = {}
    for i, f in enumerate(load_map("warehouse_a").furniture):
        if f.kind != "pallet_rack":
            continue
        parts = FW.pallet_rack(tuple(f.size), f.color, tuple((*s.rect, s.z) for s in f.surfaces))
        link(f"pallet_rack_{i}", entity_mesh(f"E_{i}", parts), i + 1)
        out[f.id] = [[p.mat, *box_of(p), p.rot[2]] for p in parts if p.mat.split("*")[0] in STOCK_MATERIALS]
    return out


def jitter(n: int) -> list[list[float]]:
    """(tilt in degrees, dx, dy) of n pallets built by `_stack` with room to spare along the rack, read from the middle
    bearer (the part of the pallet's own frame at its origin)."""
    out = []
    for seed in range(n):
        parts = FW._stack(0.0, -1.35 - 0.64, 0.0, 0.8, random.Random(f"jitter:{seed}"), (-3.0, -1.0))
        bearer = [p for p in parts if p.mat == "oak*0.6"][1]  # the one at y offset 0
        out.append([math.degrees(bearer.rot[2]), bearer.loc[0], bearer.loc[1] + 1.35 + 0.64])
    return out


first = build_racks()
second = build_racks()
args.out.write_text(json.dumps({"racks": first, "again": second, "jitter": jitter(300)}), encoding="utf-8")
