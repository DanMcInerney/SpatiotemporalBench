# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_floor_look_blender.py`: a floor with true relief and a carpet with a pile, from a
texture set made here (a flat colour, a flat roughness, and a displacement map of a plateau with one 10 cm groove down
its middle, so nothing is downloaded). Writes one JSON document. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_floor_driver.py -- --render DIR --src DIR --out R.json

- `relief`: a 2 m square floor with the set's material, the relief of `floor_look.displace` (5 mm) and its adaptive
  subdivision, seen straight down from 0.5 m on the GPU at the assets settings: the depth along the row through the
  groove, with and without the displacement; and what `displace` did to the material.
- `pile`: `wear.worn(material, "carpet")`: the sheen and the chain of bumps on its Principled node.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import cycles_setup  # noqa: E402
import floor_look  # noqa: E402
import materials  # noqa: E402
import wear  # noqa: E402

SIZE = 256  # pixels of a map: one metre
RELIEF_M = 0.005


def save(path: Path, grey: np.ndarray) -> None:
    img = bpy.data.images.new(path.stem, SIZE, SIZE, alpha=False)
    pixels = np.ones((SIZE, SIZE, 4), dtype=np.float32)
    pixels[..., :3] = grey[..., None]
    img.pixels.foreach_set(pixels.ravel())
    img.filepath_raw = str(path)
    img.file_format = "JPEG"
    img.save()


def texture_set(folder: Path) -> None:
    """Colour and roughness flat, and a displacement of 1 (the plateau) with 0 (a groove) at u in 0.45..0.55."""
    folder.mkdir(parents=True, exist_ok=True)
    u = (np.arange(SIZE) + 0.5) / SIZE
    height = np.where((u > 0.45) & (u < 0.55), 0.0, 1.0)[None, :].repeat(SIZE, axis=0)
    save(folder / "Test_2K-JPG_Color.jpg", np.full((SIZE, SIZE), 0.5))
    save(folder / "Test_2K-JPG_Roughness.jpg", np.full((SIZE, SIZE), 0.5))
    save(folder / "Test_2K-JPG_Displacement.jpg", height)


def read_depth(path: Path) -> np.ndarray:
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    i = 0
    while inp.seek_subimage(i, 0):
        if "Depth" in ",".join(inp.spec().channelnames):
            out = inp.read_image(i, 0, 0, 1, oiio.FLOAT)[..., 0]
            inp.close()
            return out
        i += 1
    raise RuntimeError("no depth in the render")


def fresh() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    return sc


def depth_row(folder: Path, relief: bool) -> dict:
    """The depth along the middle row of a frame looking straight down at x = 0.5, the groove's middle, from 0.5 m."""
    sc = fresh()
    mat = materials.ambientcg_material("acg_Test", folder, [1.0, 1.0])
    made = floor_look.displace(mat, RELIEF_M) if relief else False
    me = bpy.data.meshes.new("floor")
    me.from_pydata([(-1, -1, 0), (2, -1, 0), (2, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    me.materials.append(mat)
    ob = bpy.data.objects.new("floor", me)
    sc.collection.objects.link(ob)
    if relief:
        floor_look.subdivide(ob)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit, cam_data.angle, cam_data.clip_start = "HORIZONTAL", math.radians(60), 0.01
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.location, cam.rotation_euler = (0.5, 0.0, 0.5), (0.0, 0.0, 0.0)  # straight down
    cycles_setup.configure(sc, 160, 120, 4, "assets")
    s = sc.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "MULTI_LAYER_IMAGE"
    s.file_format, s.color_depth, s.exr_codec = "OPEN_EXR_MULTILAYER", "32", "ZIP"
    bpy.ops.render.render(write_still=False)
    exr = Path(tempfile.gettempdir()) / f"stbench_floor_{id(mat)}.exr"
    bpy.data.images["Render Result"].save_render(str(exr), scene=sc)
    try:
        row = read_depth(exr)[60]
    finally:
        exr.unlink()
    return {"made": made, "row": [round(float(v), 5) for v in row[::8]], "min": float(row.min()),
            "max": float(row.max()), "median": float(np.median(row)),
            "method": mat.displacement_method, "max_vertex": float(mat.max_vertex_displacement),
            "prop": mat.get(floor_look.PROP), "modifiers": [(m.type, m.use_adaptive_subdivision) for m in ob.modifiers]}


def pile(folder: Path) -> dict:
    fresh()
    base = materials.ambientcg_material("acg_Test", folder, [1.0, 1.0])
    out = wear.worn(base, "carpet")
    p = materials.principled(out)
    bumps = [n for n in out.node_tree.nodes if n.bl_idname == "ShaderNodeBump"]
    top = p.inputs["Normal"].links[0].from_node
    below = top.inputs["Normal"].links[0].from_node if top.inputs["Normal"].links else None
    plain = wear.worn(base, "wood")
    return {"sheen": float(p.inputs["Sheen Weight"].default_value),
            "sheen_roughness": float(p.inputs["Sheen Roughness"].default_value), "bumps": len(bumps),
            "top_is_bump": top.bl_idname == "ShaderNodeBump", "chained": below is not None
            and below.bl_idname == "ShaderNodeBump", "wood_sheen": float(
                materials.principled(plain).inputs["Sheen Weight"].default_value)}


def main() -> None:
    folder = Path(tempfile.mkdtemp(prefix="stbench_floor_")) / "set"
    texture_set(folder)
    result = {"relief": depth_row(folder, True), "flat": depth_row(folder, False), "pile": pile(folder),
              "limit": floor_look.MAX_DISPLACE_M}
    args.out.write_text(json.dumps(result), encoding="utf-8")


main()
