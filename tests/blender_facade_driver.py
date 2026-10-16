# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_facade_windows_blender.py`: renders one building of each facade style straight on with
an orthographic camera, sunlit and in shade, and measures what its windows show. Not a pytest module (Blender's Python
has no pytest).

blender --background --factory-startup --python tests/blender_facade_driver.py -- --render DIR --src DIR --out
RESULT.json

A building is 18 m wide, 12 m deep and 15.3 m high (three storeys over the 4.5 m storefront band: `facade_shaders`'s
`GROUND_H` and `FLOOR_H`). The camera looks along +y at the face on y = 0 from 40 m, 18 m across, 32 samples, the colour
pipeline of the episodes (`cycles_setup.configure`). Per window (a punched wall's: 1.7 x 1.9 m in each 3 m bay of each
storey, a curtain wall's pane: 1.3 x 2.5 m in each 1.5 m bay) the result holds the mean display luminance (0 to 1) of
the middle of the window, and per style the same for the wall between windows and for each storefront bay, and the facts
of the building's roof (a lot 12 m deep has a real one).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import facade_shaders  # noqa: E402
import facades  # noqa: E402
import numpy as np  # noqa: E402
from cycles_setup import configure  # noqa: E402
from mathutils import Vector  # noqa: E402
from scene_mesh import MeshBuilder  # noqa: E402

W, H, RES, SAMPLES = 18.0, 15.3, 640, 32
SUNS = {"sun": (0.3, -0.55, 0.78), "shade": None}  # the direction to the sun: from the front, high; or none


def clear() -> None:
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.node_groups):
        for x in list(coll):
            coll.remove(x)


def luminance(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def setup(style: str, sun) -> None:
    clear()
    mats = facades.Materials()
    mb = MeshBuilder("facade_test")
    facades.add_building(mb, mats, style, 0.0, W, 0.0, 12.0, H)
    mb.finish()
    world = bpy.data.worlds.new("sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.35, 0.5, 0.75, 1.0)
    bg.inputs["Strength"].default_value = 0.55
    if sun is not None:
        light = bpy.data.lights.new("sun", "SUN")
        light.energy, light.angle = 3.6, math.radians(0.53)
        ob = bpy.data.objects.new("sun", light)
        bpy.context.scene.collection.objects.link(ob)
        ob.rotation_euler = Vector(sun).to_track_quat("Z", "Y").to_euler()
    cam = bpy.data.cameras.new("cam")
    cam.type, cam.ortho_scale = "ORTHO", W
    ob = bpy.data.objects.new("cam", cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location, ob.rotation_euler = (W / 2, -40.0, H / 2 + 1.0), (math.pi / 2, 0.0, 0.0)
    bpy.context.scene.camera = ob


def render() -> np.ndarray:
    """The frame as display values (0..1), rows from the top."""
    bpy.ops.render.render(write_still=False)
    path = args.out.with_suffix(".png")
    bpy.data.images["Render Result"].save_render(str(path), scene=bpy.context.scene)
    img = bpy.data.images.load(str(path))
    px = np.array(img.pixels[:], dtype=np.float32).reshape(RES, RES, 4)[::-1, :, :3]  # its rows start at the bottom
    bpy.data.images.remove(img)
    return px


def box(px: np.ndarray, x0: float, x1: float, z0: float, z1: float, inset: float = 0.15) -> float:
    """The mean luminance over a window's rectangle in metres of the facade (x along it, z up), pulled in by `inset`."""
    scale, zc = RES / W, H / 2 + 1.0
    w, h = (x1 - x0) * inset, (z1 - z0) * inset
    cols = [int((x + 0.0) * scale) for x in (x0 + w, x1 - w)]
    rows = [int(RES / 2 - (z - zc) * scale) for z in (z1 - h, z0 + h)]
    return float(luminance(px[rows[0]:rows[1], cols[0]:cols[1]]).mean())


def windows(style: str) -> list[dict]:
    spec = facade_shaders.FACADES[style]
    out = []
    if "curtain" in spec:
        for k in range(3):
            for i in range(int(W / 1.5)):
                out.append({"floor": k, "bay": i, "rect": (1.5 * i + 0.1, 1.5 * i + 1.4, 4.5 + 3.6 * k + 1.0,
                                                           4.5 + 3.6 * k + 3.5)})
    else:
        for k in range(3):
            for i in range(int(W / 3.0)):
                out.append({"floor": k, "bay": i, "rect": (3.0 * i + 0.65, 3.0 * i + 2.35, 4.5 + 3.6 * k + 0.85,
                                                           4.5 + 3.6 * k + 2.75)})
    return out


def roof_facts() -> dict:
    """What the building's roof is made of (read from its mesh, in metres): the area of the tar roof that faces up at
    the roof's height, the area of the parapet's top (trim) at its height, and the highest point of the equipment."""
    me = bpy.data.objects["facade_test"].data
    tar = top = 0.0
    units = 0
    unit_top = 0.0
    for poly in me.polygons:
        name = me.materials[poly.material_index].name
        z = poly.center.z
        if name == "roof_tar" and poly.normal.z > 0.9 and abs(z - H) < 1e-3:
            tar += poly.area
        if name == "trim_stone" and poly.normal.z > 0.9 and abs(z - (H + 0.9)) < 1e-3:
            top += poly.area
        if name == "roof_unit":
            units += 1
            unit_top = max(unit_top, max(me.vertices[i].co.z for i in poly.vertices))
    return {"tar_area": tar, "parapet_top_area": top, "unit_faces": units, "unit_top": unit_top - H}


def main() -> None:
    configure(bpy.context.scene, RES, RES, SAMPLES, "assets")
    result = {}
    for style in facade_shaders.FACADES:
        for light, sun in SUNS.items():
            setup(style, sun)
            px = render()
            rows = [{**w, "lum": box(px, *w["rect"])} for w in windows(style)]
            spec = facade_shaders.FACADES[style]
            wall = None if "curtain" in spec else float(np.mean([
                box(px, 3.0 * i + 0.0, 3.0 * i + 0.3, 4.5 + 3.6 * k + 0.9, 4.5 + 3.6 * k + 2.7, 0.0)
                for k in range(3) for i in range(6)]))
            shops = [{"bay": i, "lum": box(px, 3.0 * i + 0.4, 3.0 * i + 2.6, 0.7, 3.2)} for i in range(6)]
            result[f"{style}/{light}"] = {"windows": rows, "wall": wall, "shops": shops, "roof": roof_facts()}
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print("STBENCH-FACADES done", flush=True)


main()
