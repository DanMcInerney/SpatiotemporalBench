# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_render_lighting_blender.py` and `test_render_settings_blender.py`: writes one JSON
document that the test reads. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_lighting_driver.py -- --task scene|physics|passes
        --render DIR --src DIR [--root EPISODE --private PRIVATE --manifest M.json --library L.blend] --out RESULT.json

- `scene`: builds an episode's scene with the assets appearance (no render) and reports every light, the world, the
  dressing lamps' and high-bay fixtures' boxes and the objects that must not be there.
- `physics`: renders tiny scenes on the CPU and reports what a lamp of known power, colour and balance, and an HDRI with
  a turned heading, put on a card (scene-linear values).
- `lux`: builds an episode's scene and reads grey cards (`lighting_probe`: albedo 0.18) flat at 0.75 m on a 3 x 3 grid
  of every room (or, with `--points`, at the given `{"pos", "normal"}` points, or with `--surface KIND` on every
  furniture surface of that kind), lit by the lamps alone (`--sources lamps`, the default), the daylight alone or all:
  each card's lux (a card reads 0.18 at `cycles_setup.UNIT_LUX`) and the cast of its colour (r/g, b/g). The room grid's
  cards also carry `direct`: the same three read with no bounce, the lamps' own light alone, which no wall's paint
  tints.
- `passes`: renders a tiny scene on the GPU with the settings the renders had before realism Phase 1 and with the ones
  `cycles_setup.configure` gives the assets appearance, and reports whether the depth and object index passes are the
  same bit for bit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=("scene", "physics", "passes", "lux"), required=True)
ap.add_argument("--sources", choices=("lamps", "daylight", "all"), default="lamps")
ap.add_argument("--surface", default=None, help="the Surface.kind to put cards on (the lux task)")
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
for flag in ("--root", "--private", "--manifest", "--library", "--points"):
    ap.add_argument(flag, type=Path)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import cycles_setup  # noqa: E402
import lighting  # noqa: E402
import lighting_probe  # noqa: E402
import lighting_sky  # noqa: E402

RESULT: dict = {}


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    return sc


def r(x, n: int = 4):
    return [round(float(v), n) for v in x]


def read_exr(path: Path) -> dict[str, np.ndarray]:
    """Every part of a multilayer float EXR by what it holds: 'combined' (R, G, B), 'depth' and 'index' (H, W)."""
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    out, i = {}, 0
    while inp.seek_subimage(i, 0):
        names = ",".join(inp.spec().channelnames)
        arr = inp.read_image(i, 0, 0, inp.spec().nchannels, oiio.FLOAT)
        if "Depth" in names:
            out["depth"] = np.ascontiguousarray(arr[..., 0], dtype=np.float32)
        elif "Object Index" in names:
            out["index"] = np.ascontiguousarray(arr[..., 0], dtype=np.float32)
        elif "Combined" in names:
            out["combined"] = arr[..., :3]
        i += 1
    inp.close()
    return out


def render_exr(sc) -> dict[str, np.ndarray]:
    bpy.ops.render.render(write_still=False)
    s = sc.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "MULTI_LAYER_IMAGE"
    s.file_format, s.color_depth, s.exr_codec = "OPEN_EXR_MULTILAYER", "32", "ZIP"
    path = Path(tempfile.gettempdir()) / f"stbench_driver_{os.getpid()}.exr"
    bpy.data.images["Render Result"].save_render(str(path), scene=sc)
    out = read_exr(path)
    path.unlink()
    return out


def cpu_scene(size: int = 8, samples: int = 32):
    sc = fresh()
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.cycles.device, sc.cycles.samples, sc.cycles.use_denoising = "CPU", samples, False
    sc.view_settings.view_transform = "Raw"
    vl = sc.view_layers[0]
    vl.use_pass_combined = vl.use_pass_z = vl.use_pass_object_index = True
    return sc


def plane(sc, size: float, z: float, albedo: float):
    me = bpy.data.meshes.new("plane")
    h = size / 2
    me.from_pydata([(-h, -h, z), (h, -h, z), (h, h, z), (-h, h, z)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("plane", me)
    sc.collection.objects.link(ob)
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    p = mat.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (albedo, albedo, albedo, 1)
    p.inputs["Roughness"].default_value = 1.0
    p.inputs["Specular IOR Level"].default_value = 0.0
    me.materials.append(mat)
    return ob


def camera(sc, pos, look_at, ortho: float | None = None, fov_deg: float = 20.0):
    cd = bpy.data.cameras.new("cam")
    if ortho:
        cd.type, cd.ortho_scale = "ORTHO", ortho
    else:
        cd.angle = math.radians(fov_deg)
    ob = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(ob)
    ob.location = pos
    ob.rotation_euler = (Vector(look_at) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    sc.camera = ob
    return ob


def black_world(sc):
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0


def centre(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    return a[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1].reshape(-1, a.shape[-1]).mean(axis=0)


# -- physics ---------------------------------------------------------------------------------------------------------
def physics() -> dict:
    out = {}
    cfg = {"shape": "SQUARE", "size_m": 0.36, "spread_deg": 180.0}
    # a 10 W lamp 2 m over a white card: the card's radiance is P / (pi^2 h^2) if a lamp's watts are its flux
    sc = cpu_scene()
    black_world(sc)
    plane(sc, 20.0, 0.0, 1.0)
    camera(sc, (0, 0, 1.0), (0, 0, 0), ortho=0.05)
    lamp = lighting._area_light(sc, "lamp", cfg, (0, 0, 2.0), 10.0, 6500.0, None)
    lamp.data.use_temperature = False  # a white lamp
    out["white_card_radiance_10w_2m"] = float(centre(render_exr(sc)["combined"])[1])
    # a grey card under a 3000 K lamp, balanced to 3000 K, to 3250 K and to 6500 K: r/g and b/g of what it reads
    for balance in (3000.0, 3250.0, 6500.0):
        sc = cpu_scene()
        black_world(sc)
        plane(sc, 20.0, 0.0, 0.18)
        camera(sc, (0, 0, 1.0), (0, 0, 0), ortho=0.05)
        lighting._area_light(sc, "lamp", cfg, (0, 0, 2.0), 10.0, 3000.0, balance)
        rgb = centre(render_exr(sc)["combined"])
        out[f"grey_under_3000k_balanced_to_{int(balance)}"] = {
            "r_g": float(rgb[0] / rgb[1]), "b_g": float(rgb[2] / rgb[1]),
            "luminance": float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])}
    # an HDRI with a bright block at +x on the horizon (azimuth 0 from +x, `atan2(y, x)`): which way does it face once
    # the picture is turned by `heading_deg`?
    img = bpy.data.images.new("hdri_test", 128, 64, float_buffer=True, alpha=False)
    px = np.full((64, 128, 4), 0.2, dtype=np.float32)
    px[:, :, 3] = 1.0
    px[24:40, 60:68] = 10.0  # u = 0.5 is +x in Blender's equirectangular mapping, v = 0.5 the horizon
    img.pixels.foreach_set(px.ravel())
    path = Path(tempfile.gettempdir()) / f"stbench_hdri_{os.getpid()}.exr"
    img.file_format, img.filepath_raw = "OPEN_EXR", str(path)
    img.save()
    for heading in (0.0, 90.0, -90.0):
        row = {}
        for name, d in (("+x", (1, 0, 0)), ("-y", (0, -1, 0)), ("+y", (0, 1, 0))):
            sc = cpu_scene()
            sky = {"elevation_deg": 90.0, "rotation_deg": 0.0, "strength": 0.0, "sky": 0.0}
            lighting_sky.build(sc, sky, None, path, {"camera_strength": 1.0, "light_strength": 1.0,
                                                     "heading_deg": heading})
            camera(sc, (0, 0, 0), d, fov_deg=10.0)
            row[name] = float(centre(render_exr(sc)["combined"]).mean())
        out[f"hdri_heading_{int(heading)}"] = row
    path.unlink()
    w = sc.world.node_tree
    out["hdri_world"] = sorted(n.bl_idname for n in w.nodes)
    out["hdri_clamp"] = lighting_sky.SUN_CLAMP
    # a sky set by its horizontal illuminance: a picture of 0.5 above the horizon and 5.0 below it (its first rows are
    # the bottom), at `sky_lux` 11,760, lights a small card facing up with that many lux
    img = bpy.data.images.new("sky_lux_test", 128, 64, float_buffer=True, alpha=False)
    px = np.full((64, 128, 4), 0.5, dtype=np.float32)
    px[:32, :, :3] = 5.0
    px[:, :, 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    path = Path(tempfile.gettempdir()) / f"stbench_sky_lux_{os.getpid()}.exr"
    img.file_format, img.filepath_raw = "OPEN_EXR", str(path)
    img.save()
    sc = cpu_scene()
    sky = {"elevation_deg": 90.0, "rotation_deg": 0.0, "strength": 0.0, "sky": 0.0}
    lighting_sky.build(sc, sky, None, path, {"sky_lux": 11760.0})
    plane(sc, 0.1, 0.0, 0.18)
    camera(sc, (0, 0, 1.0), (0, 0, 0), ortho=0.05)
    rgb = centre(render_exr(sc)["combined"])
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    out["sky_lux_card"] = float(luminance / 0.18 * cycles_setup.UNIT_LUX)
    path.unlink()
    return out


# -- passes ----------------------------------------------------------------------------------------------------------
def passes() -> dict:
    """A box room with a lamp, two entities with pass indices and a glossy ball, rendered from two cameras with the
    settings of the assets appearance before Phase 1 and with `cycles_setup.configure`'s."""
    sc = fresh()
    sc.render.resolution_x, sc.render.resolution_y = 96, 72
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3
    for z, size in ((0.0, 6.0), (2.5, 6.0)):
        plane(sc, size, z, 0.6)
    for i, (x, y, s) in enumerate(((0.8, 0.4, 0.5), (-0.7, 1.2, 0.7)), start=1):
        bpy.ops.mesh.primitive_cube_add(size=s, location=(x, y, s / 2))
        ob = bpy.context.object
        ob.pass_index = i
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0.0, -0.8, 0.3))
    ball = bpy.context.object
    ball.pass_index = 3
    mat = bpy.data.materials.new("glossy")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.1
    ball.data.materials.append(mat)
    ld = bpy.data.lights.new("lamp", "AREA")
    ld.size, ld.energy = 0.6, 200.0
    lo = bpy.data.objects.new("lamp", ld)
    sc.collection.objects.link(lo)
    lo.location = (0.0, 0.0, 2.4)
    fresh_cycles = {"denoising_input_passes": sc.cycles.denoising_input_passes}
    cycles_setup.configure(sc, 96, 72, 128, "assets")
    keys = ("samples", "use_adaptive_sampling", "max_bounces", "sample_clamp_indirect")
    new = {k: getattr(sc.cycles, k) for k in keys}
    runs = {}
    for label in ("assets_now", "before"):
        if label == "before":  # the settings of `configure` for the assets appearance before Phase 1
            cy = sc.cycles
            cy.samples, cy.use_adaptive_sampling = 64, False
            cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces, cy.transmission_bounces = 8, 4, 4, 8
            cy.transparent_max_bounces, cy.sample_clamp_indirect = 8, 8.0
            cy.denoising_input_passes = fresh_cycles["denoising_input_passes"]
        shots = []
        for pos in ((2.0, -2.2, 1.4), (-2.2, -1.5, 1.0)):
            camera(sc, pos, (0.0, 0.3, 0.4), fov_deg=70.0)
            shots.append(render_exr(sc))
        runs[label] = shots
    same = {"depth": True, "index": True}
    for a, b in zip(runs["assets_now"], runs["before"], strict=True):
        same["depth"] &= bool(a["depth"].tobytes() == b["depth"].tobytes())
        same["index"] &= bool(a["index"].tobytes() == b["index"].tobytes())
    now, before = runs["assets_now"], runs["before"]
    rgb = [float(np.abs(a["combined"] - b["combined"]).mean()) for a, b in zip(now, before, strict=True)]
    indices = sorted({int(v) for shot in now for v in np.unique(shot["index"])})
    return {"identical": same, "indices_seen": indices, "rgb_mean_abs_diff": rgb, "settings_now": new,
            "depth_range": [float(runs["assets_now"][0]["depth"].min()), float(runs["assets_now"][0]["depth"].max())]}


# -- scene -----------------------------------------------------------------------------------------------------------
def boxes_of(prefix: str) -> list[dict]:
    """World-space box of every object named `prefix*`."""
    out = []
    for ob in sorted((o for o in bpy.data.objects if o.name.startswith(prefix)), key=lambda o: o.name):
        co = np.array([(ob.matrix_world @ v.co)[:] for v in ob.data.vertices])
        out.append({"name": ob.name, "min": r(co.min(axis=0)), "max": r(co.max(axis=0)), "pass_index": ob.pass_index})
    return out


def scene() -> dict:
    import build_scene

    from stbench import asset_manifest
    from stbench.mapspec import load_map
    from stbench.timeline import load

    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    hfov = json.loads((args.root / "sensors" / "camera.json").read_text())["hfov_deg"]
    built = build_scene.build(tl, m, hfov, (asset_manifest.load(args.manifest), args.library))
    bpy.context.view_layer.update()  # the objects' world matrices, for the boxes
    lights = []
    for ob in bpy.data.objects:
        if ob.type != "LIGHT":
            continue
        ld = ob.data
        row = {"name": ob.name, "type": ld.type, "energy": ld.energy, "loc": r(ob.location),
               "dir": r(ob.rotation_euler.to_matrix() @ Vector((0, 0, -1)))}
        if ld.type == "AREA":
            row |= {"shape": ld.shape, "size": ld.size, "size_y": ld.size_y, "spread": math.degrees(ld.spread),
                    "portal": bool(ld.cycles.is_portal), "temperature": ld.temperature,
                    "use_temperature": ld.use_temperature, "colour": r(ld.color)}
        lights.append(row)
    nodes = [n for w in bpy.data.worlds for n in w.node_tree.nodes]
    env = next((n for n in nodes if n.bl_idname == "ShaderNodeTexEnvironment"), None)
    mats = {o.name: [s.material.name for s in o.material_slots if s.material]
            for o in bpy.data.objects if o.type == "MESH"}
    vs = bpy.context.scene.view_settings
    return {"objects": sorted(o.name for o in bpy.data.objects), "lights": lights,
            "world_nodes": sorted(n.bl_idname for n in nodes), "hdri": None if env is None else env.image.name,
            "ground_materials": sorted(o for o, ms in mats.items() if any(x == "ground" for x in ms)),
            "ceiling_lamps": boxes_of("dressing_ceiling_lamp_"), "high_bays": boxes_of("dressing_high_bay_light_"),
            "upper_cabinets": boxes_of("dressing_upper_cabinet_"),
            "skylights": boxes_of("dressing_skylight_"), "lighting": built.lighting,
            "skylight_materials": {n: ms for n, ms in mats.items() if n.startswith("dressing_skylight_")},
            "view": {"exposure": vs.exposure, "white_balance": vs.use_white_balance}}


def direct_only(sc, points: list[dict]) -> list[dict]:
    """`lighting_probe.measure` of the lamps alone with Cycles' bounces set to none: each card is lit by the lamps it
    sees, in their own colour, and not by the light the walls, floor and ceiling throw back tinted by their paint."""
    cy, bounces = sc.cycles, sc.cycles.max_bounces
    cy.max_bounces = 0
    try:
        return lighting_probe.measure(sc, points, sources="lamps")
    finally:
        cy.max_bounces = bounces


def lux() -> dict:
    """Grey cards (`lighting_probe`) in the built scene: a 3 x 3 grid in every room, or the `--points` file's points, or
    the cards on every furniture surface of the `--surface` kind, lit by the `--sources` (lamps, daylight or all). The
    room grid's cards also report `direct`, their lux and cast with no bounce (`direct_only`)."""
    import build_scene

    from stbench import asset_manifest
    from stbench.mapspec import load_map
    from stbench.timeline import load

    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    hfov = json.loads((args.root / "sensors" / "camera.json").read_text())["hfov_deg"]
    built = build_scene.build(tl, m, hfov, (asset_manifest.load(args.manifest), args.library))
    built.apply_time(0.0)
    sc = bpy.context.scene
    if args.points:
        points = json.loads(args.points.read_text(encoding="utf-8"))
    elif args.surface:
        points = lighting_probe.surface_points(m, args.surface)
    else:
        points = lighting_probe.card_points(m)
    cards = lighting_probe.measure(sc, points, sources=args.sources)
    if args.points or args.surface:
        return {"unit_lux": cycles_setup.UNIT_LUX, "cards": cards}
    rooms: dict[str, list[dict]] = {}
    for card, lit in zip(cards, direct_only(sc, points), strict=True):
        card["direct"] = {k: lit[k] for k in ("lux", "r_g", "b_g")}
        rooms.setdefault(card["room"], []).append(card)
    return {"unit_lux": cycles_setup.UNIT_LUX, "rooms": rooms}


RESULT = {"physics": physics, "passes": passes, "scene": scene, "lux": lux}[args.task]()
args.out.write_text(json.dumps(RESULT, indent=1), encoding="utf-8")
