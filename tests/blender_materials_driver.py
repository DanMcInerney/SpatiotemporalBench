# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_render_materials_blender.py`: bevels, the named materials, the warehouse floor's
markings and the wear of a built scene, printed as one JSON document that the test reads. Not a pytest module (Blender's
Python has no pytest).

    blender --background --factory-startup --python tests/blender_materials_driver.py -- \
        --render DIR --src DIR --private EPISODE_PRIVATE --cases CASES.json --out RESULT.json

Renders use Cycles on the CPU at 100 x 100 pixels and 8 samples, under a white sky and no lamp, so a surface's pixel is
its albedo: the point is what the pixels hold, not their beauty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--private", "--cases", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import build_scene  # noqa: E402
import materials  # noqa: E402
import shapes  # noqa: E402
import wear  # noqa: E402

from stbench.mapspec import load_map  # noqa: E402
from stbench.timeline import load  # noqa: E402

RESULT: dict = {}
CASES = json.loads(args.cases.read_text(encoding="utf-8"))
SCRATCH = args.out.parent


def fresh() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    shapes.reset()
    return bpy.context.scene


# -- bevels --------------------------------------------------------------------------------------------------------
def bevels() -> dict:
    """Per case of boxes (as `Part` dicts): the mesh's vertices and, per polygon, its centre and material slot."""
    out = {}
    for name, parts in CASES["bevel"].items():
        fresh()
        me = shapes.entity_mesh(name, [shapes.Part(p["shape"], tuple(p["loc"]), tuple(p["dims"]), tuple(p["rot"]),
                                                   p["mat"], (), p["bevel"], p["segments"]) for p in parts])
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        out[name] = {"verts": np.round(co.reshape(-1, 3), 6).tolist(),
                     "slots": [p.material_index for p in me.polygons],
                     "centres": [[round(x, 6) for x in p.center] for p in me.polygons],
                     "materials": [m.name for m in me.materials]}
    return out


# -- the named materials -------------------------------------------------------------------------------------------
def upstream_types(socket) -> set:
    """The node types upstream of an input socket."""
    todo, seen, found = [socket], set(), set()
    while todo:
        s = todo.pop()
        for lk in s.links:
            n = lk.from_node
            if n.name in seen:
                continue
            seen.add(n.name)
            found.add(n.bl_idname)
            todo += list(n.inputs)
    return found


def named_materials() -> dict:
    fresh()
    out = {}
    for name in materials.DRESSING_NAMES:
        mat = materials.named(name)
        p = materials.principled(mat)
        out[name] = {"nodes": len(mat.node_tree.nodes), "same": materials.named(name) is mat,
                     "emission_upstream": sorted(upstream_types(p.inputs["Emission Strength"])),
                     "emission_strength": float(p.inputs["Emission Strength"].default_value),
                     "alpha": float(p.inputs["Alpha"].default_value), "name": mat.name}
    try:
        materials.named("no_such_material")
        unknown = None
    except KeyError as e:
        unknown = str(e)
    out["_unknown_error"] = unknown
    out["_dressing_none"] = materials.dressing_material("no_such_material") is None
    out["_shader_name"] = materials.named("concrete").name
    film = materials.principled(materials.shader_material("stretch_film"))
    out["_stretch_film"] = {"base": [round(float(v), 4) for v in film.inputs["Base Color"].default_value],
                            "base_linked": bool(film.inputs["Base Color"].links),
                            "roughness": round(float(film.inputs["Roughness"].default_value), 4),
                            "transmission": float(film.inputs["Transmission Weight"].default_value),
                            "alpha": float(film.inputs["Alpha"].default_value),
                            "bump": bool(film.inputs["Normal"].links)}
    fresh()  # a rebuilt scene frees every material: the cache must not hand out a freed one
    out["_after_reset"] = materials.named("plaster_wall").name
    return out


# -- the floor ----------------------------------------------------------------------------------------------------
def slab_scene(spec: dict | None) -> None:
    sc = fresh()
    materials.use_map(spec)
    r, cy = sc.render, sc.cycles
    r.engine, cy.device, cy.samples, cy.use_denoising = "CYCLES", "CPU", 8, False
    r.resolution_x = r.resolution_y = 100
    r.resolution_percentage = 100
    cy.seed, cy.use_animated_seed = 0, False
    sc.view_settings.view_transform = "Raw"
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value, bg.inputs["Strength"].default_value = (1.0, 1.0, 1.0, 1.0), 1.0
    mat = materials.concrete_material()
    materials.principled(mat).inputs["Specular IOR Level"].default_value = 0.0  # purely diffuse
    shapes.link("floor", shapes.polygon_mesh("floor", [(0, 0), (17.6, 0), (17.6, 20.0), (0, 20.0)], 0.0, mat))
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 1.0  # 1 cm a pixel
    cam.rotation_euler = (Vector((0, 0, -1))).to_track_quat("-Z", "Y").to_euler()


def look(x: float, y: float, name: str, scale: float = 1.0) -> np.ndarray:
    """Linear RGB (100, 100, 3) of the floor straight down at (x, y), `scale` m across."""
    sc = bpy.context.scene
    cam = sc.camera
    cam.data.ortho_scale = scale
    cam.location = (x, y, 1.0)
    path = SCRATCH / f"{name}.exr"
    s = sc.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "IMAGE"
    s.file_format, s.color_depth, s.color_mode = "OPEN_EXR", "32", "RGB"
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)[..., :3]
    return px[::-1]  # Blender's first row is the bottom; this has y falling down the image, x rising along it


def yellow(px: np.ndarray) -> np.ndarray:
    """Per pixel: red well above blue (a grey slab has them equal)."""
    return px[..., 0] > 2.5 * np.maximum(px[..., 2], 1e-4)


def floor_views(spec: dict) -> dict:
    out = {}
    slab_scene(spec)
    a = look(1.5, 14.0, "line")  # across the first rack line (x 1.45 to 1.55)
    out["line_yellow_cols"] = yellow(a).mean(axis=0).round(3).tolist()  # per column: the line is 10 columns wide
    b = look(3.0, 14.0, "bare", 0.6)  # open slab between the two streaks of the west aisle
    out["bare_rgb_mean"] = b.reshape(-1, 3).mean(axis=0).tolist()
    out["bare_yellow"] = float(yellow(b).mean())
    c = look(3.7, 3.95, "hatch", 0.6)  # inside the first door's apron
    out["hatch_yellow"] = float(yellow(c).mean())
    t = look(2.45, 14.0, "track")  # across the west aisle's west streak (x 2.38 to 2.52) and the slab beside it
    lum = t.mean(axis=2)
    out["track_cols"] = lum.mean(axis=0).round(4).tolist()
    slab_scene(None)
    d = look(1.5, 14.0, "plain_line")
    out["plain_yellow"] = float(yellow(d).mean())
    return out


# -- a built scene --------------------------------------------------------------------------------------------------
def material_digest(mat) -> str:
    nt = mat.node_tree
    nodes = sorted([n.bl_idname, [round(float(v), 5) for i in n.inputs if hasattr(i, "default_value")
                                  for v in (i.default_value if hasattr(i.default_value, "__len__")
                                            else [i.default_value]) if isinstance(v, int | float)]] for n in nt.nodes)
    links = sorted([lk.from_node.bl_idname, lk.from_socket.identifier, lk.to_node.bl_idname, lk.to_socket.identifier]
                   for lk in nt.links)
    return hashlib.sha256(json.dumps([nodes, links]).encode()).hexdigest()[:16]


def slots(ob) -> list:
    return [{"name": m.name, "digest": material_digest(m),
             "base": [round(float(v), 5) for v in materials.principled(m).inputs["Base Color"].default_value]
             if not materials.principled(m).inputs["Base Color"].links else None} for m in ob.data.materials]


def worn_scene() -> dict:
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    spec = json.loads((args.src / "stbench" / "maps" / f"{m.name}.json").read_text(encoding="utf-8"))
    fresh()
    materials.use_map(spec)
    scene = build_scene.build(tl, m, 80.0, None)
    before = {mat.name: material_digest(mat) for mat in bpy.data.materials}
    entities = {}
    for eid, ob in scene.objects.items():
        e = tl.entity(eid)
        entities[eid] = {"kind": e.kind, "mobility": e.mobility, "colour": e.attributes.get("color"),
                         "before": slots(ob)}
    names = list(bpy.data.objects.keys())  # (a Blender collection iterates its objects, `keys` gives the names)
    structure = {n: slots(bpy.data.objects[n]) for n in names
                 if n == "walls" or re.fullmatch(r"(floor|ceiling)_R\d+", n)}
    structure_before = {n: [s["name"] for s in v] for n, v in structure.items()}
    wear.apply(scene, m)
    for eid, ob in scene.objects.items():
        entities[eid]["after"] = slots(ob)
    after = {name: material_digest(bpy.data.materials[name]) for name in before}
    unchanged = sorted(n for n in before if before[n] == after[n])
    return {"entities": entities, "structure_before": structure_before,
            "structure_after": {n: [s["name"] for s in slots(bpy.data.objects[n])] for n in structure},
            "unchanged_materials": unchanged, "materials_before": sorted(before)}


def surfaces() -> dict:
    """For each receptacle surface of the apartment's fixed furniture, built by `furniture.BUILDERS` as `build_scene`
    builds it: the heights a ray dropped from 3 cm above the surface meets at 35 sample points inside its rect (shrunk
    by 4 cm), and how far above the surface the first thing is that a ray cast up from 8 mm over it meets (or None)."""
    import furniture
    from mathutils.bvhtree import BVHTree

    out = {}
    fresh()
    m = load_map("apartment_a")
    for f in m.furniture:
        if not f.surfaces:
            continue
        zs = tuple(sorted(s.z for s in f.surfaces))
        me = shapes.entity_mesh(f.id, furniture.BUILDERS[f.kind](tuple(f.size), f.color, zs))
        tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        for s in f.surfaces:
            cx, cy, sx, sy = s.rect
            down, up = [], []
            for i in range(7):
                for j in range(5):
                    x = cx + (i / 6 - 0.5) * (sx - 0.08)
                    y = cy + (j / 4 - 0.5) * (sy - 0.08)
                    hit = tree.ray_cast(Vector((x, y, s.z + 0.03)), Vector((0, 0, -1)), 3.0)[0]
                    down.append(None if hit is None else round(hit.z, 5))
                    hit = tree.ray_cast(Vector((x, y, s.z + 0.008)), Vector((0, 0, 1)), 0.05)[0]
                    up.append(None if hit is None else round(hit.z - s.z, 5))
            out[f"{f.id}:{s.id}"] = {"kind": f.kind, "z": s.z, "down": down, "up": up}
    return out


def mask_nodes() -> dict:
    """The image nodes in a worn copy of a wall, a tile floor, a table and a rack's paint with no masks, with the
    synthetic ones the test wrote (three of the four sets) loaded by `wear.load_masks`, and with none again."""
    fresh()
    base = materials.new_material("plain")
    counts = {}

    def images(kind) -> int:
        return sum(1 for n in wear.worn(base, kind).node_tree.nodes if n.bl_idname == "ShaderNodeTexImage")

    wear.use_masks({})
    counts["none"] = {k: images(k) for k in ("wall", "tile", "furniture", "paint")}
    root = Path(CASES["masks_root"])
    loaded = wear.load_masks(lambda i: root / i)
    counts["loaded"] = sorted(loaded)
    counts["colourspace"] = sorted({img.colorspace_settings.name for img in loaded.values()})
    wear.use_masks(loaded)
    counts["with"] = {k: images(k) for k in ("wall", "tile", "furniture", "paint")}
    wear.use_masks({})
    counts["again"] = {k: images(k) for k in ("wall", "tile", "furniture", "paint")}
    return counts


def main() -> None:
    RESULT["bevel"] = bevels()
    RESULT["named"] = named_materials()
    RESULT["floor"] = floor_views(CASES["warehouse_spec"])
    RESULT["surfaces"] = surfaces()
    RESULT["masks"] = mask_nodes()
    RESULT["worn"] = worn_scene()
    args.out.write_text(json.dumps(RESULT), encoding="utf-8")


main()
