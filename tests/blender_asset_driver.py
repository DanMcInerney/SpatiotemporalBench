# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_render_assets_blender.py`: checks the asset library and the entities built from it,
then prints one JSON document that the test reads. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_asset_driver.py -- \
        --render DIR --src DIR --manifest M --library L.blend --out RESULT.json

Renders use Cycles on the CPU at a few pixels and a few samples: the point is what the pixels hold, not their beauty.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--manifest", "--library", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import asset_entities as AE  # noqa: E402
import materials as M  # noqa: E402

from stbench import asset_manifest  # noqa: E402

RESULT: dict = {}
ATLAS_RAMP = ((60, 90), (90, 120))  # the lit quarter of the half-used atlas, in orange (a grey ramp times ATLAS_TINT)
ATLAS_TINT = (1.0, 0.6, 0.3)


def fresh() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def library_meshes() -> dict:
    """What the library file holds, read straight from the .blend (not from the JSON the build wrote)."""
    fresh()
    with bpy.data.libraries.load(str(args.library), link=False) as (src, dst):
        dst.meshes = list(src.meshes)
    out = {}
    for me in dst.meshes:
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        out[me.name] = {"min": co.min(axis=0).tolist(), "max": co.max(axis=0).tolist(), "polygons": len(me.polygons),
                        "materials": [m.name for m in me.materials], "slot_polygons": slot_polygons(me)}
    return out


def slot_polygons(me) -> list[int]:
    """How many polygons use each material slot (an index past the last slot makes the list longer)."""
    idx = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", idx)
    return np.bincount(idx, minlength=max(1, len(me.materials))).tolist()


def procedural_slots() -> dict:
    """The polygons per material slot of every procedural builder, as `procedural.build` makes it."""
    import procedural

    fresh()
    out = {}
    for name in sorted(procedural.BUILDERS):
        me, slots = procedural.build(name, f"probe:{name}")
        idx = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get("material_index", idx)
        out[name] = {"slots": list(slots), "polygons": np.bincount(idx, minlength=len(slots)).tolist()}
    return out


def tint_targets(ob) -> list:
    """The tint group's Target of each slot that has one, as [r, g, b]."""
    out = []
    for slot in ob.material_slots:
        node = slot.material.node_tree.nodes.get(M.TINT_GROUP)
        out.append(None if node is None else [round(float(c), 5) for c in node.inputs["Target"].default_value[:3]])
    return out


def sharing(app) -> dict:
    size = (0.2, 0.1, 0.1)
    a, b, c = (app.make_object("cube_a", "mug", colour, size, f"mug_{i}", i + 1)
               for i, colour in enumerate(("brown", "brown", "blue")))
    oak = app.make_object("chair_a", "chair", "oak", (0.45, 0.45, 0.9), "chair_oak", 4)
    black = app.make_object("chair_a", "chair", "black", (0.45, 0.45, 0.9), "chair_black", 5)
    return {
        "mesh_shared": a.data is b.data is c.data,
        "materials_shared_when_identical": all(x.material is y.material
                                               for x, y in zip(a.material_slots, b.material_slots, strict=True)),
        "materials_differ_when_colour_differs": all(x.material is not y.material
                                                    for x, y in zip(a.material_slots, c.material_slots, strict=True)),
        "targets_brown": tint_targets(a), "targets_blue": tint_targets(c),
        "mesh_shared_across_assets": a.data is oak.data,
        "chair_oak_material": [s.material.name for s in oak.material_slots],
        "chair_black_targets": tint_targets(black), "chair_scale": [round(v, 6) for v in black.scale],
        "pass_indices": [o.pass_index for o in (a, b, c, oak, black)],
        "materials_per_object_slot": all(s.link == "OBJECT" for s in a.material_slots),
    }


def cpu_scene(width: int, height: int, passes: bool) -> bpy.types.Scene:
    sc = fresh()
    r, cy = sc.render, sc.cycles
    r.engine, cy.device, cy.samples, cy.use_denoising = "CYCLES", "CPU", 4, False
    r.resolution_x, r.resolution_y, r.resolution_percentage = width, height, 100
    cy.seed, cy.use_animated_seed = 0, False
    sc.view_settings.view_transform = "Raw"
    vl = sc.view_layers[0]
    vl.use_pass_combined, vl.use_pass_z, vl.use_pass_object_index = True, passes, passes
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value, bg.inputs["Strength"].default_value = (1.0, 1.0, 1.0, 1.0), 1.0
    return sc


def camera(sc, eye, target, angle_deg=None, ortho=None) -> None:
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.location = eye
    cam.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
    if ortho:
        cam.data.type, cam.data.ortho_scale = "ORTHO", ortho
    else:
        cam.data.angle = math.radians(angle_deg)


def render_passes(sc, path: Path) -> dict:
    s = sc.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "MULTI_LAYER_IMAGE"
    s.file_format, s.color_depth = "OPEN_EXR_MULTILAYER", "32"
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    found, i = {}, 0
    while inp.seek_subimage(i, 0):
        spec = inp.spec()
        data = inp.read_image(i, 0, 0, spec.nchannels, oiio.FLOAT)
        for c, name in enumerate(spec.channelnames):
            found[name] = data[..., c]
        i += 1
    inp.close()
    return found


def flat_swatch(ob, sc, path: Path) -> list[float]:
    """The mean linear RGB of a straight-down view of an object's top face (z = 0.1), purely diffuse and flat lit."""
    for slot in ob.material_slots:  # purely diffuse: no sheen of the white world on the swatch
        p = M.principled(slot.material)
        p.inputs["Specular IOR Level"].default_value, p.inputs["Roughness"].default_value = 0.0, 1.0
    camera(sc, (0.0, 0.0, 0.5), (0.0, 0.0, 0.1), ortho=0.18)
    found = render_passes(sc, path)
    return [float(found[f"ViewLayer.Combined.{c}"].mean()) for c in "RGB"]


def pass_index_and_swatch(m, scratch: Path) -> dict:
    """A close-up of the cube's top and front faces, which fill the frame: every pixel must carry the entity's pass
    index. Then a straight-down view of the top face, flat lit, whose mean is the tinted albedo."""
    sc = cpu_scene(32, 24, passes=True)
    ob = AE.Appearance(m, args.library, ["cube_a"]).make_object("cube_a", "mug", "brown", (0.2, 0.1, 0.1), "mug", 7)
    camera(sc, (0.0, -0.11, 0.2), (0.0, -0.05, 0.1), angle_deg=20)  # on the top-front edge: the frame is all cube
    found = render_passes(sc, scratch / "close.exr")
    index = found["ViewLayer.Object Index.X"]
    out = {"index_values": sorted({float(v) for v in np.unique(index)}),
           "index_all_seven": bool((index == 7.0).all()), "depth_max": float(found["ViewLayer.Depth.Z"].max())}
    sc = cpu_scene(64, 32, passes=False)
    ob = AE.Appearance(m, args.library, ["cube_a"]).make_object("cube_a", "mug", "brown", (0.2, 0.1, 0.1), "mug2", 7)
    out["swatch_mean_rgb"] = flat_swatch(ob, sc, scratch / "swatch.exr")
    node = ob.material_slots[0].material.node_tree.nodes[M.TINT_GROUP]
    out["top_slot"] = ob.material_slots[0].material.name
    out["top_mean_input"] = float(node.inputs["Mean"].default_value)
    out["top_target"] = [float(c) for c in node.inputs["Target"].default_value[:3]]
    return out


def half_atlas(scratch: Path) -> dict:
    """`finish_slots` on a cube whose top face (slot "top") maps to the lit bottom-left quarter of a 4 x 4 atlas and
    whose other faces (slot "side") map to its black top-right quarter, as a scanned object's UVs leave most of its
    atlas unused. Returns the slots' report, the atlas as stored (rows from the bottom) and a flat-lit view of the top
    face wearing "brown"."""
    import asset_library as AL

    sc = cpu_scene(64, 32, passes=False)
    img = bpy.data.images.new("atlas", 4, 4)
    px = np.zeros((4, 4, 4), np.float32)
    px[..., 3] = 1.0
    for r, row in enumerate(ATLAS_RAMP):  # rows are numbered from the bottom, as Blender stores them
        for c, grey in enumerate(row):
            px[r, c, :3] = np.array(ATLAS_TINT) * grey / 255.0
    img.pixels.foreach_set(px.ravel())
    readback = np.empty(4 * 4 * 4, np.float32)
    img.pixels.foreach_get(readback)
    mesh = bpy.data.meshes.new("half")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=0.2)
    uv = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        top = f.normal.z > 0.5
        for loop, (u, v) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
            loop[uv].uv = (0.5 * u, 0.5 * v) if top else (0.5 + 0.5 * u, 0.5 + 0.5 * v)
        f.material_index = 0 if top else 1
    bm.to_mesh(mesh)
    bm.free()
    top_mat, side_mat = M.new_material("half_top"), M.new_material("half_side")
    tex = top_mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image, tex.interpolation = img, "Closest"  # a 4 x 4 atlas: no blending of the lit texels with the black
    top_mat.node_tree.links.new(tex.outputs["Color"], M.principled(top_mat).inputs["Base Color"])
    asset = types.SimpleNamespace(id="half", tint={"mode": "luminance", "materials": ["top"]})
    report = AL.finish_slots(asset, mesh, [top_mat, side_mat], ["top", "side"])
    M.set_tint_target(top_mat, AE.linear("brown"))
    ob = bpy.data.objects.new("half", mesh)
    sc.collection.objects.link(ob)
    node = top_mat.node_tree.nodes[M.TINT_GROUP]
    return {"report": report, "atlas": np.rint(readback.reshape(4, 4, 4)[..., :3] * 255).astype(int).tolist(),
            "tint_mean_input": float(node.inputs["Mean"].default_value),
            "swatch_mean_rgb": flat_swatch(ob, sc, scratch / "half.exr")}


def frosted_glass(m, scratch: Path) -> dict:
    """A tinted frosted glass and the same glass made opaque, each on a brown floor under the white world: the mean
    linear RGB of the glass's pixels (the object-index mask), for a blue and a green glass."""
    out = {}
    for colour in ("blue", "green"):
        row = {}
        for label, weight in (("glass", None), ("opaque", 0.0)):
            sc = cpu_scene(96, 128, passes=True)
            sc.cycles.samples = 24
            floor = bpy.data.objects.new("floor", bpy.data.meshes.new("floor"))
            bm = bmesh.new()
            bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
            bm.to_mesh(floor.data)
            bm.free()
            sc.collection.objects.link(floor)
            brown = M.new_material("floor")
            M.principled(brown).inputs["Base Color"].default_value = (0.25, 0.12, 0.05, 1.0)
            floor.data.materials.append(brown)
            ob = AE.Appearance(m, args.library, ["proc_glass"]).make_object("proc_glass", "glass", colour,
                                                                           (0.07, 0.07, 0.12), "glass", 5)
            if weight is not None:
                for slot in ob.material_slots:
                    M.principled(slot.material).inputs["Transmission Weight"].default_value = weight
            camera(sc, (0.0, -0.32, 0.17), (0.0, 0.0, 0.06), angle_deg=24)
            found = render_passes(sc, scratch / f"glass_{colour}_{label}.exr")
            mask = found["ViewLayer.Object Index.X"] == 5
            row[label] = [float(found[f"ViewLayer.Combined.{c}"][mask].mean()) for c in "RGB"]
            row["pixels"] = int(mask.sum())
        out[colour] = row
    return out


def laptop_shape() -> dict:
    """The open laptop: the extents of its geometry before the exact fit to `DESIGN_BOX`, and the screen slot's faces
    (area and normal) after it."""
    import procedural

    fresh()
    bm = bmesh.new()
    procedural.BUILDERS["laptop"](bm, procedural.DESIGN_BOX["laptop"])
    co = np.array([v.co[:] for v in bm.verts])
    bm.free()
    mesh, slots = procedural.build("laptop", "probe:laptop")
    screen = slots.index("screen")
    return {"natural": (co.max(axis=0) - co.min(axis=0)).tolist(), "design": list(procedural.DESIGN_BOX["laptop"]),
            "screen": [{"area": p.area, "normal": list(p.normal)} for p in mesh.polygons if p.material_index == screen]}


def mover_extents() -> dict:
    """Per packaged mover kind with a procedural builder (the resident's body is made by a recipe, so it has none): the
    extents of its builder's geometry before the exact fit, and its `DESIGN_BOX`."""
    import procedural

    fresh()
    out = {}
    for a in asset_manifest.packaged().assets:
        if a.mobility == "mover" and a.builder:
            bm = bmesh.new()
            procedural.BUILDERS[a.builder](bm, procedural.DESIGN_BOX[a.builder])
            co = np.array([v.co[:] for v in bm.verts])
            bm.free()
            out[a.kind] = {"natural": (co.max(axis=0) - co.min(axis=0)).tolist(),
                           "design": list(procedural.DESIGN_BOX[a.builder])}
    return out


def structure() -> dict:
    """The placeholder structure of the apartment and of the warehouse (`build_scene._build_structure`): the energy of
    every ceiling light and the material on every region's floor."""
    import build_scene
    import shapes

    from stbench.mapspec import load_map

    out = {}
    for name in ("apartment_a", "warehouse_a"):
        fresh()
        shapes.reset()
        m = load_map(name)
        build_scene._build_structure(m)
        floors = {r.name: bpy.data.objects[f"floor_{r.id}"].data.materials[0] for r in m.regions}
        lights = sorted(bpy.data.lights, key=lambda x: int(x.name.rsplit("_", 1)[1]))  # ceiling_light_0, _1, ...
        out[name] = {"watts": [round(ld.energy, 4) for ld in lights],
                     "floors": {region: mat.name for region, mat in floors.items()},
                     "floor_materials": len({id(mat) for mat in floors.values()})}
    return out


def floors(m) -> dict:
    """What `Appearance.floor_material` gives: a library material for a texture floor, one shared procedural material
    for a `shader:` floor, nothing for a region the manifest does not name."""
    app = AE.Appearance(m, args.library, ["cube_a"])
    dock, again, kitchen = app.floor_material("dock"), app.floor_material("dock"), app.floor_material("kitchen")
    return {"dock": dock.name, "shared": dock is again, "kitchen": kitchen.name, "other": app.floor_material("garage")}


def concrete_floor(scratch: Path) -> dict:
    """A top-down view of the concrete slab, flat lit and purely diffuse, over a joint (x = 0) and over open slab
    (x = 3): the mean albedo of the slab and the width of the dark joint."""
    out = {}
    for label, x in (("joint", 0.0), ("slab", 3.0)):
        sc = cpu_scene(160, 8, passes=False)
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=20.0)
        floor = bpy.data.objects.new("floor", bpy.data.meshes.new("floor"))
        bm.to_mesh(floor.data)
        bm.free()
        sc.collection.objects.link(floor)
        mat = M.concrete_material()
        p = M.principled(mat)
        p.inputs["Specular IOR Level"].default_value, p.inputs["Roughness"].default_value = 0.0, 1.0
        floor.data.materials.append(mat)
        camera(sc, (x, 3.0, 1.0), (x, 3.0, 0.0), ortho=0.32)  # 2 mm a pixel
        found = render_passes(sc, scratch / f"concrete_{label}.exr")
        lum = np.mean([found[f"ViewLayer.Combined.{c}"] for c in "RGB"], axis=0)
        out[label] = lum.mean(axis=0).tolist()
    return out


def rack_stock() -> dict:
    """For each pallet rack of the warehouse map: the (bay, level) of every carton `furniture_warehouse.pallet_rack`
    draws and of every wire deck, and the same for a rack given no surface at all; and the decks' tops and thickness."""
    import furniture_warehouse as FW
    import rack_layout as RL

    from stbench.mapspec import load_map

    def cells(f, surfaces) -> list[list[int]]:
        found = {(int((p.loc[1] + f.size[1] / 2) // RL.BAY_M), int(p.loc[2] // RL.LEVEL_M))
                 for p in FW.pallet_rack(f.size, f.color, surfaces) if p.mat.startswith("beige")}
        return sorted([b, v] for b, v in found)

    def decks(f, surfaces) -> list[tuple]:
        """(bay, level, top z, thickness, x0, x1, y0, y1) of each deck."""
        out = []
        for p in FW.pallet_rack(f.size, f.color, surfaces):
            if p.mat == FW.DECK:
                top = p.loc[2] + p.dims[2] / 2
                out.append((int((p.loc[1] + f.size[1] / 2) // RL.BAY_M), round(top / RL.LEVEL_M), top, p.dims[2],
                            p.loc[0] - p.dims[0] / 2, p.loc[0] + p.dims[0] / 2, p.loc[1] - p.dims[1] / 2,
                            p.loc[1] + p.dims[1] / 2))
        return sorted(out)

    out = {}
    for f in load_map("warehouse_a").furniture:
        if f.kind == "pallet_rack":
            surfaces = [(*s.rect, s.z) for s in f.surfaces]
            out[f.id] = {"stock": cells(f, surfaces), "bare": cells(f, []),
                         "surfaces": [[s.rect[1], s.z] for s in f.surfaces], "size": list(f.size),
                         "decks": decks(f, surfaces), "bare_decks": decks(f, [])}
    return out


def main() -> None:
    m = asset_manifest.load(args.manifest)
    RESULT["library_meshes"] = library_meshes()
    RESULT["procedural"] = procedural_slots()
    RESULT["laptop"] = laptop_shape()
    RESULT["mover_extents"] = mover_extents()
    fresh()
    RESULT["sharing"] = sharing(AE.Appearance(m, args.library, ["cube_a", "chair_a", "proc_glass"]))
    RESULT["render"] = pass_index_and_swatch(m, args.out.parent)
    RESULT["half_atlas"] = half_atlas(args.out.parent)
    RESULT["frosted_glass"] = frosted_glass(m, args.out.parent)
    RESULT["palette_black"] = list(AE.linear("black"))
    RESULT["structure"] = structure()
    RESULT["floors"] = floors(m)
    RESULT["concrete"] = concrete_floor(args.out.parent)
    RESULT["rack_stock"] = rack_stock()
    args.out.write_text(json.dumps(RESULT), encoding="utf-8")


main()
