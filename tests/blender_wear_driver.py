# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_wear_home_blender.py`: what the household's look of M1's R.2 fix round does to the
materials and the furniture of the apartment, printed as one JSON document that the test reads. Not a pytest module
(Blender's Python has no pytest). The scene is the placeholder drawings of a hand-made apartment episode: nothing is
downloaded and nothing is rendered.

    blender --background --factory-startup --python tests/blender_wear_driver.py -- \
        --render DIR --src DIR --private EPISODE_PRIVATE --out RESULT.json
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
for flag in ("--render", "--src", "--private", "--out"):
    ap.add_argument(flag, type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import build_scene  # noqa: E402
import materials  # noqa: E402
import shapes  # noqa: E402
import wear  # noqa: E402
import wear_surfaces  # noqa: E402

from stbench.mapspec import load_map  # noqa: E402
from stbench.timeline import load  # noqa: E402

CABINETS = ("counter", "sink_unit", "vanity", "tv_console", "nightstand", "wardrobe")  # built by `_cabinet` or alike


def fresh() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    shapes.reset()


def with_bump(name: str) -> bpy.types.Material:
    """A material whose normal is a bump of strength 0.5 over a noise (what a library ambientCG material has)."""
    mat = materials.new_material(name)
    nt, p = mat.node_tree, materials.principled(mat)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return mat


def bump_strength(mat: bpy.types.Material) -> float:
    return round(float(next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBump").inputs["Strength"]
                       .default_value), 6)


def node_types(mat: bpy.types.Material) -> list[str]:
    return [n.bl_idname for n in mat.node_tree.nodes]


def upstream_types(socket) -> set:
    todo, seen, found = [socket], set(), set()
    while todo:
        s = todo.pop()
        for lk in s.links:
            n = lk.from_node
            if n.name not in seen:
                seen.add(n.name)
                found.add(n.bl_idname)
                todo += list(n.inputs)
    return found


# -- the library's bump, cut by surface ----------------------------------------------------------------------------
def bumps() -> dict:
    """The strength of a bump of 0.5 after `wear.worn`, per finish, in the household's look and in another catalog's."""
    out = {}
    for catalog, kinds in (("home", ("wall", "ceiling", "door", "smooth", "furniture", "fabric", "carpet", "rug")),
                           ("warehouse", ("wall", "ceiling", "furniture", "carpet"))):
        fresh()
        wear.use_catalog(catalog)
        out[catalog] = {k: bump_strength(wear.worn(with_bump(f"{catalog}_{k}"), k, (0.9, 0.04) if k in
                                                   ("door", "rug") else None)) for k in kinds}
    wear.use_catalog("")
    return out


def flat_paint() -> dict:
    """Whether the base colour of a painted wall (a material tinted to a palette colour over a texture) still reads the
    texture, in the household's look and in another catalog's."""
    out = {}
    for catalog in ("home", "warehouse"):
        fresh()
        mat = materials.new_material(f"tinted_{catalog}")
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.new("t", 4, 4)
        mat.node_tree.links.new(tex.outputs["Color"], materials.principled(mat).inputs["Base Color"])
        materials.add_tint(mat, 0.5)
        materials.set_tint_target(mat, (0.7, 0.6, 0.5))
        wear.use_catalog(catalog)
        out[catalog] = "ShaderNodeTexImage" in upstream_types(materials.principled(wear.worn(mat, "wall"))
                                                              .inputs["Base Color"])
    wear.use_catalog("")
    return out


# -- a built apartment ---------------------------------------------------------------------------------------------
def scene() -> tuple:
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    spec = json.loads((args.src / "stbench" / "maps" / f"{m.name}.json").read_text(encoding="utf-8"))
    fresh()
    materials.use_map(spec)
    sc = build_scene.build(tl, m, 80.0, None)
    return sc, tl, m


def walls(sc, m) -> dict:
    """What `wear.apply` made of the walls' material and the underlay's, in the apartment."""
    wear.apply(sc, m)
    wall = bpy.data.objects["walls"].data.materials[0]
    nodes = wall.node_tree.nodes
    paints = sorted(tuple(round(float(c), 3) for c in n.inputs[7].default_value[:3]) for n in nodes
                    if n.bl_idname == "ShaderNodeMix" and n.data_type == "RGBA" and n.blend_type == "MIX"
                    and n.inputs[0].links and not n.inputs[7].links)
    door_leaves = [ob for ob in bpy.data.objects if ob.name.startswith("door_")]
    plain = materials.new_material("plain_after")
    return {"name": wall.name, "paints": paints,
            "spots": sum(1 for n in nodes if n.bl_idname == "ShaderNodeVectorMath" and n.operation == "LENGTH"),
            "geometry_nodes": sum(1 for n in nodes if n.bl_idname == "ShaderNodeNewGeometry"),
            "underlay": bpy.data.objects["underlay"].data.materials[0].name,
            "doors": sorted({mat.name.rsplit("+", 1)[-1] for ob in door_leaves for mat in ob.data.materials}),
            "after_has_geometry": "ShaderNodeNewGeometry" in node_types(wear.worn(plain, "wall"))}


def pieces(sc, tl, m):
    """(furniture of the map, mesh of its object in the built and worn scene, a BVH tree of it) for each fixed one."""
    by_name = {f.name: f for f in m.furniture if f.mobility == "fixed" and f.blocks}
    for eid, ob in sc.objects.items():
        f = by_name.get(str(tl.entity(eid).attributes.get("name")))
        if f is not None:
            me = ob.data
            polygons = [tuple(p.vertices) for p in me.polygons]
            yield f, me, BVHTree.FromPolygons([v.co.copy() for v in me.vertices], polygons)


def finish_of(me, hit) -> str:
    """The material of the polygon a ray hit, without the worn copy's suffix ("raw+raw" is "raw")."""
    return me.materials[me.polygons[hit[2]].material_index].name.split("+")[0]


def underside_look(sc, tl, m) -> dict:
    """For each fixed piece with an open underside, rays cast up from below on a 9 x 9 grid over its footprint (3 cm in
    from its edge): of the hits at or above its underside (not a leg), the material of each."""
    out = {}
    for f, me, tree in pieces(sc, tl, m):
        if f.under_m <= 0.0:
            continue
        names: dict[str, int] = {}
        for i in range(9):
            for j in range(9):
                x = (i / 8 - 0.5) * (f.size[0] - 0.06)
                y = (j / 8 - 0.5) * (f.size[1] - 0.06)
                hit = tree.ray_cast(Vector((x, y, 0.001)), Vector((0, 0, 1)), 3.0)
                if hit[0] is not None and hit[0].z >= f.under_m - 0.003:
                    names[finish_of(me, hit)] = names.get(finish_of(me, hit), 0) + 1
        out[f.kind] = {"under_m": f.under_m, "hits": names}
    return out


def plinths(sc, tl, m) -> dict:
    """The material of a cabinet's plinth: a ray at 3 cm up, from 10 cm in front of its face along its length."""
    out = {}
    for f, me, tree in pieces(sc, tl, m):
        if f.kind in CABINETS:
            hit = tree.ray_cast(Vector((f.size[0] / 2 + 0.1, 0.0, 0.03)), Vector((-1, 0, 0)), 1.0)
            out[f.kind] = {"colour": f.color, "plinth": None if hit[0] is None else finish_of(me, hit)}
    return out


def tiles() -> dict:
    """A tile material of the bathroom's set (its bump's height is an image), dressed in the household's look beside
    three boxes on the floor: where the joints and the dust bands read from."""
    fresh()
    mat = with_bump("acg_Tiles107")
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.new("h", 4, 4)
    bump = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBump")
    nt.links.new(tex.outputs["Color"], bump.inputs["Height"])
    wear.use_catalog("home")
    wear_surfaces.use_scene([], [], [(1.0, 1.0, 0.3, 0.2, 0.0), (3.0, 1.0, 0.2, 0.2, 0.5), (5.0, 1.0, 0.4, 0.1, 1.0)])
    out = wear.worn(mat, "tile")
    bw = [n for n in out.node_tree.nodes if n.bl_idname == "ShaderNodeRGBToBW"]

    def absolutes(m: bpy.types.Material) -> int:
        return sum(1 for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "ABSOLUTE")

    result = {"joints_read_height": any("ShaderNodeTexImage" in upstream_types(n.inputs["Color"]) for n in bw),
              "absolutes": absolutes(out)}
    fresh()  # the same set with no height map to read: no joints, the dust bands stay
    bare = wear.worn(materials.new_material("acg_Tiles107"), "tile")
    result["bare"] = {"joints": any(n.bl_idname == "ShaderNodeRGBToBW" for n in bare.node_tree.nodes),
                      "absolutes": absolutes(bare)}
    wear_surfaces.use_scene([], [], [])
    wear.use_catalog("")
    return result


def main() -> None:
    result = {"bumps": bumps(), "flat_paint": flat_paint(), "tiles": tiles()}
    sc, tl, m = scene()
    result["walls"] = walls(sc, m)
    result["undersides"] = underside_look(sc, tl, m)
    result["plinths"] = plinths(sc, tl, m)
    args.out.write_text(json.dumps(result), encoding="utf-8")


main()
