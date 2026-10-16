# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_render_city_blender.py`: builds an episode's scene (no render, so no GPU) and prints
one JSON document that the test reads. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_scene_probe.py -- --render DIR --src DIR
        --root EPISODE --private PRIVATE --out RESULT.json
        [--appearance assets --library L.blend --manifest M.json]

The result holds a digest of everything the scene holds (objects, meshes, materials with their node graphs, lights, the
world, the camera), so that two builds can be compared without a render, the facts a test asserts about an outdoor scene
(its objects, lights, sky and camera), the colours the tinted slots of its entities are set to, and the names of every
shader and facade style that builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--root", "--private", "--out"):
    ap.add_argument(flag, type=Path, required=True)
for flag in ("--library", "--manifest"):
    ap.add_argument(flag, type=Path)
ap.add_argument("--appearance", choices=("placeholder", "assets"), default="placeholder")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import build_scene  # noqa: E402

from stbench.camera import Camera  # noqa: E402
from stbench.mapspec import load_map  # noqa: E402
from stbench.timeline import load  # noqa: E402


def r(x, n: int = 5):
    return [round(float(v), n) for v in x]


def socket_value(s):
    v = getattr(s, "default_value", None)
    if v is None:
        return None
    try:
        return r(v)
    except TypeError:
        return round(float(v), 5) if isinstance(v, int | float) else str(v)


def tree_signature(tree) -> list:
    """Nodes (type, name, every input's default value) and links of a node tree, in a stable order."""
    nodes = sorted(([n.bl_idname, n.name, [socket_value(i) for i in n.inputs]] for n in tree.nodes), key=str)
    links = sorted([lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier]
                   for lk in tree.links)
    return [nodes, links]


def mesh_signature(me) -> list:
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    idx = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", idx)
    return [len(me.vertices), len(me.polygons), hashlib.sha256(np.round(co, 5).tobytes()).hexdigest(),
            hashlib.sha256(idx.tobytes()).hexdigest(), [m.name if m else None for m in me.materials]]


def tint_targets() -> dict:
    """Per entity object, the colour each tinted slot is set to (linear RGB), by slot name: `stbench_tint`'s target.
    A slot is named after its material (`<asset>:<slot>[@colour]`); a tinted library texture (`acg_Fabric030@red*1`, a
    sofa's cover) has no slot name and is left out."""
    out = {}
    for ob in bpy.data.objects:
        row = {}
        for slot in ob.material_slots if ob.pass_index else ():
            node = slot.material.node_tree.nodes.get("stbench_tint") if slot.material else None
            if node is not None and ":" in slot.material.name:
                row[slot.material.name.split(":")[1].split("@")[0]] = r(node.inputs["Target"].default_value[:3], 4)
        if row:
            out[ob.name] = row
    return out


def digest() -> str:
    doc = {"objects": sorted([o.name, o.type, o.pass_index, r(o.location), r(o.rotation_euler), r(o.scale),
                              o.hide_render, o.data.name if o.data else None] for o in bpy.data.objects),
           "meshes": sorted([me.name, mesh_signature(me)] for me in bpy.data.meshes),
           "materials": sorted([m.name, tree_signature(m.node_tree) if m.use_nodes else None]
                               for m in bpy.data.materials),
           "groups": sorted([g.name, tree_signature(g)] for g in bpy.data.node_groups),
           "lights": sorted([li.name, li.type, round(li.energy, 5), round(getattr(li, "angle", 0.0), 5),
                             round(getattr(li, "size", 0.0), 5)] for li in bpy.data.lights),
           "worlds": sorted([w.name, tree_signature(w.node_tree)] for w in bpy.data.worlds),
           "cameras": sorted([c.name, round(c.angle, 5), c.clip_start, c.clip_end, c.sensor_fit]
                             for c in bpy.data.cameras)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def main() -> None:
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    hfov = json.loads((args.root / "sensors" / "camera.json").read_text())["hfov_deg"]
    Camera.load(args.root / "sensors" / "camera.json")
    assets = None
    if args.appearance == "assets":
        from stbench import asset_manifest

        assets = (asset_manifest.load(args.manifest), args.library)
    scene = build_scene.build(tl, m, hfov, assets)
    sun = next((li for li in bpy.data.lights if li.type == "SUN"), None)
    sky = next((n for w in bpy.data.worlds for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky"), None)
    bg = next((n for w in bpy.data.worlds for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"), None)
    sun_ob = next((o for o in bpy.data.objects if o.type == "LIGHT" and o.data.type == "SUN"), None)
    result = {"digest": digest(), "objects": sorted(o.name for o in bpy.data.objects),
              "indexed": sorted([o.pass_index, o.name] for o in bpy.data.objects if o.pass_index),
              "instance_map": {str(k): v for k, v in scene.instance_map.items()},
              "lights": [[li.name, li.type, li.energy] for li in bpy.data.lights],
              "sun_energy": None if sun is None else sun.energy,
              "sun_direction": None if sun_ob is None else r(sun_ob.rotation_euler.to_matrix() @ Vector((0, 0, 1))),
              "sky": None if sky is None else {"elevation": round(float(sky.sun_elevation), 5),
                                               "rotation": round(float(sky.sun_rotation), 5)},
              "sky_strength": None if bg is None else round(float(bg.inputs["Strength"].default_value), 5),
              "camera": {"clip_start": scene.camera.data.clip_start, "clip_end": scene.camera.data.clip_end},
              "tints": tint_targets()}
    import facade_shaders  # after the digest: building every shader adds materials to the file
    import materials

    result["shaders"] = sorted(n for n in materials.SHADERS if materials.shader_material(n) is not None)
    result["facade_styles"] = sorted(n for n in facade_shaders.FACADES if facade_shaders.material(n) is not None)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


main()
