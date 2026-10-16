# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The resident's body, made once: one clothed, barefoot adult in one standing pose, exported as a `.glb`.

    BLENDER_USER_RESOURCES=SCRATCH BLENDER_USER_CONFIG=SCRATCH/config blender --background --factory-startup \
        --python render/people_make.py -- --out <cache>/resident/resident.glb --extension <MPFB zip> \
        --pack <system assets zip>

Run by hand, never by `stbench`: the file is the manifest's `resident` asset, and its source `makehuman` (kind
`generated`) names what made it (`generator`: this recipe and its inputs, each with its licence and url).
The recipe installs MPFB 2 (the MakeHuman plug-in, GPL code) from its extension zip and unpacks the MakeHuman system
assets pack (CC0) into a scratch Blender profile. It refuses to run unless `BLENDER_USER_RESOURCES` is set (the
machine's own profile is never touched). It opens no network connection.
Then it builds the human (the assets named below), poses the rig once (arms lowered, feet together and flat), bakes
every mesh as the depsgraph evaluates it (pose, hidden helper geometry and one subdivision applied), turns the person to
face +x, stands it on z = 0 centred on the origin and writes one GLB: no rig, no animation, no shoe, sock or slipper.
Prints `STBENCH-PERSON <json>` with the measured box.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

EXTENSION = "bl_ext.user_default.mpfb"
SUBDIV = 1  # one subdivision level: a MakeHuman base mesh is faceted at a metre
PHENOTYPE = {"gender": 1.0, "age": 0.5, "muscle": 0.5, "weight": 0.5, "proportions": 0.5, "height": 0.5,
             "cupsize": 0.5, "firmness": 0.5, "race": {"asian": 0.0, "caucasian": 1.0, "african": 0.0}}
# What the person is made of (every asset is CC0, from the system assets pack). No shoes: the feet are the skin's.
PARTS = {"skin": "young_caucasian_male/young_caucasian_male.mhmat", "eyes": "high-poly/high-poly.mhclo",
         "eyebrows": "eyebrow001/eyebrow001.mhclo", "eyelashes": "eyelashes01/eyelashes01.mhclo",
         "hair": "short02/short02.mhclo", "clothes": "male_casualsuit01/male_casualsuit01.mhclo"}
OBJECT_OF = {"Human": "body", "Human.high-poly": "eyes", "Human.eyebrow001": "eyebrows",
             "Human.eyelashes01": "eyelashes", "Human.short02": "hair", "Human.male_casualsuit01": "clothes"}
ARM_DOWN_OUT_DEG, FOREARM_FORWARD_DEG, STANCE_DEG = 5.0, 12.0, 4.0  # the pose: relaxed, arms at the sides


class RecipeError(Exception):
    pass


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="the .glb to write")
    ap.add_argument("--extension", type=Path, action="append", default=[], help="a Blender extension zip (MPFB)")
    ap.add_argument("--pack", type=Path, action="append", default=[], help="an asset pack zip (the system assets)")
    return ap.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


# -- the inputs ----------------------------------------------------------------------------------------------------
def install(args) -> None:
    """The extension into the scratch profile, then the asset packs into the extension's user data folder."""
    if not os.environ.get("BLENDER_USER_RESOURCES"):
        raise RecipeError("set BLENDER_USER_RESOURCES (and BLENDER_USER_CONFIG) to a scratch folder: this recipe never "
                          "installs into the machine's own Blender profile")
    for path in args.extension:
        bpy.ops.extensions.package_install_files(filepath=str(path), repo="user_default", enable_on_install=True)
    if EXTENSION not in bpy.context.preferences.addons:
        raise RecipeError("MPFB did not install and enable")
    data = Path(bpy.utils.extension_path_user(EXTENSION, path="data", create=True))
    for path in args.pack:
        with zipfile.ZipFile(path) as z:
            z.extractall(data)


# -- the person ----------------------------------------------------------------------------------------------------
def build_human():
    """The human, with the default rig, the parts above, the skin and one subdivision level; returns (body, rig)."""
    humans = importlib.import_module(f"{EXTENSION}.services.humanservice").HumanService
    info = humans._create_default_human_info_dict()
    info["phenotype"] = PHENOTYPE
    info.update(rig="default", eyes=PARTS["eyes"], eyebrows=PARTS["eyebrows"], eyelashes=PARTS["eyelashes"],
                hair=PARTS["hair"], clothes=[PARTS["clothes"]], skin_mhmat=PARTS["skin"],
                skin_material_type="MAKESKIN", clothes_material_type="MAKESKIN", eyes_material_type="MAKESKIN")
    settings = humans.get_default_deserialization_settings()
    settings["subdiv_levels"] = SUBDIV
    body = humans.deserialize_from_dict(info, settings)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    return body, rig


def _pose_bone(rig, name):
    bpy.context.view_layer.update()
    return rig.pose.bones[name]


def aim(rig, name: str, direction) -> None:
    """Turn a pose bone about its head until it points along `direction` (armature space)."""
    pb = _pose_bone(rig, name)
    turn = (pb.tail - pb.head).normalized().rotation_difference(Vector(direction).normalized()).to_matrix().to_4x4()
    pb.matrix = Matrix.Translation(pb.head) @ turn @ pb.matrix.to_3x3().to_4x4()


def swing(rig, name: str, axis: str, degrees: float) -> None:
    """Turn a pose bone about its head, about an armature axis."""
    pb = _pose_bone(rig, name)
    turn = Matrix.Rotation(math.radians(degrees), 4, axis)
    pb.matrix = Matrix.Translation(pb.head) @ turn @ pb.matrix.to_3x3().to_4x4()


def pose(rig) -> None:
    """Arms down at the sides (a little out, the forearms a little forward), the feet brought together and kept flat."""
    for side in ("L", "R"):
        sx = 1.0 if _pose_bone(rig, f"upperarm01.{side}").head.x > 0 else -1.0
        a, f = math.radians(ARM_DOWN_OUT_DEG), math.radians(FOREARM_FORWARD_DEG)
        aim(rig, f"upperarm01.{side}", (sx * math.sin(a), 0.0, -math.cos(a)))
        aim(rig, f"lowerarm01.{side}", (sx * math.sin(a * 0.6), -math.sin(f), -math.cos(f)))
        swing(rig, f"upperleg01.{side}", "Y", sx * STANCE_DEG)
        swing(rig, f"foot.{side}", "Y", -sx * STANCE_DEG)
    bpy.context.view_layer.update()


def bake(rig) -> list:
    """One plain mesh object per part, as the depsgraph evaluates it (pose, hidden helpers and subdivision applied);
    the originals and the rig are deleted. Materials are named for their part."""
    for ob in bpy.data.objects:
        for mod in ob.modifiers:
            if mod.type == "SUBSURF":
                mod.levels = SUBDIV
    graph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    baked = []
    for ob in sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.name):
        part = OBJECT_OF[ob.name]
        mesh = bpy.data.meshes.new_from_object(ob.evaluated_get(graph), depsgraph=graph)
        mesh.name = part
        for mat in mesh.materials:
            mat.name = part
        new = bpy.data.objects.new(part, mesh)
        scene.collection.objects.link(new)
        baked.append(new)
    for ob in [o for o in bpy.data.objects if o not in baked]:
        bpy.data.objects.remove(ob)
    return baked


def stand(baked: list) -> list[float]:
    """Face +x (MakeHuman faces -y), stand on z = 0, centre the box on the origin; the box (x, y, z) in metres."""
    turn = Matrix.Rotation(math.radians(90.0), 4, "Z")
    for ob in baked:
        ob.data.transform(turn)
    pts = [v.co for ob in baked for v in ob.data.vertices]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    move = Matrix.Translation((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    for ob in baked:
        ob.data.transform(move)
    return [round(v, 5) for v in hi - lo]


def export(baked: list, out: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for ob in baked:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = baked[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_selection=True, export_apply=False,
                              export_yup=True, export_image_format="AUTO", export_animations=False,
                              export_skins=False, export_morph=False, export_cameras=False, export_lights=False)


def main() -> None:
    args = parse(sys.argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    install(args)
    bpy.ops.preferences.addon_enable(module=EXTENSION)
    body, rig = build_human()
    pose(rig)
    baked = bake(rig)
    box = stand(baked)
    export(baked, args.out)
    report = {"bbox_m": box, "blender": ".".join(map(str, bpy.app.version)), "parts": sorted(o.name for o in baked)}
    print(f"STBENCH-PERSON {json.dumps(report, sort_keys=True)}", flush=True)


if __name__ == "__main__":  # Blender runs `--python` scripts as __main__
    try:
        main()
    except RecipeError as e:
        print(f"STBENCH-ERROR {e}", flush=True)
        raise SystemExit(4) from e
