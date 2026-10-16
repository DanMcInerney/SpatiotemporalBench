# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Build the asset library: every model, dressing model and material of the manifest, normalised, in one `.blend`.

    blender --background --factory-startup --python render/asset_library.py -- \
        --manifest M --cache C --out L.blend --src REPO/src [--include-alternates]

Started by `stbench assets library`; never imported by `stbench`. Per model or dressing model, in id order: import the
file the manifest names (glTF or OBJ, never a `.blend`, so nothing can run), apply its declared unit scale and rotation,
join every mesh into one so that an entity is one object with one `pass_index`, optionally compose a mirrored pair,
re-centre to the base centre, and measure the box. A dressing model is built exactly so (`transform` leaves a Poly Haven
model width x depth x height, front toward -y), and has no kind, mobility or tint. Imported materials keep their
Principled trees; the manifest's `tint.materials` get the `stbench_tint` group. Procedural kinds come from
`procedural.py`, ambientCG zips from `materials.py`. Writes the `.blend` and, next to it, `<name>.json` (boxes, slots,
mean luminances).
A tinted slot's mean luminance is taken over the texels its triangles use, not over the whole image (see `texel_mean`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import materials as M  # noqa: E402
import procedural  # noqa: E402

BBOX_TOLERANCE_M = 0.001  # a later build may not measure a box this much off the one the manifest recorded
UP_AXES = ("Y", "Z")


class LibraryError(Exception):
    pass


def absolute(text: str) -> Path:
    """Blender's file API reads a relative path against the blend file, not the working directory."""
    return Path(text).resolve()


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=absolute, required=True)
    ap.add_argument("--cache", type=absolute, required=True)
    ap.add_argument("--out", type=absolute, required=True)
    ap.add_argument("--src", type=absolute, required=True, help="the directory that holds the stbench package")
    ap.add_argument("--include-alternates", action="store_true")
    return ap.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


# -- import and normalise ----------------------------------------------------------------------------------------
def _mesh_objects(before: set) -> list:
    return sorted((o for o in bpy.data.objects if o not in before and o.type == "MESH"), key=lambda o: o.name)


def import_model(a, path: Path) -> list:
    """Import the entry file; the new mesh objects, in name order."""
    before = set(bpy.data.objects)
    if a.format == "obj":
        up = (a.transform or {}).get("up", "Z")
        if up not in UP_AXES:
            raise LibraryError(f"{a.id}: an OBJ's up axis is Y or Z, not {up!r}")
        forward = "Y" if up == "Z" else "NEGATIVE_Z"  # the file's forward axis that leaves it as it is in Blender
        bpy.ops.wm.obj_import(filepath=str(path), global_scale=1.0, up_axis=up, forward_axis=forward)
    else:
        if (a.transform or {}).get("up", "Z") != "Z":
            raise LibraryError(f"{a.id}: the glTF importer already makes Z up; transform.up must be 'Z'")
        bpy.ops.import_scene.gltf(filepath=str(path))
    found = _mesh_objects(before)
    if not found:
        raise LibraryError(f"{a.id}: {path.name} holds no mesh")
    return found


def join_objects(objs: list) -> bpy.types.Mesh:
    """One mesh in world space from several objects (parent transforms baked in); the objects are deleted."""
    for o in objs:
        world = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = world
    target = objs[0]
    if len(objs) > 1:
        with bpy.context.temp_override(active_object=target, selected_editable_objects=objs, selected_objects=objs):
            bpy.ops.object.join()
    mesh = target.data
    mesh.transform(target.matrix_world)
    bpy.data.objects.remove(target)
    for o in [o for o in bpy.data.objects if o.type == "EMPTY"]:
        bpy.data.objects.remove(o)
    return mesh


def reset_rotations(a, objs: list) -> None:
    """Put the objects the manifest names in `rest_pose` back to the rotation the file's rest pose has (identity): a
    hinged door the exporter wrote open. An unknown name is an error."""
    by_name = {o.name: o for o in bpy.data.objects}
    if unknown := [n for n in a.rest_pose if n not in by_name]:
        raise LibraryError(f"{a.id}: rest_pose names {unknown}, which are not objects of the file "
                           f"(objects: {sorted(o.name for o in objs)})")
    for name in a.rest_pose:
        by_name[name].rotation_mode = "QUATERNION"
        by_name[name].rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    bpy.context.view_layer.update()


def drop_faces(a, mesh, slot_names: list[str]) -> None:
    """Delete every face on the slots the manifest names in `drop_slots` (the slot stays, unused: removing a slot would
    move every polygon to slot 0). An unknown name is an error."""
    if unknown := [n for n in a.drop_slots if n not in slot_names]:
        raise LibraryError(f"{a.id}: drop_slots names {unknown}, which are not slots of the asset "
                           f"(slots: {slot_names})")
    gone = {i for i, n in enumerate(slot_names) if n in a.drop_slots}
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in gone], context="FACES")
    bm.to_mesh(mesh)
    bm.free()


def vertices(mesh) -> np.ndarray:
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def bounds(mesh) -> tuple[np.ndarray, np.ndarray]:
    v = vertices(mesh)
    return v.min(axis=0), v.max(axis=0)


def recentre(mesh) -> list[float]:
    """Move the box's bottom centre to the origin; the box's size in metres."""
    lo, hi = bounds(mesh)
    mesh.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    return [float(x) for x in hi - lo]


def compose_pair(mesh, gap: float) -> None:
    """The mesh and its mirror image across y = 0, the two `gap` metres apart (a pair of shoes)."""
    lo, hi = bounds(mesh)
    shift = gap / 2 + (hi[1] - lo[1]) / 2
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, shift, 0.0))
    dup = bmesh.ops.duplicate(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces))["geom"]
    for v in (g for g in dup if isinstance(g, bmesh.types.BMVert)):
        v.co.y = -v.co.y
    bmesh.ops.reverse_faces(bm, faces=[g for g in dup if isinstance(g, bmesh.types.BMFace)])
    bm.to_mesh(mesh)
    bm.free()


# -- materials ---------------------------------------------------------------------------------------------------
def relink_missing(images, folder: Path) -> None:
    """Point an image whose file is not where the importer looked at the file of that name under `folder`. Blender
    resolves an MTL's `map_Kd` against the OBJ's own folder, and Google Scanned Objects keep their textures one folder
    over (`materials/textures/`)."""
    for img in images:
        if not img.filepath or img.packed_file:
            continue
        path = Path(bpy.path.abspath(img.filepath))
        if not path.is_file() and (hits := sorted(folder.rglob(path.name))):
            img.filepath = str(hits[0])
            img.reload()


def make_absolute(images) -> None:
    for img in images:
        if img.filepath and not img.packed_file:
            img.filepath = str(Path(bpy.path.abspath(img.filepath)).resolve())
            if img.size[0] == 0:
                raise LibraryError(f"texture {img.filepath} is missing or unreadable")


def tint_slots(a, slot_names: list[str]) -> list[str]:
    """The slot names the manifest says take the colour attribute; an unknown name is an error, null means none yet."""
    mode, wanted = (a.tint or {}).get("mode", "none"), (a.tint or {}).get("materials")
    if mode == "none" or wanted is None:
        return []
    if "*" in wanted:
        return list(slot_names)
    if unknown := [w for w in wanted if w not in slot_names]:
        raise LibraryError(f"{a.id}: tint.materials names {unknown}, which are not slots of the asset "
                           f"(slots: {slot_names})")
    return list(wanted)


def slot_triangles(mesh, index: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The UV corners (t, 3, 2) and 3-D areas (t,) of the triangles on material slot `index`; None for a mesh without a
    UV map (its materials project their textures on object coordinates)."""
    layer = mesh.uv_layers.active
    if layer is None:
        return None
    mesh.calc_loop_triangles()
    t = len(mesh.loop_triangles)
    loops, area, slot = np.empty(t * 3, np.int32), np.empty(t, np.float32), np.empty(t, np.int32)
    mesh.loop_triangles.foreach_get("loops", loops)
    mesh.loop_triangles.foreach_get("area", area)
    mesh.loop_triangles.foreach_get("material_index", slot)
    uv = np.empty(len(mesh.loops) * 2, np.float32)
    layer.uv.foreach_get("vector", uv)
    keep = slot == index
    return uv.reshape(-1, 2)[loops.reshape(-1, 3)[keep]].astype(np.float64), area[keep].astype(np.float64)


def finish_slots(a, mesh, bases: list[bpy.types.Material], slot_names: list[str]) -> list[dict]:
    """Name each slot's material `<asset id>:<slot>`, tint the chosen ones, put them on the mesh. A slot is replaced in
    place and never cleared: removing a mesh's slots resets every polygon to slot 0, which would put a lens, a label or
    a pot's soil on the first material. A tinted slot's mean luminance is its texture's mean over the texels that
    slot's own triangles map to (`mean_lum`); the whole image's (`mean_lum_image`) counts the unused black of a scanned
    object's atlas and is recorded only to show the correction."""
    chosen, report = set(tint_slots(a, slot_names)), []
    for i, (slot, mat) in enumerate(zip(slot_names, bases, strict=True)):
        mat.name = f"{a.id}:{slot}"
        mean = image_mean = None
        if slot in chosen:
            mean, image_mean = M.surface_mean_luminance(mat, slot_triangles(mesh, i))
            M.add_tint(mat, mean)
            mat[M.MEAN_PROP] = mean
        if i < len(mesh.materials):
            mesh.materials[i] = mat
        else:
            mesh.materials.append(mat)
        report.append({"slot": slot, "material": mat.name, "tinted": slot in chosen, "mean_lum": mean,
                       "mean_lum_image": image_mean})
    return report


def _unique_slot_names(mats) -> list[str]:
    out = []
    for m in mats:
        base = m.name.rsplit(".", 1)[0] if m.name.rsplit(".", 1)[-1].isdigit() else m.name
        out.append(base if base not in out else m.name)
    return out


# -- one model ---------------------------------------------------------------------------------------------------
def build_imported(a, cache: Path):
    from stbench import asset_manifest as AM

    entry = next(f for f in a.files if f.path == a.entry)
    objs = import_model(a, AM.file_path(a, entry, cache))
    relink_missing(bpy.data.images, AM.asset_dir(a, cache))
    make_absolute(bpy.data.images)
    tr = a.transform or {}
    if a.rest_pose:
        reset_rotations(a, objs)
    mesh = join_objects(objs)
    if a.drop_slots:
        drop_faces(a, mesh, _unique_slot_names(list(mesh.materials)))
    rot = Matrix.Identity(4)
    for axis, deg in zip("XYZ", tr.get("rotate_deg", (0, 0, 0)), strict=True):
        rot = Matrix.Rotation(math.radians(deg), 4, axis) @ rot
    mesh.transform(rot @ Matrix.Scale(float(tr.get("unit_scale", 1.0)), 4))
    recentre(mesh)
    if a.compose:
        compose_pair(mesh, float(a.compose["gap_m"]))
    return mesh, list(mesh.materials)


def build_procedural(a, library_materials: dict):
    mesh, slots = procedural.build(a.builder, f"M:{a.id}")
    wanted = a.materials or {}
    if set(wanted) != set(slots):
        raise LibraryError(f"{a.id}: the manifest's materials {sorted(wanted)} are not the builder's slots "
                           f"{sorted(slots)}")
    bases = []
    for slot in slots:
        ref = wanted[slot]
        bases.append(M.shader_material(ref.split(":", 1)[1]) if ref.startswith("shader:")
                     else library_materials[ref].copy())
    return mesh, bases, list(slots)


def build_model(a, cache: Path, library_materials: dict) -> dict:
    if a.source == "procedural":
        mesh, bases, slot_names = build_procedural(a, library_materials)
    else:
        mesh, bases = build_imported(a, cache)
        slot_names = _unique_slot_names(bases)
    mesh.name = f"M:{a.id}"
    lo, hi = bounds(mesh)
    size = [float(x) for x in hi - lo]
    if a.native_bbox_m is not None:
        off = max(abs(s - n) for s, n in zip(size, a.native_bbox_m, strict=True))
        if off > BBOX_TOLERANCE_M:
            raise LibraryError(f"{a.id}: measured box {[round(s, 4) for s in size]} m is more than 1 mm off the "
                               f"manifest's native_bbox_m {a.native_bbox_m}")
    slots = finish_slots(a, mesh, bases, slot_names)
    mesh.use_fake_user = True
    return {"role": a.role, "kind": a.kind, "mobility": a.mobility, "bbox_m": [round(s, 5) for s in size],
            "polygons": len(mesh.polygons), "slots": slots, "tint_materials": (a.tint or {}).get("materials"),
            "tint_mode": (a.tint or {}).get("mode")}


def main() -> None:
    args = parse(sys.argv)
    sys.path.insert(0, str(args.src))
    from stbench import asset_manifest as AM

    m = AM.load(args.manifest)
    if problems := AM.validate(m):
        raise LibraryError("the manifest has problems:\n  " + "\n  ".join(problems))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cache = args.cache
    mats, report_mats = {}, {}
    for a in sorted((a for a in m.assets if a.role == "material"), key=lambda a: a.id):
        mat = M.ambientcg_material(a.id, AM.extract_dir(a, cache), a.tile_m)
        mat.use_fake_user = True
        mats[a.id] = mat
        report_mats[a.id] = {"mean_lum": round(mat[M.MEAN_PROP], 5), "tile_m": a.tile_m,
                             "tile_defaulted": a.tile_m is None}
    report_assets = {}
    for a in sorted((a for a in m.assets if a.role in AM.IMPORTED), key=lambda a: a.id):
        if a.alternate_for and not args.include_alternates:
            continue
        report_assets[a.id] = build_model(a, cache, mats)
        print(f"STBENCH-ASSET {a.id} bbox_m={report_assets[a.id]['bbox_m']} "
              f"slots={[s['slot'] for s in report_assets[a.id]['slots']]}", flush=True)
    make_absolute(bpy.data.images)
    meshes = [me for me in bpy.data.meshes if me.name.startswith("M:")]  # not the importers' orphaned leftovers
    blocks = {*meshes, *mats.values(), *(mat for me in meshes for mat in me.materials), *bpy.data.node_groups}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(str(args.out), blocks, path_remap="ABSOLUTE", fake_user=True)
    doc = {"assets": report_assets, "materials": report_mats}
    args.out.with_suffix(".json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"STBENCH-LIBRARY {args.out} {len(report_assets)} models, {len(report_mats)} materials", flush=True)


if __name__ == "__main__":  # Blender runs `--python` scripts as __main__; a test driver imports the functions above
    try:
        main()
    except LibraryError as e:
        print(f"STBENCH-ERROR {e}", flush=True)
        raise SystemExit(4) from e
