# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_procedural_detail_blender.py`: builds procedural kinds with the real builders and
prints one JSON document that the test reads. Not a pytest module (Blender's Python has no pytest).

    blender --background --factory-startup --python tests/blender_procedural_driver.py -- \
        --render DIR --kinds a,b,c --out RESULT.json

Per kind: the extents of the builder's own geometry before the exact fit (`natural`) and of the mesh `procedural.build`
returns (`built`, and its lowest corner), the polygons per slot, how high the mesh stands in the front and the back
fifth of its length, how many closed pieces of it face inwards, a digest of the mesh (so that two runs can be compared)
and the share of the object's pixels that each slot covers, over eight views (four turns at 15 and at 35 degrees up),
rendered flat with a distinct emission colour per slot, one sample per pixel, so that the count is the geometry's and
not the light's; and how much of its silhouette's bounding rectangle the object fills in those views (a block fills
nearly all of it, a pallet jack's forks and tiller little).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument("--render", type=Path, required=True)
ap.add_argument("--kinds", required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path.insert(0, str(args.render))

import procedural  # noqa: E402

TILE = (128, 96)
VIEWS = [(az, el) for el in (15.0, 35.0) for az in (45.0, 135.0, 225.0, 315.0)]
FLAT = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 0.5, 0), (0.5, 0, 1), (0.5, 1, 0),
        (1, 0.5, 0.5)]  # a distinct colour per slot (black is the background)


def natural(kind: str) -> list[float]:
    bm = bmesh.new()
    procedural.BUILDERS[kind](bm, procedural.DESIGN_BOX[kind])
    co = np.array([v.co[:] for v in bm.verts])
    bm.free()
    return (co.max(axis=0) - co.min(axis=0)).tolist()


def digest(mesh) -> str:
    """A hash of the mesh as a set of shapes, whatever order its vertices and polygons were made in: each polygon's
    slot and its corners to a micrometre, sorted. (`meshkit.slab` bevels a set of edges, so vertex numbers can differ
    between two builds of the same geometry.)"""
    co = np.round(np.array([v.co[:] for v in mesh.vertices]) * 1e6).astype(np.int64)
    faces = sorted((p.material_index, tuple(sorted(tuple(co[i]) for i in p.vertices))) for p in mesh.polygons)
    return hashlib.sha256(repr(faces).encode()).hexdigest()


def inside_out(mesh) -> int:
    """How many closed pieces of the mesh (connected polygons with no open edge) have a negative signed volume, that is,
    face away from outside."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    seen, bad = set(), 0
    for first in bm.faces:
        if first in seen:
            continue
        piece, stack, closed = [], [first], True
        seen.add(first)
        while stack:
            f = stack.pop()
            piece.append(f)
            for e in f.edges:
                closed = closed and not e.is_boundary
                for g in e.link_faces:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        volume = 0.0
        for f in piece:
            vs = [v.co for v in f.verts]
            volume += sum(vs[0].dot(vs[i].cross(vs[i + 1])) for i in range(1, len(vs) - 1)) / 6.0
        bad += int(closed and volume < 0.0)
    bm.free()
    return bad


def scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    r, cy = sc.render, sc.cycles
    r.engine, cy.device, cy.samples, cy.use_denoising = "CYCLES", "CPU", 1, False
    r.resolution_x, r.resolution_y, r.resolution_percentage = *TILE, 100
    cy.seed, cy.use_animated_seed, cy.pixel_filter_type, cy.filter_width = 0, False, "BOX", 0.01
    sc.view_settings.view_transform, sc.view_settings.look = "Raw", "None"
    sc.render.image_settings.file_format, sc.render.image_settings.color_mode = "PNG", "RGB"
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    sc.collection.objects.link(cam)
    sc.camera = cam
    return sc


def flat(i: int) -> bpy.types.Material:
    mat = bpy.data.materials.new(f"flat{i}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out, em = nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*FLAT[i], 1.0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def shares(kind: str, tmp: Path) -> tuple[dict, float]:
    """Per slot, the mean over the views of its pixels' share of the object's pixels, and the least; and the mean over
    the views of the fraction of the silhouette's bounding rectangle that the object fills."""
    sc = scene()
    mesh, slots = procedural.build(kind, f"share:{kind}")
    for i in range(len(slots)):
        mesh.materials.append(flat(i))
    ob = bpy.data.objects.new(kind, mesh)
    sc.collection.objects.link(ob)
    size = Vector(procedural.DESIGN_BOX[kind])
    centre = Vector((0, 0, size.z / 2))
    cam = sc.camera
    cam.data.ortho_scale = 1.15 * size.length
    per_view, fills = [], []
    for az, el in VIEWS:
        a, e = math.radians(az), math.radians(el)
        cam.location = centre + 5.0 * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = str(tmp / f"{kind}.png")
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(tmp / f"{kind}.png"), check_existing=False)
        px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)[:, :3]
        bpy.data.images.remove(img)
        palette = np.array([(0, 0, 0), *FLAT[:len(slots)]], dtype=np.float32)
        nearest = np.argmin(((px[:, None, :] - palette[None]) ** 2).sum(axis=2), axis=1)
        counts = np.bincount(nearest, minlength=len(slots) + 1)[1:]
        per_view.append(counts / max(1, counts.sum()))
        mask = (nearest != 0).reshape(TILE[1], TILE[0])
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        fills.append(mask.sum() / ((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)))
    per_view = np.array(per_view)
    return ({slot: {"mean": float(per_view[:, i].mean()), "least": float(per_view[:, i].min())}
             for i, slot in enumerate(slots)}, float(np.mean(fills)))


def main() -> None:
    out = {}
    tmp = args.out.parent
    for kind in args.kinds.split(","):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh, slots = procedural.build(kind, f"probe:{kind}")
        again, _ = procedural.build(kind, f"probe2:{kind}")
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        idx = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("material_index", idx)
        out[kind] = {"slots": list(slots), "natural": natural(kind), "design": list(procedural.DESIGN_BOX[kind]),
                     "built": (co.max(axis=0) - co.min(axis=0)).tolist(), "built_min": co.min(axis=0).tolist(),
                     "polygons": np.bincount(idx, minlength=len(slots)).tolist(), "digest": digest(mesh),
                     "front_height": float(co[co[:, 0] > 0.3 * (co[:, 0].max() - co[:, 0].min())][:, 2].max()),
                     "back_height": float(co[co[:, 0] < -0.3 * (co[:, 0].max() - co[:, 0].min())][:, 2].max()),
                     "digest_again": digest(again), "inside_out": inside_out(mesh)}
        out[kind]["shares"], out[kind]["fill"] = shares(kind, tmp)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"STBENCH-PROCEDURAL {len(out)} kinds", flush=True)


main()
