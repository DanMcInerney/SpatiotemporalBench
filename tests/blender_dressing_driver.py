# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_dressing_build_blender.py`: builds a map's dressing twice with `dressing.build` (no
render, so no GPU, and no asset library: every asset piece is its stand-in) and writes what the scene holds as one JSON
document. It also fits a hand-made asset (a plank with a bump on its front face) into boxes at several yaws, to show
which way a Poly Haven model (width x depth x height, facing -y) ends up. Not a pytest module (Blender's Python has no
pytest).

    blender --background --factory-startup --python tests/blender_dressing_driver.py -- \
        --render DIR --src DIR --map apartment_a --catalog home --out RESULT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--out"):
    ap.add_argument(flag, type=Path, required=True)
ap.add_argument("--map", default="apartment_a")
ap.add_argument("--catalog", default="home")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import dressing  # noqa: E402
import look  # noqa: E402
import materials  # noqa: E402
import numpy as np  # noqa: E402
from dressing_spec import Dressing  # noqa: E402


def clear() -> None:
    """An empty file: the objects, meshes, materials, collections and node groups of an earlier build are gone."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.collections, bpy.data.node_groups):
        for x in list(coll):
            coll.remove(x)
    materials.use_map(None)  # forgets the named materials, which are freed with the file's data


def world_box(ob) -> list[float]:
    me = ob.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    w = np.array(ob.matrix_world)
    pts = co @ w[:3, :3].T + w[:3, 3]
    return [round(float(v), 6) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def local_box(ob) -> list[float]:
    """The mesh in the piece's own frame (x out of the wall, y across, z up): the world points relative to the object's
    place, turned back by its yaw. A piece at any yaw has the extents its builder drew."""
    me = ob.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    w = np.array(ob.matrix_world)
    rel = co.reshape(-1, 3) @ w[:3, :3].T + w[:3, 3] - np.array(ob.location)
    c, s = math.cos(-ob.rotation_euler.z), math.sin(-ob.rotation_euler.z)
    pts = rel @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return [round(float(v), 6) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def digest(objs) -> str:
    """Everything the objects hold, rounded to a tenth of a millimetre: names, transforms, geometry, materials."""
    doc = []
    for ob in sorted(objs, key=lambda o: o.name):
        me = ob.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        idx = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get("material_index", idx)
        doc.append([ob.name, me.name, ob.pass_index, [round(float(v), 4) for v in (*ob.location, *ob.rotation_euler,
                                                                                    *ob.scale)],
                    len(me.vertices), len(me.polygons), hashlib.sha256(np.round(co, 4).tobytes()).hexdigest(),
                    hashlib.sha256(idx.tobytes()).hexdigest(), [m.name if m else None for m in me.materials]])
    return hashlib.sha256(json.dumps(doc).encode()).hexdigest()


def build_once(spec: dict) -> tuple[list, float]:
    clear()
    t0 = time.perf_counter()
    objs = dressing.build(spec, args.catalog, dressing.Ctx())
    seconds = time.perf_counter() - t0
    bpy.context.view_layer.update()  # the objects' world matrices
    return objs, seconds


def fit_check() -> list[dict]:
    """A plank 0.6 wide (x), 0.04 deep (y), 0.8 high (z) with a bump on the front (-y) face at x 0.1 to 0.2, fitted into
    boxes at several yaws: where its box ends up and where its bump is."""
    clear()
    me = bpy.data.meshes.new("M:test_plank")
    verts = [(x, y, z) for x in (-0.3, 0.3) for y in (-0.02, 0.02) for z in (0.0, 0.8)]
    verts += [(x, y, z) for x in (0.1, 0.2) for y in (-0.03, -0.02) for z in (0.5, 0.6)]
    faces = []
    for base in (0, 8):
        v = [base + i for i in range(8)]
        faces += [(v[0], v[1], v[3], v[2]), (v[4], v[6], v[7], v[5]), (v[0], v[4], v[5], v[1]),
                  (v[2], v[3], v[7], v[6]), (v[0], v[2], v[6], v[4]), (v[1], v[5], v[7], v[3])]
    me.from_pydata(verts, [], faces)
    out = []
    template = bpy.data.objects.new("template", me)
    for yaw, pos, size in ((0.0, (1.0, 2.0, 1.1), (0.04, 0.5, 0.6)), (90.0, (3.0, 1.0, 0.2), (0.05, 0.8, 1.0)),
                           (180.0, (0.0, -1.0, 1.0), (0.03, 0.3, 0.4)), (-90.0, (2.0, 5.0, 0.0), (0.02, 0.6, 0.9))):
        d = Dressing("picture_frame", "asset:test_plank", pos, yaw, size)
        ob = dressing.fitted(d, template, "fit")
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.update()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        w = np.array(ob.matrix_world)
        pts = co @ w[:3, :3].T + w[:3, 3]
        bump = pts[np.array([v[1] < -0.0205 for v in co])]
        out.append({"yaw": yaw, "pos": pos, "size": size, "box": list(d.box), "world": world_box(ob),
                    "bump": [round(float(v), 5) for v in bump.mean(axis=0)]})
    return out


def library_check() -> dict:
    """A library `.blend` of one model, named as the asset library names them (`M:<id>`, its material `<id>:<slot>`),
    written here and read back through `Ctx(app)` by the id a layout uses; the manifest is a stub that lists the id as a
    dressing model. The model, its material, an unknown id and its fit into a box."""
    clear()
    me = bpy.data.meshes.new("M:test_slab")
    me.from_pydata([(x, y, z) for x in (-0.25, 0.25) for y in (-0.01, 0.01) for z in (0.0, 0.5)], [],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    me.materials.append(bpy.data.materials.new("test_slab:wood"))
    path = Path(tempfile.mkdtemp()) / "library.blend"
    bpy.data.libraries.write(str(path), {me}, fake_user=True)
    clear()
    manifest = SimpleNamespace(assets=[SimpleNamespace(id="test_slab", role="dressing"),
                                       SimpleNamespace(id="not_in_library", role="dressing")])
    ctx = dressing.Ctx(SimpleNamespace(m=manifest, library=path))
    template = ctx.asset("test_slab")
    missing = [ctx.asset("not_in_library") is None, ctx.asset("no_such_model") is None]
    d = Dressing("picture_frame", "asset:test_slab", (4.0, 1.0, 1.2), 90.0, (0.03, 0.8, 0.9))
    ob = dressing.fitted(d, template, "lib")
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    result = {"mesh": template.data.name, "materials": [m.name for m in template.data.materials], "missing": missing,
              "world": world_box(ob), "box": list(d.box)}
    ctx.release()
    result["loose"] = sorted(o.name for o in bpy.data.objects if o.name != "lib")
    return result


def builder_contract(spec: dict) -> list[dict]:
    """What each builder returns when called on its own, as the dispatcher calls it: how many objects, whether any is
    linked to a collection, carries a pass index or a transform, and where its geometry lies in the piece's frame."""
    clear()
    ctx = dressing.Ctx()
    builders = dressing.builders_of(args.catalog)
    rows = []
    for d in dressing.layout_of(args.catalog, spec):
        objs = builders_called(d, ctx, builders)
        local = []
        for ob in objs:
            co = np.empty(len(ob.data.vertices) * 3)
            ob.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            local.append([round(float(v), 6) for v in (*co.min(axis=0), *co.max(axis=0))])
        rows.append({"cls": d.cls, "n": len(objs), "linked": [o.name for o in objs if o.users_collection],
                     "pass_index": [o.pass_index for o in objs],
                     "identity": [bool(o.matrix_basis == o.matrix_basis.Identity(4)) for o in objs], "local": local,
                     "materials": [[m.name if m else None for m in o.data.materials] for o in objs]})
    return rows


def builders_called(d, ctx, builders) -> list:
    return dressing.draw(d, ctx, builders)


def main() -> None:
    spec = json.loads(resources.files("stbench.maps").joinpath(f"{args.map}.json").read_bytes())
    objs1, t1 = build_once(spec)
    pieces = dressing.layout_of(args.catalog, spec)
    rows = []
    for ob in objs1:
        rows.append({"name": ob.name, "mesh": ob.data.name, "pass_index": ob.pass_index,
                     "collections": [c.name for c in ob.users_collection], "box": world_box(ob), "local": local_box(ob),
                     "loc": [round(v, 6) for v in ob.location], "rot_z": round(ob.rotation_euler.z, 6),
                     "verts": len(ob.data.vertices), "polys": len(ob.data.polygons),
                     "materials": [m.name if m else None for m in ob.data.materials]})
    d1 = digest(objs1)
    first_names = [o.name for o in objs1]
    objs2, t2 = build_once(spec)
    result = {"pieces": [[d.cls, d.recipe, list(d.box), list(d.size), list(d.pos), d.yaw_deg] for d in pieces],
              "objects": rows, "digest": [d1, digest(objs2)], "names": [first_names, [o.name for o in objs2]],
              "seconds": [round(t1, 3), round(t2, 3)],
              "other_objects": sorted(o.name for o in bpy.data.objects if o.name not in {x.name for x in objs2}),
              "collections": sorted(c.name for c in bpy.data.collections), "fit": fit_check()}
    result["library"] = library_check()
    result["builders"] = builder_contract(spec)
    dressing.DEBUG_INDEX_FROM = 1000  # a realism probe's build: each piece has an index of its own
    probed, _ = build_once(spec)
    named = {str(k): v for k, v in dressing.INDEX.items()}
    result["debug"] = {"indices": [o.pass_index for o in probed], "index": named}
    dressing.DEBUG_INDEX_FROM = None
    plain, _ = build_once(spec)
    result["debug"]["after"] = sorted({o.pass_index for o in plain})
    clear()
    try:
        dressing.build(spec, "no_such_catalog", dressing.Ctx())
        unknown = "built"
    except look.LookError:
        unknown = "LookError"
    result["no_layout"] = [unknown, sorted(c.name for c in bpy.data.collections), len(bpy.data.objects)]
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(f"STBENCH-DRESSING {len(objs1)} objects in {t1:.2f} s", flush=True)


main()
