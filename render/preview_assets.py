# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Render the tiles of the asset contact sheets: every kind in every catalog colour from four sides, every dressing
model from four sides, every textured surface on a plane and a sphere.

    blender --background --factory-startup --python render/preview_assets.py -- \
        --manifest M --library L.blend --jobs JOBS.json --out TILES --src REPO/src [--samples N] [--tile WxH]

Started by `stbench assets preview` (which writes the job list and assembles the sheets); never imported by `stbench`.
Each entity is built exactly as an episode builds it (`Appearance.make_object`) under the sun and sky of the episode it
belongs to (the household's indoors; a kind of an outdoor map's catalog gets that map's sun and sky, on its asphalt) and
rendered on the GPU. A dressing model is shown as the library holds it (front toward -y; the household's light). Writes
`<out>/<kind>/<colour>__<view>.png`, `<out>/_dressing/<id>__<view>.png` and `<out>/_surfaces/<id>.png` (a plane) with
`<id>__sphere.png`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_scene  # noqa: E402
import materials_city  # noqa: E402
from asset_dressing import DressingAssets  # noqa: E402
from asset_entities import Appearance, squash  # noqa: E402
from cycles_setup import configure  # noqa: E402
from shapes import link, material, polygon_mesh  # noqa: E402

ELEVATION_DEG, FOV_DEG, VIEWS = 32.0, 38.0, 4
FRONT_DEG = 20.0  # the first view looks at an entity's front (+x), 20 degrees round; the others turn a quarter each
DRESSING_ELEVATION_DEG = 12.0
DRESSING_AZIMUTHS = (-90.0, -45.0, -135.0, 90.0)  # the front (it faces -y), two three-quarter views, the back
FLOOR_HALF_M = 5.0  # the floor under a kind reaches at least this far, and four times its radius


def absolute(text: str) -> Path:
    """Blender's file API reads a relative path against the blend file, not the working directory."""
    return Path(text).resolve()


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=absolute, required=True)
    ap.add_argument("--library", type=absolute, required=True)
    ap.add_argument("--jobs", type=absolute, required=True)
    ap.add_argument("--out", type=absolute, required=True)
    ap.add_argument("--src", type=absolute, required=True, help="the directory that holds the stbench package")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--tile", default="192x144")
    return ap.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def look_at(cam, eye: Vector, target: Vector) -> None:
    cam.location = eye
    cam.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()


def render_to(sc, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = sc.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "IMAGE"
    s.file_format, s.color_mode = "PNG", "RGB"
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def orbit(cam, centre: Vector, radius: float, azimuth_deg: float, elevation_deg: float = ELEVATION_DEG) -> None:
    """Place the camera on a sphere round `centre` at a distance that fits a sphere of `radius` in the frame."""
    dist = 1.15 * radius / math.sin(math.radians(FOV_DEG / 2))
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    out = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    look_at(cam, centre + dist * out, centre)


def light(sc, outdoor: dict | None) -> None:
    """The sky and sun of the kind's episode: replace the scene's with the household's (None) or an outdoor map's."""
    from stbench.mapspec import Outdoor

    for ob in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(ob)
    for ld in list(bpy.data.lights):  # removing the object leaves its light data behind, one more per change of sky
        bpy.data.lights.remove(ld)
    for w in list(bpy.data.worlds):
        bpy.data.worlds.remove(w)
    build_scene._build_sky(sc, Outdoor(**outdoor) if outdoor else None)


def sphere(name: str, radius: float, mat: bpy.types.Material) -> bpy.types.Object:
    """A smooth ball of `radius` resting on z = 0, in `mat` (a material projects its texture on object coordinates)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=radius)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = (0.0, 0.0, radius)
    sc = bpy.context.scene
    sc.collection.objects.link(ob)
    return ob


def dressing_tiles(sc, cam, args, dressing: DressingAssets, items: list[dict]) -> None:
    """Each dressing model on a floor, from `DRESSING_AZIMUTHS`; a model the library lacks is an error here (the
    preview's gate has checked the library against the manifest)."""
    for item in items:
        ob = dressing.asset(item["id"])
        if ob is None:
            raise LookupError(f"the library has no dressing model {item['id']}")
        sc.collection.objects.link(ob)
        size = Vector(dressing.native(item["id"]))
        g = max(FLOOR_HALF_M, 4 * size.length / 2)
        link("floor", polygon_mesh("floor", [(-g, -g), (g, -g), (g, g), (-g, g)], -0.001, material("underlay")))
        for v, azimuth in enumerate(DRESSING_AZIMUTHS):
            orbit(cam, Vector((0, 0, size.z / 2)), size.length / 2, azimuth, DRESSING_ELEVATION_DEG)
            render_to(sc, args.out / "_dressing" / f"{item['id']}__{v}.png")
        bpy.data.objects.remove(ob)
        bpy.data.objects.remove(bpy.data.objects["floor"])


def main() -> None:
    args = parse(sys.argv)
    sys.path.insert(0, str(args.src))
    from stbench import asset_manifest

    m, jobs = asset_manifest.load(args.manifest), json.loads(args.jobs.read_text(encoding="utf-8"))
    w, h = (int(v) for v in args.tile.split("x"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    configure(sc, w, h, args.samples, "assets")
    lit, road = "unset", None  # the sky of the last kind's map; the asphalt of an outdoor map's kinds, built once
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.sensor_fit, cam.data.angle = "HORIZONTAL", math.radians(FOV_DEG)
    sc.collection.objects.link(cam)
    sc.camera = cam
    app = Appearance(m, args.library, [i["asset"] for i in jobs["items"]])
    fit = {}
    for item in jobs["items"]:  # how far each catalog box distorts its asset: the sheet's title says so
        name = item.get("name", item["kind"])  # a clutter kind has several models, each a sheet of its own
        fit[name] = round(squash(m.asset(item["asset"]), app.native[item["asset"]], tuple(item["size"])), 3)
        if item.get("outdoor") != lit:
            lit = item.get("outdoor")
            light(sc, lit)
        g = max(FLOOR_HALF_M, 4 * Vector(item["size"]).length / 2)
        if lit and road is None:
            road = materials_city.asphalt()
        link("floor", polygon_mesh("floor", [(-g, -g), (g, -g), (g, g), (-g, g)], -0.001,
                                   road if lit else material("underlay")))
        for colour in item["colours"]:
            ob = app.make_object(item["asset"], item["kind"], colour, item["size"], f"{name}_{colour}", 1,
                                 strict=False)
            size = Vector(item["size"])
            for v in range(VIEWS):
                orbit(cam, Vector((0, 0, size.z / 2)), size.length / 2, FRONT_DEG + 90 * v)
                render_to(sc, args.out / name / f"{colour}__{v}.png")
            bpy.data.objects.remove(ob)
        bpy.data.objects.remove(bpy.data.objects["floor"])
    (args.out / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    if lit is not None:
        light(sc, None)
    dressing_tiles(sc, cam, args, DressingAssets(m, args.library), jobs.get("dressing", []))
    for s in jobs["surfaces"]:  # a 1.2 m square of each material, seen from above at an angle, and a sphere of it
        mat = app.materials[s["id"]]
        side = max(1.2, 2.5 * (s["tile_m"] or [1.0])[0])
        h2 = side / 2
        corners = [(-h2, -h2), (h2, -h2), (h2, h2), (-h2, h2)]
        square = link(f"surface_{s['id']}", polygon_mesh("square", corners, 0.0, mat))
        orbit(cam, Vector((0, 0, 0)), side / 2.2, 90)
        render_to(sc, args.out / "_surfaces" / f"{s['id']}.png")
        bpy.data.objects.remove(square)
        ball = sphere(f"sphere_{s['id']}", side / 3, mat)
        orbit(cam, Vector((0, 0, side / 3)), side / 3, 90)
        render_to(sc, args.out / "_surfaces" / f"{s['id']}__sphere.png")
        bpy.data.objects.remove(ball)
    print(f"STBENCH-PREVIEW {sum(len(i['colours']) for i in jobs['items']) * VIEWS} tiles, "
          f"{len(jobs.get('dressing', [])) * VIEWS} dressing tiles, {len(jobs['surfaces'])} surfaces", flush=True)


main()
