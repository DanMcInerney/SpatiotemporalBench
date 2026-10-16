# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Grey cards: how much light a built scene gives a place, read in the linear unit.

Runs inside Blender's own Python and is never imported by `stbench`. `tests/blender_lighting_driver.py` and the
realism probe (`stbench realism probe`) both call it. A card is a 0.1 m square of albedo 0.18, flat to the surface it
reads, seen by an orthographic camera 0.3 m away: under `cycles_setup.UNIT_LUX` lux it reads 0.18, so a card that
reads Y was lit by Y / 0.18 x UNIT_LUX lux. `measure` renders on the CPU, with the sources it is asked for on (the
lamps alone, the daylight alone, or all), and leaves the scene as it found it: the lights, the world, the camera and
the render settings it touched are put back, and the card and its camera are removed."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import bpy
import numpy as np
from cycles_setup import UNIT_LUX
from mathutils import Vector

CARD_ALBEDO, CARD_SIZE_M, CARD_DISTANCE_M = 0.18, 0.1, 0.3
LUMA = (0.2126, 0.7152, 0.0722)
SOURCES = ("lamps", "daylight", "all")


def card_points(m, z: float = 0.75, fractions=(0.2, 0.5, 0.8)) -> list[dict]:
    """A card flat (facing up) at height `z` on a grid of `fractions` x `fractions` of every region's bounding box: 3
    x 3 a room by default. `{"room": the region's name, "pos": [x, y, z], "normal": [0, 0, 1]}`."""
    out = []
    for region in m.regions:
        xs, ys = [p[0] for p in region.polygon], [p[1] for p in region.polygon]
        for fx in fractions:
            for fy in fractions:
                pos = [min(xs) + fx * (max(xs) - min(xs)), min(ys) + fy * (max(ys) - min(ys)), z]
                out.append({"room": region.name, "pos": pos, "normal": [0.0, 0.0, 1.0]})
    return out


def surface_points(m, kind: str, fractions=(0.25, 0.5, 0.75)) -> list[dict]:
    """A card flat on every furniture surface of `kind` (the map's `Surface.kind`, "kitchen counter" for a worktop),
    on a grid of `fractions` x `fractions` of the surface: `{"name": the surface's id, "pos", "normal": [0, 0, 1]}`."""
    from stbench.geometry import (  # not at the top: a script puts `stbench` on the path only when it starts
        Rect,
        to_world,
    )

    out = []
    for f in m.furniture:
        frame = Rect(f.center[0], f.center[1], 0.0, 0.0, f.yaw)
        for s in (s for s in f.surfaces if s.kind == kind):
            for fx in fractions:
                for fy in fractions:
                    x, y = to_world(frame, s.rect[0] + (fx - 0.5) * s.rect[2], s.rect[1] + (fy - 0.5) * s.rect[3])
                    out.append({"name": s.id, "pos": [x, y, s.z], "normal": [0.0, 0.0, 1.0]})
    return out


class _Restore:
    """Attribute changes made for a probe, undone in reverse order."""

    def __init__(self) -> None:
        self.saved: list[tuple[object, str, object]] = []

    def set(self, obj, name: str, value) -> None:
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self) -> None:
        for obj, name, value in reversed(self.saved):
            with contextlib.suppress(TypeError, ValueError):  # a setting that does not apply to the format it had
                setattr(obj, name, value)


def _lights(kind: str) -> list:
    """The scene's lights of a kind: "sun", or "lamp" (an area light that is not a portal)."""
    lamp = {"sun": lambda ld: ld.type == "SUN", "lamp": lambda ld: ld.type == "AREA" and not ld.cycles.is_portal}[kind]
    return [ob for ob in bpy.data.objects if ob.type == "LIGHT" and lamp(ob.data)]


def _black_world():
    world = bpy.data.worlds.new("probe_black")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    return world


def _card(sc):
    """The grey card (a plane of albedo 0.18, rough, no specular) and the camera that reads it."""
    h = CARD_SIZE_M / 2
    mesh = bpy.data.meshes.new("probe_card")
    mesh.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new("probe_card")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (CARD_ALBEDO, CARD_ALBEDO, CARD_ALBEDO, 1)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    mesh.materials.append(mat)
    card = bpy.data.objects.new("probe_card", mesh)
    cam_data = bpy.data.cameras.new("probe_cam")
    cam_data.type, cam_data.ortho_scale = "ORTHO", CARD_SIZE_M / 2
    cam = bpy.data.objects.new("probe_cam", cam_data)
    sc.collection.objects.link(card)
    sc.collection.objects.link(cam)
    return card, cam


def _read_combined(path: Path) -> np.ndarray:
    """The Combined pass of a multilayer float EXR as (H, W, 3), the colour channels picked by name."""
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    i = 0
    while inp.seek_subimage(i, 0):
        spec = inp.spec()
        if "Combined" in ",".join(spec.channelnames):
            rgb = {name.rsplit(".", 1)[-1]: c for c, name in enumerate(spec.channelnames)}
            data = inp.read_image(i, 0, 0, spec.nchannels, oiio.FLOAT)[..., [rgb["R"], rgb["G"], rgb["B"]]]
            inp.close()
            return data
        i += 1
    inp.close()
    raise RuntimeError(f"{path}: the EXR has no Combined part")


def _render_combined(sc) -> np.ndarray:
    bpy.ops.render.render(write_still=False)
    s = sc.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "MULTI_LAYER_IMAGE"
    s.file_format, s.color_depth, s.exr_codec = "OPEN_EXR_MULTILAYER", "32", "ZIP"
    path = Path(tempfile.gettempdir()) / f"stbench_probe_{os.getpid()}.exr"
    bpy.data.images["Render Result"].save_render(str(path), scene=sc)
    try:
        return _read_combined(path)
    finally:
        path.unlink()


def measure(sc, points: list[dict], *, sources: str = "lamps", samples: int = 64) -> list[dict]:
    """Each point (`{"pos", "normal", ...}`) with what a grey card there reads: `lux` (the card's luminance / 0.18 x
    `UNIT_LUX`) and the cast of its colour, `r_g` and `b_g` (red and blue over green, in linear light). `sources` says
    what lights the card: "lamps" (the sun and the sky off: the area lights of the rooms, and daylight panels that are
    materials), "daylight" (the lamps off) or "all". Rendered on the CPU at `samples`, with no denoising; the scene is
    left as it was."""
    if sources not in SOURCES:
        raise ValueError(f"sources must be one of {SOURCES}, got {sources!r}")
    undo = _Restore()
    cy, vl = sc.cycles, sc.view_layers[0]
    for obj, name, value in ((sc.render, "engine", "CYCLES"), (sc.render, "resolution_x", 8),
                             (sc.render, "resolution_y", 8), (sc.render, "resolution_percentage", 100),
                             (cy, "device", "CPU"), (cy, "samples", samples), (cy, "use_denoising", False),
                             (cy, "use_adaptive_sampling", False), (vl, "use_pass_combined", True),
                             (sc, "camera", sc.camera)):
        undo.set(obj, name, value)
    black = None
    if sources == "lamps":
        black = _black_world()
        undo.set(sc, "world", black)
        for ob in _lights("sun"):
            undo.set(ob.data, "energy", 0.0)
    if sources == "daylight":
        for ob in _lights("lamp"):
            undo.set(ob.data, "energy", 0.0)
    s = sc.render.image_settings
    for name in ("file_format", "color_depth", "exr_codec") + (("media_type",) if hasattr(s, "media_type") else ()):
        undo.set(s, name, getattr(s, name))
    card, cam = _card(sc)
    sc.camera = cam
    out = []
    try:
        for pt in points:
            pos, normal = Vector(pt["pos"]), Vector(pt["normal"]).normalized()
            card.location = pos + normal * 0.001
            card.rotation_euler = normal.to_track_quat("Z", "Y").to_euler()
            cam.location = pos + normal * CARD_DISTANCE_M
            cam.rotation_euler = (-normal).to_track_quat("-Z", "Y").to_euler()
            img = _render_combined(sc)
            h, w = img.shape[:2]
            rgb = img[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1].reshape(-1, 3).mean(axis=0)
            y = LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]
            out.append({**pt, "lux": float(y / CARD_ALBEDO * UNIT_LUX), "r_g": float(rgb[0] / rgb[1]),
                        "b_g": float(rgb[2] / rgb[1])})
    finally:
        undo.undo()
        for ob in (card, cam):
            data = ob.data
            bpy.data.objects.remove(ob)
            (bpy.data.meshes if ob is card else bpy.data.cameras).remove(data)
        bpy.data.materials.remove(bpy.data.materials["probe_card"])
        if black is not None:
            bpy.data.worlds.remove(black)
    return out
