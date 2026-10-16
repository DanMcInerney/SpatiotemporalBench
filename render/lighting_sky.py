# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The sky and the sun: the world node tree and the one sun lamp, for every appearance and map.

Runs inside Blender's own Python and is never imported by `stbench`. Two skies:
- the procedural one (Nishita sky, `sun_disc` off), which the placeholder appearance and any map whose manifest names
  no HDRI keep, exactly as `build_scene` always built it (an indoor map with the light through its windows, an outdoor
  map with its own sun and sky strength, its light desaturated: `SKY_LIGHT_DESAT`);
- an HDRI of the manifest (`hdri`, the file `asset_manifest.hdri_path` names) as `hdri_remap` turns it (clamped, turned
  by `heading_deg`, limited to a band of its columns, its sun brought to the lamp's elevation), seen by the camera at
  `camera_strength` and lighting the scene, through the windows' portals, at `light_strength` (and, outdoors, paler by
  `light_desaturation`, as the procedural sky's light is): two backgrounds mixed by Light Path "Is Camera Ray", so how
  bright the street looks through a window and how much daylight enters can be set apart (the downtown's are). The
  HDRI's own sun is clamped away (`SUN_CLAMP`): the sun lamp is the sun, from the map's direction, and a second sun in
  the picture would cast a second shadow. With `sky_lux` instead (the home's) there is one strength, for the camera and
  the light: the one at which the picture, sun clamped, lights a horizontal surface with that many lux
  (`hdri_remap.horizontal_illuminance`), so that the sky is in its real ratio to the sun and the lamps.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import hdri_remap
import numpy as np
from cycles_setup import UNIT_LUX
from mathutils import Vector

SUN = {"elevation_deg": 38.0, "rotation_deg": 215.0, "strength": 1.6, "sky": 0.3}  # indoors, procedural: the original
SKY_LIGHT_DESAT = 0.5  # outdoors, the sky's light is this much desaturated: its look to the camera is not (look-dev)
SUN_CLAMP = hdri_remap.SUN_CLAMP  # an HDRI's pixel is clipped to this radiance (`hdri_remap`)


def sun_direction(elevation_deg: float, rotation_deg: float) -> Vector:
    """The unit vector towards the sun: azimuth `rotation_deg` (clockwise from north, as Blender's sky texture reads
    it) and elevation above the horizon, in the map frame (x east, y north, z up)."""
    az, el = math.radians(rotation_deg), math.radians(elevation_deg)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


def _procedural(nt, bg, cfg: dict, outdoor) -> None:
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(cfg["elevation_deg"])
    sky.sun_rotation = math.radians(cfg["rotation_deg"])
    if outdoor is None:
        nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    else:  # the light the sky casts is desaturated, the sky the camera sees is not
        hs = nt.nodes.new("ShaderNodeHueSaturation")
        hs.inputs["Saturation"].default_value = 1.0 - SKY_LIGHT_DESAT
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        lp = nt.nodes.new("ShaderNodeLightPath")
        nt.links.new(sky.outputs["Color"], hs.inputs["Color"])
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
        nt.links.new(hs.outputs["Color"], mix.inputs[6])
        nt.links.new(sky.outputs["Color"], mix.inputs[7])
        nt.links.new(mix.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = cfg["sky"]


def _hdri(nt, out, path: Path, hdri: dict) -> None:
    """The picture the world shows (`hdri_remap`, from the manifest's file) -> environment texture -> two backgrounds
    (the camera's and the light's) mixed by Is Camera Ray -> output."""
    n = nt.nodes
    src = bpy.data.images.load(str(path), check_existing=True)
    w, h = src.size
    px = np.empty(w * h * 4, dtype=np.float32)
    src.pixels.foreach_get(px)
    shown = np.ones((h, w, 4), dtype=np.float32)
    shown[..., :3] = hdri_remap.remap(px.reshape(h, w, 4)[..., :3], hdri.get("heading_deg", 0.0), hdri.get("band_u"),
                                      hdri.get("disc"), SUN_CLAMP)
    if "sky_lux" in hdri:  # one strength for the camera and the light: the picture then lights a horizontal surface
        # with the sky's lux (a unit of irradiance is UNIT_LUX / pi lux; the sun disc is already clamped away)
        hdri["camera_strength"] = hdri["light_strength"] = (
            hdri["sky_lux"] * math.pi / UNIT_LUX / hdri_remap.horizontal_illuminance(shown[..., :3]))
    img = bpy.data.images.new(f"sky_{Path(path).stem}", w, h, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Linear Rec.709"
    img.pixels.foreach_set(shown.ravel())
    bpy.data.images.remove(src)
    env = n.new("ShaderNodeTexEnvironment")
    env.image = img
    seen, lit = n.new("ShaderNodeBackground"), n.new("ShaderNodeBackground")
    lp, mix = n.new("ShaderNodeLightPath"), n.new("ShaderNodeMixShader")
    for bg, key in ((seen, "camera"), (lit, "light")):
        bg.inputs["Strength"].default_value = hdri[f"{key}_strength"]
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    if hdri.get("light_desaturation"):  # as the procedural sky outdoors: its light is paler than the sky one sees
        hs = n.new("ShaderNodeHueSaturation")
        hs.inputs["Saturation"].default_value = 1.0 - hdri["light_desaturation"]
        nt.links.new(env.outputs["Color"], hs.inputs["Color"])
        nt.links.new(hs.outputs["Color"], lit.inputs["Color"])
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(lit.outputs["Background"], mix.inputs[1])
    nt.links.new(seen.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def build(sc, cfg: dict, outdoor=None, hdri_path: Path | None = None, hdri: dict | None = None) -> str:
    """The world and the sun for a scene. `cfg` is the sun's `elevation_deg`, `rotation_deg` and `strength` and the
    procedural sky's `sky` strength; `outdoor` the map's `Outdoor` (it changes how the procedural sky lights). With
    `hdri_path` and `hdri` (`camera_strength`, `light_strength` and `hdri_remap.remap`'s `heading_deg`, `band_u` and
    `disc`) the sky is the HDRI. Returns "hdri" or "procedural", what the world is."""
    w = bpy.data.worlds.new("sky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBackground")
    if hdri_path is None:
        _procedural(nt, bg, cfg, outdoor)
    else:
        out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld")
        nt.nodes.remove(bg)
        _hdri(nt, out, hdri_path, hdri)
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy, ld.angle = cfg["strength"], math.radians(0.53)
    ob = bpy.data.objects.new("sun", ld)
    sc.collection.objects.link(ob)
    # the sun shines from azimuth `rotation_deg` at the elevation
    ob.rotation_euler = sun_direction(cfg["elevation_deg"], cfg["rotation_deg"]).to_track_quat("Z", "Y").to_euler()
    return "procedural" if hdri_path is None else "hdri"
