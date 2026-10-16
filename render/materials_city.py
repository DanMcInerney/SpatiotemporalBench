# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's procedural vehicle and prop materials: the fixed-colour shaders its vehicles and props take (car
paint, dark glass, tyres, lamps, rubble). The street's own surfaces (asphalt, paving, sidewalk, kerb, road paint) are
`materials_street`'s; `asphalt` and `paving` are imported here for `shader` and `build_scene`.

Runs inside Blender's own Python and is never imported by `stbench`. `materials.shader_material` hands every name of
`NAMES` to `shader` (the manifest's `shader:<name>`); the street surfaces are also built by the scene directly. No image
files and no UVs: noise, ramps and math on object coordinates, as the warehouse's concrete is. Nothing is cached here (a
scene rebuild frees every material): the caller keeps what it builds.

Colours are given as sRGB and converted. Vehicle paint is a glossier Principled material with a light clear coat
(roughness 0.4, coat 0.2; a strong coat turned red roofs pink with the sky); glass is rough enough (0.2) that no
recognisable reflection of another vehicle shows, which would disagree with the ID pass.
"""

from __future__ import annotations

import bpy
from materials_street import asphalt, lin, paving
from shader_nodes import new_material, principled

FIXED = {  # name -> (sRGB, roughness, metallic, extra Principled inputs)
    "car_glass": ((0.015, 0.02, 0.03), 0.2, 0.0, {"Specular IOR Level": 0.6}),
    "tyre": ((0.05, 0.05, 0.055), 0.85, 0.0, {}),
    "metal": ((0.60, 0.60, 0.62), 0.35, 0.88, {}),
    "trim_dark": ((0.08, 0.08, 0.09), 0.6, 0.0, {}),
    "liner": ((0.12, 0.12, 0.12), 0.9, 0.0, {}),
    "matte_white": ((0.88, 0.88, 0.86), 0.5, 0.0, {}),
    "matte_grey": ((0.45, 0.45, 0.47), 0.55, 0.0, {}),
    "matte_red": ((0.80, 0.07, 0.07), 0.45, 0.0, {}),
    "matte_yellow": ((0.95, 0.80, 0.08), 0.45, 0.0, {}),
    "wood": ((0.52, 0.36, 0.20), 0.7, 0.0, {}),
    "rubble_grey": ((0.50, 0.48, 0.44), 0.95, 0.0, {}),
    "rubble_brick": ((0.55, 0.28, 0.20), 0.9, 0.0, {}),
    "rubble_dirt": ((0.40, 0.30, 0.20), 0.9, 0.0, {}),
}
LAMPS = {  # name -> (sRGB, emission strength): lit parts whose base colour is a darker copy of the glow
    "headlight": ((1.0, 0.97, 0.88), 2.0), "taillight": ((1.0, 0.05, 0.03), 2.0),
    "amber_lamp": ((1.0, 0.55, 0.05), 3.0),
    "sign_lamp": ((1.0, 0.92, 0.70), 3.0)}
NAMES = ("asphalt", "paving", "paint", "roof_panel", *FIXED, *LAMPS)
PAINT = {"roughness": 0.4, "coat": 0.2, "coat_roughness": 0.15}


def _plain(name: str, rgb, rough: float, metal: float = 0.0, **extra) -> bpy.types.Material:
    m = new_material(name)
    p = principled(m)
    p.inputs["Base Color"].default_value = (*lin(rgb), 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    for k, v in extra.items():
        p.inputs[k].default_value = v
    return m


def paint() -> bpy.types.Material:
    """Car paint: semi-gloss with a light clear coat, tinted by the entity's colour attribute."""
    return _plain("shader:paint", (0.9, 0.9, 0.9), PAINT["roughness"], **{
        "Coat Weight": PAINT["coat"], "Coat Roughness": PAINT["coat_roughness"]})


def glow(name: str, rgb, strength: float) -> bpy.types.Material:
    """A lit part: its base colour a darker copy of the glow (`rgb`, sRGB)."""
    m = new_material(name)
    p = principled(m)
    p.inputs["Base Color"].default_value = (*lin(rgb, 0.4), 1.0)
    p.inputs["Emission Color"].default_value = (*lin(rgb), 1.0)
    p.inputs["Emission Strength"].default_value = strength
    p.inputs["Roughness"].default_value = 0.3
    return m


def roof_panel() -> bpy.types.Material:
    """The milky roof of a portable toilet: a light, faintly glowing off-white."""
    m = new_material("shader:roof_panel")
    p = principled(m)
    p.inputs["Base Color"].default_value = (*lin((0.92, 0.93, 0.90)), 1.0)
    p.inputs["Emission Color"].default_value = (*lin((0.95, 0.97, 0.92)), 1.0)
    p.inputs["Emission Strength"].default_value = 0.35
    p.inputs["Roughness"].default_value = 0.5
    return m


def shader(name: str) -> bpy.types.Material:
    """One of `NAMES`, the manifest's `shader:<name>` materials of the city."""
    if name == "asphalt":
        return asphalt()
    if name == "paving":
        return paving()
    if name == "paint":
        return paint()
    if name == "roof_panel":
        return roof_panel()
    if name in LAMPS:
        return glow(f"shader:{name}", *LAMPS[name])
    rgb, rough, metal, extra = FIXED[name]
    return _plain(f"shader:{name}", rgb, rough, metal, **extra)
