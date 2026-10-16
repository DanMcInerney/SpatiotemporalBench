# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The looks only the household's dressing asks for and the material library lacks (`dressing.Ctx.material`).

Runs inside Blender's own Python and is never imported by `stbench`. A dressing builder names the look it wants
(`painted_wood_trim`, `brushed_steel`, `ceramic_tile`); the render's material library answers where it has that name
(`materials.dressing_material`, lane M) and this module answers for every other name, with a plain Principled material
or a small procedural one, so a scene builds with no download and no texture file. Every look is a finish, never a
tracked colour on a tracked kind's shape: the whites are off-white, the fabrics are muted linen, the art is ochre, sage
and slate. Deterministic: no noise has a seed of its own and nothing reads a clock.
"""

from __future__ import annotations

from collections.abc import Callable

import bpy
from shader_nodes import G
from shapes import srgb_to_linear

FILM = "stbench_film"  # custom property of a material that is clear in places (a film of dust)


def _linear(rgb: tuple[float, float, float]) -> tuple[float, float, float, float]:
    return (*(srgb_to_linear(c) for c in rgb), 1.0)


def _new(name: str) -> tuple[bpy.types.Material, bpy.types.Node]:
    m = bpy.data.materials.new(f"dressing:{name}")
    m.use_nodes = True
    return m, next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")


def pbr(name: str, rgb: tuple[float, float, float], rough: float, metal: float = 0.0, **inputs) -> bpy.types.Material:
    """A Principled material of an sRGB colour, a roughness and a metalness; `inputs` set any other socket by name
    (underscores for spaces: `Coat_Weight=0.2`)."""
    m, p = _new(name)
    p.inputs["Base Color"].default_value = _linear(rgb)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    for key, value in inputs.items():
        p.inputs[key.replace("_", " ")].default_value = value
    return m


def _bump(m: bpy.types.Material, p, height, strength: float, distance: float) -> None:
    bump = m.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value, bump.inputs["Distance"].default_value = strength, distance
    m.node_tree.links.new(height, bump.inputs["Height"])
    m.node_tree.links.new(bump.outputs["Normal"], p.inputs["Normal"])


def fabric(name: str, rgb: tuple[float, float, float], rough: float = 0.85, weave: float = 600.0) -> bpy.types.Material:
    """Woven cloth: a fine noise as a bump, and a faint slow variation in the colour (linen is never one flat tone)."""
    m, p = _new(name)
    nt = m.node_tree
    coord, grain, slow = (nt.nodes.new(t) for t in ("ShaderNodeTexCoord", "ShaderNodeTexNoise", "ShaderNodeTexNoise"))
    grain.inputs["Scale"].default_value, grain.inputs["Detail"].default_value = weave, 2.0
    slow.inputs["Scale"].default_value, slow.inputs["Detail"].default_value = 6.0, 3.0
    for n in (grain, slow):
        nt.links.new(coord.outputs["Object"], n.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = _linear(tuple(c * 0.93 for c in rgb))
    ramp.color_ramp.elements[1].color = _linear(tuple(min(1.0, c * 1.05) for c in rgb))
    nt.links.new(slow.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = rough
    _bump(m, p, grain.outputs["Fac"], 0.35, 0.0015)
    return m


def art(name: str, stops: tuple[tuple[float, float, float], ...], style: str) -> bpy.types.Material:
    """Stand-in artwork on a picture's canvas, in muted colours, never the same twice: the object's own random number (a
    hash of its name, so the same in every run) slides the pattern along the canvas by up to ten metres. `bands` is
    horizontal bands that a noise bends (a landscape), `blocks` irregular fields of colour (an abstract). Object
    coordinates: y across the canvas, z up."""
    m, p = _new(name)
    nt = m.node_tree
    new, link = nt.nodes.new, nt.links.new
    coord, sep, info = new("ShaderNodeTexCoord"), new("ShaderNodeSeparateXYZ"), new("ShaderNodeObjectInfo")
    slide, across, plane = new("ShaderNodeMath"), new("ShaderNodeMath"), new("ShaderNodeCombineXYZ")
    slide.operation, slide.inputs[1].default_value = "MULTIPLY", 10.0
    across.operation = "ADD"
    link(coord.outputs["Object"], sep.inputs["Vector"])
    link(info.outputs["Random"], slide.inputs[0])
    link(sep.outputs["Y"], across.inputs[0])
    link(slide.outputs[0], across.inputs[1])
    link(across.outputs[0], plane.inputs["X"])
    link(sep.outputs["Z"], plane.inputs["Y"])
    ramp, noise, bend, fit = (new(t) for t in ("ShaderNodeValToRGB", "ShaderNodeTexNoise", "ShaderNodeMath",
                                               "ShaderNodeMapRange"))
    els = ramp.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for i, rgb in enumerate(stops):
        els[i].position, els[i].color = i / (len(stops) - 1), _linear(rgb)
    ramp.color_ramp.interpolation = "CONSTANT"
    link(plane.outputs["Vector"], noise.inputs["Vector"])
    if style == "bands":  # the noise bends the height: z + 0.55 noise, fitted to the ramp
        noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = 2.5, 3.0
        bend.operation, bend.inputs[1].default_value = "MULTIPLY_ADD", 0.55
        fit.inputs["From Min"].default_value, fit.inputs["From Max"].default_value = 0.0, 0.8
        link(noise.outputs["Fac"], bend.inputs[0])
        link(sep.outputs["Z"], bend.inputs[2])
        link(bend.outputs[0], fit.inputs["Value"])
        link(fit.outputs["Result"], ramp.inputs["Fac"])
    else:  # soft-edged fields of colour
        noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = 3.2, 1.0
        noise.inputs["Distortion"].default_value = 0.4
        fit.inputs["From Min"].default_value, fit.inputs["From Max"].default_value = 0.3, 0.7
        link(noise.outputs["Fac"], fit.inputs["Value"])
        link(fit.outputs["Result"], ramp.inputs["Fac"])
    link(ramp.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.7
    return m


def decal(name: str, rgb: tuple[float, float, float], rough: float, opacity: Callable[[G, object], object]
          ) -> bpy.types.Material:
    """A film of dust or dirt: the colour where `opacity(g, obj)` (a 0 to 1 socket from the object's coordinates) says
    there is any, clear elsewhere. For flush grime: a plate a few millimetres thick whose soft edge is the mask's."""
    m, p = _new(name)
    m[FILM] = True  # `dressing.build` sees to the data passes of a scene that holds one
    nt = m.node_tree
    g = G(nt)
    p.inputs["Base Color"].default_value = _linear(rgb)
    p.inputs["Roughness"].default_value = rough
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs["BSDF"], mix.inputs[1])
    nt.links.new(p.outputs["BSDF"], mix.inputs[2])
    nt.links.new(opacity(g, g.node("ShaderNodeTexCoord").outputs["Object"]), mix.inputs["Fac"])
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def _dust(g: G, obj):
    """Thickest at the wall (the piece's x = -depth/2, `DUST_DEPTH` 3.5 cm), thinning to nothing 3.5 cm out, in patches
    along the wall, never opaque."""
    across = g.smooth(g.xyz(obj)[0], 0.0175, -0.0175)
    patches = g.smooth(g.noise(g.scaled(obj, 14.0, 3.0, 14.0), 3.0, 3.0, 0.55), 0.35, 0.7)
    return g.math("MULTIPLY", g.math("MULTIPLY", across, g.math("MULTIPLY_ADD", patches, 0.7, 0.3)), 0.7)


def _scuff(g: G, obj):
    """A mark of soles, 45 cm along x and 10 cm across: streaks along x, soft at the ends and the sides."""
    x, y, _ = g.xyz(obj)
    soft = g.math("MULTIPLY", g.smooth(g.math("ABSOLUTE", x), 0.225, 0.12), g.smooth(g.math("ABSOLUTE", y), 0.05, 0.02))
    streaks = g.smooth(g.noise(g.scaled(obj, 5.0, 120.0, 5.0), 3.0, 3.0, 0.6), 0.35, 0.7)
    return g.math("MULTIPLY", g.math("MULTIPLY", soft, g.math("MULTIPLY_ADD", streaks, 0.7, 0.3)), 0.4)


# What a name is: a function that makes the material. These are the looks only dressing asks for; the plain finishes
# (`painted_wood_trim`, `white_plastic`, `brushed_steel`, `ceramic_tile`, `curtain_fabric`, `led_emitter_warm` ...) are
# the material library's (`materials.dressing_material`).
FALLBACKS: dict[str, Callable[[], bpy.types.Material]] = {
    "shower_curtain_vinyl": lambda: fabric("shower_curtain_vinyl", (0.86, 0.83, 0.74), 0.55, 220.0),
    "paper_tissue": lambda: fabric("paper_tissue", (0.93, 0.92, 0.88), 0.9, 900.0),
    "cardboard_core": lambda: pbr("cardboard_core", (0.55, 0.42, 0.28), 0.9),
    "clock_face": lambda: pbr("clock_face", (0.90, 0.88, 0.80), 0.55),
    "frame_black": lambda: pbr("frame_black", (0.07, 0.07, 0.07), 0.45),
    "frame_white": lambda: pbr("frame_white", (0.90, 0.89, 0.86), 0.4),
    "frame_walnut": lambda: pbr("frame_walnut", (0.20, 0.12, 0.07), 0.45),
    "frame_gilt": lambda: pbr("frame_gilt", (0.55, 0.42, 0.20), 0.35, 0.8),
    "mat_board": lambda: pbr("mat_board", (0.91, 0.90, 0.86), 0.85),
    "art_bands": lambda: art("art_bands", ((0.76, 0.66, 0.46), (0.50, 0.55, 0.42), (0.38, 0.47, 0.55),
                                           (0.82, 0.78, 0.66)), "bands"),
    "art_blocks": lambda: art("art_blocks", ((0.60, 0.40, 0.30), (0.78, 0.72, 0.58), (0.36, 0.42, 0.44),
                                             (0.86, 0.84, 0.78)), "blocks"),
    "fan_blade": lambda: pbr("fan_blade", (0.66, 0.52, 0.36), 0.5),
    # the grime of `dressing_build_grime`: dust and a scuff are films, hairs and crumbs are solids of their own colour
    "dust": lambda: decal("dust", (0.60, 0.57, 0.52), 1.0, _dust),
    "scuff_mark": lambda: decal("scuff_mark", (0.09, 0.08, 0.075), 0.55, _scuff),
    "hair_dark": lambda: pbr("hair_dark", (0.07, 0.05, 0.04), 0.4),
    "hair_light": lambda: pbr("hair_light", (0.42, 0.32, 0.20), 0.4),
    "crumb": lambda: pbr("crumb", (0.55, 0.42, 0.26), 0.9),
}
