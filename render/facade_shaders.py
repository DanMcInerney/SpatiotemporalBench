# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's facade shaders: a window-grid building face, or a glass curtain wall, with a storefront band at the
bottom.

Runs inside Blender's own Python and is never imported by `stbench`. No image file: noise, ramps and math on object
coordinates, as the procedural concrete is, and two small atlases of window interiors made in memory
(`window_atlas.py`). A facade material works on a mesh whose object frame has x along the street face (and z up from the
street): it finds the horizontal coordinate `u` along each face from the face normal (a long face has normal +-y and
runs along x), tiles `BAY` metres across and `FLOOR_H` up above a `GROUND_H` storefront band, cuts a window out of every
cell and gives each window an inside from a hash of its cell and the object's random value, so neighbouring buildings
differ without any per-building data. The styles are those the map names on its walls (`walls[i].facade`).

No window is a black void: each pane is a variant of the window atlas (the sky in its glass over blinds, curtains, an
open room, a plainly reflective pane or a lit room, with its frame) and each shop bay one of the shop atlas (lit retail,
a dim cafe, a rolled-down shutter, papered-over glass, a reflective lobby), picked per window and per 6 m of frontage.
It is all painted on the facade's plane: a cheap interior card with no parallax.

The wall (a Principled surface in the masonry's colour) and each kind of glass are separate surfaces mixed by where the
glass is (a mix shader evaluates a branch only where its factor needs it), so a pixel of wall never pays for an interior
nor a pixel of glass for the brick. The glass is a diffuse surface in the atlas's colour, with the sky's reflection
painted into it, plus an emission shader: a Principled glass with a real specular lobe made a frame a quarter slower.

A lit room gives off light, but the material's emission sampling is off. With it on, Cycles treats every triangle of a
material that can emit anywhere as a light and samples it from every point of the frame: a whole city of facades as
lights made a frame twice as slow (0.6 s to 1.1 s at 64 samples), for a glow of a few windows. Sampled or not, a lit
window seen by the camera is as bright; only the light it throws on the street is gone.
"""

from __future__ import annotations

import bpy
import window_atlas
from materials_street import lin
from shader_nodes import G, new_material, principled

BAY, FLOOR_H, GROUND_H = 3.0, 3.6, 4.5
LIT_EMISSION = 0.4  # how much of a lit room's colour it gives off as light (on top of the daylight it takes)
FACADES = {  # style -> sRGB of the wall (brick: two tones) and of the pilasters; a curtain wall: the glass
    "brick_red": {"brick": ((0.50, 0.19, 0.12), (0.42, 0.15, 0.10)), "pilaster": (0.70, 0.66, 0.58)},
    "brick_tan": {"brick": ((0.62, 0.48, 0.32), (0.54, 0.41, 0.27)), "pilaster": (0.80, 0.74, 0.62)},
    "stone_grey": {"wall": (0.50, 0.51, 0.53), "pilaster": (0.64, 0.65, 0.67)},
    "glass_blue": {"curtain": (0.16, 0.30, 0.38)},
    "concrete_white": {"wall": (0.78, 0.78, 0.76), "pilaster": (0.88, 0.88, 0.86)},
    "stucco_cream": {"wall": (0.88, 0.82, 0.65), "pilaster": (0.95, 0.92, 0.82)},
}


def material(style: str) -> bpy.types.Material:
    """The material of one of `FACADES`; a style the file does not know is an error."""
    if style not in FACADES:
        raise KeyError(f"unknown facade style {style!r}; known: {sorted(FACADES)}")
    spec = FACADES[style]
    m = new_material(f"facade_{style}")
    m.cycles.emission_sampling = "NONE"  # a lit room glows when seen, but no building is a light to sample (see below)
    g, wall_bsdf = G(m.node_tree), principled(m)
    out = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    links = m.node_tree.links
    tc = g.node("ShaderNodeTexCoord")
    x, y, z = g.xyz(tc.outputs["Object"])
    nx, _, _ = g.xyz(tc.outputs["Normal"])
    u = g.mix(g.math("GREATER_THAN", g.math("ABSOLUTE", nx), 0.5), x, y, "FLOAT")  # the coordinate along this face
    rnd = g.node("ShaderNodeObjectInfo").outputs["Random"]
    upper = g.math("GREATER_THAN", z, GROUND_H)
    lower = g.math("SUBTRACT", 1.0, upper)
    fu = g.math("DIVIDE", u, BAY)
    iu, au = g.math("FLOOR", fu), g.math("MULTIPLY", g.math("FRACT", fu), BAY)
    fv = g.math("DIVIDE", g.math("SUBTRACT", z, GROUND_H), FLOOR_H)
    iv, av = g.math("FLOOR", fv), g.math("MULTIPLY", g.math("FRACT", fv), FLOOR_H)
    cx, cy = g.math("ADD", iu, g.math("MULTIPLY", rnd, 17.0)), g.math("ADD", iv, g.math("MULTIPLY", rnd, 9.0))
    rk = _randoms(g, cx, cy, rnd)
    if "curtain" in spec:
        wall, layers = _curtain(g, spec["curtain"], u, z, iv, av, lower, rk)
        wall_bsdf.inputs["Roughness"].default_value = 0.4
    else:
        wall, layers = _punched(g, spec, u, z, au, av, upper, lower, rk, rnd)
        wall_bsdf.inputs["Roughness"].default_value = 0.85
    # A surface per kind of glass, mixed over the wall by where that glass is. A mix shader's branches are evaluated
    # only where its factor needs them, so the heavy graph of an inside is skipped on every pixel of wall (most of a
    # facade) and the wall's brick on every pixel of glass.
    links.new(wall, wall_bsdf.inputs["Base Color"])
    shader = wall_bsdf.outputs["BSDF"]
    for colour, emit, mask in layers:
        diffuse, glow, add = (g.node(k) for k in ("ShaderNodeBsdfDiffuse", "ShaderNodeEmission", "ShaderNodeAddShader"))
        links.new(colour, diffuse.inputs["Color"])
        links.new(colour, glow.inputs["Color"])
        links.new(g.math("MULTIPLY", emit, LIT_EMISSION), glow.inputs["Strength"])
        links.new(diffuse.outputs["BSDF"], add.inputs[0])
        links.new(glow.outputs["Emission"], add.inputs[1])
        mix = g.node("ShaderNodeMixShader")
        links.new(mask, mix.inputs["Fac"])
        links.new(shader, mix.inputs[1])
        links.new(add.outputs["Shader"], mix.inputs[2])
        shader = mix.outputs["Shader"]
    links.new(shader, out.inputs["Surface"])
    return m


def _randoms(g, cx, cy, rnd):
    """`rk(k)`: the k-th random number (0..1) of this window, the same all over it and another for the next window. A
    white noise node gives three at once (its colour's channels), so nine numbers cost three lookups."""
    cache: dict[int, tuple] = {}

    def rk(k: int):
        group = k // 3
        if group not in cache:
            vec = g.combine(g.math("ADD", cx, 3.1 * group), g.math("ADD", cy, -2.3 * group),
                            g.math("ADD", rnd, 0.37 * group + 0.11))
            cache[group] = g.xyz(g.node("ShaderNodeTexWhiteNoise", {"Vector": vec}, noise_dimensions="3D").outputs[
                "Color"])
        return cache[group][k % 3]

    return rk


def _storefront_accent(g, rnd):
    return g.ramp(rnd, [(0.0, lin((0.55, 0.08, 0.08))), (0.2, lin((0.08, 0.30, 0.15))), (0.4, lin((0.08, 0.13, 0.38))),
                        (0.6, lin((0.10, 0.10, 0.11))), (0.8, lin((0.78, 0.72, 0.55)))], "CONSTANT")


def _lookup(g, image, shape: tuple[int, int], variant, wu, wv):
    """The pixel of variant `variant` (0..1, picking one of the atlas's `shape` = (columns, rows) cells) at (`wu`, `wv`)
    across and up it: (colour, emission)."""
    cols, rows = shape
    idx = g.math("MINIMUM", g.math("FLOOR", g.math("MULTIPLY", variant, cols * rows)), cols * rows - 1)
    col, row = g.math("FLOORED_MODULO", idx, cols), g.math("FLOOR", g.math("DIVIDE", idx, cols))
    wu, wv = (g.math("MINIMUM", g.math("MAXIMUM", v, 0.002), 0.998) for v in (wu, wv))  # stay inside the cell
    uv = g.combine(g.math("DIVIDE", g.math("ADD", col, wu), cols), g.math("DIVIDE", g.math("ADD", row, wv), rows), 0.0)
    tex = g.node("ShaderNodeTexImage", {"Vector": uv}, image=image, interpolation="Linear", extension="EXTEND")
    return tex.outputs["Color"], tex.outputs["Alpha"]


def _pane(g, wu, wv, rk):
    """One window pane (`wu`, `wv`: 0..1 across and up it): returns (colour, emission mask)."""
    return _lookup(g, window_atlas.window_atlas(), window_atlas.WINDOWS, rk(0), wu, wv)


def _shop(g, wu, wv, kind):
    """A shop's glazing (`kind`: 0..1, the same for 6 m of frontage): returns (colour, emission mask)."""
    return _lookup(g, window_atlas.shop_atlas(), window_atlas.SHOPS, kind, wu, wv)


def _punched(g, spec, u, z, au, av, upper, lower, rk, rnd):
    """A masonry or stucco wall with punched windows and a glazed storefront; returns the wall's colour and the glass
    layers, (colour, emission mask, where) each: the windows' panes and the storefront's glazing."""
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    if "brick" in spec:
        c1, c2 = spec["brick"]
        wall = g.node("ShaderNodeTexBrick", {"Vector": g.combine(u, z, 0.0), "Color1": (*lin(c1), 1.0),
                                             "Color2": (*lin(c2), 1.0), "Mortar": (0.30, 0.28, 0.25, 1.0),
                                             "Scale": 1.0, "Mortar Size": 0.012, "Brick Width": 0.215,
                                             "Row Height": 0.075, "Bias": 0.0}).outputs["Color"]
    else:
        wall = lin(spec["wall"])
    stain = g.math("MULTIPLY_ADD", g.noise(obj, 0.35, 3.0), 0.35, 0.82)
    wall = g.vmath("SCALE", wall, scale=stain)
    pil = g.math("MAXIMUM", g.math("LESS_THAN", au, 0.35), g.math("GREATER_THAN", au, 2.65))
    col = g.mix(pil, wall, g.vmath("SCALE", lin(spec["pilaster"]), scale=g.math("MULTIPLY_ADD", stain, 0.5, 0.5)))
    sill = g.math("MULTIPLY", g.between(av, 0.68, 0.85), g.between(au, 0.55, 2.45))
    col = g.mix(g.math("MULTIPLY", sill, upper), col, lin((0.80, 0.78, 0.72)))
    win = g.math("MULTIPLY", g.math("MULTIPLY", g.between(au, 0.65, 2.35), g.between(av, 0.85, 2.75)), upper)
    pane, pane_lit = _pane(g, g.math("DIVIDE", g.math("SUBTRACT", au, 0.65), 1.7),
                           g.math("DIVIDE", g.math("SUBTRACT", av, 0.85), 1.9), rk)
    # storefront: kick plate, glazing between piers, a sign band in a random shop colour
    shop = g.math("MULTIPLY", g.math("MULTIPLY", g.between(z, 0.55, 3.35), g.between(au, 0.30, 2.70)), lower)
    frontage = g.white(g.combine(g.math("FLOOR", g.math("DIVIDE", u, 6.0)), rnd, 1.0))
    window, window_lit = _shop(g, g.math("DIVIDE", g.math("SUBTRACT", au, 0.30), 2.4),
                               g.math("DIVIDE", g.math("SUBTRACT", z, 0.55), 2.8), frontage)
    col = g.mix(g.math("MULTIPLY", g.between(z, 3.50, 4.35), lower), col, _storefront_accent(g, rnd))
    col = g.mix(g.math("MULTIPLY", g.math("LESS_THAN", z, 0.55), lower), col, lin((0.20, 0.20, 0.21)))
    return col, [(pane, pane_lit, win), (window, window_lit, shop)]


def _curtain(g, rgb, u, z, iv, av, lower, rk):
    """A glass tower: panes 1.5 m wide between thin mullions, a slate spandrel band at every floor, a glazed lobby. Each
    pane is tinted glass over a room (`_pane`, two thirds room). Returns the wall's colour (spandrel, mullions, the
    foot) and the one glass layer."""
    pane_w = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.grid(u, 1.5, 0.07)),
                    g.math("SUBTRACT", 1.0, g.grid(z, 3.6, 0.07)))
    spandrel = g.math("MULTIPLY", g.math("LESS_THAN", av, 0.9), g.math("SUBTRACT", 1.0, lower))
    glass_col = g.vmath("SCALE", lin(rgb), scale=g.math("MULTIPLY_ADD", g.white(g.combine(g.math(
        "FLOOR", g.math("DIVIDE", u, 1.5)), iv, 0.5)), 0.5, 0.75))
    inner, inner_lit = _pane(g, g.math("FRACT", g.math("DIVIDE", u, 1.5)),
                             g.math("DIVIDE", g.math("SUBTRACT", av, 0.9), 2.7), rk)
    glass_col = g.mix(0.65, glass_col, inner)
    glass_col = g.mix(0.4, glass_col, (0.12, 0.16, 0.22))  # the sky in the glass: a tower reflects more
    col = g.mix(spandrel, lin((0.35, 0.37, 0.40)), lin((0.10, 0.12, 0.14)))
    col = g.mix(g.math("LESS_THAN", z, 0.5), col, lin((0.15, 0.15, 0.16)))
    glassy = g.math("MULTIPLY", g.math("MULTIPLY", pane_w, g.math("SUBTRACT", 1.0, spandrel)),
                    g.math("GREATER_THAN", z, 0.5))
    return col, [(glass_col, inner_lit, glassy)]
