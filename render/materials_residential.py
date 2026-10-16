# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The residential look's materials: lawn, hedge, roof shingles, the three wall finishes, fence timber, drive concrete
and bark mulch. Procedural, as the downtown's street surfaces are: no image file and no UVs, only noise, cells and math
on object coordinates (`shader_nodes`), so nothing is downloaded and every appearance draws them alike.

Runs inside Blender's own Python and is never imported by `stbench`. Nothing is cached here (a scene rebuild frees every
material): `Surfaces` makes each material of a scene once. The colours are dull and natural, never a palette colour at
full strength: a lawn or a hedge is not a tracked green, and a house wall is its attribute colour muted towards a pale
grey, so that no blue or red house reads as a car.

A part names a material by a key: `lawn`, `hedge`, `fence`, `driveway`, `mulch`, or `<finish>:<colour>` with a finish of
`siding`, `render` or `brick` and the house's colour attribute, or `shingle:<roof>` with a roof of `ROOFS`. Any other
key is the city's (`furniture_city.Surfaces`) or the palette's. Object coordinates are the piece's own (origin at its
base centre, x its front): boards, courses and bricks are laid on them, so a wall facing +x or +y has them straight.
"""

from __future__ import annotations

import bpy
from materials_street import lin
from shader_nodes import G, new_material, principled
from shapes import PALETTE

# Roof shingles, linear: charcoal asphalt, brown asphalt, slate grey and clay tile (a brick red, kept dull)
ROOFS = {"charcoal": (0.075, 0.075, 0.08), "brown": (0.14, 0.095, 0.07), "slate": (0.09, 0.10, 0.115),
         "terracotta": (0.30, 0.115, 0.07)}
# Face brick, sRGB: by the house's colour attribute (a brown or orange house is brick of that tone)
BRICKS = {"red": (0.55, 0.24, 0.17), "brown": (0.42, 0.27, 0.19), "orange": (0.62, 0.34, 0.19)}
BOARD_M, COURSE_M, BRICK_M = 0.16, 0.075, 0.215  # a siding board, a brick course and a brick's length (with mortar)
SLAB_M = 3.0  # the drive's slab joints


def wall_rgb(colour: str) -> tuple[float, float, float]:
    """A house colour as the paint on a wall: the palette colour a third of the way to a pale grey, sRGB."""
    rgb = PALETTE.get(colour, PALETTE["beige"])[0]
    return tuple(0.55 * c + 0.45 * 0.66 for c in rgb)


def _surface(name: str, mean):
    """A new material, its node-graph builder and its Principled node, whose base colour is first set to `mean`
    (linear): the graph replaces it, but the library reads a material's constant colour as its mean luminance
    (`materials.MEAN_PROP`)."""
    m = new_material(name)
    p = principled(m)
    p.inputs["Base Color"].default_value = (*mean, 1.0)
    return m, G(m.node_tree), p


def _bump(m, g, p, height, strength: float, distance: float) -> None:
    m.node_tree.links.new(g.bump(height, strength, distance), p.inputs["Normal"])


def lawn() -> bpy.types.Material:
    """Mown grass: patches of drier and greener turf a few metres wide, a centimetre grain and mowing stripes 1.2 m
    wide that darken and lighten it by a twentieth."""
    m, g, p = _surface("lawn", (0.064, 0.12, 0.024))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    x = g.xyz(obj)[0]
    patch = g.noise(obj, 0.35, 4.0)
    grain = g.noise(obj, 42.0, 3.0, 0.7)
    stripe = g.math("MULTIPLY_ADD", g.triangle(x, 2.4), 0.10, 0.95)
    col = g.ramp(patch, [(0.25, (0.040, 0.092, 0.016)), (0.55, (0.062, 0.120, 0.022)), (0.85, (0.105, 0.155, 0.032))])
    col = g.shade(col, g.math("MULTIPLY", g.math("MULTIPLY_ADD", grain, 0.5, 0.75), stripe))
    m.node_tree.links.new(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.92
    _bump(m, g, p, grain, 0.3, 0.012)
    return m


def hedge() -> bpy.types.Material:
    """A clipped hedge: dense dark leaf, mottled at a hand's width, each hedge a little different from the next."""
    m, g, p = _surface("hedge", (0.034, 0.08, 0.021))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    r = g.node("ShaderNodeObjectInfo").outputs["Random"]
    mottle = g.math("ADD", g.math("MULTIPLY", g.noise(obj, 9.0, 4.0), 0.75), g.math("MULTIPLY", r, 0.25))
    col = g.ramp(mottle, [(0.2, (0.016, 0.040, 0.012)), (0.6, (0.034, 0.080, 0.021)), (0.9, (0.066, 0.125, 0.030))])
    m.node_tree.links.new(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.7
    _bump(m, g, p, g.noise(obj, 30.0, 3.0), 0.5, 0.03)
    return m


def shingle(roof: str) -> bpy.types.Material:
    """Roof shingles in `ROOFS[roof]`: courses 0.2 m apart (along x, up the slope) with a dark line under each, tabs
    0.33 m wide each a little lighter or darker, a grain and streaks of weathering."""
    m, g, p = _surface(f"shingle_{roof}", ROOFS[roof])
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    x, y, _ = g.xyz(obj)
    tab = g.white(g.combine(g.math("FLOOR", g.math("DIVIDE", x, 0.2)), g.math("FLOOR", g.math("DIVIDE", y, 0.33)), 1.0))
    tone = g.math("MULTIPLY", g.math("MULTIPLY_ADD", tab, 0.26, 0.87),
                  g.math("MULTIPLY_ADD", g.noise(obj, 1.1, 3.0), 0.3, 0.85))
    course = g.grid(x, 0.2, 0.014)
    shade = g.math("MULTIPLY", tone, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", course, 0.45)))
    m.node_tree.links.new(g.vmath("SCALE", ROOFS[roof], scale=shade), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.9
    _bump(m, g, p, g.math("SUBTRACT", 1.0, course), 0.4, 0.006)
    return m


def siding(colour: str) -> bpy.types.Material:
    """Horizontal boards `BOARD_M` high with a shadow line under each, in the house's painted colour."""
    m, g, p = _surface(f"siding_{colour}", lin(wall_rgb(colour)))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    z = g.xyz(obj)[2]
    groove = g.grid(z, BOARD_M, 0.012)
    paint = g.math("MULTIPLY_ADD", g.noise(obj, 22.0, 3.0), 0.12, 0.94)
    shade = g.math("MULTIPLY", paint, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", groove, 0.4)))
    m.node_tree.links.new(g.vmath("SCALE", lin(wall_rgb(colour)), scale=shade), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.65
    _bump(m, g, p, g.math("SUBTRACT", 1.0, groove), 0.5, 0.008)
    return m


def render_(colour: str) -> bpy.types.Material:
    """Rendered (stuccoed) wall: a fine grain, faint stains and a grimy foot."""
    m, g, p = _surface(f"render_{colour}", lin(wall_rgb(colour)))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    z = g.xyz(obj)[2]
    grain = g.math("MULTIPLY_ADD", g.noise(obj, 28.0, 3.0), 0.14, 0.93)
    stain = g.math("MULTIPLY_ADD", g.noise(obj, 0.7, 3.0), 0.16, 0.9)
    foot = g.math("MULTIPLY_ADD", g.math("MULTIPLY_ADD", z, -1.0 / 0.6, 1.0, clamp=True), -0.18, 1.0)
    shade = g.math("MULTIPLY", g.math("MULTIPLY", grain, stain), foot)
    m.node_tree.links.new(g.vmath("SCALE", lin(wall_rgb(colour)), scale=shade), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.88
    _bump(m, g, p, grain, 0.3, 0.004)
    return m


def brick(colour: str) -> bpy.types.Material:
    """Face brick in running bond: courses `COURSE_M` high, bricks `BRICK_M` long with every other course offset by half
    a brick, mortar lines between them and a tone to each brick. The length runs along x + y, which on a wall that faces
    +-x or +-y is the way along it."""
    m, g, p = _surface(f"brick_{colour}", lin(BRICKS.get(colour, BRICKS["red"])))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    x, y, z = g.xyz(obj)
    row = g.math("FLOOR", g.math("DIVIDE", z, COURSE_M))
    u = g.math("ADD", g.math("ADD", x, y), g.math("MULTIPLY", g.math("FLOORED_MODULO", row, 2.0), BRICK_M / 2))
    cell = g.combine(g.math("FLOOR", g.math("DIVIDE", u, BRICK_M)), row, 1.0)
    tone = g.math("MULTIPLY_ADD", g.white(cell), 0.3, 0.85)
    mortar = g.math("MAXIMUM", g.grid(z, COURSE_M, 0.011), g.grid(u, BRICK_M, 0.011))
    face = g.vmath("SCALE", lin(BRICKS.get(colour, BRICKS["red"])), scale=tone)
    m.node_tree.links.new(g.mix(mortar, face, lin((0.55, 0.53, 0.50))), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.85
    _bump(m, g, p, g.math("SUBTRACT", 1.0, mortar), 0.6, 0.006)
    return m


def fence() -> bpy.types.Material:
    """Close-board timber: boards 0.12 m wide across the fence's length (its y) with a dark seam between them, a grain
    along the height and a weathered grey-brown."""
    m, g, p = _surface("fence", (0.14, 0.10, 0.06))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    y = g.xyz(obj)[1]
    seam = g.grid(y, 0.12, 0.008)
    grain = g.noise(g.scaled(obj, 3.0, 12.0, 0.4), 6.0, 3.0)
    col = g.ramp(grain, [(0.3, (0.075, 0.05, 0.03)), (0.7, (0.19, 0.135, 0.085))])
    col = g.shade(col, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", seam, 0.55)))
    m.node_tree.links.new(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.85
    return m


def driveway() -> bpy.types.Material:
    """A concrete drive: slabs `SLAB_M` square with joints, each a little lighter or darker, stains, a grain and the
    dark patches where cars have dripped oil."""
    m, g, p = _surface("driveway", (0.30, 0.29, 0.27))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    x, y, _ = g.xyz(obj)
    cell = g.combine(g.math("FLOOR", g.math("DIVIDE", x, SLAB_M)), g.math("FLOOR", g.math("DIVIDE", y, SLAB_M)), 3.0)
    tone = g.math("MULTIPLY_ADD", g.white(cell), 0.24, 0.88)
    stain = g.math("MULTIPLY_ADD", g.noise(obj, 0.5, 3.0), 0.3, 0.85)
    grain = g.math("MULTIPLY_ADD", g.noise(obj, 55.0, 4.0), 0.22, 0.9)
    oil = g.math("MULTIPLY", g.math("GREATER_THAN", g.noise(obj, 1.6, 3.0), 0.62), 0.35)
    joints = g.math("MAXIMUM", g.grid(x, SLAB_M, 0.02), g.grid(y, SLAB_M, 0.02))
    shade = g.math("MULTIPLY", g.math("MULTIPLY", tone, stain), grain)
    shade = g.math("MULTIPLY", shade, g.math("SUBTRACT", 1.0, oil))
    col = g.mix(joints, g.vmath("SCALE", (0.36, 0.35, 0.33), scale=shade), (0.07, 0.07, 0.07))
    m.node_tree.links.new(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.85
    return m


def mulch() -> bpy.types.Material:
    """Bark mulch: dark brown chips, speckled at a centimetre and lighter in patches."""
    m, g, p = _surface("mulch", (0.09, 0.05, 0.027))
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    chip = g.white(g.node("ShaderNodeTexVoronoi", {"Vector": obj, "Scale": 55.0}, voronoi_dimensions="3D",
                          feature="F1").outputs["Position"])
    patch = g.noise(obj, 1.2, 3.0)
    col = g.ramp(patch, [(0.3, (0.060, 0.030, 0.016)), (0.7, (0.115, 0.062, 0.032))])
    m.node_tree.links.new(g.shade(col, g.math("MULTIPLY_ADD", chip, 0.45, 0.7)), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.96
    return m


FACTORIES = {"lawn": lambda arg: lawn(), "hedge": lambda arg: hedge(), "fence": lambda arg: fence(),
             "driveway": lambda arg: driveway(), "mulch": lambda arg: mulch(), "shingle": shingle, "siding": siding,
             "render": render_, "brick": brick}
NAMES = frozenset(FACTORIES)  # the key stems a part may carry


class Surfaces:
    """The residential pieces' materials, each made once per scene. `city` is the scene's `furniture_city.Surfaces` (the
    kerb, paving and road paint some pieces share), or None."""

    def __init__(self, city=None):
        self.city = city
        self.made: dict[str, bpy.types.Material] = {}

    def resolver(self, base=None):
        """`key -> Material | None`: these keys, then the city's, then `base`'s (the appearance's own, or none)."""
        under = self.city.resolver(base) if self.city is not None else base

        def resolve(key: str):
            name, _, arg = key.partition(":")
            if name in FACTORIES:
                if key not in self.made:
                    self.made[key] = FACTORIES[name](arg or None)
                return self.made[key]
            return under(key) if under else None

        return resolve
