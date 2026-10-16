# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's ground materials: worn asphalt, paving and sidewalk slabs, kerb stone, road paint, manhole iron.

Runs inside Blender's own Python and is never imported by `stbench`. No image file and no UVs (an ambientCG road or
paving texture listed in the manifest may replace these once the asset library holds one): the asphalt reads two
tileable images made in memory (`ground_images`), the rest is noise, cells and math on object coordinates, as the
warehouse's concrete is. A street is not clean. The asphalt carries what a real carriageway does at 12 m: patches of
fresh or aged repair with a sealed edge, trench scars, a faint map of cracks, oil stains with a sheen and a grain. The
concrete carries slab joints, hairline cracks, grime near its edges and gum spots. Paint is thermoplastic worn through
in patches. Nothing here is a tracked colour: the surfaces are greys and beiges, as before.
"""

from __future__ import annotations

import bpy
import ground_images
from shader_nodes import G, new_material, principled


def _srgb(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(rgb, f: float = 1.0) -> tuple[float, float, float]:
    """sRGB to linear light, scaled by `f` first (clipped at 1)."""
    return tuple(_srgb(min(1.0, c * f)) for c in rgb)


def plain(name: str, linear_rgb, rough: float = 0.8) -> bpy.types.Material:
    """A constant colour given in linear light (a facade's trim, the roofs, a kerb)."""
    m = new_material(name)
    principled(m).inputs["Base Color"].default_value = (*linear_rgb, 1.0)
    principled(m).inputs["Roughness"].default_value = rough
    return m


def asphalt() -> bpy.types.Material:
    """Worn asphalt: two tileable images on object coordinates (`ground_images`: the tile has the mottling, patches of
    repair with a sealed edge, trench scars, cracks and oil stains, and its alpha the roughness; the macro tone breaks
    its repeat) and one noise for the grain close up, which also gives the bump."""
    m = new_material("shader:asphalt")
    g, p = G(m.node_tree), principled(m)
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    x, y, _ = g.xyz(obj)

    def lookup(image, metres: float):
        vec = g.combine(g.math("DIVIDE", x, metres), g.math("DIVIDE", y, metres), 0.0)
        return g.node("ShaderNodeTexImage", {"Vector": vec}, image=image, interpolation="Linear", extension="REPEAT")

    tile = lookup(ground_images.asphalt_tile(), ground_images.TILE_M)
    macro = lookup(ground_images.asphalt_macro(), ground_images.MACRO_M)
    grain = g.noise(obj, 45.0, 3.0, 0.7)
    col = g.vmath("SCALE", g.vmath("MULTIPLY", tile.outputs["Color"], macro.outputs["Color"]),
                  scale=g.math("MULTIPLY_ADD", grain, 0.5, 0.75))
    m.node_tree.links.new(col, p.inputs["Base Color"])
    m.node_tree.links.new(tile.outputs["Alpha"], p.inputs["Roughness"])
    bump = g.node("ShaderNodeBump", {"Height": grain, "Strength": 0.25, "Distance": 0.004})
    m.node_tree.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


def _slabs(name: str, pitch: float, joint: float, base, stain_scale: float, edges: bool = False,
           bond: bool = False, world: bool = False) -> bpy.types.Material:
    """Concrete slabs of `pitch` metres, each a little lighter or darker, with joints `joint` wide, stains, grain,
    hairline cracks in some slabs and gum spots. `edges` grimes the slab near its two long edges (a sidewalk's local x
    runs from the building to the kerb, +-2 m); `bond` offsets every other row by half a slab (paving flags); `world`
    reads the position in the world, not in the object, so pieces laid edge to edge show one pattern across the seam."""
    m = new_material(name)
    g, p = G(m.node_tree), principled(m)
    obj = (g.node("ShaderNodeNewGeometry").outputs["Position"] if world
           else g.node("ShaderNodeTexCoord").outputs["Object"])
    x, y, _ = g.xyz(obj)
    across = x  # the sidewalk's own x, before a bond moves it
    if bond:
        row = g.math("FLOORED_MODULO", g.math("FLOOR", g.math("DIVIDE", y, pitch)), 2.0)
        x = g.math("ADD", x, g.math("MULTIPLY", row, pitch / 2))
    cell = g.combine(g.math("FLOOR", g.math("DIVIDE", x, pitch)), g.math("FLOOR", g.math("DIVIDE", y, pitch)), 3.0)
    tone = g.math("MULTIPLY_ADD", g.white(cell), 0.28, 0.86)
    stain = g.math("MULTIPLY_ADD", g.noise(obj, stain_scale, 3.0), 0.3, 0.85)
    grain = g.math("MULTIPLY_ADD", g.noise(obj, 60.0, 4.0), 0.25, 0.9)
    shade = g.math("MULTIPLY", g.math("MULTIPLY", tone, stain), grain)
    # grime: dirt along the edges, brown leaf stain patches, dark gum spots, a faint polish where people walk
    if edges:
        edge = g.math("MULTIPLY_ADD", g.math("ABSOLUTE", across), 1.0 / 0.8, -1.2 / 0.8, clamp=True)
        shade = g.math("MULTIPLY", shade, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", edge, 0.22)))
    leaf = g.math("MULTIPLY_ADD", g.noise(obj, 0.45, 3.0), 4.0, -2.2, clamp=True)
    gum_cells = g.node("ShaderNodeTexVoronoi", {"Vector": obj, "Scale": 11.0, "Randomness": 1.0},
                       voronoi_dimensions="3D", feature="F1").outputs["Distance"]
    gum = g.math("MULTIPLY", g.math("LESS_THAN", gum_cells, 0.045), g.math("GREATER_THAN", g.white(g.combine(
        g.math("FLOOR", g.math("MULTIPLY", x, 5.0)), g.math("FLOOR", g.math("MULTIPLY", y, 5.0)), 1.0)), 0.55))
    shade = g.math("MULTIPLY", shade, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", gum, 0.35)))
    ridge = g.math("ABSOLUTE", g.math("SUBTRACT", g.noise(obj, 0.8, 4.0, 0.6), 0.62))
    hair = g.math("MULTIPLY", g.math("LESS_THAN", ridge, 0.004), g.math("GREATER_THAN", g.white(cell), 0.9))
    joints = g.math("MAXIMUM", g.grid(x, pitch, joint), g.grid(y, pitch, joint))
    col = g.vmath("SCALE", base, scale=shade)
    col = g.mix(leaf, col, g.vmath("MULTIPLY", col, (0.88, 0.8, 0.66)))
    col = g.mix(hair, col, (0.05, 0.05, 0.05))
    col = g.mix(joints, col, (0.06, 0.06, 0.06))
    m.node_tree.links.new(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.85
    return m


def sidewalk(edges: bool = False) -> bpy.types.Material:
    """Light concrete in 1.5 m slabs with 2 cm joints. `edges`: for a sidewalk's own pieces (local x across it)."""
    return _slabs("sidewalk", 1.5, 0.02, (0.34, 0.335, 0.32), 0.5, edges=edges)


def paving() -> bpy.types.Material:
    """The square's paving: warm beige flags 0.6 m square in running bond with 1.2 cm joints, in world coordinates (the
    square is two pieces with a 1 cm seam between them, and the floor shows through the yard beside them)."""
    return _slabs("shader:paving", 0.6, 0.012, (0.40, 0.34, 0.25), 0.7, bond=True, world=True)


def kerb() -> bpy.types.Material:
    """The granite edge of a sidewalk: mottled, dirtier near the road (the lower the grimier) with scuffs."""
    m = new_material("kerb")
    g, p = G(m.node_tree), principled(m)
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    _, _, z = g.xyz(obj)
    mottle = g.math("MULTIPLY_ADD", g.noise(obj, 3.0, 3.0), 0.3, 0.85)
    speck = g.math("MULTIPLY_ADD", g.noise(obj, 80.0, 2.0), 0.3, 0.85)
    low = g.math("MULTIPLY_ADD", z, 1.0 / 0.08, -0.15 / 0.08, clamp=True)  # 0 near the road, 1 over 0.08 m
    grime = g.math("MULTIPLY_ADD", low, 0.4, 0.6)
    v = g.math("MULTIPLY", g.math("MULTIPLY", mottle, speck), grime)
    m.node_tree.links.new(g.vmath("SCALE", lin((0.45, 0.45, 0.43)), scale=v), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.8
    return m


def road_paint(kind: str) -> bpy.types.Material:
    """White or yellow thermoplastic: worn in patches (the wear darkens), gritty where the asphalt shows through, a
    little dirty along its length. It never makes a hole."""
    m = new_material(f"paint_{kind}")
    g, p = G(m.node_tree), principled(m)
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    wear = g.math("MULTIPLY_ADD", g.noise(obj, 2.5, 3.0), 0.45, 0.6)
    grit = g.math("MULTIPLY_ADD", g.noise(obj, 70.0, 2.0), 0.5, 0.75)
    bare = g.math("MULTIPLY", g.math("LESS_THAN", g.noise(obj, 9.0, 3.0), 0.36), g.math("LESS_THAN", g.white(
        g.combine(g.math("FLOOR", g.math("MULTIPLY", g.xyz(obj)[0], 40.0)), g.math("FLOOR", g.math(
            "MULTIPLY", g.xyz(obj)[1], 40.0)), 2.0)), 0.5))
    base = (0.56, 0.56, 0.54) if kind == "white" else (0.66, 0.44, 0.04)
    col = g.vmath("SCALE", g.vmath("SCALE", base, scale=wear), scale=grit)
    col = g.mix(bare, col, (0.09, 0.09, 0.09))
    m.node_tree.links.new(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.6
    return m


def cast_iron(name: str, rgb=(0.07, 0.07, 0.075), rough: float = 0.55) -> bpy.types.Material:
    """Manhole and drain iron: near black with a rusty brown mottling, a little sheen where it is polished."""
    m = new_material(name)
    g, p = G(m.node_tree), principled(m)
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    rust = g.math("MULTIPLY_ADD", g.noise(obj, 9.0, 3.0), 1.0, -0.3, clamp=True)
    m.node_tree.links.new(g.mix(rust, lin(rgb), lin((0.22, 0.10, 0.05))), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = 0.6
    return m


def foliage() -> bpy.types.Material:
    """Leaves: a dull dark green that is mottled with a lighter yellow-green, each object (a tree, a shrub) a little
    different from the next. Never a saturated palette green: a street tree is not a tracked thing."""
    m = new_material("foliage")
    g, p = G(m.node_tree), principled(m)
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    r = g.node("ShaderNodeObjectInfo").outputs["Random"]
    mottle = g.math("ADD", g.math("MULTIPLY", g.noise(obj, 3.2, 3.0), 0.7), g.math("MULTIPLY", r, 0.3))
    col = g.ramp(mottle, [(0.25, (0.020, 0.050, 0.016)), (0.55, (0.042, 0.092, 0.026)), (0.85, (0.085, 0.140, 0.036))])
    m.node_tree.links.new(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.6
    return m


def bark() -> bpy.types.Material:
    """Bark: grey-brown, ridged along the trunk."""
    m = new_material("bark")
    g, p = G(m.node_tree), principled(m)
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    x, y, z = g.xyz(obj)
    ridge = g.noise(g.combine(g.math("MULTIPLY", x, 14.0), g.math("MULTIPLY", y, 14.0),
                              g.math("MULTIPLY", z, 1.5)), 6.0, 3.0)
    m.node_tree.links.new(g.ramp(ridge, [(0.3, (0.060, 0.045, 0.035)), (0.7, (0.150, 0.120, 0.095))]),
                          p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.9
    return m
