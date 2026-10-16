# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What a floor looks like from 8 cm above it (M1's R.2): relief where a bump
reads flat, and a pile.

Runs inside Blender's own Python and is never imported by `stbench`. The robot vacuum sees its floor at 5 to 27 degrees:
the rows at the bottom of a frame resolve 0.5 to 1.2 mm a pixel, and a bump map of a grout line or a plank gap shows no
depth at such an angle, so the tile and wood floors take true displacement from the displacement map of their 2K sets
(`displace`: at most `MAX_DISPLACE_M` either side of the surface, on adaptively subdivided geometry, `subdivide`: the
mesh is diced to about a pixel near the camera and not at all far away), with the bump kept for what is finer than the
dicing. Carpet and rugs take a pile (`pile`): a sheen that rises at grazing angles, as fibres scatter, and a bump of
fibre tips and tufts. No effect is a shape a perception model could take for an object: every one is a millimetre.
"""

from __future__ import annotations

import bpy

PROP = "stbench_displace_m"  # custom property of a material that has displacement: its relief, peak to peak
MAX_DISPLACE_M = 0.003  # no point of a floor moves more than this
# floor asset -> relief peak to peak, metres: the grout of the tiles, the gaps of the planks
RELIEF_M = {"acg_Tiles139": 0.005, "acg_Tiles107": 0.005, "acg_WoodFloor051": 0.004}
PIXEL_SIZE = 1.0  # the dicing of a displaced floor, in pixels


def displace(mat: bpy.types.Material, relief_m: float) -> bool:
    """True displacement for an ambientCG material of a level floor, from the same map its bump reads: `relief_m`
    peak to peak about the surface (the map's mid grey stays put), bump kept for the detail finer than the dicing.
    Its maps are projected flat on x and y, as the box projection the library gives them would for a floor: a
    displacement is evaluated before the geometry has a face, and the box projection then reads every map from the
    wrong side (one plateau, no groove). False where the material has no displacement map. A material is given it
    once."""
    if PROP in mat:
        return True
    if relief_m / 2 > MAX_DISPLACE_M + 1e-9:
        raise ValueError(f"{mat.name}: a relief of {relief_m} m moves the floor by more than {MAX_DISPLACE_M} m")
    nt = mat.node_tree
    bump = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBump" and n.inputs["Height"].links), None)
    if bump is None:
        return False
    for tex in (n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage"):
        tex.projection = "FLAT"  # see the docstring
    node = nt.nodes.new("ShaderNodeDisplacement")
    node.inputs["Midlevel"].default_value = 0.5
    node.inputs["Scale"].default_value = relief_m
    nt.links.new(bump.inputs["Height"].links[0].from_socket, node.inputs["Height"])
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output)
    nt.links.new(node.outputs["Displacement"], out.inputs["Displacement"])
    mat.displacement_method = "BOTH"
    mat.max_vertex_displacement = relief_m / 2
    mat[PROP] = relief_m
    return True


def is_displaced(mat: bpy.types.Material | None) -> bool:
    return mat is not None and PROP in mat


def subdivide(ob: bpy.types.Object) -> None:
    """Adaptive subdivision for an object that carries a displaced material: it is diced `PIXEL_SIZE` pixels near the
    camera, per frame."""
    md = ob.modifiers.new("floor_relief", "SUBSURF")
    md.subdivision_type = "SIMPLE"
    md.use_adaptive_subdivision = True
    md.adaptive_space = "PIXEL"
    md.adaptive_pixel_size = PIXEL_SIZE
    md.levels = md.render_levels = 1


def pile(g, p, obj) -> None:
    """A carpet's pile on the Principled node `p` of a worn material (`wear.worn`): sheen that rises as the view grazes
    (fibres scatter light sideways), and a bump of fibre tips (1.4 mm) and tufts (1 cm) over the bump the material has
    already. `g` is the material's `shader_nodes.G`, `obj` its object-coordinates vector."""
    fine = g.noise(obj, 700.0, 2.0, 0.65)
    tufts = g.noise(g.shifted(obj, 3.0, 5.0, 1.0), 90.0, 3.0, 0.6)
    height = g.math("ADD", g.math("MULTIPLY", fine, 0.6), g.math("MULTIPLY", tufts, 0.4))
    below = p.inputs["Normal"].links[0].from_socket if p.inputs["Normal"].links else None
    inputs = {"Height": height, "Strength": 0.8, "Distance": 0.0015}
    if below is not None:
        inputs["Normal"] = below
    g.put(p, "Normal", g.node("ShaderNodeBump", inputs).outputs["Normal"])
    g.put(p, "Sheen Weight", 0.8)
    g.put(p, "Sheen Roughness", 0.45)
