# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The finishes of a home's walls, trim, doors, thresholds and undersides (M1's R.2 fix round, after the pre-screen).

Runs inside Blender's own Python and is never imported by `stbench`. `wear` applies them (`wear.FINISHES`); each takes
the colour and roughness of a material and the object-coordinates vector of the surface and returns the colour and the
roughness of the worn one, as the finishes of `wear` do.

What a robot vacuum sees of a wall from 6 to 30 cm is smooth painted plaster: a faint roller stipple and a patch of
sheen, not a texture of stains. So the plaster's own picture (a cracked, stippled one, which read as concrete) is
replaced by the wall colour, painted per room (`look_home.WALL_PAINT`, by the room's kind: some colour, as real homes
have it), with wear only where wear gathers: dust and grubbiness at the foot of the wall, bumper and shoe scuffs in the
lowest 40 cm, and marks at the switches and sockets, the door jambs and the corners of furniture (`use_scene` gives the
spots, from the map and the dressing). The trim is painted wood that has been kicked and mopped: grime in its lower
half and scuffs. A doorway's threshold is a strip of wood, not the flat underlay. What lies under a table or a sofa is
raw board and black non-woven dust cloth, with staples along its edge and a sag between slats, and not the furniture's
top finish.
"""

from __future__ import annotations

import math

import bpy
import look
from materials_city import lin
from shader_nodes import G

DIRT = lin((0.50, 0.44, 0.38))  # what dust and grime multiply a surface by
SCUFF = lin((0.55, 0.53, 0.50))  # the grey of a rubber or shoe mark
PROBE_M = 0.06  # a wall's room is the one 6 cm out along the face's normal
STAPLE = lin((0.55, 0.55, 0.57))
COVER_EDGE_M = 0.045  # staples run this far in from the edge of an entity's box (its dust cloth is inset 2 cm)
mask_fn = None  # set by `wear`: (g, key, obj) -> a 0 to 1 socket of that mask image, or None if not loaded
_rooms: list[tuple[str, tuple[float, float, float, float]]] = []  # (kind, (x0, y0, x1, y1))
_spots: list[tuple[float, float, float, float, float, float, float]] = []  # (x, y, z, rx, ry, rz, strength)
_footprints: list[tuple[float, float, float, float, float]] = []  # (cx, cy, hx, hy, yaw) of what stands on the floor


def use_scene(rooms, spots, footprints) -> None:
    """The scene's rooms (`(kind, (x0, y0, x1, y1))`), the spots on a wall that wear marks (`(x, y, z, rx, ry, rz,
    strength)`: an ellipsoid of those radii) and the footprints on the floor that dust gathers round (`(cx, cy, hx, hy,
    yaw)`); `wear.apply` calls it before it builds a finish."""
    _rooms[:] = list(rooms)
    _spots[:] = list(spots)
    _footprints[:] = list(footprints)


def _vary(g: G, obj):
    """The object coordinates moved by a random amount of the object, so that the pieces that share a mesh and a
    material are not alike."""
    r = g.node("ShaderNodeObjectInfo").outputs["Random"]
    return g.vmath("ADD", obj, g.combine(g.math("MULTIPLY", r, 41.0), g.math("MULTIPLY", r, 17.0),
                                         g.math("MULTIPLY", r, 7.0)))


def _paint(g: G):
    """The wall paint's colour multiplier at a point, by the room the face looks into: the point moved `PROBE_M` along
    its normal is inside that room's box."""
    geo = g.node("ShaderNodeNewGeometry")
    probe = g.vmath("ADD", geo.outputs["Position"], g.vmath("SCALE", geo.outputs["Normal"], scale=PROBE_M))
    px, py, _ = g.xyz(probe)
    paint = (1.0, 1.0, 1.0)
    for kind, (x0, y0, x1, y1) in _rooms:
        paint = g.mix(g.math("MULTIPLY", g.between(px, x0, x1), g.between(py, y0, y1)), paint,
                      look.get("home", "WALL_PAINT", kind))
    return paint


def _marks(g: G, obj):
    """0 to 1: the soft marks round the scene's spots, in patches (a smudge is never a clean ellipse)."""
    out = None
    for x, y, z, rx, ry, rz, strength in _spots:
        offset = g.vmath("MULTIPLY", g.vmath("SUBTRACT", obj, (x, y, z)), (1 / rx, 1 / ry, 1 / rz))
        length = g.node("ShaderNodeVectorMath", operation="LENGTH")
        g.nt.links.new(offset, length.inputs[0])
        mark = g.math("MULTIPLY", g.smooth(length.outputs["Value"], 1.0, 0.3), strength)
        out = mark if out is None else g.math("MAXIMUM", out, mark)
    if out is None:
        return 0.0
    patchy = g.smooth(g.noise(g.shifted(obj, 1.0, 4.0, 2.0), 18.0, 3.0, 0.6), 0.3, 0.65)
    return g.math("MULTIPLY", out, g.math("MULTIPLY_ADD", patchy, 0.7, 0.3))


def wall(g: G, base, rough, obj, ctx=None):
    """Smooth painted plaster: the room's paint, a stipple of the roller a few millimetres across (about 2% of the tone)
    and a broad patch of sheen; dust and grubbiness in the lowest 40 cm, scuffs in streaks across it, and the marks
    at switches, jambs and furniture."""
    z = g.xyz(obj)[2]
    roller = g.noise(g.scaled(obj, 1.0, 1.0, 0.55), 230.0, 2.0, 0.55)
    patch = g.noise(g.shifted(obj, 3.0, 7.0, 1.0), 0.45, 3.0)
    col = g.shade(g.tint(1.0, base, _paint(g)), g.math("MULTIPLY_ADD", roller, 0.035, 0.9825))
    col = g.shade(col, g.math("MULTIPLY_ADD", patch, 0.05, 0.975))
    haze = g.smooth(g.noise(g.shifted(obj, 9.0, 2.0, 0.0), 2.5, 3.0, 0.6), 0.35, 0.7)
    col = g.tint(g.math("MULTIPLY", g.smooth(z, 0.40, 0.06), g.math("MULTIPLY_ADD", haze, 0.18, 0.06)), col, DIRT)
    smear = g.smooth(g.noise(g.scaled(obj, 0.9, 0.9, 7.0), 9.0, 2.0, 0.6), 0.60, 0.74)
    if (mask := None if mask_fn is None else mask_fn(g, "imperfections", obj)) is not None:  # the mask's smudges
        smear = g.math("MAXIMUM", g.math("MULTIPLY", smear, 0.7), g.math("MULTIPLY", mask, 0.3))
    col = g.tint(g.math("MULTIPLY", g.math("MULTIPLY", smear, g.smooth(z, 0.40, 0.08)), 0.38), col, SCUFF)
    col = g.tint(g.math("MULTIPLY", _marks(g, obj), 0.5), col, DIRT)
    return col, g.math("MULTIPLY_ADD", patch, 0.08, 0.80)


def ceiling(g: G, base, rough, obj, ctx=None):
    """A ceiling's flat paint with a broad faint patch."""
    patch = g.noise(g.shifted(obj, 5.0, 1.0, 3.0), 0.4, 3.0)
    return g.shade(base, g.math("MULTIPLY_ADD", patch, 0.06, 0.97)), 0.85


def trim(g: G, base, rough, obj, ctx=None):
    """Painted wood that has been kicked and mopped: grime in its lower half (a baseboard's 8 cm, a casing's foot),
    scuffs across it, a patch of sheen. Each piece has its own pattern (`_vary`)."""
    z = g.xyz(obj)[2]
    pos = _vary(g, obj)
    dirt = g.smooth(g.noise(g.scaled(pos, 2.5, 2.5, 10.0), 7.0, 3.0, 0.6), 0.30, 0.75)
    smear = g.smooth(g.noise(g.scaled(pos, 1.2, 1.2, 14.0), 12.0, 2.0, 0.6), 0.58, 0.72)
    col = g.tint(g.math("MULTIPLY", g.smooth(z, 0.06, 0.0), g.math("MULTIPLY_ADD", dirt, 0.30, 0.10)), base, DIRT)
    col = g.tint(g.math("MULTIPLY", g.math("MULTIPLY", smear, g.smooth(z, 0.08, 0.01)), 0.28), col, SCUFF)
    return col, g.math("ADD", rough, g.math("MULTIPLY", dirt, 0.15), clamp=True)


def door(size: tuple[float, float] | None):
    """A painted door leaf (hinge at the origin, the free edge at x = its width): the flat paint with a very faint
    grain, scuffs and grubbiness at the foot (shoes, the vacuum's bumper) and the grease of a hand by the free edge."""
    width = size[0] if size else 0.85

    def finish(g: G, base, rough, obj, ctx=None):
        x, y, z = g.xyz(obj)
        patch = g.noise(g.shifted(obj, 2.0, 4.0, 6.0), 3.0, 3.0)
        col = g.shade(base, g.math("MULTIPLY_ADD", patch, 0.04, 0.98))
        scuff = g.smooth(g.noise(g.scaled(obj, 4.0, 4.0, 9.0), 10.0, 2.0, 0.6), 0.62, 0.78)
        col = g.tint(g.math("MULTIPLY", g.smooth(z, 0.30, 0.02), g.math("MULTIPLY_ADD", scuff, 0.16, 0.04)), col, DIRT)
        hand = g.math("MULTIPLY", g.between(x, width - 0.22, width - 0.02), g.smooth(g.math("ABSOLUTE", g.math(
            "SUBTRACT", z, 1.0)), 0.20, 0.05))
        col = g.tint(g.math("MULTIPLY", hand, g.math("MULTIPLY_ADD", patch, 0.25, 0.10)), col, DIRT)
        return col, g.math("MULTIPLY_ADD", patch, 0.10, 0.62)

    return finish


def threshold(g: G, base, rough, obj, ctx=None):
    """The wooden strip of a doorway between two floors (it is the plain underlay under the walls elsewhere): a worn
    medium wood, speckled grain, a lighter worn middle, dust in its corners."""
    x, y, _ = g.xyz(obj)
    grain = g.noise(g.scaled(obj, 14.0, 14.0, 1.0), 40.0, 3.0, 0.6)
    broad = g.noise(g.shifted(obj, 4.0, 1.0, 0.0), 6.0, 3.0, 0.55)
    wood = g.ramp(grain, [(0.35, (0.095, 0.065, 0.042)), (0.65, (0.17, 0.12, 0.08))])
    worn = g.shade(wood, g.math("MULTIPLY_ADD", broad, 0.4, 0.8))
    col = g.tint(g.smooth(broad, 0.55, 0.8), worn, lin((0.8, 0.76, 0.7)))
    return col, g.math("MULTIPLY_ADD", broad, 0.2, 0.45)


def raw(g: G, base, rough, obj, ctx=None):
    """Unfinished board seen from below (the underside of a table top or a shelf): fine speckle and streaks in the tone,
    a patch of grey dust, matte. No bump: from 2 cm away at a grazing angle a bump reads as a smear."""
    y = g.xyz(obj)[1]
    fine = g.noise(g.scaled(obj, 1.0, 5.0, 1.0), 90.0, 3.0, 0.6)
    dust = g.smooth(g.noise(g.shifted(obj, 2.0, 6.0, 1.0), 7.0, 3.0, 0.6), 0.45, 0.8)
    seam = g.math("MULTIPLY", g.grid(y, 0.09, 0.003), 0.45)  # the joints between boards, every 9 cm
    col = g.shade(g.shade(base, g.math("MULTIPLY_ADD", fine, 0.22, 0.89)), g.math("SUBTRACT", 1.0, seam))
    return g.tint(g.math("MULTIPLY", dust, 0.5), col, lin((0.6, 0.58, 0.54))), 0.9


def cover(size: tuple[float, float] | None):
    """A black non-woven dust cloth stretched under a sofa or a bed (a box of `size`, its depth and width; the cloth is
    inset 2 cm): fibres, a sag between the slats and staples in a row 2.5 cm in from the edge."""
    hx, hy = (size[0] / 2, size[1] / 2) if size else (1.0, 0.8)

    def finish(g: G, base, rough, obj, ctx=None):
        x, y, _ = g.xyz(obj)
        fibre = g.noise(g.scaled(obj, 1.0, 1.0, 1.0), 800.0, 2.0, 0.6)
        mottle = g.noise(g.shifted(obj, 3.0, 8.0, 0.0), 4.0, 3.0, 0.55)
        col = g.shade(base, g.math("MULTIPLY_ADD", fibre, 0.35, 0.82))
        col = g.shade(col, g.math("MULTIPLY_ADD", mottle, 0.5, 0.75))
        dx = g.math("SUBTRACT", hx, g.math("ABSOLUTE", x))
        dy = g.math("SUBTRACT", hy, g.math("ABSOLUTE", y))
        border = g.math("MINIMUM", dx, dy)
        near_x = g.math("LESS_THAN", dx, dy)
        along = g.math("ADD", g.math("MULTIPLY", near_x, y), g.math("MULTIPLY", g.math("SUBTRACT", 1.0, near_x), x))
        staples = g.math("MULTIPLY", g.between(border, COVER_EDGE_M - 0.004, COVER_EDGE_M + 0.004),
                         g.smooth(g.sine(along, 0.06), 0.82, 0.95))
        col = g.mix(staples, col, STAPLE)
        g.put(principled_of(g), "Normal", g.bump(mottle, 0.25, 0.01))
        return col, g.math("MULTIPLY_ADD", staples, -0.5, 0.92)

    return finish


def principled_of(g: G):
    return next(n for n in g.nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")


def halo(g: G, obj, reach: float):
    """0 to 1 round the footprints of what stands on the floor (`use_scene`): a band `reach` wide where dust, spilt
    water and the mop's misses gather. Boxes are `(cx, cy, hx, hy, yaw)`; the distance is the box's, outside it."""
    x, y, _ = g.xyz(obj)
    out = None
    for cx, cy, hx, hy, yaw in _footprints:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = g.math("SUBTRACT", x, cx), g.math("SUBTRACT", y, cy)
        lx = g.math("ADD", g.math("MULTIPLY", dx, c), g.math("MULTIPLY", dy, s))
        ly = g.math("SUBTRACT", g.math("MULTIPLY", dy, c), g.math("MULTIPLY", dx, s))
        d = g.math("MAXIMUM", g.math("SUBTRACT", g.math("ABSOLUTE", lx), hx),
                   g.math("SUBTRACT", g.math("ABSOLUTE", ly), hy))
        m = g.smooth(d, reach, 0.0)
        out = m if out is None else g.math("MAXIMUM", out, m)
    return 0.0 if out is None else out


def soften_ao(mat: bpy.types.Material, kept: float) -> None:
    """The weight of the baked ambient-occlusion layer of a library material (the Mix node that multiplies its colour by
    the AO image) cut to `kept` of what it was: a carpet's loops each have a dark gap baked in, and at 5 cm they read as
    bubbles."""
    for node in mat.node_tree.nodes:
        if (node.bl_idname == "ShaderNodeMix" and node.data_type == "RGBA" and node.blend_type == "MULTIPLY"
                and node.inputs["B"].links and node.inputs["B"].links[0].from_node.bl_idname == "ShaderNodeTexImage"):
            node.inputs["Factor"].default_value *= kept


def scale_bump(mat: bpy.types.Material, factor: float) -> None:
    """The strength of every bump of a material (the library's ambientCG ones read a height map at 0.5) times `factor`:
    paint, laminate and doors are smooth to the eye at 5 to 30 cm, and the library's bump makes them stucco."""
    for node in mat.node_tree.nodes:
        if node.bl_idname == "ShaderNodeBump":
            node.inputs["Strength"].default_value *= factor
