# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""A tiny shader-node DSL for the city's procedural materials: build a node graph by calling functions on sockets and
numbers.

Runs inside Blender's own Python and is never imported by `stbench`. `G(material.node_tree)` adds nodes to a tree; every
method takes sockets or plain numbers for its inputs and returns the output socket, so a procedural shader reads as an
expression. The materials built with it need no image and no UVs: they work on object coordinates.
"""

from __future__ import annotations

import bpy

_MIX_IN = {"FLOAT": (2, 3), "VECTOR": (4, 5), "RGBA": (6, 7)}
_MIX_OUT = {"FLOAT": 0, "VECTOR": 1, "RGBA": 2}


def new_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    return mat


def principled(mat: bpy.types.Material) -> bpy.types.Node:
    return next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")


class G:
    """Graph builder for one node tree."""

    def __init__(self, tree: bpy.types.NodeTree):
        self.nt = tree

    def _put(self, socket, value) -> None:
        if isinstance(value, bpy.types.NodeSocket):
            self.nt.links.new(value, socket)
        elif value is not None:
            cur = socket.default_value
            if hasattr(cur, "__len__"):  # a colour wants four numbers, a vector three
                value = tuple(value) if hasattr(value, "__len__") else (value,) * len(cur)
                if len(value) == 3 and len(cur) == 4:
                    value = (*value, 1.0)
            socket.default_value = value

    def node(self, idname: str, inputs: dict | None = None, **props):
        n = self.nt.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (inputs or {}).items():
            self._put(n.inputs[k], v)
        return n

    def math(self, op: str, a, b=None, c=None, clamp: bool = False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            self._put(n.inputs[i], v)
        return n.outputs[0]

    def vmath(self, op: str, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self._put(n.inputs[0], a)
        self._put(n.inputs[1], b)
        if scale is not None:
            self._put(n.inputs["Scale"], scale)
        return n.outputs["Vector"]

    def mix(self, fac, a, b, kind: str = "RGBA"):
        n = self.node("ShaderNodeMix", data_type=kind)
        self._put(n.inputs[0], fac)
        ia, ib = _MIX_IN[kind]
        self._put(n.inputs[ia], a)
        self._put(n.inputs[ib], b)
        return n.outputs[_MIX_OUT[kind]]

    def xyz(self, vec):
        n = self.node("ShaderNodeSeparateXYZ", {"Vector": vec})
        return n.outputs["X"], n.outputs["Y"], n.outputs["Z"]

    def combine(self, x, y, z):
        return self.node("ShaderNodeCombineXYZ", {"X": x, "Y": y, "Z": z}).outputs["Vector"]

    def between(self, v, lo, hi):
        """1 where lo < v < hi, else 0."""
        return self.math("MULTIPLY", self.math("GREATER_THAN", v, lo), self.math("LESS_THAN", v, hi))

    def ramp(self, fac, stops, interp: str = "LINEAR"):
        """Colour ramp through `stops` = [(position, (r, g, b)), ...]."""
        n = self.node("ShaderNodeValToRGB", {"Fac": fac})
        n.color_ramp.interpolation = interp
        els = n.color_ramp.elements
        while len(els) < len(stops):
            els.new(0.5)
        for e, (pos, col) in zip(els, stops, strict=True):
            e.position, e.color = pos, (*col, 1.0)
        return n.outputs["Color"]

    def noise(self, vec, scale: float, detail: float = 2.0, rough: float = 0.5):
        n = self.node("ShaderNodeTexNoise", {"Vector": vec, "Scale": scale, "Detail": detail, "Roughness": rough})
        return n.outputs["Fac"]

    def white(self, vec):
        return self.node("ShaderNodeTexWhiteNoise", {"Vector": vec}, noise_dimensions="3D").outputs["Value"]

    def grid(self, v, pitch: float, width: float):
        """1 within `width` of a multiple of `pitch` along the scalar v (the joints of a slab pattern)."""
        shifted = self.math("ADD", v, pitch / 2)
        wrapped = self.math("FLOORED_MODULO", shifted, pitch)
        dist = self.math("ABSOLUTE", self.math("SUBTRACT", wrapped, pitch / 2))
        return self.math("LESS_THAN", dist, width / 2)

    # -- what the dressing and wear materials add ----------------------------------------------------------------
    def put(self, node, name: str, value) -> None:
        """Set the input `name` of a node to a number, a colour or vector, or link a socket to it."""
        self._put(node.inputs[name], value)

    def smooth(self, v, lo: float, hi: float):
        """0 at or below `lo`, 1 at or above `hi`, a smooth step between (the two may be given in either order)."""
        n = self.node("ShaderNodeMapRange", {"Value": v, "From Min": lo, "From Max": hi},
                      interpolation_type="SMOOTHSTEP", clamp=True)
        return n.outputs["Result"]

    def scaled(self, vec, sx: float, sy: float, sz: float):
        """A vector with each axis multiplied: a stretched coordinate, so a noise on it makes streaks."""
        return self.vmath("MULTIPLY", vec, (sx, sy, sz))

    def shifted(self, vec, dx: float, dy: float, dz: float):
        """A vector moved by an offset: another sample of the same noise, uncorrelated with the first."""
        return self.vmath("ADD", vec, (dx, dy, dz))

    def band(self, v, lo: float, hi: float):
        """1 where lo < v < hi else 0 (alias of `between`, named for masks along an axis)."""
        return self.between(v, lo, hi)

    def triangle(self, v, pitch: float):
        """A triangle wave of the scalar v: 0 at multiples of `pitch`, 1 halfway between (ribs, flutes, studs)."""
        return self.math("MULTIPLY", self.math("PINGPONG", v, pitch / 2), 2.0 / pitch)

    def sine(self, v, pitch: float):
        """0 to 1 over a `pitch`, smooth: a pleat or a weave thread."""
        return self.math("MULTIPLY_ADD", self.math("SINE", self.math("MULTIPLY", v, 6.2831853 / pitch)), 0.5, 0.5)

    def tint(self, fac, a, b):
        """The colour `a` multiplied by the colour `b` as much as `fac` (0 leaves `a`, 1 gives `a * b`)."""
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        self._put(n.inputs[0], fac)
        self._put(n.inputs[6], a)
        self._put(n.inputs[7], b)
        return n.outputs[2]

    def shade(self, colour, f):
        """The colour scaled by the scalar `f` in every channel."""
        return self.tint(1.0, colour, self.combine(f, f, f))

    def bump(self, height, strength: float, distance: float):
        """The perturbed normal of a height field."""
        return self.node("ShaderNodeBump", {"Height": height, "Strength": strength, "Distance": distance}).outputs[
            "Normal"]
