# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The residential dressing, drawn: `BUILDERS` maps a `proc:` recipe's builder name to `fn(d, ctx, text) -> [objects]`
(`dressing.build` calls it for every `Dressing` of `dressing_layout_residential.layout`).

Runs inside Blender's own Python and is never imported by `stbench`. A builder makes its piece in the Dressing's own
frame (origin at the base centre, x its facing, y across, z up; `d.size` is its box) and returns unlinked objects with
their mesh and materials; `dressing.build` places, names and links them (`pass_index` 0). Every mesh is fitted to the
box (`meshkit.fit`), so no vertex leaves what the guard checked. The trees are `street_tree`'s, unchanged: a stem (here
a plain trunk with limbs, no cast-iron pit) and a canopy of leaf lumps whose shape is the recipe's variant; the foliage
and bark are the street's (`materials_street`), made once per scene. Seeded by the piece's own place, so a map renders
the same.
"""

from __future__ import annotations

import hashlib
import math
import random

import bpy
import materials_street as ms
import street_tree as trees
from dressing_spec import Dressing
from materials_street import lin, plain
from meshkit import fit, lathe, slab
from scene_mesh import MeshBuilder, rod

FRAME_RGB, SEAT_RGB = (0.17, 0.32, 0.55), (0.06, 0.06, 0.07)  # the playground's: a muted blue frame, black seats
SLIDE_RGB, POLE_RGB = (0.78, 0.55, 0.12), (0.18, 0.20, 0.21)  # a yellow chute; the lamp post's grey


def _made(name: str, make):
    """The scene's material of this name, made on first use (a scene rebuild frees every material)."""
    return bpy.data.materials.get(name) or make()


def _rng(d: Dressing, text: str | None = None) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(repr((d.cls, d.pos, text)).encode()).digest()[:8], "big"))


def _object(d: Dressing, mb: MeshBuilder) -> bpy.types.Object:
    """The unlinked object of a finished builder: its mesh fitted to the Dressing's box."""
    fit(mb.bm, d.size)
    return bpy.data.objects.new(f"dressing_{d.cls}", mb.mesh())


def _paint(name: str, rgb, rough: float = 0.5):
    return _made(name, lambda: plain(name, lin(rgb), rough))


# -- trees and shrubs --------------------------------------------------------------------------------------------
def tree_stem(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A trunk flaring at its foot, narrowing to the box's top, and three limbs reaching up and out into the canopy."""
    sx, sy, sz = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    bark = mb.slot("bark", _made("bark", ms.bark))
    profile = [(0.0, 0.0), (0.2, 0.0), (0.16, 0.3), (0.12, 1.2), (0.10, sz * 0.8), (0.0, sz * 0.8)]
    lathe(mb.bm, profile, bark, segments=12)
    rng = _rng(d)
    for k in range(3):  # limbs spread round the top, at a random turn the piece's place draws
        a = rng.uniform(0.0, 6.28) + k * 2.09
        tip = (0.35 * sx * math.cos(a), 0.35 * sy * math.sin(a), sz)
        rod(mb.bm, (0.0, 0.0, sz * 0.7), tip, 0.045, bark, segments=6)
    return [_object(d, mb)]


def tree_canopy(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """The leaf mass and its leader and limbs: `street_tree.crown` in the box the layout gave (the variant seeds it)."""
    mb = MeshBuilder(f"dressing_{d.cls}")
    bark = mb.slot("bark", _made("bark", ms.bark))
    leaf = mb.slot("leaf", _made("foliage", ms.foliage))
    trees.crown(mb.bm, d.size, bark, leaf, _rng(d, text))
    return [_object(d, mb)]


def shrub(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A clump of leaf lumps filling the lower part of the box."""
    sx, sy, sz = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    leaf = mb.slot("leaf", _made("foliage", ms.foliage))
    u = min(sx, sy)
    spread = (0.3 * sx, 0.3 * sy, 0.28 * sz)
    trees.lumps(mb.bm, leaf, _rng(d, text), (0.0, 0.0, 0.5 * sz), spread, 12, (0.26 * u, 0.36 * u))
    return [_object(d, mb)]


# -- the lamp post -----------------------------------------------------------------------------------------------
def lamp_post(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A street lamp: a tapering pole at the back of the box, an arm reaching +x (towards the street) and a lamp head
    on its end."""
    sx, sy, sz = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    pole = mb.slot("pole", _paint("lamp_pole", POLE_RGB, 0.45))
    head = mb.slot("lamp_head", _paint("lamp_head", (0.10, 0.11, 0.12), 0.4))
    x0 = -sx / 2 + 0.15
    profile = [(0.0, 0.0), (0.09, 0.0), (0.07, 0.4), (0.05, sz - 0.45), (0.0, sz - 0.45)]
    lathe(mb.bm, profile, pole, segments=10, offset=(x0, 0.0))
    rod(mb.bm, (x0, 0.0, sz - 0.55), (sx / 2 - 0.2, 0.0, sz - 0.25), 0.035, pole, segments=8)
    slab(mb.bm, sx / 2 - 0.5, sx / 2, -0.15, 0.15, sz - 0.32, sz - 0.17, head, bevel=0.03, segments=2)
    return [_object(d, mb)]


# -- the playground ----------------------------------------------------------------------------------------------
def _plank(bm, x0: float, x1: float, a: tuple[float, float], b: tuple[float, float], t: float, slot: int) -> None:
    """A sloping plank from the point `a` = (y, z) to `b`, `t` thick below the line, between x0 and x1."""
    ends = [[bm.verts.new((x, y, z)) for y, z in ((a[0], a[1]), (b[0], b[1]), (a[0], a[1] - t), (b[0], b[1] - t))]
            for x in (x0, x1)]
    p, q = ends
    for face in ((p[0], p[1], p[3], p[2]), (q[0], q[2], q[3], q[1]), (p[0], q[0], q[1], p[1]), (p[2], p[3], q[3], q[2]),
                 (p[0], p[2], q[2], q[0]), (p[1], q[1], q[3], p[3])):
        bm.faces.new(face).material_index = slot


def swing_set(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """Two A-frames joined by a top bar (along y) and two swings on chains."""
    sx, sy, sz = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    frame = mb.slot("frame", _paint("play_frame", FRAME_RGB, 0.5))
    seat = mb.slot("seat", _paint("play_seat", SEAT_RGB, 0.8))
    hx, hy = sx / 2 - 0.05, sy / 2 - 0.1
    for s in (-1.0, 1.0):
        for lean in (-1.0, 1.0):
            rod(mb.bm, (lean * hx, s * hy, 0.0), (0.0, s * hy, sz - 0.05), 0.04, frame, segments=6)
    rod(mb.bm, (0.0, -hy, sz - 0.05), (0.0, hy, sz - 0.05), 0.04, frame, segments=6)
    for u in (-0.55, 0.55):
        for off in (-0.15, 0.15):
            rod(mb.bm, (0.0, u + off, sz - 0.05), (0.0, u + off, 0.55), 0.008, frame, segments=4)
        slab(mb.bm, -0.1, 0.1, u - 0.2, u + 0.2, 0.5, 0.56, seat)
    return [_object(d, mb)]


def slide(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A tower with a platform and a ladder at one end of the box's length (y) and a chute running down to the other."""
    sx, sy, sz = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    frame = mb.slot("frame", _paint("play_frame", FRAME_RGB, 0.5))
    chute = mb.slot("chute", _paint("play_slide", SLIDE_RGB, 0.35))
    y0, hx = -sy / 2, sx / 2 - 0.05
    tower = (y0 + 0.05, y0 + 1.15)
    for x in (-hx, hx):
        for y in tower:
            slab(mb.bm, x - 0.04, x + 0.04, y - 0.04, y + 0.04, 0.0, sz, frame)
    slab(mb.bm, -hx, hx, tower[0], tower[1], 1.45, 1.52, frame)
    for k in range(5):  # the ladder's rungs on the tower's far face
        slab(mb.bm, -hx, hx, y0 - 0.0, y0 + 0.05, 0.25 + 0.25 * k, 0.29 + 0.25 * k, frame)
    _plank(mb.bm, -0.28, 0.28, (tower[1], 1.5), (sy / 2, 0.22), 0.05, chute)
    return [_object(d, mb)]


def climbing_frame(d: Dressing, ctx, text: str | None) -> list[bpy.types.Object]:
    """A cube frame of rods with a platform and a grid of bars on two sides."""
    sx, sy, sz = d.size
    mb = MeshBuilder(f"dressing_{d.cls}")
    frame = mb.slot("frame", _paint("play_frame", FRAME_RGB, 0.5))
    hx, hy = sx / 2 - 0.05, sy / 2 - 0.05
    for x in (-hx, hx):
        for y in (-hy, hy):
            rod(mb.bm, (x, y, 0.0), (x, y, sz - 0.05), 0.04, frame, segments=6)
    for z in (0.5, 1.0, 1.5, sz - 0.05):
        for x in (-hx, hx):
            rod(mb.bm, (x, -hy, z), (x, hy, z), 0.025, frame, segments=6)
        for y in (-hy, hy):
            rod(mb.bm, (-hx, y, z), (hx, y, z), 0.025, frame, segments=6)
    slab(mb.bm, -hx, hx, -hy, hy, 1.1, 1.15, frame)
    return [_object(d, mb)]


BUILDERS = {f.__name__: f for f in (tree_stem, tree_canopy, shrub, lamp_post, swing_set, slide, climbing_frame)}
