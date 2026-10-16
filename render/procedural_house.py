# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The suburb's two buildings as lists of parts: a two-storey detached house with a pitched roof, and an apartment block
with a flat one.

Runs inside Blender's own Python (it imports `shapes`) and is never imported by `stbench`. Each function takes the
entity size (depth, width, height), the colour attribute and `detailed` and returns `Part`s in the entity's frame
(origin at the base centre, x = front, y = left, z up), every one inside the size's box, every one solid to the ground:
the blocking footprint is the box. A drone at 15 m over 8 m houses sees roofs, so the roof is drawn whole, as the top of
the piece: a house is the walls, the gable wedges that close them under the roof, two roof planes (thin boxes, tilted)
a ridge cap and a chimney; a block is the walls, a roof deck inside a parapet and a lift housing. `detailed` adds what
is seen from the ground and from 7 m: windows as a white frame, a dark pane and a mullion, doors, a plinth and gutters
(the far scenery's houses leave them out).

A house is `size[2]` high at its ridge cap (the chimney stands level with it): its eaves are `EAVES_M` up, its roof
pitch whatever joins the two over the half depth, and it has a garage door on its front when it is `GARAGE_MIN_WIDTH_M`
wide. The walls stand `OVERHANG_M` inside the box at the front and back and `GABLE_M` inside it at the gables, so the
roof's eaves are in the box; a block's walls stand `BAND_M` inside it all round, which leaves room for its window
frames, cornice and entrance canopy. A part names a palette key or a residential key (`materials_residential`): the
wall finish is the colour attribute's (`FINISH`), the roof's the house's (`ROOF`).
"""

from __future__ import annotations

import math

from shapes import Part, box, wedge

EAVES_M, STOREY_M, PARAPET_M = 5.5, 3.0, 0.9  # a house's wall top; a block's storey; the parapet on its roof
OVERHANG_M, GABLE_M, BAND_M = 0.35, 0.25, 0.4  # the box beyond a house's walls (front, back; gables); a block's
SLAB_T, CAP_W, CAP_UP = 0.12, 0.30, 0.06  # a roof plane's thickness; the ridge cap's width and rise over the planes
PLINTH_M, PROUD_M = 0.35, 0.07  # the stone course at a house's foot; how far a window frame stands off the wall
GARAGE_MIN_WIDTH_M = 9.0
SILL_M, PITCH_M = 0.9, 3.0  # a window's sill above its floor; a block's bay
FINISH = {"red": "brick", "brown": "brick", "orange": "brick", "white": "render", "beige": "render",
          "yellow": "render", "pink": "render"}  # the wall finish of a colour; any other is siding
ROOF = {"red": "charcoal", "orange": "terracotta", "brown": "brown", "beige": "brown", "yellow": "brown",
        "grey": "slate", "blue": "slate"}  # the roof of a house of a colour; any other is charcoal


def wall_key(colour: str, default: str = "siding") -> str:
    return f"{FINISH.get(colour, default)}:{colour}"


class Walls:
    """The four wall planes of a body whose walls stand at x = +-`wx` and y = +-`wy`, to lay boxes on: `on(face, ...)`
    is the box that spans `u0` to `u1` along the face (y for the faces `+x` and `-x`, x for `+y` and `-y`), `z0` to `z1`
    up, and `n0` to `n1` out of the wall."""

    def __init__(self, wx: float, wy: float):
        self.wx, self.wy = wx, wy

    def on(self, face: str, u0: float, u1: float, z0: float, z1: float, n0: float, n1: float, mat: str) -> Part:
        if face == "+x":
            return box(self.wx + n0, self.wx + n1, u0, u1, z0, z1, mat)
        if face == "-x":
            return box(-self.wx - n1, -self.wx - n0, u0, u1, z0, z1, mat)
        if face == "+y":
            return box(u0, u1, self.wy + n0, self.wy + n1, z0, z1, mat)
        return box(u0, u1, -self.wy - n1, -self.wy - n0, z0, z1, mat)


def window(w: Walls, face: str, u: float, zc: float, width: float, height: float) -> list[Part]:
    """A window centred on `u` and `zc`: a white frame, a dark pane set in it (a room behind) and a mullion."""
    a, b, z0, z1, inset = u - width / 2, u + width / 2, zc - height / 2, zc + height / 2, 0.08
    return [w.on(face, a, b, z0, z1, 0.0, PROUD_M, "frame"),
            w.on(face, a + inset, b - inset, z0 + inset, z1 - inset, 0.0, PROUD_M + 0.012, "dark"),
            w.on(face, u - 0.025, u + 0.025, z0 + inset, z1 - inset, 0.0, PROUD_M + 0.03, "frame")]


# -- the house ---------------------------------------------------------------------------------------------------
def pitch(rise: float, run: float) -> float:
    """tan of the pitch of a roof plane `SLAB_T` thick whose top surface rises `rise` over the horizontal `run` from the
    point where its underside meets the wall top."""
    t = (rise - SLAB_T) / run
    for _ in range(3):
        t = (rise - SLAB_T * math.sqrt(1.0 + t * t)) / run
    return t


def _plane(sign: float, hx: float, hy: float, ridge: float, th: float, mat: str) -> Part:
    """The roof plane over x from 0 to `sign * hx`: a thin box tilted to fall from the ridge line at x = 0 (its top
    surface passing through (0, ridge)) to the eaves at `sign * hx`."""
    ang = math.atan(th)
    c, s = math.cos(ang), math.sin(ang)
    return Part("box", (sign * (hx / 2 - s * SLAB_T / 2), 0.0, ridge - hx * th / 2 - c * SLAB_T / 2),
                (hx / c, 2 * hy, SLAB_T), (0.0, sign * ang, 0.0), mat)


def house(size, colour: str, detailed: bool = True) -> list[Part]:
    sx, sy, sz = size
    hx, hy = sx / 2, sy / 2
    wx, wy = hx - OVERHANG_M, hy - GABLE_M
    eaves = min(EAVES_M, sz - 2.5)  # a house at least 8 m high has its eaves 5.5 m up
    wall, roof = wall_key(colour), f"shingle:{ROOF.get(colour, 'charcoal')}"
    ridge = sz - CAP_UP
    th = pitch(ridge - eaves, wx)
    mid = ridge - SLAB_T * math.sqrt(1.0 + th * th) - eaves + 0.02  # the gable's height at the ridge, over the wall top
    parts = [box(-wx, wx, -wy, wy, 0.0, eaves, wall),
             wedge(-wx, 0.0, -wy, wy, eaves, 0.02, mid, wall), wedge(0.0, wx, -wy, wy, eaves, mid, 0.02, wall),
             _plane(1.0, hx, hy, ridge, th, roof), _plane(-1.0, hx, hy, ridge, th, roof),
             box(-CAP_W / 2, CAP_W / 2, -hy, hy, ridge - 0.08, sz, roof),
             box(-0.45 * hx - 0.35, -0.45 * hx + 0.35, 0.42 * hy - 0.5, 0.42 * hy + 0.5, eaves, sz - 0.06, "brick:red"),
             box(-0.45 * hx - 0.42, -0.45 * hx + 0.42, 0.42 * hy - 0.57, 0.42 * hy + 0.57, sz - 0.06, sz, "stone")]
    if detailed:
        parts += _details(Walls(wx, wy), hx, hy, wy, eaves, ridge, th, sy >= GARAGE_MIN_WIDTH_M)
    return parts


def _details(w: Walls, hx: float, hy: float, wy: float, eaves: float, ridge: float, th: float, garage: bool
             ) -> list[Part]:
    """What a house has that the far scenery's does not: the plinth, the gutters, the front door and step, the garage
    door and the windows of all four faces."""
    wx = w.wx
    eave_z = ridge - hx * th  # the top of the roof plane at its edge
    out = [box(-wx - 0.03, wx + 0.03, -wy - 0.03, wy + 0.03, 0.0, PLINTH_M, "stone")]
    out += [box(sign * (hx - 0.14) - 0.07, sign * (hx - 0.14) + 0.07, -hy + 0.1, hy - 0.1, eave_z - 0.24, eave_z - 0.04,
                "frame") for sign in (1.0, -1.0)]
    door = -0.35 * wy
    out += [w.on("+x", door - 0.5, door + 0.5, 0.0, 2.1, 0.0, PROUD_M, "door_entrance"),
            w.on("+x", door - 0.7, door + 0.7, 0.0, 0.15, 0.0, 0.3, "stone")]
    ground, upper = 1.55, eaves - 1.5
    for u in (0.15 * wy, 0.62 * wy):
        if not (garage and u > wy - 3.2):
            out += window(w, "+x", u, ground, 1.4, 1.3)
    if garage:
        out += [w.on("+x", wy - 2.7, wy - 0.3, 0.0, 2.2, 0.0, PROUD_M, "door"),
                w.on("+x", wy - 2.6, wy - 0.4, 1.0, 1.06, 0.0, PROUD_M + 0.02, "dark")]
    for u in (-0.6 * wy, 0.0, 0.6 * wy):
        out += window(w, "+x", u, upper, 1.3, 1.3)
    for u in (-0.55 * wy, 0.0, 0.55 * wy):  # the back: three windows above, the patio door and a window below
        out += window(w, "-x", u, upper, 1.3, 1.3)
    out += [w.on("-x", 0.1 * wy - 0.9, 0.1 * wy + 0.9, 0.0, 2.1, 0.0, PROUD_M, "frame"),
            w.on("-x", 0.1 * wy - 0.8, 0.1 * wy + 0.8, 0.1, 2.0, 0.0, PROUD_M + 0.012, "dark")]
    out += window(w, "-x", -0.5 * wy, ground, 1.4, 1.3)
    for face in ("+y", "-y"):  # the gables: two windows below, one above and a small one in the gable
        out += window(w, face, -0.5 * wx, ground, 1.2, 1.3) + window(w, face, 0.5 * wx, ground, 1.2, 1.3)
        out += window(w, face, 0.0, upper, 1.2, 1.3) + window(w, face, 0.0, eaves + 1.0, 0.7, 0.7)
    return out


# -- the block ---------------------------------------------------------------------------------------------------
def storeys(height: float) -> int:
    """How many storeys a block `height` high has: its parapet is `PARAPET_M` of it."""
    return max(1, round((height - PARAPET_M) / STOREY_M))


def block(size, colour: str, detailed: bool = True) -> list[Part]:
    sx, sy, sz = size
    wx, wy = sx / 2 - BAND_M, sy / 2 - BAND_M
    top, n = sz - PARAPET_M, storeys(sz)
    w, wall, pitch_z = Walls(wx, wy), wall_key(colour, "render"), top / n
    parts = [box(-wx, wx, -wy, wy, 0.0, top, wall),  # the walls, then the roof deck inside the parapet's four walls
             box(-wx + 0.3, wx - 0.3, -wy + 0.3, wy - 0.3, top, top + 0.03, "dark"),
             box(-wx, wx, -wy, -wy + 0.3, top, sz, "stone"), box(-wx, wx, wy - 0.3, wy, top, sz, "stone"),
             box(-wx, -wx + 0.3, -wy + 0.3, wy - 0.3, top, sz, "stone"),
             box(wx - 0.3, wx, -wy + 0.3, wy - 0.3, top, sz, "stone"),
             box(-1.5, 1.5, -1.5, 1.5, top, top + 0.8, "stone"),  # the lift housing and an air-conditioning unit
             box(-wx + 1.0, -wx + 2.2, wy - 2.4, wy - 1.0, top, top + 0.6, "steel")]
    if not detailed:
        return parts
    parts += [box(-wx - 0.08, wx + 0.08, -wy - 0.08, wy + 0.08, top - 0.25, top, "stone")]
    for face, length in (("+x", 2 * wy), ("-x", 2 * wy), ("+y", 2 * wx), ("-y", 2 * wx)):
        bays = max(1, int(length // PITCH_M))
        for b in range(bays):
            u = -length / 2 + (b + 0.5) * length / bays
            entrance = face == "+x" and b == bays // 2
            for k in range(n):
                if entrance and k == 0:
                    continue
                parts += window(w, face, u, k * pitch_z + SILL_M + 0.7, 1.5, 1.4)
            if entrance:
                parts += [w.on(face, u - 1.0, u + 1.0, 0.0, 2.4, 0.0, PROUD_M, "frame"),
                          w.on(face, u - 0.85, u + 0.85, 0.1, 2.3, 0.0, PROUD_M + 0.012, "dark"),
                          w.on(face, u - 1.5, u + 1.5, 2.6, 2.75, 0.0, BAND_M - 0.02, "stone")]
    return parts
