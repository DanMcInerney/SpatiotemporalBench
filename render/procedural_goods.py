# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's carried kinds as procedural geometry: a wrapped pallet (`procedural_pallet.py`), a tote, a drum, a
sack and a bin.

Runs inside Blender's own Python and is never imported by `stbench`. Each builder takes the bmesh, the kind's
`DESIGN_BOX` (the extents it is built to, metres) and `slot`, a dict from a slot name (`procedural.SLOTS`) to its
material index, and adds geometry with its origin at the base centre (x forward, y left, z up). `procedural.py`
registers them.
The dimensions are the real ones the catalog quotes (ISO 6780 pallet, Euro container, 200 l drum). The colour attribute
shows on the part a person would call the colour: the pallet's stretch wrap, the tote's, drum's and bin's body, the
sack's cloth.
"""

from __future__ import annotations

import math

from meshkit import lathe, prism, rod, solid
from partskit import Outline, loft, rings_of
from procedural_pallet import pallet
from wobble import bump, smoothstep, wave

TOTE_OUTLINE = Outline(48, 32, 5)
TOTE_TAPER, TOTE_WALL, TOTE_FLOOR = 0.9, 0.006, 0.012
RIB_PITCH, RIM = 0.05, 0.0075


def tote(bm, box, slot) -> None:
    """A 600 x 400 Euro container: an open tub that widens upwards, with vertical ribs on its walls, a stacking rim, a
    smooth label panel on one long side and a hand-hold (a dark opening, recessed in a raised surround) in each short
    end."""
    body, grip = slot["body"], slot["grip"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    rim_z, z0, z1, half = h - 0.03, 0.209, 0.261, 0.085  # the hand-hold's opening spans z0..z1 and +-half

    def wall(z: float) -> float:  # an end wall's outer face at height z, before ribs
        return (hx - RIM) * (TOTE_TAPER + (1.0 - TOTE_TAPER) * z / h)

    def profile(z: float) -> tuple[float, float, float]:
        k = TOTE_TAPER + (1.0 - TOTE_TAPER) * z / h  # the rim band's 7.5 mm brings the top to the box
        return (hx - RIM) * k, (hy - RIM) * k, 0.03

    def push(z: float, u: float, p: tuple) -> float:
        x, y, nx, ny, edge = p
        d = 0.0
        if z >= rim_z:  # the stacking rim stands out as a band, its lower edge a chamfer
            d += RIM * smoothstep(rim_z - 0.004, rim_z + 0.002, z)
        if 0.05 < z < rim_z - 0.01 and edge > 0.02:  # vertical ribs, 4 mm proud, every 5 cm along the face
            along = x if abs(ny) > 0.5 else y
            rib = bump(((along + RIB_PITCH / 2) % RIB_PITCH) - RIB_PITCH / 2, 0.011)
            label = abs(ny) > 0.5 and ny > 0 and abs(x) < 0.11 and 0.1 < z < 0.22
            grip_zone = abs(nx) > 0.5 and abs(y) < 0.125 and 0.17 < z < 0.3
            fade = smoothstep(0.05, 0.075, z) * (1.0 - smoothstep(rim_z - 0.035, rim_z - 0.01, z))
            d += 0.004 * rib * (0.0 if label or grip_zone else 1.0) * fade
        if abs(nx) > 0.5:  # the hand-hold: a raised surround round a recess
            wall_d = max(abs(y) - (half + 0.02), abs(z - (z0 + z1) / 2) - (z1 - z0) / 2 - 0.02)
            inside = smoothstep(0.012, 0.0, max(abs(y) - half, abs(z - (z0 + z1) / 2) - (z1 - z0) / 2))
            d += 0.005 * smoothstep(0.02, 0.0, wall_d) - 0.011 * inside
        return d

    outer = rings_of(TOTE_OUTLINE, [i * 0.01 for i in range(int(h / 0.01))] + [h], profile, push)
    inner_zs = [h - i * 0.012 for i in range(int((h - TOTE_FLOOR) / 0.012))] + [TOTE_FLOOR]
    inner = rings_of(TOTE_OUTLINE, inner_zs, lambda z: (profile(z)[0] - TOTE_WALL, profile(z)[1] - TOTE_WALL, 0.026))
    loft(bm, outer + inner, body, cap_first=True, cap_last=True)
    for s in (-1.0, 1.0):  # the opening itself: a dark slab 1 mm proud of the recess's floor
        prism(bm, [(s * (wall(z0) - 0.012), z0), (s * (wall(z0) - 0.005), z0), (s * (wall(z1) - 0.005), z1),
                   (s * (wall(z1) - 0.012), z1)], -half, half, grip)


def drum(bm, box, slot) -> None:
    """A 200 l drum: a body with two rolling hoops and a recessed, ribbed head, a steel lock ring with a closing lug on
    top, a rolled steel chime at the foot and two bungs. The body is the colour attribute."""
    body, rings = slot["body"], slot["rings"]
    r, h = box[0] / 2, box[2]
    rb = r - 0.008  # the body; the hoops, the lug and the chime reach r
    profile = [(0.0, 0.014), (rb - 0.025, 0.014), (rb - 0.006, 0.026), (rb, 0.045), (rb, 0.2)]
    for zc in (0.31, 0.58):  # a rolling hoop: a rounded swelling the width of a hand
        profile += [(rb + (r - rb) * math.sin(math.pi * t) ** 1.2, zc - 0.045 + 0.09 * t)
                    for t in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]
        profile += [(rb, zc + 0.05)]
    profile += [(rb, 0.8), (rb - 0.012, 0.846), (rb - 0.04, 0.86), (rb - 0.045, 0.852)]
    for rr in (0.2, 0.13):  # two stamped ribs round the head
        profile += [(rr, 0.852), (rr - 0.006, 0.858), (rr - 0.012, 0.852)]
    profile += [(0.0, 0.852)]
    lathe(bm, profile, body)
    lathe(bm, [(rb - 0.03, 0.0), (r, 0.0), (r, 0.03), (rb - 0.025, 0.036)], rings, loop=True)  # the chime
    ring_r = r - 0.012  # the lock ring, a little inside the hoops
    lathe(bm, [(rb - 0.03, 0.846), (ring_r, 0.846), (ring_r, h - 0.012), (ring_r - 0.01, h), (rb - 0.04, h),
               (rb - 0.04, 0.862)], rings, loop=True)
    for dy in (-1.0, 1.0):  # the ring's closing lug and its bolt
        solid(bm, ring_r - 0.01, r, dy * 0.034 - 0.012, dy * 0.034 + 0.012, 0.842, h - 0.004, rings, bevel=0.004,
              segments=1)
    rod(bm, (r - 0.008, -0.034, 0.862), (r - 0.008, 0.034, 0.862), 0.007, rings, segments=8)
    for rad, ang in ((0.03, 0.0), (0.019, 2.2)):  # the bungs: a flange with a raised plug
        cx, cy = 0.12 * math.cos(ang) - 0.02, 0.12 * math.sin(ang)
        rod(bm, (cx, cy, 0.852), (cx, cy, 0.862), rad + 0.008, rings, segments=16)
        rod(bm, (cx, cy, 0.862), (cx, cy, 0.874), rad, rings, segments=16)


def sack(bm, box, slot) -> None:
    """A woven sack lying on its side, filled: a flat underside and a soft, uneven dome whose short ends are pinched
    flat along a stitched seam, with creases running in from the seam and a slack, sagging middle."""
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    nu, nv = 56, 32

    def top(u: float, v: float) -> float:
        a = abs(u)
        end = (1.0 - (a / 0.97) ** 3.4) ** 0.55 if a < 0.97 else 0.0  # a long plateau and steep, pinched ends
        side = max(0.0, 1.0 - abs(v) ** 2.3) ** 0.5
        lumps = 0.06 * wave(3.0 * u, 2.5 * v, 1.0) + 0.04 * math.sin(4.2 * u + 1.0) * math.cos(3.0 * v)
        z = h * end * side * (1.0 + lumps)
        if end * side > 0.02:
            z += 0.012 * bump(a - 0.86, 0.045)  # the stitched seam is a ridge
            z += 0.006 * math.sin(14.0 * (v + 0.35 * u) + 1.0) * smoothstep(0.5, 0.9, a) * side  # creases from the seam
        return max(0.0, min(h, z))

    grid = [[bm.verts.new((u * hx, v * hy, top(u, v))) for v in (-1.0 + 2.0 * j / nv for j in range(nv + 1))]
            for u in (-1.0 + 2.0 * i / nu for i in range(nu + 1))]
    under = [[v if (i in (0, nu) or j in (0, nv)) else bm.verts.new((v.co.x, v.co.y, 0.0))
              for j, v in enumerate(row)] for i, row in enumerate(grid)]
    for g, flip in ((grid, False), (under, True)):
        for i in range(nu):
            for j in range(nv):
                quad = [g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]]
                f = bm.faces.new(quad[::-1] if flip else quad)
                f.material_index, f.smooth = slot["body"], True


def bin_(bm, box, slot) -> None:
    """An open picking bin: a full-height back with a hanging lip along its top, sides that slope down to a low front
    lip with a label holder, a ribbed floor and a rolled edge."""
    body = slot["body"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    t, lip = 0.005, 0.085
    x_back = -hx + 0.018  # the back wall's outer face; the hanging lip reaches back to -hx
    solid(bm, x_back, hx - 0.008, -hy, hy, 0.0, 0.008, body, bevel=0.003, segments=1)
    solid(bm, x_back, x_back + t, -hy, hy, 0.0, h, body, bevel=0.002, segments=1)  # the back wall
    solid(bm, -hx, x_back + t, -hy, hy, h - 0.016, h, body, bevel=0.003, segments=1)  # the hanging lip
    solid(bm, hx - 0.013, hx - 0.008, -hy, hy, 0.0, lip, body, bevel=0.002, segments=1)  # the front lip
    for y0, y1 in ((-hy, -hy + t), (hy - t, hy)):
        prism(bm, [(x_back, 0.0), (hx - 0.008, 0.0), (hx - 0.008, lip), (x_back, h - 0.016)], y0, y1, body)  # the sides
        left = y0 < 0  # the rolled edge: the side wall thickened along its sloping top
        prism(bm, [(x_back, h - 0.034), (hx - 0.014, lip - 0.012), (hx - 0.008, lip), (x_back, h - 0.016)],
              y0 if left else y0 - 0.004, y1 + 0.004 if left else y1, body)
    for y in (-0.1, -0.04, 0.04, 0.1):  # ribs on the back wall and the floor
        solid(bm, x_back, x_back + t + 0.003, y - 0.004, y + 0.004, 0.01, h - 0.02, body)
        solid(bm, x_back + t, hx - 0.013, y - 0.004, y + 0.004, 0.008, 0.013, body)
    solid(bm, hx - 0.008, hx, -0.075, 0.075, 0.012, 0.016, body)  # the label holder: a frame on the front lip
    solid(bm, hx - 0.008, hx, -0.075, 0.075, lip - 0.016, lip - 0.012, body)
    for s in (-1.0, 1.0):
        solid(bm, hx - 0.008, hx, s * 0.075 - 0.003, s * 0.075 + 0.003, 0.012, lip - 0.012, body)


BUILDERS = {"pallet": pallet, "tote": tote, "drum": drum, "sack": sack, "bin": bin_}
