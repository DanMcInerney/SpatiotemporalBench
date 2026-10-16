# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The askable household kinds that the look round rebuilt so that a picture names them from the floor (M1's
pre-screen): a tote bag and a holdall (the bag), a lidded storage box and a taped carton (the box), a pair of low
trainers with white soles and laces (shoes) and a round bar stool.

Runs inside Blender's own Python and is never imported by `stbench`. Like `procedural_clutter.py`, each builder takes
the bmesh, its `DESIGN_BOX` (the catalog's size of the kind it draws, metres) and `slot`, a dict from a slot name
(`SLOTS`) to its material index, and adds geometry with its origin at the base centre (x forward, y left, z up).
`procedural.py` registers them. Every builder reaches its box to a millimetre (a test keeps it so), so the exact fit
never squashes a detail. The colour attribute is the slot a person would call the colour (the fabric of a bag, the
upper of a shoe); the parts that make the kind readable from 40 pixels away stay in neutral materials that contrast
with it: pale soles and laces on any upper, grey webbing on any fabric, a dark latch on any lid.
"""

from __future__ import annotations

import math

from mathutils import Matrix, Vector
from meshkit import lathe, placed, rod, slab, solid
from partskit import Outline, loft, rings_of, sweep
from wobble import wave

DESIGN_BOX = {"tote_bag": (0.374, 0.304, 0.314), "holdall_bag": (0.374, 0.304, 0.314),
              "storage_box": (0.40, 0.30, 0.267), "carton": (0.40, 0.30, 0.267), "shoes_pair": (0.244, 0.191, 0.080),
              "round_stool": (0.385, 0.406, 0.579)}
SLOTS = {"tote_bag": ("fabric", "webbing"), "holdall_bag": ("fabric", "webbing", "zip"),
         "storage_box": ("body", "lid", "latch"), "carton": ("board", "tape", "print"),
         "shoes_pair": ("upper", "sole", "laces", "lining"),
         "round_stool": ("seat", "frame", "foot")}


def curve(points: tuple[tuple[float, float], ...], u: float) -> float:
    """A smooth curve through (u, value) points, cosine-eased between neighbours and flat beyond the ends."""
    if u <= points[0][0]:
        return points[0][1]
    for (u0, v0), (u1, v1) in zip(points, points[1:], strict=False):
        if u <= u1:
            t = (u - u0) / (u1 - u0)
            return v0 + (v1 - v0) * (0.5 - 0.5 * math.cos(math.pi * t))
    return points[-1][1]


def _sgn_pow(x: float, e: float) -> float:
    return math.copysign(abs(x) ** e, x)


def section(n: int, hw: float, z_lo: float, z_hi: float, x: float, y: float = 0.0, p: float = 2.6) -> list:
    """A closed superellipse in the plane of constant x: half width `hw` about `y`, from z_lo to z_hi; `p` squares it (2
    is an ellipse, more is boxier)."""
    zc, hh = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
    return [(x, y + hw * _sgn_pow(math.cos(a), 2 / p), zc + hh * _sgn_pow(math.sin(a), 2 / p))
            for a in (2 * math.pi * k / n for k in range(n))]


def extrude_xy(bm, outline: list, z0: float, z1: float, slot: int) -> None:
    """A closed solid: the polygon `outline` of (x, y) points, in order round it, from z0 to z1, flat top and bottom."""
    low = [bm.verts.new((x, y, z0)) for x, y in outline]
    high = [bm.verts.new((x, y, z1)) for x, y in outline]
    for ring in (low[::-1], high):
        bm.faces.new(ring).material_index = slot
    for i in range(len(outline)):
        j = (i + 1) % len(outline)
        f = bm.faces.new((low[i], low[j], high[j], high[i]))
        f.material_index, f.smooth = slot, False


def strap(bm, path: list, width: float, thick: float, normal: tuple, slot: int) -> None:
    """A flat band along `path`, `width` across and `thick` through, its faces towards `normal` (a webbing handle
    lying on a panel). Its ends are closed."""
    n, pts, rings = Vector(normal).normalized(), [Vector(p) for p in path], []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        s = t.cross(n).normalized()
        corners = ((1, 1), (-1, 1), (-1, -1), (1, -1))
        rings.append([tuple(p + s * a * width / 2 + n * b * thick / 2) for a, b in corners])
    loft(bm, rings, slot, smooth=False)


def arch(x0: float, x1: float, z0: float, top: float, y: float, lean: float = 0.0, steps: int = 10) -> list:
    """The path of a handle in the plane y: up from (x0, z0), over in a round arch whose crown (the centre line of the
    band) is at `top`, and down to (x1, z0); the crown leans `lean` metres in y."""
    xm, w, rise = (x0 + x1) / 2, abs(x1 - x0) / 2, top - z0
    path = [(x0, y, z0), (x0, y + lean * 0.1, z0 + 0.2 * rise)]
    for i in range(steps + 1):
        a = math.pi * i / steps
        path.append((xm - w * math.cos(a), y + lean * math.sin(a), z0 + 0.3 * rise + 0.7 * rise * math.sin(a)))
    return path + [(x1, y + lean * 0.1, z0 + 0.2 * rise), (x1, y, z0)]


# -- the bag ---------------------------------------------------------------------------------------------------------
def tote_bag(bm, box, slot) -> None:
    """A soft canvas tote with an open mouth: a rounded body that slumps in at the shoulder and bows out between its
    seams, a band of webbing and a rolled edge round the mouth, a patch pocket in the fabric on each face with a webbing
    hem, and a flat handle loop on each face that rises and leans in to meet the other over the mouth. The loops' crowns
    are the box's top."""
    fabric, web = slot["fabric"], slot["webbing"]
    hx, hy, h = box[0] / 2 - 0.004, box[1] / 2 - 0.004, box[2]
    wall, outline = 0.2, Outline(6, 6, 6)

    def profile(z: float) -> tuple[float, float, float]:  # fullest at the belly, slumped in at the mouth
        t = min(1.0, z / wall)
        f = 0.92 + 0.08 * math.sin(min(1.0, t / 0.6) * math.pi / 2) - 0.045 * max(0.0, (t - 0.6) / 0.4) ** 1.5
        return hx * f, hy * f, 0.045

    def push(z: float, u: float, p: tuple) -> float:  # the panels bow out between the seams and wrinkle a little
        t = min(1.0, z / wall)
        return (0.004 * math.sin(math.pi * t) + 0.0018 * wave(u * 9.0, z * 38.0)) * min(1.0, p[4] / 0.05)

    zs = [i * wall / 10 for i in range(11)]
    loft(bm, rings_of(outline, zs, profile, push), fabric, cap_first=True, cap_last=False)
    mouth = profile(wall)
    inner = rings_of(outline, [wall, wall - 0.05], lambda z: (profile(z)[0] - 0.006, profile(z)[1] - 0.006, 0.04))
    loft(bm, inner, fabric, cap_first=False, cap_last=True)
    band = rings_of(outline, [wall - 0.05, wall - 0.022],
                    lambda z: (profile(z)[0] + 0.0035, profile(z)[1] + 0.0035, 0.045))
    loft(bm, band, web, cap_first=False, cap_last=False)
    edge = [(math.copysign(abs(math.cos(a)) ** 0.3, math.cos(a)) * (mouth[0] - 0.003),
             math.copysign(abs(math.sin(a)) ** 0.3, math.sin(a)) * (mouth[1] - 0.003), wall)
            for a in (2 * math.pi * k / 48 for k in range(48))]
    sweep(bm, edge, 0.004, fabric, segments=8, closed=True)
    for s in (1.0, -1.0):
        face = s * (profile(0.08)[1] + 0.0038)  # the panel where the pocket sits (its bow at that height included)
        solid(bm, -0.115, 0.115, min(face - s * 0.002, face + s * 0.0025), max(face - s * 0.002, face + s * 0.0025),
              0.03, 0.12, fabric, bevel=0.002, segments=1)
        solid(bm, -0.115, 0.115, min(face - s * 0.002, face + s * 0.003), max(face - s * 0.002, face + s * 0.003),
              0.112, 0.124, web, bevel=0.001, segments=1)
        path = arch(-0.075, 0.075, wall - 0.05, h - 0.019, s * (mouth[1] + 0.001), lean=-s * 0.035)
        strap(bm, path, 0.038, 0.0035, (0.0, s, 0.0), web)


def holdall_bag(bm, box, slot) -> None:
    """A zipped holdall, the sort one takes on a short trip: a rounded body with a domed top, a zip right round its
    shoulder and along its crown, two tall handles arching from the crown, a contrasting pocket with its own zip on one
    side and a pocket on each end. The handles' crowns are the box's top."""
    fabric, web, zip_ = slot["fabric"], slot["webbing"], slot["zip"]
    hx, hy, h = box[0] / 2 - 0.0025, box[1] / 2 - 0.0035, box[2]
    outline, top = Outline(7, 7, 7), 0.215

    def profile(z: float) -> tuple[float, float, float]:  # a full body, rounded over at the shoulder to a shallow dome
        f = 0.96 + 0.04 * min(1.0, z / 0.04)
        if z > 0.16:
            f *= 1.0 - 0.17 * ((z - 0.16) / (top - 0.16)) ** 1.6
        return hx * f, hy * f, 0.07 * f

    zs = [0.0, 0.01, 0.04, 0.08, 0.12, 0.16, 0.18, 0.197, 0.208, top]
    loft(bm, rings_of(outline, zs, profile), fabric, cap_first=True, cap_last=True)
    seam = rings_of(outline, [0.146, 0.156], lambda z: (hx + 0.0025, hy + 0.0025, 0.07))  # the zip round the shoulder
    loft(bm, seam, zip_, cap_first=False, cap_last=False)
    solid(bm, -0.14, 0.14, -0.006, 0.006, top - 0.002, top + 0.0025, zip_, bevel=0.0008, segments=1)  # along the crown
    for xa, xb in ((-0.1, -0.03), (0.03, 0.1)):  # two handles stand on the crown
        strap(bm, arch(xa, xb, top - 0.002, h - 0.0185, 0.0, steps=8), 0.037, 0.0038, (0.0, 1.0, 0.0), web)
    for e in (1.0, -1.0):  # a pocket on each end
        solid(bm, min(e * (hx - 0.0065), e * (hx - 0.0005)), max(e * (hx - 0.0065), e * (hx - 0.0005)), -0.075, 0.075,
              0.04, 0.125, fabric, bevel=0.003, segments=2)
    solid(bm, -0.085, 0.085, hy - 0.0065, hy - 0.0005, 0.035, 0.125, web, bevel=0.003, segments=2)  # the side pocket
    solid(bm, -0.075, 0.075, hy - 0.0085, hy - 0.004, 0.11, 0.118, zip_, bevel=0.0008, segments=1)
    # a shoulder strap lies on the other side, clipped at both ends of the shoulder and sagging between them
    sag = [(x, -hy * profile(z)[1] / hy - 0.0015, z) for x, z in
           ((-0.145, 0.178), (-0.1, 0.13), (-0.05, 0.1), (0.0, 0.092), (0.05, 0.1), (0.1, 0.13), (0.145, 0.178))]
    strap(bm, sag, 0.03, 0.003, (0.0, -1.0, 0.0), web)
    for x in (-0.145, 0.145):
        solid(bm, x - 0.012, x + 0.012, -hy - 0.0005, -hy + 0.006, 0.172, 0.196, zip_, bevel=0.002, segments=1)


# -- the box ---------------------------------------------------------------------------------------------------------
def storage_box(bm, box, slot) -> None:
    """A plastic storage box with a clip-on lid: a tapering body in the colour, a lid of the same colour a shade paler
    with ribs across it, a dark latch at each end of both long sides and a recessed grip in the lid's rim at each
    end."""
    body, lid, latch = slot["body"], slot["lid"], slot["latch"]
    hx, hy, split = box[0] / 2, box[1] / 2 - 0.004, 0.222
    outline = Outline(5, 5, 5)
    wall = rings_of(outline, [0.0, split * 0.5, split], lambda z: (hx - 0.016 + 0.014 * z / split,
                                                                   hy - 0.008 + 0.006 * z / split, 0.028))
    loft(bm, wall, body, cap_first=True, cap_last=False)
    top = box[2] - 0.005  # the lid's flat top; its ribs stand 5 mm above it
    cover = rings_of(outline, [split - 0.002, split + 0.004, top - 0.004, top],
                     lambda z: (hx - 0.002 if z < top - 0.003 else hx - 0.012,
                                hy + 0.0 if z < top - 0.003 else hy - 0.01, 0.03))
    loft(bm, cover, lid, cap_first=False, cap_last=True)
    for x in (-0.07, 0.07):  # ribs across the lid
        solid(bm, x - 0.01, x + 0.01, -hy + 0.025, hy - 0.025, top - 0.001, box[2], lid, bevel=0.002, segments=1)
    for s in (1.0, -1.0):
        for x in (-0.115, 0.115):  # a latch bridges the join of the lid and the body
            solid(bm, x - 0.025, x + 0.025, s * (hy - 0.001), s * (hy + 0.004), 0.19, 0.248, latch, bevel=0.002,
                  segments=1)
    for e in (1.0, -1.0):
        solid(bm, e * (hx - 0.003), e * hx, -0.06, 0.06, 0.232, 0.254, latch, bevel=0.002, segments=1)


def carton(bm, box, slot) -> None:
    """A shipping carton of kraft board, closed: four walls with their corners eased, the four flaps folded across the
    top with the long pair meeting along a seam, a strip of packing tape down the seam and over each end, a shipping
    label on one long side and a dark print block on the other (arrows and a mark). The tape and print are flush to a
    millimetre."""
    board, tape, print_ = slot["board"], slot["tape"], slot["print"]
    hx, hy, h = box[0] / 2, box[1] / 2 - 0.0008, box[2]
    flap = h - 0.0045
    solid(bm, -hx, hx, -hy, hy, 0, flap, board, bevel=0.0025, segments=2)
    for s_ in (1.0, -1.0):  # the long flaps meet at a seam 3 mm wide; the short flaps lie under them (a groove between)
        solid(bm, -hx + 0.0005, hx - 0.0005, min(s_ * 0.0015, s_ * (hy - 0.0005)), max(s_ * 0.0015, s_ * (hy - 0.0005)),
              flap, h - 0.0012, board, bevel=0.0012, segments=1)
    solid(bm, -hx + 0.0006, hx - 0.0006, -0.026, 0.026, h - 0.0012, h, tape, bevel=0.0004, segments=1)  # the seam
    for e in (1.0, -1.0):  # and down each end
        solid(bm, min(e * (hx - 0.0006), e * hx), max(e * (hx - 0.0006), e * hx), -0.026, 0.026, h - 0.085, h, tape,
              bevel=0.0004, segments=1)
    solid(bm, -0.075, 0.075, hy - 0.0004, hy + 0.0008, 0.075, 0.175, tape, bevel=0.0004, segments=1)  # a label
    for z in (0.1, 0.13, 0.16):
        solid(bm, -0.06, 0.04, hy + 0.0007, hy + 0.0012, z, z + 0.006, print_)
    solid(bm, -0.11, 0.11, -hy - 0.0008, -hy + 0.0004, 0.05, 0.075, print_, bevel=0.0003, segments=1)  # print block
    solid(bm, -0.04, 0.04, -hy - 0.0012, -hy - 0.0004, 0.1, 0.19, print_, bevel=0.0003, segments=1)


# -- the shoes -------------------------------------------------------------------------------------------------------
SHOE_L, SHOE_W, GAP, SOLE = 0.244, 0.086, 0.019, 0.021
WIDTH = ((0.0, 0.0), (0.02, 0.42), (0.06, 0.60), (0.15, 0.68), (0.30, 0.74), (0.45, 0.84), (0.60, 0.95), (0.72, 1.0),
         (0.84, 0.97), (0.93, 0.80), (0.98, 0.50), (1.0, 0.0))  # half width along the length, 0 heel to 1 toe
HEIGHT = ((0.0, 0.010), (0.03, 0.040), (0.07, 0.057), (0.12, 0.0513), (0.20, 0.0493), (0.30, 0.0503), (0.42, 0.052),
          (0.50, 0.052), (0.62, 0.046), (0.75, 0.038), (0.88, 0.030), (0.96, 0.018), (1.0, 0.006))  # above the sole


def _shoe(bm, slot, side: float) -> None:
    """One low trainer lying flat with its toe towards +x: a pale outsole and midsole, an upper that rises to a padded
    collar round a dark opening at the back and falls to the toe, a pale toe cap, a tongue and four laces across the
    instep. `side` is +1 for the shoe on the left and -1 for its mirror image on the right."""
    upper, sole, laces, lining = slot["upper"], slot["sole"], slot["laces"], slot["lining"]
    y0, half = side * (GAP / 2 + SHOE_W / 2), SHOE_W / 2

    def x_of(u: float) -> float:
        return -SHOE_L / 2 + u * SHOE_L

    us = [0.5 - 0.5 * math.cos(math.pi * i / 30) for i in range(31)]
    edge = [(x_of(u), y0 + side * half * curve(WIDTH, u)) for u in us]
    edge += [(x_of(u), y0 - side * half * curve(WIDTH, u)) for u in us[-2:0:-1]]
    extrude_xy(bm, edge, 0.0, 0.009, sole)
    inset = [(x_of(u), y0 + side * half * curve(WIDTH, u) * 0.985) for u in us]
    inset += [(x_of(u), y0 - side * half * curve(WIDTH, u) * 0.985) for u in us[-2:0:-1]]
    extrude_xy(bm, inset, 0.009, SOLE, sole)
    stations = [0.0, 0.015, 0.04, 0.08, 0.14, 0.22, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.86, 0.93, 0.975, 1.0]

    def rings(scale: float, bump_: float = 0.0, u_min: float = 0.0) -> list:
        return [section(24, max(0.0015, half * 0.93 * curve(WIDTH, u) * scale), SOLE - 0.002,
                        SOLE + curve(HEIGHT, u) + bump_, x_of(u), y0, p=2.7) for u in stations if u >= u_min]

    loft(bm, rings(1.0), upper)
    loft(bm, rings(1.03, 0.0015, 0.80), sole, cap_first=False)  # the toe cap: a pale rubber bumper
    for u in (0.40, 0.46, 0.52, 0.58):  # laces across the instep, over a tongue
        z = SOLE + curve(HEIGHT, u) + 0.0012
        rod(bm, (x_of(u), y0 - 0.013, z), (x_of(u), y0 + 0.013, z), 0.0021, laces, segments=8)
    slab(bm, x_of(0.36), x_of(0.63), y0 - 0.012, y0 + 0.012, SOLE + 0.038, SOLE + curve(HEIGHT, 0.5) + 0.0004, upper,
         bevel=0.002, segments=1)
    cx, cy, top = x_of(0.215), y0, SOLE + curve(HEIGHT, 0.215)  # the opening: a dark lining in a padded collar
    ring = [(cx + 0.033 * math.cos(a), cy + 0.0185 * math.sin(a)) for a in (2 * math.pi * k / 20 for k in range(20))]
    extrude_xy(bm, ring, top - 0.0012, top + 0.0012, lining)
    collar = [(cx + 0.036 * math.cos(a), cy + 0.0215 * math.sin(a), top + 0.005 - 0.0005)
              for a in (2 * math.pi * k / 32 for k in range(32))]
    sweep(bm, collar, 0.005, upper, segments=8, closed=True)
    slab(bm, x_of(0.0), x_of(0.045), y0 - 0.008, y0 + 0.008, SOLE + 0.022, SOLE + 0.059, laces, bevel=0.002,
         segments=1)  # the heel tab


def shoes_pair(bm, box, slot) -> None:
    """A pair of low trainers side by side, toes to +x, with a gap of 19 mm between them."""
    for side in (1.0, -1.0):
        _shoe(bm, slot, side)


# -- the stool -------------------------------------------------------------------------------------------------------
def round_stool(bm, box, slot) -> None:
    """A round bar stool: a padded round seat on four splayed legs with rubber feet, a thin apron under the seat and a
    ring brace across the legs. The legs stand inside the map's leg squares; the seat is the box's width."""
    seat, frame, foot = slot["seat"], slot["frame"], slot["foot"]
    z0, top = 0.529, box[2]
    stretch = Matrix.Diagonal((1.0, box[1] / box[0] / 1.0 * 1.0, 1.0, 1.0))
    r = box[0] / 2

    placed(bm, stretch, lambda: lathe(bm, [(0, z0), (r - 0.012, z0), (r - 0.004, z0 + 0.004), (r, z0 + 0.016),
                                           (r - 0.002, z0 + 0.034), (r - 0.012, top - 0.008), (r - 0.04, top - 0.002),
                                           (r - 0.09, top), (0, top)], seat, segments=64))
    apron = [(r - 0.02, z0 - 0.012), (r - 0.02, z0), (r - 0.04, z0), (r - 0.04, z0 - 0.012)]
    placed(bm, stretch, lambda: lathe(bm, apron, frame, segments=64, loop=True))
    rt, rf = 0.130, 0.18
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        ca, sa = math.cos(a), math.sin(a)
        sweep(bm, [(rt * ca, rt * sa * stretch[1][1], z0 - 0.012), (rf * ca, rf * sa * stretch[1][1], 0.008)], 0.0,
              frame, segments=12, radii=[0.0125, 0.0095])
        rod(bm, (rf * ca, rf * sa * stretch[1][1], 0.0), (rf * ca, rf * sa * stretch[1][1], 0.012), 0.0105, foot,
            segments=12)
    zr = 0.19
    rr = rf - (rf - rt) * zr / z0
    ring = [(rr * math.cos(a) * 1.0, rr * math.sin(a) * stretch[1][1], zr)
            for a in (2 * math.pi * k / 48 for k in range(48))]
    sweep(bm, ring, 0.0065, frame, segments=8, closed=True)


BUILDERS = {"tote_bag": tote_bag, "holdall_bag": holdall_bag, "storage_box": storage_box, "carton": carton,
            "shoes_pair": shoes_pair, "round_stool": round_stool}
