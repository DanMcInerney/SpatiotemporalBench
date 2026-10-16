# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's clutter that no downloadable model covers, as procedural geometry: a floor fan, an air purifier, a
space heater, a bathroom scale, an umbrella, a rolled yoga mat, a pair of dumbbells, an extension cord, a power strip, a
cable bundle, a door stop, a floor lamp, and books standing on a shelf (three sizes).

Runs inside Blender's own Python and is never imported by `stbench`. Like the warehouse's, each builder takes the bmesh,
its `DESIGN_BOX` (the extents it is built to, metres) and `slot`, a dict from a slot name (`SLOTS`) to its material
index, and adds geometry with its origin at the base centre (x forward, y left, z up). `procedural.py` registers them.
Every builder reaches its box to a millimetre (a test keeps it so), so the exact fit never squashes a detail. The colour
attribute shows on the part a person would call the colour: the body of an appliance, the cable, the cover of a book.
A clutter entity's size is its model's box, so these boxes are the sizes that `clutter.place` draws with.
"""

from __future__ import annotations

import math

from mathutils import Matrix
from meshkit import lathe, placed, prism, rod, slab, solid
from partskit import sweep

DESIGN_BOX = {
    "floor_fan": (0.22, 0.34, 0.47), "air_purifier": (0.25, 0.25, 0.55), "space_heater": (0.16, 0.22, 0.55),
    "bathroom_scale": (0.30, 0.30, 0.028), "umbrella": (0.85, 0.075, 0.075), "yoga_mat": (0.62, 0.15, 0.15),
    "dumbbells": (0.28, 0.28, 0.11), "extension_cord": (0.50, 0.18, 0.03), "power_strip": (0.50, 0.17, 0.036),
    "router": (0.14, 0.24, 0.19),
    "cable_bundle": (0.18, 0.16, 0.034), "door_stop": (0.10, 0.045, 0.032), "floor_lamp": (0.34, 0.34, 1.50),
    "shelved_book_s": (0.085, 0.024, 0.15), "shelved_book_m": (0.09, 0.03, 0.17),
    "shelved_book_l": (0.095, 0.038, 0.19)}
SLOTS = {
    "floor_fan": ("body", "grille", "blade"), "air_purifier": ("body", "vent"), "space_heater": ("body", "vent"),
    "bathroom_scale": ("body", "glass"), "umbrella": ("canopy", "metal", "handle"), "yoga_mat": ("mat", "strap"),
    "dumbbells": ("iron", "handle"), "extension_cord": ("cable", "plug"), "power_strip": ("body", "face", "cable"),
    "router": ("body", "antenna", "face"),
    "cable_bundle": ("cable", "tie"), "door_stop": ("body",), "floor_lamp": ("body", "shade"),
    "shelved_book_s": ("cover", "pages"), "shelved_book_m": ("cover", "pages"), "shelved_book_l": ("cover", "pages")}
ALONG_X = Matrix.Rotation(math.pi / 2, 4, "Y")  # a lathe's z axis becomes x (its x axis becomes -z)


def _along_x(bm, x: float, y: float, z: float, build, spin: float = 0.0) -> None:
    """`build()` makes a lathe about the z axis; put it about the x axis through (y, z), centred at x."""
    placed(bm, Matrix.Translation((x, y, z)) @ ALONG_X @ Matrix.Rotation(spin, 4, "Z"), build)


def floor_fan(bm, box, slot) -> None:
    """A drum fan on a pedestal: a round base, a short neck, a drum with a wire grille front and back (rings and radial
    wires) round five blades on a hub. The drum's top is the box's."""
    body, grille, blade = slot["body"], slot["grille"], slot["blade"]
    r, zc, depth = box[1] / 2, box[2] - box[1] / 2, 0.08
    lathe(bm, [(0, 0), (0.105, 0), (0.11, 0.008), (0.11, 0.02), (0.05, 0.028), (0, 0.028)], body)
    rod(bm, (0, 0, 0.028), (0, 0, zc), 0.02, body)

    def head() -> None:
        lathe(bm, [(r - 0.012, -depth / 2), (r, -depth / 2), (r, depth / 2), (r - 0.012, depth / 2)], body, loop=True)
        for z in (-depth / 2, depth / 2):
            out = math.copysign(0.0016, z)
            for ring in (0.05, 0.10, r - 0.025):
                lathe(bm, [(ring - 0.0018, z), (ring + 0.0018, z), (ring + 0.0018, z + out), (ring - 0.0018, z + out)],
                      grille, segments=24, loop=True)
            for k in range(12):
                a = k * math.pi / 6
                rod(bm, (0.012 * math.cos(a), 0.012 * math.sin(a), z),
                    ((r - 0.012) * math.cos(a), (r - 0.012) * math.sin(a), z), 0.0018, grille, segments=6)
        lathe(bm, [(0, -0.022), (0.03, -0.022), (0.03, 0.022), (0, 0.022)], blade, segments=24)
        for k in range(5):
            a = k * 2 * math.pi / 5
            placed(bm, Matrix.Rotation(a, 4, "Z") @ Matrix.Translation((0.095, 0, 0)) @ Matrix.Rotation(0.5, 4, "X"),
                   lambda: slab(bm, -0.062, 0.062, -0.03, 0.03, -0.0025, 0.0025, blade, bevel=0.0015, segments=1))

    _along_x(bm, 0.0, 0.0, zc, head)


def air_purifier(bm, box, slot) -> None:
    """A tower purifier: a rounded square column on a dark plinth, a vented top, a louvred front and a small display."""
    body, vent = slot["body"], slot["vent"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx + 0.012, hx - 0.012, -hy + 0.012, hy - 0.012, 0, 0.012, vent)
    slab(bm, -hx, hx - 0.004, -hy, hy, 0.012, h, body, bevel=0.03, segments=4)
    solid(bm, -0.09, 0.09, -0.09, 0.09, h - 0.0025, h, vent, bevel=0.002, segments=1)  # the top vent
    for z in [0.32 + 0.022 * i for i in range(8)]:  # the louvres of the front stand 4 mm proud of it
        solid(bm, hx - 0.0045, hx, -0.075, 0.075, z, z + 0.008, vent, bevel=0.0012, segments=1)
    solid(bm, hx - 0.0045, hx, -0.04, 0.04, 0.5, 0.52, vent, bevel=0.001, segments=1)  # the display


def space_heater(bm, box, slot) -> None:
    """A ceramic tower heater: a base, a rounded column with a dark front vent crossed by fins, and a control knob."""
    body, vent = slot["body"], slot["vent"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx, -hy, hy, 0, 0.03, vent, bevel=0.01, segments=2)
    slab(bm, -hx + 0.01, hx - 0.01, -hy + 0.015, hy - 0.015, 0.03, h, body, bevel=0.02, segments=3)
    solid(bm, hx - 0.0125, hx - 0.0105, -0.07, 0.07, 0.1, 0.46, vent)
    for i in range(15):
        z = 0.104 + 0.0245 * i
        solid(bm, hx - 0.0125, hx - 0.006, -0.072, 0.072, z, z + 0.007, body, bevel=0.001, segments=1)
    _along_x(bm, hx - 0.0135, 0.07, 0.5, lambda: lathe(bm, [(0, 0), (0.014, 0), (0.014, 0.012), (0, 0.012)], vent,
                                                       segments=20))


def bathroom_scale(bm, box, slot) -> None:
    """A bathroom scale: a bevelled slab with a platform a few millimetres up on its top (one colour, the attribute)
    and a dark display window set in it."""
    body, glass = slot["body"], slot["glass"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx, -hy, hy, 0, h - 0.004, body, bevel=0.01, segments=3)
    solid(bm, -hx + 0.006, hx - 0.006, -hy + 0.006, hy - 0.006, h - 0.004, h, body, bevel=0.003, segments=2)
    solid(bm, 0.05, 0.13, -0.06, 0.06, h - 0.0008, h, glass, bevel=0.0004, segments=1)


def umbrella(bm, box, slot) -> None:
    """A closed umbrella lying on the floor: a pleated canopy wound round the shaft, a strap, a ferrule at one end and a
    rounded handle at the other."""
    canopy, metal, handle = slot["canopy"], slot["metal"], slot["handle"]
    hx, r = box[0] / 2, box[1] / 2
    rod(bm, (-hx, 0, r), (hx, 0, r), 0.005, metal, segments=8)
    rc = r - 0.002
    taper = [(rc * (0.35 + 0.65 * math.sin(math.pi / 2 * (i + 1) / 4)), -0.37 + 0.05 * (i + 1)) for i in range(4)]
    pleats = [(0, -0.37), *taper, (rc, 0.12), (0.8 * rc, 0.24), (0.3 * rc, 0.31), (0, 0.32)]
    _along_x(bm, 0.0, 0.0, r, lambda: lathe(bm, pleats, canopy, segments=24))
    _along_x(bm, 0.0, 0.0, r, lambda: lathe(bm, [(r - 0.0025, -0.04), (r, -0.04), (r, -0.02), (r - 0.0025, -0.02)],
                                            handle, segments=24, loop=True))
    rod(bm, (hx - 0.12, 0, r), (hx, 0, r), 0.0165, handle, segments=16)
    rod(bm, (-hx, 0, r), (-hx + 0.03, 0, r), 0.007, metal, segments=8)


def yoga_mat(bm, box, slot) -> None:
    """A rolled yoga mat lying down: the roll with its spiral end and two straps."""
    mat, strap = slot["mat"], slot["strap"]
    hx, r = box[0] / 2, box[1] / 2
    rr = r - 0.0015

    def roll() -> None:
        lathe(bm, [(0, -hx), (rr, -hx), (rr, hx), (0, hx)], mat, segments=36)
        for z in (-hx - 0.0004, hx + 0.0004):
            for k in range(1, 5):  # the spiral's turns, drawn as rings on each end
                lathe(bm, [(k * 0.015, z), (k * 0.015 + 0.0012, z)], mat, segments=36)
        for z in (-0.17, 0.17):
            lathe(bm, [(rr, z - 0.012), (r, z - 0.012), (r, z + 0.012), (rr, z + 0.012)], strap, segments=36, loop=True)

    _along_x(bm, 0.0, 0.0, r, roll)


def dumbbells(bm, box, slot) -> None:
    """A pair of hex dumbbells lying side by side: a knurled handle between two six-sided heads."""
    iron, handle = slot["iron"], slot["handle"]
    hx, h = box[0] / 2, box[2]
    rh = h / (2 * math.cos(math.pi / 6))  # a flat down: the head is the box high
    for s in (-1.0, 1.0):
        y = s * (box[1] / 2 - rh)
        rod(bm, (-hx + 0.04, y, h / 2), (hx - 0.04, y, h / 2), 0.0165, handle, segments=16)
        for e in (-1.0, 1.0):
            x0, x1 = (hx - 0.07, hx) if e > 0 else (-hx, -hx + 0.07)
            _along_x(bm, 0.0, y, h / 2, lambda x0=x0, x1=x1: lathe(bm, [(0, x0), (rh, x0), (rh, x1), (0, x1)], iron,
                                                                   segments=6), spin=math.pi / 6)


def extension_cord(bm, box, slot) -> None:
    """A cord lying in loose S-bends between a plug and a block with sockets."""
    cable, plug = slot["cable"], slot["plug"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    r = 0.0065
    amp = hy - r
    path = [(-hx + 0.055 + (box[0] - 0.115) * i / 24, amp * math.sin(2 * math.pi * 1.5 * i / 24 + math.pi / 2 * 0),
             r) for i in range(25)]
    sweep(bm, path, r, cable, segments=8)
    solid(bm, -hx, -hx + 0.06, -0.016, 0.016, 0, 0.026, plug, bevel=0.004, segments=2)
    for y in (-0.007, 0.007):
        rod(bm, (-hx + 0.06, y, 0.013), (-hx + 0.075, y, 0.013), 0.0025, plug, segments=6)
    solid(bm, hx - 0.06, hx, -0.025, 0.025, 0, h, plug, bevel=0.005, segments=2)


def power_strip(bm, box, slot) -> None:
    """A power strip: a long housing with five sockets and a lit rocker switch, and a cable that leaves one end, makes a
    loose loop on the floor and ends in a plug."""
    body, face, cable = slot["body"], slot["face"], slot["cable"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    x1 = hx - 0.20
    solid(bm, -hx, x1, -0.027, 0.027, 0, h - 0.004, body, bevel=0.006, segments=2)
    for i in range(5):
        x = -hx + 0.035 + 0.047 * i
        solid(bm, x - 0.015, x + 0.015, -0.017, 0.017, h - 0.0045, h - 0.0035, face)
    solid(bm, x1 - 0.045, x1 - 0.015, -0.015, 0.015, h - 0.004, h, face, bevel=0.002, segments=1)
    r = 0.006
    x2, amp = hx - 0.035, hy - r  # the cable swings once to each side between the housing and the plug
    sweep(bm, [(x1 + (x2 - x1) * i / 32, amp * math.sin(2 * math.pi * i / 32), r) for i in range(33)], r, cable,
          segments=8)
    solid(bm, x2, hx, -0.015, 0.015, 0, 0.026, cable, bevel=0.004, segments=2)


def cable_bundle(bm, box, slot) -> None:
    """A coil of cable: four loops stacked and slightly offset, held by a tie."""
    cable, tie = slot["cable"], slot["tie"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    r = 0.0045
    for k in range(4):
        sx, sy = (hx - r) * (1 - 0.06 * k), (hy - r) * (1 - 0.07 * k)
        z = r + 0.0075 * k
        loop = [(sx * math.cos(2 * math.pi * i / 24), sy * math.sin(2 * math.pi * i / 24), z) for i in range(24)]
        sweep(bm, loop, r, cable, segments=8, closed=True)
    solid(bm, hx - 0.026, hx - 0.018, -0.011, 0.011, 0, h, tie, bevel=0.002, segments=1)


def door_stop(bm, box, slot) -> None:
    """A rubber wedge with grip ridges on its slope."""
    body = slot["body"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    prism(bm, [(-hx, 0.0), (hx, 0.0), (hx, h)], -hy, hy, body)
    for i in range(5):
        x = -hx + 0.012 + 0.017 * i
        z = h * (x + hx) / (2 * hx)
        solid(bm, x - 0.002, x + 0.002, -hy + 0.004, hy - 0.004, z, z + 0.0012, body)


def floor_lamp(bm, box, slot) -> None:
    """A floor lamp: a round weighted base, a slim pole, three spokes and a tapered fabric shade."""
    body, shade = slot["body"], slot["shade"]
    r, h = box[0] / 2, box[2]
    z0, z1 = h - 0.30, h
    lathe(bm, [(0, 0), (0.12, 0), (0.125, 0.008), (0.125, 0.016), (0.05, 0.026), (0, 0.026)], body)
    rod(bm, (0, 0, 0.026), (0, 0, z0 + 0.14), 0.011, body, segments=16)
    lathe(bm, [(r, z0), (r - 0.0025, z0), (0.1175, z1), (0.12, z1)], shade, segments=48, loop=True)
    for k in range(3):
        a = 2 * math.pi * k / 3
        rod(bm, (0, 0, z0 + 0.14), ((r - 0.004) * math.cos(a), (r - 0.004) * math.sin(a), z0 + 0.004), 0.0025, body,
            segments=6)


def router(bm, box, slot) -> None:
    """A wireless router: a flat housing with a vented top and a row of indicator lights on its front edge, and three
    antennas standing on swivel joints along the back, the tallest at the box's top."""
    body, antenna, face = slot["body"], slot["antenna"], slot["face"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx, -hy, hy, 0, 0.032, body, bevel=0.007, segments=3)
    for i in range(7):
        y = -0.06 + 0.02 * i
        solid(bm, -0.03, 0.045, y - 0.003, y + 0.003, 0.0318, 0.0328, face)
    for i in range(4):
        solid(bm, hx - 0.0035, hx - 0.0005, -0.05 + 0.025 * i - 0.004, -0.05 + 0.025 * i + 0.004, 0.012, 0.018, face)
    for y in (-0.085, 0.0, 0.085):
        lathe(bm, [(0, 0.03), (0.011, 0.03), (0.011, 0.044), (0, 0.044)], antenna, segments=16, offset=(-hx + 0.02, y))
        rod(bm, (-hx + 0.02, y, 0.044), (-hx + 0.02, y, h), 0.0055, antenna, segments=10)


def _shelved_book(bm, box, slot) -> None:
    """A book standing spine-out: two boards, a rounded spine on the +x face and a page block between them, its edges
    inset a few millimetres."""
    cover, pages = slot["cover"], slot["pages"]
    hx, hy, h, t = box[0] / 2, box[1] / 2, box[2], 0.0016
    for y0, y1 in ((-hy, -hy + t), (hy - t, hy)):
        slab(bm, -hx, hx - 0.002, y0, y1, 0, h, cover, bevel=0.0008, segments=1)
    slab(bm, hx - 0.006, hx, -hy, hy, 0, h, cover, bevel=0.004, segments=3)
    slab(bm, -hx + 0.003, hx - 0.006, -hy + t, hy - t, 0.003, h - 0.003, pages)


BUILDERS = {"floor_fan": floor_fan, "air_purifier": air_purifier, "space_heater": space_heater,
            "bathroom_scale": bathroom_scale, "umbrella": umbrella, "yoga_mat": yoga_mat, "dumbbells": dumbbells,
            "extension_cord": extension_cord, "power_strip": power_strip, "cable_bundle": cable_bundle,
            "door_stop": door_stop, "floor_lamp": floor_lamp, "router": router, "shelved_book_s": _shelved_book,
            "shelved_book_m": _shelved_book, "shelved_book_l": _shelved_book}
