# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What a kitchen's counters and stove hold that no question names, as procedural clutter: a microwave, a stand
mixer, a blender, two storage canisters, a paper towel holder, a soap pump, a dish rack, a stock pot, a frying pan and a
pedal bin.

Runs inside Blender's own Python and is never imported by `stbench`. Like `procedural_clutter.py`, a builder takes the
bmesh, its `DESIGN_BOX` (metres) and `slot` (a slot name to its material index) and adds geometry with its origin at the
base centre, x the depth (front to back) and y the width; every one reaches its box to a millimetre (a test keeps it
so). Nothing here looks like a tracked kind: a dish rack holds grey plates and no tracked plate is grey; a soap pump is
white, yellow, orange or pink, and no tracked bottle is any of those; a pot is steel, black or red and has a lid.
"""

from __future__ import annotations

from meshkit import lathe, rod, slab, solid
from partskit import sweep

DESIGN_BOX = {
    "microwave": (0.36, 0.46, 0.27), "stand_mixer": (0.22, 0.36, 0.36), "blender": (0.17, 0.17, 0.40),
    "canister_tall": (0.12, 0.12, 0.22), "canister_short": (0.11, 0.11, 0.17), "paper_towel": (0.15, 0.15, 0.30),
    "soap_pump": (0.0625, 0.06, 0.20), "dish_rack": (0.33, 0.42, 0.165), "stock_pot": (0.271, 0.327, 0.19),
    "frying_pan": (0.45, 0.28, 0.069), "pedal_bin": (0.266, 0.266, 0.42), "bath_bin": (0.2, 0.2, 0.3)}
SLOTS = {
    "microwave": ("body", "glass", "trim"), "stand_mixer": ("body", "bowl", "trim"), "blender": ("base", "jug", "lid"),
    "canister_tall": ("body", "lid"), "canister_short": ("body", "lid"), "paper_towel": ("holder", "roll"),
    "soap_pump": ("body", "pump"), "dish_rack": ("tray", "wire", "dish"), "stock_pot": ("pot", "handle"),
    "frying_pan": ("pan", "handle"), "pedal_bin": ("body", "lid", "pedal"), "bath_bin": ("body", "lid", "pedal")}


def _disc(bm, r: float, z0: float, z1: float, slot: int, segments: int = 32, offset=(0.0, 0.0)) -> None:
    lathe(bm, [(0, z0), (r, z0), (r, z1), (0, z1)], slot, segments=segments, offset=offset)


def microwave(bm, box, slot) -> None:
    """A countertop microwave: a rounded body, a door on its front (+x) with a dark window and a bar handle, a control
    panel beside it with a dial and a row of keys."""
    body, glass, trim = slot["body"], slot["glass"], slot["trim"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx - 0.004, -hy, hy, 0.012, h, body, bevel=0.012, segments=3)
    for x in (-hx + 0.03, hx - 0.05):  # two feet
        for y in (-hy + 0.04, hy - 0.04):
            rod(bm, (x, y, 0.0), (x, y, 0.014), 0.011, trim, segments=10)
    solid(bm, hx - 0.006, hx, -hy + 0.02, hy - 0.115, 0.022, h - 0.02, trim, bevel=0.003, segments=2)  # the door frame
    solid(bm, hx - 0.002, hx + 0.0, -hy + 0.045, hy - 0.14, 0.05, h - 0.045, glass)  # the window
    rod(bm, (hx - 0.0085, hy - 0.125, 0.05), (hx - 0.0085, hy - 0.125, h - 0.05), 0.0075, trim, segments=10)  # handle
    solid(bm, hx - 0.004, hx, hy - 0.103, hy - 0.012, 0.03, h - 0.03, trim, bevel=0.002, segments=1)  # the panel
    rod(bm, (hx - 0.014, hy - 0.057, h - 0.07), (hx, hy - 0.057, h - 0.07), 0.016, body, segments=20)  # the dial
    for i in range(5):  # keys
        solid(bm, hx - 0.0065, hx - 0.001, hy - 0.095 + 0.016 * i, hy - 0.085 + 0.016 * i, 0.04, 0.05, body)


def stand_mixer(bm, box, slot) -> None:
    """A tilt-head stand mixer: a base plate, a column, a rounded head reaching along y over a steel bowl and a
    beater."""
    body, bowl, trim = slot["body"], slot["bowl"], slot["trim"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx, -hy, hy, 0.0, 0.035, body, bevel=0.012, segments=3)
    solid(bm, -hx + 0.02, hx - 0.02, hy - 0.115, hy - 0.025, 0.03, h - 0.07, body, bevel=0.02, segments=3)
    slab(bm, -hx + 0.003, hx - 0.003, -hy + 0.01, hy - 0.02, h - 0.12, h, body, bevel=0.04, segments=4)  # the head
    rod(bm, (0.0, -hy + 0.115, h - 0.17), (0.0, -hy + 0.115, h - 0.12), 0.006, trim, segments=10)  # the beater's shaft
    lathe(bm, [(0, h - 0.17), (0.02, h - 0.165), (0.03, h - 0.12), (0.0, h - 0.1)], trim, segments=16,
          offset=(0.0, -hy + 0.115))
    lathe(bm, [(0, 0.035), (0.045, 0.035), (0.06, 0.05), (0.1, 0.125), (0.105, 0.17), (0.095, 0.172), (0.09, 0.125),
               (0.055, 0.052), (0, 0.047)], bowl, segments=32, offset=(0.0, -hy + 0.115))


def blender(bm, box, slot) -> None:
    """A jug blender: a base with a dial, a tall tapered jug with a handle and a lid with a cap."""
    base, jug, lid = slot["base"], slot["jug"], slot["lid"]
    r, h = box[0] / 2, box[2]
    lathe(bm, [(0, 0), (0.95 * r, 0), (r, 0.012), (r, 0.1), (0.8 * r, 0.12), (0, 0.12)], base, segments=32)
    lathe(bm, [(0.46 * r, 0.12), (0.84 * r, 0.125), (r * 0.99, 0.2), (r, h - 0.06), (0.94 * r, h - 0.045),
               (0.94 * r, h - 0.012), (0.9 * r, h - 0.012)], jug, segments=32)
    _disc(bm, 0.96 * r, h - 0.03, h - 0.012, lid, 32)
    _disc(bm, 0.4 * r, h - 0.012, h, lid, 24)
    sweep(bm, [(0.0, r * 0.84, h - 0.075), (0.0, r * 0.9, h - 0.075), (0.0, r * 0.9, 0.2), (0.0, r * 0.84, 0.2)], 0.0,
          jug, segments=8, radii=[0.0055, 0.0055, 0.0055, 0.0055])


def _canister(bm, box, slot, flare: float) -> None:
    body, lid = slot["body"], slot["lid"]
    r, h = box[0] / 2, box[2]
    lathe(bm, [(0, 0), (0.92 * r, 0), (r, 0.01), (r, h - 0.03 - 0.01), (0.9 * r * flare, h - 0.03)], body, segments=36)
    lathe(bm, [(0.9 * r * flare, h - 0.03), (0.98 * r, h - 0.028), (0.98 * r, h - 0.012), (0.3 * r, h - 0.004),
               (0.3 * r, h), (0, h)], lid, segments=36)


def canister_tall(bm, box, slot) -> None:
    """A tall ceramic storage canister with a flat lid and a knob."""
    _canister(bm, box, slot, 1.0)


def canister_short(bm, box, slot) -> None:
    """A short, wider storage canister."""
    _canister(bm, box, slot, 1.0)


def paper_towel(bm, box, slot) -> None:
    """A paper towel holder: a round foot, a pole and a roll of towel on it, a little loose at the end."""
    holder, roll = slot["holder"], slot["roll"]
    r, h = box[0] / 2, box[2]
    lathe(bm, [(0, 0), (r, 0), (r, 0.012), (0.2 * r, 0.02), (0, 0.02)], holder, segments=36)
    rod(bm, (0, 0, 0.02), (0, 0, h - 0.003), 0.0075, holder, segments=12)
    lathe(bm, [(0.014, 0.025), (0.12 * 0.5, 0.025), (0.12 * 0.5, h - 0.045), (0.014, h - 0.045)], roll, segments=36,
          loop=True)
    lathe(bm, [(0.06, 0.025), (0.0615, 0.025), (0.0615, h - 0.045), (0.06, h - 0.045)], roll, segments=36)
    lathe(bm, [(0, h - 0.012), (0.011, h - 0.012), (0.011, h), (0, h)], holder, segments=16)


def soap_pump(bm, box, slot) -> None:
    """A pump bottle of liquid soap: a rounded bottle, a collar, a pump with a spout to +x."""
    body, pump = slot["body"], slot["pump"]
    hy, h = box[1] / 2, box[2]
    placed_r = hy
    lathe(bm, [(0, 0), (0.85 * placed_r, 0), (placed_r, 0.012), (placed_r, 0.115), (0.8 * placed_r, 0.14),
               (0.45 * placed_r, 0.15), (0, 0.15)], body, segments=28)
    rod(bm, (0, 0, 0.15), (0, 0, 0.17), 0.012, pump, segments=14)
    rod(bm, (0, 0, 0.17), (0, 0, h - 0.012), 0.005, pump, segments=10)
    solid(bm, -0.012, box[0] - hy, -0.0085, 0.0085, h - 0.024, h, pump, bevel=0.003, segments=2)  # the spout


def dish_rack(bm, box, slot) -> None:
    """A draining board with a wire rack on it: a shallow tray with a lip, a row of wires holding three grey plates on
    edge, and a cutlery cup at one end."""
    tray, wire, dish = slot["tray"], slot["wire"], slot["dish"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx, -hy, hy, 0.0, 0.02, tray, bevel=0.006, segments=2)
    for sx in (-1.0, 1.0):  # the rack's side rails
        rod(bm, (sx * (hx - 0.03), -hy + 0.04, 0.045), (sx * (hx - 0.03), hy - 0.15, 0.045), 0.003, wire, segments=8)
    top = h - 0.003  # the dividers' crossbars are the box's top
    for y in [-hy + 0.045 + i * 0.045 for i in range(7)]:  # and the dividers that stand across them
        rod(bm, (-hx + 0.03, y, 0.03), (-hx + 0.03, y, top), 0.0028, wire, segments=8)
        rod(bm, (hx - 0.03, y, 0.03), (hx - 0.03, y, top), 0.0028, wire, segments=8)
        rod(bm, (-hx + 0.03, y, top), (hx - 0.03, y, top), 0.0028, wire, segments=8)
    for y in (-hy + 0.09, -hy + 0.18, -hy + 0.27):  # a side plate on edge in each of three slots
        rod(bm, (-0.01, y - 0.004, 0.098), (-0.01, y + 0.004, 0.098), 0.066, dish, segments=32)
    lathe(bm, [(0, 0.02), (0.045, 0.02), (0.05, 0.09), (0.0, 0.09)], tray, segments=24, offset=(0.0, hy - 0.06))
    for dx, dz in ((-0.012, 0.115), (0.012, 0.105), (0.0, 0.13)):
        rod(bm, (dx, hy - 0.06 + dx, 0.07), (dx * 1.8, hy - 0.06 + dx, dz), 0.003, wire, segments=6)


def stock_pot(bm, box, slot) -> None:
    """A stock pot with a domed lid, a knob and two loop handles that reach out to the box's width (y)."""
    pot, handle = slot["pot"], slot["handle"]
    r, h = box[0] / 2.04, box[2]  # the lid's lip, 1.02 r, is the pot's width
    lathe(bm, [(0, 0), (0.96 * r, 0), (r, 0.01), (r, h - 0.06), (0.99 * r, h - 0.05)], pot, segments=40)
    lathe(bm, [(0.99 * r, h - 0.05), (1.02 * r, h - 0.045), (1.02 * r, h - 0.036), (0.9 * r, h - 0.034),
               (0.55 * r, h - 0.012), (0.2 * r, h - 0.002), (0.2 * r, h - 0.0), (0, h)], pot, segments=40)
    rod(bm, (0, 0, h - 0.006), (0, 0, h), 0.012, handle, segments=12)
    o = box[1] / 2 - 0.006  # the handles' centrelines reach this far out, and their tubes 6 mm more: the box's width
    for s in (1.0, -1.0):
        sweep(bm, [(0, s * (r - 0.004), h - 0.075), (0, s * (o - 0.004), h - 0.075), (0, s * o, h - 0.095),
                   (0, s * (r + 0.004), h - 0.1)], 0.006, handle, segments=8)


def frying_pan(bm, box, slot) -> None:
    """A frying pan with its long handle towards +x: sloped walls, a flat base and a handle with a hang hole."""
    pan, handle = slot["pan"], slot["handle"]
    r = box[1] / 2
    h = box[2]
    cx = -box[0] / 2 + r
    lathe(bm, [(0, 0), (0.78 * r, 0), (0.84 * r, 0.006), (r, h - 0.006), (r, h), (0.96 * r, h), (0.82 * r, 0.012),
               (0, 0.01)], pan, segments=40, offset=(cx, 0.0))
    reach = box[0] / 2
    sweep(bm, [(cx + r - 0.006, 0, h - 0.016), (reach - 0.05, 0, h - 0.012), (reach, 0, h - 0.009)], 0.0, handle,
          segments=12, radii=[0.011, 0.0085, 0.008])


def pedal_bin(bm, box, slot) -> None:
    """A round pedal bin (the kitchen's, and the bathroom's smaller one): a body that narrows a little at the foot, a
    domed lid with a knob, a pedal at the front (+x) and a seam where the lid meets the body."""
    body, lid, pedal = slot["body"], slot["lid"], slot["pedal"]
    r, h = box[0] / 2.02, box[2]  # the lid's lip, 1.01 r, is the bin's width
    lathe(bm, [(0, 0), (0.86 * r, 0), (0.9 * r, 0.012), (r, 0.05), (r, h - 0.06)], body, segments=40)
    lathe(bm, [(r, h - 0.06), (1.01 * r, h - 0.058), (1.01 * r, h - 0.048), (0.97 * r, h - 0.034), (0.55 * r, h - 0.01),
               (0.1 * r, h - 0.0), (0, h)], lid, segments=40)
    solid(bm, r - 0.018, r + 0.0, -0.03, 0.03, 0.0, 0.025, pedal, bevel=0.005, segments=2)
    rod(bm, (r - 0.012, 0.0, 0.02), (r - 0.012, 0.0, h - 0.17), 0.0035, pedal, segments=8)


BUILDERS = {"microwave": microwave, "stand_mixer": stand_mixer, "blender": blender, "canister_tall": canister_tall,
            "canister_short": canister_short, "paper_towel": paper_towel, "soap_pump": soap_pump,
            "dish_rack": dish_rack, "stock_pot": stock_pot, "frying_pan": frying_pan, "pedal_bin": pedal_bin,
            "bath_bin": pedal_bin}
