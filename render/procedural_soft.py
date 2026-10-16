# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The flat and soft things a lived-in home leaves about, as procedural clutter: a pile of letters, a placemat, a
jacket, a heap of clothes on the floor, a stack of folded clothes, a bath mat and a contour mat for a toilet, a boot
tray, a notepad, and the papers on a fridge door (a sheet of print, a calendar, a photo).

Runs inside Blender's own Python and is never imported by `stbench`. Like `procedural_clutter.py`, a builder takes the
bmesh, its `DESIGN_BOX` (metres) and `slot` and adds geometry with its origin at the base centre, x the depth and y the
width; each reaches its box to a millimetre (a test keeps it so). The paper on a fridge is drawn standing on its edge,
its face towards +x (the box's depth is the sheet's thickness). Nothing here is a tracked kind: clothes are beige,
black, brown, purple or pink (no tracked towel is any of those), mats the same, and a pile of letters is white or
beige (the standing books of a shelf are the only paper-coloured clutter that declares itself a book).
"""

from __future__ import annotations

import math

from mathutils import Matrix
from meshkit import placed, rod, slab, solid
from partskit import sweep
from procedural_toys import ellipsoid
from wobble import wave

DESIGN_BOX = {
    "mail_pile": (0.162, 0.207, 0.03), "placemat": (0.32, 0.44, 0.006), "jacket": (0.36, 0.493, 0.132),
    "clothes_pile": (0.30, 0.40, 0.137), "folded_clothes": (0.248, 0.289, 0.099), "bath_mat": (0.40, 0.60, 0.015),
    "pedestal_mat": (0.40, 0.50, 0.012), "boot_tray": (0.3, 0.4, 0.035), "book_stack": (0.186, 0.255, 0.075),
    "notepad": (0.21, 0.30, 0.02), "fridge_sheet": (0.004, 0.215, 0.28), "fridge_calendar": (0.004, 0.25, 0.32),
    "fridge_photo": (0.004, 0.12, 0.16)}
SLOTS = {
    "mail_pile": ("paper", "ink"), "placemat": ("mat",), "jacket": ("cloth",), "clothes_pile": ("cloth",),
    "folded_clothes": ("cloth", "edge"), "bath_mat": ("mat",), "pedestal_mat": ("mat",),
    "boot_tray": ("tray", "grid"), "book_stack": ("cover", "pages"), "notepad": ("paper", "binding"),
    "fridge_sheet": ("paper", "ink"), "fridge_calendar": ("paper", "ink"), "fridge_photo": ("paper", "ink")}


def unturned(hx: float, hy: float, deg: float) -> tuple[float, float]:
    """The half sides of a rectangle that, turned by `deg`, has the half sides `hx` and `hy` for its bounding box."""
    c, s = math.cos(math.radians(deg)), abs(math.sin(math.radians(deg)))
    k = c * c - s * s
    return (hx * c - hy * s) / k, (hy * c - hx * s) / k


def mail_pile(bm, box, slot) -> None:
    """Five envelopes in a loose pile, each a little turned from the one below, with a window and an address on the top
    one."""
    paper, ink = slot["paper"], slot["ink"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    t = (h - 0.0004) / 5
    for i in range(5):  # the lowest is as turned as the pile is wide, the rest lie inside it
        deg, (mx, my) = (-9, 5, -3, 8, -5)[i], ((0.0, 0.0), (0.008, 0.012), (0.004, 0.006), (0.012, 0.004),
                                                 (0.006, 0.01))[i]
        d, w = unturned(hx - mx, hy - my, deg)

        def envelope(i=i, w=w, d=d) -> None:
            solid(bm, -d, d, -w, w, i * t + 0.0002, (i + 1) * t, paper, bevel=0.0008, segments=1)

        placed(bm, Matrix.Rotation(math.radians(deg), 4, "Z"), envelope)
    solid(bm, -0.03, 0.03, 0.05, 0.09, h - 0.0008, h, ink)  # a stamp
    for i in range(4):  # and the address, in lines
        solid(bm, -0.05 + 0.014 * i, -0.046 + 0.014 * i, -0.08, 0.02 - 0.012 * (i % 2), h - 0.0008, h, ink)


def placemat(bm, box, slot) -> None:
    """A woven placemat: a thin slab with eased edges and a raised border."""
    mat = slot["mat"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    slab(bm, -hx, hx, -hy, hy, 0.0, h * 0.7, mat, bevel=0.0015, segments=1)
    solid(bm, -hx, hx, -hy, -hy + 0.014, 0.0, h, mat, bevel=0.0012, segments=1)
    solid(bm, -hx, hx, hy - 0.014, hy, 0.0, h, mat, bevel=0.0012, segments=1)


def jacket(bm, box, slot) -> None:
    """A jacket thrown down: a flattened torso, a rolled collar, a sleeve folded across the front and another trailing
    off one side, and a zip down the middle."""
    cloth = slot["cloth"]
    hx = box[0] / 2
    ellipsoid(bm, (0.0, 0.0, 0.052), (hx, 0.2, 0.052), cloth, segments=28)
    ellipsoid(bm, (0.04, 0.0, 0.066), (0.12, 0.17, 0.052), cloth, segments=24)  # the chest, a second fold on the torso
    ellipsoid(bm, (-hx + 0.05, 0.0, 0.056), (0.05, 0.12, 0.052), cloth, segments=24)  # the collar
    sweep(bm, [(0.05, 0.02, 0.092), (0.09, 0.09, 0.092), (0.12, 0.16, 0.084), (0.09, 0.222, 0.05)], 0.0, cloth,
          segments=12, radii=[0.04, 0.038, 0.036, 0.032])  # a sleeve folded across the front
    sweep(bm, [(-0.04, -0.02, 0.092), (-0.07, -0.1, 0.088), (-0.02, -0.17, 0.07), (0.04, -0.23, 0.042)], 0.0, cloth,
          segments=12, radii=[0.04, 0.038, 0.036, 0.032])  # and the other trailing off the side


def clothes_pile(bm, box, slot) -> None:
    """A heap of clothes on the floor: five soft lumps of different sizes piled against each other, a sleeve out of one
    side."""
    cloth = slot["cloth"]
    lumps = ((0.0, 0.0, 0.05, 0.12, 0.16, 0.05), (0.05, 0.07, 0.075, 0.085, 0.1, 0.05), (-0.06, -0.08, 0.075, 0.09,
             0.095, 0.05), (0.0, -0.02, 0.092, 0.075, 0.085, 0.045), (-0.07, 0.09, 0.05, 0.07, 0.085, 0.045))
    for cx, cy, cz, rx, ry, rz in lumps:
        ellipsoid(bm, (cx, cy, cz), (rx, ry, rz), cloth, Matrix.Rotation(cx * 9 + cy * 7, 4, "Z"), segments=24)
    sweep(bm, [(0.02, -0.13, 0.04), (0.05, -0.17, 0.034), (0.07, -0.19, 0.026)], 0.0, cloth, segments=10,
          radii=[0.034, 0.03, 0.02])
    sweep(bm, [(-0.09, 0.14, 0.032), (-0.11, 0.17, 0.022), (-0.06, 0.188, 0.016)], 0.0, cloth, segments=10,
          radii=[0.03, 0.022, 0.012])


def folded_clothes(bm, box, slot) -> None:
    """A stack of folded clothes: four rounded slabs, a little out of line, each with its fold in a paler edge."""
    cloth, edge = slot["cloth"], slot["edge"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    t = h / 4
    for i in range(4):
        dx, dy = 0.004 * wave(i * 1.3, 0.5), 0.006 * wave(i * 2.1, 1.2)
        solid(bm, -hx + 0.007 + dx - 0.002 * i, hx - 0.007 + dx + 0.002 * i, -hy + 0.0035 + dy, hy - 0.0035 + dy,
              i * t, (i + 1) * t - 0.001, cloth, bevel=0.009, segments=3)
        solid(bm, hx - 0.014 + dx + 0.002 * i, hx - 0.007 + dx + 0.002 * i, -hy + 0.0095 + dy, hy - 0.0095 + dy,
              i * t + 0.002, (i + 1) * t - 0.003, edge, bevel=0.002, segments=1)


def bath_mat(bm, box, slot) -> None:
    """A rectangular bath mat with rounded ends, a raised pile in the middle and a flat border."""
    mat = slot["mat"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    slab(bm, -hx, hx, -hy, hy, 0.0, h * 0.6, mat, bevel=0.006, segments=3)
    slab(bm, -hx + 0.03, hx - 0.03, -hy + 0.03, hy - 0.03, 0.0, h, mat, bevel=0.006, segments=3)


def book_stack(bm, box, slot) -> None:
    """Three hardbacks lying one on another, each a little turned and a little different in size: boards, a rounded
    spine on the -y side and a page block that shows at the other three edges."""
    cover, pages = slot["cover"], slot["pages"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    t = h / 3
    for i, (a, w, d) in enumerate(((-4.0, 1.0, 1.0), (3.0, 0.9, 0.93), (-1.0, 0.8, 0.85))):
        sx, sy = unturned(hx, hy, a)  # the first lies as turned as the stack is wide, the others inside it

        def one(i=i, x=sx * d, y=sy * w) -> None:
            board, spine = 0.0015, 0.007
            z0, z1 = i * t + 0.0002, (i + 1) * t - 0.0002
            slab(bm, -x, x, -y, y, z0, z0 + board, cover, bevel=0.0008, segments=1)
            slab(bm, -x, x, -y, y, z1 - board, z1, cover, bevel=0.0008, segments=1)
            slab(bm, -x, x, -y, -y + spine, z0, z1, cover, bevel=0.003, segments=2)
            slab(bm, -x + 0.003, x - 0.003, -y + spine, y - 0.003, z0 + board, z1 - board, pages)

        placed(bm, Matrix.Rotation(math.radians(a), 4, "Z"), one)


def pedestal_mat(bm, box, slot) -> None:
    """A contour mat for a toilet's pedestal: a thick rim round a bite out of the side that faces the toilet (-x)."""
    mat = slot["mat"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx + 0.1, hx, -hy, hy, 0.0, h, mat, bevel=0.005, segments=3)  # the body, its back at the toilet
    for s in (1.0, -1.0):  # and two arms that reach round the pedestal
        solid(bm, -hx, -hx + 0.14, min(s * 0.14, s * hy), max(s * 0.14, s * hy), 0.0, h, mat, bevel=0.005, segments=3)


def boot_tray(bm, box, slot) -> None:
    """A rubber boot tray: a shallow tray with a raised lip and a grid of ribs across its floor."""
    tray, grid = slot["tray"], slot["grid"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx, hx, -hy, hy, 0.0, 0.008, tray, bevel=0.003, segments=2)
    for x0, x1, y0, y1 in ((-hx, hx, -hy, -hy + 0.018), (-hx, hx, hy - 0.018, hy), (-hx, -hx + 0.018, -hy, hy),
                           (hx - 0.018, hx, -hy, hy)):
        solid(bm, x0, x1, y0, y1, 0.0, h, tray, bevel=0.004, segments=2)  # the lip
    for i in range(9):
        y = -hy + 0.04 + i * (2 * hy - 0.08) / 8
        solid(bm, -hx + 0.03, hx - 0.03, y - 0.005, y + 0.005, 0.006, 0.014, grid, bevel=0.002, segments=1)


def notepad(bm, box, slot) -> None:
    """A spiral-bound notepad lying open at its first page: a stiff back under a stack of leaves, and a row of wire
    loops along the top edge (-x)."""
    paper, binding = slot["paper"], slot["binding"]
    hx, hy, h = box[0] / 2, box[1] / 2, box[2]
    solid(bm, -hx + 0.012, hx, -hy, hy, 0.0, 0.004, binding, bevel=0.0008, segments=1)
    solid(bm, -hx + 0.012, hx, -hy, hy, 0.004, h, paper, bevel=0.0008, segments=1)
    for i in range(16):
        y = -hy + 0.015 + i * (2 * hy - 0.03) / 15
        rod(bm, (-hx + 0.0, y, h * 0.5), (-hx + 0.014, y, h * 0.5), 0.0028, binding, segments=8)


def _sheet(bm, box, slot, ink_rows: tuple) -> None:
    """A sheet stood on its edge facing +x: the paper's slab and `ink_rows` ((y0, y1, z0, z1) fractions) of print."""
    paper, ink = slot["paper"], slot["ink"]
    t, w, h = box
    solid(bm, -t / 2, t / 2 - 0.0008, -w / 2, w / 2, 0.0, h, paper, bevel=0.0003, segments=1)
    for y0, y1, z0, z1 in ink_rows:
        solid(bm, t / 2 - 0.0008, t / 2, (y0 - 0.5) * w, (y1 - 0.5) * w, z0 * h, z1 * h, ink)


def fridge_sheet(bm, box, slot) -> None:
    """A printed page held to the door: a heading and lines of text."""
    rows = [(0.1, 0.7, 0.88, 0.93)]
    rows += [(0.1, 0.9 - 0.12 * (i % 3), 0.78 - 0.065 * i, 0.805 - 0.065 * i) for i in range(10)]
    _sheet(bm, box, slot, tuple(rows))


def fridge_calendar(bm, box, slot) -> None:
    """A wall calendar: a band for the month and a grid of days."""
    rows = [(0.06, 0.94, 0.84, 0.96)]
    rows += [(0.06 + 0.88 * c / 7 + 0.004, 0.06 + 0.88 * (c + 1) / 7 - 0.004, 0.06 + 0.1 * r, 0.14 + 0.1 * r)
             for c in range(7) for r in range(6) if (c + r) % 3 == 0]
    _sheet(bm, box, slot, tuple(rows))


def fridge_photo(bm, box, slot) -> None:
    """A photograph or a child's drawing: a picture with a white margin."""
    _sheet(bm, box, slot, ((0.1, 0.9, 0.2, 0.9), ))


BUILDERS = {"mail_pile": mail_pile, "placemat": placemat, "jacket": jacket, "clothes_pile": clothes_pile,
            "folded_clothes": folded_clothes, "bath_mat": bath_mat, "notepad": notepad, "fridge_sheet": fridge_sheet,
            "fridge_calendar": fridge_calendar, "fridge_photo": fridge_photo, "pedestal_mat": pedestal_mat,
            "boot_tray": boot_tray, "book_stack": book_stack}
