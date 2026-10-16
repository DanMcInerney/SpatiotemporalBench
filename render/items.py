# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Carried items and movers as compositions of primitives, one builder per catalog kind (M1 placeholders).

A builder takes the entity's bounding box (x, y, z), already scaled by its size attribute, and its colour attribute,
and returns parts in the entity's local frame (origin at the base centre, x forward, y left, z up). Parts are drawn
in fractions of the box, so every size variant is the same shape scaled. The colour attribute shows on the part a
person would call "the red one" (the mug body, the book cover, the toy blocks); other parts use neutral materials.
No transparent parts: a "clear" glass or bottle is an opaque pale blue-grey.
"""

from __future__ import annotations

import math

from shapes import Part, ball, box, cyl, ring, wedge


def clutter_box(s, c):
    """A clutter entity in the placeholder appearance: a box of its size in its colour (clutter has dozens of kinds, and
    a world's look never rests on the placeholder, so none has a builder of its own)."""
    sx, sy, sz = s
    return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0, sz, c)]


def mug(s, c):
    sx, sy, sz = s
    r = 0.36 * min(sx, sy)
    cx = -0.12 * sx
    return [cyl(cx, 0, 0, sz, r, c), cyl(cx, 0, sz * 0.97, sz * 1.0005, r * 0.86, "brown*0.5"),
            ring(cx + r + 0.045 * sx, 0, sz * 0.5, 0.12 * sz, 0.028 * sz, "y", c)]


def glass(s, c):
    sx, sy, sz = s
    r = 0.5 * min(sx, sy)
    return [cyl(0, 0, 0, sz, r * 0.78, c, top=1.28), cyl(0, 0, sz * 0.985, sz * 1.0005, r * 0.9, c + "*0.55"),
            cyl(0, 0, 0, sz * 0.12, r * 0.8, c + "*0.7")]


def bowl(s, c):
    sx, sy, sz = s
    r = 0.5 * min(sx, sy)
    return [cyl(0, 0, 0, sz * 0.25, r * 0.4, c, top=1.0), cyl(0, 0, sz * 0.2, sz, r * 0.5, c, top=2.0),
            cyl(0, 0, sz * 0.97, sz * 1.0005, r * 0.88, c + "*0.55")]


def plate(s, c):
    sx, sy, sz = s
    r = 0.5 * min(sx, sy)
    return [cyl(0, 0, 0, sz * 0.7, r, c), cyl(0, 0, 0, sz, r * 0.62, c + "*0.82")]


def book(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    t = 0.14 * sz
    return [box(-hx, hx, -hy, hy, 0, t, c), box(-hx, hx, -hy, hy, sz - t, sz, c),
            box(-hx, hx, -hy, -hy + 0.07 * sy, 0, sz, c + "*0.8"),
            box(-hx + 0.015 * sx, hx - 0.01 * sx, -hy + 0.07 * sy, hy - 0.015 * sy, t, sz - t, "paper")]


def keys(s, c):
    sx, sy, sz = s
    return [ring(-0.3 * sx, 0, sz / 2, 0.17 * sx, 0.3 * sz, "z", "steel"),
            box(-0.12 * sx, 0.5 * sx, -0.22 * sy, 0.0, 0, sz * 0.5, "steel"),
            box(-0.12 * sx, 0.38 * sx, 0.02 * sy, 0.3 * sy, 0, sz * 0.5, "steel"),
            box(-0.5 * sx, -0.3 * sx, -0.5 * sy, -0.2 * sy, 0, sz, c)]


def phone(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    screen = "dark" if c != "black" else "black*2.5"
    return [box(-hx, hx, -hy, hy, 0, sz, c), box(-hx + 0.05 * sx, hx - 0.05 * sx, -hy + 0.08 * sy, hy - 0.08 * sy, sz,
                                                  sz * 1.15, screen),
            cyl(-hx + 0.1 * sx, hy - 0.2 * sy, sz, sz * 1.3, 0.08 * sy, "dark")]


def remote(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx, -hy, hy, 0, sz, c)]
    for i in range(4):
        x = -hx + 0.18 * sx + i * 0.2 * sx
        parts.append(box(x, x + 0.1 * sx, -0.3 * hy, 0.3 * hy, sz, sz * 1.5, "red" if i == 0 else "grey*1.6"))
    return parts


def glasses(s, c):
    sx, sy, sz = s
    hx = sx / 2
    r = 0.4 * sz
    z = r + 0.1 * sz
    return [ring(-hx + 0.3 * sx, 0, z, r, 0.1 * sz, "y", c), ring(hx - 0.3 * sx, 0, z, r, 0.1 * sz, "y", c),
            box(-0.06 * sx, 0.06 * sx, -0.05 * sy, 0.05 * sy, z + r * 0.4, z + r * 0.4 + 0.2 * sz, c),
            box(-hx, -hx + 0.1 * sx, 0.1 * sy, 0.5 * sy, z - 0.1 * sz, z + 0.1 * sz, c),
            box(hx - 0.1 * sx, hx, 0.1 * sy, 0.5 * sy, z - 0.1 * sz, z + 0.1 * sz, c)]


def bag(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0, sz * 0.74, c), box(-hx + 0.02 * sx, hx - 0.02 * sx, -hy, hy, sz * 0.5, sz * 0.58,
                                                         c + "*0.7"),
            box(-0.25 * sx, -0.2 * sx, -0.3 * hy, 0.3 * hy, sz * 0.74, sz, c + "*0.6"),
            box(0.2 * sx, 0.25 * sx, -0.3 * hy, 0.3 * hy, sz * 0.74, sz, c + "*0.6"),
            box(-0.25 * sx, 0.25 * sx, -0.3 * hy, 0.3 * hy, sz * 0.96, sz, c + "*0.6")]


def shoes(s, c):
    sx, sy, sz = s
    hx = sx / 2
    parts = []
    for y0, y1 in ((-0.5 * sy, -0.04 * sy), (0.04 * sy, 0.5 * sy)):
        parts += [box(-hx, hx, y0, y1, 0, sz * 0.14, "white*0.9"), wedge(-hx, hx, y0, y1, sz * 0.14, sz, sz * 0.4, c)]
    return parts


def laptop(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0, sz * 0.5, "grey*0.55"), box(-hx, hx, -hy, hy, sz * 0.5, sz, c),
            cyl(0, 0, sz * 0.5, sz * 0.502, 0.1 * sy, "steel", top=1.0)]


def plant(s, c):
    sx, sy, sz = s
    r = 0.5 * min(sx, sy)
    return [cyl(0, 0, 0, sz * 0.4, r * 0.8, c, top=1.2), cyl(0, 0, sz * 0.38, sz * 0.405, r * 0.95, "soil"),
            ball(0, 0, sz * 0.62, r * 0.9, r * 0.9, sz * 0.2, "foliage"),
            ball(-r * 0.3, r * 0.2, sz * 0.8, r * 0.6, r * 0.6, sz * 0.18, "foliage*1.2"),
            ball(r * 0.3, -r * 0.2, sz * 0.84, r * 0.55, r * 0.55, sz * 0.16, "foliage*0.85")]


def bottle(s, c):
    sx, sy, sz = s
    r = 0.5 * min(sx, sy)
    return [cyl(0, 0, 0, sz * 0.62, r, c), cyl(0, 0, sz * 0.62, sz * 0.78, r, c, top=0.3),
            cyl(0, 0, sz * 0.78, sz * 0.94, r * 0.3, c), cyl(0, 0, sz * 0.94, sz, r * 0.34, "dark"),
            cyl(0, 0, sz * 0.25, sz * 0.5, r * 1.01, "paper")]


def box_(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0, sz, c), box(-0.06 * sx, 0.06 * sx, -hy, hy, sz, sz * 1.003, "beige*1.2"),
            box(-0.06 * sx, 0.06 * sx, -hy - 0.002, -hy, 0, sz, "beige*1.2"),
            box(-0.06 * sx, 0.06 * sx, hy, hy + 0.002, 0, sz, "beige*1.2")]


def towel(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = []
    for i in range(3):
        d = 0.015 * (i % 2)
        shade = "*0.92" if i % 2 else ""
        parts.append(box(-hx + d, hx - d, -hy + d, hy - d, sz * i / 3, sz * (i + 1) / 3 - 0.001, c + shade))
    parts.append(box(-hx + 0.55 * sx, -hx + 0.65 * sx, -hy, hy, 0, sz * 1.002, c + "*0.6"))
    return parts


def toy(s, c):
    sx, sy, sz = s
    h = sz / 3
    return [box(-0.5 * sx, 0.5 * sx, -0.5 * sy, 0.5 * sy, 0, h, c), box(-0.34 * sx, 0.34 * sx, -0.34 * sy, 0.34 * sy, h,
                                                                        2 * h, c + "*0.8"),
            Part("box", (0.0, 0.0, 2.5 * h), (0.5 * sx, 0.5 * sy, h), (0, 0, math.radians(25)), c + "*1.2"),
            ball(0.0, 0.0, 3.0 * h - 0.2 * h, 0.12 * sx, 0.12 * sy, 0.2 * h, "white")]


def robot_vacuum(s, c):
    sx, sy, sz = s
    r = 0.5 * min(sx, sy)
    return [cyl(0, 0, 0, sz * 0.4, r, "dark"), cyl(0, 0, sz * 0.35, sz * 0.85, r * 0.97, c),
            cyl(-0.12 * r, 0, sz * 0.85, sz, r * 0.28, "dark")]


def home_robot(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -0.8 * hy, 0.8 * hy, 0, sz, c), box(0.2 * hx, hx * 1.0005, -0.6 * hy, 0.6 * hy, sz * 0.62,
                                                              sz * 0.92, "dark")]


def ball_robot(s, c):
    sx, sy, sz = s
    return [ball(0, 0, sz / 2, sx / 2, sy / 2, sz / 2, c), ball(0.44 * sx, 0, sz * 0.62, 0.07 * sx, 0.07 * sy,
                                                               0.07 * sz, "dark")]


def _wheel(cx, cy, cz, r, width, mat) -> Part:
    """A tyre with its axle along y."""
    return Part("cyl", (cx, cy, cz), (2 * r, 2 * r, width), (math.pi / 2, 0, 0), mat, (1.0,))


def pallet(s, c):
    """A wooden pallet under a load wrapped in the colour attribute."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    base = 0.145 * sz
    parts = [box(-hx, hx, -hy, hy, 0.78 * base, base, "oak")]
    parts += [box(x - 0.04 * sx, x + 0.04 * sx, -hy, hy, 0, 0.78 * base, "oak*0.7")
              for x in (-0.46 * sx, 0.0, 0.46 * sx)]
    parts.append(box(-0.49 * sx, 0.49 * sx, -0.49 * sy, 0.49 * sy, base, sz, c))
    return parts


def tote(s, c):
    """An open tub with a dark hand-hold in each short end."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    t = 0.03 * sx
    return [box(-hx, hx, -hy, hy, 0, 0.05 * sz, c), box(-hx, hx, -hy, -hy + t, 0, sz, c),
            box(-hx, hx, hy - t, hy, 0, sz, c), box(-hx, -hx + t, -hy, hy, 0, sz, c),
            box(hx - t, hx, -hy, hy, 0, sz, c),
            box(hx, hx * 1.0005, -0.2 * sy, 0.2 * sy, 0.65 * sz, 0.85 * sz, "dark"),
            box(-hx * 1.0005, -hx, -0.2 * sy, 0.2 * sy, 0.65 * sz, 0.85 * sz, "dark")]


def drum(s, c):
    """A drum with two rolling hoops, a steel lid ring and a lid."""
    sx, sy, sz = s
    r = 0.5 * min(sx, sy)
    return [cyl(0, 0, 0, sz, r * 0.96, c), cyl(0, 0, 0.3 * sz, 0.34 * sz, r, c), cyl(0, 0, 0.62 * sz, 0.66 * sz, r, c),
            cyl(0, 0, 0.96 * sz, sz, r, "steel"), cyl(0.1 * r, 0, 0.99 * sz, sz * 1.0005, 0.12 * r, "steel")]


def sack(s, c):
    """A pillow of woven cloth."""
    sx, sy, sz = s
    return [ball(0, 0, sz / 2, sx / 2, sy / 2, sz / 2, c)]


def bin_(s, c):
    """An open picking bin: a high back, sides sloping down to a low front lip."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    t = 0.03 * sx
    return [box(-hx, hx, -hy, hy, 0, 0.04 * sz, c), box(-hx, -hx + t, -hy, hy, 0, sz, c),
            box(hx - t, hx, -hy, hy, 0, 0.4 * sz, c), wedge(-hx, hx, -hy, -hy + t, 0, sz, 0.4 * sz, c),
            wedge(-hx, hx, hy - t, hy, 0, sz, 0.4 * sz, c)]


def forklift(s, c):
    """A counterbalance forklift: body and hood, an overhead guard, a mast with two forks, four tyres. Forks at +x."""
    sx, sy, sz = s
    hx = sx / 2
    parts = [box(-hx, 0.15 * sx, -0.4 * sy, 0.4 * sy, 0.1 * sz, 0.37 * sz, c),
             box(-hx, -0.1 * sx, -0.42 * sy, 0.42 * sy, 0.37 * sz, 0.48 * sz, c),
             box(-0.2 * sx, 0.12 * sx, -0.45 * sy, 0.45 * sy, 0.97 * sz, sz, c),
             box(-0.1 * sx, 0.0, -0.2 * sy, 0.2 * sy, 0.37 * sz, 0.62 * sz, "dark"),
             box(0.19 * sx, 0.26 * sx, -0.42 * sy, 0.42 * sy, 0.05 * sz, 0.34 * sz, "dark*1.6")]
    for sy_ in (-1, 1):
        parts += [box(-0.19 * sx, -0.15 * sx, sy_ * 0.42 * sy - 0.03, sy_ * 0.42 * sy + 0.03, 0.48 * sz, 0.97 * sz, c),
                  box(0.06 * sx, 0.1 * sx, sy_ * 0.42 * sy - 0.03, sy_ * 0.42 * sy + 0.03, 0.34 * sz, 0.97 * sz, c),
                  box(0.2 * sx, 0.24 * sx, sy_ * 0.2 * sy - 0.04, sy_ * 0.2 * sy + 0.04, 0.02 * sz, 0.95 * sz,
                      "dark*1.6"),
                  box(0.26 * sx, hx, sy_ * 0.27 * sy - 0.06, sy_ * 0.27 * sy + 0.06, 0, 0.03 * sz, "dark*1.6"),
                  _wheel(0.1 * sx, sy_ * 0.41 * sy, 0.14 * sz, 0.14 * sz, 0.18 * sy, "dark"),
                  _wheel(-0.33 * sx, sy_ * 0.38 * sy, 0.1 * sz, 0.1 * sz, 0.15 * sy, "dark")]
    return parts


def person(s, c):
    """A resident in the placeholder appearance: a plain box of the entity's size in a neutral grey. A person has no
    colour attribute (nobody is described by one), so `c`, which `build_scene` passes for every mover, is not used."""
    sx, sy, sz = s
    return [box(-sx / 2, sx / 2, -sy / 2, sy / 2, 0, sz, "grey")]


def tote_robot(s, c):
    """A wheeled base, a tall frame with a screen and a carrying deck."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    parts = [box(-hx, hx, -hy, hy, 0.05 * sz, 0.23 * sz, c), box(-hx, -0.2 * hx, -hy, hy, 0.23 * sz, sz, c),
             box(-0.2 * hx, hx, -0.9 * hy, 0.9 * hy, 0.4 * sz, 0.43 * sz, c),
             box(-0.2 * hx * 1.0005, -0.2 * hx * 1.0005 + 0.005, -0.25 * sy, 0.25 * sy, 0.75 * sz, 0.92 * sz,
                 "dark*2.5")]
    parts += [_wheel(ix * (hx - 0.12), iy * (hy - 0.02), 0.045 * sz, 0.045 * sz, 0.04, "dark") for ix in (-1, 1)
              for iy in (-1, 1)]
    return parts


def _cab_vehicle(s, c, cab: tuple[float, float], belt: float, roof: float = 1.0, wheel_x: float = 0.3) -> list[Part]:
    """A road vehicle in the colour attribute: a body from the sills to `belt` of the height, a cabin over the stretch
    `cab` of the length (as fractions of the half length, -1 the tail, +1 the nose) up to `roof`, a dark glass band
    round the cabin, and four dark wheels. Nose at +x."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    r = 0.17 * sz
    x0, x1 = cab[0] * hx, cab[1] * hx
    parts = [box(-hx, hx, -hy, hy, 0.5 * r, belt * sz, c), box(x0, x1, -0.86 * hy, 0.86 * hy, belt * sz, roof * sz, c),
             box(x0 - 0.002, x1 + 0.002, -0.86 * hy - 0.003, 0.86 * hy + 0.003, (belt + 0.12 * (roof - belt)) * sz,
                 (roof - 0.1 * (roof - belt)) * sz, "dark")]
    return parts + [_wheel(wx * hx, wy * (hy - 0.05 * sy), r, r, 0.12 * sy, "dark")
                    for wx in (-wheel_x - 0.3, wheel_x + 0.3) for wy in (-1, 1)]


def car(s, c):
    return _cab_vehicle(s, c, (-0.45, 0.3), 0.58, 1.0)


def taxi(s, c):
    sx, sy, sz = s
    return car(s, c) + [box(-0.08 * sx, 0.1 * sx, -0.18 * sy, 0.18 * sy, sz, sz * 1.08, "white")]


def van(s, c):
    return _cab_vehicle(s, c, (-0.9, 0.5), 0.5, 1.0)


def pickup(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return _cab_vehicle(s, c, (-0.05, 0.5), 0.5, 1.0) + [box(-hx, -0.1 * hx, -hy, hy, 0.5 * sz, 0.62 * sz, c),
                                                         box(-0.9 * hx, -0.12 * hx, -0.8 * hy, 0.8 * hy, 0.5 * sz,
                                                             0.52 * sz, "dark")]


def box_truck(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    r = 0.14 * sz
    return [box(-hx, 0.25 * hx, -hy, hy, 0.2 * sz, sz, c),
            box(0.3 * hx, hx, -0.92 * hy, 0.92 * hy, 0.2 * sz, 0.78 * sz, c),
            box(0.78 * hx, hx * 1.002, -0.8 * hy, 0.8 * hy, 0.45 * sz, 0.7 * sz, "dark"),
            box(-0.9 * hx, 0.9 * hx, -0.6 * hy, 0.6 * hy, 0.1 * sz, 0.2 * sz, "dark")] + [
        _wheel(wx * hx, wy * (hy - 0.06), r, r, 0.12 * sy, "dark") for wx in (-0.6, 0.65) for wy in (-1, 1)]


def skip(s, c):
    """An open tub with a heap of rubble."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    t = 0.04 * sx
    return [box(-0.8 * hx, 0.8 * hx, -0.7 * hy, 0.7 * hy, 0, 0.1 * sz, c), box(-hx, hx, -hy, -hy + t, 0, sz, c),
            box(-hx, hx, hy - t, hy, 0, sz, c), box(-hx, -hx + t, -hy, hy, 0, sz, c),
            box(hx - t, hx, -hy, hy, 0, sz, c),
            ball(0, 0, 0.85 * sz, 0.8 * hx, 0.7 * hy, 0.25 * sz, "grey*0.8")]


def dumpster(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0.12 * sz, 0.8 * sz, c), box(-hx, 0.05 * hx, -hy, hy, 0.8 * sz, 0.92 * sz, "dark"),
            box(0.05 * hx, hx, -hy, hy, 0.8 * sz, 0.86 * sz, "dark")] + [
        _wheel(wx * 0.85 * hx, wy * 0.6 * hy, 0.06 * sz, 0.06 * sz, 0.05, "dark") for wx in (-1, 1) for wy in (-1, 1)]


def portable_toilet(s, c):
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    return [box(-hx, hx, -hy, hy, 0.05 * sz, 0.92 * sz, c),
            box(-hx * 1.04, hx * 1.04, -hy * 1.04, hy * 1.04, 0.92 * sz, sz,
                                                               "white"),
            box(hx, hx * 1.01, -0.4 * hy, 0.4 * hy, 0.08 * sz, 0.85 * sz, c + "*0.7"),
            box(-hx * 1.02, hx * 1.02, -hy * 1.02, hy * 1.02, 0, 0.06 * sz, "grey")]


def food_cart(s, c):
    """A counter on two wheels under a flat umbrella in the colour attribute."""
    sx, sy, sz = s
    return [box(-0.35 * sx, 0.35 * sx, -0.36 * sy, 0.36 * sy, 0.18 * sz, 0.42 * sz, "steel"),
            cyl(0, 0, 0.42 * sz, 0.95 * sz, 0.03, "steel"), ball(0, 0, 0.9 * sz, 0.5 * sx, 0.5 * sy, 0.12 * sz, c)] + [
        _wheel(-0.07 * sx, wy * 0.45 * sy, 0.15 * sz, 0.15 * sz, 0.06, "dark") for wy in (-1, 1)]


def bus(s, c):
    """A long box with a dark window band, a dark windscreen and four wheels."""
    sx, sy, sz = s
    hx, hy = sx / 2, sy / 2
    r = 0.17 * sz
    return [box(-hx, hx, -hy, hy, 0.5 * r, 0.97 * sz, c), box(-0.9 * hx, 0.9 * hx, -hy * 1.003, hy * 1.003, 0.45 * sz,
                                                              0.8 * sz, "dark"),
            box(hx * 1.001, hx * 1.003, -0.85 * hy, 0.85 * hy, 0.4 * sz, 0.88 * sz, "dark")] + [
        _wheel(wx * hx, wy * (hy - 0.05), r, r, 0.12 * sy, "dark") for wx in (-0.55, 0.5) for wy in (-1, 1)]


BUILDERS = {f.__name__.rstrip("_"): f for f in (mug, glass, bowl, plate, book, keys, phone, remote, glasses, bag, shoes,
                                               laptop, plant, bottle, box_, towel, toy, robot_vacuum, home_robot,
                                               ball_robot, pallet, tote, drum, sack, bin_, forklift, tote_robot, car,
                                               van, pickup, box_truck, skip, dumpster, portable_toilet, food_cart, taxi,
                                               bus, person)}
