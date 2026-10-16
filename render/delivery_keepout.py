# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""What a flier's delivery pieces keep clear of render-only objects (lane M3-RY): plain Python (no `bpy`), so a test
reads it without Blender.

A flier hovers over a drop zone and lifts from a nest, and keeps its body and its clearance (`robots`: 0.65 m and
3.0 m) from everything it does not fly over. `validate` knows the map's pieces and walls, not the objects only the
render draws (a street tree, a lamp post): a frame could show the drone inside one, and a post in a zone would block it
in every case, a fact of the layout that a shortcut learns. So no render-only object that stands meets
- the **column** of a delivery piece: the disc about its centre of its half-diagonal plus the fliers' reach, from the
  ground to `TOP_M` (the observed flights stay under the lowest roof, 18.9 m), nor
- a nest's **yard**: the bare ground round it, out to the nearest piece of ground or wall on each side.
A plate no thicker than `FLUSH_M` (paint, iron) is flat and a flier passes over it, so it may lie anywhere. The pieces
are found by kind (`KINDS`), never by name or place, so every map with a nest or a drop zone gets the same rule.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from stbench.robots import ROBOTS

Box = tuple[float, float, float, float, float, float]  # xmin, ymin, zmin, xmax, ymax, zmax, as `dressing_spec.Box`
Rect4 = tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
Piece = tuple[str, tuple[float, float], tuple[float, float, float]]  # a map piece's kind, centre and size

KINDS = ("drop_zone", "nest")
YARD_KIND = "nest"
TOP_M = 18.0
FLUSH_M = 0.02  # a plate this thin or thinner is flat: `dressing_boxes.FLUSH_M`
REACH_M = max(max(r.body.length_m, r.body.width_m) / 2 + r.body.clearance_m for r in ROBOTS.values() if r.flies)
SIDES = ("+x", "-x", "+y", "-y")


class Column(NamedTuple):
    """The space a flier needs round a delivery piece: the disc of radius `r` about (`x`, `y`), from the ground up."""

    x: float
    y: float
    r: float

    def meets(self, box: Box) -> bool:
        dx = max(box[0] - self.x, 0.0, self.x - box[3])
        dy = max(box[1] - self.y, 0.0, self.y - box[4])
        return box[2] < TOP_M and math.hypot(dx, dy) < self.r


class Yard(NamedTuple):
    """The bare ground round a nest: `rect` runs out to the nearest piece of ground or wall on each side, and
    `ground_sides` are the sides (`SIDES`) that a piece of ground bounds, not a wall."""

    rect: Rect4
    ground_sides: tuple[str, ...]

    def meets(self, box: Box) -> bool:
        return box[0] < self.rect[1] and self.rect[0] < box[3] and box[1] < self.rect[3] and self.rect[2] < box[4]


class KeepOut(NamedTuple):
    columns: tuple[Column, ...]
    yards: tuple[Yard, ...]

    def meets(self, box: Box) -> bool:
        """Whether a render-only object whose box is `box` stands in a column or a yard."""
        return box[5] > FLUSH_M and (any(c.meets(box) for c in self.columns) or any(y.meets(box) for y in self.yards))

    def clear(self, unit: Sequence) -> list:
        """The objects of a unit (a tree's stem and crown stand or go together: each has a `box`), or none of them when
        any meets."""
        return [] if any(self.meets(d.box) for d in unit) else list(unit)


def yard(centre: tuple[float, float], grounds: Iterable[Rect4], walls: Iterable[Rect4]) -> Yard:
    """The yard of a nest at `centre`: the largest rectangle about it that runs out to the nearest of `grounds` (the
    footprints of the raised ground pieces) or `walls` (the slabs) abeam of it on each side. ValueError when a side has
    nothing to stop at (a nest in the open has no yard)."""
    x, y = centre
    found: dict[str, list[tuple[float, bool]]] = {s: [] for s in SIDES}
    for ground, rects in ((True, grounds), (False, walls)):
        for x0, x1, y0, y1 in rects:
            abeam_x, abeam_y = y0 <= y <= y1, x0 <= x <= x1  # abeam of the centre along x, and along y
            for side, near, ahead in (("+x", x0, abeam_x and x0 >= x), ("-x", x1, abeam_x and x1 <= x),
                                      ("+y", y0, abeam_y and y0 >= y), ("-y", y1, abeam_y and y1 <= y)):
                if ahead:
                    found[side].append((near, ground))
    if not all(found.values()):
        raise ValueError(f"the nest at ({x:.2f}, {y:.2f}) has nothing to stop its yard on the "
                         f"{[s for s in SIDES if not found[s]]} side")
    edge = {s: (min if s[0] == "+" else max)(found[s]) for s in SIDES}
    rect = (edge["-x"][0], edge["+x"][0], edge["-y"][0], edge["+y"][0])
    return Yard(rect, tuple(s for s in SIDES if edge[s][1]))


def columns(pieces: Iterable[Piece]) -> tuple[Column, ...]:
    """A column for each delivery piece (by `KINDS`) of the map's `pieces`."""
    return tuple(Column(c[0], c[1], math.hypot(size[0], size[1]) / 2 + REACH_M)
                 for kind, c, size in pieces if kind in KINDS)


def keep_out(pieces: Sequence[Piece], grounds: Sequence[Rect4] | None = None, walls: Sequence[Rect4] = ()) -> KeepOut:
    """The keep-out of a map: a column for each delivery piece and, when the raised `grounds` are given, a yard for each
    nest."""
    yards = tuple(yard(c, grounds, walls) for kind, c, _ in pieces if kind == YARD_KIND) if grounds is not None else ()
    return KeepOut(columns(pieces), yards)
