# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Where an outdoor map's street paint goes: centre lines, lane dashes, crosswalks and stop bars, as plain geometry.

Plain Python (no `bpy`), so the layout can be tested without Blender; `dressing_layout_downtown.py` turns the marks of
the map into dressing and `scenery_layout.py` carries the lines on past the map's open street ends. Everything is read
from the map's own pieces (a dict as the map file holds it), none of it from a name: a *street* is a region that has
a sidewalk along each long side; its carriageway lies between the two kerbs; parking lanes take their width from each
kerb; the travel lanes between them are about 3.45 m wide and split by a double yellow centre line (and, when each way
has two lanes, a white dashed line between them). Where a side street meets the street along a shared region edge the
street's dashes stop and a crosswalk crosses the side street's mouth, on the sidewalk's line, with a stop bar for the
traffic coming out of it; where a region that is no street (the square) meets the street, a crosswalk crosses the
street's carriageway 3.5 m in from each end of the shared edge, with a stop bar either way. Traffic keeps right.

A `Mark` is one piece of paint on the ground: its kind (the dressing class that draws it), the rectangle it lies in and
the direction its pattern runs; `paint_rects` is the paint itself, as rectangles. Every mark lies in the carriageway
(never on a sidewalk), and the layout keeps them off the map's parking and loading surfaces.
"""

from __future__ import annotations

import math
from typing import NamedTuple

LANE_M = 3.45  # a travel lane
CENTRE_LINE_M, CENTRE_GAP_M = 0.14, 0.12  # each yellow line of the double centre line, and the gap between them
DASH_W_M, DASH_M, DASH_PITCH_M, DASH_LEAD_M = 0.15, 3.0, 12.0, 1.0  # a lane dash, its length, its pitch, its lead-in
ZEBRA_W_M, ZEBRA_PITCH_M, ZEBRA_LEN_M = 0.5, 1.0, 3.0  # a crosswalk stripe, their pitch and the crossing's depth
BAR_M, BAR_GAP_M = 0.4, 1.2  # a stop bar's thickness and its distance from the crosswalk
CROSSWALK_INSET_M = 0.5  # a crosswalk across a mouth keeps this far from the sidewalk ends and the region edge
FLANK_M = 3.5  # a crosswalk beside a square: this far in from the end of the shared edge
CUT_M = 1.6  # lines stop this far from a crosswalk
MIN_RUN_M = 4.0  # a run of line shorter than this is dropped
TOL = 0.05
Rect4 = tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


class Mark(NamedTuple):
    """One piece of road paint: `kind` is the dressing class (`lane_marking`, `crosswalk`, `stop_bar`), `style` the
    pattern (`double`, `dash`, `zebra`, `bar`), `rect` the rectangle it lies in and `along` ("x" or "y") the direction
    its stripes, dashes or lines run."""

    kind: str
    style: str
    rect: Rect4
    along: str


class Street(NamedTuple):
    """A street: `axis` the direction it runs in, `lo`..`hi` its extent along it (the region's), `c_lo`..`c_hi` the
    carriageway between the kerbs, across it, and `park` the width of the parking lane on each side (0: none)."""

    region: str
    axis: str
    lo: float
    hi: float
    c_lo: float
    c_hi: float
    park: float

    @property
    def mid(self) -> float:
        return (self.c_lo + self.c_hi) / 2

    @property
    def lanes(self) -> int:
        """Travel lanes in all: an even number, at least two."""
        return max(2, 2 * round((self.c_hi - self.c_lo - 2 * self.park) / (2 * LANE_M)))


# -- the map's pieces -----------------------------------------------------------------------------------------------
def _quarter(yaw: float) -> bool:
    return abs(math.sin(2 * yaw)) < 1e-6


def footprint(f: dict) -> Rect4:
    """The axis-aligned rectangle of a furniture piece turned a quarter turn."""
    (cx, cy), (d, w, _) = f["center"], f["size"]
    if not _quarter(f["yaw"]):
        raise ValueError(f"{f['id']} is turned {math.degrees(f['yaw']):.1f} degrees: a rectilinear map is needed")
    hx, hy = (d / 2, w / 2) if abs(math.cos(f["yaw"])) > 0.5 else (w / 2, d / 2)
    return cx - hx, cx + hx, cy - hy, cy + hy


def _region(r: dict) -> Rect4:
    xs, ys = [p[0] for p in r["polygon"]], [p[1] for p in r["polygon"]]
    return min(xs), max(xs), min(ys), max(ys)


def _inside(b: Rect4, r: Rect4) -> bool:
    return b[0] >= r[0] - TOL and b[1] <= r[1] + TOL and b[2] >= r[2] - TOL and b[3] <= r[3] + TOL


def _across(r: Rect4, axis: str) -> tuple[float, float]:
    return (r[2], r[3]) if axis == "x" else (r[0], r[1])


def _along(r: Rect4, axis: str) -> tuple[float, float]:
    return (r[0], r[1]) if axis == "x" else (r[2], r[3])


def rect_of(axis: str, a0: float, a1: float, c0: float, c1: float) -> Rect4:
    """The rectangle that spans `a0..a1` along `axis` and `c0..c1` across it."""
    return (a0, a1, c0, c1) if axis == "x" else (c0, c1, a0, a1)


def view(m) -> dict:
    """The parts of a parsed `MapSpec` that this module reads, as the dict of the map file (for callers that hold the
    spec parsed: `scenery_layout`)."""
    return {"regions": [{"id": r.id, "polygon": [list(p) for p in r.polygon]} for r in m.regions],
            "furniture": [{"id": f.id, "kind": f.kind, "center": list(f.center), "size": list(f.size), "yaw": f.yaw,
                           "surfaces": []} for f in m.furniture]}


def streets(spec: dict) -> list[Street]:
    """The streets of the map: regions with a sidewalk along each long side, in region order."""
    walks = [footprint(f) for f in spec["furniture"] if f["kind"] == "sidewalk"]
    lanes = [f for f in spec["furniture"] if f["kind"] == "parking_lane"]
    out = []
    for r in spec["regions"]:
        rr = _region(r)
        axis = "x" if rr[1] - rr[0] >= rr[3] - rr[2] else "y"
        lo, hi = _along(rr, axis)
        a_lo, a_hi = _across(rr, axis)
        mid = (a_lo + a_hi) / 2
        mine = [b for b in walks if _inside(b, rr) and (b[1] - b[0] > b[3] - b[2]) == (axis == "x")]
        low = [_across(b, axis)[1] for b in mine if sum(_across(b, axis)) / 2 < mid]
        high = [_across(b, axis)[0] for b in mine if sum(_across(b, axis)) / 2 > mid]
        if not low or not high:
            continue
        park = max((min(f["size"][0], f["size"][1]) for f in lanes if _inside(footprint(f), rr)), default=0.0)
        out.append(Street(r["id"], axis, lo, hi, max(low), min(high), park))
    return out


def _shared(a: Rect4, b: Rect4, axis: str) -> tuple[float, float, float] | None:
    """(along-start, along-end, across line) of the edge two region rectangles share across the `axis` direction's
    perpendicular, or None: `a`'s side that touches `b`, `b` lying beyond it across the street."""
    for line in _across(a, axis):
        if any(abs(line - v) < TOL for v in _across(b, axis)):
            lo, hi = max(_along(a, axis)[0], _along(b, axis)[0]), min(_along(a, axis)[1], _along(b, axis)[1])
            if hi - lo > 1.0:
                return lo, hi, line
    return None


# -- the marks ------------------------------------------------------------------------------------------------------
def _runs(lo: float, hi: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`lo..hi` without the `cuts`, runs shorter than `MIN_RUN_M` dropped."""
    out, at = [], lo
    for a, b in sorted(cuts):
        if a > at:
            out.append((at, min(a, hi)))
        at = max(at, b)
    if at < hi:
        out.append((at, hi))
    return [(a, b) for a, b in out if b - a >= MIN_RUN_M]


def _bars(s: Street, centre: float, half_width: float) -> list[Mark]:
    """The stop bars of a mid-street crosswalk of the street centred at `centre`: each way's traffic keeps right, so
    traffic going up the axis stops on the low side of the crossing, on its right hand, and traffic coming down on the
    high side."""
    low_half = (s.c_lo + s.park, s.mid - 0.2)
    high_half = (s.mid + 0.2, s.c_hi - s.park)
    right_is_low = s.axis == "x"  # heading +x, the right hand is -y; heading +y, it is +x, the high side
    lo_edge, hi_edge = centre - half_width - BAR_GAP_M, centre + half_width + BAR_GAP_M
    first, second = (low_half, high_half) if right_is_low else (high_half, low_half)
    return [Mark("stop_bar", "bar", rect_of(s.axis, lo_edge - BAR_M, lo_edge, *first), s.axis),
            Mark("stop_bar", "bar", rect_of(s.axis, hi_edge, hi_edge + BAR_M, *second), s.axis)]


def _mouth(s: Street, side: Street, line: float) -> tuple[list[Mark], tuple[float, float], tuple[float, float]]:
    """A side street `side` that meets street `s` along the region edge `line` (across `s`): the crosswalk over its
    mouth on the sidewalk's line, the stop bar for the traffic coming out of it, the stretch of `s` its mouth takes (no
    dashes there) and the stretch of the side street's own lines that the bar and crosswalk take."""
    up = line > s.mid  # the side street lies on the high side of `s`
    sgn = 1.0 if up else -1.0
    kerb = s.c_hi if up else s.c_lo
    cross = sorted((kerb + sgn * CROSSWALK_INSET_M, line - sgn * CROSSWALK_INSET_M))
    inset = CROSSWALK_INSET_M
    walk = Mark("crosswalk", "zebra", rect_of(s.axis, side.c_lo + inset, side.c_hi - inset, *cross), side.axis)
    # the traffic heads for `s`: down the side street's axis when it lies above, else up; its right hand is the low
    # side of the side street when it heads +x or -y, i.e. when (axis x) heading is + or (axis y) heading is -
    heading = -sgn
    right_is_low = heading > 0 if side.axis == "x" else heading < 0
    half = (side.c_lo + side.park, side.mid - 0.2) if right_is_low else (side.mid + 0.2, side.c_hi - side.park)
    first = line - sgn * CROSSWALK_INSET_M + sgn * BAR_GAP_M  # the bar's near edge, past the crosswalk
    bar = sorted((first, first + sgn * BAR_M))
    stop = Mark("stop_bar", "bar", rect_of(s.axis, *half, *bar), side.axis)
    reach = (-1e9, bar[1] + CUT_M) if up else (bar[0] - CUT_M, 1e9)
    return [walk, stop], (side.c_lo - 1.0, side.c_hi + 1.0), reach


def marks(spec: dict) -> list[Mark]:
    """Every piece of paint of the map's streets, in a fixed order: crosswalks and their stop bars, then the centre
    lines, then the lane dashes."""
    regions = {r["id"]: _region(r) for r in spec["regions"]}
    all_streets = streets(spec)
    by_id = {s.region: s for s in all_streets}
    out: list[Mark] = []
    cuts: dict[str, list[tuple[float, float]]] = {s.region: [] for s in all_streets}  # of every line of a street
    dash_cuts: dict[str, list[tuple[float, float]]] = {s.region: [] for s in all_streets}  # of the dashes alone
    for s in all_streets:
        for rid, rr in regions.items():
            shared = _shared(regions[s.region], rr, s.axis) if rid != s.region else None
            if shared is None:
                continue
            a, b, line = shared
            if rid in by_id:  # a side street: a crosswalk over its mouth, and a stop bar for the traffic leaving it
                made, mouth, reach = _mouth(s, by_id[rid], line)
                out += made
                dash_cuts[s.region].append(mouth)
                cuts[rid].append(reach)
            elif b - a >= 2 * FLANK_M + ZEBRA_LEN_M:  # a square: crosswalks across the street beside its mouth
                for p in (a + FLANK_M, b - FLANK_M):
                    x0, x1 = p - ZEBRA_LEN_M / 2, p + ZEBRA_LEN_M / 2
                    out.append(Mark("crosswalk", "zebra", rect_of(s.axis, x0, x1, s.c_lo + 0.3, s.c_hi - 0.3), s.axis))
                    out += _bars(s, p, ZEBRA_LEN_M / 2)
                    cut = (x0 - BAR_GAP_M - BAR_M - CUT_M, x1 + BAR_GAP_M + BAR_M + CUT_M)
                    cuts[s.region].append(cut)
    for s in all_streets:
        out += [Mark("lane_marking", "double", rect_of(s.axis, a, b, s.mid - (CENTRE_LINE_M + CENTRE_GAP_M / 2),
                                                       s.mid + CENTRE_LINE_M + CENTRE_GAP_M / 2), s.axis)
                for a, b in _runs(s.lo, s.hi, cuts[s.region])]
    for s in all_streets:
        if s.lanes < 4:
            continue
        offset = (s.c_hi - s.c_lo - 2 * s.park) / 4
        for c in (s.mid - offset, s.mid + offset):
            out += [Mark("lane_marking", "dash", rect_of(s.axis, a, b, c - DASH_W_M / 2, c + DASH_W_M / 2), s.axis)
                    for a, b in _runs(s.lo, s.hi, [*cuts[s.region], *dash_cuts[s.region]])]
    return out


def extension(s: Street, end: str, lo: float, hi: float) -> list[Mark]:
    """The lines of street `s` carried on from its `end` ("lo" or "hi") along its axis, over `lo..hi` (map frame):
    the centre line and, where the street has them, the lane dashes."""
    out = [Mark("lane_marking", "double", rect_of(s.axis, lo, hi, s.mid - (CENTRE_LINE_M + CENTRE_GAP_M / 2),
                                                  s.mid + CENTRE_LINE_M + CENTRE_GAP_M / 2), s.axis)]
    if s.lanes >= 4:
        offset = (s.c_hi - s.c_lo - 2 * s.park) / 4
        out += [Mark("lane_marking", "dash", rect_of(s.axis, lo, hi, c - DASH_W_M / 2, c + DASH_W_M / 2), s.axis)
                for c in (s.mid - offset, s.mid + offset)]
    return out


# -- the paint ------------------------------------------------------------------------------------------------------
def paint_rects(mark: Mark, phase: float = 0.0) -> list[Rect4]:
    """The paint of a mark as rectangles (xmin, xmax, ymin, ymax): `double` two yellow lines the length of the mark,
    `dash` white dashes `DASH_M` long every `DASH_PITCH_M` from `DASH_LEAD_M` in (`phase` shifts them, to keep a
    pattern going past a seam), `zebra` stripes `ZEBRA_W_M` wide every `ZEBRA_PITCH_M` that run along the mark's `along`
    axis, `bar` the whole rectangle."""
    x0, x1, y0, y1 = mark.rect
    a0, a1 = _along(mark.rect, mark.along)
    c0, c1 = _across(mark.rect, mark.along)
    if mark.style == "bar":
        return [mark.rect]
    if mark.style == "double":
        mid = (c0 + c1) / 2
        g = CENTRE_GAP_M / 2
        return [rect_of(mark.along, a0, a1, mid - g - CENTRE_LINE_M, mid - g),
                rect_of(mark.along, a0, a1, mid + g, mid + g + CENTRE_LINE_M)]
    if mark.style == "dash":
        out, at = [], a0 + DASH_LEAD_M + phase
        while at < a1:
            out.append(rect_of(mark.along, at, min(at + DASH_M, a1), c0, c1))
            at += DASH_PITCH_M
        return out
    if mark.style == "zebra":  # stripes run along `along` (the traffic's direction) and repeat across it
        n = int((c1 - c0 - ZEBRA_W_M) / ZEBRA_PITCH_M + 1e-9) + 1
        start = (c0 + c1) / 2 - ((n - 1) * ZEBRA_PITCH_M + ZEBRA_W_M) / 2
        return [rect_of(mark.along, a0, a1, start + i * ZEBRA_PITCH_M, start + i * ZEBRA_PITCH_M + ZEBRA_W_M)
                for i in range(n)]
    raise ValueError(f"unknown paint style {mark.style!r}")
