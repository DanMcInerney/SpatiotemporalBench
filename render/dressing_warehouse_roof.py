# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's roof and what hangs under it, all above 5 m (`dressing_layout_warehouse`): trusses, purlins, the
profiled deck, roof lights, air ducts, cable trays, red ESFR sprinkler pipes and their pendent heads, and the round
high-bay fixtures.

Runs inside Blender's own Python and is never imported by `stbench`. Every builder is `fn(d, ctx, text) -> [object]`
(`dressing_warehouse_kit`) and draws to its Dressing's size: these pieces all have yaw 0, so a Dressing's x is the
map's x. A high-bay fixture is its body only: its lowest point, the lens plane, is at the Dressing's z (5.80 m, the
map's light height), nothing opaque hangs below it, and the lighting lane puts the light emitter just under the lens.
A skylight's material slot 0 is its pane, which the lighting lane gives its daylight material; slot 1 is the frame.
"""

from __future__ import annotations

from dressing_warehouse_kit import GALV, GREY, LENS_HIGH, PANE, RED, RUBBER, STEEL, WHITE, Mesh

# the C purlin's section, 80 x 140 mm with 20 mm lips and a 2.5 mm web: (x, z) from the base centre
PURLIN_SECTION = [(-0.04, 0.0), (0.04, 0.0), (0.04, 0.02), (0.0375, 0.02), (0.0375, 0.0025), (-0.0375, 0.0025),
                  (-0.0375, 0.1375), (0.0375, 0.1375), (0.0375, 0.12), (0.04, 0.12), (0.04, 0.14), (-0.04, 0.14)]
DECK_RIBS = (0.39, 0.11, 0.39, 0.11)  # a trapezoidal deck's period: crest, flank, pan, flank (fractions)
DECK_PITCH_M = 0.18
TRUSS_PANEL_M = 1.2  # nominal panel of a truss: the diagonals meet the chords this far apart


def roof_truss(d, ctx, text):
    """A Warren truss across the hall: two flat trusses 0.2 m apart (hollow-section chords, round diagonals), tied by
    cross bars at every node and ended with bearing plates."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, lane = sx / 2, sy / 2 - 0.025
    n = round(sx / TRUSS_PANEL_M)
    zb, zt = 0.035, sz - 0.04
    pitch = (sx - 0.06) / n  # the end nodes are 3 cm in from the plates
    bottom = [-hx + 0.03 + k * pitch for k in range(n + 1)]
    top = [-hx + 0.03 + (k + 0.5) * pitch for k in range(n)]
    for y in (-lane, lane):
        m.box(-hx, hx, y - 0.025, y + 0.025, 0.0, 0.07, GREY)
        m.box(-hx, hx, y - 0.025, y + 0.025, sz - 0.08, sz, GREY)
        for k in range(n):
            m.rod((bottom[k], y, zb), (top[k], y, zt), 0.016, GREY, 8)
            m.rod((top[k], y, zt), (bottom[k + 1], y, zb), 0.016, GREY, 8)
        for x in (bottom[0], bottom[-1]):
            m.rod((x, y, zb), (x, y, zt), 0.016, GREY, 8)
    for x in bottom:
        m.rod((x, -lane, zb), (x, lane, zb), 0.01, GREY, 6)
    for x in top:
        m.rod((x, -lane, zt), (x, lane, zt), 0.01, GREY, 6)
    m.box(hx - 0.006, hx, -sy / 2, sy / 2, 0.0, sz, GREY)
    m.box(-hx, -hx + 0.006, -sy / 2, sy / 2, 0.0, sz, GREY)
    return [m.finish()]


def purlin(d, ctx, text):
    """A cold-formed C section run along the hall under the deck."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    m.extrude([(x * sx / 0.08, z * sz / 0.14) for x, z in PURLIN_SECTION], "y", -sy / 2, sy / 2, GALV)
    return [m.finish()]


def roof_deck(d, ctx, text):
    """The profiled steel deck seen from below: trapezoidal ribs 0.18 m apart run across the hall (along x), its crests
    against the ceiling plane and its pans 3.5 cm lower. One sheet of faces, no thickness."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    ribs = round(sy / DECK_PITCH_M)
    period = sy / ribs
    lo, hi = 0.003, sz - 0.003
    cuts = [0.0, DECK_RIBS[0], DECK_RIBS[0] + DECK_RIBS[1], DECK_RIBS[0] + DECK_RIBS[1] + DECK_RIBS[2]]
    profile = []
    for i in range(ribs):
        y = -sy / 2 + i * period
        profile += [(y + cuts[0] * period, hi), (y + cuts[1] * period, hi), (y + cuts[2] * period, lo),
                    (y + cuts[3] * period, lo)]
    profile.append((sy / 2, hi))
    a = [m.bm.verts.new((-sx / 2, y, z)) for y, z in profile]
    b = [m.bm.verts.new((sx / 2, y, z)) for y, z in profile]
    slot = m.slot(WHITE)
    for i in range(len(profile) - 1):
        m.bm.faces.new([a[i], a[i + 1], b[i + 1], b[i]]).material_index = slot
    return [m.finish()]


def skylight(d, ctx, text):
    """A roof light under the deck: a milky glowing pane (material slot 0) in an aluminium frame with two glazing bars
    (slot 1)."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy, f = sx / 2, sy / 2, 0.05
    m.box(-hx + f - 0.002, hx - f + 0.002, -hy + f - 0.002, hy - f + 0.002, 0.005, sz - 0.004, PANE)
    for x0, x1, y0, y1 in ((-hx, hx, -hy, -hy + f), (-hx, hx, hy - f, hy), (-hx, -hx + f, -hy, hy),
                           (hx - f, hx, -hy, hy)):
        m.box(x0, x1, y0, y1, 0.0, sz, STEEL)
    for y in (-sy / 6, sy / 6):
        m.box(-hx + f, hx - f, y - 0.015, y + 0.015, 0.0, sz, STEEL)
    return [m.finish()]


def air_duct(d, ctx, text):
    """A round spiral-wound galvanised duct along x: seam rings every 0.35 m, a collar every 3 m, capped ends."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    r, hx = min(sy, sz) / 2 - 0.003, sx / 2
    m.cylinder("x", (0.0, 0.0, sz / 2), r, sx, GALV, 32)
    for i in range(1, int(sx / 0.35)):
        m.revolve([(r - 0.001, -0.003), (r + 0.002, -0.002), (r + 0.002, 0.002), (r - 0.001, 0.003)], GALV,
                  (-hx + 0.35 * i, 0.0, sz / 2), "x", 32)
    for i in range(1, int(sx / 3.0) + 1):
        m.band("x", (-hx + 3.0 * i - 1.1, 0.0, sz / 2), r, r + 0.0055, 0.05, STEEL, 32)
    return [m.finish()]


def cable_tray(d, ctx, text):
    """A ladder tray along y: side rails with a return, a rung every 0.3 m and four cables lying on them."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    hx, hy = sx / 2, sy / 2
    for x0, x1, f0, f1 in ((-hx, -hx + 0.003, -hx, -hx + 0.02), (hx - 0.003, hx, hx - 0.02, hx)):
        m.box(x0, x1, -hy, hy, 0.0, sz, GALV)
        m.box(f0, f1, -hy, hy, sz - 0.003, sz, GALV)
    for i in range(int(sy / 0.3)):
        y = -hy + 0.15 + 0.3 * i
        m.box(-hx + 0.003, hx - 0.003, y - 0.015, y + 0.015, 0.0, 0.008, GALV)
    for x, r in ((-0.08, 0.014), (-0.03, 0.011), (0.03, 0.017), (0.09, 0.012)):
        m.rod((x, -hy + 0.02, 0.008 + r), (x, hy - 0.02, 0.008 + r), r, RUBBER, 10)
    return [m.finish()]


def sprinkler_pipe(d, ctx, text):
    """A red steel pipe along the longer side of its box: grooved couplings every 3 m and hanger rings between."""
    m = Mesh(d, ctx)
    sx, sy, sz = d.size
    axis = "x" if sx > sy else "y"
    length, across = (sx, sy) if axis == "x" else (sy, sx)
    half = min(across, sz) / 2
    r = half - 0.002

    def at(a: float) -> tuple[float, float, float]:
        return (a, 0.0, sz / 2) if axis == "x" else (0.0, a, sz / 2)

    m.cylinder(axis, (0.0, 0.0, sz / 2), r, length, RED, 20)
    for i in range(int(length / 3.0)):
        a = -length / 2 + 1.2 + 3.0 * i
        m.band(axis, at(a), r, half + 0.003, 0.06, STEEL, 20)
        if a + 1.5 < length / 2 - 0.3:
            m.band(axis, at(a + 1.5), r, half + 0.002, 0.014, STEEL, 20)
    return [m.finish()]


def sprinkler_head(d, ctx, text):
    """An ESFR pendent head under a branch line: a threaded inlet, a body, two frame arms and a slotted deflector."""
    m = Mesh(d, ctx)
    sz = d.size[2]
    m.revolve([(0.0, sz), (0.0105, sz), (0.0105, sz - 0.015), (0.019, sz - 0.015), (0.019, sz - 0.05),
               (0.013, sz - 0.06), (0.0, sz - 0.06)], STEEL, segments=16)
    for s in (-1, 1):
        m.rod((s * 0.0175, 0.0, sz - 0.058), (s * 0.006, 0.0, 0.06), 0.0035, STEEL, 6)
    m.revolve([(0.0, 0.07), (0.009, 0.07), (0.009, 0.04), (0.0, 0.04)], STEEL, segments=12)
    m.revolve([(0.0, 0.04), (0.012, 0.038), (0.043, 0.032), (0.043, 0.028), (0.012, 0.034), (0.0, 0.035)], STEEL,
              segments=24)
    return [m.finish()]


def high_bay_light(d, ctx, text):
    """A round LED high-bay fixture, 0.4 m across: a flat lens just above its base (the plane of the map's light), a
    bezel whose underside is that plane, and a housing stepped like a stack of cooling fins up to a driver cap."""
    m = Mesh(d, ctx)
    sz = d.size[2]
    m.revolve([(0.0, 0.004), (0.148, 0.004), (0.148, 0.009), (0.0, 0.009)], LENS_HIGH, segments=40)
    body = [(0.148, 0.0), (0.2, 0.0), (0.2, 0.025)]
    for k in range(8):
        r, z = 0.2 - 0.0165 * k, 0.025 + 0.0135 * k
        body += [(r, z + 0.004), (r - 0.0035, z + 0.004), (r - 0.0035, z + 0.0135)]
    body += [(0.07, sz - 0.015), (0.06, sz - 0.015), (0.06, sz), (0.0, sz)]
    m.revolve(body, STEEL, segments=40)
    for a in range(0, 360, 90):  # four mounting lugs on the cap's rim
        with m.turned(a):
            m.box(0.055, 0.075, -0.01, 0.01, sz - 0.04, sz - 0.015, STEEL)
    return [m.finish()]
