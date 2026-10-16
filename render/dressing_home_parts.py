# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's trim, fittings, ceiling pieces and window dressing, placed on a `Plan` (`dressing_home_plan`).

Each function returns the `Dressing` of one family in a fixed order; `dressing_layout_home.layout` joins them. Every
piece stays `M` clear of the openings, window glass and door swings the validity zones (`stbench.dressing_zones`) keep
free, and at most `PROUD_MAX_M` off a wall face. Sizes are those of the real thing: trim 7 cm wide and 1.5 cm proud, a
switch plate 7 x 11.5 cm at 1.1 m, a socket at 0.3 m, a flush lamp 40 cm across.
"""

from __future__ import annotations

import math

from dressing_home_plan import Face, Plan
from dressing_spec import Dressing

M = 0.002  # kept off a forbidden volume, so that a rounded coordinate never touches it
CASING_W, CASING_PROUD = 0.07, 0.015
BASEBOARD_H, BASEBOARD_PROUD = 0.08, 0.012
PLATE_W, PLATE_H, PLATE_PROUD = 0.07, 0.115, 0.015
RADIATOR_Z = 0.15  # the underside of a panel radiator
CURTAIN_HANG_M = 0.20  # a curtain's hem may hang this far above the sill (else the window has a roller blind)
OUTLET_Z, SWITCH_Z = 0.24, 1.04  # the foot of a socket plate (centre 0.30 m) and of a switch plate (centre 1.1 m)
LAMP = (0.40, 0.40, 0.10)  # the flush lamp under a ceiling light, (x, y, z)
SMOKE = (0.11, 0.11, 0.04)
PENDANT = ("asset:modern_ceiling_lamp_01", (0.432, 0.432, 0.952))  # Poly Haven's real size
FAN = ("asset:ceiling_fan", (1.463, 1.463, 0.516))


def baseboards(plan: Plan) -> list[Dressing]:
    """8 cm trim along every wall face, between the casings of the door and arch openings and where no fixed furniture
    stands against the wall."""
    out = []
    for face in plan.faces:
        for a, b in plan.spans(face, 0.0, BASEBOARD_H, BASEBOARD_PROUD, gap_margin=CASING_W + M, keep=0.1):
            out.append(face.item("baseboard", (a + b) / 2, b - a, BASEBOARD_PROUD, 0.0, BASEBOARD_H))
    return out


def door_casings(plan: Plan) -> list[Dressing]:
    """A jamb on each side and a head over every door and arch opening, on each face that looks into a room."""
    out = []
    for face in plan.faces:
        for g in (g for g in face.gaps if g.kind != "window"):
            lo, hi = g.t0 - M - CASING_W, g.t1 + M + CASING_W
            for a, b, z0, h in ((lo, g.t0 - M, 0.0, g.z1), (g.t1 + M, hi, 0.0, g.z1), (lo, hi, g.z1 + M, CASING_W)):
                if plan.depth_free(face, a, b, z0, z0 + h, CASING_PROUD) >= CASING_PROUD:
                    out.append(face.item("door_casing", (a + b) / 2, b - a, CASING_PROUD, z0, h))
    return out


def window_trim(plan: Plan) -> list[Dressing]:
    """Around each window glass: a jamb each side, a head over it and a sill board under it (as deep as the furniture
    below it leaves, up to 7 cm). Nothing crosses the glass."""
    out = []
    for face in plan.faces:
        for g in (g for g in face.gaps if g.kind == "window"):
            lo, hi = g.t0 - M - CASING_W, g.t1 + M + CASING_W
            for a, b, z0, h in ((lo, g.t0 - M, g.z0, g.z1 - g.z0), (g.t1 + M, hi, g.z0, g.z1 - g.z0),
                                (lo, hi, g.z1 + M, CASING_W)):
                out.append(face.item("window_sill", (a + b) / 2, b - a, CASING_PROUD, z0, h))
            a, b, top = lo - 0.02, hi + 0.02, g.z0 - M
            depth = min(0.07, plan.depth_free(face, a, b, top - 0.03, top, 0.07) - 0.004)
            if depth >= 0.02:
                out.append(face.item("window_sill", (a + b) / 2, b - a, depth, top - 0.03, 0.03))
    return out


def outlets(plan: Plan) -> list[Dressing]:
    """Socket plates at 0.3 m, one every 2.6 m or so of free wall: where no door, furniture or counter stands."""
    out = []
    for face in plan.faces:
        if face.length < 1.0:
            continue
        for a, b in plan.spans(face, OUTLET_Z, OUTLET_Z + PLATE_H, PLATE_PROUD, gap_margin=CASING_W + 0.12, edge=0.3,
                               keep=0.3):
            n = max(1, round((b - a) / 2.6))
            out += [face.item("outlet", a + (b - a) * (k + 0.5) / n, PLATE_W, PLATE_PROUD, OUTLET_Z, PLATE_H)
                    for k in range(n)]
    return out


def switches(plan: Plan) -> list[Dressing]:
    """A switch plate at 1.1 m beside each door, inside the room it opens into, on the latch side (the hinge side if the
    wall runs out): a double plate where the room has two ceiling lights and the wall has room for it."""
    out = []
    for face in plan.faces:
        room = next(r for r in plan.rooms if r.id == face.region)
        widths = (0.115, PLATE_W) if len(room.lights) > 1 else (PLATE_W,)
        free = plan.spans(face, SWITCH_Z, SWITCH_Z + PLATE_H, PLATE_PROUD, edge=0.06)
        for g in (g for g in face.gaps if g.kind == "door" and g.enters):
            for width, jamb in ((w, j) for w in widths for j in (g.latch_t, g.hinge_t)):
                away = 1.0 if jamb >= (g.t0 + g.t1) / 2 else -1.0
                t = jamb + away * (M + CASING_W + 0.07 + width / 2)
                if any(a <= t - width / 2 and t + width / 2 <= b for a, b in free):
                    out.append(face.item("light_switch", t, width, PLATE_PROUD, SWITCH_Z, PLATE_H))
                    break
    return out


def thermostat(plan: Plan) -> list[Dressing]:
    """One thermostat, 1.45 m up the longest free stretch of wall in the hall (the living room if there is no hall)."""
    for role in ("hall", "living"):
        best = None
        for room in (r for r in plan.rooms if r.role == role):
            for face in plan.faces_of(room):
                for a, b in plan.spans(face, 1.4, 1.55, 0.03, gap_margin=0.3, edge=0.3):
                    if best is None or b - a > best[0]:
                        best = (b - a, face, (a + b) / 2)
        if best:
            return [plan.claim(best[1].item("thermostat", best[2], 0.09, 0.03, 1.455, 0.09))]
    return []


# -- the ceiling ---------------------------------------------------------------------------------------------------
def ceiling(plan: Plan) -> list[Dressing]:
    """A flush lamp under each ceiling light, a pendant over each dining table, a fan over each bed and a smoke
    detector in every room, clear of one another."""
    out: list[Dressing] = []
    avoid: dict[str, list[tuple[float, float, float]]] = {r.id: [] for r in plan.rooms}
    for room in plan.rooms:
        for x, y, _z in room.lights:
            out.append(Dressing("ceiling_lamp", "proc:ceiling_lamp", (x, y, round(room.z1 - LAMP[2], 4)), 0.0, LAMP))
            avoid[room.id].append((x, y, LAMP[0] / 2))
    for room in plan.rooms:
        for f in (f for f in room.furniture if f.mobility == "fixed"):
            for kind, (cls, (recipe, size)) in (("dining_table", ("ceiling_lamp", PENDANT)),
                                                ("bed", ("ceiling_fan", FAN))):
                if f.kind != kind:
                    continue
                r, (x0, y0, _z0, x1, y1, _z1) = size[0] / 2, f.box
                x, y = plan.ceiling_spot(room, r, avoid[room.id], inset=0.0, gap=0.0, near=f.center,
                                         within=(x0 + r + 0.01, y0 + r + 0.01, x1 - r - 0.01, y1 - r - 0.01))
                out.append(Dressing(cls, recipe, (x, y, round(room.z1 - size[2], 4)), 0.0, size))
                avoid[room.id].append((x, y, r))
    for room in plan.rooms:
        x, y = plan.ceiling_spot(room, SMOKE[0] / 2, avoid[room.id], inset=0.5)
        out.append(Dressing("smoke_detector", "proc:smoke_detector", (x, y, round(room.z1 - SMOKE[2], 4)), 0.0, SMOKE))
    return out


# -- windows -------------------------------------------------------------------------------------------------------
def window_dressing(plan: Plan) -> list[Dressing]:
    """Curtains beside a wide window whose wall below is clear to within `CURTAIN_HANG_M` of the sill height, a
    rolled-up roller blind in the head of any other. A curtain stops just above the volume things stand in on the
    furniture under it (at the sill where it can: a lamp 0.45 m up a nightstand holds the hem 0.17 m above the sill)."""
    out = []
    for face in plan.faces:
        for g in (g for g in face.gaps if g.kind == "window"):
            top = min(g.z1 + 0.18, face.ceiling - 0.1)
            panels = [(g.t0 - 0.36, g.t0 - 0.01), (g.t1 + 0.01, g.t1 + 0.36)]
            tall = [face.item("curtain", (a + b) / 2, b - a, 0.07, 0.0, top, standoff=0.03) for a, b in panels]
            hem = max(g.z0 - 0.10, max(plan.top_under(p.box) for p in tall) + 0.02)
            wide, inside = g.t1 - g.t0 >= 0.9, panels[0][0] >= 0 and panels[1][1] <= face.length
            if wide and hem <= g.z0 + CURTAIN_HANG_M and inside:
                out += [plan.claim(face.item("curtain", (a + b) / 2, b - a, 0.07, hem, top - hem, standoff=0.03))
                        for a, b in panels]
            else:
                out.append(plan.claim(face.item("roller_blind", (g.t0 + g.t1) / 2, g.t1 - g.t0, 0.07,
                                                g.z1 + CASING_W + M, 0.10)))
    return out


def radiators(plan: Plan) -> list[Dressing]:
    """A panel radiator in the living room and the bedroom: under a window where the wall below is free (here furniture
    stands under every window), else on the free stretch of wall nearest a window. Its underside is `RADIATOR_Z` up, so
    that the vacuum drives under it (the helper's swept body, which it stands in, is the guard's `EXEMPT_ABOVE_M`)."""
    out = []
    for room in (r for r in plan.rooms if r.role in ("living", "bedroom")):
        windows = [face.to_world((g.t0 + g.t1) / 2) for face in plan.faces_of(room) for g in face.gaps
                   if g.kind == "window"]
        best: tuple[float, Face, float] | None = None
        for face in plan.faces_of(room):
            for a, b in plan.spans(face, RADIATOR_Z, RADIATOR_Z + 0.55, 0.09, gap_margin=0.15, edge=0.15, keep=0.9):
                px, py = face.to_world((a + b) / 2)
                d = min((math.hypot(px - wx, py - wy) for wx, wy in windows), default=math.inf)
                if best is None or d < best[0]:
                    best = (d, face, (a + b) / 2)
        if best:
            out.append(plan.claim(best[1].item("radiator", best[2], 0.8, 0.09, RADIATOR_Z, 0.55)))
    return out
