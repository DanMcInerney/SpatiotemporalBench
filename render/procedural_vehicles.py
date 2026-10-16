# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's vehicles as procedural geometry: a car, a van, a pickup and a box truck (carried kinds), a taxi and a
bus (the movers).

Runs inside Blender's own Python and is never imported by `stbench`. Builders are called as in `procedural_goods.py`
(bmesh, `DESIGN_BOX`, slot dict) and build in the vehicle's local frame (origin at the base centre, x forward, y left, z
up); `procedural.build` then fits the result to the catalog box exactly. The colour attribute is the `body` slot. Bodies
are lofts of rounded sections with the glass painted on by rule; wheels (with a five-spoke rim), lights, mirrors and
bumpers are slabs. Detail is geometry only, inside the same boxes and in the same slots (`trim` for seams and handles,
`hub` for the blank plates, which carry no number): panel gaps and door handles on the flat sides, a grille, a plate at
each end, ribs on the truck's box. The colour attribute stays the `body` slot, and body panels stay the biggest thing.
"""

from __future__ import annotations

from vehicle_kit import interp, loft, normals, paint, ring, solid, to_box, wheels


def _body_rule(S, glass_x, belt, roof_z, hw_top, pillars=(), nose_x=None, nose_slot=None, low=0.34, rail=0.1):
    """A paint rule for a cab loft: glass on the windscreen, rear window and side windows, body elsewhere."""
    def rule(c, n):
        if nose_x is not None and n.x > 0.9 and c.x > nose_x:
            return S[nose_slot]
        if c.z < low:
            return S["trim"]
        if n.z > 0.9:
            return S["body"]
        gx0, gx1 = glass_x
        if not gx0 <= c.x <= gx1:
            return None
        if abs(n.y) < 0.55:  # windscreen and rear window
            return S["glass"] if abs(c.y) < hw_top(c.x) - 0.10 and c.z > belt else S["body"]
        if c.z < belt or c.z > roof_z(c.x) - rail or any(a <= c.x <= b for a, b in pillars):
            return S["body"]
        return S["glass"]
    return rule


def _plates(bm, S, front: tuple[float, float, float], rear: tuple[float, float, float]) -> None:
    """A blank plate (0.52 x 0.11 m) on the front and the rear: each is (x of the bumper face, z of its foot, the side
    the plate faces: +1 front, -1 rear)."""
    for x, z, side in (front, rear):
        solid(bm, x, x + side * 0.006, -0.26, 0.26, z, z + 0.11, S["hub"])


def _seams(bm, S, xs, z0: float, z1: float, y: float) -> None:
    """Shut lines of the doors on both sides: thin dark slabs at each x of `xs` on a flat side wall at |y| = `y`."""
    for s in (-1.0, 1.0):
        for x in xs:
            solid(bm, x, x + 0.012, s * (y - 0.012), s * (y + 0.006), z0, z1, S["trim"])


def _handles(bm, S, xs, z: float, y: float) -> None:
    """Door handles on both sides: dark bars 14 cm long at each x of `xs`, `z` high, on a flat wall at |y| = `y`."""
    for s in (-1.0, 1.0):
        for x in xs:
            solid(bm, x, x + 0.14, s * (y - 0.01), s * (y + 0.014), z, z + 0.03, S["trim"], bevel=0.004, segments=1)


def car(bm, box, S) -> None:
    """A three-box saloon: a rounded tub (hood, doors, boot) under a tapering greenhouse of dark glass with a
    body-colour roof, B pillar and roof rails; four wheels with dark arches, mirrors, head and tail lights, a dark
    grille and bumpers."""
    lo = [(-2.30, .68, .68, .45, .88, .14), (-2.22, .80, .80, .30, .93, .18), (-2.00, .85, .85, .24, .96, .20),
          (-1.00, .86, .86, .22, .98, .20), (.80, .86, .86, .22, .97, .20), (1.70, .85, .85, .24, .92, .20),
          (2.15, .81, .81, .28, .80, .17), (2.30, .70, .70, .40, .70, .13)]
    faces = loft(bm, [(x, ring(a, b, z0, z1, r, 3)) for x, a, b, z0, z1, r in lo], S["body"])
    cab = [(-1.45, .78, .72, .98), (-1.05, .79, .68, 1.33), (-.80, .79, .66, 1.43), (-.55, .79, .66, 1.45),
           (-.12, .79, .66, 1.45), (.04, .79, .66, 1.45), (.35, .79, .66, 1.45), (.65, .79, .67, 1.42),
           (1.15, .80, .70, .99)]
    roof_z = interp([c[0] for c in cab], [c[3] for c in cab])
    hw_top = interp([c[0] for c in cab], [c[2] for c in cab])
    faces += loft(bm, [(x, ring(a, b, .92, z1, .10, 2, belt=1.02)) for x, a, b, z1 in cab], S["body"])
    normals(bm)
    paint(faces, _body_rule(S, (-1.46, 1.16), 1.03, roof_z, hw_top, pillars=((-.12, .04),), nose_x=2.2,
                            nose_slot="trim"))
    wheels(bm, S, (1.45, -1.40), .86, .33, .22)
    for s in (-1.0, 1.0):
        solid(bm, 2.20, 2.32, s * .78, s * .45, .58, .68, S["light_f"], bevel=.01, segments=1)
        solid(bm, -2.33, -2.20, s * .78, s * .40, .74, .86, S["light_r"], bevel=.01, segments=1)
        solid(bm, .60, .78, s * .84, s * .90, .98, 1.06, S["body"], bevel=.02, segments=1)  # mirror
    solid(bm, 2.28, 2.33, -.62, .62, .36, .50, S["trim"])  # bumper valance
    solid(bm, -2.31, -2.26, -.62, .62, .34, .46, S["trim"])
    _seams(bm, S, (-1.12, -.12, .98), .50, .78, .86)  # the doors' shut lines on the flat side
    _handles(bm, S, (-.98, .02), .72, .86)
    solid(bm, 2.27, 2.31, -.48, .48, .56, .66, S["trim"], bevel=.01, segments=1)  # grille
    _plates(bm, S, (2.33, .385, 1), (-2.31, .345, -1))


def taxi(bm, box, S) -> None:
    """The car with a lit roof sign and a black-and-white checker band along each door."""
    car(bm, box, S)
    solid(bm, -.12, .18, -.27, .27, 1.44, 1.47, S["trim"])
    solid(bm, -.12, .18, -.34, .34, 1.46, 1.55, S["sign"], bevel=.03, segments=2)  # the lit sign
    for s in (-1.0, 1.0):
        for i in range(14):  # two rows of 14 squares between the wheel arches
            x = -.98 + .14 * i
            for j in range(2):
                z = .70 + .07 * j
                slot = S["check_a"] if (i + j) % 2 else S["check_b"]
                solid(bm, x, x + .14, s * .858, s * .872, z, z + .07, slot)


def van(bm, box, S) -> None:
    """A panel van with a high roof: one rounded shell from the nose over a sloping windscreen to a flat roof and a near
    vertical tail, dark front door windows, a sliding-door groove, rear door windows, wheels with arches."""
    st = [(-2.60, .78, .76, .50, 1.95, .15), (-2.52, .92, .88, .32, 2.07, .18), (-2.30, .96, .92, .28, 2.10, .20),
          (.95, .96, .92, .28, 2.10, .20), (1.25, .96, .92, .28, 2.10, .20), (1.45, .96, .90, .28, 2.03, .20),
          (1.95, .96, .88, .28, 1.28, .18), (2.30, .94, .86, .30, 1.08, .16), (2.52, .88, .80, .36, .98, .14),
          (2.60, .76, .72, .48, .88, .12)]
    faces = loft(bm, [(x, ring(a, b, z0, z1, r, 3, belt=1.2)) for x, a, b, z0, z1, r in st], S["body"])
    normals(bm)
    roof_z = interp([s[0] for s in st], [s[4] for s in st])
    hw_top = interp([s[0] for s in st], [s[2] for s in st])
    paint(faces, _body_rule(S, (.94, 2.0), 1.21, roof_z, hw_top, nose_x=2.5, nose_slot="trim", low=.40, rail=.08))
    wheels(bm, S, (1.65, -1.55), .96, .36, .24)
    for s in (-1.0, 1.0):
        solid(bm, -.62, -.60, s * .955, s * .975, .42, 1.95, S["trim"])  # the sliding door's rear edge
        solid(bm, .78, .80, s * .955, s * .975, .42, 1.95, S["trim"])
        solid(bm, 2.52, 2.62, s * .70, s * .40, .52, .62, S["light_f"], bevel=.01, segments=1)
        solid(bm, -2.64, -2.58, s * .92, s * .78, .70, 1.45, S["light_r"], bevel=.01, segments=1)
        solid(bm, 1.50, 1.68, s * .94, s * 1.00, 1.28, 1.40, S["body"], bevel=.02, segments=1)  # mirror
        solid(bm, -2.64, -2.60, s * .04, s * .62, 1.25, 1.75, S["glass"])  # rear door windows
    solid(bm, 2.56, 2.64, -.74, .74, .38, .52, S["trim"])
    solid(bm, -2.62, -2.56, -.90, .90, .34, .50, S["trim"])
    _handles(bm, S, (.55, 1.04), .98, .955)
    solid(bm, 2.52, 2.62, -.46, .46, .62, .74, S["trim"], bevel=.01, segments=1)  # grille
    _plates(bm, S, (2.64, .385, 1), (-2.62, .345, -1))


def pickup(bm, box, S) -> None:
    """A full-size pickup: a raised cab and hood (one loft with a sloping windscreen), an open bed with a dark liner,
    tailgate and wheel-arch humps, chunky tyres and bumpers."""
    cab = [(-.55, .94, .84, 1.80, .15), (-.45, .94, .84, 1.90, .18), (.60, .94, .84, 1.90, .18),
           (.90, .94, .84, 1.80, .18), (1.35, .94, .86, 1.40, .18), (1.55, .94, .88, 1.30, .18),
           (2.45, .93, .86, 1.22, .18), (2.62, .88, .82, 1.12, .15), (2.70, .78, .74, 1.0, .12)]
    faces = loft(bm, [(x, ring(a, b, .32, z1, r, 3, belt=1.38)) for x, a, b, z1, r in cab], S["body"])
    normals(bm)
    roof_z = interp([c[0] for c in cab], [c[4] for c in cab])
    hw_top = interp([c[0] for c in cab], [c[2] for c in cab])
    paint(faces, _body_rule(S, (-.50, 1.40), 1.40, roof_z, hw_top, pillars=((-.12, .06),), nose_x=2.6,
                            nose_slot="trim", low=.46, rail=.10))
    wheels(bm, S, (1.85, -1.75), .94, .40, .28)
    solid(bm, -2.70, -.60, -.94, .94, .32, .84, S["body"], bevel=.03)  # the bed's tub
    solid(bm, -2.64, -.66, -.86, .86, .83, .88, S["liner"])
    for s in (-1.0, 1.0):
        solid(bm, -2.70, -.60, s * .94, s * .84, .84, 1.20, S["body"], bevel=.02, segments=1)  # bed walls
        solid(bm, -2.30, -1.20, s * .86, s * .60, .86, 1.02, S["liner"], bevel=.02, segments=1)  # arch humps
        solid(bm, 2.60, 2.72, s * .72, s * .40, .60, .70, S["light_f"], bevel=.01, segments=1)
        solid(bm, -2.74, -2.66, s * .92, s * .78, .66, .86, S["light_r"], bevel=.01, segments=1)
        solid(bm, .95, 1.12, s * .96, s * 1.02, 1.46, 1.58, S["body"], bevel=.02, segments=1)  # mirror
    solid(bm, -2.70, -.60, -.94, -.84, .84, 1.20, S["body"])
    solid(bm, -.74, -.60, -.94, .94, .84, 1.36, S["body"], bevel=.03)  # headboard
    solid(bm, -2.76, -2.68, -.94, .94, .84, 1.20, S["body"], bevel=.02, segments=1)  # tailgate
    solid(bm, -2.78, -2.66, -.98, .98, .36, .58, S["trim"], bevel=.03)  # rear bumper
    solid(bm, 2.62, 2.74, -.98, .98, .38, .62, S["trim"], bevel=.03)
    solid(bm, -.56, -.545, -.62, .62, 1.45, 1.78, S["glass"])  # rear window
    _handles(bm, S, (-.08, .72), .98, .92)
    solid(bm, 2.62, 2.7, -.55, .55, .70, .82, S["trim"], bevel=.01, segments=1)  # grille
    solid(bm, -2.76, -2.68, -.30, .30, .94, .99, S["trim"], bevel=.01, segments=1)  # tailgate handle
    _plates(bm, S, (2.74, .42, 1), (-2.78, .40, -1))


def box_truck(bm, box, S) -> None:
    """A medium box truck: a cab-over cab (loft with a near-vertical windscreen and dark side windows) in front of a
    tall square cargo box with a rear door seam, on a dark chassis with a rear bumper, tail lights and big mirrors."""
    cab = [(1.95, 1.04, .98, 2.70, .15), (2.05, 1.10, 1.02, 2.78, .18), (2.90, 1.10, 1.02, 2.78, .18),
           (3.15, 1.10, 1.00, 2.62, .18), (3.50, 1.08, .98, 1.95, .18), (3.68, 1.05, .96, 1.55, .15),
           (3.75, .95, .90, 1.45, .12)]
    faces = loft(bm, [(x, ring(a, b, .60, z1, r, 3, belt=1.50)) for x, a, b, z1, r in cab], S["body"])
    normals(bm)
    roof_z = interp([c[0] for c in cab], [c[4] for c in cab])
    hw_top = interp([c[0] for c in cab], [c[2] for c in cab])
    paint(faces, _body_rule(S, (2.10, 3.60), 1.52, roof_z, hw_top, nose_x=3.7, nose_slot="trim", low=.80, rail=.10))
    wheels(bm, S, (2.70, -2.35), 1.25, .45, .34, arch=True)
    solid(bm, -3.75, 1.90, -1.25, 1.25, .95, 3.30, S["body"], bevel=.06)  # the cargo box
    solid(bm, -3.60, 3.40, -.60, .60, .50, .95, S["trim"])  # chassis
    solid(bm, -3.88, -3.72, -1.15, 1.15, .55, .82, S["trim"], bevel=.02)  # rear bumper
    solid(bm, 3.72, 3.84, -1.00, 1.00, .50, .82, S["trim"], bevel=.03)
    solid(bm, -3.78, -3.74, -.01, .01, 1.10, 3.20, S["trim"])  # rear door seam
    solid(bm, -3.78, -3.74, -1.15, 1.15, 2.55, 2.60, S["trim"])  # roll-up door's top edge
    for s in (-1.0, 1.0):
        solid(bm, -3.79, -3.74, s * 1.10, s * 1.22, 1.00, 1.60, S["light_r"], bevel=.01, segments=1)
        solid(bm, 3.70, 3.80, s * .86, s * .56, .72, .84, S["light_f"], bevel=.01, segments=1)
        solid(bm, 3.15, 3.30, s * 1.10, s * 1.25, 1.55, 2.30, S["trim"], bevel=.02, segments=1)  # mirror
        solid(bm, 1.92, 1.95, s * 1.25, s * 1.22, 1.0, 3.2, S["trim"])
    solid(bm, 1.90, 1.95, -1.22, 1.22, .95, 3.30, S["trim"])
    for s in (-1.0, 1.0):
        for k in range(8):  # the cargo box's ribs: a panel line every 0.65 m
            x = -3.4 + 0.65 * k
            solid(bm, x, x + 0.016, s * 1.244, s * 1.258, 1.05, 3.22, S["trim"])
    _plates(bm, S, (3.84, .56, 1), (-3.88, .60, -1))


def bus(bm, box, S) -> None:
    """A 9 m midibus: a long rounded shell with a continuous band of dark side windows between narrow pillars (six panes
    1.2 m wide, 1.4 m apart), glazed doors on the right side (-y) at the fourth and sixth panes, a big flat windscreen
    and destination sign, a contrast stripe under the windows, two roof units and two axles."""
    hl, n_panes, door_panes = 4.5, 6, (3, 5)
    x0 = -(n_panes * 1.4 - 0.2) / 2
    pane = [(round(x0 + 1.4 * i, 3), round(x0 + 1.4 * i + 1.2, 3)) for i in range(n_panes)]  # 0.2 m pillars
    doors = tuple(pane[i] for i in door_panes)
    xs = {-hl, -hl + .12, -hl + .4, hl - .4, hl - .12, hl}
    for a, b in pane:
        xs |= {a, b}
    shell = {-hl: (.90, .88, .60, 2.70, .22), -hl + .12: (1.12, 1.08, .50, 3.00, .22),
             -hl + .4: (1.20, 1.16, .45, 3.05, .20), hl - .4: (1.20, 1.16, .45, 3.05, .20),
             hl - .12: (1.17, 1.12, .50, 3.00, .22), hl: (1.05, 1.0, .60, 2.80, .22)}
    st = []
    for x in sorted(xs):
        hwb, hwt, z0, z1, r = shell.get(x, (1.20, 1.16, .45, 3.05, .20))
        st.append((x, ring(hwb, hwt, z0, z1, r, 2, belt=(.70, 1.12, 2.50))))
    faces = loft(bm, st, S["body"])
    normals(bm)

    def rule(c, n):
        if c.z < .55:
            return S["trim"]
        if abs(n.y) < .6 or n.z > .6 or abs(c.x) > hl - .4 or c.z > 2.52 or c.z < .72:
            return None
        if c.y < 0 and any(a <= c.x <= b for a, b in doors):
            return S["glass"]  # a door, glazed down to the step
        return S["glass"] if c.z > 1.1 and any(a <= c.x <= b for a, b in pane) else None

    paint(faces, rule)
    wheels(bm, S, (2.4, -1.9), 1.20, .50, .30)
    for ux0, ux1, hy, z1 in ((-2.2, .6, .85, 3.20), (1.8, 2.8, .6, 3.14)):  # air-conditioning and a roof hatch
        solid(bm, ux0, ux1, -hy, hy, 3.04, z1, S["roof_unit"], bevel=.06)
    for s in (-1.0, 1.0):
        solid(bm, -hl + .2, hl - .2, s * 1.203, s * 1.215, .88, 1.04, S["stripe"])  # the livery stripe
        solid(bm, hl - .45, hl - .30, s * 1.20, s * 1.275, 1.9, 2.5, S["trim"], bevel=.02, segments=1)  # mirror arm
        solid(bm, hl - .02, hl + .02, s * .90, s * .60, .70, .84, S["light_f"], bevel=.01, segments=1)
        solid(bm, -hl - .03, -hl + .03, s * .95, s * .70, .90, 1.3, S["light_r"], bevel=.01, segments=1)
    solid(bm, hl - .005, hl + .03, -.95, .95, 1.25, 2.62, S["glass"])  # the windscreen
    solid(bm, hl - .005, hl + .03, -.95, .95, 2.68, 2.88, S["trim"])  # destination sign
    solid(bm, hl, hl + .035, -.7, .7, 2.74, 2.82, S["amber"])
    solid(bm, -hl - .02, -hl + .02, -.8, .8, 1.5, 2.45, S["glass"])  # rear window
    _plates(bm, S, (hl, .66, 1), (-hl - .02, .66, -1))
    solid(bm, hl - .01, hl + .015, -.5, .5, .92, 1.08, S["trim"], bevel=.01, segments=1)  # grille


BUILDERS = {"car": to_box(car), "van": to_box(van), "pickup": to_box(pickup), "box_truck": to_box(box_truck),
            "taxi": to_box(taxi), "bus": to_box(bus)}
