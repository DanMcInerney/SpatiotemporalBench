# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's hand pallet truck as procedural geometry.

Runs inside Blender's own Python and is never imported by `stbench`. The builder is called as in `procedural_goods.py`
(bmesh, `DESIGN_BOX`, slot dict) and builds to the box exactly: the fork tips reach +x, the loop handle -x, the outer
faces of the forks +-y, the steering wheels and rollers the floor and the top of the handle head the box's height.
Real proportions of a 2.5 t truck, shortened to the box: two forks 0.16 wide with a 0.24 gap, tapering to a rounded
nose with a pair of load rollers under each; a cross frame behind them with the pump housing on it (a tapering,
rounded casting under a domed cap), two steering wheels either side of it; and a tiller leaning back to a black handle
head with a control lever and a loop grip. The paint (the catalog colour) is the forks, the frame, the housing and the
tiller tube; the wheels, rollers, head, lever and loop are black.
"""

from __future__ import annotations

from mathutils import Matrix, Vector
from meshkit import placed, rod, solid, wheel
from partskit import Outline, loft, rings_of, rounded_path, sweep
from wobble import smoothstep

FORK_FROM = -0.40  # where the forks start, behind the frame's front edge
ROLLER_R = 0.036
STEER_R = 0.095


def fork(bm, hx: float, yc: float, body: int, wheels: int) -> None:
    """One fork: a rounded channel from the frame to a tapered, lowered nose, and the load rollers under the nose."""
    tip = hx
    half, low = 0.078, 0.02  # half the width; the underside, which rides above the floor on the rollers
    ring = Outline(2, 2, 3)

    def top(x: float) -> float:
        return 0.088 - 0.03 * smoothstep(tip - 0.2, tip, x)

    def width(x: float) -> float:
        return half - 0.028 * smoothstep(tip - 0.2, tip, x) ** 1.5

    xs = [FORK_FROM + (tip - 0.03 - FORK_FROM) * i / 14 for i in range(15)]
    rings = []
    for x in xs:
        hu, hv = width(x), (top(x) - low) / 2
        rings.append([(x, yc + u, low + hv + v) for u, v, *_ in ring.points(hu, hv, min(0.016, hv - 0.002))])
    cz = low + (top(tip) - low) / 2
    for dx, k in ((0.018, 0.93), (0.008, 0.78), (0.0, 0.5)):  # the nose rounds off in plan and in side view
        rings.append([(tip - dx, yc + (y - yc) * k, cz + (z - cz) * k) for _, y, z in rings[14]])
    loft(bm, rings, body, cap_first=True, cap_last=True)
    for side in (-1.0, 1.0):  # a pair of rollers under the nose
        wheel(bm, tip - 0.075, yc + side * 0.043, ROLLER_R, ROLLER_R, 0.05, "y", wheels, segments=16)


def housing(bm, cx: float, body: int) -> float:
    """The pump housing: a rounded casting that tapers and leans back as it rises, under a domed cap. Returns the
    height of its top."""
    out = Outline(6, 6, 4)
    z0, z1 = 0.095, 0.275

    def profile(z: float) -> tuple[float, float, float]:
        t = (z - z0) / (z1 - z0)
        return 0.105 - 0.03 * t, 0.12 - 0.035 * t, 0.045 - 0.015 * t

    stations = [z0 + (z1 - z0) * i / 8 for i in range(9)]
    rings = rings_of(out, stations, profile)
    rings = [[(cx - 0.02 * (z - z0) / (z1 - z0) + x, y, z) for x, y, z in ring] for ring in rings]
    last = rings[-1]
    mx = sum(p[0] for p in last) / len(last)
    for s, dz in ((0.82, 0.012), (0.55, 0.02), (0.25, 0.026)):
        rings.append([(mx + (x - mx) * s, y * s, z1 + dz) for x, y, _ in last])
    loft(bm, rings, body, cap_first=True, cap_last=True)
    return z1 + 0.026


def handle(bm, origin: Vector, d: Vector, body: int, wheels: int, length: float) -> None:
    """The tiller from the frame to its head: a tube, then a black head with a lever and a loop grip, in a frame whose
    z axis runs along the tube (up and back), x towards the truck and y across."""
    z = d.normalized()
    x = Vector((z.z, 0.0, -z.x)).normalized()  # perpendicular to the tube in the vertical plane, towards the truck
    if x.x < 0:
        x = -x
    m = Matrix((x, Vector((0, 1, 0)), z)).transposed().to_4x4()
    m.translation = origin

    def head() -> None:
        solid(bm, -0.025, 0.025, -0.03, 0.03, -0.01, 0.185, wheels, bevel=0.016, segments=3)  # the head
        solid(bm, 0.025, 0.045, -0.04, 0.04, 0.05, 0.14, wheels, bevel=0.007, segments=2)  # the control lever
        grip = rounded_path([(-0.02, -0.125, 0.115), (-length, -0.125, 0.115), (-length, 0.125, 0.115),
                             (-0.02, 0.125, 0.115)], 0.05, 6)
        sweep(bm, grip, 0.0105, wheels, segments=12)  # the loop grip, rear of the head

    placed(bm, m, head)


def pallet_jack(bm, box, slot) -> None:
    """A hand pallet truck: two forks with rollers at their nose, a cross frame and pump housing between steering
    wheels, and a tiller that leans back to a black head with a loop grip. Forks run along +x, the handle is at -x."""
    body, wheels = slot["body"], slot["wheels"]
    hx, hy = box[0] / 2, box[1] / 2
    for s in (-1.0, 1.0):
        fork(bm, hx, s * (hy - 0.078), body, wheels)
        wheel(bm, -0.44, s * 0.077, STEER_R, STEER_R, 0.05, "y", wheels, segments=20)  # the steering wheels
        solid(bm, -0.54, -0.34, s * 0.112 - 0.006, s * 0.112 + 0.006, 0.014, 0.205, body, bevel=0.004,
              segments=1)  # their guards
    solid(bm, -0.50, -0.33, -hy, hy, 0.03, 0.115, body, bevel=0.03, segments=4)  # the cross frame behind the forks
    top = housing(bm, -0.42, body)
    for s in (-1.0, 1.0):  # the ears the tiller hangs from
        solid(bm, -0.475, -0.395, s * 0.032 - 0.006, s * 0.032 + 0.006, top - 0.03, top + 0.045, body, bevel=0.004,
              segments=1)
    base = Vector((-0.43, 0.0, top + 0.02))
    head = Vector((-0.623, 0.0, 1.018))  # where the head starts
    d = head - base
    tube = [tuple(base + d * t) for t in (0.0, 0.5, 1.0)]
    sweep(bm, tube, 0.019, body, segments=12)
    handle(bm, head, d, body, wheels, 0.115)
    rod(bm, (-0.43, -0.04, top + 0.02), (-0.43, 0.04, top + 0.02), 0.014, wheels)  # the pivot pin
