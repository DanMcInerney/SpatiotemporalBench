"""The volumes render-only dressing may not meet because a robot drives there.

A robot sees dressing in its depth and picture, and plans its path without it, so dressing that stands where its body
goes shows depth an obstacle no entity explains. For each ground robot planned on a catalog's maps
(`PLANNED_ROBOTS`, from design §4) this module gives:
- its **swept body**: the union of its footprint, grown by `SWEPT_GROW_M`, over every base pose that keeps its
  `clearance_m` from the wall pieces and blocking fixed footprints (a rectangle counts as its circumscribed disc), from
  its `climb_m` up to its height plus `SWEPT_GROW_M`. Poses are `open_floor` at the disc's radius plus the clearance,
  so the body comes within `clearance_m - SWEPT_GROW_M` of a wall: 0 m for the vacuum, 0.08 m for the helper and the
  AMR;
- the **underside of each fixed piece it passes under** (`robots.passes_under`), from `FLUSH_M` up to the piece's
  `under_m`, outside its leg squares (`geometry.under_obstacles`): the knee space `footprint_bands` allows dressing
  into (its premise, that nothing tracked is there, is false once a robot drives under), where a 0.09 m box on the floor
  would show as an obstacle on the vacuum's path (the review's V-7).
Beside them, the **keep-outs of the places a robot's manoeuvre uses** (`keep_out_zones`), by piece kind: a `bay`'s
footprint and a `charger`'s apron, up to `zones.TOP_M`. Dressing is no entity and no state, so a bay it stands in would
read blocked in every case (a fact of the layout that no world explains, which a layout-learning agent answers from),
and a thing on the apron would stand in the way of every undock.
Trim (at most `TRIM_PROUD_M` proud of a wall, the casings' real value, and `TRIM_SLACK_M` for rounding) and a
radiator whose underside is at least `RADIATOR_FROM_M` up are exempt from the swept bodies
(`dressing_zones.DressingGuard`); a flush plate or grime no thicker than a robot's climb lies under its body anyway.
"""

from __future__ import annotations

import math

from .dressing_boxes import FLUSH_M, Zone, open_floor, square
from .geometry import Rect, under_obstacles
from .mapspec import MapSpec
from .robots import ROBOTS, Robot, passes_under
from .zones import TOP_M

# The robots the design plans on each catalog's maps, by `robots.ROBOTS` key: `clutter.place` keeps the floor connected
# for each of them (`clutter_reach`) and the dressing guard keeps dressing out of their swept bodies. A catalog that is
# not here has none planned and keeps `dressing_zones.RETIRED_CAMERA_HEIGHTS_M`'s zones until its lane adds them.
PLANNED_ROBOTS: dict[str, tuple[str, ...]] = {"home": ("vacuum", "helper"), "warehouse": ("amr",)}
SWEPT_GROW_M = 0.02  # the swept body is the footprint and the top grown by this
TRIM_PROUD_M = 0.015  # baseboards and casings stand at most this far from the wall (the casings' real value)
TRIM_SLACK_M = 0.001  # a layout's coordinates are rounded and stand 0.5 mm off the wall face: trim has this much more
RADIATOR_FROM_M = 0.15  # a radiator with its underside this far up is exempt: the vacuum drives under it
EXEMPT_ABOVE_M = {"radiator": RADIATOR_FROM_M}  # dressing class -> the underside from which the swept bodies ignore it
APRON_DEEP_M, APRON_SIDE_M = 1.6, 0.5  # a charger's apron: out of its front face, and beyond each side of it


def planned_robots(catalog: str) -> list[Robot]:
    """The ground robots planned on a catalog's maps, in the order of `PLANNED_ROBOTS`."""
    return [ROBOTS[r] for r in PLANNED_ROBOTS.get(catalog, ()) if not ROBOTS[r].flies]


def disc_radius_m(robot: Robot) -> float:
    """A disc's radius; a rectangle is planned as its circumscribed disc (`route.robot_free_space`)."""
    body = robot.body
    return body.length_m / 2 if body.shape == "disc" else math.hypot(body.length_m, body.width_m) / 2


def swept_zones(m: MapSpec, catalog: str) -> list[Zone]:
    """The swept body of each planned robot, as boxes. A rectangle of its pose floor (`open_floor` at the disc's
    radius plus its clearance) grown by the radius and `SWEPT_GROW_M` is a little too much at a solid's corner, where
    `open_floor` rounds nothing, so each is cut to the floor at least `clearance_m - SWEPT_GROW_M` from every wall and
    blocking footprint, which no part of the body is nearer than. Every box of a robot has one reason."""
    out: list[Zone] = []
    for robot in planned_robots(catalog):
        body, grown = robot.body, disc_radius_m(robot) + SWEPT_GROW_M
        z0, z1 = body.climb_m, body.height_m + SWEPT_GROW_M
        why = f"swept body of the {robot.id} (z {z0:.2f}-{z1:.2f} m)"
        near = open_floor(m, max(body.clearance_m - SWEPT_GROW_M, 0.0))
        for x0, y0, x1, y1 in open_floor(m, disc_radius_m(robot) + body.clearance_m):
            for a0, b0, a1, b1 in near:
                lo_x, hi_x = max(x0 - grown, a0), min(x1 + grown, a1)
                lo_y, hi_y = max(y0 - grown, b0), min(y1 + grown, b1)
                if hi_x - lo_x > 1e-9 and hi_y - lo_y > 1e-9:
                    out.append((why, (lo_x, lo_y, z0, hi_x, hi_y, z1)))
    return out


def under_zones(m: MapSpec, catalog: str) -> list[Zone]:
    """The open underside of each fixed blocking piece a planned robot passes under: its footprint from `FLUSH_M` up
    to `under_m`, less the four leg squares, as three boxes (a middle band across the piece, and the two bands between
    the legs along its ends). A piece with no room between its legs leaves none."""
    robots, out = planned_robots(catalog), []
    for f in m.furniture:
        if f.mobility != "fixed" or not f.blocks or not any(passes_under(r, f.under_m) for r in robots):
            continue
        x0, y0, x1, y1 = square(f.footprint(), f"furniture {f.id}")
        legs = [square(r, "a leg") for r in under_obstacles(f.footprint(), f.under_m)]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        # where the legs on the low side end and those on the high side begin; a side the legs fill has none between
        xa = max((b[2] for b in legs if (b[0] + b[2]) / 2 < cx), default=x1)
        xb = min((b[0] for b in legs if (b[0] + b[2]) / 2 > cx), default=x0)
        ya = max((b[3] for b in legs if (b[1] + b[3]) / 2 < cy), default=y1)
        yb = min((b[1] for b in legs if (b[1] + b[3]) / 2 > cy), default=y0)
        why = f"open underside of {f.name} {f.id} (up to {f.under_m:.2f} m)"
        boxes = [(x0, ya, x1, yb), (xa, y0, xb, ya), (xa, yb, xb, y1)]
        out += [(why, (bx0, by0, FLUSH_M, bx1, by1, f.under_m)) for bx0, by0, bx1, by1 in boxes
                if bx1 - bx0 > 1e-9 and by1 - by0 > 1e-9]
    return out


def keep_out_zones(m: MapSpec) -> list[Zone]:
    """The volume of each place a robot's manoeuvre uses that dressing may not enter, by the kind of the fixed piece
    that marks it: a `bay`'s footprint, and a `charger`'s apron (`APRON_DEEP_M` out of its front, which faces along its
    yaw, on its width and `APRON_SIDE_M` beyond each side), each from the floor to `zones.TOP_M`. Piece kinds, never
    names or maps, so another warehouse's bays and chargers are kept clear the same way."""
    out: list[Zone] = []
    for f in (f for f in m.furniture if f.mobility == "fixed" and f.kind in ("bay", "charger")):
        if f.kind == "bay":
            rect, why = f.footprint(), f"bay {f.id}"
        else:
            ahead = (f.size[0] + APRON_DEEP_M) / 2  # from its centre to the middle of the apron
            rect = Rect(f.center[0] + ahead * math.cos(f.yaw), f.center[1] + ahead * math.sin(f.yaw), APRON_DEEP_M,
                        f.size[1] + 2 * APRON_SIDE_M, f.yaw)
            why = f"apron of charger {f.id}"
        x0, y0, x1, y1 = square(rect, f"furniture {f.id}")
        out.append((f"keep-out: {why} (up to {TOP_M:g} m)", (x0, y0, 0.0, x1, y1, TOP_M)))
    return out
