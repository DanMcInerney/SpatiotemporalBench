# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's Phase 1 dressing: what a pallet-rack distribution centre has around its
racks that the map does not say, as `Dressing` boxes.

Plain Python (no `bpy`). `layout(spec)` takes a warehouse map's spec and returns every dressing box of the map, in a
fixed order; its few random choices are seeded by the map's name. Nothing here is an entity, a receptacle or a floor
piece: it all stands where `src/stbench/dressing_zones.py` allows (at most 0.12 m proud of a wall face, on the roof and
its services above 5 m, on rack uprights and rack ends within 0.12 m of the row, or flush with the floor), and the
rack stock the racks are already drawn full of stays as `furniture_warehouse` draws it.

A recipe is `proc:<builder>[:<text>]` or `asset:<Poly Haven id>`. The text after the builder's name is what a sign,
label or number prints (lines apart by `|`). A Dressing's size is (depth along its facing, width across, height); it
faces away from the wall it hangs on or out of the rack face it is fixed to, and an asset is fitted to that box (Poly
Haven models are width x depth x height, facing -y).

What stands where, from the plan:
- the shell: steel columns set into the walls (every 6.6 m along the long walls, on the trusses' lines; between the
  dock doors, 3.4 m apart, on the dock wall; every 5.8 m, the racks' module, on the far wall; they start above the racks
  where a rack stands against the wall, which hides their foot), roof trusses, purlins, deck and skylights (3% of the
  roof), red ESFR sprinkler mains and heads, cable trays, air ducts, a round high-bay LED fixture at each of the map's
  nine ceiling lights (its lens at the light's height: the lighting lane sets its power), insulated cladding over a
  concrete dado with a yellow safety band, and a conduit along three walls;
- at each dock door: a foam seal (two pads and a head), two bumpers, a flush leveler, a dock light and a door number;
- on each rack row: yellow upright protectors at the foot of the front uprights, a guard at the open end, a readable
  location label on every beam (aisle-bay-level), and one readable sign per aisle on the end of a row;
- fire extinguishers and call points on the walls, a utility cabinet and a power panel (an asset, its door closed) with
  a conduit drop, and security cameras high up.
"""

from __future__ import annotations

import hashlib
import math
import random

import look
from dressing_spec import Dressing
from dressing_warehouse_parts import Face, Hall, Rack, blockers, clear, mount, runs, slab
from rack_layout import LEVEL_M

AISLE = "aisle"  # the region kind of a rack aisle (`look_warehouse.KINDS`)

# -- the shell ---------------------------------------------------------------------------------------------------
# The AMR's swept body comes within 0.08 m of every wall face and rack (`dressing_robots`: its 0.10 m clearance less
# the 0.02 m the body is grown by), so what stands on the floor against one (a column, a protector, a guard) stands out
# of it by less, with 5 mm to spare.
FLOOR_PROUD_M = 0.075
COLUMN_W, COLUMN_PROUD = 0.25, FLOOR_PROUD_M  # an H-column's flange and how far it stands out of the wall
HIDES_FOOT_M = 1.0  # a piece this high (its reach: `blockers`) hides a column's foot, which then starts above it
DECK_T, PURLIN_H, PURLIN_W, PURLIN_PITCH = 0.04, 0.14, 0.08, 1.45  # the roof from the top: deck, purlins under it
TRUSS_FROM_Z, TRUSS_W, TRUSS_BAYS = 5.05, 0.25, 3  # trusses reach down to here, are this wide, and cut the hall in 3
SKYLIGHT = (1.0, 1.7)  # width x length of a roof light: six of them are 3% of the 344 m2 roof
SKYLIGHT_ROWS = (0.41, 0.75)  # their two rows, as fractions of the hall's length
# ESFR branch lines and ducts, metres from the south wall: clear of the nine lights and the trusses (checked by test)
SPRINKLER_LINES = (1.65, 5.35, 8.3, 10.8, 14.4, 17.8)
SPRINKLER_HEAD_X = (1.8, 3.0, 6)  # first head from the west wall, spacing, count
DUCT_LINES = (3.45, 15.5)
SPRINKLER_MAIN_X = 0.9  # the cross main, from the west wall
DADO_H, DADO_W, DADO_T, BAND_H, BAND_T = 1.2, 2.4, 0.03, 0.20, 0.035  # concrete dado, its yellow band at the foot
CLAD_W, CLAD_T = 1.2, 0.04  # an insulated panel's width and how proud its profile stands
CONDUIT_Z = 4.4

# -- docks, racks, fittings -----------------------------------------------------------------------------------------
SEAL_PAD_W, SEAL_HEAD_H, BUMPER_Z = 0.25, 0.30, (1.05, 1.30)
LEVELER_D = 0.38
PROTECTOR = (FLOOR_PROUD_M, 0.22, 0.40)  # depth in front of the upright, width, height
GUARD_D, GUARD_H = FLOOR_PROUD_M, 0.60  # at the open end of a row, as far out of it as a protector
LABEL = (0.004, 0.40, 0.09)  # on a beam's front face
EXTINGUISHER = (0.12, 0.14, 0.45)  # a 2 kg unit: 0.11 m across, so it fits the 0.12 m wall band
CALL_POINT = (0.026, 0.099, 0.144)  # asset fire_alarm, real size
CAMERA = (0.554, 0.17, 0.286)  # asset security_camera_01, real size
POWER_BOX = (0.1071, 0.3455, 0.5031)  # asset power_box_01 with its door closed (the library's rest pose), as built
CAMERA_Z = (5.1, 5.3)  # a camera's base, above the 5 m line, where a piece 0.55 m deep is allowed


def layout(spec: dict) -> list[Dressing]:
    """Every Phase 1 dressing box of a warehouse map. Deterministic: the same spec gives the same list. Raises
    ValueError for a map of another catalog, or when a fitting would land on furniture."""
    if spec.get("catalog") != "warehouse":
        raise ValueError(f"map {spec.get('name')!r} is not a warehouse map (catalog {spec.get('catalog')!r})")
    rng = random.Random(int.from_bytes(hashlib.sha256(spec["name"].encode()).digest()[:8], "big"))
    hall, furniture = Hall.of(spec), spec["furniture"]
    faces = hall.faces
    held = {name: blockers(face, furniture) for name, face in faces.items()}
    doors = sorted((f for f in furniture if f["kind"] == "dock_door"), key=lambda f: (f["center"][1], f["center"][0]))
    racks = [Rack(f, hall) for f in furniture if f["kind"] == "pallet_rack"]
    out = _columns(hall, faces, held, doors)
    out += _roof(hall, rng)
    out += _services(hall, spec)
    out += _walls(hall, faces, held)
    out += _dock_doors(faces, doors)
    out += _racks(spec, racks, rng)
    out += _fittings(hall, faces, held, rng)
    out += _high(hall, faces, doors, rng)
    return out


def _roof_levels(hall: Hall) -> tuple[float, float]:
    """The z of the deck's underside and of the purlins' underside."""
    deck = hall.h - DECK_T
    return deck, deck - PURLIN_H


def _columns(hall: Hall, faces: dict[str, Face], held: dict, doors: list[dict]) -> list[Dressing]:
    """Columns on the wall faces: at the corners and truss lines of the long walls, on the dock wall between its doors
    (and a half pitch beyond the outer ones), and at the module lines of the opposite wall. Where racks stand against a
    wall a column starts above them (the rack hides its foot), so none stands in a bay; a piece lower than
    `HIDES_FOOT_M` (the AMR's charger) hides nothing, and the column stands behind it on the floor."""
    w, length = hall.x1 - hall.x0, hall.y1 - hall.y0
    truss_ys = [hall.y0 + k * length / TRUSS_BAYS for k in range(1, TRUSS_BAYS)]
    along_y = [hall.y0 + COLUMN_W / 2, *truss_ys, hall.y1 - COLUMN_W / 2]
    pitch = abs(doors[1]["center"][0] - doors[0]["center"][0]) if len(doors) > 1 else 0.0
    xs = sorted(f["center"][0] for f in doors)
    piers = [xs[0] - pitch / 2, *((a + b) / 2 for a, b in zip(xs, xs[1:], strict=False)), xs[-1] + pitch / 2] \
        if doors else []
    spots = {"S": piers, "N": [hall.x0 + k * w / 3 for k in (1, 2)], "W": along_y, "E": along_y}
    _, top = _roof_levels(hall)
    out = []
    for name, face in faces.items():
        for a in spots[name]:
            a0, a1 = a - COLUMN_W / 2, a + COLUMN_W / 2
            base = max((h for lo, hi, h in held[name] if lo < a1 and a0 < hi and h >= HIDES_FOOT_M), default=0.0)
            out.append(mount(face, a0, a1, base, top, 0.0, COLUMN_PROUD, "column", "proc:steel_column"))
    return out


def _roof(hall: Hall, rng: random.Random) -> list[Dressing]:
    """Trusses across the hall, purlins along it, the deck over them and the roof lights under the deck. The purlin
    lines are a pitch apart with the aisle centres between them, so no purlin crosses a fixture."""
    w, length = hall.x1 - hall.x0, hall.y1 - hall.y0
    deck, purlin = _roof_levels(hall)
    out = [slab("roof_truss", "proc:roof_truss", hall.x0 + COLUMN_PROUD, hall.x1 - COLUMN_PROUD,
                y - TRUSS_W / 2, y + TRUSS_W / 2, TRUSS_FROM_Z, purlin)
           for y in (hall.y0 + k * length / TRUSS_BAYS for k in range(1, TRUSS_BAYS))]
    n = round(w / PURLIN_PITCH)
    out += [slab("purlin", "proc:purlin", x - PURLIN_W / 2, x + PURLIN_W / 2, hall.y0 + CLAD_T, hall.y1 - CLAD_T,
                 purlin, deck)
            for x in (hall.x0 + (k + 0.5) * w / n for k in range(n))]
    out.append(slab("roof_deck", "proc:roof_deck", hall.x0, hall.x1, hall.y0, hall.y1, deck, hall.h))
    for row in SKYLIGHT_ROWS:
        y = hall.y0 + row * length + rng.uniform(-0.2, 0.2)
        for j in range(3):  # every fourth bay between the purlins: the aisle centres of the map
            x = hall.x0 + (2 + 4 * j) * w / n
            out.append(slab("skylight", "proc:skylight", x - SKYLIGHT[0] / 2, x + SKYLIGHT[0] / 2,
                            y - SKYLIGHT[1] / 2, y + SKYLIGHT[1] / 2, deck - 0.02, deck))
    return out


def _services(hall: Hall, spec: dict) -> list[Dressing]:
    """Overhead, in the order of their levels: air ducts (5.08-5.53), cable trays (5.56-5.64), ESFR pipes (5.62-5.79)
    with their heads under them, and the high-bay fixtures at the map's lights."""
    w = hall.x1 - hall.x0
    out = [slab("air_duct", "proc:air_duct", hall.x0 + 1.7, hall.x1 - 0.8, hall.y0 + dy - 0.225, hall.y0 + dy + 0.225,
                5.075, 5.525) for dy in DUCT_LINES]
    out += [slab("cable_tray", "proc:cable_tray", x - 0.15, x + 0.15, hall.y0 + 0.6, hall.y1 - 0.6, 5.56, 5.64)
            for x in (hall.x0 + k * w / 3 for k in (1, 2))]
    mx = hall.x0 + SPRINKLER_MAIN_X
    out.append(slab("sprinkler_main", "proc:sprinkler_pipe", mx - 0.085, mx + 0.085, hall.y0 + 0.5, hall.y1 - 0.5,
                    5.62, 5.79))
    first, step, count = SPRINKLER_HEAD_X
    for dy in SPRINKLER_LINES:
        y = hall.y0 + dy
        out.append(slab("sprinkler_main", "proc:sprinkler_pipe", mx, hall.x1 - 0.6, y - 0.038, y + 0.038, 5.68, 5.756))
        out += [slab("sprinkler_head", "proc:sprinkler_head", x - 0.045, x + 0.045, y - 0.045, y + 0.045, 5.52, 5.68)
                for x in (hall.x0 + first + step * k for k in range(count))]
    deck, _ = _roof_levels(hall)
    out += [slab("high_bay_light", "proc:high_bay_light", c["pos"][0] - 0.2, c["pos"][0] + 0.2, c["pos"][1] - 0.2,
                 c["pos"][1] + 0.2, c["pos"][2], min(c["pos"][2] + 0.16, deck)) for c in spec["ceiling_lights"]]
    return out


def _panels(a: float, b: float, width: float) -> list[tuple[float, float]]:
    n = max(1, math.ceil((b - a) / width - 1e-9))
    return [(a + (b - a) * i / n, a + (b - a) * (i + 1) / n) for i in range(n)]


def _walls(hall: Hall, faces: dict[str, Face], held: dict) -> list[Dressing]:
    """Insulated cladding up to the deck and, where nothing stands against the wall, a concrete dado with a yellow band
    at its foot below it. Wherever furniture stands against a face the cladding starts above what it holds and the
    dado is left out. A conduit runs along the three walls above the racks and the dock doors."""
    deck, _ = _roof_levels(hall)
    out = []
    for name, face in faces.items():
        for a, b, top in runs(face, held[name]):
            if b - a < 0.1:
                continue
            if top < DADO_H:
                for p, q in _panels(a, b, DADO_W):
                    out.append(mount(face, p, q, 0.0, DADO_H, 0.0, DADO_T, "wall_cladding", "proc:wall_dado"))
                    out.append(mount(face, p, q, 0.0, BAND_H, 0.0, BAND_T, "safety_band", "proc:safety_band"))
            out += [mount(face, p, q, max(top, DADO_H), deck, 0.0, CLAD_T, "wall_cladding", "proc:wall_cladding")
                    for p, q in _panels(a, b, CLAD_W)]
    for name in ("S", "W", "E"):
        face = faces[name]
        if not clear(face, face.lo + 0.3, face.hi - 0.3, CONDUIT_Z, held[name]):
            raise ValueError(f"the conduit on the {name} wall meets furniture")
        out.append(mount(face, face.lo + 0.3, face.hi - 0.3, CONDUIT_Z, CONDUIT_Z + 0.04, CLAD_T, CLAD_T + 0.04,
                         "cable_run", "proc:conduit"))
    return out


def _dock_doors(faces: dict[str, Face], doors: list[dict]) -> list[Dressing]:
    """Everything round a dock door, from the west: seal pads either side and a deeper head over it (inside the door's
    footprint column), a bumper under each pad, a flush leveler plate in front, a dock light and the door's number."""
    out = []
    for number, f in enumerate(doors, 1):
        face = min(faces.values(), key=lambda fc: abs(f["center"][1 - fc.axis] - fc.pos))
        c, (t, w, h) = f["center"][face.axis], f["size"]
        lo, hi = c - w / 2, c + w / 2
        for a0, a1 in ((lo - SEAL_PAD_W, lo), (hi, hi + SEAL_PAD_W)):
            out.append(mount(face, a0, a1, BUMPER_Z[1], h + SEAL_HEAD_H, 0.0, 0.12, "dock_door_seal",
                             "proc:dock_seal_pad"))
            out.append(mount(face, a0, a1, BUMPER_Z[0], BUMPER_Z[1], 0.0, 0.12, "dock_bumper", "proc:dock_bumper"))
        out.append(mount(face, lo, hi, h, h + SEAL_HEAD_H, 0.0, t + 0.11, "dock_door_seal", "proc:dock_seal_head"))
        (x, y), (nx, ny), d = face.point(c), face.normal, t + LEVELER_D / 2  # in front of the door's thickness
        out.append(Dressing("dock_leveler", "proc:dock_leveler", (x + nx * d, y + ny * d, 0.0), face.yaw_deg,
                            (LEVELER_D, w - 0.2, 0.004)))
        out.append(mount(face, c - 0.11, c + 0.11, h + 0.5, h + 0.7, 0.0, 0.12, "dock_light", "proc:dock_light"))
        out.append(mount(face, c - 0.2, c + 0.2, h + 0.85, h + 1.21, CLAD_T, CLAD_T + 0.01, "door_number",
                         f"proc:door_number:{number}"))
    return out


def _aisles(spec: dict, racks: list[Rack]) -> list[tuple[str, list[Rack]]]:
    """(letter, rows west to east) per aisle region (a region of the kind "aisle", whatever its name), the letters
    running from the west."""
    regions = sorted((r for r in spec["regions"] if look.kind_of(r) == AISLE), key=lambda r: r["anchor"][0])
    out = []
    for letter, r in zip("ABCDEFGH", regions, strict=False):
        xs, ys = [p[0] for p in r["polygon"]], [p[1] for p in r["polygon"]]
        rows = [k for k in racks if min(xs) <= k.cx <= max(xs) and min(ys) <= k.cy <= max(ys)]
        out.append((letter, sorted(rows, key=lambda k: k.cx)))
    return out


def _racks(spec: dict, racks: list[Rack], rng: random.Random) -> list[Dressing]:
    """Per row: a protector on each front upright (a plate in front of the post, 0.1 m out of the rack plane, so it
    never enters a bay) and a guard across the open end. Per beam: its location label (aisle letter, bay number, level
    number, as `A-03-02`: odd bays on the west side of an aisle, even on the east, level 01 is the floor). Per aisle:
    its sign on the end of one row."""
    out = []
    for k in racks:
        for i in range(k.bays + 1):
            yc = min(max(k.post_y(i), -k.hy + PROTECTOR[1] / 2), k.hy - PROTECTOR[1] / 2)
            out.append(k.item("rack_upright_protector", "proc:upright_protector", k.hx, k.hx + PROTECTOR[0],
                              yc - PROTECTOR[1] / 2, yc + PROTECTOR[1] / 2, 0.0, PROTECTOR[2]))
        if k.open_end:
            e = k.open_end
            y0, y1 = sorted((e * k.hy, e * (k.hy + GUARD_D)))
            out.append(k.item("end_of_aisle_guard", "proc:end_guard", -k.hx + 0.005, k.hx - 0.005, y0, y1, 0.0,
                              GUARD_H, turn=e))
    for letter, rows in _aisles(spec, racks):
        for side, k in enumerate(rows):
            for bay in range(k.bays):
                for level in range(1, k.levels):
                    z, yb = level * LEVEL_M, k.bay_y(bay)
                    code = f"{letter}-{2 * (k.bay_number(bay) - 1) + 1 + side % 2:02d}-{level + 1:02d}"
                    out.append(k.item("beam_label", f"proc:beam_label:{code}", k.hx - 0.02, k.hx - 0.02 + LABEL[0],
                                      yb - LABEL[1] / 2, yb + LABEL[1] / 2, z - 0.105, z - 0.105 + LABEL[2]))
        if not any(r.open_end for r in rows):
            continue
        k = rng.choice([r for r in rows if r.open_end])
        e, bays = k.open_end, k.bays
        y0, y1 = sorted((e * k.hy, e * (k.hy + 0.02)))
        out.append(k.item("aisle_sign", f"proc:aisle_sign:AISLE {letter}|BAYS 01-{2 * bays:02d}", -0.45, 0.45, y0, y1,
                          2.0, 2.45, turn=e))
    return out


def _fittings(hall: Hall, faces: dict[str, Face], held: dict, rng: random.Random) -> list[Dressing]:
    """Extinguishers with a call point beside each, a utility cabinet and a power panel with its conduit drop, on
    stretches of wall where nothing stands, in the dock (the extinguishers are 2 kg units, so they fit the 0.12 m wall
    band: Poly Haven's 6 kg one is 0.37 m deep)."""
    # the side walls' spots stand just north of the AMR's stations (which end at y 7.4 and 7.5 on `warehouse_a`) and
    # short of the cabinet and panel at y 8.6, with the call point 0.4 m beyond each
    spots = (("W", hall.y0 + 7.7), ("E", hall.y0 + 7.7), ("S", hall.x0 + 1.0), ("S", hall.x1 - 1.0),
             ("N", hall.x0 + 9.7))
    out = []
    for name, a in spots:
        step = 0.4 if a < (faces[name].lo + faces[name].hi) / 2 else -0.4
        for cls, recipe, along, size, z0, lo in (
                ("fire_extinguisher", "proc:fire_extinguisher", a, EXTINGUISHER, 1.0 + rng.uniform(0.0, 0.1), 0.0),
                ("fire_alarm", "asset:fire_alarm", a + step, CALL_POINT, 1.3 + rng.uniform(0.0, 0.1), CLAD_T)):
            out.append(_on_wall(faces[name], held[name], along, size[1], z0, size[2], lo, lo + size[0], cls, recipe))
    a = hall.y0 + 8.5
    out.append(_on_wall(faces["W"], held["W"], a, 0.5, 0.9, 0.8, 0.0, 0.12, "utility_enclosure",
                        "proc:utility_enclosure"))
    out.append(_on_wall(faces["E"], held["E"], a, POWER_BOX[1], 1.0, POWER_BOX[2], 0.0, POWER_BOX[0], "power_panel",
                        "asset:power_box_01"))
    out.append(_on_wall(faces["E"], held["E"], a, 0.04, 1.0 + POWER_BOX[2], CONDUIT_Z - 1.0 - POWER_BOX[2], CLAD_T,
                        CLAD_T + 0.04, "cable_run", "proc:conduit"))  # up to the conduit along the wall
    return out


def _on_wall(face: Face, held: list, a: float, width: float, z0: float, height: float, lo: float, hi: float, cls: str,
             recipe: str) -> Dressing:
    """A wall fitting centred at `a` along the face; it must be above whatever stands against the face there."""
    if not clear(face, a - width / 2, a + width / 2, z0, held):
        raise ValueError(f"a {cls} on the {face.name} wall at {a:.2f} m meets furniture")
    return mount(face, a - width / 2, a + width / 2, z0, z0 + height, lo, hi, cls, recipe)


def _high(hall: Hall, faces: dict[str, Face], doors: list[dict], rng: random.Random) -> list[Dressing]:
    """What hangs on the walls above the 5 m line, where depth is free: cameras (down each aisle from the far wall, over
    the dock from the dock wall, across it from the side walls)."""
    n, w = round((hall.x1 - hall.x0) / PURLIN_PITCH), hall.x1 - hall.x0
    aisle_x = [hall.x0 + (2 + 4 * j) * w / n for j in range(3)]
    dock_x = sorted(f["center"][0] for f in doors)[1::2]
    cameras = [("N", a) for a in aisle_x] + [("S", a) for a in dock_x] + [("W", hall.y0 + 7.9), ("E", hall.y0 + 7.9)]
    out = []
    for name, a in cameras:
        z = rng.uniform(*CAMERA_Z)
        out.append(mount(faces[name], a - CAMERA[1] / 2, a + CAMERA[1] / 2, z, z + CAMERA[2], 0.0, CAMERA[0],
                         "security_camera", "asset:security_camera_01"))
    return out
