# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Refined procedural geometry for the simple manufactured kinds and the movers: lathe profiles and bevelled slabs,
and the registry of every builder. The household's robots are built here and its small kinds (a glass, bottle, book,
remote, laptop, plant and laundry basket) in `procedural_home.py`; the warehouse's are in `procedural_goods.py` (and
`procedural_pallet.py`) and `procedural_equipment.py` (and `procedural_jack.py`, `procedural_forklift.py`), and the
downtown's in `procedural_vehicles.py` (a car, van, pickup, box truck, taxi and bus) and `procedural_city.py` (a skip,
dumpster, portable toilet and food cart), all of which name their material slots instead of numbering them (`_named`
hands each its dict from `SLOTS`). The household's clutter that no downloaded model covers (a floor fan, a power strip,
books on a shelf) is `procedural_clutter.py`'s, with its own boxes and slots, merged into `DESIGN_BOX` and `SLOTS` here.
The detail of the lofts and tubes is `partskit.py`'s, `meshkit.py` has the basic shapes and `wobble.py` the
deterministic unevenness.

Runs inside Blender's own Python and is never imported by `stbench`. One builder per manifest `builder` name returns a
mesh in metres with its origin at the base centre (x forward, y left, z up) and a material slot per `SLOTS` name; the
library gives each slot the material the manifest names. The mesh is fitted to `DESIGN_BOX` exactly. An entity then
scales it per axis to its own box (fit mode "exact"), so a catalog size that moves is followed without a rebuild.
`DESIGN_BOX` is the catalog's default size of the kind when this was written (the laundry basket's is the map's); a
test (`test_procedural_detail_blender.py`) keeps each builder's own geometry within a hundredth of it, so that the
exact fit never squashes a detail.
No texture coordinates: the materials project on object coordinates.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix
from meshkit import fit, lathe, placed, slab
from procedural_carried import BUILDERS as CARRIED
from procedural_carried import DESIGN_BOX as CARRIED_BOX
from procedural_carried import SLOTS as CARRIED_SLOTS
from procedural_city import BUILDERS as CITY
from procedural_clutter import BUILDERS as CLUTTER
from procedural_clutter import DESIGN_BOX as CLUTTER_BOX
from procedural_clutter import SLOTS as CLUTTER_SLOTS
from procedural_equipment import BUILDERS as EQUIPMENT
from procedural_goods import BUILDERS as GOODS
from procedural_home import BUILDERS as HOME
from procedural_hung import BUILDERS as HUNG
from procedural_hung import DESIGN_BOX as HUNG_BOX
from procedural_hung import SLOTS as HUNG_SLOTS
from procedural_kitchen import BUILDERS as KITCHEN
from procedural_kitchen import DESIGN_BOX as KITCHEN_BOX
from procedural_kitchen import SLOTS as KITCHEN_SLOTS
from procedural_room import BUILDERS as ROOM
from procedural_room import DESIGN_BOX as ROOM_BOX
from procedural_room import SLOTS as ROOM_SLOTS
from procedural_soft import BUILDERS as SOFT
from procedural_soft import DESIGN_BOX as SOFT_BOX
from procedural_soft import SLOTS as SOFT_SLOTS
from procedural_toys import BUILDERS as TOYS
from procedural_toys import DESIGN_BOX as TOYS_BOX
from procedural_toys import SLOTS as TOYS_SLOTS
from procedural_vehicles import BUILDERS as VEHICLES

DESIGN_BOX = {"glass": (0.07, 0.07, 0.12), "bottle": (0.07, 0.07, 0.25), "book": (0.22, 0.15, 0.03),
              "remote": (0.18, 0.05, 0.02), "laptop": (0.306, 0.33, 0.222), "plant": (0.16, 0.16, 0.30),
              "laundry_basket": (0.40, 0.40, 0.50), "robot_vacuum": (0.34, 0.34, 0.09),
              "home_robot": (0.35, 0.30, 0.45), "ball_robot": (0.22, 0.22, 0.22),
              # the warehouse (catalog "warehouse", map warehouse_a): carried kinds, light equipment, movers
              "pallet": (1.2, 1.0, 1.0), "tote": (0.6, 0.4, 0.32), "drum": (0.58, 0.58, 0.88),
              "sack": (0.9, 0.5, 0.25), "bin": (0.40, 0.25, 0.20), "pallet_jack": (1.55, 0.55, 1.2),
              "roll_cage": (0.8, 0.7, 1.7), "forklift": (2.6, 1.1, 2.1), "tote_robot": (0.8, 0.6, 1.3),
              # the downtown (catalog "downtown", map downtown_a): carried kinds, movers
              "car": (4.6, 1.8, 1.45), "van": (5.2, 2.0, 2.1), "pickup": (5.4, 2.0, 1.9),
              "box_truck": (7.3, 2.4, 3.3), "skip": (3.0, 1.7, 1.15), "dumpster": (1.8, 1.3, 1.4),
              "portable_toilet": (1.1, 1.1, 2.3), "food_cart": (2.0, 1.0, 2.2), "taxi": (4.6, 1.8, 1.55),
              "bus": (9.0, 2.5, 3.1)}
SLOTS = {"glass": ("glass",), "bottle": ("glass", "label"), "book": ("cover", "pages"),
         "remote": ("body", "buttons"), "laptop": ("body", "trim", "screen"), "plant": ("pot", "soil", "leaves"),
         "laundry_basket": ("wicker", "cloth"), "robot_vacuum": ("body", "bumper", "top"),
         "home_robot": ("body", "wheels", "screen"), "ball_robot": ("shell", "eye"),
         "pallet": ("wood", "wrap"), "tote": ("body", "grip"), "drum": ("body", "rings"), "sack": ("body",),
         "bin": ("body",), "pallet_jack": ("body", "wheels"), "roll_cage": ("frame", "wheels"),
         "forklift": ("body", "mast", "tyres", "cab"), "tote_robot": ("body", "wheels", "screen"),
         "car": ("body", "glass", "tyres", "hub", "trim", "light_f", "light_r"),
         "van": ("body", "glass", "tyres", "hub", "trim", "light_f", "light_r"),
         "pickup": ("body", "glass", "tyres", "hub", "trim", "light_f", "light_r", "liner"),
         "box_truck": ("body", "glass", "tyres", "hub", "trim", "light_f", "light_r"),
         "skip": ("body", "trim", "hazard", "debris_a", "debris_b", "debris_c", "wood"),
         "dumpster": ("body", "lid", "trim", "tyres"),
         "portable_toilet": ("body", "roof", "base", "trim", "indicator"),
         "food_cart": ("steel", "umbrella", "umbrella2", "glass", "tyres", "trim", "hub"),
         "taxi": ("body", "glass", "tyres", "hub", "trim", "light_f", "light_r", "sign", "check_a", "check_b"),
         "bus": ("body", "glass", "tyres", "hub", "trim", "light_f", "light_r", "stripe", "roof_unit", "amber")}
# the household's clutter and rebuilt kinds, each in its own module
DESIGN_BOX |= CLUTTER_BOX | CARRIED_BOX | TOYS_BOX | KITCHEN_BOX | SOFT_BOX | ROOM_BOX | HUNG_BOX
SLOTS |= CLUTTER_SLOTS | CARRIED_SLOTS | TOYS_SLOTS | KITCHEN_SLOTS | SOFT_SLOTS | ROOM_SLOTS | HUNG_SLOTS
SHARP_DEG = 40.0  # an edge sharper than this stays a crease; the rest is shaded smooth


def _shade(bm) -> None:
    """Creases stay sharp: an edge between faces more than SHARP_DEG apart is marked sharp."""
    for e in bm.edges:
        if e.is_manifold and e.calc_face_angle(0.0) > math.radians(SHARP_DEG):
            e.smooth = False


def _outward(bm) -> None:
    """Turn over every closed piece of the mesh that `recalc_face_normals` left inside out (its signed volume is
    negative). The recalculation judges a thin, flat piece (the sack's two skins that meet along its rim) by a heuristic
    that can choose wrongly; the volume cannot."""
    seen = set()
    for first in bm.faces:
        if first in seen:
            continue
        piece, stack, closed = [], [first], True
        seen.add(first)
        while stack:
            f = stack.pop()
            piece.append(f)
            for e in f.edges:
                closed = closed and not e.is_boundary
                for g in e.link_faces:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        if closed and sum(_volume(f) for f in piece) < 0.0:
            bmesh.ops.reverse_faces(bm, faces=piece)


def _volume(face) -> float:
    """The signed volume under a face (the divergence theorem's term): positive for a face that points away from the
    origin."""
    vs = [v.co for v in face.verts]
    return sum(vs[0].dot(vs[i].cross(vs[i + 1])) for i in range(1, len(vs) - 1)) / 6.0


# -- builders ----------------------------------------------------------------------------------------------------
def _robot_vacuum(bm, box) -> None:
    """A round robot vacuum as one sees it from standing height: a coloured shell with a domed deck over a dark rubber
    skirt (the bumper), a dark sensor window at the front and the lidar turret, a dark puck, behind the middle."""
    r, h = box[0] / 2, box[2]
    lathe(bm, [(0, 0), (0.94 * r, 0), (0.99 * r, 0.07 * h), (r, 0.14 * h), (r, 0.34 * h), (0.99 * r, 0.38 * h),
               (0, 0.38 * h)], 1)
    lathe(bm, [(0, 0.3 * h), (0.975 * r, 0.3 * h), (0.975 * r, 0.7 * h), (0.955 * r, 0.79 * h), (0.9 * r, 0.83 * h),
               (0.65 * r, 0.84 * h), (0, 0.845 * h)], 0)
    slab(bm, 0.46 * r, 0.66 * r, -0.26 * r, 0.26 * r, 0.8 * h, 0.855 * h, 2, bevel=0.0025)
    lathe(bm, [(0, 0.83 * h), (0.28 * r, 0.83 * h), (0.28 * r, 0.93 * h), (0.24 * r, 0.975 * h), (0.15 * r, h),
               (0, h)], 2, offset=(-0.12 * r, 0.0))


def _home_robot(bm, box) -> None:
    """A wheeled home robot: an egg-shaped body on two rubber wheels (their outer faces are the box's sides), and a
    rounded head whose front is a dark screen with two eyes (a face). The dimensions are the model's own, as the
    laptop's are: `DESIGN_BOX` is this geometry's extents, which a test keeps equal."""
    hy = box[1] / 2
    rw, ww = 0.07, 0.036  # wheel radius and width

    def wheel(side: float) -> None:  # the lathe's z axis becomes the axle, +z pointing out
        def tyre() -> None:
            lathe(bm, [(0.048, -ww / 2), (rw - 0.008, -ww / 2), (rw, -ww / 2 + 0.008), (rw, ww / 2 - 0.008),
                       (rw - 0.008, ww / 2), (0.048, ww / 2)], 1, loop=True)
            lathe(bm, [(0, -ww / 2 + 0.004), (0.052, -ww / 2 + 0.004), (0.052, ww / 2 - 0.003), (0.04, ww / 2 - 0.0008),
                       (0, ww / 2)], 0)

        axle = Matrix.Translation((-0.015, side * (hy - ww / 2), rw)) @ Matrix.Rotation(-side * math.pi / 2, 4, "X")
        placed(bm, axle, tyre)

    def eye(side: float) -> None:  # a disc on the screen, a little taller than wide
        place = Matrix.Translation((0.0975, side * 0.04, 0.365)) @ Matrix.Rotation(math.pi / 2, 4, "Y")
        placed(bm, place @ Matrix.Diagonal((1.3, 1.0, 1.0, 1.0)),
               lambda: lathe(bm, [(0, 0), (0.014, 0), (0.014, 0.0018), (0, 0.0018)], 0, segments=24))

    egg = [(0, 0.045), (0.8, 0.045), (0.93, 0.055), (1.0, 0.1), (1.0, 0.15), (0.93, 0.21), (0.78, 0.26), (0.62, 0.29),
           (0, 0.3)]
    placed(bm, Matrix.Diagonal((box[0] / 2, hy - ww - 0.003, 1.0, 1.0)), lambda: lathe(bm, egg, 0, segments=64))
    slab(bm, -0.075, 0.095, -0.13, 0.13, 0.275, 0.45, 0, bevel=0.03, segments=5)
    slab(bm, 0.09, 0.0975, -0.095, 0.095, 0.315, 0.41, 2, bevel=0.012, segments=3)
    for side in (1.0, -1.0):
        wheel(side)
        eye(side)


def _ball_robot(bm, box) -> None:
    """A spherical robot: a smooth shell with a dark band of glass round its middle and, at the front a little above it,
    a dark lens in a raised ring of the shell's colour. The shell is 0.5 mm inside the box, so that the band and the
    ring stand at its extents rather than past them."""
    r = box[0] / 2
    s = r - 0.0005  # the shell
    lathe(bm, [(s * math.cos(math.radians(a)), r + s * math.sin(math.radians(a))) for a in range(-90, 91, 6)], 0,
          segments=64)
    lo, hi = math.radians(-2.0), math.radians(2.0)  # the band, 0.5 mm proud of the shell
    lathe(bm, [((s + d) * math.cos(a) - e, r + (s + d) * math.sin(a)) for a, d, e in
               ((lo, 0.0, 0.01), (lo, 0.0005, 0.0), (hi, 0.0005, 0.0), (hi, 0.0, 0.01))], 1, segments=64, loop=True)

    def cap(rho: float, height: float) -> float:  # distance from the sphere's centre along the lens axis
        return math.sqrt((s + height) ** 2 - rho * rho)

    def lens() -> None:  # in a frame whose z axis runs from the sphere's centre through the lens
        lathe(bm, [(0, cap(0, 0.0005)), *((x, cap(x, 0.0005)) for x in (0.012, 0.024, 0.034)),
                   (0.034, cap(0.034, -0.01)), (0, cap(0, -0.01))], 1, segments=32)
        lathe(bm, [(0.034, cap(0.034, -0.01)), (0.034, cap(0.034, 0.0008)), (0.039, cap(0.039, 0.0012)),
                   (0.047, cap(0.047, 0.0012)), (0.051, cap(0.051, 0.0006)), (0.051, cap(0.051, -0.01))], 0,
              segments=32, loop=True)

    placed(bm, Matrix.Translation((0, 0, r)) @ Matrix.Rotation(math.radians(90.0 - 20.0), 4, "Y"), lens)


BUILDERS = {**HOME, "robot_vacuum": _robot_vacuum, "home_robot": _home_robot, "ball_robot": _ball_robot}


def _named(name: str, builder):
    """A warehouse builder as `(bmesh, box)`: it is handed the index of each slot of `SLOTS[name]` by name."""
    index = {slot: i for i, slot in enumerate(SLOTS[name])}
    return lambda bm, box: builder(bm, box, index)


BUILDERS |= {name: _named(name, builder)
             for name, builder in {**GOODS, **EQUIPMENT, **CITY, **VEHICLES, **CLUTTER, **CARRIED, **TOYS, **KITCHEN,
                                   **SOFT, **ROOM, **HUNG}.items()}


def build(builder: str, name: str) -> tuple[bpy.types.Mesh, tuple[str, ...]]:
    """The mesh of a manifest `builder` and its slot names. Raises KeyError for a builder this file does not have."""
    if builder not in BUILDERS:
        raise KeyError(f"unknown procedural builder {builder!r}; known: {sorted(BUILDERS)}")
    box = DESIGN_BOX[builder]
    bm = bmesh.new()
    BUILDERS[builder](bm, box)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    _outward(bm)
    fit(bm, box)
    _shade(bm)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh, SLOTS[builder]
