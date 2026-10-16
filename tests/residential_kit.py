"""Shared pieces of the residential kit's tests: a small synthetic residential map whose every number can be checked by
hand, the catalog and registry entries step R adds (so the tests run now and after), a loader for `render/` modules that
import `bpy` (their geometry is plain `Part`s, which need no Blender) and the drone's tall-piece rule as a function.

The kit map, 100 m by 60 m, every piece a quarter turn, no wall, the walls' height 12 m (the suburb's):

    y 60 +-------------------------+------------------------------+------------+
         | 3 Elm Street (house,    | 4 Elm Street (a block of     | car park   |
         | drive, drop zone)       | 5 storeys)                   | (the nest) |
    36.5 +-------------------------+------------------------------+------------+
         | street: verge, sidewalk, carriageway, sidewalk, verge (north then south)                |
    23.5 +-------------------------+------------------------------+--+---------+
         | 1 Elm Street (house,    | 2 Elm Street (house, hedge,  |pa| green    |
         | garage, drive, drop     | path, a bed)                 |ss| (a play- |
         | zone, hedge, fence)     |                              |  |  ground) |
       0 +-------------------------+------------------------------+--+---------+
         0                         35                            70 75       100

Hand-worked numbers the tests lean on: the streets' verges are 1.98 m across and 100 m long, 0.15 m high, their tops the
stems' base; both drop zones are 3 m square on a drive's `parking` surface at 0.05 m, 4.0 m from the nearest house wall;
the nest is at (85, 48) on bare asphalt with nothing within 3.65 m of it.
"""

from __future__ import annotations

import copy
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

from dressing_kit import RENDER, render_module
from stbench import catalog_residential as residential
from stbench.catalog import CATALOGS, DOWNTOWN_DELIVERY_COLORS, DOWNTOWN_KINDS, Catalog
from stbench.scenery import DRESSING, DressingClass

HALF_PI = math.pi / 2
DROP_ZONE_M = 3.65  # the drone's body radius (0.65 m) and its 3.0 m clearance: the lowering rule's reach


def residential_catalog() -> Catalog:
    """`CATALOGS["residential"]` once step R registers it; until then the entry `catalog_residential` documents,
    built from the downtown's objects."""
    if "residential" in CATALOGS:
        return CATALOGS["residential"]
    return Catalog("residential", {k: DOWNTOWN_KINDS[k] for k in residential.KIND_NAMES}, residential.SET_KINDS,
                   residential.LEAVING_KINDS, {}, residential.N_MOVERS, residential.N_ITEMS, residential.DELIVERY_KIND,
                   DOWNTOWN_DELIVERY_COLORS)


def register(patch) -> None:
    """Put what step R adds into the registries for as long as `patch` (a `pytest.MonkeyPatch`) lives: the catalog and
    the layout's five dressing classes (name and note from its own `CLASSES`), each only where it is missing."""
    if "residential" not in CATALOGS:
        patch.setitem(CATALOGS, "residential", residential_catalog())
    for name, note in render_module("dressing_layout_residential").CLASSES:
        if name not in DRESSING:
            patch.setitem(DRESSING, name, DressingClass(name, ("residential",), (), 1, note))


def _region(i: int, name: str, kind: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"id": f"R{i}", "name": name, "kind": kind, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "anchor": [(x0 + x1) / 2, (y0 + y1) / 2], "level": 0, "z0": 0.0, "z1": 12.0}


def _piece(fid: str, name: str, kind: str, center, size, yaw: float, color: str, blocks: bool, surfaces=()) -> dict:
    return {"id": fid, "name": name, "kind": kind, "mobility": "fixed", "center": list(center), "size": list(size),
            "yaw": yaw, "color": color, "blocks": blocks, "surfaces": list(surfaces)}


def _surface(sid: str, kind: str, rect, z: float) -> dict:
    return {"id": sid, "kind": kind, "rect": list(rect), "z": z}


_WALK = [_surface("S1", "sidewalk", (-0.1, 0.0, 1.0, 98.0), 0.15)]
_BAYS = [_surface("S1", "parking", (0.0, -3.5, 2.5, 6.7), 0.0), _surface("S2", "parking", (0.0, 3.5, 2.5, 6.7), 0.0)]
KIT: dict = {
    "name": "residential_kit", "catalog": "residential", "depth_unit": "cm",
    "outdoor": {"sun_elevation_deg": 62.0, "sun_rotation_deg": 110.0, "sun_strength": 3.6, "sky_strength": 0.55},
    "bounds": [0.0, 0.0, 100.0, 60.0], "wall_thickness": 0.6, "wall_height": 12.0,
    "regions": [_region(1, "street", "street", 0, 23.5, 100, 36.5),
                _region(2, "1 Elm Street", "garden", 0, 0, 35, 23.5),
                _region(3, "2 Elm Street", "garden", 35, 0, 70, 23.5),
                _region(4, "passage", "passage", 70, 0, 75, 23.5), _region(5, "green", "green", 75, 0, 100, 23.5),
                _region(6, "3 Elm Street", "garden", 0, 36.5, 35, 60),
                _region(7, "4 Elm Street", "garden", 35, 36.5, 70, 60),
                _region(8, "car park", "car park", 70, 36.5, 100, 60)],
    "walls": [], "openings": [], "windows": [], "ceiling_lights": [],
    "furniture": [
        # the street: sidewalks (kerbs on the carriageway's side), verges behind them, a kerbside lane of two bays
        _piece("F01", "south sidewalk", "sidewalk", (50, 26.25), (1.5, 100, 0.15), HALF_PI, "grey", True, _WALK),
        _piece("F02", "north sidewalk", "sidewalk", (50, 33.75), (1.5, 100, 0.15), -HALF_PI, "grey", True, _WALK),
        _piece("F03", "south verge", "verge", (50, 24.49), (1.98, 100, 0.15), HALF_PI, "green", True),
        _piece("F04", "north verge", "verge", (50, 35.51), (1.98, 100, 0.15), -HALF_PI, "green", True),
        _piece("F05", "kerbside bays", "parking_lane", (30, 28.3), (2.6, 14, 0.01), HALF_PI, "white", False, _BAYS),
        # 1 Elm Street: a house with a garage (9.5 m wide) and its drive, a drop zone at the drive's end, path, hedge
        _piece("F10", "1 Elm Street house", "house", (14, 11.75), (9.5, 9.5, 8.4), HALF_PI, "beige", True),
        _piece("F11", "1 Elm Street drive", "driveway", (11, 20), (7, 3.2, 0.05), HALF_PI, "grey", False,
               [_surface("S1", "parking", (0, 0, 6.8, 2.9), 0.05)]),
        _piece("F12", "1 Elm Street", "drop_zone", (11, 20.5), (3, 3, 0.055), 0.0, "white", False),
        _piece("F13", "1 Elm Street path", "path", (15.6, 20), (1.2, 7, 0.03), 0.0, "grey", False),
        _piece("F14", "1 Elm Street front lawn", "lawn", (17.5, 20), (35, 7, 0.03), 0.0, "green", False),
        _piece("F15", "1 Elm Street hedge", "hedge", (4.5, 22.9), (0.6, 9, 1.2), HALF_PI, "green", True),
        _piece("F16", "1 Elm Street fence", "fence", (34.95, 8), (0.1, 16, 1.8), 0.0, "brown", True),
        _piece("F17", "1 Elm Street tree bed", "garden_bed", (4.5, 18.5), (4, 4, 0.3), 0.0, "grey", True),
        _piece("F18", "1 Elm Street border", "garden_bed", (33.8, 8), (1.4, 14, 0.3), 0.0, "grey", True),
        # 2 Elm Street: a brick house, a path, a hedge, a bed
        _piece("F20", "2 Elm Street house", "house", (52, 11.75), (10, 8.5, 8.8), HALF_PI, "red", True),
        _piece("F22", "2 Elm Street path", "path", (54, 20.1), (1.2, 6.7, 0.03), 0.0, "grey", False),
        _piece("F23", "2 Elm Street hedge", "hedge", (60, 22.9), (0.6, 15, 1.5), HALF_PI, "green", True),
        _piece("F24", "2 Elm Street bed", "garden_bed", (64, 19.5), (6, 3, 0.3), 0.0, "brown", True),
        # the passage and the green with its playground
        _piece("F30", "passage path", "path", (72.5, 11.75), (2, 23, 0.03), 0.0, "grey", False),
        _piece("F31", "green lawn", "lawn", (87.5, 11.75), (25, 23.5, 0.03), 0.0, "green", False),
        _piece("F32", "playground bed", "garden_bed", (87.5, 11.75), (14, 9, 0.3), 0.0, "brown", True),
        # 3 Elm Street (the north side faces south): a white house, its drive and a drop zone, a bed
        _piece("F40", "3 Elm Street house", "house", (20, 48), (9, 10, 8.6), -HALF_PI, "white", True),
        _piece("F41", "3 Elm Street drive", "driveway", (23.25, 40.25), (6.5, 3.2, 0.05), -HALF_PI, "grey", False,
               [_surface("S1", "parking", (0, 0, 6.3, 2.9), 0.05)]),
        _piece("F42", "3 Elm Street", "drop_zone", (23.25, 39.5), (3, 3, 0.055), 0.0, "white", False),
        _piece("F43", "3 Elm Street bed", "garden_bed", (6, 40), (3, 3, 0.3), 0.0, "grey", True),
        # 4 Elm Street: an apartment block of five storeys (15.9 m: 5 x 3.0 and a 0.9 m parapet)
        _piece("F50", "4 Elm Street block", "block", (52.5, 48.5), (13, 30, 15.9), -HALF_PI, "red", True),
        # the car park: the nest on bare asphalt, and a lane of two bays 5.7 m from it
        _piece("F60", "nest", "nest", (85, 48), (1.6, 1.6, 0.02), HALF_PI, "grey", False),
        _piece("F61", "car park bays", "parking_lane", (85, 55), (2.6, 14, 0.01), HALF_PI, "white", False, _BAYS),
    ],
}


def kit() -> dict:
    """A fresh copy of the kit map's spec (a test may edit its own)."""
    return copy.deepcopy(KIT)


def tall_near(spec: dict, dressing: list, reach: float = DROP_ZONE_M) -> list[str]:
    """What stands within `reach` of a drop zone's or the nest's centre and is too tall for the drone to lower or take
    off beside (everything within 3.65 m of its centre must stand below its altitude less 3 m: 4 m at the lowering
    height): a `house` or a `block` of the map (a fixed piece whose footprint comes within `reach`), and a dressing box
    whose top is higher than 3.9 m. Each as a message; empty when the map keeps the rule."""
    points = [(f["name"], tuple(f["center"])) for f in spec["furniture"] if f["kind"] in ("drop_zone", "nest")]

    def gap(p, x0, x1, y0, y1) -> float:
        return math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1))

    out = []
    for name, p in points:
        for f in spec["furniture"]:
            if f["kind"] in ("house", "block"):
                hx, hy = (f["size"][0] / 2, f["size"][1] / 2) if abs(math.cos(f["yaw"])) > 0.5 \
                    else (f["size"][1] / 2, f["size"][0] / 2)
                if gap(p, f["center"][0] - hx, f["center"][0] + hx, f["center"][1] - hy, f["center"][1] + hy) < reach:
                    out.append(f"{f['name']} is within {reach} m of {name}")
        for d in dressing:
            x0, y0, z0, x1, y1, z1 = d.box
            if z1 > 3.9 and gap(p, x0, x1, y0, y1) < reach:
                out.append(f"{d.cls} at {d.pos} is within {reach} m of {name}")
    return out


@contextmanager
def blender_stubbed():
    """`render/` on the path and `bpy`, `bmesh` and `mathutils` stubbed with mocks for the duration: the builders of
    `render/` import them at the top and call them only to make meshes, while a free test reads their `Part`s. Every
    `render/` module imported under the stubs is dropped again on the way out, with the stubs, so no other test finds
    one (numpy and the rest, imported before, stay)."""
    names = ("bpy", "bmesh", "mathutils")
    before = set(sys.modules)
    for n in names:
        sys.modules[n] = MagicMock(name=n)
    sys.path.insert(0, str(RENDER))
    try:
        yield
    finally:
        sys.path.remove(str(RENDER))
        for n in set(sys.modules) - before:
            file = getattr(sys.modules[n], "__file__", None)
            if n in names or (isinstance(file, str) and Path(file).resolve().parent == RENDER):
                del sys.modules[n]
