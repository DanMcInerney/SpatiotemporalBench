"""The maker of the `fulfilment` map (M4): a small fulfilment area, public, in the
warehouse catalog, on which M3's tote AMR job `amr/1` runs unchanged. `spec()` builds the map as the dict
the map spec describes (`src/stbench/maps/README.md`); `python tools/maps/fulfilment.py [OUT]` writes it, by
default to `src/stbench/maps/fulfilment.json`. It imports `mapspec` only: a map's maker is a script, not package code.
`tests/test_fulfilment_map.py` holds the arithmetic.

One rectangular hall, 27.8 x 29.9 m inside (walls 0.2 m thick, 6.0 m high, no doors, no windows), six regions sharing
their edges with no wall between them (x east, y north, metres; S a shelving unit, T a packing table, each with its bay
in front; R a rack row, 1.1 m deep, and "." the 3.6 m lane between two rows):

    y 30.0 +----------------------------------+-----------+-----------+
           | pick area                        | aisle 1   | aisle 2   |
           |   S north wall at x 6.0 and 12.5 | R . . . R | R . . . R |   the aisles run 18.9 m from the staging
           |   S west wall at y 22.5 and 25.5 | R . . . R | R . . . R |   area to the north wall; the rows of the two
      20.1 +----------------------------------+ R . . . R | R . . . R |   aisles stand back to back at x 22.1, and
           | pack area                        | R . . . R | R . . . R |   the first row's back closes the pack and
           |   T west wall at y 12.4, 15.4,   | R . . . R | R . . . R |   pick areas' east edge (x 16.3)
           |     and 18.4                     | R . . . R | R . . . R |
      11.1 +----------------------------------+-----------+-----------+
           | staging area: four staging lanes, in line with the dock doors (x 6.0 ... 16.2, y 9.1)  |
       7.1 +----------------------------------------------------------------------------------------+
           | dock: four dock doors on the south wall, two pallet jacks and a roll cage near them;   |
           |       the charger on the east wall at y 3.6                                            |
       0.1 +----------------------------------------------------------------------------------------+
          x 0.1                              16.3        22.1        27.9

The AMR's pieces are the warehouse catalog's (`render/furniture_warehouse.py`): a `charger`, a `bay` on the face of
every station, `staging_lane`s, `pallet_rack`s. Every station faces open floor 7 m deep, so its undock's reverse
corridor is never a fact of the map. The racks are seven bays (18.9 m) long and three levels high; only the
floor-level bay-levels have a surface (a `rack` surface: the AMR's pick faces), and the rack builder stocks the rest.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

NAME = "fulfilment"
PI = math.pi
# The inside of the hall, on the walls' faces (the walls are 0.2 m thick, so the bounds are 0.1 m outside them).
X0, Y0, X1, Y1 = 0.1, 0.1, 27.9, 30.0
BOUNDS = [0.0, 0.0, 28.0, 30.1]
DOCK_Y, STAGING_Y, PACK_Y = 7.1, 11.1, 20.1  # the regions' north edges; the aisles run on from the staging edge
AISLE_X = (16.3, 22.1)  # the aisles' west edges: aisle 1 is x 16.3-22.1, aisle 2 22.1-27.9 (5.8 m: 2 x 1.1 + 3.6)

# A rack row: single-deep, `rack_layout`'s bay and level, seven bays long (18.9 m: the staging edge to the north wall).
RACK_DEPTH, BAY_M, BAYS, RACK_H = 1.1, 2.7, 7, 3.9
RACK_LEN = round(BAYS * BAY_M, 4)
RACK_SURFACE = (1.08, 2.6)  # a floor-level bay-level's surface, as warehouse_a's

BAY_DEPTH, BAY_MARGIN, BAY_H = 1.6, 0.2, 0.005  # a station's bay: the AMR's zone, a marking under the AMR's climb
STATION_LEN = 2.0
PACK_SIZE = (0.8, STATION_LEN, 0.9)  # a packing table, as warehouse_a's
SHELF_SIZE = (0.6, STATION_LEN, 2.2)  # four shelves; a tote (0.32 m) on the top one (1.85 m) stays under the top board
SHELF_Z = (0.4, 0.9, 1.4, 1.85)
LANE_SIZE = (2.8, 1.4, 0.005)  # a staging lane, as warehouse_a's after M3-maps: paint under the AMR's climb
DOOR_SIZE = (0.1, 2.4, 3.0)
CHARGER_SIZE = (0.12, 0.6, 0.3)
LIGHT_Z, LIGHT_WATTS = 5.8, 270.0  # high bay as warehouse_a's: the structure stands above 5.0 m, a lens at 5.8 at most

DOOR_X = (6.0, 9.4, 12.8, 16.2)  # pitch 3.4 m as warehouse_a's: a door's seal pads (0.25 m a side) need 0.5 m
LANE_Y = 9.1  # the lanes are 2.8 m long across the staging area (y 7.7-10.5), one in line with each door
CHARGER_Y = 3.6
PACK_YS = (12.4, 15.4, 18.4)  # packing tables on the west wall, 3 m apart
SHELF_WEST_YS = (22.5, 25.5)  # shelving on the west wall
SHELF_NORTH_XS = (6.0, 12.5)  # shelving on the north wall, clear of the dressing's fittings (x 9.8) and column (9.4)


def r(v: float) -> float:
    return round(v, 4)


def region(rid: str, name: str, box: tuple[float, float, float, float], anchor: tuple[float, float],
           kind: str | None = None) -> dict:
    x0, y0, x1, y1 = box
    return {"id": rid, "name": name} | ({"kind": kind} if kind else {}) | {
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "anchor": list(anchor), "level": 0, "z0": 0.0, "z1": 6.0}


REGIONS = (
    region("R1", "dock", (X0, Y0, X1, DOCK_Y), (14.0, 5.5)),
    region("R2", "staging area", (X0, DOCK_Y, X1, STAGING_Y), (11.1, 9.1)),
    region("R3", "pack area", (X0, STAGING_Y, AISLE_X[0], PACK_Y), (8.0, 15.0)),
    region("R4", "pick area", (X0, PACK_Y, AISLE_X[0], Y1), (8.0, 25.0)),
    region("R5", "aisle 1", (AISLE_X[0], STAGING_Y, AISLE_X[1], Y1), (19.2, 14.0), "aisle"),
    region("R6", "aisle 2", (AISLE_X[1], STAGING_Y, X1, Y1), (25.0, 14.0), "aisle"),
)


def piece(name: str, kind: str, center: tuple[float, float], size: tuple[float, float, float], yaw: float, color: str,
          *, blocks: bool = True, mobility: str = "fixed", surfaces: tuple = ()) -> dict:
    """A furniture piece without its id; a surface is `(kind, local rect, z)`, numbered with the piece."""
    return {"name": name, "kind": kind, "mobility": mobility, "center": [r(center[0]), r(center[1])],
            "size": list(size), "yaw": yaw, "color": color, "blocks": blocks, "surfaces": list(surfaces)}


def station(name: str, kind: str, wall: str, along: float, size: tuple[float, float, float], surfaces: tuple
            ) -> tuple[dict, dict]:
    """A station against the west wall (`W`, facing east) or the north wall (`N`, facing south) and its bay: 1.6 m
    deep and 0.1 m wider than the station each side, touching the face (the zone rule insets a zone by 0.05 m)."""
    depth, length, _ = size
    yaw = 0.0 if wall == "W" else -PI / 2
    out = (math.cos(yaw), math.sin(yaw))
    foot = (X0, along) if wall == "W" else (along, Y1)  # the wall's face at the station's middle

    def out_by(d: float) -> tuple[float, float]:
        return (foot[0] + out[0] * d, foot[1] + out[1] * d)

    colour = "steel" if kind == "packing_table" else "grey"
    return (piece(name, kind, out_by(depth / 2), size, yaw, colour, surfaces=surfaces),
            piece("bay", "bay", out_by(depth + BAY_DEPTH / 2), (BAY_DEPTH, length + BAY_MARGIN, BAY_H), yaw, "yellow",
                  blocks=False))


def rack(x: float, yaw: float) -> dict:
    """A pallet-rack row from the staging edge to the north wall, its front towards the lane (yaw 0: east; pi: west),
    with a `rack` surface at the middle of each floor-level bay only: the AMR's pick faces."""
    surfaces = tuple(("rack", [0.0, r(-RACK_LEN / 2 + (b + 0.5) * BAY_M), *RACK_SURFACE], 0.0) for b in range(BAYS))
    return piece("pallet rack", "pallet_rack", (x, (STAGING_Y + Y1) / 2), (RACK_DEPTH, RACK_LEN, RACK_H), yaw, "blue",
                 surfaces=surfaces)


def furniture() -> list[dict]:
    """Fixed pieces first (doors, lanes, stations, racks, charger, bays), then the light equipment, as warehouse_a."""
    doors = [piece("dock door", "dock_door", (x, 0.15), DOOR_SIZE, PI / 2, "grey") for x in DOOR_X]
    lanes = [piece("staging lane", "staging_lane", (x, LANE_Y), LANE_SIZE, PI / 2, "yellow",
                   surfaces=(("staging lane", [0.0, 0.0, 2.7, 1.3], 0.005),)) for x in DOOR_X]
    table = (("work table", [0.0, 0.0, 0.74, 1.94], PACK_SIZE[2]),)
    shelves = tuple(("shelf", [0.0, 0.0, 0.54, 1.94], z) for z in SHELF_Z)
    stations = [station("packing table", "packing_table", "W", y, PACK_SIZE, table) for y in PACK_YS]
    stations += [station("shelving unit", "shelving", "W", y, SHELF_SIZE, shelves) for y in SHELF_WEST_YS]
    stations += [station("shelving unit", "shelving", "N", x, SHELF_SIZE, shelves) for x in SHELF_NORTH_XS]
    racks = [rack(AISLE_X[0] + RACK_DEPTH / 2, 0.0), rack(AISLE_X[1] - RACK_DEPTH / 2, PI),
             rack(AISLE_X[1] + RACK_DEPTH / 2, 0.0), rack(X1 - RACK_DEPTH / 2, PI)]
    charger = piece("charger", "charger", (X1 - CHARGER_SIZE[0] / 2, CHARGER_Y), CHARGER_SIZE, PI, "black")
    light = [piece("pallet jack", "pallet_jack", (7.2, 1.9), (1.55, 0.55, 1.2), PI / 2, "red", mobility="light"),
             piece("pallet jack", "pallet_jack", (8.2, 1.9), (1.55, 0.55, 1.2), PI / 2, "yellow", mobility="light"),
             piece("roll cage", "roll_cage", (14.5, 1.3), (0.8, 0.7, 1.7), PI, "grey", mobility="light")]
    fixed = doors + lanes + [s for s, _ in stations] + racks + [charger] + [b for _, b in stations]
    out, n_surface = [], 0
    for prefix, pieces in (("F", fixed), ("L", light)):
        for i, p in enumerate(pieces, 1):
            surfaces = []
            for kind, rect, z in p["surfaces"]:
                n_surface += 1
                surfaces.append({"id": f"S{n_surface:02d}", "kind": kind, "rect": rect, "z": z})
            out.append({"id": f"{prefix}{i:02d}"} | p | {"surfaces": surfaces})
    return out


def lights() -> list[dict]:
    """24 high-bay fixtures, one a 35 m2. Each stands in a gap between two roof purlins (the dressing sets the
    purlins 1.45 m apart, rounded to a whole number across the hall) and clear of the dressing's trusses, pipes,
    ducts and trays, which it lays out by the hall's size: nothing hangs under a lens
    (`tests/test_dressing_layout_warehouse.py`)."""
    pitch = (X1 - X0) / round((X1 - X0) / 1.45)

    def at(ms: tuple[int, ...], ys: tuple[float, ...]) -> list[tuple[float, float]]:
        return [(X0 + m * pitch, y) for m in ms for y in ys]

    spots = (("R1", at((3, 7, 11, 15, 18), (4.5,))), ("R2", at((5, 10, 15), (9.2,))),
             ("R3", at((4, 9), (12.8, 17.0))), ("R4", at((4, 9), (23.0, 27.5))),
             ("R5", at((13,), (12.8, 17.0, 23.0, 27.5))), ("R6", at((17,), (12.8, 17.0, 23.0, 27.5))))
    return [{"region": rid, "pos": [r(x), r(y), LIGHT_Z], "watts": LIGHT_WATTS} for rid, xy in spots for x, y in xy]


def walls() -> list[dict]:
    corners = [(0.0, 0.0), (BOUNDS[2], 0.0), (BOUNDS[2], BOUNDS[3]), (0.0, BOUNDS[3])]
    sides = ((0, 1), (1, 2), (3, 2), (0, 3))  # south, east, north, west
    return [{"id": f"W0{i}", "a": list(corners[a]), "b": list(corners[b]), "thickness": 0.2, "height": 6.0,
             "exterior": True} for i, (a, b) in enumerate(sides, 1)]


def spec() -> dict:
    return {"name": NAME, "catalog": "warehouse", "bounds": BOUNDS, "wall_thickness": 0.2,
            "wall_height": 6.0, "regions": list(REGIONS), "walls": walls(), "openings": [], "windows": [],
            "furniture": furniture(), "ceiling_lights": lights()}


ROW_KEYS = ("regions", "walls", "openings", "windows", "furniture", "ceiling_lights")


def dump(doc: dict) -> str:
    """The map as the package keeps it: its scalar keys, then one region, wall, piece or light a line."""
    lines = [f'  "{k}": {json.dumps(v)}' for k, v in doc.items() if k not in ROW_KEYS]
    for key in ROW_KEYS:
        rows = ",\n".join(f"    {json.dumps(row)}" for row in doc[key])
        lines.append(f'  "{key}": [\n{rows}\n  ]' if rows else f'  "{key}": []')
    return "{\n" + ",\n".join(lines) + "\n}\n"


def main(argv: list[str]) -> None:
    default = Path(__file__).resolve().parents[2] / "src" / "stbench" / "maps" / f"{NAME}.json"
    out = Path(argv[1]) if len(argv) > 1 else default
    out.write_bytes(dump(spec()).encode("utf-8"))  # LF on every platform: the file's bytes are the same everywhere
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv)
