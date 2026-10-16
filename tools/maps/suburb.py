"""The maker of the `suburb` map (M4, part b): a street of detached houses, public, in
the residential catalog, on which M3's delivery drone `drone/1` runs unchanged. `spec()` builds the map as the dict
the map spec describes (`src/stbench/maps/README.md`); `python tools/maps/suburb.py [OUT]` writes it, by default
to `src/stbench/maps/suburb.json`. It imports `mapspec` only: a map's maker is a script, not package code.
`tests/test_suburb_map.py` holds the arithmetic.

One straight street, 300 m long and 12 m across (x east, y north, metres; no walls, open at both ends), with 18 plots of
15 x 35 m on each side, a green with a playground, a passage and a car park:

    y 82 +---------------------------+------+---------------------------+
         | north plots 1-17 (odd)    | car  | north plots 19-35 (odd)   |
         | nine, facing south        | park | nine, facing south        |
         |                           | (N)  |                           |
      47 +---------------------------+------+---------------------------+
         | street: verge, sidewalk, carriageway, sidewalk, verge (12 m) |
      35 +---------------------------+-+----+---------------------------+
         | south plots 2-18 (even)   |p|G   | south plots 20-36 (even)  |
         | nine, facing north        | |    | nine, facing north        |
       0 +---------------------------+-+----+---------------------------+
         x 0                        135 165                            300

N the nest (150, 64.5) on the car park's bare asphalt, p the passage (x 135-138), G the green (x 138-165).

A plot is a `garden` region named by its address ("12 Elm Street"). Its house stands 6.5-8.0 m back; a 3.4 m driveway
runs from the street to the garage side of the house, a path to its front door; the west boundary is a hedge or a fence,
the rear a fence; a front lawn, a back lawn and two garden beds (the only ground a tree or a shrub stands on).
**A drop zone stands at the street end of the driveway on a third of the plots**: the driveway's `parking` surface
carries the vehicles the zone drop moves onto it. The street keeps 1.2 m verges (street trees and lamps) against the
plots, 1.8 m sidewalks (kerb on the carriageway's side; the only ground pedestrians walk) and kerbside bays on the
carriageway, two to a lane, never opposite one another. A plot is 525 m2, under the national median of new lots
(794 m2; Census Survey of Construction, 2025) and in the 38% of new lots under 650 m2 (NAHB, eyeonhousing.org).
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

NAME = "suburb"
PI2 = math.pi / 2
LENGTH, PLOT_W, PLOT_D = 300.0, 15.0, 35.0
SOUTH_EDGE, NORTH_EDGE = 35.0, 47.0  # the plots' street boundaries: the street runs between them
BOUNDS = [0.0, 0.0, LENGTH, NORTH_EDGE + PLOT_D]
WALL_H = 12.0  # the walls' height the map would have: the dressing's trees stand under it, the drone's range is in it
OUTDOOR = {"sun_elevation_deg": 62.0, "sun_rotation_deg": 110.0, "sun_strength": 3.6, "sky_strength": 0.55}

# The street's cross-section, from the south plots' boundary: verge, sidewalk, carriageway, sidewalk, verge.
VERGE_W, WALK_W, ROAD_W = 1.2, 1.8, 6.0
ROAD_Y0 = SOUTH_EDGE + VERGE_W + WALK_W
ROAD_Y1 = ROAD_Y0 + ROAD_W
WALK_H, VERGE_H = 0.15, 0.15
LANE_W, BAY_LEN, BAY_W, BAY_PITCH = 2.6, 6.7, 2.5, 7.0  # a kerbside lane of bays, as the downtown's
KERB_M = 0.2  # the sidewalk builder's kerb stone on its front edge
WALK_SURFACE = (-0.1, 0.0, 1.5)  # local x centre, y centre and width of a sidewalk's surface (a dumpster is 1.3 m wide)
# Where the street is cut: the passage and the green (south), the car park (north).
PASSAGE_X, GREEN_X, CAR_PARK_X = (135.0, 138.0), (138.0, 165.0), (135.0, 165.0)
PLOT_XS = (*range(0, 135, 15), *range(165, 300, 15))  # the plots' west edges, on each side
NEST = (150.0, 64.5)
# Kerbside lanes of two bays: centres along x, south then north (never opposite one another: the carriageway is 6 m).
LANES = {"south": (24.0, 84.0, 144.0, 204.0, 264.0), "north": (54.0, 114.0, 174.0, 234.0, 284.0)}
CAR_PARK_BAYS = 4  # bays in each of the car park's two lanes
# The plots with a drop zone, by index along their side (west to east): a third of the plots.
DROP_PLOTS = {"south": (1, 4, 7, 10, 13, 16), "north": (0, 3, 6, 9, 12, 15)}

# A plot (u across from its west edge, d from its street boundary into the plot).
SETBACK_M = (6.5, 8.0)  # the house's front face, from the street boundary
HOUSE_DEPTH_M, HOUSE_WIDTH_M, HOUSE_H_M = (8.0, 10.0), (8.0, 11.0), (8.0, 9.0)
HOUSE_COLOURS = ("beige", "white", "grey", "red", "brown", "yellow", "blue", "beige", "white")
DRIVE_W, DRIVE_H = 3.4, 0.05
ZONE_SIZE, ZONE_D = 3.0, 1.9  # the zone, and how far its centre lies from the street boundary
GARAGE_FROM_EDGE = 1.75  # the garage door's centre, from the house's side (`procedural_house`: its 2.4 m door)
PATH_W, LAWN_H = 1.2, 0.03
HEDGE_W, FENCE_W = 0.6, 0.1


def r(v: float) -> float:
    return round(v, 4)


def piece(name: str, kind: str, center: tuple[float, float], size: tuple[float, float, float], yaw: float, color: str,
          *, blocks: bool = True, surfaces: tuple = ()) -> dict:
    """A fixed furniture piece without its id; a surface is `(kind, local rect, z)`."""
    return {"name": name, "kind": kind, "mobility": "fixed", "center": [r(center[0]), r(center[1])],
            "size": [r(v) for v in size], "yaw": yaw, "color": color, "blocks": blocks, "surfaces": list(surfaces)}


def region(name: str, kind: str, box: tuple[float, float, float, float]) -> dict:
    x0, y0, x1, y1 = box
    return {"name": name, "kind": kind, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "anchor": [r((x0 + x1) / 2), r((y0 + y1) / 2)], "level": 0, "z0": 0.0, "z1": WALL_H}


def bay_surfaces(n: int) -> tuple:
    """`n` bays along a lane's length (its local y), a line's width apart, centred."""
    return tuple(("parking", [0.0, r((i - (n - 1) / 2) * BAY_PITCH), BAY_W, BAY_LEN], 0.0) for i in range(n))


def lane(name: str, center: tuple[float, float], yaw: float, n: int) -> dict:
    """A lane of `n` painted bays, its front (the road side) towards `yaw`."""
    return piece(name, "parking_lane", center, (LANE_W, n * BAY_PITCH, 0.01), yaw, "white", blocks=False,
                 surfaces=bay_surfaces(n))


class Plot:
    """One plot's pieces. `u` runs east from the plot's west edge and `d` from its street boundary into the plot: the
    north plots face south (the house's front yaw is -pi/2), the south plots face north (pi/2). Every piece is laid out
    in these terms, then turned into the map's x and y."""

    def __init__(self, number: int, x0: float, north: bool, drop: bool):
        self.number, self.x0, self.north, self.drop = number, x0, north, drop
        self.addr = f"{number} Elm Street"
        self.rng = random.Random(number)
        self.face = -PI2 if north else PI2
        self.pieces: list[dict] = []

    def xy(self, u: float, d: float) -> tuple[float, float]:
        return (self.x0 + u, NORTH_EDGE + d if self.north else SOUTH_EDGE - d)

    def add(self, what: str, kind: str, u: float, d: float, du: float, dd: float, h: float, color: str, *,
            facing: bool = False, blocks: bool = True, surfaces: tuple = ()) -> None:
        """A piece `du` across and `dd` deep, centred at (u, d). A `facing` piece (a house, a driveway) is sized as
        `(depth, width)` with the plot's yaw; the others sit square to the map, a fence with its long axis along y."""
        center = self.xy(u, d)
        if facing:
            self.pieces.append(piece(f"{self.addr} {what}", kind, center, (dd, du, h), self.face, color,
                                     blocks=blocks, surfaces=surfaces))
        elif kind == "fence" and du > dd:
            self.pieces.append(piece(f"{self.addr} {what}", kind, center, (dd, du, h), PI2, color))
        else:
            self.pieces.append(piece(f"{self.addr} {what}", kind, center, (du, dd, h), 0.0, color, blocks=blocks,
                                     surfaces=surfaces))

    def build(self) -> list[dict]:
        rng = self.rng
        setback = round(rng.uniform(*SETBACK_M), 1)
        depth, width = (round(rng.uniform(*HOUSE_DEPTH_M), 1), round(rng.uniform(*HOUSE_WIDTH_M), 1))
        height, colour = round(rng.uniform(*HOUSE_H_M), 1), rng.choice(HOUSE_COLOURS)
        side = 1.0 if self.north else -1.0  # the garage side (the house's left) in u: east for the north plots
        half = PLOT_W / 2
        u_h = min(max(half + rng.uniform(-1.0, 1.0), width / 2 + 1.2), PLOT_W - width / 2 - 1.2)
        u_garage = u_h + side * (width / 2 - GARAGE_FROM_EDGE)
        u_door = u_h - side * 0.35 * (width / 2 - 0.25)
        d_house = setback + depth / 2
        self.add("house", "house", u_h, d_house, width, depth, height, colour, facing=True)
        self.add("drive", "driveway", u_garage, setback / 2, DRIVE_W, setback, DRIVE_H, "grey", facing=True,
                 blocks=False, surfaces=(("parking", [0.0, 0.0, r(setback - 0.2), r(DRIVE_W - 0.2)], DRIVE_H),))
        if self.drop:
            self.add("", "drop_zone", u_garage, ZONE_D, ZONE_SIZE, ZONE_SIZE, DRIVE_H + 0.005, "white", blocks=False)
            self.pieces[-1]["name"] = self.addr
        self.add("path", "path", u_door, setback / 2, PATH_W, setback, LAWN_H, "grey", facing=True, blocks=False)
        self.add("front lawn", "lawn", half, setback / 2, PLOT_W, setback, LAWN_H, "green", blocks=False)
        rear = setback + depth + 0.1
        self.add("back lawn", "lawn", half, (rear + PLOT_D - 0.1) / 2, PLOT_W, PLOT_D - 0.1 - rear, LAWN_H, "green",
                 blocks=False)
        # the boundaries: a hedge or a fence along the west side, a fence across the rear
        length = PLOT_D - 0.3
        if rng.random() < 0.6:
            self.add("hedge", "hedge", HEDGE_W / 2 + 0.005, 0.2 + length / 2, HEDGE_W, length,
                     round(rng.uniform(0.9, 1.8), 1), "green")
        else:
            self.add("fence", "fence", FENCE_W / 2 + 0.005, 0.2 + length / 2, FENCE_W, length,
                     round(rng.uniform(1.2, 1.8), 1), "brown")
        self.add("rear fence", "fence", (HEDGE_W + 0.01 + PLOT_W) / 2, PLOT_D - FENCE_W / 2 - 0.005,
                 PLOT_W - HEDGE_W - 0.01 - 0.005, FENCE_W, round(rng.uniform(1.2, 1.8), 1), "brown")
        # two beds: a tree bed at the front on the side away from the driveway, one in the back garden
        u_front = PLOT_W - 2.8 if u_garage < half else 2.8
        self.add("front bed", "garden_bed", u_front, 3.0, 3.4, 3.0, 0.3, "grey")
        self.add("back bed", "garden_bed", rng.uniform(3.0, PLOT_W - 3.0), rng.uniform(20.0, 27.0), 4.0, 3.0, 0.3,
                 rng.choice(("grey", "brown")))
        return self.pieces


def plots() -> list[Plot]:
    """Every plot, numbered along its side from the west: the south plots even, the north plots odd."""
    out = []
    for north, edge in ((False, "south"), (True, "north")):
        for i, x0 in enumerate(PLOT_XS):
            out.append(Plot(2 * (i + 1) - (1 if north else 0), float(x0), north, i in DROP_PLOTS[edge]))
    return out


def street() -> list[dict]:
    """The street: sidewalks (kerbs on the carriageway's side), verges against the plots, kerbside lanes."""
    out = []
    for name, y0, y1, yaw in (("south sidewalk", ROAD_Y0 - WALK_W, ROAD_Y0, PI2),
                              ("north sidewalk", ROAD_Y1, ROAD_Y1 + WALK_W, -PI2)):
        cx, cy, w = WALK_SURFACE
        out.append(piece(name, "sidewalk", (LENGTH / 2, (y0 + y1) / 2), (WALK_W, LENGTH, WALK_H), yaw, "grey",
                         surfaces=(("sidewalk", [cx, cy, w, LENGTH - 2.0], WALK_H),)))
    out.append(piece("south verge", "verge", (LENGTH / 2, SOUTH_EDGE + VERGE_W / 2), (VERGE_W, LENGTH, VERGE_H), PI2,
                     "green"))
    out.append(piece("north verge", "verge", (LENGTH / 2, NORTH_EDGE - VERGE_W / 2), (VERGE_W, LENGTH, VERGE_H), -PI2,
                     "green"))
    for x in LANES["south"]:
        out.append(lane("kerbside bays", (x, ROAD_Y0 + LANE_W / 2), PI2, 2))
    for x in LANES["north"]:
        out.append(lane("kerbside bays", (x, ROAD_Y1 - LANE_W / 2), -PI2, 2))
    return out


def green() -> list[dict]:
    x0, x1 = GREEN_X
    cx = (x0 + x1) / 2
    out = [piece("green lawn", "lawn", (cx, SOUTH_EDGE / 2), (x1 - x0, SOUTH_EDGE, LAWN_H), 0.0, "green", blocks=False),
           piece("playground bed", "garden_bed", (cx + 0.5, 16.0), (14.0, 9.0, 0.3), 0.0, "brown")]
    out += [piece("green tree bed", "garden_bed", (x, y), (3.0, 3.0, 0.3), 0.0, "grey")
            for x, y in ((x0 + 3.0, 30.5), (x1 - 3.0, 30.5), (x1 - 3.0, 4.0))]
    return out


def passage() -> list[dict]:
    x0, x1 = PASSAGE_X
    return [piece("passage path", "path", ((x0 + x1) / 2, SOUTH_EDGE / 2), (2.0, SOUTH_EDGE, LAWN_H), 0.0, "grey",
                  blocks=False)]


def car_park() -> list[dict]:
    """The nest on bare asphalt with nothing within 3.65 m of it, and two lanes of bays against the car park's edges."""
    x0, x1 = CAR_PARK_X
    cx = (x0 + x1) / 2
    near = lane("car park bays", (cx, NORTH_EDGE + LANE_W / 2 + 0.2), PI2, CAR_PARK_BAYS)
    far = lane("car park bays", (cx, BOUNDS[3] - LANE_W / 2 - 0.2), -PI2, CAR_PARK_BAYS)
    return [piece("nest", "nest", NEST, (1.6, 1.6, 0.02), PI2, "grey", blocks=False), near, far]


def regions() -> list[dict]:
    """The street first, then the south plots, the passage and the green, the north plots and the car park."""
    south = [region(p.addr, "garden", (p.x0, 0.0, p.x0 + PLOT_W, SOUTH_EDGE)) for p in plots() if not p.north]
    north = [region(p.addr, "garden", (p.x0, NORTH_EDGE, p.x0 + PLOT_W, BOUNDS[3])) for p in plots() if p.north]
    others = [region("passage", "passage", (PASSAGE_X[0], 0.0, PASSAGE_X[1], SOUTH_EDGE)),
              region("green", "green", (GREEN_X[0], 0.0, GREEN_X[1], SOUTH_EDGE)),
              region("car park", "car park", (CAR_PARK_X[0], NORTH_EDGE, CAR_PARK_X[1], BOUNDS[3]))]
    rows = [region("street", "street", (0.0, SOUTH_EDGE, LENGTH, NORTH_EDGE)), *south, *others, *north]
    return [{"id": f"R{i}"} | row for i, row in enumerate(rows, 1)]


def furniture() -> list[dict]:
    pieces = street() + [p for plot in plots() for p in plot.build()] + green() + passage() + car_park()
    return [{"id": f"F{i:03d}"} | p | {"surfaces": [{"id": f"S{j}", "kind": k, "rect": rect, "z": z}
                                                    for j, (k, rect, z) in enumerate(p["surfaces"], 1)]}
            for i, p in enumerate(pieces, 1)]


def spec() -> dict:
    return {"name": NAME, "catalog": "residential", "depth_unit": "cm", "outdoor": OUTDOOR,
            "bounds": BOUNDS, "wall_thickness": 0.6, "wall_height": WALL_H, "regions": regions(), "walls": [],
            "openings": [], "windows": [], "furniture": furniture(), "ceiling_lights": []}


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
    out.write_bytes(dump(spec()).encode("utf-8"))  # LF on every platform: the file is read by its bytes
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv)
