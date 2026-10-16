"""What the zone-shape tests share: `apartment_a`, and a small synthetic map for the window of a surface.

**The synthetic map** (metres; x east, y north; a piece's yaw is the direction its depth runs):
- the table `F01` at (2.0, 3.0), 0.6 deep x 1.2 wide x 0.75 high, turned to face south (yaw pi/2): footprint x
  1.4..2.6, y 2.7..3.3; its one surface is 0.5 x 1.1 m at 0.75 m, so x 1.45..2.55, y 2.75..3.25, and pulled in 0.03 m
  x 1.48..2.52, y 2.78..3.22;
- the fridge `F02` at (6.0, 3.0) (no surface), the shelf `F03` at (8.0, 3.0) (two equal surfaces, 0.4 and 0.9 m), the
  stool `F04` at (4.0, 3.0) (0.26 m surface), the bench `F07` at (4.0, 6.0) (a small surface listed before the larger);
- a light chair `F05` at (2.0, 1.9) and a rug `F06` at (2.0, 2.3), both in the way south of the table, and neither
  met by a ray: the chair moves and the rug blocks nothing.
"""

from __future__ import annotations

import math

from stbench.geometry import Rect, corners
from stbench.mapspec import Furniture, MapSpec, Surface, load_map

APARTMENT = load_map("apartment_a")
NORTH = math.pi / 2


def extents(rect: Rect) -> tuple[float, float, float, float]:
    """The x and y extents (x0, x1, y0, y1) of the rectangle, to 6 decimals."""
    xs, ys = zip(*corners(rect), strict=True)
    return round(min(xs), 6), round(max(xs), 6), round(min(ys), 6), round(max(ys), 6)


def piece(pid: str, kind: str, centre, size, yaw: float = 0.0, *, surfaces=(), mobility: str = "fixed",
          blocks: bool = True) -> Furniture:
    return Furniture(pid, kind, kind, mobility, centre, size, yaw, "grey", blocks, tuple(surfaces), 0.0)


def top(sid: str, rect, z: float) -> Surface:
    return Surface(sid, "top", rect, z)


TABLE = piece("F01", "table", (2.0, 3.0), (0.6, 1.2, 0.75), NORTH, surfaces=[top("S1", (0, 0, 0.5, 1.1), 0.75)])
FRIDGE = piece("F02", "fridge", (6.0, 3.0), (0.7, 0.7, 1.8), NORTH)
SHELF = piece("F03", "shelf", (8.0, 3.0), (0.3, 0.9, 1.5), NORTH,
              surfaces=[top("S2", (0, 0, 0.24, 0.84), 0.4), top("S3", (0, 0, 0.24, 0.84), 0.9)])
STOOL = piece("F04", "stool", (4.0, 3.0), (0.3, 0.3, 0.45), surfaces=[top("S4", (0, 0, 0.26, 0.26), 0.45)])
CHAIR = piece("F05", "chair", (2.0, 1.9), (0.4, 0.4, 0.5), mobility="light",
              surfaces=[top("S5", (0, 0, 0.3, 0.3), 0.45)])
RUG = piece("F06", "rug", (2.0, 2.3), (0.8, 0.8, 0.01), blocks=False)
BENCH = piece("F07", "bench", (4.0, 6.0), (0.5, 1.6, 0.5), NORTH,
              surfaces=[top("S6", (0, 0, 0.3, 0.4), 0.5), top("S7", (0, 0, 0.4, 1.4), 0.55)])
SYNTHETIC = MapSpec("synthetic", (0.0, 0.0, 10.0, 10.0), 0.1, 2.6, (), (), (), (),
                    (TABLE, FRIDGE, SHELF, STOOL, CHAIR, RUG, BENCH), ())
