"""The maker of `src/stbench/maps/house_floor.json`: the ground floor of a large detached two-storey house (M4's public
home layout, run by `vacuum/2`; `src/stbench/maps/README.md` describes the result).

    python tools/maps/house_floor.py        # writes src/stbench/maps/house_floor.json

`spec()` builds the map as the dict the file holds, from the plan below. It imports `stbench.mapspec` only, to check
the result (`check_map`) before it writes, so a hand edit of the JSON that this file would undo is a test failure
(`tests/test_house_floor_map.py`). Every number is a wall line, a size or a place chosen to meet a rule of the code that
reads the map: regions are axis-aligned rectangles on the walls' inner faces; furniture is at a quarter turn; no
blocking piece stands within `width / 2 + 0.3` m of an interior opening's centre; the dock's apron is clear; a door
swings into a room, never across the corridor.

    y 9.0 +--------+---------+------+------+------+-----------+
          |        | utility |cloak | bath |      |           |
          | kitchen+---+-----+------+------+ bedroom           |
          |        |   corridor            |      |           |
    y 5.0 +---+----+---+------arch----------+------+-----------+
          |dining arch| living  |   hall    |    study         |
    y 4.2 +--------+  |         | (stairs   |                  |
          | (kitchen above)     |  east)    |                  |
    y 0   +--------+--+---------+-----------+------------------+
         x 0      2.9         6.0         9.2               11.6

The stairs rise along the hall's east wall, their foot facing the front door's wall (south); the floor above is not
modelled. A task of the vacuum takes about 38 s for each square metre of floor it can reach (strips 0.2 m apart at
0.3 m/s, `job_vacuum.STRIP_M`), and the job's longest task is `TASK_MAX_S` (1,200 s): two rooms of 15 m2 and 11 m2 are
a task of 1,000 s, so no room here is larger than about 15 m2 and the two largest, the hall and the living room, are
15.2 and 14.7. Rooms are named by their kind (the map has no `kind` key): the household's clutter matches room names.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from stbench import mapspec

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "stbench" / "maps" / "house_floor.json"
NAME, W, H, T, WALL_H = "house_floor", 11.4, 8.8, 0.1, 2.5
PI, HALF_PI = math.pi, math.pi / 2

# The rooms as rectangles on the wall centre lines (x0, y0, x1, y1), in the order of their ids (R1...).
ROOMS: dict[str, tuple[float, float, float, float]] = {
    "hall": (6.0, 0.0, 9.0, 5.0), "corridor": (2.9, 5.0, 8.3, 6.4), "living room": (2.9, 0.0, 6.0, 5.0),
    "dining room": (0.0, 0.0, 2.9, 4.2), "kitchen": (0.0, 4.2, 2.9, 8.8), "utility room": (2.9, 6.4, 4.7, 8.8),
    "cloakroom": (4.7, 6.4, 6.2, 8.8), "study": (9.0, 0.0, 11.4, 5.0), "bedroom": (8.3, 5.0, 11.4, 8.8),
    "bathroom": (6.2, 6.4, 8.3, 8.8)}
ANCHORS = {"hall": (7.4, 3.0), "corridor": (5.6, 5.7), "living room": (4.3, 3.7), "dining room": (1.9, 3.6),
           "kitchen": (1.7, 5.7), "utility room": (3.6, 7.4), "cloakroom": (5.45, 7.2), "study": (10.2, 3.6),
           "bedroom": (9.4, 6.1), "bathroom": (7.2, 7.4)}
LIGHTS = {"hall": ((7.5, 1.4), (7.5, 3.8)), "corridor": ((5.6, 5.7),), "living room": ((3.9, 1.2), (4.4, 3.6)),
          "dining room": ((1.3, 3.5),), "kitchen": ((1.7, 6.4),), "utility room": ((3.7, 7.5),),
          "cloakroom": ((5.45, 7.5),), "study": ((10.2, 2.5),), "bedroom": ((9.7, 6.3),),
          "bathroom": ((7.25, 7.6),)}
EXTERIOR = {"S": "W01", "E": "W02", "N": "W03", "W": "W04"}


def inner(room: str) -> tuple[float, float, float, float]:
    """The room's floor outline: the wall lines pulled in by half a wall."""
    x0, y0, x1, y1 = ROOMS[room]
    return round(x0 + T / 2, 4), round(y0 + T / 2, 4), round(x1 - T / 2, 4), round(y1 - T / 2, 4)


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def walls() -> list[dict]:
    """The exterior walls (W01-W04, as `apartment_a`'s) and one interior wall for every maximal run of room edges on a
    line, so each region edge lies within one wall (`render/dressing_home_plan` finds a face by its wall)."""
    out = [{"id": f"W0{i}", "a": a, "b": b, "thickness": T, "height": WALL_H, "exterior": True}
           for i, (a, b) in enumerate((([0, 0], [W, 0]), ([W, 0], [W, H]), ([0, H], [W, H]), ([0, 0], [0, H])), 1)]
    runs: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for x0, y0, x1, y1 in ROOMS.values():
        for key, span in ((("h", y0), (x0, x1)), (("h", y1), (x0, x1)), (("v", x0), (y0, y1)), (("v", x1), (y0, y1))):
            runs.setdefault(key, []).append(span)
    n = 5
    for (axis, line), spans in sorted(runs.items()):
        if line in (0, W if axis == "v" else H):
            continue
        for lo, hi in _merge(spans):
            a, b = ([lo, line], [hi, line]) if axis == "h" else ([line, lo], [line, hi])
            out.append({"id": f"W{n:02d}", "a": a, "b": b, "thickness": T, "height": WALL_H, "exterior": False})
            n += 1
    return out


def _wall_for(ws: list[dict], axis: str, line: float, pos: float) -> dict:
    """The wall on a line (`axis` h: y = line, v: x = line) whose run holds `pos` along it."""
    for w in ws:
        horizontal = w["a"][1] == w["b"][1]
        c0, c1 = (w["a"][0], w["b"][0]) if horizontal else (w["a"][1], w["b"][1])
        if horizontal == (axis == "h") and w["a"][1 if horizontal else 0] == line and c0 <= pos <= c1:
            return w
    raise ValueError(f"no wall on the {axis} line {line} holds {pos}")


def _shared(a: str, b: str) -> tuple[str, float]:
    """The line two rooms share: ('h', y) or ('v', x)."""
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = ROOMS[a], ROOMS[b]
    if ax1 == bx0 or bx1 == ax0:
        return "v", ax1 if ax1 == bx0 else ax0
    if ay1 == by0 or by1 == ay0:
        return "h", ay1 if ay1 == by0 else ay0
    raise ValueError(f"{a} and {b} share no wall")


def opening(ws: list[dict], oid: str, rooms: tuple[str, str | None], pos: float, width: float, into: str | None = None,
            hinge: str = "a", height: float = 2.05) -> dict:
    """A door (`into` names the room its leaf swings into; `hinge` the jamb nearer the wall's `a` end, a or b) or, with
    no `into`, an arch between two rooms, `pos` along their shared wall. `rooms[1]` None is the outside (the front door,
    on `rooms[0]`'s south wall)."""
    ids = {name: f"R{i}" for i, name in enumerate(ROOMS, 1)}
    if rooms[1] is None:
        return {"id": oid, "type": "door", "wall": ws[0]["id"], "at": pos, "width": width, "height": height,
                "hinge": hinge, "swing": "left", "interior": False, "initial_open": False,
                "between": ["outside", ids[rooms[0]]]}
    axis, line = _shared(rooms[0], rooms[1])
    w = _wall_for(ws, axis, line, pos)
    o = {"id": oid, "type": "arch" if into is None else "door", "wall": w["id"],
         "at": round(pos - (w["a"][0] if axis == "h" else w["a"][1]), 4), "width": width, "height": height}
    if into is not None:
        cx0, cy0, cx1, cy1 = ROOMS[into]
        left = (cy0 + cy1) / 2 > line if axis == "h" else (cx0 + cx1) / 2 < line  # left of +x is north, of +y is west
        o |= {"hinge": hinge, "swing": "left" if left else "right"}
    o["interior"] = True
    if into is not None:
        o["initial_open"] = False
    o["between"] = [ids[rooms[0]], ids[rooms[1]]]
    return o


# (id, rooms, position along the shared wall, width, the room it swings into, hinge, height)
OPENINGS = (
    ("D0", ("hall", None), 7.5, 0.95, None, "a", 2.1),
    ("D1", ("hall", "living room"), 1.1, 0.8, "living room", "b", 2.05),
    ("D2", ("hall", "study"), 0.75, 0.85, "study", "a", 2.05), ("O1", ("hall", "corridor"), 6.95, 1.2, None, "a", 2.1),
    ("O2", ("living room", "dining room"), 3.3, 1.2, None, "a", 2.1),
    ("D3", ("corridor", "kitchen"), 5.7, 0.8, "kitchen", "a", 2.05),
    ("D4", ("corridor", "cloakroom"), 5.45, 0.8, "cloakroom", "b", 2.05),
    ("D5", ("corridor", "utility room"), 3.8, 0.8, "utility room", "b", 2.05),
    ("D6", ("corridor", "bedroom"), 5.7, 0.85, "bedroom", "a", 2.05),
    ("D7", ("bedroom", "bathroom"), 7.3, 0.8, "bathroom", "a", 2.05),
    ("D8", ("kitchen", "dining room"), 2.1, 0.85, "dining room", "b", 2.05),
    ("D9", ("kitchen", "utility room"), 7.4, 0.8, "utility room", "b", 2.05))
# (id, exterior wall side, position along it, width, sill, height)
WINDOWS = (("N1", "S", 1.45, 0.8, 1.0, 1.2), ("N2", "S", 4.3, 1.2, 0.9, 1.2), ("N3", "S", 10.2, 0.8, 1.0, 1.2),
           ("N4", "W", 1.2, 0.8, 1.1, 1.0), ("N5", "W", 6.55, 1.0, 1.1, 1.0), ("N6", "N", 1.5, 0.8, 1.1, 1.0),
           ("N7", "N", 3.8, 0.7, 1.3, 0.7), ("N8", "N", 5.45, 0.6, 1.3, 0.6), ("N9", "N", 7.2, 0.6, 1.3, 0.6),
           ("N10", "N", 9.95, 1.4, 1.1, 1.2), ("N11", "E", 3.9, 0.8, 1.0, 1.1), ("N12", "E", 7.4, 0.8, 0.9, 1.2))


class Furnishing:
    """The pieces of the map: ids in the order they are added, surfaces numbered across the map."""

    def __init__(self) -> None:
        self.pieces: list[dict] = []
        self._fixed = self._light = self._surface = 0

    def add(self, kind: str, name: str, center: tuple[float, float], size: tuple[float, float, float], yaw: float,
            color: str, surfaces: tuple[tuple[str, tuple[float, float, float, float], float], ...] = (),
            under: float = 0.0, light: bool = False, blocks: bool = True) -> None:
        if light:
            self._light += 1
            pid = f"L{self._light:02d}"
        else:
            self._fixed += 1
            pid = f"F{self._fixed:02d}"
        shown = []
        for skind, rect, z in surfaces:
            self._surface += 1
            shown.append({"id": f"S{self._surface:02d}", "kind": skind, "rect": [round(v, 4) for v in rect], "z": z})
        p = {"id": pid, "name": name, "kind": kind, "mobility": "light" if light else "fixed",
             "center": [round(center[0], 4), round(center[1], 4)], "size": list(size), "yaw": yaw, "color": color,
             "blocks": blocks, "surfaces": shown}
        if under:
            p["under_m"] = under
        self.pieces.append(p)

    def on_wall(self, room: str, side: str, along: float, kind: str, name: str, size: tuple[float, float, float],
                color: str, **kw) -> None:
        """A piece with its back to a wall of the room, centred at `along` (x on a south or north wall, y on a west or
        east one), facing into the room."""
        x0, y0, x1, y1 = inner(room)
        d = size[0] / 2
        centre, yaw = {"S": ((along, y0 + d), HALF_PI), "N": ((along, y1 - d), -HALF_PI), "W": ((x0 + d, along), 0.0),
                       "E": ((x1 - d, along), PI)}[side]
        self.add(kind, name, centre, size, yaw, color, **kw)


def top(size: tuple[float, float, float], kind: str, margin: float = 0.03
        ) -> tuple[str, tuple[float, float, float, float], float]:
    """A surface over a piece's whole top, `margin` in from every edge."""
    return kind, (0.0, 0.0, size[0] - 2 * margin, size[1] - 2 * margin), size[2]


def shelves(size: tuple[float, float, float], levels: tuple[float, ...]) -> tuple:
    return tuple(("shelf", (0.0, 0.0, size[0] - 0.06, size[1] - 0.06), z) for z in levels)


def furnish() -> list[dict]:
    f = Furnishing()
    # -- the hall: the front door D0 in the south wall, the stairs up the east wall (their foot at y 1.55, facing the
    # door's wall), the dock on the west wall with its apron clear, the console, the shoe rack and the mat by the door
    f.add("stairs", "stairs", (8.5, 3.25), (3.4, 0.9, 2.5), -HALF_PI, "white")
    f.on_wall("hall", "W", 3.75, "dock", "dock", (0.12, 0.28, 0.10), "black")
    f.on_wall("hall", "W", 2.4, "hall_console", "hall console", (0.3, 0.9, 0.8), "black",
              surfaces=(top((0.3, 0.9, 0.8), "hall console"),), under=0.64)
    f.on_wall("hall", "S", 6.55, "shoe_rack", "shoe rack", (0.3, 0.7, 0.35), "black",
              surfaces=(top((0.3, 0.7, 0.35), "shoe rack"),))
    f.add("floor_mat", "floor mat", (7.5, 0.45), (0.6, 0.9, 0.01), HALF_PI, "brown",
          surfaces=(("floor mat", (0.0, 0.0, 0.5, 0.8), 0.01),), blocks=False)
    # -- the living room: the television on the north wall, the sofa facing it
    f.on_wall("living room", "N", 4.3, "tv_console", "tv console", (0.4, 1.6, 0.5), "black",
              surfaces=(top((0.4, 1.6, 0.5), "tv console"),))
    f.on_wall("living room", "S", 4.3, "sofa", "sofa", (0.9, 2.0, 0.85), "grey",
          surfaces=(("sofa", (0.125, 0.0, 0.65, 1.7), 0.45),), under=0.08)
    f.add("coffee_table", "coffee table", (4.3, 2.7), (0.6, 1.1, 0.42), HALF_PI, "oak",
          surfaces=(top((0.6, 1.1, 0.42), "coffee table"),), under=0.10)
    f.add("rug", "living rug", (4.3, 2.7), (1.8, 2.0, 0.01), HALF_PI, "blue",
          surfaces=(top((1.8, 2.0, 0.01), "rug", 0.05),), blocks=False)
    f.on_wall("living room", "E", 3.6, "bookshelf", "living bookshelf", (0.3, 0.9, 1.8), "oak",
              surfaces=shelves((0.3, 0.9, 1.8), (0.4, 0.9, 1.4)))
    # -- the dining room: a table for four, a sideboard on the west wall
    f.add("dining_table", "dining table", (1.3, 2.0), (1.6, 0.9, 0.75), HALF_PI, "oak",
          surfaces=(top((1.6, 0.9, 0.75), "dining table"),), under=0.64)
    f.on_wall("dining room", "W", 3.3, "hall_console", "sideboard", (0.3, 0.9, 0.8), "black",
              surfaces=(top((0.3, 0.9, 0.8), "hall console"),), under=0.64)
    for centre, yaw, colour in (((1.3, 0.9135), HALF_PI, "beige"), ((1.3, 3.0865), -HALF_PI, "black"),
                                ((0.5635, 2.0), 0.0, "beige"), ((2.0365, 2.0), PI, "black")):
        f.add("chair", "chair", centre, (0.533, 0.401, 0.9), yaw, colour, under=0.42, light=True)
    # -- the kitchen: a run along the west wall under the window, a counter on the north wall, the fridge by the door
    f.on_wall("kitchen", "W", 4.6, "fridge", "fridge", (0.7, 0.7, 1.8), "white")
    y = 4.95
    for kind, name, width, colour in (("counter", "kitchen counter", 1.2, "oak"),
                                      ("sink_unit", "sink unit", 0.8, "grey"),
                                      ("counter", "kitchen counter", 0.8, "oak"), ("stove", "stove", 0.6, "black")):
        size = (0.6, width, 0.9)
        surface = ("sink edge", (0.15, 0.0, 0.25, 0.7), 0.9) if kind == "sink_unit" else top(size, "kitchen counter")
        f.on_wall("kitchen", "W", y + width / 2, kind, name, size, colour,
                  surfaces=() if kind == "stove" else (surface,))
        y += width
    f.on_wall("kitchen", "N", 1.6, "counter", "kitchen counter", (0.6, 1.2, 0.9), "oak",
              surfaces=(top((0.6, 1.2, 0.9), "kitchen counter"),))
    f.add("stool", "stool", (1.6, 6.6), (0.385, 0.406, 0.579), 0.0, "brown", under=0.529, light=True)
    # -- the utility room: the washer-dryer against the east wall, a short counter on the north wall
    f.on_wall("utility room", "E", 7.7, "washer_dryer", "washer dryer", (0.6, 0.6, 1.7), "white")
    f.on_wall("utility room", "N", 3.7, "counter", "utility counter", (0.6, 0.8, 0.9), "oak",
              surfaces=(top((0.6, 0.8, 0.9), "kitchen counter"),))
    f.add("laundry_basket", "laundry basket", (3.45, 7.0), (0.4, 0.4, 0.5), 0.0, "beige", light=True)
    # -- the cloakroom: a toilet on the west wall, a vanity under the window
    f.on_wall("cloakroom", "W", 7.8, "toilet", "toilet", (0.6, 0.46, 0.4), "white")
    f.on_wall("cloakroom", "N", 5.45, "vanity", "cloakroom vanity", (0.5, 0.8, 0.85), "white",
              surfaces=(("sink edge", (0.0, 0.0, 0.44, 0.74), 0.85),))
    # -- the study: the desk on the east wall with its chair, a bookshelf on the west wall
    f.on_wall("study", "E", 2.2, "desk", "desk", (0.6, 1.2, 0.75), "oak", surfaces=(top((0.6, 1.2, 0.75), "desk"),),
              under=0.60)
    f.add("chair", "chair", (10.4635, 2.2), (0.533, 0.401, 0.9), 0.0, "black", under=0.42, light=True)
    f.on_wall("study", "W", 3.6, "bookshelf", "study bookshelf", (0.3, 0.9, 1.8), "oak",
              surfaces=shelves((0.3, 0.9, 1.8), (0.4, 0.9, 1.4)))
    # -- the bedroom: the bed's head on the north wall between two nightstands, the wardrobe on the east wall
    f.on_wall("bedroom", "N", 9.95, "bed", "bed", (2.0, 1.6, 0.5), "white", surfaces=(top((2.0, 1.6, 0.5), "bed"),),
              under=0.05)
    for x in (8.91, 10.99):
        f.on_wall("bedroom", "N", x, "nightstand", "nightstand", (0.4, 0.42, 0.55), "oak",
                  surfaces=(top((0.4, 0.42, 0.55), "nightstand", 0.02),))
    f.on_wall("bedroom", "E", 6.0, "wardrobe", "wardrobe", (0.6, 1.2, 2.0), "white")
    f.add("rug", "bedroom rug", (9.8, 6.2), (1.2, 1.8, 0.01), -HALF_PI, "beige",
          surfaces=(top((1.2, 1.8, 0.01), "rug", 0.05),), blocks=False)
    # -- the bathroom (the bedroom's): the tub along the north wall under the window, the toilet, the vanity and a shelf
    f.on_wall("bathroom", "N", 7.2, "bathtub", "bathtub", (0.75, 1.7, 0.55), "white")
    f.on_wall("bathroom", "W", 7.6, "toilet", "toilet", (0.6, 0.46, 0.4), "white")
    f.on_wall("bathroom", "S", 6.7, "vanity", "vanity", (0.5, 0.8, 0.85), "white",
              surfaces=(("sink edge", (0.0, 0.0, 0.44, 0.74), 0.85),))
    f.on_wall("bathroom", "S", 7.5, "bath_shelf", "bathroom shelf", (0.25, 0.6, 1.5), "white",
              surfaces=shelves((0.25, 0.6, 1.5), (0.5, 1.0, 1.4)))
    return f.pieces


def spec() -> dict:
    ws = walls()
    regions = [{"id": f"R{i}", "name": name, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "anchor": list(ANCHORS[name]), "level": 0, "z0": 0.0, "z1": WALL_H}
               for i, name in enumerate(ROOMS, 1) for x0, y0, x1, y1 in (inner(name),)]
    ids = {name: f"R{i}" for i, name in enumerate(ROOMS, 1)}
    by_wall = {s: next(w for w in ws if w["id"] == wid) for s, wid in EXTERIOR.items()}
    windows = [{"id": wid, "wall": by_wall[side]["id"], "at": at, "width": width, "sill": sill, "height": height}
               for wid, side, at, width, sill, height in WINDOWS]
    lights = [{"region": ids[name], "pos": [x, y, 2.45]} for name, spots in LIGHTS.items() for x, y in spots]
    return {"name": NAME, "bounds": [0, 0, W, H], "wall_thickness": T,
            "wall_height": WALL_H, "regions": regions, "walls": ws,
            "openings": [opening(ws, *o) for o in OPENINGS], "windows": windows, "furniture": furnish(),
            "ceiling_lights": lights}


def render(s: dict) -> str:
    """The file's text: one line a region, wall, opening, window, piece and light, as the other maps are written."""
    def line(x: object) -> str:
        return json.dumps(x, separators=(", ", ": "))

    parts = [f'  "name": {line(s["name"])}', f'  "bounds": {line(s["bounds"])}',
             f'  "wall_thickness": {line(s["wall_thickness"])}', f'  "wall_height": {line(s["wall_height"])}']
    for key in ("regions", "walls", "openings", "windows", "furniture", "ceiling_lights"):
        parts.append(f'  "{key}": [\n' + ",\n".join("    " + line(x) for x in s[key]) + "\n  ]")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def main() -> int:
    s = spec()
    problems = mapspec.check_map(mapspec.parse_map(s))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    OUT.write_text(render(s), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
