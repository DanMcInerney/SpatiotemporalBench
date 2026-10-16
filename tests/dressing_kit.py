"""Shared pieces of the dressing tests: the maps as raw dicts (the form a layout takes; a bundled map, or a held-out one
that `STBENCH_MAPS` names, through `map_files.map_bytes`), a module of `render/`
loaded from its file (render/ is no package, and its modules import each other by name), and a small synthetic map whose
every number can be checked by hand."""

from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from stbench.map_files import map_bytes

RENDER = Path(__file__).resolve().parents[1] / "render"


def raw_map(name: str) -> dict:
    return json.loads(map_bytes(name))


@contextmanager
def render_on_path():
    """`render/` on `sys.path` for the duration (a layout imports `dressing_spec` and its own parts by name)."""
    sys.path.insert(0, str(RENDER))
    try:
        yield
    finally:
        sys.path.remove(str(RENDER))


def render_module(name: str):
    """`render/<name>.py`, imported by name with `render/` on the path (a dressing layout has no bpy, and it imports its
    siblings, `dressing_spec` first, by name, so one module object serves all)."""
    with render_on_path():
        return importlib.import_module(name)


def layout_path(catalog: str) -> Path:
    return RENDER / f"dressing_layout_{catalog}.py"


def layout_of(catalog: str, spec: dict) -> list:
    """The dressing a catalog's layout module (`render/dressing_layout_<catalog>.py`) makes for a map."""
    with render_on_path():
        return render_module(f"dressing_layout_{catalog}").layout(spec)


def _wall(i: int, a, b, exterior: bool = True) -> dict:
    return {"id": f"W0{i}", "a": a, "b": b, "thickness": 0.1, "height": 2.5, "exterior": exterior}


# A 4 x 3 m room with a counter (with a surface) and a fridge along the south wall, a window above the south wall's
# middle, a door in the north wall that swings into the room, and a chair. Walls are 0.1 m thick, so the faces the room
# sees are at x = 0.05 and 3.95 and y = 0.05 and 2.95; the ceiling is at 2.5 m.
SYNTH: dict = {
    "name": "synth", "bounds": [0, 0, 4.0, 3.0], "wall_thickness": 0.1,
    "wall_height": 2.5,
    "regions": [{"id": "R1", "name": "room", "polygon": [[0.05, 0.05], [3.95, 0.05], [3.95, 2.95], [0.05, 2.95]],
                 "anchor": [2.0, 1.5], "level": 0, "z0": 0.0, "z1": 2.5}],
    "walls": [_wall(1, [0, 0], [4, 0]), _wall(2, [4, 0], [4, 3]), _wall(3, [0, 3], [4, 3]), _wall(4, [0, 0], [0, 3])],
    "openings": [{"id": "D1", "type": "door", "wall": "W03", "at": 3.0, "width": 0.9, "height": 2.05, "hinge": "a",
                  "swing": "right", "interior": False, "initial_open": False, "between": ["R1", "outside"]}],
    "windows": [{"id": "N1", "wall": "W01", "at": 2.0, "width": 1.0, "sill": 1.0, "height": 1.0}],
    "furniture": [
        {"id": "F01", "name": "kitchen counter", "kind": "counter", "mobility": "fixed", "center": [0.65, 0.35],
         "size": [0.6, 1.2, 0.9], "yaw": 1.5707963267948966, "color": "oak", "blocks": True,
         "surfaces": [{"id": "S01", "kind": "kitchen counter", "rect": [0.0, 0.0, 0.54, 1.14], "z": 0.9}]},
        {"id": "F02", "name": "fridge", "kind": "fridge", "mobility": "fixed", "center": [3.6, 0.4],
         "size": [0.7, 0.7, 1.8], "yaw": 0.0, "color": "white", "blocks": True, "surfaces": []},
        {"id": "L01", "name": "chair", "kind": "chair", "mobility": "light", "center": [2.0, 1.5],
         "size": [0.533, 0.401, 0.9], "yaw": 0.0, "color": "beige", "blocks": True, "surfaces": []},
    ],
    "ceiling_lights": [{"region": "R1", "pos": [2.0, 1.5, 2.45]}],
}
