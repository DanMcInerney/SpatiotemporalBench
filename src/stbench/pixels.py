"""A script's pictures read as points of the map (0.7.0): a thing's pixels in a keyframe, through that keyframe's depth.

`ScriptPixels(paths)` reads a script's own files: `private/<sid>/instances/` (the instance image, whose pass index
`instance_map.json` gives to each entity id) and `private/<sid>/depth/`, with the pose rows' camera. It is what the
compiler's `Inputs.points_of` and `Inputs.gaps_of` are for a compile from files and what the oracle channel's
`floor_hull` is built from (`perception.py`): the same points, so the channel and the gold agree. It reads no keyframe
but the one it is asked for, so a gold that names keyframe `j` reads frame `j` alone, and nothing until the first ask,
so a script without instance images compiles every family that does not ask.

The module is neutral: numpy, `camera`, `nearest_part` and `render_encode`. `paths` is any object with the script's
`camera`, `pose`, `instance_map` files and its `instances(k)`, `depth(k)` image paths and `id` (`case.ScriptPaths`), so
no truth module is imported.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import grid
from .camera import Camera
from .nearest_part import visible_points
from .render_encode import read_png16


class Frame(NamedTuple):
    """Every pixel of a keyframe that has a depth: its (N, 3) map-frame points (`grid.points`' order) and, for each, the
    id of the entity whose instance it belongs to ("" for the walls, the floor and what no entity covers)."""

    points: np.ndarray
    owner: np.ndarray


PointsOf = Callable[[int, str], np.ndarray]   # (script keyframe, entity id) -> the thing's (N, 3) map-frame points
FRAMES_KEPT = 4    # keyframes whose images stay in memory (a look point asks of one frame, then the next)
THINGS_KEPT = 64   # (keyframe, thing) point sets that stay in memory


class ScriptPixels:
    """One script's pictures. `points(k, entity)` is the (N, 3) map-frame points of the entity's pixels in keyframe `k`
    that have a depth (N may be 0), `gaps(k, entity)` how many of its pixels have none (depth 0: too near, too far, the
    robot's own body). KeyError for an id that is no instance of the script or a keyframe the pose file lacks;
    FileNotFoundError for an image that is not there."""

    def __init__(self, paths):
        self.paths = paths
        self._frames: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._things: OrderedDict[tuple[int, str], tuple[np.ndarray, int]] = OrderedDict()
        self._script: dict = {}   # the camera, the instance indices by id and the pose rows, read at the first ask

    def _images(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k not in self._frames:
            self._frames[k] = (read_png16(self.paths.instances(k)), read_png16(self.paths.depth(k)))
            while len(self._frames) > FRAMES_KEPT:
                self._frames.popitem(last=False)
        return self._frames[k]

    def _read_script(self) -> None:
        if not self._script:
            read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))   # noqa: E731
            self._script["cam"] = Camera.load(self.paths.camera)
            self._script["passes"] = {eid: int(p) for p, eid in read(self.paths.instance_map).items()}
            lines = Path(self.paths.pose).read_text(encoding="utf-8").splitlines()
            self._script["rows"] = {row["k"]: row for row in (json.loads(x) for x in lines if x.strip())}

    def _thing(self, k: int, entity: str) -> tuple[np.ndarray, int]:
        self._read_script()
        if entity not in self._script["passes"]:
            raise KeyError(f"{entity!r} is no instance of script {self.paths.id}")
        if (k, entity) not in self._things:
            index, depth = self._images(k)
            camera, pass_index = self._script["rows"][k]["camera"], self._script["passes"][entity]
            points = visible_points(index, depth, self._script["cam"], camera["pos"], camera["rot"], pass_index)
            self._things[k, entity] = (points, int(np.count_nonzero(index == pass_index)) - len(points))
            while len(self._things) > THINGS_KEPT:
                self._things.popitem(last=False)
        return self._things[k, entity]

    def frame(self, k: int) -> Frame:
        """Keyframe `k`'s whole depth image as points with the entity each pixel shows (`Frame`): what a screen needs to
        say which thing is the nearest in a corridor."""
        self._read_script()
        index, depth = self._images(k)
        camera = self._script["rows"][k]["camera"]
        rows, cols = np.nonzero(depth > 0)
        pts = grid.points(depth, self._script["cam"], camera["pos"], camera["rot"])
        ids = {p: e for e, p in self._script["passes"].items()}
        passes, inverse = np.unique(index[rows, cols], return_inverse=True)
        names = np.array([ids.get(int(p), "") for p in passes], dtype=str) if len(passes) else np.array([], dtype=str)
        return Frame(pts, names[inverse] if len(passes) else np.array([], dtype=str))

    def points(self, k: int, entity: str) -> np.ndarray:
        return self._thing(k, entity)[0]

    def gaps(self, k: int, entity: str) -> int:
        return self._thing(k, entity)[1]
