"""The free-space rule applied to a script's own files: what each keyframe's depth image says of the floor.

`script_states` reads the script's depth PNGs (private: `private/<sid>/depth/`, the same bytes the `detections` channel
sends), its `camera.json` and `robot.json` and the pose rows' camera block and base height, and calls `grid`, the one
rule the baselines' mapper calls on the same bytes, so the gold and the best system agree to the bit. Nothing is
written: the states are a pure function of what a system is sent, so a release can recompute them, and the compile
asks only for the keyframes it needs.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable

from . import grid
from .camera import Camera
from .case import ScriptPaths
from .render_encode import read_png16
from .robots import from_doc
from .visibility import load_pose

Cells = dict[tuple[int, int], str]


def script_states(paths: ScriptPaths, ks: Iterable[int], floor_z: float | None = None,
                  band: tuple[float, float] | None = None, keep: Collection[tuple[int, int]] | None = None
                  ) -> dict[int, Cells]:
    """`grid.observe` over each keyframe `k` of the script (script numbering): its depth image unprojected from the
    pose row's camera block, with `floor_z` as the floor (None: the pose row's base height) and `band` as the band
    (None: the robot's, `grid.band`). A zone passes its own ground and band (`zones.Zone`), and `keep` the fine cells it
    reads: only those cells are returned (None: all), so a drone frame's tens of thousands of observed cells never sit
    in a cache. KeyError for a `k` the pose file lacks, FileNotFoundError for a depth image that is not there."""
    cam = Camera.load(paths.camera)
    band = grid.band(from_doc(json.loads(paths.robot.read_text(encoding="utf-8")))) if band is None else band
    rows = {row["k"]: row for row in load_pose(paths.pose)}
    out: dict[int, Cells] = {}
    for k in ks:
        row = rows[k]
        pts = grid.points(read_png16(paths.depth(k)), cam, row["camera"]["pos"], row["camera"]["rot"])
        states = grid.observe(pts, row["pose"]["pos"][2] if floor_z is None else floor_z, band)
        out[k] = states if keep is None else {c: s for c, s in states.items() if c in keep}
    return out
