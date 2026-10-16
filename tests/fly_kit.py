"""What the tests of the flier's `clear_path` share (0.8.0 lane L2): a place with two walls, a drone's script over it
and the compiler's inputs for it, with the depth given as points or as rendered images.

The place is `downtown_a` with its walls replaced by two, both 0.6 m thick, 20 m long (x 20 and x -15; `WE` and
`WW`), and no opening. The wall `WE` runs from (20, -10) to (20, 10), so its box is x 19.7..20.3, y -10.3..10.3,
and the bin of a point on it is `floor(y + 10)`; `WW` runs from (-15, -10) to (-15, 10), box x -15.3..-14.7, bin
`floor(y + 10)` too. A drone at (0, 0) heading +x has its body's edge (radius 0.65 m) at x 0.65, so its corridor
ahead meets `WE`'s near face (x 19.7) after 19.05 m; heading -x its edge is at x -0.65 and `WW`'s near face (x
-14.7) is 14.05 m on.

`flight` is a drone's script of `tasks` tasks of `frames` keyframes, the base at (0, 0) at `base_z`, every heading
0 unless `yaws` says (radians), in the phase `approach` after the first step. `JOB` asks `clear_path` in
`approach` from 8.35 m up. `fly_inputs` hands the compiler those keyframes' depth as `points` (keyframe -> (N, 3)
map-frame points; none: no point) and `rendered_points` ray-casts the walls as boxes through the drone's own
camera, the depth a mapper is sent.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from space_kit import JOB as BASE_JOB
from space_kit import SpaceScript, box, inputs, record, render_depth
from stbench import grid, robots
from stbench.camera import Camera
from stbench.mapspec import MapSpec, Wall, load_map
from stbench.questions_world import Inputs, World
from synth import DOWNTOWN

WE = Wall("WE", (20.0, -10.0), (20.0, 10.0), 0.6, 30.0, True)
WW = Wall("WW", (-15.0, -10.0), (-15.0, 10.0), 0.6, 30.0, True)
MAP: MapSpec = replace(load_map("downtown_a"), walls=(WE, WW), openings=())
BOXES = [box(19.7, 20.3, -10.3, 10.3, 0.0, 30.0), box(-15.3, -14.7, -10.3, 10.3, 0.0, 30.0)]
JOB = replace(BASE_JOB, id="drone/1", robot="drone", clearance=2.4, looks=("clear_path",), decisions={},
              clear_phases=frozenset({"approach"}), fly_floor_m=8.35,
              clear_directions=("ahead", "back"))
EMPTY = np.empty((0, 3))


def flight(tasks: int = 2, frames: int = 10, base_z: float = 12.0, yaws: dict[int, float] | None = None,
           poses: dict[int, tuple[float, float]] | None = None, **fields) -> SpaceScript:
    """`tasks` tasks of `frames` keyframes each over the high street, the base at (0, 0, `base_z`) heading +x; `yaws`
    (radians) and `poses` ((x, y)) move chosen keyframes."""
    n = tasks * frames
    return SpaceScript(tasks=[[("high street", frames, "approach")] for _ in range(tasks)],
                       stage=replace(DOWNTOWN, base_z=base_z),
                       poses={**dict.fromkeys(range(n), (0.0, 0.0)), **(poses or {})},
                       yaws={**dict.fromkeys(range(n), 0.0), **(yaws or {})}, depth_ks=set(), **fields)


def face(x: float, ys, z: float = 5.0) -> np.ndarray:
    """Depth points on a wall's face at x, at the heights z, one for each y."""
    return np.array([(x, y, z) for y in ys], dtype=float)


def fly_inputs(script: SpaceScript, points: dict[int, np.ndarray] | None = None, job=JOB, m: MapSpec | None = MAP,
               counter: list[int] | None = None) -> Inputs:
    """The compiler's inputs for the script over the two-wall place, with `depth_points_of` returning the given points
    (appending the keyframe to `counter` when given, to count the images read)."""
    points = points or {}

    def depth_points_of(k: int) -> np.ndarray:
        if counter is not None:
            counter.append(k)
        return points.get(k, EMPTY)

    return replace(inputs(script, job), map=m, depth_points_of=depth_points_of)


def fly_world(script: SpaceScript, points: dict[int, np.ndarray] | None = None, a: int = 0, b: int | None = None,
              job=JOB, cid: str = "c0") -> World:
    """The window over the script's tasks a..b (the first by default) of `fly_inputs`."""
    return World(fly_inputs(script, points, job), record(script, a=a, b=b, cid=cid))


def drone_camera() -> Camera:
    return Camera.load(robots.camera_doc(robots.ROBOTS["drone"]))


def rendered_depth(script: SpaceScript, k: int) -> np.ndarray:
    """Keyframe `k`'s depth image: the two walls as boxes (the ground at z 0), through the drone's own camera."""
    pos, rot = script.cameras()[k]
    return render_depth(drone_camera(), pos, rot, BOXES, 0.0)


def rendered_points(script: SpaceScript, k: int) -> np.ndarray:
    """The map-frame points of keyframe `k`'s rendered depth, as `grid.points` unprojects them."""
    pos, rot = script.cameras()[k]
    return grid.points(rendered_depth(script, k), drone_camera(), pos, rot)
