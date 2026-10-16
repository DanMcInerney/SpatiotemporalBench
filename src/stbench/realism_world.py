"""The world a probe measures: the script's own, rebuilt, and the standing camera's script over it (`stbench realism
probe`).

`rebuild_world` generates the script again from the script's own map, seed, robot, job and number of tasks and refuses
a script whose world hash is not the hash of what the generator makes, both taken without the people overlay
(`people`): a script whose world someone edited, or that the generator no longer makes, is not what the probe's pictures
would show. It returns the start world, which the generator's own steps make (`generate.build_world`, then
`clutter.place`, then `generate.light_starts` with the job's anchors, each on the stream `generate_script` gives it): no
resident, and none of the changes a longer script's gaps make. `standing_timeline` puts the home helper's camera
(1.30 m up, its default pitch, 80 degrees across) at seeded poses on the helper's free floor, `per_room` in each
room, in a timeline of that world, as one task of keyframes one second apart; it is a script like any other
(`episode.write_script`), so `stbench render` draws it. The rooms alternate in the order of the poses, so the first
few poses are in different rooms (the sheet takes the first ones at release settings)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from . import SEED_KEY, clutter, job_registry
from .changes import World
from .generate import build_world, generate_script, light_starts, to_timeline
from .geometry import points_in_polygon, q
from .mapspec import MapSpec
from .people import without_people
from .robots import ROBOTS, camera_pose
from .route import robot_free_space
from .seeding import make_rng
from .timeline import Ego, Keyframe, Task, TaskPlan, Timeline

STANDING_ROBOT = "helper"
STANDING_JOB = "standing/1"  # no job: the poses are the point, so a name that is no job's


def rebuild_world(tl: Timeline, m: MapSpec) -> World:
    """The start world of the script's seed as the generator makes it (no resident, no gap's change). Raises ValueError
    if the script's world, taken without the people overlay, differs from the generated script's."""
    ep = tl.episode
    again = generate_script(ep.map, ep.seed, ep.robot, ep.job, len(tl.ego.tasks))
    if without_people(again).world_hash != without_people(tl).world_hash:
        raise ValueError("the script's world is not the one this generator makes of its seed "
                         f"({ep.map} seed {ep.seed}): generate the script again")
    w = build_world(m, ep.seed)
    clutter.place(w, m, make_rng(m.name, ep.seed, SEED_KEY, "clutter"))
    light_starts(w, m, make_rng(m.name, ep.seed, SEED_KEY, "light_start"), job_registry.get(ep.job).anchors)
    return w


def standing_timeline(tl: Timeline, m: MapSpec, world: World, per_room: int, rng: np.random.Generator
                      ) -> tuple[Timeline, list[str]]:
    """The standing camera's timeline over `world` and the room of each frame. Poses are drawn from the free cells of
    the helper (`route.robot_free_space`, 2.5 cm cells) inside each room, a uniform heading each."""
    helper = ROBOTS[STANDING_ROBOT]
    free = robot_free_space(m, world, helper)
    cx, cy = free.cells_xy(free.free)
    per: list[list[tuple[float, float]]] = []
    for r in m.regions:
        inside = points_in_polygon(cx, cy, [list(p) for p in r.polygon])
        spots = list(zip(cx[inside].tolist(), cy[inside].tolist(), strict=True))
        picks = rng.choice(len(spots), size=min(per_room, len(spots)), replace=False) if spots else []
        per.append([spots[i] for i in picks])
    start = {k: v[1] if isinstance(v, tuple) else v for k, v in helper.status.start.items() if k != "phase"}
    pitch = q(math.radians(helper.camera.pitch_deg))
    kfs, rooms = [], []
    for turn in range(per_room):  # a pose from each room in turn
        for r, spots in zip(m.regions, per, strict=True):
            if turn < len(spots):
                x, y = spots[turn]
                yaw = q(float(rng.uniform(-math.pi, math.pi)))
                base = (q(x), q(y), 0.0)
                pos, rot = camera_pose(helper, base, yaw, pitch)
                kfs.append(Keyframe(len(kfs), float(len(kfs)), pos, rot, base, yaw, pitch, 0,
                                    helper.status.start["phase"], dict(start)))
                rooms.append(r.name)
    info = replace(tl.episode, robot=STANDING_ROBOT, job=STANDING_JOB)
    task = Task(0, 0, len(kfs), TaskPlan("standing", {}), ())
    return to_timeline(m, world, Ego(kfs, [task], []), info), rooms
