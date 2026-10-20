"""A small hand-built timeline on the apartment map, consistent by construction, for validation and query tests.

The robot is the home helper (a 0.34 x 0.33 m body, its camera 1.30 m up at pitch -15 degrees, the nearest to the 1.5 m
standing camera the body-free timeline had). Coordinates are taken from the map by hand: the dining table (F07) stands
at (2.1, 1.9) facing north, 0.8 deep and 1.4 wide, 0.75 high; the kitchen door (D1) sits in the wall x = 4.6 at y = 1.9
and, open, its leaf runs from (4.6, 1.475) west to (3.75, 1.475); (3.8, 1.0) and (3.4, 1.0) are free kitchen floor, the
stove behind them (top edge y = 0.65) and the open leaf in front (y = 1.455 to 1.495).
`keyframe` builds one keyframe of any robot from its base pose, the camera derived as `robots.camera_pose` says.
"""

from __future__ import annotations

import math
from dataclasses import replace

from stbench.geometry import q, quat_z
from stbench.robots import ROBOTS, camera_pose
from stbench.timeline import Ego, Entity, EpisodeInfo, Keyframe, Placement, Region, State, Task, TaskPlan, Timeline

TABLE, CHAIR, MUG, MUG2, DOOR = "E0001", "E0002", "E0003", "E0004", "E0005"
TASK2_T = 7200.0


def start_status(robot: str) -> dict[str, object]:
    """The robot's status keys at a task's first step, each range at its upper bound (not the phase)."""
    start = ROBOTS[robot].status.start
    return {key: v[1] if isinstance(v, tuple) else v for key, v in start.items() if key != "phase"}


def hand_job(robot: str) -> str:
    """The job a hand-built script of `robot` is stamped with: the vacuum's (its tests read `vacuum/1`'s rules) or,
    for another robot, an unregistered hand job, so `validate` applies that robot's generic rules only and no written
    job's checks or subgoals refuse a hand plan (the helper's name a thing and two pieces, the drone's an address)."""
    return f"{robot}/1" if robot == "vacuum" else f"{robot}/hand"


def keyframe(robot: str, k: int, t: float, base: tuple[float, float, float], yaw: float, *, task: int, phase: str,
             status: dict[str, object] | None = None, pitch: float | None = None) -> Keyframe:
    """A keyframe of `robot` with its base at `base` heading `yaw` (radians) and the camera at `pitch` (radians, the
    robot's default when None). Yaw and pitch are quantised first, as the file keeps them, so the camera derived from
    them is the camera the file reloads to."""
    r = ROBOTS[robot]
    base = tuple(q(v) for v in base)
    yaw = q(yaw)
    pitch = q(math.radians(r.camera.pitch_deg) if pitch is None else pitch)
    pos, rot = camera_pose(r, base, yaw, pitch)
    status = start_status(robot) if status is None else status
    return Keyframe(k, t, pos, rot, base, yaw, pitch, task, phase, dict(status))


def camera_keyframe(k: int, t: float, xy: tuple[float, float], yaw: float, *, robot: str = "helper", z: float = 1.5,
                    pitch: float = math.radians(-15.0), task: int = 0, phase: str | None = None) -> Keyframe:
    """A keyframe of `robot` whose CAMERA stands at `xy` at height `z` facing `yaw` at `pitch`, for the render tests,
    which check what a camera sees and not where a robot could drive: the base is put where the mount puts the camera
    there (the mount's offset turned by the heading, and its height taken off `z`)."""
    r = ROBOTS[robot]
    mx, my, mz = r.camera.mount_m
    c, s = math.cos(yaw), math.sin(yaw)
    base = (xy[0] - (c * mx - s * my), xy[1] - (s * mx + c * my), z - mz)
    return keyframe(robot, k, t, base, yaw, task=task, phase=r.status.start["phase"] if phase is None else phase,
                    pitch=pitch)


def minimal_timeline() -> Timeline:
    """Two tasks of three keyframes in the kitchen; a table, a chair, two mugs on the table, and an open door."""
    entities = [
        Entity(TABLE, "dining_table", {"name": "dining table"}, "fixed", (0.8, 1.4, 0.75)),
        Entity(CHAIR, "chair", {"color": "oak"}, "light", (0.45, 0.45, 0.9)),
        Entity(MUG, "mug", {"color": "red"}, "carried", (0.1, 0.1, 0.1)),
        Entity(MUG2, "mug", {"color": "blue"}, "carried", (0.1, 0.1, 0.1)),
        Entity(DOOR, "door", {"between": ["hall", "kitchen"], "hinge": "a", "swing": "left"}, "fixed",
               (0.85, 0.04, 2.05)),
    ]
    placements = [
        Placement(TABLE, 0.0, None, (2.1, 1.9, 0.0), quat_z(math.pi / 2), None),
        Placement(CHAIR, 0.0, None, (1.155, 1.9, 0.0), quat_z(0.0), None),
        Placement(MUG, 0.0, 3600.0, (2.1, 1.9, 0.75), quat_z(0.3), TABLE),
        Placement(MUG, 3600.0, None, (2.5, 1.7, 0.75), quat_z(0.3), TABLE),
        Placement(MUG2, 0.0, None, (2.4, 2.1, 0.75), quat_z(1.0), TABLE),
        Placement(DOOR, 0.0, None, (4.6, 1.9, 0.0), quat_z(math.pi / 2), None),
    ]
    states = [State(DOOR, "open", 0.0, None, True)]
    frames = [(0.0, (3.8, 1.0), 0.0, "waiting"), (1.5, (3.8, 1.0), 0.5, "navigating"),
              (3.0, (3.4, 1.0), 0.5, "navigating"), (TASK2_T, (3.8, 1.0), 1.0, "waiting"),
              (TASK2_T + 1.5, (3.8, 1.0), 1.5, "navigating"), (TASK2_T + 3.0, (3.4, 1.0), 1.5, "navigating")]
    keyframes = [keyframe("helper", k, t, (*xy, 0.0), yaw, task=k // 3, phase=phase)
                 for k, (t, xy, yaw, phase) in enumerate(frames)]
    plan = TaskPlan("fetch", {})
    return Timeline(
        episode=EpisodeInfo("apartment_a", 1, "helper", "helper/0", "2026-03-02T08:00:00"),
        regions=[Region("R3", "kitchen", "kitchen", ((0.05, 0.05), (4.55, 0.05), (4.55, 2.75), (0.05, 2.75)), 0, 0.0,
                        2.5)],
        entities=entities,
        placements=placements,
        states=states,
        ego=Ego(keyframes, [Task(0, 0, 3, plan, ()), Task(1, 3, 6, plan, ())], []),
    )


def with_placements(tl: Timeline, placements: list[Placement]) -> Timeline:
    return replace(tl, placements=placements)
