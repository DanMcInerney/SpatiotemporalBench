"""`helper/1`'s constants, its park pose, its plan and its gaps: what a script of several tasks is before any world.

Split from `job_helper.py` (which owns the path and the declarative part) so the draws are readable by themselves:
`plan_tasks`, `gaps` and `spot_blocked` take the map and the seed alone. A task is `tidy` (carry a thing from a surface
in one room to a surface in another) or `fetch` (bring a thing from a surface in a room other than the park room to a
surface of the park room, the one nearest the park pose, and drive back to the park pose); the rooms and surfaces are
the ones the helper can reach on the map (`job_helper_reach.qualified`), never named by kind.
"""

from __future__ import annotations

import math

from . import job_gaps
from .geometry import Point, distance, q, wrap_angle
from .job_helper_reach import qualified
from .mapspec import Furniture, MapSpec
from .robots import ROBOTS
from .seeding import make_rng
from .timeline_ego import TaskPlan

JOB_ID = "helper/1"
TIDY, FETCH = "tidy", "fetch"
TIDY_P = 0.5  # a task is a tidy with this probability, else a fetch
DOCK_KIND, OUTSIDE = "dock", "outside"
PARK_FROM_DOCK_M = 0.63  # the park pose: on the dock's axis, this far in front of its front face (the apron's middle)
GRASP_S, PLACE_S = (10.0, 30.0), (8.0, 20.0)  # held still while the arm works, drawn per task
PARK_HOLD_S = 2.0  # waiting at the park pose at the start and the end of a task
DRAIN_PCT_PER_MIN = 0.5  # modelling choice: a mobile manipulator's few hours on a charge
TASK_MAX_S = 600.0  # a check; the gaps' span budget counts it for every task
DOORWAY_STANDOFF_M = 0.5  # a before_doorway keyframe: the base this far before the doorway, along the path
BEFORE_DOORWAY, BEFORE_GRASP, BEFORE_PLACE = "before_doorway", "before_grasp", "before_place"
PHASES = ("waiting", "navigating", "grasping", "carrying", "placing", "navigating", "waiting")  # a task, in order
SPOT_COIN_P = 0.5  # the place spot is occupied before a task with this probability (a coin that reads no world)
TILT_DEG = -45.0  # the camera tilts down to this to look at the surface while it grasps and while it places
TILT_DPS = 30.0  # modelling choice: a pan-tilt head's rate


def dock_of(m: MapSpec) -> Furniture:
    docks = [f for f in m.furniture if f.kind == DOCK_KIND]
    if len(docks) != 1:
        raise ValueError(f"{m.name} must hold exactly one {DOCK_KIND!r} piece for {JOB_ID}, not {len(docks)}")
    return docks[0]


def park_pose(m: MapSpec) -> tuple[Point, float]:
    """Where the helper waits at the start and the end of every task: on the axis of the map's one dock piece,
    `PARK_FROM_DOCK_M` in front of its front face, facing the centre of the map's outside door (the first opening, by
    id, one of whose sides is `outside`). On `apartment_a`: (5.40, 5.05), heading -90 degrees, inside the dock's apron.
    One pose a map, never a world's: the apron is the floor every world keeps clear."""
    dock = dock_of(m)
    doors = sorted((o for o in m.openings if OUTSIDE in o.between), key=lambda o: o.id)
    if not doors:
        raise ValueError(f"{m.name} has no opening to the outside for {JOB_ID} to face")
    reach = dock.size[0] / 2 + PARK_FROM_DOCK_M
    xy = (q(dock.center[0] + reach * math.cos(dock.yaw)), q(dock.center[1] + reach * math.sin(dock.yaw)))
    yaw = q(wrap_angle(math.atan2(doors[0].center[1] - xy[1], doors[0].center[0] - xy[0])))
    return xy, yaw


def park_room(m: MapSpec) -> str:
    room = m.region_of(park_pose(m)[0])
    if room is None:
        raise ValueError(f"the park pose of {m.name} stands in no region")
    return room


def fetch_target(m: MapSpec) -> str:
    """The park room's counted piece nearest the park pose (its centre; the first by id on a tie): where a fetch puts
    the thing down. From the map alone, no draw. On `apartment_a`: the hall console F20, 0.95 m from the park pose."""
    park = park_pose(m)[0]
    pieces = m.furniture_by_id()
    return min(qualified(m, park)[park_room(m)],
               key=lambda pid: (round(distance(pieces[pid].center, park), 6), pid))


def spot_blocked(map_name: str, seed: int, i: int) -> bool:
    """Whether the place spot of task `i` is occupied before it: a coin that reads no world, from the seed alone."""
    return bool(make_rng(map_name, seed, JOB_ID, "spot", i).random() < SPOT_COIN_P)


def plan_tasks(m: MapSpec, seed: int, n_tasks: int) -> list[TaskPlan]:
    """The script's assignments, from `make_rng(m.name, seed, JOB_ID, "job")` and the map alone. Each task is a `tidy`
    with `TIDY_P`, else a `fetch`. A tidy's source room is drawn uniformly among the qualified rooms
    (`job_helper_reach`) and its target room among the others; a fetch's source room among the qualified rooms but the
    park room. A task's piece in a room is drawn uniformly among the room's counted pieces: the one it prefers (the
    world may move the helper to another piece of the room when clutter fills the preferred one's reach, never to
    another room).
    `params`: `from_room` (the room's kind word) and, for a tidy, `to_room`: the only keys a brief reads. The job's,
    never delivered: `park_room` (its kind word), `regions` (the source and target region ids; a fetch's target is the
    park room), `source` and `target` (piece ids; a fetch's target is `fetch_target`, no draw). Every task draws the
    same numbers in the same order, so no draw depends on a task's kind."""
    if n_tasks < 1:
        raise ValueError(f"a script has at least one task, not {n_tasks}")
    home = park_room(m)
    rooms = qualified(m, park_pose(m)[0])
    order = [r.id for r in m.regions if r.id in rooms]
    away = [r for r in order if r != home]
    if home not in rooms or not away:
        raise ValueError(f"{m.name} has {len(order)} rooms the {ROBOTS['helper'].id} can work in, the park room "
                         f"{'among them' if home in rooms else 'not'}: {JOB_ID} needs the park room and another")
    rng = make_rng(m.name, seed, JOB_ID, "job")
    plans: list[TaskPlan] = []
    for _ in range(n_tasks):
        kind_u, from_u, to_u, source_u, target_u = (float(rng.random()) for _ in range(5))
        tidy = kind_u < TIDY_P
        src = (order if tidy else away)[int(from_u * len(order if tidy else away))]
        others = [r for r in order if r != src]
        dst = others[int(to_u * len(others))] if tidy else home
        params: dict[str, object] = {"from_room": m.region(src).kind, "park_room": m.region(home).kind,
                                     "regions": [src, dst], "source": rooms[src][int(source_u * len(rooms[src]))]}
        params["target"] = rooms[dst][int(target_u * len(rooms[dst]))] if tidy else fetch_target(m)
        if tidy:
            params["to_room"] = m.region(dst).kind
        plans.append(TaskPlan(TIDY if tidy else FETCH, params))
    return plans


def gaps(map_name: str, seed: int, n_tasks: int) -> list[float]:
    """The n - 1 unobserved gaps between a script's tasks, in seconds: `job_gaps.gaps` with this job's id and longest
    task, from the seed alone."""
    return job_gaps.gaps(map_name, seed, JOB_ID, n_tasks, TASK_MAX_S)
