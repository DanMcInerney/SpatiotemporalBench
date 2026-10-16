"""What `drone/1`'s tests share: the downtown, its scripts cached per worker process, and the reading of a drop zone's
state from a timeline.

The numbers are `downtown_a`'s (`maps/downtown_a.json`): the nest `F25` at (40.0, -18.5), 1.6 x 1.6 x 0.02, so a drone
on it stands with its base 0.02 + 0.25 m up; the drop zones `F26` "Square West" (40, -6), `F27` "Square East" (60, -6)
and `F28` "Square South" (60, -18), 3 x 3 on the square's 0.15 m pads, `F29` "West Street" (20.3, 44.1) and `F30`
"East Street" (79.7, 44.1), 2.6 x 3 at the south end of the 14.8 m loading zones (0.01 m high: a flight lowers at
7.01 m over them, 7.15 m over a pad). A zone as the families read it is its footprint less
0.05 m on every side. A script's drop zone is blocked when its coin says so, `make_rng(map, seed, "drone/1", "zone",
i)` read below 0.5: written here from its definition, not taken from the code under test.
"""

from __future__ import annotations

from functools import cache
from unittest import mock

from stbench import build, job_drone
from stbench.generate import generate_script
from stbench.geometry import Rect, rects_overlap
from stbench.mapspec import MapSpec, load_map
from stbench.seeding import make_rng
from stbench.timeline import Timeline
from stbench.zones import ground_z

MAP, ROBOT, JOB = "downtown_a", "drone", "drone/1"
TASKS = (4, 6)  # a script's flights, drawn from the seed in this range
NEST = (40.0, -18.5)
PAD_Z = 0.27  # the nest's 0.02 m and the landing gear's 0.25 m
ZONES = ("F26", "F27", "F28", "F29", "F30")  # the zones a plan draws from on the shipped map
EDGE_M = 0.05
STAND_M = 0.3


@cache
def downtown() -> MapSpec:
    return load_map(MAP)


@cache
def generated(seed: int) -> Timeline:
    """The seed's `drone/1` script on `downtown_a`: 4-6 flights, the real world and processes (no pedestrians until
    M3-people merges). Cached per worker process; a test mutates nothing of it."""
    return generate_script(MAP, seed, ROBOT, JOB, build.tasks_for(TASKS, MAP, seed, ROBOT, JOB))


@cache
def forced(seed: int, blocked: bool) -> Timeline:
    """The seed's script with every flight's coin forced (`job_drone.coin_blocked` patched to say `blocked` for each),
    the flights, the plan and the seed otherwise the same as `generated(seed)`'s. Cached per worker process."""
    n = len(generated(seed).ego.tasks)
    with mock.patch.object(job_drone, "coin_blocked", lambda map_name, s, i: blocked):
        return generate_script(MAP, seed, ROBOT, JOB, n)


def coin_says_blocked(seed: int, i: int) -> bool:
    """Flight `i`'s coin, from its definition."""
    return float(make_rng(MAP, seed, JOB, "zone", i).random()) < 0.5


def zone_rect(m: MapSpec, zone_id: str) -> Rect:
    """The zone `zone_id` as the families read it: its piece's footprint less 0.05 m on every side."""
    f = m.furniture_by_id()[zone_id]
    return Rect(f.center[0], f.center[1], f.size[0] - 2 * EDGE_M, f.size[1] - 2 * EDGE_M, f.yaw)


def blocked_at_start(tl: Timeline, task) -> bool:
    """Whether the flight's drop zone holds a standing item at its first keyframe (`zone_state`'s "blocked")."""
    piece = downtown().furniture_by_id()[task.plan.params["zone"]]
    return zone_state(tl, downtown(), piece.id, ground_z(downtown(), piece), tl.keyframe(task.k0).t) == "blocked"


def zone_state(tl: Timeline, m: MapSpec, zone_id: str, ground: float, t: float) -> str:
    """The zone's state at time `t` from the timeline's placements alone: "blocked" when some carried thing taller than
    0.3 m above the zone's ground has a footprint that overlaps it, "clear" when none comes within 0.05 m of it, else
    "near" (a thing beside it)."""
    zone, state = zone_rect(m, zone_id), "clear"
    for e in tl.entities:
        p = tl.placement_of(e.id, t) if e.mobility == "carried" else None
        if p is None or p.pos[2] + e.size[2] - ground <= STAND_M:
            continue
        foot = Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)
        if rects_overlap(foot, zone):
            return "blocked"
        if rects_overlap(foot, zone, EDGE_M):
            state = "near"
    return state
