"""`drone/1`'s constants, its plan, its gaps and its zone coin: what a script of flights is before any world exists.

Split from `job_drone.py` (which owns the job and its script) so the draws stay readable by themselves: `plan_flights`,
`gaps` and `coin_blocked` are the map's, the seed's and the job's alone, never the world's, so each is testable without
a world and no world can move a flight's kind, its drop zone, its time or its coin. A flight is a `deliver` (the
package is lowered and released) or its swap partner, an `inspect` (the same flight, the lowering not made); its drop
zone is one of the map's `drop_zone` pieces, uniform, in map order.
"""

from __future__ import annotations

from functools import lru_cache

from . import job_gaps
from .job_drone_zone import settable
from .mapspec import Furniture, MapSpec
from .seeding import make_rng
from .timeline_ego import TaskPlan

JOB_ID = "drone/1"
DELIVER, INSPECT = "deliver", "inspect"
NEST_KIND, ZONE_KIND = "nest", "drop_zone"  # the map's pieces the job reads, by kind only
INSPECT_P = 0.25  # a flight is an inspection with this probability: the swap partner, the same flight
ZONE_COIN_P = 0.5  # a flight's drop zone is blocked with this probability, a coin that reads no world
ALT_M = 15.0  # transit and inspection height above the map's floor (Wing's approach, design §4)
LOWER_M = 7.0  # lowering height above the drop zone's ground (Wing's, design §4)
LEG_M = 0.25  # the landing gear: on the pad the body's underside (the base) stands this high
PITCH_DEG = {"preflight": -20.0, "transit": -45.0, "down": -90.0}  # the gimbal (limits -90 to -20)
GIMBAL_DPS = 60.0  # modelling choice: the gimbal's turn rate
FLIGHT_GRID_M = 0.5  # the flight's own free grid, never `robot_free_space`'s 0.025 m ground grid
PREFLIGHT_S, INSPECT_S, RELEASE_S = 6.0, (6.0, 12.0), 2.0
DRAIN_PCT_PER_MIN = 2.5  # modelling choice: a delivery drone's tens of minutes of flight
TASK_MAX_S = 600.0  # the most a flight may take (a check); the span budget counts it for every flight
BEFORE_DESCENT, BEFORE_LOWER, BEFORE_RETURN = "before_descent", "before_lower", "before_return"
ABORT, GO = "abort", "go"  # the response at `before_lower`: a delivery over a zone that holds an item aborts, else go
SPEED_PHASES = ("climb", "approach", "descend", "lower")  # a flight's speeds, drawn once in this order


def nest_of(m: MapSpec) -> Furniture:
    """The map's one `nest` piece (the job's anchor)."""
    nests = [f for f in m.furniture if f.kind == NEST_KIND]
    if len(nests) != 1:
        raise ValueError(f"{m.name} must hold exactly one {NEST_KIND!r} piece for {JOB_ID}, not {len(nests)}")
    return nests[0]


def drop_zones(m: MapSpec) -> list[Furniture]:
    """The map's `drop_zone` pieces, in map order; ValueError when it has none."""
    zones = [f for f in m.furniture if f.kind == ZONE_KIND]
    if not zones:
        raise ValueError(f"{m.name} holds no {ZONE_KIND!r} piece for {JOB_ID}")
    return zones


@lru_cache(maxsize=16)
def _settable_ids(m: MapSpec) -> tuple[str, ...]:
    zones = drop_zones(m)
    return tuple(z.id for z in zones if settable(m, z, zones))


def settable_zones(m: MapSpec) -> list[Furniture]:
    """The drop zones a plan draws from, in map order: those the world can always make blocked or clear
    (`job_drone_zone.settable`, from the map and its catalog alone). ValueError when none is."""
    found = [z for z in drop_zones(m) if z.id in _settable_ids(m)]
    if not found:
        raise ValueError(f"no {ZONE_KIND!r} piece of {m.name} is one the world can make blocked and clear")
    return found


def region_kind_at(m: MapSpec, piece: Furniture) -> str:
    """The kind word of the region a piece stands in."""
    region = m.region_of(piece.center)
    if region is None:
        raise ValueError(f"the {piece.kind} {piece.id} of {m.name} stands in no region")
    return m.region(region).kind


def plan_flights(m: MapSpec, seed: int, n_flights: int) -> list[TaskPlan]:
    """The script's assignments, from `make_rng(m.name, seed, JOB_ID, "job")` and the map alone: each flight is an
    `inspect` with `INSPECT_P`, else a `deliver`, to a drop zone uniform among the map's. Every flight draws the same
    two numbers in the same order whatever they decide. `params` hold `address` (the zone's name, the only key a brief
    reads), `zone` (its piece id), the zone's centre as `zone_x` and `zone_y` (the point `Relevance.near` is measured
    from) and the region kinds of the zone and of the nest (`relevance`): the job's, never delivered. ValueError for no
    flight."""
    if n_flights < 1:
        raise ValueError(f"a script has at least one flight, not {n_flights}")
    zones, nest = settable_zones(m), nest_of(m)
    rng = make_rng(m.name, seed, JOB_ID, "job")
    plans = []
    for _ in range(n_flights):
        inspect, pick = float(rng.random()) < INSPECT_P, int(rng.integers(len(zones)))
        zone = zones[pick]
        plans.append(TaskPlan(INSPECT if inspect else DELIVER,
                              {"address": zone.name, "zone": zone.id,
                               "zone_x": zone.center[0], "zone_y": zone.center[1],
                               "region": region_kind_at(m, zone), "nest_region": region_kind_at(m, nest)}))
    return plans


def coin_blocked(map_name: str, seed: int, i: int) -> bool:
    """Whether flight `i`'s drop zone is blocked: a seeded coin of `ZONE_COIN_P` that reads no world, so no form, key or
    pool can predict it. The world is made to say it (`job_drone_zone.enforce`)."""
    return float(make_rng(map_name, seed, JOB_ID, "zone", i).random()) < ZONE_COIN_P


def gaps(map_name: str, seed: int, n_flights: int) -> list[float]:
    """The n - 1 unobserved gaps between a script's flights, in seconds: `job_gaps.gaps` (the law every job shares) with
    this job's id and longest flight. It adds no validation: the job's checks hold the range (`job_gap`)."""
    return job_gaps.gaps(map_name, seed, JOB_ID, n_flights, TASK_MAX_S)
