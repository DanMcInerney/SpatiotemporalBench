"""What the zone tests share, on top of `family_kit` and `space_kit`: a hand map, a job that declares its zones, a
script of the tote robot to put them to, and two ways to give the compile a zone's states.

**The map** (metres; every marking at most 0.005 m, as `warehouse_a`'s, so a ground robot drives over it):
- two bays, B1 `F01` at (1.5, 1.5) and B2 `F02` at (1.5, 4.5) (3 m apart; B2 turned by pi), each 1.6 x 2.2 m, so
  B1 is x 0.7..2.3, y 0.4..2.6 and its zone x 0.75..2.25, y 0.45..2.55: 16 x 22 = 352 fine cells (a centre on an edge
  is in); a shelving station would stand against its west face, x 0.2..0.7;
- a 0.15 m plaza `F03` (x 1..5, y 6.5..10.5) holding a 0.155 m drop zone `F04` at (3.0, 8.5), 2.4 m square: zone x
  1.85..4.15, y 7.35..9.65, 24 x 24 = 576 cells, on the plaza's ground 0.15;
- four staging lanes `F05`..`F08` (1.4 x 2.8 m, turned by pi/2) at x 10, 12.5, 15 and 17.5, y 2: zones 1.3 x 2.7 m.
Each piece is a fixed timeline entity `P01`..`P08` in the script, as the generator makes them.

**The job** is the tote robot's: `before_station` asks `zone_clear` of the bay ahead, the events "when you reached the
station" (the first `at_station` step) and "when you finished at the station" (the last) name it for `clearance_then`,
and `slots_empty` lists the staging lanes. **The script** is `script(n)`: n tasks of 40 keyframes, the base at
(3.9, 1.5) facing west (the bay B1 ahead, B2 3 m off) throughout: waiting 0, transit 1-7, approach 8-17,
at_station 18-25, undocking 26-33, waiting 34-39; the decision at 17. The stream is cut after 17 and 39: look windows
0-7, 8-15, 16-17, 18-25, 26-33 and 34-39 (the last is the end look window).

**The states**: `Views` hands the compile hand-set cells (`paint`): `zone[k]` is what every zone band sees at keyframe
k, `robot[k]` what the robot's own band sees (a family that read it would be wrong), and `calls` records each request's
(floor, band). `seen` ray-casts a zone from the tote robot's camera (millimetre depth, `space_kit.render_depth`) and
observes it with the zone's own ground and band; the AMR's camera is 0.5 m up pitched 10 degrees down, so it sees
the floor from 0.55 m to 4 m ahead of the camera: from (3.9, 1.5) the whole of B1, and from (3.0, 5.5) facing north the
whole of the drop zone.

**The apartment** (`apartment_script`, `WAY_JOB`, `SPOT_JOB`) puts the same script on `apartment_a`, for the zones that
are no piece's footprint: the way of an interior opening (`WAY`, 0.02 m up, as a vacuum's) and the window of the
surface the base faces (`SPOT`, 0.01 to 0.5 m above it). Every piece and door of the map is a fixed entity `E<id>`
(`EF07`, the dining table, `ED1`, the door D1; the arch O1 has none), as the generator makes them. Hand values the
tests lean on: from (3.9, 1.1) facing north-east (pi/4) the nearest way is D1's (x 4.6..4.95, y 1.575..2.225: 24
cells) and the tote camera sees all of it; from (2.1, 0.9) facing north the window is the dining table's (x 1.9..2.3,
y 1.56..2.24 at 0.75 m: 24 cells), facing south the sink unit F03's (0.9 m), facing east there is none.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cache

import family_kit as fk
from builders import keyframe
from shape_kit import APARTMENT
from space_kit import JOB as SPACE_JOB
from space_kit import SpaceScript, render_depth
from space_kit import record as space_record
from stbench import grid, zones
from stbench.case import CaseRecord
from stbench.events import EventRule
from stbench.job_registry import JobSpec
from stbench.mapspec import Furniture, MapSpec
from stbench.questions_world import Inputs, World, cached_states
from stbench.zones import Zone, ZoneRule
from synth import WAREHOUSE, camera

BAY = ZoneRule("the bay ahead of you", "bay")
DROP = ZoneRule("the drop zone", "drop_zone")
LANES = ZoneRule("the staging lanes", "staging_lane", piece_words="staging lanes")
REACHED, FINISHED = "when you reached the station", "when you finished at the station"
EVENTS = {REACHED: EventRule("first", "at_station"), FINISHED: EventRule("last", "at_station")}
JOB: JobSpec = replace(SPACE_JOB, id="amr/1", robot="amr", domain="warehouse",
                       decisions={"before_station": ("zone_clear",), "before_crossing": ()}, events=EVENTS,
                       zones={"before_station": BAY, REACHED: BAY, FINISHED: BAY}, slot_zones=(LANES,),
                       ends=("clearance_then", "slots_empty"))


def piece(pid: str, name: str, kind: str, centre, size, yaw: float = 0.0, blocks: bool = False) -> Furniture:
    return Furniture(pid, name, kind, "fixed", centre, size, yaw, "yellow", blocks, (), 0.0)


PIECES = (
    piece("F01", "bay", "bay", (1.5, 1.5), (1.6, 2.2, 0.005)),
    piece("F02", "bay", "bay", (1.5, 4.5), (1.6, 2.2, 0.005), math.pi),
    piece("F03", "plaza", "plaza", (3.0, 8.5), (4.0, 4.0, 0.15), blocks=True),
    piece("F04", "Square West", "drop_zone", (3.0, 8.5), (2.4, 2.4, 0.155)),
    *(piece(f"F0{5 + i}", "staging lane", "staging_lane", (10.0 + 2.5 * i, 2.0), (2.8, 1.4, 0.005), math.pi / 2,
            blocks=True) for i in range(4)))
MAP = MapSpec("warehouse_a", (0.0, 0.0, 20.0, 11.0), 0.1, 3.0, (), (), (), (), PIECES, ())
B1, B2 = zones.zones_of(MAP, BAY)
(DROP_ZONE,) = zones.zones_of(MAP, DROP)
L1, L2, L3, L4 = zones.zones_of(MAP, LANES)
LANE_M = 1.125  # slots_empty's match radius on these lanes: the least of 2.7 / 2 and 0.45 x the 2.5 m between two

N = 40  # keyframes a task
BASE, FACING = (3.9, 1.5), math.pi
LEGS = [("dock", 1), ("dock", 7, "transit"), ("dock", 10, "approach"), ("dock", 8, "at_station"),
        ("dock", 8, "undocking"), ("dock", 6, "waiting")]
DECISION_K, REACHED_K, FINISHED_K = 17, 18, 25  # in task 0; task t's are 40 t later
END_WINDOW = range(34, 40)  # task 0's last look window


def script(n_tasks: int = 1, **fields) -> SpaceScript:
    """`n_tasks` tasks of the tote robot (module docstring), the eight pieces as fixed entities, no depth."""
    s = SpaceScript(tasks=[list(LEGS) for _ in range(n_tasks)], stage=WAREHOUSE,
                    decisions=[(t * N + DECISION_K, "before_station") for t in range(n_tasks)],
                    poses=dict.fromkeys(range(n_tasks * N), BASE), yaws=dict.fromkeys(range(n_tasks * N), FACING),
                    depth_ks=set(), **fields)
    for i, f in enumerate(PIECES, 1):
        s.item(f"P{i:02d}", f.kind, {"name": f.name}, [(0, None, (*f.center, 0.0))], mobility="fixed", size=f.size)
    return s


# -- hand states ------------------------------------------------------------------------------------------------
def paint(zone: Zone, state: str = "free", n: int | None = None, start: int = 0) -> dict[tuple[int, int], str]:
    """The zone's cells `start`..`start + n` (all from `start` when n is None; in cell order, u then v) in `state`."""
    cells = zones.zone_cells(zone)
    return dict.fromkeys(cells[start:] if n is None else cells[start:start + n], state)


class Views:
    """The free-space cells a test hands the compile: `zone[k]` is what a zone band sees at keyframe k (every band
    the same), `robot[k]` what the robot's own band sees; a keyframe not in them saw nothing. `calls` lists the
    (floor, band) of every request, the robot's as (None, None)."""

    def __init__(self, zone: dict[int, dict] | None = None, robot: dict[int, dict] | None = None):
        self.zone, self.robot, self.calls = zone or {}, robot or {}, []

    def __call__(self, ks, floor_z=None, band=None):
        self.calls.append((floor_z, band))
        source = self.robot if floor_z is None and band is None else self.zone
        return {k: source.get(k, {}) for k in ks}


def inputs(s: SpaceScript, views: Views, job: JobSpec = JOB, m: MapSpec = MAP) -> Inputs:
    """The compiler's inputs for the script over the map `m`, its states those of `views`."""
    return replace(fk.inputs(s, job), states_of=cached_states(views), map=m)


def record(s: SpaceScript, a: int = 0, b: int | None = None, cid: str = "c0", length: str = "short") -> CaseRecord:
    """The case over tasks a..b of the script (b defaults to a)."""
    return space_record(s, a=a, b=b, cid=cid, length=length)


def world(s: SpaceScript, views: Views, a: int = 0, b: int | None = None, cid: str = "c0", length: str = "short",
          job: JobSpec = JOB, m: MapSpec = MAP) -> World:
    """The window of tasks a..b of the script (b defaults to a) as the compiler's `World`, over `views`."""
    return World(inputs(s, views, job, m), record(s, a, b, cid, length))


# -- the apartment: a doorway's way and the window faced --------------------------------------------------------
WAY = ZoneRule("the doorway ahead of you", zones.DOORWAY, height_m=0.02)
SPOT = ZoneRule("the spot in front of you where you are putting it down", zones.SPOT, height_m=0.01, top_m=0.5)
WAY_JOB: JobSpec = replace(JOB, decisions={"before_room": ("zone_clear",)}, events={}, zones={"before_room": WAY},
                           slot_zones=(), ends=())
SPOT_JOB: JobSpec = replace(JOB, decisions={"before_place": ("zone_clear",)},
                            zones={"before_place": SPOT, REACHED: SPOT}, slot_zones=(), ends=("clearance_then",))


def apartment_script(kind: str, base, yaw: float) -> SpaceScript:
    """`script()` with its decision (keyframe 17) of kind `kind` at `base` heading `yaw`, and `apartment_a`'s pieces and
    doors as fixed entities `E<id>`."""
    s = script()
    s.decisions = [(DECISION_K, kind)]
    s.poses[DECISION_K], s.yaws[DECISION_K] = base, yaw
    for f in APARTMENT.furniture:
        s.item(f"E{f.id}", f.kind, {"name": f.name}, [(0, None, (*f.center, 0.0))], mobility=f.mobility, size=f.size)
    for o in APARTMENT.doors():
        s.item(f"E{o.id}", "door", {}, [(0, None, (*o.center, 0.0))], mobility="fixed", size=(o.width, 0.04, o.height))
    return s


# -- ray-cast states --------------------------------------------------------------------------------------------
CAM = camera("mm")


@cache
def _cast(base: tuple, yaw: float, boxes: tuple):
    kf = keyframe("amr", 0, 0.0, (*base, 0.0), yaw, task=0, phase="waiting")
    return kf, render_depth(CAM, kf.pos, kf.rot, list(boxes))


def look(zone: Zone, boxes=(), base=BASE, yaw: float = FACING) -> dict[tuple[int, int], str]:
    """What the tote robot's camera at `base` heading `yaw` makes of a scene of `boxes` (`space_kit.box`), observed with
    the zone's own ground and band: the frame's fine cells, as `Views` takes them."""
    kf, depth = _cast(tuple(base), yaw, tuple(boxes))
    return grid.observe(grid.points(depth, CAM, kf.pos, kf.rot), zone.base_z, zone.band)


def seen(zone: Zone, boxes=(), base=BASE, yaw: float = FACING) -> dict[tuple[int, int], tuple[str, int]]:
    """`look` as `grid.latest` gives it (the frame is keyframe 0): what `zones.zone_state` takes."""
    return grid.latest({0: look(zone, boxes, base, yaw)})

