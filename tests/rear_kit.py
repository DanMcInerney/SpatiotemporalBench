"""What the tests of `rear_clear` (`questions_space`) share, on top of `space_kit`: scripts of the tote AMR undocking
from a station, with ray-cast depth from the AMR's own camera, and a job that asks the family at the undock.

The place (map frame, metres; no map is read): a station's face stands east of the bay, so the robot docks facing +x at
`DOCK` (0.4 m of body behind its base: the rear edge `REAR_X`, 6.6), and its straight reverse corridor runs west from
the rear edge, 0.58 m wide and 5 m long, along y = `AXIS_Y`. The axis is 4.98, not 5: the 0.1 m fine cells then have
their centres 0.27 m off it (the cell [5.2, 5.3) of y), inside the body's half-width of 0.29 m, so what stands 0.31 m
off the axis lies in a cell the old rule would count (M1's door-frame error) and only the sub-cells tell it apart.

Both scripts are one task of 23 keyframes, 1.5 s apart, with a decision point `before_undock` at offset 20 (the last
`at_station` step); the stream is cut after offsets 7, 15, 20 and 22, so the decision's look window is offsets 16 to 20.
A keyframe outside `depth_ks` has no depth, so a test names the frames that look (`space_kit.SpaceScript`):
- `lane_run` is the side arrival of `M3-W.md` §3: offsets 0-5 drive north along the lane x = `LANE_X` (6.0), 1 m apart
  from y = -1 to 4; 6 reaches `T` (6.0, 4.98); 7 turns to face the station; 8-10 drive the 1 m into the bay; 11-20 hold
  at `DOCK`; 21-22 undock. The camera, facing north from the lane, sees the corridor from the side: the box `NEAR_BOX`
  (below) from offset 3 (base (6.0, 2.0)).
- `line_run` drives in along the corridor's own line (the vacuum's back-out, which `rear_clear` also has to answer):
  offsets 0-16 wait at (1.0, 4.98) facing the station, 16-20 drive in 1.5 m a frame to `DOCK`, 21-22 undock (the base
  stays). The camera at (1.0, 4.98) (offset 16) sees the corridor's whole length ahead of it.
`look_back` turns one keyframe round at the dock to look down the corridor; `world` is the compiler's window of a
script, `asked` its during questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from space_kit import JOB as SPACE_JOB
from space_kit import NO_DEPTH, Box, SpaceScript, box, render_depth
from space_kit import record as space_record
from stbench import episode as ep
from stbench import grid, questionnaire
from stbench.camera import Camera
from stbench.events import EventRule
from stbench.job_registry import JobSpec, Relevance
from stbench.questions_select import Cand
from stbench.questions_world import Inputs, World
from stbench.robots import ROBOTS, camera_doc
from synth import WAREHOUSE

AMR = ROBOTS["amr"]
CAM = Camera.load({**camera_doc(AMR), "depth_unit": "mm"})   # the AMR's own: 0.2 to 10 m, millimetres
AXIS_Y, LANE_X = 4.98, 6.0
DOCK = (7.0, AXIS_Y)                  # the docked base, facing +x
T = (LANE_X, AXIS_Y)                  # where the lane meets the approach line
REAR_X = DOCK[0] - AMR.body.length_m / 2   # 6.6: the corridor starts here and runs west
FRAMES, DECISION = 23, 20             # keyframes a task, and the `before_undock` offset
KIND = "before_undock"

JOB = replace(
    SPACE_JOB, id="amr/1", robot="amr", domain="warehouse", decisions={KIND: ("rear_clear",)},
    events={"when you reached the station": EventRule("first", "at_station")},
    relevance=lambda plan: Relevance(frozenset({"tote", "pallet", "drum", "bin", "sack"}),
                                     frozenset(WAREHOUSE.rooms), frozenset({"transit", "approach", "at_station"})))

# -- the boxes the tests put in the corridor: (x0, x1, y0, y1, z0, z1) ------------------------------------------------
# NEAR_BOX stands across the corridor, its east face 2.11 m behind the rear edge (x 4.49): the sub-cell [4.48, 4.50)
# holds that face, its centre less half a sub-cell is 2.1 m from the rear edge.
NEAR_BOX = box(3.99, 4.49, AXIS_Y - 0.25, AXIS_Y + 0.25, 0.0, 0.5)


def beside(offset: float, x0: float = 4.97, x1: float = 5.49) -> Box:
    """A box 0.5 m deep and 0.5 m tall whose near side lies `offset` m north of the corridor's axis and which stands
    beyond it (x0..x1): the corridor's width is 0.58, so beyond 0.29 it is beside it, inside that it is across it."""
    return box(x0, x1, AXIS_Y + offset, AXIS_Y + offset + 0.5, 0.0, 0.5)


@dataclass
class RearScript(SpaceScript):
    """`SpaceScript` whose depth is rendered through the AMR's camera (`CAM`), not the vacuum's."""

    def depth_of(self, k: int):
        if self.depth_ks is not None and k not in self.depth_ks:
            return NO_DEPTH
        pos, rot = (self._cameras or self.cameras())[k]
        return render_depth(CAM, pos, rot, self.boxes + self.frame_boxes.get(k, []), self.stage.base_z)


def _run(legs: list, place, n_tasks: int, fields: dict) -> RearScript:
    poses, yaws = {}, {}
    for t in range(n_tasks):
        for o in range(FRAMES):
            poses[t * FRAMES + o], yaws[t * FRAMES + o] = place(o)
    setup = {"tasks": [list(legs) for _ in range(n_tasks)], "stage": WAREHOUSE, "poses": poses, "yaws": yaws,
             "decisions": [(t * FRAMES + DECISION, KIND) for t in range(n_tasks)], "depth_ks": set()}
    return RearScript(**{**setup, **fields})


def _lane(o: int) -> tuple[tuple[float, float], float]:
    if o <= 5:
        return (LANE_X, float(o - 1)), math.pi / 2
    if o == 6:
        return T, math.pi / 2
    if o <= 10:
        return (round(LANE_X + (o - 7) / 3, 2), AXIS_Y), 0.0
    return DOCK, 0.0


def _line(o: int) -> tuple[tuple[float, float], float]:
    x = 1.0 if o <= 16 else 1.0 + 1.5 * (o - 16) if o <= 19 else DOCK[0]
    return (x, AXIS_Y), 0.0


def lane_run(n_tasks: int = 1, **fields) -> RearScript:
    """`n_tasks` side arrivals (the module docstring); `fields` replace `SpaceScript`'s (`boxes`, `depth_ks`,
    `frame_boxes`, `seed` ...). No keyframe has depth unless `depth_ks` names it."""
    legs = [("west aisle", 8, "transit"), ("dock", 3, "approach"), ("dock", 10, "at_station"),
            ("dock", 2, "undocking")]
    return _run(legs, _lane, n_tasks, fields)


def line_run(n_tasks: int = 1, **fields) -> RearScript:
    """`n_tasks` straight-in drives (the module docstring); `fields` as `lane_run`'s."""
    legs = [("west aisle", 16, "transit"), ("dock", 4, "approach"), ("dock", 1, "at_station"),
            ("dock", 2, "undocking")]
    return _run(legs, _line, n_tasks, fields)


def look_back(script: RearScript, k: int) -> RearScript:
    """Keyframe `k` of the script's first task stands at the dock turned round to face the aisle, so its camera looks
    back along the corridor from the rear edge (the floor is in view from 0.53 m beyond its front edge, which is now
    the old rear edge); the script is changed in place and returned."""
    script.poses[k], script.yaws[k] = DOCK, math.pi
    return script


def inputs(script: RearScript, job: JobSpec = JOB) -> Inputs:
    """The compiler's inputs for the script in memory: free-space cells from its own ray-cast depth through the pose
    rows' cameras, the AMR's band (`space_kit.inputs` renders through the vacuum's camera)."""
    tl = script.timeline()
    cameras, band, done = script.cameras(), grid.band(AMR), {}

    def states_of(ks):
        for k in ks:
            if k not in done:
                done[k] = grid.observe(grid.points(script.depth_of(k), CAM, *cameras[k]), tl.keyframe(k).base[2], band)
        return {k: done[k] for k in ks}

    return Inputs(tl, CAM, AMR, job, ep.pose_rows(tl), script.visibility(), states_of)


def world(script: RearScript, a: int = 0, b: int | None = None, cid: str = "c0", length: str = "short") -> World:
    """The window of tasks a..b of the script (b defaults to a) as the compiler's `World`."""
    return World(inputs(script), space_record(script, a=a, b=b, cid=cid, length=length))


def asked(script: RearScript) -> list[Cand]:
    """The during questions of a one-task window: with the script's decision, one."""
    return questionnaire.during(world(script))
