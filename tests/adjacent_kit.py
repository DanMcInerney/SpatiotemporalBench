"""What the `adjacency` and `adjacent` tests share: a hand-made map of six rooms, and the robot's walks over it.

`kit_map()` is a rectilinear apartment (walls 0.1 m, rooms on their walls' inner faces, so a wall between two rooms is a
0.1 m gap between their polygons), x east and y north:

    study    x 0..3,   y 3.1..6.1    kitchen   x 3.1..6.0, y 3.1..6.1    bedroom  x 6.1..9.1,  y 2..5
    hall     x 0..3,   y 0..3        corridor  x 3.1..6.1, y 0..3        bathroom x 8.6..11.6, y 0.5..2

R1 hall, R2 corridor, R3 kitchen, R4 study, R5 bedroom, R6 bathroom; each name is its kind. Joined: hall-corridor (door
D1, at (3.05, 1.5)), corridor-kitchen (door D2, (4.6, 3.05)), kitchen-study (arch O1, (3.05, 4.6)) and corridor-bedroom
(an open shared edge, x = 6.1, y 2..3: 1.0 m). Not joined: hall and study (a wall between, nothing cut), kitchen and
bedroom (a wall, x 6.0..6.1), bedroom and bathroom (an open shared edge of 0.5 m, x 8.6..9.1, y = 2). The entrance D0
(outside to the hall) joins nothing. `kit_map(hall_study_door=True)` cuts a door D5 in the wall between hall and study,
`kit_map(kitchen_bedroom_door=True)` a door D6 in the wall between kitchen and bedroom (with the corridor, a triangle).

A walk is a list of base positions, one a keyframe, 1.5 s apart (`HALL`, `COR`, `KIT` ... a place in a room; `D1`, `D2`,
`ARCH` a doorway, which is in no room). `script(walk, ...)` is one task of the vacuum along the walk (a list of walks:
one task each); `world` is the window of it as the compiler's `World`, over the kit map, a job that asks `adjacent`, and
hand free-space states (each room's `observed` share of its fine cells, from the lower left, in the given state, at
the window's last keyframe). The end look window is the last 8 steps of the window (`look_windows.LOOK_WINDOW`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import family_kit as fk
from space_kit import JOB as SPACE_JOB
from space_kit import SpaceScript
from space_kit import record as space_record
from stbench import mapspec
from stbench.catalog import CATALOGS
from stbench.job_registry import JobSpec, Relevance
from stbench.questions_world import Inputs, World, cached_states
from stbench.timeline_ego import TaskPlan
from synth import HOME, Stage

MAP_NAME = "adjacent_kit"
ROOMS = {"R1": ("hall", 0.0, 0.0, 3.0, 3.0), "R2": ("corridor", 3.1, 0.0, 6.1, 3.0),
         "R3": ("kitchen", 3.1, 3.1, 6.0, 6.1), "R4": ("study", 0.0, 3.1, 3.0, 6.1),
         "R5": ("bedroom", 6.1, 2.0, 9.1, 5.0), "R6": ("bathroom", 8.6, 0.5, 11.6, 2.0)}  # id: name, x0 y0 x1 y1
HALL_ID, COR_ID, KIT_ID, STUDY_ID, BED_ID, BATH_ID = ("R1", "R2", "R3", "R4", "R5", "R6")
HALL, COR, COR_E, KIT, STUDY, BED = (1.5, 1.5), (4.6, 1.5), (5.9, 2.5), (4.6, 4.6), (1.5, 4.6), (7.5, 3.5)
D1, D2, ARCH = (3.05, 1.5), (4.6, 3.05), (3.05, 4.6)  # COR_E: the corridor, 0.2 m from its open edge to the bedroom
STAGE = Stage(MAP_NAME, "vacuum", HOME.rooms, CATALOGS["home"], 0, None, None)
JOB: JobSpec = replace(SPACE_JOB, decisions={}, ends=("adjacent",),
                       assignments={"clean_room": "Its current assignment is to clean the {room}."},
                       relevance=lambda plan: Relevance(frozenset(), frozenset(), frozenset()))


def kit_map(*, hall_study_door: bool = False, kitchen_bedroom_door: bool = False) -> mapspec.MapSpec:
    """The six-room apartment of the module docstring."""
    def door(oid, wall, at, between, width=0.9, kind="door", interior=True):
        return {"id": oid, "type": kind, "wall": wall, "at": at, "width": width, "height": 2.0,
                "between": list(between), "interior": interior, "hinge": "a", "swing": "left"}

    def wall(wid, a, b, exterior=False):
        return {"id": wid, "a": list(a), "b": list(b), "thickness": 0.1, "height": 2.5, "exterior": exterior}

    openings = [door("D0", "W0", 1.5, ("outside", "R1"), 0.95, interior=False), door("D1", "W1", 1.5, ("R1", "R2")),
                door("O1", "W1", 4.6, ("R4", "R3"), 1.2, "arch"), door("D2", "W2", 4.6, ("R2", "R3"))]
    if hall_study_door:
        openings.append(door("D5", "W2", 1.5, ("R1", "R4")))
    if kitchen_bedroom_door:
        openings.append(door("D6", "W3", 1.0, ("R3", "R5")))
    raw = {"name": MAP_NAME, "bounds": [0.0, -0.1, 11.6, 6.1], "wall_thickness": 0.1,
           "wall_height": 2.5,
           "regions": [{"id": rid, "name": name, "kind": name, "level": 0, "z0": 0.0, "z1": 2.5,
                        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "anchor": [(x0 + x1) / 2, (y0 + y1) / 2]}
                       for rid, (name, x0, y0, x1, y1) in ROOMS.items()],
           "walls": [wall("W0", (0.0, -0.05), (3.0, -0.05), True), wall("W1", (3.05, 0.0), (3.05, 6.1)),
                     wall("W2", (0.0, 3.05), (6.1, 3.05)), wall("W3", (6.05, 3.1), (6.05, 5.0))],
           "openings": openings, "windows": [], "furniture": [], "ceiling_lights": []}
    return mapspec.parse_map(raw)


def room_cells(room: str) -> list[tuple[int, int]]:
    """The room's inside fine cells, row by row from the lower left: every room's walls lie on tenths, so these are
    the 0.1 m squares between them (900 for a 3 x 3 m room)."""
    _, x0, y0, x1, y1 = ROOMS[room]
    return [(u, v) for v in range(round(y0 * 10), round(y1 * 10)) for u in range(round(x0 * 10), round(x1 * 10))]


@dataclass
class KitScript(SpaceScript):
    """A `SpaceScript` whose tasks' plans may name any room: `rooms[t]` is task t's plan parameter `room`, the word the
    brief says (None: the stage's first room, "kitchen"... as `Synth` names it)."""

    rooms: tuple[str, ...] = ()

    def timeline(self):
        tl = super().timeline()
        if not self.rooms:
            return tl
        tasks = [replace(t, plan=TaskPlan(t.plan.kind, {"room": room})) for t, room in zip(tl.ego.tasks, self.rooms,
                                                                                            strict=True)]
        return replace(tl, ego=replace(tl.ego, tasks=tasks))


def script(*walks, rooms: tuple[str, ...] = ("living room",), **fields) -> KitScript:
    """A task of the vacuum along each walk (a walk is a list of (x, y)); the window's last 8 steps are the end look
    window. `rooms` names the room of each task's plan, the word its brief says ("living room": no room of the map);
    `fields` replace `SpaceScript`'s (an unseen item, say)."""
    walks = [list(walk) for walk in walks]
    starts = [sum(len(w) for w in walks[:i]) for i in range(len(walks))]
    setup = {"tasks": [[("living room", len(walk))] for walk in walks], "depth_ks": set(),
             "poses": {start + i: xy for start, walk in zip(starts, walks, strict=True) for i, xy in enumerate(walk)},
             "decisions": [], "rooms": rooms if len(rooms) == len(walks) else rooms * len(walks), "stage": STAGE}
    return KitScript(**{**setup, **fields})


def inputs(s: KitScript, job: JobSpec = JOB, observed: dict[str, tuple[float, str]] | None = None,
           kit: mapspec.MapSpec | None = None) -> Inputs:
    """The compiler's inputs for the script over the kit map: every room observed (`observed` maps a room id to
    (share, state) and replaces a room's default (1.0, "free"))."""
    seen = {rid: (1.0, "free") for rid in ROOMS} | (observed or {})
    cells = {c: state for rid, (share, state) in seen.items()
             for c in room_cells(rid)[:round(share * len(room_cells(rid)))]}
    return replace(fk.inputs(s, job), map=kit or kit_map(), states_of=cached_states(
        lambda ks: {k: cells if k == max(ks) else {} for k in ks}))


def world(s: KitScript, job: JobSpec = JOB, *, observed=None, kit=None, first: int = 0, cid: str = "c0") -> World:
    """The window of the script's tasks from `first` to its last as the compiler's `World`; the end look window is its
    last 8 keyframes."""
    w = World(inputs(s, job, observed, kit), space_record(s, a=first, b=len(s.tasks) - 1, cid=cid))
    assert list(w.end_window) == list(range(w.rec.k1 - 8, w.rec.k1))
    return w
