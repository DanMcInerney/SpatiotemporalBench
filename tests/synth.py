"""A hand-built script for compiler, harness and baseline tests: two rooms, three tasks, hand-set sights.

Rooms: the stage's first room x 0..4 and second room x 4..8 (both y 0..3): the kitchen and the living room of
`apartment_a` (`HOME`, the vacuum), the dock and west aisle of `warehouse_a` (`WAREHOUSE`, the tote robot), whose items
are drawn from the warehouse catalog, or the high street and west street of `downtown_a` (`DOWNTOWN`, the drone at base
height 12 m), whose items are drawn from the downtown catalog and whose depth images are in centimetres (the robot's
`depth.unit`; the camera file says so).

A task is a list of (room, frames[, phase]) legs; each leg is a look-around of `frames` keyframes (heading 0, 45, ...
degrees) with the base at the room's centre and the camera at the robot's default pitch, 1.5 s apart; `poses` moves
chosen keyframes' bases to another (x, y), to put the robot near a room's outline. Each task's first keyframe is in the
robot's start set (its start phase, each status range at its upper bound); the other keyframes of a leg are in the leg's
phase (the start phase when none), and `status` overrides a keyframe's status keys. Tasks start 7200 s apart.
`decisions` lists (keyframe, kind) decision points.

`see(k, entity, px)` says how many pixels an entity covered at keyframe k (>= 269 is seen by the compiler and >= 215
detected by the oracle channel; 1..268 is the compiler's dead band; SEEN = 300 and GLIMPSE = 100 sit clear of both
edges; the recognisability floor is 1024 px, a luminance of 0.01, a box 2 px clear of the image's edge and a depth, so
RECOG = 1500 at LUM = 0.25 in the box BBOX at the depth DEPTH_M clears all four; `see`'s `bbox` and `depth` (None: no
depth pixel) set the box and median depth of a sighting); nothing else is inferred from geometry, so every expectation
in a test is hand-computed. A sighting's `rgb_med` is `see`'s `rgb`, else the faithful colour of the entity's palette
word (`RENDER_RGB`), so a thing renders as its label unless a test says otherwise; `rgb_med = False` gives an older
root's log, with no `rgb_med` (the label stands for the render).

`write` lays the whole thing out as a script: timeline, robot, camera, pose, visibility, depth, oracle channel and
perception ids, and `write_case` adds a case over a window of its tasks. The flat depth image and `depth_with_wall`
count in the stage's unit. The script is not valid against its map (the rooms are not the map's):
`conftest.script_on_map` makes one that is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from builders import keyframe, start_status
from stbench import case as case_mod
from stbench import episode as ep
from stbench import robots
from stbench.camera import Camera, map_to_cam
from stbench.catalog import CATALOGS, Catalog
from stbench.geometry import quat_z
from stbench.timeline import (
    DecisionPoint,
    Ego,
    Entity,
    EpisodeInfo,
    Placement,
    Region,
    Task,
    TaskPlan,
    Timeline,
)

RECTS = (((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)), ((4.0, 0.0), (8.0, 0.0), (8.0, 3.0), (4.0, 3.0)))
CENTRE_XY = ((2.0, 1.5), (6.0, 1.5))
TASK_GAP_S = 7200.0
DT = 1.5
FLAT_DEPTH_M = 6  # the depth of every synthetic depth image where nothing nearer is set
DEPTH_UNIT_PER_M = {"mm": 1000, "cm": 100, "m": 1}  # depth image counts in a metre, by the camera file's `depth_unit`
DEPTH_MM = FLAT_DEPTH_M * DEPTH_UNIT_PER_M["mm"]
START = "2026-03-02T08:00:00"
SEEN, GLIMPSE = 300, 100
RECOG, LUM = 1500, 0.25  # a view the Perceive set recognises (>= 1024 px) and the mean luminance every sighting has
BBOX, DEPTH_M = [100, 100, 140, 140], 2.0  # a sighting's box (clear of the 640 x 480 image's edges) and median depth
# The sRGB median a faithful picture gives a palette colour (the visibility log's `rgb_med`), written by hand: each is
# one that `colour_seen.classify` names with a single word, the colour's own (test_colour_seen holds the kit to that).
RENDER_RGB = {"black": (15, 15, 15), "grey": (85, 85, 85), "white": (225, 225, 225), "beige": (190, 175, 150),
              "brown": (75, 55, 40), "red": (200, 30, 30), "orange": (240, 130, 20), "yellow": (240, 210, 30),
              "green": (30, 150, 50), "blue": (30, 60, 200), "purple": (140, 40, 160), "pink": (240, 140, 180)}


def camera(depth_unit: str = "mm") -> Camera:
    return Camera.load({**robots.camera_doc(robots.ROBOTS["vacuum"]), "depth_unit": depth_unit})


@dataclass(frozen=True)
class Stage:
    """Where a synthetic script is set: a bundled map (its name decides the catalog the compiler samples the never-seen
    descriptions from), the robot and the height of its base, two rooms side by side, the catalog the carried items are
    drawn from, how many there are, and the light furniture (`chair`: a kind and size, with the colours of its two
    pieces, the second of which shifts; `stool`: a kind, colour and size; None on a map that has none, as the city)."""

    map_name: str
    robot: str
    rooms: tuple[str, str]
    catalog: Catalog
    n_items: int
    chair: tuple[str, tuple[float, float, float], tuple[str, str]] | None
    stool: tuple[str, str, tuple[float, float, float]] | None
    base_z: float = 0.0

    @property
    def depth_unit(self) -> str:
        return robots.ROBOTS[self.robot].depth.unit


HOME = Stage("apartment_a", "vacuum", ("kitchen", "living room"), CATALOGS["home"], 34,
             ("chair", (0.4, 0.4, 0.9), ("oak", "oak")), ("stool", "oak", (0.3, 0.3, 0.45)))
# 24 items, as the generator's 22-28: `busy` draws every description of its unique items from the catalog's (kind,
# colour) pairs without repeats, and the warehouse's 31 pairs cannot hold the 34 of the home stage (35 are drawn)
WAREHOUSE = Stage("warehouse_a", "amr", ("dock", "west aisle"), CATALOGS["warehouse"], 24,
                  ("pallet_jack", (1.55, 0.55, 1.2), ("red", "yellow")), ("roll_cage", "grey", (0.8, 0.7, 1.7)))
# 28 items, as the city's 26-34; no light furniture (the city has none) and depth in centimetres. The rooms are the
# stage's 4 x 3 m rooms, as the warehouse's: what the stage tests is the compiler's rules, not the drone's route.
DOWNTOWN = Stage("downtown_a", "drone", ("high street", "west street"), CATALOGS["downtown"], 28, None, None,
                 base_z=12.0)


@dataclass
class Synth:
    tasks: list[list[tuple]]  # per task, (room, frames) or (room, frames, phase) legs
    seed: int = 1
    start: str = START
    kind: str = "clean_room"  # the plan kind of every task
    gap_s: float = TASK_GAP_S  # seconds from the start of one task to the start of the next
    entities: list[Entity] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)
    px: dict[int, dict[str, int]] = field(default_factory=dict)
    lum: dict[int, dict[str, float]] = field(default_factory=dict)  # per-sighting override of the mean luminance (LUM)
    bbox: dict[int, dict[str, list[int]]] = field(default_factory=dict)  # per-sighting override of the box (BBOX)
    median_m: dict[int, dict[str, float | None]] = field(default_factory=dict)  # of the median depth (DEPTH_M)
    rgb: dict[int, dict[str, tuple[int, int, int]]] = field(default_factory=dict)  # of the sRGB median (RENDER_RGB)
    rgb_med: bool = True  # False: the log of an old root, no `rgb_med`: the label stands for the render
    depth: dict[int, np.ndarray] = field(default_factory=dict)  # per-keyframe override of the flat depth image
    poses: dict[int, tuple[float, float]] = field(default_factory=dict)  # per-keyframe override of the base (x, y)
    decisions: list[tuple[int, str]] = field(default_factory=list)  # (keyframe, kind)
    status: dict[int, dict[str, object]] = field(default_factory=dict)  # per-keyframe override of the status keys
    stage: Stage = HOME

    # -- building ----------------------------------------------------------------------------------------------
    def item(self, eid: str, kind: str, attrs: dict, where: list[tuple[float, float | None, tuple]],
             mobility: str = "carried", size=(0.1, 0.1, 0.1)) -> str:
        """An entity with placements [(t0, t1, (x, y, z))]; z is the base height."""
        self.entities.append(Entity(eid, kind, dict(attrs), mobility, tuple(size)))
        self.placements += [Placement(eid, t0, t1, tuple(pos), quat_z(0.0), None) for t0, t1, pos in where]
        return eid

    def see(self, ks, eid: str, px: int = SEEN, lum: float | None = None, bbox: list[int] | None = None,
            depth: float | None = DEPTH_M, rgb: tuple[int, int, int] | None = None) -> None:
        """The entity covered `px` pixels at each keyframe, of mean luminance `lum` (LUM when not given), in the box
        `bbox` (BBOX), at the median depth `depth` metres (DEPTH_M; None: no pixel of it has a depth), in the sRGB
        median `rgb` (the faithful one of its colour word, `RENDER_RGB`, when not given)."""
        for k in [ks] if isinstance(ks, int) else ks:
            self.px.setdefault(k, {})[eid] = px
            if rgb is None:
                self.rgb.get(k, {}).pop(eid, None)
            else:
                self.rgb.setdefault(k, {})[eid] = rgb
            for given, default, table in ((lum, None, self.lum), (bbox, None, self.bbox),
                                          (depth, DEPTH_M, self.median_m)):
                if given == default:
                    table.get(k, {}).pop(eid, None)
                else:
                    table.setdefault(k, {})[eid] = given

    # -- derived -----------------------------------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return sum(leg[1] for task in self.tasks for leg in task)

    def t_of(self, k: int) -> float:
        """The time of keyframe k: 7200 s a task plus 1.5 s a keyframe inside it."""
        starts = self.task_starts()
        v = max(i for i, k0 in enumerate(starts) if k0 <= k)
        return v * self.gap_s + DT * (k - starts[v])

    def task_starts(self) -> list[int]:
        out, k = [], 0
        for task in self.tasks:
            out.append(k)
            k += sum(leg[1] for leg in task)
        return out

    def timeline(self) -> Timeline:
        robot = robots.ROBOTS[self.stage.robot]
        start_phase, base_status = robot.status.start["phase"], start_status(self.stage.robot)
        frames, tasks, k = [], [], 0
        for vi, legs in enumerate(self.tasks):
            k0, t = k, vi * self.gap_s
            for room, n, *phase in legs:
                for i in range(n):
                    cx, cy = self.poses.get(k, CENTRE_XY[self.stage.rooms.index(room)])
                    first = k == k0
                    frames.append(keyframe(
                        self.stage.robot, k, t, (cx, cy, self.stage.base_z), math.radians(45 * (i % 8)), task=vi,
                        phase=start_phase if first or not phase else phase[0],
                        status={**base_status, **self.status.get(k, {})}))
                    k, t = k + 1, t + DT
            tasks.append(Task(vi, k0, k, TaskPlan(self.kind, {"room": legs[0][0]}), ()))
        regions = [Region(f"R{i + 1}", name, name, poly, 0, 0.0, 2.5)
                   for i, (name, poly) in enumerate(zip(self.stage.rooms, RECTS, strict=True))]
        info = EpisodeInfo(self.stage.map_name, self.seed, self.stage.robot, f"{self.stage.robot}/1", self.start)
        return Timeline(info, regions, list(self.entities), list(self.placements), [],
                        Ego(frames, tasks, [DecisionPoint(dk, kind) for dk, kind in self.decisions]))

    def visibility(self) -> list[dict]:
        """The log: rows with an `rgb_med` for every entity of a palette colour (a thing with no such colour, a
        person, has none), or with `rgb_med` False, rows without it (an old root's)."""
        colour = {e.id: e.attributes.get("color") for e in self.entities}

        def record(k: int, e: str, p: int) -> dict:
            rec = {"id": e, "px": p, "bbox": self.bbox.get(k, {}).get(e, BBOX),
                   "depth_med_m": self.median_m.get(k, {}).get(e, DEPTH_M), "lum": self.lum.get(k, {}).get(e, LUM)}
            rgb = self.rgb.get(k, {}).get(e) or RENDER_RGB.get(colour.get(e))
            if self.rgb_med and rgb is not None:
                rec["rgb_med"] = list(rgb)
            return rec

        return [{"k": k,
                 "entities": [record(k, e, p) for e, p in sorted(self.px.get(k, {}).items())]}
                for k in range(self.n_frames)]

    def depth_of(self, k: int) -> np.ndarray:
        flat = FLAT_DEPTH_M * DEPTH_UNIT_PER_M[self.stage.depth_unit]
        return self.depth.get(k, np.full((480, 640), flat, dtype=np.uint16))

    # -- on disk -----------------------------------------------------------------------------------------------
    def write(self, dataset: Path, perception: bool = True) -> case_mod.ScriptPaths:
        """The script as files: agent-facing under scripts/<sid>, truth under private/<sid>."""
        tl = self.timeline()
        paths = ep.write_script(dataset, tl)
        paths.mkdirs()
        ep.write_jsonl(paths.visibility, self.visibility())
        for k in range(self.n_frames):
            Image.fromarray(self.depth_of(k).astype(np.uint16)).save(paths.depth(k))
        if perception:
            rows, ids = self.oracle(tl)
            ep.write_jsonl(paths.oracle, rows)
            ep.write_jsonl(paths.perception_ids, ids)
        return paths

    def write_case(self, dataset: Path, a: int = 0, b: int | None = None, length: str = "short") -> case_mod.CasePaths:
        """The script (`write`) and a case over its tasks a..b (b defaults to a)."""
        return case_mod.write_case(dataset, self.write(dataset).id, a, a if b is None else b, length)

    def oracle(self, tl: Timeline) -> tuple[list[dict], list[dict]]:
        """Oracle detections for every entity seen (>= 215 px), with true camera-frame centres, and the id map."""
        rows, ids = [], []
        for kf in tl.ego.keyframes:
            dets, key = [], {}
            for j, (eid, px) in enumerate(sorted(self.px.get(kf.k, {}).items())):
                if px < 215:
                    continue
                e = tl.entity(eid)
                pos = tl.placement_of(eid, kf.t).pos
                centre = np.array([pos[0], pos[1], pos[2] + e.size[2] / 2])
                d = f"{kf.k:04x}{j:04x}"
                attrs = {a: v for a, v in e.attributes.items() if a in ("color", "size")}
                dets.append({"d": d, "kind": e.kind, "attributes": attrs, "bbox_px": [0, 0, 20, 20],
                             "center_cam": [round(float(x), 4) for x in map_to_cam(centre, kf.pos, kf.rot)],
                             "center_map": [round(float(x), 4) for x in centre],
                             "extent_m": list(e.size)})
                key[d] = eid
            rows.append({"k": kf.k, "clock": tl.clock(kf.t), "detections": dets})
            ids.append({"k": kf.k, "ids": key})
        return rows, ids


def depth_with_wall(wall_m: float, u0: int = 0, u1: int = 640, unit: str = "mm") -> np.ndarray:
    """A depth image (counts of `unit`) of flat depth (6 m) with a nearer surface (wall_m metres away) across columns
    u0..u1."""
    per_m = DEPTH_UNIT_PER_M[unit]
    img = np.full((480, 640), FLAT_DEPTH_M * per_m, dtype=np.uint16)
    img[:, u0:u1] = int(wall_m * per_m)
    return img


def clock_of(t: float) -> str:
    """The delivered clock for t seconds after the start, written out by hand: START plus t rounded half up to the
    tenth, as `YYYY-MM-DDTHH:MM:SS.s` with its one decimal always (the case's clock, `case.case_clock`)."""
    from datetime import datetime, timedelta

    tenths = int(t * 10 + 0.5)
    at = datetime.fromisoformat(START) + timedelta(milliseconds=100 * tenths)
    return at.strftime("%Y-%m-%dT%H:%M:%S") + f".{tenths % 10}"


def three_tasks(seed: int = 1, stage: Stage = HOME) -> Synth:
    """Tasks k 0-15 (first room 0-7, second room 8-15), 16-31 (same), 32-39 (first room); the third task starts at
    keyframe 32. The first room is the kitchen and the second the living room, on the home stage.

    Keyframe k of task v at offset i is at t = 7200 v + 1.5 i, e.g. k=20 is t=7206 and k=3 is t=4.5.
    """
    a, b = stage.rooms
    return Synth(tasks=[[(a, 8), (b, 8)], [(a, 8), (b, 8)], [(a, 8)]], seed=seed, stage=stage)


def busy(seed: int, stage: Stage = HOME) -> Synth:
    """A world with enough of everything for every family's quota: 34 unique carried items in six fates, a pair of
    identical items (B1, B2), identical oak chairs, a stool (the home stage; the warehouse stage has 24 items from the
    warehouse catalog, a red and a yellow pallet jack and a roll cage; the downtown stage has 28 items from the downtown
    catalog and no light furniture). An item has one of six fates: 0 seen in task 0 then removed, 1 static and seen in
    tasks 0 and 1, 2 seen in both and moved, 3 appears and is seen in task 1, 4 static and seen in task 1 only, 5
    never seen. Descriptions, fates, positions, sighting frames and the start clock all come from `seed`, so nothing
    about one seed predicts another. Fates 1 and 2 come in same-kind pairs, so moved_between's per-kind balance has
    pairs to pick."""
    import random
    from datetime import datetime, timedelta

    kinds = stage.catalog.kinds
    first, second = stage.rooms
    rng = random.Random(seed)
    s = three_tasks(seed, stage)
    s.start = (datetime.fromisoformat(START) + timedelta(days=3 * seed, minutes=7 * seed)).isoformat()
    taken: list[tuple[float, float]] = []

    def spot(room: str | None = None) -> tuple[float, float, float]:
        for _ in range(2000):
            where = room or rng.choice([first, second])
            x = rng.uniform(0.4, 3.6) + (4.0 if where == second else 0.0)
            y = rng.uniform(0.4, 2.6)
            if all(math.dist((x, y), p) > 0.3 for p in taken):
                taken.append((x, y))
                return (round(x, 2), round(y, 2), 0.9)
        raise RuntimeError("no free spot in the synthetic world")

    def leg_frame(task: int, pos: tuple) -> int:
        return 16 * task + (8 if pos[0] >= 4 else 0) + rng.randrange(8)

    # moved_between balances yes and no inside each kind, so the moved items (fate 2) and the static ones seen in both
    # tasks (fate 1) come in pairs of one kind, in two colours (six pairs of 34 items); every other description is
    # drawn from the rest, one more than the items need (the last is the identical pair's)
    fates = [i % 6 for i in range(stage.n_items)]
    n_pairs = fates.count(1)
    kinds6 = rng.sample(sorted(k for k in kinds if len(kinds[k].colors) >= 2), n_pairs)  # a pair needs two colours
    paired = {k: rng.sample(kinds[k].colors, 2) for k in kinds6}
    statics = [(k, cs[0]) for k, cs in paired.items()]
    movers = [(k, cs[1]) for k, cs in paired.items()]
    rest = rng.sample([(k, c) for k in kinds for c in kinds[k].colors if (k, c) not in statics + movers],
                      len(fates) - n_pairs - fates.count(2) + 1)
    pair_kind, pair_color = rest.pop()
    rng.shuffle(fates)
    pools = {1: iter(statics), 2: iter(movers)}
    rest_it = iter(rest)
    combos = [next(pools[f]) if f in pools else next(rest_it) for f in fates]
    for i, ((kind, color), fate) in enumerate(zip(combos, fates, strict=True)):
        eid, a = f"I{i:02d}", spot()
        attrs = {"color": color}
        if fate == 0:
            s.item(eid, kind, attrs, [(0, 3600, a)])
            s.see(leg_frame(0, a), eid)
        elif fate == 1:
            s.item(eid, kind, attrs, [(0, None, a)])
            s.see([leg_frame(0, a), leg_frame(1, a)], eid)
        elif fate == 2:
            b = spot()
            s.item(eid, kind, attrs, [(0, 3600, a), (3600, None, b)])
            s.see([leg_frame(0, a), leg_frame(1, b)], eid)
        elif fate == 3:
            s.item(eid, kind, attrs, [(3600, None, a)])
            s.see(leg_frame(1, a), eid)
        elif fate == 4:
            s.item(eid, kind, attrs, [(0, None, a)])
            s.see(leg_frame(1, a), eid)
        else:
            s.item(eid, kind, attrs, [(0, None, a)])
    for eid in ("B1", "B2"):
        s.item(eid, pair_kind, {"color": pair_color}, [(0, None, spot(first))])
        s.see([4, 20], eid)
    if stage.chair and stage.stool:
        chair, size, (colour1, colour2) = stage.chair
        floor = {"mobility": "light", "size": size}
        c1, c2, c2b = (spot(first)[:2] + (0.0,) for _ in range(3))
        s.item("L1", chair, {"color": colour1}, [(0, None, c1)], **floor)
        s.item("L2", chair, {"color": colour2}, [(0, 3600, c2), (3600, None, c2b)], **floor)
        s.see([5, 21], "L1")
        s.see([5, 21], "L2")
        stool = spot(second)[:2] + (0.0,)
        kind, colour, size = stage.stool
        s.item("S1", kind, {"color": colour}, [(0, None, stool)], mobility="light", size=size)
        s.see([leg_frame(0, stool), leg_frame(1, stool)], "S1")
    return s
