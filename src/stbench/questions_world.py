"""What the question compiler knows about a case: its window of a script, read through what a system is sent.

`Inputs` is everything a compile reads (the private timeline and visibility log, the script's pose rows, camera and
robot, the job, the free-space rule over its depth, and the points of a thing's pixels); `load_inputs` reads them from a
case's files. `World` is the
window `[rec.k0, rec.k1)` only:
- **Sightings come from the window's visibility rows alone**, so nothing before the window is evidence (design §5): a
  thing seen only before it has no pixel here and is unknown.
- **A sighting is evidence only if its own frame is recognisable** (`World.recognised`, `recognise.recognisable`:
  enough pixels and light, a box clear of the image's edge, a depth). A gold cites the last such frame at or before its
  moment; a clipped exit frame or a depth-less one is a sighting (`World.sighted`) but never the evidence. `stale`,
  `arrived` and `steady` say whether a place a thing had is out of date, was reached by a move, or is kept by every
  sighting of a stretch.
- **A colour word is read from the render, not the label** (`World.rendered`, `may_be_taken_for`, `colour_holds`,
  `sighted_looks`): a recognisable frame's `rgb_med` is named by `colour_seen`, and the screens of
  `questions_where` ask what a viewer of the pictures could take a thing for. An older root's visibility log has no
  `rgb_med`: the label stands, a neutral one for all three neutrals (black, grey and white are one group when two
  things are compared).
- **A thing's visible part comes from the pictures** (`World.nearest_part`: the floor hull of its own instance pixels,
  through the keyframe's depth, over the keyframes asked, and its point nearest a ground point; `nearest_part`), so a
  gold that names keyframes up to `j` reads no frame after `j`.
- **Every pose a gold uses is the delivered one** (`sensors/pose.jsonl`: `yaw_deg` is rounded to 3 places there), never
  `tl.keyframe(k)`'s floats, so a system's mapper reads exactly what the gold read and the oracle reproduces it.
- **The recent view slides** (0.10.0): a look point is any frame, and `ego_mem` and `person_mem` take what was seen
  before `look_windows.recent` (the last `LOOK_WINDOW` frames of the current task up to the point) and not in it. The
  case's last `LOOK_WINDOW` frames are `end_window`, the view the end questions are asked from. A system never sees
  either: it gets one frame at a time, so a `now` gold reads the latest frame `k` alone.
- **Decision windows** (`windows`, `window_ks`, `window_of`): `look_windows.windows` over the window, cut after every
  decision point (asked or not) and every task's last step. They are the window of a decision point's own families
  (`path_ahead`, `where_was_i`, `out_of_view`, `zone_clear`, `clearance_then`; no job asks one since 0.6.0) and what
  `stbench view` marks; no look family reads them.
- **Events come from the job, task by task.** Each wording of `JobSpec.events` is resolved by `events.resolve` over the
  delivered steps of each task of the window (a case's task numbering), as a system resolves it; no phase or wording is
  named here. A window of one task has the one.
- **The map is the public one** (`Inputs.map`, or `mapspec.load_map` of the script's layout): what a family reads of a
  room's polygon or a fixed piece, as the table systems read theirs.
- **Free-space states are computed once a keyframe and band** (`cached_states`), whichever families read them. A
  zone's states are observed with the zone's own ground and band (`World.zone_states`) and keep only the cells of the
  job's zones of that band, so a drone frame's tens of thousands of cells never sit in the cache.
- **What the robot's own log tells about is never asked** (`World.told`): the thing a task carries (the third of its
  `targets`) and every thing standing in the zone of any decision of the job's `zones`, asked or not (the response
  after the coin, and the subgoal that follows it, tell the zone was blocked and by what). `things` leaves them out, so
  no end question names one; a question during the job reads `told(upto=k)`, what the log had told by then.
- The timeline is read through its query helpers (`placement_of`, `decisions_in`, `task_of`, `keyframe`, `entities`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from . import colour_seen, freespace, grid, job_registry, look_windows, nearest_part, visibility, zones
from .camera import Camera
from .case import CasePaths, CaseRecord, case_clock, case_k, case_rows, case_task, script_k
from .catalog import item_size
from .descriptions import CONFUSED_KINDS, Key, desc_key
from .ego_frame import slant_m
from .events import resolve
from .geometry import Point, Rect, corners
from .job_registry import JobSpec, Relevance
from .mapspec import MapSpec, load_map
from .nearest_part import Part
from .pixels import Frame, ScriptPixels
from .recognise import recognisable
from .render_encode import read_png16
from .robots import Robot, from_doc
from .rules import MOVED_NO_M, STALE_M, widening
from .sightings import Sightings, kinds_of
from .timeline import DESCRIBABLE, Entity, Timeline, load

Cells = dict[tuple[int, int], str]
STANDING = ("light", "carried", "clutter", "mover")  # what can stand in a zone: not the map's fixed pieces or structure


@dataclass(frozen=True)
class Inputs:
    """What a compile reads of one script: `poses` and `vis` are the pose and visibility rows in script numbering, and
    `states_of(ks, floor_z=None, band=None)` the free-space cells of those keyframes (`freespace.script_states` over the
    script's files, behind `cached_states`): the robot's band on the base's floor, or a zone's ground and band;
    `map` is the layout's public map (None: `World.map` loads it from the timeline's layout); `points_of(k, id)` the
    (N, 3) map-frame points of the entity's pixels in keyframe `k` (`pixels.ScriptPixels` over the script's files; None:
    a compile of the ego families raises) and `gaps_of(k, id)` how many of its pixels there have no depth (None: none
    do); `depth_points_of(k)` the (N, 3) map-frame points of keyframe `k`'s whole depth image (None: a flier's
    `clear_path` raises), which `wall_seen` reads for the wall record; `frame_of(k)` the whole frame as points with the
    entity each shows (`pixels.Frame`; None: a ground robot's `clear_path` stopper screen cannot be read and raises)."""

    tl: Timeline
    cam: Camera
    robot: Robot
    job: JobSpec
    poses: list[dict]
    vis: list[dict]
    states_of: Callable[..., dict[int, Cells]]
    map: MapSpec | None = None
    points_of: Callable[[int, str], np.ndarray] | None = None
    gaps_of: Callable[[int, str], int] | None = None
    depth_points_of: Callable[[int], np.ndarray] | None = None
    frame_of: Callable[[int], Frame] | None = None


def cached_states(states_of: Callable[..., dict[int, Cells]]) -> Callable[..., dict[int, Cells]]:
    """`states_of` that computes a keyframe's cells once per band and hands the same dict to every later asker: a depth
    image is read once per compile whichever families read it. The cells of a keyframe are a pure function of the
    script's files, so an `Inputs` with another timeline or visibility log (the swap check's) may share it. The cache
    keys by (keyframe, floor, band); `floor_z` and `band` reach `states_of` only when given, so a function of the
    keyframes alone still serves the robot's band."""
    done: dict[tuple, Cells] = {}

    def states(ks: Iterable[int], floor_z: float | None = None, band: tuple[float, float] | None = None
               ) -> dict[int, Cells]:
        ks = list(ks)
        given = {name: v for name, v in (("floor_z", floor_z), ("band", band)) if v is not None}
        if missing := [k for k in dict.fromkeys(ks) if (k, floor_z, band) not in done]:
            done.update({(k, floor_z, band): cells for k, cells in states_of(missing, **given).items()})
        return {k: done[(k, floor_z, band)] for k in ks}

    return states


def load_inputs(case: CasePaths) -> Inputs:
    """The inputs of a case's script, from its files. A zone band's states keep only the fine cells of the job's zones
    of that ground and band (`zones.band_cells`); the robot's band is kept whole."""
    script = case.script
    tl = load(script.timeline)
    robot = from_doc(json.loads(script.robot.read_text(encoding="utf-8")))
    job, m = job_registry.get(tl.episode.job), load_map(tl.episode.map)
    keep = zones.band_cells(m, [*job.zones.values(), *job.slot_zones])

    def states(ks: Iterable[int], floor_z: float | None = None, band: tuple[float, float] | None = None
               ) -> dict[int, Cells]:
        if floor_z is None and band is None:
            return freespace.script_states(script, ks)
        return freespace.script_states(script, ks, floor_z, band, keep=keep.get((floor_z, band), frozenset()))

    pixels, cam, poses = ScriptPixels(script), Camera.load(script.camera), visibility.load_pose(script.pose)
    rows = {row["k"]: row for row in poses}

    def depth_points(k: int) -> np.ndarray:
        return grid.points(read_png16(script.depth(k)), cam, rows[k]["camera"]["pos"], rows[k]["camera"]["rot"])

    return Inputs(tl, cam, robot, job, poses, visibility.read(script.visibility), cached_states(states), m,
                  pixels.points, pixels.gaps, depth_points, pixels.frame)


def _overlaps(rect: Rect, polygon: tuple[Point, ...]) -> bool:
    """Whether the rectangle and the convex polygon share area (touching edges do not): no edge normal of either
    separates them."""
    a, b = corners(rect), list(polygon)
    for poly in (a, b):
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True):
            nx, ny = y1 - y0, x0 - x1
            pa, pb = [nx * x + ny * y for x, y in a], [nx * x + ny * y for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return False
    return True


class World:
    def __init__(self, inp: Inputs, rec: CaseRecord):
        self.inp, self.rec = inp, rec
        self.tl, self.cam, self.robot, self.job = inp.tl, inp.cam, inp.robot, inp.job
        window = [r for r in inp.vis if rec.k0 <= r["k"] < rec.k1]
        self.rows = {r["k"]: r for r in inp.poses if rec.k0 <= r["k"] < rec.k1}  # the delivered pose rows
        self.vis = {r["k"]: {e["id"]: e for e in r["entities"]} for r in window}
        self.sight = Sightings(window, self.cam, kinds_of(self.tl))
        self._recognised: dict[str, list[int]] = {}
        self._rendered: dict[str, tuple[frozenset[str], ...]] = {}
        self._told: tuple[dict[int, frozenset[str]], dict[int, frozenset[str]]] | None = None  # `told`'s, by keyframe
        self.decisions = self.tl.decisions_in(rec.k0, rec.k1)
        self.plan = self.tl.ego.tasks[rec.b].plan  # the window's last task: what `relevance` is of
        self.relevance = self.job.relevance(self.plan)
        cuts = {case_k(rec, d.k) for d in self.decisions}
        cuts |= {case_k(rec, self.tl.ego.tasks[i].k1 - 1) for i in range(rec.a, rec.b + 1)}
        self.windows = look_windows.windows(rec.k1 - rec.k0, cuts)  # case-relative [lo, hi)
        self.groups: dict[Key, list[Entity]] = {}
        for e in self.tl.entities:
            if e.mobility in DESCRIBABLE:
                self.groups.setdefault(desc_key(e.kind, e.attributes), []).append(e)

    # -- the stream ------------------------------------------------------------------------------------------
    def window_ks(self, k: int) -> range:
        """The script keyframes of the decision window that ends at script keyframe `k` (a decision point, a task's last
        step, or `look_windows.LOOK_WINDOW` steps after the last cut); ValueError when none ends there. A look point
        is any frame and ends none: its families read `look_windows.recent`."""
        end = case_k(self.rec, k) + 1
        for lo, hi in self.windows:
            if hi == end:
                return range(script_k(self.rec, lo), script_k(self.rec, hi))
        raise ValueError(f"no decision window ends at keyframe {k}: windows are cut only after a decision point, a "
                         f"task's last step or `look_windows.LOOK_WINDOW` steps")

    def window_of(self, k: int) -> range:
        """The script keyframes of the decision window that holds script keyframe `k` (whether or not it ends there)."""
        at = case_k(self.rec, k)
        for lo, hi in self.windows:
            if lo <= at < hi:
                return range(script_k(self.rec, lo), script_k(self.rec, hi))
        raise ValueError(f"no decision window holds keyframe {k}: it lies outside the window")

    @property
    def end_window(self) -> range:
        """The case window's last `look_windows.LOOK_WINDOW` frames (all of them when the window is shorter): the view
        just before the end questions."""
        return range(max(self.rec.k0, self.rec.k1 - look_windows.LOOK_WINDOW), self.rec.k1)

    def recent(self, k: int) -> range:
        """The recent view at script keyframe `k` (`look_windows.recent`): the last `look_windows.LOOK_WINDOW` frames
        of the task that holds `k`, up to and including it."""
        return look_windows.recent(self.tl.task_of(k).k0, k)

    def base(self, k: int) -> tuple[float, float]:
        """The delivered base position (x, y) at script keyframe k."""
        return tuple(self.rows[k]["pose"]["pos"][:2])

    def yaw(self, k: int) -> float:
        """The delivered heading at script keyframe k, in radians, from the row's `yaw_deg`."""
        return math.radians(self.rows[k]["pose"]["yaw_deg"])

    def latest_cells(self, ks: Iterable[int]) -> dict[tuple[int, int], tuple[str, int]]:
        """`grid.latest` over the free-space cells of the keyframes (each keyframe's cells computed once per
        `Inputs`)."""
        return grid.latest(self.inp.states_of(ks))

    def zone_states(self, ks: Iterable[int], zone: zones.Zone) -> dict[tuple[int, int], tuple[str, int]]:
        """`grid.latest` over the free-space cells of the keyframes observed with the zone's own ground and band (each
        keyframe's cells computed once per band): what `zones.zone_state` and `zones.coverage` take."""
        return grid.latest(self.inp.states_of(ks, zone.base_z, zone.band))

    # -- the window's tasks ----------------------------------------------------------------------------------
    @cached_property
    def tasks(self) -> list[tuple[int, int, int]]:
        """(case task index, script k0, script k1) of each task of the window, in order."""
        return [(case_task(self.rec, t.index), t.k0, t.k1) for t in self.tl.ego.tasks[self.rec.a:self.rec.b + 1]]

    @property
    def multi(self) -> bool:
        """The window holds more than one task."""
        return len(self.tasks) > 1

    def task_index(self, k: int) -> int:
        """The case task index of the task that holds script keyframe `k`."""
        return case_task(self.rec, self.tl.task_of(k).index)

    def relevance_of(self, i: int) -> Relevance:
        """The job's relevance of case task `i`'s plan (`relevance` is the last task's)."""
        return self.job.relevance(self.tl.ego.tasks[self.rec.a + i].plan)

    def camera_at(self, k: int) -> tuple[tuple[float, float, float], float]:
        """The camera's position (x, y, z) and heading in degrees at script keyframe `k`, from the delivered pose row:
        its `camera.pos` (the mount turned by the heading and added to the base, so a drone's hangs below it) and the
        base's `pose.yaw_deg`, which the camera shares. The frame of an `answers.Offset` (`ego_frame`)."""
        row = self.rows[k]
        return tuple(row["camera"]["pos"][:3]), float(row["pose"]["yaw_deg"])

    def middle(self, e: Entity, k: int) -> tuple[float, float, float] | None:
        """The middle of the entity's bounding box in the map at script keyframe `k`: the placement's base centre (x, y)
        and half the kind's height above its base. None when the entity does not exist then."""
        p = self.tl.placement_of(e.id, self.tl.keyframe(k).t)
        if p is None:
            return None
        return (float(p.pos[0]), float(p.pos[1]), float(p.pos[2]) + e.size[2] / 2)

    def slant(self, e: Entity, k: int) -> float | None:
        """The 3-D distance in metres from the delivered camera at keyframe `k` to the middle of the entity there: the
        range `rules.relative_ladder` scales with. None when the entity does not exist then."""
        if (mid := self.middle(e, k)) is None:
            return None
        return slant_m(self.camera_at(k)[0], mid)

    def nearest_part(self, entity: str, ks: Iterable[int], ground_xy: tuple[float, float]) -> Part | None:
        """The gold of a look (`nearest_part.nearest_part`): the point of the union of the floor hulls of the entity's
        pixels in the script keyframes `ks`, each seen from that keyframe's delivered camera, nearest `ground_xy`. None
        when it has no pixel with a depth in any of them. ValueError for a keyframe outside the window, or for inputs
        built without the pictures (`Inputs.points_of`)."""
        ks = list(ks)
        if self.inp.points_of is None:
            raise ValueError("these inputs carry no pictures (Inputs.points_of), so no nearest part can be read")
        if outside := [k for k in ks if k not in self.rows]:
            raise ValueError(f"keyframe {outside[0]} lies outside the window [{self.rec.k0}, {self.rec.k1})")
        return nearest_part.nearest_part([(self.camera_at(k)[0], self.inp.points_of(k, entity)) for k in ks], ground_xy)

    def contact_error(self, entity: str, ks: Iterable[int], xy: tuple[float, float], bases: Iterable[float | None]
                      ) -> float | None:
        """The contact tie's one-pixel error at the gold point `xy` (`nearest_part.contact_error`): the entity's pixels
        in the script keyframes `ks`, each seen from its delivered camera, against the base z it rests on at that
        keyframe (`bases`, None for a thing with none: `ego_support.contact_bases`), at the job's clearance. None when
        no keyframe shows a contact pixel. Raises as `nearest_part` does."""
        ks = list(ks)
        if self.inp.points_of is None:
            raise ValueError("these inputs carry no pictures (Inputs.points_of), so no contact can be read")
        return nearest_part.contact_error([(self.camera_at(k)[0], self.inp.points_of(k, entity), base)
                                           for k, base in zip(ks, bases, strict=True)], xy, self.cam.fy,
                                          self.job.clearance_m)

    def gaps(self, entity: str, k: int) -> int:
        """How many pixels of the entity in script keyframe `k` have no depth (`Inputs.gaps_of`; 0 without it): a part
        read from the pixels that have one is missing them."""
        return 0 if self.inp.gaps_of is None else self.inp.gaps_of(k, entity)

    def clock(self, k: int) -> str:
        """The case's clock at script keyframe `k`, as the stream delivers it."""
        return case_clock(self.rec, self.tl.keyframe(k).t)

    @cached_property
    def events(self) -> dict[tuple[int, str], int]:
        """For every task `i` of the window (case numbering) and every wording of the job's events that resolves there,
        the script keyframe: `events.resolve` over the delivered steps of task `i`, as a system resolves it. A pair that
        does not resolve (the task never has the phase) is absent."""
        steps: dict[int, list[dict]] = {}
        for row in case_rows(self.rec, self.tl, self.inp.poses):
            steps.setdefault(row["status"]["task"], []).append(row)
        found = {(i, wording): resolve(rule, steps.get(i, [])) for i, _, _ in self.tasks
                 for wording, rule in self.job.events.items()}
        return {pair: script_k(self.rec, k) for pair, k in found.items() if k is not None}

    @cached_property
    def map(self) -> MapSpec:
        """The layout's public map: the inputs' own, else `mapspec.load_map` of the timeline's layout (which finds a
        held-out map as `map_files` does)."""
        return self.inp.map if self.inp.map is not None else load_map(self.tl.episode.map)

    # -- things ----------------------------------------------------------------------------------------------
    def unique(self, e: Entity) -> bool:
        """Its description (kind and describable attributes) is the only one of its kind in the world."""
        return len(self.groups[desc_key(e.kind, e.attributes)]) == 1

    def things(self, relevance: Relevance | None = None) -> list[Entity]:
        """What an end question may ask about: carried things of the task's relevance kinds (`relevance`: the last
        task's when not given) whose description is unique among the world's describable things and that the robot's
        own log does not tell about (`told`). Light furniture and clutter are never asked."""
        kinds, told = (self.relevance if relevance is None else relevance).kinds, self.told()
        return [e for e in self.tl.entities
                if e.mobility == "carried" and e.kind in kinds and self.unique(e) and e.id not in told]

    def told(self, upto: int | None = None) -> frozenset[str]:
        """The entities the robot's own log tells about, which no question names (they would be answered from the log,
        not from memory of the world): every entity named by the third `target` of a task of the window (the helper's
        carried thing: its subgoal names it, where it came from and where it went), and every thing whose footprint
        overlaps the zone of a decision of the window, at that decision's time (the blockers: the response after a
        coin and the subgoal that follows it tell the zone was blocked). A decision counts when its kind has a zone in
        `JobSpec.zones`, whether or not a family is asked there (none is, from 0.6.0: the coin still decides the
        response, so its blocker is told). The zone is the one the map and the delivered pose give; all the window's
        decisions count, asked or not.

        `upto` (a script keyframe) is what the log had told by then, for a question asked during the job: only the
        tasks that had begun (`Task.k0 <= upto`, since their subgoals name the carried thing from the first step) and
        the decisions at or before it (`DecisionPoint.k <= upto`). An end question asks after the window, with no
        `upto`: everything the window told."""
        if self._told is None:
            known = {e.id for e in self.tl.entities}
            self._told = (
                {t.k0: frozenset({t.targets[2]} & known) if len(t.targets) > 2 else frozenset()
                 for t in self.tl.ego.tasks[self.rec.a:self.rec.b + 1]},
                {d.k: self._standing(d.k, self.job.zones[d.kind])
                 for d in self.decisions if d.kind in self.job.zones})
        carried, standing = self._told
        return frozenset().union(*(ids for k, ids in (*carried.items(), *standing.items())
                                   if upto is None or k <= upto))

    def _standing(self, k: int, rule: zones.ZoneRule) -> frozenset[str]:
        """The ids of the things whose footprint overlaps the rule's zone at the decision at script keyframe `k`."""
        if (zone := zones.zone_of(self.map, rule, self.base(k), self.yaw(k))) is None:
            return frozenset()
        t = self.tl.keyframe(k).t
        return frozenset(e.id for e in self.tl.entities if e.mobility in STANDING
                         and (rect := self.tl.footprint(e.id, t)) is not None and _overlaps(rect, zone.polygon))

    @cached_property
    def sighted_looks(self) -> dict[str, frozenset[str]]:
        """The ids of the things with a sighting in the window (`sighted`), by kind: the describable entities, and the
        clutter of a kind some `descriptions.CONFUSABLE` group holds (a clothes pile, a book stack: never asked, but a
        picture shows one for a toy or a book). What the look-alike screens of `questions_where` read, and no more (no
        place, no time, no gold)."""
        found: dict[str, set[str]] = {}
        for e in self.tl.entities:
            named = e.mobility in DESCRIBABLE or (e.mobility == "clutter" and e.kind in CONFUSED_KINDS)
            if named and self.sighted(e.id):
                found.setdefault(e.kind, set()).add(e.id)
        return {kind: frozenset(ids) for kind, ids in found.items()}

    def in_window(self, entity: str) -> bool:
        """The entity has a visibility row (at least one pixel) at some keyframe of the window."""
        return any(entity in rows for rows in self.vis.values())

    def in_ks(self, entity: str, ks: Iterable[int]) -> bool:
        """The entity has a visibility row, at any pixel count, at one of the script keyframes `ks`."""
        return any(entity in self.vis.get(k, ()) for k in ks)

    def in_end_window(self, entity: str) -> bool:
        """The entity has a visibility row, at any pixel count, in the case window's last look window."""
        return self.in_ks(entity, self.end_window)

    def xy_at(self, entity: str, k: int) -> tuple[float, float] | None:
        """The placement's (x, y), the base centre and so the bounding box's, at script keyframe k, to 4 decimals; None
        when the entity does not exist then."""
        p = self.tl.placement_of(entity, self.tl.keyframe(k).t)
        return None if p is None else (round(float(p.pos[0]), 4), round(float(p.pos[1]), 4))

    def on_floor(self, entity: str, k: int) -> bool:
        """The placement's base height at script keyframe `k` is within the robot's `climb_m` of the delivered base
        height there: the thing stands on the floor the robot drives on. False when it does not exist then."""
        p = self.tl.placement_of(entity, self.tl.keyframe(k).t)
        return p is not None and abs(p.pos[2] - self.rows[k]["pose"]["pos"][2]) <= self.robot.body.climb_m + grid.EPS

    # -- sightings -------------------------------------------------------------------------------------------
    def sighted(self, entity: str) -> list[int]:
        """The window keyframes where the entity counted as a sighting (its kind's glimpse bound or more), sorted."""
        return sorted(self.sight.seen(entity) + self.sight.band(entity))

    def recognised(self, entity: str) -> list[int]:
        """The window keyframes with a recognisable sighting of the entity, sorted: its record of that one frame passes
        `recognise.recognisable` (enough pixels and light, a bounding box clear of the image's edge, a depth)."""
        if entity not in self._recognised:
            self._recognised[entity] = [k for k in self.sighted(entity)
                                        if recognisable(self.vis[k][entity], self.cam, f"{entity} at keyframe {k}")]
        return self._recognised[entity]

    def rendered(self, entity: str) -> tuple[frozenset[str], ...]:
        """What the picture shows of the entity's colour, one set a recognisable sighting: the words that frame's
        `rgb_med` may be taken for (`colour_seen.classify`: one word when sure, else the words of the boundary it is
        near). Empty when nothing was measured: an older root's visibility log, a thing never recognisable, or a label
        that is no palette word."""
        if entity not in self._rendered:
            palette = self.tl.entity(entity).attributes.get("color") in colour_seen.WORDS
            self._rendered[entity] = tuple(
                colour_seen.classify(rgb) for k in self.recognised(entity)
                if palette and (rgb := self.vis[k][entity].get("rgb_med")) is not None)
        return self._rendered[entity]

    def may_be_taken_for(self, entity: str, colour: object) -> bool:
        """Whether a viewer of the entity's sightings could take it for something of the colour word `colour`: the
        colour any of its recognisable frames shows it in, or sits next to a boundary of (`colour_seen.could_be`); the
        three neutrals (black, grey, white) are one group, so a thing that may be taken for any of them hides every
        neutral. A thing with nothing measured (`rendered`) is taken for its label, a neutral label for all three
        neutrals: the screen of an older root's visibility log."""
        classes = self.rendered(entity)
        if classes:
            return colour_seen.could_be(colour_seen.taken_for(classes), colour)
        return colour_seen.could_be(colour_seen.label_reads(self.tl.entity(entity).attributes.get("color")), colour)

    def colour_holds(self, entity: str) -> bool:
        """The picture bears out the entity's colour word: the colour most of its recognisable frames confidently show
        it in (`colour_seen.held`) is its label. True when nothing was measured (`rendered`): the label stands."""
        classes = self.rendered(entity)
        return not classes or colour_seen.held(classes) == self.tl.entity(entity).attributes.get("color")

    def _places(self, entity: str, ks: Iterable[int]) -> list[tuple[float, float]]:
        """The placement's (x, y) at each of the keyframes `ks` where the entity exists."""
        return [xy for k in ks if (xy := self.xy_at(entity, k)) is not None]

    def _far(self, e: Entity) -> float:
        """How far a recognisable sighting must lie from a place to show the thing was elsewhere: `STALE_M` plus the
        kind's widening (`rules.widening`: a large thing's extent)."""
        return STALE_M + widening(item_size(e.kind)) - grid.EPS

    def stale(self, e: Entity, k: int, xy: tuple[float, float]) -> bool:
        """The place `xy` that the entity `e` had at keyframe `k` is out of date: some recognisable sighting of it after
        `k` lies at least `STALE_M` plus the kind's widening from `xy`."""
        far = self._far(e)
        later = (j for j in self.recognised(e.id) if j > k)
        return any(math.dist(p, xy) >= far for p in self._places(e.id, later))

    def arrived(self, e: Entity, k: int, xy: tuple[float, float]) -> bool:
        """The entity `e` came to the place `xy` it had at keyframe `k` from elsewhere (`stale`'s mirror): some
        recognisable sighting of it before `k` lies at least `STALE_M` plus the kind's widening from `xy`. A thing at
        such a place is at its latest state after a move."""
        far = self._far(e)
        earlier = (j for j in self.recognised(e.id) if j < k)
        return any(math.dist(p, xy) >= far for p in self._places(e.id, earlier))

    def steady(self, e: Entity, k: int, xy: tuple[float, float], until: int, within: float = MOVED_NO_M) -> bool:
        """Every sighting of the entity `e` (any at its kind's glimpse bound or more, recognisable or not) from keyframe
        `k` up to keyframe `until`, both included, lies within `within` (`MOVED_NO_M`; the ego families pass
        `UNCHANGED_M`) of `xy`: nothing seen in that stretch shows it anywhere else."""
        near = within - grid.EPS
        stretch = (j for j in self.sighted(e.id) if k <= j <= until)
        return all(math.dist(p, xy) < near for p in self._places(e.id, stretch))


def scope_of(w: World, k: int, evidence: Iterable[int]) -> str:
    """The scope an `"evidence"` family's question has: `now` exactly when the evidence is non-empty and every evidence
    keyframe is `k` itself, the latest frame (what a system holds when it is asked there; a system gets one frame at a
    time), else `memory`."""
    keyframes = list(evidence)
    return "now" if keyframes and all(e == k for e in keyframes) else "memory"
