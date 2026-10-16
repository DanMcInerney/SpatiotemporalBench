"""The people overlay: a catalog's people over a finished script (docs/people.md; M2's lane P1, widened by M3-people).

People are an overlay on the finished timeline, not a call before the path as the other world processes are: a person
must keep clear of the robot's path, which exists only after the job's script. `generate.generate_script` ends with
`add`, and `validate_ego` falls back to `person_kind` when no catalog has a mover's kind, so `validate` reads a person's
box and camera clearance as it reads a mover's. A catalog's people are the module `people_<catalog>` (`people_home`),
which declares `KIND`, `MODEL`, `BOX_M`, `COUNT`, `P_HOME`, `SPEED_MPS`, `DWELL_S`, `SAME_ROOM`, `CAMERA_M`,
`ROBOT_KEEP_M` and `START_ROBOT_M`, and may declare `WALK_ON`, `KEEP_OFF` and `GRID_M`; a catalog with no such module
has no people.

A person exists only during the tasks it is present for (`presence`, drawn first and alone). Its walk in a task is
`mover_walk.Walker` on the free floor of what stands through the task and the gaps around it (`_obstacles`), kept clear
of the robot's body, of the body's sweep between keyframes and of the camera: the robot never reacts to a person (its
path is planned first), so the person keeps off the robot. A person arrives and leaves at a whole second inside a
gap, `changes.EDGE_BUFFER` from both its ends, so it never appears or vanishes in view; in the script's first task it
is there from t = 0. Several people (`COUNT`) are drawn one after another, each treating the placements of the ones
before it as obstacles.

What a catalog may add to the home's flat floor, each a no-op where its module declares none:
- `WALK_ON`, fixed-piece kinds people walk on (a sidewalk, a plaza). A person then stands at their one height and keeps
  its whole footprint on them (the union of their footprints, pieces within `SEAM_M` of each other one surface), so a
  body never overhangs a kerb; a fixed piece whose top is at most that height is ground, never an obstacle, and a thing
  on the floor obstructs when its base is within `FLOOR_ITEM_Z` of it (a cart on the plaza does, a car in the roadway
  below does not). Walk pieces are axis-aligned and of one height (`ValueError` naming the map otherwise);
- `KEEP_OFF`, fixed-piece kinds a walk never enters (a robot's zones: a person standing in one would turn the coin that
  balances the zone's answers);
- `GRID_M`, the walk grid's cell (the downtown's is 0.1 m over 100 x 96 m).
Where the robot flies over a person (`robots.flies_over`, at the person's own height, its floor included) nothing of it
keeps the person off at that keyframe; elsewhere the person keeps `max(ROBOT_KEEP_M, robot clearance + 0.01)` from the
robot's body.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from types import ModuleType

import numpy as np

from . import robots
from .catalog import CATALOGS, FLOOR_ITEM_Z, MoverKind
from .changes import EDGE_BUFFER, when_in_gap
from .geometry import Point, Rect, corners, dist_to_segment, rect_distance, rects_overlap
from .ledger import Ledger
from .mapspec import MapSpec
from .mover_walk import Walker
from .route import FreeSpace
from .timeline import Entity, Timeline
from .validate import SURFACE_Z_TOL, UNDER_TOL, collision_class, door_entities, door_leaf_rect, non_blocking

# Draws of one person's walk in one task before it is not present for that task. The walk is drawn again, from a new
# start, when the robot's body or camera comes up to a person that is waiting or boxed in.
ATTEMPTS = 1000
GRID_RES = 0.05  # the free grid's cell: an 0.8 m door leaves 0.14 m of free width at 0.33 m of clearance, three cells
CLEARANCE_PAD_M = 0.05  # the grid keeps half the box's width plus this from every wall and obstacle
SEAM_M = 0.01  # walk pieces this far apart are one surface (M3-maps insets neighbouring pieces by 0.005 m)
CLEARANCE_MARGIN_M = 0.01  # the robot's own clearance from a person is kept this far over, for `validate`'s sampling
Pose = tuple[Point, float]  # floor position, yaw
Ground = tuple[float, list[Rect]]  # the height people stand at; the rectangles a walk never enters


@dataclass(frozen=True)
class Report:
    """What an overlay did, for the lane's measures: the draws each walk took (one per task a person was drawn present
    for and found a walk in), and the tasks it was drawn present for and found none in `ATTEMPTS` draws (not there)."""

    attempts: tuple[int, ...] = ()
    dropped: int = 0


def add(tl: Timeline, m: MapSpec, rng: np.random.Generator) -> Timeline:
    """The catalog's people over a finished script: new entities (ids after every other) with placements during the
    tasks they are present for, kept clear of the robot, the furniture, the things on the floor and each other. A
    catalog with no people module, or a script with no task, comes back unchanged; a person that finds no walk in
    `ATTEMPTS` draws is not there for that task (`overlay` counts them). Raises ValueError for a gap with no moment 30 s
    from both ends where a person must arrive or leave, and for a map whose walk pieces are not one height or not
    axis-aligned."""
    return overlay(tl, m, rng)[0]


def overlay(tl: Timeline, m: MapSpec, rng: np.random.Generator) -> tuple[Timeline, Report]:
    """`add` and its `Report`. `rng` is split into one stream a person, and each of those into one for its presence
    and one a task, so a task's walk does not depend on how many tasks the script has."""
    mod = _module(m.catalog)
    tasks = tl.ego.tasks
    if mod is None or not tasks:
        return tl, Report()
    kind = _kind(mod)
    robot = robots.ROBOTS[tl.episode.robot]
    ground = _ground(m, mod)
    first_id = 1 + max(int(e.id[1:]) for e in tl.entities)
    attempts: list[int] = []
    dropped = 0
    for r, person in enumerate(rng.spawn(mod.COUNT)):
        eid = f"E{first_id + r:04d}"
        presence_rng, *task_rngs = person.spawn(1 + len(tasks))
        wants = presence(presence_rng, len(tasks), mod.P_HOME)
        ledger, here = Ledger(), False
        for i, task in enumerate(tasks):
            times = [tl.keyframe(k).t for k in range(task.k0, task.k1)]
            lo = tl.keyframe(tasks[i - 1].k1 - 1).t if i else None
            moment = _moment(task_rngs[i], lo, times[0], f"task {i}") if i and (wants[i] or here) else 0.0
            poses = None
            if wants[i]:
                t_lo = 0.0 if lo is None else lo
                t_hi = tl.keyframe(tasks[i + 1].k0).t if i + 1 < len(tasks) else math.inf
                poses, n = _walk(tl, m, mod, kind, robot, task.k0, task.k1,
                                 _obstacles(tl, m, t_lo, t_hi, ground[0]), task_rngs[i], ground)
                if poses is None:
                    dropped += 1
                else:
                    attempts.append(n)
            if poses is not None:
                _place(ledger, eid, poses, times, moment, ground[0])
                here = True
            elif here:
                ledger.remove(eid, moment)
                here = False
        placements = ledger.finish()[0]
        if placements:
            person_entity = Entity(eid, mod.KIND, {}, "mover", mod.BOX_M, asset=mod.MODEL)
            tl = replace(tl, entities=[*tl.entities, person_entity]).with_placements({eid: placements})
    return tl, Report(tuple(attempts), dropped)


def presence(rng: np.random.Generator, n_tasks: int, p_home: float) -> list[bool]:
    """Whether a person is present for each of `n_tasks` tasks: independently, with probability `p_home`."""
    return [bool(u < p_home) for u in rng.random(n_tasks)]


def person_kind(name: str) -> MoverKind:
    """The kind record (box, speeds, dwell, camera clearance) of a person kind of any catalog; KeyError when none. Every
    people module of the kind shares its camera clearance, so the first catalog's record serves `validate`; a mover's
    own box is its entity's."""
    for catalog in sorted(CATALOGS):
        mod = _module(catalog)
        if mod is not None and name == mod.KIND:
            return _kind(mod)
    raise KeyError(name)


def without_people(tl: Timeline) -> Timeline:
    """`tl` less the overlay: its person entities and their placements, which is the script's world as the generator's
    steps before `add` made it (the probe compares worlds this way)."""
    ids = []
    for e in tl.entities:
        if e.mobility == "mover":
            try:
                person_kind(e.kind)
            except KeyError:
                continue
            ids.append(e.id)
    if not ids:
        return tl
    kept = replace(tl, entities=[e for e in tl.entities if e.id not in ids])
    return kept.with_placements({eid: [] for eid in ids})


def walk_height(m: MapSpec, mod: ModuleType) -> float:
    """The height people stand at: the one top of the fixed pieces of the module's `WALK_ON` kinds, 0 where it declares
    none. Raises ValueError naming the map when it has no such piece or their tops differ by more than
    `validate.SURFACE_Z_TOL` (a body stands at one height)."""
    kinds = getattr(mod, "WALK_ON", ())
    if not kinds:
        return 0.0
    tops = [f.size[2] for f in _pieces(m, kinds)]
    if not tops:
        raise ValueError(f"map {m.name!r} has no fixed piece of the kinds {kinds} that its people walk on")
    if max(tops) - min(tops) > SURFACE_Z_TOL:
        raise ValueError(f"map {m.name!r}: the pieces {kinds} that people walk on stand from {min(tops):g} to "
                         f"{max(tops):g} m: a person needs one height (to {SURFACE_Z_TOL:g} m)")
    return max(tops)


def _pieces(m: MapSpec, kinds) -> list:
    return [f for f in m.furniture if f.mobility == "fixed" and f.kind in kinds]


def _ground(m: MapSpec, mod: ModuleType) -> Ground:
    """The height people stand at, and what a walk never enters whatever stands: the map off its walk pieces and the
    footprints of the module's `KEEP_OFF` kinds (the free grid keeps the walker's clearance from each)."""
    walk = getattr(mod, "WALK_ON", ())
    off = _off_walk(m, walk) if walk else []
    return walk_height(m, mod), off + [f.footprint() for f in _pieces(m, getattr(mod, "KEEP_OFF", ()))]


def _off_walk(m: MapSpec, kinds) -> list[Rect]:
    """Rectangles that cover the map except its walk pieces, each grown by `SEAM_M` so that neighbours with a seam are
    one surface. As obstacles of the free grid they keep every free cell's whole footprint on the pieces, exactly: by
    their distance, and by the overlap test each step makes. The pieces must be axis-aligned (a quarter turn), which
    reduces the complement to rectangles by a sweep over the pieces' edges; ValueError names the map otherwise."""
    boxes = []
    for f in _pieces(m, kinds):
        turns = f.yaw / (math.pi / 2)
        if abs(turns - round(turns)) > 1e-6:
            raise ValueError(f"map {m.name!r}: the walk piece {f.id} is turned {f.yaw:g} rad, not by quarter turns")
        xs, ys = zip(*corners(f.footprint()), strict=True)
        boxes.append((min(xs) - SEAM_M, min(ys) - SEAM_M, max(xs) + SEAM_M, max(ys) + SEAM_M))
    x0, y0, x1, y1 = m.bounds
    cuts_x = sorted({round(v, 6) for b in boxes for v in (b[0], b[2])} | {x0 - 1.0, x1 + 1.0})
    cuts_y = sorted({round(v, 6) for b in boxes for v in (b[1], b[3])} | {y0 - 1.0, y1 + 1.0})
    rows = []  # for each strip between two y cuts: the runs of x cuts no walk piece covers
    for ya, yb in zip(cuts_y, cuts_y[1:], strict=False):
        runs, start = [], None
        for xa, xb in zip(cuts_x, cuts_x[1:], strict=False):
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            covered = any(b[0] <= mx <= b[2] and b[1] <= my <= b[3] for b in boxes)
            if start is None and not covered:
                start = xa
            elif start is not None and covered:
                runs.append((start, xa))
                start = None
        rows.append((ya, runs + ([(start, cuts_x[-1])] if start is not None else [])))
    out: list[Rect] = []
    live: dict[tuple[float, float], float] = {}  # a run of x -> where its rectangle starts in y (repeated runs join)
    for ya, runs in [*rows, (cuts_y[-1], [])]:
        for run in [r for r in live if r not in runs]:
            lo = live.pop(run)
            out.append(Rect((run[0] + run[1]) / 2, (lo + ya) / 2, run[1] - run[0], ya - lo, 0.0))
        for run in runs:
            live.setdefault(run, ya)
    return out


def _kind(mod: ModuleType) -> MoverKind:
    return MoverKind(mod.KIND, mod.BOX_M, (), mod.SPEED_MPS, mod.DWELL_S, mod.SAME_ROOM, mod.CAMERA_M)


def _module(catalog: str) -> ModuleType | None:
    """The catalog's people module, imported on first use; None when the catalog has none."""
    name = f"{__package__}.people_{catalog}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as err:
        if err.name == name:
            return None
        raise


def _moment(rng: np.random.Generator, lo: float | None, hi: float, what: str) -> float:
    """A whole second strictly inside the gap (lo, hi) before `what`, at least `EDGE_BUFFER` from both ends (`lo`
    rounded up first: `when_in_gap` rounds it down, which can fall a fraction of a second short of the buffer)."""
    start = math.ceil(lo)
    if hi - start < 2 * EDGE_BUFFER:
        raise ValueError(f"the gap before {what}, ({lo}, {hi}) s, has no moment {EDGE_BUFFER:g} s from both its ends")
    return when_in_gap(rng, float(start), hi)


def _obstacles(tl: Timeline, m: MapSpec, t_lo: float, t_hi: float, z0: float = 0.0) -> list[Rect]:
    """What a person standing at height `z0` may not overlap at some time in [t_lo, t_hi], the span from the previous
    task's last keyframe to the next task's first: its first and last poses of a task stand through the gaps around it,
    through every change the world makes in them. Blocking fixed furniture that stands above `z0` (a piece whose top is
    at most `z0`, to `UNDER_TOL`, is the ground), every placement in that span of light furniture, of a carried item or
    clutter whose base is within `FLOOR_ITEM_Z` of `z0` and of another mover, and each door leaf in every state the
    door has then (the entrance's shut leaf keeps a person out of the doorway)."""
    rects = [f.footprint() for f in m.furniture if f.mobility == "fixed" and f.blocks and f.size[2] > z0 + UNDER_TOL]
    for e in tl.entities:
        cls = collision_class(e)
        if cls not in ("light", "carried", "mover") or non_blocking(e):
            continue
        for p in tl.placements_of(e.id):
            on_floor = cls != "carried" or abs(p.pos[2] - z0) < FLOOR_ITEM_Z
            if p.t0 < t_hi and (p.t1 is None or p.t1 > t_lo) and on_floor:
                rects.append(Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw))
    for door_id, eid in door_entities(tl, m).items():
        opening, thickness = next(o for o in m.doors() if o.id == door_id), tl.entity(eid).size[1]
        states = {bool(v) for v in tl.values_between(eid, "open", t_lo, t_hi)} or {False, True}
        rects += [door_leaf_rect(m, opening, is_open, thickness) for is_open in sorted(states)]
    return rects


def _walk(tl: Timeline, m: MapSpec, mod: ModuleType, kind: MoverKind, robot: robots.Robot, k0: int, k1: int,
          obstacles: list[Rect], rng: np.random.Generator, ground: Ground | None = None
          ) -> tuple[list[Pose] | None, int]:
    """One person's pose at each keyframe of a task, and the draws it took; None when `ATTEMPTS` draws found no walk.
    A step is not taken (the person waits and the walk resumes from there) when the new pose overlaps an obstacle, comes
    within `max(ROBOT_KEEP_M, robot clearance + CLEARANCE_MARGIN_M)` of the robot's body at the keyframe or of the body
    swept to the keyframes before and after it, or puts the person's centre within `CAMERA_M` of the camera. At a
    keyframe where the robot flies over the person (`robots.flies_over`, the person's floor included) only the obstacles
    count. If the pose it waits in is itself too near by then, the whole walk is drawn again, from a new start.
    `ground` is `_ground(m, mod)` (given by `overlay`, which draws it once)."""
    z0, never = ground if ground is not None else _ground(m, mod)
    kfs = [tl.keyframe(k) for k in range(k0, k1)]
    times, bases, cams = [kf.t for kf in kfs], [kf.base[:2] for kf in kfs], [kf.pos[:2] for kf in kfs]
    body = robot.body
    reach = body.length_m / 2 if body.shape == "disc" else math.hypot(body.length_m, body.width_m) / 2
    keep = max(mod.ROBOT_KEEP_M, body.clearance_m + CLEARANCE_MARGIN_M)
    over = [robots.flies_over(robots.over_m(robot, kf.base[2]), z0 + kind.size[2]) for kf in kfs]
    n = len(kfs)  # the robot's body at keyframe j, and swept from the keyframe before it and to the one after it
    sweeps = [[(bases[j], bases[j])] + ([(bases[j - 1], bases[j])] if j else [])
              + ([(bases[j], bases[j + 1])] if j + 1 < n else []) for j in range(n)]
    free = FreeSpace(m, [*obstacles, *never], kind.size[1] / 2 + CLEARANCE_PAD_M, getattr(mod, "GRID_M", GRID_RES))
    walker = Walker(kind, free, rng)
    radius = math.hypot(kind.size[0], kind.size[1]) / 2
    boxes = np.array([(r.cx, r.cy, r.sx / 2, r.sy / 2, math.cos(r.yaw), math.sin(r.yaw)) for r in free.rects])

    def near(pos: Point) -> np.ndarray:
        """The obstacles within the body's circumscribed radius of `pos`: the only ones its footprint can overlap."""
        dx, dy = pos[0] - boxes[:, 0], pos[1] - boxes[:, 1]
        lx, ly = boxes[:, 4] * dx + boxes[:, 5] * dy, boxes[:, 4] * dy - boxes[:, 5] * dx
        gap = np.hypot(np.maximum(np.abs(lx) - boxes[:, 2], 0.0), np.maximum(np.abs(ly) - boxes[:, 3], 0.0))
        return np.flatnonzero(gap <= radius)

    def clear(j: int, pos: Point, yaw: float) -> bool:
        if not over[j] and math.dist(pos, cams[j]) < kind.camera_m:
            return False
        foot = Rect(pos[0], pos[1], kind.size[0], kind.size[1], yaw)
        if any(rects_overlap(foot, free.rects[i]) for i in near(pos)):
            return False
        if over[j]:
            return True
        for a, b in sweeps[j]:
            if dist_to_segment(pos[0], pos[1], a, b) - radius - reach >= keep:
                continue  # too far apart for the outlines to come within the margin
            mid, heading = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), math.atan2(b[1] - a[1], b[0] - a[0])
            line = Rect(mid[0], mid[1], math.dist(a, b), 0.0, heading)  # the sweep as a rectangle with no width
            if rect_distance(foot, line) - reach < keep:
                return False
        return True

    keepout = [(bases[0], mod.START_ROBOT_M)]
    for attempt in range(1, ATTEMPTS + 1):
        st = walker.start(keepout)
        if st is None or not clear(0, st.pos, st.yaw):
            continue
        out = [(st.pos, st.yaw)]
        for j in range(1, n):
            nxt = walker.advance(st, times[j] - times[j - 1])
            if not clear(j, nxt.pos, nxt.yaw):
                nxt = st
                if not clear(j, st.pos, st.yaw):
                    break
            st = nxt
            out.append((st.pos, st.yaw))
        else:
            return out, attempt
    return None, ATTEMPTS


def _place(ledger: Ledger, eid: str, poses: list[Pose], times: list[float], first: float, z0: float = 0.0) -> None:
    """A placement at each keyframe whose pose differs from the one before (a waiting person keeps its placement), at
    height `z0`. The first starts at `first`, a whole second inside the gap before the task (0 in the first task), and
    closes the previous task's last."""
    last: Pose | None = None
    for j, pose in enumerate(poses):
        if j == 0 or pose != last:
            ledger.place(eid, first if j == 0 else times[j], (pose[0][0], pose[0][1], z0), pose[1], None)
        last = pose
