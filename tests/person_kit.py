"""What the people tests share (`closest_person`, `person_mem`, the person evidence rule): `ego_kit`'s one-task vacuum
script with people put in by hand, so every distance is worked out in the test.

The look point is `ego_kit.K` = 13 (the look window 6-13 ends there; the other look windows are 0-5, 14-18, 19-21,
22-27). The camera is mounted 0.13 m ahead of the base and 0.08 m up: at keyframe 13 the base is (3.0, 1.0) heading
north, so the camera is at (3.0, 1.13, 0.08); at keyframe 2 the base is the kitchen's centre (2.0, 1.5) heading north,
camera (2.0, 1.63, 0.08). A point (x, y) is `y - camera y` metres ahead of a north-heading camera and `camera x - x` to
its left. `EGO_JOB`'s clearance is 2.0 m, so the contact tie holds out to about 5.5 m (its one-pixel error at the
person's nearest foot is `(d^2 + 0.08^2) / (0.08 fy)` with fy = 381.36, under C/2 = 1.0 m); a test of the tie sets its
own.

A person is `PERSON_SIZE` (the home resident's box, to the centimetre) standing at (x, y), as a `mover` of kind `person`
with no attributes. `put_person` gives it placements and sightings (1,500 px, enough light, a box clear of every edge, a
depth, unless the test says otherwise) and the points of its pixels there: `body`, the outline of its footprint every
2 cm at the floor, at 0.9 m and at its top, so its floor hull is the footprint's rectangle and its feet are in the
frame.

`PeopleScript` is `SpaceScript` whose oracle channel gives a person the evidence rule's hull: a person's `floor_hull` is
`person_seen.hull`, as `perception.py` sets it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import family_kit as fk
from ego_kit import CLEARANCE_M, EGO_JOB, K
from space_kit import SpaceScript, record, vacuum_run
from stbench import person_seen
from stbench.questions_world import World
from synth import camera

PERSON_SIZE = (0.34, 0.56, 1.76)   # the resident's box (people_home.BOX_M, to the centimetre)
HALF_X, HALF_Y = 0.17, 0.28


def body(xy: tuple[float, float], size: tuple[float, float, float] = PERSON_SIZE) -> np.ndarray:
    """The points of a person's pixels: the outline of the footprint every 2 cm at heights 0, 0.9 m and the top."""
    xs = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, 0.02)
    ys = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, 0.02)
    ring = [(x, -size[1] / 2) for x in xs] + [(x, size[1] / 2) for x in xs]
    ring += [(-size[0] / 2, y) for y in ys] + [(size[0] / 2, y) for y in ys]
    return np.array([(xy[0] + x, xy[1] + y, z) for x, y in ring for z in (0.0, 0.9, size[2])])


@dataclass
class PeopleScript(SpaceScript):
    """`SpaceScript` whose oracle channel gives a person the evidence rule's hull at the clearance `c_m`."""

    c_m: float = CLEARANCE_M

    def oracle(self, tl):
        rows, ids = super().oracle(tl)
        seen = {(r["k"], e["id"]): e for r in self.visibility() for e in r["entities"]}
        cameras, cam = self._cameras or self.cameras(), camera(self.stage.depth_unit)
        for row, key in zip(rows, ids, strict=True):
            for d in row["detections"]:
                eid = key["ids"][d["d"]]
                if tl.entity(eid).kind == person_seen.PERSON_KIND:
                    frame = person_seen.Frame(tl, cam, row["k"], tuple(cameras[row["k"]][0]), seen[row["k"], eid],
                                              lambda k, e: self.parts.get((k, e), np.empty((0, 3))),
                                              lambda k, e: self.gaps.get((k, e), 0))
                    d["floor_hull"] = person_seen.hull(frame, eid, self.c_m)
        return rows, ids


def scene(n_tasks: int = 1, **fields) -> PeopleScript:
    """The vacuum's `n_tasks`-task script with the base at (3.0, 1.0) heading north at keyframe `K`
    (`ego_kit.scene`)."""
    poses, yaws = {K: (3.0, 1.0), **fields.pop("poses", {})}, {K: np.pi / 2, **fields.pop("yaws", {})}
    base = vacuum_run(n_tasks, poses=poses, yaws=yaws, depth_ks=set(), **fields)
    return PeopleScript(**{name: getattr(base, name) for name in base.__dataclass_fields__})


def put_person(s: SpaceScript, pid: str, where, seen=(), *, px: int = 1500, lum: float | None = None,
               bbox: list[int] | None = None, depth: float | None = 2.0, pts=None, gaps: int = 0) -> str:
    """The person `pid` standing at (x, y) (`where` a point), or at the placements `where` ([(t0, t1, (x, y, z))]),
    sighted with `px` pixels at the keyframes `seen` (in the box `bbox`, at the median depth `depth`, mean luminance
    `lum`), the points of its pixels there `pts` (default: `body` of where it stands at that keyframe's time) and `gaps`
    of its pixels with no depth."""
    s.item(pid, "person", {}, [(0, None, (*where, 0.0))] if isinstance(where, tuple) and len(where) == 2 else where,
           mobility="mover", size=PERSON_SIZE)
    seen = [seen] if isinstance(seen, int) else list(seen)
    if seen:
        s.see(seen, pid, px, lum=lum, bbox=bbox, depth=depth)
        for k in seen:
            placed = s.timeline().placement_of(pid, s.timeline().keyframe(k).t)
            s.parts[k, pid] = np.array(pts) if pts is not None else body((float(placed.pos[0]), float(placed.pos[1])))
            if gaps:
                s.gaps[k, pid] = gaps
    return pid


def world(s: SpaceScript, job=EGO_JOB, **kw) -> World:
    return fk.world(s, job=job, **kw)


def cut(s: SpaceScript, k: int = K, job=EGO_JOB) -> World:
    """The window of `s`'s first task ended at keyframe `k` (the look point): nothing after it was delivered."""
    return World(fk.inputs(s, job), replace(record(s), k1=k + 1))


def one_person(job=EGO_JOB):
    """The job as the helper declares it: its world holds the one resident."""
    return replace(job, one_person=True)
