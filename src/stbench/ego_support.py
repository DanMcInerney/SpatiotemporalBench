"""The support side of the surface tie of `ego_now` and `ego_mem` (0.7.0): the set-back screen, the edge tie built on
it, and the base z a thing rests on.

**The set-back screen** (the dev pilot's log review). A camera below the top of a table, a desk or a rack shelf sees the
thing's nearest face and the surface's front edge, but not how far back from that edge the thing stands: the top is
edge-on or hidden, so no depth ties the thing to the floor below it, and a reader cannot tell a box on the edge of a
0.9 m table from one half a metre back (AMR case 5, a box 0.23 m back on a packing table above the 0.5 m camera; the
vacuum's picture frame on a desk).

A thing is **set back** when:
- its placement at every sighting the gold uses (`Placement.on`: the supporting entity's id, empty for the floor)
  names a supporting entity, and its base (the surface's top) is at or above that sighting's camera;
- and its gold point lies more than C/2 behind the support's near edge, along the sight line: the camera's floor point
  to the gold point, entering the support's true footprint (`Timeline.footprint`, a rectangle). A gold point outside
  the footprint (a thing mounted on a face) is not behind it.

**The edge tie** (`edge_tie`). In a sighting, a thing that rests on top of a support (`Placement.on` names it and the
gold point lies inside its footprint, by `EDGE_TOL_M`) whose top is at or above that sighting's camera, and is not set
back from that camera, is tied by the support's near edge: the edge itself, within C/2 of the point, is what the camera
reads. The distance is along the sight line, not to the footprint's point nearest the camera: the whole near edge is
seen, so a thing at the edge is tied wherever along it.

**The base** (`contact_bases`). The z the thing rests on in a sighting, which the contact tie looks for pixels at
(`nearest_part.contact_error`): the placement's base z, the floor's 0 or a support's top. A thing hanging on a support's
face (`on` names a support and the gold point lies outside its footprint: a coat on a wardrobe's side) rests on nothing
there, so it has no base. A support with no footprint cannot put the point outside it, and the base stands.

The screens read the timeline and the cameras, never a gold or a score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .geometry import Rect, to_local
from .questions_world import World

EDGE_TOL_M = 0.01   # a point this far outside the footprint is on its edge (placements and hulls round to the mm)
EPS = 1e-9


def inside(rect: Rect, point_xy: Sequence[float]) -> bool:
    """Whether the point lies in the rectangle, or within `EDGE_TOL_M` outside it."""
    px, py = to_local(rect, *point_xy)
    return abs(px) <= rect.sx / 2 + EDGE_TOL_M and abs(py) <= rect.sy / 2 + EDGE_TOL_M


def set_back_m(rect: Rect, camera_xy: Sequence[float], point_xy: Sequence[float]) -> float:
    """How far behind the rectangle's near edge the point lies, along the line from the camera's floor point to it: the
    length of the part of that segment inside the rectangle after it first enters. 0 for a point outside the rectangle
    (by `EDGE_TOL_M`); the whole distance from a camera inside it."""
    hx, hy = rect.sx / 2, rect.sy / 2
    (px, py), (cx, cy) = to_local(rect, *point_xy), to_local(rect, *camera_xy)
    if not inside(rect, point_xy):
        return 0.0
    enter = 0.0
    for c, p, h in ((cx, px, hx), (cy, py, hy)):
        d = p - c
        if abs(d) < EPS:
            if abs(c) > h:
                return 0.0   # runs parallel to a side outside it: never enters
            continue
        enter = max(enter, min((-h - c) / d, (h - c) / d))
    return (1.0 - min(enter, 1.0)) * math.dist(camera_xy, point_xy)


def set_back(w: World, entity: str, evidence: Sequence[int], camera_xy: Sequence[float],
             point_xy: Sequence[float]) -> bool:
    """Whether the thing stands set back on a surface its cameras cannot see the top of (the module docstring):
    `evidence` the keyframes whose pixels give the gold point `point_xy`, `camera_xy` the floor point of the camera the
    point is measured from. A support with no footprint then counts as set back: nothing says where its edge is."""
    support = None
    for j in evidence:
        p = w.tl.placement_of(entity, w.tl.keyframe(j).t)
        if p is None or not p.on or p.pos[2] < w.camera_at(j)[0][2]:
            return False
        support = p.on
    rect = None if support is None else w.tl.footprint(support, w.tl.keyframe(evidence[-1]).t)
    return support is not None and (rect is None
                                    or set_back_m(rect, camera_xy, point_xy) > w.job.clearance_m / 2 + EPS)


def edge_tie(w: World, entity: str, evidence: Sequence[int], point_xy: Sequence[float]) -> bool:
    """Whether some sighting in `evidence` ties the gold point `point_xy` by a support's near edge (the module
    docstring): the thing rests on top of a support (inside its footprint) whose top is at or above that sighting's
    camera, and is not set back from that camera."""
    for j in evidence:
        p = w.tl.placement_of(entity, w.tl.keyframe(j).t)
        cam = w.camera_at(j)[0]
        if p is None or not p.on or p.pos[2] < cam[2]:
            continue
        rect = w.tl.footprint(p.on, w.tl.keyframe(j).t)
        if rect is not None and inside(rect, point_xy) and not set_back(w, entity, [j], cam[:2], point_xy):
            return True
    return False


def contact_bases(w: World, entity: str, evidence: Sequence[int], point_xy: Sequence[float]) -> list[float | None]:
    """The z the thing rests on in each sighting of `evidence`, for the contact tie (the module docstring): its
    placement's base z, or None where it hangs on a support's face or has no placement."""
    out: list[float | None] = []
    for j in evidence:
        t = w.tl.keyframe(j).t
        p = w.tl.placement_of(entity, t)
        rect = w.tl.footprint(p.on, t) if p is not None and p.on else None
        out.append(None if p is None or (rect is not None and not inside(rect, point_xy)) else float(p.pos[2]))
    return out
