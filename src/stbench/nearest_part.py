"""The nearest visible part of a thing (0.7.0): the one owner of the geometry `ego_now` and `ego_mem` ask for.

**The target.** Of what the camera showed of a thing, the point nearest the camera's floor point, measured along the
floor (docs/families/ego_now.md). It is built in three steps, each a function here:
1. `visible_points`: the thing's own instance pixels in a keyframe, unprojected through that keyframe's depth at the
   pixel centres and moved to the map by the pose row's camera, as `grid.points` does for the free-space rule (a
   pixel of depth 0 adds no point).
2. `floor_hull`: those points dropped onto the floor plane as (x, y), and their convex hull, counter-clockwise, to the
   millimetre. This is what the oracle channel sends of a recognisable detection (`perception.py`, `floor_hull`), so a
   table and the gold take the same polygon.
3. `nearest_on_hull`: the point of the filled hull nearest the ground point. It lies on an edge as often as at a vertex,
   and is the ground point itself when the camera is over the thing (a drone lowering): the answer (0, 0) is valid.

`ego_mem` takes the hull of the union of the hulls of every recognisable sighting of the thing at its current place
(`union_hull`, a hull of hull vertices), so a side that no sighting showed does not count. `nearest_part` does the whole
of it for the gold and also gives the height of the visible point nearest the answer (for the report's slant).

**The resolvability screen (the surface tie).** A point is asked only where a camera-only reader, with no depth, could
place it: where the gold point is tied to a surface whose height the reader can read. At least one evidence frame the
gold uses must give one of two ties.
1. **The contact tie** (`contact_error`, `surface_tie`). The frame shows a visible pixel where the thing meets the
   surface it rests on, within C/2 along the floor of the gold point. The surface is the floor (z = 0) or the support's
   top, the placement's base z, whether a slab, a shelf or a table. A **contact pixel** is a visible point of the thing
   whose z is within `max(CONTACT_M, slant / fy)` of the base z: 2 cm, or one pixel's height at that point's range when
   that is more. It must also read there to within C/2: one pixel moves a point on the base plane by
   `contact_pixel_error` = `(d^2 + H^2) / (H * fy)` metres, d the floor distance from that frame's camera to the
   contact point and H the camera's height above that plane, `|camera z - base z|`; a camera within `CONTACT_M` of the
   plane reads nothing. The error of the gold point is the smallest over the contact pixels and the sightings, each from
   its own camera. A thing hanging on a support's face (its placement names a support and the gold point lies outside
   the support's footprint) has no base to rest on: the caller gives it none, and it has no contact.
2. **The edge tie** (`ego_support.edge_tie`, not here). The thing rests on top of a support whose top is at or above
   the camera, so the contact is hidden from below, and it is not set back: within C/2 of the support's near edge along
   the sight line, where the edge itself is the tie. `surface_tie` combines the two.

A floor lamp whose nearest point is a shade overhanging a base more than C/2 away (or hidden), and a coat hanging on a
wardrobe's side, have neither tie and are not asked. The earlier rules read each point at its true height (a reader
with no depth does not know it) and then only a floor pixel (which dropped every raised thing); the surface tie reads
the height of the surface the thing rests on, which a camera can see. The rule is the same for every job and reads only
the cameras, the poses, the points, the placements' bases and supports, and C.

A thing cut by the frame's edge has its visible part, which counts (the gold's evidence is always a recognisable
sighting, whose box clears the edge); a thing with no pixel that has a depth has no points and no part (None). The
nearest of equally near points goes to the smaller x, then y, then z, and distances compare to a nanometre, so float
noise never picks the point. The module is neutral (numpy, `camera`, `grid`): the gold and the table systems choose
their point here; `path_log` takes the nearest of the points it learned (`baselines_path_ego`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from . import grid
from .camera import Camera

DIST_DECIMALS = 9   # distances compare to a nanometre
HULL_DECIMALS = 3   # a hull vertex is given to the millimetre
CONTACT_M = 0.02    # a visible point this near its base (or one pixel's height, if more) is a contact pixel
UNCHANGED_M = 0.01  # sightings of a thing less than this apart in the map are at one place: `ego_mem` unions them


class Part(NamedTuple):
    """The gold of one look: `xy` the point of the floor hull nearest the ground point, `z` the height of the visible
    3-D point nearest it on the floor."""

    xy: tuple[float, float]
    z: float


def visible_points(index: np.ndarray, depth: np.ndarray, cam: Camera, cam_pos: Sequence[float],
                   cam_rot: Sequence[float], pass_index: int) -> np.ndarray:
    """The (N, 3) map-frame points of the pixels of instance `pass_index` in `index` that have a depth in `depth`
    (counts of `cam.depth_unit`; both images are the camera's height by width), unprojected as `grid.points` does and
    moved by the camera's pose. Shape (0, 3) when the thing has no such pixel. ValueError when the images disagree."""
    if index.shape != depth.shape:
        raise ValueError(f"instance image {index.shape} and depth image {depth.shape} differ")
    return grid.points(np.where(index == pass_index, depth, 0), cam, cam_pos, cam_rot)


def floor_hull(points: Sequence[Sequence[float]] | np.ndarray) -> list[list[float]]:
    """The convex hull of the points' (x, y), counter-clockwise from the lowest (smallest x, then y) vertex, every
    vertex rounded to the millimetre; collinear points are not vertices. A single point gives one vertex, collinear
    points the two ends, no point an empty list. Points that round to one millimetre cell count once."""
    pts = sorted({(round(float(p[0]), HULL_DECIMALS), round(float(p[1]), HULL_DECIMALS)) for p in points})
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def chain(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = chain(pts), chain(reversed(pts))
    return [list(p) for p in lower[:-1] + upper[:-1]]


def union_hull(hulls: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    """The hull of the union of hulls: `floor_hull` of all their vertices."""
    return floor_hull([v for hull in hulls for v in hull])


def _on_segment(a: Sequence[float], b: Sequence[float], q: Sequence[float]) -> tuple[float, float]:
    """The point of the segment ab nearest q."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else min(1.0, max(0.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / length2))
    return (a[0] + t * dx, a[1] + t * dy)


def nearest_on_hull(hull: Sequence[Sequence[float]], ground_xy: Sequence[float]) -> tuple[float, float] | None:
    """The point of the filled hull (a `floor_hull`) nearest `ground_xy`: the ground point itself when it lies inside or
    on the hull, else the nearest point of the boundary. None for an empty hull. Ties go to the smaller x, then y."""
    if not hull:
        return None
    q = (float(ground_xy[0]), float(ground_xy[1]))
    n = len(hull)
    if n >= 3 and all((hull[(i + 1) % n][0] - hull[i][0]) * (q[1] - hull[i][1])
                      - (hull[(i + 1) % n][1] - hull[i][1]) * (q[0] - hull[i][0]) >= 0 for i in range(n)):
        return q
    edges = [(hull[i], hull[(i + 1) % n]) for i in range(n if n > 2 else n - 1)] or [(hull[0], hull[0])]
    found = [_on_segment(a, b, q) for a, b in edges]
    return min(found, key=lambda p: (round(math.hypot(p[0] - q[0], p[1] - q[1]), DIST_DECIMALS), p))


def nearest_point(points: np.ndarray, ground_xy: Sequence[float]) -> tuple[float, float, float] | None:
    """The point of `points` ((N, 3), map frame) nearest `ground_xy` in the floor plane, as (x, y, z); None for no
    point. Ties (to `DIST_DECIMALS`) go to the smaller x, then y, then z."""
    if len(points) == 0:
        return None
    dist = np.round(np.hypot(points[:, 0] - ground_xy[0], points[:, 1] - ground_xy[1]), DIST_DECIMALS)
    tied = points[dist == dist.min()]
    first = tied[np.lexsort((tied[:, 2], tied[:, 1], tied[:, 0]))[0]]
    return float(first[0]), float(first[1]), float(first[2])


def contact_pixel_error(cam_pos: Sequence[float], point_xy: Sequence[float], base_z: float, fy: float) -> float | None:
    """How far one pixel moves a point on the plane z = `base_z` at `point_xy`, seen from a camera at `cam_pos` with
    focal length `fy` pixels: `(d^2 + H^2) / (H fy)`, d the floor distance and H = |camera z - base z|. None within
    `CONTACT_M` of the plane (the ray runs along it: no bound)."""
    d, h = math.hypot(point_xy[0] - cam_pos[0], point_xy[1] - cam_pos[1]), abs(cam_pos[2] - base_z)
    return None if h <= CONTACT_M else (d * d + h * h) / (h * fy)


def contact_error(seen: Sequence[tuple[Sequence[float], np.ndarray, float | None]], xy: Sequence[float], fy: float,
                  c_m: float) -> float | None:
    """The one-pixel error of the gold point `xy` at its contact: the smallest `contact_pixel_error` over the contact
    pixels of the thing, in any of the sightings `seen` (each the position of its camera, the (N, 3) points it saw and
    the base z the thing rests on there, None for a thing with no base: one hanging on a face), each from its own
    camera. A contact pixel is a visible point within `max(CONTACT_M, slant / fy)` of the base z (slant the 3-D distance
    from the camera) and within C/2 (`c_m` / 2) of `xy` along the floor. None when no sighting has one."""
    errors = []
    for cam, pts, base in seen:
        if base is None or not len(pts):
            continue
        slant = np.sqrt(((pts - np.asarray(cam, dtype=float)) ** 2).sum(axis=1))
        near = (np.abs(pts[:, 2] - base) <= np.maximum(CONTACT_M, slant / fy) + grid.EPS) \
            & (np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1]) <= c_m / 2 + grid.EPS)
        errors += [e for p in pts[near] if (e := contact_pixel_error(cam, p, base, fy)) is not None]
    return min(errors, default=None)


def nearest_part(sightings: Sequence[tuple[Sequence[float], np.ndarray]], ground_xy: Sequence[float]) -> Part | None:
    """The gold of a look. `sightings` are the recognisable sightings of the thing at its place, each the position of
    the camera that saw it and the (N, 3) points of the thing's pixels there (`visible_points`). The point asked is the
    point of the union of their floor hulls nearest `ground_xy`; its height is that of the visible point nearest it on
    the floor (the first on a tie). None when no sighting has a point."""
    seen = [(cam, pts) for cam, pts in sightings if len(pts)]
    if not seen or (xy := nearest_on_hull(union_hull([floor_hull(pts) for _, pts in seen]), ground_xy)) is None:
        return None
    dists = [np.round(np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1]), DIST_DECIMALS) for _, pts in seen]
    nearest = min(float(d.min()) for d in dists)
    z = next(nearest_point(pts, xy)[2] for (_, pts), d in zip(seen, dists, strict=True) if d.min() == nearest)
    return Part(xy, z)


def resolvable(contact_m: float | None, c_m: float) -> bool:
    """Whether the contact tie holds: the contact's one-pixel error is at most C/2 metres. False for no error (no
    contact pixel near the point)."""
    return contact_m is not None and contact_m <= c_m / 2 + 1e-9


def surface_tie(contact_m: float | None, edge: bool, c_m: float) -> str | None:
    """Which tie asks the question: "contact" when the contact's error is within C/2 (`resolvable`), else "edge" when
    the edge tie holds (`ego_support.edge_tie`), else None. The contact tie goes first, so a thing both ties admit is
    counted once, and read at its contact."""
    return "contact" if resolvable(contact_m, c_m) else "edge" if edge else None


def fallback_ladder(contact_m: float | None, c_m: float) -> list[float] | None:
    """The rungs of a question the screen would drop, kept (FD-0.7.0 review, change 5, the declared fallback): `max(C m,
    2 m e)` for m in 0.5, 1, 2, with e the contact tie's one-pixel error, so the full rung is what one pixel could
    miss. None for no error: a thing with no contact pixel near the point is never kept, and one the edge tie asks needs
    no fallback."""
    return None if contact_m is None else [round(max(c_m * m, 2 * m * contact_m), 6) for m in (0.5, 1.0, 2.0)]
