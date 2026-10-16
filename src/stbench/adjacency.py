"""Which rooms of a layout are joined, which room connects two others, and which joins a path crossed (`adjacent`).

Two regions are **joined** when a door or an arch cuts through the wall between them (both `between` sides are regions:
the entrance and "outside" join nothing), or when their polygons share a boundary segment on which no wall piece
stands, the open-plan edge the planner crosses (`maps/README.md`, "regions"): collinear polygon edges of regions on one
level, within `EDGE_TOL_M`, whose overlap holds a run of at least `JOIN_MIN_M` clear of every wall piece
(`MapSpec.wall_pieces`, door and arch cuts excluded). Two rooms with only a wall between them have polygons a wall
thickness apart, so they share no edge and join nothing; a shared edge shorter than `JOIN_MIN_M`, or cut into shorter
runs by a wall, joins nothing either: no robot passes there.

`connector` is the one region joined to both of two rooms that are not joined to each other; `regions_along`,
`crossed` and `first_crossings` read a path's base positions through the same map; `toured` says whether the cells a
robot observed show a room, and `answer_point` is the point of the connector that the gold names. The gold of `adjacent`
and the table systems both call this module, so they agree on every edge, every tour and every point. It is neutral:
`mapspec`, `geometry`, `grid` and the standard library only, no truth and no world.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache, lru_cache
from itertools import combinations, pairwise

from . import grid
from .geometry import Point, Rect, dist_to_segment, point_in_polygon, to_local
from .mapspec import MapSpec, Region

JOIN_MIN_M = 0.8  # check_map's narrowest door: a shared edge shorter than this joins nothing
EDGE_TOL_M = 0.002  # two polygon edges this close to one line are collinear
TOURED = 0.5  # the share of a room's inside fine cells a robot must observe before it counts as toured (`changed`'s)
ERODE_M = 0.2  # the margin of the gold row's `region` credit: the gold point lies this far inside the connector's walls
Pair = frozenset[str]
Cells = dict[tuple[int, int], tuple[str, int]]   # `grid.latest`: each observed fine cell's latest state and its step


@lru_cache(maxsize=16)
def joins(m: MapSpec) -> frozenset[Pair]:
    """The pairs of region ids joined by an opening or an open shared edge (the module docstring)."""
    ids = {r.id for r in m.regions}
    found = {frozenset(o.between) for o in m.openings if o.between[0] != o.between[1] and set(o.between) <= ids}
    pieces = m.wall_pieces()
    for a, b in combinations(m.regions, 2):
        if frozenset((a.id, b.id)) not in found and a.level == b.level and _open_edge(a, b, pieces):
            found.add(frozenset((a.id, b.id)))
    return frozenset(found)


def _open_edge(a: Region, b: Region, pieces: list[Rect]) -> bool:
    """Whether some edge of `a` and some edge of `b` overlap along a run of `JOIN_MIN_M` or more that no wall piece
    stands on."""
    for p, q in _edges(a.polygon):
        length = math.dist(p, q)
        if length < 1e-9:
            continue
        axis = ((q[0] - p[0]) / length, (q[1] - p[1]) / length)
        for r, s in _edges(b.polygon):
            along = [_along(p, axis, t) for t in (r, s)]
            if any(abs(off) > EDGE_TOL_M for _, off in along):
                continue
            lo, hi = max(0.0, min(t for t, _ in along)), min(length, max(t for t, _ in along))
            if hi - lo >= JOIN_MIN_M - 1e-9 and _longest_clear(p, axis, lo, hi, pieces) >= JOIN_MIN_M - 1e-9:
                return True
    return False


def _edges(polygon: Sequence[Point]) -> list[tuple[Point, Point]]:
    return [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]


def _along(origin: Point, axis: Point, point: Point) -> tuple[float, float]:
    """The point's distance along the axis from the origin, and across it (the left of the axis is positive)."""
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    return dx * axis[0] + dy * axis[1], dy * axis[0] - dx * axis[1]


def _longest_clear(origin: Point, axis: Point, lo: float, hi: float, pieces: list[Rect]) -> float:
    """The length of the longest run of the segment origin + t * axis, t in [lo, hi], that lies inside no wall piece
    (a piece counts `EDGE_TOL_M` smaller on every side, so a wall face flush with the edge stands on none of it)."""
    covered = sorted(c for piece in pieces if (c := _inside(origin, axis, lo, hi, piece)) is not None)
    best, cursor = 0.0, lo
    for start, end in covered:
        best, cursor = max(best, start - cursor), max(cursor, end)
    return max(best, hi - cursor)


def _inside(origin: Point, axis: Point, lo: float, hi: float, piece: Rect) -> tuple[float, float] | None:
    """The part [t0, t1] of the segment origin + t * axis, t in [lo, hi], inside the wall piece; None when none."""
    c, s = math.cos(piece.yaw), math.sin(piece.yaw)
    start = to_local(piece, origin[0], origin[1])
    step = (c * axis[0] + s * axis[1], -s * axis[0] + c * axis[1])
    for p0, d, half in zip(start, step, (piece.sx / 2 - EDGE_TOL_M, piece.sy / 2 - EDGE_TOL_M), strict=True):
        if abs(d) < 1e-12:
            if abs(p0) > half:
                return None
        else:
            a, b = (-half - p0) / d, (half - p0) / d
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return (lo, hi) if hi > lo else None


def connector(m: MapSpec, a: str, b: str) -> str | None:
    """The one region joined to both `a` and `b` when they are different regions not joined to each other and exactly
    one such region exists; else None."""
    pairs = joins(m)
    if a == b or frozenset((a, b)) in pairs:
        return None
    between = [r.id for r in m.regions if r.id not in (a, b)
               and frozenset((a, r.id)) in pairs and frozenset((b, r.id)) in pairs]
    return between[0] if len(between) == 1 else None


def _runs(m: MapSpec, xys: Sequence[Point]) -> list[tuple[str, int]]:
    """(region id, index of its first position) for each stretch of the path in one region, a position in no region (a
    doorway, a wall) dropped and a repeat collapsed."""
    runs: list[tuple[str, int]] = []
    for i, xy in enumerate(xys):
        room = m.region_of(xy)
        if room is not None and (not runs or runs[-1][0] != room):
            runs.append((room, i))
    return runs


def regions_along(m: MapSpec, xys: Sequence[Point]) -> list[str]:
    """The regions of a path's base positions in order, positions in no region dropped and repeats collapsed."""
    return [room for room, _ in _runs(m, xys)]


def crossed(m: MapSpec, xys: Sequence[Point]) -> frozenset[Pair]:
    """The joined pairs the path moves between: consecutive entries of `regions_along` that `joins` holds."""
    return frozenset(first_crossings(m, xys))


def first_crossings(m: MapSpec, xys: Sequence[Point]) -> dict[Pair, int]:
    """Each pair of `crossed`, mapped to the index of the position where the path first enters the second region of a
    move between them (in either direction)."""
    pairs, out = joins(m), {}
    for (a, _), (b, i) in pairwise(_runs(m, xys)):
        if (pair := frozenset((a, b))) in pairs:
            out.setdefault(pair, i)
    return out


def interior_point(polygon: Sequence[Point], margin_m: float) -> Point | None:
    """The polygon's area centroid to 4 decimals when it lies at least `margin_m` inside the polygon, else None: a
    sliver too narrow to hold the margin, or a concave room (an L-shaped one, say) whose centroid falls outside it or
    too near a wall."""
    if len(polygon) < 3:
        return None
    twice, cx, cy = 0.0, 0.0, 0.0  # twice the signed area, and the centroid's sums
    for (x0, y0), (x1, y1) in _edges(polygon):
        cross = x0 * y1 - x1 * y0
        twice, cx, cy = twice + cross, cx + (x0 + x1) * cross, cy + (y0 + y1) * cross
    if abs(twice) < 1e-12:
        return None
    x, y = round(cx / (3 * twice), 4), round(cy / (3 * twice), 4)
    if not point_in_polygon(x, y, list(polygon)):
        return None
    return (x, y) if min(dist_to_segment(x, y, p, q) for p, q in _edges(polygon)) >= margin_m else None


@cache
def inside_cells(region: Region) -> tuple[tuple[int, int], ...]:
    """The fine cells whose centres lie in the room."""
    poly = list(region.polygon)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    (u0, v0), (u1, v1) = grid.fine_cell(min(xs), min(ys)), grid.fine_cell(max(xs), max(ys))
    return tuple((u, v) for u in range(u0, u1 + 1) for v in range(v0, v1 + 1)
                 if point_in_polygon((u + 0.5) * grid.FINE_M, (v + 0.5) * grid.FINE_M, poly))


def toured(region: Region, cells: Cells) -> bool:
    """At least `TOURED` of the room's inside fine cells are among the observed `cells`, in any state: the camera showed
    the room."""
    inside = inside_cells(region)
    return bool(inside) and sum(c in cells for c in inside) >= TOURED * len(inside)


def answer_point(via: Region) -> Point | None:
    """The point of the connecting room that the gold names: its centroid when that lies `ERODE_M` inside the room,
    else None (the room has no point the `region` credit would accept)."""
    return interior_point(via.polygon, ERODE_M)
