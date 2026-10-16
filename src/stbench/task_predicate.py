"""The predicate of a job's `task_things` question (0.6.0): which things the question is about, declared over the log.

`task_things` asks "where were the things {phrase} in your {ordinal} task?" The predicate is declarative over what the
robot was sent (the steps of the task: base positions and `status.phase`) and the public map, never over the job's
plan, because the scripted baselines that must reproduce the gold (the oracle, `memory_table`) are fenced from plans.
This module holds the declaration (`TaskThings`, one per job, in `JobSpec.task_things`) and the geometry that
evaluates it, which the gold (`questions_task`) and the tables (`table_task`) both call: the anchor of a task from its
steps (`anchor_of`), a footprint's distance to it and its verdict (`verdict`, `judge`), and what a table lists
(`reaches`).

- `phrase` is the predicate in words as the question says it, so the text states the distance the gold uses.
- `anchor` is what the things are near: `path` (the base positions over the steps of `phases`, up to the task's last
  step of `until`), `rooms` (the regions holding a step of `phases`) or `point` (the median base position over
  `phases`).
- `reach_m` is the distance (to the footprint's nearest point) within which a thing meets the predicate of a `path` or
  `point` anchor; `band_m` the don't-care band around it (reach +- band), and for `rooms` (band 0) the doorway ways and
  a footprint across a room's edge.
- `floor` keeps only things on the robot's floor as items, the rest don't-care; `min_height_m` the floor-level cutoff;
  `dont_care_kinds` the kinds that are never an item; `item_kinds` (0.11.0) the only kinds of a carried thing or clutter
  that can be an item (empty: every kind), and a kind that neither it nor `dont_care_kinds` holds is no item and no
  don't-care place either; `empty_keep_percent` the share of empty-gold candidates the draw keeps, per job (100: all).

Neutral: the geometry and the public map, nothing that reads the world.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from . import zone_shapes
from .geometry import Rect, corners, point_in_polygon, rect_distance, rects_overlap
from .mapspec import INTERIOR_TOL, MapSpec, Region

ANCHORS = ("path", "rooms", "point")


@dataclass(frozen=True)
class TaskThings:
    phrase: str                  # the predicate by the task's role, as the question says it
    anchor: str                  # "path" | "rooms" | "point", built from the delivered steps of the task
    phases: frozenset[str]       # the steps whose base positions make the anchor
    until: str | None = None     # path: only steps up to the task's last step of this phase
    reach_m: float = 0.0         # path, point: the predicate's distance (to the footprint's nearest point)
    band_m: float = 0.0          # don't-care within reach +- band; rooms: the doorway ways
    floor: bool = False          # only things on the robot's floor are items; the rest don't-care
    min_height_m: float = 0.0
    dont_care_kinds: frozenset[str] = frozenset()
    item_kinds: frozenset[str] = frozenset()   # empty: every kind may be an item (the vacuum's: what its brush catches)
    # Of the candidates whose gold is empty, the percent the draw keeps, per job: one is kept when
    # `stable_hash(case, family, key, "empty") % 100 < empty_keep_percent`. 100 keeps all; a job whose empties pass a
    # third is set to 100 (1 - p) / (2 p) before the binding read.
    empty_keep_percent: int = 100

    def __post_init__(self) -> None:
        if not self.phrase or not isinstance(self.phrase, str):
            raise ValueError(f"phrase must be a non-empty string, got {self.phrase!r}")
        if self.anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {list(ANCHORS)}, got {self.anchor!r}")
        for name in ("phases", "dont_care_kinds", "item_kinds"):
            names = getattr(self, name)
            if isinstance(names, str) or not all(isinstance(n, str) and n for n in names):
                raise ValueError(f"{name} must be a collection of non-empty strings (a bare string would match by "
                                 f"substring), got {names!r}")
        if not self.phases:
            raise ValueError("phases must name at least one phase")
        if self.until is not None and self.anchor != "path":
            raise ValueError(f"until is for a path anchor, got it with {self.anchor!r}")
        if isinstance(self.empty_keep_percent, bool) or not isinstance(self.empty_keep_percent, int) \
                or not 0 <= self.empty_keep_percent <= 100:
            raise ValueError(f"empty_keep_percent must be a whole number 0 to 100, got {self.empty_keep_percent!r}")
        for name in ("reach_m", "band_m", "min_height_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be at least 0, got {getattr(self, name)!r}")


# -- the geometry the predicate is evaluated with (the gold and the tables call this one code) -------------------
EPS = 1e-9
IN, EDGE, OUT = "in", "edge", "out"   # a footprint inside the predicate, within its band, or outside it
Xy = tuple[float, float]


@dataclass(frozen=True)
class Anchor:
    """What a task's things are near, built from its delivered steps: `segments` (a `path` anchor: the base positions
    of the matched steps, a segment between each two that follow one another in the task; a `point` anchor: the one
    median point, as a segment of no length) or `rooms` and the `ways` through their openings (a `rooms` anchor)."""

    kind: str
    segments: tuple[tuple[Xy, Xy], ...] = ()
    rooms: tuple[Region, ...] = ()
    ways: tuple[Rect, ...] = ()


def anchor_of(pred: TaskThings, steps: Sequence[dict], m: MapSpec | None = None) -> Anchor | None:
    """The anchor of one task from its delivered steps (each with `pose.pos` and `status.phase`, in step order) and the
    public map (`rooms` only). The matched steps are those whose phase is in `pred.phases`, up to the task's last step
    of `pred.until` (a task without that phase has no anchor: None). `path`: their base positions, a segment between
    two matched steps that follow one another; `point`: the median x and the median y of their base positions;
    `rooms`: the regions holding the base of a matched step, and the way through each opening of those regions.
    None when no step matches, or no matched base stands in a room."""
    stop = len(steps)
    if pred.until is not None:
        last = [i for i, s in enumerate(steps) if s["status"]["phase"] == pred.until]
        if not last:
            return None
        stop = last[-1] + 1
    kept = [i for i in range(stop) if steps[i]["status"]["phase"] in pred.phases]
    if not kept:
        return None
    at = [(float(steps[i]["pose"]["pos"][0]), float(steps[i]["pose"]["pos"][1])) for i in kept]
    if pred.anchor == "point":
        p = (statistics.median(x for x, _ in at), statistics.median(y for _, y in at))
        return Anchor("point", segments=((p, p),))
    if pred.anchor == "path":
        follows = [kept[n + 1] == kept[n] + 1 for n in range(len(at) - 1)]
        joined = [(at[n], at[n + 1]) for n, f in enumerate(follows) if f]
        alone = [(p, p) for n, p in enumerate(at) if not (follows[n] if n < len(follows) else False)
                 and not (follows[n - 1] if n else False)]
        return Anchor("path", segments=tuple([*joined, *alone]))
    if m is None:
        raise ValueError("a rooms anchor is built from the public map: anchor_of needs it")
    ids = {r for p in at if (r := m.region_of(p)) is not None}
    if not ids:
        return None
    ways = tuple(zone_shapes.doorway_way(m, o) for o in m.openings if ids & set(o.between))
    return Anchor("rooms", rooms=tuple(r for r in m.regions if r.id in ids), ways=ways)


def detected_rect(centre: Xy, size: Sequence[float]) -> Rect:
    """The footprint a detection gives: the thing's extents about its centre, with no heading (a detection carries
    none)."""
    return Rect(centre[0], centre[1], size[0], size[1], 0.0)


def _segment_distance(rect: Rect, a: Xy, b: Xy) -> float:
    """The least distance from the rectangle to the segment a-b (0 when they touch), the segment taken as a rectangle
    of no width."""
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return rect_distance(rect, Rect(mid[0], mid[1], math.dist(a, b), 0.0, math.atan2(b[1] - a[1], b[0] - a[0])))


def distance(anchor: Anchor, rect: Rect) -> float:
    """The distance from the footprint's nearest point to a `path` or `point` anchor (0 when it lies on it)."""
    return min(_segment_distance(rect, a, b) for a, b in anchor.segments)


def _overlaps(rect: Rect, polygon: Sequence[Xy]) -> bool:
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


def _room_verdict(anchor: Anchor, rect: Rect) -> str:
    """`rooms`: a footprint on a way is `EDGE`; wholly inside one of the rooms is `IN`; across a room's edge `EDGE`;
    else `OUT`."""
    if any(rects_overlap(rect, way) for way in anchor.ways):
        return EDGE
    inside = corners(rect, INTERIOR_TOL)
    polygons = [r.polygon for r in anchor.rooms]
    if any(all(point_in_polygon(x, y, list(p)) for x, y in inside) for p in polygons):
        return IN
    return EDGE if any(_overlaps(rect, p) for p in polygons) else OUT


def verdict(pred: TaskThings, anchor: Anchor, rect: Rect) -> str:
    """Where one footprint stands against the predicate. `path` and `point`: `IN` within `reach_m - band_m` of the
    anchor (to the footprint's nearest point), `EDGE` within `reach_m + band_m`, else `OUT`. `rooms`: see
    `_room_verdict`."""
    if anchor.kind == "rooms":
        return _room_verdict(anchor, rect)
    d = distance(anchor, rect)
    if d <= pred.reach_m - pred.band_m + EPS:
        return IN
    return EDGE if d <= pred.reach_m + pred.band_m + EPS else OUT


def judge(pred: TaskThings, anchor: Anchor, rects: Iterable[Rect]) -> str:
    """The verdict on a thing whose footprint is any of `rects` (the true one and the detected one, `detected_rect`):
    `IN` when every one is, `OUT` when every one is, else `EDGE`. A thing whose distance depends on a heading no
    detection carries is don't-care wherever the footprints disagree, so a table that sees only the detected one can
    not be charged for it."""
    found = {verdict(pred, anchor, r) for r in rects}
    return found.pop() if len(found) == 1 else EDGE


def reaches(pred: TaskThings, anchor: Anchor, rect: Rect) -> bool:
    """What a table lists from a detection: the footprint is within `reach_m` of a `path` or `point` anchor, or wholly
    inside the rooms and off their ways. Every thing the gold counts as an item passes it, and every thing it passes
    is, to the gold, an item or within its band (`judge`)."""
    if anchor.kind == "rooms":
        return _room_verdict(anchor, rect) == IN
    return distance(anchor, rect) <= pred.reach_m + EPS


def on_floor(bottom_z: float, base_z: float, climb_m: float) -> bool:
    """A thing whose bottom is `bottom_z` stands on the floor the robot drives on: within the robot's `climb_m` of its
    delivered base height."""
    return abs(bottom_z - base_z) <= climb_m + EPS
