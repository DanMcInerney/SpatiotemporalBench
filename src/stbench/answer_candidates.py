"""The answers a best response tries: every answer type's candidates over a pool of gold rows (plan §4.6).

The prior systems (`blind`, `path_log`, `current_view_prior`) answer a question with whichever candidate scores best
over the gold rows of the pool its key found (`answer_keys.levels`). A type this module did not know would give the
pool no candidate worth trying, and its shortcut gate would pass because the shortcut never tried an answer, so an
unknown type raises and every type of `answers.TYPES` is covered. This module is neutral baseline-side code: it reads
gold rows as data and imports no module that reads truth.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from fractions import Fraction

from .answers import MAX_CELLS, TYPES
from .places import dense_places
from .score_set import boundary_distance, inside

MAX_CANDIDATES = 60
# a cell joins a candidate set when at least this share of the pool's golds hold it
CELL_SHARES = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
GRID = 24            # the grid, per axis, that `interior_point` searches
UNKNOWN = {"type": "unknown"}

Point = tuple[float, float]


def _depth(p: Point, polygon: Sequence[Point]) -> float:
    """How far inside the polygon `p` stands (negative outside): its distance to the boundary."""
    d = boundary_distance(p, polygon)
    return d if inside(p, polygon) else -d


def interior_point(polygon: Sequence[Point]) -> Point:
    """A point deep inside `polygon`: the one farthest from its boundary among the area centroid, the bounding box's
    centre and a `GRID` x `GRID` grid over the box (the first of equals, so a rectangle gets its centroid)."""
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    edges = list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges) / 2
    centroid = ((sum((a[0] + b[0]) * (a[0] * b[1] - b[0] * a[1]) for a, b in edges) / (6 * area),
                 sum((a[1] + b[1]) * (a[0] * b[1] - b[0] * a[1]) for a, b in edges) / (6 * area))
                if area else (statistics.fmean(xs), statistics.fmean(ys)))
    x0, y0, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    seeds = [centroid, (x0 + w / 2, y0 + h / 2)]
    grid = [(x0 + (i + 0.5) * w / GRID, y0 + (j + 0.5) * h / GRID) for j in range(GRID) for i in range(GRID)]
    return max([*seeds, *grid], key=lambda p: _depth(p, polygon))


def _position(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The centroid and median of the pool's points, then each point; for `region` golds one interior point of each
    region instead (the points the pool names are no better than another point inside the room)."""
    regions = {tuple(map(tuple, r["region"])) for r in rows if r.get("credit") == "region"}
    points = [tuple((g["gold"].get("xy") or g["gold"]["xyz"])[:2]) for g in rows if g.get("credit") != "region"]
    summary = [_at(interior_point(region)) for region in sorted(regions)]
    if points:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        summary += [_at((statistics.fmean(xs), statistics.fmean(ys))),
                    _at((statistics.median(xs), statistics.median(ys)))]
    return summary, [_at(p) for p in points]


def _at(p: Point) -> dict:
    return {"type": "position", "xy": [p[0], p[1]]}


def _offset(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The mean and the median of the pool's offsets, then each offset: a constant guess in the camera's frame."""
    ahead, left = [g["gold"]["ahead_m"] for g in rows], [g["gold"]["left_m"] for g in rows]

    def at(a: float, l_: float) -> dict:
        return {"type": "offset", "ahead_m": a, "left_m": l_}

    summary = [at(statistics.fmean(ahead), statistics.fmean(left)),
               at(statistics.median(ahead), statistics.median(left))]
    return summary, [at(a, l_) for a, l_ in zip(ahead, left, strict=True)]


def _number(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    values = [g["gold"]["value"] for g in rows]
    return [{"type": "number", "value": statistics.median(values)}], [{"type": "number", "value": v} for v in values]


def _time(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The median clock, then each clock: the credit is a ladder of seconds, so a middle time can beat every one."""
    clocks = sorted(datetime.fromisoformat(g["gold"]["clock"]) for g in rows)
    middle = clocks[(len(clocks) - 1) // 2] + (clocks[len(clocks) // 2] - clocks[(len(clocks) - 1) // 2]) / 2
    return [{"type": "time", "clock": middle.isoformat()}], [{"type": "time", "clock": c.isoformat()} for c in clocks]


def _values(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """`yesno` and `count`: each value the pool holds."""
    return [], [dict(g["gold"]) for g in rows]


def _set(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    return [{"type": "set", "items": []}], []


def _positions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The empty list, then the pool's 1 to 8 densest places (`places.dense_places`)."""
    spots = dense_places(rows)
    return [{"type": "positions", "items": []},
            *({"type": "positions", "items": [{"xy": [s[0], s[1]]} for s in spots[:n]]}
              for n in range(1, len(spots) + 1))], []


def _cells(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The empty set, then the cells present in at least 1/4, 1/2 and 3/4 of the pool."""
    pool = [{tuple(c) for c in g["gold"]["items"]} for g in rows]
    seen = Counter(c for cells in pool for c in cells)
    sets = [sorted(c for c, k in seen.items() if Fraction(k, len(pool)) >= share) for share in CELL_SHARES]
    return [{"type": "cells", "items": []},
            *({"type": "cells", "items": [list(c) for c in cells]} for cells in sets if len(cells) <= MAX_CELLS)], []


def _route(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Each valid sequence of the pool's route golds, as an interior point of each of its regions."""
    routes = []
    for g in rows:
        regions = {name: interior_point(polygon) for name, polygon in g["gold"]["regions"].items()}
        routes += [{"type": "route", "points": [list(regions[name]) for name in seq]} for seq in g["gold"]["valid"]]
    return routes, []


def _heading(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The four headings of the compass (0, 90, 180 and -90 degrees left) and the median of the pool's, then each gold's
    own heading: a constant turn, ahead, behind or to a side, and where the pool's points mostly lie."""
    degs = [g["gold"]["deg_left"] for g in rows]

    def at(deg: float) -> dict:
        return {"type": "heading", "deg_left": float(deg)}

    return [at(d) for d in (0, 90, 180, -90, statistics.median(degs))], [at(d) for d in degs]


_HANDLERS = {"position": _position, "offset": _offset, "number": _number, "time": _time, "yesno": _values,
             "count": _values, "set": _set, "positions": _positions, "cells": _cells, "route": _route,
             "heading": _heading}


def candidates(golds: list[dict]) -> list[dict]:
    """The answers a best response tries over a pool of gold rows: `unknown` first, then each present type's summaries
    (centroids and medians, an offset's mean and median, the empty answer, the densest places, the cells most golds
    share, an interior point of each region, each valid route), then the pool's own values; no repeats, at most
    `MAX_CANDIDATES`, so the summaries are never the ones cut. A gold of a type outside `answers.TYPES` raises
    ValueError."""
    by_type: dict[str, list[dict]] = {}
    for g in golds:
        kind = g["gold"]["type"]
        if kind != "unknown" and kind not in TYPES:
            raise ValueError(f"gold {g.get('qid')!r} has type {kind!r}, which no candidates are defined for "
                             f"(they are for {list(TYPES)})")
        if kind != "unknown":
            by_type.setdefault(kind, []).append(g)
    summary, own = [dict(UNKNOWN)], []
    for kind, rows in by_type.items():
        found_summary, found_own = _HANDLERS[kind](rows)
        summary, own = [*summary, *found_summary], [*own, *found_own]
    unique, seen = [], set()
    for c in [*summary, *own]:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:MAX_CANDIDATES]
