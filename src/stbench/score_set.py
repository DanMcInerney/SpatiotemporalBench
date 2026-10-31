"""The credits that compare places, cells and regions: F1 over lists, and the room tests of `route` and `region`.

One F1 core serves two answer types. A `set` answer item (a change event) matches a gold item when it is within
`match_m` metres in xy and has the same change type; a `positions` answer item (a place) matches when it is within
`match_m` in xy, whatever the height. Answers are matched first: F1 only depends on how many pairs are matched, so the
best assignment is a maximum bipartite matching (augmenting paths); a greedy nearest-first choice can match fewer. Only
then are the answer items that matched nothing and lie within a don't-care region's radius dropped, so they are neither
true nor false positives; every other unmatched item is a false positive. A correct answer inside a region is therefore
a true positive (the radius never costs it its match). Where two answers could take the same gold item, the one inside a
region is left unmatched, so which one matches never changes the score. An answer point repeated twice matches one gold
point once, and the repeat is a false positive (unless it lies in a region).

A `graded_set` scores a `positions` answer per item rather than by count: the answer's items are matched one to one to
the gold's by maximum summed credit (`assignment`, the Hungarian method), each pair within `MATCH_REACH` match radii
earning the four quarters of `credit_kernel.graded` at its own gold item's resolution. The items outside every
don't-care circle are paired first; those inside a circle then take only the gold items left over, so a circle can add a
match to a list and never cost it one. A list item is a place (0.11.0): the pair earns its direction quarter within the
match radius M, as a single place does within R (docs/methodology.md rule 7), and not at any distance up to the reach;
the other three quarters are unchanged. A missed gold item costs a whole item and a false one (an unmatched answer
outside every don't-care circle) `fp_weight` of one. When there is nothing to list, the empty list is the one thing to
find, so false places dilute it as they dilute a list of one thing (0.11.0): one false place no longer scores as ten do.

A `cells` answer is a set of grid cells (`grid.cell`): F1 of the two sets, with the gold's don't-care cells dropped from
both sides (the list credits above drop unmatched answer items only).

A `route` answer is judged by the rooms it passes through, and a `region` position by whether it stands in a room away
from its walls: both reduce to the point-in-polygon test below. A point on a polygon's edge, within `EPS`, is inside it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from stbench.answers import Cells, DontCare, Positions, Route, RouteGold, SetAnswer
from stbench.credit_kernel import MATCH_REACH, graded

EPS = 1e-9

Item = tuple[tuple[float, float], str | None]  # (xy, change type); a place has none
Point = tuple[float, float]
Polygon = Sequence[Point]


def _matched(edges: list[list[int]], n_right: int) -> set[int]:
    """The left vertices of a maximum matching; `edges[i]` lists the right vertices left vertex i may take. A matched
    left vertex stays matched (an augmenting path only re-routes), so of two vertices that cannot both be matched the
    earlier one in `edges` is the one that is."""
    owner = [-1] * n_right

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] < 0 or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(edges)):
        augment(i, set())
    return {i for i in owner if i >= 0}


def _f1(answer: Sequence[Item], gold: Sequence[Item], dont_care: Sequence[DontCare], match_m: float) -> float:
    """F1 of `answer` against `gold`: answers are matched first, then the unmatched ones inside a `dont_care` region
    are dropped; with tp the matches and kept the matches plus the unmatched answers outside every region, F1 =
    2 tp / (kept + |gold|), and 1 if kept and gold are both empty."""

    def ok(a: Item, g: Item) -> bool:
        return a[1] == g[1] and math.dist(a[0], g[0]) <= match_m + EPS

    # answers outside every region come first, so of two answers that could take one gold item the one in a region is
    # left over and dropped
    marked = sorted(((a, any(math.dist(a[0], r.xy) <= r.radius + EPS for r in dont_care)) for a in answer),
                    key=lambda m: m[1])
    matched = _matched([[j for j, g in enumerate(gold) if ok(a, g)] for a, _ in marked], len(gold))
    kept = len(matched) + sum(not ignored for i, (_, ignored) in enumerate(marked) if i not in matched)
    if not kept and not gold:
        return 1.0
    return 2 * len(matched) / (kept + len(gold))


def set_f1(answer: SetAnswer, gold: SetAnswer, match_m: float = 0.5) -> float:
    """F1 of a `set` answer: items match on place and change type."""
    return _f1([(i.xy, i.change) for i in answer.items], [(i.xy, i.change) for i in gold.items], gold.dont_care,
               match_m)


def positions_f1(answer: Positions, gold: Positions, match_m: float = 0.5) -> float:
    """F1 of a `positions` answer: items match on their floor-plane place alone."""
    return _f1([(p.xy, None) for p in answer.items], [(p.xy, None) for p in gold.items], gold.dont_care, match_m)


def assignment(credit: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """The one-to-one pairs (row, column) of a rectangular matrix of credits with the largest summed credit: the
    Hungarian method (shortest augmenting paths), so no scipy. Every row of the shorter side is paired, with a
    credit of 0 where nothing better is left; ties go to the earlier row."""
    if not credit or not credit[0]:
        return []
    if len(credit) > len(credit[0]):
        return sorted((i, j) for j, i in assignment([list(column) for column in zip(*credit, strict=True)]))
    n, m = len(credit), len(credit[0])
    u, v, owner, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0], j0 = i, 0
        low, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], math.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cost = -credit[i0 - 1][j - 1] - u[i0] - v[j]
                if cost < low[j]:
                    low[j], way[j] = cost, j0
                if low[j] < delta:
                    delta, j1 = low[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    low[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    return sorted((owner[j] - 1, j - 1) for j in range(1, m + 1) if owner[j])


def graded_set(answer: Positions, gold: Positions, match_m: float, resolutions_m: Sequence[float],
               fp_weight: float) -> float:
    """Credit of a `positions` answer per item: `s = sum of the matched items' credit / (|gold| + fp_weight * |false|)`,
    and `1 / (1 + fp_weight * |false|)` for an empty gold, whose one thing is the empty list. A pair is an answer item
    within `MATCH_REACH * match_m` of a gold item in the floor plane, earning `credit_kernel.graded(d, resolutions_m[j],
    match_m, d <= match_m, False)` for gold item j, `d` their distance: a list item is a place, so a pair earns its
    direction quarter within the match radius, as a place does within R (methodology rule 7), and nowhere else. The
    answer items outside every don't-care circle of the gold are paired first, by the assignment of the largest summed
    credit, and one that is in no pair is false. The items inside a circle are then paired the same way with the gold
    items left over: they can earn a match but never take a gold item from an item outside, so adding an item inside a
    circle never lowers the credit, and an unmatched one is neither true nor false."""
    inside_circle = [any(math.dist(a.xy, r.xy) <= r.radius + EPS for r in gold.dont_care) for a in answer.items]
    outside = [a for a, ig in zip(answer.items, inside_circle, strict=True) if not ig]
    circled = [a for a, ig in zip(answer.items, inside_circle, strict=True) if ig]
    reach = MATCH_REACH * match_m + EPS

    def credits(items: Sequence[Point], golds: Sequence[int]) -> list[list[float]]:
        return [[graded(d, resolutions_m[j], match_m, d <= match_m + EPS, False)
                 if (d := math.dist(a, gold.items[j].xy)) <= reach else 0.0 for j in golds] for a in items]

    every = range(len(gold.items))
    credit = credits([a.xy for a in outside], every)
    pairs = [(i, j) for i, j in assignment(credit) if credit[i][j] > 0]
    total, taken = sum(credit[i][j] for i, j in pairs), {j for _, j in pairs}
    left = [j for j in every if j not in taken]
    credit = credits([a.xy for a in circled], left)
    total += sum(credit[i][k] for i, k in assignment(credit) if credit[i][k] > 0)
    false = len(outside) - len(pairs)
    if not gold.items:
        return 1 / (1 + fp_weight * false)
    return total / (len(gold.items) + fp_weight * false)


def cells_f1(answer: Cells, gold: Cells) -> float:
    """F1 of a `cells` answer, the gold's don't-care cells dropped from both sides: A = answer - DC, G = gold - DC;
    1 if both are empty, 0 if exactly one is, else 2|A & G| / (|A| + |G|)."""
    a, g = answer.items - gold.dont_care, gold.items - gold.dont_care
    if not a and not g:
        return 1.0
    if not a or not g:
        return 0.0
    return 2 * len(a & g) / (len(a) + len(g))


def boundary_distance(p: Point, polygon: Polygon) -> float:
    """The distance from `p` to the nearest edge of `polygon`."""
    best = math.inf
    for a, b in zip(polygon, [*polygon[1:], polygon[0]], strict=True):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
        best = min(best, math.dist(p, (a[0] + t * dx, a[1] + t * dy)))
    return best


def inside(p: Point, polygon: Polygon) -> bool:
    """Whether `p` is inside `polygon`, its boundary included (within `EPS`)."""
    if boundary_distance(p, polygon) <= EPS:
        return True
    crossed = False
    for a, b in zip(polygon, [*polygon[1:], polygon[0]], strict=True):
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            crossed = not crossed
    return crossed


def region_credit(xy: Point, polygon: Polygon, erode_m: float) -> float:
    """1 if `xy` is inside `polygon` and at least `erode_m` from its boundary (within `EPS`), else 0. A distance test:
    the eroded polygon is never built, so no float noise moves a point across it."""
    return 1.0 if inside(xy, polygon) and boundary_distance(xy, polygon) >= erode_m - EPS else 0.0


def route_credit(answer: Route, gold: RouteGold) -> float:
    """1 if the rooms the route passes through, in order, are one of the gold's `valid` sequences, else 0.

    Each point belongs to the first region (in the gold's order) that holds it, boundary included. A point in no region
    is dropped (a doorway lies between room polygons), then consecutive repeats collapse. An empty sequence scores 0."""
    rooms: list[str] = []
    for point in answer.points:
        name = next((n for n, polygon in gold.regions if inside(point, polygon)), None)
        if name is not None and (not rooms or rooms[-1] != name):
            rooms.append(name)
    return 1.0 if rooms and tuple(rooms) in gold.valid else 0.0
