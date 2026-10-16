"""The strict parser's primitives (split from `answers`, which re-exports them): the schema violation `_Bad`, the finite
number, the coordinate list, the key check, the cell list and the polygon. They build no answer type, so this module
imports nothing of the package and `answers` can import it without a cycle."""

from __future__ import annotations

import math

MAX_CELLS = 1000  # a 6 x 5 m room is 120 cells, past the shared 100-item cap


class _Bad(Exception):
    """Internal: a schema violation, turned into Malformed at the public boundary."""


def _number(v: object, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise _Bad(f"{what} must be a finite number, got {v!r}")
    return float(v)


def _coords(v: object, n: int, what: str) -> tuple[float, ...]:
    if not isinstance(v, list | tuple) or len(v) != n:
        raise _Bad(f"{what} must be a list of {n} numbers, got {v!r}")
    return tuple(_number(c, what) for c in v)


def _keys(obj: dict, required: set[str], optional: set[str] = frozenset()) -> None:
    missing, extra = required - obj.keys(), obj.keys() - required - optional
    if missing or extra:
        raise _Bad(f"keys must be {sorted(required | optional)}; missing {sorted(missing)}, unexpected {sorted(extra)}")


def _cell_list(v: object, what: str) -> frozenset[tuple[int, int]]:
    """A list of at most `MAX_CELLS` cells as given (before duplicates collapse), each a list of two integers."""
    if not isinstance(v, list | tuple) or len(v) > MAX_CELLS:
        raise _Bad(f"{what} must be a list of at most {MAX_CELLS} cells")
    cells = []
    for c in v:
        if (not isinstance(c, list | tuple) or len(c) != 2
                or not all(isinstance(n, int) and not isinstance(n, bool) for n in c)):
            raise _Bad(f"each cell of {what} must be a list of two integers, got {c!r}")
        cells.append((c[0], c[1]))
    return frozenset(cells)


def _polygon(v: object, what: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(v, list | tuple) or len(v) < 3:
        raise _Bad(f"{what} must be a polygon of three or more points, got {v!r}")
    return tuple(_coords(p, 2, what) for p in v)


def is_polygon(v: object) -> bool:
    """Whether `v` is a polygon as a route gold's regions are: three or more [x, y] points of finite numbers."""
    try:
        _polygon(v, "polygon")
    except _Bad:
        return False
    return True
