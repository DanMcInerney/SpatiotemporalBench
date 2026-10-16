"""The typed answer schema (V0 §8) and its strict parser.

An answer is the inner object of a response `{"qid", "answer": {...}}`. Every answer names its `type`:

| type       | fields                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| `position` | exactly one of `xy` ([x, y]) or `xyz` ([x, y, z]), metres, map frame    |
| `offset`   | `ahead_m` and `left_m`: metres ahead of the camera and to its left, along the floor       |
| `time`     | `clock`: local ISO clock, `YYYY-MM-DDTHH:MM[:SS[.ffffff]]`, no zone     |
| `yesno`    | `value`: a JSON boolean                                                 |
| `count`    | `value`: a non-negative JSON integer                                    |
| `number`   | `value`: a finite JSON number (metres for a distance)                   |
| `set`      | `items`: [{`xy`: [x, y], `change`: added, removed or moved}], at most 100 |
| `positions`| `items`: [{`xy`: [x, y]} or {`xyz`: [x, y, z]}], at most 100            |
| `cells`    | `items`: [[i, j]] answer-grid cells (`grid.py`), integers, at most 1,000 |
| `route`    | `points`: [[x, y]], 1 to 50 map points, in the order they are passed    |
| `heading`  | `deg_left`: a finite number from -180 to 180, degrees left of the way the robot faces |
| `unknown`  | nothing else                                                            |

Parsing never raises on bad input: it returns `Malformed(reason)`, which the scorer credits with 0. Unknown keys,
wrong JSON types and non-finite numbers are all malformed. Gold uses the same shapes (`parse_gold`), and a gold `set`
or `positions` may also carry `dont_care`: [{`xy`, `radius`}] regions, a gold `cells` a `dont_care` list of cells; an
answer may carry neither. A `positions` item has no `change` (it is a place; a `set` item is an event at a place), so
each shape is malformed as the other. A gold `route` is a different object under the same type name: `regions` (name to
polygon) and `valid` (the region sequences that score), with no `points`.

`TYPES` is the one list of every type a question's `answer_type` may name, and `PROMPT_SHAPES` is what the reference
agent's prompt shows for each type its job asks (`brief.answer_types`). A parse always knows all of `TYPES`: with the
question's `answer_type`, an answer of another type (`unknown` aside) is malformed with the mismatch reason, so there
is no Recall/Perceive split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .answers_parse import MAX_CELLS, _Bad, _cell_list, _coords, _keys, _number, _polygon, is_polygon

__all__ = ["MAX_CELLS", "is_polygon"]   # the parser's primitives live in `answers_parse`; these two are public here

TYPES = ("position", "offset", "time", "yesno", "count", "number", "set", "positions", "cells", "route",
         "heading")  # the one list
CHANGES = ("added", "removed", "moved")
MAX_SET_ITEMS = 100
MAX_ROUTE_POINTS = 50
MAX_HEADING_DEG = 180.0

# What the reference agent's prompt shows for each type, beside the parser that reads it: every type of `TYPES`, and
# `unknown`. `llm_render` and the tool server (`codex_mcp`, for a repair) read this table and keep no copy.
PROMPT_SHAPES = {
    "position": '{"type": "position", "xy": [x, y]} (metres, map frame; "xyz": [x, y, z] also works)',
    "offset": ('{"type": "offset", "ahead_m": a, "left_m": l} (metres from the camera along the floor: a ahead of it '
               'along the robot\'s heading, l to its left; negative is behind it or to its right)'),
    "time": ('{"type": "time", "clock": "YYYY-MM-DDTHH:MM:SS.s"} (the local clock time as the frames give it, to the '
             'tenth of a second; no time zone)'),
    "yesno": '{"type": "yesno", "value": true} (true or false)',
    "count": '{"type": "count", "value": N} (N is a whole number, 0 or more)',
    "number": '{"type": "number", "value": X} (a number; metres for a distance)',
    "set": ('{"type": "set", "items": [{"xy": [x, y], "change": "added"}]} (one item per change; change is added, '
            'removed or moved; xy is a map position in metres; an empty list means nothing changed)'),
    # the list's form alone: what a question counts is that question's own text (`questions_task`, `questions_hazard`)
    "positions": ('{"type": "positions", "items": [{"xy": [x, y]}]} (one item per place the question asks for, each a '
                  'map position in metres; "xyz": [x, y, z] also works for an item; an empty list means there are '
                  'none)'),
    "cells": ('{"type": "cells", "items": [[i, j]]} (each cell is [i, j] with i = floor(x / 0.5) and '
              'j = floor(y / 0.5), x and y in metres in the map frame; an empty list means none)'),
    "route": ('{"type": "route", "points": [[x, y]]} (1 to 50 map positions in metres, in the order you would pass '
              'them)'),
    "heading": ('{"type": "heading", "deg_left": x} (degrees to the left of the way you face now, from -180 to 180; '
                'to your right is negative)'),
    "unknown": '{"type": "unknown"} (you cannot know)',
}

_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?")


@dataclass(frozen=True)
class Position:
    """A point in the map frame; `coords` has two entries (xy) or three (xyz)."""

    coords: tuple[float, ...]

    @property
    def xy(self) -> tuple[float, float]:
        return (self.coords[0], self.coords[1])

    @property
    def has_z(self) -> bool:
        return len(self.coords) == 3


@dataclass(frozen=True)
class Offset:
    """A place in the camera's ground frame: `ahead_m` metres along its heading, `left_m` metres to its left, measured
    along the floor (the height is ignored). Negative is behind or to the right (`ego_frame`)."""

    ahead_m: float
    left_m: float


@dataclass(frozen=True)
class Time:
    """A naive local clock time."""

    clock: datetime


@dataclass(frozen=True)
class YesNo:
    value: bool


@dataclass(frozen=True)
class Count:
    value: int


@dataclass(frozen=True)
class Number:
    value: float


@dataclass(frozen=True)
class SetItem:
    xy: tuple[float, float]
    change: str


@dataclass(frozen=True)
class DontCare:
    """A gold-only region where answers are neither true nor false positives."""

    xy: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class SetAnswer:
    items: tuple[SetItem, ...]
    dont_care: tuple[DontCare, ...] = ()


@dataclass(frozen=True)
class Positions:
    """A list of places (the Perceive set): no change types; a gold may carry don't-care regions."""

    items: tuple[Position, ...]
    dont_care: tuple[DontCare, ...] = ()


@dataclass(frozen=True)
class Cells:
    """A set of answer-grid cells: cell (i, j) covers [0.5i, 0.5i + 0.5) x [0.5j, 0.5j + 0.5) (`grid.cell`)."""

    items: frozenset[tuple[int, int]]
    dont_care: frozenset[tuple[int, int]] = frozenset()  # gold only


@dataclass(frozen=True)
class Route:
    """A route answer: 1 to `MAX_ROUTE_POINTS` map points, in the order they are passed."""

    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RouteGold:
    """A route gold (its type name is `route` too): the named regions and the region sequences that score."""

    regions: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]  # (name, polygon), in the gold's key order
    valid: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Heading:
    """A direction to turn in place toward: `deg_left` degrees to the left of the way the robot faces, -180 to 180
    (to its right is negative)."""

    deg_left: float


@dataclass(frozen=True)
class Unknown:
    pass


@dataclass(frozen=True)
class Malformed:
    """What `parse_answer` returns for input that violates the schema."""

    reason: str


Answer = (Position | Offset | Time | YesNo | Count | Number | SetAnswer | Positions | Cells | Route | RouteGold
          | Heading | Unknown | Malformed)

_TYPE_NAMES = {Position: "position", Offset: "offset", Time: "time", YesNo: "yesno", Count: "count", Number: "number",
               SetAnswer: "set", Positions: "positions", Cells: "cells", Route: "route", RouteGold: "route",
               Heading: "heading", Unknown: "unknown", Malformed: "malformed"}


def answer_type_of(answer: Answer) -> str:
    """The schema type name of a parsed answer (`malformed` for a Malformed marker)."""
    return _TYPE_NAMES[type(answer)]


def _place(obj: dict, head: set[str]) -> Position:
    """A point: exactly one of `xy` or `xyz`, beside the keys of `head` (an answer's `type`, nothing for an item)."""
    if "xyz" in obj:
        _keys(obj, head | {"xyz"})
        return Position(_coords(obj["xyz"], 3, "xyz"))
    _keys(obj, head | {"xy"})
    return Position(_coords(obj["xy"], 2, "xy"))


def _position(obj: dict) -> Position:
    return _place(obj, {"type"})


def _offset(obj: dict) -> Offset:
    _keys(obj, {"type", "ahead_m", "left_m"})
    return Offset(_number(obj["ahead_m"], "ahead_m"), _number(obj["left_m"], "left_m"))


def _heading(obj: dict) -> Heading:
    _keys(obj, {"type", "deg_left"})
    deg = _number(obj["deg_left"], "deg_left")
    if abs(deg) > MAX_HEADING_DEG:
        raise _Bad(f"deg_left must be from -{MAX_HEADING_DEG:g} to {MAX_HEADING_DEG:g}, got {deg}")
    return Heading(deg)


def _clock(obj: dict) -> Time:
    _keys(obj, {"type", "clock"})
    text = obj["clock"]
    if not isinstance(text, str) or not _CLOCK.fullmatch(text):
        raise _Bad(f"clock must be a local ISO time 'YYYY-MM-DDTHH:MM:SS' with no zone, got {text!r}")
    try:
        return Time(datetime.fromisoformat(text))
    except ValueError as e:
        raise _Bad(f"clock {text!r} is not a valid time: {e}") from None


def _value(obj: dict, make: type) -> Answer:
    _keys(obj, {"type", "value"})
    v = obj["value"]
    if make is YesNo:
        if not isinstance(v, bool):
            raise _Bad(f"value must be true or false, got {v!r}")
        return YesNo(v)
    if make is Count:
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise _Bad(f"value must be a non-negative integer, got {v!r}")
        return Count(v)
    return Number(_number(v, "value"))


def _items(obj: dict, gold: bool) -> list:
    """The `items` of a list-shaped answer (`set` or `positions`); `dont_care` only on a gold."""
    _keys(obj, {"type", "items"}, {"dont_care"} if gold else set())
    items = obj["items"]
    if not isinstance(items, list | tuple) or len(items) > MAX_SET_ITEMS:
        raise _Bad(f"items must be a list of at most {MAX_SET_ITEMS} entries")
    return list(items)


def _item(it: object) -> dict:
    """An item of a list-shaped answer, checked as it is read, one at a time (the order of Release 0.2's reasons)."""
    if not isinstance(it, dict):
        raise _Bad(f"each item must be an object, got {it!r}")
    return it


def _regions(obj: dict) -> tuple[DontCare, ...]:
    regions = []
    for r in obj.get("dont_care", ()):
        if not isinstance(r, dict):
            raise _Bad(f"each dont_care region must be an object, got {r!r}")
        _keys(r, {"xy", "radius"})
        radius = _number(r["radius"], "radius")
        if radius <= 0:
            raise _Bad(f"radius must be positive, got {radius}")
        regions.append(DontCare(_coords(r["xy"], 2, "xy"), radius))
    return tuple(regions)


def _set(obj: dict, gold: bool) -> SetAnswer:
    parsed = []
    for it in _items(obj, gold):
        _keys(_item(it), {"xy", "change"})
        if it["change"] not in CHANGES:
            raise _Bad(f"change must be one of {list(CHANGES)}, got {it['change']!r}")
        parsed.append(SetItem(_coords(it["xy"], 2, "xy"), it["change"]))
    return SetAnswer(tuple(parsed), _regions(obj))


def _positions(obj: dict, gold: bool) -> Positions:
    # an item is exactly `xy` or `xyz`: a `change` (a set's item), a `type` or anything else is malformed
    return Positions(tuple(_place(_item(it), set()) for it in _items(obj, gold)), _regions(obj))


def _cells(obj: dict, gold: bool) -> Cells:
    _keys(obj, {"type", "items"}, {"dont_care"} if gold else set())
    items, dont_care = _cell_list(obj["items"], "items"), _cell_list(obj.get("dont_care", []), "dont_care")
    if items & dont_care:
        raise _Bad(f"cells {sorted(items & dont_care)} are in both items and dont_care")
    return Cells(items, dont_care)


def _route_gold(obj: dict) -> RouteGold:
    _keys(obj, {"type", "regions", "valid"})
    regions, valid = obj["regions"], obj["valid"]
    if not isinstance(regions, dict) or not regions:
        raise _Bad(f"regions must be a non-empty object of name to polygon, got {regions!r}")
    polygons = tuple((name, _polygon(poly, f"region {name!r}")) for name, poly in regions.items())
    if not isinstance(valid, list | tuple) or not valid:
        raise _Bad(f"valid must be a non-empty list of region-name lists, got {valid!r}")
    sequences = []
    for seq in valid:
        if not isinstance(seq, list | tuple) or not seq:
            raise _Bad(f"each valid sequence must be a non-empty list of region names, got {seq!r}")
        for name in seq:
            if not isinstance(name, str) or name not in regions:
                raise _Bad(f"valid names {name!r}, which is not a region")
        sequences.append(tuple(seq))
    return RouteGold(polygons, tuple(sequences))


def _route(obj: dict, gold: bool) -> Route | RouteGold:
    if gold:
        return _route_gold(obj)
    _keys(obj, {"type", "points"})
    points = obj["points"]
    if not isinstance(points, list | tuple) or not 1 <= len(points) <= MAX_ROUTE_POINTS:
        raise _Bad(f"points must be a list of 1 to {MAX_ROUTE_POINTS} points")
    return Route(tuple(_coords(p, 2, "point") for p in points))


def _parse(obj: object, gold: bool) -> Answer:
    if not isinstance(obj, dict):
        raise _Bad(f"an answer must be a JSON object, got {type(obj).__name__}")
    kind = obj.get("type")
    if kind == "position":
        return _position(obj)
    if kind == "offset":
        return _offset(obj)
    if kind == "time":
        return _clock(obj)
    if kind == "yesno":
        return _value(obj, YesNo)
    if kind == "count":
        return _value(obj, Count)
    if kind == "number":
        return _value(obj, Number)
    if kind == "set":
        return _set(obj, gold)
    if kind == "positions":
        return _positions(obj, gold)
    if kind == "cells":
        return _cells(obj, gold)
    if kind == "route":
        return _route(obj, gold)
    if kind == "heading":
        return _heading(obj)
    if kind == "unknown":
        _keys(obj, {"type"})
        return Unknown()
    raise _Bad(f"type must be one of {[*TYPES, 'unknown']}, got {kind!r}")


def parse_answer(obj: object, answer_type: str | None = None) -> Answer:
    """Validate `obj` strictly. Returns a Malformed marker, never raises, for bad input.

    Every type of `TYPES` parses. With `answer_type` (the question's, one of `TYPES`), any answer of another type
    except `unknown` is Malformed with the mismatch reason.
    """
    if answer_type is not None and answer_type not in TYPES:
        raise ValueError(f"answer_type must be one of {list(TYPES)}, got {answer_type!r}")
    try:
        answer = _parse(obj, gold=False)
    except _Bad as e:
        return Malformed(str(e))
    got = answer_type_of(answer)
    if answer_type is not None and got not in (answer_type, "unknown"):
        return Malformed(f"answer type {got!r} does not match the question's answer_type {answer_type!r}")
    return answer


def parse_gold(obj: object) -> Answer:
    """Parse a gold value (same shapes as answers; a set, positions or cells may carry `dont_care`, and a route is
    `regions` and `valid`). Raises ValueError if invalid."""
    try:
        return _parse(obj, gold=True)
    except _Bad as e:
        raise ValueError(f"invalid gold value: {e}") from None


def _with_regions(out: dict, dont_care: tuple[DontCare, ...]) -> dict:
    if dont_care:
        out["dont_care"] = [{"xy": list(r.xy), "radius": r.radius} for r in dont_care]
    return out


def _coords_json(p: Position) -> dict:
    return {"xyz" if p.has_z else "xy": list(p.coords)}


def to_json(answer: Answer) -> dict:
    """The JSON object for a parsed answer; `parse_answer(to_json(a)) == a`. Malformed has no JSON form."""
    match answer:
        case Position():
            return {"type": "position", **_coords_json(answer)}
        case Offset(ahead_m, left_m):
            return {"type": "offset", "ahead_m": ahead_m, "left_m": left_m}
        case Time(clock):
            return {"type": "time", "clock": clock.isoformat()}
        case YesNo(value) | Count(value) | Number(value):
            return {"type": _TYPE_NAMES[type(answer)], "value": value}
        case SetAnswer(items, dont_care):
            out: dict = {"type": "set", "items": [{"xy": list(i.xy), "change": i.change} for i in items]}
            return _with_regions(out, dont_care)
        case Positions(items, dont_care):
            return _with_regions({"type": "positions", "items": [_coords_json(p) for p in items]}, dont_care)
        case Cells(items, dont_care):
            cells: dict = {"type": "cells", "items": [list(c) for c in sorted(items)]}
            if dont_care:
                cells["dont_care"] = [list(c) for c in sorted(dont_care)]
            return cells
        case Route(points):
            return {"type": "route", "points": [list(p) for p in points]}
        case RouteGold(regions, valid):
            return {"type": "route", "regions": {name: [list(p) for p in polygon] for name, polygon in regions},
                    "valid": [list(sequence) for sequence in valid]}
        case Heading(deg_left):
            return {"type": "heading", "deg_left": deg_left}
        case Unknown():
            return {"type": "unknown"}
    raise ValueError(f"cannot serialise {answer!r}")
