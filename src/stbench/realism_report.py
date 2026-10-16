"""The arithmetic of `stbench realism report` (M1's R.2, the plan's section 4.10 item 7): what a look
is, as numbers, against targets fixed before any render. Pure: arrays, lists and dicts in, numbers out; nothing here
reads a file, so a test can hand-build every input (`tests/test_realism_report.py`). `realism_data` reads a dataset and
a probe folder into these inputs and `realism_cmd` writes `realism.json` and `realism.md`.

A measure is a `Row`: its name, the value, the target as words, and a status: "ok" or "outside" for a measure that
has a target, "reported" for one that has none (a number to look at beside the references), "no data" where the
inputs were not given. The targets are `TARGETS` (and `LAMP_ROOM_BOUNDS`, a room's own band); `TRIM` is what is no
object in a count. The noise is `realism_noise`'s and the crop check `realism_crops`'s."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

TRIM = ("baseboard", "door_casing", "window_sill", "backsplash")  # trim is no object: Hypersim's count leaves it out
DEBUG_BASE = 1000  # the pass index of the first dressing piece in a probe's index frames (entities are below it)
TARGETS = {  # (low, high): a number inside it is "ok"; None for a side that is open, or for a measure with no target
    "entities": (160, 250), "lamps only, a room's median lux": (100, 300),
    "lamps only, the worktop's lux": (300, 750), "lamps' colour temperature, K": (2700, 3500),
    # one view at 1.30 m covers about a fifth of a room: its object count is reported, not a target (DECISIONS 10-06)
    "objects, standing frames": None,
    # the vacuum's own camera: the real Roomba j7 frames' mean sRGB luminance, the flat fills left out, median 0.30
    "exposure, mean sRGB luminance": (0.22, 0.36), "luminance floor, share of a kind under RECOG_LUM": (None, 0.05),
    # a window in a daylight frame over the room's median: a sky-lit window is 5-6 stops over an interior wall
    "window over the room's median, stops": (4.5, None),
}
LAMP_ROOM_BOUNDS = {"kitchen": (150, 500)}  # a room's own lamps-alone band where it differs (kitchens are lit higher)
FLOOR_SHARE = (20, 45)  # floor clutter a script, the quota (VISION: 0.5-1.2 a m2 of the apartment's 39 m2)
CROP_RIGHT = 0.8  # of a kind's crops that must be named right, kind and colour (4 of 5)
WINDOW_PERCENTILE = 90.0  # a window's luminance: this percentile of the linear luma inside its sill's box
WINDOW_MIN_PX = 200  # the pixels of a window's sill that make the window "in view"


@dataclass(frozen=True)
class Row:
    name: str
    value: object
    target: str
    status: str  # ok | outside | reported | no data

    def to_dict(self) -> dict:
        return asdict(self)


def within(value: float, bounds: tuple[float | None, float | None]) -> bool:
    lo, hi = bounds
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def words(bounds: tuple[float | None, float | None]) -> str:
    lo, hi = bounds
    if lo is not None and hi is not None:
        return f"{lo} to {hi}"
    return f"at least {lo}" if hi is None else f"at most {hi}"


def median(values: Iterable[float]) -> float | None:
    vs = list(values)
    return float(statistics.median(vs)) if vs else None


def spread(values: Sequence[float]) -> dict | None:
    """Minimum, median and maximum of a list, or None for an empty one."""
    return {"min": min(values), "median": float(statistics.median(values)), "max": max(values)} if values else None


# -- the world ---------------------------------------------------------------------------------------------------
def entity_counts(entities: Iterable[tuple[str, bool]]) -> dict:
    """What a script holds, by mobility: `entities` are (mobility, on the floor) pairs. `total`, each mobility's count
    and `floor_clutter`, the clutter that lies on the floor with no support."""
    rows = list(entities)
    out = dict(Counter(m for m, _ in rows))
    out["total"] = len(rows)
    out["floor_clutter"] = sum(1 for m, floor in rows if m == "clutter" and floor)
    return out


def objects_in_frame(px: dict[int, int], dressing: dict[int, str], min_px: int, trim: Sequence[str] = TRIM) -> dict:
    """The objects of one frame from its debug index: entities (pass index under `DEBUG_BASE`, structure is 0) and
    dressing pieces (`dressing` maps a piece's index to its class) of at least `min_px` pixels, the pieces of trim left
    out. `entities` and `pieces` count each, `objects` is their sum."""
    entities = sum(1 for i, n in px.items() if 0 < i < DEBUG_BASE and n >= min_px)
    pieces = sum(1 for i, n in px.items() if i >= DEBUG_BASE and n >= min_px and dressing.get(i) not in trim)
    return {"entities": entities, "pieces": pieces, "objects": entities + pieces}


# -- light -------------------------------------------------------------------------------------------------------
def light_table(cards: dict) -> dict:
    """`cards.json` (the probe's grey cards: each `{room or name, lux, r_g, b_g}`) as medians: lamps-only lux a room,
    the worktop's, daylight-only lux a room, and the cast (red over green, blue over green) of the lamps' light. A card
    that no light reached (0 lux: inside a solid) has no colour, a ratio of two zeros, so it counts in the lux medians
    and not in the cast."""
    def by_room(points: list[dict]) -> dict[str, float]:
        rooms: dict[str, list[float]] = defaultdict(list)
        for p in points:
            rooms[p["room"]].append(p["lux"])
        return {r: float(statistics.median(v)) for r, v in sorted(rooms.items())}

    lamps, day, top = cards.get("lamps", []), cards.get("daylight", []), cards.get("worktop", [])
    return {"lamps": by_room(lamps), "daylight": by_room(day), "worktop": median(p["lux"] for p in top),
            "cast": {k: median(p[k] for p in lamps if math.isfinite(p[k])) for k in ("r_g", "b_g")}}


def lamp_cct(lamps: Sequence[dict]) -> dict | None:
    """The colour temperatures the rig record gives the lamps (`render_log.json`'s `lighting.lamps`)."""
    ks = [float(lamp["cct_k"]) for lamp in lamps]
    return {"min": min(ks), "max": max(ks)} if ks else None


# -- exposure, clipping, noise -----------------------------------------------------------------------------------
def by_room(frames: Sequence[dict]) -> dict[str, dict]:
    """Per room, over frames `{room, ev, lum, clipped, crushed}` (the frame log's numbers and the mean sRGB luminance of
    the JPEG): the median EV and luminance, the mean share clipped and crushed, and the number of frames."""
    rooms: dict[str, list[dict]] = defaultdict(list)
    for f in frames:
        rooms[f["room"]].append(f)
    def mean(v: list[dict], key: str) -> float:
        return float(np.mean([f[key] for f in v]))

    return {r: {"frames": len(v), "ev": median(f["ev"] for f in v), "lum": median(f["lum"] for f in v),
                "clipped": mean(v, "clipped"), "crushed": mean(v, "crushed")} for r, v in sorted(rooms.items())}


def stops(linear_lum: np.ndarray) -> dict:
    """A frame's range in stops of the scene-linear luminance: the brightest 0.1% (a window, a sun patch) over the
    median (the room) as `bright`, and the median over the 5th percentile (the dark under furniture) as `dark`."""
    lo, mid, hi = np.percentile(linear_lum, [5.0, 50.0, 99.9])
    eps = 1e-6
    lo, mid, hi = max(lo, eps), max(mid, eps), max(hi, eps)
    return {"bright": float(np.log2(hi / mid)), "dark": float(np.log2(mid / lo))}


def window_stops(luma: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """How far a window stands over its room, in stops of the scene-linear luma: `WINDOW_PERCENTILE` of the luma inside
    `box` (u0, v0, u1, v1, the box of the window's sill, with the glass in it) over the median of the whole frame."""
    u0, v0, u1, v1 = box
    eps = 1e-6
    return float(np.log2(max(np.percentile(luma[v0:v1, u0:u1], WINDOW_PERCENTILE), eps) / max(np.median(luma), eps)))


# -- the luminance floor -----------------------------------------------------------------------------------------
def luminance_floor(sightings: Sequence[dict], kinds: Iterable[str], recog_lum: float, min_px: int = 1024) -> dict:
    """The 1st, 2nd and 5th percentiles of `lum` over the sightings (`{kind, px, lum}`) of at least `min_px` pixels of
    the given kinds, the candidate `RECOG_LUM` (the midpoint of the 2nd and the 5th), and per kind how many sightings
    there were and which share of them is under `recog_lum`."""
    wanted = set(kinds)
    seen = [s for s in sightings if s["kind"] in wanted and s["px"] >= min_px]
    if not seen:
        return {"sightings": 0, "percentiles": None, "candidate": None, "per_kind": {}, "recog_lum": recog_lum}
    lum = np.array([s["lum"] for s in seen])
    p1, p2, p5 = (float(v) for v in np.percentile(lum, [1.0, 2.0, 5.0]))
    per_kind = {k: {"sightings": len(v), "share_under": sum(1 for s in v if s["lum"] < recog_lum) / len(v)}
                for k in sorted(wanted) if (v := [s for s in seen if s["kind"] == k])}
    return {"sightings": len(seen), "percentiles": {"p1": p1, "p2": p2, "p5": p5}, "candidate": (p2 + p5) / 2,
            "per_kind": per_kind, "recog_lum": recog_lum}


# -- the report --------------------------------------------------------------------------------------------------
def row(name: str, value: object, bounds: tuple[float | None, float | None] | None, number: float | None = None) -> Row:
    """A row; `number` is what the target is tested on (the value itself when it is a number)."""
    n = value if number is None and isinstance(value, int | float) else number
    if n is None:
        return Row(name, value, "none" if bounds is None else words(bounds), "no data" if value is None else "reported")
    return Row(name, value, "none" if bounds is None else words(bounds),
               "reported" if bounds is None else "ok" if within(n, bounds) else "outside")


def render_md(rows: Sequence[Row], title: str = "Realism report") -> str:
    """The rows as a markdown table, a line to a measure."""
    def cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v) if not isinstance(v, dict | list) else str(v).replace("|", "/")

    lines = [f"# {title}", "", "| measure | value | target | status |", "| --- | --- | --- | --- |"]
    lines += [f"| {r.name} | {cell(r.value)} | {r.target} | {r.status} |" for r in rows]
    return "\n".join(lines) + "\n"
