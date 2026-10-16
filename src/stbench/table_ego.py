"""The table systems' answers to `ego_now` and `ego_mem` (a module of `table_answers.ANSWER_MODULES`; the gold is
`questions_ego`), each from the floor hulls the tracks keep of the sightings (`Track.hulls`: the detection's
`floor_hull`, which `perception.py` computes with `nearest_part.floor_hull`, the function the gold uses), put in the
camera frame of the step the ask is about (`ctx["now"]`: the delivered camera position and the base's heading,
`ego_frame`).

The question asks for the point of the thing nearest the camera, measured along the floor, and the gold is the point of
the floor hull nearest the camera's floor point (`nearest_part.nearest_on_hull`), a hull edge as often as a vertex.
- `ego_now`: the description's sighting at the ask step itself. Exactly one is the thing; none (it is not in the frame
  the detections came from) or several (a twin is detected beside it, which the table cannot tell apart) give
  `unknown`, and so does a sighting with no hull (not recognisable, which the gold never asks).
- `ego_mem`: the hull of the union of the hulls of the track's sightings at the place of its latest one
  (`nearest_part.UNCHANGED_M`), the track being the description's with the latest hulled sighting; `unknown` when it
  has none. `current_view`'s store holds the latest frame alone, so a thing out of that frame is `unknown` to it, which
  is what the memory-needed gate holds under the floor.
The gold asks only what the table reads as the gold does: a name no other sighted thing could be taken for, a thing
unmoved from its last recognisable sighting, and for `ego_mem` none in the recent view, so the oracle and `memory_table`
read the same hulls and score 1. Offsets are rounded to 4 places, as the gold's are.
"""

from __future__ import annotations

import math

from .answer_candidates import UNKNOWN
from .ego_frame import to_offset
from .nearest_part import UNCHANGED_M, nearest_on_hull, union_hull
from .table import Answerer, Store, Track


def _answer(hull: list[list[float]], ctx: dict) -> dict:
    """The offset of the hull's point nearest the camera of the step the ask is about."""
    now = ctx["now"]
    cam_xy, yaw = now["camera"]["pos"][:2], now["pose"]["yaw_deg"]
    ahead, left = to_offset(nearest_on_hull(hull, cam_xy), cam_xy, yaw)
    return {"type": "offset", "ahead_m": round(ahead, 4), "left_m": round(left, 4)}


def _tracks(store: Store, slots: dict) -> list[Track]:
    """The tracks the description names."""
    return list(store.matching(slots["kind"], slots["attributes"]))


def ego_now(store: Store, slots: dict, ctx: dict) -> dict:
    """The offset of the hull of the description's one sighting at the ask step; `unknown` with none or more than one,
    or when that sighting has no hull."""
    k = ctx["now"]["k"]
    here = [(t, s) for t in _tracks(store, slots) for s in t.sightings if s[0] == k]
    if len(here) != 1 or k not in here[0][0].hulls:
        return dict(UNKNOWN)
    return _answer(here[0][0].hulls[k][1], ctx)


def ego_mem(store: Store, slots: dict, ctx: dict) -> dict:
    """The offset of the union of the hulls of the sightings at the place of the latest hulled one; `unknown` when no
    track the description names has one."""
    hulled = [t for t in _tracks(store, slots) if t.hulls]
    if not hulled:
        return dict(UNKNOWN)
    track = max(hulled, key=lambda t: max(t.hulls))
    last = track.hulls[max(track.hulls)][0]
    here = [hull for centre, hull in track.hulls.values() if math.dist(centre, last) < UNCHANGED_M]
    return _answer(union_hull(here), ctx)


ANSWERERS: dict[str, Answerer] = {"ego_now": ego_now, "ego_mem": ego_mem}
