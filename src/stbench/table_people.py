"""The table systems' answers to `closest_person` and `person_mem` (a module of `table_answers.ANSWER_MODULES`; the
gold is `questions_people`), each from the floor hulls the tracks keep of the people they detected (`Track.hulls`: the
detection's `floor_hull`, which `perception.py` sets for a person exactly when the sighting is evidence,
`person_seen.hull`, the function the gold calls). A person detected without a hull is not evidence and counts for
nothing, as for the gold.

- `closest_person`: the least, over the steps of the task the question names (`slots["task"]`, the case index) and the
  hulls of every person track at them, of the floor distance from that step's delivered camera to the hull's nearest
  point (`nearest_part.nearest_on_hull`), capped at `CAP_M` (10 m, "no one"), to 4 decimals. Any person counts: the
  question names none.
- `person_mem`: the hull of the latest sighting with a hull of any person track, its point nearest the camera of the
  step the ask is about (`ctx["now"]`), put in that camera's frame (`ego_frame.to_offset`); `unknown` when no person
  has one. The gold asks only where nothing seen puts the person elsewhere since that sighting, so this reads it as the
  gold does.
`current_view`'s store holds the latest frame alone, so a task or a sighting out of it is `no one` or `unknown` to it,
which is what the memory-needed gate holds under the floor. The kind word is the catalogs' one (`person`); the cap is
`questions_people.CAP_M`'s (a test holds the two equal).
"""

from __future__ import annotations

import math

from .answer_candidates import UNKNOWN
from .ego_frame import to_offset
from .nearest_part import nearest_on_hull
from .table import Answerer, Store

PERSON = "person"
CAP_M = 10.0


def _hulls(store: Store) -> list[tuple[int, list[list[float]]]]:
    """(step k, hull) of every sighting with a hull of every track of a person."""
    return [(k, hull) for t in store.tracks if t.kind == PERSON for k, (_, hull) in t.hulls.items()]


def closest_person(store: Store, slots: dict, ctx: dict) -> dict:
    """The least camera-to-hull floor distance over the task's steps, at most `CAP_M` (the module docstring)."""
    in_task = {s["k"] for s in store.task_steps(slots["task"])}
    nearest = CAP_M
    for k, hull in _hulls(store):
        if k in in_task:
            cam = store.steps[k]["camera"]["pos"][:2]
            nearest = min(nearest, math.dist(nearest_on_hull(hull, cam), cam))
    return {"type": "number", "value": round(nearest, 4)}


def person_mem(store: Store, slots: dict, ctx: dict) -> dict:
    """The offset, in the ask step's camera frame, of the latest hull's point nearest that camera; `unknown` with no
    hull."""
    hulls = _hulls(store)
    if not hulls:
        return dict(UNKNOWN)
    hull = max(hulls, key=lambda kh: kh[0])[1]
    now = ctx["now"]
    cam_xy, yaw = now["camera"]["pos"][:2], now["pose"]["yaw_deg"]
    ahead, left = to_offset(nearest_on_hull(hull, cam_xy), cam_xy, yaw)
    return {"type": "offset", "ahead_m": round(ahead, 4), "left_m": round(left, 4)}


ANSWERERS: dict[str, Answerer] = {"closest_person": closest_person, "person_mem": person_mem}
