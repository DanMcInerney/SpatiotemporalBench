"""The table systems' answers to `drop_zone_people` and `tangle_ahead` (a module of `table_answers.ANSWER_MODULES`; the
gold is `questions_hazard`), from the floor hulls the tracks keep (`Track.hulls`: the detection's `floor_hull`, set for
a person exactly when the sighting is evidence and for any other thing when it is recognisable). Neither answer reads
the mapper: the gold's coverage rules choose which points are asked, not what the answer is.

- `drop_zone_people`: the floor centroid (`approach_rule.centroid`) of the hull, in the ask's frame, of every person
  track whose hull has a point within `hazard_rule.ZONE_M` of the ask's camera floor point, as `positions`, to 4
  decimals and sorted. A person detected without a hull is not evidence and counts for nothing, as for the gold.
- `tangle_ahead`: the middle of every track of `hazard_rule.TANGLE_KINDS` whose last hull (its last recognisable
  sighting, which the store holds up to the ask) has its nearest point in reach (`hazard_rule.in_reach`) of the ask's
  camera floor point and heading, and which stood on the floor then: its centre's height less half its extent is within
  the robot's climb of the delivered base height of that step (`World.on_floor`'s rule). A track whose later detections
  put it `rules.MOVED_NO_M` or more from that place has moved since and is left out, as for the gold.
`current_view`'s store holds the latest frame alone, so it lists what that frame shows (`now` rows) and nothing it saw
earlier.
"""

from __future__ import annotations

import math

from . import grid
from .approach_rule import centroid
from .hazard_rule import TANGLE_KINDS, in_reach, in_zone
from .rules import MOVED_NO_M
from .table import Answerer, Store, Track
from .table_people import PERSON


def _answer(places: list[tuple[float, float]]) -> dict:
    return {"type": "positions", "items": [{"xy": [round(x, 4), round(y, 4)]} for x, y in sorted(places)]}


def drop_zone_people(store: Store, slots: dict, ctx: dict) -> dict:
    """The centroids of the hulls of the people in the ask's frame within `ZONE_M` of its camera floor point."""
    k, cam = ctx["now"]["k"], ctx["now"]["camera"]["pos"][:2]
    return _answer([centroid(t.hulls[k][1]) for t in store.tracks
                    if t.kind == PERSON and k in t.hulls and in_zone(t.hulls[k][1], cam)])


def _on_floor(store: Store, t: Track, j: int, climb_m: float) -> bool:
    """The track's centre at step `j`, less half its extent, is within `climb_m` of the delivered base height there."""
    z = next(s[3] for s in t.sightings if s[0] == j)
    return abs(z - t.size[2] / 2 - store.steps[j]["pose"]["pos"][2]) <= climb_m + grid.EPS


def tangle_ahead(store: Store, slots: dict, ctx: dict) -> dict:
    """The middles of the tangle things last recognised in reach (the module docstring)."""
    now, robot = ctx["now"], ctx["robot"]
    cam, yaw = now["camera"]["pos"][:2], math.radians(now["pose"]["yaw_deg"])
    places = []
    for t in store.tracks:
        if t.kind not in TANGLE_KINDS or not t.hulls:
            continue
        j = max(t.hulls)
        place, hull = t.hulls[j]
        later = [(s[1], s[2]) for s in t.sightings if s[0] > j]
        if in_reach(hull, cam, yaw) and _on_floor(store, t, j, robot.body.climb_m) \
                and all(math.dist((x, y), place) < MOVED_NO_M - grid.EPS for x, y in later):
            places.append(place)
    return _answer(places)


ANSWERERS: dict[str, Answerer] = {"drop_zone_people": drop_zone_people, "tangle_ahead": tangle_ahead}
