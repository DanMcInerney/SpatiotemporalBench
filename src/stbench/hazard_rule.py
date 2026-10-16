"""The two list questions of the hazards (0.10.0, docs/families/drop_zone_people.md and tangle_ahead.md): which people
stand within `ZONE_M` of the point below a drone, and which things a vacuum's brush could catch lie within `REACH_M` of
it, ahead or to its sides.

One owner of the tests and the kinds, called by the gold (`questions_hazard`) and the table systems (`table_hazard`), so
the two cannot differ by a constant. Both read a floor hull (what the oracle channel sends of an evidence sighting of a
person, or of a recognisable thing: `nearest_part.floor_hull`) and the camera's floor point, never a picture. It is
neutral: `descriptions`, `nearest_part` and the standard library.

**The kinds a brush catches** (`CATCH_KINDS`) are cords and cloth. Those whose instances recur at the same spot from
script to script are a layout prior, not a sighting (`SPOT_KINDS`): a system that learned where the bath mat lies would
list it without a camera. The test, on the dev root's 16 vacuum scripts (`docs/families/tangle_ahead.md` has the numbers
per kind): a kind recurs when at least half of its floor instances lie within one free-space cell (0.10 m,
`grid.FINE_M`) of an instance of the same kind in another script of the map. The brief's radius is 0.25 m; at that
radius every kind recurs on the dev root, the cables too (65-72%), because 0.25 m is the scale of the density of
instances (a random placement of the same number of cables gives 45-64%), not of a spot. The bath mat (94%), the
pedestal mat (82%) and the clothes pile (69%) recur at 0.10 m; the cables (33%, 33%, 43%) and the floor towel (0%) do
not.

`catch_words` says these kinds in a question's words (the vacuum's two lists of floor things name them).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .descriptions import kind_word
from .nearest_part import nearest_on_hull

Xy = tuple[float, float]
Hull = Sequence[Sequence[float]]
EPS = 1e-9

# Each constant and its source (FD-questions-0.10 sections 2.2 and 2.3; review B2, S4):
# at the drone's lowering height of about 7 m the nadir view covers +-5.9 m x +-4.4 m (fx = fy = 381, 640 x 480): a 4 m
# disc fits inside the short half-axis
ZONE_M = 4.0
REACH_M = 1.0     # about 3 s of the vacuum's 0.28-0.35 m/s: its next strip and turn
# "ahead or to your sides" is 90 degrees either way from the heading; the camera's floor point is 0.13 m ahead of the
# base centre that the 0.25 m brush disc is centred on, so a thing at the brush's side lies a few centimetres behind the
# camera's line: 10 degrees more keep it
ARC_DEG = 100.0
FRONT_M = 0.3     # the catastrophe zone: 1 s of travel at the vacuum's 0.3 m/s from the body's front edge

CATCH_KINDS = ("bath_mat", "cable_bundle", "clothes_pile", "extension_cord", "pedestal_mat", "power_strip", "towel")
SPOT_KINDS = ("bath_mat", "clothes_pile", "pedestal_mat")   # the share at a spot (dev root): 94%, 69%, 82%
TANGLE_KINDS = tuple(k for k in CATCH_KINDS if k not in SPOT_KINDS)   # 33% (extension cord), 33% (power strip), 43%
#                                                                       (cable bundle), 0% (floor towel, 11 of them)


def _each(kinds: Sequence[str]) -> str:
    """The kinds as words with their articles, "a towel", "an extension cord", joined "a, b or c"."""
    words = [(("an " if (word := kind_word(kind))[0] in "aeiou" else "a ") + word) for kind in kinds]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


def catch_words() -> str:
    """The kinds a brush catches in the words of a question (final review S4, 0.11.0): what the list asks for
    (`TANGLE_KINDS`) and what it counts neither way (`SPOT_KINDS`), "(a cable bundle, an extension cord, a power strip
    or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way)". The vacuum's two lists of floor
    things, `tangle_ahead` and `task_things`, both say it, so a reader is never charged for a thing the words leave
    open: a kind in neither list is a false place."""
    return f"({_each(TANGLE_KINDS)}; {_each(SPOT_KINDS)} counts neither way)"


def nearest(hull: Hull, cam: Xy) -> tuple[Xy, float]:
    """The point of the hull nearest the camera's floor point, and its distance."""
    point = nearest_on_hull(hull, cam)
    return point, math.dist(point, cam)


def in_zone(hull: Hull, centre: Xy) -> bool:
    """The hull has a point within `ZONE_M` of `centre`."""
    return nearest(hull, centre)[1] <= ZONE_M + EPS


def bearing_deg(point: Xy, cam: Xy, yaw_rad: float) -> float:
    """The angle in degrees, 0 to 180, between the heading `yaw_rad` and the way from the camera's floor point to
    `point`."""
    away = math.atan2(point[1] - cam[1], point[0] - cam[0]) - yaw_rad
    return abs(math.degrees(math.remainder(away, math.tau)))


def reaches(point: Xy, distance: float, cam: Xy, yaw_rad: float) -> bool:
    """A point at `distance` from the camera's floor point is in reach: within `REACH_M` and at most `ARC_DEG` from the
    heading (a point under the camera is in reach whatever the heading)."""
    return distance <= REACH_M + EPS and (distance <= EPS or bearing_deg(point, cam, yaw_rad) <= ARC_DEG + EPS)


def in_reach(hull: Hull, cam: Xy, yaw_rad: float) -> bool:
    """The hull's nearest point to the camera's floor point is in reach (`reaches`)."""
    point, distance = nearest(hull, cam)
    return reaches(point, distance, cam, yaw_rad)
