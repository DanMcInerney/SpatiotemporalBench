"""The downtown's processes between flights: what the streets do while the drone is home, at what rates
(docs/job-drone.md). `world_processes.gaps` loads this module for a map of the downtown catalog and calls `start` before
the first flight and `gap` before each later one.

Every thing is drawn again by the law it started with (`world_processes_home.move_item`: a new place on a receptacle,
the world's own split of typical and other surfaces), with the chance `half_life_p(gap, H)` of its half-life `H`, so a
short gap moves little and a long one most things. The changes of one gap all happen at the same whole second `t_g` at
least 30 s from both ends of the gap, in the world's order, each checked against the world as the ones before it left
it, with the thing that moves taken out of it, as the home's are. No thing is added or removed; there are no doors, so
`keep_open` and `forced` are not read.
"""

from __future__ import annotations

import numpy as np

from .changes import World, when_in_gap
from .mapspec import MapSpec
from .world_processes import half_life_p
from .world_processes_home import move_item

H_VEHICLE_S = 1 * 3600.0  # cars, vans, pickups and box trucks: on-street parking turns over about hourly
H_CART_S = 6 * 3600.0  # a street vendor's cart stays for a shift
H_FIXTURE_S = 48 * 3600.0  # skips, dumpsters and portable toilets stay for days
HALF_LIFE_S = {"car": H_VEHICLE_S, "van": H_VEHICLE_S, "pickup": H_VEHICLE_S, "box_truck": H_VEHICLE_S,
               "food_cart": H_CART_S, "skip": H_FIXTURE_S, "dumpster": H_FIXTURE_S, "portable_toilet": H_FIXTURE_S}

SOURCES: dict[str, str] = {
    "H_VEHICLE_S": (
        "modelling choice, bounded by PRIOR-ART §7's parking stays: a downtown study found an average stay of 1 h "
        "52 min (Everett 2015; an exponential stay of that mean has a half-life of 1.3 h), another 74% of stays under "
        "2 h (Carpinteria), and retail bays turn over in under 30 min (Melbourne sensors); 1 h draws half the parked "
        "vehicles again over an hour and 88% over three"),
    "H_CART_S": (
        "modelling choice, no primary source: a street vendor trades for a shift, so a cart is drawn again with the "
        "chance of a 6 h half-life (21% over a 2 h gap, 29% over 3 h); PRIOR-ART §7 gives the unmetered kinds no stay"),
    "H_FIXTURE_S": (
        "modelling choice, no primary source: a builders' skip is hired for days, a front-load container is emptied on "
        "a weekly round and a portable toilet is serviced weekly, so a 48 h half-life (1% over an hour, 4% over "
        "three); PRIOR-ART §7 gives the unmetered kinds no stay"),
}


def start(w: World, m: MapSpec, rng: np.random.Generator, keep_open: frozenset[str]) -> None:
    """t = 0: the downtown has no doors and nothing to set; the pieces start where `build_world` put them."""


def movers(rng: np.random.Generator, w: World, gap_s: float) -> list[str]:
    """The things a gap of `gap_s` seconds draws again, in the world's order, one draw each: every thing with the chance
    `half_life_p` of its kind's half-life."""
    return [eid for eid, c in w.carried.items() if rng.random() < half_life_p(gap_s, HALF_LIFE_S[c.spec.kind])]


def gap(w: World, m: MapSpec, rng: np.random.Generator, lo: float, hi: float, keep_open: frozenset[str],
        forced: frozenset[str] = frozenset()) -> dict[str, tuple[int, int]]:
    """The changes of the gap (lo, hi), all at one whole second `t_g` at least 30 s from both ends: each thing drawn
    again (`movers`) moves by the start's law (`move_item`), in the world's order. Returns {"items": (drawn, moved)}; a
    thing drawn that finds no spot stays."""
    t = when_in_gap(rng, lo, hi)
    drawn = movers(rng, w, hi - lo)
    return {"items": (len(drawn), sum(1 for eid in drawn if move_item(w, m, rng, eid, t)))}
