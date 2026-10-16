"""The warehouse's processes between tasks: what the hall does while no one watches, at what rates (docs/job-amr.md,
docs/world.md). `world_processes.gaps` loads this module for a map of the `warehouse` catalog and calls `start` before
the first task and `gap` before each later one.

Every process re-draws a thing's place by the law it started with, so the world at any task is distributed as the world
at the first and nothing reveals how many tasks came before a case's window: a carried thing (a pallet, a tote, a bin, a
box, a drum, a sack) gets a new spot by the start's split (`world_processes_home.move_item`: the warehouse's kinds have
no floor anchors, so it goes to a receptacle, a rack, a staging lane, a shelf or a work table), a piece of light
equipment (a pallet jack, the roll cage) a new place by `generate.light_start`, drawn from its map place and never from
where it stands. Each thing is drawn again with the chance `half_life_p(gap, H)` of its half-life `H`, so a short gap
moves little and a long one most things. No thing is added or removed. The hall has no doors, so `start` does nothing.

All the changes of one gap happen at the same whole second, `t_g`, at least `changes.EDGE_BUFFER` from both its ends, in
the order light equipment, then carried things, and each is checked against the world as the changes before it left it,
with the thing that moves taken out of it (`World.without`): it has left at `t_g`, so another thing may take its place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from . import generate
from .changes import World, when_in_gap
from .mapspec import MapSpec
from .world_processes import half_life_p
from .world_processes_home import move_item

H_PALLET_S = 2 * 3600.0  # pallets: inbound and outbound flow through the staging lanes and the racks
H_TOTE_S = 3 * 3600.0  # totes, bins and boxes: picked, replenished and returned
H_DRUM_S = 8 * 3600.0  # drums and sacks: bulk stock, moved a few times a shift
H_LIGHT_S = 1 * 3600.0  # pallet jacks and the roll cage: used and parked all shift
HALF_LIFE_S = {"pallet": H_PALLET_S, "tote": H_TOTE_S, "bin": H_TOTE_S, "box": H_TOTE_S, "drum": H_DRUM_S,
               "sack": H_DRUM_S}
ANCHORS = frozenset({"charger"})  # the furniture kinds a robot starts from: light pieces keep off their apron

SOURCES: dict[str, str] = {
    "H_PALLET_S": (
        "modelling choice: pallets are the warehouse's fastest stock (inbound and outbound flow through the staging "
        "lanes, PRIOR-ART §7: 'slot contents hourly; carts, pallets and beds'), so a 2 h half-life draws 29% of them "
        "again over a 1 h gap and 65% over a 3 h gap; no primary source gives a rate for a hall this small"),
    "H_TOTE_S": (
        "modelling choice: totes, bins and boxes are picked and returned through a shift, slower than whole pallets; a "
        "3 h half-life draws 21% of them again over a 1 h gap; no primary source (PRIOR-ART §7)"),
    "H_DRUM_S": (
        "modelling choice: drums and sacks are bulk stock that moves a few times a shift (a 3 h gap draws 23% of them "
        "again); no primary source (PRIOR-ART §7)"),
    "H_LIGHT_S": (
        "modelling choice: a pallet jack or a roll cage is used and parked again within the hour (a 1 h gap draws half "
        "of them again, as the home's chairs are pulled out at meals); no primary source (PRIOR-ART §7)"),
}


def _drawn(rng: np.random.Generator, ids: Iterable[str], p: Callable[[str], float]) -> list[str]:
    """The ids, in order, each kept with its own chance, one draw each."""
    return [i for i in ids if rng.random() < p(i)]


def light_movers(rng: np.random.Generator, w: World, gap_s: float) -> list[str]:
    """The pieces of light equipment a gap of `gap_s` seconds draws again, each with `half_life_p` of `H_LIGHT_S`."""
    return _drawn(rng, w.light_rects, lambda _: half_life_p(gap_s, H_LIGHT_S))


def item_movers(rng: np.random.Generator, w: World, gap_s: float) -> list[str]:
    """The carried things a gap draws again, each with `half_life_p` of its kind's half-life (`HALF_LIFE_S`)."""
    return _drawn(rng, w.carried, lambda eid: half_life_p(gap_s, HALF_LIFE_S[w.carried[eid].spec.kind]))


def start(w: World, m: MapSpec, rng: np.random.Generator, keep_open: frozenset[str]) -> None:
    """t = 0: nothing. The hall has no doors, and the pieces start where `build_world`, `clutter.place` and
    `generate.light_starts` put them."""


def _run(ids: list[str], move: Callable[[str], bool]) -> tuple[int, int]:
    """(drawn, moved): `move` called on each id in turn, so each sees the world as the ones before it left it."""
    return len(ids), sum(1 for i in ids if move(i))


def gap(w: World, m: MapSpec, rng: np.random.Generator, lo: float, hi: float, keep_open: frozenset[str],
        forced: frozenset[str] = frozenset()) -> dict[str, tuple[int, int]]:
    """The changes of the gap (lo, hi), all at one whole second `t_g` at least 30 s from both ends, in this order: light
    equipment, carried things. Returns (drawn, moved) by process, "light" and "items"; a thing drawn that finds no spot
    stays."""
    t, span = when_in_gap(rng, lo, hi), hi - lo
    done = {"light": _run(light_movers(rng, w, span), lambda e: generate.light_start(w, m, rng, e, ANCHORS, t))}
    done["items"] = _run(item_movers(rng, w, span), lambda e: move_item(w, m, rng, e, t))
    return done
