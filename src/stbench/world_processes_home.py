"""The home's processes between tasks: what the apartment does while no one watches, at what rates, and in what order
(docs/world.md). `world_processes.gaps` loads this module for a map of the home catalog and calls `start` before the
first task and `gap` before each later one.

Every process re-draws a thing's state by the law it started with, so the world at any task is distributed as the world
at the first and nothing reveals how many tasks came before a case's window: a piece of light furniture gets a new
place by `generate.light_start` (drawn from its map place, never from where it stands), a carried thing a new spot by
the start's split (a thing of a floor kind is left on the floor with `clutter.DROP_SHARE`, else it goes to a
receptacle), a floor clutter piece a new place by its kind's places, a door is open with `P_OPEN`. Each thing is drawn
again with the chance `half_life_p(gap, H)` of its half-life `H`, so a short gap moves little and a long one most
things. No thing is added or removed, and surface clutter stays put.

All the changes of one gap happen at the same whole second, `t_g`, in the order doors, light furniture, carried
things, floor clutter, and each is checked against the world as the changes before it left it, with the thing that
moves taken out of it (`World.without`): it has left at `t_g`, so another thing may take its place. Then no change
lands where another thing has not yet left, and the check is exact. The doors come first so that the floor guard
(`clutter_reach.FloorGuard`) of every later change sees each door leaf as it will stand at the end of the call, which
is how the next task is planned; every door of `doors_on_ways(m, keep_open)` is open then, so no door's state ever
changes that task's path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from . import clutter, generate
from .catalog import FLOOR_ITEM_Z
from .changes import World, when_in_gap
from .clutter_reach import FloorGuard
from .mapspec import MapSpec
from .placement import Spot, find_spot, occupied_entry
from .world_processes import doors_on_ways, half_life_p, set_doors

H_ITEM_S = 5 * 3600.0  # carried things: a 3-hour gap moves about a third (V0's calibration, PRIOR-ART §7)
H_CHAIR_S = 6 * 3600.0  # chairs, pulled out and pushed in at meals and work
H_LIGHT_S = 12 * 3600.0  # the stool and the laundry basket
H_FLOOR_CLUTTER_S = 12 * 3600.0  # floor clutter (a fan, a bin, a lamp); surface clutter stays put
H_DOOR_S = 2 * 3600.0  # an interior door's state is drawn again
P_OPEN = 0.7  # ... open with this probability, as at the start
ANCHORS = frozenset({"dock"})  # the furniture kinds a robot starts from: light pieces keep off their apron
RECEPTACLE_TRIES = 6  # spots a thing gets on a receptacle that lies on the floor (a mat, a rug) before it stays

SOURCES: dict[str, str] = {
    "H_ITEM_S": (
        "modelling choice, calibrated to V0's: a 3 h gap moves 34% of the carried things (V0 moved about 10 of its "
        "30-40 things a gap, 2 a room, over gaps of 0.5-6 h) and a 10 min gap 2%, short gaps moving little as "
        "3RScan's sparse rescans do (about 3.3 annotated changes a rescan; PRIOR-ART §7: home move rates beyond "
        "3RScan's have no primary source)"),
    "H_CHAIR_S": (
        "modelling choice: slower than a carried thing and quicker than the rest of the light furniture, a chair "
        "being pulled out and pushed in at meals and work; 3RScan counts chairs among the most-changed objects "
        "(PRIOR-ART §7); a 3 h gap draws 29% of the chairs again"),
    "H_LIGHT_S": (
        "modelling choice: the stool and the laundry basket stay where they are most days, so a 3 h gap draws 16% "
        "of them again; no source gives a home rate for light furniture (PRIOR-ART §7)"),
    "H_FLOOR_CLUTTER_S": (
        "modelling choice: a fan, a bin or a lamp on the floor moves when someone cleans or rearranges, about as "
        "rarely as the stool (a 3 h gap draws 16% of them again); no primary source (PRIOR-ART §7)"),
    "H_DOOR_S": (
        "modelling choice: doors and people dominate change in STRANDS (104 days, an office and a care home), and "
        "FreMEn models a door's state by a few daily and weekly cycles (PRIOR-ART §7), but no source gives a home "
        "door's switching rate; a 2 h half-life draws 65% of the doors again over a 3 h gap"),
    "P_OPEN": (
        "modelling choice: a home's interior doors are mostly open; 0.7 is what the world starts with, and the "
        "law each draw repeats, so a door is as likely open at any task as at the first"),
}


# -- who moves (each selection is a function of its own) ---------------------------------------------------------------
def _drawn(rng: np.random.Generator, ids: Iterable[str], p: Callable[[str], float]) -> list[str]:
    """The ids, in order, each kept with its own chance, one draw each."""
    return [i for i in ids if rng.random() < p(i)]


def light_movers(rng: np.random.Generator, w: World, gap_s: float) -> list[str]:
    """The pieces of light furniture a gap of `gap_s` seconds draws again: each chair with `half_life_p` of
    `H_CHAIR_S`, every other piece of `H_LIGHT_S`."""
    kinds = {e.id: e.kind for e in w.entities if e.mobility == "light"}

    def p(eid: str) -> float:
        return half_life_p(gap_s, H_CHAIR_S if kinds[eid] == generate.CHAIR_KIND else H_LIGHT_S)

    return _drawn(rng, w.light_rects, p)


def item_movers(rng: np.random.Generator, w: World, gap_s: float) -> list[str]:
    """The carried things a gap draws again, each with `half_life_p` of `H_ITEM_S`."""
    return _drawn(rng, w.carried, lambda _: half_life_p(gap_s, H_ITEM_S))


def floor_clutter(w: World) -> list[str]:
    """The ids of the world's floor clutter: the clutter entities of a kind the catalog stands on the floor."""
    on_floor = {k.name for k in w.catalog.clutter if k.on_floor}
    return [e.id for e in w.entities if e.mobility == "clutter" and e.kind in on_floor and e.id in w.clutter]


def clutter_movers(rng: np.random.Generator, w: World, gap_s: float) -> list[str]:
    """The floor clutter pieces a gap draws again, each with `half_life_p` of `H_FLOOR_CLUTTER_S`."""
    return _drawn(rng, floor_clutter(w), lambda _: half_life_p(gap_s, H_FLOOR_CLUTTER_S))


def door_redraws(rng: np.random.Generator, m: MapSpec, gap_s: float, forced: frozenset[str]) -> list[str]:
    """The interior doors (never the entrance) a gap draws again, in id order: every door forced open for the previous
    task, whatever the gap, and each other with `half_life_p` of `H_DOOR_S`."""
    ids = sorted(o.id for o in m.toggling_doors())
    return _drawn(rng, ids, lambda i: 1.0 if i in forced else half_life_p(gap_s, H_DOOR_S))


# -- how they move -----------------------------------------------------------------------------------------------------
def _receptacle(w: World, rest: World, rng: np.random.Generator, eid: str, t: float) -> Spot | None:
    """A spot on a receptacle other than the thing's own, at least `placement.MIN_MOVE` away (`placement.find_spot`). A
    thing that lies on the floor there (on a mat or a rug) must also split no planned robot's free floor
    (`FloorGuard`): it is drawn again, up to `RECEPTACLE_TRIES` times."""
    c, guard = w.carried[eid], None
    for _ in range(RECEPTACLE_TRIES):
        spot = find_spot(rng, c.spec, w.slots, rest.occupied(), avoid=(c.spot.pos[0], c.spot.pos[1]),
                         avoid_slot=c.spot.slot, catalog=w.catalog)
        if spot is None:
            return None
        rect, z0, z1 = occupied_entry(c.spec, spot.pos, spot.yaw)
        if z0 >= FLOOR_ITEM_Z:
            return spot
        guard = guard or FloorGuard(rest, t)
        if guard.allows(rect, z1 - z0):
            return spot
    return None


def move_item(w: World, m: MapSpec, rng: np.random.Generator, eid: str, t: float) -> bool:
    """A carried thing's new spot from `t` on, by the start's split: a thing of a floor kind is left on the floor
    (`clutter.floor_spot`, within reach of one of its kind's anchor pieces) with `clutter.DROP_SHARE`, else it goes to a
    receptacle (`_receptacle`). False, and the thing stays, when no spot holds it. (The start's second try for a floor
    drop, its room's open floor, is not repeated.)"""
    c = w.carried[eid]
    rest = w.without(eid)
    anchors = dict(w.catalog.floor_kinds).get(c.spec.kind)
    if anchors is not None and rng.random() < clutter.DROP_SHARE:
        at = clutter.floor_spot(rest, m, rng, c.spec.size, t=t, anchors=anchors)
        spot = None if at is None else Spot(clutter.floor_slot(m, at[0], at[1]), (at[0], at[1], 0.0), at[2])
    else:
        spot = _receptacle(w, rest, rng, eid, t)
    if spot is None:
        return False
    w.ledger.place(eid, t, spot.pos, spot.yaw, spot.slot.furniture_entity or None)
    c.spot = spot
    return True


def redraw_doors(w: World, m: MapSpec, rng: np.random.Generator, t: float, drawn: list[str],
                 keep_open: frozenset[str]) -> int:
    """Each door of `drawn` open with `P_OPEN`, then every door of `doors_on_ways(m, keep_open)` open, from `t` on;
    only a door whose state changes is written. Returns how many changed."""
    wanted = {door: bool(rng.random() < P_OPEN) for door in drawn}
    wanted.update(dict.fromkeys(sorted(doors_on_ways(m, sorted(keep_open))), True))
    return set_doors(w, t, wanted)


# -- the callbacks ---------------------------------------------------------------------------------------------------
def start(w: World, m: MapSpec, rng: np.random.Generator, keep_open: frozenset[str]) -> None:
    """t = 0: each interior door is open with `P_OPEN`, those of `doors_on_ways(m, keep_open)` open (the doors are
    states written over `build_world`'s all-open ones). The pieces start where `build_world`, `clutter.place` and
    `generate.light_starts` put them."""
    redraw_doors(w, m, rng, 0.0, sorted(o.id for o in m.toggling_doors()), keep_open)


def _run(ids: list[str], move: Callable[[str], bool]) -> tuple[int, int]:
    """(drawn, moved): `move` called on each id in turn, so each sees the world as the ones before it left it."""
    return len(ids), sum(1 for i in ids if move(i))


def gap(w: World, m: MapSpec, rng: np.random.Generator, lo: float, hi: float, keep_open: frozenset[str],
        forced: frozenset[str] = frozenset()) -> dict[str, tuple[int, int]]:
    """The changes of the gap (lo, hi), all at one whole second `t_g` at least 30 s from both ends, in this order: doors
    (`forced`: the doors forced open for the previous task, drawn again with `P_OPEN`, and the others by half-life; then
    every door of `doors_on_ways(m, keep_open)` is opened), light furniture, carried things, floor clutter. Returns
    (drawn, moved) by process: "doors" (the doors drawn and the doors whose state changed), "light", "items" and
    "floor"; a thing drawn that finds no spot stays."""
    t, span = when_in_gap(rng, lo, hi), hi - lo
    drawn = door_redraws(rng, m, span, forced)
    done = {"doors": (len(drawn), redraw_doors(w, m, rng, t, drawn, keep_open))}
    done["light"] = _run(light_movers(rng, w, span), lambda e: generate.light_start(w, m, rng, e, ANCHORS, t))
    done["items"] = _run(item_movers(rng, w, span), lambda e: move_item(w, m, rng, e, t))
    done["floor"] = _run(clutter_movers(rng, w, span), lambda e: clutter.move_floor(w, m, rng, e, t))
    return done

