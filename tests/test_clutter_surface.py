"""`clutter_surface`: the moves that stay on surfaces, which `helper/1`'s place-spot coin is made of. The worlds are the
apartment's seeds 101 and 102 as the generator builds one before it plans a path (`process_kit.fresh`: built,
cluttered, the light furniture started). Hand values are the map's: the hall console F20 at (5.4, 6.0) turned -90
degrees, its surface S13 0.26 m deep and 0.86 m long at 0.8 m, with picture frames and mail on its back strip, and the
0.2 x 0.4 m window the helper faces from (5.3875, 5.3625) heading +90 degrees."""

from __future__ import annotations

import math

import pytest

from process_kit import fresh
from stbench import clutter
from stbench.catalog import FLOOR_ITEM_Z
from stbench.clutter_surface import free_spot, move_surface, move_to_surface, slide_surface, standing
from stbench.geometry import Rect, rects_overlap
from stbench.mapspec import load_map
from stbench.placement import Slot
from stbench.seeding import make_rng

pytestmark = pytest.mark.xdist_group(name=__name__)  # the worlds are cached per worker process
M = load_map("apartment_a")
BAND = (0.01, 0.5)  # the helper's window: taller than 0.01 m, base lower than 0.5 m above the surface
CONSOLE = "S13"  # the hall console's surface
WINDOW = Rect(5.3875, 6.0, 0.2, 0.4, -math.pi / 2)  # by hand: 0.4 m of the console's long side about the hit point


def console(w) -> Slot:
    return next(s for s in w.slots if s.surface_id == CONSOLE)


def clutter_on(w, slot: Slot) -> list[str]:
    """The clutter pieces standing on the slot's surface (their base at its height, their footprint on it)."""
    return [e for e, (box, z0, _) in w.clutter.items() if abs(z0 - slot.z) < 0.01 and rects_overlap(box, slot.rect)]


def test_standing_lists_what_reaches_into_the_column():
    """Contract: `standing` lists the carried items and clutter pieces whose footprint is within `margin` of the
    rectangle and which reach into the band above the ground: top higher than ground + band[0], base lower than ground +
    band[1]. Hand case on seed 101's hall console (ground 0.8 m): its surface holds four clutter pieces (picture frames
    and mail, all taller than 0.01 m) and all four are listed; the same rectangle with the band 3 m up lists none
    (nothing reaches that high), and with the band's top at the surface itself lists none (nothing starts below it); a
    rectangle 2 m east of the console lists none, and lists all four (and more) once the margin is 3 m. Regression
    caught: a band read from the floor (the window's surface is 0.8 m up), a footprint test that ignores the margin, or
    clutter left out of the column."""
    w = fresh(101)
    slot = console(w)
    on = clutter_on(w, slot)
    assert len(on) == 4
    assert set(standing(w, slot.rect, slot.z, BAND)) == set(on)
    assert standing(w, slot.rect, slot.z, (3.0, 4.0)) == []
    assert standing(w, slot.rect, slot.z, (-1.0, 0.0)) == []
    away = Rect(slot.rect.cx + 2.0, slot.rect.cy, 0.2, 0.2, 0.0)
    assert standing(w, away, slot.z, BAND) == []
    assert set(on) <= set(standing(w, away, slot.z, BAND, 3.0))


def test_a_clutter_piece_moves_by_its_kinds_places_and_a_refused_spot_is_not_taken():
    """Contract: `move_surface` places a surface clutter piece again from `t` on by its kind's places, in the world as
    it stands without it: its ledger placement and its `World.clutter` entry change, it stands on a raised support, and
    a spot `refuse` turns down is never the one written (here: any spot within 0.05 m of the hall console's window);
    when no spot is left it returns False and writes nothing. A kind that stands on the floor is refused with
    ValueError. Hand case: of the clutter pieces of seed 101's hall console that stand in the window, at least one
    moves, at t = 5000 s, to a spot outside it, with one new placement in the ledger, on a support, at the footprint the
    table keeps; each one that does not move leaves the ledger and the table as they were. Regression caught: a moved
    piece left in the ledger and the footprint table at two places, one put back where the coin wants it empty, a
    failed move that writes half a change, or a floor clutter piece moved by the surface rule."""
    refuse = lambda box, z0, z1: rects_overlap(box, WINDOW, 0.05)  # noqa: E731
    inside = [e for e in clutter_on(fresh(101), console(fresh(101))) if refuse(fresh(101).clutter[e][0], 0, 0)]
    assert len(inside) >= 2, "seed 101's hall console holds pieces in the window"
    moved = 0
    for eid in inside:
        w = fresh(101)
        before, n_before = w.clutter[eid], len(w.ledger.finish()[0])
        if not move_surface(w, M, make_rng("test_move_surface", 1), eid, 5000.0, refuse):
            assert w.clutter[eid] == before and len(w.ledger.finish()[0]) == n_before
            continue
        moved += 1
        box, z0, z1 = w.clutter[eid]
        assert box != before[0] and not refuse(box, z0, z1)
        placed = [p for p in w.ledger.finish()[0] if p.entity == eid and p.t0 == 5000.0]
        assert len(placed) == 1 and placed[0].on is not None and placed[0].pos[2] >= FLOOR_ITEM_Z
        assert placed[0].pos[:2] == pytest.approx((box.cx, box.cy), abs=1e-5)
        assert z1 - z0 == pytest.approx(before[2] - before[1])
    assert moved >= 1
    floor = next(e for e, (_, z, _) in fresh(101).clutter.items() if z < FLOOR_ITEM_Z)
    with pytest.raises(ValueError, match="does not stand on a surface"):
        move_surface(fresh(101), M, make_rng("test_floor_piece", 1), floor, 5000.0)


def test_a_carried_item_moves_by_the_worlds_law_but_never_onto_the_floor(monkeypatch):
    """Contract: `move_to_surface` moves a carried item by `world_processes_home.move_item` and writes only a result on
    a raised surface. With every floor kind forced to the floor (`clutter.DROP_SHARE` 1: the start's split drops a shoe,
    a bag, a toy or a box on the floor whenever it can), a floor-kind item is not moved at all: the call is False, the
    ledger and the item's spot are as they were; a mug (no floor kind) moves to a raised surface other than its own, as
    `move_item` would. A `refuse` that turns every spot down leaves the world as it was. Regression caught: a floor
    drop written by a coin that must not change the floor (every planned path would cross it), or an item that stays
    where the coin wants it empty."""
    w, again = fresh(102), fresh(102)  # built before the patch: a world built under it would hold no raised floor kind
    monkeypatch.setattr(clutter, "DROP_SHARE", 1.0)
    kinds = dict(w.catalog.floor_kinds)
    shoe = next(e for e, c in w.carried.items() if c.spec.kind in kinds and c.spot.pos[2] >= FLOOR_ITEM_Z)
    mug = next(e for e, c in w.carried.items() if c.spec.kind not in kinds and c.spot.pos[2] >= FLOOR_ITEM_Z)
    ledger_before = len(w.ledger.finish()[0])
    spot_before = w.carried[shoe].spot
    assert not move_to_surface(w, M, make_rng("test_floor_kind", 1), shoe, 5000.0)
    assert w.carried[shoe].spot == spot_before and len(w.ledger.finish()[0]) == ledger_before
    own = w.carried[mug].spot.slot
    assert move_to_surface(w, M, make_rng("test_mug", 1), mug, 5000.0)
    moved = w.carried[mug].spot
    assert moved.pos[2] >= FLOOR_ITEM_Z and moved.slot.furniture_entity and moved.slot != own
    placed = [p for p in w.ledger.finish()[0] if p.entity == mug and p.t0 == 5000.0]
    assert len(placed) == 1 and placed[0].on == moved.slot.furniture_entity
    assert not move_to_surface(again, M, make_rng("test_refuse_all", 1), mug, 5000.0, lambda spot: True)
    assert again.carried[mug].spot.slot == own and len(again.ledger.finish()[0]) == ledger_before


def test_free_spot_is_the_nearest_clear_place_on_the_surface():
    """Contract: `free_spot` gives the spot on a surface nearest the aim point where a box lies wholly on it
    (`EDGE_MARGIN` 0.01 m from its edges), a quarter turn to it, clear of everything by `ITEM_MARGIN` (0.015 m), and
    not within `margin` of the keep-out rectangle; the aim point itself comes first. By hand on a 1.0 x 0.5 m surface
    at 0.8 m with one 0.2 x 0.2 m box at its centre and a 0.1 x 0.1 m thing aimed at the centre: the aim is taken, and
    the nearest clear places are on the grid 0.02 m apart (-0.44 m along and -0.19 m across from the centre, the edge
    limits): 0.1 + 0.015 + 0.05 = 0.165 m across from the box, so 0.17 m on either side of it (the grid across holds the
    odd hundredths). With no box but a keep-out 0.4 m long and 0.1 m wide on the centre line and 0.04 m of margin: 0.05
    + 0.04 + 0.05 = 0.14 m across, so 0.15 m. A box the surface cannot hold gives None. Regression caught: a spot
    overlapping a neighbour, off the surface's edge, inside the keep-out, or not the nearest."""
    slot = Slot("E1", "S1", "table", Rect(0.0, 0.0, 1.0, 0.5, 0.0), 0.8, None)
    box = (Rect(0.0, 0.0, 0.2, 0.2, 0.0), 0.8, 1.0)
    size = (0.1, 0.1, 0.1)
    x, y, yaw = free_spot([box], size, slot, (0.0, 0.0))
    assert (round(x, 6), round(abs(y), 6), yaw) == (0.0, 0.17, 0.0)
    assert not rects_overlap(Rect(x, y, 0.1, 0.1, yaw), box[0], 0.015 - 1e-9)
    keep_out = Rect(0.0, 0.0, 0.4, 0.1, 0.0)
    x, y, yaw = free_spot([], size, slot, (0.0, 0.0), keep_out, 0.04)
    assert (round(x, 6), round(abs(y), 6)) == (0.0, 0.15)
    assert not rects_overlap(Rect(x, y, 0.1, 0.1, yaw), keep_out, 0.04)
    assert free_spot([], (0.6, 0.6, 0.1), slot, (0.0, 0.0)) is None


def test_a_clutter_piece_slides_off_a_place_to_the_nearest_spot_outside_it():
    """Contract: `slide_surface` slides a clutter piece standing on a slot to the free spot nearest the aim that is not
    within `margin` of the keep-out, writing its new placement and footprint from `t` on, turned a quarter turn to the
    surface, at the surface's height; a piece on another surface is refused (False, nothing written). Hand case on seed
    102's hall console: the smallest piece standing in the window slides out of it (+0.05 m of margin), keeps 0.015 m
    from every other thing and stays on the console. Regression caught: a clutter piece that cannot be moved off a
    crowded surface (the coin's clear window would stay blocked), or one slid onto another thing."""
    w = fresh(102)
    slot = console(w)
    inside = [e for e in clutter_on(w, slot) if rects_overlap(w.clutter[e][0], WINDOW)]
    assert inside, "seed 102's hall console holds a piece in the window"
    eid = min(inside, key=lambda e: w.clutter[e][0].sx * w.clutter[e][0].sy)
    old = w.clutter[eid][0]
    assert slide_surface(w, eid, 5000.0, slot, (WINDOW.cx, WINDOW.cy), WINDOW, 0.05)
    box, z0, z1 = w.clutter[eid]
    assert z0 == pytest.approx(slot.z) and not rects_overlap(box, WINDOW, 0.05)
    assert (box.cx, box.cy) != (old.cx, old.cy) and rects_overlap(box, slot.rect)
    others = [r for r, lo, hi in w.without(eid).occupied() if lo < z1 and z0 < hi]
    assert not any(rects_overlap(box, r, 0.015 - 1e-9) for r in others)
    far = next(e for e, (_, z, _) in w.clutter.items() if abs(z - slot.z) > 0.2)
    assert not slide_surface(w, far, 5000.0, slot, (WINDOW.cx, WINDOW.cy), WINDOW, 0.05)
    assert [p for p in w.ledger.finish()[0] if p.entity == far and p.t0 == 5000.0] == []
