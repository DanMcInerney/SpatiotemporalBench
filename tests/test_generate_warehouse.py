"""Generator contracts for the warehouse (generator 0.6.0, plan-warehouse §7.2): `build_world` on `warehouse_a` makes
valid worlds of the warehouse catalog, its items squared up on its racks, its deliveries and its departures (the
processes of `changes` over an explicit gap). The apartment's contracts are in `test_generate.py`."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

import test_generate
from amr_kit import amr_checks_off  # noqa: F401  (a fixture)
from conftest import script_on_map
from stbench import timeline as T
from stbench.catalog import CATALOGS, KINDS
from stbench.geometry import Rect, corners, distance, to_local
from stbench.mapspec import surface_world_rect
from stbench.validate import validate
from world_kit import GAP, after_gap, changes, initial_carried, initial_world, manifest_model_of

WAREHOUSE = CATALOGS["warehouse"]
DOCK_FLOOR = [(8.8, 6.0, 0.0, 0.0), (9.8, 6.0, 0.0, 0.0), (9.8, 7.0, 0.0, math.pi / 2)]  # open floor of the dock


@pytest.mark.usefixtures("amr_checks_off")
def test_warehouse_worlds_build_and_validate_with_an_amr_script(wh):
    """Contract (plan-warehouse §7.2, §9.1): `build_world` on `warehouse_a` makes a world that passes `validate`, before
    and after a gap's changes, for seeds 1 and 2 (seeds 1-30 are proven by the build's generate stage), with a tote
    robot's hand-made script on the dock's open floor (not a whole task of `amr/1`, whose own checks are off here), the
    map's name and hash, and the four regions. Regression caught:
    a warehouse that the collision rules reject (a forklift walk through a rack), a script that does not name its map,
    or the aisles' kind lost on the way into the timeline (a job reading them by kind would find none)."""
    for seed in (1, 2):
        tl = script_on_map("warehouse_a", seed, "amr", DOCK_FLOOR)
        assert validate(tl, wh) == [], seed
        assert validate(after_gap("warehouse_a", seed), wh) == [], seed
        assert (tl.episode.map, tl.episode.robot) == ("warehouse_a", "amr")
        assert [r.name for r in tl.regions] == ["dock", "west aisle", "middle aisle", "east aisle"]
        assert [r.kind for r in tl.regions] == ["dock", "aisle", "aisle", "aisle"]


def test_a_warehouse_world_holds_the_warehouse_catalog_and_an_apartment_world_the_homes():
    """Contract (plan-warehouse §2: the catalog is picked per map): a warehouse world starts with 22-28 carried items
    of the warehouse's kinds, each in a colour of its kind with its kind's box and manifest model, and holds the map's
    light equipment (two pallet jacks and a roll cage, with their models) and no mover; an apartment world holds none
    of the warehouse's own kinds (the box is the one kind both share). Regression caught: the household's 30-40 items
    or kinds in a warehouse (a mug on a rack), the warehouse's pallets in an apartment, a catalog read from the seed
    instead of the map, or an entity without its asset."""
    own = set(WAREHOUSE.kinds) - set(KINDS)
    assert own == {"pallet", "tote", "drum", "sack", "bin"}
    for seed in (1, 2):
        tl = initial_world("warehouse_a", seed)
        items = initial_carried(tl)
        assert 22 <= len(items) <= 28, seed
        for e in items:
            kind = WAREHOUSE.kinds[e.kind]
            assert set(e.attributes) == {"color"} and e.attributes["color"] in kind.colors, (seed, e)
            assert e.size == kind.size and e.asset == manifest_model_of(e.kind, e.attributes.get("color")), (seed, e)
        light = [e for e in tl.entities if e.mobility == "light"]
        assert sorted(e.kind for e in light) == ["pallet_jack", "pallet_jack", "roll_cage"], seed
        assert all(e.asset == manifest_model_of(e.kind, e.attributes.get("color")) for e in light), seed
        assert not [e for e in tl.entities if e.mobility == "mover"], seed
    for seed in range(1, 9):
        assert not {e.kind for e in initial_carried(initial_world("apartment_a", seed))} & own, seed


def test_warehouse_items_lie_wholly_on_a_surface_of_the_map_squared_up_at_every_time(wh):
    """Contract (VISION rule 2, V0 §3: truth is where things are): every placement of a carried item of a warehouse
    world, at the start and after a gap (moves and arrivals), has its whole footprint inside one surface of its
    support's map furniture, at that surface's height, turned a quarter turn to the surface within 0.04 rad (a pallet
    squared up in its bay). Checked against the map file, not the sampler. Hand case: a rack bay's surface is 1.08 x
    2.6 m, so a 1.2 x 1.0 m pallet turned 30 degrees (1.45 m across) would stick out of it. Regression caught: a
    footprint test dropped for a turned item (a pallet poking out of its bay, still 'on' the rack for `validate`, which
    tests the rack's whole outline), a surface chosen at the wrong level, or a yaw drawn at any angle."""
    seen = 0
    for seed in (1, 2, 3, 4, 5):
        tl = after_gap("warehouse_a", seed)
        for e in (e for e in tl.entities if e.mobility == "carried"):
            for p in tl.placements_of(e.id):
                support = tl.placement_of(p.on, p.t0)
                pieces = [f for f in wh.furniture if f.mobility == "fixed"
                          and math.dist(f.center, support.pos[:2]) < 1e-6 and abs(f.yaw - support.yaw) < 1e-4]
                assert len(pieces) == 1, (seed, e.id, p.on)
                rects = [surface_world_rect(pieces[0].center, pieces[0].yaw, s) for s in pieces[0].surfaces
                         if abs(s.z - p.pos[2]) < 1e-6]
                assert rects, (seed, e.id, p.pos)
                body = corners(Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw))
                inside = [all(abs(x) <= r.sx / 2 + 1e-3 and abs(y) <= r.sy / 2 + 1e-3
                              for x, y in (to_local(r, cx, cy) for cx, cy in body)) for r in rects]
                assert any(inside), (seed, e.id, p.pos)
                turn = (p.yaw - rects[0].yaw) % (math.pi / 2)
                assert min(turn, math.pi / 2 - turn) <= 0.04 + 1e-4, (seed, e.id, p.yaw)
                seen += 1
    assert seen > 70


def test_warehouse_gaps_deliver_pallets_and_take_pallets_boxes_and_totes_away():
    """Contract (plan-warehouse §3.4; `Catalog.delivery_kind`, `leaving_kinds`): in a warehouse gap what arrives is a
    pallet in a wrap colour that no item of the world has had (a description that matches no earlier item), what leaves
    is a pallet, a box or a tote (a kind of the catalog's `leaving_kinds`, which every world has some of), and the
    carried items move at least 0.6 m, all inside the gap, 30 s from its ends. Seeds 1-24 hold at least three
    deliveries and three departures between them. Regression caught: the household's delivery box or its leaving bag
    in a warehouse (a KeyError, or a `bag` that does not exist), an arrival that duplicates a description (a question
    about it is ambiguous), a sack or drum that leaves, or a change drawn outside the gap."""
    lo, hi = GAP
    added, removed = [], []
    for seed in range(1, 25):
        tl = after_gap("warehouse_a", seed)
        for kind, eid, t, prev, nxt in changes(tl):
            assert lo + 30 <= t <= hi - 30, (seed, kind, eid, t)
            e = tl.entity(eid)
            if kind == "add":
                added.append((seed, e.kind, e.attributes["color"]))
                assert e.kind == WAREHOUSE.delivery_kind and e.attributes["color"] in WAREHOUSE.delivery_colors
                twins = [o for o in tl.entities if o.id != eid and o.kind == e.kind and o.attributes == e.attributes]
                assert twins == [], (seed, e)
            elif kind == "remove":
                removed.append((seed, e.kind))
                assert e.kind in WAREHOUSE.leaving_kinds, (seed, e)
            elif kind == "move":
                assert distance(prev.pos[:2], nxt.pos[:2]) >= 0.6 - 1e-6, (seed, eid)
    assert len(added) >= 3 and len(removed) >= 3, (added, removed)
    assert {kind for _, kind, *_ in added} == {"pallet"}


def test_a_warehouse_world_is_the_same_bytes_in_another_process():
    """Contract (as `test_hash_is_stable_across_processes`, for the new catalog): nothing in the warehouse's sampler or
    changes depends on Python's salted hash or on the iteration order of a set (the catalogs are dicts): a subprocess
    with another PYTHONHASHSEED makes the world of seed 2 after a gap that this process made. Regression caught: a set
    or hash-ordered draw in the warehouse path (the build's workers would disagree with the tests, and worlds would
    not reproduce)."""
    out = subprocess.run([sys.executable, test_generate.__file__, "after_gap", "warehouse_a", "2"],
                         capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "777"}).stdout.strip()
    assert out == T.digest(after_gap("warehouse_a", 2))
