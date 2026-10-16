"""Placement by catalog: the `catalog=` parameter of `sample_plan`, `sample_items`,
`place_set` and `find_spot` defaults to the household catalog, so the apartment's draws do not move, and a warehouse
catalog fills a warehouse map."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from conftest import apartment, warehouse
from stbench.catalog import CATALOGS, for_map
from stbench.geometry import corners, to_local
from stbench.placement import (
    ItemSpec,
    build_slots,
    find_spot,
    item_rect,
    occupied_entry,
    place_set,
    sample_items,
    sample_plan,
)
from stbench.seeding import make_rng

HOME, WAREHOUSE = CATALOGS["home"], CATALOGS["warehouse"]


def slots_of(m):
    return build_slots(m, {f.id: f"E{f.id}" for f in m.furniture if f.mobility == "fixed"})


def draw_world(m, rng, **catalog):
    """The items of a world as `generate._build_world` places them: each identical set together, then the others one by
    one. (kind, colour, surface, x, y, yaw) per item, with None where an item found no room."""
    slots, occupied, out = slots_of(m), [], []
    plan = sample_plan(rng, **catalog)

    def record(spec, spot):
        out.append((spec.kind, spec.attributes["color"], None if spot is None else spot.slot.surface_id,
                    None if spot is None else (round(spot.pos[0], 4), round(spot.pos[1], 4), round(spot.yaw, 4))))
        if spot is not None:
            occupied.append(occupied_entry(spec, spot.pos, spot.yaw))

    for specs in plan.sets:
        for spec, spot in zip(specs, place_set(rng, specs, slots, occupied, **catalog), strict=True):
            record(spec, spot)
    for spec in plan.others:
        record(spec, find_spot(rng, spec, slots, occupied, **catalog))
    return out


def test_the_default_catalog_is_the_household_one_and_fills_the_apartment():
    """Contract (plan-warehouse §2, §7.1): `sample_plan`, `place_set`, `find_spot` and `sample_items` called without
    `catalog=` draw what they draw with the household catalog named, on seeds 1 to 3 (plans) and seed 7 (the spots on
    the apartment's surfaces), and the apartment's draws repeat: the plan holds 30 to 40 items, and every item that
    found room rests on a surface of the apartment. No draw is recorded. Regression caught: a default that is not the
    household catalog, a draw made in another order or from another list (the kinds, the set kinds, the item count), a
    receptacle prior read from the wrong catalog, or an item placed on a surface the map lacks."""
    def sets_and_singles(seed, **catalog):
        plan = sample_plan(make_rng("catalog-default", seed), **catalog)
        return ([(s[0].kind, s[0].attributes["color"], len(s)) for s in plan.sets],
                [(o.kind, o.attributes["color"]) for o in plan.others][:4], len(plan.items))

    for seed in (1, 2, 3):
        assert sets_and_singles(seed, catalog=HOME) == sets_and_singles(seed)
        assert 30 <= sets_and_singles(seed)[2] <= 40, seed
    placed = draw_world(apartment(), make_rng("catalog-default", 7))
    assert draw_world(apartment(), make_rng("catalog-default", 7), catalog=HOME) == placed
    assert draw_world(apartment(), make_rng("catalog-default", 7)) == placed
    surfaces = {slot.surface_id for slot in slots_of(apartment())}
    assert placed and {p[2] for p in placed if p[2] is not None} <= surfaces
    assert sample_items(make_rng("catalog-default", 1), catalog=HOME) == sample_items(make_rng("catalog-default", 1))


def test_a_warehouse_plan_is_22_to_28_items_of_the_warehouse_kinds_in_sets_and_lookalikes():
    """Contract (plan-warehouse §3.1, §3.2): with the warehouse catalog a world draws 22-28 items from its six kinds,
    2-3 identical sets of 2-4 from tote, drum, box and bin (a different kind each), and every other description once,
    so a description is ambiguous only inside a set or a lookalike group; the colours are the kind's own. Regression
    caught: the household's 30-40 items (a plan the 31 pairs cannot fill: "catalog too small for a fresh item"), a set
    of a pallet (no one keeps pallets in matching sets), or a colour no kind of the warehouse has."""
    kinds = WAREHOUSE.kinds
    for seed in range(1, 41):
        plan = sample_plan(make_rng("wh-plan", seed), catalog=WAREHOUSE)
        assert 22 <= len(plan.items) <= 28, seed
        assert 2 <= len(plan.sets) <= 3 and all(2 <= len(s) <= 4 and len({m.key for m in s}) == 1 for s in plan.sets)
        set_kinds = [s[0].kind for s in plan.sets]
        assert set(set_kinds) <= set(WAREHOUSE.set_kinds) and len(set(set_kinds)) == len(set_kinds), (seed, set_kinds)
        others = Counter(o.key for o in plan.others)
        assert max(others.values()) == 1 and not set(others) & {s[0].key for s in plan.sets}, seed
        assert all(i.kind in kinds and i.attributes["color"] in kinds[i.kind].colors for i in plan.items), seed
    # the sampler is deterministic per rng, and a different catalog is a different world
    assert sample_items(make_rng("wh-plan", 1), catalog=WAREHOUSE) == sample_items(make_rng("wh-plan", 1),
                                                                                   catalog=WAREHOUSE)
    assert {i.kind for i in sample_items(make_rng("wh-plan", 1))} <= set(HOME.kinds)


def test_a_warehouse_world_finds_a_spot_for_every_item_on_the_warehouse_surfaces():
    """Contract (plan-warehouse §0, §3.1): the 48 surfaces of `warehouse_a` hold a whole world of 22-28 items of the
    warehouse catalog (pallets 1.2 x 1.0 m among them), every item resting wholly on its surface, so no sampled item is
    left out of a world (which would shrink it unseen). Hand case: a pallet needs 1.0 m of a rack bay's 1.08 m depth
    (sideways to the beam) and 2.5 m of its 2.6 m length for two. Regression caught: a rack surface too small for a
    pallet, or a catalog whose items fit nowhere (a world that silently loses its pallets)."""
    wh = warehouse()
    assert for_map(wh) is WAREHOUSE
    for seed in range(1, 11):
        placed = draw_world(wh, make_rng("wh-fit", seed), catalog=WAREHOUSE)
        left_out = [p for p in placed if p[2] is None]
        assert 22 <= len(placed) <= 28 and not left_out, (seed, left_out)
    pallet = ItemSpec("pallet", {"color": "blue"})
    for seed in range(20):
        spot = find_spot(make_rng("wh-fit", 99, seed), pallet, slots_of(wh), [], catalog=WAREHOUSE)
        local = [to_local(spot.slot.rect, *c) for c in corners(item_rect(pallet, spot.pos[0], spot.pos[1], spot.yaw))]
        assert spot.slot.kind in ("rack", "staging lane"), seed  # a pallet fits no shelf and no table
        assert all(abs(x) <= spot.slot.rect.sx / 2 and abs(y) <= spot.slot.rect.sy / 2 for x, y in local), seed


def test_find_spot_reads_the_receptacle_priors_of_the_catalog_it_is_given():
    """Contract (priors must not be exploitable; plan-warehouse §3.1): an item goes on a receptacle kind its kind lists
    in the given catalog 70% of the time and elsewhere 30%, not always. A tote (rack, shelf, work table) on an empty
    warehouse: 60-80% of 200 draws land on those, the rest on a staging lane; hand value 0.7. Without `catalog=` the
    household's priors apply, in which a pallet is no kind at all. Regression caught: priors read from the household
    catalog (a KeyError on the first warehouse item), or typical placement made certain."""
    wh = warehouse()
    slots = slots_of(wh)
    tote = ItemSpec("tote", {"color": "blue"})
    typical = sum(find_spot(make_rng("wh-prior", i), tote, slots, [], catalog=WAREHOUSE).slot.kind in
                  WAREHOUSE.kinds["tote"].receptacles for i in range(200))
    assert 120 <= typical <= 160
    with pytest.raises(KeyError, match="tote"):  # the default catalog is the household one, which has no tote
        find_spot(make_rng("wh-prior", 0), tote, slots, [])


def quarter_turn_error(spot) -> float:
    """How far the item's yaw is from the nearest multiple of 90 degrees to its surface (radians)."""
    r = (spot.yaw - spot.slot.rect.yaw) % (math.pi / 2)
    return min(r, math.pi / 2 - r)


def test_a_square_kind_sits_a_quarter_turn_to_its_surface_give_or_take_0_04_rad():
    """Contract (plan-warehouse §2, "an item's yaw is uniform"; `Kind.square`): a warehouse kind lands at a multiple of
    90 degrees to its surface within 0.04 rad, whichever of the four turns the surface leaves room for; the household's
    kinds are not squared, the box included although the warehouse shares it. Hand cases: a tote (0.6 x 0.4 m) on a
    staging lane (surface 2.7 x 1.3 m) fits all four turns; a pallet (1.2 x 1.0) fits a rack bay (surface 1.08 across
    x 2.6 along the beam) only with its 1.2 m along the beam, which is the surface's turns 1 and 3 (turned the other way
    it needs 1.2 m of the 1.08 m across); with a uniform yaw
    P(error > 0.2 rad) = 1 - 0.2 / (pi / 4) = 0.75, so over 200 draws the household box is squared in at most 25%.
    Regression caught: the flag ignored (a pallet or tote at any angle, which does not fit a bay), the tolerance
    loosened, only two of the four turns drawn (every pallet facing one way), or the household box squared (the
    apartment is redrawn, and the 90 kept episodes stop reproducing)."""
    wh_slots = slots_of(warehouse())
    lane = next(s for s in wh_slots if s.kind == "staging lane")
    tote = ItemSpec("tote", {"color": "blue"})
    spots = [find_spot(make_rng("sq-lane", i), tote, [lane], [], catalog=WAREHOUSE) for i in range(200)]
    assert all(quarter_turn_error(sp) <= 0.04 + 1e-9 for sp in spots)
    assert {round(((sp.yaw - lane.rect.yaw) % (2 * math.pi)) / (math.pi / 2)) % 4 for sp in spots} == {0, 1, 2, 3}
    rack = next(s for s in wh_slots if s.kind == "rack")
    pallet = ItemSpec("pallet", {"color": "red"})
    on_rack = [find_spot(make_rng("sq-rack", i), pallet, [rack], [], catalog=WAREHOUSE) for i in range(100)]
    assert all(quarter_turn_error(sp) <= 0.04 + 1e-9 for sp in on_rack)
    assert {round(((sp.yaw - rack.rect.yaw) % (2 * math.pi)) / (math.pi / 2)) % 4 for sp in on_rack} == {1, 3}
    box = ItemSpec("box", {"color": "brown"})
    apt_slots = slots_of(apartment())
    home = [find_spot(make_rng("sq-home", i), box, apt_slots, [], catalog=HOME) for i in range(200)]
    squared = sum(quarter_turn_error(sp) <= 0.04 for sp in home)
    assert squared <= 50 and max(quarter_turn_error(sp) for sp in home) > 0.7
    wh_box = [find_spot(make_rng("sq-wh", i), box, wh_slots, [], catalog=WAREHOUSE) for i in range(200)]
    assert all(quarter_turn_error(sp) <= 0.04 + 1e-9 for sp in wh_box)


def test_the_members_of_a_square_kinds_set_sit_squared_together_and_a_household_set_keeps_its_slack():
    """Contract (plan-warehouse §2; `placement.SQUARE_YAW`, `SET_YAW`): the members of an identical set share one yaw, a
    quarter turn to the surface within 0.04 rad when the kind is square (three warehouse totes) and within 0.12 rad
    when it is not (three mugs): over 60 draws the mugs stray beyond 0.04 (the largest of 60 draws of uniform(-0.12,
    0.12) is about 0.118) and the totes never do. Regression caught: a warehouse set at the household's looser slack
    (a row of totes visibly crooked on a rack), or the household set squared (the apartment is redrawn)."""
    slots = slots_of(warehouse())
    totes = (ItemSpec("tote", {"color": "green"}),) * 3
    for i in range(60):
        spots = place_set(make_rng("sq-set", i), totes, slots, [], catalog=WAREHOUSE)
        assert None not in spots and len({round(sp.yaw, 9) for sp in spots}) == 1, i
        assert quarter_turn_error(spots[0]) <= 0.04 + 1e-9, i
    mugs = (ItemSpec("mug", {"color": "red"}),) * 3
    apt_slots = slots_of(apartment())
    errors = [quarter_turn_error(place_set(make_rng("sq-mug", i), mugs, apt_slots, [])[0]) for i in range(60)]
    assert max(errors) > 0.06 and max(errors) <= 0.12 + 1e-9
