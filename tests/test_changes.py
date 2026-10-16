"""World changes: the delivery a gap adds, per catalog, the shift of light furniture, and what `World.occupied` lists
(the move, shift and remove rules are covered over an explicit gap in `test_generate.py`)."""

from __future__ import annotations

import numpy as np

from conftest import apartment, warehouse
from stbench.catalog import DELIVERY_COLORS, DELIVERY_KIND, KINDS, WAREHOUSE_KINDS
from stbench.changes import World, _fresh_delivery_spec, shift_light_furniture
from stbench.generate import build_world
from stbench.geometry import Rect
from stbench.ledger import Ledger
from stbench.mapspec import load_map
from stbench.placement import ItemSpec, find_spot
from stbench.seeding import make_rng


def world(used: set, m=None) -> World:
    w = World(m or apartment(), Ledger(), [], [])
    w.used_keys |= used
    return w


def test_a_delivery_box_is_brown_first_then_the_other_colours_and_carries_a_colour_only():
    """Contract (V0 section 4: an added item never matches the description of an item already in the world; DECISIONS,
    "Probe run 1": the size attribute is dropped): the delivered box is a cardboard brown one if no brown box has been
    in the world, then each other colour in turn, none when every colour is taken, and its only attribute is the
    colour. Hand case: with the five colours taken one by one, the box that comes is the next one in the order.
    Regression caught: a delivery that duplicates a description already in use (an ambiguous item that no set
    announces), brown no longer first, a size attribute back on the box, or a crash once the colours run out."""
    assert DELIVERY_KIND == "box" and set(DELIVERY_COLORS) == set(KINDS[DELIVERY_KIND].colors)
    assert DELIVERY_COLORS[0] == "brown"
    used: set = set()
    for taken, colour in enumerate(DELIVERY_COLORS):
        spec = _fresh_delivery_spec(world(used))
        assert spec == ItemSpec("box", {"color": colour}), taken
        used.add(spec.key)
    assert _fresh_delivery_spec(world(used)) is None
    brown_and_blue = {ItemSpec("box", {"color": c}).key for c in ("brown", "blue")}
    assert _fresh_delivery_spec(world(brown_and_blue)) == ItemSpec("box", {"color": "white"})  # the free ones in order


def test_a_warehouse_delivery_is_a_pallet_white_first_then_the_pallets_colours_and_none_when_all_are_taken():
    """Contract (plan-warehouse §3.4: a pallet of an unused wrap colour arrives; V0 section 4: an added item never
    matches an existing description): the warehouse's delivery is a pallet in the first of its six wrap colours that
    no item of the world has had, white first and then blue, red, green, yellow and black, none once all six are
    taken, and its only attribute is the colour. Hand case: with white and green taken, the pallet that comes is the
    blue one. The household box stays what it is (the test above). Regression caught: the household's box delivered
    into a warehouse (a cardboard box arriving at a dock), a colour a pallet is not drawn in, or a crash once the
    colours run out."""
    m = warehouse()
    assert set(WAREHOUSE_KINDS["pallet"].colors) == {"white", "blue", "red", "green", "yellow", "black"}
    taken = {ItemSpec("pallet", {"color": c}).key for c in ("white", "green")}
    assert _fresh_delivery_spec(world(taken, m)) == ItemSpec("pallet", {"color": "blue"})
    used: set = set()
    order = []
    for _ in range(6):
        spec = _fresh_delivery_spec(world(used, m))
        order.append((spec.kind, spec.attributes["color"]))
        used.add(spec.key)
    assert order == [("pallet", c) for c in ("white", "blue", "red", "green", "yellow", "black")]
    assert _fresh_delivery_spec(world(used, m)) is None
    # a box in the world's history does not use up a pallet colour
    assert _fresh_delivery_spec(world({ItemSpec("box", {"color": "white"}).key}, m)) == ItemSpec(
        "pallet", {"color": "white"})


def test_a_world_without_light_furniture_draws_its_shift_count_and_shifts_nothing():
    """Contract (plan-city §2, §3.3: the downtown has no light furniture; generator 0.7.0): `shift_light_furniture`
    draws the 0-2 shift count from the gap's rng on every map, so the draws after it are the same whether a world has
    light furniture or not, and where there is none it shifts nothing and draws nothing more: it returns 0, the ledger
    holds no placement, and the rng has advanced by exactly one `integers(0, 3)`, whatever the count was. Hand case: for
    20 seeds the rng ends in the state of a twin that made that one draw. Regression caught: `rng.integers(0, 0)` on the
    empty list of pieces (a ValueError on every downtown T2 and T3 episode), or the count not drawn (the gap's moves,
    adds and removals would shift with it)."""
    for seed in range(20):
        w = World(load_map("downtown_a"), Ledger(), [], [])
        rng, twin = make_rng("shift", seed), make_rng("shift", seed)
        twin.integers(0, 3)
        assert shift_light_furniture(w, rng, 1000.0, 9000.0, np.empty((0, 2))) == 0
        assert rng.bit_generator.state == twin.bit_generator.state, seed
        assert w.ledger.finish() == ([], [])


def test_two_shifts_drawn_for_one_light_piece_move_it_at_most_once():
    """Contract (`changes.shift_light_furniture`: a piece moves at most once a gap, so a world with fewer pieces than
    the count it draws shifts each of them and stops): with the apartment's one chair as its only light furniture and a
    count of 2 drawn, the second shift has no piece left and is skipped, the chair moves 0 or 1 times, never raises.
    Hand case: of seeds 0-39, those whose first draw is 2 are found with a twin rng, and the chair moves for at least
    one of them. Regression caught: `integers(0, 0)` on the pieces that have not moved (a ValueError in a world with a
    single light piece), or a piece moved twice in one gap (two overlapping placements)."""
    apt = apartment()
    chair = next(f for f in apt.furniture if f.mobility == "light")
    two_drawn = [s for s in range(40) if make_rng("shift", s).integers(0, 3) == 2]
    assert len(two_drawn) >= 5
    moved = 0
    for seed in two_drawn:
        w = World(apt, Ledger(), [], [])
        w.light_rects["E0001"] = chair.footprint()
        w.ledger.place("E0001", 0.0, (chair.center[0], chair.center[1], 0.0), chair.yaw, None)
        done = shift_light_furniture(w, make_rng("shift", seed), 1000.0, 9000.0, np.empty((0, 2)))
        placements, _ = w.ledger.finish()
        assert done == len(placements) - 1 <= 1, seed
        moved += done
    assert moved >= 1


def test_occupied_includes_clutter():
    """Contract (map X-H8): `World.occupied()` lists carried items and clutter, so a later move cannot land inside
    clutter. Hand case: clutter 0.5 m tall over the whole of the largest surface leaves `find_spot` no spot on it, and
    with no clutter it finds one. Regression caught: a later move landing inside clutter."""
    w = build_world(apartment(), 1)
    n = len(w.carried)
    assert len(w.occupied()) == n
    slot = max(w.slots, key=lambda s: s.area)
    entry = (slot.rect, slot.z, slot.z + 0.5)
    spec = ItemSpec("mug", {"color": "red"})
    assert find_spot(make_rng("clutter", 1), spec, [slot], [], catalog=w.catalog) is not None
    w.clutter["E9001"] = entry
    assert w.occupied()[-1] == entry and len(w.occupied()) == n + 1
    assert find_spot(make_rng("clutter", 1), spec, [slot], w.occupied(), catalog=w.catalog) is None


def test_a_world_without_a_thing_leaves_the_rest_as_it_stands():
    """Contract (`World.without`, M2-A2W): a thing that moves is not in the world it finds at the instant it leaves, so
    it never blocks its own new place, and the world it left is untouched. Hand case on seed 1's world with one clutter
    piece added: without a carried item the world has one carried item and one occupied entry fewer, without the
    clutter piece one clutter entry fewer, without a light piece one light rect fewer, each time with the same ledger
    and everything else as it was; the original still holds all of them; an id that is not there changes nothing.
    Regression caught: a mover blocking its own new place (a chair that cannot turn where it stands), or a copy that
    edits the world it came from."""
    w = build_world(apartment(), 1)
    chair, item = next(iter(w.light_rects)), next(iter(w.carried))
    w.clutter["E9001"] = (Rect(1.0, 1.0, 0.2, 0.2, 0.0), 0.0, 0.1)
    n_carried, n_occupied, n_light = len(w.carried), len(w.occupied()), len(w.light_rects)
    no_item = w.without(item)
    assert (len(no_item.carried), len(no_item.occupied()), len(no_item.clutter)) == (n_carried - 1, n_occupied - 1, 1)
    assert no_item.light_rects == w.light_rects and no_item.ledger is w.ledger
    assert len(w.without("E9001").clutter) == 0 and len(w.without("E9001").carried) == n_carried
    assert len(w.without(chair).light_rects) == n_light - 1 and chair not in w.without(chair).light_rects
    assert (len(w.carried), len(w.light_rects), len(w.clutter)) == (n_carried, n_light, 1)
    assert w.without("nothing").carried == w.carried and w.without("nothing").clutter == w.clutter
