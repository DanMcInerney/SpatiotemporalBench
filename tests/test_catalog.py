"""Catalog: one attribute (the colour), and every kind has what the generator and renderer need."""

from __future__ import annotations

import dataclasses
import math

import pytest

from stbench import asset_manifest as A
from stbench import descriptions
from stbench.catalog import (
    CATALOGS,
    KINDS,
    LEAVING_KINDS,
    MOVER_CAMERA_M,
    MOVER_KINDS,
    N_MOVERS,
    SET_KINDS,
    ClutterKind,
    Kind,
    describe,
    for_map,
    item_size,
    mover_kind,
    place_of,
)
from stbench.mapspec import load_map
from test_mapspec import palette_keys


def every_kind() -> list[Kind]:
    """The carried kinds of every catalog, once each (the box is in both)."""
    return list({k.name: k for c in CATALOGS.values() for k in c.kinds.values()}.values())


def every_mover_kind() -> list:
    return [k for c in CATALOGS.values() for k in c.mover_kinds.values()]


def test_no_kind_has_a_size_attribute():
    """Contract (DECISIONS, "Probe run 1": the predeclared consequence of size accuracy .389 over n 36 against the bar
    .80): a kind has no size variants, so an item's box is its kind's box whatever attributes it carries, and a
    description is colour then kind. Regression caught: the size attribute reintroduced, as a `Kind.sizes` field, a
    scale applied to the box, or a size word in a description."""
    assert "sizes" not in {f.name for f in dataclasses.fields(Kind)}
    for name, kind in KINDS.items():
        assert item_size(name) == kind.size, name
    assert describe("bowl", {"color": "red"}) == "red bowl"
    assert describe("mug", {}) == "mug"


def test_dropped_kinds_are_gone_from_catalog_descriptions_and_manifest():
    """Contract (VISION rule 10, DECISIONS "Probe run 1" and "v0 scope"): a kind the models did not recognise is
    dropped, whatever the cause, so keys (a tag colour nobody sees), phone (a face-up screen) and glasses (not reliably
    recognised even at 256-512 px, and seen in under half of their worlds) are in no catalog, no description wording
    and no manifest model, and nothing of them is generated or rendered. Regression caught: a dropped kind left in one
    layer, so a world could still contain it (the catalog), or a library build still carry its asset (the manifest)."""
    dropped = {"keys", "phone", "glasses"}
    assert dropped.isdisjoint(KINDS)
    assert descriptions.kind_word("keys") == "keys"  # no longer a "set of" unit
    assert descriptions.kind_word("glasses") == "glasses"  # nor a "pair of" unit
    models = [a for a in A.packaged().assets if a.role == "model"]
    assert dropped.isdisjoint({a.kind for a in models}) and dropped.isdisjoint({a.builder for a in models})


def test_catalog_covers_the_kinds_with_what_generation_needs():
    """The kinds are all present, each with at least one colour, receptacle priors, a renderer geometry hint and a
    plausible size (nothing over half a metre). Exactly the plant (its leaves are green) has one colour; every other
    kind has several."""
    expected = {"mug", "glass", "bowl", "plate", "book", "remote", "bag", "shoes", "laptop", "plant", "bottle", "box",
                "towel", "toy"}
    assert set(KINDS) == expected
    for k in KINDS.values():
        assert k.colors and k.receptacles and k.geometry
        assert all(0 < v <= 0.5 for v in k.size)
    assert {k.name for k in KINDS.values() if len(k.colors) == 1} == {"plant"}
    assert KINDS["plant"].colors == ("green",)


def test_the_laptop_is_open():
    """Contract (VISION rule 10: recognisable in one frame; DECISIONS "Probe run 1": a closed laptop is a slab, and 6 of
    42 white ones were called a laptop, the rest a book): the laptop is drawn open, with a screen, so its box is as high
    as a lid stands, far above a book's, and its asset has a screen slot. Regression caught: the closed laptop back."""
    assert KINDS["laptop"].size[2] >= 0.2 > 3 * KINDS["book"].size[2]
    assert "screen" in A.packaged().model_for("laptop").materials


def test_no_kind_has_a_clear_colour():
    """Contract (M2 plan 4.4, V0 §5): the ID pass and the RGB image must agree, and clear glass would show what lies
    behind it in RGB while the ID pass shows the glass, so "clear" is no colour of any kind (glass and bottle are
    tinted). Regression caught: "clear" reintroduced to glass or bottle."""
    assert [k.name for k in every_kind() if "clear" in k.colors] == []
    assert KINDS["glass"].colors and KINDS["bottle"].colors  # the tinted ones remain


def test_a_kind_with_several_colours_has_an_asset_that_takes_a_tint():
    """Contract (VISION rule 10: the colour attribute is what the pixels show): a kind with more than one colour is
    rendered in each, so its asset's tint mode is not "none" (`luminance` or `swap`); an asset with tint "none" keeps
    its own look and is therefore the kind's one colour (the plant: white pot, green leaves, labelled green).
    Regression caught: a second colour added to the plant, or a tint mode set to none on a kind with colours, which
    renders every colour of it the same."""
    m = A.packaged()
    for kind in every_kind():
        for colour in kind.colors:
            mode = m.model_for(kind.name, colour).tint["mode"]
            assert (len(kind.colors) > 1) <= (mode != "none"), (kind.name, mode, kind.colors)
    assert m.model_for("plant").tint["mode"] == "none"


def test_set_kinds_are_the_five_things_people_keep_in_matching_sets():
    """Contract (M2 plan 4.1): identical sets are drawn from mug, glass, bowl, plate and towel only, and each of them
    has a second colour (a one-attribute lookalike needs a sibling). Regression caught: a kind that people do not own
    in sets (a phone, a book) in the list, or a set kind the sampler cannot build a lookalike for."""
    assert set(SET_KINDS) == {"mug", "glass", "bowl", "plate", "towel"} and len(SET_KINDS) == 5
    assert all(len(KINDS[k].colors) >= 2 for k in SET_KINDS)


def test_catalog_sizes_follow_the_boxes_of_their_assets():
    """Contract (M2 plan 4.3, V0 section 4): `Kind.size` is the truth that footprints, `extent_m` and the in-view cover
    use, and the renderer fits the asset's mesh to it, so it must describe the mesh. A `per_axis` model's catalog box
    scales each native axis (the manifest's `native_bbox_m`, which the library build checks against the mesh to 1 mm) by
    factors that differ by at most the manifest's `max_anisotropy`; a procedural model is built to its catalog box, to
    1 mm. Regression caught: a catalog box that disagrees with its mesh (the mug was squashed 36%, the plate 53%, the
    laptop scan 862%), so the footprints the generator and compiler use lie about what is rendered."""
    m = A.packaged()
    for kind in every_kind():
        for asset in {m.model_for(kind.name, c) for c in kind.colors}:  # every model of the kind, whichever its colour
            assert asset.native_bbox_m is not None, kind.name
            scales = [want / have for want, have in zip(kind.size, asset.native_bbox_m, strict=True)]
            if asset.fit["mode"] == "per_axis":
                assert max(scales) / min(scales) - 1 <= asset.fit["max_anisotropy"], (kind.name, scales)
            else:
                assert asset.fit["mode"] == "exact"
                assert all(abs(want - have) <= 0.001 for want, have in zip(kind.size, asset.native_bbox_m,
                                                                          strict=True)), (
                    kind.name, kind.size, asset.native_bbox_m)


def test_mover_colours_are_colours_the_renderer_can_draw():
    """Contract (VISION rule 10: the colour attribute is what the pixels show): every colour of a
    mover kind is a key of the renderer's palette (`render/shapes.PALETTE`, read from its source), so a mover is drawn
    in the colour its description names. Regression caught: a colour with no palette entry ("silver"), which passes
    generation and fails only in the Blender build of a T3 episode, after the render has started."""
    palette = palette_keys()
    for kind in every_mover_kind():
        assert kind.colors and set(kind.colors) <= palette, (kind.name, set(kind.colors) - palette)
        assert "clear" not in kind.colors, kind.name  # the ID pass and RGB must agree (as for carried kinds)


def test_mover_kinds_are_a_separate_vocabulary_that_the_sampler_can_fill():
    """Contract: a mover kind never shares a name with a carried kind (a description's kind
    word alone says which class it is, and a mover is never mistaken for an item of the same description), there are
    enough kinds for `N_MOVERS` different ones per episode, the speed and dwell ranges are positive and ordered, the
    dwell stays under the 20 s cap that keeps it inside `where_at`'s 30 s unknown window, `same_room` is a probability,
    and a mover's box and words come from the same functions as an item's. Regression caught: a mover kind named like a
    carried one (two classes under one description), `N_MOVERS` raised above the kinds there are (the sampler would
    draw fewer), a dwell of a minute (a mover that never moves inside the window), or `item_size` raising KeyError for
    a mover (the generator sizes every entity by it)."""
    assert set(MOVER_KINDS).isdisjoint(KINDS) and N_MOVERS == 2 <= len(MOVER_KINDS)
    for name, kind in MOVER_KINDS.items():
        assert kind.name == name and item_size(name) == kind.size and all(0 < v <= 0.5 for v in kind.size), name
        assert 0 < kind.speed[0] <= kind.speed[1] <= 1.0 and 0 <= kind.dwell[0] <= kind.dwell[1] <= 20, name
        assert 0.0 <= kind.same_room <= 1.0, name
    assert descriptions.phrase("robot_vacuum", {"color": "white"}) == "white robot vacuum"
    assert descriptions.phrase("ball_robot", {"color": "orange"}) == "orange ball robot"
    assert descriptions.phrase("home_robot", {"color": "grey"}) == "grey home robot"


# -- catalogs ------------------------------------------------------------------------------------------------------
HOME_KINDS = ["mug", "glass", "bowl", "plate", "book", "remote", "bag", "shoes", "laptop", "plant", "bottle", "box",
              "towel", "toy"]  # written out: a change to the order redraws every apartment world


def test_the_home_catalog_is_the_household_constants_unchanged_in_value_and_order():
    """Contract (plan-warehouse §2, §7.1): `CATALOGS["home"]` is the household catalog the apartment's worlds are drawn
    from, so its kinds, their order, its set and leaving kinds, its mover kinds, the number of movers and of items are
    the values generator 0.5.0 draws with, and a home mover keeps the camera 0.6 m away as before. Regression caught:
    any reordering or edit of these (the draws are indexed by position, so an apartment would be redrawn and the kept
    90 episodes stop reproducing), or a mover kind that quietly gets another camera clearance."""
    home = CATALOGS["home"]
    assert list(home.kinds) == HOME_KINDS and home.kinds is KINDS
    assert home.set_kinds == ("mug", "glass", "bowl", "plate", "towel") == SET_KINDS
    assert home.leaving_kinds == ("bag", "box") == LEAVING_KINDS
    assert list(home.mover_kinds) == ["robot_vacuum", "home_robot", "ball_robot"] and home.mover_kinds is MOVER_KINDS
    assert (home.n_movers, home.n_items) == (2, (30, 40)) and N_MOVERS == 2
    assert MOVER_CAMERA_M == 0.6 and {k.camera_m for k in home.mover_kinds.values()} == {0.6}


def test_for_map_picks_the_catalog_the_map_names():
    """Contract (plan-warehouse §1, §7.3; plan-city §7.3): `for_map(m)` is the catalog `m.catalog` names, "home" for the
    apartment (which names none), "warehouse" for `warehouse_a` and "downtown" for `downtown_a`; one that does not exist
    is an error naming it and the known ones. Regression caught: the warehouse sampled with household kinds (a pallet
    that is a mug), the city with the warehouse's, or the apartment losing its default."""
    assert for_map(load_map("apartment_a")) is CATALOGS["home"]
    assert for_map(load_map("warehouse_a")) is CATALOGS["warehouse"]
    assert for_map(load_map("downtown_a")) is CATALOGS["downtown"]
    assert set(CATALOGS) == {"home", "warehouse", "downtown", "residential"}
    with pytest.raises(ValueError, match="'storehouse'.*home, warehouse, downtown, residential"):
        for_map(dataclasses.replace(load_map("apartment_a"), catalog="storehouse"))


def test_a_kind_two_catalogs_share_has_one_size_and_item_size_finds_every_kind():
    """Contract (plan-warehouse §2): the box is the one kind the household and the warehouse share, and it is one asset
    at one size, and the warehouse's colours are the household's but brown; every other name belongs to one catalog, and
    a mover's name is never a carried kind's (a description's kind word alone says which class it is). `item_size` finds
    a kind of any catalog.
    Regression caught: the warehouse box resized (two sizes for one model: the renderer fits the mesh to one of them), a
    warehouse kind that reuses a household name with another size, or `item_size` failing on a forklift (the generator
    sizes every entity by it)."""
    carried = [name for c in CATALOGS.values() for name in c.kinds]
    assert sorted({n for n in carried if carried.count(n) > 1}) == ["box", "car", "dumpster", "pickup", "skip", "van"]
    home, warehouse = CATALOGS["home"], CATALOGS["warehouse"]
    assert home.kinds["box"].size == warehouse.kinds["box"].size == (0.40, 0.30, 0.267)
    assert home.kinds["box"].colors == ("brown", "white", "blue", "red", "green")
    assert warehouse.kinds["box"].colors == ("white", "blue", "red", "green")
    movers = [name for c in CATALOGS.values() for name in c.mover_kinds]
    assert len(movers) == len(set(movers)) == 7 and set(movers).isdisjoint(carried)
    assert item_size("box") == (0.40, 0.30, 0.267) and item_size("forklift") == (2.6, 1.1, 2.1)
    assert item_size("pallet") == (1.2, 1.0, 1.0) and item_size("robot_vacuum") == (0.34, 0.34, 0.09)
    with pytest.raises(KeyError):
        item_size("spoon")


def test_the_warehouse_catalog_is_six_kinds_of_31_colour_pairs_with_stock_sizes():
    """Contract (plan-warehouse §3.1, §3.2): pallet, tote, drum, box, sack and bin, with the colours and boxes of real
    stock (a 1200 x 1000 mm pallet and a 1.0 m load, a 600 x 400 mm Euro tote, a 585 mm drum), 6+6+5+4+5+5 = 31 (kind,
    colour) pairs (the box is the household's without brown: the racks' stock is brown cartons, so "the brown box" would
    not be unique in the pictures), 22-28 items a world, identical sets of tote, drum, box and bin, and pallets, boxes
    and totes leaving.
    Every kind has at least two colours (a lookalike needs a sibling) and receptacle priors in the order of the plan.
    Regression caught: a kind or colour dropped (the look-alike pressure and the never-seen pool shrink), a set kind
    with one colour (a sampler that cannot build the set's lookalike), a leaving kind not in the catalog, or the brown
    box back in the warehouse (a description that matches hundreds of cartons of stock in the pictures)."""
    w = CATALOGS["warehouse"]
    assert list(w.kinds) == ["pallet", "tote", "drum", "box", "sack", "bin"]
    assert {k.name: k.size for k in w.kinds.values()} == {
        "pallet": (1.2, 1.0, 1.0), "tote": (0.6, 0.4, 0.32), "drum": (0.58, 0.58, 0.88), "box": (0.40, 0.30, 0.267),
        "sack": (0.9, 0.5, 0.25), "bin": (0.40, 0.25, 0.20)}
    assert [len(k.colors) for k in w.kinds.values()] == [6, 6, 5, 4, 5, 5]
    assert w.kinds["box"].colors == ("white", "blue", "red", "green")
    assert "brown" in CATALOGS["home"].kinds["box"].colors
    assert w.kinds["pallet"].receptacles == ("rack", "staging lane") and w.kinds["bin"].receptacles[0] == "shelf"
    assert all(k.colors and k.receptacles and k.geometry and len(set(k.colors)) == len(k.colors)
               for k in w.kinds.values())
    assert w.set_kinds == ("tote", "drum", "box", "bin") and w.leaving_kinds == ("pallet", "box", "tote")
    assert set(w.set_kinds) | set(w.leaving_kinds) <= set(w.kinds)
    assert all(len(w.kinds[k].colors) >= 2 for k in w.set_kinds)
    assert (w.n_movers, w.n_items) == (2, (22, 28)) and w.n_movers <= len(w.mover_kinds)


def test_every_catalog_leaves_a_pool_of_descriptions_that_no_world_holds():
    """Contract (docs/questions.md, the never-seen pool): the compiler draws unknown golds from descriptions that other
    seeds' worlds hold and this world does not, so a world must never use up the catalog. A world of at most `hi` items
    has at most `hi - 2` distinct descriptions (it holds at least two identical sets of at least two). Hand values: the
    warehouse 31 pairs - (28 - 2) = 5 left at the very least, the household 67 - 38 = 29, the city 43 - (34 - 2) = 11.
    (The plan asked for 1.3 times as many pairs as items; 31 / 28 is 1.11, but the worlds hold 17.3 distinct
    descriptions on average: 13.7 are left.)
    Regression caught: items raised, or colours cut, until a world can hold every description."""
    pairs = {name: sum(len(k.colors) for k in c.kinds.values()) for name, c in CATALOGS.items()}
    assert pairs == {"home": 67, "warehouse": 31, "downtown": 43, "residential": 27}
    for name, c in CATALOGS.items():
        assert pairs[name] - (c.n_items[1] - 2) >= 5, name


def test_a_mover_keeps_the_camera_outside_its_own_body():
    """Contract (plan-warehouse §2, §3.5; `MoverKind.camera_m`): a mover's centre stays `camera_m` from the camera, and
    that is clear of its own body by at least 0.3 m: the plan's circumradius plus 0.35 m, to the centimetre. Hand values
    (circumradius = half the diagonal of the box): forklift 1.4116 (1.76 - 1.4116 = 0.348), tote robot 0.5 (0.35),
    robot vacuum 0.2404 (0.36), home robot 0.2305 (0.37), ball robot 0.1556 (0.44). Regression caught: the one global
    0.6 m used for a 2.6 m forklift, which puts the camera inside it (the render is black and the visibility log says it
    is seen), or a kind added with a clearance smaller than its body."""
    for kind in every_mover_kind():
        circumradius = math.hypot(kind.size[0], kind.size[1]) / 2
        assert kind.camera_m >= circumradius + 0.3, (kind.name, kind.camera_m, circumradius)
    w = CATALOGS["warehouse"].mover_kinds
    assert (w["forklift"].camera_m, w["tote_robot"].camera_m) == (1.76, 0.85)


def test_warehouse_movers_are_a_forklift_and_a_tote_robot_that_move_within_the_dwell_cap():
    """Contract (as the household movers): a forklift (2.6 x 1.1 x 2.1 m;
    0.8-1.5 m/s) and a tote robot (0.8 x 0.6 x 1.3 m; 0.6-1.0 m/s), each pausing 3-20 s (under the 30 s window of
    `where_at`'s unknown rule) and choosing a waypoint in its room half the time; two movers of different kinds so every
    description is unique, and their words read as a person says them. Regression caught: a dwell of a minute (a mover
    that never moves inside the window), speeds that are not ordered, or a kind name that reads wrong in a question
    ("tote_robot")."""
    movers = CATALOGS["warehouse"].mover_kinds
    assert {n: (k.size, k.speed, k.dwell, k.same_room) for n, k in movers.items()} == {
        "forklift": ((2.6, 1.1, 2.1), (0.8, 1.5), (3, 20), 0.5),
        "tote_robot": ((0.8, 0.6, 1.3), (0.6, 1.0), (3, 20), 0.5)}
    assert {n: k.colors for n, k in movers.items()} == {"forklift": ("yellow", "orange", "red"),
                                                       "tote_robot": ("blue", "grey", "orange")}
    assert descriptions.phrase("forklift", {"color": "yellow"}) == "yellow forklift"
    assert descriptions.phrase("tote_robot", {"color": "blue"}, plural=True) == "blue tote robots"
    assert descriptions.phrase("pallet_jack", {"color": "red"}) == "red pallet jack"


def test_no_household_kind_sits_square_to_its_surface_and_every_warehouse_kind_does():
    """Contract (plan-warehouse §2, "an item's yaw is uniform"): `Kind.square` says a kind is turned a quarter turn at a
    time to its surface. Every warehouse kind is square (stock on racks is squared up: a pallet at 30 degrees does not
    fit a bay), no household kind is, and the box is square in the warehouse and not in the household, although it is
    one kind at one size in both (the downtown's kinds, all square too, are held with its catalog below). Regression
    caught: a household kind flagged square (the apartment's draws change, so the 90 kept episodes stop reproducing),
    the shared box flagged square in the home catalog for the same reason, or a warehouse kind left unflagged (a pallet
    turned at any angle, which rarely fits a bay)."""
    home, warehouse = CATALOGS["home"], CATALOGS["warehouse"]
    assert not any(k.square for k in home.kinds.values())
    assert all(k.square for k in warehouse.kinds.values())
    assert (home.kinds["box"].square, warehouse.kinds["box"].square) == (False, True)
    assert home.kinds["box"].size == warehouse.kinds["box"].size


def test_each_catalog_names_what_is_delivered_and_what_leaves():
    """Contract (plan-warehouse §2, §3.4): a catalog says what arrives between tasks (a kind of its own and the colours
    that kind comes in, none repeated) and what leaves. The household's delivery is a box, brown first, as in generator
    0.5.0; the warehouse's is a pallet, in a colour a pallet has. Regression caught: a delivery in a colour the kind
    does not come in (the render has no such tint), or a delivery kind the catalog does not have (a KeyError in the
    middle of a gap)."""
    home, warehouse = CATALOGS["home"], CATALOGS["warehouse"]
    assert (home.delivery_kind, home.delivery_colors) == ("box", ("brown", "white", "blue", "red", "green"))
    assert warehouse.delivery_kind == "pallet" and warehouse.delivery_colors[0] == "white"
    for c in (home, warehouse):
        assert c.delivery_kind in c.kinds and len(set(c.delivery_colors)) == len(c.delivery_colors), c.name
        assert set(c.delivery_colors) <= set(c.kinds[c.delivery_kind].colors) and c.delivery_colors, c.name


def test_mover_kind_finds_a_kind_of_any_catalog():
    """Contract (`validate` reads a mover's `camera_m` by its kind alone, whichever map the timeline is of): the mover
    kinds of the catalogs are disjoint, so a name finds one kind, and an unknown name is a KeyError. Regression caught:
    a lookup that only knows the household's movers (a forklift would raise in `validate` or take the household's
    0.6 m camera clearance)."""
    forklift = CATALOGS["warehouse"].mover_kinds["forklift"]
    assert mover_kind("forklift") is forklift and forklift.camera_m == 1.76
    assert mover_kind("robot_vacuum") is MOVER_KINDS["robot_vacuum"] and mover_kind("robot_vacuum").camera_m == 0.6
    with pytest.raises(KeyError):
        mover_kind("dragon")


# -- downtown ------------------------------------------------------------------------------------------------------
def test_the_downtown_catalog_is_eight_kinds_of_43_pairs_and_a_taxi_and_a_bus():
    """Contract (plan-city §3.1, §3.5, §7.3): the city's eight kinds in the plan's order with its real boxes (a 4.6 m
    sedan, a 7.3 m box truck, a 2.3 m portable toilet), colours (5+6+6+6+5+5+4+6 = 43 pairs) and receptacles (the most
    typical first), all square to their surface (they stand along the kerb); 26-34 items a world, so the never-seen pool
    keeps 43 - 32 = 11; identical toilets and dumpsters; cars, vans and pickups leaving; a van delivered, white first;
    and two movers, a taxi (4.6 x 1.8 x 1.55 m, 4-8 m/s) and a 9 m bus (3-7 m/s), each pausing 3-20 s with the camera
    kept circumradius + 0.35 m away. Regression caught: a kind or colour dropped (the look-alike pressure and the pool
    shrink, the yields were measured on exactly these), a size that drifts from the look-dev's vehicles, kinds reordered
    (the draws are indexed by position), or a mover given a different speed (the `where_at` exploits were scored on
    these)."""
    c = CATALOGS["downtown"]
    assert c.name == "downtown" and list(c.kinds) == [
        "car", "van", "pickup", "box_truck", "skip", "dumpster", "portable_toilet", "food_cart"]
    assert {n: k.size for n, k in c.kinds.items()} == {
        "car": (4.6, 1.8, 1.45), "van": (5.2, 2.0, 2.1), "pickup": (5.4, 2.0, 1.9), "box_truck": (7.3, 2.4, 3.3),
        "skip": (3.0, 1.7, 1.15), "dumpster": (1.8, 1.3, 1.4), "portable_toilet": (1.1, 1.1, 2.3),
        "food_cart": (2.0, 1.0, 2.2)}
    assert [len(k.colors) for k in c.kinds.values()] == [5, 6, 6, 6, 5, 5, 4, 6]
    assert {n: k.colors for n, k in c.kinds.items()} == {
        "car": ("white", "red", "blue", "green", "beige"),
        "van": ("white", "blue", "red", "green", "yellow", "orange"),
        "pickup": ("red", "blue", "white", "green", "orange", "beige"),
        "box_truck": ("white", "yellow", "red", "blue", "green", "orange"),
        "skip": ("yellow", "orange", "green", "blue", "red"),
        "dumpster": ("green", "blue", "red", "orange", "yellow"),
        "portable_toilet": ("blue", "green", "orange", "red"),
        "food_cart": ("red", "blue", "yellow", "green", "white", "orange")}
    assert {n: k.receptacles for n, k in c.kinds.items()} == {
        "car": ("parking",), "van": ("parking", "loading zone"), "pickup": ("parking",),
        "box_truck": ("loading zone",), "skip": ("parking", "loading zone"),
        "dumpster": ("sidewalk", "loading zone"), "portable_toilet": ("plaza", "sidewalk"),
        "food_cart": ("plaza", "sidewalk")}
    assert all(k.square and k.geometry and len(set(k.colors)) == len(k.colors) for k in c.kinds.values())
    assert (c.n_movers, c.n_items) == (2, (26, 34)) and sum(len(k.colors) for k in c.kinds.values()) == 43
    assert c.set_kinds == ("portable_toilet", "dumpster") and c.leaving_kinds == ("car", "van", "pickup")
    assert (c.delivery_kind, c.delivery_colors) == ("van", ("white", "blue", "red", "green", "yellow", "orange"))
    assert c.delivery_colors == c.kinds["van"].colors
    assert all(len(c.kinds[k].colors) >= 2 for k in c.set_kinds)  # a look-alike needs a sibling
    assert {n: (k.size, k.colors, k.speed, k.dwell, k.same_room, k.camera_m) for n, k in c.mover_kinds.items()} == {
        "taxi": ((4.6, 1.8, 1.55), ("yellow", "orange"), (4.0, 8.0), (3, 20), 0.5, 2.82),
        "bus": ((9.0, 2.5, 3.1), ("blue", "red", "white", "green"), (3.0, 7.0), (3, 20), 0.5, 5.02)}
    assert mover_kind("bus") is c.mover_kinds["bus"] and item_size("bus") == (9.0, 2.5, 3.1)
    assert item_size("box_truck") == (7.3, 2.4, 3.3)


def test_no_downtown_kind_has_a_black_or_grey_colour():
    """Contract (VISION rule 10: the colour attribute is what the pixels show): the look-dev
    found black and grey lost in shade at 12-35 m from a drone (a black car and a grey dumpster read as shadow), so no
    carried kind of the city and no mover has either. Regression caught: a black or grey colour added to a kind, whose
    golds a perceiver could not tell from its neighbours."""
    for kind in (*CATALOGS["downtown"].kinds.values(), *CATALOGS["downtown"].mover_kinds.values()):
        assert not {"black", "grey"} & set(kind.colors), kind.name


def test_the_taxis_colours_are_no_cars():
    """Contract (plan-city §3.2): a mover is never a carried kind's look-alike, and the taxi is a sedan with a roof
    sign, so its colours (yellow, orange) are none of the carried car's. Regression caught: a yellow or orange car (a
    parked car told from the taxi by its roof sign alone, which a perceiver at 20 m may not see), or a white taxi."""
    c = CATALOGS["downtown"]
    assert set(c.mover_kinds["taxi"].colors).isdisjoint(c.kinds["car"].colors)


def test_the_home_floor_kinds_are_tracked_kinds_dropped_beside_furniture_the_apartment_has():
    """Contract (plan-redesign §4.10 item 1: real floors hold shoes, a bag, a toy and a box, and `clutter.place` moves
    about half of them off their receptacles): `floor_kinds` names tracked kinds of the household, each with the
    furniture kinds it is dropped beside, written out here from the plan (shoes: floor mat, shoe rack, hall console,
    wardrobe, sofa; bag: desk, hall console, sofa; toy: sofa, coffee table, rug; box: hall console, dining table), and
    every one of those is a piece the apartment has. The warehouse and the downtown name none. Regression caught: a
    floor kind the catalog does not draw (nothing would ever be dropped), an anchor spelt as a name the map has no piece
    of (a drop with no place to go), or a floor rule that leaks into a catalog whose kinds stand on racks and kerbs."""
    home = CATALOGS["home"]
    assert dict(home.floor_kinds) == {
        "shoes": ("floor_mat", "shoe_rack", "hall_console", "wardrobe", "sofa"),
        "bag": ("desk", "hall_console", "sofa"),
        "toy": ("sofa", "coffee_table", "rug"), "box": ("hall_console", "dining_table")}
    assert set(dict(home.floor_kinds)) <= set(home.kinds)
    pieces = {f.kind for f in load_map("apartment_a").furniture if f.mobility == "fixed"}
    assert {a for anchors in dict(home.floor_kinds).values() for a in anchors} <= pieces
    assert CATALOGS["warehouse"].floor_kinds == () == CATALOGS["downtown"].floor_kinds


def test_a_place_token_is_a_surface_kind_or_the_floor_and_may_name_a_room():
    """Contract (`ClutterKind.places`): "floor" and a surface kind such as "kitchen counter" are places, and an `@`
    names the room the kind stays in ("sink edge@bathroom", "floor@bedroom"); `on_floor` is true only of a kind all of
    whose places are the floor. Regression caught: a room name read as part of a surface kind (no surface of the map
    matches it, so the kind never lands), or a kind that stands on a counter placed on the floor."""
    assert place_of("floor") == ("floor", None) and place_of("floor@bedroom") == ("floor", "bedroom")
    assert place_of("sink edge@bathroom") == ("sink edge", "bathroom") and place_of("desk") == ("desk", None)
    box = (0.1, 0.1, 0.1)
    assert ClutterKind("a", box, ("grey",), ("floor", "floor@bedroom"), (0, 1), ("m",)).on_floor
    assert not ClutterKind("b", box, ("grey",), ("floor", "desk"), (0, 1), ("m",)).on_floor
    assert not ClutterKind("c", box, ("grey",), ("kitchen counter",), (0, 1), ("m",)).on_floor
