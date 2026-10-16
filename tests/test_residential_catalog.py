"""The residential kit's catalog, look and pieces (M4-suburb part a, §1-§3), free: no Blender (the builders import
`bpy`, which `residential_kit.blender_stubbed` stands in for, since a free test reads their `Part`s, not their
meshes). What the meshes do in Blender is `test_residential_blender.py`'s. The catalog is `CATALOGS["residential"]`
once step R registers it, and until then the entry `catalog_residential` documents
(`residential_kit.residential_catalog`)."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from dressing_kit import RENDER, render_module
from residential_kit import KIT, blender_stubbed, kit, residential_catalog
from stbench.catalog import DOWNTOWN_DELIVERY_COLORS, DOWNTOWN_KINDS

SIZES = {  # per kind: the brief's range at both ends and the kit's own (depth, width, height)
    "house": [(8.0, 8.0, 8.0), (11.0, 10.0, 9.0), (9.5, 9.5, 8.4), (10.0, 8.5, 8.8), (9.0, 10.0, 8.6)],
    "block": [(12.0, 24.0, 9.9), (14.0, 40.0, 18.9), (13.0, 30.0, 15.9)],
    "lawn": [(35.0, 7.0, 0.03)], "driveway": [(7.0, 3.2, 0.05)], "path": [(1.2, 7.0, 0.03)],
    "hedge": [(0.6, 9.0, 1.2), (0.8, 15.0, 1.8)], "fence": [(0.1, 16.0, 1.8), (0.1, 3.0, 1.0)],
    "garden_bed": [(4.0, 4.0, 0.3), (1.4, 14.0, 0.3)], "verge": [(1.98, 100.0, 0.15)],
}


def test_the_residential_catalog_shares_the_downtowns_kinds():
    """Contract (M4-suburb §1): the residential catalog asks about exactly the downtown's car, van, pickup, skip and
    dumpster, each the downtown's own `Kind` (one kind, one size, one set of colours and receptacles), with no movers
    and no clutter, dumpsters as its identical sets, cars, vans and pickups leaving and a van arriving in the
    downtown's colours, 18 to 24 items a world (its 27 (kind, colour) pairs less the 22 distinct descriptions a world
    of 24 holds at most leave the 5 the never-seen pool needs; the downtown's 26 to 34 would use the catalog up); and
    every kind has a surface on the kit's map that it can be put on, the delivered van's among them. Regression
    caught: a second size for one model (the same car two lengths on the pictures), a box truck or a food cart in a
    front garden, a world that can hold every description, or a delivery no surface takes."""
    cat = residential_catalog()
    assert cat.name == "residential" and list(cat.kinds) == ["car", "van", "pickup", "skip", "dumpster"]
    assert all(kind is DOWNTOWN_KINDS[name] for name, kind in cat.kinds.items())
    assert (cat.set_kinds, cat.leaving_kinds, cat.n_items) == (("dumpster",), ("car", "van", "pickup"), (18, 24))
    assert sum(len(k.colors) for k in cat.kinds.values()) == 27 and 27 - (cat.n_items[1] - 2) >= 5
    assert (cat.mover_kinds, cat.n_movers, cat.clutter, cat.floor_kinds) == ({}, 0, (), ())
    assert (cat.delivery_kind, cat.delivery_colors) == ("van", DOWNTOWN_DELIVERY_COLORS)
    surface_kinds = {s["kind"] for f in KIT["furniture"] for s in f["surfaces"]}
    assert surface_kinds == {"sidewalk", "parking"}
    for name, kind in cat.kinds.items():  # parking: car, van, pickup, skip; the sidewalk: the dumpster
        assert set(kind.receptacles) & surface_kinds, name
    assert set(cat.kinds["van"].receptacles) & surface_kinds == {"parking"}


def test_the_look_names_the_ground_and_scenery_the_core_hunk_reads():
    """Contract (M4-suburb §2): the residential look is the six region kinds of the design (the suburb uses all but
    `courtyard`), asphalt under the roads and paving under the rest in the placeholder render (it has no grass), no
    wear and no ceiling lamps, the downtown's sky and sun value for value, `GROUND = "lawn"` and `SCENERY =
    "residential"` (the two keys `build_scene` reads beyond the tables, so `render/scenery_residential.py` must
    exist), and its dressing module is `residential`'s; every region kind of the kit is one of them. Regression
    caught: a kind used by a map and missing from the vocabulary (a `LookError` in the middle of a render), a sky that
    differs from the downtown's by one number, or a scenery key naming a module that is not there."""
    look = render_module("look")
    mod, downtown = look.module("residential"), look.module("downtown")
    kinds, floors = mod.KINDS, mod.PLACEHOLDER_FLOOR
    assert kinds == {"street", "garden", "car park", "green", "passage", "courtyard"}
    assert floors == {"street": "asphalt", "car park": "asphalt", "garden": "paving", "green": "paving",
                      "passage": "paving", "courtyard": "paving"}
    assert mod.FLOOR_FINISH is None and mod.RIG["lux"] is None and mod.RIG["cct_k"] is None
    assert mod.RIG == downtown.RIG
    assert (mod.GROUND, mod.SCENERY, mod.DRESSING) == ("lawn", "residential", "residential")
    assert not hasattr(downtown, "GROUND") and not hasattr(downtown, "SCENERY")  # the core hunk defaults for it
    for stem in (f"scenery_{mod.SCENERY}", f"dressing_layout_{mod.DRESSING}", f"dressing_build_{mod.DRESSING}"):
        assert (RENDER / f"{stem}.py").is_file(), stem
    assert {look.kind_of(r) for r in kit()["regions"]} <= mod.KINDS


def _extent(part) -> tuple[float, ...]:
    """The axis-aligned box of a part: a box turned about y (the roof planes) or not turned at all, a wedge."""
    (x, y, z), (dx, dy, dz) = part.loc, part.dims
    turn = abs(part.rot[1])
    assert part.rot[0] == part.rot[2] == 0.0 and part.shape in ("box", "wedge")
    c, s = math.cos(turn), math.sin(turn)
    hx, hz = (c * dx + s * dz) / 2, (s * dx + c * dz) / 2
    if part.shape == "wedge":  # `loc` is the centre of its base
        return x - hx, x + hx, y - dy / 2, y + dy / 2, z, z + dz
    return x - hx, x + hx, y - dy / 2, y + dy / 2, z - hz, z + hz


@pytest.fixture(scope="module")
def pieces():
    """Every residential builder's parts at every size of `SIZES` in three colours, loaded once under stubbed Blender
    modules (the stubs stay for the module's tests and go with its last)."""
    with blender_stubbed():
        furniture, materials, shapes = (render_module(n) for n in ("furniture_residential", "materials_residential",
                                                                   "shapes"))
        house = render_module("procedural_house")
        built = {(kind, size, color): furniture.BUILDERS[kind](size, color, ())
                 for kind, sizes in SIZES.items() for size in sizes for color in ("beige", "red", "grey")}
        yield {"built": built, "builders": set(furniture.BUILDERS), "roofs": set(materials.ROOFS),
               "stems": set(materials.NAMES), "palette": set(shapes.PALETTE), "house": house}


def test_every_residential_builder_stays_in_its_box_solid_to_the_ground_with_materials_the_scene_can_make(pieces):
    """Contract (M4-suburb §3; the fixed pieces are blocking footprints and solid to the ground, drawn by palette keys
    and the residential keys alone): there is a builder for every kind the brief lists (house, block, lawn, driveway,
    path, hedge, fence, garden bed, verge); at every size and colour, every part lies in the piece's box (to a
    micron), none below the ground, and the lowest stands on it; every part names a palette key, `paving` (the city's)
    or a residential key (`lawn`, `hedge`, `fence`, `driveway` and `mulch` alone, `siding`, `render` and `brick` after
    a colon and a colour, `shingle` after a colon and a roof of `ROOFS`). Regression caught: a part outside the box the
    planner keeps other pieces and the drone out of, a floating piece, or a key no resolver can make (a `KeyError` at
    the first render of a map that holds the piece)."""
    assert pieces["builders"] == set(SIZES)
    needs_arg = {"siding", "render", "brick", "shingle"}
    for (kind, size, color), parts in pieces["built"].items():
        boxes = [_extent(p) for p in parts]
        low = [min(b[i] for b in boxes) for i in (0, 2, 4)]
        high = [max(b[i] for b in boxes) for i in (1, 3, 5)]
        assert all(v >= -h - 1e-6 for v, h in zip(low[:2], (size[0] / 2, size[1] / 2), strict=True)), (kind, size, low)
        assert all(v <= h + 1e-6 for v, h in zip(high, (size[0] / 2, size[1] / 2, size[2]), strict=True)), (kind, size)
        assert low[2] == pytest.approx(0.0, abs=1e-9), (kind, size, color)
        for p in parts:
            stem, _, arg = p.mat.partition(":")
            residential = stem in pieces["stems"] and bool(arg) == (stem in needs_arg)
            assert p.mat in pieces["palette"] | {"paving"} or residential, (kind, p.mat)
            assert stem != "shingle" or arg in pieces["roofs"], (kind, p.mat)


def test_a_house_is_walls_under_two_roof_planes_that_meet_at_the_ridge_with_the_eaves_over_the_walls(pieces):
    """Contract (M4-suburb §3: a drone at 15 m over 8 m houses sees roofs, so the roof is drawn whole): the house of
    the kit (9.5 deep, 9.5 wide, 8.4 high) has walls 8.8 by 9.0 and 5.5 m high (its eaves), two roof planes whose top
    surfaces meet at the ridge line, 0.06 m under the top of the piece, and come down over the walls' top edge 0.12 m
    thick (the underside meeting the wall top there), its highest point (the ridge cap and the chimney) is the piece's
    height, and the house has a garage door because it is 9.5 m wide, where a house 8.5 m wide has none. By hand:
    walls at x +-4.4 and y +-4.5; the plane's top passes (0, 8.34) and (4.4, 5.5 + 0.12 sec of the pitch). Regression
    caught: a roof drawn at the wrong height (the ridge in the depth image), planes that leave a gap at the ridge or
    float over the wall, a house taller than its box, or a garage for every house."""
    parts = pieces["built"]["house", (9.5, 9.5, 8.4), "beige"]
    walls = parts[0]
    assert walls.dims == pytest.approx((8.8, 9.0, 5.5)) and walls.loc == pytest.approx((0.0, 0.0, 2.75))
    planes = [p for p in parts if p.rot[1] != 0.0]
    assert len(planes) == 2 and planes[0].rot[1] == -planes[1].rot[1]
    for p in planes:
        sign, angle = math.copysign(1.0, p.rot[1]), abs(p.rot[1])
        c, s, length, thick = math.cos(angle), math.sin(angle), p.dims[0], p.dims[2]
        # the top surface's two ends, along the plane's own x axis from its centre
        ridge_x = p.loc[0] - sign * (length / 2 * c - thick / 2 * s)
        ridge_z = p.loc[2] + length / 2 * s + thick / 2 * c
        eave_x = p.loc[0] + sign * (length / 2 * c + thick / 2 * s)
        eave_z = p.loc[2] - length / 2 * s + thick / 2 * c
        assert ridge_x == pytest.approx(0.0, abs=1e-9) and ridge_z == pytest.approx(8.34, abs=1e-9)
        assert abs(eave_x) == pytest.approx(4.75, abs=1e-9) and p.dims[1] == pytest.approx(9.5)
        wall_x = 4.4  # where the plane's top is above the wall's outer face: its underside is the wall top, 5.5 m
        top_at_wall = ridge_z - (ridge_z - eave_z) * wall_x / 4.75
        assert top_at_wall - thick / c == pytest.approx(5.5, abs=0.02)
    assert max(_extent(p)[5] for p in parts) == pytest.approx(8.4, abs=1e-9)
    assert Counter(p.mat for p in parts)["door"] == 1  # the garage door's panel; the entrance is `door_entrance`
    assert "door" not in {p.mat for p in pieces["built"]["house", (10.0, 8.5, 8.8), "red"]}


def test_a_block_has_its_storeys_a_flat_roof_inside_a_parapet_and_an_entrance(pieces):
    """Contract (M4-suburb §3: an apartment block, 3 to 6 storeys of 3.0 m, a flat roof with a parapet, rows of windows,
    an entrance): a block 15.9 m high is five storeys under a 0.9 m parapet, its walls 0.4 m inside the box, its roof
    deck a few centimetres over 15.0 m inside four parapet walls that reach the box's height, and a window of five
    rows in each of its bays (30 m long: 9 bays on each long face), none on the ground floor of the entrance bay.
    By hand: the walls 12.2 by 29.2, the deck at 15.03, the parapet to 15.9, and (9 bays x 5 rows - 1 for the
    entrance) window frames on the front, 9 x 5 on the back, 2 x 4 x 5 on the gables (12.2 m: 4 bays). Regression
    caught: a roof under its parapet (the drone's depth at the wrong height), storeys that do not add up to the
    height, or an entrance that is a window."""
    house = pieces["house"]
    assert [house.storeys(h) for h in (9.9, 12.9, 15.9, 18.9)] == [3, 4, 5, 6]
    parts = pieces["built"]["block", (13.0, 30.0, 15.9), "red"]
    assert parts[0].dims == pytest.approx((12.2, 29.2, 15.0))
    assert max(_extent(p)[5] for p in parts) == pytest.approx(15.9)
    assert {round(_extent(p)[5], 2) for p in parts if p.mat == "dark" and p.dims[2] == pytest.approx(0.03)} == {15.03}
    assert sum(1 for p in parts if p.mat == "stone" and p.dims[2] > 0.85) >= 4  # the parapet's four walls
    frames = [p for p in parts if p.mat == "frame" and p.dims[2] == pytest.approx(1.4)]
    assert len(frames) == (9 * 5 - 1) + 9 * 5 + 2 * 4 * 5
