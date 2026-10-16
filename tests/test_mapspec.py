"""The apartment map spec: realism, self-consistency and the door/wall geometry the renderer will rely on."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from stbench import asset_manifest as A
from stbench.catalog import CATALOGS, KINDS, for_map
from stbench.geometry import Rect, dist_to_rect, point_in_rect, to_world
from stbench.mapspec import (
    DOCK_APRON_FRONT_M,
    DOCK_APRON_SIDE_M,
    CeilingLight,
    MapSpec,
    Outdoor,
    bundled_maps,
    check_map,
    load_map,
    parse_map,
)
from stbench.robots import ROBOTS, passes_under


def test_committed_map_is_self_consistent(apt):
    """Contract: `check_map` finds nothing on the committed apartment: furniture inside rooms and apart, surfaces on
    their furniture, openings in walls, windows on exterior walls, clear door swings. Regression caught: an edit of
    apartment_a.json that leaves a piece outside its room, two pieces overlapping or a door swing blocked, which every
    script on the map would inherit."""
    assert check_map(apt) == []


def test_apartment_is_a_realistic_one_bedroom(apt):
    """Contract: the apartment is about 55 m2 gross, with five named rooms, four interior doors plus the entrance and
    an open kitchen/living arch, doors 0.8-0.95 m wide and walls 0.1 m by 2.5 m. Regression caught: an edit that turns
    it into another plan (a room dropped, a doorway no door fits, walls of another thickness) while `check_map` still
    passes."""
    x0, y0, x1, y1 = apt.bounds
    assert 54.0 <= (x1 - x0) * (y1 - y0) <= 57.0
    assert sorted(r.name for r in apt.regions) == ["bathroom", "bedroom", "hall", "kitchen", "living room"]
    assert len(apt.toggling_doors()) == 4 and len(apt.doors()) == 5
    assert [o.id for o in apt.openings if o.type == "arch"] == ["O1"]
    assert all(0.8 <= o.width <= 0.95 for o in apt.doors())
    assert all(w.thickness == 0.1 and w.height == 2.5 for w in apt.walls)
    assert all(w.exterior for w in apt.walls if w.id in {"W01", "W02", "W03", "W04"})


def test_furniture_heights_are_realistic(apt):
    """Contract: counters 0.9 m, tables 0.75 m, a coffee table lower, a sofa seat near 0.45 m. Regression caught: a
    surface height edited to an unrealistic one (a 0.6 m counter), which `check_map` accepts and every "on the
    counter" answer and render would carry."""
    heights = {}
    for f in apt.furniture:
        for s in f.surfaces:
            heights.setdefault(s.kind, set()).add(s.z)
    assert heights["kitchen counter"] == {0.9}
    assert heights["dining table"] == {0.75} and heights["desk"] == {0.75}
    assert heights["coffee table"] == {0.42} and heights["sofa"] == {0.45}
    assert {0.4, 0.9, 1.4} <= heights["shelf"]


def test_light_furniture_is_four_chairs_a_stool_and_a_basket(apt):
    """Contract: the apartment's light furniture (mobility `light`: each kind has a model of its own in the manifest) is
    four chairs, a stool and a laundry basket. Regression caught: a map edit that adds, drops or re-kinds a light piece
    unnoticed, so the models, the moves and the counts that read this set drift from the map."""
    light = sorted(f.kind for f in apt.furniture if f.mobility == "light")
    assert light == ["chair", "chair", "chair", "chair", "laundry_basket", "stool"]


def test_light_furniture_boxes_follow_their_assets(apt):
    """Contract (M2 plan 4.5): a light piece is drawn from its asset, fitted by height (`uniform`: chairs and the stool)
    or built to its box (`exact`: the laundry basket), so the map's `size` must be the asset's own proportions at the
    piece's height, to 1 mm. Footprints, the route's clearances and the in-view cover all read this box. Regression
    caught: a map box that disagrees with the mesh (the chairs were 0.45 x 0.45, the asset is 0.58 x 0.43), so the
    robot clears and sees a chair that is not the one rendered."""
    m = A.packaged()
    for f in apt.furniture:
        if f.mobility != "light":
            continue
        asset = m.model_for(f.kind)
        native = asset.native_bbox_m
        assert native is not None and asset.mobility == "light", f.id
        scale = f.size[2] / native[2] if asset.fit["mode"] == "uniform" else None
        want = [n * scale for n in native] if scale else list(native)
        if scale is None:
            assert asset.fit["mode"] == "exact", f.id
        assert all(abs(got - exp) <= 0.001 for got, exp in zip(f.size, want, strict=True)), (f.id, f.size, want)


def palette_keys() -> set[str]:
    """The keys of `render/shapes.py`'s PALETTE, read from its source (the module imports bpy)."""
    source = Path(__file__).resolve().parents[1] / "render" / "shapes.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "PALETTE":
            return {key.value for key in node.value.keys}
    raise AssertionError(f"no top-level PALETTE in {source}")


def test_every_light_piece_is_tinted_to_the_colour_the_map_gives_it(apt):
    """Contract (VISION rule 10: the colour attribute is what the pixels show; DECISIONS "Generator 0.4.0"): each light
    piece's asset takes a luminance tint on named slots, and its map colour is a palette colour (the renderer's
    palette) that is not a material word (`palette_materials`: oak, stone, steel name textures, not colours). The stool
    is "brown" and the laundry basket "beige". Regression caught: the laundry basket's tint left at none (labelled
    "beige", it rendered brown wicker: 54 of 54 probe answers "brown", 0 right), or a material word as a colour (the
    stool "oak" was a texture swap that the models called brown, oak or beige)."""
    m = A.packaged()
    palette = palette_keys()
    for f in (f for f in apt.furniture if f.mobility == "light"):
        tint = m.model_for(f.kind).tint
        assert tint["mode"] == "luminance" and tint["materials"], f.id
        assert f.color in palette and f.color not in m.palette_materials, (f.id, f.color)
    assert {f.kind: f.color for f in apt.furniture if f.kind in ("stool", "laundry_basket")} == {
        "stool": "brown", "laundry_basket": "beige"}


def test_every_catalog_receptacle_kind_exists_on_the_map(apt):
    """Contract: every receptacle a catalog kind names is the kind of a surface somewhere on the apartment. Regression
    caught: a typo in a kind's receptacles (or a surface renamed in the map), which would make the 'typical' share
    silently atypical."""
    surface_kinds = {s.kind for f in apt.furniture for s in f.surfaces}
    for kind in KINDS.values():
        assert set(kind.receptacles) <= surface_kinds, kind.name


def flat(leaf):
    return [*leaf[0], *leaf[1]]


def test_door_leaf_closed_and_open_positions(apt):
    """D1 is in the wall x=4.6 (direction +y, left normal -x) at y=1.9, 0.85 wide, hinge at the south jamb, swinging
    left. Hand values: jambs at y=1.475 and 2.325; closed leaf from (4.6,1.475) to (4.6,2.325); open leaf from the
    hinge to (3.75,1.475), into the kitchen. Regression caught: a leaf drawn about the wrong hinge or on the wrong
    side (a door that opens into the other room, so its swept floor and the render disagree)."""
    d1 = next(o for o in apt.openings if o.id == "D1")
    assert flat(apt.door_leaf(d1, False)) == pytest.approx([4.6, 1.475, 4.6, 2.325])
    assert flat(apt.door_leaf(d1, True)) == pytest.approx([4.6, 1.475, 3.75, 1.475])
    assert d1.swings_into == "R3"
    d4 = next(o for o in apt.openings if o.id == "D4")  # hinge b, swings right (+x) into the bedroom
    assert flat(apt.door_leaf(d4, True)) == pytest.approx([6.2, 3.725, 7.05, 3.725])
    assert d4.swings_into == "R4"


def test_wall_pieces_leave_the_openings_open(apt):
    """Wall W05 (x=4.6, y 0..6.2) has D1 (0.85) and D2 (0.9): its solid pieces total 6.2 + 0.1 - 1.75 = 4.55 m long,
    and no piece covers a doorway centre. Regression caught: a wall piece across a doorway (the door walled shut in the
    render and in the planner's grid), or wall length lost at a jamb."""
    pieces = [r for r in apt.wall_pieces() if abs(r.cx - 4.6) < 1e-9 and abs(r.yaw - math.pi / 2) < 1e-9]
    assert sum(r.sx for r in pieces) == pytest.approx(4.55)
    for door in ("D1", "D2"):
        c = next(o for o in apt.openings if o.id == door).center
        assert not any(point_in_rect(r, c[0], c[1]) for r in pieces)


def test_contains_accepts_rooms_and_openings_only(apt):
    """Contract: points in rooms and in door gaps are inside the map; points in plain wall or outside are not.
    Regression caught: a doorway judged outside (a robot could not cross it), or a point in plain wall judged inside."""
    assert apt.contains((2.0, 1.0)) and apt.contains((4.6, 1.9)) and apt.contains((3.6, 2.8))
    assert not apt.contains((4.6, 4.8)) and not apt.contains((-1.0, 1.0))
    assert apt.region_of((4.6, 1.9)) is None and apt.region_of((5.4, 3.0)) == "R1"


def test_rect_inside_rejects_straddling_a_wall(apt):
    """Contract: a chair-sized rectangle centred in the kitchen but reaching into the hall crosses the wall: not
    inside. Regression caught: a rectangle straddling a wall judged inside a room, so a piece or a body could stand
    half through a wall."""
    assert apt.rect_inside(Rect(4.3, 1.0, 0.45, 0.45, 0.0))
    assert not apt.rect_inside(Rect(4.5, 1.0, 0.45, 0.45, 0.0))


def test_check_map_reports_a_broken_spec(apt):
    """Contract: `check_map` fails on bad specs: an overlapping piece, and a door that fits no wall. Regression
    caught: a check that returns no problem whatever the spec, so `test_committed_map_is_self_consistent` passes
    vacuously."""
    bad_furn = replace(apt.furniture[1], center=apt.furniture[2].center)
    broken = replace(apt, furniture=(apt.furniture[0], bad_furn, *apt.furniture[2:]))
    assert any("overlap" in p for p in check_map(broken))
    wide = replace(apt.openings[1], width=9.0)
    too_wide = replace(apt, openings=(apt.openings[0], wide, *apt.openings[2:]))
    assert any("does not fit" in p for p in check_map(too_wide))


def test_a_map_file_with_a_schema_key_still_parses():
    """Contract: `parse_map` reads only the keys it names, so a map file of an earlier build (it carried a `schema`)
    parses to the same map, whatever that key holds, and the bundled files carry none. Regression caught: a loader that
    refuses a map for a key it no longer reads, which stops every private map written before the change from loading."""
    raw = json.loads(map_bytes("apartment_a"))
    assert "schema" not in raw
    assert parse_map({"schema": "old", **raw}) == parse_map({"schema": "other", **raw}) == parse_map(raw)


DOOR_BAND_M = 0.5  # the clear floor out from both faces of the wall, across the width of a doorway
ON_LEGS = {"F07", "F10", "F11", "F20", "F30", "F33", "L01", "L02", "L03", "L04", "L05"}  # the pieces with an underside


def test_the_apartment_map_file_has_no_optional_keys_and_legs_on_eleven_pieces(apt):
    """Contract: the optional keys `catalog` and `watts` are absent from the file (its bytes need neither, see its
    README); and `under_m` is on the eleven pieces that have an open underside (the tables, the sofa, the bed, the hall
    console, the desk, the chairs and the stool) and on no other. Regression caught: the apartment given a
    `catalog: home` or a `watts` for symmetry, or an underside given to a piece that stands on the floor (the planner
    would drive a robot under it)."""
    data = resources.files("stbench.maps").joinpath("apartment_a.json").read_bytes()
    assert b'"catalog"' not in data and b'"watts"' not in data
    assert apt.catalog == "home" and {light.watts for light in apt.ceiling_lights} == {45.0}
    assert {f["id"] for f in json.loads(data)["furniture"] if "under_m" in f} == ON_LEGS


def test_the_vacuum_passes_under_exactly_the_pieces_on_legs(apt):
    """Contract (plan-redesign section 2.1: the vacuum is 0.09 m high and clears an underside of 0.10 m or more, so it
    drives under the tables, the hall console, the desk, the chairs and the stool, and not under the sofa (0.08 m) or
    the bed (0.05 m), whose skirts it would jam in): the pieces whose `under_m` the vacuum passes are the nine of that
    hand list. Regression caught: a vacuum planned under the sofa or the bed, or never under the tables."""
    passes = {f.id for f in apt.furniture if passes_under(ROBOTS["vacuum"], f.under_m)}
    assert passes == {"F07", "F11", "F20", "F33", "L01", "L02", "L03", "L04", "L05"}


def apron_problems(m: MapSpec) -> list[str]:
    """What stops the vacuum undocking from, and driving back onto, the map's one `dock`. Its apron is the floor from
    the dock's back face to `DOCK_APRON_FRONT_M` in front of its front face and `DOCK_APRON_SIDE_M` beyond each side
    (in the dock's frame), sampled every 0.01 m. It must lie inside the dock's room and touch no other blocking
    footprint, no door's swept quarter disc (the hinge, the door's width as radius, between the closed and the open
    leaf) and no opening's band (its width, out `DOOR_BAND_M` from both faces of the wall)."""
    (dock,) = [f for f in m.furniture if f.kind == "dock"]
    depth, width = dock.size[:2]
    apron = Rect(*to_world(dock.footprint(), DOCK_APRON_FRONT_M / 2, 0.0), depth + DOCK_APRON_FRONT_M,
                 width + 2 * DOCK_APRON_SIDE_M, dock.yaw)
    problems = [] if m.rect_inside(apron) else ["leaves its room"]
    lx, ly = np.meshgrid(np.arange(-depth / 2, depth / 2 + DOCK_APRON_FRONT_M + 1e-9, 0.01),
                         np.arange(-apron.sy / 2, apron.sy / 2 + 1e-9, 0.01))
    c, s = math.cos(dock.yaw), math.sin(dock.yaw)
    xs, ys = dock.center[0] + c * lx - s * ly, dock.center[1] + s * lx + c * ly

    def touches(rect: Rect) -> bool:
        return bool((dist_to_rect(xs, ys, rect) < 1e-9).any())

    problems += [f"touches {f.id}" for f in m.furniture if f.blocks and f.id != dock.id and touches(f.footprint())]
    for o in m.openings:
        if touches(Rect(*o.center, o.width, m.wall_thickness + 2 * DOOR_BAND_M, o.yaw)):
            problems.append(f"touches the band of {o.id}")
        if o.type == "door":
            hinge, open_tip = m.door_leaf(o, True)
            swept = np.hypot(xs - hinge[0], ys - hinge[1]) <= o.width
            for tip in (open_tip, m.door_leaf(o, False)[1]):  # the leaves are at a right angle: the sweep is a quadrant
                swept &= (xs - hinge[0]) * (tip[0] - hinge[0]) + (ys - hinge[1]) * (tip[1] - hinge[1]) >= 0
            if swept.any():
                problems.append(f"touches the swing of {o.id}")
    return problems


def test_the_dock_apron_is_irobots_guidance():
    """Contract (plan-redesign section 2.9, iRobot's home-base guidance: 4 ft clear in front of the dock and 1.5 ft at
    each side): the apron that the docking path and the clutter keep-out share is 1.2 m in front and 0.5 m each side.
    These are the plan's numbers, written out once here, because `apron_problems` below reads the constants and would
    shrink with them. Regression caught: an apron cut to 0.3 m in front, which `test_the_dock_apron_is_clear` still
    passes while the vacuum has no room to undock."""
    assert (DOCK_APRON_FRONT_M, DOCK_APRON_SIDE_M) == (1.2, 0.5)


def test_the_dock_apron_is_clear(apt):
    """Contract (iRobot's home-base guidance, `mapspec.DOCK_APRON_FRONT_M` and `DOCK_APRON_SIDE_M`: the vacuum needs a
    clear apron to undock from the dock and to find it again): the apron of the apartment's dock, 1.2 m in front and
    0.5 m each side, lies inside the hall and touches no other piece that blocks, no door's swing and no doorway's band.
    The check is shown to bite once for each of its four rules, by hand. The hall console back where it stood (4.8,
    5.5, yaw 0) is in the apron. The dock moved to x 5.3 (front face 5.36) has its apron's front edge at 5.36 + 1.2
    = 6.56, past the hall's east wall face at 6.15: it leaves its room. The dock at (4.71, 4.5) has its apron at x
    4.65-5.97, y 3.86-5.14, which overlaps the band of door D2 (x 4.05-5.15, y 3.45-4.35). The dock moved into the
    bathroom at (6.9, 0.8) facing north has its apron at x 6.26-7.54, y 0.74-2.06, which covers the quarter disc
    that door D3 sweeps (hinge (6.2, 1.7), radius 0.8, x 6.2-7.0, y 0.9-1.7) as well as its band. Regression caught:
    a dock the vacuum cannot undock from or back onto, because a piece, a door or a doorway was moved into its
    apron; or a rule of this check (the room, the swing) that could never fail."""

    def moved(piece_id: str, **changes) -> MapSpec:
        return replace(apt, furniture=tuple(replace(f, **changes) if f.id == piece_id else f for f in apt.furniture))

    assert apron_problems(apt) == []
    assert apron_problems(moved("F20", center=(4.8, 5.5), yaw=0.0)) == ["touches F20"]
    assert apron_problems(moved("F23", center=(5.3, 5.05))) == ["leaves its room"]
    assert apron_problems(moved("F23", center=(4.71, 4.5))) == ["touches the band of D2"]
    assert apron_problems(moved("F23", center=(6.9, 0.8), yaw=math.pi / 2)) == [
        "touches the band of D3", "touches the swing of D3"]


def test_a_map_names_its_catalog_and_its_lights_their_power_or_gets_the_defaults(apt):
    """Contract (plan-warehouse §1): `catalog` (absent means "home") and `ceiling_lights[i].watts` (absent means 45.0)
    are optional keys of the map file; the loader reads them and `check_map` reports a catalog that does not exist
    (`catalog.for_map` follows the key, test_catalog). Regression caught: a loader that ignores `catalog` (a warehouse
    sampled with household kinds), a default other than the apartment's 45 W (a re-lit apartment), or a typo in a
    catalog name that passes the check and fails in the middle of a generate."""
    raw = json.loads(resources.files("stbench.maps").joinpath("apartment_a.json").read_bytes())
    assert (parse_map(raw).catalog, parse_map(raw).ceiling_lights[0].watts) == ("home", 45.0)
    raw["catalog"], raw["ceiling_lights"][0]["watts"] = "warehouse", 270
    m = parse_map(raw)
    assert (m.catalog, m.ceiling_lights[0].watts, m.ceiling_lights[1].watts) == ("warehouse", 270.0, 45.0)
    assert check_map(replace(apt, catalog="storehouse")) == [
        "unknown catalog 'storehouse'; known: home, warehouse, downtown, residential"]


def test_the_bundled_maps_are_listed_and_hashed_by_their_bytes():
    """Contract: `bundled_maps` lists the maps in `stbench/maps` (the manifest's coverage check asks every one for its
    light furniture), sorted, each of which `load_map` loads by name. Regression caught: a map file added but invisible
    to the coverage, so its light furniture would have no model."""
    assert bundled_maps() == ("apartment_a", "downtown_a", "house_floor", "suburb", "warehouse_a")
    assert [load_map(n).name for n in bundled_maps()] == [
        "apartment_a", "downtown_a", "house_floor", "suburb", "warehouse_a"]


def test_what_a_map_delivers_fits_its_surfaces_and_its_light_pieces_are_no_catalog_kind():
    """Contract (plan-warehouse §2, §3.3: a description's kind word alone says which class a thing is): on every bundled
    map the kind the catalog delivers has a receptacle kind that the map has (a delivery that no surface of the map
    takes would never land: the gap would add nothing, silently), and no light piece (a chair, a pallet jack) is named
    like a carried or mover kind of any catalog. Regression caught: a delivery kind whose receptacles the map lacks, and
    a light piece called `pallet` or `box` (two classes under one description)."""
    taken = set()
    for name in bundled_maps():
        m = load_map(name)
        surfaces = {s.kind for f in m.furniture if f.mobility == "fixed" for s in f.surfaces}
        catalog = for_map(m)
        assert set(catalog.kinds[catalog.delivery_kind].receptacles) & surfaces, name
        taken |= {f.kind for f in m.furniture if f.mobility == "light"}
    names = {k for c in CATALOGS.values() for k in (*c.kinds, *c.mover_kinds)}
    assert {"chair", "pallet_jack", "roll_cage"} <= taken and not taken & names


# -- the optional keys of the city ----------------------------------------
def map_bytes(name: str) -> bytes:
    return resources.files("stbench.maps").joinpath(f"{name}.json").read_bytes()


def test_the_depth_unit_the_sun_and_the_facades_are_optional_keys_that_the_indoor_maps_lack():
    """Contract (plan-city §1): `depth_unit` (absent means "mm"), `outdoor` (absent means indoors) and `walls[i].facade`
    (absent means the plain wall) are optional, so `apartment_a.json` and `warehouse_a.json` have none of them and parse
    to mm, indoors and unstyled walls; `downtown_a` has all three: centimetres, a sun and a facade on each of its twelve
    walls. Regression caught: a key written into an indoor map for symmetry (its hash, which every episode records,
    changes), a loader that ignores a key (the city rendered in the apartment's light and its depth read as
    millimetres), or a default other than the one the kept episodes were written with."""
    for name in ("apartment_a", "warehouse_a"):
        data, m = map_bytes(name), load_map(name)
        assert b'"depth_unit"' not in data and b'"outdoor"' not in data and b'"facade"' not in data, name
        assert m.depth_unit == "mm" and m.outdoor is None and {w.facade for w in m.walls} == {None}, name
    city = load_map("downtown_a")
    assert city.depth_unit == "cm"
    assert city.outdoor == Outdoor(sun_elevation_deg=62.0, sun_rotation_deg=110.0, sun_strength=3.6, sky_strength=0.55)
    assert len(city.walls) == 12 and all(w.facade for w in city.walls)


def test_the_loader_reads_the_optional_keys_and_defaults_them_when_absent():
    """Contract: `parse_map` takes `depth_unit`, the four numbers of `outdoor` and a wall's `facade` from the file and
    leaves a map that names none with the defaults, equal to what the loader gave before the keys existed. Regression
    caught: a key read from the wrong place (a facade on the wrong wall), an `outdoor` that loses a number, or a default
    that makes the apartment parse to an object unequal to itself."""
    raw = json.loads(map_bytes("apartment_a"))
    plain = parse_map(raw)
    assert (plain.depth_unit, plain.outdoor) == ("mm", None) and plain.walls[0].facade is None
    raw["depth_unit"], raw["walls"][1]["facade"] = "cm", "brick_red"
    raw["outdoor"] = {"sun_elevation_deg": 30, "sun_rotation_deg": 200, "sun_strength": 3, "sky_strength": 0.5}
    m = parse_map(raw)
    assert m.depth_unit == "cm" and m.outdoor == Outdoor(30.0, 200.0, 3.0, 0.5)
    assert [w.facade for w in m.walls[:3]] == [None, "brick_red", None]
    assert replace(m, depth_unit="mm", outdoor=None, walls=plain.walls) == plain


def test_check_map_reports_an_unknown_depth_unit():
    """Contract (plan-city §2): a map's `depth_unit` is "mm" or "cm", and nothing else: a camera file may say "m"
    (`camera.DEPTH_UNITS` reads it), but a step of a metre rounds a depth by up to 0.5 m, beyond `rules.DEPTH_TOL_M`
    (0.1 m), so the in-view test would read a visible point as occluded behind its own surface. Regression caught: a
    typo ("cms") that passes the check and fails in the middle of a render, when the encoder meets a unit it does not
    know, or a map that names "m" and is accepted (every depth-based in-view answer on it would be wrong)."""
    apt = load_map("apartment_a")
    for bad in ("cms", "m"):
        problems = check_map(replace(apt, depth_unit=bad))
        assert problems == [f"unsupported depth unit {bad!r}; a map names one of: mm, cm"]
    assert all(check_map(replace(apt, depth_unit=unit)) == [] for unit in ("mm", "cm"))


def test_check_map_holds_the_map_to_the_range_of_its_depth_unit():
    """Contract (plan-city §2, §5.4): a depth image is 16 bits, so it holds 65,535 steps of its unit (mm: 65.535 m,
    cm: 655.35 m), and a map's farthest point, the diagonal of its box (bounds and wall height, in 3D),
    must lie inside it (a depth of 0 reads as "nothing nearer", right only if the range covers every occluder). Hand
    values: bounds 65.5 m long with a 5 m wall give sqrt(65.5^2 + 5^2) = 65.69 m, past the millimetre range although
    the plan view alone (65.5) is inside it; 65.0 m gives 65.19 m, inside; the city, 101.2 x 97.2 x 45, is 147.4 m, so
    millimetres fail and centimetres hold. Regression caught: a check on the floor plan alone (a tall map passes), a
    limit in the wrong unit, or the city given millimetres."""
    apt = load_map("apartment_a")
    inside = replace(apt, bounds=(0.0, 0.0, 65.0, 0.0), wall_height=5.0)
    past = replace(apt, bounds=(0.0, 0.0, 65.5, 0.0), wall_height=5.0)
    assert check_map(inside) == []
    assert check_map(past) == ["the map's 3D diagonal 65.7 m is past the 65.535 m that a depth image in mm holds"]
    assert check_map(replace(past, depth_unit="cm")) == []
    city = load_map("downtown_a")
    assert check_map(city) == []
    assert check_map(replace(city, depth_unit="mm")) == [
        "the map's 3D diagonal 147.4 m is past the 65.535 m that a depth image in mm holds"]


def test_check_map_refuses_ceiling_lights_on_an_outdoor_map():
    """Contract (plan-city §1): an outdoor map has a sun and a sky and no ceilings, so it has no ceiling lights, while
    an indoor map keeps its lights (the apartment's and the warehouse's pass the same check). Regression caught: a
    light left in an outdoor map (the renderer would hang a fixture in the sky), or the check applied to indoor maps."""
    city = load_map("downtown_a")
    lit = replace(city, ceiling_lights=(CeilingLight("R1", (50.0, 13.5, 12.0)),))
    assert check_map(lit) == ["an outdoor map has no ceilings, so it has no ceiling lights"]
    assert check_map(load_map("apartment_a")) == [] and load_map("apartment_a").ceiling_lights
    assert check_map(load_map("warehouse_a")) == []


def test_under_m_and_region_kind_are_optional_keys():
    """Contract (plan §2.2): `furniture[i].under_m` (the height of an open underside; absent means 0, solid to the
    floor) and `regions[i].kind` (the room's kind word; absent means its name) are optional keys, and `check_map`
    requires 0 <= under_m < the piece's height. Written so it holds whichever pieces the file gives an underside.
    Regression caught: a key read wrongly (a kind from the wrong region, an underside on the wrong piece), a default
    other than 0 and the name (every map without the keys would change), or an underside as tall as the piece or below
    the floor (a piece with nothing to stand on)."""
    raw = json.loads(map_bytes("apartment_a"))
    for piece in raw["furniture"]:
        piece.pop("under_m", None)
    for room in raw["regions"]:
        room.pop("kind", None)
    plain = parse_map(raw)
    assert [f.under_m for f in plain.furniture] == [0.0] * len(raw["furniture"])
    assert [r.kind for r in plain.regions] == [room["name"] for room in raw["regions"]]
    assert check_map(plain) == []

    raw["furniture"][2]["under_m"], raw["regions"][1]["kind"] = 0.5, "corridor"
    m = parse_map(raw)
    assert [f.under_m for f in m.furniture] == [0.5 if i == 2 else 0.0 for i in range(len(raw["furniture"]))]
    assert [r.kind for r in m.regions] == [
        "corridor" if i == 1 else room["name"] for i, room in enumerate(raw["regions"])]
    assert check_map(m) == []

    piece = raw["furniture"][2]
    for bad in (-0.1, piece["size"][2]):
        raw["furniture"][2]["under_m"] = bad
        assert check_map(parse_map(raw)) == [f"furniture {piece['id']} ({piece['name']}) has under_m {bad}; it must be "
                                             f"at least 0 and below its height {piece['size'][2]}"]
