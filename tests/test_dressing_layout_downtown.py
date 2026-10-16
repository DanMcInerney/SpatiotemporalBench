"""The downtown's Phase 1 dressing layout (`render/dressing_layout_downtown.py`): what is particular to it. The rules
every layout must meet (classes registered, every class used, no box in a forbidden volume or outside an allowed place,
a layout that depends on the map alone) are `test_dressing.py`'s; here are the things a block has on its street and the
numbers they are placed by, worked out by hand from `downtown_a.json` (sidewalks 4 m deep whose kerb strip is local x
1.0 to 2.0 from the middle, 0.15 m high; paving in two pieces round the nest's bare yard, 30 x 13 m and 20 x 11 m, with
five 3.5 m pads and the building walls round three of the square's sides; the tallest tracked kind 3.3 m; the drone 12 m
up clearing what is under 9 m). The layout needs no Blender, but it imports `stbench`."""

from __future__ import annotations

import ast
import math
from collections import Counter
from functools import cache

import pytest

from dressing_kit import RENDER, layout_of, raw_map, render_module
from stbench.catalog import CATALOGS
from stbench.geometry import Rect, corners, rects_overlap, to_local
from stbench.mapspec import load_map, parse_map, surface_world_rect
from stbench.scenery import DRESSING, class_problems, dressing_of, names_a_kind

SCENERY = render_module("scenery_layout")
WALK_Z = 0.15  # a sidewalk's top: what its kerb strip's pieces stand on
FLUSH_M = 0.02
DRONE_CLEARS_M = 9.0  # the flying profile passes over what is lower than its 12 m less its 3 m clearance
STRIP_CLASSES = ("street_tree", "bench", "planter", "bike_rack")
FLAT = ("lane_marking", "crosswalk", "stop_bar", "manhole_cover", "storm_drain")


@cache
def _spec() -> dict:
    return raw_map("downtown_a")


@cache
def _layout() -> tuple:
    return tuple(layout_of("downtown", _spec()))


def of(*classes: str) -> list:
    return [d for d in _layout() if d.cls in classes]


def corners_of(d) -> list[tuple[float, float]]:
    """The four corners of a Dressing's box as it stands, turned by its yaw (the box itself is axis-aligned)."""
    return corners(Rect(d.pos[0], d.pos[1], d.size[0], d.size[1], math.radians(d.yaw_deg)))


def sidewalk_of(d) -> dict | None:
    """The sidewalk the piece stands on (its centre in the sidewalk's footprint), else None."""
    for f in _spec()["furniture"]:
        if f["kind"] == "sidewalk":
            lx, ly = to_local(Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"]), d.pos[0], d.pos[1])
            if abs(lx) <= f["size"][0] / 2 and abs(ly) <= f["size"][1] / 2:
                return f
    return None


def plaza_of(d) -> dict:
    """The piece of paving the piece stands on (its centre in the plaza's footprint)."""
    for f in _spec()["furniture"]:
        if f["kind"] == "plaza":
            lx, ly = to_local(Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"]), d.pos[0], d.pos[1])
            if abs(lx) <= f["size"][0] / 2 and abs(ly) <= f["size"][1] / 2:
                return f
    raise AssertionError(f"{d.cls} at {d.pos} stands on no paving")


def local_corners(d, f) -> list[tuple[float, float]]:
    frame = Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"])
    return [to_local(frame, x, y) for x, y in corners_of(d)]


def overlap_2d(a, b) -> bool:
    return a[0] < b[3] - 1e-9 and b[0] < a[3] - 1e-9 and a[1] < b[4] - 1e-9 and b[1] < a[4] - 1e-9


def test_the_downtowns_registered_classes_are_disjoint_from_every_tracked_kind():
    """Contract (the look-alike rule for the city: nothing may look like a parked or
    moving vehicle, a skip, a dumpster, a portable toilet or a food cart; no kiosk or phone box, no parasol or cart):
    the downtown's phase 1 classes are the ten the layout draws, in the order the registry lists them, and none declares
    a resemblance (a class that resembles a tracked kind would be a thing the mission tells the robot to ignore) or is
    named like a kind or a synonym of one (`names_a_kind`, over the downtown's carried and mover kinds). Regression
    caught: a class added for the city under a vehicle's, container's or cart's name or look (a `kiosk`, a `parasol`, a
    parked `car`) with nothing declared."""
    classes = dressing_of("downtown", 1)
    assert [c.name for c in classes] == ["lane_marking", "crosswalk", "stop_bar", "manhole_cover", "storm_drain",
                                         "street_tree", "tree_crown", "bench", "planter", "bike_rack"]
    assert dressing_of("downtown", 0) == [] and dressing_of("downtown", 2) == []
    tracked = {*CATALOGS["downtown"].kinds, *CATALOGS["downtown"].mover_kinds}
    assert tracked == {"car", "van", "pickup", "box_truck", "skip", "dumpster", "portable_toilet", "food_cart", "taxi",
                       "bus"}
    for c in classes:
        assert c.resembles == () and c.catalogs == ("downtown",) and not class_problems(c), c.name
        assert names_a_kind(c.name, tracked) == [], c.name
    assert not {"kiosk", "phone_box", "parasol", "food_cart", "car", "van", "skip", "dumpster"} & set(DRESSING)


def test_the_layout_holds_what_a_block_has_on_its_street_in_numbers_worked_out_by_the_stations():
    """Contract (plan-realism section 4, "City": street trees, benches, planters and bike racks in the kerb strip, plus
    the paint and iron of the road): the high street's and the side streets' sidewalks (four 19 to 38 m long on the
    north, one 100 m on the south, four 45 m on the side streets: 356 m of kerb) take about one item in ten metres,
    trees half of them, so the map has between 10 and 24 street trees on its sidewalks (and a crown with each), 1 or
    more benches (3 are stationed, and the columns of the two street zones, at the south ends of their loading zones,
    take the two at (18.5, 42.38) and (81.5, 38.84)), 4 or more planters and 3 or more bike racks, the square's walled
    edges at least 1 tree, 1 bench and 1 planter of their own (they would have 5 trees, 5 benches and 2 planters but for
    the columns of the drop zones and the nest, which take the rest: `test_dressing_downtown_zones.py`), the four
    crossings, six stop bars, five centre lines and six runs of dashes of the paint, the yard's two white lines (13
    `lane_marking` in all), and at least 5 manholes and 6 storm drains. Regression caught: a layout that stations
    nothing (an empty street), or fills the strip so full of crowns that the pictures lose their cars."""
    counts = Counter(d.cls for d in _layout())
    on_walks = Counter(d.cls for d in _layout() if sidewalk_of(d) is not None)
    assert 10 <= on_walks["street_tree"] <= 24 and counts["tree_crown"] == counts["street_tree"]
    assert on_walks["bench"] >= 1 and on_walks["planter"] >= 4 and on_walks["bike_rack"] >= 3
    square = Counter(d.cls for d in _layout() if sidewalk_of(d) is None and d.cls in STRIP_CLASSES)
    assert square["street_tree"] >= 1 and square["bench"] >= 1 and square["planter"] >= 1
    assert counts["crosswalk"] == 4 and counts["stop_bar"] == 6 and counts["lane_marking"] == 13
    assert counts["manhole_cover"] >= 5 and counts["storm_drain"] >= 6


def test_every_kerb_strip_item_stands_in_the_first_metre_from_the_kerb_clear_of_the_posts_and_the_strips_ends():
    """Contract (the kerb strip `scenery_layout` reserves: the first `KERB_M` = 1 m of a sidewalk from the kerb, local x
    1.0 to 2.0 in the sidewalk's own frame (x towards the kerb, the surface ending at 1.0), where no receptacle surface
    is; the lamp posts, signal poles and hydrants of the scenery layout stand in the same strip): every tree pit, bench,
    planter and bike rack on a sidewalk has all four corners of its box in that strip and at least 4 m from either end
    of the sidewalk (the signal poles stand 0.5 m in from an end), and its box is at least 0.6 m from every post's box
    (a lamp's, a signal's, a hydrant's) of the strip. Regression caught: a piece on the sidewalk's surface (a bench
    where a dumpster is put), over the kerb into the road, or against a lamp post."""
    posts = SCENERY.layout(load_map("downtown_a")).posts
    checked = 0
    for d in of(*STRIP_CLASSES):
        f = sidewalk_of(d)
        if f is None:
            continue
        checked += 1
        local = local_corners(d, f)
        assert all(1.0 - 1e-6 <= x <= 2.0 + 1e-6 for x, _ in local), (d.cls, d.pos, local)
        assert all(abs(y) <= f["size"][1] / 2 - 4.0 + 1e-6 for _, y in local), (d.cls, d.pos)
        assert d.pos[2] == pytest.approx(WALK_Z)
        for p in posts:
            if p.kind != "tree" and p.rect:  # the scenery's own trees are beyond the map: not in these strips
                piece = Rect(d.pos[0], d.pos[1], d.size[0], d.size[1], math.radians(d.yaw_deg))
                assert not rects_overlap(piece, p.rect, margin=0.6), (d.cls, d.pos, p.kind, (p.x, p.y))
    assert checked >= 20  # a station about every 10 m along 356 m of kerb strip, and some stations are skipped


def test_a_tree_is_a_stem_in_a_pit_and_a_crown_above_the_tallest_tracked_thing_over_the_sidewalk_only():
    """Contract (the look-alike and occlusion rules: dressing hides only what is not tracked and never crosses a tracked
    kind's volume; the drone flies over what is lower than 9 m): each street tree is two pieces, a stem standing on the
    sidewalk (z 0.15, 3.25 m of it, to 3.4 m) and a crown whose foot is at 3.4 m, above the 3.3 m of the tallest tracked
    kind (a box truck) and the 2.3 m of the tallest thing that may stand on a sidewalk or in the square, 2.9 to 3.5 m
    tall (to 6.3 to 6.9 m, under the 9 m the drone clears), 1.8 to 2.1 m across and 2.3 to 3.0 m along the sidewalk (no
    two trees alike), centred 0.6 m from the stem towards the building on a sidewalk (so that it hangs over the walk and
    never past the kerb at local x 2.0) and over the stem in the square, wholly inside the footprint of the sidewalk or
    the paving it stands on. Regression caught: a crown over the road (a canopy across a mover's lane), a crown low
    enough to hide a portable toilet, or a tree tall enough to be flown through."""
    kinds = CATALOGS["downtown"].kinds
    assert max(k.size[2] for k in kinds.values()) == 3.3
    stems, crowns = of("street_tree"), of("tree_crown")
    assert len(stems) == len(crowns) >= 14
    for s, c in zip(stems, crowns, strict=True):
        assert s.size == (0.8, 1.0, 3.25)
        assert 1.8 <= c.size[0] <= 2.1 and 2.3 <= c.size[1] <= 3.0 and 2.9 <= c.size[2] <= 3.5
        assert s.pos[2] == pytest.approx(0.15) and c.pos[2] == pytest.approx(3.4)
        assert c.box[5] == pytest.approx(3.4 + c.size[2]) and c.box[5] < DRONE_CLEARS_M - 2.0 and c.pos[2] > 3.3
        walk = sidewalk_of(c)
        d = math.dist(s.pos[:2], c.pos[:2])
        assert d == pytest.approx(0.6 if walk is not None else 0.0, abs=1e-6), (s.pos, c.pos)
        f = walk or plaza_of(c)
        frame = Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"])
        for x, y in corners_of(c):
            lx, ly = to_local(frame, x, y)
            assert abs(lx) <= f["size"][0] / 2 + 1e-6 and abs(ly) <= f["size"][1] / 2 + 1e-6, (c.pos, lx, ly)


def test_paint_and_iron_are_flush_plates_on_the_road_that_never_overlap_each_other():
    """Contract (the guard's flush floor: a plate no thicker than 2 cm needs no clearance; a layout puts none on top of
    another, which would flicker): every piece of paint, every manhole cover and every storm drain stands on the ground
    (z 0) and is 2 cm high or less (paint 1.2 cm, iron 1.4 cm), no two of them overlap in plan and none overlaps the
    footprint of any of the map's 28 furniture pieces (sidewalks, paving, parking lanes, loading zones: entities whose
    pixels it would hide). A manhole lies in a travel lane (the high street's y 6.6 to 20.4, or a side street's x 21.6
    to 28.4 and 71.6 to 78.4, 3.45 m lanes between the parking lanes) and a storm drain stands 0.3 m beyond a sidewalk's
    kerb face, its long side along the kerb. Regression caught: a cover in a parking bay, a drain on the sidewalk or in
    the middle of the street, paint drawn twice over the same place (two coplanar layers)."""
    flat = of(*FLAT)
    assert all(d.pos[2] == 0.0 and d.size[2] <= FLUSH_M for d in flat)
    assert {d.cls: max(x.size[2] for x in flat if x.cls == d.cls) for d in flat} == {
        "lane_marking": 0.012, "crosswalk": 0.012, "stop_bar": 0.012, "manhole_cover": 0.014, "storm_drain": 0.014}
    assert not [(a.cls, b.cls) for i, a in enumerate(flat) for b in flat[i + 1:] if overlap_2d(a.box, b.box)]
    pieces = []  # the map's own ground pieces are entities: a plate over one would hide its pixels from the index pass
    for f in load_map("downtown_a").furniture:
        xs, ys = [p[0] for p in corners(f.footprint())], [p[1] for p in corners(f.footprint())]
        pieces.append((f.kind, (min(xs), min(ys), 0.0, max(xs), max(ys), 0.0)))
    assert len(pieces) == 28
    assert not [(d.cls, kind) for d in flat for kind, box in pieces if overlap_2d(d.box, box)]
    for d in of("manhole_cover"):
        x, y = d.pos[:2]
        assert (6.6 <= y <= 20.4 and 0.0 <= x <= 100.0) or (21.6 <= x <= 28.4 and 27.0 <= y <= 72.0) \
            or (71.6 <= x <= 78.4 and 27.0 <= y <= 72.0), (x, y)
    for d in of("storm_drain"):
        beside = []
        for f in (f for f in _spec()["furniture"] if f["kind"] == "sidewalk"):
            lx, ly = to_local(Rect(f["center"][0], f["center"][1], 0.0, 0.0, f["yaw"]), *d.pos[:2])
            if abs(lx - 2.3) < 1e-6 and abs(ly) <= f["size"][1] / 2:
                beside.append(f)
        assert len(beside) == 1 and d.size == (0.5, 0.9, 0.014), (d.pos, beside)
        assert math.cos(math.radians(d.yaw_deg) - beside[0]["yaw"]) == pytest.approx(1.0, abs=1e-6)  # faces the kerb


def test_the_square_has_its_trees_benches_and_planters_along_the_walls_and_none_by_its_pads_or_its_mouth():
    """Contract (the square's rule: pieces on its paving, clear of the five pads' volumes and of the street, along the
    edges a building wall closes): the paving is x 35 to 65 and y -24 to 0 less the nest's yard (x 35 to 45, y -24
    to -13), in two pieces (F21: x 35 to 65, y -13 to 0; F24: x 45 to 65, y -24 to -13), with walls along its north
    (face at y = -24), west (x = 35) and east (x = 65) sides and its mouth open to the street at y = 0; its pads are
    3.5 m squares at (40, -6), (50, -6), (60, -6), (50, -18) and (60, -18). Every item of the square stands inside the
    paving, at least 1 m from a pad, at least 2 m from the mouth (y <= -2); a bench has its back within 1.2 m of a wall
    (its centre 0.9 m in) and faces into the square; a tree stands 2.3 m in and a planter 1.0 m in. Only three stand:
    the columns of the drop zones and the nest (`test_dressing_downtown_zones.py`) take the rest of the walled edges,
    and what is left is on the north wall between the nest's column and the south drop zone's. Regression caught: a
    tree on a pad (a thing put down on a pad would stand inside its stem), a bench across the open mouth, or an item on
    the wrong side of a wall."""
    m = load_map("downtown_a")
    pads = [s_rect for f in m.furniture if f.kind == "plaza" for s_rect in
            (surface_world_rect(f.center, f.yaw, s) for s in f.surfaces)]
    assert len(pads) == 5
    pad_boxes = [(min(p[0] for p in corners(r)), min(p[1] for p in corners(r)), max(p[0] for p in corners(r)),
                  max(p[1] for p in corners(r))) for r in pads]
    items = [d for d in of(*STRIP_CLASSES) if sidewalk_of(d) is None]
    assert len(items) == 3  # a tree, a bench and a planter on the north wall: the rest is in a column
    for d in items:
        xs, ys = [c[0] for c in corners_of(d)], [c[1] for c in corners_of(d)]
        assert min(xs) >= 35.0 and max(xs) <= 65.0 and min(ys) >= -24.0 and max(ys) <= -2.0, (d.cls, d.pos)
        for x0, y0, x1, y1 in pad_boxes:
            clear = min(xs) >= x1 + 1.0 or max(xs) <= x0 - 1.0 or min(ys) >= y1 + 1.0 or max(ys) <= y0 - 1.0
            assert clear, (d.cls, d.pos)
        wall = min(d.pos[1] + 24.0, d.pos[0] - 35.0, 65.0 - d.pos[0])  # the distance to the nearest wall face
        inset = {"bench": 0.9, "street_tree": 2.3, "planter": 1.0}.get(d.cls)
        if inset is not None:
            assert wall == pytest.approx(inset, abs=0.01) or d.cls == "planter", (d.cls, d.pos, wall)
    for b in (d for d in items if d.cls == "bench"):  # faces into the square: the back is on the wall side
        facing = (math.cos(math.radians(b.yaw_deg)), math.sin(math.radians(b.yaw_deg)))
        centre_dir = (50.0 - b.pos[0], -12.0 - b.pos[1])
        assert facing[0] * centre_dir[0] + facing[1] * centre_dir[1] > 0.0, (b.pos, b.yaw_deg)


def test_recipes_name_the_builders_of_the_class_and_the_paint_text_is_a_known_style_and_axis():
    """Contract (`dressing.build` dispatches `proc:<builder>[:<text>]` to `dressing_build_downtown.BUILDERS`; nothing
    here is an asset, so the scene needs no download): every recipe is `proc:` and its builder is one the builders
    module exports (read from its source: it needs Blender to import), by class: paint for the three painted classes
    (`double`, `dash` or, for the yard's lines, `bar` on x or y for a lane marking, `zebra` for a crossing, `bar` for a
    stop bar), `manhole`,
    `storm_drain`, `street_tree`, `tree_crown`, `bench`, `planter` and `bike_rack` for the others. Regression caught: a
    class with no builder (the scene would not build), or a paint style the builder cannot draw."""
    tree = ast.parse((RENDER / "dressing_build_downtown.py").read_text(encoding="utf-8"))
    table = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BUILDERS")
    names = {e.id for e in table.value.generators[0].iter.elts}
    assert names == {"paint", "manhole", "storm_drain", "street_tree", "tree_crown", "bench", "planter", "bike_rack"}
    expect = {"lane_marking": "paint", "crosswalk": "paint", "stop_bar": "paint", "manhole_cover": "manhole",
              "storm_drain": "storm_drain", "street_tree": "street_tree", "tree_crown": "tree_crown", "bench": "bench",
              "planter": "planter", "bike_rack": "bike_rack"}
    styles = {"lane_marking": {"double", "dash", "bar"}, "crosswalk": {"zebra"}, "stop_bar": {"bar"}}
    for d in _layout():
        kind, _, rest = d.recipe.partition(":")
        builder, _, text = rest.partition(":")
        assert kind == "proc" and builder == expect[d.cls] and builder in names, d.recipe
        if builder == "paint":
            style, axis = text.split(":")
            assert style in styles[d.cls] and axis in ("x", "y") and d.yaw_deg == 0.0, d.recipe
        elif builder == "tree_crown":
            assert text in {"0", "1", "2", "3", "4", "5"}, d.recipe  # which of six shapes
        else:
            assert text == "", d.recipe


def test_the_stations_come_from_the_maps_name_and_a_catalog_that_is_not_the_downtowns_is_refused():
    """Contract (VISION rule 13, `dressing_spec`: any randomness is seeded from the map's name): a map called something
    else has the same paint (the marks come from the map's streets, not the seed) and some of the stationed pieces
    elsewhere (which kind stands where, the gaps between them, which lane a manhole is in). The apartment's spec is not
    a downtown's: `layout` raises ValueError. Regression caught: a seed that ignores the name, a layout that quietly
    dresses a map of another catalog."""
    other = layout_of("downtown", {**_spec(), "name": "downtown_b"})
    assert [d for d in other if d.cls in ("lane_marking", "crosswalk", "stop_bar")] == [
        d for d in _layout() if d.cls in ("lane_marking", "crosswalk", "stop_bar")]
    assert [d.pos for d in other if d.cls == "street_tree"] != [d.pos for d in of("street_tree")]
    with pytest.raises(ValueError, match="not a downtown map"):
        layout_of("downtown", raw_map("apartment_a"))
    assert parse_map(_spec()).catalog == "downtown"
