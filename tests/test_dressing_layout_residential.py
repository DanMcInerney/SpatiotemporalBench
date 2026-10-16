"""The residential dressing layout and far scenery (`render/dressing_layout_residential.py`,
`render/scenery_residential.py`, M4-suburb §4) on the kit map (`residential_kit`: 100 m by 60 m, hand-worked there):
what the layout emits and where, that the dressing guard accepts it, that nothing tall stands by a drop zone or the
nest, and that the scenery lies outside the bounds. The rules every layout must meet on a bundled map are
`test_dressing.py`'s and run on the suburb when part b lands. Free: both modules are plain Python. The catalog and the
dressing classes step R registers are put in the registries here for the module (`residential_kit.register`), so the
tests hold before it and after."""

from __future__ import annotations

import copy
import math
from collections import Counter
from functools import cache

import pytest

from dressing_kit import render_module
from residential_kit import KIT, kit, register, residential_catalog, tall_near
from stbench.dressing_zones import DressingGuard
from stbench.geometry import Rect
from stbench.mapspec import check_map, parse_map
from stbench.scenery import class_problems, dressing_of, layout_problems, names_a_kind

LAYOUT, SCENERY = render_module("dressing_layout_residential"), render_module("scenery_residential")
AABB = render_module("scenery_layout").aabb
CLASSES = ["tree_stem", "tree_canopy", "lamp_post", "shrub", "playground"]  # the registry's rows, by hand
Z_VERGE, Z_BED, STRUCTURE_M, WALLS_M = 0.15, 0.3, 5.0, 12.0
Box = tuple[float, float, float, float]


@pytest.fixture(scope="module", autouse=True)
def step_r():
    with pytest.MonkeyPatch.context() as patch:
        register(patch)
        yield


@cache
def _layout() -> tuple:
    return tuple(LAYOUT.layout(kit()))


def of(*classes: str) -> list:
    return [d for d in _layout() if d.cls in classes]


def footprint(f: dict) -> Box:
    """x0, x1, y0, y1 of a piece of the spec (a quarter turn)."""
    sx, sy = f["size"][0] / 2, f["size"][1] / 2
    hx, hy = (sx, sy) if abs(math.cos(f["yaw"])) > 0.5 else (sy, sx)
    return f["center"][0] - hx, f["center"][0] + hx, f["center"][1] - hy, f["center"][1] + hy


def inside(inner: Box, outer: Box, margin: float = 1e-6) -> bool:
    return (outer[0] - margin <= inner[0] and inner[1] <= outer[1] + margin
            and outer[2] - margin <= inner[2] and inner[3] <= outer[3] + margin)


def overlap(a: Box, b: Box) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def pieces(kind: str) -> list[dict]:
    return [f for f in KIT["furniture"] if f["kind"] == kind]


def area(d) -> Box:
    """A Dressing's box as x0, x1, y0, y1."""
    return d.box[0], d.box[3], d.box[1], d.box[4]


def stands_on(d, f: dict) -> bool:
    return inside(area(d), footprint(f))


def test_the_kit_is_a_valid_rectilinear_residential_place():
    """Contract (the fixture's own: part b's tests and the drone's job read the same kind of place): `check_map` finds
    nothing in the kit (every piece inside its region and clear of other blockers, surfaces on their pieces, a known
    catalog), there is one nest and two drop zones, each drop zone's 3 m square lying inside a driveway's parking
    surface 0.005 m under its own top, none of them within 3.65 m of a house or block, no wall, and every region a
    rectangle. Regression caught: a kit that is not a place (so the layout tests would pass on a map the job
    refuses)."""
    spec = kit()
    assert check_map(parse_map(spec)) == [] and spec["walls"] == [] and spec["catalog"] == "residential"
    assert Counter(f["kind"] for f in spec["furniture"] if f["kind"] in ("nest", "drop_zone")) == {"nest": 1,
                                                                                                    "drop_zone": 2}
    drives = [(footprint(f), s) for f in pieces("driveway") for s in f["surfaces"] if s["kind"] == "parking"]
    for zone in pieces("drop_zone"):
        assert any(inside(footprint(zone), drive) and s["z"] + 0.005 == pytest.approx(zone["size"][2])
                   for drive, s in drives), zone["name"]
    assert tall_near(spec, []) == []
    assert all(len(r["polygon"]) == 4 for r in spec["regions"])


def test_the_registered_classes_are_the_five_the_layout_draws_and_none_looks_like_a_tracked_kind():
    """Contract (M4-suburb §4: street trees and lamps on verges, shrubs and trees on garden beds, a playground on a
    green, nothing else; a wheelie bin would read as a dumpster): the residential phase 1 classes are tree_stem,
    tree_canopy, lamp_post, shrub and playground, in the order the layout emits them, none declares a resemblance,
    none is named like the car, van, pickup, skip or dumpster or a synonym of one, and every one the layout emits is
    registered, in phase 1, with every registered class emitted (`layout_problems`). Regression caught: a class added
    to the layout and not to the registry (an unreviewed look-alike), a wheelie bin or a mailbox that reads as a
    dumpster, or a class no picture shows."""
    classes = dressing_of("residential", 1)
    assert [c.name for c in classes] == CLASSES == [c[0] for c in LAYOUT.CLASSES]
    assert dressing_of("residential", 0) == [] and dressing_of("residential", 2) == []
    tracked = set(residential_catalog().kinds)
    assert tracked == {"car", "van", "pickup", "skip", "dumpster"}
    for c in classes:
        assert c.resembles == () and c.catalogs == ("residential",) and not class_problems(c), c.name
        assert names_a_kind(c.name, tracked) == [], c.name
    assert layout_problems("residential", [(d.cls, d.phase) for d in _layout()]) == []
    assert [d.cls for d in _layout()] == sorted((d.cls for d in _layout()), key=CLASSES.index)  # the registry's order


def test_the_kit_map_emits_every_residential_class_and_passes_the_guard():
    """Contract (render-only dressing cannot touch the timeline): on the kit the layout
    emits all five classes, every box lies in an allowed place (over a blocking verge or garden bed, or in the
    structure band from 5 m up, never over open ground a drone or a person uses) and meets no forbidden volume
    (`DressingGuard` is clean), two runs are equal, and it depends on the map alone, not the order of its keys.
    Regression caught: a stem on open lawn, a canopy under 5 m over the street, a shrub on a drive's parking surface,
    or a layout drawn from a clock."""
    guard = DressingGuard(kit(), residential_catalog())
    assert Counter(d.cls for d in _layout()).keys() == set(CLASSES)
    assert [(d.cls, d.pos, guard.problems(d.box, d.cls)) for d in _layout() if guard.problems(d.box, d.cls)] == []
    assert list(_layout()) == LAYOUT.layout(kit()) == LAYOUT.layout(dict(reversed(list(kit().items()))))


def test_streets_trees_and_lamps_stand_along_both_verges_with_each_stem_under_its_canopy():
    """Contract (M4-suburb §4): each of the kit's two verges (100 m, 1.98 m across, 0.15 m high) takes between 4 and 8
    street trees (a tree every 13 to 19 m from 3 m in, less the few that would stand within 4 m of a drop zone, a house
    or another canopy) and 4 to 7 lamp posts between them; a stem is 0.5 m square on the verge's top (z 0.15) up to
    5.0 m, its canopy 3.4 to 5.0 m wide and 3.2 to 5.0 m high standing from 5.0 m over it, topped under the walls'
    height (12 m plus the guard's 0.02); a lamp post is 5 m high, 0.9 m across the verge, and its arm faces the street
    (yaw 90 degrees from the south verge, which has the street to its north, and -90 from the north one); a canopy
    meets no house, no block and no other canopy. Regression caught: a street without trees (or one so full of crowns
    that the pictures lose the cars), a lamp whose arm hangs over a garden, or a crown through a roof."""
    stems, canopies = of("tree_stem"), of("tree_canopy")
    assert len(stems) == len(canopies)
    verges = {f["id"]: f for f in pieces("verge")}
    for vid, yaw in (("F03", 90.0), ("F04", -90.0)):
        assert 4 <= sum(stands_on(d, verges[vid]) for d in stems) <= 8, vid
        lamps = [d for d in of("lamp_post") if stands_on(d, verges[vid])]
        assert 4 <= len(lamps) <= 7 and {d.yaw_deg for d in lamps} == {yaw}, vid
        assert all(d.size == (0.9, 0.3, 5.0) and d.pos[2] == Z_VERGE for d in lamps)
    for stem, canopy in zip(stems, canopies, strict=True):
        assert stem.size[:2] == (0.5, 0.5) and stem.pos[2] + stem.size[2] == pytest.approx(STRUCTURE_M)
        assert stem.pos[2] in (Z_VERGE, Z_BED) and canopy.pos == (stem.pos[0], stem.pos[1], STRUCTURE_M)
        assert 3.4 <= canopy.size[0] <= 5.0 and 3.4 <= canopy.size[1] <= 5.0 and 3.2 <= canopy.size[2] <= 5.0
        assert canopy.pos[2] + canopy.size[2] <= WALLS_M + 0.02
    buildings = [footprint(f) for f in pieces("house") + pieces("block")]
    for i, c in enumerate(canopies):
        assert not any(overlap(area(c), b) for b in buildings)
        assert not any(overlap(area(c), area(o)) for o in canopies[i + 1:])


def test_shrubs_stand_in_garden_beds_and_the_playground_only_on_the_bed_in_the_green():
    """Contract (M4-suburb §4: shrubs and trees on garden beds, a playground on a green): every shrub (0.7 to 1.3 m
    high) lies inside one of the kit's garden beds (0.3 m high) and none in the green's playground bed, which holds
    6 pieces of play equipment (swings, a slide, a climbing frame, two rows in a 14 m by 9 m bed) all inside it and no
    taller than 2.4 m; each garden bed carries at most one tree. Regression caught: shrubs on a lawn or a drive (a
    thing the planner counts as free floor), play equipment outside its bed or in the gardens, or a tree bed with two
    trees."""
    beds = {f["id"]: f for f in pieces("garden_bed")}
    play = beds["F32"]
    shrubs, equipment = of("shrub"), of("playground")
    assert len(shrubs) >= 6 and all(0.7 <= d.size[2] <= 1.3 and d.pos[2] == Z_BED for d in shrubs)
    for d in shrubs:
        assert sum(stands_on(d, b) for b in beds.values()) == 1 and not stands_on(d, play)
    assert len(equipment) == 6 and all(stands_on(d, play) and d.size[2] <= 2.4 for d in equipment)
    assert {d.recipe for d in equipment} == {"proc:swing_set", "proc:slide", "proc:climbing_frame"}
    for bed in beds.values():
        assert sum(stands_on(d, bed) for d in of("tree_stem")) <= 1


def test_nothing_tall_stands_by_a_drop_zone():
    """Contract (M3-Y §1, M4-suburb §4: at lowering, everything within 3.65 m of the drone's centre must stand below its
    altitude less 3 m, 4 m; nothing within that of the nest either): on the kit, with two drop zones and the nest, no
    house or block and no dressing taller than 3.9 m (stem, canopy, lamp post) is within 3.65 m of any of their
    centres. The rule bites: a house moved to 3 m from a drop zone, and a canopy box 0.3 m from the nest, are both
    reported. Regression caught: a lowering point the drone's own rule forbids, or a street tree growing over a drop
    zone."""
    spec = kit()
    assert tall_near(spec, _layout()) == []
    near = copy.deepcopy(spec)
    next(f for f in near["furniture"] if f["id"] == "F10")["center"][1] += 1.0  # its front 3.0 m from (11, 20.5)
    canopy = render_module("dressing_spec").Dressing("tree_canopy", "proc:tree_canopy:0", (87.0, 48.0, 5.0), 0.0,
                                                     (3.4, 3.4, 3.2))
    assert tall_near(near, [canopy]) == ["1 Elm Street house is within 3.65 m of 1 Elm Street",
                                         "tree_canopy at (87.0, 48.0, 5.0) is within 3.65 m of nest"]


def test_the_layout_refuses_another_catalog_and_walls_too_low_for_a_canopy_over_its_stem():
    """Contract (M4-suburb §4): `layout` raises ValueError for a map of another catalog and for one whose walls stand
    under 8.2 m (5 m for the stem and the lowest canopy of 3.2 m), where a canopy would have no structure band to stand
    in. Regression caught: the residential dressing laid over the downtown, or canopies the guard would refuse on every
    map with low walls."""
    other = kit()
    other["catalog"] = "downtown"
    with pytest.raises(ValueError, match="not a residential map"):
        LAYOUT.layout(other)
    low = kit()
    low["wall_height"] = 8.0
    with pytest.raises(ValueError, match="no room for a canopy"):
        LAYOUT.layout(low)
    assert len(LAYOUT.layout({**kit(), "wall_height": 8.2})) > 0


# -- the far scenery ---------------------------------------------------------------------------------------------------
@cache
def _plan():
    return SCENERY.plan(parse_map(kit()))


def test_the_far_scenery_continues_the_street_past_its_ends_and_lies_outside_the_bounds():
    """Contract (M4-suburb §4: the street continued past the map's open ends, rows of houses and trees beyond its long
    sides, low-detail, out to about 250 m, nothing inside the map's bounds): on the kit, the street's two ends (x 100
    and x 0; the plots' open back edges are no street) continue as carriageways 6 m wide (y 27 to 33, what the
    sidewalks 25.5-27 and 33-34.5 and the verges leave) 250 m long, with four sidewalks of 1.5 m beside them; no
    building, tree, road or sidewalk of the scenery enters the bounds (0 to 100 by 0 to 60) or reaches 270 m past them;
    no two buildings overlap and none overlaps a road or sidewalk; there are at least 150 houses, 5 blocks (a row of
    their own, 150 m out) and 100 trees; the same map gives the same plan and an indoor one none. By hand: the east
    arm's frame is at (100, 23.5) looking +x, the west's at (0, 36.5) looking -x. Regression caught: a street that ends
    at the map's edge in a field, asphalt over the plots' back gardens, a house standing in a garden of the map, or a
    scenery that depends on the episode."""
    plan = _plan()
    assert sorted((a.o, a.n) for a in plan.arms) == [((0.0, 36.5), (-1.0, -0.0)), ((100.0, 23.5), (1.0, -0.0))]
    roads = sorted(AABB(r) for r in plan.roads)
    assert roads == [pytest.approx((-250.0, 0.0, 27.0, 33.0)), pytest.approx((100.0, 350.0, 27.0, 33.0))]
    assert sorted(round(AABB(w.rect)[3] - AABB(w.rect)[2], 3) for w in plan.walks) == [1.5] * 4
    x0, y0, x1, y1 = KIT["bounds"]
    ground = [AABB(r) for r in plan.roads] + [AABB(w.rect) for w in plan.walks]
    buildings = [AABB(s.rect) for s in plan.sites]
    trees = [AABB(Rect(t.x, t.y, 4.0, 4.0, 0.0)) for t in plan.trees]
    for a in [*ground, *buildings, *trees]:
        assert a[0] >= x1 - 1e-6 or a[1] <= x0 + 1e-6 or a[2] >= y1 - 1e-6 or a[3] <= y0 + 1e-6, a
        assert x0 - 270.0 <= a[0] and a[1] <= x1 + 270.0 and y0 - 270.0 <= a[2] and a[3] <= y1 + 270.0, a
    for i, a in enumerate(buildings):
        assert not any(overlap(a, b) for b in [*buildings[i + 1:], *ground]), plan.sites[i]
    blocks = [s for s in plan.sites if s.kind == "block"]
    assert len(blocks) >= 5 and len(plan.sites) - len(blocks) >= 150 and len(plan.trees) >= 100
    for s in blocks:  # beyond a long side, 150 m out or more
        a = AABB(s.rect)
        assert max(y0 - a[3], a[2] - y1, x0 - a[1], a[0] - x1) >= 149.0, s
    assert SCENERY.plan(parse_map(kit())) == plan
    assert SCENERY.plan(parse_map({**kit(), "outdoor": None})).sites == []
