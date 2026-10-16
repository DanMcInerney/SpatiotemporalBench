"""Clutter, the world content no question names: how many entities a world holds, what they are, where they may and may
not lie, that the floor stays connected for the robots planned on it, that the things left on the floor are drawn from
a stream of their own, and that a cluttered world is a valid timeline. The worlds are `realism_kit`'s (the apartment's
seeds, `build_world` then `place`, cached per process); every number is written out here from the plan and the map."""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict

import numpy as np
import pytest

from realism_kit import (
    apartment,
    clutter_ids,
    clutter_rng,
    clutter_timeline,
    cluttered,
    docked_script,
    footprint,
    keepout_hits,
    on_floor,
)
from stbench import asset_manifest as A
from stbench import scenery
from stbench.catalog import CATALOGS, place_of
from stbench.clutter import place
from stbench.clutter_reach import FloorGuard
from stbench.generate import build_world
from stbench.geometry import (
    Rect,
    corners,
    point_in_rect,
    rect_distance,
    rect_inside_rect,
    rects_overlap,
    to_local,
    under_obstacles,
)
from stbench.mapspec import load_map, surface_world_rect
from stbench.placement import free_of
from stbench.robots import ROBOTS
from stbench.route import robot_free_space
from stbench.validate import validate
from test_mapspec import palette_keys
from validate_kit import generic_rules_only  # noqa: F401  (a fixture)
from world_kit import world_timeline

# One xdist group (`--dist loadgroup`): the 20 worlds are cached per process, so one worker builds them once.
pytestmark = pytest.mark.xdist_group(name=__name__)
SEEDS = range(1, 21)
HOME = CATALOGS["home"]


def floor_ids(tl) -> list[str]:
    return [i for i in clutter_ids(tl) if tl.placement_of(i, 0.0).on is None]


def test_clutter_counts_land_in_the_target():
    """Contract (plan-redesign §4.10 item 1; recommendations §2.5: the apartment holds 160-250 objects, 3-4.5 a m² of
    its 55 m²): on seeds 1-20 every world holds 160-250 entities in all, 20-45 pieces of clutter on the floor
    (about 0.5-1.2 a m² of its ~39 m² of free floor) and 65-140 on surfaces, no kind more often than its count
    allows, and the worlds differ in how much they hold. Regression caught: clutter far from a real home's density
    (the plan's apartment held 7 objects in a median frame), a count that ignores what `build_world` made (68-78
    entities), or one world copied for every seed."""
    totals, floors = set(), set()
    for seed in SEEDS:
        tl = clutter_timeline(seed)
        ids = clutter_ids(tl)
        floor = floor_ids(tl)
        assert 160 <= len(tl.entities) <= 250, (seed, len(tl.entities))
        assert 20 <= len(floor) <= 45, (seed, len(floor))
        assert 65 <= len(ids) - len(floor) <= 140, (seed, len(ids) - len(floor))
        for kind, n in Counter(tl.entity(i).kind for i in ids).items():
            assert HOME.clutter_kind(kind).count[0] <= n <= HOME.clutter_kind(kind).count[1], (seed, kind, n)
        totals.add(len(tl.entities))
        floors.add(len(floor))
    assert len(totals) >= 10 and len(floors) >= 10


def test_a_clutter_entity_is_its_models_size_and_colour_and_nothing_a_question_can_name():
    """Contract (`clutter.place`): each clutter entity has the `clutter` mobility, an asset that is a clutter model of
    its own kind, that model's measured box for its size, and a colour that is a palette key: the model's own where it
    is untinted, else one of its kind's. The mobility is never describable and it collides as a carried thing.
    Regression caught: an entity sized by a catalog box that its model does not have (the renderer would squash it), a
    colour that names no paint, or clutter that a question could ask about."""
    from stbench import timeline as T

    m = A.packaged()
    palette = palette_keys()
    for seed in (1, 2, 3):
        tl = clutter_timeline(seed)
        for eid in clutter_ids(tl):
            e = tl.entity(eid)
            model = m.asset(e.asset)
            assert model.role == "clutter" and model.kind == e.kind, e
            assert tuple(e.size) == tuple(model.native_bbox_m), e
            colour = e.attributes["color"]
            assert colour in palette and colour in HOME.clutter_kind(e.kind).colors, e
            assert model.colour in (None, colour), e
            assert e.id in cluttered(seed).clutter
    rule = T.MOBILITY["clutter"]
    assert not rule.describable and rule.collides_as == "carried" and "clutter" not in T.DESCRIBABLE


ASKABLE_SYNONYM_FIXES = ("cushion", "pillow", "socks", "waste_bin")


def test_clutter_is_never_askable():
    """Contract (VISION rule 10; recommendations §2.5 item 2): no clutter kind is called what an askable kind is called
    (the household's 14, the apartment's chairs, stool and laundry basket, the movers, the door) or any synonym of
    one, unless it declares the resemblance, which it may only do in colours no tracked thing of that kind has; so
    there is no cushion, pillow, sock or waste bin, the side table is white or grey, the heater, fan and purifier grey
    or black and the dumbbells black or grey (the plan's list: the box is brown, white, blue, red or green, the toy
    red, blue, yellow, green, orange or pink). The guard has teeth: a kind called a cushion, a sock, a waste bin or a
    shoe box is refused, as are a resemblance in a colour the tracked kind has, a kind that is the floor and a
    counter, and two of a name. Regression caught: a clutter object read as "the black shoe", or a list fix undone."""
    m = apartment()
    askable = {*HOME.kinds, *HOME.mover_kinds, "door", *(f.kind for f in m.furniture if f.mobility == "light")}
    assert {"chair", "stool", "laundry_basket", "robot_vacuum"} <= askable
    assert scenery.clutter_problems(HOME, askable) == []
    names = {k.name for k in HOME.clutter}
    assert not names & {*ASKABLE_SYNONYM_FIXES, "cushions", "throw_pillow", "sock", "waste_basket", "waste_bin"}
    assert dict(HOME.clutter_kind("shelved_book").resembles) == dict(scenery.DRESSING["shelved_book"].resembles)
    assert {c for k in ("side_table",) for c in HOME.clutter_kind(k).colors} == {"white", "grey"}
    for kind in ("space_heater", "floor_fan", "air_purifier", "dumbbells"):
        assert set(HOME.clutter_kind(kind).colors) <= {"grey", "black"}, kind
    base = dict(size=(0.1, 0.1, 0.1), colors=("grey",), places=("floor",), count=(0, 1), models=("proc_floor_fan",))

    def problems(**extra):
        kind = dataclasses.replace(HOME.clutter_kind("floor_fan"), **{k: v for k, v in {**base, **extra}.items()})
        return scenery.clutter_problems(dataclasses.replace(HOME, clutter=(kind,)), askable)

    for name in ("floor_cushion", "throw_pillow", "socks", "waste_bin", "shoe_box", "trash_can", "side_chair"):
        assert any("is called what" in p for p in problems(name=name)), name
    assert problems(name="shelf_book", resembles=(("book", ("white", "beige")),)) == []  # declared, untracked colours
    assert any("tracked" in p for p in problems(name="shelf_book", resembles=(("book", ("red",)),)))
    assert any("floor and on a surface" in p for p in problems(name="x", places=("floor", "desk")))
    assert any("two clutter kinds" in p for p in scenery.clutter_problems(
        dataclasses.replace(HOME, clutter=(HOME.clutter[0], HOME.clutter[0])), askable))


def test_clutter_never_lies_where_a_door_swings_a_doorway_the_apron_or_a_door_band():
    """Contract (plan §4.10 item 1: never in a doorway, a door's swing, the dock's apron or within 0.5 m in front of a
    door): on seeds 1-20 no clutter entity and no tracked thing left on the floor has any part of its footprint in a
    doorway, in the band 0.5 m out from both faces of a doorway's wall across its width, in the quarter disc a door leaf
    sweeps, or in the dock's apron (1.2 m in front, 0.5 m each side): the geometry is `realism_kit`'s, from the map, as
    `test_mapspec.apron_problems` writes the apron. Regression caught: a vacuum that cannot leave its dock or enter a
    room because something lies across it."""
    m = apartment()
    for seed in SEEDS:
        tl = clutter_timeline(seed)
        floor = [*floor_ids(tl), *(c for c in cluttered(seed).carried if on_floor(tl, c))]
        for eid in floor:
            assert keepout_hits(m, footprint(tl, eid)) == [], (seed, eid, tl.entity(eid).kind)


def test_clutter_leaves_every_floor_a_planned_robot_could_reach_reachable():
    """Contract (plan §4.10 item 1: every room stays reachable; `route` owns where a robot can go): on seeds 1-20, for
    each robot planned on the home (the vacuum and the helper), every cell of `route.robot_free_space` that the robot
    could reach from the middle of the hall level with the vacuum's dock before the clutter and the left things were
    placed, it can still reach after, and that start is still free (the dock's apron keeps it so). The plan's wording
    is one component a room; the map itself has pockets a robot's margin shuts off (the vacuum's: 12 cells at 0.1 m
    in the kitchen, 10 in the bathroom, 58-74 in the bedroom), so the contract is that clutter cuts none off.
    Regression caught: a script a robot cannot finish, a room it cannot enter, or clutter placed against another idea
    of where the robot can go than the planner's (the guard's own 0.1 m grid and margin closed the bathroom to
    `job_vacuum` on seed 104)."""
    m = apartment()
    (dock,) = [f for f in m.furniture if f.kind == "dock"]
    front = (5.4, dock.center[1])  # the middle of the hall (x 4.65-6.15) level with the dock, in its apron
    for seed in SEEDS:
        for name in ("vacuum", "helper"):
            robot = ROBOTS[name]
            before = robot_free_space(m, build_world(m, seed), robot)
            after = robot_free_space(m, cluttered(seed), robot)
            assert after.is_free(front), (seed, name)
            cut = before.reachable_from(front) & after.free & ~after.reachable_from(front)
            assert not cut.any(), (seed, name, int(cut.sum()))


def test_floor_drops_vary():
    """Contract (plan §4.10 item 1: things left on the floor are what real floors look like, and `blind`'s dense places
    would learn a mat and two rugs): on seeds 1-20 each floor kind (shoes, bag, toy, box) is left on open floor in
    25-75% of its draws (the draw is a half) and at 3 or more spots more than 0.5 m apart, and each lies on the floor
    with no support, off every keep-out and out of every blocking footprint. Regression caught: floor things that stay
    on their receptacles, or only ever land on the mat and the rugs."""
    m = apartment()
    blocking = [f.footprint() for f in m.furniture if f.mobility == "fixed" and f.blocks]
    tracked, left, where = Counter(), Counter(), defaultdict(list)
    for seed in SEEDS:
        w, tl = cluttered(seed), clutter_timeline(seed)
        for eid, c in w.carried.items():
            if c.spec.kind not in dict(HOME.floor_kinds):
                continue
            tracked[c.spec.kind] += 1
            if on_floor(tl, eid):
                left[c.spec.kind] += 1
                fp = footprint(tl, eid)
                where[c.spec.kind].append((fp.cx, fp.cy))
                assert keepout_hits(m, fp) == [] and not any(rects_overlap(fp, b) for b in blocking), (seed, eid)
    assert set(tracked) == {"shoes", "bag", "toy", "box"}
    for kind, n in tracked.items():
        assert 0.25 <= left[kind] / n <= 0.75, (kind, left[kind], n)
        apart: list[tuple[float, float]] = []
        for p in where[kind]:
            if all(math.dist(p, q) > 0.5 for q in apart):
                apart.append(p)
        assert len(apart) >= 3, (kind, apart)


def _surface_of(tl, m, eid):
    """The map piece and the surface a clutter entity rests on, found by the support's position and the entity's height.
    """
    p = tl.placement_of(eid, 0.0)
    support = tl.entity(p.on)
    pos = tl.placement_of(p.on, 0.0).pos
    f = next(f for f in m.furniture if f.kind == support.kind and math.dist(f.center, pos[:2]) < 1e-3)
    for s in f.surfaces:
        if abs(s.z - p.pos[2]) < 1e-6 and point_in_rect(surface_world_rect(f.center, f.yaw, s), p.pos[0], p.pos[1],
                                                         1e-6):
            return f, surface_world_rect(f.center, f.yaw, s)
    return f, f.footprint()  # a piece the map gives no surface there: the top of the stove, the face of the fridge


@pytest.mark.usefixtures("generic_rules_only")
def test_a_cluttered_world_validates_and_each_piece_lies_where_its_rule_puts_it():
    """Contract (plan §4.10 item 1; `validate`): on seeds 1-5 the cluttered world under a docked vacuum passes
    `validate`'s generic rules (no empty interval, no clutter inside another piece or a mug; `vacuum/1`'s own checks
    are off, for a one-keyframe docked script is not a whole task of the job, and `test_job_vacuum` tests them); every
    surface piece lies wholly inside the back strip of its surface (four fifths of a surface 0.5 m deep or more, else
    the rear 0.25 m or half its depth, or as deep as the kind's own depth asks, at most four fifths, in its piece's
    frame; the stove's top is a surface of the stove's footprint), a sheet on the fridge or a coat on the wardrobe
    hangs on its front face between 0.8 m and the top, inside the face's width; every floor piece is at least 0.12 m
    from every wall and blocking footprint (a mat that lies beside a piece, 0.03 m: it is lower than any dressing), or
    lies under a piece with an open underside, wholly in its footprint, off its leg squares and lower than its open
    height. Regression caught: clutter inside furniture, an empty interval (the ledger rule), or floor things that wall
    dressing (up to 0.12 m proud) could meet."""
    m = apartment()
    walls = m.wall_pieces()
    blocking = [f.footprint() for f in m.furniture if f.mobility == "fixed" and f.blocks]
    passable = [f for f in m.furniture if f.mobility == "fixed" and f.blocks and f.under_m > 0]
    for seed in range(1, 6):
        assert validate(docked_script(seed), m) == [], seed
        tl = clutter_timeline(seed)
        for eid in clutter_ids(tl):
            fp, e, p = footprint(tl, eid), tl.entity(eid), tl.placement_of(eid, 0.0)
            if p.on is not None:
                f, surface = _surface_of(tl, m, eid)
                if HOME.clutter_kind(e.kind).on_front:  # on the front face (+x of the piece), its back on the face
                    assert f.kind in ("fridge", "wardrobe"), (seed, eid)
                    assert 0.8 - 1e-6 <= p.pos[2] <= f.size[2] - e.size[2] + 1e-6, (seed, eid)
                    for x, y in corners(fp):
                        lx, ly = to_local(surface, x, y)
                        assert surface.sx / 2 - e.size[0] <= lx <= surface.sx / 2 + e.size[0], (seed, eid)
                        assert abs(ly) <= surface.sy / 2 + 1e-6, (seed, eid)
                    continue
                depth = 0.8 * surface.sx if surface.sx >= 0.5 else min(0.25, surface.sx / 2)  # a deep one: four fifths
                depth = max(depth, min(0.8 * surface.sx, e.size[0] + 0.02))  # and as deep as the kind needs
                for x, y in corners(fp):
                    lx, ly = to_local(surface, x, y)
                    assert -surface.sx / 2 - 1e-6 <= lx <= -surface.sx / 2 + depth + 1e-6, (seed, eid, e.kind)
                    assert abs(ly) <= surface.sy / 2 + 1e-6, (seed, eid, e.kind)
                continue
            under = [f for f in passable if rect_inside_rect(fp, f.footprint())]
            if under:
                (f,) = under
                assert e.size[2] < f.under_m, (seed, eid, e.kind)
                assert not any(rects_overlap(fp, leg) for leg in under_obstacles(f.footprint(), f.under_m))
            else:
                apart = 0.03 if any(":" in place_of(t)[0] for t in HOME.clutter_kind(e.kind).places) else 0.12
                assert all(rect_distance(fp, r) >= apart - 1e-6 for r in [*walls, *blocking]), (seed, eid, e.kind)


def test_clutter_stands_on_the_stove_hangs_on_a_door_and_lies_beside_the_tub_toilet_and_rack():
    """Contract (`catalog_clutter` places `on:`, `front:` and `floor:<piece>`; the pre-screen's fix 9: the kitchen's
    pots, the fridge's papers, the wardrobe's coats, the bathroom's mats and the hall's boot tray): on seeds 1-20, the
    map's numbers written out here. The stove F05 is 0.6 m square at (3.85, 0.35), 0.9 m high: a pot or a pan stands on
    it at z 0.9, wholly inside x 3.55-4.15, y 0.05-0.65. The fridge F01 at (0.4, 0.4), 0.7 x 0.7 x 1.8 m, faces +x, its
    front at x 0.75: a sheet of paper hangs with its centre 1.5 mm out (x 0.7515, half its 4 mm depth less half a
    millimetre) between y 0.05 and 0.75 and from 0.8 m to the top, turned to face +x. The wardrobe F34 at (7.8, 2.75),
    0.6 x 1.2 x 2.0, yaw 90 degrees, faces +y, its front at y 3.05: a coat is seated 8.5 mm out (3.0585: a thick thing
    keeps its centre on the piece, `validate`'s 1 cm), x 7.2-8.4. A mat is within 0.35 m of the piece it lies beside
    (the bath mat the tub, which spans x 8.2-8.95, y 0.6-2.3; the contour mat the toilet, x 7.7-8.1, y 0.05-0.65; the
    boot tray the shoe rack, x 4.65-4.95, y 0.15-0.85). Each of them is placed in some world, and a fridge always has
    four to nine sheets. Regression caught: a pot floating in the kitchen, a paper in the middle of the room or inside
    the fridge, a coat that `validate` refuses, or a mat on the far side of the bathroom."""
    seen = Counter()
    for seed in SEEDS:
        tl = clutter_timeline(seed)
        sheets = 0
        for eid in clutter_ids(tl):
            e, p = tl.entity(eid), tl.placement_of(eid, 0.0)
            fp = footprint(tl, eid)
            seen[e.kind] += 1
            if e.kind in ("stock_pot", "frying_pan"):
                assert tl.entity(p.on).kind == "stove" and p.pos[2] == pytest.approx(0.9), (seed, eid)
                assert all(3.55 - 1e-6 <= x <= 4.15 + 1e-6 and 0.05 - 1e-6 <= y <= 0.65 + 1e-6 for x, y in corners(fp))
            elif e.kind == "fridge_paper":
                sheets += 1
                assert tl.entity(p.on).kind == "fridge" and p.pos[0] == pytest.approx(0.7515, abs=1e-3), (seed, eid)
                assert p.pos[1] - e.size[1] / 2 >= 0.05 and p.pos[1] + e.size[1] / 2 <= 0.75, (seed, eid)
                assert 0.8 - 1e-6 <= p.pos[2] <= 1.8 - e.size[2] + 1e-6 and p.yaw == pytest.approx(0.0), (seed, eid)
            elif e.kind == "hanging_coat":
                assert tl.entity(p.on).kind == "wardrobe" and p.pos[1] == pytest.approx(3.0585, abs=1e-3), (seed, eid)
                assert 7.2 <= p.pos[0] <= 8.4 and p.yaw == pytest.approx(math.pi / 2), (seed, eid)
            elif e.kind in ("bath_mat", "pedestal_mat", "boot_tray"):
                piece = {"bath_mat": Rect(8.575, 1.45, 0.75, 1.7, 0.0), "pedestal_mat": Rect(7.9, 0.35, 0.4, 0.6, 0.0),
                         "boot_tray": Rect(4.8, 0.5, 0.3, 0.7, 0.0)}[e.kind]
                assert p.on is None and rect_distance(fp, piece) <= 0.35 + 1e-6, (seed, eid)
        assert 4 <= sheets <= 9, (seed, sheets)
    assert {"stock_pot", "frying_pan", "hanging_coat", "bath_mat", "pedestal_mat", "boot_tray"} <= set(seen), seen


def test_a_bathroom_floor_holds_a_mat_a_bin_a_brush_and_laundry_and_no_cable():
    """Contract (the look round's fix 3, `catalog_clutter`'s bathroom kinds): on seeds 1-20 the bathroom (x 6.25-8.95,
    y 0.05-2.35 in the map) holds on its floor, in every world, a bath bin, a toilet brush, the tracked laundry basket
    and a mat (the tub's or the toilet's), and no extension cord, power strip or cable bundle (cables lie on the other
    rooms' floors). Regression caught: the mats laid after the rest, which lose the little open floor a bathroom has (a
    bath mat in three worlds of ten), a bin that is drawn out of the bathroom, and a cable on a wet floor."""
    for seed in SEEDS:
        tl = clutter_timeline(seed)
        kinds = [e.kind for e in tl.entities if e.mobility in ("clutter", "light")
                 if (p := tl.placement_of(e.id, 0.0)).on is None and 6.25 <= p.pos[0] <= 8.95
                 and 0.05 <= p.pos[1] <= 2.35]
        assert {"bath_bin", "toilet_brush", "laundry_basket"} <= set(kinds), (seed, kinds)
        assert {"bath_mat", "pedestal_mat"} & set(kinds), (seed, kinds)
        assert not {"extension_cord", "power_strip", "cable_bundle"} & set(kinds), (seed, kinds)


def test_clutter_streams_are_independent(monkeypatch):
    """Contract (`clutter.place` takes two streams from its rng): the same rng gives the same world twice; and changing
    one clutter kind's count, which redraws what the first stream draws, leaves the tracked things left on the floor
    where they were, because the second stream draws them and what they avoid is the same. (The edit has teeth: the
    cable bundles it removes are gone and the rest of the world differs.) Regression caught: floor drops redrawn by an
    unrelated edit, which would move every script's dropped shoes whenever the clutter table changes."""
    m, seed = apartment(), 3

    def world():
        w = build_world(m, seed)
        place(w, m, clutter_rng(seed))
        return w, world_timeline(m, w, seed)

    (w1, t1), (_, t2) = world(), world()
    assert t1.entities == t2.entities and t1.placements == t2.placements

    def left(w, tl):
        return {eid: tl.placement_of(eid, 0.0) for eid, c in w.carried.items() if c.spec.kind in dict(HOME.floor_kinds)}

    fewer = tuple(dataclasses.replace(k, count=(0, 0)) if k.name == "cable_bundle" else k for k in HOME.clutter)
    monkeypatch.setitem(CATALOGS, "home", dataclasses.replace(HOME, clutter=fewer))
    w3, t3 = world()
    assert left(w1, t1) == left(w3, t3) and any(p.on is None for p in left(w1, t1).values())
    assert not any(t3.entity(i).kind == "cable_bundle" for i in clutter_ids(t3))
    assert any(t1.entity(i).kind == "cable_bundle" for i in clutter_ids(t1))
    assert t1.placements != t3.placements


def test_a_thing_that_would_cut_a_robots_floor_in_two_is_refused():
    """Contract (`clutter_reach.FloorGuard`, on `route.robot_free_space`, the floor each planned robot's planner
    walks): a floor thing taller than a robot climbs that fills the doorway of the bathroom, whose only door (D3, 0.8 m
    wide) leaves 0.8 - 2 x 0.208 = 0.38 m of floor for the vacuum (a 0.34 m disc, 0.02 m clearance, and the grid's
    half-diagonal 0.018 m), is refused: for the vacuum (climbs 0.02 m) a 0.2 m box 0.1 m high, but not one of 0.015 m,
    which it drives over; for the helper (a 0.34 x 0.33 m box planned as its 0.47 m disc, climbs 0.01 m, clearance
    0.1 m: 0.35 m from every wall) both, and not a 5 mm one. The same box in the middle of the hall, 1.5 m wide, leaves
    the vacuum 0.23 m of floor each side and the helper none: only the helper's space refuses it. A thing in the
    living room's open floor is allowed for both. The guard covers exactly the robots planned on the catalog
    (`dressing_robots.PLANNED_ROBOTS`: the vacuum and the helper at home, the AMR alone at the warehouse, none in the
    downtown, whose drone flies). Regression caught: a bathroom the clutter closes, a guard with its own idea of where a
    robot can go, or one that refuses what a robot drives over or lets a gap close for the larger robot."""
    w = build_world(apartment(), 1)
    guard = FloorGuard(w)
    spaces = {s.robot.id: s for s in guard.spaces}
    assert list(spaces) == ["vacuum", "helper"]
    door = Rect(6.2, 1.3, 0.2, 0.2, 0.0)  # D3 stands in the wall x = 6.2 at y = 1.3
    assert not spaces["vacuum"].allows(door, 0.1) and spaces["vacuum"].allows(door, 0.015)
    assert not spaces["helper"].allows(door, 0.1) and not spaces["helper"].allows(door, 0.015)
    assert spaces["helper"].allows(door, 0.005) and not guard.allows(door, 0.1)
    hall = Rect(5.4, 3.0, 0.2, 0.2, 0.0)  # the hall runs x 4.65-6.15
    assert spaces["vacuum"].allows(hall, 0.1) and not spaces["helper"].allows(hall, 0.1)
    assert guard.allows(Rect(3.9, 5.4, 0.2, 0.2, 0.0), 0.1)
    wh = load_map("warehouse_a")
    assert [s.robot.id for s in FloorGuard(build_world(wh, 1)).spaces] == ["amr"]
    assert load_map("apartment_a").region_of((6.2, 1.3)) is None  # the door's centre is in no room
    before = int(spaces["vacuum"].fs.free.sum())
    spaces["vacuum"].block(door, 0.1)
    assert int(spaces["vacuum"].fs.free.sum()) < before


def test_the_clearance_test_is_the_plain_pairwise_overlap_test_made_faster():
    """Contract (`placement.free_of`): a box is clear of what stands at its height exactly when it is clear, by the
    margin, of each by the plain separating-axis test `geometry.rects_overlap`: the cheap centre-distance test in front
    of it only skips pairs that cannot be that close, never changes an answer (every carried item's draw, which
    `test_placement_catalog` pins, goes through it). Checked on 6,000 seeded pairs of turned rectangles at two margins,
    near and far, both kinds of answer among them, and a pair at another height is never an obstacle. Regression caught:
    a prefilter that is too keen and lets two boxes overlap by a margin, which moves every draw after it."""
    rng = np.random.default_rng(7)
    seen = Counter()
    for i in range(6000):
        a = Rect(*rng.uniform(0, 0.6, 2), *rng.uniform(0.02, 0.5, 2), rng.uniform(-math.pi, math.pi))
        b = Rect(*rng.uniform(0, 0.6, 2), *rng.uniform(0.02, 0.5, 2), rng.uniform(-math.pi, math.pi))
        margin = (0.015, 0.004)[i % 2]
        want = not rects_overlap(a, b, margin)
        assert free_of(a, 0.0, 0.1, [(b, 0.0, 0.1)], margin) is want
        seen[want] += 1
    assert min(seen.values()) > 500
    assert free_of(Rect(0, 0, 0.2, 0.2, 0.0), 0.0, 0.1, [(Rect(0, 0, 0.2, 0.2, 0.0), 0.1, 0.2)])
