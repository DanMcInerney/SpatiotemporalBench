"""The forbidden volumes and allowed places of render-only dressing (`dressing_zones.py`): where a Phase 1 box may not
stand because something tracked or the robot can be there, and the only places it may. Everything is free: the maps are
data, the generator is the one component that decides where things really are, and the expected numbers are worked out
by hand from the map files and the catalogs."""

from __future__ import annotations

import copy

import pytest

from dressing_kit import SYNTH, raw_map, render_module
from stbench import dressing_boxes as db
from stbench import dressing_zones as dz
from stbench.geometry import Rect, corners, yaw_of
from stbench.mapspec import check_map, load_map, parse_map
from world_kit import after_gap

MAPS = ("apartment_a", "warehouse_a")
FLOOR_EGO, FLOOR_HEAVY = "open floor the camera", "open floor light furniture"


def _zones(spec: dict, prefix: str) -> list[db.Box]:
    return [box for why, box in dz.forbidden(spec) if why.startswith(prefix)]


def _tiny(x: float, y: float, z0: float, z1: float, r: float = 1e-6) -> db.Box:
    return (x - r, y - r, z0, x + r, y + r, z1)


def test_the_synthetic_map_is_a_legal_map():
    """Contract: the hand-checkable map of these tests is a real map (rooms, openings, windows and
    furniture all consistent), so what the zones say about it holds for maps. Regression caught: a synthetic map that
    drifts from the schema, which would make every hand-worked number below a statement about nothing."""
    assert check_map(parse_map(SYNTH)) == []


@pytest.mark.parametrize(("map_name", "furniture", "surface", "top"), [
    # counter 0.54 x 1.14 m: the blender (0.17 x 0.17 x 0.40, a counter's clutter) is the tallest kind that fits, taller
    # than the bag (0.314): 0.9 + 0.40 + 0.05
    ("apartment_a", "F02", "S01", 1.35),
    # nightstand 0.36 x 0.38 m: the bag needs 0.304 x 0.374 plus 2 x 0.01, and 0.374 > 0.36 either way round, so the
    # plant (0.30) is the tallest tracked kind, but the desk lamp (clutter on desks and nightstands, 0.45 high) fits and
    # is taller: 0.55 + 0.45 + 0.05
    ("apartment_a", "F31", "S17", 1.05),
    # desk 0.54 x 1.14 m: the bag fits (0.314), and the desk lamp is taller: 0.75 + 0.45 + 0.05
    ("apartment_a", "F33", "S19", 1.25),
    ("apartment_a", "F13", "S11", 1.75),  # shelf 0.24 x 0.84 at 1.4 m: plant 0.30 (a box is 0.30 wide: no)
    ("apartment_a", "F14", "S12", 0.374),  # living rug at 0.01 m: the bag again
    ("warehouse_a", "F09", "S05", 1.83),  # packing table 0.74 x 1.94 at 0.9 m: a drum (0.58 wide) fits, a pallet not
    ("warehouse_a", "F11", "S07", 0.87),  # shelving 0.54 wide at 0.5 m: no drum (squared kinds do not turn): a tote
    ("warehouse_a", "F13", "S14", 3.65),  # rack level at 2.6 m: a pallet (1.0 high) fits across the 1.08 m depth
    ("warehouse_a", "F05", "S01", 1.055),  # staging lane at 0.005 m (M3-maps: a marking): a pallet, 0.005 + 1.0 + 0.05
])
def test_a_receptacle_volume_reaches_the_tallest_kind_that_fits_plus_five_centimetres(map_name, furniture, surface,
                                                                                    top):
    """Contract (`placement._try_slot`, `find_spot`): any surface takes any kind that fits it (70% of draws go to a
    typical receptacle, the rest anywhere), so a surface's volume rises from its height to the tallest kind of the
    catalog whose footprint plus 1 cm on each side fits its rect at some angle (a quarter turn for the warehouse's
    squared stock), plus 5 cm, a clutter kind counting where its places name the surface's kind (a desk lamp on a desk
    or a nightstand, not a kettle: it is a counter's, and shorter than the bag). Worked by hand per case above.
    Regression caught: a margin dropped (an item touching a sill), a kind that fits left out, the typical receptacles
    read as the only ones (a bag on a nightstand), a non-fitting kind counted (a drum on a shelf), or clutter taller
    than what the volume holds (a dressing piece in a desk lamp's way)."""
    why = f"receptacle {furniture}/{surface} "
    (box,) = [b for w, b in dz.forbidden(raw_map(map_name)) if w.startswith(why)]
    assert box[5] == pytest.approx(top)
    assert box[2] < box[5]


def test_door_swings_openings_and_window_glass_are_the_boxes_the_map_implies():
    """Contract: D1 is 0.85 m wide in wall W05 (x = 4.6, y runs up), hinged at its low end (y = 1.475) and swinging to
    the left, x < 4.6 (`MapSpec.door_leaf`), so its swing is the square [3.75, 4.6] x [1.475, 2.325] up to its 2.05 m;
    the arch O1 (1.2 m wide at x = 3.6 in W07, y = 2.8) is cut through the 0.1 m wall and 0.12 m before it on both
    sides: x 3.0..4.2, y 2.63..2.97, 2.1 m high; window N1 (1.0 m wide at x = 2.35 in W01, y = 0, sill 1.1, 1.0 high) is
    x 1.85..2.85, y -0.17..0.17, z 1.1..2.1. Regression caught: a swing on the wrong side or too short (a casing the
    leaf would hit), an opening that stops at the wall (trim across the doorway) and glass that stops at the pane (a
    curtain hung in front of it)."""
    zones = dict(dz.forbidden(raw_map("apartment_a")))
    assert zones["swing of door D1"] == pytest.approx((3.75, 1.475, 0.0, 4.6, 2.325, 2.05))
    assert zones["arch opening O1"] == pytest.approx((3.0, 2.63, 0.0, 4.2, 2.97, 2.1))
    assert zones["glass of window N1"] == pytest.approx((1.85, -0.17, 1.1, 2.85, 0.17, 2.1))


def test_the_open_floor_is_the_floor_a_hand_away_from_walls_and_blocking_furniture():
    """Contract: the camera keeps 0.25 m from every wall piece and blocking footprint (`route.FreeSpace`), light
    furniture stands anywhere clear of them and movers drive the free grid, so the open floor is every point of the
    room at least 0.12 m (the most a wall piece may stand proud) from a wall face or a blocking footprint, and it is
    no box that overlaps another. On the synthetic room (faces at 0.05 and 3.95 / 2.95; counter x 0.05..1.25, y
    0.05..0.65; fridge x 3.25..3.95, y 0.05..0.75): the middle is open; 0.05 m off the south wall, 0.05 m off the
    counter's front and 0.05 m off the fridge's front are not; 0.1 m beyond the counter's end and fridge's side are.
    Regression caught: an inset that grows or shrinks, footprints left out, or a room's floor eroded along a line
    that is no wall."""
    rects = db.open_floor(parse_map(SYNTH))
    boxes = [(r[0], r[1], 0.0, r[2], r[3], 1.0) for r in rects]
    for a in range(len(boxes)):
        assert not any(db.overlaps(boxes[a], b) for b in boxes[a + 1:])

    def inside(x: float, y: float) -> bool:
        return db.covered(_tiny(x, y, 0.4, 0.5), boxes)

    assert inside(2.0, 1.5) and inside(3.0, 0.5) and inside(0.7, 0.9) and inside(1.4, 0.4) and inside(3.5, 0.9)
    assert not inside(2.0, 0.10) and not inside(0.7, 0.7) and not inside(3.5, 0.8) and not inside(3.9, 1.5)
    assert not inside(5.0, 5.0)  # outside the room
    hall = [(r[0], r[1], 0.0, r[2], r[3], 1.0) for r in db.open_floor(load_map("warehouse_a"))]
    assert db.covered(_tiny(8.8, 9.1, 0.4, 0.5), hall)  # the line between two regions of one hall is no wall


def test_the_floor_volumes_reach_the_camera_and_the_tallest_light_piece_mover_or_floor_clutter():
    """Contract: where no robot is planned (the downtown) the camera part of the open floor rises to the highest camera
    height of the retired profiles that fits under the map's ceiling (the drone's 12 m under the city's 45 m ceiling)
    plus what is left of the 0.25 m world clearance after a 0.12 m wall piece, 0.13 m; where robots are planned (the
    home, and the warehouse, where the AMR is) there is none, their swept bodies stand in its place. The part of light
    furniture, movers and floor clutter rises to the tallest of them plus 5 cm: at home the floor lamp (1.5 m, floor
    clutter) beats the chair (0.9 m) and the home robot (0.45 m), so 1.55; at the warehouse the forklift, 2.1 m, beats
    the roll cage, so 2.15; and every one starts 2 cm above the floor, so a flush plate stands clear. Regression caught:
    a camera height or margin changed without the zone, a camera zone left where robots are planned (it would forbid
    what the swept bodies allow), or a lamp too tall for its zone (a piece of dressing it would stand in)."""
    assert pytest.approx(0.13) == dz.EGO_BUFFER_M
    assert _zones(SYNTH, FLOOR_EGO) == [] and _zones(raw_map("warehouse_a"), FLOOR_EGO) == []
    for boxes, top in ((_zones(SYNTH, FLOOR_HEAVY), 1.55),
                       (_zones(raw_map("downtown_a"), FLOOR_EGO), 12.13),
                       (_zones(raw_map("warehouse_a"), FLOOR_HEAVY), 2.15)):
        assert boxes and all(b[2] == 0.02 and b[5] == pytest.approx(top) for b in boxes)


# Boxes on the synthetic room and what a layout author is told. Each is (box, the problems' words, in order).
NOT_IN_A_BAND = "not wholly inside"
SWEPT_VACUUM, SWEPT_HELPER = "swept body of the vacuum", "swept body of the helper"
CASES = {
    "a baseboard on the wall, 1.5 cm proud": ((1.3, 0.03, 0.0, 3.2, 0.065, 0.08), ()),
    "an upper cabinet over the counter": ((0.1, 0.06, 1.45, 1.2, 0.4, 2.1), ()),
    "the same cabinet 25 cm lower, in the volume things stand in": (
        (0.1, 0.06, 1.2, 1.2, 0.4, 1.9), ("receptacle F01/S01",)),
    "a curtain across the glass": ((1.4, 0.05, 1.2, 2.6, 0.12, 2.2), ("glass of window N1",)),
    "a curtain beside the glass": ((1.3, 0.05, 1.0, 1.5, 0.12, 2.3), ()),
    "a sill under the glass": ((1.4, 0.05, 0.97, 2.6, 0.13, 1.0), ()),
    "a radiator 25 cm proud": (
        (2.0, 0.05, 0.1, 2.8, 0.30, 0.7), (NOT_IN_A_BAND, SWEPT_VACUUM, SWEPT_HELPER, FLOOR_HEAVY)),
    "a flush plate 1 cm thick": ((1.5, 0.3, 0.0, 2.5, 1.2, 0.01), ()),
    "a flush plate 1.5 cm thick, which the helper (it climbs 1 cm) cannot drive over": (
        (1.5, 0.3, 0.0, 2.5, 1.2, 0.015), (SWEPT_HELPER,)),
    "a plate 3 cm thick": (
        (1.5, 0.3, 0.0, 2.5, 1.2, 0.03), (NOT_IN_A_BAND, SWEPT_VACUUM, SWEPT_HELPER, FLOOR_HEAVY)),
    "a ceiling lamp": ((1.8, 1.3, 2.1, 2.2, 1.7, 2.5), ()),
    "a lamp floating at head height": ((1.8, 1.3, 1.8, 2.2, 1.7, 2.2), (NOT_IN_A_BAND,)),
    "a lamp through the ceiling": ((1.8, 1.3, 2.1, 2.2, 1.7, 2.6), (NOT_IN_A_BAND,)),
    "trim over the door's head": ((2.55, 2.93, 2.05, 3.45, 2.97, 2.15), ()),
    "a hook in the doorway": (
        (2.7, 2.8, 1.2, 2.8, 2.95, 1.6),
        (NOT_IN_A_BAND, SWEPT_HELPER, FLOOR_HEAVY, "door opening D1", "swing of door D1")),
    "something on top of the fridge": ((3.3, 0.1, 1.85, 3.9, 0.6, 2.2), ()),
    "something inside the fridge's body, where no one can see it": ((3.3, 0.1, 0.5, 3.9, 0.6, 1.0), ()),
}


@pytest.mark.parametrize("name", CASES)
def test_a_box_is_judged_by_where_it_stands(name):
    """Contract: a Phase 1 box passes if it lies wholly inside an allowed place (a wall
    band at most 0.12 m proud, the ceiling's slab, structure above 5 m in a hall, the column over a blocking piece,
    a flush plate) and meets no forbidden volume; otherwise the guard says each reason once, in the order allowed place,
    receptacle, the vacuum's and the helper's swept bodies, light furniture and mover floor, door, window. Worked by
    hand on the synthetic room per case above. Regression caught: a band that stops short of what trim needs (a casing
    over a door, a sill under glass), a zone that is too wide (a curtain beside the window), too narrow (a lamp in the
    doorway, a cabinet in the volume of the counter's items, a plate thick enough to trip a mover) or missing."""
    box, words = CASES[name]
    problems = dz.DressingGuard(SYNTH).problems(box)
    assert len(problems) == len(words), problems
    assert all(w in p for w, p in zip(words, problems, strict=True)), problems


def test_the_wall_bands_are_the_strips_along_faces_that_look_into_a_room():
    """Contract: a wall band is a strip along a wall piece's room-side face, 0.12 m out and 2 cm into the wall, floor to
    ceiling (2 cm into the ceiling); the outside of the building has none; a door's cut is no wall but the wall over its
    head is. The synthetic room has six: the south face, the east face, the north face either side of the door, the west
    face and the head over the door. Regression caught: a band on the outside of an exterior wall (dressing floating
    in the street), a band across a doorway, or no band over a door (no casing head, no lintel trim)."""
    bands = {tuple(round(v, 3) for v in b) for _, b in dz.wall_bands(SYNTH)}
    assert bands == {
        (-0.05, 0.03, 0.0, 4.05, 0.17, 2.52),  # south
        (3.83, -0.05, 0.0, 3.97, 3.05, 2.52),  # east
        (-0.05, 2.83, 0.0, 2.55, 2.97, 2.52),  # north, west of the door
        (3.45, 2.83, 0.0, 4.05, 2.97, 2.52),  # north, east of it
        (0.03, -0.05, 0.0, 0.17, 3.05, 2.52),  # west
        (2.55, 2.83, 2.05, 3.45, 2.97, 2.52),  # over the door
    }


@pytest.mark.parametrize("map_name", MAPS)
def test_wall_ceiling_and_structure_bands_never_reach_the_floor_volumes(map_name):
    """Contract: the places Phase 1 allows on walls and under roofs are outside the open floor that the camera, light
    furniture and movers use, so a layout that stays in them cannot meet either. Regression caught: a band that grew
    past the 0.12 m inset or a ceiling slab deep enough to reach the camera's 1.63 m or the forklift's 2.15 m."""
    spec = raw_map(map_name)
    floors = [b for w, b in dz.forbidden(spec) if w.startswith("open floor")]
    for why, band in [*dz.wall_bands(spec), *dz.ceiling_bands(spec), *dz.structure_band(spec)]:
        assert not any(db.overlaps(band, f) for f in floors), why


def test_a_hall_has_structure_above_five_metres_and_a_house_has_none():
    """Contract: building structure (the roof, trusses, services, lights) is allowed above 5 m, which exists in the 6 m
    warehouse (one slab per region, 5.0 m up to 2 cm into the roof) and not in the 2.5 m apartment, whose ceiling band
    is 0.5 m deep. Regression caught: structure allowed in a room too low for it (a truss at head height)."""
    assert dz.structure_band(raw_map("apartment_a")) == []
    hall = dz.structure_band(raw_map("warehouse_a"))
    assert len(hall) == 4 and all(b[2] == 5.0 and b[5] == pytest.approx(6.02) for _, b in hall)
    assert {b[2] for _, b in dz.ceiling_bands(raw_map("apartment_a"))} == {2.0}


def test_covered_is_the_union_of_the_bands_and_nothing_more():
    """Contract: `covered` asks whether a box lies wholly inside the union of the bands: a box that spans two touching
    bands is covered (a cabinet run over two counters), one that leaves them by any amount, or crosses a gap between
    them, is not. By hand on unit cubes. Regression caught: a check against one band at a time (runs rejected) or one
    that only tests the box's corners (a box over a gap accepted)."""
    a, b = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 2.0, 1.0, 1.0)
    assert db.covered((0.5, 0.2, 0.2, 1.5, 0.8, 0.8), [a, b])
    assert not db.covered((0.5, 0.2, 0.2, 1.5, 0.8, 0.8), [a])
    assert not db.covered((0.5, 0.2, 0.2, 1.5, 0.8, 0.8), [a, (1.1, 0.0, 0.0, 2.0, 1.0, 1.0)])  # a gap of 0.1
    assert not db.covered((0.5, 0.2, 0.2, 1.5, 0.8, 1.01), [a, b])  # 1 cm too high
    assert db.covered((0.5, 0.2, 0.2, 1.5, 0.8, 1.0 + 1e-12), [a, b])  # float noise is no gap


def test_the_overlap_rule_is_the_one_the_renders_dressing_module_has():
    """Contract (`render/dressing_spec.py`): boxes that touch share no volume and one that is inside another does, in
    the guard and in the shared interface a layout author uses, so a layout's own checks agree with the guard's. The
    box of a `Dressing` is its extents after the yaw: 1 x 2 m turned a quarter turn about the base centre is 2 x 1.
    Regression caught: the two copies of the rule drifting apart (one counting touching faces), and a yaw that is not
    applied to the box."""
    spec = render_module("dressing_spec")
    a, touching, inside = (0, 0, 0, 1, 1, 1), (1, 0, 0, 2, 1, 1), (0.2, 0.2, 0.2, 0.8, 0.8, 0.8)
    for other, expected in ((touching, False), (inside, True), ((0.9, 0.9, 0.9, 2, 2, 2), True)):
        assert db.overlaps(a, other) is expected and spec.overlaps(a, other) is expected
    d = spec.Dressing("baseboard", "box", (0.0, 0.0, 0.5), 90.0, (1.0, 2.0, 0.5))
    assert d.box == pytest.approx((-1.0, -0.5, 0.5, 1.0, 0.5, 1.0))


@pytest.mark.parametrize("map_name", MAPS)
def test_the_zones_hold_everything_the_world_puts_and_moves(map_name):
    """Contract (the proof that dressing cannot touch a timeline): the forbidden volumes contain the whole box of every
    carried item at every place it is ever put (a receptacle volume holds it), the centre of every light piece at every
    place (the floor volume of light furniture holds it), and every position of a door leaf, open or shut (the swing
    and opening zones hold it). Worked by the generator, not by this module: the worlds of seeds 1-3 at the start and
    after a gap's moves, shifts, arrivals and departures (`changes`). Regression caught: a margin or inset too small
    for what `placement` and `changes` really do, which would let dressing stand where a thing is put."""
    spec = raw_map(map_name)
    zones = dz.forbidden(spec)
    openings = [b for w, b in zones if "opening" in w]
    heavy = [b for w, b in zones if w.startswith(FLOOR_HEAVY)] + openings
    receptacles = [b for w, b in zones if w.startswith("receptacle")]
    m = load_map(map_name)
    for seed in (1, 2, 3):
        tl = after_gap(map_name, seed)
        kinds = {e.id: e for e in tl.entities}
        for p in tl.placements:
            e = kinds[p.entity]
            if e.mobility == "carried":
                pts = corners(Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], yaw_of(p.rot)))
                box = (min(q[0] for q in pts), min(q[1] for q in pts), p.pos[2], max(q[0] for q in pts),
                       max(q[1] for q in pts), p.pos[2] + e.size[2])
                assert db.covered(box, receptacles), (seed, e.kind, box)
            elif e.mobility == "light":
                assert db.covered(_tiny(p.pos[0], p.pos[1], 0.03, 0.05), heavy), (seed, e.kind, p.pos)
    swings = [b for w, b in zones if w.startswith("swing")]
    for door in m.doors():  # the leaf's line from the hinge to the tip, shut and open, at 0.5-1.0 m
        for is_open in (False, True):
            (hx, hy), (tx, ty) = m.door_leaf(door, is_open)
            for f in (0.01, 0.25, 0.5, 0.75, 0.99):
                assert db.covered(_tiny(hx + f * (tx - hx), hy + f * (ty - hy), 0.5, 1.0, 1e-10), swings + openings), (
                    door.id, is_open, f)


def test_a_map_that_is_not_rectilinear_is_refused_rather_than_boxed_loosely():
    """Contract: where an axis-aligned box would be too permissive (an allowed band over a turned wall, open floor
    around a turned footprint) the module raises instead of guessing, so a map with one cannot pass the guard on a
    loose box. Regression caught: the guard accepting a map it cannot judge."""
    turned = copy.deepcopy(SYNTH)
    turned["furniture"][1]["yaw"] = 0.5  # the fridge
    with pytest.raises(ValueError, match="rectilinear"):
        dz.forbidden(turned)
    with pytest.raises(ValueError, match="rectilinear"):
        dz.allowed_phase1(turned)


def _meets(box: tuple, cls: str | None = None, spec: dict = SYNTH, catalog: str | None = None) -> list[str]:
    return dz.DressingGuard(spec, catalog).problems(box, cls)


def test_no_dressing_inside_any_planned_robots_volume():
    """Contract (M1's R.2: the swept body of each robot planned on the home, from its `climb_m` up to its height plus
    2 cm, over every base pose that keeps its `clearance_m` from the walls and blocking footprints, grown by 2 cm; the
    robots are the vacuum, a 0.34 m disc 0.09 m high that climbs 2 cm and keeps 2 cm, and the helper, 0.34 x 0.33 m,
    1.41 m, climbing 1 cm and keeping 0.1 m). By hand on the synthetic room, whose south wall face is at y = 0.05: the
    vacuum's body reaches the face (0.02 - 0.02) up to 0.11 m, the helper's comes within 0.08 m of it up to 1.43 m, so a
    0.075 m piece at 0.5 m stands clear of the helper and a 0.085 m one meets it; a 2 cm strip at 0-0.08 m meets the
    vacuum and not the helper (its body is 8 cm off the wall); a radiator 0.09 m proud at 0.12-0.67 m meets the helper
    only (0.09 > 0.08), and when it is a radiator from 0.15 m up it meets nothing; trim, a strip 1.5 cm proud, stands in
    both; a plate 1 cm thick lies under both bodies, and 1.5 cm meets the helper's. At the warehouse the AMR is planned:
    a 0.80 x 0.58 m rectangle planned as its circumscribed disc (radius 0.494 m), 0.65 m high, climbing 1 cm and
    keeping 0.10 m, so its body (grown 2 cm, z 0.01 to 0.67 m) comes within 0.08 m of a wall face, a rack's front and a
    rack's end, as the helper does of a wall. By hand on `warehouse_a`: a piece 0.075 m proud of the dock wall (face at
    y = 0.1) between the west wall and the first door (x 1.8 to 2.2) stands clear and one 0.085 m proud meets it, a
    piece 0.115 m proud from 0.7 m up (wholly above the body) meets nothing and from 0.6 m up meets it; the protector
    at the foot of the west rack's first upright (x 1.2 out, y 9.1 to 9.32, 0.4 m high) and the guard across that
    row's end (y 9.1 back, 0.6 m high) clear at 0.075 m and meet it at 0.085. Where no robot is planned (the downtown)
    no volume names one. Regression caught: dressing a robot drives through, which depth sees and the planner does not
    (the 42 columns, protectors and guards that stood 0.10 and 0.115 m out of the warehouse's walls and racks), a trim
    exemption wide enough for a 2 cm strip, or a guard that names a robot no catalog plans."""
    x0, x1 = 1.8, 2.4  # along the south wall's face, under the window's span
    assert _meets((x0, 0.05, 0.5, x1, 0.125, 0.6)) == []
    assert _meets((x0, 0.05, 0.5, x1, 0.135, 0.6)) == [f"meets {SWEPT_HELPER} (z 0.01-1.43 m)"]
    assert _meets((x0, 0.05, 0.0, x1, 0.07, 0.08)) == [f"meets {SWEPT_VACUUM} (z 0.02-0.11 m)"]
    assert _meets((x0, 0.05, 0.0, x1, 0.065, 0.08)) == []  # a strip of trim's thickness: 1.5 cm proud
    radiator = (1.7, 0.05, 0.12, 2.5, 0.14, 0.67)
    assert _meets(radiator) == _meets(radiator, "radiator") == [f"meets {SWEPT_HELPER} (z 0.01-1.43 m)"]
    assert _meets((1.7, 0.05, 0.15, 2.5, 0.14, 0.70)) == [f"meets {SWEPT_HELPER} (z 0.01-1.43 m)"]
    assert _meets((1.7, 0.05, 0.15, 2.5, 0.14, 0.70), "radiator") == []
    assert _meets((1.5, 0.3, 0.0, 2.5, 1.2, 0.01)) == []
    assert _meets((1.5, 0.3, 0.0, 2.5, 1.2, 0.015)) == [f"meets {SWEPT_HELPER} (z 0.01-1.43 m)"]
    wh, amr = raw_map("warehouse_a"), "meets swept body of the amr (z 0.01-0.67 m)"
    assert {w for w, _ in dz.forbidden(wh) if w.startswith("swept body")} == {amr.removeprefix("meets ")}
    assert _meets((1.8, 0.1, 0.3, 2.2, 0.175, 0.5), spec=wh) == []
    assert _meets((1.8, 0.1, 0.3, 2.2, 0.185, 0.5), spec=wh) == [amr]
    assert _meets((1.8, 0.1, 0.7, 2.2, 0.215, 0.9), spec=wh) == []
    assert _meets((1.8, 0.1, 0.6, 2.2, 0.215, 0.8), spec=wh) == [amr]
    assert _meets((1.2, 9.1, 0.0, 1.275, 9.32, 0.4), spec=wh) == []
    assert _meets((1.2, 9.1, 0.0, 1.285, 9.32, 0.4), spec=wh) == [amr]
    assert _meets((0.105, 9.025, 0.0, 1.195, 9.1, 0.6), spec=wh) == []
    assert _meets((0.105, 9.015, 0.0, 1.195, 9.1, 0.6), spec=wh) == [amr]
    assert not any(w.startswith("swept body") for w, _ in dz.forbidden(raw_map("downtown_a")))
    assert {w for w, _ in dz.forbidden(SYNTH) if w.startswith("swept body")} == {
        "swept body of the vacuum (z 0.02-0.11 m)", "swept body of the helper (z 0.01-1.43 m)"}


# Where the map puts the middle of the desk F33, the dining table F07, the hall console F20 and the coffee table F11.
KNEES = {"F33": (8.65, 3.85), "F07": (2.1, 1.9), "F20": (5.4, 6.0), "F11": (2.4, 4.4)}


def test_no_dressing_under_what_a_robot_passes_under():
    """Contract (the review's V-7; `dressing_robots.under_zones`): the vacuum passes under a piece whose underside is
    0.10 m up or more (0.09 m and a 0.01 m margin), so it drives under the desk F33 (0.6 m), the dining table F07
    (0.64), the hall console F20 (0.64) and the coffee table F11 (0.10), not under the sofa (0.08) or the bed (0.05);
    what it drives over there must be what the world holds. A 0.2 x 0.2 x 0.09 m box on the floor under the middle of
    each of the four is refused (it was allowed once: `footprint_bands` lets dressing into the column over a
    blocking piece), a box of 0.1 x 0.1 m wholly inside a leg square of the desk and a flush 1 cm plate under it are
    still allowed, and the same box under the sofa or the bed is too (no robot goes there). The open underside of the
    desk, by hand: its footprint, x 8.35 to 8.95 and y 3.25 to 4.45, from 2 cm to 0.6 m, less four 0.15 m leg squares (a
    0.6 x 0.9 m band between the legs across the piece and two 0.3 x 0.15 m bands along its ends: 0.63 m2). Regression
    caught: dressing in the knee space the vacuum crosses, which depth shows as an obstacle no entity explains against
    the robot's own path, or knee space refused that is a leg."""
    guard = dz.DressingGuard(raw_map("apartment_a"))
    for piece, (x, y) in KNEES.items():
        problems = guard.problems((x - 0.1, y - 0.1, 0.0, x + 0.1, y + 0.1, 0.09))
        rest = [p for p in problems if not (p.startswith("meets open underside of ") and f" {piece} " in p)]
        assert len(rest) == len(problems) - 1, piece
        assert rest == (["meets receptacle F14/S12 (up to 0.37 m)"] if piece == "F11" else []), piece  # on the rug
        assert guard.problems((x - 0.1, y - 0.1, 0.0, x + 0.1, y + 0.1, 0.01)) == [], piece  # a flush plate
    desk = [b for why, b in guard.forbidden if why.startswith("open underside") and " F33 " in why]
    assert sum((b[3] - b[0]) * (b[4] - b[1]) for b in desk) == pytest.approx(0.63)
    assert all(b[2] == 0.02 and b[5] == 0.6 for b in desk)
    assert guard.problems((8.375, 3.275, 0.0, 8.475, 3.375, 0.09)) == []  # inside the leg square x 8.35-8.5, y 3.25-3.4
    for x, y in ((2.4, 5.7), (7.5, 5.15)):  # the sofa and the bed
        assert guard.problems((x - 0.1, y - 0.1, 0.0, x + 0.1, y + 0.1, 0.05)) == []
    assert not any(w.startswith("open underside") for w, _ in dz.forbidden(raw_map("warehouse_a")))
