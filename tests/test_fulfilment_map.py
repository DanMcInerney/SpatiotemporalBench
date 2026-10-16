"""The `fulfilment` map (M4-fulfilment, part 1): a 27.8 x 29.9 m hall (x 0.1-27.9, y 0.1-30.0)
of six regions, built by `tools/maps/fulfilment.py` from the warehouse catalog's pieces for `amr/1`. The map is
not in the package yet (the coordinator's rows for its region kinds come first), so the fixture writes the maker's
`spec()` to a temporary file and loads that by path. Every expected number is worked out by hand in its docstring:
the hall's faces (x 0.1 and 27.9, y 0.1 and 30.0); the dock y 0.1-7.1, the staging area 7.1-11.1, the pack area (x
0.1-16.3) y 11.1-20.1 and the pick area y 20.1-30.0; aisle 1 x 16.3-22.1 and aisle 2 22.1-27.9 (rows 1.1 m deep,
lanes 3.6 m) from y 11.1 to the north wall."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from dressing_kit import layout_of, render_module
from stbench import generate
from stbench.catalog import for_map
from stbench.dressing_boxes import rooms, square
from stbench.generate import build_world
from stbench.geometry import Rect, dist_to_rect, rect_distance, rects_overlap
from stbench.mapspec import check_map, load_map, surface_world_rect
from stbench.robots import ROBOTS
from stbench.route import robot_free_space
from stbench.scenery import layout_problems
from stbench.seeding import make_rng

MAKER = Path(__file__).resolve().parents[1] / "tools" / "maps" / "fulfilment.py"


@cache
def maker():
    """`tools/maps/fulfilment.py` loaded by path (`tools/` is no package)."""
    spec = importlib.util.spec_from_file_location("fulfilment_maker", MAKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def doc() -> dict:
    return maker().spec()


@pytest.fixture(scope="module")
def ful(doc, tmp_path_factory):
    path = tmp_path_factory.mktemp("fulfilment") / "fulfilment.json"
    path.write_text(maker().dump(doc), encoding="utf-8")
    return load_map(path)


@pytest.fixture(scope="module")
def bare_world(ful):
    """The map's world with no carried item and no clutter: only its fixed pieces and its light equipment at their
    map places, which is what a robot's free space is made of before a script puts anything on the floor."""
    return dataclasses.replace(build_world(ful, 101), carried={}, clutter={})


def area(polygon) -> float:
    """Shoelace."""
    edges = zip(polygon, polygon[1:] + polygon[:1], strict=True)
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in edges)) / 2


def of_kind(m, kind: str) -> list:
    return [f for f in m.furniture if f.kind == kind]


def face_of(st) -> tuple[tuple[float, float], tuple[float, float]]:
    """(the middle of a station's front face, the unit vector out of it)."""
    out = (math.cos(st.yaw), math.sin(st.yaw))
    return (st.center[0] + out[0] * st.size[0] / 2, st.center[1] + out[1] * st.size[0] / 2), out


# the seven stations, by hand: (kind, centre, bay's centre, yaw). A packing table (0.8 x 2.0 m) stands against the west
# wall (face x 0.1 + 0.8 = 0.9, bay centre 0.9 + 0.8 = 1.7), a west shelving unit (0.6 deep: face x 0.7, bay x 1.5) too,
# and a north one (centre y 30.0 - 0.3 = 29.7, face y 29.4, bay centre y 29.4 - 0.8 = 28.6) faces south
STATIONS = [
    ("packing_table", (0.5, 12.4), (1.7, 12.4), 0.0), ("packing_table", (0.5, 15.4), (1.7, 15.4), 0.0),
    ("packing_table", (0.5, 18.4), (1.7, 18.4), 0.0), ("shelving", (0.4, 22.5), (1.5, 22.5), 0.0),
    ("shelving", (0.4, 25.5), (1.5, 25.5), 0.0), ("shelving", (6.0, 29.7), (6.0, 28.6), -math.pi / 2),
    ("shelving", (12.5, 29.7), (12.5, 28.6), -math.pi / 2),
]


def stations(m) -> list:
    return [f for f in m.furniture if f.kind in ("packing_table", "shelving")]


def test_fulfilment_is_a_valid_rectilinear_warehouse(doc, ful):
    """Contract (brief §1; `mapspec.check_map`, `dressing_boxes`): one rectangular hall of the warehouse catalog, 0.2
    m walls 6.0 m high, no door and no window, six regions that tile it with no wall between them, rectilinear (the
    dressing and the zones refuse a piece turned off a quarter turn), the four region kinds a question names unique
    (`dock`, `staging area`, `pack area`, `pick area`; the two aisles share `aisle` and are never named), 24 high-bay
    lights at 5.8 m and 270 W, one in the region that holds it, and the light equipment at `warehouse_a`'s sizes; and
    the file the maker writes is its `spec()`. Hand values: hall 27.8 x 29.9 = 831.22 m2 = dock 27.8 x 7.0 = 194.6 +
    staging 27.8 x 4.0 = 111.2 + pack 16.2 x 9.0 = 145.8 + pick 16.2 x 9.9 = 160.38 + two aisles of 5.8 x 18.9 =
    109.62; lights 5 (dock) + 3 (staging) + 4 + 4 (pack, pick) + 4 + 4 (the aisles), 831.22 / 24 = 34.6 m2 each.
    Regression caught: a piece through a wall or a region's edge, a region name a question could not use, a light off
    its region or above the lens limit, a piece turned off a quarter turn (met only at render time), a maker whose
    file is not its spec."""
    assert check_map(ful) == []
    assert (ful.name, ful.catalog, ful.bounds, ful.wall_thickness, ful.wall_height) == (
        "fulfilment", "warehouse", (0.0, 0.0, 28.0, 30.1), 0.2, 6.0)
    assert len(ful.walls) == 4 and all(w.exterior and w.thickness == 0.2 and w.height == 6.0 for w in ful.walls)
    assert ful.openings == () and ful.windows == ()
    assert [(r.name, r.kind) for r in ful.regions] == [
        ("dock", "dock"), ("staging area", "staging area"), ("pack area", "pack area"), ("pick area", "pick area"),
        ("aisle 1", "aisle"), ("aisle 2", "aisle")]
    assert [round(area(r.polygon), 2) for r in ful.regions] == [194.6, 111.2, 145.8, 160.38, 109.62, 109.62]
    assert sum(area(r.polygon) for r in ful.regions) == pytest.approx(27.8 * 29.9) == pytest.approx(831.22)
    xs = [x for r in ful.regions for x, _ in r.polygon]
    ys = [y for r in ful.regions for _, y in r.polygon]
    assert (min(xs), min(ys), max(xs), max(ys)) == (0.1, 0.1, 27.9, 30.0)
    assert all((r.level, r.z0, r.z1) == (0, 0.0, 6.0) for r in ful.regions)
    boxes = [Rect((a + c) / 2, (b + d) / 2, c - a, d - b, 0.0) for a, b, c, d in rooms(ful)]  # raises if off-axis
    assert not [(i, j) for i in range(6) for j in range(i) if rects_overlap(boxes[i], boxes[j])]  # no overlap
    kinds = [r.kind for r in ful.regions]
    assert all(kinds.count(k) == 1 for k in ("dock", "staging area", "pack area", "pick area"))
    for f in ful.furniture:
        square(f.footprint(), f.id)  # a quarter turn, or a ValueError
    for wall in ful.wall_pieces():
        square(wall, "a wall piece")
    lights = ful.ceiling_lights
    assert len(lights) == 24 and all(light.pos[2] == 5.8 and light.watts == 270.0 for light in lights)
    assert all(ful.region_of(light.pos[:2]) == light.region for light in lights)
    assert [sum(light.region == r.id for light in lights) for r in ful.regions] == [5, 3, 4, 4, 4, 4]
    light = sorted((f.kind, f.size) for f in ful.furniture if f.mobility == "light")
    assert light == [("pallet_jack", (1.55, 0.55, 1.2))] * 2 + [("roll_cage", (0.8, 0.7, 1.7))]
    assert json.loads(maker().dump(doc)) == doc


def test_the_warehouse_dressing_builds_on_the_hall_and_its_fixed_wall_spots_are_free(doc):
    """Contract (brief §1, "what the warehouse dressing assumes"): the warehouse dressing
    (`render/dressing_layout_warehouse.py`), which reads the hall from the map's bounds less half a wall and puts
    its fittings at fixed wall spots (west and east walls at y 7.6 and 8.6, the south wall at x 1.1 and 26.9, the
    north wall at x 9.8), builds on this map without raising (a fitting that would meet a piece raises), emits every
    one of its 27 classes, and draws what the layout must have: five extinguishers with a call point beside each (the
    five spots), one utility enclosure and one power panel, one roof light fixture a map light (24), a door number a
    dock door (4), a guard on the open end of each of the four rack rows and one sign an aisle (2). That the boxes
    meet no forbidden volume is `test_dressing`'s once the map is in the package. Regression caught: a station or
    lane standing at a fitting's spot (a ValueError at the first render), or a layout of the hall that leaves a
    dressing class undrawn."""
    boxes = layout_of("warehouse", doc)
    assert layout_problems("warehouse", [(d.cls, d.phase) for d in boxes]) == []
    counts = Counter(d.cls for d in boxes)
    assert (counts["fire_extinguisher"], counts["fire_alarm"], counts["utility_enclosure"],
            counts["power_panel"]) == (5, 5, 1, 1)
    assert (counts["high_bay_light"], counts["door_number"], counts["end_of_aisle_guard"],
            counts["aisle_sign"]) == (24, 4, 4, 2)


def test_every_station_has_its_bay_and_the_charger_its_apron(ful):
    """Contract (brief §1; M3-W §1): three `packing_table` and four `shelving` stations, each with exactly one `bay`
    on its front face (touching it, in front of it), 1.6 m deep and 0.2 m wider than the station's 2.0 m, at most
    0.005 m high, `blocks: false`, no surface, wholly inside the station's own region; no bay that is no station's;
    and one `charger`, 0.12 x 0.60 x 0.30 m, its back on the east wall's face (x 27.9) facing west, with 1.6 m in
    front and 0.5 m either side clear of every blocking piece, wall and light piece. Hand values: the stations and
    bays are STATIONS' (the packing tables' faces at x 0.9, bays at x 1.7; the west shelving's at 0.7 and 1.5; the
    north shelving's at y 29.4 and 28.6); the charger's centre is (27.84, 3.6), its face x 27.78, so the apron is x
    26.18 to 27.9 and y 2.8 to 4.4. Regression caught: a station the AMR cannot approach (no bay, two bays, one off
    the face), a bay that stands in two regions, or a charger the AMR cannot back off."""
    assert (len(of_kind(ful, "packing_table")), len(of_kind(ful, "shelving")), len(of_kind(ful, "bay"))) == (3, 4, 7)
    assert len(stations(ful)) == len(STATIONS)
    bays = of_kind(ful, "bay")
    for kind, centre, bay_centre, yaw in STATIONS:
        (st,) = [f for f in ful.furniture if f.kind == kind and f.center == pytest.approx(centre)]
        assert st.yaw == pytest.approx(yaw)
        _, out = face_of(st)
        on_face = [b for b in bays if rect_distance(b.footprint(), st.footprint()) < 1e-9
                   and (b.center[0] - st.center[0]) * out[0] + (b.center[1] - st.center[1]) * out[1] > 0]
        assert len(on_face) == 1, (kind, centre)
        (b,) = on_face
        assert b.center == pytest.approx(bay_centre) and b.yaw == pytest.approx(yaw), (kind, centre)
        assert b.size == pytest.approx((1.6, 2.2, 0.005)) and not b.blocks and b.surfaces == (), (kind, centre)
        assert b.size[2] <= 0.005 and ful.rect_inside(b.footprint())
        assert ful.region_of(b.center) == ful.region_of(st.center), (kind, centre)
    (charger,) = of_kind(ful, "charger")
    assert (charger.mobility, charger.blocks, charger.size, charger.color) == ("fixed", True, (0.12, 0.6, 0.3), "black")
    assert charger.surfaces == () and charger.yaw == pytest.approx(math.pi)
    assert charger.center == pytest.approx((27.84, 3.6))
    assert charger.center[0] + charger.size[0] / 2 == pytest.approx(27.9)  # its back on the east wall's face
    apron = Rect(27.04, 3.6, 1.72, 1.6, 0.0)  # x 26.18 to 27.9, y 2.8 to 4.4
    others = [f.footprint() for f in ful.furniture if f.blocks and f is not charger] + ful.wall_pieces()
    assert [r for r in others if rects_overlap(apron, r)] == []
    assert rects_overlap(apron, Rect(26.5, 4.0, 0.4, 0.4, 0.0))  # control: a piece on the apron is seen


def approach_gap(lanes: list[Rect], sts: list, t_face: float, leg: float = 2.0) -> float:
    """The least distance from a lane to a station's T (its base `t_face` out from the face, centred on the bay) or
    to the straight leg of `leg` metres along the face on either side of T."""
    gap = math.inf
    for st in sts:
        face, out = face_of(st)
        t = (face[0] + out[0] * t_face, face[1] + out[1] * t_face)
        for sign in (-1.0, 1.0):
            end = (t[0] - sign * out[1] * leg, t[1] + sign * out[0] * leg)
            points = np.linspace(t, end, 201)
            gap = min(gap, *(float(dist_to_rect(points[:, 0], points[:, 1], lane).min()) for lane in lanes))
    return gap


def test_no_staging_lane_lies_by_a_stations_approach(ful):
    """Contract (M3-W §1, §3; brief §1): `amr/1` turns to face a station at T, 2.2 m out from its face and centred on
    its bay, after a straight leg along the face from an arrival waypoint 2.0 m from T on either side (1.5 and 1.0 m
    if that is not free), and plans with every staging lane as a keep-out. No `staging lane` surface lies within 0.6
    m of any station's T or of either leg; and the same at 0.65 m for T anywhere in 1.8 to 2.4 m out in 0.1 m steps
    (the range M3-W may settle `T_FACE_M` in, and its map test's margin). Hand control: a lane surface (2.7 x 1.3 m)
    laid with its near edge 0.5 m from the first packing table's T (3.1, 12.4) is 0.5 m from it; one laid across T is
    0. Regression caught: a lane `amr/1`'s contract refuses, where a station's approach passes over its slot (a
    pallet on it blocks the visit in a third of the worlds, M3-maps' B1)."""
    lanes = [surface_world_rect(f.center, f.yaw, s) for f in of_kind(ful, "staging_lane") for s in f.surfaces]
    sts = stations(ful)
    assert len(lanes) == 4 and len(sts) == 7
    assert approach_gap(lanes, sts, 2.2) >= 0.6
    assert all(approach_gap(lanes, sts, tenths / 10) >= 0.65 for tenths in range(18, 25))
    first = [f for f in sts if f.center == pytest.approx((0.5, 12.4))]
    assert approach_gap([Rect(3.1 + 0.5 + 0.65, 12.4, 2.7, 1.3, math.pi / 2)], first, 2.2) == pytest.approx(0.5)
    assert approach_gap([Rect(3.1, 12.4, 2.7, 1.3, math.pi / 2)], first, 2.2) == 0.0


def test_every_reverse_corridor_runs_into_open_floor(ful):
    """Contract (brief §1; M3-W §3, gate (g)): after a visit the AMR reverses out of its bay along its approach line,
    and the corridor `rear_clear` reads is its width (0.58 m), 5 m behind the docked rear edge (0.92 m from the
    face), so 5.92 m out. In front of every station's face the band the bay's width (2.2 m) wide and 7 m deep holds
    no fixed piece but the station's own bay: not another station, a lane, a rack, the charger, a door, and no other
    station's bay. Hand control: a piece 6.5 m from the first packing table's face (x 0.9 + 6.5 = 7.4) is in its
    band, one 7.5 m out is not (x 8.4, past the band's end at 7.9). Regression caught: a station whose undock a map
    raycast answers (a wall or a rack a few metres behind the robot, M2's back-out failure), or a station facing
    another across a narrow way."""
    fixed = [f for f in ful.furniture if f.mobility == "fixed"]
    bays = of_kind(ful, "bay")
    for st in stations(ful):
        face, out = face_of(st)
        band = Rect(face[0] + out[0] * 3.5, face[1] + out[1] * 3.5, 7.0, 2.2, st.yaw)  # 7 m deep from the face
        (own,) = [b for b in bays if rect_distance(b.footprint(), st.footprint()) < 1e-9
                  and (b.center[0] - st.center[0]) * out[0] + (b.center[1] - st.center[1]) * out[1] > 0]
        hits = [f.id for f in fixed if f is not st and rects_overlap(band, f.footprint())]
        assert hits == [own.id], (st.id, hits)
    band = Rect(0.9 + 3.5, 12.4, 7.0, 2.2, 0.0)  # the first packing table's: x 0.9 to 7.9, y 11.3 to 13.5
    assert rects_overlap(band, Rect(7.4, 12.4, 0.5, 0.5, 0.0))
    assert not rects_overlap(band, Rect(8.4, 12.4, 0.5, 0.5, 0.0))


def flood(free: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """The cells of `free` (a boolean grid) 4-connected to `seed` (row, column), grown by dilation."""
    reached = np.zeros_like(free)
    reached[seed] = free[seed]
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if (grown == reached).all():
            return reached
        reached = grown


def test_the_amr_can_reach_every_bay_and_lane(ful, bare_world):
    """Contract (brief §1; M3-W §1, §3): on `route`'s free space for the `amr` robot (its base 0.612 m from every
    wall and piece, on 2.5 cm cells; read at 10 cm, a cell counting when any of its 16 fine cells is free) the point
    1.0 m out from the charger (27.78 - 0.4 - 1.0 = 26.38, 3.6) reaches every bay's centre (the seven of STATIONS),
    every staging lane's centre ((6.0, 9.1), (9.4, 9.1), (12.8, 9.1), (16.2, 9.1)), every station's T (2.2 m out from
    its face: x 3.1 for the packing tables, 2.9 for the west shelving, y 27.2 for the north), and the 28 pick poses
    (the body's centre 0.64 m from a rack face, 0.35 m clear: x 18.04 and 20.36 in aisle 1, 23.84 and 26.16 in aisle
    2, at y 12.45 + 2.7 b for the seven bays b). Control: 0.55 m from a rack face (under 0.612) is not free, and a
    point in a rack is not. Regression caught: an aisle or a gap narrower than the AMR's 1.25 m, or a piece that
    walls a station or a lane off."""
    fs = robot_free_space(ful, bare_world, ROBOTS["amr"])
    assert fs.nx % 4 == 0 and fs.ny % 4 == 0
    coarse = fs.free.reshape(fs.ny // 4, 4, fs.nx // 4, 4).any(axis=(1, 3))
    start = (26.38, 3.6)
    assert fs.is_free(start)
    reached = flood(coarse, (fs.cell(start)[1] // 4, fs.cell(start)[0] // 4))
    wanted = [b for _, _, b, _ in STATIONS] + [(x, 9.1) for x in (6.0, 9.4, 12.8, 16.2)]
    wanted += [(3.1, y) for y in (12.4, 15.4, 18.4)] + [(2.9, y) for y in (22.5, 25.5)] + [(6.0, 27.2), (12.5, 27.2)]
    wanted += [(x, round(12.45 + 2.7 * b, 2)) for x in (18.04, 20.36, 23.84, 26.16) for b in range(7)]
    assert len(wanted) == 7 + 4 + 7 + 28
    unreached = [p for p in wanted if not (fs.is_free(p) and reached[fs.cell(p)[1] // 4, fs.cell(p)[0] // 4])]
    assert unreached == []
    assert not fs.is_free((17.4 + 0.55, 20.55)) and not fs.is_free((16.85, 20.55)) and fs.is_free((17.4 + 0.64, 20.55))


def test_every_pick_face_has_its_surface(ful):
    """Contract (brief §1; M4-integration call 23): four `pallet_rack` rows (1.1 x 18.9 x 3.9 m: seven 2.7 m bays,
    three levels at z 0.0, 1.3 and 2.6), two lining each aisle region with their fronts 3.6 m apart facing each other
    (centres x 16.85 and 21.55 in aisle 1, 22.65 and 27.35 in aisle 2; fronts at 17.4, 21.0, 23.2 and 26.8), from the
    staging edge (y 11.1) to the north wall (30.0); every floor-level bay-level has its `rack` surface (1.08 x 2.6 m
    at the bay's middle, local y -8.1, -5.4, -2.7, 0, 2.7, 5.4, 8.1) and no other bay-level has one, so `rack_layout`
    says the builder stocks exactly the 14 others of a row: 56 of 84, two thirds (reported, not set). Regression
    caught: an aisle with a missing pick face, a surface on a stocked level (an item put where the stock stands), or
    a fill rule drifting."""
    rack_layout = render_module("rack_layout")
    racks = of_kind(ful, "pallet_rack")
    assert [(f.center, round(f.yaw, 4)) for f in racks] == [
        ((16.85, 20.55), 0.0), ((21.55, 20.55), round(math.pi, 4)), ((22.65, 20.55), 0.0),
        ((27.35, 20.55), round(math.pi, 4))]
    aisles = [r for r in ful.regions if r.kind == "aisle"]
    assert len(aisles) == 2
    for i, f in enumerate(racks):
        assert f.size == (1.1, 18.9, 3.9) and (f.center[1] - 9.45, f.center[1] + 9.45) == pytest.approx((11.1, 30.0))
        assert ful.region_of(f.center) == aisles[i // 2].id
        assert len(f.surfaces) == 7 and all(s.kind == "rack" and s.z == 0.0 and s.rect[2:] == (1.08, 2.6)
                                            for s in f.surfaces)
        assert sorted(s.rect[1] for s in f.surfaces) == pytest.approx([-8.1, -5.4, -2.7, 0.0, 2.7, 5.4, 8.1])
        surfaces = [(*s.rect, s.z) for s in f.surfaces]
        assert rack_layout.layout(f.size, surfaces) == (7, 3, {(b, 0) for b in range(7)})
        assert rack_layout.stock(f.size, surfaces) == [(b, v) for b in range(7) for v in (1, 2)]
    for first, second in ((racks[0], racks[1]), (racks[2], racks[3])):
        assert (second.center[0] - 0.55) - (first.center[0] + 0.55) == pytest.approx(3.6)


def test_every_receptacle_the_warehouse_catalog_names_exists_on_the_map(ful):
    """Contract (brief §1: "the delivered pallet's receptacle (`rack`, `staging lane`) present"; as `warehouse_a`'s):
    the map's surface kinds are exactly the receptacles the warehouse catalog names, `rack`, `staging lane`, `shelf`
    and `work table`, so a delivery lands and no kind of item is left without a place. Hand count: 28 rack + 4 lane +
    3 table + 4 shelving x 4 shelves = 51 surfaces. Regression caught: a typo in a surface kind (the typical share
    silently atypical), or a delivery with no receptacle."""
    counts = Counter(s.kind for f in ful.furniture for s in f.surfaces)
    assert counts == {"rack": 28, "staging lane": 4, "work table": 3, "shelf": 16} and sum(counts.values()) == 51
    catalog = for_map(ful)
    assert {r for k in catalog.kinds.values() for r in k.receptacles} == set(counts)
    assert set(catalog.kinds[catalog.delivery_kind].receptacles) == {"rack", "staging lane"}


def test_no_light_piece_starts_in_a_bay_or_on_the_chargers_apron(ful, bare_world):
    """Contract (M3-maps §1; M3-W §4: light pieces start far from every bay, so the bay coin alone says its state; the
    charger's apron stays clear): over seeds 101-130, `light_starts` (a piece moves 0.3 to 1.0 m, `SHIFT_M`, from its
    map place, and stays in its own room) puts no light piece's footprint over a bay or over the AMR's apron (x 26.18
    to 27.9, y 2.8 to 4.4). By hand: the jacks (7.2 and 8.2, 1.9) and the roll cage (14.5, 1.3) stand in the dock; the
    roll cage's footprint ends at x 14.9, 26.18 - 14.9 = 11.3 m from the apron, and the nearest jack (footprint x 6.925
    to 7.475, y up to 2.675) is 4.4 m in x and 8.6 m in y from the first packing table's bay (x 0.9 to 2.5, y from
    11.3), 9.7 m off, so no start reaches either. Regression caught: a pallet jack or roll cage parked near the
    charger or a station, which a start could then drop in front of it in every world."""
    bays = [f.footprint() for f in of_kind(ful, "bay")]
    apron = Rect(27.04, 3.6, 1.72, 1.6, 0.0)
    for seed in range(101, 131):
        w = copy.deepcopy(bare_world)
        rng = make_rng(ful.name, seed, generate.SEED_KEY, "light_start")
        generate.light_starts(w, ful, rng, frozenset({"charger"}))
        assert len(w.light_rects) == 3, seed
        assert [e for e, r in w.light_rects.items() if any(rects_overlap(r, b) for b in bays + [apron])] == [], seed
