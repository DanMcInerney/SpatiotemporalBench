"""The warehouse map (`warehouse_a`): the hall, its regions, its 48 surfaces and the room the
forklift and the robot need. Everything here is read from the committed map file; the expected numbers are worked out
by hand in each docstring."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import space_kit
from builders import keyframe
from dressing_kit import render_module
from stbench import asset_manifest as A
from stbench import generate, grid
from stbench.camera import Camera
from stbench.catalog import for_map
from stbench.geometry import Rect, corners, dist_to_rect, points_in_polygon, rect_distance, rects_overlap
from stbench.mapspec import check_map, surface_world_rect
from stbench.robots import ROBOTS, camera_doc
from stbench.route import FreeSpace, blocking_rects
from stbench.seeding import make_rng
from test_mapspec import palette_keys

REPO = Path(__file__).resolve().parents[1]


def racks(wh):
    return [f for f in wh.furniture if f.kind == "pallet_rack"]


def surface_kinds(wh) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in wh.furniture:
        for s in f.surfaces:
            out[s.kind] = out.get(s.kind, 0) + 1
    return out


def area(polygon) -> float:
    """Shoelace."""
    edges = zip(polygon, polygon[1:] + polygon[:1], strict=True)
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in edges)) / 2


def test_the_warehouse_map_is_self_consistent_and_names_the_warehouse_catalog(wh):
    """Contract (plan §1): `check_map` finds nothing in the committed map (furniture inside its region and apart from
    other blockers, every surface on its piece, a known catalog), and the map names the warehouse catalog. Regression
    caught: a rack row pushed through a wall or into its neighbour, a surface larger than its piece, or the `catalog`
    key dropped (the map would silently be sampled with household kinds)."""
    assert check_map(wh) == []
    assert wh.catalog == "warehouse" and wh.name == "warehouse_a"


def test_the_hall_is_a_dock_and_three_aisles_sharing_their_edges(wh):
    """Contract (plan §1): a 17.4 x 19.8 m hall (345 m2) of four open-plan regions, the dock first (the route grows
    from region 0's anchor) and three aisles of 5.8 m by 10.8 m north of it; walls 0.2 m thick and 6.0 m high all round,
    no doors and no windows. Hand values: 17.4 x 9.0 = 156.6 m2 of dock, 3 x (5.8 x 10.8 = 62.64) = 187.92 m2 of aisles,
    344.52 m2 in all; inner faces at x 0.1 and 17.5, y 0.1 and 19.9. Regression caught: a shorter dock (the forklift
    walks fail at 7 m), a fourth aisle, a region renamed (the `room` slot of every question), or a wall on the hall."""
    assert [r.name for r in wh.regions] == ["dock", "west aisle", "middle aisle", "east aisle"]
    assert [round(area(r.polygon), 2) for r in wh.regions] == [156.6, 62.64, 62.64, 62.64]
    assert sum(area(r.polygon) for r in wh.regions) == pytest.approx(344.52)
    xs = [x for r in wh.regions for x, _ in r.polygon]
    ys = [y for r in wh.regions for _, y in r.polygon]
    assert (min(xs), min(ys), max(xs), max(ys)) == (0.1, 0.1, 17.5, 19.9)
    assert wh.bounds == (0.0, 0.0, 17.6, 20.0)
    assert all((r.level, r.z0, r.z1) == (0, 0.0, 6.0) for r in wh.regions)
    assert len(wh.walls) == 4 and all(w.exterior and w.thickness == 0.2 and w.height == 6.0 for w in wh.walls)
    assert wh.openings == () and wh.windows == ()
    assert [wh.region_of(p) for p in ((8.8, 5.0), (3.0, 9.0), (3.0, 9.2), (8.8, 14.0), (14.6, 19.0))] == [
        "R1", "R1", "R2", "R3", "R4"]


def test_the_dock_has_the_planned_pieces_and_the_hall_has_48_surfaces(wh):
    """Contract (plan §0, §1; M3-maps §1): the dock holds four dock doors (0.1 m deep panels against the south wall, no
    surface), four staging lanes (2.8 x 1.4 m floor markings in front of them, blocking, one surface each), two packing
    tables and two shelving units of three shelves, the AMR's charger and one bay in front of each station (no
    surface); the aisles six pallet-rack rows; and light equipment of two pallet jacks and a roll cage. Surfaces by
    hand: 4 lanes + 2 tables + 2 x 3 shelves + 36 rack bay-levels = 48 (the plan's "48 free surfaces"); the charger and
    the bays add none. Regression caught: a lane or a shelf lost (the surfaces the yields were measured on), a dock door
    given a surface (a receptacle nobody can see items on), a bay given one (an item put down inside a zone the world
    never draws onto), or the equipment moved into the forklift's band."""
    by_kind: dict[str, int] = {}
    for f in wh.furniture:
        by_kind[f"{f.mobility}:{f.kind}"] = by_kind.get(f"{f.mobility}:{f.kind}", 0) + 1
    assert by_kind == {"fixed:dock_door": 4, "fixed:staging_lane": 4, "fixed:packing_table": 2, "fixed:shelving": 2,
                       "fixed:pallet_rack": 6, "fixed:charger": 1, "fixed:bay": 4, "light:pallet_jack": 2,
                       "light:roll_cage": 1}
    assert surface_kinds(wh) == {"staging lane": 4, "work table": 2, "shelf": 6, "rack": 36}
    for f in (f for f in wh.furniture if f.kind in ("dock_door", "charger", "bay")):
        assert f.surfaces == (), f.id
    for f in (f for f in wh.furniture if f.kind == "dock_door"):
        assert f.size[0] == 0.1 and f.footprint().cy - 0.05 == pytest.approx(0.1), f.id
    for f in (f for f in wh.furniture if f.kind == "staging_lane"):
        assert f.size[:2] == (2.8, 1.4) and f.blocks, f.id
        assert (f.center[1] - 1.4, f.center[1] + 1.4) == pytest.approx((0.6, 3.4)), f.id  # y 0.6 to 3.4
    shelves = [f for f in wh.furniture if f.kind == "shelving"]
    assert all(f.center[0] - f.size[0] / 2 == pytest.approx(0.1) for f in shelves)  # on the west wall
    assert [sorted(s.z for s in f.surfaces) for f in shelves] == [[0.5, 1.1, 1.7]] * 2
    assert all(f.center[1] < 3.4 for f in wh.furniture if f.mobility == "light")  # parked south of the cross band


def test_a_rack_row_is_single_deep_four_bays_long_and_faces_the_lane(wh):
    """Contract (plan §1): in each aisle a west row and an east row of pallet racks, 1.1 m deep, 10.8 m (four 2.7 m
    bays) long and 3.9 m high, from the dock's edge (y 9.1) to the north wall (y 19.9), their fronts 3.6 m apart. Hand
    values for the west aisle: west row x 0.1-1.2 (centre 0.65, front east), east row x 4.8-5.9 (centre 5.35, front
    west), the lane between them 4.8 - 1.2 = 3.6 m; the same, 5.8 m on, for the other aisles. Regression caught: a
    lane narrower than the forklift's clearance, a row that stops short of the wall (a through aisle: a different
    layout and 2.5 times the keyframes), or a front facing the wall (items on the wrong side of the rack)."""
    rows = racks(wh)
    assert len(rows) == 6 and all(f.size == (1.1, 10.8, 3.9) for f in rows)
    for i, (x0, x1) in enumerate(((0.1, 5.9), (5.9, 11.7), (11.7, 17.5))):
        west, east = rows[2 * i], rows[2 * i + 1]
        assert (west.yaw, east.yaw) == (0.0, math.pi)
        assert west.center == pytest.approx((x0 + 0.55, 14.5)) and east.center == pytest.approx((x1 - 0.55, 14.5))
        assert (east.center[0] - 0.55) - (west.center[0] + 0.55) == pytest.approx(3.6)
        for f in (west, east):
            assert (f.center[1] - 5.4, f.center[1] + 5.4) == pytest.approx((9.1, 19.9))
            assert wh.region_of(f.center) == f"R{i + 2}"


def test_half_the_rack_bay_levels_are_free_surfaces_in_the_same_pattern_in_every_aisle(wh):
    """Contract (plan §1, §3.2): a rack row has three levels (z 0.0, 1.3, 2.6) of four bays, 72 bay-levels in all; the
    bay-levels with no surface hold static stock (the rack builder draws it, it is never an entity) and the 36 that have
    one are the items' free surfaces: 12 per aisle, 6 per row, 12 per level, and the same bay-levels in every aisle (the
    look-alike aisles). A free surface is 1.08 m deep and 2.6 m long, so a pallet (1.2 x 1.0 m) fits it lengthways, two
    to a bay. Hand count: 3 aisles x 2 rows x 4 bays x 3 levels = 72; half is 36. Regression caught: a pattern that
    differs between aisles (landmarks that break the aliasing), a level that is all stock (the top level of free
    surfaces is the one rarely seen), or a rack surface larger than its bay."""
    free: dict[int, set] = {0: set(), 1: set(), 2: set()}
    levels: dict[float, int] = {}
    for f in racks(wh):
        aisle = int(f.center[0] // 5.8)  # 0.65 -> 0, 5.35 -> 0, 6.45 -> 1, ...
        assert len(f.surfaces) == 6, f.id
        for s in f.surfaces:
            assert s.kind == "rack" and s.rect[2:] == (1.08, 2.6) and s.z in (0.0, 1.3, 2.6), s
            r = surface_world_rect(f.center, f.yaw, s)
            free[aisle].add((round(r.cx - aisle * 5.8, 3), round(r.cy, 3), s.z))
            levels[s.z] = levels.get(s.z, 0) + 1
    assert free[0] == free[1] == free[2] and len(free[0]) == 12
    assert levels == {0.0: 12, 1.3: 12, 2.6: 12}
    bays = sorted({y for _, y, _ in free[0]})
    assert bays == pytest.approx([10.45, 13.15, 15.85, 18.55])  # 9.1 + 1.35 + 2.7 b


def test_the_forklift_can_drive_the_cross_band_and_every_lane_to_its_dead_end(wh):
    """Contract (plan §1: a 9 m dock, a 3.6 m lane): on the grid a forklift walks on (cells at its circumradius plus 0.1
    m from every wall and blocker; the forklift's box is 2.6 x 1.1 m, so 1.4116 + 0.1 = 1.5116 m), the cell at the
    dock's anchor reaches the middle of each aisle's lane right up to the dead end, and the cross band has a free strip
    across it. Hand values: the band (y 3.4-9.1) is 5.7 m wide and keeps 5.7 - 2 x 1.5116 = 2.68 m of free cells; a
    lane (3.6 m) keeps 3.6 - 3.02 = 0.58 m. Regression caught: the 7 m dock (a 3.7 m band, a forklift walk that failed
    at the attempt limit), a narrower lane, or equipment parked in the band."""
    kind = for_map(wh).mover_kinds["forklift"]
    grid = FreeSpace(wh, blocking_rects(wh, {f.id: f.footprint() for f in wh.furniture if f.mobility == "light"}),
                     math.hypot(kind.size[0], kind.size[1]) / 2 + 0.1)
    reach = grid.reachable_from(wh.regions[0].anchor)
    for x in (3.0, 8.8, 14.6):  # the lane centres
        ix, iy = grid.cell((x, 18.3))  # 1.6 m short of the north wall's inner face
        assert reach[iy, ix], x
    # x = 5.9 is under the back-to-back racks of aisles 1 and 2, so the band is closed to the north there
    column = [y for y in (3.4 + 0.1 * i for i in range(57)) if reach[grid.cell((5.9, y))[1], grid.cell((5.9, y))[0]]]
    assert len(column) >= 26  # at least 2.6 m of the 5.7 m band
    assert wh.regions[0].anchor == (8.8, 6.0) and grid.is_free(wh.regions[0].anchor)


def test_the_robot_reaches_every_region_from_the_docks_anchor(wh):
    """Contract (route.py: viewpoints are free cells reachable from region 0's anchor): on the ego's grid (0.25 m from
    every wall and blocker) the anchor of every region is a free cell reachable from the dock's, and every region has
    reachable cells. Regression caught: an anchor on a rack or a table (the first viewpoint of a tour then lands
    elsewhere), or a region walled off by furniture (its tours would silently be skipped)."""
    light = {f.id: f.footprint() for f in wh.furniture if f.mobility == "light"}
    grid = FreeSpace(wh, blocking_rects(wh, light), 0.25)
    reach = grid.reachable_from(wh.regions[0].anchor)
    cx, cy = grid.cells_xy(reach)
    for r in wh.regions:
        ix, iy = grid.cell(r.anchor)
        assert reach[iy, ix] and wh.region_of(r.anchor) == r.id, r.name
        assert points_in_polygon(cx, cy, [list(p) for p in r.polygon]).sum() > 100, r.name


def test_light_equipment_boxes_equal_the_procedural_builders_design_boxes(wh):
    """Contract (maps/README, as for the apartment's light pieces): a light piece's `size` is the box its procedural
    builder is built to (`DESIGN_BOX` in render/procedural.py, read from its source: it imports bpy), which the
    manifest records as the model's `native_bbox_m` to 1 mm, since footprints, clearances and the in-view cover all read
    this box and the renderer fits the mesh to it. Regression caught: a pallet jack resized in the map and not in its
    builder (the renderer would squash it), or the reverse."""
    source = REPO / "render" / "procedural.py"
    boxes = next(ast.literal_eval(n.value) for n in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "DESIGN_BOX")
    light = [f for f in wh.furniture if f.mobility == "light"]
    assert {f.kind for f in light} == {"pallet_jack", "roll_cage"}
    for f in light:
        asset = A.packaged().model_for(f.kind)
        assert (asset.mobility, asset.fit["mode"], asset.builder) == ("light", "exact", f.kind), f.id
        assert f.size == pytest.approx(boxes[f.kind], abs=1e-9), f.id
        assert f.size == pytest.approx(asset.native_bbox_m, abs=0.001), f.id


def test_every_warehouse_colour_is_a_colour_the_renderer_can_draw(wh):
    """Contract (VISION rule 10; as the apartment's light pieces): the `color` of every piece of furniture is a key of
    the renderer's palette (render/shapes.PALETTE, read from its source), and a light piece's is a colour, not a
    material word (oak, stone, steel name textures), since its asset is tinted to it. Regression caught: a colour with
    no palette entry ("orange" misspelt), found only in the Blender build of the first warehouse episode."""
    palette = palette_keys()
    assert {f.color for f in wh.furniture} <= palette
    light = {f.name: f.color for f in wh.furniture if f.mobility == "light"}
    assert sorted(f.color for f in wh.furniture if f.kind == "pallet_jack") == ["red", "yellow"]
    assert light["roll cage"] == "grey" and not {*light.values()} & set(A.packaged().palette_materials)


def test_nine_high_bay_lights_give_their_own_power(wh, apt):
    """Contract (plan §1, §2): nine fixtures at z 5.8, three over the dock and two over each aisle, each in its own
    region, at 270 W (about six times the apartment's 45 W for a fixture 2.4 times as high); the apartment's own lights,
    which name no power, keep 45 W at 2.45 m. Regression caught: `watts` ignored by the loader (a dark hall that the
    renderer lights as an apartment), a fixture outside its region, or the default drifting from the apartment's 45."""
    lights = wh.ceiling_lights
    assert [(sum(light.region == r.id for light in lights)) for r in wh.regions] == [3, 2, 2, 2]
    assert len(lights) == 9 and all(light.pos[2] == 5.8 and light.watts == 270.0 for light in lights)
    assert all(wh.region_of(light.pos[:2]) == light.region for light in lights)
    assert len(apt.ceiling_lights) == 6 and all(light.watts == 45.0 and light.pos[2] == 2.45
                                                for light in apt.ceiling_lights)


def test_every_receptacle_the_warehouse_catalog_names_exists_on_the_map(wh):
    """Contract (as the apartment's, test_mapspec): the catalog's priors name map surface kinds, so a typo ("racks")
    would make the 70% typical share silently atypical. Regression caught: a receptacle with no surface on the map."""
    on_map = set(surface_kinds(wh))
    for kind in for_map(wh).kinds.values():
        assert set(kind.receptacles) <= on_map, kind.name
    assert {r for k in for_map(wh).kinds.values() for r in k.receptacles} == on_map  # and every surface kind is used


# -- M3-maps: the aisles' kind, the charger, the bays, the moved lanes and the 5 mm markings (M3-maps.md)
def test_the_aisles_are_of_kind_aisle(wh):
    """Contract (M3-maps §1): the dock keeps its name as its kind and each aisle is of kind "aisle", whatever its
    name, so a job reads rack aisles by kind and M4's warehouses use the same kind (no special case); the render's
    look vocabulary for the warehouse is those two kinds and M4 step W's three (a fulfilment hall's pick, pack and
    staging areas) and the manifest names a floor for each and for none of the three aisle names (that every look
    table has exactly the vocabulary is `test_render_looks`'). Hand values: kinds [dock, aisle, aisle, aisle] over the
    names [dock, west aisle, middle aisle, east aisle], every floor `shader:concrete`. Regression caught: a job finding
    aisles by name, or a look table or floor left on the old kinds (a `LookError` or a placeholder floor at the first
    render)."""
    assert [(r.name, r.kind) for r in wh.regions] == [
        ("dock", "dock"), ("west aisle", "aisle"), ("middle aisle", "aisle"), ("east aisle", "aisle")]
    look = render_module("look").module("warehouse")
    assert set(look.KINDS) == {"dock", "aisle", "pick area", "pack area", "staging area"}
    floors = A.packaged().floors
    assert {floors[k] for k in look.KINDS} == {"shader:concrete"}
    assert not {"west aisle", "middle aisle", "east aisle"} & set(floors)


def test_the_charger_and_its_apron_are_clear(wh):
    """Contract (M3-maps §1; M3-W §1): one charger, fixed and blocking, 0.12 x 0.60 x 0.30 m, black, with no surface,
    its back on the east wall's face (x 17.5) and its front facing west into the dock, and the AMR's apron clear: 1.6 m
    in front of its face and 0.5 m to either side. Hand values: the face is at x 17.5 - 0.12 = 17.38, so the apron is x
    15.78 to 17.5 and y 0.6 to 2.2 (the charger's y 1.1 to 1.7, 0.5 either way); packing table F10 starts at y 3.1, so
    it is 3.1 - 1.7 = 1.4 m from the charger, and rack F18 starts at y 9.1, 9.1 - 1.7 = 7.4 m from it. Regression
    caught: a piece, a wall or a piece of light equipment on the apron (an AMR that cannot back off its charger)."""
    (charger,) = [f for f in wh.furniture if f.kind == "charger"]
    assert (charger.mobility, charger.blocks, charger.size, charger.color) == ("fixed", True, (0.12, 0.6, 0.3), "black")
    assert charger.surfaces == () and charger.yaw == pytest.approx(math.pi) and wh.region_of(charger.center) == "R1"
    assert charger.center[0] + charger.size[0] / 2 == pytest.approx(17.5)
    assert check_map(wh) == []
    apron = Rect(16.64, 1.4, 1.72, 1.6, 0.0)  # x 15.78 to 17.5, y 0.6 to 2.2
    others = [f.footprint() for f in wh.furniture if f.blocks and f is not charger] + wh.wall_pieces()
    assert [r for r in others if rects_overlap(apron, r)] == []
    by_id = {f.id: f for f in wh.furniture}
    assert rect_distance(charger.footprint(), by_id["F10"].footprint()) == pytest.approx(1.4)
    assert rect_distance(charger.footprint(), by_id["F18"].footprint()) == pytest.approx(7.4)


def test_each_station_has_a_bay_on_its_face(wh):
    """Contract (M3-maps §1; M3-W §1): each `packing_table` and `shelving` piece has exactly one `bay` touching its
    front face, 1.6 m deep, 2.0 + 0.1 + 0.1 = 2.2 m wide (0.1 m past the station each side), 5 mm high, with the
    station's yaw, wholly inside the dock; so a job finds a station's bay by kind and geometry. Hand values: packing
    table F09 (17.1, 6.5) faces west, its face at x 16.7, so its bay is centred at x 16.7 - 0.8 = 15.9 and F10's at
    (15.9, 4.1); shelving F11 (0.4, 6.4) faces east, its face at x 0.7, so its bay is at (1.5, 6.4) and F12's at (1.5,
    4.2). Regression caught: a bay a job cannot find (off its face, or two on one), or one the wrong size."""
    bays = [f for f in wh.furniture if f.kind == "bay"]
    stations = {f.id: f for f in wh.furniture if f.kind in ("packing_table", "shelving")}
    expected = {"F09": ("F20", (15.9, 6.5)), "F10": ("F21", (15.9, 4.1)), "F11": ("F22", (1.5, 6.4)),
                "F12": ("F23", (1.5, 4.2))}
    assert sorted(stations) == sorted(expected) and len(bays) == 4
    for sid, st in stations.items():
        front = (math.cos(st.yaw), math.sin(st.yaw))
        on_face = [b for b in bays if rect_distance(b.footprint(), st.footprint()) < 1e-9
                   and (b.center[0] - st.center[0]) * front[0] + (b.center[1] - st.center[1]) * front[1] > 0]
        assert [b.id for b in on_face] == [expected[sid][0]], sid
        (b,) = on_face
        assert b.center == pytest.approx(expected[sid][1]) and b.yaw == pytest.approx(st.yaw), sid
        assert b.size == pytest.approx((1.6, st.size[1] + 0.2, 0.005)) and not b.blocks and b.mobility == "fixed", sid
        assert wh.rect_inside(b.footprint()) and wh.region_of(b.center) == "R1", sid


def test_no_light_piece_starts_in_a_bay(wh):
    """Contract (M3-maps §1; M3-W §4: light pieces start far from every bay, so the coin is the bay's state): over
    seeds 101-130, `build_world` and `light_starts` put no piece of light equipment's footprint over a bay (a start
    moves a piece at most 1 m from its map place; the roll cage's, at (11.7, 1.3), ends at x 12.1, 3.0 m from the
    nearest bay's edge x 15.1, and the jacks' are 5.5 m or more off one). Regression caught: a roll cage left standing
    in bay F20 (the old (15.6, 1.4)), which would block a station in every world."""
    bays = [f.footprint() for f in wh.furniture if f.kind == "bay"]
    for seed in range(101, 131):
        w = generate.build_world(wh, seed)
        generate.light_starts(w, wh, make_rng(wh.name, seed, generate.SEED_KEY, "light_start"), frozenset({"charger"}))
        assert len(w.light_rects) == 3, seed
        assert [e for e, r in w.light_rects.items() if any(rects_overlap(r, b) for b in bays)] == [], seed


def lane_gap(lanes: list[Rect], stations: list, dock) -> float:
    """The least distance from a lane to a station's T or to its straight side leg, over `T_FACE_M` 1.8 to 2.4 m in 0.1
    m steps and `SIDE_M` 1.0, 1.5 and 2.0 m (M3-W §2, written here by hand), on whichever side lies in the dock."""
    polygon = [list(p) for p in dock.polygon]

    def in_dock(p) -> bool:
        return bool(points_in_polygon(np.array([p[0]]), np.array([p[1]]), polygon).all())

    gap = math.inf
    for st in stations:
        out = (math.cos(st.yaw), math.sin(st.yaw))  # out of the station's face
        face = (st.center[0] + out[0] * st.size[0] / 2, st.center[1] + out[1] * st.size[0] / 2)
        for tenths in range(18, 25):
            t = (face[0] + out[0] * tenths / 10, face[1] + out[1] * tenths / 10)
            assert in_dock(t)
            for side in (1.0, 1.5, 2.0):
                for sign in (-1.0, 1.0):
                    arrival = (t[0] - sign * out[1] * side, t[1] + sign * out[0] * side)
                    if in_dock(arrival):
                        leg = np.linspace(t, arrival, 101)
                        gap = min(gap, *(float(dist_to_rect(leg[:, 0], leg[:, 1], lane).min()) for lane in lanes))
    return gap


def test_no_staging_lane_near_a_stations_arrival(wh):
    """Contract (the plan review's B1; M3-W §1): no staging lane lies within 0.65 m of any station's T (the AMR's base
    1.8 to 2.4 m out from the face, centred on the bay) or of the straight leg along the face from its arrival
    waypoint (1.0, 1.5 or 2.0 m from T, on whichever side lies in the dock) to T, for every T in 0.1 m steps; the lanes
    are `amr/1`'s keep-outs too, and a pallet on a lane would otherwise block the visit in a third of the worlds. Hand
    values: shelving F12's T at 2.4 m is (3.1, 4.2), 3.8 - 3.1 = 0.7 m from the moved lane F05 (x 3.8 to 5.2), and
    packing table F10's is (14.3, 4.1), 14.3 - 13.6 = 0.7 m from F08 (x 12.2 to 13.6), each along its leg south along
    the face (the lanes end at y 3.4; F09 and F11, at y 6.5 and 6.4, are 2.7 m or more from them); with the lanes where
    they stood (F05 at x 3.7, F08 at 13.9) the leg runs into one (F12's, x 3.1 from y 4.2 to 2.2, enters F05, x 3.0 to
    4.4, at y 3.4), and at the first proposal (4.3 and 13.2) F08's footprint (x 12.5 to 13.9) is 14.3 - 13.9 = 0.4 m
    from the leg. Regression caught: a lane moved back onto a visit, or one closing a
    station's side leg."""
    lanes = [f.footprint() for f in wh.furniture if f.kind == "staging_lane"]
    stations = [f for f in wh.furniture if f.kind in ("packing_table", "shelving")]
    assert len(lanes) == 4 and len(stations) == 4

    def with_ends(f05_x: float, f08_x: float) -> list[Rect]:
        return [Rect(x, 2.0, 2.8, 1.4, math.pi / 2) for x in (f05_x, 7.1, 10.5, f08_x)]

    assert lane_gap(lanes, stations, wh.regions[0]) == pytest.approx(0.7)
    assert lane_gap(with_ends(4.5, 12.9), stations, wh.regions[0]) == pytest.approx(0.7)  # the map's own, by hand
    assert lane_gap(with_ends(3.7, 13.9), stations, wh.regions[0]) == 0.0
    assert lane_gap(with_ends(4.3, 13.2), stations, wh.regions[0]) == pytest.approx(0.4)


def test_markings_are_floor_to_the_amr(wh):
    """Contract (the plan review's M2): every staging lane and bay is at most 0.005 m high, its surface no higher, and
    to the AMR it is floor: from its camera (mount 0.38 m ahead and 0.50 m up, pitched down 10 degrees) 1.5 m from a
    marking's near edge, millimetre ray-cast depth (`space_kit.render_depth`) read by `grid.observe` with the AMR's band
    (climb 0.01 m, so its floor reaches 0.013 m) gives no occupied fine cell, and every fine cell whose centre lies 0.1
    m or more inside the marking is free. Hand values: a bay is 1.6 x 2.2 m, so 14 x 20 = 280 such cells; a lane is 2.8
    x 1.4 m, so 26 x 12 = 312. Control: the same bay 0.014 m high (1 mm over the floor) reads occupied (at least the
    footprint's 16 x 22 = 352 fine cells, none of its inside free), as the review found, so the instrument can tell.
    Regression caught: paint read as an obstacle in every frame (`rear_clear` stopping at a bay's stripe, a pallet that
    left a lane listed for good)."""
    amr = ROBOTS["amr"]
    cam = Camera.load({**camera_doc(amr), "depth_unit": "mm"})
    marks = [f for f in wh.furniture if f.kind in ("staging_lane", "bay")]
    assert len(marks) == 8 and grid.band(amr)[0] == 0.013
    inside_cells = {"bay": 280, "staging_lane": 312}

    def read(piece, height: float) -> tuple[list, list, list]:
        """(occupied cells, the cells 0.1 m or more inside the piece, those of them that are free) from 1.5 m off."""
        front = (math.cos(piece.yaw), math.sin(piece.yaw))
        reach = piece.size[0] / 2 + 1.5  # the camera, from the piece's centre out of its front
        yaw = piece.yaw + math.pi
        mount = amr.camera.mount_m[0]
        base = (piece.center[0] + front[0] * reach - mount * math.cos(yaw),
                piece.center[1] + front[1] * reach - mount * math.sin(yaw), 0.0)
        kf = keyframe("amr", 0, 0.0, base, yaw, task=0, phase="transit")
        xs, ys = zip(*corners(piece.footprint()), strict=True)  # the pieces are axis-aligned
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        depth = space_kit.render_depth(cam, kf.pos, kf.rot, [space_kit.box(x0, x1, y0, y1, 0.0, height)])
        states = grid.observe(grid.points(depth, cam, kf.pos, kf.rot), 0.0, grid.band(amr))
        inside = [(u, v) for u in range(math.floor(x0 * 10), math.ceil(x1 * 10))
                  for v in range(math.floor(y0 * 10), math.ceil(y1 * 10))
                  if x0 + 0.1 <= (u + 0.5) / 10 <= x1 - 0.1 and y0 + 0.1 <= (v + 0.5) / 10 <= y1 - 0.1]
        occupied = [c for c, s in states.items() if s == "occupied"]
        return occupied, inside, [c for c in inside if states.get(c) == "free"]

    for f in marks:
        assert f.size[2] <= 0.005 and all(s.z <= 0.005 for s in f.surfaces), f.id
        occupied, inside, free = read(f, f.size[2])
        assert occupied == [] and len(inside) == inside_cells[f.kind] and free == inside, f.id
    bay = next(f for f in marks if f.kind == "bay")
    occupied, inside, free = read(bay, 0.014)
    assert len(occupied) >= 352 and free == []


def test_the_new_piece_kinds_have_registered_placeholder_builders():
    """Contract (M3-maps §3): `charger` and `bay` are in `furniture_warehouse.BUILDERS` and `nest` and `drop_zone` in
    `furniture_city.BUILDERS` (read from the sources, which import bpy), the tables `build_scene` draws a fixed piece
    from, so a placeholder render of either map raises nothing. That a function exists for every kind of every map is
    `test_asset_render`'s; this is that it is registered. Regression caught: a builder written and left out of the
    table (`no placeholder geometry for fixed kind 'charger'` in the middle of the first render)."""
    for module, kinds in (("furniture_warehouse.py", {"charger", "bay"}), ("furniture_city.py", {"nest", "drop_zone"})):
        tree = ast.parse((REPO / "render" / module).read_text(encoding="utf-8"))
        (table,) = [n.value for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "BUILDERS"]
        registered = {e.id for e in table.generators[0].iter.elts}  # {f.__name__: f for f in (a, b, ...)}
        assert kinds <= registered, (module, sorted(kinds - registered))
