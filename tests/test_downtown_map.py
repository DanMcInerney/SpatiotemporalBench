"""The downtown map (`downtown_a`): four streets, their surfaces, the blocks around them and the
manifest's models and floors for them. Everything here is read from the committed map file; the expected numbers are
worked out by hand in each docstring."""

from __future__ import annotations

import math

import pytest

from stbench import asset_manifest as A
from stbench.catalog import CATALOGS, DOWNTOWN_KINDS, DOWNTOWN_MOVER_KINDS, for_map
from stbench.geometry import Rect, nearest_on_rect, point_in_rect, rect_inside_rect, rects_overlap
from stbench.mapspec import check_map, load_map, surface_world_rect
from stbench.robots import ROBOTS
from stbench.route import blocking_rects


@pytest.fixture(scope="module")
def city():
    return load_map("downtown_a")


def of_kind(m, kind):
    return [f for f in m.furniture if f.kind == kind]


def surface_kinds(m) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in m.furniture:
        for s in f.surfaces:
            out[s.kind] = out.get(s.kind, 0) + 1
    return out


def test_the_downtown_map_is_self_consistent_and_names_the_city_catalog(city):
    """Contract (plan-city §1): `check_map` finds nothing in the committed map (furniture inside its street and apart
    from other blockers, every surface on its piece, a known catalog and depth unit, the diagonal inside the depth
    range), the map names the downtown catalog, centimetre depth, and an outdoor sun, has no door, window or ceiling
    light. Regression caught: a street piece pushed through a facade or into its neighbour, the `catalog` key dropped
    (the city would be sampled with household kinds), or a door or a light left in an outdoor map."""
    assert check_map(city) == []
    assert (city.name, city.catalog, city.depth_unit) == ("downtown_a", "downtown", "cm")
    assert for_map(city) is CATALOGS["downtown"]
    assert city.outdoor is not None and 0 < city.outdoor.sun_elevation_deg < 90
    assert city.outdoor.sun_strength > 0 and city.outdoor.sky_strength > 0
    assert city.openings == () and city.windows == () and city.ceiling_lights == ()
    assert city.wall_thickness == 0.6 and city.wall_height == 45.0


def test_the_city_is_a_high_street_two_side_streets_and_a_square_sharing_their_edges(city):
    """Contract (plan-city §0, §1): four rectangular regions, the high street first (the route grows from region 0's
    anchor): the 100 x 27 m high street (R1), two 20 x 45 m side streets north of it (R2 at x 15-35, R3 at x 65-85) and
    a 30 x 24 m square south of it (R4 at x 35-65), each with its anchor at its middle (the side streets 12 m in from
    the high street), 45 m tall. Hand values: 100 x 27 = 2,700 m2, 2 x (20 x 45 = 900), 30 x 24 = 720, 5,220 m2 in
    all. The side streets' south edges lie on the high street's north edge (y 27) and the square's north edge on its
    south edge (y 0), so the planner joins them; the bounds are the regions' extent plus a wall thickness. Regression
    caught: a third side street (the route is 236-275 keyframes and 1-4% of attempts land in range), a region renamed
    (the `room` slot of every question reads "in the high street"), or a square that opens beside a side street (the
    streets would see each other and `where_last` would die of re-sightings)."""
    assert [(r.id, r.name) for r in city.regions] == [
        ("R1", "high street"), ("R2", "west street"), ("R3", "east street"), ("R4", "square")]
    boxes = {r.name: (min(x for x, _ in r.polygon), min(y for _, y in r.polygon), max(x for x, _ in r.polygon),
                      max(y for _, y in r.polygon)) for r in city.regions}
    assert boxes == {"high street": (0.0, 0.0, 100.0, 27.0), "west street": (15.0, 27.0, 35.0, 72.0),
                     "east street": (65.0, 27.0, 85.0, 72.0), "square": (35.0, -24.0, 65.0, 0.0)}
    for r in city.regions:
        x0, y0, x1, y1 = boxes[r.name]
        assert sorted(r.polygon) == sorted([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), r.id  # a rectangle
        assert (r.level, r.z0, r.z1) == (0, 0.0, 45.0), r.id
    assert sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes.values()) == pytest.approx(5220.0)
    assert {r.name: r.anchor for r in city.regions} == {
        "high street": (50.0, 13.5), "west street": (25.0, 39.0), "east street": (75.0, 39.0), "square": (50.0, -12.0)}
    assert city.bounds == (-0.6, -24.6, 100.6, 72.6)
    assert [city.region_of(p) for p in ((25, 26.9), (25, 27.1), (75, 27.1), (50, -0.1), (50, 0.1), (25, -5), (10, 50))
            ] == ["R1", "R2", "R3", "R4", "R1", None, None]  # the last two stand in a building or beside the square


def test_the_street_ends_are_open_and_the_square_is_closed_behind_by_a_facade(city):
    """Contract (plan-city §1, "Open street ends"): the high street's two ends and the side streets' north ends are
    region edges with no wall (the renderer continues the city beyond them as scenery), while the square's south end is
    a facade. A line from each street's anchor out through its end crosses no wall piece for the first three kinds and
    one for the square. Regression caught: a wall across a street end (a closed courtyard: the scenery and the
    open sightlines the look-dev rendered are gone), or the square left open to the south (nothing to stop the
    camera's view of a boundless plane)."""
    pieces = city.wall_pieces()

    def blocked(start, end) -> bool:
        n = 400
        pts = [(start[0] + (end[0] - start[0]) * i / n, start[1] + (end[1] - start[1]) * i / n) for i in range(n + 1)]
        return any(point_in_rect(r, x, y) for r in pieces for x, y in pts)

    assert not blocked((50.0, 13.5), (-5.0, 13.5)) and not blocked((50.0, 13.5), (105.0, 13.5))
    assert not blocked((25.0, 39.0), (25.0, 80.0)) and not blocked((75.0, 39.0), (75.0, 80.0))
    assert blocked((50.0, -12.0), (50.0, -30.0))


def test_the_square_keeps_five_pads(city):
    """Contract (plan-city §1, "Fixed furniture"; M3-maps §2): 28 fixed pieces: 8 sidewalk strips (grey), 10 parking
    lanes (white) with 22 bays between them, 2 loading zones (yellow), the square's paving in two pieces (beige, F21 and
    F24) round the nest's bare yard with 5 pads (3 and 2), the nest and 5 drop zones. Surfaces by hand: bays 2+3+2+3 on
    the high street and 1+1+4 and 4+1+1 on the side streets = 22, 2 loading zones, 8 sidewalk surfaces, 5 pads = 37; the
    nest and the drop zones have none. Regression caught: a lane or a pad lost or moved unnoticed (the 22 bays hold
    14.4 things at the start, which the yields were measured on), a loading zone dropped (box trucks have nowhere to
    stand), a piece left without its surface (a receptacle nobody can place on), or a surface on a nest or a drop zone
    (the world would draw items onto the marked ground)."""
    assert len(city.furniture) == 28 and {f.mobility for f in city.furniture} == {"fixed"}
    kinds = ("sidewalk", "parking_lane", "loading_zone", "plaza", "nest", "drop_zone")
    assert {k: len(of_kind(city, k)) for k in kinds} == {
        "sidewalk": 8, "parking_lane": 10, "loading_zone": 2, "plaza": 2, "nest": 1, "drop_zone": 5}
    assert surface_kinds(city) == {"sidewalk": 8, "parking": 22, "loading zone": 2, "plaza": 5}
    assert sorted(len(f.surfaces) for f in of_kind(city, "parking_lane")) == [1, 1, 1, 1, 2, 2, 3, 3, 4, 4]
    assert sorted(len(f.surfaces) for f in of_kind(city, "plaza")) == [2, 3]
    assert {k: {f.color for f in of_kind(city, k)} for k in kinds[:4]} == {
        "sidewalk": {"grey"}, "parking_lane": {"white"}, "loading_zone": {"yellow"}, "plaza": {"beige"}}
    assert sum(len(f.surfaces) for f in city.furniture) == 37
    assert [s for f in city.furniture if f.kind in ("nest", "drop_zone") for s in f.surfaces] == []
    pads = sorted((round(r.cx, 3), round(r.cy, 3)) for f in of_kind(city, "plaza") for s in f.surfaces
                  for r in [surface_world_rect(f.center, f.yaw, s)])
    assert pads == [(40.0, -6.0), (50.0, -18.0), (50.0, -6.0), (60.0, -18.0), (60.0, -6.0)]  # (40, -18) is the yard's


def test_sidewalks_and_the_paving_block_and_painted_lanes_do_not(city):
    """Contract (plan-city §1, "Sidewalks block, painted lanes do not"): a sidewalk (4 m x 0.15 m high) and the square's
    paving block the ego's grid and the movers' (so every ground mover stays on the roadway with no new code, and the
    drone flies over them), while a parking lane or loading zone (2.6 m wide, 1 cm high) is a floor marking that blocks
    nothing: a mover may pull into an empty bay; so are the nest and the drop zones, which are marked ground and block
    nothing. Hand count: 8 sidewalks + the two pieces of paving = 10 blocking footprints. Regression caught: a lane
    that blocks (the bus could never use the high street between parked rows), a sidewalk that does not (a taxi cuts
    across the kerb), or a drop zone or nest that does (a zone no robot may hover over)."""
    assert {f.kind: f.blocks for f in city.furniture} == {
        "sidewalk": True, "parking_lane": False, "loading_zone": False, "plaza": True, "nest": False,
        "drop_zone": False}
    assert len(blocking_rects(city)) == 10
    assert {(f.size[0], f.size[2]) for f in of_kind(city, "sidewalk")} == {(4.0, 0.15)}
    assert {(f.size[0], f.size[2]) for k in ("parking_lane", "loading_zone") for f in of_kind(city, k)} == {(2.6, 0.01)}
    assert {f.size[2] for f in of_kind(city, "plaza")} == {0.15}


def test_a_bay_is_seven_metres_of_lane_and_a_loading_zone_fifteen(city):
    """Contract (plan-city §1, "Parking"): a bay is a 2.5 x 6.7 m surface in a 2.6 m lane, bays 7 m apart (the lane is
    7 m times its bays long), so a 2.4 m box truck fits a loading zone's 2.5 m width and a 4.6 m car a bay; a loading
    zone is one 2.5 x 14.8 m surface in a 15 m piece, two on the side streets. Hand values: F14, four bays at +-3.5 and
    +-10.5 along its 28 m. Regression caught: bays that overlap or leave gaps (items placed across a painted line),
    surface sizes that drift from the lane."""
    for f in of_kind(city, "parking_lane"):
        assert f.size[0] == 2.6 and f.size[1] == pytest.approx(7.0 * len(f.surfaces)), f.id
        along = sorted(s.rect[1] for s in f.surfaces)
        assert along == pytest.approx([7.0 * (k - (len(along) - 1) / 2) for k in range(len(along))]), f.id
        assert {(s.rect[0], s.rect[2:], s.z) for s in f.surfaces} == {(0.0, (2.5, 6.7), 0.0)}, f.id
    for f in of_kind(city, "loading_zone"):
        assert f.size[:2] == (2.6, 15.0) and [(s.rect, s.z) for s in f.surfaces] == [((0.0, 0.0, 2.5, 14.8), 0.0)]
    assert sorted(f.id for f in of_kind(city, "loading_zone")) == ["F12", "F19"]
    f14 = next(f for f in city.furniture if f.id == "F14")
    assert [s.rect[1] for s in f14.surfaces] == [10.5, 3.5, -3.5, -10.5]


def test_every_sidewalk_surface_starts_one_metre_in_from_its_kerb_and_the_kerb_faces_the_roadway(city):
    """Contract (plan-city §1, "The sidewalk surface is the furniture zone"; lane B's scenery strip): a sidewalk is 4 m
    deep with its front towards the roadway, and its one surface is 1.6 m deep at local x 0.2, so it spans local x
    -0.6 to 1.0 and ends 2.0 - 1.0 = 1.0 m short of the kerb (lamp posts, signal poles and hydrants stand in that metre,
    never on a surface); it is 1 m shorter than the strip and at the piece's height (0.15 m). The point 0.5 m beyond
    the kerb is on the roadway: inside a street and under no blocking piece, and 0.5 m behind it is on the sidewalk.
    Regression caught: a surface that reaches the kerb (a lamp post on an item), a sidewalk turned the wrong way (items
    in front of the facade, scenery in the street), or a surface that floats off the sidewalk."""
    walks = of_kind(city, "sidewalk")
    assert len(walks) == 8
    blockers = [f.footprint() for f in city.furniture if f.blocks]
    for f in walks:
        (s,) = f.surfaces
        cx, cy, sx, sy = s.rect
        assert f.size[0] == 4.0 and (sx, cx) == (1.6, 0.2) and f.size[0] / 2 - (cx + sx / 2) == pytest.approx(1.0), f.id
        assert sy == pytest.approx(f.size[1] - 1.0) and cy == 0.0 and s.z == f.size[2] == 0.15, f.id
        r = surface_world_rect(f.center, f.yaw, s)  # the surface sits 0.2 m from the piece's middle, towards the kerb
        towards_kerb = (r.cx - f.center[0]) * math.cos(f.yaw) + (r.cy - f.center[1]) * math.sin(f.yaw)
        assert towards_kerb == pytest.approx(0.2), f.id
        front = (f.center[0] + 2.5 * math.cos(f.yaw), f.center[1] + 2.5 * math.sin(f.yaw))
        behind = (f.center[0] + 1.5 * math.cos(f.yaw), f.center[1] + 1.5 * math.sin(f.yaw))
        assert city.region_of(front) is not None and not any(point_in_rect(b, *front) for b in blockers), f.id
        assert point_in_rect(f.footprint(), *behind), f.id


def test_the_roadways_are_nineteen_and_twelve_metres_between_their_sidewalks(city):
    """Contract (plan-city §1, "Street widths"): the high street's carriageway between its sidewalks is 19 m (y 4 to
    23: two parking lanes and four travel lanes), a side street's 12 m (two 2.6 m lanes and two 3.4 m travel lanes), and
    on the high street the bus has 20.4 - 6.6 = 13.8 m between its parked rows, over the 9.5 m it needs. Regression
    caught: a high street too narrow for the bus (the T3 walks fail), or a side street that is no 20 m right of way."""
    kerbs = {f.id: f for f in city.furniture}
    south, north = kerbs["F05"], kerbs["F01"]
    assert (south.center[1] + 2.0, north.center[1] - 2.0) == (4.0, 23.0)  # kerb to kerb: 19 m
    lanes = sorted((f.center[1] - 1.3, f.center[1] + 1.3) for f in (kerbs["F06"], kerbs["F03"]))
    assert [edge for lane in lanes for edge in lane] == pytest.approx([4.0, 6.6, 20.4, 23.0])
    assert lanes[1][0] - lanes[0][1] >= 9.5
    west, east = kerbs["F09"], kerbs["F10"]
    assert (west.center[0] + 2.0, east.center[0] - 2.0) == (19.0, 31.0)  # kerb to kerb: 12 m


def test_a_vehicle_fits_a_bay_and_neither_a_sidewalk_surface_nor_a_pad(city):
    """Contract (plan-city §3.1, "Where things sit"; §1, "The square's pads"): surface sizes keep vehicles where they
    belong. By hand, as a box on a rectangle either way round, with the catalog's boxes (car 4.6 x 1.8, van 5.2 x 2.0,
    pickup 5.4 x 2.0, box truck 7.3 x 2.4, skip 3.0 x 1.7) against a bay 6.7 x 2.5, a loading zone 14.8 x 2.5, a
    sidewalk surface 1.6 x 18-99 and a pad 3.5 x 3.5: cars, vans and pickups fit the bays and the loading zones; the box
    truck (7.3 > 6.7) only a loading zone; none of the four fits a sidewalk (every box is over 1.6 m wide) or a pad
    (over 3.5 m long); the skip (1.7 > 1.6) fits no sidewalk but a pad (3.0 <= 3.5) and a bay; a dumpster (1.8 x 1.3),
    portable toilet (1.1 x 1.1) and food cart (2.0 x 1.0) fit all four. Regression caught: a pad enlarged until a van
    parks on the square, a bay shortened under a pickup, or a sidewalk widened to a car (parked on the pavement)."""
    sizes: dict[str, set[tuple[float, float]]] = {}
    for f in city.furniture:
        for s in f.surfaces:
            sizes.setdefault(s.kind, set()).add(tuple(sorted(s.rect[2:], reverse=True)))  # (long side, short side)
    assert sizes["parking"] == {(6.7, 2.5)} and sizes["loading zone"] == {(14.8, 2.5)}
    assert sizes["plaza"] == {(3.5, 3.5)} and {s[1] for s in sizes["sidewalk"]} == {1.6}

    def fits(kind: str, surface: str) -> bool:
        a, b = sorted(DOWNTOWN_KINDS[kind].size[:2], reverse=True)
        return any(a <= long_ and b <= short for long_, short in sizes[surface])

    for kind in ("car", "van", "pickup"):
        assert {s for s in sizes if fits(kind, s)} == {"parking", "loading zone"}, kind
    assert {s for s in sizes if fits("box_truck", s)} == {"loading zone"}
    assert {s for s in sizes if fits("skip", s)} == {"parking", "loading zone", "plaza"}
    for kind in ("dumpster", "portable_toilet", "food_cart"):
        assert {s for s in sizes if fits(kind, s)} == set(sizes), kind
    for kind, k in DOWNTOWN_KINDS.items():  # and each kind's receptacle priors name surfaces the map has
        assert set(k.receptacles) <= set(sizes), kind


def test_every_wall_is_a_facade_taller_than_the_drone_flies(city):
    """Contract (plan-city §1, "Walls", "Buildings"): twelve exterior walls 0.6 m thick, each with a facade style, in
    six buildings (a block is a hollow box of facade walls: the walls of one building share their height and style, and
    no two buildings a style); a height is 4.5 m plus 3.6 m storeys, 18.9 to 44.1 m, all above the 12 m the drone
    flies, so no roof ever shows. Hand values: 22.5 = 4.5 + 5 x 3.6, 33.3 (8), 18.9 (4), 44.1 (11), 29.7 (7), 26.1
    (6); walls W01-W02, W03-W05, W06-W07, W08-W09, W10-W11 and W12. Regression caught: a building lower than the drone
    (it would fly through the facade, or see a roof), two walls of one block in two styles, or an interior wall."""
    walls = {w.id: w for w in city.walls}
    assert len(walls) == 12 and all(w.exterior and w.thickness == 0.6 and w.facade for w in walls.values())
    buildings = [("W01", "W02"), ("W03", "W04", "W05"), ("W06", "W07"), ("W08", "W09"), ("W10", "W11"), ("W12",)]
    heights = [22.5, 33.3, 18.9, 44.1, 29.7, 26.1]
    styles = []
    for ids, h in zip(buildings, heights, strict=True):
        assert {walls[i].height for i in ids} == {h} and len({walls[i].facade for i in ids}) == 1, ids
        styles.append(walls[ids[0]].facade)
    assert len(set(styles)) == 6
    assert min(w.height for w in walls.values()) == 18.9 > 12.0  # the drone flies at 12 m
    assert all(round((h - 4.5) / 3.6) * 3.6 + 4.5 == pytest.approx(h) for h in heights)


def test_the_manifest_holds_the_citys_ten_procedural_models_and_four_floors():
    """Contract (plan-city §7.3, step 0): the eight carried kinds and the taxi and the bus each have one procedural
    model of the right mobility, built to the catalog's box (fit `exact`), every material a `shader:` name (nothing is
    downloaded) and tinted by the colour attribute on its body (the cart's umbrella: it is the colour); the four regions
    have floors, `shader:asphalt` for the streets and `shader:paving` for the square; and the whole manifest passes its
    own validation, which asks for exactly one model per queried kind. Regression caught: a city kind with no model
    (the first assets render fails), a model the colour does not reach (every colour renders alike), or an asset that
    needs a download."""
    m = A.packaged()
    wanted = {**{k: ("carried", v.size) for k, v in DOWNTOWN_KINDS.items()},
              **{k: ("mover", v.size) for k, v in DOWNTOWN_MOVER_KINDS.items()}}
    assert len(wanted) == 10
    for kind, (mobility, size) in wanted.items():
        (a,) = [a for a in m.assets if a.role == "model" and a.kind == kind]
        assert (a.source, a.mobility, a.fit["mode"], a.builder) == (A.PROCEDURAL, mobility, "exact", kind), kind
        assert all(abs(have - want) <= 0.001 for have, want in zip(a.native_bbox_m, size, strict=True)), kind
        assert all(v.startswith("shader:") for v in a.materials.values()) and not a.files, kind
        assert a.tint["mode"] == "luminance" and a.tint["materials"], kind
        assert set(a.tint["materials"]) <= set(a.materials), kind
        assert a.tint["materials"] == (["umbrella", "umbrella2"] if kind == "food_cart" else ["body"]), kind
    assert {m.floors[r.name] for r in load_map("downtown_a").regions if r.name != "square"} == {"shader:asphalt"}
    assert m.floors["square"] == "shader:paving"
    carried, light, movers = A.queried_kinds()
    assert A.validate(m, carried, light, movers, colours=A.queried_colours()) == []


# -- M3-maps: the drone's nest, its yard and the five drop zones (M3-maps §2) ----------------------
DRONE_REACH_M = 3.65  # the drone's body radius (1.30 m across / 2 = 0.65) plus its clearance (3.0 m): nothing is nearer


def test_the_nest_stands_on_bare_ground(city):
    """Contract (M3-maps §2; M3-Y §1): one `nest`, a fixed, non-blocking, grey 1.6 x 1.6 x 0.02 m pad with no surface at
    (40.0, -18.5) in the square's south-west corner, on a bare yard (x 35.005 to 45.005, y -23.995 to -12.995) that
    no piece covers: no other fixed piece holds its centre or meets the yard, and every wall face, every other fixed
    piece and every walk-surface piece (`sidewalk`, `plaza`) is at least 3.65 m from its centre, the drone's 0.65 m
    radius plus its 3.0 m clearance (`ROBOTS["drone"].body`). Hand values: the facade W09's face is at x 35.0, 5.0 m
    away; W12's inner face at y -24.0, 5.5 m; the paving F24's west edge x 45.005, 5.005 m; F21's south edge y
    -12.995, 5.505 m. Regression caught: a drone that cannot land without breaking the body rule, or a piece of paving,
    a pad or a drop zone put back on the yard."""
    body = ROBOTS["drone"].body
    assert body.length_m / 2 + body.clearance_m == pytest.approx(DRONE_REACH_M)
    (nest,) = of_kind(city, "nest")
    assert (nest.id, nest.mobility, nest.blocks, nest.color, nest.size, nest.surfaces) == (
        "F25", "fixed", False, "grey", (1.6, 1.6, 0.02), ())
    assert nest.center == (40.0, -18.5) and city.region_of(nest.center) == "R4"
    yard = Rect(40.005, -18.495, 10.0, 11.0, 0.0)  # x 35.005 to 45.005, y -23.995 to -12.995
    others = [f for f in city.furniture if f is not nest]
    assert [f.id for f in others if point_in_rect(f.footprint(), *nest.center)] == []
    assert [f.id for f in others if rects_overlap(f.footprint(), yard)] == []
    assert rect_inside_rect(nest.footprint(), yard)
    assert city.openings == ()  # a wall is one piece, in the walls' order
    faces = {w.id: nearest_on_rect(r, *nest.center)[1] for w, r in zip(city.walls, city.wall_pieces(), strict=True)}
    assert (faces["W09"], faces["W12"]) == pytest.approx((5.0, 5.5)) and min(faces.values()) >= DRONE_REACH_M
    near = {f.id: nearest_on_rect(f.footprint(), *nest.center)[1] for f in others}
    assert min(near.values()) >= DRONE_REACH_M
    assert near["F24"] == pytest.approx(5.005) and near["F21"] == pytest.approx(5.505)


def test_drop_zones_lie_on_their_ground(city):
    """Contract (M3-maps §2; M3-zones-C §1): five `drop_zone` pieces, fixed, non-blocking, white, with no surface (no
    item is ever drawn onto one by the world's placement), named by the address a brief names, each a marking 0.005 m
    thick on the ground under it, so its height is that ground's plus 0.005 within 1e-6 m (the ground: the top of
    the tallest other fixed piece of at most 0.2 m whose footprint holds its centre), with the yaw of that piece and
    wholly inside it; each centre is at least 3.65 m from every wall face (the drone hovers over it 7 m up and keeps
    that from walls) and lies in a region. Hand values: the square's three zones stand on the paving's 0.15 m (0.155);
    the street zones on the loading zones' 0.01 m (0.015). Regression caught: a marking drawn inside the paving (its
    height from the street's level), a zone no drone may hover over, or a zone outside its piece."""
    zones = {f.id: f for f in of_kind(city, "drop_zone")}
    expected = {  # id: (address, centre, size, ground)
        "F26": ("Square West", (40.0, -6.0), (3.0, 3.0, 0.155), 0.15),
        "F27": ("Square East", (60.0, -6.0), (3.0, 3.0, 0.155), 0.15),
        "F28": ("Square South", (60.0, -18.0), (3.0, 3.0, 0.155), 0.15),
        "F29": ("West Street", (20.3, 44.1), (2.6, 3.0, 0.015), 0.01),
        "F30": ("East Street", (79.7, 44.1), (2.6, 3.0, 0.015), 0.01)}
    assert sorted(zones) == sorted(expected)
    for zid, (address, centre, size, ground) in expected.items():
        z = zones[zid]
        assert (z.name, z.center, z.size) == (address, centre, size), zid
        assert (z.mobility, z.blocks, z.color, z.surfaces) == ("fixed", False, "white", ()), zid
        under = [g for g in city.furniture if g is not z and g.size[2] <= 0.2 and point_in_rect(g.footprint(), *centre)]
        holder = max(under, key=lambda g: g.size[2])
        assert holder.size[2] == pytest.approx(ground) and z.size[2] == pytest.approx(ground + 0.005, abs=1e-6), zid
        assert z.yaw == pytest.approx(holder.yaw) and rect_inside_rect(z.footprint(), holder.footprint()), zid
        assert min(nearest_on_rect(r, *centre)[1] for r in city.wall_pieces()) >= DRONE_REACH_M, zid
        assert city.region_of(centre) is not None and city.rect_inside(z.footprint()), zid
