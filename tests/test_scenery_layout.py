"""The scenery beyond the downtown's edge, as geometry (`render/scenery_layout.py`, plain Python: no Blender needed).

The renderer continues the city past the high street's two ends and the side streets' north ends as static scenery that
is never an entity, with street furniture in the first metre from each kerb (lamp posts, signal poles, hydrants and,
along the continued sidewalks, street trees) and the centre line and lane dashes of each street carried on as paint.
What that layout may and may not touch is checked here against the map's own surfaces, regions, walls and kerb strips,
which are computed from the map file independently of the layout's own helpers."""

from __future__ import annotations

import math
from functools import cache

import pytest

from conftest import apartment, warehouse
from dressing_kit import render_module
from stbench.geometry import Rect, corners, rect_inside_rect, rects_overlap, to_world
from stbench.mapspec import MapSpec, load_map, surface_world_rect

KERB_M = 1.0  # plan-city section 1: a sidewalk's surface starts a metre in from its kerb


@cache
def downtown() -> MapSpec:
    return load_map("downtown_a")


@pytest.fixture(scope="module")
def L():
    """`render/scenery_layout.py`, which has no bpy, imported by name with render/ on the path (it is no package, and
    its street paint is `street_marks.py`, a sibling)."""
    return render_module("scenery_layout")


@pytest.fixture(scope="module")
def lay(L):
    return L.layout(downtown())


def region_rects(m) -> list[Rect]:
    """Each region (a rectangle, from the map's polygon) as a Rect."""
    out = []
    for r in m.regions:
        xs, ys = [p[0] for p in r.polygon], [p[1] for p in r.polygon]
        out.append(Rect((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys), 0.0))
    return out


def kerb_strips(m) -> list[Rect]:
    """The first metre from the kerb of every sidewalk of the map, from its pose: the piece's front (local +x) is its
    kerb side, so the strip is its last metre of depth."""
    out = []
    for f in m.furniture:
        if f.kind == "sidewalk":
            frame = Rect(f.center[0], f.center[1], 0.0, 0.0, f.yaw)
            cx, cy = to_world(frame, f.size[0] / 2 - KERB_M / 2, 0.0)
            out.append(Rect(cx, cy, KERB_M, f.size[1], f.yaw))
    return out


def test_scenery_continues_exactly_the_four_street_ends_the_map_leaves_open(L):
    """Contract (plan-city section 1): the high street's two ends and the side streets' north ends are open region
    edges, so the city continues there and nowhere else: each continued street faces away from the map. The edges a wall
    covers, the square's mouth and the streets' shared edges are not street ends. Regression caught: scenery on a walled
    side (a street running through a building), or a street end left as a view of nothing."""
    arms = L.open_edges(downtown())
    found = sorted((round(a.pt(0.0, a.width / 2)[0], 1), round(a.pt(0.0, a.width / 2)[1], 1), round(a.n[0]),
                    round(a.n[1]), round(a.width, 1)) for a in arms)
    assert found == [(0.0, 13.5, -1, 0, 27.0), (25.0, 72.0, 0, 1, 20.0), (75.0, 72.0, 0, 1, 20.0),
                     (100.0, 13.5, 1, 0, 27.0)]


def test_no_scenery_lies_on_a_surface_or_inside_a_region_beyond_a_kerb_strip(lay):
    """Contract (plan-city section 7.3, lane B): the scenery is outside the map: no footprint (a road, a sidewalk, a
    marked bay, a building, a post) overlaps a receptacle surface, and none overlaps the inside of a region except a
    lamp post, signal pole or hydrant standing wholly in the kerb strip of one of the map's sidewalks (street trees
    stand beyond the map only). Regression caught: street furniture on a surface (a hydrant in a dumpster's place: a
    carried thing placed inside it), scenery drawn inside the streets, or a road laid over the map."""
    m = downtown()
    surfaces = [surface_world_rect(f.center, f.yaw, s) for f in m.furniture for s in f.surfaces]
    regions, strips = region_rects(m), kerb_strips(m)
    posts = {"lamp", "signal", "hydrant", "tree"}
    inside = 0
    for kind, rect in lay.footprints():
        assert not any(rects_overlap(rect, s) for s in surfaces), (kind, rect)
        if not any(rects_overlap(rect, r) for r in regions):
            continue
        assert kind in posts, f"a {kind} overlaps a region: {rect}"
        assert any(rect_inside_rect(rect, s, tol=1e-6) for s in strips), f"a {kind} in a region, not in a kerb strip"
        inside += 1
    assert inside >= 19  # the map's own sidewalks carry lamp posts and signal poles (20 but for the West Street zone's)


def kerb_band(walk) -> Rect:
    """The first metre from the kerb of a scenery sidewalk slab (axis aligned; its kerb faces along `front`)."""
    r, (fx, fy) = walk.rect, (round(math.cos(walk.front)), round(math.sin(walk.front)))
    x0, x1, y0, y1 = r.cx - r.sx / 2, r.cx + r.sx / 2, r.cy - r.sy / 2, r.cy + r.sy / 2
    if fx:
        x0, x1 = (x1 - KERB_M, x1) if fx > 0 else (x0, x0 + KERB_M)
    else:
        y0, y1 = (y1 - KERB_M, y1) if fy > 0 else (y0, y0 + KERB_M)
    return Rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0)


def test_scenery_posts_beyond_the_map_stand_in_the_first_metre_of_a_scenery_sidewalk(lay):
    """Contract: furniture of the scenery's own sidewalks obeys the same kerb-strip rule, so a street's furniture stands
    in the same place whichever side of the edge it is on. Regression caught: a lamp post in the road or against a
    building beyond the edge."""
    regions = region_rects(downtown())
    outside = [p for p in lay.posts if not any(rects_overlap(p.rect, r) for r in regions)]
    assert len(outside) > 40
    bands = [kerb_band(w) for w in lay.walks]
    for p in outside:
        assert any(rect_inside_rect(p.rect, b, tol=1e-6) for b in bands), (p.kind, p.x, p.y)


def test_street_trees_stand_along_the_continued_sidewalks_only_every_seventeen_metres_or_so_clear_of_the_posts(lay):
    """Contract (realism Phase 1: street trees in the kerb strip; the map's own trees are the dressing's, validated
    against the map's zones, so the scenery keeps off the map's sidewalks): the 8 continued sidewalks (250 m each: the
    high street's two ends on both sides, the four side street ends) carry about one tree in 17 m (14 candidates a
    sidewalk, those within 3.5 m of a lamp, signal or hydrant dropped): between 60 and 90 trees in all, each standing in
    the kerb strip of a scenery sidewalk (the first metre from its kerb, the trunk's 0.8 m box wholly inside) and none
    in a region of the map; no two within 3.5 m of each other or of another post. Regression caught: trees in the map's
    own kerb strips (twice as many as the dressing places), in a carriageway or against a lamp post, or none at all
    beyond the edge (a bare end of a tree-lined street)."""
    regions = region_rects(downtown())
    trees = [p for p in lay.posts if p.kind == "tree"]
    assert 60 <= len(trees) <= 90
    bands = [kerb_band(w) for w in lay.walks]
    others = [p for p in lay.posts if p.kind != "tree"]
    for t in trees:
        assert any(rect_inside_rect(t.rect, b, tol=1e-6) for b in bands), (t.x, t.y)
        assert not any(rects_overlap(t.rect, r) for r in regions), (t.x, t.y)
        assert all(math.dist((t.x, t.y), (q.x, q.y)) > 3.5 for q in others), (t.x, t.y)
    assert all(math.dist((a.x, a.y), (b.x, b.y)) > 8.0 for i, a in enumerate(trees) for b in trees[i + 1:])


def test_a_street_end_carries_its_centre_line_and_lane_dashes_on_as_paint_that_is_no_footprint(L, lay):
    """Contract (the continued streets keep their paint: the high street's double yellow centre line on y = 13.5 and its
    two white lane dashes on y = 13.5 -+ 3.45, the side streets' on x = 25 and 75, which carry on from the map's own
    over the 250 m of scenery, from x = 100 east, x = 0 west and y = 72 north): 8 marks, a centre line and two dashes at
    each end of the high street and a centre line at the north end of each side street, the east ones from x 100 to 350
    and the west from -250 to 0, the north ones from y 72 to 322. Paint is flat, so it is no footprint: the layout's
    `footprints()` do not list it and the bare carriageway test above still holds. Regression caught: a street whose
    lines stop at the map's edge, lines carried on at another offset (a seam), or paint listed as an object."""
    got = sorted((m.style, tuple(round(v, 3) for v in m.rect)) for m in lay.lines)
    assert got == sorted([
        ("double", (100.0, 350.0, 13.3, 13.7)), ("dash", (100.0, 350.0, 9.975, 10.125)),
        ("dash", (100.0, 350.0, 16.875, 17.025)), ("double", (-250.0, 0.0, 13.3, 13.7)),
        ("dash", (-250.0, 0.0, 9.975, 10.125)), ("dash", (-250.0, 0.0, 16.875, 17.025)),
        ("double", (24.8, 25.2, 72.0, 322.0)), ("double", (74.8, 75.2, 72.0, 322.0))])
    assert L.layout(apartment()).lines == [] and L.layout(warehouse()).lines == []
    assert not any(kind == "line" for kind, _ in lay.footprints())


def test_no_building_stands_in_a_wall_or_in_another_building(lay):
    """Contract: the scenery's buildings begin past the end of the map's wall slabs and never overlap each other (the
    two rows between the north streets meet in the middle of their block instead). Regression caught: a lot overlapping
    the map's own wall (two facades in one plane, a flicker), or rows of the two north arms drawn through each other."""
    m = downtown()
    walls = [Rect(*w.point_at(w.length / 2), w.length + w.thickness, w.thickness, w.yaw) for w in m.walls]
    buildings = [b.rect for b in lay.buildings]
    assert not any(rects_overlap(b, w) for b in buildings for w in walls)
    assert not any(rects_overlap(a, b) for i, a in enumerate(buildings) for b in buildings[i + 1:])


def test_each_row_continues_the_building_beside_it(lay):
    """Contract (plan-city section 1): past an edge a street's two rows of buildings carry on from the map's own walls:
    the first lot of each row has the style and height of the wall it continues. Hand-read from the plan's wall table
    (W06 stone_grey 18.9 m, W10 concrete_white 29.7, W01 brick_red 22.5, W08 glass_blue 44.1, W02 brick_red 22.5, W04
    and W05 brick_tan 33.3, W07 stone_grey 18.9). Regression caught: a building that changes its facade or height the
    moment it leaves the map, a seam in the middle of a block."""
    def first(near: float, axis: str, side: str, across: tuple[float, float]):
        """The lot whose `side` ('lo' or 'hi') edge on `axis` is at `near`, its centre within the `across` range."""
        hits = []
        for b in lay.buildings:
            r = b.rect
            x0, x1, y0, y1 = r.cx - r.sx / 2, r.cx + r.sx / 2, r.cy - r.sy / 2, r.cy + r.sy / 2
            lo, hi, mid = (x0, x1, r.cy) if axis == "x" else (y0, y1, r.cx)
            if abs((lo if side == "lo" else hi) - near) < 0.01 and across[0] < mid < across[1]:
                hits.append(b)
        assert len(hits) == 1, (near, axis, side, len(hits))
        return hits[0].style, hits[0].height

    assert first(100.3, "x", "lo", (27, 60)) == ("stone_grey", 18.9)  # beyond the east end, north of the street: W06
    assert first(100.3, "x", "lo", (-30, 0)) == ("concrete_white", 29.7)  # south of it: W10
    assert first(-0.3, "x", "hi", (27, 60)) == ("brick_red", 22.5)  # beyond the west end, north: W01
    assert first(-0.3, "x", "hi", (-30, 0)) == ("glass_blue", 44.1)  # south: W08
    assert first(72.3, "y", "lo", (-10, 15)) == ("brick_red", 22.5)  # north of the west street, its west side: W02
    assert first(72.3, "y", "lo", (35, 50)) == ("brick_tan", 33.3)  # its east side: W04
    assert first(72.3, "y", "lo", (50, 65)) == ("brick_tan", 33.3)  # the east street's west side: W05
    assert first(72.3, "y", "lo", (85, 110)) == ("stone_grey", 18.9)  # its east side: W07


def test_scenery_reaches_about_250_metres_and_no_further(lay):
    """Contract (plan-city section 5.4): the scenery stays within about 250 m of the map so that the farthest depth
    stays well under the 655 m a centimetre depth image holds, and reaches far enough that a ray 2.8 degrees below the
    horizon from 12 m up (it meets the ground 245 m out) still meets city. Regression caught: a street run out to 400 m
    (depth past the 16-bit range, which `locate` refuses), or one that stops short and shows the end of the world."""
    m = downtown()
    xmin, ymin, xmax, ymax = m.bounds
    far = 0.0
    for _, r in lay.footprints():
        for x, y in corners(r):
            far = max(far, xmin - x, x - xmax, ymin - y, y - ymax)
    assert 245.0 <= far <= 255.0, far


def test_the_layout_is_the_same_every_time_and_an_indoor_map_has_none(L, lay):
    """Contract: scenery is seeded by the layout alone (never by the episode), so every episode of a map sees the same
    city beyond its edge; an indoor map has no scenery. Regression caught: a layout drawn from unseeded randomness (the
    same map rendered with another city in each episode), or scenery added to the apartment."""
    again = L.layout(downtown())
    assert [(k, tuple(r)) for k, r in again.footprints()] == [(k, tuple(r)) for k, r in lay.footprints()]
    assert [b.style for b in again.buildings] == [b.style for b in lay.buildings]
    assert L.layout(apartment()).footprints() == [] and L.layout(warehouse()).footprints() == []


def test_bays_are_marked_on_from_the_lane_that_ends_near_the_edge(lay):
    """Contract: past the east end the south parking lane's bays carry on at its own pitch (7 m) and across width (2.5
    m, the lane's centre line y = 5.3), none starting inside the map. Regression caught: bays drawn inside the map
    (lines over the map's own lane), at another pitch (a seam), or off the lane's line."""
    east = sorted((r for r in lay.bays if r.cx > 100 and abs(r.cy - 5.3) < 0.01), key=lambda r: r.cx)
    assert len(east) > 30
    assert all(abs(r.sx - 6.7) < 1e-6 and abs(r.sy - 2.5) < 1e-6 for r in east)
    assert east[0].cx - 3.35 >= 100.0 + 6.0 - 1e-6  # unmarked for the first 6 m (see the carriageway test)
    assert all(abs((b.cx - a.cx) - 7.0) < 1e-6 for a, b in zip(east, east[1:], strict=False))


def test_the_carriageway_beyond_every_street_end_is_bare_of_objects_and_unmarked_for_six_metres(L, lay):
    """Contract (the movers' free grid keeps only a mover's centre inside a region, so a bus or taxi body reaches up to
    4.4 m past an open end): nothing the scenery draws as an object stands on the carriageway of a continued street, and
    nothing at all, not even a painted bay, lies on it within 6 m of the edge. The carriageway is the street's road
    between its sidewalks; lamp posts and hydrants stand on the sidewalks (kerb strips). Regression caught: a scenery
    car, bollard or building across the road where a bus may be standing, or bay lines painted under it."""
    arms = L.open_edges(downtown())
    assert len(arms) == 4
    walks = [w.rect for w in lay.walks]
    for arm in arms:
        carriageway = []  # the arm's road, cut into the stretches between its sidewalks, as (c0, c1)
        cuts = sorted(arm.span(w)[2:] for w in walks
                      if abs(arm.span(w)[0]) < 0.05 and arm.span(w)[2] > -0.05 and arm.span(w)[3] < arm.width + 0.05)
        at = 0.0
        for c0, c1 in cuts:
            if c0 > at + 1e-6:
                carriageway.append((at, c0))
            at = max(at, c1)
        if at < arm.width - 1e-6:
            carriageway.append((at, arm.width))
        assert sum(b - a for a, b in carriageway) >= 12.0, carriageway  # 19 m of the high street, 12 of a side street
        bare = [arm.rect(0.0, 6.0, a, b) for a, b in carriageway]
        whole = [arm.rect(0.0, 250.0, a, b) for a, b in carriageway]
        for kind, rect in lay.footprints():
            if kind in ("road", "walk"):
                continue
            assert not any(rects_overlap(rect, z) for z in bare), f"a {kind} on the first 6 m of a carriageway: {rect}"
            if kind != "bay":  # a painted bay is a marking; anything else is an object
                assert not any(rects_overlap(rect, z) for z in whole), f"a {kind} on a carriageway: {rect}"
