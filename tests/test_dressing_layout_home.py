"""What is particular to the household's Phase 1 dressing layout (`render/dressing_layout_home.py`): that it is a
function of the map, that its trim covers the walls, that fittings stand where a real apartment puts them and fixtures
follow the furniture they belong to. Where dressing may stand at all (receptacles, open floor, door swings, openings,
glass, the registry) is `test_dressing_zones.py`'s and lane V's; here only what that cannot say. Free: the map is data
and every number is worked out by hand from `apartment_a.json`."""

from __future__ import annotations

import copy
import itertools
import math
import random
from functools import cache

import pytest

from dressing_kit import layout_of, raw_map, render_module
from stbench.catalog import CATALOGS, KINDS, place_of
from stbench.geometry import corners
from stbench.mapspec import load_map

CEILING = 2.5
CASING_W = 0.07  # a casing is 7 cm wide
GRIME = ("dust_line", "hair", "crumbs", "scuff")  # M1's R.2: flush, on the floor, seeded from the map's name
# What each Poly Haven asset measures, in millimetres as `api.polyhaven.com/info/<id>` gives them (x width, y depth, z
# height): a piece's size is (depth, width, height), so (y, x, z).
POLY_HAVEN_MM = {
    "hanging_picture_frame_01": (594, 16, 841), "hanging_picture_frame_02": (752, 33, 500),
    "hanging_picture_frame_03": (385, 32, 502), "fancy_picture_frame_01": (603, 20, 464), "wall_clock": (320, 47, 320),
    "modern_ceiling_lamp_01": (432, 432, 952), "ceiling_fan": (1463, 1463, 516),
}


@cache
def _dressing() -> tuple:
    return tuple(layout_of("home", raw_map("apartment_a")))


@cache
def _map():
    return load_map("apartment_a")


def _of(cls: str) -> list:
    return [d for d in _dressing() if d.cls == cls]


def _fixed(kind: str) -> list:
    return [f for f in _map().furniture if f.kind == kind and f.mobility == "fixed"]


def _rect(furniture) -> tuple[float, float, float, float]:
    pts = corners(furniture.footprint())
    return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)


def _inside_footprint(box, furniture, eps=1e-6) -> bool:
    x0, y0, x1, y1 = _rect(furniture)
    return box[0] >= x0 - eps and box[1] >= y0 - eps and box[3] <= x1 + eps and box[4] <= y1 + eps


def _meet(a, b) -> bool:
    """Whether two (xmin, ymin, zmin, xmax, ymax, zmax) boxes share volume."""
    return all(a[i] < b[i + 3] - 1e-9 and b[i] < a[i + 3] - 1e-9 for i in range(3))


def _along(wall, x: float, y: float) -> float:
    """The distance of a map point from the wall's `a` end, along it."""
    ux, uy = wall.direction
    return (x - wall.a[0]) * ux + (y - wall.a[1]) * uy


def _across(wall, x: float, y: float) -> float:
    """The signed distance of a map point from the wall's centre line, positive on its left."""
    nx, ny = wall.normal
    return (x - wall.a[0]) * nx + (y - wall.a[1]) * ny


def test_the_layout_depends_on_the_map_alone_and_never_on_global_random_state():
    """Contract (dressing_spec: every episode of a map has the same dressing): `layout(spec)` is a pure function of the
    map spec, its few random choices (which picture hangs where, where the grime lies) seeded from the map's name.
    Called again on a copy it gives the same list, it leaves `random`'s global state alone, and a map of another name
    changes the pictures and the grime and nothing else. Regression caught: the global generator (a layout that
    differs by process or by which test seeded `random` last), or a choice that depends on anything but the name."""
    spec = raw_map("apartment_a")
    random.seed(11)
    before = random.getstate()
    first = layout_of("home", spec)
    assert random.getstate() == before
    assert layout_of("home", copy.deepcopy(spec)) == first == list(_dressing())
    renamed = layout_of("home", {**spec, "name": "apartment_z"})
    seeded = ("picture_frame", *GRIME)
    assert [d for d in renamed if d.cls not in seeded] == [d for d in first if d.cls not in seeded]
    assert [d for d in renamed if d.cls == "picture_frame"] != [d for d in first if d.cls == "picture_frame"]
    assert [d for d in renamed if d.cls in GRIME] != [d for d in first if d.cls in GRIME]


def _side(room, side: str) -> tuple[float, float, float, bool]:
    """(line, lo, hi, horizontal): where a side of a room (xmin, ymin, xmax, ymax) lies and how far it runs."""
    x0, y0, x1, y1 = room
    line = {"south": y0, "north": y1, "west": x0, "east": x1}[side]
    return (line, x0, x1, True) if side in ("south", "north") else (line, y0, y1, False)


def _baseboard_runs(room, side: str) -> list[tuple[float, float]]:
    """The stretches of one side of a room that baseboards cover, merged, read off the baseboard boxes."""
    line, lo, hi, horizontal = _side(room, side)
    spans = []
    for d in _of("baseboard"):
        b = d.box
        a0, a1, c0, c1 = (b[0], b[3], b[1], b[4]) if horizontal else (b[1], b[4], b[0], b[3])
        near = c0 if side in ("south", "west") else c1
        if abs(near - line) < 0.002 and c1 - c0 < 0.03 and a0 >= lo - 0.002 and a1 <= hi + 0.002:
            spans.append((a0, a1))
    merged: list[tuple[float, float]] = []
    for a0, a1 in sorted(spans):
        if merged and a0 <= merged[-1][1] + 0.002:
            merged[-1] = (merged[-1][0], max(a1, merged[-1][1]))
        else:
            merged.append((a0, a1))
    return merged


def _expected_runs(room, side: str) -> list[tuple[float, float]]:
    """By hand: a side of a room less its doorways (and 7 cm of casing each side) and less where fixed furniture stands
    flush against it, fragments under 10 cm dropped."""
    m = _map()
    line, lo, hi, horizontal = _side(room, side)
    holes = []
    for o in m.openings:
        across, mid = (o.center[1], o.center[0]) if horizontal else (o.center[0], o.center[1])
        if abs(across - line) <= m.wall_thickness / 2 + 1e-6 and lo - 1e-6 <= mid <= hi + 1e-6:
            holes.append((mid - o.width / 2 - CASING_W, mid + o.width / 2 + CASING_W))
    for f in m.furniture:
        if f.mobility == "fixed" and f.blocks:
            fx0, fy0, fx1, fy1 = _rect(f)
            near = {"south": fy0, "north": fy1, "west": fx0, "east": fx1}[side]
            a0, a1 = (fx0, fx1) if horizontal else (fy0, fy1)
            if abs(near - line) < 0.012 and a1 > lo and a0 < hi:
                holes.append((a0, a1))
    out, cursor = [], lo
    for h0, h1 in sorted(holes):
        if h0 - cursor >= 0.1:
            out.append((cursor, h0))
        cursor = max(cursor, h1)
    if hi - cursor >= 0.1:
        out.append((cursor, hi))
    return out


@pytest.mark.parametrize("region", [r.id for r in load_map("apartment_a").regions])
def test_baseboards_cover_every_wall_run_except_doorways_and_where_furniture_stands_against_it(region):
    """Contract (plan §4: 8 cm baseboards along every wall run except door openings): on each side of each room the
    baseboards cover exactly the wall less the doors and arches (stopping at their 7 cm casings) and less the places a
    fixed blocking piece stands flush against it (the counters, fridge, tub, ...), whose trim nobody sees. Worked out
    by hand per side from the openings and furniture of the map. Regression caught: a bare stretch of wall, trim run
    across a doorway or behind a counter, or a run that ends short of its corner."""
    r = next(r for r in _map().regions if r.id == region)
    xs, ys = [p[0] for p in r.polygon], [p[1] for p in r.polygon]
    room = (min(xs), min(ys), max(xs), max(ys))
    for side in ("south", "north", "west", "east"):
        got, want = _baseboard_runs(room, side), _expected_runs(room, side)
        assert len(got) == len(want), (region, side, got, want)
        for (g0, g1), (w0, w1) in zip(got, want, strict=True):
            assert abs(g0 - w0) < 0.01 and abs(g1 - w1) < 0.01, (region, side, got, want)
    assert all(d.size[0] == 0.012 and d.size[2] == 0.08 for d in _of("baseboard"))


def test_every_opening_has_a_jamb_each_side_and_a_head_on_each_face_that_looks_into_a_room():
    """Contract (plan §4: door casings around every door opening): each door and arch has, on each of its faces that
    looks into a room (the entrance has only the hall's), two 7 cm jambs standing from the floor to the top of the
    opening and a head over it, the jambs against the opening's edges, which is 3 pieces a face: 33 for the six
    openings. Regression caught: a casing missing on one side of a door, a head that starts below the opening (across
    the doorway), or jambs a hand's width off the jamb."""
    m = _map()
    pieces = _of("door_casing")
    assert len(pieces) == 33
    for o in m.openings:
        w = m.wall(o.wall)
        nx, ny = w.normal
        faces = 0
        for sign in (1, -1):
            reach = w.thickness / 2 + 0.1
            probe = (o.center[0] + sign * nx * reach, o.center[1] + sign * ny * reach)
            if m.region_of(probe) is None:
                continue
            faces += 1
            mine = [d for d in pieces if abs(_along(w, *d.pos[:2]) - o.at) < o.width / 2 + 0.12
                    and abs(_across(w, *d.pos[:2]) - sign * w.thickness / 2) < 0.02]
            jambs = sorted((d for d in mine if d.pos[2] == 0.0), key=lambda d: _along(w, *d.pos[:2]))
            heads = [d for d in mine if d.pos[2] > 0.0]
            assert len(jambs) == 2 and len(heads) == 1, (o.id, sign)
            assert abs(_along(w, *jambs[0].pos[:2]) - (o.at - o.width / 2 - CASING_W / 2)) < 0.006
            assert abs(_along(w, *jambs[1].pos[:2]) - (o.at + o.width / 2 + CASING_W / 2)) < 0.006
            assert all(abs(d.size[2] - o.height) < 1e-3 and abs(d.size[1] - CASING_W) < 1e-3 for d in jambs)
            assert heads[0].pos[2] >= o.height and abs(heads[0].size[1] - (o.width + 2 * CASING_W)) < 0.01
        assert faces == len([r for r in o.between if r != "outside"]), o.id


def _glass(win, wall) -> tuple[float, float, float, float, float, float]:
    """A window's own rectangle through the wall (no margin), as a box."""
    ax, ay = abs(wall.direction[0]), abs(wall.direction[1])
    hx, hy = (ax * win.width + ay * wall.thickness) / 2, (ay * win.width + ax * wall.thickness) / 2
    x, y = win.center
    return x - hx, y - hy, win.sill, x + hx, y + hy, win.sill + win.height


def test_each_window_is_framed_by_two_jambs_a_head_and_a_sill_and_nothing_stands_in_the_glass():
    """Contract (plan §4: window sills and trim; the registry's `window_sill`: under and beside the glass, never across
    it): every window has trim at both edges of its glass and over it, and a sill board under it, and no piece of any
    class overlaps the glass itself (the window's own rectangle through the wall). Regression caught: a curtain,
    blind, picture or casing across a pane (the render would show it over the sky and the depth would disagree), or a
    window with no trim on one side."""
    m = _map()
    sills = _of("window_sill")
    assert len(sills) == 4 * len(m.windows)
    for win in m.windows:
        w = m.wall(win.wall)
        glass = _glass(win, w)
        assert [d.cls for d in _dressing() if _meet(d.box, glass)] == [], win.id
        mine = [d for d in sills if math.dist(d.pos[:2], win.center) < win.width / 2 + 0.2 + w.thickness
                and win.sill - 0.05 <= d.pos[2] + d.size[2] / 2 <= win.sill + win.height + 0.15]
        assert len(mine) == 4, win.id
        jambs = sorted((d for d in mine if abs(d.size[2] - win.height) < 1e-3), key=lambda d: _along(w, *d.pos[:2]))
        assert [abs(_along(w, *d.pos[:2]) - win.at) for d in jambs] == pytest.approx(
            [win.width / 2 + CASING_W / 2] * 2, abs=0.006)
        assert any(abs(d.pos[2] - (win.sill + win.height)) < 0.01 for d in mine)  # the head, over the glass
        assert any(abs(d.pos[2] + d.size[2] - win.sill) < 0.01 and d.size[2] == 0.03 for d in mine)  # the sill board


def test_one_smoke_detector_in_each_room_on_the_ceiling_clear_of_the_lamps_and_walls():
    """Contract (plan §4: a smoke detector per room, on the ceiling): each of the five rooms holds exactly one, its top
    flush with the ceiling, at least 0.4 m from every wall and 0.3 m (in the plan) from every lamp, fan and pendant,
    which a detector must not be under. Regression caught: two in a room or none (the hall has two ceiling lights),
    one beside a wall (the dead air of a corner) or one hidden under a lamp."""
    m = _map()
    detectors = _of("smoke_detector")
    assert sorted(m.region_of(d.pos[:2]) for d in detectors) == sorted(r.id for r in m.regions)
    hung = [d for d in _dressing() if d.cls in ("ceiling_lamp", "ceiling_fan")]
    for d in detectors:
        assert abs(d.pos[2] + d.size[2] - CEILING) < 1e-6
        room = next(r for r in m.regions if r.id == m.region_of(d.pos[:2]))
        xs, ys = [p[0] for p in room.polygon], [p[1] for p in room.polygon]
        assert min(d.pos[0] - min(xs), max(xs) - d.pos[0], d.pos[1] - min(ys), max(ys) - d.pos[1]) >= 0.4
        for h in hung:
            gap_x = max(h.box[0] - d.box[3], d.box[0] - h.box[3], 0.0)
            gap_y = max(h.box[1] - d.box[4], d.box[1] - h.box[4], 0.0)
            clear = math.hypot(gap_x, gap_y)
            if h.cls == "ceiling_fan":  # its blades sweep a circle, not its box
                clear = math.dist(d.pos[:2], h.pos[:2]) - h.size[0] / 2
            assert clear >= 0.3, (d.pos, h.recipe)


def test_each_door_has_a_switch_at_1_1_m_on_its_latch_side_inside_the_room_it_opens_into():
    """Contract (plan §4: switches beside the doors at 1.1 m on the latch side): for each of the five doors a switch
    plate centred 1.1 m up stands inside the room the door swings into, on the wall of the door, beyond its casing on
    the side of the latch (not the hinge), within 0.4 m of the latch jamb; the hall's, with two lights, is a plate the
    wall has room for. Worked out from each door's hinge and swing in the map. Regression caught: a switch behind the
    open leaf (the hinge side), outside the room, across the doorway, or at the wrong height."""
    m = _map()
    switches = _of("light_switch")
    assert len(switches) == len(m.doors())
    for o in m.doors():
        w = m.wall(o.wall)
        hinge, latch = m.door_leaf(o, False)
        near = [s for s in switches if m.region_of(s.pos[:2]) == o.swings_into
                and abs(_along(w, *s.pos[:2]) - o.at) < o.width / 2 + 0.5]
        assert len(near) == 1, o.id
        s = near[0]
        assert abs(s.pos[2] + s.size[2] / 2 - 1.1) < 0.01
        assert math.dist(s.pos[:2], latch) < math.dist(s.pos[:2], hinge)
        assert CASING_W < abs(_along(w, *s.pos[:2]) - _along(w, *latch)) < 0.4


def _over_counters(box, homes) -> bool:
    """Whether a box lies inside the union of the footprints of the counters it runs over (a run of cabinets crosses
    from one counter to the next): each end of it is inside a footprint and no gap lies between."""
    rects = sorted(_rect(f) for f in homes)
    xs = sorted({box[0], box[3], *(v for r in rects for v in (r[0], r[2]) if box[0] < v < box[3])})
    ys = sorted({box[1], box[4], *(v for r in rects for v in (r[1], r[3]) if box[1] < v < box[4])})
    return all(any(r[0] <= (x0 + x1) / 2 <= r[2] and r[1] <= (y0 + y1) / 2 <= r[3] for r in rects)
               for x0, x1 in itertools.pairwise(xs) for y0, y1 in itertools.pairwise(ys))


def test_the_kitchen_fixtures_follow_the_counters_the_stove_and_the_fridge():
    """Contract (plan §4): upper cabinets from 1.45 m up, each over the footprint of counters, sink units or (the tall
    one) the fridge; a range hood over the stove, centred on it; the cabinets in modules of at most 0.62 m (the
    fridge's, 0.68, is the fridge's width) with the stove and the window left clear. By hand from the map: the
    kitchen's run along the south wall is the fridge, counter, sink unit, counter and stove, with the window over the
    sink unit, along the west wall a counter; so two cabinets west of the window, one east of it, two on the west run
    and the fridge's: 6. Regression caught: a cabinet
    floating over the open floor, one across the window or over the stove, or the fridge's cabinet drawn twice (the
    fridge stands in a corner, against two walls)."""
    cabinets = _of("upper_cabinet")
    assert len(cabinets) == 6
    homes = [f for f in _map().furniture if f.kind in ("counter", "sink_unit", "fridge") and f.mobility == "fixed"]
    for c in cabinets:
        assert c.pos[2] >= 1.45 - 1e-9 and c.pos[2] + c.size[2] <= CEILING + 1e-9 and c.size[1] <= 0.7
        assert _over_counters(c.box, homes), c.pos
    stove = _fixed("stove")[0]
    (hood,) = _of("range_hood")
    assert _inside_footprint(hood.box, stove) and abs(hood.pos[0] - stove.center[0]) < 0.02
    assert hood.pos[2] == 1.55 and abs(hood.pos[2] + hood.size[2] - CEILING) < 1e-6
    glass = _glass(_map().windows[0], _map().wall("W01"))
    for c in cabinets:
        assert not _meet(c.box, (stove.center[0] - 0.3, 0.0, 1.45, stove.center[0] + 0.3, 0.65, 2.15))
        assert c.box[3] <= glass[0] - CASING_W or c.box[0] >= glass[3] + CASING_W or c.box[1] > 0.3


def test_the_television_hangs_centred_over_the_console_above_what_may_rest_on_it():
    """Contract (plan §4: a wall-mounted TV above the console): one 50-inch screen (1.12 m) centred on the console and
    standing at least 5 cm above the tallest kind that fits on it (the console is 0.5 m: 0.5 + 0.314 + 0.05), on the
    wall (5 cm deep). Regression caught: a screen standing on the console where a book or a remote may lie, or off
    centre."""
    (tv,) = _of("wall_tv")
    console = _fixed("tv_console")[0]
    tallest = max(k.size[2] for k in KINDS.values())
    assert tv.pos[2] >= console.size[2] + tallest + 0.05 - 1e-9
    assert abs(tv.pos[1] - console.center[1]) < 0.01 and abs(tv.size[1] - 1.12) < 1e-3 and tv.size[0] <= 0.06
    assert tv.pos[2] + tv.size[2] < 2.0


def test_the_layout_keeps_clear_of_the_tallest_thing_that_rests_on_a_surface():
    """Contract (`dressing_home_plan.ITEM_CLEAR_M` against `dressing_zones.receptacle_volumes`): the plan keeps every
    piece this far above a receptacle surface, which must be the tallest kind of the catalog that rests on a surface,
    tracked or clutter, plus the zones' 0.05 m margin: the bag's 0.314 m for the tracked kinds, and a desk lamp's 0.45 m
    for clutter on desks and nightstands, so 0.50 m (the floor lamp, 1.5 m, stands on the floor). Regression caught: a
    plan that hangs a curtain or a cabinet in the volume a clutter kind stands in, which the guard then refuses."""
    plan = render_module("dressing_home_plan")
    rests = [k for k in CATALOGS["home"].clutter if any(place_of(p)[0].partition(":")[0] not in ("floor", "front", "on")
                                                        for p in k.places)]  # on a surface, not the floor or a face
    surface = [k.size[2] for k in rests]
    assert max(surface) == 0.45 and max(k.size[2] for k in KINDS.values()) == 0.314
    assert pytest.approx(max(surface) + 0.05) == plan.ITEM_CLEAR_M


def test_the_bathroom_has_a_cabinet_over_the_toilet_a_roll_holder_beside_it_and_a_curtain_over_the_tub():
    """Contract (plan §4: a closed bathroom cabinet, a toilet-roll holder, a shower curtain and rail at the tub): the
    cabinet hangs over the toilet (above the cistern, no mirror: a plain door), the holder beside the toilet at about
    0.7 m on the wall, and the curtain, with its rail, along the tub's open side above its rim and wholly inside its
    footprint (the tub carries no surface, so nothing tracked can stand under it). Regression caught: a curtain
    drawn down into the tub's mesh or out over the floor, or a holder hidden behind the toilet."""
    toilet, tub = _fixed("toilet")[0], _fixed("bathtub")[0]
    (cabinet,), (holder,), (curtain,) = _of("bathroom_cabinet"), _of("toilet_roll_holder"), _of("shower_curtain")
    assert abs(cabinet.pos[0] - toilet.center[0]) < 0.02 and cabinet.pos[2] >= toilet.size[2] + 0.5
    tx0, _, tx1, _ = _rect(toilet)
    assert holder.box[3] <= tx0 or holder.box[0] >= tx1  # beside it, not behind it
    assert holder.pos[2] >= 0.6 and holder.pos[2] + holder.size[2] <= 0.8
    assert _inside_footprint(curtain.box, tub) and curtain.pos[2] >= tub.size[2]
    assert curtain.pos[2] + curtain.size[2] <= 2.0  # the rail, under the ceiling's band of lights
    open_side = (math.cos(tub.yaw), math.sin(tub.yaw))
    toward_room = (curtain.pos[0] - tub.center[0]) * open_side[0] + (curtain.pos[1] - tub.center[1]) * open_side[1]
    assert toward_room > 0.25  # on the open side, not against the wall


def test_the_fan_hangs_over_the_bed_the_pendant_over_the_dining_table_and_a_lamp_under_every_light():
    """Contract (plan §4: a ceiling lamp per room, a fan in the bedroom): a flush lamp is centred on each of the six
    ceiling lights of the map (its underside above the light's 2.45 m is the render's business: 2.40 m); the 1.46 m fan
    is wholly over the bed's footprint (nobody can stand under it: it hangs to 1.98 m) and the pendant, 0.95 m long,
    wholly over the dining table's (its lowest point 1.55 m is above any camera there). Regression caught: a pendant
    over open floor where the camera passes (0.12 m of margin short of the 1.5 m camera) or a fan the bed does not
    cover."""
    lights = {(c.pos[0], c.pos[1]) for c in _map().ceiling_lights}
    flush = [d for d in _of("ceiling_lamp") if d.recipe == "proc:ceiling_lamp"]
    assert {(d.pos[0], d.pos[1]) for d in flush} == lights and len(flush) == 6
    (fan,) = _of("ceiling_fan")
    (pendant,) = [d for d in _of("ceiling_lamp") if d.recipe.startswith("asset:")]
    assert _inside_footprint(fan.box, _fixed("bed")[0]) and abs(fan.pos[2] + fan.size[2] - CEILING) < 1e-6
    assert _inside_footprint(pendant.box, _fixed("dining_table")[0])
    assert abs(pendant.pos[2] + pendant.size[2] - CEILING) < 1e-6
    assert fan.pos[2] > 1.9 and pendant.pos[2] > 1.5 + 0.02


def test_no_piece_cuts_into_the_body_of_fixed_furniture():
    """Contract: dressing stands on walls and ceilings or over furniture, never inside a counter, fridge, bed or tub
    (the zones keep it off receptacles and the floor, not out of a body that carries no surface). Each piece's box meets
    no fixed piece's body, from the floor to its height, but at a face; grime may lie on the open floor under a
    table, the sofa or the bed. Regression caught: an outlet or baseboard
    through a cabinet, or a cabinet through the fridge."""
    bodies = []
    for f in _map().furniture:
        if f.mobility == "fixed" and f.size[2] > 0.011:
            x0, y0, x1, y1 = _rect(f)
            bodies.append((f.id, (x0, y0, 0.0, x1, y1, f.size[2])))
    solid = {f.id for f in _map().furniture if f.under_m == 0.0}  # the floor under a table or a bed is open
    clashes = [(d.cls, d.pos, fid) for d, (fid, b) in itertools.product(_dressing(), bodies)
               if _meet(d.box, b) and (d.cls not in GRIME or fid in solid)]
    assert clashes == []


def test_no_two_pieces_share_volume_except_the_baseboards_where_two_walls_meet():
    """Contract: every piece is a thing of its own the builder draws, so none lies inside another: the sockets over the
    worktop stand on the backsplash (their box starts at the tile's face) and the tile stops short of the window trim.
    The one allowed overlap is the 1.2 cm square where the baseboards of two walls meet in a corner. Regression caught:
    a socket buried in the tile (the plate never showed in a render), tile running under the window casings, a picture
    over the television, or a casing through the trim beside it."""
    pieces = _dressing()
    clashes = [(a.cls, a.pos, b.cls, b.pos) for a, b in itertools.combinations(pieces, 2)
               if _meet(a.box, b.box) and not (a.cls == b.cls == "baseboard")]
    assert clashes == []
    worktop = [d for d in _of("outlet") if d.pos[2] > 1.0]
    tiles = _of("backsplash")
    assert worktop and all(any(_touch_front(o, t) for t in tiles) for o in worktop)


def _touch_front(plate, tile) -> bool:
    """Whether a plate stands on a tile: it lies within the tile along the wall and in height, and its box starts where
    the tile's ends, at the tile's face."""
    x0, y0, z0, x1, y1, z1 = plate.box
    tx0, ty0, tz0, tx1, ty1, tz1 = tile.box
    if not (z0 >= tz0 - 1e-6 and z1 <= tz1 + 1e-6):
        return False
    if tx1 - tx0 < 0.02:  # a tile on a wall that runs along y
        return x0 >= tx0 - 1e-6 and (abs(x0 - tx1) < 0.002 or abs(x1 - tx0) < 0.002) and ty0 <= y0 and y1 <= ty1
    return y0 >= ty0 - 1e-6 and (abs(y0 - ty1) < 0.002 or abs(y1 - ty0) < 0.002) and tx0 <= x0 and x1 <= tx1


def test_fittings_stand_at_their_standard_heights():
    """Contract (plan §4; the realism of a real apartment): socket plates centred 0.3 m up on the walls and 1.1 m over
    the worktop, switch plates 1.1 m, the thermostat 1.5 m, the wall clocks and pictures where eyes are (1.1 to 2.3 m),
    the radiators with their foot 15 cm off the floor (the vacuum drives under one, and the guard exempts a radiator
    from the helper's swept body from there up). Regression caught: sockets at the ceiling, a thermostat on the floor,
    or a picture that hangs at the skirting."""
    def centre(d):
        return d.pos[2] + d.size[2] / 2

    assert {round(centre(d), 2) for d in _of("outlet")} == {0.30, 1.11}
    assert {round(centre(d), 2) for d in _of("light_switch")} == {1.10}
    assert {round(centre(d), 2) for d in _of("thermostat")} == {1.50}
    assert all(d.pos[2] >= 1.1 and d.pos[2] + d.size[2] <= 2.3 for d in _of("picture_frame") + _of("wall_clock"))
    assert {d.pos[2] for d in _of("radiator")} == {0.15}


def test_every_recipe_is_procedural_or_a_pinned_poly_haven_asset_at_its_real_size():
    """Contract (plan §4: Poly Haven's real sizes; the builder dispatches on the recipe): a recipe is `proc:<class>` for
    a procedural piece or `asset:<id>` for one of the plan's nine assets, and an asset's box is that model's measured
    size (depth, width, height) to the millimetre: no model stretched to fit a wall. The numbers are what Poly Haven's
    API reports for each model; a hair or a cluster of crumbs has the variant it is drawn as after its name
    (`proc:hair:2`). Regression caught: a frame scaled to a 1 m span, an unpinned asset id, or a procedural recipe whose
    name is not its class (the builder would not know it)."""
    seen = set()
    for d in _dressing():
        if d.recipe.startswith("asset:"):
            asset = d.recipe.removeprefix("asset:")
            x, y, z = POLY_HAVEN_MM[asset]
            assert [v * 1000 for v in d.size] == pytest.approx([y, x, z], abs=0.6), asset
            seen.add(asset)
        else:
            variant = d.cls in ("hair", "crumbs") and d.recipe.startswith(f"proc:{d.cls}:")
            assert d.recipe == f"proc:{d.cls}" or variant, d
    assert seen == set(POLY_HAVEN_MM)


def test_every_room_is_dressed_to_the_plans_density():
    """Contract (plan §2: the apartment's target is 160-250 objects with Phase 2, and a median frame of at least 20):
    Phase 1 alone leaves at least 25 pieces in every room and between 150 and 250 in the apartment. Regression caught:
    a layout that quietly dropped a family (a room with no trim or no pictures), which would leave frames as bare as
    today's."""
    m = _map()
    per_room = {r.id: 0 for r in m.regions}
    for d in _dressing():
        per_room[m.region_of(d.pos[:2])] += 1
    assert min(per_room.values()) >= 25, per_room
    assert 150 <= len(_dressing()) <= 250
    assert all(d.phase == 1 for d in _dressing())
