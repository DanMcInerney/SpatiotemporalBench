"""The warehouse's Phase 1 dressing layout (`render/dressing_layout_warehouse.py`):
what is specific to this layout. The rules every layout must meet (classes registered, every class used, no box in a
forbidden volume or outside an allowed place) are `test_dressing.py`'s; here are the things a distribution centre has
and the numbers they are placed by, worked out by hand from `warehouse_a` (17.4 x 19.8 m inside, 6 m high; four dock
doors 2.4 m wide at x = 3.7, 7.1, 10.5 and 13.9 on the south wall; three aisles 3.6 m wide between racks whose fronts
are at x = 1.2 and 4.8, 7.0 and 10.6, 12.8 and 16.4, running from y = 9.1 to the north wall at 19.9; the nine lights
over the aisle centres at 5.8 m). The layout needs no Blender: `render/` is not a package, so it is put on the path."""

from __future__ import annotations

import ast
import copy
import json
import math
import re
import sys
from collections import Counter
from importlib import resources
from pathlib import Path

import pytest

RENDER = Path(__file__).resolve().parents[1] / "render"
sys.path.insert(0, str(RENDER))  # its modules import each other by bare name

import dressing_layout_warehouse as L  # noqa: E402
from dressing_spec import overlaps  # noqa: E402

from stbench.mapspec import load_map, surface_world_rect  # noqa: E402

AISLES = {"A": (1.2, 4.8), "B": (7.0, 10.6), "C": (12.8, 16.4)}  # x of the racks' fronts either side of each aisle
DOORS = (3.7, 7.1, 10.5, 13.9)
LIGHTS = [(x, y) for x in (3.0, 8.8, 14.6) for y in (12.0, 17.0)] + [(x, 4.6) for x in (3.0, 8.8, 14.6)]
PLAN_ASSETS = {"fire_alarm", "security_camera_01", "power_box_01"}  # the plan's picks that fit, and are in the manifest
ASSET_SIZE = {"fire_alarm": (0.0264, 0.0994, 0.1436), "security_camera_01": (0.5537, 0.1696, 0.2861),
              "power_box_01": (0.1071, 0.3455, 0.5031)}  # the library's measured native boxes: (y, x, z)


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(resources.files("stbench.maps").joinpath("warehouse_a.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dressing(spec):
    return L.layout(spec)


def of(dressing, cls: str):
    return [d for d in dressing if d.cls == cls]


def text(d) -> str:
    """What a sign, label or number prints: the recipe after its builder's name."""
    return d.recipe.split(":", 2)[2]


def test_each_aisle_has_one_readable_sign_on_the_end_of_a_row_facing_the_dock(dressing):
    """Contract (readable aisle signs and beam labels, one aisle
    sign per aisle): exactly three signs, one for each of the aisles A, B and C from the west, each printing its aisle
    and its bays (01-08: four bays a side, odd on the west), each standing on the south end of a rack row of its own
    aisle (the row ends are at y = 9.1; the rows of aisle A are at x 0.1 to 1.2 and 4.8 to 5.9, of B 5.9 to 7.0 and
    10.6 to 11.7, of C 11.7 to 12.8 and 16.4 to 17.5), above the floor guard and below the top beam, facing south (yaw
    270 degrees) into the dock where the robot passes. Regression caught: a sign missing or doubled, one that prints a
    different aisle's name, one turned to the wall, or one on the far end of a row where nothing can read it."""
    signs = of(dressing, "aisle_sign")
    assert sorted(text(s) for s in signs) == [f"AISLE {a}|BAYS 01-08" for a in "ABC"]
    rows = {"A": ((0.1, 1.2), (4.8, 5.9)), "B": ((5.9, 7.0), (10.6, 11.7)), "C": ((11.7, 12.8), (16.4, 17.5))}
    for s in signs:
        x0, y0, z0, x1, y1, z1 = s.box
        letter = text(s)[6]
        assert any(lo - 1e-9 <= x0 and x1 <= hi + 1e-9 for lo, hi in rows[letter]), (letter, s.box)
        assert y1 == pytest.approx(9.1) and y1 - y0 < 0.05 and z0 >= 2.0 and z1 <= 2.6, s.box
        assert s.yaw_deg % 360 == 270


def test_every_beam_has_its_own_location_label_on_its_front_face(dressing):
    """Contract (barcode labels on the beams, readable): the 6 rows have 2 beams
    above the floor in each of 4 bays, so 48 labels, each printing a code `<aisle>-<bay>-<level>` that no other label
    has: bays 01 to 08 along an aisle from the dock, odd on the west row and even on the east row (01 and 02 are the
    first bay of each), levels 02 and 03 for the beams at 1.3 and 2.6 m (01 is the floor). A label stands on its beam's
    front face (the beam's front face is 2 cm behind the rack's front plane, 0.53 m from its middle, and 12 cm high up
    to the level's height) and in its own bay. Regression caught: two labels with one code, a label on the wrong row or
    level, a label floating off its beam, or in the wrong bay (the robot would read a false place)."""
    labels = of(dressing, "beam_label")
    codes = [text(d) for d in labels]
    assert len(labels) == 48 and len(set(codes)) == 48
    assert set(codes) == {f"{a}-{n:02d}-{lv:02d}" for a in "ABC" for n in range(1, 9) for lv in (2, 3)}
    for d in labels:
        aisle, n, lv = text(d).split("-")
        west, bay = int(n) % 2 == 1, (int(n) + 1) // 2
        x0, y0, z0, x1, y1, z1 = d.box
        face = AISLES[aisle][0] - 0.02 if west else AISLES[aisle][1] + 0.02
        assert (x0 if west else x1) == pytest.approx(face), d.box
        beam = 1.3 * (int(lv) - 1)
        assert beam - 0.12 <= z0 and z1 <= beam, d.box
        assert 9.1 + 2.7 * (bay - 1) < y0 and y1 < 9.1 + 2.7 * bay, d.box


def test_every_dock_door_is_dressed_and_numbered_from_the_west_and_none_covers_a_shutter(dressing):
    """Contract (plan-realism section 4: "seals, bumpers, levelers, dock lights and door numbers at each dock door"):
    within 1.5 m of each of the four doors' middles there are three seal pieces (a pad each side of the 2.4 m door,
    outside its x range, and a head over it from 3.0 m up), two bumpers under the pads, one flush leveler (no more than
    2 cm thick) between the door and the staging lane (y from 0.2 to 0.58, the lane begins at 0.6), one dock light and
    one door number printing 1 to 4 from the west. No dressing box meets a door's own box (x +-1.2 of its middle, y 0.1
    to 0.2, z 0 to 3.0), so the shutter stays whole in the pictures. Regression caught: a door without its seal, two
    doors with one number, a leveler that fills the staging lane's marking, or a seal over the shutter."""
    for number, c in enumerate(DOORS, 1):
        near = [d for d in dressing if abs(d.pos[0] - c) <= 1.5 and d.pos[1] < 1.0 and d.cls in {
            "dock_door_seal", "dock_bumper", "dock_leveler", "dock_light", "door_number"}]
        assert Counter(d.cls for d in near) == {"dock_door_seal": 3, "dock_bumper": 2, "dock_leveler": 1,
                                                "dock_light": 1, "door_number": 1}, c
        pads = [d for d in near if d.cls == "dock_door_seal" and d.pos[2] < 2.0]
        heads = [d for d in near if d.cls == "dock_door_seal" and d.pos[2] >= 2.0]
        assert len(pads) == 2 and len(heads) == 1 and heads[0].pos[2] == pytest.approx(3.0)
        assert all(abs(d.pos[0] - c) > 1.2 for d in pads) and all(d.pos[2] >= 1.3 for d in pads)
        leveler = next(d for d in near if d.cls == "dock_leveler")
        assert leveler.box[1] >= 0.2 - 1e-9 and leveler.box[4] <= 0.58 + 1e-9 and leveler.size[2] <= 0.02
        assert text(next(d for d in near if d.cls == "door_number")) == str(number)
        shutter = (c - 1.2, 0.1, 0.0, c + 1.2, 0.2, 3.0)
        assert not [d.cls for d in dressing if overlaps(d.box, shutter)], c


def test_no_dressing_enters_an_aisle_corridor_or_the_opening_of_a_free_bay_level(dressing, spec):
    """Contract (the robot's clearance and the receptacle rule, held to a stricter standard than `dressing_zones`'s):
    the robot keeps 0.25 m from the rack fronts, so each aisle's corridor is 0.25 m in from each front (x 1.45 to 4.55,
    7.25 to 10.35 and 13.05 to 16.15, from the rows' ends at y = 9.1 to 19.65, 0.25 m off the north wall), and nothing
    below the 2.15 m of the tallest mover is in it. A free bay-level is open the whole way from its surface up to the
    next beam: 1.18 m, though a thing on it is 1.05 m at most. No dressing box meets any of those 36 openings (worked
    out from the map's surface rects) at any height. Regression caught: a protector, guard or label that grows into the
    corridor, a sign or label that blocks the opening an item is carried into (a pallet cannot be put on a level that
    has a sign in it)."""
    corridors = [(a + 0.25, 9.1, 0.0, b - 0.25, 19.65, 2.15) for a, b in AISLES.values()]
    m = load_map("warehouse_a")
    openings = []
    for f in m.furniture:
        for s in f.surfaces:
            if f.kind == "pallet_rack":
                r = surface_world_rect(f.center, f.yaw, s)
                c, sn = abs(math.cos(r.yaw)), abs(math.sin(r.yaw))
                hx, hy = (r.sx * c + r.sy * sn) / 2, (r.sx * sn + r.sy * c) / 2
                openings.append((r.cx - hx, r.cy - hy, s.z, r.cx + hx, r.cy + hy, s.z + 1.18))
    assert len(openings) == 36 and len(corridors) == 3
    for d in dressing:
        assert not [o for o in corridors + openings if overlaps(d.box, o)], (d.cls, d.recipe, d.box)


def test_the_nine_high_bay_fixtures_sit_at_the_lights_and_nothing_hangs_under_a_lens(dressing):
    """Contract (plan-realism section 4: "keep the nine"; the lighting lane sets their power and puts the emitter at the
    lens, which is the map's light height): one round fixture 0.4 m across at each of the nine lights, its base (the
    lens) at the light's height 5.8 m and no taller than the deck's underside at 5.96. Nothing else stands in the 0.6 x
    0.6 m square under a lens from the 5 m line down to the lens (a truss, pipe, duct, tray or camera would shadow it).
    Regression caught: a fixture off its light, hung from the deck so that its lens is above the light, or a purlin or
    pipe laid across a light."""
    fixtures = of(dressing, "high_bay_light")
    assert sorted((round(d.pos[0], 6), round(d.pos[1], 6)) for d in fixtures) == sorted(LIGHTS)
    assert all(d.pos[2] == pytest.approx(5.8) and d.box[5] <= 5.96 + 1e-9 for d in fixtures)
    for x, y in LIGHTS:
        under = (x - 0.3, y - 0.3, 5.0, x + 0.3, y + 0.3, 5.8)
        assert not [d.cls for d in dressing if overlaps(d.box, under)], (x, y)


def test_roof_lights_are_three_percent_of_the_roof_and_no_structure_is_below_five_metres(dressing):
    """Contract (plan-realism section 4: "skylight panels at about 3% of the roof area"; the roof structure and its
    services stand above 5 m, where `dressing_zones` allows them anywhere): the six roof lights, 1.0 x 1.7 m, make 10.2
    m2 of the 344.5 m2 roof (2.96%: between 2.5% and 3.5%); each of the trusses, purlins, deck, lights, pipes, heads,
    trays and ducts has its lowest point above 5.0 m. Regression caught: a roof with no lights or with a fifth of it
    glazed (the lighting lane's calibration would be off), or a truss that hangs into the robot's view."""
    roof = 17.4 * 19.8
    area = sum((d.box[3] - d.box[0]) * (d.box[4] - d.box[1]) for d in of(dressing, "skylight"))
    assert 0.025 * roof <= area <= 0.035 * roof and len(of(dressing, "skylight")) == 6
    for cls in ("roof_truss", "purlin", "roof_deck", "high_bay_light", "sprinkler_main", "sprinkler_head", "cable_tray",
                "air_duct", "skylight", "security_camera"):
        assert all(d.pos[2] >= 5.0 for d in of(dressing, cls)), cls


def test_fire_equipment_hangs_at_reaching_height_with_a_call_point_beside_each_extinguisher(dressing):
    """Contract (plan-realism section 4: "fire extinguishers and fire alarm call points at realistic intervals"; OSHA
    colours, red for fire equipment): five 2 kg extinguishers (0.11 m across, so within the 0.12 m wall band; Poly
    Haven's 6 kg unit is 0.37 m deep) hung with their base 1.0 to 1.1 m up and five call points 1.3 to 1.4 m up, each
    call point within half a metre of an extinguisher on the same wall, none within 3 m of another extinguisher
    (they are spread, 4 on the dock walls and 1 at an aisle's end). Regression caught: an extinguisher on the floor or
    near the roof, a call point nowhere near one, or all of them in one corner."""
    ext, alarms = of(dressing, "fire_extinguisher"), of(dressing, "fire_alarm")
    assert len(ext) == len(alarms) == 5
    assert all(1.0 <= d.pos[2] <= 1.1 + 1e-9 for d in ext) and all(1.3 <= d.pos[2] <= 1.4 + 1e-9 for d in alarms)
    for a in alarms:
        assert [e for e in ext if math.dist(e.pos[:2], a.pos[:2]) <= 0.5]
    assert all(math.dist(p.pos[:2], q.pos[:2]) > 3.0 for i, p in enumerate(ext) for q in ext[i + 1:])


def test_cameras_look_down_each_aisle_from_its_far_wall_and_across_the_dock(dressing):
    """Contract (plan-realism section 4: "security cameras", Poly Haven `security_camera_01`, 0.55 m deep, so above the
    5 m line where depth is free): seven, three on the north wall at the aisle centres (x 3.0, 8.8, 14.6) facing south
    down the aisles, two on the dock wall over doors 2 and 4 facing north, one on each side wall facing across the dock,
    all with their base between 5.1 and 5.3 m. Regression caught: a camera below the 5 m line (inside the robot's view
    and clearance), or aisles nobody watches."""
    cams = of(dressing, "security_camera")
    assert len(cams) == 7 and all(5.1 <= d.pos[2] <= 5.3 for d in cams)
    north = sorted(d.pos[0] for d in cams if d.yaw_deg % 360 == 270)
    assert north == pytest.approx([3.0, 8.8, 14.6])
    assert Counter(round(d.yaw_deg % 360) for d in cams) == {270: 3, 90: 2, 0: 1, 180: 1}


def test_recipes_are_builders_or_the_plans_poly_haven_picks_at_their_real_sizes(dressing):
    """Contract: every recipe is `proc:<name>` (a name to build, with
    the printed text after a second colon) or `asset:<id>` with an id the plan picks from Poly Haven and the library
    pins; an asset's box is the box the library measured for it, depth x width x height (the native y, x and z: fire
    alarm 26 x 99 x 144 mm, camera 554 x 170 x 286, the power box with its door closed 107 x 346 x 503; models face -y),
    to within a centimetre (the library refuses a layout box more than 5% off). The three the plan picks that do not
    fit the wall band or the roof (the 6 kg extinguisher, the utility cabinet, the duct kit) and the cable kit, a
    catalogue of loose parts that reads as a pinboard, are not placed. Regression caught: a recipe with a typo, an asset
    the library does not have, or a box that stretches or shrinks an asset."""
    for d in dressing:
        kind, _, name = d.recipe.partition(":")
        assert kind in ("proc", "asset") and re.fullmatch(r"[a-z0-9_]+(:[ -~]+)?", name), d.recipe
        if kind == "asset":
            assert name in PLAN_ASSETS and name in ASSET_SIZE and ":" not in name, d.recipe
            assert d.size == pytest.approx(ASSET_SIZE[name], abs=0.01), d.recipe


def test_the_layout_is_the_same_every_time_and_its_random_choices_come_from_the_maps_name(spec, dressing):
    """Contract (rule 13: every episode of a map has the same dressing; `dressing_spec`: "any randomness is seeded from
    the map's name"): laying the same spec out again, and a copy of it, gives the same boxes in the same order; a map
    called something else has the same classes in the same numbers (the structure comes from the map, not the seed) with
    some boxes moved (the seeded choices: roof light rows, which row holds each sign, hang heights). Regression caught:
    a clock or process seed (the renders of one map differ), or a seed that ignores the name."""
    assert L.layout(copy.deepcopy(spec)) == dressing
    other = L.layout({**spec, "name": "warehouse_b"})
    assert Counter(d.cls for d in other) == Counter(d.cls for d in dressing) and other != dressing


def test_the_rack_constants_are_the_ones_furniture_warehouse_draws_with():
    """Contract (the labels and protectors are placed on the rack the builder draws, which `render/` cannot import
    outside Blender): `POST_Y`, `BEAM_H` and the beams' front face set in 0.02 m from the rack's front are read from
    `render/furniture_warehouse.py`'s source and are the numbers the layout uses. Regression caught: the builder's
    beams or posts resized while the labels and protectors stay where they were (labels floating off their beams)."""
    tree = ast.parse((RENDER / "furniture_warehouse.py").read_text(encoding="utf-8"))
    consts = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "elts", None)
                  and [t.id for t in n.targets[0].elts] == ["POST_X", "POST_Y", "BEAM_H", "BEAM_D"])
    post_x, post_y, beam_h, beam_d = ast.literal_eval(consts.value)
    rack = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pallet_rack")
    xo = next(n for n in ast.walk(rack) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "xo")
    set_in = ast.literal_eval(xo.value.right.right)  # xo = sign * (hx - 0.02)
    import dressing_warehouse_parts as P

    assert (post_y, beam_h, set_in) == (P.POST_Y, P.BEAM_H, P.BEAM_SET_IN)
    assert post_x == 0.08 and beam_d == 0.14  # the protectors' and labels' numbers assume these
