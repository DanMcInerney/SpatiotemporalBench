"""The light of the photoreal appearance in Blender (realism Phase 1): the physics of a
lamp's power, colour and balance and of a turned HDRI, rendered on tiny scenes; the lamps, portals, sun and sky
`lighting.setup` puts in the apartment's and the warehouse's scenes, checked against the dressing's own fixtures; and
the lux and the colour cast that grey cards in those scenes report. Needs Blender; marked `blender`, skipped when it is
not installed. The renders run on the CPU (the scenes are tiny or only built). The libraries are the procedural models
of the packaged manifest (nothing is downloaded), so the maps' skies are the procedural one. About two minutes for the
module (one library build, six driver runs, no GPU needed)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from conftest import apartment, warehouse
from stbench import blender_run
from stbench import episode as ep
from stbench.render_cli import blender_path
from test_render_city_blender import DOWNTOWN_KINDS, apartment_timeline, procedural_library
from test_render_warehouse_blender import warehouse_timeline

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
UNIT_LUX = 240.0  # `cycles_setup.UNIT_LUX`, for every catalog: the lux at which a grey card reads 0.18
# By hand, the lumen method `lm = lux * area / (0.75 * lamps in the room)` and watts = lm * pi / 240, for
# apartment_a: living room 4.5 x 3.3 m at 150 lx, kitchen 4.5 x 2.7 at 300, bedroom 2.7 x 3.7 at 120, bathroom 2.7 x
# 2.3 at 200, hall 1.5 x 6.1 at 115 (15% over the band's floor: asked at 100 it measured 96) with two lamps. Per region
# id: (lumens of one lamp, colour temperature of the room).
APARTMENT_LAMPS = {"R1": (701.5, 3000.0), "R2": (2970.0, 3000.0), "R3": (4860.0, 3500.0), "R4": (1598.0, 3000.0),
                   "R5": (1656.0, 3500.0)}
# By hand, the cast of a lamp's direct light on a grey card, through the apartment's 3250 K balance: the lamp's
# blackbody over the balance's, per channel, red and blue over green. The blackbodies' linear sRGB (r/g, b/g; Planckian
# locus to CIE xy to sRGB): 3000 K 2.089 and 0.322, 3250 K 1.901 and 0.389, 3500 K 1.752 and 0.454. So 3000 K reads
# 2.089 / 1.901 = 1.10 and 0.322 / 0.389 = 0.83, and 3500 K 0.92 and 1.17. Per colour temperature: (red over green,
# blue over green).
DIRECT_CAST = {3000.0: (1.10, 0.83), 3500.0: (0.92, 1.17)}
CAST_TOLERANCE = 0.05  # the rooms' medians read 0.01 to 0.02 off these (the hall's cards also see the kitchen's lamp)
HIGH_BAY_LUMENS = 17000.0  # 130 W of LED at 130 lm/W
FIXTURE_Z, LENS_Z = 2.40, 5.80  # the underside of the flush lamp, the lens of the high bay


def driver(task: str, out: Path, *extra: str) -> dict:
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_lighting_driver.py"), "--", "--task", task, "--render", str(REPO / "render"),
           "--src", str(REPO / "src"), "--out", str(out), *extra]
    blender_run.run(cmd, f"the lighting driver ({task})")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """The procedural models of both indoor maps (the packaged manifest's), floors of the warehouse's shader."""
    return procedural_library(tmp_path_factory.mktemp("lighting"), lambda kind: kind not in DOWNTOWN_KINDS,
                              "warehouse_a")


@pytest.fixture(scope="module")
def apartment_scene(library, tmp_path_factory):
    manifest, blend = library
    paths = ep.write_script(tmp_path_factory.mktemp("apartment"), apartment_timeline(assets=True))
    return driver("scene", tmp_path_factory.mktemp("out") / "apartment.json", "--root", str(paths.root), "--private",
                  str(paths.private), "--manifest", str(manifest), "--library", str(blend))


@pytest.fixture(scope="module")
def warehouse_scene(library, tmp_path_factory):
    manifest, blend = library
    paths = ep.write_script(tmp_path_factory.mktemp("warehouse"), warehouse_timeline())
    return driver("scene", tmp_path_factory.mktemp("out") / "warehouse.json", "--root", str(paths.root), "--private",
                  str(paths.private), "--manifest", str(manifest), "--library", str(blend))


@pytest.fixture(scope="module")
def physics(tmp_path_factory):
    return driver("physics", tmp_path_factory.mktemp("physics") / "physics.json")


def emitters(scene: dict, prefix: str = "ceiling_light_") -> dict[str, dict]:
    """The area lights (portals aside) whose names start with `prefix`: the rooms' lamps, or `under_cabinet_light_`."""
    return {li["name"]: li for li in scene["lights"]
            if li["type"] == "AREA" and not li["portal"] and li["name"].startswith(prefix)}


def over(boxes: list[dict], xy) -> list[dict]:
    """The boxes whose footprint holds the point `xy`."""
    return [b for b in boxes if b["min"][0] <= xy[0] <= b["max"][0] and b["min"][1] <= xy[1] <= b["max"][1]]


# -- the physics -----------------------------------------------------------------------------------------------------
def test_a_lamps_watts_are_its_flux_in_radiometric_units(physics):
    """Contract (`lighting` docstring, "Units"): a Lambertian area light of P Blender watts lights a white card at
    height h under it to a radiance P / (pi^2 h^2), the card being a Lambertian reflector (L = E / pi) and the lamp,
    small against h, a point source of intensity P / pi. By hand 10 W at 2 m: 10 / (9.8696 * 4) = 0.2533; the lamp
    is 0.36 m across, so the card reads about 1% less. Regression caught: `render_watts` or the lumen method resting
    on another conversion (a lamp that is 4 or pi times too dim or bright, so every room off its lux by that), found
    by the lux meter only after a full render."""
    assert physics["white_card_radiance_10w_2m"] == pytest.approx(0.2533, rel=0.03)


def test_a_source_balanced_to_its_own_temperature_is_white_and_the_lumens_do_not_change(physics):
    """Contract (`lighting.balanced`): the white balance is a gain per channel on a source's colour, so a 3000 K lamp
    balanced to 3000 K lights a grey card with no cast (r = g = b), one balanced to 3250 K (the apartment's) keeps a
    small warm cast, one balanced to 6500 K (daylight) stays orange (a blackbody of 3000 K has r/g about 2.1 and b/g
    about 0.32 in linear sRGB), and the card's luminance is the same in all three (the lumens are lumens, not a colour's
    brightness). Regression caught: a balance that does not cancel the temperature (catalog colours shifted by the lamp,
    the plan's validity risk), or one that changes the room's brightness (rooms off their lux by the colour)."""
    own, apartment_k, daylight = (physics[f"grey_under_3000k_balanced_to_{k}"] for k in (3000, 3250, 6500))
    assert own["r_g"] == pytest.approx(1.0, abs=0.02) and own["b_g"] == pytest.approx(1.0, abs=0.02)
    assert 1.05 < apartment_k["r_g"] < 1.15 and 0.78 < apartment_k["b_g"] < 0.90
    assert daylight["r_g"] > 1.8 and daylight["b_g"] < 0.4
    assert apartment_k["luminance"] == pytest.approx(own["luminance"], rel=0.03)
    assert daylight["luminance"] == pytest.approx(own["luminance"], rel=0.03)


def test_an_hdri_turned_by_a_heading_shows_its_features_that_far_the_other_way_round(physics):
    """Contract (`lighting_sky._hdri`, `lighting.setup`): a feature of the picture at azimuth phi (`atan2(y, x)`)
    appears at phi - heading_deg, so that the city's HDRI is turned `sun_atan2 - map_sun_atan2` to put its own sun
    under the map's sun lamp. The test HDRI is dim (0.2) with a bright block (10) on the horizon at +x. Heading 0
    shows the block at +x, 90 at -y, -90 at +y; and the world is the environment texture, its clamp and the Is
    Camera Ray mix of the two backgrounds. Regression caught: a turn the wrong way (the sky's sun 2 x 14 degrees off
    the lamp's), or a heading that does nothing."""
    for heading, bright in ((0, "+x"), (90, "-y"), (-90, "+y")):
        row = physics[f"hdri_heading_{heading}"]
        dim = [v for k, v in row.items() if k != bright]
        assert row[bright] > 5 * max(dim), (heading, row)
    assert physics["hdri_world"] == sorted(["ShaderNodeBackground", "ShaderNodeBackground", "ShaderNodeLightPath",
                                            "ShaderNodeMixShader", "ShaderNodeOutputWorld", "ShaderNodeTexEnvironment"])
    assert physics["hdri_clamp"] == 20.0


def test_a_sky_set_by_its_illuminance_lights_a_card_facing_up_with_that_many_lux(physics):
    """Contract (`lighting_sky._hdri` with `sky_lux`, `hdri_remap.horizontal_illuminance`): a picture of radiance 0.5
    above the horizon and 5.0 below it (first rows the bottom, as Blender stores pixels), at `sky_lux` 11,760, lights a
    small card facing up with 11,760 lx in the linear unit, within 3%. By hand its horizontal illuminance is pi x 0.5 =
    1.5708 (the ground half counts for nothing), so the strength is 11,760 pi / 240 / 1.5708 = 98; a card of albedo 0.18
    under 11,760 lx reads 0.18 x 11,760 / 240 = 8.82. The picture is not uniform, so a calibration that counts the
    wrong hemisphere (5.0 for 0.5) is ten times off. Regression caught: a sky calibrated against another integral than
    the one Cycles lights a surface with, or a strength that skips the unit."""
    assert physics["sky_lux_card"] == pytest.approx(11760.0, rel=0.03)


# -- the apartment ---------------------------------------------------------------------------------------------------
def test_the_apartment_has_no_green_ground_plane_no_old_panels_and_a_portal_in_each_window(apartment_scene):
    """Contract (plan section 4, home light): the photoreal apartment has no ground plane (depth through a window is
    "none", as the sky in the RGB says), none of the 0.6 m glowing `fixture_i` panels (the dressing's lamps are the
    fixtures now), and a Cycles portal in each of the map's five windows, in its wall's plane and facing into the
    building. Regression caught: the green `ground` plane back (a lawn in every window), the old panels drawn beside
    the new lamps, or a window without a portal (daylight sampled blindly through a hole)."""
    objects = apartment_scene["objects"]
    assert "ground" not in objects and apartment_scene["ground_materials"] == []
    assert not [n for n in objects if n.startswith("fixture_")], objects
    portals = {li["name"]: li for li in apartment_scene["lights"] if li["type"] == "AREA" and li["portal"]}
    assert sorted(portals) == [f"portal_N{i}" for i in range(1, 6)]
    # N3 is on the north wall (y = 6.2) of the living room: 1.6 x 1.2 m, its centre at z = 1.0 + 0.6, facing south (-y)
    n3 = portals["portal_N3"]
    assert [n3["size"], n3["size_y"]] == pytest.approx([1.6, 1.2]) and n3["loc"] == pytest.approx([2.4, 6.2, 1.6])
    assert n3["dir"] == pytest.approx([0, -1, 0], abs=1e-4)


def test_each_apartment_ceiling_light_is_an_area_light_inside_its_dressing_lamp_with_its_rooms_power(apartment_scene):
    """Contract (plan section 4; the home layout's flush lamp: 0.40 x 0.40 m at z 2.40-2.50 centred on each map ceiling
    light): the six lights of the map are six 0.36 m area lights facing down at (x, y, 2.398), under the lamp's
    underside (2.40) and inside its footprint, each of the lumens its room's lux calls for (APARTMENT_LAMPS, by hand) at
    the camera's exposure, in its room's colour temperature (3000 K, kitchen and bathroom 3500 K) with the balance to
    3250 K on top. Regression caught: an emitter inside the lamp's body (a shadow over the room), outside it, or above
    it; a room off its lux by the conversion; a kitchen at the living room's colour."""
    lights, lamps = emitters(apartment_scene), apartment_scene["ceiling_lamps"]
    assert sorted(lights) == [f"ceiling_light_{i}" for i in range(6)]
    for i, spec in enumerate(apartment().ceiling_lights):
        li = lights[f"ceiling_light_{i}"]
        lumens, kelvin = APARTMENT_LAMPS[spec.region]
        assert li["loc"][:2] == pytest.approx(list(spec.pos[:2])) and li["loc"][2] == pytest.approx(2.398)
        assert (li["shape"], li["size"], li["spread"]) == ("SQUARE", pytest.approx(0.36), pytest.approx(180.0))
        assert li["dir"] == pytest.approx([0, 0, -1], abs=1e-4)
        assert li["energy"] == pytest.approx(lumens * math.pi / UNIT_LUX, rel=0.02)
        assert (li["use_temperature"], li["temperature"]) == (True, kelvin)
        under = over(lamps, spec.pos)
        assert under, f"no dressing lamp over {spec.pos}"
        assert li["loc"][2] <= FIXTURE_Z and min(b["min"][2] for b in under) >= FIXTURE_Z - 1e-3
        assert all(b["pass_index"] == 0 for b in under)


def test_the_apartments_sun_is_the_clear_skys_from_the_south_east_and_the_world_is_procedural_without_an_hdri(
        apartment_scene):
    """Contract (DECISIONS 2026-10-06, M1's calls; `look_home.RIG`): the apartment's sun shines from azimuth 141
    degrees at 38 degrees elevation (10:00 solar time at 45 N near an equinox), so its rays travel along
    -(sin141 cos38, cos141 cos38, sin38) = (-0.496, 0.612, -0.616) by hand; its strength is the clear sky's sun, 59,500
    lx on a horizontal surface, in the linear unit: 59,500 pi / 240 / sin 38 = 1265 W/m2; a manifest with no sky for the
    catalog (this test's) leaves the procedural sky, with no environment image; and the scene sets no exposure shift and
    no view-transform white balance (the balance is in the lamps' colours). Regression caught: a sun turned by the sky
    code, a sun in watts that skips the unit (a sunlit floor then reads 1/1000 of what it should), or a second white
    balance on top of the lamps'."""
    (sun,) = [li for li in apartment_scene["lights"] if li["type"] == "SUN"]
    assert sun["dir"] == pytest.approx([-0.496, 0.612, -0.616], abs=2e-3)
    assert sun["energy"] == pytest.approx(1265.0, rel=0.01)
    assert apartment_scene["hdri"] is None and "ShaderNodeTexSky" in apartment_scene["world_nodes"]
    assert apartment_scene["view"] == {"exposure": 0.0, "white_balance": False}
    assert apartment_scene["lighting"]["rig"] == "home" and apartment_scene["lighting"]["portals"] == 5


def test_each_wall_cabinet_has_a_strip_light_under_it_as_long_as_it_is(apartment_scene):
    """Contract (`lighting.under_cabinet_lights`): every `dressing_upper_cabinet_*` has one strip light, a rectangle
    4 cm across and as long as the cabinet, 2 mm under its underside and inside its footprint, of 500 lm a metre (500
    x length x pi / 240 watts by hand: 6.5 W a metre) at the kitchen's 3500 K. Regression caught: a cabinet with no
    strip (the worktop under it in its shadow), a strip inside the cabinet's body, or lumens that do not follow the
    length."""
    cabinets, strips = apartment_scene["upper_cabinets"], emitters(apartment_scene, "under_cabinet_light_")
    assert len(cabinets) >= 5 and sorted(strips) == sorted(f"under_cabinet_light_{i}" for i in range(len(cabinets)))
    for i, box in enumerate(cabinets):
        li = strips[f"under_cabinet_light_{i}"]
        extent = [hi - lo for lo, hi in zip(box["min"], box["max"], strict=True)]
        length = max(extent[:2])
        assert li["shape"] == "RECTANGLE" and sorted([li["size"], li["size_y"]]) == pytest.approx([0.04, length])
        assert li["loc"][2] == pytest.approx(box["min"][2] - 0.002, abs=1e-3) and over([box], li["loc"])
        assert li["energy"] == pytest.approx(500.0 * length * math.pi / UNIT_LUX, rel=0.01)
        assert (li["use_temperature"], li["temperature"]) == (True, 3500.0)
        assert li["dir"] == pytest.approx([0, 0, -1], abs=1e-4)


def test_the_kitchen_worktop_has_300_to_750_lx_from_the_lamps_alone(library, tmp_path_factory):
    """Contract (plan 4.10 item 5 and M1's brief: 300-750 lx on the kitchen worktop, measured with daylight off): a
    grey card flat on the worktops (every `kitchen counter` surface, a 3 x 3 grid on each) reads, at the median of
    the cards that are not buried in a thing on the counter (over 50 lx), within 300 to 750 lx, and at the 10th
    percentile at least 250. Measured 537 median with the strips, 227 with the ceiling lamp alone, whose light the
    wall cabinets shade. Regression caught: the strips under the cabinets lost, or put where they give the worktop
    nothing."""
    manifest, blend = library
    paths = ep.write_script(tmp_path_factory.mktemp("worktop"), apartment_timeline(assets=True))
    out = driver("lux", tmp_path_factory.mktemp("luxout") / "worktop.json", "--root", str(paths.root), "--private",
                 str(paths.private), "--manifest", str(manifest), "--library", str(blend), "--surface",
                 "kitchen counter")
    lux = sorted(c["lux"] for c in out["cards"] if c["lux"] > 50)
    assert len(lux) >= 15 and 300.0 <= lux[len(lux) // 2] <= 750.0 and lux[len(lux) // 10] >= 250.0, lux


def test_a_card_above_the_roof_in_daylight_reads_the_suns_lux_by_hand(library, tmp_path_factory):
    """Contract (`look_home.RIG["sun"]`, the unit): with the lamps off, a card facing up above the building,
    where the sun shines on it unobstructed, reads the sun's 59,500 lx on a horizontal surface (the strength 1265
    W/m2 times sin 38, over the unit: 1265 x 0.61566 x 240 / pi = 59,500) plus the procedural sky of this test's
    manifest (a few tens of lux, uncalibrated: the shipped sky is the manifest's HDRI, set by `sky_lux`): between 59,000
    and 61,000 lx. Regression caught: a sun in watts that skips the unit or the sine, which `exposure` would hide
    from the eye."""
    manifest, blend = library
    paths = ep.write_script(tmp_path_factory.mktemp("roof"), apartment_timeline(assets=True))
    points = tmp_path_factory.mktemp("points") / "points.json"
    points.write_text(json.dumps([{"name": "roof", "pos": [4.5, 3.0, 6.0], "normal": [0, 0, 1]}]), encoding="utf-8")
    out = driver("lux", tmp_path_factory.mktemp("luxout") / "roof.json", "--root", str(paths.root), "--private",
                 str(paths.private), "--manifest", str(manifest), "--library", str(blend), "--points", str(points),
                 "--sources", "daylight")
    assert 59000.0 <= out["cards"][0]["lux"] <= 61000.0, out["cards"]


# -- the warehouse ---------------------------------------------------------------------------------------------------
def test_each_warehouse_ceiling_light_is_a_5000k_disc_under_its_high_bay_lens_and_the_skylights_glow(warehouse_scene):
    """Contract (plan section 4, warehouse light; EN 12464-1): the nine lights of `warehouse_a` are nine 0.36 m
    discs at z 5.798, under the lens plane of the dressing's high-bay fixture (z 5.80) and inside its 0.4 m
    footprint, each of 17,000 lm (a 130 W LED high bay) at the warehouse's 200 lx exposure, 5000 K and balanced to
    5000 K, and no old panels, ground plane, sunlight or portals (a closed building's world is black); every
    skylight pane (slot 0 of `dressing_skylight_*`, slot 1 being the frame) carries the daylight panel material.
    Regression caught: a lens above its light (shadowed by the fixture), the old 0.6 m panels beside the new
    fixtures, a frame that glows, or a skylight that lights nothing."""
    lights, bays = emitters(warehouse_scene), warehouse_scene["high_bays"]
    assert sorted(lights) == [f"ceiling_light_{i}" for i in range(9)] and len(bays) == 9
    for i, spec in enumerate(warehouse().ceiling_lights):
        li = lights[f"ceiling_light_{i}"]
        assert li["loc"] == pytest.approx([spec.pos[0], spec.pos[1], 5.798])
        assert (li["shape"], li["size"], li["temperature"]) == ("DISK", pytest.approx(0.36), 5000.0)
        assert li["energy"] == pytest.approx(HIGH_BAY_LUMENS * math.pi / UNIT_LUX, rel=0.01)
        # the lamp's colour is the inverse of a 5000 K light's (1.213, 0.961, 0.762), so that 5000 K comes out white
        assert li["colour"] == pytest.approx([0.824, 1.041, 1.312], abs=2e-3)
        under = over(bays, spec.pos)
        assert len(under) == 1 and under[0]["min"][2] == pytest.approx(LENS_Z, abs=1e-3) and li["loc"][2] < LENS_Z
    objects = warehouse_scene["objects"]
    assert "ground" not in objects and not [n for n in objects if n.startswith("fixture_")]
    assert not [li for li in warehouse_scene["lights"] if li.get("portal") or (li["type"] == "SUN" and li["energy"])]
    assert warehouse_scene["lighting"]["skylights"] == len(warehouse_scene["skylights"]) == 6
    assert all(ms[0] == "daylight_panel" and ms[1:] != ["daylight_panel"]
               for ms in warehouse_scene["skylight_materials"].values()), warehouse_scene["skylight_materials"]


def test_the_apartments_rooms_have_the_lux_the_lumen_method_aims_at_in_a_neutral_light(library, tmp_path_factory):
    """Contract (plan section 2 and 4: 100-300 lx in living rooms, 300-750 lx for kitchen tasks; a fixed exposure and
    balance so that the warm lamps do not shift catalog colours): with the lamps alone, a grey card flat at 0.75 m over
    a 3 x 3 grid of each room reads, at the median, within 0.7 to 1.4 of the room's target (hall 115, living room 150,
    kitchen 300, bedroom 120, bathroom 200 lx; measured 0.82 to 1.11 of it on this scene, its walls painted per room),
    and the lamps' own cast on those cards is their colour through the balance: read from the direct light alone (no
    bounce, `direct_only` in the driver), at the median of the cards the lamps see (over 20 lx), red over green and
    blue over green are within 0.05 of `DIRECT_CAST` for the room's lamps (by hand: 3000 K 1.10 and 0.83, 3500 K 0.92
    and 1.17; measured within 0.02 of them). The bounce is left out of the cast because the walls tint it with their
    paint, not the lamps' colour: in the bathroom, aqua (0.72, 0.92, 0.95 of the white), it took red over green from
    0.92 to 0.87. Regression caught: a lamp shadowed by its own fixture or put in the wrong place (a room at a tenth of
    its lux), the watts conversion off by pi or 4 (every room 3 to 4 times off), a utilisation factor off by half
    (every room 1.5 times off), or a balance that does not hold the colours (a lamp unbalanced reads 2.1 and 0.32; one
    250 K off its balance, 0.1 off in red)."""
    manifest, blend = library
    paths = ep.write_script(tmp_path_factory.mktemp("lux"), apartment_timeline(assets=True))
    out = driver("lux", tmp_path_factory.mktemp("luxout") / "lux.json", "--root", str(paths.root), "--private",
                 str(paths.private), "--manifest", str(manifest), "--library", str(blend))
    target = {"hall": 115.0, "living room": 150.0, "kitchen": 300.0, "bedroom": 120.0, "bathroom": 200.0}
    kelvin = {region.name: APARTMENT_LAMPS[region.id][1] for region in apartment().regions}
    assert out["unit_lux"] == UNIT_LUX
    for room, cards in out["rooms"].items():
        lux = sorted(c["lux"] for c in cards)[len(cards) // 2]
        assert 0.7 * target[room] <= lux <= 1.4 * target[room], (room, round(lux), "lx; target", target[room])
        # the cast is the lamps' own (direct light); a card inside a piece of furniture reads nothing worth a colour
        seen = [c["direct"] for c in cards if c["direct"]["lux"] > 20]
        assert seen, (room, "no card sees a lamp")
        r_g, b_g = (sorted(c[k] for c in seen)[len(seen) // 2] for k in ("r_g", "b_g"))
        want = DIRECT_CAST[kelvin[room]]
        assert (r_g, b_g) == pytest.approx(want, abs=CAST_TOLERANCE), (room, round(r_g, 2), round(b_g, 2), "want", want)


def test_the_warehouse_has_150_lx_on_its_gangways_and_about_200_lx_on_its_rack_faces(library, tmp_path_factory):
    """Contract (plan section 2 and 4, EN 12464-1: at least 150 lx on gangway floors and 200 lx on rack faces; the
    nine high bays "aimed" at both): with the fixtures alone, a grey card on the floor at the centre of each of the
    three aisles reads 150 lx or more (measured about 275 on this plain-palette scene, 170 to 290 on the shipped one),
    the mean of the cards on the face of the west rack at 0.6, 1.5, 2.4 and 3.3 m at three places along it reads 170 to
    260 lx (measured about 205; the lowest level gets about half of the highest, the fixtures being overhead), and the
    cards' cast is small (red over green 0.95 to 1.1, blue over green 0.9 to 1.05 at the median: 5000 K light balanced
    to 5000 K). Regression caught: a fixture shadowed by its own housing or its lens put above its light (a dark hall),
    the conversion off by pi or 4, or lights that give the floor but not the racks."""
    manifest, blend = library
    paths = ep.write_script(tmp_path_factory.mktemp("warehouse_lux"), warehouse_timeline())
    floor = [{"name": "floor", "pos": [x, 14.5, 0.02], "normal": [0, 0, 1]} for x in (3.0, 8.8, 14.6)]
    faces = [{"name": "face", "pos": [1.23, y, z], "normal": [1, 0, 0]} for y in (10.5, 14.5, 18.5)
             for z in (0.6, 1.5, 2.4, 3.3)]  # the face of the west rack F13 (x 1.2), at three places along it
    points = tmp_path_factory.mktemp("points") / "points.json"
    points.write_text(json.dumps(floor + faces), encoding="utf-8")
    out = driver("lux", tmp_path_factory.mktemp("luxout") / "warehouse.json", "--root", str(paths.root), "--private",
                 str(paths.private), "--manifest", str(manifest), "--library", str(blend), "--points", str(points))
    cards = out["cards"]
    assert out["unit_lux"] == UNIT_LUX
    assert all(c["lux"] >= 150.0 for c in cards if c["name"] == "floor"), cards
    on_faces = [c["lux"] for c in cards if c["name"] == "face"]
    assert 170.0 <= sum(on_faces) / len(on_faces) <= 260.0, on_faces
    r_g, b_g = (sorted(c[k] for c in cards)[len(cards) // 2] for k in ("r_g", "b_g"))
    assert 0.95 <= r_g <= 1.1 and 0.9 <= b_g <= 1.05, (r_g, b_g)
