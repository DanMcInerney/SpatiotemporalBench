"""The downtown in Blender: a hand-built episode on `downtown_a` (the map's own furniture, one of each of its eight
carried kinds and two movers, a drone at 12 m looking down 35 degrees) is rendered with the placeholder drawings and
with the procedural assets, and what the renders show is checked against the timeline; the outdoor scene is built
without a render and checked for what it must and must not hold, and the indoor scenes' graphs (drawn as placeholders
and from the procedural assets, as the shipped renders are) are built twice and hashed to the same digest (no digest is
recorded). Needs Blender and a GPU; marked `blender`, skipped when Blender is not installed. The assets come from
libraries built here from the packaged manifest's procedural entries (the downtown's, and the indoor maps' with their
textures swapped for a shader: nothing is downloaded). The module is two library builds, twelve Blender scripts that
only build (the indoor scenes twice each) and two six-frame renders.

The world is hand-built, not generated: `generate` takes about 40 s for a T3 world and does not promise every kind in
view of a few frames, whereas this test asserts that each of the ten kinds is seen."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from builders import camera_keyframe, minimal_timeline
from conftest import apartment, warehouse
from stbench import asset_manifest as A
from stbench import asset_render as AR
from stbench import blender_run
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera
from stbench.case import ScriptPaths
from stbench.catalog import CATALOGS, item_size
from stbench.geometry import Rect, dist_to_rect, quat_z, to_world
from stbench.mapspec import load_map
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16
from stbench.timeline import Ego, Entity, Placement, Region, Task, TaskPlan, load
from test_render_warehouse_blender import warehouse_timeline

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
DOWNTOWN_KINDS = tuple(CATALOGS["downtown"].kinds) + tuple(CATALOGS["downtown"].mover_kinds)
GROUND_PIECES = ("sidewalk", "parking_lane", "loading_zone", "plaza")
DRONE_HEIGHT_M, DRONE_PITCH = 12.0, math.radians(-35.0)
# (kind, colour, furniture id, surface id, metres along the surface): things on the map's own bays, loading zone,
# sidewalks and pads. The lanes' and zones' long axis is their local y; a sidewalk's too.
ON_SURFACES = [("car", "white", "F06", "S2", 0.0), ("van", "red", "F06", "S1", 0.0),
               ("pickup", "green", "F06", "S3", 0.0),
               ("box_truck", "yellow", "F12", "S1", 0.0), ("dumpster", "blue", "F09", "S1", -8.0),
               ("portable_toilet", "orange", "F09", "S1", 6.0), ("food_cart", "red", "F10", "S1", 0.0),
               ("skip", "yellow", "F21", "S1", 0.0), ("portable_toilet", "blue", "F21", "S3", 0.0),
               ("food_cart", "green", "F24", "S1", 0.0), ("car", "blue", "F08", "S2", 0.0)]
# movers on the high street's roadway: (kind, colour, x, y, yaw)
MOVERS = [("bus", "white", 56.0, 14.0, 0.0), ("taxi", "yellow", 44.0, 11.0, 0.4)]
# where the drone is and which way it looks (yaw): down the south lane, up the west street, over the square, along the
# roadway, out of the east end and out of the north end of the east street
POSES = [((17.5, 22.0), -math.pi / 2), ((25.0, 30.0), math.pi / 2), ((50.0, 4.0), -math.pi / 2), ((38.0, 14.0), 0.0),
         ((82.0, 13.5), 0.0), ((75.0, 50.0), math.pi / 2)]
# (kind, mobility, colour) of what the apartment's assets scene holds besides its table and door: one of each of the
# apartment's procedural models (the others are scanned or photographed, so they need a download)
APARTMENT_MODELS = [("glass", "carried", "white"), ("book", "carried", "red"), ("remote", "carried", "black"),
                    ("laptop", "carried", "grey"), ("plant", "carried", "green"), ("bottle", "carried", "blue"),
                    ("laundry_basket", "light", "beige"), ("robot_vacuum", "mover", "white"),
                    ("home_robot", "mover", "grey"), ("ball_robot", "mover", "yellow")]

def downtown_timeline():
    """The map's furniture as entities, the things on its surfaces, the two movers and the drone's poses above."""
    m, base = load_map("downtown_a"), minimal_timeline()
    entities, placements, furniture_entity = [], [], {}

    def add(kind, attributes, mobility, pos, yaw, asset=None, parent=None, size=None):
        eid = f"E{len(entities) + 1:04d}"
        entities.append(Entity(eid, kind, attributes, mobility, size or item_size(kind), asset))
        placements.append(Placement(eid, 0.0, None, pos, quat_z(yaw), parent))
        return eid

    for f in m.furniture:
        furniture_entity[f.id] = add(f.kind, {"name": f.name, "color": f.color}, f.mobility, (*f.center, 0.0), f.yaw,
                                     size=f.size)
    furniture = {f.id: f for f in m.furniture}
    for kind, colour, fid, sid, along in ON_SURFACES:
        f = furniture[fid]
        surface = next(s for s in f.surfaces if s.id == sid)
        frame = Rect(f.center[0], f.center[1], 0.0, 0.0, f.yaw)
        x, y = to_world(frame, surface.rect[0], surface.rect[1] + along)
        add(kind, {"color": colour}, "carried", (x, y, surface.z), f.yaw + math.pi / 2, f"proc_{kind}",
            furniture_entity[fid])
    for kind, colour, x, y, yaw in MOVERS:
        add(kind, {"color": colour}, "mover", (x, y, 0.0), yaw, f"proc_{kind}")
    keyframes = [camera_keyframe(k, 1.5 * k, xy, yaw, robot="drone", z=DRONE_HEIGHT_M, pitch=DRONE_PITCH)
                 for k, (xy, yaw) in enumerate(POSES)]
    regions = [Region(r.id, r.name, r.kind, tuple(map(tuple, r.polygon)), r.level, r.z0, r.z1) for r in m.regions]
    episode = dataclasses.replace(base.episode, map=m.name, robot="drone", job="drone/1")
    ego = Ego(keyframes, [Task(0, 0, len(keyframes), TaskPlan("hand", {}), ())], [])
    return dataclasses.replace(base, episode=episode, regions=regions, entities=entities, placements=placements,
                               states=[], ego=ego)


def apartment_timeline(assets: bool = False):
    """`minimal_timeline` on the apartment map. For the assets scene, where every carried, light and mover entity needs
    a procedural model, its chair and mugs (scanned meshes, so downloads) give way to one of each of `APARTMENT_MODELS`;
    the table and the door stay."""
    base = minimal_timeline()
    if not assets:
        return base
    entities = [e for e in base.entities if e.mobility == "fixed"]
    placements = [p for p in base.placements if p.entity in {e.id for e in entities}]
    for i, (kind, mobility, colour) in enumerate(APARTMENT_MODELS):
        eid = f"E{len(base.entities) + i + 1:04d}"  # after the ids the base timeline has, the door's the last
        size = next(f.size for f in apartment().furniture if f.kind == kind) if mobility == "light" else item_size(kind)
        entities.append(Entity(eid, kind, {"color": colour}, mobility, size, f"proc_{kind}"))
        placements.append(Placement(eid, 0.0, None, (1.0 + 0.3 * i, 1.0, 0.0), quat_z(0.0), None))
    return dataclasses.replace(base, entities=entities, placements=placements)


def procedural_library(root: Path, keep, map_name: str):
    """A library of the packaged manifest's procedural models of the kinds `keep` accepts (so no files), built by the
    real build, with the floors of `map_name`'s regions. A model's texture (a pallet's wood, a book's cloth) becomes
    the plastic shader, and the walls and palette textures go: nothing is downloaded and the cache is not read.
    Returns the manifest and the .blend."""
    raw = json.loads(A.packaged_path().read_text(encoding="utf-8"))
    raw["assets"] = [a for a in raw["assets"] if a["source"] == "procedural" and keep(a.get("kind"))]
    for a in raw["assets"]:
        a["materials"] = {slot: ref if ref.startswith("shader:") else "shader:plastic"
                          for slot, ref in a["materials"].items()}
    raw["sources"] = {"procedural": raw["sources"]["procedural"]}
    raw.update(palette_materials={}, walls=None, skies={}, floors={r.kind: raw["floors"][r.kind]
                                                         for r in load_map(map_name).regions
                                                         if raw["floors"][r.kind].startswith("shader:")})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STBENCH_ASSET_CACHE", str(root / "cache"))
        blend = root / "library" / "procedural.blend"
        AR.build_library(manifest, blend, root / "cache")
    return manifest, blend


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """The downtown's models; the streets' floors (`shader:asphalt`, `shader:paving`) stay."""
    return procedural_library(tmp_path_factory.mktemp("downtown"), lambda kind: kind in DOWNTOWN_KINDS, "downtown_a")


@pytest.fixture(scope="module")
def indoor_library(tmp_path_factory):
    """The apartment's and the warehouse's procedural models; the warehouse's floors (`shader:concrete`) stay, and the
    apartment's textured ones (ambientCG) go, so its floors are the placeholders'."""
    return procedural_library(tmp_path_factory.mktemp("indoor"), lambda kind: kind not in DOWNTOWN_KINDS, "warehouse_a")


@pytest.fixture(scope="module")
def renders(library, tmp_path_factory):
    """The episode rendered fast, once with the placeholder drawings and once with the procedural assets."""
    manifest, blend = library
    out = {}
    for appearance in ("placeholder", "assets"):
        paths = ep.write_script(tmp_path_factory.mktemp(appearance), downtown_timeline())
        render(paths, fast=True, appearance=appearance, library=blend, manifest=manifest)
        out[appearance] = paths
    return out


def probe(paths: ScriptPaths, out: Path, *extra: str) -> dict:
    """What `tests/blender_scene_probe.py` found in the scene Blender builds for an episode (the placeholder drawings,
    or with `extra` flags the library's assets)."""
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_scene_probe.py"), "--", "--render", str(REPO / "render"), "--src",
           str(REPO / "src"), "--root", str(paths.root), "--private", str(paths.private), "--out", str(out), *extra]
    blender_run.run(cmd, "building a scene")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def outdoor_scene(renders, tmp_path_factory):
    return probe(renders["placeholder"], tmp_path_factory.mktemp("probe") / "downtown.json")


@pytest.mark.parametrize("appearance", ["placeholder", "assets"])
def test_every_seen_downtown_entity_is_where_the_timeline_puts_it(appearance, renders):
    """Contract (V0 section 12's cross-check, plan-city section 7.2): for each entity seen in a rendered frame (215 px
    or more), the ground-plane position of its visible pixels lies within 0.1 m of its footprint in the timeline, with
    the placeholder drawings and with the procedural assets alike, from a drone 12 m up. The scene holds all ten kinds:
    cars, vans, a pickup, a box truck, skips, a dumpster, toilets and carts on bays, a loading zone, sidewalks and pads,
    a taxi and a 9 m bus on the roadway, and the four ground pieces themselves (sidewalks, parking lanes, a loading
    zone, the paving); each is seen in some frame. Regression caught: an entity drawn at the wrong place, scale or pivot
    (a vehicle built off its centre or facing the wrong way, a lane or slab off its pose, an item not moved to its
    surface), a mesh whose extents differ from the footprint the timeline uses, a kind with no geometry in one
    appearance, or a depth image in the wrong unit (centimetres on this map) that would put every centroid 10 times too
    near or too far."""
    paths = renders[appearance]
    tl, cam = load(paths.timeline), Camera.load(paths.camera)
    assert cam.depth_unit == "cm"
    imap, poses = V.load_instance_map(paths.instance_map), V.load_pose(paths.pose)
    seen = set()
    for k in range(tl.n_keyframes):
        with Image.open(paths.rgb(k)) as im:
            rgb = np.asarray(im.convert("RGB"))
        for r in V.frame_entities(read_png16(paths.instances(k)), read_png16(paths.depth(k)), rgb, cam,
                                  poses[k]["camera"]["pos"], poses[k]["camera"]["rot"], imap):
            if V.status(r["px"], cam) != "seen":
                continue
            e, p = tl.entity(r["id"]), tl.placement_of(r["id"], tl.keyframe(k).t)
            x, y = r["centroid_map"][:2]
            off = float(dist_to_rect(np.array([x]), np.array([y]), Rect(p.pos[0], p.pos[1], e.size[0], e.size[1],
                                                                        p.yaw))[0])
            assert off <= 0.1, f"k={k} {e.kind}: pixel centroid {off:.2f} m from where the timeline places it"
            seen.add(e.kind)
    assert set(DOWNTOWN_KINDS) <= seen, sorted(set(DOWNTOWN_KINDS) - seen)
    assert set(GROUND_PIECES) <= seen, sorted(set(GROUND_PIECES) - seen)


def test_depth_and_ids_of_the_city_are_identical_across_runs_and_sample_counts(renders, library, tmp_path):
    """Contract (V0 section 3): depth and instance PNGs stay bit-identical between two runs (a new Blender process each)
    and between 8 and 4 samples, in a scene whose far geometry (scenery 245 m out, 24,500 depth counts) and many objects
    could make them depend on the run. Regression caught: scenery or a layout drawn from unseeded randomness, a pass
    that is blended at the edges of the far facades, or a Cycles seed that moves."""
    manifest, blend = library
    again = ep.write_script(tmp_path, downtown_timeline())
    render(again, samples=4, appearance="placeholder", library=blend, manifest=manifest)
    first = renders["placeholder"]
    for k in range(len(POSES)):
        for name in ("depth", "instances"):
            assert getattr(again, name)(k).read_bytes() == getattr(first, name)(k).read_bytes(), (name, k)


@pytest.mark.parametrize("appearance", ["placeholder", "assets"])
def test_every_pixel_of_every_frame_has_a_surface_within_the_depth_range(appearance, renders):
    """Contract (plan-city sections 5.4 and 7.3): the city goes on past the map's edge far enough that no ray of a
    drone's frame, not even the topmost, which meets the ground 245 m out, leaves the world: no depth pixel is 0
    ("nothing nearer"), and none is farther than 300 m, well inside the 655 m a centimetre depth image holds. The frames
    include two that look out of the east end and out of the north end of the east street. Regression caught: scenery
    that stops short (a void or a skybox at the end of a street), or that runs on past the depth range (pixels that
    become 0 and read as unoccluded)."""
    paths = renders[appearance]
    for k in range(len(POSES)):
        depth = read_png16(paths.depth(k))
        assert (depth > 0).all(), f"k={k}: {(depth == 0).mean():.1%} of the pixels have no surface"
        assert depth.max() * 0.01 < 300.0, f"k={k}: the farthest pixel is {depth.max() * 0.01:.0f} m away"


def test_the_carts_second_set_of_umbrella_gores_is_a_lighter_tint_of_its_colour(library, renders, tmp_path):
    """Contract (plan-city section 3.1, manifest `tint.lighten`): the colour attribute sets a vehicle's, skip's,
    dumpster's and toilet's body to the palette colour and nothing else of it, and sets the food cart's umbrella to it
    and the umbrella's second set of gores half way to white (so a red cart's alternate gores are pink). Palette numbers
    are `shapes.PALETTE`'s (sRGB); targets are linear light. Regression caught: a tint on the glass or tyres (every
    colour reads dark), a cart whose umbrella is one flat colour, or a lighter tint applied on the wrong slot or
    twice."""
    manifest, blend = library
    out = probe(renders["assets"], tmp_path / "assets.json", "--appearance", "assets", "--library", str(blend),
                "--manifest", str(manifest))
    palette = {"white": (0.90, 0.90, 0.88), "red": (0.80, 0.07, 0.07), "green": (0.10, 0.58, 0.16),
               "yellow": (0.95, 0.80, 0.08), "blue": (0.08, 0.25, 0.80), "orange": (0.95, 0.42, 0.04)}

    def linear(rgb):
        return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]

    tl, tints = downtown_timeline(), out["tints"]
    object_of = {out["instance_map"][str(pid)]: name for pid, name in out["indexed"]}
    seen_bodies, carts = 0, 0
    for e in tl.entities:
        if e.mobility not in ("carried", "mover"):
            continue
        row, colour = tints[object_of[e.id]], e.attributes["color"]
        if e.kind == "food_cart":
            assert set(row) == {"umbrella", "umbrella2"}, row
            assert row["umbrella"] == pytest.approx(linear(palette[colour]), abs=2e-3)
            light = [c + (1.0 - c) * 0.5 for c in palette[colour]]
            assert row["umbrella2"] == pytest.approx(linear(light), abs=2e-3)
            carts += 1
        else:
            assert set(row) == {"body"}, (e.kind, row)
            assert row["body"] == pytest.approx(linear(palette[colour]), abs=2e-3), (e.kind, colour)
            seen_bodies += 1
    assert carts == 2 and seen_bodies == len(ON_SURFACES) - 2 + len(MOVERS)


def test_an_outdoor_scene_has_the_maps_sun_and_sky_and_no_ceiling_underlay_or_ceiling_light(outdoor_scene):
    """Contract (plan-city section 2, `build_scene`): the downtown's scene holds the map's own sun (3.6 W/m2 at 62
    degrees elevation, rotation 110) and sky (strength 0.55), one light only (no ceiling fixtures), no ceiling and no
    underlay, a camera that sees 1,000 m, and the map's twelve walls as facades. Regression caught: the apartment's
    indoor sun and sky on a street, a ceiling over a street (a black lid: the drone would see nothing), or the indoor
    clip distance of 200 m cutting the scenery off."""
    sky = load_map("downtown_a").outdoor
    assert outdoor_scene["lights"] == [["sun", "SUN", pytest.approx(sky.sun_strength)]]
    assert outdoor_scene["sun_energy"] == pytest.approx(3.6)
    assert outdoor_scene["sky"] == {"elevation": pytest.approx(math.radians(62.0), abs=1e-4),
                                    "rotation": pytest.approx(math.radians(110.0), abs=1e-4)}
    assert outdoor_scene["sky_strength"] == pytest.approx(0.55)
    names = outdoor_scene["objects"]
    assert not [n for n in names if n.startswith("ceiling") or n.startswith("fixture") or n == "underlay"], names
    assert sum(n.startswith("floor_") for n in names) == 4 and "ground" in names
    assert sum(n.startswith("facade_W") for n in names) == 12
    assert outdoor_scene["camera"]["clip_end"] == 1000.0
    # the sun shines from above, towards the south-west: 62 degrees up, from east-south-east
    x, y, z = outdoor_scene["sun_direction"]
    assert z == pytest.approx(math.sin(math.radians(62.0)), abs=1e-3)
    assert x > 0 > y


def test_the_scenery_is_never_an_entity(outdoor_scene):
    """Contract (plan-city section 7.3, lane B): the city beyond the edge and its street furniture are static structure
    with pass_index 0, so the instance pass and the oracle never name them: every object with a pass index is one of the
    timeline's entities, and the scene holds buildings, lamp posts, signal poles and hydrants. Regression caught:
    scenery given an index (the instance map would list a building the world does not have), in particular one that
    shares an entity's index, which a map from index to object would hide behind that entity."""
    assert len(outdoor_scene["instance_map"]) == len(downtown_timeline().entities)
    indexed = outdoor_scene["indexed"]  # (pass index, object name) of every object with an index, one pair each
    assert [pid for pid, _ in indexed] == sorted(map(int, outdoor_scene["instance_map"])), indexed
    names = outdoor_scene["objects"]
    for prefix in ("scenery_building_", "scenery_lamp_", "scenery_signal_", "scenery_hydrant_", "scenery_tree_"):
        assert any(n.startswith(prefix) for n in names), prefix
    assert not any(name.startswith("scenery_") for _, name in indexed)


def test_the_dressing_is_never_an_entity_and_the_photoreal_scene_is_the_same_every_build(library, renders, tmp_path):
    """Contract (realism Phase 1: dressing is never an entity, and the dressing
    and the scenery depend on the map alone): the downtown's scene built from the asset library holds the dressing
    (paint, iron, trees, benches, planters, bike racks: 82 objects named `dressing_<class>_<n>`, M3-RY's keep-out of the
    drone's reach having taken 18 of the 98), the scenery's trees and the paint of its continued streets, none of which
    has a pass index, so the `instance_map` is the timeline's entities and nothing else; and two builds in two Blender
    processes hash to the same scene (every object, mesh, material and its node graph, light, the world, the camera).
    Regression caught: dressing given a pass index (the oracle would list a street tree), an `instance_map` that grows
    with the dressing, or a layout, builder or material seeded from a clock, `random`'s state or the order of a set (a
    different city in every render). This scene's digest was not recorded before: it is the first the realism lanes'
    outdoor scene has."""
    manifest, blend = library
    extra = ("--appearance", "assets", "--library", str(blend), "--manifest", str(manifest))
    first = probe(renders["assets"], tmp_path / "first.json", *extra)
    again = probe(renders["assets"], tmp_path / "again.json", *extra)
    assert first["digest"] == again["digest"]
    names = first["objects"]
    assert sum(n.startswith("dressing_") for n in names) >= 75
    for prefix in ("dressing_street_tree_", "dressing_tree_crown_", "dressing_crosswalk_", "dressing_manhole_cover_",
                   "scenery_tree_"):
        assert any(n.startswith(prefix) for n in names), prefix
    assert len(first["instance_map"]) == len(downtown_timeline().entities)
    assert [pid for pid, _ in first["indexed"]] == sorted(map(int, first["instance_map"]))
    assert not any(name.startswith(("dressing_", "scenery_")) for _, name in first["indexed"])


def test_the_indoor_scenes_graphs_are_the_same_in_every_build(indoor_library, tmp_path_factory):
    """Contract (plan-city section 7.3, lane B: indoor maps build the same scene every time): the apartment's and the
    warehouse's scenes, built from a hand-made episode each, as placeholder drawings and from a library of the
    procedural models (the path the shipped renders take: models, tinted copies of their materials, the warehouse's
    concrete shader), hash to the same digest in two Blender processes, over every object, mesh, material and its node
    graph, light, the world and the camera. No digest is recorded, so this does not say an indoor scene is unchanged
    since the city was added; it says the scene depends on the episode and the library alone. The textured surfaces of
    the shipped library (the apartment's floors, walls and cloth) are not in it: they need a download. Regression
    caught: an indoor builder, layout or material seeded from a clock, `random`'s state or the order of a set (a
    different apartment or warehouse in every render)."""
    manifest, blend = indoor_library
    assets = ("--appearance", "assets", "--library", str(blend), "--manifest", str(manifest))
    scenes = {"apartment": (apartment_timeline(), ()), "warehouse": (warehouse_timeline(), ()),
              "apartment_assets": (apartment_timeline(assets=True), assets),
              "warehouse_assets": (warehouse_timeline(), assets)}
    root = tmp_path_factory.mktemp("indoor")
    for name, (tl, extra) in scenes.items():
        paths = ep.write_script(root / name, tl)
        first = probe(paths, root / f"{name}.json", *extra)
        again = probe(paths, root / f"{name}_again.json", *extra)
        assert first["digest"] == again["digest"] and len(first["digest"]) == 64, name
    assert warehouse().name == "warehouse_a"


def test_every_shader_and_facade_the_scene_needs_builds(outdoor_scene):
    """Contract: every name in `materials.SHADERS` builds into a material, and every facade style the map names is one
    `facade_shaders` has. Regression caught: a shader listed and not built, or a facade style on the map with no shader
    (met only when the first frame shows that wall)."""
    assert outdoor_scene["shaders"] == sorted(outdoor_scene["shaders"]) and "asphalt" in outdoor_scene["shaders"]
    styles = {w.facade for w in load_map("downtown_a").walls}
    assert styles <= set(outdoor_scene["facade_styles"])
