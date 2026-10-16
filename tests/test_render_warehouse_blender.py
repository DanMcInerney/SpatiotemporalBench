"""The warehouse in Blender: a hand-built episode on `warehouse_a` (racks, a dock, a forklift, a tote robot, pallet
jacks, a roll cage and items on rack beams) is rendered with the placeholder drawings and with the procedural assets,
and what the renders show is checked against the timeline. Needs Blender and a GPU; marked `blender`, skipped when
Blender is not installed. The assets come from a library built here from the packaged manifest's warehouse entries
(procedural; their ambientCG materials are swapped for a shader, so nothing is downloaded and the cache is not read).
About half a minute for the module (one library build, two five-frame renders)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from builders import camera_keyframe, minimal_timeline
from conftest import warehouse
from stbench import asset_manifest as A
from stbench import asset_render as AR
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera
from stbench.catalog import item_size
from stbench.geometry import Rect, dist_to_rect, quat_z
from stbench.mapspec import surface_world_rect
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16
from stbench.timeline import Ego, Entity, Placement, Region, Task, TaskPlan, load

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
WAREHOUSE_KINDS = ("pallet", "tote", "drum", "sack", "bin", "pallet_jack", "roll_cage", "forklift", "tote_robot")
# (kind, colour, rack, surface, along the surface in m, yaw relative to the rack): items on the west rack of the west
# aisle, which faces +x into the lane; a pallet's 1.2 m side runs along the rack (a quarter turn), a tote's 0.6 m
# across it
ON_RACK = [("pallet", "white", "F13", "S15", -0.6, math.pi / 2), ("tote", "red", "F13", "S15", 0.6, 0.0),
           ("bin", "blue", "F13", "S15", 1.05, 0.0), ("drum", "green", "F13", "S16", -0.5, 0.0),
           ("sack", "yellow", "F13", "S16", 0.5, math.pi / 2)]
# (kind, colour, x, y, yaw): the movers in the lane of the west aisle
MOVERS = [("tote_robot", "orange", 3.6, 15.4, math.pi), ("forklift", "yellow", 2.6, 18.0, -math.pi / 2)]
# the camera looks along its yaw from 1.5 m, 15 degrees down: three in the west aisle, two in the dock (the helper
# robot carries it there: `camera_keyframe`)
POSES = [((4.3, 13.6), math.pi), ((4.3, 14.5), math.pi), ((3.0, 11.0), math.pi / 2), ((9.0, 5.6), -math.pi / 2),
         ((14.0, 5.0), -math.pi / 2)]


def warehouse_timeline():
    """The map's furniture as entities, the items on its racks, the two movers and the camera poses above."""
    wh, base = warehouse(), minimal_timeline()
    entities, placements, rack_entity = [], [], {}

    def add(kind, attributes, mobility, pos, yaw, asset=None, parent=None, size=None):
        eid = f"E{len(entities) + 1:04d}"
        entities.append(Entity(eid, kind, attributes, mobility, size or item_size(kind), asset))
        placements.append(Placement(eid, 0.0, None, pos, quat_z(yaw), parent))
        return eid

    for f in wh.furniture:
        asset = f"proc_{f.kind}" if f.mobility == "light" else None
        rack_entity[f.id] = add(f.kind, {"name": f.name, "color": f.color}, f.mobility, (*f.center, 0.0), f.yaw, asset,
                                size=f.size)
    furniture = {f.id: f for f in wh.furniture}
    for kind, colour, rack_id, surface_id, along, turn in ON_RACK:
        f = furniture[rack_id]
        surface = next(s for s in f.surfaces if s.id == surface_id)
        centre = surface_world_rect(f.center, f.yaw, surface)
        x, y = centre.cx - math.sin(f.yaw) * along, centre.cy + math.cos(f.yaw) * along  # along the rack (local y)
        add(kind, {"color": colour}, "carried", (x, y, surface.z), f.yaw + turn, f"proc_{kind}", rack_entity[rack_id])
    for kind, colour, x, y, yaw in MOVERS:
        add(kind, {"color": colour}, "mover", (x, y, 0.0), yaw, f"proc_{kind}")
    keyframes = [camera_keyframe(k, 1.5 * k, xy, yaw) for k, (xy, yaw) in enumerate(POSES)]
    regions = [Region(r.id, r.name, r.kind, tuple(map(tuple, r.polygon)), r.level, r.z0, r.z1) for r in wh.regions]
    episode = dataclasses.replace(base.episode, map=wh.name)
    ego = Ego(keyframes, [Task(0, 0, len(keyframes), TaskPlan("hand", {}), ())], [])
    return dataclasses.replace(base, episode=episode, regions=regions, entities=entities, placements=placements,
                               states=[], ego=ego)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A library of the packaged manifest's warehouse models only, built by the real build. Their ambientCG materials
    (the pallet's wood, the drum's rings, the sack's cloth) become the plastic shader here: the point is the
    geometry."""
    root = tmp_path_factory.mktemp("warehouse")
    raw = json.loads(A.packaged_path().read_text(encoding="utf-8"))
    raw["assets"] = [a for a in raw["assets"] if a.get("kind") in WAREHOUSE_KINDS and a["source"] == "procedural"]
    for a in raw["assets"]:
        a["materials"] = {slot: ref if ref.startswith("shader:") else "shader:plastic"
                          for slot, ref in a["materials"].items()}
    raw["sources"] = {"procedural": raw["sources"]["procedural"]}
    raw.update(palette_materials={}, walls=None, skies={},
               floors={r.name: "shader:concrete" for r in warehouse().regions})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STBENCH_ASSET_CACHE", str(root / "cache"))
        blend = root / "library" / "warehouse.blend"
        AR.build_library(manifest, blend, root / "cache")
        yield manifest, blend


@pytest.fixture(scope="module")
def renders(library, tmp_path_factory):
    """The episode rendered fast, once with the placeholder drawings and once with the procedural assets."""
    manifest, blend = library
    out = {}
    for appearance in ("placeholder", "assets"):
        paths = ep.write_script(tmp_path_factory.mktemp(appearance), warehouse_timeline())
        render(paths, fast=True, appearance=appearance, library=blend, manifest=manifest)
        out[appearance] = paths
    return out


@pytest.mark.parametrize("appearance", ["placeholder", "assets"])
def test_every_seen_warehouse_entity_is_where_the_timeline_puts_it(appearance, renders):
    """Contract (V0 section 12's cross-check, plan-warehouse section 7.2): for each entity seen in a rendered frame
    (215 px or more), the ground-plane position of its visible pixels lies within 0.1 m of its footprint in the
    timeline, with the placeholder drawings and with the procedural assets alike. The scene holds every kind the
    warehouse adds: pallets, totes, bins, a drum and a sack on rack beams and on the floor of a rack bay, a forklift and
    a tote robot standing in a lane, pallet jacks and a roll cage on the dock, and the racks, staging lanes and dock
    doors themselves; each of the nine kinds is seen in some frame. Regression caught: an entity drawn at the wrong
    place, scale or pivot (a rack built off its centre, an item not moved to its surface's pose, a mesh whose extents
    differ from the footprint the timeline and the visibility rules use), or a kind with no geometry in one
    appearance."""
    paths = renders[appearance]
    tl, cam = load(paths.timeline), Camera.load(paths.camera)
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
    assert set(WAREHOUSE_KINDS) <= seen, sorted(set(WAREHOUSE_KINDS) - seen)
    assert {"pallet_rack", "staging_lane", "dock_door"} <= seen


def test_the_warehouse_renders_are_lit_and_not_empty(renders):
    """Contract (plan-warehouse section 8, risk 2): the 270 W high-bay lights light the hall in both appearances: no
    frame is dark (mean luminance over 40 of 255 and 5% of its pixels over 60) and none is blown out (under 1% at 255).
    Regression caught: a warehouse lit by the apartment's 45 W lamps at 5.8 m (near black), or by far too many watts."""
    for appearance, paths in renders.items():
        for k in range(len(POSES)):
            with Image.open(paths.rgb(k)) as im:
                lum = np.asarray(im.convert("L")).astype(float)
            assert lum.mean() > 40 and (lum > 60).mean() > 0.05 and (lum >= 255).mean() < 0.01, (appearance, k)
