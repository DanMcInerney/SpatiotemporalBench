"""The suburb in Blender (M4 suburb part b, §6): a hand-made episode on `suburb` (the map's own furniture, a vehicle on
a driveway, a kerbside bay, the car park and a sidewalk, and a drone at 15 m over the street) has its scene built from
the procedural asset library by `tests/blender_suburb_driver.py`, with no render and no GPU, and the scene is checked
for what the residential look must give it. What the map, its dressing and its people are as plain data is
`test_suburb_map.py`'s, `test_dressing.py`'s and `test_render_looks.py`'s; what the kit's pieces are in Blender is
`test_residential_blender.py`'s. Needs Blender; marked `blender`, skipped when it is not installed. The library is built
here from the packaged manifest's procedural models of the catalog's five kinds, with the floors that are shaders
(asphalt, the lawn, paving): nothing is downloaded. About ten seconds of Blender work (a library build, a scene build).
The driver brings the view layer up to date before it reads the boxes: an object made in a script keeps an identity
`matrix_world` until then, so without it every box of the scenery is its model's own, at the origin, inside the map."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from builders import camera_keyframe, minimal_timeline
from stbench import asset_manifest as A
from stbench import asset_render as AR
from stbench import blender_run
from stbench import episode as ep
from stbench.catalog import CATALOGS, item_size
from stbench.geometry import Rect, quat_z, to_world
from stbench.mapspec import load_map
from stbench.render_cli import blender_path
from stbench.timeline import Ego, Entity, Placement, Region, Task, TaskPlan

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
KINDS = tuple(CATALOGS["residential"].kinds)
DRONE_HEIGHT_M = 15.0
DRESSING = {"tree_stem", "tree_canopy", "lamp_post", "shrub", "playground"}  # the residential classes, by hand
# (kind, colour, piece kind, which of that kind's pieces, metres along its first surface): a car on the driveway of the
# first drop zone ("zone drive") and a van on another driveway, a pickup on a kerbside bay, a car and a skip in the car
# park's two lanes and a dumpster on a sidewalk
ON_SURFACES = [("car", "white", "zone drive", 0, 0.0), ("van", "red", "driveway", 7, 0.0),
               ("pickup", "green", "parking_lane", 0, 0.0), ("car", "blue", "parking_lane", 10, 7.0),
               ("skip", "yellow", "parking_lane", 11, 0.0), ("dumpster", "orange", "sidewalk", 0, -20.0)]


def on_surface(m, entity_of: dict, kind: str, colour: str, piece_kind: str, n: int, along: float, add) -> None:
    """A thing of `kind` on the `n`th piece of `piece_kind` that has a surface, `along` its first surface (its long
    axis is the surface's local y for a lane or a sidewalk, its local x for a driveway)."""
    drive = piece_kind in ("driveway", "zone drive")
    if piece_kind == "zone drive":
        zone = next(p for p in m.furniture if p.kind == "drop_zone")
        f = next(p for p in m.furniture if p.kind == "driveway" and abs(p.center[0] - zone.center[0]) < 1.0
                 and abs(p.center[1] - zone.center[1]) < 6.0)
    else:
        f = [p for p in m.furniture if p.kind == piece_kind and p.surfaces][n]
    s = f.surfaces[0]
    frame = Rect(f.center[0], f.center[1], 0.0, 0.0, f.yaw)
    x, y = to_world(frame, s.rect[0] + (along if drive else 0.0), s.rect[1] + (0.0 if drive else along))
    add(kind, {"color": colour}, "carried", (x, y, s.z), f.yaw + (0.0 if drive else math.pi / 2), f"proc_{kind}",
        entity_of[f.id])


def suburb_timeline():
    """The map's furniture as entities, the things on its surfaces and the drone's poses: at the nest looking south,
    over the first drop zone looking down, out of each end of the street and over each side."""
    m, base = load_map("suburb"), minimal_timeline()
    entities, placements, entity_of = [], [], {}

    def add(kind, attributes, mobility, pos, yaw, asset=None, parent=None, size=None):
        eid = f"E{len(entities) + 1:04d}"
        entities.append(Entity(eid, kind, attributes, mobility, size or item_size(kind), asset))
        placements.append(Placement(eid, 0.0, None, pos, quat_z(yaw), parent))
        return eid

    for f in m.furniture:
        entity_of[f.id] = add(f.kind, {"name": f.name, "color": f.color}, f.mobility, (*f.center, 0.0), f.yaw,
                              size=f.size)
    for kind, colour, piece_kind, n, along in ON_SURFACES:
        on_surface(m, entity_of, kind, colour, piece_kind, n, along, add)
    nest = next(f for f in m.furniture if f.kind == "nest")
    zone = next(f for f in m.furniture if f.kind == "drop_zone")
    poses = [(nest.center, -math.pi / 2, math.radians(-45.0)), (zone.center, 0.0, math.radians(-90.0)),
             ((8.0, 41.0), math.pi, math.radians(-35.0)), ((292.0, 41.0), 0.0, math.radians(-35.0)),
             ((150.0, 14.0), -math.pi / 2, math.radians(-35.0)), ((150.0, 70.0), math.pi / 2, math.radians(-35.0))]
    keyframes = [camera_keyframe(k, 1.5 * k, xy, yaw, robot="drone", z=DRONE_HEIGHT_M, pitch=pitch)
                 for k, (xy, yaw, pitch) in enumerate(poses)]
    regions = [Region(r.id, r.name, r.kind, tuple(map(tuple, r.polygon)), r.level, r.z0, r.z1) for r in m.regions]
    episode = dataclasses.replace(base.episode, map=m.name, robot="drone", job="drone/1")
    ego = Ego(keyframes, [Task(0, 0, len(keyframes), TaskPlan("hand", {}), ())], [])
    return dataclasses.replace(base, episode=episode, regions=regions, entities=entities, placements=placements,
                               states=[], ego=ego)


def residential_library(root: Path):
    """A library of the packaged manifest's procedural models of the catalog's kinds (so no files), built by the real
    build, with the floors of the map's region kinds that are shaders. A model's texture becomes the plastic shader and
    the walls and palette textures go: nothing is downloaded. Returns the manifest and the .blend."""
    raw = json.loads(A.packaged_path().read_text(encoding="utf-8"))
    raw["assets"] = [a for a in raw["assets"] if a["source"] == "procedural" and a.get("kind") in KINDS]
    for a in raw["assets"]:
        a["materials"] = {slot: ref if ref.startswith("shader:") else "shader:plastic"
                          for slot, ref in a["materials"].items()}
    raw["sources"] = {"procedural": raw["sources"]["procedural"]}
    kinds = {r.kind for r in load_map("suburb").regions}
    raw.update(palette_materials={}, walls=None, skies={},
               floors={k: ref for k, ref in raw["floors"].items() if k in kinds and ref.startswith("shader:")})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STBENCH_ASSET_CACHE", str(root / "cache"))
        blend = root / "library" / "procedural.blend"
        AR.build_library(manifest, blend, root / "cache")
    return manifest, blend


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """What `blender_suburb_driver.py` found in the scene it built for the hand-made episode from the library."""
    root = tmp_path_factory.mktemp("suburb")
    manifest, blend = residential_library(root)
    paths = ep.write_script(root / "script", suburb_timeline())
    out = root / "scene.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_suburb_driver.py"), "--", "--render", str(REPO / "render"), "--src",
                     str(REPO / "src"), "--root", str(paths.root), "--private", str(paths.private), "--library",
                     str(blend), "--manifest", str(manifest), "--out", str(out)], "building the suburb's scene")
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_suburb_builds_with_its_scenery(built):
    """Contract (M4 suburb §6; `look_residential.GROUND` and `SCENERY`, the core hunk of `build_scene`): from the
    procedural library the suburb's scene builds (no `LookError`: every piece kind has its builder, every region kind a
    floor and the look names its ground and scenery) as an outdoor scene: a floor for each of the 40 regions, the
    ground, no ceiling, underlay, light fixture or facade; the ground beyond the map and the garden and green floors are
    lawn, the street's is not; the far scenery is the residential one (houses, trees and the street's ground carried on
    past both ends, one mesh) and every object of it but that ground lies outside the map's bounds (0 to 300 by 0 to
    82); the dressing holds all five residential classes; and the objects with a pass index are the timeline's
    entities and no scenery or dressing. Regression caught: a suburb drawn on the downtown's asphalt and towers (the
    look keys dropped, the core hunk reverted), a house drawn inside the map, a tree with a pass index (the oracle would
    list it), or a region kind without a floor."""
    names = built["objects"]
    assert sum(n.startswith("floor_") for n in names) == 40 and "ground" in names
    assert not [n for n in names if n.startswith(("ceiling", "fixture", "facade", "underlay"))]
    assert all(m and m.startswith("lawn") for m in built["ground"])
    lawn = {n for n, mats in built["floors"].items() if all(m and m.startswith("lawn") for m in mats)}
    street = built["floors"]["floor_R1"]
    assert "floor_R1" not in lawn and not any(m and m.startswith("lawn") for m in street)
    assert len(lawn) == 36 + 1  # the 36 gardens and the green (the passage and the car park are paving and asphalt)
    scenery = built["scenery"]
    kinds = {s["name"].rsplit("_", 1)[0] for s in scenery}
    assert {"scenery", "scenery_house", "scenery_tree"} <= kinds and "scenery_building" not in kinds
    x0, y0, x1, y1 = load_map("suburb").bounds
    for s in (s for s in scenery if s["name"] != "scenery_ground"):
        a0, b0, _, a1, b1, _ = s["box"]
        assert a0 >= x1 - 1e-3 or a1 <= x0 + 1e-3 or b0 >= y1 - 1e-3 or b1 <= y0 + 1e-3, s["name"]
    assert set(built["dressing"]) == DRESSING
    indexed = built["indexed"]
    assert [pid for pid, _ in indexed] == sorted(map(int, built["instance_map"]))
    assert len(built["instance_map"]) == len(suburb_timeline().entities)
    assert not any(name.startswith(("dressing_", "scenery")) for _, name in indexed)
    assert built["clip_end"] >= 600.0
