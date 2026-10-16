"""`house_floor`'s staircase and scene in Blender: a hand-made episode on the map (its fixed furniture, its doors and
the laundry basket, whose model is procedural) is built from the asset library, no render and no GPU, and rays are cast
at the staircase. The library is the packaged manifest's procedural models of the kinds the episode holds (so nothing
is downloaded: `test_render_city_blender.procedural_library`), and the scene is the shipped one (the household's
dressing, the lamps, the wear) with the textured surfaces left out. Needs Blender; marked `blender`, skipped when it is
not installed. About thirty seconds (one library build of one model, one script that only builds)."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from builders import camera_keyframe, minimal_timeline
from stbench import blender_run
from stbench import episode as ep
from stbench.geometry import Rect, quat_z, to_local, to_world
from stbench.mapspec import load_map
from stbench.render_cli import blender_path
from stbench.timeline import Ego, Entity, Placement, Region, State, Task, TaskPlan
from test_dressing import HOME_PLAN
from test_render_city_blender import procedural_library

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
TOL_M = 0.01


def house_timeline():
    """The map's fixed furniture and doors as entities (a chair and the stool are scanned meshes, so downloads: they are
    left out, as the apartment's are in `test_render_city_blender.apartment_timeline`), the laundry basket, and two
    camera poses in the hall."""
    m, base = load_map("house_floor"), minimal_timeline()
    entities, placements, states = [], [], []

    def add(kind, attributes, mobility, pos, yaw, size, asset=None):
        eid = f"E{len(entities) + 1:04d}"
        entities.append(Entity(eid, kind, attributes, mobility, size, asset))
        placements.append(Placement(eid, 0.0, None, pos, quat_z(yaw), None))
        return eid

    for f in m.furniture:
        if f.mobility == "fixed" or f.kind == "laundry_basket":
            add(f.kind, {"name": f.name, "color": f.color}, f.mobility, (*f.center, 0.0), f.yaw, f.size,
                f"proc_{f.kind}" if f.mobility == "light" else None)
    names = {r.id: r.name for r in m.regions} | {"outside": "outside"}
    for o in m.doors():
        eid = add("door", {"between": [names[i] for i in o.between], "hinge": o.hinge, "swing": o.swing}, "fixed",
                  (*o.center, 0.0), o.yaw, (o.width, 0.04, o.height))
        states.append(State(eid, "open", 0.0, None, bool(o.interior)))
    keyframes = [camera_keyframe(k, 1.5 * k, xy, yaw) for k, (xy, yaw) in enumerate([((7.4, 1.0), math.pi / 2),
                                                                                       ((7.0, 4.5), -math.pi / 2)])]
    regions = [Region(r.id, r.name, r.kind, tuple(map(tuple, r.polygon)), r.level, r.z0, r.z1) for r in m.regions]
    episode = dataclasses.replace(base.episode, map=m.name)
    ego = Ego(keyframes, [Task(0, 0, len(keyframes), TaskPlan("hand", {}), ())], [])
    return dataclasses.replace(base, episode=episode, regions=regions, entities=entities, placements=placements,
                               states=states, ego=ego)


def rays(stairs) -> tuple[Rect, list[dict]]:
    """Five rays in the staircase's frame (x towards the foot, y to the left, z up), as map-frame rays: from 0.8 m
    before the foot at 0.05 m and at 1.3 m, along the floor at 0.05 m from the open side, and over the top steps down
    from 2.4 m (it starts inside the bulkhead) and up from 2.1 m (it starts under it)."""
    hx, hy = stairs.size[0] / 2, stairs.size[1] / 2
    frame = Rect(stairs.center[0], stairs.center[1], stairs.size[0], stairs.size[1], stairs.yaw)
    c, s = math.cos(stairs.yaw), math.sin(stairs.yaw)
    out = []
    for name, (lx, ly, z), (dx, dy, dz) in (("foot_low", (hx + 0.8, 0.0, 0.05), (-1, 0, 0)),
                                            ("foot_high", (hx + 0.8, 0.0, 1.3), (-1, 0, 0)),
                                            ("side_floor", (0.25, -hy - 0.8, 0.05), (0, 1, 0)),
                                            ("over_top", (-1.0, 0.0, 2.4), (0, 0, -1)),
                                            ("under_top", (-1.0, 0.0, 2.1), (0, 0, 1))):
        x, y = to_world(frame, lx, ly)
        out.append({"name": name, "origin": [x, y, z], "direction": [c * dx - s * dy, s * dx + c * dy, dz]})
    return frame, out


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """What the driver reports for the episode: the library of the basket's model, the script written from the
    hand-made timeline, the scene built from the library and the five rays cast at the staircase."""
    root = tmp_path_factory.mktemp("house")
    manifest, blend = procedural_library(tmp_path_factory.mktemp("library"), lambda kind: kind == "laundry_basket",
                                         "house_floor")
    paths = ep.write_script(root / "script", house_timeline())
    stairs = next(f for f in load_map("house_floor").furniture if f.kind == "stairs")
    frame, cast = rays(stairs)
    (root / "rays.json").write_text(json.dumps(cast), encoding="utf-8")
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_house_floor_driver.py"), "--", "--render", str(REPO / "render"), "--src",
           str(REPO / "src"), "--root", str(paths.root), "--private", str(paths.private), "--manifest", str(manifest),
           "--library", str(blend), "--rays", str(root / "rays.json"), "--out", str(root / "result.json")]
    blender_run.run(cmd, "building the house floor's scene")
    return json.loads((root / "result.json").read_text(encoding="utf-8")), stairs, frame


def test_the_stairs_are_drawn_solid_and_lit(scene):
    """Contract (M4 house_floor §2; `under_m` 0, `build_scene.py:177-181`, `lighting.lamps`): the scene of the map
    builds from the assets (no `LookError`), draws every class of the household's dressing, has one lamp for each
    ceiling light of each room; a ray from 0.05 m and one from 1.3 m up, towards the staircase's foot, meet the
    staircase first (the bottom riser, then a higher riser); a ray along the floor at 0.05 m towards the flight's side
    meets its panelling there, so nothing of the underside is open; a ray down from 2.4 m over the top steps meets the
    bulkhead's underside, 0.3 m under the ceiling, and so does one up from 2.1 m. Regression caught: a staircase the
    vacuum would see under or drive beneath (a gap under the steps, an open side), one with no bulkhead where the flight
    meets the ceiling, or a map the assets render refuses (a kind with no drawing, a room kind with no look, a room
    with no lamp)."""
    result, stairs, frame = scene
    m, hx, hy, top = load_map("house_floor"), stairs.size[0] / 2, stairs.size[1] / 2, stairs.size[2]
    classes = {n.removeprefix("dressing_").rsplit("_", 1)[0] for n in result["dressing"]}
    assert set(HOME_PLAN) <= classes, sorted(set(HOME_PLAN) - classes)
    rooms = {r.id: r.name for r in m.regions}
    assert Counter(lamp["room"] for lamp in result["lamps"]) == Counter(rooms[c.region] for c in m.ceiling_lights)
    hits = result["hits"]
    assert all(h["hit"] and h["object"] == result["stairs"] for h in hits.values()), hits
    local = {name: (*to_local(frame, *h["point"][:2]), h["point"][2]) for name, h in hits.items()}
    assert local["foot_low"][0] == pytest.approx(hx, abs=TOL_M)
    assert -hx < local["foot_high"][0] < hx and 0.8 < local["foot_high"][2] < top
    assert local["side_floor"][1] == pytest.approx(-hy, abs=TOL_M) and local["side_floor"][2] == pytest.approx(0.05)
    assert local["over_top"][2] == pytest.approx(top - 0.3, abs=TOL_M)
    assert local["under_top"][2] == pytest.approx(top - 0.3, abs=TOL_M)
