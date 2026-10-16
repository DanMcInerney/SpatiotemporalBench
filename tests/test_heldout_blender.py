"""Every held-out map present, built in Blender (M4-integration call 8): in assets mode, on a library of procedural
models, the scene of a hand-made one-keyframe episode on the map builds through the renderer's own scene builder with
`STBENCH_MAPS` set. Held-out maps come from the variable as in `test_heldout_maps.py`; no layout fact is stated.
Needs Blender; marked `blender`, skipped when Blender is not installed or no held-out map is present. The scenes are
built, not rendered, so no GPU; about ten seconds a map (one library build, one scene build)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from builders import camera_keyframe, keyframe, minimal_timeline
from stbench import asset_manifest, job_registry, map_files
from stbench.catalog import CATALOGS
from stbench.episode import write_script
from stbench.generate import build_world
from stbench.mapspec import load_map
from stbench.render_cli import blender_path
from stbench.robots import ROBOTS
from stbench.timeline import Ego, Region, Task, TaskPlan
from test_heldout_maps import JOBS, maps_present
from test_render_city_blender import probe, procedural_library

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
FLIER_HEIGHT_M, FLIER_PITCH_DEG = 12.0, -35.0  # a flier's one keyframe, as the downtown's hand-made episode puts it


def one_keyframe_timeline(m):
    """A hand-made episode on `m`: the map's fixed furniture and doors as entities (the first things the scene draws; a
    light or carried entity would need a model that is a download), the robot of the catalog's M4 job, and one keyframe
    of it at the first region's anchor (a flier at its altitude, pitched down)."""
    job_id = JOBS[m.catalog]
    robot = ROBOTS[job_registry.get(job_id).robot]
    base = minimal_timeline()
    world = build_world(m, 1)
    placements, states = world.ledger.finish()
    keep = {e.id for e in world.entities if e.mobility == "fixed"}
    x, y = m.regions[0].anchor
    if robot.flies:
        kf = camera_keyframe(0, 0.0, (x, y), 0.0, robot=robot.id, z=FLIER_HEIGHT_M, pitch=math.radians(FLIER_PITCH_DEG))
    else:
        kf = keyframe(robot.id, 0, 0.0, (x, y, 0.0), 0.0, task=0, phase=robot.status.start["phase"])
    episode = dataclasses.replace(base.episode, map=m.name, robot=robot.id, job=job_id)
    return dataclasses.replace(
        base, episode=episode,
        regions=[Region(r.id, r.name, r.kind, tuple(map(tuple, r.polygon)), r.level, r.z0, r.z1) for r in m.regions],
        entities=[e for e in world.entities if e.id in keep], placements=[p for p in placements if p.entity in keep],
        states=[s for s in states if s.entity in keep], ego=Ego([kf], [Task(0, 0, 1, TaskPlan("hand", {}), ())], []))


def floors_of(m) -> str:
    """The map whose regions' floors the library gets (`procedural_library` reads the manifest's floors by region name):
    the held-out map itself when every region name is a floor of the manifest (the home maps' names are their kinds),
    else a bundled map of its catalog; a catalog with none skips."""
    if all(r.name in asset_manifest.packaged().floors for r in m.regions):
        return m.name
    for name in map_files.bundled_maps():
        if load_map(name).catalog == m.catalog:
            return name
    pytest.skip(f"the {m.catalog} catalog has no bundled map whose floors the library could take")


@pytest.fixture(scope="module", params=maps_present())
def held_out(request):
    return load_map(request.param)


@pytest.fixture(scope="module")
def library(held_out, tmp_path_factory):
    """The procedural models of the map's catalog (light and carried kinds; the dressing's scanned pieces and the
    textured floors are downloads and give way to their placeholders), built with the map's floors."""
    if held_out.catalog not in JOBS:
        pytest.skip(f"the {held_out.catalog} catalog has no M4 job named in test_heldout_maps")
    cat = CATALOGS[held_out.catalog]
    kinds = {*cat.kinds, *cat.mover_kinds, *(f.kind for f in held_out.furniture if f.mobility == "light")}
    return procedural_library(tmp_path_factory.mktemp(held_out.name), lambda kind: kind in kinds, floors_of(held_out))


def test_every_held_out_map_builds(held_out, library, tmp_path):
    """Contract (M4-integration call 8): in assets mode on the procedural library, a hand-made one-keyframe timeline
    of the map builds through `build_scene.build` with the variable set (no `LookError` for a region kind, no failure
    of the dressing's `ceiling_spot`); an indoor map has one lamp per ceiling light (`ceiling_light_<i>`); and the
    scene holds dressing. Regression caught: a held-out map that renders only with a fallback, or not at all."""
    manifest, blend = library
    paths = write_script(tmp_path / "data", one_keyframe_timeline(held_out))
    scene = probe(paths, tmp_path / "probe.json", "--appearance", "assets", "--library", str(blend), "--manifest",
                  str(manifest))
    assert any(o.startswith("dressing_") for o in scene["objects"])
    if held_out.outdoor is None:
        lamps = [light for light in scene["lights"] if light[0].startswith("ceiling_light_")]
        assert len(lamps) == len(held_out.ceiling_lights) > 0
