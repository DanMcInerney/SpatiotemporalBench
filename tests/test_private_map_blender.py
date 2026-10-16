"""A private map in Blender (M4.0): the renderer finds a map that lives outside the package through `STBENCH_MAPS`, in
assets mode too, where the scene reads the map's own file for its dressing and its lamps. Needs Blender; marked
`blender`, skipped when Blender is not installed. The scenes are built, not rendered, so no GPU; about twenty seconds
(one library build, two scene builds)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from stbench import episode as ep
from stbench.map_files import MAPS_ENV, map_bytes
from stbench.render_cli import blender_path
from test_render_city_blender import APARTMENT_MODELS, apartment_timeline, probe, procedural_library

# One xdist group a module (`--dist loadgroup`): its module fixture is built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
PRIVATE = "apartment_private_copy"


@pytest.fixture(scope="module")
def indoor_library(tmp_path_factory):
    """The procedural models of the apartment's assets scene (`apartment_timeline`), built with the indoor floors of
    `test_render_city_blender`'s library."""
    kinds = {kind for kind, _, _ in APARTMENT_MODELS}
    return procedural_library(tmp_path_factory.mktemp("indoor"), lambda kind: kind in kinds, "warehouse_a")


def test_a_private_map_builds_its_dressing_and_its_lamps(indoor_library, tmp_path, monkeypatch):
    """Contract (M4.0): in assets mode, the scene of an episode on a private map (a renamed copy of the apartment, in a
    folder `STBENCH_MAPS` names) builds through the renderer's own scene builder with the variable set: its lights
    equal the bundled apartment's (the lamps read the spec's ceiling lights and room names, which the copy keeps), and
    it holds dressing pieces, as the bundled apartment's scene does. The full digests are not compared: the home decor
    is drawn from the map's name, so a renamed copy is dressed differently. Regression caught: `build_scene._raw_spec`
    reading only the package, so a held-out map fails to build in assets mode, or builds without its dressing and
    lamps."""
    manifest, blend = indoor_library
    doc = json.loads(map_bytes("apartment_a"))
    doc["name"] = PRIVATE
    data = json.dumps(doc, indent=1).encode("utf-8")
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / f"{PRIVATE}.json").write_bytes(data)
    monkeypatch.setenv(MAPS_ENV, str(tmp_path / "maps"))
    assets = ("--appearance", "assets", "--library", str(blend), "--manifest", str(manifest))
    bundled = apartment_timeline(assets=True)
    private = dataclasses.replace(bundled, episode=dataclasses.replace(bundled.episode, map=PRIVATE))
    scenes = {name: probe(ep.write_script(tmp_path / name, tl), tmp_path / f"{name}.json", *assets)
              for name, tl in (("bundled", bundled), ("private", private))}
    dressing = {name: [o for o in scene["objects"] if o.startswith("dressing_")] for name, scene in scenes.items()}
    assert dressing["bundled"]  # the bundled scene holds dressing, so the check below is not vacuous
    assert dressing["private"]
    lights = {name: sorted((kind, energy) for _, kind, energy in scene["lights"]) for name, scene in scenes.items()}
    assert len(lights["bundled"]) > 1  # the apartment's lamps are in the scene, so the equality is not vacuous
    assert lights["private"] == lights["bundled"]
