"""The dressing models of the asset library in Blender, on a tiny world written by the test (one glTF of two boxes, a
named door node turned open, three manifest entries of it; a texture set and an HDRI the library must leave alone): the
library build's two edits to a published mesh (`rest_pose`, `drop_slots`) and what `DressingAssets` hands a dressing
builder. Needs Blender; marked `blender`, skipped when it is not installed. Nothing is downloaded. About twenty seconds
(one library build, one run of the driver in Blender)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

import asset_world as W
from stbench import asset_fetch as F
from stbench import asset_manifest as A
from stbench import asset_render as AR
from stbench import blender_run
from stbench.render_cli import blender_path

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
HOME = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))  # the door turned a quarter about the file's up axis


class Dressing(NamedTuple):
    manifest_path: Path
    cache: Path
    blend: Path
    doc: dict
    driver: dict


def door_gltf() -> bytes:
    """`W.gltf_boxes()` with its nodes named and the second turned open: the file's two boxes, `base` (x 0..0.2) and
    `door` (x 0.2..0.3, on the second material), the door swung a quarter turn about the origin of its node, up axis
    Y, so it lies at x 0..0.2 and z -0.3..-0.2 in the file's frame."""
    doc = json.loads(W.gltf_boxes())
    doc["nodes"][1]["name"] = "base"
    doc["nodes"][2].update(name="door", rotation=list(HOME))
    return json.dumps(doc).encode()


def entry(path: str, **extra) -> dict:
    return {"path": path, "url": f"https://{W.HOST}/{path}", **extra}


def dressing_asset(asset_id: str, **extra) -> dict:
    return {"id": asset_id, "source": "cc0", "role": "dressing", "title": asset_id, "page": f"https://{W.HOST}/{asset_id}",
            "licence": "CC0-1.0", "files": [entry("dress.gltf")], "entry": "dress.gltf", "format": "gltf",
            "transform": {"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 0]}, "pivot": "base_centre",
            "fit": {"mode": "per_axis", "max_anisotropy": 0.05}, "uses": ["a test door"], **extra}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> Dressing:
    root = tmp_path_factory.mktemp("dressing")
    cache, hdr = root / "cache", b"#?RADIANCE\nnot read by the library\n"
    # the file's frame is Y up; Blender's is Z up, front toward -y: file (x, y, z) is Blender (x, -z, y)
    raw = {"sources": {"cc0": {"name": "Cee Zero", "licence": "CC0-1.0", "licence_url": f"https://{W.HOST}/licence",
                               "hosts": [W.HOST]}},
           "assets": [dressing_asset("dress_rest", rest_pose=["door"], native_bbox_m=[0.3, 0.2, 0.1]),
                      dressing_asset("dress_open", native_bbox_m=[0.2, 0.5, 0.1]),
                      dressing_asset("dress_drop", drop_slots=["mat_two"], native_bbox_m=[0.2, 0.2, 0.1]),
                      {"id": "tex_set", "source": "cc0", "role": "texture", "title": "Texture set",
                       "page": f"https://{W.HOST}/tex_set", "licence": "CC0-1.0",
                       "files": [entry("tex_1K.zip", extract=True)]},
                      {"id": "sky", "source": "cc0", "role": "hdri", "title": "Sky", "page": f"https://{W.HOST}/sky",
                       "licence": "CC0-1.0", "files": [entry("sky_2k.hdr")], "entry": "sky_2k.hdr",
                       "format": "hdr"}],
           "palette_materials": {}, "floors": {}, "walls": None, "skies": {"home": "sky"}}
    for a in raw["assets"]:
        for f in a["files"]:
            data = {"dress.gltf": door_gltf(), "tex_1K.zip": W.mat_zip(), "sky_2k.hdr": hdr}[f["path"]]
            target = cache / a["id"] / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    F.extract_zip(cache / "tex_set" / "tex_1K.zip", cache / "tex_set" / A.EXTRACT_DIR)
    path = root / "manifest.json"
    path.write_bytes((json.dumps(raw, indent=1) + "\n").encode("utf-8"))
    assert A.validate(A.load(path)) == []
    blend = root / "library" / "test.blend"
    doc = AR.build_library(path, blend, cache)
    out = root / "driver.json"
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_dressing_asset_driver.py"), "--", "--render", str(REPO / "render"), "--src",
           str(REPO / "src"), "--manifest", str(path), "--library", str(blend), "--out", str(out)]
    blender_run.run(cmd, "checking the dressing assets")
    return Dressing(path, cache, blend, doc, json.loads(out.read_text(encoding="utf-8")))


def test_rest_pose_closes_the_door_and_an_open_door_widens_the_box(world):
    """Contract (manifest `rest_pose`): the library resets the rotation of the glTF nodes the manifest names before it
    joins the mesh, so a hinged door the exporter wrote open is built closed. Hand-derived: the file's boxes are x
    0..0.2 (base) and x 0.2..0.3 (door), both 0.1 up and z 0..0.2, so closed they measure 0.3 wide, 0.2 deep, 0.1 high
    in Blender's frame; turned a quarter about the door node's origin the door lies at x 0..0.2 and z -0.3..-0.2, so the
    open pair is 0.2 wide and 0.5 deep. Regression caught: a rest pose that is ignored (the box of a power panel with
    its door ajar protrudes 0.4 m into the hall), or applied to every asset."""
    boxes = {a: e["bbox_m"] for a, e in world.doc["assets"].items()}
    assert boxes["dress_rest"] == [0.3, 0.2, 0.1] and boxes["dress_open"] == [0.2, 0.5, 0.1]
    assert world.driver["native"]["dress_rest"] == [0.3, 0.2, 0.1]


def test_drop_slots_removes_the_faces_of_a_slot_and_keeps_the_rest(world):
    """Contract (manifest `drop_slots`): the faces on the named material slots are deleted (glass the exporter left
    opaque), the slot stays so no polygon changes slot, and the box is that of what is left. Hand-derived: the door is
    the second material's twelve triangles, so what remains is the base box, 0.2 x 0.2 x 0.1, with 12 of the 24
    triangles. Regression caught: a drop that resets every polygon to slot 0, deletes the wrong slot, or leaves the
    dangling vertices that would widen the box."""
    assert world.driver["polygons"] == {"dress_rest": 24, "dress_open": 24, "dress_drop": 12}
    assert world.driver["native"]["dress_drop"] == [0.2, 0.2, 0.1]
    assert world.driver["materials"]["dress_drop"] == ["dress_drop:mat_one", "dress_drop:mat_two"]


def test_the_library_holds_each_dressing_model_but_nothing_for_a_texture_set_or_an_hdri(world):
    """Contract: only `model` and `dressing` assets are meshes of the library; a `texture`
    set and an `hdri` are read from the cache, so the build leaves them out and the gate still accepts the library.
    Regression caught: a build that imports an HDRI or a texture zip as a mesh, or a gate that demands a library entry
    for a role the library does not build."""
    assert sorted(world.doc["assets"]) == ["dress_drop", "dress_open", "dress_rest"] and world.doc["materials"] == {}
    assert world.driver["mesh_names"] == ["M:dress_drop", "M:dress_open", "M:dress_rest"]
    assert world.driver["in_library"] == ["dress_drop", "dress_open", "dress_rest"]
    assert AR.check_library(A.load(world.manifest_path), world.blend, world.cache) == []


def test_a_dressing_asset_is_an_unlinked_scale_one_object_on_the_shared_mesh_with_pass_index_zero(world):
    """Contract (`render/asset_dressing.py`, `dressing.Ctx.asset`): `asset(id)` returns a new object on the library's
    one mesh of the model, at scale 1, with `pass_index` 0 (dressing is never an entity), linked to no collection (the
    builder links and fits it), and None for an id that is no dressing model (an entity's mesh is not dressing).
    Regression caught: a dressing piece with a pass index (it would appear in the oracle channel as an entity), a mesh
    copied per call, or an entity asset handed out as dressing."""
    d = world.driver
    assert d["pass_index"] == 0 and d["scale"] == [1.0, 1.0, 1.0] and d["collections"] == 0
    assert d["shared_mesh"] is True and d["absent_is_none"] is True
    assert d["has"] == {"dress_rest": True, "dress_open": True, "dress_drop": True, "proc_glass": False, "nope": False}


def test_the_pivot_is_the_base_centre_with_the_front_toward_minus_y(world):
    """Contract (`render/asset_dressing.py`): a dressing mesh stands on z = 0, centred on x and y, its width along x,
    its depth along y and its height along z; the library keeps the importer's orientation (a Poly Haven glTF faces -y)
    rather than turning it to face +x like an entity. Hand-derived for the closed pair: file (x, y, z) is Blender
    (x, -z, y), so the base box spans x -0.15..0.15, y -0.1..0.1 and z 0..0.1, and `native` is (0.3, 0.2, 0.1).
    Regression caught: a dressing model rotated like an entity (width and depth swapped), or sunk below the floor."""
    lo, hi = world.driver["extent"]["dress_rest"]
    assert np.allclose(lo, (-0.15, -0.1, 0.0), atol=2e-5) and np.allclose(hi, (0.15, 0.1, 0.1), atol=2e-5)
