"""A tiny asset world on disk for the library and render tests: real (but minute) OBJ, PNG, JPEG and zip files, a
manifest that points at them, and a cache folder holding them. Nothing is downloaded and no real asset is used.

Models: `cube_a` (a box 0.2 x 0.1 x 0.1 m, off the origin, in two objects with two materials, one textured; kind mug;
laid out like a Google Scanned Object: its MTL names `texture.png`, which lives in `materials/textures/`, one folder
away from where Blender looks),
`slab_b` (a Y-up OBJ turned a quarter turn and composed into a mirrored pair; kind bowl), `chair_a` (a box 0.45 x 0.45 x
0.9 m, light furniture whose "oak" is a texture swap; kind chair), `gltf_d` (two boxes under a moved parent node in one
glTF with the buffer inline; kind bag) and `proc_glass` (the procedural glass with the frosted-glass shader). Material:
`mat_a`, a zip with a colour and a roughness image.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from stbench import asset_fetch as F
from stbench import asset_manifest as A

HOST = "assets.example.org"
CUBE = (0.2, 0.1, 0.1)  # cube_a's box in metres
CHAIR = (0.45, 0.45, 0.9)
GLTF = (0.3, 0.2, 0.1)  # gltf_d's box in Blender's frame: the file's 0.3 x 0.1 (up) x 0.2 turned Z up
SLAB = (0.2, 0.61, 0.05)  # slab_b's: 0.3 x 0.2 x 0.05 turned a quarter turn, then two of them 0.01 m apart across y
TEX_GREYS = np.array([[60, 90, 120, 150], [90, 120, 150, 180], [120, 150, 180, 210], [150, 180, 210, 240]], np.uint8)
TEX_TINT = (1.0, 0.6, 0.3)  # the texture is orange: a tint that kept its hue would show
CUBE_MTL = b"newmtl side\nKd 0.8 0.2 0.2\nnewmtl top\nKd 1 1 1\nmap_Kd texture.png\n"
SLAB_MTL = b"newmtl plain\nKd 0.5 0.5 0.5\n"
CHAIR_MTL = b"newmtl wood\nKd 0.6 0.4 0.2\n"
# a box's faces by name, as vertex numbers (1-based) of its eight corners listed x fastest, then y, then z
FACES = {"bottom": (1, 3, 4, 2), "top": (5, 6, 8, 7), "front": (1, 2, 6, 5), "back": (3, 7, 8, 4),
         "left": (1, 5, 7, 3), "right": (2, 4, 8, 6)}


class World(NamedTuple):
    manifest_path: Path
    cache: Path
    raw: dict


def texture_pixels() -> np.ndarray:
    """The 4 x 4 x 3 uint8 texture of `cube_a`: a grey ramp in orange."""
    return np.stack([np.rint(TEX_GREYS * f) for f in TEX_TINT], axis=-1).astype(np.uint8)


def texture_png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(texture_pixels()).save(buf, format="PNG")
    return buf.getvalue()


def jpeg(value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (value, value // 2, 40)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def box_obj(xs, ys, zs, mtl: str, parts: list[tuple[str, str, list[str]]], y_up: bool = False) -> bytes:
    """An OBJ box: (low, high) per axis, the material file's name and `(object, material, face names)` parts. With
    `y_up` the box's z extent is written on the file's y axis (a Y-up file) and its y extent on the file's z axis."""
    corners = [(x, y, z) for z in zs for y in ys for x in xs]
    if y_up:
        corners = [(x, z, y) for x, y, z in corners]
    lines = [f"mtllib {mtl}"] + [f"v {x} {y} {z}" for x, y, z in corners] + ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    for name, material, faces in parts:
        lines += [f"o {name}", f"usemtl {material}"]
        lines += ["f " + " ".join(f"{v}/{i + 1}" for i, v in enumerate(FACES[f])) for f in faces]
    return ("\n".join(lines) + "\n").encode()


def cube_obj() -> bytes:
    """x 1.0..1.2, y 2.0..2.1, z 0.5..0.6 (Z up): `caps` on the textured material, `sides` on the plain one."""
    return box_obj((1.0, 1.2), (2.0, 2.1), (0.5, 0.6), "model.mtl",
                   [("caps", "top", ["bottom", "top"]), ("sides", "side", ["front", "back", "left", "right"])])


def slab_obj() -> bytes:
    """A Y-up box: x 0.3 m, y (up) 0.05 m, z 0.2 m, one plain material."""
    return box_obj((0.0, 0.3), (0.0, 0.2), (0.0, 0.05), "slab.mtl", [("slab", "plain", list(FACES))], y_up=True)


def chair_obj() -> bytes:
    """0.45 x 0.45 x 0.9 m (Z up), centred on x and y with its base on z = 0, one object on one material."""
    return box_obj((-0.225, 0.225), (-0.225, 0.225), (0.0, 0.9), "chair.mtl", [("chair", "wood", list(FACES))])


def gltf_boxes() -> bytes:
    """A glTF (Y up) of two boxes on two materials, in two mesh nodes under a parent that is moved by (1, 0, 2): x
    0..0.2 and x 0.2..0.3, y (up) 0..0.1, z 0..0.2, the geometry in an inline base64 buffer."""
    corners, tris = [], []
    for x0, x1 in ((0.0, 0.2), (0.2, 0.3)):
        corners.append(np.array([(x, y, z) for z in (0.0, 0.2) for y in (0.0, 0.1) for x in (x0, x1)], np.float32))
        tris.append(np.array([[f[i] - 1 for i in tri] for f in FACES.values() for tri in ((0, 1, 2), (0, 2, 3))],
                             np.uint16))
    blob = b"".join(a.tobytes() for a in corners) + b"".join(a.tobytes() for a in tris)
    pos_len, idx_len = corners[0].nbytes, tris[0].nbytes
    views = [{"buffer": 0, "byteOffset": i * pos_len, "byteLength": pos_len, "target": 34962} for i in range(2)]
    views += [{"buffer": 0, "byteOffset": 2 * pos_len + i * idx_len, "byteLength": idx_len, "target": 34963}
              for i in range(2)]
    accessors = [{"bufferView": i, "componentType": 5126, "count": 8, "type": "VEC3",
                  "min": corners[i].min(axis=0).tolist(), "max": corners[i].max(axis=0).tolist()} for i in range(2)]
    accessors += [{"bufferView": 2 + i, "componentType": 5123, "count": 36, "type": "SCALAR"} for i in range(2)]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"children": [1, 2], "translation": [1.0, 0.0, 2.0]}, {"mesh": 0}, {"mesh": 1}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": i}, "indices": 2 + i, "material": i}]}
                      for i in range(2)],
           "materials": [{"name": n, "pbrMetallicRoughness": {"baseColorFactor": c, "metallicFactor": 0.0}}
                         for n, c in (("mat_one", [0.2, 0.2, 0.8, 1.0]), ("mat_two", [0.8, 0.8, 0.2, 1.0]))],
           "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(blob),
                        "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}]}
    return json.dumps(doc).encode()


def mat_zip() -> bytes:
    """The material's zip, the same bytes on every call (a fixed timestamp: `writestr` of a bare name stamps the clock,
    and the cache and the manifest each call this, so a call a second apart made another zip)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in (("Mat_1K-JPG_Color.jpg", jpeg(180)), ("Mat_1K-JPG_Roughness.jpg", jpeg(120))):
            zf.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data)
    return buf.getvalue()


def file_bytes() -> dict[str, dict[str, bytes]]:
    """{asset id: {file path: bytes}} for the files the manifest lists, as written into the cache."""
    return {"cube_a": {"meshes/model.obj": cube_obj(), "meshes/model.mtl": CUBE_MTL,
                       "materials/textures/texture.png": texture_png()},
            "slab_b": {"slab.obj": slab_obj(), "slab.mtl": SLAB_MTL},
            "chair_a": {"chair.obj": chair_obj(), "chair.mtl": CHAIR_MTL}, "gltf_d": {"boxes.gltf": gltf_boxes()},
            "mat_a": {"mat_a_1K-JPG.zip": mat_zip()}}


def _entry(path: str, **extra) -> dict:
    return {"path": path, "url": f"https://{HOST}/{path}", **extra}


def _model(asset_id: str, kind: str, entry: str, **extra) -> dict:
    files = file_bytes()[asset_id]
    return {"id": asset_id, "source": "ccby", "role": "model", "kind": kind, "mobility": "carried", "title": asset_id,
            "page": f"https://{HOST}/{asset_id}", "licence": "CC-BY-4.0", "changes": "rescaled, recoloured",
            "files": [_entry(p) for p in files], "entry": entry, "format": "obj",
            "transform": {"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 0]}, "pivot": "base_centre",
            "native_bbox_m": None, "fit": {"mode": "per_axis", "max_anisotropy": 0.15},
            "tint": {"mode": "luminance", "materials": ["*"]}, "compose": None, "logo_review": "none", **extra}


def raw_manifest() -> dict:
    return {
        "sources": {"ccby": {"name": "Mug Makers", "licence": "CC-BY-4.0", "licence_url": f"https://{HOST}/licence",
                             "hosts": [HOST], "attribution": "Mugs by Mug Makers, licensed CC BY 4.0"},
                    "cc0": {"name": "Cee Zero", "licence": "CC0-1.0", "licence_url": f"https://{HOST}/licence",
                            "hosts": [HOST]},
                    "procedural": {"name": "our code", "licence": "GPL-3.0-or-later", "hosts": []}},
        "assets": [
            _model("cube_a", "mug", "meshes/model.obj", native_bbox_m=list(CUBE)),
            _model("slab_b", "bowl", "slab.obj", native_bbox_m=list(SLAB),
                   compose={"mode": "pair_mirror", "gap_m": 0.01},
                   transform={"unit_scale": 1.0, "up": "Y", "rotate_deg": [0, 0, 90]}),
            _model("chair_a", "chair", "chair.obj", mobility="light", native_bbox_m=list(CHAIR),
                   fit={"mode": "uniform"}, tint={"mode": "swap", "materials": ["*"]}),
            _model("gltf_d", "bag", "boxes.gltf", native_bbox_m=list(GLTF), format="gltf"),
            {"id": "mat_a", "source": "cc0", "role": "material", "title": "Material A", "page": f"https://{HOST}/mat_a",
             "licence": "CC0-1.0", "tile_m": [0.5, 0.5], "uses": ["floor:kitchen"],
             "files": [_entry("mat_a_1K-JPG.zip", extract=True)]},
            {"id": "proc_glass", "source": "procedural", "role": "model", "kind": "glass", "mobility": "carried",
             "title": "procedural glass", "licence": "GPL-3.0-or-later", "builder": "glass",
             "materials": {"glass": "shader:frosted_glass"}, "pivot": "base_centre", "native_bbox_m": None,
             "fit": {"mode": "exact"}, "tint": {"mode": "luminance", "materials": ["glass"]}, "compose": None},
        ],
        "palette_materials": {"oak": "mat_a"},
        "floors": {"kitchen": "mat_a", "dock": "shader:concrete"},
        "walls": "mat_a",
    }


def build(root: Path, edit=None) -> World:
    """Write the cache and the manifest under `root`. `edit(raw)` may change the manifest before it is written."""
    cache = root / "cache"
    for asset_id, files in file_bytes().items():
        for path, data in files.items():
            target = cache / asset_id / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    F.extract_zip(cache / "mat_a" / "mat_a_1K-JPG.zip", cache / "mat_a" / A.EXTRACT_DIR)
    raw = raw_manifest()
    if edit:
        edit(raw)
    path = root / "manifest.json"
    path.write_bytes((json.dumps(raw, indent=1) + "\n").encode("utf-8"))
    return World(path, cache, raw)
