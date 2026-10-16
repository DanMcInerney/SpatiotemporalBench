"""Synthetic asset manifests, zips and a fake network for the asset tests: tiny bytes, no download, no network.

The world has a CC BY source (a mug and an alternate mug, like Google Scanned Objects), a CC0 source (a zip material,
like ambientCG) and a procedural glass that uses that material.
"""

from __future__ import annotations

import copy
import http.client
import io
import json
import zipfile
from pathlib import Path

from stbench import asset_manifest as A

CC0, CCBY = "cc0.example.org", "ccby.example.org"
MUG_OBJ, MUG_TEX, ALT_OBJ = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", b"tiny-png-bytes", b"v 5 5 5\n"
MUG_OBJ_URL, MUG_TEX_URL = f"https://{CCBY}/mug_a/model.obj", f"https://{CCBY}/mug_a/texture.png"
ALT_URL, ZIP_URL = f"https://{CCBY}/mug_b/model.obj", f"https://{CC0}/get?file=mat_a_1K.zip"


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


MAT_ZIP = make_zip({"color.jpg": b"jpg-1", "normal.jpg": b"jpg-2", "maps/rough.jpg": b"jpg-3"})
CONTENT = {MUG_OBJ_URL: MUG_OBJ, MUG_TEX_URL: MUG_TEX, ALT_URL: ALT_OBJ, ZIP_URL: MAT_ZIP}


def _entry(path: str, url: str, **extra) -> dict:
    return {"path": path, "url": url, **extra}


def _model(asset_id: str, files: list[dict], **extra) -> dict:
    return {"id": asset_id, "source": "ccby", "role": "model", "kind": "mug", "mobility": "carried",
            "title": f"Mug {asset_id[-1].upper()}", "page": f"https://{CCBY}/pages/{asset_id}",
            "licence": "CC-BY-4.0", "changes": "rescaled, recoloured", "files": files, "estimate_bytes": 1234,
            "entry": "model.obj", "format": "obj", "transform": {"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 0]},
            "pivot": "base_centre", "native_bbox_m": None, "fit": {"mode": "per_axis", "max_anisotropy": 0.15},
            "tint": {"mode": "luminance", "materials": ["*"]}, "compose": None, "logo_review": "none", **extra}


def raw_manifest() -> dict:
    """A valid manifest as parsed JSON."""
    return {
        "sources": {
            "cc0": {"name": "Cee Zero", "licence": "CC0-1.0", "licence_url": f"https://{CC0}/licence",
                    "hosts": [CC0]},
            "ccby": {"name": "Mug Makers", "licence": "CC-BY-4.0", "licence_url": f"https://{CCBY}/licence",
                     "hosts": [CCBY], "attribution": "Mugs by Mug Makers, licensed CC BY 4.0"},
            "procedural": {"name": "our code", "licence": "GPL-3.0-or-later", "hosts": []},
        },
        "assets": [
            _model("mug_a", [_entry("model.obj", MUG_OBJ_URL), _entry("tex/texture.png", MUG_TEX_URL)]),
            _model("mug_b", [_entry("model.obj", ALT_URL)], alternate_for="mug_a"),
            {"id": "mat_a", "source": "cc0", "role": "material", "title": "Material A", "page": f"https://{CC0}/a/mat_a",
             "licence": "CC0-1.0", "tile_m": [1.0, 1.0], "uses": ["floor:hall"],
             "files": [_entry("mat_a_1K.zip", ZIP_URL, extract=True)]},
            {"id": "proc_glass", "source": "procedural", "role": "model", "kind": "glass", "mobility": "carried",
             "title": "procedural glass", "licence": "GPL-3.0-or-later", "builder": "glass",
             "materials": {"glass": "shader:frosted_glass", "base": "mat_a"}, "fit": {"mode": "exact"},
             "tint": {"mode": "luminance", "materials": ["glass"]}},
        ],
        "palette_materials": {"oak": "mat_a"},
        "floors": {"hall": "mat_a"},
        "walls": "mat_a",
    }


def asset(raw: dict, asset_id: str) -> dict:
    return next(a for a in raw["assets"] if a["id"] == asset_id)


def manifest(edit=None) -> A.Manifest:
    raw = copy.deepcopy(raw_manifest())
    if edit:
        edit(raw)
    return A.parse(raw)


def write_manifest(path: Path, raw: dict) -> Path:
    path.write_bytes((json.dumps(raw, indent=1) + "\n").encode("utf-8"))
    return path


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, final_url: str, headers: dict[str, str] | None = None):
        super().__init__(data)
        self.final_url = final_url
        self.headers = headers or {}  # what the server said: Content-Length, Content-Encoding
        self.bytes_read = 0

    def read(self, n: int | None = -1) -> bytes:
        data = super().read(n)
        self.bytes_read += len(data)
        return data

    def geturl(self) -> str:
        return self.final_url


class FakeNet:
    """An opener serving CONTENT by URL. It records every call, can answer from another final URL (a redirect) and can
    announce headers that differ from what it serves (`headers[url]`, e.g. a Content-Length longer than the body)."""

    def __init__(self, redirects: dict[str, str] | None = None):
        self.content = dict(CONTENT)
        self.redirects = redirects or {}
        self.headers: dict[str, dict[str, str]] = {}
        self.calls: list[str] = []
        self.responses: list[FakeResponse] = []

    def __call__(self, url: str, hosts: tuple[str, ...]):
        self.calls.append(url)
        self.responses.append(FakeResponse(self.content[url], self.redirects.get(url, url), self.headers.get(url)))
        return self.responses[-1]


class WireResponse:
    """The real `http.client.HTTPResponse` parsing canned wire bytes (no socket), reporting the final URL as urllib's
    response does. The bytes end where the test cuts them, which is what a dropped connection looks like."""

    def __init__(self, wire: bytes, final_url: str):
        class Sock:
            def makefile(self, mode: str):
                return io.BytesIO(wire)

        self.resp = http.client.HTTPResponse(Sock())
        self.resp.begin()
        self.headers, self.final_url = self.resp.headers, final_url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.resp.close()
        return False

    def geturl(self) -> str:
        return self.final_url

    def read(self, n: int) -> bytes:
        return self.resp.read(n)


def cut_body(body: bytes, announce: int, chunked: bool = False) -> bytes:
    """Wire bytes of an HTTP/1.1 reply that announces `announce` bytes (one chunk when `chunked`) and delivers
    `body`."""
    if chunked:
        return b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + b"%x\r\n" % announce + body
    return b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % announce + body


class EndlessResponse:
    """A response that never ends: only a size cap can stop a download from it."""

    def __init__(self):
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def geturl(self) -> str:
        return MUG_OBJ_URL

    def read(self, n: int) -> bytes:
        self.reads += 1
        if self.reads > 20:  # fail rather than hang when nothing cuts the download off
            raise AssertionError("the download was not cut off")
        return b"x" * n
