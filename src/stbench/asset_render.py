"""The Blender asset library and the gate that guards it: `stbench assets library`, `stbench assets preview`.

    stbench assets library [--manifest M] [--out L.blend] [--include-alternates]
    stbench assets preview --out DIR [--kinds mug,glass] [--samples N]

`library` runs `render/asset_library.py` in Blender once: every model and dressing model imported, normalised, joined
into one mesh and measured, every ambientCG material built (a `texture` set and an `hdri` are read from the cache, not
from the library), all written to `<cache>/library/<manifest sha12>-<blender version>.blend`
with `<same name>.json` beside it (boxes, slots, mean luminances). A manifest or a Blender that differs names another
file, so a library built from another is never found. `require_library` is the gate in front of every
`--appearance assets` render: it refuses unless every file the manifest names is in the cache, the library holds every
model and material, and nothing in the manifest is still unmeasured. There is no fallback to placeholders (VISION
rule 10: a render labelled photoreal is photoreal).
`preview` renders a contact sheet per kind (every catalog colour, four views), one of the dressing models, one of the
textured surfaces (each on a plane and a sphere) and one of the texture sets, for a person to look at before any episode
is rendered.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from . import asset_cmd, blender_run
from . import asset_fetch as F
from .asset_manifest import (
    GENERATED,
    IMPORTED,
    PROCEDURAL,
    AssetError,
    Manifest,
    cache_dir,
    file_path,
    load,
    packaged_path,
    validate,
)
from .asset_preview import TILE, preview
from .blender_run import SRC_DIR, BlenderError

BBOX_TOLERANCE_M = 0.001
TINTED = ("luminance", "swap")


class LibraryError(AssetError):
    """The library cannot be built or used; the message lists what is wrong and what to do."""


class LibraryInfo(NamedTuple):
    blend: Path
    manifest_path: Path


def library_path(m: Manifest, blender_version: str, cache: Path | None = None) -> Path:
    return (cache or cache_dir()) / "library" / f"{m.sha256[:12]}-{blender_version}.blend"


def json_path(blend: Path) -> Path:
    return blend.with_suffix(".json")


def _manifest(path: Path | None) -> tuple[Path, Manifest]:
    path = Path(path) if path else packaged_path()
    m = load(path)
    if problems := validate(m):
        raise LibraryError("the manifest has problems:\n  " + "\n  ".join(problems))
    return path, m


def _blender_version() -> str:
    try:
        return blender_run.blender_version()
    except BlenderError as e:
        raise LibraryError(str(e)) from e


# -- the gate ----------------------------------------------------------------------------------------------------
def _coverage_problems(m: Manifest, doc: dict, complete: bool = True) -> list[str]:
    """Every model, dressing model and material is in the library, and each box the manifest records is the one
    measured. With `complete`, every box and tint slot must also be recorded (the preview, which a person looks at
    before filling them in, does not ask for that)."""
    out, built, mats = [], doc.get("assets", {}), doc.get("materials", {})
    for a in m.assets:
        if a.role == "material":
            if a.id not in mats:
                out.append(f"{a.id}: not in the library")
        elif a.role in IMPORTED and not a.alternate_for:
            entry = built.get(a.id)
            if entry is None:
                out.append(f"{a.id}: not in the library")
                continue
            if a.native_bbox_m is None:
                if complete and a.source != PROCEDURAL:
                    out.append(f"{a.id}: native_bbox_m is not recorded in the manifest (the library measured "
                               f"{entry['bbox_m']}); record it, then rebuild")
            elif max(abs(x - y) for x, y in zip(entry["bbox_m"], a.native_bbox_m, strict=True)) > BBOX_TOLERANCE_M:
                out.append(f"{a.id}: the library measured {entry['bbox_m']} m, more than 1 mm off the manifest's "
                           f"native_bbox_m {a.native_bbox_m}")
            tint = a.tint or {}
            if complete and tint.get("mode") in TINTED and tint.get("materials") is None:
                out.append(f"{a.id}: tint.materials is null; its slots are {[s['slot'] for s in entry['slots']]}: "
                           "name the ones that take the colour attribute, then rebuild")
    return out


def cache_problems(m: Manifest, cache: Path, include_alternates: bool = False) -> list[str]:
    """What the library needs and the cache lacks: a downloaded file that is missing or not (fully) unpacked (`fetch`
    brings it), and a generated file that is missing (its source's recipe makes it). An alternate counts only when
    asked for."""
    out = [f"{t.label}: " + ("missing" if not t.dest.is_file() else "not fully unpacked")
           for t in F.plan(m, include_alternates=include_alternates, cache=cache)]
    for a in m.assets:
        source = m.sources[a.source]
        if source.kind != GENERATED or (a.alternate_for and not include_alternates):
            continue
        out += [f"{a.id} {f.path}: missing (generated, not downloaded: run {source.generator['recipe']}, whose "
                "docstring gives the command)" for f in a.files if not file_path(a, f, cache).is_file()]
    return out


def check_library(m: Manifest, blend: Path, cache: Path | None = None, complete: bool = True) -> list[str]:
    """Why the library at `blend` may not be used for this manifest, or an empty list: a file of the manifest missing
    from the cache, no library or no JSON beside it, or (`complete`: see `_coverage_problems`) a model or material the
    library lacks or measures differently."""
    problems = cache_problems(m, cache or cache_dir())
    jpath = json_path(blend)
    if not blend.is_file() or not jpath.is_file():
        return [*problems, f"no asset library at {blend}: run `stbench assets library`"]
    try:
        doc = json.loads(jpath.read_text(encoding="utf-8"))
    except ValueError as e:
        return [*problems, f"{jpath.name} is not valid JSON: {e}"]
    return problems + _coverage_problems(m, doc, complete)


def require_library(manifest: Path | None = None, library: Path | None = None, cache: Path | None = None,
                    blender_version: str | None = None, complete: bool = True) -> LibraryInfo:
    """The checked library for an `assets` render, or `LibraryError` listing everything that is wrong. The preview asks
    for `complete=False`: it is how a person sees the assets before recording their boxes and tint slots."""
    path, m = _manifest(manifest)
    cache = cache or cache_dir()
    blend = Path(library) if library else library_path(m, blender_version or _blender_version(), cache)
    if problems := check_library(m, blend, cache, complete):
        raise LibraryError("the asset library cannot be used:\n  " + "\n  ".join(problems))
    return LibraryInfo(blend.resolve(), path.resolve())


# -- building ----------------------------------------------------------------------------------------------------
def build_library(manifest: Path | None = None, out: Path | None = None, cache: Path | None = None,
                  include_alternates: bool = False) -> dict:
    """Run the library build in Blender; returns the JSON it wrote."""
    path, m = _manifest(manifest)
    cache = cache or cache_dir()
    if problems := cache_problems(m, cache, include_alternates):
        raise LibraryError("cannot build the library until every file is cached (`stbench assets fetch`):\n  "
                           + "\n  ".join(problems))
    out = Path(out) if out else library_path(m, _blender_version(), cache)
    args = ["--manifest", str(path), "--cache", str(cache), "--out", str(out), "--src", str(SRC_DIR)]
    try:
        extra = ["--include-alternates"] if include_alternates else []
        blender_run.run(blender_run.command("asset_library.py", *args, *extra), "building the asset library")
    except BlenderError as e:
        raise LibraryError(str(e)) from e
    return json.loads(json_path(out).read_text(encoding="utf-8"))


def review_table(m: Manifest, doc: dict) -> str:
    """What a person fills into the manifest after a build: each model's box and its material slots."""
    lines = [f"{'asset':36} {'box (m)':22} slots (* = takes the colour attribute)"]
    for a in m.assets:
        e = doc["assets"].get(a.id)
        if e:
            slots = ", ".join(f"{s['slot']}{'*' if s['tinted'] else ''}" for s in e["slots"])
            lines.append(f"{a.id:36} {str(e['bbox_m']):22} {slots}")
    unset = [a.id for a in m.assets if a.role == "material" and a.tile_m is None]
    if unset:
        lines.append(f"materials with no tile_m (the build used 1 m): {', '.join(unset)}")
    return "\n".join(lines)


# -- commands ----------------------------------------------------------------------------------------------------
def _guarded(handler: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except AssetError as e:
            print(f"assets: {e}", file=sys.stderr)
            return 2

    return run


def _library(args: argparse.Namespace) -> int:
    doc = build_library(args.manifest, args.out, include_alternates=args.include_alternates)
    _, m = _manifest(args.manifest)
    print(review_table(m, doc))
    print(f"library built: {len(doc['assets'])} models, {len(doc['materials'])} materials")
    return 0


def _tile(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a size like 192x144") from None
    return width, height


def _preview(args: argparse.Namespace) -> int:
    info = require_library(args.manifest, args.library, complete=False)
    for path in preview(info.blend, info.manifest_path, args.out, args.kinds, args.samples, args.tile):
        print(path)
    return 0


def register(sub) -> None:
    commands = asset_cmd.group(sub)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, help="a manifest file (default: the packaged one)")
    lib = commands.add_parser("library", parents=[common], help="build the Blender asset library from the cache")
    lib.add_argument("--out", type=Path, help="the .blend to write (default: <cache>/library/<sha12>-<version>.blend)")
    lib.add_argument("--include-alternates", action="store_true", help="also build the alternates (must be cached)")
    lib.set_defaults(func=_guarded(_library))
    pre = commands.add_parser("preview", parents=[common], help="render contact sheets of every kind and surface")
    pre.add_argument("--out", type=Path, required=True, help="a folder for the sheets (outside any dataset)")
    pre.add_argument("--library", type=Path, help="the library .blend (default: the one built for the manifest)")
    pre.add_argument("--kinds", type=lambda s: [x for x in s.split(",") if x], help="only these kinds, comma separated")
    pre.add_argument("--samples", type=int, default=32, help="samples per tile")
    pre.add_argument("--tile", type=_tile, default=TILE, help="a tile's size, WxH (default 192x144)")
    pre.set_defaults(func=_guarded(_preview))
